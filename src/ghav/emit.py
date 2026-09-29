"""WIR -> GitHub Actions YAML.

Emission is a pure, deterministic function of the Program whose guiding rule is the round
trip: every emitted field is chosen so that ``lower`` reads the instance back (bindings,
conditions, matrix, runs-on, env, check names). Where the raw YAML satisfies that, the raw
text is kept (readability); otherwise the field is written from the WIR.

1. Grouping. Frozen instances of one original job are emitted once from ``raw``. Non-frozen
   instances of one original matrix job start in one group, which is refined until every
   member has the same emitted body (steps, conditions, needs, runs-on, env, ...) and the
   same set of dependents (``needs`` on a matrix job waits for every combination, so a job
   that needs one member must need all of them). A group of several members is folded back
   into a matrix job; every other instance becomes its own job. Refinement only splits
   groups, so it terminates.
2. Job keys. A group that is a whole original job (or a fold/frozen group) keeps the
   sanitized original key; other instances get a key derived from the instance id, sanitized
   to GitHub job-id syntax (long ids keep a prefix plus a stable hash). Collisions get
   ``-2``, ``-3``, ... in a fixed order.
3. Steps come from ``Step.raw`` when lowering the raw mapping in the emitted job gives the
   step's ``Binding`` (matrix interpolation and workflow-level ``defaults.run`` included),
   else from the ``Binding``. ``if:`` keeps the raw text when ``expr.parse`` +
   ``expr.substitute`` + the default ``success() &&`` fold reproduces the explicit
   condition (compared after ``expr.simplify`` and and/or normalization); otherwise the
   condition is rendered from the Expr with its status references explicit (an explicit
   ``success()`` is kept) and ``!cancelled()`` where only the implicit ``success()`` must be
   removed (never ``always()``). A job's own ``defaults.run`` is kept when every step fits it
   (so matrix members keep one body); ``timeout-minutes`` (job and step) is emitted verbatim.
4. Matrix instances keep a ``strategy.matrix``: the raw one when the members are exactly its
   expansion (computed by ``lower.expand_matrix``), else an ``include:`` list with the
   original typed values; so ``${{ matrix.* }}`` references stay valid. ``name:`` is the
   first candidate whose lowering reproduces the members' check names.
5. Jobs expanded from a local reusable workflow are inlined; their ``needs.<job>``
   references (in raw text or in lowered ``needs['caller/job']`` form) are renamed to the
   emitted keys. A frozen call (lowering gives its jobs the caller's YAML as ``raw``,
   ``call_key``) is written back verbatim as the call, so it re-lowers identically.
6. Transfers use one idiom (``transfer_steps``, ``contracts.transfer_bindings``): tar the
   declared out paths relative to the workspace root, upload the tarball with
   ``if-no-files-found: error`` and ``retention-days: 1``, download, untar. Artifact names
   carry the reserved prefix ``ghav-``. Every emitted matrix (frozen ones included) states
   ``fail-fast`` explicitly.

Output: workflow and job keys in a fixed conventional order, ``jobs`` sorted by key,
``with``/``env`` sorted, step order kept, matrix keys in their original order (it determines
default check names). Strings that a YAML 1.1 or 1.2 reader would retype are quoted.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import yaml

from pathlib import Path

from ghav import contracts as gcontracts
from ghav import expr as gexpr
from ghav.contracts import TRANSFER_PREFIX, parse_transfer, transfer_bindings, valid_transfer_name
from ghav.lower import LowerError
from ghav.lower import default_check_name as _lower_default_check_name
from ghav.lower import expand_matrix as _lower_expand_matrix
from ghav.wir import TOP_CONTRACT, TRUE, Binding, Expr, JobInstance, Program, Step

UPLOAD_ACTION = gcontracts.TRANSFER_UPLOAD
DOWNLOAD_ACTION = gcontracts.TRANSFER_DOWNLOAD
TRANSFER_RETENTION_DAYS = 1
TRANSFER_ORIGIN_INDEX = -3  # origin index of transfer steps (never an original step)

SUCCESS = Expr("status", ("success",))
FALSE = Expr("lit", (False,))
NOT_CANCELLED = Expr("not", (Expr("status", ("cancelled",)),))

_WORKFLOW_KEY_ORDER = ("name", "on", "permissions", "concurrency", "env", "defaults", "jobs")
_JOB_KEY_ORDER = ("name", "needs", "if", "runs-on", "environment", "permissions", "strategy",
                  "timeout-minutes", "env", "defaults", "outputs", "steps")
_STEP_KEY_ORDER = ("name", "id", "if", "uses", "run", "shell", "working-directory", "with", "env",
                   "timeout-minutes")


class EmitError(Exception):
    """The program cannot be expressed as a GitHub Actions workflow."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def emit(p: Program) -> str:
    """Stable YAML text for ``p``."""
    header = f"# Generated by GHAV from {p.source}\n" if p.source else "# Generated by GHAV\n"
    body = yaml.dump(emit_program(p), Dumper=_Dumper, sort_keys=False,
                     default_flow_style=False, allow_unicode=True, width=1 << 30)
    return header + body


def emit_program(p: Program) -> dict:
    """The workflow as a plain mapping (key order is the emission order)."""
    plan = _plan(p)
    wf: dict[str, Any] = {}
    if p.name and p.name != p.source:  # lowering names unnamed workflows by their path
        wf["name"] = p.name
    wf["on"] = _plain(p.on) if p.on is not None else {}
    if p.permissions is not None:
        wf["permissions"] = _plain(p.permissions)
    if p.concurrency is not None:
        wf["concurrency"] = _plain(p.concurrency)
    if p.env:
        wf["env"] = {k: v for k, v in sorted(p.env)}
    if p.defaults is not None:
        wf["defaults"] = _plain(p.defaults)
    wf["jobs"] = {key: plan.bodies[key] for key in sorted(plan.bodies)}
    return {k: wf[k] for k in _WORKFLOW_KEY_ORDER if k in wf}


def transfer_name(producer: JobInstance, out_paths: Sequence[str], tag: str = "") -> str:
    """A transfer artifact name: the reserved prefix ``ghav-``, the producer's instance id
    (job key and every matrix value, sanitized), and a digest of the instance id, the paths
    and ``tag`` (distinct producers or path sets never share a name)."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", producer.id).strip("-")[:60] or "job"
    digest = hashlib.sha1(json.dumps([producer.id, sorted(out_paths), tag]).encode("utf-8")).hexdigest()[:8]
    return f"{TRANSFER_PREFIX}{base}-{digest}"


def transfer_steps(producer_job: JobInstance, consumer_job: JobInstance, out_paths: Sequence[str],
                   name: str, *, repo_root: Path | None = None) -> tuple[tuple[Step, Step], tuple[Step, Step]]:
    """The transfer idiom (the only way GHAV moves ``out`` between jobs), as
    ((pack, upload) for the producer, (download, unpack) for the consumer).

    The producer tars the declared ``out_paths`` (workspace-relative) into
    ``$RUNNER_TEMP/<name>.tar`` and uploads that file with ``if-no-files-found: error`` and
    ``retention-days: 1``; the consumer downloads it into ``$RUNNER_TEMP`` and extracts
    exactly those paths into the workspace root. Each pair must stay adjacent (the
    semantics treats the transfer as an exact copy of ``out`` only then, see
    ``semantics.transfer_plan``). ``name`` must be a valid ``ghav-`` name (``transfer_name``).
    Steps get the default condition ``success()``, origin ``(<orig job>, -3)``,
    ``origin_instance`` = the job's instance id, and their guide contracts (``repo_root``
    only refines ``out`` directories from tsconfig).
    """
    if not valid_transfer_name(name):
        raise EmitError(f"invalid transfer artifact name {name!r} (reserved prefix {TRANSFER_PREFIX})")
    try:
        pack, upload, download, unpack = transfer_bindings(name, out_paths)
    except ValueError as exc:
        raise EmitError(str(exc)) from exc
    root = Path(repo_root) if repo_root is not None else Path("/nonexistent-ghav-root")

    def mk(b: Binding, job: JobInstance) -> Step:
        step = Step(b, SUCCESS, TOP_CONTRACT, (job.orig_job, TRANSFER_ORIGIN_INDEX),
                    None, job.id)
        return Step(b, SUCCESS, gcontracts.infer(step, job, root), step.origin, None, job.id)

    return ((mk(pack, producer_job), mk(upload, producer_job)),
            (mk(download, consumer_job), mk(unpack, consumer_job)))


def job_keys(p: Program) -> dict[str, str]:
    """Instance id -> emitted job key."""
    plan = _plan(p)
    return {i: plan.keys[i] for i in sorted(plan.keys)}


_JOB_ID_BAD = re.compile(r"[^A-Za-z0-9_-]+")


MAX_JOB_ID_LENGTH = 64


def sanitize_job_id(s: str) -> str:
    """Map an arbitrary instance id to GitHub job-id syntax ``[A-Za-z_][A-Za-z0-9_-]*``.

    Ids longer than ``MAX_JOB_ID_LENGTH`` keep a readable prefix plus a stable hash."""
    out = _JOB_ID_BAD.sub("-", s).strip("-")
    out = re.sub(r"-{2,}", "-", out)
    if not out:
        return "job"
    if not (out[0].isalpha() or out[0] == "_"):
        out = "j-" + out
    if len(out) > MAX_JOB_ID_LENGTH:
        digest = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
        out = out[: MAX_JOB_ID_LENGTH - 9].rstrip("-") + "-" + digest
    return out


# ---------------------------------------------------------------------------
# Expression rendering
# ---------------------------------------------------------------------------

def _rename_refs(e: Expr, needs_map: Callable[[str], str], step_map: Callable[[str], str]) -> Expr:
    if e.op == "step":
        return Expr("step", (step_map(str(e.args[0])), *e.args[1:]))
    if e.op == "needs":
        return Expr("needs", (needs_map(str(e.args[0])), *e.args[1:]))
    if e.op == "opaque":
        return e
    return Expr(e.op, tuple(_rename_refs(a, needs_map, step_map) if isinstance(a, Expr) else a
                            for a in e.args))


def render_expr(e: Expr, needs_map: Callable[[str], str] | None = None,
                step_map: Callable[[str], str] | None = None) -> str:
    """GitHub expression text for ``e`` (without ``${{ }}``), via ``expr.unparse``.

    ``needs_map``/``step_map`` rename job and step references (identity by default). A
    reference that must be renamed but sits inside opaque text raises :class:`EmitError`.
    """
    nm = needs_map or (lambda s: s)
    sm = step_map or (lambda s: s)
    for sid in sorted(gexpr.step_refs(e)):
        if sm(sid) != sid and sid in _opaque_refs(e, "steps"):
            raise EmitError(f"cannot rename step {sid!r} inside opaque expression text")
    needs_rename = {j: nm(j) for j in sorted(gexpr.needs_refs(e)) if nm(j) != j}
    renamed = gexpr.rename_needs(e, needs_rename) if needs_rename else e
    return gexpr.unparse(_rename_refs(renamed, lambda s: s, sm))


def _opaque_refs(e: Expr, root: str) -> set[str]:
    out: set[str] = set()
    if e.op == "opaque":
        for ref in e.args[1:]:
            parts = str(ref).split(".")
            if parts[0].lower() == root and len(parts) > 1:
                out.add(parts[1])
        return out
    for a in e.args:
        if isinstance(a, Expr):
            out |= _opaque_refs(a, root)
    return out


def uses_status_function(e: Expr) -> bool:
    return gexpr.uses_status_function(e)


def _normalize(e: Expr) -> Expr:
    """Canonical form for comparing conditions: flattened, deduplicated, sorted and/or;
    ``true`` dropped from conjunctions and ``false`` from disjunctions; ==/!= operands sorted."""
    if e.op in ("and", "or"):
        unit = TRUE if e.op == "and" else Expr("lit", (False,))
        items: list[Expr] = []

        def flat(x: Expr) -> None:
            if x.op == e.op:
                flat(x.args[0])
                flat(x.args[1])
            else:
                items.append(_normalize(x))

        flat(e)
        uniq = sorted({repr(x): x for x in items if x != unit}.items())
        xs = [x for _, x in uniq]
        if not xs:
            return unit
        if len(xs) == 1:
            return xs[0]
        return Expr(e.op, tuple(xs))
    if e.op in ("eq", "ne"):
        a, b = _normalize(e.args[0]), _normalize(e.args[1])
        return Expr(e.op, tuple(sorted((a, b), key=repr)))
    if e.op in ("not", "call"):
        return Expr(e.op, tuple(_normalize(x) if isinstance(x, Expr) else x for x in e.args))
    return e


def _canonical_cond(e: Expr) -> Expr:
    return _normalize(gexpr.simplify(e, boolean=True))


def same_condition(a: Expr, b: Expr) -> bool:
    """Equal as ``if:`` conditions up to constant folding and and/or reordering."""
    return _canonical_cond(a) == _canonical_cond(b)


def _if_text(s: str) -> str:
    return "${{ " + s + " }}" if s.startswith("!") else s


# ---------------------------------------------------------------------------
# Predicting what lowering derives from raw text
# ---------------------------------------------------------------------------

def _matrix_envs(job: JobInstance) -> list[Mapping[str, Any]]:
    """Plausible substitution environments for the instance's matrix: the typed values from
    the raw YAML matrix (what lowering substitutes) and, as a fallback, the lowered strings."""
    if not job.matrix:
        return [{}]
    strings = {k: v for k, v in job.matrix}
    typed = _typed_combo(job) or strings
    envs: list[Mapping[str, Any]] = [{"matrix": typed}]
    if typed != strings:
        envs.append({"matrix": strings})
    return envs


def _predictions(raw_if: Any, job: JobInstance, default_fold: bool, use_matrix: bool) -> list[Expr]:
    """Conditions that lowering may derive from a raw ``if:`` value in the emitted context."""
    if raw_if is None or raw_if == "":
        return [SUCCESS if default_fold else TRUE]
    base = gexpr.parse(raw_if)
    out: list[Expr] = []
    for env in (_matrix_envs(job) if use_matrix else [{}]):
        e = gexpr.substitute(base, env) if env else base
        if default_fold and not uses_status_function(e):
            e = Expr("and", (SUCCESS, e))
        out.append(e)
    return out


def _reproduces(raw_if: Any, job: JobInstance, default_fold: bool, cond: Expr,
                use_matrix: bool = True) -> bool:
    return any(same_condition(e, cond)
               for e in _predictions(raw_if, job, default_fold, use_matrix))


# ---------------------------------------------------------------------------
# Value stringification (raw YAML values vs lowered strings)
# ---------------------------------------------------------------------------

def _str_candidates(v: Any) -> set[str]:
    """Every plausible lowered string for a raw YAML value."""
    out = {gexpr.to_str(v)}
    if v is None:
        return out | {"", "None", "null"}
    if isinstance(v, bool):
        return out | {str(v).lower(), str(v)}
    if isinstance(v, (int, float)):
        out |= {str(v), json.dumps(v)}
        if isinstance(v, float) and v.is_integer():
            out.add(str(int(v)))
        return out
    if isinstance(v, str):
        return out | {v}
    out.add(str(v))
    for sk in (False, True):
        for sep in ((",", ":"), (", ", ": ")):
            try:
                out.add(json.dumps(v, sort_keys=sk, separators=sep))
            except TypeError:
                pass
    return out


def _raw_matches(raw_value: Any, lowered: str | None, job: JobInstance) -> bool:
    """Does lowering ``raw_value`` in ``job``'s context (matrix interpolation) give ``lowered``?"""
    if lowered is None:
        return raw_value is None
    if raw_value is None:
        return False
    if lowered in _str_candidates(raw_value):
        return True
    if isinstance(raw_value, str) and job.matrix and "${{" in raw_value:
        return any(gexpr.interpolate(raw_value, env) == lowered for env in _matrix_envs(job))
    return False


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def _is_transfer(b: Binding) -> str | None:
    """The transfer-idiom role of a binding (pack, upload, download, unpack), else None."""
    t = parse_transfer(b)
    return None if t is None else t.role


def _run_defaults(defaults: Any) -> Mapping[str, Any]:
    run = defaults.get("run") if isinstance(defaults, Mapping) else None
    return run if isinstance(run, Mapping) else {}


def _raw_consistent(step: Step, job: JobInstance, wf_run_defaults: Mapping[str, Any]) -> bool:
    """Is ``step.binding`` what lowering derives from ``step.raw`` inside the emitted job?

    Emitted non-frozen jobs carry no job-level ``defaults``; only the workflow-level
    ``defaults.run`` (emitted verbatim) applies to ``run`` steps."""
    raw = step.raw
    if not isinstance(raw, Mapping):
        return False
    b = step.binding
    kind = "uses" if "uses" in raw else "run" if "run" in raw else None
    if kind != b.kind:
        return False
    text = raw.get(kind)
    if kind == "uses":
        if gexpr.to_str(text).strip() != b.text:
            return False
    elif not _raw_matches(text, b.text, job):
        return False
    wd, sh = raw.get("working-directory"), raw.get("shell")
    if kind == "run":
        wd = wd if wd is not None else wf_run_defaults.get("working-directory")
        sh = sh if sh is not None else wf_run_defaults.get("shell")
    for rv, lowered in ((wd, b.working_directory), (sh, b.shell), (raw.get("name"), b.name)):
        if not _raw_matches(rv, lowered, job):
            return False
    rid = raw.get("id")
    if (gexpr.to_str(rid) if rid is not None else None) != b.step_id:
        return False
    for key, pairs in (("with", b.with_), ("env", b.env)):
        rv = raw.get(key) or {}
        if not isinstance(rv, Mapping):
            return False
        lowered_map = dict(pairs)
        if sorted(str(k) for k in rv) != sorted(lowered_map):
            return False
        for k in rv:
            if not _raw_matches(rv[k], lowered_map[str(k)], job):
                return False
    return True


def _defaults_fit(job: JobInstance, run_defaults: Mapping[str, Any]) -> bool:
    """Can the job be emitted with these ``defaults.run``? Every ``run`` step must either be
    emitted from its raw YAML under them, or state the defaulted fields explicitly."""
    for s in job.steps:
        b = s.binding
        if b.kind != "run" or _raw_consistent(s, job, run_defaults):
            continue
        if ("shell" in run_defaults and b.shell is None) or \
                ("working-directory" in run_defaults and b.working_directory is None):
            return False
    return True


def _step_from_binding(b: Binding, raw: Mapping | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {}
    if b.name is not None:
        d["name"] = b.name
    if b.step_id:
        d["id"] = b.step_id
    d[b.kind] = b.text
    if b.shell:
        d["shell"] = b.shell
    if b.working_directory:
        d["working-directory"] = b.working_directory
    with_ = {k: v for k, v in sorted(b.with_)}
    if _is_transfer(b) == "upload":
        with_.setdefault("retention-days", TRANSFER_RETENTION_DAYS)
        with_.setdefault("if-no-files-found", "error")
    if with_:
        d["with"] = {k: with_[k] for k in sorted(with_)}
    if b.env:
        d["env"] = {k: v for k, v in sorted(b.env)}
    if isinstance(raw, Mapping) and "timeout-minutes" in raw:
        d["timeout-minutes"] = _plain(raw["timeout-minutes"])  # no correctness effect; verbatim
    return d


def _step_from_raw(raw: Mapping) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k in raw:
        if k == "if":
            continue
        v = _plain(raw[k])
        if k in ("with", "env") and isinstance(v, dict):
            v = {kk: v[kk] for kk in sorted(v, key=str)}
        d[str(k)] = v
    return d


def _ordered(d: Mapping[str, Any], order: Sequence[str]) -> dict[str, Any]:
    out = {k: d[k] for k in order if k in d}
    for k in sorted(d):
        if k not in out:
            out[k] = d[k]
    return out


_STEP_TEXT_REF = re.compile(r"steps\.([A-Za-z0-9_-]+)\.")


@dataclass
class _StepIds:
    """Emitted step ids of one job, with deterministic renaming of duplicates."""

    by_origin: dict[tuple[str, str], str]
    latest: dict[str, str]
    used: set[str]

    @staticmethod
    def new() -> "_StepIds":
        return _StepIds({}, {}, set())

    def resolve(self, origin_job: str, sid: str) -> str:
        return self.by_origin.get((origin_job, sid), self.latest.get(sid, sid))

    def register(self, origin_job: str, sid: str) -> str:
        new = sid
        n = 2
        while new in self.used:
            new = f"{sid}-{n}"
            n += 1
        self.used.add(new)
        self.by_origin[(origin_job, sid)] = new
        self.latest[sid] = new
        return new


def _emit_step(step: Step, job: JobInstance, ids: _StepIds, needs_map: Callable[[str], str],
               wf_run_defaults: Mapping[str, Any], text_rename: Mapping[str, str]) -> dict[str, Any]:
    origin_job = step.origin[0]

    def step_map(s: str) -> str:
        return ids.resolve(origin_job, s)

    if _raw_consistent(step, job, wf_run_defaults):
        d = _step_from_raw(step.raw)  # type: ignore[arg-type]
    else:
        d = _step_from_binding(step.binding, step.raw)
    d = _rewrite_needs_text(d, text_rename)

    # Text references to renamed step ids cannot be rewritten without changing the binding.
    b = step.binding
    texts = [b.text] + [v for _, v in b.with_] + [v for _, v in b.env]
    for t in texts:
        for sid in _STEP_TEXT_REF.findall(t):
            if step_map(sid) != sid:
                raise EmitError(f"step text in {job.id} references duplicated step id {sid!r}")

    # Condition.
    refs_ok = all(step_map(s) == s for s in sorted(gexpr.step_refs(step.cond))) and \
        all(needs_map(n) == n for n in sorted(gexpr.needs_refs(step.cond)))
    raw_if = step.raw.get("if") if isinstance(step.raw, Mapping) else None
    if refs_ok and isinstance(step.raw, Mapping) and \
            _reproduces(raw_if, job, default_fold=True, cond=step.cond):
        if raw_if is not None and raw_if != "":
            d["if"] = _plain(raw_if)
    else:
        text = _render_step_cond(step.cond, needs_map, step_map)
        if text is not None:
            d["if"] = text

    if b.step_id:
        new_id = ids.register(origin_job, b.step_id)
        if new_id != b.step_id:
            d["id"] = new_id
    return _ordered(d, _STEP_KEY_ORDER)


def _render_step_cond(cond: Expr, needs_map: Callable[[str], str],
                      step_map: Callable[[str], str]) -> str | None:
    n = _canonical_cond(cond)
    if n == SUCCESS:
        return None
    if n == TRUE:
        return _if_text("!cancelled()")
    if n == FALSE:
        return "false"
    cond = gexpr.simplify(cond, boolean=True)
    if uses_status_function(cond):
        e = cond  # status references stay explicit (lowering keeps them as written)
    else:
        e = Expr("and", (NOT_CANCELLED, cond))  # no status function: suppress the default fold
    return _if_text(render_expr(e, needs_map, step_map))


def _render_job_cond(cond: Expr, has_needs: bool, needs_map: Callable[[str], str]) -> str | None:
    n = _canonical_cond(cond)
    if n == SUCCESS:
        return None  # with needs: the default; without needs: success() is true
    if n == TRUE:
        return _if_text("!cancelled()") if has_needs else None
    if n == FALSE:
        return "false"
    cond = gexpr.simplify(cond, boolean=True)
    if not uses_status_function(cond):
        e = Expr("and", (NOT_CANCELLED, cond)) if has_needs else cond
    else:
        e = cond  # status references stay explicit (lowering keeps them as written)
    return _if_text(render_expr(e, needs_map))


def _render_output(e: Expr, needs_map: Callable[[str], str], step_map: Callable[[str], str]) -> str:
    if e.op == "opaque" and "${{" in str(e.args[0]):
        return str(e.args[0])
    if e.op == "lit" and isinstance(e.args[0], str):
        return e.args[0]
    return "${{ " + render_expr(e, needs_map, step_map) + " }}"


# ---------------------------------------------------------------------------
# Matrix handling (expansion is delegated to lowering so both agree exactly)
# ---------------------------------------------------------------------------

class _Unresolved(Exception):
    pass


def _resolve_static(v: Any) -> Any:
    """Resolve constant ``${{ }}`` values (e.g. ``fromJSON('[...]')``) as lowering does."""
    if isinstance(v, str) and "${{" in v:
        ok, value = gexpr.constant_value(gexpr.simplify(gexpr.parse(v)))
        if not ok:
            raise _Unresolved(v)
        return value
    if isinstance(v, list):
        return [_resolve_static(x) for x in v]
    if isinstance(v, Mapping):
        return {str(k): _resolve_static(x) for k, x in v.items()}
    return v


@functools.lru_cache(maxsize=512)
def _expansion_json(canon: str) -> str | None:
    try:
        resolved = _resolve_static(json.loads(canon))
        if not isinstance(resolved, Mapping):
            return None
        combos, order = _lower_expand_matrix(resolved)
    except (_Unresolved, LowerError):
        return None
    return json.dumps([combos, order])


def _expansion(m: Any) -> tuple[list[dict[str, Any]], list[str]] | None:
    """(combinations in GitHub order, matrix-key order) of a raw ``strategy.matrix``."""
    if not isinstance(m, Mapping):
        return None
    out = _expansion_json(json.dumps(_plain(m), default=str))
    if out is None:
        return None
    combos, order = json.loads(out)
    return combos, order


def expand_matrix(m: Mapping) -> list[dict[str, Any]] | None:
    """GitHub matrix expansion (cross product, exclude, include); None if not static."""
    e = _expansion(m)
    return None if e is None else e[0]


def _raw_matrix(job: JobInstance) -> Any:
    raw = job.raw if isinstance(job.raw, Mapping) else None
    strategy = raw.get("strategy") if raw else None
    return strategy.get("matrix") if isinstance(strategy, Mapping) else None


def _combo_matches(j: JobInstance, c: Mapping) -> bool:
    mv = dict(j.matrix)
    return sorted(mv) == sorted(str(k) for k in c) and \
        all(mv[str(k)] in _str_candidates(v) for k, v in c.items())


def _typed_combo(job: JobInstance) -> dict[str, Any] | None:
    """The instance's combination with its original YAML-typed values: from the raw matrix
    expansion when it is static, else by reading JSON-shaped lowered values back."""
    if not job.matrix:
        return None
    e = _expansion(_raw_matrix(job))
    for c in (e[0] if e else []):
        if _combo_matches(job, c):
            return c
    out: dict[str, Any] = {}
    for k, v in job.matrix:
        out[k] = v
        if v[:1] in ("{", "["):
            try:
                out[k] = json.loads(v)
            except ValueError:
                pass
    return out


def _expansion_rank(j: JobInstance) -> int:
    e = _expansion(_raw_matrix(j))
    combos = e[0] if e else []
    for i, c in enumerate(combos):
        if _combo_matches(j, c):
            return i
    return len(combos)


def _matrix_key_order(members: Sequence[JobInstance]) -> list[str]:
    e = _expansion(_raw_matrix(members[0]))
    order = list(e[1]) if e else []
    present = {k for j in members for k, _ in j.matrix}
    return [k for k in order if k in present] + sorted(present - set(order))


def _members_are_expansion(members: Sequence[JobInstance], m: Any) -> bool:
    combos = expand_matrix(m) if isinstance(m, Mapping) else None
    if combos is None or len(combos) != len(members):
        return False
    free = list(members)
    for c in combos:
        hit = next((j for j in free if _combo_matches(j, c)), None)
        if hit is None:
            return False
        free.remove(hit)
    return not free


def _strategy(members: Sequence[JobInstance]) -> dict[str, Any]:
    fail_fast = members[0].fail_fast_group is not None
    m = _raw_matrix(members[0])
    raw_strategy = members[0].raw.get("strategy") if isinstance(members[0].raw, Mapping) else None
    same_raw = all(_canon(_plain(_raw_matrix(j))) == _canon(_plain(m)) for j in members)
    if isinstance(raw_strategy, Mapping) and same_raw and _members_are_expansion(members, m):
        s = {str(k): _plain(v) for k, v in raw_strategy.items()}
        s["fail-fast"] = fail_fast  # always explicit (RESEARCH_SCOPE: emission writes fail-fast)
        return {k: s[k] for k in ("fail-fast", *[k for k in s if k != "fail-fast"])}
    return {"fail-fast": fail_fast, "matrix": {"include": _include_list(members)}}


def _include_list(members: Sequence[JobInstance]) -> list[dict[str, Any]]:
    """The members' combinations as an explicit ``include:`` list, in GitHub order."""
    order = _matrix_key_order(members)
    include = []
    for j in sorted(members, key=lambda x: (_expansion_rank(x), x.id)):
        typed = _typed_combo(j) or {}
        mv = dict(j.matrix)
        include.append({k: (_plain(typed[k]) if k in typed else mv[k]) for k in order if k in mv})
    return include


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Group:
    members: tuple[str, ...]      # sorted instance ids
    frozen: bool


@dataclass(frozen=True)
class _Plan:
    keys: dict[str, str]          # instance id -> job key
    bodies: dict[str, dict]       # job key -> body


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def call_key(job: JobInstance) -> str | None:
    """Top-level caller job key of a frozen reusable-workflow call (lowering gives every job
    of a frozen call the caller's YAML as ``raw``); None for any other instance."""
    if job.frozen and isinstance(job.raw, Mapping) and "uses" in job.raw:
        return job.orig_job.split("/", 1)[0]
    return None


def _initial_groups(p: Program) -> list[_Group]:
    frozen: dict[tuple[str, str], list[str]] = defaultdict(list)
    matrix: dict[tuple[str, str], list[str]] = defaultdict(list)
    singles: list[_Group] = []
    for j in sorted(p.jobs, key=lambda x: x.id):
        if j.frozen and isinstance(j.raw, Mapping):
            frozen[(call_key(j) or j.orig_job, _canon(_plain(j.raw)))].append(j.id)
        elif j.matrix:
            matrix[(j.orig_job, j.fail_fast_group or "")].append(j.id)
        else:
            singles.append(_Group((j.id,), False))
    groups = [_Group(tuple(sorted(v)), True) for _, v in sorted(frozen.items())]
    groups += [_Group(tuple(sorted(v)), False) for _, v in sorted(matrix.items())]
    groups += singles
    return sorted(groups, key=lambda g: g.members)


def _assign_keys(p: Program, groups: Sequence[_Group]) -> dict[str, str]:
    jobs = {j.id: j for j in p.jobs}
    count: dict[str, int] = defaultdict(int)
    for j in p.jobs:
        count[j.orig_job] += 1

    def preferred(g: _Group) -> str:
        first = jobs[g.members[0]]
        if g.frozen and call_key(first) is not None:
            return sanitize_job_id(call_key(first) or "")  # the call is emitted verbatim
        whole = len(g.members) == count[first.orig_job]
        if g.frozen or len(g.members) > 1 or whole:
            return sanitize_job_id(first.orig_job)
        return sanitize_job_id(first.id)

    prefs = [(preferred(g), g) for g in groups]
    all_prefs = {k for k, _ in prefs}
    used: set[str] = set()
    keys: dict[str, str] = {}
    for pref, g in sorted(prefs, key=lambda x: (x[0], -len(x[1].members), x[1].members)):
        key = pref
        n = 2
        while key in used or (key != pref and key in all_prefs):
            key = f"{pref}-{n}"
            n += 1
        used.add(key)
        for m in g.members:
            keys[m] = key
    return keys


def _needs_mapper(p: Program, keys: Mapping[str, str]) -> Callable[[str], str]:
    by_orig: dict[str, set[str]] = defaultdict(set)
    for j in p.jobs:
        by_orig[j.orig_job].add(keys[j.id])
        ck = call_key(j)
        if ck is not None:
            by_orig[ck].add(keys[j.id])

    def mapper(ref: str) -> str:
        if ref in keys:
            return keys[ref]
        if ref in by_orig and len(by_orig[ref]) == 1:
            return next(iter(by_orig[ref]))
        return ref

    return mapper


def _runs_on(job: JobInstance) -> Any:
    raw = job.raw if isinstance(job.raw, Mapping) else None
    rv = raw.get("runs-on") if raw else None
    if isinstance(rv, str) and _raw_matches(rv, job.runs_on, job):
        return rv
    if isinstance(rv, (list, Mapping)):
        return _plain(rv)  # label lists/groups: lowering resolved runs_on from this value
    return job.runs_on


def _job_env(job: JobInstance) -> dict[str, Any]:
    raw = job.raw if isinstance(job.raw, Mapping) else None
    renv = raw.get("env") if raw else None
    renv = renv if isinstance(renv, Mapping) else {}
    out: dict[str, Any] = {}
    for k, v in sorted(job.env):
        rv = renv.get(k)
        out[k] = _plain(rv) if rv is not None and _raw_matches(rv, v, job) else v
    return out


def _member_body(p: Program, job: JobInstance, keys: Mapping[str, str],
                 needs_map: Callable[[str], str]) -> dict[str, Any]:
    own = keys[job.id]
    needs = sorted({keys[n] if n in keys else sanitize_job_id(n) for n in job.needs} - {own})
    body: dict[str, Any] = {}
    if needs:
        body["needs"] = needs

    raw = job.raw if isinstance(job.raw, Mapping) else None
    refs_ok = all(needs_map(n) == n for n in sorted(gexpr.needs_refs(job.cond)))
    if refs_ok and raw is not None and \
            _reproduces(raw.get("if"), job, default_fold=bool(needs), cond=job.cond,
                        use_matrix=False):
        if raw is not None and raw.get("if") not in (None, ""):
            body["if"] = _plain(raw["if"])
    else:
        text = _render_job_cond(job.cond, bool(needs), needs_map)
        if text is not None:
            body["if"] = text

    body["runs-on"] = _runs_on(job)
    if raw is not None and "timeout-minutes" in raw:
        body["timeout-minutes"] = _plain(raw["timeout-minutes"])  # no correctness effect; verbatim
    if isinstance(job.permissions, str):
        body["permissions"] = job.permissions
    elif job.permissions:
        body["permissions"] = {k: v for k, v in sorted(job.permissions)}
    if job.env:
        body["env"] = _job_env(job)

    ids = _StepIds.new()
    run_defaults = _run_defaults(p.defaults)
    job_run = _run_defaults(raw.get("defaults")) if raw is not None else {}
    if job_run and _defaults_fit(job, {**run_defaults, **job_run}):
        # keep the job's own `defaults.run` (e.g. `shell: ${{ matrix.shell }}`), so raw steps
        # stay verbatim and matrix members keep one body
        run_defaults = {**run_defaults, **job_run}
        body["defaults"] = {"run": {str(k): _plain(v) for k, v in job_run.items()}}
    rename = _needs_text_rename(job, needs_map)
    if "env" in body:
        body["env"] = _rewrite_needs_text(body["env"], rename)
    steps = [_emit_step(s, job, ids, needs_map, run_defaults, rename) for s in job.steps]

    if job.outputs:
        def out_step_map(s: str) -> str:
            return ids.latest.get(s, s)
        renamed = any(out_step_map(s) != s for _, e in job.outputs for s in gexpr.step_refs(e))
        routs = raw.get("outputs") if raw else None
        if isinstance(routs, Mapping) and not renamed and \
                sorted(str(k) for k in routs) == sorted(k for k, _ in job.outputs) and \
                all(needs_map(n) == n for _, e in job.outputs for n in gexpr.needs_refs(e)) and \
                all(_output_reproduces(routs[k], e, job) for k, e in job.outputs):
            body["outputs"] = {str(k): _plain(routs[k]) for k in sorted(routs, key=str)}
        else:
            body["outputs"] = {k: _render_output(e, needs_map, out_step_map)
                               for k, e in sorted(job.outputs, key=lambda x: x[0])}
    body["steps"] = steps
    return body


def _output_reproduces(raw_value: Any, e: Expr, job: JobInstance) -> bool:
    base = gexpr.parse_template(raw_value)
    for env in _matrix_envs(job):
        if gexpr.simplify(gexpr.substitute(base, env) if env else base) == e:
            return True
    return False


def _lowered_name(candidate: Any, job: JobInstance, key: str, order: Sequence[str]) -> set[str]:
    """Check names lowering derives for ``job`` emitted under ``key`` with ``name: candidate``
    (GitHub appends the matrix values to a name without ``${{ }}``)."""
    if isinstance(candidate, str) and "${{" in candidate:
        return {gexpr.interpolate(candidate, env) for env in _matrix_envs(job)}
    base = key if candidate is None else gexpr.to_str(candidate)
    combo = (_typed_combo(job) or {}) if job.matrix else None
    return {_lower_default_check_name(base, combo, list(order))}


def _job_name(members: Sequence[JobInstance], key: str) -> Any:
    """``name:`` for an emitted job; ``None`` to omit it.

    The first candidate whose lowering reproduces every member's check name wins: no name,
    the raw name, the raw name behind the literal prefix lowering added (expanded reusable
    workflows: ``caller / callee``), the check name without its matrix suffix as a static
    name (GitHub appends the values), the same with the suffix spelled as ``${{ matrix.* }}``
    expressions, and finally the literal check name."""
    job = members[0]
    order = _matrix_key_order(members) if job.matrix else []
    raw = job.raw if isinstance(job.raw, Mapping) else None
    raw_name = raw.get("name") if raw is not None else None
    candidates: list[Any] = [None]
    if raw_name is not None:
        candidates.append(_plain(raw_name))
        if isinstance(raw_name, str):
            own = gexpr.interpolate(raw_name, _matrix_envs(job)[0])
            if own and job.check_name.endswith(own) and job.check_name != own:
                prefix = job.check_name[: -len(own)]
                if "${{" not in prefix:
                    candidates.append(prefix + raw_name)
    if job.matrix:
        values = dict(job.matrix)
        own_keys = [k for k in order if k in values]
        suffix = " (" + ", ".join(values[k] for k in own_keys) + ")"
        if job.check_name.endswith(suffix):
            base = job.check_name[: -len(suffix)]
            if "${{" not in base:
                candidates.append(base)  # static name: GitHub appends the matrix values
                candidates.append(base + " (" + ", ".join("${{ matrix." + k + " }}" for k in own_keys) + ")")
    candidates.append(job.check_name)
    for c in candidates:
        if all(m.check_name in _lowered_name(c, m, key, order) for m in members):
            return c
    return job.check_name


_NEEDS_TEXT = re.compile(r"(?<![A-Za-z0-9_.-])needs\.([A-Za-z0-9_\-/\[\]=,.#]+?)\.(result|outputs)\b")
_NEEDS_INDEX = re.compile(r"(?<![A-Za-z0-9_.-])needs\[\s*'((?:[^']|'')*)'\s*\]\.(result|outputs)\b")


def _needs_text_rename(job: JobInstance, needs_map: Callable[[str], str]) -> dict[str, str]:
    """Job references that may appear in ``job``'s texts, mapped to emitted job keys.

    Covers lowered group keys (``caller/callee``) and, for jobs expanded from a reusable
    workflow, the callee-local names used in their raw YAML."""
    refs: dict[str, str] = {}
    local_prefix = ""
    if "/" in job.orig_job:
        local = job.orig_job.rsplit("/", 1)[1]
        group = job.id.split("[", 1)[0] if not job.id.endswith("]") else job.id[: job.id.rfind("[")]
        if group.endswith("/" + local):
            local_prefix = group[: -len(local)]
    for n in sorted(job.needs):
        group = n[: n.rfind("[")] if n.endswith("]") else n
        for ref in (n, group):
            refs[ref] = needs_map(ref)
        if local_prefix and group.startswith(local_prefix) and "/" not in group[len(local_prefix):]:
            refs[group[len(local_prefix):]] = needs_map(group)
    return {k: v for k, v in sorted(refs.items()) if k != v}


def _renamed_index(m: re.Match, rename: Mapping[str, str]) -> str:
    ref = m.group(1).replace("''", "'")
    if ref in rename:
        return f"needs.{rename[ref]}.{m.group(2)}"
    return m.group(0)


def _rewrite_needs_text(obj: Any, rename: Mapping[str, str]) -> Any:
    """Rename ``needs.<job>`` references inside ``${{ }}`` segments of every string."""
    if not rename:
        return obj
    if isinstance(obj, str):
        if "${{" not in obj or "needs" not in obj:
            return obj
        out: list[str] = []
        for is_expr, _inner, raw in gexpr.template_segments(obj):
            if is_expr:
                raw = _NEEDS_TEXT.sub(lambda m: f"needs.{rename.get(m.group(1), m.group(1))}.{m.group(2)}", raw)
                raw = _NEEDS_INDEX.sub(lambda m: _renamed_index(m, rename), raw)
            out.append(raw)
        return "".join(out)
    if isinstance(obj, list):
        return [_rewrite_needs_text(x, rename) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite_needs_text(v, rename) for k, v in obj.items()}
    return obj


def _patch_frozen_step(d: Any, occurrences: Sequence[tuple[Step, JobInstance]],
                       run_defaults: Mapping[str, Any]) -> Any:
    """Make a verbatim step's working directory and shell explicit when the emitted context
    would resolve them differently (e.g. defaults of an inlined reusable workflow).

    ``occurrences`` are the step at this position in every member instance of the frozen
    group. A default is kept when lowering it (with each member's matrix, as in
    ``defaults.run.shell: ${{ matrix.shell }}``) gives every member's value back; otherwise
    the value is written explicitly when all members agree on it."""
    step0 = occurrences[0][0]
    if not isinstance(d, dict) or step0.binding.kind != "run":
        return d
    out = dict(d)
    for key, attr in (("working-directory", "working_directory"), ("shell", "shell")):
        values = [getattr(st.binding, attr) for st, _ in occurrences]
        if key in out or any(v is None for v in values):
            continue
        default = run_defaults.get(key)
        if all(_raw_matches(default, v, j) for v, (_, j) in zip(values, occurrences)):
            continue
        if len(set(values)) == 1:
            out[key] = values[0]
    return _ordered(out, _STEP_KEY_ORDER) if out != d else d


def _frozen_body(p: Program, members: Sequence[JobInstance], keys: Mapping[str, str],
                 needs_map: Callable[[str], str]) -> dict:
    """The raw job, verbatim except where lowering the raw text in the emitted workflow would
    not give the instance back: ``needs`` (emitted keys), ``if`` and ``name`` (jobs expanded
    from a reusable workflow carry their caller's condition and check-name prefix), and
    ``needs.<job>`` references to renamed jobs."""
    job = members[0]
    raw = job.raw
    assert isinstance(raw, Mapping)
    key = keys[job.id]
    needs = sorted({keys[n] if n in keys else sanitize_job_id(n)
                    for j in members for n in j.needs} - {key})
    if call_key(job) is not None:
        return _call_body(p, members, keys, needs_map, needs)
    rename = _needs_text_rename(job, needs_map)
    body: dict[str, Any] = {str(k): _rewrite_needs_text(_plain(v), rename)
                            for k, v in raw.items()}
    raw_needs = raw.get("needs")
    raw_list = [raw_needs] if isinstance(raw_needs, str) else list(raw_needs or [])
    if sorted(map(str, raw_list)) == needs:
        body["needs"] = _plain(raw_needs) if raw_needs is not None else None
    else:
        body["needs"] = needs
    if not body["needs"]:
        del body["needs"]

    strategy = body.get("strategy")
    if job.matrix and isinstance(strategy, dict) and \
            not _members_are_expansion(members, _raw_matrix(job)):
        # Only some combinations are frozen (e.g. a matrix-dependent continue-on-error).
        body["strategy"] = {**strategy, "matrix": {"include": _include_list(members)}}
    if "strategy" in body:
        body["strategy"] = _explicit_fail_fast(body["strategy"], job)
    raw_env = raw.get("env")
    raw_env_keys = sorted(map(str, raw_env)) if isinstance(raw_env, Mapping) else []
    if job.env and raw_env_keys != sorted(k for k, _ in job.env):
        body["env"] = _rewrite_needs_text(_job_env(job), rename)  # e.g. a callee's workflow env
    if job.steps and isinstance(body.get("steps"), list) and \
            all(len(body["steps"]) == len(m.steps) for m in members):
        run_defaults = {**_run_defaults(p.defaults), **_run_defaults(raw.get("defaults"))}
        body["steps"] = [_patch_frozen_step(d, [(m.steps[i], m) for m in members], run_defaults)
                         for i, d in enumerate(body["steps"])]

    refs_ok = all(needs_map(n) == n for n in sorted(gexpr.needs_refs(job.cond)))
    if not (refs_ok and _reproduces(raw.get("if"), job, default_fold=bool(needs),
                                    cond=job.cond, use_matrix=False)):
        text = _render_job_cond(job.cond, bool(needs), needs_map)
        body.pop("if", None)
        if text is not None:
            body["if"] = text
    name = _job_name(members, key)
    if name != _plain(raw.get("name")):
        body.pop("name", None)
        if name is not None:
            body["name"] = name
    # Keep the raw key order; keys added above go to their conventional position.
    order = [str(k) for k in raw if str(k) in body]
    for k in _JOB_KEY_ORDER:
        if k in body and k not in order:
            pos = next((i for i, o in enumerate(order)
                        if o in _JOB_KEY_ORDER and _JOB_KEY_ORDER.index(o) > _JOB_KEY_ORDER.index(k)),
                       len(order))
            order.insert(pos, k)
    return {k: body[k] for k in order}


def _explicit_fail_fast(strategy: Any, job: JobInstance) -> Any:
    """A raw strategy with ``fail-fast`` written explicitly (RESEARCH_SCOPE: emission writes
    fail-fast on every matrix); unchanged when it already states it or has no matrix."""
    if not isinstance(strategy, dict) or "matrix" not in strategy or "fail-fast" in strategy:
        return strategy
    return {"fail-fast": job.fail_fast_group is not None, **strategy}


def _call_body(p: Program, members: Sequence[JobInstance], keys: Mapping[str, str],
               needs_map: Callable[[str], str], needs: list[str]) -> dict:
    """A frozen reusable-workflow call, written back as the caller's YAML: verbatim except
    ``needs`` (emitted keys of the jobs outside the call), ``needs.<job>`` references to
    renamed jobs, and an explicit ``fail-fast`` on its matrix."""
    job = members[0]
    raw = job.raw
    assert isinstance(raw, Mapping)
    ids = {m.id for m in members}
    rename: dict[str, str] = {}
    by_id = {j.id: j for j in p.jobs}
    for m in members:
        for n in sorted(m.needs - ids):
            dep = by_id.get(n)
            top = (call_key(dep) or dep.orig_job.split("/", 1)[0]) if dep is not None else n
            if top != keys.get(n, top):
                rename[top] = keys[n]
    body: dict[str, Any] = {str(k): _rewrite_needs_text(_plain(v), rename) for k, v in raw.items()}
    raw_needs = raw.get("needs")
    raw_list = [raw_needs] if isinstance(raw_needs, str) else list(raw_needs or [])
    if sorted(map(str, raw_list)) == needs:
        body["needs"] = _plain(raw_needs) if raw_needs is not None else None
    else:
        body["needs"] = needs
    if not body["needs"]:
        del body["needs"]
    if "strategy" in body:
        body["strategy"] = _explicit_fail_fast(body["strategy"], job)
    order = [str(k) for k in raw if str(k) in body]
    order += [k for k in _JOB_KEY_ORDER if k in body and k not in order]
    return {k: body[k] for k in order}


def _plan(p: Program) -> _Plan:
    jobs = {j.id: j for j in p.jobs}
    if len(jobs) != len(p.jobs):
        raise EmitError("duplicate job instance ids")
    dependents: dict[str, frozenset[str]] = {
        i: frozenset(j.id for j in p.jobs if i in j.needs) for i in sorted(jobs)}
    groups = _initial_groups(p)
    while True:
        keys = _assign_keys(p, groups)
        needs_map = _needs_mapper(p, keys)
        refined: list[_Group] = []
        for g in groups:
            if g.frozen or len(g.members) == 1:
                refined.append(g)
                continue
            buckets: dict[str, list[str]] = defaultdict(list)
            for m in g.members:
                job = jobs[m]
                sig = _canon([sorted(dependents[m]), sorted(job.needs),
                              _member_body(p, job, keys, needs_map)])
                buckets[sig].append(m)
            for _, ms in sorted(buckets.items()):
                refined.append(_Group(tuple(sorted(ms)), False))
        refined.sort(key=lambda g: g.members)
        if refined == groups:
            break
        groups = refined

    bodies: dict[str, dict] = {}
    for g in groups:
        members = [jobs[m] for m in g.members]
        key = keys[g.members[0]]
        if g.frozen:
            bodies[key] = _frozen_body(p, members, keys, needs_map)
            continue
        first = members[0]
        body = _member_body(p, first, keys, needs_map)
        name = _job_name(members, key)
        if name is not None:
            body["name"] = name
        if first.matrix:
            body["strategy"] = _strategy(members)
        bodies[key] = _ordered(body, _JOB_KEY_ORDER)
    return _Plan(keys=keys, bodies=bodies)


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

def _plain(obj: Any) -> Any:
    """Deep copy into plain dict/list/scalars (keeps mapping order)."""
    if isinstance(obj, Mapping):
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): _plain(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(x) for x in obj]
    return obj


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True

    def increase_indent(self, flow: bool = False, indentless: bool = False):  # type: ignore[override]
        return super().increase_indent(flow, False)


# Plain scalars that a YAML 1.2 core-schema reader (GitHub, ghav.lower) or a YAML 1.1 reader
# would not read back as strings; such strings are always quoted.
_NON_STRING_PLAIN = re.compile(
    r"^(?:null|Null|NULL|~|true|True|TRUE|false|False|FALSE"
    r"|[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+"
    r"|[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
)


def _repr_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    if data == "" or _NON_STRING_PLAIN.match(data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def _repr_none(dumper: yaml.SafeDumper, data: None) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:null", "")


_Dumper.add_representer(str, _repr_str)
_Dumper.add_representer(type(None), _repr_none)
