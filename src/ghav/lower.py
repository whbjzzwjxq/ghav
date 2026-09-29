"""GHA Core -> WIR lowering.

`lower(workflow_path, repo_root)` reads one workflow file and produces a `Program`:

* Matrix expansion follows GitHub: cross product of the dimensions, then `exclude`, then
  `include` (an include entry extends every post-exclude combination whose *original*
  values it agrees with, overwriting only include-added keys; otherwise it becomes a new
  combination). One `JobInstance` per combination.
* Local reusable workflows (`uses: ./.github/workflows/x.yml` at job level) are expanded
  recursively. Callee jobs become instances with id `<caller-instance>/<callee-job>[...]`,
  `inputs.*` and passed `secrets.*` are substituted, callee-internal `needs` are rewritten to
  instance ids (and, in expressions, to the callee *group* key `<caller-instance>/<job>`),
  and the check name is `<caller check> / <callee check>`.
* Local composite actions (`uses: ./path` at step level) stay one step.
* Default conditions are folded in (see `_combine_cond`).
* Constructs outside GHA Core freeze a job (see `FROZEN_JOB_KEYS` and `frozen_reasons`).
  Lowering fails closed: any job, step, strategy, `defaults` or workflow key it does not know
  freezes (`unsupported-*`). `timeout-minutes` has no correctness effect (every task is
  assumed to terminate): it is kept in `raw`, emitted verbatim, and freezes nothing.
* Every step records its instance in `Step.origin_instance` (finer check groups, occurrence
  keys of external writes, historical timings).
* Post-lowering rules (`_apply_core_rules`, RESEARCH_SCOPE "Verification gaps and
  resolutions") freeze what the checker cannot transform: readers of `needs.<M>.result` of a
  multi-instance M; readers of `needs.<M>.outputs` unless every instance of M produces the
  same term; fail-fast matrices that violate the fail-fast rule; and every job of a
  top-level reusable-workflow call that has a frozen job or is named by `needs.<call>`
  (such a call is kept verbatim: `raw` of its jobs is the caller's YAML). The fail-fast rule
  and the output rule need guide contracts, which are computed here for those steps only.
* Contracts are not stored: every step gets `wir.TOP_CONTRACT` (see `contracts.annotate`).

Substitution policy. Known matrix values, reusable-workflow inputs and passed secrets are
substituted into every lowered field (`Binding` texts, `with`/`env` values, names,
conditions, `runs_on`, job `env` and `outputs`). Other `${{ }}` text is kept. `Step.raw`
and `JobInstance.raw` are the original YAML mappings; for jobs expanded from a reusable
workflow they are copies with `inputs.*`/`secrets.*` substituted (the callee's own
`matrix.*` stays symbolic), so they are valid in the caller workflow.

Determinism: jobs are processed in a topological order with sorted tie-breaks, every
mapping that is turned into a WIR tuple is sorted, and matrix combinations keep the YAML
order (which GitHub also uses for check names).
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

import yaml

from ghav import expr as X
from ghav.wir import TOP_CONTRACT, Binding, Expr, JobInstance, Program, Step

__all__ = [
    "FAIL_FAST_REASON",
    "FROZEN_CALL_REASON",
    "FROZEN_JOB_KEYS",
    "FROZEN_STEP_KEYS",
    "KNOWN_CALL_JOB_KEYS",
    "KNOWN_JOB_KEYS",
    "KNOWN_NORMAL_JOB_KEYS",
    "KNOWN_STEP_KEYS",
    "KNOWN_STRATEGY_KEYS",
    "KNOWN_WORKFLOW_KEYS",
    "MULTI_OUTPUT_REASON",
    "MULTI_RESULT_REASON",
    "may_run_after_cancel",
    "LowerError",
    "default_check_name",
    "expand_matrix",
    "instance_id",
    "load_yaml",
    "lower",
    "needs_members",
]

# Keys of a job with steps and of a reusable-workflow call (lowering fails closed: any other
# key freezes the job with reason "unsupported-key:<k>", e.g. `environment`, whose protection
# rules and deployment records are not modeled, or `cache-mode`). `timeout-minutes` has no
# correctness effect (every task is assumed to terminate) and is emitted verbatim.
KNOWN_NORMAL_JOB_KEYS = frozenset(
    {"defaults", "env", "if", "name", "needs", "outputs", "permissions", "runs-on", "steps", "strategy",
     "timeout-minutes"}
)
KNOWN_CALL_JOB_KEYS = frozenset({"if", "name", "needs", "permissions", "secrets", "strategy", "uses", "with"})
KNOWN_JOB_KEYS = KNOWN_NORMAL_JOB_KEYS | KNOWN_CALL_JOB_KEYS
# Job keys outside GHA Core (frozen with a named reason).
FROZEN_JOB_KEYS = {
    "concurrency": "concurrency",
    "container": "container",
    "continue-on-error": "continue-on-error",
    "services": "services",
}
KNOWN_STEP_KEYS = frozenset(
    {"env", "id", "if", "name", "run", "shell", "timeout-minutes", "uses", "with", "working-directory"}
)
FROZEN_STEP_KEYS = {
    "continue-on-error": "step-continue-on-error",
}
KNOWN_STRATEGY_KEYS = frozenset({"fail-fast", "matrix", "max-parallel"})
KNOWN_WORKFLOW_KEYS = frozenset({"concurrency", "defaults", "env", "jobs", "name", "on", "permissions", "run-name"})
KNOWN_RUN_DEFAULT_KEYS = frozenset({"shell", "working-directory"})

# Reasons added by the post-lowering rules (RESEARCH_SCOPE, "Verification gaps and resolutions").
FAIL_FAST_REASON = "fail-fast"                    # a fail-fast matrix the checker cannot transform
MULTI_RESULT_REASON = "multi-instance-result"     # reads needs.<M>.result of a multi-instance M
MULTI_OUTPUT_REASON = "multi-instance-output"     # reads needs.<M>.outputs whose instances differ
FROZEN_CALL_REASON = "frozen-call"                # a job of a reusable-workflow call with a frozen job
MAX_MATRIX_COMBINATIONS = 256
MAX_REUSABLE_DEPTH = 10


class LowerError(Exception):
    """The workflow is not valid GitHub Actions YAML (lowering cannot proceed)."""


# ---------------------------------------------------------------------------
# YAML (1.2 core schema, as used by GitHub: `on`/`yes`/`no` are strings, no timestamps)
# ---------------------------------------------------------------------------

class _Loader(yaml.SafeLoader):
    pass


_REPLACED_TAGS = {
    "tag:yaml.org,2002:bool",
    "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float",
    "tag:yaml.org,2002:timestamp",
}
_Loader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for tag, rx in resolvers if tag not in _REPLACED_TAGS]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$"),
    list("-+0123456789"),
)
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
        r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)


def _construct_int(loader: yaml.SafeLoader, node: yaml.Node) -> int:
    text = loader.construct_scalar(node)
    if text.startswith("0o"):
        return int(text[2:], 8)
    if text.startswith("0x"):
        return int(text[2:], 16)
    return int(text, 10)


_Loader.add_constructor("tag:yaml.org,2002:int", _construct_int)


def load_yaml(text: str) -> object:
    """Parse YAML with the 1.2 core schema. Raises LowerError on malformed input."""
    try:
        return yaml.load(text, Loader=_Loader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise LowerError(f"invalid YAML: {exc}") from exc


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

def _value_eq(a: object, b: object) -> bool:
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool):
        return float(a) == float(b)
    return type(a) is type(b) and a == b


def expand_matrix(matrix: Mapping) -> tuple[list[dict], list[str]]:
    """Expand a static `strategy.matrix` mapping.

    Returns (combinations in GitHub order, key order). The key order is the YAML order of the
    dimensions followed by include-added keys in order of first appearance; it is the order
    GitHub uses for default check names.
    """
    if not isinstance(matrix, Mapping):
        raise LowerError("strategy.matrix must be a mapping")
    dims: list[tuple[str, list]] = []
    for k, v in matrix.items():
        if k in ("include", "exclude"):
            continue
        if not isinstance(v, list):
            raise LowerError(f"matrix dimension {k!r} is not a list")
        dims.append((str(k), v))
    key_order = [k for k, _ in dims]
    base: list[dict] = [{}] if dims else []
    for k, values in dims:
        base = [{**c, k: v} for c in base for v in values]
    excludes = matrix.get("exclude") or []
    includes = matrix.get("include") or []
    if not isinstance(excludes, list) or not all(isinstance(e, Mapping) for e in excludes):
        raise LowerError("matrix exclude must be a list of mappings")
    if not isinstance(includes, list) or not all(isinstance(e, Mapping) for e in includes):
        raise LowerError("matrix include must be a list of mappings")

    def excluded(c: dict) -> bool:
        return any(all(str(k) in c and _value_eq(c[str(k)], v) for k, v in ex.items()) for ex in excludes)

    combos = [c for c in base if not excluded(c)]
    original = set(key_order)
    extended = [dict(c) for c in combos]
    added: list[dict] = []
    for inc in includes:
        entry = {str(k): v for k, v in inc.items()}
        matched = False
        for i, c in enumerate(combos):
            if all(_value_eq(c[k], v) for k, v in entry.items() if k in original):
                matched = True
                for k, v in entry.items():
                    if k not in original:
                        extended[i][k] = v
        if not matched:
            added.append(dict(entry))
        for k in entry:
            if k not in key_order:
                key_order.append(k)
    result = extended + added
    if not result:
        raise LowerError("matrix produces no job")
    if len(result) > MAX_MATRIX_COMBINATIONS:
        raise LowerError(f"matrix produces {len(result)} jobs (> {MAX_MATRIX_COMBINATIONS})")
    return result, key_order


def _matrix_pairs(combo: Mapping | None) -> tuple[tuple[str, str], ...]:
    if not combo:
        return ()
    return tuple(sorted((k, X.to_str(v)) for k, v in combo.items()))


def instance_id(job: str, matrix: tuple[tuple[str, str], ...]) -> str:
    """`job[k1=v1,k2=v2]` over the sorted matrix pairs, or `job` without a matrix."""
    if not matrix:
        return job
    return f"{job}[{','.join(f'{k}={v}' for k, v in sorted(matrix))}]"


def default_check_name(job: str, combo: Mapping | None, key_order: list[str]) -> str:
    """GitHub's check name for a job without `name:`: `job (v1, v2, ...)` in matrix-key order.

    GitHub applies the same suffix to a literal `name:` (one without `${{ }}`); a name that
    contains an expression is used as interpolated, without a suffix.
    """
    if not combo:
        return job
    values = [X.to_str(combo[k]) for k in key_order if k in combo]
    return f"{job} ({', '.join(values)})"


def needs_members(program: Program, key: str) -> tuple[str, ...]:
    """Instance ids denoted by a `needs.<key>` reference (or a `needs:` entry) after lowering.

    `key` is a top-level job key, or a callee group key `<caller-instance>/<callee-job>`
    (the form lowering writes into expressions inside expanded reusable workflows). Members
    are the instances whose id is `key`, or starts with `key[` (matrix instances) or `key/`
    (jobs of an expanded reusable workflow).
    """
    return tuple(
        j.id for j in program.jobs if j.id == key or j.id.startswith(key + "[") or j.id.startswith(key + "/")
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_on(on: object) -> dict:
    if isinstance(on, str):
        return {on: None}
    if isinstance(on, list):
        return {X.to_str(x): None for x in on}
    if isinstance(on, Mapping):
        return dict(on)
    return {}


def _resolve_value(v: object, env: Mapping[str, object]) -> tuple[bool, object]:
    """Resolve `${{ }}` inside a YAML value to a concrete value, if possible."""
    if isinstance(v, str):
        if "${{" not in v:
            return True, v
        segs = X.template_segments(v)
        if len(segs) == 1 and segs[0][0]:
            return X.constant_value(X.simplify(X.substitute(X.parse(segs[0][1]), env)))
        s = X.interpolate(v, env)
        return ("${{" not in s), s
    if isinstance(v, list):
        out = []
        for x in v:
            ok, r = _resolve_value(x, env)
            if not ok:
                return False, None
            out.append(r)
        return True, out
    if isinstance(v, Mapping):
        out_map = {}
        for k, x in v.items():
            ok, r = _resolve_value(x, env)
            if not ok:
                return False, None
            out_map[k] = r
        return True, out_map
    return True, v


def _interp_obj(obj: object, env: Mapping[str, object]) -> object:
    """Recursively interpolate every string in a YAML value."""
    if not env:
        return obj
    if isinstance(obj, str):
        return X.interpolate(obj, env)
    if isinstance(obj, list):
        return [_interp_obj(x, env) for x in obj]
    if isinstance(obj, Mapping):
        return {k: _interp_obj(v, env) for k, v in obj.items()}
    return obj


def _norm_permissions(p: object) -> tuple[tuple[str, str], ...] | str | None:
    if p is None:
        return None
    if isinstance(p, str):
        return p
    if isinstance(p, Mapping):
        return tuple(sorted((str(k), X.to_str(v)) for k, v in p.items()))
    return X.to_str(p)


def _defaults_reasons(defaults: object) -> set[str]:
    """Fail closed on `defaults` other than `run.shell` / `run.working-directory`."""
    if defaults is None:
        return set()
    if not isinstance(defaults, Mapping):
        return {"unsupported-defaults"}
    out = {f"unsupported-defaults-key:{k}" for k in map(str, defaults) if k != "run"}
    run = defaults.get("run")
    if run is not None and not isinstance(run, Mapping):
        out.add("unsupported-defaults")
    elif isinstance(run, Mapping):
        out |= {f"unsupported-defaults-key:run.{k}" for k in map(str, run) if k not in KNOWN_RUN_DEFAULT_KEYS}
    return out


def _workflow_reasons(doc: Mapping, *, callee: bool) -> tuple[str, ...]:
    """Reasons that freeze every job of a workflow: unknown top-level keys, unsupported
    `defaults`, and (for a called workflow) workflow-level `concurrency`."""
    out = {f"unsupported-workflow-key:{k}" for k in map(str, doc) if k not in KNOWN_WORKFLOW_KEYS}
    out |= _defaults_reasons(doc.get("defaults"))
    if callee and doc.get("concurrency") is not None:
        out.add("concurrency")
    return tuple(sorted(out))


def _run_defaults(*layers: object) -> dict:
    out: dict = {}
    for layer in layers:
        if isinstance(layer, Mapping) and isinstance(layer.get("run"), Mapping):
            out.update(layer["run"])
    return out


def _needs_list(job: Mapping, key: str, jobs: Mapping) -> list[str]:
    n = job.get("needs")
    if n is None:
        return []
    if isinstance(n, str):
        n = [n]
    if not isinstance(n, list) or not all(isinstance(x, str) for x in n):
        raise LowerError(f"job {key!r}: needs must be a string or a list of strings")
    for x in n:
        if x not in jobs:
            raise LowerError(f"job {key!r} needs unknown job {x!r}")
    return sorted(set(n))


def _topo_order(jobs: Mapping) -> list[str]:
    deps = {k: set(_needs_list(jobs[k], k, jobs)) for k in sorted(jobs)}
    order: list[str] = []
    done: set[str] = set()
    while len(order) < len(deps):
        ready = sorted(k for k in deps if k not in done and deps[k] <= done)
        if not ready:
            raise LowerError("cyclic needs among jobs " + ", ".join(sorted(set(deps) - done)))
        order.extend(ready)
        done.update(ready)
    return order


def _combine_cond(sides: tuple[tuple[frozenset[str], Expr | None], ...]) -> tuple[Expr, bool]:
    """Fold default conditions.

    A side is (needs instance ids, parsed `if:` or None) for one nesting level: the job itself,
    and for jobs of an expanded reusable workflow also each enclosing caller job. With no
    status function anywhere, the condition is `success() && if_1 && ...` (`success()` only if
    some side has needs). If exactly one side uses a status function and no other side has
    needs, the status function already ranges over all needs and the `if`s are conjoined as
    they are. Otherwise the condition cannot be written with one job-level status scope and
    the second component is False (the caller freezes the job).
    """
    all_needs: frozenset[str] = frozenset().union(*(n for n, _ in sides)) if sides else frozenset()
    ifs = [e for _, e in sides if e is not None]
    status = [i for i, (_, e) in enumerate(sides) if e is not None and X.uses_status_function(e)]
    if not status:
        return X.conj(([X.SUCCESS] if all_needs else []) + ifs), True
    other_needs = [n for i, (n, _) in enumerate(sides) if i not in status and n]
    if len(status) == 1 and not other_needs:
        return X.conj(ifs), True
    return X.conj(ifs), False


def _mentions_needs_opaque(e: Expr) -> bool:
    stack = [e]
    while stack:
        n = stack.pop()
        if n.op == "opaque":
            if any(r.split(".", 1)[0].lower() == "needs" for r in n.args[1:]):
                return True
            continue
        if n.op in ("lit", "ctx", "status", "step", "needs"):
            continue
        stack.extend(a for a in (n.args[1:] if n.op == "call" else n.args) if isinstance(a, Expr))
    return False


def _coerce_input(value: object, typ: str) -> object:
    if isinstance(value, Expr):
        return value
    if typ == "boolean":
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "false"):
                return low == "true"
        return X.truthy(value) if not isinstance(value, bool) else value
    if typ == "number":
        if isinstance(value, int | float) and not isinstance(value, bool):
            return value
        n = X.to_number(value)
        if n != n:  # NaN: keep the given value
            return value
        return int(n) if n.is_integer() else n
    return X.to_str(value)


def _eval_value(raw: object, env: Mapping[str, object], rename: Mapping[str, str]) -> object:
    if not isinstance(raw, str):
        return raw
    e = X.simplify(X.substitute(X.parse_template(raw), env, rename))
    ok, value = X.constant_value(e)
    return value if ok else e


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Scope:
    """Context of the workflow whose jobs are being lowered (top level or a callee)."""

    id_prefix: str = ""                 # "" or "<caller-instance-id>/"
    orig_prefix: str = ""               # "" or "<caller-orig-job>/"
    check_prefix: str = ""              # "" or "<caller check> / "
    subst: Mapping[str, object] = field(default_factory=dict)  # inputs / passed secrets
    sides: tuple[tuple[frozenset[str], Expr | None], ...] = ()  # enclosing callers' (needs, if)
    wf_env: tuple[tuple[str, str], ...] = ()  # callee workflow env (merged into job env)
    wf_defaults: object = None
    default_permissions: object = None
    fail_fast: str | None = None
    frozen: tuple[str, ...] = ()
    stack: tuple[str, ...] = ()         # workflow files being expanded (cycle detection)
    callee: bool = False


class _Lowerer:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        # instance id -> (top-level caller job key, caller job YAML) for every instance that a
        # top-level reusable-workflow call expands into (a frozen call is emitted verbatim)
        self.calls: dict[str, tuple[str, Mapping]] = {}
        # fail-fast groups of matrix jobs that call a reusable workflow (rule (iii))
        self.call_ff_groups: set[str] = set()

    # -- workflow ----------------------------------------------------------

    def lower_jobs(self, doc: Mapping, scope: _Scope) -> list[JobInstance]:
        jobs = doc.get("jobs")
        if not isinstance(jobs, Mapping) or not jobs:
            raise LowerError("workflow has no jobs")
        for k in sorted(jobs, key=str):
            if not isinstance(k, str) or not isinstance(jobs[k], Mapping):
                raise LowerError(f"job {k!r} is not a mapping")
        rename = {k: scope.id_prefix + k for k in sorted(jobs)} if scope.id_prefix else {}
        members: dict[str, list[str]] = {}
        out: list[JobInstance] = []
        for key in _topo_order(jobs):
            job = jobs[key]
            need_ids = frozenset(i for n in _needs_list(job, key, jobs) for i in members[n])
            insts = self.lower_job(key, job, need_ids, scope, rename)
            members[key] = [i.id for i in insts]
            out.extend(insts)
        return out

    # -- one job -----------------------------------------------------------

    def lower_job(
        self, key: str, job: Mapping, need_ids: frozenset[str], scope: _Scope, rename: Mapping[str, str]
    ) -> list[JobInstance]:
        orig_job = scope.orig_prefix + key
        group = scope.id_prefix + key
        reasons: set[str] = set(scope.frozen)
        subst = dict(scope.subst)

        known = KNOWN_CALL_JOB_KEYS if "uses" in job else KNOWN_NORMAL_JOB_KEYS
        for k in sorted(job, key=str):
            if k == "continue-on-error":
                continue  # decided per matrix combination below
            if k in FROZEN_JOB_KEYS:
                reasons.add(FROZEN_JOB_KEYS[k])
            elif k not in known:
                reasons.add(f"unsupported-key:{k}")
        reasons |= _defaults_reasons(job.get("defaults"))

        strategy = job.get("strategy")
        if isinstance(strategy, Mapping):
            reasons.update(f"unsupported-strategy-key:{k}" for k in sorted(map(str, strategy))
                           if k not in KNOWN_STRATEGY_KEYS)
        combos, key_order, dynamic = self._matrix(strategy, subst)
        has_matrix = isinstance(strategy, Mapping) and "matrix" in strategy or isinstance(strategy, str)
        if dynamic:
            reasons.add("dynamic-matrix")
        if isinstance(strategy, Mapping) and "max-parallel" in strategy:
            reasons.add("max-parallel")
        fail_fast = scope.fail_fast
        if fail_fast is None and has_matrix:
            ff = strategy.get("fail-fast", True) if isinstance(strategy, Mapping) else True
            ok, ff_value = _resolve_value(ff, subst)
            if not ok or X.truthy(ff_value):
                fail_fast = group
                if "uses" in job:
                    self.call_ff_groups.add(group)

        job_if: Expr | None = None
        if "if" in job:
            job_if = X.simplify(X.substitute(X.parse(job["if"]), subst, rename), boolean=True)
            if _mentions_needs_opaque(job_if):
                reasons.add("opaque-needs-condition")
        sides = scope.sides + ((need_ids, job_if),)

        results: list[JobInstance] = []
        seen: dict[str, int] = {}
        for combo in combos:
            env = dict(subst)
            if not dynamic:
                env["matrix"] = combo if combo is not None else {}
            mpairs = _matrix_pairs(combo)
            iid = instance_id(group, mpairs)
            seen[iid] = seen.get(iid, 0) + 1
            if seen[iid] > 1:
                iid = f"{iid}#{seen[iid]}"
            name = job.get("name")
            if isinstance(name, str) and "${{" in name:
                own_check = X.interpolate(name, env, rename)
            else:
                own_check = default_check_name(key if name is None else X.to_str(name), combo, key_order)
            check = scope.check_prefix + own_check
            inst_reasons = set(reasons)
            if "continue-on-error" in job and not self._falsy(job["continue-on-error"], env):
                inst_reasons.add(FROZEN_JOB_KEYS["continue-on-error"])
            if "uses" in job:
                expanded = self.lower_call(key, job, iid, orig_job, check, mpairs, env, sides, inst_reasons,
                                           fail_fast, scope, rename)
                if not scope.callee:
                    for inst in expanded:
                        self.calls[inst.id] = (key, job)
                results.extend(expanded)
            else:
                results.append(
                    self.lower_normal(key, job, iid, orig_job, check, mpairs, env, sides, inst_reasons, fail_fast, scope, rename)
                )
        return results

    def _falsy(self, value: object, env: Mapping[str, object]) -> bool:
        ok, v = _resolve_value(value, env)
        return ok and not X.truthy(v)

    def _matrix(self, strategy: object, env: Mapping[str, object]) -> tuple[list, list[str], bool]:
        if isinstance(strategy, str):
            return [None], [], True
        if not isinstance(strategy, Mapping) or "matrix" not in strategy:
            return [None], [], False
        ok, m = _resolve_value(strategy["matrix"], env)
        if not ok or not isinstance(m, Mapping):
            return [None], [], True
        combos, key_order = expand_matrix(m)
        return combos, key_order, False

    # -- normal job --------------------------------------------------------

    def lower_normal(
        self,
        key: str,
        job: Mapping,
        iid: str,
        orig_job: str,
        check: str,
        mpairs: tuple[tuple[str, str], ...],
        env: Mapping[str, object],
        sides: tuple[tuple[frozenset[str], Expr | None], ...],
        reasons: set[str],
        fail_fast: str | None,
        scope: _Scope,
        rename: Mapping[str, str],
    ) -> JobInstance:
        cond, ok = _combine_cond(sides)
        if not ok:
            reasons.add("reusable-status-condition")
        cond = X.simplify(cond, boolean=True)
        needs = frozenset().union(*(n for n, _ in sides))

        runs_on, ro_reasons = self._runs_on(job.get("runs-on"), env, rename)
        reasons |= ro_reasons

        steps_raw = job.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise LowerError(f"job {key!r} has no steps")
        run_defaults = _run_defaults(scope.wf_defaults, job.get("defaults"))
        steps: list[Step] = []
        for idx, s in enumerate(steps_raw):
            steps.extend(self._steps(key, iid, orig_job, idx, s, env, run_defaults, reasons, scope, rename))

        env_pairs = dict(scope.wf_env)
        job_env = job.get("env")
        if isinstance(job_env, Mapping):
            for k in sorted(job_env, key=str):
                env_pairs[str(k)] = X.interpolate(job_env[k], env, rename)
        elif job_env is not None:
            reasons.add("dynamic-env")

        if "permissions" in job:
            permissions = _norm_permissions(job["permissions"])
        else:
            permissions = scope.default_permissions

        outputs: list[tuple[str, Expr]] = []
        job_outputs = job.get("outputs")
        if isinstance(job_outputs, Mapping):
            for name in sorted(job_outputs, key=str):
                e = X.simplify(X.substitute(X.parse_template(job_outputs[name]), env, rename))
                if X.has_opaque(e):
                    reasons.add("opaque-output")
                outputs.append((str(name), e))
        elif job_outputs is not None:
            reasons.add("opaque-output")

        return JobInstance(
            id=iid,
            orig_job=orig_job,
            check_name=check,
            matrix=mpairs,
            runs_on=runs_on,
            cond=cond,
            needs=needs,
            steps=tuple(steps),
            env=tuple(sorted(env_pairs.items())),
            permissions=permissions,
            outputs=tuple(outputs),
            fail_fast_group=fail_fast,
            frozen=bool(reasons),
            frozen_reasons=tuple(sorted(reasons)),
            raw=_interp_obj(job, scope.subst) if scope.callee else job,
        )

    def _runs_on(
        self, spec: object, env: Mapping[str, object], rename: Mapping[str, str]
    ) -> tuple[str, set[str]]:
        reasons: set[str] = set()
        if spec is None:
            return "", {"missing-runs-on"}
        if isinstance(spec, Mapping):
            reasons.add("runner-group")
            spec = spec.get("labels", spec.get("group"))
            if spec is None:
                return "", reasons
        labels: list[str] = []
        for item in spec if isinstance(spec, list) else [spec]:
            ok, value = _resolve_value(item, env)
            if not ok:
                reasons.add("unresolved-runs-on")
                labels.append(X.interpolate(item, env, rename))
            elif isinstance(value, list):
                labels.extend(X.to_str(x) for x in value)
            else:
                labels.append(X.to_str(value))
        if any(label.strip().casefold() == "self-hosted" for label in labels):
            reasons.add("self-hosted-runner")
        return (labels[0] if len(labels) == 1 else ",".join(sorted(labels))), reasons

    def _steps(
        self,
        key: str,
        iid: str,
        orig_job: str,
        idx: int,
        s: object,
        env: Mapping[str, object],
        run_defaults: Mapping,
        reasons: set[str],
        scope: _Scope,
        rename: Mapping[str, str],
    ) -> list[Step]:
        """Lower one YAML step. A step with neither `run` nor `uses` (e.g. a `parallel:` group)
        freezes the job; the steps nested in a `parallel:` list are lowered in order (all with
        origin index `idx`) so that the frozen job still has a cost."""
        if isinstance(s, Mapping) and "run" not in s and "uses" not in s:
            unknown = sorted(str(k) for k in s if k not in KNOWN_STEP_KEYS)
            if not unknown:
                raise LowerError(f"job {key!r} step {idx} has neither run nor uses")
            reasons.update(f"unsupported-step-key:{k}" for k in unknown)
            nested = s.get("parallel")
            out: list[Step] = []
            if isinstance(nested, list):
                for child in nested:
                    out.extend(self._steps(key, iid, orig_job, idx, child, env, run_defaults, reasons, scope, rename))
            return out
        return [self._step(key, iid, orig_job, idx, s, env, run_defaults, reasons, scope, rename)]

    def _step(
        self,
        key: str,
        iid: str,
        orig_job: str,
        idx: int,
        s: object,
        env: Mapping[str, object],
        run_defaults: Mapping,
        reasons: set[str],
        scope: _Scope,
        rename: Mapping[str, str],
    ) -> Step:
        if not isinstance(s, Mapping):
            raise LowerError(f"job {key!r} step {idx} is not a mapping")
        for k in sorted(s, key=str):
            if k in FROZEN_STEP_KEYS:
                if k == "continue-on-error" and self._falsy(s[k], env):
                    continue
                reasons.add(FROZEN_STEP_KEYS[k])
            elif k not in KNOWN_STEP_KEYS:
                reasons.add(f"unsupported-step-key:{k}")

        def interp(v: object) -> str:
            return X.interpolate(v, env, rename)

        if "run" in s:
            kind, text = "run", interp(s["run"])
            wd = s.get("working-directory", run_defaults.get("working-directory"))
            sh = s.get("shell", run_defaults.get("shell"))
        elif "uses" in s:
            kind, text = "uses", X.to_str(s["uses"]).strip()
            wd = s.get("working-directory")
            sh = s.get("shell")
        else:
            raise LowerError(f"job {key!r} step {idx} has neither run nor uses")

        with_: tuple[tuple[str, str], ...] = ()
        if isinstance(s.get("with"), Mapping):
            with_ = tuple(sorted((str(k), interp(v)) for k, v in s["with"].items()))
        elif s.get("with") is not None:
            reasons.add("dynamic-with")
        step_env: tuple[tuple[str, str], ...] = ()
        if isinstance(s.get("env"), Mapping):
            step_env = tuple(sorted((str(k), interp(v)) for k, v in s["env"].items()))
        elif s.get("env") is not None:
            reasons.add("dynamic-env")

        binding = Binding(
            kind=kind,
            text=text,
            with_=with_,
            env=step_env,
            working_directory=interp(wd) if wd is not None else None,
            shell=interp(sh) if sh is not None else None,
            name=interp(s["name"]) if s.get("name") is not None else None,
            step_id=X.to_str(s["id"]) if s.get("id") is not None else None,
        )
        if "if" in s:
            e = X.substitute(X.parse(s["if"]), env, rename)
            cond = e if X.uses_status_function(e) else X.conj([X.SUCCESS, e])
        else:
            cond = X.SUCCESS
        return Step(
            binding=binding,
            cond=X.simplify(cond, boolean=True),
            contract=TOP_CONTRACT,
            origin=(orig_job, idx),
            raw=_interp_obj(s, scope.subst) if scope.callee else s,
            origin_instance=iid,
        )

    # -- reusable-workflow call ---------------------------------------------

    def lower_call(
        self,
        key: str,
        job: Mapping,
        iid: str,
        orig_job: str,
        check: str,
        mpairs: tuple[tuple[str, str], ...],
        env: Mapping[str, object],
        sides: tuple[tuple[frozenset[str], Expr | None], ...],
        reasons: set[str],
        fail_fast: str | None,
        scope: _Scope,
        rename: Mapping[str, str],
    ) -> list[JobInstance]:
        uses = X.to_str(job["uses"]).strip()

        def frozen_call(reason: str) -> list[JobInstance]:
            reasons.add(reason)
            cond, _ = _combine_cond(sides)
            return [
                JobInstance(
                    id=iid,
                    orig_job=orig_job,
                    check_name=check,
                    matrix=mpairs,
                    runs_on="",
                    cond=X.simplify(cond, boolean=True),
                    needs=frozenset().union(*(n for n, _ in sides)),
                    steps=(),
                    permissions=_norm_permissions(job["permissions"]) if "permissions" in job else scope.default_permissions,
                    fail_fast_group=fail_fast,
                    frozen=True,
                    frozen_reasons=tuple(sorted(reasons)),
                    raw=_interp_obj(job, scope.subst) if scope.callee else job,
                )
            ]

        if not uses.startswith("./"):
            return frozen_call("remote-reusable-workflow")
        rel = posixpath.normpath(uses[2:].split("@", 1)[0])
        if rel in scope.stack:
            return frozen_call("reusable-cycle")
        if len(scope.stack) > MAX_REUSABLE_DEPTH:
            return frozen_call("reusable-depth")
        path = self.repo_root / rel
        if not path.is_file():
            return frozen_call("missing-local-workflow")
        callee = load_yaml(path.read_text(encoding="utf-8"))
        if not isinstance(callee, Mapping) or not isinstance(callee.get("jobs"), Mapping):
            return frozen_call("invalid-reusable-workflow")
        on = _normalize_on(callee.get("on"))
        if "workflow_call" not in on:
            return frozen_call("invalid-reusable-workflow")
        call_spec = on["workflow_call"] if isinstance(on["workflow_call"], Mapping) else {}

        given = job.get("with") if isinstance(job.get("with"), Mapping) else {}
        inputs = self._inputs(call_spec.get("inputs"), given, env, rename)
        subst: dict[str, object] = {"inputs": inputs}
        secrets = job.get("secrets")
        if isinstance(secrets, Mapping):
            for k in sorted(secrets, key=str):
                subst[f"secrets.{k}"] = _eval_value(secrets[k], env, rename)

        wf_env: tuple[tuple[str, str], ...] = ()
        if isinstance(callee.get("env"), Mapping):
            wf_env = tuple(sorted((str(k), X.interpolate(v, subst)) for k, v in callee["env"].items()))
        if "permissions" in callee:
            default_perm = _norm_permissions(callee["permissions"])
        elif "permissions" in job:
            default_perm = _norm_permissions(job["permissions"])
        else:
            default_perm = scope.default_permissions

        child = _Scope(
            id_prefix=iid + "/",
            orig_prefix=orig_job + "/",
            check_prefix=check + " / ",
            subst=subst,
            sides=sides,
            wf_env=wf_env,
            wf_defaults=_interp_obj(callee.get("defaults"), subst),
            default_permissions=default_perm,
            fail_fast=fail_fast,
            frozen=tuple(sorted(reasons | set(_workflow_reasons(callee, callee=True)))),
            stack=scope.stack + (rel,),
            callee=True,
        )
        return self.lower_jobs(callee, child)

    def _inputs(
        self, spec: object, given: Mapping, env: Mapping[str, object], rename: Mapping[str, str]
    ) -> dict[str, object]:
        if not isinstance(spec, Mapping):
            return {}
        given_l = {str(k).lower(): given[k] for k in sorted(given, key=str)}
        out: dict[str, object] = {}
        for name in sorted(spec, key=str):
            decl = spec[name] if isinstance(spec[name], Mapping) else {}
            typ = X.to_str(decl.get("type", "string"))
            if str(name).lower() in given_l:
                value = _eval_value(given_l[str(name).lower()], env, rename)
            elif "default" in decl:
                value = _eval_value(decl["default"], {}, {})
            else:
                value = {"boolean": False, "number": 0}.get(typ, "")
            out[str(name)] = _coerce_input(value, typ)
        return out


# ---------------------------------------------------------------------------
# Post-lowering rules (RESEARCH_SCOPE, "Verification gaps and resolutions")
# ---------------------------------------------------------------------------

class _StatusUnknown(Exception):
    pass


def may_run_after_cancel(cond: Expr) -> bool:
    """Can a step with this explicit condition run after its job was cancelled? Status
    functions are evaluated as after a cancellation (always() and cancelled() true,
    success() and failure() false); opaque text that mentions a status function may."""

    def sub(e: Expr) -> Expr:
        if e.op == "status":
            return Expr("lit", (str(e.args[0]).lower() in ("always", "cancelled"),))
        if e.op == "opaque":
            if X.uses_status_function(e):
                raise _StatusUnknown
            return e
        if e.op in ("lit", "ctx", "step", "needs"):
            return e
        if e.op == "call":
            return Expr("call", (e.args[0], *(sub(a) for a in e.args[1:])))
        return Expr(e.op, tuple(sub(a) if isinstance(a, Expr) else a for a in e.args))

    try:
        e2 = X.simplify(sub(cond), boolean=True)
    except _StatusUnknown:
        return True
    return not (e2.op == "lit" and not X.truthy(e2.args[0]))


def _only_default_status(cond: Expr) -> bool:
    """No status function except top-level `success()` conjuncts (the default condition)."""
    parts: list[Expr] = []
    stack = [cond]
    while stack:
        e = stack.pop()
        if e.op == "and":
            stack.extend(e.args)
        else:
            parts.append(e)
    return not any(X.uses_status_function(e) for e in parts if e != X.SUCCESS)


def _job_needs_refs(job: JobInstance) -> frozenset[tuple[str, str]]:
    """Every `needs.<key>.<field>` reference of a job: conditions, outputs, env, step texts."""
    refs: set[tuple[str, str]] = set()
    for e in [job.cond, *(e for _, e in job.outputs), *(s.cond for s in job.steps)]:
        refs |= X.needs_field_refs(e)
    texts = [v for _, v in job.env]
    for s in job.steps:
        b = s.binding
        texts += [b.text, *(v for _, v in b.with_), *(v for _, v in b.env),
                  b.working_directory or "", b.name or ""]
    for t in texts:
        refs |= X.template_needs_refs(t)
    return frozenset(refs)


def _ref_members(job: JobInstance, key: str) -> list[str]:
    """Needed instances a `needs.<key>` reference of `job` denotes (`*`: every need)."""
    if key == "*":
        return sorted(job.needs)
    return sorted(n for n in job.needs if n == key or n.startswith(key + "[") or n.startswith(key + "/"))


def _scope_prefix(jid: str) -> str:
    """`<caller-instance>/` for a job of an expanded reusable workflow, else "" (the last `/`
    outside matrix brackets)."""
    depth, cut = 0, -1
    for i, ch in enumerate(jid):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "/" and depth == 0:
            cut = i
    return jid[: cut + 1]


def _is_multi(job: JobInstance, key: str, members: list[str]) -> bool:
    """Does `needs.<key>` of `job` denote several instances of one needed job? For `needs.*`,
    the needed instances are grouped by the job key they belong to (in the reader's scope)."""
    if key != "*":
        return len(members) > 1
    prefix = _scope_prefix(job.id)
    groups: dict[str, int] = {}
    for n in members:
        rest = n[len(prefix):] if n.startswith(prefix) else n
        g = rest.split("[", 1)[0].split("/", 1)[0]
        groups[g] = groups.get(g, 0) + 1
    return any(c > 1 for c in groups.values())


def _freeze(job: JobInstance, reason: str) -> JobInstance:
    if reason in job.frozen_reasons:
        return job
    return replace(job, frozen=True, frozen_reasons=tuple(sorted(set(job.frozen_reasons) | {reason})))


def _dependents(p: Program) -> dict[str, set[str]]:
    """Transitive dependents of every instance."""
    direct: dict[str, set[str]] = {j.id: set() for j in p.jobs}
    for j in p.jobs:
        for n in j.needs:
            if n in direct:
                direct[n].add(j.id)
    out: dict[str, set[str]] = {}
    for jid in sorted(direct):
        seen: set[str] = set()
        stack = sorted(direct[jid])
        while stack:
            x = stack.pop()
            if x not in seen:
                seen.add(x)
                stack.extend(sorted(direct[x]))
        out[jid] = seen
    return out


def _fail_fast_violation(p: Program, members: list[JobInstance], call_group: bool,
                         dependents: Mapping[str, set[str]], repo_root: Path) -> str | None:
    """Why a fail-fast matrix is not transformable (None if it is): (iii) it calls a
    reusable workflow; (ii) a downstream job has a non-default status condition or reads
    `needs.<M>` of the matrix; (i) a member step that can run after a cancellation writes
    persistent state (`ext:*` or an artifact, per its guide contract)."""
    if call_group:
        return "calls a reusable workflow"
    ids = {m.id for m in members}
    by_id = {j.id: j for j in p.jobs}
    down = sorted(set().union(*(dependents.get(m, set()) for m in ids)) - ids)
    for d in down:
        job = by_id[d]
        if not _only_default_status(job.cond):
            return f"downstream job {d} has a status condition"
        for key, _field in sorted(_job_needs_refs(job)):
            if set(_ref_members(job, key)) & ids:
                return f"downstream job {d} reads needs.{key}"
    from ghav import contracts  # lazy: contracts do not depend on lowering

    for m in sorted(members, key=lambda j: j.id):
        for k, s in enumerate(m.steps):
            if not may_run_after_cancel(s.cond):
                continue
            c = contracts.infer(s, m, repo_root, workflow_env=p.env, workflow_defaults=p.defaults)
            if c.top or any(r.kind in ("ext", "run") for r in c.writes):
                return f"step {k} of {m.id} can run after a cancellation and writes persistent state"
    return None


def _multi_output_readers(p: Program) -> list[tuple[str, str, tuple[str, ...]]]:
    """(reader id, output name or `*`, member ids) of every `needs.<M>.outputs` reference of a
    non-frozen job where M denotes several needed instances."""
    out: list[tuple[str, str, tuple[str, ...]]] = []
    for job in sorted(p.jobs, key=lambda j: j.id):
        if job.frozen:
            continue
        for key, fld in sorted(_job_needs_refs(job)):
            members = _ref_members(job, key)
            if not _is_multi(job, key, members):
                continue
            if fld.startswith("outputs.") and key != "*":
                out.append((job.id, fld[len("outputs."):], tuple(members)))
            elif fld in ("outputs", "*"):
                out.append((job.id, "*", tuple(members)))
    return out


def _apply_core_rules(p: Program, lowerer: "_Lowerer", repo_root: Path) -> Program:
    """Freeze what the checker cannot transform (RESEARCH_SCOPE, verification gaps):

    1. a job reading `needs.<M>.result` of a multi-instance M (`multi-instance-result`);
    2. every member of a fail-fast matrix that violates `_fail_fast_violation` (`fail-fast`);
    3. a job reading `needs.<M>.outputs.<o>` of a multi-instance M unless every instance of M
       produces the same term for `o` (symbolic evaluation with guide contracts;
       `multi-instance-output`);
    4. every job of a top-level reusable-workflow call of which some job is frozen
       (`frozen-call`); such a call is kept verbatim: `raw` of each of its jobs is the caller
       job's YAML, and emission writes the call back unchanged.
    Rules 3 and 4 are repeated until nothing changes (freezing changes evaluation).
    """
    jobs = {j.id: j for j in p.jobs}
    call_of = {iid: key for iid, (key, _raw) in lowerer.calls.items()}
    for jid in sorted(jobs):
        job = jobs[jid]
        for key, fld in sorted(_job_needs_refs(job)):
            members = _ref_members(job, key)
            if (fld in ("result", "*") or key == "*") and _is_multi(job, key, members):
                jobs[jid] = _freeze(jobs[jid], MULTI_RESULT_REASON)
            # `needs.<call>` names a reusable-workflow call as a whole; the reference stays
            # meaningful only if the call is kept (not inlined), so the call is frozen.
            for m in members:
                if m in call_of and call_of[m] != call_of.get(jid) and \
                        (key == "*" or key == call_of[m]) and m != key:
                    jobs[m] = _freeze(jobs[m], FROZEN_CALL_REASON)
    p = p.with_jobs(jobs.values())

    groups: dict[str, list[JobInstance]] = {}
    for j in sorted(p.jobs, key=lambda x: x.id):
        if j.fail_fast_group:
            groups.setdefault(j.fail_fast_group, []).append(j)
    dependents = _dependents(p)
    for g, members in sorted(groups.items()):
        if len(members) < 2 and g not in lowerer.call_ff_groups:
            continue
        if _fail_fast_violation(p, members, g in lowerer.call_ff_groups, dependents, repo_root):
            for m in members:
                jobs[m.id] = _freeze(jobs[m.id], FAIL_FAST_REASON)
    p = p.with_jobs(jobs.values())

    for _ in range(len(jobs) + 1):
        changed = False
        readers = _multi_output_readers(p)
        if readers:
            from ghav import contracts, semantics  # lazy: only for matrix-output readers

            annotated = contracts.annotate(p, repo_root, observed={})
            obs = semantics.evaluate(annotated, semantics.SymbolicContext.for_programs(annotated))
            for rid, name, members in readers:
                terms = [obs.outputs.get((m, name)) for m in members] if name != "*" else [None]
                if any(t is None for t in terms) or not all(t.eq(terms[0]) for t in terms):
                    jobs[rid] = _freeze(jobs[rid], MULTI_OUTPUT_REASON)
                    changed = True
        calls: dict[str, list[str]] = {}
        for iid, (key, _raw) in sorted(lowerer.calls.items()):
            if iid in jobs:
                calls.setdefault(key, []).append(iid)
        for key, iids in sorted(calls.items()):
            if not any(jobs[i].frozen for i in iids):
                continue
            raw = lowerer.calls[iids[0]][1]
            for i in iids:
                job = jobs[i] if jobs[i].frozen else _freeze(jobs[i], FROZEN_CALL_REASON)
                if job.raw is not raw or not jobs[i].frozen:
                    changed = changed or not jobs[i].frozen
                    jobs[i] = replace(job, raw=raw)
        p = p.with_jobs(jobs.values())
        if not changed:
            break
    return p


def lower(workflow_path: Path, repo_root: Path) -> Program:
    """Lower one workflow file of the repository checked out at `repo_root`."""
    repo_root = Path(repo_root)
    path = Path(workflow_path)
    if not path.is_absolute() and (repo_root / path).exists():
        path = repo_root / path
    try:
        source = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        source = path.as_posix()
    doc = load_yaml(path.read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping):
        raise LowerError("workflow is not a mapping")
    if "on" not in doc:
        raise LowerError("workflow has no 'on'")
    wf_env = doc.get("env")
    env_pairs = (
        tuple(sorted((str(k), X.to_str(v)) for k, v in wf_env.items())) if isinstance(wf_env, Mapping) else ()
    )
    scope = _Scope(wf_defaults=doc.get("defaults"), stack=(source,),
                   frozen=_workflow_reasons(doc, callee=False))
    lowerer = _Lowerer(repo_root)
    jobs = lowerer.lower_jobs(doc, scope)
    ids = [j.id for j in jobs]
    if len(ids) != len(set(ids)):
        raise LowerError("duplicate job instance ids")
    program = Program(
        name=X.to_str(doc["name"]) if doc.get("name") is not None else source,
        on=_normalize_on(doc["on"]),
        jobs=tuple(sorted(jobs, key=lambda j: j.id)),
        env=env_pairs,
        permissions=doc.get("permissions"),
        concurrency=doc.get("concurrency"),
        defaults=doc.get("defaults"),
        source=source,
    )
    return _apply_core_rules(program, lowerer, repo_root)
