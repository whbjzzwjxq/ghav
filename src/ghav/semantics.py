"""Symbolic semantics of WIR programs (z3) and well-formedness checks.

A program denotes, for every execution context, the final values of its persistent
regions and the verdict of every check. This module computes that denotation as z3
terms in one fixed topological order of the job instances; the order does not matter
for well-formed (race-free) programs because their instances commute.

Encoding (see docs/ARCHITECTURE.md, "semantics.py"):

* One uninterpreted sort ``Val``. Concrete strings (matrix values, platform labels,
  literals in conditions, env values) are literal constants that are pairwise distinct.
  ``github.*``, ``secrets.*``, ``vars.*`` and ``inputs.*`` context keys are symbolic
  constants shared by every instance; ``github.event_name`` is constrained to the
  supported events.
* A task is abstracted by its reads: for every (binding key, written region) there is an
  uninterpreted function ``F`` over the values of the task's reads (sorted by region, with
  the owning instance erased from job-local and context regions, so the same step computes
  the same function in every job), and ``ok`` is an uninterpreted Bool function of the same
  arguments. A step with guard ``g`` writes ``If(g, F(args), old)``; its status is
  ``If(g, If(ok, success, failure), skipped)``.
* Job-local writes are updates. A task changes part of a job-local region (some paths, some
  PATH entries) and the rest survives, so when it writes a kind (src, dep, out, tool) that it
  does not read, the region becomes ``U_kind(old, F(args))`` with ``U_kind`` uninterpreted
  (a task that in fact replaces the whole region is one interpretation of ``U``). A task that
  reads the kind it writes gets plain ``F(args)``: ``old`` is already among its arguments.
  Without this, a contract that omits a read would make earlier writers of the region dead
  (e.g. ``pnpm/action-setup`` before ``actions/setup-node``, both writing ``dep``).
* ``Status`` is an enumeration {success, failure, skipped, cancelled}. A check passes iff
  its job's status is success or skipped. Cancellation is not produced by the semantics
  (fail-fast is handled by the checker's provisional policy).
* Official actions have interpreted semantics. Caches follow cold-cache semantics: a cache
  step has no effect, always succeeds (unless ``fail-on-cache-miss``), and its outputs are
  empty; a restore that breaks cache transparency has the ``cache-opaque`` contract
  (contracts guide 7) and is an ordinary uninterpreted task. The transfer idiom
  (``contracts.parse_transfer``: tar the declared ``out`` paths, upload, download, untar) is
  an exact copy of ``out`` when ``transfer_plan`` shows that the packed paths cover the
  producer's whole ``out`` region and the consumer's ``out`` is still empty; any other
  artifact (plain upload/download) is uninterpreted. A download fails iff the artifact does
  not exist; an upload fails iff an artifact of that name already exists (unless
  ``overwrite``), or, for a plain upload, possibly under ``if-no-files-found: error``
  (exact transfers are assumed to succeed). A clean checkout empties ``out``.
* External-service writes are keyed by the originating step occurrence
  (``ext:<service>@<instance>#<index>``, ``ext_write_key``); the catch-all ``ext:*`` is not.
* A ⊤ step reads all of its instance's job-local and context regions (including
  ``job-identity``), all non-transfer artifacts, all external objects and its needs'
  outputs, and writes its job-local regions and every external object including the
  catch-all ``ext:*``. A frozen job is one opaque task whose effect is the union of its
  steps' effects (``frozen_effect``); a job frozen for a construct whose effects the steps do
  not describe, or without steps, is a blanket ⊤ task that may also create every artifact
  some task reads and set any output.
* ``needs.<M>.outputs.<o>`` of a multi-instance M is that term when every instance produces
  the same term (otherwise lowering froze the reader); the outputs of a reusable-workflow
  call are uninterpreted in everything the call's jobs produce.
* Guards are kept in a canonical form (flattened And/Or with operands sorted by AST id) and a
  task's arguments are simplified under its own guard (an ``If`` whose condition the guard
  decides is resolved). Both are exact, and they make equal computations syntactically equal,
  which keeps solver queries small.
* Every ``SymbolicContext`` owns a fresh z3 context, so results never depend on earlier
  solver work in the same process.
* Rename invariance: instance ids are internal names and differ between an original and a
  re-lowered or developer-written program. Everything that enters a term is therefore named
  by ``instance_identity`` (check name and matrix) instead: the value of ``job-identity``
  and other per-instance context keys, the argument slots of job outputs and job statuses,
  ``needs.<job>`` references inside step texts (``task_key``), and the tie-break of the
  evaluation order.

Persistent regions ``P`` are ``ext:*`` (every external object of either program, including
``ext:check:<name>`` objects written by steps) and ``run:<name>`` for artifacts that the
ORIGINAL program uploads, excluding ``ghav-`` transfer artifacts. Cross-run caches are not
persistent. Cache-transparency findings (``cache_issues``) are reported, never blocking.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import posixpath
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import z3

from ghav.contracts import TRANSFER_PREFIX, parse_transfer
from ghav.wir import (
    CTX,
    DEP,
    EVENTS,
    EXT,
    JOB_LOCAL_KINDS,
    OUT,
    OUTPUT,
    RUN,
    SRC,
    ST_JOB,
    ST_STEP,
    TOOL,
    Binding,
    Effect,
    Expr,
    JobInstance,
    Program,
    Region,
    Step,
)

z3.set_param("smt.random_seed", 0)
z3.set_param("sat.random_seed", 0)

# ---------------------------------------------------------------------------
# Constants. Sorts live in a fresh z3 context per SymbolicContext, so results do not
# depend on what else the process has solved before (see SymbolicContext).
# ---------------------------------------------------------------------------

STATUS_NAMES = ("success", "failure", "skipped", "cancelled")
SUCCESS, FAILURE, SKIPPED, CANCELLED = STATUS_NAMES

TOP_EXT_KEY = "*"                   # catch-all external object written by ⊤ tasks
LOCAL_STATE_KINDS = (SRC, DEP, OUT, TOOL)
SHARED_KINDS = frozenset({RUN, EXT, OUTPUT})
STRUCTURAL_ISSUES = frozenset({"duplicate-id", "unknown-need", "cycle"})
RACE_ISSUES = frozenset({"race", "duplicate-artifact"})

GLOBAL_CTX_PREFIXES = ("github.", "secrets.", "vars.", "inputs.")
INSTANCE_CTX_KEYS = frozenset({"github.job"})  # github.* keys that identify the job instance

UPLOAD_TEMPLATE = "upload-artifact"
DOWNLOAD_TEMPLATE = "download-artifact"
CACHE_TEMPLATE = "cache"
CHECKOUT_TEMPLATE = "checkout"
OPAQUE_CACHE_TEMPLATE = "cache-opaque"

# Frozen-job reasons after which the job's step contracts still describe its effects (the
# construct changes how the steps run, not what they read and write). Any other reason, or a
# frozen job without steps, makes the job a blanket ⊤ task (see ``frozen_effect``).
TRANSPARENT_FROZEN_REASONS = frozenset({
    "concurrency", "container", "continue-on-error", "services", "max-parallel",
    "self-hosted-runner", "runner-group", "unresolved-runs-on", "missing-runs-on",
    "step-continue-on-error", "dynamic-matrix", "dynamic-env", "dynamic-with", "opaque-output",
    "opaque-needs-condition", "reusable-status-condition", "fail-fast", "multi-instance-result",
    "multi-instance-output", "frozen-call",
})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def region_key(r: Region) -> tuple[str, str, str]:
    """Total, deterministic sort key for regions (Region's own order fails on None)."""
    return (r.kind, r.job or "", r.key or "")


def slot_key(r: Region) -> tuple[str, str, str]:
    """Argument slot of a read or written region, independent of the executing instance."""
    if r.kind in JOB_LOCAL_KINDS or r.kind == CTX:
        return (r.kind, "", r.key or "")
    return (r.kind, r.job or "", r.key or "")


def is_transfer_artifact(name: str | None) -> bool:
    return bool(name) and str(name).startswith(TRANSFER_PREFIX)


def is_global_ctx(key: str) -> bool:
    """Context keys whose value is the same for every job instance of a run."""
    if key in INSTANCE_CTX_KEYS or key.startswith("github.action"):
        return False
    return key.startswith(GLOBAL_CTX_PREFIXES)


def merged_env(p: Program, job: JobInstance) -> dict[str, str]:
    env = dict(p.env)
    env.update(dict(job.env))
    return env


def binding_label(b: Binding) -> str:
    if b.name:
        return b.name
    if b.step_id:
        return b.step_id
    first = (b.text or "").strip().splitlines()[0] if (b.text or "").strip() else b.kind
    return first if len(first) <= 60 else first[:57] + "..."


def _digest(obj: object) -> str:
    return hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()[:20]


def _with(step: Step) -> dict[str, str]:
    return {k: v for k, v in step.binding.with_}


def _truthy_text(text: str | None) -> bool:
    return (text or "").strip().lower() in ("true", "1", "yes")


_OP_TRUE, _OP_FALSE = z3.Z3_OP_TRUE, z3.Z3_OP_FALSE
_OP_AND, _OP_OR, _OP_NOT, _OP_ITE = z3.Z3_OP_AND, z3.Z3_OP_OR, z3.Z3_OP_NOT, z3.Z3_OP_ITE


def _op(x: z3.ExprRef) -> int:
    """Declaration kind of an application term (all terms built here are applications)."""
    c = x.ctx_ref()
    return z3.Z3_get_decl_kind(c, z3.Z3_get_app_decl(c, x.as_ast()))


def _kids(x: z3.ExprRef) -> tuple[tuple[z3.ExprRef, ...], frozenset[int]]:
    """Children of an application term and their AST ids, memoized on its z3 context (the
    memo keeps the term alive, so its id cannot be reused; each SymbolicContext owns a fresh
    context, so the memo never outlives one comparison)."""
    zc = x.ctx
    memo = getattr(zc, "_ghav_kids", None)
    if memo is None:
        memo = {}
        zc._ghav_kids = memo  # type: ignore[attr-defined]
    i = x.get_id()
    hit = memo.get(i)
    if hit is None:
        kids = tuple(x.children())
        hit = (x, kids, frozenset(k.get_id() for k in kids))
        memo[i] = hit
    return hit[1], hit[2]


def _junction(zc: z3.Context, xs: Sequence[z3.BoolRef], conj: bool) -> z3.BoolRef:
    """Flattened, deduplicated And/Or with its operands sorted by AST id.

    Equal operand sets give the identical AST (ids are deterministic in the fresh context
    of a SymbolicContext), so guards built along different paths coincide syntactically.
    """
    unit, zero, same = (_OP_TRUE, _OP_FALSE, _OP_AND) if conj else (_OP_FALSE, _OP_TRUE, _OP_OR)
    items: dict[int, z3.BoolRef] = {}
    negated: list[int] = []
    stack = list(reversed(xs))
    while stack:
        x = stack.pop()
        k = _op(x)
        if k == zero:
            return z3.BoolVal(not conj, zc)
        if k == unit:
            continue
        if k == same:
            stack.extend(reversed(_kids(x)[0]))
            continue
        items[x.get_id()] = x
        if k == _OP_NOT:
            negated.append(x.arg(0).get_id())
    if any(i in items for i in negated):
        return z3.BoolVal(not conj, zc)
    if not items:
        return z3.BoolVal(conj, zc)
    if len(items) == 1:
        return next(iter(items.values()))
    ordered = [items[i] for i in sorted(items)]
    return z3.And(*ordered) if conj else z3.Or(*ordered)


def _and(zc: z3.Context, *xs: z3.BoolRef) -> z3.BoolRef:
    """Conjunction with folding (``zc`` gives the context of an empty result)."""
    return _junction(zc, xs, True)


def _or(zc: z3.Context, *xs: z3.BoolRef) -> z3.BoolRef:
    return _junction(zc, xs, False)


def _not(x: z3.BoolRef) -> z3.BoolRef:
    k = _op(x)
    if k == _OP_TRUE:
        return z3.BoolVal(False, x.ctx)
    if k == _OP_FALSE:
        return z3.BoolVal(True, x.ctx)
    if k == _OP_NOT:
        return x.arg(0)
    return z3.Not(x)


def _conjuncts(x: z3.BoolRef) -> tuple[z3.BoolRef, ...]:
    k = _op(x)
    if k == _OP_TRUE:
        return ()
    if k == _OP_AND:
        return _kids(x)[0]
    return (x,)


def _conjunct_ids(x: z3.BoolRef) -> frozenset[int]:
    k = _op(x)
    if k == _OP_TRUE:
        return frozenset()
    if k == _OP_AND:
        return _kids(x)[1]
    return frozenset({x.get_id()})


def _implies(a: z3.BoolRef, b: z3.BoolRef) -> bool:
    """Syntactic check: every conjunct of ``b`` is a conjunct of ``a``."""
    if _op(a) == _OP_FALSE:
        return True
    return _conjunct_ids(b) <= _conjunct_ids(a)


class _Assuming:
    """Conjuncts of a guard, for simplifying values that only matter when it holds."""

    def __init__(self, guard: z3.BoolRef) -> None:
        cs = _conjuncts(guard)
        self.pos = _conjunct_ids(guard)
        self.neg = {c.arg(0).get_id() for c in cs if _op(c) == _OP_NOT}

    def _decides(self, c: z3.BoolRef) -> bool | None:
        ids = _conjunct_ids(c)
        if ids <= self.pos:
            return True
        if ids & self.neg:
            return False
        for k in _conjuncts(c):
            if _op(k) == _OP_NOT and k.arg(0).get_id() in self.pos:
                return False
        return None

    def value(self, v: z3.ExprRef) -> z3.ExprRef:
        """``v`` with top-level If-terms resolved when the guard decides their condition."""
        while _op(v) == _OP_ITE:
            c, a, b = v.children()
            d = self._decides(c)
            if d is None:
                break
            v = a if d else b
        return v


def _ite(c: z3.BoolRef, a: z3.ExprRef, b: z3.ExprRef) -> z3.ExprRef:
    kc = _op(c)
    if kc == _OP_TRUE:
        return a
    if kc == _OP_FALSE:
        return b
    if a.eq(b):
        return a
    if z3.is_bool(a):
        zc = c.ctx
        ka, kb = _op(a), _op(b)
        if kb == _OP_FALSE:
            return _and(zc, c, a)
        if ka == _OP_FALSE:
            return _and(zc, _not(c), b)
        if ka == _OP_TRUE:
            return _or(zc, c, b)
        if kb == _OP_TRUE:
            return _or(zc, _not(c), a)
    return z3.If(c, a, b)


def _is_status_const(st: z3.ExprRef) -> bool:
    return _op(st) == z3.Z3_OP_DT_CONSTRUCTOR and st.num_args() == 0


def status_is(st: z3.ExprRef, name: str) -> z3.BoolRef:
    """``st == <name>`` for a Status term, folded through If-terms over status constants."""
    if _is_status_const(st):
        return z3.BoolVal(st.decl().name() == name, st.ctx)
    if _op(st) == _OP_ITE:
        c, a, b = st.children()
        return _ite(c, status_is(a, name), status_is(b, name))
    return st == st.sort().constructor(STATUS_NAMES.index(name))()


def passes(st: z3.ExprRef) -> z3.BoolRef:
    """Check verdict of a job status: success or skipped (cancelled counts as fail)."""
    return _or(st.ctx, status_is(st, SUCCESS), status_is(st, SKIPPED))


# ---------------------------------------------------------------------------
# Issues and program structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Issue:
    """A well-formedness problem.

    kind: ``race`` | ``duplicate-artifact`` | ``unknown-need`` | ``cycle`` | ``duplicate-id``.
    ``origins`` are the ``Step.origin`` values of the conflicting accesses (job outputs and
    frozen jobs use ``unit_origin``); they identify a race independently of instance ids.
    """

    kind: str
    message: str
    region: str | None = None
    jobs: tuple[str, ...] = ()
    origins: tuple[tuple[str, int], ...] = ()

    def sort_key(self) -> tuple:
        return (self.kind, self.region or "", self.jobs, self.origins, self.message)

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "message": self.message,
            "region": self.region,
            "jobs": list(self.jobs),
            "origins": [list(o) for o in self.origins],
        }


def _needs_graph(p: Program) -> dict[str, tuple[str, ...]]:
    ids = {j.id for j in p.jobs}
    return {j.id: tuple(sorted(n for n in j.needs if n in ids)) for j in p.jobs}


def instance_identity(job: JobInstance) -> str:
    """Identity of a job instance that is stable across programs.

    It is the instance's check name plus its matrix assignment: what GitHub shows for the
    instance, and what emission preserves when it renames jobs (for example when it
    flattens a local reusable workflow). Instance ids are internal names; they may differ
    between an original and a re-lowered or developer-written candidate. The value of
    ``ctx:job-identity`` and of other per-instance context keys is derived from it, so a
    ⊤ task keeps its behavior under a rename that keeps the check name, and changes it
    when it moves to an instance with another check name. Assumption: tasks do not branch
    on the YAML job key (``GITHUB_JOB``) beyond what the check name determines.
    """
    return json.dumps([job.check_name, [list(kv) for kv in sorted(job.matrix)]], sort_keys=True)


def unit_origin(job: JobInstance) -> tuple[str, int]:
    """Origin of a job-level access (a frozen job as one task, or a job output).

    It is the original job of the instance's first step (so it follows recovered step
    origins, see ``checker.recover_origins``), or ``orig_job`` for a job without steps.
    """
    job_key = str(job.steps[0].origin[0]) if job.steps else job.orig_job
    return (job_key, -1)


_TEMPLATE_RE = re.compile(r"\$\{\{(.*?)\}\}", re.S)
_NEEDS_REF_RE = re.compile(r"""needs\s*(?:\.\s*([A-Za-z_][A-Za-z0-9_-]*)|\[\s*(['"])(.*?)\2\s*\])""")


def needs_targets(job: JobInstance, target: str, by_id: Mapping[str, JobInstance]) -> list[str]:
    """Instances a ``needs.<target>`` reference of ``job`` denotes: the needed instance with
    that id; else the needed instances of that matrix or reusable-workflow call
    (``target[...]``, ``target/...``, as ``lower.needs_members``); else every needed instance
    whose original job is ``target`` or a job of the call ``target``."""
    needs = sorted(n for n in job.needs if n in by_id)
    if target in needs:
        return [target]
    members = [n for n in needs if n.startswith(target + "[") or n.startswith(target + "/")]
    if members:
        return members
    return [n for n in needs if by_id[n].orig_job == target or by_id[n].orig_job.startswith(target + "/")]


def is_call_target(target: str, instances: Sequence[str], by_id: Mapping[str, JobInstance]) -> bool:
    """Does ``needs.<target>`` denote the jobs of a reusable-workflow call?"""
    return any(n.startswith(target + "/") or by_id[n].orig_job.startswith(target + "/")
               for n in instances if n in by_id)


def task_key(step: Step, job: JobInstance, by_id: Mapping[str, JobInstance]) -> tuple:
    """``Binding.key()`` with every ``needs.<job>`` reference inside ``${{ }}`` in the
    step's texts replaced by the identities of the instances it denotes.

    Emission renames jobs (for example when it flattens a local reusable workflow) and
    rewrites such references accordingly; the task is the same, so its key must be too.
    References that do not resolve are kept verbatim.
    """
    b = step.binding
    texts = [b.text, *(v for _, v in b.with_), *(v for _, v in b.env)]
    if not any("needs" in t for t in texts):
        return b.key()

    def ref(m: re.Match) -> str:
        target = m.group(1) if m.group(1) is not None else m.group(3)
        inst = needs_targets(job, target, by_id)
        if not inst:
            return m.group(0)
        return "needs[" + json.dumps(sorted(instance_identity(by_id[n]) for n in inst)) + "]"

    def norm(text: str) -> str:
        return _TEMPLATE_RE.sub(lambda m: "${{" + _NEEDS_REF_RE.sub(ref, m.group(1)) + "}}", text)

    return (
        b.kind,
        norm(b.text),
        tuple((k, norm(v)) for k, v in b.with_),
        tuple((k, norm(v)) for k, v in b.env),
        b.working_directory,
        b.shell,
    )


def _order_key(job: JobInstance) -> tuple[str, int, str, str]:
    first = min((s.origin for s in job.steps), default=(job.orig_job, 1 << 30))
    return (str(first[0]), int(first[1]), instance_identity(job), job.id)


def topological_order(p: Program) -> tuple[str, ...]:
    """Instances in dependency order; ties broken by (first step origin, identity, id).

    Unknown needs are ignored (``well_formed`` reports them); a cycle raises ValueError.
    """
    by_id = {j.id: j for j in p.jobs}
    preds = _needs_graph(p)
    succs: dict[str, list[str]] = {jid: [] for jid in by_id}
    indeg = {jid: len(preds[jid]) for jid in by_id}
    for jid in sorted(by_id):
        for n in preds[jid]:
            succs[n].append(jid)
    heap = [(_order_key(by_id[jid]), jid) for jid in sorted(by_id) if indeg[jid] == 0]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        _, jid = heapq.heappop(heap)
        order.append(jid)
        for s in sorted(succs[jid]):
            indeg[s] -= 1
            if indeg[s] == 0:
                heapq.heappush(heap, (_order_key(by_id[s]), s))
    if len(order) != len(by_id):
        rest = sorted(set(by_id) - set(order))
        raise ValueError("needs cycle among: " + ", ".join(rest))
    return tuple(order)


def ancestors(p: Program) -> dict[str, frozenset[str]]:
    """Transitive ``needs`` of every instance."""
    preds = _needs_graph(p)
    anc: dict[str, frozenset[str]] = {}
    for jid in topological_order(p):
        acc: set[str] = set()
        for n in preds[jid]:
            acc.add(n)
            acc |= anc[n]
        anc[jid] = frozenset(acc)
    return anc


# ---------------------------------------------------------------------------
# Region universe and effects
# ---------------------------------------------------------------------------

def _expr_ctx_keys(e: object) -> set[str]:
    keys: set[str] = set()
    stack = [e]
    while stack:
        x = stack.pop()
        if not isinstance(x, Expr):
            continue
        if x.op == "ctx" and x.args:
            keys.add(str(x.args[0]))
        elif x.op == "opaque":
            keys.update(str(a) for a in x.args[1:] if isinstance(a, str))
        stack.extend(a for a in x.args if isinstance(a, Expr))
    return keys


@dataclass(frozen=True)
class Universe:
    """Shared regions and global context keys mentioned by a set of programs.

    ⊤ tasks read/write "everything"; the universe fixes what that means so that the
    original and the candidate give a ⊤ task the same arguments.
    """

    ext_keys: tuple[str, ...] = ()
    run_names: tuple[str, ...] = ()
    global_ctx: tuple[str, ...] = ("github.event_name",)
    read_runs: tuple[str, ...] = ()   # artifact names some (non-⊤) task reads

    @staticmethod
    def of(programs: Iterable[Program]) -> Universe:
        ext: set[str] = set()
        runs: set[str] = set()
        read_runs: set[str] = set()
        glob: set[str] = {"github.event_name"}
        top = False
        for p in programs:
            for job in p.jobs:
                top = top or (job.frozen and blanket_frozen(job))
                exprs: list[object] = [job.cond] + [e for _, e in job.outputs]
                for s in job.steps:
                    c = s.contract
                    top = top or c.top
                    exprs.append(s.cond)
                    for r in c.reads | c.writes:
                        if r.kind == EXT and r.key is not None:
                            ext.add(ext_write_key(r.key, s, job) if r in c.writes else r.key)
                        elif r.kind == RUN and r.key is not None:
                            runs.add(r.key)
                        elif r.kind == CTX and r.key is not None and is_global_ctx(r.key):
                            glob.add(r.key)
                    for r in c.reads:
                        if r.kind == EXT and r.key is not None:
                            ext.add(r.key)
                        elif r.kind == RUN and r.key is not None:
                            read_runs.add(r.key)
                for e in exprs:
                    glob.update(k for k in _expr_ctx_keys(e) if is_global_ctx(k))
        if top:
            ext.add(TOP_EXT_KEY)
        return Universe(tuple(sorted(ext)), tuple(sorted(runs)), tuple(sorted(glob)),
                        tuple(sorted(read_runs)))


def instance_ctx_keys(job: JobInstance, p: Program, universe: Universe) -> tuple[str, ...]:
    """Every context key a ⊤ task of ``job`` may read."""
    keys = {"platform", "job-identity"}
    keys.update(f"matrix.{k}" for k, _ in job.matrix)
    keys.update(f"env.{k}" for k in merged_env(p, job))
    keys.update(universe.global_ctx)
    return tuple(sorted(keys))


def _needs_output_regions(job: JobInstance, p: Program) -> set[Region]:
    by_id = {j.id: j for j in p.jobs}
    return {
        Region(OUTPUT, n, name)
        for n in sorted(job.needs)
        if n in by_id
        for name, _ in by_id[n].outputs
    }


def occurrence(step: Step, job: JobInstance) -> str:
    """The originating step occurrence of ``step``: its original instance
    (``Step.origin_instance``; the original job key when unknown) and original index. It is
    stable when synthesis or emission moves the step to another job."""
    inst = step.origin_instance if step.origin_instance is not None else str(step.origin[0])
    return f"{inst}#{step.origin[1]}"


def ext_write_key(key: str, step: Step, job: JobInstance) -> str:
    """External-service writes are keyed by the originating step occurrence
    (``<service>@<occurrence>``), so separate uploads to one service are separate objects
    of P (no false overwrites, no false races); the ⊤ catch-all ``*`` is not keyed."""
    return key if key == TOP_EXT_KEY else f"{key}@{occurrence(step, job)}"


def step_effect(step: Step, job: JobInstance, p: Program, universe: Universe) -> Effect:
    """The step's effect instantiated for ``job``, with ⊤ expanded over ``universe``."""
    if not step.contract.top:
        eff = step.contract.instantiate(job.id)
        if any(r.kind == EXT for r in eff.writes):
            eff = Effect(eff.reads, frozenset(
                Region(EXT, None, ext_write_key(r.key or "", step, job)) if r.kind == EXT else r
                for r in eff.writes), eff.top)
        return eff
    local = {Region(k, job.id) for k in LOCAL_STATE_KINDS}
    ctx = {Region(CTX, job.id, k) for k in instance_ctx_keys(job, p, universe)}
    runs = {Region(RUN, None, n) for n in universe.run_names if not is_transfer_artifact(n)}
    ext = {Region(EXT, None, k) for k in universe.ext_keys}
    reads = local | ctx | runs | ext | _needs_output_regions(job, p)
    return Effect(frozenset(reads), frozenset(local | ext))


def _local_reads(eff: Effect, job: JobInstance) -> frozenset[str]:
    """Job-local kinds of ``job`` that an effect reads (writes of the others are updates)."""
    return frozenset(r.kind for r in eff.reads
                     if r.kind in LOCAL_STATE_KINDS and (r.job is None or r.job == job.id))


def blanket_frozen(job: JobInstance) -> bool:
    """A frozen job whose effect its step contracts do not describe: it has no steps (e.g.
    a remote reusable workflow) or a reason outside ``TRANSPARENT_FROZEN_REASONS``."""
    return not job.steps or any(r not in TRANSPARENT_FROZEN_REASONS for r in job.frozen_reasons)


def frozen_effect(job: JobInstance, p: Program, universe: Universe) -> Effect:
    """Effect of a frozen job treated as one opaque task.

    The union of its steps' effects (RESEARCH_SCOPE: frozen jobs get the union of their
    steps' guide contracts, ⊤ only for unclassifiable steps) without its own job-local
    regions, plus every context value of the instance, the outputs of its needs, and its own
    outputs. A ``blanket_frozen`` job instead reads every shared region and writes every
    external object and every (non-transfer) artifact name of the universe.
    """
    ctx = {Region(CTX, job.id, k) for k in instance_ctx_keys(job, p, universe)}
    outs = {Region(OUTPUT, job.id, name) for name, _ in job.outputs}
    needs_out = _needs_output_regions(job, p)
    if blanket_frozen(job):
        runs = {Region(RUN, None, n) for n in universe.run_names if not is_transfer_artifact(n)}
        ext = {Region(EXT, None, k) for k in universe.ext_keys}
        uploads = {
            Region(RUN, None, r.key)
            for s in job.steps
            for r in s.contract.writes
            if r.kind == RUN and r.key is not None
        }
        # It may upload any artifact; the names that matter are the ones some task reads.
        maybe = {Region(RUN, None, n) for n in universe.read_runs if not is_transfer_artifact(n)}
        return Effect(frozenset(ctx | runs | ext | needs_out), frozenset(ext | maybe | uploads | outs))
    reads: set[Region] = set(ctx) | needs_out
    writes: set[Region] = set(outs)
    internal = set(LOCAL_STATE_KINDS) | {ST_STEP}
    for s in job.steps:
        e = step_effect(s, job, p, universe)
        reads |= {r for r in e.reads if r.kind not in internal}
        writes |= {r for r in e.writes if r.kind not in internal}
    return Effect(frozenset(reads), frozenset(writes))


# ---------------------------------------------------------------------------
# Well-formedness
# ---------------------------------------------------------------------------

def _shared_access(job: JobInstance, p: Program, universe: Universe):
    reads: dict[Region, set] = defaultdict(set)
    writes: dict[Region, set] = defaultdict(set)
    unit = unit_origin(job)
    if job.frozen:
        eff = frozen_effect(job, p, universe)
        for r in eff.reads:
            if r.kind in SHARED_KINDS:
                reads[r].add(unit)
        for r in eff.writes:
            if r.kind in SHARED_KINDS:
                writes[r].add(unit)
        return reads, writes
    for s in job.steps:
        eff = step_effect(s, job, p, universe)
        for r in eff.reads:
            if r.kind in SHARED_KINDS:
                reads[r].add(tuple(s.origin))
        for r in eff.writes:
            if r.kind in SHARED_KINDS:
                writes[r].add(tuple(s.origin))
    for name, _ in job.outputs:
        writes[Region(OUTPUT, job.id, name)].add(unit)
    return reads, writes


def well_formed(p: Program) -> list[Issue]:
    """Structural problems and races between instances not ordered by ``needs*``.

    Two unordered instances race on a shared region (run, ext, output) when one writes it
    and the other reads or writes it. Two unordered uploads of one artifact name are
    reported as ``duplicate-artifact`` (a GitHub error for v4 artifacts).
    """
    issues: list[Issue] = []
    counts = Counter(j.id for j in p.jobs)
    for jid in sorted(counts):
        if counts[jid] > 1:
            issues.append(Issue("duplicate-id", f"instance id {jid!r} occurs {counts[jid]} times", jobs=(jid,)))
    ids = set(counts)
    for job in sorted(p.jobs, key=lambda j: j.id):
        for n in sorted(job.needs):
            if n not in ids:
                issues.append(Issue("unknown-need", f"{job.id} needs unknown instance {n}", jobs=(job.id, n)))
    try:
        topological_order(p)
    except ValueError as exc:
        issues.append(Issue("cycle", str(exc)))
    if issues:
        return sorted(issues, key=Issue.sort_key)

    universe = Universe.of([p])
    anc = ancestors(p)
    by_id = {j.id: j for j in p.jobs}
    access = {jid: _shared_access(by_id[jid], p, universe) for jid in sorted(by_id)}
    for a, b in combinations(sorted(by_id), 2):
        if a in anc[b] or b in anc[a]:
            continue
        ra, wa = access[a]
        rb, wb = access[b]
        for r in sorted(set(wa) | set(wb), key=region_key):
            both = r in wa and r in wb
            a_then_b = r in wa and r in rb
            b_then_a = r in wb and r in ra
            if not (both or a_then_b or b_then_a):
                continue
            origins = set(wa.get(r, ())) | set(wb.get(r, ()))
            if r in wa:
                origins |= rb.get(r, set())
            if r in wb:
                origins |= ra.get(r, set())
            if both and r.kind == RUN:
                kind, what = "duplicate-artifact", f"unordered instances {a} and {b} both upload artifact {r.key!r}"
            else:
                kind, what = "race", f"unordered instances {a} and {b} conflict on {r}"
            issues.append(Issue(kind, what, str(r), (a, b), tuple(sorted(origins))))
    return sorted(issues, key=Issue.sort_key)


def cache_issues(p: Program) -> list[Issue]:
    """Findings of the cache-transparency check (guide 7): every cache restore that
    ``contracts.annotate`` made opaque because a later task of its job writes ``dep``. They
    are reported, never blocking: the opaque contract already makes the semantics
    conservative."""
    out: list[Issue] = []
    for job in sorted(p.jobs, key=lambda j: j.id):
        for i, s in enumerate(job.steps):
            if OPAQUE_CACHE_TEMPLATE in s.contract.template.split("+"):
                out.append(Issue("cache-opaque",
                                 f"{job.id} step {i} ({binding_label(s.binding)}): a later task "
                                 "writes dep, so the cache is treated as opaque",
                                 f"{DEP}:{job.id}", (job.id,), (tuple(s.origin),)))
    return out


def inherited(issue: Issue, original_issues: Sequence[Issue]) -> bool:
    """A race of the candidate that the original already has (same kind, region, origins)."""
    if issue.kind not in RACE_ISSUES:
        return False
    pool: set = set()
    for o in original_issues:
        if o.kind == issue.kind and o.region == issue.region:
            pool.update(o.origins)
    return bool(pool) and set(issue.origins) <= pool


def blocking_issues(original_issues: Sequence[Issue], candidate_issues: Sequence[Issue]) -> list[Issue]:
    """Candidate issues that the original does not already have (``inherited``), computed
    with an index of the original's race origins per (kind, region)."""
    pools: dict[tuple[str, str | None], set] = defaultdict(set)
    for o in original_issues:
        if o.kind in RACE_ISSUES:
            pools[(o.kind, o.region)].update(o.origins)
    out: list[Issue] = []
    for i in candidate_issues:
        pool = pools.get((i.kind, i.region)) if i.kind in RACE_ISSUES else None
        if not (pool and set(i.origins) <= pool):
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# Artifact layout (exact-copy transfer semantics)
# ---------------------------------------------------------------------------

def _norm_path(text: str) -> str | None:
    t = (text or "").strip()
    if not t or "${{" in t or any(ch in t for ch in "*?[]!") or t.startswith(("/", "~")):
        return None
    n = posixpath.normpath(t)
    if n == ".." or n.startswith("../"):
        return None
    return n


def _covers(u: str, q: str) -> bool:
    return u == "." or q == u or q.startswith(u + "/")


def artifact_root(paths: Sequence[str]) -> str:
    """Directory that upload-artifact@v4 uses as the artifact root for these paths."""
    if len(paths) == 1:
        return paths[0]
    parts = [p.split("/") for p in paths]
    common: list[str] = []
    for comps in zip(*parts):
        if all(c == comps[0] for c in comps):
            common.append(comps[0])
        else:
            break
    return "/".join(common) or "."


def _out_contribution(step: Step) -> list[str] | None:
    c = step.contract
    paths = list(c.out_paths)
    if not paths and c.template == DOWNLOAD_TEMPLATE:
        paths = [_with(step).get("path") or "."]
    if not paths:
        return None
    norm = [_norm_path(x) for x in paths]
    if any(n is None for n in norm):
        return None
    return [n for n in norm if n is not None]


def cleans_workspace(step: Step) -> bool:
    """A checkout that resets the workspace (``clean`` not ``false``): afterwards ``out`` is
    empty, whatever earlier steps wrote there."""
    return step.contract.template == CHECKOUT_TEMPLATE and \
        (_with(step).get("clean") or "").strip().lower() != "false"


@dataclass(frozen=True)
class TransferPlan:
    """Exact-copy transfers of a program (see ``transfer_plan``).

    ``exact``: artifact name -> declared paths, for names whose every upload is an exact
    transfer; ``packs``: (instance id, step index) of the pack steps that carry the whole
    ``out`` region of their job.
    """

    exact: Mapping[str, tuple[str, ...]]
    packs: frozenset[tuple[str, int]]


def transfer_plan(p: Program) -> TransferPlan:
    """Which artifacts are exact copies of a job's ``out`` region.

    A transfer is exact when it follows the idiom (``contracts.parse_transfer``): a pack step
    that tars declared paths covering every path written into ``out`` since the job started
    or its workspace was last cleaned (each earlier ``out`` writer must declare its
    ``out_paths``; a ⊤ step makes the content unknown), immediately followed by the upload of
    that tarball. An artifact name is exact when every upload of it is such a transfer and all
    carry the same paths; any other upload (a plain ``actions/upload-artifact``, whose
    archive rebases paths, drops hidden files and resets permissions) is uninterpreted.
    """
    carried: dict[str, set] = defaultdict(set)
    packs: set[tuple[str, int]] = set()
    for job in sorted(p.jobs, key=lambda j: j.id):
        if job.frozen:
            for s in job.steps:
                for r in s.contract.writes:
                    if r.kind == RUN and r.key is not None:
                        carried[r.key].add(None)
            continue
        contrib: list[str] | None = []
        packed: tuple[int, str, tuple[str, ...]] | None = None
        for i, s in enumerate(job.steps):
            c = s.contract
            t = None if c.top else parse_transfer(s.binding)
            if t is not None and t.role == "pack":
                covers = contrib is not None and all(any(_covers(u, q) for u in t.paths) for q in contrib)
                if covers:
                    packs.add((job.id, i))
                    packed = (i, t.name, t.paths)
            elif t is not None and t.role == "upload":
                ok = packed is not None and packed[0] == i - 1 and packed[1] == t.name
                carried[t.name].add(packed[2] if ok and packed is not None else None)
            elif not c.top and c.template == UPLOAD_TEMPLATE:
                for r in c.writes:
                    if r.kind == RUN and r.key is not None:
                        carried[r.key].add(None)
            if c.top:
                contrib = None
            elif cleans_workspace(s):
                contrib = []
            elif any(r.kind == OUT for r in c.writes):
                paths = list(t.paths) if t is not None and t.role == "unpack" else _out_contribution(s)
                contrib = None if contrib is None or paths is None else contrib + paths
    exact = {n: next(iter(v)) for n, v in sorted(carried.items()) if len(v) == 1 and None not in v}
    return TransferPlan(exact, frozenset(packs))


def artifact_plan(p: Program) -> dict[str, tuple[str, ...] | None]:
    """Declared paths of every uploaded artifact name that is an exact transfer, else None."""
    plan = transfer_plan(p)
    names: set[str] = set(plan.exact)
    for job in p.jobs:
        for s in job.steps:
            names |= {r.key for r in s.contract.writes if r.kind == RUN and r.key is not None}
            t = None if s.contract.top else parse_transfer(s.binding)
            if t is not None and t.role == "upload":
                names.add(t.name)
    return {n: plan.exact.get(n) for n in sorted(names)}


def persistent_regions(original: Program, universe: Universe) -> tuple[str, ...]:
    """P: every external object of the universe, plus the original's non-transfer artifacts."""
    names = {f"ext:{k}" for k in universe.ext_keys}
    for job in original.jobs:
        for s in job.steps:
            for r in s.contract.writes:
                if r.kind == RUN and r.key is not None and not is_transfer_artifact(r.key):
                    names.add(f"run:{r.key}")
    return tuple(sorted(names))


# ---------------------------------------------------------------------------
# Symbolic context
# ---------------------------------------------------------------------------

class SymbolicContext:
    """Constants and uninterpreted functions shared by the programs being compared.

    Build one context per comparison (``SymbolicContext.for_programs(original, candidate)``)
    and evaluate both programs in it, so that equal computations get equal terms. Every
    SymbolicContext owns a fresh ``z3.Context``: solver behavior (models, resource counts)
    then depends only on this comparison, not on earlier work in the same process.
    """

    def __init__(self, universe: Universe) -> None:
        self.universe = universe
        self.z3 = z3.Context()
        self.val_sort = z3.DeclareSort("Val", self.z3)
        self.status_sort, consts = z3.EnumSort("Status", list(STATUS_NAMES), ctx=self.z3)
        self.status: dict[str, z3.ExprRef] = dict(zip(STATUS_NAMES, consts))
        self.true = z3.BoolVal(True, self.z3)
        self.false = z3.BoolVal(False, self.z3)
        self._literals: dict[str, tuple[z3.ExprRef, bool]] = {}
        self._symbols: dict[str, z3.ExprRef] = {}
        self._funcs: dict[tuple, z3.FuncDecl] = {}
        self._func_names: dict[str, tuple] = {}
        self.labels: dict[str, str] = {}
        self.truthy = z3.Function("truthy", self.val_sort, z3.BoolSort(self.z3))
        self.event_name = self.symbol("ctx:github.event_name")
        for e in EVENTS:
            self.lit(e)

    @classmethod
    def for_programs(cls, *programs: Program) -> SymbolicContext:
        return cls(Universe.of(programs))

    # constants ---------------------------------------------------------------
    def literal(self, name: str, truthy: bool) -> z3.ExprRef:
        """A constant distinct from every other literal (asserted in ``background``)."""
        hit = self._literals.get(name)
        if hit is None:
            hit = (z3.Const(name, self.val_sort), truthy)
            self._literals[name] = hit
        return hit[0]

    def lit(self, s: str) -> z3.ExprRef:
        return self.literal("str:" + s, s != "")

    def symbol(self, name: str) -> z3.ExprRef:
        c = self._symbols.get(name)
        if c is None:
            c = z3.Const(name, self.val_sort)
            self._symbols[name] = c
        return c

    def func(self, kind: str, key: tuple, arity: int, rng: str = "val", label: str = "") -> z3.FuncDecl:
        """Uninterpreted function ``Val^arity -> Val`` (rng="val") or ``-> Bool`` (rng="bool")."""
        full = (kind, key, arity, rng)
        f = self._funcs.get(full)
        if f is not None:
            return f
        name = f"{kind}:{_digest(full)}"
        other = self._func_names.get(name)
        if other is not None and other != full:  # pragma: no cover - digest collision
            raise RuntimeError(f"function name collision for {name}")
        self._func_names[name] = full
        sort = z3.BoolSort(self.z3) if rng == "bool" else self.val_sort
        f = z3.Function(name, *([self.val_sort] * arity), sort)
        self._funcs[full] = f
        if label:
            self.labels[name] = label
        return f

    def literals(self) -> list[tuple[str, z3.ExprRef]]:
        return [(n, self._literals[n][0]) for n in sorted(self._literals)]

    def global_symbols(self) -> list[tuple[str, z3.ExprRef]]:
        return [(n[len("ctx:"):], c) for n, c in sorted(self._symbols.items()) if n.startswith("ctx:")]

    def background(self) -> list[z3.BoolRef]:
        """Axioms: literals are distinct, literal truthiness, and the supported events."""
        lits = [self._literals[n] for n in sorted(self._literals)]
        axioms: list[z3.BoolRef] = []
        if len(lits) >= 2:
            axioms.append(z3.Distinct(*[c for c, _ in lits]))
        for c, t in lits:
            axioms.append(self.truthy(c) == z3.BoolVal(t, self.z3))
        axioms.append(z3.Or(*[self.event_name == self.lit(e) for e in EVENTS]))
        return axioms


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WriteEvent:
    """One write of a shared (run/ext) region; ``guard`` is the condition it happens."""

    region: str
    guard: z3.BoolRef
    job: str
    fail_fast_group: str | None
    origin: tuple[str, int]


@dataclass
class Observation:
    order: tuple[str, ...]
    ext: dict[str, z3.ExprRef]
    run_value: dict[str, z3.ExprRef]
    run_exists: dict[str, z3.BoolRef]
    job_status: dict[str, z3.ExprRef]
    job_ran: dict[str, z3.BoolRef]
    checks: dict[str, z3.BoolRef]            # check name -> verdict (conjunction over instances)
    check_of: dict[str, str]                 # instance id -> check name
    step_status: dict[tuple[str, int], z3.ExprRef]
    step_labels: dict[tuple[str, int], str]
    step_reads: dict[tuple[str, int], tuple[tuple[tuple[str, str, str], z3.ExprRef], ...]]
    writes: dict[str, tuple[WriteEvent, ...]]
    fail_fast_groups: dict[str, tuple[str, ...]]
    ctx: SymbolicContext
    outputs: dict[tuple[str, str], z3.ExprRef] = field(default_factory=dict)  # (instance, name)
    _nofail_cache: dict[str, z3.BoolRef] = field(default_factory=dict, repr=False)

    def verdict(self, instance_id: str) -> z3.BoolRef:
        return passes(self.job_status[instance_id])

    def no_member_failed(self, group: str) -> z3.BoolRef:
        hit = self._nofail_cache.get(group)
        if hit is None:
            members = self.fail_fast_groups.get(group, ())
            hit = _and(self.ctx.z3, *[_not(status_is(self.job_status[m], FAILURE)) for m in members])
            self._nofail_cache[group] = hit
        return hit

    def region_value(self, name: str) -> tuple[z3.BoolRef, z3.ExprRef]:
        """(exists, value) of a shared region named ``run:<n>`` or ``ext:<k>``."""
        kind, _, key = name.partition(":")
        if kind == RUN:
            return (
                self.run_exists.get(key, self.ctx.false),
                self.run_value.get(key, self.ctx.symbol(f"init:run:{key}")),
            )
        if kind == EXT:
            return self.ctx.true, self.ext.get(key, self.ctx.symbol(f"init:ext:{key}"))
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_NONE = object()


@dataclass(frozen=True)
class _V:
    """An expression value: a z3 term, whether it is Bool-sorted, and a concrete value."""

    term: z3.ExprRef
    is_bool: bool
    py: object = _NONE


@dataclass
class _StepRec:
    ran: z3.BoolRef
    status: z3.ExprRef
    key: tuple
    args: tuple
    template: str


@dataclass
class _JobState:
    local: dict[str, z3.ExprRef]
    nofail: z3.BoolRef  # no step of the job failed so far, assuming the job runs
    statuses: list[z3.ExprRef] = field(default_factory=list)
    records: list["_StepRec"] = field(default_factory=list)
    by_step_id: dict[str, _StepRec] = field(default_factory=dict)
    out_dirty: bool = False
    packed: dict[str, tuple[int, z3.ExprRef]] = field(default_factory=dict)   # exact pack: (index, out)
    fetched: dict[str, tuple[int, z3.ExprRef]] = field(default_factory=dict)  # exact download


def _num_str(x: float) -> str:
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if x.is_integer():
            return str(int(x))
    return str(x)


def _gh_truthy(x: object) -> bool:
    if x is None or x is False:
        return False
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return not (x == 0 or (isinstance(x, float) and math.isnan(x)))
    if isinstance(x, str):
        return x != ""
    return True


def _gh_num(x: object) -> float:
    if x is None:
        return 0.0
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return 0.0
        try:
            return float(int(s, 16)) if s.lower().startswith("0x") else float(s)
        except ValueError:
            return math.nan
    return math.nan


_EXPR_MODULE: object = None
_EXPR_LOADED = False


def _expr_module():
    """ghav.expr, imported lazily (None if unavailable); used for concrete folding only."""
    global _EXPR_MODULE, _EXPR_LOADED
    if not _EXPR_LOADED:
        try:
            import ghav.expr as mod
        except ImportError:  # pragma: no cover - expr.py is part of the package
            mod = None
        _EXPR_MODULE, _EXPR_LOADED = mod, True
    return _EXPR_MODULE


_BASE_FOLD_ERRORS = (ArithmeticError, AttributeError, KeyError, IndexError, TypeError, ValueError)


def _fold_errors() -> tuple[type[BaseException], ...]:
    """Exceptions that mean "cannot fold concretely" (includes ghav.expr.Unresolved)."""
    mod = _expr_module()
    return _BASE_FOLD_ERRORS + ((mod.Unresolved,) if mod is not None else ())


def _truthy_py(x: object) -> bool:
    mod = _expr_module()
    return mod.truthy(x) if mod is not None else _gh_truthy(x)


def _loose_eq(a: object, b: object) -> bool:
    mod = _expr_module()
    return mod.loose_eq(a, b) if mod is not None else _gh_eq(a, b)


def _gh_eq(a: object, b: object) -> bool:
    """GitHub's loose equality on concrete values (strings compare case-insensitively)."""
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        return a is b
    if type(a) is type(b):
        return a == b
    return _gh_num(a) == _gh_num(b)


def _gh_format(fmt: str, args: Sequence[object]) -> str:
    out: list[str] = []
    i = 0
    while i < len(fmt):
        ch = fmt[i]
        if fmt.startswith("{{", i):
            out.append("{")
            i += 2
        elif fmt.startswith("}}", i):
            out.append("}")
            i += 2
        elif ch == "{":
            j = fmt.find("}", i)
            idx = fmt[i + 1 : j] if j > 0 else ""
            if not idx.isdigit() or int(idx) >= len(args):
                raise ValueError("bad format")
            v = args[int(idx)]
            out.append(v if isinstance(v, str) else json.dumps(v))
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


class _Evaluator:
    def __init__(self, program: Program, ctx: SymbolicContext, assume_success: bool) -> None:
        self.p = program
        self.ctx = ctx
        self.zc = ctx.z3
        self.T = ctx.true
        self.F = ctx.false
        self.S = ctx.status
        self.assume_success = assume_success
        self.by_id = {j.id: j for j in program.jobs}
        self.order = topological_order(program)
        self.anc = ancestors(program)
        self.plan = transfer_plan(program)
        self.frozen_tasks: dict[str, tuple[tuple, tuple, z3.BoolRef, str]] = {}
        uni = ctx.universe
        self.ext: dict[str, z3.ExprRef] = {k: ctx.symbol(f"init:ext:{k}") for k in uni.ext_keys}
        self.run_value: dict[str, z3.ExprRef] = {n: ctx.symbol(f"init:run:{n}") for n in uni.run_names}
        self.run_exists: dict[str, z3.BoolRef] = {n: self.F for n in uni.run_names}
        self.outputs: dict[tuple[str, str], z3.ExprRef] = {}
        self.job_status: dict[str, z3.ExprRef] = {}
        self.job_ran: dict[str, z3.BoolRef] = {}
        self.step_status: dict[tuple[str, int], z3.ExprRef] = {}
        self.step_labels: dict[tuple[str, int], str] = {}
        self.step_reads: dict[tuple[str, int], tuple] = {}
        self.writes: dict[str, list[WriteEvent]] = defaultdict(list)
        self.env_cache: dict[str, dict[str, str]] = {}

    # entry -------------------------------------------------------------------
    def run(self) -> Observation:
        for jid in self.order:
            self._job(self.by_id[jid])
        checks: dict[str, list[z3.BoolRef]] = defaultdict(list)
        check_of: dict[str, str] = {}
        groups: dict[str, list[str]] = defaultdict(list)
        for jid in sorted(self.by_id):
            job = self.by_id[jid]
            checks[job.check_name].append(passes(self.job_status[jid]))
            check_of[jid] = job.check_name
            if job.fail_fast_group:
                groups[job.fail_fast_group].append(jid)
        return Observation(
            order=self.order,
            ext=dict(sorted(self.ext.items())),
            run_value=dict(sorted(self.run_value.items())),
            run_exists=dict(sorted(self.run_exists.items())),
            job_status=dict(sorted(self.job_status.items())),
            job_ran=dict(sorted(self.job_ran.items())),
            checks={n: _and(self.zc, *checks[n]) for n in sorted(checks)},
            check_of=check_of,
            step_status=dict(sorted(self.step_status.items())),
            step_labels=dict(sorted(self.step_labels.items())),
            step_reads=dict(sorted(self.step_reads.items())),
            writes={r: tuple(self.writes[r]) for r in sorted(self.writes)},
            fail_fast_groups={g: tuple(sorted(m)) for g, m in sorted(groups.items())},
            ctx=self.ctx,
            outputs={k: self.outputs[k] for k in sorted(self.outputs)},
        )

    # jobs --------------------------------------------------------------------
    def _job(self, job: JobInstance) -> None:
        ran = self._bool(self._expr(job.cond, job, None))
        self.job_ran[job.id] = ran
        if job.frozen:
            self.job_status[job.id] = self._frozen(job, ran)
            return
        js = _JobState(local={k: self.ctx.symbol(f"init:{k}") for k in LOCAL_STATE_KINDS}, nofail=self.T)
        for i, step in enumerate(job.steps):
            self._step(job, i, step, js, ran)
        self.job_status[job.id] = _ite(ran, _ite(js.nofail, self.S["success"], self.S["failure"]), self.S["skipped"])
        for name, e in sorted(job.outputs, key=lambda kv: kv[0]):
            val = self._val(self._expr(e, job, js))
            self.outputs[(job.id, name)] = _ite(ran, val, self.ctx.lit(""))

    def _frozen(self, job: JobInstance, ran: z3.BoolRef) -> z3.ExprRef:
        eff = frozen_effect(job, self.p, self.ctx.universe)
        reads = sorted(eff.reads, key=self._read_order)
        slots = tuple(self._slot(r) for r in reads)
        assuming = _Assuming(ran)
        args = tuple(assuming.value(self._read(r, job, None)) for r in reads)
        bkey = ("frozen", job.check_name, tuple(task_key(s, job, self.by_id) for s in job.steps))
        key = (bkey, slots)
        label = f"frozen job {job.id}"
        self.frozen_tasks[job.id] = (key, args, ran, label)
        ok = self._ok(key, args, label)
        for w in sorted(eff.writes, key=region_key):
            new = self._F(key, w, args, label)
            if w.kind == RUN:
                made = self.ctx.func("E", key + (self._slot(w),), len(args), "bool", label)(*args)
                wrote = _and(self.zc, ran, made)
                self._record(w, wrote, job, unit_origin(job))
                self.run_value[w.key] = _ite(wrote, new, self._run_value(w.key))
                self.run_exists[w.key] = _or(self.zc, self.run_exists.get(w.key, self.F), wrote)
            elif w.kind == OUTPUT:
                self.outputs[(w.job, w.key)] = _ite(ran, new, self.ctx.lit(""))
            else:
                self._write(w, new, ran, job, unit_origin(job), None)
        return _ite(ran, _ite(ok, self.S["success"], self.S["failure"]), self.S["skipped"])

    # steps -------------------------------------------------------------------
    def _step(self, job: JobInstance, i: int, step: Step, js: _JobState, ran: z3.BoolRef) -> None:
        cond = self._bool(self._expr(step.cond, job, js))
        g = _and(self.zc, ran, cond)
        eff = step_effect(step, job, self.p, self.ctx.universe)
        reads = sorted(eff.reads, key=self._read_order)
        slots = tuple(self._slot(r) for r in reads)
        # Arguments only matter when the step runs, so simplify them under its guard.
        assuming = _Assuming(g)
        args = tuple(assuming.value(self._read(r, job, js)) for r in reads)
        key = (task_key(step, job, self.by_id), slots)
        template = "top" if step.contract.top else step.contract.template
        label = binding_label(step.binding)
        origin = tuple(step.origin)
        transfer = None if step.contract.top else parse_transfer(step.binding)
        exact = transfer is not None and self._exact_transfer(transfer, job, i, js)
        if exact:
            ok = self._transfer(transfer, g, js, i, job, origin)
        elif template == UPLOAD_TEMPLATE:
            ok = self._upload(job, step, eff, key, args, g, js, label, origin)
        elif template == DOWNLOAD_TEMPLATE:
            ok = self._download(job, step, eff, key, args, g, js, label, origin)
        elif template == CACHE_TEMPLATE:
            fails = _truthy_text(_with(step).get("fail-on-cache-miss"))
            ok = self.F if fails and not self.assume_success else self.T
            self._generic_writes(eff, key, args, g, job, js, label, origin)
        else:
            ok = self._ok(key, args, label)
            self._generic_writes(eff, key, args, g, job, js, label, origin)
        if cleans_workspace(step):
            js.out_dirty = False  # the checkout emptied `out`
        status = _ite(g, _ite(ok, self.S["success"], self.S["failure"]), self.S["skipped"])
        js.statuses.append(status)
        if _implies(js.nofail, cond):  # the step runs whenever nothing failed so far
            js.nofail = _and(self.zc, js.nofail, ok)
        else:
            js.nofail = _and(self.zc, js.nofail, _or(self.zc, _not(cond), ok))
        rec = _StepRec(ran=g, status=status, key=key, args=args, template=template)
        js.records.append(rec)
        if step.binding.step_id:
            js.by_step_id[step.binding.step_id] = rec
        self.step_status[(job.id, i)] = status
        self.step_labels[(job.id, i)] = label
        self.step_reads[(job.id, i)] = tuple(zip(slots, args))

    def _generic_writes(self, eff, key, args, g, job, js, label, origin) -> None:
        read = _local_reads(eff, job)
        for w in sorted(eff.writes, key=region_key):
            self._write(w, self._F(key, w, args, label), g, job, origin, js, read)

    def _exact_transfer(self, t, job: JobInstance, i: int, js: _JobState) -> bool:
        """Is this idiom step part of an exact transfer (``transfer_plan``)?"""
        paths = self.plan.exact.get(t.name)
        if paths is None:
            return False
        if t.role == "pack":
            return (job.id, i) in self.plan.packs and tuple(t.paths) == paths
        if t.role == "upload":
            hit = js.packed.get(t.name)
            return hit is not None and hit[0] == i - 1
        if t.role == "download":
            return True
        hit = js.fetched.get(t.name)  # unpack
        return hit is not None and hit[0] == i - 1 and tuple(t.paths) == paths and not js.out_dirty

    def _transfer(self, t, g: z3.BoolRef, js: _JobState, i: int, job: JobInstance, origin) -> z3.BoolRef:
        """Exact transfer steps: the artifact carries exactly the producer's ``out`` region and
        the unpack restores it into the consumer's empty ``out`` region. The idiom's tarball
        lives in a GHAV-private file under ``$RUNNER_TEMP`` that no original task reads, so pack,
        download and unpack change nothing else; transfers are assumed to succeed except that a
        download fails when the artifact does not exist and an upload fails when it exists."""
        if t.role == "pack":
            js.packed[t.name] = (i, _Assuming(g).value(js.local[OUT]))
            return self.T
        if t.role == "upload":
            ok = self.T if self.assume_success else _not(self.run_exists.get(t.name, self.F))
            wrote = _and(self.zc, g, ok)
            value = js.packed[t.name][1]
            self._write(Region(RUN, None, t.name), _Assuming(wrote).value(value), wrote, job, origin, js)
            return ok
        if t.role == "download":
            ok = self.T if self.assume_success else self.run_exists.get(t.name, self.F)
            js.fetched[t.name] = (i, _Assuming(_and(self.zc, g, ok)).value(self._run_value(t.name)))
            return ok
        value = js.fetched[t.name][1]
        js.local[OUT] = _ite(g, _Assuming(g).value(value), js.local[OUT])
        js.out_dirty = True
        return self.T

    def _upload(self, job, step, eff, key, args, g, js, label, origin) -> z3.BoolRef:
        """A plain upload (not an exact transfer): the artifact is an uninterpreted function
        of the upload's reads. It fails when the name already exists (unless ``overwrite``),
        and possibly (uninterpreted) under ``if-no-files-found: error``."""
        w = _with(step)
        runs = sorted((r for r in eff.writes if r.kind == RUN), key=region_key)
        ok = self.T
        if not _truthy_text(w.get("overwrite")):
            ok = _and(self.zc, *[_not(self.run_exists.get(r.key, self.F)) for r in runs])
        if (w.get("if-no-files-found") or "").strip().lower() == "error":
            ok = _and(self.zc, ok, self._ok(key, args, label))
        if self.assume_success:
            ok = self.T
        wrote = _and(self.zc, g, ok)
        for r in runs:
            self._write(r, self._F(key, r, args, label), wrote, job, origin, js)
        read = _local_reads(eff, job)
        for r in sorted(eff.writes, key=region_key):
            if r.kind != RUN:
                self._write(r, self._F(key, r, args, label), g, job, origin, js, read)
        return ok

    def _download(self, job, step, eff, key, args, g, js, label, origin) -> z3.BoolRef:
        """A plain download (not an exact transfer): fails iff the artifact does not exist;
        what it writes is an uninterpreted function of its reads (the artifact included)."""
        runs = sorted((r for r in eff.reads if r.kind == RUN), key=region_key)
        if len(runs) == 1:
            ok = self.T if self.assume_success else self.run_exists.get(runs[0].key, self.F)
        else:
            ok = self._ok(key, args, label)
        wrote = _and(self.zc, g, ok)
        read = _local_reads(eff, job)
        for r in sorted(eff.writes, key=region_key):
            self._write(r, self._F(key, r, args, label), wrote, job, origin, js, read)
        return ok

    # slots -------------------------------------------------------------------
    def _slot(self, r: Region) -> tuple[str, str, str]:
        """``slot_key`` with the producing instance of a job output or job status named by
        its identity, so that the slot does not depend on instance ids (which differ between
        an original and a re-lowered or developer-written candidate)."""
        if r.kind in (OUTPUT, ST_JOB) and r.job in self.by_id:
            return (r.kind, "@" + instance_identity(self.by_id[r.job]), r.key or "")
        return slot_key(r)

    def _read_order(self, r: Region) -> tuple:
        return (self._slot(r), region_key(r))

    # state -------------------------------------------------------------------
    def _run_value(self, name: str) -> z3.ExprRef:
        return self.run_value.get(name, self.ctx.symbol(f"init:run:{name}"))

    def _record(self, r: Region, guard, job: JobInstance, origin) -> None:
        self.writes[str(r)].append(WriteEvent(str(r), guard, job.id, job.fail_fast_group, tuple(origin)))

    def _write(self, r: Region, new, cond, job: JobInstance, origin, js: _JobState | None,
               read: frozenset[str] = frozenset()) -> None:
        """Write ``new`` into ``r`` when ``cond`` holds. A job-local write is an update: when
        the task does not read the kind (``read``), ``new`` is only the task's change, and the
        region becomes ``U(old, new)`` (``_update``)."""
        k = r.kind
        if k in LOCAL_STATE_KINDS:
            if js is None or (r.job is not None and r.job != job.id):
                return
            if k not in read:
                new = self._update(k, _Assuming(cond).value(js.local[k]), new)
            js.local[k] = _ite(cond, new, js.local[k])
            if k == OUT:
                js.out_dirty = True
        elif k == EXT:
            self._record(r, cond, job, origin)
            self.ext[r.key] = _ite(cond, new, self.ext.get(r.key, self.ctx.symbol(f"init:ext:{r.key}")))
        elif k == RUN:
            self._record(r, cond, job, origin)
            self.run_value[r.key] = _ite(cond, new, self._run_value(r.key))
            self.run_exists[r.key] = _or(self.zc, self.run_exists.get(r.key, self.F), cond)
        elif k == OUTPUT:
            old = self.outputs.get((r.job, r.key), self.ctx.lit(""))
            self.outputs[(r.job, r.key)] = _ite(cond, new, old)
        # ctx and status cells are not writable by tasks

    def _read(self, r: Region, job: JobInstance, js: _JobState | None) -> z3.ExprRef:
        k = r.kind
        if k in LOCAL_STATE_KINDS:
            if js is None or (r.job is not None and r.job != job.id):
                return self.ctx.symbol(f"init:{k}")
            return js.local[k]
        if k == CTX:
            return self._ctx(job, r.key or "").term
        if k == RUN:
            return self._run_value(r.key or "")
        if k == EXT:
            return self.ext.get(r.key, self.ctx.symbol(f"init:ext:{r.key}"))
        if k == OUTPUT:
            return self._output(r.job or "", r.key or "")
        if k == ST_STEP:
            idx = int(r.key) if r.key and r.key.isdigit() else -1
            if js is not None and 0 <= idx < len(js.records):
                return self._step_cell(js.records[idx])
            return self.ctx.lit("")
        if k == ST_JOB:
            st = self.job_status.get(r.job or "")
            return self._status_val(st) if st is not None else self.ctx.lit("")
        return self.ctx.symbol(f"region:{r}")

    def _step_cell(self, rec: _StepRec) -> z3.ExprRef:
        """Value of a step's status cell as read by a later step: its status together with
        all of its outputs (``steps.<id>.outputs.*``), which are a function of its reads."""
        if rec.template == CACHE_TEMPLATE:  # cold cache: every output is empty
            outs = self.ctx.lit("")
        else:
            f = self.ctx.func("O*", rec.key, len(rec.args), "val", "outputs")
            outs = _ite(rec.ran, f(*rec.args), self.ctx.lit(""))
        cell = self.ctx.func("cell", (), 2, "val", "step cell")
        return cell(self._status_val(rec.status), outs)

    def _output(self, jid: str, name: str) -> z3.ExprRef:
        """Value of job output ``name`` of instance ``jid``. A blanket frozen job (e.g. a
        remote reusable workflow) may set outputs it does not declare: those are
        uninterpreted functions of the frozen task's reads (empty if it did not run)."""
        v = self.outputs.get((jid, name))
        if v is not None:
            return v
        task = self.frozen_tasks.get(jid)
        job = self.by_id.get(jid)
        if task is not None and job is not None and blanket_frozen(job):
            key, args, ran, label = task
            f = self._F(key, Region(OUTPUT, jid, name), args, f"{label} -> output {name}")
            return _ite(ran, f, self.ctx.lit(""))
        return self.ctx.lit("")

    def _update(self, kind: str, old: z3.ExprRef, change: z3.ExprRef) -> z3.ExprRef:
        """``U_kind(old, change)``: a job-local region after a task changed part of it (some
        paths, some PATH entries) without reading it. The rest of the region survives, so
        the new value depends on the old one; ``U`` is uninterpreted, so a task that in fact
        replaces the whole region is covered as well."""
        return self.ctx.func("U", (kind,), 2, "val", f"update {kind}")(old, change)

    def _F(self, key: tuple, w: Region, args: tuple, label: str) -> z3.ExprRef:
        f = self.ctx.func("F", (key[0], self._slot(w), key[1]), len(args), "val", f"{label} -> {w.kind}:{w.key or ''}")
        return f(*args)

    def _ok(self, key: tuple, args: tuple, label: str) -> z3.BoolRef:
        if self.assume_success:
            return self.T
        return self.ctx.func("ok", key, len(args), "bool", f"ok({label})")(*args)

    # context -----------------------------------------------------------------
    def _env(self, job: JobInstance) -> dict[str, str]:
        env = self.env_cache.get(job.id)
        if env is None:
            env = merged_env(self.p, job)
            self.env_cache[job.id] = env
        return env

    def _ctx(self, job: JobInstance, key: str) -> _V:
        if key == "platform":
            return self._py(job.runs_on)
        if key == "job-identity":
            return _V(self.ctx.literal(f"jobid:{instance_identity(job)}", True), False)
        if key == "matrix":
            return self._py({k: v for k, v in job.matrix})
        if key == "matrix.*":
            return self._py([v for _, v in sorted(job.matrix)])
        if key.startswith("matrix."):
            m = dict(job.matrix)
            name = key[len("matrix."):]
            return self._py(m.get(name))
        if key.startswith("env."):
            env = self._env(job)
            name = key[len("env."):]
            if name in env:
                return self._py(env[name])
            return _V(self.ctx.symbol(f"env-default:{name}@{job.runs_on}"), False)
        if key.startswith("runner."):
            return _V(self.ctx.symbol(f"runner:{key[len('runner.'):]}@{job.runs_on}"), False)
        if is_global_ctx(key):
            return _V(self.ctx.symbol(f"ctx:{key}"), False)
        return _V(self.ctx.symbol(f"ctx@{instance_identity(job)}:{key}"), False)

    # values ------------------------------------------------------------------
    def _py(self, x: object) -> _V:
        if isinstance(x, bool):
            return _V(z3.BoolVal(x, self.zc), True, x)
        if x is None:
            return _V(self.ctx.lit(""), False, None)
        if isinstance(x, (int, float)):
            return _V(self.ctx.lit(_num_str(x)), False, x)
        if isinstance(x, str):
            return _V(self.ctx.lit(x), False, x)
        text = json.dumps(x, sort_keys=True)
        return _V(self.ctx.literal("json:" + text, True), False, x)

    def _bool(self, v: _V) -> z3.BoolRef:
        if v.py is not _NONE:
            return z3.BoolVal(_truthy_py(v.py), self.zc)
        if v.is_bool:
            return v.term
        return self.ctx.truthy(v.term)

    def _val(self, v: _V) -> z3.ExprRef:
        if not v.is_bool:
            return v.term
        t = self.ctx.literal("bool:true", True)
        f = self.ctx.literal("bool:false", False)
        if v.py is not _NONE:
            return t if v.py else f
        return _ite(v.term, t, f)

    def _status_val(self, st: z3.ExprRef) -> z3.ExprRef:
        if _is_status_const(st):
            return self.ctx.lit(st.decl().name())
        lits = [self.ctx.lit(n) for n in STATUS_NAMES]
        return _ite(
            status_is(st, SUCCESS),
            lits[0],
            _ite(status_is(st, FAILURE), lits[1], _ite(status_is(st, SKIPPED), lits[2], lits[3])),
        )

    # expressions -------------------------------------------------------------
    def _expr(self, e: object, job: JobInstance, js: _JobState | None) -> _V:
        if not isinstance(e, Expr):
            return self._py(e if isinstance(e, (str, int, float, bool)) or e is None else str(e))
        op, a = e.op, e.args
        if op == "lit":
            return self._py(a[0] if a else None)
        if op == "ctx":
            return self._ctx(job, str(a[0]))
        if op == "status":
            return _V(self._status_fn(str(a[0]), job, js), True)
        if op == "step":
            return self._step_ref(js, str(a[0]), str(a[1]) if len(a) > 1 else "outcome")
        if op == "needs":
            return self._needs_ref(job, str(a[0]), str(a[1]) if len(a) > 1 else "result")
        if op == "not":
            v = self._expr(a[0], job, js)
            if v.py is not _NONE:
                return self._py(not _truthy_py(v.py))
            return _V(_not(self._bool(v)), True)
        if op in ("and", "or"):
            vs = [self._expr(x, job, js) for x in a]
            if all(v.py is not _NONE for v in vs):
                pys = [_truthy_py(v.py) for v in vs]
                return self._py(all(pys) if op == "and" else any(pys))
            bs = [self._bool(v) for v in vs]
            return _V(_and(self.zc, *bs) if op == "and" else _or(self.zc, *bs), True)
        if op in ("eq", "ne"):
            eq = self._eq(self._expr(a[0], job, js), self._expr(a[1], job, js))
            return eq if op == "eq" else (self._py(not eq.py) if eq.py is not _NONE else _V(_not(eq.term), True))
        if op in ("lt", "le", "gt", "ge"):
            x, y = self._expr(a[0], job, js), self._expr(a[1], job, js)
            if x.py is not _NONE and y.py is not _NONE:
                folded = _fold(Expr(op, (x.py, y.py)))
                if folded is not _NONE:
                    return self._py(folded)
                nx, ny = _gh_num(x.py), _gh_num(y.py)
                res = {"lt": nx < ny, "le": nx <= ny, "gt": nx > ny, "ge": nx >= ny}[op]
                return self._py(res)
            f = self.ctx.func("cmp", (op,), 2, "bool", op)
            return _V(f(self._val(x), self._val(y)), True)
        if op == "call":
            return self._call(str(a[0]), [self._expr(x, job, js) for x in a[1:]], job, js)
        if op == "opaque":
            text = str(a[0]) if a else ""
            refs = sorted({str(r) for r in a[1:]})
            folded = self._fold_opaque(e, job)
            if folded is not _NONE:
                return self._py(folded)
            vals = [self._ref_value(job, js, r) for r in refs]
            vals.append(self._val(_V(self._status_fn("success", job, js), True)))
            vals.append(self._val(_V(self._status_fn("failure", job, js), True)))
            f = self.ctx.func("opaque", (text, tuple(refs)), len(vals), "val", text)
            return _V(f(*vals), False)
        # Unknown operator: uninterpreted in its Expr arguments.
        sub = [self._val(self._expr(x, job, js)) for x in a if isinstance(x, Expr)]
        consts = tuple(repr(x) for x in a if not isinstance(x, Expr))
        f = self.ctx.func("op", (op, consts), len(sub), "val", op)
        return _V(f(*sub), False)

    def _fold_opaque(self, e: Expr, job: JobInstance) -> object:
        """Concrete value of an opaque expression whose references are all concrete."""
        mod = _expr_module()
        if mod is None:
            return _NONE

        def lookup(n: Expr) -> object:
            if n.op == "ctx":
                v = self._ctx(job, str(n.args[0]))
                if v.py is not _NONE:
                    return v.py
            raise KeyError(n)

        try:
            return mod.evaluate(e, lookup)
        except _fold_errors():  # unresolved reference or a GitHub runtime error: stay symbolic
            return _NONE

    def _ref_value(self, job: JobInstance, js: _JobState | None, path: str) -> z3.ExprRef:
        """Value of a reference path mentioned by an opaque expression."""
        parts = path.split(".")
        head = parts[0].lower()
        if head == "steps" and len(parts) >= 3:
            return self._val(self._step_ref(js, parts[1], ".".join(parts[2:])))
        if head == "needs" and len(parts) >= 3:
            return self._val(self._needs_ref(job, parts[1], ".".join(parts[2:])))
        return self._val(self._ctx(job, path))

    def _eq(self, x: _V, y: _V) -> _V:
        if x.py is not _NONE and y.py is not _NONE:
            return self._py(_loose_eq(x.py, y.py))
        if x.is_bool and y.is_bool:
            return _V(x.term == y.term, True)
        return _V(self._val(x) == self._val(y), True)

    def _status_fn(self, fn: str, job: JobInstance, js: _JobState | None) -> z3.BoolRef:
        fn = fn.lower()
        if fn == "always":
            return self.T
        if fn == "cancelled":
            return self.F
        if js is not None:  # step level: earlier steps of this job
            if fn == "success":
                return js.nofail
            if fn == "failure":
                return _not(js.nofail)
        else:  # job level: every transitive dependency
            deps = sorted(self.anc.get(job.id, frozenset()))
            sts = [self.job_status[d] for d in deps if d in self.job_status]
            if fn == "success":
                return _and(self.zc, *[status_is(s, SUCCESS) for s in sts])
            if fn == "failure":
                return _or(self.zc, *[status_is(s, FAILURE) for s in sts])
        return self.ctx.truthy(self.ctx.symbol(f"status:{fn}@{instance_identity(job)}"))

    def _step_ref(self, js: _JobState | None, sid: str, fld: str) -> _V:
        rec = js.by_step_id.get(sid) if js is not None else None
        if rec is None:
            return self._py(None)
        if fld in ("outcome", "conclusion"):
            return _V(self._status_val(rec.status), False)
        if fld.startswith("outputs."):
            name = fld[len("outputs."):]
            if rec.template == CACHE_TEMPLATE:  # cold cache: every output is empty
                return self._py("")
            f = self.ctx.func("O", (rec.key, name), len(rec.args), "val", f"output {name}")
            return _V(_ite(rec.ran, f(*rec.args), self.ctx.lit("")), False)
        return self._py(None)

    def _needs_ref(self, job: JobInstance, target: str, fld: str) -> _V:
        inst = [n for n in needs_targets(job, target, self.by_id) if n in self.job_status]
        if not inst:
            return self._py(None)
        if fld == "result":
            sts = [self.job_status[n] for n in inst]
            if len(sts) == 1:
                agg = sts[0]
            else:
                agg = _ite(
                    _or(self.zc, *[status_is(s, FAILURE) for s in sts]),
                    self.S["failure"],
                    _ite(
                        _or(self.zc, *[status_is(s, CANCELLED) for s in sts]),
                        self.S["cancelled"],
                        _ite(_and(self.zc, *[status_is(s, SKIPPED) for s in sts]), self.S["skipped"], self.S["success"]),
                    ),
                )
            return _V(self._status_val(agg), False)
        if fld.startswith("outputs.") and is_call_target(target, inst, self.by_id):
            # A call's outputs map the callee's job outputs (``on.workflow_call.outputs``),
            # which lowering does not model: uninterpreted in everything the callee produced.
            name = fld[len("outputs."):]
            vals = []
            for n in inst:
                vals.append(self._status_val(self.job_status[n]))
                vals += [self._output(n, o) for o, _ in self.by_id[n].outputs]
            f = self.ctx.func("call-output", (target, name, tuple(instance_identity(self.by_id[n]) for n in inst)),
                              len(vals), "val", f"needs.{target}.outputs.{name}")
            return _V(f(*vals), False)
        if fld.startswith("outputs."):
            name = fld[len("outputs."):]
            vals = [self._output(n, name) for n in inst]
            if len(vals) == 1 or all(v.eq(vals[0]) for v in vals):
                return _V(vals[0], False)  # every instance produces the same term
            f = self.ctx.func("matrix-output", (target, name), len(vals), "val", f"needs.{target}.outputs.{name}")
            return _V(f(*vals), False)
        return self._py(None)

    def _call(self, fname: str, args: list[_V], job: JobInstance, js: _JobState | None) -> _V:
        name = fname.lower()
        concrete = all(v.py is not _NONE for v in args)
        if name in ("success", "failure", "always", "cancelled") and not args:
            return _V(self._status_fn(name, job, js), True)
        if concrete and name != "hashfiles":
            folded = _fold(Expr("call", (fname, *[v.py for v in args])))
            if folded is not _NONE:
                return self._py(folded)
        if name in ("contains", "startswith", "endswith") and len(args) == 2:
            hay, needle = args
            if name == "contains" and isinstance(hay.py, list):
                return _V(_or(self.zc, *[self._eq(self._py(item), needle).term for item in hay.py]), True)
            if concrete and not isinstance(hay.py, (list, dict)) and not isinstance(needle.py, (list, dict)):
                h = _str_of(hay.py).lower()
                n = _str_of(needle.py).lower()
                res = n in h if name == "contains" else (h.startswith(n) if name == "startswith" else h.endswith(n))
                return self._py(res)
            f = self.ctx.func("call", (name,), 2, "bool", name)
            return _V(f(self._val(hay), self._val(needle)), True)
        if name == "fromjson" and len(args) == 1 and isinstance(args[0].py, str):
            try:
                return self._py(json.loads(args[0].py))
            except ValueError:
                pass
        if name == "tojson" and len(args) == 1 and args[0].py is not _NONE:
            return self._py(json.dumps(args[0].py, indent=2, sort_keys=True))
        if name == "format" and concrete and args and isinstance(args[0].py, str):
            try:
                return self._py(_gh_format(args[0].py, [a.py for a in args[1:]]))
            except ValueError:
                pass
        vals = [self._val(v) for v in args]
        if name == "hashfiles":  # hashes workspace files: depends on the job-local state
            if js is not None:
                vals = [js.local[SRC], js.local[DEP], js.local[OUT]] + vals
            else:
                vals = [self.ctx.symbol(f"init:{k}") for k in (SRC, DEP, OUT)] + vals
        f = self.ctx.func("call", (name,), len(vals), "val", name)
        return _V(f(*vals), False)


def _fold(e: Expr) -> object:
    """Concretely evaluate ``e`` whose non-Expr arguments are Python values, via ghav.expr."""
    mod = _expr_module()
    if mod is None:
        return _NONE
    if e.op == "call":
        name = mod.FUNCTIONS.get(str(e.args[0]).lower(), str(e.args[0]))
        args = (name, *(_as_expr(mod, a) for a in e.args[1:]))
    else:
        args = tuple(_as_expr(mod, a) for a in e.args)
    try:
        return mod.evaluate(Expr(e.op, args))
    except _fold_errors():  # unresolved or a runtime error in GitHub semantics: do not fold
        return _NONE


def _as_expr(mod, v: object) -> Expr:
    return v if isinstance(v, Expr) else mod.from_value(v)


def _str_of(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return _num_str(x)
    return str(x)


def evaluate(p: Program, ctx: SymbolicContext, *, assume_success: bool = False) -> Observation:
    """Symbolically execute ``p`` in ``ctx``.

    With ``assume_success=True`` every task succeeds (``ok`` is true); the resulting
    ``step_reads`` are the read-value terms used to form read classes.
    Raises ValueError if ``p`` has a needs cycle.
    """
    return _Evaluator(p, ctx, assume_success).run()


__all__ = [
    "CANCELLED",
    "FAILURE",
    "SKIPPED",
    "STATUS_NAMES",
    "SUCCESS",
    "TOP_EXT_KEY",
    "TRANSFER_PREFIX",
    "Issue",
    "Observation",
    "SymbolicContext",
    "Universe",
    "WriteEvent",
    "ancestors",
    "TRANSPARENT_FROZEN_REASONS",
    "TransferPlan",
    "artifact_plan",
    "artifact_root",
    "binding_label",
    "blanket_frozen",
    "blocking_issues",
    "cache_issues",
    "cleans_workspace",
    "ext_write_key",
    "occurrence",
    "transfer_plan",
    "evaluate",
    "frozen_effect",
    "instance_identity",
    "inherited",
    "is_global_ctx",
    "is_transfer_artifact",
    "passes",
    "persistent_regions",
    "region_key",
    "slot_key",
    "status_is",
    "step_effect",
    "needs_targets",
    "is_call_target",
    "task_key",
    "topological_order",
    "unit_origin",
    "well_formed",
]
