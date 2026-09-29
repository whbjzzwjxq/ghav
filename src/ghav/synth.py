"""Synthesis: read classes, sketch moves, deterministic completion, pruning, search.

``synthesize(p, cfg, repo_root=...)`` searches for a cheaper program equivalent to ``p``
(docs/ARCHITECTURE.md, "synth.py"; RESEARCH_SCOPE "Synthesis" and "Provisional decisions").
There are no rewrite rules with premises: moves only generate sketches, completion turns a
sketch into a WIR program deterministically, effect pruning rejects sketches before
checking, and every improving candidate is validated end to end by ``checker.check`` and by
the emission round trip ``checker.check(original, lower(emit(candidate)))``.

Read classes. The original program is evaluated symbolically with every task succeeding
(``semantics.evaluate(..., assume_success=True)``). Two step occurrences of non-frozen
instances are in one read class when their task keys (``semantics.task_key``) and the terms
of their read values (``Observation.step_reads``, compared by z3 AST identity) are equal.
Classes are named by a digest of their member occurrences. With ``read_classes=False`` the
task key alone forms the class (a syntactic, coarser grouping; the checker then refutes the
spurious members).

Sketch. A sketch is a set of moves applied to the original program:

* ``share(class, producer)``: one occurrence of a read class (with members in at least two
  instances) runs; every other member is displaced. ``producer`` is the encoding choice:
  ``"job"`` creates a NEW producer job that runs the backward slice of one representative
  occurrence (the earlier steps of its instance whose job-local writes it transitively
  reads), the step, and the transfer upload; an instance id makes that EXISTING occurrence
  the producer (it keeps its step and adds the upload right after it). A displaced member
  becomes a download of the transfer artifact plus a ``needs`` edge on the producer when the
  step writes ``out`` with non-empty ``out_paths`` (the only transferable region); otherwise
  it is simply dropped (its verdict is accounted for by the class, see
  ``checker.read_class_accounts``). One producer serves exactly one read class, so no
  matrix producer is ever shared across groups. Shareable steps have a default-fold
  condition (``success() && X``), are not artifact uploads/downloads, and belong to
  non-frozen instances.
* ``merge(a, b)``: the jobs of two groups of original instances become one job. The host
  (the member whose steps need their instance identity or matrix, else the least id) gives
  the job its id, check name, matrix, runs-on and settings. The merged step list is the
  longest common prefix of equal slots (same read class and condition) once, then the
  host's remaining steps, then each guest's remaining steps.

Completion (deterministic):

* Shares are applied per original instance (slots), then merges fuse the slot lists.
* Conditions take the exact form: a step whose kept original predecessors (earlier slots of
  its own original instance, including shared prefix slots and transfers standing in for
  them) are all earlier slots of its job keeps its condition; otherwise ``success() && X``
  becomes ``!cancelled() && X && steps.<p>.outcome != 'failure'`` over its kept
  predecessors ``p`` (step ids ``ghav_<n>`` are added where needed). A job that gains
  ``needs`` beyond its original ones (on a producer) gets ``X && needs.<n>.result ==
  'success'`` over its original needs ``n`` (emitted with ``!cancelled()``), so a failing
  producer never skips a consumer.
* Transfers are built from read/write sets: the transferred paths are the step's
  ``out_paths``, moved with the transfer idiom (contract guide 9,
  ``contracts.transfer_bindings``: pack and upload right after the producer's occurrence,
  download and unpack in place of each displaced one). Artifact names are ``ghav-`` plus a
  stable hash plus every instance-distinguishing value of the producing instance (its
  matrix values and those of its reusable-call path). Transfer steps carry origin index
  ``-(1000 + i)`` for original step ``i`` (``is_synth_transfer``), and their contracts come
  from ``contracts.infer`` like any other step.
* Check grouping is derived by the checker (origins plus read-class accounts).

Hard constraints (a sketch that violates them cannot be completed): merged members have
the same reusable-call path, job condition, original needs and permissions, compatible
job env (no key with two values), and no guest with job outputs or with dependents that
reference it in expressions; conditions must be expressible in the exact form; the needs
graph must stay acyclic.

Effect pruning (``effect_pruning``) rejects a completed sketch before checking when

1. ``untransferred-read``: a kept step reads a job-local region (src, dep, out, tool) whose
   writer chain in its new job differs, by read class, from its chain in the original
   instance, or a step status/output whose writer differs (a displaced writer's
   non-transferable effects are lost), or a transfer artifact is not an exact copy
   (``semantics.artifact_plan``). Job-local writes are updates (``semantics``), so the value
   of a region is determined by its chain: the last writer that also read the kind, then
   every later writer (``chain_step``); an exact transfer gives the consumer the producer's
   ``out`` chain (``Analysis.producer_out``). The backward slice of a new producer job
   follows the same chains;
   ``status-dependency``: a dropped occurrence (no transfer stands in for it) gated, through
   the status cells read by ``success()``/``failure()``, a later persistent write of its
   instance, or a dependent job that writes persistent state and is not also a dependent of
   the producer;
2. ``settings``: a moved step reads a context value (platform, matrix, env, instance
   identity) that differs in its new job;
3. ``well-formed``: ``semantics.well_formed`` reports an issue the original does not have;
4. ``fail-fast``: it transforms a member of a fail-fast matrix that is not transformable
   (RESEARCH_SCOPE: a member step with a status-function condition writes persistent state,
   a dependent uses a non-default condition or reads ``needs.<M>``, or the matrix calls a
   reusable workflow). Without effect pruning the same rule is applied after checking, as
   an acceptance guard, so the ablation never accepts more than the full configuration.

Search: best-first on estimated expected machine time (the parent's exact cost minus the
move's estimated saving), ties broken by the canonical sketch key. Branching is bounded by
counts: each expansion pushes its ``max_children`` best children, and a class tries the new
job plus at most ``max_producers`` member instances as producers (those that run most
often). Every popped sketch is one expansion (budget ``max_expansions``); it is completed and priced exactly
(``cost.expected_machine_time``); with ``cost_bound`` it is dropped when its lower bound
(cost minus the largest saving any further moves can realize) is not below the best
verified cost. An improving sketch is checked; rejected sketches are not expanded, and a
sketch rejected by the checker, the round trip, a hard constraint, or a monotone pruning
reason (all but ``untransferred-read``) is a nogood: no superset of it is evaluated.

Variants (``SynthConfig.variant``): ``full``; ``rules`` (merge plus the whole-prefix hoist of
a single matrix job into a NEW producer job when every instance shares the step and its
whole prefix, i.e. what a standard rule produces); ``no-effect-pruning``; ``no-cost-bound``;
``no-read-classes``.

Determinism: every collection is iterated in sorted order; the frontier is ordered by
(rounded estimate, canonical key); budgets count expansions; wall-clock time is reported
only (``SynthResult.wall_ms``, excluded from ``to_json(include_timing=False)``).
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ghav import checker
from ghav import cost as costmod
from ghav import emit as emitmod
from ghav import expr as gexpr
from ghav import semantics as sem
from ghav.wir import (
    CTX,
    DEP,
    EXT,
    OUT,
    RUN,
    SRC,
    ST_STEP,
    TOOL,
    TOP_CONTRACT,
    TRUE,
    Binding,
    Contract,
    Expr,
    JobInstance,
    Program,
    Step,
)

__all__ = [
    "VARIANTS",
    "Analysis",
    "Incomplete",
    "Sketch",
    "SynthConfig",
    "SynthResult",
    "artifact_name",
    "call_path",
    "complete",
    "instance_values",
    "occurrence_classes",
    "prune",
    "redundancy",
    "annotate_like",
    "default_observed",
    "roundtrip_check",
    "synthesize",
]

LOCAL_KINDS = (SRC, DEP, OUT, TOOL)
SUCCESS = gexpr.SUCCESS
NOT_CANCELLED = Expr("not", (Expr("status", ("cancelled",)),))
TRANSFER_ORIGIN_OFFSET = 1000   # transfer steps get origin index -(OFFSET + index): never an original step
EPS = 1e-6
PRODUCER_PREFIX = "ghav-share-"
STEP_ID_PREFIX = "ghav_"
MAX_ARTIFACT_NAME = 125         # `ghav-` plus at most 120 characters (the transfer idiom's limit)
_ARTIFACT_BAD = re.compile(r"[^A-Za-z0-9._-]+")
_STEP_TEXT_REF = re.compile(r"steps\.([A-Za-z_][A-Za-z0-9_-]*)\.")
_NEEDS_TEXT_REF = re.compile(r"needs\.([A-Za-z_][A-Za-z0-9_\-/\[\]=,.]*?)\.(?:result|outputs)")
# Pruning reasons whose failure no further move can repair (supersets are skipped).
MONOTONE_PRUNING = frozenset({"settings", "well-formed", "fail-fast"})


# ---------------------------------------------------------------------------
# Configuration and result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynthConfig:
    """Search configuration. ``moves``: ``full`` or ``rules``. ``rlimit`` bounds every
    solver query (z3 resource limit, deterministic; 0 = unbounded)."""

    moves: str = "full"
    effect_pruning: bool = True
    cost_bound: bool = True
    read_classes: bool = True
    max_expansions: int = 2000
    rlimit: int = 0
    variant: str = "full"
    max_children: int = 256        # children pushed per expansion (best estimates first)
    max_producers: int = 8         # instance-as-producer encodings tried per class

    def to_json(self) -> dict:
        return {
            "variant": self.variant,
            "moves": self.moves,
            "effect_pruning": self.effect_pruning,
            "cost_bound": self.cost_bound,
            "read_classes": self.read_classes,
            "max_expansions": self.max_expansions,
            "rlimit": self.rlimit,
            "max_children": self.max_children,
            "max_producers": self.max_producers,
        }

    @staticmethod
    def for_variant(name: str, **overrides: Any) -> "SynthConfig":
        if name not in VARIANTS:
            raise ValueError(f"unknown synthesis variant {name!r} (known: {', '.join(sorted(VARIANTS))})")
        return replace(VARIANTS[name], **overrides)


VARIANTS: Mapping[str, SynthConfig] = {
    "full": SynthConfig(variant="full"),
    "rules": SynthConfig(moves="rules", variant="rules"),
    "no-effect-pruning": SynthConfig(effect_pruning=False, variant="no-effect-pruning"),
    "no-cost-bound": SynthConfig(cost_bound=False, variant="no-cost-bound"),
    "no-read-classes": SynthConfig(read_classes=False, variant="no-read-classes"),
}


@dataclass
class SynthResult:
    """Outcome of ``synthesize``. Costs are expected machine seconds under the cost model.

    ``accepted`` counts verified improving candidates (each better than the previous best);
    ``checked`` counts candidates sent to the checker; ``rejected_*`` count unique sketches.
    """

    program: Program
    best_cost: float
    original_cost: float
    expansions: int
    generated: int
    checked: int
    accepted: int
    rejected_pruning: dict[str, int]
    rejected_completion: dict[str, int]
    rejected_checker: int
    rejected_roundtrip: int
    rejected_guard: int
    bound_pruned: int
    nogood_skipped: int
    shareable_classes: int
    budget_exhausted: bool
    best_moves: list[dict]
    config: SynthConfig
    cost_model: str = "mock"
    original_billed_minutes: float = 0.0
    best_billed_minutes: float = 0.0
    check: dict | None = None
    roundtrip: dict | None = None
    accepted_trace: list[dict] = field(default_factory=list)
    wall_ms: int = 0

    @property
    def improved(self) -> bool:
        return self.best_cost < self.original_cost - EPS

    def to_json(self, *, include_timing: bool = False) -> dict:
        out = {
            "config": self.config.to_json(),
            "cost_model": self.cost_model,
            "original_cost": _r(self.original_cost),
            "best_cost": _r(self.best_cost),
            "saving": _r(self.original_cost - self.best_cost),
            "saving_share": _r((self.original_cost - self.best_cost) / self.original_cost)
            if self.original_cost > 0 else 0.0,
            "original_billed_minutes": _r(self.original_billed_minutes),
            "best_billed_minutes": _r(self.best_billed_minutes),
            "improved": self.improved,
            "expansions": self.expansions,
            "generated": self.generated,
            "checked": self.checked,
            "accepted": self.accepted,
            "rejected": {
                "pruning": dict(sorted(self.rejected_pruning.items())),
                "completion": dict(sorted(self.rejected_completion.items())),
                "checker": self.rejected_checker,
                "roundtrip": self.rejected_roundtrip,
                "guard": self.rejected_guard,
                "bound": self.bound_pruned,
                "nogood": self.nogood_skipped,
            },
            "shareable_classes": self.shareable_classes,
            "budget_exhausted": self.budget_exhausted,
            "best_moves": self.best_moves,
            "accepted_trace": self.accepted_trace,
            "check": self.check,
            "roundtrip": self.roundtrip,
        }
        if include_timing:
            out["wall_ms"] = self.wall_ms
        return out


def _r(x: float) -> float:
    return round(float(x), 6)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

Occ = tuple[str, int]   # (original instance id, original step index)


def _digest(obj: object, n: int = 10) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:n]


def call_path(instance_id: str) -> str:
    """Reusable-call path of an instance: its id up to the last ``/`` outside brackets
    (``caller[os=x]/callee[...]`` -> ``caller[os=x]``; ``""`` for top-level jobs)."""
    depth, last = 0, -1
    for i, ch in enumerate(instance_id):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "/" and depth == 0:
            last = i
    return instance_id[:last] if last >= 0 else ""


def _conjuncts(e: Expr) -> list[Expr]:
    if e.op == "and":
        return [x for a in e.args for x in _conjuncts(a)]
    if e == TRUE:
        return []
    return [e]


def default_split(cond: Expr) -> list[Expr] | None:
    """The conjuncts X of a default-fold condition ``success() && X`` (or of a condition
    without status functions); None if another status function occurs."""
    parts = _conjuncts(cond)
    rest = [x for x in parts if x != SUCCESS]
    if any(gexpr.uses_status_function(x) for x in rest):
        return None
    return rest


def _local_access(step: Step) -> tuple[frozenset[str], frozenset[str]]:
    """(job-local kinds read, written) by a step; ⊤ reads and writes every kind."""
    c = step.contract
    if c.top:
        return frozenset(LOCAL_KINDS), frozenset(LOCAL_KINDS)
    reads = frozenset(r.kind for r in c.reads if r.kind in LOCAL_KINDS)
    writes = frozenset(r.kind for r in c.writes if r.kind in LOCAL_KINDS)
    return reads, writes


def _step_refs(step: Step) -> set[str]:
    """Step ids a step references (condition and texts)."""
    refs = set(gexpr.step_refs(step.cond))
    b = step.binding
    for t in [b.text, *(v for _, v in b.with_), *(v for _, v in b.env)]:
        refs.update(_STEP_TEXT_REF.findall(t or ""))
    return refs


def _st_step_reads(step: Step) -> set[int]:
    return {int(r.key) for r in step.contract.reads
            if r.kind == ST_STEP and r.key is not None and str(r.key).isdigit()}


def _identity_key(key: str) -> bool:
    """Context keys whose value is tied to the instance (identity or matrix)."""
    if key == "platform" or key.startswith(("env.", "runner.")):
        return False
    if key.startswith("matrix"):
        return True
    return not sem.is_global_ctx(key)


def _needs_mentions(job: JobInstance) -> set[str]:
    """``needs.<x>`` targets mentioned by a job (condition, step conditions and texts, outputs)."""
    out = set(gexpr.needs_refs(job.cond))
    for _, e in job.outputs:
        out |= set(gexpr.needs_refs(e))
    for s in job.steps:
        out |= set(gexpr.needs_refs(s.cond))
        b = s.binding
        for t in [b.text, *(v for _, v in b.with_), *(v for _, v in b.env)]:
            out.update(m.group(1) for m in _NEEDS_TEXT_REF.finditer(t or ""))
    return out


def _persistent_writer(step: Step) -> bool:
    """Does the step write a persistent region (external state, a non-transfer artifact)?"""
    c = step.contract
    if c.top:
        return True
    return any(r.kind == EXT or (r.kind == RUN and not sem.is_transfer_artifact(r.key)) for r in c.writes)


def _status_gated(cond: Expr) -> bool:
    """Does the condition depend on earlier steps' statuses (success()/failure())?"""
    return _mentions_success_or_failure(cond)


def _group_key(iid: str) -> str:
    return iid[: iid.rfind("[")] if iid.endswith("]") and "[" in iid else iid


def _is_artifact_step(step: Step) -> bool:
    return not step.contract.top and step.contract.template in (sem.UPLOAD_TEMPLATE, sem.DOWNLOAD_TEMPLATE)


def instance_values(instance_id: str, matrix: Sequence[tuple[str, str]]) -> list[str]:
    """Every value that distinguishes an instance: the matrix values of the reusable-call
    path (``caller[k=v]/...``) followed by its own matrix values (sorted by key)."""
    out: list[str] = []
    path = call_path(instance_id)
    for seg in re.findall(r"\[([^\[\]]*)\]", path):
        out += [kv.split("=", 1)[1] for kv in seg.split(",") if "=" in kv]
    out += [str(v) for _, v in sorted(matrix)]
    return out


def artifact_name(class_id: str, producer: str, matrix: Sequence[tuple[str, str]],
                  instance_id: str = "") -> str:
    """``ghav-<hash>[-<value>...]``: the reserved prefix, a stable hash of the class and
    producer, and every instance-distinguishing value of the producing instance (sanitized;
    ``instance_values``)."""
    parts = [sem.TRANSFER_PREFIX + _digest([class_id, producer], 8)]
    for v in instance_values(instance_id, matrix):
        parts.append(_ARTIFACT_BAD.sub("_", str(v)).strip("_") or "_")
    name = "-".join(parts)
    if len(name) > MAX_ARTIFACT_NAME:  # the hash keeps truncated names unique
        name = name[:MAX_ARTIFACT_NAME].rstrip("-.")
    return name


# ---------------------------------------------------------------------------
# Read classes
# ---------------------------------------------------------------------------

def occurrence_classes(p: Program, *, read_values: bool = True) -> dict[Occ, tuple]:
    """Grouping key of every step occurrence of a non-frozen instance.

    With ``read_values`` the key is (task key, read-value terms when every task succeeds),
    the read-class definition shared with ``checker.read_class_accounts``; otherwise the
    task key alone.
    """
    by_id = {j.id: j for j in p.jobs}
    obs = None
    if read_values:
        ctx = sem.SymbolicContext.for_programs(p)
        obs = sem.evaluate(p, ctx, assume_success=True)
    keys: dict[Occ, tuple] = {}
    for j in sorted(p.jobs, key=lambda x: x.id):
        if j.frozen:
            continue
        for i, s in enumerate(j.steps):
            tk = sem.task_key(s, j, by_id)
            if obs is not None:
                reads = tuple((slot, term.get_id()) for slot, term in obs.step_reads[(j.id, i)])
                keys[(j.id, i)] = (repr(tk), reads)
            else:
                keys[(j.id, i)] = (repr(tk),)
    return keys


@dataclass(frozen=True)
class ReadClass:
    id: str
    members: tuple[Occ, ...]        # one occurrence per instance, sorted
    template: str
    transferable: bool
    out_paths: tuple[str, ...]
    label: str


# ---------------------------------------------------------------------------
# Analysis of the original program
# ---------------------------------------------------------------------------

class Analysis:
    """Everything the search needs about the original program, computed once."""

    def __init__(self, p: Program, cfg: SynthConfig, model: costmod.CostModel,
                 contexts: Sequence[tuple[str, float]], repo_root: Path | None) -> None:
        self.p = p
        self.cfg = cfg
        self.model = model
        self.contexts = tuple(contexts)
        self.repo_root = Path(repo_root) if repo_root is not None else Path(".")
        self.by_id = {j.id: j for j in p.jobs}
        self.ids = tuple(sorted(self.by_id))
        self.movable = tuple(i for i in self.ids if not self.by_id[i].frozen)
        self.orig_issues = sem.well_formed(p)
        self._transfer_contracts: dict[tuple, Contract] = {}

        # classes
        keys = occurrence_classes(p, read_values=cfg.read_classes)
        groups: dict[tuple, list[Occ]] = defaultdict(list)
        for occ in sorted(keys):
            groups[keys[occ]].append(occ)
        self.class_of: dict[Occ, str] = {}
        self.class_members: dict[str, tuple[Occ, ...]] = {}
        for _, occs in sorted(groups.items(), key=lambda kv: kv[1][0]):
            cid = "c" + _digest([list(o) for o in occs])
            self.class_members[cid] = tuple(occs)
            for o in occs:
                self.class_of[o] = cid

        # execution weights and prices (all tasks succeed)
        self.weight: dict[Occ, float] = defaultdict(float)
        self.job_weight: dict[str, float] = defaultdict(float)
        self.seconds: dict[Occ, float] = {}
        for event, w in self.contexts:
            runs = costmod.simulate(p, event, model)
            for jid in self.ids:
                run = runs[jid]
                job = self.by_id[jid]
                if run.ran:
                    self.job_weight[jid] += w
                for i, s in enumerate(job.steps):
                    if i < len(run.steps) and run.steps[i]:
                        self.weight[(jid, i)] += w
        for jid in self.ids:
            job = self.by_id[jid]
            for i, s in enumerate(job.steps):
                self.seconds[(jid, i)] = self._price(job, s)
        self.overhead = {jid: float(model.overhead(self.by_id[jid])) if hasattr(model, "overhead")
                         else float(model.job_overhead_seconds) for jid in self.ids}
        self.upload_seconds = costmod.template_seconds("upload-artifact", model)
        self.download_seconds = costmod.template_seconds("download-artifact", model)

        # Original reaching definitions of job-local regions and step references. Job-local
        # writes are updates (``semantics``: a task that writes a kind without reading it
        # changes part of it), so the value a step reads is determined by a chain of writers:
        # the last writer that also read the kind, then every later writer (``chain_step``).
        self.local_reads: dict[Occ, frozenset[str]] = {}
        self.local_writes: dict[Occ, frozenset[str]] = {}
        self.chain_before: dict[Occ, dict[str, tuple[Occ, ...]]] = {}
        self.ref_targets: dict[Occ, dict[str, Occ | None]] = {}
        for jid in self.movable:
            job = self.by_id[jid]
            cur: dict[str, tuple[Occ, ...]] = {k: () for k in LOCAL_KINDS}
            ids_so_far: dict[str, Occ] = {}
            for i, s in enumerate(job.steps):
                occ = (jid, i)
                r, w = _local_access(s)
                self.local_reads[occ] = r
                self.local_writes[occ] = w
                self.chain_before[occ] = dict(cur)
                targets: dict[str, Occ | None] = {sid: ids_so_far.get(sid) for sid in sorted(_step_refs(s))}
                for idx in sorted(_st_step_reads(s)):
                    if idx < i:
                        targets[f"#{idx}"] = (jid, idx)
                self.ref_targets[occ] = targets
                cur = chain_step(cur, occ, r, w)
                if s.binding.step_id:
                    ids_so_far[s.binding.step_id] = occ

        # shareable classes
        self.shareable: dict[str, ReadClass] = {}
        for cid, occs in sorted(self.class_members.items()):
            first: dict[str, Occ] = {}
            for o in occs:
                first.setdefault(o[0], o)
            members = tuple(sorted(first.values()))
            if len(members) < 2:
                continue
            j0, i0 = members[0]
            s0 = self.by_id[j0].steps[i0]
            if _is_artifact_step(s0):
                continue
            if any(default_split(self.by_id[j].steps[i].cond) is None for j, i in members):
                continue
            c = s0.contract
            transferable = (not c.top) and any(r.kind == OUT for r in c.writes) and bool(c.out_paths)
            self.shareable[cid] = ReadClass(cid, members, "top" if c.top else c.template, transferable,
                                            tuple(c.out_paths), sem.binding_label(s0.binding))
        if cfg.moves == "rules":
            self.shareable = {cid: rc for cid, rc in self.shareable.items() if self._rule_premise(rc)}
        # Largest saving a class can give (it runs once): used by the cost lower bound. Every
        # class counts, shareable or not, because merges also deduplicate common prefixes.
        self.class_saving: dict[str, float] = {}
        for cid, occs in sorted(self.class_members.items()):
            vals = sorted(self.occ_saving(o) for o in occs)
            if len(vals) >= 2:
                self.class_saving[cid] = sum(vals[1:])
        self.total_class_saving = sum(self.class_saving[c] for c in sorted(self.class_saving))

        # fail-fast groups that may be transformed
        self.ff_ok: dict[str, bool] = {}
        for g in sorted({j.fail_fast_group for j in p.jobs if j.fail_fast_group}):
            self.ff_ok[g] = self._ff_transformable(g)

        # Control dependencies: which occurrences gate (through success()/failure()) a later
        # persistent write of their instance, and which instances have dependents that write
        # persistent state (a failing step skips them).
        self.gates_persistent: dict[Occ, bool] = {}
        for jid in self.movable:
            steps = self.by_id[jid].steps
            later = False
            for i in range(len(steps) - 1, -1, -1):
                self.gates_persistent[(jid, i)] = later
                s = steps[i]
                if _persistent_writer(s) and _status_gated(s.cond):
                    later = True
        writes_p = {jid: self.by_id[jid].frozen or any(_persistent_writer(s) for s in self.by_id[jid].steps)
                    for jid in self.ids}
        self.ancestors = sem.ancestors(p)
        # Dependents that write persistent state and are skipped when an ancestor fails
        # (job-level success()/failure() range over every transitive ancestor).
        gated = {d: writes_p[d] and (_status_gated(self.by_id[d].cond) or bool(_needs_mentions(self.by_id[d])))
                 for d in self.ids}
        self.gated_dependents: dict[str, tuple[str, ...]] = {
            jid: tuple(sorted(d for d in self.ids if gated[d] and jid in self.ancestors.get(d, frozenset())))
            for jid in self.ids}

        self.mentions = {jid: _needs_mentions(self.by_id[jid]) for jid in self.ids}
        self.dependents: dict[str, tuple[str, ...]] = {
            jid: tuple(sorted(d for d in self.ids if jid in self.by_id[d].needs)) for jid in self.ids}
        self.identity_bound = {jid: self._identity_bound(jid) for jid in self.movable}
        self._slices: dict[Occ, tuple[int, ...]] = {}

    # prices -----------------------------------------------------------------
    def _price(self, job: JobInstance, step: Step) -> float:
        if hasattr(self.model, "price_step"):
            return float(self.model.price_step(job, step)[0])
        return float(costmod.step_seconds(step, self.model))

    def occ_saving(self, occ: Occ) -> float:
        return self.seconds[occ] * self.weight[occ]

    # structure --------------------------------------------------------------
    def step(self, occ: Occ) -> Step:
        return self.by_id[occ[0]].steps[occ[1]]

    def token(self, occ: Occ) -> str:
        return self.class_of[occ]

    def slice_of(self, occ: Occ) -> tuple[int, ...]:
        """Indices of the earlier steps of the occurrence's instance that it transitively
        depends on through job-local regions or step references (its backward slice)."""
        hit = self._slices.get(occ)
        if hit is not None:
            return hit
        jid, k = occ
        need: set[int] = set()
        todo = [k]
        while todo:
            t = todo.pop()
            o = (jid, t)
            chains = self.chain_before[o]
            deps = [w for k in sorted(self.local_reads[o]) for w in chains[k]]
            deps += [w for w in self.ref_targets[o].values() if w is not None]
            for w in deps:
                if w[1] not in need:
                    need.add(w[1])
                    todo.append(w[1])
        out = tuple(sorted(need))
        self._slices[occ] = out
        return out

    def tokens(self, chain: Sequence[Occ]) -> tuple[str, ...]:
        return tuple(self.token(o) for o in chain)

    def chain_after(self, occs: Sequence[Occ], kind: str) -> tuple[Occ, ...]:
        """Writer chain of ``kind`` after running the original occurrences ``occs`` (in order)
        in a fresh job."""
        cur: dict[str, tuple[Occ, ...]] = {k: () for k in LOCAL_KINDS}
        for o in occs:
            cur = chain_step(cur, o, self.local_reads[o], self.local_writes[o])
        return cur[kind]

    def producer_out(self, rc: ReadClass, enc: str) -> tuple[str, ...]:
        """Tokens of the ``out`` writer chain an exact transfer of ``rc`` carries: the
        producer's ``out`` right after its occurrence of the class (its original instance
        prefix for an instance producer; the backward slice for a new producer job)."""
        if enc == "job":
            rep = self.representative(rc)
            occs = [(rep[0], i) for i in self.slice_of(rep)] + [rep]
        else:
            rep = next(o for o in rc.members if o[0] == enc)
            occs = [(rep[0], i) for i in range(rep[1] + 1)]
        return self.tokens(self.chain_after(occs, OUT))

    def representative(self, rc: ReadClass) -> Occ:
        """Member whose instance runs most often (then the least occurrence)."""
        return min(rc.members, key=lambda o: (-round(self.job_weight[o[0]], 9), o))

    def _identity_bound(self, jid: str) -> bool:
        """Do the instance's steps depend on its identity (⊤ steps, or reads of
        ``job-identity``, matrix values, or other per-instance context keys)?"""
        for s in self.by_id[jid].steps:
            if s.contract.top:
                return True
            if any(r.kind == CTX and r.key and _identity_key(r.key) for r in s.contract.reads):
                return True
        return False

    def _rule_premise(self, rc: ReadClass) -> bool:
        """Whole-prefix hoist of a single matrix job: the class covers every non-frozen
        instance of one original job (at least two), at the same index, and every earlier
        step is shared by all of them too."""
        insts = [o[0] for o in rc.members]
        jobs = {self.by_id[i].orig_job for i in insts}
        if len(jobs) != 1:
            return False
        orig = next(iter(jobs))
        all_insts = sorted(j.id for j in self.p.jobs if j.orig_job == orig and not j.frozen)
        if sorted(insts) != all_insts or any(j.frozen for j in self.p.jobs if j.orig_job == orig):
            return False
        idxs = {o[1] for o in rc.members}
        if len(idxs) != 1:
            return False
        k = next(iter(idxs))
        for t in range(k):
            toks = {self.class_of.get((i, t)) for i in insts}
            if len(toks) != 1 or None in toks:
                return False
        return True

    def _ff_transformable(self, group: str) -> bool:
        """RESEARCH_SCOPE fail-fast rule (i)-(iii) for one fail-fast group."""
        members = [j for j in self.p.jobs if j.fail_fast_group == group]
        ids = {j.id for j in members}
        if any(call_path(j.id) for j in members):          # (iii) calls a reusable workflow
            return False
        read_elsewhere = {r.key for j in self.p.jobs if j.id not in ids for s in j.steps
                          for r in s.contract.reads if r.kind == RUN}
        for j in members:                                 # (i)
            for s in j.steps:
                if default_split(s.cond) is not None:
                    continue
                c = s.contract
                if c.top:
                    return False
                for r in c.writes:
                    if r.kind == EXT or (r.kind == RUN and (not sem.is_transfer_artifact(r.key)
                                                            or r.key in read_elsewhere)):
                        return False
        names = ids | {j.orig_job for j in members} | {_group_key(j.id) for j in members}
        for d in self.p.jobs:                             # (ii)
            if d.id in ids or not (d.needs & ids):
                continue
            if default_split(d.cond) is None:
                return False
            if _needs_mentions(d) & names:
                return False
        return True

    # transfers --------------------------------------------------------------
    def transfer_steps(self, kind: str, name: str, paths: Sequence[str], cond: Expr,
                       origin: tuple[str, int], origin_instance: str, job: JobInstance) -> list[Step]:
        up, down = transfer_idiom(name, paths)
        items = up if kind == "upload" else down
        steps: list[Step] = []
        for binding, contract in items:
            if contract is None:
                contract = self._contract_of(binding, job)
            steps.append(Step(binding=binding, cond=cond, contract=contract, origin=origin, raw=None,
                              origin_instance=origin_instance))
        return steps

    def _contract_of(self, binding: Binding, job: JobInstance) -> Contract:
        key = (binding, job.runs_on)
        hit = self._transfer_contracts.get(key)
        if hit is None:
            from ghav import contracts  # local import: contracts is heavy and only needed here
            probe = Step(binding=binding, cond=SUCCESS, contract=TOP_CONTRACT, origin=("ghav", -1))
            try:
                hit = contracts.infer(probe, job, self.repo_root, workflow_env=self.p.env,
                                      workflow_defaults=self.p.defaults)
            except Exception:  # noqa: BLE001 - an unanalyzable transfer is ⊤ (conservative)
                hit = TOP_CONTRACT
            self._transfer_contracts[key] = hit
        return hit


def _norm_paths(paths: Sequence[str]) -> list[str]:
    out = []
    for x in paths:
        t = str(x).strip()
        if t.startswith("./"):
            t = t[2:]
        t = t.rstrip("/") or "."
        out.append(t)
    return sorted(set(out))


def transfer_idiom(name: str, paths: Sequence[str]) -> tuple[list[tuple[Binding, Contract | None]],
                                                             list[tuple[Binding, Contract | None]]]:
    """(producer side, consumer side) of one transfer of ``paths`` named ``name``: the idiom's
    (pack, upload) and (download, unpack) bindings from ``contracts.transfer_bindings``, the
    single definition of the idiom that ``emit.transfer_steps`` uses as well (contract guide
    9). Contracts are inferred per job by the caller. Raises Incomplete when the idiom rejects
    the name or the paths."""
    from ghav import contracts
    try:
        pack, upload, download, unpack = contracts.transfer_bindings(name, _norm_paths(paths))
    except ValueError as exc:
        raise Incomplete("transfer-idiom", str(exc)) from exc
    return [(pack, None), (upload, None)], [(download, None), (unpack, None)]


def is_synth_transfer(step: Step) -> bool:
    """A transfer step added by completion (its origin index is below every original one)."""
    return step.origin[1] <= -TRANSFER_ORIGIN_OFFSET


# ---------------------------------------------------------------------------
# Sketches and moves
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sketch:
    shares: tuple[tuple[str, str], ...] = ()          # sorted (class id, producer: "job" | instance id)
    merges: tuple[tuple[str, ...], ...] = ()          # sorted groups of >= 2 sorted instance ids

    @property
    def key(self) -> str:
        return json.dumps([[list(s) for s in self.shares], [list(g) for g in self.merges]],
                          separators=(",", ":"))

    def shared_classes(self) -> frozenset[str]:
        return frozenset(c for c, _ in self.shares)

    def with_share(self, cid: str, producer: str) -> "Sketch":
        return Sketch(tuple(sorted(self.shares + ((cid, producer),))), self.merges)

    def with_merge(self, a: Sequence[str], b: Sequence[str]) -> "Sketch":
        union = tuple(sorted(set(a) | set(b)))
        rest = [g for g in self.merges if not (set(g) & set(union))]
        return Sketch(self.shares, tuple(sorted(rest + [union])))

    def covers(self, other: "Sketch") -> bool:
        """Does this sketch contain every move of ``other``?"""
        if not set(other.shares) <= set(self.shares):
            return False
        return all(any(set(g) <= set(h) for h in self.merges) for g in other.merges)

    def size(self) -> int:
        return len(self.shares) + sum(len(g) - 1 for g in self.merges)


class NogoodIndex:
    """Sketches whose supersets are skipped, indexed so that ``covers`` only inspects
    nogoods that can apply: by their least share, or (share-free ones) by the two least
    members of their least merge group."""

    def __init__(self) -> None:
        self.by_share: dict[tuple[str, str], list[Sketch]] = defaultdict(list)
        self.by_pair: dict[tuple[str, str], list[Sketch]] = defaultdict(list)
        self.count = 0

    def add(self, sk: Sketch) -> None:
        if sk.shares:
            self.by_share[sk.shares[0]].append(sk)
        elif sk.merges:
            g = sk.merges[0]
            self.by_pair[(g[0], g[1])].append(sk)
        else:
            return
        self.count += 1

    def covers(self, sk: Sketch) -> bool:
        """Does ``sk`` contain every move of some nogood?"""
        for sh in sk.shares:
            if any(sk.covers(n) for n in self.by_share.get(sh, ())):
                return True
        for g in sk.merges:
            for x in range(len(g)):
                for y in range(x + 1, len(g)):
                    if any(sk.covers(n) for n in self.by_pair.get((g[x], g[y]), ())):
                        return True
        return False


def groups_of(an: Analysis, sk: Sketch) -> list[tuple[str, ...]]:
    """Current job groups of original non-frozen instances (merged groups and singletons)."""
    merged = {i for g in sk.merges for i in g}
    out = [tuple(g) for g in sk.merges] + [(i,) for i in an.movable if i not in merged]
    return sorted(out)


def merge_hard_ok(an: Analysis, group: Sequence[str]) -> str | None:
    """Hard (completion) constraints of a merge group; returns a reason or None."""
    jobs = [an.by_id[i] for i in group]
    first = jobs[0]
    for j in jobs[1:]:
        if call_path(j.id) != call_path(first.id):
            return "merge-call-path"
        if j.cond != first.cond:
            return "merge-condition"
        if j.needs != first.needs:
            return "merge-needs"
        if j.permissions != first.permissions:
            return "merge-permissions"
    env: dict[str, str] = {}
    for j in jobs:
        for k, v in j.env:
            if env.setdefault(k, v) != v:
                return "merge-env"
    return None


def choose_host(an: Analysis, group: Sequence[str]) -> str:
    return min(group, key=lambda i: (not an.identity_bound[i], i))


def merge_guest_ok(an: Analysis, group: Sequence[str], host: str) -> str | None:
    for g in group:
        if g == host:
            continue
        job = an.by_id[g]
        if job.outputs:
            return "merge-guest-outputs"
        names = {g, _group_key(g)}
        if sum(1 for j in an.p.jobs if j.orig_job == job.orig_job) == 1:
            names.add(job.orig_job)
        for d in an.dependents[g]:
            if an.mentions[d] & names:
                return "merge-guest-referenced"
    return None


def merge_settings_ok(an: Analysis, group: Sequence[str]) -> bool:
    """Cheap settings compatibility (the ``settings`` pruning reason, before completion):
    one runs-on, one fail-fast group, at most one member bound to its identity."""
    jobs = [an.by_id[i] for i in group]
    if len({j.runs_on for j in jobs}) != 1:
        return False
    if len({j.fail_fast_group for j in jobs}) != 1:
        return False
    return sum(1 for j in jobs if an.identity_bound[j.id]) <= 1


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

class Incomplete(Exception):
    """The sketch has no completion (a hard constraint fails); ``reason`` names it."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass
class _Slot:
    kind: str                     # "orig" | "down" | "up"
    occ: Occ                      # the occurrence (or the one the transfer stands for)
    share: str | None = None      # class id for transfers
    owners: set[str] = field(default_factory=set)
    steps: list[Step] = field(default_factory=list)   # transfer steps (one or more)

    def token(self, an: Analysis) -> tuple:
        if self.kind == "orig":
            return ("orig", an.token(self.occ), repr(an.step(self.occ).cond))
        return (self.kind, self.share or "", repr(an.step(self.occ).cond))


@dataclass
class Completion:
    program: Program
    slots: dict[str, list[_Slot]]           # final job id -> slots (non-frozen changed jobs only)
    producers: dict[str, str]               # class id -> producer job id
    artifacts: dict[str, str]               # class id -> artifact name
    hosts: dict[str, tuple[str, ...]]       # final job id -> original member instances
    changed: frozenset[str]                 # original instances whose job changed
    dropped: tuple[tuple[Occ, str], ...] = ()   # (displaced occurrence, producer job) with no transfer
    encodings: tuple[tuple[str, str], ...] = ()  # (class id, "job" | producer instance) = Sketch.shares


def complete(an: Analysis, sk: Sketch) -> Completion:
    """The deterministic completion of a sketch (raises Incomplete)."""
    p = an.p
    slots: dict[str, list[_Slot]] = {
        jid: [_Slot("orig", (jid, i), owners={jid}) for i in range(len(an.by_id[jid].steps))]
        for jid in an.movable}
    extra_needs: dict[str, set[str]] = defaultdict(set)
    producers: dict[str, str] = {}
    artifacts: dict[str, str] = {}
    new_jobs: list[JobInstance] = []
    changed: set[str] = set()
    dropped: list[tuple[Occ, str]] = []

    for cid, enc in sk.shares:
        rc = an.shareable.get(cid)
        if rc is None:
            raise Incomplete("unknown-class", cid)
        if enc == "job":
            rep = an.representative(rc)
            pid = PRODUCER_PREFIX + _digest([cid, "job"], 8)
            prod_matrix = an.by_id[rep[0]].matrix
            consumers = list(rc.members)
        else:
            if enc not in {o[0] for o in rc.members}:
                raise Incomplete("unknown-producer", enc)
            rep = next(o for o in rc.members if o[0] == enc)
            pid = enc
            prod_matrix = an.by_id[enc].matrix
            consumers = [o for o in rc.members if o != rep]
        art = artifact_name(cid, enc, prod_matrix, rep[0])
        producers[cid] = pid
        artifacts[cid] = art
        rep_step = an.step(rep)
        up_origin = (rep_step.origin[0], -(TRANSFER_ORIGIN_OFFSET + rep[1]))
        if enc == "job":
            new_jobs.append(_producer_job(an, rc, rep, pid, art, up_origin))
        elif rc.transferable:
            sl = slots[rep[0]]
            pos = next(k for k, s in enumerate(sl) if s.kind == "orig" and s.occ == rep)
            up = _Slot("up", rep, cid, owners={rep[0]})
            up.steps = an.transfer_steps("upload", art, rc.out_paths, rep_step.cond, up_origin, rep[0],
                                         an.by_id[rep[0]])
            sl.insert(pos + 1, up)
            changed.add(rep[0])
        for occ in consumers:
            sl = slots[occ[0]]
            pos = next(k for k, s in enumerate(sl) if s.kind == "orig" and s.occ == occ)
            changed.add(occ[0])
            if rc.transferable:
                st = an.step(occ)
                down = _Slot("down", occ, cid, owners={occ[0]})
                down.steps = an.transfer_steps(
                    "download", art, rc.out_paths, st.cond,
                    (st.origin[0], -(TRANSFER_ORIGIN_OFFSET + occ[1])), occ[0], an.by_id[occ[0]])
                sl[pos] = down
                if pid != occ[0]:
                    extra_needs[occ[0]].add(pid)
            else:
                del sl[pos]
                dropped.append((occ, pid))

    # merges -------------------------------------------------------------------
    remap: dict[str, str] = {}
    members_of: dict[str, tuple[str, ...]] = {}
    fused: dict[str, list[_Slot]] = {}
    for group in sk.merges:
        reason = merge_hard_ok(an, group)
        if reason:
            raise Incomplete(reason, ",".join(group))
        host = choose_host(an, group)
        reason = merge_guest_ok(an, group, host)
        if reason:
            raise Incomplete(reason, ",".join(group))
        order = [host] + [g for g in group if g != host]
        acc = list(slots[host])
        for guest in order[1:]:
            gs = slots[guest]
            n = 0
            while n < len(gs) and n < len(acc) and acc[n].token(an) == gs[n].token(an):
                acc[n].owners.add(guest)
                n += 1
            acc.extend(gs[n:])
            remap[guest] = host
        fused[host] = acc
        members_of[host] = tuple(order)
        changed.update(group)

    # jobs -----------------------------------------------------------------------
    def rn(n: str) -> str:
        return remap.get(n, n)

    final_slots: dict[str, list[_Slot]] = {}
    jobs: list[JobInstance] = []
    for jid in an.ids:
        job = an.by_id[jid]
        if jid in remap:
            continue
        members = members_of.get(jid, (jid,))
        orig_needs = frozenset(rn(n) for m in members for n in an.by_id[m].needs) - {jid}
        new_needs = set(orig_needs)
        for m in members:
            new_needs |= {rn(n) for n in extra_needs.get(m, ())}
        new_needs.discard(jid)
        new_needs = frozenset(new_needs)
        if job.frozen or (jid not in changed and new_needs == job.needs):
            if new_needs != job.needs:
                job = replace(job, needs=new_needs)
            jobs.append(job)
            continue
        sl = fused.get(jid, slots[jid])
        steps = _assign_conditions(an, jid, sl)
        cond = job.cond
        if new_needs != orig_needs:
            cond = _exact_job_cond(job.cond, orig_needs)
        env = dict(job.env)
        for m in members:
            env.update(dict(an.by_id[m].env))
        jobs.append(replace(job, steps=tuple(steps), needs=new_needs, cond=cond,
                            env=tuple(sorted(env.items()))))
        final_slots[jid] = sl
    for pj in new_jobs:
        jobs.append(replace(pj, needs=frozenset(rn(n) for n in pj.needs)))
    prog = p.with_jobs(jobs)
    try:
        sem.topological_order(prog)
    except ValueError as exc:
        raise Incomplete("cycle", str(exc)) from exc
    hosts = {h: m for h, m in members_of.items()}
    return Completion(prog, final_slots, producers, artifacts, hosts, frozenset(changed),
                      tuple(sorted(dropped)), tuple(sk.shares))


def _exact_job_cond(cond: Expr, orig_needs: Iterable[str]) -> Expr:
    """Original job condition with ``success()`` over the ORIGINAL needs only."""
    rest = default_split(cond)
    if rest is None:
        if gexpr.uses_status_function(cond) and _mentions_success_or_failure(cond):
            raise Incomplete("inexpressible-condition", gexpr.unparse(cond))
        return cond
    parts = list(rest)
    if SUCCESS in _conjuncts(cond):
        parts += [Expr("eq", (Expr("needs", (n, "result")), Expr("lit", ("success",))))
                  for n in sorted(orig_needs)]
    return gexpr.conj(parts) if parts else TRUE


def _mentions_success_or_failure(e: Expr) -> bool:
    if e.op == "status" and e.args and str(e.args[0]).lower() in ("success", "failure"):
        return True
    return any(_mentions_success_or_failure(a) for a in e.args if isinstance(a, Expr))


def _assign_conditions(an: Analysis, jid: str, sl: list[_Slot]) -> list[Step]:
    """Materialize the slots of one job with exact-form conditions."""
    # Which slots need a rewritten condition, and which predecessors they reference.
    needs_rewrite: dict[int, list[int]] = {}
    for k, s in enumerate(sl):
        preds = [t for t in range(k) if sl[t].owners & s.owners]
        if len(preds) != k:
            needs_rewrite[k] = preds
    referenced = sorted({t for preds in needs_rewrite.values() for t in preds})

    # Step ids for referenced predecessors (the last step of a multi-step transfer).
    materialized: list[list[Step]] = []
    for s in sl:
        materialized.append(list(s.steps) if s.kind != "orig" else [an.step(s.occ)])
    used = Counter(st.binding.step_id for sts in materialized for st in sts if st.binding.step_id)
    ref_id: dict[int, str] = {}
    for t in referenced:
        st = materialized[t][-1]
        sid = st.binding.step_id
        if sid and used[sid] > 1:
            raise Incomplete("ambiguous-step-id", sid)
        if not sid:
            n, sid = t, f"{STEP_ID_PREFIX}{t}"
            while sid in used:
                n += 1000
                sid = f"{STEP_ID_PREFIX}{n}"
            used[sid] += 1
            materialized[t][-1] = replace(st, binding=replace(st.binding, step_id=sid))
        ref_id[t] = sid

    out: list[Step] = []
    for k, s in enumerate(sl):
        sts = materialized[k]
        if k in needs_rewrite:
            base = sts[0].cond
            rest = default_split(base)
            if rest is None:
                if _mentions_success_or_failure(base):
                    raise Incomplete("inexpressible-condition", gexpr.unparse(base))
                new = base
            else:
                preds = [Expr("ne", (Expr("step", (ref_id[t], "outcome")), Expr("lit", ("failure",))))
                         for t in needs_rewrite[k]]
                new = gexpr.conj([NOT_CANCELLED] + list(rest) + preds)
            sts = [replace(st, cond=new) for st in sts]
        if s.kind == "orig":
            st = sts[0]
            if st.origin_instance is None:
                st = replace(st, origin_instance=s.occ[0])
            sts = [st]
        out.extend(sts)
    return out


def _producer_job(an: Analysis, rc: ReadClass, rep: Occ, pid: str, art: str,
                  up_origin: tuple[str, int]) -> JobInstance:
    """A NEW job running the backward slice of ``rep``, the step, and the upload."""
    from ghav.lower import default_check_name
    src = an.by_id[rep[0]]
    idxs = list(an.slice_of(rep)) + [rep[1]]
    steps = [replace(src.steps[i], origin_instance=rep[0]) for i in idxs]
    rep_step = src.steps[rep[1]]
    job = replace(src, id=pid, orig_job=pid, check_name=pid, outputs=(), fail_fast_group=None,
                  frozen=False, frozen_reasons=(), raw=None, steps=())
    if rc.transferable:
        steps += an.transfer_steps("upload", art, rc.out_paths, rep_step.cond, up_origin, rep[0], job)
    combo = {k: v for k, v in src.matrix}
    check = default_check_name(pid, combo, sorted(combo)) if combo else pid
    return replace(job, check_name=check, steps=tuple(steps))


# ---------------------------------------------------------------------------
# Effect pruning
# ---------------------------------------------------------------------------

def _ctx_value(p: Program, job: JobInstance, key: str) -> object:
    """A descriptor of the value of a context key in ``job`` (for comparing jobs)."""
    if key == "platform":
        return ("platform", job.runs_on)
    if key in ("job-identity",) or key == "github.job" or key.startswith("github.action"):
        return ("identity", sem.instance_identity(job))
    if key.startswith("matrix"):
        m = dict(job.matrix)
        if key in ("matrix", "matrix.*"):
            return ("matrix", tuple(sorted(m.items())))
        return ("matrix", m.get(key[len("matrix."):]))
    if key.startswith("env."):
        env = dict(p.env)
        env.update(dict(job.env))
        name = key[len("env."):]
        return ("env", env.get(name), job.runs_on if name not in env else None)
    if key.startswith("runner."):
        return ("runner", job.runs_on)
    if sem.is_global_ctx(key):
        return ("global",)
    return ("identity", sem.instance_identity(job))


def prune(an: Analysis, comp: Completion) -> str | None:
    """The first effect-pruning reason that rejects the completion, or None."""
    cand = comp.program
    cbid = {j.id: j for j in cand.jobs}
    universe = sem.Universe.of([an.p])
    # 2. settings: every kept step must see the context values of its original instance.
    placed: list[tuple[JobInstance, Occ]] = []
    for jid, sl in sorted(comp.slots.items()):
        placed += [(cbid[jid], s.occ) for s in sl if s.kind == "orig"]
    for pid in sorted(set(comp.producers.values())):
        if pid in an.by_id or pid not in cbid:
            continue
        job = cbid[pid]
        placed += [(job, (st.origin_instance, st.origin[1])) for st in job.steps
                   if not is_synth_transfer(st) and st.origin_instance in an.by_id]
    for job, occ in placed:
        orig = an.by_id[occ[0]]
        st = an.step(occ)
        if st.contract.top:
            keys = sem.instance_ctx_keys(orig, an.p, universe)
        else:
            keys = tuple(sorted({r.key for r in st.contract.reads if r.kind == CTX and r.key}))
        for key in keys:
            if _ctx_value(an.p, job, key) != _ctx_value(an.p, orig, key):
                return "settings"
    # 1b. status cells: a dropped occurrence (no transfer stands in for it) whose failure used
    # to skip a later persistent write of its instance, or dependents that write persistent
    # state, changes when those writes happen.
    for occ, pid in comp.dropped:
        if an.gates_persistent.get(occ):
            return "status-dependency"
        # A dependent that the producer's failure skips as well keeps its gate.
        for d in an.gated_dependents.get(occ[0], ()):
            if d != pid and pid not in an.ancestors.get(d, frozenset()):
                return "status-dependency"
    # 1. untransferred job-local reads.
    encodings = dict(comp.encodings)
    for jid, sl in sorted(comp.slots.items()):
        if _untransferred(an, sl, encodings):
            return "untransferred-read"
    plan = sem.artifact_plan(cand)
    for cid, art in sorted(comp.artifacts.items()):
        if an.shareable[cid].transferable and plan.get(art) is None:
            return "untransferred-read"
    # 4. fail-fast rule.
    for jid in sorted(comp.changed):
        g = an.by_id[jid].fail_fast_group
        if g and not an.ff_ok.get(g, False):
            return "fail-fast"
    # 3. well-formedness.
    if sem.blocking_issues(an.orig_issues, sem.well_formed(cand)):
        return "well-formed"
    return None


def chain_step(cur: Mapping[str, tuple[Occ, ...]], occ: Occ, reads: frozenset[str],
               writes: frozenset[str]) -> dict[str, tuple[Occ, ...]]:
    """Writer chains after ``occ``: a written kind it also reads is determined by ``occ``
    alone (its read values are part of its read class); a written kind it does not read is
    updated, so ``occ`` is appended to the chain."""
    out = dict(cur)
    for k in sorted(writes):
        out[k] = (occ,) if k in reads else tuple(cur[k]) + (occ,)
    return out


def _untransferred(an: Analysis, sl: list[_Slot], producers: Mapping[str, str] | None = None) -> bool:
    """Does a kept step read a job-local value (a writer chain, compared by read class) or a
    step reference that differs from the original? ``producers`` maps class ids to their
    encodings (``Sketch.shares``), used for what an exact transfer carries."""
    state: dict[str, tuple[str, ...]] = {k: () for k in LOCAL_KINDS}
    ids: dict[str, str] = {}
    for s in sl:
        if s.kind == "orig":
            occ = s.occ
            for kind in sorted(an.local_reads[occ]):
                if state[kind] != an.tokens(an.chain_before[occ][kind]):
                    return True
            for ref, w in sorted(an.ref_targets[occ].items()):
                if ref.startswith("#"):
                    continue
                want = an.token(w) if w is not None else None
                if ids.get(ref) != want:
                    return True
            reads = an.local_reads[occ]
            for kind in sorted(an.local_writes[occ]):
                state[kind] = (an.token(occ),) if kind in reads else state[kind] + (an.token(occ),)
            sid = an.step(occ).binding.step_id
            if sid:
                ids[sid] = an.token(occ)
        elif s.kind == "down":
            # An exact transfer (``semantics.transfer_plan``) replaces the consumer's (clean)
            # out by the producer's out right after its occurrence of the class; the idiom's
            # archive is private to GHAV, so nothing else changes.
            enc = (producers or {}).get(s.share or "")
            rc = an.shareable.get(s.share or "")
            state[OUT] = an.producer_out(rc, enc) if rc is not None and enc is not None \
                else (an.token(s.occ),)
    return False


def fail_fast_guard(an: Analysis, comp: Completion) -> bool:
    return all(not an.by_id[j].fail_fast_group or an.ff_ok.get(an.by_id[j].fail_fast_group, False)
               for j in comp.changed)


# ---------------------------------------------------------------------------
# Costs, bounds, estimates
# ---------------------------------------------------------------------------

def program_cost(an: Analysis, prog: Program) -> float:
    return float(costmod.expected_machine_time(prog, an.contexts, an.model))


def billed_minutes(an: Analysis, prog: Program) -> float:
    total = 0.0
    for event, w in an.contexts:
        runs = costmod.simulate(prog, event, an.model)
        total += w * sum(costmod.billed_minutes_for(runs[j].seconds, runs[j].ran) for j in sorted(runs))
    return total


def _merge_bucket(an: Analysis, jid: str) -> tuple:
    j = an.by_id[jid]
    return (call_path(jid), repr(j.cond), tuple(sorted(j.needs)), repr(j.permissions), j.runs_on)


def lower_bound(an: Analysis, sk: Sketch, c: float) -> float:
    """Cost minus the largest saving further moves can realize: every unshared class runs
    once, and every merge-compatible bucket of jobs pays one overhead."""
    saving = an.total_class_saving - sum(an.class_saving.get(cid, 0.0) for cid in sorted(sk.shared_classes()))
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for g in groups_of(an, sk):
        host = g[0]
        buckets[_merge_bucket(an, host)].append(an.overhead[host] * an.job_weight[host])
    for _, ws in sorted(buckets.items()):
        ws.sort()
        saving += sum(ws[1:])
    return c - saving


def share_estimate(an: Analysis, rc: ReadClass, enc: str) -> float:
    """Estimated saving of ``share(rc, enc)`` relative to the unshared program."""
    if enc == "job":
        rep = an.representative(rc)
        consumers = list(rc.members)
        w = an.job_weight[rep[0]]
        added = an.overhead[rep[0]] * w + sum(an.seconds[(rep[0], i)] for i in an.slice_of(rep)) * w \
            + an.seconds[rep] * w + (an.upload_seconds * w if rc.transferable else 0.0)
    else:
        consumers = [o for o in rc.members if o[0] != enc]
        added = an.upload_seconds * an.job_weight[enc] if rc.transferable else 0.0
    saved = sum(an.occ_saving(o) for o in consumers)
    if rc.transferable:
        added += sum(an.download_seconds * an.weight[o] for o in consumers)
    return saved - added


def merge_estimate(an: Analysis, a: Sequence[str], b: Sequence[str]) -> float:
    ha, hb = a[0], b[0]
    sa, sb = an.by_id[ha].steps, an.by_id[hb].steps
    n = 0
    saved = 0.0
    while n < len(sa) and n < len(sb) and an.token((ha, n)) == an.token((hb, n)) \
            and sa[n].cond == sb[n].cond:
        saved += min(an.occ_saving((ha, n)), an.occ_saving((hb, n)))
        n += 1
    return saved + min(an.overhead[ha] * an.job_weight[ha], an.overhead[hb] * an.job_weight[hb])


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def default_observed(p: Program, repo_root: Path) -> object:
    """The observed contracts ``contracts.annotate`` uses for ``p`` by default (its
    benchmark's trace file, if any), as a mapping; ``{}`` when the contract layer has none."""
    from ghav import contracts
    trace_path = getattr(contracts, "trace_path", None)
    load = getattr(contracts, "load_observed", None)
    if trace_path is None or load is None or not p.source:
        return {}
    return load(trace_path(Path(repo_root), p.source))


def annotate_like(p: Program, repo_root: Path, observed: object) -> Program:
    """``contracts.annotate`` with the given observed contracts (when supported)."""
    import inspect

    from ghav import contracts
    if "observed" in inspect.signature(contracts.annotate).parameters:
        return contracts.annotate(p, Path(repo_root), observed=observed)
    return contracts.annotate(p, Path(repo_root))  # a contract layer without observed contracts


def roundtrip_check(original: Program, candidate: Program, repo_root: Path, *,
                    rlimit: int = 0, observed: object = None, canonical: bool = True
                    ) -> tuple[checker.CheckResult | None, str, str]:
    """``checker.check(original, lower(emit(candidate)))`` with contracts re-inferred.

    Returns (result or None on an emission/lowering error, emitted YAML, error text).
    The emitted file is lowered from a temporary directory against ``repo_root``, so local
    actions and reusable workflows resolve as in the original repository, and annotated
    with the ORIGINAL's observed contracts (``observed``; None = ``default_observed``).
    """
    from ghav.lower import LowerError, lower
    if observed is None:
        observed = default_observed(original, repo_root)
    try:
        text = emitmod.emit(candidate)
    except emitmod.EmitError as exc:
        return None, "", f"emit: {exc}"
    with tempfile.TemporaryDirectory(prefix="ghav-roundtrip-") as d:
        path = Path(d) / "ghav-roundtrip.yml"
        path.write_text(text, encoding="utf-8")
        try:
            again = annotate_like(lower(path, Path(repo_root)), Path(repo_root), observed)
        except (LowerError, OSError, ValueError) as exc:
            return None, text, f"lower: {exc}"
    return checker.check(original, again, rlimit=rlimit, canonical=canonical), text, ""


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def describe(an: Analysis, sk: Sketch, comp: Completion | None = None) -> list[dict]:
    """Human-readable moves of a sketch (deterministic)."""
    out: list[dict] = []
    for cid, enc in sk.shares:
        rc = an.shareable[cid]
        out.append({
            "move": "share",
            "class": cid,
            "producer": enc,
            "producer_job": comp.producers.get(cid) if comp else None,
            "artifact": comp.artifacts.get(cid) if comp and rc.transferable else None,
            "step": rc.label,
            "template": rc.template,
            "transfer": rc.transferable,
            "members": [f"{i}#{k}" for i, k in rc.members],
        })
    for g in sk.merges:
        out.append({"move": "merge", "instances": list(g), "host": choose_host(an, g)})
    return out


def producer_choices(an: Analysis, rc: ReadClass) -> list[str]:
    """Encodings tried for a class: the new job, and at most ``max_producers`` member
    instances (those that run most often, then by id)."""
    if an.cfg.moves == "rules":
        return ["job"]
    members = sorted({o[0] for o in rc.members}, key=lambda i: (-round(an.job_weight[i], 9), i))
    return ["job"] + sorted(members[: an.cfg.max_producers])


def _hard_bucket(an: Analysis, group: Sequence[str]) -> tuple:
    j = an.by_id[group[0]]
    return (call_path(j.id), repr(j.cond), tuple(sorted(j.needs)), repr(j.permissions))


def _children(an: Analysis, sk: Sketch) -> list[tuple[float, Sketch]]:
    """(estimated saving, child sketch) for every move applicable to ``sk``. Merge pairs are
    enumerated only within groups that agree on the hard merge constraints."""
    out: list[tuple[float, Sketch]] = []
    shared = sk.shared_classes()
    for cid, rc in sorted(an.shareable.items()):
        if cid in shared:
            continue
        for enc in producer_choices(an, rc):
            out.append((share_estimate(an, rc, enc), sk.with_share(cid, enc)))
    buckets: dict[tuple, list[tuple[str, ...]]] = defaultdict(list)
    for g in groups_of(an, sk):
        buckets[_hard_bucket(an, g)].append(g)
    for _, groups in sorted(buckets.items()):
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                a, b = groups[x], groups[y]
                if merge_hard_ok(an, tuple(sorted(set(a) | set(b)))):
                    continue
                out.append((merge_estimate(an, a, b), sk.with_merge(a, b)))
    return out


def synthesize(p: Program, cfg: SynthConfig = SynthConfig(), *, repo_root: Path | str | None = None,
               model: costmod.CostModel | None = None,
               contexts: Sequence[tuple[str, float]] | None = None,
               observed: object = None) -> SynthResult:
    """Search for the cheapest verified equivalent of ``p`` (see the module docstring).

    ``repo_root`` is the repository the workflow belongs to; the round-trip check lowers the
    emitted YAML against it (required). ``observed`` are the observed contracts ``p`` was
    annotated with (None = the contract layer's default for ``p``; ``{}`` = static guide
    only); the round trip re-annotates with the same. ``model``/``contexts`` default to the
    mock cost model and its event weights.
    """
    if repo_root is None:
        raise ValueError("synthesize needs repo_root: acceptance requires the emission round trip")
    start = time.perf_counter()
    if observed is None:
        observed = default_observed(p, Path(repo_root))
    model = model or costmod.MOCK
    if contexts is None:
        contexts = getattr(model, "contexts", None) or costmod.DEFAULT_CONTEXTS
    an = Analysis(p, cfg, model, contexts, Path(repo_root))

    root = Sketch()
    orig_cost = program_cost(an, p)
    best_cost, best_sketch, best_prog = orig_cost, root, p
    best_comp: Completion | None = None
    best_check: dict | None = None
    best_rt: dict | None = None
    trace: list[dict] = []

    pruning: Counter = Counter()
    completion: Counter = Counter()
    rejected_checker = rejected_roundtrip = rejected_guard = 0
    bound_pruned = nogood_skipped = 0
    checked = accepted = 0
    expansions = 0
    generated = 1
    nogoods = NogoodIndex()
    is_nogood = nogoods.covers

    heap: list[tuple[float, str]] = [(_r(orig_cost), root.key)]
    sketches: dict[str, Sketch] = {root.key: root}
    seen: set[str] = {root.key}   # sketches pushed or rejected

    while heap and expansions < cfg.max_expansions:
        _, key = heapq.heappop(heap)
        sk = sketches.pop(key)
        if sk.size() and is_nogood(sk):
            nogood_skipped += 1
            continue
        expansions += 1
        if sk.size() == 0:
            comp: Completion | None = None
            cand, c = p, orig_cost
        else:
            try:
                comp = complete(an, sk)
            except Incomplete as exc:
                completion[exc.reason] += 1
                nogoods.add(sk)
                continue
            cand = comp.program
            c = program_cost(an, cand)
        if cfg.cost_bound and sk.size() and lower_bound(an, sk, c) >= best_cost - EPS:
            bound_pruned += 1
            continue
        if comp is not None and cfg.effect_pruning:
            reason = prune(an, comp)
            if reason:
                pruning[reason] += 1
                if reason in MONOTONE_PRUNING:
                    nogoods.add(sk)
                continue
        if comp is not None and c < best_cost - EPS:
            checked += 1
            # Counterexamples of rejected candidates are not reported: skip canonicalization.
            res = checker.check(p, cand, rlimit=cfg.rlimit, canonical=False)
            if not res.equivalent:
                rejected_checker += 1
                nogoods.add(sk)
                continue
            rt, _, err = roundtrip_check(p, cand, Path(repo_root), rlimit=cfg.rlimit, observed=observed,
                                         canonical=False)
            if rt is None or not rt.equivalent:
                rejected_roundtrip += 1
                nogoods.add(sk)
                continue
            if not cfg.effect_pruning and not fail_fast_guard(an, comp):
                rejected_guard += 1
                nogoods.add(sk)
                continue
            accepted += 1
            best_cost, best_sketch, best_prog, best_comp = c, sk, cand, comp
            best_check = res.to_json(include_timing=False)
            best_rt = rt.to_json(include_timing=False)
            trace.append({"expansion": expansions, "cost": _r(c), "sketch": sk.key})
        fresh: list[tuple[float, str, Sketch]] = []
        for saving, child in _children(an, sk):
            ck = child.key
            if ck in seen or ck in sketches:
                continue
            if is_nogood(child):
                seen.add(ck)
                nogood_skipped += 1
                continue
            if cfg.effect_pruning and child.merges != sk.merges:
                bad = [g for g in child.merges if g not in sk.merges and not merge_settings_ok(an, g)]
                if bad:
                    seen.add(ck)
                    pruning["settings"] += 1
                    nogoods.add(child)
                    continue
            fresh.append((_r(c - saving), ck, child))
        # Push the best ``max_children`` (deterministic order); the others may be generated
        # again from another parent.
        for est, ck, child in sorted(fresh, key=lambda t: (t[0], t[1]))[: cfg.max_children]:
            seen.add(ck)
            generated += 1
            sketches[ck] = child
            heapq.heappush(heap, (est, ck))

    budget_exhausted = bool(heap) and expansions >= cfg.max_expansions
    if best_check is None:
        # The original itself: report its self-check and round trip for completeness.
        best_check = checker.check(p, p, rlimit=cfg.rlimit).to_json(include_timing=False)
        rt, _, err = roundtrip_check(p, p, Path(repo_root), rlimit=cfg.rlimit, observed=observed)
        best_rt = rt.to_json(include_timing=False) if rt is not None else {"equivalent": False, "error": err}
    return SynthResult(
        program=best_prog,
        best_cost=best_cost,
        original_cost=orig_cost,
        expansions=expansions,
        generated=generated,
        checked=checked,
        accepted=accepted,
        rejected_pruning=dict(pruning),
        rejected_completion=dict(completion),
        rejected_checker=rejected_checker,
        rejected_roundtrip=rejected_roundtrip,
        rejected_guard=rejected_guard,
        bound_pruned=bound_pruned,
        nogood_skipped=nogood_skipped,
        shareable_classes=len(an.shareable),
        budget_exhausted=budget_exhausted,
        best_moves=describe(an, best_sketch, best_comp),
        config=cfg,
        cost_model=getattr(model, "label", "mock"),
        original_billed_minutes=billed_minutes(an, p),
        best_billed_minutes=billed_minutes(an, best_prog),
        check=best_check,
        roundtrip=best_rt,
        accepted_trace=trace,
        wall_ms=round((time.perf_counter() - start) * 1000),
    )


# ---------------------------------------------------------------------------
# Redundancy (docs/EXPERIMENTS.md), shared with pipeline.workflow_stats
# ---------------------------------------------------------------------------

def redundancy(p: Program, model: costmod.CostModel | None = None,
               contexts: Sequence[tuple[str, float]] | None = None) -> tuple[float, float]:
    """(redundant seconds, total expected machine seconds): the expected mock seconds of
    step occurrences whose read class already occurs in an earlier job instance (instances
    in topological order), and the workflow's expected machine time."""
    model = model or costmod.MOCK
    if contexts is None:
        contexts = getattr(model, "contexts", None) or costmod.DEFAULT_CONTEXTS
    total = float(costmod.expected_machine_time(p, contexts, model))
    keys = occurrence_classes(p, read_values=True)
    by_id = {j.id: j for j in p.jobs}
    weight: dict[Occ, float] = defaultdict(float)
    for event, w in contexts:
        runs = costmod.simulate(p, event, model)
        for jid in sorted(runs):
            for i, flag in enumerate(runs[jid].steps):
                if flag:
                    weight[(jid, i)] += w
    seen_in: dict[tuple, str] = {}
    redundant = 0.0
    for jid in sem.topological_order(p):
        job = by_id[jid]
        if job.frozen:
            continue
        for i, s in enumerate(job.steps):
            k = keys[(jid, i)]
            owner = seen_in.setdefault(k, jid)
            if owner != jid:
                price = model.price_step(job, s)[0] if hasattr(model, "price_step") else costmod.step_seconds(s, model)
                redundant += float(price) * weight[(jid, i)]
    return redundant, total
