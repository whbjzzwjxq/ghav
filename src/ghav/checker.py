"""End-to-end equivalence checking of WIR programs with counterexamples.

``check(original, candidate)`` decides whether the candidate has the same observable
behavior as the original for every execution context of a supported event
(``github.event_name`` in {push, pull_request}) and every repository content (tasks are
uninterpreted functions of their reads). Observable behavior is:

* the final value of every persistent region in P (``semantics.persistent_regions``); and
* the conjunction of check verdicts of every check group. Groups are derived, not searched:
  a new check is related to the old check of the original instance that contributed a step
  to it (``Step.origin_instance``; to every old check of the original job, ``Step.origin``,
  when the instance is unknown), and to the old check of every original occurrence that the new check's
  steps account for through a shared read class (``read_class_accounts``: a displaced
  occurrence is accounted for by every candidate step with the same binding and equal read
  values); groups are the connected components of that relation, extended so that
  every fail-fast group falls into one check group. A side of a group without checks has
  verdict true (a removed check must never have failed; a new unrelated check must never fail).

Programs not produced by synthesis (a re-lowered emitted workflow, a developer-written
change) carry origins that name their own jobs. ``recover_origins`` maps every step whose
origin does not name an original step with the same binding back to one that does
(preferring the original instance with the same check name), and its ``origin_instance`` to
the original occurrence's instance, before any of the above. Cache-transparency findings of
either program are reported as non-blocking issues.

Fail-fast (provisional policy): the semantics never cancels; this is sound for check groups
because a fail-fast cancellation only happens after some member failed, and all members share
one group. A persistent region written by a fail-fast member is additionally checked as a
refinement: the candidate's write condition must lie between the original's lower bound
(own guard and no member of the writer's fail-fast group failed) and upper bound (own guard).
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import z3

from ghav import semantics as sem
from ghav.contracts import parse_transfer
from ghav.wir import EVENTS, JobInstance, Program, Step


@dataclass(frozen=True)
class CheckGroup:
    old: tuple[str, ...]
    new: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{','.join(self.old)}=>{','.join(self.new)}"

    def to_json(self) -> dict:
        return {"old": list(self.old), "new": list(self.new)}


@dataclass
class CheckResult:
    """Outcome of ``check``.

    ``reason``: ``equivalent`` | ``counterexample`` | ``ill-formed`` (the candidate has a
    structural problem or a race the original does not have) | ``original-ill-formed`` |
    ``unknown`` (solver gave up within ``rlimit``). ``smt_ms`` is wall-clock time, reported
    only; it never influences the result.
    """

    equivalent: bool
    counterexample: dict | None
    issues: list[dict]
    smt_ms: int
    reason: str = ""
    groups: list[CheckGroup] = field(default_factory=list)
    persistent: tuple[str, ...] = ()
    recovered_origins: int = 0

    def to_json(self, *, include_timing: bool = True) -> dict:
        out = {
            "equivalent": self.equivalent,
            "reason": self.reason,
            "counterexample": self.counterexample,
            "issues": self.issues,
            "groups": [g.to_json() for g in self.groups],
            "persistent": list(self.persistent),
            "recovered_origins": self.recovered_origins,
        }
        if include_timing:
            out["smt_ms"] = self.smt_ms
        return out


# ---------------------------------------------------------------------------
# Origin recovery
# ---------------------------------------------------------------------------

def is_transfer_step(step: Step) -> bool:
    """A step of the transfer idiom (``contracts.parse_transfer``: pack, upload, download or
    unpack of a ``ghav-`` artifact), added by synthesis or emission."""
    return not step.contract.top and parse_transfer(step.binding) is not None


def _unique_by(jobs, key) -> dict:
    groups: dict = {}
    for j in sorted(jobs, key=lambda x: x.id):
        groups.setdefault(key(j), []).append(j)
    return {k: v[0] for k, v in groups.items() if len(v) == 1}


def recover_origins(original: Program, candidate: Program) -> tuple[Program, int]:
    """Give every candidate step an origin in the original, where one exists.

    A step keeps its origin when the origin names an original step with the same binding
    key, or when it is a transfer step. Any other step (typically every step of a
    re-lowered emitted workflow whose jobs were renamed, or of a developer-written
    workflow) is matched by binding key: first to the k-th occurrence of that binding in
    its partner, the original instance with the same identity (check name and matrix),
    where k counts earlier steps of the candidate job with that binding; otherwise to the
    least original origin with that binding. A step whose binding occurs nowhere in the
    original is new: it gets origin ``(job, -2 - index)``, where ``job`` is the partner's
    original job (or its own), so it joins that job's checks and never matches an original
    step. Returns the candidate with recovered origins and the number of steps changed.
    Deterministic: every choice is by sorted keys.
    """
    obid = {j.id: j for j in original.jobs}
    cbid = {j.id: j for j in candidate.jobs}
    orig_keys: dict[tuple[str, int], set[tuple]] = {}
    # (origin instance, origin) -> binding keys: the original's step occurrences
    occ_keys: dict[tuple[str | None, tuple[str, int]], set[tuple]] = {}
    occ_insts: dict[tuple[tuple[str, int], tuple], set[str | None]] = {}
    by_binding: dict[tuple, list[tuple[str, int]]] = {}
    for j in sorted(original.jobs, key=lambda x: x.id):
        for s in j.steps:
            o = (str(s.origin[0]), int(s.origin[1]))
            k = sem.task_key(s, j, obid)  # matrix instances of one origin may differ in text
            orig_keys.setdefault(o, set()).add(k)
            occ_keys.setdefault((s.origin_instance, o), set()).add(k)
            occ_insts.setdefault((o, k), set()).add(s.origin_instance)
            by_binding.setdefault(k, []).append(o)
    least = {k: min(v) for k, v in by_binding.items()}
    partners = _unique_by(original.jobs, sem.instance_identity)
    cand_unique = _unique_by(candidate.jobs, sem.instance_identity)

    changed = 0
    jobs: list[JobInstance] = []
    for cj in sorted(candidate.jobs, key=lambda x: x.id):
        ident = sem.instance_identity(cj)
        partner = partners.get(ident) if ident in cand_unique else None
        seen: dict[tuple, int] = {}
        steps: list[Step] = []
        for idx, s in enumerate(cj.steps):
            k = sem.task_key(s, cj, cbid)
            rank = seen.get(k, 0)
            seen[k] = rank + 1
            o = (str(s.origin[0]), int(s.origin[1]))
            inst = s.origin_instance
            if is_transfer_step(s):
                if inst is not None and inst not in obid:  # e.g. a re-lowered job's own id
                    changed += 1
                    s = replace(s, origin_instance=partner.id if partner is not None else None)
                steps.append(s)
                continue
            if k in orig_keys.get(o, ()):
                if k not in occ_keys.get((inst, o), ()):
                    # The origin is right; the instance must name an original occurrence:
                    # the partner's, else the only one with that origin and binding.
                    same = [ps.origin_instance for ps in (partner.steps if partner is not None else ())
                            if (str(ps.origin[0]), int(ps.origin[1])) == o
                            and sem.task_key(ps, partner, obid) == k]
                    insts = sorted(occ_insts.get((o, k), set()), key=lambda x: (x is None, x or ""))
                    new_inst = same[0] if same else (insts[0] if len(insts) == 1 else None)
                    if new_inst != inst:
                        changed += 1
                        s = replace(s, origin_instance=new_inst)
                steps.append(s)
                continue
            new: tuple[str, int] | None = None
            new_inst = None
            if partner is not None:
                same = [ps for ps in partner.steps if sem.task_key(ps, partner, obid) == k]
                if rank < len(same):
                    new = (str(same[rank].origin[0]), int(same[rank].origin[1]))
                    new_inst = same[rank].origin_instance
            if new is None:
                new = least.get(k)  # stands for some instance: no instance-level origin
            if new is None:
                new = (partner.orig_job if partner is not None else cj.orig_job, -2 - idx)
            if new != o or new_inst != inst:
                changed += 1
                s = replace(s, origin=new, origin_instance=new_inst)
            steps.append(s)
        jobs.append(replace(cj, steps=tuple(steps)))
    return candidate.with_jobs(jobs), changed


# ---------------------------------------------------------------------------
# Check correspondence
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, str], tuple[str, str]] = {}

    def add(self, x: tuple[str, str]) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: tuple[str, str]) -> tuple[str, str]:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: tuple[str, str], b: tuple[str, str]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def displaced_occurrences(original: Program, candidate: Program) -> list[tuple[str, int]]:
    """Original step occurrences that no longer execute in their own instance.

    An occurrence ``(instance, index)`` of a non-frozen original instance is in place when
    the candidate instances with the same identity (``semantics.instance_identity``) run
    its binding at least as often as the original instance does up to and including it.
    Every other occurrence is displaced: it was removed, moved, or replaced by a transfer.
    """
    obid = {j.id: j for j in original.jobs}
    cbid = {j.id: j for j in candidate.jobs}
    counts: dict[str, Counter] = {}
    for cj in sorted(candidate.jobs, key=lambda x: x.id):
        c = counts.setdefault(sem.instance_identity(cj), Counter())
        c.update(sem.task_key(s, cj, cbid) for s in cj.steps)
    out: list[tuple[str, int]] = []
    for oj in sorted(original.jobs, key=lambda x: x.id):
        if oj.frozen:
            continue
        have = counts.get(sem.instance_identity(oj), Counter())
        seen: Counter = Counter()
        for idx, s in enumerate(oj.steps):
            k = sem.task_key(s, oj, obid)
            seen[k] += 1
            if seen[k] > have[k]:
                out.append((oj.id, idx))
    return out


def _class_key(obs: sem.Observation, by_id: dict[str, JobInstance], job: JobInstance, idx: int) -> tuple:
    reads = obs.step_reads[(job.id, idx)]
    return (sem.task_key(job.steps[idx], job, by_id), tuple((slot, term.get_id()) for slot, term in reads))


def read_class_accounts(
    original: Program, candidate: Program, ctx: sem.SymbolicContext
) -> list[tuple[str, str]]:
    """Pairs (candidate instance, original instance) from shared read classes.

    A displaced original occurrence is accounted for by every candidate step in its read
    class: same binding key and syntactically equal read values when every task succeeds
    (both programs evaluated in ``ctx``). This is the origin rule of completion (C4): the
    executed member of a shared class accounts for the whole class. It is derived from the
    programs, so it also applies to candidates not produced by synthesis. Transfer steps
    account for nothing; frozen jobs are fixed and account only through their origins.
    """
    displaced = displaced_occurrences(original, candidate)
    if not displaced:
        return []
    obs_o = sem.evaluate(original, ctx, assume_success=True)
    obs_c = sem.evaluate(candidate, ctx, assume_success=True)
    index: dict[tuple, set[str]] = {}
    cbid = {j.id: j for j in candidate.jobs}
    for cj in sorted(candidate.jobs, key=lambda x: x.id):
        if cj.frozen:
            continue
        for idx, s in enumerate(cj.steps):
            if not is_transfer_step(s):
                index.setdefault(_class_key(obs_c, cbid, cj, idx), set()).add(cj.id)
    by_id = {j.id: j for j in original.jobs}
    pairs: set[tuple[str, str]] = set()
    for oid, idx in displaced:
        for cid in index.get(_class_key(obs_o, by_id, by_id[oid], idx), ()):
            pairs.add((cid, oid))
    return sorted(pairs)


def check_groups(
    original: Program,
    candidate: Program,
    accounts: Sequence[tuple[str, str]] = (),
) -> list[CheckGroup]:
    """Connected components of the check correspondence between old and new check names.

    A new check is related to the old check of the original instance that contributed a step
    to it (``Step.origin_instance``, when it names an original instance of the step's original
    job), else to every old check of that original job (``Step.origin``); to the old check of
    the original instance with the same identity
    (check name and matrix; this covers jobs without steps, such as calls of remote
    reusable workflows), and to the old check of every original instance in ``accounts``
    (pairs of candidate and original instance ids, see ``read_class_accounts``).
    """
    uf = _UnionFind()
    old_by_job: dict[str, set[str]] = {}
    old_inst: dict[str, JobInstance] = {j.id: j for j in original.jobs}
    for j in original.jobs:
        uf.add(("old", j.check_name))
        old_by_job.setdefault(j.orig_job, set()).add(j.check_name)
    for j in candidate.jobs:
        uf.add(("new", j.check_name))
    for j in sorted(candidate.jobs, key=lambda x: x.id):
        for s in j.steps:
            src = old_inst.get(s.origin_instance) if s.origin_instance is not None else None
            if src is not None and src.orig_job == str(s.origin[0]):
                # the step comes from one original instance: relate to that instance's check
                uf.union(("new", j.check_name), ("old", src.check_name))
                continue
            for name in sorted(old_by_job.get(str(s.origin[0]), ())):
                uf.union(("new", j.check_name), ("old", name))
    old_by_ident: dict[str, set[str]] = {}
    for j in original.jobs:
        old_by_ident.setdefault(sem.instance_identity(j), set()).add(j.check_name)
    for j in sorted(candidate.jobs, key=lambda x: x.id):
        for name in sorted(old_by_ident.get(sem.instance_identity(j), ())):
            uf.union(("new", j.check_name), ("old", name))
    new_check = {j.id: j.check_name for j in candidate.jobs}
    old_check = {j.id: j.check_name for j in original.jobs}
    for cid, oid in sorted(accounts):
        if cid in new_check and oid in old_check:
            uf.union(("new", new_check[cid]), ("old", old_check[oid]))
    for side, prog in (("old", original), ("new", candidate)):
        members: dict[str, list[str]] = {}
        for j in sorted(prog.jobs, key=lambda x: x.id):
            if j.fail_fast_group:
                members.setdefault(j.fail_fast_group, []).append(j.check_name)
        for _, names in sorted(members.items()):
            for n in names[1:]:
                uf.union((side, names[0]), (side, n))
    comps: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
    for node in sorted(uf.parent):
        old, new = comps.setdefault(uf.find(node), (set(), set()))
        (old if node[0] == "old" else new).add(node[1])
    groups = [CheckGroup(tuple(sorted(o)), tuple(sorted(n))) for o, n in comps.values()]
    return sorted(groups, key=lambda g: (g.old, g.new))


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------

def _elapsed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def _issue_rows(orig: list[sem.Issue], cand: list[sem.Issue], blocking: list[sem.Issue]) -> list[dict]:
    rows = [{"program": "original", "blocking": False, **i.to_json()} for i in orig]
    block = {i.sort_key() for i in blocking}
    rows += [{"program": "candidate", "blocking": i.sort_key() in block, **i.to_json()} for i in cand]
    return rows


def _region_differs(name: str, o: sem.Observation, c: sem.Observation) -> z3.BoolRef:
    eo, vo = o.region_value(name)
    ec, vc = c.region_value(name)
    if name.startswith("run:"):
        return z3.Or(eo != ec, z3.And(eo, vo != vc))
    return vo != vc


def _write_bounds(events: tuple[sem.WriteEvent, ...], obs: sem.Observation) -> tuple[z3.BoolRef, z3.BoolRef]:
    zc = obs.ctx.z3
    lower = [
        sem._and(zc, e.guard, obs.no_member_failed(e.fail_fast_group)) if e.fail_fast_group else e.guard
        for e in events
    ]
    upper = [e.guard for e in events]
    return sem._or(zc, *lower), sem._or(zc, *upper)


def _differences(
    persistent: tuple[str, ...],
    groups: list[CheckGroup],
    o: sem.Observation,
    c: sem.Observation,
) -> list[tuple[str, z3.BoolRef, dict]]:
    zc = o.ctx.z3
    diffs: list[tuple[str, z3.BoolRef, dict]] = []
    for name in persistent:
        diffs.append((f"region:{name}", _region_differs(name, o, c), {"region": name}))
    for g in groups:
        vo = sem._and(zc, *[o.checks[n] for n in g.old])
        vc = sem._and(zc, *[c.checks[n] for n in g.new])
        diffs.append((f"group:{g.label}", vo != vc, {"group": g, "old": vo, "new": vc}))
    for name in persistent:
        eo, ec = o.writes.get(name, ()), c.writes.get(name, ())
        if not any(e.fail_fast_group for e in eo + ec):
            continue
        lo, uo = _write_bounds(eo, o)
        lc, uc = _write_bounds(ec, c)
        diffs.append((f"fail-fast-lower:{name}", z3.And(lo, z3.Not(lc)), {"region": name}))
        diffs.append((f"fail-fast-upper:{name}", z3.And(uc, z3.Not(uo)), {"region": name}))
    return diffs


def _is_true(m: z3.ModelRef, e: z3.BoolRef) -> bool:
    return z3.is_true(m.eval(e, model_completion=True))


def _run_summary(m: z3.ModelRef, obs: sem.Observation) -> dict:
    failed_steps = [
        f"{jid}#{idx} {obs.step_labels[(jid, idx)]}"
        for (jid, idx), st in obs.step_status.items()
        if _is_true(m, sem.status_is(st, sem.FAILURE))
    ]
    failed_jobs = [jid for jid, st in obs.job_status.items() if _is_true(m, sem.status_is(st, sem.FAILURE))]
    skipped_jobs = [jid for jid, st in obs.job_status.items() if _is_true(m, sem.status_is(st, sem.SKIPPED))]
    return {"failed_steps": failed_steps, "failed_jobs": failed_jobs, "skipped_jobs": skipped_jobs}


def _context_summary(m: z3.ModelRef, ctx: sem.SymbolicContext) -> dict[str, str]:
    literals = [(n, c) for n, c in ctx.literals() if n.startswith("str:")]
    out: dict[str, str] = {}
    for key, sym in ctx.global_symbols():
        value = "<other>"
        for name, lit in literals:
            if _is_true(m, sym == lit):
                value = name[len("str:"):]
                break
        out[key] = value
    return out


MAX_PREFERENCE_QUERIES = 256


def _is_decided_true(solver: z3.Solver, f: z3.BoolRef) -> bool:
    """Was ``f`` itself asserted (as a preference that held)?"""
    return any(a.eq(f) for a in solver.assertions())


def _canonical_model(
    solver: z3.Solver,
    ctx: sem.SymbolicContext,
    diffs: list[tuple[str, z3.BoolRef, dict]],
    o: sem.Observation,
    c: sem.Observation,
    rlimit: int = 0,
) -> z3.ModelRef | None:
    """A model fixed by a sequence of preferences, independent of solver heuristics.

    Every preference query is bounded by ``rlimit`` (deterministic; 0 = unbounded). A
    query that gives up adds nothing. Returns None if the final query gives up, and the
    caller then reports the solver's first model.

    Preferences, in order: event push; each step (original, then candidate) does not fail;
    each job runs; each difference holds; each global context value differs from every
    literal (else equals the first literal that is possible). Every field of the reported
    counterexample is a function of these facts, so the report is canonical. Runs of
    preferences are tried as blocks first (same result as one at a time), and at most
    ``MAX_PREFERENCE_QUERIES`` queries are made; both keep large workflows tractable
    without depending on time.
    """
    solver.set("rlimit", rlimit)

    budget = [MAX_PREFERENCE_QUERIES]

    def prefer(p: z3.BoolRef) -> bool:
        if budget[0] <= 0:  # deterministic cap on the number of queries
            return False
        budget[0] -= 1
        solver.push()
        solver.add(p)
        res = solver.check()
        solver.pop()
        if res == z3.sat:
            solver.add(p)
            return True
        if res == z3.unsat:
            solver.add(z3.Not(p))
        return False

    def prefer_each(ps: list[z3.BoolRef]) -> None:
        """Same result as ``prefer`` on each of ``ps`` in order (lexicographic greedy), but
        tries whole blocks first: a block whose conjunction is possible is taken at once."""
        if not ps:
            return
        if len(ps) == 1:
            prefer(ps[0])
            return
        if prefer(z3.And(*ps)):
            return
        mid = len(ps) // 2
        prefer_each(ps[:mid])
        prefer_each(ps[mid:])

    prefer(ctx.event_name == ctx.lit(EVENTS[0]))
    prefer_each([z3.Not(sem.status_is(st, sem.FAILURE)) for obs in (o, c) for st in obs.step_status.values()])
    prefer_each([ran for obs in (o, c) for ran in obs.job_ran.values()])
    for _, d, _ in diffs:
        prefer(d)
    literals = [lit for name, lit in ctx.literals() if name.startswith("str:")]
    syms = [sym for key, sym in ctx.global_symbols() if key != "github.event_name"]
    if literals and syms:
        fresh = [z3.And(*[sym != lit for lit in literals]) for sym in syms]
        prefer_each(fresh)
        for sym, f in zip(syms, fresh):
            if _is_decided_true(solver, f):
                continue
            for lit in literals:
                if prefer(sym == lit):
                    break
    if solver.check() != z3.sat:  # only when a bounded query gives up
        return None
    return solver.model()


def _counterexample(
    m: z3.ModelRef,
    ctx: sem.SymbolicContext,
    diffs: list[tuple[str, z3.BoolRef, dict]],
    o: sem.Observation,
    c: sem.Observation,
) -> dict:
    event = next((e for e in EVENTS if _is_true(m, ctx.event_name == ctx.lit(e))), "<other>")
    hits = [(label, info) for label, d, info in diffs if _is_true(m, d)]
    region = next((info["region"] for label, info in hits if label.startswith(("region:", "fail-fast-"))), None)
    group = None
    for label, info in hits:
        if label.startswith("group:"):
            g: CheckGroup = info["group"]
            group = {
                "old": list(g.old),
                "new": list(g.new),
                "original_verdict": _is_true(m, info["old"]),
                "candidate_verdict": _is_true(m, info["new"]),
            }
            break
    return {
        "event": event,
        "differences": [label for label, _ in hits],
        "region": region,
        "group": group,
        "context": _context_summary(m, ctx),
        "original": _run_summary(m, o),
        "candidate": _run_summary(m, c),
    }


def check(
    original: Program,
    candidate: Program,
    *,
    rlimit: int = 0,
    canonical: bool = True,
    recover: bool = True,
    read_classes: bool = True,
) -> CheckResult:
    """Decide whether ``candidate`` is equivalent to ``original``.

    ``rlimit`` (z3 resource limit, deterministic) bounds the decision query; 0 means
    unbounded. ``canonical`` makes the counterexample independent of solver heuristics
    (a few extra incremental queries, only when a counterexample exists). ``recover``
    first maps candidate origins onto the original (``recover_origins``); it changes
    nothing for candidates whose origins already name original steps. ``read_classes``
    adds the read-class accounts of displaced occurrences to the check correspondence
    (``read_class_accounts``); without it, only origins relate checks.
    """
    start = time.perf_counter()
    recovered = 0
    if recover:
        candidate, recovered = recover_origins(original, candidate)
    orig_issues = sem.well_formed(original)
    cand_issues = sem.well_formed(candidate)
    blocking = sem.blocking_issues(orig_issues, cand_issues)
    issues = _issue_rows(orig_issues, cand_issues, blocking)
    # Cache-transparency findings are reported, never blocking (the opaque cache contract
    # already makes the comparison conservative).
    issues += [{"program": side, "blocking": False, **i.to_json()}
               for side, prog in (("original", original), ("candidate", candidate))
               for i in sem.cache_issues(prog)]
    if any(i.kind in sem.STRUCTURAL_ISSUES for i in orig_issues):
        return CheckResult(False, None, issues, _elapsed_ms(start), reason="original-ill-formed",
                           recovered_origins=recovered)
    if blocking:
        return CheckResult(False, None, issues, _elapsed_ms(start), reason="ill-formed",
                           recovered_origins=recovered)

    ctx = sem.SymbolicContext.for_programs(original, candidate)
    obs_o = sem.evaluate(original, ctx)
    obs_c = sem.evaluate(candidate, ctx)
    persistent = sem.persistent_regions(original, ctx.universe)
    accounts = read_class_accounts(original, candidate, ctx) if read_classes else []
    groups = check_groups(original, candidate, accounts)
    diffs = _differences(persistent, groups, obs_o, obs_c)

    solver = z3.Solver(ctx=ctx.z3)
    if rlimit:
        solver.set("rlimit", rlimit)
    solver.add(*ctx.background())
    solver.add(sem._or(ctx.z3, *[d for _, d, _ in diffs]))
    res = solver.check()
    if res == z3.unsat:
        return CheckResult(True, None, issues, _elapsed_ms(start), "equivalent", groups, persistent, recovered)
    if res == z3.sat:
        model = solver.model()
        if canonical:
            model = _canonical_model(solver, ctx, diffs, obs_o, obs_c, rlimit) or model
        cex = _counterexample(model, ctx, diffs, obs_o, obs_c)
        return CheckResult(False, cex, issues, _elapsed_ms(start), "counterexample", groups, persistent, recovered)
    return CheckResult(False, None, issues, _elapsed_ms(start), "unknown", groups, persistent, recovered)


__all__ = [
    "CheckGroup",
    "CheckResult",
    "check",
    "check_groups",
    "displaced_occurrences",
    "is_transfer_step",
    "read_class_accounts",
    "recover_origins",
]
