#!/usr/bin/env python3
"""RQ4: check every developer-written optimization (before/after pair) with ``checker.check``.

Input: ``scope/devopt.pairs.json`` (built by ``scripts/scope/build_devopt_scope.py``). For each
pair, the before workflow is lowered against the before snapshot and the after workflow against
the after snapshot, both annotated with the static contract guide (no traces), and
``checker.check(before, after)`` decides equivalence. Verdicts:

* ``equivalent``: the checker proves the after workflow equivalent to the before workflow;
* ``counterexample``: the checker returns a context (event, failing tasks) in which a
  persistent region or a check-group verdict differs; the summary names them;
* ``unsupported``: GHAV cannot give a verdict: a side fails to lower, the before workflow is
  ill-formed, the after workflow has a structural issue or race the before one does not have
  (``ill-formed``), or the solver gives up (``unknown``). The reason is recorded.
  Also unsupported (``original-duplicate-artifact``) when the checker says equivalent or
  counterexample but the before workflow has a ``duplicate-artifact`` issue, i.e. two unordered
  instances upload an artifact whose name the semantics cannot tell apart (typically an
  unresolved ``${{ steps.*.outputs.* }}`` name). The semantics then makes one upload fail in
  every context, so every job downstream of it never runs in the model and a verdict about
  those jobs is vacuous. The checker's own reason is kept in ``checker_reason``.

Each pair also records mock-cost machine seconds and billed minutes of both sides (the
developer's change priced with the same table; no historical timings exist for these commits),
lowering statistics, and how many after steps have a task key (binding) that occurs nowhere in
the before workflow (new tasks to the checker, e.g. a sharded test command or a bumped action
version).

Output: ``output/stages/devopt.pairs/<pair-id>__<benchmark-id>/devopt_check.json`` (sorted keys;
no wall-clock fields) and ``timing.json`` next to them (reported only).

Usage:
  uv run python scripts/ablation/check_devopt.py [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAIRS = REPO_ROOT / "scope" / "devopt.pairs.json"
PRS = REPO_ROOT / "resources" / "devopt" / "prs.json"
OUT = REPO_ROOT / "output" / "stages" / "devopt.pairs"
DEFAULT_WORKERS = 8
MAX_LISTED = 5


def dumps(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def pair_key(pair: dict) -> str:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from ghav.pipeline import parse_scope_line
    return f"{pair['id']}__{parse_scope_line(pair['before']).id}"


def _load(line: str):
    from ghav import pipeline
    return pipeline.load_program(line, use_traces=False)


def _stats(p) -> dict:
    from ghav import cost as costmod
    frozen = sum(1 for j in p.jobs if j.frozen)
    steps = sum(len(j.steps) for j in p.jobs)
    tops = sum(1 for j in p.jobs for s in j.steps if s.contract.top)
    rep = costmod.cost_report(p, None, costmod.MOCK)
    return {"instances": len(p.jobs), "frozen_instances": frozen, "steps": steps, "top_steps": tops,
            "mock_seconds": round(float(rep["expected_seconds"]), 6),
            "mock_billed_minutes": round(float(rep["expected_billed_minutes"]), 6)}


def _label(step) -> str:
    b = step.binding
    if b.name:
        return str(b.name)
    lines = [ln for ln in str(b.text or "").splitlines() if ln.strip()]
    return lines[0].strip() if lines else ""


def _new_steps(before, after) -> dict:
    """After steps whose task key (``semantics.task_key``: the binding, instance-independent)
    occurs nowhere in the before workflow: new tasks to the checker, e.g. a test command with a
    shard argument or a bumped action version."""
    from ghav import semantics as sem
    bb = {j.id: j for j in before.jobs}
    ab = {j.id: j for j in after.jobs}
    keys = {repr(sem.task_key(st, j, bb)) for j in before.jobs for st in j.steps}
    new = [f"{j.id}#{i} {_label(st)}".strip()
           for j in sorted(after.jobs, key=lambda x: x.id) for i, st in enumerate(j.steps)
           if repr(sem.task_key(st, j, ab)) not in keys]
    return {"count": len(new), "steps": new[:MAX_LISTED]}


def _top_involved(labels: list[str], p) -> bool:
    """Whether some listed failing step (``<job>#<i> ...``) of ``p`` is ⊤ or in a frozen job."""
    by_id = {j.id: j for j in p.jobs}
    for lab in labels:
        head = lab.split(" ", 1)[0]
        jid, _, idx = head.rpartition("#")
        j = by_id.get(jid)
        if j is None:
            continue
        if j.frozen:
            return True
        try:
            k = int(idx)
        except ValueError:
            continue
        if 0 <= k < len(j.steps) and j.steps[k].contract.top:
            return True
    return False


def summarize_counterexample(cex: dict) -> str:
    diffs = cex.get("differences") or []
    parts = [f"event={cex.get('event')}"]
    if cex.get("region"):
        parts.append(f"region {cex['region']} differs")
    g = cex.get("group")
    if g:
        parts.append(f"check group {','.join(g['old'])} => {','.join(g['new'])}: before "
                     f"{'pass' if g['original_verdict'] else 'fail'}, after "
                     f"{'pass' if g['candidate_verdict'] else 'fail'}")
    parts.append(f"{len(diffs)} difference(s)")
    for side in ("original", "candidate"):
        fs = (cex.get(side) or {}).get("failed_steps") or []
        if fs:
            shown = "; ".join(fs[:2]) + (f"; +{len(fs) - 2}" if len(fs) > 2 else "")
            parts.append(f"{'before' if side == 'original' else 'after'} failing: {shown}")
    return " | ".join(parts)


def check_pair(pair: dict) -> dict:
    from ghav import checker
    rec: dict = {k: pair[k] for k in ("id", "pr", "reason", "before", "after")}
    try:
        before, _ = _load(pair["before"])
    except Exception as exc:  # noqa: BLE001
        return {**rec, "verdict": "unsupported", "verdict_reason": "before-lower-error",
                "error": f"{type(exc).__name__}: {exc}"}
    try:
        after, _ = _load(pair["after"])
    except Exception as exc:  # noqa: BLE001
        return {**rec, "verdict": "unsupported", "verdict_reason": "after-lower-error",
                "error": f"{type(exc).__name__}: {exc}", "before_stats": _stats(before)}
    rec["before_stats"] = _stats(before)
    rec["after_stats"] = _stats(after)
    rec["new_steps"] = _new_steps(before, after)
    try:
        res = checker.check(before, after)
    except Exception as exc:  # noqa: BLE001
        return {**rec, "verdict": "unsupported", "verdict_reason": "checker-error",
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    rec["check"] = {k: v for k, v in res.to_json(include_timing=False).items() if k != "issues"}
    rec["checker_reason"] = res.reason
    counts: dict[str, int] = {}
    for i in res.issues:
        key = f"{i.get('program')}:{i.get('kind')}:{'blocking' if i.get('blocking') else 'non-blocking'}"
        counts[key] = counts.get(key, 0) + 1
    rec["issue_counts"] = dict(sorted(counts.items()))
    kinds: dict[str, int] = {}
    for i in res.issues:
        if i.get("program") == "original":
            kinds[i.get("kind", "?")] = kinds.get(i.get("kind", "?"), 0) + 1
    rec["original_issue_kinds"] = dict(sorted(kinds.items()))
    if kinds.get("duplicate-artifact") and res.reason in ("equivalent", "counterexample"):
        rec["verdict"], rec["verdict_reason"] = "unsupported", "original-duplicate-artifact"
        if res.counterexample:
            rec["counterexample_summary"] = summarize_counterexample(res.counterexample)
    elif res.reason == "equivalent":
        rec["verdict"], rec["verdict_reason"] = "equivalent", "equivalent"
    elif res.reason == "counterexample":
        cex = res.counterexample or {}
        rec["verdict"], rec["verdict_reason"] = "counterexample", "counterexample"
        rec["counterexample_summary"] = summarize_counterexample(cex)
        # ext:* is the catch-all external region, written only by ⊤ steps and blanket-⊤ jobs
        rec["counterexample_involves_top_or_frozen"] = (
            cex.get("region") == "ext:*"
            or _top_involved((cex.get("original") or {}).get("failed_steps") or [], before)
            or _top_involved((cex.get("candidate") or {}).get("failed_steps") or [], after))
    else:
        rec["verdict"], rec["verdict_reason"] = "unsupported", res.reason
        rec["blocking_issues"] = [{k: i.get(k) for k in ("kind", "message", "program", "region")}
                                  for i in res.issues if i.get("blocking")][:MAX_LISTED]
    return rec


def run_one(pair: dict) -> tuple[str, dict, float]:
    t = time.perf_counter()
    rec = check_pair(pair)
    return pair_key(pair), rec, round(time.perf_counter() - t, 3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", type=Path, default=PAIRS)
    ap.add_argument("--prs", type=Path, default=PRS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = ap.parse_args(argv)
    pairs = sorted(json.loads(args.pairs.read_text(encoding="utf-8"))["pairs"],
                   key=lambda p: (p["id"], p["before"]))
    titles = {}
    if args.prs.exists():
        titles = {k: v.get("title") for k, v in json.loads(args.prs.read_text(encoding="utf-8")).items()}
    t0 = time.perf_counter()
    timing = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers), max_tasks_per_child=1) as ex:
        for key, rec, wall in ex.map(run_one, pairs):
            rec["title"] = titles.get(rec["id"])
            path = args.out / key / "devopt_check.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dumps(rec), encoding="utf-8")
            timing[key] = wall
            print(f"[devopt] {key}: {rec['verdict']} ({rec['verdict_reason']})"
                  + (f" -- {rec['counterexample_summary']}" if rec.get("counterexample_summary") else "")
                  + (f" -- {rec['error']}" if rec.get("error") else ""), flush=True)
    (args.out / "timing.json").write_text(
        dumps({"workers": args.workers, "wall_s": round(time.perf_counter() - t0, 3), "pairs": timing}),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
