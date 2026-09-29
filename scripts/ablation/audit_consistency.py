#!/usr/bin/env python3
"""Audit every workflow of a scope for disagreement between the cost simulation and the semantics.

For each workflow, ``run_stages.consistency_json`` lists the job instances and steps that
``cost.simulate`` executes (every task succeeding) but that ``semantics.evaluate`` skips in every
context of the same event. Output: ``output/audit/consistency.<scope-stem>.json`` with the
per-workflow counts (sorted keys) and totals. Offline and deterministic.

Usage:
  uv run python scripts/ablation/audit_consistency.py --scope scope/workflows.scope [--workers 8]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stages  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "output" / "audit"


def one(line: str) -> tuple[str, dict]:
    try:
        r = run_stages.consistency_json(line, "mock")
    except Exception as exc:  # noqa: BLE001 - reported per workflow
        return line, {"error": f"{type(exc).__name__}: {exc}"}
    return line, {
        "benchmark": r["benchmark"],
        "consistent": r["consistent"],
        "jobs_never_run_in_semantics": r["jobs_never_run_in_semantics"],
        "steps_never_run_in_semantics": r["steps_never_run_in_semantics"],
        "examples": sorted({s for e in r["events"].values() for s in e["jobs"] + e["steps"]})[:5],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scope", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=run_stages.DEFAULT_WORKERS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    lines = run_stages.read_scope_lines(args.scope)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        results = dict(ex.map(one, lines, chunksize=8))
    rows = {results[ln].get("benchmark", ln): {**results[ln], "scope_line": ln} for ln in lines}
    inconsistent = sorted(k for k, v in rows.items() if v.get("consistent") is False)
    errors = sorted(k for k, v in rows.items() if "error" in v)
    report = {
        "scope": args.scope.name,
        "workflows": len(lines),
        "errors": len(errors),
        "inconsistent_workflows": len(inconsistent),
        "jobs_never_run_in_semantics": sum(v.get("jobs_never_run_in_semantics", 0) for v in rows.values()),
        "steps_never_run_in_semantics": sum(v.get("steps_never_run_in_semantics", 0) for v in rows.values()),
        "inconsistent": inconsistent,
        "workflows_detail": {k: rows[k] for k in sorted(rows)},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"consistency.{args.scope.stem}.json"
    path.write_text(run_stages.dumps(report), encoding="utf-8")
    print(f"{args.scope.name}: {len(inconsistent)}/{len(lines)} workflows inconsistent "
          f"({report['jobs_never_run_in_semantics']} jobs, {report['steps_never_run_in_semantics']} steps), "
          f"{len(errors)} errors -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
