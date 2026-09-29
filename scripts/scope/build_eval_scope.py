#!/usr/bin/env python3
"""Build scope/eval.scope from scope/workflows.scope (offline, deterministic).

A workflow qualifies when, under ghav.pipeline.workflow_stats (mock cost model):
  - it has at least 2 job instances,
  - at least half of its job instances are not frozen, and
  - its redundancy ratio is at least 0.2.
If more than 100 qualify, a uniform sample of 100 is drawn with random.Random(42) from the sorted
list. Workflows whose stats cannot be computed are excluded and listed in the summary.

Usage:
  uv run python scripts/scope/build_eval_scope.py [--workflows scope/workflows.scope]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scopelib import OUTPUT_DIR, SCOPE_DIR, SEED, collect_stats, load_workflow_stats, read_scope, rel_to_root, write_json, write_scope  # noqa: E402

MIN_INSTANCES = 2
MIN_REDUNDANCY = 0.2
CAP = 100


def eligible(st: dict) -> bool:
    return (st["instances"] >= MIN_INSTANCES
            and 2 * st["frozen"] <= st["instances"]
            and st["redundancy_ratio"] >= MIN_REDUNDANCY)


def select_eval(stats: dict[str, dict], cap: int = CAP, seed: int = SEED) -> tuple[list[str], list[str]]:
    """(qualifying lines, selected lines), both sorted."""
    qualifying = sorted(k for k in stats if eligible(stats[k]))
    selected = qualifying if len(qualifying) <= cap else sorted(random.Random(seed).sample(qualifying, cap))
    return qualifying, selected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workflows", type=Path, default=SCOPE_DIR / "workflows.scope")
    ap.add_argument("--out", type=Path, default=SCOPE_DIR / "eval.scope")
    ap.add_argument("--summary", type=Path, default=OUTPUT_DIR / "scope" / "eval.summary.json")
    args = ap.parse_args()
    fn = load_workflow_stats()
    stats, errors = collect_stats(read_scope(args.workflows), fn)
    qualifying, selected = select_eval(stats)
    n = write_scope(args.out, selected)
    write_json(args.summary, {
        "criteria": {"min_instances": MIN_INSTANCES, "min_not_frozen_share": 0.5,
                     "min_redundancy_ratio": MIN_REDUNDANCY, "cap": CAP, "seed": SEED},
        "workflows": len(stats) + len(errors), "stats_errors": errors, "qualifying": len(qualifying),
        "selected": len(selected), "stats": stats})
    print(f"eval.scope: {n} workflows ({len(qualifying)} qualifying of {len(stats)} with stats, "
          f"{len(errors)} stats errors) -> {rel_to_root(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
