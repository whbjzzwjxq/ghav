#!/usr/bin/env python3
"""Build scope/random.scope from scope/workflows.scope (offline, deterministic).

Uniform sample of 30 workflows, drawn with random.Random(42) from the sorted list of workflows in
workflows.scope that have at least 2 job instances under ghav.pipeline.workflow_stats. There is no
redundancy filter; the scope is reported next to eval.scope so the effect of that filter is visible.

Usage:
  uv run python scripts/scope/build_random_scope.py [--workflows scope/workflows.scope]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scopelib import OUTPUT_DIR, SCOPE_DIR, SEED, collect_stats, load_workflow_stats, read_scope, rel_to_root, write_json, write_scope  # noqa: E402

MIN_INSTANCES = 2
SAMPLE = 30


def select_random(stats: dict[str, dict], k: int = SAMPLE, seed: int = SEED) -> tuple[list[str], list[str]]:
    """(eligible lines, sampled lines), both sorted."""
    pool = sorted(key for key in stats if stats[key]["instances"] >= MIN_INSTANCES)
    sample = pool if len(pool) <= k else sorted(random.Random(seed).sample(pool, k))
    return pool, sample


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workflows", type=Path, default=SCOPE_DIR / "workflows.scope")
    ap.add_argument("--out", type=Path, default=SCOPE_DIR / "random.scope")
    ap.add_argument("--summary", type=Path, default=OUTPUT_DIR / "scope" / "random.summary.json")
    args = ap.parse_args()
    fn = load_workflow_stats()
    stats, errors = collect_stats(read_scope(args.workflows), fn)
    pool, sample = select_random(stats)
    n = write_scope(args.out, sample)
    write_json(args.summary, {
        "criteria": {"min_instances": MIN_INSTANCES, "sample": SAMPLE, "seed": SEED},
        "workflows": len(stats) + len(errors), "stats_errors": errors, "eligible": len(pool),
        "selected": len(sample)})
    print(f"random.scope: {n} workflows (sampled from {len(pool)} with >= {MIN_INSTANCES} instances, "
          f"{len(errors)} stats errors) -> {rel_to_root(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
