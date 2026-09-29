#!/usr/bin/env python3
"""Build scope/pool.scope from resources/pool.json (offline, deterministic).

pool.scope = the fetched pool (first 300 repositories of the search, in rank order) minus every
repository that appears in scope/dev.scope, each pinned to its last default-branch commit before
the cutoff. Repositories without such a commit are dropped and counted. Output lines are
`owner/repo@sha`, sorted.

Usage:
  uv run python scripts/scope/build_pool_scope.py [--pool resources/pool.json] [--dev scope/dev.scope]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scopelib import OUTPUT_DIR, POOL_SIZE, RESOURCES, SCOPE_DIR, ScopeLine, read_scope, rel_to_root, write_json, write_scope  # noqa: E402


def build_pool(pool: dict, dev: list[ScopeLine], size: int = POOL_SIZE) -> tuple[list[ScopeLine], dict]:
    dev_repos = {ln.repo.lower() for ln in dev}
    repos = sorted(pool["repos"], key=lambda r: r["rank"])[:size]
    lines: list[ScopeLine] = []
    summary = {"pool": len(repos), "dev_overlap": [], "unpinned": [], "selected": 0}
    for r in repos:
        names = {r["full_name"].lower(), (r.get("resolved_name") or r["full_name"]).lower()}
        if names & dev_repos:
            summary["dev_overlap"].append(r["full_name"])
            continue
        if not r.get("pinned_sha"):
            summary["unpinned"].append(r["full_name"])
            continue
        lines.append(ScopeLine(r["full_name"], r["pinned_sha"]))
    summary["dev_overlap"].sort(key=str.lower)
    summary["unpinned"].sort(key=str.lower)
    summary["selected"] = len(lines)
    return sorted(lines), summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", type=Path, default=RESOURCES / "pool.json")
    ap.add_argument("--dev", type=Path, default=SCOPE_DIR / "dev.scope")
    ap.add_argument("--out", type=Path, default=SCOPE_DIR / "pool.scope")
    ap.add_argument("--summary", type=Path, default=OUTPUT_DIR / "scope" / "pool.summary.json")
    args = ap.parse_args()
    if not args.pool.exists():
        print(f"error: {rel_to_root(args.pool)} missing; run `make fetch-pool` first", file=sys.stderr)
        return 1
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    missing = [r["full_name"] for r in pool["repos"] if "pinned_sha" not in r]
    if missing:
        print(f"error: {len(missing)} pool repos have no pin yet; re-run `make fetch-pool`", file=sys.stderr)
        return 1
    lines, summary = build_pool(pool, read_scope(args.dev))
    summary.update({"query": pool["query"], "retrieved_at": pool["retrieved_at"], "cutoff": pool["cutoff"]})
    n = write_scope(args.out, lines)
    write_json(args.summary, summary)
    print(f"pool.scope: {n} repos (pool {summary['pool']}, dev overlap {len(summary['dev_overlap'])}, "
          f"unpinned {len(summary['unpinned'])}) -> {rel_to_root(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
