#!/usr/bin/env python3
"""Fetch the candidate repository pool into resources/pool.json.

Query (docs/EXPERIMENTS.md): GitHub repository search
`language:TypeScript stars:>2500 fork:false archived:false`, sorted by stars descending,
first 300 repositories. For each repository the script also resolves the pinned commit: the
last commit on the default branch whose commit date is before the cutoff 2026-09-01T00:00:00Z
(GraphQL `history(first: 1, until: cutoff)`).

The pool is frozen once written: re-running the script only fills in missing pins. Pass
`--force` to repeat the search (which changes the benchmark; the new retrieval date is recorded).

Usage:
  uv run python scripts/resources/fetch_pool.py [--out resources/pool.json] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "scope")]

from ghlib import Gh, GhError, gql_str, log, utc_now_iso  # noqa: E402
from scopelib import (  # noqa: E402
    CUTOFF,
    POOL_ORDER,
    POOL_QUERY,
    POOL_SIZE,
    POOL_SORT,
    RESOURCES,
    write_json,
)

DEFAULT_OUT = RESOURCES / "pool.json"
PIN_BATCH = 25
UNRESOLVABLE = "repository not resolvable"

_KEEP = ("full_name", "node_id", "stargazers_count", "default_branch", "language", "fork",
         "archived", "created_at", "pushed_at")


def search_pool(gh: Gh, size: int = POOL_SIZE) -> tuple[list[dict], int]:
    """First `size` unique repositories of the pool query, in rank order."""
    seen: set[str] = set()
    repos: list[dict] = []
    total = 0
    page = 1
    while len(repos) < size and page <= 10:
        resp = gh.rest("search/repositories", {"q": POOL_QUERY, "sort": POOL_SORT,
                                               "order": POOL_ORDER, "per_page": 100, "page": page})
        total = int(resp.get("total_count", 0))
        items = resp.get("items") or []
        if not items:
            break
        for it in items:
            key = it["full_name"].lower()
            if key in seen:
                continue
            seen.add(key)
            repos.append({k: it.get(k) for k in _KEEP})
        page += 1
    repos = repos[:size]
    for rank, r in enumerate(repos, 1):
        r["rank"] = rank
    return repos, total


def _pin_query(node_ids: list[str]) -> str:
    ids = ", ".join(gql_str(i) for i in node_ids)
    return (
        "query { rateLimit { cost remaining resetAt } "
        f"nodes(ids: [{ids}]) {{ ... on Repository {{ id nameWithOwner isFork isArchived "
        "defaultBranchRef { name target { ... on Commit { "
        f"history(first: 1, until: {gql_str(CUTOFF)}) {{ nodes {{ oid committedDate }} }} }} }} }} }} }} }}"
    )


def resolve_pins(gh: Gh, repos: list[dict]) -> None:
    """Fill `pinned_sha`/`pinned_committed_date` (None if no commit precedes the cutoff)."""
    # Unresolvable repositories are retried on every run (the failure may be transient).
    todo = [r for r in repos if "pinned_sha" not in r or r.get("pin_error") == UNRESOLVABLE]
    for i in range(0, len(todo), PIN_BATCH):
        batch = todo[i:i + PIN_BATCH]
        resp = gh.graphql(_pin_query([r["node_id"] for r in batch]))
        nodes = (resp.get("data") or {}).get("nodes") or []
        if len(nodes) != len(batch):
            raise GhError(f"nodes() returned {len(nodes)} results for {len(batch)} ids")
        for r, n in zip(batch, nodes, strict=True):   # nodes() answers in the order of the ids
            r.pop("pin_error", None)
            if not isinstance(n, dict) or not n.get("nameWithOwner"):
                r["pinned_sha"] = None
                r["pin_error"] = UNRESOLVABLE
                continue
            ref = n.get("defaultBranchRef") or {}
            hist = ((ref.get("target") or {}).get("history") or {}).get("nodes") or []
            r["resolved_name"] = n.get("nameWithOwner")
            r["pinned_branch"] = ref.get("name")
            r["pinned_sha"] = hist[0]["oid"] if hist else None
            r["pinned_committed_date"] = hist[0]["committedDate"] if hist else None
            if not hist:
                r["pin_error"] = "no default-branch commit before cutoff"
        log(f"[pool] pinned {min(i + PIN_BATCH, len(todo))}/{len(todo)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="repeat the search even if the pool exists")
    args = ap.parse_args()

    gh = Gh()
    if args.out.exists() and not args.force:
        pool = json.loads(args.out.read_text(encoding="utf-8"))
        log(f"[pool] {args.out} exists ({len(pool['repos'])} repos, retrieved {pool['retrieved_at']}); "
            "resolving missing pins only")
    else:
        repos, total = search_pool(gh)
        pool = {
            "query": POOL_QUERY, "sort": POOL_SORT, "order": POOL_ORDER, "size": POOL_SIZE,
            "cutoff": CUTOFF, "retrieved_at": utc_now_iso(), "total_count": total,
            "pin_rule": "last commit of the default branch with commit date before the cutoff "
                        "(GraphQL history(first: 1, until: cutoff))",
            "repos": repos,
        }
        write_json(args.out, pool)
        log(f"[pool] search returned {len(repos)} repos (total_count={total})")
    resolve_pins(gh, pool["repos"])
    write_json(args.out, pool)
    pinned = sum(1 for r in pool["repos"] if r.get("pinned_sha"))
    print(f"pool: {len(pool['repos'])} repos, {pinned} pinned -> {args.out} ({gh.calls} API calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
