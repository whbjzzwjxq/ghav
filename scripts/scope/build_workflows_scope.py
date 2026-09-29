#!/usr/bin/env python3
"""Build scope/workflows.scope from the pool snapshots (offline, deterministic).

A workflow is included when it is a `.github/workflows/*.yml|yaml` file of a pooled repository
snapshot, parses as YAML, and its `on:` includes `push` or `pull_request`.

The second criterion of docs/EXPERIMENTS.md ("lowers without error") is applied with
`--check-lowering`, which calls `ghav.lower.lower(workflow_path, repo_root)` and drops files it
rejects; `make build-workflows-scope` always passes it (the flag stays optional so the script
also runs without the ghav package). The eval/random builders additionally drop workflows whose
stats fail.

Usage:
  uv run python scripts/scope/build_workflows_scope.py [--pool-scope scope/pool.scope] [--check-lowering]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scopelib import (  # noqa: E402
    OUTPUT_DIR,
    REPOS_DIR,
    SCOPE_DIR,
    ScopeLine,
    load_manifest,
    load_yaml,
    read_scope,
    rel_to_root,
    snapshot_dir,
    triggers_core_event,
    write_json,
    write_scope,
)

LowerCheck = Callable[[Path, Path], str | None]   # (workflow file, repo root) -> error or None


def lowering_check() -> LowerCheck:
    """Hook for the lowering criterion; imported lazily so this script runs before lower.py exists."""
    try:
        from ghav.lower import lower  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - depends on other modules
        raise SystemExit(f"error: --check-lowering needs ghav.lower.lower ({e})") from e

    def check(wf: Path, root: Path) -> str | None:
        try:
            lower(wf, root)
        except Exception as e:  # noqa: BLE001 - any lowering failure excludes the workflow
            return f"{type(e).__name__}: {e}"
        return None

    return check


def build_workflows(pool: list[ScopeLine], repos_dir: Path = REPOS_DIR,
                    lower_check: LowerCheck | None = None) -> tuple[list[ScopeLine], dict]:
    summary: dict = {"repos": len(pool), "repos_missing_snapshot": [], "repos_not_ok": [],
                     "repos_with_workflows": 0, "repos_with_selected": 0, "workflow_files": 0,
                     "yaml_errors": 0, "no_core_trigger": 0, "lowering_errors": {}, "selected": 0}
    lines: list[ScopeLine] = []
    for ln in sorted(pool):
        if ln.sha is None:
            raise ValueError(f"pool line is not pinned: {ln}")
        m = load_manifest(ln.repo, ln.sha, repos_dir)
        if m is None or not m.get("complete"):
            summary["repos_missing_snapshot"].append(str(ln))
            continue
        if m.get("status") != "ok":
            summary["repos_not_ok"].append(f"{ln} ({m.get('status')})")
            continue
        wfs = sorted(m.get("workflows", []))
        summary["workflow_files"] += len(wfs)
        summary["repos_with_workflows"] += bool(wfs)
        snap = snapshot_dir(ln.repo, ln.sha, repos_dir)
        picked = 0
        for wf in wfs:
            doc = load_yaml((snap / wf).read_text(encoding="utf-8", errors="replace"))
            if not isinstance(doc, dict):
                summary["yaml_errors"] += 1
                continue
            if not triggers_core_event(doc):
                summary["no_core_trigger"] += 1
                continue
            line = ScopeLine(ln.repo, ln.sha, wf)
            if lower_check is not None:
                err = lower_check(snap / wf, snap)
                if err is not None:
                    summary["lowering_errors"][str(line)] = err
                    continue
            lines.append(line)
            picked += 1
        summary["repos_with_selected"] += bool(picked)
    summary["selected"] = len(lines)
    summary["lowering_checked"] = lower_check is not None
    return sorted(lines), summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool-scope", type=Path, default=SCOPE_DIR / "pool.scope")
    ap.add_argument("--out", type=Path, default=SCOPE_DIR / "workflows.scope")
    ap.add_argument("--summary", type=Path, default=OUTPUT_DIR / "scope" / "workflows.summary.json")
    ap.add_argument("--repos-dir", type=Path, default=REPOS_DIR)
    ap.add_argument("--check-lowering", action="store_true")
    args = ap.parse_args()
    if not args.pool_scope.exists():
        print(f"error: {rel_to_root(args.pool_scope)} missing; run `make build-pool-scope`", file=sys.stderr)
        return 1
    lines, summary = build_workflows(read_scope(args.pool_scope), args.repos_dir,
                                     lowering_check() if args.check_lowering else None)
    if summary["repos_missing_snapshot"]:
        print(f"error: {len(summary['repos_missing_snapshot'])} pool snapshots missing; "
              "run `make fetch-pool-snapshots`", file=sys.stderr)
        return 1
    n = write_scope(args.out, lines)
    write_json(args.summary, summary)
    print(f"workflows.scope: {n} workflows from {summary['repos_with_selected']} repos "
          f"({summary['workflow_files']} workflow files, {summary['no_core_trigger']} without push/pull_request, "
          f"{summary['yaml_errors']} YAML errors, {len(summary['repos_not_ok'])} repos unavailable) "
          f"-> {rel_to_root(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
