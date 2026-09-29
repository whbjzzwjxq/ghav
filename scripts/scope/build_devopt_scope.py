#!/usr/bin/env python3
"""Build scope/devopt.scope and scope/devopt.pairs.json (offline, deterministic).

Input: resources/devopt/prs.json and the before/after snapshots written by
scripts/resources/fetch_devopt.py (`make fetch-devopt`).

A PR contributes one (before, after) pair per affected workflow file:
  - a workflow file the PR modifies or renames, or
  - a workflow whose local closure (reusable workflows and composite actions, transitively) at the
    before commit contains a file the PR modifies.
A pair is kept when the workflow exists at both commits and its before version triggers on
`push` or `pull_request`. Workflows the PR adds or removes, and PRs whose change is not captured
by the snapshot pair, are listed under `excluded` with a reason.

devopt.scope lists the before versions (the benchmark workflows, `owner/repo@before:path`);
devopt.pairs.json maps each to its after version for RQ4.

Usage:
  uv run python scripts/scope/build_devopt_scope.py
"""

from __future__ import annotations

import argparse
import json
import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scopelib import (  # noqa: E402
    ACTION_FILES,
    REPOS_DIR,
    RESOURCES,
    SCOPE_DIR,
    ScopeLine,
    is_workflow_file,
    load_manifest,
    load_yaml,
    local_closure,
    rel_to_root,
    snapshot_dir,
    triggers_core_event,
    write_json,
    write_scope,
)

USABLE_RULES = ("merge-first-parent", "pr-base")


def affected_workflows(entry: dict, before_m: dict, before_snap: Path) -> dict[str, tuple[str, str]]:
    """before path -> (after path, reason) for every workflow the PR affects."""
    out: dict[str, tuple[str, str]] = {}
    changed_paths: set[str] = set()
    for ch in entry["changed_files"]:
        new, old = ch["filename"], ch.get("previous_filename") or ch["filename"]
        changed_paths.update({new, old})
        if ch["status"] in ("modified", "changed", "renamed") and is_workflow_file(old) and is_workflow_file(new):
            out[old] = (new, "workflow-changed")
    changed_action_dirs = {posixpath.dirname(p) for p in changed_paths
                           if posixpath.basename(p) in ACTION_FILES}
    for wf in sorted(before_m.get("workflows", [])):
        if wf in out:
            continue
        actions, workflows = local_closure(before_snap, wf)
        if actions & changed_action_dirs:
            out[wf] = (wf, "local-action-changed")
        elif workflows & changed_paths:
            out[wf] = (wf, "reusable-workflow-changed")
    return out


def build_devopt(prs: dict, repos_dir: Path = REPOS_DIR) -> tuple[list[ScopeLine], dict]:
    pairs: list[dict] = []
    excluded: list[dict] = []
    for pid in sorted(prs):
        e = prs[pid]
        pr = f"{e['repo']}#{e['number']}"
        if e.get("before_rule") not in USABLE_RULES:
            excluded.append({"id": pid, "pr": pr, "reason": f"before rule {e.get('before_rule')}"})
            continue
        repo, before, after = e["repo"], e["before_sha"], e["after_sha"]
        bm, am = load_manifest(repo, before, repos_dir), load_manifest(repo, after, repos_dir)
        if bm is None or am is None:
            raise SystemExit(f"error: snapshots for {pid} missing; run `make fetch-devopt`")
        bsnap, asnap = snapshot_dir(repo, before, repos_dir), snapshot_dir(repo, after, repos_dir)
        for ch in e["changed_files"]:
            if ch["status"] in ("added", "removed") and is_workflow_file(ch["filename"]):
                excluded.append({"id": pid, "pr": pr, "path": ch["filename"], "reason": f"workflow {ch['status']}"})
        for bpath, (apath, reason) in sorted(affected_workflows(e, bm, bsnap).items()):
            item = {"id": pid, "pr": pr, "path": bpath}
            if not (bsnap / bpath).is_file() or not (asnap / apath).is_file():
                excluded.append({**item, "reason": "missing in a snapshot"})
                continue
            if not triggers_core_event(load_yaml((bsnap / bpath).read_text(encoding="utf-8"))):
                excluded.append({**item, "reason": "no push/pull_request trigger"})
                continue
            pairs.append({"id": pid, "pr": pr, "reason": reason, "before_rule": e["before_rule"],
                          "before": str(ScopeLine(repo, before, bpath)),
                          "after": str(ScopeLine(repo, after, apath))})
    lines = sorted(ScopeLine.parse(p["before"]) for p in pairs)
    report = {"pairs": pairs, "excluded": excluded, "prs": len(prs),
              "prs_with_pairs": len({p["id"] for p in pairs})}
    return lines, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prs", type=Path, default=RESOURCES / "devopt" / "prs.json")
    ap.add_argument("--out", type=Path, default=SCOPE_DIR / "devopt.scope")
    ap.add_argument("--pairs", type=Path, default=SCOPE_DIR / "devopt.pairs.json")
    ap.add_argument("--repos-dir", type=Path, default=REPOS_DIR)
    args = ap.parse_args()
    if not args.prs.exists():
        print(f"error: {rel_to_root(args.prs)} missing; run `make fetch-devopt`", file=sys.stderr)
        return 1
    lines, report = build_devopt(json.loads(args.prs.read_text(encoding="utf-8")), args.repos_dir)
    n = write_scope(args.out, lines)
    write_json(args.pairs, report)
    print(f"devopt.scope: {n} workflow pairs from {report['prs_with_pairs']}/{report['prs']} PRs, "
          f"{len(report['excluded'])} excluded -> {rel_to_root(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
