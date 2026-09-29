#!/usr/bin/env python3
"""Fetch developer-optimization PRs (resources/devopt_prs.txt) and their before/after snapshots.

For each line `Dxx owner/repo#N` this records, in resources/devopt/prs.json:

  - PR metadata (state, merged, base ref and sha, head sha, merge commit sha, commit count),
  - the PR's changed files under `.github/` and root package.json / tsconfig*.json,
  - the parents of the merge commit,
  - `after_sha` = the merge commit, and `before_sha` chosen by a fixed rule:
      1. `merge-first-parent`: the merge commit's first parent, if every relevant changed file
         (workflow files, and local action files fetched into either snapshot) differs between it
         and the merge commit as the PR's file status says (added/removed/modified/renamed);
      2. `pr-base`: otherwise the PR's recorded base sha, under the same test
         (rebase merges of multi-commit PRs land earlier commits before the merge commit);
      3. `uncaptured`: neither captures the change; the PR is reported and excluded.
    A PR that changes no relevant file gets `no-tracked-change` (before = first candidate).

Both snapshots are fetched with fetch_snapshots (skipped when cached). Completed PR entries are
not re-fetched, so later runs are offline.

Usage:
  uv run python scripts/resources/fetch_devopt.py [--prs resources/devopt_prs.txt]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "scope")]

from fetch_snapshots import fetch_snapshot  # noqa: E402
from ghlib import Gh, GhError, NotFound, log, utc_now_iso  # noqa: E402
from scopelib import REPOS_DIR, RESOURCES, ROOT_FILES, is_workflow_file, load_manifest, write_json  # noqa: E402

DEFAULT_PRS = RESOURCES / "devopt_prs.txt"
DEFAULT_OUT = RESOURCES / "devopt" / "prs.json"

_PR_LINE = re.compile(r"(?P<id>\S+)\s+(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)#(?P<num>\d+)")


def parse_pr_list(text: str) -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _PR_LINE.fullmatch(line)
        if m is None:
            raise ValueError(f"malformed PR line: {line!r}")
        out.append((m.group("id"), m.group("repo"), int(m.group("num"))))
    return sorted(out)


def is_tracked_change(path: str) -> bool:
    """Changed files that can affect a workflow's lowering or contracts."""
    return path.startswith(".github/") or path in ROOT_FILES


def captures(changed: list[dict], before: dict, after: dict) -> bool | None:
    """Does the snapshot pair (before, after) show every relevant change of the PR?

    A change is relevant when it touches a workflow file or a file fetched into either snapshot
    (a local action or a root file); others (e.g. .github/dependabot.yml) are ignored. Returns
    None when the PR has no relevant change at all.
    """
    bf, af = before.get("files", {}), after.get("files", {})

    def oid(files: dict, path: str) -> str | None:
        return (files.get(path) or {}).get("oid")

    relevant = 0
    for ch in changed:
        new = ch["filename"]
        old = ch.get("previous_filename") or new
        if not any(is_workflow_file(p) or p in bf or p in af for p in (old, new)):
            continue
        relevant += 1
        status = ch["status"]
        if status == "added":
            ok = oid(bf, new) is None and oid(af, new) is not None
        elif status == "removed":
            ok = oid(bf, old) is not None and oid(af, old) is None
        elif status == "renamed":
            ok = oid(bf, old) is not None and oid(af, new) is not None and oid(bf, new) is None
        else:  # modified / changed
            ok = None not in (oid(bf, new), oid(af, new)) and oid(bf, new) != oid(af, new)
        if not ok:
            return False
    return True if relevant else None


def fetch_pr(gh: Gh, pid: str, repo: str, num: int) -> dict:
    pr = gh.rest(f"repos/{repo}/pulls/{num}")
    entry: dict = {
        "id": pid, "repo": repo, "number": num, "title": pr.get("title"),
        "state": pr.get("state"), "merged": bool(pr.get("merged")), "merged_at": pr.get("merged_at"),
        "base_ref": (pr.get("base") or {}).get("ref"), "base_sha": (pr.get("base") or {}).get("sha"),
        "head_sha": (pr.get("head") or {}).get("sha"), "merge_commit_sha": pr.get("merge_commit_sha"),
        "commits": pr.get("commits"), "changed_files_total": pr.get("changed_files"),
        "retrieved_at": utc_now_iso(),
    }
    files = gh.rest_paginate(f"repos/{repo}/pulls/{num}/files")
    entry["changed_files"] = sorted(
        ({k: f[k] for k in ("filename", "status", "previous_filename") if f.get(k) is not None}
         for f in files if is_tracked_change(f["filename"])
         or is_tracked_change(f.get("previous_filename") or "")),
        key=lambda f: f["filename"])
    if entry["merged"] and entry["merge_commit_sha"]:
        c = gh.rest(f"repos/{repo}/git/commits/{entry['merge_commit_sha']}")
        entry["merge_parents"] = [p["sha"] for p in c.get("parents", [])]
    return entry


def choose_before(gh: Gh | None, entry: dict, repos_dir: Path) -> None:
    """Apply the before-commit rule, fetching snapshots as needed (gh=None: cached only)."""
    repo, after = entry["repo"], entry["merge_commit_sha"]

    def snap(sha: str) -> dict:
        if gh is None:
            m = load_manifest(repo, sha, repos_dir)
            if m is None:
                raise GhError(f"{repo}@{sha}: snapshot not cached")
            return m
        return fetch_snapshot(gh, repo, sha, repos_dir)[0]

    after_m = snap(after)
    candidates: list[tuple[str, str]] = []
    if entry.get("merge_parents"):
        candidates.append(("merge-first-parent", entry["merge_parents"][0]))
    if entry.get("base_sha") and all(entry["base_sha"] != c[1] for c in candidates):
        candidates.append(("pr-base", entry["base_sha"]))
    entry["after_sha"] = after
    entry["before_rule"], entry["before_sha"] = "uncaptured", None
    for rule, sha in candidates:
        verdict = captures(entry["changed_files"], snap(sha), after_m)
        if verdict is None:
            entry["before_rule"], entry["before_sha"] = "no-tracked-change", sha
            return
        if verdict:
            entry["before_rule"], entry["before_sha"] = rule, sha
            return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prs", type=Path, default=DEFAULT_PRS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--repos-dir", type=Path, default=REPOS_DIR)
    args = ap.parse_args()

    prs = parse_pr_list(args.prs.read_text(encoding="utf-8"))
    db: dict = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    gh = Gh()
    failed = 0
    for pid, repo, num in prs:
        entry = db.get(pid)
        if entry and entry.get("complete") and entry["repo"] == repo and entry["number"] == num:
            continue
        try:
            entry = fetch_pr(gh, pid, repo, num)
            if entry["merged"] and entry.get("merge_commit_sha"):
                choose_before(gh, entry, args.repos_dir)
            else:
                entry["before_rule"], entry["before_sha"], entry["after_sha"] = "not-merged", None, None
            entry["complete"] = True
        except (GhError, NotFound) as e:
            failed += 1
            log(f"[devopt] {pid} {repo}#{num} FAILED: {str(e)[:300]}")
            continue
        db[pid] = entry
        write_json(args.out, db)
        log(f"[devopt] {pid} {repo}#{num}: {entry['before_rule']} "
            f"({len(entry['changed_files'])} tracked changed files)")
    write_json(args.out, db)
    rules: dict[str, int] = {}
    for pid in sorted(db):
        rules[db[pid]["before_rule"]] = rules.get(db[pid]["before_rule"], 0) + 1
    print(json.dumps({"prs": len(prs), "failed": failed, "before_rules": rules,
                      "api_calls": gh.calls}, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
