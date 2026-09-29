#!/usr/bin/env python3
"""Fetch repository snapshots for every repository in one or more scope files.

For each `owner/repo@sha` (lines without a sha are pinned to the last default-branch commit before
the cutoff, cached in resources/pins.json) this writes, under resources/repos/<owner>__<repo>/<sha>/:

  - every `.github/workflows/*.yml|yaml`,
  - every local action (`uses: ./dir` -> dir/action.yml|yaml) referenced by those workflows,
    transitively through composite actions,
  - every local reusable workflow they reference (these live in .github/workflows as well),
  - root package.json, tsconfig.json, tsconfig.build.json when present,
  - manifest.json: file hashes, root directory listing, local references, and fetch status.

Snapshots whose manifest says `complete: true` are skipped, so the script resumes after
interruptions and later runs are offline.

Usage:
  uv run python scripts/resources/fetch_snapshots.py --scope scope/dev.scope [--scope ...] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "scope")]

from ghlib import Gh, GhError, NotFound, gql_str, log, utc_now_iso  # noqa: E402
from scopelib import (  # noqa: E402
    CUTOFF,
    REPOS_DIR,
    RESOURCES,
    ROOT_FILES,
    WORKFLOWS_DIR,
    ScopeLine,
    action_file_candidates,
    git_blob_oid,
    is_workflow_file,
    load_yaml,
    local_refs_in_action,
    local_refs_in_workflow,
    read_scope,
    safe_relpath,
    sha256_bytes,
    snapshot_dir,
    write_json,
)

PINS_FILE = RESOURCES / "pins.json"
SYMLINK_MODE = 0o120000

_BLOB = "fragment B on Blob { oid byteSize isBinary isTruncated text }"


# ---------------------------------------------------------------------------
# Pinning
# ---------------------------------------------------------------------------


def resolve_pin(gh: Gh, repo: str, pins_file: Path = PINS_FILE) -> str | None:
    pins = json.loads(pins_file.read_text(encoding="utf-8")) if pins_file.exists() else {}
    if repo in pins:
        return pins[repo].get("sha")
    owner, name = repo.split("/", 1)
    q = (
        "query { rateLimit { cost remaining resetAt } "
        f"repository(owner: {gql_str(owner)}, name: {gql_str(name)}) {{ nameWithOwner "
        "defaultBranchRef { name target { ... on Commit { "
        f"history(first: 1, until: {gql_str(CUTOFF)}) {{ nodes {{ oid committedDate }} }} }} }} }} }} }}"
    )
    data = gh.graphql(q).get("data") or {}
    r = data.get("repository") or {}
    ref = r.get("defaultBranchRef") or {}
    hist = ((ref.get("target") or {}).get("history") or {}).get("nodes") or []
    pins[repo] = {"branch": ref.get("name"), "cutoff": CUTOFF,
                  "sha": hist[0]["oid"] if hist else None,
                  "committed_date": hist[0]["committedDate"] if hist else None}
    write_json(pins_file, pins)
    return pins[repo]["sha"]


# ---------------------------------------------------------------------------
# Snapshot fetching
# ---------------------------------------------------------------------------


def _blob_or_none(obj: dict | None) -> dict | None:
    """A GraphQL object that is a Blob (has 'oid' and 'byteSize'), else None."""
    if isinstance(obj, dict) and "byteSize" in obj:
        return obj
    return None


class SnapshotFetcher:
    def __init__(self, gh: Gh, repo: str, sha: str, repos_dir: Path = REPOS_DIR):
        self.gh, self.repo, self.sha = gh, repo, sha
        self.owner, self.name = repo.split("/", 1)
        self.dir = snapshot_dir(repo, sha, repos_dir)
        self.files: dict[str, dict] = {}
        self.skipped: list[dict] = []

    # -- content ----------------------------------------------------------

    def _store(self, rel: str, blob: dict) -> bool:
        path = safe_relpath(rel)
        if path is None or path == "":
            self.skipped.append({"path": rel, "reason": "unsafe path"})
            return False
        if blob.get("isBinary"):
            self.skipped.append({"path": path, "reason": "binary"})
            return False
        data = None if blob.get("isTruncated") else (blob.get("text") or "").encode("utf-8")
        if data is None or (blob.get("oid") and git_blob_oid(data) != blob["oid"]):
            # Truncated text, or text whose escapes did not survive the JSON round trip through
            # `gh`: fetch the exact bytes (base64 blob API by oid, else the raw contents API).
            if blob.get("oid"):
                data = self.gh.rest_blob(self.repo, blob["oid"])
            else:
                quoted = urllib.parse.quote(path)
                data = self.gh.rest_raw(f"repos/{self.repo}/contents/{quoted}?ref={self.sha}")
        if blob.get("oid") and git_blob_oid(data) != blob["oid"]:
            raise GhError(f"{self.repo}@{self.sha}:{path}: content does not match blob {blob['oid']}")
        out = self.dir / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        self.files[path] = {"bytes": len(data), "oid": blob.get("oid"), "sha256": sha256_bytes(data)}
        return True

    def _text(self, rel: str) -> str:
        return (self.dir / rel).read_text(encoding="utf-8", errors="replace")

    # -- queries ------------------------------------------------------------

    def _query_base(self) -> dict:
        s = self.sha
        extra = " ".join(f"f{i}: object(expression: {gql_str(f'{s}:{p}')}) {{ ...B }}"
                         for i, p in enumerate(ROOT_FILES))
        q = (
            "query($owner: String!, $name: String!) { rateLimit { cost remaining resetAt } "
            "repository(owner: $owner, name: $name) { nameWithOwner "
            f"commit: object(oid: {gql_str(s)}) {{ ... on Commit {{ oid committedDate }} }} "
            f"root: object(expression: {gql_str(s + ':')}) {{ ... on Tree {{ entries {{ name type mode }} }} }} "
            f"workflows: object(expression: {gql_str(f'{s}:{WORKFLOWS_DIR}')}) "
            "{ ... on Tree { entries { name type mode object { ...B } } } } "
            f"{extra} }} }} {_BLOB}"
        )
        return self.gh.graphql(q, {"owner": self.owner, "name": self.name})

    def _query_paths(self, paths: list[str]) -> dict[str, dict | None]:
        """Fetch blobs for repository paths in one query; missing paths map to None."""
        result: dict[str, dict | None] = {}
        for i in range(0, len(paths), 40):
            chunk = paths[i:i + 40]
            fields = " ".join(f"p{j}: object(expression: {gql_str(f'{self.sha}:{p}')}) {{ ...B }}"
                              for j, p in enumerate(chunk))
            q = ("query($owner: String!, $name: String!) { rateLimit { cost remaining resetAt } "
                 f"repository(owner: $owner, name: $name) {{ {fields} }} }} {_BLOB}")
            data = (self.gh.graphql(q, {"owner": self.owner, "name": self.name}).get("data") or {})
            repo = data.get("repository") or {}
            for j, p in enumerate(chunk):
                result[p] = _blob_or_none(repo.get(f"p{j}"))
        return result

    # -- driver --------------------------------------------------------------

    def fetch(self) -> dict:
        resp = self._query_base()
        data = resp.get("data") or {}
        repo = data.get("repository")
        manifest: dict = {"repo": self.repo, "sha": self.sha, "cutoff": CUTOFF,
                          "retrieved_at": utc_now_iso(), "complete": True}
        if repo is None:
            errs = resp.get("errors") or []
            if any(isinstance(e, dict) and e.get("type") == "NOT_FOUND" for e in errs):
                manifest["status"] = "repo-not-found"
                return self._finish(manifest)
            raise GhError(f"{self.repo}: unexpected GraphQL response: {json.dumps(errs)[:300]}")
        manifest["resolved_name"] = repo.get("nameWithOwner")
        commit = repo.get("commit")
        if not isinstance(commit, dict) or not commit.get("oid"):
            manifest["status"] = "commit-not-found"
            return self._finish(manifest)
        manifest["committed_date"] = commit.get("committedDate")
        manifest["status"] = "ok"

        root = repo.get("root") or {}
        manifest["root_entries"] = sorted(
            ({"name": e["name"], "type": e["type"]} for e in root.get("entries") or []),
            key=lambda e: e["name"])

        workflows: list[str] = []
        for e in sorted((repo.get("workflows") or {}).get("entries") or [], key=lambda e: e["name"]):
            rel = f"{WORKFLOWS_DIR}/{e['name']}"
            if e.get("type") != "blob" or not is_workflow_file(rel):
                continue
            if int(e.get("mode") or 0) == SYMLINK_MODE:
                self.skipped.append({"path": rel, "reason": "symlink"})
                continue
            blob = _blob_or_none(e.get("object"))
            if blob is not None and self._store(rel, blob):
                workflows.append(rel)
        manifest["workflows"] = workflows

        for i, p in enumerate(ROOT_FILES):
            blob = _blob_or_none(repo.get(f"f{i}"))
            if blob is not None:
                self._store(p, blob)

        # Local references: reusable workflows are already fetched with the workflows directory;
        # local actions are fetched transitively through composite actions.
        yaml_errors: list[str] = []
        unresolved: set[str] = set()
        local_workflows: dict[str, bool] = {}
        pending: set[str] = set()
        for wf in workflows:
            doc = load_yaml(self._text(wf))
            if doc is None:
                yaml_errors.append(wf)
                continue
            refs = local_refs_in_workflow(doc)
            unresolved.update(refs.unresolved)
            pending.update(refs.actions)
            for w in refs.workflows:
                local_workflows[w] = w in self.files
        local_actions: dict[str, str | None] = {}
        while pending:
            batch = sorted(pending - set(local_actions))
            pending = set()
            if not batch:
                break
            cands = [c for d in batch for c in action_file_candidates(d)]
            blobs = self._query_paths(cands)
            for d in batch:
                found = None
                for c in action_file_candidates(d):
                    if blobs.get(c) is not None and self._store(c, blobs[c]):
                        found = c
                        break
                local_actions[d] = found
                if found is not None:
                    doc = load_yaml(self._text(found))
                    if doc is None:
                        yaml_errors.append(found)
                        continue
                    refs = local_refs_in_action(doc)
                    unresolved.update(refs.unresolved)
                    pending.update(a for a in refs.actions if a not in local_actions)

        manifest["local_actions"] = dict(sorted(local_actions.items()))
        manifest["local_workflows"] = dict(sorted(local_workflows.items()))
        manifest["unresolved_uses"] = sorted(unresolved)
        manifest["yaml_errors"] = sorted(set(yaml_errors))
        return self._finish(manifest)

    def _finish(self, manifest: dict) -> dict:
        manifest["files"] = dict(sorted(self.files.items()))
        manifest["skipped"] = sorted(self.skipped, key=lambda s: (s["path"], s["reason"]))
        write_json(self.dir / "manifest.json", manifest)
        return manifest


def fetch_snapshot(gh: Gh, repo: str, sha: str, repos_dir: Path = REPOS_DIR, force: bool = False) -> tuple[dict, bool]:
    """Fetch one snapshot unless a complete manifest exists. Returns (manifest, fetched_now)."""
    mpath = snapshot_dir(repo, sha, repos_dir) / "manifest.json"
    if mpath.exists() and not force:
        m = json.loads(mpath.read_text(encoding="utf-8"))
        if m.get("complete"):
            return m, False
    return SnapshotFetcher(gh, repo, sha, repos_dir).fetch(), True


def snapshots_for_scope(gh: Gh | None, lines: list[ScopeLine]) -> list[tuple[str, str]]:
    """Unique (repo, sha) pairs of a scope, pinning unpinned lines (needs `gh` only then)."""
    pairs: set[tuple[str, str]] = set()
    for ln in lines:
        sha = ln.sha
        if sha is None:
            if gh is None:
                raise GhError(f"unpinned scope line needs network: {ln}")
            sha = resolve_pin(gh, ln.repo)
            if sha is None:
                log(f"[snap] {ln.repo}: no commit before cutoff; skipped")
                continue
        pairs.add((ln.repo, sha))
    return sorted(pairs, key=lambda p: (p[0].lower(), p[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scope", type=Path, action="append", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="refetch even complete snapshots")
    ap.add_argument("--repos-dir", type=Path, default=REPOS_DIR)
    args = ap.parse_args()

    gh = Gh()
    lines = [ln for s in args.scope for ln in read_scope(s)]
    pairs = snapshots_for_scope(gh, lines)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    stats = {"snapshots": len(pairs), "fetched": 0, "cached": 0, "ok": 0, "failed": 0,
             "not_found": 0, "workflows": 0, "local_actions": 0}
    for k, (repo, sha) in enumerate(pairs, 1):
        try:
            m, fetched = fetch_snapshot(gh, repo, sha, args.repos_dir, args.force)
        except (GhError, NotFound) as e:
            stats["failed"] += 1
            log(f"[snap] {k}/{len(pairs)} {repo}@{sha[:10]} FAILED: {str(e)[:300]}")
            continue
        stats["fetched" if fetched else "cached"] += 1
        if m.get("status") == "ok":
            stats["ok"] += 1
            stats["workflows"] += len(m.get("workflows", []))
            stats["local_actions"] += sum(1 for v in m.get("local_actions", {}).values() if v)
        else:
            stats["not_found"] += 1
        if fetched:
            log(f"[snap] {k}/{len(pairs)} {repo}@{sha[:10]} {m.get('status')} "
                f"workflows={len(m.get('workflows', []))} actions={len(m.get('local_actions', {}))}")
    # Scope lines naming a workflow must exist in their snapshot.
    missing = [str(ln) for ln in lines if ln.path and ln.sha
               and not (snapshot_dir(ln.repo, ln.sha, args.repos_dir) / ln.path).is_file()]
    for s in sorted(missing):
        log(f"[snap] missing workflow file: {s}")
    stats["missing_workflow_files"] = len(missing)
    stats["api_calls"] = gh.calls
    print(json.dumps(stats, sort_keys=True))
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
