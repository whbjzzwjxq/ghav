"""Offline tests for the benchmark scripts in scripts/scope and scripts/resources."""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts" / "scope"), str(ROOT / "scripts" / "resources")]

import build_devopt_scope  # noqa: E402
import build_eval_scope  # noqa: E402
import build_pool_scope  # noqa: E402
import build_random_scope  # noqa: E402
import build_workflows_scope  # noqa: E402
import fetch_devopt  # noqa: E402
import fetch_pool  # noqa: E402
import fetch_snapshots  # noqa: E402
import ghlib  # noqa: E402
import scopelib  # noqa: E402
from scopelib import ScopeLine  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "scripts"
REPOS = FIX / "repos"
SHA1 = "1" * 40
ACME = ScopeLine("acme/app", SHA1)


# ---------------------------------------------------------------------------
# scopelib
# ---------------------------------------------------------------------------


def test_scope_line_roundtrip_and_id():
    s = f"acme/app@{SHA1}:.github/workflows/ci.yml"
    ln = ScopeLine.parse(s)
    assert (ln.repo, ln.sha, ln.path) == ("acme/app", SHA1, ".github/workflows/ci.yml")
    assert str(ln) == s
    assert ln.benchmark_id() == "acme__app__ci"
    assert str(ScopeLine.parse("acme/app")) == "acme/app"
    assert ScopeLine.parse(f"acme/app@{SHA1}").path is None
    with pytest.raises(ValueError):
        ScopeLine.parse("not a line")
    with pytest.raises(ValueError):
        ScopeLine.parse("acme/app@abc")  # short sha is rejected


def test_read_and_format_scope(tmp_path):
    p = tmp_path / "x.scope"
    p.write_text(f"# comment\n\nb/b@{SHA1}\na/a\nb/b@{SHA1}\n")
    lines = scopelib.read_scope(p)
    assert [str(x) for x in lines] == [f"b/b@{SHA1}", "a/a", f"b/b@{SHA1}"]
    assert scopelib.format_scope(lines) == f"a/a\nb/b@{SHA1}\n"
    assert scopelib.format_scope([]) == ""


@pytest.mark.parametrize("text,expected", [
    ("on: push\njobs: {}", {"push"}),
    ("on: [pull_request, workflow_dispatch]\njobs: {}", {"pull_request", "workflow_dispatch"}),
    ("on:\n  push:\n    branches: [main]\n  schedule: []\n", {"push", "schedule"}),
    ("'on': workflow_call\n", {"workflow_call"}),
    ("name: x\n", set()),
])
def test_workflow_triggers(text, expected):
    doc = scopelib.load_yaml(text)
    assert scopelib.workflow_triggers(doc) == expected
    assert scopelib.triggers_core_event(doc) == bool(expected & {"push", "pull_request"})


def test_load_yaml_error_returns_none():
    assert scopelib.load_yaml("jobs: [unclosed") is None


@pytest.mark.parametrize("uses,expected", [
    ("./.github/actions/setup", ".github/actions/setup"),
    ("./.github/actions/setup/", ".github/actions/setup"),
    ("./", ""),
    ("./../outside", None),
    ("./${{ matrix.action }}", None),
    ("actions/checkout@v4", None),
])
def test_normalize_local_ref(uses, expected):
    assert scopelib.normalize_local_ref(uses) == expected


def test_local_refs_and_closure():
    snap = REPOS / "acme__app" / SHA1
    doc = scopelib.load_yaml((snap / ".github/workflows/ci.yml").read_text())
    refs = scopelib.local_refs_in_workflow(doc)
    assert refs.actions == (".github/actions/setup",)
    assert refs.workflows == (".github/workflows/reusable.yml",)
    actions, workflows = scopelib.local_closure(snap, ".github/workflows/ci.yml")
    assert actions == {".github/actions/setup", ".github/actions/node"}
    assert workflows == {".github/workflows/reusable.yml"}
    assert scopelib.find_action_file(snap, ".github/actions/node") == ".github/actions/node/action.yaml"


def test_git_blob_oid_matches_git():
    # `printf 'hello\n' | git hash-object --stdin`
    assert scopelib.git_blob_oid(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


# ---------------------------------------------------------------------------
# Scope builders
# ---------------------------------------------------------------------------


def test_build_pool_excludes_dev_and_unpinned():
    pool = json.loads((FIX / "pool.json").read_text())
    dev = scopelib.read_scope(FIX / "dev.scope")
    lines, summary = build_pool_scope.build_pool(pool, dev)
    assert [str(x) for x in lines] == [f"Big/Star@{'a' * 40}", f"acme/app@{SHA1}", f"gone/repo@{'2' * 40}"]
    assert summary["dev_overlap"] == ["Dev/Repo"]          # case-insensitive match against dev.scope
    assert summary["unpinned"] == ["new/repo"]
    lines2, _ = build_pool_scope.build_pool(pool, dev, size=2)
    assert [x.repo for x in lines2] == ["Big/Star", "acme/app"]


def test_build_workflows_filters_triggers_and_yaml():
    gone = ScopeLine("gone/repo", "2" * 40)
    lines, summary = build_workflows_scope.build_workflows([gone, ACME], REPOS)
    assert [x.path for x in lines] == [".github/workflows/ci.yml", ".github/workflows/list.yaml"]
    assert summary["workflow_files"] == 5
    assert summary["yaml_errors"] == 1
    assert summary["no_core_trigger"] == 2                 # nightly (schedule), reusable (workflow_call)
    assert summary["repos_not_ok"] == [f"gone/repo@{'2' * 40} (repo-not-found)"]
    assert summary["lowering_checked"] is False


def test_build_workflows_lowering_hook_and_missing_snapshot():
    def reject_list(wf: Path, root: Path) -> str | None:
        assert root == REPOS / "acme__app" / SHA1
        return "unsupported" if wf.name == "list.yaml" else None

    lines, summary = build_workflows_scope.build_workflows([ACME], REPOS, reject_list)
    assert [x.path for x in lines] == [".github/workflows/ci.yml"]
    assert list(summary["lowering_errors"].values()) == ["unsupported"]
    _, summary = build_workflows_scope.build_workflows([ScopeLine("big/star", "a" * 40)], REPOS)
    assert summary["repos_missing_snapshot"] == [f"big/star@{'a' * 40}"]


def _stats(n: int, ratio: float = 0.5, instances: int = 3, frozen: int = 0) -> dict[str, dict]:
    return {f"o/r{i:03d}@{SHA1}:.github/workflows/w.yml":
            {"instances": instances, "frozen": frozen, "redundancy_ratio": ratio} for i in range(n)}


def test_eval_eligibility_boundaries():
    e = build_eval_scope.eligible
    assert e({"instances": 2, "frozen": 1, "redundancy_ratio": 0.2})       # half frozen, ratio at bound
    assert not e({"instances": 3, "frozen": 2, "redundancy_ratio": 0.9})   # more than half frozen
    assert not e({"instances": 1, "frozen": 0, "redundancy_ratio": 0.9})   # single instance
    assert not e({"instances": 4, "frozen": 0, "redundancy_ratio": 0.19})


def test_eval_selection_is_seeded_and_sorted():
    stats = _stats(150)
    stats.update({k.replace("/r", "/low"): {**v, "redundancy_ratio": 0.0} for k, v in _stats(5).items()})
    qualifying, selected = build_eval_scope.select_eval(stats)
    assert len(qualifying) == 150 and len(selected) == 100
    assert selected == sorted(random.Random(42).sample(sorted(qualifying), 100))
    assert selected == build_eval_scope.select_eval(stats)[1]              # deterministic
    small = _stats(7)
    assert build_eval_scope.select_eval(small)[1] == sorted(small)


def test_random_selection():
    stats = {**_stats(40), **{k.replace("/r", "/one"): {**v, "instances": 1} for k, v in _stats(10).items()}}
    pool, sample = build_random_scope.select_random(stats)
    assert len(pool) == 40 and len(sample) == 30
    assert sample == sorted(random.Random(42).sample(pool, 30))


def test_stats_validation_and_collection():
    assert scopelib.validate_stats({"instances": 2, "frozen": 0, "redundancy_ratio": 1}) == \
        {"instances": 2, "frozen": 0, "redundancy_ratio": 1.0}
    for bad in ({"instances": 2}, {"instances": 2, "frozen": 3, "redundancy_ratio": 0.1},
                {"instances": 2, "frozen": True, "redundancy_ratio": 0.1},
                {"instances": 2, "frozen": 0, "redundancy_ratio": 1.5}):
        with pytest.raises(ValueError):
            scopelib.validate_stats(bad)

    def fn(line: str) -> dict:
        if "bad" in line:
            raise RuntimeError("cannot lower")
        return {"instances": 2, "frozen": 0, "redundancy_ratio": 0.5}

    lines = [ScopeLine("o/bad", SHA1, ".github/workflows/a.yml"), ScopeLine("o/good", SHA1, ".github/workflows/a.yml")]
    stats, errors = scopelib.collect_stats(lines, fn)
    assert list(stats) == [str(lines[1])]
    assert errors == {str(lines[0]): "RuntimeError: cannot lower"}


def test_missing_pipeline_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "ghav.pipeline", None)   # forces ImportError
    with pytest.raises(SystemExit, match="ghav.pipeline.workflow_stats"):
        scopelib.load_workflow_stats()


# ---------------------------------------------------------------------------
# DevOpt
# ---------------------------------------------------------------------------


def test_parse_pr_list():
    assert fetch_devopt.parse_pr_list("# x\nD02 b/b#2\nD01 a/a#10\n") == [("D01", "a/a", 10), ("D02", "b/b", 2)]
    with pytest.raises(ValueError):
        fetch_devopt.parse_pr_list("D01 a/a 10")


def test_captures_rules():
    wf = ".github/workflows/ci.yml"
    mod = [{"filename": wf, "status": "modified"}]
    assert fetch_devopt.captures(mod, {"files": {wf: {"oid": "a"}}}, {"files": {wf: {"oid": "b"}}}) is True
    assert fetch_devopt.captures(mod, {"files": {wf: {"oid": "a"}}}, {"files": {wf: {"oid": "a"}}}) is False
    add = [{"filename": wf, "status": "added"}]
    assert fetch_devopt.captures(add, {"files": {}}, {"files": {wf: {"oid": "b"}}}) is True
    rem = [{"filename": wf, "status": "removed"}]
    assert fetch_devopt.captures(rem, {"files": {wf: {"oid": "a"}}}, {"files": {}}) is True
    ren = [{"filename": ".github/workflows/new.yml", "previous_filename": wf, "status": "renamed"}]
    assert fetch_devopt.captures(ren, {"files": {wf: {"oid": "a"}}},
                                 {"files": {".github/workflows/new.yml": {"oid": "a"}}}) is True
    other = [{"filename": ".github/dependabot.yml", "status": "modified"}]
    assert fetch_devopt.captures(other, {"files": {}}, {"files": {}}) is None


def _write_snapshot(repos: Path, repo: str, sha: str, files: dict[str, str]) -> dict:
    snap = scopelib.snapshot_dir(repo, sha, repos)
    meta = {}
    for rel, text in files.items():
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap / rel).write_text(text)
        meta[rel] = {"oid": scopelib.git_blob_oid(text.encode())}
    m = {"repo": repo, "sha": sha, "status": "ok", "complete": True, "files": meta,
         "workflows": sorted(f for f in files if scopelib.is_workflow_file(f))}
    scopelib.write_json(snap / "manifest.json", m)
    return m


def test_build_devopt_pairs(tmp_path):
    before, after = "b" * 40, "c" * 40
    base = {
        ".github/workflows/ci.yml": "on: push\njobs:\n  a:\n    runs-on: x\n    steps:\n      - uses: ./.github/actions/s\n",
        ".github/workflows/other.yml": "on: pull_request\njobs:\n  a:\n    runs-on: x\n    steps:\n      - run: echo\n",
        ".github/workflows/call.yml": "on: workflow_call\njobs:\n  a:\n    runs-on: x\n    steps:\n      - run: echo\n",
        ".github/actions/s/action.yml": "runs:\n  using: composite\n  steps: []\n",
    }
    _write_snapshot(tmp_path, "o/r", before, base)
    changed = dict(base)
    changed[".github/actions/s/action.yml"] = "runs:\n  using: composite\n  steps:\n    - run: x\n      shell: bash\n"
    changed[".github/workflows/call.yml"] = base[".github/workflows/call.yml"] + "# edit\n"
    changed[".github/workflows/new.yml"] = "on: push\njobs: {}\n"
    _write_snapshot(tmp_path, "o/r", after, changed)
    prs = {
        "D01": {"repo": "o/r", "number": 1, "before_rule": "merge-first-parent", "before_sha": before,
                "after_sha": after, "changed_files": [
                    {"filename": ".github/actions/s/action.yml", "status": "modified"},
                    {"filename": ".github/workflows/call.yml", "status": "modified"},
                    {"filename": ".github/workflows/new.yml", "status": "added"}]},
        "D02": {"repo": "o/r", "number": 2, "before_rule": "uncaptured", "changed_files": []},
    }
    lines, report = build_devopt_scope.build_devopt(prs, tmp_path)
    assert [str(x) for x in lines] == [f"o/r@{before}:.github/workflows/ci.yml"]
    assert report["pairs"][0]["after"] == f"o/r@{after}:.github/workflows/ci.yml"
    assert report["pairs"][0]["reason"] == "local-action-changed"
    reasons = sorted((x["id"], x["reason"]) for x in report["excluded"])
    assert reasons == [("D01", "no push/pull_request trigger"), ("D01", "workflow added"),
                       ("D02", "before rule uncaptured")]
    assert report["prs_with_pairs"] == 1


def test_choose_before_falls_back_to_pr_base(tmp_path):
    wf = ".github/workflows/ci.yml"
    _write_snapshot(tmp_path, "o/r", "a" * 40, {wf: "v1"})      # PR base
    _write_snapshot(tmp_path, "o/r", "b" * 40, {wf: "v2"})      # merge first parent (rebase merge)
    _write_snapshot(tmp_path, "o/r", "c" * 40, {wf: "v2"})      # merge commit
    entry = {"repo": "o/r", "merge_commit_sha": "c" * 40, "merge_parents": ["b" * 40],
             "base_sha": "a" * 40, "changed_files": [{"filename": wf, "status": "modified"}]}
    fetch_devopt.choose_before(None, entry, tmp_path)
    assert (entry["before_rule"], entry["before_sha"], entry["after_sha"]) == ("pr-base", "a" * 40, "c" * 40)


# ---------------------------------------------------------------------------
# Network layer with fake runners
# ---------------------------------------------------------------------------


class FakeRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    def __call__(self, args, stdin):
        self.calls.append(list(args))
        rc, out, err = self.responses.pop(0)
        return ghlib.Completed(rc, out.encode(), err.encode())


def _gh(responses) -> tuple[ghlib.Gh, FakeRunner, list[float]]:
    slept: list[float] = []
    runner = FakeRunner(responses)
    return ghlib.Gh(runner=runner, sleep=slept.append, clock=lambda: 0.0, min_interval=0.0), runner, slept


def test_gh_retries_rate_limit_then_succeeds():
    gh, runner, slept = _gh([(1, "", "HTTP 403: API rate limit exceeded"), (0, '{"ok": 1}', "")])
    assert gh.rest("repos/o/r", {"b": 2, "a": 1}) == {"ok": 1}
    assert runner.calls[0] == ["gh", "api", "-X", "GET", "repos/o/r", "-f", "a=1", "-f", "b=2"]
    assert slept and slept[0] >= 60


def test_gh_not_found_and_fatal():
    gh, _, _ = _gh([(1, "", "HTTP 404: Not Found")])
    with pytest.raises(ghlib.NotFound):
        gh.rest("repos/o/missing")
    gh, _, _ = _gh([(1, "", "HTTP 422: Validation Failed")])
    with pytest.raises(ghlib.GhError):
        gh.rest("search/repositories")


def test_graphql_failure_and_low_budget():
    gh, _, _ = _gh([(1, '{"errors": [{"message": "Expected NAME"}]}', "")])
    with pytest.raises(ghlib.GhError, match="Expected NAME"):
        gh.graphql("query { x }")
    low = {"data": {"rateLimit": {"remaining": 3, "resetAt": "2000-01-01T00:00:00Z"}}}
    gh, _, slept = _gh([(0, json.dumps(low), "")])
    assert gh.graphql("query { x }")["data"]["rateLimit"]["remaining"] == 3
    assert slept == [5.0]   # reset already passed: minimal wait


def test_search_pool_dedupes_and_ranks():
    class G:
        def rest(self, path, params):
            page = params["page"]
            items = {1: [{"full_name": "a/a"}, {"full_name": "b/b"}], 2: [{"full_name": "B/B"}, {"full_name": "c/c"}]}
            return {"total_count": 4, "items": items.get(page, [])}

    repos, total = fetch_pool.search_pool(G(), size=3)
    assert [(r["rank"], r["full_name"]) for r in repos] == [(1, "a/a"), (2, "b/b"), (3, "c/c")]
    assert total == 4


def test_resolve_pins_positional():
    class G:
        def graphql(self, q, variables=None):
            assert '"x1", "x2"' in q
            return {"data": {"nodes": [
                {"id": "new1", "nameWithOwner": "a/a", "defaultBranchRef": {"name": "main", "target": {
                    "history": {"nodes": [{"oid": "f" * 40, "committedDate": "2026-08-31T00:00:00Z"}]}}}},
                None]}}

    repos = [{"node_id": "x1", "full_name": "a/a"}, {"node_id": "x2", "full_name": "b/b"}]
    fetch_pool.resolve_pins(G(), repos)
    assert repos[0]["pinned_sha"] == "f" * 40 and "pin_error" not in repos[0]
    assert repos[1]["pinned_sha"] is None and repos[1]["pin_error"] == fetch_pool.UNRESOLVABLE


class FakeGraphGh:
    """Serves snapshot queries from an in-memory repository."""

    def __init__(self, files: dict[str, str]):
        self.files = files
        self.queries = 0

    def _blob(self, path):
        if path not in self.files:
            return None
        data = self.files[path].encode()
        return {"oid": scopelib.git_blob_oid(data), "byteSize": len(data), "isBinary": False,
                "isTruncated": False, "text": self.files[path]}

    def graphql(self, q, variables=None):
        self.queries += 1
        if "workflows: object" in q:
            wf = sorted(p for p in self.files if p.startswith(".github/workflows/"))
            repo = {"nameWithOwner": "acme/app",
                    "commit": {"oid": SHA1, "committedDate": "2026-08-30T00:00:00Z"},
                    "root": {"entries": [{"name": "package.json", "type": "blob", "mode": 33188}]},
                    "workflows": {"entries": [{"name": p.rsplit("/", 1)[1], "type": "blob", "mode": 33188,
                                               "object": self._blob(p)} for p in wf]}}
            for i, p in enumerate(scopelib.ROOT_FILES):
                repo[f"f{i}"] = self._blob(p)
            return {"data": {"repository": repo}}
        repo = {}
        for alias, path in re.findall(r'(p\d+): object\(expression: "[0-9a-f]{40}:([^"]+)"\)', q):
            repo[alias] = self._blob(path)
        return {"data": {"repository": repo}}

    def rest_raw(self, path):  # pragma: no cover - not reached with untruncated blobs
        raise AssertionError(path)


def test_snapshot_fetcher_writes_closure_and_skips_complete(tmp_path):
    src = REPOS / "acme__app" / SHA1
    files = {p.relative_to(src).as_posix(): p.read_text() for p in sorted(src.rglob("*"))
             if p.is_file() and p.name != "manifest.json"}
    files["README.md"] = "not fetched"
    gh = FakeGraphGh(files)
    m, fetched = fetch_snapshots.fetch_snapshot(gh, "acme/app", SHA1, tmp_path)
    assert fetched and m["status"] == "ok" and m["complete"]
    assert m["workflows"] == [".github/workflows/broken.yml", ".github/workflows/ci.yml",
                              ".github/workflows/list.yaml", ".github/workflows/nightly.yml",
                              ".github/workflows/reusable.yml"]
    assert m["local_actions"] == {".github/actions/node": ".github/actions/node/action.yaml",
                                  ".github/actions/setup": ".github/actions/setup/action.yml"}
    assert m["local_workflows"] == {".github/workflows/reusable.yml": True}
    assert m["yaml_errors"] == [".github/workflows/broken.yml"]
    assert "README.md" not in m["files"] and "package.json" in m["files"]
    snap = scopelib.snapshot_dir("acme/app", SHA1, tmp_path)
    assert (snap / ".github/actions/node/action.yaml").read_text() == files[".github/actions/node/action.yaml"]
    # Same manifest as the committed fixture, except provenance fields.
    fixture = json.loads((src / "manifest.json").read_text())
    for key in ("files", "workflows", "local_actions", "local_workflows", "yaml_errors"):
        assert m[key] == fixture[key], key
    q = gh.queries
    m2, fetched2 = fetch_snapshots.fetch_snapshot(gh, "acme/app", SHA1, tmp_path)
    assert not fetched2 and gh.queries == q and m2 == m


def test_snapshot_fetcher_repo_not_found(tmp_path):
    class G:
        def graphql(self, q, variables=None):
            return {"data": {"repository": None}, "errors": [{"type": "NOT_FOUND"}]}

    m, _ = fetch_snapshots.fetch_snapshot(G(), "gone/repo", SHA1, tmp_path)
    assert m["status"] == "repo-not-found" and m["complete"] and m["files"] == {}


def test_scripts_do_not_import_network_in_scope_builders():
    for p in sorted((ROOT / "scripts" / "scope").glob("*.py")):
        text = p.read_text()
        assert not re.search(r"^\s*(from|import) (ghlib|subprocess|fetch_)", text, re.M), p.name
