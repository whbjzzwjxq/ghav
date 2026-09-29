"""Offline tests for historical timings (scripts/resources/fetch_timings.py) and local traces
(scripts/traces/trace_workflow.py), plus checks on the committed Zustand dev-set resources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts" / "scope"), str(ROOT / "scripts" / "resources"), str(ROOT / "scripts" / "traces")]

import fetch_timings as ft  # noqa: E402
import ghlib  # noqa: E402
import trace_workflow as tw  # noqa: E402
from scopelib import ScopeLine  # noqa: E402

from ghav import cost  # noqa: E402
from ghav.wir import TRUE, Binding, Contract, Expr, JobInstance, Program, Step  # noqa: E402

SHA = "a" * 40
LINE = ScopeLine("acme/app", SHA, ".github/workflows/ci.yml")


def gstep(number, name, start, end, conclusion="success"):
    return {"number": number, "name": name, "status": "completed", "conclusion": conclusion,
            "started_at": f"2026-08-01T00:{start}Z", "completed_at": f"2026-08-01T00:{end}Z"}


def gjob(jid, name, steps, start="00:00", end="01:00", conclusion="success", run_id=1):
    return {"id": jid, "run_id": run_id, "name": name, "conclusion": conclusion, "status": "completed",
            "started_at": f"2026-08-01T00:{start}Z", "completed_at": f"2026-08-01T00:{end}Z",
            "labels": ["ubuntu-latest"], "steps": steps, "html_url": "https://x", "url": "https://y"}


def standard_steps(test_end="00:50"):
    return [
        gstep(1, "Set up job", "00:01", "00:03"),
        gstep(2, "Run actions/checkout@v4", "00:03", "00:05"),
        gstep(3, "Setup Node", "00:05", "00:10"),
        gstep(4, "Run npm ci", "00:10", "00:30"),
        gstep(5, "Run npm test", "00:30", test_end),
        gstep(6, "Deploy", "00:50", "00:50", conclusion="skipped"),
        gstep(10, "Post Setup Node", "00:50", "00:54"),
        gstep(11, "Post Run actions/checkout@v4", "00:54", "00:55"),
        gstep(12, "Complete job", "00:55", "00:56"),
    ]


# ---------------------------------------------------------------------------
# fetch_timings: pure derivation
# ---------------------------------------------------------------------------


def test_classify_steps_pairs_hooks_and_infra():
    mains, infra = ft.classify_steps(standard_steps())
    assert [m["step"]["name"] for m in mains] == [
        "Run actions/checkout@v4", "Setup Node", "Run npm ci", "Run npm test", "Deploy"]
    assert [h["name"] for h in mains[1]["hooks"]] == ["Post Setup Node"]
    assert [h["name"] for h in mains[0]["hooks"]] == ["Post Run actions/checkout@v4"]
    assert [s["name"] for s in infra] == ["Set up job", "Complete job"]


def test_classify_steps_same_name_posts_pair_in_reverse():
    steps = [gstep(2, "Run actions/cache@v4", "00:00", "00:01"), gstep(3, "Run actions/cache@v4", "00:01", "00:02"),
             gstep(8, "Post Run actions/cache@v4", "00:05", "00:15"), gstep(9, "Post Run actions/cache@v4", "00:15", "00:17")]
    mains, _ = ft.classify_steps(steps)
    assert mains[1]["hooks"][0]["number"] == 8 and mains[0]["hooks"][0]["number"] == 9
    # a user step called "Post results" is a main step when no step is called "results"
    mains, _ = ft.classify_steps([gstep(2, "Post results", "00:00", "00:01")])
    assert [m["step"]["name"] for m in mains] == ["Post results"]


def test_job_sample_attributes_hooks_and_overhead():
    s = ft.job_sample(gjob(1, "test (a)", standard_steps()))
    assert s["duration"] == 60.0
    by = {x["name"]: x for x in s["steps"]}
    assert by["Setup Node"]["seconds"] == 5.0 + 4.0          # main + post hook
    assert by["Run actions/checkout@v4"]["seconds"] == 2.0 + 1.0
    assert by["Run npm ci"]["seconds"] == 20.0 and by["Run npm test"]["seconds"] == 20.0
    assert by["Deploy"]["executed"] is False and by["Deploy"]["seconds"] is None
    assert s["overhead"] == 60.0 - (3 + 9 + 20 + 20 + 0)      # set up, complete job and gaps
    assert s["main_steps"] == 5
    assert ft.job_sample(gjob(2, "x", standard_steps(), conclusion="skipped")) is None


def test_summarize_jobs_medians_and_positions():
    runs = {
        2: [gjob(20, "test (a)", standard_steps("00:40"), run_id=2)],
        1: [gjob(10, "test (a)", standard_steps("00:50")), gjob(11, "lint", [], conclusion="skipped")],
    }
    out = ft.summarize_jobs(runs)
    assert sorted(out) == ["lint", "test (a)"]
    t = out["test (a)"]
    assert t["samples"] == 2 and t["durations"] == [60.0, 60.0]
    assert t["main_step_counts"] == [5]
    test_step = next(s for s in t["steps"] if s["name"] == "Run npm test")
    assert test_step["durations"] == [20.0, 10.0]             # run 1 first, then run 2
    assert test_step["seconds"] == 15.0 and test_step["positions"] == [3]
    deploy = next(s for s in t["steps"] if s["name"] == "Deploy")
    assert deploy["seconds"] is None and deploy["skipped"] == 2
    assert [s["name"] for s in t["steps"]][:2] == ["Run actions/checkout@v4", "Setup Node"]
    assert out["lint"]["samples"] == 0 and out["lint"]["skipped"] == 1


def test_count_windows_end_before_cutoff():
    w = ft.count_windows("2026-09-01T00:00:00Z")
    assert w[0] == (30, "2026-08-02T00:00:00Z", "2026-08-31T23:59:59Z")
    assert [d for d, _, _ in w] == list(ft.COUNT_WINDOWS_DAYS)


def test_project_job_drops_urls_and_sorts_steps():
    j = ft.project_job(gjob(1, "x", list(reversed(standard_steps()))))
    assert "url" not in j and "html_url" not in j
    assert [s["number"] for s in j["steps"]] == sorted(s["number"] for s in j["steps"])


# ---------------------------------------------------------------------------
# fetch_timings: network layer with a fake gh
# ---------------------------------------------------------------------------


class RoutingRunner:
    """Answers `gh api` calls from a table keyed by endpoint (and event for runs queries)."""

    def __init__(self, table):
        self.table = table
        self.calls: list[list[str]] = []

    def __call__(self, args, stdin):
        self.calls.append(list(args))
        a = list(args)[2:]
        if a and a[0] == "graphql":
            return ghlib.Completed(0, json.dumps(self.table["graphql"]).encode(), b"")
        path = next(x for x in a if x.startswith("repos/"))
        params = dict(a[i + 1].split("=", 1) for i, x in enumerate(a) if x == "-f")
        key = path.split("?")[0]
        if key.endswith("/runs"):
            key = (key, params.get("event"), "created" in params and ".." in params["created"])
        value = self.table.get(key)
        if value is None:
            return ghlib.Completed(1, b"", b"HTTP 404: Not Found")
        return ghlib.Completed(0, json.dumps(value).encode(), b"")


def fake_table():
    runs = "repos/acme/app/actions/workflows/ci.yml/runs"
    run = lambda i, ev: {"id": i, "event": ev, "head_sha": SHA, "conclusion": "success",  # noqa: E731
                         "html_url": "u", "repository": {"big": 1}}
    return {
        (runs, "push", False): {"total_count": 7, "workflow_runs": [run(2, "push"), run(1, "push")]},
        (runs, "pull_request", False): {"total_count": 0, "workflow_runs": []},
        (runs, "push", True): {"total_count": 30},
        (runs, "pull_request", True): {"total_count": 10},
        "repos/acme/app/actions/runs/1/jobs": [{"total_count": 1, "jobs": [gjob(10, "test (a)", standard_steps())]}],
        "repos/acme/app/actions/runs/2/jobs": [{"total_count": 1, "jobs": [gjob(20, "test (a)", standard_steps("00:40"), run_id=2)]}],
        "graphql": {"data": {"repository": {"v0": {"oid": "f" * 40}}}},
    }


def test_fetcher_writes_cache_and_medians_then_reuses_cache(tmp_path):
    runner = RoutingRunner(fake_table())
    gh = ghlib.Gh(runner=runner, sleep=lambda s: None, clock=lambda: 0.0, min_interval=0.0)
    m = ft.TimingsFetcher(gh, LINE, tmp_path / "timings", tmp_path / "repos").fetch()
    d = tmp_path / "timings" / "acme__app__ci"
    assert sorted(p.name for p in d.iterdir()) == [
        "counts.json", "jobs", "medians.json", "runs_pull_request.json", "runs_push.json", "versions.json"]
    assert m["status"] == "ok"
    assert m["events"]["push"] == {"runs": [2, 1], "window_count": 30}
    assert m["events"]["pull_request"] == {"runs": [], "window_count": 10}
    assert m["count_window"]["days"] == 30 and m["count_window"]["capped"] is False
    assert m["jobs"]["test (a)"]["samples"] == 2
    raw = json.loads((d / "runs_push.json").read_text())
    assert "html_url" not in raw["workflow_runs"][0] and "repository" not in raw["workflow_runs"][0]
    assert raw["query"]["created"] == "<2026-09-01T00:00:00Z" and raw["query"]["status"] == "success"
    assert m["workflow_versions"]["runs_total"] == 1
    n_calls = len(runner.calls)
    again = ft.TimingsFetcher(None, LINE, tmp_path / "timings", tmp_path / "repos").fetch()  # offline
    assert again == m and len(runner.calls) == n_calls


def test_fetcher_narrows_count_window_when_capped(tmp_path):
    table = fake_table()
    runs = "repos/acme/app/actions/workflows/ci.yml/runs"
    table[(runs, "push", True)] = {"total_count": 2500}
    gh = ghlib.Gh(runner=RoutingRunner(table), sleep=lambda s: None, clock=lambda: 0.0, min_interval=0.0)
    c = ft.TimingsFetcher(gh, LINE, tmp_path, tmp_path).counts()
    assert c["days"] == ft.COUNT_WINDOWS_DAYS[-1] and c["capped"] is True


def test_fetcher_missing_workflow(tmp_path):
    gh = ghlib.Gh(runner=RoutingRunner({}), sleep=lambda s: None, clock=lambda: 0.0, min_interval=0.0)
    m = ft.TimingsFetcher(gh, LINE, tmp_path, tmp_path).fetch()
    assert m["status"] == "workflow-not-found" and m["jobs"] == {}
    with pytest.raises(ghlib.GhError):
        ft.TimingsFetcher(None, ScopeLine("acme/other", SHA, ".github/workflows/x.yml"), tmp_path, tmp_path).fetch()


def test_medians_file_feeds_historical_model(tmp_path):
    gh = ghlib.Gh(runner=RoutingRunner(fake_table()), sleep=lambda s: None, clock=lambda: 0.0, min_interval=0.0)
    ft.TimingsFetcher(gh, LINE, tmp_path, tmp_path).fetch()
    succ = Expr("status", ("success",))
    steps = (
        Step(Binding("uses", "actions/checkout@v4"), succ, Contract(template="checkout"), ("test", 0)),
        Step(Binding("uses", "actions/setup-node@v4", name="Setup Node"), succ,
             Contract(template="setup-runtime"), ("test", 1)),
        Step(Binding("run", "npm ci"), succ, Contract(template="install-pinned"), ("test", 2)),
        Step(Binding("run", "npm test"), succ, Contract(template="test"), ("test", 3)),
        Step(Binding("run", "./deploy.sh", name="Deploy"), succ, Contract(template="top"), ("test", 4)),
    )
    j = JobInstance(id="test[v=a]", orig_job="test", check_name="test (a)", matrix=(("v", "a"),),
                    runs_on="ubuntu-latest", cond=TRUE, needs=frozenset(), steps=steps)
    p = Program(name="ci", on={"push": None, "pull_request": None}, jobs=(j,))
    m = cost.load_historical_model(p, "acme__app__ci", tmp_path)
    assert m.coverage["steps_priced_historical"] == 4 and m.coverage["steps_total"] == 5
    assert m.contexts == (("push", 0.75), ("pull_request", 0.25))
    run = cost.simulate(p, "push", m)["test[v=a]"]
    assert run.step_seconds == (3.0, 9.0, 20.0, 15.0, 30.0)   # Deploy was always skipped -> mock
    # overheads: 8 s (run 1) and 18 s (run 2 has a 10 s gap after its shorter test step)
    assert m.overhead_table == {"ubuntu-latest": 13.0}
    assert run.seconds == 13.0 + 77.0 and run.billed_minutes == 2


# ---------------------------------------------------------------------------
# trace_workflow: pure helpers
# ---------------------------------------------------------------------------


def test_parse_and_select_instance():
    assert tw.parse_instance("env=development, build=cjs") == (None, (("build", "cjs"), ("env", "development")))
    assert tw.parse_instance("t[build=cjs]") == ("t[build=cjs]", ())
    mk = lambda jid, m: JobInstance(id=jid, orig_job=jid.split("[")[0], check_name=jid, matrix=m,  # noqa: E731
                                    runs_on="ubuntu-latest", cond=TRUE, needs=frozenset(), steps=())
    p = Program(name="w", on={}, jobs=(mk("a[b=1]", (("b", "1"),)), mk("c[b=1]", (("b", "1"),))))
    with pytest.raises(SystemExit):
        tw.select_instance(p, "b=1")
    assert tw.select_instance(p, "b=1", job_key="c").id == "c[b=1]"
    assert tw.select_instance(p, "a[b=1]").id == "a[b=1]"


def test_package_manager_field(tmp_path):
    pj = tmp_path / "package.json"
    pj.write_text(json.dumps({"packageManager": "pnpm@11.3.0+sha512.abc"}))
    assert tw.package_manager(pj) == ("pnpm", "11.3.0")
    pj.write_text(json.dumps({"packageManager": "pnpm@1.0; rm -rf /"}))
    assert tw.package_manager(pj) is None
    pj.write_text(json.dumps({"name": "x"}))
    assert tw.package_manager(pj) is None
    assert tw.package_manager(tmp_path / "missing.json") is None


def test_snapshot_diff_and_collapse(tmp_path):
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / ".git").mkdir()
    (ws / ".git" / "HEAD").write_text("ref")
    (ws / "src" / "a.ts").write_text("a")
    (ws / "src" / "b.ts").write_text("b")
    (ws / "node_modules").mkdir()
    (ws / "node_modules" / "x.js").write_text("x")
    snap = tw.Snapshotter(ws)
    before = snap.snapshot()
    assert sorted(before) == ["node_modules/x.js", "src/a.ts", "src/b.ts"]
    (ws / "dist" / "esm").mkdir(parents=True)
    (ws / "dist" / "index.js").write_text("i")
    (ws / "dist" / "esm" / "index.mjs").write_text("m")
    (ws / "src" / "a.ts").write_text("A")
    (ws / "src" / "a.ts~").write_text("a")
    (ws / "src" / "b.ts").write_text("b")           # rewritten with identical content
    (ws / "node_modules" / ".vite" / "v").mkdir(parents=True)
    (ws / "node_modules" / ".vite" / "v" / "r.json").write_text("{}")
    (ws / "node_modules" / "x.js").unlink()
    (ws / ".git" / "HEAD").write_text("changed")    # .git is excluded
    (ws / "link").symlink_to("src")
    after = snap.snapshot()
    assert after["link"] == "link:src"
    d = tw.diff(before, after)
    assert d["created"] == ["dist/", "link", "node_modules/.vite/", "src/a.ts~"]
    assert d["modified"] == ["src/a.ts"]
    assert d["deletes"] == ["node_modules/x.js"]
    assert d["writes"] == ["dist/", "link", "node_modules/.vite/", "src/a.ts", "src/a.ts~"]
    assert d["counts"] == {"created": 5, "modified": 1, "deleted": 1}
    # deleting a whole directory collapses too
    import shutil
    shutil.rmtree(ws / "dist")
    assert tw.diff(after, snap.snapshot())["deletes"] == ["dist/"]


def test_merge_trace_and_paths(tmp_path):
    line = ScopeLine("o/r", SHA, ".github/workflows/t.yml")
    assert tw.trace_path(line, tmp_path) == tmp_path / "o__r" / SHA / "o__r__t.json"
    b1 = tw.binding_json(Binding("run", "make", env=(("A", "1"),)))
    b2 = tw.binding_json(Binding("uses", "actions/checkout@v4"))

    def rec(i, kind, binding, executed=True, code=0, writes=("dist/",)):
        return {"index": i, "kind": kind, "binding": binding, "executed": executed, "exit_code": code,
                "writes": list(writes), "deletes": []}

    d = tw.merge_trace(None, str(line), "j[b=2]", {"steps": [rec(0, "run", b1)]})
    d = tw.merge_trace(d, str(line), "j[b=1]", {"steps": [rec(0, "uses", b2, executed=False),
                                                          rec(1, "run", b1, code=2),
                                                          rec(2, "run", b1, executed=False)]})
    d = tw.merge_trace(d, str(line), "j[b=2]", {"steps": [rec(0, "run", b1, writes=("lib/",))]})
    assert list(d["instances"]) == ["j[b=1]", "j[b=2]"]
    assert d["format"] == tw.FORMAT == "ghav-trace/1"
    # only executed run steps that exited 0 reach the loader-facing list
    assert d["steps"] == [{"binding": b1, "writes": ["lib/"], "deletes": [], "external": [],
                           "instance": "j[b=2]", "index": 0}]


def test_known_effects_and_git_guard(tmp_path):
    assert "clone" in tw.known_effects("actions/checkout@abc")
    assert tw.known_effects("someone/thing@v1").startswith("Unknown action")
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    (tmp_path / "repo" / "sub").mkdir()
    assert tw.inside_git_repo(tmp_path / "repo" / "sub")
    b = Binding("run", "echo", env=(("A", "1"),))
    assert tw.binding_json(b) == {"kind": "run", "text": "echo", "with": {}, "env": {"A": "1"},
                                  "working_directory": None, "shell": None}


def test_trace_file_is_read_by_contract_loader(tmp_path):
    from ghav.contracts import TRACE_FORMAT, load_observed
    assert tw.FORMAT == TRACE_FORMAT
    b = Binding("run", "pnpm run build", with_=(), env=(("NODE_ENV", "development"),), working_directory="pkg")
    entry = {"steps": [{"index": 3, "kind": "run", "binding": tw.binding_json(b), "executed": True,
                        "exit_code": 0, "writes": ["dist/"], "deletes": ["old.txt"]}]}
    doc = tw.merge_trace(None, "o/r@x:.github/workflows/t.yml", "t", entry)
    path = tmp_path / "t.json"
    path.write_text(json.dumps(doc))
    obs = load_observed(path)
    assert list(obs) == [b.key()]
    assert obs[b.key()].writes == ("dist/", "old.txt")


# ---------------------------------------------------------------------------
# Committed dev-set resources (Zustand motivating example)
# ---------------------------------------------------------------------------

ZUSTAND = ScopeLine.parse(
    "pmndrs/zustand@b57db4f86ef179285da216eeb291266da82c361c:.github/workflows/test-multiple-builds.yml")


def _zustand_program():
    from ghav.contracts import annotate
    from ghav.lower import lower
    root = ROOT / "resources" / "repos" / "pmndrs__zustand" / ZUSTAND.sha
    return annotate(lower(root / ZUSTAND.path, root), root)


@pytest.mark.skipif(not cost.timings_path(ZUSTAND.benchmark_id()).exists(), reason="timings not fetched")
def test_zustand_historical_model():
    p = _zustand_program()
    m = cost.load_historical_model(p, ZUSTAND.benchmark_id())
    cov = m.coverage
    assert cov["jobs_matched"] == 2 and cov["github_jobs_unmatched"] == []
    assert cov["steps_matched_by_name"] == 20
    # only "Patch for PRD-ONLY" (never executed on GitHub) lacks a sample in both instances
    assert cov["steps_priced_historical"] == 18
    r = cost.cost_report(p, model=m)
    for ev in ("push", "pull_request"):
        e = r["events"][ev]
        assert e["executed_steps"] == e["executed_steps_historical"]
        assert e["billed_minutes"] == 2                       # both jobs bill one minute
        for j in e["jobs"].values():
            assert 20 <= j["seconds"] <= 60                    # real jobs take 26-49 s
    mock = cost.cost_report(p)
    assert mock["expected_seconds"] > r["expected_seconds"]


@pytest.mark.skipif(not tw.trace_path(ZUSTAND).exists(), reason="trace not recorded")
def test_zustand_trace_matches_manual_evidence():
    doc = json.loads(tw.trace_path(ZUSTAND).read_text())
    assert doc["format"] == tw.FORMAT and doc["benchmark"] == str(ZUSTAND)
    assert sorted(doc["instances"]) == ["test_multiple_builds[build=cjs,env=development]",
                                        "test_multiple_builds[build=esm,env=development]"]
    for iid, inst in doc["instances"].items():
        assert inst["complete"]
        steps = {s["index"]: s for s in inst["steps"]}
        assert [steps[i]["skipped_reason"] for i in (0, 1, 2)] == ["action"] * 3
        install, build, test = steps[3], steps[4], steps[9]
        assert install["writes"] == ["node_modules/"] and install["deletes"] == []
        # the build writes only dist (out); no tracked file, no node_modules change
        assert build["writes"] == ["dist/"] and build["modified"] == [] and build["deletes"] == []
        assert build["counts"]["created"] == 52
        # tests do not rebuild dist; their only write is the vite cache under node_modules
        assert all(w.startswith("node_modules/") for w in test["writes"]) and test["deletes"] == []
        assert steps[6]["executed"] is False                   # PRD-ONLY patch never runs
        patched = [p for i in (5, 7, 8) if steps[i]["executed"] for p in steps[i]["writes"]]
        assert patched and all(not p.startswith(("dist/", "node_modules/")) for p in patched)
    # the loader-facing list holds the executed run steps of both instances
    assert len(doc["steps"]) == 2 * 5
    from ghav.contracts import load_observed
    obs = load_observed(tw.trace_path(ZUSTAND))
    build_key = ("run", "pnpm run build", (), (), None, None)
    assert obs[build_key].writes == ("dist/",)
