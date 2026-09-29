"""Tests of the experiment scripts (scripts/ablation, scripts/aggregate).

Run with ``make test-scripts`` (``uv run pytest -q scripts/ablation scripts/aggregate``). They use
committed resources only (a pool snapshot and a developer pair), no network.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "ablation"), str(ROOT / "scripts" / "aggregate")]

import aggregate  # noqa: E402
import check_devopt  # noqa: E402
import compare_outputs  # noqa: E402
import export_tables  # noqa: E402
import run_stages  # noqa: E402

def _pinned(repo: str, path: str) -> str:
    """The committed pinned scope line of a small pool workflow (two jobs)."""
    for line in (ROOT / "scope" / "workflows.scope").read_text(encoding="utf-8").splitlines():
        if line.startswith(repo + "@") and line.endswith(":" + path):
            return line.strip()
    pytest.skip(f"{repo}:{path} not in scope/workflows.scope")


@pytest.fixture(scope="module")
def scope_file(tmp_path_factory) -> Path:
    lines = [_pinned("GrapesJS/grapesjs", ".github/workflows/test.yml"),
             _pinned("slab/quill", ".github/workflows/pull-request.yml")]
    d = tmp_path_factory.mktemp("scope")
    p = d / "tiny.scope"
    p.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def two_runs(scope_file, tmp_path_factory) -> tuple[Path, Path]:
    a = tmp_path_factory.mktemp("run_a")
    b = tmp_path_factory.mktemp("run_b")
    for stages, workers in ((a, 1), (b, 3)):
        rc = run_stages.main(["--scope", str(scope_file), "--variants", "full,rules", "--cost", "mock",
                              "--workers", str(workers), "--stages", str(stages), "--max-expansions", "50"])
        assert rc == 0
    return a, b


def test_stage_name_and_scope_lines(tmp_path):
    p = tmp_path / "eval.scope"
    p.write_text("# c\nb/b@" + "1" * 40 + ":x.yml\n\na/a@" + "2" * 40 + ":y.yml\na/a@" + "2" * 40 + ":y.yml\n")
    assert run_stages.stage_name(p, "historical") == "eval.historical"
    assert run_stages.read_scope_lines(p) == ["a/a@" + "2" * 40 + ":y.yml", "b/b@" + "1" * 40 + ":x.yml"]


def test_run_stages_writes_stage_files(two_runs, scope_file):
    a, _ = two_runs
    stage = a / "tiny.mock"
    manifest = json.loads((stage / "manifest.json").read_text())
    assert manifest["cost_backend"] == "mock"
    assert manifest["runs"]["full"]["max_expansions"] == 50
    assert set(manifest["runs"]) == {"full", "rules"}
    for bid in manifest["benchmarks"]:
        d = stage / bid
        for name in ("lowered.json", "contracts.json", "cost.json", "redundancy.json", "consistency.json",
                     "synth_full.json",
                     "synth_rules.json", "check_full.json", "optimized_full.yml", "timing_full.json"):
            assert (d / name).is_file(), name
        red = json.loads((d / "redundancy.json").read_text())
        syn = json.loads((d / "synth_full.json").read_text())
        # the breakdown sums to the scope criterion's redundant seconds (mock cost)
        assert red["redundant_seconds"] == pytest.approx(syn["stats"]["redundant_seconds"], abs=1e-6)
        assert sum(red["by_category"].values()) == pytest.approx(red["redundant_seconds"], abs=1e-4)
        assert syn["config"]["max_expansions"] == 50
        assert "wall_ms" not in syn


def test_results_do_not_depend_on_worker_count(two_runs):
    a, b = two_runs
    n, problems = compare_outputs.compare(a, b)
    assert n > 10
    assert problems == []


def test_compare_outputs_masks_only_wall_clock(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d, wall, val in ((a, "1.0", "3"), (b, "2.0", "3")):
        (d / "tables").mkdir(parents=True)
        (d / "x.csv").write_text(f"k,wall_s,v\nq,{wall},{val}\n")
        (d / "timing_full.json").write_text(json.dumps({"wall_ms": wall}))
        (d / "tables" / "rq3.tex").write_text(f"Cost & Wall s \\\\\nmock & {wall} \\\\\n")
    assert compare_outputs.compare(a, b)[1] == []
    (b / "x.csv").write_text("k,wall_s,v\nq,2.0,4\n")
    assert compare_outputs.compare(a, b)[1] == ["differs: x.csv"]
    (b / "extra.json").write_text("{}")
    assert any("only in" in p for p in compare_outputs.compare(a, b)[1])


def test_aggregate_tables_and_csv(two_runs, tmp_path):
    a, _ = two_runs
    # the aggregator reads <scope>.<backend> directories; present the tiny run as the eval scope
    stages = tmp_path / "stages"
    (stages / "eval.mock").mkdir(parents=True)
    for p in (a / "tiny.mock").iterdir():
        target = stages / "eval.mock" / p.name
        target.symlink_to(p, target_is_directory=p.is_dir())
    out = tmp_path / "out"
    assert aggregate.main(["--stages", str(stages), "--out", str(out)]) == 0
    rows = list(csv.DictReader((out / "rq1_effectiveness.csv").open()))
    assert len(rows) == 2 and {r["backend"] for r in rows} == {"mock"}
    assert all(r["status"] == "ok" and r["roundtrip_equivalent"] == "true" for r in rows)
    rq2 = list(csv.DictReader((out / "rq2_rules.csv").open()))
    assert len(rq2) == 2
    rq3 = list(csv.DictReader((out / "rq3_ablation.csv").open()))
    assert sorted({r["variant"] for r in rq3}) == ["full", "rules"]
    for name in ("rq1", "rq2", "rq3", "rq4"):
        text = (out / "tables" / f"{name}.tex").read_text()
        assert r"\toprule" in text and r"\bottomrule" in text and r"\small" in text
        assert r"\caption{" in text
    rq1 = (out / "tables" / "rq1.tex").read_text()
    assert "mock cost table" in rq1 and "billed minutes" in rq1 and "machine seconds" in rq1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["rq1"]["eval.mock"]["workflows"] == 2


def test_tex_escape():
    assert aggregate.tex_escape("a_b & 50% #1") == r"a\_b \& 50\% \#1"


def test_export_tables(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    (src / "rq1.tex").write_text("x\n")
    (src / "notes.txt").write_text("ignored\n")
    assert export_tables.export(src, dest) == [("rq1.tex", "added")]
    assert export_tables.export(src, dest) == [("rq1.tex", "unchanged")]
    (src / "rq1.tex").write_text("y\n")
    assert export_tables.export(src, dest) == [("rq1.tex", "updated")]
    assert not (dest / "notes.txt").exists()


def test_counterexample_summary():
    cex = {"event": "push", "differences": ["region:ext:*", "group:a=>b"], "region": "ext:*",
           "group": {"old": ["a"], "new": ["b"], "original_verdict": True, "candidate_verdict": False},
           "original": {"failed_steps": []}, "candidate": {"failed_steps": ["b#1 x", "b#2 y", "b#3 z"]}}
    s = check_devopt.summarize_counterexample(cex)
    assert s.startswith("event=push | region ext:* differs | check group a => b: before pass, after fail")
    assert "after failing: b#1 x; b#2 y; +1" in s


def test_devopt_pair_classification():
    pairs = json.loads((ROOT / "scope" / "devopt.pairs.json").read_text())["pairs"]
    pair = next((p for p in pairs if p["before"].endswith("compat.yml")), None)
    if pair is None:
        pytest.skip("no compat.yml pair")
    rec = check_devopt.check_pair(pair)
    assert rec["verdict"] in ("equivalent", "counterexample", "unsupported")
    assert rec["checker_reason"] in ("equivalent", "counterexample", "ill-formed", "original-ill-formed",
                                     "unknown")
    if rec["verdict"] == "counterexample":
        assert rec["counterexample_summary"].startswith("event=")
    assert "smt_ms" not in rec["check"]


def test_consistency_audit_on_a_consistent_workflow():
    """The audit compares both executions per event; a plain two-job workflow is consistent."""
    line = _pinned("GrapesJS/grapesjs", ".github/workflows/test.yml")
    r = run_stages.consistency_json(line, "mock")
    assert r["consistent"] is True
    assert r["events"]["push"]["simulated_steps"] > 0
    assert set(r["events"]) == {"push", "pull_request"}
