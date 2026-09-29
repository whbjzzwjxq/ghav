"""Tests for ghav.pipeline (workflow_stats, run_benchmark) and ghav.cli."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from ghav import cli, pipeline

FIX = Path(__file__).parent / "fixtures" / "synth"
SCOPE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "scope"
SHA = "0123456789abcdef0123456789abcdef01234567"
LINE = f"acme/demo@{SHA}:.github/workflows/ci.yml"
STAGE_FILES = ("lowered.json", "contracts.json", "synth_full.json", "optimized_full.yml", "check_full.json")


@pytest.fixture()
def resources(tmp_path: Path) -> Path:
    """A resources directory with the `share` fixture as snapshot acme/demo@SHA."""
    res = tmp_path / "resources"
    shutil.copytree(FIX / "share", res / "repos" / "acme__demo" / SHA)
    return res


def test_parse_scope_line():
    b = pipeline.parse_scope_line(LINE)
    assert (b.repo, b.sha, b.path) == ("acme/demo", SHA, ".github/workflows/ci.yml")
    assert b.id == "acme__demo__ci" and b.line == LINE
    assert b.snapshot(Path("/r")) == Path("/r/repos/acme__demo") / SHA
    with pytest.raises(ValueError):
        pipeline.parse_scope_line("acme/demo")


def test_workflow_stats_matches_scope_builders(resources: Path):
    st = pipeline.workflow_stats(LINE, resources=resources)
    assert st["instances"] == 2 and st["frozen"] == 0 and st["frozen_instances"] == 0
    assert st["steps"] == 12 and st["top_steps"] == 0
    assert st["machine_seconds"] == pytest.approx(570.0)
    # mode=b repeats checkout, setup, install, build and lint: 150 of 570 seconds.
    assert st["redundancy_ratio"] == pytest.approx(150.0 / 570.0, abs=1e-6)
    sys.path.insert(0, str(SCOPE_DIR))
    try:
        import scopelib  # noqa: PLC0415 - the scope builders' own validator
        assert scopelib.validate_stats(st)["instances"] == 2
        assert set(scopelib.STATS_KEYS) <= set(st)
    finally:
        sys.path.remove(str(SCOPE_DIR))


def test_run_benchmark_writes_deterministic_stage_files(resources: Path, tmp_path: Path):
    outs = []
    for k in range(2):
        out = tmp_path / f"stage{k}"
        summary = pipeline.run_benchmark(LINE, "full", out, resources=resources)
        assert summary["benchmark"] == "acme__demo__ci" and summary["roundtrip_equivalent"]
        assert summary["best_cost"] < summary["original_cost"]
        outs.append(out)
    for name in STAGE_FILES:
        a, b = (outs[0] / name).read_bytes(), (outs[1] / name).read_bytes()
        assert a == b, name
        if name.endswith(".json"):
            text = a.decode()
            assert text == pipeline.dumps(json.loads(text))  # sorted keys, indent 2
    record = json.loads((outs[0] / "synth_full.json").read_text())
    assert record["scope_line"] == LINE and record["stats"]["instances"] == 2
    assert "wall_ms" not in record
    assert "wall_ms" in json.loads((outs[0] / "timing_full.json").read_text())
    assert json.loads((outs[0] / "check_full.json").read_text())["equivalent"]
    lowered = json.loads((outs[0] / "lowered.json").read_text())
    assert [j["id"] for j in lowered["jobs"]] == ["test[mode=a]", "test[mode=b]"]


def test_run_benchmark_rejects_unknown_variant_and_backend(resources: Path, tmp_path: Path):
    with pytest.raises(ValueError):
        pipeline.run_benchmark(LINE, "nope", tmp_path / "x", resources=resources)
    with pytest.raises(ValueError):
        pipeline.run_benchmark(LINE, "full", tmp_path / "y", resources=resources, cost_backend="nope")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_commands(resources: Path, tmp_path: Path, capsys):
    wf = str(FIX / "merge" / ".github" / "workflows" / "ci.yml")

    assert cli.main(["stats", LINE, "--resources", str(resources)]) == 0
    assert json.loads(capsys.readouterr().out)["instances"] == 2

    assert cli.main(["lower", wf]) == 0
    assert [j["id"] for j in json.loads(capsys.readouterr().out)["jobs"]] == ["lint", "typecheck"]

    assert cli.main(["contracts", wf]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["steps"] == 8

    assert cli.main(["emit", wf]) == 0
    assert "jobs:" in capsys.readouterr().out

    out = tmp_path / "opt.yml"
    report = tmp_path / "report.json"
    assert cli.main(["optimize", wf, "--out", str(out), "--report", str(report)]) == 0
    capsys.readouterr()
    assert json.loads(report.read_text())["best_cost"] == pytest.approx(145.0)

    assert cli.main(["check", wf, str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["equivalent"] is True

    # A candidate that drops the typecheck is refuted (exit status 1).
    bad = tmp_path / "bad.yml"
    bad.write_text(out.read_text().split("      - if:")[0])
    assert cli.main(["check", wf, str(bad)]) == 1
    assert json.loads(capsys.readouterr().out)["equivalent"] is False

    assert cli.main(["lower", str(tmp_path / "missing.yml")]) == 2


# ---------------------------------------------------------------------------
# The motivating example (committed snapshot and trace of pmndrs/zustand)
# ---------------------------------------------------------------------------

ZUSTAND = ("pmndrs/zustand@b57db4f86ef179285da216eeb291266da82c361c:"
           ".github/workflows/test-multiple-builds.yml")


def _zustand_available() -> bool:
    b = pipeline.parse_scope_line(ZUSTAND)
    return (b.snapshot() / b.path).is_file()


@pytest.mark.skipif(not _zustand_available(), reason="zustand snapshot not in resources/repos")
def test_zustand_observed_contracts_give_instance_producer_build_share(tmp_path: Path):
    """With the trace (observed contracts) the build writes only dist, and synthesis under the
    mock cost shares it through one matrix instance; the static guide leaves the build ⊤,
    so nothing moves. pnpm/action-setup is never dropped (job-local writes are updates)."""
    from ghav import synth
    bench = pipeline.parse_scope_line(ZUSTAND)
    p, root = pipeline.load_program(bench, use_traces=True)
    observed = pipeline.observed_for(p, root, True)
    res = synth.synthesize(p, synth.SynthConfig(), repo_root=root, observed=observed)
    assert res.improved and res.check["equivalent"] and res.roundtrip["equivalent"]
    shares = [m for m in res.best_moves if m["move"] == "share"]
    assert [(m["step"], m["template"], m["transfer"]) for m in shares] == [("pnpm run build", "observed", True)]
    assert shares[0]["producer"] in {j.id for j in p.jobs}
    for job in res.program.jobs:
        texts = [s.binding.text for s in job.steps]
        assert texts.count("pnpm/action-setup@0e279bb959325dab635dd2c09392533439d90093") == 1
    # The CLI runs the same search under the historical backend: no improvement there
    # (the build's median is about 9 s; a transfer is priced at 20 mock seconds).
    out = tmp_path / "z.yml"
    assert cli.main(["optimize", ZUSTAND, "--traces", "--cost", "historical", "--out", str(out)]) == 0
    assert "ghav: pack" not in out.read_text()
    static, _ = pipeline.load_program(bench)
    none = synth.synthesize(static, synth.SynthConfig(), repo_root=root, observed={})
    assert not none.improved and not [m for m in none.best_moves if m["move"] == "share"]
