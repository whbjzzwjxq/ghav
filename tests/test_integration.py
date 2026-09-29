"""Cross-module integration: lower -> contracts -> emit -> lower -> checker.

The ts-jest fixture is the real repository (see tests/fixtures/integration/README.md). The
round trip must be equivalent: emission renames the jobs expanded from the local reusable
workflow, and the checker recovers the correspondence (``checker.recover_origins``, check
identity = check name and matrix). Hand-built candidates exercise the checker on the real
program: a dropped check step, and artifact sharing in ``typescript-compatibility`` under
the v1 guide (rejected: the build step is ⊤) and under a labeled what-if contract.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

from ghav import expr as gexpr
from ghav.checker import check
from ghav.contracts import annotate, infer
from ghav.emit import emit, transfer_steps
from ghav.lower import lower
from ghav.wir import TRUE, Binding, Expr, JobInstance, Program, Step

FIXTURES = Path(__file__).parent / "fixtures" / "integration"
TS_JEST = FIXTURES / "ts-jest"
CI = Path(".github/workflows/ci.yml")
SUCCESS = Expr("status", ("success",))
TSC = "typescript-compatibility"


def lowered(root: Path, workflow: Path = CI) -> Program:
    return annotate(lower(root / workflow, root), root)


def relowered(root: Path, text: str, tmp_path: Path) -> Program:
    """Lower emitted YAML inside a copy of the repository (local actions stay resolvable)."""
    repo = tmp_path / "repo"
    shutil.copytree(root, repo)
    out = repo / ".github" / "workflows" / "ghav-emitted.yml"
    out.write_text(text)
    return annotate(lower(out, repo), repo)


def canonical(res) -> str:
    return json.dumps(res.to_json(include_timing=False), sort_keys=True)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

def test_ts_jest_round_trip_is_equivalent(tmp_path):
    original = lowered(TS_JEST)
    assert len(original.jobs) == 24
    assert not any(j.frozen for j in original.jobs)
    text = emit(original)
    again = relowered(TS_JEST, text, tmp_path)
    # Emission flattens the 18 reusable-call instances into renamed jobs; check names survive.
    assert sorted(j.check_name for j in again.jobs) == sorted(j.check_name for j in original.jobs)
    assert {j.id for j in again.jobs} != {j.id for j in original.jobs}

    res = check(original, again)
    assert res.equivalent, res.counterexample or [i for i in res.issues if i["blocking"]]
    assert res.recovered_origins == 18 * 4  # every step of every renamed instance
    # The original's races between ⊤ steps (catch-all ext:*) are inherited, not blocking.
    assert res.issues and not any(i["blocking"] for i in res.issues)
    # Deterministic, and emission is a fixed point on its own output.
    assert canonical(check(original, again)) == canonical(res)
    assert emit(again).split("\n", 1)[1] == emit(relowered(TS_JEST, text, tmp_path / "b")).split("\n", 1)[1]


def test_ts_jest_guide_classification():
    """What the v1 guide makes of ts-jest (the paper's overview depends on this)."""
    p = lowered(TS_JEST)
    by_text: dict[str, set[str]] = {}
    for j in p.jobs:
        for s in j.steps:
            by_text.setdefault(s.binding.text.strip(), set()).add(s.contract.template)
    # The composite restores caches; where a later step writes dep (a ⊤ build, the
    # compatibility jobs' extra install) the restore is opaque (guide 7).
    assert by_text["./.github/actions/setup-workspace"] == {
        "setup-runtime+cache+install-pinned", "setup-runtime+cache-opaque+install-pinned"}
    assert by_text["npm run lint"] == {"lint"}
    assert by_text["npm run test"] == {"test"}
    # prebuild (rimraf) and postbuild (node scripts/post-build.js) have no template.
    assert by_text["npm run build"] == {"top"}
    assert by_text["npm run build\nnpm run postbuild"] == {"top"}


def test_frozen_matrix_shell_default_survives_round_trip(tmp_path):
    root = FIXTURES / "frozen-matrix-shell"
    original = lowered(root)
    assert all(j.frozen for j in original.jobs) and len(original.jobs) == 2
    shells = {j.check_name: {s.binding.shell for s in j.steps if s.binding.kind == "run"} for j in original.jobs}
    assert shells == {"Test (linux)": {"bash"}, "Test (windows)": {"pwsh"}}
    again = relowered(root, emit(original), tmp_path)
    assert {j.check_name: {s.binding.shell for s in j.steps if s.binding.kind == "run"} for j in again.jobs} == shells
    assert check(original, again).equivalent


def test_frozen_job_upload_reaches_downstream_download():
    """A frozen job's upload must exist for the consumer; otherwise its later steps never
    run in the semantics and removing them would wrongly check as equivalent."""
    p = lowered(FIXTURES / "frozen-upload")
    assert p.job("build").frozen and not p.job("test").frozen
    assert check(p, p).equivalent
    test = p.job("test")
    assert test.steps[-1].binding.text == "npm test"
    dropped = p.with_jobs([j for j in p.jobs if j.id != "test"] + [replace(test, steps=test.steps[:-1])])
    res = check(p, dropped)
    assert not res.equivalent
    assert res.counterexample["group"] == {
        "old": ["test"], "new": ["test"], "original_verdict": False, "candidate_verdict": True}
    assert res.counterexample["original"]["failed_steps"] == ["test#3 npm test"]


# ---------------------------------------------------------------------------
# Candidates on the real program
# ---------------------------------------------------------------------------

def test_ts_jest_dropping_a_lint_step_is_refuted():
    original = lowered(TS_JEST)
    lint = original.job("lint")
    assert lint.steps[3].binding.text == "npm run lint-prettier-ci"
    candidate = original.with_jobs(
        [j for j in original.jobs if j.id != "lint"] + [replace(lint, steps=lint.steps[:3])])
    res = check(original, candidate)
    assert not res.equivalent and res.reason == "counterexample"
    assert res.counterexample["group"] == {
        "old": ["lint"], "new": ["lint"], "original_verdict": False, "candidate_verdict": True}
    assert res.counterexample["original"]["failed_steps"] == ["lint#3 Run prettier 💅"]


def build_script_what_if(p: Program, root: Path, *, build_writes_tool: bool = False) -> Program:
    """WHAT-IF contracts (not the guide): every ``npm run build`` step gets the guide
    contract of the build script's own command, as if the guide classified the prebuild
    and postbuild hooks as having no effect, and every cache restore keeps its transparent
    (cold-cache) contract, as if no later step broke cache transparency (guide 7: the
    compatibility jobs' extra ``npm install`` does). Unless ``build_writes_tool``, the build
    writes only its ``out`` paths (as an observed contract would say): the guide's build
    template also writes ``tool``, which is not transferable and, since job-local writes are
    updates, reaches the compatibility jobs' later ⊤ step. Used only to exercise sharing on
    the real workflow structure; the guide itself gives the builds ⊤ and those restores
    ``cache-opaque``."""
    script = json.loads((root / "package.json").read_text())["scripts"]["build"]
    jobs = []
    for j in p.jobs:
        steps = []
        for s in j.steps:
            if s.binding.kind == "run" and s.binding.text.strip().startswith("npm run build"):
                probe = replace(s, binding=Binding("run", script))
                c = infer(probe, j, root)
                if not build_writes_tool:
                    c = replace(c, writes=frozenset(r for r in c.writes if r.kind != "tool"))
                s = replace(s, contract=c)
            elif "cache-opaque" in s.contract.template.split("+"):
                s = replace(s, contract=infer(s, j, root))  # the step alone: no job-level rule
            steps.append(s)
        jobs.append(replace(j, steps=tuple(steps)))
    return p.with_jobs(jobs)




def hoist_tsc_build(p: Program, root: Path, *, source: str | None = None) -> Program:
    """Share the build of the four ``typescript-compatibility`` instances.

    With ``source=None`` a new producer job runs checkout, setup and the build of the first
    instance and uploads ``dist`` (the paper's hoist). With ``source=<instance id>`` the
    build of that other job is reused instead (the paper's second rejected candidate).
    Consumers download ``dist`` in place of their build and wait for the producer.
    """
    consumers = sorted((j for j in p.jobs if j.orig_job == TSC), key=lambda j: j.id)
    first = consumers[0]
    build_idx = 2
    assert first.steps[build_idx].binding.text.strip() == "npm run build"
    art = "ghav-build-" + (source or TSC).split("/")[0].replace("[", "-").replace("=", "-").rstrip("]")
    jobs = [j for j in p.jobs if j.orig_job != TSC and j.id != source]
    if source is None:
        producer = replace(first, id=f"ghav-{TSC}", orig_job=f"ghav-{TSC}", check_name=f"ghav {TSC} build",
                           matrix=(), cond=TRUE, needs=frozenset(), outputs=(), fail_fast_group=None, raw=None)
        (pack, up), _ = transfer_steps(producer, consumers[0], ["dist"], art, repo_root=root)
        jobs.append(replace(producer, steps=first.steps[: build_idx + 1] + (pack, up)))
        producer_id = producer.id
    else:
        src = p.job(source)
        k = next(i for i, s in enumerate(src.steps) if s.binding.text.strip().startswith("npm run build"))
        (pack, up), _ = transfer_steps(src, consumers[0], ["dist"], art, repo_root=root)
        jobs.append(replace(src, steps=src.steps[: k + 1] + (pack, up) + src.steps[k + 1:]))
        producer_id = source
    for c in consumers:
        _, (down, unpack) = transfer_steps(p.job(producer_id) if source else jobs[-1], c, ["dist"], art,
                                           repo_root=root)
        steps = c.steps[:build_idx] + (down, unpack) + c.steps[build_idx + 1:]
        cond = SUCCESS if c.cond == TRUE else gexpr.conj([SUCCESS, c.cond])
        jobs.append(replace(c, steps=steps, needs=frozenset({producer_id}), cond=cond))
    return p.with_jobs(jobs)


def test_ts_jest_hoist_is_rejected_under_the_guide():
    """With the v1 guide the build is ⊤: it reads its instance identity and writes the
    catch-all external region, so no build can move to a producer."""
    original = lowered(TS_JEST)
    res = check(original, hoist_tsc_build(original, TS_JEST))
    assert not res.equivalent and res.reason == "counterexample"
    assert "region:ext:*" in res.counterexample["differences"]


def test_ts_jest_hoist_is_refuted_when_the_build_also_writes_tool():
    """Job-local writes are updates: the build's (untransferred) ``tool`` write survives the
    later ``npm install`` and is read by the compatibility script (⊤), so hoisting the build
    changes what that script sees."""
    original = build_script_what_if(lowered(TS_JEST), TS_JEST, build_writes_tool=True)
    res = check(original, hoist_tsc_build(original, TS_JEST))
    assert not res.equivalent and res.reason == "counterexample"
    assert "region:ext:*" in res.counterexample["differences"]


def test_ts_jest_hoist_under_what_if_contract_is_equivalent_and_round_trips(tmp_path):
    original = build_script_what_if(lowered(TS_JEST), TS_JEST)
    candidate = hoist_tsc_build(original, TS_JEST)
    res = check(original, candidate)
    assert res.equivalent, res.counterexample or [i for i in res.issues if i["blocking"]]
    tsc_checks = sorted(j.check_name for j in original.jobs if j.orig_job == TSC)
    assert {"old": tsc_checks, "new": sorted(tsc_checks + [f"ghav {TSC} build"])} in [g.to_json() for g in res.groups]

    # Emit the candidate, lower it again, and check it against the original (the round trip).
    text = emit(candidate)
    assert "actions/upload-artifact@v4" in text and "retention-days: '1'" in text
    assert 'tar -cf "$RUNNER_TEMP/' in text and 'tar -xf "$RUNNER_TEMP/' in text
    again = build_script_what_if(relowered(TS_JEST, text, tmp_path), TS_JEST)
    rt = check(original, again)
    assert rt.equivalent, rt.counterexample or [i for i in rt.issues if i["blocking"]]


def test_ts_jest_reusing_the_unit_test_build_is_refuted_under_what_if_contract():
    """The paper's second rejected candidate: test.yml runs build and postbuild, the
    compatibility job only build; different bindings are different tasks."""
    original = build_script_what_if(lowered(TS_JEST), TS_JEST)
    source = "test-ubuntu[node-version=22]/unit-test"
    res = check(original, hoist_tsc_build(original, TS_JEST, source=source))
    assert not res.equivalent and res.reason == "counterexample"
    assert "region:ext:*" in res.counterexample["differences"]
