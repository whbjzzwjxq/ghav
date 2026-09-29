from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from ghav import emit as E
from ghav import expr
from ghav.emit import EmitError, emit, emit_program, job_keys, render_expr, sanitize_job_id
from ghav.wir import TRUE, Binding, Contract, Expr, JobInstance, Program, Step

FIXTURES = Path(__file__).parent / "fixtures" / "emit"
SUCCESS = Expr("status", ("success",))


# ---------------------------------------------------------------------------
# Builders for hand-built programs (same conventions as lowering: known matrix values are
# interpolated into bindings, runs-on and conditions; default conditions are folded in)
# ---------------------------------------------------------------------------

def pairs(combo: dict | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, expr.to_str(v)) for k, v in (combo or {}).items()))


def raw_binding(raw: dict, combo: dict | None = None) -> Binding:
    env = {"matrix": combo} if combo else {}
    interp = lambda v: expr.interpolate(v, env)  # noqa: E731
    kind = "uses" if "uses" in raw else "run"
    return Binding(
        kind=kind, text=interp(raw[kind]) if kind == "run" else raw[kind],
        with_=tuple(sorted((k, interp(v)) for k, v in (raw.get("with") or {}).items())),
        env=tuple(sorted((k, interp(v)) for k, v in (raw.get("env") or {}).items())),
        working_directory=raw.get("working-directory"), shell=raw.get("shell"),
        name=interp(raw["name"]) if "name" in raw else None, step_id=raw.get("id"),
    )


def lowered_cond(raw_if, combo=None, fold=True) -> Expr:
    if raw_if is None:
        return SUCCESS if fold else TRUE
    e = expr.parse(raw_if)
    if combo:
        e = expr.substitute(e, {"matrix": combo})
    if fold and not expr.uses_status_function(e):
        e = expr.conj([SUCCESS, e])
    return expr.simplify(e, boolean=True)


def raw_step(raw: dict, origin, combo=None, template="noop") -> Step:
    return Step(raw_binding(raw, combo), lowered_cond(raw.get("if"), combo),
                Contract(template=template), origin, raw)


def instance(orig, raw_job, combo=None, *, jid=None, steps=None, needs=(), cond=None,
             frozen=False, check_name=None, fail_fast=True) -> JobInstance:
    mp = pairs(combo)
    if jid is None:
        jid = orig if not mp else f"{orig}[{','.join(f'{k}={v}' for k, v in mp)}]"
    if steps is None:
        steps = tuple(raw_step(s, (orig, i), combo) for i, s in enumerate(raw_job.get("steps", [])))
    ro = raw_job.get("runs-on", "ubuntu-latest")
    runs_on = expr.interpolate(ro, {"matrix": combo}) if combo else ro
    if cond is None:
        cond = lowered_cond(raw_job.get("if"), fold=bool(needs))
    name = raw_job.get("name")
    if check_name is None and isinstance(name, str) and "${{" in name:
        check_name = expr.interpolate(name, {"matrix": combo} if combo else {})
    if check_name is None:
        base = orig if name is None else name
        check_name = base if not combo else f"{base} ({', '.join(expr.to_str(v) for v in combo.values())})"
    return JobInstance(
        id=jid, orig_job=orig, check_name=check_name, matrix=mp, runs_on=runs_on, cond=cond,
        needs=frozenset(needs), steps=tuple(steps),
        fail_fast_group=(orig if combo and fail_fast else None), frozen=frozen, raw=raw_job,
    )


def program(*jobs, name="CI", **kw) -> Program:
    return Program(name=name, on={"push": {"branches": ["main"]}, "pull_request": None},
                   jobs=tuple(sorted(jobs, key=lambda j: j.id)), **kw)


RAW_TEST = {
    "runs-on": "${{ matrix.os }}",
    "strategy": {"matrix": {"os": ["ubuntu-latest", "windows-latest"], "node": [18, 20]}},
    "steps": [
        {"uses": "actions/checkout@v4"},
        {"uses": "actions/setup-node@v4", "with": {"node-version": "${{ matrix.node }}"}},
        {"run": "npm ci"},
        {"name": "Build", "run": "npm run build"},
        {"run": "npm test", "if": "matrix.os == 'ubuntu-latest'"},
    ],
}
COMBOS = [{"os": o, "node": n} for o in ("ubuntu-latest", "windows-latest") for n in (18, 20)]


def matrix_program() -> Program:
    return program(*(instance("test", RAW_TEST, c) for c in COMBOS))


def shared_build_program() -> Program:
    """A synthesized candidate: one producer builds and uploads (the transfer idiom); each
    matrix instance downloads instead of building."""
    raws = RAW_TEST["steps"]
    producer_steps = (
        raw_step(raws[0], ("test", 0)),
        Step(Binding("uses", "actions/setup-node@v4", with_=(("node-version", "20"),)), SUCCESS,
             Contract(template="setup-runtime"), ("test", 1), None),
        raw_step(raws[2], ("test", 2)),
        raw_step(raws[3], ("test", 3)),
    )
    producer = JobInstance(
        id="ghav-build", orig_job="ghav-build", check_name="ghav build", matrix=(),
        runs_on="ubuntu-latest", cond=TRUE, needs=frozenset(), steps=producer_steps)
    consumers = []
    for c in COMBOS:
        consumer = instance("test", RAW_TEST, c, needs=["ghav-build"])
        (pack, up), (down, unpack) = E.transfer_steps(producer, consumer, ["dist"], "ghav-1a2b3c")
        steps = (
            raw_step(raws[0], ("test", 0), c),
            raw_step(raws[1], ("test", 1), c),
            raw_step(raws[2], ("test", 2), c),
            down, unpack,
            raw_step(raws[4], ("test", 4), c),
        )
        consumers.append(replace(consumer, steps=steps))
    producer = replace(producer, steps=producer_steps + (pack, up))
    return program(producer, *consumers)


def golden(name: str, text: str) -> None:
    path = FIXTURES / name
    assert path.exists(), f"missing golden file {path}"
    assert text == path.read_text(), f"{name} differs from golden output"


def with_steps(j: JobInstance, steps) -> JobInstance:
    return replace(j, steps=tuple(steps))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_sanitize_job_id():
    assert sanitize_job_id("test[node=18,os=ubuntu-latest]") == "test-node-18-os-ubuntu-latest"
    assert sanitize_job_id("caller[x=1]/callee") == "caller-x-1-callee"
    assert sanitize_job_id("1st job") == "j-1st-job"
    assert sanitize_job_id("node=18.x") == "node-18-x"
    assert sanitize_job_id("_a") == "_a"
    assert sanitize_job_id("[]") == "job"


def test_render_expr_precedence_and_renaming():
    e = expr.parse("!(a == 'x' || github.ref == 'r') && needs.build.result == 'success' "
                   "&& steps.s.outcome != 'failure'")
    text = render_expr(e, needs_map=lambda j: j + "-2", step_map=lambda s: s + "-2")
    assert "needs.build-2.result" in text and "steps.s-2.outcome" in text
    assert expr.parse(render_expr(e)) == e
    assert E._if_text("!cancelled()") == "${{ !cancelled() }}"
    assert E._if_text("success()") == "success()"


def test_render_expr_rejects_step_rename_inside_opaque():
    e = Expr("opaque", ("steps.s.outputs.x[0]", "steps.s.outputs.x"))
    assert "steps.s.outputs.x" in render_expr(e)
    with pytest.raises(EmitError):
        render_expr(e, step_map=lambda s: s + "-2")


def test_same_condition_modulo_folding_and_order():
    a = expr.parse("success() && (github.event_name == 'push' && 'a' == 'a')")
    b = expr.parse("github.event_name == 'push' && success()")
    assert E.same_condition(a, b)
    assert not E.same_condition(a, expr.parse("success()"))


def test_expand_matrix_github_rules():
    m = {"os": ["u", "w"], "node": [18, 20],
         "exclude": [{"os": "w", "node": 18}],
         "include": [{"os": "u", "experimental": True}, {"os": "m", "node": 22}]}
    assert E.expand_matrix(m) == [
        {"os": "u", "node": 18, "experimental": True},
        {"os": "u", "node": 20, "experimental": True},
        {"os": "w", "node": 20},
        {"os": "m", "node": 22},
    ]
    assert E.expand_matrix({"os": "${{ fromJSON(x) }}"}) is None
    assert E.expand_matrix({"include": [{"a": 1}, {"a": 2}]}) == [{"a": 1}, {"a": 2}]


# ---------------------------------------------------------------------------
# Emission of hand-built programs
# ---------------------------------------------------------------------------

def test_untouched_matrix_folds_back_to_original_strategy():
    p = matrix_program()
    wf = emit_program(p)
    assert list(wf) == ["name", "on", "jobs"]
    assert list(wf["jobs"]) == ["test"]
    job = wf["jobs"]["test"]
    assert job["strategy"] == {"fail-fast": True, **RAW_TEST["strategy"]}  # always explicit
    assert job["runs-on"] == "${{ matrix.os }}"
    assert job["steps"] == [dict(sorted(s.items(), key=lambda kv: E._STEP_KEY_ORDER.index(kv[0])))
                            for s in RAW_TEST["steps"]]
    assert "name" not in job and "needs" not in job and "if" not in job
    assert set(job_keys(p).values()) == {"test"}
    golden("matrix.yml", emit(p))


def test_shared_build_candidate():
    p = shared_build_program()
    wf = emit_program(p)
    assert sorted(wf["jobs"]) == ["ghav-build", "test"]
    prod = wf["jobs"]["ghav-build"]
    assert prod["name"] == "ghav build"
    assert prod["runs-on"] == "ubuntu-latest"
    pack, upload = prod["steps"][-2:]
    assert pack == {"name": "ghav: pack ghav-1a2b3c",
                    "run": 'tar -cf "$RUNNER_TEMP/ghav-1a2b3c.tar" -C "$GITHUB_WORKSPACE" dist',
                    "shell": "bash"}
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["with"] == {"if-no-files-found": "error", "name": "ghav-1a2b3c",
                              "path": "${{ runner.temp }}/ghav-1a2b3c.tar", "retention-days": "1"}
    test = wf["jobs"]["test"]
    assert test["needs"] == ["ghav-build"]
    assert test["strategy"] == {"fail-fast": True, **RAW_TEST["strategy"]}
    download, unpack = test["steps"][3:5]
    assert download["uses"] == "actions/download-artifact@v4"
    assert download["with"] == {"name": "ghav-1a2b3c", "path": "${{ runner.temp }}"}
    assert unpack["run"] == 'tar -xf "$RUNNER_TEMP/ghav-1a2b3c.tar" -C "$GITHUB_WORKSPACE" dist'
    assert test["steps"][5]["if"] == "matrix.os == 'ubuntu-latest'"
    golden("shared_build.yml", emit(p))


def test_transfer_steps_helper():
    prod = instance("build", {"runs-on": "ubuntu-latest", "steps": [{"run": "npm run build"}]})
    cons = instance("test", RAW_TEST, COMBOS[0])
    name = E.transfer_name(prod, ["dist", "lib"])
    assert name.startswith("ghav-build-") and name == E.transfer_name(prod, ["lib", "dist"])
    assert name != E.transfer_name(cons, ["dist", "lib"])  # instance-distinguishing
    (pack, up), (down, unpack) = E.transfer_steps(prod, cons, ["lib", "dist"], name)
    assert [E._is_transfer(s.binding) for s in (pack, up, down, unpack)] == ["pack", "upload", "download", "unpack"]
    assert pack.origin == up.origin == ("build", E.TRANSFER_ORIGIN_INDEX)
    assert (down.origin_instance, unpack.origin_instance) == (cons.id, cons.id)
    assert unpack.contract.out_paths == ("dist", "lib") and not any(s.contract.top for s in (pack, up, down, unpack))
    with pytest.raises(EmitError):
        E.transfer_steps(prod, cons, ["dist"], "dist")  # reserved prefix required
    with pytest.raises(EmitError):
        E.transfer_steps(prod, cons, ["../x"], name)


def test_partial_fold_uses_include_list_and_standalone_job():
    insts = [instance("test", RAW_TEST, c) for c in COMBOS]
    extra = Step(Binding("run", "npm run lint"), SUCCESS, Contract(template="lint"), ("lint", 0))
    insts[3] = with_steps(insts[3], insts[3].steps + (extra,))  # windows/20 differs now
    p = program(*insts)
    wf = emit_program(p)
    assert sorted(wf["jobs"]) == ["test", "test-node-20-os-windows-latest"]
    folded = wf["jobs"]["test"]
    assert folded["strategy"]["matrix"] == {"include": [
        {"os": "ubuntu-latest", "node": 18},
        {"os": "ubuntu-latest", "node": 20},
        {"os": "windows-latest", "node": 18},
    ]}
    assert folded["strategy"]["fail-fast"] is True
    assert "name" not in folded
    single = wf["jobs"]["test-node-20-os-windows-latest"]
    assert single["name"] == "test"  # GitHub appends the matrix values: "test (windows-latest, 20)"
    assert single["strategy"]["matrix"] == {"include": [{"os": "windows-latest", "node": 20}]}
    assert single["steps"][-1] == {"run": "npm run lint"}
    assert single["steps"][1]["with"] == {"node-version": "${{ matrix.node }}"}


def test_fail_fast_false_is_emitted():
    p = program(*(instance("test", RAW_TEST, c, fail_fast=False) for c in COMBOS))
    assert emit_program(p)["jobs"]["test"]["strategy"]["fail-fast"] is False


def test_dependent_on_single_instance_prevents_folding():
    insts = [instance("test", RAW_TEST, c) for c in COMBOS]
    only = insts[0].id
    report = instance("report", {"runs-on": "ubuntu-latest", "needs": "test",
                                 "steps": [{"run": "echo done"}]}, needs=[only])
    p = program(*insts, report)
    wf = emit_program(p)
    key = job_keys(p)[only]
    assert key != "test"
    assert wf["jobs"]["report"]["needs"] == [key]
    assert wf["jobs"]["test"]["strategy"]["matrix"]["include"] == [
        {"os": "ubuntu-latest", "node": 20}, {"os": "windows-latest", "node": 18},
        {"os": "windows-latest", "node": 20}]


def test_changed_conditions_are_rendered_unchanged_kept():
    raw = {"runs-on": "ubuntu-latest", "if": "github.event_name == 'push'",
           "steps": [{"run": "npm ci"}, {"run": "npm test", "if": "github.event_name == 'push'"},
                     {"run": "npm run lint"}]}
    j = instance("ci", raw)
    changed = replace(j.steps[2], cond=expr.parse("success() && needs.gen.result == 'success'"))
    gen = instance("gen", {"runs-on": "ubuntu-latest", "steps": [{"run": "make"}]})
    ci = emit_program(program(with_steps(j, j.steps[:2] + (changed,)), gen))["jobs"]["ci"]
    assert ci["if"] == "github.event_name == 'push'"
    assert ci["steps"][1]["if"] == "github.event_name == 'push'"
    # status references stay explicit (lowering keeps an explicit success() as written)
    assert ci["steps"][2]["if"] == "success() && needs.gen.result == 'success'"


def test_job_condition_default_fold_depends_on_emitted_needs():
    raw = {"runs-on": "ubuntu-latest", "if": "github.event_name == 'push'", "steps": [{"run": "x"}]}
    j = replace(instance("ci", raw), needs=frozenset({"gen"}))  # candidate added a needs edge
    gen = instance("gen", {"runs-on": "ubuntu-latest", "steps": [{"run": "make"}]})
    ci = emit_program(program(j, gen))["jobs"]["ci"]
    assert ci["needs"] == ["gen"]
    assert ci["if"] == "${{ !cancelled() && github.event_name == 'push' }}"


def test_negated_condition_is_wrapped():
    raw = {"runs-on": "ubuntu-latest", "steps": [{"run": "x", "if": "!cancelled()"}]}
    j = instance("ci", raw)
    assert emit_program(program(j))["jobs"]["ci"]["steps"][0]["if"] == "!cancelled()"
    s2 = replace(j.steps[0], cond=expr.parse("!cancelled() && github.event_name == 'push'"))
    p = program(with_steps(j, [s2]))
    assert emit_program(p)["jobs"]["ci"]["steps"][0]["if"] == \
        "${{ !cancelled() && github.event_name == 'push' }}"
    assert yaml.safe_load(emit(p))["jobs"]["ci"]["steps"][0]["if"].startswith("${{ !cancelled()")


def test_binding_used_when_raw_is_inconsistent():
    raw = {"runs-on": "ubuntu-latest", "defaults": {"run": {"working-directory": "packages/core"}},
           "steps": [{"run": "npm test"}]}
    j = instance("ci", raw)
    b = replace(j.steps[0].binding, working_directory="packages/core")  # from job defaults
    step = replace(j.steps[0], binding=b)
    job = emit_program(program(with_steps(j, [step])))["jobs"]["ci"]
    # The job's own defaults are kept, so the raw step stays verbatim.
    assert job["steps"][0] == {"run": "npm test"}
    assert job["defaults"] == {"run": {"working-directory": "packages/core"}}
    # A synthesized step without a working directory cannot live under those defaults: the
    # defaults are dropped and the raw step becomes explicit.
    extra = Step(Binding("run", "npm run lint"), SUCCESS, Contract(template="lint"), ("lint", 0))
    job = emit_program(program(with_steps(j, [step, extra])))["jobs"]["ci"]
    assert job["steps"][0] == {"run": "npm test", "working-directory": "packages/core"}
    assert "defaults" not in job


def test_workflow_defaults_keep_raw_steps():
    raw = {"runs-on": "ubuntu-latest", "steps": [{"run": "npm test"}]}
    j = instance("ci", raw)
    b = replace(j.steps[0].binding, shell="bash")
    p = program(with_steps(j, [replace(j.steps[0], binding=b)]), defaults={"run": {"shell": "bash"}})
    wf = emit_program(p)
    assert wf["defaults"] == {"run": {"shell": "bash"}}
    assert wf["jobs"]["ci"]["steps"][0] == {"run": "npm test"}


def test_frozen_job_verbatim_with_needs_rewritten():
    raw_frozen = {"name": "Deploy", "needs": ["test"], "runs-on": "ubuntu-latest",
                  "container": "node:20", "timeout-minutes": 10, "steps": [{"run": "deploy"}]}
    insts = [instance("test", RAW_TEST, c) for c in COMBOS]
    f = instance("deploy", raw_frozen, needs=[i.id for i in insts], frozen=True, steps=())
    assert emit_program(program(*insts, f))["jobs"]["deploy"] == raw_frozen
    # When the job it needs changes, the frozen job's needs follow the emitted keys.
    p2 = shared_build_program()
    f2 = instance("deploy", raw_frozen, needs=["ghav-build"], frozen=True, steps=())
    deploy = emit_program(replace(p2, jobs=p2.jobs + (f2,)))["jobs"]["deploy"]
    assert deploy["needs"] == ["ghav-build"]
    assert list(deploy) == list(raw_frozen)


def test_duplicate_step_ids_are_renamed_and_references_follow():
    a = {"runs-on": "ubuntu-latest", "steps": [
        {"id": "v", "run": "echo a"}, {"run": "use a", "if": "steps.v.outcome == 'success'"}]}
    b = {"runs-on": "ubuntu-latest", "steps": [
        {"id": "v", "run": "echo b"}, {"run": "use b", "if": "steps.v.outcome == 'success'"}]}
    ja, jb = instance("a", a), instance("b", b)
    merged = replace(ja, id="a+b", orig_job="a+b", check_name="a+b", raw=None,
                     steps=ja.steps + jb.steps)
    steps = emit_program(program(merged))["jobs"]["a-b"]["steps"]
    assert [s.get("id") for s in steps] == ["v", None, "v-2", None]
    assert steps[1]["if"] == "steps.v.outcome == 'success'"
    assert steps[3]["if"] == "success() && steps.v-2.outcome == 'success'"
    c = {"runs-on": "ubuntu-latest", "steps": [{"id": "v", "run": "echo c"},
                                               {"run": "echo ${{ steps.v.outputs.x }}"}]}
    bad = replace(merged, steps=ja.steps + instance("c", c).steps)
    with pytest.raises(EmitError):
        emit(program(bad))


def test_job_key_collisions_are_resolved_deterministically():
    raw = {"runs-on": "ubuntu-latest", "steps": [{"run": "x"}]}
    j1 = instance("build", raw)
    j2 = replace(instance("build", raw), id="build!", check_name="build!")
    j3 = replace(instance("o3", raw), id="o3[x=1]", orig_job="o3", matrix=(("x", "1"),),
                 check_name="o3 (1)", raw=None)
    j4 = replace(j3, id="o3[x=2]", matrix=(("x", "2"),), check_name="o3 (2)",
                 steps=j3.steps + j3.steps)
    keys = job_keys(program(j1, j2, j3, j4))
    assert keys == {"build": "build", "build!": "build-2", "o3[x=1]": "o3-x-1", "o3[x=2]": "o3-x-2"}


def test_workflow_level_fields_and_outputs():
    raw = {"runs-on": "ubuntu-latest", "outputs": {"v": "${{ steps.s.outputs.v }}"},
           "steps": [{"id": "s", "run": "echo v=1 >> $GITHUB_OUTPUT"}]}
    j = replace(instance("gen", raw), outputs=(("v", expr.parse("steps.s.outputs.v")),),
                env=(("A", "1"), ("B", "2")), permissions=(("contents", "read"),))
    synth = replace(j, id="ghav-x", orig_job="ghav-x", raw=None, check_name="ghav-x")
    p = program(j, synth, env=(("CI", "true"),), permissions={"contents": "read"},
                concurrency={"group": "ci-${{ github.ref }}", "cancel-in-progress": True},
                defaults={"run": {"shell": "bash"}}, source="o/r@abc:.github/workflows/ci.yml")
    wf = emit_program(p)
    assert list(wf) == ["name", "on", "permissions", "concurrency", "env", "defaults", "jobs"]
    assert wf["jobs"]["gen"]["outputs"] == {"v": "${{ steps.s.outputs.v }}"}
    assert wf["jobs"]["ghav-x"]["outputs"] == {"v": "${{ steps.s.outputs.v }}"}
    assert list(wf["jobs"]["gen"]["env"]) == ["A", "B"]
    assert wf["jobs"]["gen"]["permissions"] == {"contents": "read"}
    text = emit(p)
    assert text.startswith("# Generated by GHAV from o/r@abc:.github/workflows/ci.yml\n")
    assert "'on':" in text and "  pull_request:\n" in text
    assert "name" not in emit_program(replace(p, name=p.source))


def test_yaml_parses_back_to_the_same_mapping_and_is_deterministic():
    for p in (matrix_program(), shared_build_program()):
        text = emit(p)
        assert yaml.safe_load(text) == emit_program(p)
        assert emit(replace(p, jobs=tuple(reversed(p.jobs)))) == text


def test_multiline_run_uses_block_style():
    raw = {"runs-on": "ubuntu-latest", "steps": [{"run": "npm ci\nnpm test\n"}]}
    text = emit(program(instance("ci", raw)))
    assert "run: |\n" in text
    assert yaml.safe_load(text)["jobs"]["ci"]["steps"][0]["run"] == "npm ci\nnpm test\n"


# ---------------------------------------------------------------------------
# Round trip through lowering: lower(emit(p)) reproduces p
# ---------------------------------------------------------------------------

REPO = FIXTURES / "repo"
SOURCE = REPO / ".github" / "workflows" / "ci.yml"


def _lower(text: str, tmp_path: Path) -> Program:
    from ghav.lower import lower

    wf = tmp_path / ".github" / "workflows" / "ci.yml"
    wf.parent.mkdir(parents=True, exist_ok=True)
    wf.write_text(text)
    return lower(wf, tmp_path)


def _same_program(p: Program, q: Program, keys: dict[str, str]) -> None:
    """q (re-lowered emission) matches p up to job renaming."""
    for jp in p.jobs:
        key = keys[jp.id]
        cands = [j for j in q.jobs if (j.id == key or j.id.startswith(key + "["))
                 and j.matrix == jp.matrix]
        assert len(cands) == 1, jp.id
        jq = cands[0]
        assert jq.runs_on == jp.runs_on
        assert jq.frozen == jp.frozen
        assert jq.fail_fast_group is None or jp.fail_fast_group is not None
        assert {keys[n] for n in jp.needs} == {
            next(k for k in set(keys.values()) if n == k or n.startswith(k + "[")) for n in jq.needs}
        assert E.same_condition(jq.cond, jp.cond), jp.id
        if jp.frozen:
            continue
        assert len(jq.steps) == len(jp.steps)
        for sq, sp in zip(jq.steps, jp.steps):
            bp = sp.binding
            assert sq.binding == bp
            assert E.same_condition(sq.cond, sp.cond)


def test_round_trip_lowered_workflow(tmp_path):
    from ghav.lower import lower

    p = lower(SOURCE, REPO)
    text = emit(p)
    golden("ci.roundtrip.yml", text)
    q = _lower(text, tmp_path)
    assert sorted(j.id for j in q.jobs) == sorted(j.id for j in p.jobs)
    assert {j.id: j.check_name for j in q.jobs} == {j.id: j.check_name for j in p.jobs}
    _same_program(p, q, job_keys(p))
    # Emission is a fixed point on its own output (up to the header line).
    assert emit(q).split("\n", 1)[1] == text.split("\n", 1)[1]


def test_round_trip_synthesized_candidate(tmp_path):
    from ghav.lower import lower

    p = lower(SOURCE, REPO)
    tests = [j for j in p.jobs if j.orig_job == "test"]
    lint = p.job("lint")
    build = tests[0].steps[3]
    producer = replace(lint, id="ghav-build", orig_job="ghav-build", check_name="ghav build",
                       raw=None, steps=lint.steps[:3] + (replace(build, raw=None),))
    consumers = []
    for t in tests:
        (pack, upload), (down, unpack) = E.transfer_steps(producer, t, ["dist"], "ghav-0f0f")
        consumers.append(replace(t, needs=t.needs | {"ghav-build"},
                                 steps=t.steps[:3] + (down, unpack) + t.steps[4:]))
    producer = replace(producer, steps=producer.steps + (pack, upload))
    cand = replace(p, jobs=tuple(sorted([j for j in p.jobs if j.orig_job != "test"]
                                        + [producer] + consumers, key=lambda j: j.id)))
    text = emit(cand)
    golden("ci.candidate.yml", text)
    wf = yaml.safe_load(text)
    assert wf["jobs"]["test"]["strategy"] == {
        "fail-fast": True, **yaml.safe_load(SOURCE.read_text())["jobs"]["test"]["strategy"]}
    assert wf["jobs"]["test"]["needs"] == ["ghav-build", "lint"]
    q = _lower(text, tmp_path)
    _same_program(cand, q, job_keys(cand))


def test_strings_that_yaml_would_retype_are_quoted():
    raw = {"runs-on": "ubuntu-latest",
           "steps": [{"uses": "actions/checkout@v4", "with": {"ref": "45e7696", "depth": "0"}}]}
    j = instance("ci", raw)
    text = emit(program(with_steps(j, [replace(j.steps[0], raw=None)])))
    assert "ref: '45e7696'" in text and "depth: '0'" in text


def test_frozen_partial_matrix_keeps_only_its_combinations():
    raw = {"runs-on": "ubuntu-latest", "continue-on-error": "${{ matrix.advisory }}",
           "strategy": {"fail-fast": False, "matrix": {"include": [
               {"node": 20, "advisory": False}, {"node": 22, "advisory": True}]}},
           "steps": [{"run": "npm test"}]}
    ok = instance("pkg", raw, {"node": 20, "advisory": False}, fail_fast=False)
    frozen = instance("pkg", raw, {"node": 22, "advisory": True}, fail_fast=False, frozen=True)
    wf = emit_program(program(ok, frozen))
    assert len(wf["jobs"]) == 2
    combos = sorted(json_combo for body in wf["jobs"].values()
                    for json_combo in map(str, body["strategy"]["matrix"]["include"]))
    assert combos == [str({"node": 20, "advisory": False}), str({"node": 22, "advisory": True})]


def test_round_trip_inlines_local_reusable_workflow(tmp_path):
    from ghav.lower import lower

    src = FIXTURES / "reusable"
    p = lower(src / ".github" / "workflows" / "ci.yml", src)
    text = emit(p)
    golden("reusable.roundtrip.yml", text)
    wf = yaml.safe_load(text)
    keys = job_keys(p)
    build = wf["jobs"][keys["checks/build"]]
    assert build["name"] == "checks / Build"
    assert build["needs"] == ["prep"]
    assert build["if"] == "success() && github.event_name == 'push'"  # explicit status reference
    test = wf["jobs"][keys["checks/test[node=20]"]]
    # callee-internal `needs.build` refers to the renamed job
    assert f"needs.{keys['checks/build']}.result" in test["steps"][1]["run"]
    q = _lower(text, tmp_path)
    assert {j.check_name for j in q.jobs} == {j.check_name for j in p.jobs}
    for jp in p.jobs:
        jq = [j for j in q.jobs if j.check_name == jp.check_name and j.matrix == jp.matrix]
        assert len(jq) == 1
        assert [s.binding.key()[:2] for s in jq[0].steps if "needs" not in s.binding.text] == \
            [s.binding.key()[:2] for s in jp.steps if "needs" not in s.binding.text]


def test_condition_without_status_function_keeps_it_off():
    raw = {"runs-on": "ubuntu-latest", "steps": [{"run": "x"}]}
    j = instance("ci", raw)
    s2 = replace(j.steps[0], cond=expr.parse("github.event_name == 'push'"))  # no default fold
    s3 = replace(j.steps[0], cond=TRUE)
    steps = emit_program(program(with_steps(j, [s2, s3])))["jobs"]["ci"]["steps"]
    assert steps[0]["if"] == "${{ !cancelled() && github.event_name == 'push' }}"
    assert steps[1]["if"] == "${{ !cancelled() }}"


# ---------------------------------------------------------------------------
# Verification-gap resolutions: frozen calls kept verbatim, explicit fail-fast, timeouts
# ---------------------------------------------------------------------------

CALLEE = """
on:
  workflow_call:
    inputs:
      os: {type: string, required: true}
jobs:
  build:
    runs-on: ${{ inputs.os }}
    steps: [{run: npm ci}, {run: npm run build}]
  test:
    needs: build
    runs-on: ${{ inputs.os }}
    steps: [{run: npm test}]
"""


def _repo(tmp_path: Path, text: str) -> Path:
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True, exist_ok=True)
    (wf / "lib.yml").write_text(CALLEE)
    (wf / "ci.yml").write_text(text)
    return wf / "ci.yml"


def test_frozen_call_is_emitted_verbatim_and_round_trips(tmp_path):
    from ghav.checker import check
    from ghav.lower import lower

    path = _repo(tmp_path, """
on: [push, pull_request]
jobs:
  prep:
    runs-on: ubuntu-latest
    steps: [{run: echo prep}]
  call:
    needs: prep
    strategy:
      matrix: {os: [ubuntu-latest, macos-latest]}
    uses: ./.github/workflows/lib.yml
    with: {os: "${{ matrix.os }}"}
  after:
    needs: call
    runs-on: ubuntu-latest
    steps: [{run: echo after}]
""")
    p = lower(path, tmp_path)
    call_jobs = [j for j in p.jobs if j.orig_job.startswith("call/")]
    assert len(call_jobs) == 4 and all(j.frozen_reasons == ("fail-fast",) for j in call_jobs)
    wf = emit_program(p)
    assert sorted(wf["jobs"]) == ["after", "call", "prep"]
    assert wf["jobs"]["call"] == {"needs": "prep", "strategy": {"fail-fast": True, "matrix": {
        "os": ["ubuntu-latest", "macos-latest"]}}, "uses": "./.github/workflows/lib.yml",
        "with": {"os": "${{ matrix.os }}"}}
    assert wf["jobs"]["after"]["needs"] == ["call"]
    out = path.parent / "emitted.yml"
    out.write_text(emit(p))
    q = lower(out, tmp_path)
    assert sorted((j.id, j.frozen_reasons) for j in q.jobs) == sorted((j.id, j.frozen_reasons) for j in p.jobs)
    assert check(p, q).equivalent


def test_frozen_matrix_job_states_fail_fast():
    raw = {"runs-on": "ubuntu-latest", "services": {"db": {"image": "postgres"}},
           "strategy": {"matrix": {"node": [18]}}, "steps": [{"run": "npm test"}]}
    j = instance("test", raw, {"node": 18}, frozen=True)
    assert emit_program(program(j))["jobs"]["test"]["strategy"] == {"fail-fast": True, "matrix": {"node": [18]}}


def test_timeout_minutes_is_emitted_verbatim():
    raw = {"runs-on": "ubuntu-latest", "timeout-minutes": 30,
           "steps": [{"run": "npm test", "timeout-minutes": 5}]}
    j = instance("ci", raw)
    b = replace(j.steps[0].binding, text="npm run test")  # inconsistent raw: emitted from the binding
    job = emit_program(program(with_steps(j, [replace(j.steps[0], binding=b)])))["jobs"]["ci"]
    assert job["timeout-minutes"] == 30
    assert job["steps"][0] == {"run": "npm run test", "timeout-minutes": 5}


def test_conditions_never_introduce_always():
    raw = {"runs-on": "ubuntu-latest", "steps": [{"run": "x"}]}
    j = replace(instance("ci", raw), needs=frozenset({"gen"}), cond=TRUE)
    gen = instance("gen", {"runs-on": "ubuntu-latest", "steps": [{"run": "make"}]})
    steps = [replace(j.steps[0], cond=TRUE)]
    wf = emit_program(program(with_steps(j, steps), gen))
    assert wf["jobs"]["ci"]["if"] == "${{ !cancelled() }}"
    assert wf["jobs"]["ci"]["steps"][0]["if"] == "${{ !cancelled() }}"
    assert "always()" not in emit(program(with_steps(j, steps), gen))
