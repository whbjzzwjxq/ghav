"""Tests for ghav.lower (GHA Core -> WIR)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ghav import expr as X
from ghav.lower import LowerError, expand_matrix, instance_id, load_yaml, lower, needs_members
from ghav.wir import TOP_CONTRACT, TRUE, Expr, Program

FIXTURES = Path(__file__).parent / "fixtures" / "lower"
TS_JEST = FIXTURES / "ts-jest"
SHIKI = FIXTURES / "shiki"
SYN = FIXTURES / "synthetic"


def syn(name: str) -> Program:
    return lower(SYN / ".github" / "workflows" / f"{name}.yml", SYN)


def by_id(p: Program) -> dict:
    return {j.id: j for j in p.jobs}


# ---------------------------------------------------------------------------
# YAML and matrix helpers
# ---------------------------------------------------------------------------

def test_load_yaml_uses_core_schema():
    doc = load_yaml("on: push\na: yes\nb: 012\nc: 3.10\nd: 2024-01-01\ne: True\nf: 0x1f\ng: ~\n")
    assert doc == {"on": "push", "a": "yes", "b": 12, "c": 3.1, "d": "2024-01-01", "e": True, "f": 31, "g": None}


def test_load_yaml_error():
    with pytest.raises(LowerError):
        load_yaml("a: [unclosed")


def test_expand_matrix_include_docs_example():
    combos, order = expand_matrix(
        {
            "fruit": ["apple", "pear"],
            "animal": ["cat", "dog"],
            "include": [
                {"color": "green"},
                {"color": "pink", "animal": "cat"},
                {"fruit": "apple", "shape": "circle"},
                {"fruit": "banana"},
                {"fruit": "banana", "animal": "cat"},
            ],
        }
    )
    assert combos == [
        {"fruit": "apple", "animal": "cat", "color": "pink", "shape": "circle"},
        {"fruit": "apple", "animal": "dog", "color": "green", "shape": "circle"},
        {"fruit": "pear", "animal": "cat", "color": "pink"},
        {"fruit": "pear", "animal": "dog", "color": "green"},
        {"fruit": "banana"},
        {"fruit": "banana", "animal": "cat"},
    ]
    assert order == ["fruit", "animal", "color", "shape"]


def test_expand_matrix_exclude_docs_example():
    combos, _ = expand_matrix(
        {
            "os": ["macos-latest", "windows-latest"],
            "version": [12, 14, 16],
            "environment": ["staging", "production"],
            "exclude": [
                {"os": "macos-latest", "version": 12, "environment": "production"},
                {"os": "windows-latest", "version": 16},
            ],
        }
    )
    assert len(combos) == 9
    assert {"os": "macos-latest", "version": 12, "environment": "production"} not in combos
    assert not any(c["os"] == "windows-latest" and c["version"] == 16 for c in combos)


def test_expand_matrix_include_only_and_errors():
    combos, order = expand_matrix({"include": [{"a": 1}, {"a": 2, "b": 3}]})
    assert combos == [{"a": 1}, {"a": 2, "b": 3}]
    assert order == ["a", "b"]
    with pytest.raises(LowerError):
        expand_matrix({"a": []})
    with pytest.raises(LowerError):
        expand_matrix({"a": "not-a-list"})
    with pytest.raises(LowerError):
        expand_matrix({"a": list(range(20)), "b": list(range(20))})


def test_instance_id():
    assert instance_id("test", ()) == "test"
    assert instance_id("test", (("os", "ubuntu"), ("node", "18"))) == "test[node=18,os=ubuntu]"


# ---------------------------------------------------------------------------
# ts-jest (kulshekhar/ts-jest@3d7203b0)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ts_jest() -> Program:
    return lower(TS_JEST / ".github" / "workflows" / "ci.yml", TS_JEST)


def test_ts_jest_instance_count(ts_jest):
    ids = [j.id for j in ts_jest.jobs]
    assert ids == sorted(ids)
    calls = [j.id for j in ts_jest.jobs if j.orig_job.startswith(("test-ubuntu/", "test-windows/"))]
    assert len(calls) == 2 * 3 * 3
    expected_calls = {
        f"test-{os}[node-version={v}]/{job}"
        for os in ("ubuntu", "windows")
        for v in (18, 20, 22)
        for job in ("unit-test", "e2e-test", "example-test")
    }
    assert set(calls) == expected_calls
    compat = [i for i in ids if i.startswith("typescript-compatibility[")]
    assert len(compat) == 4
    assert set(ids) - set(calls) - set(compat) == {"lint", "sonar-scan"}
    assert len(ids) == 24
    assert not any(j.frozen for j in ts_jest.jobs)


def test_ts_jest_program_fields(ts_jest):
    assert ts_jest.name == "CI"
    assert ts_jest.source == ".github/workflows/ci.yml"
    assert sorted(ts_jest.on) == ["pull_request", "push"]
    assert ts_jest.concurrency == {
        "group": "${{ github.workflow }}-${{ github.head_ref || github.run_id }}",
        "cancel-in-progress": True,
    }
    assert ts_jest.permissions == {"contents": "read"}


def test_ts_jest_reusable_instance(ts_jest):
    j = ts_jest.job("test-windows[node-version=20]/unit-test")
    assert j.orig_job == "test-windows/unit-test"
    assert j.check_name == "test-windows (20) / Unit tests on Node v20 on windows-latest"
    assert j.runs_on == "windows-latest"
    assert j.matrix == ()
    assert j.cond == TRUE
    assert j.needs == frozenset()
    assert j.permissions == (("checks", "write"), ("contents", "read"))
    assert j.fail_fast_group is None
    names = [s.binding.name for s in j.steps]
    assert names == ["Checkout 🛎️", "Setup workspace ⚙️", "Build 🔧", "Run tests 🧪"]
    setup = j.steps[1]
    assert setup.binding.kind == "uses"
    assert setup.binding.text == "./.github/actions/setup-workspace"  # composite action stays one step
    assert setup.binding.with_ == (("node-version", "20"), ("os", "windows-latest"))
    assert setup.raw["with"] == {"os": "windows-latest", "node-version": "20"}  # inputs substituted in raw
    assert j.steps[2].binding.text == "npm run build\nnpm run postbuild\n"
    assert j.steps[3].binding.env == (("CI", "true"), ("DISABLE_MOCKED_WARNING", "true"))
    assert [s.origin for s in j.steps] == [("test-windows/unit-test", i) for i in range(4)]
    assert all(s.cond == X.SUCCESS for s in j.steps)
    assert all(s.contract == TOP_CONTRACT for s in j.steps)
    assert j.raw["runs-on"] == "windows-latest"


def test_ts_jest_include_only_matrix(ts_jest):
    compat = [j for j in ts_jest.jobs if j.orig_job == "typescript-compatibility"]
    assert [j.check_name for j in compat] == [
        "TypeScript 4.3 compatibility",
        "TypeScript 5.9 compatibility",
        "TypeScript 6 compatibility API compatibility",
        "TypeScript 7 side-by-side aliases compatibility",
    ]
    j = compat[0]
    assert j.id == (
        "typescript-compatibility[name=TypeScript 4.3,packages=typescript@4.3.5,"
        "test-args=--api-major 4 --compiler-bin tsc]"
    )
    assert j.matrix == (
        ("name", "TypeScript 4.3"),
        ("packages", "typescript@4.3.5"),
        ("test-args", "--api-major 4 --compiler-bin tsc"),
    )
    assert j.steps[3].binding.text == (
        "npm install --ignore-scripts --legacy-peer-deps --no-save --package-lock=false typescript@4.3.5"
    )
    assert j.steps[3].binding.name == "Install TypeScript 4.3 🔧"
    # the original YAML keeps the matrix reference
    assert "${{ matrix.packages }}" in j.steps[3].raw["run"]
    assert j.fail_fast_group is None


def test_ts_jest_plain_jobs(ts_jest):
    lint = ts_jest.job("lint")
    assert lint.check_name == "lint"
    assert [s.binding.text for s in lint.steps][2:] == ["npm run lint", "npm run lint-prettier-ci"]
    assert lint.permissions is None
    sonar = ts_jest.job("sonar-scan")
    assert sonar.steps[4].binding.env == (("SONAR_TOKEN", "${{ secrets.SONAR_TOKEN }}"),)
    assert sonar.steps[0].binding.with_ == (("fetch-depth", "0"), ("fetch-tags", "false"))


def test_ts_jest_deterministic(ts_jest):
    again = lower(TS_JEST / ".github" / "workflows" / "ci.yml", TS_JEST)
    assert again == ts_jest


# ---------------------------------------------------------------------------
# shiki (shikijs/shiki@48cd2cc6)
# ---------------------------------------------------------------------------

def test_shiki():
    p = lower(SHIKI / ".github" / "workflows" / "ci.yml", SHIKI)
    ids = [j.id for j in p.jobs]
    assert len(ids) == 8
    assert {"lint", "typecheck"} <= set(ids)
    j = p.job("test[node=22.x,os=windows-latest]")
    assert j.check_name == "test (22.x, windows-latest)"
    assert j.runs_on == "windows-latest"
    assert j.permissions == (("id-token", "write"),)
    assert j.fail_fast_group is None
    assert j.steps[2].binding.with_ == (("node-version", "22.x"),)
    assert j.steps[2].binding.name == "Set node 22.x"
    assert p.job("lint").steps[3].binding.text == "npm i -g @antfu/ni"


# ---------------------------------------------------------------------------
# Synthetic workflows
# ---------------------------------------------------------------------------

def test_matrix_workflow():
    p = syn("matrix")
    assert p.on == {"push": None, "pull_request": None}
    jobs = by_id(p)
    include = [j for j in p.jobs if j.orig_job == "docs-include"]
    assert len(include) == 6
    assert jobs["docs-include[fruit=banana]"].check_name == "docs-include (banana)"
    assert jobs["docs-include[animal=cat,color=pink,fruit=apple,shape=circle]"].check_name == (
        "docs-include (apple, cat, pink, circle)"
    )
    # missing matrix keys interpolate as empty strings
    assert jobs["docs-include[fruit=banana]"].steps[0].binding.text == "echo banana  "
    assert all(j.fail_fast_group == "docs-include" for j in include)

    exclude = [j for j in p.jobs if j.orig_job == "docs-exclude"]
    assert len(exclude) == 9
    assert all(j.fail_fast_group is None for j in exclude)
    assert {j.runs_on for j in exclude} == {"macos-latest", "windows-latest"}


def test_matrix_typed_values():
    jobs = by_id(syn("matrix"))
    t = jobs["typed[experimental=false,py=3.10,word=yes]"]
    assert t.check_name == "Typed false"
    assert t.steps[0].cond == X.FALSE  # `if: matrix.experimental` with boolean false
    assert t.steps[1].cond == X.FALSE  # '3.10' is a string, not equal to '3.1'
    assert t.steps[2].binding.text == "echo yes"
    f = jobs["typed[experimental=true,py=3.1,word=yes]"]
    assert f.steps[0].cond == X.SUCCESS
    assert f.steps[1].cond == X.SUCCESS  # YAML float 3.10 == '3.1' after number coercion
    assert f.fail_fast_group == "typed"


def test_literal_name_gets_matrix_suffix():
    jobs = by_id(syn("matrix"))
    assert jobs["named[target=a]"].check_name == "Build (a)"


def test_needs_workflow():
    jobs = by_id(syn("needs"))
    build = jobs["build"]
    assert build.cond == TRUE
    conds = [X.unparse(s.cond) for s in build.steps]
    assert conds == [
        "success()",
        "success()",
        "failure()",
        "success() && github.event_name == 'push'",
        "!cancelled()",
        "false",
    ]
    assert build.outputs == (
        ("label", Expr("call", ("format", Expr("lit", ("v-{0}",)), Expr("step", ("v", "outputs.version"))))),
        ("version", Expr("step", ("v", "outputs.version"))),
    )
    assert build.steps[1].binding.step_id == "v"

    test18 = jobs["test[node=18]"]
    assert test18.needs == frozenset({"build"})
    assert test18.cond == X.SUCCESS
    assert test18.fail_fast_group == "test"
    assert test18.steps[0].binding.text == "echo ${{ needs.build.outputs.version }} 18"

    report = jobs["report"]
    assert report.needs == frozenset({"build", "test[node=18]", "test[node=20]"})
    assert report.cond == Expr("status", ("always",))

    assert X.unparse(jobs["deploy"].cond) == "success() && github.ref == 'refs/heads/main'"
    assert X.unparse(jobs["independent"].cond) == "github.event_name == 'push'"

    gate = jobs["gate"]
    # `needs.*.result` over the two-instance matrix `test` is an aggregate: the reader freezes.
    assert gate.frozen and gate.frozen_reasons == ("multi-instance-result", "opaque-needs-condition")
    assert X.uses_status_function(gate.cond)
    # `test` is a fail-fast matrix and `report` (downstream) runs under always(): rule (ii).
    assert test18.frozen_reasons == ("fail-fast",) and jobs["test[node=20]"].frozen_reasons == ("fail-fast",)
    assert not jobs["deploy"].frozen  # default condition downstream


@pytest.mark.parametrize(
    "job,reasons",
    [
        ("with-container", ("container",)),
        ("with-services", ("services",)),
        ("job-coe", ("continue-on-error",)),
        ("job-coe-false", ()),
        ("job-timeout", ()),  # timeout-minutes has no correctness effect (emitted verbatim)
        ("step-timeout", ("step-continue-on-error",)),
        ("job-concurrency", ("concurrency",)),
        ("remote", ("remote-reusable-workflow",)),
        ("self-hosted", ("self-hosted-runner",)),
        ("max-par[shard=1]", ("max-parallel",)),
        ("max-par[shard=2]", ("max-parallel",)),
        ("opaque-out", ("opaque-output",)),
        ("env-job", ("unsupported-key:environment",)),
        ("dyn-matrix", ("dynamic-matrix",)),
        ("dyn-runs-on", ("unresolved-runs-on",)),
        ("runner-group", ("runner-group",)),
        ("self-hosted-dynamic", ("self-hosted-runner", "unresolved-runs-on")),
        ("parallel-steps", ("unsupported-step-key:parallel",)),
        ("coe-matrix[experimental=true]", ("continue-on-error",)),
        ("coe-matrix[experimental=false]", ()),
        ("plain", ()),
    ],
)
def test_frozen_reasons(job, reasons):
    j = syn("frozen").job(job)
    assert j.frozen_reasons == reasons
    assert j.frozen == bool(reasons)


def test_frozen_jobs_keep_raw_and_shape():
    p = syn("frozen")
    remote = p.job("remote")
    assert remote.steps == () and remote.runs_on == ""
    assert remote.raw == {"uses": "octo-org/octo-repo/.github/workflows/reusable.yml@v1"}
    assert p.job("self-hosted").runs_on == "linux,self-hosted"
    par = p.job("parallel-steps")
    assert [s.binding.text for s in par.steps] == ["npm ci", "npm run lint", "npm test"]
    assert [s.origin for s in par.steps] == [("parallel-steps", 0), ("parallel-steps", 1), ("parallel-steps", 1)]
    assert p.job("self-hosted-dynamic").runs_on == "JobId=x-${{ github.run_id }},self-hosted"
    dyn = p.job("dyn-matrix")
    assert dyn.needs == frozenset({"opaque-out"})
    assert dyn.steps[0].binding.text == "echo ${{ matrix.x }}"  # matrix unknown: kept symbolic


def test_defaults_are_folded_into_bindings():
    p = syn("defaults")
    assert p.env == (("CI", "true"), ("TOKEN", "${{ secrets.TOKEN }}"))
    assert p.defaults == {"run": {"shell": "bash"}}
    a = p.job("a")
    assert a.env == (("NODE", "20"),)
    b0, b1, b2 = (s.binding for s in a.steps)
    assert (b0.working_directory, b0.shell) == ("web", "bash")
    assert (b1.working_directory, b1.shell) == ("api", "sh")
    assert (b2.kind, b2.text, b2.working_directory, b2.shell) == ("uses", "./.github/actions/setup", None, None)
    assert b2.with_ == (("node-version", "20"),)


def test_reusable_expansion():
    p = syn("caller")
    jobs = by_id(p)
    assert sorted(jobs) == [
        "after",
        "call-always/build",
        "call-always/test[shard=1]",
        "call-always/test[shard=2]",
        "call[os=macos-latest]/build",
        "call[os=macos-latest]/test[shard=1]",
        "call[os=macos-latest]/test[shard=2]",
        "call[os=ubuntu-latest]/build",
        "call[os=ubuntu-latest]/test[shard=1]",
        "call[os=ubuntu-latest]/test[shard=2]",
        "missing",
        "prep",
    ]
    b = jobs["call[os=macos-latest]/build"]
    assert b.orig_job == "call/build"
    assert b.check_name == "call (macos-latest) / build"
    assert b.runs_on == "macos-latest"
    assert b.needs == frozenset({"prep"})
    assert X.unparse(b.cond) == "success() && github.event_name == 'push'"
    assert b.env == (("LOCAL", "x"), ("SHARED", "callee-macos-latest"))
    assert b.permissions == (("contents", "read"),)  # inherited from the caller job
    assert b.fail_fast_group == "call"  # caller matrix with default fail-fast
    s0, s1 = b.steps
    # inputs: expression (needs output), number default, missing string -> ''
    assert s0.binding.text == 'echo ${{ needs.prep.outputs.tag }} 2 ""'
    assert s0.binding.env == (("TOKEN", "${{ secrets.NPM_TOKEN }}"),)  # passed secret
    assert s0.binding.working_directory == "packages/core"  # callee workflow defaults
    assert s1.cond == X.SUCCESS  # `if: inputs.flag` with flag: true
    assert s0.origin == ("call/build", 0)
    assert s0.raw["run"] == 'echo ${{ needs.prep.outputs.tag }} 2 ""'

    t = jobs["call[os=ubuntu-latest]/test[shard=2]"]
    assert t.orig_job == "call/test"
    assert t.check_name == "call (ubuntu-latest) / Test on ubuntu-latest"
    assert t.matrix == (("shard", "2"),)
    assert t.needs == frozenset({"prep", "call[os=ubuntu-latest]/build"})
    assert X.unparse(t.cond) == "success() && github.event_name == 'push'"
    assert t.steps[0].binding.text == "npm test -- --shard=2"
    assert t.steps[0].binding.working_directory == "."
    # callee-internal needs references point at the callee group of this caller instance
    assert t.steps[1].binding.text == "echo ${{ needs['call[os=ubuntu-latest]/build'].result }}"
    # The caller is a fail-fast matrix calling a reusable workflow (rule (iii)): the whole call
    # is frozen and kept verbatim, so every job of it carries the caller's YAML.
    assert t.frozen_reasons == ("fail-fast",) and b.frozen_reasons == ("fail-fast",)
    assert t.raw is b.raw and t.raw["uses"] == "./.github/workflows/callee.yml"
    assert t.steps[0].origin_instance == t.id

    after = jobs["after"]
    assert after.needs == frozenset(i for i in jobs if i.startswith("call["))
    assert after.cond == X.SUCCESS

    missing = jobs["missing"]
    assert missing.frozen_reasons == ("missing-local-workflow",)


def test_reusable_status_condition():
    jobs = by_id(syn("caller"))
    b = jobs["call-always/build"]
    assert b.cond == Expr("status", ("always",))
    assert b.steps[0].binding.text == 'echo  2 ""'
    assert b.steps[1].cond == X.FALSE  # flag defaults to false
    assert b.permissions is None
    t = jobs["call-always/test[shard=1]"]
    assert t.frozen_reasons == ("reusable-status-condition",)
    assert t.fail_fast_group == "call-always/test"
    # A frozen job freezes its whole call, which is then kept verbatim (raw = caller YAML).
    assert b.frozen_reasons == ("frozen-call",)
    assert b.raw == t.raw == {"needs": "prep", "if": "always()", "uses": "./.github/workflows/callee.yml",
                              "with": {"os": "ubuntu-latest"}}


def test_needs_members():
    p = syn("caller")
    assert needs_members(p, "call") == tuple(sorted(i.id for i in p.jobs if i.id.startswith("call[")))
    assert needs_members(p, "call[os=ubuntu-latest]/build") == ("call[os=ubuntu-latest]/build",)
    assert needs_members(p, "call-always/test") == ("call-always/test[shard=1]", "call-always/test[shard=2]")
    assert needs_members(p, "prep") == ("prep",)
    assert needs_members(syn("needs"), "test") == ("test[node=18]", "test[node=20]")


def test_cycle_is_error():
    with pytest.raises(LowerError):
        syn("cycle")


def test_invalid_workflows(tmp_path):
    wf = tmp_path / "w.yml"
    wf.write_text("jobs: {a: {runs-on: x, steps: [{run: a}]}}\n")
    with pytest.raises(LowerError):
        lower(wf, tmp_path)  # no `on`
    wf.write_text("on: push\njobs: {a: {needs: zzz, runs-on: x, steps: [{run: a}]}}\n")
    with pytest.raises(LowerError):
        lower(wf, tmp_path)
    wf.write_text("on: push\njobs: {a: {runs-on: x}}\n")
    with pytest.raises(LowerError):
        lower(wf, tmp_path)
    wf.write_text("on: push\njobs: {a: {runs-on: x, steps: [{name: nothing}]}}\n")
    with pytest.raises(LowerError):
        lower(wf, tmp_path)


def test_relative_workflow_path():
    p = lower(Path(".github/workflows/needs.yml"), SYN)
    assert p.source == ".github/workflows/needs.yml"
    assert p == syn("needs")


# ---------------------------------------------------------------------------
# Verification-gap resolutions (RESEARCH_SCOPE): fail closed, timeouts, origin instances,
# fail-fast, multi-instance readers, frozen calls
# ---------------------------------------------------------------------------

def lower_text(tmp_path: Path, text: str, extra: dict[str, str] | None = None) -> Program:
    wf = tmp_path / ".github" / "workflows" / "ci.yml"
    wf.parent.mkdir(parents=True, exist_ok=True)
    wf.write_text(text)
    for name, body in (extra or {}).items():
        (wf.parent / name).write_text(body)
    return lower(wf, tmp_path)


def test_origin_instance_names_the_lowered_instance():
    for p in (syn("needs"), syn("caller"), syn("frozen")):
        for j in p.jobs:
            assert all(s.origin_instance == j.id for s in j.steps), j.id


def test_timeout_minutes_is_kept_but_does_not_freeze(tmp_path):
    p = lower_text(tmp_path, """
on: push
jobs:
  a:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: npm test
        timeout-minutes: 5
""")
    a = p.job("a")
    assert not a.frozen and a.raw["timeout-minutes"] == 10 and a.steps[0].raw["timeout-minutes"] == 5


def test_lowering_fails_closed_on_unknown_keys(tmp_path):
    p = lower_text(tmp_path, """
on: push
jobs:
  strat:
    runs-on: ubuntu-latest
    strategy:
      matrix: {n: [1]}
      bogus: 1
    steps: [{run: x}]
  defs:
    runs-on: ubuntu-latest
    defaults: {run: {shell: bash, env: x}}
    steps: [{run: x}]
  normal-with:
    runs-on: ubuntu-latest
    with: {a: b}
    steps: [{run: x}]
  bg:
    runs-on: ubuntu-latest
    steps: [{run: x, background: true}]
  mode:
    runs-on: ubuntu-latest
    cache-mode: read
    steps: [{run: x}]
""")
    assert p.job("strat[n=1]").frozen_reasons == ("unsupported-strategy-key:bogus",)
    assert p.job("defs").frozen_reasons == ("unsupported-defaults-key:run.env",)
    assert p.job("normal-with").frozen_reasons == ("unsupported-key:with",)
    assert p.job("bg").frozen_reasons == ("unsupported-step-key:background",)
    assert p.job("mode").frozen_reasons == ("unsupported-key:cache-mode",)
    q = lower_text(tmp_path, "on: push\nsnapshot: x\njobs: {a: {runs-on: x, steps: [{run: a}]}}\n")
    assert q.job("a").frozen_reasons == ("unsupported-workflow-key:snapshot",)
    call = lower_text(tmp_path, """
on: push
jobs:
  c:
    runs-on: ubuntu-latest
    uses: ./.github/workflows/lib.yml
""", {"lib.yml": "on: workflow_call\nconcurrency: x\njobs: {b: {runs-on: x, steps: [{run: b}]}}\n"})
    assert call.job("c/b").frozen_reasons == ("concurrency", "unsupported-key:runs-on")


def test_may_run_after_cancel():
    from ghav.lower import may_run_after_cancel

    assert may_run_after_cancel(X.parse("always()"))
    assert may_run_after_cancel(X.parse("cancelled() || failure()"))
    assert may_run_after_cancel(X.parse("always() && github.event_name == 'push'"))
    assert not may_run_after_cancel(X.parse("success() && always()"))
    assert not may_run_after_cancel(X.parse("!cancelled()"))
    assert not may_run_after_cancel(X.parse("failure()"))
    assert not may_run_after_cancel(X.SUCCESS)


FF_MATRIX = """
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      {ff}matrix: {{node: [18, 20]}}
    outputs:
      v: ${{{{ steps.s.outputs.v }}}}
    steps:
      - uses: actions/checkout@v4
      - id: s
        run: npm test
      - if: {cond}
        uses: actions/upload-artifact@v4
        with: {{name: "r-${{{{ matrix.node }}}}", path: coverage}}
{down}
"""


@pytest.mark.parametrize("ff,cond,down,frozen", [
    ("", "always()", "", True),               # (i) a member writes P after a cancellation
    ("", "failure()", "", False),             # failure() is false after a cancellation
    ("", "success()", "", False),
    ("fail-fast: false\n      ", "always()", "", False),  # no fail-fast: no cancellation
    ("", "success()", "  gate:\n    needs: test\n    if: always()\n    runs-on: x\n    steps: [{run: y}]", True),  # (ii)
    ("", "success()", "  gate:\n    needs: test\n    runs-on: x\n    steps: [{run: 'echo ${{ needs.test.outputs.v }}'}]", True),
    ("", "success()", "  gate:\n    needs: test\n    if: github.ref == 'x'\n    runs-on: x\n    steps: [{run: y}]", False),
])
def test_fail_fast_rule(tmp_path, ff, cond, down, frozen):
    p = lower_text(tmp_path, FF_MATRIX.format(ff=ff, cond=cond, down=down))
    members = [j for j in p.jobs if j.orig_job == "test"]
    assert len(members) == 2
    assert all(("fail-fast" in j.frozen_reasons) == frozen for j in members)


MULTI = """
on: push
jobs:
  m:
    runs-on: ubuntu-latest
    strategy: {{fail-fast: false, matrix: {{n: [1, 2]}}}}
    outputs:
      same: fixed
      differs: ${{{{ steps.s.outputs.v }}}}
    steps:
      - id: s
        run: echo "v=${{{{ matrix.n }}}}" > out.txt
  reader:
    needs: m
    runs-on: ubuntu-latest
    steps:
      - run: echo {ref}
"""


@pytest.mark.parametrize("ref,reason", [
    ("${{ needs.m.result }}", "multi-instance-result"),
    ("${{ needs.m.outputs.differs }}", "multi-instance-output"),
    ("${{ needs.m.outputs.same }}", None),
    ("${{ toJSON(needs.m.outputs) }}", "multi-instance-output"),
])
def test_multi_instance_readers(tmp_path, ref, reason):
    p = lower_text(tmp_path, MULTI.format(ref=ref))
    reader = p.job("reader")
    assert reader.frozen_reasons == ((reason,) if reason else ())
    assert not any(j.frozen for j in p.jobs if j.orig_job == "m")


def test_a_call_named_by_needs_is_frozen_and_kept_verbatim(tmp_path):
    lib = """
on: workflow_call
jobs:
  a: {runs-on: x, steps: [{run: a}]}
  b: {runs-on: x, steps: [{run: b}]}
"""
    p = lower_text(tmp_path, """
on: push
jobs:
  c:
    uses: ./.github/workflows/lib.yml
  gate:
    needs: c
    if: always()
    runs-on: x
    steps:
      - run: test "${{ needs.c.result }}" = success
""", {"lib.yml": lib})
    assert p.job("gate").frozen_reasons == ("multi-instance-result",)
    for jid in ("c/a", "c/b"):
        j = p.job(jid)
        assert j.frozen_reasons == ("frozen-call",) and j.raw == {"uses": "./.github/workflows/lib.yml"}
    # Without the reference the call is expanded and not frozen.
    q = lower_text(tmp_path, "on: push\njobs:\n  c:\n    uses: ./.github/workflows/lib.yml\n", {"lib.yml": lib})
    assert not any(j.frozen for j in q.jobs) and q.job("c/a").raw == {"runs-on": "x", "steps": [{"run": "a"}]}


def test_needs_star_is_an_aggregate_only_over_a_multi_instance_need(tmp_path):
    base = """
on: push
jobs:
  a: {runs-on: x, steps: [{run: a}]}
  b: {runs-on: x, steps: [{run: b}]}
  m:
    runs-on: x
    strategy: {fail-fast: false, matrix: {n: [1, 2]}}
    steps: [{run: m}]
  gate:
    needs: [a, b%s]
    if: ${{ always() && contains(needs.*.result, 'failure') }}
    runs-on: x
    steps: [{run: g}]
"""
    assert "multi-instance-result" not in lower_text(tmp_path, base % "").job("gate").frozen_reasons
    assert "multi-instance-result" in lower_text(tmp_path, base % ", m").job("gate").frozen_reasons
