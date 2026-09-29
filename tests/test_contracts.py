"""Tests for ghav.contracts (docs/CONTRACT_GUIDE.md)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from ghav import contracts as C
from ghav.wir import (
    CTX,
    DEP,
    EXT,
    OUT,
    OUTPUT,
    RUN,
    SRC,
    ST_JOB,
    ST_STEP,
    TOOL,
    TOP_CONTRACT,
    TRUE,
    Binding,
    Contract,
    JobInstance,
    Program,
    Region,
    Step,
)

FIX = Path(__file__).parent / "fixtures" / "contracts"
SYNTH = FIX / "synth"
TS_JEST = FIX / "ts_jest"
BERRY = FIX / "berry"
LIFECYCLE = FIX / "lifecycle"
GUIDE = Path(__file__).parent.parent / "docs" / "CONTRACT_GUIDE.md"

PLATFORM = Region(CTX, None, "platform")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run(text: str, *, env=(), wd=None, shell=None, step_id=None) -> Step:
    return Step(Binding("run", text, (), tuple(sorted(env)), wd, shell, None, step_id), TRUE,
                TOP_CONTRACT, ("job", 0))


def uses(ref: str, with_=(), env=(), step_id=None) -> Step:
    w = tuple(sorted(with_.items() if isinstance(with_, dict) else with_))
    return Step(Binding("uses", ref, w, tuple(sorted(env)), step_id=step_id), TRUE, TOP_CONTRACT,
                ("job", 0))


def job(*steps: Step, matrix=(), env=(), needs=(), jid="job", frozen=False, raw=None) -> JobInstance:
    return JobInstance(jid, jid.split("[")[0], jid, tuple(sorted(matrix)), "ubuntu-latest", TRUE,
                       frozenset(needs), tuple(steps), env=tuple(sorted(env)), frozen=frozen,
                       raw=raw)


def infer1(step: Step, root: Path = SYNTH, **kw) -> Contract:
    j = kw.pop("job", None) or job(step, **kw)
    return C.infer(step, j, root)


def explain1(step: Step, root: Path = SYNTH, **kw) -> C.StepReport:
    j = kw.pop("job", None) or job(step, **kw)
    return C.explain(step, j, root)


def R(*specs: str) -> frozenset[Region]:
    out = set()
    for s in specs:
        kind, _, key = s.partition(":")
        out.add(Region(kind, None, key or None))
    return frozenset(out)


def yaml_step(raw: dict) -> Step:
    """Tiny stand-in for lowering: one YAML step to a Step."""
    if "run" in raw:
        b = Binding("run", raw["run"], (), tuple(sorted((k, str(v)) for k, v in (raw.get("env") or {}).items())),
                    raw.get("working-directory"), raw.get("shell"), raw.get("name"), raw.get("id"))
    else:
        w = raw.get("with") or {}
        b = Binding("uses", raw["uses"], tuple(sorted((k, str(v)) for k, v in w.items())),
                    tuple(sorted((k, str(v)) for k, v in (raw.get("env") or {}).items())),
                    name=raw.get("name"), step_id=raw.get("id"))
    return Step(b, TRUE, TOP_CONTRACT, ("x", 0), raw)


def ts_jest_job(name: str, matrix=()) -> JobInstance:
    wf = yaml.safe_load((TS_JEST / "ci.yml").read_text())
    steps = tuple(yaml_step(s) for s in wf["jobs"][name]["steps"])
    return job(*steps, matrix=matrix, jid=name)


# ---------------------------------------------------------------------------
# shell splitting
# ---------------------------------------------------------------------------


def words(script: str) -> list[tuple[str, ...]]:
    return [c.words for c in C.split_commands(script)]


def test_split_on_guide_separators():
    assert words("a 1\nb 2 && c 3 || d 4; e 5") == [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4"),
                                                  ("e", "5")]


def test_split_pipes_quotes_comments_continuations():
    assert words("npm test | tee log") == [("npm", "test"), ("tee", "log")]
    assert words("# comment\necho 'a b' \"c d\"  # trailing") == [("echo", "a b", "c d")]
    assert words("npx jest \\\n  --ci") == [("npx", "jest", "--ci")]
    assert words("echo a#b") == [("echo", "a#b")]


def test_split_keeps_expressions_as_one_word():
    assert words("npm i ${{ matrix.a || 'x y' }} z") == [("npm", "i", "${{ matrix.a || 'x y' }}", "z")]


def test_split_redirections():
    ok = C.split_commands("tsc > /dev/null 2>&1; eslint . >&2")
    assert [c.problem for c in ok] == [None, None]
    assert ok[0].words == ("tsc",)
    bad = C.split_commands('echo "v=1" >> "$GITHUB_OUTPUT"')
    assert bad[0].problem and "redirection" in bad[0].problem


@pytest.mark.parametrize("script,needle", [
    ("npm start &", "background"),
    ("echo $(git rev-parse HEAD)", "substitution"),
    ("echo `date`", "substitution"),
    ("cat <<EOF\nx\nEOF", "here-document"),
    ("(cd a && make)", "subshell"),
    ("node < input.txt", "input"),
])
def test_split_unsupported_constructs(script, needle):
    problems = [c.problem for c in C.split_commands(script) if c.problem]
    assert problems and needle in problems[0]


# ---------------------------------------------------------------------------
# expressions and paths
# ---------------------------------------------------------------------------


def test_expression_refs():
    refs, funcs = C.expression_refs(
        "matrix['node-version'] == '18' && startsWith(github.event.pull_request.head.ref, 'x') "
        "|| hashFiles('**/lock') && 'github.sha'")
    assert ("matrix", "node-version") in refs
    assert ("github", "event", "pull_request", "head", "ref") in refs
    assert ("github", "sha") not in refs  # inside a string literal
    assert funcs == ("startsWith", "hashFiles")


@pytest.mark.parametrize("path,kinds", [
    ("dist", {OUT}), ("dist/**", {OUT}), ("./lib/index.js", {OUT}), ("my-out", {OUT}),
    ("node_modules/.cache", {DEP}), ("coverage/lcov.info", {TOOL}), ("x.tsbuildinfo", {TOOL}),
    ("~/.npm", {TOOL}), ("${{ runner.temp }}/x", {TOOL}),
    ("${{ github.workspace }}/dist", {OUT}),
    ("packages/a/dist", {SRC, OUT}), ("report.json", {SRC, OUT, TOOL}),
    (".", {SRC, DEP, OUT, TOOL}), ("${{ matrix.dir }}", {SRC, DEP, OUT, TOOL}),
])
def test_path_kinds(path, kinds):
    assert C.path_kinds(path, ("my-out",)) == frozenset(kinds)


# ---------------------------------------------------------------------------
# templates (guide 5), one test per template
# ---------------------------------------------------------------------------


def test_checkout():
    # Checkout resets the workspace: it reads and writes every workspace-resident kind.
    c = infer1(uses("actions/checkout@v4"))
    assert c.template == "checkout"
    assert c.reads == R("ctx:platform", "ctx:github.sha", "ctx:github.ref", "src", "dep", "out", "tool")
    assert c.writes == R("src", "dep", "out", "tool") and c.out_paths == ()
    # fetch-depth 0 fetches every ref: the refs are an external read.
    assert infer1(uses("actions/checkout@v4", {"fetch-depth": "0"})).reads == c.reads | R("ext:refs")
    assert infer1(uses("actions/checkout@v4", {"fetch-depth": "${{ inputs.depth }}"})).reads >= R("ext:refs")
    assert infer1(uses("actions/checkout@v4", {"fetch-depth": "1"})).reads == c.reads
    ok = infer1(uses("actions/checkout@v4", {"ref": "${{ github.event.pull_request.head.sha }}"}))
    assert Region(CTX, None, "github.event.pull_request.head.sha") in ok.reads
    assert infer1(uses("actions/checkout@v4", {"ref": "main"})).top
    assert infer1(uses("actions/checkout@v4", {"repository": "other/repo"})).top


def test_setup_runtime():
    c = infer1(uses("actions/setup-node@v4", {"node-version": "${{ matrix.node }}", "cache": "npm"}),
               matrix=(("node", "20"),))
    assert c.template == "setup-runtime"
    assert c.reads == R("ctx:platform", "src", "ctx:matrix.node")
    assert c.writes == R("dep")
    assert infer1(uses("pnpm/action-setup@v4", {"version": "9"})).template == "setup-runtime"
    assert infer1(uses("pnpm/action-setup@v4", {"run_install": "true"})).top
    assert not infer1(uses("pnpm/action-setup@v4", {"run_install": "false"})).top


@pytest.mark.parametrize("cmd,root", [
    ("npm ci", SYNTH), ("npm ci --ignore-scripts", TS_JEST), ("pnpm install", SYNTH),
    ("pnpm i --frozen-lockfile", SYNTH), ("yarn install --frozen-lockfile", SYNTH),
    ("yarn --immutable", SYNTH), ("yarn", BERRY), ("yarn install", BERRY),
    ("bun install --frozen-lockfile", SYNTH), ("nci", SYNTH),
])
def test_install_pinned(cmd, root):
    c = infer1(run(cmd), root)
    assert c.template == "install-pinned"
    assert c.reads == R("ctx:platform", "src", "dep")  # pinned-input rule: no ext read
    assert c.writes == R("dep", "tool")


@pytest.mark.parametrize("cmd,src_write", [
    ("npm i -g pnpm", False), ("npm install --no-save typescript@5", False),
    ("npm install left-pad", True), ("npm install", False), ("pnpm add -g turbo", False),
    ("pnpm add zod", True), ("pnpm install --no-frozen-lockfile", False), ("yarn", False),
    ("yarn add zod", True), ("yarn global add turbo", False), ("bun install", False),
])
def test_install_unpinned(cmd, src_write):
    c = infer1(run(cmd))
    assert c.template == "install-unpinned"
    assert Region(EXT, None, "registry") in c.reads
    assert c.writes == (R("dep", "tool", "src") if src_write else R("dep", "tool"))


def test_build_out_paths_from_tsconfig_and_flags():
    c = infer1(run("tsc"))
    assert c.template == "build"
    assert c.reads == R("ctx:platform", "src", "dep", "out")
    assert c.writes == R("out", "tool")
    assert c.out_paths == ("lib",)  # outDir of tsconfig.json overrides its base (JSONC parsed)
    assert infer1(run("vite build --outDir web-dist")).out_paths == ("web-dist",)
    assert infer1(run("tsup src/index.ts")).out_paths == ("lib",)  # guide: tsconfig outDir
    assert infer1(run("babel src -d out-babel")).out_paths == ("out-babel",)
    assert infer1(run("tsc", wd="packages/app")).out_paths == ("packages/app/build",)
    assert infer1(run("tsc"), TS_JEST).out_paths == ("dist",)  # no outDir -> dist
    assert infer1(run("tsc -p tsconfig.build.json"), TS_JEST).out_paths == ("dist",)
    assert infer1(run("babel src")).top  # babel only with -d/--out-dir


def test_tsc_with_noemit_config_is_typecheck():
    c = infer1(run("tsc -p tsconfig.check.json"))
    assert c.template == "typecheck"
    assert c.writes == R("tool") and c.out_paths == ()


def test_build_orchestrated():
    for cmd in ("turbo run build", "nx affected -t build", "lerna run build", "pnpm -r build",
                "pnpm --filter web run build"):
        c = infer1(run(cmd))
        assert c.template == "build-orchestrated", cmd
        assert c.writes == R("out", "tool", "src") and c.out_paths == ()
    assert infer1(run("pnpm -r exec rm -rf dist")).top


def test_typecheck():
    c = infer1(run("tsc --noEmit -p tsconfig.json"))
    assert c.template == "typecheck"
    assert c.reads == R("ctx:platform", "src", "dep", "out") and c.writes == R("tool")
    assert infer1(run("vue-tsc --noEmit")).template == "typecheck"


def test_lint():
    c = infer1(run("eslint ."))
    assert c.template == "lint" and c.writes == frozenset()
    assert infer1(run("eslint . --cache")).writes == R("tool")
    for cmd in ("prettier --check .", "biome check .", "biome ci", "oxlint", "knip", "publint",
                "oxfmt --check", "madge --circular src", "xo", "stylelint '**/*.css'"):
        assert infer1(run(cmd)).template == "lint", cmd
    assert infer1(run("biome check --write .")).top  # not in the guide: top, not lint
    assert infer1(run("prettier .")).top


def test_format_write():
    for cmd in ("prettier --write .", "biome format --write .", "eslint --fix ."):
        c = infer1(run(cmd))
        assert c.template == "format-write", cmd
        assert c.writes == R("src", "tool")
    # script arguments reach the script body
    assert infer1(run("npm run lint -- --fix")).template == "format-write"


def test_test():
    for cmd in ("jest", "vitest run --coverage", "mocha", "ava", "node --test",
                "playwright test", "cypress run"):
        c = infer1(run(cmd))
        assert c.template == "test", cmd
        assert c.writes == R("tool")
    assert infer1(run("jest -u")).writes == R("tool", "src")
    assert infer1(run("jest --watch")).top


def test_upload_artifact():
    step = uses("actions/upload-artifact@v4", {"name": "dist-${{ matrix.os }}", "path": "dist\n!dist/*.map"})
    c = infer1(step, matrix=(("os", "ubuntu"),))
    assert c.template == "upload-artifact"
    assert c.reads == R("ctx:platform", "out", "ctx:matrix.os")
    assert c.writes == frozenset({Region(RUN, None, "dist-ubuntu")})
    default_name = infer1(uses("actions/upload-artifact@v4", {"path": "coverage"}))
    assert default_name.writes == frozenset({Region(RUN, None, "artifact")})
    assert default_name.reads == R("ctx:platform", "tool")
    assert infer1(uses("actions/upload-artifact@v4", {"name": "x"})).top  # no path
    assert infer1(uses("actions/upload-artifact@v4",
                       {"name": "r-${{ strategy.job-index }}", "path": "dist"})).top


def test_download_artifact():
    c = infer1(uses("actions/download-artifact@v4", {"name": "dist", "path": "dist"}))
    assert c.template == "download-artifact"
    assert c.reads == frozenset({PLATFORM, Region(RUN, None, "dist")})
    assert c.writes == R("out") and c.out_paths == ("dist",)
    assert infer1(uses("actions/download-artifact@v4", {"path": "dist"})).top  # all artifacts
    assert infer1(uses("actions/download-artifact@v4", {"pattern": "d-*", "name": "x"})).top


def test_cache_has_no_correctness_effect():
    c = infer1(uses("actions/cache@v4", {"path": "~/.npm", "key": "${{ hashFiles('**/lock') }}"}))
    assert c.template == "cache"
    assert c.reads == frozenset({PLATFORM}) and c.writes == frozenset()
    assert infer1(uses("actions/cache/restore@v4", {"key": "k"})).template == "cache"


def test_report_upload():
    c = infer1(uses("codecov/codecov-action@v4", {"token": "${{ secrets.CODECOV_TOKEN }}"}))
    assert c.template == "report-upload"
    assert c.reads == R("ctx:platform", "src", "tool", "out", "ctx:github.sha",
                        "ctx:secrets.CODECOV_TOKEN")
    assert c.writes == R("ext:codecov")
    assert infer1(uses("SonarSource/sonarqube-scan-action@v6")).writes == R("ext:sonar")


def test_publish():
    c = infer1(run("npm publish --provenance"))
    assert c.template == "publish"
    assert c.reads == R("ctx:platform", "src", "dep", "out", "ctx:github.ref")
    assert c.writes == R("ext:npm-registry")
    assert infer1(run("npm run release")).template == "publish"  # changeset publish
    assert infer1(run("gh release create v1")).writes == R("ext:github-releases")
    assert infer1(uses("actions/deploy-pages@v4")).writes == R("ext:github-pages")


def test_noop():
    c = infer1(run("set -euo pipefail\ncd .\necho hello\nFOO=1\nexport BAR=2"))
    assert c.template == "noop+noop+noop+noop+noop"
    assert c.reads == frozenset({PLATFORM}) and c.writes == frozenset()
    assert infer1(run("")).template == "noop"
    # exit/true/false only set the step status (the task's ok), no region.
    assert infer1(run("exit 1")).template == "noop" and not infer1(run("false || exit 1")).top
    assert infer1(run('echo "::set-output name=x::1"')).top
    logs = uses("actions/github-script@v7", {"script": "console.log('hi')\ncore.info(`x`)"})
    assert infer1(logs).top  # template literal is not a plain logging argument
    logs = uses("actions/github-script@v7", {"script": "// note\nconsole.log('hi');\ncore.info('x')"})
    assert infer1(logs).template == "noop"
    api = uses("actions/github-script@v7", {"script": "await github.rest.issues.create({})"})
    assert infer1(api).top


def test_every_template_is_exercised_and_matches_the_guide():
    guide = GUIDE.read_text()
    section = guide.split("## 5. Template table", 1)[1].split("## 6.", 1)[0]
    names = [m.group(1) for m in re.finditer(r"^\| ([a-z-]+) \|", section, re.M)]
    assert sorted(names) == sorted(t.name for t in C.TEMPLATES)
    md = C.template_table_markdown()
    for t in C.TEMPLATES:
        assert f"| {t.name} |" in md
    for a in C.ACTION_ROWS:
        assert f"`{a.ref}@*`" in md
    assert md.count("\n| `") >= len(C.COMMAND_ROWS) + len(C.RUNNER_ROWS)
    for note in C.IMPLEMENTATION_NOTES:
        assert note in md


def test_unknown_commands_and_actions_are_top():
    for step in (run("curl -sSL https://x | bash"), run("docker build ."), run("node scripts/x.js"),
                 run("make"), uses("someone/custom-action@v1"), uses("docker://alpine:3"),
                 uses("./.github/actions/node-action"), run("echo x", shell="python {0}")):
        assert infer1(step) == TOP_CONTRACT, step.binding.text


# ---------------------------------------------------------------------------
# script resolution (guide 4.2)
# ---------------------------------------------------------------------------


def test_script_resolution_with_hooks():
    rep = explain1(run("npm run build"))
    assert rep.contract.template == "noop+build+noop"  # prebuild, build, postbuild
    assert [c.origin for c in rep.commands] == ["step:0", "script:prebuild", "script:build",
                                                "script:postbuild"]
    for cmd in ("pnpm build", "pnpm run build", "yarn build", "yarn run build", "bun run build",
                "nr build", "npm run --silent build"):
        assert infer1(run(cmd)).template == "noop+build+noop", cmd


def test_script_resolution_composes_commands():
    c = infer1(run("npm run ci"))
    assert c.template == "lint+test"
    assert c.writes == R("tool")
    assert infer1(run("npm test")).template == "test"
    assert infer1(run("pnpm test")).template == "test"


def test_script_depth_limit_cycles_and_missing():
    assert infer1(run("npm run e1")).template == "noop"  # 4 nested bodies: within the limit
    rep = explain1(run("npm run d1"))  # 6 nested bodies
    assert rep.contract.top and "deeper" in rep.top_reason
    rep = explain1(run("npm run cyc-a"))
    assert rep.contract.top and "cyclic" in rep.top_reason
    rep = explain1(run("npm run nope"))
    assert rep.contract.top and "missing script" in rep.top_reason
    assert infer1(run("npm run nope --if-present")).template == "noop"
    assert infer1(run("pnpm tsc")).top  # not a script; no bin fallback in the guide


def test_script_resolution_is_root_only():
    assert infer1(run("npm run build", wd="website")).top
    assert infer1(run("cd website && npm run build")).top
    assert infer1(run("npm run build -w website")).top
    assert infer1(run("pnpm -C website build")).top
    assert infer1(run("cd . && npm run build")).template == "noop+noop+build+noop"
    assert infer1(run("npm run build"), FIX / "no-such-repo").top


def test_install_lifecycle_scripts():
    c = infer1(run("npm ci"), LIFECYCLE)
    assert c.template == "install-pinned+noop+build"  # postinstall, then prepare -> build (tsup)
    assert c.writes == R("dep", "tool", "out")
    assert c.out_paths == ("dist",)
    assert infer1(run("npm ci --ignore-scripts"), LIFECYCLE).template == "install-pinned"
    assert infer1(run("npm ci", wd="website")).top  # website has a postinstall hook
    assert infer1(run("npm ci --ignore-scripts", wd="website")).template == "install-pinned"


def test_package_runners():
    c = infer1(run("npx tsc --noEmit"))
    assert c.template == "typecheck"  # typescript is a root dependency: local binary
    c = infer1(run("npx changeset publish"))
    assert c.template == "publish"
    c = infer1(run("npx -y publint"))
    assert c.template == "install-unpinned+lint"  # not a dependency: downloaded
    assert Region(EXT, None, "registry") in c.reads
    assert infer1(run("npx prettier@3 --check .")).template == "install-unpinned+lint"
    assert infer1(run("pnpm dlx @biomejs/biome ci")).template == "install-unpinned+lint"
    assert infer1(run("pnpm exec eslint .")).template == "lint"
    assert infer1(run("./node_modules/.bin/vitest run")).template == "test"
    assert infer1(run("npx some-unknown-tool")).top


# ---------------------------------------------------------------------------
# composite actions (guide 4.4)
# ---------------------------------------------------------------------------


def test_composite_union_and_cold_cache():
    c = infer1(uses("./.github/actions/setup"))
    assert c.template == "setup-runtime+cache+install-pinned"
    # hashFiles in the cache key and the internal steps.cache reference add no reads
    assert c.reads == R("ctx:platform", "src", "dep")
    assert c.writes == R("dep", "tool")


def test_nested_composite_and_failures():
    c = infer1(uses("./.github/actions/nested"))
    assert c.template == "setup-runtime+cache+install-pinned+noop+build+noop"
    assert c.out_paths == ("lib",)
    assert explain1(uses("./.github/actions/cycle-a")).top_reason.startswith("cyclic local action")
    assert infer1(uses("./.github/actions/missing")).top
    assert infer1(uses("./.github/actions/bad-step")).top


def test_composite_inputs_and_conditions():
    step = uses("./.github/actions/with-inputs", {"greeting": "${{ matrix.lang }}"})
    c = infer1(step, matrix=(("lang", "en"),), env=(("DEPLOY_TARGET", "prod"),))
    assert c.template == "noop+noop"
    assert c.reads == R("ctx:platform", "ctx:matrix.lang", "ctx:github.token",
                        "ctx:github.event_name", "ctx:env.DEPLOY_TARGET")


# ---------------------------------------------------------------------------
# context reads (guide 3)
# ---------------------------------------------------------------------------


def test_every_step_reads_platform():
    for step in (run("echo"), run("npm ci"), uses("actions/cache@v4")):
        assert PLATFORM in infer1(step).reads


def test_expression_reads_in_run_with_env():
    step = run("vitest run --shard=${{ matrix.shard }}/${{ strategy.job-total }}",
               env=(("TOKEN", "${{ secrets.T }}"), ("REF", "${{ github.head_ref }}")))
    c = infer1(step, matrix=(("shard", "1"),))
    assert c.reads == R("ctx:platform", "src", "dep", "out", "ctx:matrix.shard",
                        "ctx:job-identity", "ctx:secrets.T", "ctx:github.head_ref")


def test_env_names_read_only_when_they_appear():
    j_env = (("REPORTER", "dot"), ("NODE_OPTIONS", "--x"), ("UNUSED", "1"))
    c = infer1(run("npm run env-use"), env=j_env)
    assert Region(CTX, None, "env.REPORTER") in c.reads  # appears in the resolved script text
    assert Region(CTX, None, "env.UNUSED") not in c.reads
    c = infer1(run('node -e "process.env.NODE_OPTIONS" || true && vitest --x=${NODE_OPTIONS}'),
               env=j_env)
    assert c.top  # node -e is unknown
    c = infer1(run("vitest --x=${NODE_OPTIONS}"), env=j_env)
    assert Region(CTX, None, "env.NODE_OPTIONS") in c.reads
    shadowed = infer1(run("vitest --x=$REPORTER", env=(("REPORTER", "json"),)), env=j_env)
    assert Region(CTX, None, "env.REPORTER") not in shadowed.reads


def test_workflow_env_via_annotate():
    step = run("jest --ci --shard=$SHARD")
    p = Program("wf", {"push": None}, (job(step),), env=(("SHARD", "1/2"),))
    c = C.annotate(p, SYNTH).jobs[0].steps[0].contract
    assert Region(CTX, None, "env.SHARD") in c.reads


def test_job_identity_and_default_variables():
    c = infer1(run("jest --outputFile=reports/$GITHUB_JOB.json"))
    assert Region(CTX, None, "job-identity") in c.reads
    c = infer1(run("jest --ci --changedSince=$GITHUB_BASE_REF"))
    assert Region(CTX, None, "github.base_ref") in c.reads
    assert Region(CTX, None, "job-identity") not in c.reads
    assert infer1(run("jest ${{ github.job }}")).reads >= R("ctx:job-identity")


def test_step_and_needs_references():
    first = run("echo hi", step_id="ver")
    second = run("npm publish --tag ${{ steps.ver.outputs.tag }}")
    j = job(first, second, needs=("build", "test[node=18]", "test[node=20]"))
    c = C.infer(second, j, SYNTH)
    assert Region(ST_STEP, None, "0") in c.reads
    assert c.instantiate("job").reads >= {Region(ST_STEP, "job", "0")}
    c = C.infer(run("jest ${{ needs.build.outputs.shard }} ${{ needs.build.result }}"), j, SYNTH)
    assert Region(OUTPUT, "build", "shard") in c.reads
    assert Region(ST_JOB, "build", None) in c.reads
    assert C.infer(run("jest ${{ needs.test.outputs.x }}"), j, SYNTH).top  # matrix: ambiguous
    assert C.infer(run("jest ${{ steps.later.outputs.x }}"), j, SYNTH).top
    assert C.infer(run("jest ${{ toJSON(github) }}"), j, SYNTH).top


def test_matrix_selected_commands_resolve_per_instance():
    step = run("npm run ${{ matrix.task }}")
    lint = infer1(step, matrix=(("task", "lint"),))
    test = infer1(step, matrix=(("task", "test"),))
    assert lint.template == "lint" and test.template == "test"
    assert Region(CTX, None, "matrix.task") in lint.reads


def test_instantiate_fills_job_for_local_kinds():
    c = infer1(run("tsc"))
    e = c.instantiate("build[os=linux]")
    assert Region(SRC, "build[os=linux]") in e.reads
    assert Region(CTX, "build[os=linux]", "platform") in e.reads
    assert Region(OUT, "build[os=linux]") in e.writes


# ---------------------------------------------------------------------------
# ts-jest (real files at 3d7203b0)
# ---------------------------------------------------------------------------


def test_ts_jest_build_step_resolves_through_hooks_and_is_top():
    j = ts_jest_job("sonar-scan")
    build = j.steps[2]
    assert build.binding.text == "npm run build\nnpm run postbuild\n"
    rep = C.explain(build, j, TS_JEST)
    trace = [(c.origin, c.text, c.outcome) for c in rep.commands]
    assert trace == [
        ("step:2", "npm run build", "resolve"),
        ("script:prebuild", "rimraf --glob dist coverage *.tgz", "file-util"),
        ("script:build", "tsc -p tsconfig.build.json", "build"),
        ("script:postbuild", "node scripts/post-build.js", "top"),
        ("step:2", "npm run postbuild", "resolve"),
        ("script:postbuild", "node scripts/post-build.js", "top"),
    ]
    # rimraf is a file utility (guide 4.5); `node <script>` stays top, so the whole step is top.
    assert rep.contract == TOP_CONTRACT
    assert "node scripts/post-build.js" in rep.top_reason


def test_ts_jest_other_steps():
    j = ts_jest_job("sonar-scan")
    got = [C.infer(s, j, TS_JEST) for s in j.steps]
    assert [c.template for c in got] == [
        "checkout", "setup-runtime+cache+install-pinned", "top", "test", "report-upload"]
    setup = got[1]
    assert setup.reads == R("ctx:platform", "src", "dep") and setup.writes == R("dep", "tool")
    assert Region(CTX, None, "env.CI") not in got[3].reads  # step env, not job env
    assert got[4].reads >= R("ctx:secrets.SONAR_TOKEN")
    lint = ts_jest_job("lint")
    assert [C.infer(s, lint, TS_JEST).template for s in lint.steps] == [
        "checkout", "setup-runtime+cache+install-pinned", "lint", "lint"]


def test_ts_jest_typescript_compatibility_matrix():
    matrix = (("name", "TypeScript 4.3"), ("packages", "typescript@4.3.5"),
              ("test-args", "--api-major 4 --compiler-bin tsc"))
    j = ts_jest_job("typescript-compatibility", matrix=matrix)
    got = [C.infer(s, j, TS_JEST) for s in j.steps]
    assert [c.template for c in got] == [
        "checkout", "setup-runtime+cache+install-pinned", "top", "install-unpinned", "top"]
    assert got[3].reads == R("ctx:platform", "src", "dep", "ext:registry", "ctx:matrix.packages")


# ---------------------------------------------------------------------------
# annotate, report, determinism
# ---------------------------------------------------------------------------


def test_annotate_fills_every_step_including_frozen_jobs():
    a = job(uses("actions/checkout@v4"), run("npm ci"), jid="a")
    up = uses("actions/upload-artifact@v4", with_=(("name", "dist"), ("path", "dist")))
    f = job(run("npm ci"), up, jid="f", frozen=True)
    p = Program("wf", {"push": None}, (a, f))
    q = C.annotate(p, SYNTH)
    assert [s.contract.template for s in q.job("a").steps] == ["checkout", "install-pinned"]
    # Frozen jobs get their steps' guide contracts, so the semantics sees their uploads.
    assert [s.contract.template for s in q.job("f").steps] == ["install-pinned", "upload-artifact"]
    assert Region(RUN, None, "dist") in q.job("f").steps[1].contract.writes
    assert p.job("a").steps[0].contract == TOP_CONTRACT  # input unchanged
    assert C.annotate(p, SYNTH) == q  # deterministic


def test_report_summary():
    a = job(uses("actions/checkout@v4"), run("npm run e2e"), run("npm run ci"), jid="a")
    rep = C.report(Program("wf", {}, (a,)), SYNTH)
    assert rep["summary"]["steps"] == 3 and rep["summary"]["top_steps"] == 1
    assert rep["summary"]["template_counts"] == {"checkout": 1, "lint": 1, "test": 1, "top": 1}
    assert rep["steps"][1]["top_reason"].startswith("unknown command")


def test_default_working_directory_from_job_raw():
    step = run("tsc")
    j = job(step, raw={"defaults": {"run": {"working-directory": "packages/app"}}})
    assert C.infer(step, j, SYNTH).out_paths == ("packages/app/build",)
    assert C.infer(run("tsc"), job(run("tsc")), SYNTH, workflow_defaults={
        "run": {"working-directory": "packages/app"}}).out_paths == ("packages/app/build",)


# ---------------------------------------------------------------------------
# guide v1.1: file utilities (4.5), downloads into the runner temp (5), the transfer idiom (9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmd,reads,writes,out_paths", [
    ("rm -rf dist coverage", (), ("out", "tool"), ("dist",)),
    ("rimraf --glob dist coverage *.tgz", (), ("src", "dep", "out", "tool"), (".", "dist")),
    ("shx rm -rf lib", (), ("out",), ("lib",)),
    ("del-cli dist '!dist/keep'", (), ("out",), ("dist",)),
    ("mkdir -p dist/assets", (), ("out",), ("dist/assets",)),
    ("touch dist/.nojekyll", (), ("out",), ("dist/.nojekyll",)),
    ("cp -r README.md dist/", ("src", "out", "tool"), ("out",), ("dist",)),
    ("cp -t dist a.txt b.txt", ("src", "out", "tool"), ("out",), ("dist",)),
    ("mv dist/x.js lib/x.js", ("out",), ("out",), ("dist/x.js", "lib/x.js")),
    ("ln -s ../x node_modules/y", (), ("dep",), ()),
    ("sed -i 's/a/b/' src/x.ts", ("src", "out", "tool"), ("src", "out", "tool"), ("src/x.ts",)),
    ("sed -i.bak -e s/a/b/ README.md", ("src", "out", "tool"), ("src", "out", "tool"), ("README.md",)),
    ("json -I -f package.json -e 'this.private=true'", ("src", "out", "tool"), ("src", "out", "tool"),
     ("package.json",)),
    ("rm -rf ~/.npm node_modules", (), ("dep", "tool"), ()),
    ("tar -czf out.tgz -C dist .", ("out",), ("src", "out", "tool"), ("out.tgz",)),
])
def test_file_utilities(cmd, reads, writes, out_paths):
    c = infer1(run(cmd))
    assert c.template == "file-util", cmd
    assert c.reads == R("ctx:platform", *reads), cmd
    assert c.writes == R(*writes), cmd
    assert c.out_paths == out_paths, cmd


@pytest.mark.parametrize("cmd", [
    "rm -rf $FOO", "rm -rf /usr/lib/node", "cp a", "tar xzf a.tgz", "tar -cf x.tar -T list.txt",
    "tar -xf a.tar --strip-components=1 dist", "node scripts/post-build.js", "node -e 1",
    "sed s/a/b/ x", "json -f package.json version",
])
def test_file_utility_top_cases(cmd):
    assert infer1(run(cmd)).top, cmd


def test_file_utilities_follow_the_working_directory():
    assert infer1(run("rm -rf dist", wd="packages/a")).writes == R("src", "out")  # packages/a/dist
    assert infer1(run("cd dist && rm -rf assets")).out_paths == ("dist/assets",)


def test_download_into_the_runner_temp_writes_tool():
    c = infer1(uses("actions/download-artifact@v4", {"name": "x", "path": "${{ runner.temp }}"}))
    assert c.writes == R("tool") and c.out_paths == ()


def test_transfer_idiom_contracts_and_reserved_prefix():
    pack, up, down, unpack = C.transfer_bindings("ghav-a1", ("lib", "dist"))
    assert [C.parse_transfer(b) for b in (pack, up, down, unpack)] == [
        C.TransferStep("pack", "ghav-a1", ("dist", "lib")), C.TransferStep("upload", "ghav-a1"),
        C.TransferStep("download", "ghav-a1"), C.TransferStep("unpack", "ghav-a1", ("dist", "lib"))]

    def contract(b):
        return C.infer(Step(b, TRUE, TOP_CONTRACT, ("j", 0)), job(), SYNTH)

    cp, cu, cd, cx = map(contract, (pack, up, down, unpack))
    assert cp.template == "file-util" and Region(OUT) in cp.reads and cp.writes == R("tool")
    assert cu.reads == R("ctx:platform", "tool") and cu.writes == frozenset({Region(RUN, None, "ghav-a1")})
    assert cd.writes == R("tool")
    assert cx.writes == R("out") and cx.out_paths == ("dist", "lib")
    # The prefix is reserved: a plain upload or download of a ghav- name is top.
    assert infer1(uses("actions/upload-artifact@v4", {"name": "ghav-a1", "path": "dist"})).top
    assert infer1(uses("actions/download-artifact@v4", {"name": "ghav-a1", "path": "dist"})).top
    with pytest.raises(ValueError):
        C.transfer_bindings("dist", ("dist",))
    with pytest.raises(ValueError):
        C.transfer_bindings("ghav-x", ("../up",))
    with pytest.raises(ValueError):
        C.transfer_bindings("ghav-x", ("dist", "dist/sub"))


# ---------------------------------------------------------------------------
# cache transparency (guide 7)
# ---------------------------------------------------------------------------


def _annotated(*steps: Step) -> list[Step]:
    p = Program("wf", {"push": None}, (job(*steps, jid="a"),))
    return list(C.annotate(p, SYNTH, observed={}).job("a").steps)


def test_cache_restore_followed_by_a_pinned_install_stays_transparent():
    cache = uses("actions/cache@v4", {"path": "node_modules", "key": "${{ hashFiles('package-lock.json') }}"})
    steps = _annotated(cache, run("npm ci"), run("npm test"))
    assert steps[0].contract.template == "cache" and steps[0].contract.writes == frozenset()


def test_cache_restore_followed_by_a_dep_writer_is_opaque():
    cache = uses("actions/cache@v4", {"path": "node_modules", "key": "${{ hashFiles('package-lock.json') }}"})
    for later in (run("npm install --no-save typescript@4"), run("./scripts/anything.sh"),
                  run("rm -rf node_modules/.cache")):
        c = _annotated(cache, run("npm ci"), later)[0].contract
        assert c.template == "cache-opaque", later.binding.text
        assert c.reads >= R("src", "dep", "out", "tool") and c.writes >= R("dep", "tool")
    setup = uses("actions/setup-node@v4", {"node-version": "20", "cache": "npm"})
    c = _annotated(setup, run("npm ci"), run("./x.sh"))[0].contract
    assert c.template == "setup-runtime+cache-opaque"
    assert _annotated(uses("actions/setup-node@v4", {"node-version": "20"}), run("./x.sh"))[0].contract.template \
        == "setup-runtime"
    # A cache save does not restore; restores are also found inside composite actions.
    save = uses("actions/cache/save@v4", {"path": "node_modules", "key": "k"})
    assert _annotated(save, run("./x.sh"))[0].contract.template == "cache"
    ts = ts_jest_job("typescript-compatibility", matrix=(("name", "TypeScript 4.3"), ("packages", "typescript@4.3.5"),
                                                         ("test-args", "--api-major 4")))
    q = C.annotate(Program("ci", {"push": None}, (ts,)), TS_JEST, observed={})
    assert "cache-opaque" in q.jobs[0].steps[1].contract.template.split("+")
    rep = C.report(Program("ci", {"push": None}, (ts,)), TS_JEST, observed={})
    assert rep["summary"]["opaque_caches"] == 1
    assert rep["steps"][1]["note"].startswith("cache restore is opaque: step 2")


# ---------------------------------------------------------------------------
# observed contracts (guide 8)
# ---------------------------------------------------------------------------


def _trace_repo(tmp_path: Path, steps: list) -> tuple[Path, Program]:
    root = tmp_path / "resources" / "repos" / "own__repo" / "abc123"
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "package.json").write_text('{"scripts": {"build": "node scripts/build.js"}}')
    trace = tmp_path / "resources" / "traces" / "own__repo" / "abc123" / "own__repo__ci.json"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps({"format": "ghav-trace/1", "steps": steps}))
    p = Program("ci", {"push": None}, (job(run("npm run build"), run("node -e 1"), jid="a"),),
                source=".github/workflows/ci.yml")
    return root, p


def test_observed_contract_from_a_trace(tmp_path):
    entry = {"binding": {"kind": "run", "text": "npm run build"},
             "writes": ["dist/index.js", "dist/esm/index.mjs", "node_modules/.cache/x"],
             "external": []}
    root, p = _trace_repo(tmp_path, [entry])
    assert C.trace_path(root, p.source) == tmp_path / "resources/traces/own__repo/abc123/own__repo__ci.json"
    q = C.annotate(p, root)
    c = q.job("a").steps[0].contract
    assert c.template == "observed" and not c.top
    assert c.writes == R("out", "dep") and c.out_paths == ("dist",)
    assert c.reads == R("ctx:platform", "src", "dep", "out", "tool")
    assert q.job("a").steps[1].contract.top  # not in the trace
    assert C.annotate(p, root, observed={}).job("a").steps[0].contract.top  # no trace given
    rep = C.report(p, root)
    assert rep["summary"]["observed_steps"] == 1 and rep["steps"][0]["note"].startswith("observed contract")


def test_observed_contract_needs_mappable_paths_and_a_valid_file(tmp_path):
    root, p = _trace_repo(tmp_path, [{"binding": {"kind": "run", "text": "npm run build"},
                                      "writes": ["/usr/lib/x"]}])
    assert C.annotate(p, root).job("a").steps[0].contract.top
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "other"}')
    with pytest.raises(ValueError):
        C.load_observed(bad)
    assert C.load_observed(tmp_path / "missing.json") == {}
