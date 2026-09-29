from __future__ import annotations

import pytest

from ghav import cost
from ghav.cost import (
    DEFAULT_CONTEXTS,
    JOB_OVERHEAD_SECONDS,
    UNKNOWN,
    EvalEnv,
    Prefix,
    evaluate,
    executes,
    expected_machine_time,
    loose_equal,
    machine_time,
    simulate,
    step_seconds,
    template_seconds,
)
from ghav.wir import TOP_CONTRACT, TRUE, Binding, Contract, Expr, JobInstance, Program, Step

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUCCESS = Expr("status", ("success",))
ALWAYS = Expr("status", ("always",))


def lit(v):
    return Expr("lit", (v,))


def ctx(k):
    return Expr("ctx", (k,))


def AND(a, b):
    return Expr("and", (a, b))


def OR(a, b):
    return Expr("or", (a, b))


def NOT(a):
    return Expr("not", (a,))


def EQ(a, b):
    return Expr("eq", (a, b))


def NE(a, b):
    return Expr("ne", (a, b))


def call(f, *args):
    return Expr("call", (f, *args))


def step(template, text=None, cond=SUCCESS, step_id=None, kind="run", origin=("j", 0)):
    return Step(
        binding=Binding(kind=kind, text=text or template, step_id=step_id),
        cond=cond,
        contract=Contract(template=template),
        origin=origin,
    )


def job(jid, steps, cond=TRUE, needs=(), matrix=(), runs_on="ubuntu-latest", frozen=False,
        raw=None, env=(), orig=None):
    return JobInstance(
        id=jid, orig_job=orig or jid.split("[")[0], check_name=jid, matrix=tuple(matrix),
        runs_on=runs_on, cond=cond, needs=frozenset(needs), steps=tuple(steps), env=tuple(env),
        frozen=frozen, raw=raw,
    )


def program(*jobs, on=None, env=()):
    return Program(name="wf", on=on if on is not None else {"push": None, "pull_request": None},
                   jobs=tuple(sorted(jobs, key=lambda j: j.id)), env=tuple(env))


def env(**ctxvals):
    return EvalEnv(ctx=ctxvals)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def test_table_matches_cost_model_doc():
    assert JOB_OVERHEAD_SECONDS == 15
    expected = {
        "checkout": 5, "setup-runtime": 10, "install-pinned": 45, "install-unpinned": 45,
        "build": 60, "build-orchestrated": 90, "typecheck": 40, "lint": 30,
        "format-write": 20, "test": 120, "upload-artifact": 10, "download-artifact": 8,
        "cache": 5, "report-upload": 10, "publish": 30, "noop": 1, "top": 30,
        # guide v1.1 templates copy the row of what they refine
        "file-util": 1, "cache-opaque": 5, "observed": 30,
    }
    assert dict(cost.TEMPLATE_SECONDS) == expected


def test_template_seconds_unknown_and_composite():
    assert template_seconds("test") == 120
    assert template_seconds("unknown") == 30
    assert template_seconds("") == 30
    assert template_seconds("install-pinned+build") == 105
    assert step_seconds(Step(Binding("run", "x"), SUCCESS, TOP_CONTRACT, ("j", 0))) == 30


def test_transfer_step_priced_from_action_reference():
    s = Step(Binding("uses", "actions/upload-artifact@v4", with_=(("name", "ghav-x"),)),
             SUCCESS, Contract(template="unknown"), ("j", 0))
    assert step_seconds(s) == 10
    d = Step(Binding("uses", "actions/download-artifact@v4"), SUCCESS, Contract(), ("j", 0))
    assert step_seconds(d) == 8


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def test_loose_equality_rules():
    assert loose_equal("Push", "push") is True
    assert loose_equal("18", 18) is True
    assert loose_equal(None, 0) is True
    assert loose_equal(None, None) is True
    assert loose_equal(True, "1") is True
    assert loose_equal("abc", 0) is False  # NaN
    assert loose_equal("", 0) is True
    assert loose_equal(UNKNOWN, "x") is None
    assert loose_equal(Prefix("refs/pull/"), "refs/heads/main") is False
    assert loose_equal(Prefix("refs/pull/"), "refs/pull/1/merge") is None


def test_and_or_not_three_valued():
    e = env()
    unk = ctx("github.repository")
    assert evaluate(AND(unk, lit(False)), e) is False
    assert evaluate(AND(unk, lit(True)), e) is UNKNOWN
    assert evaluate(OR(unk, lit(True)), e) is True
    assert evaluate(OR(unk, lit(False)), e) is UNKNOWN
    assert evaluate(NOT(unk), e) is UNKNOWN
    assert evaluate(AND(lit("a"), lit("b")), e) == "b"
    assert evaluate(OR(lit(""), lit("b")), e) == "b"


def test_unknown_and_opaque_count_as_true():
    e = env()
    assert executes(Expr("opaque", ("weird[0]", "github.x")), e)
    assert executes(NOT(Expr("opaque", ("weird[0]",))), e)
    assert executes(EQ(ctx("github.repository"), lit("o/r")), e)
    assert executes(NE(ctx("github.repository"), lit("o/r")), e)
    assert not executes(lit(False), e)


def test_functions():
    e = env(**{"github.event_name": "push"})
    arr = call("fromJSON", lit('["push", "pull_request"]'))
    assert evaluate(call("contains", arr, ctx("github.event_name")), e) is True
    assert evaluate(call("contains", lit("Hello World"), lit("world")), e) is True
    assert evaluate(call("startsWith", lit("refs/tags/v1"), lit("refs/TAGS/")), e) is True
    assert evaluate(call("endsWith", lit("abc"), lit("x")), e) is False
    assert evaluate(call("format", lit("{0}-{1}{{x}}"), lit("a"), lit(1)), e) == "a-1{x}"
    assert evaluate(call("hashFiles", lit("**/package-lock.json")), e) is UNKNOWN
    assert evaluate(call("join", call("fromJSON", lit("[1,2]")), lit("+")), e) == "1+2"


def test_prefix_ref_on_pull_request():
    p_env = EvalEnv(ctx={"github.ref": Prefix("refs/pull/")})
    assert not executes(EQ(ctx("github.ref"), lit("refs/heads/main")), p_env)
    assert not executes(call("startsWith", ctx("github.ref"), lit("refs/tags/")), p_env)
    assert executes(call("startsWith", ctx("github.ref"), lit("refs/pull")), p_env)


def test_step_and_needs_references():
    e = EvalEnv(ctx={}, steps={"a": True, "b": False}, needs={"build": False})
    assert evaluate(Expr("step", ("a", "outcome")), e) == "success"
    assert evaluate(Expr("step", ("b", "conclusion")), e) == "skipped"
    assert evaluate(Expr("step", ("a", "outputs.cache-hit")), e) is UNKNOWN
    assert evaluate(Expr("step", ("missing", "outcome")), e) is None
    assert evaluate(Expr("needs", ("build", "result")), e) == "skipped"


# ---------------------------------------------------------------------------
# Program cost
# ---------------------------------------------------------------------------

def test_single_job_machine_time():
    p = program(job("ci", [step("checkout"), step("install-pinned"), step("test")]))
    assert machine_time(p, "push") == 15 + 5 + 45 + 120
    assert expected_machine_time(p) == 185


def test_event_condition_on_job():
    only_push = AND(SUCCESS, EQ(ctx("github.event_name"), lit("push")))
    p = program(job("deploy", [step("publish")], cond=EQ(ctx("github.event_name"), lit("push"))),
                job("lint", [step("lint", cond=only_push)]))
    assert machine_time(p, "push") == (15 + 30) + (15 + 30)
    assert machine_time(p, "pull_request") == 15
    assert expected_machine_time(p) == 0.5 * 90 + 0.5 * 15


def test_skipped_need_propagates_under_default_condition():
    a = job("a", [step("build")], cond=EQ(ctx("github.event_name"), lit("push")))
    b = job("b", [step("test")], cond=SUCCESS, needs=["a"])
    c = job("c", [step("noop")], cond=ALWAYS, needs=["a"])
    d = job("d", [step("noop")], cond=AND(ALWAYS, EQ(Expr("needs", ("a", "result")), lit("skipped"))),
            needs=["a"])
    p = program(a, b, c, d)
    runs = simulate(p, "pull_request")
    assert not runs["a"].ran and not runs["b"].ran
    assert runs["c"].ran and runs["d"].ran
    runs = simulate(p, "push")
    assert runs["a"].ran and runs["b"].ran and runs["c"].ran and not runs["d"].ran


def test_matrix_and_runner_os_conditions():
    only_linux = AND(SUCCESS, EQ(ctx("runner.os"), lit("Linux")))
    only_18 = AND(SUCCESS, EQ(ctx("matrix.node"), lit(18)))
    j1 = job("t[node=18,os=ubuntu-latest]", [step("lint", cond=only_linux), step("test", cond=only_18)],
             matrix=[("node", "18"), ("os", "ubuntu-latest")], runs_on="ubuntu-latest")
    j2 = job("t[node=20,os=windows-latest]", [step("lint", cond=only_linux), step("test", cond=only_18)],
             matrix=[("node", "20"), ("os", "windows-latest")], runs_on="windows-latest")
    p = program(j1, j2)
    runs = simulate(p, "push")
    assert runs[j1.id].steps == (True, True)
    assert runs[j2.id].steps == (False, False)
    assert machine_time(p, "push") == (15 + 30 + 120) + 15


def test_step_outcome_reference_and_cache_hit_cold():
    s1 = step("noop", step_id="gate", cond=AND(SUCCESS, EQ(ctx("github.event_name"), lit("push"))))
    s2 = step("build", cond=AND(SUCCESS, EQ(Expr("step", ("gate", "outcome")), lit("success"))))
    s3 = step("cache", step_id="cache", kind="uses", text="actions/cache@v4")
    s4 = step("install-pinned", cond=AND(SUCCESS, NE(Expr("step", ("cache", "outputs.cache-hit")), lit("true"))))
    p = program(job("j", [s1, s2, s3, s4]))
    assert simulate(p, "push")["j"].steps == (True, True, True, True)
    assert simulate(p, "pull_request")["j"].steps == (False, False, True, True)


def test_env_context_literal_only():
    cond = AND(SUCCESS, EQ(ctx("env.DEPLOY"), lit("yes")))
    p = program(job("j", [step("publish", cond=cond)], env=[("DEPLOY", "no")]))
    assert machine_time(p, "push") == 15
    p2 = program(job("j", [step("publish", cond=cond)], env=[("DEPLOY", "${{ vars.D }}")]))
    assert machine_time(p2, "push") == 45


def test_frozen_job_costs_overhead_plus_top_per_step():
    f = job("svc", [], frozen=True, raw={"runs-on": "ubuntu-latest", "services": {},
                                         "steps": [{"run": "a"}, {"run": "b"}, {"run": "c"}]})
    g = job("svc2", [step("test"), step("build")], frozen=True, raw={"runs-on": "x"})
    h = job("remote", [], frozen=True, raw={"uses": "o/r/.github/workflows/w.yml@v1"})
    p = program(f, g, h)
    assert simulate(p, "push")["svc"].seconds == 15 + 3 * 30
    assert simulate(p, "push")["svc2"].seconds == 15 + 2 * 30
    assert simulate(p, "push")["remote"].seconds == 15 + 30


def test_triggers_gate_events():
    j = job("j", [step("test")])
    assert machine_time(program(j, on={"push": {"branches": ["main"]}}), "pull_request") == 0
    assert machine_time(program(j, on="push"), "push") == 135
    assert machine_time(program(j, on=["pull_request"]), "push") == 0
    assert machine_time(program(j, on={}), "pull_request") == 135  # empty on: every event


def test_push_payload_facts():
    cond = AND(SUCCESS, NE(ctx("github.event.pull_request.head.repo.full_name"), lit("o/r")))
    p = program(job("j", [step("test", cond=cond)]))
    # push: pull_request payload is null, null != 'o/r' holds; pull_request: unknown -> true
    assert simulate(p, "push")["j"].steps == (True,)
    assert simulate(p, "pull_request")["j"].steps == (True,)
    cond2 = AND(SUCCESS, EQ(ctx("github.head_ref"), lit("")))
    p2 = program(job("j", [step("test", cond=cond2)]))
    assert simulate(p2, "push")["j"].steps == (True,)


def test_unknown_needs_and_cycles_raise():
    with pytest.raises(ValueError):
        simulate(program(job("a", [], needs=["zz"])), "push")
    with pytest.raises(ValueError):
        simulate(program(job("a", [], needs=["b"]), job("b", [], needs=["a"])), "push")


def test_matrix_needs_reference_by_original_job_key():
    b1 = job("b[n=1]", [step("build")], matrix=[("n", "1")], orig="b")
    b2 = job("b[n=2]", [step("build")], matrix=[("n", "2")], orig="b")
    c = job("c", [step("test")], cond=AND(ALWAYS, EQ(Expr("needs", ("b", "result")), lit("success"))),
            needs=[b1.id, b2.id])
    assert simulate(program(b1, b2, c), "push")["c"].ran


def test_deterministic_and_report_json_ready():
    import json
    p = program(job("a", [step("build")]), job("b", [step("test")], needs=["a"], cond=SUCCESS))
    r1 = json.dumps(cost.cost_report(p), sort_keys=True)
    r2 = json.dumps(cost.cost_report(p), sort_keys=True)
    assert r1 == r2
    assert cost.cost_report(p)["expected_seconds"] == (15 + 60) + (15 + 120)
    assert DEFAULT_CONTEXTS == (("push", 0.5), ("pull_request", 0.5))
    with pytest.raises(ValueError):
        expected_machine_time(p, (("push", -1.0),))


def test_needs_reference_by_reusable_group_key():
    b = job("call[os=x]/build", [step("build")], orig="call/build",
            cond=EQ(ctx("github.event_name"), lit("push")))
    t = job("call[os=x]/test", [step("test")], orig="call/test", needs=[b.id],
            cond=AND(ALWAYS, EQ(Expr("needs", ("call[os=x]/build", "result")), lit("skipped"))))
    p = program(b, t)
    assert not simulate(p, "push")["call[os=x]/test"].ran
    assert simulate(p, "pull_request")["call[os=x]/test"].ran


# ---------------------------------------------------------------------------
# Billed minutes
# ---------------------------------------------------------------------------

def test_billed_minutes_rounding():
    assert cost.billed_minutes_for(0.0, ran=False) == 0
    assert cost.billed_minutes_for(125.0, ran=False) == 0
    assert cost.billed_minutes_for(0.0) == 1          # an executed job bills at least a minute
    assert cost.billed_minutes_for(0.5) == 1
    assert cost.billed_minutes_for(60.0) == 1
    assert cost.billed_minutes_for(60.0000000001) == 1  # float noise never adds a minute
    assert cost.billed_minutes_for(60.5) == 2
    assert cost.billed_minutes_for(185.0) == 4


def test_billed_minutes_per_job_and_expected():
    only_push = EQ(ctx("github.event_name"), lit("push"))
    a = job("a", [step("checkout"), step("install-pinned"), step("test")])   # 185 s -> 4 min
    b = job("b", [step("lint")], cond=only_push)                             # 45 s -> 1 min
    p = program(a, b)
    runs = simulate(p, "push")
    assert runs["a"].billed_minutes == 4 and runs["b"].billed_minutes == 1
    assert simulate(p, "pull_request")["b"].billed_minutes == 0
    assert cost.billed_minutes(p, "push") == 5
    assert cost.billed_minutes(p, "pull_request") == 4
    assert cost.expected_billed_minutes(p) == 0.5 * 5 + 0.5 * 4
    # two short jobs bill more minutes than one merged job with the same machine seconds
    merged = program(job("m", [step("checkout"), step("lint")]))             # 50 s -> 1 min
    split = program(job("x", [step("checkout")]), job("y", [step("lint")]))  # 20 s + 45 s
    assert machine_time(split, "push") > machine_time(merged, "push")
    assert cost.billed_minutes(split, "push") == 2 and cost.billed_minutes(merged, "push") == 1


def test_cost_report_has_seconds_and_billed_minutes():
    p = program(job("a", [step("build"), step("test", cond=lit(False))]))
    r = cost.cost_report(p)
    assert r["model"] == "mock"
    assert r["expected_seconds"] == 75 and r["expected_billed_minutes"] == 2
    ev = r["events"]["push"]
    assert ev["seconds"] == 75 and ev["billed_minutes"] == 2
    assert ev["jobs"]["a"]["billed_minutes"] == 2
    assert ev["jobs"]["a"]["step_seconds"] == [60.0, 0.0]
    assert ev["jobs"]["a"]["step_historical"] == [False, False]
    assert ev["executed_steps"] == 1 and ev["executed_steps_historical"] == 0
    assert r["contexts"] == [["push", 0.5], ["pull_request", 0.5]]
    assert r["coverage"] is None
    assert cost.cost_report(p, (("push", 1.0),))["expected_billed_minutes"] == 2


# ---------------------------------------------------------------------------
# Historical backend
# ---------------------------------------------------------------------------

def gh_step(name, durations, positions, occurrence=0, skipped=0):
    import statistics
    return {"name": name, "occurrence": occurrence, "positions": positions, "durations": durations,
            "seconds": statistics.median(durations) if durations else None, "skipped": skipped}


def gh_job(steps, durations=(40.0,), overheads=(4.0, 6.0), counts=None):
    return {"samples": len(durations), "skipped": 0, "durations": list(durations),
            "overheads": list(overheads), "main_step_counts": counts or [len(steps)], "steps": steps}


def uses_step(ref, idx, template="checkout", origin_job="t", origin_instance=None):
    return Step(Binding("uses", ref), SUCCESS, Contract(template=template), (origin_job, idx),
                origin_instance=origin_instance)


def run_step(text, idx, template, name=None, origin_job="t", cond=SUCCESS, origin_instance=None):
    return Step(Binding("run", text, name=name), cond, Contract(template=template), (origin_job, idx),
                origin_instance=origin_instance)


def matrix_program():
    def inst(v):
        steps = [uses_step("actions/checkout@v4", 0),
                 run_step("npm ci\n", 1, "install-pinned"),
                 run_step("npm run build", 2, "build", name=f"Build {v}"),
                 run_step("npm test", 3, "test"),
                 run_step("./weird.sh", 4, "top")]
        return JobInstance(id=f"t[v={v}]", orig_job="t", check_name=f"t ({v})", matrix=(("v", v),),
                           runs_on="ubuntu-latest", cond=TRUE, needs=frozenset(), steps=tuple(steps))
    return program(inst("a"), inst("b"))


def matrix_timings():
    def steps(v, test):
        return [gh_step("Run actions/checkout@v4", [1.0, 3.0], [0]),
                gh_step("Run npm ci", [10.0, 20.0], [1]),
                gh_step(f"Build {v}", [30.0], [2]),
                gh_step("Run npm test", test, [3])]
    return {
        "benchmark": "acme/app@x:.github/workflows/ci.yml",
        "events": {"push": {"runs": [1], "window_count": 30}, "pull_request": {"runs": [2], "window_count": 10}},
        "jobs": {"t (a)": gh_job(steps("a", [50.0]), counts=[4]),
                 "t (b)": gh_job(steps("b", [70.0]), overheads=(10.0,), counts=[4]),
                 "stale job": gh_job([gh_step("Run x", [5.0], [0])])},
    }


def test_step_names_and_matching():
    assert cost.github_step_name(run_step("\n  npm ci  \nnpm test", 0, "x")) == "Run npm ci"
    assert cost.github_step_name(uses_step("actions/checkout@v4", 0)) == "Run actions/checkout@v4"
    assert cost.github_step_name(run_step("x", 0, "x", name="Build it")) == "Build it"
    assert cost.names_match("Test  ${{ matrix.os }}", "Test ubuntu-latest")
    assert cost.names_match("Run echo 18", "Run echo ${{ matrix.node }}")
    assert not cost.names_match("Run npm test", "Run npm test:ci")
    long = "Lint with build (4/4, parser, type-utils, rule-schema-to-typescript-types, typescript-eslint, repo)"
    assert cost.names_match(long, long[:90] + "...")                   # GitHub truncation
    assert not cost.names_match("Lint (1/4)", "Lint (1/4)...")         # too short to be truncated
    assert not cost.names_match(long.replace("4/4", "3/4"), long[:90] + "...")


def test_historical_model_matches_jobs_and_steps():
    p = matrix_program()
    m = cost.historical_model(p, matrix_timings())
    assert m.label == "historical"
    assert m.step_table[("t[v=a]", "t", 1)] == 15.0 and m.step_table[("t[v=b]", "t", 3)] == 70.0
    assert m.step_table[("t[v=a]", "t", 2)] == 30.0     # matched by the (interpolated) name
    assert ("t[v=a]", "t", 4) not in m.step_table      # no GitHub step -> mock fallback
    assert m.origin_table[("t", 3)] == 60.0             # pooled over instances
    assert m.overhead_table["ubuntu-latest"] == 6.0     # median of 4, 6, 10
    assert m.contexts == (("push", 0.75), ("pull_request", 0.25))
    cov = m.coverage
    assert cov["jobs_matched"] == 2 and cov["github_jobs_unmatched"] == ["stale job"]
    assert cov["steps_total"] == 10 and cov["steps_priced_historical"] == 8
    assert cov["matched_fraction"] == 0.8 and cov["event_weights_source"] == "run-counts"
    runs = simulate(p, "push", m)
    # overhead 6 + checkout 2 + ci 15 + build 30 + test 50 + unmatched weird.sh 30 (mock top)
    assert runs["t[v=a]"].seconds == 6 + 2 + 15 + 30 + 50 + 30
    assert runs["t[v=a]"].step_historical == (True, True, True, True, False)
    assert runs["t[v=a]"].billed_minutes == 3
    r = cost.cost_report(p, model=m)
    assert r["contexts"] == [["push", 0.75], ["pull_request", 0.25]]
    assert r["coverage"]["matched_fraction"] == 0.8
    assert r["events"]["push"]["executed_steps"] == 10
    assert r["events"]["push"]["executed_steps_historical"] == 8
    assert expected_machine_time(p, model=m) == machine_time(p, "push", m)  # both events equal here


def test_historical_index_fallback_when_names_differ():
    s = [run_step("make build", 0, "build", origin_job="ci"), run_step("make test", 1, "test", origin_job="ci")]
    j = JobInstance(id="ci", orig_job="ci", check_name="ci", matrix=(), runs_on="ubuntu-latest",
                    cond=TRUE, needs=frozenset(), steps=tuple(s))
    t = {"jobs": {"ci": gh_job([gh_step("Run make build", [12.0], [0]),
                                gh_step("Run make  test --ci", [34.0], [1])])}}
    m = cost.historical_model(program(j), t)
    assert m.step_table == {("ci", "ci", 0): 12.0, ("ci", "ci", 1): 34.0}
    assert m.coverage["steps_matched_by_name"] == 1 and m.coverage["steps_matched_by_index"] == 1
    # no index fallback when the fetched runs had a different number of steps
    t2 = {"jobs": {"ci": gh_job(t["jobs"]["ci"]["steps"], counts=[2, 3])}}
    assert cost.historical_model(program(j), t2).step_table == {("ci", "ci", 0): 12.0}


def test_historical_prices_candidate_programs_through_origins():
    p = matrix_program()
    m = cost.historical_model(p, matrix_timings())
    shared = JobInstance(
        id="ghav-merged", orig_job="ghav-merged", check_name="merged", matrix=(), runs_on="ubuntu-latest",
        cond=TRUE, needs=frozenset(),
        steps=(uses_step("actions/checkout@v4", 0, origin_instance="t[v=b]"),    # instance table
               run_step("npm test", 3, "test"),                                   # job-level table
               Step(Binding("uses", "actions/upload-artifact@v4"), SUCCESS, Contract(template="unknown"),
                    ("ghav-transfer", 0))))                                      # mock
    cand = program(shared)
    run = simulate(cand, "push", m)["ghav-merged"]
    assert run.step_seconds == (2.0, 60.0, 10.0)
    assert run.step_historical == (True, True, False)
    assert run.seconds == 6.0 + 72.0                    # overhead by runner label
    other = JobInstance(id="mac", orig_job="mac", check_name="mac", matrix=(), runs_on="macos-latest",
                        cond=TRUE, needs=frozenset(), steps=())
    assert m.overhead(other) == 6.0                     # pooled default overhead
    # a job that reuses an original instance id but holds a step of another original job
    reuse = JobInstance(id="t[v=a]", orig_job="t", check_name="t (a)", matrix=(), runs_on="ubuntu-latest",
                        cond=TRUE, needs=frozenset(), steps=(run_step("make x", 1, "build", origin_job="u"),))
    assert m.price_step(reuse, reuse.steps[0]) == (60.0, False)


def test_historical_frozen_job_uses_median_duration_and_weights_default():
    f = JobInstance(id="svc", orig_job="svc", check_name="svc", matrix=(), runs_on="ubuntu-latest",
                    cond=TRUE, needs=frozenset(), steps=(run_step("a", 0, "top", origin_job="svc"),
                                                         run_step("b", 1, "top", origin_job="svc")),
                    frozen=True)
    g = JobInstance(id="other", orig_job="other", check_name="other", matrix=(), runs_on="ubuntu-latest",
                    cond=TRUE, needs=frozenset(), steps=(run_step("c", 0, "top", origin_job="other"),),
                    frozen=True)
    t = {"jobs": {"svc": gh_job([], durations=(100.0, 120.0), overheads=(4.0,), counts=[5])}}
    m = cost.historical_model(program(f, g), t)
    runs = simulate(program(f, g), "push", m)
    assert runs["svc"].seconds == 110.0 and runs["svc"].step_historical == (True, True)
    assert runs["other"].seconds == 4.0 + 30.0          # unmatched frozen job: overhead + mock
    assert m.contexts == DEFAULT_CONTEXTS and m.coverage["event_weights_source"] == "default"
    assert m.coverage["steps_priced_historical"] == 2


def test_ambiguous_job_names_do_not_match():
    j = JobInstance(id="t", orig_job="t", check_name="Test ${{ github.event_name }}", matrix=(),
                    runs_on="ubuntu-latest", cond=TRUE, needs=frozenset(), steps=())
    t = {"jobs": {"Test push": gh_job([]), "Test pull_request": gh_job([])}}
    assert cost.match_jobs(program(j), t["jobs"]) == {}
    assert cost.match_jobs(program(j), {"Test push": gh_job([])}) == {"t": "Test push"}
    # two instances matching one GitHub job through wildcards: neither is matched
    k = JobInstance(id="u", orig_job="u", check_name="Test ${{ inputs.x }}", matrix=(),
                    runs_on="ubuntu-latest", cond=TRUE, needs=frozenset(), steps=())
    assert cost.match_jobs(program(j, k), {"Test push": gh_job([])}) == {}


def test_load_historical_model_without_file_is_mock_priced(tmp_path):
    p = program(job("a", [step("build")]))
    m = cost.load_historical_model(p, "nobody__nothing__ci", tmp_path)
    assert m.label == "historical" and m.coverage["matched_fraction"] == 0.0
    assert machine_time(p, "push", m) == machine_time(p, "push")


def test_github_job_names_flatten_structured_matrix_values():
    def mk(name, matrix=(("x", "1"),)):
        return JobInstance(id="j", orig_job="j", check_name=name, matrix=matrix, runs_on="ubuntu-latest",
                           cond=TRUE, needs=frozenset(), steps=())
    j = mk('Lint (2/4, , ["ast-spec","utils"])')
    assert cost.github_job_names(j) == [j.check_name, "Lint (2/4, ast-spec, utils)"]
    t = mk('Unit (ubuntu-latest, 18, {"name":"eslint-plugin","params":"--shard=1/4"})')
    assert cost.github_job_names(t)[1] == "Unit (ubuntu-latest, 18, eslint-plugin, --shard=1/4)"
    assert cost.github_job_names(mk("test (a, b)")) == ["test (a, b)"]
    assert cost.github_job_names(mk("plain", matrix=())) == ["plain"]
    gh = {"Lint (2/4, ast-spec, utils)": gh_job([]), "Lint (1/4, x)": gh_job([])}
    assert cost.match_jobs(program(j), gh) == {"j": "Lint (2/4, ast-spec, utils)"}
