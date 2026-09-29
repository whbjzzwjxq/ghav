"""Tests for ghav.semantics: well-formedness, artifact layouts, symbolic evaluation."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import z3

from ghav import semantics as S
from ghav.wir import Expr

_spec = importlib.util.spec_from_file_location(
    "ghav_test_builders", Path(__file__).parent / "fixtures" / "checker" / "builders.py"
)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def valid(ctx: S.SymbolicContext, formula) -> bool:
    """``formula`` holds in every model of the background axioms."""
    s = z3.Solver(ctx=ctx.z3)
    s.add(*ctx.background())
    s.add(z3.Not(formula))
    return s.check() == z3.unsat


def satisfiable(ctx: S.SymbolicContext, formula) -> bool:
    s = z3.Solver(ctx=ctx.z3)
    s.add(*ctx.background())
    s.add(formula)
    return s.check() == z3.sat


def evaluate(p, **kw):
    ctx = S.SymbolicContext.for_programs(p)
    return ctx, S.evaluate(p, ctx, **kw)


def simple(jid, *, needs=(), cond=None, contract=None, orig=None):
    c = contract or B.TEST
    return B.job(jid, [B.checkout((orig or jid, 0)), B.run(f"run {jid}", c, (orig or jid, 1))],
                 needs=needs, cond=cond, orig=orig)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_topological_order_respects_needs_and_is_deterministic():
    p = B.program(simple("c", needs=("a", "b")), simple("b"), simple("a"), simple("d", needs=("c",)))
    assert S.topological_order(p) == ("a", "b", "c", "d")
    assert S.ancestors(p)["d"] == frozenset({"a", "b", "c"})


def test_structural_issues():
    dup = B.program(simple("a"), simple("a"))
    assert [i.kind for i in S.well_formed(dup)] == ["duplicate-id"]
    unknown = B.program(simple("a", needs=("zzz",)))
    assert [(i.kind, i.jobs) for i in S.well_formed(unknown)] == [("unknown-need", ("a", "zzz"))]
    cyc = B.program(simple("a", needs=("b",)), simple("b", needs=("a",)))
    assert [i.kind for i in S.well_formed(cyc)] == ["cycle"]


def test_external_writes_are_keyed_by_originating_occurrence():
    def reporter(jid, needs=(), origin=None):
        o = origin or jid
        return B.job(jid, [B.checkout((o, 0)), B.run("npm test", B.TEST, (o, 1)),
                           B.uses("codecov/codecov-action@v4", B.CODECOV, (o, 2))], needs=needs)

    # Two uploads to one service are two objects of P: no overwrite, no race.
    assert S.well_formed(B.program(reporter("a"), reporter("b"))) == []
    uni = S.Universe.of([B.program(reporter("a"), reporter("b"))])
    assert uni.ext_keys == ("codecov@a#2", "codecov@b#2")
    # The same occurrence run by two unordered jobs writes one object: a race.
    racy = S.well_formed(B.program(reporter("a"), reporter("b", origin="a")))
    assert [(i.kind, i.region, i.jobs, i.origins) for i in racy] == [
        ("race", "ext:codecov@a#2", ("a", "b"), (("a", 2),))
    ]
    assert S.well_formed(B.program(reporter("a"), reporter("b", needs=("a",), origin="a"))) == []
    # The key follows Step.origin_instance when it is set (lowering sets it).
    step = replace(reporter("x").steps[2], origin_instance="test[node=18]")
    assert S.occurrence(step, reporter("x")) == "test[node=18]#2"
    assert S.ext_write_key("codecov", step, reporter("x")) == "codecov@test[node=18]#2"
    assert S.ext_write_key("*", step, reporter("x")) == "*"


def test_top_step_races_with_external_writer_but_not_with_local_steps():
    top = B.job("t", [B.checkout(("t", 0)), B.run("./deploy.sh", B.TOP, ("t", 1))])
    local = simple("l")
    assert S.well_formed(B.program(top, local)) == []
    pub = B.job("p", [B.checkout(("p", 0)), B.run("npm publish", B.PUBLISH, ("p", 1))])
    issues = S.well_formed(B.program(top, pub))
    assert {(i.kind, i.region) for i in issues} == {("race", "ext:npm@p#1")}


def test_unordered_read_of_job_output_is_a_race():
    producer = B.job("a", [B.checkout(("a", 0))], outputs=(("v", Expr("lit", ("x",))),))
    reader_contract = B.contract("test", ("src", "output:a:v"), ())
    reader = B.job("b", [B.checkout(("b", 0)), B.run("use", reader_contract, ("b", 1))])
    assert [(i.kind, i.region) for i in S.well_formed(B.program(producer, reader))] == [("race", "output:a:v")]


def test_blocking_issues_ignore_inherited_races():
    racy = S.Issue("race", "m", "ext:codecov", ("a", "b"), (("test", 2),))
    new = S.Issue("race", "m", "ext:codecov", ("x", "y"), (("test", 2),))
    other = S.Issue("race", "m", "ext:npm", ("x", "y"), (("test", 3),))
    assert S.blocking_issues([racy], [new, other]) == [other]


# ---------------------------------------------------------------------------
# Artifact layouts and persistent regions
# ---------------------------------------------------------------------------

def _producer(jid, extra_steps, path, name="art"):
    return B.job(jid, [B.checkout((jid, 0))] + extra_steps + [B.upload(name, (jid, 9), path=path)])


def _packer(jid, extra_steps, paths, name="ghav-art", before=()):
    return B.job(jid, list(before) + [B.checkout((jid, 0))] + extra_steps + B.pack_upload(name, paths, (jid, 9)))


def test_artifact_plan_exact_copies_only_through_the_transfer_idiom():
    build = B.build(("a", 1))
    lib = B.run("npm run build:lib", B.BUILD_LIB, ("a", 2))
    assert S.artifact_plan(B.program(_packer("a", [build], ("dist",)))) == {"ghav-art": ("dist",)}
    assert S.artifact_plan(B.program(_packer("a", [build, lib], ("dist", "lib")))) == \
        {"ghav-art": ("dist", "lib")}
    # The pack must cover everything written into `out` since the last clean checkout.
    assert S.artifact_plan(B.program(_packer("a", [build, lib], ("dist",)))) == {"ghav-art": None}
    before = [B.run("npm run build:lib", B.BUILD_LIB, ("a", 8))]  # wiped by the checkout
    assert S.artifact_plan(B.program(_packer("a", [build], ("dist",), before=before))) == {"ghav-art": ("dist",)}
    top = B.run("./gen.sh", B.TOP, ("a", 2))
    assert S.artifact_plan(B.program(_packer("a", [build, top], ("dist",)))) == {"ghav-art": None}
    # A plain upload is never an exact copy (upload-artifact rebases paths, drops hidden
    # files, resets permissions): its artifact is uninterpreted.
    assert S.artifact_plan(B.program(_producer("a", [build], "dist"))) == {"art": None}
    # Two exact uploads of one name that carry different paths cannot be exact.
    both = B.program(_packer("a", [build], ("dist",)), _packer("b", [B.build(("b", 1))], ("dist", "lib")))
    assert S.artifact_plan(both) == {"ghav-art": None}
    # The upload must directly follow its pack.
    j = _packer("a", [build], ("dist",))
    gap = replace(j, steps=j.steps[:-1] + (B.run("npm test", B.TEST, ("a", 5)),) + j.steps[-1:])
    assert S.artifact_plan(B.program(gap)) == {"ghav-art": None}
    assert S.artifact_root(["dist/esm", "dist/cjs"]) == "dist"


def test_persistent_regions_exclude_transfers_and_include_top_catch_all():
    p = B.program(
        _producer("a", [B.build(("a", 1))], "dist", name="site"),
        _packer("b", [B.build(("b", 1))], ("dist",), name="ghav-123"),
        B.job("c", [B.run("./x.sh", B.TOP, ("c", 0))]),
        B.job("d", [B.checkout(("d", 0)), B.run("npm publish", B.PUBLISH, ("d", 1))]),
    )
    ctx = S.SymbolicContext.for_programs(p)
    assert S.persistent_regions(p, ctx.universe) == ("ext:*", "ext:npm@d#1", "run:site")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_skipped_job_passes_and_job_level_success_is_transitive():
    always = Expr("status", ("always",))
    p = B.program(simple("a"), simple("b", needs=("a",), cond=always), simple("c", needs=("b",)))
    ctx, obs = evaluate(p)
    a_fails = S.status_is(obs.job_status["a"], S.FAILURE)
    # b runs anyway; c's default success() also looks at a, so c is skipped and passes.
    assert valid(ctx, z3.Implies(a_fails, obs.job_ran["b"]))
    assert valid(ctx, z3.Implies(a_fails, S.status_is(obs.job_status["c"], S.SKIPPED)))
    assert valid(ctx, z3.Implies(a_fails, obs.checks["c"]))
    assert valid(ctx, z3.Implies(a_fails, z3.Not(obs.checks["a"])))


def test_context_values_shared_and_instance_specific():
    c = B.contract("test", ("ctx:github.sha", "ctx:matrix.node"), ())
    jobs = [B.job(f"t[node={n}]", [B.run("npm test", c, ("t", 0))], orig="t", matrix=(("node", n),)) for n in ("18", "20")]
    ctx, obs = evaluate(B.program(*jobs))
    reads18 = dict(obs.step_reads[("t[node=18]", 0)])
    reads20 = dict(obs.step_reads[("t[node=20]", 0)])
    sha = ("ctx", "", "github.sha")
    node = ("ctx", "", "matrix.node")
    assert reads18[sha].eq(reads20[sha])
    assert not reads18[node].eq(reads20[node])
    assert valid(ctx, reads18[node] != reads20[node])


def test_download_of_missing_artifact_fails_and_duplicate_upload_fails():
    lone = B.job("a", [B.download("nope", ("a", 0))])
    ctx, obs = evaluate(B.program(lone))
    assert valid(ctx, S.status_is(obs.step_status[("a", 0)], S.FAILURE))

    twice = B.job("a", [B.checkout(("a", 0)), B.build(("a", 1)), B.upload("x", ("a", 2)),
                        B.upload("x", ("a", 3), cond=Expr("status", ("always",)))])
    ctx, obs = evaluate(B.program(twice))
    first_ok = S.status_is(obs.step_status[("a", 2)], S.SUCCESS)
    assert valid(ctx, z3.Implies(first_ok, S.status_is(obs.step_status[("a", 3)], S.FAILURE)))


def test_exact_transfer_copies_out_region():
    prod = B.job("p", B.prefix("p") + B.pack_upload("ghav-1", ("dist",), ("p", 4)))
    cons = B.job("c", [B.checkout(("c", 0))] + B.download_unpack("ghav-1", ("dist",), ("c", 1))
                 + [B.run("npm test", B.TEST, ("c", 2))], needs=("p",))
    ctx, obs = evaluate(B.program(prod, cons), assume_success=True)
    out_after_build = dict(obs.step_reads[("p", 4)])[("out", "", "")]
    out_in_consumer = dict(obs.step_reads[("c", 3)])[("out", "", "")]
    assert out_after_build.eq(out_in_consumer)
    # The transfer steps change nothing else (their tarball is GHAV-private): a reader of the
    # consumer's tool state sees the same value as without the transfer.
    report = B.uses("codecov/codecov-action@v4", B.CODECOV, ("c", 5))
    with_t = B.job("c", list(cons.steps[:3]) + [report], needs=("p",))
    without = B.job("c", [B.checkout(("c", 0)), report], needs=("p",))
    _, o1 = evaluate(B.program(prod, with_t), assume_success=True)
    _, o2 = evaluate(B.program(prod, without), assume_success=True)
    assert dict(o1.step_reads[("c", 3)])[("tool", "", "")].sexpr() == \
        dict(o2.step_reads[("c", 1)])[("tool", "", "")].sexpr()


def test_transfer_into_a_dirty_out_region_or_with_plain_actions_is_uninterpreted():
    prod = B.job("p", B.prefix("p") + B.pack_upload("ghav-1", ("dist",), ("p", 4)))
    dirty = B.job("c", [B.checkout(("c", 0)), B.run("npm run build:lib", B.BUILD_LIB, ("c", 1))]
                  + B.download_unpack("ghav-1", ("dist",), ("c", 2)) + [B.run("npm test", B.TEST, ("c", 4))],
                  needs=("p",))
    ctx, obs = evaluate(B.program(prod, dirty), assume_success=True)
    out_after_build = dict(obs.step_reads[("p", 4)])[("out", "", "")]
    assert not dict(obs.step_reads[("c", 4)])[("out", "", "")].eq(out_after_build)
    plain_prod = B.job("p", B.prefix("p") + [B.upload("dist", ("p", 4))])
    plain = B.job("c", [B.checkout(("c", 0)), B.download("dist", ("c", 1)), B.run("npm test", B.TEST, ("c", 2))],
                  needs=("p",))
    ctx, obs = evaluate(B.program(plain_prod, plain), assume_success=True)
    out_after_build = dict(obs.step_reads[("p", 4)])[("out", "", "")]
    assert not dict(obs.step_reads[("c", 2)])[("out", "", "")].eq(out_after_build)


def test_read_classes_under_assume_success():
    jobs = []
    for shard in ("1", "2"):
        steps = B.prefix("t") + [B.run("npm test", B.TEST_SHARD, ("t", 4))]
        jobs.append(B.job(f"t[shard={shard}]", steps, orig="t", matrix=(("shard", shard),)))
    _, obs = evaluate(B.program(*jobs), assume_success=True)

    def key(jid, idx):
        return tuple((slot, term.sexpr()) for slot, term in obs.step_reads[(jid, idx)])

    assert key("t[shard=1]", 3) == key("t[shard=2]", 3)  # build reads identical values
    assert key("t[shard=1]", 4) != key("t[shard=2]", 4)  # test reads the shard


def test_job_local_writes_are_updates():
    """A task that writes a job-local kind without reading it changes part of the region:
    an earlier writer's contribution survives (pnpm/action-setup before setup-node both
    write ``dep``; the install reads what both left), so dropping the earlier writer changes
    the later reads. A task that reads the kind it writes gets no extra update term."""
    pnpm = B.uses("pnpm/action-setup@v4", B.SETUP, ("t", 1))
    both = B.job("t", [B.checkout(("t", 0)), pnpm, B.setup(("t", 2)), B.install(("t", 3))])
    only = B.job("t", [B.checkout(("t", 0)), B.setup(("t", 2)), B.install(("t", 3))])
    p = B.program(both)
    ctx = S.SymbolicContext.for_programs(p, B.program(only))
    a = S.evaluate(p, ctx, assume_success=True)
    b = S.evaluate(B.program(only), ctx, assume_success=True)
    dep = dict(a.step_reads[("t", 3)])[("dep", "", "")]
    dep_without = dict(b.step_reads[("t", 2)])[("dep", "", "")]
    assert satisfiable(ctx, dep != dep_without)
    assert "U:" in dep.sexpr()
    # The checker agrees: the setup step is not dead.
    from ghav.checker import check
    assert not check(p, B.program(only)).equivalent
    # checkout reads every kind it writes: its writes are plain functions of its reads.
    co = dict(a.step_reads[("t", 1)])[("src", "", "")]
    assert "U:" not in co.sexpr()


def test_conditions_contains_fromjson_and_opaque_folding():
    event_ok = Expr("call", ("contains", Expr("call", ("fromJSON", B.lit('["push","pull_request"]'))), B.ctx("github.event_name")))
    node_ge = Expr("opaque", ("matrix.node >= 18", "matrix.node"))
    run_ge = Expr("opaque", ("github.run_number >= 18", "github.run_number"))
    j = B.job("t[node=20]", [
        B.run("a", B.LINT, ("t", 0), cond=event_ok),
        B.run("b", B.LINT, ("t", 1), cond=node_ge),
        B.run("c", B.LINT, ("t", 2), cond=run_ge),
    ], orig="t", matrix=(("node", "20"),))
    ctx, obs = evaluate(B.program(j))

    def ran(i):
        return z3.Not(S.status_is(obs.step_status[("t[node=20]", i)], S.SKIPPED))

    assert valid(ctx, ran(0))  # event_name is push or pull_request
    assert valid(ctx, ran(1))  # folded concretely: 20 >= 18
    assert satisfiable(ctx, ran(2)) and satisfiable(ctx, z3.Not(ran(2)))  # symbolic reference


def test_cache_outputs_are_empty_under_cold_cache():
    miss = B.ne(Expr("step", ("cache", "outputs.cache-hit")), B.lit("true"))
    hit = B.eq(Expr("step", ("cache", "outputs.cache-hit")), B.lit("true"))
    j = B.job("a", [
        B.uses("actions/cache@v4", B.CACHE, ("a", 0), with_=(("key", "k"), ("path", "~/.npm")), step_id="cache"),
        B.run("npm ci", B.INSTALL, ("a", 1), cond=miss),
        B.run("echo hit", B.contract("noop"), ("a", 2), cond=hit),
    ])
    ctx, obs = evaluate(B.program(j))
    assert valid(ctx, S.status_is(obs.step_status[("a", 0)], S.SUCCESS))
    assert valid(ctx, z3.Not(S.status_is(obs.step_status[("a", 1)], S.SKIPPED)))
    assert valid(ctx, S.status_is(obs.step_status[("a", 2)], S.SKIPPED))


def test_event_condition_and_needs_outputs():
    on_push = B.eq(B.ctx("github.event_name"), B.lit("push"))
    producer = B.job("a", [B.checkout(("a", 0)), B.run("detect", B.LINT, ("a", 1), step_id="d")],
                     outputs=(("changed", Expr("step", ("d", "outputs.changed"))),))
    gate = B.ne(Expr("needs", ("a", "outputs.changed")), B.lit(""))
    consumer = B.job("b", [B.run("npm publish", B.PUBLISH, ("b", 0), cond=on_push)], needs=("a",),
                     cond=B.and_(B.SUCCESS, gate))
    ctx, obs = evaluate(B.program(producer, consumer))
    npm = obs.ext["npm@b#0"]
    init = ctx.symbol("init:ext:npm@b#0")
    assert valid(ctx, z3.Implies(ctx.event_name == ctx.lit("pull_request"), npm == init))
    assert satisfiable(ctx, z3.And(ctx.event_name == ctx.lit("push"), npm != init))
    assert [w.region for w in obs.writes["ext:npm@b#0"]] == ["ext:npm@b#0"]
    assert ("a", "changed") in obs.outputs


def test_evaluation_is_deterministic():
    def run_once():
        p = B.program(simple("a"), simple("b", needs=("a",)), B.job("c", [B.run("./x.sh", B.TOP, ("c", 0))]))
        _, obs = evaluate(p)
        return [obs.job_status[j].sexpr() for j in sorted(obs.job_status)] + [obs.ext[k].sexpr() for k in sorted(obs.ext)]

    assert run_once() == run_once()


def test_frozen_job_is_one_opaque_task_with_the_union_of_its_step_contracts():
    frozen = B.job("e2e", [B.checkout(("e2e", 0)), B.upload("report", ("e2e", 1), path="report")], frozen=True)
    p = B.program(frozen, B.job("d", [B.checkout(("d", 0)), B.run("npm publish", B.PUBLISH, ("d", 1))]))
    uni = S.Universe.of([p])
    assert not S.blanket_frozen(frozen)  # `services`: the steps still describe the effects
    eff = S.frozen_effect(frozen, p, uni)
    assert {str(r) for r in eff.writes} == {"run:report"}
    assert all(r.kind not in ("src", "dep", "out", "tool") for r in eff.reads)  # job-local: internal
    assert S.well_formed(p) == []  # no blanket write of ext: no race with the publisher
    ctx, obs = evaluate(p)
    assert satisfiable(ctx, S.status_is(obs.job_status["e2e"], S.FAILURE))
    assert "run:report" in obs.writes


def test_blanket_frozen_job_may_write_every_external_object_read_artifact_and_output():
    remote = B.job("remote", [], reasons=("remote-reusable-workflow",))
    top = B.job("t", [B.run("./x.sh", B.TOP, ("t", 0))])
    reader = B.job("r", [B.download("dist", ("r", 0)), B.run("npm test", B.TEST, ("r", 1))], needs=("remote",),
                   cond=B.and_(B.SUCCESS, B.ne(Expr("needs", ("remote", "outputs.v")), B.lit(""))))
    p = B.program(remote, top, reader)
    uni = S.Universe.of([p])
    assert S.blanket_frozen(remote) and uni.read_runs == ("dist",)
    eff = S.frozen_effect(remote, p, uni)
    assert {str(r) for r in eff.writes} == {"ext:*", "run:dist"}
    ctx, obs = evaluate(p)
    # The download may succeed (the remote workflow may upload `dist`) and the undeclared
    # output is unknown, so the reader's test may run.
    assert satisfiable(ctx, z3.And(obs.job_ran["r"],
                                   z3.Not(S.status_is(obs.step_status[("r", 1)], S.SKIPPED))))


def test_needs_of_a_reusable_call_and_matrix_outputs():
    lit = B.lit
    a = B.job("c/a", [B.run("a", B.LINT, ("c/a", 0))], orig="c/a", outputs=(("v", lit("1")),))
    b = B.job("c/b", [B.run("b", B.LINT, ("c/b", 0))], orig="c/b", outputs=(("v", lit("1")),))
    reader = B.job("r", [B.run("r", B.LINT, ("r", 0))], needs=("c/a", "c/b"),
                   cond=B.and_(B.SUCCESS, B.eq(Expr("needs", ("c", "outputs.v")), lit("1"))))
    p = B.program(a, b, reader)
    by_id = {j.id: j for j in p.jobs}
    assert S.needs_targets(reader, "c", by_id) == ["c/a", "c/b"]
    assert S.is_call_target("c", ["c/a", "c/b"], by_id)
    ctx, obs = evaluate(p)
    # A call's outputs are not modeled: uninterpreted, so the reader may or may not run.
    assert satisfiable(ctx, obs.job_ran["r"]) and satisfiable(ctx, z3.Not(obs.job_ran["r"]))
    # Matrix instances that produce the same term: the reference is that term.
    m = [B.job(f"m[n={n}]", [B.run("m", B.LINT, ("m", 0))], orig="m", matrix=(("n", n),),
               outputs=(("v", lit("1")),)) for n in ("1", "2")]
    reader = B.job("r", [B.run("r", B.LINT, ("r", 0))], needs=tuple(j.id for j in m),
                   cond=B.and_(B.SUCCESS, B.eq(Expr("needs", ("m", "outputs.v")), lit("1"))))
    ctx, obs = evaluate(B.program(*m, reader), assume_success=True)
    assert valid(ctx, obs.job_ran["r"])


def test_cache_issues_name_the_opaque_restores():
    opaque = B.contract("cache-opaque", ("src",), ("dep",))
    j = B.job("a", [B.uses("actions/cache@v4", opaque, ("a", 0)), B.run("x", B.TOP, ("a", 1))])
    assert [(i.kind, i.jobs, i.origins) for i in S.cache_issues(B.program(j))] == [
        ("cache-opaque", ("a",), (("a", 0),))]
