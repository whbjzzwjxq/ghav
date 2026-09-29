"""Tests for ghav.checker: equivalence of small WIR programs built from dataclasses."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ghav.checker import check, check_groups
from ghav.wir import Expr

_spec = importlib.util.spec_from_file_location(
    "ghav_test_builders", Path(__file__).parent / "fixtures" / "checker" / "builders.py"
)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def lint_and_test():
    """Two independent check jobs that both build."""
    lint = B.job("lint", B.prefix("lint") + [B.run("npm run lint", B.LINT, ("lint", 4))])
    test = B.job("test", B.prefix("test") + [B.run("npm test", B.TEST, ("test", 4))])
    return B.program(lint, test)


def sharded_tests(ff: str | None = "test"):
    """A matrix job with two shards; only the test step reads the shard."""
    jobs = []
    for shard in ("1", "2"):
        steps = B.prefix("test") + [B.run("npm test", B.TEST_SHARD, ("test", 4))]
        jobs.append(B.job(f"test[shard={shard}]", steps, orig="test", check=f"test ({shard})", matrix=(("shard", shard),), ff=ff))
    return B.program(*jobs)


def shared_build_candidate(ff: str | None = "test"):
    producer = B.job("ghav-build", B.prefix("test") + B.pack_upload("ghav-0a1b", ("dist",), ("test", 3)),
                     orig="ghav-build")
    jobs = [producer]
    for shard in ("1", "2"):
        steps = [
            B.checkout(("test", 0)),
            B.setup(("test", 1)),
            B.install(("test", 2)),
            *B.download_unpack("ghav-0a1b", ("dist",), ("test", 3)),
            B.run("npm test", B.TEST_SHARD, ("test", 4)),
        ]
        jobs.append(
            B.job(
                f"test[shard={shard}]", steps, orig="test", check=f"test ({shard})",
                matrix=(("shard", shard),), ff=ff, needs=("ghav-build",),
            )
        )
    return B.program(*jobs)


# ---------------------------------------------------------------------------
# Required scenarios
# ---------------------------------------------------------------------------

def test_identical_programs_are_equivalent():
    p = lint_and_test()
    res = check(p, p)
    assert res.equivalent, res.counterexample
    assert res.reason == "equivalent"
    assert res.counterexample is None
    assert [g.to_json() for g in res.groups] == [
        {"old": ["lint"], "new": ["lint"]},
        {"old": ["test"], "new": ["test"]},
    ]


def test_merging_independent_check_jobs_is_equivalent_under_group_conjunction():
    original = lint_and_test()
    merged = B.job(
        "ci",
        B.prefix("lint") + [B.run("npm run lint", B.LINT, ("lint", 4)), B.run("npm test", B.TEST, ("test", 4))],
        orig="lint",
    )
    candidate = B.program(merged)
    res = check(original, candidate)
    assert res.equivalent, res.counterexample
    assert [g.to_json() for g in res.groups] == [{"old": ["lint", "test"], "new": ["ci"]}]


def test_sharing_a_build_through_a_ghav_artifact_is_equivalent():
    original = sharded_tests()
    candidate = shared_build_candidate()
    res = check(original, candidate)
    assert res.equivalent, res.counterexample
    # ghav- transfer artifacts are not persistent.
    assert all(not r.startswith("run:ghav-") for r in res.persistent)
    assert [g.to_json() for g in res.groups] == [
        {"old": ["test (1)", "test (2)"], "new": ["ghav-build", "test (1)", "test (2)"]}
    ]


def two_platforms():
    """Two check jobs on different platforms that both build (like two calls of test.yml)."""
    linux = B.job("linux", B.prefix("linux") + [B.run("npm test", B.TEST, ("linux", 4))])
    windows = B.job("windows", B.prefix("windows") + [B.run("npm test", B.TEST, ("windows", 4))],
                    runs_on="windows-latest")
    return B.program(linux, windows)


def platform_producers(consumers_need_all: bool):
    """One producer per platform; consumers download their platform's build.

    With ``consumers_need_all`` every consumer waits for every producer, as it would for a
    single producer matrix job (the first rejected candidate of the paper's overview).
    """
    jobs = []
    for plat, runs_on in (("linux", "ubuntu-latest"), ("windows", "windows-latest")):
        jobs.append(B.job(f"build-{plat}", B.prefix(plat) + B.pack_upload(f"ghav-{plat}", ("dist",), (plat, 3)),
                          runs_on=runs_on))
    for plat, runs_on in (("linux", "ubuntu-latest"), ("windows", "windows-latest")):
        needs = ("build-linux", "build-windows") if consumers_need_all else (f"build-{plat}",)
        steps = [B.checkout((plat, 0)), B.setup((plat, 1)), B.install((plat, 2)),
                 *B.download_unpack(f"ghav-{plat}", ("dist",), (plat, 3)), B.run("npm test", B.TEST, (plat, 4))]
        jobs.append(B.job(plat, steps, runs_on=runs_on, needs=needs))
    return B.program(*jobs)


def test_producer_shared_by_its_read_class_forms_one_group_and_is_equivalent():
    """A producer accounts for every displaced member of its read class (completion rule C4)."""
    original = lint_and_test()
    producer = B.job("build", B.prefix("lint") + B.pack_upload("ghav-dist", ("dist",), ("lint", 3)))
    lint = B.job(
        "lint",
        [B.checkout(("lint", 0)), B.setup(("lint", 1)), B.install(("lint", 2)),
         *B.download_unpack("ghav-dist", ("dist",), ("lint", 3)), B.run("npm run lint", B.LINT, ("lint", 4))],
        needs=("build",),
    )
    test = B.job(
        "test",
        [B.checkout(("test", 0)), B.setup(("test", 1)), B.install(("test", 2)),
         *B.download_unpack("ghav-dist", ("dist",), ("test", 3)), B.run("npm test", B.TEST, ("test", 4))],
        needs=("build",),
    )
    res = check(original, B.program(producer, lint, test))
    assert res.equivalent, res.counterexample
    assert [g.to_json() for g in res.groups] == [{"old": ["lint", "test"], "new": ["build", "lint", "test"]}]
    # Without read-class accounts the producer relates only to lint (its origins), and a
    # failing build would skip test in a group of its own.
    assert not check(original, B.program(producer, lint, test), read_classes=False).equivalent


def test_per_platform_producers_are_equivalent():
    res = check(two_platforms(), platform_producers(consumers_need_all=False))
    assert res.equivalent, res.counterexample
    assert [g.to_json() for g in res.groups] == [
        {"old": ["linux"], "new": ["build-linux", "linux"]},
        {"old": ["windows"], "new": ["build-windows", "windows"]},
    ]


def test_ts_jest_style_producer_skipping_other_group_is_not_equivalent():
    """A consumer that also waits for another group's producer is skipped when it fails.

    Skipped counts as pass, so the consumer's group passes although its own test failed.
    """
    res = check(two_platforms(), platform_producers(consumers_need_all=True))
    assert not res.equivalent
    assert res.reason == "counterexample"
    cex = res.counterexample
    assert cex["event"] in ("push", "pull_request")
    assert cex["group"] == {
        "old": ["linux"], "new": ["build-linux", "linux"], "original_verdict": False, "candidate_verdict": True,
    }
    assert "linux" in cex["candidate"]["skipped_jobs"]
    assert "build-windows" in cex["candidate"]["failed_jobs"]
    assert "linux" in cex["original"]["failed_jobs"]
    assert cex["candidate"]["failed_steps"]


def test_moving_a_step_to_a_job_with_a_different_ctx_value_is_not_equivalent():
    test = B.job("test[node=18]", [B.checkout(("test", 0)), B.run("npm test", B.TEST_NODE, ("test", 1))],
                 orig="test", check="test (18)", matrix=(("node", "18"),))
    lint = B.job("lint", [B.checkout(("lint", 0)), B.run("npm run lint", B.LINT, ("lint", 1))])
    original = B.program(test, lint)
    moved = B.job("lint", [B.checkout(("lint", 0)), B.run("npm run lint", B.LINT, ("lint", 1)),
                           B.run("npm test", B.TEST_NODE, ("test", 1))])
    res = check(original, B.program(moved))
    assert not res.equivalent
    assert res.counterexample["group"]["old"] == ["lint", "test (18)"]

    # Control: the same move into a job that has the same matrix value is equivalent.
    moved18 = B.job("lint", moved.steps, matrix=(("node", "18"),))
    assert check(original, B.program(moved18)).equivalent


def test_race_two_unordered_jobs_uploading_same_artifact_name():
    def uploader(jid):
        return B.job(jid, [B.checkout((jid, 0)), B.build((jid, 1)), B.upload("dist", (jid, 2))])

    original = B.program(uploader("a"))
    candidate = B.program(uploader("a"), uploader("b"))
    res = check(original, candidate)
    assert not res.equivalent
    assert res.reason == "ill-formed"
    blocking = [i for i in res.issues if i["blocking"]]
    assert [(i["kind"], i["region"], i["jobs"]) for i in blocking] == [("duplicate-artifact", "run:dist", ["a", "b"])]


# ---------------------------------------------------------------------------
# Further behavior
# ---------------------------------------------------------------------------

def test_dropping_an_event_condition_yields_event_counterexample():
    on_push = B.and_(B.SUCCESS, B.eq(B.ctx("github.event_name"), B.lit("push")))
    release = B.job("release", [B.checkout(("release", 0)), B.run("npm publish", B.PUBLISH, ("release", 1), cond=on_push)])
    unguarded = B.job("release", [B.checkout(("release", 0)), B.run("npm publish", B.PUBLISH, ("release", 1))])
    res = check(B.program(release), B.program(unguarded))
    assert not res.equivalent
    assert res.counterexample["event"] == "pull_request"
    assert res.counterexample["region"] == "ext:npm@release#1"
    assert res.counterexample["context"]["github.event_name"] == "pull_request"


def test_removing_a_cache_step_is_equivalent_under_cold_cache_semantics():
    miss = B.and_(B.SUCCESS, B.ne(Expr("step", ("cache", "outputs.cache-hit")), B.lit("true")))
    cached = B.job("test", [
        B.checkout(("test", 0)),
        B.uses("actions/cache@v4", B.CACHE, ("test", 1), with_=(("key", "npm-${{ hashFiles('package-lock.json') }}"), ("path", "~/.npm")), step_id="cache"),
        B.run("npm ci", B.INSTALL, ("test", 2), cond=miss),
        B.run("npm test", B.TEST, ("test", 3)),
    ])
    uncached = B.job("test", [B.checkout(("test", 0)), B.install(("test", 2)), B.run("npm test", B.TEST, ("test", 3))])
    assert check(B.program(cached), B.program(uncached)).equivalent


def _two_build_shards(pack_paths: tuple[str, ...] | None, unpack_paths: tuple[str, ...] | None = None):
    """Shards whose out region holds dist and lib; optionally share both builds."""
    builds = [B.run("npm run build:lib", B.BUILD_LIB, ("test", 4))]
    test_step = B.run("npm test", B.TEST_SHARD, ("test", 5))
    jobs = []
    if pack_paths is not None:
        jobs.append(B.job("ghav-build", B.prefix("test") + builds + B.pack_upload("ghav-x", pack_paths, ("test", 3))))
    for shard in ("1", "2"):
        if pack_paths is None:
            steps = B.prefix("test") + builds + [test_step]
            needs = ()
        else:
            steps = [B.checkout(("test", 0)), B.setup(("test", 1)), B.install(("test", 2)),
                     *B.download_unpack("ghav-x", unpack_paths or pack_paths, ("test", 3)), test_step]
            needs = ("ghav-build",)
        jobs.append(B.job(f"test[shard={shard}]", steps, orig="test", check=f"test ({shard})",
                          matrix=(("shard", shard),), needs=needs))
    return B.program(*jobs)


def test_share_that_uploads_only_part_of_out_is_not_equivalent():
    original = _two_build_shards(None)
    # out holds dist and lib, but only dist is packed: the transfer is not an exact copy.
    assert not check(original, _two_build_shards(("dist",))).equivalent
    # Packing both paths and unpacking them is exact.
    assert check(original, _two_build_shards(("dist", "lib"))).equivalent
    # Unpacking other paths than the packed ones is not.
    assert not check(original, _two_build_shards(("dist", "lib"), ("dist",))).equivalent


def test_fail_fast_member_writes_are_checked_as_refinement():
    def shards(extra_member: bool):
        jobs = []
        for shard in ("1", "2"):
            steps = B.prefix("test") + [B.run("npm test", B.TEST_SHARD, ("test", 4))]
            if shard == "1":
                on1 = B.and_(B.SUCCESS, B.eq(B.ctx("matrix.shard"), B.lit("1")))
                steps.append(B.run("npm publish", B.PUBLISH, ("test", 5), cond=on1))
            jobs.append(B.job(f"test[shard={shard}]", steps, orig="test", check=f"test ({shard})",
                              matrix=(("shard", shard),), ff="test"))
        lint = B.job("lint", [B.checkout(("lint", 0)), B.run("npm run lint", B.LINT, ("lint", 1))],
                     ff="test" if extra_member else None)
        return B.program(*jobs, lint)

    original = shards(False)
    assert check(original, original).equivalent
    res = check(original, shards(True))
    assert not res.equivalent
    assert "fail-fast-lower:ext:npm@test#5" in res.counterexample["differences"]
    assert "lint" in res.counterexample["candidate"]["failed_jobs"]


def test_fail_fast_members_share_one_group():
    groups = check_groups(sharded_tests(), sharded_tests())
    assert [g.to_json() for g in groups] == [{"old": ["test (1)", "test (2)"], "new": ["test (1)", "test (2)"]}]


def test_top_step_cannot_move_between_instances():
    a = B.job("a", [B.checkout(("a", 0)), B.run("./scripts/check.sh", B.TOP, ("a", 1))])
    b = B.job("b", [B.checkout(("b", 0)), B.run("npm run lint", B.LINT, ("b", 1))])
    original = B.program(a, b)
    assert check(original, original).equivalent
    merged = B.job("b", [B.checkout(("b", 0)), B.run("npm run lint", B.LINT, ("b", 1)), B.run("./scripts/check.sh", B.TOP, ("a", 1))])
    res = check(original, B.program(merged))
    assert not res.equivalent


def test_frozen_job_is_kept_and_compared():
    frozen = B.job("e2e", [B.checkout(("e2e", 0)), B.run("npm run e2e", B.TEST, ("e2e", 1))], frozen=True)
    original = B.program(frozen, *lint_and_test().jobs)
    assert check(original, original).equivalent
    without = B.program(*lint_and_test().jobs)
    res = check(original, without)
    assert not res.equivalent
    assert res.counterexample["group"]["old"] == ["e2e"]
    assert res.counterexample["group"]["new"] == []


def test_inherited_race_is_tolerated_but_recorded():
    def reporter(jid):
        return B.job(jid, [B.checkout((jid, 0)), B.run("npm test", B.TEST, (jid, 1)),
                           B.uses("codecov/codecov-action@v4", B.CODECOV, ("test", 2))], orig="test")

    p = B.program(reporter("t1"), reporter("t2"))
    res = check(p, p)
    assert res.equivalent
    assert [(i["program"], i["kind"], i["blocking"]) for i in res.issues] == [
        ("original", "race", False),
        ("candidate", "race", False),
    ]


def test_result_is_deterministic():
    original = two_platforms()
    candidate = platform_producers(consumers_need_all=True)
    runs = []
    for other in (None, sharded_tests(), shared_build_candidate()):
        if other is not None:  # unrelated solver work in between must not change the result
            check(other, other)
        runs.append(json.dumps(check(original, candidate).to_json(include_timing=False), sort_keys=True))
    assert runs[0] == runs[1] == runs[2]
    # The canonical counterexample prefers push and keeps steps succeeding in sorted order
    # as long as a difference remains. Every difference needs a failing test in one group
    # and a failing build in the other, so the first steps that must fail are linux's test
    # and windows' build (whose producer fails in the candidate).
    cex = json.loads(runs[0])["counterexample"]
    assert cex["event"] == "push"
    assert cex["original"]["failed_steps"] == ["linux#4 npm test", "windows#3 npm run build"]
    assert cex["candidate"]["failed_steps"] == ["build-windows#3 npm run build"]
    assert cex["candidate"]["skipped_jobs"] == ["linux", "windows"]


def test_conditions_from_expr_parser_when_available():
    expr = pytest.importorskip("ghav.expr")
    cond = expr.parse("success() && github.event_name == 'push'")
    release = B.job("release", [B.checkout(("release", 0)), B.run("npm publish", B.PUBLISH, ("release", 1), cond=cond)])
    manual = B.and_(B.SUCCESS, B.eq(B.ctx("github.event_name"), B.lit("push")))
    same = B.job("release", [B.checkout(("release", 0)), B.run("npm publish", B.PUBLISH, ("release", 1), cond=manual)])
    assert check(B.program(release), B.program(same)).equivalent


def test_resource_limit_gives_unknown_not_equivalent():
    res = check(lint_and_test(), shared_build_candidate(), rlimit=1)
    assert (res.equivalent, res.reason, res.counterexample) == (False, "unknown", None)


# ---------------------------------------------------------------------------
# Programs not produced by synthesis: origin recovery and rename invariance
# ---------------------------------------------------------------------------

def _renamed(job, new_id: str):
    """The job as a re-lowered emission would give it: new id and key, origins in the new key."""
    from dataclasses import replace

    steps = tuple(replace(s, origin=(new_id, i)) for i, s in enumerate(job.steps))
    return replace(job, id=new_id, orig_job=new_id, steps=steps)


def test_recover_origins_maps_renamed_jobs_by_check_name():
    from ghav.checker import recover_origins

    a = B.job("ci[os=a]/test", B.prefix("ci/test") + [B.run("./scripts/e2e.sh", B.TOP, ("ci/test", 4))],
              orig="ci/test", check="ci (a) / test")
    b = B.job("lint", [B.checkout(("lint", 0)), B.run("./scripts/check.sh", B.TOP, ("lint", 1))])
    original = B.program(a, b)
    candidate = B.program(_renamed(a, "ci-a-test"), b)
    recovered, n = recover_origins(original, candidate)
    assert n == 5
    assert [s.origin for s in recovered.job("ci-a-test").steps] == [s.origin for s in a.steps]
    res = check(original, candidate)
    # ⊤ steps read their instance identity (check name and matrix), which the rename keeps.
    assert res.equivalent, res.counterexample
    assert res.recovered_origins == 5
    # A race between the two ⊤ steps is inherited through the recovered origins.
    assert [(i["program"], i["blocking"]) for i in res.issues] == [("original", False), ("candidate", False)]


def test_recover_origins_keeps_synthesized_origins_and_marks_new_steps():
    from ghav.checker import recover_origins

    assert recover_origins(sharded_tests(), shared_build_candidate())[1] == 0
    original = lint_and_test()
    lint = original.job("lint")
    extra = B.job("lint", list(lint.steps) + [B.run("npm run typecheck", B.TYPECHECK, ("lint", 5))])
    recovered, n = recover_origins(original, B.program(extra, original.job("test")))
    assert n == 1
    assert recovered.job("lint").steps[-1].origin == ("lint", -2 - 5)
    assert not check(original, B.program(extra, original.job("test"))).equivalent


def test_renaming_a_top_job_to_another_check_name_is_not_equivalent():
    a = B.job("a", [B.checkout(("a", 0)), B.run("./scripts/check.sh", B.TOP, ("a", 1))])
    moved = _renamed(a, "b")
    from dataclasses import replace

    assert check(B.program(a), B.program(moved)).equivalent  # same check name "a"
    res = check(B.program(a), B.program(replace(moved, check_name="b")))
    assert not res.equivalent


def test_jobs_without_steps_relate_by_identity():
    """A call of a remote reusable workflow is frozen and has no steps: only its check name
    relates it to the original."""
    call = B.job("deploy", [], frozen=True)
    p = B.program(call, *lint_and_test().jobs)
    res = check(p, p)
    assert res.equivalent, res.counterexample
    assert {"old": ["deploy"], "new": ["deploy"]} in [g.to_json() for g in res.groups]
    renamed = B.program(_renamed(call, "deploy-renamed"), *lint_and_test().jobs)
    assert check(p, renamed).equivalent


def test_needs_references_in_step_text_follow_job_renaming():
    """``needs.<job>`` in a step's text names an instance; a rename that keeps the check name
    keeps the task (``semantics.task_key``)."""
    from dataclasses import replace

    out = Expr("step", ("v", "outputs.version"))
    build = B.job("build", [B.checkout(("build", 0)), B.run("./version.sh", B.contract("noop"), ("build", 1), step_id="v")],
                  outputs=(("version", out),))

    def publisher(target: str):
        text = f"npm publish --tag ${{{{ needs.{target}.outputs.version }}}}"
        c = B.contract("publish", ("src", "dep", "out", "ctx:github.ref", f"output:{target}:version"), ("ext:npm",))
        return B.job("release", [B.checkout(("release", 0)), B.run(text, c, ("release", 1))], needs=(target,))

    original = B.program(build, publisher("build"))
    renamed_build = replace(_renamed(build, "build-renamed"), outputs=build.outputs)
    candidate = B.program(renamed_build, publisher("build-renamed"))
    res = check(original, candidate)
    assert res.equivalent, res.counterexample


# ---------------------------------------------------------------------------
# Instance-level check groups (Step.origin_instance), cache findings
# ---------------------------------------------------------------------------

def _with_instances(p):
    """``p`` with every step's ``origin_instance`` set to its instance (as lowering does)."""
    return p.with_jobs([replace(j, steps=tuple(replace(s, origin_instance=j.id) for s in j.steps)) for j in p.jobs])


def test_origin_instance_gives_one_check_group_per_instance():
    p = _with_instances(sharded_tests(ff=None))
    assert [g.to_json() for g in check_groups(p, p)] == [
        {"old": ["test (1)"], "new": ["test (1)"]}, {"old": ["test (2)"], "new": ["test (2)"]}]
    # Without instance origins the relation is per original job (coarser).
    assert [g.to_json() for g in check_groups(sharded_tests(ff=None), sharded_tests(ff=None))] == [
        {"old": ["test (1)", "test (2)"], "new": ["test (1)", "test (2)"]}]
    # Fail-fast members still share one group.
    ff = _with_instances(sharded_tests())
    assert len(check_groups(ff, ff)) == 1


def test_instance_groups_localize_a_dropped_step_to_its_check():
    original = _with_instances(sharded_tests(ff=None))
    one, two = sorted(original.jobs, key=lambda j: j.id)
    cand = original.with_jobs([replace(one, steps=one.steps[:-1]), two])  # shard 1 drops its test
    res = check(original, cand)
    assert not res.equivalent
    assert res.counterexample["group"] == {
        "old": ["test (1)"], "new": ["test (1)"], "original_verdict": False, "candidate_verdict": True}
    coarse = check(sharded_tests(ff=None), sharded_tests(ff=None).with_jobs(
        [replace(j, steps=j.steps[:-1]) if j.id == one.id else j for j in sharded_tests(ff=None).jobs]))
    assert coarse.counterexample["group"]["old"] == ["test (1)", "test (2)"]


def test_recover_origins_sets_instance_origins_of_renamed_jobs():
    from ghav.checker import recover_origins

    original = _with_instances(sharded_tests(ff=None))
    renamed = original.with_jobs([replace(j, id=j.id.replace("test", "t2"),
                                          steps=tuple(replace(s, origin=("t2", s.origin[1]), origin_instance=None)
                                                      for s in j.steps)) for j in original.jobs])
    recovered, _ = recover_origins(original, renamed)
    for j in recovered.jobs:
        partner = next(o for o in original.jobs if o.check_name == j.check_name)
        assert {s.origin_instance for s in j.steps} == {partner.id}
    assert check(original, renamed).equivalent


def test_cache_opacity_is_reported_but_never_blocking():
    opaque = B.contract("setup-runtime+cache-opaque", ("src", "dep", "out", "tool"), ("dep", "tool"))
    j = B.job("a", [B.checkout(("a", 0)), B.uses("actions/setup-node@v4", opaque, ("a", 1)),
                    B.run("./x.sh", B.TOP, ("a", 2))])
    res = check(B.program(j), B.program(j))
    assert res.equivalent
    rows = [i for i in res.issues if i["kind"] == "cache-opaque"]
    assert [(r["program"], r["blocking"]) for r in rows] == [("original", False), ("candidate", False)]
