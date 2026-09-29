"""Tests for ghav.synth: read classes, merge, both share encodings, cross-group producers,
the rule-restricted variant, and determinism. Fixtures: tests/fixtures/synth (see README)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghav import checker, emit, synth
from ghav import cost as costmod
from ghav.lower import lower
from ghav.wir import OUT, Program

FIX = Path(__file__).parent / "fixtures" / "synth"
WF = Path(".github/workflows/ci.yml")
MOCK = costmod.MOCK
CONTEXTS = costmod.DEFAULT_CONTEXTS


def load(name: str) -> tuple[Program, Path]:
    """The fixture program with static-guide contracts (no observed traces)."""
    root = FIX / name
    return synth.annotate_like(lower(root / WF, root), root, {}), root


def analysis(p: Program, root: Path, **cfg) -> synth.Analysis:
    return synth.Analysis(p, synth.SynthConfig(**cfg), MOCK, CONTEXTS, root)


def class_with(an: synth.Analysis, text: str) -> synth.ReadClass:
    hits = [rc for rc in an.shareable.values() if an.step(rc.members[0]).binding.text == text]
    assert len(hits) == 1, [rc.label for rc in an.shareable.values()]
    return hits[0]


def run(p: Program, root: Path, variant: str = "full", **overrides) -> synth.SynthResult:
    return synth.synthesize(p, synth.SynthConfig.for_variant(variant, **overrides), repo_root=root, observed={})


# ---------------------------------------------------------------------------
# Helpers and read classes
# ---------------------------------------------------------------------------

def test_call_path_and_artifact_names():
    assert synth.call_path("test[os=a/b]") == ""
    assert synth.call_path("ci[node=18]/unit[os=x]") == "ci[node=18]"
    assert synth.call_path("a/b/c") == "a/b"
    name = synth.artifact_name("cabc", "test[mode=a]", (("mode", "a"), ("os", "ubuntu latest")))
    assert name.startswith("ghav-") and name.endswith("-a-ubuntu_latest")
    assert name == synth.artifact_name("cabc", "test[mode=a]", (("os", "ubuntu latest"), ("mode", "a")))
    assert name != synth.artifact_name("cabc", "job", (("mode", "a"), ("os", "ubuntu latest")))
    long = synth.artifact_name("c", "job", tuple((f"k{i}", "v" * 30) for i in range(10)))
    assert len(long) <= synth.MAX_ARTIFACT_NAME
    # Callee instances of a reusable workflow carry their caller's matrix values too.
    assert synth.instance_values("ci[node=18]/unit[os=x]", (("os", "x"),)) == ["18", "x"]
    callee = synth.artifact_name("c", "job", (("os", "x"),), "ci[node=18]/unit[os=x]")
    assert callee.endswith("-18-x")


def test_read_classes_follow_read_values():
    p, root = load("share")
    an = analysis(p, root)
    build = class_with(an, "npm run build")
    assert build.members == (("test[mode=a]", 3), ("test[mode=b]", 3))
    assert build.transferable and build.out_paths == ("dist",)
    lint = class_with(an, "npm run lint")
    assert not lint.transferable
    # The test steps differ by their MODE env, so they are singletons.
    assert an.token(("test[mode=a]", 5)) != an.token(("test[mode=b]", 5))
    # The build of mode=a is in the backward slice of its lint; the slice is closed.
    assert an.slice_of(("test[mode=a]", 4)) == (0, 1, 2, 3)


def test_read_classes_split_by_node_version():
    p, root = load("groups")
    an = analysis(p, root)
    for rc in an.shareable.values():
        nodes = {dict(an.by_id[i].matrix)["node"] for i, _ in rc.members}
        if an.step(rc.members[0]).binding.text != "actions/checkout@v4":
            assert len(nodes) == 1, rc  # one producer per read class: never across node groups
    builds = [rc for rc in an.shareable.values() if rc.label == "npm run build"]
    assert len(builds) == 2
    # Without read values the binding alone groups all four builds together.
    coarse = analysis(p, root, read_classes=False)
    assert [len(rc.members) for rc in coarse.shareable.values() if rc.label == "npm run build"] == [4]


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def test_merge_fuses_common_prefix_and_verifies():
    p, root = load("merge")
    res = run(p, root)
    assert res.original_cost == pytest.approx(220.0)
    assert res.best_cost == pytest.approx(145.0)  # one overhead and checkout/setup/install saved
    assert res.best_moves == [{"move": "merge", "instances": ["lint", "typecheck"], "host": "lint"}]
    assert res.accepted >= 1 and res.check["equivalent"] and res.roundtrip["equivalent"]
    (job,) = res.program.jobs
    assert [s.binding.text for s in job.steps] == [
        "actions/checkout@v4", "actions/setup-node@v4", "npm ci", "npm run lint", "npm run typecheck"]
    # Exact-form condition: typecheck runs whenever its own predecessors did not fail.
    last = emit.render_expr(job.steps[-1].cond)
    assert last.startswith("!cancelled()") and last.count("outcome != 'failure'") == 3
    assert "steps." in last and job.steps[3].cond == p.job("lint").steps[3].cond


def test_merge_constraints():
    p, root = load("share")
    an = analysis(p, root)
    # The two instances carry conflicting job env (TARGET): no completion exists.
    assert synth.merge_hard_ok(an, ("test[mode=a]", "test[mode=b]")) == "merge-env"
    with pytest.raises(synth.Incomplete) as exc:
        synth.complete(an, synth.Sketch(merges=(("test[mode=a]", "test[mode=b]"),)))
    assert exc.value.reason == "merge-env"


# ---------------------------------------------------------------------------
# Share encodings
# ---------------------------------------------------------------------------

def encodings(an: synth.Analysis, rc: synth.ReadClass) -> dict[str, synth.Completion]:
    return {enc: synth.complete(an, synth.Sketch().with_share(rc.id, enc))
            for enc in ("job", rc.members[0][0])}


def test_share_encodings_dedup_instance_wins_new_job_loses():
    """Sharing the lint of a two-instance matrix (no transfer): the instance-as-producer
    encoding is cheaper than the original, the new-job encoding is not; both verify."""
    p, root = load("share")
    an = analysis(p, root)
    orig = synth.program_cost(an, p)
    rc = class_with(an, "npm run lint")
    comps = encodings(an, rc)
    inst, job = comps["test[mode=a]"], comps["job"]
    lint = costmod.step_seconds(an.step(rc.members[0]))
    assert synth.program_cost(an, inst.program) == pytest.approx(orig - lint)
    assert synth.program_cost(an, job.program) > orig
    # The consumer drops its lint and gains no needs edge (nothing is transferred).
    b = inst.program.job("test[mode=b]")
    assert "npm run lint" not in [s.binding.text for s in b.steps] and not b.needs
    # The new producer runs the backward slice of the lint, and nothing else.
    prod = job.program.job(job.producers[rc.id])
    assert [s.binding.text for s in prod.steps] == [
        "actions/checkout@v4", "actions/setup-node@v4", "npm ci", "npm run build", "npm run lint"]
    assert prod.matrix == p.job("test[mode=a]").matrix and not prod.fail_fast_group
    for comp in (inst, job):
        assert synth.prune(an, comp) is None
        assert checker.check(p, comp.program).equivalent


def test_share_encodings_transfer_structure_and_cost():
    """Sharing the build (transferable out): the producer instance uploads right after its
    build, the consumer downloads in place of its build and needs the producer; the new-job
    encoding is more expensive than the original for two instances."""
    p, root = load("share")
    an = analysis(p, root)
    orig = synth.program_cost(an, p)
    rc = class_with(an, "npm run build")
    comps = encodings(an, rc)
    inst, job = comps["test[mode=a]"], comps["job"]
    art = inst.artifacts[rc.id]
    assert art.startswith("ghav-") and art.endswith("-a")
    a, b = inst.program.job("test[mode=a]"), inst.program.job("test[mode=b]")
    texts_a = [s.binding.text for s in a.steps]
    k = texts_a.index("npm run build")
    ups = [s for s in a.steps[k + 1:] if synth.is_synth_transfer(s)]
    assert ups and all(s.origin_instance == "test[mode=a]" for s in ups)
    assert any(any(r.kind == "run" and r.key == art for r in s.contract.writes) for s in ups)
    assert "npm run build" not in [s.binding.text for s in b.steps]
    downs = [s for s in b.steps if synth.is_synth_transfer(s)]
    assert downs and any(any(r.kind == OUT for r in s.contract.writes) for s in downs)
    assert b.needs == frozenset({"test[mode=a]"})
    assert emit.render_expr(b.cond) == "true"  # exact form: the producer's status is not required
    transfer = sum(costmod.step_seconds(s) for s in ups + downs)
    build = costmod.step_seconds(an.step(rc.members[0]))
    assert synth.program_cost(an, inst.program) == pytest.approx(orig - build + transfer)
    assert synth.program_cost(an, job.program) > orig
    assert synth.program_cost(an, job.program) > synth.program_cost(an, inst.program)
    # Both encodings are exact transfers, pass pruning, and verify end to end.
    for comp in (inst, job):
        art = comp.artifacts[rc.id]
        assert synth.sem.artifact_plan(comp.program)[art] == ("dist",)
        assert synth.prune(an, comp) is None
        assert checker.check(p, comp.program).equivalent
        rt, _, err = synth.roundtrip_check(p, comp.program, root, observed={})
        assert rt is not None and rt.equivalent, err


def test_status_dependency_pruning():
    """Dropping a step whose failure used to skip a later persistent write is pruned; the
    checker agrees (the write now happens when the dropped step would have failed)."""
    p, root = load("share")
    an = analysis(p, root)
    rc = class_with(an, "npm run lint")
    occ = rc.members[1]
    an.gates_persistent[occ] = True   # as if a later step of mode=b published something
    comp = synth.complete(an, synth.Sketch().with_share(rc.id, rc.members[0][0]))
    assert comp.dropped == ((occ, rc.members[0][0]),)
    assert synth.prune(an, comp) == "status-dependency"


def test_share_search_picks_instance_producer():
    p, root = load("share")
    res = run(p, root)
    assert res.improved and res.check["equivalent"] and res.roundtrip["equivalent"]
    shares = [m for m in res.best_moves if m["move"] == "share"]
    assert shares and all(m["producer"] != "job" for m in shares)
    assert "npm run lint" in {m["step"] for m in shares}
    assert res.best_cost <= res.original_cost - 30.0 + 1e-6
    # The emitted optimum lowers back to an equivalent program.
    rt, text, err = synth.roundtrip_check(p, res.program, root, observed={})
    assert rt is not None and rt.equivalent, err


# ---------------------------------------------------------------------------
# Job-local writes are updates
# ---------------------------------------------------------------------------

def test_setup_steps_that_write_dep_form_a_chain():
    """pnpm/action-setup and setup-node both write dep without reading it; the install reads
    what both left (semantics: job-local writes are updates). Dropping the shared-class
    pnpm/action-setup from one instance is pruned, and the checker agrees."""
    p, root = load("setup-chain")
    an = analysis(p, root)
    a, b = "test[mode=a]", "test[mode=b]"
    assert an.chain_before[(b, 3)]["dep"] == ((b, 0), (b, 1), (b, 2))
    assert an.chain_before[(b, 4)]["out"] == ((b, 0),)
    # The backward slice follows the whole chain, so a new producer job keeps both setups.
    assert an.slice_of((a, 4)) == (0, 1, 2, 3)
    setup = class_with(an, "pnpm/action-setup@v4")
    comp = synth.complete(an, synth.Sketch().with_share(setup.id, a))
    assert comp.dropped == (((b, 1), a),)
    assert synth.prune(an, comp) == "untransferred-read"
    assert not checker.check(p, comp.program).equivalent


def test_setup_chain_build_share_verifies_and_search_keeps_every_setup():
    p, root = load("setup-chain")
    an = analysis(p, root)
    build = class_with(an, "pnpm run build")
    # The consumer's out chain is the producer's (an exact transfer of dist).
    assert an.producer_out(build, "test[mode=a]") == (build.id,)
    comp = synth.complete(an, synth.Sketch().with_share(build.id, "test[mode=a]"))
    assert synth.prune(an, comp) is None and checker.check(p, comp.program).equivalent
    # The search (here a merge wins) never loses a setup step before an install.
    res = run(p, root)
    assert res.improved and res.check["equivalent"] and res.roundtrip["equivalent"]
    for job in res.program.jobs:
        texts = [s.binding.text for s in job.steps]
        for k, t in enumerate(texts):
            if t == "pnpm install":
                assert {"pnpm/action-setup@v4", "actions/setup-node@v4"} <= set(texts[:k])


# ---------------------------------------------------------------------------
# Cross-group producers
# ---------------------------------------------------------------------------

def test_cross_group_producer_is_rejected():
    """A producer shared across read classes (node 18 lint standing in for node 20) is
    refuted by the checker; synthesis never proposes it (classes never cross groups)."""
    p, root = load("groups")
    an = analysis(p, root)
    lints = sorted(o for rc in an.shareable.values() if rc.label == "npm run lint" for o in rc.members)
    assert len(lints) == 4
    fake = synth.ReadClass("cfake", tuple(lints), "lint", False, (), "npm run lint")
    an.shareable = {**an.shareable, "cfake": fake}
    producer = next(i for i, _ in lints if dict(an.by_id[i].matrix)["node"] == "18")
    comp = synth.complete(an, synth.Sketch().with_share("cfake", producer))
    res = checker.check(p, comp.program)
    assert not res.equivalent and res.reason == "counterexample"

    best = run(p, root)
    assert best.improved
    for m in best.best_moves:
        if m["move"] == "share":
            nodes = {dict(p.job(x.split("#")[0]).matrix)["node"] for x in m["members"]}
            assert len(nodes) == 1 or m["step"] == "actions/checkout@v4"


def test_no_read_classes_variant_is_refuted_on_spurious_classes():
    p, root = load("groups")
    res = run(p, root, "no-read-classes")
    assert res.rejected_checker >= 1          # the coarse classes are refuted, not accepted
    assert res.check["equivalent"] and res.best_cost <= res.original_cost


# ---------------------------------------------------------------------------
# Variants and determinism
# ---------------------------------------------------------------------------

def test_rules_variant_only_hoists_into_new_jobs():
    p, root = load("share")
    an = analysis(p, root, moves="rules")
    # Whole-prefix premise: every step before a shared step is shared too.
    assert sorted(rc.label for rc in an.shareable.values()) == sorted(
        ["actions/checkout@v4", "actions/setup-node@v4", "npm ci", "npm run build", "npm run lint"])
    res = run(p, root, "rules")
    assert all(m["producer"] == "job" for m in res.best_moves if m["move"] == "share")
    full = run(p, root, "full")
    assert full.best_cost <= res.best_cost + 1e-6


def test_no_effect_pruning_checks_more_but_finds_the_same():
    p, root = load("share")
    full = run(p, root)
    loose = run(p, root, "no-effect-pruning")
    assert loose.best_cost == pytest.approx(full.best_cost)
    assert loose.checked > full.checked and not loose.rejected_pruning
    assert sum(full.rejected_pruning.values()) > 0


@pytest.mark.parametrize("variant", sorted(synth.VARIANTS))
def test_variants_never_accept_unverified(variant):
    p, root = load("merge")
    res = run(p, root, variant)
    assert res.config.variant == variant
    assert res.check["equivalent"] and res.roundtrip["equivalent"]
    assert res.best_cost <= res.original_cost


def test_budget_counts_expansions():
    p, root = load("share")
    res = run(p, root, max_expansions=3)
    assert res.expansions <= 3 and res.budget_exhausted


def test_synthesis_is_deterministic():
    p, root = load("share")
    a = json.dumps(run(p, root).to_json(), sort_keys=True, indent=2)
    b = json.dumps(run(p, root).to_json(), sort_keys=True, indent=2)
    assert a == b
    assert "wall_ms" not in json.loads(a)
    q, _ = load("share")
    assert emit.emit(run(q, root).program) == emit.emit(run(p, root).program)


def test_redundancy_counts_later_duplicates():
    p, _ = load("share")
    redundant, total = synth.redundancy(p)
    # mode=b repeats checkout, setup, install, build and lint of mode=a.
    assert redundant == pytest.approx(5 + 10 + 45 + 60 + 30)
    assert total == pytest.approx(costmod.expected_machine_time(p))


def test_nogood_index_skips_supersets_only():
    idx = synth.NogoodIndex()
    bad_share = synth.Sketch().with_share("cA", "job")
    bad_merge = synth.Sketch().with_merge(("a",), ("b",))
    idx.add(bad_share)
    idx.add(bad_merge)
    assert idx.covers(bad_share.with_share("cB", "x"))
    assert idx.covers(synth.Sketch().with_merge(("a", "b"), ("c",)))
    assert not idx.covers(synth.Sketch().with_share("cA", "x"))
    assert not idx.covers(synth.Sketch().with_merge(("a",), ("c",)))
    assert idx.count == 2


def test_synthesize_requires_repo_root():
    p, _ = load("merge")
    with pytest.raises(ValueError):
        synth.synthesize(p, synth.SynthConfig())
