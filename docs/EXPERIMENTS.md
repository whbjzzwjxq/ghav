# Experiments

Everything here is deterministic given the committed `resources/` and `scope/` files.
Network access happens only in `make fetch-*` targets, and their results are committed.

## Benchmark scopes

The criteria come from the problem definition (redundant machine time), not from the method.
The development set is kept separate from the evaluation set so that the contract guide and the
synthesizer are not tuned on the evaluation workflows.

| Scope | Built by | Definition |
|---|---|---|
| `dev.scope` | committed | 30 workflows (S01–S10, M01–M10, H01–H10) inspected while designing GHAV, pinned to commits before 2026-09-01. Used only for development and the qualitative case studies. Labels in `resources/dev_labels.json`. |
| `pool.scope` | `scripts/scope/build_pool_scope.py` from `resources/pool.json` | GitHub search `language:TypeScript stars:>2500 fork:false archived:false`, sorted by stars descending, first 300 repositories, minus repositories in `dev.scope`. Each repository is pinned to the last default-branch commit before `2026-09-01T00:00:00Z`. |
| `workflows.scope` | `scripts/scope/build_workflows_scope.py` | Every `.github/workflows/*.yml|yaml` in a pooled repository whose `on:` includes `push` or `pull_request` and that lowers without error. |
| `eval.scope` | `scripts/scope/build_eval_scope.py` | Workflows in `workflows.scope` with at least 2 job instances, at least half of them not frozen, and redundancy ratio ≥ 0.2 under the mock cost model. If more than 100 qualify, a uniform sample of 100 with seed 42. |
| `random.scope` | `scripts/scope/build_random_scope.py` | Uniform sample of 30 workflows from `workflows.scope` with at least 2 job instances, seed 42, with no redundancy filter. Reported next to `eval.scope` so the effect of the filter is visible. |
| `devopt.scope` | `scripts/scope/build_devopt_scope.py` from `resources/devopt_prs.txt` | Developer-implemented optimizations: for each PR, the workflow files it changes, before (base commit) and after (merge commit). |

Redundancy ratio of a workflow: the mock-cost seconds of step occurrences whose binding and
read values (all tasks succeeding) equal an occurrence in an earlier job instance, divided by
the workflow's total mock machine time.

## Research questions

| RQ | Question | Scope | Output |
|---|---|---|---|
| RQ1 | How much machine time does GHAV save? | eval, random | `output/rq1_effectiveness.csv`, `tables/rq1.tex` |
| RQ2 | What does synthesis find beyond a rule-restricted search space? | eval | `output/rq2_rules.csv`, `tables/rq2.tex` |
| RQ3 | How much do effect pruning, cost bounds, and read classes contribute to search? | eval | `output/rq3_ablation.csv`, `tables/rq3.tex` |
| RQ4 | Does the checker validate or refute developer-written optimizations? | devopt | `output/rq4_devopt.csv`, `tables/rq4.tex` |
| RQ5 | Does the semantics agree with GitHub? | differential tests | deferred to v2 (needs real runs) |

An LLM baseline (same inputs, fixed prompt, checked with the same checker) is planned for v2.

## Stages per workflow

`ghav.pipeline.run_benchmark(line, variant, out_dir, cost_backend=...)` writes, into
`output/stages/<scope>/<benchmark-id>/`:

1. `lowered.json`: the lowered WIR (contracts not yet inferred);
2. `contracts.json`: template and effect of every step, ⊤ share, observed contracts and opaque
   caches (`contracts.report`);
3. `synth_<variant>.json` and `optimized_<variant>.yml`: the synthesis record (config, cost
   model, original and best cost in machine seconds and billed minutes, search counters,
   rejections by reason, best moves, the best candidate's check and round trip) and the emitted
   best program;
4. `check_<variant>.json`: the round trip, i.e. the optimized YAML lowered again and checked
   against the original;
5. `timing_<variant>.json`: wall-clock time, kept apart so every other file is byte-identical
   across runs.

Variants: `full`, `rules`, `no-effect-pruning`, `no-cost-bound`, `no-read-classes`.
Budget: `max_expansions = 2000` for every variant, with at most 256 children pushed per
expansion and 8 instance-as-producer encodings per class; supersets of refuted sketches are
skipped (docs/ARCHITECTURE.md, synth.py). These caps are deterministic and apply to every
benchmark.

Contracts: every evaluation run uses the static guide (`use_traces=False`). Observed contracts
from `resources/traces` are used only for dev-set case studies and are labeled as such.

Scope statistics (`ghav.pipeline.workflow_stats`, static guide, mock cost): `instances`,
`frozen` (= `frozen_instances`), `steps`, `top_steps`, `redundancy_ratio`,
`redundant_seconds`, `machine_seconds`.

## Reporting rules

- Every cost number states its backend: "mock cost" (docs/COST_MODEL.md table) or "historical
  cost (medians of the last 5 successful runs per event before 2026-09-01)" with the matched
  fraction (`CostModel.coverage`). Transfer steps inserted by synthesis have no history and are
  always priced by the mock table; say so where they matter.
- Report both metrics, expected machine seconds and expected billed minutes (each executed job
  rounded up to a whole minute), until the user fixes the primary one. Synthesis minimizes
  machine seconds; a candidate can save seconds and still add billed minutes.
- Report counts of workflows that are unsupported, frozen, or all-⊤, not only successes.
- Machine: record CPU model, OS, Python and z3 versions in `output/environment.json`.
