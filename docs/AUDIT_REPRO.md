# Audit: reproducibility and overfitting (2026-09-28)

Scope of this audit: the experiment code and outputs as of the source digest
`d9bc3233…6396f9b` (sha256 over `src/ghav/*.py`, identical to `output/environment.json`).
The audit was read-only. Every check below was re-run offline into a scratch directory;
nothing under `scope/`, `resources/`, `output/` or the paper repository was modified.

Severity: **high** = a published number, the paper build, or a provenance claim is wrong or
unprovable; **medium** = results are valid today but the documented procedure differs from what
ran, or a silent failure mode exists; **low** = hygiene.

## 1. Summary

| Check | Result |
|---|---|
| Repository/workflow/job-name special cases in `src/`, `scripts/`, `docs/`, `Makefile` | none found (section 2) |
| Contract guide rules are general | yes, but code implements rules the guide does not state, and one deviates from it (M5) |
| Scopes derived only from committed inputs by scripts | yes; all five builders reproduce byte-identical scope files offline; but nothing is committed (H1) |
| Offline re-run reproduces outputs | yes for every file checked (section 3) |
| Seeds and budgets fixed | seed 42 everywhere; budget is fixed at 250, but the docs say 2000 (M1) |
| `dev.scope` does not leak into eval/random | no repository overlap; `devopt.scope` (RQ4) does reuse two dev-set workflows (M3) |
| Mock cost table unchanged after results | code equals `docs/COST_MODEL.md`; every stage output is newer than `cost.py`; not provable without VCS history (H1) |

## 2. Special-case search (no violations)

- Every owner and repository name, workflow file stem, and job key/check name occurring in
  `dev`, `eval`, `random` and `devopt` (124 names, 53 stems, 680 job names, taken from the
  `lowered.json` files) was grepped as a word and as a quoted literal in `src/`, `scripts/`,
  `docs/` and `Makefile`.
- Hits in `src/` are tool and template names only (`jest`, `vite`, `playwright`, `vue-tsc`,
  `babel`, `build`, `lint`, `check`, ...) in the generic command and template tables
  (`src/ghav/contracts.py:186-337`, `src/ghav/cost.py:79-104`).
- Hits in `scripts/` are fixtures in `scripts/ablation/test_experiment_scripts.py:35-36,172`
  (GrapesJS, quill), used as test inputs only.
- Hits in `docs/` are the Zustand trace-format example and a remark
  (`docs/CONTRACT_GUIDE.md:175,204`). Neither is a rule.
- No string comparison in `lower`, `semantics`, `checker`, `synth`, `cost`, `emit`, `pipeline`
  or the experiment scripts tests a benchmark, repository, workflow or job name.
  The comparisons on `orig_job` and `check_name` are structural: same job, same check.
- `check_devopt.py` classifies pairs by verdict and issue kind only. It never looks at pair ids.
- The only trace file (`resources/traces/pmndrs__zustand/…`) is from the dev set.
  `run_stages.py` calls `run_benchmark(..., use_traces=False)`, which gives `observed={}` (`src/ghav/pipeline.py:253-254`), and every
  `contracts.json` under `output/stages` reports `"observed_steps": 0`.

## 3. Offline re-run (what was reproduced)

All commands ran without network access to GitHub. Outputs went to a scratch directory and
were then compared byte for byte with the committed files. Only `timing*.json` was excluded.

| Artifact | Command | Result |
|---|---|---|
| `scope/pool.scope`, `output/scope/pool.summary.json` | `build_pool_scope.py` | identical |
| `scope/workflows.scope`, `workflows.summary.json` | `build_workflows_scope.py --check-lowering` | identical (1805 workflows) |
| `scope/eval.scope`, `eval.summary.json` | `build_eval_scope.py` | identical (20 of 1805 qualify) |
| `scope/random.scope`, `random.summary.json` | `build_random_scope.py` | identical (30 of 749) |
| `scope/devopt.scope`, `devopt.pairs.json` | `build_devopt_scope.py` | identical (13 pairs) |
| `resources/timings/*/medians.json` (83 dirs) | `fetch_timings.py --offline --coverage` on a copy | identical |
| `output/audit/consistency.{eval,random,dev,devopt,workflows}.json` | `audit_consistency.py` | identical (32/1805 inconsistent) |
| `output/stages/{eval,random,devopt}.*`, `devopt.pairs` | `run_stages.py`, `check_devopt.py`, 5 workers (original runs: 6 to 8); eval with full, rules, no-cost-bound and no-read-classes under both backends; random full under both; devopt full (mock) and the 13 RQ4 pairs | 1280/1282 files identical; the 2 that differ are the eval `manifest.json` files, which lack the `no-effect-pruning` entry this re-run did not produce |
| `output/rq{1..4}*.csv`, `summary.json`, `tables/rq{1..4}.tex` | `aggregate.py` over the re-run (committed no-effect-pruning files added) | identical (`wall_s` masked as in `compare_outputs.py`) |
| no-effect-pruning on TriliumNext dev and GrapesJS (mock) | `run_stages.py --only …` | identical, including the unsound 5820 → 4100 s candidate |
| full, rules, no-cost-bound on eval (mock) at budget 2000 | `run_stages.py --max-expansions 2000` | 60/60 `synth_*.json` equal to the 250-budget files except `config.max_expansions`; every other file identical |
| test suites | `pytest -q`; `pytest -q scripts/ablation scripts/aggregate` | 473 passed; 10 passed |
| headline numbers of the experiment report | `output/summary.json` | match (RQ1 totals, frozen 6/254 and 150/335, ⊤ 40.5%/60.5%, pruning 897/724/54, RQ3 1115/1112 and 1149/1146, RQ4 2 counterexample + 11 unsupported) |

No-effect-pruning was re-run only on the two workflows above. The full run takes about
98 minutes of CPU per backend, most of it on jest. The other 18 × 2 no-effect-pruning results
are checked only by the original author's two-run comparison.

## 4. Findings

### High

**H1. Nothing is under version control, so "fixed before any experiment" cannot be shown.**
- `git status` in `/Users/hongbowen/code-whbjzzwjxq/ghav` reports `No commits yet`.
  `src/`, `scripts/`, `docs/`, `resources/`, `scope/`, `output/` and `Makefile` are all untracked.
- `docs/ARCHITECTURE.md:20-21` says "Cached files are committed so the experiments reproduce
  offline". `docs/EXPERIMENTS.md:4` says the fetch results "are committed". Both are false
  today.
- The two central non-tuning claims rest only on file modification times:
  - the mock table was fixed before results (`docs/COST_MODEL.md` "Fixed before any
    experiment and not tuned"; `src/ghav/cost.py:74`);
  - the contract guide was not tuned on eval (`docs/CONTRACT_GUIDE.md:4-6`).
- What the modification times show (all 2026-09-28):
  - `cost.py` 03:33, `COST_MODEL.md` 03:47, `contracts.py` and `CONTRACT_GUIDE.md` 03:53.
  - `eval.scope` 04:21. Eval and random timings 04:47.
  - First surviving stage output 05:42. `Makefile` (budget 250) 05:42:07.
  - The Makefile comment (`Makefile:67-73`) cites per-expansion times, so earlier runs
    existed and were overwritten. What code they ran is not recoverable.
- The mock table in code equals the doc table row for row, including job overhead 15 s. No
  other copy of the table exists in either repository to compare against.
- Fix: the user makes an initial commit of the ghav repository (inputs, scopes, docs, code;
  `output/stages/` is gitignored) and tags it before any further experiment. The paper
  should cite that commit.

**H2. Exported tables break the paper build.**
- `main.tex` wraps each generated file in its own float:
  - `\begin{table}` + `\caption` + `\label{tab:rqN}` + `\input{tables/rqN.tex}`;
  - RQ1 at `../ghav-pldi2027/main.tex:1135-1139`, RQ2 at `:1149-1153`, RQ3 at `:1162-1166`,
    RQ4 at `:1177-1181`.
- `scripts/aggregate/aggregate.py` emits complete `table*` floats that carry their own caption
  and label (`tables/rq1.tex:2-7`).
- Now that `tables/` exists, `pdflatex main.tex` stops with
  `LaTeX Error: Not in outer par mode` at `tables/rq1.tex` l.2. The labels are also defined twice.
- The report's "they compile in acmsmall with no overfull boxes" holds only for the tables
  compiled on their own.
- Other mismatches:
  - The paper's captions still say "(mock cost)", but the tables report both backends.
  - `tables/scopes.tex` (`main.tex:1080`) is not produced by any script.
- Fix, one of two:
  - the paper uses `\input{tables/rqN.tex}` bare;
  - or `aggregate.py` emits only the tabular.
- Also add a `scopes` table to `aggregate.py`.

**H3. The soundness bug in the experiment report is confirmed and reproduces
deterministically.** This is not a reproducibility defect, but it limits what the reported
RQ3 numbers mean.
- `src/ghav/semantics.py:1618-1622` returns the literal text of a job or workflow `env`
  value. For example, `AFFECTED: ${{ needs.changes.outputs.server }}` becomes that string.
  So `if: env.AFFECTED == 'true'` is constantly false.
- The re-run reproduces the "verified" TriliumNext dev candidate byte for byte (mock
  5820 → 4100 s, 101 → 71 billed minutes).
- The consistency audit flags 32/1805 workflows.
- Until this is fixed:
  - no verdict on a workflow flagged by `output/audit/consistency.*.json` should be reported
    as verified;
  - the RQ3 sentence "pruning discarded nothing the checker would have accepted" must not be
    stated.

### Medium

**M1. The budget in the docs is not the budget that ran.**
- The docs say 2000:
  - `docs/EXPERIMENTS.md:55` ("`max_expansions = 2000` for every variant");
  - `docs/ARCHITECTURE.md:224` (default 2000).
- Every result was produced with `MAX_EXPANSIONS ?= 250` (`Makefile:78`). This is recorded in
  each `synth_*.json` `config` and in the table captions.
- The 250 was chosen after the fact, as stated in the comment at `Makefile:67-73`: "the
  largest need is 244 expansions". That is jest `full`, 244 of 250, 2.4% headroom.
- It is non-binding for `full`, `rules`, `no-cost-bound` and `no-read-classes`:
  - `budget_exhausted` is 0 in every such row;
  - this audit re-ran the three cheap variants at 2000 and found them identical.
- It is binding for `no-effect-pruning` on 6/20 eval workflows (jest, repomix, html2canvas,
  three TriliumNext). RQ3's "pruning loses nothing" comparison therefore covers only a
  truncated search on those six.
- Any code change that makes a search longer can silently push `full` past 250. The
  aggregate already reports `budget_exhausted`, so this stays visible. RQ1/RQ2 text should
  check that it is 0.
- Fix: update `EXPERIMENTS.md` and `ARCHITECTURE.md`. State the reason for 250 and the
  truncation in RQ3.

**M2. Stage outputs carry no code provenance, so partial re-runs can silently mix code
versions.**
- `run_stages.py:108-113` hashes only `src/ghav/*.py`, not `scripts/`, `Makefile` or `uv.lock`.
- That hash goes into a single `output/environment.json`, which every invocation overwrites.
- Neither `synth_*.json` nor `manifest.json` records it.
- `run_stages.py:391-400` merges the previous manifest's `runs`.
- `aggregate.py:506-508` includes any variant file that exists (RQ3).
- Result: `make run-stages VARIANTS=full` after a code change leaves the other variants'
  files from the old code, and `make aggregate` mixes them without warning.
- Today the outputs are consistent: the digest equals the current source, and every stage
  file is newer than the last `src/` edit (04:02).
- Fix: write the source and script digests into each `synth_*.json`, `check_*.json` and
  `devopt_check.json`. Have `aggregate.py` refuse mixed digests.

**M3. RQ4 reuses dev-set workflows, and RQ4 overlaps random.**
- `devopt.scope` contains `kulshekhar/ts-jest .github/workflows/ci.yml` (D10) and
  `mswjs/msw .github/workflows/compat.yml` (D09, plus msw `ci.yml`). The first is dev
  workflow M03, the former motivating example. The second is dev M09. Both are at other
  commits: `resources/dev_labels.json`, `resources/devopt_prs.txt:9-10`.
- The PR list is hand-curated and has the same timestamp as `dev.scope`, so RQ4 is not a
  held-out set.
- `strapi/strapi .github/workflows/tests.yml` is in both `random.scope:27` (2026 pin) and
  `devopt.scope:8` (D01, older commit).
- There is no dev overlap with `pool`, `workflows`, `eval` or `random`: 17 dev repositories
  are excluded case-insensitively by `build_pool_scope.py:25-42`.
- Fix: disclose in the paper that RQ4 is curated and includes two dev-set workflows. Report
  RQ4 with and without D09/D10.

**M4. Result-driven analysis rules are implemented in scripts but not documented in `docs/`.**
None branches on names, so none is overfitting in the grep sense. But each was introduced
after results were seen, and `docs/EXPERIMENTS.md` does not define it:
- RQ4 `original-duplicate-artifact` reclassification (`scripts/ablation/check_devopt.py:15-22,173-174`):
  - it turns the checker's `equivalent` verdict on D04 into `unsupported`;
  - it is conservative (it removes a success), but readers need the rule.
- Stage layout `output/stages/<scope>.<backend>/` versus the documented `<scope>/<id>`:
  `docs/EXPERIMENTS.md:40`, `docs/ARCHITECTURE.md:48,257`.
- The consistency audit (`audit_consistency.py`, `consistency.json`) and the redundancy
  categories (`redundancy.json`).
- A further risk: the experiment report proposes counting only transfer-carriable redundancy
  in the eval criterion. That change would be made after observing zero improvements.
  `docs/EXPERIMENTS.md:8` says the criteria come "from the problem definition … not from the
  method". Selecting for what transfers can carry would be selecting for the method. If it is
  adopted anyway:
  - keep the current `eval.scope` as the primary result;
  - report the new filter as a second, labeled scope.

**M5. Contract code goes beyond, and in one place against, `docs/CONTRACT_GUIDE.md`.**
- The guide calls itself "the single, repository-independent procedure" (`CONTRACT_GUIDE.md:3`).
- The code also applies rules that appear only in `IMPLEMENTATION_NOTES`
  (`src/ghav/contracts.py:571-631`: pipe splitting, shell constructs → ⊤, snapshot-update
  flags write `src`, lint `-o` writes `tool`, package-runner rules, install lifecycle scripts)
  and in the name lists `_TOOL_NAMES`/`_TOOL_SUFFIXES` (`contracts.py:558-561`).
- These rules are general and printed by `ghav contracts`, but they are not in the guide.
- One rule deviates from the guide in an unsound direction:
  - the guide's `build-orchestrated` row covers "`pnpm -r`/`--filter` running builds";
  - `contracts.py:289-290` classifies *every* recursive or filtered script run as
    `build-orchestrated`, which writes no `ext:*`;
  - probe: `pnpm -r run release`, where `release` is `npm publish`, gets `build-orchestrated`,
    with writes `out, src, tool` only;
  - the root `pnpm run release` correctly gets `publish` with `ext:npm-registry`.
- Other rows missing from the guide: `yarn publish`/`yarn npm publish` (`contracts.py:333`),
  `nx run-many` (`contracts.py:285`).
- Fix:
  - move the notes into the guide (v1.2);
  - make recursive or filtered runs of anything but a resolvable build script ⊤;
  - re-run everything.

### Low

- **L1.** `build-devopt-scope` depends on `fetch-devopt` (`Makefile:58`), so
  `make build-scopes` enters a network script. It makes no API call when every PR is cached,
  but it still rewrites `resources/devopt/prs.json`. Drop the prerequisite.
- **L2.** `make check-determinism` (`Makefile:119-127`):
  - it compares all of `output/`, including `output/audit/` and `output/scope/`, which it
    does not regenerate, so up to 9 of the reported 1412 "identical" files are unchanged copies;
  - it does not cover scope building or the audit (section 3 does).
- **L3.** The Python version is not pinned: `pyproject.toml` has `>=3.11,<3.14` and there is
  no `.python-version`. z3 5.1.0.0 and PyYAML 6.0.3 are pinned by `uv.lock`. Each `uv run`
  reinstalls the project, which needs `hatchling` from the uv cache when offline.
- **L4.** `make test` does not run the 10 experiment-script tests (`pyproject.toml:25`
  `testpaths = ["tests"]`); they need `make test-scripts`.
- **L5.** `random.scope` does not exclude eval workflows (`build_random_scope.py:27-31`). The
  report's "no overlap with eval" is true for this sample, but by chance, not by design.
- **L6.** `tables/rq3.tex` contains a wall-clock column. It is masked by `compare_outputs.py`,
  but the paper must state the machine next to it (`output/environment.json`).
- **L7.** `audit_consistency.py` reuses worker processes; `run_stages.py` uses
  `max_tasks_per_child=1`. Its outputs reproduced anyway, since the answers are sat/unsat with
  no solver limit.

## 5. Verified determinism facts

- Randomness: only `random.Random(42)` in `scripts/scope/build_{eval,random}_scope.py`
  (`scopelib.SEED = 42`). z3 seeds are set once at import (`src/ghav/semantics.py:111-112`),
  and `rlimit` defaults to 0.
- No wall-clock value reaches a result. `time.perf_counter` appears only in the reported
  `wall_ms`/`smt_ms` fields.
- All 635 cached timing runs were created before `2026-09-01T00:00:00Z` (latest
  `2026-08-31T23:58:54Z`).
