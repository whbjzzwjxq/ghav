# Correctness audit of accepted optimizations (2026-09-28)

Read-only audit of the GHAV outputs after the experiment run of 2026-09-28. Only this file was
written in the repository; every run below wrote to a scratch directory outside the repository.

## 1. What was audited and how

**Sample.** `output/` holds only **two** accepted optimizations: TriliumNext/Trilium `dev.yml`
under `no-effect-pruning`, one per cost backend. Every `full`, `rules`, `no-cost-bound` and
`no-read-classes` run in `output/` and the RQ4 checks accepted nothing. To reach ten cases, I ran
the same `pipeline.run_benchmark` code into a scratch directory on:

- `no-effect-pruning` (mock, budget 250) over `dev.scope` and `random.scope`: 2 more accepted
  candidates, langfuse `pipeline.yml` and fast-check `build-status.yml`.
  `strapi__strapi__tests` had not finished when this report was written; the other 28 random
  workflows accepted nothing.
- `full` over `dev.scope` (static guide): 0 accepted.

I added the four accepted optimizations of `tests/fixtures/synth` and two small probes that `full`
accepts. Zustand was run with its trace (`full`, both backends).

**Method.** For each case:
- I read the original and optimized YAML.
- I re-derived by hand whether the final persistent state (external writes, artifacts of the
  original) and the check verdicts stay the same for `push` and `pull_request`. The semantics is
  the one in RESEARCH_SCOPE (tasks are deterministic functions of their reads, infrastructure
  failures are excluded, and checks are compared as group conjunctions).
- I looked for:
  - reads missing from contracts;
  - transfers that miss files;
  - changed conditions, check names or context values;
  - fail-fast changes;
  - permission and env differences.

**Validation.** I ran `actionlint` 1.7.12 and `yamllint` from a temporary venv outside the
repositories over:
- all 273 `optimized_*.yml` files in `output/stages` (compared with their originals);
- every audited candidate.

I also diffed every identity emission (`optimized_full.yml` of the 113 stage directories) against
its original, structurally.

## 2. Verdicts

The costs are before → after, in expected mock seconds unless marked (h), meaning historical.

| # | Case | Source | Cost | Verdict |
|---|---|---|---|---|
| 1 | Trilium `dev`, no-effect-pruning, mock | output/ | 5820 → 4100 (101 → 71 billed min) | **UNSOUND** (B1, B2) |
| 2 | Trilium `dev`, no-effect-pruning, historical | output/ | (h) 1597 → 1473 (31 → 29 billed min) | **UNSOUND** (B1, B2) |
| 3 | langfuse `pipeline`, no-effect-pruning, mock | random scope, rerun | 8485 → 8410 | **UNSOUND** (B1) |
| 4 | fast-check `build-status`, no-effect-pruning, mock | dev scope, rerun | 7309 → 7308 | Sound (trivial) |
| 5 | Probe A: matrix with `strategy.job-total`, `full` | probe | 510 → 470 | **UNSOUND** (B2) |
| 6 | Probe G: env-guarded Codecov upload, `full` | probe | 485 → 435 | **UNSOUND** (B1) |
| 7 | fixture `share`, `full` | tests/fixtures | 570 → 500 | Sound under assumptions A1, A2 |
| 8 | fixture `groups`, `full` | tests/fixtures | 1140 → 1000 | Sound under assumptions A1, A2 |
| 9 | fixture `setup-chain` (merge), `full` | tests/fixtures | 530 → 385 | Sound under assumption A2; the check `test (b)` disappears |
| 10 | fixture `merge` (lint + typecheck), `full` | tests/fixtures | 220 → 145 | Sound under assumption A2; the check `typecheck` disappears |
| Z | Zustand `test-multiple-builds`, `full`, with the trace | motivating example | mock 477 → 467 (8 → **9** billed min); (h) 74 → 74, no change | Sound under the trace and assumption A2; no real saving shown |

**Summary.**
- Every real-workflow candidate that saves more than 1 s is unsound.
- Both root causes (B1, B2) also affect the `full` variant: probes 5 and 6 pass effect pruning, the
  checker and the round trip.
- The experiments report "0 improvements" for `full`, so B1 and B2 do not change any RQ1 or RQ2
  number. They do invalidate:
  - the RQ3 no-effect-pruning "improvements";
  - any soundness claim for the checker as it stands, including RQ4 verdicts on workflows that
    read env values containing expressions.

## 3. Bugs

### B1 (critical): `env.X` evaluates to the literal text of an expression value

- **Location:** `src/ghav/semantics.py:1618-1622` (`_Evaluator._ctx`) returns `self._py(env[name])`.
  The env value comes from `merged_env` (`semantics.py:176-179`) and is the raw YAML text.
- **Effect:**
  - For `AFFECTED: ${{ needs.changes.outputs.server }}` and `if: env.AFFECTED == 'true'`, the
    condition compares the string `"${{ needs.changes.outputs.server }}"` with `"true"`. The
    condition is therefore constant false, and every step or job it guards never runs in the
    model.
  - The checker compares two programs in which that code is dead, so any change to it is
    "equivalent".
  - Synthesis sees the dead steps as shareable; effect pruning is a heuristic and does not rule
    them out.
- **Minimal checker repro (probe B):**
  - The job has `env: {EVENT: ${{ github.event_name }}}` and a last step `if: env.EVENT == 'push'`
    that runs `codecov/codecov-action@v4`.
  - The candidate deletes that step.
  - `checker.check` returns `equivalent`, with persistent region `ext:codecov@release#4`.
  - On GitHub the upload runs on every push.
- **Minimal `full` repro (probe G, case 6):**
  - Workflow `env: {UPLOAD: ${{ github.event_name == 'push' }}}`.
  - Two jobs each end in `if: env.UPLOAD == 'true'` → Codecov.
  - `full` shares the upload through the read class and emits a workflow in which job `unit` no
    longer uploads. The checker and the round trip say equivalent.
  - Control: the same workflow with `if: github.event_name == 'push'` is refuted, and `full` finds
    nothing.
- **Real instances:**
  - Trilium: `AFFECTED`.
  - langfuse: `CI_CACHE_ALLOWED: ${{ github.event_name != 'pull_request' || ... }}`, with the
    "with pnpm cache" `setup-node` guarded by `env.CI_CACHE_ALLOWED == 'true'`.
  - The experiment report counts 46 workflows (385 conditions) in `workflows.scope` with this
    pattern.
- **Also affected:** a step that reads `$NAME` gets `ctx:env.NAME` bound to the same literal
  text. A `needs.*.outputs` value passed through env is therefore frozen as text, and changes
  upstream of it are invisible.
- **Fix:**
  - Evaluate `${{ }}` in workflow- and job-level env values in the job's context: `needs`,
    `github`, `matrix`, and `vars`/`secrets` as symbols.
  - Or make any env value containing `${{` an opaque per-instance symbol that depends on the
    context keys it references.
  - `synth._ctx_value` (`synth.py:1221`) compares the same raw text and needs the same treatment.

### B2 (high): splitting a matrix changes `strategy.job-total`, `strategy.job-index` and `github.job`, and the model cannot see it

- **Location:**
  - `contracts.py:1232-1233` maps every `strategy.*` reference to `ctx:job-identity`.
  - `semantics.py:1608-1609` gives `job-identity` the value `instance_identity(job)`, which is the
    check name plus the matrix (`semantics.py:417-428`).
  - Emission (`emit.py`, grouping rule 1) splits an original matrix into jobs that each have a
    one-entry `include:` matrix whenever member bodies differ. Instance-as-producer sharing always
    causes this.
  - The split keeps the check name and the matrix, so the model's identity is unchanged, but on
    GitHub `strategy.job-total` becomes 1 and `strategy.job-index` becomes 0.
  - `github.job` / `GITHUB_JOB` become the new job key; the docstring of `instance_identity`
    assumes this away.
- **Repro (probe A, case 5):**
  - `test` has a matrix `shard: [1, 2]` and runs `npx vitest run --shard=${{ matrix.shard }}/${{ strategy.job-total }}`.
  - `full` shares `npx tsc` from `test[shard=1]` and emits `test-shard-1` / `test-shard-2`, each
    with `include: [{shard: N}]`. The checker and the round trip say equivalent.
  - On GitHub, shard 1 runs `--shard=1/1` (the whole suite) and shard 2 runs `--shard=2/1`, which
    vitest rejects.
- **Real instance:** both Trilium candidates split `test_ckeditor5` in the same way, and its step
  uses `--shard=${{ matrix.shard }}/${{ strategy.job-total }}`.
- **Fix, one of:**
  - Include the matrix size and index, and the emitted job key, in the value of `job-identity`.
    A split that changes them is then refuted whenever a step reads `strategy.*` or `github.job`.
  - Forbid splits of matrices whose steps read `strategy.*` or `github.job`.

### B3 (high, only for local reusable workflows): inlining exposes caller env and unpassed secrets

- **Background (GitHub docs, "Reusing workflows"):**
  - Workflow-level `env` of the caller is **not** propagated to a called workflow.
  - A secret that is not passed (no `secrets:` entry, no `secrets: inherit`) is empty inside the
    callee.
- **Lowering and semantics get both wrong:**
  - `merged_env` (`semantics.py:176-179`) merges `p.env` into callee jobs.
  - `lower.py:876-880` substitutes only passed secrets and leaves the other `secrets.*`
    references as global symbols shared with the caller.
- **Emission then makes the model true:** `emit.py` rule 5 inlines the callee jobs into the
  caller, where caller env and every repository secret are visible. The round trip agrees with
  itself and says equivalent.
- **Repro (probe C):**
  - Caller has `env: {UPLOAD: 'yes'}` and a job that calls `./.github/workflows/callee.yml` with no
    secrets.
  - The callee has steps `if: env.UPLOAD == 'yes'` → Codecov and `if: secrets.NPM_TOKEN != ''` →
    `npm publish`.
  - The **identity** emission inlines both steps, and the round trip says equivalent.
  - On GitHub the original skips both steps. The emitted workflow uploads, and publishes whenever
    the repository has an `NPM_TOKEN`.
- **Exposure:**
  - A textual scan finds 11 workflows in `workflows.scope` whose callee reads a secret the caller
    does not pass (for example RocketChat `ci.yml`: `NPM_TOKEN`; kriasoft `ci.yml`:
    `CLOUDFLARE_API_TOKEN`).
  - None are in `eval`, `random` or `dev`, and no caller-env leak occurs in the corpus.
- **Fix:**
  - Lower unpassed secrets to `''`, except `GITHUB_TOKEN` and `secrets: inherit`.
  - Do not merge the caller's workflow env into callee jobs.
  - When inlining, write explicit job-level env overrides (or keep the call frozen).

### B4 (medium, security): `permissions: {}` is dropped by emission

- **Location:** `emit.py:858-861` uses `elif job.permissions:`. The lowered value of
  `permissions: {}` is `()`, which is falsy, so the key is omitted.
- **Effect:** the job then inherits the workflow or default token permissions.
- **Observed:** ueberdosis/tiptap `publish.yml`, jobs `notify-slack` and `notify-slack-failure`,
  in the `optimized_full.yml` of both random backends. These jobs go from no token scopes to
  `contents: read`. Under a repository whose default is write-all, the token would get write
  access.
- **Why the round trip misses it:** the semantics ignores permissions.
- **Fix:** `elif job.permissions is not None:`.

### B5 (medium): typed reusable-workflow inputs become strings in emitted YAML

- **Location:** `lower.py:323-333` (`_interp_obj`) interpolates every string of a callee's `raw`
  (`lower.py:687`, `852`). `timeout-minutes: ${{ inputs.timeout-minutes }}` (a `number` input,
  default 20) becomes the string `'20'`, which `emit.py:856-857` writes verbatim.
- **Observed:** n8n `ci-master.yml`, job `verify-single-instance-npm-verify`. actionlint reports
  a `syntax-check` error ("expecting a single ${{...}} expression or float number literal").
- **Status:** PLAUSIBLE that GitHub rejects the workflow file; not tried on GitHub.
- **Fix:** keep the input's declared type when the whole value is one `${{ inputs.x }}`.

### B6 (medium): transfer `out_paths` default to `dist` even when the tool writes elsewhere

- **Location:** `contracts.py:1656-1663`. The order is: explicit flag, else the tsconfig `outDir`
  of the project (not its `references`), else `dist`, for every `build` row (vite, rollup, tsup,
  webpack, esbuild, unbuild, tsc).
- **Why it matters:** the semantics trusts these paths for exact copies.
- **Repro (probe K):**
  - Root `tsconfig.json` is `{"files": [], "references": [a, b]}`; each package has
    `outDir: lib`. Two matrix instances run `npx tsc -b`.
  - `full` shares the build with a transfer of `dist`; the checker and the round trip say
    equivalent.
  - On GitHub, `tar -cf … dist` fails because `dist` does not exist, so the producer job fails.
    The consumer would miss `packages/*/lib` in any case.
- **Other affected cases:** a `vite build` whose config sets `build.outDir` is affected the same
  way, as is any build under a tsconfig whose `outDir` the tool ignores.
- **Fix:**
  - Follow `references` for `tsc -b`.
  - Treat `out_paths` of non-tsc build tools as unknown (not transferable) unless a flag or a
    trace gives them. Traces (guide 8) already do this for Zustand.

### B7 (low; PLAUSIBLE, not tried on GitHub): the transfer idiom on Windows runners

- **Location:** `contracts.py:484-493`. The idiom runs `tar -cf "$RUNNER_TEMP/…"` under
  `shell: bash`.
- **Probe J:** `full` emits it for `runs-on: windows-latest`.
- **Risk:** in Git Bash, `RUNNER_TEMP` is `D:\a\_temp`, and GNU tar reads `D:` as a remote host
  unless `--force-local` is given.
- **Fix:** restrict transfers to Linux/macOS, or add `--force-local` / use `cygpath`.

### Not bugs, but assumptions the paper must state

- **A1: job- and workflow-level env is read only when its name appears in the command text**
  (guide 3.3).
  - The `build`, `lint` and `test` templates execute repository config files
    (`vite.config.*`, `eslint.config.*`, `jest.config.*`), and those can read any
    `process.env`.
  - The `share` and `groups` fixtures share `npm run build` and `npm run lint` across instances
    whose `TARGET: ${{ matrix.mode }}` differs. This is sound only if no config reads `TARGET`.
- **A2: a check correspondence may drop check names.**
  - A merge removes a check: `test (b)` in `setup-chain`, `typecheck` in `merge`. A required
    status check with that name would then never report.
  - A share moves a failure: in `share`, a lint failure now fails only `test (a)`.
  - Both are within the "group conjunction" definition, but `ext:check:*` is otherwise treated
    as persistent state.
- **A3: templates with known missing writes are used for acceptance.**
  - The Zustand trace shows `vitest` writing `node_modules/.vite` (`dep`), which the `test`
    template omits (guide 8 keeps the template).
  - A merge that puts two test runs in one job relies on this omission.
- **A4: the transfer idiom uses tag refs** (`actions/upload-artifact@v4`) in repositories that
  pin every action by SHA (Zustand). Organizations that enforce SHA pinning would reject the
  emitted workflow.

## 4. Per-case notes

1. **Trilium, mock (case 1).**
   - `pnpm/action-setup`, `setup-node` and `pnpm install` are shared from
     `test_ckeditor5[shard=1]` with no transfer. They are dropped from `test_ckeditor5[shard=2]`,
     `test_ckeditor5_merge`, `test_server` and `test_standalone`.
   - Every dropped step is guarded by `env.AFFECTED == 'true'`, which is dead in the model (B1).
   - On any run that touches those suites, the tests run without Node 24 or dependencies.
   - The shard split also changes `strategy.job-total` (B2).
   - YAML is valid (actionlint clean); workflow permissions and concurrency are kept.
2. **Trilium, historical (case 2).**
   - Same moves, except that `setup-node` is shared from `test_standalone`.
   - `test_ckeditor5-shard-1` therefore installs without `setup-node`, and `test_standalone`
     tests without an install. Unsound for the same reasons.
3. **langfuse (case 3).**
   - "Use Node.js … with pnpm cache" is dropped from `knip`, `tests-eslint-plugin`,
     `tests-in-app-agent-sandbox-runtime`, `tests-storybook` and `tests-web-client`.
   - Its guard `env.CI_CACHE_ALLOWED == 'true'` is dead in the model (B1). On pushes and
     same-repo PRs, those jobs run on the runner's default Node.
   - The 24 actionlint issues are unknown `blacksmith-*` runner labels, identical to the
     original's.
4. **fast-check (case 4).**
   - Drops `echo "Success"` under `if: needs.pre_all_checks_passed.result == 'success'` in
     `all_checks_passed`.
   - The job verdict is still decided by the `exit 1` step, and logs are not in P. Sound; 1 s.
5. **Probes A and G (cases 5, 6).** See B2 and B1.
6. **`share` and `groups` (cases 7, 8).**
   - The build is shared by instance-as-producer: consumer `needs: producer`, `if: !cancelled()`,
     then download and untar of `dist` after `npm ci`.
   - The producer packs right after the build, before lint and test.
   - The declared outDir `dist` matches `tsconfig.json`.
   - Conditions: if the producer's build fails, the consumer's download fails. The original
     consumer fails its own build, which reads the same values, so the verdicts are equal.
   - `lint` is dropped from the consumer, which relies on A1 and A2.
   - Check names are kept: `name: test` + one-entry matrix renders `test (a)`, which lowering
     models.
7. **Merges (cases 9, 10).**
   - The merged step's condition is `!cancelled() && steps.ghav_i.outcome != 'failure' …` over
     the guest's original predecessors, so the second test or typecheck still runs after the
     first fails.
   - `dep` is rebuilt identically (same steps, same order).
   - Check names are lost (A2), and `node-version: 20` becomes `'20'` (harmless).
8. **Zustand (Z).**
   - `pnpm run build` has an observed contract (trace: writes only `dist/`; `prebuild` deletes
     `dist` first).
   - `esm` downloads `cjs`'s `dist` after `pnpm install` and before the matrix-guarded
     `sed` patches. Those patches touch `tests/` and `vitest.config.mts` (`src`), not `dist`.
   - NODE_ENV is set only on the patch and test steps, and the build does not read matrix values.
   - Check names are kept, and `fail-fast: false` is kept.
   - Sound under the trace (taken on macOS and node 22, not on ubuntu CI) and A2/A3.
   - The mock saving of 10 s comes from pricing the observed build at the ⊤ row (30 s) against a
     20 s transfer. Billed minutes rise from 8 to 9, and the historical backend finds no
     improvement (build ≈ 9 s).

## 5. YAML and schema validation

- **actionlint, 273 `optimized_*.yml` in `output/stages`:** 24 files have findings absent from
  their original.
  - 20 are slab/quill: `actions/checkout@v3` "too old", inherited from the inlined callee
    `_test.yml`, which a lint of the caller alone does not see.
  - 2 are n8n `ci-master` (B5, and a constant `if: 'false'` from correct partial evaluation).
  - 2 are pear-desktop (a constant `if: 'false'`, correct because the job has no matrix).
- **actionlint and yamllint on all audited candidates, probes and fixtures:** no findings except
  langfuse's inherited runner labels.
- **Identity emission versus original, 113 directories:** 24 differ.
  - Semantically neutral:
    - `on` list → map;
    - `name:` equal to the key dropped;
    - `needs: []` dropped;
    - job renames with `needs` rewritten, when reusable calls are inlined or matrices over calls
      are split;
    - constant-folded `if`.
  - Semantically relevant: B4 (tiptap `permissions: {}`) and B5 (n8n `timeout-minutes`).
- **Fail-fast (probe F):** a default fail-fast matrix whose members upload to Codecov was left
  untouched, which is conservative.
- **Conditions (probe I):** with an event-guarded job and an always-running job that both build,
  `full` picks the always-running job as producer, which is correct.
- **`working-directory` builds (probe H):** transfer paths are rebased to the workspace root
  (`packages/a/lib`), which is correct.

## 6. Requests

- **`semantics.py`:** fix B1 (and the env text comparison in `synth._ctx_value`). Also fix the
  callee part of B3: unpassed secrets are empty, and caller workflow env is not visible.
- **`semantics.py` / `contracts.py` / `emit.py`:** fix B2 by making matrix size and index, and the
  job key, part of the identity, or by refusing such splits.
- **`emit.py`:** fix B4. For B3, stop inlining callees whose steps read caller env or unpassed
  secrets, or emit explicit overrides.
- **`lower.py`:** fix B5 (keep typed inputs) and the secrets substitution of B3.
- **`contracts.py` and `docs/CONTRACT_GUIDE.md`:**
  - fix B6 (`tsc -b` references; no `dist` default for non-tsc tools);
  - fix B7 (no tar idiom on Windows, or `--force-local`);
  - state A1 and A3 as assumptions.
- **Tests (`tests/`):** add probes A, B, C, G and K as regression tests. Each is under 20 lines of
  YAML plus `package.json` and `tsconfig.json`, and today each must fail.
- **Paper / RESEARCH_SCOPE:** state A2 (check names may disappear or move within a group) and A4.
  Do not present the Trilium no-effect-pruning result, or any no-effect-pruning "improvement",
  as a verified saving.
