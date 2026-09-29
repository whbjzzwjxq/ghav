# GHAV Architecture

GHAV optimizes a GitHub Actions workflow by synthesis. The specification is the original
workflow's semantics on persistent state; the components are its steps, each with a
region-effect contract; candidates are validated end to end by an SMT-based equivalence
checker. There are no hand-written rewrite rules.

Design decisions are recorded in the paper repo at
`../ghav-pldi2027/tmp/research/RESEARCH_SCOPE.md`. This document is the implementation
contract between modules. Every module must follow the determinism rules below.

## Determinism rules (all modules)

1. Iterate every dict/set in sorted order. Never depend on hash order.
2. No wall-clock-dependent decisions. Search budgets count node expansions, not seconds.
   Wall-clock time may be *reported* but must never change a result.
3. Randomness only through `random.Random(SEED)` with `SEED = 42` passed explicitly.
4. z3: create solvers with `z3.SolverFor("QF_UF")` or `z3.Solver()` and set
   `set_param("smt.random_seed", 0)` and `set_param("sat.random_seed", 0)` once at import.
5. Network access happens only in `scripts/resources/`; everything else reads the cached
   files under `resources/`. Cached files are committed so the experiments reproduce offline.
6. Outputs are JSON with `sort_keys=True, indent=2` or CSV with sorted rows.

## Layout

```
src/ghav/
  wir.py          data model (Region, Effect, Contract, Expr, Binding, Step, JobInstance, Program)
  expr.py         GitHub expression parser/evaluator for the supported subset
  lower.py        GHA Core -> WIR (matrix expansion, reusable-workflow expansion, default conditions, freezing)
  contracts.py    contract inference from the global contract guide (docs/CONTRACT_GUIDE.md)
  semantics.py    symbolic evaluation of a Program into z3 terms; race/well-formedness checks
  checker.py      end-to-end equivalence checking with counterexamples
  cost.py         machine-time cost model (mock table and historical medians; machine seconds and billed minutes)
  synth.py        synthesis: read classes, sketch moves, pruning, deterministic completion, search
  emit.py         WIR -> GitHub Actions YAML
  pipeline.py     per-workflow driver used by the CLI and the experiment scripts
  cli.py          `ghav lower|contracts|check|optimize|emit|stats`
scripts/scope/      deterministic scope builders (candidate pool, snapshots, eval set, random sample)
scripts/resources/  network fetchers (gh api) that populate resources/ (repos, pool, devopt, timings)
scripts/traces/     local filesystem traces of dev-set workflows -> resources/traces/ (observed contracts)
scripts/aggregate/  turn output/stages/* into CSV and LaTeX tables in output/tables/
scripts/ablation/   run synthesis variants
scope/*.scope       one "owner/repo@sha:.github/workflows/file.yml" per line, sorted
resources/repos/<owner>__<repo>/<sha>/...   cached repository files needed by lowering and contracts
resources/timings/<benchmark-id>/            cached GitHub run/job timings and derived medians.json
resources/traces/<owner>__<repo>/<sha>/      ghav-trace/1 files (observed contracts, contract guide 8)
output/stages/<scope>/<benchmark-id>/        per-benchmark results
tests/              pytest; every module has unit tests; tests/fixtures holds small workflows
```

Benchmark id: `<owner>__<repo>__<workflow-file-stem>`.

## Module contracts

### expr.py
- `parse(text: str) -> Expr`: parses the body of `${{ }}` or a bare `if:` string. Unsupported
  syntax yields `Expr("opaque", (text, *sorted_ctx_refs))`; never raises on valid YAML strings.
- `uses_status_function(e: Expr) -> bool`: true if `success()/failure()/always()/cancelled()` occurs.
- `substitute(e: Expr, env: Mapping[str, str]) -> Expr`: replace known ctx keys (matrix.*, inputs.*) by literals.
- `ctx_refs(e: Expr) -> frozenset[str]`; `step_refs`, `needs_refs` likewise.
- `interpolate(s: str, env) -> str`: substitute `${{ matrix.x }}`/`${{ inputs.x }}` inside strings.

### lower.py
`lower(workflow_path: Path, repo_root: Path) -> Program`
- Only triggers are recorded; the program is interpreted for `push` and `pull_request`.
- Matrix expansion follows GitHub rules (cross product, then `exclude`, then `include`).
  Instance id: `f"{job}[{','.join(f'{k}={v}' for k,v in sorted(matrix))}]"`, or `job` without a matrix.
  `check_name` follows GitHub: `name:` interpolated if present, else `job (v1, v2, ...)` in matrix-key order
  of the YAML; for reusable calls `caller / callee-check`.
- Local reusable workflows (`uses: ./.github/workflows/x.yml`) are expanded: callee jobs become
  instances with id `caller[...]/callee-job`, inputs substituted, `needs` rewritten.
  Remote reusable workflows make the caller frozen.
- Local composite actions (`uses: ./path`) stay ONE step (opaque task). Their `action.yml`
  text is available to contracts via `repo_root`.
- Default conditions are folded in: a step's cond becomes `success() && <its if>` unless its `if`
  uses a status function; same for jobs with `needs`.
- Every step records `Step.origin` (original job key, step index) and `Step.origin_instance`
  (the original job instance), used for check groups, occurrence keys of external writes, and
  historical prices.
- Lowering fails closed: a job, step, strategy, `defaults` or workflow key it does not know
  freezes the job (`unsupported-*`). A job is frozen (kept verbatim; its effect is the union of
  its steps' contracts, see semantics.py) if it uses: `container`, `services`,
  `continue-on-error` (job or step), job-level `concurrency`, `environment` (an unknown key),
  remote reusable workflows, a self-hosted runner label, `strategy.max-parallel`, or an
  expression lowering cannot parse into a non-opaque form inside `needs`/`outputs`.
  `timeout-minutes` (job and step) has no correctness effect (every task is assumed to
  terminate): it freezes nothing and is emitted verbatim. `frozen_reasons` lists the causes.
- Post-lowering rules (RESEARCH_SCOPE "Verification gaps and resolutions") freeze: readers of
  `needs.<M>.result` of a multi-instance M (`multi-instance-result`); readers of
  `needs.<M>.outputs` unless every instance of M produces the same term
  (`multi-instance-output`); fail-fast matrices that break rule (i)-(iii) (`fail-fast`); and every
  job of a reusable-workflow call that contains a frozen job or is named by `needs.<call>`
  (`frozen-call`; the call keeps the caller's YAML as `raw` and is emitted back unchanged).
- Workflow-level `concurrency`, `permissions`, `env`, `defaults` are kept and emitted verbatim.
- `fail_fast_group`: for a matrix job whose `strategy.fail-fast` is not false, all instances get
  group `job`.

### contracts.py
Implements docs/CONTRACT_GUIDE.md (v1.1) mechanically; pure and deterministic.
- `infer(step, job, repo_root, *, workflow_env=(), workflow_defaults=None, observed=None) -> Contract`:
  path-to-region mapping, context-read rules, the template table keyed by action reference or
  command, `package.json` script resolution (root only), file utilities (guide 4.5), and ⊤
  fallback. Every step reads `ctx:platform`. Unknown commands give `TOP_CONTRACT`; a ⊤ step whose
  binding key has an observed contract (`observed`, guide 8) gets template `observed`.
- `annotate(program, repo_root, *, observed=None) -> Program` infers every step (frozen jobs
  included) and then applies the cache-transparency rule (guide 7: a violating restore gets
  `cache-opaque`). `observed=None` loads the benchmark's trace file if one exists
  (`trace_path`, `load_observed`); `{}` means the static guide only. `report(...)` is the audit
  behind `ghav contracts`.
- Official artifact actions: `upload-artifact` reads the kinds covering `with.path` and writes
  `run:<name>`; `download-artifact` reads `run:<name>` and writes `out` (or `tool` outside the
  workspace). Cache actions have no correctness effect (cold-cache semantics).
- The transfer idiom (guide 9) is defined here: `transfer_bindings(name, paths)` gives the
  (pack, upload, download, unpack) bindings and `parse_transfer(binding)` recognizes them. An
  upload or download of a `ghav-` name that is not an idiom step is ⊤.

### semantics.py
- `well_formed(p: Program) -> list[Issue]`: race-freedom. Two instances not ordered by `needs*`
  must have independent effects on shared regions (RUN, EXT, OUTPUT), except writes of the same
  artifact name are an error in GitHub (v4 names are unique per run) and are reported.
  `blocking_issues(original, candidate)` ignores races the original already has.
- `evaluate(p, ctx, *, assume_success=False) -> Observation`: symbolic execution in one fixed
  topological order (valid because well-formed programs are confluent). Returns, for every
  persistent region, its final z3 term, for every check name a z3 Bool verdict (pass iff status
  in {success, skipped}; cancelled counts as fail), and the read values of every step
  (`step_reads`, used for read classes).
- Encoding: one uninterpreted sort `Val`; each context key is a `Val` constant (concrete strings map
  to distinct constants); each (task key, written region) is an uninterpreted function `F` over
  the values of the task's reads (sorted by region); `ok` is an uninterpreted Bool function of the
  same arguments. A step with guard g writes `If(g, F(args), old)`; its status is
  `If(g, If(ok, success, failure), skipped)`.
- Job-local writes are updates: when a task writes a job-local kind (src, dep, out, tool) that it
  does not read, the kind becomes `U_kind(old, F(args))` (`U` uninterpreted), because a task
  changes part of a region and the rest survives. A contract therefore never needs to read a
  kind only to keep earlier writers alive.
- The transfer idiom is an exact copy of the producer's `out` when `transfer_plan` shows that the
  packed paths cover everything written into `out` since the producer's last clean checkout and
  the consumer's `out` is still clean; plain uploads/downloads are uninterpreted. A clean checkout
  empties `out`.
- External writes are keyed by the originating step occurrence (`ext:<service>@<instance>#<index>`);
  ⊤ steps write the catch-all `ext:*`. A frozen job is one opaque task whose effect is the union of
  its steps' contracts; a job frozen for a reason its steps do not describe (or without steps) is a
  blanket ⊤ task.
- Persistent regions P: `ext:*` (including `ext:check:<name>`) and `run:<name>` for artifacts the
  ORIGINAL workflow uploads. Artifacts whose name starts with `ghav-` are internal transfers and
  are excluded from P. Cross-run caches are not in P; cache-transparency findings
  (`cache_issues`) are reported, never blocking.

### checker.py
`check(original, candidate, *, rlimit=0, canonical=True, recover=True, read_classes=True) -> CheckResult`
- Runs `well_formed` on the candidate (and records original issues; inherited races are not blocking).
- `recover_origins` first maps steps of programs not produced by synthesis (a re-lowered emitted
  workflow, a developer-written change) back to original steps with the same binding, and their
  `origin_instance` to the original occurrence's instance. Transfer steps are recognized by the
  idiom (`contracts.parse_transfer`).
- Check correspondence is derived, not searched: relate each new check to the old check of the
  original instance that contributed a step (`Step.origin_instance`, else every old check of
  `Step.origin`'s job) and to the old checks of displaced occurrences that a step accounts for
  through a shared read class (`read_class_accounts`); groups are connected components (extended
  so every fail-fast group falls into one group); compare the conjunction of verdicts per group.
- Asserts `event_name ∈ {push, pull_request}` and asks z3 for a context where some persistent
  region or group verdict differs. Unsat → equivalent. Sat → counterexample: event, failing tasks,
  differing region.
- fail-fast (provisional, see RESEARCH_SCOPE): all members of a fail-fast group fall into one
  check group; persistent writes by members are checked as refinement (the candidate's write
  condition must lie between the original's "no member failed" and "own guard" conditions).
- `CheckResult`: `equivalent`, `reason`, `counterexample`, `issues`, `groups`, `persistent`,
  `recovered_origins`, `smt_ms` (reported only).

### cost.py (see docs/COST_MODEL.md)
- One interface, `CostModel`, two backends: `MOCK` (the documented template table, keyed by
  contract template; guide-v1.1 templates copy the row of what they refine) and the historical
  model (`historical_model(p, medians)`, `load_historical_model(p, benchmark_id, timings_dir)`
  reading `resources/timings/<benchmark-id>/medians.json`, matched to the ORIGINAL program).
  `CostModel.price_step(job, step) -> (seconds, from history?)` prices through `Step.origin` and
  `Step.origin_instance`; `CostModel.overhead(job)` is the per-job fixed cost.
- `simulate(p, event, model)` evaluates conditions concretely per event, assuming every task
  succeeds. `expected_machine_time(p, contexts=None, model=MOCK)` and
  `expected_billed_minutes(...)` weight events by `contexts`, or by the model's event weights
  when `contexts=None` (0.5/0.5 for the mock model). Billed minutes: each executed job bills
  `max(1, ceil(seconds / 60))`. `cost_report` gives both metrics per job, per event and in
  expectation, with price provenance and the historical match coverage.

### synth.py
`synthesize(p, cfg, *, repo_root, model=None, contexts=None, observed=None) -> SynthResult`
- `repo_root` is required (acceptance needs the emission round trip, which lowers the emitted
  YAML against it); `observed` are the observed contracts `p` was annotated with (None = the
  contract layer's default for `p`, `{}` = static guide) and are reused by the round trip.
- Read classes: symbolically evaluate the program with every task succeeding; two step
  occurrences are in one class when their task keys and read-value terms are identical.
- Moves (search-space generators, no premises): `merge(a, b)` fuses two groups of compatible
  instances (same reusable-call path, condition, needs, permissions, runs-on, compatible env,
  neither frozen); `share(class, producer)` runs one occurrence of a read class and displaces the
  others. Producer encodings: `"job"` (a NEW job running the backward slice of a representative,
  the step, and the upload) or an existing member instance (instance-as-producer: the upload
  goes right after its occurrence). One producer serves one read class (never across groups).
  A displaced occurrence becomes download + unpack plus a `needs` edge when the step writes
  `out` with `out_paths`; otherwise it is dropped (its verdict is accounted for by the class).
- Completion is deterministic: conditions take the exact form (original condition and its kept
  original predecessors did not fail: `!cancelled() && X && steps.<p>.outcome != 'failure'`;
  jobs with new needs get `needs.<n>.result == 'success'` over their original needs only);
  transfers use the idiom with `ghav-` names carrying every instance-distinguishing value;
  transfer steps get origin index `-(1000 + i)`; check grouping is derived by the checker.
  Steps from different reusable-call paths never share a job.
- Effect pruning (flag `effect_pruning`) rejects a completed sketch before checking with reason
  `untransferred-read` (a kept step's job-local writer chain or step reference differs by read
  class from the original, or a transfer is not exact), `status-dependency` (a dropped
  occurrence gated, through `success()`/`failure()`, a later persistent write of its instance or
  a gated persistent-writing dependent not also downstream of the producer), `settings` (a moved
  step sees a different context value; merges also need one runs-on, one fail-fast group, at most
  one identity-bound member), `well-formed`, or `fail-fast` (the RESEARCH_SCOPE rule). Without
  effect pruning the fail-fast rule is an acceptance guard.
- Search: best-first on estimated machine time, deterministic tie-break on the canonical sketch
  key; budget `max_expansions` (expansions, never seconds); each expansion pushes at most
  `max_children` (256) children and a class tries at most `max_producers` (8) member instances
  as producers; `cost_bound` drops sketches whose lower bound is not below the best verified
  cost; sketches refuted by the checker, the round trip, a hard constraint, or a monotone pruning
  reason are nogoods whose supersets are skipped. Every improving candidate goes to
  `checker.check` and then to the round trip `check(original, lower(emit(candidate)))`; only
  candidates passing both count.
- `SynthConfig`: `moves ∈ {"full", "rules"}` ("rules" = merge + whole-prefix hoist of a single
  matrix job into a new producer job), `effect_pruning`, `cost_bound`, `read_classes`,
  `max_expansions` (default 2000), `rlimit`, `max_children`, `max_producers`; variants `full`,
  `rules`, `no-effect-pruning`, `no-cost-bound`, `no-read-classes`.
- `SynthResult`: best program, its cost and the original's (machine seconds, and billed minutes
  reported alongside), expansions, candidates generated/checked/accepted, rejections by pruning
  reason / completion reason / checker / round trip / guard / bound / nogood, the best moves, the
  best candidate's check and round-trip results, wall-clock ms (reported only).

### emit.py
`emit(p: Program) -> str` (YAML). Frozen jobs are emitted from `raw` (a frozen reusable-workflow
call as the original call). Instances of one original matrix job with identical bodies are folded
back into a matrix; others become separate jobs keyed from their instance ids, with a
`strategy.matrix` (`include:`) so check names and `${{ matrix.* }}` stay valid. Steps are emitted
from `Step.raw` when it lowers back to the same binding. Rewritten conditions keep explicit status
references and use `!cancelled()` (never `always()`). Every emitted matrix states `fail-fast`;
`timeout-minutes` is written verbatim. Transfers use the idiom (`transfer_steps(producer_job,
consumer_job, out_paths, name, *, repo_root=None)`, built on `contracts.transfer_bindings`).
Emission is not trusted: `check(original, lower(emit(p)))` is part of acceptance.

### pipeline.py and cli.py
- `workflow_stats(scope_line)` (used by the scope builders): `instances`, `frozen`
  (= `frozen_instances`), `steps`, `top_steps`, `redundancy_ratio`, `redundant_seconds`,
  `machine_seconds` (mock cost, static guide).
- `run_benchmark(scope_line, variant, out_dir, *, cost_backend="mock"|"historical",
  resources=None, max_expansions=None, use_traces=False)` writes `lowered.json`,
  `contracts.json`, `synth_<variant>.json`, `optimized_<variant>.yml`, `check_<variant>.json`
  (the round trip) with sorted keys, and wall-clock time only to `timing_<variant>.json`.
  Evaluation runs use the static guide (`use_traces=False`) for every benchmark.
- `ghav lower|contracts|check|optimize|emit|stats`; workflow arguments are paths or pinned scope
  lines; `--traces` adds the benchmark's observed contracts; `optimize --cost historical` prices
  with the benchmark's medians (scope lines only).

## Experiments (see docs/EXPERIMENTS.md)

Scopes are built by scripts from committed inputs; stages write to `output/stages/<scope>/<id>/`;
aggregate scripts write `output/*.csv` and `output/tables/*.tex`; `make export-tables` copies the
tables into the paper repo. The Makefile exposes one target per step.
