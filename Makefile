.PHONY: help sync test test-scripts \
        fetch-pool fetch-dev fetch-pool-snapshots fetch-devopt fetch-timings fetch-all \
        build-pool-scope build-workflows-scope build-eval-scope build-random-scope build-devopt-scope build-scopes \
        run-stages run-eval-historical run-eval-mock run-random run-devopt run-all audit-consistency aggregate export-tables check-determinism

UV ?= uv
PY = $(UV) run python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-24s %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────────────────

sync: ## Install dependencies into .venv (uv sync)
	$(UV) sync

test: ## Run the test suite (offline)
	$(UV) run pytest -q

test-scripts: ## Run the tests of the experiment scripts (scripts/ablation, scripts/aggregate)
	$(UV) run pytest -q scripts/ablation scripts/aggregate

# ── Resources (network via `gh api`; outputs are committed, cached items are skipped) ──

fetch-pool: ## Search GitHub for the candidate pool and pin commits -> resources/pool.json (FORCE=1 re-searches)
	$(PY) scripts/resources/fetch_pool.py $(if $(filter 1,$(FORCE)),--force)

fetch-dev: ## Fetch repository snapshots for scope/dev.scope -> resources/repos/
	$(PY) scripts/resources/fetch_snapshots.py --scope scope/dev.scope

fetch-pool-snapshots: ## Fetch repository snapshots for scope/pool.scope -> resources/repos/ (LIMIT=N)
	$(PY) scripts/resources/fetch_snapshots.py --scope scope/pool.scope $(if $(LIMIT),--limit $(LIMIT))

fetch-devopt: ## Fetch PR metadata and before/after snapshots for resources/devopt_prs.txt
	$(PY) scripts/resources/fetch_devopt.py

TIMING_SCOPES ?= scope/dev.scope scope/eval.scope scope/random.scope

fetch-timings: ## Fetch historical run/job timings for TIMING_SCOPES -> resources/timings/ (cached items skipped)
	$(PY) scripts/resources/fetch_timings.py $(foreach s,$(TIMING_SCOPES),--scope $(s))

fetch-all: fetch-dev fetch-pool build-pool-scope fetch-pool-snapshots fetch-devopt fetch-timings ## Every network step, in order

# ── Scopes (offline, deterministic) ──────────────────────────────────

build-pool-scope: ## pool.json minus dev.scope repos, pinned -> scope/pool.scope
	$(PY) scripts/scope/build_pool_scope.py

build-workflows-scope: ## Pool workflows triggered by push/pull_request that lower -> scope/workflows.scope
	$(PY) scripts/scope/build_workflows_scope.py --check-lowering

build-eval-scope: ## Redundancy-filtered sample (needs ghav.pipeline.workflow_stats) -> scope/eval.scope
	$(PY) scripts/scope/build_eval_scope.py

build-random-scope: ## Unfiltered random sample of 30 (needs ghav.pipeline.workflow_stats) -> scope/random.scope
	$(PY) scripts/scope/build_random_scope.py

build-devopt-scope: fetch-devopt ## Developer-optimization before/after pairs -> scope/devopt.scope, scope/devopt.pairs.json
	$(PY) scripts/scope/build_devopt_scope.py

build-scopes: build-pool-scope build-workflows-scope build-eval-scope build-random-scope build-devopt-scope ## All scopes

# ── Experiments (offline, deterministic; docs/EXPERIMENTS.md) ─────────
#
# Stage files go to output/stages/<scope-stem>.<cost>/<benchmark-id>/. Every (benchmark, variant)
# runs in a fresh worker process; results do not depend on WORKERS.
#
# MAX_EXPANSIONS is applied uniformly to every variant. It is 250 instead of the SynthConfig
# default 2000 because no-effect-pruning checks every candidate and needs about 23 s per
# expansion on the largest evaluation workflow (94 job instances), i.e. hours at 2000. At 250
# every other variant finishes its search on every workflow of eval/random/devopt before the
# budget (the largest need is 244 expansions), so only no-effect-pruning is capped.

SCOPE ?= scope/eval.scope
COST ?= historical
VARIANTS ?= full,rules,no-effect-pruning,no-cost-bound,no-read-classes
WORKERS ?= 8
MAX_EXPANSIONS ?= 250
EXP_ARGS = --workers $(WORKERS) $(if $(MAX_EXPANSIONS),--max-expansions $(MAX_EXPANSIONS))

define require-script
	@test -f $(1) || { echo "error: $(1) is not implemented yet" >&2; exit 2; }
endef

run-stages: ## Run VARIANTS over SCOPE with cost backend COST -> output/stages/<scope>.<cost>/ (WORKERS, MAX_EXPANSIONS)
	$(call require-script,scripts/ablation/run_stages.py)
	$(PY) scripts/ablation/run_stages.py --scope $(SCOPE) --variants $(VARIANTS) --cost $(COST) $(EXP_ARGS)

run-eval-historical: ## eval.scope, every variant, historical cost
	$(MAKE) run-stages SCOPE=scope/eval.scope COST=historical VARIANTS=$(VARIANTS)

run-eval-mock: ## eval.scope, every variant, mock cost
	$(MAKE) run-stages SCOPE=scope/eval.scope COST=mock VARIANTS=$(VARIANTS)

run-random: ## random.scope, variant full, both cost backends
	$(MAKE) run-stages SCOPE=scope/random.scope COST=historical VARIANTS=full
	$(MAKE) run-stages SCOPE=scope/random.scope COST=mock VARIANTS=full

run-devopt: ## devopt.scope, variant full (mock cost: no timings for these commits), and RQ4 pair checks
	$(MAKE) run-stages SCOPE=scope/devopt.scope COST=mock VARIANTS=full
	$(PY) scripts/ablation/check_devopt.py --workers $(WORKERS)

run-all: run-eval-historical run-eval-mock run-random run-devopt ## Every experiment (independent; `make -j4 run-all` runs them in parallel)

audit-consistency: ## Cost simulation vs semantics disagreements for every scope -> output/audit/
	$(foreach s,workflows eval random dev devopt,$(PY) scripts/ablation/audit_consistency.py --scope scope/$(s).scope --workers $(WORKERS) &&) true

aggregate: ## Turn output/stages/* into output/rq*.csv, output/summary.json and output/tables/rq*.tex
	$(call require-script,scripts/aggregate/aggregate.py)
	$(PY) scripts/aggregate/aggregate.py

export-tables: ## Copy output/tables/*.tex into the paper repo (../ghav-pldi2027/tables/)
	$(call require-script,scripts/aggregate/export_tables.py)
	$(PY) scripts/aggregate/export_tables.py

DETERMINISM_DIR ?= /tmp/ghav-determinism
WORKERS2 ?= 3

check-determinism: ## run-all + aggregate twice from clean stages (WORKERS, then WORKERS2) and compare outputs
	rm -rf $(DETERMINISM_DIR) output/stages && mkdir -p $(DETERMINISM_DIR)
	$(MAKE) -j4 run-all WORKERS=$(WORKERS)
	$(MAKE) aggregate
	cp -R output $(DETERMINISM_DIR)/run1
	rm -rf output/stages
	$(MAKE) -j4 run-all WORKERS=$(WORKERS2)
	$(MAKE) aggregate
	$(PY) scripts/aggregate/compare_outputs.py $(DETERMINISM_DIR)/run1 output
