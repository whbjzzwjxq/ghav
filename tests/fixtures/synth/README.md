Fixtures for tests/test_synth.py and tests/test_pipeline.py (synthetic, not from a real
repository). Each directory is a minimal repository: one workflow plus the `package.json`
(and `tsconfig.json`) that contract inference reads.

- `merge/`: two jobs with an identical checkout/setup/install prefix (lint, typecheck); the
  expected optimization fuses them into one job.
- `share/`: a two-instance matrix whose instances build and lint identically but test
  differently and carry conflicting job env (so they cannot be fused). The lint class is
  shared without a transfer (dedup) and the build class with a transfer; for two instances
  the instance-as-producer encoding beats the original and the new-job encoding does not.
- `groups/`: a node-version x mode matrix; builds and lints are equal only within one node
  version, so every producer serves one read class, and a producer shared across node
  versions is refuted by the checker.
- `setup-chain/`: a two-instance matrix whose `pnpm/action-setup` and `actions/setup-node`
  both write `dep` without reading it; the install reads what both left. Job-local writes
  are updates, so dropping the (shared-class) `pnpm/action-setup` from one instance is
  pruned and refuted, while sharing the build still verifies.
