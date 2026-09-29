Fixtures for tests/test_integration.py.

- `ts-jest/`: kulshekhar/ts-jest at commit 3d7203b09625b369d87ccbc4bce9a68ac355e98a
  (the files lowering and contract inference read: both workflows, both local composite
  actions, `package.json`, and the tsconfig files), copied unmodified from
  `tests/fixtures/lower/ts-jest` and `tests/fixtures/contracts/ts_jest`.
- `frozen-upload/`, `frozen-matrix-shell/`: synthetic workflows for two integration bugs
  (a frozen job's upload must be visible to a downstream download; a frozen matrix job's
  `defaults.run.shell: ${{ matrix.shell }}` must survive emission). The jobs are frozen by
  `continue-on-error: true` (`timeout-minutes` no longer freezes; `frozen-matrix-shell` keeps
  it to check that it is emitted verbatim).
