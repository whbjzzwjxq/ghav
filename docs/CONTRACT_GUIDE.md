# Contract Guide (v1.1)

This guide is the single, repository-independent procedure for giving a step its
region-effect contract. `src/ghav/contracts.py` implements it mechanically. Contracts are never
edited per benchmark; if a benchmark needs a new rule, the rule is added here, applies to every
benchmark, and every benchmark is re-run.

Changes from v1 (2026-09-28): checkout reads and writes every workspace-resident kind and reads
`ext:refs` with `fetch-depth: 0` (section 5); general file-utility rules (4.5); downloads into
`$RUNNER_TEMP` write `tool` (5); the cache-transparency rule (7); observed contracts from
filesystem traces (8); the transfer idiom and the reserved `ghav-` artifact prefix (9).

## 1. Regions a contract may mention

Job-local (instantiated per job instance): `src`, `dep`, `out`, `tool`.
Context: `ctx:<key>` (read-only). Shared: `run:<artifact>`. External: `ext:<object>`.

A write of a job-local kind means the task *changes part of* that region (some paths, some
`PATH` entries); the rest survives. The semantics models such a write as an update
`U(old, F(reads))` whenever the contract does not also read the kind, so a contract lists a
job-local read only when the task's behavior depends on it, never just to keep earlier writers
alive (for example `setup-runtime` reads `src` only, yet `pnpm/action-setup` followed by
`actions/setup-node` leaves both runtimes in `dep` for the install that reads it).

A contract names external services (`ext:codecov`, `ext:npm-registry`, ...). The semantics keys
every external *write* by the originating step occurrence (`ext:codecov@<instance>#<index>`),
so two uploads to one service are two objects, not an overwrite; logs, annotations and job
summaries are not modeled.

## 2. Path-to-region mapping

| Written location | Region |
|---|---|
| Any path tracked by git at checkout, including `.git`; generated files written into tracked paths | `src` (not transferable) |
| `node_modules` anywhere, package-manager stores and caches, runtimes installed by `setup-*`, `PATH` changes | `dep` (not transferable) |
| New untracked files inside the workspace produced by a build (`dist`, `lib`, `build`, `out`, `outDir` from tsconfig) | `out` (transferable; listed in `out_paths`) |
| Tool caches (`.eslintcache`, `*.tsbuildinfo`, `.turbo`, `.nx`), temp files, coverage and test reports, files under `HOME` | `tool` |
| Network services | `ext:<service>` |

A workspace path whose status is unknown (for example `report.json`) covers every kind it may
be (`src`, `out`, `tool`); `.` covers all four job-local kinds.

## 3. Context reads

1. Every step reads `ctx:platform` (the resolved `runs-on`).
2. Every `${{ <expr> }}` inside the step's `run`, `with`, or `env` adds `ctx:<key>` for each
   context key it references (`matrix.*`, `inputs.*`, `github.*`, `env.*`, `vars.*`, `secrets.*`).
3. A job- or workflow-level `env` variable is read by a step only if its name appears in the
   step's resolved command text (`$NAME`, `${NAME}`, `process.env.NAME`, `%NAME%`).
4. Job-identity variables (`GITHUB_JOB`, `GITHUB_RUN_ID`, `RUNNER_NAME`, `GITHUB_ACTION`,
   `github.job`, `strategy.*`) are read only when they appear in the resolved text or in a `${{ }}`;
   then the step reads `ctx:job-identity`, which differs for every instance.
5. An external read whose content is uniquely fixed by an input (lockfile install, action pinned by
   commit SHA) is recorded as a read of that input, not of `ext`.

## 4. Command resolution

1. A `run` script is split into commands on newlines, `&&`, `||`, and `;`. Comment and
   `echo`/`printf`/`cd`/`set -e`-style shell-only commands have no effect (except `cd`, which changes
   the working directory used for path resolution). `exit`, `true` and `false` are shell-only as
   well: they only set the step's status, which the semantics models as the task's success.
2. `npm run X`, `npm X` for lifecycle scripts, `pnpm X`, `pnpm run X`, `yarn X`, `yarn run X`,
   `bun run X`, `nr X` resolve through the root `package.json` `scripts` (including `preX`/`postX`)
   recursively, depth at most 5; cycles or missing scripts give ⊤.
3. The contract of a step is the union of its commands' contracts. Any ⊤ command makes the step ⊤.
4. A local composite action (`uses: ./path`) is one step; its contract is the union of its
   `action.yml` steps' contracts, resolved with the same rules.
5. **File utilities** (template `file-util`). Their writes are the regions their target path
   arguments map to under section 2, and their reads are the regions of their source paths
   (paths are relative to the command's working directory; `$GITHUB_WORKSPACE/...` is the
   workspace root; `~`, `$HOME`, `$RUNNER_TEMP` and `/tmp` are `tool`, or `dep` below a
   `node_modules`). A glob maps by its fixed prefix (`dist/*.js` is `dist`, `*.tgz` is `.`);
   a negated pattern (`!dist/keep`) covers nothing. An unknown or variable path (`$VAR`,
   `${{ }}` other than matrix values, an absolute path outside `/tmp`, a path escaping the
   workspace, an unknown working directory) makes the command ⊤. Written `out` paths are
   the command's `out_paths` (their fixed prefix).

   | Utility | Sources (read) | Targets (written) |
   |---|---|---|
   | `rm`, `rimraf`, `del`/`del-cli`, `shx rm` | – | every path |
   | `mkdir`, `touch`, `shx mkdir`, `shx touch` | `touch -r FILE` | every path |
   | `cp`, `shx cp` | all but the last path (or all with `-t DIR`) | the last path (or `DIR`) |
   | `mv`, `shx mv` | as `cp` | the destination and the sources |
   | `ln` | the targets (not for `-s`) | the link name (or `DIR`) |
   | `sed -i` with explicit files | the files (and `-f SCRIPT`) | the files |
   | `json -I -f FILE` | `FILE` | `FILE` |
   | `tar -c -f A [-C D] M...` | the members, relative to `D` | `A` |
   | `tar -x -f A [-C D] M...` (members required) | `A` | the members, relative to `D` |

   `sed` without `-i` and `json` without `-I -f` are not file utilities (⊤ as unknown
   commands). `tar` without listed members on extraction, or with `-T`/`--files-from`,
   `--transform`, `--strip-components`, `--to-command` and similar options, is ⊤.
   `node <repository script>` and `node -e` stay ⊤ (unless an observed contract exists, section 8).

## 5. Template table

Reads always include `ctx:platform` (rule 3.1). `pinned` means the external read is fixed by an input.

| Template | Matches | Reads | Writes | out_paths |
|---|---|---|---|---|
| checkout | `actions/checkout@*` | `ctx:github.sha`, `ctx:github.ref`, `src`, `dep`, `out`, `tool`; `ext:refs` when `fetch-depth` is `0` or an expression | `src`, `dep`, `out`, `tool` (it resets the workspace) | – |
| setup-runtime | `actions/setup-node@*`, `setup-python`, `setup-go`, `setup-java`, `pnpm/action-setup@*`, `oven-sh/setup-bun@*`, `denoland/setup-deno@*` | `src` (version files) | `dep` | – |
| install-pinned | `npm ci`, `pnpm install` (CI default frozen), `pnpm i`, `yarn install --frozen-lockfile`/`--immutable`, `yarn` (berry in CI), `bun install --frozen-lockfile`, `nci` | `src`, `dep` | `dep`, `tool` | – |
| install-unpinned | `npm install`/`npm i` without lockfile guarantee, `npm i -g`, `pnpm add`, `yarn add`, `npx <pkg>` of an unpinned package | `src`, `dep`, `ext:registry` | `dep`, `tool` | – |
| build | `tsc` (without `--noEmit`), `tsc -b`, `vite build`, `rollup`, `tsup`, `esbuild`, `webpack`, `unbuild`, `babel` with `-d/--out-dir` | `src`, `dep`, `out` | `out`, `tool` | tsconfig `outDir` if set, else `dist` |
| build-orchestrated | `turbo run`, `nx run`/`nx affected`, `lerna run`, `pnpm -r`/`--filter` running builds | `src`, `dep`, `out` | `out`, `tool`, `src` | – (not transferable) |
| typecheck | `tsc --noEmit`, `vue-tsc --noEmit`, `tsc -p … --noEmit` | `src`, `dep`, `out` | `tool` | – |
| lint | `eslint`, `prettier --check`, `biome check`/`lint`/`ci`, `oxlint`, `stylelint`, `xo`, `oxfmt --check`, `madge`, `publint`, `knip` | `src`, `dep`, `out` | `tool` if `--cache` present, else nothing | – |
| format-write | `prettier --write`, `biome format --write`, `eslint --fix` | `src`, `dep` | `src`, `tool` | – |
| test | `jest`, `vitest`, `mocha`, `ava`, `node --test`, `playwright test`, `cypress run` | `src`, `dep`, `out` | `tool` | – |
| upload-artifact | `actions/upload-artifact@*` | kinds covering `with.path` | `run:<with.name>` | – |
| download-artifact | `actions/download-artifact@*` | `run:<with.name>` | `out`; `tool` when `with.path` is outside the workspace (e.g. `${{ runner.temp }}`) | `with.path` (when `out`) |
| cache | `actions/cache@*`, `actions/cache/restore@*`, `actions/cache/save@*` | – | – (cold-cache semantics) | – |
| cache-opaque | a cache restore that breaks the cache-transparency rule (section 7) | its reads, `src`, `dep`, `out`, `tool`, the context values of its inputs | its writes, `dep`, `tool`, the kinds of its cached paths | – |
| report-upload | `codecov/codecov-action@*`, `coverallsapp/github-action@*`, `SonarSource/*`, `actions/upload-pages-artifact@*` | `src`, `tool`, `out`, `ctx:github.sha` | `ext:<service>` | – |
| publish | `npm publish`, `pnpm publish`, `changeset publish`, `gh release`, `semantic-release`, deploy actions | `src`, `dep`, `out`, `ctx:github.ref` | `ext:<service>` | – |
| noop | shell-only commands (rule 4.1), `actions/github-script` that only logs | – | – | – |
| file-util | file utilities (rule 4.5) | regions of the source paths | regions of the target paths | written `out` prefixes |
| observed | a ⊤ step with an observed contract (section 8) | `src`, `dep`, `out`, `tool`, its context reads (rules 3.2-3.4) | regions of the observed paths, listed external objects | written `out` prefixes |

Anything else (unknown third-party actions, unknown commands, `docker`, `curl`/`wget` without a
pinned digest, scripts that start background processes with `&`) gives ⊤.

## 6. Auditing

`ghav contracts <workflow>` prints each step's template and contract, including the steps of
frozen jobs (with `frozen: true`: a frozen job is one opaque task whose effect is the union of
its steps' contracts). Aggregates report the share of steps per template, the share of ⊤ steps
(overall and among unfrozen jobs), and the number of observed contracts and opaque caches, so
the coverage of this guide is visible.

## 7. Cache transparency

Correctness uses cold-cache semantics (a restore is a miss). That is sound only if a hit and a
miss lead to the same state, which fails when a later task changes what the job saves (the saved
snapshot is taken at job end). The checkable part of this hypothesis:

> After a cache-restoring task, no later task in the same job writes `dep`.

Cache-restoring tasks: `actions/cache@*` and `actions/cache/restore@*`; `actions/setup-node`,
`setup-python`, `setup-java` and `denoland/setup-deno` with a `cache` input other than `false`;
`actions/setup-go` unless `cache: false` (its default from v4 on, and for a version given as a
commit); and a local composite action containing any of these. A later task *writes `dep`* when
it is ⊤ or when a command other than a pinned install writes `dep` (a pinned install's `dep` is
fixed by the lockfile, so it is the same after a hit and after a miss).

When the rule fails, the restore is reported (`cache-opaque` finding, never blocking) and treated
as opaque: template `cache-opaque` in place of `cache`, reading every job-local kind and the
context values of its inputs (the cache key, including `hashFiles`), writing `dep`, `tool` and the
kinds of its cached paths (every job-local kind for a composite action), with uninterpreted
outputs (`cache-hit`).

## 8. Observed contracts

A step that the guide gives ⊤ gets an *observed* contract when a trace of a local run lists its
binding. Trace files are committed at

    resources/traces/<owner>__<repo>/<sha>/<benchmark-id>.json

(benchmark id `<owner>__<repo>__<workflow-file-stem>`). Any benchmark may have one; v1 has traces
only for development-set examples, and evaluation runs use the static guide, so no benchmark is
treated specially. The observed contract writes the regions of every path the trace saw
created, modified or deleted (mapped through section 2 and rule 4.5; an unmappable path leaves
the step ⊤) plus the listed external objects, and reads conservatively: every job-local kind,
`ctx:platform`, the step's `${{ }}` context references and the env names in its resolved text
(rules 3.2-3.4). A round trip re-annotates the emitted workflow with the original benchmark's
trace.

Trace file format (JSON, `format` must be `ghav-trace/1`):

```json
{
  "format": "ghav-trace/1",
  "benchmark": "pmndrs__zustand__test-multiple-builds",
  "repository": "pmndrs/zustand",
  "sha": "b57db4f86ef179285da216eeb291266da82c361c",
  "provenance": "how the trace was taken (machine, tool versions, command)",
  "steps": [
    {
      "binding": {"kind": "run", "text": "pnpm run build", "with": {}, "env": {},
                  "working_directory": null, "shell": null},
      "writes": ["dist/index.js", "dist/esm/index.mjs"],
      "deletes": [],
      "external": []
    }
  ]
}
```

`binding` is the lowered binding (after matrix substitution), matched on its key: `kind`,
`text`, `with`, `env`, `working_directory`, `shell` (step names and ids do not matter).
`writes` and `deletes` are workspace-relative paths (or `~/`, `$HOME/`, `$RUNNER_TEMP/` paths);
a path ending in `/` stands for a whole directory that was created or deleted (for example
`node_modules/`). Both count as writes. `external` names external objects the step writes (a
filesystem diff cannot see them, so the trace author states them; traces produced by
`scripts/traces/trace_workflow.py` leave it empty). Several entries for one binding are
united. Other fields are provenance and are ignored by contract inference: per-step
`instance` and `index` (where the step was observed), and a top-level `instances` object with
per-instance detail (tool versions, platform, local wall-clock seconds, skipped steps). Only
steps that ran and exited 0 are listed in `steps`. A malformed trace file is an error, never
silently ignored.

An observed contract replaces only a step the guide gives ⊤; a step with a template keeps it
even when its trace shows more writes (the Zustand trace shows `vitest` writing
`node_modules/.vite`, i.e. `dep`, which the `test` template does not list). Validating
templates against traces is v2 work (RESEARCH_SCOPE).

## 9. The transfer idiom and the reserved artifact prefix

GHAV moves build output between jobs with one idiom only (`ghav.emit.transfer_steps`):

1. producer: `tar -cf "$RUNNER_TEMP/<name>.tar" -C "$GITHUB_WORKSPACE" <paths>` (shell `bash`),
   immediately followed by `actions/upload-artifact@v4` with `name: <name>`,
   `path: ${{ runner.temp }}/<name>.tar`, `if-no-files-found: error`, `retention-days: 1`;
2. consumer: `actions/download-artifact@v4` with `name: <name>`, `path: ${{ runner.temp }}`,
   immediately followed by `tar -xf "$RUNNER_TEMP/<name>.tar" -C "$GITHUB_WORKSPACE" <paths>`.

Under this guide these steps get `file-util`, `upload-artifact` and `download-artifact`
contracts (pack: reads `out`, writes `tool`; unpack: reads `tool`, writes `out`). The semantics
treats the idiom as an exact copy of `out` when the packed paths cover everything written into
`out` in the producer since its last clean checkout and the consumer's `out` is still empty; a
plain upload/download pair is uninterpreted (it rebases paths, drops hidden files and resets
permissions).

Artifact names starting with `ghav-` are reserved for this idiom: an upload or download of a
`ghav-` name that is not exactly an idiom step is ⊤, and transfer names never enter the
persistent regions.
