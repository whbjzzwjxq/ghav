# Cost Model

Metric: expected machine time per run = Σ over executed job instances of
(job overhead + Σ executed step seconds), averaged over contexts weighted by their frequency.
Conditions are evaluated concretely for each event assuming every task succeeds
(`ghav.cost.simulate`).

Every cost report (`ghav.cost.cost_report`) gives two numbers side by side, per job, per event,
and in expectation:

- **machine seconds**: the raw sum above;
- **billed minutes**: Σ over executed jobs of `max(1, ceil(seconds / 60))`, i.e. each executed
  job rounded up to a whole minute as GitHub bills it (seconds are rounded to 6 decimals first
  so float noise never adds a minute; jobs that do not run bill 0).

Which of the two is the primary metric is pending the user's choice (RESEARCH_SCOPE,
"Provisional decisions"). They can disagree: removing seconds from a job that stays under a
minute saves machine seconds but no billed minutes, and splitting work into more jobs can add
billed minutes without adding seconds.

There are two backends with one interface (`ghav.cost.CostModel`); the mock model is the
default.

## Mock durations

Fixed before any experiment and not tuned. Relative results (reduction percentages) are
reported, and every table produced with this model is labeled "mock cost". Event weights are
0.5/0.5 for `push` and `pull_request`.

| Template | Seconds |
|---|---|
| job overhead (provisioning, set-up and complete-job) | 15 |
| checkout | 5 |
| setup-runtime | 10 |
| install-pinned | 45 |
| install-unpinned | 45 |
| build | 60 |
| build-orchestrated | 90 |
| typecheck | 40 |
| lint | 30 |
| format-write | 20 |
| test | 120 |
| upload-artifact | 10 |
| download-artifact | 8 |
| cache | 5 |
| report-upload | 10 |
| publish | 30 |
| noop | 1 |
| file-util | 1 |
| cache-opaque | 5 |
| observed | 30 |
| top (unknown) | 30 |

The `file-util`, `cache-opaque` and `observed` rows were added with contract guide v1.1
(2026-09-28), which introduced those templates; before, they fell to the ⊤ row. No new
duration was chosen: each copies the row of what it refines. A file utility (`rm`, `cp`,
`tar`, ...) is a shell-level command like `noop`; an opaque cache restore is still a cache
step; an observed contract describes the effects of a step the guide leaves at ⊤, not its
duration. This matters for sharing: the transfer idiom's pack and unpack steps are
`file-util`, so a transfer costs 1 + 10 (upload) + 8 (download) + 1 = 20 mock seconds.

Composite templates (`a+b`) cost the sum; unknown templates cost the ⊤ row. Transfer steps
inserted by synthesis are priced by their contract templates (`file-util`, `upload-artifact`,
`download-artifact`; an official artifact action without a known template falls back to its
action reference). A frozen job costs the job overhead plus 30 s (⊤) per step.

## Historical durations

### Fetching (`scripts/resources/fetch_timings.py`; network; results committed)

For each benchmark workflow in a scope, through `gh api` (`scripts/resources/ghlib.py`):

1. Runs: the last 5 runs with conclusion `success` per event (`push`, `pull_request`) created
   before `2026-09-01T00:00:00Z`
   (`GET repos/<o>/<r>/actions/workflows/<file>/runs?event=E&status=success&created=<cutoff`).
2. The jobs of each run (latest attempt, `filter=latest`) with their step timestamps.
3. Event counts: the number of runs per event (any conclusion) created in the 30 days before
   the cutoff. The API caps `total_count` for filtered queries, so if either count reaches 1000
   the 7-day and then the 1-day window is used; the window used is recorded (`count_window`).
4. Workflow versions: the blob oid of the workflow file at each run's head commit, compared with
   the pinned commit's (`workflow_versions.runs_same_version`). Runs of other versions are still
   used; name matching makes them comparable step by step.

Cache layout, `resources/timings/<benchmark-id>/`: `runs_<event>.json`, `counts.json`,
`jobs/<run-id>.json`, `versions.json` (field-projected raw responses: URLs and nested
repository/actor objects are dropped; every field the derivation reads is kept), and the derived
`medians.json`. Cached responses are reused; `--offline` re-derives `medians.json` from the
cache, and `--coverage` lowers each workflow and writes the match coverage to
`resources/timings/coverage.<scope>.json`.

### Derivation (`medians.json`)

Per GitHub job name, over the successful jobs of the fetched runs (runs in increasing id order):

- GitHub's steps are split into *workflow steps* and *infrastructure* (`Set up job`,
  `Complete job`, container set-up). `Pre X`/`Post X` hooks are attributed to workflow step
  `X` (pre hooks pair in order, post hooks in reverse order, as the runner executes them).
- A step sample is its duration plus its hooks' durations, taken only when the step executed
  (conclusion success, failure or neutral); skipped steps are counted, not sampled.
- Job overhead sample = job duration − Σ durations of workflow steps and their hooks, i.e.
  `Set up job`, `Complete job`, and the gaps between steps.
- Steps are identified by (name, occurrence of that name in the job) and carry the positions
  (0-based index among workflow steps) at which they were observed, plus the number of workflow
  steps per job (`main_step_counts`).

Timestamps have 1-second resolution, so a step shorter than a second samples as 0.

### Matching (`ghav.cost.historical_model(program, medians)`)

Matching is against the ORIGINAL lowered program and is repository-independent:

1. Jobs: a GitHub job name matches `JobInstance.check_name` or, for a default matrix suffix
   `job (v1, v2, …)`, GitHub's rendering of it (JSON array/object values flattened to their
   values, empty values omitted). Names are whitespace-normalized. An exact match wins;
   otherwise an unevaluated `${{ … }}` on either side matches any text, and a GitHub name of at
   least 40 characters ending in `...` (GitHub's truncation) matches the names it is a prefix
   of. A non-exact match must be unique in both directions (one GitHub job for the instance, one
   instance for the GitHub job); otherwise the instance is unmatched.
2. Steps: the expected GitHub name of a WIR step is its `name:`, else `Run <first non-empty
   line of run>`, else `Run <uses ref>`. Names are aligned greedily and monotonically in step
   order. An unmatched step then falls back to its position, only if every fetched run of that
   job had exactly as many workflow steps as the WIR job and the GitHub step at that position
   is not otherwise matched.
3. A matched step with at least one sample costs the median of its samples (table keyed by
   original instance id and step index). A step whose own instance has no sample uses the
   median pooled over every matched instance of the same original job and step index.
   Anything else, including transfer steps inserted by synthesis, costs its mock value.
4. Job overhead: the median of the overhead samples pooled over matched jobs with the same
   `runs_on` label; for other labels, the median over all matched jobs; with no match at all,
   the mock 15 s. New jobs created by synthesis get the overhead of their `runs_on` label.
5. Frozen jobs (kept verbatim) matched to a GitHub job cost that job's median duration;
   otherwise they cost the job overhead plus 30 s per step.
6. Event weights: the event counts normalized over `push` and `pull_request`; 0.5/0.5 when no
   count is available (for example, no runs in any window).

Candidate programs produced by synthesis are priced through `Step.origin` and
`Step.origin_instance` (falling back to the containing job's id), so the same tables price the
original program and every candidate.

The model reports its coverage (`CostModel.coverage`, copied into `cost_report`): jobs matched,
GitHub job names left unmatched, steps matched by name and by position, and
`matched_fraction` = step occurrences priced from history / all step occurrences. Per event,
the report also counts the executed steps priced from history.

Tables produced with this model are labeled "historical cost (medians of the last 5 successful
runs per event before 2026-09-01)" and state the matched fraction.

### Known limitations of the historical backend

- Conditions are still evaluated with cold-cache semantics, so a step that GitHub skipped on
  every fetched run because of a cache hit (no sample) is charged its mock value.
- Samples come from the last 5 successful runs per event, possibly of other workflow versions
  (`workflow_versions`); instances or steps that did not exist in those runs fall back to mock.
- A frozen remote reusable-workflow call is one WIR instance but several GitHub jobs; it is
  matched only when a GitHub job carries exactly its check name.
- Step durations have 1-second resolution and include queueing inside the job only; runner
  provisioning before the job starts is not billed and not counted.
