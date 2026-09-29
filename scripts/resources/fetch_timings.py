#!/usr/bin/env python3
"""Fetch historical GitHub Actions timings for every workflow in one or more scope files.

For each benchmark line `owner/repo@sha:.github/workflows/<file>` this writes, under
resources/timings/<benchmark-id>/:

  - runs_<event>.json   the last RUNS_PER_EVENT successful runs of the workflow per event
                        (push, pull_request) created before the cutoff (2026-09-01T00:00:00Z),
  - counts.json         run counts per event in a window before the cutoff (event weights),
  - jobs/<run-id>.json  the jobs of each run (latest attempt) with their step timings,
  - versions.json       the workflow file's blob oid at each run's head commit vs the pinned one,
  - medians.json        derived: per GitHub job name, per step, the durations and medians that
                        `ghav.cost.historical_model` matches against a lowered program.

Raw responses are cached field-projected (URLs and nested repository/actor objects dropped; every
field the derivation uses is kept, see RUN_KEYS/JOB_KEYS/STEP_KEYS). Cached files are reused, so
the script resumes after interruptions and later runs are offline; `medians.json` is always
re-derived from the cache. Network access goes through `ghlib.Gh` (retries, pacing, rate limits).

Usage:
  uv run python scripts/resources/fetch_timings.py --scope scope/dev.scope [--limit N] [--force]
  uv run python scripts/resources/fetch_timings.py --scope scope/dev.scope --offline --coverage

`--offline` only re-derives medians from the cache; `--coverage` lowers every workflow and prints
(and writes to resources/timings/coverage.<scope>.json) the historical model's match coverage.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "scope")]

from ghlib import Gh, GhError, NotFound, gql_str, log  # noqa: E402
from scopelib import (  # noqa: E402
    CUTOFF,
    EVENTS,
    REPOS_DIR,
    RESOURCES,
    ScopeLine,
    load_manifest,
    read_scope,
    snapshot_dir,
    write_json,
)

TIMINGS_DIR = RESOURCES / "timings"
RUNS_PER_EVENT = 5
# Event weights: runs per event created in the COUNT_WINDOWS_DAYS[i] days before the cutoff.
# The API caps `total_count` for filtered queries, so when either count reaches COUNT_CAP_GUARD
# the next (narrower) window is used; the window actually used is recorded.
COUNT_WINDOWS_DAYS = (30, 7, 1)
COUNT_CAP_GUARD = 1000

RUN_KEYS = ("conclusion", "created_at", "event", "head_branch", "head_sha", "id", "name", "path",
            "run_attempt", "run_number", "run_started_at", "status", "updated_at", "workflow_id")
JOB_KEYS = ("completed_at", "conclusion", "created_at", "head_sha", "id", "labels", "name",
            "run_attempt", "run_id", "runner_group_name", "runner_name", "started_at", "status",
            "workflow_name")
STEP_KEYS = ("completed_at", "conclusion", "name", "number", "started_at", "status")

# Runner-generated steps that belong to the job, not to any workflow step.
INFRA_STEPS = frozenset({"Set up job", "Complete job", "Initialize containers", "Stop containers"})
EXECUTED = frozenset({"success", "failure", "neutral"})


# ---------------------------------------------------------------------------
# Pure helpers (tested offline)
# ---------------------------------------------------------------------------


def _project(obj: Mapping[str, Any], keys: Iterable[str]) -> dict:
    return {k: obj.get(k) for k in keys}


def project_run(run: Mapping[str, Any]) -> dict:
    return _project(run, RUN_KEYS)


def project_job(job: Mapping[str, Any]) -> dict:
    out = _project(job, JOB_KEYS)
    steps = [_project(s, STEP_KEYS) for s in job.get("steps") or [] if isinstance(s, Mapping)]
    out["steps"] = sorted(steps, key=lambda s: (s.get("number") or 0, s.get("name") or ""))
    return out


def parse_time(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def seconds_between(a: Any, b: Any) -> float | None:
    ta, tb = parse_time(a), parse_time(b)
    if ta is None or tb is None:
        return None
    return max(0.0, (tb - ta).total_seconds())


def classify_steps(steps: list[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Split a job's GitHub steps into workflow steps and infrastructure steps.

    Returns (main, infra). Each main entry is {"step": <github step>, "hooks": [<pre/post
    steps>]}. GitHub names the hooks of an action step "Pre <name>" and "Post <name>"; pre hooks
    run in the order of their main steps and post hooks in reverse order, which is how
    same-named steps are paired."""
    ordered = sorted(steps, key=lambda s: (s.get("number") or 0, s.get("name") or ""))
    infra = [s for s in ordered if s.get("name") in INFRA_STEPS]
    rest = [s for s in ordered if s.get("name") not in INFRA_STEPS]
    names = {s.get("name") for s in rest}
    pre: list[dict] = []
    post: list[dict] = []
    mains: list[dict] = []
    for s in rest:
        name = s.get("name") or ""
        if name.startswith("Post ") and name[5:] in names and name[5:] != name:
            post.append(s)
        elif name.startswith("Pre ") and name[4:] in names and name[4:] != name:
            pre.append(s)
        else:
            mains.append({"step": s, "hooks": []})
    for hooks, forward, cut in ((pre, True, 4), (post, False, 5)):
        by_name: dict[str, list[dict]] = {}
        for m in mains:
            by_name.setdefault(m["step"].get("name") or "", []).append(m)
        for n in sorted(by_name):
            if not forward:
                by_name[n] = list(reversed(by_name[n]))
        taken: dict[str, int] = {}
        for h in hooks:
            target = (h.get("name") or "")[cut:]
            k = taken.get(target, 0)
            cands = by_name.get(target, [])
            if k < len(cands):
                cands[k]["hooks"].append(h)
                taken[target] = k + 1
            else:
                infra.append(h)  # an unpaired hook still costs time; count it as overhead
    return mains, infra


def job_sample(job: Mapping[str, Any]) -> dict | None:
    """One successful job's measurements, or None when the job did not succeed."""
    if job.get("conclusion") != "success":
        return None
    duration = seconds_between(job.get("started_at"), job.get("completed_at"))
    if duration is None:
        return None
    mains, _infra = classify_steps(list(job.get("steps") or []))
    attributed = 0.0
    steps: list[dict] = []
    seen: dict[str, int] = {}
    for pos, m in enumerate(mains):
        s = m["step"]
        name = s.get("name") or ""
        occ = seen.get(name, 0)
        seen[name] = occ + 1
        secs = 0.0
        valid = True
        for part in [s, *m["hooks"]]:
            d = seconds_between(part.get("started_at"), part.get("completed_at"))
            if d is None:
                valid = valid and part is not s
                continue
            secs += d
        attributed += secs
        executed = s.get("conclusion") in EXECUTED and valid
        steps.append({"name": name, "occurrence": occ, "position": pos,
                      "executed": executed, "seconds": secs if executed else None})
    return {"name": job.get("name") or "", "duration": duration,
            "overhead": max(0.0, duration - attributed), "steps": steps,
            "main_steps": len(mains)}


def _median(xs: list[float]) -> float | None:
    return float(statistics.median(sorted(xs))) if xs else None


def summarize_jobs(jobs_by_run: Mapping[int, list[Mapping[str, Any]]]) -> dict:
    """Per GitHub job name: durations, overheads, and per-step durations over all runs.

    Runs are visited in increasing run id and jobs in (name, id) order, so sample lists are
    deterministic. Step entries are keyed by (name, occurrence) and listed in order of their
    first observed position."""
    jobs: dict[str, dict] = {}
    for run_id in sorted(jobs_by_run):
        for job in sorted(jobs_by_run[run_id], key=lambda j: (j.get("name") or "", j.get("id") or 0)):
            name = job.get("name") or ""
            entry = jobs.setdefault(name, {"samples": 0, "skipped": 0, "durations": [], "overheads": [],
                                           "main_step_counts": set(), "steps": {}})
            sample = job_sample(job)
            if sample is None:
                entry["skipped"] += 1
                continue
            entry["samples"] += 1
            entry["durations"].append(sample["duration"])
            entry["overheads"].append(sample["overhead"])
            entry["main_step_counts"].add(sample["main_steps"])
            for st in sample["steps"]:
                key = (st["name"], st["occurrence"])
                se = entry["steps"].setdefault(key, {"name": st["name"], "occurrence": st["occurrence"],
                                                     "positions": set(), "durations": [], "skipped": 0})
                se["positions"].add(st["position"])
                if st["executed"]:
                    se["durations"].append(st["seconds"])
                else:
                    se["skipped"] += 1
    out: dict[str, dict] = {}
    for name in sorted(jobs):
        e = jobs[name]
        steps = sorted(e["steps"].values(), key=lambda s: (min(s["positions"]), s["name"], s["occurrence"]))
        out[name] = {
            "samples": e["samples"],
            "skipped": e["skipped"],
            "durations": e["durations"],
            "overheads": e["overheads"],
            "duration_seconds": _median(e["durations"]),
            "overhead_seconds": _median(e["overheads"]),
            "main_step_counts": sorted(e["main_step_counts"]),
            "steps": [{"name": s["name"], "occurrence": s["occurrence"],
                       "positions": sorted(s["positions"]), "durations": s["durations"],
                       "seconds": _median(s["durations"]), "skipped": s["skipped"]} for s in steps],
        }
    return out


def summarize(line: ScopeLine, runs: Mapping[str, Mapping[str, Any] | None], counts: Mapping[str, Any] | None,
              jobs_by_run: Mapping[int, list[Mapping[str, Any]]], versions: Mapping[str, Any] | None) -> dict:
    """The derived medians file for one benchmark (see docs/COST_MODEL.md)."""
    events: dict[str, dict] = {}
    run_ids: list[int] = []
    not_found = False
    for ev in EVENTS:
        r = runs.get(ev)
        if r is None or r.get("status") == "workflow-not-found":
            if r is not None:
                not_found = True
            events[ev] = {"runs": [], "window_count": None}
            continue
        ids = [int(x["id"]) for x in r.get("workflow_runs") or []]
        run_ids.extend(ids)
        wc = ((counts or {}).get("counts") or {}).get(ev)
        events[ev] = {"runs": ids, "window_count": wc}
    used = {rid: jobs_by_run[rid] for rid in sorted(set(run_ids)) if rid in jobs_by_run}
    if not_found:
        status = "workflow-not-found"
    elif not run_ids:
        status = "no-runs"
    else:
        status = "ok"
    return {
        "benchmark": str(line),
        "benchmark_id": line.benchmark_id(),
        "cutoff": CUTOFF,
        "runs_per_event": RUNS_PER_EVENT,
        "status": status,
        "events": events,
        "count_window": {k: v for k, v in (counts or {}).items() if k != "counts"} or None,
        "workflow_versions": versions,
        "jobs": summarize_jobs(used),
    }


def count_windows(cutoff: str = CUTOFF) -> list[tuple[int, str, str]]:
    """(days, start, end) for each count window; `end` is one second before the cutoff."""
    c = parse_time(cutoff)
    assert c is not None
    end = (c - timedelta(seconds=1)).astimezone(timezone.utc)
    out = []
    for days in COUNT_WINDOWS_DAYS:
        start = (c - timedelta(days=days)).astimezone(timezone.utc)
        out.append((days, start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")))
    return out


# ---------------------------------------------------------------------------
# Network (gh api) with a file cache
# ---------------------------------------------------------------------------


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def runs_endpoint(line: ScopeLine) -> str:
    return f"repos/{line.repo}/actions/workflows/{posixpath.basename(line.path or '')}/runs"


class TimingsFetcher:
    def __init__(self, gh: Gh | None, line: ScopeLine, timings_dir: Path = TIMINGS_DIR,
                 repos_dir: Path = REPOS_DIR, force: bool = False):
        if line.sha is None or line.path is None:
            raise ValueError(f"not a pinned workflow line: {line}")
        self.gh, self.line, self.force = gh, line, force
        self.dir = timings_dir / line.benchmark_id()
        self.repos_dir = repos_dir

    def _cached(self, path: Path) -> bool:
        return path.exists() and not self.force

    def _need_gh(self) -> Gh:
        if self.gh is None:
            raise GhError(f"{self.line}: cache miss in offline mode")
        return self.gh

    def runs(self, event: str) -> dict:
        path = self.dir / f"runs_{event}.json"
        if self._cached(path):
            return _load(path)
        params = {"event": event, "status": "success", "created": f"<{CUTOFF}",
                  "per_page": RUNS_PER_EVENT, "exclude_pull_requests": "true"}
        try:
            resp = self._need_gh().rest(runs_endpoint(self.line), params)
            out = {"query": params, "status": "ok", "total_count": resp.get("total_count"),
                   "workflow_runs": [project_run(r) for r in (resp.get("workflow_runs") or [])][:RUNS_PER_EVENT]}
        except NotFound:
            out = {"query": params, "status": "workflow-not-found", "total_count": 0, "workflow_runs": []}
        write_json(path, out)
        return out

    def counts(self) -> dict:
        path = self.dir / "counts.json"
        if self._cached(path):
            return _load(path)
        gh = self._need_gh()
        out: dict[str, Any] = {}
        windows = count_windows()
        for i, (days, start, end) in enumerate(windows):
            got: dict[str, int] = {}
            try:
                for ev in EVENTS:
                    resp = gh.rest(runs_endpoint(self.line), {"event": ev, "created": f"{start}..{end}", "per_page": 1})
                    got[ev] = int(resp.get("total_count") or 0)
            except NotFound:
                got = {ev: 0 for ev in EVENTS}
            capped = max(got.values()) >= COUNT_CAP_GUARD
            out = {"days": days, "start": start, "end": end, "counts": got, "capped": capped}
            if not capped or i == len(windows) - 1:
                break
        write_json(path, out)
        return out

    def jobs(self, run_id: int) -> list[dict]:
        path = self.dir / "jobs" / f"{run_id}.json"
        if self._cached(path):
            return _load(path)["jobs"]
        pages = self._need_gh().rest_paginate(f"repos/{self.line.repo}/actions/runs/{run_id}/jobs?filter=latest")
        jobs = [project_job(j) for page in pages if isinstance(page, dict) for j in page.get("jobs") or []]
        jobs.sort(key=lambda j: (j.get("id") or 0))
        write_json(path, {"run_id": run_id, "jobs": jobs})
        return jobs

    def versions(self, runs: Mapping[str, dict]) -> dict:
        """Whether each run executed the pinned version of the workflow file (blob oid)."""
        path = self.dir / "versions.json"
        heads = sorted({r["head_sha"] for ev in EVENTS for r in (runs.get(ev) or {}).get("workflow_runs") or []
                        if r.get("head_sha")})
        manifest = load_manifest(self.line.repo, self.line.sha, self.repos_dir) or {}
        pinned = ((manifest.get("files") or {}).get(self.line.path) or {}).get("oid")
        if self._cached(path):
            cached = _load(path)
            if sorted(cached.get("oids") or {}) == heads:
                return cached
        oids: dict[str, str | None] = {}
        if heads:
            owner, name = self.line.repo.split("/", 1)
            fields = " ".join(f"v{i}: object(expression: {gql_str(f'{h}:{self.line.path}')}) {{ ... on Blob {{ oid }} }}"
                              for i, h in enumerate(heads))
            q = ("query { rateLimit { cost remaining resetAt } "
                 f"repository(owner: {gql_str(owner)}, name: {gql_str(name)}) {{ {fields} }} }}")
            data = (self._need_gh().graphql(q).get("data") or {}).get("repository") or {}
            for i, h in enumerate(heads):
                obj = data.get(f"v{i}")
                oids[h] = obj.get("oid") if isinstance(obj, dict) else None
        out = {"pinned_oid": pinned, "oids": oids,
               "runs_same_version": sum(1 for h in heads if pinned and oids.get(h) == pinned),
               "runs_total": len(heads)}
        write_json(path, out)
        return out

    def fetch(self) -> dict:
        runs = {ev: self.runs(ev) for ev in EVENTS}
        found = any(r.get("status") == "ok" for r in runs.values())
        counts = self.counts() if found else None
        ids = sorted({int(r["id"]) for ev in EVENTS for r in runs[ev].get("workflow_runs") or []})
        jobs_by_run = {rid: self.jobs(rid) for rid in ids}
        versions = self.versions(runs) if ids else None
        medians = summarize(self.line, runs, counts, jobs_by_run, versions)
        write_json(self.dir / "medians.json", medians)
        return medians


# ---------------------------------------------------------------------------
# Coverage (offline): lower each workflow and match it against its medians file
# ---------------------------------------------------------------------------


def coverage_for(line: ScopeLine, timings_dir: Path = TIMINGS_DIR, repos_dir: Path = REPOS_DIR) -> dict:
    from ghav import cost
    from ghav.contracts import annotate
    from ghav.lower import lower

    root = snapshot_dir(line.repo, line.sha or "", repos_dir)
    try:
        program = annotate(lower(root / (line.path or ""), root), root)
    except Exception as e:  # noqa: BLE001 - reported, not fatal
        return {"error": f"{type(e).__name__}: {e}"}
    model = cost.load_historical_model(program, line.benchmark_id(), timings_dir)
    cov = dict(model.coverage or {})
    report = cost.cost_report(program, None, model)
    mock = cost.cost_report(program, None, cost.MOCK)
    cov["expected_seconds_historical"] = report["expected_seconds"]
    cov["expected_billed_minutes_historical"] = report["expected_billed_minutes"]
    cov["expected_seconds_mock"] = mock["expected_seconds"]
    cov["expected_billed_minutes_mock"] = mock["expected_billed_minutes"]
    ex = sum(report["events"][e]["executed_steps"] * w for e, w in report["contexts"])
    exh = sum(report["events"][e]["executed_steps_historical"] * w for e, w in report["contexts"])
    cov["executed_matched_fraction"] = (exh / ex) if ex else 0.0
    medians = cost.timings_path(line.benchmark_id(), timings_dir)
    cov["status"] = cost.load_timings(medians).get("status") if medians.exists() else "missing"
    return cov


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scope", type=Path, action="append", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="refetch cached responses")
    ap.add_argument("--offline", action="store_true", help="only re-derive medians from the cache")
    ap.add_argument("--coverage", action="store_true", help="lower each workflow and report match coverage")
    ap.add_argument("--timings-dir", type=Path, default=TIMINGS_DIR)
    ap.add_argument("--repos-dir", type=Path, default=REPOS_DIR)
    args = ap.parse_args()

    lines: list[ScopeLine] = []
    for sc in args.scope:
        lines.extend(ln for ln in read_scope(sc) if ln.sha and ln.path)
    lines = sorted(set(lines))
    if args.limit is not None:
        lines = lines[: args.limit]
    gh = None if args.offline else Gh()
    failures: dict[str, str] = {}
    for i, ln in enumerate(lines, 1):
        try:
            m = TimingsFetcher(gh, ln, args.timings_dir, args.repos_dir, args.force).fetch()
            log(f"[{i}/{len(lines)}] {ln.benchmark_id()}: {m['status']}, "
                f"{sum(len(m['events'][e]['runs']) for e in EVENTS)} runs, {len(m['jobs'])} job names")
        except GhError as e:
            failures[str(ln)] = str(e)
            log(f"[{i}/{len(lines)}] {ln.benchmark_id()}: FAILED {e}")
    if args.coverage:
        cov = {ln.benchmark_id(): coverage_for(ln, args.timings_dir, args.repos_dir) for ln in lines}
        name = "+".join(sorted(sc.stem for sc in args.scope))
        write_json(args.timings_dir / f"coverage.{name}.json", cov)
        for bid in sorted(cov):
            c = cov[bid]
            if "error" in c:
                print(f"{bid}\tERROR {c['error']}")
                continue
            print(f"{bid}\t{c['status']}\tjobs {c['jobs_matched']}/{c['jobs_total']}\t"
                  f"steps {c['steps_priced_historical']}/{c['steps_total']} ({c['matched_fraction']:.2f})\t"
                  f"exec {c['executed_matched_fraction']:.2f}\t"
                  f"w(push)={c['event_weights'].get('push', 0):.2f} [{c['event_weights_source']}]\t"
                  f"hist {c['expected_seconds_historical']:.1f}s/{c['expected_billed_minutes_historical']:.2f}min\t"
                  f"mock {c['expected_seconds_mock']:.1f}s/{c['expected_billed_minutes_mock']:.2f}min")
    if failures:
        for k in sorted(failures):
            log(f"failed: {k}: {failures[k]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
