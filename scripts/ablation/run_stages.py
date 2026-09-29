#!/usr/bin/env python3
"""Run the per-benchmark stages (lower, contracts, synthesis, round trip) over a scope.

For every workflow line of ``--scope`` and every variant of ``--variants`` this calls
``ghav.pipeline.run_benchmark(line, variant, out_dir, cost_backend=COST,
max_expansions=MAX_EXPANSIONS)`` (docs/EXPERIMENTS.md, "Stages per workflow") with the static
contract guide (``use_traces=False``, every benchmark alike). Stage files go to

    output/stages/<scope-stem>.<cost>/<benchmark-id>/

(one directory per scope and cost backend, since the stage file names only carry the variant).
Besides the pipeline's files, each benchmark directory gets

* ``cost.json``: the cost report of the ORIGINAL program under the backend (expected machine
  seconds and billed minutes, event weights, and for the historical backend the match coverage);
* ``redundancy.json``: the redundant seconds (``synth.redundancy``'s definition, priced by the
  backend) split into ⊤ steps, transferable steps (write ``out`` with declared paths) and
  non-transferable steps (checkout, set-up, install, ...), and by contract template;
* ``consistency.json``: job instances and steps that the cost model's simulation executes (all
  tasks succeeding) but that the symbolic semantics skips in every context of the same event
  (``consistency_json``); a non-empty list flags a modeling error that can make verdicts
  vacuous;
* ``error_<variant>.json``: only when ``run_benchmark`` raised (type, message, traceback).

The stage directory gets ``manifest.json`` (scope file digest, variants, backend, budget,
benchmarks, errors) and ``timing.json`` (worker count and wall-clock per task; reported only).
``environment.json`` next to the stages directory (``output/environment.json`` by default) records
the machine (CPU, OS, Python, z3, PyYAML, source digest).

Determinism: every (benchmark, variant) task runs in a fresh worker process
(``max_tasks_per_child=1``), results are keyed by benchmark and written by the task itself, and
no output depends on completion order, so every file except ``timing*.json`` is byte-identical
across runs and independent of ``--workers``. Budgets are expansion counts (``max_expansions``),
never seconds.

Usage:
  uv run python scripts/ablation/run_stages.py --scope scope/eval.scope \
      --variants full,rules --cost historical [--workers 8] [--max-expansions 2000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "output"
STAGES = OUTPUT / "stages"
DEFAULT_WORKERS = 8
COST_BACKENDS = ("mock", "historical")


def dumps(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj), encoding="utf-8")


def read_scope_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        t = raw.strip()
        if t and not t.startswith("#"):
            lines.append(t)
    return sorted(set(lines))


def stage_name(scope: Path, cost: str) -> str:
    """``<scope-stem>.<cost>``, e.g. ``eval.historical``."""
    return f"{scope.stem}.{cost}"


def variant_files(variant: str) -> tuple[str, ...]:
    return (f"synth_{variant}.json", f"optimized_{variant}.yml", f"check_{variant}.json",
            f"timing_{variant}.json", f"error_{variant}.json")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, check=True).stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or "unknown"


def source_digest(root: Path = REPO_ROOT) -> str:
    """sha256 over src/ghav/*.py (sorted by name), identifying the code that produced outputs."""
    h = hashlib.sha256()
    for p in sorted((root / "src" / "ghav").glob("*.py")):
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def environment() -> dict:
    import yaml
    import z3
    return {
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "z3": z3.get_version_string(),
        "pyyaml": yaml.__version__,
        "ghav_source_sha256": source_digest(),
    }


# ---------------------------------------------------------------------------
# One task (runs in a fresh worker process)
# ---------------------------------------------------------------------------

def cost_json(line: str, cost: str) -> dict:
    """Cost report of the ORIGINAL program (static guide) under ``cost``, without per-job detail."""
    from ghav import cost as costmod
    from ghav import pipeline
    bench = pipeline.parse_scope_line(line)
    p, _ = pipeline.load_program(bench, use_traces=False)
    model = pipeline.cost_model_for(cost, p, bench)
    rep = costmod.cost_report(p, None, model)
    events = {e: {k: v for k, v in d.items() if k != "jobs"} for e, d in sorted(rep["events"].items())}
    return {
        "benchmark": bench.id,
        "backend": cost,
        "model": rep["model"],
        "contexts": rep["contexts"],
        "expected_seconds": round(float(rep["expected_seconds"]), 6),
        "expected_billed_minutes": round(float(rep["expected_billed_minutes"]), 6),
        "coverage": rep["coverage"],
        "events": events,
    }


def redundancy_category(step) -> str:
    """``top`` (⊤ contract), ``transferable`` (writes ``out`` with declared paths, so a transfer
    can carry it), or ``non-transferable`` (only src/dep/tool/external writes: checkout, set-up,
    install, most tests)."""
    c = step.contract
    if c.top:
        return "top"
    if c.out_paths and any(str(r) == "out" for r in c.writes):
        return "transferable"
    return "non-transferable"


def redundancy_json(line: str, cost: str) -> dict:
    """Redundant seconds (``synth.redundancy``'s definition, priced by ``cost``) split by
    category and by contract template; explains which redundancy the search space can remove."""
    from collections import defaultdict

    from ghav import cost as costmod
    from ghav import pipeline, synth
    from ghav import semantics as sem
    bench = pipeline.parse_scope_line(line)
    p, _ = pipeline.load_program(bench, use_traces=False)
    model = pipeline.cost_model_for(cost, p, bench)
    contexts = model.contexts or costmod.DEFAULT_CONTEXTS
    keys = synth.occurrence_classes(p, read_values=True)
    by_id = {j.id: j for j in p.jobs}
    weight: dict[tuple[str, int], float] = defaultdict(float)
    for event, w in contexts:
        runs = costmod.simulate(p, event, model)
        for jid in sorted(runs):
            for i, flag in enumerate(runs[jid].steps):
                if flag:
                    weight[(jid, i)] += w
    seen: dict[tuple, str] = {}
    by_cat: dict[str, float] = defaultdict(float)
    by_tpl: dict[str, float] = defaultdict(float)
    for jid in sem.topological_order(p):
        job = by_id[jid]
        if job.frozen:
            continue
        for i, st in enumerate(job.steps):
            if seen.setdefault(keys[(jid, i)], jid) == jid:
                continue
            secs = float(model.price_step(job, st)[0]) * weight[(jid, i)]
            by_cat[redundancy_category(st)] += secs
            by_tpl[st.contract.template or "?"] += secs
    total = float(costmod.expected_machine_time(p, contexts, model))
    red = sum(by_cat.values())
    return {
        "benchmark": bench.id,
        "backend": cost,
        "machine_seconds": round(total, 6),
        "redundant_seconds": round(red, 6),
        "redundancy_ratio": round(min(1.0, red / total), 6) if total > 0 else 0.0,
        "by_category": {k: round(v, 6) for k, v in sorted(by_cat.items())},
        "by_template": {k: round(v, 6) for k, v in sorted(by_tpl.items())},
    }


MAX_LISTED = 20


def consistency_json(line: str, cost: str) -> dict:
    """Agreement between the two executions GHAV computes for the ORIGINAL program, with every
    task succeeding: the cost model's concrete simulation (``cost.simulate``) and the symbolic
    semantics (``semantics.evaluate(assume_success=True)``).

    For each event, a job instance or step occurrence that the simulation executes but that the
    semantics skips in EVERY context of that event is listed. Such a disagreement means one of
    the two models is wrong about when the step runs; if the semantics is the wrong one, the
    checker compares the step as dead code and may accept candidates that drop it, so
    improvements on such a workflow are not trustworthy (reported, never used to filter)."""
    import z3

    from ghav import cost as costmod
    from ghav import pipeline
    from ghav import semantics as sem
    from ghav.wir import EVENTS
    bench = pipeline.parse_scope_line(line)
    p, _ = pipeline.load_program(bench, use_traces=False)
    ctx = sem.SymbolicContext.for_programs(p)
    obs = sem.evaluate(p, ctx, assume_success=True)
    solver = z3.Solver(ctx=ctx.z3)
    solver.add(*ctx.background())
    out: dict = {"benchmark": bench.id, "events": {}}
    total_jobs = total_steps = 0
    for event in EVENTS:
        runs = costmod.simulate(p, event, costmod.MOCK)
        jobs_never: list[str] = []
        steps_never: list[str] = []
        executed = 0
        solver.push()
        solver.add(ctx.event_name == ctx.lit(event))
        for jid in sorted(runs):
            r = runs[jid]
            if not r.ran:
                continue
            ran = obs.job_ran.get(jid)
            if ran is not None and solver.check(ran) == z3.unsat:
                jobs_never.append(jid)
                continue
            for i, flag in enumerate(r.steps):
                st = obs.step_status.get((jid, i))
                if not flag or st is None:
                    continue
                executed += 1
                if solver.check(z3.Not(sem.status_is(st, sem.SKIPPED))) == z3.unsat:
                    steps_never.append(f"{jid}#{i} {obs.step_labels.get((jid, i), '')}".strip())
        solver.pop()
        total_jobs += len(jobs_never)
        total_steps += len(steps_never)
        out["events"][event] = {
            "simulated_steps": executed,
            "jobs_never_run_in_semantics": len(jobs_never),
            "steps_never_run_in_semantics": len(steps_never),
            "jobs": jobs_never[:MAX_LISTED],
            "steps": steps_never[:MAX_LISTED],
        }
    out["jobs_never_run_in_semantics"] = total_jobs
    out["steps_never_run_in_semantics"] = total_steps
    out["consistent"] = total_jobs == 0 and total_steps == 0
    return out


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run_task(task: tuple[str, str | None, str, str, int | None]) -> dict:
    """One (benchmark, variant) run, or the benchmark's ``cost.json`` when ``variant`` is None.

    ``run_benchmark`` writes into a private temporary directory; the files are then moved into
    the benchmark directory with ``os.replace`` (atomic), so concurrent variants of one
    benchmark never interleave writes of the shared ``lowered.json``/``contracts.json``.
    """
    import tempfile
    line, variant, cost, out_dir, max_expansions = task
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if variant is None:
        status, summary = "ok", {}
        for fname, fn in (("cost.json", cost_json), ("redundancy.json", redundancy_json),
                          ("consistency.json", consistency_json)):
            try:
                obj = fn(line, cost)
            except Exception as exc:  # noqa: BLE001
                obj = {"error": f"{type(exc).__name__}: {exc}", "backend": cost}
                status, summary = "error", {"error": obj["error"]}
            _atomic_write_text(out / fname, dumps(obj))
        return {"line": line, "variant": None, "status": status,
                "wall_s": round(time.perf_counter() - start, 3), **summary}

    from ghav import pipeline
    for name in variant_files(variant):
        (out / name).unlink(missing_ok=True)
    status = "ok"
    with tempfile.TemporaryDirectory(prefix="ghav-run-") as tmp:
        tmpd = Path(tmp)
        try:
            summary = pipeline.run_benchmark(line, variant, tmpd, cost_backend=cost,
                                             max_expansions=max_expansions, use_traces=False)
        except Exception as exc:  # noqa: BLE001 - recorded per benchmark, never fatal for the scope
            status = "error"
            summary = {"error": f"{type(exc).__name__}: {exc}"}
            (tmpd / f"error_{variant}.json").write_text(dumps({
                "variant": variant, "scope_line": line, "type": type(exc).__name__,
                "message": str(exc), "traceback": traceback.format_exc()}), encoding="utf-8")
        for f in sorted(tmpd.iterdir()):
            if f.is_file():
                _atomic_write_text(out / f.name, f.read_text(encoding="utf-8"))
    return {"line": line, "variant": variant, "status": status,
            "wall_s": round(time.perf_counter() - start, 3), **summary}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def benchmark_id(line: str) -> str:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from ghav.pipeline import parse_scope_line
    return parse_scope_line(line).id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scope", type=Path, required=True)
    ap.add_argument("--variants", default="full")
    ap.add_argument("--cost", default="historical", choices=COST_BACKENDS)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--max-expansions", type=int, default=None,
                    help="uniform search budget for every variant (default: SynthConfig's 2000)")
    ap.add_argument("--stages", type=Path, default=STAGES)
    ap.add_argument("--only", default=None, help="comma-separated benchmark ids (debugging)")
    args = ap.parse_args(argv)

    from ghav import synth
    variants = [v for v in args.variants.split(",") if v]
    unknown = [v for v in variants if v not in synth.VARIANTS]
    if unknown:
        ap.error(f"unknown variants {unknown} (known: {sorted(synth.VARIANTS)})")
    lines = read_scope_lines(args.scope)
    ids = {line: benchmark_id(line) for line in lines}
    if len(set(ids.values())) != len(ids):
        ap.error("benchmark ids are not unique in this scope")
    if args.only:
        keep = set(args.only.split(","))
        lines = [ln for ln in lines if ids[ln] in keep]
    name = stage_name(args.scope, args.cost)
    root = args.stages / name
    max_exp = args.max_expansions if args.max_expansions is not None else synth.SynthConfig().max_expansions

    _atomic_write_text(args.stages.parent / "environment.json", dumps(environment()))
    tasks = [(ln, None, args.cost, str(root / ids[ln]), max_exp) for ln in lines]
    # Scheduling only (results do not depend on it): the variants that send many candidates to
    # the checker start first.
    first = {"no-effect-pruning": 0, "no-read-classes": 1}
    order = sorted(variants, key=lambda v: (first.get(v, 2), variants.index(v)))
    tasks += [(ln, v, args.cost, str(root / ids[ln]), max_exp) for v in order for ln in lines]
    t0 = time.perf_counter()
    results: dict[tuple[str, str], dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers), max_tasks_per_child=1) as ex:
        for task, res in zip(tasks, ex.map(run_task, tasks)):
            results[(task[0], task[1])] = res
            if task[1] is None:
                continue
            print(f"[{name}] {ids[task[0]]} {task[1]}: {res['status']}"
                  + (f" {res.get('original_cost')} -> {res.get('best_cost')}" if res["status"] == "ok"
                     else f" {res.get('error')}") + f" ({res['wall_s']}s)", flush=True)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    runs = manifest.get("runs", {})
    for v in variants:
        runs[v] = {
            "max_expansions": max_exp,
            "benchmarks": sorted(ids[ln] for ln in lines),
            "errors": {ids[ln]: results[(ln, v)]["error"] for ln in lines
                       if results[(ln, v)]["status"] == "error"},
        }
    write(manifest_path, {
        "scope": str(args.scope.resolve().relative_to(REPO_ROOT)) if args.scope.resolve().is_relative_to(REPO_ROOT)
        else str(args.scope),
        "scope_sha256": hashlib.sha256(args.scope.read_bytes()).hexdigest(),
        "cost_backend": args.cost,
        "contracts": "static guide (use_traces=False)",
        "benchmarks": {ids[ln]: ln for ln in read_scope_lines(args.scope)},
        "runs": dict(sorted(runs.items())),
    })
    timing_path = root / "timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}
    for v in variants:
        timing[v] = {"workers": args.workers, "wall_s": round(time.perf_counter() - t0, 3),
                     "tasks": {ids[ln]: results[(ln, v)]["wall_s"] for ln in lines}}
    write(timing_path, dict(sorted(timing.items())))
    n_err = sum(1 for (ln, v), r in results.items() if v is not None and r["status"] == "error")
    print(f"{name}: {len(tasks)} tasks, {n_err} errors, {time.perf_counter() - t0:.1f}s "
          f"with {args.workers} workers -> {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
