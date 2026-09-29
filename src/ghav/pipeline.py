"""Per-workflow driver used by the CLI and the experiment scripts.

A benchmark is one scope line ``owner/repo@<sha>:<workflow path>`` (docs/EXPERIMENTS.md).
Its repository snapshot lives under ``resources/repos/<owner>__<repo>/<sha>/`` (committed,
read offline). Everything here is deterministic; JSON is written with ``sort_keys=True,
indent=2``.

* ``workflow_stats(line)`` gives the statistics the scope builders select on
  (``scripts/scope/scopelib.py``): ``instances``, ``frozen`` (= ``frozen_instances``),
  ``steps``, ``top_steps``, ``redundancy_ratio`` and ``machine_seconds`` (mock cost).
  Redundancy ratio: the expected mock seconds of step occurrences whose read class (binding
  and read values, all tasks succeeding) already occurs in an earlier job instance
  (topological order), divided by the expected mock machine time (``synth.redundancy``).
* ``run_benchmark(line, variant, out_dir)`` runs the stages of one benchmark into
  ``out_dir`` (the benchmark's own directory, e.g. ``output/stages/<scope>/<id>/``):
  ``lowered.json``, ``contracts.json``, ``synth_<variant>.json``, ``optimized_<variant>.yml``
  and ``check_<variant>.json`` (the round trip: the optimized YAML lowered again and checked
  against the original). Wall-clock times go to ``timing_<variant>.json`` only, so every
  other file is byte-identical across runs.
"""

from __future__ import annotations

import json
import posixpath
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ghav import checker, contracts
from ghav import cost as costmod
from ghav import emit as emitmod
from ghav import synth
from ghav.lower import lower
from ghav.wir import Contract, Expr, JobInstance, Program, Step

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCES = REPO_ROOT / "resources"

_LINE_RE = re.compile(
    r"(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@(?P<sha>[0-9a-f]{40}):(?P<path>\S+)")

__all__ = [
    "Benchmark",
    "COST_BACKENDS",
    "contracts_report",
    "cost_model_for",
    "dumps",
    "load_program",
    "observed_for",
    "parse_scope_line",
    "program_json",
    "run_benchmark",
    "workflow_stats",
]

COST_BACKENDS = ("mock", "historical")


@dataclass(frozen=True)
class Benchmark:
    """One workflow of a pinned repository."""

    repo: str
    sha: str
    path: str

    @property
    def line(self) -> str:
        return f"{self.repo}@{self.sha}:{self.path}"

    @property
    def id(self) -> str:
        """``<owner>__<repo>__<workflow-file-stem>`` (docs/ARCHITECTURE.md)."""
        owner, name = self.repo.split("/", 1)
        stem = posixpath.splitext(posixpath.basename(self.path))[0]
        return f"{owner}__{name}__{stem}"

    def snapshot(self, resources: Path | None = None) -> Path:
        owner, name = self.repo.split("/", 1)
        return Path(resources or RESOURCES) / "repos" / f"{owner}__{name}" / self.sha


def parse_scope_line(line: str) -> Benchmark:
    m = _LINE_RE.fullmatch(str(line).strip())
    if m is None:
        raise ValueError(f"not a pinned workflow scope line (owner/repo@<sha>:<path>): {line!r}")
    return Benchmark(m.group("repo"), m.group("sha"), m.group("path"))


def observed_for(p: Program, root: Path, use_traces: bool) -> object:
    """Observed contracts to annotate with: ``{}`` (static guide only, the default for every
    benchmark, RESEARCH_SCOPE "Provisional decisions") or the benchmark's trace file."""
    return synth.default_observed(p, root) if use_traces else {}


def load_program(bench: Benchmark | str, resources: Path | None = None, *,
                 use_traces: bool = False) -> tuple[Program, Path]:
    """The lowered program with its contracts, and the snapshot root it was read from."""
    b = parse_scope_line(bench) if isinstance(bench, str) else bench
    root = b.snapshot(resources)
    wf = root / b.path
    if not wf.is_file():
        raise FileNotFoundError(f"workflow {b.path} not in snapshot {root}")
    p = lower(wf, root)
    return synth.annotate_like(p, root, observed_for(p, root, use_traces)), root


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Serialization of WIR (lowered.json)
# ---------------------------------------------------------------------------

def _expr_json(e: Any) -> Any:
    if isinstance(e, Expr):
        return [e.op, *[_expr_json(a) for a in e.args]]
    if isinstance(e, (list, tuple)):
        return [_expr_json(x) for x in e]
    return e


def _contract_json(c: Contract) -> dict:
    return {
        "template": c.template,
        "top": c.top,
        "reads": sorted(str(r) for r in c.reads),
        "writes": sorted(str(r) for r in c.writes),
        "out_paths": list(c.out_paths),
    }


def _step_json(s: Step) -> dict:
    b = s.binding
    return {
        "kind": b.kind,
        "text": b.text,
        "with": [list(kv) for kv in b.with_],
        "env": [list(kv) for kv in b.env],
        "working_directory": b.working_directory,
        "shell": b.shell,
        "name": b.name,
        "id": b.step_id,
        "cond": _expr_json(s.cond),
        "contract": _contract_json(s.contract),
        "origin": list(s.origin),
        "origin_instance": s.origin_instance,
    }


def _job_json(j: JobInstance) -> dict:
    perms = j.permissions
    return {
        "id": j.id,
        "orig_job": j.orig_job,
        "check_name": j.check_name,
        "matrix": [list(kv) for kv in j.matrix],
        "runs_on": j.runs_on,
        "cond": _expr_json(j.cond),
        "needs": sorted(j.needs),
        "env": [list(kv) for kv in j.env],
        "permissions": [list(kv) for kv in perms] if isinstance(perms, tuple) else perms,
        "outputs": [[k, _expr_json(e)] for k, e in j.outputs],
        "fail_fast_group": j.fail_fast_group,
        "frozen": j.frozen,
        "frozen_reasons": list(j.frozen_reasons),
        "steps": [_step_json(s) for s in j.steps],
    }


def program_json(p: Program) -> dict:
    """A JSON view of a WIR program (raw YAML mappings are omitted)."""
    return {
        "name": p.name,
        "source": p.source,
        "on": p.on,
        "env": [list(kv) for kv in p.env],
        "permissions": p.permissions,
        "concurrency": p.concurrency,
        "defaults": p.defaults,
        "jobs": [_job_json(j) for j in sorted(p.jobs, key=lambda x: x.id)],
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def stats_of(p: Program) -> dict:
    redundant, total = synth.redundancy(p, costmod.MOCK)
    frozen = sum(1 for j in p.jobs if j.frozen)
    steps = sum(len(j.steps) for j in p.jobs)
    tops = sum(1 for j in p.jobs for s in j.steps if s.contract.top)
    ratio = min(1.0, max(0.0, redundant / total)) if total > 0 else 0.0
    return {
        "instances": len(p.jobs),
        "frozen": frozen,
        "frozen_instances": frozen,
        "steps": steps,
        "top_steps": tops,
        "redundancy_ratio": round(ratio, 6),
        "redundant_seconds": round(redundant, 6),
        "machine_seconds": round(total, 6),
    }


def workflow_stats(scope_line: str, *, resources: Path | None = None, use_traces: bool = False) -> dict:
    """Statistics of one benchmark (mock cost model); see the module docstring."""
    p, _ = load_program(scope_line, resources, use_traces=use_traces)
    return stats_of(p)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def cost_model_for(backend: str, p: Program, bench: Benchmark | None,
                   resources: Path | None = None) -> costmod.CostModel:
    """``mock`` (docs/COST_MODEL.md) or ``historical`` (medians under resources/timings)."""
    if backend == "mock":
        return costmod.MOCK
    if backend == "historical":
        loader = getattr(costmod, "load_historical_model", None)
        if loader is None or bench is None:
            raise ValueError("historical cost model unavailable")
        timings_dir = Path(resources or RESOURCES) / "timings"
        return loader(p, bench.id, timings_dir)
    raise ValueError(f"unknown cost backend {backend!r} (known: {', '.join(COST_BACKENDS)})")


def run_benchmark(scope_line: str, variant: str, out_dir: Path | str, *, cost_backend: str = "mock",
                  resources: Path | None = None, max_expansions: int | None = None,
                  use_traces: bool = False) -> dict:
    """Run lower, contracts, synthesis (``variant``) and the round trip for one benchmark,
    writing the stage files into ``out_dir``. Returns a summary."""
    bench = parse_scope_line(scope_line)
    out = Path(out_dir)
    root = bench.snapshot(resources)
    wf = root / bench.path
    if not wf.is_file():
        raise FileNotFoundError(f"workflow {bench.path} not in snapshot {root}")
    lowered = lower(wf, root)
    _write(out / "lowered.json", dumps(program_json(lowered)))
    observed = observed_for(lowered, root, use_traces)
    p = synth.annotate_like(lowered, root, observed)
    _write(out / "contracts.json", dumps(contracts_report(p, root, observed)))

    model = cost_model_for(cost_backend, p, bench, resources)
    overrides = {} if max_expansions is None else {"max_expansions": max_expansions}
    cfg = synth.SynthConfig.for_variant(variant, **overrides)
    res = synth.synthesize(p, cfg, repo_root=root, model=model, observed=observed)
    record = res.to_json(include_timing=False)
    record["benchmark"] = bench.id
    record["scope_line"] = bench.line
    record["stats"] = stats_of(p)
    _write(out / f"synth_{variant}.json", dumps(record))

    try:
        text = emitmod.emit(res.program)
    except emitmod.EmitError as exc:
        text = ""
        rt_json: dict = {"equivalent": False, "reason": "emit-error", "error": str(exc)}
    else:
        rt_json = _roundtrip_json(p, text, root, cfg.rlimit, observed)
    _write(out / f"optimized_{variant}.yml", text)
    _write(out / f"check_{variant}.json", dumps(rt_json))
    _write(out / f"timing_{variant}.json", dumps({"wall_ms": res.wall_ms}))
    return {
        "benchmark": bench.id,
        "variant": variant,
        "original_cost": record["original_cost"],
        "best_cost": record["best_cost"],
        "accepted": record["accepted"],
        "expansions": record["expansions"],
        "roundtrip_equivalent": bool(rt_json.get("equivalent")),
    }


def contracts_report(p: Program, root: Path, observed: object) -> dict:
    """``contracts.report`` with the given observed contracts (when supported)."""
    import inspect
    if "observed" in inspect.signature(contracts.report).parameters:
        return contracts.report(p, root, observed=observed)
    return contracts.report(p, root)


def _roundtrip_json(original: Program, text: str, root: Path, rlimit: int, observed: object) -> dict:
    from ghav.lower import LowerError
    with tempfile.TemporaryDirectory(prefix="ghav-stage-") as d:
        path = Path(d) / "optimized.yml"
        path.write_text(text, encoding="utf-8")
        try:
            again = synth.annotate_like(lower(path, root), root, observed)
        except (LowerError, OSError, ValueError) as exc:
            return {"equivalent": False, "reason": "lower-error", "error": str(exc)}
    return checker.check(original, again, rlimit=rlimit).to_json(include_timing=False)
