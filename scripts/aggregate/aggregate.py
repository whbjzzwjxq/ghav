#!/usr/bin/env python3
"""Turn output/stages/* into the RQ1-RQ4 CSV files and LaTeX tables (docs/EXPERIMENTS.md).

Inputs (written by scripts/ablation/run_stages.py and scripts/ablation/check_devopt.py):

    output/stages/<scope>.<backend>/<benchmark-id>/{synth,check,timing,error}_<variant>.*,
                                                   contracts.json, cost.json, redundancy.json
    output/stages/devopt.pairs/<pair-id>__<benchmark-id>/devopt_check.json

Outputs:

    output/rq1_effectiveness.csv   variant full, one row per (scope, backend, benchmark)
    output/rq2_rules.csv           full vs rules on eval, one row per (backend, benchmark)
    output/rq3_ablation.csv        every variant on eval, one row per (backend, benchmark, variant)
    output/rq4_devopt.csv          one row per developer before/after pair
    output/summary.json            the headline numbers of every table
    output/tables/rq{1,2,3,4}.tex  booktabs tables (small), captions state backend and metric

Reporting rules (docs/EXPERIMENTS.md): every number states its cost backend; mock-cost numbers
are labeled "mock"; both machine seconds and billed minutes are reported; counts of unsupported
(errors), frozen and all-⊤ workflows are reported next to successes. Rows are sorted; the only
non-deterministic values are the wall-clock columns (``wall_s``) of RQ3.

Usage:
  uv run python scripts/aggregate/aggregate.py [--stages output/stages] [--out output]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "output"
STAGES = OUTPUT / "stages"

VARIANTS = ("full", "rules", "no-effect-pruning", "no-cost-bound", "no-read-classes")
BACKENDS = ("historical", "mock")
SCOPES_RQ1 = ("eval", "random")
EPS = 1e-6
HIST_LABEL = ("historical cost (medians of the last 5 successful runs per event before "
              "2026-09-01; transfer steps and unmatched steps priced by the mock table)")
MOCK_LABEL = "mock cost (docs/COST_MODEL.md table)"
PRUNING_REASONS = ("settings", "status-dependency", "untransferred-read", "well-formed", "fail-fast")


# ---------------------------------------------------------------------------
# Reading stage files
# ---------------------------------------------------------------------------

def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stage_dirs(stages: Path) -> dict[tuple[str, str], Path]:
    """(scope, backend) -> directory, for directories named ``<scope>.<backend>``."""
    out = {}
    if not stages.is_dir():
        return out
    for d in sorted(stages.iterdir()):
        if d.is_dir() and "." in d.name:
            scope, backend = d.name.rsplit(".", 1)
            if backend in BACKENDS:
                out[(scope, backend)] = d
    return out


def benchmark_dirs(stage: Path) -> list[Path]:
    manifest = _load(stage / "manifest.json") or {}
    names = sorted(manifest.get("benchmarks", {})) or sorted(
        d.name for d in stage.iterdir() if d.is_dir())
    return [stage / n for n in names]


def _r(x: float | None, nd: int = 3) -> float | str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return round(float(x), nd)


def moves_summary(moves: list[dict] | None) -> str:
    counts: dict[str, int] = defaultdict(int)
    for m in moves or []:
        kind = str(m.get("move") or "?")
        if kind == "share":
            enc = "new-job" if m.get("producer") == "job" else "instance"
            kind = f"share-{'transfer' if m.get('transfer') else 'drop'}-{enc}"
        counts[kind] += 1
    return " ".join(f"{k}x{v}" for k, v in sorted(counts.items()))


def contracts_summary(bdir: Path) -> dict:
    c = _load(bdir / "contracts.json") or {}
    return c.get("summary", {}) if isinstance(c, dict) else {}


def issue_count(check: dict | None, program: str, kind: str) -> int:
    if not isinstance(check, dict):
        return 0
    return sum(1 for i in check.get("issues") or [] if i.get("program") == program and i.get("kind") == kind)


def variant_row(bdir: Path, variant: str) -> dict:
    """Everything the tables need about one (benchmark, variant) run."""
    row: dict[str, Any] = {"benchmark": bdir.name}
    err = _load(bdir / f"error_{variant}.json")
    syn = _load(bdir / f"synth_{variant}.json")
    chk = _load(bdir / f"check_{variant}.json")
    tim = _load(bdir / f"timing_{variant}.json") or {}
    cost = _load(bdir / "cost.json") or {}
    red = _load(bdir / "redundancy.json") or {}
    cons = _load(bdir / "consistency.json") or {}
    summ = contracts_summary(bdir)
    stats = (syn or {}).get("stats") or {}
    row["status"] = "error" if err or syn is None else "ok"
    row["error"] = (f"{err.get('type')}: {err.get('message')}" if err else ("missing" if syn is None else ""))
    inst = stats.get("instances")
    frozen = stats.get("frozen_instances", stats.get("frozen"))
    unfrozen_steps = summ.get("unfrozen_steps")
    unfrozen_top = summ.get("unfrozen_top_steps")
    row.update({
        "instances": inst if inst is not None else "",
        "frozen_instances": frozen if frozen is not None else "",
        "steps": stats.get("steps", ""),
        "top_steps": stats.get("top_steps", ""),
        "unfrozen_top_share": _r(summ.get("unfrozen_top_share"), 4),
        "all_frozen": bool(inst) and frozen == inst,
        "all_top": bool(unfrozen_steps) and unfrozen_top == unfrozen_steps,
        "redundancy_ratio_mock": _r(stats.get("redundancy_ratio"), 4),
        "redundant_seconds": _r(red.get("redundant_seconds")),
        "redundant_top": _r((red.get("by_category") or {}).get("top", 0.0)),
        "redundant_transferable": _r((red.get("by_category") or {}).get("transferable", 0.0)),
        "redundant_nontransferable": _r((red.get("by_category") or {}).get("non-transferable", 0.0)),
    })
    row["model_consistent"] = bool(cons.get("consistent")) if cons else ""
    row["semantics_never_jobs"] = cons.get("jobs_never_run_in_semantics", "") if cons else ""
    row["semantics_never_steps"] = cons.get("steps_never_run_in_semantics", "") if cons else ""
    cov = cost.get("coverage") or {}
    row["matched_fraction"] = _r(cov.get("matched_fraction"), 4) if cov else ""
    weights = {e: w for e, w in cost.get("contexts") or []}
    row["weight_push"] = _r(weights.get("push"), 4) if weights else ""
    if syn is not None:
        o, b = float(syn["original_cost"]), float(syn["best_cost"])
        om, bm = float(syn["original_billed_minutes"]), float(syn["best_billed_minutes"])
        rej = syn.get("rejected") or {}
        pruning = rej.get("pruning") or {}
        row.update({
            "cost_model": syn.get("cost_model", ""),
            "max_expansions": (syn.get("config") or {}).get("max_expansions", ""),
            "original_seconds": _r(o), "best_seconds": _r(b), "saving_seconds": _r(o - b),
            "saving_pct": _r(100.0 * (o - b) / o if o > 0 else 0.0, 3),
            "original_billed_minutes": _r(om), "best_billed_minutes": _r(bm),
            "saving_billed_minutes": _r(om - bm),
            "improved": bool(syn.get("improved")),
            # a verified improvement on a workflow whose semantics skips steps the cost
            # simulation executes may rest on a vacuous comparison (run_stages.consistency_json)
            "improved_model_inconsistent": bool(syn.get("improved")) and cons.get("consistent") is False,
            "expansions": syn.get("expansions", ""), "generated": syn.get("generated", ""),
            "checked": syn.get("checked", ""), "accepted": syn.get("accepted", ""),
            "rejected_checker": rej.get("checker", ""), "rejected_roundtrip": rej.get("roundtrip", ""),
            "rejected_guard": rej.get("guard", ""), "bound_pruned": rej.get("bound", ""),
            "nogood_skipped": rej.get("nogood", ""),
            "pruned": sum(int(v) for v in pruning.values()),
            **{f"pruned_{k}": int(pruning.get(k, 0)) for k in PRUNING_REASONS},
            "pruned_other": sum(int(v) for k, v in pruning.items() if k not in PRUNING_REASONS),
            "completion_rejected": sum(int(v) for v in (rej.get("completion") or {}).values()),
            "shareable_classes": syn.get("shareable_classes", ""),
            "budget_exhausted": bool(syn.get("budget_exhausted")),
            "best_moves": moves_summary(syn.get("best_moves")),
        })
    row["roundtrip_equivalent"] = bool(chk.get("equivalent")) if isinstance(chk, dict) else False
    row["roundtrip_reason"] = chk.get("reason", "") if isinstance(chk, dict) else ""
    row["original_duplicate_artifact"] = issue_count(chk, "original", "duplicate-artifact")
    row["wall_s"] = _r(tim.get("wall_ms", 0) / 1000.0, 1) if tim else ""
    return row


# ---------------------------------------------------------------------------
# CSV and LaTeX helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("true" if v is True else "false" if v is False else v) for k, v in r.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")


def tex_escape(s: Any) -> str:
    text = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"), ("⊤", r"$\top$"), ("→", r"$\to$")):
        text = text.replace(a, b)
    return text


def fmt(x: Any, nd: int = 1) -> str:
    if x == "" or x is None:
        return "--"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}".replace(",", "{,}")
    if isinstance(x, float):
        s = f"{x:,.{nd}f}".replace(",", "{,}")
        return s
    return tex_escape(x)


def table(label: str, caption: str, header: list[str], rows: list[list[str]], align: str,
          notes: str = "", star: bool = False, midrules: Iterable[int] = (),
          blocks: list[tuple[list[str], list[list[str]], str, Iterable[int]]] | None = None,
          colsep: str = "3pt") -> str:
    """A booktabs table (``\\small``). ``blocks`` adds further tabulars below the first one."""
    env = "table*" if star else "table"

    def tabular(hdr: list[str], rws: list[list[str]], al: str, mids_: Iterable[int]) -> list[str]:
        mids = set(mids_)
        body = []
        for i, r in enumerate(rws):
            if i in mids:
                body.append(r"\midrule")
            body.append(" & ".join(r) + r" \\")
        return [rf"\begin{{tabular}}{{{al}}}", r"\toprule", " & ".join(hdr) + r" \\", r"\midrule",
                *body, r"\bottomrule", r"\end{tabular}"]

    out = [
        "% Generated by scripts/aggregate/aggregate.py from output/stages; do not edit.",
        rf"\begin{{{env}}}[t]",
        r"\small",
        r"\centering",
        rf"\setlength{{\tabcolsep}}{{{colsep}}}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        *tabular(header, rows, align, midrules),
    ]
    for hdr, rws, al, mids_ in blocks or []:
        out += [r"\par\medskip", *tabular(hdr, rws, al, mids_)]
    if notes:
        out.append(rf"\par\smallskip{{\footnotesize {notes}}}")
    out.append(rf"\end{{{env}}}")
    return "\n".join(out) + "\n"


def backend_label(backend: str) -> str:
    return HIST_LABEL if backend == "historical" else MOCK_LABEL


def short_backend(backend: str) -> str:
    return "historical" if backend == "historical" else "mock"


def pct(a: float, b: float) -> float:
    return 100.0 * a / b if b > 0 else 0.0


# ---------------------------------------------------------------------------
# RQ1: effectiveness (variant full) on eval and random
# ---------------------------------------------------------------------------

RQ1_COLUMNS = [
    "scope", "backend", "benchmark", "status", "error", "instances", "frozen_instances", "all_frozen",
    "steps", "top_steps", "unfrozen_top_share", "all_top", "redundancy_ratio_mock", "redundant_seconds",
    "redundant_top", "redundant_transferable", "redundant_nontransferable", "matched_fraction",
    "weight_push", "original_seconds", "best_seconds", "saving_seconds", "saving_pct",
    "original_billed_minutes", "best_billed_minutes", "saving_billed_minutes", "improved",
    "accepted", "checked", "expansions", "budget_exhausted", "shareable_classes", "best_moves",
    "roundtrip_equivalent", "roundtrip_reason", "original_duplicate_artifact", "model_consistent",
    "semantics_never_jobs", "semantics_never_steps", "improved_model_inconsistent", "max_expansions",
]


def scope_summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    o = sum(float(r["original_seconds"]) for r in ok)
    b = sum(float(r["best_seconds"]) for r in ok)
    om = sum(float(r["original_billed_minutes"]) for r in ok)
    bm = sum(float(r["best_billed_minutes"]) for r in ok)
    improved = [r for r in ok if r["improved"]]
    inst = sum(int(r["instances"] or 0) for r in rows)
    frozen = sum(int(r["frozen_instances"] or 0) for r in rows)
    steps = sum(int(r["steps"] or 0) for r in rows)
    tops = sum(int(r["top_steps"] or 0) for r in rows)
    red = sum(float(r["redundant_seconds"] or 0) for r in rows)
    red_tr = sum(float(r["redundant_transferable"] or 0) for r in rows)
    red_top = sum(float(r["redundant_top"] or 0) for r in rows)
    mf = [float(r["matched_fraction"]) for r in rows if r["matched_fraction"] != ""]
    shares = [float(r["saving_pct"]) for r in improved]
    return {
        "workflows": len(rows),
        "errors": len(rows) - len(ok),
        "all_frozen": sum(1 for r in rows if r["all_frozen"]),
        "all_top": sum(1 for r in rows if r["all_top"]),
        "with_frozen": sum(1 for r in rows if int(r["frozen_instances"] or 0) > 0),
        "instances": inst, "frozen_instances": frozen,
        "steps": steps, "top_steps": tops, "top_share": round(pct(tops, steps), 3),
        "improved": len(improved),
        "roundtrip_equivalent": sum(1 for r in ok if r["roundtrip_equivalent"]),
        "original_duplicate_artifact": sum(1 for r in rows if r["original_duplicate_artifact"]),
        "model_inconsistent": sum(1 for r in rows if r["model_consistent"] is False),
        "improved_model_inconsistent": sum(1 for r in ok if r.get("improved_model_inconsistent")),
        "original_seconds": round(o, 3), "best_seconds": round(b, 3),
        "saving_seconds": round(o - b, 3), "saving_pct": round(pct(o - b, o), 3),
        "original_billed_minutes": round(om, 3), "best_billed_minutes": round(bm, 3),
        "saving_billed_minutes": round(om - bm, 3), "saving_billed_pct": round(pct(om - bm, om), 3),
        "median_saving_pct_improved": round(statistics.median(shares), 3) if shares else 0.0,
        "redundant_seconds": round(red, 3), "redundant_transferable_seconds": round(red_tr, 3),
        "redundant_top_seconds": round(red_top, 3),
        "redundant_transferable_pct": round(pct(red_tr, red), 3),
        "matched_fraction_median": round(statistics.median(mf), 4) if mf else None,
        "matched_fraction_mean": round(sum(mf) / len(mf), 4) if mf else None,
        "no_history": sum(1 for x in mf if x == 0.0),
        "budget_exhausted": sum(1 for r in ok if r.get("budget_exhausted")),
        "checked": sum(int(r.get("checked") or 0) for r in ok),
    }


def rq1(stages: dict[tuple[str, str], Path], out: Path) -> dict:
    rows = []
    for (scope, backend), d in sorted(stages.items()):
        if scope not in SCOPES_RQ1:
            continue
        for bdir in benchmark_dirs(d):
            if not (bdir / "synth_full.json").exists() and not (bdir / "error_full.json").exists():
                continue
            rows.append({"scope": scope, "backend": backend, **variant_row(bdir, "full")})
    rows.sort(key=lambda r: (r["scope"], r["backend"], r["benchmark"]))
    write_csv(out / "rq1_effectiveness.csv", rows, RQ1_COLUMNS)
    summary = {}
    for scope in SCOPES_RQ1:
        for backend in BACKENDS:
            rs = [r for r in rows if r["scope"] == scope and r["backend"] == backend]
            if rs:
                summary[f"{scope}.{backend}"] = scope_summary(rs)
    # table: effectiveness per scope and backend, then scope characteristics
    header = ["Scope", "Cost", "WF", "Impr.", "RT", "s before", "s after", "s saved",
              "min before", "min after", "min saved"]
    body = []
    for key, s in summary.items():
        scope, backend = key.rsplit(".", 1)
        body.append([
            tex_escape(scope), short_backend(backend), fmt(s["workflows"]),
            fmt(s["improved"]) + (f" ({fmt(s['improved_model_inconsistent'])} incons.)"
                                  if s["improved_model_inconsistent"] else ""),
            f"{fmt(s['roundtrip_equivalent'])}/{fmt(s['workflows'] - s['errors'])}",
            fmt(s["original_seconds"], 0), fmt(s["best_seconds"], 0), f"{fmt(s['saving_pct'])}\\%",
            fmt(s["original_billed_minutes"]), fmt(s["best_billed_minutes"]),
            f"{fmt(s['saving_billed_pct'])}\\%",
        ])
    header2 = ["Scope", "WF", "Err.", r"All-frz.", r"All-$\top$", "Incons.", r"Frz.\ inst.",
               r"$\top$ steps", r"Red.\ tr.\ (hist.)", r"Red.\ tr.\ (mock)"]
    body2 = []
    for scope in SCOPES_RQ1:
        ss = {b: summary.get(f"{scope}.{b}") for b in BACKENDS}
        s = ss.get("historical") or ss.get("mock")
        if s is None:
            continue
        body2.append([
            tex_escape(scope), fmt(s["workflows"]), fmt(s["errors"]), fmt(s["all_frozen"]),
            fmt(s["all_top"]), fmt(s["model_inconsistent"]),
            f"{fmt(s['frozen_instances'])}/{fmt(s['instances'])}", f"{fmt(s['top_share'])}\\%",
            *[f"{fmt(ss[b]['redundant_transferable_pct'])}\\%" if ss.get(b) else "--" for b in BACKENDS],
        ])
    hist = [s for k, s in summary.items() if k.endswith(".historical")]
    cov = ", ".join(f"{k.split('.')[0]}: median {fmt(s['matched_fraction_median'], 2)}, "
                    f"{fmt(s['no_history'])} without history"
                    for k, s in summary.items() if k.endswith(".historical")) if hist else ""
    max_exp = sorted({r.get("max_expansions") for r in rows if r.get("max_expansions") != ""})
    caption = ("RQ1: expected machine time per run before and after GHAV (variant \\emph{full}, static "
               "contract guide, budget " + ", ".join(fmt(x) for x in max_exp) + " expansions). "
               "Rows labeled \\emph{mock} use the mock cost table; rows labeled \\emph{historical} use "
               "historical cost (medians of the last 5 successful runs per event before 2026-09-01; "
               "unmatched and transfer steps priced by the mock table" +
               (f"; fraction of step occurrences priced from history per workflow, {cov}" if cov else "") +
               ")" +
               ". Metrics: expected machine seconds and expected billed minutes (each executed job "
               "rounded up to a whole minute), summed over the scope; the event weights come from the "
               "backend.")
    notes = ("WF: workflows; Err.: synthesis errors (unsupported); All-frz.: every instance frozen; "
             r"All-$\top$: every unfrozen step $\top$; Incons.: workflows where the semantics never runs a "
             "step that the cost simulation runs (all tasks succeeding), a modeling error that can make "
             r"verdicts vacuous; Frz.\ inst.: frozen/total job instances; "
             r"$\top$ steps: share of steps with the $\top$ contract; Red.\ tr.: share of redundant seconds "
             "in steps whose output a transfer can carry (the rest is checkout, set-up, install, or "
             r"$\top$); Impr.: workflows with a verified improvement; RT: emitted workflows whose round "
             "trip is equivalent; s: expected machine seconds; min: expected billed minutes; saved: share "
             "of the scope total.")
    text = table("tab:rq1", caption, header, body, "llrrrrrrrrr", notes=notes, star=True,
                 blocks=[(header2, body2, "lrrrrrrrrr", ())])
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tables" / "rq1.tex").write_text(text, encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# RQ2: full vs rules on eval
# ---------------------------------------------------------------------------

RQ2_COLUMNS = ["backend", "benchmark", "status_full", "status_rules", "original_seconds",
               "full_saving_seconds", "rules_saving_seconds", "full_saving_billed_minutes",
               "rules_saving_billed_minutes", "full_improved", "rules_improved", "full_only",
               "rules_only", "full_best_moves", "rules_best_moves"]


def rq2(stages: dict[tuple[str, str], Path], out: Path) -> dict:
    rows = []
    for (scope, backend), d in sorted(stages.items()):
        if scope != "eval":
            continue
        for bdir in benchmark_dirs(d):
            f, r = variant_row(bdir, "full"), variant_row(bdir, "rules")
            if not (bdir / "synth_rules.json").exists() and not (bdir / "error_rules.json").exists():
                continue
            fi, ri = bool(f.get("improved")), bool(r.get("improved"))
            rows.append({
                "backend": backend, "benchmark": bdir.name, "status_full": f["status"],
                "status_rules": r["status"], "original_seconds": f.get("original_seconds", ""),
                "full_saving_seconds": f.get("saving_seconds", ""), "rules_saving_seconds": r.get("saving_seconds", ""),
                "full_saving_billed_minutes": f.get("saving_billed_minutes", ""),
                "rules_saving_billed_minutes": r.get("saving_billed_minutes", ""),
                "full_improved": fi, "rules_improved": ri,
                "full_only": fi and (not ri or float(f["saving_seconds"]) > float(r["saving_seconds"]) + EPS),
                "rules_only": ri and (not fi or float(r["saving_seconds"]) > float(f["saving_seconds"]) + EPS),
                "full_best_moves": f.get("best_moves", ""), "rules_best_moves": r.get("best_moves", ""),
            })
    rows.sort(key=lambda x: (x["backend"], x["benchmark"]))
    write_csv(out / "rq2_rules.csv", rows, RQ2_COLUMNS)
    summary = {}
    for backend in BACKENDS:
        rs = [x for x in rows if x["backend"] == backend]
        if not rs:
            continue

        def tot(k: str) -> float:
            return round(sum(float(x[k] or 0) for x in rs), 3)
        summary[backend] = {
            "workflows": len(rs),
            "errors_full": sum(1 for x in rs if x["status_full"] != "ok"),
            "errors_rules": sum(1 for x in rs if x["status_rules"] != "ok"),
            "full_improved": sum(1 for x in rs if x["full_improved"]),
            "rules_improved": sum(1 for x in rs if x["rules_improved"]),
            "full_better": sum(1 for x in rs if x["full_only"]),
            "rules_better": sum(1 for x in rs if x["rules_only"]),
            "original_seconds": tot("original_seconds"),
            "full_saving_seconds": tot("full_saving_seconds"), "rules_saving_seconds": tot("rules_saving_seconds"),
            "full_saving_billed_minutes": tot("full_saving_billed_minutes"),
            "rules_saving_billed_minutes": tot("rules_saving_billed_minutes"),
        }
    header = ["Cost", "WF", "Space", "Err.", "Impr.", "Better", "Saved s", "Saved min"]
    body = []
    for backend, s in summary.items():
        for space in ("full", "rules"):
            body.append([short_backend(backend), fmt(s["workflows"]),
                         space,
                         fmt(s[f"errors_{space}"]), fmt(s[f"{space}_improved"]),
                         fmt(s["full_better" if space == "full" else "rules_better"]),
                         f"{fmt(s[f'{space}_saving_seconds'])} ({fmt(pct(s[f'{space}_saving_seconds'], s['original_seconds']))}\\%)",
                         fmt(s[f"{space}_saving_billed_minutes"])])
    caption = ("RQ2: synthesis over the full search space vs.\\ the rule-restricted space (merge and "
               "whole-prefix hoist) on the evaluation scope, static contract guide. Savings are summed "
               "expected machine seconds and expected billed minutes per run; rows labeled \\emph{mock} "
               "use the mock cost table, rows labeled \\emph{historical} historical medians (transfer "
               "steps priced by the mock table).")
    notes = ("full: synthesis over the full search space; rules: merge and whole-prefix hoist only; "
             "Impr.: workflows with a verified improvement; Better: workflows where this space saves "
             "strictly more machine seconds than the other; Saved s: with the share of the scope total.")
    mid = [i for i in range(2, len(body), 2)]
    text = table("tab:rq2", caption, header, body, "lrlrrrrr", notes=notes, midrules=mid)
    (out / "tables" / "rq2.tex").write_text(text, encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# RQ3: ablation on eval
# ---------------------------------------------------------------------------

RQ3_COLUMNS = ["backend", "benchmark", "variant", "status", "error", "original_seconds", "best_seconds",
               "saving_seconds", "saving_billed_minutes", "improved", "improved_model_inconsistent",
               "model_consistent", "best_moves", "expansions", "generated", "checked",
               "accepted", "rejected_checker", "rejected_roundtrip", "rejected_guard", "pruned",
               *[f"pruned_{k}" for k in PRUNING_REASONS], "pruned_other", "completion_rejected",
               "bound_pruned", "nogood_skipped", "budget_exhausted", "max_expansions", "wall_s"]


def rq3(stages: dict[tuple[str, str], Path], out: Path) -> dict:
    rows = []
    for (scope, backend), d in sorted(stages.items()):
        if scope != "eval":
            continue
        for bdir in benchmark_dirs(d):
            for v in VARIANTS:
                if (bdir / f"synth_{v}.json").exists() or (bdir / f"error_{v}.json").exists():
                    rows.append({"backend": backend, "variant": v, **variant_row(bdir, v)})
    order = {v: i for i, v in enumerate(VARIANTS)}
    rows.sort(key=lambda x: (x["backend"], x["benchmark"], order[x["variant"]]))
    write_csv(out / "rq3_ablation.csv", rows, RQ3_COLUMNS)
    summary: dict[str, dict] = {}
    for backend in BACKENDS:
        for v in VARIANTS:
            rs = [x for x in rows if x["backend"] == backend and x["variant"] == v]
            if not rs:
                continue
            ok = [x for x in rs if x["status"] == "ok"]

            def tot(k: str) -> int:
                return sum(int(x.get(k) or 0) for x in ok)
            summary[f"{backend}.{v}"] = {
                "workflows": len(rs), "errors": len(rs) - len(ok),
                "improved": sum(1 for x in ok if x["improved"]),
                "improved_model_inconsistent": sum(1 for x in ok if x.get("improved_model_inconsistent")),
                "saving_seconds": round(sum(float(x["saving_seconds"]) for x in ok), 3),
                "saving_billed_minutes": round(sum(float(x["saving_billed_minutes"]) for x in ok), 3),
                "expansions": tot("expansions"), "generated": tot("generated"), "checked": tot("checked"),
                "accepted": tot("accepted"), "rejected_checker": tot("rejected_checker"),
                "rejected_roundtrip": tot("rejected_roundtrip"), "pruned": tot("pruned"),
                **{f"pruned_{k}": tot(f"pruned_{k}") for k in PRUNING_REASONS},
                "bound_pruned": tot("bound_pruned"), "nogood_skipped": tot("nogood_skipped"),
                "budget_exhausted": sum(1 for x in ok if x["budget_exhausted"]),
                "wall_s": round(sum(float(x["wall_s"] or 0) for x in rs), 1),
            }
    header = ["Cost", "Variant", "Impr.", "Saved s", "Saved min", "Budget", "Wall s"]
    header2 = ["Cost", "Variant", "Expans.", "Checked", "Refuted", "Pruned", "Bound", "Nogood"]
    body, body2 = [], []
    mids = []
    last = None
    for key, s in summary.items():
        backend, v = key.split(".", 1)
        if last is not None and backend != last:
            mids.append(len(body))
        last = backend
        impr = fmt(s["improved"]) + (f" ({fmt(s['improved_model_inconsistent'])} incons.)"
                                      if s["improved_model_inconsistent"] else "")
        body.append([short_backend(backend), tex_escape(v), impr, fmt(s["saving_seconds"]),
                     fmt(s["saving_billed_minutes"]), fmt(s["budget_exhausted"]), fmt(s["wall_s"], 0)])
        body2.append([short_backend(backend), tex_escape(v), fmt(s["expansions"]), fmt(s["checked"]),
                      fmt(s["rejected_checker"] + s["rejected_roundtrip"]), fmt(s["pruned"]),
                      fmt(s["bound_pruned"]), fmt(s["nogood_skipped"])])
    n = max((s["workflows"] for s in summary.values()), default=0)
    budgets = sorted({x.get("max_expansions") for x in rows if x.get("max_expansions") != ""})
    reached = sorted({x["variant"] for x in rows if x.get("budget_exhausted")}, key=lambda v: order[v])
    budget_text = (f" Budget: {', '.join(fmt(b) for b in budgets)} expansions per workflow for every variant"
                   + (f"; reached only by {', '.join(tex_escape(v) for v in reached)}." if reached else
                      "; never reached.")) if budgets else ""
    caption = (f"RQ3: search effort of each variant on the evaluation scope ({n} workflows, static contract "
               "guide), summed over workflows." + budget_text + " Savings are expected machine seconds and billed minutes per "
               "run under the stated cost backend (\\emph{mock}: mock cost table; \\emph{historical}: "
               "historical medians, transfer steps priced by the mock table). Wall-clock seconds are "
               "reported only; budgets count expansions.")
    notes = ("Impr.: workflows improved (incons.: of which on workflows whose semantics never runs a step "
             "the cost simulation runs; such savings rest on a vacuous comparison); Expans.: popped sketches; Checked: candidates sent to the checker; "
             "Refuted: rejected by the checker or the round trip; Pruned: rejected by effect pruning before "
             "checking; Bound: dropped by the cost bound; Nogood: supersets of refuted sketches skipped; "
             "Budget: workflows that exhausted the expansion budget.")
    text = table("tab:rq3", caption, header, body, "llrrrrr", notes=notes, star=True, midrules=mids,
                 blocks=[(header2, body2, "llrrrrrr", mids)])
    (out / "tables" / "rq3.tex").write_text(text, encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# RQ4: developer-written optimizations
# ---------------------------------------------------------------------------

RQ4_COLUMNS = ["pair", "id", "pr", "title", "workflow", "verdict", "verdict_reason", "checker_reason",
               "counterexample_summary", "counterexample_involves_top_or_frozen", "new_steps",
               "before_instances", "after_instances", "before_frozen", "after_frozen", "before_top_steps",
               "after_top_steps", "before_mock_seconds", "after_mock_seconds", "before_mock_billed_minutes",
               "after_mock_billed_minutes", "blocking_issue_kinds", "original_issue_kinds",
               "synth_full_mock_saving_seconds", "synth_full_mock_improved", "error"]


def rq4(stages_root: Path, stages: dict[tuple[str, str], Path], out: Path) -> dict:
    root = stages_root / "devopt.pairs"
    rows = []
    synth_dir = stages.get(("devopt", "mock"))
    if root.is_dir():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            rec = _load(d / "devopt_check.json")
            if not rec:
                continue
            b, a = rec.get("before_stats") or {}, rec.get("after_stats") or {}
            blocking: dict[str, int] = defaultdict(int)
            for k, v in (rec.get("issue_counts") or {}).items():
                prog, kind, blk = k.split(":", 2)
                if prog == "candidate" and blk == "blocking":
                    blocking[kind] += v
            bench = d.name.split("__", 1)[1]
            srow = variant_row(synth_dir / bench, "full") if synth_dir and (synth_dir / bench).is_dir() else {}
            rows.append({
                "pair": d.name, "id": rec.get("id"), "pr": rec.get("pr"), "title": rec.get("title") or "",
                "workflow": rec.get("before", "").split(":", 1)[-1],
                "verdict": rec.get("verdict"), "verdict_reason": rec.get("verdict_reason"),
                "checker_reason": rec.get("checker_reason", ""),
                "counterexample_summary": rec.get("counterexample_summary", ""),
                "counterexample_involves_top_or_frozen": rec.get("counterexample_involves_top_or_frozen", ""),
                "new_steps": (rec.get("new_steps") or {}).get("count", ""),
                "before_instances": b.get("instances", ""), "after_instances": a.get("instances", ""),
                "before_frozen": b.get("frozen_instances", ""), "after_frozen": a.get("frozen_instances", ""),
                "before_top_steps": b.get("top_steps", ""), "after_top_steps": a.get("top_steps", ""),
                "before_mock_seconds": b.get("mock_seconds", ""), "after_mock_seconds": a.get("mock_seconds", ""),
                "before_mock_billed_minutes": b.get("mock_billed_minutes", ""),
                "after_mock_billed_minutes": a.get("mock_billed_minutes", ""),
                "blocking_issue_kinds": " ".join(f"{k}:{v}" for k, v in sorted(blocking.items())),
                "original_issue_kinds": " ".join(f"{k}:{v}" for k, v in sorted((rec.get("original_issue_kinds") or {}).items())),
                "synth_full_mock_saving_seconds": srow.get("saving_seconds", ""),
                "synth_full_mock_improved": srow.get("improved", ""),
                "error": rec.get("error", ""),
            })
    rows.sort(key=lambda r: r["pair"])
    write_csv(out / "rq4_devopt.csv", rows, RQ4_COLUMNS)
    counts: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["verdict"]] += 1
        reasons[f"{r['verdict']}:{r['verdict_reason']}"] += 1
    summary = {"pairs": len(rows), "prs": len({r["id"] for r in rows}),
               "verdicts": dict(sorted(counts.items())), "reasons": dict(sorted(reasons.items())),
               "synth_improved_before": sum(1 for r in rows if r["synth_full_mock_improved"] is True)}
    header = ["ID", "Workflow", "Verdict", "Reason", "New", r"Mock s", r"Mock min"]
    body = []
    for r in rows:
        reason = tex_escape(r["verdict_reason"])
        if r["verdict"] == "counterexample":
            summ = r["counterexample_summary"]
            reason = tex_escape(summ.split(" | ")[1] if " | " in summ else r["verdict_reason"])
        elif r["blocking_issue_kinds"]:
            kinds = ", ".join(f"{tex_escape(k)} ({fmt(int(v))})" for k, v in
                              (kv.split(":", 1) for kv in r["blocking_issue_kinds"].split()))
            reason = f"{reason}: {kinds}"
        repo = str(r["pr"]).split("#", 1)[0]
        body.append([tex_escape(r["id"]), f"{tex_escape(repo)} {tex_escape(r['workflow'].rsplit('/', 1)[-1])}",
                     tex_escape(r["verdict"]), reason, fmt(r["new_steps"]),
                     f"{fmt(r['before_mock_seconds'], 0)}$\\to${fmt(r['after_mock_seconds'], 0)}",
                     f"{fmt(r['before_mock_billed_minutes'])}$\\to${fmt(r['after_mock_billed_minutes'])}"])
    caption = ("RQ4: the checker on developer-written workflow optimizations (before $\\to$ after of each "
               "merged PR, static contract guide). Costs are expected machine seconds and billed minutes "
               "per run under the \\emph{mock} cost table (no historical timings exist for these commits).")
    notes = ("equivalent: proved equivalent; counterexample: a context in which a persistent region or a "
             "check-group verdict differs; unsupported: no verdict (the after workflow races or duplicates "
             "an artifact name in the model, a side fails to lower, or the before workflow duplicates an "
             "artifact name, which makes the verdict vacuous). New: after steps whose task key (command or "
             "action with its inputs) occurs nowhere in the before workflow; Mock s, Mock min: before $\\to$ after.")
    text = table("tab:rq4", caption, header, body, "lp{2.9cm}lp{2.9cm}rrr", notes=notes, star=True)
    (out / "tables" / "rq4.tex").write_text(text, encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stages", type=Path, default=STAGES)
    ap.add_argument("--out", type=Path, default=OUTPUT)
    args = ap.parse_args(argv)
    stages = stage_dirs(args.stages)
    if not stages and not (args.stages / "devopt.pairs").is_dir():
        print(f"error: no stage directories under {args.stages}; run `make run-stages`", file=sys.stderr)
        return 1
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    summary = {
        "rq1": rq1(stages, args.out),
        "rq2": rq2(stages, args.out),
        "rq3": {k: {kk: vv for kk, vv in v.items() if kk != "wall_s"} for k, v in rq3(stages, args.out).items()},
        "rq4": rq4(args.stages, stages, args.out),
        "stages": sorted(f"{s}.{b}" for s, b in stages),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
