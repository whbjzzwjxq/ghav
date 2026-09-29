"""Command line: ``ghav lower|contracts|check|optimize|emit|stats``.

A workflow argument is either a path to a workflow file (the repository root is ``--repo``,
else the nearest ancestor directory containing ``.github``, else the current directory) or
a pinned scope line ``owner/repo@<sha>:<path>`` (read from ``resources/repos``).

    ghav lower WORKFLOW              lowered WIR as JSON (contracts not yet inferred)
    ghav contracts WORKFLOW          contract audit (template and effect of every step)
    ghav check ORIGINAL CANDIDATE    equivalence check of two workflows of one repository
    ghav optimize WORKFLOW           synthesize; optimized YAML to stdout (or --out);
                                     --cost mock|historical (historical needs a scope line)
    ghav emit WORKFLOW               lower, infer contracts, emit again (round-trip view)
    ghav stats SCOPE_LINE            workflow statistics used by the scope builders

Contracts come from the static guide (docs/CONTRACT_GUIDE.md); ``--traces`` adds the
benchmark's observed contracts from ``resources/traces``. JSON goes to stdout with sorted keys. Exit status: 0 on success; ``check`` exits 1 when the
workflows are not equivalent; 2 on usage or input errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ghav import checker, pipeline, synth
from ghav.emit import EmitError, emit
from ghav.lower import LowerError, lower
from ghav.wir import Program


def _repo_root(workflow: Path, repo: str | None) -> Path:
    if repo:
        return Path(repo)
    for parent in Path(workflow).resolve().parents:
        if (parent / ".github").is_dir():
            return parent
    return Path.cwd()


def _resolve(arg: str, repo: str | None, resources: str | None) -> tuple[Path, Path]:
    """(workflow file, repository root) of a path or a pinned scope line."""
    try:
        bench = pipeline.parse_scope_line(arg)
    except ValueError:
        path = Path(arg)
        return path, _repo_root(path, repo)
    root = bench.snapshot(Path(resources) if resources else None)
    return root / bench.path, root


def _load(arg: str, args: argparse.Namespace, *, annotate: bool = True) -> tuple[Program, Path, object]:
    """(program, repository root, observed contracts used to annotate it)."""
    wf, root = _resolve(arg, getattr(args, "repo", None), getattr(args, "resources", None))
    p = lower(wf, root)
    if not annotate:
        return p, root, {}
    observed = pipeline.observed_for(p, root, getattr(args, "traces", False))
    return synth.annotate_like(p, root, observed), root, observed


def _print_json(obj: object) -> None:
    sys.stdout.write(pipeline.dumps(obj))


def cmd_lower(args: argparse.Namespace) -> int:
    p, _, _ = _load(args.workflow, args, annotate=False)
    _print_json(pipeline.program_json(p))
    return 0


def cmd_contracts(args: argparse.Namespace) -> int:
    p, root, observed = _load(args.workflow, args)
    _print_json(pipeline.contracts_report(p, root, observed))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    original, root, observed = _load(args.original, args)
    cand_path = Path(args.candidate)
    candidate = synth.annotate_like(lower(cand_path, root), root, observed)
    res = checker.check(original, candidate, rlimit=args.rlimit)
    _print_json(res.to_json(include_timing=args.timing))
    return 0 if res.equivalent else 1


def cmd_optimize(args: argparse.Namespace) -> int:
    p, root, observed = _load(args.workflow, args)
    overrides = {"rlimit": args.rlimit}
    if args.max_expansions is not None:
        overrides["max_expansions"] = args.max_expansions
    cfg = synth.SynthConfig.for_variant(args.variant, **overrides)
    bench = None
    if args.cost == "historical":
        try:
            bench = pipeline.parse_scope_line(args.workflow)
        except ValueError:
            raise ValueError("--cost historical needs a pinned scope line (timings are per benchmark)") from None
    model = pipeline.cost_model_for(args.cost, p, bench,
                                    Path(args.resources) if args.resources else None)
    res = synth.synthesize(p, cfg, repo_root=root, model=model, observed=observed)
    text = emit(res.program)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    report = res.to_json(include_timing=args.timing)
    if args.report:
        Path(args.report).write_text(pipeline.dumps(report), encoding="utf-8")
    summary = {k: report[k] for k in ("cost_model", "original_cost", "best_cost", "original_billed_minutes",
                                       "best_billed_minutes", "accepted", "expansions", "checked")}
    sys.stderr.write(pipeline.dumps(summary))
    return 0


def cmd_emit(args: argparse.Namespace) -> int:
    p, _, _ = _load(args.workflow, args)
    sys.stdout.write(emit(p))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    _print_json(pipeline.workflow_stats(args.scope_line,
                                        resources=Path(args.resources) if args.resources else None,
                                        use_traces=args.traces))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ghav", description="GHAV: verified optimization of GitHub Actions workflows")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--repo", help="repository root (default: nearest ancestor with .github)")
        sp.add_argument("--resources", help="resources directory for scope lines (default: ./resources)")
        sp.add_argument("--traces", action="store_true",
                        help="use the benchmark's observed contracts (resources/traces) besides the guide")

    sp = sub.add_parser("lower", help="lowered WIR as JSON")
    sp.add_argument("workflow")
    common(sp)
    sp.set_defaults(fn=cmd_lower)

    sp = sub.add_parser("contracts", help="contract audit as JSON")
    sp.add_argument("workflow")
    common(sp)
    sp.set_defaults(fn=cmd_contracts)

    sp = sub.add_parser("check", help="equivalence check of two workflows")
    sp.add_argument("original")
    sp.add_argument("candidate")
    sp.add_argument("--rlimit", type=int, default=0)
    sp.add_argument("--timing", action="store_true", help="include solver time (not deterministic)")
    common(sp)
    sp.set_defaults(fn=cmd_check)

    sp = sub.add_parser("optimize", help="synthesize an optimized workflow")
    sp.add_argument("workflow")
    sp.add_argument("--variant", default="full", choices=sorted(synth.VARIANTS))
    sp.add_argument("--max-expansions", type=int)
    sp.add_argument("--cost", default="mock", choices=pipeline.COST_BACKENDS,
                    help="cost backend (historical: scope lines with resources/timings only)")
    sp.add_argument("--rlimit", type=int, default=0)
    sp.add_argument("--out", help="write the optimized YAML here instead of stdout")
    sp.add_argument("--report", help="write the synthesis report (JSON) here")
    sp.add_argument("--timing", action="store_true", help="include wall-clock time in the report")
    common(sp)
    sp.set_defaults(fn=cmd_optimize)

    sp = sub.add_parser("emit", help="lower and emit again")
    sp.add_argument("workflow")
    common(sp)
    sp.set_defaults(fn=cmd_emit)

    sp = sub.add_parser("stats", help="workflow statistics of a scope line")
    sp.add_argument("scope_line")
    sp.add_argument("--resources", help="resources directory (default: ./resources)")
    sp.add_argument("--traces", action="store_true", help="use observed contracts (resources/traces)")
    sp.set_defaults(fn=cmd_stats)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (LowerError, EmitError, FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"ghav {args.command}: error: {exc}\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
