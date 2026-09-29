#!/usr/bin/env python3
"""Compare two output trees (e.g. two complete runs) byte by byte, except wall-clock fields.

Excluded or masked, because they hold wall-clock time (docs/ARCHITECTURE.md, determinism rule 2):

* files named ``timing*.json`` (per-benchmark ``timing_<variant>.json``, per-scope ``timing.json``);
* the ``wall_s`` column of CSV files;
* the ``Wall s`` column (the last one) of ``tables/rq3.tex``.

Every other file must be byte-identical, and both trees must contain the same files.
Exit status 0 when identical, 1 otherwise; the differing files are listed.

Usage:
  uv run python scripts/aggregate/compare_outputs.py RUN1_DIR RUN2_DIR
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

WALL_COLUMNS = ("wall_s",)


def excluded(rel: Path) -> bool:
    return rel.name.startswith("timing") and rel.suffix == ".json"


def mask_csv(text: str) -> str:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return text
    idx = [i for i, h in enumerate(rows[0]) if h in WALL_COLUMNS]
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(rows[0])
    for r in rows[1:]:
        w.writerow(["<wall>" if i in idx else v for i, v in enumerate(r)])
    return out.getvalue()


def mask_rq3_tex(text: str) -> str:
    """Replace the last cell of every data row (the wall-clock column) by a placeholder."""
    lines = []
    for line in text.splitlines():
        if line.rstrip().endswith(r"\\") and " & " in line and not line.startswith("Cost"):
            cells = line.rstrip()[:-2].rstrip().split(" & ")
            cells[-1] = "<wall>"
            line = " & ".join(cells) + r" \\"
        lines.append(line)
    return "\n".join(lines) + "\n"


def normalized(path: Path, rel: Path) -> bytes:
    data = path.read_bytes()
    if rel.suffix == ".csv":
        return mask_csv(data.decode("utf-8")).encode("utf-8")
    if rel.name == "rq3.tex":
        return mask_rq3_tex(data.decode("utf-8")).encode("utf-8")
    return data


def files(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file() and not excluded(p.relative_to(root))}


def compare(a: Path, b: Path) -> tuple[int, list[str]]:
    fa, fb = files(a), files(b)
    problems = [f"only in {a}: {p}" for p in sorted(fa - fb)] + [f"only in {b}: {p}" for p in sorted(fb - fa)]
    both = sorted(fa & fb)
    for rel in both:
        if normalized(a / rel, rel) != normalized(b / rel, rel):
            problems.append(f"differs: {rel}")
    return len(both), problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run1", type=Path)
    ap.add_argument("run2", type=Path)
    args = ap.parse_args(argv)
    n, problems = compare(args.run1, args.run2)
    for p in problems:
        print(p)
    print(f"compared {n} files (timing*.json excluded, wall_s masked): "
          f"{'identical' if not problems else f'{len(problems)} problem(s)'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
