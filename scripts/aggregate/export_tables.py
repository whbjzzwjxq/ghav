#!/usr/bin/env python3
"""Copy output/tables/*.tex into the paper repository (default ../ghav-pldi2027/tables/).

Only ``*.tex`` files are copied; files in the destination that this script did not produce are
left alone. Prints what changed.

Usage:
  uv run python scripts/aggregate/export_tables.py [--src output/tables] [--dest ../ghav-pldi2027/tables]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "output" / "tables"
DEST = REPO_ROOT.parent / "ghav-pldi2027" / "tables"


def export(src: Path, dest: Path) -> list[tuple[str, str]]:
    files = sorted(src.glob("*.tex"))
    if not files:
        raise SystemExit(f"error: no tables in {src}; run `make aggregate`")
    dest.mkdir(parents=True, exist_ok=True)
    changes = []
    for f in files:
        target = dest / f.name
        old = target.read_bytes() if target.exists() else None
        new = f.read_bytes()
        if old == new:
            changes.append((f.name, "unchanged"))
            continue
        shutil.copyfile(f, target)
        changes.append((f.name, "updated" if old is not None else "added"))
    return changes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args(argv)
    for name, what in export(args.src, args.dest):
        print(f"{what:9s} {args.dest / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
