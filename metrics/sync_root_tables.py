"""Sync the three cross-method tables into the two root READMEs.

Why this exists
---------------
GUIDELINE §15.4 requires the A1-A34, B curve-shape and PS-MC cross-method tables to be
**byte-identical** in ``README.md`` and ``results/README.md``, and to come from
``render_tables.py`` rather than being hand-maintained. Adding a method changes every
row of all three tables (a new column plus a possibly-new Winner in each row), so
hand-editing them is both tedious and the obvious place for the two files to drift
apart. This script removes that failure mode.

What it touches
---------------
Only the **first three** ``<table>...</table>`` blocks of each file, in order:

    1. A1-A34 TABLE
    2. B CURVE-SHAPE TABLE
    3. PS-MC TABLE

Each README also carries a **fourth**, hand-written table (the methods / key-differences
summary). It is left alone. The script asserts each file contains exactly four blocks,
so any layout change fails loudly instead of silently clobbering the wrong one. A
``.bak`` copy is written before each file is rewritten.

The ``grid_tvd`` block that ``render_tables.py`` also emits is a visual side-check and
is not ranked, so it is not spliced in.

Usage
-----
    /home/tbasseras/gpu-venv/bin/python metrics/sync_root_tables.py
    /home/tbasseras/gpu-venv/bin/python metrics/sync_root_tables.py --check   # dry run

``--check`` exits 1 if either README is out of date, which makes it usable as a CI gate.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
RENDER = os.path.join(HERE, "render_tables.py")
TARGETS = ("README.md", os.path.join("results", "README.md"))
ORDER = ["A1-A34 TABLE", "B CURVE-SHAPE TABLE", "PS-MC TABLE"]
TABLE_RE = re.compile(r"^<table>.*?^</table>", re.S | re.M)


def render() -> dict:
    """Run render_tables.py and return {block name -> <table>...</table>}."""
    out = subprocess.run(
        [sys.executable, RENDER, "--which", "all"],
        cwd=BENCH, capture_output=True, text=True, check=True).stdout
    blocks = {}
    for part in out.split("<!-- =====")[1:]:
        name = part.split("=====")[0].strip()
        m = TABLE_RE.search(part)
        if m:
            blocks[name] = m.group(0)
    missing = [k for k in ORDER if k not in blocks]
    if missing:
        raise SystemExit(f"render_tables.py did not emit: {missing}")
    return blocks


def splice(src: str, blocks: dict, label: str) -> str:
    hits = list(TABLE_RE.finditer(src))
    if len(hits) != 4:
        raise SystemExit(
            f"{label}: expected 4 <table> blocks (3 generated + 1 hand-written), "
            f"got {len(hits)}. Layout changed -- fix this script before running it.")
    out, last = [], 0
    for i, m in enumerate(hits[:3]):
        out.append(src[last:m.start()])
        out.append(blocks[ORDER[i]])
        last = m.end()
    out.append(src[last:])
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report drift without writing; exit 1 if out of date")
    args = ap.parse_args()

    blocks = render()
    stale = []
    for rel in TARGETS:
        path = os.path.join(BENCH, rel)
        src = open(path).read()
        new = splice(src, blocks, rel)
        if new == src:
            print(f"{rel}: up to date")
            continue
        stale.append(rel)
        if args.check:
            print(f"{rel}: OUT OF DATE")
            continue
        shutil.copyfile(path, path + ".bak")
        open(path, "w").write(new)
        print(f"{rel}: 3 tables updated (backup at {rel}.bak)")

    if args.check and stale:
        print(f"\n{len(stale)} file(s) out of date; run without --check to fix.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
