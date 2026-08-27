#!/usr/bin/env python
"""Concatenate the per-seed generation-time CSVs into ``generation_time.csv``.

Why this file exists
--------------------
``generate_reference_true.py`` writes ``generation_time_seed_<N>.csv`` when it is
invoked with exactly one seed, and ``generation_time.csv`` when it is invoked
with several.  The five reference banks on this panel are produced as five
CONCURRENT one-seed processes -- the workload is a 127-step Euler loop that
saturates a single core, so five cores finish in the wall time of one -- and five
processes writing one shared CSV would be five writers racing on one path.  The
last to close the file would win and the other four rows would be lost, silently:
the file would still parse, still have a header, and still contain a plausible
row.  Nothing downstream could detect it.

So each process writes its own file and this script joins them afterwards.  The
alternative -- a lock around the shared CSV -- would serialise the only part of
the run that is worth parallelising and would still leave a partial file if a
process died holding the lock.

What it refuses to do
---------------------
It does not invent rows.  If a seed's file is missing the script says which one
and exits non-zero rather than emitting a four-row CSV that looks complete.  A
generation-time table with a seed quietly absent is worse than no table, because
the README renderer would report a mean over four seeds and label it five.

It also does not accept two files claiming the same seed, or files that disagree
on ``n_samples``/``T``/``d``: those mean the banks were generated under different
settings and the per-seed times are not comparable, which is the only reason to
put them in one table.

Usage
-----
    python merge_generation_time.py                 # default losses dir
    python merge_generation_time.py --seeds 0 1 2   # expect exactly these
    python merge_generation_time.py --losses-dir /path/to/losses
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHOD_ROOT = HERE.parent

FIELDS = ["seed", "n_samples", "T", "d", "device", "elapsed_sec", "elapsed_min"]

# The columns that describe WHAT was generated rather than how long it took.
# Rows that disagree on these are not rows of the same table.
SHAPE_FIELDS = ("n_samples", "T", "d")


def _read_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise SystemExit(
            f"ABORT: {path.name} holds {len(rows)} rows, expected exactly 1. "
            f"This file is written by a single-seed invocation of "
            f"generate_reference_true.py; more than one row means it was not."
        )
    row = rows[0]
    missing = [f for f in FIELDS if f not in row or row[f] is None]
    if missing:
        raise SystemExit(f"ABORT: {path.name} is missing column(s) {missing}")
    return row


def merge(losses_dir: Path, seeds: list[int]) -> Path:
    rows: list[dict[str, str]] = []
    absent: list[int] = []
    for seed in seeds:
        path = losses_dir / f"generation_time_seed_{seed}.csv"
        if not path.exists():
            absent.append(seed)
            continue
        rows.append(_read_one(path))

    if absent:
        raise SystemExit(
            f"ABORT: no generation-time file for seed(s) {absent} in {losses_dir}. "
            f"Those seeds did not finish, or were never run. Refusing to write a "
            f"{len(rows)}-seed CSV that downstream code would read as complete -- "
            f"re-run generate_reference_true.py for the missing seed(s) first."
        )

    seen = [int(r["seed"]) for r in rows]
    duplicates = sorted({s for s in seen if seen.count(s) > 1})
    if duplicates:
        raise SystemExit(f"ABORT: seed(s) {duplicates} appear in more than one file")

    for field in SHAPE_FIELDS:
        values = {r[field] for r in rows}
        if len(values) > 1:
            raise SystemExit(
                f"ABORT: seeds disagree on '{field}' ({sorted(values)}). The banks "
                f"were generated under different settings, so their generation "
                f"times do not belong in one table."
            )

    rows.sort(key=lambda r: int(r["seed"]))
    out = losses_dir / "generation_time.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row[f] for f in FIELDS})

    secs = [float(r["elapsed_sec"]) for r in rows]
    print(f"[merge] {len(rows)} seeds: {', '.join(str(s) for s in sorted(seen))}")
    print(
        f"[merge] elapsed_sec  min {min(secs):.1f}  mean {sum(secs) / len(secs):.1f}  "
        f"max {max(secs):.1f}"
    )
    print(f"[out] {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--losses-dir",
        type=Path,
        default=METHOD_ROOT / "losses",
        help="directory holding generation_time_seed_<N>.csv",
    )
    args = parser.parse_args(argv)
    if not args.losses_dir.is_dir():
        raise SystemExit(f"ABORT: {args.losses_dir} is not a directory")
    merge(args.losses_dir, sorted(set(args.seeds)))


if __name__ == "__main__":
    sys.exit(main())
