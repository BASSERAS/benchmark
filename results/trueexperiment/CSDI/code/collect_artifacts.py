#!/usr/bin/env python3
"""Verify every CSDI bank against the guideline section 4 contract, then tabulate cost.

THIS IS A GATE, NOT A REPORT
----------------------------
Guideline section 4 requires that the shape, dtype, S0 anchor, finiteness and
positivity be **recomputed from the .npy**, and that a violation exit non-zero.
The SBTS sibling of this script reads `metadata.json` and stops there, which
cannot catch the failure mode that actually happens: the array on disk is
truncated, half-overwritten by a second run, or stale from a previous build,
while the JSON beside it still describes the bank the author meant to write.
Metadata is the author's claim; the `.npy` is the evidence. This script checks
the evidence and treats the claim as a second, independent witness -- when the
two disagree that is itself a failure, because one of them is feeding the
metric scripts a lie.

The arrays are opened with `mmap_mode="r"`, so a 6144x128x8 float64 bank (50 MB)
or the 8192-path pool (67 MB) is streamed rather than resident. Reductions like
`Xg.min()` still touch every page, but sequentially and without a second copy.

WHAT IS CHECKED, AND WHY EACH ONE HAS BURNED SOMEBODY
-----------------------------------------------------
  1. shape is 3-D (N, T, d)               -- a (N, T) bank from a d=1 script
                                             broadcasts silently in the metrics.
  2. filename NxTxd tag == actual shape   -- the tag is what downstream globs
                                             match on; a stale tag loads the
                                             wrong file with no error.
  3. dtype is float64                     -- float32 banks change the 6th digit
                                             of vol_err and are not comparable
                                             with the other methods' tables.
  4. S[:, 0, :] == 100.0 exactly          -- section 4. Not "close to": the
                                             generators assign it, so anything
                                             else means the file is not what
                                             the generator wrote.
  5. all finite and strictly positive     -- a single inf makes every aggregate
                                             nan; a single non-positive price
                                             makes log-returns nan.
  6. metadata agrees with the array       -- see above.
  7. metadata["seed"] == directory seed   -- catches a bank written into the
                                             wrong seed_N/ folder, which
                                             quietly turns 4 seeds into 3.

Usage
-----
  R=results/trueexperiment/CSDI
  /home/tbasseras/.cc-venv/bin/python collect_artifacts.py --results-dir $R
  echo $?   # 0 = every bank honours the contract; 1 = at least one does not
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.dirname(HERE)

S0 = 100.0
TAG_RE = re.compile(r"generated_paths_(\d+)x(\d+)x(\d+)\.npy$")
FIELDS = ["seed", "role", "n_paths", "seq_len", "d",
          "train_min", "gen_min", "epochs", "params", "sec_per_epoch"]


def check_bank(npy_path, meta, seed_from_dir, role, problems):
    """Recompute the section 4 contract from `npy_path`. Returns a CSV row dict.

    Every failure is appended to `problems` rather than raised, so one run
    reports all of them instead of making the reader fix them one per
    invocation.
    """
    where = os.path.relpath(npy_path, DEFAULT_RESULTS)
    Xg = np.load(npy_path, mmap_mode="r")

    if Xg.ndim != 3:
        problems.append(f"{where}: ndim={Xg.ndim}, expected 3 (N, T, d)")
        return None
    n, t, d = (int(v) for v in Xg.shape)

    m = TAG_RE.search(os.path.basename(npy_path))
    if m:
        tag = tuple(int(v) for v in m.groups())
        if tag != (n, t, d):
            problems.append(f"{where}: filename says {tag} but array is {(n, t, d)} "
                            f"-- downstream globs match on the filename")
    else:
        problems.append(f"{where}: filename does not carry an NxTxd tag")

    if Xg.dtype != np.float64:
        problems.append(f"{where}: dtype={Xg.dtype}, expected float64")

    first = np.asarray(Xg[:, 0, :])
    if not np.all(first == S0):
        worst = float(np.abs(first - S0).max())
        problems.append(f"{where}: S[:, 0, :] != {S0} exactly "
                        f"(max deviation {worst:.3e}) -- section 4")

    finite = bool(np.isfinite(Xg).all())
    smin, smax = float(np.min(Xg)), float(np.max(Xg))
    if not finite:
        problems.append(f"{where}: contains non-finite entries")
    if smin <= 0.0:
        problems.append(f"{where}: min price {smin:.6g} <= 0 -- log-returns are nan")

    # Independent-witness cross-check. Tolerances are float-print tolerances,
    # not modelling tolerances: the metadata numbers were written from the same
    # array by the generator, so anything beyond round-trip noise means the
    # file changed after the JSON was written.
    if meta is not None:
        if list(meta.get("shape", [])) != [n, t, d]:
            problems.append(f"{where}: metadata shape {meta.get('shape')} != array {(n, t, d)}")
        if str(meta.get("dtype", "")) != str(Xg.dtype):
            problems.append(f"{where}: metadata dtype {meta.get('dtype')!r} != array {Xg.dtype}")
        for key, got in (("S_min", smin), ("S_max", smax)):
            claimed = meta.get(key)
            if claimed is not None and not np.isclose(float(claimed), got, rtol=1e-9, atol=0.0):
                problems.append(f"{where}: metadata {key}={claimed!r} != recomputed {got!r} "
                                f"-- the .npy changed after metadata.json was written")
        if meta.get("seed") is not None and int(meta["seed"]) != seed_from_dir:
            problems.append(f"{where}: metadata seed={meta['seed']} but lives in "
                            f"seed_{seed_from_dir}/ -- bank written to the wrong folder")
        if meta.get("S0_exact") is False:
            problems.append(f"{where}: metadata records S0_exact=false")
        if meta.get("all_finite_positive") is False:
            problems.append(f"{where}: metadata records all_finite_positive=false")
        nonpos_s0 = meta.get("n_nonpositive_s0_before_rescale")
        if nonpos_s0:
            print(f"  ! {where}: {nonpos_s0} non-positive FIRST prices were clipped before "
                  f"rescaling -- each rescales its whole path by ~1e8, see rescale_to_s0",
                  file=sys.stderr)

    meta = meta or {}
    gen_sec = meta.get("gen_time_sec")
    train_sec = meta.get("train_time_sec")
    return {
        "seed": seed_from_dir, "role": role,
        "n_paths": n, "seq_len": t, "d": d,
        "train_min": round(train_sec / 60.0, 1) if train_sec else "",
        "gen_min": round(gen_sec / 60.0, 1) if gen_sec else "",
        "epochs": meta.get("epochs_run", ""),
        "params": meta.get("params", ""),
        "sec_per_epoch": meta.get("sec_per_epoch") or "",
    }


def scan(root, role, problems, rows):
    """Walk `root`/seed_*/ and check every bank found. Ignores --tag probe runs."""
    if not os.path.isdir(root):
        return
    for d_seed in sorted(glob.glob(os.path.join(root, "seed_*"))):
        try:
            seed = int(os.path.basename(d_seed).split("_")[1])
        except (IndexError, ValueError):
            problems.append(f"{d_seed}: cannot parse a seed from the directory name")
            continue
        npys = sorted(p for p in glob.glob(os.path.join(d_seed, "generated_paths_*.npy"))
                      if TAG_RE.search(os.path.basename(p)))
        if not npys:
            problems.append(f"{d_seed}: no generated_paths_*.npy")
            continue
        if len(npys) > 1:
            problems.append(f"{d_seed}: {len(npys)} banks present ({[os.path.basename(p) for p in npys]}) "
                            f"-- ambiguous, the metric scripts glob and take the first")
        meta_path = os.path.join(d_seed, "metadata.json")
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                meta = json.load(fh)
        else:
            problems.append(f"{d_seed}: no metadata.json -- provenance unrecoverable")
        for npy in npys:
            row = check_bank(npy, meta, seed, role, problems)
            if row is not None:
                rows.append(row)
                print(f"  ok {os.path.relpath(npy, DEFAULT_RESULTS)}  "
                      f"({row['n_paths']}x{row['seq_len']}x{row['d']}, {role})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS)
    ap.add_argument("--expect-seeds", type=int, default=5,
                    help="how many A/B-bank seeds must be present (0 disables the check)")
    a = ap.parse_args()

    problems, rows = [], []
    print(f"[collect] scanning {a.results_dir}", flush=True)
    scan(os.path.join(a.results_dir, "generated_paths"), "ab_bank", problems, rows)
    scan(os.path.join(a.results_dir, "crps_banks", "generated_paths"), "crps_pool", problems, rows)

    if not rows:
        raise SystemExit("ABORT: no banks found -- nothing to verify")

    ab_seeds = sorted({r["seed"] for r in rows if r["role"] == "ab_bank"})
    if a.expect_seeds and len(ab_seeds) != a.expect_seeds:
        problems.append(f"expected {a.expect_seeds} A/B-bank seeds, found {len(ab_seeds)}: {ab_seeds}")

    # All A/B banks must agree on (T, d) or the A-table is averaging over
    # different objects. N may legitimately differ between roles, not within one.
    for role in ("ab_bank", "crps_pool"):
        shapes = {(r["n_paths"], r["seq_len"], r["d"]) for r in rows if r["role"] == role}
        if len(shapes) > 1:
            problems.append(f"{role}: seeds disagree on shape: {sorted(shapes)}")

    # Cross-check the recorded parameter count against the saved configs. A
    # differing count between seeds means they are not the same architecture.
    cfgs = sorted(glob.glob(os.path.join(a.results_dir, "weights", "seed_*_config.json")))
    nparams = set()
    for c in cfgs:
        with open(c) as fh:
            nparams.add(json.load(fh).get("params"))
    if len(nparams) > 1:
        problems.append(f"weights/*_config.json disagree on params: {sorted(nparams)} "
                        f"-- the seeds are not the same model")

    losses_dir = os.path.join(a.results_dir, "losses")
    os.makedirs(losses_dir, exist_ok=True)
    out_csv = os.path.join(losses_dir, "generation_time.csv")
    rows.sort(key=lambda r: (r["role"], r["seed"]))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[collect] wrote {os.path.relpath(out_csv, DEFAULT_RESULTS)} ({len(rows)} rows)", flush=True)

    if problems:
        print(f"\nFAIL: {len(problems)} contract violation(s)", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[collect] PASS: {len(rows)} bank(s) honour the section 4 contract", flush=True)


if __name__ == "__main__":
    main()
