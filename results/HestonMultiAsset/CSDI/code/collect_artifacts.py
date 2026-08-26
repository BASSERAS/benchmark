"""CSDI d = 8 -- post-training artefact collector and contract check.

Adapted from ``LS4/code/collect_artifacts.py``; MULTIASSET_GUIDELINE.md section 4
tells every method to copy it, and the reason applies here unchanged.

The guideline warns that a runner rewriting ``generation_time.csv`` in ``"w"`` mode
from an in-memory list silently deletes the rows for seeds it did not re-run. CSDI
trains its 5 seeds **concurrently across 2 GPUs**, so an in-run writer would not
merely lose rows -- live processes would race on the same file. No seed process
writes a shared path at all; the timing table is rebuilt here, once, from the
per-seed ``metadata.json`` files each process wrote independently. Re-running one
seed and then re-running this script yields the same 5-row table either way.

Every field is recomputed from the ``.npy`` itself rather than copied forward, so
the audit record cannot drift from the array it describes -- which matters
precisely because the array is gitignored (132 MB, over GitHub's 100 MB hard limit)
and the metadata is the only thing tracked.

Contract checked per seed (GUIDELINE section 4):
    shape (8192, 252, 8), dtype float64, S[:, 0, :] == 100.0 exactly,
    finite and strictly positive everywhere.
A violation is reported and the script exits non-zero. It does not repair. Wire the
exit code so a violation stops the chain *before* compute_all_multiasset.py, which
costs 55 minutes and produces normal-looking numbers on a broken array.

Usage:
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/CSDI/code/collect_artifacts.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSDI_DIR = os.path.dirname(HERE)
GEN_ROOT = os.path.join(CSDI_DIR, "generated_paths")
WEIGHTS_DIR = os.path.join(CSDI_DIR, "weights")
LOSSES_DIR = os.path.join(CSDI_DIR, "losses")

SEEDS = [0, 1, 2, 3, 4]
EXPECTED_SHAPE = (8192, 252, 8)
S0 = 100.0
# every seed ran under taskset -c 16-23 / 24-31 with OMP_NUM_THREADS=8
N_WORKERS = 8


def main():
    rows, ok = [], True
    for seed in SEEDS:
        gen_dir = os.path.join(GEN_ROOT, f"seed_{seed}")
        npy = os.path.join(gen_dir, "generated_paths_8192x252x8.npy")
        meta_path = os.path.join(gen_dir, "metadata.json")
        if not (os.path.exists(npy) and os.path.exists(meta_path)):
            print(f"  seed {seed}: MISSING artefacts -> FAIL")
            ok = False
            continue

        S = np.load(npy)
        with open(meta_path) as f:
            old = json.load(f)
        cfg_path = os.path.join(WEIGHTS_DIR, f"seed_{seed}_config.json")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)

        s0_exact = bool(np.all(S[:, 0, :] == S0))
        finite_pos = bool(np.isfinite(S).all() and (S > 0).all())
        good = (S.shape == EXPECTED_SHAPE and S.dtype == np.float64
                and s0_exact and finite_pos)
        ok &= good

        gen_sec = float(old.get("gen_time_sec", 0.0))
        meta = {
            "method": "CSDI",
            "seed": seed,
            "shape": list(S.shape),
            "dtype": str(S.dtype),
            "d": int(S.shape[2]),
            "S0": S0,
            "S0_exact": s0_exact,
            "S_min": float(S.min()),
            "S_max": float(S.max()),
            "all_finite_positive": finite_pos,
            "generated_mean": float(S.mean()),
            "generated_std": float(S.std()),
            "gen_time_sec": round(gen_sec, 1),
            "train_time_sec": float(old.get("train_time_sec", 0.0)),
            "sec_per_epoch": old.get("sec_per_epoch"),
            "epochs_run": old.get("epochs_run", cfg.get("epochs")),
            "batch_size": old.get("batch_size", cfg.get("batch_size")),
            "params": old.get("params", cfg.get("params")),
            "num_steps": old.get("num_steps", cfg.get("num_steps")),
            "min_total_loss": old.get("min_total_loss"),
            "min_val_loss": old.get("min_val_loss"),
            "first_nan_step": old.get("first_nan_step"),
            "raw_price_min": old.get("raw_price_min"),
            "raw_price_max": old.get("raw_price_max"),
            "n_nonpositive_s0_before_rescale": old.get("n_nonpositive_s0_before_rescale"),
            "n_nonpositive_total_before_rescale": old.get("n_nonpositive_total_before_rescale"),
            "n_entries": old.get("n_entries", int(S.size)),
            "s0_rescaled": old.get("s0_rescaled", True),
            "gpu": "A100-SXM4-80GB",
            "date": old.get("date"),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        rows.append({
            "seed": seed, "n_samples": S.shape[0], "T": S.shape[1], "d": S.shape[2],
            "epochs": meta["epochs_run"], "batch_size": meta["batch_size"],
            "num_steps": meta["num_steps"], "n_workers": N_WORKERS,
            "elapsed_sec": round(gen_sec, 2),
            "elapsed_min": round(gen_sec / 60.0, 2),
        })
        clipped = meta["n_nonpositive_total_before_rescale"]
        print(f"  seed {seed}: {S.shape} price [{S.min():.2f}, {S.max():.2f}]  "
              f"S0 exact: {s0_exact}  finite&positive: {finite_pos}  "
              f"clipped<=0: {clipped}  gen {gen_sec:.1f}s  -> {'OK' if good else 'FAIL'}")

    if not rows:
        raise SystemExit("ABORT: no seeds found -- run train_multiasset.py first.")

    os.makedirs(LOSSES_DIR, exist_ok=True)
    dst = os.path.join(LOSSES_DIR, "generation_time.csv")
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {dst}  ({len(rows)} rows)")

    if len(rows) != len(SEEDS):
        print(f"WARNING: {len(rows)} rows, expected {len(SEEDS)} -- "
              f"do NOT render the README until this is 5.")
        ok = False

    print("All seeds satisfy the section 4 contract." if ok
          else "CONTRACT VIOLATION -- see FAIL rows above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
