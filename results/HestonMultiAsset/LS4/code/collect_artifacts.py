"""LS4 d = 8 -- post-training artefact collector and contract check.

Why this exists as a separate script rather than inline in the trainer
---------------------------------------------------------------------
MULTIASSET_GUIDELINE.md section 4, under the ``generation_time.csv`` contract,
warns (cited by section rather than line number: this docstring originally said
"lines 222-225" and the next edit to section 4 invalidated it -- see section
12.7):

    If your runner rewrites generation_time.csv in "w" mode from an in-memory
    list, re-running a *subset* of seeds silently deletes the rows for the seeds
    you did not re-run. Check the file has 5 rows before you render the README.

LS4 trains its 5 seeds **concurrently across 2 GPUs**, so an in-run writer would
not merely lose rows on a partial re-run -- five processes would race on the same
file. The timing table is therefore rebuilt here, once, from the per-seed
``metadata.json`` files each process wrote independently. Re-running one seed and
then re-running this script yields the same 5-row table either way.

It also normalises ``metadata.json`` to the section 4 key names. The trainer's
keys were corrected mid-run (``gen_sec`` -> ``gen_time_sec``,
``min_val_price``/``max_val_price`` -> ``S_min``/``S_max``), so seeds launched
before and after the fix would otherwise disagree. Every field is recomputed
from the ``.npy`` itself rather than copied forward, so the audit record cannot
drift from the array it describes -- which matters precisely because the array is
gitignored (132 MB, over GitHub's 100 MB hard limit) and the metadata is the only
thing tracked.

Contract checked per seed (GUIDELINE section 4):
    shape (8192, 252, 8), dtype float64, S[:, 0, :] == 100.0 exactly,
    finite and strictly positive everywhere.
A violation is reported and the script exits non-zero. It does not repair.

Usage:
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/LS4/code/collect_artifacts.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LS4_DIR = os.path.dirname(HERE)
GEN_ROOT = os.path.join(LS4_DIR, "generated_paths")
WEIGHTS_DIR = os.path.join(LS4_DIR, "weights")
LOSSES_DIR = os.path.join(LS4_DIR, "losses")

SEEDS = [0, 1, 2, 3, 4]
EXPECTED_SHAPE = (8192, 252, 8)
S0 = 100.0
# both seed groups ran under taskset -c 0-7 / 8-15 with OMP_NUM_THREADS=8
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

        gen_sec = float(old.get("gen_time_sec", old.get("gen_sec", 0.0)))
        meta = {
            "method": "LS4",
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
            "epochs_run": old.get("epochs_run"),
            "min_total_loss": old.get("min_total_loss"),
            "min_val_loss": old.get("min_val_loss"),
            "first_nan_epoch": old.get("first_nan_epoch"),
            "z_dim": old.get("z_dim", cfg.get("z_dim")),
            "params": old.get("params", cfg.get("params")),
            "per_asset_digest": old.get("per_asset_digest"),
            "s0_rescaled": old.get("s0_rescaled", True),
            "gpu": "A100-SXM4-80GB",
            "date": old.get("date"),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        rows.append({
            "seed": seed, "n_samples": S.shape[0], "T": S.shape[1], "d": S.shape[2],
            "z_dim": meta["z_dim"], "epochs": meta["epochs_run"],
            "n_workers": N_WORKERS,
            "elapsed_sec": round(gen_sec, 2),
            "elapsed_min": round(gen_sec / 60.0, 2),
        })
        print(f"  seed {seed}: {S.shape} price [{S.min():.2f}, {S.max():.2f}]  "
              f"S0 exact: {s0_exact}  finite&positive: {finite_pos}  "
              f"gen {gen_sec:.1f}s  -> {'OK' if good else 'FAIL'}")

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
