"""
Perfect-recovery FLOOR for the log-return-preprocessing experiment (N=4096).

Identical methodology to metrics/compute_perfect_recovery.py, only at the 4096
sample count and pointed at dataset/Heston/preprocessing_with_log_returns/. It
reuses that script's untouched ``compute_one_seed`` (all A1-A34 + B curve code),
so the floor is directly comparable to the LS4/CSDI columns of this experiment.

The floor is a property of the DATASET, not the preprocessing: a fresh,
INDEPENDENT true-Heston draw (seed 1000+i) is scored against the held-out TEST
set (seed 1), exactly as every generator is. The log-return round-trip is exact
on genuine Heston paths (they already start at S0=100), so passing the draw
through the SBTS transform would be a no-op — the raw independent draw is the
correct, method-agnostic floor.

Usage:
  CUDA_VISIBLE_DEVICES=0 /home/tbasseras/gpu-venv/bin/python compute_perfect_4096.py --seeds 5
  CUDA_VISIBLE_DEVICES=0 /home/tbasseras/gpu-venv/bin/python compute_perfect_4096.py --seeds 5 --no-pytorch
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                          # .../perfect_recovery
EXP_DIR = os.path.dirname(HERE)                                            # .../preprocessing_with_log_returns
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(EXP_DIR)))    # repo root
METRICS_DIR = os.path.join(BENCH_ROOT, "metrics")
DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns")

N = 4096
IND_SEED_BASE = 1000

# reuse the canonical floor scorer (compute_one_seed) + the Heston generator,
# without editing either.
sys.path.insert(0, METRICS_DIR)
from compute_perfect_recovery import compute_one_seed  # noqa: E402
sys.path.insert(0, os.path.join(BENCH_ROOT, "dataset", "Heston"))
from generate_heston import generate_heston  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-pytorch", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_pytorch = not args.no_pytorch

    S_real = np.load(os.path.join(DATA_DIR, f"heston_S_test_{N}x128.npy"))
    S_disc = np.load(os.path.join(DATA_DIR, f"heston_S_disc_{N}x128.npy"))
    print(f"TEST S {S_real.shape}  DISC S {S_disc.shape}  pytorch={run_pytorch}")

    out_dir = os.path.join(HERE, "results")
    seeds_dir = os.path.join(HERE, "dataset", "seeds")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(seeds_dir, exist_ok=True)

    all_results = []
    for seed in range(args.seeds):
        t0 = time.time()
        print(f"\n--- Seed {seed} (draw {IND_SEED_BASE + seed}) ---")
        S_gen, v_gen = generate_heston(n_samples=N, seq_len=S_real.shape[1],
                                       seed=IND_SEED_BASE + seed)
        S_gen = np.asarray(S_gen); v_gen = np.asarray(v_gen)
        np.save(os.path.join(seeds_dir, f"heston_S_independent_seed{seed}.npy"), S_gen)
        np.save(os.path.join(seeds_dir, f"heston_v_independent_seed{seed}.npy"), v_gen)

        d = compute_one_seed(S_real=S_real, S_gen=S_gen, v_gen=v_gen,
                             S_disc=S_disc, seed=seed,
                             run_pytorch=run_pytorch, device=args.device)
        d["compute_time_sec"] = round(time.time() - t0, 1)
        with open(os.path.join(out_dir, f"seed_{seed}_metrics.json"), "w") as f:
            json.dump(d, f, indent=2)
        print(f"  saved seed_{seed}_metrics.json ({d['compute_time_sec']}s)")
        all_results.append(d)

    all_keys = sorted({k for d in all_results for k in d
                       if isinstance(d[k], (int, float)) and k not in ("seed", "compute_time_sec")})
    rows = []
    for k in all_keys:
        vals = [d[k] for d in all_results if isinstance(d.get(k), (int, float))]
        if not vals:
            continue
        row = {"metric": k, "mean": f"{float(np.nanmean(vals)):.6f}", "std": f"{float(np.nanstd(vals)):.6f}"}
        for i, v in enumerate(vals):
            row[f"seed_{i}"] = f"{v:.6f}"
        rows.append(row)
    fieldnames = ["metric", "mean", "std"] + [f"seed_{i}" for i in range(args.seeds)]
    with open(os.path.join(out_dir, "metrics_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nDone -> {out_dir}/metrics_summary.csv")


if __name__ == "__main__":
    main()
