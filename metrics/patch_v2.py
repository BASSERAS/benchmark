#!/usr/bin/env python3
"""
patch_v2.py -- Add A25-A34 + new B curve metrics to existing seed JSONs.

Does NOT re-run A13/A14 (slow PyTorch training).
Reads existing seed JSONs, computes new metrics, writes back.
Also regenerates metrics_summary.csv.

Usage:
    /home/tbasseras/gpu-venv/bin/python metrics/patch_v2.py --method TimeGAN
    /home/tbasseras/gpu-venv/bin/python metrics/patch_v2.py --method SBTS
"""

import argparse, json, csv, os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO       = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from metrics_np import (
    mean_path_rmse, vol_path_rmse,
    ks_logreturns, skewness_error,
    qq_rmse, tail_qq_error,
    rolling_vol_ks, vol_of_vol_error,
    terminal_ks, hill_tail_index_error,
)
from stylized_metrics import compute_curve_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--method",  default="TimeGAN")
parser.add_argument("--dataset", default="Heston")
parser.add_argument("--seeds",   type=int, default=5)
args = parser.parse_args()

METHOD  = args.method
DATASET = args.dataset

DATASET_DIR   = os.path.join(REPO, "dataset", DATASET)
GENERATED_DIR = os.path.join(REPO, "methods", METHOD, "generated_paths")
RESULTS_DIR   = os.path.join(REPO, "results", DATASET, METHOD)

S_real = np.load(os.path.join(DATASET_DIR, "heston_S_8192x128.npy"))  # (8192, 128)
real3  = S_real[:, :, None]
print(f"Real: {S_real.shape}")

all_results = []

for seed in range(args.seeds):
    json_path = os.path.join(RESULTS_DIR, f"seed_{seed}_metrics.json")
    print(f"\n--- Seed {seed} ---")

    with open(json_path) as f:
        d = json.load(f)

    fake = np.load(os.path.join(GENERATED_DIR, f"seed_{seed}",
                                "generated_paths_8192x128.npy"))
    fake3 = fake[:, :, None]
    print(f"  Generated: {fake.shape}")

    # A25-A34
    d["A25_mean_path_rmse"]        = float(mean_path_rmse(real3, fake3))
    d["A26_vol_path_rmse"]         = float(vol_path_rmse(real3, fake3))
    d["A27_ks_logreturns"]         = float(ks_logreturns(real3, fake3))
    d["A28_skewness_error"]        = float(skewness_error(real3, fake3))
    d["A29_qq_rmse"]               = float(qq_rmse(real3, fake3))
    d["A30_tail_qq_error"]         = float(tail_qq_error(real3, fake3))
    d["A31_rolling_vol_ks"]        = float(rolling_vol_ks(real3, fake3))
    d["A32_vol_of_vol_error"]      = float(vol_of_vol_error(real3, fake3))
    d["A33_terminal_ks"]           = float(terminal_ks(real3, fake3))
    d["A34_hill_tail_index_error"] = float(hill_tail_index_error(real3, fake3))

    for k in ["A25_mean_path_rmse","A26_vol_path_rmse","A27_ks_logreturns",
              "A28_skewness_error","A29_qq_rmse","A30_tail_qq_error",
              "A31_rolling_vol_ks","A32_vol_of_vol_error","A33_terminal_ks",
              "A34_hill_tail_index_error"]:
        print(f"    {k}: {d[k]:.6f}")

    # New B curve metrics
    print("  B curve metrics ...", end=" ", flush=True)
    curve = compute_curve_metrics(S_real, fake)
    d.update(curve)
    print("done")

    with open(json_path, "w") as f:
        json.dump(d, f, indent=2)
    all_results.append(d)

# Regenerate metrics_summary.csv
print("\nRegenerating metrics_summary.csv ...")
all_keys = sorted({k for d in all_results
                   for k in d if isinstance(d[k], (int, float)) and k != "seed"})
rows_out = []
for k in all_keys:
    vals = [d[k] for d in all_results if isinstance(d.get(k), (int, float))]
    if not vals:
        continue
    mean = float(np.mean(vals))
    std  = float(np.std(vals))
    row  = {"metric": k, "mean": f"{mean:.6f}", "std": f"{std:.6f}"}
    for i, v in enumerate(vals):
        row[f"seed_{i}"] = f"{v:.6f}"
    rows_out.append(row)

fieldnames = ["metric", "mean", "std"] + [f"seed_{i}" for i in range(args.seeds)]
with open(os.path.join(RESULTS_DIR, "metrics_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows_out)
print(f"Done -- {METHOD}/{DATASET}, A25-A34 + B curve metrics added.")
