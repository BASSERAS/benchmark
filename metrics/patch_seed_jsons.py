#!/usr/bin/env python3
"""patch_seed_jsons.py — Patch existing seed JSONs: rename B keys, remove old A16-A20,
compute and insert new A16-A24, regenerate metrics_summary.csv.

Usage:
  python metrics/patch_seed_jsons.py --method TimeGAN --dataset Heston
  python metrics/patch_seed_jsons.py --method SBTS    --dataset Heston
"""
import json, csv, os, sys, argparse
import numpy as np

METRICS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(METRICS_DIR)
sys.path.insert(0, METRICS_DIR)

from metrics_np import (
    logreturn_std_error, abs_return_quantile_error, kurtosis_ratio,
    sigma_mean_error, learned_oracle_sigma_corr,
    acf_lag1_abs_error, acf_lag1_sq_error, rv_law_loss,
)

parser = argparse.ArgumentParser()
parser.add_argument("--method",  default="TimeGAN")
parser.add_argument("--dataset", default="Heston")
parser.add_argument("--seeds",   type=int, default=5)
args = parser.parse_args()

DATASET_DIR   = os.path.join(REPO_ROOT, "dataset", args.dataset)
GENERATED_DIR = os.path.join(REPO_ROOT, "methods", args.method, "generated_paths")
RESULTS_DIR   = os.path.join(REPO_ROOT, "results",  args.dataset, args.method)

S  = np.load(os.path.join(DATASET_DIR, "heston_S_8192x128.npy"))
vt = np.load(os.path.join(DATASET_DIR, "heston_v_8192x128.npy"))
real3 = S[:, :, None]

B_RENAMES = {
    "B9_acf_lag1_sq":       "B8_acf_lag1_sq",
    "B11_rolling_vol_ks":   "B9_rolling_vol_ks",
    "B12_vol_of_vol_error": "B10_vol_of_vol_error",
    "B13_terminal_ks":      "B11_terminal_ks",
    "B14_tail_index_error": "B12_tail_index_error",
}
B_REMOVE   = {"B8_arch_persistence_error", "B10_garch_persistence_error"}
A_REMOVE   = {"A16_tail_survival","A16_q90_error","A16_q95_error","A16_q99_error",
              "A17_oracle_mae","A18_agent_mae","A19_oa_corr","A20_rv_law_loss"}

all_results = []
for seed in range(args.seeds):
    path = os.path.join(RESULTS_DIR, f"seed_{seed}_metrics.json")
    if not os.path.exists(path):
        print(f"MISSING {path}"); continue
    with open(path) as f:
        d = json.load(f)

    for ok, nk in B_RENAMES.items():
        if ok in d: d[nk] = d.pop(ok)
    for k in B_REMOVE: d.pop(k, None)
    for k in A_REMOVE: d.pop(k, None)

    fake  = np.load(os.path.join(GENERATED_DIR, f"seed_{seed}", "generated_paths_8192x128.npy"))
    fake3 = fake[:, :, None]

    print(f"\n--- {args.method} seed {seed} ---")
    d["A16_logreturn_std_error"]       = float(logreturn_std_error(real3, fake3))
    d["A17_abs_r_q95_error"]           = float(abs_return_quantile_error(real3, fake3, q=0.95))
    d["A18_abs_r_q99_error"]           = float(abs_return_quantile_error(real3, fake3, q=0.99))
    d["A19_kurtosis_ratio"]            = float(kurtosis_ratio(real3, fake3))
    d["A20_sigma_mean_error"]          = float(sigma_mean_error(real3, fake3))
    try:
        d["A21_learned_oracle_sigma_corr"] = float(learned_oracle_sigma_corr(fake3, vt))
    except Exception as e:
        d["A21_learned_oracle_sigma_corr"] = None; print(f"  A21 SKIP: {e}")
    d["A22_acf_lag1_abs_error"]        = float(acf_lag1_abs_error(real3, fake3))
    d["A23_acf_lag1_sq_error"]         = float(acf_lag1_sq_error(real3, fake3))
    d["A24_rv_law_loss"]               = float(rv_law_loss(real3, fake3))

    for k in ["A16_logreturn_std_error","A17_abs_r_q95_error","A18_abs_r_q99_error",
              "A19_kurtosis_ratio","A20_sigma_mean_error","A21_learned_oracle_sigma_corr",
              "A22_acf_lag1_abs_error","A23_acf_lag1_sq_error","A24_rv_law_loss"]:
        print(f"  {k} = {d[k]}")

    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    all_results.append(d)

if all_results:
    keys = [k for k in all_results[0] if k not in ("seed","compute_time_sec")]
    sumpath = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    with open(sumpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric","mean","std",
                                          "seed_0","seed_1","seed_2","seed_3","seed_4"])
        w.writeheader()
        for mk in keys:
            vals, ps = [], {}
            for r in all_results:
                v = r.get(mk); ps[f"seed_{r['seed']}"] = v
                if v is not None:
                    try: vals.append(float(v))
                    except: pass
            row = {"metric": mk,
                   "mean": round(np.mean(vals),6) if vals else None,
                   "std":  round(np.std(vals), 6) if vals else None}
            row.update(ps); w.writerow(row)
    print(f"\nSaved {sumpath}")
