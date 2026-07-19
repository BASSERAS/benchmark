#!/usr/bin/env python3
"""
compute_perfect_recovery.py
───────────────────────────
Row-shuffle baseline: treats a row-permuted copy of the real dataset as
"generated" and evaluates the full metric suite (A1–A34 + B curve metrics).

This gives the empirical FLOOR for every metric — the best a generative
model could score given finite-sample estimation noise.  The time dimension
is never touched; only the path-index order is permuted.

Usage
─────
    /home/tbasseras/gpu-venv/bin/python metrics/compute_perfect_recovery.py
    /home/tbasseras/gpu-venv/bin/python metrics/compute_perfect_recovery.py --no-pytorch
    /home/tbasseras/gpu-venv/bin/python metrics/compute_perfect_recovery.py --seeds 5

Output
──────
    methods/perfect_recovery/results/seed_{0..4}_metrics.json
    methods/perfect_recovery/results/metrics_summary.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO       = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ── imports ───────────────────────────────────────────────────────────────────
from metrics_np import (
    # A1–A6: distribution (MMD / SWD)
    mmd2, terminal_mmd2, increment_mmd2, volatility_mmd,
    terminal_swd, path_swd,
    # A7–A12: statistical moments + ACF over lags
    terminal_cov_error, terminal_mean_rmse,
    return_std_error, return_kurtosis_error,
    acf_error,
    # A15: Heston teacher-sigma
    teacher_sigma_metrics,
    # A16–A24: log-return / vol / predictability
    logreturn_std_error, abs_return_quantile_error,
    kurtosis_ratio, sigma_mean_error,
    acf_lag1_abs_error, acf_lag1_sq_error,
    rv_law_loss,
    # A25–A34: shape / tail / curve-derived
    mean_path_rmse, vol_path_rmse, ks_logreturns, skewness_error,
    qq_rmse, tail_qq_error, rolling_vol_ks, vol_of_vol_error,
    terminal_ks, hill_tail_index_error,
)
from stylized_metrics import compute_curve_metrics

# ── constants ─────────────────────────────────────────────────────────────────
N_SEEDS = 5
N_SUB   = 1024   # subsample for O(N²) MMD/SWD (A1–A6)
DATASET = "Heston"


def compute_one_seed(
    S_real: np.ndarray,    # (N, T)  full real dataset — "real" side
    S_shuf: np.ndarray,    # (N, T)  row-shuffled real — "generated" side
    v_shuf: np.ndarray,    # (N, T)  true variance for S_shuf rows (A33 sigma corr)
    seed:   int,
    run_pytorch: bool = True,
    device: str = "cuda",
) -> dict:
    """Compute A1–A34 + B curve metrics between S_real ("real") and S_shuf ("generated")."""
    N = len(S_real)
    real3 = S_real[:, :, None]   # (N, T, 1)
    fake3 = S_shuf[:, :, None]   # (N, T, 1)

    out: dict = {"seed": seed}

    # ── subsample for O(N²) MMD/SWD ────────────────────────────────────────────
    rng   = np.random.default_rng(seed + 100)   # independent from shuffle rng
    idx_r = rng.choice(N, size=min(N_SUB, N), replace=False)
    idx_f = rng.choice(N, size=min(N_SUB, N), replace=False)
    R = real3[idx_r]
    F = fake3[idx_f]

    lr_real = np.diff(np.log(np.maximum(real3, 1e-10)), axis=1)   # (N, T-1, 1)
    lr_fake = np.diff(np.log(np.maximum(fake3, 1e-10)), axis=1)

    # ── Fat Tail (A1–A5) ──────────────────────────────────────────────────────
    out["A1_kurtosis_error"]      = float(return_kurtosis_error(real3, fake3))
    out["A2_abs_r_q95_error"]     = float(abs_return_quantile_error(real3, fake3, q=0.95))
    out["A3_abs_r_q99_error"]     = float(abs_return_quantile_error(real3, fake3, q=0.99))
    out["A4_tail_qq_error"]       = float(tail_qq_error(real3, fake3))
    out["A5_hill_tail_index_error"] = float(hill_tail_index_error(real3, fake3))

    # ── Distribution (A6–A17) ─────────────────────────────────────────────────
    out["A6_path_mmd2"]       = float(mmd2(R, F))
    out["A7_terminal_mmd2"]   = float(terminal_mmd2(R, F))
    out["A8_increment_mmd2"]  = float(increment_mmd2(R, F))
    out["A9_volatility_mmd"]  = float(volatility_mmd(R, F))
    out["A10_terminal_swd"]   = float(terminal_swd(R, F))
    out["A11_path_swd"]       = float(path_swd(R, F))
    out["A12_rv_law_loss"]    = float(rv_law_loss(real3, fake3))
    out["A13_mean_path_rmse"] = float(mean_path_rmse(real3, fake3))
    out["A14_ks_logreturns"]  = float(ks_logreturns(real3, fake3))
    out["A15_skewness_error"] = float(skewness_error(real3, fake3))
    out["A16_qq_rmse"]        = float(qq_rmse(real3, fake3))
    out["A17_terminal_ks"]    = float(terminal_ks(real3, fake3))

    # ── Adversarial (A18): discriminative score (GRU + MLP) ────────────────────
    if run_pytorch:
        from discriminative_score import compute_discriminative_score
        lr_a = lr_real.squeeze(-1)
        lr_b = lr_fake.squeeze(-1)
        d18 = compute_discriminative_score(lr_a, lr_b, n_steps=2000, device=device)
        out["A18_disc_score_gru"] = float(d18["disc_score_gru"])
        out["A18_disc_score_mlp"] = float(d18["disc_score_mlp"])
    else:
        out["A18_disc_score_gru"] = float("nan")
        out["A18_disc_score_mlp"] = float("nan")

    # ── Predictive (A19): TSTR predictive score (GRU + MLP) ────────────────────
    if run_pytorch:
        from predictive_score import compute_predictive_score
        lr_a = lr_real.squeeze(-1)
        lr_b = lr_fake.squeeze(-1)
        d19 = compute_predictive_score(lr_a, lr_b, n_steps=5000, device=device)
        out["A19_pred_score_gru"] = float(d19["pred_score_gru"])
        out["A19_pred_score_mlp"] = float(d19["pred_score_mlp"])
    else:
        out["A19_pred_score_gru"] = float("nan")
        out["A19_pred_score_mlp"] = float("nan")

    # ── Temporal (A20–A24) ────────────────────────────────────────────────────
    out["A20_cov_error"]          = float(terminal_cov_error(real3, fake3))
    out["A21_acf_abs"]            = float(acf_error(np.abs(lr_fake), np.abs(lr_real)))
    out["A22_acf_sq"]             = float(acf_error(lr_fake ** 2, lr_real ** 2))
    out["A23_acf_lag1_abs_error"] = float(acf_lag1_abs_error(real3, fake3))
    out["A24_acf_lag1_sq_error"]  = float(acf_lag1_sq_error(real3, fake3))

    # ── Vol (A25–A32) ─────────────────────────────────────────────────────────
    out["A25_mean_rmse"]          = float(terminal_mean_rmse(real3, fake3))
    out["A26_std_error"]          = float(return_std_error(real3, fake3))
    out["A27_logreturn_std_error"] = float(logreturn_std_error(real3, fake3))
    out["A28_kurtosis_ratio"]     = float(kurtosis_ratio(real3, fake3))
    out["A29_sigma_mean_error"]   = float(sigma_mean_error(real3, fake3))
    out["A30_vol_path_rmse"]      = float(vol_path_rmse(real3, fake3))
    out["A31_rolling_vol_ks"]     = float(rolling_vol_ks(real3, fake3))
    out["A32_vol_of_vol_error"]   = float(vol_of_vol_error(real3, fake3))

    # ── Heston Spec (A33–A34): teacher-sigma ──────────────────────────────────
    corr_ts, rmse_ts = teacher_sigma_metrics(fake3, v_shuf)
    out["A33_sigma_corr"] = float(corr_ts)
    out["A34_sigma_rmse"] = float(rmse_ts)

    # ── B curve metrics (6 plots × 3 sub-metrics) ─────────────────────────────
    print("  B curve metrics ...", end=" ", flush=True)
    curve = compute_curve_metrics(S_real, S_shuf)
    out.update(curve)
    print("done")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",      type=int, default=N_SEEDS,
                        help="Number of row-shuffle seeds (default 5)")
    parser.add_argument("--no-pytorch", action="store_true",
                        help="Skip A13/A14 (no GPU needed, fast)")
    parser.add_argument("--device",     default="cuda",
                        help="PyTorch device for A13/A14 (default: cuda)")
    args = parser.parse_args()

    run_pytorch = not args.no_pytorch

    # ── load real data ─────────────────────────────────────────────────────────
    data_dir = os.path.join(REPO, "dataset", DATASET)
    S_real   = np.load(os.path.join(data_dir, "heston_S_8192x128.npy"))  # (8192, 128)
    v_real   = np.load(os.path.join(data_dir, "heston_v_8192x128.npy"))  # (8192, 128)
    N        = len(S_real)
    print(f"Loaded {DATASET}: S {S_real.shape}, v {v_real.shape}")
    if run_pytorch:
        print("A13/A14 ENABLED (use --no-pytorch to skip)")
    else:
        print("A13/A14 SKIPPED (--no-pytorch)")

    # ── output directory ───────────────────────────────────────────────────────
    out_dir = os.path.join(REPO, "methods", "perfect_recovery", "results")
    os.makedirs(out_dir, exist_ok=True)

    # ── run seeds ─────────────────────────────────────────────────────────────
    all_results = []

    for seed in range(args.seeds):
        t0 = time.time()
        print(f"\n--- Seed {seed} ---")

        rng    = np.random.default_rng(seed)
        idx    = rng.permutation(N)
        S_shuf = S_real[idx]   # same distribution, different row order
        v_shuf = v_real[idx]   # matching oracle variance (same permutation)

        d = compute_one_seed(
            S_real      = S_real,
            S_shuf      = S_shuf,
            v_shuf      = v_shuf,
            seed        = seed,
            run_pytorch = run_pytorch,
            device      = args.device,
        )
        d["compute_time_sec"] = round(time.time() - t0, 1)

        for k in ["A6_path_mmd2", "A20_cov_error", "A21_acf_abs",
                  "A14_ks_logreturns", "A17_terminal_ks"]:
            if k in d and not (isinstance(d[k], float) and np.isnan(d[k])):
                print(f"    {k}: {d[k]:.6f}")
        print(f"    elapsed: {d['compute_time_sec']}s")

        json_path = os.path.join(out_dir, f"seed_{seed}_metrics.json")
        with open(json_path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"  Saved: {json_path}")

        all_results.append(d)

    # ── metrics_summary.csv ────────────────────────────────────────────────────
    print("\nGenerating metrics_summary.csv ...")
    all_keys = sorted({
        k for d in all_results
        for k in d
        if isinstance(d[k], (int, float)) and k not in ("seed", "compute_time_sec")
    })

    rows_out = []
    for k in all_keys:
        vals = [d[k] for d in all_results if isinstance(d.get(k), (int, float))]
        if not vals:
            continue
        mean_v = float(np.nanmean(vals))
        std_v  = float(np.nanstd(vals))
        row    = {"metric": k, "mean": f"{mean_v:.6f}", "std": f"{std_v:.6f}"}
        for i, v in enumerate(vals):
            row[f"seed_{i}"] = f"{v:.6f}"
        rows_out.append(row)

    fieldnames = ["metric", "mean", "std"] + [f"seed_{i}" for i in range(args.seeds)]
    csv_path   = os.path.join(out_dir, "metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    print(f"\nDone. Results saved to {out_dir}")
    print("\n--- Perfect-recovery floors ---")
    for k in ["A6_path_mmd2", "A20_cov_error", "A21_acf_abs",
              "A33_sigma_corr", "A34_sigma_rmse",
              "A14_ks_logreturns", "A17_terminal_ks"]:
        if k in all_results[0]:
            vals = [d[k] for d in all_results if isinstance(d.get(k), (int, float))]
            m, s = float(np.nanmean(vals)), float(np.nanstd(vals))
            print(f"  {k:40s} {m:.6f} ± {s:.6f}")


if __name__ == "__main__":
    main()
