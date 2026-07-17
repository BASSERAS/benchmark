#!/usr/bin/env python3
"""
Perfect-recovery baseline: compute all A1-A20 metrics on two independent
random halves of the real dataset.

Usage
-----
    python metrics/perfect_recovery.py --dataset Heston
    # Output: results/Heston/perfect_recovery.json

What it computes
----------------
Split real data (8 192 paths) into two halves of 4 096, compute all A1-A20
metrics + B1-B14 stylized metrics treating one half as "real" and the other as "generated".
Repeat N_RUNS=5 times with different random seeds.

This gives the **empirical floor** for each metric — what a perfect generative
model would achieve (i.e., the noise floor from finite-sample estimation).

A15 interpretation
------------------
- Sigma Corr floor ≈ 0.505 (not 1.0): quadratic variation is a noisy
  estimator of instantaneous variance.  Two halves of real data share the
  same underlying variance path but have independent return noise, so their
  QV-based sigma estimates correlate at ~0.5, not 1.0.
- Sigma RMSE floor ≈ 1.054: both SBTS and TimeGAN score BELOW this floor
  due to variance compression in their generated paths.  A lower-than-floor
  Sigma RMSE is a warning sign (variance compression), not a win.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ── resolve project root ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "metrics"))

from metrics_np import (
    mmd2,
    terminal_mmd2,
    increment_mmd2,
    volatility_mmd,
    terminal_swd,
    path_swd,
    terminal_cov_error,
    terminal_mean_rmse,
    return_std_error,
    return_kurtosis_error,
    acf_error,
    teacher_sigma_metrics,
    tail_survival_error,
    rv_law_loss,
    oracle_metrics,
)
from discriminative_score import compute_discriminative_score
from predictive_score import compute_predictive_score
from stylized_metrics import compute_stylized_metrics

# ── constants ─────────────────────────────────────────────────────────────────
N_RUNS = 5      # number of random half-splits
N_SUB  = 1024   # subsample size for MMD / SWD (A1-A6) to keep O(N²) tractable
DEVICE = "cuda" # for A13 / A14 (falls back to cpu if cuda unavailable)


# ── single-run computation ────────────────────────────────────────────────────

def compute_one_run(
    S_a: np.ndarray,   # (N//2, T) — treated as "real"
    S_b: np.ndarray,   # (N//2, T) — treated as "generated"
    v_a: np.ndarray,   # (N//2, T) — true variance matching S_b rows (NOT S_a rows!)
    rng: np.random.Generator,
    run_idx: int = 0,  # seed for oracle_metrics
) -> dict:
    """Compute all A1-A20 metrics + B1-B14 stylized metrics between S_a ("real") and S_b ("generated")."""
    N = len(S_a)

    # 3D view needed by most metric functions
    real3 = S_a[:, :, None]   # (N, T, 1)
    fake3 = S_b[:, :, None]   # (N, T, 1)

    # ── A1-A6: subsample to keep MMD/SWD tractable ───────────────────────────
    idx_r = rng.choice(N, size=min(N_SUB, N), replace=False)
    idx_f = rng.choice(N, size=min(N_SUB, N), replace=False)
    R = real3[idx_r]  # (N_SUB, T, 1)
    F = fake3[idx_f]  # (N_SUB, T, 1)

    out = {}
    out["A1_path_mmd2"]      = float(mmd2(R, F))
    out["A2_terminal_mmd2"]  = float(terminal_mmd2(R, F))
    out["A3_increment_mmd2"] = float(increment_mmd2(R, F))
    out["A4_volatility_mmd"] = float(volatility_mmd(R, F))
    out["A5_terminal_swd"]   = float(terminal_swd(R, F))
    out["A6_path_swd"]       = float(path_swd(R, F))

    # ── A7-A12: use full half-dataset ─────────────────────────────────────────
    out["A7_cov_error"]       = float(terminal_cov_error(real3, fake3))
    out["A8_mean_rmse"]       = float(terminal_mean_rmse(real3, fake3))
    out["A9_std_error"]       = float(return_std_error(real3, fake3))
    out["A10_kurtosis_error"] = float(return_kurtosis_error(real3, fake3))

    _lr_real = np.diff(np.log(real3), axis=1)  # log-returns (correct for ARCH/ACF)
    _lr_fake = np.diff(np.log(fake3), axis=1)
    out["A11_acf_abs"] = float(acf_error(np.abs(_lr_fake), np.abs(_lr_real)))
    out["A12_acf_sq"]  = float(acf_error(_lr_fake ** 2, _lr_real ** 2))

    # ── A13: discriminative score (GRU + MLP, GPU) ────────────────────────────
    _lr_a = np.diff(np.log(S_a[:, :, None]), axis=1).squeeze(-1)
    _lr_b = np.diff(np.log(S_b[:, :, None]), axis=1).squeeze(-1)
    d13 = compute_discriminative_score(_lr_a, _lr_b, n_steps=2000, device=DEVICE)
    out["A13_disc_gru"] = float(d13["disc_score_gru"])
    out["A13_disc_mlp"] = float(d13["disc_score_mlp"])

    # ── A14: predictive score TSTR (GRU + MLP, GPU) ──────────────────────────
    d14 = compute_predictive_score(_lr_a, _lr_b, n_steps=5000, device=DEVICE)
    out["A14_pred_gru"] = float(d14["pred_score_gru"])
    out["A14_pred_mlp"] = float(d14["pred_score_mlp"])

    # ── A15: teacher-sigma (Heston-specific) ──────────────────────────────────
    corr15, rmse15 = teacher_sigma_metrics(fake3, v_a)
    out["A15_sigma_corr"] = float(corr15)
    out["A15_sigma_rmse"] = float(rmse15)

    # ── A16: tail survival error ───────────────────────────────────────────────
    a16_rms, a16_q90, a16_q95, a16_q99 = tail_survival_error(real3, fake3)
    out["A16_tail_survival"] = a16_rms
    out["A16_q90_error"]     = a16_q90
    out["A16_q95_error"]     = a16_q95
    out["A16_q99_error"]     = a16_q99

    # A17-A19 Oracle AR(5) TSTR
    o_mean, a_mean, oa_corr = oracle_metrics(real3, fake3, ar_order=5, seed=run_idx)
    out["A17_oracle_mae"] = o_mean
    out["A18_agent_mae"]  = a_mean
    out["A19_oa_corr"]    = oa_corr

    # A20 RV Law Loss
    out["A20_rv_law_loss"] = rv_law_loss(real3, fake3)

    # B1-B14 Stylized metrics
    stylized = compute_stylized_metrics(S_a, S_b)
    out.update(stylized)

    return out


# ── summary statistics ────────────────────────────────────────────────────────

def summarise(per_run: list) -> dict:
    keys = [k for k in per_run[0] if k != "elapsed_sec"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in per_run]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "runs": vals,
        }
    summary["n_runs"]      = N_RUNS
    summary["description"] = (
        f"Perfect recovery: real vs real (independent halves), {N_RUNS} random splits"
    )
    return summary


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute perfect-recovery baseline for a dataset."
    )
    parser.add_argument("--dataset", default="Heston",
                        help="Dataset name (must match dataset/<dataset>/ folder)")
    args = parser.parse_args()

    dataset  = args.dataset
    data_dir = os.path.join(ROOT, "dataset", dataset)
    out_dir  = os.path.join(ROOT, "results", dataset)
    os.makedirs(out_dir, exist_ok=True)

    # ── load real data ────────────────────────────────────────────────────────
    S_all = np.load(os.path.join(data_dir, "heston_S_8192x128.npy"))  # (8192, 128)
    v_all = np.load(os.path.join(data_dir, "heston_v_8192x128.npy"))  # (8192, 128)
    N     = len(S_all)
    half  = N // 2
    print(f"Loaded {dataset}: S {S_all.shape}, v {v_all.shape}")

    # ── N_RUNS splits ─────────────────────────────────────────────────────────
    per_run = []
    for run in range(N_RUNS):
        t0  = time.time()
        rng = np.random.default_rng(run)
        idx = rng.permutation(N)
        idx_a, idx_b = idx[:half], idx[half:]

        S_a = S_all[idx_a]   # (4096, 128) — "real"
        S_b = S_all[idx_b]   # (4096, 128) — "generated"
        # A15: teacher_sigma_metrics compares S_b (generated) to its own variance path.
        # Must use v_b (variance for S_b rows), NOT v_a (variance for S_a rows).
        v_b = v_all[idx_b]   # (4096, 128) — variance for A15, same rows as S_b

        result = compute_one_run(S_a, S_b, v_b, rng, run_idx=run)
        result["elapsed_sec"] = round(time.time() - t0, 1)

        print(
            f"  run {run} done in {result['elapsed_sec']}s: "
            f"A1={result['A1_path_mmd2']:.4f} "
            f"A7={result['A7_cov_error']:.2f} "
            f"A13_gru={result['A13_disc_gru']:.4f}"
        )
        per_run.append(result)

    # ── save ──────────────────────────────────────────────────────────────────
    summary  = summarise(per_run)
    out_path = os.path.join(out_dir, "perfect_recovery.json")
    with open(out_path, "w") as f:
        json.dump({"per_run": per_run, "summary": summary}, f, indent=2)

    print(f"\nDone. Saved to {out_path}")
    print(f"  A1  MMD²         floor: {summary['A1_path_mmd2']['mean']:.4f} ± {summary['A1_path_mmd2']['std']:.4f}")
    print(f"  A7  Cov Error    floor: {summary['A7_cov_error']['mean']:.2f} ± {summary['A7_cov_error']['std']:.2f}")
    print(f"  A13 Disc GRU     floor: {summary['A13_disc_gru']['mean']:.4f} ± {summary['A13_disc_gru']['std']:.4f}")
    print(f"  A15 Sigma Corr   floor: {summary['A15_sigma_corr']['mean']:.4f} ± {summary['A15_sigma_corr']['std']:.4f}")
    print(f"  A15 Sigma RMSE   floor: {summary['A15_sigma_rmse']['mean']:.4f} ± {summary['A15_sigma_rmse']['std']:.4f}")
    print(f"  A16 Tail Surv    floor: {summary['A16_tail_survival']['mean']:.4f} ± {summary['A16_tail_survival']['std']:.4f}")


if __name__ == "__main__":
    main()
