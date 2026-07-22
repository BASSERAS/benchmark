"""
compute_all.py — Run all 34 A-metrics + 18 B curve metrics for each of 5 seeds.

Metrics are numbered in category-display order:
  Fat Tail (A1–A5) · Distribution (A6–A17) · Adversarial (A18, PyTorch GRU+MLP)
  · Predictive (A19, PyTorch GRU+MLP) · Temporal (A20–A24) · Vol (A25–A32)
  · Heston Spec (A33–A34).

For each seed:
  - Loads real paths from dataset/<dataset>/
  - Loads generated paths from methods/<method>/generated_paths/seed_i/
  - Computes A1–A34 (numpy + PyTorch) and B curve metrics (36 keys, numpy)
  - Saves results/<dataset>/<method>/seed_i_metrics.json
  - Generates PCA + t-SNE plots

After all seeds: writes results/<dataset>/<method>/metrics_summary.csv

All numpy metric functions live in metrics/metrics.py.
B curve metrics are computed by metrics/metrics.py::compute_curve_metrics.
"""

import json, os, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")

METRICS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(METRICS_DIR)

# ── CLI args — choose method and dataset ─────────────────────────────────
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--method",  default="TimeGAN")
_parser.add_argument("--dataset", default="Heston")
_parser.add_argument("--seeds",   type=int, default=5)
_cli, _ = _parser.parse_known_args()

METHOD      = _cli.method
DATASET     = _cli.dataset
N_SEEDS     = _cli.seeds

DATASET_DIR   = os.path.join(REPO_ROOT, "dataset", DATASET)
GENERATED_DIR = os.path.join(REPO_ROOT, "methods", METHOD, "generated_paths")
RESULTS_DIR   = os.path.join(REPO_ROOT, "results", DATASET, METHOD)
PLOTS_DIR     = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

sys.path.insert(0, METRICS_DIR)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

from metrics import (
    # A1–A6: Distribution (MMD / SWD)
    mmd2, terminal_mmd2, increment_mmd2, volatility_mmd,
    terminal_swd, path_swd,
    # A7–A12: Statistical moments + ACF over lags
    terminal_cov_error, terminal_mean_rmse,
    return_std_error, return_kurtosis_error,
    acf_error,
    # A15: Heston teacher-sigma
    teacher_sigma_metrics,
    # A16–A24: Log-return vol / predictability / realized vol
    logreturn_std_error,
    abs_return_quantile_error,
    kurtosis_ratio,
    sigma_mean_error,
    acf_lag1_abs_error,
    acf_lag1_sq_error,
    rv_law_loss,
    # A25–A34: Distributional shape / tail / curve-derived
    mean_path_rmse, vol_path_rmse, ks_logreturns, skewness_error,
    qq_rmse, tail_qq_error, rolling_vol_ks, vol_of_vol_error,
    terminal_ks, hill_tail_index_error,
)
from discriminative_score import compute_discriminative_score
from predictive_score import compute_predictive_score
from metrics import compute_curve_metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_data():
    # "real" reference for ALL metrics/plots = the held-out TEST set (seed 1),
    # independent of the seed-0 set the generators were trained on.
    S = np.load(os.path.join(DATASET_DIR, "heston_S_test_8192x128.npy"))   # (8192, 128)
    v = np.load(os.path.join(DATASET_DIR, "heston_v_test_8192x128.npy"))   # (8192, 128)
    # Note: file names are dataset-specific; update for new datasets
    return S, v


def load_disc():
    """Independent judge set (seed 2), used ONLY as the 'real' class for the A18
    discriminative score and the A19 predictive-TSTR test set — a third Heston
    draw, distinct from both the training (seed 0) and test (seed 1) sets, so the
    adversarial/predictive judges never see the exact reference used elsewhere."""
    return np.load(os.path.join(DATASET_DIR, "heston_S_disc_8192x128.npy"))   # (8192, 128)


def load_generated(seed: int) -> np.ndarray:
    path = os.path.join(GENERATED_DIR, f"seed_{seed}", "generated_paths_8192x128.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}. Run TimeGan/train.py first.")
    return np.load(path)   # (8192, 128)


def compute_metrics_for_seed(seed: int, S: np.ndarray, v: np.ndarray,
                             S_disc: np.ndarray) -> dict:
    print(f"\n=== Seed {seed} ===")
    fake = load_generated(seed)   # (8192, 128)

    # Expand to (N, T, 1) for metrics that expect 3D input
    real3 = S[:, :, None]
    fake3 = fake[:, :, None]

    results = {"seed": seed}

    t0 = time.perf_counter()

    # Subsample for expensive metrics (use 1024 points for MMD/SWD)
    N_sub = 1024
    idx_r = np.random.choice(len(S), N_sub, replace=False)
    idx_f = np.random.choice(len(fake), N_sub, replace=False)
    R = real3[idx_r]; F = fake3[idx_f]

    # Log-return views reused by several categories
    _lr_real_3d = np.diff(np.log(real3), axis=1)          # (N, T-1, 1) increments (ACF)
    _lr_fake_3d = np.diff(np.log(fake3), axis=1)
    _lr_S       = _lr_real_3d.squeeze(-1)                 # (N, T-1) for GRU/MLP
    _lr_F       = _lr_fake_3d.squeeze(-1)
    # A18/A19 use the independent judge set (seed 2) as their 'real' class, NOT
    # the test set — so the adversarial/predictive judges score against a Heston
    # draw distinct from every reference used by the A1–A17/A20–A34 + B metrics.
    _lr_disc    = np.diff(np.log(S_disc), axis=1)         # (N, T-1)
    import csv

    # ── Fat Tail (A1–A5) ──────────────────────────────────────────────────────
    print("  A1  kurtosis error ...", end=" ", flush=True)
    results["A1_kurtosis_error"]      = float(return_kurtosis_error(real3, fake3)); print(f"{results['A1_kurtosis_error']:.6f}")
    print("  A2  |r| q95 error ...", end=" ", flush=True)
    results["A2_abs_r_q95_error"]     = float(abs_return_quantile_error(real3, fake3, q=0.95)); print(f"{results['A2_abs_r_q95_error']:.6f}")
    print("  A3  |r| q99 error ...", end=" ", flush=True)
    results["A3_abs_r_q99_error"]     = float(abs_return_quantile_error(real3, fake3, q=0.99)); print(f"{results['A3_abs_r_q99_error']:.6f}")
    print("  A4  tail QQ error ...", end=" ", flush=True)
    results["A4_tail_qq_error"]       = float(tail_qq_error(real3, fake3)); print(f"{results['A4_tail_qq_error']:.6f}")
    print("  A5  Hill tail index error ...", end=" ", flush=True)
    results["A5_hill_tail_index_error"] = float(hill_tail_index_error(real3, fake3)); print(f"{results['A5_hill_tail_index_error']:.6f}")

    # ── Distribution (A6–A17) ─────────────────────────────────────────────────
    print("  A6  path MMD2 ...", end=" ", flush=True)
    results["A6_path_mmd2"]           = float(mmd2(R, F)); print(f"{results['A6_path_mmd2']:.6f}")
    print("  A7  terminal MMD2 ...", end=" ", flush=True)
    results["A7_terminal_mmd2"]       = float(terminal_mmd2(R, F)); print(f"{results['A7_terminal_mmd2']:.6f}")
    print("  A8  increment MMD2 ...", end=" ", flush=True)
    results["A8_increment_mmd2"]      = float(increment_mmd2(R, F)); print(f"{results['A8_increment_mmd2']:.6f}")
    print("  A9  volatility MMD ...", end=" ", flush=True)
    results["A9_volatility_mmd"]      = float(volatility_mmd(R, F)); print(f"{results['A9_volatility_mmd']:.6f}")
    print("  A10 terminal SWD ...", end=" ", flush=True)
    results["A10_terminal_swd"]       = float(terminal_swd(R, F)); print(f"{results['A10_terminal_swd']:.6f}")
    print("  A11 path SWD ...", end=" ", flush=True)
    results["A11_path_swd"]           = float(path_swd(R, F)); print(f"{results['A11_path_swd']:.6f}")
    print("  A12 RV law loss ...", end=" ", flush=True)
    results["A12_rv_law_loss"]        = float(rv_law_loss(real3, fake3)); print(f"{results['A12_rv_law_loss']:.6f}")
    print("  A13 mean path RMSE ...", end=" ", flush=True)
    results["A13_mean_path_rmse"]     = float(mean_path_rmse(real3, fake3)); print(f"{results['A13_mean_path_rmse']:.6f}")
    print("  A14 KS log-returns ...", end=" ", flush=True)
    results["A14_ks_logreturns"]      = float(ks_logreturns(real3, fake3)); print(f"{results['A14_ks_logreturns']:.6f}")
    print("  A15 skewness error ...", end=" ", flush=True)
    results["A15_skewness_error"]     = float(skewness_error(real3, fake3)); print(f"{results['A15_skewness_error']:.6f}")
    print("  A16 QQ RMSE (300-pt) ...", end=" ", flush=True)
    results["A16_qq_rmse"]            = float(qq_rmse(real3, fake3)); print(f"{results['A16_qq_rmse']:.6f}")
    print("  A17 terminal KS ...", end=" ", flush=True)
    results["A17_terminal_ks"]        = float(terminal_ks(real3, fake3)); print(f"{results['A17_terminal_ks']:.6f}")

    # ── Adversarial (A18) — Discriminative Score on log-returns (GRU + MLP) ────
    print("  A18 discriminative (GRU + MLP) ...", flush=True)
    d18 = compute_discriminative_score(_lr_disc, _lr_F, n_steps=2000, device=DEVICE)
    results["A18_disc_score_gru"] = d18["disc_score_gru"]
    results["A18_disc_score_mlp"] = d18["disc_score_mlp"]
    print(f"       GRU={d18['disc_score_gru']:.4f}  MLP={d18['disc_score_mlp']:.4f}")
    for arch in ("gru", "mlp"):
        loss_path = os.path.join(RESULTS_DIR, f"seed_{seed}_disc_{arch}_loss.csv")
        with open(loss_path, "w", newline="") as lf:
            w = csv.DictWriter(lf, fieldnames=["step", "train_bce"])
            w.writeheader(); w.writerows(d18[f"loss_history_{arch}"])

    # ── Predictive (A19) — TSTR Predictive Score on log-returns (GRU + MLP) ────
    print("  A19 predictive TSTR (GRU + MLP) ...", flush=True)
    d19 = compute_predictive_score(_lr_disc, _lr_F, n_steps=5000, device=DEVICE)
    results.update({f"A19_{k}": v for k, v in d19.items()
                    if not k.startswith("loss_history")})
    print(f"       GRU={d19['pred_score_gru']:.4f}  MLP={d19['pred_score_mlp']:.4f}")
    for arch in ("gru", "mlp"):
        csv_path = os.path.join(RESULTS_DIR, f"seed_{seed}_pred_{arch}_loss.csv")
        with open(csv_path, "w", newline="") as lf:
            w = csv.DictWriter(lf, fieldnames=["step", "train_mae"])
            w.writeheader(); w.writerows(d19[f"loss_history_{arch}"])

    # ── Temporal (A20–A24) ────────────────────────────────────────────────────
    print("  A20 cov error ...", end=" ", flush=True)
    results["A20_cov_error"]          = float(terminal_cov_error(real3, fake3)); print(f"{results['A20_cov_error']:.6f}")
    print("  A21 ACF abs returns ...", end=" ", flush=True)
    results["A21_acf_abs"]            = float(acf_error(np.abs(_lr_fake_3d), np.abs(_lr_real_3d))); print(f"{results['A21_acf_abs']:.6f}")
    print("  A22 ACF sq returns ...", end=" ", flush=True)
    results["A22_acf_sq"]             = float(acf_error(_lr_fake_3d**2, _lr_real_3d**2)); print(f"{results['A22_acf_sq']:.6f}")
    print("  A23 ACF |r| lag-1 error ...", end=" ", flush=True)
    results["A23_acf_lag1_abs_error"] = float(acf_lag1_abs_error(real3, fake3)); print(f"{results['A23_acf_lag1_abs_error']:.6f}")
    print("  A24 ACF r2 lag-1 error ...", end=" ", flush=True)
    results["A24_acf_lag1_sq_error"]  = float(acf_lag1_sq_error(real3, fake3)); print(f"{results['A24_acf_lag1_sq_error']:.6f}")

    # ── Vol (A25–A32) ─────────────────────────────────────────────────────────
    print("  A25 mean RMSE ...", end=" ", flush=True)
    results["A25_mean_rmse"]          = float(terminal_mean_rmse(real3, fake3)); print(f"{results['A25_mean_rmse']:.6f}")
    print("  A26 return std error ...", end=" ", flush=True)
    results["A26_std_error"]          = float(return_std_error(real3, fake3)); print(f"{results['A26_std_error']:.6f}")
    print("  A27 logreturn std error ...", end=" ", flush=True)
    results["A27_logreturn_std_error"] = float(logreturn_std_error(real3, fake3)); print(f"{results['A27_logreturn_std_error']:.6f}")
    print("  A28 kurtosis ratio ...", end=" ", flush=True)
    results["A28_kurtosis_ratio"]     = float(kurtosis_ratio(real3, fake3)); print(f"{results['A28_kurtosis_ratio']:.4f}")
    print("  A29 sigma mean error ...", end=" ", flush=True)
    results["A29_sigma_mean_error"]   = float(sigma_mean_error(real3, fake3)); print(f"{results['A29_sigma_mean_error']:.6f}")
    print("  A30 vol path RMSE ...", end=" ", flush=True)
    results["A30_vol_path_rmse"]      = float(vol_path_rmse(real3, fake3)); print(f"{results['A30_vol_path_rmse']:.6f}")
    print("  A31 rolling vol KS ...", end=" ", flush=True)
    results["A31_rolling_vol_ks"]     = float(rolling_vol_ks(real3, fake3)); print(f"{results['A31_rolling_vol_ks']:.6f}")
    print("  A32 vol-of-vol error ...", end=" ", flush=True)
    results["A32_vol_of_vol_error"]   = float(vol_of_vol_error(real3, fake3)); print(f"{results['A32_vol_of_vol_error']:.6f}")

    # ── Heston Spec (A33–A34) — Teacher-Sigma ─────────────────────────────────
    print("  A33/A34 teacher-sigma ...", end=" ", flush=True)
    try:
        corr_ts, rmse_ts = teacher_sigma_metrics(fake3, v)
        results["A33_sigma_corr"] = float(corr_ts)
        results["A34_sigma_rmse"] = float(rmse_ts)
        print(f"corr={results['A33_sigma_corr']:.4f}  rmse={results['A34_sigma_rmse']:.4f}")
    except Exception as ex:
        results["A33_sigma_corr"] = None
        results["A34_sigma_rmse"] = None
        print(f"SKIPPED ({ex})")

    # B curve metrics: 6 plots × 3 sub-metrics (funct / der / sec_der)
    # Each metric = MSE between the real and generated curve (or its finite difference)
    print("  B curve metrics (6 plots × 3) ...", end=" ", flush=True)
    curve = compute_curve_metrics(S, fake)
    results.update(curve)
    print(f"B_log_ret_hist_funct={curve['B_log_ret_hist_funct']:.4f}  B_qq_plot_funct={curve['B_qq_plot_funct']:.6f}")

    results["compute_time_sec"] = round(time.perf_counter() - t0, 2)
    print(f"  Done in {results['compute_time_sec']:.1f}s")
    return results


def save_plots(seed: int, S: np.ndarray, fake: np.ndarray):
    N_plot = 500
    idx_r  = np.random.choice(len(S),    N_plot, replace=False)
    idx_f  = np.random.choice(len(fake), N_plot, replace=False)

    # Flatten to 2D for PCA/t-SNE
    R_flat = S[idx_r]        # (500, 128)
    F_flat = fake[idx_f]     # (500, 128)
    X_all  = np.vstack([R_flat, F_flat])

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_pca[:N_plot, 0], X_pca[:N_plot, 1], alpha=0.5, s=8, label="Real",  c="#2196F3")
    ax.scatter(X_pca[N_plot:, 0], X_pca[N_plot:, 1], alpha=0.5, s=8, label="Fake",  c="#FF5722")
    ax.set_title(f"PCA — Seed {seed}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"seed_{seed}_pca.png"), dpi=120)
    plt.close()

    # t-SNE (on subset of 200)
    N_tsne = 200
    X_tsne_in = np.vstack([R_flat[:N_tsne], F_flat[:N_tsne]])
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
    X_tsne = tsne.fit_transform(X_tsne_in)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_tsne[:N_tsne,0], X_tsne[:N_tsne,1], alpha=0.6, s=10, label="Real", c="#2196F3")
    ax.scatter(X_tsne[N_tsne:,0], X_tsne[N_tsne:,1], alpha=0.6, s=10, label="Fake", c="#FF5722")
    ax.set_title(f"t-SNE — Seed {seed}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"seed_{seed}_tsne.png"), dpi=120)
    plt.close()
    print(f"  Plots saved for seed {seed}")


def main():
    S, v = load_data()
    S_disc = load_disc()
    print(f"Real data: {S.shape}  min={S.min():.2f}  max={S.max():.2f}")
    print(f"Disc judge data (seed 2): {S_disc.shape}")

    all_results = []
    for seed in range(N_SEEDS):
        try:
            res = compute_metrics_for_seed(seed, S, v, S_disc)
            out_path = os.path.join(RESULTS_DIR, f"seed_{seed}_metrics.json")
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  Saved {out_path}")
            fake = load_generated(seed)
            save_plots(seed, S, fake)
            all_results.append(res)
        except Exception as ex:
            print(f"  ERROR seed {seed}: {ex}")

    # Summary CSV
    if all_results:
        import csv
        metric_keys = [k for k in all_results[0] if k not in ("seed","compute_time_sec")]
        summary_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["metric","mean","std","seed_0","seed_1","seed_2","seed_3","seed_4"])
            w.writeheader()
            for mk in metric_keys:
                vals = []
                per_seed = {}
                for r in all_results:
                    v_val = r.get(mk)
                    per_seed[f"seed_{r['seed']}"] = v_val
                    if v_val is not None:
                        try: vals.append(float(v_val))
                        except: pass
                row = {"metric": mk,
                       "mean": round(float(np.mean(vals)), 6) if vals else None,
                       "std":  round(float(np.std(vals)),  6) if vals else None}
                row.update(per_seed)
                w.writerow(row)
        print(f"\nSaved {summary_path}")


if __name__ == "__main__":
    main()
