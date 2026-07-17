"""
compute_all.py — Run all 14 metrics (A1-A14 + A15) for each of 5 seeds.

For each seed:
  - Loads real paths from dataset/<dataset>/
  - Loads generated paths from methods/<method>/generated_paths/seed_i/
  - Computes A1-A12 (numpy), A13 (PyTorch GRU+MLP), A14 (PyTorch GRU+MLP), A15 (numpy)
  - Saves results/<dataset>/<method>/seed_i_metrics.json
  - Generates PCA + t-SNE plots

After all seeds: writes results/<dataset>/<method>/metrics_summary.csv
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

from metrics_np import (
    mmd2, terminal_mmd2, increment_mmd2, volatility_mmd,
    terminal_swd, path_swd,
    terminal_cov_error, terminal_mean_rmse,
    return_std_error, return_kurtosis_error,
    acf_error,
    teacher_sigma_metrics,
    tail_survival_error,
)
from discriminative_score import compute_discriminative_score
from predictive_score import compute_predictive_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_data():
    S = np.load(os.path.join(DATASET_DIR, "heston_S_8192x128.npy"))   # (8192, 128)
    v = np.load(os.path.join(DATASET_DIR, "heston_v_8192x128.npy"))   # (8192, 128)
    # Note: file names are dataset-specific; update for new datasets
    return S, v


def load_generated(seed: int) -> np.ndarray:
    path = os.path.join(GENERATED_DIR, f"seed_{seed}", "generated_paths_8192x128.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}. Run TimeGan/train.py first.")
    return np.load(path)   # (8192, 128)


def compute_metrics_for_seed(seed: int, S: np.ndarray, v: np.ndarray) -> dict:
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

    print("  A1  path MMD2  ...", end=" ", flush=True)
    results["A1_path_mmd2"]        = float(mmd2(R, F));          print(f"{results['A1_path_mmd2']:.6f}")
    print("  A2  terminal MMD2 ...", end=" ", flush=True)
    results["A2_terminal_mmd2"]    = float(terminal_mmd2(R, F)); print(f"{results['A2_terminal_mmd2']:.6f}")
    print("  A3  increment MMD2 ...", end=" ", flush=True)
    results["A3_increment_mmd2"]   = float(increment_mmd2(R, F));print(f"{results['A3_increment_mmd2']:.6f}")
    print("  A4  volatility MMD ...", end=" ", flush=True)
    results["A4_volatility_mmd"]   = float(volatility_mmd(R, F));print(f"{results['A4_volatility_mmd']:.6f}")
    print("  A5  terminal SWD ...", end=" ", flush=True)
    results["A5_terminal_swd"]     = float(terminal_swd(R, F));  print(f"{results['A5_terminal_swd']:.6f}")
    print("  A6  path SWD ...", end=" ", flush=True)
    results["A6_path_swd"]         = float(path_swd(R, F));      print(f"{results['A6_path_swd']:.6f}")

    # Statistical metrics use full dataset
    print("  A7  cov error ...", end=" ", flush=True)
    results["A7_cov_error"]        = float(terminal_cov_error(real3, fake3)); print(f"{results['A7_cov_error']:.6f}")
    print("  A8  mean RMSE ...", end=" ", flush=True)
    results["A8_mean_rmse"]        = float(terminal_mean_rmse(real3, fake3)); print(f"{results['A8_mean_rmse']:.6f}")
    print("  A9  std error ...", end=" ", flush=True)
    results["A9_std_error"]        = float(return_std_error(real3, fake3));   print(f"{results['A9_std_error']:.6f}")
    print("  A10 kurtosis error ...", end=" ", flush=True)
    results["A10_kurtosis_error"]  = float(return_kurtosis_error(real3, fake3));print(f"{results['A10_kurtosis_error']:.6f}")
    print("  A11 ACF abs returns ...", end=" ", flush=True)
    _d_real = np.diff(real3, axis=1); _d_fake = np.diff(fake3, axis=1)
    results["A11_acf_abs"]         = float(acf_error(np.abs(_d_fake), np.abs(_d_real))); print(f"{results['A11_acf_abs']:.6f}")
    print("  A12 ACF sq returns ...", end=" ", flush=True)
    results["A12_acf_sq"]          = float(acf_error(_d_fake**2, _d_real**2));           print(f"{results['A12_acf_sq']:.6f}")

    # A13 Discriminative Score
    print("  A13 discriminative (GRU + MLP) ...", flush=True)
    d13 = compute_discriminative_score(S, fake, n_steps=2000, device=DEVICE)
    results["A13_disc_score_gru"] = d13["disc_score_gru"]
    results["A13_disc_score_mlp"] = d13["disc_score_mlp"]
    print(f"       GRU={d13['disc_score_gru']:.4f}  MLP={d13['disc_score_mlp']:.4f}")
    # Save classifier loss curves
    import csv
    for arch in ("gru", "mlp"):
        loss_path = os.path.join(RESULTS_DIR, f"seed_{seed}_disc_{arch}_loss.csv")
        with open(loss_path, "w", newline="") as lf:
            w = csv.DictWriter(lf, fieldnames=["step", "train_bce"])
            w.writeheader(); w.writerows(d13[f"loss_history_{arch}"])

    # A14 Predictive Score (TSTR)
    print("  A14 predictive TSTR (GRU + MLP) ...", flush=True)
    d14 = compute_predictive_score(S, fake, n_steps=5000, device=DEVICE)
    results.update({f"A14_{k}": v for k, v in d14.items()
                    if not k.startswith("loss_history")})
    print(f"       GRU={d14['pred_score_gru']:.4f}  MLP={d14['pred_score_mlp']:.4f}")
    # Save predictive loss histories
    for arch in ("gru", "mlp"):
        csv_path = os.path.join(RESULTS_DIR, f"seed_{seed}_pred_{arch}_loss.csv")
        with open(csv_path, "w", newline="") as lf:
            w = csv.DictWriter(lf, fieldnames=["step", "train_mae"])
            w.writeheader(); w.writerows(d14[f"loss_history_{arch}"])

    # A15 Teacher-Sigma (Heston-specific)
    print("  A15 teacher-sigma ...", end=" ", flush=True)
    try:
        corr15, rmse15 = teacher_sigma_metrics(fake3, v)
        results["A15_sigma_corr"] = float(corr15)
        results["A15_sigma_rmse"] = float(rmse15)
        print(f"corr={results['A15_sigma_corr']:.4f}  rmse={results['A15_sigma_rmse']:.4f}")
    except Exception as ex:
        results["A15_sigma_corr"] = None
        results["A15_sigma_rmse"] = None
        print(f"SKIPPED ({ex})")

    # A16 Tail Survival Error
    print("  A16 tail survival ...", end=" ", flush=True)
    results["A16_tail_survival"] = float(tail_survival_error(real3, fake3))
    print(f"{results['A16_tail_survival']:.6f}")

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
    print(f"Real data: {S.shape}  min={S.min():.2f}  max={S.max():.2f}")

    all_results = []
    for seed in range(N_SEEDS):
        try:
            res = compute_metrics_for_seed(seed, S, v)
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
