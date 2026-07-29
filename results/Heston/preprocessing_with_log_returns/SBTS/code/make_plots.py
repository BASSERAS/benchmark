"""
Generate the 3 non-PCA/tSNE plots the LS4-style README needs, for SBTS.

Reuses the benchmark's canonical plotting code WITHOUT editing it:
  metrics/plot_diagnostics.plot_diagnostics  → heston_diagnostics.png
  metrics/plot_score_losses._panel           → disc_classifier_loss.png,
                                                pred_score_loss.png

The score-loss module hardcodes the main-benchmark results dir, so instead of
calling its main() we reuse its per-panel helper pointed at THIS folder's
per-seed loss CSVs (written by compute_metrics_logret.py).

Run (gpu-venv):
  python make_plots.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))                    # .../SBTS/code
METHOD_DIR = os.path.dirname(HERE)                                   # .../SBTS
BENCH_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
METRICS_DIR = os.path.join(BENCH_ROOT, "metrics")
DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns")
PLOTS_DIR = os.path.join(METHOD_DIR, "plots")
N = 4096
DPI = 150

sys.path.insert(0, METRICS_DIR)
from plot_diagnostics import plot_diagnostics    # noqa: E402
from plot_score_losses import _panel             # noqa: E402


def make_diagnostics():
    S_real = np.load(os.path.join(DATA_DIR, f"heston_S_test_{N}x128.npy"))
    S_gen = np.load(os.path.join(METHOD_DIR, "generated_paths", "seed_0",
                                 f"generated_paths_{N}x128.npy"))
    out = os.path.join(PLOTS_DIR, "heston_diagnostics.png")
    plot_diagnostics(S_real, S_gen, method="SBTS (log-ret preproc)",
                     seed=0, out_path=out)
    print(f"Saved {out}")


def make_loss_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # A18 discriminative (BCE)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    _panel(a1, METHOD_DIR, "disc", "gru", "train_bce", "Train BCE",
           "GRU discriminator loss")
    _panel(a2, METHOD_DIR, "disc", "mlp", "train_bce", "Train BCE",
           "MLP discriminator loss")
    plt.suptitle("SBTS — A18 Discriminative classifier loss (Heston, 5 seeds)",
                 fontsize=13)
    plt.tight_layout()
    out1 = os.path.join(PLOTS_DIR, "disc_classifier_loss.png")
    plt.savefig(out1, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {out1}")

    # A19 predictive (MAE)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    _panel(a1, METHOD_DIR, "pred", "gru", "train_mae", "Train MAE",
           "GRU predictor loss")
    _panel(a2, METHOD_DIR, "pred", "mlp", "train_mae", "Train MAE",
           "MLP predictor loss")
    plt.suptitle("SBTS — A19 Predictive score loss (Heston, 5 seeds)",
                 fontsize=13)
    plt.tight_layout()
    out2 = os.path.join(PLOTS_DIR, "pred_score_loss.png")
    plt.savefig(out2, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


if __name__ == "__main__":
    make_diagnostics()
    make_loss_plots()
