"""
Generate the 4 non-PCA/tSNE figures the LS4-style README needs, for TimeDiT.

Reuses the benchmark's canonical plotting code WITHOUT editing it:
  metrics/plot_diagnostics.plot_diagnostics  -> plots/heston_diagnostics.png
                                               baseline_no_preproc/plots/heston_diagnostics.png
  metrics/plot_score_losses._panel           -> plots/disc_classifier_loss.png
                                               plots/pred_score_loss.png

The training-loss figure has no shared generator (every method logs a different
objective), so it is drawn here from this folder's own
losses/seed_{0..4}_losses.csv (cols: step,phase,loss_total).

TimeDiT's objective is a single scalar: the DDPM hybrid noise-prediction loss.
There is no KLD/NLL/EMA/LR schedule to plot (Adam lr=3e-4 constant, wd=0, no EMA),
so loss_convergence.png is a 2-panel linear + log-y view across the 5 seeds.

Run (gpu-venv, CPU only, ~10 s):
  /home/tbasseras/gpu-venv/bin/python make_plots.py
  /home/tbasseras/gpu-venv/bin/python make_plots.py --only baseline
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))                     # .../TimeDiT/code
METHOD_DIR = os.path.dirname(HERE)                                    # .../TimeDiT
BASELINE_DIR = os.path.join(METHOD_DIR, "baseline_no_preproc")
BENCH_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
METRICS_DIR = os.path.join(BENCH_ROOT, "metrics")
DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston",
                        "preprocessing_with_log_returns")
PLOTS_DIR = os.path.join(METHOD_DIR, "plots")
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")
N = 4096
DPI = 150
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

sys.path.insert(0, METRICS_DIR)
from plot_diagnostics import plot_diagnostics    # noqa: E402
from plot_score_losses import _panel             # noqa: E402


def _real_test():
    return np.load(os.path.join(DATA_DIR, f"heston_S_test_{N}x128.npy"))


def make_diagnostics():
    """8-panel stylised-facts figure, log-return-preprocessed variant, seed 0."""
    S_gen = np.load(os.path.join(METHOD_DIR, "generated_paths", "seed_0",
                                 f"generated_paths_{N}x128.npy"))
    out = os.path.join(PLOTS_DIR, "heston_diagnostics.png")
    plot_diagnostics(_real_test(), S_gen, method="TimeDiT (log-ret preproc)",
                     seed=0, out_path=out)
    print(f"Saved {out}")


def make_baseline_diagnostics():
    """Same 8 panels for the raw / no-preproc control (GUIDELINE 10.0.1 B.4).

    Not produced by the baseline metrics runner: generated here from the
    already-saved baseline paths. No retrain, CPU only.
    """
    S_gen = np.load(os.path.join(BASELINE_DIR, "generated_paths", "seed_0",
                                 f"generated_paths_{N}x128.npy"))
    out_dir = os.path.join(BASELINE_DIR, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "heston_diagnostics.png")
    plot_diagnostics(_real_test(), S_gen, method="TimeDiT raw (no-preproc)",
                     seed=0, out_path=out)
    print(f"Saved {out}")


def _read_losses(seed):
    path = os.path.join(LOSSES_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None, None
    xs, ys = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            xs.append(int(r["step"]))
            ys.append(float(r["loss_total"]))
    return xs, ys


def make_loss_convergence():
    """DDPM hybrid loss across the 5 seeds: linear + log-y."""
    os.makedirs(LOSSES_DIR, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    for seed in range(5):
        xs, ys = _read_losses(seed)
        if xs is None:
            continue
        lab = f"seed {seed} (min={min(ys):.6f}, last={ys[-1]:.6f})"
        a1.plot(xs, ys, color=COLORS[seed], alpha=0.85, linewidth=1.3, label=lab)
        a2.plot(xs, ys, color=COLORS[seed], alpha=0.85, linewidth=1.3, label=lab)
    a1.set_title("DDPM hybrid loss (linear)", fontsize=12)
    a2.set_title("DDPM hybrid loss (log y)", fontsize=12)
    a2.set_yscale("log")
    for ax in (a1, a2):
        ax.set_xlabel("Training step")
        ax.set_ylabel("loss_total")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.suptitle("TimeDiT + log-return preprocessing — training loss "
                 "(Heston, 5 seeds, 15 000 steps)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(LOSSES_DIR, "loss_convergence.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def make_score_loss_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # A18 discriminative (BCE); ln2 ~ 0.693 == indistinguishable
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    _panel(a1, METHOD_DIR, "disc", "gru", "train_bce", "Train BCE",
           "GRU discriminator loss")
    _panel(a2, METHOD_DIR, "disc", "mlp", "train_bce", "Train BCE",
           "MLP discriminator loss")
    plt.suptitle("TimeDiT — A18 Discriminative classifier loss "
                 "(Heston, 5 seeds)", fontsize=13)
    plt.tight_layout()
    out1 = os.path.join(PLOTS_DIR, "disc_classifier_loss.png")
    plt.savefig(out1, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {out1}")

    # A19 predictive TSTR (MAE)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    _panel(a1, METHOD_DIR, "pred", "gru", "train_mae", "Train MAE",
           "GRU predictor loss")
    _panel(a2, METHOD_DIR, "pred", "mlp", "train_mae", "Train MAE",
           "MLP predictor loss")
    plt.suptitle("TimeDiT — A19 Predictive score loss (Heston, 5 seeds)",
                 fontsize=13)
    plt.tight_layout()
    out2 = os.path.join(PLOTS_DIR, "pred_score_loss.png")
    plt.savefig(out2, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all", "baseline", "losses", "diagnostics", "scores"])
    a = ap.parse_args()
    if a.only in ("all", "diagnostics"):
        make_diagnostics()
    if a.only in ("all", "baseline"):
        make_baseline_diagnostics()
    if a.only in ("all", "losses"):
        make_loss_convergence()
    if a.only in ("all", "scores"):
        make_score_loss_plots()
