"""Discriminative (A18) + predictive (A19) classifier loss-curve plots.

Recreates the two result figures that live next to every method's metrics:
    results/Heston/<method>/plots/disc_classifier_loss.png   (GRU+MLP BCE, 5 seeds)
    results/Heston/<method>/plots/pred_score_loss.png        (GRU+MLP MAE, 5 seeds)

The original TimeGAN generator (commit 941f07b) was ad-hoc and never committed;
this is the reusable version so any method can regenerate identical figures.

Reads   results/Heston/<method>/seed_{0..4}_disc_gru_loss.csv   (cols: step,train_bce)
                                seed_{0..4}_disc_mlp_loss.csv   (cols: step,train_bce)
                                seed_{0..4}_pred_gru_loss.csv   (cols: step,train_mae)
                                seed_{0..4}_pred_mlp_loss.csv   (cols: step,train_mae)
Writes  results/Heston/<method>/plots/{disc_classifier_loss,pred_score_loss}.png

Usage:
    /home/tbasseras/gpu-venv/bin/python metrics/plot_score_losses.py --method FourierFlow
"""
from __future__ import annotations
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
DPI = 150


def _read(path, ycol):
    if not os.path.exists(path):
        return None, None
    xs, ys = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            xs.append(int(r["step"]))
            ys.append(float(r[ycol]))
    return xs, ys


def _panel(ax, res_dir, kind, arch, ycol, ylabel, title):
    for seed in range(5):
        xs, ys = _read(os.path.join(res_dir, f"seed_{seed}_{kind}_{arch}_loss.csv"), ycol)
        if xs is None:
            continue
        ax.plot(xs, ys, color=COLORS[seed], alpha=0.85, linewidth=1.3,
                label=f"seed {seed} (last={ys[-1]:.4f})")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="FourierFlow")
    ap.add_argument("--dataset", default="Heston")
    args = ap.parse_args()

    res_dir = os.path.join(REPO, "results", args.dataset, args.method)
    plots_dir = os.path.join(res_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- A18 discriminative classifier loss (BCE) ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    _panel(a1, res_dir, "disc", "gru", "train_bce", "Train BCE",
           "GRU discriminator loss")
    _panel(a2, res_dir, "disc", "mlp", "train_bce", "Train BCE",
           "MLP discriminator loss")
    plt.suptitle(f"{args.method} — A18 Discriminative classifier loss "
                 f"({args.dataset}, 5 seeds)", fontsize=13)
    plt.tight_layout()
    out1 = os.path.join(plots_dir, "disc_classifier_loss.png")
    plt.savefig(out1, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {out1}")

    # --- A19 predictive score loss (MAE) ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    _panel(a1, res_dir, "pred", "gru", "train_mae", "Train MAE",
           "GRU predictor loss")
    _panel(a2, res_dir, "pred", "mlp", "train_mae", "Train MAE",
           "MLP predictor loss")
    plt.suptitle(f"{args.method} — A19 Predictive score loss "
                 f"({args.dataset}, 5 seeds)", fontsize=13)
    plt.tight_layout()
    out2 = os.path.join(plots_dir, "pred_score_loss.png")
    plt.savefig(out2, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
