"""LS4 — Heston loss convergence plot (5 seeds overlaid).

LS4 is a VAE-style latent state-space model trained on the ELBO
(``total = kld_loss + nll_loss``; ``mse_loss`` is a reconstruction diagnostic;
see ../train_heston.py).  This mirrors methods/TimeVAE/code/plot_losses.py but
with LS4's Total / KLD / NLL / MSE panels + the ReduceLROnPlateau LR schedule.

Reads   ../losses/seed_{0..4}_losses.csv
        (columns: epoch,total_loss,kld_loss,nll_loss,mse_loss,lr)
Writes  ../losses/loss_convergence.png

Usage:
    /home/tbasseras/gpu-venv/bin/python methods/LS4/code/plot_losses.py
"""
from __future__ import annotations
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)                     # methods/LS4
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

FIELDS = ["total_loss", "kld_loss", "nll_loss", "mse_loss", "lr"]
TITLES = ["Total Loss (ELBO)", "KLD Loss", "NLL Loss", "MSE (recon diag.)",
          "Learning Rate"]


def _read(seed: int):
    path = os.path.join(LOSSES_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    last_totals = []

    for seed in range(5):
        rows = _read(seed)
        if rows is None:
            continue
        ep = [int(r["epoch"]) for r in rows]
        for ax, fld in zip(axes, FIELDS):
            vals = [float(r[fld]) for r in rows]
            ax.plot(ep, vals, color=COLORS[seed], alpha=0.85,
                    linewidth=1.2, label=f"seed {seed}")
        last_totals.append(float(rows[-1]["total_loss"]))

    for ax, title, fld in zip(axes, TITLES, FIELDS):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if fld == "lr":
            ax.set_yscale("log")

    axes[-1].axis("off")   # 6th panel unused (5 tracked fields)

    band = (f"total_last ∈ [{min(last_totals):.2f}, {max(last_totals):.2f}]"
            if last_totals else "")
    plt.suptitle(
        "LS4 Loss Convergence — Heston seq_len=128  "
        f"(ELBO = KLD + NLL; EMA lamb=0.99 start_step=200; "
        f"ReduceLROnPlateau patience=20; {band})",
        fontsize=13)
    plt.tight_layout()
    out = os.path.join(LOSSES_DIR, "loss_convergence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}  ({band})")


if __name__ == "__main__":
    main()
