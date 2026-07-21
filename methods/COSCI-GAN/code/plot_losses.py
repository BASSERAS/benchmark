"""COSCI-GAN — Heston loss convergence plot (5 seeds overlaid).

COSCI-GAN is a 3-player game: per-channel Discriminator(s), per-channel
Generator(s), and one Central Discriminator (CD).  For Heston C=1 there is a
single channel, so each epoch logs three losses (see ../code/train_heston.py):

    loss_D_0  — channel-0 discriminator BCE
    loss_G_0  — channel-0 generator loss  (local adversarial − gamma * CD term)
    loss_CD   — central discriminator BCE

For C=1 the CD is fed the same 128-dim vector as the channel discriminator, so
loss_CD sits at ln 2 ≈ 0.693 (CD at chance) — the documented C=1 degeneracy, and
the healthy equilibrium signature. This mirrors methods/TimeVAE/code/plot_losses.py
(2x2 grid) but with the three COSCI-GAN loss panels + a shared overlay panel.

Reads   ../losses/seed_{0..4}_losses.csv
        (columns: epoch,loss_D_0,loss_G_0,loss_CD)
Writes  ../losses/loss_convergence.png

Usage:
    /home/tbasseras/gpu-venv/bin/python methods/COSCI-GAN/code/plot_losses.py
"""
from __future__ import annotations
import csv, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)                     # methods/COSCI-GAN
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

FIELDS = ["loss_D_0", "loss_G_0", "loss_CD"]
TITLES = ["Channel Discriminator (loss_D_0)",
          "Channel Generator (loss_G_0)",
          "Central Discriminator (loss_CD)"]


def _read(seed: int):
    path = os.path.join(LOSSES_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.flatten()

    for seed in range(5):
        rows = _read(seed)
        if rows is None:
            continue
        ep = [int(r["epoch"]) for r in rows]
        for ax, fld in zip(axes[:3], FIELDS):
            vals = [float(r[fld]) for r in rows]
            ax.plot(ep, vals, color=COLORS[seed], alpha=0.85,
                    linewidth=1.2, label=f"seed {seed}")
        cd = [float(r["loss_CD"]) for r in rows]
        axes[3].plot(ep, cd, color=COLORS[seed], alpha=0.85,
                     linewidth=1.2, label=f"seed {seed}")

    for ax, title in zip(axes[:3], TITLES):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[3].axhline(math.log(2), color="black", linestyle="--", linewidth=1.0,
                    label="ln 2 (CD at chance)")
    axes[3].set_title("Central Discriminator vs ln 2 (C=1 equilibrium)", fontsize=12)
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("loss_CD")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=8)

    fig.suptitle("COSCI-GAN — Heston C=1 training loss convergence (5 seeds)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(LOSSES_DIR, "loss_convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
