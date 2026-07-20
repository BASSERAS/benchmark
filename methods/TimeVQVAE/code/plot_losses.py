#!/usr/bin/env python
"""Plot TimeVQVAE training-loss convergence across the 5 Heston seeds.

Standalone script (run manually):
    /home/tbasseras/tvqvae-venv/bin/python methods/TimeVQVAE/code/plot_losses.py

Reads   ../losses/seed_{0..4}_stage1_losses.csv
        ../losses/seed_{0..4}_stage2_losses.csv
Writes  ../losses/loss_convergence.png   (2x2 grid, 1600x900, 150 dpi)

TimeVQVAE is a two-stage model, so the grid shows:
  (1) Stage-1 VQ-VAE total loss            (log scale)
  (2) Stage-1 reconstruction loss (time)   (log scale)
  (3) Stage-2 MaskGIT prior loss           (masked-token cross-entropy)
  (4) Stage-1 codebook perplexity          (LF solid / HF dashed, max=32)
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOSSD = os.path.join(HERE, "..", "losses")
SEEDS = [0, 1, 2, 3, 4]
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]


def _read(seed, stage):
    path = os.path.join(LOSSD, f"seed_{seed}_stage{stage}_losses.csv")
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _col(rows, key):
    return [float(r[key]) for r in rows]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    (ax1, ax2), (ax3, ax4) = axes

    for seed, color in zip(SEEDS, COLORS):
        s1 = _read(seed, 1)
        s2 = _read(seed, 2)
        ep1 = _col(s1, "epoch")
        ep2 = _col(s2, "epoch")

        ax1.plot(ep1, _col(s1, "loss"), color=color, lw=1.3, label=f"seed {seed}")
        ax2.plot(ep1, _col(s1, "recons_loss.time"), color=color, lw=1.3, label=f"seed {seed}")
        ax3.plot(ep2, _col(s2, "prior_loss"), color=color, lw=1.3, label=f"seed {seed}")
        ax4.plot(ep1, _col(s1, "perplexity.LF"), color=color, lw=1.3, label=f"seed {seed} LF")
        ax4.plot(ep1, _col(s1, "perplexity.HF"), color=color, lw=1.0, ls="--", alpha=0.7)

    ax1.set_title("Stage-1 VQ-VAE Total Loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("total loss")
    ax1.set_yscale("log"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    ax2.set_title("Stage-1 Reconstruction Loss (time domain)")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("recons_loss.time")
    ax2.set_yscale("log"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    ax3.set_title("Stage-2 MaskGIT Prior Loss (masked-token CE)")
    ax3.set_xlabel("epoch"); ax3.set_ylabel("prior loss")
    ax3.grid(alpha=0.3); ax3.legend(fontsize=8)

    ax4.set_title("Stage-1 Codebook Perplexity (LF solid / HF dashed, max=32)")
    ax4.set_xlabel("epoch"); ax4.set_ylabel("perplexity")
    ax4.axhline(32, color="gray", ls=":", lw=1, alpha=0.6)
    ax4.grid(alpha=0.3); ax4.legend(fontsize=8)

    fig.suptitle("TimeVQVAE — Training Loss Convergence (5 Heston seeds)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(LOSSD, "loss_convergence.png")
    fig.savefig(out, dpi=150)
    print(f"Saved → {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
