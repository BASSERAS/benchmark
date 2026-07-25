"""TimesFM — Heston fine-tune loss convergence plot (5 seeds overlaid).

TimesFM-1.0-200m is a pretrained decoder-only patched forecaster; the only
training performed for this benchmark is a *full* fine-tune of the released
``google/timesfm-1.0-200m-pytorch`` checkpoint on the 8192 Heston training
paths (see ../path_shadowing/finetune_heston.py):

  finetune_mode="full"   all decoder weights updated
  context=path[:64]      2 input patches of length 32
  future=path[64:128]    64-step forecasting objective
  ft_steps=1000          Adam steps, batch 256, lr 1e-4, weight_decay 0.01

The fit objective is the official TimesFM finetuner loss with
``use_quantile_loss=True``: masked MSE on the mean head PLUS the sum of the
9 pinball terms over the quantile heads (levels 0.1..0.9), all masked to the
first 64 output steps.  This is logged as a single scalar per step (column
``mse`` in the CSV holds this total loss).  Unlike the GAN/VAE methods there
is a single loss curve, so this script overlays the 5 seeds' loss (log-y).

Reads   ../weights/seed_{0..4}_losses.csv   (columns: step,mse)
Writes  ../weights/loss_convergence.png

Usage:
    /home/tbasseras/timesfm-v1-venv/bin/python \
        methods/TimesFM/path_shadowing/plot_losses.py
"""
from __future__ import annotations
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))         # path_shadowing
METHOD_DIR = os.path.dirname(HERE)                        # methods/TimesFM
WEIGHTS_DIR = os.path.join(METHOD_DIR, "weights")
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]


def _read(seed: int):
    path = os.path.join(WEIGHTS_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    first_loss, last_loss = [], []

    for seed in range(5):
        rows = _read(seed)
        if rows is None:
            continue
        step = [int(r["step"]) for r in rows]
        loss = [float(r["mse"]) for r in rows]
        ax.plot(step, loss, color=COLORS[seed], alpha=0.85, linewidth=1.3,
                label=f"seed {seed}")
        first_loss.append(loss[0])
        last_loss.append(loss[-1])

    ax.set_title("Fine-tune loss (masked MSE + 9-quantile pinball)", fontsize=12)
    ax.set_xlabel("Adam step")
    ax.set_ylabel("total loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    band = (f"loss_last in [{min(last_loss):.3f}, {max(last_loss):.3f}]"
            if last_loss else "")
    fig.suptitle(
        "TimesFM-1.0-200m Fine-tune Loss Convergence — Heston seq_len=128  "
        f"(full fine-tune, 1000 steps, batch 256, lr 1e-4; {band})",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(WEIGHTS_DIR, "loss_convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}  ({band})")


if __name__ == "__main__":
    main()
