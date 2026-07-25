"""Chronos-2 — Heston fine-tune loss convergence plot (5 seeds overlaid).

Chronos-2 is a pretrained probabilistic forecaster; the only training performed
for this benchmark is a *full* fine-tune of the released ``amazon/chronos-2``
checkpoint on the 8192 Heston training paths (see ../train_heston.py):

  finetune_mode="full"   all encoder + decoder weights updated
  prediction_length=16   next-16-step quantile forecasting objective
  num_steps=1000         HuggingFace Trainer steps, batch_size=256, lr=1e-4
                         (cosine-decayed, logged every 10 steps)

The fit objective is Chronos-2's native quantile / cross-entropy training loss
(a single scalar ``train_loss`` per logged step) — there is no adversarial or
reconstruction sub-loss, so unlike the GAN/VAE methods there is a single loss
curve.  This script overlays the 5 seeds' ``train_loss`` (log-y) and the shared
learning-rate schedule.

Reads   ../losses/seed_{0..4}_losses.csv   (columns: step,train_loss,lr)
Writes  ../losses/loss_convergence.png

Usage:
    /home/tbasseras/gpu-venv/bin/python methods/Chronos2/code/plot_losses.py
"""
from __future__ import annotations
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)                     # methods/Chronos2
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]


def _read(seed: int):
    path = os.path.join(LOSSES_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(x):
    if x is None or x == "":
        return None
    return float(x)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    last_loss = []

    for seed in range(5):
        rows = _read(seed)
        if rows is None:
            continue
        step = [int(r["step"]) for r in rows]
        loss = [_f(r["train_loss"]) for r in rows]
        lr = [_f(r["lr"]) for r in rows]
        xs = [s for s, v in zip(step, loss) if v is not None]
        ys = [v for v in loss if v is not None]
        if ys:
            axes[0].plot(xs, ys, color=COLORS[seed], alpha=0.85, linewidth=1.3,
                         label=f"seed {seed}")
            last_loss.append(ys[-1])
        lxs = [s for s, v in zip(step, lr) if v is not None]
        lys = [v for v in lr if v is not None]
        if lys:
            axes[1].plot(lxs, lys, color=COLORS[seed], alpha=0.85, linewidth=1.3,
                         label=f"seed {seed}")

    axes[0].set_title("Fine-tune loss (train_loss)", fontsize=12)
    axes[0].set_xlabel("Trainer step")
    axes[0].set_ylabel("train_loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Learning-rate schedule", fontsize=12)
    axes[1].set_xlabel("Trainer step")
    axes[1].set_ylabel("lr")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    band = (f"train_loss_last in [{min(last_loss):.3f}, {max(last_loss):.3f}]"
            if last_loss else "")
    fig.suptitle(
        "Chronos-2 Fine-tune Loss Convergence — Heston seq_len=128  "
        f"(full fine-tune, 1000 steps, batch 256, lr 1e-4; {band})",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(LOSSES_DIR, "loss_convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}  ({band})")


if __name__ == "__main__":
    main()
