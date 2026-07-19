"""Fourier Flow — Heston loss convergence plot (5 seeds overlaid).

Fourier Flow trains by exact maximum likelihood, so there is a SINGLE loss curve
per seed: the mean negative-log-likelihood ``(-log_pz - log_jacob).mean()`` per
full-batch epoch (see ../train_heston.py). This mirrors
methods/TimeGAN/code/train.py::plot_losses but with one panel pair instead of the
GAN's four sub-losses.

Reads   ../losses/seed_{0..4}_losses.csv   (columns: epoch,loss)
Writes  ../losses/loss_convergence.png

Usage:
    /home/tbasseras/gpu-venv/bin/python methods/FourierFlow/code/plot_losses.py
"""
from __future__ import annotations
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)                      # methods/FourierFlow
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
ZOOM_FROM = 100                                        # plateau-zoom start epoch


def _read(seed: int):
    path = os.path.join(LOSSES_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None, None
    ep, ls = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"]))
            ls.append(float(r["loss"]))
    return ep, ls


def main():
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 6))
    last_vals = []
    for seed in range(5):
        ep, ls = _read(seed)
        if ep is None:
            continue
        ax_full.plot(ep, ls, color=COLORS[seed], alpha=0.85,
                     linewidth=1.2, label=f"seed {seed}")
        zi = [i for i, e in enumerate(ep) if e >= ZOOM_FROM]
        if zi:
            ax_zoom.plot([ep[i] for i in zi], [ls[i] for i in zi],
                         color=COLORS[seed], alpha=0.85, linewidth=1.2,
                         label=f"seed {seed} (last={ls[-1]:.1f})")
        last_vals.append(ls[-1])

    ax_full.set_title("Full NLL trajectory (epochs 0–1000)", fontsize=12)
    ax_zoom.set_title(f"Plateau zoom (epochs {ZOOM_FROM}–1000)", fontsize=12)
    for ax in (ax_full, ax_zoom):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Negative log-likelihood  (−log p_z − log|J|).mean()")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    band = f"loss_last ∈ [{min(last_vals):.1f}, {max(last_vals):.1f}]" if last_vals else ""
    plt.suptitle(
        "Fourier Flow Loss Convergence — Heston seq_len=128  "
        f"(normalize=True + zero-std clamp + grad-clip=1.0; no NaN; {band})",
        fontsize=13)
    plt.tight_layout()
    out = os.path.join(LOSSES_DIR, "loss_convergence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}  ({band})")


if __name__ == "__main__":
    main()
