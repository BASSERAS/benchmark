"""CSDI -- multi-asset Heston (d = 8) loss convergence, 5 seeds overlaid.

CSDI optimises a **single** term: the DDPM denoising objective
``E_t || eps - eps_theta(x_t, t) ||^2`` restricted to ``target_mask``, which in the
unconditional regime is 1 everywhere (see ``train_multiasset.py``). There is no
KLD/NLL/reconstruction decomposition to break out, so the figure is two panels
rather than LS4's five: train and validation, each with all 5 seeds overlaid.

The two panels are **not** the same quantity plotted twice. ``calc_loss`` draws one
random diffusion step ``t`` per sample, so the train curve is a one-sample estimate
and is dominated by that sampling noise; ``calc_loss_valid`` averages over **all 50**
diffusion steps, so the validation curve is the low-variance one and is the series
to read convergence from. Their levels are not comparable for the same reason --
they are different estimators of different averages, not train/test versions of one
number, so a validation curve sitting *below* the training curve here is expected
and is not evidence of anything.

This is NOT a copy of ``methods/CSDI/code/plot_losses.py``: that script reads the
d = 1 column names (``step,loss``, logged every step, no validation phase) and
writes into ``losses/``. The multi-asset contract (MULTIASSET_GUIDELINE.md
section 3.2) fixes ``step,phase,loss_total`` logged every 100 steps and puts the
figure in ``plots/``.

Reads   ../losses/seed_{0..4}_losses.csv   columns: step,phase,loss_total
Writes  ../plots/loss_convergence.png      (1600x900 at 150 dpi)

Usage:
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/CSDI/code/plot_losses.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSDI_DIR = os.path.dirname(HERE)                       # results/HestonMultiAsset/CSDI
LOSSES_DIR = os.path.join(CSDI_DIR, "losses")
PLOTS_DIR = os.path.join(CSDI_DIR, "plots")

COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
PANELS = [
    ("train", "Training loss -- DDPM eps-MSE, one random t per sample"),
    ("val",   "Validation loss -- same objective averaged over all 50 diffusion steps"),
]


def _read(seed):
    path = os.path.join(LOSSES_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _series(rows, phase):
    """(steps, values) for one phase, skipping blank and non-finite cells."""
    xs, ys = [], []
    for r in rows:
        if r.get("phase") != phase:
            continue
        v = r.get("loss_total", "")
        if v == "" or v is None:
            continue
        y = float(v)
        if y != y:          # nan: a skipped validation epoch
            continue
        xs.append(int(r["step"]))
        ys.append(y)
    return xs, ys


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # 1600x900 at 150 dpi  ->  figsize = (1600/150, 900/150)
    fig, axes = plt.subplots(1, 2, figsize=(1600 / 150, 900 / 150), dpi=150)

    final_val, n_seeds = [], 0
    for seed in range(5):
        rows = _read(seed)
        if rows is None:
            continue
        n_seeds += 1
        for ax, (phase, _) in zip(axes, PANELS):
            xs, ys = _series(rows, phase)
            if xs:
                ax.plot(xs, ys, color=COLORS[seed], alpha=0.85, linewidth=1.0,
                        marker="o" if phase == "val" else None, markersize=2.5,
                        label=f"Seed {seed}")
        vx, vy = _series(rows, "val")
        if vy:
            final_val.append(vy[-1])

    for ax, (_, title) in zip(axes, PANELS):
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("loss_total", fontsize=7)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    band = (f"final validation loss in [{min(final_val):.4f}, {max(final_val):.4f}]"
            if final_val else "no seeds found")
    fig.suptitle(
        f"CSDI loss convergence -- multi-asset Heston d = 8, T = 252, {n_seeds} seeds  |  "
        f"log y-axis  |  {band}",
        fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(PLOTS_DIR, "loss_convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  ({n_seeds} seeds)")


if __name__ == "__main__":
    main()
