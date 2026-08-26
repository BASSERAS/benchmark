"""LS4 -- multi-asset Heston (d = 8) loss convergence, 5 seeds overlaid.

LS4 is a VAE-style latent state-space model trained on the ELBO
(``loss_total = kld_loss + nll_loss``; ``mse_loss`` is a reconstruction
diagnostic, not part of the objective -- see ``train_multiasset.py``).

This is NOT a copy of ``methods/LS4/code/plot_losses.py``: that script hard-codes
``methods/LS4`` as its root, reads the d = 1 column names
(``epoch,total_loss,kld_loss,...``) and writes into ``losses/``. The multi-asset
contract (MULTIASSET_GUIDELINE.md section 3.2) fixes different column names
(``step,phase,loss_total``), requires the x-axis to be **step, not epoch**, and
puts the figure in ``plots/``.

Reads   ../losses/seed_{0..4}_losses.csv
        columns: step,phase,loss_total,epoch,kld_loss,nll_loss,mse_loss,lr
Writes  ../plots/loss_convergence.png   (1600x900 at 150 dpi)

Usage:
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/LS4/code/plot_losses.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LS4_DIR = os.path.dirname(HERE)                        # results/HestonMultiAsset/LS4
LOSSES_DIR = os.path.join(LS4_DIR, "losses")
PLOTS_DIR = os.path.join(LS4_DIR, "plots")

COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
PANELS = [
    ("loss_total", "Total loss (ELBO = KLD + NLL)"),
    ("kld_loss",   "KLD term"),
    ("nll_loss",   "NLL term"),
    ("mse_loss",   "MSE (reconstruction diagnostic)"),
    ("lr",         "Learning rate (ReduceLROnPlateau)"),
]


def _read(seed):
    path = os.path.join(LOSSES_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _series(rows, phase, field):
    """(steps, values) for one phase, skipping blank cells.

    Validation rows carry only ``loss_total``; the decomposed terms are left
    empty rather than filled with a duplicate of the training value, so blanks
    are dropped here instead of being silently coerced to 0.0.
    """
    xs, ys = [], []
    for r in rows:
        if r.get("phase") != phase:
            continue
        v = r.get(field, "")
        if v == "" or v is None:
            continue
        xs.append(int(r["step"]))
        ys.append(float(v))
    return xs, ys


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # 1600x900 at 150 dpi  ->  figsize = (1600/150, 900/150)
    fig, axes = plt.subplots(2, 3, figsize=(1600 / 150, 900 / 150), dpi=150)
    axes = axes.flatten()

    last_totals, n_seeds = [], 0
    for seed in range(5):
        rows = _read(seed)
        if rows is None:
            continue
        n_seeds += 1
        for ax, (field, _) in zip(axes, PANELS):
            xs, ys = _series(rows, "train", field)
            if xs:
                ax.plot(xs, ys, color=COLORS[seed], alpha=0.85, linewidth=1.0,
                        label=f"Seed {seed}")
        # validation ELBO on the total-loss panel only, dashed
        vx, vy = _series(rows, "val", "loss_total")
        if vx:
            axes[0].plot(vx, vy, color=COLORS[seed], alpha=0.55, linewidth=1.0,
                         linestyle="--", label=f"Seed {seed} (val)")
        tx, ty = _series(rows, "train", "loss_total")
        if ty:
            last_totals.append(ty[-1])

    for ax, (field, title) in zip(axes, PANELS):
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Step", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        if field == "lr":
            ax.set_yscale("log")
        ax.legend(fontsize=5, ncol=2)

    axes[-1].axis("off")   # 6th panel unused: 5 tracked fields

    band = (f"final train ELBO in [{min(last_totals):.3f}, {max(last_totals):.3f}]"
            if last_totals else "no seeds found")
    fig.suptitle(
        f"LS4 loss convergence -- multi-asset Heston d = 8, T = 252, {n_seeds} seeds  |  "
        f"solid = train, dashed = validation  |  {band}",
        fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(PLOTS_DIR, "loss_convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}  ({n_seeds} seeds, {band})")


if __name__ == "__main__":
    main()
