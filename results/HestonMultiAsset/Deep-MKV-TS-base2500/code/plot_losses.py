"""Deep-MKV-TS -- multi-asset Heston (d = 8) loss convergence, 5 seeds overlaid.

Why this is NOT the CSDI script
-------------------------------
``methods/CSDI/code/plot_losses.py`` and its multi-asset sibling split the two
panels on the ``phase`` column, train vs val.  Deep-MKV-TS has no such split:
``train_multiasset.py`` line 293 hardcodes ``"phase": "train"`` at the only
row-writing site, so a phase-keyed layout renders one empty axis and a
"no seeds found" suptitle.  Algorithm 1 has no held-out forward pass to log --
model selection happens *after* training, in ``select_checkpoint_multiasset.py``,
by scoring saved checkpoints on the validation bank.

The panels are therefore keyed on COLUMN, and they are two genuinely different
quantities, not the same number twice:

``loss_total``  is L_adj = (1/PM) sum || Zhat^theta - Z^proxy ||^2, the adjoint
    matching loss AdamW actually minimises.  Its target MOVES: ``Z^proxy`` is
    re-estimated by ridge regression on freshly sampled paths at every outer
    iteration, so ``loss_total`` is a distance to a target that is itself
    drifting.  Descending is necessary but NOT sufficient -- it can fall simply
    because the proxy got easier to hit.  Do not read convergence from it alone.

``objective``   is the path-functional discrepancy between generated paths and
    the training bank (``build_heston_discrepancy``, 13 blocks at d = 8).  This
    is a fixed yardstick and is the same functional form used by
    ``select_checkpoint_multiasset.py`` to choose the reported checkpoint.  Read
    convergence from this panel.

Both are logged on the same rows, every ``LOG_EVERY = 100`` steps, so the x-axes
coincide and the panels can be compared step-for-step.

Reads   ../losses/seed_{0,2,4,5,6}_losses.csv   (seeds 1 and 3 diverged; see README.md)
        columns: step,phase,loss_total,objective,grad_norm,control_rms,elapsed_sec
Writes  ../plots/loss_convergence.png      (1600x900 at 150 dpi)

Usage:
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/Deep-MKV-TS/code/plot_losses.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)              # results/HestonMultiAsset/Deep-MKV-TS
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")
PLOTS_DIR = os.path.join(METHOD_DIR, "plots")

COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
# Not range(5): TWO seeds diverged with a non-finite control -- seed 3 at step 2
# and seed 1 at step ~1450 -- and were replaced by seeds 5 and 6 respectively
# (see "The caveat came true twice" in README.md). seed_1_losses.csv and
# seed_3_losses.csv still exist and hold the steps logged before each death;
# plotting a truncated curve alongside five complete 3000-step ones would be a
# misleading panel, not a record of the failure. The failures are recorded in the
# README with their step counts, where they can actually be read.
# Colours are indexed by POSITION, not by seed id, so seed 6 does not run off
# the end of a 5-entry palette.
SEEDS = [0, 2, 4, 5, 6]
PANELS = [
    ("loss_total",
     "Adjoint-matching loss L_adj -- minimised by AdamW (target moves each outer iter)"),
    ("objective",
     "Path-functional discrepancy -- fixed yardstick, the selection criterion"),
]


def _read(seed):
    path = os.path.join(LOSSES_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _series(rows, column):
    """(steps, values) for one column, skipping blank and non-finite cells."""
    xs, ys = [], []
    for r in rows:
        v = r.get(column, "")
        if v == "" or v is None:
            continue
        y = float(v)
        if y != y:          # nan: metric unavailable on that row
            continue
        xs.append(int(r["step"]))
        ys.append(y)
    return xs, ys


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # 1600x900 at 150 dpi  ->  figsize = (1600/150, 900/150)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(1600 / 150, 900 / 150), dpi=150)

    final_obj, n_seeds = [], 0
    for i, seed in enumerate(SEEDS):
        rows = _read(seed)
        if rows is None:
            continue
        n_seeds += 1
        for ax, (column, _) in zip(axes, PANELS):
            xs, ys = _series(rows, column)
            if xs:
                ax.plot(xs, ys, color=COLORS[i % len(COLORS)], alpha=0.85,
                        linewidth=1.0, label=f"Seed {seed}")
        _, oy = _series(rows, "objective")
        if oy:
            final_obj.append(oy[-1])

    for ax, (_, title) in zip(axes, PANELS):
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Outer iteration (Algorithm 1 step)", fontsize=7)
        ax.set_ylabel("value", fontsize=7)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    band = (f"final objective in [{min(final_obj):.4f}, {max(final_obj):.4f}]"
            if final_obj else "no seeds found")
    fig.suptitle(
        f"Deep-MKV-TS loss convergence -- multi-asset Heston d = 8, T = 252, "
        f"{n_seeds} seeds  |  log y-axis  |  {band}",
        fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(PLOTS_DIR, "loss_convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  ({n_seeds} seeds)")


if __name__ == "__main__":
    main()
