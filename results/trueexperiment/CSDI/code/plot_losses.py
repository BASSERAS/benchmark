"""CSDI -- TrueDataset (real crypto, d = 8) loss convergence, all seeds overlaid.

CSDI optimises a **single** term: the DDPM denoising objective
``E_t || eps - eps_theta(x_t, t) ||^2`` restricted to ``target_mask``, which in the
unconditional regime is 1 everywhere (see ``csdi_true.CSDI_TrueData.forward``).
There is no KLD/NLL/reconstruction decomposition to break out, so the figure is
two panels rather than LS4's five: train and validation, each with every seed
overlaid.

THE TWO PANELS ARE NOT THE SAME QUANTITY PLOTTED TWICE
-------------------------------------------------------
``calc_loss`` draws one random diffusion step ``t`` per sample, so the train curve
is a one-sample estimate and is dominated by that sampling noise. ``calc_loss_valid``
averages over **all 50** diffusion steps, so the validation curve is the
low-variance one and is the series to read convergence from. Their *levels* are
not comparable for the same reason -- they are two different estimators of two
different averages, not train/test versions of one number. A validation curve
sitting *below* the training curve here is expected and is evidence of nothing.
Do not report the gap between them as an overfitting measure.

WHAT THIS FIGURE IS ACTUALLY FOR ON THIS BUILD
-----------------------------------------------
``epochs = 200`` is the authors' published number, kept verbatim (see
``csdi_true.BASE_CONFIG``). On TrueDataset that is 384 steps/epoch = 76 800
steps, 25 % fewer than the d = 8 Heston sibling got at the same nominal setting.
Nobody re-tuned it. So the validation panel is the only evidence that 200 was
enough here, and it is a precondition for trusting the A-table, not an
appendix figure: if the val curve is still falling at the right edge, the bank
was drawn from an undertrained model and every metric downstream inherits that.

The validation series carries 20 POINTS, not 200: ``train_true.py`` sets
``val_every = max(1, epochs // 20)``, i.e. one pass every 10 epochs at
``epochs=200``, because a full-split validation pass costs ~50x a training epoch
(it averages over all 50 diffusion steps). Skipped epochs are written as
blank/nan and dropped here, which is why the val panel is marker-dotted and the
train panel is not -- and why ``tail_drop`` below is computed over the last
``len(vy)//5`` = 4 validation points, i.e. the last 20 % of TRAINING, not the
last 20 % of the plotted markers by accident.

Reads   ../losses/seed_{i}_losses.csv   columns: step,phase,loss_total
Writes  ../plots/loss_convergence.png   (1600x900 at 150 dpi)

Usage:
    /home/tbasseras/gpu-venv/bin/python \
        results/trueexperiment/CSDI/code/plot_losses.py
    ... --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)                     # results/trueexperiment/CSDI
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")
PLOTS_DIR = os.path.join(METHOD_DIR, "plots")

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    os.makedirs(PLOTS_DIR, exist_ok=True)
    # 1600x900 at 150 dpi  ->  figsize = (1600/150, 900/150)
    fig, axes = plt.subplots(1, 2, figsize=(1600 / 150, 900 / 150), dpi=150)

    final_val, tail_drop, n_seeds = [], [], 0
    for i, seed in enumerate(seeds):
        rows = _read(seed)
        if rows is None:
            print(f"  seed {seed}: no losses/seed_{seed}_losses.csv -- skipped")
            continue
        n_seeds += 1
        for ax, (phase, _) in zip(axes, PANELS):
            xs, ys = _series(rows, phase)
            if xs:
                ax.plot(xs, ys, color=COLORS[i % len(COLORS)], alpha=0.85, linewidth=1.0,
                        marker="o" if phase == "val" else None, markersize=2.5,
                        label=f"Seed {seed}")
        vx, vy = _series(rows, "val")
        if vy:
            final_val.append(vy[-1])
            # How much of the total descent happened in the last 20 % of training.
            # Large => the curve was still moving when the budget ran out.
            if len(vy) >= 5:
                k = max(1, len(vy) // 5)
                span = max(vy) - min(vy)
                if span > 0:
                    tail_drop.append((vy[-k] - vy[-1]) / span)

    if not n_seeds:
        raise SystemExit(f"ABORT: no seed_*_losses.csv under {LOSSES_DIR}")

    for ax, (_, title) in zip(axes, PANELS):
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("loss_total", fontsize=7)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    band = (f"final validation loss in [{min(final_val):.4f}, {max(final_val):.4f}]"
            if final_val else "no validation points")
    tail = (f"  |  last 20 % of steps account for "
            f"{100 * max(tail_drop):.1f} % of the val descent (worst seed)"
            if tail_drop else "")
    fig.suptitle(
        f"CSDI loss convergence -- TrueDataset d = 8, T = 128, {n_seeds} seeds  |  "
        f"log y-axis  |  {band}{tail}",
        fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(PLOTS_DIR, "loss_convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  ({n_seeds} seeds)")
    if tail_drop and max(tail_drop) > 0.10:
        print(f"  ! the val curve was still descending at the right edge "
              f"({100 * max(tail_drop):.1f} % of the drop in the last 20 % of steps) "
              f"-- 200 epochs may be short. Say so in the README rather than "
              f"quoting the A-table as if the model had converged.")


if __name__ == "__main__":
    main()
