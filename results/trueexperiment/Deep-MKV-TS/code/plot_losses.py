"""Deep-MKV-TS -- TrueDataset (real crypto, d = 8) convergence, all seeds overlaid.

WHY THIS FIGURE HAS TWO ROWS AND NOT TWO PANELS
------------------------------------------------
The CSDI/LS4 siblings plot a train curve beside a validation curve because both
are the SAME objective evaluated on two splits.  Deep-MKV-TS has no such pair.
Its trainer writes one phase only -- ``train`` -- because the quantity that
decides which checkpoint is reported is not the training loss at all: it is the
section-7 screen run afterwards by ``select_checkpoint_true.py``, on the
VALIDATION split, over the six saved checkpoints.

So the two rows are two different things, and reading them as train-vs-val is
the mistake this docstring exists to prevent:

    row 1  OPTIMISATION   what the optimiser saw     losses/seed_N_losses.csv
    row 2  SELECTION      what chose the checkpoint  selection/seed_N_selection.json

Row 2 has SIX points per seed, not 3000: ``CHECKPOINT_STEPS = (500, 1000, 1500,
2000, 2500, 3000)``.  Scoring a checkpoint means sampling a bank and computing
the envelope statistics, which costs far more than a training step, so the
selection grid is coarse by construction.  The markers are the whole series --
they are not a subsample of a denser curve.

WHAT ROW 1 IS AND IS NOT EVIDENCE OF
-------------------------------------
``loss_total`` is the MMD discrepancy against the target bank plus the running
cost; ``objective`` is the running cost alone.  They are on wildly different
scales (~1e3 versus ~1e1), which is why they get their own panels rather than
being overlaid, and why the y-axes are logarithmic.  Neither is a held-out
quantity.  A ``loss_total`` that keeps falling is NOT evidence the reported
model improved -- section 7 forbids selecting on goodness of fit, so a seed can
descend beautifully here and still be selected at step 1000.  That disagreement
is the point of showing both rows together.

``grad_norm`` is plotted because this method has a documented divergence mode:
one NaN in the summed running cost makes every gradient NaN, and
``clip_grad_norm_`` rescales finite gradients rather than filtering them, so it
cannot repair it (see ``eigh_float64.py``).  A grad_norm spike immediately
before a seed disappears from the figure is the signature.

WHY THE CEILINGS ARE READ AND NOT WRITTEN DOWN
-----------------------------------------------
The vol and corr ceilings drawn on row 2 are properties of the DATASET VARIANT,
measured real-against-real.  They are read out of each seed's selection JSON
(``envelope.vol_err_pct_max``, ``envelope.corr_err_max``) rather than pasted in
as constants, so that rebuilding the variant moves the lines automatically
instead of leaving the figure quietly asserting last month's envelope.

Reads   ../losses/seed_{i}_losses.csv
            step,phase,loss_total,objective,grad_norm,control_rms,elapsed_sec
        ./selection/seed_{i}_selection.json    (optional -- row 2 is skipped
            with a printed note until select_checkpoint_true.py has run)
Writes  ../plots/loss_convergence.png   (1800x1000 at 150 dpi)

Usage:
    /home/tbasseras/gpu-venv/bin/python \
        results/trueexperiment/Deep-MKV-TS/code/plot_losses.py
    ... --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)  # results/trueexperiment/Deep-MKV-TS
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")
PLOTS_DIR = os.path.join(METHOD_DIR, "plots")
SELECTION_DIR = os.path.join(HERE, "selection")

COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

# (csv column, panel title, log y-axis?)
TRAIN_PANELS = [
    ("loss_total", "Training: total loss -- MMD discrepancy + running cost", True),
    ("objective", "Training: running cost alone (specific-entropy control)", True),
    ("grad_norm", "Training: gradient norm (pre-clip, clip = 5.0)", True),
]
# (selection key, panel title, ceiling key or None)
SELECT_PANELS = [
    ("abs_log_nn_ratio", "Selection: |log NNratio| -- MINIMISED (section 7)", None),
    ("vol_err_pct", "Selection: vol_err % -- hard ceiling", "vol_err_pct_max"),
    ("corr_err", "Selection: corr_err -- hard ceiling", "corr_err_max"),
]


def _read_losses(seed: int):
    path = os.path.join(LOSSES_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _read_selection(seed: int):
    path = os.path.join(SELECTION_DIR, f"seed_{seed}_selection.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _series(rows, column: str):
    """(steps, values, n_dropped) for one column, dropping unplottable cells.

    Non-finite cells are DROPPED from the line rather than plotted, because a
    single inf would rescale a log axis and erase every real value on it.  The
    count is returned so the caller can say so instead of hiding it.
    """
    xs, ys, dropped = [], [], 0
    for r in rows:
        if r.get("phase") != "train":
            continue
        raw = r.get(column, "")
        if raw in ("", None):
            continue
        y = float(raw)
        if not math.isfinite(y) or y <= 0.0:
            dropped += 1
            continue
        xs.append(int(r["step"]))
        ys.append(y)
    return xs, ys, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(1800 / 150, 1000 / 150), dpi=150)

    n_train, n_select, dropped_total = 0, 0, 0
    final_steps, selected_steps, ceilings = [], [], {}

    for i, seed in enumerate(seeds):
        color = COLORS[i % len(COLORS)]

        rows = _read_losses(seed)
        if rows is None:
            print(f"  seed {seed}: no losses/seed_{seed}_losses.csv -- skipped")
        else:
            n_train += 1
            for ax, (column, _, _) in zip(axes[0], TRAIN_PANELS):
                xs, ys, dropped = _series(rows, column)
                dropped_total += dropped
                if xs:
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        alpha=0.85,
                        linewidth=1.0,
                        label=f"Seed {seed}",
                    )
            xs, _, _ = _series(rows, "loss_total")
            if xs:
                final_steps.append(max(xs))

        record = _read_selection(seed)
        if record is None:
            continue
        n_select += 1
        env = record.get("envelope", {})
        for key in ("vol_err_pct_max", "corr_err_max"):
            if key in env:
                ceilings[key] = float(env[key])
        chosen = record.get("selected_step")
        if chosen is not None:
            selected_steps.append(int(chosen))

        # A diverged checkpoint is recorded with +inf, deliberately (see
        # select_checkpoint_true.py).  Dropping it from the line would make a
        # seed that blew up at step 2000 look like one never scored there, so it
        # is drawn as an x at the top of the axis instead.
        per_step = sorted(record.get("per_step", []), key=lambda r: r["step"])
        for ax, (key, _, _) in zip(axes[1], SELECT_PANELS):
            good = [r for r in per_step if math.isfinite(r.get(key, float("inf")))]
            bad = [r for r in per_step if not math.isfinite(r.get(key, float("inf")))]
            gx = [r["step"] for r in good]
            gy = [r[key] for r in good]
            if gx:
                ax.plot(
                    gx,
                    gy,
                    color=color,
                    alpha=0.85,
                    linewidth=1.0,
                    marker="o",
                    markersize=3.5,
                    label=f"Seed {seed}",
                )
            if bad and gy:
                ax.plot(
                    [r["step"] for r in bad],
                    [max(gy)] * len(bad),
                    color=color,
                    alpha=0.9,
                    linestyle="none",
                    marker="x",
                    markersize=6,
                )
            if chosen is not None:
                hit = [r for r in good if r["step"] == chosen]
                if hit:
                    ax.plot(
                        [chosen],
                        [hit[0][key]],
                        color=color,
                        marker="*",
                        markersize=13,
                        linestyle="none",
                        markeredgecolor="black",
                        markeredgewidth=0.4,
                    )

    if not n_train:
        raise SystemExit(f"ABORT: no seed_*_losses.csv under {LOSSES_DIR}")

    for ax, (_, title, log_y) in zip(axes[0], TRAIN_PANELS):
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("Optimizer step", fontsize=7)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    for ax, (key, title, ceil_key) in zip(axes[1], SELECT_PANELS):
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("Checkpoint step", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        if not n_select:
            ax.text(
                0.5,
                0.5,
                "select_checkpoint_true.py\nhas not run yet",
                ha="center",
                va="center",
                fontsize=8,
                color="#777777",
                transform=ax.transAxes,
            )
            continue
        if ceil_key and ceil_key in ceilings:
            ax.axhline(
                ceilings[ceil_key],
                color="black",
                linestyle="--",
                linewidth=1.0,
                label=f"ceiling {ceilings[ceil_key]:.4g}",
            )
        if key == "abs_log_nn_ratio":
            ax.axhline(
                0.0,
                color="black",
                linestyle=":",
                linewidth=1.0,
                label="NNratio = 1 (target)",
            )
        ax.legend(fontsize=6)

    budget = f"{max(final_steps)} steps" if final_steps else "unknown budget"
    if selected_steps and final_steps:
        at_edge = sum(1 for s in selected_steps if s == max(final_steps))
        sel = (
            f"selected steps {sorted(selected_steps)}"
            f"  ({at_edge}/{len(selected_steps)} at the last checkpoint)"
        )
    else:
        sel = "no checkpoint selected yet"
    fig.suptitle(
        f"Deep-MKV-TS convergence and section-7 selection -- TrueDataset "
        f"d = 8, T = 128, {n_train} seeds, {budget}  |  log y on row 1  |  {sel}",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(PLOTS_DIR, "loss_convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  ({n_train} seeds trained, {n_select} seeds selected)")

    if dropped_total:
        print(
            f"  ! {dropped_total} non-finite or non-positive training cells were "
            f"dropped from the log axes. That is a divergence signature, not a "
            f"plotting detail -- check losses/failed_seeds.txt before reporting."
        )
    if selected_steps and final_steps and max(selected_steps) == max(final_steps):
        n_edge = sum(1 for s in selected_steps if s == max(final_steps))
        print(
            f"  ! {n_edge} seed(s) selected the LAST checkpoint ({max(final_steps)}). "
            f"The selection grid is bounded above by the training budget, so this "
            f"is a boundary and not demonstrably an optimum -- section 7.2. Say so "
            f"in the README rather than quoting the A-table as converged."
        )


if __name__ == "__main__":
    main()
