"""SBBTS — Heston loss convergence plot (5 seeds overlaid).

SBBTS trains by DSBM: K outer iterations, each an early-stopped inner fit of the
score network to the current bridge coupling. The model and the Adam state are
built ONCE outside the K loop (see sbbts_torch.training_sbbts_dsbm), so every
outer iteration warm-starts from the previous one and the loss is a single
continuous trajectory rather than K independent restarts. That is why this plots
one uninterrupted curve per seed with the iteration boundaries drawn as vertical
guides, instead of K separate panels.

Two components are logged per step, so two panels (GUIDELINE 4.4 asks for one
subplot per loss component): the training loss and the held-out validation loss
that drives early stopping.

Reads   ../losses/seed_{0..4}_losses.csv   (columns: step,phase,loss_total,val_loss)
Writes  ../losses/loss_convergence.png     (1600x900 px, 150 dpi, GUIDELINE 4.4)

Usage:
    /home/tbasseras/gpu-venv/bin/python methods/SBBTS/code/plot_losses.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)                       # methods/SBBTS
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]


def _read(seed):
    """-> (steps, train, val, boundaries) or None if the seed has no CSV yet.

    ``boundaries`` are the steps at which ``phase`` changes, i.e. where one DSBM
    outer iteration hands over to the next.
    """
    path = os.path.join(LOSSES_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None
    steps, train, val, bounds, prev = [], [], [], [], None
    with open(path) as f:
        for r in csv.DictReader(f):
            step = int(r["step"])
            if prev is not None and r["phase"] != prev:
                bounds.append(step)
            prev = r["phase"]
            steps.append(step)
            train.append(float(r["loss_total"]))
            val.append(float(r["val_loss"]))
    return steps, train, val, bounds


def main():
    # 1600 x 900 px at 150 dpi.
    fig, (ax_tr, ax_va) = plt.subplots(1, 2, figsize=(16 / 1.5, 9 / 1.5), dpi=150)

    final_val, all_bounds, n_seeds = [], [], 0
    for seed in range(5):
        got = _read(seed)
        if got is None:
            continue
        n_seeds += 1
        steps, train, val, bounds = got
        all_bounds.append(bounds)
        ax_tr.plot(steps, train, color=COLORS[seed], alpha=0.85, linewidth=1.1,
                   label=f"Seed {seed}")
        ax_va.plot(steps, val, color=COLORS[seed], alpha=0.85, linewidth=1.1,
                   label=f"Seed {seed} (last={val[-1]:.3f})")
        final_val.append(val[-1])

    if not n_seeds:
        raise SystemExit(f"no seed_*_losses.csv under {LOSSES_DIR}; run training first")

    # Draw DSBM iteration handovers from seed 0 only -- early stopping makes the
    # boundaries seed-dependent, so overlaying all five would be unreadable.
    for b in (all_bounds[0] if all_bounds else []):
        for ax in (ax_tr, ax_va):
            ax.axvline(b, color="0.55", linestyle=":", linewidth=0.8, zorder=0)

    ax_tr.set_title("DSBM training loss", fontsize=12)
    ax_va.set_title("Validation loss (drives early stopping)", fontsize=12)
    for ax in (ax_tr, ax_va):
        ax.set_xlabel("Step (epoch, cumulative across DSBM iterations)")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    band = (f"final val ∈ [{min(final_val):.3f}, {max(final_val):.3f}]"
            if final_val else "")
    plt.suptitle(
        f"SBBTS Loss Convergence — Heston seq_len=128, {n_seeds} seeds  "
        f"(dotted = DSBM outer-iteration handover, seed 0; {band})",
        fontsize=12)
    plt.tight_layout()

    out = os.path.join(LOSSES_DIR, "loss_convergence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}  ({n_seeds} seeds; {band})")


if __name__ == "__main__":
    main()
