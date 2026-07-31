"""Loss-convergence figure for the README's Losses section.

Reads every ``losses/seed_*_losses.csv`` under a method directory and draws one panel per
logged quantity. Method-neutral: the panel list is taken from the CSV header, so a method
that logs different columns than LS4's ELBO split still plots.

Usage:
  python plot_losses.py --model-dir ../experiment_A/LS4 --title "LS4 (Experiment A)"
"""
import os
import csv
import glob
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 120
WARMUP_EPOCHS = 5


def read_series(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--title", default="Training loss")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.model_dir, "losses", "seed_*_losses.csv")))
    if not paths:
        raise SystemExit(f"no seed_*_losses.csv under {a.model_dir}/losses")

    series = {os.path.basename(p).split("_")[1]: read_series(p) for p in paths}
    columns = [c for c in next(iter(series.values())) if c != "epoch"]

    ncol = min(len(columns), 3)
    nrow = (len(columns) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.8 * nrow), squeeze=False)
    flat = axes.ravel()

    for ax, col in zip(flat, columns):
        tail = []
        for seed, s in sorted(series.items()):
            ax.plot(s["epoch"], s[col], lw=1.2, label=f"seed {seed}")
            tail += [v for e, v in zip(s["epoch"], s[col]) if e >= WARMUP_EPOCHS]
        ax.set_title(col)
        ax.set_xlabel("epoch")
        if col == "lr":
            ax.set_yscale("log")
        elif tail:
            # epoch-0 values are 1-2 orders above convergence; without clipping the
            # converged region is a flat line and the panel says nothing.
            lo, hi = min(tail), max(tail)
            pad = 0.1 * (hi - lo) + 1e-9
            ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.3)
    for ax in flat[len(columns):]:
        ax.axis("off")
    flat[0].legend(fontsize=8)

    fig.suptitle(f"{a.title} — {len(series)} seeds", fontsize=13)
    fig.tight_layout()
    out = os.path.join(a.model_dir, "losses", "loss_convergence.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(series)} seeds: {', '.join(sorted(series))})")


if __name__ == "__main__":
    main()
