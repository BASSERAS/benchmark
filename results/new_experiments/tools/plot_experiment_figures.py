"""Experiment-specific figures for README section 1 (Stage 4 of the methodology).

Experiment A: the memory structure. future_rv split by early_hit, real vs generated -- this is
the picture behind ``early_history_incremental_r2``. A model with no memory shows the two
generated histograms on top of each other.

Experiment B: the mixture structure. Oracle-predicted regime proportions, target vs generated,
plus the per-parameter spread that ``std_ratio`` summarises. A collapsed model shows a narrow
parameter spread even when its 8 bars look right.

Experiment A features are re-derived here from the frozen manifest rather than imported: the
protocol scripts are vendored verbatim and expose no importable feature API. The printed
hit-gap must match ``future_rv_hit_gap`` in the evaluator JSON -- check it.

Usage:
  python plot_experiment_figures.py --experiment A \
      --model-dir ../experiment_A/LS4 --seed 0 --out ../experiment_A/LS4/plots
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REAL_COL = "#2196F3"
GEN_COL = "#FF5722"
DPI = 120

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def data_dir(experiment):
    return os.path.join(BENCH_ROOT, "dataset", "Heston", "new_experiments",
                        f"experiment_{experiment}")


def drawdown_features(prices, cfg):
    """Mirrors evaluate_drawdown_memory.drawdown_features, for plotting only.

    Drawdown is in LOG space and rv is sqrt(ann * mean(r**2)) with ann = 1/dt --
    not 1 - P/max and not std * sqrt(250). Diverging from this silently shifts
    every number in the figure away from the evaluator's.
    """
    prices = np.asarray(prices, dtype=np.float64)
    log_paths = np.log(prices / prices[:, :1])
    returns = np.diff(log_paths, axis=1)
    drawdown = np.maximum.accumulate(log_paths, axis=1) - log_paths

    cutoff = int(cfg["history_cutoff"])
    split = int(cfg["split_index"])
    horizon = int(cfg["future_horizon"])
    window = int(cfg["recent_window"])
    threshold = float(cfg["drawdown_threshold"])
    ann = 1.0 / float(cfg["dt"])

    early_hit = (drawdown[:, : cutoff + 1].max(axis=1) >= threshold).astype(float)
    recent_rv = np.sqrt(ann * np.mean(returns[:, split - window: split] ** 2, axis=1))
    future_rv = np.sqrt(ann * np.mean(returns[:, split: split + horizon] ** 2, axis=1))
    return early_hit, recent_rv, future_rv


def figure_a(args):
    cfg = json.load(open(os.path.join(data_dir("A"), "manifest.json")))["configuration"]
    real = np.load(os.path.join(data_dir("A"), "test.npy"))
    gen = np.load(os.path.join(args.model_dir, "generated_paths", f"seed_{args.seed}",
                               "generated_paths_8192x128.npy"))

    hr, _, fr = drawdown_features(real, cfg)
    hg, _, fg = drawdown_features(gen, cfg)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.linspace(0, max(fr.max(), fg.max()) * 1.02, 60)

    for ax, hit, fut, title, col in (
        (axes[0], hr, fr, "Real (test.npy)", REAL_COL),
        (axes[1], hg, fg, f"{args.label} seed {args.seed}", GEN_COL),
    ):
        ax.hist(fut[hit == 0], bins=bins, alpha=0.65, density=True,
                color=col, label="no early drawdown")
        ax.hist(fut[hit == 1], bins=bins, alpha=0.65, density=True,
                color="#333333", label="early drawdown (hit)")
        gap = fut[hit == 1].mean() - fut[hit == 0].mean()
        ax.set_title(f"{title}\nhit gap = {gap:.4f}")
        ax.set_xlabel("future realised vol")
        ax.legend(fontsize=8)

    ax = axes[2]
    width = 0.35
    for i, (hit, fut, name, col) in enumerate(
            ((hr, fr, "Real", REAL_COL), (hg, fg, args.label, GEN_COL))):
        ax.bar([0 + (i - 0.5) * width, 1 + (i - 0.5) * width],
               [fut[hit == 0].mean(), fut[hit == 1].mean()],
               width, color=col, label=name)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no hit", "early hit"])
    ax.set_ylabel("mean future realised vol")
    ax.set_title("Volatility response to early drawdown")
    ax.legend(fontsize=8)

    fig.suptitle("Experiment A - delayed drawdown memory", fontsize=13)
    fig.tight_layout()
    out = os.path.join(args.out, f"memory_structure_seed{args.seed}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  real hit_gap = {fr[hr == 1].mean() - fr[hr == 0].mean():.6f}")
    print(f"  gen  hit_gap = {fg[hg == 1].mean() - fg[hg == 0].mean():.6f}")


def figure_b(args):
    ev = json.load(open(os.path.join(args.model_dir, "pdf_metrics",
                                     f"seed_{args.seed}_heston_mixture.json")))
    mf = ev["mixture_fidelity"]
    target = np.asarray(mf["target_regime_proportions"], dtype=float)
    generated = np.asarray(mf["generated_regime_proportions"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    idx = np.arange(8)
    width = 0.38
    axes[0].bar(idx - width / 2, target, width, color=REAL_COL, label="Real (test.npy)")
    axes[0].bar(idx + width / 2, generated, width, color=GEN_COL, label=args.label)
    axes[0].axhline(0.125, ls="--", lw=1, color="#666666", label="balanced 1/8")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels([f"{l}\nT{(l >> 2) & 1}X{(l >> 1) & 1}R{l & 1}" for l in idx],
                            fontsize=8)
    axes[0].set_ylabel("oracle-predicted proportion")
    axes[0].set_title(f"Regime proportions  (TVD = {mf['regime_proportion_tvd']:.5f})")
    axes[0].legend(fontsize=8)

    names = ["theta", "xi", "rho"]
    ratios = [mf["parameters"][n]["std_ratio"] for n in names]
    axes[1].bar(names, ratios, color=GEN_COL)
    axes[1].axhline(1.0, ls="--", lw=1, color="#666666", label="target spread")
    axes[1].set_ylim(0, max(1.25, max(ratios) * 1.15))
    axes[1].set_ylabel("std(generated) / std(target)")
    axes[1].set_title("Parameter spread  (<1 = mode collapse)")
    axes[1].legend(fontsize=8)
    for i, r in enumerate(ratios):
        axes[1].text(i, r + 0.02, f"{r:.4f}", ha="center", fontsize=9)

    fig.suptitle("Experiment B - balanced Heston parameter mixture", fontsize=13)
    fig.tight_layout()
    out = os.path.join(args.out, f"mixture_structure_seed{args.seed}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  tvd = {mf['regime_proportion_tvd']:.6f}   std_ratios = "
          + ", ".join(f"{n}={r:.4f}" for n, r in zip(names, ratios)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=("A", "B"))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="LS4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    (figure_a if args.experiment == "A" else figure_b)(args)


if __name__ == "__main__":
    main()
