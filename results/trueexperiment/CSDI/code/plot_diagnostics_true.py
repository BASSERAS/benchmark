#!/usr/bin/env python3
"""
plot_diagnostics_true.py
────────────────────────
The 8-panel stylised-facts figure for the d = 8 **real-data** (TrueDataset) run.
Panel-for-panel identical to `results/HestonMultiAsset/*/code/
plot_diagnostics_multiasset.py` and to the copy in every sibling method tree, so
every README displays the same figure in the same slot and a reader can put the
synthetic and the real result -- or two methods -- side by side without
re-learning the layout. This file is a verbatim copy of
`results/trueexperiment/SBTS/code/plot_diagnostics_true.py` apart from renaming
one path constant; keep it that way, because a panel that drifts between two
method trees turns a comparison into a category error.

Two deliberate differences from the Heston copy
───────────────────────────────────────────────
1. **The reference curve is different, and weaker.** On Heston the third curve
   is an *independent draw from the true SDE* -- a genuine second sample of the
   data-generating law. Real markets have no such law to re-draw from, so the
   third curve here is a **real-vs-real** reference: a held-out REAL split
   (train / val / valdisc) plotted against the test split. It answers a narrower
   question -- "how far apart do two real samples of this market sit?" -- and
   because the build is holdout-era, part of that distance is the regime change
   across the embargo, not finite-sample noise. Do not read it as a floor a
   generator *should* reach; read it as the score a perfect memoriser of the
   training era gets on the test era.
2. **No theory curve at all.** `metrics/heston_theory.py` is a closed-form
   Heston reference and there is no analogue for BTC/ETH/... 30-second bars.

Which asset
-----------
One asset, not eight. The *metrics* are per-asset-averaged (see
metrics_per_asset.csv); the *figure* is a single representative asset, and the
title says which. Pooling eight assets with different volatilities into one
histogram would show the mixing, not the model.

Usage
-----
    /home/tbasseras/gpu-venv/bin/python \
        results/trueexperiment/CSDI/code/plot_diagnostics_true.py \
        --data-dir dataset/TrueDataset/variants/om_2022-07_N6144 \
        --seq-tag 6144x128x8 --seed 0 --asset 0
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Named for what it is rather than for one method: this file is copied verbatim
# into every method tree, and a variable called SBTS sitting in CSDI/code/ is a
# lie the reader has to decode. The value is identical either way.
METHOD_DIR = os.path.dirname(HERE)                 # .../CSDI
MA = os.path.dirname(METHOD_DIR)                   # .../trueexperiment
REPO = os.path.abspath(os.path.join(MA, "../.."))

# TrueDataset ships as *variants*: a build is pinned by (period, N, split-mode),
# so there is no single canonical directory or tag the way HestonMultiAsset has
# one. Both are CLI arguments with the locked build as the default.
DEFAULT_DATA = os.path.join(REPO, "dataset", "TrueDataset", "variants",
                            "om_2022-07_N6144")
DEFAULT_TAG = "6144x128x8"

# benchmark-standard constants -- identical to metrics/plot_diagnostics.py
N_SHOW = 50        # individual paths drawn in panels 0 & 1
N_STAT = 5000      # paths used for the statistical panels 2-7
ACF_LAGS = 20
VOL_WIN = 5
RNG_SEED = 42
REAL_COL = "#2196F3"
GEN_COL = "#FF5722"
FLOOR_COL = "#000000"
FLOOR_LBL = "Real split (real-vs-real)"   # overwritten from metadata.json

# The floor seeds are REAL splits standing in for generated banks, and they are
# not interchangeable: seed 0 is `train`, seed 1 `val`, seed 2 `valdisc`. Calling
# any of them "held-out" in the legend would be a lie for seed 0, so the legend
# is read off the floor's own metadata.json rather than asserted here.
SPLIT_NICE = {"true_S": "train", "true_S_val": "val",
              "true_S_valdisc": "valdisc", "true_S_test": "test"}
DPI = 120


def _log_returns(S):
    """(N, T) prices -> (N, T-1) log-returns."""
    return np.log(S[:, 1:] / S[:, :-1])


def _acf_mean(X, max_lag=20):
    """Cross-path mean ACF. X: (N, T) -> array of length max_lag."""
    X = X - X.mean(axis=1, keepdims=True)
    var = (X ** 2).mean(axis=1)
    out = []
    for lag in range(1, max_lag + 1):
        cov = (X[:, lag:] * X[:, :-lag]).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(var > 0, cov / var, 0.0)
        out.append(float(r.mean()))
    return np.array(out)


def _rolling_vol_flat(S, window=5):
    """Flattened rolling std of log-returns over all paths."""
    R = _log_returns(S)
    cols = [R[:, t - window + 1: t + 1].std(axis=1)
            for t in range(window - 1, R.shape[1])]
    return np.stack(cols, axis=1).ravel()


def plot_diagnostics_ma(S_real, S_gen, S_floor, method, seed, asset, out_path,
                        floor_lbl=FLOOR_LBL):
    """Save the 8-panel figure. All three inputs are (N, T) price paths."""
    rng = np.random.default_rng(RNG_SEED)
    n_stat = min(N_STAT, len(S_real), len(S_gen))

    idx_show_r = rng.choice(len(S_real), N_SHOW, replace=False)
    idx_show_g = rng.choice(len(S_gen), N_SHOW, replace=False)
    idx_stat_r = rng.choice(len(S_real), n_stat, replace=False)
    idx_stat_g = rng.choice(len(S_gen), n_stat, replace=False)

    R_real = _log_returns(S_real[idx_stat_r])
    R_gen = _log_returns(S_gen[idx_stat_g])
    r_flat_r = R_real.ravel()
    r_flat_g = R_gen.ravel()

    have_floor = S_floor is not None
    if have_floor:
        idx_stat_f = rng.choice(len(S_floor), min(n_stat, len(S_floor)), replace=False)
        R_floor = _log_returns(S_floor[idx_stat_f])
        r_flat_f = R_floor.ravel()

    t = np.arange(S_real.shape[1])
    lags = np.arange(1, ACF_LAGS + 1)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(
        f"TrueDataset (real crypto, d = 8) diagnostics -- Real vs {method}  "
        f"(seed {seed}, asset {asset} of 8)",
        fontsize=14, fontweight="bold", y=0.998,
    )

    # -- [0, 0] Real sample paths --------------------------------------------
    ax = axes[0, 0]
    for i in idx_show_r:
        ax.plot(t, S_real[i], color=REAL_COL, alpha=0.25, lw=0.6)
    ax.plot(t, S_real[idx_show_r].mean(axis=0), color=REAL_COL, lw=2.0, label="mean")
    ax.set_title(f"Real paths  (n={N_SHOW} shown / {len(S_real)} total)")
    ax.set_xlabel("t (steps)"); ax.set_ylabel("S(t)")
    ax.legend(fontsize=8)

    # -- [0, 1] Generated sample paths ---------------------------------------
    ax = axes[0, 1]
    for i in idx_show_g:
        ax.plot(t, S_gen[i], color=GEN_COL, alpha=0.25, lw=0.6)
    ax.plot(t, S_gen[idx_show_g].mean(axis=0), color=GEN_COL, lw=2.0, label="mean")
    ax.set_title(f"{method} paths  (n={N_SHOW} shown / {len(S_gen)} total)")
    ax.set_xlabel("t (steps)"); ax.set_ylabel("S(t)")
    ax.legend(fontsize=8)

    # -- [1, 0] Log-return distribution --------------------------------------
    ax = axes[1, 0]
    lo, hi = np.quantile(np.concatenate([r_flat_r, r_flat_g]), [0.002, 0.998])
    bins = np.linspace(lo, hi, 80)
    ax.hist(r_flat_r, bins=bins, alpha=0.5, color=REAL_COL, density=True, label="Real")
    ax.hist(r_flat_g, bins=bins, alpha=0.5, color=GEN_COL, density=True, label=method)
    if have_floor:
        dens, edges = np.histogram(r_flat_f, bins=bins, density=True)
        ax.plot(0.5 * (edges[1:] + edges[:-1]), dens, color=FLOOR_COL,
                lw=1.6, ls="--", label=floor_lbl)
    ax.set_title("Log-return distribution")
    ax.set_xlabel("log-return"); ax.set_ylabel("density")
    ax.legend(fontsize=8)

    # -- [1, 1] QQ plot ------------------------------------------------------
    ax = axes[1, 1]
    qq_grid = np.linspace(0.005, 0.995, 300)
    q_r = np.quantile(r_flat_r, qq_grid)
    q_g = np.quantile(r_flat_g, qq_grid)
    ax.scatter(q_r, q_g, s=5, color=GEN_COL, alpha=0.6, label=method)
    lo_ = min(q_r.min(), q_g.min()); hi_ = max(q_r.max(), q_g.max())
    ax.plot([lo_, hi_], [lo_, hi_], color="gray", ls=":", lw=1, label="y = x (perfect)")
    if have_floor:
        ax.plot(q_r, np.quantile(r_flat_f, qq_grid), color=FLOOR_COL,
                lw=1.6, ls="--", label=floor_lbl)
    ax.set_title("QQ plot -- generated vs real log-returns")
    ax.set_xlabel("Real quantiles"); ax.set_ylabel("Gen / floor quantiles")
    ax.legend(fontsize=8)

    # -- [2, 0] ACF of |log-returns| -----------------------------------------
    ax = axes[2, 0]
    ax.plot(lags, _acf_mean(np.abs(R_real), ACF_LAGS),
            color=REAL_COL, marker="o", ms=3, label="Real")
    ax.plot(lags, _acf_mean(np.abs(R_gen), ACF_LAGS),
            color=GEN_COL, marker="o", ms=3, label=method)
    if have_floor:
        ax.plot(lags, _acf_mean(np.abs(R_floor), ACF_LAGS), color=FLOOR_COL,
                lw=1.6, ls="--", marker="s", ms=2.5, label=floor_lbl)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_title("ACF of |log-returns|  (volatility clustering)")
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("ACF")
    ax.legend(fontsize=8)

    # -- [2, 1] ACF of squared returns ---------------------------------------
    ax = axes[2, 1]
    ax.plot(lags, _acf_mean(R_real ** 2, ACF_LAGS),
            color=REAL_COL, marker="o", ms=3, label="Real")
    ax.plot(lags, _acf_mean(R_gen ** 2, ACF_LAGS),
            color=GEN_COL, marker="o", ms=3, label=method)
    if have_floor:
        ax.plot(lags, _acf_mean(R_floor ** 2, ACF_LAGS), color=FLOOR_COL,
                lw=1.6, ls="--", marker="s", ms=2.5, label=floor_lbl)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_title("ACF of squared log-returns  (GARCH effect)")
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("ACF")
    ax.legend(fontsize=8)

    # -- [3, 0] Rolling volatility histogram ---------------------------------
    ax = axes[3, 0]
    vol_r = _rolling_vol_flat(S_real[idx_stat_r], VOL_WIN)
    vol_g = _rolling_vol_flat(S_gen[idx_stat_g], VOL_WIN)
    hi_v = np.quantile(np.concatenate([vol_r, vol_g]), 0.99)
    bins_v = np.linspace(0, hi_v, 60)
    ax.hist(vol_r, bins=bins_v, alpha=0.5, color=REAL_COL, density=True, label="Real")
    ax.hist(vol_g, bins=bins_v, alpha=0.5, color=GEN_COL, density=True, label=method)
    if have_floor:
        vol_f = _rolling_vol_flat(S_floor[idx_stat_f], VOL_WIN)
        dens, edges = np.histogram(vol_f, bins=bins_v, density=True)
        ax.plot(0.5 * (edges[1:] + edges[:-1]), dens, color=FLOOR_COL,
                lw=1.6, ls="--", label=floor_lbl)
    ax.set_title(f"Rolling volatility  (window={VOL_WIN} days)")
    ax.set_xlabel("rolling std of log-returns"); ax.set_ylabel("density")
    ax.legend(fontsize=8)

    # -- [3, 1] Tail survival (log-log) --------------------------------------
    ax = axes[3, 1]
    term_r = S_real[:, -1]
    term_g = S_gen[:, -1]
    qq_tail = np.linspace(0.50, 0.999, 300)
    thresh = np.quantile(term_r, qq_tail)
    surv_r = np.array([(term_r > thr).mean() for thr in thresh])
    surv_g = np.array([(term_g > thr).mean() for thr in thresh])
    mask = (surv_r > 0) & (surv_g > 0)
    ax.plot(1 - qq_tail[mask], surv_r[mask], color=REAL_COL, label="Real")
    ax.plot(1 - qq_tail[mask], surv_g[mask], color=GEN_COL, label=method)
    if have_floor:
        term_f = S_floor[:, -1]
        surv_f = np.array([(term_f > thr).mean() for thr in thresh])
        mf = surv_f > 0
        ax.plot(1 - qq_tail[mf], surv_f[mf], color=FLOOR_COL, lw=1.6, ls="--",
                label=floor_lbl)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Tail survival -- terminal price S_T  (log-log)")
    ax.set_xlabel("1 - quantile"); ax.set_ylabel("P(S_T > threshold)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    # Default = the directory this copy of the script lives in, so a method that
    # copies the file into its own code/ slot gets the right tree with no flag.
    # --method must steer the DATA as well as the title, otherwise passing it
    # would silently label another method's figure over this one's paths.
    ap.add_argument("--method", default=os.path.basename(METHOD_DIR))
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--seq-tag", default=DEFAULT_TAG)
    ap.add_argument("--floor-method", default="real_floor",
                    help="method directory holding the real-vs-real reference. "
                         "Pass '' to plot without a third curve.")
    # Deliberately NOT tied to --seed. The floor seeds are three different real
    # splits (0=train, 1=val, 2=valdisc), so seed 0 of the floor is the training
    # era -- useful as a reference, but it is not held out, and there are only 3
    # of them against the generator's 5.
    ap.add_argument("--floor-seed", type=int, default=1,
                    help="which real split to draw as the third curve. Default 1 "
                         "= val, a genuinely held-out split. 0 = train.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--asset", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    method_dir = os.path.join(MA, args.method)
    if not os.path.isdir(method_dir):
        raise SystemExit(f"ABORT: no such method directory {method_dir}")

    tag = args.seq_tag
    real_p = os.path.join(args.data_dir, f"true_S_test_{tag}.npy")
    gen_p = os.path.join(method_dir, "generated_paths", f"seed_{args.seed}",
                         f"generated_paths_{tag}.npy")

    if not os.path.exists(gen_p):
        raise SystemExit(f"ABORT: {gen_p} missing -- generate the banks first.")

    j = args.asset
    S_real = np.load(real_p)[:, :, j]
    S_gen = np.load(gen_p)[:, :, j]

    S_floor, floor_lbl = None, FLOOR_LBL
    if args.floor_method:
        floor_dir = os.path.join(MA, args.floor_method, "generated_paths",
                                 f"seed_{args.floor_seed}")
        floor_p = os.path.join(floor_dir, f"generated_paths_{tag}.npy")
        if os.path.exists(floor_p):
            S_floor = np.load(floor_p)[:, :, j]
            # Name the curve after the split it actually is.
            meta_p = os.path.join(floor_dir, "metadata.json")
            if os.path.exists(meta_p):
                with open(meta_p) as fh:
                    src = json.load(fh).get("source_split", "")
                stem = re.sub(rf"_{tag}\.npy$", "", src)
                nice = SPLIT_NICE.get(stem, stem or "real")
                floor_lbl = f"Real `{nice}` split (real-vs-real)"
            print(f"  floor curve: {floor_lbl}")
        else:
            print(f"  [warn] {floor_p} missing -- plotting without the third curve")

    out = args.out or os.path.join(method_dir, "plots", "truedata_diagnostics.png")
    print(f"real {S_real.shape}  gen {S_gen.shape}  asset {j}")
    plot_diagnostics_ma(S_real, S_gen, S_floor, args.method, args.seed, j, out,
                        floor_lbl=floor_lbl)


if __name__ == "__main__":
    main()
