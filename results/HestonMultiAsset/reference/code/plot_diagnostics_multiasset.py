#!/usr/bin/env python3
"""
plot_diagnostics_multiasset.py
──────────────────────────────
The 8-panel stylised-facts figure for the d = 8 multi-asset Heston run, laid
out panel-for-panel like the d = 1 benchmark's `metrics/plot_diagnostics.py`
so the two READMEs display the same figure in the same slot.

Why this is not just a call into metrics/plot_diagnostics.py
────────────────────────────────────────────────────────────
That script draws a third "theory" reference curve from `metrics/heston_theory`,
whose parameters are hard-wired to the d = 1 dataset (kappa=2.0, theta=0.04,
rho=-0.7, dt=1/250, T=128) and which loads the d = 1 path bank. The d = 8 data
comes from a different law entirely -- per-asset Ait-Sahalia & Kimmel (2007)
parameters, dt=1/252, T=252 -- so that curve would be a *fabricated* reference
drawn on top of unrelated data.

The third curve here is instead the **independent-draw floor**: a fresh draw
from the true multi-asset Heston SDE with the identical frozen parameters and a
brand-new RNG seed (GUIDELINE 5.4, seeds 1000..1004). It is not a closed-form
formula but it is honest, it is already computed, and it answers the question
the theory curve was there to answer: *how far apart do two independent draws of
the truth sit?* Any gap between Real and the generated paths that the floor curve
also shows is finite-sample noise, not a modelling failure.

Which asset
-----------
One asset, not eight. The instruction was "compute also for each asset the
stylised metrics and do an average for the metrics **no need to do all the
plots**" -- the *metrics* are per-asset-averaged (see metrics_per_asset.csv),
the *figure* is a single representative asset. Asset 0 is used and the title
says so, so nobody mistakes it for a pooled-over-assets view. Pooling would be
worse than useless: the eight assets have deliberately different volatilities
(sigma 0.0097-0.0165), so a pooled histogram is a mixture of eight different
marginals and its shape would reflect the mixing, not the model.

Usage
-----
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/reference/code/plot_diagnostics_multiasset.py
    ... --seed 0 --asset 0
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)                 # .../Deep-MKV-TS
MA = os.path.dirname(METHOD_DIR)                   # .../HestonMultiAsset
REPO = os.path.abspath(os.path.join(MA, "../.."))
DATA = os.path.join(REPO, "dataset", "HestonMultiAsset")
FLOOR = os.path.join(MA, "perfect_recovery")

TAG = "8192x252x8"

# benchmark-standard constants -- identical to metrics/plot_diagnostics.py
N_SHOW = 50        # individual paths drawn in panels 0 & 1
N_STAT = 5000      # paths used for the statistical panels 2-7
ACF_LAGS = 20
VOL_WIN = 5
RNG_SEED = 42
REAL_COL = "#2196F3"
GEN_COL = "#FF5722"
FLOOR_COL = "#000000"
FLOOR_LBL = "Independent draw (floor)"
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


def plot_diagnostics_ma(S_real, S_gen, S_floor, method, seed, asset, out_path):
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
        f"Multi-asset Heston diagnostics -- Real vs {method}  "
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
                lw=1.6, ls="--", label=FLOOR_LBL)
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
                lw=1.6, ls="--", label=FLOOR_LBL)
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
                lw=1.6, ls="--", marker="s", ms=2.5, label=FLOOR_LBL)
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
                lw=1.6, ls="--", marker="s", ms=2.5, label=FLOOR_LBL)
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
                lw=1.6, ls="--", label=FLOOR_LBL)
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
                label=FLOOR_LBL)
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
    # would silently label another method's figure over this method's paths.
    ap.add_argument("--method", default=os.path.basename(METHOD_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--asset", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    method_dir = os.path.join(MA, args.method)
    if not os.path.isdir(method_dir):
        raise SystemExit(f"ABORT: no such method directory {method_dir}")

    real_p = os.path.join(DATA, f"heston_ma_S_test_{TAG}.npy")
    gen_p = os.path.join(method_dir, "generated_paths", f"seed_{args.seed}",
                         f"generated_paths_{TAG}.npy")
    floor_p = os.path.join(FLOOR, "generated_paths", f"seed_{args.seed}",
                           f"generated_paths_{TAG}.npy")

    if not os.path.exists(gen_p):
        raise SystemExit(f"ABORT: {gen_p} missing -- run run_all_multiasset.py first.")

    j = args.asset
    S_real = np.load(real_p)[:, :, j]
    S_gen = np.load(gen_p)[:, :, j]
    if os.path.exists(floor_p):
        S_floor = np.load(floor_p)[:, :, j]
    else:
        S_floor = None
        print(f"  [warn] {floor_p} missing -- plotting without the floor curve")

    out = args.out or os.path.join(method_dir, "plots", "heston_diagnostics.png")
    print(f"real {S_real.shape}  gen {S_gen.shape}  asset {j}")
    plot_diagnostics_ma(S_real, S_gen, S_floor, args.method, args.seed, j, out)


if __name__ == "__main__":
    main()
