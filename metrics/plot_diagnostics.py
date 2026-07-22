#!/usr/bin/env python3
"""
Reusable 8-panel Heston diagnostics figure — consistent across ALL methods.

Usage (CLI):
    python metrics/plot_diagnostics.py --method SBTS    --dataset Heston --seed 0
    python metrics/plot_diagnostics.py --method TimeGAN --dataset Heston --seed 0

Usage (import):
    from plot_diagnostics import plot_diagnostics
    plot_diagnostics(S_real, S_gen, method="SBTS", seed=0, out_path="...")

Output:
    results/<Dataset>/<Method>/plots/heston_diagnostics.png

Design decisions
----------------
- N_SHOW = 50 individual paths in panels 0 & 1 (alpha=0.25, thin lines).
  Enough to show path diversity; thick line = cross-sectional mean.
- Statistics panels (2–7) use up to N_STAT=5000 paths, fixed RNG seed=42.
- Colors: Real = #2196F3 (blue), Generated = #FF5722 (orange-red).
  These are the benchmark-standard colors across ALL methods.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import heston_theory as ht  # third reference curve (theory / semi-theory / empirical)

# ── Constants (benchmark standard — do not change per method) ────────────────
BENCH    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_SHOW   = 50      # paths drawn in panels 0 & 1
N_STAT   = 5000    # paths used for statistical panels 2–7
ACF_LAGS = 20      # max lag for ACF panels
VOL_WIN  = 5       # rolling vol window (days)
RNG_SEED = 42      # fixed for reproducibility across methods
REAL_COL = "#2196F3"
GEN_COL  = "#FF5722"
THEORY_COL = ht.THEORY_COL   # black — third curve on the 6 statistical panels
DPI      = 120


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_real(dataset: str) -> np.ndarray:
    """Return real price paths shape (N, T).

    The 'real' reference is the held-out TEST set (seed 1) — the same reference
    used by every A/B metric — NOT the seed-0 set the generators trained on.
    """
    p = os.path.join(BENCH, "dataset", dataset, "heston_S_test_8192x128.npy")
    return np.load(p)


def _load_generated(method: str, dataset: str, seed: int) -> np.ndarray:
    """Return generated price paths shape (N, T)."""
    p = os.path.join(
        BENCH, "methods", method, "generated_paths",
        f"seed_{seed}", "generated_paths_8192x128.npy",
    )
    return np.load(p)


# ── Statistical helpers ───────────────────────────────────────────────────────

def _log_returns(S: np.ndarray) -> np.ndarray:
    """(N, T) price → (N, T-1) log-returns."""
    return np.log(S[:, 1:] / S[:, :-1])


def _acf_mean(X: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Cross-path mean ACF.  X: (N, T) → array of length max_lag."""
    X = X - X.mean(axis=1, keepdims=True)
    var = (X ** 2).mean(axis=1)
    out = []
    for lag in range(1, max_lag + 1):
        cov = (X[:, lag:] * X[:, :-lag]).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(var > 0, cov / var, 0.0)
        out.append(float(r.mean()))
    return np.array(out)


def _rolling_vol_flat(S: np.ndarray, window: int = 5) -> np.ndarray:
    """Return flattened rolling std of log-returns for all paths."""
    R = _log_returns(S)
    cols = [R[:, t - window + 1 : t + 1].std(axis=1)
            for t in range(window - 1, R.shape[1])]
    return np.stack(cols, axis=1).ravel()


# ── Main figure ───────────────────────────────────────────────────────────────

def plot_diagnostics(
    S_real:   np.ndarray,
    S_gen:    np.ndarray,
    method:   str,
    seed:     int,
    out_path: str,
) -> None:
    """
    Generate and save the standard 8-panel Heston diagnostics PNG.

    Parameters
    ----------
    S_real   : (N, T) real price paths (price space, S0 ≈ 100)
    S_gen    : (N, T) generated price paths (price space, S0 ≈ 100)
    method   : method name used in labels and title (e.g. "SBTS", "TimeGAN")
    seed     : seed index — appears in the figure title only
    out_path : full output path for the PNG
    """
    rng    = np.random.default_rng(RNG_SEED)
    N_stat = min(N_STAT, len(S_real), len(S_gen))

    # Fixed third reference curve for the 6 stat panels (theory / semi / empirical).
    # Method-independent → computed once and cached; missing file → build it now.
    try:
        TB = ht.compute_theory_bundle()
    except Exception as _ex:
        print(f"  [warn] theory bundle unavailable ({_ex}); plotting without 3rd curve")
        TB = None

    idx_show_r = rng.choice(len(S_real), N_SHOW,  replace=False)
    idx_show_g = rng.choice(len(S_gen),  N_SHOW,  replace=False)
    idx_stat_r = rng.choice(len(S_real), N_stat,  replace=False)
    idx_stat_g = rng.choice(len(S_gen),  N_stat,  replace=False)

    R_real   = _log_returns(S_real[idx_stat_r])   # (N_stat, T-1)
    R_gen    = _log_returns(S_gen[idx_stat_g])
    r_flat_r = R_real.ravel()
    r_flat_g = R_gen.ravel()
    t        = np.arange(S_real.shape[1])
    lags     = np.arange(1, ACF_LAGS + 1)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(
        f"Heston diagnostics — Real vs {method}  (seed {seed})",
        fontsize=14, fontweight="bold", y=0.998,
    )

    # ── [0, 0]  Real sample paths ─────────────────────────────────────────────
    ax = axes[0, 0]
    for i in idx_show_r:
        ax.plot(t, S_real[i], color=REAL_COL, alpha=0.25, lw=0.6)
    ax.plot(t, S_real[idx_show_r].mean(axis=0), color=REAL_COL, lw=2.0, label="mean")
    ax.set_title(f"Real paths  (n={N_SHOW} shown / 8192 total)")
    ax.set_xlabel("t (steps)"); ax.set_ylabel("S(t)")
    ax.legend(fontsize=8)

    # ── [0, 1]  Generated sample paths ───────────────────────────────────────
    ax = axes[0, 1]
    for i in idx_show_g:
        ax.plot(t, S_gen[i], color=GEN_COL, alpha=0.25, lw=0.6)
    ax.plot(t, S_gen[idx_show_g].mean(axis=0), color=GEN_COL, lw=2.0, label="mean")
    ax.set_title(f"{method} paths  (n={N_SHOW} shown / 8192 total)")
    ax.set_xlabel("t (steps)"); ax.set_ylabel("S(t)")
    ax.legend(fontsize=8)

    # ── [1, 0]  Log-return distribution ──────────────────────────────────────
    ax = axes[1, 0]
    lo, hi = np.quantile(np.concatenate([r_flat_r, r_flat_g]), [0.002, 0.998])
    bins = np.linspace(lo, hi, 80)
    ax.hist(r_flat_r, bins=bins, alpha=0.5, color=REAL_COL, density=True, label="Real")
    ax.hist(r_flat_g, bins=bins, alpha=0.5, color=GEN_COL,  density=True, label=method)
    if TB is not None:
        r_line = np.linspace(lo, hi, 400)
        ax.plot(r_line, ht.logreturn_marginal_pdf(r_line), color=THEORY_COL,
                lw=1.6, ls="--", label=ht.LABELS["log_ret_hist"])
    ax.set_title("Log-return distribution")
    ax.set_xlabel("log-return"); ax.set_ylabel("density")
    ax.legend(fontsize=8)

    # ── [1, 1]  QQ plot ───────────────────────────────────────────────────────
    ax = axes[1, 1]
    qq_grid = np.linspace(0.005, 0.995, 300)
    q_r = np.quantile(r_flat_r, qq_grid)
    q_g = np.quantile(r_flat_g, qq_grid)
    ax.scatter(q_r, q_g, s=5, color=GEN_COL, alpha=0.6, label=method)
    lo_ = min(q_r.min(), q_g.min()); hi_ = max(q_r.max(), q_g.max())
    ax.plot([lo_, hi_], [lo_, hi_], color="gray", ls=":", lw=1, label="y = x (perfect)")
    if TB is not None:
        # theoretical log-return quantiles vs the real (test) quantiles
        ax.plot(q_r, TB["qq_theory"], color=THEORY_COL, lw=1.6, ls="--",
                label=ht.LABELS["qq"])
    ax.set_title("QQ plot — generated vs real log-returns")
    ax.set_xlabel("Real quantiles"); ax.set_ylabel("Gen / theory quantiles")
    ax.legend(fontsize=8)

    # ── [2, 0]  ACF of |log-returns| ─────────────────────────────────────────
    ax = axes[2, 0]
    ax.plot(lags, _acf_mean(np.abs(R_real), ACF_LAGS),
            color=REAL_COL, marker="o", ms=3, label="Real")
    ax.plot(lags, _acf_mean(np.abs(R_gen),  ACF_LAGS),
            color=GEN_COL,  marker="o", ms=3, label=method)
    if TB is not None:
        ax.plot(TB["lags"], TB["acf_abs"], color=THEORY_COL, lw=1.6, ls="--",
                marker="s", ms=2.5, label=ht.LABELS["acf_abs"])
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_title("ACF of |log-returns|  (volatility clustering)")
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("ACF")
    ax.legend(fontsize=8)

    # ── [2, 1]  ACF of squared returns ───────────────────────────────────────
    ax = axes[2, 1]
    ax.plot(lags, _acf_mean(R_real ** 2, ACF_LAGS),
            color=REAL_COL, marker="o", ms=3, label="Real")
    ax.plot(lags, _acf_mean(R_gen  ** 2, ACF_LAGS),
            color=GEN_COL,  marker="o", ms=3, label=method)
    if TB is not None:
        ax.plot(TB["lags"], TB["acf_sq"], color=THEORY_COL, lw=1.6, ls="--",
                marker="s", ms=2.5, label=ht.LABELS["acf_sq"])
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_title("ACF of squared log-returns  (GARCH effect)")
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("ACF")
    ax.legend(fontsize=8)

    # ── [3, 0]  Rolling volatility histogram ─────────────────────────────────
    ax = axes[3, 0]
    vol_r = _rolling_vol_flat(S_real[idx_stat_r], VOL_WIN)
    vol_g = _rolling_vol_flat(S_gen[idx_stat_g],  VOL_WIN)
    hi_v  = np.quantile(np.concatenate([vol_r, vol_g]), 0.99)
    bins_v = np.linspace(0, hi_v, 60)
    ax.hist(vol_r, bins=bins_v, alpha=0.5, color=REAL_COL, density=True, label="Real")
    ax.hist(vol_g, bins=bins_v, alpha=0.5, color=GEN_COL,  density=True, label=method)
    if TB is not None:
        m = TB["rvol_grid"] <= hi_v
        ax.plot(TB["rvol_grid"][m], TB["rvol_dens"][m], color=THEORY_COL,
                lw=1.6, ls="--", label=ht.LABELS["rolling_vol"])
    ax.set_title(f"Rolling volatility  (window={VOL_WIN} days)")
    ax.set_xlabel("rolling std of log-returns"); ax.set_ylabel("density")
    ax.legend(fontsize=8)

    # ── [3, 1]  Tail survival (log-log) ──────────────────────────────────────
    ax = axes[3, 1]
    term_r  = S_real[:, -1]
    term_g  = S_gen[:,  -1]
    qq_tail = np.linspace(0.50, 0.999, 300)
    thresh  = np.quantile(term_r, qq_tail)
    surv_r  = np.array([(term_r > thr).mean() for thr in thresh])
    surv_g  = np.array([(term_g > thr).mean() for thr in thresh])
    mask    = (surv_r > 0) & (surv_g > 0)
    ax.plot(1 - qq_tail[mask], surv_r[mask], color=REAL_COL, label="Real")
    ax.plot(1 - qq_tail[mask], surv_g[mask], color=GEN_COL,  label=method)
    if TB is not None:
        mt = TB["tail_surv"] > 0
        ax.plot(TB["tail_oneminusq"][mt], TB["tail_surv"][mt], color=THEORY_COL,
                lw=1.6, ls="--", label=ht.LABELS["tail_surv"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Tail survival — terminal price S_T  (log-log)")
    ax.set_xlabel("1 − quantile"); ax.set_ylabel("P(S_T > threshold)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="8-panel Heston diagnostics figure — works for any method."
    )
    ap.add_argument("--method",  default="SBTS",   help="e.g. SBTS, TimeGAN")
    ap.add_argument("--dataset", default="Heston", help="Dataset name")
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument(
        "--out", default=None,
        help="Output PNG path (default: results/<Dataset>/<Method>/plots/heston_diagnostics.png)",
    )
    args = ap.parse_args()

    S_real = _load_real(args.dataset)
    S_gen  = _load_generated(args.method, args.dataset, args.seed)

    out = args.out or os.path.join(
        BENCH, "results", args.dataset, args.method,
        "plots", "heston_diagnostics.png",
    )
    print(f"Real : {S_real.shape}")
    print(f"Gen  : {S_gen.shape}")
    plot_diagnostics(S_real, S_gen, args.method, args.seed, out)


if __name__ == "__main__":
    main()
