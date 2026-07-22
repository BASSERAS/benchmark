#!/usr/bin/env python3
"""
heston_theory.py — reference "theory" curves for the 6 statistical panels of the
8-panel Heston diagnostics figure.

Each stat panel gets a THIRD curve besides Real (test set) and Generated. The
curve is one of three kinds, following an explicit honesty taxonomy:

    "theoretical"      — an exact closed form.
    "semi theoretical" — a semi-closed form (a 1-D numerical integral / a clean
                         analytic expression derived from the model parameters).
    "empirical"        — no clean closed form exists, so the curve is a
                         high-resolution Monte-Carlo estimate taken from the
                         100M-path large empirical dataset (dataset/Heston/
                         large_dataset/), generated with the *identical* Euler
                         scheme and dt as the benchmark train/test sets.

Panel → curve kind
------------------
[1,0] Log-return marginal   → semi theoretical  (CIR-stationary Gamma mixture of Normals)
[1,1] QQ (log-returns)      → semi theoretical  (quantiles of the same marginal)
[2,0] ACF of |log-returns|  → empirical         (autocorr of sqrt(v_t): no clean form)
[2,1] ACF of squared returns→ empirical         (per-path ACF estimator ≠ population
                                                 CIR autocorr; see acf_sq_returns_theory)
[3,0] Rolling-vol histogram → empirical         (distribution of a 5-day rolling std)
[3,1] Tail survival of S_T  → empirical          (terminal-price survival, Euler scheme)

All model constants MUST match dataset/Heston/generate_heston.py and
generate_heston_large.py exactly.
"""

import os
from math import gamma as _gamma_fn, pi, sqrt

import numpy as np

# ── Heston constants (identical to the generators) ───────────────────────────
MU = 0.05
KAPPA = 2.0
THETA = 0.04
XI = 0.3
RHO = -0.7
S0 = 100.0
V0 = 0.04
DT = 1.0 / 250.0

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
_LARGE_DIR = os.path.join(_BENCH, "dataset", "Heston", "large_dataset")


# ── CIR stationary law of the variance process ───────────────────────────────
def _cir_stationary_gamma():
    """Stationary law of the CIR variance v: Gamma(shape a, scale b).

    v_inf ~ Gamma(a = 2 kappa theta / xi^2, scale b = xi^2 / (2 kappa)).
    Mean a*b = theta, variance a*b^2 = theta xi^2 / (2 kappa).
    """
    a = 2.0 * KAPPA * THETA / XI ** 2
    b = XI ** 2 / (2.0 * KAPPA)
    return a, b


def _gamma_pdf(v, a, b):
    """Gamma(a, scale=b) density, evaluated elementwise (no scipy dependency)."""
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros_like(v)
    pos = v > 0
    out[pos] = (
        v[pos] ** (a - 1.0)
        * np.exp(-v[pos] / b)
        / (b ** a * _gamma_fn(a))
    )
    return out


def _v_quadrature(n_v=600, tail=1e-5):
    """Grid of variance values + Gamma weights (already times dv) for quadrature."""
    a, b = _cir_stationary_gamma()
    # support: from a small positive value to mean + many std (Gamma is right-skewed)
    mean = a * b
    std = sqrt(a) * b
    v_hi = mean + 12.0 * std
    v = np.linspace(1e-7, v_hi, n_v)
    dv = v[1] - v[0]
    w = _gamma_pdf(v, a, b) * dv
    w = w / w.sum()  # renormalise the truncated/discretised Gamma
    return v, w


# ── [1,0] / [1,1] Log-return marginal — semi theoretical ─────────────────────
def logreturn_marginal_pdf(r_grid, n_v=600):
    """Semi-theoretical marginal density of one-step log-returns.

    Conditional on the stationary variance v, one Euler log-return is
    approximately Normal( (mu - v/2) dt , v dt ). Marginalising v over its CIR
    stationary Gamma law gives a Gamma-mixture-of-Normals — a 1-D integral we
    evaluate by quadrature.
    """
    r_grid = np.asarray(r_grid, dtype=np.float64)
    v, w = _v_quadrature(n_v)
    m = (MU - 0.5 * v) * DT           # (n_v,)
    s2 = v * DT                       # (n_v,)
    R = r_grid[:, None]               # (G, 1)
    norm = np.exp(-0.5 * (R - m[None, :]) ** 2 / s2[None, :]) / np.sqrt(2.0 * pi * s2[None, :])
    return (norm * w[None, :]).sum(axis=1)


def logreturn_marginal_quantiles(q_levels, n_grid=20001, r_span=0.12, n_v=600):
    """Quantiles of the semi-theoretical log-return marginal at levels q_levels."""
    q_levels = np.asarray(q_levels, dtype=np.float64)
    r = np.linspace(-r_span, r_span, n_grid)
    pdf = logreturn_marginal_pdf(r, n_v=n_v)
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    return np.interp(q_levels, cdf, r)


# ── [2,1] ACF of squared log-returns — population form (NOT used on the panel) ─
def acf_sq_returns_theory(lags):
    """Population (cross-time, stationary) ACF of squared log-returns.

    Cov(r_t^2, r_{t+h}^2) = dt^2 Cov(v_t, v_{t+h}) = dt^2 Var(v) exp(-kappa dt h)
    Var(r^2)             = dt^2 (3 E[v^2] - theta^2)
    so rho(h) = [Var(v) / (3 E[v^2] - theta^2)] exp(-kappa dt h).

    NOTE — this closed form is kept for reference only. The diagnostics panel
    estimates ACF *per path* (each 128-step path is de-meaned and normalised by
    its own variance, then averaged across paths). Over one short path the
    variance cannot decorrelate, so the per-path estimator is dominated by the
    conditional-normal noise of the returns and sits ~3x below this population
    curve with a faster decay (validated against the 100M set). The panel
    therefore uses the *empirical* per-path curve (acf_sq_returns_empirical),
    computed with the identical estimator, and is labelled "empirical".
    """
    a, b = _cir_stationary_gamma()
    var_v = a * b ** 2
    Ev2 = var_v + THETA ** 2
    prefactor = var_v / (3.0 * Ev2 - THETA ** 2)
    lags = np.asarray(lags, dtype=np.float64)
    return prefactor * np.exp(-KAPPA * DT * lags)


# ── Empirical high-res curves from the 100M large dataset ────────────────────
_S_CACHE = {"n": 0, "arr": None}


def _load_large_S(n_paths):
    """Memory-map the first large shard and return the first n_paths rows (copy).

    A tiny module-level cache avoids re-reading the shard when several empirical
    panels ask for the same (or a smaller) number of paths within one figure.
    """
    if _S_CACHE["arr"] is not None and _S_CACHE["n"] >= n_paths:
        return _S_CACHE["arr"][:n_paths]
    p = os.path.join(_LARGE_DIR, "large_S_shard0.npy")
    mm = np.load(p, mmap_mode="r")
    n = min(n_paths, mm.shape[0])
    arr = np.asarray(mm[:n], dtype=np.float64)
    _S_CACHE["n"] = n
    _S_CACHE["arr"] = arr
    return arr


def _log_returns(S):
    return np.log(S[:, 1:] / S[:, :-1])


def _acf_mean(X, max_lag=20):
    """Cross-path mean ACF (matches plot_diagnostics._acf_mean)."""
    X = X - X.mean(axis=1, keepdims=True)
    var = (X ** 2).mean(axis=1)
    out = []
    for lag in range(1, max_lag + 1):
        cov = (X[:, lag:] * X[:, :-lag]).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(var > 0, cov / var, 0.0)
        out.append(float(r.mean()))
    return np.array(out)


def acf_abs_returns_empirical(max_lag=20, n_paths=1_000_000):
    """[2,0] Empirical ACF of |log-returns| from the large dataset (per-path estimator)."""
    S = _load_large_S(n_paths)
    R = _log_returns(S)
    return _acf_mean(np.abs(R), max_lag)


def acf_sq_returns_empirical(max_lag=20, n_paths=1_000_000):
    """[2,1] Empirical ACF of squared log-returns from the large dataset.

    Uses the identical per-path estimator as the panel, so the curve is directly
    comparable to the Real/Generated ACFs (unlike the population closed form in
    acf_sq_returns_theory, which does not describe the per-path estimator)."""
    S = _load_large_S(n_paths)
    R = _log_returns(S)
    return _acf_mean(R ** 2, max_lag)


def rolling_vol_samples_empirical(window=5, n_paths=400_000):
    """[3,0] Empirical pool of 5-day rolling-vol values from the large dataset."""
    S = _load_large_S(n_paths)
    R = _log_returns(S)
    cols = [R[:, t - window + 1: t + 1].std(axis=1)
            for t in range(window - 1, R.shape[1])]
    return np.stack(cols, axis=1).ravel()


def terminal_survival_empirical(thresholds, n_paths=25_000_000):
    """[3,1] Empirical survival P(S_T > thr) of terminal price from the large dataset.

    Uses only the terminal column, so a full 25M-path shard is cheap (~200 MB).
    """
    p = os.path.join(_LARGE_DIR, "large_S_shard0.npy")
    mm = np.load(p, mmap_mode="r")
    n = min(n_paths, mm.shape[0])
    term = np.asarray(mm[:n, -1], dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    return np.array([(term > thr).mean() for thr in thresholds])


# ── Cached bundle of the fixed reference curves ──────────────────────────────
# The third curve on every stat panel depends ONLY on (a) the held-out test set
# and (b) the 100M large dataset — never on the method being plotted. So the
# expensive empirical curves (which read ~1M paths from disk) are computed once,
# cached to an .npz, and reused for all 9 methods.
_BUNDLE_PATH = os.path.join(_LARGE_DIR, "theory_curves_bundle.npz")

# grids shared with plot_diagnostics (must stay in sync)
QQ_GRID = np.linspace(0.005, 0.995, 300)
ACF_LAGS = 20
TAIL_QQ = np.linspace(0.50, 0.999, 300)
VOL_WIN = 5


def _load_test_S():
    p = os.path.join(_BENCH, "dataset", "Heston", "heston_S_test_8192x128.npy")
    return np.load(p).astype(np.float64)


def compute_theory_bundle(force=False):
    """Compute (or load cached) the fixed reference curves for all 6 stat panels.

    Returns a dict of numpy arrays:
        qq_grid, qq_theory            — [1,1] semi-theoretical log-ret quantiles
        lags, acf_abs, acf_sq         — [2,0]/[2,1] empirical per-path ACFs
        rvol_grid, rvol_dens          — [3,0] empirical rolling-vol density (line)
        tail_oneminusq, tail_surv     — [3,1] empirical terminal-price survival
    """
    if (not force) and os.path.exists(_BUNDLE_PATH):
        d = np.load(_BUNDLE_PATH)
        return {k: d[k] for k in d.files}

    # [1,1] QQ — semi theoretical
    qq_theory = logreturn_marginal_quantiles(QQ_GRID)

    # [2,0]/[2,1] ACF — empirical (per-path estimator, 1M paths)
    lags = np.arange(1, ACF_LAGS + 1)
    S1 = _load_large_S(1_000_000)
    R1 = _log_returns(S1)
    acf_abs = _acf_mean(np.abs(R1), ACF_LAGS)
    acf_sq = _acf_mean(R1 ** 2, ACF_LAGS)

    # [3,0] rolling-vol — empirical density line (400k paths)
    rvol = rolling_vol_samples_empirical(window=VOL_WIN, n_paths=400_000)
    rvhi = np.quantile(rvol, 0.999)
    rvol_grid = np.linspace(0.0, rvhi, 400)
    # smooth density via fine histogram midpoints
    hist, edges = np.histogram(rvol, bins=600, range=(0.0, rvhi), density=True)
    mids = 0.5 * (edges[1:] + edges[:-1])
    rvol_dens = np.interp(rvol_grid, mids, hist)

    # [3,1] tail survival — empirical, thresholds from the (fixed) test set
    term_test = _load_test_S()[:, -1]
    thresh = np.quantile(term_test, TAIL_QQ)
    tail_surv = terminal_survival_empirical(thresh, n_paths=25_000_000)
    tail_oneminusq = 1.0 - TAIL_QQ

    bundle = dict(
        qq_grid=QQ_GRID, qq_theory=qq_theory,
        lags=lags, acf_abs=acf_abs, acf_sq=acf_sq,
        rvol_grid=rvol_grid, rvol_dens=rvol_dens,
        tail_oneminusq=tail_oneminusq, tail_surv=tail_surv,
    )
    np.savez(_BUNDLE_PATH, **bundle)
    return bundle


# ── Curve-kind labels (legend text) ──────────────────────────────────────────
LABELS = {
    "log_ret_hist": "semi theoretical",
    "qq":           "semi theoretical",
    "acf_abs":      "empirical",
    "acf_sq":       "empirical",
    "rolling_vol":  "empirical",
    "tail_surv":    "empirical",
}
THEORY_COL = "#111111"  # black, dashed — third curve on every stat panel


if __name__ == "__main__":
    # Self-check: print the semi-theoretical closed forms + validate vs 100M set.
    print("CIR stationary Gamma:", _cir_stationary_gamma())
    print("ACF(r^2) theory lags 1,5,20:",
          acf_sq_returns_theory([1, 5, 20]))
    lg = _load_large_S(200_000)
    R = _log_returns(lg).ravel()
    qs = [0.01, 0.5, 0.99]
    print("log-ret quantiles  theory:", logreturn_marginal_quantiles(qs))
    print("log-ret quantiles  large :", np.quantile(R, qs))
