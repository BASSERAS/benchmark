"""
stylized_metrics.py — B1–B14: Quantitative metrics extracted from the 8 diagnostic plots.

Each metric corresponds directly to one (or two paired) diagnostic panels so that
every number in this table can be visually verified against the PNG diagnostic figure.

References
----------
Cont R. (2001) "Empirical properties of asset returns: stylized facts and
  statistical issues." Quantitative Finance 1(2), 223–236.
Bollerslev T. (1986) "Generalized Autoregressive Conditional Heteroskedasticity."
  Journal of Econometrics 31(3), 307–327.
Ding Z., Granger C.W.J., Engle R.F. (1993) "A long memory property of stock
  market returns and a new model." Journal of Empirical Finance 1(1), 83–106.
Hill B.M. (1975) "A simple general approach to inference about the tail of a
  distribution." Annals of Statistics 3(5), 1163–1174.
Massey F.J. (1951) "The Kolmogorov-Smirnov test for goodness of fit."
  Journal of the American Statistical Association 46(253), 68–78.

Metric map
----------
Plots 1+2  (sample paths)          B1  Mean Path RMSE
                                   B2  Cross-Sectional Vol RMSE
Plot  3    (log-return histogram)  B3  KS Statistic (log-returns)
                                   B4  Skewness Error
Plot  4    (QQ plot)               B5  QQ RMSE
                                   B6  Tail QQ Error (5th/95th extremes)
Plot  5    (ACF |r|)               B7  ACF Lag-1 Error (|r|)
                                   B8  ARCH Persistence Error
Plot  6    (ACF r²)                B9  ACF Lag-1 Error (r²)
                                   B10 GARCH Persistence Error
Plot  7    (rolling vol histogram)  B11 Rolling Vol KS Statistic
                                   B12 Vol-of-Vol Error
Plot  8    (tail survival log-log) B13 Terminal Price KS Statistic
                                   B14 Tail Index Error (Hill estimator)
"""

import numpy as np
from scipy.stats import ks_2samp, skew
from typing import Dict


# ── helpers ──────────────────────────────────────────────────────────────────

def _log_returns(S: np.ndarray) -> np.ndarray:
    """(N, T) price paths → (N, T-1) log-returns  r_{i,t} = log(S_{i,t+1}/S_{i,t})."""
    return np.log(np.maximum(S[:, 1:], 1e-10) / np.maximum(S[:, :-1], 1e-10))


def _rolling_vol(S: np.ndarray, window: int = 5) -> np.ndarray:
    """(N, T) → (N, T-window) rolling std of log-returns over a sliding window."""
    R = _log_returns(S)                # (N, T-1)
    T1 = R.shape[1]
    cols = [R[:, t - window + 1: t + 1].std(axis=1)
            for t in range(window - 1, T1)]
    return np.stack(cols, axis=1)     # (N, T-1-(window-1))


def _acf_mean(X: np.ndarray, lag: int) -> float:
    """Mean cross-path ACF at a given lag.  X : (N, T)."""
    X_c = X - X.mean(axis=1, keepdims=True)
    var = (X_c ** 2).mean(axis=1)
    cov = (X_c[:, lag:] * X_c[:, :-lag]).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(var > 0, cov / var, 0.0)
    return float(r.mean())


def _hill_estimator(x: np.ndarray, k_frac: float = 0.10) -> float:
    """Hill (1975) tail index α̂.

    Uses the top k = max(10, int(k_frac * n)) order statistics.
    α̂ = 1 / mean_i(log X_{(n-i+1)} − log X_{(n-k)})   for i=1..k

    Returns α̂ (higher = lighter tail; Pareto tail index).
    Returns NaN if insufficient data.
    """
    x_pos = np.sort(x[x > 0])
    n = len(x_pos)
    k = max(10, int(k_frac * n))
    if k >= n:
        return float("nan")
    threshold = x_pos[-(k + 1)]
    exceedances = x_pos[-k:]
    diffs = np.log(exceedances) - np.log(threshold)
    if diffs.mean() <= 0:
        return float("nan")
    return float(1.0 / diffs.mean())


# ── main function ─────────────────────────────────────────────────────────────

def compute_stylized_metrics(
    S_real: np.ndarray,
    S_gen: np.ndarray,
    acf_lags: int = 20,
    vol_window: int = 5,
    qq_grid: int = 300,
) -> Dict[str, float]:
    """Compute B1–B14 stylized metrics.

    Parameters
    ----------
    S_real : (N, T)  real price paths (price space, S0 ≈ 100)
    S_gen  : (N, T)  generated price paths
    acf_lags   : max lag for ACF panels (default 20, matches plot)
    vol_window : rolling vol window in days (default 5, matches plot)
    qq_grid    : number of quantile points for QQ RMSE (default 300, matches plot)

    Returns
    -------
    dict with keys B1..B14 (floats, all ↓ lower is better)
    """
    N_r, T = S_real.shape
    N_g    = S_gen.shape[0]

    # log-returns
    R_real = _log_returns(S_real)   # (N_r, T-1)
    R_gen  = _log_returns(S_gen)    # (N_g, T-1)

    r_flat_r = R_real.ravel()
    r_flat_g = R_gen.ravel()

    out: Dict[str, float] = {}

    # ── B1  Mean Path RMSE (Plots 1+2) ───────────────────────────────────────
    # RMSE between cross-sectional mean paths over time
    # Formula: sqrt(1/T * Σ_t (E_real[S(t)] - E_gen[S(t)])²)
    mu_r = S_real.mean(axis=0)   # (T,)
    mu_g = S_gen.mean(axis=0)
    out["B1_mean_path_rmse"] = float(np.sqrt(np.mean((mu_r - mu_g) ** 2)))

    # ── B2  Cross-Sectional Vol RMSE (Plots 1+2) ─────────────────────────────
    # RMSE between cross-sectional standard deviation at each t
    # Formula: sqrt(1/T * Σ_t (std_real(t) - std_gen(t))²)
    sig_r = S_real.std(axis=0)
    sig_g = S_gen.std(axis=0)
    out["B2_vol_path_rmse"] = float(np.sqrt(np.mean((sig_r - sig_g) ** 2)))

    # ── B3  KS Statistic — log-returns (Plot 3) ──────────────────────────────
    # Two-sample Kolmogorov-Smirnov: D = max_x |F_real(x) - F_gen(x)|
    # Ref: Massey (1951). Perfect: 0. ↓
    ks_stat, _ = ks_2samp(r_flat_r, r_flat_g)
    out["B3_ks_logreturns"] = float(ks_stat)

    # ── B4  Skewness Error (Plot 3) ──────────────────────────────────────────
    # |skew_real - skew_gen|, skew = E[(r-μ)³]/σ³ (Fisher, unbiased)
    # Heston generates negative skew via ρ<0 (leverage effect). Ref: Cont 2001.
    sk_r = float(skew(r_flat_r, bias=False))
    sk_g = float(skew(r_flat_g, bias=False))
    out["B4_skewness_error"] = float(abs(sk_r - sk_g))

    # ── B5  QQ RMSE (Plot 4) ─────────────────────────────────────────────────
    # sqrt(1/G * Σ_g (Q_real(g) - Q_gen(g))²) over uniform quantile grid
    # Ref: Wilk & Gnanadesikan (1968). Perfect: 0. ↓
    qq_grid_pts = np.linspace(0.005, 0.995, qq_grid)
    q_r = np.quantile(r_flat_r, qq_grid_pts)
    q_g = np.quantile(r_flat_g, qq_grid_pts)
    out["B5_qq_rmse"] = float(np.sqrt(np.mean((q_r - q_g) ** 2)))

    # ── B6  Tail QQ Error (Plot 4) ───────────────────────────────────────────
    # Mean absolute deviation at extreme quantile levels [0.01..0.05, 0.95..0.99]
    # Captures how well tails match; complements B5 which is dominated by the bulk.
    tail_levels = np.concatenate([np.linspace(0.01, 0.05, 5), np.linspace(0.95, 0.99, 5)])
    q_r_tail = np.quantile(r_flat_r, tail_levels)
    q_g_tail = np.quantile(r_flat_g, tail_levels)
    out["B6_tail_qq_error"] = float(np.mean(np.abs(q_r_tail - q_g_tail)))

    # ── B7  ACF Lag-1 Error |r| (Plot 5) ─────────────────────────────────────
    # |ACF_real(|r|, lag=1) - ACF_gen(|r|, lag=1)|
    # Lag-1 captures the dominant ARCH signal. Ref: Ding, Granger & Engle (1993).
    acf1_abs_r = _acf_mean(np.abs(R_real), lag=1)
    acf1_abs_g = _acf_mean(np.abs(R_gen),  lag=1)
    out["B7_acf_lag1_abs"] = float(abs(acf1_abs_r - acf1_abs_g))

    # ── B8  ARCH Persistence Error (Plot 5) ──────────────────────────────────
    # |mean_k(ACF_real(|r|,k)) - mean_k(ACF_gen(|r|,k))| for k=1..acf_lags
    # Integrated autocorrelation of absolute returns measures ARCH persistence.
    acf_abs_r = np.array([_acf_mean(np.abs(R_real), k) for k in range(1, acf_lags + 1)])
    acf_abs_g = np.array([_acf_mean(np.abs(R_gen),  k) for k in range(1, acf_lags + 1)])
    out["B8_arch_persistence_error"] = float(abs(acf_abs_r.mean() - acf_abs_g.mean()))

    # ── B9  ACF Lag-1 Error r² (Plot 6) ──────────────────────────────────────
    # |ACF_real(r², lag=1) - ACF_gen(r², lag=1)|
    # Squared returns ACF captures GARCH effect. Ref: Bollerslev (1986).
    acf1_sq_r = _acf_mean(R_real ** 2, lag=1)
    acf1_sq_g = _acf_mean(R_gen  ** 2, lag=1)
    out["B9_acf_lag1_sq"] = float(abs(acf1_sq_r - acf1_sq_g))

    # ── B10  GARCH Persistence Error (Plot 6) ────────────────────────────────
    # |mean_k(ACF_real(r²,k)) - mean_k(ACF_gen(r²,k))| for k=1..acf_lags
    acf_sq_r = np.array([_acf_mean(R_real ** 2, k) for k in range(1, acf_lags + 1)])
    acf_sq_g = np.array([_acf_mean(R_gen  ** 2, k) for k in range(1, acf_lags + 1)])
    out["B10_garch_persistence_error"] = float(abs(acf_sq_r.mean() - acf_sq_g.mean()))

    # ── B11  Rolling Vol KS Statistic (Plot 7) ───────────────────────────────
    # Two-sample KS on rolling volatility distributions (window=vol_window)
    # Ref: Cont (2001) stylized fact: "volatility clusters in time"
    rv_r = _rolling_vol(S_real, vol_window).ravel()
    rv_g = _rolling_vol(S_gen,  vol_window).ravel()
    ks_vol, _ = ks_2samp(rv_r, rv_g)
    out["B11_rolling_vol_ks"] = float(ks_vol)

    # ── B12  Vol-of-Vol Error (Plot 7) ───────────────────────────────────────
    # |std(roll_vol_real) - std(roll_vol_gen)| normalized by real std
    # Measures whether the dispersion of the vol distribution is reproduced.
    # Ref: Hull & White (1987) — vol-of-vol drives option smile.
    std_rv_r = float(rv_r.std())
    std_rv_g = float(rv_g.std())
    out["B12_vol_of_vol_error"] = float(abs(std_rv_r - std_rv_g))

    # ── B13  Terminal Price KS Statistic (Plot 8) ─────────────────────────────
    # Two-sample KS on terminal price S_T distribution
    # Tests whether the generator reproduces the terminal marginal (log-normal for GBM).
    term_r = S_real[:, -1]
    term_g = S_gen[:,  -1]
    ks_term, _ = ks_2samp(term_r, term_g)
    out["B13_terminal_ks"] = float(ks_term)

    # ── B14  Tail Index Error — Hill estimator (Plot 8) ───────────────────────
    # |α̂_real - α̂_gen| where α̂ = Hill(1975) estimator on terminal prices
    # Uses top 10% of terminal prices. α > 4 → finite 4th moment; α > 2 → finite variance.
    # Ref: Hill (1975); Mandelbrot (1963) — empirical tail indices for stocks ≈ 3–5.
    alpha_r = _hill_estimator(term_r, k_frac=0.10)
    alpha_g = _hill_estimator(term_g, k_frac=0.10)
    if np.isfinite(alpha_r) and np.isfinite(alpha_g):
        out["B14_tail_index_error"] = float(abs(alpha_r - alpha_g))
    else:
        out["B14_tail_index_error"] = float("nan")

    return out


# ── public ────────────────────────────────────────────────────────────────────

__all__ = ["compute_stylized_metrics"]
