"""
Path Shadowing Monte-Carlo (PS-MC) for time-series generation evaluation.

Reference: Morel, Mallat, Bouchaud (2023) — arXiv:2308.01486
           "Path Shadowing Monte Carlo"

Algorithm
---------
1. Embed every path prefix as a vector of multi-scale log-returns (eq. 13).
2. For each real query path, retrieve the K nearest generated paths by L2
   distance in that embedding space.
3. Use the futures of those K paths as a forecast ensemble (price-anchored).
4. Evaluate with CRPS, MAE, RMSE at multiple forecast horizons.

Embedding — eq. (13) of arXiv:2308.01486
-----------------------------------------
    h_{α,β}(x)[m] = ( log S_t − log S_{t−ℓ_m} ) / ℓ_m^β,   ℓ_m = ⌊α^m⌋

Optimal parameters calibrated on S&P (Table I of the paper):
    α = 1.15  →  lags are geometrically spaced (coarser for the distant past)
    β = 0.9   →  power-law down-weighting of distant increments

With prefix_len=64 (63 available returns) this yields M=22 unique lags
{1,2,3,4,5,6,7,8,9,10,12,14,16,18,21,24,28,32,37,43,50,57}
and an embedding of dimension 22.  (Paper achieves dim=34 with its 126-day
prefix; we are limited by the shorter Heston path length of 128 steps.)

Gaussian bandwidth — per-query normalisation
--------------------------------------------
    η_i = η̃ · ‖h(x̃_past_i)‖,    η̃ = 0.075  (calibrated in the paper)

This normalises the L2 distance by the query's own embedding norm, making
the kernel scale-invariant across different market regimes.

KNN selection — NOT combinatorial
----------------------------------
sklearn NearestNeighbors computes L2(query_emb, pool_emb) for all pool paths
and returns the K with the smallest distance.  Cost: O(N × M) per batch.
No subset enumeration; no combinatorial search.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


# ── Embedding ─────────────────────────────────────────────────────────────────

def _get_lags(prefix_len, alpha=1.15):
    """Return sorted list of unique lags ℓ = ⌊α^m⌋ ≤ prefix_len − 1."""
    lags, m = [], 1
    while True:
        lag = int(alpha ** m)
        if lag > prefix_len - 1:
            break
        if not lags or lag != lags[-1]:
            lags.append(lag)
        m += 1
    return lags


def _multiscale_embedding(X, prefix_len, alpha=1.15, beta=0.9):
    """
    Multi-scale log-return embedding — eq. (13) of arXiv:2308.01486.

    For each lag ℓ_m = ⌊α^m⌋:
        component_m = ( log S_{t} − log S_{t−ℓ_m} ) / ℓ_m^β

    where t = prefix_len − 1 is the last observed prefix step.

    Unlike flat log-returns (our previous approach), this embedding:
    - emphasises recent dynamics (divided by ℓ^0.9 ≈ small for small ℓ)
    - is lower-dimensional (22D vs 63D) → less curse of dimensionality
    - uses logarithmic lag spacing → scale-invariant representation

    Parameters
    ----------
    X          : (N, T) price paths
    prefix_len : int — steps available for embedding
    alpha      : float — geometric spacing (paper: 1.15)
    beta       : float — power-law decay (paper: 0.9)

    Returns
    -------
    emb : (N, M) float32
    """
    lags = _get_lags(prefix_len, alpha)
    log_p = np.log(np.clip(
        X[:, :prefix_len].astype(np.float64), 1e-8, None
    ))                                     # (N, prefix_len)
    t = prefix_len - 1                     # index of last prefix step
    components = [
        (log_p[:, t] - log_p[:, t - ell]) / (ell ** beta)
        for ell in lags
    ]
    return np.stack(components, axis=1).astype(np.float32)   # (N, M)


# ── Core retrieval ────────────────────────────────────────────────────────────

def ps_mc_retrieve(X_real, X_fake, prefix_len=64, K=77, alpha=1.15, beta=0.9):
    """
    Retrieve K nearest fake paths for every real query path.

    Distance is computed in the multi-scale embedding space (eq. 13).

    Parameters
    ----------
    X_real     : (N, T) real paths
    X_fake     : (M, T) generated paths (search pool)
    prefix_len : int — steps used for embedding / distance
    K          : int — number of nearest neighbours
    alpha, beta: embedding hyperparameters (paper: 1.15, 0.9)

    Returns
    -------
    ensemble        : (N, K, T-prefix_len) — price-anchored futures of K neighbours
    distances       : (N, K) — L2 distances in embedding space
    indices         : (N, K) — pool indices of the K neighbours
    real_emb_norms  : (N,)   — ‖h(x̃_past_i)‖, needed for per-query η
    """
    real_emb = _multiscale_embedding(X_real, prefix_len, alpha, beta)  # (N, M)
    fake_emb = _multiscale_embedding(X_fake, prefix_len, alpha, beta)  # (P, M)
    fake_fut = X_fake[:, prefix_len:]                                   # (P, H)

    nn = NearestNeighbors(n_neighbors=K, metric="euclidean",
                          algorithm="auto", n_jobs=16)
    nn.fit(fake_emb)
    distances, indices = nn.kneighbors(real_emb)          # both (N, K)

    # Raw futures of the K neighbours
    ensemble_raw = fake_fut[indices]                       # (N, K, H)

    # Price anchoring: scale each fake future so it starts at the real path's
    # last prefix price.  Without this, a fake path at level 98 → 110 would be
    # compared against a real path that ended at 103 — a spurious level offset.
    S_real_last = np.clip(X_real[:, prefix_len - 1], 1e-8, None)         # (N,)
    S_fake_last = np.clip(X_fake[indices, prefix_len - 1], 1e-8, None)   # (N, K)
    scale       = S_real_last[:, None] / S_fake_last                      # (N, K)
    ensemble    = ensemble_raw * scale[:, :, None]                        # (N, K, H)

    # Embedding norms of real query paths (for per-query Gaussian bandwidth)
    real_emb_norms = np.linalg.norm(
        real_emb.astype(np.float64), axis=1
    )                                                                      # (N,)

    return ensemble, distances, indices, real_emb_norms


# ── Weight computation ────────────────────────────────────────────────────────

def uniform_weights(N, K, dtype=np.float32):
    """Return uniform 1/K weights of shape (N, K)."""
    return np.full((N, K), 1.0 / K, dtype=dtype)


def gaussian_weights(distances, real_emb_norms=None, eta_tilde=0.075):
    """
    Gaussian weights: w_k ∝ exp(−d_k² / (2η_i²)).

    Per-query bandwidth following the paper (Section IV-B):
        η_i = η̃ · ‖h(x̃_past_i)‖

    This normalises the distance by the query path's embedding norm, making
    the kernel scale-invariant (η̃ = 0.075 calibrated on S&P in the paper).

    Parameters
    ----------
    distances      : (N, K) — L2 distances from retrieve step
    real_emb_norms : (N,) or None — ‖h(x̃_past_i)‖ per query
                     If None, falls back to global median bandwidth.
    eta_tilde      : float — normalised bandwidth (paper: 0.075)

    Returns
    -------
    weights  : (N, K) float32 — normalised Gaussian weights
    eta_used : float — representative η (mean of per-query ηs, or global)
    """
    if real_emb_norms is not None:
        # Per-query η: shape (N,)
        eta_q   = (eta_tilde * real_emb_norms).astype(np.float32)   # (N,)
        eta_used = float(np.mean(eta_q))
        log_w   = -(distances.astype(np.float32) ** 2) / (
            2.0 * (eta_q[:, None] ** 2) + 1e-30
        )
    else:
        # Fallback: global median bandwidth
        eta      = float(np.median(distances))
        eta_used = eta
        log_w    = -(distances.astype(np.float32) ** 2) / (
            2.0 * float(eta) ** 2 + 1e-30
        )

    log_w -= log_w.max(axis=1, keepdims=True)
    w      = np.exp(log_w)
    w     /= w.sum(axis=1, keepdims=True)
    return w.astype(np.float32), eta_used


# ── Forecast ──────────────────────────────────────────────────────────────────

def ensemble_forecast(ensemble, weights):
    """
    Weighted ensemble mean.

    Parameters
    ----------
    ensemble : (N, K, H)
    weights  : (N, K)

    Returns
    -------
    forecast : (N, H)
    """
    return np.einsum("nk,nkh->nh",
                     weights.astype(np.float32),
                     ensemble.astype(np.float32)).astype(np.float64)


# ── CRPS (vectorised over N, not H) ──────────────────────────────────────────

def crps(ensemble, y_real, weights=None, batch_n=512):
    """
    Energy-score CRPS: E_w|Z−y| − ½ E_w|Z−Z'|.

    Vectorised over N-batches (not H-steps) for efficiency.
    Uses float32 internally; returns float64.

    Parameters
    ----------
    ensemble : (N, K, H) float32 or float64
    y_real   : (N, H)    float32 or float64
    weights  : (N, K) or None (uniform 1/K)
    batch_n  : int — paths processed per batch

    Returns
    -------
    crps_vals : (N, H) float64 — pointwise CRPS (mean = scalar CRPS)
    """
    N, K, H = ensemble.shape
    ens = ensemble.astype(np.float32)
    y   = y_real.astype(np.float32)

    if weights is None:
        w = np.full((N, K), 1.0 / K, dtype=np.float32)
    else:
        w = weights.astype(np.float32)

    # Term 1: Σ_k w_k |z_k − y|          shape (N, H)
    term1 = np.einsum("nk,nkh->nh", w, np.abs(ens - y[:, None, :]))

    # Term 2: ½ Σ_{j,k} w_j w_k |z_j − z_k|  (N-batch loop to cap RAM)
    term2 = np.zeros((N, H), dtype=np.float32)
    for n0 in range(0, N, batch_n):
        n1   = min(n0 + batch_n, N)
        e_b  = ens[n0:n1]                                  # (B, K, H)
        w_b  = w[n0:n1]                                    # (B, K)
        ww_b = w_b[:, :, None] * w_b[:, None, :]          # (B, K, K)
        diff = np.abs(e_b[:, :, None, :] - e_b[:, None, :, :])  # (B,K,K,H)
        term2[n0:n1] = 0.5 * np.einsum("bjk,bjkh->bh", ww_b, diff)

    return (term1 - term2).astype(np.float64)


# ── Horizon-level evaluation ──────────────────────────────────────────────────

def evaluate_horizon(ensemble, y_future, weights_uniform, weights_gaussian,
                     h_start=0, h_end=None, batch_n=512):
    """
    Compute CRPS / MAE / RMSE for a horizon slice [h_start:h_end].

    Returns dict with keys:
      CRPS_uniform, MAE_uniform, RMSE_uniform,
      CRPS_gaussian, MAE_gaussian, RMSE_gaussian
    """
    if h_end is None:
        h_end = ensemble.shape[2]

    e = ensemble[:, :, h_start:h_end]    # (N, K, H)
    y = y_future[:, h_start:h_end]       # (N, H)

    out = {}
    for tag, wt in [("uniform", weights_uniform), ("gaussian", weights_gaussian)]:
        f = ensemble_forecast(e, wt)
        out[f"MAE_{tag}"]  = float(np.mean(np.abs(f - y)))
        out[f"RMSE_{tag}"] = float(np.sqrt(np.mean((f - y) ** 2)))
        out[f"CRPS_{tag}"] = float(crps(e, y, weights=wt, batch_n=batch_n).mean())
    return out


# ── Naive baseline ────────────────────────────────────────────────────────────

def naive_baseline(X_real, prefix_len=64):
    """
    Flat (random-walk) baseline: repeat last prefix value for all future steps.

    Returns dict: MAE_h32, RMSE_h32, CRPS_h32, MAE_h64, RMSE_h64, CRPS_h64.
    (CRPS of a deterministic forecast equals MAE.)
    """
    last  = X_real[:, prefix_len - 1:prefix_len]    # (N, 1)
    y_fut = X_real[:, prefix_len:]                   # (N, T-prefix_len)
    fc    = np.repeat(last, y_fut.shape[1], axis=1)  # (N, T-prefix_len)

    out = {}
    for h_end, tag in [(32, "h32"), (64, "h64")]:
        y    = y_fut[:, :h_end]
        f    = fc[:, :h_end]
        mae  = float(np.mean(np.abs(f - y)))
        rmse = float(np.sqrt(np.mean((f - y) ** 2)))
        out[f"MAE_{tag}"]  = mae
        out[f"RMSE_{tag}"] = rmse
        out[f"CRPS_{tag}"] = mae   # deterministic forecast: CRPS = MAE
    return out
