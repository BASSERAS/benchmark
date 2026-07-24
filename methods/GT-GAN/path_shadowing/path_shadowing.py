"""
Path Shadowing Monte-Carlo (PS-MC) for time-series generation evaluation.

Reference: Morel, Mallat, Bouchaud (2023) — arXiv:2308.01486
           "Path Shadowing Monte Carlo"

Algorithm
---------
1. Embed every path prefix as a 65D feature vector (see Embedding below).
2. For each real query path, retrieve the K nearest generated paths by L2
   distance in the z-scored embedding space.
3. Use the futures of those K paths as a forecast ensemble (price-anchored).
4. Evaluate with CRPS, MAE, RMSE at multiple forecast horizons.

Embedding — 65D murex-style prefix features
--------------------------------------------
Adapted from Murex/deep-mkv-gen-path-dt (experiments/path_dt_experiments/shadowing.py).
For a prefix of length L (L−1 available log-returns):

    Component 1 — full log-return trajectory (L−1 dims):
        r_t = log S_t − log S_{t−1},  for t = 1 … L−1

    Component 2 — terminal cumulative return (1 dim):
        R = log S_{L−1} − log S_0

    Component 3 — realized volatility (1 dim):
        σ = sqrt( mean(r_t²) )

Total dimension: (L−1) + 1 + 1 = L+1.  With prefix_len=64 → 65D.

Each feature dimension is z-scored using the mean and std of the GENERATED
pool, so distances are scale-invariant across feature dimensions.

Why this outperforms the 22D eq.(13) embedding from arXiv:2308.01486:
    Eq.(13) captures only endpoint-relative multi-scale differences
    (market regime), not the full path shape.  Two paths with different
    trajectories can share the same 22D embedding if their endpoint-to-lag
    differences coincide.  The 65D embedding retains the full return
    trajectory, so the KNN selects paths whose PREFIX SHAPE is closest.

Gaussian bandwidth — adaptive per-run calibration
--------------------------------------------------
    η̃_adapt = median(distances) / median(‖z(x̃_past)‖)
    η_i     = η̃_adapt · ‖z(x̃_past_i)‖

where ‖z(·)‖ is the L2 norm of the z-scored feature vector.
This replaces the paper's fixed η̃=0.075 (calibrated on S&P) with a
data-driven value matched to the Heston embedding scale.

KNN selection — NOT combinatorial
----------------------------------
sklearn NearestNeighbors computes L2(query_emb, pool_emb) for all pool paths
and returns the K with the smallest distance.  Cost: O(N × D) per batch.
No subset enumeration; no combinatorial search.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


# ── Embedding ─────────────────────────────────────────────────────────────────

def _murex_embedding(X, prefix_len, eps=1e-8):
    """
    65D prefix embedding:
      - (prefix_len-1) log-returns, flattened
      - 1 terminal cumulative return
      - 1 realized volatility = sqrt(mean(r²))

    Returns (N, prefix_len+1) float32.
    """
    prefix   = np.log(np.clip(X[:, :prefix_len].astype(np.float64), 1e-8, None))
    returns  = prefix[:, 1:] - prefix[:, :-1]          # (N, prefix_len-1)
    terminal = prefix[:, -1] - prefix[:, 0]             # (N,)
    rvol     = np.sqrt(np.mean(returns ** 2, axis=1))   # (N,)
    return np.concatenate(
        [returns, terminal[:, None], rvol[:, None]], axis=1
    ).astype(np.float32)                                 # (N, prefix_len+1)


# ── Core retrieval ────────────────────────────────────────────────────────────

def ps_mc_retrieve(X_real, X_fake, prefix_len=64, K=77, eps=1e-8, **_ignored):
    """
    Retrieve K nearest fake paths for every real query path.

    Distance is computed in the 65D murex embedding space, z-scored using
    the generated pool distribution.

    Parameters
    ----------
    X_real     : (N, T) real paths
    X_fake     : (P, T) generated paths (search pool)
    prefix_len : int — steps used for embedding / distance
    K          : int — number of nearest neighbours
    eps        : float — numerical floor for std normalisation

    Returns
    -------
    ensemble        : (N, K, T-prefix_len) — price-anchored futures of K neighbours
    distances       : (N, K) — L2 distances in z-scored embedding space
    indices         : (N, K) — pool indices of the K neighbours
    real_emb_norms  : (N,)   — ‖z(x̃_past_i)‖, needed for adaptive Gaussian η
    """
    real_emb = _murex_embedding(X_real, prefix_len, eps)  # (N, D)
    fake_emb = _murex_embedding(X_fake, prefix_len, eps)  # (P, D)

    # Z-score each dimension using the generated pool
    mean = fake_emb.mean(axis=0, keepdims=True).astype(np.float32)
    std  = fake_emb.std(axis=0, keepdims=True).astype(np.float32)
    std  = np.where(std < float(eps), float(eps), std)

    real_std = (real_emb - mean) / std                    # (N, D)
    fake_std = (fake_emb - mean) / std                    # (P, D)

    nn = NearestNeighbors(n_neighbors=K, metric="euclidean",
                          algorithm="auto", n_jobs=16)
    nn.fit(fake_std)
    distances, indices = nn.kneighbors(real_std)           # both (N, K)

    fake_fut     = X_fake[:, prefix_len:]                  # (P, H)
    ensemble_raw = fake_fut[indices]                       # (N, K, H)

    # Price anchoring: scale each fake future so it starts at the real path's
    # last prefix price.
    S_real_last = np.clip(X_real[:, prefix_len - 1], 1e-8, None)        # (N,)
    S_fake_last = np.clip(X_fake[indices, prefix_len - 1], 1e-8, None)  # (N, K)
    scale       = S_real_last[:, None] / S_fake_last                     # (N, K)
    ensemble    = ensemble_raw * scale[:, :, None]                       # (N, K, H)

    # Norms of z-scored real embeddings (for adaptive Gaussian bandwidth)
    real_emb_norms = np.linalg.norm(
        real_std.astype(np.float64), axis=1
    )                                                                     # (N,)

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
