"""
Path Shadowing Monte-Carlo (PS-MC) for time-series generation evaluation.

Reference: Morel, Mallat, Bouchaud (2023) — arXiv:2308.01486
           "Path Shadowing Monte Carlo"

Algorithm
---------
1. Given a real path prefix of length `prefix_len`, retrieve the K nearest
   generated paths from the synthetic pool (L2 distance on the prefix).
2. Use the futures of those K paths (steps prefix_len:T) as a forecast ensemble.
3. Evaluate with CRPS, MAE, RMSE at multiple forecast horizons.

Two variants
------------
- **Uniform (basic):**     flat 1/K weight on every retrieved path.
- **Gaussian (advanced):** weight ∝ exp(-d²/(2η²)), η = median NN distance.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


# ── Core retrieval ────────────────────────────────────────────────────────────

def ps_mc_retrieve(X_real, X_fake, prefix_len=64, K=77):
    """
    Retrieve K nearest fake paths for every real query path.

    Parameters
    ----------
    X_real     : (N, T) real paths
    X_fake     : (M, T) generated paths (search pool)
    prefix_len : int — steps used for distance computation
    K          : int — number of nearest neighbours

    Returns
    -------
    ensemble  : (N, K, T-prefix_len) — futures of the K neighbours
    distances : (N, K) — L2 distances to each neighbour
    indices   : (N, K) — pool indices of the K neighbours
    """
    real_pre = X_real[:, :prefix_len]   # (N, prefix_len)
    fake_pre = X_fake[:, :prefix_len]   # (M, prefix_len)
    fake_fut = X_fake[:, prefix_len:]   # (M, T-prefix_len)

    nn = NearestNeighbors(n_neighbors=K, metric="euclidean",
                          algorithm="auto", n_jobs=16)
    nn.fit(fake_pre)
    distances, indices = nn.kneighbors(real_pre)    # both (N, K)

    ensemble = fake_fut[indices]                    # (N, K, T-prefix_len)
    return ensemble, distances, indices


# ── Weight computation ────────────────────────────────────────────────────────

def uniform_weights(N, K, dtype=np.float32):
    """Return uniform 1/K weights of shape (N, K)."""
    return np.full((N, K), 1.0 / K, dtype=dtype)


def gaussian_weights(distances, eta=None):
    """
    Gaussian weights: w_k ∝ exp(−d_k² / (2η²)).

    Parameters
    ----------
    distances : (N, K) — L2 distances from retrieve step
    eta       : float or None — bandwidth (default: median of all distances)

    Returns
    -------
    weights : (N, K) float32 — normalised Gaussian weights
    eta     : float — bandwidth actually used
    """
    if eta is None:
        eta = float(np.median(distances))
    log_w = -(distances.astype(np.float32) ** 2) / (2.0 * float(eta) ** 2 + 1e-30)
    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)
    return w.astype(np.float32), eta


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
    return np.einsum("nk,nkh->nh", weights.astype(np.float32),
                     ensemble.astype(np.float32)).astype(np.float64)


# ── CRPS (vectorised over N, not H) ──────────────────────────────────────────

def crps(ensemble, y_real, weights=None, batch_n=512):
    """
    Energy-score CRPS: E_w|Z−y| − ½ E_w|Z−Z'|.

    Vectorised implementation: loops over N-batches (not H-steps).
    Uses float32 internally for speed; returns float64.

    Parameters
    ----------
    ensemble : (N, K, H) float32 or float64
    y_real   : (N, H)    float32 or float64
    weights  : (N, K) or None (uniform 1/K)
    batch_n  : int — paths processed per batch (tune to VRAM/RAM)

    Returns
    -------
    crps_vals : (N, H) float64 — pointwise CRPS (mean over N,H = scalar CRPS)
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

    # Term 2: ½ Σ_{j,k} w_j w_k |z_j − z_k|
    # Vectorised over H; Python loop only over N-batches.
    term2 = np.zeros((N, H), dtype=np.float32)
    for n0 in range(0, N, batch_n):
        n1   = min(n0 + batch_n, N)
        e_b  = ens[n0:n1]              # (B, K, H)
        w_b  = w[n0:n1]               # (B, K)
        ww_b = w_b[:, :, None] * w_b[:, None, :]           # (B, K, K)
        diff = np.abs(e_b[:, :, None, :] - e_b[:, None, :, :])  # (B, K, K, H)
        term2[n0:n1] = 0.5 * np.einsum("bjk,bjkh->bh", ww_b, diff)

    return (term1 - term2).astype(np.float64)


# ── Horizon-level evaluation ──────────────────────────────────────────────────

def evaluate_horizon(ensemble, y_future, weights_uniform, weights_gaussian,
                     h_start=0, h_end=None, batch_n=512):
    """
    Compute CRPS / MAE / RMSE at a single horizon slice [h_start:h_end].

    Returns dict:
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
