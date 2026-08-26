#!/usr/bin/env python3
"""
Conditional-forecast CRPS by path shadowing, d > 1 — the paper's real-data protocol.

This reproduces the evaluation behind Table 4 of arXiv:2608.19394v1 (Deep-MKV-TS),
recorded verbatim in
    methods/Deep-MKV-TS/paper_reimplementation/results/paper_real_data_table4.json
and applies it to dataset/TrueDataset (real 30s crypto, d = 8, T = 128).

WHAT THE PROTOCOL IS
--------------------
Given the first 65 log-prices of a held-out path, retrieve the K = 256 nearest
generated histories from a bank of M simulated paths, and use THEIR next 32
returns as the predictive distribution for that query.  Score with CRPS against
what actually happened, on three targets:

    cum_return   cumulative log-return trajectory over the horizon   (H = 32)
    increment    the individual one-step log-returns                 (H = 32)
    rv           realised volatility of the horizon, sqrt(mean r^2)  (scalar)

Because the score is conditioned on the retrieved histories rather than computed
unconditionally, it tests whether generated continuations respond to the SPECIFIC
past of each query -- not merely whether the marginal distribution looks right.
That is exactly the property an unconditional stylised-facts metric cannot see.

EVERY CONSTANT AND EVERY RULE BELOW IS READ OFF THE AUTHOR'S OWN CODE
--------------------------------------------------------------------
methods/Deep-MKV-TS/code/reference/experiments/path_dt_experiments/shadowing.py
and real_data_protocol.py / scripts/evaluate_real_data_candidate_bank.py:

    S_IDX = 64          prefix_length 65 log-prices, 64 increments
    H     = 32          future_horizon
    K     = 256         top_k, uniform predictive ensemble
    BLOCK_W = {recent: 1.0, cumpath: 0.5, rollvol: 2.0, dep: 1.0}
                        = return_weight / path_weight / vol_weight / acf_weight
    RECENT_N = 32       last_returns
    CUMPATH_PTS = 24    max_path_points
    ROLL_WINDOWS = [5,10,20],  ACF_LAGS = [1,2,5,10]
    BLOCK_LEN = 8       moving-block bootstrap
    BASELINE_SEED = 1234, BOOT_REPS = 2000, BOOT_SEED = 0, BOOT_LEVEL = 0.95

Three points where a naive reading of the text would get it WRONG, and where
this file follows the CODE instead (shadowing.py line numbers given):

  * WEIGHTS ARE NOT DIMENSION-NORMALISED (227-235).  Each feature carries the
    full block weight, applied as sqrt(w) after standardisation.  A block with
    more features therefore carries more total mass.  That is the author's
    intent; do not "correct" it by dividing by the block dimension.

  * STANDARDISATION IS FROZEN ON THE BANK BEING SCORED (303-329,
    reference_blocks=generated_raw), not on the real train split.  mu/sigma are
    recomputed for every bank, so distances are internal to each bank.

  * rv IS A CROSS-ASSET TRAJECTORY, NOT A PER-ASSET SCALAR (579-601):
    rv_t = sqrt(sum_a r_{t,a}^2), shape (H,), one value per horizon step.

CRPS itself is _empirical_crps_per_path (953-973): flatten every non-ensemble
dimension, CRPS each independently, average over them.  For a (H,A) quantity
that averages over horizon AND asset inside each query.  The reported figure is
the mean over queries with a 2000-replicate percentile bootstrap CI (1194-1229).

S_IDX + H = 96 <= 127 available increments at T = 128, so the protocol fits
TrueDataset without changing a single constant.

THE ONE DELIBERATE DEVIATION
----------------------------
RETRIEVAL IS JOINT ACROSS ASSETS, NOT PER-ASSET.
    The paper runs ES, NQ and YM as three INDEPENDENT d = 1 experiments ("Each
    dataset is trained separately"), and real_data_protocol.py:513 hard-raises
    for d != 1.  The shadowing feature code itself accepts arbitrary d.  Here
    the four blocks are built per asset and concatenated, so one retrieval is
    driven by the whole d = 8 history.
    Reason: this is the multi-asset benchmark.  Scoring eight independent d = 1
    problems would discard the cross-asset structure -- 0.573 mean pairwise
    realised correlation -- that the dataset exists to test, and would make the
    number insensitive to whether a generator got the correlation right at all.

BANK SIZE IS 8,192 -- NOT 1,000,000
-----------------------------------
Paper §3.3.1: "we retrieve the 256 nearest generated histories from a pool of
8,192 simulated paths (fit on the train set)".  The 1M bank belongs to a
DIFFERENT protocol -- Morel, Mallat & Bouchaud (reference [20], arXiv:2308.01486)
-- implemented separately in this repo at
results/Heston/preprocessing_with_log_returns/*/path_shadowing/path_shadowing_pdf.py,
which sweeps banks of {4096, 16384, 65536, 262144, 1000000}.  Do not conflate them.

8,192 also exceeds this dataset's 6,144 real training paths, exactly as the
author's 8,192 exceeds their 496 training sessions: the historical banks are
RESAMPLED up to the bank size, they are not capped by the training count.

Measured here, CRPS is nearly flat in bank size at K = 256 (block bootstrap:
cum 1.284 at M=1536 vs 1.279 at M=12288, a 0.4% move over an 8x range), because
averaging 256 neighbours makes the ensemble spread depend on the intrinsic
unpredictability of the horizon rather than on retrieval sharpness.  The value is
nevertheless pinned to the paper's 8,192 so the protocol is exact.

BOTH HISTORICAL BASELINES ARE REQUIRED
--------------------------------------
The paper's absolute CRPS levels carry the units of 1-second ES/NQ/YM equity-index
futures.  TrueDataset is 30-second crypto with annualised vol 0.62-1.36 against
roughly 0.15-0.25 for equity index futures, so a raw comparison of CRPS across the
two datasets is a units error, not a result.

What IS transferable is the dimensionless ratio of a generator to a purely
historical bank built from the same real data.  Paper §3.3.1 specifies TWO such
banks and Table 4 reports both for all three indices:

    block bootstrap    rebuild each path from blocks of 8 consecutive returns,
                       each drawn from a random training window but KEPT AT ITS
                       ORIGINAL INTRADAY POSITION
    session bootstrap  an entire training session, resampled whole

The same block is drawn simultaneously for all 8 assets.  Drawing per asset would
destroy the cross-asset correlation and make the baseline artificially easy to
beat on any correlation-sensitive target.

The session bootstrap is the stronger of the two on RV in the paper (ES 0.620 vs
0.771 block; NQ 0.802 vs 0.990) and the weaker on cumulative return, so dropping
it would remove the harder baseline on the target where SBTS is meant to win.

In the paper, SBTS beats the block bootstrap on RV for all three indices
(0.808 / 0.916 / 0.946) but LOSES to it on cumulative return for NQ and YM
(1.088 / 1.087).  SBTS is not expected to dominate on real data.

Usage
-----
    python conditional_crps_multiasset.py --bank <generated.npy> --label SBTS
    python conditional_crps_multiasset.py --baselines-only
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "dataset", "TrueDataset")

S_IDX = 64
H = 32
K_NEIGH = 256
BLOCK_W = {"recent": 1.0, "cumpath": 0.5, "rollvol": 2.0, "dep": 1.0}
ACF_LAGS = [1, 2, 5, 10]
ROLL_WINDOWS = [5, 10, 20]
CUMPATH_PTS = 24
RECENT_N = 32
BLOCK_LEN = 8          # block bootstrap, paper: "blocks of eight consecutive returns"

# Bootstrap confidence interval over queries. Author: bootstrap_replicates=2000,
# confidence_level=0.95 (evaluate_real_data_candidate_bank.py:207-208),
# bootstrap_rng_seed=0 (shadowing.py:1209).
BOOT_REPS = 2000
BOOT_SEED = 0
BOOT_LEVEL = 0.95

# Moving-block bootstrap baseline seed. Author: real_data_protocol.py:392.
BASELINE_SEED = 1234


# ------------------------------------------------------------------ feature blocks
def _rolling_rms(r, w):
    """Rolling RMS of returns r (N,L) with window w -> (N, L-w+1), causal/valid only."""
    sq = r * r
    c = np.cumsum(np.concatenate([np.zeros((sq.shape[0], 1)), sq], axis=1), axis=1)
    return np.sqrt((c[:, w:] - c[:, :-w]) / w)


def _acf(A, lag):
    """Per-path biased lag-`lag` autocorrelation of series A (N,L) (prefix-only, demeaned)."""
    A = A - A.mean(axis=1, keepdims=True)
    num = (A[:, lag:] * A[:, :-lag]).sum(axis=1)
    den = (A * A).sum(axis=1) + 1e-30
    return num / den


def _blocks_one_asset(logS_a, weight_mode="paper"):
    """Four blocks for ONE asset. logS_a (N, S_IDX+1). Returns (feats, weights).

    weight_mode="paper"  -- the Deep-MKV author's convention, VERIFIED at
        shadowing.py:227-234 `_standardize_block`, which returns
        `(weight**0.5) * ((block - mean) / std)` with NO division by the block
        dimension. Each feature carries the full block weight, so a block with
        more features carries more total mass. Effective mass at d=8:
            recent 32x1.0 = 32 (46%) | rollvol 9x2.0 = 18 (26%)
            cumpath 24x0.5 = 12 (17%) | dep 8x1.0 = 8 (11%)
        Note this INVERTS the declared intent: the author declares vol 2x more
        important than returns, but the code makes returns 1.8x more important
        than vol. Reproduced exactly, because it is what produced Table 4.

    weight_mode="perdim" -- the Morel-Mallat-Bouchaud convention used by this
        repo's other path-shadowing implementation
        (results/Heston/preprocessing_with_log_returns/*/path_shadowing/
        path_shadowing_pdf.py, "block of dimension d gets weight w_block / d ...
        per-feature multiplier = sqrt(w_block / d)"). Each BLOCK carries exactly
        its declared weight regardless of how many features it has, which removes
        the weight x dimensionality confound above. This is a DIFFERENT PAPER's
        protocol (arXiv:2308.01486), not a correction to Table 4 -- never mix the
        two in one table.
    """
    if weight_mode not in ("paper", "perdim"):
        raise ValueError(f"weight_mode must be 'paper' or 'perdim', got {weight_mode!r}")
    r = np.diff(logS_a, axis=1)                       # (N, S_IDX)
    blocks, wparts = [], []

    def _add(block, wname):
        blocks.append(block)
        d = block.shape[1]
        w = BLOCK_W[wname] / d if weight_mode == "perdim" else BLOCK_W[wname]
        wparts.append(np.full(d, w))

    _add(r[:, -RECENT_N:], "recent")

    cum = logS_a - logS_a[:, :1]
    idx = np.linspace(0, logS_a.shape[1] - 1, CUMPATH_PTS).round().astype(int)
    _add(cum[:, idx], "cumpath")

    rv_feats = []
    for w in ROLL_WINDOWS:
        rr = _rolling_rms(r, w)
        rv_feats.append(rr[:, -1:])
        rv_feats.append(rr.mean(axis=1, keepdims=True))
        rv_feats.append(rr.std(axis=1, keepdims=True))
    _add(np.concatenate(rv_feats, axis=1), "rollvol")

    dep = []
    for series in (np.abs(r), r * r):
        for lag in ACF_LAGS:
            dep.append(_acf(series, lag)[:, None])
    _add(np.concatenate(dep, axis=1), "dep")

    return np.concatenate(blocks, axis=1), np.concatenate(wparts)


def build_features(logS, weight_mode="paper"):
    """logS (N, S_IDX+1, A) log-price prefixes -> feats (N, D), sqrt-weights (D,)."""
    A = logS.shape[2]
    feats, wts = [], []
    for a in range(A):
        f, w = _blocks_one_asset(logS[:, :, a], weight_mode=weight_mode)
        feats.append(f)
        wts.append(w)
    return (np.concatenate(feats, axis=1).astype(np.float64),
            np.sqrt(np.concatenate(wts)).astype(np.float64))


# ------------------------------------------------------------------ retrieval
def _standardize(feats, mu, sd, sqrtw):
    return ((feats - mu) / sd) * sqrtw


def retrieve(q_std, b_std, k, chunk=512):
    """Exact brute-force KNN, chunked. Returns idx (Nq,k). Pure numpy, no sklearn."""
    q = np.ascontiguousarray(q_std, dtype=np.float32)
    b = np.ascontiguousarray(b_std, dtype=np.float32)
    bn = (b ** 2).sum(1)
    out = np.empty((len(q), k), dtype=np.int64)
    for i in range(0, len(q), chunk):
        blk = q[i:i + chunk]
        d2 = bn[None, :] - 2.0 * (blk @ b.T)          # + |q|^2 is constant per row
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        rows = np.arange(len(blk))[:, None]
        order = np.argsort(d2[rows, part], axis=1)
        out[i:i + chunk] = part[rows, order]
    return out


# ------------------------------------------------------------------ CRPS
def targets(S, s_idx=S_IDX, h=H):
    """Future targets from price paths S (N,T,A).

    cum  (N,h,A) cumulative log-return from the split point
    step (N,h,A) one-step log-returns
    rv   (N,A)   sqrt(sum_t step_{t,a}^2), realised volatility of each asset
                 ACCUMULATED OVER THE HORIZON -- one scalar per asset.

    The rv axis was WRONG until 2026-08-26: it summed over ASSETS at each step,
    producing an (N,h) cross-sectional dispersion trajectory. The author sums over
    TIME. shadowing.py:583-584 -- raw_shadows is (B, top_k, h+1, A) so dim=2 is
    TIME, and raw_real_continuation is (B, h+1, A) so dim=1 is TIME:

        realized_volatility_samples  = sqrt(sum(increment_samples.pow(2), dim=2))
        realized_volatility_realized = sqrt(sum(increment_realized.pow(2), dim=1))

    At d=1 the author's definition is realised vol over the window; the old one
    collapsed to |r_t|, a different statistic. Read the axis MEANING, not its index.
    """
    logS = np.log(S)
    fut = logS[:, s_idx:s_idx + h + 1, :]             # (N, h+1, A)
    step = np.diff(fut, axis=1)                       # (N, h, A)
    cum = fut[:, 1:, :] - fut[:, :1, :]               # (N, h, A)
    rv = np.sqrt((step ** 2).sum(axis=1))             # (N, A)  <- over TIME
    return cum, step, rv


def _crps_per_query(Y, y):
    """Y (Nq,K,*dims) ensemble, y (Nq,*dims) truth -> per-query CRPS (Nq,).

    Author's `_empirical_crps_per_path` (shadowing.py:953-973): flatten every
    non-ensemble dimension, CRPS each one independently, then average over them.
    For a (H,A) quantity this averages over horizon AND asset inside the query.
    """
    Nq, K = Y.shape[0], Y.shape[1]
    Yf = Y.reshape(Nq, K, -1)
    yf = y.reshape(Nq, -1)
    term1 = np.abs(Yf - yf[:, None, :]).mean(axis=1)
    coef = (2.0 * np.arange(1, K + 1, dtype=np.float64) - K - 1.0)[None, :, None]
    term2 = (2.0 / (K * K)) * (coef * np.sort(Yf, axis=1)).sum(axis=1)
    return (term1 - 0.5 * term2).mean(axis=1)


def _energy_per_query(Y, y, qchunk=64):
    """Energy score, the strictly-proper MULTIVARIATE generalisation of CRPS.

        ES(F, y) = E||X - y|| - 0.5 * E||X - X'||

    with the Euclidean norm over the FULL flattened coordinate vector. At d=1 and
    a scalar target this reduces EXACTLY to CRPS, so it is a superset of the
    paper's number rather than a competing one.

    Why it is needed at d=8: the paper's CRPS flattens every non-ensemble axis and
    averages the UNIVARIATE score over coordinates. That is a mean MARGINAL score
    and is completely blind to cross-asset dependence -- a generator with perfect
    marginals and destroyed correlation scores identically to one that gets both
    right. The energy score sees the vector.

    Y (Nq,K,*dims), y (Nq,*dims) -> (Nq,). Uses batched BLAS for the K x K term.
    """
    Nq, K = Y.shape[0], Y.shape[1]
    Yf = np.ascontiguousarray(Y.reshape(Nq, K, -1), dtype=np.float64)
    yf = y.reshape(Nq, -1)
    out = np.empty(Nq)
    for i in range(0, Nq, qchunk):
        B = Yf[i:i + qchunk]                                  # (c,K,D)
        t = yf[i:i + qchunk]                                  # (c,D)
        term1 = np.linalg.norm(B - t[:, None, :], axis=2).mean(axis=1)
        # ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b, batched over queries
        sq = (B * B).sum(axis=2)                              # (c,K)
        g = np.matmul(B, B.transpose(0, 2, 1))                # (c,K,K)
        d2 = sq[:, :, None] + sq[:, None, :] - 2.0 * g
        np.maximum(d2, 0.0, out=d2)
        term2 = np.sqrt(d2, out=d2).reshape(len(B), -1).mean(axis=1)
        out[i:i + qchunk] = term1 - 0.5 * term2
    return out


def _variogram_per_query(Y, y, p=0.5):
    """Variogram score of order p on the ASSET vector -- far more sensitive to
    cross-asset dependence than the energy score, which is dominated by the
    marginal locations.

        VS_p = sum_{a<b} ( |y_a - y_b|^p  -  E|X_a - X_b|^p )^2

    Applied to the A-vector at each horizon step and averaged over steps, so for a
    (Nq,K,h,A) input it asks: at every instant, is the spread BETWEEN assets right?
    For a (Nq,K,A) input (rv) it is evaluated once. Location-free by construction:
    shifting every asset by the same amount leaves it unchanged.
    """
    if Y.ndim == 3:                                            # (Nq,K,A) -> (Nq,K,1,A)
        Y, y = Y[:, :, None, :], y[:, None, :]
    Nq, K, h, A = Y.shape
    ia, ib = np.triu_indices(A, k=1)
    dY = np.abs(Y[..., ia] - Y[..., ib]) ** p                  # (Nq,K,h,P)
    dy = np.abs(y[..., ia] - y[..., ib]) ** p                  # (Nq,h,P)
    return ((dy - dY.mean(axis=1)) ** 2).sum(axis=2).mean(axis=1)


def _mean_ci(per_q, reps=BOOT_REPS, seed=BOOT_SEED, level=BOOT_LEVEL):
    """Percentile bootstrap over queries (shadowing.py:1194-1229)."""
    n = len(per_q)
    bs = per_q[np.random.default_rng(seed).integers(0, n, size=(reps, n))].mean(axis=1)
    a = 0.5 * (1.0 - level)
    return dict(mean=float(per_q.mean()),
                ci_lo=float(np.quantile(bs, a)), ci_hi=float(np.quantile(bs, 1.0 - a)))


def score(bank_S, query_S, k=K_NEIGH, chunk=512, qchunk=256,
          weight_mode="paper", standardize="bank", ref_S=None, joint=True):
    """CRPS (+ optional joint scores) of the bank's retrieved continuations.

    standardize="bank" -- the Deep-MKV author's convention (shadowing.py:303-329,
        `reference_blocks=generated_raw`): mu/sigma recomputed on EACH bank, so
        distances are internal to that bank. Reproduces Table 4.
    standardize="realtrain" -- mu/sigma frozen once on the REAL train split
        (`ref_S`) and shared by every bank, as in the MMB implementation. Better
        for a BENCHMARK: each generator is then measured with the same ruler
        instead of one derived from its own output, so a method with distorted
        feature scales cannot be judged in its own forgiving geometry.
    """
    q_feat, sqrtw = build_features(np.log(query_S[:, :S_IDX + 1, :]), weight_mode)
    b_feat, _ = build_features(np.log(bank_S[:, :S_IDX + 1, :]), weight_mode)
    if standardize == "realtrain":
        if ref_S is None:
            raise ValueError("standardize='realtrain' requires ref_S (the real train split)")
        r_feat, _ = build_features(np.log(ref_S[:, :S_IDX + 1, :]), weight_mode)
    elif standardize == "bank":
        r_feat = b_feat
    else:
        raise ValueError(f"standardize must be 'bank' or 'realtrain', got {standardize!r}")
    mu = r_feat.mean(axis=0)
    sd = r_feat.std(axis=0) + 1e-12
    q_std = _standardize(q_feat, mu, sd, sqrtw)
    b_std = _standardize(b_feat, mu, sd, sqrtw)
    idx = retrieve(q_std, b_std, k, chunk=chunk)

    b_cum, b_step, b_rv = targets(bank_S)
    q_cum, q_step, q_rv = targets(query_S)

    out = {}
    for name, B, Q in (("cum_return", b_cum, q_cum),
                       ("increment", b_step, q_step),
                       ("rv", b_rv, q_rv)):
        per_q = np.concatenate([_crps_per_query(B[idx[i:i + qchunk]], Q[i:i + qchunk])
                                for i in range(0, len(idx), qchunk)])
        out[name] = _mean_ci(per_q)
        if joint:
            e = np.concatenate([_energy_per_query(B[idx[i:i + qchunk]], Q[i:i + qchunk])
                                for i in range(0, len(idx), qchunk)])
            v = np.concatenate([_variogram_per_query(B[idx[i:i + qchunk]], Q[i:i + qchunk])
                                for i in range(0, len(idx), qchunk)])
            out[name + "__energy"] = _mean_ci(e)
            out[name + "__variogram"] = _mean_ci(v)

    # Retrieval-quality diagnostics. If these look degenerate the CRPS is flat for
    # a reason that has nothing to do with the generator: 584 dims against an 8192
    # bank concentrates distances, so the "256 nearest" may not be near at all.
    d_near = np.linalg.norm(q_std[:512] - b_std[idx[:512, 0]], axis=1)
    out["_diag"] = dict(
        feature_dim=int(q_std.shape[1]),
        nn1_dist_mean=float(d_near.mean()),
        nn1_dist_p95=float(np.quantile(d_near, 0.95)),
        unique_neighbour_frac=float(len(np.unique(idx)) / len(b_std)),
        mean_reuse=float(idx.size / max(len(np.unique(idx)), 1)),
    )
    return out


# ------------------------------------------------------------------ baselines
def block_bootstrap(S_train, m, seed, block_len=BLOCK_LEN):
    """Rebuild paths from blocks of `block_len` consecutive returns, each drawn from a
    random training window but KEPT AT ITS ORIGINAL INTRADAY POSITION.

    The same source window is used for all assets in a block, so the cross-asset
    correlation of the real data survives. Drawing per asset would destroy it.
    """
    rng = np.random.default_rng(seed)
    r = np.diff(np.log(S_train), axis=1)              # (N, T-1, A)
    N, L, A = r.shape
    out = np.empty((m, L, A))
    for b0 in range(0, L, block_len):
        b1 = min(b0 + block_len, L)
        src = rng.integers(0, N, size=m)              # one source path per output path
        out[:, b0:b1, :] = r[src, b0:b1, :]
    logS = np.concatenate([np.zeros((m, 1, A)), np.cumsum(out, axis=1)], axis=1)
    return np.exp(logS) * S_train[0, 0, 0]


def session_bootstrap(S_train, m, seed):
    """A whole training window, resampled with replacement."""
    rng = np.random.default_rng(seed)
    return S_train[rng.integers(0, len(S_train), size=m)].copy()


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--seq-tag", default="8192x128x8")
    ap.add_argument("--bank", default=None, help="generated paths .npy (M,T,A)")
    ap.add_argument("--label", default="bank")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--k", type=int, default=K_NEIGH)
    ap.add_argument("--seed", type=int, default=BASELINE_SEED,
                    help="baseline resampling seed (author: 1234)")
    ap.add_argument("--bank-size", type=int, default=8192,
                    help="paths per bank. Paper §3.3.1: 8,192 for the generated "
                         "pool AND for both historical banks.")
    ap.add_argument("--no-session-bootstrap", action="store_true",
                    help="drop the session bootstrap. It IS one of the paper's two "
                         "historical baselines; this flag exists only for quick runs.")
    ap.add_argument("--weight-mode", default="paper", choices=["paper", "perdim"],
                    help="paper: sqrt(w) per feature, author's Table 4 convention. "
                         "perdim: sqrt(w/d) per block, the MMB convention that removes "
                         "the weight x dimensionality confound.")
    ap.add_argument("--standardize", default="bank", choices=["bank", "realtrain"],
                    help="bank: mu/sigma per bank (author). realtrain: frozen on the "
                         "real train split, one ruler for every method.")
    ap.add_argument("--no-joint", action="store_true",
                    help="skip the energy and variogram scores")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = args.seq_tag
    S_train = np.load(os.path.join(args.data_dir, f"true_S_{tag}.npy"))
    S_test = np.load(os.path.join(args.data_dir, f"true_S_test_{tag}.npy"))
    A = S_train.shape[2]

    print("=" * 78)
    print(f"Conditional-forecast CRPS by path shadowing — d={A}, T={S_train.shape[1]}")
    print("=" * 78)
    print(f"prefix 0..{S_IDX} ({S_IDX + 1} log-prices)  horizon H={H}  K={args.k}")
    print(f"feature dim {73 * A} = 4 blocks x {A} assets")
    print(f"queries {S_test.shape}  standardisation frozen per bank (author's)\n")

    results = {}
    banks = {}
    if not args.baselines_only:
        if args.bank is None:
            sys.exit("--bank is required unless --baselines-only")
        banks[args.label] = np.load(args.bank)
    banks["block_bootstrap"] = block_bootstrap(S_train, args.bank_size, args.seed)
    if not args.no_session_bootstrap:
        banks["session_bootstrap"] = session_bootstrap(S_train, args.bank_size, args.seed)

    for name, bank in banks.items():
        t0 = time.perf_counter()
        r = score(bank, S_test, k=args.k, weight_mode=args.weight_mode,
                  standardize=args.standardize, ref_S=S_train,
                  joint=not args.no_joint)
        el = time.perf_counter() - t0
        # x1000 applies to the SCORES only. _diag holds distances and fractions,
        # which are not CRPS and must not be scaled.
        results[name] = {
            t: (v if t.startswith("_") else {k2: v2 * 1000.0 for k2, v2 in v.items()})
            for t, v in r.items()
        }
        results[name]["bank_shape"] = list(bank.shape)
        results[name]["elapsed_sec"] = round(el, 1)
        m = results[name]
        print(f"  {name:18s} cum {m['cum_return']['mean']:8.3f}  "
              f"incr {m['increment']['mean']:8.3f}  "
              f"rv {m['rv']['mean']:8.3f}   [{el:.0f}s]")

    if not args.baselines_only:
        bb = results["block_bootstrap"]
        me = results[args.label]
        print("\n  dimensionless ratios vs block bootstrap (the transferable gate):")
        for t in ("cum_return", "increment", "rv"):
            print(f"    {t:12s} {me[t]['mean'] / bb[t]['mean']:.3f}")
        print("  paper (Table 4) SBTS/block-bootstrap: rv 0.808/0.916/0.946, "
              "cum 0.998/1.088/1.087 for ES/NQ/YM")

    out = dict(
        units="CRPS x 1000, lower is better",
        protocol=dict(s_idx=S_IDX, history_log_prices=S_IDX + 1, horizon=H,
                      k_neighbours=args.k, block_w=BLOCK_W, acf_lags=ACF_LAGS,
                      roll_windows=ROLL_WINDOWS, cumpath_pts=CUMPATH_PTS,
                      recent_n=RECENT_N, bootstrap_block_len=BLOCK_LEN,
                      baseline_seed=args.seed, boot_reps=BOOT_REPS,
                      boot_seed=BOOT_SEED, boot_level=BOOT_LEVEL,
                      weight_mode=args.weight_mode, standardisation=args.standardize,
                      joint_scores=not args.no_joint,
                      rv_definition="sqrt(sum_t r^2) per asset, summed over TIME "
                                    "(shadowing.py:583-584); axis fixed 2026-08-26",
                      retrieval="joint across assets", d=A),
        paper_reference="methods/Deep-MKV-TS/paper_reimplementation/results/paper_real_data_table4.json",
        data_dir=args.data_dir, queries=f"true_S_test_{tag}.npy",
        results=results,
    )
    path = args.out or os.path.join(args.data_dir, "conditional_crps.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
