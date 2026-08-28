#!/usr/bin/env python3
"""
Intraday portfolio VaR / ES backtest on the path-shadowing conditional ensemble.

WHAT THIS IS, IN ONE SENTENCE
-----------------------------
Take the K = 256 retrieved continuations that conditional_crps_multiasset.py
already scores with CRPS, read the alpha-quantile of their equal-weight portfolio
return off the ensemble, call it a VaR forecast, and count how often the real
continuation breached it.

WHY IT IS NOT REDUNDANT WITH CRPS
---------------------------------
CRPS is a strictly proper score of the WHOLE predictive law, averaged over
coordinates.  It is dominated by the centre of the distribution: a forecast that
nails the bulk and collapses the tail loses very little CRPS.  Coverage is the
opposite -- it looks ONLY at the tail and it is a frequency, not a distance, so
it cannot be bought back by being sharp in the middle.

That difference is not hypothetical here.  Table C ranks CSDI and Deep-MKV-TS
BELOW the real-data floor on cum_return CRPS (1.248 vs 1.257).  A CRPS below the
real-data floor is the memoriser/over-sharpness signature (see README section 7.1);
coverage is the instrument that says whether that sharpness is earned or is simply
a distribution too narrow to be used for risk.

THE HORIZON IS 16 MINUTES.  SAY SO.
-----------------------------------
H = 32 bars x 30 s = 960 s = 16 minutes.  This is an intraday LIQUIDATION-horizon
risk figure -- roughly the time to unwind a crypto position -- and it is NOT an
FRTB/Basel capital number, which is a 10-day quantity.  Do not scale it to one day
by sqrt(t): that requires returns to be independent across time, which is the
exact assumption the volatility-clustering results in this repo refute.  There is
no sqrt-scaling anywhere in this file, on purpose.

THE HEADLINE FINDING, AND WHY THE FLOOR ROW IS MANDATORY
--------------------------------------------------------
Measured (K = 256, equal-weight 8 assets, holdout-era split):

    bank                 alpha=1%   5%      10%     ens_sd/bank_sd
    real train (floor)     3.80%   12.02%  18.66%      0.62
    block bootstrap        2.06%    5.27%   8.41%      0.99
    SBTS                   4.80%    9.38%  14.25%      0.97

The REAL TRAINING DATA, used as its own bank, over-breaches by 2.4x at alpha=5%.
No generator is involved in that row.  So the raw exception rate is NOT a clean
verdict on a generator -- the retrieval step itself over-tightens the predictive
law, discarding ~38% of the unconditional spread (ens_sd/bank_sd = 0.62) in
exchange for conditioning that does not pay for itself out of sample.

The block bootstrap is the control that proves the mechanism: it rebuilds paths
from blocks drawn at random, so its "nearest neighbours" carry no real
conditioning information, its ensemble keeps ~99% of the unconditional spread,
and its coverage is close to nominal at every level.  The dumbest baseline wins
the risk test.  That is the result, not a bug.

CONSEQUENCE FOR THE TABLE: report the exception rate AND its ratio to the real-
data floor, exactly as Table C reports xfloor.  The ratio absorbs the protocol's
own coverage bias and isolates what the generator contributed.  Reporting the raw
rate alone would blame the generators for the retrieval step.

PART OF THE MISS IS THE MARKET, NOT THE MODEL
---------------------------------------------
Between the two eras (train 2022-07..2024-11, test 2025-02..2026-07):

    per-asset bar vol   0.000967 -> 0.001036   (+7.1%)
    mean pair corr        0.5644 -> 0.6288     (+0.064)
    portfolio bar vol   0.000750 -> 0.000845   (+12.7%)

Portfolio vol rose nearly twice as fast as per-asset vol.  The gap is
diversification decay: an equal-weight book got riskier even where the individual
legs did not.  Any bank fitted on the train era under-prices the test era for
that reason alone, which is what the `uncond` column isolates (floor: 1.42% at
alpha=1%, i.e. 1.4x nominal BEFORE retrieval tightens it to 3.80%).

WHY alpha = 5% IS THE HEADLINE AND alpha = 1% IS A FOOTNOTE
-----------------------------------------------------------
With K = 256 the empirical alpha-quantile sits at order statistic ~alpha*K.  At
alpha = 5% that is the 13th of 256 -- a stable estimate.  At alpha = 1% it is the
~2.6th, and the finite-sample bias of an empirical tail quantile makes VaR
slightly too tight (under a normal, E[X_(3)] of 256 draws is Phi^-1(3/257) =
-2.27 vs the true -2.33, so ~1.17% realised at 1% nominal).  That bias is small
next to the 3.80% observed, so it does not explain the finding -- but it is a
reason to headline the level where the estimator is not fighting you.

TESTS REPORTED
--------------
kupiec  LR_uc, Kupiec (1995) unconditional coverage, chi2(1).
        H0: P(breach) = alpha.  Frequency only; blind to clustering.

chris   LR_cc = LR_uc + LR_ind, Christoffersen (1998), chi2(2), with LR_ind the
        first-order Markov independence test.  H0: breaches are Bernoulli(alpha)
        AND independent.  A model can pass Kupiec and fail this by getting the
        average right while bunching every breach into one afternoon.

        THE INDEPENDENCE TEST IS LEGITIMATE HERE, which is unusual.  Most VaR
        backtests on overlapping windows cannot run it, because an H-step
        overlapping return induces mechanical serial dependence in the breach
        sequence and inflates rejections.  This dataset's windows are DISJOINT:
        dataset_stats.json gives n_windows_available = 33570 against ~4.21M bars
        at seq_len 128 (4.21e6/128 = 32.9e3), i.e. stride == seq_len, no overlap.
        Transitions are still evaluated WITHIN contiguous blocks only (--block-
        size, default 256 = dataset_stats.json block_windows) so that the seam
        between two non-adjacent blocks is never counted as a transition.

es      Expected Shortfall at the same alpha, reported as the ratio
            realised mean loss given breach / predicted mean loss given breach
        ES is NOT elicitable (Gneiting 2011), so there is deliberately no p-value
        attached to it.  It is a calibration diagnostic: > 1 means that when the
        model is wrong it is wrong by more than it admitted.  Cite it as a
        magnitude, never as a test.

Usage
-----
    python var_backtest_multiasset.py                 # full sweep, all banks
    python var_backtest_multiasset.py --k-sweep       # + mechanism experiment
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import conditional_crps_multiasset as M  # noqa: E402

# THE LOCKED BUILD.  Not dataset/TrueDataset -- that top-level directory also holds
# an 8192-row build, and every published table in results/trueexperiment/ is computed
# on the N6144 variant instead (read it off any losses/crps_configs/paper__seed_*.json:
# "data_dir": ".../variants/om_2022-07_N6144", "queries": "true_S_test_6144x128x8.npy").
#
# Defaulting to the top-level directory silently produced a Table D on 8192 different
# queries than Table C, while every number still looked plausible.  The defaults below
# and the queries cross-check in results/trueexperiment/table_d.py exist so that
# mistake cannot be made again without something failing loudly.
DATA = os.path.join(REPO, "dataset", "TrueDataset", "variants", "om_2022-07_N6144")
SEQ_TAG = "6144x128x8"
TE = os.path.join(REPO, "results", "trueexperiment")

LEVELS = (0.01, 0.05, 0.10)
BLOCK_WINDOWS = 256          # dataset_stats.json block_windows
BAR_SECONDS = 30             # dataset_stats.json bar_seconds
METHODS = ("SBTS", "CSDI", "Deep-MKV-TS", "reference")
SEEDS = (0, 1, 2, 3, 4)
BASELINE_SEEDS = (1234, 1235, 1236, 1237, 1238)
EWMA_LAMBDA = 0.94           # RiskMetrics (J.P. Morgan 1996), the desk default


# ------------------------------------------------------------------ statistics
def _chi2_sf(x, df):
    """Upper tail of chi2, df in {1,2}, closed form -- avoids a scipy dependency.

    df=1: P(X>x) = erfc(sqrt(x/2)).   df=2: P(X>x) = exp(-x/2).
    """
    if x is None or (isinstance(x, float) and math.isnan(x)) or x < 0:
        return float("nan")
    if df == 1:
        return math.erfc(math.sqrt(x / 2.0))
    if df == 2:
        return math.exp(-x / 2.0)
    raise ValueError(f"df must be 1 or 2, got {df}")


def kupiec_lr(n, x, alpha):
    """Kupiec (1995) POF unconditional-coverage LR. n trials, x breaches -> (LR, p).

    LR_uc = -2 log [ (1-a)^(n-x) a^x / (1-p)^(n-x) p^x ],  p = x/n,  ~ chi2(1).
    Returns (nan, nan) for the degenerate x in {0, n}, where the MLE is on the
    boundary and the asymptotic chi2 does not hold.
    """
    if n == 0 or x == 0 or x == n:
        return float("nan"), float("nan")
    p = x / n
    ll0 = (n - x) * math.log(1 - alpha) + x * math.log(alpha)
    ll1 = (n - x) * math.log(1 - p) + x * math.log(p)
    lr = -2.0 * (ll0 - ll1)
    return lr, _chi2_sf(lr, 1)


def christoffersen_lr(breach, alpha, block=BLOCK_WINDOWS):
    """Christoffersen (1998) conditional coverage. breach (N,) bool -> dict.

    LR_ind tests first-order Markov independence via the 2x2 transition counts
    n_ij (i = state at t-1, j = state at t).  LR_cc = LR_uc + LR_ind ~ chi2(2).

    Transitions are counted only WITHIN each contiguous run of `block` queries.
    The test split is a set of blocks that are contiguous in time internally but
    not adjacent to each other, so the seam between block b and block b+1 is not
    a real time step and must not contribute a transition.  Passing block=0
    disables the split and treats the whole sequence as contiguous.
    """
    b = np.asarray(breach).astype(np.int64)
    n = len(b)
    segs = ([b[i:i + block] for i in range(0, n, block)] if block else [b])
    n00 = n01 = n10 = n11 = 0
    for s in segs:
        if len(s) < 2:
            continue
        prev, cur = s[:-1], s[1:]
        n00 += int(((prev == 0) & (cur == 0)).sum())
        n01 += int(((prev == 0) & (cur == 1)).sum())
        n10 += int(((prev == 1) & (cur == 0)).sum())
        n11 += int(((prev == 1) & (cur == 1)).sum())

    lr_uc, _ = kupiec_lr(n, int(b.sum()), alpha)
    denom0, denom1 = n00 + n01, n10 + n11
    if denom0 == 0 or denom1 == 0 or n01 + n11 == 0:
        return dict(lr_ind=float("nan"), p_ind=float("nan"),
                    lr_cc=float("nan"), p_cc=float("nan"),
                    pi0=float("nan"), pi1=float("nan"))
    pi0 = n01 / denom0                       # P(breach_t | no breach_{t-1})
    pi1 = n11 / denom1                       # P(breach_t | breach_{t-1})
    pi = (n01 + n11) / (denom0 + denom1)

    def _l(k, p):
        return k * math.log(p) if k and p > 0 else 0.0

    ll_ind = _l(n00, 1 - pi0) + _l(n01, pi0) + _l(n10, 1 - pi1) + _l(n11, pi1)
    ll_0 = _l(n00 + n10, 1 - pi) + _l(n01 + n11, pi)
    lr_ind = -2.0 * (ll_0 - ll_ind)
    lr_cc = (lr_uc + lr_ind) if not math.isnan(lr_uc) else float("nan")
    return dict(lr_ind=lr_ind, p_ind=_chi2_sf(lr_ind, 1),
                lr_cc=lr_cc,
                p_cc=(_chi2_sf(lr_cc, 2) if not math.isnan(lr_cc) else float("nan")),
                pi0=pi0, pi1=pi1)


def _norm_ppf(p):
    """Acklam's inverse normal CDF, |error| < 1.15e-9.  Avoids a scipy dependency."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ------------------------------------------------------------------ core
def portfolio(cum, w):
    """Terminal equal-weight portfolio log-return from cum (N,H,A) -> (N,).

    Uses the FINAL horizon step: the H-bar (16 min) holding-period return, which
    is the quantity a liquidation-horizon VaR is about.  Log-returns are combined
    across assets with weights w; at these magnitudes (|r| ~ 1e-3) the difference
    from a properly compounded arithmetic portfolio is O(1e-6), far below the
    Monte-Carlo error of a 256-member ensemble.
    """
    return cum[:, -1, :] @ w


def _tail_mean(row, v):
    """Mean of the ensemble members at or below its own VaR.  Falls back to the
    minimum when the quantile lands below every member (possible at small k)."""
    m = row[row <= v]
    return float(m.mean()) if m.size else float(row.min())


def _ewma_var(S_test, w, levels):
    """RiskMetrics EWMA(0.94) portfolio VaR from the query prefix alone.

    The practitioner benchmark.  For each query: build the equal-weight portfolio
    return series over the 64 observed prefix bars, run an EWMA variance recursion
    with lambda = 0.94, scale the one-bar sd to the H-bar horizon by sqrt(H), and
    read a Gaussian quantile off it.

    The sqrt(H) here is INTERNAL TO THE BENCHMARK, not applied to our own numbers:
    the whole point of this row is to show what a desk running the standard
    conditionally-Gaussian recipe would have produced, and sqrt-scaling is part of
    that recipe.  Its failure or success is therefore informative about the recipe.
    """
    r = np.diff(np.log(S_test[:, :M.S_IDX + 1, :]), axis=1) @ w     # (Nq, 64)
    v = (r[:, :20] ** 2).mean(axis=1)                               # burn-in seed
    for t in range(20, r.shape[1]):
        v = EWMA_LAMBDA * v + (1.0 - EWMA_LAMBDA) * r[:, t] ** 2
    sd_h = np.sqrt(v * M.H)
    var, es = {}, {}
    for a in levels:
        z = _norm_ppf(a)
        var[a] = z * sd_h
        # Gaussian ES: -phi(z)/a * sigma, closed form.
        es[a] = -(math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)) / a * sd_h
    return var, es


def backtest_one(job):
    """One (bank, K) cell.  Runs in a worker process; must stay picklable."""
    label, path, kind, seed, k, levels, data_dir, tag, split = job
    t0 = time.perf_counter()
    S_test = np.load(os.path.join(data_dir, f"true_S_{split}_{tag}.npy"))
    S_tr = np.load(os.path.join(data_dir, f"true_S_{tag}.npy"))
    A = S_test.shape[2]
    w = np.full(A, 1.0 / A)
    # The bank size is whatever the bank file holds (8192 for the generated pools,
    # per paper 3.3.1).  The historical baselines are resampled UP to that same
    # size, which is why bank_size is read off the generated pool rather than off
    # the train split -- the train split only has 6144 paths.
    bank_size = 8192

    if kind == "file":
        bank = np.load(path)
    elif kind == "real_train":
        bank = S_tr
    elif kind == "block":
        bank = M.block_bootstrap(S_tr, bank_size, seed)
    elif kind == "session":
        bank = M.session_bootstrap(S_tr, bank_size, seed)
    elif kind == "ewma":
        bank = None
    else:
        raise ValueError(f"unknown kind {kind!r}")

    q_cum, _, _ = M.targets(S_test)
    real = portfolio(q_cum, w)

    if kind == "ewma":
        var, es_pred = _ewma_var(S_test, w, levels)
        es_pred = {a: np.full(len(real), 0.0) + es_pred[a] for a in levels}
        diag = dict(model="RiskMetrics EWMA", lam=EWMA_LAMBDA)
    else:
        qf, sqrtw = M.build_features(np.log(S_test[:, :M.S_IDX + 1, :]), "paper")
        bf, _ = M.build_features(np.log(bank[:, :M.S_IDX + 1, :]), "paper")
        mu, sd = bf.mean(axis=0), bf.std(axis=0) + 1e-12
        idx = M.retrieve(M._standardize(qf, mu, sd, sqrtw),
                         M._standardize(bf, mu, sd, sqrtw), k)
        b_cum, _, _ = M.targets(bank)
        bank_pf = portfolio(b_cum, w)
        ens = bank_pf[idx]                                    # (Nq, k)
        var = {a: np.quantile(ens, a, axis=1) for a in levels}
        es_pred = {a: np.array([_tail_mean(ens[i], var[a][i]) for i in range(len(ens))])
                   for a in levels}
        # ens_sd/bank_sd is the shrinkage the retrieval step applies.  ~1.0 means
        # the "conditioning" carries no information; << 1.0 means it is confident,
        # and coverage says whether that confidence was earned.
        diag = dict(ens_sd=float(ens.std(axis=1).mean()),
                    bank_sd=float(bank_pf.std()),
                    shrinkage=float(ens.std(axis=1).mean() / (bank_pf.std() + 1e-30)),
                    bank_shape=list(bank.shape))

    out = {}
    for a in levels:
        br = real < var[a]
        x, n = int(br.sum()), len(br)
        lr, p = kupiec_lr(n, x, a)
        cc = christoffersen_lr(br, a)
        # Realised ES: mean loss GIVEN a breach.  Predicted ES: the model's own
        # mean below its VaR, averaged over the queries that breached -- compared
        # like for like, on the same subset.
        es_real = float(real[br].mean()) if x else float("nan")
        es_p = float(es_pred[a][br].mean()) if x else float("nan")
        out[f"alpha_{a:.2f}"] = dict(
            alpha=a, n=n, breaches=x, rate=x / n, expected=a * n,
            kupiec_lr=lr, kupiec_p=p, **cc,
            es_realised=es_real, es_predicted=es_p,
            es_ratio=(es_real / es_p if x and es_p and not math.isnan(es_p)
                      else float("nan")),
        )
    return dict(label=label, kind=kind, seed=seed, k=k, split=split,
                elapsed_sec=round(time.perf_counter() - t0, 2),
                # realised_sd is the yardstick the ensemble width is judged against
                # and it belongs in the artefact, not in a renderer's default arg.
                realised_sd=float(real.std()),
                diag=diag, levels=out)


# ------------------------------------------------------------------ manifest
def _bank_path(method, seed):
    """Generated 8192-path pool for one method/seed, or None.

    This is the SAME artefact conditional_crps_multiasset.py scores -- see the
    --bank argument in results/trueexperiment/CSDI/code/run_pipeline.sh:245.  The
    pool is 8192 regardless of the 6144-row dataset variant, per paper 3.3.1.
    """
    p = os.path.join(TE, method, "crps_banks", "generated_paths",
                     f"seed_{seed}", "generated_paths_8192x128x8.npy")
    return p if os.path.exists(p) else None


def build_jobs(k_list, levels, data_dir, tag, splits=("test",)):
    """Job manifest.  `splits` > 1 runs the in-era control (val) alongside test."""
    jobs = []
    for split in splits:
        common = (levels, data_dir, tag, split)
        jobs.append(("real_train_bank", None, "real_train", 0, M.K_NEIGH) + common)
        jobs.append(("ewma", None, "ewma", 0, 0) + common)
        for s in BASELINE_SEEDS:
            jobs.append(("block_bootstrap", None, "block", s, M.K_NEIGH) + common)
            jobs.append(("session_bootstrap", None, "session", s, M.K_NEIGH) + common)
        for m in METHODS:
            for s in SEEDS:
                p = _bank_path(m, s)
                if p:
                    jobs.append((m, p, "file", s, M.K_NEIGH) + common)
        # Mechanism experiment: sweep K on one seed per bank.  As K -> M the
        # ensemble converges to the unconditional bank distribution, so if the
        # over-breaching is caused by retrieval shrinkage the coverage must walk
        # back toward nominal.  Run on the test split only; the val split is a
        # control on the era, not on K.
        if split != "test":
            continue
        for k in k_list:
            if k == M.K_NEIGH:
                continue
            jobs.append(("real_train_bank", None, "real_train", 0, k) + common)
            jobs.append(("block_bootstrap", None, "block", 1234, k) + common)
            for m in METHODS:
                p = _bank_path(m, 0)
                if p:
                    jobs.append((m, p, "file", 0, k) + common)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA,
                    help="MUST match the data_dir in losses/crps_configs/paper__seed_*.json, "
                         "or Table D is computed on different queries than Table C")
    ap.add_argument("--seq-tag", default=SEQ_TAG)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k-sweep", action="store_true",
                    help="also sweep K to demonstrate the shrinkage mechanism")
    ap.add_argument("--val-control", action="store_true",
                    help="also backtest against the validation split, which is the SAME "
                         "era as the bank. Separates protocol failure from regime shift.")
    ap.add_argument("--out", default=os.path.join(TE, "var_backtest.json"))
    args = ap.parse_args()

    k_list = [16, 32, 64, 128, 512, 1024, 2048] if args.k_sweep else []
    splits = ("test", "val") if args.val_control else ("test",)
    jobs = build_jobs(k_list, LEVELS, args.data_dir, args.seq_tag, splits)

    print("=" * 78)
    print(f"Intraday portfolio VaR backtest — H={M.H} bars x {BAR_SECONDS}s = "
          f"{M.H * BAR_SECONDS // 60} min, equal-weight d=8")
    print("=" * 78)
    print(f"data_dir {args.data_dir}")
    print(f"queries  true_S_test_{args.seq_tag}.npy   splits {splits}")
    print(f"{len(jobs)} cells, {args.workers} workers\n")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(backtest_one, jobs))
    el = time.perf_counter() - t0

    base = [r for r in res if r["k"] in (M.K_NEIGH, 0) and r["split"] == "test"]
    print(f"{'bank':22s} {'seed':>5s} {'1%':>7s} {'5%':>7s} {'10%':>7s} "
          f"{'kup_p@5%':>9s} {'ES@5%':>7s} {'shrink':>7s}")
    for r in sorted(base, key=lambda z: (z["label"], z["seed"])):
        L = r["levels"]
        sh = r["diag"].get("shrinkage")
        print(f"{r['label']:22s} {r['seed']:5d} "
              f"{L['alpha_0.01']['rate']*100:6.2f}% {L['alpha_0.05']['rate']*100:6.2f}% "
              f"{L['alpha_0.10']['rate']*100:6.2f}% {L['alpha_0.05']['kupiec_p']:9.2e} "
              f"{L['alpha_0.05']['es_ratio']:7.3f} "
              f"{(f'{sh:.3f}' if sh is not None else '-'):>7s}")

    out = dict(
        units="exception rate is a fraction of queries; es_ratio is dimensionless",
        horizon=dict(bars=M.H, bar_seconds=BAR_SECONDS,
                     minutes=M.H * BAR_SECONDS / 60,
                     note="intraday liquidation horizon, NOT an FRTB 10-day capital "
                          "figure; no sqrt(t) scaling is applied to any reported number"),
        protocol=dict(s_idx=M.S_IDX, k_neighbours=M.K_NEIGH, levels=list(LEVELS),
                      portfolio="equal weight, terminal H-bar log-return",
                      block_windows=BLOCK_WINDOWS, ewma_lambda=EWMA_LAMBDA,
                      independence_test="Christoffersen first-order Markov, transitions "
                                        "within contiguous 256-window blocks only; valid "
                                        "because stride == seq_len (windows are disjoint)"),
        # data_dir and queries are recorded so table_d.py can assert they match the
        # CRPS artefact. Table C and Table D claim to read the SAME ensemble; if
        # these two strings ever diverge that claim is false and the tables must
        # not be printed side by side.
        data_dir=args.data_dir,
        queries=f"true_S_test_{args.seq_tag}.npy",
        realised_sd=next((r["realised_sd"] for r in res if r["split"] == "test"), None),
        elapsed_sec=round(el, 1),
        results=res,
    )
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n{len(jobs)} cells in {el:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
