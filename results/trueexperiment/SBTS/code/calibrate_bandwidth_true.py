#!/usr/bin/env python3
"""
(h, K) calibration for SBTS on dataset/TrueDataset (real 30s crypto, d = 8).

The criterion is NOT the one used for the d = 8 Heston panel, and the reason is
not convenience.  It is written down in full, and pre-registered, in
    results/TrueDataset/SBTS/losses/selection_criterion.md
Read that file before changing anything here.

WHY THE HESTON CRITERION IS NOT REUSED
--------------------------------------
Heston gates on absolute tolerances against a reference split: vol error < 5%,
excess-kurtosis error < 20%.  That is well posed there because the five Heston
splits are five INDEPENDENT DRAWS FROM ONE KNOWN SDE, so train and val agree to
within sampling noise.

There is only ONE history of the real market.  The five TrueDataset splits are
carved out of it, so they differ by genuine regime variation.  Measured:

    scored against val        vol err     excess-kurtosis err
    train                       7.05%              33.10%
    test                        9.84%             348.26%
    disc                        8.47%            1012.81%
    valdisc                    11.73%              53.84%

A PERFECT generator -- one emitting the training distribution exactly -- scores
7.05% / 33.10% and is rejected by both Heston constraints.  The criterion is not
merely hard to satisfy on real data, it is unsatisfiable by anything including
the data itself.  Reporting "SBTS fails" under it would be an artefact.

THE REPLACEMENT: A REAL-vs-REAL ENVELOPE, ZERO FREE PARAMETERS
--------------------------------------------------------------
Every threshold is the extreme value attained by genuine market data scored
against genuine market data, over the 20 ordered pairs of the five splits.  It
is recomputed at runtime from whatever --data-dir is given, which is REQUIRED:
the dataset-variant sweep (history length, split mode, train size) changes the
splits, and therefore changes the envelope.  Hardcoding it would silently apply
one variant's envelope to another.

  Hard constraints -- a violation disqualifies whatever else the row says:
    1. no path collapse: min(S_gen) > MIN_PRICE, all prices finite and positive;
    2. vol_err        <= max real-vs-real vol_err   (18.06% on the committed build);
    3. corr_err       <= max real-vs-real corr_err  (0.1173 on the committed build).

  Excess kurtosis is REPORTED BUT GATES NOTHING.  Its real-vs-real spread is
  29.7% to 1012.8%, a factor of 34.  A statistic that varies by 34x between two
  honest samples of the same market carries no information about generator
  quality at this sample size; gating on it would be selecting on noise.

  Objective among survivors: MINIMISE |log NNratio|, i.e. drive
      NNratio = median NN(gen -> train) / median NN(val -> train)
  toward 1 in flattened log-return space.  This diagnostic comes out unusually
  well calibrated here -- the four genuinely held-out real splits span only
  0.980 to 1.050.  NNratio << 1 is memorisation; NNratio >> 1 is the opposite
  failure, paths that have left the data manifold.  The Heston file says
  "maximise"; that was a proxy for "push toward 1" valid because every Heston row
  sat below 1 with kurtosis bounding it from above.  With kurtosis demoted an
  unbounded maximise would reward off-manifold paths.  The two rules COINCIDE
  whenever NNratio < 1.

  Tie-break (|log NNratio| within 0.02): lowest corr_err.

WHY K IS SWEPT AND NOT HELD AT THE AUTHOR'S VALUE
-------------------------------------------------
The Heston runs fix K = 20, N_pi = 50.  On real data K = 20 is measurably
pathological.  Post-overflow-fix effective sample size ESS = (sum w)^2/sum w^2
out of 8192, zero-weight fraction 0.0000 everywhere:

    h \\ K      1       3       5      10      20
    0.16     3884    1433     557     245    2.91
    0.22     5456    3084    1952     967     332
    0.31     6663    4817    3785    2402    1231
    0.45     7449    6330    5546    4233    2771

At h = 0.16, K = 20 the drift is conditioned on THREE training paths out of
8192 -- the memorisation regime by construction.  Holding K fixed would confound
"SBTS is unsuited to real data" with "this hyperparameter was never re-tuned off
Heston".  N_pi = 50 stays fixed.

SPLIT DISCIPLINE
----------------
(h, K) is scored on the VALIDATION split (true_S_val), never on test.  Test is
the reference for every reported metric; tuning on it would leak.

Usage
-----
    setsid nohup taskset -c 0-15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
        /home/tbasseras/sbts-venv/bin/python calibrate_bandwidth_true.py \\
        > ../losses/calibrate.log 2>&1 < /dev/null & disown
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SBTS = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(SBTS, "../../.."))
DATA = os.path.join(REPO, "dataset", "TrueDataset")
sys.path.insert(0, HERE)

from sbts_generate_true import generate_paths, warmup_jit, DT, S0  # noqa: E402

MIN_PRICE = 10.0
SPLIT_SUFFIXES = ["", "_test", "_disc", "_val", "_valdisc"]
SPLIT_LABELS = ["train", "test", "disc", "val", "valdisc"]

BARS_PER_YEAR = None   # filled from dataset_stats.json in main()


def logret(S):
    return np.diff(np.log(S), axis=1)


def ann_vol(S):
    """Per-asset annualised volatility, using the dataset's real bar rate."""
    return logret(S).std(axis=(0, 1)) * np.sqrt(BARS_PER_YEAR)


def excess_kurt(S):
    r = logret(S)
    out = np.empty(r.shape[2])
    for j in range(r.shape[2]):
        x = r[:, :, j].ravel()
        out[j] = ((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3.0
    return out


def cross_corr(S):
    """Off-diagonal realised correlations of the log-returns, flattened."""
    C = np.corrcoef(logret(S).reshape(-1, S.shape[2]), rowvar=False)
    return C[np.triu_indices_from(C, k=1)]


def min_nn(query, ref, chunk=256):
    """L2 distance from each row of `query` to its nearest row in `ref`."""
    q = np.ascontiguousarray(query, dtype=np.float32)
    r = np.ascontiguousarray(ref, dtype=np.float32)
    rn = (r ** 2).sum(1)
    out = np.empty(len(q), dtype=np.float64)
    for i in range(0, len(q), chunk):
        blk = q[i:i + chunk]
        d2 = (blk ** 2).sum(1)[:, None] + rn[None, :] - 2.0 * (blk @ r.T)
        out[i:i + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def rel_err(a, ref):
    """Mean absolute relative error in percent."""
    return float(np.mean(np.abs(a - ref) / np.abs(ref)) * 100.0)


def real_vs_real_envelope(data_dir, score_suffix="_val", seq_tag="8192x128x8"):
    """Thresholds derived from the data itself. See selection_criterion.md.

    The ceilings are the WORST value attained by genuine market data scored
    against genuine market data, so a generated sample that clears them is
    indistinguishable, on that statistic, from another honest slice of the
    same market. Recomputed per dataset variant -- never hardcode these.
    """
    S = {}
    for suf in SPLIT_SUFFIXES:
        p = os.path.join(data_dir, f"true_S{suf}_{seq_tag}.npy")
        if os.path.exists(p):
            S[suf] = np.load(p)
    have = list(S)
    V = {k: ann_vol(v) for k, v in S.items()}
    K_ = {k: excess_kurt(v) for k, v in S.items()}
    C = {k: cross_corr(v) for k, v in S.items()}

    pairs = list(itertools.permutations(have, 2))
    vol_errs = {(a, b): rel_err(V[a], V[b]) for a, b in pairs}
    kurt_errs = {(a, b): rel_err(K_[a], K_[b]) for a, b in pairs}
    corr_errs = {(a, b): float(np.mean(np.abs(C[a] - C[b]))) for a, b in pairs}

    flat = {k: logret(v).reshape(len(v), -1) for k, v in S.items()}
    den = float(np.median(min_nn(flat[score_suffix], flat[""])))
    nn_band = {}
    for suf in have:
        if suf == "":
            continue
        nn_band[SPLIT_LABELS[SPLIT_SUFFIXES.index(suf)]] = \
            float(np.median(min_nn(flat[suf], flat[""]))) / den

    lbl = lambda s: SPLIT_LABELS[SPLIT_SUFFIXES.index(s)]  # noqa: E731
    return dict(
        splits_found=[lbl(s) for s in have],
        median_nn_heldout=den,
        vol_err_pct_max=max(vol_errs.values()),
        vol_err_pct_min=min(vol_errs.values()),
        vol_err_pct_median=float(np.median(list(vol_errs.values()))),
        kurt_err_pct_max=max(kurt_errs.values()),
        kurt_err_pct_min=min(kurt_errs.values()),
        kurt_err_pct_median=float(np.median(list(kurt_errs.values()))),
        corr_err_max=max(corr_errs.values()),
        corr_err_min=min(corr_errs.values()),
        corr_err_median=float(np.median(list(corr_errs.values()))),
        nn_ratio_by_split=nn_band,
        nn_ratio_band=[min(nn_band.values()), max(nn_band.values())],
        train_vs_score=dict(vol_err_pct=vol_errs[("", score_suffix)],
                            kurt_err_pct=kurt_errs[("", score_suffix)],
                            corr_err=corr_errs[("", score_suffix)]),
    )


def main():
    global BARS_PER_YEAR

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="0.10,0.16,0.22,0.31,0.45,0.65",
                    help="bandwidths h")
    ap.add_argument("--K-grid", default="1,3,5,10,20",
                    help="K values; 20 is the SBTS author's, kept as a column")
    ap.add_argument("--N-pi", type=int, default=50)
    ap.add_argument("--m-probe", type=int, default=512,
                    help="paths per (h,K); the winner is re-verified at 8192")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3, help="validation seed, matching Heston")
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--seq-tag", default="8192x128x8")
    ap.add_argument("--train", default=None)
    ap.add_argument("--score", default=None)
    ap.add_argument("--out", default=os.path.join(SBTS, "losses", "bandwidth_selection.json"))
    args = ap.parse_args()

    train_name = args.train or f"true_S_{args.seq_tag}.npy"
    score_name = args.score or f"true_S_val_{args.seq_tag}.npy"

    with open(os.path.join(args.data_dir, "dataset_stats.json")) as fh:
        BARS_PER_YEAR = json.load(fh)["bars_per_year"]

    S_train = np.load(os.path.join(args.data_dir, train_name))
    S_val = np.load(os.path.join(args.data_dir, score_name))
    d = S_train.shape[2]

    vol_ref = ann_vol(S_val)
    kurt_ref = excess_kurt(S_val)
    corr_ref = cross_corr(S_val)

    env = real_vs_real_envelope(args.data_dir, seq_tag=args.seq_tag)
    VOL_MAX = env["vol_err_pct_max"]
    CORR_MAX = env["corr_err_max"]
    NN_LO, NN_HI = env["nn_ratio_band"]

    train_flat = logret(S_train).reshape(len(S_train), -1)
    med_base = env["median_nn_heldout"]

    hs = [float(x) for x in args.grid.split(",")]
    Ks = [int(x) for x in args.K_grid.split(",")]

    print("=" * 84)
    print(f"SBTS (h, K) calibration — TrueDataset d={d}, T={S_train.shape[1]}")
    print("=" * 84)
    print(f"train {S_train.shape}  score-on {score_name}  m_probe={args.m_probe}  "
          f"N_pi={args.N_pi}  workers={args.workers}")
    print(f"grid: h={hs}  K={Ks}  -> {len(hs)*len(Ks)} runs")
    print(f"reference (validation): ann.vol   {np.round(vol_ref, 3)}")
    print(f"                        exc.kurt  {np.round(kurt_ref, 1)}")
    print(f"                        mean cross-corr {corr_ref.mean():.4f}")
    print()
    print("--- real-vs-real envelope (derived from this data-dir, not hardcoded) ---")
    print(f"  vol_err%   min {env['vol_err_pct_min']:7.2f}  median {env['vol_err_pct_median']:7.2f}"
          f"  CEILING {VOL_MAX:7.2f}")
    print(f"  kurt_err%  min {env['kurt_err_pct_min']:7.2f}  median {env['kurt_err_pct_median']:7.2f}"
          f"  max {env['kurt_err_pct_max']:7.2f}   (REPORTED, NOT GATED)")
    print(f"  corr_err   min {env['corr_err_min']:.4f}  median {env['corr_err_median']:.4f}"
          f"  CEILING {CORR_MAX:.4f}")
    print(f"  NNratio band {NN_LO:.4f} .. {NN_HI:.4f}   {env['nn_ratio_by_split']}")
    print(f"  a PERFECT generator (= train) would score vol {env['train_vs_score']['vol_err_pct']:.2f}%"
          f"  kurt {env['train_vs_score']['kurt_err_pct']:.2f}%")
    print(f"  median NN(val -> train) = {med_base:.6f}   <- NNratio denominator\n")

    warmup_jit(d)
    rows = []
    for h, K in itertools.product(hs, Ks):
        t0 = time.perf_counter()
        S_gen, _ = generate_paths(S_train, M_simu=args.m_probe, h=h, K=K,
                                  N_pi=args.N_pi, n_workers=args.workers, seed=args.seed)
        elapsed = time.perf_counter() - t0

        finite = bool(np.all(np.isfinite(S_gen)) and np.all(S_gen > 0.0))
        smin = float(S_gen.min()) if finite else 0.0
        vol_gen = ann_vol(S_gen)
        vol_err = rel_err(vol_gen, vol_ref)
        # signed: < 1 is over-smoothing (compression), > 1 is under-smoothing (blow-up).
        # vol_err is an absolute error and is only monotone in h while this stays below 1.
        vol_ratio = float(np.mean(vol_gen / vol_ref))
        kurt_err = rel_err(excess_kurt(S_gen), kurt_ref)
        corr_err = float(np.mean(np.abs(cross_corr(S_gen) - corr_ref)))
        nn_ratio = float(np.median(min_nn(logret(S_gen).reshape(len(S_gen), -1),
                                          train_flat)) / med_base)
        log_nn = abs(np.log(nn_ratio)) if nn_ratio > 0 else float("inf")

        admissible = bool(finite and smin > MIN_PRICE
                          and vol_err <= VOL_MAX and corr_err <= CORR_MAX)
        rows.append(dict(h=h, K=K, N_pi=args.N_pi, m_probe=args.m_probe,
                         finite=finite, min_price=smin, vol_err_pct=vol_err,
                         vol_ratio=vol_ratio,
                         kurt_err_pct=kurt_err, corr_err=corr_err,
                         nn_ratio=nn_ratio, abs_log_nn_ratio=log_nn,
                         in_nn_band=bool(NN_LO <= nn_ratio <= NN_HI),
                         admissible=admissible, elapsed_sec=round(elapsed, 1)))
        print(f"  h={h:<5} K={K:<3} vol={vol_err:6.2f}% (x{vol_ratio:.3f})  kurt={kurt_err:8.2f}%  "
              f"corr={corr_err:.4f}  minS={smin:8.2f}  NN={nn_ratio:.4f} "
              f"|log|={log_nn:.3f}  {'OK ' if admissible else 'REJ'}"
              f"{' <BAND>' if NN_LO <= nn_ratio <= NN_HI else ''}  [{elapsed/60:.1f} min]",
              flush=True)

        with open(args.out + ".partial", "w") as fh:
            json.dump(rows, fh, indent=2)

    ok = [r for r in rows if r["admissible"]]
    if ok:
        best_nn = min(r["abs_log_nn_ratio"] for r in ok)
        tied = [r for r in ok if r["abs_log_nn_ratio"] - best_nn <= 0.02]
        best = min(tied, key=lambda r: r["corr_err"])
    else:
        best = None

    out = dict(
        dataset="TrueDataset", d=d, T=int(S_train.shape[1]), dt_gauge=DT, S0=S0,
        bars_per_year=BARS_PER_YEAR,
        criterion_source="results/TrueDataset/SBTS/losses/selection_criterion.md",
        criterion_note=("Heston's absolute tolerances are unsatisfiable on real data: "
                        "train-vs-val alone is 7.05% vol / 33.10% kurt. Thresholds here are "
                        "the real-vs-real envelope, derived at runtime from --data-dir."),
        real_vs_real_envelope=env,
        hard_constraints=dict(min_price=MIN_PRICE, vol_err_pct_max=VOL_MAX,
                              corr_err_max=CORR_MAX,
                              kurt_err_pct_max=None),
        objective="minimise |log nn_ratio| among admissible; tie-break (0.02) lowest corr_err",
        nn_ratio_band=[NN_LO, NN_HI],
        scored_on=score_name, trained_on=train_name, data_dir=args.data_dir,
        median_nn_heldout=med_base,
        reference=dict(ann_vol=vol_ref.tolist(), excess_kurt=kurt_ref.tolist(),
                       mean_cross_corr=float(corr_ref.mean())),
        selected=best, sweep=rows,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print()
    if best is None:
        print("NO ADMISSIBLE (h, K) — report the failure honestly, do not relax the envelope.")
        closest = min(rows, key=lambda r: r["abs_log_nn_ratio"])
        print(f"  closest on the objective anyway: h={closest['h']} K={closest['K']} "
              f"NN={closest['nn_ratio']:.4f} vol={closest['vol_err_pct']:.2f}% "
              f"(ceiling {VOL_MAX:.2f}%)")
    else:
        print(f"selected h = {best['h']}, K = {best['K']}  (NNratio {best['nn_ratio']:.4f}, "
              f"vol_err {best['vol_err_pct']:.2f}%, corr_err {best['corr_err']:.4f})")
    print(f"wrote {args.out}")
    return 0 if best is not None else 1


if __name__ == "__main__":
    sys.exit(main())
