"""SBBTS — the paper's OWN Table 4 metrics, applied to our benchmark Heston paths.

Why this file exists
--------------------
GUIDELINE §15.2 Section 6 requires the results README to compare against **the
paper's own metrics**, and requires the "Ours — Heston" column to come from a
committed JSON produced by a reusable driver in ``paper_reimplementation/metric/``
— never hand-typed.

For SBBTS that anchor is **Table 4** (Appendix C.2.1), the paper's only
quantitative *generative-quality* table:

    Table 4: Comparison of tail-risk metrics (VaR, ES) and annualized performance
             statistics averaged across instruments.

                     Real   SBBTS   SBTS
        VaR99% (%)   3.60    3.57   3.49
        VaR95% (%)   2.11    2.19   2.17
        ES99%  (%)   4.65    4.44   4.36
        ES95%  (%)   3.15    3.18   3.07
        Ann. Ret (%) 19.02  16.31  14.68
        Ann. Std (%) 24.22  24.38  23.06

Two properties make Table 4 the right anchor, and they are the reason this driver
replaces the d=2 rerun that was previously scoped for Section 6:

1. **It is computable from univariate price paths.** Every entry is a functional of
   the daily return series alone. The paper's §5.1 Heston metric (per-path bivariate
   MLE of κ, θ, ξ, ρ, r) is NOT — it reads ``S = X[:,0]; v = X[:,1]`` and needs the
   variance leg, so on price-only paths ρ and ξ are unidentifiable. That metric
   therefore stays where it belongs, in ``reproduce_heston.py`` on the paper's own
   d=2 dataset. Section 6 of the results README uses Table 4 instead, and no extra
   GPU time is spent.
2. **It already contains an SBTS column**, so our table can reproduce the paper's
   own three-way Real / SBBTS / SBTS layout using the benchmark's SBTS run.

⚠️ What is and is not comparable
--------------------------------
The paper's Table 4 numbers are computed on **real S&P 500 instruments** (daily
returns, long histories). Ours are computed on **Heston** paths, T=128 steps at
dt=1/250. The levels are therefore NOT expected to match — the Heston generator is
parameterised with μ=0.05, θ=0.04 (long-run vol 20%), so the *real* column here
must land near Ann. Ret ≈ 5% and Ann. Std ≈ 20%, not the paper's 19.02 / 24.22.

What IS comparable, and what the table is for, is the **generated-vs-real gap** and
its **sign**, which is exactly the claim Table 4 makes: SBBTS should sit closer to
Real than SBTS does, and both should slightly under-shoot the annualised return.

Estimator definitions (reconstructed — the paper states no formulas)
-------------------------------------------------------------------
The paper names the statistics but gives no equations. The standard definitions
below are pinned down by the paper's own internal consistency: from Ann. Std =
24.22% the implied daily σ is 24.22/√252 = 1.526%, and a Gaussian VaR99 would be
2.326 × 1.526 = 3.55% against the reported 3.60% (and ES99 4.65% > the Gaussian
4.07%, i.e. a fat left tail). That reproduces Table 4's Real column to within
rounding, which is the evidence for this reconstruction.

Per path i, with log-returns r_i = diff(log S_i) (127 values for T=128):

    VaR_q(i)  = -quantile(r_i, 1 - q)                        × 100
    ES_q(i)   = -mean(r_i[r_i <= quantile(r_i, 1 - q)])      × 100
    AnnRet(i) =  mean(r_i)          × PERIODS_PER_YEAR       × 100
    AnnStd(i) =  std(r_i, ddof=1)   × sqrt(PERIODS_PER_YEAR) × 100

then averaged over paths ("averaged across all instruments", Table 4 caption).

``PERIODS_PER_YEAR = 250`` here, not the paper's 252, because the benchmark Heston
dataset is generated with ``DT = 1.0/250.0`` (``dataset/Heston/generate_heston.py:12``).
Using the dataset's own clock is what makes "annualised" mean anything. The
difference vs 252 is 0.4% on the annualised rows and zero on VaR/ES.

Two aggregations are reported, because one of them is fragile here:

* ``per_path``  — the paper's stated procedure (per instrument, then average).
  ⚠️ With only 127 returns per path, the 1% quantile is the ~1.3th order statistic,
  i.e. essentially the path minimum. That estimator is strongly biased for VaR99 /
  ES99. Averaging over 8192 paths kills the *variance* of the mean but not the bias.
* ``pooled``    — all 8192 × 127 returns in one sample (1.04M points), one quantile.
  Stable, but it mixes cross-path dispersion into the tail.

Both are computed **identically for the real and the generated pools**, so the
gap being reported is fair under either. The README quotes ``pooled`` for VaR/ES
and ``per_path`` for the annualised rows, and shows both.

Reads
-----
    dataset/Heston/heston_S_test_8192x128.npy              float64 (8192, 128), prices
    methods/<M>/generated_paths/seed_{0..4}/generated_paths_8192x128.npy
                                                           float64 (8192, 128), prices

Writes
------
    methods/SBBTS/paper_reimplementation/results/heston_paper_metrics.json

Usage
-----
    /home/tbasseras/gpu-venv/bin/python \
        methods/SBBTS/paper_reimplementation/metric/heston_paper_metrics.py

    # or a subset / different real file:
    ... heston_paper_metrics.py --methods SBBTS --real dataset/Heston/heston_S_8192x128.npy

Missing methods and missing seeds are skipped with a printed warning, so the driver
is safe to run before the SBBTS 5-seed pipeline has finished and to re-run after.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_REIMPL = os.path.dirname(HERE)                       # methods/SBBTS/paper_reimplementation
METHOD_DIR = os.path.dirname(PAPER_REIMPL)                 # methods/SBBTS
METHODS_DIR = os.path.dirname(METHOD_DIR)                  # methods
BENCH = os.path.dirname(METHODS_DIR)                       # benchmark root

PERIODS_PER_YEAR = 250.0        # dataset/Heston/generate_heston.py:12 -> DT = 1/250
LEVELS = (0.99, 0.95)

# Paper Table 4 (Appendix C.2.1), transcribed from the PDF. Real S&P 500 instruments.
PAPER_TABLE4 = {
    "VaR99_pct":  {"Real": 3.60,  "SBBTS": 3.57,  "SBTS": 3.49},
    "VaR95_pct":  {"Real": 2.11,  "SBBTS": 2.19,  "SBTS": 2.17},
    "ES99_pct":   {"Real": 4.65,  "SBBTS": 4.44,  "SBTS": 4.36},
    "ES95_pct":   {"Real": 3.15,  "SBBTS": 3.18,  "SBTS": 3.07},
    "AnnRet_pct": {"Real": 19.02, "SBBTS": 16.31, "SBTS": 14.68},
    "AnnStd_pct": {"Real": 24.22, "SBBTS": 24.38, "SBTS": 23.06},
}


def log_returns(prices: np.ndarray) -> np.ndarray:
    """(N, T) prices -> (N, T-1) log-returns. Raises on non-positive prices."""
    if prices.ndim != 2:
        raise ValueError(f"expected (N, T) price array, got shape {prices.shape}")
    if not np.all(np.isfinite(prices)):
        raise ValueError("price array contains non-finite values")
    if prices.min() <= 0:
        raise ValueError(f"non-positive price {prices.min()}, log-return undefined")
    return np.diff(np.log(prices), axis=1)


def _var_es(r: np.ndarray, q: float, axis):
    """Return (VaR, ES) as POSITIVE loss fractions at confidence ``q``.

    ``axis=1``  -> per-path vectors;  ``axis=None`` -> scalars over the pooled sample.
    ES is the mean of the returns at or below the (1-q) quantile.
    """
    if axis is None:
        cut = np.quantile(r, 1.0 - q)
        return -cut, -r[r <= cut].mean()
    cut = np.quantile(r, 1.0 - q, axis=axis, keepdims=True)
    tail_mask = r <= cut
    # Row-wise mean over a ragged mask: sum/count. count >= 1 is guaranteed because
    # the quantile always lies at or above at least one observation of the row.
    cnt = tail_mask.sum(axis=1)
    es = -(np.where(tail_mask, r, 0.0).sum(axis=1) / np.maximum(cnt, 1))
    return -np.squeeze(cut, axis=axis), es


def table4_stats(prices: np.ndarray) -> dict:
    """Compute the six Table-4 rows, under both aggregations, in percent."""
    r = log_returns(prices)
    out = {}

    # -- per_path: statistic per instrument, then average (the paper's wording) --
    for q in LEVELS:
        var, es = _var_es(r, q, axis=1)
        tag = str(int(round(q * 100)))
        out["per_path/VaR%s_pct" % tag] = float(var.mean() * 100.0)
        out["per_path/ES%s_pct" % tag] = float(es.mean() * 100.0)
    out["per_path/AnnRet_pct"] = float(
        (r.mean(axis=1) * PERIODS_PER_YEAR).mean() * 100.0)
    out["per_path/AnnStd_pct"] = float(
        (r.std(axis=1, ddof=1) * np.sqrt(PERIODS_PER_YEAR)).mean() * 100.0)

    # -- pooled: one sample of N*(T-1) returns --
    flat = r.ravel()
    for q in LEVELS:
        var, es = _var_es(flat, q, axis=None)
        tag = str(int(round(q * 100)))
        out["pooled/VaR%s_pct" % tag] = float(var * 100.0)
        out["pooled/ES%s_pct" % tag] = float(es * 100.0)
    out["pooled/AnnRet_pct"] = float(flat.mean() * PERIODS_PER_YEAR * 100.0)
    out["pooled/AnnStd_pct"] = float(flat.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR) * 100.0)

    out["n_paths"] = int(prices.shape[0])
    out["n_returns_per_path"] = int(r.shape[1])
    return out


def mean_std(dicts, key):
    """Mean and sample std of ``key`` across per-seed dicts (std = 0.0 if n == 1)."""
    vals = np.array([d[key] for d in dicts], dtype=float)
    return float(vals.mean()), (float(vals.std(ddof=1)) if vals.size > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real", default="dataset/Heston/heston_S_test_8192x128.npy",
                    help="real price file, relative to the benchmark root")
    ap.add_argument("--methods", default="SBBTS,SBTS",
                    help="comma-separated method names under methods/")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out",
                    default=os.path.join(PAPER_REIMPL, "results",
                                         "heston_paper_metrics.json"))
    args = ap.parse_args()

    t0 = time.time()
    real_path = os.path.join(BENCH, args.real)
    real = np.load(real_path)
    print("real  %s  shape=%s  range=[%.2f, %.2f]"
          % (args.real, real.shape, real.min(), real.max()))
    real_stats = table4_stats(real)

    per_method = {}
    for m in [s.strip() for s in args.methods.split(",") if s.strip()]:
        seeds_out = []
        for s in range(args.seeds):
            p = os.path.join(METHODS_DIR, m, "generated_paths", "seed_%d" % s,
                             "generated_paths_8192x128.npy")
            if not os.path.exists(p):
                print("  [skip] %s seed %d: no %s" % (m, s, os.path.relpath(p, BENCH)))
                continue
            g = np.load(p)
            st = table4_stats(g)
            st["seed"] = s
            st["path"] = os.path.relpath(p, BENCH)
            seeds_out.append(st)
            print("  %s seed %d: pooled VaR99=%.3f%%  AnnStd=%.3f%%"
                  % (m, s, st["pooled/VaR99_pct"], st["pooled/AnnStd_pct"]))
        if not seeds_out:
            print("  [skip] %s: no seeds found, method omitted from the JSON" % m)
            continue
        keys = [k for k in real_stats if k.startswith(("per_path/", "pooled/"))]
        agg = {}
        for k in keys:
            mu, sd = mean_std(seeds_out, k)
            agg[k] = {"mean": mu, "std": sd}
        per_method[m] = {"n_seeds": len(seeds_out), "aggregate": agg,
                         "per_seed": seeds_out}

    doc = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "Heston",
        "seq_len": int(real.shape[1]),
        "periods_per_year": PERIODS_PER_YEAR,
        "periods_per_year_note": (
            "dataset/Heston/generate_heston.py:12 uses DT = 1/250; the paper "
            "annualises daily market data with 252. VaR/ES rows are unaffected."),
        "return_convention": "log-returns, diff(log S) along the time axis",
        "metric_source": (
            "SBBTS paper (arXiv:2604.07159) Table 4, Appendix C.2.1 - tail-risk "
            "(VaR, ES at 95%/99%) + annualised return and std, averaged across "
            "instruments. Formulas are not given in the paper; the standard "
            "definitions used here reproduce the paper's own Real column to "
            "within rounding (see this file's docstring)."),
        "comparability_warning": (
            "Paper Table 4 is computed on real S&P 500 instruments; these numbers "
            "are computed on Heston (mu=0.05, theta=0.04 -> long-run vol 20%), "
            "T=128, dt=1/250. Absolute levels are NOT expected to match. The "
            "comparable quantity is the generated-vs-real gap and its sign."),
        "paper_table4_sp500": PAPER_TABLE4,
        "real": dict({"path": args.real}, **real_stats),
        "methods": per_method,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print("\nWrote %s  (%.1fs)" % (args.out, time.time() - t0))

    # -- console preview in the paper's own Real / SBBTS / SBTS layout --
    rows = [("VaR99% (%)", "VaR99_pct"), ("VaR95% (%)", "VaR95_pct"),
            ("ES99%  (%)", "ES99_pct"), ("ES95%  (%)", "ES95_pct"),
            ("Ann. Ret (%)", "AnnRet_pct"), ("Ann. Std (%)", "AnnStd_pct")]
    names = list(per_method)
    for agg_name in ("pooled", "per_path"):
        print("\n[%s]  %-8s%9s" % (agg_name, "", "Real")
              + "".join("%18s" % n for n in names))
        for label, key in rows:
            line = "%-15s%9.3f" % (label, real_stats["%s/%s" % (agg_name, key)])
            for n in names:
                a = per_method[n]["aggregate"]["%s/%s" % (agg_name, key)]
                line += "%12.3f +-%5.3f" % (a["mean"], a["std"])
            print(line)


if __name__ == "__main__":
    main()
