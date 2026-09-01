"""Paper Table-4 statistics on the paper's OWN dataset (§5.1 Heston, d=2).

Why this file exists
--------------------
GUIDELINE §15.2 Section 6 asks for a three-column comparison:

    Paper (Table 4, S&P 500) | Ours - paper dataset | Ours - benchmark Heston

``heston_paper_metrics.py`` produces the third column (benchmark Heston, 5 seeds,
8192x128 price paths). This driver produces the **second**: the same six Table-4
statistics, same estimator code, evaluated on the paper's own §5.1 Heston dataset
(5000 x 253 x 2, dt = 1/252, T = 1 year) and on the generated pools from the
paper-reimplementation sweep.

Without this the middle column would have to be hand-typed, which §15.2 forbids
("Never hand-type a paper-metric value, compute it").

What is scored
--------------
* **Real** — ``dataset/X_heston_paper_5000x253x2.npy``, price channel.
* **SBBTS** — the four ``beta=100`` base-arm replicates ``t00 / t00s1 / t00s2 /
  t00s3`` (trial index 0 of ``sweep_paper.py:TRIALS``, seeds 0-3). Reported as
  mean +- std across the four arms, so the spread is the seed spread at the
  shipped setting.
* **SBTS** — ``sbtsh0p05``, the h = 0.05 kernel baseline. One arm, hence no std.
  h = 0.05 is the author-mandated bandwidth for this benchmark.

Only the price channel (index 0) is used: every Table-4 entry is a functional of
the return series alone, so the variance leg is irrelevant here. That is exactly
why Table 4 works as the cross-dataset anchor while the §5.1 MLE metric does not.

``PERIODS_PER_YEAR = 252`` here, NOT the 250 used by ``heston_paper_metrics.py``.
The two datasets have different clocks: the paper's §5.1 Heston is generated with
N = 252 steps over T = 1 year (``dataset/generate_paper_heston.py``), while the
benchmark Heston uses ``DT = 1/250`` (``dataset/Heston/generate_heston.py:12``).
Each is annualised on its own clock. VaR/ES rows are unaffected by this choice.

Path filtering
--------------
A generated path is dropped if it is non-finite or hits a non-positive price
(``log`` undefined). Same rule as ``rebuild_corr_cache.py``. The count kept is
recorded per arm in the JSON so the filtering is auditable.

Reads
-----
    methods/SBBTS/paper_reimplementation/dataset/X_heston_paper_5000x253x2.npy
    methods/SBBTS/paper_reimplementation/results/sweep/X_sbbts_1000x252x2_*.npy
    methods/SBBTS/paper_reimplementation/results/sweep/X_sbts_1000x252x2_sbtsh0p05.npy

Writes
------
    methods/SBBTS/paper_reimplementation/results/paper_dataset_table4.json

Usage
-----
    /home/tbasseras/gpu-venv/bin/python \
        methods/SBBTS/paper_reimplementation/metric/paper_dataset_table4.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_REIMPL = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import heston_paper_metrics as hpm  # noqa: E402  (needs sys.path above)

# The paper's §5.1 Heston is simulated with N = 252 steps over T = 1 year.
hpm.PERIODS_PER_YEAR = 252.0

DATASET = os.path.join(
    PAPER_REIMPL, "dataset", "X_heston_paper_5000x253x2.npy")
SWEEP = os.path.join(PAPER_REIMPL, "results", "sweep")
OUT = os.path.join(PAPER_REIMPL, "results", "paper_dataset_table4.json")

# beta = 100 base arm, seeds 0-3 (trial index 0 of sweep_paper.py:TRIALS).
SBBTS_ARMS = ["t00", "t00s1", "t00s2", "t00s3"]
SBTS_ARM = "sbtsh0p05"          # author-mandated h = 0.05

ROWS = ["VaR99_pct", "VaR95_pct", "ES99_pct",
        "ES95_pct", "AnnRet_pct", "AnnStd_pct"]


def usable_prices(arr):
    """(M, T, 2) -> (M', T) price channel, dropping unusable paths."""
    prices = arr[:, :, 0].astype(np.float64)
    keep = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    return prices[keep], int(keep.sum()), int(keep.size)


def main():
    t0 = time.time()
    real_raw = np.load(DATASET)
    real, n_keep, n_tot = usable_prices(real_raw)
    print(f"real   {real.shape}  ({n_keep}/{n_tot} paths usable)")
    real_stats = hpm.table4_stats(real)

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "paper §5.1 Heston (arXiv:2604.07159), d=2, T=1y, N=252",
        "periods_per_year": hpm.PERIODS_PER_YEAR,
        "periods_per_year_note": (
            "252 here because the paper's §5.1 Heston uses N=252 steps over T=1; "
            "heston_paper_metrics.py uses 250 because the benchmark Heston uses "
            "DT=1/250. VaR/ES rows are unaffected."),
        "channel_note": "price channel (index 0) only; Table 4 is a functional of returns",
        "paper_table4_sp500": hpm.PAPER_TABLE4,
        "real": {"path": os.path.relpath(DATASET, PAPER_REIMPL),
                 "n_paths": n_keep, **real_stats},
        "methods": {},
    }

    for method, arms, pattern in (
            ("SBBTS", SBBTS_ARMS, "X_sbbts_1000x252x2_%s.npy"),
            ("SBTS", [SBTS_ARM], "X_sbts_1000x252x2_%s.npy")):
        per_arm = {}
        for tag in arms:
            path = os.path.join(SWEEP, pattern % tag)
            if not os.path.isfile(path):
                print(f"  MISSING {path} -- skipped")
                continue
            gen, k, t = usable_prices(np.load(path))
            if k < 50:
                print(f"  {tag}: only {k} usable paths -- skipped")
                continue
            per_arm[tag] = {"n_paths": k, "n_dropped": t - k,
                            **hpm.table4_stats(gen)}
            print(f"  {method:6s} {tag:7s} ({k}/{t} usable)")
        agg = {}
        for key in (per_arm[next(iter(per_arm))] if per_arm else []):
            if key in ("n_paths", "n_dropped"):
                continue
            vals = np.array([a[key] for a in per_arm.values()], dtype=np.float64)
            agg[key] = {"mean": float(vals.mean()),
                        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                        "n": int(vals.size)}
        out["methods"][method] = {"arms": per_arm, "aggregate": agg}

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {OUT}  ({time.time() - t0:.1f}s)")

    # -- readable board, pooled aggregation (what the README quotes) --
    print("\n| row | Real | SBBTS (n=%d) | SBTS |"
          % out["methods"]["SBBTS"]["aggregate"]["pooled/VaR99_pct"]["n"])
    for row in ROWS:
        k = "pooled/" + row
        b = out["methods"]["SBBTS"]["aggregate"][k]
        s = out["methods"]["SBTS"]["aggregate"][k]
        print(f"| {row:11s} | {out['real'][k]:8.3f} | "
              f"{b['mean']:8.3f} +- {b['std']:.3f} | {s['mean']:8.3f} |")


if __name__ == "__main__":
    main()
