#!/usr/bin/env python3
"""Merge per-seed split-half metric JSONs into the same four aggregate artefacts
that compute_all_multiasset.py main() writes at lines 374-424.

The per-seed jobs were run one seed per process to parallelise, so nobody ever
held the full `all_results` list in memory. This rebuilds that list from the
seed JSONs and replays the identical aggregation -- including the SAME imported
aggregate_curve_metrics -- so the numbers are what a single 5-seed process would
have produced. Nothing here recomputes a metric.

Writes only under /tmp/splithalf.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/tbasseras/benchmark/metrics")
from metrics import aggregate_curve_metrics, GRID_TVD_DEFAULT_BINS  # noqa: E402

OUT = "/tmp/splithalf"
SKIP = {"seed", "compute_time_sec", "per_asset", "native_keys", "d"}


def merge(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "s*", "seed_*_metrics.json")))
    if not files:
        return None
    all_results = sorted((json.load(open(f)) for f in files), key=lambda r: r["seed"])
    native = set(all_results[0].get("native_keys") or [])
    keys = [k for k in all_results[0] if k not in SKIP]

    # ---- metrics_summary.csv (compute_all_multiasset.py:377-390) --------------
    summary = os.path.join(results_dir, "metrics_summary.csv")
    with open(summary, "w", newline="") as fh:
        cols = ["metric", "scope", "mean", "std"] + [f"seed_{r['seed']}" for r in all_results]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k in keys:
            vals = [float(r[k]) for r in all_results if r.get(k) is not None]
            row = {"metric": k,
                   "scope": "native" if k in native else "per-asset-mean",
                   "mean": round(float(np.mean(vals)), 8) if vals else None,
                   "std": round(float(np.std(vals)), 8) if vals else None}
            row.update({f"seed_{r['seed']}": r.get(k) for r in all_results})
            w.writerow(row)

    # ---- metrics_per_asset.csv (compute_all_multiasset.py:392-405) -----------
    pa = os.path.join(results_dir, "metrics_per_asset.csv")
    pkeys = sorted(all_results[0]["per_asset"])
    with open(pa, "w", newline="") as fh:
        cols = ["metric", "asset"] + [f"seed_{r['seed']}" for r in all_results] + ["mean"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k in pkeys:
            for j in range(all_results[0]["d"]):
                vals = [r["per_asset"][k][j] for r in all_results]
                row = {"metric": k, "asset": j, "mean": round(float(np.mean(vals)), 8)}
                row.update({f"seed_{r['seed']}": r["per_asset"][k][j] for r in all_results})
                w.writerow(row)

    # ---- curve_b_aggregate.json (compute_all_multiasset.py:407-413) ----------
    with open(os.path.join(results_dir, "curve_b_aggregate.json"), "w") as fh:
        json.dump(aggregate_curve_metrics(all_results), fh, indent=2)

    # ---- grid_tvd_aggregate.json (compute_all_multiasset.py:415-424) ---------
    g = [float(r["grid_tvd"]) for r in all_results]
    with open(os.path.join(results_dir, "grid_tvd_aggregate.json"), "w") as fh:
        json.dump({"name": "grid_tvd (path-cloud 2D-histogram TVD, %), mean over assets",
                   "n_bins": list(GRID_TVD_DEFAULT_BINS),
                   "mean": float(np.mean(g)), "std": float(np.std(g)),
                   "per_seed": g,
                   "per_seed_per_asset": [r["per_asset"]["grid_tvd"] for r in all_results]},
                  fh, indent=2)
    return [r["seed"] for r in all_results]


if __name__ == "__main__":
    for half in ("A", "B"):
        for m in ("Deep-MKV-TS", "reference", "SBTS", "CSDI", "real_floor"):
            d = os.path.join(OUT, f"out{half}", m)
            seeds = merge(d) if os.path.isdir(d) else None
            print(f"{half}/{m:14s} {'seeds ' + str(seeds) if seeds else 'NO SEED JSONS'}")
