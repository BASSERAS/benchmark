#!/usr/bin/env python3
"""
recompute_grid_tvd.py
─────────────────────
Deterministic (numpy-only, no GPU) recompute of the grid_tvd visual sanity-check
metric for a method's 5 seeds, and re-aggregation into the mean ± std summary
shown in the READMEs.

grid_tvd is the 2D-histogram Total Variation Distance (%) between the real and
generated (time, value) path clouds — exactly what the first two diagnostic
panels scatter. It is a VISUAL side-check, not part of the A/B rankings. The grid
resolution is locked at metrics.GRID_TVD_DEFAULT_BINS (50×50) for every method so
all curves stay comparable.

It:
  1. loads the real Heston S TEST matrix and each seed's generated paths,
  2. calls metrics.grid_tvd on the flattened (t, x) clouds,
  3. writes the "grid_tvd" scalar into each seed_i_metrics.json,
  4. writes results/<dataset>/<method>/grid_tvd_aggregate.json
     ({mean, std, per_seed, n_bins}).

The real reference is the held-out TEST set (seed 1). For method ==
"perfect_recovery" the "generated" side is the materialised independent Heston
draw for that seed (written by compute_perfect_recovery.py).

Usage
─────
    python metrics/recompute_grid_tvd.py --method TimeGAN
    python metrics/recompute_grid_tvd.py --method perfect_recovery
"""
import argparse, json, os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO       = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from metrics import grid_tvd, paths_to_points, GRID_TVD_DEFAULT_BINS

DATASET = "Heston"


def load_real():
    d = os.path.join(REPO, "dataset", DATASET)
    return np.load(os.path.join(d, "heston_S_test_8192x128.npy"))   # TEST set (seed 1)


def load_generated(method, seed):
    if method == "perfect_recovery":
        p = os.path.join(REPO, "methods", "perfect_recovery", "dataset",
                         "seeds", f"heston_S_independent_seed{seed}.npy")
        return np.load(p)
    p = os.path.join(REPO, "methods", method, "generated_paths",
                     f"seed_{seed}", "generated_paths_8192x128.npy")
    return np.load(p)


def results_dir(method):
    if method == "perfect_recovery":
        return os.path.join(REPO, "methods", "perfect_recovery", "results")
    return os.path.join(REPO, "results", DATASET, method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    S_real = load_real()
    real_pts = paths_to_points(S_real)
    rdir = results_dir(args.method)
    per_seed = []

    for seed in range(args.seeds):
        fake = load_generated(args.method, seed)
        tvd, _, _, _ = grid_tvd(real_pts, paths_to_points(fake),
                                n_bins=GRID_TVD_DEFAULT_BINS)
        per_seed.append(float(tvd))

        jpath = os.path.join(rdir, f"seed_{seed}_metrics.json")
        with open(jpath) as f:
            doc = json.load(f)
        doc["grid_tvd"] = float(tvd)
        with open(jpath, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"  updated {os.path.relpath(jpath, REPO)}  grid_tvd={tvd:.4f}")

    arr = np.array(per_seed)
    agg = {
        "name": "grid_tvd (path-cloud 2D-histogram TVD, %)",
        "n_bins": list(GRID_TVD_DEFAULT_BINS),
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "per_seed": per_seed,
    }
    dump = os.path.join(rdir, "grid_tvd_aggregate.json")
    with open(dump, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n===== {args.method}  grid_tvd @ "
          f"{GRID_TVD_DEFAULT_BINS[0]}×{GRID_TVD_DEFAULT_BINS[1]} =====")
    print(f"  mean ± std = {arr.mean():.4f} ± {arr.std():.4f}  "
          f"seeds=[{', '.join(f'{x:.4f}' for x in per_seed)}]")
    print(f"  wrote {os.path.relpath(dump, REPO)}")


if __name__ == "__main__":
    main()
