#!/usr/bin/env python3
"""
recompute_curve_b.py
────────────────────
Deterministic (numpy-only, no GPU) recompute of the B curve-shape metrics for a
method's 5 seeds — now including the percentage-error variant — and re-aggregation
into the two-subline (MSE / % error) summary shown in the READMEs.

It:
  1. loads the real Heston S matrix and each seed's generated paths,
  2. calls metrics.compute_curve_metrics (36 keys: MSE + % per plot),
  3. overwrites the B_* keys in each results/<dataset>/<method>/seed_i_metrics.json,
  4. rebuilds metrics_summary.csv from the (updated) seed JSONs, and
  5. prints the per-plot aggregate table (aggregate_curve_metrics) for the README.

For method == "perfect_recovery" the "generated" side is a row-shuffled copy of
the real dataset (same permutation rng as compute_perfect_recovery.py, seed=i).

Usage
─────
    python metrics/recompute_curve_b.py --method TimeGAN
    python metrics/recompute_curve_b.py --method SBTS
    python metrics/recompute_curve_b.py --method perfect_recovery
"""
import argparse, csv, json, os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO       = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from metrics import compute_curve_metrics, aggregate_curve_metrics, CURVE_PLOTS

DATASET = "Heston"


def load_real():
    d = os.path.join(REPO, "dataset", DATASET)
    return np.load(os.path.join(d, "heston_S_8192x128.npy"))


def load_generated(method, seed, S_real):
    if method == "perfect_recovery":
        rng = np.random.default_rng(seed)
        return S_real[rng.permutation(len(S_real))]
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
    rdir   = results_dir(args.method)
    per_seed = []

    for seed in range(args.seeds):
        fake = load_generated(args.method, seed, S_real)
        curve = compute_curve_metrics(S_real, fake)
        per_seed.append(curve)

        jpath = os.path.join(rdir, f"seed_{seed}_metrics.json")
        with open(jpath) as f:
            doc = json.load(f)
        # drop any stale B_* keys, then write the fresh 36
        for k in [k for k in doc if k.startswith("B_")]:
            del doc[k]
        doc.update(curve)
        with open(jpath, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"  updated {os.path.relpath(jpath, REPO)}")

    # ── rebuild metrics_summary.csv from the (updated) seed JSONs ───────────────
    docs = []
    for seed in range(args.seeds):
        with open(os.path.join(rdir, f"seed_{seed}_metrics.json")) as f:
            docs.append(json.load(f))
    keys = [k for k in docs[0] if k not in ("seed", "compute_time_sec")]
    csv_path = os.path.join(rdir, "metrics_summary.csv")
    fieldnames = ["metric", "mean", "std"] + [f"seed_{i}" for i in range(args.seeds)]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for k in keys:
            vals, per = [], {}
            for d in docs:
                v = d.get(k)
                per[f"seed_{d.get('seed', docs.index(d))}"] = v
                if isinstance(v, (int, float)) and v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            row = {"metric": k,
                   "mean": f"{np.mean(vals):.6f}" if vals else "",
                   "std":  f"{np.std(vals):.6f}"  if vals else ""}
            row.update({kk: (f"{vv:.6f}" if isinstance(vv, (int, float)) else "")
                        for kk, vv in per.items()})
            w.writerow(row)
    print(f"  rebuilt {os.path.relpath(csv_path, REPO)}")

    # ── aggregate + print README table ─────────────────────────────────────────
    agg = aggregate_curve_metrics(per_seed)
    print(f"\n===== {args.method}  B curve-shape aggregate =====")
    for prefix, name in CURVE_PLOTS:
        row = agg[prefix]
        m, p = row["mse"], row["pct"]
        print(f"\n{name}")
        print(f"  MSE  {m['mean']:.6g} ± {m['std']:.6g}   "
              f"seeds=[{', '.join(f'{x:.6g}' for x in m['per_seed'])}]")
        print(f"  %err {p['mean']:.6g} ± {p['std']:.6g}   "
              f"seeds=[{', '.join(f'{x:.6g}' for x in p['per_seed'])}]")

    # machine-readable dump for README rendering
    dump = os.path.join(rdir, "curve_b_aggregate.json")
    with open(dump, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n  wrote {os.path.relpath(dump, REPO)}")


if __name__ == "__main__":
    main()
