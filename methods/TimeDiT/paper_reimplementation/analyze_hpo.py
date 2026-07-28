"""Marginal / coordinate analysis of the TimeDiT Sine-disc HPO results.

Reads every hpo_results_shard*.jsonl trial and, for each swept hyperparameter axis,
reports the mean discriminative score per value (the *marginal* effect of that axis).
The best value per axis is combined into a single "coordinate-descent candidate"
config, which is the seed for the refined second-stage grid.  Also prints the raw
top-K configs and how often each value appears among them (winner-direction voting).

Target: disc -> paper's 0.0086.

Usage:
  python analyze_hpo.py                      # analyse stage-1 (hpo_results_shard*.jsonl)
  python analyze_hpo.py --glob 'hpo_stage2_shard*.jsonl' --out hpo_stage2_analysis.json
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

AXES = ["norm", "schedule", "lr", "sampler", "learn_sigma", "ddim_steps", "steps", "ema"]


def load(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        for line in open(path):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # keep only finite disc
    return [r for r in rows if isinstance(r.get("disc_mean"), (int, float))
            and r["disc_mean"] == r["disc_mean"]]


def marginals(rows):
    """For each axis present in the data, mean disc per value."""
    out = {}
    for ax in AXES:
        buckets = defaultdict(list)
        for r in rows:
            if ax in r:
                buckets[r[ax]].append(r["disc_mean"])
        if buckets:
            out[ax] = {str(v): (float(np.mean(d)), float(np.min(d)), len(d))
                       for v, d in buckets.items()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="hpo_results_shard*.jsonl")
    p.add_argument("--out", default="hpo_stage1_analysis.json")
    p.add_argument("--topk", type=int, default=8)
    args = p.parse_args()

    rows = load(args.glob)
    if not rows:
        print(f"[analyze] no trials matched {args.glob}")
        return
    rows.sort(key=lambda r: r["disc_mean"])
    print(f"[analyze] {len(rows)} trials from {args.glob}\n")

    # ---- raw leaderboard ----
    print(f"=== TOP {args.topk} CONFIGS (by disc) ===")
    for r in rows[:args.topk]:
        extra = " ".join(f"{k}={r[k]}" for k in ("ddim_steps", "steps", "ema") if k in r)
        print(f"  disc={r['disc_mean']:.4f} pred={r['pred_mean']:.4f} | "
              f"{r['norm']:<8} {r['sampler']:<12} ls={str(r['learn_sigma'])[0]} "
              f"{r['schedule']:<6} lr={r['lr']:.0e} {extra}")

    # ---- marginal effect per axis ----
    marg = marginals(rows)
    print("\n=== MARGINAL disc per axis (mean | best | n) — lower mean is better ===")
    best_value = {}
    for ax, vals in marg.items():
        if len(vals) < 2:
            continue  # axis not actually varied
        ordered = sorted(vals.items(), key=lambda kv: kv[1][0])
        best_value[ax] = ordered[0][0]
        cells = "  ".join(f"{v}:{m:.3f}/{lo:.3f}(n{n})" for v, (m, lo, n) in ordered)
        print(f"  {ax:<12} -> best='{ordered[0][0]}'   {cells}")

    # ---- winner-direction voting among top-K ----
    print(f"\n=== VALUE FREQUENCY among top {args.topk} ===")
    for ax in marg:
        cnt = defaultdict(int)
        for r in rows[:args.topk]:
            if ax in r:
                cnt[str(r[ax])] += 1
        if len(cnt) > 1:
            print(f"  {ax:<12} {dict(cnt)}")

    # ---- combined coordinate-descent candidate ----
    print("\n=== COORDINATE-DESCENT CANDIDATE (best value per axis) ===")
    print(f"  {best_value}")
    print("  ^ verify this exact combo in stage-2 (it may not exist in stage-1).")

    with open(args.out, "w") as f:
        json.dump({"n_trials": len(rows), "top": rows[:args.topk],
                   "marginals": marg, "best_value": best_value}, f, indent=2)
    print(f"\n[analyze] wrote {args.out}")


if __name__ == "__main__":
    main()
