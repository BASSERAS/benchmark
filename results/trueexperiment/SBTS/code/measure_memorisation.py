#!/usr/bin/env python3
"""
measure_memorisation.py
-----------------------
Nearest-neighbour memorisation diagnostic for the d = 8 SBTS output on real
crypto data.

The question a kernel generator must answer honestly: are the "generated" paths
genuinely new samples, or near-copies of the training set? For SBTS this is not
a formality -- the bandwidth `h` is precisely the knob that trades fit against
copying, and a small enough `h` will beat every distributional metric in the
panel by reproducing the training set.

Diagnostic
----------
Work in **log-return space** (every path is anchored at S0 = 100, so raw price
distance is dominated by the shared anchor and understates similarity). Flatten
each path to a (T-1)*d = 1016-vector, then compute

    nn_ratio = median_i  min_j ||gen_i  - train_j||
             -----------------------------------------
               median_i  min_j ||real_i - train_j||

Reading it
----------
  ratio ~ 1.0   generated paths sit no closer to the training set than genuine
                real data does -- no memorisation.
  ratio << 1.0  generated paths hug the training set; 1/ratio is how many times
                closer they sit.

WHICH REAL SPLIT GOES IN THE DENOMINATOR -- the one thing this file does
differently from its Heston sibling
-----------------------------------------------------------------------
The Heston version uses the `test` split, and there that is exactly right: every
Heston split is an i.i.d. draw from one known SDE, so `test` is "genuine data
from the same law the generator never saw" and nothing else.

TrueDataset is not that. `dataset_stats.json` records `split_mode:
holdout-era`: train / val / valdisc are drawn from 2022-07 -> 2024-12, while
disc / test come from 2025-02 -> 2026-07, and annualised vol FALLS between the
two eras on six of the eight assets (BTC 0.498 -> 0.429, SOL 1.039 -> 0.733,
DOGE 1.107 -> 0.800).

Measured on this build:

    median NN(val  -> train) = 0.018943
    median NN(test -> train) = 0.017647     = 0.932x the val distance

Note the direction, because it is the opposite of the intuition. The test era is
not FURTHER from train despite being a different regime -- it is CLOSER. Nothing
about it is more similar to training data; its returns are simply smaller, so
every distance in the space contracts with them. The test denominator is
measuring a volatility level, not a similarity.

Using it anyway biases the diagnostic in the dangerous direction. A smaller
denominator makes every ratio LARGER: the same SBTS banks score 1.081 against
val and 1.161 against test. Memorisation is the low-ratio failure, so the test
denominator pushes a memorising generator UP, towards the healthy-looking end,
and it does so silently. That is the wrong yardstick for a copying diagnostic.

So the primary denominator here is **`val`** -- same era as train, therefore the
honest answer to "how close does real data of this kind sit to the training
set". The `test` denominator is reported alongside it as
`nn_ratio_vs_test_era`, and the gap between the two numbers is a direct readout
of the regime change. Neither is hidden; only the primary one is used for the
pass/fail call, and it is the same denominator the (h, K) selection was run
against, so the number in this file is comparable to the calibration sweep.

Usage
-----
    /home/tbasseras/gpu-venv/bin/python \
        results/trueexperiment/SBTS/code/measure_memorisation.py
    ... --seeds 0,1,2,3,4 --device cuda
"""

import argparse
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SBTS = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(SBTS, "../../.."))
DEFAULT_DATA = os.path.join(REPO, "dataset", "TrueDataset", "variants",
                            "om_2022-07_N6144")
DEFAULT_TAG = "6144x128x8"


def logret_flat(S):
    """(N, T, d) prices -> (N, (T-1)*d) float32 log-returns."""
    lr = np.diff(np.log(S), axis=1)
    return np.ascontiguousarray(lr.reshape(len(lr), -1), dtype=np.float32)


def min_dists(query, ref, device, chunk=256):
    """For each row of `query`, the L2 distance to its nearest row in `ref`."""
    q = torch.as_tensor(query, device=device)
    r = torch.as_tensor(ref, device=device)
    out = torch.empty(len(q), device=device)
    for i in range(0, len(q), chunk):
        out[i:i + chunk] = torch.cdist(q[i:i + chunk], r).min(dim=1).values
    return out.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--seq-tag", default=DEFAULT_TAG)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    tag = args.seq_tag

    def _load(name):
        return logret_flat(np.load(os.path.join(args.data_dir, name)))

    train = _load(f"true_S_{tag}.npy")
    val = _load(f"true_S_val_{tag}.npy")       # same era as train  -> primary
    test = _load(f"true_S_test_{tag}.npy")     # later era          -> context
    print(f"train {train.shape}  val {val.shape}  test {test.shape}  "
          f"device={args.device}")

    # Denominator: how close does real, same-era data sit to the training set?
    base = min_dists(val, train, args.device, args.chunk)
    med_base = float(np.median(base))
    med_test = float(np.median(min_dists(test, train, args.device, args.chunk)))
    print(f"median NN(val  -> train) = {med_base:.6f}   <- calibration (same era)")
    print(f"median NN(test -> train) = {med_test:.6f}   "
          f"({med_test / med_base:.3f}x the val distance -- this gap IS the "
          f"regime change)")

    per_seed, ratios = [], []
    train_rows = {r.tobytes() for r in train}
    for s in seeds:
        p = os.path.join(SBTS, "generated_paths", f"seed_{s}",
                         f"generated_paths_{tag}.npy")
        if not os.path.exists(p):
            print(f"  seed {s}: MISSING {p} -- skipped")
            continue
        gen = logret_flat(np.load(p))
        dg = min_dists(gen, train, args.device, args.chunk)
        med_gen = float(np.median(dg))
        ratio = med_gen / med_base
        dup = sum(1 for r in gen if r.tobytes() in train_rows)
        per_seed.append({"seed": s, "median_nn_generated": med_gen,
                         "nn_ratio": ratio,
                         "nn_ratio_vs_test_era": med_gen / med_test,
                         "n_exact_duplicates": dup})
        ratios.append(ratio)
        print(f"  seed {s}: median NN(gen -> train) = {med_gen:.6f}   "
              f"ratio = {ratio:.4f}   duplicates = {dup}")

    if not per_seed:
        raise SystemExit("ABORT: no generated arrays found under "
                         f"{SBTS}/generated_paths/seed_*/")

    mean_gen = float(np.mean([p["median_nn_generated"] for p in per_seed]))
    out = {
        "diagnostic": "nearest-neighbour memorisation ratio, log-return space",
        "definition": "median_i min_j ||gen_i - train_j||  /  "
                      "median_i min_j ||real_val_i - train_j||",
        "denominator_split": "val",
        "why_val_not_test":
            "TrueDataset splits by era (dataset_stats.json: split_mode = "
            "holdout-era). train/val/valdisc are 2022-07 -> 2024-12; disc/test "
            "are 2025-02 -> 2026-07, and annualised vol FALLS between the eras "
            "on 6 of 8 assets. Measured, the test era sits CLOSER to train than "
            "val does (0.932x the distance) -- not because it is more similar, "
            "but because its returns are smaller, so every distance contracts "
            "with them. A smaller denominator inflates every ratio, and "
            "memorisation is the LOW-ratio failure, so a test denominator "
            "pushes a memorising generator up towards the healthy end. val is "
            "the same era as train and is the denominator the (h, K) sweep was "
            "scored against. nn_ratio_vs_test_era is reported for context.",
        "interpretation": "1.0 = generated paths sit no closer to the training "
                          "set than real same-era data; <1 = memorisation, "
                          "1/ratio = how many times closer they sit",
        "space": f"log-returns, flattened to (T-1)*d = {train.shape[1]} dims",
        "median_nn_heldout": med_base,
        "median_nn_test_era": med_test,
        "nn_ratio": float(np.mean(ratios)),
        "nn_ratio_std": float(np.std(ratios)),
        "nn_ratio_vs_test_era": mean_gen / med_test,
        "median_nn_generated": mean_gen,
        "n_exact_duplicates": int(sum(p["n_exact_duplicates"] for p in per_seed)),
        "per_seed": per_seed,
        "note": "Exact duplicates being 0 does NOT clear the method: SBTS "
                "interpolates between training paths, so it produces "
                "near-copies rather than copies. The ratio is the number that "
                "matters. This ratio is not a passive report -- it is the "
                "objective the bandwidth was selected on: (h, K) were chosen to "
                "minimise |log nn_ratio| subject to the real-vs-real envelope, "
                "with corr_err as the tie-break. Rejected candidates make the "
                "point: K=3, h=0.05 scores vol_err 6.06 %, better than the "
                "real-vs-real MINIMUM of 8.72 %, and its nn_ratio is 0.197. "
                "Beating the floor on a distributional metric is what "
                "memorisation looks like from inside the panel; only this "
                "diagnostic sees it.",
    }
    dst = os.path.join(SBTS, "losses", "memorisation.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nmean nn_ratio = {out['nn_ratio']:.4f} (vs val, same era)")
    print(f"     vs test-era denominator = {out['nn_ratio_vs_test_era']:.4f}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
