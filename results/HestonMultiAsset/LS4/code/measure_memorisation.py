#!/usr/bin/env python3
"""
measure_memorisation.py
───────────────────────
Nearest-neighbour memorisation diagnostic for the d = 8 LS4 output.

The question any generator must answer honestly: are the "generated" paths
genuinely new samples, or near-copies of the training set?

The estimator is byte-identical to `SBTS/code/measure_memorisation.py` -- same
space, same splits, same statistic. It has to be, or the two methods' numbers
would not be comparable. Only the method-specific prose differs.

Diagnostic
──────────
Work in **log-return space** (every path is anchored at S0 = 100, so raw price
distance is dominated by the shared anchor and understates similarity). Flatten
each path to a (T-1)*d = 2008-vector, then compute

    nn_ratio = median_i  min_j ||gen_i  - train_j||
             -----------------------------------------
               median_i  min_j ||real_i - train_j||

where `real` is the HELD-OUT test split: data from the same law the generator
never saw.

Reading it
──────────
  ratio ~ 1.0   generated paths sit no closer to the training set than genuine
                held-out data does -- no memorisation.
  ratio << 1.0  generated paths hug the training set; 1/ratio is how many times
                closer they sit than real data does.

The denominator is the whole point: it calibrates "how close is close" using the
true process itself, so the number cannot be gamed by the dataset's intrinsic
scale or dimension.

Also reported: exact duplicates (bitwise-identical flattened vectors), the
extreme failure. Their absence does NOT clear the method. For a latent-variable
model an exact duplicate is essentially impossible by construction -- the decoder
output is a continuous function of a Gaussian prior sample, so bitwise equality
has probability zero. A zero duplicate count is therefore uninformative here and
must not be read as evidence of novelty; the RATIO is the number that matters.

Note the asymmetry with SBTS worth keeping in view when comparing the two
columns: SBTS is an explicit kernel average over training paths, so memorisation
is its *default* failure mode, while LS4 never touches a training path at
generation time and can only memorise by having encoded the training set into its
weights. The same low ratio would mean different things; the same diagnostic is
still the right way to detect it.

Usage
-----
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/LS4/code/measure_memorisation.py
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
DATA = os.path.join(REPO, "dataset", "HestonMultiAsset")


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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    train = logret_flat(np.load(os.path.join(DATA, "heston_ma_S_8192x252x8.npy")))
    real = logret_flat(np.load(os.path.join(DATA, "heston_ma_S_test_8192x252x8.npy")))
    print(f"train {train.shape}   held-out real {real.shape}   device={args.device}")

    # denominator: how close does GENUINE held-out data sit to the training set?
    base = min_dists(real, train, args.device, args.chunk)
    med_base = float(np.median(base))
    print(f"median NN(held-out real -> train) = {med_base:.6f}   <- calibration")

    per_seed, ratios = [], []
    train_rows = {r.tobytes() for r in train}
    for s in seeds:
        p = os.path.join(SBTS, "generated_paths", f"seed_{s}",
                         "generated_paths_8192x252x8.npy")
        if not os.path.exists(p):
            print(f"  seed {s}: MISSING {p} -- skipped")
            continue
        gen = logret_flat(np.load(p))
        dg = min_dists(gen, train, args.device, args.chunk)
        med_gen = float(np.median(dg))
        ratio = med_gen / med_base
        dup = sum(1 for r in gen if r.tobytes() in train_rows)
        per_seed.append({"seed": s, "median_nn_generated": med_gen,
                         "nn_ratio": ratio, "n_exact_duplicates": dup})
        ratios.append(ratio)
        print(f"  seed {s}: median NN(gen -> train) = {med_gen:.6f}   "
              f"ratio = {ratio:.4f}   ({1/ratio:.2f}x closer)   duplicates = {dup}")

    if not per_seed:
        raise SystemExit("ABORT: no generated arrays found -- run run_all_multiasset.py first.")

    out = {
        "diagnostic": "nearest-neighbour memorisation ratio, log-return space",
        "definition": "median_i min_j ||gen_i - train_j||  /  "
                      "median_i min_j ||real_test_i - train_j||",
        "interpretation": "1.0 = generated paths sit no closer to the training set than "
                          "genuine held-out data; <1 = memorisation, 1/ratio = how many "
                          "times closer they sit",
        "space": "log-returns, flattened to (T-1)*d = 2008 dims",
        "median_nn_heldout": med_base,
        "nn_ratio": float(np.mean(ratios)),
        "nn_ratio_std": float(np.std(ratios)),
        "median_nn_generated": float(np.mean([p["median_nn_generated"] for p in per_seed])),
        "n_exact_duplicates": int(sum(p["n_exact_duplicates"] for p in per_seed)),
        "per_seed": per_seed,
        "note": "Exact duplicates being 0 does NOT clear the method, and for LS4 it is "
                "close to uninformative: the decoder output is a continuous function of "
                "a Gaussian prior sample, so bitwise equality with a training path has "
                "probability zero by construction. The ratio is the number that matters. "
                "Estimator byte-identical to SBTS/code/measure_memorisation.py so the two "
                "columns are directly comparable; SBTS d = 8 scores 0.2189 +/- 0.0003 on "
                "it, and SBTS d = 1 scores 0.158.",
    }
    dst = os.path.join(SBTS, "losses", "memorisation.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nmean nn_ratio = {out['nn_ratio']:.4f} "
          f"({1/out['nn_ratio']:.2f}x closer than held-out real data)")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
