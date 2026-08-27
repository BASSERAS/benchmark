#!/usr/bin/env python3
"""
measure_memorisation.py
───────────────────────
Nearest-neighbour memorisation diagnostic for the REFERENCE SDE on TrueDataset.

The question any generator must answer honestly: are the "generated" paths
genuinely new samples, or near-copies of the training set?

The estimator is byte-identical to the SBTS, CSDI and Deep-MKV-TS versions --
same space, same splits, same statistic. It has to be, or the methods' numbers
would not be comparable. Only the method-specific prose differs.

Why run it on a model that provably cannot memorise
───────────────────────────────────────────────────
Because that is exactly what makes the number useful.

The reference SDE has **25 fitted numbers in total** -- 7 covariance-diagonal
coefficients, 7 off-diagonal, 5 drift, 2 trend half-lives, 2 activity half-lives
and 2 mixing weights, all listed in ``weights/reference_kernel.json`` -- against
6144*128*8 ~ 6.29M training values. Nothing else is fitted: the neural control is
built but never trained, and ``code/generate_reference_true.py`` verifies
``Zhat == 0`` before sampling. Memorisation through 25 numbers is not
implausible, it is arithmetically impossible.

So this row is not a test of the reference SDE. It is a **calibration of the
diagnostic itself**: it reports what ``nn_ratio`` reads for a generator that
certainly did not memorise, on this dataset, in this metric, at this sample size.
Without it, "method X scored 0.9" is hard to read -- 0.9 could be mild
memorisation or simply what a clean model scores here. With it, the reference
column supplies the null and the comparison becomes interpretable.

Note the asymmetry across the columns, worth keeping in view. SBTS is an explicit
kernel average over training paths, so memorisation is its *default* failure
mode; on this panel it nonetheless measures 1.0815, i.e. its generated paths sit
no closer to the training set than held-out same-era real data does. CSDI
measures 0.5612 -- its paths sit about 1.8x closer than real data does. The
reference SDE never touches a training path at generation time: ``model.sample``
only rolls the fitted drift and control forward from ``x0 = log(100)`` driven by
fresh Gaussian noise. The same statistic means different things in the different
columns, which is the argument for reading them side by side rather than ranking
them.

The denominator is `val`, NOT `test`
────────────────────────────────────
This is the one substantive difference from the Heston version, and it is
mandated by section 9.1 of ``truedatasetguideline.md``.

TrueDataset is ONE realisation of a real market, split by era: train, val and
valdisc come from 2022-07 to 2024-12, while disc and test come from 2025-02
onward, behind a 45.5-day embargo. The test split is therefore not an
exchangeable draw from the same law -- it is a different market regime. Using it
as the denominator would measure "how far is 2025 from 2023", a regime-shift
distance, and then charge the generator for failing to match it. The measured
consequence is in the envelope: NN(test -> train) sits at 0.9316 of NN(val ->
train), so a test denominator would inflate every ratio by about 7% for a reason
that has nothing to do with the generator.

`val` is the same-era honest sample. It is what a non-memorising generator
should look like, so it is what calibrates "how close is close".

Diagnostic
──────────
Work in **log-return space** -- every path is anchored at S0 = 100, so raw price
distance is dominated by the shared anchor and understates similarity. Flatten
each path to a (T-1)*d = 127*8 = 1016-vector, then compute

    nn_ratio = median_i  min_j ||gen_i - train_j||
             ---------------------------------------
               median_i  min_j ||val_i - train_j||

Reading it
──────────
  ratio ~ 1.0   generated paths sit no closer to the training set than a genuine
                same-era held-out sample does -- no memorisation.
  ratio << 1.0  generated paths hug the training set; 1/ratio is how many times
                closer they sit than real data does.
  ratio >> 1.0  the generator is not memorising, but it is also not landing in
                the right region at all. High is not automatically good, which is
                exactly why the selection criterion minimises |log ratio| rather
                than maximising the ratio.

Also reported: exact duplicates (bitwise-identical flattened vectors), the
extreme failure. Their absence does NOT clear the method, and for the reference
SDE it is close to uninformative: a sample is a 127-step Euler-Maruyama rollout
of ``X_{k+1} = X_k + b^ref(X_k) + sigma(X_k) eps_k`` driven by fresh Gaussian
``eps_k``, so bitwise equality with a training path has probability zero by
construction. A zero duplicate count must not be read as evidence of novelty;
the RATIO is the number that matters.

Where memorisation could enter for this method
──────────────────────────────────────────────
Nowhere, and that is the point of running it -- but the argument has to be stated
precisely rather than waved at.

The training bank enters this method exactly once: through the 300-step Gaussian
NLL fit that produced ``weights/reference_kernel.json``. That fit compresses the
whole training split into the 25 numbers above. At generation time nothing else
is consulted: ``model.sample`` rolls the fitted drift and control forward from
``x0 = log(100)`` driven by fresh Gaussian noise, with no lookup, no kernel
average over training paths, and no nearest-neighbour step. There is no trained
neural control to smuggle capacity in either -- both output heads have a zero
final layer, so ``Zhat == 0`` exactly and the sampler integrates ``sigma^ref``
itself.

That is a capacity argument, and "arithmetically impossible" is still an
argument, not a measurement -- which is why this script runs anyway. A reference
row that came back at 0.3 would not mean the reference SDE memorised; it would
mean the ESTIMATOR is mis-calibrated on this dataset, and that every other
column read through it is suspect. This row is the instrument check.

Usage
-----
    /home/tbasseras/gpu-venv/bin/python \
        results/trueexperiment/reference/code/measure_memorisation.py
    ... --seeds 0,1,2,3,4 --device cuda
"""

import argparse
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)
DATA = "/home/tbasseras/benchmark/dataset/TrueDataset/variants/om_2022-07_N6144"
SEQ_TAG = "6144x128x8"
# The split whose distance to train calibrates the ratio. Named as a constant so
# that changing it is a visible edit, not a silently different denominator.
DENOMINATOR_SPLIT = "val"


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
        out[i : i + chunk] = torch.cdist(q[i : i + chunk], r).min(dim=1).values
    return out.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=256)
    # Explicit rather than implicit: every other stage of the pipeline is handed
    # the dataset and the shape tag on the command line, and a memorisation
    # ratio computed against a DIFFERENT train split than the one the model saw
    # is meaningless.  Defaults keep the previous behaviour.
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--seq-tag", default=SEQ_TAG)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    data_dir = args.data_dir
    seq_tag = args.seq_tag

    train = logret_flat(np.load(os.path.join(data_dir, f"true_S_{seq_tag}.npy")))
    real = logret_flat(
        np.load(os.path.join(data_dir, f"true_S_{DENOMINATOR_SPLIT}_{seq_tag}.npy"))
    )
    print(
        f"train {train.shape}   held-out real ({DENOMINATOR_SPLIT}) {real.shape}   "
        f"device={args.device}"
    )

    # denominator: how close does a GENUINE same-era sample sit to the train set?
    base = min_dists(real, train, args.device, args.chunk)
    med_base = float(np.median(base))
    print(
        f"median NN({DENOMINATOR_SPLIT} -> train) = {med_base:.6f}   <- calibration"
    )

    per_seed, ratios = [], []
    train_rows = {r.tobytes() for r in train}
    for s in seeds:
        p = os.path.join(
            METHOD_DIR,
            "generated_paths",
            f"seed_{s}",
            f"generated_paths_{seq_tag}.npy",
        )
        if not os.path.exists(p):
            print(f"  seed {s}: MISSING {p} -- skipped")
            continue
        gen = logret_flat(np.load(p))
        dg = min_dists(gen, train, args.device, args.chunk)
        med_gen = float(np.median(dg))
        ratio = med_gen / med_base
        dup = sum(1 for r in gen if r.tobytes() in train_rows)
        per_seed.append(
            {
                "seed": s,
                "median_nn_generated": med_gen,
                "nn_ratio": ratio,
                "n_exact_duplicates": dup,
            }
        )
        ratios.append(ratio)
        print(
            f"  seed {s}: median NN(gen -> train) = {med_gen:.6f}   "
            f"ratio = {ratio:.4f}   ({1 / ratio:.2f}x closer)   duplicates = {dup}"
        )

    if not per_seed:
        raise SystemExit(
            "ABORT: no generated arrays found -- run generate_bank_true.py first."
        )

    n_dims = train.shape[1]
    out = {
        "diagnostic": "nearest-neighbour memorisation ratio, log-return space",
        "definition": (
            "median_i min_j ||gen_i - train_j||  /  "
            "median_i min_j ||val_i - train_j||"
        ),
        "interpretation": (
            "1.0 = generated paths sit no closer to the training set than a "
            "genuine same-era held-out sample; <1 = memorisation, 1/ratio = how "
            "many times closer they sit; >1 = the generator is not landing in the "
            "right region"
        ),
        "space": f"log-returns, flattened to (T-1)*d = {n_dims} dims",
        "denominator_split": DENOMINATOR_SPLIT,
        "denominator_rationale": (
            "truedatasetguideline section 9.1. TrueDataset is one realisation "
            "split by era; test sits behind a 45.5-day embargo in a later market "
            "regime, so NN(test -> train) measures regime shift, not novelty. The "
            "measured gap is 0.9316 -- using test would inflate every ratio by "
            "about 7% for a reason unrelated to the generator."
        ),
        "median_nn_heldout": med_base,
        "nn_ratio": float(np.mean(ratios)),
        "nn_ratio_std": float(np.std(ratios)),
        "median_nn_generated": float(
            np.mean([p["median_nn_generated"] for p in per_seed])
        ),
        "n_exact_duplicates": int(sum(p["n_exact_duplicates"] for p in per_seed)),
        "per_seed": per_seed,
        "note": (
            "Exact duplicates being 0 does NOT clear a method, and here it is "
            "entirely uninformative: a sample is a 127-step Euler-Maruyama "
            "rollout driven by fresh Gaussian increments, so bitwise equality "
            "with a training path has probability zero by construction. The "
            "ratio is the number that matters. Estimator byte-identical to "
            "SBTS/code/measure_memorisation.py and "
            "CSDI/code/measure_memorisation.py so the columns are directly "
            "comparable. THIS row is a calibration of the diagnostic rather "
            "than a test of the generator: the reference SDE has 25 fitted "
            "numbers (7 covariance-diagonal, 7 off-diagonal, 5 drift, 2 trend "
            "half-lives, 2 activity half-lives, 2 mixing weights) against "
            "6144*128*8 ~ 6.29M training values, and its neural control is "
            "verified to be exactly zero before sampling, so memorisation is "
            "arithmetically impossible. The value it returns is therefore what "
            "nn_ratio reads for a generator that certainly did not memorise, "
            "which is the null the other columns need."
        ),
        "n_fitted_coefficients": 25,
        "n_gamma_coefficients": 19,
        "control_is_zero": True,
    }
    dst = os.path.join(METHOD_DIR, "losses", "memorisation.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(
        f"\nmean nn_ratio = {out['nn_ratio']:.4f} "
        f"({1 / out['nn_ratio']:.2f}x closer than held-out real data)"
    )
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
