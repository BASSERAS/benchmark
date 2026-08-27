#!/usr/bin/env python3
"""
measure_memorisation.py
───────────────────────
Nearest-neighbour memorisation diagnostic for Deep-MKV-TS on TrueDataset.

The question any generator must answer honestly: are the "generated" paths
genuinely new samples, or near-copies of the training set?

The estimator is byte-identical to the SBTS and LS4 versions -- same space, same
statistic. It has to be, or the methods' numbers would not be comparable. Only
the denominator split and the method-specific prose differ.

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
extreme failure. Their absence does NOT clear the method, and for Deep-MKV-TS it
is close to uninformative: a sample is a 127-step Euler-Maruyama rollout of
``X_{k+1} = X_k + b^ref(X_k) + sigma(X_k) eps_k`` driven by fresh Gaussian
``eps_k``, so bitwise equality with a training path has probability zero by
construction. A zero duplicate count must not be read as evidence of novelty;
the RATIO is the number that matters.

Where memorisation could enter for this method
──────────────────────────────────────────────
Deep-MKV-TS never touches a training path at generation time -- ``model.sample``
only rolls the fitted drift and control forward from ``x0 = log(100)``. But it is
not in the same position as a pure likelihood model either. During TRAINING the
ridge Z-proxy regresses the adjoint moments onto ``Phi^ref``, a flattened prefix
of the sampled path, and the training bank enters through the path-functional
discrepancy. Memorisation would therefore have to arrive through the fitted
control network's weights, whose exact count is recorded as ``n_parameters`` in
``weights/seed_0_config.json`` and is small against the 6144*128*8 ~ 6.3M
training values. That is a capacity argument, and "implausible" is an argument,
not a measurement -- which is why this script exists.

Usage
-----
    /home/tbasseras/gpu-venv/bin/python \
        results/trueexperiment/Deep-MKV-TS/code/measure_memorisation.py
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
    # The later-era arm. NOT the denominator -- see denominator_rationale below --
    # but measured rather than asserted, for two reasons. First, the 0.9316 figure
    # quoted in that rationale was a hand-typed constant until now; computing it
    # here makes the claim track the data instead of a comment. Second, SBTS and
    # CSDI both report `median_nn_test_era` / `nn_ratio_vs_test_era`
    # (SBTS/code/measure_memorisation.py:184,187) and this method's
    # render_readme.py:456,459 reads both, so an artefact without them cannot be
    # rendered and is not comparable to the columns beside it.
    test = logret_flat(np.load(os.path.join(data_dir, f"true_S_test_{seq_tag}.npy")))
    print(
        f"train {train.shape}   held-out real ({DENOMINATOR_SPLIT}) {real.shape}   "
        f"test-era {test.shape}   device={args.device}"
    )

    # denominator: how close does a GENUINE same-era sample sit to the train set?
    base = min_dists(real, train, args.device, args.chunk)
    med_base = float(np.median(base))
    med_test = float(np.median(min_dists(test, train, args.device, args.chunk)))
    print(
        f"median NN({DENOMINATOR_SPLIT} -> train) = {med_base:.6f}   <- calibration"
    )
    print(
        f"median NN(test -> train) = {med_test:.6f}   "
        f"({med_test / med_base:.4f}x the {DENOMINATOR_SPLIT} distance -- this gap "
        f"IS the regime change, and is why test is not the denominator)"
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
                "nn_ratio_vs_test_era": med_gen / med_test,
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
            f"measured gap is {med_test / med_base:.4f} -- using test would inflate "
            f"every ratio by about {100 * (med_base / med_test - 1):.0f}% for a "
            "reason unrelated to the generator."
        ),
        "median_nn_heldout": med_base,
        "median_nn_test_era": med_test,
        "nn_ratio": float(np.mean(ratios)),
        "nn_ratio_std": float(np.std(ratios)),
        "nn_ratio_vs_test_era": float(
            np.mean([p["median_nn_generated"] for p in per_seed])
        )
        / med_test,
        "median_nn_generated": float(
            np.mean([p["median_nn_generated"] for p in per_seed])
        ),
        "n_exact_duplicates": int(sum(p["n_exact_duplicates"] for p in per_seed)),
        "per_seed": per_seed,
        "note": (
            "Exact duplicates being 0 does NOT clear the method, and for "
            "Deep-MKV-TS it is close to uninformative: a sample is a 127-step "
            "Euler-Maruyama rollout driven by fresh Gaussian increments, so "
            "bitwise equality with a training path has probability zero by "
            "construction. The ratio is the number that matters. Estimator "
            "byte-identical to SBTS/code/measure_memorisation.py and "
            "LS4/code/measure_memorisation.py so the columns are directly "
            "comparable."
        ),
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
