"""CSDI -- is the low NN ratio memorisation, or is it shrunken support?

THE QUESTION THIS ANSWERS
--------------------------
``measure_memorisation.py`` reports

    nn_ratio = median_i min_j ||gen_i - train_j||
             / median_i min_j ||val_i - train_j||

and CSDI scores ~0.561 against a real-vs-real band of 0.932-1.000. Read naively
that says the generated paths sit far CLOSER to the training set than genuine
held-out data does, which is the signature of copying. It also reports
``duplicates = 0``, so whatever is happening is not literal reproduction.

The confound is that ``measure_memorisation.py`` works in RAW flattened
log-return space -- ``logret_flat`` does no standardisation. L2 distance in that
space is homogeneous of degree 1 in the scale of the returns. CSDI's realised
volatility is ~0.52x the real value (see ``screen_envelope.py``). A cloud that
is uniformly half as wide has smaller nearest-neighbour distances for purely
geometric reasons, with zero information copied. And 0.561 is suspiciously close
to 0.52.

So the low ratio has two candidate explanations that the statistic cannot
separate on its own:

    (H1) memorisation  -- generated paths are drawn near training examples
    (H2) shrunken support -- the whole cloud is ~2x too narrow

TWO TESTS, AND WHY THE SECOND ONE IS THE DECISIVE ONE
------------------------------------------------------
TEST A (rescale the generated data UP). Rescale each generated asset's
log-returns so their std matches the real training std, rebuild, and recompute
the ratio. Under H2 the ratio should climb into the real-vs-real band. Under H1
it should stay low, because rescaling does not move a copy away from its source.

TEST B (rescale the REAL VALIDATION data DOWN) -- the positive control, and the
one that actually settles it. Take ``val``, which is genuine market data from
the same era and therefore CANNOT be a copy of train in any meaningful sense,
and shrink its log-returns by CSDI's own volatility ratio. If that alone drives
val's nn_ratio down to ~0.56, then the statistic demonstrably produces CSDI's
score from shrinkage ALONE, on data known to be innocent. That is a constructive
proof that the low ratio is not evidence of copying.

Test B is the stronger argument because it does not depend on any property of
CSDI. It is a statement about what the DIAGNOSTIC does when handed a narrow
cloud, established on data whose provenance is not in question.

WHAT AN HONEST NEGATIVE RESULT LOOKS LIKE
------------------------------------------
If Test A leaves the ratio near 0.56 AND Test B leaves val near 0.93, then
shrinkage does NOT explain the gap and the memorisation reading stands. Report
that outcome exactly as loudly. This script is written to be able to falsify the
author's preferred explanation, which is the only reason running it is worth
anything.

Reads   ../../../dataset/TrueDataset/variants/om_2022-07_N6144/true_S_<tag>.npy
        ../generated_paths/seed_{i}/generated_paths_<tag>.npy
Writes  ../losses/memorisation_scale_check.json

Usage:
    CUDA_VISIBLE_DEVICES=0 /home/tbasseras/gpu-venv/bin/python \
        results/trueexperiment/CSDI/code/check_memorisation_scale.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CSDI = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(CSDI, "../../.."))
DEFAULT_DATA = os.path.join(REPO, "dataset", "TrueDataset", "variants",
                            "om_2022-07_N6144")
DEFAULT_TAG = "6144x128x8"

# Band from the SBTS real-vs-real calibration; see measure_memorisation.py.
BAND_LO, BAND_HI = 0.932, 1.000


def logret(S):
    """(N, T, d) prices -> (N, T-1, d) log-returns. Kept 3-D so the per-asset
    std used for rescaling is unambiguous; flattening happens later."""
    return np.diff(np.log(S), axis=1)


def flat(lr):
    return np.ascontiguousarray(lr.reshape(len(lr), -1), dtype=np.float32)


def min_dists(query, ref, device, chunk=256):
    """Identical to measure_memorisation.min_dists -- same metric, same chunking,
    so the numbers here are comparable to that script's output line for line."""
    q = torch.as_tensor(query, device=device)
    r = torch.as_tensor(ref, device=device)
    out = torch.empty(len(q), device=device)
    for i in range(0, len(q), chunk):
        out[i:i + chunk] = torch.cdist(q[i:i + chunk], r).min(dim=1).values
    return out.cpu().numpy()


def per_asset_std(lr):
    """std over (paths, time) for each asset -> (d,)."""
    return lr.std(axis=(0, 1))


def rescale_to(lr, target_std):
    """Match each asset's log-return std to target_std, preserving the mean.

    Only the SCALE is touched: the temporal ordering, the cross-asset
    correlation structure and the shape of the marginal are all untouched, so
    anything this changes is attributable to scale alone. That is the entire
    point -- if it were a general-purpose 'make it look real' transform the test
    would prove nothing.
    """
    mu = lr.mean(axis=(0, 1), keepdims=True)
    sd = lr.std(axis=(0, 1), keepdims=True)
    return mu + (lr - mu) * (target_std.reshape(1, 1, -1) / sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--seq-tag", default=DEFAULT_TAG)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    tag, dev = args.seq_tag, args.device

    lr_train = logret(np.load(os.path.join(args.data_dir, f"true_S_{tag}.npy")))
    lr_val = logret(np.load(os.path.join(args.data_dir, f"true_S_val_{tag}.npy")))
    f_train = flat(lr_train)
    sd_train = per_asset_std(lr_train)

    med_base = float(np.median(min_dists(flat(lr_val), f_train, dev, args.chunk)))
    print(f"median NN(val -> train) = {med_base:.6f}   <- denominator (unchanged)")
    print()

    out = {
        "question": "is nn_ratio ~0.56 memorisation, or shrunken support?",
        "band": [BAND_LO, BAND_HI],
        "median_nn_val_to_train": med_base,
        "test_a_rescale_generated_up": {},
        "test_b_control_shrink_real_val": {},
    }

    # ---------------- TEST A -------------------------------------------------
    print("TEST A -- rescale each seed's log-returns UP to the real training std")
    print(f"{'seed':>5} {'vol ratio':>10} {'nn_ratio raw':>13} "
          f"{'nn_ratio rescaled':>18}  reading")
    print("-" * 72)
    for s in seeds:
        p = os.path.join(CSDI, "generated_paths", f"seed_{s}",
                         f"generated_paths_{tag}.npy")
        if not os.path.exists(p):
            print(f"{s:>5}   (bank missing -- skipped)")
            continue
        lr_gen = logret(np.load(p))
        vol_ratio = float((per_asset_std(lr_gen) / sd_train).mean())
        raw = float(np.median(min_dists(flat(lr_gen), f_train, dev, args.chunk))) / med_base
        resc = float(np.median(min_dists(flat(rescale_to(lr_gen, sd_train)),
                                         f_train, dev, args.chunk))) / med_base
        inband = BAND_LO <= resc <= BAND_HI
        # Three-way, not two-way. The band is an INTERVAL and the interesting
        # failure here is overshoot, not undershoot: landing ABOVE 1.000 means
        # the rescaled sample sits FARTHER from train than genuine held-out data
        # does, which is the opposite of memorisation. Collapsing that into
        # "still below band" would report the refutation as a confirmation.
        where = ("IN BAND" if inband else
                 "below band -- closer to train than real data" if resc < BAND_LO else
                 "ABOVE band -- farther from train than real val is")
        out["test_a_rescale_generated_up"][str(s)] = dict(
            vol_ratio=vol_ratio, nn_ratio_raw=raw, nn_ratio_rescaled=resc,
            rescaled_in_real_vs_real_band=bool(inband),
            rescaled_position=("in" if inband else "below" if resc < BAND_LO else "above"))
        print(f"{s:>5} {vol_ratio:>10.3f} {raw:>13.4f} {resc:>18.4f}  {where}")

    # ---------------- TEST B -------------------------------------------------
    # The control. Real data, shrunk. Any drop is caused by scale and nothing
    # else, because val is not derived from train in any way.
    print()
    print("TEST B (control) -- shrink the REAL val set by each seed's own vol ratio")
    print("   val cannot be a copy of train, so whatever drop appears here is")
    print("   produced by scale alone.")
    print(f"{'shrink':>8} {'val nn_ratio':>13}  note")
    print("-" * 52)
    factors = sorted({round(v["vol_ratio"], 3)
                      for v in out["test_a_rescale_generated_up"].values()})
    factors = [1.0] + factors
    for f in factors:
        shrunk = rescale_to(lr_val, sd_train * f)
        r = float(np.median(min_dists(flat(shrunk), f_train, dev, args.chunk))) / med_base
        out["test_b_control_shrink_real_val"][f"{f:.3f}"] = r
        note = "unshrunk real val (sanity: must be ~1.000)" if f == 1.0 else \
               "real data, CSDI's own scale"
        print(f"{f:>8.3f} {r:>13.4f}  {note}")

    # ---------------- verdict ------------------------------------------------
    a = out["test_a_rescale_generated_up"]
    b = out["test_b_control_shrink_real_val"]
    if a:
        mean_raw = float(np.mean([v["nn_ratio_raw"] for v in a.values()]))
        mean_res = float(np.mean([v["nn_ratio_rescaled"] for v in a.values()]))
        n_in = sum(v["rescaled_in_real_vs_real_band"] for v in a.values())
        ctrl = [(f, r) for f, r in b.items() if float(f) < 0.999]
        out["summary"] = dict(mean_nn_ratio_raw=mean_raw,
                              mean_nn_ratio_rescaled=mean_res,
                              n_seeds_in_band_after_rescale=n_in,
                              n_seeds=len(a))
        print()
        print(f"mean nn_ratio  raw {mean_raw:.4f}  ->  rescaled {mean_res:.4f}"
              f"   ({n_in}/{len(a)} seeds in the real-vs-real band)")
        shrink_reproduces = bool(ctrl) and all(r < 0.75 for _, r in ctrl)
        # One-sided ON PURPOSE. The worry is memorisation, i.e. being too CLOSE
        # to train; anything at or above BAND_LO has cleared that worry. But say
        # which side it landed on rather than claiming "restored to the band"
        # when it overshot -- overshoot is a different fact and a reader who is
        # checking will notice 1.08 is not inside [0.932, 1.000].
        rescue = mean_res >= BAND_LO
        overshoot = mean_res > BAND_HI
        if rescue and shrink_reproduces:
            landed = (f"OVERSHOOTS it ({mean_res:.4f} > {BAND_HI}), i.e. the "
                      "rescaled paths sit FARTHER from the training set than "
                      "genuine held-out data does") if overshoot else \
                     f"lands inside it ({mean_res:.4f})"
            verdict = ("SHRUNKEN SUPPORT. Rescaling to the real volatility "
                       f"moves the ratio from {mean_raw:.4f} to a value that "
                       f"{landed}; and shrinking genuine val data reproduces "
                       "CSDI's score on data that cannot be a copy. The low "
                       "nn_ratio is a restatement of the volatility failure, "
                       "not a second independent finding, and is NOT evidence "
                       "of copying.")
        elif rescue:
            verdict = ("Rescaling restores the ratio, but the control did not "
                       "reproduce the effect on real data -- scale explains "
                       "part of the gap. Report both, do not claim a clean "
                       "mechanism.")
        else:
            verdict = ("NOT explained by scale. Rescaling to the real "
                       "volatility leaves the ratio below the band, so the "
                       "memorisation reading survives the confound and must be "
                       "reported as a genuine second finding.")
        out["verdict"] = verdict
        print()
        print("VERDICT: " + verdict)

    dst = os.path.join(CSDI, "losses", "memorisation_scale_check.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
