"""Deep-MKV-TS -- TrueDataset section-7 real-vs-real envelope screen.

Below this docstring this file is a BYTE-FOR-BYTE copy of
``CSDI/code/screen_envelope.py`` (source sha256 ea3ce789ef4374c8) and must stay
one. The envelope is a property of the DATASET VARIANT, not of the method, so
the methods have to be screened by identical arithmetic or their PASS/FAIL
verdicts are not comparable. If this screen ever needs to change, change it in
both places or in neither -- do not "improve" it here.

WHY THIS FILE EXISTS SEPARATELY FROM THE A-TABLE
-------------------------------------------------
``render_readme.py`` renders the A-table against ``real_floor/``. It never
computes the section-7 envelope, so without this script the two HARD
CONSTRAINTS of the selection criterion are simply absent from the
README -- the A-table would show the damage spread across A29/A30/A31 without
ever stating the verdict. That is the failure mode section 10.4 was written to
prevent: numbers present, conclusion missing.

WHAT A FAIL HERE MEANS FOR THIS METHOD SPECIFICALLY
----------------------------------------------------
Deep-MKV-TS already applies these same two ceilings INSIDE
``select_checkpoint_true.py``, which refuses to report a checkpoint that misses
them. A FAIL here is therefore not redundant with that gate -- it is strictly
more informative, because selection screens a small probe bank on the
VALIDATION split while this screens the full reported bank. A checkpoint that
passed selection and fails here means the probe was optimistic, and that gap is
itself the finding.

WHAT THE ENVELOPE IS
---------------------
The ceilings are the WORST value attained by genuine market data scored against
genuine market data on this build. A generated sample that clears them is
indistinguishable, on that statistic, from another honest slice of the same
market. They are properties of the DATASET VARIANT, not of any method, and are
recomputed by ``SBTS/code/calibrate_bandwidth_true.py`` whenever the variant
changes. They are quoted here, not re-derived, and the constants below must be
re-read from that script if the variant is ever rebuilt.

    vol_err  <= 24.59 %      annualised-vol relative error, mean over 8 assets
    corr_err <= 0.1843       mean |Delta| of the 28 off-diagonal correlations

THE ONE DEVIATION FROM THE SBTS DEFINITIONS, AND WHY IT IS EXACT
-----------------------------------------------------------------
``calibrate_bandwidth_true.py`` defines ``ann_vol`` with a ``sqrt(BARS_PER_YEAR)``
factor. That factor is a constant multiplying BOTH arguments of ``rel_err``,
which is scale-invariant, so it cancels identically. It is omitted here rather
than re-imported so this script has no dependency on the SBTS tree. This is an
algebraic identity, not an approximation -- do not "fix" it by adding the factor
back and do not use ``ann_vol`` from this file for anything except ``rel_err``.

WHY corr_err PASSING WHILE vol_err FAILS IS THE INTERESTING PART
-----------------------------------------------------------------
These two statistics fail independently. A model that had simply not learned
would miss both. A model that reproduces the cross-asset correlation structure
while halving the realised volatility has learned the SHAPE of the joint law and
lost its SCALE at the highest frequency -- it is smoothing the bar-to-bar
increments. Report the two numbers separately; an average over them is
meaningless and hides exactly the diagnosis.

Reads   ../../../dataset/TrueDataset/variants/om_2022-07_N6144/true_S_<tag>.npy
        ../generated_paths/seed_{i}/generated_paths_<tag>.npy
Writes  ../losses/envelope_screen.json

Usage:
    /home/tbasseras/.cc-venv/bin/python \
        results/trueexperiment/CSDI/code/screen_envelope.py
    ... --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)                     # results/trueexperiment/CSDI
TE = os.path.dirname(METHOD_DIR)                       # results/trueexperiment
REPO = os.path.dirname(os.path.dirname(TE))            # benchmark/
DEFAULT_VARIANT = os.path.join(
    REPO, "dataset", "TrueDataset", "variants", "om_2022-07_N6144")

# Section-7 ceilings for the om_2022-07_N6144 build. See module docstring: these
# are dataset properties, re-derived by SBTS/code/calibrate_bandwidth_true.py.
VOL_CEIL_PCT = 24.59
CORR_CEIL = 0.1843

ASSETS = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "LINK"]


def logret(S):
    return np.diff(np.log(S), axis=1)


def ann_vol(S):
    """Per-asset log-return std. The sqrt(BARS_PER_YEAR) factor of the SBTS
    definition is omitted because it cancels in rel_err -- see docstring."""
    return logret(S).std(axis=(0, 1))


def cross_corr(S):
    """Off-diagonal realised correlations of the log-returns, flattened."""
    C = np.corrcoef(logret(S).reshape(-1, S.shape[2]), rowvar=False)
    return C[np.triu_indices_from(C, k=1)]


def rel_err(a, ref):
    """Mean absolute relative error in percent."""
    return float(np.mean(np.abs(a - ref) / np.abs(ref)) * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_VARIANT)
    ap.add_argument("--seq-tag", default="6144x128x8")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    tag = args.seq_tag

    real = np.load(os.path.join(args.data_dir, f"true_S_{tag}.npy"))
    rv, rc = ann_vol(real), cross_corr(real)

    # The val split scored against train is the SCALE for reading the table: it
    # is two honest samples of the same market, so its distance from 1.000 is
    # what "indistinguishable" actually costs on this dataset.
    val_path = os.path.join(args.data_dir, f"true_S_val_{tag}.npy")
    val_ratio = None
    if os.path.exists(val_path):
        val = np.load(val_path)
        val_ratio = dict(
            vol_ratio_mean=float((ann_vol(val) / rv).mean()),
            vol_err_pct=rel_err(ann_vol(val), rv),
            corr_err=float(np.mean(np.abs(cross_corr(val) - rc))),
        )

    out = {
        "ceilings": {"vol_err_pct": VOL_CEIL_PCT, "corr_err": CORR_CEIL},
        "reference": "real train split, true_S_%s.npy" % tag,
        "real_val_vs_train": val_ratio,
        "seeds": {},
    }

    print(f"real train split {real.shape}")
    if val_ratio:
        print(f"real val vs train: vol_err {val_ratio['vol_err_pct']:.2f} %  "
              f"corr_err {val_ratio['corr_err']:.4f}  "
              f"(vol ratio {val_ratio['vol_ratio_mean']:.3f})")
    print()
    print(f"{'seed':>5} {'vol_err %':>10} {'corr_err':>9} {'vol ratio':>10} "
          f"{'finite>0':>9}  verdict")
    print("-" * 60)

    for seed in seeds:
        p = os.path.join(METHOD_DIR, "generated_paths", f"seed_{seed}",
                         f"generated_paths_{tag}.npy")
        if not os.path.exists(p):
            print(f"{seed:>5}   (bank not written yet -- skipped)")
            continue
        g = np.load(p)
        ok_fin = bool(np.isfinite(g).all() and (g > 0).all())
        gv = ann_vol(g)
        ve = rel_err(gv, rv)
        ce = float(np.mean(np.abs(cross_corr(g) - rc)))
        ratio = gv / rv
        passed = ok_fin and ve <= VOL_CEIL_PCT and ce <= CORR_CEIL
        out["seeds"][str(seed)] = dict(
            vol_err_pct=ve, corr_err=ce,
            vol_ratio_mean=float(ratio.mean()),
            vol_ratio_per_asset={a: float(r) for a, r in zip(ASSETS, ratio)},
            finite_and_positive=ok_fin,
            passes_vol=bool(ve <= VOL_CEIL_PCT),
            passes_corr=bool(ce <= CORR_CEIL),
            passes=bool(passed),
        )
        print(f"{seed:>5} {ve:>10.2f} {ce:>9.4f} {ratio.mean():>10.3f} "
              f"{str(ok_fin):>9}  {'PASS' if passed else 'FAIL'}")

    print("-" * 60)
    print(f"ceilings: vol_err <= {VOL_CEIL_PCT} %, corr_err <= {CORR_CEIL}")

    if out["seeds"]:
        n_fail = sum(1 for v in out["seeds"].values() if not v["passes"])
        vol_bad = [k for k, v in out["seeds"].items() if not v["passes_vol"]]
        corr_bad = [k for k, v in out["seeds"].items() if not v["passes_corr"]]
        out["n_seeds"] = len(out["seeds"])
        out["n_failing"] = n_fail
        out["seeds_failing_vol"] = vol_bad
        out["seeds_failing_corr"] = corr_bad
        if vol_bad and not corr_bad:
            # The diagnosis, not just the verdict. See module docstring.
            print()
            print("  ! vol_err fails on seeds " + ",".join(vol_bad)
                  + " while corr_err passes on every seed.")
            print("    The cross-asset correlation structure is reproduced and the")
            print("    realised volatility is not: the model has the shape of the")
            print("    joint law and not its scale at the highest frequency. Report")
            print("    these as two findings, never as one averaged score.")

    dst = os.path.join(METHOD_DIR, "losses", "envelope_screen.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
