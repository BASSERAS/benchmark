#!/usr/bin/env python3
"""
gen_perfect_recovery_multiasset.py
──────────────────────────────────
Materialise the **independent-draw floor** for the d = 8 multi-asset Heston
dataset (GUIDELINE §5.4).

The floor is the score a *perfect* generator would obtain: one whose output is
distributionally identical to the true process. We realise it honestly by
drawing 5 fresh, INDEPENDENT sets of 8192 paths from the *same* SDE with the
*same* frozen per-asset parameters, using brand-new RNG seeds
(`seed = IND_SEED_BASE + i` = 1000…1004), and then scoring those draws with the
exact same metric code every method is scored with.

Because both sides come from the identical law but independent randomness, every
metric lands on its irreducible finite-sample noise floor. That floor is
**non-zero everywhere** — two independent 8192-path draws never produce identical
histograms, ACFs, quantiles, covariance matrices or moments.

This script only WRITES THE PATHS. The metrics are then produced by the single
shared driver, so the floor and every method go through byte-identical code:

    /home/tbasseras/gpu-venv/bin/python metrics/compute_all_multiasset.py \
        --method perfect_recovery --subdir HestonMultiAsset

which reads
    methods/perfect_recovery/HestonMultiAsset/generated_paths/seed_{i}/generated_paths_*.npy
and writes
    results/HestonMultiAsset/perfect_recovery/

Not a permutation baseline
──────────────────────────
Row-shuffling the test set would preserve every column-wise statistic exactly
and collapse most metrics to 0 — a misleading target. An independent
re-simulation of the SDE is the honest lower bound a real generator can reach.

Not method-specific
───────────────────
The floor depends only on the test set + the Heston law + the protocol, so the
same `Perfect floor` column is valid in every per-method README under
`results/HestonMultiAsset/`.

Usage
─────
    /home/tbasseras/sbts-venv/bin/python metrics/gen_perfect_recovery_multiasset.py
    /home/tbasseras/sbts-venv/bin/python metrics/gen_perfect_recovery_multiasset.py --seeds 5
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(REPO, "dataset", "HestonMultiAsset")
sys.path.insert(0, DATASET_DIR)

from generate_heston_multiasset import (  # noqa: E402
    D, N_SAMPLES, SEQ_LEN,
    generate_multiasset_heston, sample_parameters, _params_to_json,
)

IND_SEED_BASE = 1000                     # GUIDELINE §5.4: 1000 + i, i = 0..4
OUT_ROOT = os.path.join(REPO, "methods", "perfect_recovery", "HestonMultiAsset")


def _digest(obj) -> str:
    """Stable hash of the per-asset parameter block."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()
    ).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    # ── 1. reproduce the frozen parameters and PROVE they still match ────────
    params = sample_parameters()
    fresh = _params_to_json(params)

    committed_path = os.path.join(DATASET_DIR, "parameters.json")
    with open(committed_path) as fh:
        committed = json.load(fh)

    h_fresh = _digest(fresh["per_asset"])
    h_commit = _digest(committed["per_asset"])
    if h_fresh != h_commit:
        raise SystemExit(
            "ABORT: sample_parameters() no longer reproduces the committed\n"
            f"  {committed_path}\n"
            f"  fresh per_asset digest    = {h_fresh}\n"
            f"  committed per_asset digest= {h_commit}\n"
            "The independent draws would come from a DIFFERENT law than the\n"
            "train/test/disc splits, which would silently invalidate the floor.\n"
            "numpy's Generator bit-stream is not guaranteed stable across\n"
            "versions -- check the numpy version this env pins."
        )
    if not np.allclose(np.asarray(fresh["sigma_s"]),
                       np.asarray(committed["sigma_s"])):
        raise SystemExit("ABORT: sigma_s mismatch vs committed parameters.json")

    print(f"parameter check OK  (per_asset digest {h_fresh})")
    print(f"d={D}  N={N_SAMPLES}  T={SEQ_LEN}  seeds={IND_SEED_BASE}"
          f"..{IND_SEED_BASE + args.seeds - 1}")

    # ── 2. draw and persist ──────────────────────────────────────────────────
    for i in range(args.seeds):
        draw_seed = IND_SEED_BASE + i
        out_dir = os.path.join(OUT_ROOT, "generated_paths", f"seed_{i}")
        os.makedirs(out_dir, exist_ok=True)

        t0 = time.time()
        S, v = generate_multiasset_heston(seed=draw_seed, params=params,
                                          n_samples=N_SAMPLES, seq_len=SEQ_LEN)
        dt_sec = time.time() - t0

        assert S.shape == (N_SAMPLES, SEQ_LEN, D), S.shape
        assert S.dtype == np.float64, S.dtype
        assert np.all(np.isfinite(S)) and np.all(S > 0.0), "non-finite / non-positive prices"

        tag = f"{N_SAMPLES}x{SEQ_LEN}x{D}"
        np.save(os.path.join(out_dir, f"generated_paths_{tag}.npy"), S)
        np.save(os.path.join(out_dir, f"variance_{tag}.npy"), v)

        meta = {
            "kind": "independent-draw perfect-recovery floor",
            "note": "fresh true-Heston draw, identical frozen parameters, "
                    "independent RNG seed -- NOT a permutation of the real data",
            "guideline": "GUIDELINE 5.4",
            "model": fresh["model"],
            "parameter_source": fresh["parameter_source"],
            "per_asset_digest": h_fresh,
            "run_seed": i,
            "draw_seed": draw_seed,
            "ind_seed_base": IND_SEED_BASE,
            "shape": list(S.shape),
            "dtype": str(S.dtype),
            "S0": fresh["S0"],
            "dt": fresh["dt"],
            "S_min": float(S.min()),
            "S_max": float(S.max()),
            "S0_exact": bool(np.all(S[:, 0, :] == fresh["S0"])),
            "v_min": float(v.min()),
            "generation_time_sec": round(dt_sec, 1),
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

        print(f"  seed {i} (draw {draw_seed}): S {S.shape} "
              f"[{S.min():.2f}, {S.max():.2f}]  S0 exact={meta['S0_exact']}  "
              f"{dt_sec:.1f}s")

    print(f"\nwrote {args.seeds} independent draws to")
    print(f"  {os.path.join(OUT_ROOT, 'generated_paths')}")
    print("\nnext:")
    print("  /home/tbasseras/gpu-venv/bin/python metrics/compute_all_multiasset.py \\")
    print("      --method perfect_recovery --subdir HestonMultiAsset")


if __name__ == "__main__":
    main()
