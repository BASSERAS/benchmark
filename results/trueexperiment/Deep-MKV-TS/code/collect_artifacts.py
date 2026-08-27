"""Deep-MKV-TS on TrueDataset -- post-generation artefact collector and contract gate.

Section 4 of ``truedatasetguideline.md`` tells every method to run this, and the
reasoning is unchanged from the Heston campaign.

The guideline warns that a runner rewriting ``generation_time.csv`` in ``"w"``
mode from an in-memory list silently deletes the rows for seeds it did not
re-run. Deep-MKV-TS trains its seeds in **waves of 3 concurrent processes on 3
GPUs**, so an in-run writer would not merely lose rows -- live processes would
race on the same file. No seed process writes a shared path at all; the timing
table is rebuilt here, once, from the per-seed ``metadata.json`` files each
process wrote independently. Re-running one seed and then re-running this script
yields the same table either way.

Deep-MKV-TS reports a checkpoint chosen on validation, not the last optimizer
step, so ``selected_step`` is carried into every row: a timing table that did not
name it would describe a model that was never reported. ``ridge_lambda`` is
carried for the same reason -- it is re-selected on this dataset, and the run is
not reproducible without it.

Every field is recomputed from the ``.npy`` itself rather than copied forward, so
the audit record cannot drift from the array it describes -- which matters
precisely because the array is gitignored and the metadata is the only thing
tracked.

Why SEEDS is discovered, not hardcoded
--------------------------------------
The Heston campaign lost two of six seeds to a non-finite control and had to
hand-edit its seed list afterwards. Here the list is read off disk, and the
script states plainly how many it found versus how many were launched. A seed
that diverged therefore shows up as a smaller count with its number missing --
visible -- rather than as a list someone quietly renumbered into a clean run.

Contract checked per seed (section 4):
    shape (6144, 128, 8), dtype float64, S[:, 0, :] == 100.0 exactly,
    finite and strictly positive everywhere.
A violation is reported and the script exits non-zero. It does not repair. Wire
the exit code so a violation stops the chain BEFORE the metrics stage, which is
expensive and produces normal-looking numbers on a broken array.

Usage:
    /home/tbasseras/gpu-venv/bin/python \
        results/trueexperiment/Deep-MKV-TS/code/collect_artifacts.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)
GEN_ROOT = os.path.join(METHOD_DIR, "generated_paths")
WEIGHTS_DIR = os.path.join(METHOD_DIR, "weights")
SELECTION_DIR = os.path.join(HERE, "selection")
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")

EXPECTED_SHAPE = (6144, 128, 8)
S0 = 100.0
LAUNCHED_SEEDS = [0, 1, 2, 3, 4]
# every seed ran under taskset -c 0-7 / 8-15 / 16-23 with OMP_NUM_THREADS=8,
# three concurrent processes per wave on three GPUs
N_WORKERS = 8
BANK_NAME = "generated_paths_{}x{}x{}.npy".format(*EXPECTED_SHAPE)


def discovered_seeds() -> list[int]:
    if not os.path.isdir(GEN_ROOT):
        return []
    found = []
    for name in os.listdir(GEN_ROOT):
        match = re.fullmatch(r"seed_(\d+)", name)
        if match and os.path.exists(os.path.join(GEN_ROOT, name, BANK_NAME)):
            found.append(int(match.group(1)))
    return sorted(found)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # A COUNT, not a list: the pipeline knows how many seeds it launched, and
    # the seeds are always 0..N-1.  Seeds are still DISCOVERED from disk -- this
    # only says how many should have been found, so a diverged seed shows up as
    # an explicit missing index rather than as a quietly shorter run.
    parser.add_argument("--expect-seeds", type=int, default=len(LAUNCHED_SEEDS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    expected = list(range(int(args.expect_seeds)))
    seeds = discovered_seeds()
    rows, ok = [], True
    for seed in seeds:
        gen_dir = os.path.join(GEN_ROOT, f"seed_{seed}")
        npy = os.path.join(gen_dir, BANK_NAME)
        meta_path = os.path.join(gen_dir, "metadata.json")
        if not os.path.exists(meta_path):
            print(f"  seed {seed}: MISSING metadata.json -> FAIL")
            ok = False
            continue

        S = np.load(npy)
        with open(meta_path) as f:
            old = json.load(f)
        cfg = {}
        cfg_path = os.path.join(WEIGHTS_DIR, f"seed_{seed}_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        selection = {}
        sel_path = os.path.join(SELECTION_DIR, f"seed_{seed}_selection.json")
        if os.path.exists(sel_path):
            with open(sel_path) as f:
                selection = json.load(f)

        s0_exact = bool(np.all(S[:, 0, :] == S0))
        finite_pos = bool(np.isfinite(S).all() and (S > 0).all())
        good = (
            S.shape == EXPECTED_SHAPE
            and S.dtype == np.float64
            and s0_exact
            and finite_pos
        )
        ok &= good

        gen_sec = float(old.get("gen_time_sec", 0.0))
        chosen = selection.get("selected", {})
        meta = {
            "method": "Deep-MKV-TS",
            "dataset": old.get("dataset", "TrueDataset om_2022-07_N6144"),
            "seed": seed,
            "shape": list(S.shape),
            "dtype": str(S.dtype),
            "d": int(S.shape[2]),
            "dt": old.get("dt"),
            "S0": S0,
            "S0_exact": s0_exact,
            "S_min": float(S.min()),
            "S_max": float(S.max()),
            "all_finite_positive": finite_pos,
            "generated_mean": float(S.mean()),
            "generated_std": float(S.std()),
            "gen_time_sec": round(gen_sec, 1),
            "train_time_sec": float(
                old.get("train_time_sec", cfg.get("train_time_sec", 0.0))
            ),
            # Algorithm 1 counts OUTER ITERATIONS, not epochs: each step resamples
            # a fresh batch of paths and re-fits the Z-proxy, so there is no pass
            # over a fixed dataset to call an epoch.
            "n_steps": old.get("n_steps", cfg.get("n_steps")),
            "selected_step": old.get("selected_step", cfg.get("selected_step")),
            # The section-7 numbers, not a fit discrepancy: these are what chose
            # the reported checkpoint, so they are what the audit record must name.
            "selection_rule": cfg.get("selection_rule"),
            "selection_file": cfg.get("selection_file"),
            "selection_nn_ratio": chosen.get("nn_ratio", cfg.get("selection_nn_ratio")),
            "selection_vol_err_pct": chosen.get(
                "vol_err_pct", cfg.get("selection_vol_err_pct")
            ),
            "selection_corr_err": chosen.get("corr_err", cfg.get("selection_corr_err")),
            "fit_rule_would_have_chosen": selection.get("fit_rule_would_have_chosen"),
            "ridge_lambda": old.get("ridge_lambda", cfg.get("ridge_lambda")),
            "sigma_max": old.get("sigma_max", cfg.get("sigma_max")),
            "batch_size": old.get("batch_size", cfg.get("batch_size")),
            "n_parameters": old.get("n_parameters", cfg.get("n_parameters")),
            "num_steps": old.get("num_steps", cfg.get("seq_len")),
            "generation_seed": old.get("generation_seed"),
            "n_entries": int(S.size),
            "s0_rescaled": old.get("s0_rescaled", True),
            # float32 round-off on exp(log(100)), removed by the section 4 rescale.
            # Recorded so a reader can see the residual was numerical, not structural.
            "s0_max_abs_residual_before_rescale": old.get(
                "s0_max_abs_residual_before_rescale"
            ),
            "gpu": old.get("gpu", "A100-SXM4-80GB"),
            "date": old.get("date"),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        rows.append(
            {
                "seed": seed,
                "n_samples": S.shape[0],
                "T": S.shape[1],
                "d": S.shape[2],
                "n_steps": meta["n_steps"],
                "selected_step": meta["selected_step"],
                "ridge_lambda": meta["ridge_lambda"],
                "sigma_max": meta["sigma_max"],
                "batch_size": meta["batch_size"],
                "n_workers": N_WORKERS,
                "elapsed_sec": round(gen_sec, 2),
                "elapsed_min": round(gen_sec / 60.0, 2),
            }
        )
        residual = meta["s0_max_abs_residual_before_rescale"]
        residual_txt = "n/a" if residual is None else f"{float(residual):.3e}"
        print(
            f"  seed {seed}: {S.shape} price [{S.min():.2f}, {S.max():.2f}]  "
            f"S0 exact: {s0_exact}  finite&positive: {finite_pos}  "
            f"selected step: {meta['selected_step']}  "
            f"pre-rescale S0 residual: {residual_txt}  "
            f"gen {gen_sec:.1f}s  -> {'OK' if good else 'FAIL'}"
        )

    if not rows:
        raise SystemExit(
            "ABORT: no seeds found -- the chain is "
            "train_true.py -> select_checkpoint_true.py -> "
            "generate_bank_true.py -> this script."
        )

    os.makedirs(LOSSES_DIR, exist_ok=True)
    dst = os.path.join(LOSSES_DIR, "generation_time.csv")
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {dst}  ({len(rows)} rows)")

    missing = sorted(set(expected) - set(seeds))
    if missing:
        print(
            f"WARNING: launched seeds {expected} but banked {seeds}; "
            f"missing {missing}. Do NOT render the README until this is resolved -- "
            "either the seeds are rerun, or the README states plainly that they "
            "diverged. Renumbering them away would hide a stability failure."
        )
        ok = False

    print(
        "All seeds satisfy the section 4 contract."
        if ok
        else "CONTRACT VIOLATION -- see FAIL rows above."
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
