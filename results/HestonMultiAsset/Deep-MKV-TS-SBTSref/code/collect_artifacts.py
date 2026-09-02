"""Deep-MKV-TS-SBTSref d = 8 -- post-training artefact collector and contract check.

Adapted from ``Deep-MKV-TS/code/collect_artifacts.py``; MULTIASSET_GUIDELINE.md
section 4 tells every method to copy it, and the reason applies here unchanged.

The guideline warns that a runner rewriting ``generation_time.csv`` in ``"w"`` mode
from an in-memory list silently deletes the rows for seeds it did not re-run. This
method trains its 5 seeds as **two concurrent chains on two GPUs** (seeds 0/2/4 on
GPU 0, seeds 1/3 on GPU 1), so an in-run writer would not merely lose rows -- live
processes would race on the same file. No seed process writes a shared path at all;
the timing table is rebuilt here, once, from the per-seed ``metadata.json`` files
each process wrote independently. Re-running one seed and then re-running this
script yields the same 5-row table either way.

This method reports a checkpoint chosen on validation, not the last optimizer step,
so ``selected_step`` is carried into every row: a timing table that did not name it
would describe a model that was never reported.

What this file carries that the sibling's does not
--------------------------------------------------
The sibling's reference drift is fitted once and frozen into the weights. Here the
reference is an SBTS kernel that is NOT stored in the checkpoint -- it is rebuilt at
load time from ``(h, markov_order, npi, weight_grad_mode)`` plus the train-split
price bank. Two runs with identical weights and different ``h`` are different
models. Those four knobs are therefore promoted into ``generation_time.csv`` itself
rather than being left to the config files, because the CSV is what gets read back
when the numbers are questioned.

``weight_grad_mode`` in particular is a CORRECTNESS setting, not a tuned one:
``detached`` drops ``d b_ref / d x`` from the backward pass entirely, and the method
is specified to carry it. A row that does not say which mode produced it cannot be
audited. See SWEEP.md.

Every field is recomputed from the ``.npy`` itself rather than copied forward, so
the audit record cannot drift from the array it describes -- which matters
precisely because the array is gitignored (132 MB, over GitHub's 100 MB hard limit)
and the metadata is the only thing tracked.

Contract checked per seed (GUIDELINE section 4):
    shape (8192, 252, 8), dtype float64, S[:, 0, :] == 100.0 exactly,
    finite and strictly positive everywhere.
A violation is reported and the script exits non-zero. It does not repair. Wire the
exit code so a violation stops the chain *before* compute_all_multiasset.py, which
costs 55 minutes and produces normal-looking numbers on a broken array.

Divergence is not laundered
---------------------------
``SEEDS`` is the honest ``[0, 1, 2, 3, 4]``. A seed with missing or non-finite
artefacts is reported as FAIL and the script exits non-zero; it is not dropped from
the list and the survivors are not renumbered. A five-row table that was really a
three-seed campaign is the specific lie this guard exists to prevent.

Usage:
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code/collect_artifacts.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)
GEN_ROOT = os.path.join(METHOD_DIR, "generated_paths")
WEIGHTS_DIR = os.path.join(METHOD_DIR, "weights")
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")

SEEDS = [0, 1, 2, 3, 4]
EXPECTED_SHAPE = (8192, 252, 8)
S0 = 100.0
# every seed ran under taskset -c 0-7 / 8-15 with OMP_NUM_THREADS=8, two
# concurrent chains on two GPUs
N_WORKERS = 8


def main():
    rows, ok = [], True
    for seed in SEEDS:
        gen_dir = os.path.join(GEN_ROOT, f"seed_{seed}")
        npy = os.path.join(gen_dir, "generated_paths_8192x252x8.npy")
        meta_path = os.path.join(gen_dir, "metadata.json")
        if not (os.path.exists(npy) and os.path.exists(meta_path)):
            print(f"  seed {seed}: MISSING artefacts -> FAIL")
            ok = False
            continue

        S = np.load(npy)
        with open(meta_path) as f:
            old = json.load(f)
        cfg_path = os.path.join(WEIGHTS_DIR, f"seed_{seed}_config.json")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)

        s0_exact = bool(np.all(S[:, 0, :] == S0))
        finite_pos = bool(np.isfinite(S).all() and (S > 0).all())
        good = (S.shape == EXPECTED_SHAPE and S.dtype == np.float64
                and s0_exact and finite_pos)
        ok &= good

        gen_sec = float(old.get("gen_time_sec", 0.0))
        meta = {
            "method": "Deep-MKV-TS-SBTSref",
            "seed": seed,
            "shape": list(S.shape),
            "dtype": str(S.dtype),
            "d": int(S.shape[2]),
            "S0": S0,
            "S0_exact": s0_exact,
            "S_min": float(S.min()),
            "S_max": float(S.max()),
            "all_finite_positive": finite_pos,
            "generated_mean": float(S.mean()),
            "generated_std": float(S.std()),
            "gen_time_sec": round(gen_sec, 1),
            "train_time_sec": float(
                old.get("train_time_sec", cfg.get("train_time_sec", 0.0))),
            # Algorithm 1 counts OUTER ITERATIONS, not epochs: each step resamples a
            # fresh batch of paths and re-fits the Z-proxy, so there is no pass over
            # a fixed dataset to call an epoch.
            "n_steps": old.get("n_steps", cfg.get("n_steps")),
            "selected_step": old.get("selected_step", cfg.get("selected_step")),
            "selection_file": cfg.get("selection_file"),
            "selection_val_discrepancy": cfg.get("selection_val_discrepancy"),
            "ridge_lambda": old.get("ridge_lambda", cfg.get("ridge_lambda")),
            "batch_size": old.get("batch_size", cfg.get("batch_size")),
            "n_parameters": old.get("n_parameters", cfg.get("n_parameters")),
            "num_steps": old.get("num_steps", cfg.get("seq_len")),
            # Chunked generation: the bank is a function of (base, seed, chunk),
            # so the chunk size is part of the reproduction recipe, not a detail.
            "generation_seed_base": old.get("generation_seed_base"),
            "generation_seed_first_chunk": old.get("generation_seed_first_chunk"),
            "generation_chunk": old.get("generation_chunk"),
            "n_entries": old.get("n_entries", int(S.size)),
            "s0_rescaled": old.get("s0_rescaled", True),
            # float32 round-off on exp(log(100)), removed by the section 4 rescale.
            # Recorded so a reader can see the residual was numerical, not structural.
            "s0_max_abs_residual_before_rescale": old.get(
                "s0_max_abs_residual_before_rescale"),
            # ---- the reference, which is NOT in the checkpoint ---------------- #
            "reference_kernel": cfg.get("reference_kernel"),
            "reference_family": old.get(
                "reference_family", cfg.get("reference_family")),
            "reference_h": old.get("reference_h", cfg.get("reference_h")),
            "reference_markov_order": old.get(
                "reference_markov_order", cfg.get("reference_markov_order")),
            "reference_npi": old.get("reference_npi", cfg.get("reference_npi")),
            "reference_weight_grad_mode": old.get(
                "reference_weight_grad_mode", cfg.get("reference_weight_grad_mode")),
            "reference_bank_paths": cfg.get("reference_bank_paths"),
            "reference_bank_split": cfg.get("reference_bank_split"),
            "reference_sigma_constant": cfg.get("reference_sigma_constant"),
            "allow_drift_correction": old.get(
                "allow_drift_correction", cfg.get("allow_drift_correction")),
            "eigh_fallback": cfg.get("eigh_fallback"),
            "gpu": old.get("gpu", "A100-SXM4-80GB"),
            "date": old.get("date"),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        rows.append({
            "seed": seed, "n_samples": S.shape[0], "T": S.shape[1], "d": S.shape[2],
            "n_steps": meta["n_steps"], "selected_step": meta["selected_step"],
            "ridge_lambda": meta["ridge_lambda"],
            "h": meta["reference_h"],
            "markov_order": meta["reference_markov_order"],
            "npi": meta["reference_npi"],
            "weight_grad_mode": meta["reference_weight_grad_mode"],
            "batch_size": meta["batch_size"],
            "n_workers": N_WORKERS,
            "elapsed_sec": round(gen_sec, 2),
            "elapsed_min": round(gen_sec / 60.0, 2),
        })
        residual = meta["s0_max_abs_residual_before_rescale"]
        residual_txt = "n/a" if residual is None else f"{float(residual):.3e}"
        print(f"  seed {seed}: {S.shape} price [{S.min():.2f}, {S.max():.2f}]  "
              f"S0 exact: {s0_exact}  finite&positive: {finite_pos}  "
              f"selected step: {meta['selected_step']}  "
              f"pre-rescale S0 residual: {residual_txt}  "
              f"gen {gen_sec:.1f}s  -> {'OK' if good else 'FAIL'}")

    if not rows:
        raise SystemExit(
            "ABORT: no seeds found -- the chain is "
            "train_multiasset.py -> select_checkpoint_multiasset.py -> "
            "run_all_multiasset.py -> this script."
        )

    # A single campaign must not mix reference settings. If it did, the five banks
    # would not be five samples of one method and averaging them would be
    # meaningless -- so this is a hard stop, not a warning.
    knobs = {
        (r["h"], r["markov_order"], r["npi"], r["weight_grad_mode"]) for r in rows
    }
    if len(knobs) > 1:
        print("\nABORT: seeds do not share one reference configuration:")
        for row in rows:
            print(f"  seed {row['seed']}: h={row['h']} K={row['markov_order']} "
                  f"npi={row['npi']} grad={row['weight_grad_mode']}")
        sys.exit(1)

    os.makedirs(LOSSES_DIR, exist_ok=True)
    dst = os.path.join(LOSSES_DIR, "generation_time.csv")
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {dst}  ({len(rows)} rows)")

    if len(rows) != len(SEEDS):
        print(f"WARNING: {len(rows)} rows, expected {len(SEEDS)} -- "
              f"do NOT render the README until this is {len(SEEDS)}.")
        ok = False

    print("All seeds satisfy the section 4 contract." if ok
          else "CONTRACT VIOLATION -- see FAIL rows above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
