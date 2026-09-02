#!/usr/bin/env python3
"""Generate the 8192-path bank for the FROZEN REFERENCE SDE, 5 seeds, d = 8.

What this "method" is
---------------------
It is Deep-MKV-TS with the learning switched off, and nothing else.

Algorithm 1 parameterises the diffusion as a correction to a reference SDE::

    Theta = eta * (sigma^ref)^{-1} + Zhat / sqrt(dt)

where ``Zhat`` is the network output.  Algorithm 1 **zero-initialises the final
layer of both output heads**, so a model that has been BUILT but never FITTED
emits ``Zhat == 0`` exactly -- not approximately, not to float tolerance, but
because the weight and the bias of the last layer are both the zero tensor and
``0 @ h + 0`` is zero for every hidden state ``h``.  With ``Zhat == 0`` the
control collapses to ``eta * (sigma^ref)^{-1}``, i.e. ``sigma == sigma^ref``,
and the sampler integrates the reference SDE itself.

So: build the model, skip ``fit``, sample.  That is the whole method.
``_assert_zero_control`` below checks the four tensors that make the claim true
rather than trusting this docstring, and records the measured maxima in
``metadata.json`` so a reader can audit the claim without rerunning anything.

Why the folder exists
---------------------
Every other row of the comparison table answers "how close to the truth did this
generator get".  None of them answers the prior question for Deep-MKV-TS:
**how much of its score is the learned control, and how much was already in the
reference SDE it starts from?**  Without this row, a good Deep-MKV-TS number is
unattributable -- it could be Algorithm 1 working, or it could be a well-fitted
``sigma^ref`` carrying the result while the control does nothing.

This is a different question from ``perfect_recovery/``.  That folder draws from
the TRUE Heston SDE with the true per-asset parameters and gives the
independent-draw floor: the best any generator could do.  This folder draws from
the FITTED reference SDE and gives the starting line: what Deep-MKV-TS scores
before a single gradient step.  Floor and starting line bracket the method from
opposite sides.

Relation to ``Deep-MKV-TS/code/null_control_baseline.py``
--------------------------------------------------------
Same object, different scope.  That script measures ONE number -- the discrepancy
functional on the validation split -- to decide whether the ridge sweep had
selected a null model; it answered no, 0.300095 for the null against 0.023836
for the selected candidate.  This file runs the full 5-seed benchmark pipeline on
the same object so the answer lands in the comparison table in the same units as
every other method.

Why the generation seeds are shared with Deep-MKV-TS
----------------------------------------------------
``GEN_SEED_BASE`` is imported from ``Deep-MKV-TS/code/run_all_multiasset.py``
rather than redefined, so seed ``i`` here draws the SAME Brownian stream as seed
``i`` there.  The two banks then differ ONLY by the learned control, which makes
the comparison paired rather than merely independent -- and a paired difference
is resolvable at 5 seeds where an unpaired one frequently is not.

Caveat, stated because it is easy to miss: the Deep-MKV-TS campaign runs seeds
{0, 1, 2, 4, 5} after seed 3 diverged at step 2, while this folder runs the
conventional {0, 1, 2, 3, 4} that SBTS, LS4, CSDI and ``perfect_recovery`` use.
Seeds 0, 1, 2 and 4 are therefore genuinely paired; seed 3 here has no
Deep-MKV-TS counterpart and seed 5 there has none here.  Both table rows are
5-seed means regardless, which is what the comparison consumes.

The S0 contract
---------------
Not reimplemented.  ``S0``, ``EXPECTED_SHAPE`` and ``GEN_SEED_BASE`` are imported
from ``Deep-MKV-TS/code/run_all_multiasset.py``, and the two-line rescale
prescribed by section 4 of ``MULTIASSET_GUIDELINE.md`` is applied identically::

    prices = prices * (S0 / prices[:, 0:1, :])
    prices[:, 0, :] = S0

Multiplying a whole path by a constant is exactly a shift of the log-price level,
so every log-return is bit-identical before and after.  The residual it removes
is float32 round-off in ``exp(log(100))`` -- measured at 7.6e-6 -- and is recorded
as ``s0_max_abs_residual_before_rescale`` rather than discarded.

No GPU required
---------------
Sampling 8192 paths costs about 2 minutes on a SINGLE core, because the workload
is a sequential 251-step Euler loop over tiny ``(8192, 8, 8)`` tensors and is
kernel-launch-bound rather than compute-bound.  ``--device cpu`` is the default
for exactly that reason: the GPUs are busy with the training campaign and this
job does not need one.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    D=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS/code
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
    PYTHONPATH="$R/src:$R/experiments:$D" taskset -c 15 \
    /home/tbasseras/gpu-venv/bin/python generate_reference_multiasset.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
METHOD_ROOT = HERE.parent
DMKV_CODE = METHOD_ROOT.parent / "Deep-MKV-TS" / "code"
REFERENCE = Path("/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference")
for _path in (HERE, DMKV_CODE, REFERENCE / "src", REFERENCE / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402

import eigh_fallback  # noqa: E402
import train_multiasset as tm  # noqa: E402
from fit_reference_multiasset import DT, STATE_DIM  # noqa: E402

# Imported, never redefined: a local copy would let this bank drift out of the
# paired relationship with the Deep-MKV-TS bank without anything failing loudly.
from run_all_multiasset import EXPECTED_SHAPE, GEN_SEED_BASE, S0  # noqa: E402

eigh_fallback.install()

METHOD_NAME = "reference"

# The final layer of each output head. Algorithm 1 zero-initialises exactly
# these; if any becomes non-zero the sampler is no longer the reference SDE and
# this whole folder is mislabelled, so they are checked rather than assumed.
#   expected_adjoint_next_head.2        -> (8, 96),  output dim d = 8
#   expected_adjoint_noise_next_head.2  -> (64, 96), output dim d^2 = 64, i.e. Zhat
ZERO_INIT_TENSORS = (
    "expected_adjoint_next_head.2.weight",
    "expected_adjoint_next_head.2.bias",
    "expected_adjoint_noise_next_head.2.weight",
    "expected_adjoint_noise_next_head.2.bias",
)


def _assert_zero_control(model) -> dict[str, float]:
    """Verify Zhat == 0 by inspecting the weights, and return the measured maxima.

    The claim "this bank is a draw from the reference SDE" rests entirely on the
    final layer of both heads being the zero tensor.  That is a property of the
    freshly built model which a future change to ``build_model`` -- a different
    initialiser, an added bias, an accidental checkpoint load -- would silently
    break, and the resulting bank would look completely normal while actually
    being a draw from some half-trained model.  Checking costs microseconds; not
    checking costs a benchmark row that is wrong in a way no downstream gate can
    detect.

    A zero weight AND a zero bias in the last layer is sufficient: the head's
    output is ``W @ h + b`` and is therefore identically zero for every hidden
    state ``h``, regardless of what the GRU below it computes.
    """
    state = model.checkpoint_state()["network_state_dict"]
    missing = [k for k in ZERO_INIT_TENSORS if k not in state]
    if missing:
        raise SystemExit(
            f"ABORT: expected zero-initialised tensors absent from the network: "
            f"{missing}. build_model() no longer matches the architecture this "
            f"file assumes, so 'Zhat == 0' can no longer be verified and the "
            f"generated bank would not be the reference SDE."
        )
    maxima = {k: float(state[k].abs().max()) for k in ZERO_INIT_TENSORS}
    nonzero = {k: v for k, v in maxima.items() if v != 0.0}
    if nonzero:
        raise SystemExit(
            f"ABORT: the control is not zero -- {nonzero}. The model appears to "
            f"have been fitted or restored from a checkpoint. This file must "
            f"sample an UNFITTED model; that is the entire definition of the "
            f"'reference' method."
        )
    return maxima


def generate_one(seed: int, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)

    # Derive the horizon from the training bank rather than hard-coding 251, so
    # a regenerated dataset cannot silently desynchronise this from Deep-MKV-TS.
    probe = tm.load_log_prices(num_paths=1, device="cpu")
    num_steps = int(probe.shape[1]) - 1
    del probe
    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)

    model = tm.build_model(
        grid=grid, device=device, seed=int(seed), kernel_steps=num_steps
    )
    # NO fit(). NO load_checkpoint_state(). That omission is the method.
    zhat_maxima = _assert_zero_control(model)

    x0 = torch.full((1, STATE_DIM), math.log(S0), device=device, dtype=model.dtype)
    started = time.perf_counter()
    with torch.no_grad():
        log_paths = model.sample(
            num_paths=int(args.num_paths),
            x0=x0,
            seed=GEN_SEED_BASE + int(seed),
        ).paths
    prices = torch.exp(log_paths.double()).cpu().numpy()
    gen_sec = time.perf_counter() - started

    if prices.shape != EXPECTED_SHAPE:
        raise SystemExit(
            f"ABORT: seed {seed} produced {prices.shape}, want {EXPECTED_SHAPE}"
        )
    if not np.isfinite(prices).all():
        raise SystemExit(f"ABORT: seed {seed} produced non-finite prices")
    if not (prices > 0.0).all():
        raise SystemExit(f"ABORT: seed {seed} produced non-positive prices")

    # Section 4 contract, identical to Deep-MKV-TS/code/run_all_multiasset.py.
    residual = float(np.abs(prices[:, 0, :] - S0).max())
    prices = prices * (S0 / prices[:, 0:1, :])
    prices[:, 0, :] = S0
    s0_exact = bool(np.all(prices[:, 0, :] == S0))
    if not s0_exact:
        raise SystemExit(f"ABORT: seed {seed} failed S[:, 0, :] == 100.0 after rescale")

    gen_dir = METHOD_ROOT / "generated_paths" / f"seed_{int(seed)}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    npy_path = gen_dir / "generated_paths_8192x252x8.npy"
    np.save(npy_path, prices)

    metadata = {
        "method": METHOD_NAME,
        "seed": int(seed),
        "shape": list(prices.shape),
        "dtype": str(prices.dtype),
        "d": int(prices.shape[2]),
        "S0": S0,
        "S0_exact": s0_exact,
        "S_min": float(prices.min()),
        "S_max": float(prices.max()),
        "all_finite_positive": True,
        "generated_mean": float(prices.mean()),
        "generated_std": float(prices.std()),
        "gen_time_sec": round(gen_sec, 1),
        "train_time_sec": 0.0,
        "generation_seed": GEN_SEED_BASE + int(seed),
        "s0_rescaled": True,
        "s0_max_abs_residual_before_rescale": residual,
        "zero_control_verified": True,
        "zhat_head_max_abs": zhat_maxima,
        "n_parameters": int(sum(p.numel() for p in model.network.parameters())),
        "n_trained_parameters": 0,
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "note": (
            "Frozen reference SDE. The model was BUILT but never FITTED; both "
            "output heads have a zero final layer, so Zhat == 0 exactly and the "
            "control reduces to eta * (sigma^ref)^-1, i.e. sigma == sigma^ref. "
            "n_parameters counts the architecture Deep-MKV-TS would train; "
            "n_trained_parameters is 0 because no gradient step was taken. "
            "Generation seeds are shared with Deep-MKV-TS/run_all_multiasset.py, "
            "so seeds 0, 1, 2 and 4 are paired with that method's banks."
        ),
        "date": date.today().isoformat(),
    }
    (gen_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"  seed {seed}: {prices.shape} price "
        f"[{prices.min():.2f}, {prices.max():.2f}]  S0 exact: {s0_exact}  "
        f"pre-rescale residual {residual:.3e}  "
        f"Zhat max {max(zhat_maxima.values()):g}  gen {gen_sec:.1f}s",
        flush=True,
    )
    print(f"  [out] {npy_path}", flush=True)

    del model, log_paths
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def _write_generation_time(rows: list[dict]) -> Path:
    """Write losses/generation_time.csv and return its path.

    Deep-MKV-TS rebuilds this file with ``collect_artifacts.py`` because it also
    has to gate a training-time contract.  There is no training here, so the CSV
    is written directly -- but it is rewritten after EVERY seed rather than once
    at the end, so an interrupted run still leaves a record of what completed.
    """
    losses_dir = METHOD_ROOT / "losses"
    losses_dir.mkdir(parents=True, exist_ok=True)
    path = losses_dir / "generation_time.csv"
    fields = ["seed", "n_samples", "T", "d", "device", "elapsed_sec", "elapsed_min"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for meta in rows:
            writer.writerow(
                {
                    "seed": meta["seed"],
                    "n_samples": meta["shape"][0],
                    "T": meta["shape"][1],
                    "d": meta["d"],
                    "device": meta["device"],
                    "elapsed_sec": round(meta["gen_time_sec"], 2),
                    "elapsed_min": round(meta["gen_time_sec"] / 60.0, 2),
                }
            )
    return path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-paths", type=int, default=8192)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not args.seeds:
        raise SystemExit("ABORT: --seeds is empty, nothing to generate")
    print(
        f"reference SDE bank -- seeds {args.seeds}  device={args.device}  "
        f"paths={args.num_paths}",
        flush=True,
    )
    rows: list[dict] = []
    csv_path = METHOD_ROOT / "losses" / "generation_time.csv"
    started = time.perf_counter()
    for seed in args.seeds:
        rows.append(generate_one(int(seed), args))
        # Rewritten after every seed so a crash still leaves a usable record.
        csv_path = _write_generation_time(rows)
    print(
        f"\nAll {len(rows)} seeds done in "
        f"{(time.perf_counter() - started) / 60:.1f} min",
        flush=True,
    )
    print(f"[out] {csv_path}", flush=True)
    print(
        "\nNext: metrics/compute_all_multiasset.py --method reference "
        "--dataset HestonMultiAsset --seeds 5",
        flush=True,
    )


if __name__ == "__main__":
    main()
