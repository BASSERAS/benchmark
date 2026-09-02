#!/usr/bin/env python3
"""Generate the 8192-path bank for each seed from its SELECTED checkpoint.

Order of operations
-------------------
    train_multiasset.py             -> runs/seed_i/training_checkpoints/step_*.pt
    select_checkpoint_multiasset.py -> weights/seed_i_model.pt  (argmin on val)
    run_all_multiasset.py           -> generated_paths/seed_i/*.npy   <- THIS FILE
    collect_artifacts.py            -> losses/generation_time.csv + contract gate

Generation is deliberately NOT folded into the trainer the way LS4 and CSDI do
it.  Those two report their final model; Deep-MKV-TS reports a checkpoint chosen
on validation, which is not known until training has finished, so the bank
cannot be sampled from an in-memory model at the end of the fit.

Why the model is rebuilt through ``select_checkpoint_multiasset._rebuild``
--------------------------------------------------------------------------
This method's model is not fully described by its weights: the SBTS reference
kernel carries the price bank, ``h``, ``markov_order``, ``npi`` and the gradient
mode, and the control carries the constant ``sigma_ref``.  Reconstructing that
from module-level defaults would silently generate from a DIFFERENT reference
than the one trained against, the moment those defaults move.  ``_rebuild``
reads the knobs back out of the run's own ``config.json``, and importing it
rather than copying it means selection and generation cannot drift apart.

Why the full checkpoint is loaded
---------------------------------
``DiscreteMPModel.checkpoint_state()`` documents that the timewise target scales
and the noise-target baseline feed the adjoint moments passed to the control
map.  A bare ``network.state_dict()`` therefore does NOT reproduce a fitted
model's rollouts.  ``select_checkpoint_multiasset.py`` writes the full payload
into ``weights/seed_i_model.pt`` and this file restores it with
``load_checkpoint_state``.

Why generation is chunked
-------------------------
Sampling is not memory-flat in the number of paths.  At every step the SBTS
kernel forms a ``(paths, bank)`` log-weight matrix, so an 8192-path bank against
an 8192-path reference bank is a 67M-entry intermediate PER STEP -- a 32x jump
over the 256-path training batch, at a point in the pipeline where an OOM costs
a whole seed rather than one optimiser step.  Chunking at 1024 matches the
sweep's scoring chunk and holds that intermediate at 8.4M entries.

The batched ``eigh`` is NOT the binding constraint here: sampling projects Theta
one date at a time, so its batch is ``(paths, d, d)`` -- 8192 leading matrices,
well under the measured cuSOLVER knee of 32768.  The knee bites in
``running_cost``, which eighs a ``(B, N, d, d)`` tensor.

Chunking makes the bank a function of ``(GEN_SEED_BASE, seed, chunk)``, not of
``(GEN_SEED_BASE, seed)`` alone, because each chunk draws its own stream.  The
chunk size is therefore recorded in ``metadata.json``; changing it changes the
bank, and that has to be legible rather than silent.

The S0 contract
---------------
The generator integrates LOG prices from ``x0 = log(100)`` exactly, so it meets
``S[:, 0, :] == 100.0`` natively up to float round-off -- but the model runs in
float32, where ``exp(log(100))`` is not bit-exactly 100.  Section 4 of
``MULTIASSET_GUIDELINE.md`` prescribes the fix, applied here in float64:

    Xg = Xg * (100.0 / Xg[:, 0:1, :])
    Xg[:, 0, :] = 100.0

Multiplying a whole path by a constant is exactly a shift of the log-price
level, so every log-return is bit-identical before and after.  ``s0_rescaled``
is set to ``true`` in ``metadata.json`` so a reader can tell a method that met
the contract natively from one that was rescaled onto it.  The residual the
rescale removes is recorded as ``s0_max_abs_residual_before_rescale``; it is
float32 round-off, not a structural miss.

Divergence is NOT laundered
---------------------------
The default seed list is the honest ``0 1 2 3 4``.  A seed whose bank comes out
non-finite ABORTS the run naming that seed.  It is not replaced by a spare and
the surviving seeds are not renumbered: a stability failure is a result.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
    PYTHONPATH="$R/src:$R/experiments" taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python run_all_multiasset.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

# _rebuild / _run_config are imported rather than duplicated so that the model
# generation samples from is provably the model selection scored.  Importing
# also fixes sys.path for the frozen reference package as a side effect.
from select_checkpoint_multiasset import METHOD_ROOT, _rebuild, _run_config

import train_multiasset as tm

S0 = 100.0
EXPECTED_SHAPE = (8192, 252, 8)
# Generation seeds are disjoint from training seeds (0-4) and from the sweep's
# scoring seed (20260831), so no bank is ever drawn from a stream the model was
# fitted or selected on.
GEN_SEED_BASE = 90000
DEFAULT_CHUNK = 1024


def _sample_bank(
    model,
    *,
    seed: int,
    num_paths: int,
    chunk: int,
    device: torch.device,
) -> torch.Tensor:
    """Log-price paths ``(num_paths, T + 1, d)``, drawn in chunks.

    Each chunk gets its own stream ``GEN_SEED_BASE + 1000 * seed + index`` so the
    chunk boundary is explicit and reproducible instead of depending on however
    the sampler happens to consume its generator.
    """

    x0 = torch.full((1, tm.STATE_DIM), math.log(S0), device=device, dtype=model.dtype)
    pieces: list[torch.Tensor] = []
    drawn = 0
    index = 0
    while drawn < int(num_paths):
        size = min(int(chunk), int(num_paths) - drawn)
        with torch.no_grad():
            piece = model.sample(
                num_paths=size,
                x0=x0,
                seed=GEN_SEED_BASE + 1000 * int(seed) + index,
            ).paths
        pieces.append(piece.detach())
        drawn += size
        index += 1
    return torch.cat(pieces, dim=0)


def generate_one(seed: int, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)

    weights_path = METHOD_ROOT / "weights" / f"seed_{int(seed)}_model.pt"
    if not weights_path.is_file():
        raise SystemExit(
            f"ABORT: {weights_path} missing -- run select_checkpoint_multiasset.py"
        )

    run_dir = Path(args.run_root) / f"seed_{int(seed)}"
    config = _run_config(int(seed), run_dir)
    model, _grid, _num_steps = _rebuild(int(seed), config, device=device)

    payload = torch.load(weights_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "network_state_dict" not in payload:
        raise SystemExit(
            f"ABORT: {weights_path} is not a checkpoint_state() payload; "
            "select_checkpoint_multiasset.py must run before generation"
        )
    model.load_checkpoint_state(payload)

    started = time.perf_counter()
    log_paths = _sample_bank(
        model,
        seed=int(seed),
        num_paths=int(args.num_paths),
        chunk=int(args.chunk),
        device=device,
    )
    prices = torch.exp(log_paths.double()).cpu().numpy()
    gen_sec = time.perf_counter() - started

    if prices.shape != EXPECTED_SHAPE:
        raise SystemExit(
            f"ABORT: seed {seed} produced {prices.shape}, want {EXPECTED_SHAPE}"
        )
    if not np.isfinite(prices).all():
        raise SystemExit(
            f"ABORT: seed {seed} produced non-finite prices. This is a divergence, "
            "not a missing file: report it, do not substitute another seed."
        )
    if not (prices > 0.0).all():
        raise SystemExit(f"ABORT: seed {seed} produced non-positive prices")

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

    published_path = METHOD_ROOT / "weights" / f"seed_{int(seed)}_config.json"
    published = (
        json.loads(published_path.read_text(encoding="utf-8"))
        if published_path.is_file()
        else config
    )
    metadata = {
        "method": "Deep-MKV-TS-SBTSref",
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
        "train_time_sec": float(published.get("train_time_sec", 0.0)),
        "selected_step": published.get("selected_step"),
        "ridge_lambda": published.get("ridge_lambda"),
        "n_parameters": published.get("n_parameters"),
        "generation_seed_base": GEN_SEED_BASE,
        "generation_seed_first_chunk": GEN_SEED_BASE + 1000 * int(seed),
        "generation_chunk": int(args.chunk),
        "s0_rescaled": True,
        "s0_max_abs_residual_before_rescale": residual,
        # The reference this bank was generated against -- the one thing that
        # distinguishes this method from the sibling, so it travels with the data.
        "reference_family": published.get("reference_family"),
        "reference_h": published.get("reference_h"),
        "reference_markov_order": published.get("reference_markov_order"),
        "reference_npi": published.get("reference_npi"),
        "reference_weight_grad_mode": published.get("reference_weight_grad_mode"),
        "allow_drift_correction": published.get("allow_drift_correction"),
        "gpu": (torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        "date": date.today().isoformat(),
    }
    (gen_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"  seed {seed}: {prices.shape} price "
        f"[{prices.min():.2f}, {prices.max():.2f}]  S0 exact: {s0_exact}  "
        f"pre-rescale residual {residual:.3e}  gen {gen_sec:.1f}s",
        flush=True,
    )
    print(f"  [out] {npy_path}", flush=True)

    del model, log_paths
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-paths", type=int, default=8192)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument(
        "--run-root", type=Path, default=Path(__file__).resolve().parent / "runs"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    for seed in args.seeds:
        generate_one(int(seed), args)
    print(
        "\nNow run collect_artifacts.py -- it rebuilds generation_time.csv from "
        "the per-seed metadata.json files and gates the section 4 contract.",
        flush=True,
    )


if __name__ == "__main__":
    main()
