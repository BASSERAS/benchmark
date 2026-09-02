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

Why the full checkpoint is loaded
---------------------------------
``DiscreteMPModel.checkpoint_state()`` documents that the timewise target scales
and the noise-target baseline feed the adjoint moments passed to the control
map.  A bare ``network.state_dict()`` therefore does NOT reproduce a fitted
model's rollouts.  ``select_checkpoint_multiasset.py`` writes the full payload
into ``weights/seed_i_model.pt`` and this file restores it with
``load_checkpoint_state``.

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

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
    PYTHONPATH="$R/src:$R/experiments" taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python run_all_multiasset.py --seeds 0 2 4 5 6
"""

from __future__ import annotations

import argparse
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
REFERENCE = Path("/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference")
for _path in (HERE, REFERENCE / "src", REFERENCE / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402

import train_multiasset as tm  # noqa: E402
from fit_reference_multiasset import DT, STATE_DIM  # noqa: E402

S0 = 100.0
EXPECTED_SHAPE = (8192, 252, 8)
GEN_SEED_BASE = 90000  # generation seed = base + seed; never reuses a training seed


def generate_one(seed: int, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)

    weights_path = METHOD_ROOT / "weights" / f"seed_{int(seed)}_model.pt"
    if not weights_path.is_file():
        raise SystemExit(
            f"ABORT: {weights_path} missing -- run select_checkpoint_multiasset.py"
        )

    probe = tm.load_log_prices(num_paths=1, device="cpu")
    num_steps = int(probe.shape[1]) - 1
    del probe
    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)

    model = tm.build_model(
        grid=grid, device=device, seed=int(seed), kernel_steps=num_steps
    )
    payload = torch.load(weights_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "network_state_dict" not in payload:
        raise SystemExit(
            f"ABORT: {weights_path} is not a checkpoint_state() payload; "
            "select_checkpoint_multiasset.py must run before generation"
        )
    model.load_checkpoint_state(payload)

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

    config_path = METHOD_ROOT / "weights" / f"seed_{int(seed)}_config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    metadata = {
        "method": "Deep-MKV-TS-quarter",
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
        "train_time_sec": float(config.get("train_time_sec", 0.0)),
        "selected_step": config.get("selected_step"),
        "ridge_lambda": config.get("ridge_lambda"),
        "n_parameters": config.get("n_parameters"),
        "generation_seed": GEN_SEED_BASE + int(seed),
        "s0_rescaled": True,
        "s0_max_abs_residual_before_rescale": residual,
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
    # Not range(5): seed 3 diverged at step 2 with a non-finite control and was
    # replaced by seed 5. See "The caveat came true" in README.md. The gap stays
    # visible -- renumbering 5 -> 3 would launder a stability failure.
    # Not range(5): TWO seeds diverged with a non-finite control -- seed 3 at
    # step 2, seed 1 at step ~1450 -- and were replaced by seeds 5 and 6. See
    # "The caveat came true twice" in README.md. The gaps stay visible;
    # renumbering would launder a 33% stability failure rate.
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 2, 4, 5, 6])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-paths", type=int, default=8192)
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
