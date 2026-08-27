#!/usr/bin/env python3
"""Generate the (6144, 128, 8) path bank per seed from its SELECTED checkpoint.

Order of operations
-------------------
    train_true.py             -> runs/seed_i/training_checkpoints/step_*.pt
    select_checkpoint_true.py -> weights/seed_i_model.pt   (section-7 rule)
    generate_bank_true.py     -> generated_paths/seed_i/*.npy   <- THIS FILE
    collect_artifacts.py      -> losses/generation_time.csv + contract gate

Generation is deliberately NOT folded into the trainer. Deep-MKV-TS reports a
checkpoint chosen on validation, and which checkpoint that is cannot be known
until training has finished, so the bank cannot be sampled from the in-memory
model at the end of the fit.

Why the full checkpoint is loaded
---------------------------------
``DiscreteMPModel.checkpoint_state()`` documents that the timewise target scales
and the noise-target baseline feed the adjoint moments handed to the control
map. A bare ``network.state_dict()`` therefore does NOT reproduce a fitted
model's rollouts. ``select_checkpoint_true.py`` writes the full payload into
``weights/seed_i_model.pt`` and this file restores it with
``load_checkpoint_state``. The payload is type-checked below rather than
trusted, because the trainer also writes a bare state_dict to that same path as
a provisional value -- loading that one would silently generate from a
half-restored model.

The S0 contract
---------------
The generator integrates LOG prices from ``x0 = log(100)`` exactly, so it meets
``S[:, 0, :] == 100.0`` natively up to float round-off -- but the model runs in
float32, where ``exp(log(100))`` is not bit-exactly 100. The fix, applied in
float64:

    Xg = Xg * (100.0 / Xg[:, 0:1, :])
    Xg[:, 0, :] = 100.0

Multiplying a whole path by a constant is exactly a shift of the log-price
level, so every log-return is bit-identical before and after: this normalises
the reported level without touching the quantity every metric consumes.
``s0_rescaled`` is recorded as ``true`` so a reader can distinguish a method
that met the contract natively from one rescaled onto it, and the residual the
rescale removed is recorded as ``s0_max_abs_residual_before_rescale``.

Shape
-----
``(6144, 128, 8)``: 6144 paths matching the train split, 128 time points =
127 steps plus the initial point, 8 assets. The shape is asserted rather than
inferred -- a bank one step short still loads, still plots, and silently
misscales every annualised statistic downstream.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \\
    PYTHONPATH="$R/src:$R/experiments" taskset -c 0-7 \\
    /home/tbasseras/gpu-venv/bin/python generate_bank_true.py --seeds 0 1 2 3 4
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

import eigh_fallback  # noqa: E402
import train_true as tt  # noqa: E402
from fit_reference_true import DATASET, DT, SEQ_TAG, STATE_DIM  # noqa: E402

S0 = 100.0
EXPECTED_SHAPE = (6144, 128, 8)
GEN_SEED_BASE = 90000  # generation seed = base + seed; never reuses a training seed


def _parse_seq_tag(tag: str) -> tuple[int, int, int]:
    """``"6144x128x8"`` -> ``(6144, 128, 8)``.

    The tag names the REAL bank.  Its N is the A/B bank size; the T and d are
    contract constants that every bank -- real, A/B, or CRPS pool -- must meet.
    A CRPS pool overrides only N (via ``--m-simu``), never T or d.
    """
    parts = tag.lower().split("x")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise SystemExit(f"ABORT: --seq-tag {tag!r} is not of the form NxTxd")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def generate_one(seed: int, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    _, tag_t, tag_d = _parse_seq_tag(args.seq_tag)
    expected_shape = (int(args.num_paths), tag_t, tag_d)

    # --out-root already redirects where banks LAND; this redirects where the
    # weights are READ FROM. They are separate because a CRPS pool is written to
    # crps_banks/ while still coming off the published weights. A retune needs
    # the other combination: retuned weights, banks beside them.
    weights_root = Path(getattr(args, "weights_root", None) or METHOD_ROOT)
    weights_path = weights_root / "weights" / f"seed_{int(seed)}_model.pt"
    if not weights_path.is_file():
        raise SystemExit(
            f"ABORT: {weights_path} missing -- run select_checkpoint_true.py"
        )

    probe = tt.load_log_prices(num_paths=1, device="cpu")
    num_steps = int(probe.shape[1]) - 1
    del probe
    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)

    model, sigma_max = tt.build_model(
        grid=grid, device=device, seed=int(seed), kernel_steps=num_steps
    )
    payload = torch.load(weights_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "network_state_dict" not in payload:
        raise SystemExit(
            f"ABORT: {weights_path} is not a checkpoint_state() payload; "
            "select_checkpoint_true.py must run before generation"
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

    if prices.shape != expected_shape:
        raise SystemExit(
            f"ABORT: seed {seed} produced {prices.shape}, want {expected_shape}"
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

    out_root = Path(args.out_root).resolve()
    gen_dir = out_root / "generated_paths" / f"seed_{int(seed)}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    n, t, d = expected_shape
    npy_path = gen_dir / f"generated_paths_{n}x{t}x{d}.npy"
    np.save(npy_path, prices)

    config_path = weights_root / "weights" / f"seed_{int(seed)}_config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    metadata = {
        "method": "Deep-MKV-TS",
        "dataset": "TrueDataset om_2022-07_N6144",
        "seed": int(seed),
        "data_dir": str(args.data_dir),
        "seq_tag": str(args.seq_tag),
        # An A/B bank (N = the tag's N) and a CRPS pool (N = --m-simu) come off
        # the SAME selected checkpoint but are SEPARATE draws with the same
        # generation seed per seed index.  Recorded so a reader can tell which
        # role a bank on disk was written for.
        "bank_role": ("ab_bank" if int(args.num_paths) == _parse_seq_tag(args.seq_tag)[0] else "crps_pool"),
        "shape": list(prices.shape),
        "dtype": str(prices.dtype),
        "d": int(prices.shape[2]),
        "dt": float(DT),
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
        "selection_rule": config.get("selection_rule"),
        "selection_nn_ratio": config.get("selection_nn_ratio"),
        "ridge_lambda": config.get("ridge_lambda"),
        "sigma_max": float(sigma_max),
        "n_parameters": config.get("n_parameters"),
        "generation_seed": GEN_SEED_BASE + int(seed),
        "s0_rescaled": True,
        "s0_max_abs_residual_before_rescale": residual,
        "eigh_fallback": eigh_fallback.counters(),
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
    # --seed (one) and --seeds (many) both exist because the two callers differ:
    # the A/B stage sweeps all seeds in one process to amortise model build,
    # while the CRPS stage fans one seed per GPU worker.
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    # --m-simu is the pipeline's spelling of --num-paths.  Same knob, and giving
    # both with different values is an error rather than a silent precedence.
    parser.add_argument("--num-paths", type=int, default=None)
    parser.add_argument("--m-simu", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=str(DATASET))
    parser.add_argument("--seq-tag", type=str, default=SEQ_TAG)
    parser.add_argument("--out-root", type=str, default=str(METHOD_ROOT))
    parser.add_argument(
        "--weights-root", type=str, default=None,
        help="directory holding weights/seed_<S>_model.pt (default: the "
             "published Deep-MKV-TS method root). Independent of --out-root: a "
             "CRPS pool redirects the OUTPUT only, a retune redirects the INPUT.",
    )
    args = parser.parse_args(argv)

    if args.seed is not None and args.seeds is not None:
        raise SystemExit("ABORT: pass --seed or --seeds, not both")
    if args.seed is not None:
        args.seeds = [int(args.seed)]
    elif args.seeds is None:
        args.seeds = [0, 1, 2, 3, 4]

    if args.num_paths is not None and args.m_simu is not None:
        if int(args.num_paths) != int(args.m_simu):
            raise SystemExit(
                f"ABORT: --num-paths {args.num_paths} != --m-simu {args.m_simu}; "
                "they are the same knob"
            )
    if args.m_simu is not None:
        args.num_paths = int(args.m_simu)
    elif args.num_paths is None:
        args.num_paths = _parse_seq_tag(args.seq_tag)[0]

    # The model was fitted to ONE dataset.  Sampling it and labelling the output
    # with a different --data-dir would attach the wrong provenance to a bank
    # that metrics then compare against that other dataset's real split, so a
    # mismatch is fatal rather than a warning.
    given = Path(args.data_dir).resolve()
    if given != DATASET.resolve():
        raise SystemExit(
            f"ABORT: --data-dir {given} but the fitted reference kernel and the "
            f"trained checkpoints belong to {DATASET.resolve()}"
        )
    _, tag_t, tag_d = _parse_seq_tag(args.seq_tag)
    if tag_d != STATE_DIM:
        raise SystemExit(f"ABORT: --seq-tag d={tag_d} but STATE_DIM={STATE_DIM}")
    return args


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
