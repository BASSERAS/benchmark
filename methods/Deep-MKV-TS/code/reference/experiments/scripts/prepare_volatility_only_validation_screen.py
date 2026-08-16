#!/usr/bin/env python3
"""Prepare leak-free N=128 adapters for the locked volatility-only screen."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.data import (  # noqa: E402
    HestonConfig,
    simulate_heston_log_paths,
)


TASKS = ("heston", "mixture", "ES", "NQ", "RTY", "YM")
SEQUENCE_LENGTH = 128
SPLIT_INDEX = 64
HISTORY_CUTOFF = 40
FUTURE_HORIZON = 32
RECENT_WINDOW = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mixture-root",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "heston_parameter_mixture_seed0_20260730"
            / "data"
        ),
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=REPO_ROOT / "data" / "ICAIF_3Y" / "canonical_rth_128",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    return parser.parse_args()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return sha256(contiguous.view(np.uint8)).hexdigest()


def _drawdown_threshold(train: np.ndarray) -> float:
    logs = np.log(train / train[:, :1])
    prefix = logs[:, : HISTORY_CUTOFF + 1]
    drawdown = np.maximum.accumulate(prefix, axis=1) - prefix
    return float(np.quantile(np.max(drawdown, axis=1), 0.5))


def _validate_prices(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(f"{name} must have shape (B, {SEQUENCE_LENGTH})")
    if array.shape[0] < 2 or not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain at least two finite positive paths")
    if not np.allclose(array[:, 0], 100.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"{name} paths must start at 100")
    return array


def _write_adapter(
    *,
    root: Path,
    task: str,
    train: np.ndarray,
    validation: np.ndarray,
    dt: float,
    provenance: dict[str, object],
) -> None:
    train = _validate_prices(f"{task}/train", train)
    validation = _validate_prices(f"{task}/validation", validation)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "train.npy", train)
    np.save(root / "disc.npy", validation)
    # The legacy runner checks for this filename.  It deliberately aliases the
    # validation array; no canonical or synthetic test split is read.
    np.save(root / "test.npy", validation)
    manifest = {
        "configuration": {
            "name": f"locked_volatility_only_validation_{task}",
            "dt": float(dt),
            "sequence_length": SEQUENCE_LENGTH,
            "split_index": SPLIT_INDEX,
            "history_cutoff": HISTORY_CUTOFF,
            "future_horizon": FUTURE_HORIZON,
            "recent_window": RECENT_WINDOW,
            "drawdown_threshold": _drawdown_threshold(train),
            "s0": 100.0,
        },
        "screen_protocol": {
            "solver": "finite_objective_gated_nested_mp",
            "admissible_control": "volatility_only_alpha_fixed_zero",
            "model_seed": 0,
            "objective": "locked_n128_broad_plus_joint_prefix_future_rv",
            "test_split_used": False,
            "test_filename_aliases_validation": True,
        },
        "provenance": {
            **provenance,
            "train_sha256": _array_sha256(train),
            "validation_sha256": _array_sha256(validation),
            "train_shape": list(train.shape),
            "validation_shape": list(validation.shape),
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _heston_paths(*, seed: int) -> np.ndarray:
    dt = 0.004
    paths = simulate_heston_log_paths(
        num_paths=8192,
        config=HestonConfig(
            T=(SEQUENCE_LENGTH - 1) * dt,
            num_steps=SEQUENCE_LENGTH - 1,
            mu=0.0,
            kappa=3.0,
            theta=0.04,
            vol_of_vol=0.6,
            rho=-0.7,
            v0=0.04,
            x0=0.0,
        ),
        seed=int(seed),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )[..., 0].numpy()
    return 100.0 * np.exp(paths - paths[:, :1])


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    tasks = tuple(str(task) for task in args.tasks)
    if "heston" in tasks:
        _write_adapter(
            root=output_root / "heston",
            task="heston",
            train=_heston_paths(seed=0),
            validation=_heston_paths(seed=2),
            dt=0.004,
            provenance={
                "source": "fresh_standard_heston_simulation",
                "train_seed": 0,
                "validation_seed": 2,
                "parameters": {
                    "mu": 0.0,
                    "kappa": 3.0,
                    "theta": 0.04,
                    "xi": 0.6,
                    "rho": -0.7,
                    "v0": 0.04,
                },
            },
        )
    if "mixture" in tasks:
        mixture_root = args.mixture_root.resolve()
        _write_adapter(
            root=output_root / "mixture",
            task="mixture",
            train=np.load(mixture_root / "train.npy", allow_pickle=False),
            validation=np.load(mixture_root / "disc.npy", allow_pickle=False),
            dt=0.004,
            provenance={
                "source": str(mixture_root),
                "source_train": "train.npy",
                "source_validation": "disc.npy",
            },
        )
    canonical_root = args.canonical_root.resolve()
    for index in ("ES", "NQ", "RTY", "YM"):
        if index not in tasks:
            continue
        index_root = canonical_root / index
        _write_adapter(
            root=output_root / index,
            task=index,
            train=np.load(index_root / "prices_train.npy", allow_pickle=False),
            validation=np.load(
                index_root / "prices_validation.npy", allow_pickle=False
            ),
            dt=1.0 / (SEQUENCE_LENGTH - 1),
            provenance={
                "source": str(index_root),
                "source_train": "prices_train.npy",
                "source_validation": "prices_validation.npy",
                "canonical_test_loaded": False,
            },
        )
    print(f"prepared {len(tasks)} validation-only adapters in {output_root}")


if __name__ == "__main__":
    main()
