#!/usr/bin/env python3
"""Evaluate one SBTS bank with the locked delayed-memory scaling protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt.discrepancies.mmd import rbf_mmd2  # noqa: E402
from evaluate_drawdown_memory import drawdown_features, summarize_memory  # noqa: E402
from summarize_volatility_only_nested_scaling import _path_metrics  # noqa: E402


MMD_BANDWIDTHS = (0.5, 1.0, 2.0, 4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--generated-data", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsample-size", type=int, default=1024)
    parser.add_argument("--swd-projections", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _prices(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(f"{path} must contain finite positive price paths")
    return values


def _path_mmd(
    train: np.ndarray,
    target: np.ndarray,
    generated: np.ndarray,
    *,
    device: torch.device,
) -> float:
    train_log = np.log(train / train[:, :1])[:, 1:]
    target_log = np.log(target / target[:, :1])[:, 1:]
    generated_log = np.log(generated / generated[:, :1])[:, 1:]
    center = train_log.mean(axis=0, keepdims=True)
    scale = max(float(train_log.std()), 1e-12)
    target_features = torch.as_tensor(
        (target_log - center) / scale, dtype=torch.float64, device=device
    )
    generated_features = torch.as_tensor(
        (generated_log - center) / scale, dtype=torch.float64, device=device
    )
    value = rbf_mmd2(
        generated_features,
        target_features,
        bandwidths=MMD_BANDWIDTHS,
        unbiased=False,
    )
    return math.sqrt(max(float(value.item()), 0.0))


def evaluate_bank(
    *,
    dataset_root: Path,
    generated_path: Path,
    sequence_length: int,
    seed: int,
    subsample_size: int,
    swd_projections: int,
    device: torch.device,
) -> dict[str, object]:
    train = _prices(dataset_root / "train.npy")
    validation = _prices(dataset_root / "disc.npy")
    generated = _prices(generated_path)
    expected_shape = (8192, int(sequence_length))
    for name, values in (
        ("train", train), ("validation", validation), ("generated", generated)
    ):
        if tuple(values.shape) != expected_shape:
            raise ValueError(f"{name} has shape {values.shape}, expected {expected_shape}")

    rng = np.random.default_rng(20260804 + 1000 * int(sequence_length) + int(seed))
    target_indices = rng.choice(len(validation), int(subsample_size), replace=False)
    generated_indices = rng.choice(len(generated), int(subsample_size), replace=False)
    target_subsample = validation[target_indices]
    generated_subsample = generated[generated_indices]
    metrics = _path_metrics(
        target_subsample,
        generated_subsample,
        device=device,
        swd_projections=int(swd_projections),
        swd_seed=20260804 + int(seed),
    )
    metrics["path_mmd"] = _path_mmd(
        train,
        target_subsample,
        generated_subsample,
        device=device,
    )

    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["configuration"]
    memory_kwargs = {
        "threshold": float(config["drawdown_threshold"]),
        "split": int(config["split_index"]),
        "history_cutoff": int(config["history_cutoff"]),
        "recent_window": int(config["recent_window"]),
        "horizon": int(config["future_horizon"]),
        "annualization": 1.0 / float(config["dt"]),
    }
    target_memory = summarize_memory(drawdown_features(validation, **memory_kwargs))
    generated_memory = summarize_memory(drawdown_features(generated, **memory_kwargs))
    metrics.update(
        {
            "future_rv_hit_gap_error": abs(
                float(generated_memory["future_rv_hit_gap"])
                - float(target_memory["future_rv_hit_gap"])
            ),
            "early_hit_future_rv_correlation_error": abs(
                float(generated_memory["early_hit_future_rv_correlation"])
                - float(target_memory["early_hit_future_rv_correlation"])
            ),
            "early_history_incremental_r2_error": abs(
                float(generated_memory["early_history_incremental_r2"])
                - float(target_memory["early_history_incremental_r2"])
            ),
            "early_hit_rate_error": abs(
                float(generated_memory["early_hit_rate"])
                - float(target_memory["early_hit_rate"])
            ),
        }
    )
    return {
        "status": "complete",
        "sequence_length": int(sequence_length),
        "seed": int(seed),
        "decision_split": "disc.npy",
        "test_split_used": False,
        "subsample_size": int(subsample_size),
        "swd_projections": int(swd_projections),
        "mmd": {
            "estimator": "biased_rbf_mmd_distance",
            "path_standardization": "train_log_path_coordinate_mean_and_scalar_std",
            "bandwidths": list(MMD_BANDWIDTHS),
        },
        "metrics": metrics,
        "target_memory": target_memory,
        "generated_memory": generated_memory,
        "sources": {
            "train": str((dataset_root / "train.npy").resolve()),
            "validation": str((dataset_root / "disc.npy").resolve()),
            "generated": str(generated_path.resolve()),
        },
    }


def main() -> None:
    args = parse_args()
    generation_manifest = json.loads(
        args.generation_manifest.read_text(encoding="utf-8")
    )
    if bool(generation_manifest.get("test_split_used", True)):
        raise RuntimeError("SBTS generation manifest does not certify validation-only use")
    payload = evaluate_bank(
        dataset_root=args.dataset_root.resolve(),
        generated_path=args.generated_data.resolve(),
        sequence_length=int(args.sequence_length),
        seed=int(args.seed),
        subsample_size=int(args.subsample_size),
        swd_projections=int(args.swd_projections),
        device=torch.device(args.device),
    )
    payload["training_resources"] = {
        "training_wall_seconds": float(generation_manifest["training_wall_seconds"]),
        "cuda_incremental_peak_allocated_bytes": int(
            generation_manifest["cuda_incremental_peak_allocated_bytes"]
        ),
        "peak_process_rss_mb": float(generation_manifest["peak_process_rss_mb"]),
        "peak_child_rss_mb": float(generation_manifest["peak_child_rss_mb"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sequence_length": payload["sequence_length"],
                "seed": payload["seed"],
                "metrics": payload["metrics"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
