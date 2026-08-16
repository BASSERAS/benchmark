#!/usr/bin/env python3
"""Paired moving-date-block interval for one real-data shadow metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, action="append", required=True)
    parser.add_argument("--comparison-root", type=Path, action="append", required=True)
    parser.add_argument("--metrics-file", default="test_per_path_metrics.npz")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20_260_801)
    parser.add_argument(
        "--pairs-per-group",
        type=int,
        default=0,
        help=(
            "For an index-major panel, the number of matched seeds per index. "
            "This affects the reported between-seed estimates, while the date "
            "bootstrap always clusters the cross-index mean by date."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metric(root: Path, filename: str, metric: str) -> np.ndarray:
    path = root / filename
    with np.load(path, allow_pickle=False) as payload:
        if metric not in payload.files:
            raise KeyError(f"{metric!r} is unavailable in {path}")
        values = np.asarray(payload[metric], dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"{path}:{metric} must contain at least two finite values")
    return values


def moving_block_means(
    values: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    generator: np.random.Generator,
) -> np.ndarray:
    num_dates = int(values.size)
    if block_length < 1 or block_length > num_dates:
        raise ValueError("block_length must lie between one and the phase length")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    num_blocks = (num_dates + block_length - 1) // block_length
    starts = generator.integers(
        0,
        num_dates - block_length + 1,
        size=(replicates, num_blocks),
    )
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[..., None] + offsets).reshape(replicates, -1)
    return values[indices[:, :num_dates]].mean(axis=1)


def main() -> None:
    args = parse_args()
    if len(args.primary_root) != len(args.comparison_root):
        raise ValueError("primary and comparison roots must have equal counts")
    paired = []
    for primary_root, comparison_root in zip(
        args.primary_root, args.comparison_root, strict=True
    ):
        primary = load_metric(primary_root, args.metrics_file, args.metric)
        comparison = load_metric(comparison_root, args.metrics_file, args.metric)
        if primary.shape != comparison.shape:
            raise ValueError(
                f"paired arrays differ: {primary.shape} != {comparison.shape}"
            )
        paired.append(primary - comparison)
    shapes = {values.shape for values in paired}
    if len(shapes) != 1:
        raise ValueError(f"paired arrays have different shapes: {sorted(shapes)}")

    paired_array = np.stack(paired)
    date_differences = paired_array.mean(axis=0)
    bootstrap = moving_block_means(
        date_differences,
        block_length=int(args.block_length),
        replicates=int(args.replicates),
        generator=np.random.default_rng(int(args.seed)),
    )
    pair_estimates = paired_array.mean(axis=1)
    if args.pairs_per_group:
        if args.pairs_per_group < 1 or pair_estimates.size % args.pairs_per_group:
            raise ValueError("pairs_per_group must divide the number of root pairs")
        seed_estimates = pair_estimates.reshape(-1, args.pairs_per_group).mean(axis=0)
        num_groups = pair_estimates.size // args.pairs_per_group
    else:
        seed_estimates = pair_estimates
        num_groups = 1

    alpha = 0.5 * (1.0 - float(args.confidence_level))
    payload = {
        "metric": args.metric,
        "metrics_file": args.metrics_file,
        "difference_direction": "primary_minus_comparison",
        "estimate": float(date_differences.mean()),
        "bootstrap_standard_error": float(bootstrap.std(ddof=0)),
        "ci_lower": float(np.quantile(bootstrap, alpha)),
        "ci_upper": float(np.quantile(bootstrap, 1.0 - alpha)),
        "confidence_level": float(args.confidence_level),
        "num_dates": int(date_differences.size),
        "block_length_dates": int(args.block_length),
        "bootstrap_replicates": int(args.replicates),
        "bootstrap_seed": int(args.seed),
        "seed_estimates": [float(value) for value in seed_estimates],
        "seed_sample_standard_deviation": (
            float(seed_estimates.std(ddof=1)) if seed_estimates.size > 1 else None
        ),
        "num_seed_pairs": len(paired),
        "num_groups": int(num_groups),
        "paired_seeds_per_group": int(args.pairs_per_group or pair_estimates.size),
        "sources": {
            "primary_roots": [str(path.resolve()) for path in args.primary_root],
            "comparison_roots": [
                str(path.resolve()) for path in args.comparison_root
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
