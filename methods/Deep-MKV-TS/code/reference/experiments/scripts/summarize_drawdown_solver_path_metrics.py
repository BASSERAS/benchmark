#!/usr/bin/env python3
"""Score matched delayed-memory solvers with standard whole-path metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


DEFAULT_SEEDS = (0, 1, 3, 4)
DEFAULT_ARMS = ("generic_ce_nested", "generic_ce_online")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--benchmark-metrics-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--arms", nargs=2, default=DEFAULT_ARMS)
    parser.add_argument(
        "--arm-path",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH_TEMPLATE"),
        help="Two repeatable label/path templates containing {seed}.",
    )
    parser.add_argument("--subsample-size", type=int, default=1024)
    parser.add_argument("--include-volatility-mmd", action="store_true")
    parser.add_argument("--seed", type=int, default=8080)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.benchmark_metrics_dir.resolve()))
    from metrics import (  # type: ignore[import-not-found]  # noqa: PLC0415
        increment_mmd2,
        mmd2,
        path_swd,
        terminal_mmd2,
        terminal_swd,
        volatility_mmd,
    )

    real = np.asarray(np.load(args.test_data), dtype=np.float64)
    if real.ndim != 2 or not np.isfinite(real).all():
        raise ValueError("test data must be a finite two-dimensional path bank")
    real = real[..., None]
    metric_functions = {
        "path_mmd2": mmd2,
        "terminal_mmd2": terminal_mmd2,
        "increment_mmd2": increment_mmd2,
        "terminal_swd": terminal_swd,
        "path_swd": path_swd,
    }
    if args.include_volatility_mmd:
        metric_functions["volatility_mmd"] = volatility_mmd
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    seeds = tuple(int(seed) for seed in args.seeds)
    if args.arm_path:
        if len(args.arm_path) != 2:
            raise ValueError("--arm-path must be supplied exactly twice")
        arms = tuple(str(item[0]) for item in args.arm_path)
        path_templates = {
            str(label): str(template) for label, template in args.arm_path
        }
        if any("{seed}" not in template for template in path_templates.values()):
            raise ValueError("each --arm-path template must contain {seed}")
    else:
        if args.run_root is None:
            raise ValueError("--run-root is required without --arm-path")
        arms = tuple(str(arm) for arm in args.arms)
        path_templates = {
            arm: str(
                args.run_root
                / "seed_{seed}"
                / arm
                / "generated_paths_8192x128.npy"
            )
            for arm in arms
        }
    for seed in seeds:
        generator = np.random.default_rng(int(args.seed) + int(seed))
        real_indices = generator.choice(
            len(real), int(args.subsample_size), replace=False
        )
        fake_indices = generator.choice(8192, int(args.subsample_size), replace=False)
        real_subsample = real[real_indices]
        seed_rows: dict[str, dict[str, float]] = {}
        for arm in arms:
            path = Path(path_templates[arm].format(seed=seed))
            fake = np.asarray(np.load(path), dtype=np.float64)
            if fake.shape != real.shape[:2] or not np.isfinite(fake).all():
                raise ValueError(f"invalid generated bank {path}: {fake.shape}")
            fake_subsample = fake[fake_indices, :, None]
            seed_rows[arm] = {
                name: float(function(real_subsample, fake_subsample))
                for name, function in metric_functions.items()
            }
        per_seed[str(seed)] = seed_rows

    aggregate = {}
    for arm in arms:
        aggregate[arm] = {
            metric: summarize(
                [per_seed[str(seed)][arm][metric] for seed in seeds]
            )
            for metric in metric_functions
        }
    difference_label = f"{arms[0]}_minus_{arms[1]}"
    aggregate[difference_label] = {
        metric: summarize(
            [
                per_seed[str(seed)][arms[0]][metric]
                - per_seed[str(seed)][arms[1]][metric]
                for seed in seeds
            ]
        )
        for metric in metric_functions
    }
    payload = {
        "seeds": list(seeds),
        "arms": list(arms),
        "subsample_size": int(args.subsample_size),
        "subsample_seed": int(args.seed),
        "shared_subsamples_between_arms": True,
        "metric_direction": "lower_is_better",
        "per_seed": per_seed,
        "aggregate": aggregate,
        "sources": {
            "test_data": str(args.test_data.resolve()),
            "generated_path_templates": path_templates,
            "benchmark_metrics_dir": str(args.benchmark_metrics_dir.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
