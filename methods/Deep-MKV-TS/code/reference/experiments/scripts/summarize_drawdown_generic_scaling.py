#!/usr/bin/env python3
"""Aggregate accuracy, stability, runtime, and memory across solver resolutions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    128: REPO_ROOT / "runs" / "drawdown_memory_seed0_comparison_20260730" / "data",
    256: REPO_ROOT / "data" / "drawdown_memory_scaling" / "n256",
    512: REPO_ROOT / "data" / "drawdown_memory_scaling" / "n512",
}
ARMS = ("direct_bptt", "generic_ce_online", "generic_ce_nested")
ERROR_METRICS = (
    "abs_return_acf_rmse_lags_1_50",
    "future_rv_wasserstein",
    "future_rv_hit_gap_error",
    "early_history_incremental_r2_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 3, 4))
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=(128, 256, 512))
    parser.add_argument(
        "--benchmark-metrics-dir",
        type=Path,
        default=Path("/home/tbasseras/benchmark/metrics"),
    )
    parser.add_argument("--subsample-size", type=int, default=1024)
    parser.add_argument("--skip-path-metrics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    path_functions = {}
    if not args.skip_path_metrics:
        sys.path.insert(0, str(args.benchmark_metrics_dir.resolve()))
        from metrics import increment_mmd2, mmd2, path_swd, terminal_swd  # type: ignore[import-not-found]  # noqa: PLC0415

        path_functions = {
            "path_mmd2": mmd2,
            "increment_mmd2": increment_mmd2,
            "path_swd": path_swd,
            "terminal_swd": terminal_swd,
        }

    per_resolution: dict[str, object] = {}
    for sequence_length in tuple(int(value) for value in args.sequence_lengths):
        dataset_root = DATASETS[sequence_length]
        real_all = np.asarray(np.load(dataset_root / "test.npy"), dtype=np.float64)[..., None]
        rows: dict[str, dict[str, object]] = {}
        for seed in tuple(int(value) for value in args.seeds):
            rng = np.random.default_rng(8080 + 1000 * sequence_length + seed)
            real_indices = rng.choice(
                len(real_all), int(args.subsample_size), replace=False
            )
            fake_indices = rng.choice(8192, int(args.subsample_size), replace=False)
            real = real_all[real_indices]
            seed_rows: dict[str, object] = {}
            for arm in ARMS:
                arm_dir = run_root / f"n{sequence_length}" / f"seed_{seed}" / arm
                test_path = arm_dir / "test_metrics.json"
                manifest_path = arm_dir / "training_manifest.json"
                bank_path = arm_dir / f"generated_paths_8192x{sequence_length}.npy"
                if not (test_path.exists() and manifest_path.exists() and bank_path.exists()):
                    seed_rows[arm] = {"status": "missing"}
                    continue
                test = json.loads(test_path.read_text(encoding="utf-8"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                fake_all = np.asarray(np.load(bank_path), dtype=np.float64)[..., None]
                fake = fake_all[fake_indices]
                path_metrics = {
                    name: float(function(real, fake))
                    for name, function in path_functions.items()
                }
                final = manifest.get("fit_final", {})
                seed_rows[arm] = {
                    "status": "complete",
                    "errors": {
                        name: float(test["errors"][name]) for name in ERROR_METRICS
                    },
                    "path_metrics": path_metrics,
                    "training_resources": manifest["training_resources"],
                    "final_training": {
                        "complete_objective_value": float(
                            final.get("complete_objective_value", float("nan"))
                        ),
                        "parameter_nonfinite_fraction": float(
                            final.get("parameter_nonfinite_fraction", 0.0)
                        ),
                        "gradient_nonfinite_fraction": float(
                            final.get("gradient_nonfinite_fraction", 0.0)
                        ),
                    },
                }
            rows[str(seed)] = seed_rows

        aggregate: dict[str, object] = {}
        for arm in ARMS:
            complete = [
                rows[str(seed)][arm]
                for seed in args.seeds
                if rows[str(seed)][arm].get("status") == "complete"  # type: ignore[union-attr]
            ]
            arm_summary: dict[str, object] = {
                "complete_count": len(complete),
                "failure_or_missing_count": len(tuple(args.seeds)) - len(complete),
            }
            if complete:
                arm_summary["errors"] = {
                    metric: summarize(
                        [float(row["errors"][metric]) for row in complete]  # type: ignore[index]
                    )
                    for metric in ERROR_METRICS
                }
                arm_summary["path_metrics"] = {
                    metric: summarize(
                        [float(row["path_metrics"][metric]) for row in complete]  # type: ignore[index]
                    )
                    for metric in path_functions
                }
                resource_names = (
                    "training_wall_seconds",
                    "cuda_peak_allocated_bytes",
                    "cuda_incremental_peak_allocated_bytes",
                )
                arm_summary["training_resources"] = {
                    metric: summarize(
                        [float(row["training_resources"][metric]) for row in complete]  # type: ignore[index]
                    )
                    for metric in resource_names
                }
            aggregate[arm] = arm_summary
        per_resolution[str(sequence_length)] = {
            "per_seed": rows,
            "aggregate": aggregate,
        }

    payload = {
        "seeds": [int(value) for value in args.seeds],
        "sequence_lengths": [int(value) for value in args.sequence_lengths],
        "arms": list(ARMS),
        "metric_direction": "lower_is_better",
        "subsample_size": int(args.subsample_size),
        "per_resolution": per_resolution,
        "sources": {
            "run_root": str(run_root),
            "benchmark_metrics_dir": str(args.benchmark_metrics_dir.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value["aggregate"] for key, value in per_resolution.items()}, indent=2), flush=True)  # type: ignore[index]


if __name__ == "__main__":
    main()
