#!/usr/bin/env python3
"""Summarize the matched reference/full/volatility-only scaling experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wasserstein_distance
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.metrics import (  # noqa: E402
    acf,
    sliced_wasserstein_distance,
)


ARMS = (
    "generic_reference_only",
    "generic_ce_nested_reference",
    "generic_ce_nested_volatility_only",
    "direct_bptt_volatility_only",
)
REFERENCE_ARM = "generic_reference_only"
VOLATILITY_ONLY_NESTED_ARM = "generic_ce_nested_volatility_only"
FULL_CONTROL_NESTED_ARM = "generic_ce_nested_reference"
VOLATILITY_ONLY_DIRECT_ARM = "direct_bptt_volatility_only"

PRIMARY_DISTRIBUTION_METRICS = (
    "path_swd_normalized",
    "terminal_w1_normalized",
    "increment_w1_normalized",
    "realized_volatility_w1_normalized",
    "maximum_drawdown_w1_normalized",
    "return_qq_rmse_normalized",
)
DEPENDENCE_METRICS = (
    "return_acf_rmse",
    "absolute_return_acf_rmse",
    "squared_return_acf_rmse",
    "future_rv_hit_gap_error",
    "early_hit_future_rv_correlation_error",
)
LOCAL_CALIBRATION_METRICS = (
    "timewise_return_mean_rmse_normalized",
    "timewise_return_std_rmse_normalized",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 3, 4))
    parser.add_argument("--subsample-size", type=int, default=1024)
    parser.add_argument("--swd-projections", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _w1(left: np.ndarray, right: np.ndarray) -> float:
    return float(wasserstein_distance(left.reshape(-1), right.reshape(-1)))


def _maximum_drawdown(log_paths: np.ndarray) -> np.ndarray:
    return np.max(np.maximum.accumulate(log_paths, axis=1) - log_paths, axis=1)


def _path_metrics(
    target_prices: np.ndarray,
    generated_prices: np.ndarray,
    *,
    device: torch.device,
    swd_projections: int,
    swd_seed: int,
) -> dict[str, float]:
    target = np.log(target_prices / target_prices[:, :1])
    generated = np.log(generated_prices / generated_prices[:, :1])
    target_returns = np.diff(target, axis=1)
    generated_returns = np.diff(generated, axis=1)
    return_scale = max(float(target_returns.std()), 1e-12)
    path_scale = max(float(target[:, 1:].std()), 1e-12)
    terminal_scale = max(float(target[:, -1].std()), 1e-12)
    target_rv = np.sqrt(np.sum(target_returns**2, axis=1))
    generated_rv = np.sqrt(np.sum(generated_returns**2, axis=1))
    rv_scale = max(float(target_rv.mean()), 1e-12)
    target_drawdown = _maximum_drawdown(target)
    generated_drawdown = _maximum_drawdown(generated)
    drawdown_scale = max(float(target_drawdown.mean()), 1e-12)
    levels = np.linspace(0.01, 0.99, 99)
    target_quantiles = np.quantile(target_returns, levels)
    generated_quantiles = np.quantile(generated_returns, levels)
    lags = tuple(range(1, min(50, target_returns.shape[1] - 1) + 1))
    target_tensor = torch.as_tensor(target[..., None], dtype=torch.float64, device=device)
    generated_tensor = torch.as_tensor(
        generated[..., None], dtype=torch.float64, device=device
    )
    target_return_tensor = torch.diff(target_tensor, dim=1)
    generated_return_tensor = torch.diff(generated_tensor, dim=1)

    return {
        "path_swd_normalized": sliced_wasserstein_distance(
            generated_tensor / path_scale,
            target_tensor / path_scale,
            num_projections=int(swd_projections),
            seed=int(swd_seed),
        ),
        "terminal_w1_normalized": _w1(generated[:, -1], target[:, -1])
        / terminal_scale,
        "increment_w1_normalized": _w1(generated_returns, target_returns)
        / return_scale,
        "realized_volatility_w1_normalized": _w1(generated_rv, target_rv)
        / rv_scale,
        "maximum_drawdown_w1_normalized": _w1(
            generated_drawdown, target_drawdown
        )
        / drawdown_scale,
        "return_qq_rmse_normalized": float(
            np.sqrt(np.mean((generated_quantiles - target_quantiles) ** 2))
            / return_scale
        ),
        "return_acf_rmse": float(
            torch.mean(
                (
                    acf(generated_return_tensor, lags=lags)
                    - acf(target_return_tensor, lags=lags)
                ).pow(2)
            )
            .sqrt()
            .item()
        ),
        "absolute_return_acf_rmse": float(
            torch.mean(
                (
                    acf(generated_return_tensor.abs(), lags=lags)
                    - acf(target_return_tensor.abs(), lags=lags)
                ).pow(2)
            )
            .sqrt()
            .item()
        ),
        "squared_return_acf_rmse": float(
            torch.mean(
                (
                    acf(generated_return_tensor.pow(2), lags=lags)
                    - acf(target_return_tensor.pow(2), lags=lags)
                ).pow(2)
            )
            .sqrt()
            .item()
        ),
        "timewise_return_mean_rmse_normalized": float(
            np.sqrt(
                np.mean(
                    (generated_returns.mean(axis=0) - target_returns.mean(axis=0))
                    ** 2
                )
            )
            / return_scale
        ),
        "timewise_return_std_rmse_normalized": float(
            np.sqrt(
                np.mean(
                    (generated_returns.std(axis=0) - target_returns.std(axis=0))
                    ** 2
                )
            )
            / max(float(target_returns.std(axis=0).mean()), 1e-12)
        ),
    }


def _closure(reference: float, candidate: float) -> float | None:
    if not math.isfinite(reference) or reference <= 1e-12 or not math.isfinite(candidate):
        return None
    return (reference - candidate) / reference


def _group_closure(
    reference: dict[str, float],
    candidate: dict[str, float],
    metrics: tuple[str, ...],
) -> dict[str, float | int | None]:
    values = [
        value
        for name in metrics
        if (value := _closure(reference[name], candidate[name])) is not None
    ]
    if not values:
        return {
            "metric_count": 0,
            "improved_count": 0,
            "median_gap_closed": None,
            "mean_gap_closed": None,
        }
    return {
        "metric_count": len(values),
        "improved_count": sum(value > 0.0 for value in values),
        "median_gap_closed": float(np.median(values)),
        "mean_gap_closed": float(np.mean(values)),
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    dataset_root = args.dataset_root.resolve()
    sequence_length = int(args.sequence_length)
    target_all = np.asarray(np.load(dataset_root / "disc.npy"), dtype=np.float64)
    if int(args.subsample_size) > len(target_all):
        raise ValueError("subsample-size exceeds the validation-bank size")
    device = torch.device(str(args.device))
    per_seed: dict[str, object] = {}
    for seed in tuple(int(value) for value in args.seeds):
        rng = np.random.default_rng(20260804 + 1000 * sequence_length + seed)
        target_indices = rng.choice(len(target_all), int(args.subsample_size), replace=False)
        fake_indices = rng.choice(8192, int(args.subsample_size), replace=False)
        target = target_all[target_indices]
        seed_rows: dict[str, object] = {}
        for arm in ARMS:
            arm_dir = run_root / f"seed_{seed}" / arm
            metrics_path = arm_dir / "validation_metrics.json"
            manifest_path = arm_dir / "training_manifest.json"
            bank_path = arm_dir / f"generated_paths_8192x{sequence_length}.npy"
            if not (metrics_path.exists() and manifest_path.exists() and bank_path.exists()):
                seed_rows[arm] = {"status": "missing"}
                continue
            evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if bool(manifest.get("test_split_used", True)):
                raise RuntimeError(f"{manifest_path} does not certify an untouched test split")
            generated_all = np.asarray(np.load(bank_path), dtype=np.float64)
            if len(generated_all) != 8192:
                raise ValueError(f"{bank_path} must contain 8192 paths")
            enhanced = _path_metrics(
                target,
                generated_all[fake_indices],
                device=device,
                swd_projections=int(args.swd_projections),
                swd_seed=20260804 + seed,
            )
            enhanced.update(
                {
                    "future_rv_hit_gap_error": float(
                        evaluation["errors"]["future_rv_hit_gap_error"]
                    ),
                    "early_hit_future_rv_correlation_error": abs(
                        float(
                            evaluation["generated_memory"][
                                "early_hit_future_rv_correlation"
                            ]
                        )
                        - float(
                            evaluation["target_memory"][
                                "early_hit_future_rv_correlation"
                            ]
                        )
                    ),
                }
            )
            seed_rows[arm] = {
                "status": "complete",
                "metrics": enhanced,
                "training_resources": manifest["training_resources"],
                "admissible_control": manifest["admissible_control"],
                "initialization": manifest["initialization"],
            }

        reference_row = seed_rows.get(REFERENCE_ARM, {})
        if reference_row.get("status") == "complete":  # type: ignore[union-attr]
            reference_metrics = reference_row["metrics"]  # type: ignore[index]
            for arm in ARMS[1:]:
                row = seed_rows.get(arm, {})
                if row.get("status") != "complete":  # type: ignore[union-attr]
                    continue
                candidate_metrics = row["metrics"]  # type: ignore[index]
                row["gap_closed"] = {  # type: ignore[index]
                    name: _closure(reference_metrics[name], candidate_metrics[name])
                    for name in reference_metrics
                }
                row["group_gap_closed"] = {  # type: ignore[index]
                    "distribution": _group_closure(
                        reference_metrics,
                        candidate_metrics,
                        PRIMARY_DISTRIBUTION_METRICS,
                    ),
                    "dependence": _group_closure(
                        reference_metrics, candidate_metrics, DEPENDENCE_METRICS
                    ),
                    "local_calibration": _group_closure(
                        reference_metrics,
                        candidate_metrics,
                        LOCAL_CALIBRATION_METRICS,
                    ),
                }
        per_seed[str(seed)] = seed_rows

    aggregate: dict[str, object] = {}
    for arm in ARMS:
        complete = [
            per_seed[str(seed)][arm]
            for seed in args.seeds
            if per_seed[str(seed)][arm].get("status") == "complete"  # type: ignore[index,union-attr]
        ]
        row: dict[str, object] = {
            "complete_count": len(complete),
            "failure_or_missing_count": len(tuple(args.seeds)) - len(complete),
        }
        if complete:
            row["metrics"] = {
                name: _summary([float(item["metrics"][name]) for item in complete])  # type: ignore[index]
                for name in complete[0]["metrics"]  # type: ignore[index]
            }
            row["training_resources"] = {
                name: _summary(
                    [float(item["training_resources"][name]) for item in complete]  # type: ignore[index]
                )
                for name in (
                    "training_wall_seconds",
                    "cuda_incremental_peak_allocated_bytes",
                )
            }
        if arm != REFERENCE_ARM:
            closures = [item for item in complete if "gap_closed" in item]
            if closures:
                row["gap_closed"] = {
                    name: {
                        **_summary(
                            [float(item["gap_closed"][name]) for item in closures]  # type: ignore[index]
                        ),
                        "improved_seed_count": sum(
                            float(item["gap_closed"][name]) > 0.0  # type: ignore[index]
                            for item in closures
                        ),
                    }
                    for name in closures[0]["gap_closed"]  # type: ignore[index]
                    if all(item["gap_closed"][name] is not None for item in closures)  # type: ignore[index]
                }
                row["group_gap_closed"] = {
                    group: {
                        "median_gap_closed": _summary(
                            [
                                float(item["group_gap_closed"][group]["median_gap_closed"])  # type: ignore[index]
                                for item in closures
                            ]
                        ),
                        "positive_seed_count": sum(
                            float(item["group_gap_closed"][group]["median_gap_closed"])  # type: ignore[index]
                            > 0.0
                            for item in closures
                        ),
                    }
                    for group in (
                        "distribution",
                        "dependence",
                        "local_calibration",
                    )
                }
        aggregate[arm] = row

    def pairwise(candidate_arm: str, comparator_arm: str) -> dict[str, object]:
        matched = []
        for seed in args.seeds:
            seed_row = per_seed[str(seed)]
            candidate = seed_row[candidate_arm]
            comparator = seed_row[comparator_arm]
            if (
                candidate.get("status") == "complete"  # type: ignore[union-attr]
                and comparator.get("status") == "complete"  # type: ignore[union-attr]
            ):
                matched.append((candidate, comparator))
        result: dict[str, object] = {"matched_seed_count": len(matched)}
        if not matched:
            return result
        metric_names = tuple(matched[0][0]["metrics"])  # type: ignore[index]
        result["relative_error_reduction"] = {
            name: {
                **_summary(
                    [
                        float(
                            _closure(
                                float(comparator["metrics"][name]),  # type: ignore[index]
                                float(candidate["metrics"][name]),  # type: ignore[index]
                            )
                        )
                        for candidate, comparator in matched
                    ]
                ),
                "candidate_better_seed_count": sum(
                    float(candidate["metrics"][name])  # type: ignore[index]
                    < float(comparator["metrics"][name])  # type: ignore[index]
                    for candidate, comparator in matched
                ),
            }
            for name in metric_names
        }
        result["group_relative_error_reduction"] = {
            group: {
                "median_across_metrics": _summary(
                    [
                        float(
                            _group_closure(
                                comparator["metrics"],  # type: ignore[index]
                                candidate["metrics"],  # type: ignore[index]
                                names,
                            )["median_gap_closed"]
                        )
                        for candidate, comparator in matched
                    ]
                )
            }
            for group, names in (
                ("distribution", PRIMARY_DISTRIBUTION_METRICS),
                ("dependence", DEPENDENCE_METRICS),
                ("local_calibration", LOCAL_CALIBRATION_METRICS),
            )
        }
        result["training_wall_time_ratio_candidate_over_comparator"] = _summary(
            [
                float(candidate["training_resources"]["training_wall_seconds"])  # type: ignore[index]
                / max(
                    float(
                        comparator["training_resources"]["training_wall_seconds"]  # type: ignore[index]
                    ),
                    1e-12,
                )
                for candidate, comparator in matched
            ]
        )
        result[
            "peak_allocated_memory_ratio_candidate_over_comparator"
        ] = _summary(
            [
                float(
                    candidate["training_resources"][
                        "cuda_incremental_peak_allocated_bytes"
                    ]
                )
                / max(
                    float(
                        comparator["training_resources"][
                            "cuda_incremental_peak_allocated_bytes"
                        ]
                    ),
                    1.0,
                )
                for candidate, comparator in matched
            ]
        )
        return result

    pairwise_results = {
        "volatility_only_nested_vs_full_control_nested": pairwise(
            VOLATILITY_ONLY_NESTED_ARM, FULL_CONTROL_NESTED_ARM
        ),
        "volatility_only_nested_vs_volatility_only_direct": pairwise(
            VOLATILITY_ONLY_NESTED_ARM, VOLATILITY_ONLY_DIRECT_ARM
        ),
    }

    volatility_row = aggregate[VOLATILITY_ONLY_NESTED_ARM]
    distribution_positive = int(
        volatility_row.get("group_gap_closed", {})
        .get("distribution", {})
        .get("positive_seed_count", 0)
    )
    dependence_positive = int(
        volatility_row.get("group_gap_closed", {})
        .get("dependence", {})
        .get("positive_seed_count", 0)
    )
    local_gap_mean = float(
        volatility_row.get("group_gap_closed", {})
        .get("local_calibration", {})
        .get("median_gap_closed", {})
        .get("mean", float("-inf"))
    )
    versus_full = pairwise_results[
        "volatility_only_nested_vs_full_control_nested"
    ]
    distribution_vs_full = float(
        versus_full.get("group_relative_error_reduction", {})
        .get("distribution", {})
        .get("median_across_metrics", {})
        .get("mean", float("-inf"))
    )
    dependence_vs_full = float(
        versus_full.get("group_relative_error_reduction", {})
        .get("dependence", {})
        .get("median_across_metrics", {})
        .get("mean", float("-inf"))
    )
    gate = {
        "complete_four_of_four": volatility_row["complete_count"] == len(args.seeds),
        "distribution_median_improves_at_least_three_seeds": distribution_positive
        >= 3,
        "dependence_median_improves_at_least_three_seeds": dependence_positive >= 3,
        "local_calibration_median_degradation_at_most_ten_percent": local_gap_mean
        >= -0.10,
        "broad_distribution_competitive_with_full_control": distribution_vs_full
        >= -0.10,
        "dependence_not_worse_than_full_control": dependence_vs_full >= 0.0,
    }
    gate["advance_to_longer_horizons"] = all(bool(value) for value in gate.values())

    payload = {
        "protocol": {
            "sequence_length": sequence_length,
            "seeds": [int(value) for value in args.seeds],
            "decision_split": "disc.npy",
            "test_split_used": False,
            "subsample_size": int(args.subsample_size),
            "swd_projections": int(args.swd_projections),
            "metric_direction": "lower_is_better",
            "matched_reference_gap_closed": "(reference_error-candidate_error)/reference_error",
        },
        "metric_groups": {
            "distribution": list(PRIMARY_DISTRIBUTION_METRICS),
            "dependence": list(DEPENDENCE_METRICS),
            "local_calibration": list(LOCAL_CALIBRATION_METRICS),
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "pairwise": pairwise_results,
        "n128_gate": gate,
        "sources": {
            "run_root": str(run_root),
            "dataset_root": str(dataset_root),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"aggregate": aggregate, "n128_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
