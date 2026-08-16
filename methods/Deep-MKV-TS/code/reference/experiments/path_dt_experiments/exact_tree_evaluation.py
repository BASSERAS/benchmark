from __future__ import annotations

import math
from typing import Mapping

import torch

from deep_mkv_gen_path_dt.discrepancies.mmd import rbf_mmd2
from path_dt_experiments.exact_scenario_tree import (
    ExactBinaryTreeProblem,
    ExactTreeSolution,
)
from path_dt_experiments.metrics import acf, sliced_wasserstein_distance


PRIMARY_DISTRIBUTION_METRICS = (
    "path_mmd2",
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
    "delayed_future_rv_gap_error",
    "delayed_future_rv_correlation_error",
)

LOCAL_CALIBRATION_METRICS = (
    "timewise_return_mean_rmse_normalized",
    "timewise_return_std_rmse_normalized",
)


def _paths(solution: ExactTreeSolution) -> torch.Tensor:
    return solution.leaf_paths.detach().double().unsqueeze(-1)


def _wasserstein_1(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().double().reshape(-1)
    right = right.detach().double().reshape(-1)
    count = max(int(left.numel()), int(right.numel()))
    levels = torch.linspace(
        0.0,
        1.0,
        count,
        device=left.device,
        dtype=torch.float64,
    )
    return float(
        torch.mean(
            torch.abs(torch.quantile(left, levels) - torch.quantile(right, levels))
        ).item()
    )


def _maximum_drawdown(paths: torch.Tensor) -> torch.Tensor:
    return (torch.cummax(paths, dim=1).values - paths).amax(dim=1)


def exact_tree_metric_errors(
    problem: ExactBinaryTreeProblem,
    solution: ExactTreeSolution,
    *,
    swd_projections: int = 1024,
    swd_seed: int = 20260804,
) -> dict[str, float]:
    """Evaluate a candidate law against the exact target leaf distribution.

    Every returned quantity is an error for which zero is ideal.  Scalar
    distribution distances are normalized by immutable target scales so that
    gap-closure fractions remain interpretable across metric families.
    """

    candidate = _paths(solution)
    target = problem.target.leaf_paths.detach().double().unsqueeze(-1)
    candidate_returns = torch.diff(candidate, dim=1)
    target_returns = torch.diff(target, dim=1)
    target_return_scale = target_returns.std(unbiased=False).clamp_min(1e-12)
    target_path_scale = target[:, 1:, :].std(unbiased=False).clamp_min(1e-12)
    target_terminal_scale = target[:, -1, :].std(unbiased=False).clamp_min(1e-12)

    candidate_rv = torch.sqrt(torch.sum(candidate_returns.pow(2), dim=1))
    target_rv = torch.sqrt(torch.sum(target_returns.pow(2), dim=1))
    target_rv_scale = target_rv.mean().clamp_min(1e-12)
    candidate_drawdown = _maximum_drawdown(candidate)
    target_drawdown = _maximum_drawdown(target)
    target_drawdown_scale = target_drawdown.mean().clamp_min(1e-12)

    standardized_candidate = (
        candidate[:, 1:, 0] - problem.feature_center
    ) / problem.feature_scale
    path_mmd = rbf_mmd2(
        standardized_candidate,
        problem.target_features,
        bandwidths=problem.config.bandwidths,
        unbiased=False,
    )
    normalized_candidate = candidate / target_path_scale
    normalized_target = target / target_path_scale

    quantiles = torch.linspace(
        0.01,
        0.99,
        99,
        device=candidate.device,
        dtype=torch.float64,
    )
    candidate_return_quantiles = torch.quantile(candidate_returns.reshape(-1), quantiles)
    target_return_quantiles = torch.quantile(target_returns.reshape(-1), quantiles)
    lag_values = tuple(range(1, min(3, int(problem.config.horizon) - 1) + 1))

    candidate_return_acf = acf(candidate_returns, lags=lag_values)
    target_return_acf = acf(target_returns, lags=lag_values)
    candidate_abs_acf = acf(candidate_returns.abs(), lags=lag_values)
    target_abs_acf = acf(target_returns.abs(), lags=lag_values)
    candidate_squared_acf = acf(candidate_returns.pow(2), lags=lag_values)
    target_squared_acf = acf(target_returns.pow(2), lags=lag_values)

    candidate_time_mean = candidate_returns.mean(dim=0)
    target_time_mean = target_returns.mean(dim=0)
    candidate_time_std = candidate_returns.std(dim=0, unbiased=False)
    target_time_std = target_returns.std(dim=0, unbiased=False)
    mean_target_time_std = target_time_std.mean().clamp_min(1e-12)

    memory = problem.delayed_memory_metrics(solution)
    target_memory = problem.delayed_memory_metrics(
        problem.solution(problem.oracle_start())
    )

    return {
        "path_mmd2": float(path_mmd.item()),
        "path_swd_normalized": sliced_wasserstein_distance(
            normalized_candidate,
            normalized_target,
            num_projections=int(swd_projections),
            seed=int(swd_seed),
        ),
        "terminal_w1_normalized": _wasserstein_1(
            candidate[:, -1, :], target[:, -1, :]
        )
        / float(target_terminal_scale.item()),
        "increment_w1_normalized": _wasserstein_1(
            candidate_returns, target_returns
        )
        / float(target_return_scale.item()),
        "realized_volatility_w1_normalized": _wasserstein_1(
            candidate_rv, target_rv
        )
        / float(target_rv_scale.item()),
        "maximum_drawdown_w1_normalized": _wasserstein_1(
            candidate_drawdown, target_drawdown
        )
        / float(target_drawdown_scale.item()),
        "return_qq_rmse_normalized": float(
            torch.mean(
                (candidate_return_quantiles - target_return_quantiles).pow(2)
            ).sqrt().item()
            / target_return_scale.item()
        ),
        "return_acf_rmse": float(
            torch.mean((candidate_return_acf - target_return_acf).pow(2)).sqrt().item()
        ),
        "absolute_return_acf_rmse": float(
            torch.mean((candidate_abs_acf - target_abs_acf).pow(2)).sqrt().item()
        ),
        "squared_return_acf_rmse": float(
            torch.mean((candidate_squared_acf - target_squared_acf).pow(2)).sqrt().item()
        ),
        "delayed_future_rv_gap_error": abs(
            float(memory["future_rv_hit_gap"])
            - float(target_memory["future_rv_hit_gap"])
        ),
        "delayed_future_rv_correlation_error": abs(
            float(memory["early_hit_future_rv_correlation"])
            - float(target_memory["early_hit_future_rv_correlation"])
        ),
        "timewise_return_mean_rmse_normalized": float(
            torch.mean((candidate_time_mean - target_time_mean).pow(2)).sqrt().item()
            / target_return_scale.item()
        ),
        "timewise_return_std_rmse_normalized": float(
            torch.mean((candidate_time_std - target_time_std).pow(2)).sqrt().item()
            / mean_target_time_std.item()
        ),
    }


def gap_closed(
    reference_errors: Mapping[str, float],
    candidate_errors: Mapping[str, float],
    *,
    zero_tolerance: float = 1e-12,
) -> dict[str, float | None]:
    """Return the fraction of each reference error removed by a candidate."""

    if set(reference_errors) != set(candidate_errors):
        raise ValueError("reference and candidate metric keys must match")
    result: dict[str, float | None] = {}
    for name, reference in reference_errors.items():
        denominator = float(reference)
        candidate = float(candidate_errors[name])
        if (
            not math.isfinite(denominator)
            or denominator <= float(zero_tolerance)
            or not math.isfinite(candidate)
        ):
            result[name] = None
        else:
            result[name] = (denominator - candidate) / denominator
    return result


def summarize_gap_closure(
    closure: Mapping[str, float | None],
    names: tuple[str, ...],
) -> dict[str, float | int | None]:
    values = [float(closure[name]) for name in names if closure[name] is not None]
    if not values:
        return {
            "metric_count": 0,
            "improved_count": 0,
            "median_gap_closed": None,
            "mean_gap_closed": None,
        }
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "metric_count": len(values),
        "improved_count": sum(value > 0.0 for value in values),
        "median_gap_closed": float(torch.quantile(tensor, 0.5).item()),
        "mean_gap_closed": float(torch.mean(tensor).item()),
    }


def fraction_of_exact_improvement_captured(
    one_stage_closure: Mapping[str, float | None],
    exact_closure: Mapping[str, float | None],
    names: tuple[str, ...],
) -> dict[str, float | int | None]:
    """Summarize one-stage closure relative to positive exact-MP closure."""

    ratios = []
    for name in names:
        one_stage = one_stage_closure[name]
        exact = exact_closure[name]
        if (
            one_stage is not None
            and exact is not None
            and math.isfinite(float(one_stage))
            and math.isfinite(float(exact))
            and float(exact) > 0.0
        ):
            ratios.append(float(one_stage) / float(exact))
    if not ratios:
        return {
            "metric_count": 0,
            "median_fraction_captured": None,
            "mean_fraction_captured": None,
        }
    tensor = torch.tensor(ratios, dtype=torch.float64)
    return {
        "metric_count": len(ratios),
        "median_fraction_captured": float(torch.quantile(tensor, 0.5).item()),
        "mean_fraction_captured": float(torch.mean(tensor).item()),
    }


__all__ = [
    "DEPENDENCE_METRICS",
    "LOCAL_CALIBRATION_METRICS",
    "PRIMARY_DISTRIBUTION_METRICS",
    "exact_tree_metric_errors",
    "fraction_of_exact_improvement_captured",
    "gap_closed",
    "summarize_gap_closure",
]
