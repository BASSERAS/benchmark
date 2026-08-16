from __future__ import annotations

import math
from typing import Mapping

import torch

from deep_mkv_gen_path_dt.discrepancies.mmd import rbf_mmd2
from path_dt_experiments.garch_tree_oracle import (
    ExactGarchTreeProblem,
    GarchTreeSolution,
)
from path_dt_experiments.metrics import acf, sliced_wasserstein_distance


GARCH_DISTRIBUTION_METRICS = (
    "path_mmd2",
    "path_swd_normalized",
    "terminal_w1_normalized",
    "increment_w1_normalized",
    "realized_volatility_w1_normalized",
    "maximum_drawdown_w1_normalized",
    "return_qq_rmse_normalized",
)

GARCH_DEPENDENCE_METRICS = (
    "absolute_return_acf_rmse",
    "squared_return_acf_rmse",
    "squared_return_lag1_correlation_error",
    "next_absolute_return_response_gap_error",
)

GARCH_LOCAL_CALIBRATION_METRICS = (
    "timewise_return_mean_rmse_normalized",
    "timewise_return_std_rmse_normalized",
)


def _wasserstein_1(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().double().reshape(-1)
    right = right.detach().double().reshape(-1)
    count = max(int(left.numel()), int(right.numel()))
    levels = torch.linspace(
        0.0, 1.0, count, device=left.device, dtype=torch.float64
    )
    return float(
        torch.mean(
            torch.abs(
                torch.quantile(left, levels) - torch.quantile(right, levels)
            )
        ).item()
    )


def _maximum_drawdown(paths: torch.Tensor) -> torch.Tensor:
    return (torch.cummax(paths, dim=1).values - paths).amax(dim=1)


def _garch_dependence(paths: torch.Tensor) -> dict[str, float]:
    returns = torch.diff(paths, dim=1)
    current = returns[:, :-1].reshape(-1)
    following = returns[:, 1:].reshape(-1)
    current_square = current.pow(2)
    following_square = following.pow(2)
    correlation = float(
        torch.corrcoef(torch.stack((current_square, following_square)))[
            0, 1
        ].item()
    )
    absolute = current.abs()
    lower = torch.quantile(absolute, 1.0 / 3.0)
    upper = torch.quantile(absolute, 2.0 / 3.0)
    low_mask = absolute <= lower
    high_mask = absolute >= upper
    response_gap = float(
        following[high_mask].abs().mean().item()
        - following[low_mask].abs().mean().item()
    )
    return {
        "squared_return_lag1_correlation": correlation,
        "next_absolute_return_response_gap": response_gap,
    }


def garch_tree_metric_errors(
    problem: ExactGarchTreeProblem,
    solution: GarchTreeSolution,
    *,
    swd_projections: int = 512,
    swd_seed: int = 20260805,
) -> dict[str, float]:
    candidate = solution.leaf_paths.detach().double().unsqueeze(-1)
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
    path_mmd = rbf_mmd2(
        problem._features(solution.leaf_paths),
        problem.target_features,
        bandwidths=problem.config.bandwidths,
        unbiased=False,
    )
    quantiles = torch.linspace(
        0.01, 0.99, 99, device=candidate.device, dtype=torch.float64
    )
    candidate_quantiles = torch.quantile(candidate_returns.reshape(-1), quantiles)
    target_quantiles = torch.quantile(target_returns.reshape(-1), quantiles)
    lags = tuple(range(1, min(3, int(problem.config.horizon) - 1) + 1))
    candidate_abs_acf = acf(candidate_returns.abs(), lags=lags)
    target_abs_acf = acf(target_returns.abs(), lags=lags)
    candidate_squared_acf = acf(candidate_returns.pow(2), lags=lags)
    target_squared_acf = acf(target_returns.pow(2), lags=lags)
    candidate_time_mean = candidate_returns.mean(dim=0)
    target_time_mean = target_returns.mean(dim=0)
    candidate_time_std = candidate_returns.std(dim=0, unbiased=False)
    target_time_std = target_returns.std(dim=0, unbiased=False)
    mean_target_time_std = target_time_std.mean().clamp_min(1e-12)
    candidate_dependence = _garch_dependence(candidate[..., 0])
    target_dependence = _garch_dependence(target[..., 0])

    return {
        "path_mmd2": float(path_mmd.item()),
        "path_swd_normalized": sliced_wasserstein_distance(
            candidate / target_path_scale,
            target / target_path_scale,
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
            torch.mean((candidate_quantiles - target_quantiles).pow(2))
            .sqrt()
            .item()
            / target_return_scale.item()
        ),
        "absolute_return_acf_rmse": float(
            torch.mean((candidate_abs_acf - target_abs_acf).pow(2))
            .sqrt()
            .item()
        ),
        "squared_return_acf_rmse": float(
            torch.mean((candidate_squared_acf - target_squared_acf).pow(2))
            .sqrt()
            .item()
        ),
        "squared_return_lag1_correlation_error": abs(
            candidate_dependence["squared_return_lag1_correlation"]
            - target_dependence["squared_return_lag1_correlation"]
        ),
        "next_absolute_return_response_gap_error": abs(
            candidate_dependence["next_absolute_return_response_gap"]
            - target_dependence["next_absolute_return_response_gap"]
        ),
        "timewise_return_mean_rmse_normalized": float(
            torch.mean((candidate_time_mean - target_time_mean).pow(2))
            .sqrt()
            .item()
            / target_return_scale.item()
        ),
        "timewise_return_std_rmse_normalized": float(
            torch.mean((candidate_time_std - target_time_std).pow(2))
            .sqrt()
            .item()
            / mean_target_time_std.item()
        ),
    }


def gap_closed(
    reference_errors: Mapping[str, float],
    candidate_errors: Mapping[str, float],
    *,
    zero_tolerance: float = 1e-12,
) -> dict[str, float | None]:
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
    closure: Mapping[str, float | None], names: tuple[str, ...]
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
        "mean_gap_closed": float(tensor.mean().item()),
    }


__all__ = [
    "GARCH_DEPENDENCE_METRICS",
    "GARCH_DISTRIBUTION_METRICS",
    "GARCH_LOCAL_CALIBRATION_METRICS",
    "gap_closed",
    "garch_tree_metric_errors",
    "summarize_gap_closure",
]
