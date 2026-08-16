from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class RVShadowDiagnosticResult:
    summary: dict[str, object]
    path_metrics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class RVShadowRegimeDiagnosticResult:
    """Per-prefix-regime realized-volatility forecast diagnostics."""

    summary: dict[str, object]
    path_metrics: dict[str, torch.Tensor]
    regime_indices: torch.Tensor
    regime_thresholds: torch.Tensor


def _validate_inputs(
    real_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    selected_indices: torch.Tensor,
    *,
    prefix_length: int,
    horizons: Sequence[int],
) -> tuple[int, ...]:
    if real_paths.ndim != 3 or generated_paths.ndim != 3:
        raise ValueError("real_paths and generated_paths must have shape (B, N + 1, d)")
    if tuple(real_paths.shape[1:]) != tuple(generated_paths.shape[1:]):
        raise ValueError("real and generated paths must have matching path shapes")
    if selected_indices.ndim != 2 or int(selected_indices.shape[0]) != int(real_paths.shape[0]):
        raise ValueError("selected_indices must have shape (real paths, top_k)")
    if int(selected_indices.shape[1]) < 2:
        raise ValueError("selected_indices must contain at least two neighbors per real path")
    if int(selected_indices.min().item()) < 0 or int(selected_indices.max().item()) >= int(
        generated_paths.shape[0]
    ):
        raise ValueError("selected_indices contain an out-of-range generated path")
    prefix_length = int(prefix_length)
    if prefix_length < 2 or prefix_length >= int(real_paths.shape[1]):
        raise ValueError("prefix_length must lie between 2 and the number of path points - 1")
    values = tuple(int(value) for value in horizons)
    maximum = int(real_paths.shape[1]) - prefix_length
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError("horizons must be strictly increasing without duplicates")
    if any(value < 2 or value > maximum for value in values):
        raise ValueError("each horizon must lie between 2 and the available future increments")
    return values


def _scalar_interval_path_metrics(
    samples: torch.Tensor,
    realized: torch.Tensor,
    *,
    name: str,
) -> tuple[dict[str, float | int], dict[str, torch.Tensor]]:
    if samples.ndim != 3 or realized.ndim != 2:
        raise ValueError("scalar forecast inputs must have shapes (B, K, d) and (B, d)")
    quantiles = torch.quantile(
        samples,
        torch.tensor((0.05, 0.95), device=samples.device, dtype=samples.dtype),
        dim=1,
    )
    lower, upper = quantiles[0], quantiles[1]
    lower_miss = (realized < lower).to(realized.dtype).mean(dim=1)
    upper_miss = (realized > upper).to(realized.dtype).mean(dim=1)
    covered = ((realized >= lower) & (realized <= upper)).to(realized.dtype).mean(dim=1)
    width = (upper - lower).mean(dim=1)
    path_metrics = {
        f"{name}_coverage_90": covered.detach().cpu(),
        f"{name}_lower_miss": lower_miss.detach().cpu(),
        f"{name}_upper_miss": upper_miss.detach().cpu(),
        f"{name}_band_width_90": width.detach().cpu(),
    }
    summary: dict[str, float | int] = {
        "coverage_90": float(covered.mean().item()),
        "lower_miss_rate": float(lower_miss.mean().item()),
        "upper_miss_rate": float(upper_miss.mean().item()),
        "band_width_90": float(width.mean().item()),
        "covered_paths": int(torch.sum(covered > 0.5).item()),
        "lower_miss_paths": int(torch.sum(lower_miss > 0.5).item()),
        "upper_miss_paths": int(torch.sum(upper_miss > 0.5).item()),
    }
    return summary, path_metrics


def _conditional_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    selected = values[mask]
    return None if int(selected.numel()) == 0 else float(selected.mean().item())


def _empirical_scalar_crps_per_path(
    samples: torch.Tensor,
    realized: torch.Tensor,
) -> torch.Tensor:
    """Empirical CRPS averaged over assets, independently for each path."""

    scenario_count = int(samples.shape[1])
    mean_abs = torch.mean(torch.abs(samples - realized[:, None, :]), dim=1)
    sorted_values, _ = torch.sort(samples, dim=1)
    coefficients = (
        2.0
        * torch.arange(
            1,
            scenario_count + 1,
            device=samples.device,
            dtype=samples.dtype,
        )
        - float(scenario_count)
        - 1.0
    )
    pairwise_abs_mean = (
        2.0
        * torch.sum(sorted_values * coefficients[None, :, None], dim=1)
        / float(scenario_count**2)
    )
    return torch.mean(mean_abs - 0.5 * pairwise_abs_mean, dim=1)


def _masked_metric_summary(
    path_metrics: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> dict[str, float | int]:
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError("every prefix-volatility regime must contain at least one path")
    return {
        "count": count,
        **{
            name: float(values[mask].to(dtype=torch.float64).mean().item())
            for name, values in path_metrics.items()
        },
    }


def diagnose_shadow_by_prefix_volatility_regime(
    *,
    real_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    selected_indices: torch.Tensor,
    prefix_length: int,
    future_horizon: int,
    prefix_window: int = 16,
) -> RVShadowRegimeDiagnosticResult:
    """Decompose RV interval calibration by observed prefix-volatility tertile.

    Regimes are defined from the real prefixes shared by every candidate law,
    so model/oracle comparisons never condition on candidate-generated states.
    Bias is forecast mean minus realized RV: a negative value indicates that
    the conditional forecast is centered below the realized continuation.
    """

    (future_horizon,) = _validate_inputs(
        real_paths,
        generated_paths,
        selected_indices,
        prefix_length=prefix_length,
        horizons=(int(future_horizon),),
    )
    prefix_window = int(prefix_window)
    prefix_return_count = int(prefix_length) - 1
    if prefix_window < 1 or prefix_window > prefix_return_count:
        raise ValueError("prefix_window must fit within the observed prefix returns")

    generated = generated_paths.to(device=real_paths.device, dtype=real_paths.dtype)
    indices = selected_indices.to(device=generated.device)
    start = int(prefix_length) - 1
    selected_paths = generated[
        indices,
        start : start + int(future_horizon) + 1,
        :,
    ]
    real_continuation = real_paths[
        :,
        start : start + int(future_horizon) + 1,
        :,
    ]
    sample_increments = selected_paths[:, :, 1:, :] - selected_paths[:, :, :-1, :]
    realized_increments = real_continuation[:, 1:, :] - real_continuation[:, :-1, :]
    sample_rv = torch.sqrt(sample_increments.pow(2).sum(dim=2).clamp_min(0.0))
    realized_rv = torch.sqrt(realized_increments.pow(2).sum(dim=1).clamp_min(0.0))
    quantiles = torch.quantile(
        sample_rv,
        torch.tensor((0.05, 0.95), device=sample_rv.device, dtype=sample_rv.dtype),
        dim=1,
    )
    lower, upper = quantiles[0], quantiles[1]
    forecast_mean = sample_rv.mean(dim=1)

    real_prefix = real_paths[:, : int(prefix_length), :]
    prefix_increments = real_prefix[:, 1:, :] - real_prefix[:, :-1, :]
    recent_prefix = prefix_increments[:, -prefix_window:, :]
    prefix_rv = torch.sqrt(recent_prefix.pow(2).mean(dim=(1, 2)).clamp_min(0.0))
    thresholds = torch.quantile(
        prefix_rv.detach().double(),
        torch.tensor((1.0 / 3.0, 2.0 / 3.0), dtype=torch.float64),
    )
    regime_indices = torch.bucketize(prefix_rv.detach().double(), thresholds)

    path_metrics = {
        "coverage_90": ((realized_rv >= lower) & (realized_rv <= upper))
        .to(realized_rv.dtype)
        .mean(dim=1),
        "lower_miss_rate": (realized_rv < lower).to(realized_rv.dtype).mean(dim=1),
        "upper_miss_rate": (realized_rv > upper).to(realized_rv.dtype).mean(dim=1),
        "band_width_90": (upper - lower).mean(dim=1),
        "crps": _empirical_scalar_crps_per_path(sample_rv, realized_rv),
        "mean_bias": (forecast_mean - realized_rv).mean(dim=1),
        "absolute_error": torch.abs(forecast_mean - realized_rv).mean(dim=1),
        "forecast_mean_rv": forecast_mean.mean(dim=1),
        "realized_rv": realized_rv.mean(dim=1),
        "prefix_rms_volatility": prefix_rv,
    }
    path_metrics_cpu = {
        name: values.detach().cpu() for name, values in path_metrics.items()
    }
    regime_indices_cpu = regime_indices.detach().cpu()
    regimes = {
        name: _masked_metric_summary(
            path_metrics_cpu,
            regime_indices_cpu == regime_number,
        )
        for regime_number, name in enumerate(("low", "middle", "high"))
    }
    overall_mask = torch.ones_like(regime_indices_cpu, dtype=torch.bool)
    return RVShadowRegimeDiagnosticResult(
        summary={
            "num_real_paths": int(real_paths.shape[0]),
            "top_k": int(selected_indices.shape[1]),
            "prefix_length": int(prefix_length),
            "prefix_window": prefix_window,
            "future_horizon": int(future_horizon),
            "regime_definition": "observed-prefix RMS-volatility tertiles",
            "regime_thresholds": [float(value.item()) for value in thresholds],
            "overall": _masked_metric_summary(path_metrics_cpu, overall_mask),
            "regimes": regimes,
        },
        path_metrics=path_metrics_cpu,
        regime_indices=regime_indices_cpu,
        regime_thresholds=thresholds.detach().cpu(),
    )


def paired_bootstrap_regime_difference(
    primary: RVShadowRegimeDiagnosticResult,
    comparison: RVShadowRegimeDiagnosticResult,
    *,
    num_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Paired primary-minus-comparison bootstrap within each real-prefix regime."""

    if int(num_replicates) < 2:
        raise ValueError("num_replicates must be >= 2")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if set(primary.path_metrics) != set(comparison.path_metrics):
        raise ValueError("regime diagnostics must expose identical path metrics")
    if not torch.equal(primary.regime_indices, comparison.regime_indices):
        raise ValueError("regime diagnostics must use identical real-prefix regimes")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    alpha = 0.5 * (1.0 - float(confidence_level))
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    masks = {
        "overall": torch.ones_like(primary.regime_indices, dtype=torch.bool),
        **{
            name: primary.regime_indices == regime_number
            for regime_number, name in enumerate(("low", "middle", "high"))
        },
    }
    for regime_name, mask in masks.items():
        selected_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        count = int(selected_indices.numel())
        if count == 0:
            raise ValueError("every prefix-volatility regime must contain at least one path")
        bootstrap_positions = torch.randint(
            count,
            (int(num_replicates), count),
            generator=generator,
        )
        sampled_indices = selected_indices[bootstrap_positions]
        metrics: dict[str, dict[str, float]] = {}
        for name in sorted(primary.path_metrics):
            difference = (
                primary.path_metrics[name].to(dtype=torch.float64)
                - comparison.path_metrics[name].to(dtype=torch.float64)
            )
            selected_difference = difference[selected_indices]
            replicate_means = difference[sampled_indices].mean(dim=1)
            metrics[name] = {
                "estimate": float(selected_difference.mean().item()),
                "standard_error": float(replicate_means.std(unbiased=False).item()),
                "ci_lower": float(torch.quantile(replicate_means, alpha).item()),
                "ci_upper": float(torch.quantile(replicate_means, 1.0 - alpha).item()),
                "confidence_level": float(confidence_level),
                "num_paths": float(count),
                "num_replicates": float(num_replicates),
            }
        summaries[regime_name] = metrics
    return summaries


def diagnose_shadow_realized_variation(
    *,
    real_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    selected_indices: torch.Tensor,
    prefix_length: int,
    horizons: Sequence[int] = (8, 16, 32),
    eps: float = 1e-12,
) -> RVShadowDiagnosticResult:
    """Separate isolated-shock and persistent-variation shadowing failures."""

    horizon_values = _validate_inputs(
        real_paths,
        generated_paths,
        selected_indices,
        prefix_length=prefix_length,
        horizons=horizons,
    )
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be positive and finite")
    generated = generated_paths.to(device=real_paths.device, dtype=real_paths.dtype)
    indices = selected_indices.to(device=generated.device)
    start = int(prefix_length) - 1
    maximum_horizon = max(horizon_values)
    selected_paths = generated[indices, start : start + maximum_horizon + 1, :]
    real_continuation = real_paths[:, start : start + maximum_horizon + 1, :]
    generated_increments = selected_paths[:, :, 1:, :] - selected_paths[:, :, :-1, :]
    real_increments = real_continuation[:, 1:, :] - real_continuation[:, :-1, :]

    summary_by_horizon: dict[str, object] = {}
    all_path_metrics: dict[str, torch.Tensor] = {}
    for horizon in horizon_values:
        samples = generated_increments[:, :, :horizon, :]
        realized = real_increments[:, :horizon, :]
        sample_squared = samples.pow(2)
        realized_squared = realized.pow(2)
        sample_sum = sample_squared.sum(dim=2)
        realized_sum = realized_squared.sum(dim=1)
        sample_max = sample_squared.max(dim=2).values
        realized_max = realized_squared.max(dim=1).values
        sample_rv = torch.sqrt(sample_sum.clamp_min(0.0))
        realized_rv = torch.sqrt(realized_sum.clamp_min(0.0))
        sample_leave_max = torch.sqrt((sample_sum - sample_max).clamp_min(0.0))
        realized_leave_max = torch.sqrt((realized_sum - realized_max).clamp_min(0.0))
        sample_bipower = torch.sqrt(
            (0.5 * math.pi)
            * (samples[:, :, 1:, :].abs() * samples[:, :, :-1, :].abs()).sum(dim=2)
        )
        realized_bipower = torch.sqrt(
            (0.5 * math.pi)
            * (realized[:, 1:, :].abs() * realized[:, :-1, :].abs()).sum(dim=1)
        )
        sample_max_abs = samples.abs().max(dim=2).values
        realized_max_abs = realized.abs().max(dim=1).values

        statistics = {
            "realized_volatility": (sample_rv, realized_rv),
            "leave_largest_return_out_volatility": (sample_leave_max, realized_leave_max),
            "bipower_volatility": (sample_bipower, realized_bipower),
            "maximum_absolute_return": (sample_max_abs, realized_max_abs),
        }
        horizon_summary: dict[str, object] = {}
        statistic_path_metrics: dict[str, dict[str, torch.Tensor]] = {}
        for statistic, (sample_values, realized_values) in statistics.items():
            prefix = f"h{horizon}_{statistic}"
            statistic_summary, path_metrics = _scalar_interval_path_metrics(
                sample_values,
                realized_values,
                name=prefix,
            )
            horizon_summary[statistic] = statistic_summary
            statistic_path_metrics[statistic] = path_metrics
            all_path_metrics.update(path_metrics)

        dominance_realized = (realized_max / realized_sum.clamp_min(float(eps))).mean(dim=1)
        dominance_samples = sample_max / sample_sum.clamp_min(float(eps))
        rv_upper_miss = statistic_path_metrics["realized_volatility"][
            f"h{horizon}_realized_volatility_upper_miss"
        ].to(device=dominance_realized.device) > 0.5
        rv_lower_miss = statistic_path_metrics["realized_volatility"][
            f"h{horizon}_realized_volatility_lower_miss"
        ].to(device=dominance_realized.device) > 0.5
        rv_covered = statistic_path_metrics["realized_volatility"][
            f"h{horizon}_realized_volatility_coverage_90"
        ].to(device=dominance_realized.device) > 0.5
        dominance_summary = {
            "realized_mean": float(dominance_realized.mean().item()),
            "realized_median": float(torch.quantile(dominance_realized, 0.5).item()),
            "realized_q90": float(torch.quantile(dominance_realized, 0.9).item()),
            "selected_scenarios_mean": float(dominance_samples.mean().item()),
            "selected_scenarios_median": float(torch.quantile(dominance_samples, 0.5).item()),
            "selected_scenarios_q90": float(torch.quantile(dominance_samples, 0.9).item()),
            "realized_mean_when_rv_upper_miss": _conditional_mean(
                dominance_realized,
                rv_upper_miss,
            ),
            "realized_mean_when_rv_lower_miss": _conditional_mean(
                dominance_realized,
                rv_lower_miss,
            ),
            "realized_mean_when_rv_covered": _conditional_mean(
                dominance_realized,
                rv_covered,
            ),
        }
        horizon_summary["largest_return_squared_share"] = dominance_summary
        standard_coverage = float(
            dict(horizon_summary["realized_volatility"])["coverage_90"]
        )
        leave_out_coverage = float(
            dict(horizon_summary["leave_largest_return_out_volatility"])["coverage_90"]
        )
        horizon_summary["coverage_change_after_removing_largest_return"] = float(
            leave_out_coverage - standard_coverage
        )
        summary_by_horizon[str(horizon)] = horizon_summary

    return RVShadowDiagnosticResult(
        summary={
            "num_real_paths": int(real_paths.shape[0]),
            "top_k": int(selected_indices.shape[1]),
            "prefix_length": int(prefix_length),
            "horizons": summary_by_horizon,
        },
        path_metrics=all_path_metrics,
    )


def paired_bootstrap_diagnostic_difference(
    primary: RVShadowDiagnosticResult,
    comparison: RVShadowDiagnosticResult,
    *,
    num_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Paired path bootstrap of primary-minus-comparison diagnostic metrics."""

    if int(num_replicates) < 2:
        raise ValueError("num_replicates must be >= 2")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if set(primary.path_metrics) != set(comparison.path_metrics):
        raise ValueError("diagnostic results must expose identical path metrics")
    counts = {
        (int(primary.path_metrics[name].numel()), int(comparison.path_metrics[name].numel()))
        for name in primary.path_metrics
    }
    if len(counts) != 1:
        raise ValueError("diagnostic path metrics must have consistent lengths")
    primary_count, comparison_count = counts.pop()
    if primary_count != comparison_count:
        raise ValueError("diagnostic results must contain the same real paths")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        primary_count,
        (int(num_replicates), primary_count),
        generator=generator,
    )
    alpha = 0.5 * (1.0 - float(confidence_level))
    summaries: dict[str, dict[str, float]] = {}
    for name in sorted(primary.path_metrics):
        difference = primary.path_metrics[name].to(dtype=torch.float64).reshape(-1) - comparison.path_metrics[
            name
        ].to(dtype=torch.float64).reshape(-1)
        replicate_means = difference[indices].mean(dim=1)
        summaries[name] = {
            "estimate": float(difference.mean().item()),
            "standard_error": float(replicate_means.std(unbiased=False).item()),
            "ci_lower": float(torch.quantile(replicate_means, alpha).item()),
            "ci_upper": float(torch.quantile(replicate_means, 1.0 - alpha).item()),
            "confidence_level": float(confidence_level),
            "num_paths": float(primary_count),
            "num_replicates": float(num_replicates),
        }
    return summaries


__all__ = [
    "RVShadowDiagnosticResult",
    "RVShadowRegimeDiagnosticResult",
    "diagnose_shadow_realized_variation",
    "diagnose_shadow_by_prefix_volatility_regime",
    "paired_bootstrap_diagnostic_difference",
    "paired_bootstrap_regime_difference",
]
