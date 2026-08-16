from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Mapping, Sequence

import torch

from path_dt_experiments.metrics import acf, path_returns


REGIME_NAMES = ("low", "middle", "high")


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _require_positive_int(name: str, value: object) -> int:
    result = _require_int(name, value)
    if result < 1:
        raise ValueError(f"{name} must be >= 1")
    return result


def _require_probability(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must lie strictly between zero and one")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result >= 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


@dataclass(frozen=True)
class VolatilityPathStatistics:
    """Path-level quantities used by the volatility diagnostic report."""

    num_paths: int
    asset: int
    prefix_length: int
    prefix_window: int
    horizons: tuple[int, ...]
    lags: tuple[int, ...]
    prefix_rms_volatility: torch.Tensor
    future_realized_volatility: dict[int, torch.Tensor]
    absolute_return_acf: torch.Tensor
    squared_return_acf: torch.Tensor


def volatility_path_statistics(
    paths: torch.Tensor,
    *,
    prefix_length: int,
    prefix_window: int,
    horizons: Sequence[int],
    lags: Sequence[int],
    asset: int = 0,
) -> VolatilityPathStatistics:
    """
    Compute unconditional and prefix-conditioned volatility quantities.

    ``prefix_length`` counts observed path points, so the future starts at the
    return whose left endpoint is ``prefix_length - 1``. Prefix volatility is
    the RMS of the last ``prefix_window`` observed returns. Future realized
    volatility follows the shadowing convention ``sqrt(sum(r_t**2))``.
    """
    if not isinstance(paths, torch.Tensor) or paths.ndim != 3:
        raise ValueError("paths must be a torch.Tensor with shape (B, N + 1, d)")
    if int(paths.shape[0]) < 1 or int(paths.shape[1]) < 2 or int(paths.shape[2]) < 1:
        raise ValueError("paths must contain at least one path, two points, and one asset")
    if not torch.is_floating_point(paths):
        paths = paths.float()

    prefix_length = _require_positive_int("prefix_length", prefix_length)
    prefix_window = _require_positive_int("prefix_window", prefix_window)
    asset = _require_int("asset", asset)
    horizon_values = tuple(_require_positive_int("horizons", value) for value in horizons)
    lag_values = tuple(_require_positive_int("lags", value) for value in lags)
    if len(horizon_values) == 0:
        raise ValueError("horizons must be non-empty")
    if len(set(horizon_values)) != len(horizon_values):
        raise ValueError("horizons must not contain duplicates")
    if len(lag_values) == 0:
        raise ValueError("lags must be non-empty")
    if len(set(lag_values)) != len(lag_values):
        raise ValueError("lags must not contain duplicates")
    if asset < 0 or asset >= int(paths.shape[2]):
        raise ValueError("asset out of range")

    returns = path_returns(paths)[..., asset : asset + 1]
    num_steps = int(returns.shape[1])
    future_start = prefix_length - 1
    if future_start < prefix_window:
        raise ValueError("prefix_length - 1 must be >= prefix_window")
    if future_start >= num_steps:
        raise ValueError("prefix_length must leave at least one future return")
    if max(horizon_values) > num_steps - future_start:
        raise ValueError("each horizon must fit after the prefix endpoint")
    if max(lag_values) >= num_steps:
        raise ValueError("each lag must be smaller than the number of returns")

    prefix_returns = returns[:, future_start - prefix_window : future_start, 0]
    prefix_rms = torch.sqrt(torch.mean(prefix_returns.pow(2), dim=1).clamp_min(0.0))
    future_rv = {
        horizon: torch.sqrt(
            torch.sum(returns[:, future_start : future_start + horizon, 0].pow(2), dim=1).clamp_min(0.0)
        )
        for horizon in horizon_values
    }
    absolute_acf = acf(returns.abs(), lags=lag_values)[:, 0]
    squared_acf = acf(returns.pow(2), lags=lag_values)[:, 0]
    return VolatilityPathStatistics(
        num_paths=int(paths.shape[0]),
        asset=asset,
        prefix_length=prefix_length,
        prefix_window=prefix_window,
        horizons=horizon_values,
        lags=lag_values,
        prefix_rms_volatility=prefix_rms.detach().cpu(),
        future_realized_volatility={horizon: values.detach().cpu() for horizon, values in future_rv.items()},
        absolute_return_acf=absolute_acf.detach().cpu(),
        squared_return_acf=squared_acf.detach().cpu(),
    )


def _summary(values: torch.Tensor, *, quantiles: tuple[float, ...]) -> dict[str, object]:
    values = values.detach().double().reshape(-1)
    levels = torch.tensor(quantiles, dtype=values.dtype)
    quantile_values = torch.quantile(values, levels)
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
        "quantiles": {
            f"{level:g}": float(value.item())
            for level, value in zip(quantiles, quantile_values)
        },
    }


def _safe_ratio(numerator: float, denominator: float, *, eps: float = 1e-12) -> float | None:
    if abs(float(denominator)) <= eps:
        return None
    return float(numerator) / float(denominator)


def _correlation(left: torch.Tensor, right: torch.Tensor, *, eps: float = 1e-12) -> float:
    left = left.detach().double().reshape(-1)
    right = right.detach().double().reshape(-1)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.sqrt(torch.mean(left_centered.pow(2)) * torch.mean(right_centered.pow(2)))
    if float(denominator.item()) <= eps:
        return 0.0
    return float((torch.mean(left_centered * right_centered) / denominator).item())


def _regression_slope(predictor: torch.Tensor, response: torch.Tensor, *, eps: float = 1e-12) -> float:
    predictor = predictor.detach().double().reshape(-1)
    response = response.detach().double().reshape(-1)
    predictor_centered = predictor - predictor.mean()
    variance = torch.mean(predictor_centered.pow(2))
    if float(variance.item()) <= eps:
        return 0.0
    covariance = torch.mean(predictor_centered * (response - response.mean()))
    return float((covariance / variance).item())


def _quantile_wasserstein_approximation(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    num_quantiles: int = 999,
) -> float:
    levels = torch.linspace(0.001, 0.999, int(num_quantiles), dtype=torch.float64)
    left_quantiles = torch.quantile(left.detach().double().reshape(-1), levels)
    right_quantiles = torch.quantile(right.detach().double().reshape(-1), levels)
    return float(torch.mean(torch.abs(left_quantiles - right_quantiles)).item())


def _regime_indices(prefix_volatility: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(prefix_volatility.detach().double(), thresholds.detach().double())


def _dataset_report(
    statistics: VolatilityPathStatistics,
    *,
    regime_thresholds: torch.Tensor,
    quantiles: tuple[float, ...],
) -> dict[str, object]:
    regime_index = _regime_indices(statistics.prefix_rms_volatility, regime_thresholds)
    regime_payload: dict[str, object] = {}
    for regime_number, regime_name in enumerate(REGIME_NAMES):
        selected = regime_index == regime_number
        count = int(selected.sum().item())
        by_horizon: dict[str, object] = {}
        for horizon in statistics.horizons:
            values = statistics.future_realized_volatility[horizon][selected]
            by_horizon[str(horizon)] = None if count == 0 else _summary(values, quantiles=quantiles)
        regime_payload[regime_name] = {
            "count": count,
            "fraction": float(count / statistics.num_paths),
            "future_realized_volatility": by_horizon,
        }

    persistence: dict[str, object] = {}
    for horizon in statistics.horizons:
        future = statistics.future_realized_volatility[horizon]
        low_values = future[regime_index == 0]
        high_values = future[regime_index == 2]
        persistence[str(horizon)] = {
            "prefix_future_correlation": _correlation(statistics.prefix_rms_volatility, future),
            "prefix_future_regression_slope": _regression_slope(statistics.prefix_rms_volatility, future),
            "high_to_low_future_mean_ratio": None
            if low_values.numel() == 0 or high_values.numel() == 0
            else _safe_ratio(float(high_values.mean().item()), float(low_values.mean().item())),
        }

    return {
        "num_paths": statistics.num_paths,
        "prefix_rms_volatility": _summary(statistics.prefix_rms_volatility, quantiles=quantiles),
        "future_realized_volatility": {
            str(horizon): _summary(values, quantiles=quantiles)
            for horizon, values in statistics.future_realized_volatility.items()
        },
        "absolute_return_acf": {
            str(lag): float(value.item())
            for lag, value in zip(statistics.lags, statistics.absolute_return_acf)
        },
        "squared_return_acf": {
            str(lag): float(value.item())
            for lag, value in zip(statistics.lags, statistics.squared_return_acf)
        },
        "regimes": regime_payload,
        "persistence": persistence,
    }


def _distribution_comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | None]:
    reference = reference.detach().double().reshape(-1)
    candidate = candidate.detach().double().reshape(-1)
    reference_mean = float(reference.mean().item())
    candidate_mean = float(candidate.mean().item())
    reference_std = float(reference.std(unbiased=False).item())
    candidate_std = float(candidate.std(unbiased=False).item())
    reference_iqr = float((torch.quantile(reference, 0.75) - torch.quantile(reference, 0.25)).item())
    candidate_iqr = float((torch.quantile(candidate, 0.75) - torch.quantile(candidate, 0.25)).item())
    wasserstein = _quantile_wasserstein_approximation(reference, candidate)
    return {
        "candidate_to_reference_mean_ratio": _safe_ratio(candidate_mean, reference_mean),
        "candidate_to_reference_std_ratio": _safe_ratio(candidate_std, reference_std),
        "candidate_to_reference_iqr_ratio": _safe_ratio(candidate_iqr, reference_iqr),
        "quantile_wasserstein_1_approx": wasserstein,
        "normalized_quantile_wasserstein_1_approx": _safe_ratio(wasserstein, reference_mean),
    }


def build_volatility_diagnostic_report(
    statistics_by_label: Mapping[str, VolatilityPathStatistics],
    *,
    reference_label: str,
    quantiles: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> dict[str, object]:
    """Compare named path laws against one reference using common volatility regimes."""
    if reference_label not in statistics_by_label:
        raise ValueError("reference_label is not present in statistics_by_label")
    if len(statistics_by_label) < 2:
        raise ValueError("at least one candidate in addition to the reference is required")
    quantile_values = tuple(_require_probability("quantiles", value) for value in quantiles)
    if len(quantile_values) == 0:
        raise ValueError("quantiles must be non-empty")
    if tuple(sorted(quantile_values)) != quantile_values or len(set(quantile_values)) != len(quantile_values):
        raise ValueError("quantiles must be strictly increasing")

    reference = statistics_by_label[reference_label]
    for label, statistics in statistics_by_label.items():
        if statistics.horizons != reference.horizons:
            raise ValueError(f"{label} horizons do not match the reference")
        if statistics.lags != reference.lags:
            raise ValueError(f"{label} lags do not match the reference")
        if statistics.prefix_length != reference.prefix_length or statistics.prefix_window != reference.prefix_window:
            raise ValueError(f"{label} prefix configuration does not match the reference")

    regime_thresholds = torch.quantile(
        reference.prefix_rms_volatility.detach().double(),
        torch.tensor([1.0 / 3.0, 2.0 / 3.0], dtype=torch.float64),
    )
    datasets = {
        label: _dataset_report(
            statistics,
            regime_thresholds=regime_thresholds,
            quantiles=quantile_values,
        )
        for label, statistics in statistics_by_label.items()
    }

    comparisons: dict[str, object] = {}
    reference_regimes = _regime_indices(reference.prefix_rms_volatility, regime_thresholds)
    for label, candidate in statistics_by_label.items():
        if label == reference_label:
            continue
        candidate_regimes = _regime_indices(candidate.prefix_rms_volatility, regime_thresholds)
        conditional: dict[str, object] = {}
        for regime_number, regime_name in enumerate(REGIME_NAMES):
            reference_selected = reference_regimes == regime_number
            candidate_selected = candidate_regimes == regime_number
            conditional[regime_name] = {
                str(horizon): None
                if int(reference_selected.sum()) == 0 or int(candidate_selected.sum()) == 0
                else _distribution_comparison(
                    reference.future_realized_volatility[horizon][reference_selected],
                    candidate.future_realized_volatility[horizon][candidate_selected],
                )
                for horizon in reference.horizons
            }

        comparisons[label] = {
            "future_realized_volatility": {
                str(horizon): _distribution_comparison(
                    reference.future_realized_volatility[horizon],
                    candidate.future_realized_volatility[horizon],
                )
                for horizon in reference.horizons
            },
            "absolute_return_acf_error_rms": float(
                torch.sqrt(torch.mean((candidate.absolute_return_acf - reference.absolute_return_acf).double().pow(2))).item()
            ),
            "squared_return_acf_error_rms": float(
                torch.sqrt(torch.mean((candidate.squared_return_acf - reference.squared_return_acf).double().pow(2))).item()
            ),
            "conditional_future_realized_volatility": conditional,
            "persistence_difference": {
                str(horizon): {
                    "prefix_future_correlation_difference": _correlation(
                        candidate.prefix_rms_volatility,
                        candidate.future_realized_volatility[horizon],
                    )
                    - _correlation(
                        reference.prefix_rms_volatility,
                        reference.future_realized_volatility[horizon],
                    ),
                    "prefix_future_regression_slope_difference": _regression_slope(
                        candidate.prefix_rms_volatility,
                        candidate.future_realized_volatility[horizon],
                    )
                    - _regression_slope(
                        reference.prefix_rms_volatility,
                        reference.future_realized_volatility[horizon],
                    ),
                }
                for horizon in reference.horizons
            },
        }

    return {
        "reference_label": reference_label,
        "configuration": {
            "asset": reference.asset,
            "prefix_length": reference.prefix_length,
            "prefix_window": reference.prefix_window,
            "horizons": list(reference.horizons),
            "lags": list(reference.lags),
            "quantiles": list(quantile_values),
            "regime_definition": "reference prefix RMS-volatility tertiles",
            "regime_thresholds": [float(value.item()) for value in regime_thresholds],
        },
        "datasets": datasets,
        "comparisons_to_reference": comparisons,
    }


def volatility_statistics_artifact(
    statistics_by_label: Mapping[str, VolatilityPathStatistics],
) -> dict[str, object]:
    """Return a CPU tensor payload suitable for ``torch.save``."""
    return {
        label: {
            "num_paths": statistics.num_paths,
            "asset": statistics.asset,
            "prefix_length": statistics.prefix_length,
            "prefix_window": statistics.prefix_window,
            "horizons": statistics.horizons,
            "lags": statistics.lags,
            "prefix_rms_volatility": statistics.prefix_rms_volatility,
            "future_realized_volatility": statistics.future_realized_volatility,
            "absolute_return_acf": statistics.absolute_return_acf,
            "squared_return_acf": statistics.squared_return_acf,
        }
        for label, statistics in statistics_by_label.items()
    }
