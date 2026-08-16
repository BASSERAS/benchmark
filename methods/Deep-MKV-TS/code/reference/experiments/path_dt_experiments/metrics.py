from __future__ import annotations

import math
import operator
from typing import Sequence

import torch


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


def _require_finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_paths(paths: torch.Tensor, *, name: str = "paths") -> torch.Tensor:
    if not isinstance(paths, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if paths.ndim != 3:
        raise ValueError(f"{name} must have shape (B, N + 1, d)")
    if int(paths.shape[0]) < 1:
        raise ValueError(f"{name} must contain at least one path")
    if int(paths.shape[1]) < 2:
        raise ValueError(f"{name} must contain at least two time points")
    if int(paths.shape[2]) < 1:
        raise ValueError(f"{name} must contain at least one state dimension")
    if not torch.is_floating_point(paths):
        paths = paths.float()
    return paths


def _check_same_path_shape(left: torch.Tensor, right: torch.Tensor) -> None:
    if int(left.shape[1]) != int(right.shape[1]):
        raise ValueError("path time dimensions must match")
    if int(left.shape[2]) != int(right.shape[2]):
        raise ValueError("path state dimensions must match")


def _flatten_batch_time(values: torch.Tensor) -> torch.Tensor:
    return values.reshape(int(values.shape[0]) * int(values.shape[1]), int(values.shape[2]))


def _rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(values.detach().pow(2))).item())


def path_returns(paths: torch.Tensor) -> torch.Tensor:
    """Return increments with shape ``(B, N, d)``."""
    paths = _validate_paths(paths)
    return paths[:, 1:, :] - paths[:, :-1, :]


def return_covariance(paths: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Empirical covariance of all path returns pooled over batch and time."""
    eps = _require_finite_float("eps", eps)
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    returns = _flatten_batch_time(path_returns(paths))
    centered = returns - returns.mean(dim=0, keepdim=True)
    denom = max(1, int(centered.shape[0]) - 1)
    return centered.transpose(0, 1) @ centered / float(denom)


def return_correlation(paths: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Empirical return correlation matrix pooled over batch and time."""
    eps = _require_finite_float("eps", eps)
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    cov = return_covariance(paths, eps=eps)
    scale = torch.sqrt(torch.diagonal(cov).clamp_min(eps))
    return cov / (scale[:, None] * scale[None, :]).clamp_min(eps)


def excess_kurtosis(returns: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Excess kurtosis by asset for a return tensor with shape ``(B, N, d)``."""
    if returns.ndim != 3:
        raise ValueError("returns must have shape (B, N, d)")
    eps = _require_finite_float("eps", eps)
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    centered = returns - returns.mean(dim=(0, 1), keepdim=True)
    variance = torch.mean(centered.pow(2), dim=(0, 1)).clamp_min(eps)
    fourth = torch.mean(centered.pow(4), dim=(0, 1))
    return fourth / variance.pow(2) - 3.0


def acf(values: torch.Tensor, *, lags: Sequence[int], eps: float = 1e-12) -> torch.Tensor:
    """
    Pooled autocorrelation by asset.

    ``values`` must have shape ``(B, N, d)``.  The output has shape
    ``(len(lags), d)``.
    """
    if values.ndim != 3:
        raise ValueError("values must have shape (B, N, d)")
    eps = _require_finite_float("eps", eps)
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    lag_values = tuple(_require_positive_int("lags", lag) for lag in lags)
    if len(lag_values) == 0:
        raise ValueError("lags must be non-empty")
    num_steps = int(values.shape[1])
    if any(lag >= num_steps for lag in lag_values):
        raise ValueError("lags must satisfy lag < number of time points")

    centered = values - values.mean(dim=(0, 1), keepdim=True)
    denominator = torch.mean(centered.pow(2), dim=(0, 1)).clamp_min(eps)
    correlations = []
    for lag in lag_values:
        numerator = torch.mean(centered[:, :-lag, :] * centered[:, lag:, :], dim=(0, 1))
        correlations.append(numerator / denominator)
    return torch.stack(correlations, dim=0)


def tail_survival(abs_returns: torch.Tensor, *, thresholds: torch.Tensor) -> torch.Tensor:
    """
    Survival probabilities ``P(|r| > threshold)`` by asset.

    ``abs_returns`` must have shape ``(B, N, d)`` and ``thresholds`` must have
    shape ``(Q, d)`` or ``(Q, 1)``.
    """
    if abs_returns.ndim != 3:
        raise ValueError("abs_returns must have shape (B, N, d)")
    if thresholds.ndim != 2:
        raise ValueError("thresholds must have shape (Q, d)")
    if int(thresholds.shape[1]) not in {1, int(abs_returns.shape[2])}:
        raise ValueError("thresholds second dimension must be 1 or match state dimension")
    thresholds = thresholds.to(device=abs_returns.device, dtype=abs_returns.dtype)
    values = abs_returns.unsqueeze(0)
    levels = thresholds[:, None, None, :]
    return (values > levels).to(abs_returns.dtype).mean(dim=(1, 2))


def sliced_wasserstein_distance(
    left_paths: torch.Tensor,
    right_paths: torch.Tensor,
    *,
    num_projections: int = 128,
    seed: int | None = 1234,
    p: float = 2.0,
) -> float:
    """
    Sliced Wasserstein distance between flattened path distributions.

    The two batches may have different sizes.  Quantile grids are used after
    random projection so the comparison is distributional rather than paired.
    """
    left_paths = _validate_paths(left_paths, name="left_paths")
    right_paths = _validate_paths(right_paths, name="right_paths").to(device=left_paths.device, dtype=left_paths.dtype)
    _check_same_path_shape(left_paths, right_paths)
    num_projections = _require_positive_int("num_projections", num_projections)
    p = _require_finite_float("p", p)
    if p <= 0.0:
        raise ValueError("p must be > 0")

    left = left_paths.reshape(int(left_paths.shape[0]), -1)
    right = right_paths.reshape(int(right_paths.shape[0]), -1)
    generator = torch.Generator(device=left.device if left.device.type == "cpu" else "cpu")
    if seed is not None:
        generator.manual_seed(_require_int("seed", seed))
    projections = torch.randn(
        int(left.shape[1]),
        num_projections,
        generator=generator,
        device=left.device if left.device.type == "cpu" else "cpu",
        dtype=left.dtype,
    ).to(device=left.device)
    projections = projections / torch.linalg.norm(projections, dim=0, keepdim=True).clamp_min(1e-12)
    left_projected = left @ projections
    right_projected = right @ projections
    quantile_count = max(int(left.shape[0]), int(right.shape[0]))
    quantiles = torch.linspace(0.0, 1.0, quantile_count, device=left.device, dtype=left.dtype)
    left_quantiles = torch.quantile(left_projected, quantiles, dim=0)
    right_quantiles = torch.quantile(right_projected, quantiles, dim=0)
    distance_power = torch.mean(torch.abs(left_quantiles - right_quantiles).pow(p))
    return float(distance_power.pow(1.0 / p).item())


def stylized_fact_metrics(
    generated_paths: torch.Tensor,
    target_paths: torch.Tensor,
    *,
    lags: Sequence[int] = (1, 5, 10),
    tail_quantiles: Sequence[float] = (0.90, 0.95, 0.99),
    include_swd: bool = True,
    swd_projections: int = 128,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute a compact stylized-fact comparison between generated and target paths."""
    generated_paths = _validate_paths(generated_paths, name="generated_paths")
    target_paths = _validate_paths(target_paths, name="target_paths").to(
        device=generated_paths.device,
        dtype=generated_paths.dtype,
    )
    _check_same_path_shape(generated_paths, target_paths)
    eps = _require_finite_float("eps", eps)
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    lag_values = tuple(_require_positive_int("lags", lag) for lag in lags)
    if len(lag_values) == 0:
        raise ValueError("lags must be non-empty")
    tail_q = tuple(_require_finite_float("tail_quantiles", value) for value in tail_quantiles)
    if len(tail_q) == 0:
        raise ValueError("tail_quantiles must be non-empty")
    if any(value <= 0.0 or value >= 1.0 for value in tail_q):
        raise ValueError("tail_quantiles must lie in (0, 1)")

    generated_returns = path_returns(generated_paths)
    target_returns = path_returns(target_paths)
    generated_abs = generated_returns.abs()
    target_abs = target_returns.abs()
    generated_squared = generated_returns.pow(2)
    target_squared = target_returns.pow(2)

    generated_cov = return_covariance(generated_paths, eps=eps)
    target_cov = return_covariance(target_paths, eps=eps)
    generated_corr = return_correlation(generated_paths, eps=eps)
    target_corr = return_correlation(target_paths, eps=eps)

    flat_target_abs = _flatten_batch_time(target_abs)
    thresholds = torch.quantile(
        flat_target_abs,
        torch.tensor(tail_q, device=flat_target_abs.device, dtype=flat_target_abs.dtype),
        dim=0,
    )
    generated_survival = tail_survival(generated_abs, thresholds=thresholds)
    target_survival = tail_survival(target_abs, thresholds=thresholds)

    generated_abs_acf = acf(generated_abs, lags=lag_values, eps=eps)
    target_abs_acf = acf(target_abs, lags=lag_values, eps=eps)
    generated_squared_acf = acf(generated_squared, lags=lag_values, eps=eps)
    target_squared_acf = acf(target_squared, lags=lag_values, eps=eps)

    generated_kurtosis = excess_kurtosis(generated_returns, eps=eps)
    target_kurtosis = excess_kurtosis(target_returns, eps=eps)

    metrics = {
        "return_mean_error_rms": _rms(generated_returns.mean(dim=(0, 1)) - target_returns.mean(dim=(0, 1))),
        "return_std_error_rms": _rms(generated_returns.std(dim=(0, 1), unbiased=False) - target_returns.std(dim=(0, 1), unbiased=False)),
        "excess_kurtosis_error_rms": _rms(generated_kurtosis - target_kurtosis),
        "tail_survival_error_rms": _rms(generated_survival - target_survival),
        "abs_return_acf_error_rms": _rms(generated_abs_acf - target_abs_acf),
        "squared_return_acf_error_rms": _rms(generated_squared_acf - target_squared_acf),
        "return_covariance_error_rms": _rms(generated_cov - target_cov),
        "return_correlation_error_rms": _rms(generated_corr - target_corr),
        "generated_return_abs_mean": float(generated_abs.mean().item()),
        "target_return_abs_mean": float(target_abs.mean().item()),
        "generated_excess_kurtosis_mean": float(generated_kurtosis.mean().item()),
        "target_excess_kurtosis_mean": float(target_kurtosis.mean().item()),
    }
    if include_swd:
        metrics["path_swd"] = sliced_wasserstein_distance(
            generated_paths,
            target_paths,
            num_projections=swd_projections,
            seed=1234,
        )
    return metrics

