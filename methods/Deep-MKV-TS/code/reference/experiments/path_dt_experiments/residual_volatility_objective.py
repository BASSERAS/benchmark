from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from deep_mkv_gen_path_dt.discrepancies import PathFunctionalDiscrepancyResult
from deep_mkv_gen_path_dt.discrepancies.mmd import (
    rbf_mmd2,
    rbf_mmd_lions_potential,
)
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


def _positive_finite(name: str, value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _reference_moments(
    paths: torch.Tensor,
    *,
    reference_kernel: object,
    grid: DiscreteTimeGrid,
    detach_reference_outputs: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    evaluate_all = getattr(reference_kernel, "evaluate_all", None)
    if not callable(evaluate_all):
        raise TypeError("residual volatility objective requires reference.evaluate_all")
    drift, sigma = evaluate_all(paths)
    expected = (
        int(paths.shape[0]),
        int(grid.num_steps),
        int(paths.shape[-1]),
    )
    if tuple(drift.shape) != expected or tuple(sigma.shape) != expected:
        raise ValueError(f"reference drift and volatility must have shape {expected}")
    if detach_reference_outputs:
        drift = drift.detach()
        sigma = sigma.detach()
    return drift, sigma.pow(2) * float(grid.dt)


def reference_residual_volatility_features(
    paths: torch.Tensor,
    *,
    reference_kernel: object,
    grid: DiscreteTimeGrid,
    split_index: int,
    prefix_window: int,
    future_horizon: int,
    prefix_variance_floor: float,
    future_variance_floor: float,
    drawdown_variance_floor: float,
    include_drawdown: bool,
    detach_reference_outputs: bool = False,
) -> torch.Tensor:
    """Dimensionless reference-residualized prefix/future variance features.

    Innovations are measured relative to the frozen causal reference's fitted
    conditional drift.  Both fitted drift and volatility are re-evaluated on
    generated paths, preserving their path derivatives; reference parameters
    and target-path evaluations remain frozen.
    """

    if paths.ndim != 3 or int(paths.shape[-1]) != 1:
        raise ValueError("residual volatility features require shape (B, N + 1, 1)")
    if int(paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("paths must align with the supplied grid")
    split = int(split_index)
    prefix = int(prefix_window)
    future = int(future_horizon)
    if prefix < 1 or future < 1 or split < prefix:
        raise ValueError("prefix window must fit before a positive split")
    if split + future > int(grid.num_steps):
        raise ValueError("future horizon must fit after the split")
    prefix_floor = _positive_finite("prefix_variance_floor", prefix_variance_floor)
    future_floor = _positive_finite("future_variance_floor", future_variance_floor)
    drawdown_floor = _positive_finite(
        "drawdown_variance_floor", drawdown_variance_floor
    )
    returns = paths[:, 1:, :] - paths[:, :-1, :]
    reference_drift, reference_variance = _reference_moments(
        paths,
        reference_kernel=reference_kernel,
        grid=grid,
        detach_reference_outputs=bool(detach_reference_outputs),
    )

    def log_ratio(left: int, right: int, floor: float) -> torch.Tensor:
        innovations = returns - float(grid.dt) * reference_drift
        realized = innovations[:, left:right, :].pow(2).sum(dim=1)
        predicted = reference_variance[:, left:right, :].sum(dim=1)
        return torch.log((realized + floor) / (predicted + floor))

    prefix_residual = log_ratio(split - prefix, split, prefix_floor)
    future_residual = log_ratio(split, split + future, future_floor)
    features = [prefix_residual]
    if bool(include_drawdown):
        prefix_paths = paths[:, : split + 1, :]
        running_max = torch.cummax(prefix_paths, dim=1).values
        current_drawdown = running_max[:, -1, :] - prefix_paths[:, -1, :]
        predicted_prefix_variance = reference_variance[:, :split, :].sum(dim=1)
        features.append(
            current_drawdown
            / torch.sqrt(predicted_prefix_variance + drawdown_floor)
        )
    features.append(future_residual)
    return torch.cat(features, dim=1)


def median_heuristic_bandwidths(
    standardized_features: torch.Tensor,
    *,
    multipliers: Sequence[float] = (0.5, 1.0, 2.0),
    maximum_paths: int = 1024,
) -> tuple[float, ...]:
    if standardized_features.ndim != 2 or int(standardized_features.shape[0]) < 2:
        raise ValueError("bandwidth fitting requires at least two feature rows")
    factors = tuple(_positive_finite("bandwidth multiplier", value) for value in multipliers)
    sample = standardized_features.detach()[: int(maximum_paths)].double()
    distances = torch.pdist(sample, p=2).pow(2)
    positive = distances[distances > torch.finfo(distances.dtype).eps]
    if int(positive.numel()) == 0:
        base = 1.0
    else:
        median = float(torch.median(positive).item())
        base = math.sqrt(median / (2.0 * float(sample.shape[1])))
    return tuple(max(torch.finfo(torch.float64).eps, base * value) for value in factors)


@dataclass(frozen=True)
class FixedReferenceResidualConditionalVolatilityMMD:
    reference_kernel: object
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    split_index: int
    prefix_window: int
    future_horizon: int
    prefix_variance_floor: float
    future_variance_floor: float
    drawdown_variance_floor: float
    include_drawdown: bool
    bandwidths: tuple[float, ...]
    name: str = "reference_residual_conditional_volatility"

    def __post_init__(self) -> None:
        if not isinstance(self.target_mean, torch.Tensor) or not isinstance(
            self.target_scale, torch.Tensor
        ):
            raise ValueError("target statistics must be tensors")
        if self.target_mean.ndim != 2 or int(self.target_mean.shape[0]) != 1:
            raise ValueError("target_mean must have shape (1, feature_dim)")
        if tuple(self.target_scale.shape) != tuple(self.target_mean.shape):
            raise ValueError("target_scale must match target_mean")
        expected_dim = 3 if bool(self.include_drawdown) else 2
        if int(self.target_mean.shape[1]) != expected_dim:
            raise ValueError("target statistics do not match the selected feature set")
        if not bool(torch.isfinite(self.target_mean).all().item()) or not bool(
            torch.isfinite(self.target_scale).all().item()
        ):
            raise ValueError("target statistics must be finite")
        if bool(torch.any(self.target_scale <= 0.0).item()):
            raise ValueError("target_scale must be positive")
        bandwidths = tuple(_positive_finite("bandwidth", value) for value in self.bandwidths)
        if not bandwidths:
            raise ValueError("bandwidths cannot be empty")
        for name in (
            "prefix_variance_floor",
            "future_variance_floor",
            "drawdown_variance_floor",
        ):
            object.__setattr__(self, name, _positive_finite(name, getattr(self, name)))
        object.__setattr__(self, "target_mean", self.target_mean.detach().cpu().clone())
        object.__setattr__(self, "target_scale", self.target_scale.detach().cpu().clone())
        object.__setattr__(self, "bandwidths", bandwidths)
        object.__setattr__(self, "include_drawdown", bool(self.include_drawdown))

    def raw_features(
        self,
        paths: torch.Tensor,
        *,
        grid: DiscreteTimeGrid,
        detach_reference_outputs: bool = False,
    ) -> torch.Tensor:
        return reference_residual_volatility_features(
            paths,
            reference_kernel=self.reference_kernel,
            grid=grid,
            split_index=int(self.split_index),
            prefix_window=int(self.prefix_window),
            future_horizon=int(self.future_horizon),
            prefix_variance_floor=float(self.prefix_variance_floor),
            future_variance_floor=float(self.future_variance_floor),
            drawdown_variance_floor=float(self.drawdown_variance_floor),
            include_drawdown=bool(self.include_drawdown),
            detach_reference_outputs=bool(detach_reference_outputs),
        )

    def features(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        raw = self.raw_features(paths, grid=grid)
        mean = self.target_mean.to(device=paths.device, dtype=paths.dtype)
        scale = self.target_scale.to(device=paths.device, dtype=paths.dtype)
        return (raw - mean) / scale

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        if int(generated_paths.shape[0]) < 2 or int(target_paths.shape[0]) < 2:
            raise ValueError("residual conditional MMD requires at least two paths per law")
        generated = self.features(generated_paths, grid=grid)
        target = self.features(
            target_paths.detach().to(
                device=generated_paths.device, dtype=generated_paths.dtype
            ),
            grid=grid,
        ).detach()
        value = rbf_mmd2(generated, target, bandwidths=self.bandwidths)
        return PathFunctionalDiscrepancyResult(
            value=value,
            metrics={
                f"{self.name}_value": float(value.detach().item()),
                f"{self.name}_feature_dim": float(generated.shape[1]),
                f"{self.name}_drawdown_augmented": float(self.include_drawdown),
            },
        )

    def lions_potential(
        self,
        *,
        query_paths: torch.Tensor,
        population_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> torch.Tensor:
        query = self.features(query_paths, grid=grid)
        population = self.features(
            population_paths.detach().to(
                device=query_paths.device, dtype=query_paths.dtype
            ),
            grid=grid,
        ).detach()
        target = self.features(
            target_paths.detach().to(
                device=query_paths.device, dtype=query_paths.dtype
            ),
            grid=grid,
        ).detach()
        return rbf_mmd_lions_potential(
            query,
            population,
            target,
            bandwidths=self.bandwidths,
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "split_index": int(self.split_index),
            "prefix_window": int(self.prefix_window),
            "future_horizon": int(self.future_horizon),
            "include_drawdown": bool(self.include_drawdown),
            "prefix_variance_floor": float(self.prefix_variance_floor),
            "future_variance_floor": float(self.future_variance_floor),
            "drawdown_variance_floor": float(self.drawdown_variance_floor),
            "target_mean": self.target_mean.tolist(),
            "target_scale": self.target_scale.tolist(),
            "bandwidths": list(self.bandwidths),
            "reference_parameters_frozen": True,
            "generated_reference_path_derivatives_included": True,
            "reference_drift_removed_from_innovations": True,
        }


def fit_fixed_reference_residual_conditional_volatility_mmd(
    target_paths: torch.Tensor,
    *,
    reference_kernel: object,
    grid: DiscreteTimeGrid,
    split_index: int,
    prefix_window: int,
    future_horizon: int,
    include_drawdown: bool,
    floor_fraction: float = 1e-3,
    bandwidth_multipliers: Sequence[float] = (0.5, 1.0, 2.0),
) -> FixedReferenceResidualConditionalVolatilityMMD:
    fraction = _positive_finite("floor_fraction", floor_fraction)
    _, reference_variance = _reference_moments(
        target_paths,
        reference_kernel=reference_kernel,
        grid=grid,
        detach_reference_outputs=True,
    )
    split = int(split_index)
    prefix = int(prefix_window)
    future = int(future_horizon)
    prefix_integrated = reference_variance[:, split - prefix : split].sum(dim=1)
    future_integrated = reference_variance[:, split : split + future].sum(dim=1)
    drawdown_integrated = reference_variance[:, :split].sum(dim=1)
    floors = tuple(
        fraction * max(float(torch.median(value).item()), torch.finfo(torch.float64).eps)
        for value in (prefix_integrated, future_integrated, drawdown_integrated)
    )
    raw = reference_residual_volatility_features(
        target_paths,
        reference_kernel=reference_kernel,
        grid=grid,
        split_index=split,
        prefix_window=prefix,
        future_horizon=future,
        prefix_variance_floor=floors[0],
        future_variance_floor=floors[1],
        drawdown_variance_floor=floors[2],
        include_drawdown=bool(include_drawdown),
        detach_reference_outputs=False,
    )
    mean = raw.mean(dim=0, keepdim=True).detach()
    scale = raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6).detach()
    standardized = (raw.detach() - mean) / scale
    bandwidths = median_heuristic_bandwidths(
        standardized,
        multipliers=bandwidth_multipliers,
    )
    return FixedReferenceResidualConditionalVolatilityMMD(
        reference_kernel=reference_kernel,
        target_mean=mean,
        target_scale=scale,
        split_index=split,
        prefix_window=prefix,
        future_horizon=future,
        prefix_variance_floor=floors[0],
        future_variance_floor=floors[1],
        drawdown_variance_floor=floors[2],
        include_drawdown=bool(include_drawdown),
        bandwidths=bandwidths,
    )


__all__ = [
    "FixedReferenceResidualConditionalVolatilityMMD",
    "fit_fixed_reference_residual_conditional_volatility_mmd",
    "median_heuristic_bandwidths",
    "reference_residual_volatility_features",
]
