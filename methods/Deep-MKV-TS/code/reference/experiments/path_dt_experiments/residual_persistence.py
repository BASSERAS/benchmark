from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch
import torch.nn as nn

from deep_mkv_gen_path_dt import DiscreteMPModel, DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.discrepancies import (
    CompositePathFunctionalDiscrepancy,
    PathFunctionalDiscrepancyResult,
    WeightedDiscrepancy,
)
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid
from deep_mkv_gen_path_dt.networks import AdjointMoments
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals
from path_dt_experiments.metrics import path_returns, stylized_fact_metrics
from path_dt_experiments.volatility_diagnostics import volatility_path_statistics


def _require_path_tensor(name: str, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> None:
    if not isinstance(paths, torch.Tensor) or paths.ndim != 3:
        raise ValueError(f"{name} must have shape (B, N + 1, d)")
    if int(paths.shape[0]) < 2:
        raise ValueError(f"{name} must contain at least two paths")
    if int(paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError(f"{name} time dimension must align with grid")
    if int(paths.shape[2]) < 1:
        raise ValueError(f"{name} must contain at least one state coordinate")


def _volatility_pair(
    paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    window: int,
    split_index: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_path_tensor("paths", paths, grid=grid)
    if int(window) < 1:
        raise ValueError("window must be >= 1")
    if int(split_index) < int(window) or int(grid.num_steps) - int(split_index) < int(window):
        raise ValueError("window must fit before and after split_index")
    returns = (paths[:, 1:, :] - paths[:, :-1, :]) / math.sqrt(float(grid.dt))
    eps_squared = float(eps) ** 2
    prefix = torch.sqrt(
        returns[:, int(split_index) - int(window) : int(split_index), :]
        .pow(2)
        .mean(dim=1)
        + eps_squared
    )
    future = torch.sqrt(
        returns[:, int(split_index) : int(split_index) + int(window), :]
        .pow(2)
        .mean(dim=1)
        + eps_squared
    )
    return prefix, future


def _centered_correlation(
    prefix: torch.Tensor,
    future: torch.Tensor,
    *,
    prefix_scale_floor: torch.Tensor,
    future_scale_floor: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix_centered = prefix - prefix.mean(dim=0, keepdim=True)
    future_centered = future - future.mean(dim=0, keepdim=True)
    eps_squared = float(eps) ** 2
    prefix_scale = torch.sqrt(prefix_centered.pow(2).mean(dim=0) + eps_squared)
    future_scale = torch.sqrt(future_centered.pow(2).mean(dim=0) + eps_squared)
    stabilized_prefix = torch.maximum(prefix_scale, prefix_scale_floor)
    stabilized_future = torch.maximum(future_scale, future_scale_floor)
    correlation = (prefix_centered * future_centered).mean(dim=0) / (
        stabilized_prefix * stabilized_future
    )
    return correlation, prefix_scale, future_scale


@dataclass(frozen=True)
class FixedTargetVolatilityCorrelationDiscrepancy:
    """Single-window correlation discrepancy with full-bank target statistics.

    ``target_paths`` in ``__call__`` is intentionally ignored. The target
    correlation and stabilization scales are fitted once from the complete
    training bank, removing target-minibatch noise without introducing a
    parametric or latent-volatility reference model.
    """

    window: int
    split_index: int
    target_correlation: torch.Tensor
    prefix_scale_floor: torch.Tensor
    future_scale_floor: torch.Tensor
    eps: float = 1e-6
    name: str = "fixed_target_prefix_future_volatility_correlation"

    def __post_init__(self) -> None:
        window = int(self.window)
        split_index = int(self.split_index)
        eps = float(self.eps)
        if window < 1:
            raise ValueError("window must be >= 1")
        if split_index < window:
            raise ValueError("split_index must be >= window")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        tensors = (
            self.target_correlation,
            self.prefix_scale_floor,
            self.future_scale_floor,
        )
        if any(not isinstance(value, torch.Tensor) or value.ndim != 1 for value in tensors):
            raise ValueError("fixed target statistics must be one-dimensional tensors")
        if len({int(value.numel()) for value in tensors}) != 1:
            raise ValueError("fixed target statistics must have matching dimensions")
        if any(int(value.numel()) < 1 for value in tensors):
            raise ValueError("fixed target statistics must be non-empty")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("fixed target statistics must be finite")
        if bool((self.prefix_scale_floor <= 0.0).any()) or bool(
            (self.future_scale_floor <= 0.0).any()
        ):
            raise ValueError("scale floors must be positive")
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "split_index", split_index)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "target_correlation", self.target_correlation.detach().cpu().clone())
        object.__setattr__(self, "prefix_scale_floor", self.prefix_scale_floor.detach().cpu().clone())
        object.__setattr__(self, "future_scale_floor", self.future_scale_floor.detach().cpu().clone())

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        del target_paths
        prefix, future = _volatility_pair(
            generated_paths,
            grid=grid,
            window=int(self.window),
            split_index=int(self.split_index),
            eps=float(self.eps),
        )
        target_correlation = self.target_correlation.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        prefix_floor = self.prefix_scale_floor.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        future_floor = self.future_scale_floor.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        generated_correlation, prefix_scale, future_scale = _centered_correlation(
            prefix,
            future,
            prefix_scale_floor=prefix_floor,
            future_scale_floor=future_floor,
            eps=float(self.eps),
        )
        difference = generated_correlation - target_correlation
        value = difference.pow(2).mean()
        metrics: dict[str, float] = {
            f"{self.name}_value": float(value.detach().item()),
            f"{self.name}_window": float(self.window),
            f"{self.name}_split_index": float(self.split_index),
            f"{self.name}_fixed_target": 1.0,
            f"{self.name}_correlation_rmse": float(
                torch.sqrt(difference.detach().pow(2).mean()).item()
            ),
            f"{self.name}_scale_floor_active_fraction": float(
                torch.cat(
                    [
                        (prefix_scale < prefix_floor).reshape(-1),
                        (future_scale < future_floor).reshape(-1),
                    ]
                )
                .to(dtype=generated_paths.dtype)
                .mean()
                .detach()
                .item()
            ),
        }
        for asset in range(int(generated_correlation.numel())):
            metrics[f"{self.name}_generated_asset{asset}"] = float(
                generated_correlation[asset].detach().item()
            )
            metrics[f"{self.name}_target_asset{asset}"] = float(
                target_correlation[asset].detach().item()
            )
            metrics[f"{self.name}_error_asset{asset}"] = float(
                difference[asset].detach().item()
            )
        return PathFunctionalDiscrepancyResult(value=value, metrics=metrics)


def fit_fixed_target_volatility_correlation(
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    window: int,
    split_index: int | None = None,
    scale_floor_fraction: float = 0.25,
    eps: float = 1e-6,
) -> FixedTargetVolatilityCorrelationDiscrepancy:
    _require_path_tensor("target_paths", target_paths, grid=grid)
    if not math.isfinite(float(scale_floor_fraction)) or not 0.0 < float(
        scale_floor_fraction
    ) <= 1.0:
        raise ValueError("scale_floor_fraction must be in (0, 1]")
    split = int(grid.num_steps) // 2 if split_index is None else int(split_index)
    prefix, future = _volatility_pair(
        target_paths,
        grid=grid,
        window=int(window),
        split_index=split,
        eps=float(eps),
    )
    prefix_scale = prefix.std(dim=0, unbiased=False).clamp_min(float(eps))
    future_scale = future.std(dim=0, unbiased=False).clamp_min(float(eps))
    target_correlation, _, _ = _centered_correlation(
        prefix,
        future,
        prefix_scale_floor=float(scale_floor_fraction) * prefix_scale,
        future_scale_floor=float(scale_floor_fraction) * future_scale,
        eps=float(eps),
    )
    return FixedTargetVolatilityCorrelationDiscrepancy(
        window=int(window),
        split_index=split,
        target_correlation=target_correlation,
        prefix_scale_floor=float(scale_floor_fraction) * prefix_scale,
        future_scale_floor=float(scale_floor_fraction) * future_scale,
        eps=float(eps),
    )


def _local_realized_volatility(
    paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    window: int,
    endpoint: int,
    eps: float,
) -> torch.Tensor:
    _require_path_tensor("paths", paths, grid=grid)
    if int(window) < 1:
        raise ValueError("window must be >= 1")
    if int(endpoint) < int(window) or int(endpoint) > int(grid.num_steps):
        raise ValueError("endpoint must satisfy window <= endpoint <= num_steps")
    returns = (paths[:, 1:, :] - paths[:, :-1, :]) / math.sqrt(float(grid.dt))
    return torch.sqrt(
        returns[:, int(endpoint) - int(window) : int(endpoint), :]
        .pow(2)
        .mean(dim=1)
        + float(eps) ** 2
    )


@dataclass(frozen=True)
class FixedTargetTimeMarginalVolatilityDiscrepancy:
    """Full-bank Wasserstein discrepancies for selected volatility marginals.

    Every ``(window, endpoint)`` pair defines a one-time marginal of local RMS
    realized volatility.  Target means, scales, IQRs, and empirical quantile
    functions are fitted once from the complete training bank.  At training
    time the generated empirical quantiles are compared with interpolated
    target quantiles, independently for every asset and date.  This gives a
    deterministic, dimensionless one-dimensional Wasserstein-2 anchor without
    introducing a latent-volatility or parametric reference model.
    """

    window_endpoint_pairs: Sequence[tuple[int, int]]
    target_mean: torch.Tensor
    target_std: torch.Tensor
    target_iqr: torch.Tensor
    target_sorted_standardized: torch.Tensor
    eps: float = 1e-6
    value_transform: str = "identity"
    target_raw_mean: torch.Tensor | None = None
    target_raw_std: torch.Tensor | None = None
    target_raw_iqr: torch.Tensor | None = None
    name: str = "fixed_target_time_marginal_volatility"

    def __post_init__(self) -> None:
        pairs = tuple((int(window), int(endpoint)) for window, endpoint in self.window_endpoint_pairs)
        if len(pairs) == 0:
            raise ValueError("window_endpoint_pairs must be non-empty")
        if len(set(pairs)) != len(pairs):
            raise ValueError("window_endpoint_pairs must not contain duplicates")
        if any(window < 1 or endpoint < window for window, endpoint in pairs):
            raise ValueError("every pair must satisfy 1 <= window <= endpoint")
        eps = float(self.eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        value_transform = str(self.value_transform).strip().lower()
        if value_transform not in {"identity", "log"}:
            raise ValueError("value_transform must be identity or log")
        if self.target_mean.ndim != 2:
            raise ValueError("target_mean must have shape (K, d)")
        expected_shape = tuple(self.target_mean.shape)
        for name, value in (
            ("target_std", self.target_std),
            ("target_iqr", self.target_iqr),
        ):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                raise ValueError(f"{name} must have shape (K, d)")
        sorted_target = self.target_sorted_standardized
        if not isinstance(sorted_target, torch.Tensor) or sorted_target.ndim != 3:
            raise ValueError("target_sorted_standardized must have shape (K, B, d)")
        if int(sorted_target.shape[0]) != len(pairs) or (
            int(sorted_target.shape[2]) != int(self.target_mean.shape[1])
        ):
            raise ValueError("fixed target tensors must have matching pair and asset dimensions")
        if int(sorted_target.shape[1]) < 2:
            raise ValueError("fixed target marginals require at least two target paths")
        tensors = (self.target_mean, self.target_std, self.target_iqr, sorted_target)
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("fixed target marginal statistics must be finite")
        if bool((self.target_std <= 0.0).any()) or bool((self.target_iqr < 0.0).any()):
            raise ValueError("target marginal scales must be valid")
        if bool((sorted_target[:, 1:, :] < sorted_target[:, :-1, :]).any()):
            raise ValueError("target standardized values must be sorted")
        raw_tensors = (
            self.target_raw_mean,
            self.target_raw_std,
            self.target_raw_iqr,
        )
        if any(value is not None for value in raw_tensors):
            if any(value is None for value in raw_tensors):
                raise ValueError("all target_raw statistics must be supplied together")
            for raw_name, raw_value in zip(
                ("target_raw_mean", "target_raw_std", "target_raw_iqr"),
                raw_tensors,
            ):
                assert raw_value is not None
                if tuple(raw_value.shape) != expected_shape:
                    raise ValueError(f"{raw_name} must have shape (K, d)")
                if not bool(torch.isfinite(raw_value).all()):
                    raise ValueError(f"{raw_name} must be finite")
            assert self.target_raw_mean is not None
            assert self.target_raw_std is not None
            assert self.target_raw_iqr is not None
            if bool((self.target_raw_mean <= 0.0).any()) or bool(
                (self.target_raw_std <= 0.0).any()
            ) or bool((self.target_raw_iqr < 0.0).any()):
                raise ValueError("raw target volatility statistics must be valid")
        object.__setattr__(self, "window_endpoint_pairs", pairs)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "value_transform", value_transform)
        for name in (
            "target_mean",
            "target_std",
            "target_iqr",
            "target_sorted_standardized",
            "target_raw_mean",
            "target_raw_std",
            "target_raw_iqr",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, value.detach().cpu().clone())

    @staticmethod
    def _interpolated_quantiles(sorted_values: torch.Tensor, *, count: int) -> torch.Tensor:
        target_count = int(sorted_values.shape[0])
        if int(count) < 2:
            raise ValueError("generated marginals require at least two paths")
        if int(count) == target_count:
            return sorted_values
        coordinate = torch.linspace(
            0.0,
            float(target_count - 1),
            int(count),
            device=sorted_values.device,
            dtype=sorted_values.dtype,
        )
        lower = torch.floor(coordinate).to(dtype=torch.long)
        upper = torch.ceil(coordinate).to(dtype=torch.long)
        upper_weight = (coordinate - lower.to(dtype=coordinate.dtype)).unsqueeze(-1)
        return (
            (1.0 - upper_weight) * sorted_values.index_select(0, lower)
            + upper_weight * sorted_values.index_select(0, upper)
        )

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        del target_paths
        _require_path_tensor("generated_paths", generated_paths, grid=grid)
        target_mean = self.target_mean.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        target_std = self.target_std.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        target_iqr = self.target_iqr.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        sorted_target = self.target_sorted_standardized.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        if int(generated_paths.shape[2]) != int(target_mean.shape[1]):
            raise ValueError("generated and fixed-target asset dimensions must match")

        marginal_values: list[torch.Tensor] = []
        metrics: dict[str, float] = {
            f"{self.name}_fixed_target": 1.0,
            f"{self.name}_marginal_count": float(
                len(tuple(self.window_endpoint_pairs)) * int(target_mean.shape[1])
            ),
            f"{self.name}_target_path_count": float(sorted_target.shape[1]),
            f"{self.name}_log_transform": float(self.value_transform == "log"),
        }
        quantile_levels = torch.tensor(
            [0.25, 0.75],
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        for position, (window, endpoint) in enumerate(tuple(self.window_endpoint_pairs)):
            generated_volatility = _local_realized_volatility(
                generated_paths,
                grid=grid,
                window=int(window),
                endpoint=int(endpoint),
                eps=float(self.eps),
            )
            generated_comparison = (
                torch.log(generated_volatility.clamp_min(float(self.eps)))
                if self.value_transform == "log"
                else generated_volatility
            )
            generated_standardized = (
                generated_comparison - target_mean[position].unsqueeze(0)
            ) / target_std[position].unsqueeze(0)
            generated_sorted = torch.sort(generated_standardized, dim=0).values
            target_quantiles = self._interpolated_quantiles(
                sorted_target[position],
                count=int(generated_sorted.shape[0]),
            )
            wasserstein_by_asset = (generated_sorted - target_quantiles).pow(2).mean(dim=0)
            marginal_values.extend(wasserstein_by_asset.unbind())

            generated_mean = generated_comparison.mean(dim=0)
            generated_std = generated_comparison.std(dim=0, unbiased=False)
            generated_quartiles = torch.quantile(
                generated_comparison,
                quantile_levels,
                dim=0,
            )
            generated_iqr = generated_quartiles[1] - generated_quartiles[0]
            generated_raw_mean = generated_volatility.mean(dim=0)
            generated_raw_std = generated_volatility.std(dim=0, unbiased=False)
            generated_raw_quartiles = torch.quantile(
                generated_volatility,
                quantile_levels,
                dim=0,
            )
            generated_raw_iqr = generated_raw_quartiles[1] - generated_raw_quartiles[0]
            label = f"w{int(window)}_e{int(endpoint)}"
            for asset in range(int(generated_paths.shape[2])):
                suffix = f"{label}_asset{asset}"
                metrics[f"{self.name}_{suffix}_wasserstein2"] = float(
                    wasserstein_by_asset[asset].detach().item()
                )
                for statistic, generated_value, target_value in (
                    ("mean", generated_mean[asset], target_mean[position, asset]),
                    ("std", generated_std[asset], target_std[position, asset]),
                    ("iqr", generated_iqr[asset], target_iqr[position, asset]),
                ):
                    generated_scalar = float(generated_value.detach().item())
                    target_scalar = float(target_value.detach().item())
                    metrics[f"{self.name}_{suffix}_generated_{statistic}"] = generated_scalar
                    metrics[f"{self.name}_{suffix}_target_{statistic}"] = target_scalar
                    metrics[f"{self.name}_{suffix}_{statistic}_ratio"] = generated_scalar / max(
                        abs(target_scalar),
                        torch.finfo(torch.float64).eps,
                    )
                if self.target_raw_mean is not None:
                    assert self.target_raw_std is not None
                    assert self.target_raw_iqr is not None
                    for statistic, generated_value, target_value in (
                        ("raw_mean", generated_raw_mean[asset], self.target_raw_mean[position, asset]),
                        ("raw_std", generated_raw_std[asset], self.target_raw_std[position, asset]),
                        ("raw_iqr", generated_raw_iqr[asset], self.target_raw_iqr[position, asset]),
                    ):
                        generated_scalar = float(generated_value.detach().item())
                        target_scalar = float(target_value.detach().item())
                        metrics[f"{self.name}_{suffix}_generated_{statistic}"] = generated_scalar
                        metrics[f"{self.name}_{suffix}_target_{statistic}"] = target_scalar
                        metrics[f"{self.name}_{suffix}_{statistic}_ratio"] = (
                            generated_scalar
                            / max(abs(target_scalar), torch.finfo(torch.float64).eps)
                        )
        value = torch.stack(marginal_values).mean()
        metrics[f"{self.name}_value"] = float(value.detach().item())
        return PathFunctionalDiscrepancyResult(value=value, metrics=metrics)

    def lions_potential(
        self,
        *,
        query_paths: torch.Tensor,
        population_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> torch.Tensor:
        """Frozen-law potential for the one-dimensional W2 marginals.

        For a scalar marginal, the spatial derivative of squared
        Wasserstein-2 is ``2 * (z - T_mu_to_nu(z))``.  The empirical
        transport rank is computed from the detached unconditional population
        law and is therefore held fixed while differentiating the query path.
        The returned query average includes the averaging over dates and
        assets used by :meth:`__call__`.
        """

        del target_paths
        _require_path_tensor("query_paths", query_paths, grid=grid)
        _require_path_tensor("population_paths", population_paths, grid=grid)
        if int(query_paths.shape[2]) != int(population_paths.shape[2]):
            raise ValueError("query and population asset dimensions must match")

        target_mean = self.target_mean.to(
            device=query_paths.device,
            dtype=query_paths.dtype,
        )
        target_std = self.target_std.to(
            device=query_paths.device,
            dtype=query_paths.dtype,
        )
        sorted_target = self.target_sorted_standardized.to(
            device=query_paths.device,
            dtype=query_paths.dtype,
        )
        pair_count = len(tuple(self.window_endpoint_pairs))
        state_dim = int(query_paths.shape[2])
        population_count = int(population_paths.shape[0])
        potential = query_paths.new_tensor(0.0)

        for position, (window, endpoint) in enumerate(
            tuple(self.window_endpoint_pairs)
        ):
            query_volatility = _local_realized_volatility(
                query_paths,
                grid=grid,
                window=int(window),
                endpoint=int(endpoint),
                eps=float(self.eps),
            )
            population_volatility = _local_realized_volatility(
                population_paths,
                grid=grid,
                window=int(window),
                endpoint=int(endpoint),
                eps=float(self.eps),
            ).detach()
            if self.value_transform == "log":
                query_comparison = torch.log(
                    query_volatility.clamp_min(float(self.eps))
                )
                population_comparison = torch.log(
                    population_volatility.clamp_min(float(self.eps))
                )
            else:
                query_comparison = query_volatility
                population_comparison = population_volatility

            query_standardized = (
                query_comparison - target_mean[position].unsqueeze(0)
            ) / target_std[position].unsqueeze(0)
            population_standardized = (
                population_comparison - target_mean[position].unsqueeze(0)
            ) / target_std[position].unsqueeze(0)
            target_at_population_ranks = self._interpolated_quantiles(
                sorted_target[position],
                count=population_count,
            )
            coefficients = torch.empty_like(query_comparison)
            for asset in range(state_dim):
                population_sorted = torch.sort(
                    population_standardized[:, asset]
                ).values
                query_values = query_standardized[:, asset]
                ranks = torch.searchsorted(
                    population_sorted,
                    query_values.detach().contiguous(),
                    right=False,
                ).clamp(max=population_count - 1)
                transported_target = target_at_population_ranks[
                    :, asset
                ].index_select(0, ranks)
                coefficients[:, asset] = (
                    2.0
                    * (query_values.detach() - transported_target)
                    / (
                        float(pair_count * state_dim)
                        * target_std[position, asset]
                    )
                )
            potential = potential + (
                coefficients.detach() * query_comparison
            ).sum(dim=1).mean()
        return potential

    def select_pairs(
        self,
        pairs: Sequence[tuple[int, int]],
    ) -> FixedTargetTimeMarginalVolatilityDiscrepancy:
        """Return a view-like fixed target containing only selected dates."""

        selected = tuple((int(window), int(endpoint)) for window, endpoint in pairs)
        if len(selected) == 0:
            raise ValueError("pairs must be non-empty")
        available = tuple(self.window_endpoint_pairs)
        if any(pair not in available for pair in selected):
            raise ValueError("every selected pair must exist in the fitted target")
        positions = torch.tensor(
            [available.index(pair) for pair in selected],
            dtype=torch.long,
        )
        return FixedTargetTimeMarginalVolatilityDiscrepancy(
            window_endpoint_pairs=selected,
            target_mean=self.target_mean.index_select(0, positions),
            target_std=self.target_std.index_select(0, positions),
            target_iqr=self.target_iqr.index_select(0, positions),
            target_sorted_standardized=self.target_sorted_standardized.index_select(
                0,
                positions,
            ),
            eps=float(self.eps),
            value_transform=str(self.value_transform),
            target_raw_mean=(
                None
                if self.target_raw_mean is None
                else self.target_raw_mean.index_select(0, positions)
            ),
            target_raw_std=(
                None
                if self.target_raw_std is None
                else self.target_raw_std.index_select(0, positions)
            ),
            target_raw_iqr=(
                None
                if self.target_raw_iqr is None
                else self.target_raw_iqr.index_select(0, positions)
            ),
            name=str(self.name),
        )


def fit_fixed_target_time_marginal_volatility(
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    window_endpoint_pairs: Sequence[tuple[int, int]],
    eps: float = 1e-6,
    value_transform: str = "identity",
) -> FixedTargetTimeMarginalVolatilityDiscrepancy:
    _require_path_tensor("target_paths", target_paths, grid=grid)
    pairs = tuple((int(window), int(endpoint)) for window, endpoint in window_endpoint_pairs)
    if len(pairs) == 0:
        raise ValueError("window_endpoint_pairs must be non-empty")
    volatility = torch.stack(
        [
            _local_realized_volatility(
                target_paths,
                grid=grid,
                window=window,
                endpoint=endpoint,
                eps=float(eps),
            )
            for window, endpoint in pairs
        ],
        dim=0,
    )
    transform = str(value_transform).strip().lower()
    if transform not in {"identity", "log"}:
        raise ValueError("value_transform must be identity or log")
    comparison = (
        torch.log(volatility.clamp_min(float(eps)))
        if transform == "log"
        else volatility
    )
    target_mean = comparison.mean(dim=1)
    target_std = comparison.std(dim=1, unbiased=False).clamp_min(float(eps))
    standardized = (comparison - target_mean.unsqueeze(1)) / target_std.unsqueeze(1)
    sorted_standardized = torch.sort(standardized, dim=1).values
    quartile_levels = torch.tensor(
        [0.25, 0.75],
        device=volatility.device,
        dtype=volatility.dtype,
    )
    quartiles = torch.quantile(comparison, quartile_levels, dim=1)
    target_iqr = quartiles[1] - quartiles[0]
    raw_quartiles = torch.quantile(volatility, quartile_levels, dim=1)
    return FixedTargetTimeMarginalVolatilityDiscrepancy(
        window_endpoint_pairs=pairs,
        target_mean=target_mean,
        target_std=target_std,
        target_iqr=target_iqr,
        target_sorted_standardized=sorted_standardized,
        eps=float(eps),
        value_transform=transform,
        target_raw_mean=(volatility.mean(dim=1) if transform == "log" else None),
        target_raw_std=(
            volatility.std(dim=1, unbiased=False).clamp_min(float(eps))
            if transform == "log"
            else None
        ),
        target_raw_iqr=(raw_quartiles[1] - raw_quartiles[0] if transform == "log" else None),
        name=(
            "fixed_target_log_time_marginal_volatility"
            if transform == "log"
            else "fixed_target_time_marginal_volatility"
        ),
    )


class NoiseAdjointResidualNetwork(nn.Module):
    """Causal residual network for the noise-adjoint moment only."""

    def __init__(
        self,
        *,
        state_dim: int,
        noise_adjoint_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 1,
        zero_final: bool = True,
    ) -> None:
        super().__init__()
        if min(int(state_dim), int(noise_adjoint_dim), int(hidden_dim), int(num_layers)) < 1:
            raise ValueError("residual network dimensions must be >= 1")
        self.state_dim = int(state_dim)
        self.noise_adjoint_dim = int(noise_adjoint_dim)
        self.gru = nn.GRU(
            input_size=int(state_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(noise_adjoint_dim)),
        )
        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        if bool(zero_final):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def reset_output_layer(self) -> None:
        """Restore the final linear layer's ordinary nonzero initialization."""

        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        final.reset_parameters()

    def forward(self, x_path: torch.Tensor) -> torch.Tensor:
        if x_path.ndim != 3 or int(x_path.shape[1]) < 2:
            raise ValueError("x_path must have shape (B, N + 1, d)")
        if int(x_path.shape[2]) != int(self.state_dim):
            raise ValueError("x_path state dimension does not match residual network")
        output, _ = self.gru(x_path[:, :-1, :])
        return self.head(output)

    def forward_step(
        self,
        *,
        x_t: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_t.ndim != 2 or int(x_t.shape[1]) != int(self.state_dim):
            raise ValueError("x_t must have shape (B, state_dim)")
        output, next_hidden = self.gru(x_t.unsqueeze(1), hidden)
        return self.head(output[:, -1, :]), next_hidden


class CenteredNoiseAdjointResidualNetwork(nn.Module):
    """Exactly-zero residual with the full trainable-network tangent active.

    The frozen anchor is evaluated on the same evolving path as the trainable
    network.  Its parameters do not receive gradients, while its derivative
    with respect to the path remains active.  At initialization the two
    networks are identical, so their values and input-state Jacobians cancel;
    gradients with respect to every trainable parameter remain available.
    """

    def __init__(self, trainable_network: NoiseAdjointResidualNetwork) -> None:
        super().__init__()
        self.trainable_network = trainable_network
        self.anchor_network = copy.deepcopy(trainable_network)
        self.state_dim = int(trainable_network.state_dim)
        self.noise_adjoint_dim = int(trainable_network.noise_adjoint_dim)
        for parameter in self.anchor_network.parameters():
            parameter.requires_grad_(False)

    def forward(self, x_path: torch.Tensor) -> torch.Tensor:
        return self.trainable_network(x_path) - self.anchor_network(x_path)

    def forward_step(
        self,
        *,
        x_t: torch.Tensor,
        hidden: tuple[torch.Tensor | None, torch.Tensor | None] | None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        if hidden is None:
            trainable_hidden = None
            anchor_hidden = None
        else:
            trainable_hidden, anchor_hidden = hidden
        trainable_output, next_trainable_hidden = self.trainable_network.forward_step(
            x_t=x_t,
            hidden=trainable_hidden,
        )
        anchor_output, next_anchor_hidden = self.anchor_network.forward_step(
            x_t=x_t,
            hidden=anchor_hidden,
        )
        return (
            trainable_output - anchor_output,
            (next_trainable_hidden, next_anchor_hidden),
        )


class FrozenBaseResidualRNetwork(nn.Module):
    """Frozen base adjoint network plus an additive raw-R correction.

    The enclosing ``DiscreteMPModel`` keeps the base target preconditioner. The
    residual is trained in its own normalized units and converted into base
    normalized units before the model decodes both through the unchanged
    preconditioner and entropy-barrier control map.
    """

    def __init__(
        self,
        *,
        base_network: nn.Module,
        residual_network: nn.Module,
        base_noise_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        if not hasattr(residual_network, "state_dim") or not hasattr(
            residual_network,
            "noise_adjoint_dim",
        ):
            raise ValueError(
                "residual_network must expose state_dim and noise_adjoint_dim"
            )
        if base_noise_scale.ndim != 2:
            raise ValueError("base_noise_scale must have shape (N, noise_adjoint_dim)")
        if int(base_noise_scale.shape[1]) != int(residual_network.noise_adjoint_dim):
            raise ValueError("base scale output dimension must match residual network")
        self.base_network = base_network
        self.residual_network = residual_network
        self.state_dim = int(getattr(base_network, "state_dim"))
        self.adjoint_dim = int(getattr(base_network, "adjoint_dim"))
        self.noise_adjoint_dim = int(getattr(base_network, "noise_adjoint_dim"))
        if int(residual_network.state_dim) != int(self.state_dim):
            raise ValueError("base and residual state dimensions must match")
        if int(residual_network.noise_adjoint_dim) != int(self.noise_adjoint_dim):
            raise ValueError("base and residual noise-adjoint dimensions must match")
        for parameter in self.base_network.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("base_noise_scale", base_noise_scale.detach().clone())
        self.register_buffer("residual_noise_scale", torch.ones_like(base_noise_scale))

    def set_residual_noise_scale(self, scale: torch.Tensor) -> None:
        if tuple(scale.shape) != tuple(self.residual_noise_scale.shape):
            raise ValueError("residual scale shape must match the base time/output shape")
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0.0).any()):
            raise ValueError("residual scale must be finite and positive")
        self.residual_noise_scale.copy_(
            scale.detach().to(
                device=self.residual_noise_scale.device,
                dtype=self.residual_noise_scale.dtype,
            )
        )

    def forward(self, x_path: torch.Tensor) -> AdjointMoments:
        # The base parameters are frozen in ``__init__``, but its response to
        # ``x_path`` must remain differentiable.  A no-grad wrapper here would
        # suppress the feedback derivative through the evolving generated
        # state, so the residual would be trained against a different local
        # system from the one evaluated by perturbed rollouts.
        base = self.base_network(x_path)
        residual_normalized = self.residual_network(x_path)
        ratio = (self.residual_noise_scale / self.base_noise_scale).unsqueeze(0)
        return AdjointMoments(
            expected_adjoint_next=base.expected_adjoint_next,
            expected_adjoint_noise_next=(
                base.expected_adjoint_noise_next + ratio * residual_normalized
            ),
        )

    def forward_step(
        self,
        *,
        x_t: torch.Tensor,
        hidden: tuple[torch.Tensor | None, torch.Tensor | None, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, int]]:
        if hidden is None:
            base_hidden = None
            residual_hidden = None
            step_index = 0
        else:
            base_hidden, residual_hidden, step_index = hidden
        if not 0 <= int(step_index) < int(self.base_noise_scale.shape[0]):
            raise ValueError("residual rollout step exceeds the stored timewise scale")
        base_p, base_r, next_base_hidden = self.base_network.forward_step(
            x_t=x_t,
            hidden=base_hidden,
        )
        residual_r, next_residual_hidden = self.residual_network.forward_step(
            x_t=x_t,
            hidden=residual_hidden,
        )
        ratio = self.residual_noise_scale[int(step_index)] / self.base_noise_scale[int(step_index)]
        return (
            base_p,
            base_r + ratio.unsqueeze(0) * residual_r,
            (next_base_hidden, next_residual_hidden, int(step_index) + 1),
        )


@dataclass(frozen=True)
class ResidualRTrainingConfig:
    num_steps: int = 100
    batch_size: int = 2048
    calibration_batches: int = 8
    min_scale: float = 1e-3
    lr: float = 2e-3
    weight_decay: float = 1e-5
    eval_every: int = 10
    seed: int = 1234

    def __post_init__(self) -> None:
        if min(
            int(self.num_steps),
            int(self.batch_size),
            int(self.calibration_batches),
            int(self.eval_every),
        ) < 1:
            raise ValueError("residual training counts must be >= 1")
        for name, value in (
            ("min_scale", self.min_scale),
            ("lr", self.lr),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a positive finite float")
        if not math.isfinite(float(self.weight_decay)) or float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be a non-negative finite float")


@dataclass(frozen=True)
class ResidualRFitResult:
    history: list[dict[str, float]]
    evaluations: list[dict[str, object]]
    checkpoints: dict[int, dict[str, torch.Tensor]]
    residual_noise_scale: torch.Tensor


def _state_dict_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _parameter_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    return math.sqrt(sum(float(parameter.detach().pow(2).sum().item()) for parameter in parameters))


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    return math.sqrt(
        sum(
            float(parameter.grad.detach().pow(2).sum().item())
            for parameter in parameters
            if parameter.grad is not None
        )
    )


def _update_norm(
    parameters: Sequence[torch.nn.Parameter],
    before: Sequence[torch.Tensor],
) -> float:
    return math.sqrt(
        sum(
            float((parameter.detach() - previous).pow(2).sum().item())
            for parameter, previous in zip(parameters, before)
        )
    )


def _direction_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().double().reshape(-1)).item())


def _direction_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.detach().double().reshape(-1)
    right_flat = right.detach().double().reshape(-1)
    denominator = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return 0.0
    return float(torch.dot(left_flat, right_flat).div(denominator).item())


def residual_block_direction_diagnostics(
    *,
    model: DiscreteMPModel,
    generated_paths: torch.Tensor,
    noise: torch.Tensor,
    target_paths: torch.Tensor,
    source_training: DiscreteMPTrainingConfig,
    blocks: Sequence[WeightedDiscrepancy],
) -> dict[str, object]:
    """Measure block interaction in path-adjoint and noise-adjoint target space."""

    if len(tuple(blocks)) < 1:
        raise ValueError("blocks must be non-empty")
    if generated_paths.ndim != 3 or noise.ndim != 3:
        raise ValueError("generated_paths and noise must be three-dimensional")
    if int(generated_paths.shape[0]) != int(noise.shape[0]):
        raise ValueError("generated_paths and noise must have matching batch sizes")
    if int(generated_paths.shape[1]) != int(noise.shape[1]) + 1:
        raise ValueError("generated paths must have one more time point than noise")

    pathwise_directions: dict[str, torch.Tensor] = {}
    noise_directions: dict[str, torch.Tensor] = {}
    parameter_directions: dict[str, torch.Tensor] = {}
    block_payload: dict[str, object] = {}
    residual_network = getattr(model.network, "residual_network", None)
    residual_scale = getattr(model.network, "residual_noise_scale", None)
    trainable = (
        tuple(
            parameter
            for parameter in residual_network.parameters()
            if parameter.requires_grad
        )
        if isinstance(residual_network, nn.Module)
        else ()
    )
    has_parameter_diagnostics = (
        len(trainable) > 0
        and isinstance(residual_scale, torch.Tensor)
        and residual_scale.ndim == 2
    )
    target_used = target_paths[: max(2, min(int(target_paths.shape[0]), 16))]
    for block in tuple(blocks):
        single_block = CompositePathFunctionalDiscrepancy(blocks=(block,))
        pathwise_target, target_metrics = model._path_functional_adjoint_targets(
            generated_path=generated_paths,
            target_path=target_used,
            training=source_training,
            discrepancy=single_block,
            include_running_cost=False,
        )
        noise_target = model._adjoint_noise_target(
            pathwise_adjoint_target=pathwise_target,
            moments_noise_target_dim=int(model._default_noise_adjoint_dim(model.architecture)),
            noise=noise,
            control_variate=None,
        ).detach()
        pathwise_directions[str(block.name)] = pathwise_target.detach()
        noise_directions[str(block.name)] = noise_target
        payload: dict[str, object] = {
            "weight": float(block.weight),
            "pathwise_target_norm": _direction_norm(pathwise_target),
            "pathwise_target_rms": float(
                torch.sqrt(pathwise_target.detach().double().pow(2).mean()).item()
            ),
            "noise_adjoint_target_norm": _direction_norm(noise_target),
            "noise_adjoint_target_rms": float(
                torch.sqrt(noise_target.detach().double().pow(2).mean()).item()
            ),
            "target_metrics": target_metrics,
        }
        if has_parameter_diagnostics:
            normalized_prediction = residual_network(generated_paths.detach())
            normalized_target = noise_target / residual_scale.unsqueeze(0)
            block_loss = (normalized_prediction - normalized_target).pow(2).mean()
            gradients = torch.autograd.grad(
                block_loss,
                trainable,
                allow_unused=True,
            )
            flattened = torch.cat(
                [
                    (
                        torch.zeros_like(parameter).reshape(-1)
                        if gradient is None
                        else gradient.detach().reshape(-1)
                    )
                    for parameter, gradient in zip(trainable, gradients)
                ]
            )
            parameter_directions[str(block.name)] = flattened
            payload.update(
                {
                    "normalized_adjoint_mse": float(block_loss.detach().item()),
                    "parameter_gradient_norm": _direction_norm(flattened),
                }
            )
        block_payload[str(block.name)] = payload

    names = tuple(pathwise_directions)
    pairwise: dict[str, object] = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            pair_payload = {
                "pathwise_target_cosine": _direction_cosine(
                    pathwise_directions[left_name],
                    pathwise_directions[right_name],
                ),
                "noise_adjoint_target_cosine": _direction_cosine(
                    noise_directions[left_name],
                    noise_directions[right_name],
                ),
            }
            if has_parameter_diagnostics:
                pair_payload["parameter_gradient_cosine"] = _direction_cosine(
                    parameter_directions[left_name],
                    parameter_directions[right_name],
                )
            pairwise[f"{left_name}__{right_name}"] = pair_payload

    combined_pathwise = torch.stack(tuple(pathwise_directions.values()), dim=0).sum(dim=0)
    combined_noise = torch.stack(tuple(noise_directions.values()), dim=0).sum(dim=0)
    pathwise_norm_sum = sum(_direction_norm(value) for value in pathwise_directions.values())
    noise_norm_sum = sum(_direction_norm(value) for value in noise_directions.values())
    combined_payload = {
        "pathwise_target_norm": _direction_norm(combined_pathwise),
        "pathwise_cancellation_ratio": _direction_norm(combined_pathwise)
        / max(pathwise_norm_sum, torch.finfo(torch.float64).eps),
        "noise_adjoint_target_norm": _direction_norm(combined_noise),
        "noise_adjoint_cancellation_ratio": _direction_norm(combined_noise)
        / max(noise_norm_sum, torch.finfo(torch.float64).eps),
    }
    if has_parameter_diagnostics:
        combined_parameter = torch.stack(tuple(parameter_directions.values()), dim=0).sum(dim=0)
        parameter_norm_sum = sum(
            _direction_norm(value) for value in parameter_directions.values()
        )
        combined_payload.update(
            {
                "parameter_gradient_norm": _direction_norm(combined_parameter),
                "parameter_gradient_cancellation_ratio": _direction_norm(combined_parameter)
                / max(parameter_norm_sum, torch.finfo(torch.float64).eps),
            }
        )
    return {
        "blocks": block_payload,
        "pairwise": pairwise,
        "combined": combined_payload,
    }


def fit_residual_r_correction(
    *,
    model: DiscreteMPModel,
    network: FrozenBaseResidualRNetwork,
    target_paths: torch.Tensor,
    discrepancy,
    source_training: DiscreteMPTrainingConfig,
    config: ResidualRTrainingConfig,
    evaluation_callback: Callable[[int], dict[str, object]] | None = None,
) -> ResidualRFitResult:
    if model.network is not network:
        raise ValueError("model.network must be the supplied frozen-base residual network")
    target = target_paths.to(device=model.device, dtype=model.dtype)
    target_count, _ = model._validate_target_paths(target)
    cpu_generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    dummy_target = target[:2]

    squared_sum: torch.Tensor | None = None
    calibration_count = 0
    for batch_index in range(int(config.calibration_batches)):
        indices = model._sample_indices(
            size=target_count,
            count=int(config.batch_size),
            generator=cpu_generator,
        )
        x0 = target.index_select(0, indices.to(model.device))[:, 0, :]
        noise = sample_standard_normals(
            grid=model.grid,
            batch_size=int(config.batch_size),
            noise_dim=int(model.architecture.noise_dim),
            device=model.device,
            dtype=model.dtype,
            seed=derive_stream_seed(base_seed=int(config.seed), stream_offset=80_000 + batch_index),
        )
        with torch.no_grad():
            rollout = model.rollout(x0=x0, noise=noise)
        pathwise_target, _ = model._path_functional_adjoint_targets(
            generated_path=rollout.paths,
            target_path=dummy_target,
            training=source_training,
            discrepancy=discrepancy,
            include_running_cost=False,
        )
        r_target = model._adjoint_noise_target(
            pathwise_adjoint_target=pathwise_target,
            moments_noise_target_dim=int(network.noise_adjoint_dim),
            noise=noise,
            control_variate=None,
        ).detach()
        batch_squared_sum = r_target.to(dtype=torch.float64).pow(2).sum(dim=0)
        squared_sum = batch_squared_sum if squared_sum is None else squared_sum + batch_squared_sum
        calibration_count += int(r_target.shape[0])
    if squared_sum is None or calibration_count < 1:
        raise RuntimeError("residual target calibration produced no samples")
    residual_scale = torch.sqrt(squared_sum / float(calibration_count)).clamp_min(float(config.min_scale))
    residual_scale = residual_scale.to(device=model.device, dtype=model.dtype)
    network.set_residual_noise_scale(residual_scale)

    trainable = [
        parameter
        for parameter in network.residual_network.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    history: list[dict[str, float]] = []
    evaluations: list[dict[str, object]] = []
    checkpoints: dict[int, dict[str, torch.Tensor]] = {
        0: _state_dict_cpu(network.residual_network)
    }
    if evaluation_callback is not None:
        evaluations.append({"step": 0, "metrics": evaluation_callback(0)})

    for step_index in range(int(config.num_steps)):
        indices = model._sample_indices(
            size=target_count,
            count=int(config.batch_size),
            generator=cpu_generator,
        )
        x0 = target.index_select(0, indices.to(model.device))[:, 0, :]
        noise = sample_standard_normals(
            grid=model.grid,
            batch_size=int(config.batch_size),
            noise_dim=int(model.architecture.noise_dim),
            device=model.device,
            dtype=model.dtype,
            seed=derive_stream_seed(base_seed=int(config.seed), stream_offset=step_index + 1),
        )
        with torch.no_grad():
            rollout = model.rollout(x0=x0, noise=noise)
        pathwise_target, target_metrics = model._path_functional_adjoint_targets(
            generated_path=rollout.paths,
            target_path=dummy_target,
            training=source_training,
            discrepancy=discrepancy,
            include_running_cost=False,
        )
        r_target = model._adjoint_noise_target(
            pathwise_adjoint_target=pathwise_target,
            moments_noise_target_dim=int(network.noise_adjoint_dim),
            noise=noise,
            control_variate=None,
        ).detach()
        normalized_target = r_target / residual_scale.unsqueeze(0)
        normalized_prediction = network.residual_network(rollout.paths.detach())
        loss = (normalized_prediction - normalized_target).pow(2).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameter_norm = _parameter_norm(trainable)
        gradient_norm = _gradient_norm(trainable)
        before = [parameter.detach().clone() for parameter in trainable]
        optimizer.step()
        update_norm = _update_norm(trainable, before)
        target_rms = float(torch.sqrt(normalized_target.detach().pow(2).mean()).item())
        prediction_rms = float(torch.sqrt(normalized_prediction.detach().pow(2).mean()).item())
        row = {
            "step": float(step_index + 1),
            "loss": float(loss.detach().item()),
            "normalized_target_rms": target_rms,
            "normalized_prediction_rms": prediction_rms,
            "prediction_target_rms_ratio": prediction_rms
            / max(target_rms, torch.finfo(torch.float64).eps),
            "gradient_norm": float(gradient_norm),
            "parameter_norm_before_update": float(parameter_norm),
            "parameter_update_norm": float(update_norm),
            "update_to_parameter_norm": float(
                update_norm / max(parameter_norm, torch.finfo(torch.float64).eps)
            ),
            "gradient_nonfinite_fraction": float(
                sum(
                    int((~torch.isfinite(parameter.grad)).sum().item())
                    for parameter in trainable
                    if parameter.grad is not None
                )
                / max(
                    1,
                    sum(
                        int(parameter.grad.numel())
                        for parameter in trainable
                        if parameter.grad is not None
                    ),
                )
            ),
        }
        for name, value in target_metrics.items():
            row[f"target_{name}"] = float(value)
        history.append(row)

        step_number = int(step_index) + 1
        if evaluation_callback is not None and (
            step_number % int(config.eval_every) == 0
            or step_number == int(config.num_steps)
        ):
            checkpoints[step_number] = _state_dict_cpu(network.residual_network)
            evaluations.append({"step": step_number, "metrics": evaluation_callback(step_number)})

    return ResidualRFitResult(
        history=history,
        evaluations=evaluations,
        checkpoints=checkpoints,
        residual_noise_scale=residual_scale.detach().cpu(),
    )


def _safe_ratio(left: float, right: float, *, eps: float = 1e-12) -> float:
    return float(left) / max(float(right), eps)


def _iqr(values: torch.Tensor) -> float:
    quantiles = torch.quantile(
        values.detach().double().reshape(-1),
        torch.tensor([0.25, 0.75], dtype=torch.float64),
    )
    return float((quantiles[1] - quantiles[0]).item())


def persistence_gate_metrics(
    generated_paths: torch.Tensor,
    reference_paths: torch.Tensor,
    *,
    prefix_length: int = 65,
    prefix_window: int = 32,
    horizons: Sequence[int] = (5, 10, 20, 32),
    lags: Sequence[int] = (1, 2, 5, 10, 20),
    swd_projections: int = 32,
) -> dict[str, object]:
    generated = generated_paths.detach().cpu()
    reference = reference_paths.detach().cpu()
    generated_stats = volatility_path_statistics(
        generated,
        prefix_length=int(prefix_length),
        prefix_window=int(prefix_window),
        horizons=horizons,
        lags=lags,
    )
    reference_stats = volatility_path_statistics(
        reference,
        prefix_length=int(prefix_length),
        prefix_window=int(prefix_window),
        horizons=horizons,
        lags=lags,
    )
    stylized = stylized_fact_metrics(
        generated,
        reference,
        lags=lags,
        include_swd=True,
        swd_projections=int(swd_projections),
    )
    horizon_payload: dict[str, object] = {}
    for horizon in tuple(int(value) for value in horizons):
        generated_rv = generated_stats.future_realized_volatility[horizon]
        reference_rv = reference_stats.future_realized_volatility[horizon]
        generated_mean = float(generated_rv.mean().item())
        reference_mean = float(reference_rv.mean().item())
        generated_std = float(generated_rv.std(unbiased=False).item())
        reference_std = float(reference_rv.std(unbiased=False).item())
        mean_ratio = _safe_ratio(generated_mean, reference_mean)
        std_ratio = _safe_ratio(generated_std, reference_std)
        iqr_ratio = _safe_ratio(_iqr(generated_rv), _iqr(reference_rv))
        prefix = generated_stats.prefix_rms_volatility.double()
        future = generated_rv.double()
        reference_prefix = reference_stats.prefix_rms_volatility.double()
        reference_future = reference_rv.double()

        def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
            left_centered = left - left.mean()
            right_centered = right - right.mean()
            denominator = torch.sqrt(
                left_centered.pow(2).mean() * right_centered.pow(2).mean()
            ).clamp_min(1e-12)
            return float((left_centered * right_centered).mean().div(denominator).item())

        generated_correlation = correlation(prefix, future)
        reference_correlation = correlation(reference_prefix, reference_future)
        horizon_payload[str(horizon)] = {
            "future_rv_mean_ratio": mean_ratio,
            "future_rv_mean_abs_error": abs(mean_ratio - 1.0),
            "future_rv_std_ratio": std_ratio,
            "future_rv_std_abs_error": abs(std_ratio - 1.0),
            "future_rv_iqr_ratio": iqr_ratio,
            "future_rv_iqr_abs_error": abs(iqr_ratio - 1.0),
            "prefix_future_correlation": generated_correlation,
            "reference_prefix_future_correlation": reference_correlation,
            "prefix_future_correlation_abs_error": abs(
                generated_correlation - reference_correlation
            ),
        }
    return {
        "horizons": horizon_payload,
        "path_swd": float(stylized["path_swd"]),
        "abs_return_acf_error_rms": float(stylized["abs_return_acf_error_rms"]),
        "squared_return_acf_error_rms": float(stylized["squared_return_acf_error_rms"]),
        "excess_kurtosis_error_rms": float(stylized["excess_kurtosis_error_rms"]),
        "return_abs_mean_ratio": _safe_ratio(
            float(path_returns(generated).abs().mean().item()),
            float(path_returns(reference).abs().mean().item()),
        ),
    }


def persistence_gate_decision(
    candidate: dict[str, object],
    baseline: dict[str, object],
    *,
    horizon: int = 32,
    required_correlation_error_reduction: float = 0.25,
    scale_error_tolerance: float = 0.0,
    swd_factor_limit: float = 1.10,
) -> dict[str, object]:
    candidate_horizons = candidate.get("horizons")
    baseline_horizons = baseline.get("horizons")
    if not isinstance(candidate_horizons, dict) or not isinstance(baseline_horizons, dict):
        raise ValueError("gate metrics must contain horizon dictionaries")
    candidate_horizon = candidate_horizons.get(str(int(horizon)))
    baseline_horizon = baseline_horizons.get(str(int(horizon)))
    if not isinstance(candidate_horizon, dict) or not isinstance(baseline_horizon, dict):
        raise ValueError("requested gate horizon is missing")
    scale_checks = {
        name: float(candidate_horizon[name])
        <= float(baseline_horizon[name]) + float(scale_error_tolerance)
        for name in (
            "future_rv_mean_abs_error",
            "future_rv_std_abs_error",
            "future_rv_iqr_abs_error",
        )
    }
    swd_check = float(candidate["path_swd"]) <= float(swd_factor_limit) * float(
        baseline["path_swd"]
    )
    baseline_correlation_error = float(
        baseline_horizon["prefix_future_correlation_abs_error"]
    )
    candidate_correlation_error = float(
        candidate_horizon["prefix_future_correlation_abs_error"]
    )
    correlation_reduction = (
        baseline_correlation_error - candidate_correlation_error
    ) / max(baseline_correlation_error, 1e-12)
    correlation_check = correlation_reduction >= float(required_correlation_error_reduction)
    constraints_pass = all(scale_checks.values()) and swd_check
    return {
        "pass": bool(constraints_pass and correlation_check),
        "constraints_pass": bool(constraints_pass),
        "correlation_check": bool(correlation_check),
        "correlation_error_reduction": float(correlation_reduction),
        "required_correlation_error_reduction": float(required_correlation_error_reduction),
        "scale_checks": scale_checks,
        "scale_error_tolerance": float(scale_error_tolerance),
        "swd_check": bool(swd_check),
        "swd_factor": _safe_ratio(float(candidate["path_swd"]), float(baseline["path_swd"])),
        "swd_factor_limit": float(swd_factor_limit),
    }


def bootstrap_persistence_gate_decision(
    candidate: dict[str, object],
    baseline: dict[str, object],
    *,
    standard_errors: dict[str, float],
    horizon: int = 32,
    required_correlation_error_reduction: float = 0.10,
    marginal_standard_error_budget: float = 0.5,
    swd_factor_limit: float = 1.10,
) -> dict[str, object]:
    """Apply the persistence gate with bootstrap-sized marginal allowances.

    ``standard_errors`` are estimated once at the unperturbed validation
    baseline.  This makes every checkpoint face the same uncertainty budget
    and mirrors the Pareto budgets used by the direct causal-R feasibility
    audit.  The correlation requirement remains an effect-size threshold, not
    a significance claim.
    """

    if not math.isfinite(float(required_correlation_error_reduction)) or not (
        0.0 <= float(required_correlation_error_reduction) <= 1.0
    ):
        raise ValueError("required_correlation_error_reduction must be in [0, 1]")
    if not math.isfinite(float(marginal_standard_error_budget)) or float(
        marginal_standard_error_budget
    ) < 0.0:
        raise ValueError("marginal_standard_error_budget must be non-negative")
    if not math.isfinite(float(swd_factor_limit)) or float(swd_factor_limit) <= 0.0:
        raise ValueError("swd_factor_limit must be positive")

    candidate_horizons = candidate.get("horizons")
    baseline_horizons = baseline.get("horizons")
    if not isinstance(candidate_horizons, dict) or not isinstance(
        baseline_horizons,
        dict,
    ):
        raise ValueError("gate metrics must contain horizon dictionaries")
    candidate_horizon = candidate_horizons.get(str(int(horizon)))
    baseline_horizon = baseline_horizons.get(str(int(horizon)))
    if not isinstance(candidate_horizon, dict) or not isinstance(
        baseline_horizon,
        dict,
    ):
        raise ValueError("requested gate horizon is missing")

    marginal_names = (
        "future_rv_mean_abs_error",
        "future_rv_std_abs_error",
        "future_rv_iqr_abs_error",
    )
    allowances: dict[str, float] = {}
    changes: dict[str, float] = {}
    marginal_checks: dict[str, bool] = {}
    for name in marginal_names:
        if name not in standard_errors:
            raise ValueError(f"standard_errors is missing {name}")
        standard_error = float(standard_errors[name])
        if not math.isfinite(standard_error) or standard_error < 0.0:
            raise ValueError(f"standard error for {name} must be non-negative")
        allowance = float(marginal_standard_error_budget) * standard_error
        change = float(candidate_horizon[name]) - float(baseline_horizon[name])
        allowances[name] = allowance
        changes[name] = change
        marginal_checks[name] = change <= allowance

    baseline_correlation_error = float(
        baseline_horizon["prefix_future_correlation_abs_error"]
    )
    candidate_correlation_error = float(
        candidate_horizon["prefix_future_correlation_abs_error"]
    )
    correlation_error_change = candidate_correlation_error - baseline_correlation_error
    correlation_error_reduction = -correlation_error_change / max(
        baseline_correlation_error,
        1e-12,
    )
    correlation_check = correlation_error_reduction >= float(
        required_correlation_error_reduction
    )
    swd_factor = _safe_ratio(float(candidate["path_swd"]), float(baseline["path_swd"]))
    swd_check = swd_factor <= float(swd_factor_limit)
    constraints_pass = bool(all(marginal_checks.values()) and swd_check)
    return {
        "pass": bool(constraints_pass and correlation_check),
        "constraints_pass": constraints_pass,
        "correlation_check": bool(correlation_check),
        "correlation_error_change": float(correlation_error_change),
        "correlation_error_reduction": float(correlation_error_reduction),
        "required_correlation_error_reduction": float(
            required_correlation_error_reduction
        ),
        "marginal_checks": marginal_checks,
        "marginal_error_changes": changes,
        "marginal_error_allowances": allowances,
        "marginal_standard_error_budget": float(marginal_standard_error_budget),
        "baseline_standard_errors": {
            name: float(standard_errors[name]) for name in marginal_names
        },
        "swd_check": bool(swd_check),
        "swd_factor": float(swd_factor),
        "swd_factor_limit": float(swd_factor_limit),
    }


__all__ = [
    "CenteredNoiseAdjointResidualNetwork",
    "FixedTargetVolatilityCorrelationDiscrepancy",
    "FrozenBaseResidualRNetwork",
    "NoiseAdjointResidualNetwork",
    "ResidualRFitResult",
    "ResidualRTrainingConfig",
    "fit_fixed_target_volatility_correlation",
    "fit_residual_r_correction",
    "bootstrap_persistence_gate_decision",
    "persistence_gate_decision",
    "persistence_gate_metrics",
]
