from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from deep_mkv_gen_path_dt.discrepancies.analytic import (
    feature_path_vjp,
    returns_path_vjp,
)
from deep_mkv_gen_path_dt.discrepancies.mmd import (
    rbf_mmd2_value_and_feature_gradient,
    rbf_mmd_lions_potential_value_and_feature_gradient,
)
from deep_mkv_gen_path_dt.discrepancies import (
    ACFLagProductFeature,
    AbsReturnsFeature,
    CompositePathFunctionalDiscrepancy,
    MMDPathFunctionalDiscrepancy,
    ObservedPathFeature,
    PathFeatureMap,
    PathFunctionalDiscrepancyResult,
    RealizedVolatilityFeature,
    ReturnsFeature,
    RollingVolatilityACFLagProductFeature,
    SquaredReturnsFeature,
    TerminalFeature,
    TerminalReturnFeature,
    WeightedDiscrepancy,
    rbf_mmd2,
    rbf_mmd_lions_potential,
)
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


def _valid_lags(lags: Sequence[int], *, num_returns: int) -> tuple[int, ...]:
    values = tuple(int(lag) for lag in lags if 1 <= int(lag) < int(num_returns))
    if len(values) == 0:
        return (1,)
    return values


def _returns(paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
    if paths.ndim != 3:
        raise ValueError("paths must have shape (B, N + 1, d)")
    if int(paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("paths time dimension must align with grid")
    return paths[:, 1:, :] - paths[:, :-1, :]


@dataclass(frozen=True)
class RealizedVarianceFeature:
    name: str = "realized_variance"

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        rv = _returns(paths, grid=grid).pow(2) / float(grid.dt)
        return rv.reshape(int(paths.shape[0]), -1)

    def analytic_path_vjp(
        self,
        paths: torch.Tensor,
        feature_cotangent: torch.Tensor,
        *,
        grid: DiscreteTimeGrid,
    ) -> torch.Tensor:
        returns = _returns(paths, grid=grid)
        cotangent = feature_cotangent.reshape_as(returns)
        return returns_path_vjp(
            paths,
            2.0 * returns * cotangent / float(grid.dt),
        )


@dataclass(frozen=True)
class GlobalRealizedVarianceFeature:
    name: str = "global_realized_variance"

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        return (_returns(paths, grid=grid).pow(2) / float(grid.dt)).mean(dim=1)

    def analytic_path_vjp(
        self,
        paths: torch.Tensor,
        feature_cotangent: torch.Tensor,
        *,
        grid: DiscreteTimeGrid,
    ) -> torch.Tensor:
        returns = _returns(paths, grid=grid)
        cotangent = feature_cotangent.reshape(
            int(paths.shape[0]),
            1,
            int(paths.shape[2]),
        )
        return returns_path_vjp(
            paths,
            2.0
            * returns
            * cotangent
            / (float(grid.dt) * float(returns.shape[1])),
        )


@dataclass(frozen=True)
class StateRealizedVarianceFeature:
    """Flattened state/local-variance pairs for a path-dependent MMD block."""

    name: str = "state_realized_variance"

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        realized_variance = _returns(paths, grid=grid).pow(2) / float(grid.dt)
        pairs = torch.cat((paths[:, :-1, :], realized_variance), dim=-1)
        return pairs.reshape(int(paths.shape[0]), -1)

    def analytic_path_vjp(
        self,
        paths: torch.Tensor,
        feature_cotangent: torch.Tensor,
        *,
        grid: DiscreteTimeGrid,
    ) -> torch.Tensor:
        returns = _returns(paths, grid=grid)
        dimension = int(paths.shape[2])
        pair_bar = feature_cotangent.reshape(
            int(paths.shape[0]),
            int(grid.num_steps),
            2 * dimension,
        )
        result = torch.zeros_like(paths)
        result[:, :-1, :].add_(pair_bar[..., :dimension])
        result.add_(
            returns_path_vjp(
                paths,
                2.0
                * returns
                * pair_bar[..., dimension:]
                / float(grid.dt),
            )
        )
        return result


@dataclass(frozen=True)
class FixedTargetStandardizedFeatureMMD:
    """MMD after immutable, train-law feature standardization.

    The center and scale are fitted once on the complete training law.  Both
    generated and minibatched target paths use those same statistics, so the
    discrepancy remains sensitive to level and dispersion while its bandwidths
    are dimensionless and portable across Heston and intraday-return scales.
    """

    feature_map: PathFeatureMap
    target_center: torch.Tensor
    target_scale: torch.Tensor
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0)
    name: str = "fixed_target_standardized_mmd"

    def __post_init__(self) -> None:
        bandwidths = tuple(float(value) for value in self.bandwidths)
        if len(bandwidths) == 0 or any(
            not math.isfinite(value) or value <= 0.0 for value in bandwidths
        ):
            raise ValueError("bandwidths must be non-empty finite positive floats")
        if self.target_center.ndim != 2 or int(self.target_center.shape[0]) != 1:
            raise ValueError("target_center must have shape (1, feature_dim)")
        if self.target_scale.shape != self.target_center.shape:
            raise ValueError("target_scale must match target_center")
        if not torch.isfinite(self.target_center).all():
            raise ValueError("target_center must be finite")
        if not torch.isfinite(self.target_scale).all() or not torch.all(
            self.target_scale > 0.0
        ):
            raise ValueError("target_scale must be finite and strictly positive")
        object.__setattr__(self, "bandwidths", bandwidths)

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        generated_features = self.feature_map(generated_paths, grid=grid)
        target_paths = target_paths.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        target_features = self.feature_map(target_paths, grid=grid)
        center = self.target_center.to(
            device=generated_features.device,
            dtype=generated_features.dtype,
        )
        scale = self.target_scale.to(
            device=generated_features.device,
            dtype=generated_features.dtype,
        )
        if int(generated_features.shape[1]) != int(center.shape[1]):
            raise ValueError("feature dimension does not match fitted target statistics")
        generated_standardized = (generated_features - center) / scale
        target_standardized = (target_features - center) / scale
        value = rbf_mmd2(
            generated_standardized,
            target_standardized,
            bandwidths=tuple(self.bandwidths),
        )
        return PathFunctionalDiscrepancyResult(
            value=value,
            metrics={
                f"{self.name}_value": float(value.detach().item()),
                f"{self.name}_feature_dim": float(generated_features.shape[1]),
                f"{self.name}_target_standardization_fixed": 1.0,
                f"{self.name}_target_scale_min": float(scale.min().detach().item()),
                f"{self.name}_target_scale_median": float(scale.median().detach().item()),
                f"{self.name}_target_scale_max": float(scale.max().detach().item()),
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
        """MMD Lions potential evaluated at a frozen generated law."""

        query_features = self.feature_map(query_paths, grid=grid)
        population_features = self.feature_map(
            population_paths, grid=grid
        ).detach()
        target_features = self.feature_map(target_paths, grid=grid).detach()
        center = self.target_center.to(
            device=query_features.device,
            dtype=query_features.dtype,
        )
        scale = self.target_scale.to(
            device=query_features.device,
            dtype=query_features.dtype,
        )
        return rbf_mmd_lions_potential(
            (query_features - center) / scale,
            (population_features - center) / scale,
            (target_features - center) / scale,
            bandwidths=tuple(self.bandwidths),
        )

    def analytic_value_and_path_gradient(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> tuple[PathFunctionalDiscrepancyResult, torch.Tensor]:
        with torch.no_grad():
            generated_features = self.feature_map(generated_paths.detach(), grid=grid)
            target_features = self.feature_map(
                target_paths.detach().to(
                    device=generated_paths.device,
                    dtype=generated_paths.dtype,
                ),
                grid=grid,
            )
            center = self.target_center.to(
                device=generated_paths.device,
                dtype=generated_paths.dtype,
            )
            scale = self.target_scale.to(
                device=generated_paths.device,
                dtype=generated_paths.dtype,
            )
            value, standardized_gradient = rbf_mmd2_value_and_feature_gradient(
                (generated_features - center) / scale,
                (target_features - center) / scale,
                bandwidths=tuple(self.bandwidths),
            )
            path_gradient = feature_path_vjp(
                self.feature_map,
                generated_paths,
                standardized_gradient / scale,
                grid=grid,
            )
        return (
            PathFunctionalDiscrepancyResult(
                value=value,
                metrics={
                    f"{self.name}_value": float(value.item()),
                    f"{self.name}_feature_dim": float(generated_features.shape[1]),
                    f"{self.name}_target_standardization_fixed": 1.0,
                    f"{self.name}_target_scale_min": float(scale.min().item()),
                    f"{self.name}_target_scale_median": float(scale.median().item()),
                    f"{self.name}_target_scale_max": float(scale.max().item()),
                },
            ),
            path_gradient,
        )

    def analytic_lions_potential_and_path_gradient(
        self,
        *,
        query_paths: torch.Tensor,
        population_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            query_features = self.feature_map(query_paths.detach(), grid=grid)
            population_features = self.feature_map(
                population_paths.detach().to(
                    device=query_paths.device,
                    dtype=query_paths.dtype,
                ),
                grid=grid,
            )
            target_features = self.feature_map(
                target_paths.detach().to(
                    device=query_paths.device,
                    dtype=query_paths.dtype,
                ),
                grid=grid,
            )
            center = self.target_center.to(
                device=query_paths.device,
                dtype=query_paths.dtype,
            )
            scale = self.target_scale.to(
                device=query_paths.device,
                dtype=query_paths.dtype,
            )
            value, standardized_gradient = (
                rbf_mmd_lions_potential_value_and_feature_gradient(
                    (query_features - center) / scale,
                    (population_features - center) / scale,
                    (target_features - center) / scale,
                    bandwidths=tuple(self.bandwidths),
                )
            )
            path_gradient = feature_path_vjp(
                self.feature_map,
                query_paths,
                standardized_gradient / scale,
                grid=grid,
            )
        return value, path_gradient


def fit_fixed_target_standardized_feature_mmd(
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    feature_map: PathFeatureMap,
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0),
    name: str | None = None,
) -> FixedTargetStandardizedFeatureMMD:
    """Fit robust train-only center/scale statistics for one feature block."""

    with torch.no_grad():
        features = feature_map(target_paths, grid=grid).to(dtype=torch.float64)
        if features.ndim != 2 or int(features.shape[0]) < 2:
            raise ValueError("target feature matrix must have shape (B, p) with B >= 2")
        center = torch.quantile(features, 0.5, dim=0, keepdim=True)
        mad = torch.quantile((features - center).abs(), 0.5, dim=0, keepdim=True)
        robust_scale = 1.4826 * mad
        fallback_scale = features.std(dim=0, keepdim=True, unbiased=False)
        positive = fallback_scale[fallback_scale > 0.0]
        global_floor = (
            float(positive.median().item()) * 1e-6
            if int(positive.numel()) > 0
            else 1.0
        )
        scale = torch.where(
            robust_scale > global_floor,
            robust_scale,
            fallback_scale,
        )
        scale = torch.where(
            scale > global_floor,
            scale,
            torch.ones_like(scale),
        )
    return FixedTargetStandardizedFeatureMMD(
        feature_map=feature_map,
        target_center=center.detach(),
        target_scale=scale.detach(),
        bandwidths=tuple(float(value) for value in bandwidths),
        name=(
            f"fixed_target_standardized_{feature_map.name}"
            if name is None
            else str(name)
        ),
    )


@dataclass(frozen=True)
class FixedTargetACFMomentDiscrepancy:
    """Direct train-fitted autocorrelation law functional.

    The existing lag-product MMD matches the distribution of pathwise lag
    products, but it need not match their pooled correlation.  This functional
    instead matches that population moment explicitly.  It depends only on
    observed returns, remains reference-free, and exposes an exact frozen-law
    Lions potential through its finite collection of law statistics.
    """

    target_acf: torch.Tensor
    target_variance: torch.Tensor
    lags: Sequence[int]
    transform: str = "absolute"
    variance_floor_fraction: float = 0.05
    eps: float = 1e-12
    name: str = "acf_moments"

    def __post_init__(self) -> None:
        lags = tuple(int(value) for value in self.lags)
        if len(lags) == 0 or any(value < 1 for value in lags):
            raise ValueError("lags must be non-empty positive integers")
        if tuple(sorted(set(lags))) != lags:
            raise ValueError("lags must be strictly increasing")
        if str(self.transform) not in {"absolute", "squared"}:
            raise ValueError("transform must be absolute or squared")
        fraction = float(self.variance_floor_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("variance_floor_fraction must lie in (0, 1]")
        eps = float(self.eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")
        if self.target_acf.ndim != 2 or int(self.target_acf.shape[0]) != len(lags):
            raise ValueError("target_acf must have shape (len(lags), state_dim)")
        if tuple(self.target_variance.shape) != (int(self.target_acf.shape[1]),):
            raise ValueError("target_variance must have shape (state_dim,)")
        if not bool(torch.isfinite(self.target_acf).all()):
            raise ValueError("target_acf must be finite")
        if not bool(torch.isfinite(self.target_variance).all()) or not bool(
            torch.all(self.target_variance > 0.0)
        ):
            raise ValueError("target_variance must be finite and positive")
        object.__setattr__(self, "lags", lags)
        object.__setattr__(self, "transform", str(self.transform))
        object.__setattr__(self, "variance_floor_fraction", fraction)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "target_acf", self.target_acf.detach().cpu().clone())
        object.__setattr__(
            self,
            "target_variance",
            self.target_variance.detach().cpu().clone(),
        )

    def _values(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        returns = _returns(paths, grid=grid)
        return returns.abs() if self.transform == "absolute" else returns.pow(2)

    def _law_statistics(
        self,
        paths: torch.Tensor,
        *,
        grid: DiscreteTimeGrid,
    ) -> torch.Tensor:
        values = self._values(paths, grid=grid)
        num_steps = int(values.shape[1])
        if any(int(lag) >= num_steps for lag in self.lags):
            raise ValueError("every ACF lag must be smaller than the return count")
        rows = [values.mean(dim=1), values.pow(2).mean(dim=1)]
        for lag in self.lags:
            left = values[:, :-int(lag), :]
            right = values[:, int(lag) :, :]
            rows.extend(
                (
                    left.mean(dim=1),
                    right.mean(dim=1),
                    (left * right).mean(dim=1),
                )
            )
        return torch.cat(rows, dim=1)

    def _acf_from_mean_statistics(self, mean_statistics: torch.Tensor) -> torch.Tensor:
        state_dim = int(self.target_acf.shape[1])
        offset = 0
        mean = mean_statistics[offset : offset + state_dim]
        offset += state_dim
        second = mean_statistics[offset : offset + state_dim]
        offset += state_dim
        variance = (second - mean.pow(2)).clamp_min(0.0)
        target_variance = self.target_variance.to(
            device=mean_statistics.device,
            dtype=mean_statistics.dtype,
        )
        denominator = torch.maximum(
            variance,
            float(self.variance_floor_fraction) * target_variance,
        ).clamp_min(float(self.eps))
        correlations = []
        for _lag in self.lags:
            left_mean = mean_statistics[offset : offset + state_dim]
            offset += state_dim
            right_mean = mean_statistics[offset : offset + state_dim]
            offset += state_dim
            cross = mean_statistics[offset : offset + state_dim]
            offset += state_dim
            covariance = cross - mean * left_mean - mean * right_mean + mean.pow(2)
            correlations.append(covariance / denominator)
        return torch.stack(correlations, dim=0)

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        del target_paths
        generated_statistics = self._law_statistics(
            generated_paths,
            grid=grid,
        ).mean(dim=0)
        generated_acf = self._acf_from_mean_statistics(generated_statistics)
        target_acf = self.target_acf.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        errors = generated_acf - target_acf
        value = errors.pow(2).mean()
        metrics = {
            f"{self.name}_value": float(value.detach().item()),
            f"{self.name}_rmse": float(torch.sqrt(value.detach()).item()),
            f"{self.name}_variance_floor_fraction": float(
                self.variance_floor_fraction
            ),
        }
        for lag_index, lag in enumerate(self.lags):
            metrics[f"{self.name}_lag_{lag}_generated_mean"] = float(
                generated_acf[lag_index].mean().detach().item()
            )
            metrics[f"{self.name}_lag_{lag}_target_mean"] = float(
                target_acf[lag_index].mean().detach().item()
            )
        return PathFunctionalDiscrepancyResult(value=value, metrics=metrics)

    def lions_potential(
        self,
        *,
        query_paths: torch.Tensor,
        population_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> torch.Tensor:
        del target_paths
        query_statistics = self._law_statistics(query_paths, grid=grid)
        with torch.no_grad():
            population_mean = self._law_statistics(
                population_paths,
                grid=grid,
            ).mean(dim=0)
        population_mean_req = population_mean.detach().clone().requires_grad_(True)
        population_acf = self._acf_from_mean_statistics(population_mean_req)
        target_acf = self.target_acf.to(
            device=population_mean_req.device,
            dtype=population_mean_req.dtype,
        )
        population_value = (population_acf - target_acf).pow(2).mean()
        coefficients = torch.autograd.grad(
            population_value,
            population_mean_req,
        )[0].detach()
        return (query_statistics * coefficients).sum(dim=1).mean()


def fit_fixed_target_acf_moments(
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    lags: Sequence[int],
    transform: str,
    variance_floor_fraction: float = 0.05,
    eps: float = 1e-12,
    name: str | None = None,
) -> FixedTargetACFMomentDiscrepancy:
    """Fit immutable pooled ACF targets on the complete training law."""

    lag_values = _valid_lags(lags, num_returns=int(grid.num_steps))
    returns = _returns(target_paths, grid=grid)
    if str(transform) == "absolute":
        values = returns.abs()
    elif str(transform) == "squared":
        values = returns.pow(2)
    else:
        raise ValueError("transform must be absolute or squared")
    centered = values - values.mean(dim=(0, 1), keepdim=True)
    variance = centered.pow(2).mean(dim=(0, 1)).clamp_min(float(eps))
    correlations = torch.stack(
        [
            (centered[:, :-lag, :] * centered[:, lag:, :]).mean(dim=(0, 1))
            / variance
            for lag in lag_values
        ],
        dim=0,
    )
    return FixedTargetACFMomentDiscrepancy(
        target_acf=correlations,
        target_variance=variance,
        lags=lag_values,
        transform=str(transform),
        variance_floor_fraction=float(variance_floor_fraction),
        eps=float(eps),
        name=(f"{transform}_return_acf_moments" if name is None else str(name)),
    )


@dataclass(frozen=True)
class FixedTargetPooledReturnWassersteinDiscrepancy:
    """Train-fitted empirical Wasserstein-2 discrepancy for pooled returns.

    Returns are pooled over paths and time, but kept separate by asset.  The
    target empirical quantile function and its standardization are fitted once
    on the complete training split.  Consequently, minibatch target sampling
    does not inject noise into this law-level path functional, and no
    parametric return model is introduced.
    """

    target_mean: torch.Tensor
    target_std: torch.Tensor
    target_iqr: torch.Tensor
    target_sorted_standardized: torch.Tensor
    eps: float = 1e-12
    name: str = "fixed_target_pooled_return_wasserstein"

    def __post_init__(self) -> None:
        eps = float(self.eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        if not isinstance(self.target_mean, torch.Tensor) or self.target_mean.ndim != 1:
            raise ValueError("target_mean must have shape (d,)")
        if (
            not isinstance(self.target_std, torch.Tensor)
            or self.target_std.shape != self.target_mean.shape
        ):
            raise ValueError("target_std must have shape (d,)")
        if (
            not isinstance(self.target_iqr, torch.Tensor)
            or self.target_iqr.shape != self.target_mean.shape
        ):
            raise ValueError("target_iqr must have shape (d,)")
        sorted_target = self.target_sorted_standardized
        if not isinstance(sorted_target, torch.Tensor) or sorted_target.ndim != 2:
            raise ValueError("target_sorted_standardized must have shape (M, d)")
        if int(sorted_target.shape[0]) < 2 or int(sorted_target.shape[1]) != int(
            self.target_mean.numel()
        ):
            raise ValueError("target quantiles must contain at least two returns per asset")
        tensors = (self.target_mean, self.target_std, self.target_iqr, sorted_target)
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("fixed target return statistics must be finite")
        if bool((self.target_std <= 0.0).any()) or bool((self.target_iqr < 0.0).any()):
            raise ValueError("fixed target return scales must be valid")
        if bool((sorted_target[1:] < sorted_target[:-1]).any()):
            raise ValueError("target standardized returns must be sorted")
        object.__setattr__(self, "eps", eps)
        for field_name in (
            "target_mean",
            "target_std",
            "target_iqr",
            "target_sorted_standardized",
        ):
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, value.detach().cpu().clone())

    @staticmethod
    def _interpolated_quantiles(
        sorted_values: torch.Tensor,
        *,
        count: int,
    ) -> torch.Tensor:
        target_count = int(sorted_values.shape[0])
        if int(count) < 2:
            raise ValueError("generated return law requires at least two returns")
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
        generated_returns = _returns(generated_paths, grid=grid).reshape(
            -1,
            int(generated_paths.shape[2]),
        )
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
        if int(generated_returns.shape[1]) != int(target_mean.numel()):
            raise ValueError("generated and fixed-target asset dimensions must match")

        generated_standardized = (generated_returns - target_mean) / target_std
        generated_sorted = torch.sort(generated_standardized, dim=0).values
        target_quantiles = self._interpolated_quantiles(
            sorted_target,
            count=int(generated_sorted.shape[0]),
        )
        wasserstein_by_asset = (generated_sorted - target_quantiles).pow(2).mean(dim=0)
        value = wasserstein_by_asset.mean()

        generated_mean = generated_returns.mean(dim=0)
        generated_std = generated_returns.std(dim=0, unbiased=False)
        quantile_levels = torch.tensor(
            [0.25, 0.75],
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        generated_quartiles = torch.quantile(
            generated_returns,
            quantile_levels,
            dim=0,
        )
        generated_iqr = generated_quartiles[1] - generated_quartiles[0]
        metrics: dict[str, float] = {
            f"{self.name}_value": float(value.detach().item()),
            f"{self.name}_fixed_target": 1.0,
            f"{self.name}_target_return_count": float(sorted_target.shape[0]),
        }
        tiny = torch.finfo(torch.float64).eps
        for asset in range(int(generated_returns.shape[1])):
            prefix = f"{self.name}_asset{asset}"
            metrics[f"{prefix}_wasserstein2"] = float(
                wasserstein_by_asset[asset].detach().item()
            )
            for statistic, generated_value, target_value in (
                ("mean", generated_mean[asset], target_mean[asset]),
                ("std", generated_std[asset], target_std[asset]),
                ("iqr", generated_iqr[asset], target_iqr[asset]),
            ):
                generated_scalar = float(generated_value.detach().item())
                target_scalar = float(target_value.detach().item())
                metrics[f"{prefix}_generated_{statistic}"] = generated_scalar
                metrics[f"{prefix}_target_{statistic}"] = target_scalar
                metrics[f"{prefix}_{statistic}_ratio"] = generated_scalar / max(
                    abs(target_scalar), tiny
                )
        return PathFunctionalDiscrepancyResult(value=value, metrics=metrics)

    def lions_potential(
        self,
        *,
        query_paths: torch.Tensor,
        population_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> torch.Tensor:
        """Frozen-law one-dimensional Wasserstein influence potential.

        In one dimension the Lions derivative of squared W2 has spatial
        derivative ``2 * (z - T_mu_to_nu(z))``.  The empirical transport map
        is frozen from the unconditional population ranks; the returned
        query average includes the path/time/asset normalization of the pooled
        return functional.
        """

        del target_paths
        query_returns = _returns(query_paths, grid=grid)
        population_returns = _returns(population_paths, grid=grid).detach()
        state_dim = int(query_returns.shape[2])
        num_returns = int(query_returns.shape[1])
        if int(population_returns.shape[2]) != state_dim:
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
        query_standardized = (query_returns - target_mean) / target_std
        population_standardized = (
            population_returns - target_mean
        ) / target_std
        coefficients = torch.empty_like(query_returns)
        population_count = int(population_standardized.shape[0]) * num_returns
        target_at_population_ranks = self._interpolated_quantiles(
            sorted_target,
            count=population_count,
        )
        for asset in range(state_dim):
            population_sorted = torch.sort(
                population_standardized[:, :, asset].reshape(-1)
            ).values
            query_values = query_standardized[:, :, asset]
            ranks = torch.searchsorted(
                population_sorted,
                query_values.detach().contiguous(),
                right=False,
            ).clamp(max=population_count - 1)
            transported_target = target_at_population_ranks[:, asset].index_select(
                0,
                ranks.reshape(-1),
            ).reshape_as(query_values)
            coefficients[:, :, asset] = (
                2.0
                * (query_values.detach() - transported_target)
                / (
                    float(num_returns * state_dim)
                    * target_std[asset]
                )
            )
        return (coefficients.detach() * query_returns).sum(dim=(1, 2)).mean()


def fit_fixed_target_pooled_return_wasserstein(
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    eps: float = 1e-12,
) -> FixedTargetPooledReturnWassersteinDiscrepancy:
    """Fit a dimensionless pooled-return empirical quantile functional."""

    returns = _returns(target_paths, grid=grid).reshape(
        -1,
        int(target_paths.shape[2]),
    )
    if int(returns.shape[0]) < 2:
        raise ValueError("target return law requires at least two returns")
    target_mean = returns.mean(dim=0)
    target_std = returns.std(dim=0, unbiased=False).clamp_min(float(eps))
    target_standardized = (returns - target_mean) / target_std
    sorted_standardized = torch.sort(target_standardized, dim=0).values
    quartiles = torch.quantile(
        returns,
        torch.tensor([0.25, 0.75], device=returns.device, dtype=returns.dtype),
        dim=0,
    )
    return FixedTargetPooledReturnWassersteinDiscrepancy(
        target_mean=target_mean,
        target_std=target_std,
        target_iqr=quartiles[1] - quartiles[0],
        target_sorted_standardized=sorted_standardized,
        eps=float(eps),
    )


def build_real_data_standardized_discrepancy(
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0),
    acf_lags: Sequence[int] = (1, 2, 5, 10),
) -> CompositePathFunctionalDiscrepancy:
    """Equal-weight, scale-portable path-law discrepancy for real sessions.

    All numerical scales are estimated from the training law.  The ACF maps
    deliberately normalize within each law before their outer train-standardized
    MMD, making those two blocks target dependence rather than volatility level.
    """

    if int(target_paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("target_paths must align with grid")
    lags = _valid_lags(acf_lags, num_returns=int(grid.num_steps))
    window = min(5, int(grid.num_steps))
    feature_maps: tuple[tuple[str, PathFeatureMap], ...] = (
        ("path_law_observed_path", ObservedPathFeature()),
        ("path_law_returns", ReturnsFeature()),
        ("volatility_law_abs_returns", AbsReturnsFeature()),
        ("volatility_law_squared_returns", SquaredReturnsFeature()),
        ("volatility_law_global_rv", GlobalRealizedVarianceFeature()),
        (
            "volatility_law_rolling_rv",
            RealizedVolatilityFeature(window=window, eps=1e-16),
        ),
        ("volatility_law_state_rv", StateRealizedVarianceFeature()),
        (
            "volatility_law_abs_return_acf",
            ACFLagProductFeature(
                lags=lags,
                squared=False,
                eps=1e-12,
                smoothing_epsilon=0.0,
            ),
        ),
        (
            "volatility_law_squared_return_acf",
            ACFLagProductFeature(
                lags=lags,
                squared=True,
                eps=1e-16,
                smoothing_epsilon=0.0,
            ),
        ),
    )
    blocks = tuple(
        WeightedDiscrepancy(
            name=block_name,
            weight=1.0,
            discrepancy=fit_fixed_target_standardized_feature_mmd(
                target_paths,
                grid=grid,
                feature_map=feature_map,
                bandwidths=bandwidths,
                name=f"mmd_{block_name}",
            ),
        )
        for block_name, feature_map in feature_maps
    )
    return CompositePathFunctionalDiscrepancy(blocks=blocks)


@dataclass(frozen=True)
class TargetStandardizedStateRealizedVarianceMMD:
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0, 4.0)
    eps: float = 1e-6
    name: str = "state_realized_variance"

    def __post_init__(self) -> None:
        bandwidths = tuple(float(value) for value in self.bandwidths)
        if len(bandwidths) == 0:
            raise ValueError("bandwidths must be non-empty")
        if any(value <= 0.0 for value in bandwidths):
            raise ValueError("bandwidths must be > 0")
        if float(self.eps) <= 0.0:
            raise ValueError("eps must be > 0")
        object.__setattr__(self, "bandwidths", bandwidths)
        object.__setattr__(self, "eps", float(self.eps))

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        generated_rv = _returns(generated_paths, grid=grid).pow(2) / float(grid.dt)
        target_paths = target_paths.to(device=generated_paths.device, dtype=generated_paths.dtype)
        target_rv = _returns(target_paths, grid=grid).pow(2) / float(grid.dt)
        generated_pair = torch.cat([generated_paths[:, :-1, :], generated_rv], dim=-1)
        target_pair = torch.cat([target_paths[:, :-1, :], target_rv], dim=-1)
        mean = target_pair.mean(dim=(0, 1), keepdim=True).detach()
        std = target_pair.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(float(self.eps)).detach()
        generated_features = ((generated_pair - mean) / std).reshape(int(generated_paths.shape[0]), -1)
        target_features = ((target_pair - mean) / std).reshape(int(target_paths.shape[0]), -1)
        value = rbf_mmd2(generated_features, target_features, bandwidths=tuple(self.bandwidths))
        return PathFunctionalDiscrepancyResult(
            value=value,
            metrics={
                f"{self.name}_value": float(value.detach().item()),
                f"{self.name}_feature_dim": float(generated_features.shape[1]),
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
        """Frozen-law potential for the target-standardized state/RV MMD."""

        target_paths = target_paths.to(
            device=query_paths.device,
            dtype=query_paths.dtype,
        )

        def pairs(paths: torch.Tensor) -> torch.Tensor:
            realized_variance = _returns(paths, grid=grid).pow(2) / float(grid.dt)
            return torch.cat((paths[:, :-1, :], realized_variance), dim=-1)

        target_pair = pairs(target_paths)
        target_mean = target_pair.mean(dim=(0, 1), keepdim=True).detach()
        target_std = (
            target_pair.std(dim=(0, 1), keepdim=True, unbiased=False)
            .clamp_min(float(self.eps))
            .detach()
        )

        def features(paths: torch.Tensor) -> torch.Tensor:
            standardized = (pairs(paths) - target_mean) / target_std
            return standardized.reshape(int(paths.shape[0]), -1)

        return rbf_mmd_lions_potential(
            features(query_paths),
            features(
                population_paths.to(
                    device=query_paths.device,
                    dtype=query_paths.dtype,
                )
            ).detach(),
            features(target_paths).detach(),
            bandwidths=tuple(self.bandwidths),
        )

    def _analytic_features_and_vjp(
        self,
        paths: torch.Tensor,
        *,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> tuple[torch.Tensor, object]:
        returns = _returns(paths, grid=grid)
        realized_variance = returns.pow(2) / float(grid.dt)
        pair = torch.cat((paths[:, :-1, :], realized_variance), dim=-1)
        features = ((pair - target_mean) / target_std).reshape(
            int(paths.shape[0]),
            -1,
        )

        def vjp(feature_cotangent: torch.Tensor) -> torch.Tensor:
            pair_bar = feature_cotangent.reshape_as(pair) / target_std
            dimension = int(paths.shape[2])
            result = torch.zeros_like(paths)
            result[:, :-1, :].add_(pair_bar[..., :dimension])
            result.add_(
                returns_path_vjp(
                    paths,
                    2.0
                    * returns
                    * pair_bar[..., dimension:]
                    / float(grid.dt),
                )
            )
            return result

        return features, vjp

    def analytic_value_and_path_gradient(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> tuple[PathFunctionalDiscrepancyResult, torch.Tensor]:
        with torch.no_grad():
            generated_paths = generated_paths.detach()
            target_paths = target_paths.detach().to(
                device=generated_paths.device,
                dtype=generated_paths.dtype,
            )
            target_returns = _returns(target_paths, grid=grid)
            target_pair = torch.cat(
                (
                    target_paths[:, :-1, :],
                    target_returns.pow(2) / float(grid.dt),
                ),
                dim=-1,
            )
            target_mean = target_pair.mean(dim=(0, 1), keepdim=True)
            target_std = target_pair.std(
                dim=(0, 1),
                keepdim=True,
                unbiased=False,
            ).clamp_min(float(self.eps))
            generated_features, generated_vjp = self._analytic_features_and_vjp(
                generated_paths,
                target_mean=target_mean,
                target_std=target_std,
                grid=grid,
            )
            target_features = ((target_pair - target_mean) / target_std).reshape(
                int(target_paths.shape[0]),
                -1,
            )
            value, feature_gradient = rbf_mmd2_value_and_feature_gradient(
                generated_features,
                target_features,
                bandwidths=tuple(self.bandwidths),
            )
            path_gradient = generated_vjp(feature_gradient)
        return (
            PathFunctionalDiscrepancyResult(
                value=value,
                metrics={
                    f"{self.name}_value": float(value.item()),
                    f"{self.name}_feature_dim": float(generated_features.shape[1]),
                },
            ),
            path_gradient,
        )

    def analytic_lions_potential_and_path_gradient(
        self,
        *,
        query_paths: torch.Tensor,
        population_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            query_paths = query_paths.detach()
            population_paths = population_paths.detach().to(
                device=query_paths.device,
                dtype=query_paths.dtype,
            )
            target_paths = target_paths.detach().to(
                device=query_paths.device,
                dtype=query_paths.dtype,
            )
            target_returns = _returns(target_paths, grid=grid)
            target_pair = torch.cat(
                (
                    target_paths[:, :-1, :],
                    target_returns.pow(2) / float(grid.dt),
                ),
                dim=-1,
            )
            target_mean = target_pair.mean(dim=(0, 1), keepdim=True)
            target_std = target_pair.std(
                dim=(0, 1),
                keepdim=True,
                unbiased=False,
            ).clamp_min(float(self.eps))
            query_features, query_vjp = self._analytic_features_and_vjp(
                query_paths,
                target_mean=target_mean,
                target_std=target_std,
                grid=grid,
            )
            population_features, _ = self._analytic_features_and_vjp(
                population_paths,
                target_mean=target_mean,
                target_std=target_std,
                grid=grid,
            )
            target_features = ((target_pair - target_mean) / target_std).reshape(
                int(target_paths.shape[0]),
                -1,
            )
            value, feature_gradient = (
                rbf_mmd_lions_potential_value_and_feature_gradient(
                    query_features,
                    population_features,
                    target_features,
                    bandwidths=tuple(self.bandwidths),
                )
            )
            path_gradient = query_vjp(feature_gradient)
        return value, path_gradient


@dataclass(frozen=True)
class TimeMarginalRealizedVolatilityMMD:
    """Average MMD over local realized-volatility marginals at selected dates.

    Each date is standardized with that target marginal's mean and standard
    deviation, but the generated law uses those same fixed target statistics.
    Consequently level, dispersion, and shape mismatches remain visible while
    every date contributes on a comparable dimensionless scale. Marginals are
    compared independently, leaving their temporal dependence to a separate
    path-dependent discrepancy.
    """

    window: int = 4
    endpoint_indices: Sequence[int] = ()
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0)
    eps: float = 1e-6
    name: str = "time_marginal_realized_volatility"

    def __post_init__(self) -> None:
        window = int(self.window)
        endpoints = tuple(int(value) for value in self.endpoint_indices)
        bandwidths = tuple(float(value) for value in self.bandwidths)
        eps = float(self.eps)
        if window < 1:
            raise ValueError("window must be >= 1")
        if any(value < 1 for value in endpoints):
            raise ValueError("endpoint_indices must contain positive integers")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("endpoint_indices must not contain duplicates")
        if tuple(sorted(endpoints)) != endpoints:
            raise ValueError("endpoint_indices must be strictly increasing")
        if len(bandwidths) == 0 or any(
            not math.isfinite(value) or value <= 0.0 for value in bandwidths
        ):
            raise ValueError("bandwidths must be non-empty finite positive floats")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "endpoint_indices", endpoints)
        object.__setattr__(self, "bandwidths", bandwidths)
        object.__setattr__(self, "eps", eps)

    def _endpoints(self, *, num_steps: int) -> tuple[int, ...]:
        if self.endpoint_indices:
            endpoints = tuple(int(value) for value in self.endpoint_indices)
        else:
            endpoints = tuple(
                dict.fromkeys(
                    max(int(self.window), min(num_steps, (fraction * num_steps) // 4))
                    for fraction in (1, 2, 3, 4)
                )
            )
        if any(value < int(self.window) or value > num_steps for value in endpoints):
            raise ValueError("every endpoint must satisfy window <= endpoint <= num_steps")
        return endpoints

    def _local_volatility(
        self,
        paths: torch.Tensor,
        *,
        grid: DiscreteTimeGrid,
        endpoints: tuple[int, ...],
    ) -> torch.Tensor:
        scaled_returns = _returns(paths, grid=grid) / math.sqrt(float(grid.dt))
        eps_squared = float(self.eps) ** 2
        return torch.stack(
            [
                torch.sqrt(
                    scaled_returns[:, endpoint - int(self.window) : endpoint, :]
                    .pow(2)
                    .mean(dim=1)
                    + eps_squared
                )
                for endpoint in endpoints
            ],
            dim=1,
        )

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        if int(generated_paths.shape[0]) < 2 or int(target_paths.shape[0]) < 2:
            raise ValueError("time-marginal volatility MMD requires at least two paths per law")
        endpoints = self._endpoints(num_steps=int(grid.num_steps))
        target_paths = target_paths.to(device=generated_paths.device, dtype=generated_paths.dtype)
        generated_volatility = self._local_volatility(
            generated_paths,
            grid=grid,
            endpoints=endpoints,
        )
        target_volatility = self._local_volatility(
            target_paths,
            grid=grid,
            endpoints=endpoints,
        )
        marginal_values = []
        metrics: dict[str, float] = {
            f"{self.name}_window": float(self.window),
            f"{self.name}_endpoint_count": float(len(endpoints)),
        }
        for position, endpoint in enumerate(endpoints):
            target_at_time = target_volatility[:, position, :]
            generated_at_time = generated_volatility[:, position, :]
            target_mean = target_at_time.mean(dim=0, keepdim=True).detach()
            target_std = target_at_time.std(dim=0, keepdim=True, unbiased=False).clamp_min(
                float(self.eps)
            ).detach()
            generated_standardized = (generated_at_time - target_mean) / target_std
            target_standardized = (target_at_time - target_mean) / target_std
            marginal_value = rbf_mmd2(
                generated_standardized,
                target_standardized,
                bandwidths=tuple(self.bandwidths),
            )
            marginal_values.append(marginal_value)
            generated_mean = generated_at_time.mean()
            generated_std = generated_at_time.std(unbiased=False)
            target_mean_scalar = target_at_time.mean()
            target_std_scalar = target_at_time.std(unbiased=False)
            metrics[f"{self.name}_endpoint_{endpoint}_mmd"] = float(
                marginal_value.detach().item()
            )
            metrics[f"{self.name}_endpoint_{endpoint}_generated_mean"] = float(
                generated_mean.detach().item()
            )
            metrics[f"{self.name}_endpoint_{endpoint}_target_mean"] = float(
                target_mean_scalar.detach().item()
            )
            metrics[f"{self.name}_endpoint_{endpoint}_generated_std"] = float(
                generated_std.detach().item()
            )
            metrics[f"{self.name}_endpoint_{endpoint}_target_std"] = float(
                target_std_scalar.detach().item()
            )
        value = torch.stack(marginal_values).mean()
        metrics[f"{self.name}_value"] = float(value.detach().item())
        return PathFunctionalDiscrepancyResult(value=value, metrics=metrics)


@dataclass(frozen=True)
class MarginalInvariantConditionalVolatilityMMD:
    """Marginal-invariant MMD for multiscale volatility dependence.

    RMS and absolute-return volatility over matching prefix windows and future
    horizons are standardized within each empirical law.  The feature vector
    contains all prefix/future interactions.  Matching it targets volatility
    dependence while leaving marginal scale and dispersion to the existing
    volatility-law blocks, without introducing a latent-volatility model or
    reference kernel.
    """

    windows: Sequence[int] = (1, 2, 4, 8)
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0)
    split_index: int | None = None
    eps: float = 1e-6
    name: str = "conditional_multiscale_volatility"

    def __post_init__(self) -> None:
        windows = tuple(int(value) for value in self.windows)
        bandwidths = tuple(float(value) for value in self.bandwidths)
        if len(windows) == 0 or any(value < 1 for value in windows):
            raise ValueError("windows must be non-empty positive integers")
        if len(set(windows)) != len(windows):
            raise ValueError("windows must not contain duplicates")
        if len(bandwidths) == 0 or any(not math.isfinite(value) or value <= 0.0 for value in bandwidths):
            raise ValueError("bandwidths must be non-empty finite positive floats")
        split_index = None if self.split_index is None else int(self.split_index)
        if split_index is not None and split_index < 1:
            raise ValueError("split_index must be >= 1 when provided")
        eps = float(self.eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "bandwidths", bandwidths)
        object.__setattr__(self, "split_index", split_index)
        object.__setattr__(self, "eps", eps)

    def _base_features(
        self,
        paths: torch.Tensor,
        *,
        grid: DiscreteTimeGrid,
        split_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scaled_returns = _returns(paths, grid=grid) / math.sqrt(float(grid.dt))
        prefix_rms = []
        prefix_abs = []
        future_rms = []
        future_abs = []
        eps_squared = float(self.eps) ** 2
        for window in tuple(int(value) for value in self.windows):
            prefix = scaled_returns[:, split_index - window : split_index, :]
            future = scaled_returns[:, split_index : split_index + window, :]
            prefix_rms.append(torch.sqrt(prefix.pow(2).mean(dim=1) + eps_squared))
            prefix_abs.append(torch.sqrt(prefix.pow(2) + eps_squared).mean(dim=1))
            future_rms.append(torch.sqrt(future.pow(2).mean(dim=1) + eps_squared))
            future_abs.append(torch.sqrt(future.pow(2) + eps_squared).mean(dim=1))
        return (
            torch.stack(prefix_rms, dim=1),
            torch.stack(prefix_abs, dim=1),
            torch.stack(future_rms, dim=1),
            torch.stack(future_abs, dim=1),
        )

    def _interaction_features(self, standardized: torch.Tensor) -> torch.Tensor:
        window_count = len(tuple(self.windows))
        prefix_rms, prefix_abs, future_rms, future_abs = standardized.split(window_count, dim=1)
        rms_interactions = prefix_rms.unsqueeze(2) * future_rms.unsqueeze(1)
        abs_interactions = prefix_abs.unsqueeze(2) * future_abs.unsqueeze(1)
        return torch.cat(
            [
                rms_interactions.reshape(int(standardized.shape[0]), -1),
                abs_interactions.reshape(int(standardized.shape[0]), -1),
            ],
            dim=1,
        )

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        num_steps = int(grid.num_steps)
        split_index = int(self.split_index) if self.split_index is not None else num_steps // 2
        largest_window = max(tuple(int(value) for value in self.windows))
        if split_index < largest_window or num_steps - split_index < largest_window:
            raise ValueError("every window must fit before and after the conditional-volatility split")
        target_paths = target_paths.to(device=generated_paths.device, dtype=generated_paths.dtype)
        generated_base = torch.cat(
            self._base_features(generated_paths, grid=grid, split_index=split_index),
            dim=1,
        )
        target_base = torch.cat(
            self._base_features(target_paths, grid=grid, split_index=split_index),
            dim=1,
        )
        target_mean = target_base.mean(dim=0, keepdim=True).detach()
        target_scale = target_base.std(dim=0, keepdim=True, unbiased=False).clamp_min(float(self.eps)).detach()
        generated_mean = generated_base.mean(dim=0, keepdim=True)
        generated_scale = generated_base.std(dim=0, keepdim=True, unbiased=False).clamp_min(float(self.eps))
        generated_features = self._interaction_features(
            (generated_base - generated_mean) / generated_scale
        )
        target_features = self._interaction_features(
            (target_base - target_mean) / target_scale
        )
        value = rbf_mmd2(generated_features, target_features, bandwidths=tuple(self.bandwidths))
        return PathFunctionalDiscrepancyResult(
            value=value,
            metrics={
                f"{self.name}_value": float(value.detach().item()),
                f"{self.name}_feature_dim": float(generated_features.shape[1]),
                f"{self.name}_split_index": float(split_index),
                f"{self.name}_largest_window": float(largest_window),
                f"{self.name}_marginal_invariant": 1.0,
            },
        )


def _joint_prefix_future_volatility_features(
    paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    split_index: int,
    future_horizon: int,
    prefix_windows: tuple[int, ...],
    recent_return_window: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if paths.ndim != 3:
        raise ValueError("paths must have shape (B, N + 1, d)")
    if int(paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("paths time dimension must align with grid")
    if int(split_index) < 1:
        raise ValueError("split_index must be >= 1")
    if int(future_horizon) < 1:
        raise ValueError("future_horizon must be >= 1")
    if (
        len(prefix_windows) == 0
        or any(int(value) < 1 for value in prefix_windows)
        or tuple(sorted(set(int(value) for value in prefix_windows))) != prefix_windows
    ):
        raise ValueError("prefix_windows must be strictly increasing positive integers")
    if int(recent_return_window) < 1:
        raise ValueError("recent_return_window must be >= 1")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be a positive finite float")
    largest_window = max(prefix_windows)
    if split_index < largest_window or split_index < int(recent_return_window):
        raise ValueError("prefix windows must fit before the split")
    if split_index + int(future_horizon) > int(grid.num_steps):
        raise ValueError("future_horizon must fit after the split")

    returns = _returns(paths, grid=grid)
    scaled_returns = returns / math.sqrt(float(grid.dt))
    eps_squared = float(eps) ** 2
    prefix_log_rms = [
        torch.log(
            torch.sqrt(
                scaled_returns[:, split_index - window : split_index, :]
                .pow(2)
                .mean(dim=1)
                + eps_squared
            )
        )
        for window in prefix_windows
    ]
    recent_return = (
        paths[:, split_index, :]
        - paths[:, split_index - int(recent_return_window), :]
    ) / math.sqrt(float(recent_return_window) * float(grid.dt))
    prefix = paths[:, : split_index + 1, :]
    drawdown = (prefix.max(dim=1).values - prefix[:, -1, :]) / math.sqrt(
        float(split_index) * float(grid.dt)
    )
    future_returns = returns[:, split_index : split_index + int(future_horizon), :]
    future_realized_volatility = torch.sqrt(
        future_returns.pow(2).sum(dim=1) + eps_squared
    )
    future_log_realized_volatility = torch.log(future_realized_volatility)
    return (
        torch.cat(
            [*prefix_log_rms, recent_return, drawdown, future_log_realized_volatility],
            dim=1,
        ),
        future_realized_volatility,
    )


@dataclass(frozen=True)
class FixedTargetJointPrefixFutureVolatilityMMD:
    """Target-standardized joint MMD for prefix state and future volatility.

    Unlike :class:`MarginalInvariantConditionalVolatilityMMD`, generated-law
    moments are not removed.  A single set of location and scale statistics is
    fitted on the complete training target and then frozen.  The MMD therefore
    sees conditional spread and tail errors in future realized volatility as
    well as errors in its dependence on compact causal prefix features.
    """

    target_mean: torch.Tensor
    target_scale: torch.Tensor
    split_index: int
    future_horizon: int
    prefix_windows: Sequence[int] = (4, 8, 16)
    recent_return_window: int = 4
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0)
    eps: float = 1e-6
    name: str = "joint_prefix_future_volatility"

    def __post_init__(self) -> None:
        split_index = int(self.split_index)
        future_horizon = int(self.future_horizon)
        prefix_windows = tuple(int(value) for value in self.prefix_windows)
        recent_return_window = int(self.recent_return_window)
        bandwidths = tuple(float(value) for value in self.bandwidths)
        eps = float(self.eps)
        if split_index < 1:
            raise ValueError("split_index must be >= 1")
        if future_horizon < 1:
            raise ValueError("future_horizon must be >= 1")
        if (
            len(prefix_windows) == 0
            or any(value < 1 for value in prefix_windows)
            or tuple(sorted(set(prefix_windows))) != prefix_windows
        ):
            raise ValueError("prefix_windows must be strictly increasing positive integers")
        if recent_return_window < 1:
            raise ValueError("recent_return_window must be >= 1")
        if len(bandwidths) == 0 or any(
            not math.isfinite(value) or value <= 0.0 for value in bandwidths
        ):
            raise ValueError("bandwidths must be non-empty finite positive floats")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        if not isinstance(self.target_mean, torch.Tensor) or not isinstance(
            self.target_scale, torch.Tensor
        ):
            raise ValueError("target_mean and target_scale must be tensors")
        if self.target_mean.ndim != 2 or int(self.target_mean.shape[0]) != 1:
            raise ValueError("target_mean must have shape (1, feature_dim)")
        if tuple(self.target_scale.shape) != tuple(self.target_mean.shape):
            raise ValueError("target_scale must have the same shape as target_mean")
        if int(self.target_mean.shape[1]) < 1:
            raise ValueError("target feature dimension must be positive")
        if not torch.isfinite(self.target_mean).all() or not torch.isfinite(
            self.target_scale
        ).all():
            raise ValueError("target standardization statistics must be finite")
        if not torch.all(self.target_scale > 0.0):
            raise ValueError("target_scale must be strictly positive")
        object.__setattr__(self, "split_index", split_index)
        object.__setattr__(self, "future_horizon", future_horizon)
        object.__setattr__(self, "prefix_windows", prefix_windows)
        object.__setattr__(self, "recent_return_window", recent_return_window)
        object.__setattr__(self, "bandwidths", bandwidths)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "target_mean", self.target_mean.detach().cpu().clone())
        object.__setattr__(self, "target_scale", self.target_scale.detach().cpu().clone())

    def _features(
        self,
        paths: torch.Tensor,
        *,
        grid: DiscreteTimeGrid,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _joint_prefix_future_volatility_features(
            paths,
            grid=grid,
            split_index=int(self.split_index),
            future_horizon=int(self.future_horizon),
            prefix_windows=tuple(int(value) for value in self.prefix_windows),
            recent_return_window=int(self.recent_return_window),
            eps=float(self.eps),
        )

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        if int(generated_paths.shape[0]) < 2 or int(target_paths.shape[0]) < 2:
            raise ValueError("joint prefix/future MMD requires at least two paths per law")
        target_paths = target_paths.to(device=generated_paths.device, dtype=generated_paths.dtype)
        generated_features, generated_future_rv = self._features(generated_paths, grid=grid)
        target_features, target_future_rv = self._features(target_paths, grid=grid)
        if int(generated_features.shape[1]) != int(self.target_mean.shape[1]):
            raise ValueError("fitted target feature dimension does not match paths")
        target_mean = self.target_mean.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        target_scale = self.target_scale.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        generated_standardized = (generated_features - target_mean) / target_scale
        target_standardized = (target_features - target_mean) / target_scale
        value = rbf_mmd2(
            generated_standardized,
            target_standardized,
            bandwidths=tuple(self.bandwidths),
        )
        generated_rv_mean = generated_future_rv.mean()
        target_rv_mean = target_future_rv.mean()
        generated_rv_std = generated_future_rv.std(unbiased=False)
        target_rv_std = target_future_rv.std(unbiased=False)
        return PathFunctionalDiscrepancyResult(
            value=value,
            metrics={
                f"{self.name}_value": float(value.detach().item()),
                f"{self.name}_feature_dim": float(generated_features.shape[1]),
                f"{self.name}_split_index": float(self.split_index),
                f"{self.name}_future_horizon": float(self.future_horizon),
                f"{self.name}_target_standardization_fixed": 1.0,
                f"{self.name}_generated_future_rv_mean": float(
                    generated_rv_mean.detach().item()
                ),
                f"{self.name}_target_future_rv_mean": float(target_rv_mean.detach().item()),
                f"{self.name}_generated_future_rv_std": float(
                    generated_rv_std.detach().item()
                ),
                f"{self.name}_target_future_rv_std": float(target_rv_std.detach().item()),
                f"{self.name}_future_rv_mean_ratio": float(
                    (generated_rv_mean / target_rv_mean.clamp_min(float(self.eps)))
                    .detach()
                    .item()
                ),
                f"{self.name}_future_rv_std_ratio": float(
                    (generated_rv_std / target_rv_std.clamp_min(float(self.eps)))
                    .detach()
                    .item()
                ),
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
        query_features, _ = self._features(query_paths, grid=grid)
        population_features, _ = self._features(population_paths, grid=grid)
        target_features, _ = self._features(target_paths, grid=grid)
        target_mean = self.target_mean.to(
            device=query_paths.device,
            dtype=query_paths.dtype,
        )
        target_scale = self.target_scale.to(
            device=query_paths.device,
            dtype=query_paths.dtype,
        )
        return rbf_mmd_lions_potential(
            (query_features - target_mean) / target_scale,
            ((population_features - target_mean) / target_scale).detach(),
            ((target_features - target_mean) / target_scale).detach(),
            bandwidths=tuple(self.bandwidths),
        )


def fit_fixed_target_joint_prefix_future_volatility_mmd(
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    split_index: int,
    future_horizon: int,
    prefix_windows: Sequence[int] = (4, 8, 16),
    recent_return_window: int = 4,
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0),
    eps: float = 1e-6,
) -> FixedTargetJointPrefixFutureVolatilityMMD:
    """Fit immutable standardization statistics on the complete target law."""

    windows = tuple(int(value) for value in prefix_windows)
    features, _ = _joint_prefix_future_volatility_features(
        target_paths,
        grid=grid,
        split_index=int(split_index),
        future_horizon=int(future_horizon),
        prefix_windows=windows,
        recent_return_window=int(recent_return_window),
        eps=float(eps),
    )
    target_mean = features.mean(dim=0, keepdim=True).detach()
    target_scale = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(
        float(eps)
    ).detach()
    return FixedTargetJointPrefixFutureVolatilityMMD(
        target_mean=target_mean,
        target_scale=target_scale,
        split_index=int(split_index),
        future_horizon=int(future_horizon),
        prefix_windows=windows,
        recent_return_window=int(recent_return_window),
        bandwidths=tuple(float(value) for value in bandwidths),
        eps=float(eps),
    )


@dataclass(frozen=True)
class FixedTargetConditionalResidualVolatilityMMD:
    """MMD for the future-volatility innovation left by a causal predictor.

    A linear projection of standardized future log realized volatility on the
    generic causal prefix features is fit once on the complete training law
    and then frozen.  The discrepancy therefore targets conditional
    dispersion without imposing a volatility model or modifying the reference.
    """

    prefix_mean: torch.Tensor
    prefix_scale: torch.Tensor
    future_mean: torch.Tensor
    future_scale: torch.Tensor
    predictor_coefficients: torch.Tensor
    target_residual_scale: torch.Tensor
    target_standardized_residuals: torch.Tensor
    target_residual_first_moment: torch.Tensor
    target_residual_second_moment: torch.Tensor
    split_index: int
    future_horizon: int
    prefix_windows: Sequence[int] = (16, 32, 64)
    recent_return_window: int = 16
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0)
    moment_weight: float = 1.0
    eps: float = 1e-6
    name: str = "conditional_residual_future_volatility"

    def __post_init__(self) -> None:
        prefix_windows = tuple(int(value) for value in self.prefix_windows)
        bandwidths = tuple(float(value) for value in self.bandwidths)
        if int(self.split_index) < 1 or int(self.future_horizon) < 1:
            raise ValueError("split_index and future_horizon must be positive")
        if (
            len(prefix_windows) == 0
            or any(value < 1 for value in prefix_windows)
            or tuple(sorted(set(prefix_windows))) != prefix_windows
        ):
            raise ValueError("prefix_windows must be strictly increasing")
        if int(self.recent_return_window) < 1:
            raise ValueError("recent_return_window must be positive")
        if len(bandwidths) == 0 or any(
            not math.isfinite(value) or value <= 0.0 for value in bandwidths
        ):
            raise ValueError("bandwidths must be finite and positive")
        if not math.isfinite(float(self.eps)) or float(self.eps) <= 0.0:
            raise ValueError("eps must be positive and finite")
        if not math.isfinite(float(self.moment_weight)) or float(
            self.moment_weight
        ) < 0.0:
            raise ValueError("moment_weight must be finite and nonnegative")
        tensor_names = (
            "prefix_mean",
            "prefix_scale",
            "future_mean",
            "future_scale",
            "predictor_coefficients",
            "target_residual_scale",
            "target_standardized_residuals",
            "target_residual_first_moment",
            "target_residual_second_moment",
        )
        tensors = tuple(getattr(self, name) for name in tensor_names)
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise ValueError("conditional residual parameters must be tensors")
        if self.prefix_mean.ndim != 2 or int(self.prefix_mean.shape[0]) != 1:
            raise ValueError("prefix_mean must have shape (1, prefix_dim)")
        if tuple(self.prefix_scale.shape) != tuple(self.prefix_mean.shape):
            raise ValueError("prefix_scale must match prefix_mean")
        if self.future_mean.ndim != 2 or int(self.future_mean.shape[0]) != 1:
            raise ValueError("future_mean must have shape (1, state_dim)")
        if tuple(self.future_scale.shape) != tuple(self.future_mean.shape):
            raise ValueError("future_scale must match future_mean")
        if tuple(self.target_residual_scale.shape) != tuple(self.future_mean.shape):
            raise ValueError("target_residual_scale must match future_mean")
        expected_coefficients = (
            int(self.prefix_mean.shape[1]) + 1,
            int(self.future_mean.shape[1]),
        )
        if tuple(self.predictor_coefficients.shape) != expected_coefficients:
            raise ValueError("predictor coefficient shape is inconsistent")
        if self.target_standardized_residuals.ndim != 2 or int(
            self.target_standardized_residuals.shape[1]
        ) != int(self.future_mean.shape[1]):
            raise ValueError("target residuals must have shape (B, state_dim)")
        if int(self.target_standardized_residuals.shape[0]) < 2:
            raise ValueError("at least two target residuals are required")
        for moment_name in (
            "target_residual_first_moment",
            "target_residual_second_moment",
        ):
            if tuple(getattr(self, moment_name).shape) != tuple(self.future_mean.shape):
                raise ValueError("target residual moments must match future_mean")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("conditional residual parameters must be finite")
        if (
            bool((self.prefix_scale <= 0.0).any())
            or bool((self.future_scale <= 0.0).any())
            or bool((self.target_residual_scale <= 0.0).any())
        ):
            raise ValueError("conditional residual scales must be positive")
        object.__setattr__(self, "split_index", int(self.split_index))
        object.__setattr__(self, "future_horizon", int(self.future_horizon))
        object.__setattr__(self, "prefix_windows", prefix_windows)
        object.__setattr__(self, "recent_return_window", int(self.recent_return_window))
        object.__setattr__(self, "bandwidths", bandwidths)
        object.__setattr__(self, "moment_weight", float(self.moment_weight))
        object.__setattr__(self, "eps", float(self.eps))
        for field_name in tensor_names:
            object.__setattr__(
                self,
                field_name,
                getattr(self, field_name).detach().cpu().clone(),
            )

    def _standardized_residuals(
        self,
        paths: torch.Tensor,
        *,
        grid: DiscreteTimeGrid,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, _ = _joint_prefix_future_volatility_features(
            paths,
            grid=grid,
            split_index=int(self.split_index),
            future_horizon=int(self.future_horizon),
            prefix_windows=tuple(int(value) for value in self.prefix_windows),
            recent_return_window=int(self.recent_return_window),
            eps=float(self.eps),
        )
        state_dim = int(self.future_mean.shape[1])
        prefix = features[:, :-state_dim]
        future = features[:, -state_dim:]
        prefix_mean = self.prefix_mean.to(device=paths.device, dtype=paths.dtype)
        prefix_scale = self.prefix_scale.to(device=paths.device, dtype=paths.dtype)
        future_mean = self.future_mean.to(device=paths.device, dtype=paths.dtype)
        future_scale = self.future_scale.to(device=paths.device, dtype=paths.dtype)
        coefficients = self.predictor_coefficients.to(
            device=paths.device,
            dtype=paths.dtype,
        )
        residual_scale = self.target_residual_scale.to(
            device=paths.device,
            dtype=paths.dtype,
        )
        standardized_prefix = (prefix - prefix_mean) / prefix_scale
        design = torch.cat(
            (
                torch.ones(
                    (int(paths.shape[0]), 1),
                    device=paths.device,
                    dtype=paths.dtype,
                ),
                standardized_prefix,
            ),
            dim=1,
        )
        standardized_future = (future - future_mean) / future_scale
        raw_residual = standardized_future - design @ coefficients
        return raw_residual / residual_scale, raw_residual

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        del target_paths
        if int(generated_paths.shape[0]) < 2:
            raise ValueError("conditional residual MMD requires at least two paths")
        generated, raw_residual = self._standardized_residuals(
            generated_paths,
            grid=grid,
        )
        target = self.target_standardized_residuals.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        mmd_value = rbf_mmd2(
            generated,
            target,
            bandwidths=tuple(self.bandwidths),
        )
        target_first = self.target_residual_first_moment.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        ).squeeze(0)
        target_second = self.target_residual_second_moment.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        ).squeeze(0)
        first_error = generated.mean(dim=0) - target_first
        second_error = generated.pow(2).mean(dim=0) - target_second
        moment_value = (first_error.pow(2) + second_error.pow(2)).mean()
        value = mmd_value + float(self.moment_weight) * moment_value
        generated_scale = raw_residual.std(dim=0, unbiased=False)
        target_scale = self.target_residual_scale.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        ).squeeze(0)
        return PathFunctionalDiscrepancyResult(
            value=value,
            metrics={
                f"{self.name}_value": float(value.detach().item()),
                f"{self.name}_mmd_value": float(mmd_value.detach().item()),
                f"{self.name}_moment_value": float(moment_value.detach().item()),
                f"{self.name}_moment_weight": float(self.moment_weight),
                f"{self.name}_fixed_target": 1.0,
                f"{self.name}_residual_std_ratio": float(
                    (generated_scale / target_scale).mean().detach().item()
                ),
                f"{self.name}_split_index": float(self.split_index),
                f"{self.name}_future_horizon": float(self.future_horizon),
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
        del target_paths
        query, _ = self._standardized_residuals(query_paths, grid=grid)
        population, _ = self._standardized_residuals(population_paths, grid=grid)
        target = self.target_standardized_residuals.to(
            device=query_paths.device,
            dtype=query_paths.dtype,
        )
        mmd_potential = rbf_mmd_lions_potential(
            query,
            population.detach(),
            target.detach(),
            bandwidths=tuple(self.bandwidths),
        )
        population_detached = population.detach()
        target_first = self.target_residual_first_moment.to(
            device=query_paths.device,
            dtype=query_paths.dtype,
        )
        target_second = self.target_residual_second_moment.to(
            device=query_paths.device,
            dtype=query_paths.dtype,
        )
        first_error = population_detached.mean(dim=0, keepdim=True) - target_first
        second_error = (
            population_detached.pow(2).mean(dim=0, keepdim=True) - target_second
        )
        moment_potential = (
            2.0
            * (
                first_error * query
                + second_error * query.pow(2)
            ).mean(dim=1)
        ).mean()
        return mmd_potential + float(self.moment_weight) * moment_potential


def fit_fixed_target_conditional_residual_volatility_mmd(
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    split_index: int,
    future_horizon: int,
    prefix_windows: Sequence[int] = (16, 32, 64),
    recent_return_window: int = 16,
    ridge: float = 1e-2,
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0),
    moment_weight: float = 1.0,
    eps: float = 1e-6,
) -> FixedTargetConditionalResidualVolatilityMMD:
    """Fit the immutable target-law conditional residual transform."""

    if not math.isfinite(float(ridge)) or float(ridge) <= 0.0:
        raise ValueError("ridge must be positive and finite")
    windows = tuple(int(value) for value in prefix_windows)
    features, _ = _joint_prefix_future_volatility_features(
        target_paths,
        grid=grid,
        split_index=int(split_index),
        future_horizon=int(future_horizon),
        prefix_windows=windows,
        recent_return_window=int(recent_return_window),
        eps=float(eps),
    )
    state_dim = int(target_paths.shape[2])
    prefix = features[:, :-state_dim]
    future = features[:, -state_dim:]
    prefix_mean = prefix.mean(dim=0, keepdim=True)
    prefix_scale = prefix.std(dim=0, keepdim=True, unbiased=False).clamp_min(
        float(eps)
    )
    future_mean = future.mean(dim=0, keepdim=True)
    future_scale = future.std(dim=0, keepdim=True, unbiased=False).clamp_min(
        float(eps)
    )
    standardized_prefix = (prefix - prefix_mean) / prefix_scale
    standardized_future = (future - future_mean) / future_scale
    design = torch.cat(
        (
            torch.ones(
                (int(target_paths.shape[0]), 1),
                device=target_paths.device,
                dtype=target_paths.dtype,
            ),
            standardized_prefix,
        ),
        dim=1,
    )
    penalty = torch.eye(
        int(design.shape[1]),
        device=target_paths.device,
        dtype=target_paths.dtype,
    )
    penalty[0, 0] = 0.0
    coefficients = torch.linalg.solve(
        design.transpose(0, 1) @ design + float(ridge) * penalty,
        design.transpose(0, 1) @ standardized_future,
    )
    residual = standardized_future - design @ coefficients
    residual_scale = residual.std(dim=0, keepdim=True, unbiased=False).clamp_min(
        float(eps)
    )
    standardized_residual = residual / residual_scale
    return FixedTargetConditionalResidualVolatilityMMD(
        prefix_mean=prefix_mean,
        prefix_scale=prefix_scale,
        future_mean=future_mean,
        future_scale=future_scale,
        predictor_coefficients=coefficients,
        target_residual_scale=residual_scale,
        target_standardized_residuals=standardized_residual,
        target_residual_first_moment=standardized_residual.mean(
            dim=0,
            keepdim=True,
        ),
        target_residual_second_moment=standardized_residual.pow(2).mean(
            dim=0,
            keepdim=True,
        ),
        split_index=int(split_index),
        future_horizon=int(future_horizon),
        prefix_windows=windows,
        recent_return_window=int(recent_return_window),
        bandwidths=tuple(float(value) for value in bandwidths),
        moment_weight=float(moment_weight),
        eps=float(eps),
    )


def _prefix_future_log_volatility(
    paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    split_index: int,
    prefix_window: int,
    future_horizon: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if paths.ndim != 3 or int(paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("paths must align with the discrepancy grid")
    if int(prefix_window) < 1 or int(prefix_window) > int(split_index):
        raise ValueError("prefix_window must fit before split_index")
    if int(future_horizon) < 1 or int(split_index) + int(future_horizon) > int(
        grid.num_steps
    ):
        raise ValueError("future_horizon must fit after split_index")
    scaled_returns = _returns(paths, grid=grid) / math.sqrt(float(grid.dt))
    prefix = scaled_returns[:, split_index - int(prefix_window) : split_index, :]
    future = scaled_returns[:, split_index : split_index + int(future_horizon), :]
    eps_squared = float(eps) ** 2
    prefix_log_rms = torch.log(torch.sqrt(prefix.pow(2).mean(dim=1) + eps_squared))
    future_log_rv = torch.log(torch.sqrt(future.pow(2).sum(dim=1) + eps_squared))
    return prefix_log_rms, future_log_rv


def _soft_prefix_regime_claims(
    prefix_log_rms: torch.Tensor,
    future_log_rv: torch.Tensor,
    *,
    prefix_thresholds: torch.Tensor,
    gate_temperature: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    future_quantiles: torch.Tensor,
    tail_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    thresholds = prefix_thresholds.to(
        device=prefix_log_rms.device,
        dtype=prefix_log_rms.dtype,
    )
    temperature = gate_temperature.to(
        device=prefix_log_rms.device,
        dtype=prefix_log_rms.dtype,
    ).clamp_min(torch.finfo(prefix_log_rms.dtype).eps)
    lower_switch = torch.sigmoid((prefix_log_rms - thresholds[0]) / temperature)
    upper_switch = torch.sigmoid((prefix_log_rms - thresholds[1]) / temperature)
    gates = torch.stack(
        (1.0 - lower_switch, lower_switch - upper_switch, upper_switch),
        dim=1,
    )
    mean = future_mean.to(device=future_log_rv.device, dtype=future_log_rv.dtype)
    scale = future_scale.to(device=future_log_rv.device, dtype=future_log_rv.dtype)
    standardized_future = (future_log_rv - mean) / scale
    future_by_regime = standardized_future[:, None, :]
    quantiles = future_quantiles.to(
        device=future_log_rv.device,
        dtype=future_log_rv.dtype,
    )
    tail_temperature = float(tail_temperature)
    lower_cdf = torch.sigmoid(
        (quantiles[None, :, 0, :] - future_by_regime) / tail_temperature
    )
    median_cdf = torch.sigmoid(
        (quantiles[None, :, 1, :] - future_by_regime) / tail_temperature
    )
    upper_survival = torch.sigmoid(
        (future_by_regime - quantiles[None, :, 2, :]) / tail_temperature
    )
    claim_values = torch.stack(
        (
            torch.ones_like(future_by_regime).expand_as(gates),
            future_by_regime.expand_as(gates),
            future_by_regime.pow(2).expand_as(gates),
            lower_cdf,
            median_cdf,
            upper_survival,
        ),
        dim=-1,
    )
    weighted_claims = gates.unsqueeze(-1) * claim_values
    return weighted_claims.reshape(int(prefix_log_rms.shape[0]), -1), gates, standardized_future


@dataclass(frozen=True)
class FixedTargetPrefixRegimeFutureVolatilityClaims:
    """Smooth prefix-regime conditional future-volatility moment claims.

    Target prefix-volatility tertiles define three soft regimes. Within each
    regime the claims match its probability mass, standardized future log-RV
    first and second moments, and smooth lower/median/upper tail probabilities.
    They are ordinary path-dependent functionals of the observed prefix and
    future continuation; no latent-volatility or parametric reference model is
    introduced.
    """

    target_prefix_thresholds: torch.Tensor
    target_gate_temperature: torch.Tensor
    target_future_mean: torch.Tensor
    target_future_scale: torch.Tensor
    target_future_quantiles: torch.Tensor
    target_claim_mean: torch.Tensor
    target_claim_scale: torch.Tensor
    target_regime_future_mean: torch.Tensor
    target_regime_future_std: torch.Tensor
    split_index: int
    prefix_window: int
    future_horizon: int
    claim_weight: float = 1.0
    conditional_moment_weight: float = 0.0
    conditional_mean_weight: float = 1.0
    conditional_std_weight: float = 1.0
    claim_component_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    future_quantile_levels: Sequence[float] = (0.10, 0.50, 0.90)
    regime_weights: Sequence[float] = (1.0, 1.0, 1.0)
    tail_temperature: float = 0.15
    eps: float = 1e-6
    name: str = "prefix_regime_future_volatility_claims"

    def __post_init__(self) -> None:
        split_index = int(self.split_index)
        prefix_window = int(self.prefix_window)
        future_horizon = int(self.future_horizon)
        claim_weight = float(self.claim_weight)
        conditional_moment_weight = float(self.conditional_moment_weight)
        conditional_mean_weight = float(self.conditional_mean_weight)
        conditional_std_weight = float(self.conditional_std_weight)
        claim_component_weights = tuple(
            float(value) for value in self.claim_component_weights
        )
        future_quantile_levels = tuple(
            float(value) for value in self.future_quantile_levels
        )
        regime_weights = tuple(float(value) for value in self.regime_weights)
        tail_temperature = float(self.tail_temperature)
        eps = float(self.eps)
        if split_index < 1:
            raise ValueError("split_index must be >= 1")
        if prefix_window < 1 or prefix_window > split_index:
            raise ValueError("prefix_window must fit before split_index")
        if future_horizon < 1:
            raise ValueError("future_horizon must be >= 1")
        if not math.isfinite(claim_weight) or claim_weight < 0.0:
            raise ValueError("claim_weight must be nonnegative and finite")
        if not math.isfinite(conditional_moment_weight) or conditional_moment_weight < 0.0:
            raise ValueError("conditional_moment_weight must be nonnegative and finite")
        if not math.isfinite(conditional_mean_weight) or conditional_mean_weight < 0.0:
            raise ValueError("conditional_mean_weight must be nonnegative and finite")
        if not math.isfinite(conditional_std_weight) or conditional_std_weight < 0.0:
            raise ValueError("conditional_std_weight must be nonnegative and finite")
        if conditional_mean_weight + conditional_std_weight <= 0.0:
            raise ValueError(
                "conditional mean and std weights cannot both be zero"
            )
        if (
            len(claim_component_weights) != 6
            or any(
                not math.isfinite(value) or value < 0.0
                for value in claim_component_weights
            )
            or sum(claim_component_weights) <= 0.0
        ):
            raise ValueError(
                "claim_component_weights must contain six nonnegative finite "
                "values with positive sum"
            )
        if (
            len(future_quantile_levels) != 3
            or any(
                not math.isfinite(value) or not 0.0 < value < 1.0
                for value in future_quantile_levels
            )
            or not (
                future_quantile_levels[0]
                < future_quantile_levels[1]
                < future_quantile_levels[2]
            )
        ):
            raise ValueError(
                "future_quantile_levels must contain three increasing probabilities"
            )
        if claim_weight == 0.0 and conditional_moment_weight == 0.0:
            raise ValueError("claim_weight and conditional_moment_weight cannot both be zero")
        if (
            len(regime_weights) != 3
            or any(not math.isfinite(value) or value < 0.0 for value in regime_weights)
            or sum(regime_weights) <= 0.0
        ):
            raise ValueError(
                "regime_weights must contain three nonnegative finite values "
                "with positive sum"
            )
        if not math.isfinite(tail_temperature) or tail_temperature <= 0.0:
            raise ValueError("tail_temperature must be positive and finite")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")
        tensors = (
            self.target_prefix_thresholds,
            self.target_gate_temperature,
            self.target_future_mean,
            self.target_future_scale,
            self.target_future_quantiles,
            self.target_claim_mean,
            self.target_claim_scale,
            self.target_regime_future_mean,
            self.target_regime_future_std,
        )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise ValueError("all fitted target quantities must be tensors")
        state_dim = int(self.target_prefix_thresholds.shape[-1])
        if tuple(self.target_prefix_thresholds.shape) != (2, state_dim):
            raise ValueError("target_prefix_thresholds must have shape (2, d)")
        if tuple(self.target_gate_temperature.shape) != (state_dim,):
            raise ValueError("target_gate_temperature must have shape (d,)")
        if tuple(self.target_future_mean.shape) != (1, state_dim):
            raise ValueError("target_future_mean must have shape (1, d)")
        if tuple(self.target_future_scale.shape) != (1, state_dim):
            raise ValueError("target_future_scale must have shape (1, d)")
        if tuple(self.target_future_quantiles.shape) != (3, 3, state_dim):
            raise ValueError("target_future_quantiles must have shape (3, 3, d)")
        feature_dim = 3 * state_dim * 6
        if tuple(self.target_claim_mean.shape) != (1, feature_dim):
            raise ValueError("target_claim_mean has the wrong feature dimension")
        if tuple(self.target_claim_scale.shape) != (1, feature_dim):
            raise ValueError("target_claim_scale has the wrong feature dimension")
        if tuple(self.target_regime_future_mean.shape) != (3, state_dim):
            raise ValueError("target_regime_future_mean must have shape (3, d)")
        if tuple(self.target_regime_future_std.shape) != (3, state_dim):
            raise ValueError("target_regime_future_std must have shape (3, d)")
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("fitted target quantities must be finite")
        if not torch.all(self.target_gate_temperature > 0.0):
            raise ValueError("target_gate_temperature must be strictly positive")
        if not torch.all(self.target_future_scale > 0.0):
            raise ValueError("target_future_scale must be strictly positive")
        if not torch.all(self.target_claim_scale > 0.0):
            raise ValueError("target_claim_scale must be strictly positive")
        object.__setattr__(self, "split_index", split_index)
        object.__setattr__(self, "prefix_window", prefix_window)
        object.__setattr__(self, "future_horizon", future_horizon)
        object.__setattr__(self, "claim_weight", claim_weight)
        object.__setattr__(self, "conditional_moment_weight", conditional_moment_weight)
        object.__setattr__(self, "conditional_mean_weight", conditional_mean_weight)
        object.__setattr__(self, "conditional_std_weight", conditional_std_weight)
        object.__setattr__(self, "claim_component_weights", claim_component_weights)
        object.__setattr__(self, "future_quantile_levels", future_quantile_levels)
        object.__setattr__(self, "regime_weights", regime_weights)
        object.__setattr__(self, "tail_temperature", tail_temperature)
        object.__setattr__(self, "eps", eps)
        for field_name in (
            "target_prefix_thresholds",
            "target_gate_temperature",
            "target_future_mean",
            "target_future_scale",
            "target_future_quantiles",
            "target_claim_mean",
            "target_claim_scale",
            "target_regime_future_mean",
            "target_regime_future_std",
        ):
            object.__setattr__(self, field_name, getattr(self, field_name).detach().cpu().clone())

    def _claims(
        self,
        paths: torch.Tensor,
        *,
        grid: DiscreteTimeGrid,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix, future = _prefix_future_log_volatility(
            paths,
            grid=grid,
            split_index=int(self.split_index),
            prefix_window=int(self.prefix_window),
            future_horizon=int(self.future_horizon),
            eps=float(self.eps),
        )
        return _soft_prefix_regime_claims(
            prefix,
            future,
            prefix_thresholds=self.target_prefix_thresholds,
            gate_temperature=self.target_gate_temperature,
            future_mean=self.target_future_mean,
            future_scale=self.target_future_scale,
            future_quantiles=self.target_future_quantiles,
            tail_temperature=float(self.tail_temperature),
        )

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        del target_paths
        generated_claims, gates, standardized_future = self._claims(
            generated_paths,
            grid=grid,
        )
        target_mean = self.target_claim_mean.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        target_scale = self.target_claim_scale.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        state_dim = int(self.target_future_mean.shape[1])
        regime_weights = generated_paths.new_tensor(self.regime_weights)
        claim_component_weights = generated_paths.new_tensor(
            self.claim_component_weights
        )
        standardized_errors = (
            (generated_claims.mean(dim=0, keepdim=True) - target_mean)
            / target_scale
        ).reshape(3, state_dim, 6)
        claim_value = (
            standardized_errors.pow(2)
            * regime_weights[:, None, None]
            * claim_component_weights[None, None, :]
        ).sum() / (
            regime_weights.sum()
            * float(state_dim)
            * claim_component_weights.sum()
        )
        mass = gates.mean(dim=0)
        generated_regime_mean = (
            (gates * standardized_future[:, None, :]).mean(dim=0)
            / mass.clamp_min(float(self.eps))
        )
        generated_regime_second = (
            (gates * standardized_future[:, None, :].pow(2)).mean(dim=0)
            / mass.clamp_min(float(self.eps))
        )
        generated_regime_std = torch.sqrt(
            (generated_regime_second - generated_regime_mean.pow(2)).clamp_min(0.0)
        )
        target_regime_mean = self.target_regime_future_mean.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        target_regime_std = self.target_regime_future_std.to(
            device=generated_paths.device,
            dtype=generated_paths.dtype,
        )
        conditional_weight_sum = float(
            self.conditional_mean_weight + self.conditional_std_weight
        )
        conditional_moment_value = (
            (
                float(self.conditional_mean_weight)
                * (generated_regime_mean - target_regime_mean).pow(2)
                + float(self.conditional_std_weight)
                * (generated_regime_std - target_regime_std).pow(2)
            )
            * regime_weights[:, None]
        ).sum() / (
            regime_weights.sum() * float(state_dim) * conditional_weight_sum
        )
        value = (
            float(self.claim_weight) * claim_value
            + float(self.conditional_moment_weight) * conditional_moment_value
        )
        metrics: dict[str, float] = {
            f"{self.name}_value": float(value.detach().item()),
            f"{self.name}_claim_value": float(claim_value.detach().item()),
            f"{self.name}_claim_weight": float(self.claim_weight),
            f"{self.name}_conditional_moment_value": float(
                conditional_moment_value.detach().item()
            ),
            f"{self.name}_conditional_moment_weight": float(
                self.conditional_moment_weight
            ),
            f"{self.name}_conditional_mean_weight": float(
                self.conditional_mean_weight
            ),
            f"{self.name}_conditional_std_weight": float(
                self.conditional_std_weight
            ),
            f"{self.name}_feature_dim": float(generated_claims.shape[1]),
            f"{self.name}_split_index": float(self.split_index),
            f"{self.name}_prefix_window": float(self.prefix_window),
            f"{self.name}_future_horizon": float(self.future_horizon),
            f"{self.name}_standardized_claim_rmse": float(
                torch.sqrt(value.detach()).item()
            ),
        }
        regime_names = ("low", "middle", "high")
        for regime_number, regime_name in enumerate(regime_names):
            metrics[f"{self.name}_{regime_name}_weight"] = float(
                self.regime_weights[regime_number]
            )
            metrics[f"{self.name}_{regime_name}_mass_mean"] = float(
                mass[regime_number].mean().detach().item()
            )
            metrics[f"{self.name}_{regime_name}_future_z_mean"] = float(
                generated_regime_mean[regime_number].mean().detach().item()
            )
            metrics[f"{self.name}_{regime_name}_future_z_std"] = float(
                generated_regime_std[regime_number].mean().detach().item()
            )
            metrics[f"{self.name}_{regime_name}_target_future_z_mean"] = float(
                target_regime_mean[regime_number].mean().detach().item()
            )
            metrics[f"{self.name}_{regime_name}_target_future_z_std"] = float(
                target_regime_std[regime_number].mean().detach().item()
            )
        return PathFunctionalDiscrepancyResult(value=value, metrics=metrics)

    @staticmethod
    def _law_statistics(
        claims: torch.Tensor,
        gates: torch.Tensor,
        standardized_future: torch.Tensor,
    ) -> torch.Tensor:
        future_by_regime = standardized_future[:, None, :]
        return torch.cat(
            (
                claims,
                gates.reshape(int(gates.shape[0]), -1),
                (gates * future_by_regime).reshape(int(gates.shape[0]), -1),
                (gates * future_by_regime.pow(2)).reshape(
                    int(gates.shape[0]), -1
                ),
            ),
            dim=1,
        )

    def _value_from_mean_statistics(
        self,
        mean_statistics: torch.Tensor,
    ) -> torch.Tensor:
        state_dim = int(self.target_future_mean.shape[1])
        claim_dim = int(self.target_claim_mean.shape[1])
        regime_dim = 3 * state_dim
        claim_mean = mean_statistics[:claim_dim].reshape(1, claim_dim)
        offset = claim_dim
        mass = mean_statistics[offset : offset + regime_dim].reshape(
            3, state_dim
        )
        offset += regime_dim
        weighted_first = mean_statistics[offset : offset + regime_dim].reshape(
            3, state_dim
        )
        offset += regime_dim
        weighted_second = mean_statistics[offset : offset + regime_dim].reshape(
            3, state_dim
        )
        target_claim_mean = self.target_claim_mean.to(
            device=mean_statistics.device,
            dtype=mean_statistics.dtype,
        )
        target_claim_scale = self.target_claim_scale.to(
            device=mean_statistics.device,
            dtype=mean_statistics.dtype,
        )
        regime_weights = mean_statistics.new_tensor(self.regime_weights)
        claim_component_weights = mean_statistics.new_tensor(
            self.claim_component_weights
        )
        claim_errors = (
            (claim_mean - target_claim_mean) / target_claim_scale
        ).reshape(3, state_dim, 6)
        claim_value = (
            claim_errors.pow(2)
            * regime_weights[:, None, None]
            * claim_component_weights[None, None, :]
        ).sum() / (
            regime_weights.sum()
            * float(state_dim)
            * claim_component_weights.sum()
        )
        regime_mean = weighted_first / mass.clamp_min(float(self.eps))
        regime_second = weighted_second / mass.clamp_min(float(self.eps))
        regime_std = torch.sqrt(
            (regime_second - regime_mean.pow(2)).clamp_min(0.0)
        )
        target_regime_mean = self.target_regime_future_mean.to(
            device=mean_statistics.device,
            dtype=mean_statistics.dtype,
        )
        target_regime_std = self.target_regime_future_std.to(
            device=mean_statistics.device,
            dtype=mean_statistics.dtype,
        )
        conditional_weight_sum = float(
            self.conditional_mean_weight + self.conditional_std_weight
        )
        conditional_moment_value = (
            (
                float(self.conditional_mean_weight)
                * (regime_mean - target_regime_mean).pow(2)
                + float(self.conditional_std_weight)
                * (regime_std - target_regime_std).pow(2)
            )
            * regime_weights[:, None]
        ).sum() / (
            regime_weights.sum() * float(state_dim) * conditional_weight_sum
        )
        return (
            float(self.claim_weight) * claim_value
            + float(self.conditional_moment_weight) * conditional_moment_value
        )

    def lions_potential(
        self,
        *,
        query_paths: torch.Tensor,
        population_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> torch.Tensor:
        """Influence potential with the unconditional law held fixed.

        This functional depends on the generated law only through expectations
        of smooth claim, gate, gate-times-future, and gate-times-future-squared
        statistics.  The Lions potential is their query value dotted with the
        derivative of the scalar functional with respect to those population
        expectations.
        """

        del target_paths
        query_claims, query_gates, query_future = self._claims(
            query_paths,
            grid=grid,
        )
        with torch.no_grad():
            population_claims, population_gates, population_future = (
                self._claims(population_paths, grid=grid)
            )
            population_mean = self._law_statistics(
                population_claims,
                population_gates,
                population_future,
            ).mean(dim=0)
        population_mean_req = population_mean.detach().clone().requires_grad_(
            True
        )
        population_value = self._value_from_mean_statistics(
            population_mean_req
        )
        influence_coefficients = torch.autograd.grad(
            population_value,
            population_mean_req,
        )[0].detach()
        query_statistics = self._law_statistics(
            query_claims,
            query_gates,
            query_future,
        )
        return (query_statistics * influence_coefficients).sum(dim=1).mean()


def fit_fixed_target_prefix_regime_future_volatility_claims(
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    split_index: int,
    prefix_window: int,
    future_horizon: int,
    gate_temperature_fraction: float = 0.25,
    tail_temperature: float = 0.15,
    conditional_moment_weight: float = 0.0,
    conditional_mean_weight: float = 1.0,
    conditional_std_weight: float = 1.0,
    claim_component_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    future_quantile_levels: Sequence[float] = (0.10, 0.50, 0.90),
    claim_weight: float = 1.0,
    regime_weights: Sequence[float] = (1.0, 1.0, 1.0),
    eps: float = 1e-6,
) -> FixedTargetPrefixRegimeFutureVolatilityClaims:
    """Fit immutable empirical regime thresholds and conditional RV claims."""

    gate_temperature_fraction = float(gate_temperature_fraction)
    if not math.isfinite(gate_temperature_fraction) or gate_temperature_fraction <= 0.0:
        raise ValueError("gate_temperature_fraction must be positive and finite")
    prefix, future = _prefix_future_log_volatility(
        target_paths,
        grid=grid,
        split_index=int(split_index),
        prefix_window=int(prefix_window),
        future_horizon=int(future_horizon),
        eps=float(eps),
    )
    thresholds = torch.quantile(
        prefix,
        torch.tensor((1.0 / 3.0, 2.0 / 3.0), device=prefix.device, dtype=prefix.dtype),
        dim=0,
    ).detach()
    gate_temperature = (
        gate_temperature_fraction * (thresholds[1] - thresholds[0]).abs()
    ).clamp_min(float(eps)).detach()
    future_mean = future.mean(dim=0, keepdim=True).detach()
    future_scale = future.std(dim=0, keepdim=True, unbiased=False).clamp_min(float(eps)).detach()
    standardized_future = (future - future_mean) / future_scale
    state_dim = int(target_paths.shape[-1])
    hard_regimes = torch.stack(
        [
            torch.bucketize(
                prefix[:, asset].contiguous(),
                thresholds[:, asset].contiguous(),
            )
            for asset in range(state_dim)
        ],
        dim=1,
    )
    quantile_levels = torch.tensor(
        tuple(float(value) for value in future_quantile_levels),
        device=target_paths.device,
        dtype=target_paths.dtype,
    )
    regime_quantiles = target_paths.new_empty((3, 3, state_dim))
    for regime_number in range(3):
        for asset in range(state_dim):
            selected = standardized_future[:, asset][hard_regimes[:, asset] == regime_number]
            if int(selected.numel()) < 2:
                raise ValueError("every target prefix regime must contain at least two paths")
            regime_quantiles[regime_number, :, asset] = torch.quantile(
                selected,
                quantile_levels,
            )
    claims, gates, standardized_future = _soft_prefix_regime_claims(
        prefix,
        future,
        prefix_thresholds=thresholds,
        gate_temperature=gate_temperature,
        future_mean=future_mean,
        future_scale=future_scale,
        future_quantiles=regime_quantiles,
        tail_temperature=float(tail_temperature),
    )
    regime_mass = gates.mean(dim=0)
    regime_mean = (
        (gates * standardized_future[:, None, :]).mean(dim=0)
        / regime_mass.clamp_min(float(eps))
    )
    regime_second = (
        (gates * standardized_future[:, None, :].pow(2)).mean(dim=0)
        / regime_mass.clamp_min(float(eps))
    )
    regime_std = torch.sqrt((regime_second - regime_mean.pow(2)).clamp_min(0.0))
    return FixedTargetPrefixRegimeFutureVolatilityClaims(
        target_prefix_thresholds=thresholds,
        target_gate_temperature=gate_temperature,
        target_future_mean=future_mean,
        target_future_scale=future_scale,
        target_future_quantiles=regime_quantiles,
        target_claim_mean=claims.mean(dim=0, keepdim=True).detach(),
        target_claim_scale=claims.std(dim=0, keepdim=True, unbiased=False)
        .clamp_min(float(eps))
        .detach(),
        target_regime_future_mean=regime_mean.detach(),
        target_regime_future_std=regime_std.detach(),
        split_index=int(split_index),
        prefix_window=int(prefix_window),
        future_horizon=int(future_horizon),
        conditional_moment_weight=float(conditional_moment_weight),
        conditional_mean_weight=float(conditional_mean_weight),
        conditional_std_weight=float(conditional_std_weight),
        claim_component_weights=tuple(
            float(value) for value in claim_component_weights
        ),
        future_quantile_levels=tuple(
            float(value) for value in future_quantile_levels
        ),
        claim_weight=float(claim_weight),
        regime_weights=tuple(float(value) for value in regime_weights),
        tail_temperature=float(tail_temperature),
        eps=float(eps),
    )


@dataclass(frozen=True)
class FixedTargetPrefixFutureVolatilityKernelMoments:
    """Continuous train-fitted prefix/future log-volatility moments."""

    prefix_mean: torch.Tensor
    prefix_scale: torch.Tensor
    kernel_centers: torch.Tensor
    kernel_bandwidth: torch.Tensor
    future_mean: torch.Tensor
    future_scale: torch.Tensor
    target_conditional_mean: torch.Tensor
    target_conditional_std: torch.Tensor
    split_index: int
    prefix_window: int
    future_horizon: int
    mean_weight: float = 1.0
    std_weight: float = 1.0
    eps: float = 1e-6
    name: str = "prefix_future_volatility_kernel_moments"

    def __post_init__(self) -> None:
        tensors = (
            self.prefix_mean,
            self.prefix_scale,
            self.kernel_centers,
            self.kernel_bandwidth,
            self.future_mean,
            self.future_scale,
            self.target_conditional_mean,
            self.target_conditional_std,
        )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise ValueError("kernel-moment fitted quantities must be tensors")
        kernel_count, state_dim = self.kernel_centers.shape
        if kernel_count < 2 or tuple(self.prefix_mean.shape) != (1, state_dim):
            raise ValueError("kernel moments require shape-compatible centers")
        expected_shapes = (
            (self.prefix_scale, (1, state_dim)),
            (self.kernel_bandwidth, (1, state_dim)),
            (self.future_mean, (1, state_dim)),
            (self.future_scale, (1, state_dim)),
            (self.target_conditional_mean, (kernel_count, state_dim)),
            (self.target_conditional_std, (kernel_count, state_dim)),
        )
        if any(tuple(value.shape) != shape for value, shape in expected_shapes):
            raise ValueError("kernel-moment fitted tensor shape is invalid")
        if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
            raise ValueError("kernel-moment fitted quantities must be finite")
        if any(
            bool((value <= 0).any().item())
            for value in (self.prefix_scale, self.kernel_bandwidth, self.future_scale)
        ):
            raise ValueError("kernel-moment scales must be positive")
        mean_weight, std_weight = float(self.mean_weight), float(self.std_weight)
        if min(mean_weight, std_weight) < 0.0 or mean_weight + std_weight <= 0.0:
            raise ValueError("kernel mean/std weights must be nonnegative and active")
        for field_name in (
            "prefix_mean", "prefix_scale", "kernel_centers", "kernel_bandwidth",
            "future_mean", "future_scale", "target_conditional_mean",
            "target_conditional_std",
        ):
            object.__setattr__(self, field_name, getattr(self, field_name).detach().cpu().clone())
        object.__setattr__(self, "mean_weight", mean_weight)
        object.__setattr__(self, "std_weight", std_weight)

    def _path_statistics(self, paths: torch.Tensor, grid: DiscreteTimeGrid) -> torch.Tensor:
        prefix, future = _prefix_future_log_volatility(
            paths, grid=grid, split_index=int(self.split_index),
            prefix_window=int(self.prefix_window), future_horizon=int(self.future_horizon),
            eps=float(self.eps),
        )
        prefix_z = (prefix - self.prefix_mean.to(prefix)) / self.prefix_scale.to(prefix)
        future_z = (future - self.future_mean.to(future)) / self.future_scale.to(future)
        distance = (prefix_z[:, None, :] - self.kernel_centers.to(prefix)[None]) / self.kernel_bandwidth.to(prefix)[None]
        gates = torch.softmax(-0.5 * distance.pow(2), dim=1)
        expanded = future_z[:, None, :]
        return torch.cat((gates, gates * expanded, gates * expanded.pow(2)), dim=1).reshape(int(paths.shape[0]), -1)

    def _value(self, statistics: torch.Tensor) -> torch.Tensor:
        kernel_count, state_dim = self.target_conditional_mean.shape
        block = int(kernel_count * state_dim)
        mass = statistics[:block].reshape(kernel_count, state_dim).clamp_min(float(self.eps))
        mean = statistics[block : 2 * block].reshape(kernel_count, state_dim) / mass
        second = statistics[2 * block :].reshape(kernel_count, state_dim) / mass
        std = torch.sqrt((second - mean.pow(2)).clamp_min(0.0))
        error = (
            float(self.mean_weight) * (mean - self.target_conditional_mean.to(statistics)).pow(2)
            + float(self.std_weight) * (std - self.target_conditional_std.to(statistics)).pow(2)
        )
        return error.mean() / float(self.mean_weight + self.std_weight)

    def __call__(self, *, generated_paths: torch.Tensor, target_paths: torch.Tensor, grid: DiscreteTimeGrid) -> PathFunctionalDiscrepancyResult:
        del target_paths
        value = self._value(self._path_statistics(generated_paths, grid).mean(dim=0))
        return PathFunctionalDiscrepancyResult(value=value, metrics={f"{self.name}_value": float(value.detach()), f"{self.name}_kernel_count": float(self.kernel_centers.shape[0])})

    def lions_potential(self, *, query_paths: torch.Tensor, population_paths: torch.Tensor, target_paths: torch.Tensor, grid: DiscreteTimeGrid) -> torch.Tensor:
        del target_paths
        query = self._path_statistics(query_paths, grid)
        with torch.no_grad():
            population_mean = self._path_statistics(population_paths, grid).mean(dim=0)
        population_req = population_mean.detach().clone().requires_grad_(True)
        coefficients = torch.autograd.grad(self._value(population_req), population_req)[0].detach()
        return (query * coefficients).sum(dim=1).mean()


def fit_fixed_target_prefix_future_volatility_kernel_moments(
    target_paths: torch.Tensor, *, grid: DiscreteTimeGrid, split_index: int,
    prefix_window: int, future_horizon: int,
    kernel_quantiles: Sequence[float] = (0.10, 0.30, 0.50, 0.70, 0.90),
    bandwidth_multiplier: float = 1.0, mean_weight: float = 1.0,
    std_weight: float = 1.0, eps: float = 1e-6,
) -> FixedTargetPrefixFutureVolatilityKernelMoments:
    prefix, future = _prefix_future_log_volatility(
        target_paths, grid=grid, split_index=int(split_index),
        prefix_window=int(prefix_window), future_horizon=int(future_horizon), eps=float(eps),
    )
    levels = tuple(float(value) for value in kernel_quantiles)
    if len(levels) < 2 or any(not 0.0 < value < 1.0 for value in levels):
        raise ValueError("kernel quantiles must lie in (0, 1)")
    prefix_mean = prefix.mean(0, keepdim=True).detach()
    prefix_scale = prefix.std(0, keepdim=True, unbiased=False).clamp_min(float(eps)).detach()
    prefix_z = (prefix - prefix_mean) / prefix_scale
    centers = torch.quantile(prefix_z, torch.tensor(levels, device=prefix.device, dtype=prefix.dtype), dim=0).detach()
    bandwidth = (float(bandwidth_multiplier) * (centers[1:] - centers[:-1]).abs().median(0).values).clamp_min(float(eps)).reshape(1, -1).detach()
    future_mean = future.mean(0, keepdim=True).detach()
    future_scale = future.std(0, keepdim=True, unbiased=False).clamp_min(float(eps)).detach()
    future_z = (future - future_mean) / future_scale
    gates = torch.softmax(-0.5 * ((prefix_z[:, None, :] - centers[None]) / bandwidth[None]).pow(2), dim=1)
    mass = gates.mean(0).clamp_min(float(eps))
    conditional_mean = (gates * future_z[:, None, :]).mean(0) / mass
    conditional_second = (gates * future_z[:, None, :].pow(2)).mean(0) / mass
    conditional_std = torch.sqrt((conditional_second - conditional_mean.pow(2)).clamp_min(0.0))
    return FixedTargetPrefixFutureVolatilityKernelMoments(
        prefix_mean, prefix_scale, centers, bandwidth, future_mean, future_scale,
        conditional_mean.detach(), conditional_std.detach(), int(split_index),
        int(prefix_window), int(future_horizon), float(mean_weight),
        float(std_weight), float(eps),
    )


@dataclass(frozen=True)
class StabilizedPrefixFutureVolatilityCorrelationDiscrepancy:
    """Direct discrepancy between prefix/future realized-volatility correlations.

    For each window, prefix and future RMS realized volatility are centered
    within each empirical law and divided by their own empirical scales.  A
    detached fraction of the target scale floors each generated-law scale.
    Above that floor the generated statistic is the ordinary Pearson
    correlation, so changing marginal volatility amplitude cannot satisfy the
    discrepancy.  The floor keeps the statistic and its gradient finite while
    generated volatility-state dispersion is initially collapsed.

    By default only matched window/horizon pairs are compared;
    ``full_matrix=True`` compares every prefix window with every future horizon.
    """

    windows: Sequence[int] = (4, 8)
    full_matrix: bool = False
    split_index: int | None = None
    scale_floor_fraction: float = 0.25
    eps: float = 1e-6
    name: str = "prefix_future_volatility_correlations"

    def __post_init__(self) -> None:
        windows = tuple(int(value) for value in self.windows)
        if len(windows) == 0 or any(value < 1 for value in windows):
            raise ValueError("windows must be non-empty positive integers")
        if len(set(windows)) != len(windows):
            raise ValueError("windows must not contain duplicates")
        if not isinstance(self.full_matrix, bool):
            raise ValueError("full_matrix must be a bool")
        split_index = None if self.split_index is None else int(self.split_index)
        if split_index is not None and split_index < 1:
            raise ValueError("split_index must be >= 1 when provided")
        scale_floor_fraction = float(self.scale_floor_fraction)
        if not math.isfinite(scale_floor_fraction) or not 0.0 < scale_floor_fraction <= 1.0:
            raise ValueError("scale_floor_fraction must be in (0, 1]")
        eps = float(self.eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "split_index", split_index)
        object.__setattr__(self, "scale_floor_fraction", scale_floor_fraction)
        object.__setattr__(self, "eps", eps)

    def _realized_volatility(
        self,
        paths: torch.Tensor,
        *,
        grid: DiscreteTimeGrid,
        split_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scaled_returns = _returns(paths, grid=grid) / math.sqrt(float(grid.dt))
        prefix_values = []
        future_values = []
        eps_squared = float(self.eps) ** 2
        for window in tuple(int(value) for value in self.windows):
            prefix = scaled_returns[:, split_index - window : split_index, :]
            future = scaled_returns[:, split_index : split_index + window, :]
            prefix_values.append(torch.sqrt(prefix.pow(2).mean(dim=1) + eps_squared))
            future_values.append(torch.sqrt(future.pow(2).mean(dim=1) + eps_squared))
        return torch.stack(prefix_values, dim=1), torch.stack(future_values, dim=1)

    def _correlations(
        self,
        prefix_volatility: torch.Tensor,
        future_volatility: torch.Tensor,
        *,
        prefix_scale_floor: torch.Tensor,
        future_scale_floor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_mean = prefix_volatility.mean(dim=0, keepdim=True)
        future_mean = future_volatility.mean(dim=0, keepdim=True)
        prefix_centered = prefix_volatility - prefix_mean
        future_centered = future_volatility - future_mean
        eps_squared = float(self.eps) ** 2
        prefix_scale = torch.sqrt(torch.mean(prefix_centered.pow(2), dim=0) + eps_squared)
        future_scale = torch.sqrt(torch.mean(future_centered.pow(2), dim=0) + eps_squared)
        stabilized_prefix_scale = torch.maximum(prefix_scale, prefix_scale_floor)
        stabilized_future_scale = torch.maximum(future_scale, future_scale_floor)
        if bool(self.full_matrix):
            covariance = torch.mean(
                prefix_centered.unsqueeze(2) * future_centered.unsqueeze(1),
                dim=0,
            )
            correlations = covariance / (
                stabilized_prefix_scale.unsqueeze(1) * stabilized_future_scale.unsqueeze(0)
            )
        else:
            covariance = torch.mean(prefix_centered * future_centered, dim=0)
            correlations = covariance / (stabilized_prefix_scale * stabilized_future_scale)
        prefix_floor_active = prefix_scale < prefix_scale_floor
        future_floor_active = future_scale < future_scale_floor
        return correlations, prefix_floor_active, future_floor_active

    def _law_statistics(
        self,
        prefix_volatility: torch.Tensor,
        future_volatility: torch.Tensor,
    ) -> torch.Tensor:
        cross = (
            prefix_volatility.unsqueeze(2) * future_volatility.unsqueeze(1)
            if bool(self.full_matrix)
            else prefix_volatility * future_volatility
        )
        batch_size = int(prefix_volatility.shape[0])
        return torch.cat(
            (
                prefix_volatility.reshape(batch_size, -1),
                future_volatility.reshape(batch_size, -1),
                prefix_volatility.pow(2).reshape(batch_size, -1),
                future_volatility.pow(2).reshape(batch_size, -1),
                cross.reshape(batch_size, -1),
            ),
            dim=1,
        )

    def _correlations_from_mean_statistics(
        self,
        mean_statistics: torch.Tensor,
        *,
        prefix_scale_floor: torch.Tensor,
        future_scale_floor: torch.Tensor,
    ) -> torch.Tensor:
        window_count = len(tuple(self.windows))
        state_dim = int(prefix_scale_floor.shape[1])
        marginal_dim = window_count * state_dim
        offset = 0
        prefix_mean = mean_statistics[offset : offset + marginal_dim].reshape(
            window_count, state_dim
        )
        offset += marginal_dim
        future_mean = mean_statistics[offset : offset + marginal_dim].reshape(
            window_count, state_dim
        )
        offset += marginal_dim
        prefix_second = mean_statistics[offset : offset + marginal_dim].reshape(
            window_count, state_dim
        )
        offset += marginal_dim
        future_second = mean_statistics[offset : offset + marginal_dim].reshape(
            window_count, state_dim
        )
        offset += marginal_dim
        cross_shape = (
            (window_count, window_count, state_dim)
            if bool(self.full_matrix)
            else (window_count, state_dim)
        )
        cross = mean_statistics[offset:].reshape(cross_shape)
        eps_squared = float(self.eps) ** 2
        prefix_scale = torch.sqrt(
            (prefix_second - prefix_mean.pow(2)).clamp_min(0.0)
            + eps_squared
        )
        future_scale = torch.sqrt(
            (future_second - future_mean.pow(2)).clamp_min(0.0)
            + eps_squared
        )
        stabilized_prefix = torch.maximum(prefix_scale, prefix_scale_floor)
        stabilized_future = torch.maximum(future_scale, future_scale_floor)
        if bool(self.full_matrix):
            covariance = cross - prefix_mean.unsqueeze(1) * future_mean.unsqueeze(0)
            return covariance / (
                stabilized_prefix.unsqueeze(1) * stabilized_future.unsqueeze(0)
            )
        covariance = cross - prefix_mean * future_mean
        return covariance / (stabilized_prefix * stabilized_future)

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        if int(generated_paths.shape[0]) < 2 or int(target_paths.shape[0]) < 2:
            raise ValueError("conditional volatility correlations require at least two paths per law")
        num_steps = int(grid.num_steps)
        split_index = int(self.split_index) if self.split_index is not None else num_steps // 2
        largest_window = max(tuple(int(value) for value in self.windows))
        if split_index < largest_window or num_steps - split_index < largest_window:
            raise ValueError("every window must fit before and after the conditional-volatility split")
        target_paths = target_paths.to(device=generated_paths.device, dtype=generated_paths.dtype)
        generated_prefix, generated_future = self._realized_volatility(
            generated_paths,
            grid=grid,
            split_index=split_index,
        )
        target_prefix, target_future = self._realized_volatility(
            target_paths,
            grid=grid,
            split_index=split_index,
        )
        target_prefix_scale = target_prefix.std(dim=0, unbiased=False).clamp_min(float(self.eps)).detach()
        target_future_scale = target_future.std(dim=0, unbiased=False).clamp_min(float(self.eps)).detach()
        generated_correlations, prefix_floor_active, future_floor_active = self._correlations(
            generated_prefix,
            generated_future,
            prefix_scale_floor=float(self.scale_floor_fraction) * target_prefix_scale,
            future_scale_floor=float(self.scale_floor_fraction) * target_future_scale,
        )
        target_correlations, _, _ = self._correlations(
            target_prefix,
            target_future,
            prefix_scale_floor=float(self.scale_floor_fraction) * target_prefix_scale,
            future_scale_floor=float(self.scale_floor_fraction) * target_future_scale,
        )
        target_correlations = target_correlations.detach()
        differences = generated_correlations - target_correlations
        value = torch.mean(differences.pow(2))
        metrics = {
            f"{self.name}_value": float(value.detach().item()),
            f"{self.name}_correlation_count": float(generated_correlations.numel()),
            f"{self.name}_correlation_rmse": float(
                torch.sqrt(torch.mean(differences.detach().pow(2))).item()
            ),
            f"{self.name}_split_index": float(split_index),
            f"{self.name}_largest_window": float(largest_window),
            f"{self.name}_full_matrix": float(bool(self.full_matrix)),
            f"{self.name}_scale_floor_fraction": float(self.scale_floor_fraction),
            f"{self.name}_generated_scale_floor_active_fraction": float(
                torch.cat(
                    [prefix_floor_active.reshape(-1), future_floor_active.reshape(-1)]
                )
                .to(dtype=generated_paths.dtype)
                .mean()
                .detach()
                .item()
            ),
        }
        state_dim = int(generated_paths.shape[-1])
        windows = tuple(int(value) for value in self.windows)
        pairs = (
            tuple(
                (prefix_index, future_index)
                for prefix_index in range(len(windows))
                for future_index in range(len(windows))
            )
            if bool(self.full_matrix)
            else tuple((index, index) for index in range(len(windows)))
        )
        for prefix_index, future_index in pairs:
            for asset in range(state_dim):
                generated_value = (
                    generated_correlations[prefix_index, future_index, asset]
                    if bool(self.full_matrix)
                    else generated_correlations[prefix_index, asset]
                )
                target_value = (
                    target_correlations[prefix_index, future_index, asset]
                    if bool(self.full_matrix)
                    else target_correlations[prefix_index, asset]
                )
                metric_suffix = f"w{windows[prefix_index]}_h{windows[future_index]}_asset{asset}"
                metrics[f"{self.name}_generated_{metric_suffix}"] = float(
                    generated_value.detach().item()
                )
                metrics[f"{self.name}_target_{metric_suffix}"] = float(target_value.detach().item())
                metrics[f"{self.name}_error_{metric_suffix}"] = float(
                    (generated_value.detach() - target_value.detach()).item()
                )
        return PathFunctionalDiscrepancyResult(value=value, metrics=metrics)

    def lions_potential(
        self,
        *,
        query_paths: torch.Tensor,
        population_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> torch.Tensor:
        """Frozen-law influence potential for the correlation functional.

        Correlations depend on the generated law only through first, second,
        and prefix/future cross moments.  The gradient of the scalar
        correlation error with respect to those unconditional moments gives
        coefficients whose dot product with query-path statistics is the
        Lions potential.
        """

        num_steps = int(grid.num_steps)
        split_index = (
            int(self.split_index)
            if self.split_index is not None
            else num_steps // 2
        )
        query_prefix, query_future = self._realized_volatility(
            query_paths,
            grid=grid,
            split_index=split_index,
        )
        with torch.no_grad():
            population_prefix, population_future = self._realized_volatility(
                population_paths,
                grid=grid,
                split_index=split_index,
            )
            target_prefix, target_future = self._realized_volatility(
                target_paths.to(
                    device=query_paths.device,
                    dtype=query_paths.dtype,
                ),
                grid=grid,
                split_index=split_index,
            )
            target_prefix_scale = target_prefix.std(
                dim=0,
                unbiased=False,
            ).clamp_min(float(self.eps))
            target_future_scale = target_future.std(
                dim=0,
                unbiased=False,
            ).clamp_min(float(self.eps))
            prefix_floor = (
                float(self.scale_floor_fraction) * target_prefix_scale
            )
            future_floor = (
                float(self.scale_floor_fraction) * target_future_scale
            )
            target_correlations, _, _ = self._correlations(
                target_prefix,
                target_future,
                prefix_scale_floor=prefix_floor,
                future_scale_floor=future_floor,
            )
            population_mean = self._law_statistics(
                population_prefix,
                population_future,
            ).mean(dim=0)

        population_mean_req = population_mean.detach().clone().requires_grad_(
            True
        )
        population_correlations = self._correlations_from_mean_statistics(
            population_mean_req,
            prefix_scale_floor=prefix_floor,
            future_scale_floor=future_floor,
        )
        population_value = (
            population_correlations - target_correlations.detach()
        ).pow(2).mean()
        influence_coefficients = torch.autograd.grad(
            population_value,
            population_mean_req,
        )[0].detach()
        query_statistics = self._law_statistics(query_prefix, query_future)
        return (query_statistics * influence_coefficients).sum(dim=1).mean()


def _mmd_block(name: str, weight: float, feature_map, bandwidths: Sequence[float]) -> WeightedDiscrepancy:
    return WeightedDiscrepancy(
        name=name,
        weight=float(weight),
        discrepancy=MMDPathFunctionalDiscrepancy(
            feature_map=feature_map,
            bandwidths=tuple(float(value) for value in bandwidths),
            name=f"mmd_{name}",
        ),
    )


def _discrepancy_block(name: str, weight: float, discrepancy) -> WeightedDiscrepancy:
    return WeightedDiscrepancy(name=name, weight=float(weight), discrepancy=discrepancy)


def _old_w0p25_sigma_like_config() -> dict[str, tuple[float, ...] | float]:
    return {
        "sigma_like_terminal_return_weight": 0.4424396519610169,
        "sigma_like_terminal_return_bandwidths": (0.17280253767967224, 0.3456050753593445, 0.08640126883983612),
        "sigma_like_returns_weight": 0.6262497370003193,
        "sigma_like_returns_bandwidths": (0.02287130430340767, 0.04574260860681534, 0.09148521721363068),
        "sigma_like_realized_vol_weight": 0.7056226192588566,
        "sigma_like_realized_vol_bandwidths": (0.02173430845141411, 0.04346861690282822, 0.010867154225707054),
        "sigma_like_abs_returns_weight": 0.72636567982937,
        "sigma_like_abs_returns_bandwidths": (0.032140329480171204, 0.016070164740085602, 0.06428065896034241),
        "sigma_like_squared_returns_weight": 0.25,
        "sigma_like_squared_returns_bandwidths": (0.01, 0.011117583140730858),
        "abs_return_acf_weight": 0.25,
        "abs_return_acf_bandwidths": (2.169067621231079, 4.338135242462158, 1.0845338106155396),
        "squared_return_acf_weight": 0.125,
        "squared_return_acf_bandwidths": (2.128969192504883, 1.0644845962524414, 0.5322422981262207),
    }


def build_heston_discrepancy(
    *,
    num_steps: int,
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0),
    include_acf: bool = True,
    acf_lags: Sequence[int] = (1, 2, 5, 10),
    preset: str = "simple",
    lambda_scale: float = 50.0,
    kappa_scale: float = 50.0,
    include_conditional_volatility: bool = False,
    include_conditional_volatility_correlation: bool = False,
    conditional_volatility_windows: Sequence[int] = (1, 2, 4, 8),
    conditional_volatility_correlation_windows: Sequence[int] = (4, 8),
    conditional_volatility_weight: float = 1.0,
    conditional_volatility_correlation_weight: float = 0.025,
    conditional_volatility_bandwidths: Sequence[float] = (0.5, 1.0, 2.0),
    conditional_volatility_correlation_full_matrix: bool = False,
    conditional_volatility_correlation_scale_floor_fraction: float = 0.25,
    time_marginal_volatility_window: int | None = None,
    time_marginal_volatility_endpoints: Sequence[int] = (),
    time_marginal_volatility_weight: float = 1.0,
    abs_return_acf_weight: float | None = None,
    squared_return_acf_weight: float | None = None,
) -> CompositePathFunctionalDiscrepancy:
    """
    Build a Heston-oriented MMD discrepancy.

    The blocks compare return-level information, volatility proxies, and, when
    enabled, lag products for volatility-clustering diagnostics.
    """
    if num_steps < 2:
        raise ValueError("num_steps must be >= 2")
    if float(lambda_scale) <= 0.0:
        raise ValueError("lambda_scale must be > 0")
    if float(kappa_scale) < 0.0:
        raise ValueError("kappa_scale must be >= 0")
    if float(conditional_volatility_weight) < 0.0:
        raise ValueError("conditional_volatility_weight must be >= 0")
    if float(conditional_volatility_correlation_weight) < 0.0:
        raise ValueError("conditional_volatility_correlation_weight must be >= 0")
    if float(time_marginal_volatility_weight) < 0.0:
        raise ValueError("time_marginal_volatility_weight must be >= 0")
    if abs_return_acf_weight is not None and float(abs_return_acf_weight) < 0.0:
        raise ValueError("abs_return_acf_weight must be >= 0 when supplied")
    if squared_return_acf_weight is not None and float(squared_return_acf_weight) < 0.0:
        raise ValueError("squared_return_acf_weight must be >= 0 when supplied")
    if not 0.0 < float(conditional_volatility_correlation_scale_floor_fraction) <= 1.0:
        raise ValueError("conditional_volatility_correlation_scale_floor_fraction must be in (0, 1]")
    if preset not in {"simple", "old_fullv_w0p25", "reduced_time_local"}:
        raise ValueError("preset must be simple, old_fullv_w0p25, or reduced_time_local")

    lags = _valid_lags(acf_lags, num_returns=int(num_steps))
    window = min(5, int(num_steps))
    rolling_volatility_points = int(num_steps) - window + 1
    rv_lags = (
        _valid_lags(acf_lags, num_returns=rolling_volatility_points)
        if rolling_volatility_points >= 2
        else ()
    )

    if preset == "reduced_time_local":
        path_bandwidths = (0.2, 0.5, 1.0, 2.0, 4.0)
        vol_scale = float(kappa_scale) / float(lambda_scale)
        marginal_window = (
            max(1, int(num_steps) // 8)
            if time_marginal_volatility_window is None
            else int(time_marginal_volatility_window)
        )
        blocks = [
            _mmd_block(
                "path_law_observed_path",
                1.0,
                ObservedPathFeature(),
                path_bandwidths,
            ),
            _discrepancy_block(
                "volatility_law_time_marginal_rv",
                vol_scale * float(time_marginal_volatility_weight),
                TimeMarginalRealizedVolatilityMMD(
                    window=marginal_window,
                    endpoint_indices=time_marginal_volatility_endpoints,
                    bandwidths=conditional_volatility_bandwidths,
                ),
            ),
        ]
        if bool(include_conditional_volatility_correlation):
            blocks.append(
                _discrepancy_block(
                    "volatility_law_conditional_correlation",
                    vol_scale * float(conditional_volatility_correlation_weight),
                    StabilizedPrefixFutureVolatilityCorrelationDiscrepancy(
                        windows=conditional_volatility_correlation_windows,
                        full_matrix=bool(conditional_volatility_correlation_full_matrix),
                        scale_floor_fraction=float(
                            conditional_volatility_correlation_scale_floor_fraction
                        ),
                    ),
                )
            )
        return CompositePathFunctionalDiscrepancy(blocks=tuple(blocks))

    if preset == "old_fullv_w0p25":
        path_bandwidths = (0.2, 0.5, 1.0, 2.0, 4.0)
        vol_bandwidths = (0.01, 0.025, 0.05, 0.1, 0.2)
        state_rv_bandwidths = (0.5, 1.0, 2.0, 4.0)
        sigma = _old_w0p25_sigma_like_config()
        if abs_return_acf_weight is not None:
            sigma["abs_return_acf_weight"] = float(abs_return_acf_weight)
        if squared_return_acf_weight is not None:
            sigma["squared_return_acf_weight"] = float(squared_return_acf_weight)
        vol_scale = float(kappa_scale) / float(lambda_scale)
        blocks = [
            _mmd_block("path_law_observed_path", 2.0, ObservedPathFeature(), path_bandwidths),
            _mmd_block("path_law_terminal", 1.0, TerminalFeature(), path_bandwidths),
            _mmd_block("path_law_increments", 2.0, ReturnsFeature(), path_bandwidths),
            _mmd_block("volatility_law_instantaneous_rv", vol_scale, RealizedVarianceFeature(), vol_bandwidths),
            _discrepancy_block(
                "volatility_law_state_rv",
                vol_scale,
                TargetStandardizedStateRealizedVarianceMMD(bandwidths=state_rv_bandwidths),
            ),
            _mmd_block("volatility_law_global_rv", vol_scale, GlobalRealizedVarianceFeature(), vol_bandwidths),
            _mmd_block(
                "volatility_law_sigma_like_terminal_return",
                vol_scale * float(sigma["sigma_like_terminal_return_weight"]),
                TerminalReturnFeature(),
                sigma["sigma_like_terminal_return_bandwidths"],  # type: ignore[arg-type]
            ),
            _mmd_block(
                "volatility_law_sigma_like_returns",
                vol_scale * float(sigma["sigma_like_returns_weight"]),
                ReturnsFeature(),
                sigma["sigma_like_returns_bandwidths"],  # type: ignore[arg-type]
            ),
            _mmd_block(
                "volatility_law_sigma_like_realized_vol",
                vol_scale * float(sigma["sigma_like_realized_vol_weight"]),
                RealizedVolatilityFeature(window=window, smoothing_epsilon=0.001),
                sigma["sigma_like_realized_vol_bandwidths"],  # type: ignore[arg-type]
            ),
            _mmd_block(
                "volatility_law_sigma_like_abs_returns",
                vol_scale * float(sigma["sigma_like_abs_returns_weight"]),
                AbsReturnsFeature(smoothing_epsilon=0.001),
                sigma["sigma_like_abs_returns_bandwidths"],  # type: ignore[arg-type]
            ),
            _mmd_block(
                "volatility_law_sigma_like_squared_returns",
                vol_scale * float(sigma["sigma_like_squared_returns_weight"]),
                SquaredReturnsFeature(),
                sigma["sigma_like_squared_returns_bandwidths"],  # type: ignore[arg-type]
            ),
        ]
        if include_acf:
            blocks.extend(
                [
                    _mmd_block(
                        "volatility_law_abs_return_acf",
                        vol_scale * float(sigma["abs_return_acf_weight"]),
                        ACFLagProductFeature(lags=lags, squared=False, smoothing_epsilon=0.001),
                        sigma["abs_return_acf_bandwidths"],  # type: ignore[arg-type]
                    ),
                    _mmd_block(
                        "volatility_law_squared_return_acf",
                        vol_scale * float(sigma["squared_return_acf_weight"]),
                        ACFLagProductFeature(lags=lags, squared=True, smoothing_epsilon=0.001),
                        sigma["squared_return_acf_bandwidths"],  # type: ignore[arg-type]
                    ),
                ]
            )
        if bool(include_conditional_volatility):
            blocks.append(
                _discrepancy_block(
                    "volatility_law_conditional_multiscale",
                    vol_scale * float(conditional_volatility_weight),
                    MarginalInvariantConditionalVolatilityMMD(
                        windows=conditional_volatility_windows,
                        bandwidths=conditional_volatility_bandwidths,
                    ),
                )
            )
        if bool(include_conditional_volatility_correlation):
            blocks.append(
                _discrepancy_block(
                    "volatility_law_conditional_correlation",
                    vol_scale * float(conditional_volatility_correlation_weight),
                    StabilizedPrefixFutureVolatilityCorrelationDiscrepancy(
                        windows=conditional_volatility_correlation_windows,
                        full_matrix=bool(conditional_volatility_correlation_full_matrix),
                        scale_floor_fraction=float(
                            conditional_volatility_correlation_scale_floor_fraction
                        ),
                    ),
                )
            )
        return CompositePathFunctionalDiscrepancy(blocks=tuple(blocks))

    blocks = [
        _mmd_block("returns", 1.0, ReturnsFeature(), bandwidths),
        _mmd_block("abs_returns", 0.5, AbsReturnsFeature(smoothing_epsilon=1e-6), bandwidths),
        _mmd_block("squared_returns", 0.25, SquaredReturnsFeature(), bandwidths),
        _mmd_block(
            "realized_volatility",
            0.25,
            RealizedVolatilityFeature(window=window, smoothing_epsilon=1e-6),
            bandwidths,
        ),
    ]

    if include_acf:
        blocks.extend(
            [
                _mmd_block(
                    "abs_return_acf",
                    0.25,
                    ACFLagProductFeature(lags=lags, squared=False, smoothing_epsilon=1e-6),
                    bandwidths,
                ),
                _mmd_block(
                    "squared_return_acf",
                    0.25,
                    ACFLagProductFeature(lags=lags, squared=True, smoothing_epsilon=1e-6),
                    bandwidths,
                ),
            ]
        )
        if len(rv_lags) > 0:
            blocks.append(
                _mmd_block(
                    "rolling_volatility_acf",
                    0.25,
                    RollingVolatilityACFLagProductFeature(
                        lags=rv_lags,
                        window=window,
                        smoothing_epsilon=1e-6,
                    ),
                    bandwidths,
                )
            )

    if bool(include_conditional_volatility):
        blocks.append(
            _discrepancy_block(
                "conditional_multiscale_volatility",
                (float(kappa_scale) / float(lambda_scale)) * float(conditional_volatility_weight),
                MarginalInvariantConditionalVolatilityMMD(
                    windows=conditional_volatility_windows,
                    bandwidths=conditional_volatility_bandwidths,
                ),
            )
        )

    if bool(include_conditional_volatility_correlation):
        blocks.append(
            _discrepancy_block(
                "conditional_volatility_correlation",
                (float(kappa_scale) / float(lambda_scale))
                * float(conditional_volatility_correlation_weight),
                StabilizedPrefixFutureVolatilityCorrelationDiscrepancy(
                    windows=conditional_volatility_correlation_windows,
                    full_matrix=bool(conditional_volatility_correlation_full_matrix),
                    scale_floor_fraction=float(
                        conditional_volatility_correlation_scale_floor_fraction
                    ),
                ),
            )
        )

    return CompositePathFunctionalDiscrepancy(blocks=tuple(blocks))


def build_path_volatility_discrepancy(
    *,
    path_bandwidths: Sequence[float] = (0.2, 0.5, 1.0, 2.0, 4.0),
    vol_bandwidths: Sequence[float] = (0.01, 0.025, 0.05, 0.1, 0.2),
    state_rv_bandwidths: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    observed_path_weight: float = 2.0,
    terminal_weight: float = 1.0,
    increments_weight: float = 2.0,
    vol_instantaneous_weight: float = 1.0,
    vol_state_rv_weight: float = 1.0,
    vol_global_weight: float = 1.0,
    lambda_scale: float = 50.0,
    kappa_scale: float = 0.0,
) -> CompositePathFunctionalDiscrepancy:
    """
    Build the generic path-law plus realized-volatility discrepancy used by
    migrated discrete-time experiments.

    The model core multiplies the returned value by ``lambda_scale``.  Volatility
    blocks are therefore stored with weight ``kappa_scale / lambda_scale`` so
    the effective source is ``lambda * G_path + kappa * G_vol``.
    """
    if float(lambda_scale) <= 0.0:
        raise ValueError("lambda_scale must be > 0")
    if float(kappa_scale) < 0.0:
        raise ValueError("kappa_scale must be >= 0")
    blocks = [
        _mmd_block("path_law_observed_path", float(observed_path_weight), ObservedPathFeature(), path_bandwidths),
        _mmd_block("path_law_terminal", float(terminal_weight), TerminalFeature(), path_bandwidths),
        _mmd_block("path_law_increments", float(increments_weight), ReturnsFeature(), path_bandwidths),
    ]
    if float(kappa_scale) > 0.0:
        vol_scale = float(kappa_scale) / float(lambda_scale)
        blocks.extend(
            [
                _mmd_block(
                    "volatility_law_instantaneous_rv",
                    vol_scale * float(vol_instantaneous_weight),
                    RealizedVarianceFeature(),
                    vol_bandwidths,
                ),
                _discrepancy_block(
                    "volatility_law_state_rv",
                    vol_scale * float(vol_state_rv_weight),
                    TargetStandardizedStateRealizedVarianceMMD(bandwidths=state_rv_bandwidths),
                ),
                _mmd_block(
                    "volatility_law_global_rv",
                    vol_scale * float(vol_global_weight),
                    GlobalRealizedVarianceFeature(),
                    vol_bandwidths,
                ),
            ]
        )
    return CompositePathFunctionalDiscrepancy(blocks=tuple(blocks))
