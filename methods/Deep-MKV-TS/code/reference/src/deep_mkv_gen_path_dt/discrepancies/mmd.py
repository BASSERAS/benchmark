from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from deep_mkv_gen_path_dt.discrepancies.base import PathFunctionalDiscrepancyResult
from deep_mkv_gen_path_dt.discrepancies.analytic import feature_path_vjp
from deep_mkv_gen_path_dt.discrepancies.features import IdentityPathFeature, PathFeatureMap
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


def _require_bandwidth(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("bandwidths must be finite positive floats")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("bandwidths must be finite")
    if result <= 0.0:
        raise ValueError("bandwidths must be > 0")
    return result


def _pairwise_squared_distances(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_norm = left.pow(2).sum(dim=1, keepdim=True)
    right_norm = right.pow(2).sum(dim=1, keepdim=True).transpose(0, 1)
    distances = left_norm + right_norm - 2.0 * left @ right.transpose(0, 1)
    return distances.clamp_min(0.0)


def rbf_mmd2(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    bandwidths: Sequence[float],
    unbiased: bool = False,
) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left and right must have shape (B, p)")
    if int(left.shape[1]) != int(right.shape[1]):
        raise ValueError("left and right feature dimensions must match")
    if int(left.shape[1]) < 1:
        raise ValueError("MMD feature dimension must be >= 1")
    right = right.to(device=left.device, dtype=left.dtype)
    bandwidth_values = tuple(_require_bandwidth(value) for value in bandwidths)
    if len(bandwidth_values) == 0:
        raise ValueError("bandwidths must be non-empty")

    d_xx = _pairwise_squared_distances(left, left)
    d_yy = _pairwise_squared_distances(right, right)
    d_xy = _pairwise_squared_distances(left, right)
    k_xx = left.new_zeros(d_xx.shape)
    k_yy = left.new_zeros(d_yy.shape)
    k_xy = left.new_zeros(d_xy.shape)
    feature_scale = float(max(1, int(left.shape[1])))
    for bandwidth in bandwidth_values:
        scale = 2.0 * float(bandwidth) ** 2 * feature_scale
        k_xx = k_xx + torch.exp(-d_xx / scale)
        k_yy = k_yy + torch.exp(-d_yy / scale)
        k_xy = k_xy + torch.exp(-d_xy / scale)
    k_xx = k_xx / float(len(bandwidth_values))
    k_yy = k_yy / float(len(bandwidth_values))
    k_xy = k_xy / float(len(bandwidth_values))

    if not unbiased:
        return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

    n = int(left.shape[0])
    m = int(right.shape[0])
    if n < 2 or m < 2:
        raise ValueError("unbiased MMD requires at least two samples per batch")
    xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / float(n * (n - 1))
    yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / float(m * (m - 1))
    return xx + yy - 2.0 * k_xy.mean()


def rbf_mmd_lions_potential(
    query: torch.Tensor,
    population: torch.Tensor,
    target: torch.Tensor,
    *,
    bandwidths: Sequence[float],
) -> torch.Tensor:
    """Mean MMD Lions potential at queries for a frozen source law.

    For ``G(mu, nu) = E k(X,X') - 2 E k(X,Y) + const``, a
    representative functional derivative is
    ``2 E k(x,X) - 2 E k(x,Y)``.  The returned scalar averages that
    potential over ``query``.  Multiplying its autograd gradient by the query
    count therefore recovers one pathwise Lions source per query.
    """

    if query.ndim != 2 or population.ndim != 2 or target.ndim != 2:
        raise ValueError("query, population, and target must have shape (B, p)")
    feature_dim = int(query.shape[1])
    if feature_dim < 1 or int(population.shape[1]) != feature_dim or int(
        target.shape[1]
    ) != feature_dim:
        raise ValueError("all MMD feature dimensions must match and be positive")
    if int(query.shape[0]) < 1 or int(population.shape[0]) < 1 or int(
        target.shape[0]
    ) < 1:
        raise ValueError("query, population, and target laws must be non-empty")
    population = population.to(device=query.device, dtype=query.dtype)
    target = target.to(device=query.device, dtype=query.dtype)
    bandwidth_values = tuple(_require_bandwidth(value) for value in bandwidths)
    if len(bandwidth_values) == 0:
        raise ValueError("bandwidths must be non-empty")
    query_population_distances = _pairwise_squared_distances(query, population)
    query_target_distances = _pairwise_squared_distances(query, target)
    query_population_kernel = query.new_zeros(
        query_population_distances.shape
    )
    query_target_kernel = query.new_zeros(query_target_distances.shape)
    feature_scale = float(max(1, feature_dim))
    for bandwidth in bandwidth_values:
        scale = 2.0 * float(bandwidth) ** 2 * feature_scale
        query_population_kernel = query_population_kernel + torch.exp(
            -query_population_distances / scale
        )
        query_target_kernel = query_target_kernel + torch.exp(
            -query_target_distances / scale
        )
    divisor = float(len(bandwidth_values))
    potential = 2.0 * (
        query_population_kernel.mean(dim=1) / divisor
        - query_target_kernel.mean(dim=1) / divisor
    )
    return potential.mean()


def _rbf_kernel_and_left_gradient_sum(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    bandwidths: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the averaged kernel and ``sum_j d_left k(left_i,right_j)``."""

    bandwidth_values = tuple(_require_bandwidth(value) for value in bandwidths)
    if len(bandwidth_values) == 0:
        raise ValueError("bandwidths must be non-empty")
    distances = _pairwise_squared_distances(left, right)
    kernel = left.new_zeros(distances.shape)
    gradient_sum = left.new_zeros(left.shape)
    feature_scale = float(max(1, int(left.shape[1])))
    differences = left[:, None, :] - right[None, :, :]
    for bandwidth in bandwidth_values:
        scale = 2.0 * float(bandwidth) ** 2 * feature_scale
        component = torch.exp(-distances / scale)
        kernel.add_(component)
        gradient_sum.add_(
            (-2.0 / scale) * (component[..., None] * differences).sum(dim=1)
        )
    divisor = float(len(bandwidth_values))
    return kernel / divisor, gradient_sum / divisor


def rbf_mmd2_value_and_feature_gradient(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    bandwidths: Sequence[float],
    unbiased: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Biased/unbiased RBF MMD and its closed-form gradient in ``left``."""

    left = left.detach()
    right = right.detach().to(device=left.device, dtype=left.dtype)
    value = rbf_mmd2(
        left,
        right,
        bandwidths=bandwidths,
        unbiased=bool(unbiased),
    )
    _, xx_gradient_sum = _rbf_kernel_and_left_gradient_sum(
        left,
        left,
        bandwidths=bandwidths,
    )
    _, xy_gradient_sum = _rbf_kernel_and_left_gradient_sum(
        left,
        right,
        bandwidths=bandwidths,
    )
    n = int(left.shape[0])
    m = int(right.shape[0])
    if bool(unbiased):
        if n < 2 or m < 2:
            raise ValueError("unbiased MMD requires at least two samples per batch")
        xx_gradient = 2.0 * xx_gradient_sum / float(n * (n - 1))
    else:
        xx_gradient = 2.0 * xx_gradient_sum / float(n * n)
    xy_gradient = -2.0 * xy_gradient_sum / float(n * m)
    return value, xx_gradient + xy_gradient


def rbf_mmd_lions_potential_value_and_feature_gradient(
    query: torch.Tensor,
    population: torch.Tensor,
    target: torch.Tensor,
    *,
    bandwidths: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen-law MMD potential and its closed-form query gradient."""

    query = query.detach()
    population = population.detach().to(device=query.device, dtype=query.dtype)
    target = target.detach().to(device=query.device, dtype=query.dtype)
    potential = rbf_mmd_lions_potential(
        query,
        population,
        target,
        bandwidths=bandwidths,
    )
    _, population_gradient_sum = _rbf_kernel_and_left_gradient_sum(
        query,
        population,
        bandwidths=bandwidths,
    )
    _, target_gradient_sum = _rbf_kernel_and_left_gradient_sum(
        query,
        target,
        bandwidths=bandwidths,
    )
    query_count = int(query.shape[0])
    gradient = (2.0 / float(query_count)) * (
        population_gradient_sum / float(population.shape[0])
        - target_gradient_sum / float(target.shape[0])
    )
    return potential, gradient


@dataclass(frozen=True)
class MMDPathFunctionalDiscrepancy:
    feature_map: PathFeatureMap = IdentityPathFeature()
    bandwidths: Sequence[float] = (1.0,)
    unbiased: bool = False
    name: str = "mmd"

    def __post_init__(self) -> None:
        if not isinstance(self.unbiased, bool):
            raise ValueError("unbiased must be a bool")
        bandwidths = tuple(_require_bandwidth(value) for value in self.bandwidths)
        if len(bandwidths) == 0:
            raise ValueError("bandwidths must be non-empty")
        object.__setattr__(self, "bandwidths", bandwidths)

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        generated_features = self.feature_map(generated_paths, grid=grid)
        target_features = self.feature_map(target_paths, grid=grid)
        value = rbf_mmd2(
            generated_features,
            target_features,
            bandwidths=tuple(self.bandwidths),
            unbiased=bool(self.unbiased),
        )
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
        if bool(self.unbiased):
            raise TypeError(
                "the frozen-law Lions potential is implemented for the "
                "population/biased MMD estimator only"
            )
        return rbf_mmd_lions_potential(
            self.feature_map(query_paths, grid=grid),
            self.feature_map(population_paths, grid=grid).detach(),
            self.feature_map(target_paths, grid=grid).detach(),
            bandwidths=tuple(self.bandwidths),
        )

    def analytic_value_and_path_gradient(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> tuple[PathFunctionalDiscrepancyResult, torch.Tensor]:
        """Evaluate the empirical discrepancy and its path derivative by hand."""

        with torch.no_grad():
            generated_features = self.feature_map(generated_paths.detach(), grid=grid)
            target_features = self.feature_map(
                target_paths.detach().to(
                    device=generated_paths.device,
                    dtype=generated_paths.dtype,
                ),
                grid=grid,
            )
            value, feature_gradient = rbf_mmd2_value_and_feature_gradient(
                generated_features,
                target_features,
                bandwidths=tuple(self.bandwidths),
                unbiased=bool(self.unbiased),
            )
            path_gradient = feature_path_vjp(
                self.feature_map,
                generated_paths,
                feature_gradient,
                grid=grid,
            )
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
        if bool(self.unbiased):
            raise TypeError(
                "the frozen-law Lions potential is implemented for the "
                "population/biased MMD estimator only"
            )
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
            potential, feature_gradient = (
                rbf_mmd_lions_potential_value_and_feature_gradient(
                    query_features,
                    population_features,
                    target_features,
                    bandwidths=tuple(self.bandwidths),
                )
            )
            path_gradient = feature_path_vjp(
                self.feature_map,
                query_paths,
                feature_gradient,
                grid=grid,
            )
        return potential, path_gradient
