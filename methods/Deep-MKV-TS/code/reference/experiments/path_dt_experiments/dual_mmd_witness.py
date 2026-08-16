from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from deep_mkv_gen_path_dt.discrepancies import PathFunctionalDiscrepancyResult
from deep_mkv_gen_path_dt.discrepancies.mmd import rbf_mmd2
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


def _rbf_kernel(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    bandwidths: tuple[float, ...],
) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2 or int(left.shape[1]) != int(right.shape[1]):
        raise ValueError("kernel inputs must have compatible shape (B,f)")
    left_norm = left.pow(2).sum(dim=1, keepdim=True)
    right_norm = right.pow(2).sum(dim=1, keepdim=True).transpose(0, 1)
    distance = (left_norm + right_norm - 2.0 * left @ right.transpose(0, 1)).clamp_min(0.0)
    feature_scale = float(max(1, int(left.shape[1])))
    result = left.new_zeros(distance.shape)
    for bandwidth in bandwidths:
        result = result + torch.exp(
            -distance / (2.0 * float(bandwidth) ** 2 * feature_scale)
        )
    return result / float(len(bandwidths))


@dataclass(frozen=True)
class FixedEmpiricalRBFMMDWitness:
    """Exact finite-support RKHS dual witness for a biased RBF MMD.

    The witness is represented in the empirical RKHS span as
    ``w = mean_i phi(x_i) - mean_j phi(y_j)``.  Holding these supports fixed
    removes population-law regeneration from the inner pathwise source while
    retaining the original RBF kernel without random-feature approximation.
    """

    residual_feature_map: object
    population_features: torch.Tensor
    target_features: torch.Tensor
    bandwidths: tuple[float, ...]
    witness_scale: float = 1.0
    name: str = "fixed_dual_residual_conditional_volatility"

    def __post_init__(self) -> None:
        population = self.population_features.detach().cpu().clone()
        target = self.target_features.detach().cpu().clone()
        if population.ndim != 2 or target.ndim != 2:
            raise ValueError("witness supports must have shape (B,f)")
        if int(population.shape[1]) != int(target.shape[1]) or int(population.shape[1]) < 1:
            raise ValueError("witness feature dimensions must agree and be positive")
        bandwidths = tuple(float(value) for value in self.bandwidths)
        if not bandwidths or any(not math.isfinite(value) or value <= 0.0 for value in bandwidths):
            raise ValueError("bandwidths must be positive and finite")
        scale = float(self.witness_scale)
        if not math.isfinite(scale):
            raise ValueError("witness_scale must be finite")
        object.__setattr__(self, "population_features", population)
        object.__setattr__(self, "target_features", target)
        object.__setattr__(self, "bandwidths", bandwidths)
        object.__setattr__(self, "witness_scale", scale)

    def _features(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        features = getattr(self.residual_feature_map, "features", None)
        if not callable(features):
            raise TypeError("residual_feature_map must expose features(paths, grid=...)")
        return features(paths, grid=grid)

    def witness_values(self, features: torch.Tensor) -> torch.Tensor:
        population = self.population_features.to(device=features.device, dtype=features.dtype)
        target = self.target_features.to(device=features.device, dtype=features.dtype)
        return float(self.witness_scale) * (
            _rbf_kernel(features, population, bandwidths=self.bandwidths).mean(dim=1)
            - _rbf_kernel(features, target, bandwidths=self.bandwidths).mean(dim=1)
        )

    def witness_norm_squared(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        population = self.population_features.to(device=device, dtype=dtype)
        target = self.target_features.to(device=device, dtype=dtype)
        return rbf_mmd2(
            population,
            target,
            bandwidths=self.bandwidths,
            unbiased=False,
        )

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        del target_paths
        generated = self._features(generated_paths, grid=grid)
        frozen_target = self.target_features.to(
            device=generated.device, dtype=generated.dtype
        )
        pairing = self.witness_values(generated).mean() - self.witness_values(
            frozen_target
        ).mean()
        norm = self.witness_norm_squared(
            device=generated.device, dtype=generated.dtype
        )
        scale = float(self.witness_scale)
        value = 2.0 * pairing - scale * scale * norm
        return PathFunctionalDiscrepancyResult(
            value=value,
            metrics={
                f"{self.name}_value": float(value.detach().item()),
                f"{self.name}_witness_norm_squared": float(norm.detach().item()),
                f"{self.name}_population_support": float(
                    self.population_features.shape[0]
                ),
                f"{self.name}_target_support": float(self.target_features.shape[0]),
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
        del population_paths, target_paths
        query = self._features(query_paths, grid=grid)
        return 2.0 * self.witness_values(query).mean()

    def summary(self) -> dict[str, object]:
        population_count = int(self.population_features.shape[0])
        target_count = int(self.target_features.shape[0])
        return {
            "name": self.name,
            "representation": "finite empirical expansion in the exact RBF RKHS",
            "population_support": population_count,
            "target_support": target_count,
            "population_coefficients": {
                "count": population_count,
                "common_value": float(self.witness_scale) / float(population_count),
            },
            "target_coefficients": {
                "count": target_count,
                "common_value": -float(self.witness_scale) / float(target_count),
            },
            "bandwidths": list(self.bandwidths),
            "population_source_frozen": True,
        }


__all__ = ["FixedEmpiricalRBFMMDWitness"]
