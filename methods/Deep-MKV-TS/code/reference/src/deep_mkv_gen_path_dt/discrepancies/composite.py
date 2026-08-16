from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from deep_mkv_gen_path_dt.config import _require_finite_float
from deep_mkv_gen_path_dt.discrepancies.base import (
    PathFunctionalDiscrepancy,
    PathFunctionalDiscrepancyResult,
)
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


@dataclass(frozen=True)
class WeightedDiscrepancy:
    name: str
    weight: float
    discrepancy: PathFunctionalDiscrepancy

    def __post_init__(self) -> None:
        weight = _require_finite_float("weight", self.weight)
        if weight < 0.0:
            raise ValueError("weight must be >= 0")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class CompositePathFunctionalDiscrepancy:
    blocks: Sequence[WeightedDiscrepancy]

    def __post_init__(self) -> None:
        if len(tuple(self.blocks)) == 0:
            raise ValueError("blocks must be non-empty")

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        total = generated_paths.new_tensor(0.0)
        metrics: dict[str, float] = {}
        for block in tuple(self.blocks):
            result = block.discrepancy(
                generated_paths=generated_paths,
                target_paths=target_paths,
                grid=grid,
            )
            weighted = float(block.weight) * result.value
            total = total + weighted
            metrics[f"{block.name}_weight"] = float(block.weight)
            metrics[f"{block.name}_weighted_value"] = float(weighted.detach().item())
            metrics.update({f"{block.name}_{key}": float(value) for key, value in result.metrics.items()})
        metrics["discrepancy_total"] = float(total.detach().item())
        return PathFunctionalDiscrepancyResult(value=total, metrics=metrics)

    def lions_potential(
        self,
        *,
        query_paths: torch.Tensor,
        population_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> torch.Tensor:
        """Weighted Lions potential with the source population held fixed."""

        total = query_paths.new_tensor(0.0)
        for block in tuple(self.blocks):
            potential = getattr(block.discrepancy, "lions_potential", None)
            if not callable(potential):
                raise TypeError(
                    "nested conditional targets require a frozen-law Lions "
                    f"potential; discrepancy block {block.name!r} does not "
                    "provide one"
                )
            total = total + float(block.weight) * potential(
                query_paths=query_paths,
                population_paths=population_paths,
                target_paths=target_paths,
                grid=grid,
            )
        return total

    def analytic_value_and_path_gradient(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> tuple[PathFunctionalDiscrepancyResult, torch.Tensor]:
        """Evaluate every block and assemble its closed-form path derivative."""

        total = generated_paths.new_tensor(0.0)
        gradient = torch.zeros_like(generated_paths)
        metrics: dict[str, float] = {}
        for block in tuple(self.blocks):
            evaluator = getattr(
                block.discrepancy,
                "analytic_value_and_path_gradient",
                None,
            )
            if not callable(evaluator):
                raise TypeError(
                    "analytical path sources are unavailable for discrepancy "
                    f"block {block.name!r} ({type(block.discrepancy).__name__})"
                )
            result, block_gradient = evaluator(
                generated_paths=generated_paths,
                target_paths=target_paths,
                grid=grid,
            )
            weighted = float(block.weight) * result.value
            total = total + weighted
            gradient.add_(float(block.weight) * block_gradient)
            metrics[f"{block.name}_weight"] = float(block.weight)
            metrics[f"{block.name}_weighted_value"] = float(weighted.item())
            metrics.update(
                {
                    f"{block.name}_{key}": float(value)
                    for key, value in result.metrics.items()
                }
            )
        metrics["discrepancy_total"] = float(total.item())
        return PathFunctionalDiscrepancyResult(value=total, metrics=metrics), gradient

    def analytic_lions_potential_and_path_gradient(
        self,
        *,
        query_paths: torch.Tensor,
        population_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble the frozen-law potential and analytical query derivative."""

        total = query_paths.new_tensor(0.0)
        gradient = torch.zeros_like(query_paths)
        for block in tuple(self.blocks):
            evaluator = getattr(
                block.discrepancy,
                "analytic_lions_potential_and_path_gradient",
                None,
            )
            if not callable(evaluator):
                raise TypeError(
                    "analytical frozen-law sources are unavailable for discrepancy "
                    f"block {block.name!r} ({type(block.discrepancy).__name__})"
                )
            value, block_gradient = evaluator(
                query_paths=query_paths,
                population_paths=population_paths,
                target_paths=target_paths,
                grid=grid,
            )
            total = total + float(block.weight) * value
            gradient.add_(float(block.weight) * block_gradient)
        return total, gradient
