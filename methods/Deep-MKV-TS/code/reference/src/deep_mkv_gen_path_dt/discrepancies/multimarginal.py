from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from deep_mkv_gen_path_dt.config import _require_finite_float, _require_int
from deep_mkv_gen_path_dt.discrepancies.base import (
    PathFunctionalDiscrepancy,
    PathFunctionalDiscrepancyResult,
)
from deep_mkv_gen_path_dt.discrepancies.composite import WeightedDiscrepancy
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


@dataclass(frozen=True)
class TimeIndexedWeightedDiscrepancy:
    """A law cost injected at one non-terminal date of the MP recursion."""

    name: str
    endpoint_index: int
    weight: float
    discrepancy: PathFunctionalDiscrepancy

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        endpoint_index = _require_int("endpoint_index", self.endpoint_index)
        if endpoint_index < 1:
            raise ValueError("endpoint_index must be >= 1")
        weight = _require_finite_float("weight", self.weight)
        if weight < 0.0:
            raise ValueError("weight must be >= 0")
        object.__setattr__(self, "endpoint_index", endpoint_index)
        object.__setattr__(self, "weight", weight)


def _prefix_grid(grid: DiscreteTimeGrid, *, endpoint_index: int) -> DiscreteTimeGrid:
    endpoint = int(endpoint_index)
    if not 1 <= endpoint <= int(grid.num_steps):
        raise ValueError("endpoint_index must lie on the supplied grid")
    times = grid.time_points(device="cpu", dtype=torch.float64)[: endpoint + 1]
    return DiscreteTimeGrid(
        T=float(times[-1].item()),
        num_steps=endpoint,
        point_times=tuple(float(value) for value in times.tolist()),
        allow_nonuniform=True,
    )


@dataclass(frozen=True)
class MultiMarginalPathFunctionalDiscrepancy:
    """Discrete-time running law costs plus terminal law costs.

    A running block at index ``k`` is evaluated only on ``X_{0:k}``.  Its
    autograd source therefore has support no later than ``k``.  The model's
    existing reverse cumulative source sum turns these dated Lions-gradient
    sources into the jump terms of the discrete path-dependent adjoint.  This
    is the numerical counterpart of a cost

    ``sum_k f_k(pr_k # Law(X)) + g(Law(X))``.
    """

    running_blocks: Sequence[TimeIndexedWeightedDiscrepancy] = ()
    terminal_blocks: Sequence[WeightedDiscrepancy] = ()

    def __post_init__(self) -> None:
        running = tuple(self.running_blocks)
        terminal = tuple(self.terminal_blocks)
        if len(running) + len(terminal) == 0:
            raise ValueError("at least one running or terminal block is required")
        names = tuple(block.name for block in running) + tuple(
            block.name for block in terminal
        )
        if len(set(names)) != len(names):
            raise ValueError("multi-marginal block names must be unique")
        object.__setattr__(self, "running_blocks", running)
        object.__setattr__(self, "terminal_blocks", terminal)

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        if generated_paths.ndim != 3 or target_paths.ndim != 3:
            raise ValueError("generated_paths and target_paths must have shape (B, N + 1, d)")
        expected_points = int(grid.num_steps) + 1
        if int(generated_paths.shape[1]) != expected_points or int(
            target_paths.shape[1]
        ) != expected_points:
            raise ValueError("multi-marginal paths must align with the supplied grid")
        if int(generated_paths.shape[2]) != int(target_paths.shape[2]):
            raise ValueError("generated and target state dimensions must match")

        total = generated_paths.new_tensor(0.0)
        metrics: dict[str, float] = {
            "multi_marginal": 1.0,
            "multi_marginal_running_block_count": float(len(self.running_blocks)),
            "multi_marginal_terminal_block_count": float(len(self.terminal_blocks)),
            "multi_marginal_running_endpoint_count": float(
                len({block.endpoint_index for block in self.running_blocks})
            ),
        }
        for block in self.running_blocks:
            endpoint = int(block.endpoint_index)
            if endpoint >= int(grid.num_steps):
                raise ValueError(
                    "running block endpoint_index must be strictly before the terminal grid point"
                )
            prefix_grid = _prefix_grid(grid, endpoint_index=endpoint)
            result = block.discrepancy(
                generated_paths=generated_paths[:, : endpoint + 1, :],
                target_paths=target_paths[:, : endpoint + 1, :],
                grid=prefix_grid,
            )
            weighted = float(block.weight) * result.value
            total = total + weighted
            prefix = f"running_{block.name}"
            metrics[f"{prefix}_endpoint_index"] = float(endpoint)
            metrics[f"{prefix}_endpoint_time"] = float(prefix_grid.T)
            metrics[f"{prefix}_weight"] = float(block.weight)
            metrics[f"{prefix}_weighted_value"] = float(weighted.detach().item())
            metrics.update(
                {f"{prefix}_{key}": float(value) for key, value in result.metrics.items()}
            )

        for block in self.terminal_blocks:
            result = block.discrepancy(
                generated_paths=generated_paths,
                target_paths=target_paths,
                grid=grid,
            )
            weighted = float(block.weight) * result.value
            total = total + weighted
            prefix = f"terminal_{block.name}"
            metrics[f"{prefix}_weight"] = float(block.weight)
            metrics[f"{prefix}_weighted_value"] = float(weighted.detach().item())
            metrics.update(
                {f"{prefix}_{key}": float(value) for key, value in result.metrics.items()}
            )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("multi-marginal discrepancy is non-finite")
        metrics["multi_marginal_total"] = float(total.detach().item())
        return PathFunctionalDiscrepancyResult(value=total, metrics=metrics)


__all__ = [
    "MultiMarginalPathFunctionalDiscrepancy",
    "TimeIndexedWeightedDiscrepancy",
]
