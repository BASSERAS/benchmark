from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from deep_mkv_gen_path_dt.discrepancies import (
    MultiMarginalPathFunctionalDiscrepancy,
    PathFunctionalDiscrepancy,
    TimeIndexedWeightedDiscrepancy,
    WeightedDiscrepancy,
)
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid
from path_dt_experiments.residual_persistence import (
    FixedTargetTimeMarginalVolatilityDiscrepancy,
    fit_fixed_target_time_marginal_volatility,
)


@dataclass(frozen=True)
class DatedVolatilityMarginalBuild:
    augmented_discrepancy: MultiMarginalPathFunctionalDiscrepancy
    marginal_only_discrepancy: MultiMarginalPathFunctionalDiscrepancy
    fixed_target_marginals: FixedTargetTimeMarginalVolatilityDiscrepancy
    metadata: dict[str, object]


def multi_marginal_window_endpoint_pairs(
    *,
    num_steps: int,
    endpoint_spacing: int,
    windows: Sequence[int],
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    num_steps = int(num_steps)
    endpoint_spacing = int(endpoint_spacing)
    window_values = tuple(int(value) for value in windows)
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if endpoint_spacing < 1:
        raise ValueError("endpoint_spacing must be >= 1")
    if (
        not window_values
        or tuple(sorted(set(window_values))) != window_values
        or any(value < 1 for value in window_values)
    ):
        raise ValueError("windows must be strictly increasing positive integers")
    endpoints = list(range(endpoint_spacing, num_steps + 1, endpoint_spacing))
    if not endpoints or endpoints[-1] != num_steps:
        endpoints.append(num_steps)
    pairs = tuple(
        (window, endpoint)
        for endpoint in endpoints
        for window in window_values
        if window <= endpoint
    )
    if not pairs:
        raise ValueError("no requested window fits at a marginal endpoint")
    return tuple(endpoints), pairs


def _observed_discrepancy_law(
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
) -> tuple[torch.Tensor, DiscreteTimeGrid, int]:
    indices = grid.observed_point_indices(device=torch.device("cpu"))
    if int(indices.numel()) < 2:
        raise ValueError("the observed grid must contain at least two points")
    differences = indices[1:] - indices[:-1]
    if not bool((differences == differences[0]).all()):
        raise ValueError("dated full-grid windows require uniformly spaced observations")
    observed_every = int(differences[0].item())
    times = grid.time_points(device=torch.device("cpu"), dtype=torch.float64).index_select(
        0,
        indices,
    )
    observed_grid = DiscreteTimeGrid(
        T=float(grid.T),
        num_steps=int(indices.numel()) - 1,
        point_times=tuple(float(value) for value in times.tolist()),
        allow_nonuniform=True,
    )
    return (
        target_paths.index_select(1, indices.to(target_paths.device)),
        observed_grid,
        observed_every,
    )


def build_dated_volatility_marginals(
    *,
    base_discrepancy: PathFunctionalDiscrepancy,
    target_paths: torch.Tensor,
    grid: DiscreteTimeGrid,
    observed_only: bool,
    total_weight: float,
    endpoint_spacing_full_grid: int,
    windows_full_grid: Sequence[int],
) -> DatedVolatilityMarginalBuild:
    """Add dated realized-volatility laws without changing the terminal objective."""

    weight = float(total_weight)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("total_weight must be a positive finite float")
    if target_paths.ndim != 3:
        raise ValueError("target_paths must have shape (B, N + 1, d)")
    if int(target_paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("target_paths must align with grid")
    if bool(observed_only):
        target_used, discrepancy_grid, observed_every = _observed_discrepancy_law(
            target_paths,
            grid=grid,
        )
    else:
        target_used = target_paths
        discrepancy_grid = grid
        observed_every = 1
    endpoint_spacing_full = int(endpoint_spacing_full_grid)
    windows_full = tuple(int(value) for value in windows_full_grid)
    if endpoint_spacing_full % observed_every != 0:
        raise ValueError(
            "endpoint_spacing_full_grid must be divisible by the observation spacing"
        )
    if any(value % observed_every != 0 for value in windows_full):
        raise ValueError("every full-grid window must be divisible by observation spacing")
    endpoint_spacing = endpoint_spacing_full // observed_every
    windows = tuple(value // observed_every for value in windows_full)
    endpoints, pairs = multi_marginal_window_endpoint_pairs(
        num_steps=int(discrepancy_grid.num_steps),
        endpoint_spacing=endpoint_spacing,
        windows=windows,
    )
    fixed_marginals = fit_fixed_target_time_marginal_volatility(
        target_used,
        grid=discrepancy_grid,
        window_endpoint_pairs=pairs,
    )
    pairs_by_endpoint = {
        endpoint: tuple(pair for pair in pairs if pair[1] == endpoint)
        for endpoint in endpoints
    }
    date_weight = weight / float(len(endpoints))
    running_blocks = tuple(
        TimeIndexedWeightedDiscrepancy(
            name=f"volatility_marginal_t{endpoint}",
            endpoint_index=endpoint,
            weight=date_weight,
            discrepancy=fixed_marginals.select_pairs(pairs_by_endpoint[endpoint]),
        )
        for endpoint in endpoints
        if endpoint < int(discrepancy_grid.num_steps)
    )
    terminal_marginal = WeightedDiscrepancy(
        name=f"volatility_marginal_t{int(discrepancy_grid.num_steps)}",
        weight=date_weight,
        discrepancy=fixed_marginals.select_pairs(
            pairs_by_endpoint[int(discrepancy_grid.num_steps)]
        ),
    )
    marginal_only = MultiMarginalPathFunctionalDiscrepancy(
        running_blocks=running_blocks,
        terminal_blocks=(terminal_marginal,),
    )
    augmented = MultiMarginalPathFunctionalDiscrepancy(
        running_blocks=running_blocks,
        terminal_blocks=(
            terminal_marginal,
            WeightedDiscrepancy(
                name="preexisting_complete_path_objective",
                weight=1.0,
                discrepancy=base_discrepancy,
            ),
        ),
    )
    return DatedVolatilityMarginalBuild(
        augmented_discrepancy=augmented,
        marginal_only_discrepancy=marginal_only,
        fixed_target_marginals=fixed_marginals,
        metadata={
            "total_weight": weight,
            "date_weight": date_weight,
            "observed_only": bool(observed_only),
            "observed_every": observed_every,
            "endpoint_spacing_full_grid": endpoint_spacing_full,
            "endpoint_spacing_discrepancy_grid": endpoint_spacing,
            "windows_full_grid": windows_full,
            "windows_discrepancy_grid": windows,
            "endpoints_discrepancy_grid": endpoints,
            "window_endpoint_pairs_discrepancy_grid": pairs,
            "running_endpoint_count": len(running_blocks),
            "terminal_endpoint_count": 1,
        },
    )


__all__ = [
    "DatedVolatilityMarginalBuild",
    "build_dated_volatility_marginals",
    "multi_marginal_window_endpoint_pairs",
]
