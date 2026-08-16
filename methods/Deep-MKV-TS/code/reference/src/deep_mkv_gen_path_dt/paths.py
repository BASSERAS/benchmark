from __future__ import annotations

import torch

from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


def validate_path(x_path: torch.Tensor, *, grid: DiscreteTimeGrid) -> tuple[int, int]:
    if x_path.ndim != 3:
        raise ValueError("x_path must have shape (B, N + 1, d)")
    if int(x_path.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("x_path time dimension must align with the grid")
    return int(x_path.shape[0]), int(x_path.shape[2])


def gather_observed_path(x_path: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
    validate_path(x_path, grid=grid)
    return x_path.index_select(1, grid.observed_point_indices(device=x_path.device))


def scatter_observed_sources(
    observed_sources: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    time_points: int | None = None,
) -> torch.Tensor:
    if observed_sources.ndim != 3:
        raise ValueError("observed_sources must have shape (B, observed_count, d)")
    if int(observed_sources.shape[1]) != int(grid.observed_count):
        raise ValueError("observed_sources must align with observed_indices")
    total_points = int(grid.num_steps) + 1 if time_points is None else int(time_points)
    if total_points < int(grid.num_steps) + 1:
        raise ValueError("time_points cannot be smaller than grid.num_steps + 1")
    sources = observed_sources.new_zeros(
        int(observed_sources.shape[0]),
        total_points,
        int(observed_sources.shape[2]),
    )
    sources[:, grid.observed_point_indices(device=observed_sources.device), :] = observed_sources
    return sources


def future_source_sums(source_path: torch.Tensor) -> torch.Tensor:
    if source_path.ndim != 3:
        raise ValueError("source_path must have shape (B, N + 1, d)")
    future_inclusive = torch.flip(torch.cumsum(torch.flip(source_path, dims=(1,)), dim=1), dims=(1,))
    return future_inclusive[:, 1:, :]

