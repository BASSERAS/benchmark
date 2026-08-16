from __future__ import annotations

import math

import torch

from deep_mkv_gen_path_dt.config import _require_int, _require_optional_int
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


def make_cpu_generator(seed: int | None) -> torch.Generator | None:
    seed = _require_optional_int("seed", seed)
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def derive_stream_seed(*, base_seed: int | None, stream_offset: int) -> int | None:
    base_seed = _require_optional_int("base_seed", base_seed)
    stream_offset = _require_int("stream_offset", stream_offset)
    if base_seed is None:
        return None
    return int((base_seed + 1_000_003 * stream_offset) % (2**31 - 1))


def sample_standard_normals(
    *,
    grid: DiscreteTimeGrid,
    batch_size: int,
    noise_dim: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    batch_size = _require_int("batch_size", batch_size)
    noise_dim = _require_int("noise_dim", noise_dim)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if noise_dim < 1:
        raise ValueError("noise_dim must be >= 1")
    eps = torch.randn(
        batch_size,
        int(grid.num_steps),
        noise_dim,
        generator=make_cpu_generator(seed),
        dtype=dtype if dtype is not None else torch.float32,
        device="cpu",
    )
    return eps.to(device=device)


def sample_brownian_increments(
    *,
    grid: DiscreteTimeGrid,
    batch_size: int,
    noise_dim: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    if not grid.is_uniform:
        raise ValueError("sample_brownian_increments requires a uniform grid")
    return sample_standard_normals(
        grid=grid,
        batch_size=batch_size,
        noise_dim=noise_dim,
        device=device,
        dtype=dtype,
        seed=seed,
    ).mul_(math.sqrt(float(grid.dt)))
