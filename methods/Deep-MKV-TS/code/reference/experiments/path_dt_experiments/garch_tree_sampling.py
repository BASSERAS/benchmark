from __future__ import annotations

from dataclasses import replace

import torch

from path_dt_experiments.garch_tree_oracle import GarchTreeConfig
from path_dt_experiments.lifted_collocation_solver import ScenarioCollocationBank


def sample_garch_tree_noise(
    config: GarchTreeConfig,
    *,
    batch_size: int,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample iid shocks from the exact tree's finite innovation law."""

    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        int(config.branching),
        (int(batch_size), int(config.horizon), 1),
        generator=generator,
        device="cpu",
    )
    values = torch.tensor(config.innovations, dtype=dtype, device="cpu")
    return values.index_select(0, indices.reshape(-1)).reshape(indices.shape).to(
        device=torch.device(device), dtype=dtype
    )


def bank_with_garch_tree_noise(
    bank: ScenarioCollocationBank,
    *,
    config: GarchTreeConfig,
    seed: int,
) -> ScenarioCollocationBank:
    """Replace only a scenario bank's Gaussian base shocks by tree shocks."""

    return replace(
        bank,
        base_noise=sample_garch_tree_noise(
            config,
            batch_size=int(bank.initial_indices.numel()),
            seed=int(seed),
            device=torch.device("cpu"),
            dtype=bank.base_noise.dtype,
        ),
        antithetic=False,
    )


def replay_garch_tree_branch_noise(
    *,
    config: GarchTreeConfig,
    bank: ScenarioCollocationBank,
    base_noise: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    """Replay iid conditional futures from the same law as the exact tree."""

    step = int(step_index)
    if not 0 <= step < int(config.horizon):
        raise IndexError("step_index is outside the GARCH horizon")
    batch_size = int(bank.initial_indices.numel())
    expected = (batch_size, int(config.horizon), 1)
    if tuple(base_noise.shape) != expected:
        raise ValueError(f"base_noise must have shape {expected}")
    branches = sample_garch_tree_noise(
        config,
        batch_size=batch_size * int(bank.num_branches),
        seed=int(bank.branch_seed) + int(bank.branch_stream_offset) + step,
        device=base_noise.device,
        dtype=base_noise.dtype,
    ).reshape(
        batch_size,
        int(bank.num_branches),
        int(config.horizon),
        1,
    )
    if step > 0:
        branches[:, :, :step, :] = base_noise[:, None, :step, :]
    return branches


def pool_garch_tree_prefix_targets(
    config: GarchTreeConfig,
    *,
    base_noise: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Average repeated Monte Carlo labels for identical observable prefixes."""

    if base_noise.ndim != 3 or tuple(base_noise.shape[1:]) != (
        int(config.horizon),
        1,
    ):
        raise ValueError("base_noise does not match the GARCH tree")
    if targets.ndim != 3 or tuple(targets.shape[:2]) != tuple(
        base_noise.shape[:2]
    ):
        raise ValueError("targets must align with base_noise")
    innovations = torch.tensor(
        config.innovations, device=base_noise.device, dtype=base_noise.dtype
    )
    digits = torch.argmin(
        torch.abs(base_noise.squeeze(-1).unsqueeze(-1) - innovations), dim=-1
    )
    reconstructed = innovations.index_select(0, digits.reshape(-1)).reshape(
        digits.shape
    )
    if not bool(
        torch.isclose(
            reconstructed,
            base_noise.squeeze(-1),
            rtol=0.0,
            atol=10.0 * torch.finfo(base_noise.dtype).eps,
        )
        .all()
        .item()
    ):
        raise ValueError("base_noise contains shocks outside the tree law")

    pooled = torch.empty_like(targets)
    prefix_id = torch.zeros(
        int(base_noise.shape[0]), device=base_noise.device, dtype=torch.long
    )
    for step in range(int(config.horizon)):
        if step > 0:
            prefix_id = (
                prefix_id * int(config.branching) + digits[:, step - 1]
            )
        unique, inverse = torch.unique(prefix_id, sorted=True, return_inverse=True)
        sums = torch.zeros(
            int(unique.numel()),
            int(targets.shape[-1]),
            device=targets.device,
            dtype=targets.dtype,
        )
        counts = torch.zeros(
            int(unique.numel()), 1, device=targets.device, dtype=targets.dtype
        )
        sums.index_add_(0, inverse.to(targets.device), targets[:, step])
        counts.index_add_(
            0,
            inverse.to(targets.device),
            torch.ones(
                int(targets.shape[0]), 1, device=targets.device, dtype=targets.dtype
            ),
        )
        pooled[:, step] = (sums / counts).index_select(
            0, inverse.to(targets.device)
        )
    return pooled


__all__ = [
    "bank_with_garch_tree_noise",
    "pool_garch_tree_prefix_targets",
    "replay_garch_tree_branch_noise",
    "sample_garch_tree_noise",
]
