from __future__ import annotations

from typing import Any

import torch

from deep_mkv_gen_path_dt.discrepancies import CompositePathFunctionalDiscrepancy
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


def _validate_paths(paths: torch.Tensor, *, name: str, grid: DiscreteTimeGrid) -> None:
    if paths.ndim != 3:
        raise ValueError(f"{name} must have shape (B, N + 1, d)")
    if int(paths.shape[0]) < 2:
        raise ValueError(f"{name} must contain at least two paths")
    if int(paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError(f"{name} time dimension must align with grid")


def _diffusion_scaled_paths(paths: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    increments = paths[:, 1:, :] - paths[:, :-1, :]
    mean_increment = increments.mean(dim=0, keepdim=True).detach()
    innovations = increments - mean_increment
    scaled_increments = mean_increment + innovations * torch.exp(log_scales).reshape(1, -1, 1)
    return torch.cat(
        [paths[:, :1, :], paths[:, :1, :] + torch.cumsum(scaled_increments, dim=1)],
        dim=1,
    )


def _region_means(gradient: torch.Tensor) -> dict[str, float]:
    regions = torch.tensor_split(gradient, 3)
    return {
        f"{name}_gradient_mean": float(values.mean().item())
        for name, values in zip(("early", "middle", "late"), regions)
        if int(values.numel()) > 0
    }


def discrepancy_diffusion_scale_sensitivity(
    *,
    discrepancy: CompositePathFunctionalDiscrepancy,
    generated_paths: torch.Tensor,
    target_paths: torch.Tensor,
    grid: DiscreteTimeGrid,
) -> dict[str, Any]:
    """Measure each discrepancy block's local gradient along diffusion scale.

    Generated increments are decomposed into their cross-path mean and centered
    innovations.  A differentiable timewise log scale multiplies only the
    innovations, leaving the empirical drift unchanged.  The resulting
    derivative is a local, path-law diagnostic: it reports whether each block
    asks for more or less diffusion and whether block gradients reinforce or
    cancel one another.
    """

    _validate_paths(generated_paths, name="generated_paths", grid=grid)
    _validate_paths(target_paths, name="target_paths", grid=grid)
    if int(generated_paths.shape[2]) != int(target_paths.shape[2]):
        raise ValueError("generated_paths and target_paths must have the same state dimension")
    target_paths = target_paths.to(device=generated_paths.device, dtype=generated_paths.dtype)

    block_values: dict[str, float] = {}
    weighted_gradients: dict[str, torch.Tensor] = {}
    raw_gradients: dict[str, torch.Tensor] = {}
    for block in tuple(discrepancy.blocks):
        log_scales = torch.zeros(
            int(grid.num_steps),
            device=generated_paths.device,
            dtype=generated_paths.dtype,
            requires_grad=True,
        )
        perturbed = _diffusion_scaled_paths(generated_paths, log_scales)
        result = block.discrepancy(
            generated_paths=perturbed,
            target_paths=target_paths,
            grid=grid,
        )
        gradient = torch.autograd.grad(result.value, log_scales)[0].detach()
        raw_gradients[str(block.name)] = gradient
        weighted_gradients[str(block.name)] = float(block.weight) * gradient
        block_values[str(block.name)] = float(result.value.detach().item())

    total_gradient = torch.stack(tuple(weighted_gradients.values()), dim=0).sum(dim=0)
    total_norm = torch.linalg.vector_norm(total_gradient)
    total_norm_squared = total_norm.pow(2)
    eps = torch.finfo(total_gradient.dtype).eps
    blocks: dict[str, dict[str, float | str]] = {}
    for block in tuple(discrepancy.blocks):
        name = str(block.name)
        raw = raw_gradients[name]
        weighted = weighted_gradients[name]
        weighted_norm = torch.linalg.vector_norm(weighted)
        dot_total = torch.sum(weighted * total_gradient)
        global_gradient = float(weighted.sum().item())
        if abs(global_gradient) <= eps:
            direction = "neutral"
        else:
            direction = "increase" if global_gradient < 0.0 else "decrease"
        blocks[name] = {
            "weight": float(block.weight),
            "value": block_values[name],
            "weighted_value": float(block.weight) * block_values[name],
            "raw_timewise_gradient_rms": float(torch.sqrt(torch.mean(raw.pow(2))).item()),
            "weighted_timewise_gradient_rms": float(torch.sqrt(torch.mean(weighted.pow(2))).item()),
            "weighted_gradient_l2": float(weighted_norm.item()),
            "global_log_diffusion_scale_gradient": global_gradient,
            "local_descent_direction": direction,
            "cosine_with_total_gradient": float(
                (dot_total / (weighted_norm * total_norm).clamp_min(eps)).item()
            ),
            "fraction_of_total_gradient_projection": float(
                (dot_total / total_norm_squared.clamp_min(eps)).item()
            ),
        } | _region_means(weighted)

    total_global_gradient = float(total_gradient.sum().item())
    if abs(total_global_gradient) <= eps:
        total_direction = "neutral"
    else:
        total_direction = "increase" if total_global_gradient < 0.0 else "decrease"
    return {
        "interpretation": {
            "perturbation": "timewise scaling of centered generated increments; mean increments are fixed",
            "gradient_sign": "negative asks gradient descent to increase diffusion; positive asks it to decrease",
            "scope": "local sensitivity of the empirical discrepancy, not a causal attribution of trained parameters",
        },
        "num_generated_paths": int(generated_paths.shape[0]),
        "num_target_paths": int(target_paths.shape[0]),
        "num_steps": int(grid.num_steps),
        "blocks": blocks,
        "total": {
            "timewise_gradient_rms": float(torch.sqrt(torch.mean(total_gradient.pow(2))).item()),
            "gradient_l2": float(total_norm.item()),
            "global_log_diffusion_scale_gradient": total_global_gradient,
            "local_descent_direction": total_direction,
        }
        | _region_means(total_gradient),
    }


__all__ = ["discrepancy_diffusion_scale_sensitivity"]
