from __future__ import annotations

import math
from typing import Sequence

import torch


def common_noise_path_transition(
    previous_paths: torch.Tensor,
    current_paths: torch.Tensor,
) -> dict[str, float]:
    """Measure a checkpoint transition on path banks sharing the same noise."""

    if previous_paths.ndim != 3 or current_paths.ndim != 3:
        raise ValueError("path banks must have shape (B, N + 1, d)")
    if tuple(previous_paths.shape) != tuple(current_paths.shape):
        raise ValueError("common-noise path banks must have identical shapes")
    if int(previous_paths.shape[1]) < 2:
        raise ValueError("path banks require at least one increment")
    previous = previous_paths.detach().to(dtype=torch.float64)
    current = current_paths.detach().to(device=previous.device, dtype=previous.dtype)
    previous_returns = previous[:, 1:, :] - previous[:, :-1, :]
    current_returns = current[:, 1:, :] - current[:, :-1, :]
    previous_rv = torch.sqrt(torch.mean(previous_returns.pow(2), dim=1))
    current_rv = torch.sqrt(torch.mean(current_returns.pow(2), dim=1))
    previous_terminal = previous[:, -1, :] - previous[:, 0, :]
    current_terminal = current[:, -1, :] - current[:, 0, :]

    def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
        left_flat = left.reshape(-1)
        right_flat = right.reshape(-1)
        left_centered = left_flat - left_flat.mean()
        right_centered = right_flat - right_flat.mean()
        denominator = torch.sqrt(
            torch.sum(left_centered.pow(2)) * torch.sum(right_centered.pow(2))
        )
        if float(denominator.item()) == 0.0:
            return 1.0 if bool(torch.allclose(left_flat, right_flat)) else 0.0
        return float(torch.sum(left_centered * right_centered).div(denominator).item())

    return {
        "path_rms_change": float(torch.sqrt(torch.mean((current - previous).pow(2))).item()),
        "increment_rms_change": float(
            torch.sqrt(torch.mean((current_returns - previous_returns).pow(2))).item()
        ),
        "increment_correlation": correlation(previous_returns, current_returns),
        "realized_volatility_correlation": correlation(previous_rv, current_rv),
        "terminal_return_correlation": correlation(previous_terminal, current_terminal),
        "current_realized_volatility_std": float(current_rv.std(unbiased=False).item()),
        "current_terminal_return_std": float(current_terminal.std(unbiased=False).item()),
    }


def finite_pearson_correlation(
    records: Sequence[dict[str, float]],
    *,
    left: str,
    right: str,
) -> float:
    """Pearson correlation for two finite scalar fields."""

    if len(records) < 2:
        raise ValueError("correlation requires at least two records")
    left_values = torch.tensor([float(record[left]) for record in records], dtype=torch.float64)
    right_values = torch.tensor([float(record[right]) for record in records], dtype=torch.float64)
    if not bool(torch.isfinite(left_values).all() and torch.isfinite(right_values).all()):
        raise ValueError("correlation fields must be finite")
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    denominator = torch.sqrt(
        torch.sum(left_centered.pow(2)) * torch.sum(right_centered.pow(2))
    )
    if float(denominator.item()) == 0.0:
        raise ValueError("correlation fields must have non-zero variance")
    result = float(torch.sum(left_centered * right_centered).div(denominator).item())
    if not math.isfinite(result):
        raise ValueError("correlation must be finite")
    return result


__all__ = ["common_noise_path_transition", "finite_pearson_correlation"]
