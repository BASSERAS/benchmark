from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from deep_mkv_gen_path_dt.controls import HamiltonianControlInputs, HamiltonianControlMap


@dataclass(frozen=True)
class AdjointComponentAblationControl:
    """Intervene on the P/R inputs of an existing Hamiltonian control map."""

    base_control: HamiltonianControlMap
    zero_p: bool = False
    zero_r: bool = False

    def __post_init__(self) -> None:
        if not callable(self.base_control):
            raise ValueError("base_control must be callable")
        if not bool(self.zero_p) and not bool(self.zero_r):
            raise ValueError("at least one of zero_p or zero_r must be enabled")

    def __call__(self, inputs: HamiltonianControlInputs) -> torch.Tensor:
        return self.base_control(
            HamiltonianControlInputs(
                step_index=int(inputs.step_index),
                x_prefix=inputs.x_prefix,
                expected_adjoint_next=(
                    torch.zeros_like(inputs.expected_adjoint_next)
                    if bool(self.zero_p)
                    else inputs.expected_adjoint_next
                ),
                expected_adjoint_noise_next=(
                    torch.zeros_like(inputs.expected_adjoint_noise_next)
                    if bool(self.zero_r)
                    else inputs.expected_adjoint_noise_next
                ),
                parameters=inputs.parameters,
            )
        )


def _correlation(left: torch.Tensor, right: torch.Tensor, *, eps: float = 1e-12) -> float:
    left_flat = left.detach().double().reshape(-1)
    right_flat = right.detach().double().reshape(-1)
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(right_centered)
    if float(denominator.item()) <= float(eps):
        return float("nan")
    return float(torch.dot(left_centered, right_centered).item() / denominator.item())


def specific_entropy_r_effect_on_realized_prefixes(
    *,
    base_control: HamiltonianControlMap,
    rollout,
    state_dim: int,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Measure the algebraic R contribution while holding realized prefixes fixed."""
    state_width = int(state_dim)
    if state_width < 1:
        raise ValueError("state_dim must be >= 1")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and > 0")
    if rollout.paths.ndim != 3 or rollout.controls.ndim != 3:
        raise ValueError("rollout paths and controls must be three-dimensional")
    if int(rollout.controls.shape[-1]) < 2 * state_width:
        raise ValueError("rollout controls do not contain diagonal alpha and sigma blocks")

    r_zero_control = AdjointComponentAblationControl(base_control=base_control, zero_r=True)
    counterfactual_controls = []
    for step_index in range(int(rollout.controls.shape[1])):
        counterfactual_controls.append(
            r_zero_control(
                HamiltonianControlInputs(
                    step_index=int(step_index),
                    x_prefix=rollout.paths[:, : step_index + 1, :],
                    expected_adjoint_next=rollout.expected_adjoint_next[:, step_index, :],
                    expected_adjoint_noise_next=(
                        rollout.expected_adjoint_noise_next[:, step_index, :]
                    ),
                )
            )
        )
    r_zero_controls = torch.stack(counterfactual_controls, dim=1)
    sigma_full = rollout.controls[..., state_width : 2 * state_width]
    sigma_r_zero = r_zero_controls[..., state_width : 2 * state_width]
    difference = sigma_full - sigma_r_zero
    relative_difference = difference.abs() / sigma_r_zero.abs().clamp_min(float(eps))
    ratios = sigma_full / sigma_r_zero.clamp_min(float(eps))
    ratio_quantiles = torch.quantile(
        ratios.detach().double().reshape(-1),
        torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64, device=ratios.device),
    )
    r_values = rollout.expected_adjoint_noise_next
    return {
        "r_rms": float(torch.sqrt(torch.mean(r_values.detach().double().pow(2))).item()),
        "sigma_full_mean": float(sigma_full.detach().double().mean().item()),
        "sigma_r_zero_mean": float(sigma_r_zero.detach().double().mean().item()),
        "sigma_difference_mean": float(difference.detach().double().mean().item()),
        "sigma_difference_abs_mean": float(difference.detach().double().abs().mean().item()),
        "sigma_difference_rms": float(torch.sqrt(torch.mean(difference.detach().double().pow(2))).item()),
        "sigma_relative_difference_abs_mean": float(relative_difference.detach().double().mean().item()),
        "sigma_relative_difference_gt_1pct_fraction": float((relative_difference > 0.01).double().mean().item()),
        "sigma_relative_difference_gt_5pct_fraction": float((relative_difference > 0.05).double().mean().item()),
        "sigma_relative_difference_gt_10pct_fraction": float((relative_difference > 0.10).double().mean().item()),
        "sigma_full_to_r_zero_ratio_q05": float(ratio_quantiles[0].item()),
        "sigma_full_to_r_zero_ratio_q50": float(ratio_quantiles[1].item()),
        "sigma_full_to_r_zero_ratio_q95": float(ratio_quantiles[2].item()),
        "sigma_full_r_zero_correlation": _correlation(sigma_full, sigma_r_zero, eps=float(eps)),
    }
