from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from deep_mkv_gen_path_dt.controls.base import (
    HamiltonianControlInputs,
    RunningCostInputs,
    RunningCostResult,
)


def _require_positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite float")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return result


@dataclass(frozen=True)
class EntropyBarrierDiagonalControl:
    """Reference-free diagonal volatility control with energy and entropy costs.

    The local running cost for each coordinate is

        0.5 * alpha**2 + 0.5 * kappa * sigma**2 - tau * log(sigma),

    on positive ``sigma``.  Its uncontrolled volatility scale is
    ``sqrt(tau / kappa)`` and does not depend on a fitted prefix kernel.
    """

    dt: float
    kappa: float
    tau: float
    sigma_min: float = 1e-4
    sigma_max: float | None = None

    def __post_init__(self) -> None:
        dt = _require_positive_finite("dt", self.dt)
        kappa = _require_positive_finite("kappa", self.kappa)
        tau = _require_positive_finite("tau", self.tau)
        sigma_min = _require_positive_finite("sigma_min", self.sigma_min)
        sigma_max = None if self.sigma_max is None else _require_positive_finite("sigma_max", self.sigma_max)
        if sigma_max is not None and sigma_max <= sigma_min:
            raise ValueError("sigma_max must be > sigma_min when provided")
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "sigma_min", sigma_min)
        object.__setattr__(self, "sigma_max", sigma_max)

    @property
    def baseline_sigma(self) -> float:
        return math.sqrt(float(self.tau) / float(self.kappa))

    def __call__(self, inputs: HamiltonianControlInputs) -> torch.Tensor:
        p = inputs.expected_adjoint_next
        r = inputs.expected_adjoint_noise_next
        if p.ndim != 2:
            raise ValueError("expected_adjoint_next must have shape (B, d)")
        if r.ndim != 2:
            raise ValueError("expected_adjoint_noise_next must have shape (B, d)")
        if tuple(p.shape) != tuple(r.shape):
            raise ValueError("entropy-barrier diagonal control expects P and R to have the same shape")

        alpha = -p
        linear_coefficient = r / math.sqrt(float(self.dt))
        discriminant_root = torch.hypot(
            linear_coefficient,
            torch.full_like(linear_coefficient, math.sqrt(4.0 * float(self.kappa) * float(self.tau))),
        )
        direct_root = (discriminant_root - linear_coefficient) / (2.0 * float(self.kappa))
        rationalized_root = 2.0 * float(self.tau) / (discriminant_root + linear_coefficient)
        sigma = torch.where(linear_coefficient >= 0.0, rationalized_root, direct_root)
        sigma = sigma.clamp_min(float(self.sigma_min))
        if self.sigma_max is not None:
            sigma = sigma.clamp_max(float(self.sigma_max))
        return torch.cat([alpha, sigma], dim=1)

    def running_cost(self, inputs: RunningCostInputs) -> RunningCostResult:
        """Evaluate the state-independent energy--entropy running cost."""

        paths = inputs.paths
        controls = (
            inputs.controls
            if bool(inputs.differentiate_controls)
            else inputs.controls.detach()
        )
        if paths.ndim != 3:
            raise ValueError("running-cost paths must have shape (B, N + 1, d)")
        if controls.ndim != 3:
            raise ValueError("running-cost controls must have shape (B, N, 2 * d)")
        batch_size, point_count, state_dim = paths.shape
        expected_control_shape = (int(batch_size), int(point_count) - 1, 2 * int(state_dim))
        if tuple(controls.shape) != expected_control_shape:
            raise ValueError(
                "entropy-barrier running-cost controls must have shape "
                f"{expected_control_shape}"
            )
        if int(inputs.grid.num_steps) != int(point_count) - 1:
            raise ValueError("running-cost paths must align with the supplied grid")
        if abs(float(inputs.grid.dt) - float(self.dt)) > 1e-12:
            raise ValueError("running-cost grid.dt must match the control-map dt")

        alpha = controls[..., :state_dim]
        sigma = controls[..., state_dim:]
        if not bool(torch.isfinite(controls).all().item()):
            raise ValueError("running-cost controls must be finite")
        if bool(torch.any(sigma <= 0.0).item()):
            raise ValueError("entropy-barrier running cost requires positive sigma")
        drift_cost = 0.5 * alpha.pow(2)
        volatility_cost = 0.5 * float(self.kappa) * sigma.pow(2) - float(self.tau) * torch.log(sigma)
        value = float(self.dt) * (drift_cost + volatility_cost).sum(dim=(1, 2)).mean()
        return RunningCostResult(
            value=value,
            metrics={
                "running_cost_value": float(value.detach().item()),
                "running_cost_drift_value": float(
                    (float(self.dt) * drift_cost.sum(dim=(1, 2)).mean()).detach().item()
                ),
                "running_cost_volatility_value": float(
                    (float(self.dt) * volatility_cost.sum(dim=(1, 2)).mean()).detach().item()
                ),
                "running_cost_integrand_value": float(
                    (drift_cost + volatility_cost).sum(dim=(1, 2)).mean().detach().item()
                ),
            },
        )


__all__ = ["EntropyBarrierDiagonalControl"]
