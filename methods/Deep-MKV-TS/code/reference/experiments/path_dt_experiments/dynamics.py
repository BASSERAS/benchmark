from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from deep_mkv_gen_path_dt.controls import DynamicsStepInputs


@dataclass(frozen=True)
class DriftVolatilityEulerStep:
    """
    Euler step for controls packed as ``[alpha, sigma]``.

    For state dimension ``d`` and noise dimension ``d``, the control must have
    shape ``(B, 2 * d)``.  The first half is the drift control and the second
    half is the diagonal volatility control.
    """

    dt: float

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")

    def __call__(self, inputs: DynamicsStepInputs) -> torch.Tensor:
        x_t = inputs.x_prefix[:, -1, :]
        state_dim = int(x_t.shape[-1])
        if int(inputs.noise_next.shape[-1]) != state_dim:
            raise ValueError("DriftVolatilityEulerStep expects noise_dim == state_dim")
        if inputs.control.ndim != 2 or int(inputs.control.shape[-1]) != 2 * state_dim:
            raise ValueError("control must have shape (B, 2 * state_dim)")
        alpha = inputs.control[:, :state_dim]
        sigma = inputs.control[:, state_dim:]
        return x_t + alpha * float(self.dt) + sigma * math.sqrt(float(self.dt)) * inputs.noise_next

