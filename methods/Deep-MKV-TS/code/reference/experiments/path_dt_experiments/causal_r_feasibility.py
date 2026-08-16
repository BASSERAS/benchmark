from __future__ import annotations

import itertools
import math
from typing import Sequence

import torch
import torch.nn as nn


class CausalVolatilityBasisResidualNetwork(nn.Module):
    """Time-local, model-free causal basis for direct perturbations of ``R``.

    The only path inputs are rolling RMS realized volatilities computed from
    returns available at the current time.  Their linear, quadratic, and
    pairwise-product features are standardized once on an unperturbed
    generated bank.  A separate coefficient vector is exposed at each time,
    so this diagnostic bypasses a learned conditional representation without
    using future information or a latent-volatility reference model.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        noise_adjoint_dim: int,
        num_steps: int,
        dt: float,
        windows: Sequence[int] = (4, 8, 16, 32),
        eps: float = 1e-8,
        min_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        if min(int(state_dim), int(noise_adjoint_dim), int(num_steps)) < 1:
            raise ValueError("state_dim, noise_adjoint_dim, and num_steps must be >= 1")
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be a positive finite float")
        window_values = tuple(int(value) for value in windows)
        if (
            len(window_values) == 0
            or any(value < 1 for value in window_values)
            or tuple(sorted(set(window_values))) != window_values
        ):
            raise ValueError("windows must be strictly increasing positive integers")
        if not math.isfinite(float(eps)) or float(eps) <= 0.0:
            raise ValueError("eps must be a positive finite float")
        if not math.isfinite(float(min_scale)) or float(min_scale) <= 0.0:
            raise ValueError("min_scale must be a positive finite float")

        self.state_dim = int(state_dim)
        self.noise_adjoint_dim = int(noise_adjoint_dim)
        self.num_steps = int(num_steps)
        self.dt = float(dt)
        self.windows = window_values
        self.eps = float(eps)
        self.min_scale = float(min_scale)
        self.raw_feature_dim = len(self.windows) * self.state_dim
        self.interaction_pairs = tuple(
            itertools.combinations(range(self.raw_feature_dim), 2)
        )
        self.expanded_feature_dim = (
            2 * self.raw_feature_dim + len(self.interaction_pairs)
        )
        self.feature_dim = 1 + self.expanded_feature_dim

        self.coefficients = nn.Parameter(
            torch.zeros(
                self.num_steps,
                self.feature_dim,
                self.noise_adjoint_dim,
            )
        )
        self.register_buffer(
            "log_rv_center",
            torch.zeros(self.num_steps, self.raw_feature_dim),
        )
        self.register_buffer(
            "log_rv_scale",
            torch.ones(self.num_steps, self.raw_feature_dim),
        )
        self.register_buffer(
            "expanded_center",
            torch.zeros(self.num_steps, self.expanded_feature_dim),
        )
        self.register_buffer(
            "expanded_scale",
            torch.ones(self.num_steps, self.expanded_feature_dim),
        )
        self.register_buffer("calibrated", torch.tensor(False, dtype=torch.bool))

    @property
    def feature_names(self) -> tuple[str, ...]:
        raw_names = tuple(
            f"log_rv_w{window}_asset{asset}"
            for window in self.windows
            for asset in range(self.state_dim)
        )
        interaction_names = tuple(
            f"{raw_names[left]}__x__{raw_names[right]}"
            for left, right in self.interaction_pairs
        )
        return (
            "intercept",
            *(f"linear_{name}" for name in raw_names),
            *(f"quadratic_{name}" for name in raw_names),
            *interaction_names,
        )

    def _log_rolling_rv(self, x_path: torch.Tensor) -> torch.Tensor:
        if x_path.ndim != 3:
            raise ValueError("x_path must have shape (B, N + 1, state_dim)")
        if int(x_path.shape[1]) != self.num_steps + 1:
            raise ValueError("x_path time dimension does not match num_steps")
        if int(x_path.shape[2]) != self.state_dim:
            raise ValueError("x_path state dimension does not match the basis")
        returns = (x_path[:, 1:, :] - x_path[:, :-1, :]) / math.sqrt(self.dt)
        values = []
        for step_index in range(self.num_steps):
            available = step_index
            per_window = []
            for window in self.windows:
                if available == 0:
                    rv = torch.full_like(x_path[:, 0, :], self.eps)
                else:
                    start = max(0, available - int(window))
                    rv = torch.sqrt(
                        returns[:, start:available, :].pow(2).mean(dim=1)
                        + self.eps**2
                    )
                per_window.append(torch.log(rv.clamp_min(self.eps)))
            values.append(torch.cat(per_window, dim=1))
        return torch.stack(values, dim=1)

    def _expanded(self, standardized_log_rv: torch.Tensor) -> torch.Tensor:
        components = [standardized_log_rv, standardized_log_rv.pow(2)]
        if self.interaction_pairs:
            components.append(
                torch.stack(
                    [
                        standardized_log_rv[..., left]
                        * standardized_log_rv[..., right]
                        for left, right in self.interaction_pairs
                    ],
                    dim=-1,
                )
            )
        return torch.cat(components, dim=-1)

    def fit_calibration(self, x_path: torch.Tensor) -> None:
        """Fix feature centers and scales from an unperturbed generated bank."""

        with torch.no_grad():
            log_rv = self._log_rolling_rv(
                x_path.to(device=self.coefficients.device, dtype=self.coefficients.dtype)
            )
            log_center = log_rv.mean(dim=0)
            log_scale = log_rv.std(dim=0, unbiased=False).clamp_min(self.min_scale)
            standardized = (log_rv - log_center.unsqueeze(0)) / log_scale.unsqueeze(0)
            expanded = self._expanded(standardized)
            expanded_center = expanded.mean(dim=0)
            expanded_scale = expanded.std(dim=0, unbiased=False).clamp_min(
                self.min_scale
            )
            self.log_rv_center.copy_(log_center)
            self.log_rv_scale.copy_(log_scale)
            self.expanded_center.copy_(expanded_center)
            self.expanded_scale.copy_(expanded_scale)
            self.calibrated.fill_(True)

    def features(self, x_path: torch.Tensor) -> torch.Tensor:
        log_rv = self._log_rolling_rv(x_path)
        standardized = (
            log_rv - self.log_rv_center.to(log_rv).unsqueeze(0)
        ) / self.log_rv_scale.to(log_rv).unsqueeze(0)
        expanded = self._expanded(standardized)
        normalized = (
            expanded - self.expanded_center.to(expanded).unsqueeze(0)
        ) / self.expanded_scale.to(expanded).unsqueeze(0)
        intercept = torch.ones_like(normalized[..., :1])
        return torch.cat([intercept, normalized], dim=-1)

    def forward(self, x_path: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "bnf,nfq->bnq",
            self.features(x_path),
            self.coefficients,
        )

    def _step_log_rolling_rv(self, points: torch.Tensor) -> torch.Tensor:
        available = int(points.shape[1]) - 1
        if available == 0:
            per_window = [
                torch.full_like(points[:, 0, :], self.eps)
                for _ in self.windows
            ]
        else:
            returns = (points[:, 1:, :] - points[:, :-1, :]) / math.sqrt(self.dt)
            per_window = []
            for window in self.windows:
                start = max(0, available - int(window))
                rv = torch.sqrt(
                    returns[:, start:available, :].pow(2).mean(dim=1)
                    + self.eps**2
                )
                per_window.append(rv)
        return torch.cat(
            [torch.log(value.clamp_min(self.eps)) for value in per_window],
            dim=1,
        )

    def forward_step(
        self,
        *,
        x_t: torch.Tensor,
        hidden: tuple[torch.Tensor, int] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, int]]:
        if x_t.ndim != 2 or int(x_t.shape[1]) != self.state_dim:
            raise ValueError("x_t must have shape (B, state_dim)")
        if hidden is None:
            points = x_t.unsqueeze(1)
            step_index = 0
        else:
            previous_points, step_index = hidden
            points = torch.cat([previous_points, x_t.unsqueeze(1)], dim=1)
        if not 0 <= int(step_index) < self.num_steps:
            raise ValueError("basis rollout step exceeds num_steps")

        log_rv = self._step_log_rolling_rv(points)
        standardized = (
            log_rv - self.log_rv_center[int(step_index)].to(log_rv).unsqueeze(0)
        ) / self.log_rv_scale[int(step_index)].to(log_rv).unsqueeze(0)
        expanded = self._expanded(standardized)
        normalized = (
            expanded
            - self.expanded_center[int(step_index)].to(expanded).unsqueeze(0)
        ) / self.expanded_scale[int(step_index)].to(expanded).unsqueeze(0)
        features = torch.cat([torch.ones_like(normalized[:, :1]), normalized], dim=1)
        output = features @ self.coefficients[int(step_index)]
        return output, (points, int(step_index) + 1)


__all__ = ["CausalVolatilityBasisResidualNetwork"]
