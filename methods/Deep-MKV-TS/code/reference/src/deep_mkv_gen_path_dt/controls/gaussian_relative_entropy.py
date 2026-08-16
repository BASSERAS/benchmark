from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from deep_mkv_gen_path_dt.causal_reference import (
    CausalVolatilityReferenceKernel,
    ShrunkCausalVolatilityReferenceKernel,
)
from deep_mkv_gen_path_dt.controls.base import (
    HamiltonianControlInputs,
    RunningCostInputs,
    RunningCostResult,
)
from deep_mkv_gen_path_dt.reference import (
    GuyonLekeufackReferenceKernel,
    LocalGaussianReferenceKernel,
)


ReferenceKernel = (
    LocalGaussianReferenceKernel
    | GuyonLekeufackReferenceKernel
    | CausalVolatilityReferenceKernel
    | ShrunkCausalVolatilityReferenceKernel
)


def _require_positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite float")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return result


@dataclass(frozen=True)
class GaussianRelativeEntropyDiagonalControl:
    """Diagonal control with Gaussian scale-relative entropy.

    The running integrand is
    ``0.5*drift_penalty*alpha**2 + 0.5*eta*(q-1-log(q))`` coordinatewise,
    where ``q=(sigma/sigma_ref)**2``.  The default drift penalty is one.  A
    non-unit value permits an exact linear change of state units: for
    ``X = state_scale * Y``, use ``drift_penalty = state_scale**2`` in the
    dimensionless ``Y`` coordinates.  The scale term is the relative entropy
    between centered Gaussian increments.  It is strictly convex and
    quadratic, rather than linear, at large volatility ratios.
    """

    dt: float
    reference_kernel: ReferenceKernel
    eta: float = 1.0
    drift_penalty: float = 1.0
    sigma_min: float = 1e-4
    sigma_max: float | None = None

    def __post_init__(self) -> None:
        dt = _require_positive_finite("dt", self.dt)
        eta = _require_positive_finite("eta", self.eta)
        drift_penalty = _require_positive_finite(
            "drift_penalty", self.drift_penalty
        )
        sigma_min = _require_positive_finite("sigma_min", self.sigma_min)
        sigma_max = (
            None
            if self.sigma_max is None
            else _require_positive_finite("sigma_max", self.sigma_max)
        )
        if sigma_max is not None and sigma_max <= sigma_min:
            raise ValueError("sigma_max must be > sigma_min when provided")
        if not isinstance(
            self.reference_kernel,
            (
                LocalGaussianReferenceKernel,
                GuyonLekeufackReferenceKernel,
                CausalVolatilityReferenceKernel,
                ShrunkCausalVolatilityReferenceKernel,
            ),
        ):
            raise ValueError("reference_kernel must be a fitted Gaussian reference kernel")
        if abs(float(self.reference_kernel.grid.dt) - dt) > 1e-12:
            raise ValueError("dt must match reference_kernel.grid.dt")
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "drift_penalty", drift_penalty)
        object.__setattr__(self, "sigma_min", sigma_min)
        object.__setattr__(self, "sigma_max", sigma_max)

    def _reference_sigma(
        self,
        x_prefix: torch.Tensor,
        *,
        step_index: int,
    ) -> torch.Tensor:
        _, sigma_ref = self.reference_kernel.evaluate(
            x_prefix,
            step_index=int(step_index),
        )
        sigma_ref = sigma_ref.clamp_min(float(self.sigma_min))
        if self.sigma_max is not None:
            sigma_ref = sigma_ref.clamp_max(float(self.sigma_max))
        return sigma_ref

    def _reference_sigmas_all(self, paths: torch.Tensor) -> torch.Tensor:
        evaluate_all = getattr(self.reference_kernel, "evaluate_all", None)
        if callable(evaluate_all):
            _, sigma_refs = evaluate_all(paths)
            sigma_refs = sigma_refs.clamp_min(float(self.sigma_min))
            if self.sigma_max is not None:
                sigma_refs = sigma_refs.clamp_max(float(self.sigma_max))
            return sigma_refs
        return torch.stack(
            [
                self._reference_sigma(
                    paths[:, : step + 1, :],
                    step_index=step,
                )
                for step in range(int(paths.shape[1]) - 1)
            ],
            dim=1,
        )

    def __call__(self, inputs: HamiltonianControlInputs) -> torch.Tensor:
        p = inputs.expected_adjoint_next
        r = inputs.expected_adjoint_noise_next
        if p.ndim != 2 or r.ndim != 2 or tuple(p.shape) != tuple(r.shape):
            raise ValueError("Gaussian-relative-entropy control expects matching (B, d) P/R")
        alpha = -p / float(self.drift_penalty)
        sigma_ref = self._reference_sigma(
            inputs.x_prefix,
            step_index=int(inputs.step_index),
        )
        linear = r / math.sqrt(float(self.dt))
        discriminant_root = torch.hypot(
            linear,
            2.0 * float(self.eta) / sigma_ref,
        )
        # Stable forms of the positive root of
        # eta*sigma**2/sigma_ref**2 + linear*sigma - eta = 0.
        direct = (
            sigma_ref.pow(2)
            * (discriminant_root - linear)
            / (2.0 * float(self.eta))
        )
        rationalized = 2.0 * float(self.eta) / (
            discriminant_root + linear
        )
        sigma = torch.where(linear >= 0.0, rationalized, direct)
        sigma = sigma.clamp_min(float(self.sigma_min))
        if self.sigma_max is not None:
            sigma = sigma.clamp_max(float(self.sigma_max))
        return torch.cat((alpha, sigma), dim=1)

    def controls_from_moments(
        self,
        *,
        paths: torch.Tensor,
        expected_adjoint_next: torch.Tensor,
        expected_adjoint_noise_next: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the bounded Hamiltonian map on a frozen path bank."""

        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        expected = (
            int(paths.shape[0]),
            int(paths.shape[1]) - 1,
            int(paths.shape[2]),
        )
        if tuple(expected_adjoint_next.shape) != expected:
            raise ValueError(f"expected_adjoint_next must have shape {expected}")
        if tuple(expected_adjoint_noise_next.shape) != expected:
            raise ValueError(
                f"expected_adjoint_noise_next must have shape {expected}"
            )
        return torch.stack(
            [
                self(
                    HamiltonianControlInputs(
                        step_index=step,
                        x_prefix=paths[:, : step + 1, :],
                        expected_adjoint_next=expected_adjoint_next[:, step, :],
                        expected_adjoint_noise_next=(
                            expected_adjoint_noise_next[:, step, :]
                        ),
                    )
                )
                for step in range(int(paths.shape[1]) - 1)
            ],
            dim=1,
        )

    def moments_from_controls(
        self,
        *,
        paths: torch.Tensor,
        controls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the bounded-KKT moment representative of physical controls."""

        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        batch, points, state_dim = paths.shape
        expected = (int(batch), int(points) - 1, 2 * int(state_dim))
        if tuple(controls.shape) != expected:
            raise ValueError(f"controls must have shape {expected}")
        alpha = controls[..., :state_dim]
        sigma = controls[..., state_dim:]
        if not bool(torch.isfinite(controls).all().item()) or bool(
            torch.any(sigma <= 0.0).item()
        ):
            raise ValueError("controls must be finite with positive volatility")
        sigma_ref = self._reference_sigmas_all(paths)
        p = -float(self.drift_penalty) * alpha
        r = -math.sqrt(float(self.dt)) * float(self.eta) * (
            sigma / sigma_ref.pow(2) - sigma.reciprocal()
        )
        return p, r

    def transition_kernel_kl(
        self,
        *,
        controls: torch.Tensor,
        current_controls: torch.Tensor,
    ) -> torch.Tensor:
        """Conditional Gaussian transition KL, independent of running-cost scale."""

        if tuple(controls.shape) != tuple(current_controls.shape):
            raise ValueError("transition controls must have matching shapes")
        if controls.ndim != 3 or int(controls.shape[-1]) % 2 != 0:
            raise ValueError("controls must have shape (B, N, 2*d)")
        state_dim = int(controls.shape[-1]) // 2
        alpha = controls[..., :state_dim]
        sigma = controls[..., state_dim:]
        current_alpha = current_controls[..., :state_dim]
        current_sigma = current_controls[..., state_dim:]
        if bool(torch.any(sigma <= 0.0).item()) or bool(
            torch.any(current_sigma <= 0.0).item()
        ):
            raise ValueError("transition volatilities must be positive")
        variance_ratio = sigma.pow(2) / current_sigma.pow(2)
        mean_term = (
            float(self.dt)
            * (alpha - current_alpha).pow(2)
            / current_sigma.pow(2)
        )
        return 0.5 * (
            variance_ratio - 1.0 - torch.log(variance_ratio) + mean_term
        ).sum(dim=-1)

    def running_cost(self, inputs: RunningCostInputs) -> RunningCostResult:
        paths = inputs.paths
        controls = (
            inputs.controls
            if bool(inputs.differentiate_controls)
            else inputs.controls.detach()
        )
        if paths.ndim != 3 or controls.ndim != 3:
            raise ValueError("running-cost paths and controls must be rank three")
        batch_size, point_count, state_dim = paths.shape
        expected = (batch_size, point_count - 1, 2 * state_dim)
        if tuple(controls.shape) != expected:
            raise ValueError(f"controls must have shape {expected}")
        if int(inputs.grid.num_steps) != int(point_count) - 1:
            raise ValueError("running-cost paths must align with the supplied grid")
        if abs(float(inputs.grid.dt) - float(self.dt)) > 1e-12:
            raise ValueError("running-cost grid.dt must match the control-map dt")
        if not bool(torch.isfinite(controls).all().item()):
            raise ValueError("running-cost controls must be finite")
        alpha = controls[..., :state_dim]
        sigma = controls[..., state_dim:]
        if bool(torch.any(sigma <= 0.0).item()):
            raise ValueError("Gaussian scale relative entropy requires positive sigma")
        sigma_refs = self._reference_sigmas_all(paths)
        scale_ratio = sigma / sigma_refs
        variance_ratio = scale_ratio.pow(2)
        drift_cost = 0.5 * float(self.drift_penalty) * alpha.pow(2)
        volatility_cost = 0.5 * float(self.eta) * (
            variance_ratio - 1.0 - torch.log(variance_ratio)
        )
        value = float(self.dt) * (
            drift_cost + volatility_cost
        ).sum(dim=(1, 2)).mean()
        return RunningCostResult(
            value=value,
            metrics={
                "running_cost_value": float(value.detach().item()),
                "running_cost_drift_value": float(
                    (float(self.dt) * drift_cost.sum(dim=(1, 2)).mean())
                    .detach()
                    .item()
                ),
                "running_cost_volatility_value": float(
                    (float(self.dt) * volatility_cost.sum(dim=(1, 2)).mean())
                    .detach()
                    .item()
                ),
                "running_cost_integrand_value": float(
                    (drift_cost + volatility_cost)
                    .sum(dim=(1, 2))
                    .mean()
                    .detach()
                    .item()
                ),
                "running_cost_sigma_reference_mean": float(
                    sigma_refs.detach().mean().item()
                ),
                "running_cost_sigma_ratio_mean": float(
                    scale_ratio.detach().mean().item()
                ),
            },
        )


@dataclass(frozen=True)
class VolatilityOnlyGaussianRelativeEntropyDiagonalControl(
    GaussianRelativeEntropyDiagonalControl
):
    """Gaussian-relative-entropy control on the restricted set ``alpha = 0``.

    The full P/R backward system is unchanged, but the P channel is passive in
    the Hamiltonian map.  This is the zero-controlled-drift analogue of
    :class:`VolatilityOnlySpecificEntropyDiagonalControl`.
    """

    @staticmethod
    def _zero_drift(controls: torch.Tensor, *, state_dim: int) -> torch.Tensor:
        return torch.cat(
            (torch.zeros_like(controls[..., :state_dim]), controls[..., state_dim:]),
            dim=-1,
        )

    @staticmethod
    def _require_zero_drift(controls: torch.Tensor, *, state_dim: int) -> None:
        alpha = controls[..., :state_dim]
        if not bool(torch.all(alpha == 0.0).item()):
            raise ValueError(
                "volatility-only controls require the drift coordinate alpha "
                "to be exactly zero"
            )

    def __call__(self, inputs: HamiltonianControlInputs) -> torch.Tensor:
        controls = super().__call__(inputs)
        return self._zero_drift(
            controls, state_dim=int(inputs.expected_adjoint_next.shape[-1])
        )

    def controls_from_moments(
        self,
        *,
        paths: torch.Tensor,
        expected_adjoint_next: torch.Tensor,
        expected_adjoint_noise_next: torch.Tensor,
    ) -> torch.Tensor:
        controls = super().controls_from_moments(
            paths=paths,
            expected_adjoint_next=expected_adjoint_next,
            expected_adjoint_noise_next=expected_adjoint_noise_next,
        )
        return self._zero_drift(controls, state_dim=int(paths.shape[-1]))

    def moments_from_controls(
        self,
        *,
        paths: torch.Tensor,
        controls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_zero_drift(controls, state_dim=int(paths.shape[-1]))
        _, r = super().moments_from_controls(paths=paths, controls=controls)
        return torch.zeros_like(r), r

    def running_cost(self, inputs: RunningCostInputs) -> RunningCostResult:
        self._require_zero_drift(
            inputs.controls, state_dim=int(inputs.paths.shape[-1])
        )
        return super().running_cost(inputs)


__all__ = [
    "GaussianRelativeEntropyDiagonalControl",
    "VolatilityOnlyGaussianRelativeEntropyDiagonalControl",
]
