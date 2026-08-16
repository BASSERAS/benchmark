from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from deep_mkv_gen_path_dt.controls.base import (
    HamiltonianControlInputs,
    RunningCostInputs,
    RunningCostResult,
)
from deep_mkv_gen_path_dt.controls.gaussian_relative_entropy import (
    GaussianRelativeEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.controls.specific_entropy import (
    SpecificEntropyDiagonalControl,
)


class _ReferenceDriftMixin:
    """Evaluate the fitted physical drift used as the control origin."""

    reference_kernel: object

    def _reference_drift(
        self,
        x_prefix: torch.Tensor,
        *,
        step_index: int,
    ) -> torch.Tensor:
        drift, _ = self.reference_kernel.evaluate(  # type: ignore[attr-defined]
            x_prefix,
            step_index=int(step_index),
        )
        return drift

    def _reference_drifts_all(self, paths: torch.Tensor) -> torch.Tensor:
        evaluate_drifts_all = getattr(
            self.reference_kernel,
            "evaluate_drifts_all",
            None,
        )
        if callable(evaluate_drifts_all):
            return evaluate_drifts_all(paths)
        evaluate_all = getattr(self.reference_kernel, "evaluate_all", None)
        if callable(evaluate_all):
            drifts, _ = evaluate_all(paths)
            return drifts
        return torch.stack(
            [
                self._reference_drift(
                    paths[:, : step + 1, :],
                    step_index=step,
                )
                for step in range(int(paths.shape[1]) - 1)
            ],
            dim=1,
        )


@dataclass(frozen=True)
class ReferenceDriftSpecificEntropyDiagonalControl(
    _ReferenceDriftMixin,
    SpecificEntropyDiagonalControl,
):
    """Specific entropy in correction coordinates around fitted drift.

    The physical drift is ``b_ref + delta_alpha``.  With
    ``allow_drift_correction=False`` the admissible set fixes
    ``delta_alpha=0`` while retaining the complete P/R backward targets.
    """

    allow_drift_correction: bool = True

    def __call__(self, inputs: HamiltonianControlInputs) -> torch.Tensor:
        state_dim = int(inputs.expected_adjoint_next.shape[-1])
        reference_drift, sigma_ref = self.reference_kernel.evaluate(  # type: ignore[attr-defined]
            inputs.x_prefix,
            step_index=int(inputs.step_index),
        )
        sigma_ref = sigma_ref.clamp_min(float(self.sigma_min))
        if self.sigma_max is not None:
            sigma_ref = sigma_ref.clamp_max(float(self.sigma_max))
        correction_control = self._control_from_reference_sigma(
            p=inputs.expected_adjoint_next,
            r=inputs.expected_adjoint_noise_next,
            sigma_ref=sigma_ref,
        )
        correction = correction_control[..., :state_dim]
        physical_drift = reference_drift + (
            correction if bool(self.allow_drift_correction) else torch.zeros_like(correction)
        )
        return torch.cat((physical_drift, correction_control[..., state_dim:]), dim=-1)

    def running_cost_value_and_path_gradient(
        self,
        inputs: RunningCostInputs,
    ) -> tuple[RunningCostResult, torch.Tensor] | None:
        """Exact structural-Guyon fast path in drift-correction coordinates."""

        if bool(inputs.differentiate_controls):
            return None
        paths = inputs.paths
        controls = inputs.controls.detach()
        if paths.ndim != 3 or controls.ndim != 3:
            raise ValueError("running-cost paths and controls must be rank three")
        batch, points, state_dim = paths.shape
        expected = (int(batch), int(points) - 1, 2 * int(state_dim))
        if tuple(controls.shape) != expected:
            raise ValueError(f"controls must have shape {expected}")
        physical_drift = controls[..., :state_dim]
        if bool(self.allow_drift_correction):
            reference_drift = self._reference_drifts_all(paths.detach())
            drift_correction = (physical_drift - reference_drift).detach()
        else:
            # The restricted admissible set fixes the correction coordinate
            # to zero.  Rollout controls already contain the physical fitted
            # drift, so re-running every prefix regression here would only
            # reconstruct a quantity that is zero by construction.
            reference_drift = physical_drift
            drift_correction = torch.zeros_like(physical_drift)
        correction_controls = torch.cat(
            (drift_correction, controls[..., state_dim:]),
            dim=-1,
        )
        combined = super().running_cost_value_and_path_gradient(
            RunningCostInputs(
                paths=paths,
                controls=correction_controls,
                grid=inputs.grid,
                parameters=inputs.parameters,
                differentiate_controls=False,
            )
        )
        if combined is None:
            return None
        result, gradient = combined
        metrics = dict(result.metrics)
        metrics["running_cost_reference_drift_mean"] = float(
            reference_drift.mean().item()
        )
        metrics["running_cost_drift_correction_rms"] = float(
            torch.sqrt(torch.mean(drift_correction.pow(2))).item()
        )
        return RunningCostResult(value=result.value, metrics=metrics), gradient

    def moments_from_controls(
        self,
        *,
        paths: torch.Tensor,
        controls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if paths.ndim != 3 or controls.ndim != 3:
            raise ValueError("paths and controls must be rank three")
        state_dim = int(paths.shape[-1])
        reference_drift = self._reference_drifts_all(paths)
        correction_controls = torch.cat(
            (controls[..., :state_dim] - reference_drift, controls[..., state_dim:]),
            dim=-1,
        )
        p, r = super().moments_from_controls(
            paths=paths,
            controls=correction_controls,
        )
        if not bool(self.allow_drift_correction):
            p = torch.zeros_like(p)
        return p, r

    def running_cost(self, inputs: RunningCostInputs) -> RunningCostResult:
        paths = inputs.paths
        controls = inputs.controls
        if paths.ndim != 3 or controls.ndim != 3:
            raise ValueError("running-cost paths and controls must be rank three")
        batch, points, state_dim = paths.shape
        expected = (int(batch), int(points) - 1, 2 * int(state_dim))
        if tuple(controls.shape) != expected:
            raise ValueError(f"controls must have shape {expected}")
        if not bool(torch.isfinite(controls).all().item()):
            raise ValueError("running-cost controls must be finite")
        physical_drift = controls[..., :state_dim]
        sigma = controls[..., state_dim:]
        if not bool(inputs.differentiate_controls):
            physical_drift = physical_drift.detach()
            sigma = sigma.detach()
        if bool(torch.any(sigma <= 0.0).item()):
            raise ValueError("specific-entropy running cost requires positive sigma")
        reference_drift = self._reference_drifts_all(paths)
        sigma_ref = self._reference_sigmas_all(paths)
        drift_correction = physical_drift - reference_drift
        # The admissible drift coordinate is delta_alpha, not physical drift.
        # During the SMP partial path derivative, delta_alpha is held fixed;
        # otherwise subtracting b_ref from a detached physical drift creates a
        # spurious running-cost contribution to P.
        if not bool(inputs.differentiate_controls):
            drift_correction = drift_correction.detach()
        ratio = sigma / sigma_ref
        drift_cost = 0.5 * drift_correction.pow(2)
        volatility_cost = float(self.eta) * (ratio - 1.0 - torch.log(ratio))
        value = float(self.dt) * (
            drift_cost + volatility_cost
        ).sum(dim=(1, 2)).mean()
        return RunningCostResult(
            value=value,
            metrics={
                "running_cost_value": float(value.detach().item()),
                "running_cost_drift_value": float(
                    (float(self.dt) * drift_cost.sum(dim=(1, 2)).mean()).detach().item()
                ),
                "running_cost_volatility_value": float(
                    (float(self.dt) * volatility_cost.sum(dim=(1, 2)).mean())
                    .detach()
                    .item()
                ),
                "running_cost_integrand_value": float(
                    (drift_cost + volatility_cost).sum(dim=(1, 2)).mean().detach().item()
                ),
                "running_cost_reference_drift_mean": float(
                    reference_drift.detach().mean().item()
                ),
                "running_cost_drift_correction_rms": float(
                    torch.sqrt(torch.mean(drift_correction.detach().pow(2))).item()
                ),
                "running_cost_sigma_reference_mean": float(
                    sigma_ref.detach().mean().item()
                ),
                "running_cost_sigma_ratio_mean": float(ratio.detach().mean().item()),
            },
        )


@dataclass(frozen=True)
class ReferenceDriftGaussianRelativeEntropyDiagonalControl(
    _ReferenceDriftMixin,
    GaussianRelativeEntropyDiagonalControl,
):
    """Gaussian relative entropy in correction coordinates around fitted drift."""

    allow_drift_correction: bool = True

    def __call__(self, inputs: HamiltonianControlInputs) -> torch.Tensor:
        correction_control = super().__call__(inputs)
        state_dim = int(inputs.expected_adjoint_next.shape[-1])
        reference_drift = self._reference_drift(
            inputs.x_prefix,
            step_index=int(inputs.step_index),
        )
        correction = correction_control[..., :state_dim]
        physical_drift = reference_drift + (
            correction if bool(self.allow_drift_correction) else torch.zeros_like(correction)
        )
        return torch.cat((physical_drift, correction_control[..., state_dim:]), dim=-1)

    def moments_from_controls(
        self,
        *,
        paths: torch.Tensor,
        controls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if paths.ndim != 3 or controls.ndim != 3:
            raise ValueError("paths and controls must be rank three")
        state_dim = int(paths.shape[-1])
        reference_drift = self._reference_drifts_all(paths)
        correction_controls = torch.cat(
            (controls[..., :state_dim] - reference_drift, controls[..., state_dim:]),
            dim=-1,
        )
        p, r = super().moments_from_controls(
            paths=paths,
            controls=correction_controls,
        )
        if not bool(self.allow_drift_correction):
            p = torch.zeros_like(p)
        return p, r

    def running_cost(self, inputs: RunningCostInputs) -> RunningCostResult:
        paths = inputs.paths
        controls = inputs.controls
        if paths.ndim != 3 or controls.ndim != 3:
            raise ValueError("running-cost paths and controls must be rank three")
        batch, points, state_dim = paths.shape
        expected = (int(batch), int(points) - 1, 2 * int(state_dim))
        if tuple(controls.shape) != expected:
            raise ValueError(f"controls must have shape {expected}")
        if not bool(torch.isfinite(controls).all().item()):
            raise ValueError("running-cost controls must be finite")
        physical_drift = controls[..., :state_dim]
        sigma = controls[..., state_dim:]
        if not bool(inputs.differentiate_controls):
            physical_drift = physical_drift.detach()
            sigma = sigma.detach()
        if bool(torch.any(sigma <= 0.0).item()):
            raise ValueError("Gaussian relative entropy requires positive sigma")
        reference_drift = self._reference_drifts_all(paths)
        sigma_ref = self._reference_sigmas_all(paths)
        drift_correction = physical_drift - reference_drift
        if not bool(inputs.differentiate_controls):
            drift_correction = drift_correction.detach()
        variance_ratio = (sigma / sigma_ref).pow(2)
        drift_cost = 0.5 * float(self.drift_penalty) * drift_correction.pow(2)
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
                    (float(self.dt) * drift_cost.sum(dim=(1, 2)).mean()).detach().item()
                ),
                "running_cost_volatility_value": float(
                    (float(self.dt) * volatility_cost.sum(dim=(1, 2)).mean())
                    .detach()
                    .item()
                ),
                "running_cost_integrand_value": float(
                    (drift_cost + volatility_cost).sum(dim=(1, 2)).mean().detach().item()
                ),
                "running_cost_reference_drift_mean": float(
                    reference_drift.detach().mean().item()
                ),
                "running_cost_drift_correction_rms": float(
                    torch.sqrt(torch.mean(drift_correction.detach().pow(2))).item()
                ),
                "running_cost_sigma_reference_mean": float(
                    sigma_ref.detach().mean().item()
                ),
                "running_cost_sigma_ratio_mean": float(
                    torch.sqrt(variance_ratio.detach()).mean().item()
                ),
            },
        )


__all__ = [
    "ReferenceDriftGaussianRelativeEntropyDiagonalControl",
    "ReferenceDriftSpecificEntropyDiagonalControl",
]
