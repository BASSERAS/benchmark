"""Fine-grid MP control around a reference fitted only at observed dates.

The generated state and the adjoint network live on the complete fine grid.
The fitted local reference lives on the observed grid and is lifted across the
fine steps inside each observed interval.  Its drift and volatility provide a
fixed baseline during that interval, while the MP volatility correction is
recomputed at every fine step from the current fine-grid causal history.

Path-law discrepancies are handled by ``DiscreteMPModel(observed_only=True)``
and therefore see observed endpoints only.  The running cost below is summed
on every fine step.  Nothing in this module changes the production solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from deep_mkv_gen_path_dt.controls import (
    HamiltonianControlInputs,
    RunningCostInputs,
    RunningCostResult,
)
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid
from deep_mkv_gen_path_dt.reference import LocalGaussianReferenceKernel


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


@dataclass(frozen=True)
class ObservedReferenceFineGridLift:
    """Evaluate an observed-grid reference from a fine-grid path prefix."""

    observed_reference: LocalGaussianReferenceKernel
    fine_grid: DiscreteTimeGrid
    substeps: int = 4

    def __post_init__(self) -> None:
        count = _positive_integer("substeps", self.substeps)
        object.__setattr__(self, "substeps", count)
        observed_grid = self.observed_reference.grid
        if int(self.fine_grid.num_steps) != count * int(observed_grid.num_steps):
            raise ValueError(
                "fine_grid.num_steps must equal substeps * observed reference steps"
            )
        if abs(float(self.fine_grid.T) - float(observed_grid.T)) > 1e-12:
            raise ValueError("fine and observed reference grids must have the same T")
        required_indices = tuple(
            range(0, int(self.fine_grid.num_steps) + 1, count)
        )
        if tuple(int(index) for index in self.fine_grid.observed_indices) != required_indices:
            raise ValueError(
                "fine_grid observed indices must be the block endpoints"
            )

    @property
    def sigma_min(self) -> float:
        return float(self.observed_reference.sigma_min)

    @property
    def sigma_max(self) -> float | None:
        value = self.observed_reference.sigma_max
        return None if value is None else float(value)

    def observed_step(self, fine_step: int) -> int:
        step = int(fine_step)
        if not 0 <= step < int(self.fine_grid.num_steps):
            raise ValueError("fine_step is out of bounds")
        return step // int(self.substeps)

    def observed_prefix(
        self,
        fine_prefix: torch.Tensor,
        *,
        fine_step: int,
    ) -> torch.Tensor:
        if fine_prefix.ndim != 3:
            raise ValueError("fine_prefix must have shape (B, fine_step + 1, d)")
        step = int(fine_step)
        if int(fine_prefix.shape[1]) != step + 1:
            raise ValueError("fine_prefix length must match fine_step")
        block = self.observed_step(step)
        last_observed = block * int(self.substeps)
        indices = torch.arange(
            0,
            last_observed + 1,
            int(self.substeps),
            device=fine_prefix.device,
        )
        return fine_prefix.index_select(1, indices)

    def evaluate(
        self,
        fine_prefix: torch.Tensor,
        *,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.observed_step(int(step_index))
        prefix = self.observed_prefix(fine_prefix, fine_step=int(step_index))
        return self.observed_reference.evaluate(prefix, step_index=block)

    def evaluate_all(self, fine_paths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if fine_paths.ndim != 3:
            raise ValueError("fine_paths must have shape (B, N + 1, d)")
        if int(fine_paths.shape[1]) != int(self.fine_grid.num_steps) + 1:
            raise ValueError("fine_paths must align with fine_grid")
        values = [
            self.evaluate(
                fine_paths[:, : step + 1, :],
                step_index=step,
            )
            for step in range(int(self.fine_grid.num_steps))
        ]
        return (
            torch.stack([value[0] for value in values], dim=1),
            torch.stack([value[1] for value in values], dim=1),
        )

    def summary(self) -> dict[str, float]:
        return self.observed_reference.summary() | {
            "reference_fitted_points": float(
                int(self.observed_reference.grid.num_steps) + 1
            ),
            "fine_grid_points": float(int(self.fine_grid.num_steps) + 1),
            "fine_substeps_per_observation": float(self.substeps),
        }


@dataclass(frozen=True)
class FineGridVolatilityOnlySpecificEntropyControl:
    """Fixed lifted reference drift with a new MP sigma at each fine step."""

    dt: float
    reference_lift: ObservedReferenceFineGridLift
    eta: float = 1.0
    sigma_min: float = 1e-4
    sigma_max: float | None = None
    denominator_eps: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("dt", "eta", "sigma_min", "denominator_eps"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.sigma_max is not None:
            sigma_max = float(self.sigma_max)
            if not math.isfinite(sigma_max) or sigma_max <= float(self.sigma_min):
                raise ValueError("sigma_max must be finite and greater than sigma_min")
            object.__setattr__(self, "sigma_max", sigma_max)
        if abs(float(self.dt) - float(self.reference_lift.fine_grid.dt)) > 1e-12:
            raise ValueError("dt must match the fine grid")

    @property
    def reference_kernel(self) -> ObservedReferenceFineGridLift:
        return self.reference_lift

    def _reference(
        self,
        paths: torch.Tensor,
        *,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        drift, sigma = self.reference_lift.evaluate(paths, step_index=int(step_index))
        sigma = sigma.clamp_min(float(self.sigma_min))
        if self.sigma_max is not None:
            sigma = sigma.clamp_max(float(self.sigma_max))
        return drift, sigma

    def __call__(self, inputs: HamiltonianControlInputs) -> torch.Tensor:
        p = inputs.expected_adjoint_next
        r = inputs.expected_adjoint_noise_next
        if p.ndim != 2 or r.ndim != 2 or tuple(p.shape) != tuple(r.shape):
            raise ValueError("P and R must have the same shape (B, d)")
        reference_drift, reference_sigma = self._reference(
            inputs.x_prefix,
            step_index=int(inputs.step_index),
        )
        denominator = float(self.eta) / reference_sigma + r / math.sqrt(float(self.dt))
        if self.sigma_max is None:
            if bool(torch.any(denominator <= float(self.denominator_eps)).item()):
                raise ValueError("fine-grid specific-entropy denominator is nonpositive")
            sigma = float(self.eta) / denominator.clamp_min(float(self.denominator_eps))
        else:
            unconstrained = float(self.eta) / denominator.clamp_min(
                float(self.denominator_eps)
            )
            sigma = torch.where(
                denominator > float(self.denominator_eps),
                unconstrained,
                torch.full_like(unconstrained, float(self.sigma_max)),
            ).clamp_max(float(self.sigma_max))
        sigma = sigma.clamp_min(float(self.sigma_min))
        # P remains a learned diagnostic, but it cannot alter the physical law.
        return torch.cat((reference_drift, sigma), dim=-1)

    def controls_from_moments(
        self,
        *,
        paths: torch.Tensor,
        expected_adjoint_next: torch.Tensor,
        expected_adjoint_noise_next: torch.Tensor,
    ) -> torch.Tensor:
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
            raise ValueError(f"expected_adjoint_noise_next must have shape {expected}")
        return torch.stack(
            [
                self(
                    HamiltonianControlInputs(
                        step_index=step,
                        x_prefix=paths[:, : step + 1, :],
                        expected_adjoint_next=expected_adjoint_next[:, step, :],
                        expected_adjoint_noise_next=expected_adjoint_noise_next[:, step, :],
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
        if paths.ndim != 3 or controls.ndim != 3:
            raise ValueError("paths and controls must be rank three")
        batch, points, state_dim = paths.shape
        expected = (int(batch), int(points) - 1, 2 * int(state_dim))
        if tuple(controls.shape) != expected:
            raise ValueError(f"controls must have shape {expected}")
        _, sigma_reference = self.reference_lift.evaluate_all(paths)
        sigma = controls[..., state_dim:]
        if bool(torch.any(sigma <= 0.0).item()):
            raise ValueError("volatility controls must be positive")
        p = torch.zeros_like(sigma)
        r = math.sqrt(float(self.dt)) * float(self.eta) * (
            sigma.reciprocal() - sigma_reference.reciprocal()
        )
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
        if int(points) != int(self.reference_lift.fine_grid.num_steps) + 1:
            raise ValueError("running-cost paths must align with the fine grid")
        if not bool(torch.isfinite(controls).all().item()):
            raise ValueError("running-cost controls must be finite")
        sigma = controls[..., state_dim:]
        if not bool(inputs.differentiate_controls):
            sigma = sigma.detach()
        if bool(torch.any(sigma <= 0.0).item()):
            raise ValueError("specific-entropy running cost requires positive sigma")
        reference_drift, reference_sigma = self.reference_lift.evaluate_all(paths)
        reference_sigma = reference_sigma.clamp_min(float(self.sigma_min))
        if self.sigma_max is not None:
            reference_sigma = reference_sigma.clamp_max(float(self.sigma_max))
        ratio = sigma / reference_sigma
        volatility_integrand = float(self.eta) * (ratio - 1.0 - torch.log(ratio))
        value = float(self.dt) * volatility_integrand.sum(dim=(1, 2)).mean()
        physical_drift = controls[..., :state_dim]
        return RunningCostResult(
            value=value,
            metrics={
                "running_cost_value": float(value.detach().item()),
                "running_cost_drift_value": 0.0,
                "running_cost_volatility_value": float(value.detach().item()),
                "running_cost_integrand_value": float(
                    volatility_integrand.sum(dim=(1, 2)).mean().detach().item()
                ),
                "running_cost_reference_drift_mean": float(
                    reference_drift.detach().mean().item()
                ),
                "running_cost_drift_correction_rms": float(
                    torch.sqrt(
                        torch.mean((physical_drift.detach() - reference_drift.detach()).pow(2))
                    ).item()
                ),
                "running_cost_sigma_reference_mean": float(
                    reference_sigma.detach().mean().item()
                ),
                "running_cost_sigma_ratio_mean": float(ratio.detach().mean().item()),
            },
        )


__all__ = [
    "FineGridVolatilityOnlySpecificEntropyControl",
    "ObservedReferenceFineGridLift",
]
