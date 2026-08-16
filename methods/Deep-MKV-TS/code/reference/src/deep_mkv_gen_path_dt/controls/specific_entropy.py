from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from deep_mkv_gen_path_dt.controls.base import (
    HamiltonianControlInputs,
    RunningCostInputs,
    RunningCostResult,
)
from deep_mkv_gen_path_dt.reference import (
    GuyonLekeufackReferenceKernel,
    LocalGaussianReferenceKernel,
)
from deep_mkv_gen_path_dt.causal_reference import (
    CausalVolatilityReferenceKernel,
    ShrunkCausalVolatilityReferenceKernel,
)


class SpecificEntropyDomainError(ValueError):
    """Raised when the specific-entropy Hamiltonian has no finite unconstrained sigma optimum."""


def _require_finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class SpecificEntropyDiagonalControl:
    """
    Diagonal control map for a drift-quadratic and specific-entropy volatility
    running cost around a fitted local reference volatility.

    The packed control is `[alpha, sigma]`, both with shape `(B, d)`.
    """

    dt: float
    reference_kernel: (
        LocalGaussianReferenceKernel
        | GuyonLekeufackReferenceKernel
        | CausalVolatilityReferenceKernel
        | ShrunkCausalVolatilityReferenceKernel
    )
    eta: float = 1.0
    sigma_min: float = 1e-4
    sigma_max: float | None = None
    denominator_eps: float = 1e-8

    def __post_init__(self) -> None:
        dt = _require_finite_float("dt", self.dt)
        eta = _require_finite_float("eta", self.eta)
        sigma_min = _require_finite_float("sigma_min", self.sigma_min)
        sigma_max = None if self.sigma_max is None else _require_finite_float("sigma_max", self.sigma_max)
        denominator_eps = _require_finite_float("denominator_eps", self.denominator_eps)
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        if eta <= 0.0:
            raise ValueError("eta must be > 0")
        if sigma_min <= 0.0:
            raise ValueError("sigma_min must be > 0")
        if sigma_max is not None and sigma_max <= sigma_min:
            raise ValueError("sigma_max must be > sigma_min when provided")
        if denominator_eps <= 0.0:
            raise ValueError("denominator_eps must be > 0")
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
        if int(self.reference_kernel.grid.num_steps) <= 0:
            raise ValueError("reference_kernel must be fitted on a non-empty grid")
        if abs(float(self.reference_kernel.grid.dt) - dt) > 1e-12:
            raise ValueError("dt must match reference_kernel.grid.dt")
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "sigma_min", sigma_min)
        object.__setattr__(self, "sigma_max", sigma_max)
        object.__setattr__(self, "denominator_eps", denominator_eps)

    def _reference_sigma(self, x_prefix: torch.Tensor, *, step_index: int) -> torch.Tensor:
        _, sigma_ref = self.reference_kernel.evaluate(
            x_prefix,
            step_index=int(step_index),
        )
        sigma_ref = sigma_ref.clamp_min(float(self.sigma_min))
        if self.sigma_max is not None:
            sigma_ref = sigma_ref.clamp_max(float(self.sigma_max))
        return sigma_ref

    def _reference_sigmas_all(self, paths: torch.Tensor) -> torch.Tensor:
        evaluate_sigmas_all = getattr(
            self.reference_kernel,
            "evaluate_sigmas_all",
            None,
        )
        if callable(evaluate_sigmas_all):
            sigma_refs = evaluate_sigmas_all(paths)
            sigma_refs = sigma_refs.clamp_min(float(self.sigma_min))
            if self.sigma_max is not None:
                sigma_refs = sigma_refs.clamp_max(float(self.sigma_max))
            return sigma_refs
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

    def running_cost_value_and_path_gradient(
        self,
        inputs: RunningCostInputs,
    ) -> tuple[RunningCostResult, torch.Tensor] | None:
        """Return the running cost and an explicit path derivative together."""

        reference = self.reference_kernel
        local_vjp = getattr(reference, "sigma_path_vjp", None)
        structural_guyon = (
            isinstance(reference, GuyonLekeufackReferenceKernel)
            and reference.activity_update == "structural_variance"
        )
        if not callable(local_vjp) and not structural_guyon:
            return None
        if bool(inputs.differentiate_controls):
            return None
        paths = inputs.paths
        controls = inputs.controls.detach()
        if paths.ndim != 3:
            raise ValueError("running-cost paths must have shape (B, N + 1, d)")
        batch_size, point_count, state_dim = paths.shape
        expected = (int(batch_size), int(point_count) - 1, 2 * int(state_dim))
        if controls.ndim != 3 or tuple(controls.shape) != expected:
            raise ValueError(
                "specific-entropy running-cost controls must have shape "
                f"{expected}"
            )
        if int(inputs.grid.num_steps) != int(point_count) - 1:
            raise ValueError("running-cost paths must align with the supplied grid")
        sigma = controls[..., state_dim:]
        if not bool(torch.isfinite(controls).all().item()):
            raise ValueError("running-cost controls must be finite")
        if bool(torch.any(sigma <= 0.0).item()):
            raise ValueError("specific-entropy running cost requires positive sigma")

        with torch.no_grad():
            kernel_sigmas = reference.evaluate_sigmas_all(paths.detach())
            sigma_refs = kernel_sigmas.clamp_min(float(self.sigma_min))
            outer_active = kernel_sigmas >= float(self.sigma_min)
            if self.sigma_max is not None:
                sigma_refs = sigma_refs.clamp_max(float(self.sigma_max))
                outer_active = outer_active & (
                    kernel_sigmas <= float(self.sigma_max)
                )
            ratio = sigma / sigma_refs
            sigma_cotangent = (
                float(self.dt)
                * float(self.eta)
                / float(batch_size)
                * (1.0 - ratio)
                / sigma_refs
            )
            sigma_cotangent = sigma_cotangent * outer_active.to(
                dtype=paths.dtype
            )
            path_gradient = (
                reference.structural_sigma_path_vjp(
                    paths.detach(),
                    sigma_cotangent,
                )
                if structural_guyon
                else local_vjp(paths.detach(), sigma_cotangent)
            )
            alpha = controls[..., :state_dim]
            drift_cost = 0.5 * alpha.pow(2)
            volatility_cost = float(self.eta) * (
                ratio - 1.0 - torch.log(ratio)
            )
            path_cost = float(self.dt) * (
                drift_cost + volatility_cost
            ).sum(dim=(1, 2))
            value = path_cost.mean()
            result = RunningCostResult(
                value=value,
                metrics={
                    "running_cost_value": float(value.item()),
                    "running_cost_drift_value": float(
                        (
                            float(self.dt)
                            * drift_cost.sum(dim=(1, 2)).mean()
                        ).item()
                    ),
                    "running_cost_volatility_value": float(
                        (
                            float(self.dt)
                            * volatility_cost.sum(dim=(1, 2)).mean()
                        ).item()
                    ),
                    "running_cost_integrand_value": float(
                        (drift_cost + volatility_cost)
                        .sum(dim=(1, 2))
                        .mean()
                        .item()
                    ),
                    "running_cost_sigma_reference_mean": float(
                        sigma_refs.mean().item()
                    ),
                    "running_cost_sigma_ratio_mean": float(ratio.mean().item()),
                },
            )
            return result, path_gradient

    def running_cost_path_gradient(
        self,
        inputs: RunningCostInputs,
    ) -> torch.Tensor | None:
        """Compatibility wrapper returning only the exact fast derivative."""

        combined = self.running_cost_value_and_path_gradient(inputs)
        return None if combined is None else combined[1]

    def _control_from_reference_sigma(
        self,
        *,
        p: torch.Tensor,
        r: torch.Tensor,
        sigma_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the closed-form control with a precomputed reference sigma."""

        if p.ndim != 2:
            raise ValueError("expected_adjoint_next must have shape (B, d)")
        if r.ndim != 2:
            raise ValueError("expected_adjoint_noise_next must have shape (B, d)")
        if tuple(p.shape) != tuple(r.shape):
            raise ValueError("specific-entropy diagonal control expects P and R to have the same shape")
        if tuple(sigma_ref.shape) != tuple(p.shape):
            raise ValueError("reference sigma must match the adjoint shape")
        alpha = -p
        denominator = float(self.eta) / sigma_ref + r / (float(self.dt) ** 0.5)
        if self.sigma_max is not None:
            sigma_unclamped = float(self.eta) / denominator.clamp_min(float(self.denominator_eps))
            sigma = torch.where(
                denominator > float(self.denominator_eps),
                sigma_unclamped,
                torch.full_like(sigma_unclamped, float(self.sigma_max)),
            )
            sigma = sigma.clamp_max(float(self.sigma_max))
        else:
            if bool(torch.any(denominator <= float(self.denominator_eps)).item()):
                raise SpecificEntropyDomainError(
                    "specific-entropy diagonal control has no finite unconstrained sigma optimum; "
                    "provide sigma_max to cap the volatility"
                )
            sigma = float(self.eta) / denominator.clamp_min(float(self.denominator_eps))
        sigma = sigma.clamp_min(float(self.sigma_min))
        return torch.cat([alpha, sigma], dim=1)

    def __call__(self, inputs: HamiltonianControlInputs) -> torch.Tensor:
        sigma_ref = self._reference_sigma(
            inputs.x_prefix,
            step_index=int(inputs.step_index),
        )
        return self._control_from_reference_sigma(
            p=inputs.expected_adjoint_next,
            r=inputs.expected_adjoint_noise_next,
            sigma_ref=sigma_ref,
        )

    def controls_from_moments(
        self,
        *,
        paths: torch.Tensor,
        expected_adjoint_next: torch.Tensor,
        expected_adjoint_noise_next: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the Hamiltonian map at every date of a frozen path bank."""

        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        expected = (
            int(paths.shape[0]),
            int(paths.shape[1]) - 1,
            int(paths.shape[2]),
        )
        if tuple(expected_adjoint_next.shape) != expected:
            raise ValueError(
                "expected_adjoint_next must have shape (B, N, d)"
            )
        if tuple(expected_adjoint_noise_next.shape) != expected:
            raise ValueError(
                "expected_adjoint_noise_next must have shape (B, N, d)"
            )
        if int(paths.shape[1]) - 1 != int(self.reference_kernel.grid.num_steps):
            raise ValueError("paths must align with the reference grid")
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
        """Invert the bounded diagonal Hamiltonian map on a frozen path bank.

        At an active volatility bound the returned ``R`` is the boundary KKT
        representative which maps back to that same bounded control.  This is
        the useful inverse for numerical control-space relaxation.
        """

        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        batch, points, state_dim = paths.shape
        expected = (int(batch), int(points) - 1, 2 * int(state_dim))
        if tuple(controls.shape) != expected:
            raise ValueError(f"controls must have shape {expected}")
        if int(points) - 1 != int(self.reference_kernel.grid.num_steps):
            raise ValueError("paths must align with the reference grid")
        if not bool(torch.isfinite(controls).all().item()):
            raise ValueError("controls must be finite")
        alpha = controls[..., :state_dim]
        sigma = controls[..., state_dim:]
        if bool(torch.any(sigma <= 0.0).item()):
            raise ValueError("volatility controls must be strictly positive")
        sigma_ref = self._reference_sigmas_all(paths)
        p = -alpha
        r = math.sqrt(float(self.dt)) * float(self.eta) * (
            sigma.reciprocal() - sigma_ref.reciprocal()
        )
        return p, r

    def proximal_transition_kl_controls(
        self,
        *,
        paths: torch.Tensor,
        current_controls: torch.Tensor,
        expected_adjoint_next: torch.Tensor,
        expected_adjoint_noise_next: torch.Tensor,
        rho: float,
    ) -> torch.Tensor:
        """Minimize the Hamiltonian plus conditional transition-kernel KL.

        The proposal term is ``KL(p_u || p_current) / rho`` for the diagonal
        Gaussian Euler increments.  It is algorithmic only: at a fixed point
        its gradient vanishes and the original Hamiltonian stationarity
        condition is recovered.
        """

        rho_value = _require_finite_float("rho", rho)
        if rho_value <= 0.0:
            raise ValueError("rho must be positive")
        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        batch, points, state_dim = paths.shape
        moment_shape = (int(batch), int(points) - 1, int(state_dim))
        control_shape = (
            int(batch),
            int(points) - 1,
            2 * int(state_dim),
        )
        if tuple(current_controls.shape) != control_shape:
            raise ValueError(f"current_controls must have shape {control_shape}")
        if tuple(expected_adjoint_next.shape) != moment_shape:
            raise ValueError(
                f"expected_adjoint_next must have shape {moment_shape}"
            )
        if tuple(expected_adjoint_noise_next.shape) != moment_shape:
            raise ValueError(
                "expected_adjoint_noise_next must have shape "
                f"{moment_shape}"
            )
        if not bool(torch.isfinite(current_controls).all().item()):
            raise ValueError("current_controls must be finite")
        current_alpha = current_controls[..., :state_dim]
        current_sigma = current_controls[..., state_dim:]
        if bool(torch.any(current_sigma <= 0.0).item()):
            raise ValueError("current volatility must be strictly positive")

        p = expected_adjoint_next
        r = expected_adjoint_noise_next
        sigma_ref = self._reference_sigmas_all(paths)
        inverse_rho = 1.0 / rho_value
        drift_weight = (
            float(self.dt) * inverse_rho / current_sigma.pow(2)
        )
        alpha = (drift_weight * current_alpha - p) / (1.0 + drift_weight)

        quadratic = inverse_rho / current_sigma.pow(2)
        linear = float(self.eta) / sigma_ref + r / math.sqrt(float(self.dt))
        constant = float(self.eta) + inverse_rho
        discriminant = torch.sqrt(
            linear.pow(2) + 4.0 * quadratic * constant
        )
        positive_linear_root = 2.0 * constant / (
            linear + discriminant
        ).clamp_min(float(self.denominator_eps))
        negative_linear_root = (-linear + discriminant) / (
            2.0 * quadratic
        ).clamp_min(float(self.denominator_eps))
        sigma = torch.where(
            linear >= 0.0,
            positive_linear_root,
            negative_linear_root,
        )
        sigma = sigma.clamp_min(float(self.sigma_min))
        if self.sigma_max is not None:
            sigma = sigma.clamp_max(float(self.sigma_max))
        return torch.cat((alpha, sigma), dim=-1)

    def proximal_transition_kl_control(
        self,
        *,
        inputs: HamiltonianControlInputs,
        current_control: torch.Tensor,
        rho: float,
    ) -> torch.Tensor:
        """Single-date version used during recursive predictor rollouts."""

        rho_value = _require_finite_float("rho", rho)
        if rho_value <= 0.0:
            raise ValueError("rho must be positive")
        p = inputs.expected_adjoint_next
        r = inputs.expected_adjoint_noise_next
        if p.ndim != 2 or tuple(r.shape) != tuple(p.shape):
            raise ValueError("proximal transition moments must match with shape (B, d)")
        state_dim = int(p.shape[1])
        if tuple(current_control.shape) != (
            int(p.shape[0]),
            2 * state_dim,
        ):
            raise ValueError("current_control must have shape (B, 2 * d)")
        current_alpha = current_control[..., :state_dim]
        current_sigma = current_control[..., state_dim:]
        if (
            not bool(torch.isfinite(current_control).all().item())
            or bool(torch.any(current_sigma <= 0.0).item())
        ):
            raise ValueError(
                "current_control must be finite with positive volatility"
            )
        sigma_ref = self._reference_sigma(
            inputs.x_prefix,
            step_index=int(inputs.step_index),
        )
        inverse_rho = 1.0 / rho_value
        drift_weight = (
            float(self.dt) * inverse_rho / current_sigma.pow(2)
        )
        alpha = (drift_weight * current_alpha - p) / (1.0 + drift_weight)
        quadratic = inverse_rho / current_sigma.pow(2)
        linear = float(self.eta) / sigma_ref + r / math.sqrt(float(self.dt))
        constant = float(self.eta) + inverse_rho
        discriminant = torch.sqrt(
            linear.pow(2) + 4.0 * quadratic * constant
        )
        positive_linear_root = 2.0 * constant / (
            linear + discriminant
        ).clamp_min(float(self.denominator_eps))
        negative_linear_root = (-linear + discriminant) / (
            2.0 * quadratic
        ).clamp_min(float(self.denominator_eps))
        sigma = torch.where(
            linear >= 0.0,
            positive_linear_root,
            negative_linear_root,
        ).clamp_min(float(self.sigma_min))
        if self.sigma_max is not None:
            sigma = sigma.clamp_max(float(self.sigma_max))
        return torch.cat((alpha, sigma), dim=-1)

    def transition_kernel_kl(
        self,
        *,
        controls: torch.Tensor,
        current_controls: torch.Tensor,
    ) -> torch.Tensor:
        """Conditional ``KL(p_u || p_current)`` for each path and date."""

        if tuple(controls.shape) != tuple(current_controls.shape):
            raise ValueError("transition controls must have matching shapes")
        if controls.ndim != 3 or int(controls.shape[-1]) % 2 != 0:
            raise ValueError("controls must have shape (B, N, 2 * d)")
        state_dim = int(controls.shape[-1]) // 2
        alpha = controls[..., :state_dim]
        sigma = controls[..., state_dim:]
        current_alpha = current_controls[..., :state_dim]
        current_sigma = current_controls[..., state_dim:]
        if (
            not bool(torch.isfinite(controls).all().item())
            or not bool(torch.isfinite(current_controls).all().item())
            or bool(torch.any(sigma <= 0.0).item())
            or bool(torch.any(current_sigma <= 0.0).item())
        ):
            raise ValueError(
                "transition controls must be finite with positive volatility"
            )
        variance_ratio = sigma.pow(2) / current_sigma.pow(2)
        mean_term = (
            float(self.dt)
            * (alpha - current_alpha).pow(2)
            / current_sigma.pow(2)
        )
        component = 0.5 * (
            variance_ratio
            - 1.0
            - torch.log(variance_ratio)
            + mean_term
        )
        return component.sum(dim=-1)

    def proximal_moment_targets(
        self,
        *,
        paths: torch.Tensor,
        current_controls: torch.Tensor,
        target_adjoint_next: torch.Tensor,
        target_adjoint_noise_next: torch.Tensor,
        alpha_fraction: float,
        log_sigma_fraction: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return moment targets for a proximal step in control coordinates.

        Drift is relaxed linearly and volatility geometrically.  Equivalently,
        the update is the minimizer of a quadratic proximal problem in
        ``(alpha, log(sigma))`` between the current control and the frozen-law
        Hamiltonian target.  Repeated refreshes have the original stationarity
        solutions as fixed points; this method changes only the numerical step.
        """

        alpha_rate = _require_finite_float("alpha_fraction", alpha_fraction)
        sigma_rate = _require_finite_float(
            "log_sigma_fraction", log_sigma_fraction
        )
        if not 0.0 <= alpha_rate <= 1.0:
            raise ValueError("alpha_fraction must lie in [0, 1]")
        if not 0.0 <= sigma_rate <= 1.0:
            raise ValueError("log_sigma_fraction must lie in [0, 1]")
        if alpha_rate == 0.0 and sigma_rate == 0.0:
            raise ValueError("at least one proximal fraction must be positive")

        target_controls = self.controls_from_moments(
            paths=paths,
            expected_adjoint_next=target_adjoint_next,
            expected_adjoint_noise_next=target_adjoint_noise_next,
        )
        if tuple(current_controls.shape) != tuple(target_controls.shape):
            raise ValueError("current_controls must match target controls")
        state_dim = int(paths.shape[2])
        current_alpha = current_controls[..., :state_dim]
        current_sigma = current_controls[..., state_dim:]
        target_alpha = target_controls[..., :state_dim]
        target_sigma = target_controls[..., state_dim:]
        if not bool(torch.isfinite(current_controls).all().item()):
            raise ValueError("current_controls must be finite")
        if bool(torch.any(current_sigma <= 0.0).item()):
            raise ValueError("current volatility must be strictly positive")

        relaxed_alpha = current_alpha + alpha_rate * (
            target_alpha - current_alpha
        )
        relaxed_log_sigma = torch.log(current_sigma) + sigma_rate * (
            torch.log(target_sigma) - torch.log(current_sigma)
        )
        relaxed_sigma = torch.exp(relaxed_log_sigma)
        relaxed_controls = torch.cat((relaxed_alpha, relaxed_sigma), dim=-1)
        relaxed_p, relaxed_r = self.moments_from_controls(
            paths=paths,
            controls=relaxed_controls,
        )
        return relaxed_p, relaxed_r, relaxed_controls, target_controls

    def running_cost(self, inputs: RunningCostInputs) -> RunningCostResult:
        """Evaluate the path-dependent cost whose Hamiltonian this map minimizes.

        The realized controls are held fixed.  Only the reference volatility is
        re-evaluated on ``inputs.paths``, so autograd returns the path partial
        derivative required by the discrete adjoint equation.
        """

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
                "specific-entropy running-cost controls must have shape "
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
            raise ValueError("specific-entropy running cost requires positive sigma")

        sigma_refs = self._reference_sigmas_all(paths)
        ratio = sigma / sigma_refs
        drift_cost = 0.5 * alpha.pow(2)
        volatility_cost = float(self.eta) * (ratio - 1.0 - torch.log(ratio))
        # The control formula uses the Hamiltonian divided by dt.  The actual
        # stage cost in the discrete objective carries the dt factor, as in
        # the paper's 0.5 * |alpha|^2 * dt generative example.
        path_cost = float(self.dt) * (drift_cost + volatility_cost).sum(dim=(1, 2))
        value = path_cost.mean()
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
                "running_cost_sigma_reference_mean": float(sigma_refs.detach().mean().item()),
                "running_cost_sigma_ratio_mean": float(ratio.detach().mean().item()),
            },
        )


@dataclass(frozen=True)
class VolatilityOnlySpecificEntropyDiagonalControl(SpecificEntropyDiagonalControl):
    """Specific-entropy Hamiltonian map on the restricted set ``alpha = 0``.

    ``P`` remains an input because it is part of the backward adjoint system,
    but it is diagnostically passive: the admissible control set contains only
    volatility.  Controls retain the repository-wide packed representation
    ``[alpha, sigma]`` with an identically zero drift coordinate.
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
                "volatility-only controls require the drift coordinate alpha to be exactly zero"
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
        p, r = super().moments_from_controls(paths=paths, controls=controls)
        return torch.zeros_like(p), r

    def proximal_transition_kl_controls(
        self,
        *,
        paths: torch.Tensor,
        current_controls: torch.Tensor,
        expected_adjoint_next: torch.Tensor,
        expected_adjoint_noise_next: torch.Tensor,
        rho: float,
    ) -> torch.Tensor:
        self._require_zero_drift(
            current_controls, state_dim=int(paths.shape[-1])
        )
        controls = super().proximal_transition_kl_controls(
            paths=paths,
            current_controls=current_controls,
            expected_adjoint_next=expected_adjoint_next,
            expected_adjoint_noise_next=expected_adjoint_noise_next,
            rho=rho,
        )
        return self._zero_drift(controls, state_dim=int(paths.shape[-1]))

    def proximal_transition_kl_control(
        self,
        *,
        inputs: HamiltonianControlInputs,
        current_control: torch.Tensor,
        rho: float,
    ) -> torch.Tensor:
        state_dim = int(inputs.expected_adjoint_next.shape[-1])
        self._require_zero_drift(current_control, state_dim=state_dim)
        controls = super().proximal_transition_kl_control(
            inputs=inputs,
            current_control=current_control,
            rho=rho,
        )
        return self._zero_drift(controls, state_dim=state_dim)

    def transition_kernel_kl(
        self,
        *,
        controls: torch.Tensor,
        current_controls: torch.Tensor,
    ) -> torch.Tensor:
        state_dim = int(controls.shape[-1]) // 2
        self._require_zero_drift(controls, state_dim=state_dim)
        self._require_zero_drift(current_controls, state_dim=state_dim)
        return super().transition_kernel_kl(
            controls=controls,
            current_controls=current_controls,
        )

    def running_cost(self, inputs: RunningCostInputs) -> RunningCostResult:
        self._require_zero_drift(
            inputs.controls, state_dim=int(inputs.paths.shape[-1])
        )
        return super().running_cost(inputs)
