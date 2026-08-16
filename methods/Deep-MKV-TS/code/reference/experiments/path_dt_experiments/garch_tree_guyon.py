"""Guyon-reference variant of the finite GARCH-tree verification problem.

This module is deliberately separate from the established constant-reference
tree implementation.  It lets the paper's current volatility reference be
checked on the same exactly enumerable tree without changing existing runs.
"""

from __future__ import annotations

from dataclasses import replace
import math

import torch

from deep_mkv_gen_path_dt import (
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteTimeGrid,
    GuyonLekeufackReferenceKernel,
    fit_guyon_lekeufack_likelihood_reference_kernel,
)
from deep_mkv_gen_path_dt.controls import VolatilityOnlySpecificEntropyDiagonalControl
from path_dt_experiments.discrepancies import build_heston_discrepancy
from path_dt_experiments.dynamics import DriftVolatilityEulerStep
from path_dt_experiments.garch_tree_oracle import (
    ExactGarchTreeProblem,
    GarchTreeConfig,
    _children,
)
from path_dt_experiments.garch_tree_production import (
    ProductionGarchTreeProblem,
    constant_zero_drift_reference,
)


def fit_zero_drift_guyon_tree_reference(
    config: GarchTreeConfig,
    *,
    target_paths: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    calibration_seed: int = 314159,
) -> GuyonLekeufackReferenceKernel:
    """Fit Guyon volatility to tree paths, then lock physical drift to zero."""

    grid = DiscreteTimeGrid(
        T=float(config.horizon) * float(config.dt),
        num_steps=int(config.horizon),
    )
    paths = target_paths.to(device=device, dtype=dtype).unsqueeze(-1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(calibration_seed))
    permutation = torch.randperm(int(paths.shape[0]), generator=generator).to(device)
    paths = paths.index_select(0, permutation)
    fitted = fit_guyon_lekeufack_likelihood_reference_kernel(
        target_paths=paths,
        grid=grid,
        ridge=1e-3,
        variance_ridge=1e-3,
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=float(config.sigma_min),
        sigma_max=float(config.sigma_max),
        calibration_fraction=0.8,
        activity_update="structural_variance",
        optimization_steps=400,
        optimization_batch_size=min(512, int(paths.shape[0])),
        learning_rate=2e-2,
        validation_every=20,
        seed=int(calibration_seed),
    )
    zero_drift = constant_zero_drift_reference(
        grid=grid,
        sigma=float(config.long_run_sigma),
        sigma_min=float(config.sigma_min),
        sigma_max=float(config.sigma_max),
        device=device,
        dtype=dtype,
    )
    return replace(fitted, drift_kernel=zero_drift)


def _reference_tree_start(
    problem: "GuyonProductionGarchTreeProblem",
) -> torch.Tensor:
    """Roll the Guyon reference itself through every node of the finite tree."""

    branching = int(problem.config.branching)
    innovations = problem.innovations
    prefix = torch.zeros(1, 1, 1, device=problem.device, dtype=problem.dtype)
    sigma_levels: list[torch.Tensor] = []
    for step in range(int(problem.config.horizon)):
        _, sigma = problem.reference_kernel.evaluate(prefix, step_index=step)
        node_sigma = sigma[:, 0]
        sigma_levels.append(node_sigma)
        children = _children(
            prefix[:, -1, 0],
            node_sigma,
            dt=float(problem.config.dt),
            innovations=innovations,
        )
        prefix = torch.cat(
            (
                prefix.repeat_interleave(branching, dim=0),
                children.reshape(-1, 1, 1),
            ),
            dim=1,
        )
    return problem.pack(tuple(sigma_levels))


class GuyonProductionGarchTreeProblem(ProductionGarchTreeProblem):
    """Production tree objective with a calibrated Guyon volatility reference."""

    def __init__(
        self,
        config: GarchTreeConfig,
        *,
        lambda_scale: float = 50.0,
        kappa_scale: float = 100.0,
        calibration_seed: int = 314159,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        ExactGarchTreeProblem.__init__(self, config, device=device, dtype=dtype)
        self.lambda_scale = float(lambda_scale)
        self.kappa_scale = float(kappa_scale)
        if self.lambda_scale <= 0.0 or self.kappa_scale < 0.0:
            raise ValueError("production discrepancy scales are invalid")
        self.grid = DiscreteTimeGrid(
            T=float(config.horizon) * float(config.dt),
            num_steps=int(config.horizon),
        )
        self.reference_kernel = fit_zero_drift_guyon_tree_reference(
            config,
            target_paths=self.target.leaf_paths,
            device=self.device,
            dtype=self.dtype,
            calibration_seed=int(calibration_seed),
        )
        self.control_map = VolatilityOnlySpecificEntropyDiagonalControl(
            dt=float(self.grid.dt),
            eta=float(config.eta),
            reference_kernel=self.reference_kernel,
            sigma_min=float(config.sigma_min),
            sigma_max=float(config.sigma_max),
        )
        self.discrepancy = build_heston_discrepancy(
            num_steps=int(config.horizon),
            include_acf=True,
            preset="old_fullv_w0p25",
            lambda_scale=float(lambda_scale),
            kappa_scale=float(kappa_scale),
            include_conditional_volatility=False,
            include_conditional_volatility_correlation=False,
        )
        self._guyon_reference_start = _reference_tree_start(self).detach()

    def reference_start(self) -> torch.Tensor:
        return self._guyon_reference_start.detach().clone()

    def _expand_node_levels(
        self, levels: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        columns = []
        branching = int(self.config.branching)
        horizon = int(self.config.horizon)
        for level, values in enumerate(levels):
            columns.append(
                values.repeat_interleave(branching ** (horizon - level))
            )
        return torch.stack(columns, dim=1).unsqueeze(-1)

    def _reference_sigma_levels(self, solution) -> tuple[torch.Tensor, ...]:
        pathwise = self.reference_kernel.evaluate_sigmas_all(
            solution.leaf_paths.unsqueeze(-1)
        )[..., 0]
        levels = []
        branching = int(self.config.branching)
        horizon = int(self.config.horizon)
        for level in range(horizon):
            nodes = branching**level
            descendants = branching ** (horizon - level)
            grouped = pathwise[:, level].reshape(nodes, descendants)
            if float((grouped - grouped[:, :1]).abs().max().item()) > 1e-10:
                raise RuntimeError("Guyon reference is not nodewise on the exact tree")
            levels.append(grouped[:, 0])
        return tuple(levels)

    def hamiltonian_best_response(self, variables: torch.Tensor) -> torch.Tensor:
        solution = self.solution(variables.detach())
        p_levels, r_levels = self.conditional_adjoint_targets_for_solution(solution)
        controls = self.control_map.controls_from_moments(
            paths=solution.leaf_paths.unsqueeze(-1),
            expected_adjoint_next=self._expand_node_levels(p_levels),
            expected_adjoint_noise_next=self._expand_node_levels(r_levels),
        )
        pathwise_sigma = controls[..., 1]
        branching = int(self.config.branching)
        horizon = int(self.config.horizon)
        levels = []
        for level in range(horizon):
            nodes = branching**level
            descendants = branching ** (horizon - level)
            grouped = pathwise_sigma[:, level].reshape(nodes, descendants)
            levels.append(grouped[:, 0])
        return self.pack(tuple(levels)).detach()

    def exact_adjoint_audit(self, variables: torch.Tensor) -> dict[str, object]:
        """Check the exact adjoint formula using the pathwise Guyon reference."""

        point = variables.detach().clone().requires_grad_(True)
        solution = self.solution(point)
        p_levels, r_levels = self.conditional_adjoint_targets_for_solution(solution)
        reference_levels = self._reference_sigma_levels(solution)
        best_variables = self.hamiltonian_best_response(point)
        best_levels = self.unpack(best_variables)
        formula = []
        log_best_response = []
        for level, (sigma, sigma_ref, r, best) in enumerate(
            zip(solution.sigma_levels, reference_levels, r_levels, best_levels)
        ):
            probability = self.level_node_probability(level)
            physical_gradient = probability * float(self.config.dt) * (
                float(self.config.eta) * (1.0 / sigma_ref - 1.0 / sigma)
                + r / math.sqrt(float(self.config.dt))
            )
            formula.append(physical_gradient * sigma)
            log_best_response.append(torch.log(sigma) - torch.log(best))
        expected = torch.cat(formula)
        direct = torch.autograd.grad(self.objective(point), point)[0]
        relative_error = float(
            torch.linalg.vector_norm((direct - expected).double())
            / torch.linalg.vector_norm(direct.double()).clamp_min(
                torch.finfo(torch.float64).eps
            )
        )
        sigma_flat = torch.cat(solution.sigma_levels)
        lower = float(self.config.sigma_min)
        upper = float(self.config.sigma_max)
        return {
            "P_levels": p_levels,
            "R_levels": r_levels,
            "volatility_log_best_response_rms": float(
                torch.cat(log_best_response).double().pow(2).mean().sqrt().item()
            ),
            "direct_gradient_rms": float(
                direct.double().pow(2).mean().sqrt().item()
            ),
            "independent_adjoint_gradient_relative_error": relative_error,
            "sigma_lower_activity": float(
                (sigma_flat <= lower * (1.0 + 1e-6)).double().mean().item()
            ),
            "sigma_upper_activity": float(
                (sigma_flat >= upper * (1.0 - 1e-6)).double().mean().item()
            ),
            "minimum_sigma_bound_margin": float(
                torch.minimum(sigma_flat - lower, upper - sigma_flat).min().item()
            ),
        }

    def build_model(
        self,
        *,
        hidden_dim: int,
        num_layers: int = 1,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> DiscreteMPModel:
        model = DiscreteMPModel(
            architecture=DiscreteMPArchitectureConfig(
                state_dim=1,
                noise_dim=1,
                hidden_dim=int(hidden_dim),
                num_layers=int(num_layers),
            ),
            grid=self.grid,
            dynamics_step=DriftVolatilityEulerStep(dt=float(self.grid.dt)),
            control_map=self.control_map,
            discrepancy=self.discrepancy,
            init_seed=int(seed),
        )
        model.network.to(device=self.device, dtype=dtype)
        return model


__all__ = [
    "GuyonProductionGarchTreeProblem",
    "fit_zero_drift_guyon_tree_reference",
]
