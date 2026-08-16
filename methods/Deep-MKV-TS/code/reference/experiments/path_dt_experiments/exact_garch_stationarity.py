from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import torch

from path_dt_experiments.garch_tree_production import ProductionGarchTreeProblem


@dataclass(frozen=True)
class ExactGarchStationarityEvaluation:
    """Exact volatility MP stationarity system on a complete GARCH tree."""

    variables: torch.Tensor
    pointwise_residual: torch.Tensor
    projected_residual: torch.Tensor
    weighted_objective_gradient: torch.Tensor
    p_levels: tuple[torch.Tensor, ...]
    r_levels: tuple[torch.Tensor, ...]
    metrics: dict[str, float]


@dataclass(frozen=True)
class FiniteDifferenceJacobian:
    value: torch.Tensor
    base_residual: torch.Tensor
    metrics: dict[str, float]


def _nodewise_reference_sigma(
    problem: ProductionGarchTreeProblem,
    *,
    leaf_paths: torch.Tensor,
    level: int,
) -> torch.Tensor:
    _, sigma_pathwise = problem.reference_kernel.evaluate(
        x_prefix=leaf_paths[:, : level + 1].unsqueeze(-1),
        step_index=int(level),
    )
    nodes = int(problem.config.branching) ** int(level)
    descendants = int(problem.config.branching) ** (
        int(problem.config.horizon) - int(level)
    )
    grouped = sigma_pathwise[:, 0].reshape(nodes, descendants)
    disagreement = float((grouped - grouped[:, :1]).abs().max().item())
    if disagreement > 1e-12:
        raise RuntimeError("the causal reference is not nodewise on the exact tree")
    return grouped[:, 0]


def exact_garch_mp_stationarity(
    problem: ProductionGarchTreeProblem,
    variables: torch.Tensor,
    *,
    gradient_audit: bool = False,
) -> ExactGarchStationarityEvaluation:
    """Evaluate exact conditional-adjoint Hamiltonian stationarity.

    The returned pointwise residual removes the tree-node probability and time
    weight from the objective gradient.  This row scaling does not change its
    zeros and prevents late tree levels from being numerically ignored.
    """

    if variables.ndim != 1 or int(variables.numel()) != problem.variable_count:
        raise ValueError("variables have the wrong dimension")
    point = variables.detach().to(device=problem.device, dtype=problem.dtype)
    solution = problem.solution(point)
    p_levels, r_levels = problem.conditional_adjoint_targets_for_solution(solution)
    pointwise_levels = []
    weighted_levels = []
    projected_levels = []
    lower = math.log(float(problem.config.sigma_min))
    upper = math.log(float(problem.config.sigma_max))
    dt = float(problem.config.dt)
    eta = float(problem.config.eta)
    level_sizes = [
        int(problem.config.branching) ** index
        for index in range(int(problem.config.horizon))
    ]
    log_sigma_levels = point.split(level_sizes)
    for level, (log_sigma, sigma, r) in enumerate(
        zip(log_sigma_levels, solution.sigma_levels, r_levels)
    ):
        sigma_ref = _nodewise_reference_sigma(
            problem,
            leaf_paths=solution.leaf_paths,
            level=level,
        ).to(sigma)
        residual = sigma * (
            eta * (1.0 / sigma_ref - 1.0 / sigma)
            + r / math.sqrt(dt)
        )
        weighted = problem.level_node_probability(level) * dt * residual
        projected = residual.clone()
        projected = torch.where(
            (log_sigma <= lower + 1e-10) & (projected > 0.0),
            torch.zeros_like(projected),
            projected,
        )
        projected = torch.where(
            (log_sigma >= upper - 1e-10) & (projected < 0.0),
            torch.zeros_like(projected),
            projected,
        )
        pointwise_levels.append(residual)
        weighted_levels.append(weighted)
        projected_levels.append(projected)

    pointwise = torch.cat(pointwise_levels).detach()
    weighted = torch.cat(weighted_levels).detach()
    projected = torch.cat(projected_levels).detach()
    gradient_relative_error = math.nan
    if gradient_audit:
        audit_point = point.detach().clone().requires_grad_(True)
        direct = torch.autograd.grad(problem.objective(audit_point), audit_point)[0]
        gradient_relative_error = float(
            torch.linalg.vector_norm((direct.detach() - weighted).double())
            / torch.linalg.vector_norm(direct.detach().double()).clamp_min(1e-15)
        )
    return ExactGarchStationarityEvaluation(
        variables=point,
        pointwise_residual=pointwise,
        projected_residual=projected,
        weighted_objective_gradient=weighted,
        p_levels=p_levels,
        r_levels=r_levels,
        metrics={
            "conditional_expectations_exact": 1.0,
            "neural_estimator_used": 0.0,
            "pointwise_residual_rms": float(
                pointwise.double().pow(2).mean().sqrt().item()
            ),
            "projected_residual_rms": float(
                projected.double().pow(2).mean().sqrt().item()
            ),
            "projected_residual_max_abs": float(
                projected.double().abs().max().item()
            ),
            "weighted_gradient_rms": float(
                weighted.double().pow(2).mean().sqrt().item()
            ),
            "direct_gradient_relative_error": gradient_relative_error,
        },
    )


def finite_difference_stationarity_jacobian(
    problem: ProductionGarchTreeProblem,
    variables: torch.Tensor,
    *,
    epsilon: float = 1e-4,
    progress: Callable[[int, int], None] | None = None,
) -> FiniteDifferenceJacobian:
    """Central finite-difference Jacobian of the exact projected MP residual."""

    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    point = variables.detach().to(device=problem.device, dtype=problem.dtype)
    base = exact_garch_mp_stationarity(problem, point).projected_residual
    dimension = int(point.numel())
    lower = math.log(float(problem.config.sigma_min))
    upper = math.log(float(problem.config.sigma_max))
    columns = []
    started = time.perf_counter()
    for coordinate in range(dimension):
        plus = point.clone()
        minus = point.clone()
        plus[coordinate] = min(float(plus[coordinate].item()) + epsilon, upper)
        minus[coordinate] = max(float(minus[coordinate].item()) - epsilon, lower)
        width = float((plus[coordinate] - minus[coordinate]).item())
        if width <= 0.0:
            raise RuntimeError("finite-difference coordinate has zero feasible width")
        plus_residual = exact_garch_mp_stationarity(
            problem, plus
        ).projected_residual
        minus_residual = exact_garch_mp_stationarity(
            problem, minus
        ).projected_residual
        columns.append((plus_residual - minus_residual) / width)
        if progress is not None:
            progress(coordinate + 1, dimension)
    jacobian = torch.stack(columns, dim=1)
    elapsed = time.perf_counter() - started
    singular_values = torch.linalg.svdvals(jacobian.double())
    maximum = singular_values.max().clamp_min(1e-30)
    positive = singular_values[singular_values > maximum * 1e-12]
    condition = (
        float((positive.max() / positive.min()).item())
        if int(positive.numel()) > 0
        else math.inf
    )
    return FiniteDifferenceJacobian(
        value=jacobian.detach(),
        base_residual=base.detach(),
        metrics={
            "epsilon": float(epsilon),
            "dimension": float(dimension),
            "wall_seconds": float(elapsed),
            "spectral_norm": float(singular_values.max().item()),
            "minimum_retained_singular_value": (
                float(positive.min().item())
                if int(positive.numel()) > 0
                else 0.0
            ),
            "condition_number_1e12_cutoff": condition,
            "numerical_rank_1e12_cutoff": float(positive.numel()),
        },
    )


__all__ = [
    "ExactGarchStationarityEvaluation",
    "FiniteDifferenceJacobian",
    "exact_garch_mp_stationarity",
    "finite_difference_stationarity_jacobian",
]
