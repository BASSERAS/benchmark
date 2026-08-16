from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from deep_mkv_gen_path_dt import (
    ConditionalAdjointTargets,
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteTimeGrid,
    LocalGaussianReferenceKernel,
)
from deep_mkv_gen_path_dt.controls import (
    RunningCostInputs,
    VolatilityOnlySpecificEntropyDiagonalControl,
)
from path_dt_experiments.discrepancies import build_heston_discrepancy
from path_dt_experiments.dynamics import DriftVolatilityEulerStep
from path_dt_experiments.garch_tree_oracle import (
    ExactGarchTreeProblem,
    GarchTreeConfig,
    GarchTreeSolution,
    _leaf_paths,
)


def constant_zero_drift_reference(
    *,
    grid: DiscreteTimeGrid,
    sigma: float,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> LocalGaussianReferenceKernel:
    """Construct a production reference kernel with zero drift and fixed sigma."""

    if not 0.0 < float(sigma_min) <= float(sigma) <= float(sigma_max):
        raise ValueError("reference sigma must lie inside its bounds")
    drift_coefficients = []
    log_variance_coefficients = []
    feature_means = []
    feature_stds = []
    for step in range(int(grid.num_steps)):
        width = step + 1
        drift_coefficients.append(
            torch.zeros(width + 1, 1, device=device, dtype=dtype)
        )
        log_variance = torch.zeros(
            width + 1, 1, device=device, dtype=dtype
        )
        log_variance[0, 0] = 2.0 * math.log(float(sigma))
        log_variance_coefficients.append(log_variance)
        feature_means.append(
            torch.zeros(1, width, device=device, dtype=dtype)
        )
        feature_stds.append(
            torch.ones(1, width, device=device, dtype=dtype)
        )
    return LocalGaussianReferenceKernel(
        grid=grid,
        drift_coefficients=tuple(drift_coefficients),
        log_variance_coefficients=tuple(log_variance_coefficients),
        feature_means=tuple(feature_means),
        feature_stds=tuple(feature_stds),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
    )


def build_production_garch_components(
    config: GarchTreeConfig,
    *,
    lambda_scale: float = 50.0,
    kappa_scale: float = 100.0,
    hidden_dim: int = 96,
    num_layers: int = 1,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[
    DiscreteMPModel,
    LocalGaussianReferenceKernel,
    VolatilityOnlySpecificEntropyDiagonalControl,
]:
    """Instantiate the actual production Deep MKV model for the GARCH tree."""

    device = torch.device(device)
    grid = DiscreteTimeGrid(
        T=float(config.horizon) * float(config.dt),
        num_steps=int(config.horizon),
    )
    reference = constant_zero_drift_reference(
        grid=grid,
        sigma=float(config.long_run_sigma),
        sigma_min=float(config.sigma_min),
        sigma_max=float(config.sigma_max),
        device=device,
        dtype=dtype,
    )
    control = VolatilityOnlySpecificEntropyDiagonalControl(
        dt=float(grid.dt),
        eta=float(config.eta),
        reference_kernel=reference,
        sigma_min=float(config.sigma_min),
        sigma_max=float(config.sigma_max),
    )
    discrepancy = build_heston_discrepancy(
        num_steps=int(config.horizon),
        include_acf=True,
        preset="old_fullv_w0p25",
        lambda_scale=float(lambda_scale),
        kappa_scale=float(kappa_scale),
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )
    model = DiscreteMPModel(
        architecture=DiscreteMPArchitectureConfig(
            state_dim=1,
            noise_dim=1,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
        ),
        grid=grid,
        dynamics_step=DriftVolatilityEulerStep(dt=float(grid.dt)),
        control_map=control,
        discrepancy=discrepancy,
        init_seed=int(seed),
    )
    model.network.to(device=device, dtype=dtype)
    return model, reference, control


class ProductionGarchTreeProblem(ExactGarchTreeProblem):
    """Exact tree using the production Deep MKV discrepancy and running cost."""

    def __init__(
        self,
        config: GarchTreeConfig,
        *,
        lambda_scale: float = 50.0,
        kappa_scale: float = 100.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(config, device=device, dtype=dtype)
        self.lambda_scale = float(lambda_scale)
        self.kappa_scale = float(kappa_scale)
        if self.lambda_scale <= 0.0 or self.kappa_scale < 0.0:
            raise ValueError("production discrepancy scales are invalid")
        model, reference, control = build_production_garch_components(
            config,
            lambda_scale=self.lambda_scale,
            kappa_scale=self.kappa_scale,
            hidden_dim=1,
            num_layers=1,
            seed=0,
            device=self.device,
            dtype=self.dtype,
        )
        self.grid = model.grid
        self.discrepancy = model.discrepancy
        self.reference_kernel = reference
        self.control_map = control
        del model

    def controls_for_solution(
        self, solution: GarchTreeSolution
    ) -> torch.Tensor:
        leaves = int(self.config.leaf_count)
        branching = int(self.config.branching)
        sigma_columns = []
        for level, sigma in enumerate(solution.sigma_levels):
            repeat = branching ** (int(self.config.horizon) - level)
            expanded = sigma.repeat_interleave(repeat)
            if int(expanded.numel()) != leaves:
                raise RuntimeError("tree-control expansion is inconsistent")
            sigma_columns.append(expanded)
        sigma = torch.stack(sigma_columns, dim=1).unsqueeze(-1)
        return torch.cat((torch.zeros_like(sigma), sigma), dim=-1)

    def objective_components_for_solution(
        self, solution: GarchTreeSolution
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        paths = solution.leaf_paths.unsqueeze(-1)
        controls = self.controls_for_solution(solution)
        discrepancy = self.discrepancy(
            generated_paths=paths,
            target_paths=self.target.leaf_paths.unsqueeze(-1),
            grid=self.grid,
        )
        running = self.control_map.running_cost(
            RunningCostInputs(
                paths=paths,
                controls=controls,
                grid=self.grid,
                differentiate_controls=True,
            )
        )
        scaled_discrepancy = self.lambda_scale * discrepancy.value
        total = running.value + scaled_discrepancy
        return total, {
            "total": total,
            "volatility_cost": running.value,
            "discrepancy": discrepancy.value,
            "scaled_discrepancy": scaled_discrepancy,
        }

    def objective_components(
        self, variables: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.objective_components_for_solution(self.solution(variables))

    def objective(self, variables: torch.Tensor) -> torch.Tensor:
        return self.objective_components(variables)[0]

    def conditional_adjoint_targets_for_solution(
        self, solution: GarchTreeSolution
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        independent_states = tuple(
            level.detach().clone().requires_grad_(True)
            for level in solution.state_levels[1:]
        )
        state_levels = (
            torch.zeros(
                1,
                device=solution.state_levels[0].device,
                dtype=solution.state_levels[0].dtype,
            ),
            *independent_states,
        )
        leaf_paths = _leaf_paths(
            state_levels, branching=int(self.config.branching)
        )
        controls = self.controls_for_solution(solution).detach()
        discrepancy = self.discrepancy(
            generated_paths=leaf_paths.unsqueeze(-1),
            target_paths=self.target.leaf_paths.unsqueeze(-1),
            grid=self.grid,
        )
        running = self.control_map.running_cost(
            RunningCostInputs(
                paths=leaf_paths.unsqueeze(-1),
                controls=controls,
                grid=self.grid,
                differentiate_controls=False,
            )
        )
        source_objective = self.lambda_scale * discrepancy.value + running.value
        sources_tail = torch.autograd.grad(source_objective, independent_states)
        sources = (torch.zeros_like(state_levels[0]), *sources_tail)
        costate: list[torch.Tensor] = [
            torch.empty(0, device=leaf_paths.device, dtype=leaf_paths.dtype)
        ] * (int(self.config.horizon) + 1)
        costate[-1] = sources[-1]
        branching = int(self.config.branching)
        for level in range(int(self.config.horizon) - 1, -1, -1):
            descendants = costate[level + 1].reshape(
                -1, branching
            ).sum(dim=1)
            costate[level] = sources[level] + descendants
        p_levels = []
        r_levels = []
        innovations = self.innovations.to(leaf_paths)
        for level in range(int(self.config.horizon)):
            child = costate[level + 1].reshape(-1, branching)
            probability = self.level_node_probability(level)
            p_levels.append((child.sum(dim=1) / probability).detach())
            r_levels.append(
                (
                    (child * innovations[None, :]).sum(dim=1)
                    / probability
                ).detach()
            )
        return tuple(p_levels), tuple(r_levels)


@dataclass(frozen=True)
class ExactGarchTreeMPResponse:
    """Production Hamiltonian response driven by exact tree conditional moments.

    This object deliberately contains no neural-network state.  The nodewise
    conditional adjoints are computed by complete descendant enumeration and
    are injected directly into the production volatility-only control map.
    """

    variables: torch.Tensor
    solution: GarchTreeSolution
    expected_adjoint_next: torch.Tensor
    expected_adjoint_noise_next: torch.Tensor
    controls: torch.Tensor
    metrics: dict[str, float]


def _expand_node_levels_to_leaves(
    levels: tuple[torch.Tensor, ...],
    *,
    config: GarchTreeConfig,
) -> torch.Tensor:
    """Expand one value per causal tree node to one value per leaf and date."""

    if len(levels) != int(config.horizon):
        raise ValueError("one node tensor is required per decision level")
    columns = []
    branching = int(config.branching)
    horizon = int(config.horizon)
    for level, values in enumerate(levels):
        expected_nodes = branching**level
        if values.ndim != 1 or int(values.numel()) != expected_nodes:
            raise ValueError("node tensor has the wrong size")
        descendants = branching ** (horizon - level)
        columns.append(values.repeat_interleave(descendants))
    return torch.stack(columns, dim=1).unsqueeze(-1)


def exact_garch_tree_mp_response(
    problem: ProductionGarchTreeProblem,
    variables: torch.Tensor,
) -> ExactGarchTreeMPResponse:
    """Apply one strict exact-CE MP response through the production control map.

    The input law is frozen while its complete finite tree is enumerated.  No
    regression, Monte Carlo approximation, or GRU is used: exact nodewise
    conditional P/R values are sent directly to ``controls_from_moments``.
    """

    if variables.ndim != 1 or int(variables.numel()) != problem.variable_count:
        raise ValueError("variables have the wrong dimension")
    current = variables.detach().to(device=problem.device, dtype=problem.dtype)
    solution = problem.solution(current)
    p_levels, r_levels = problem.conditional_adjoint_targets_for_solution(solution)
    p = _expand_node_levels_to_leaves(p_levels, config=problem.config)
    r = _expand_node_levels_to_leaves(r_levels, config=problem.config)
    paths = solution.leaf_paths.unsqueeze(-1)
    controls = problem.control_map.controls_from_moments(
        paths=paths,
        expected_adjoint_next=p,
        expected_adjoint_noise_next=r,
    )
    drift = controls[..., 0]
    if not bool(torch.all(drift == 0.0).item()):
        raise RuntimeError("the volatility-only production map produced nonzero drift")

    sigma_pathwise = controls[..., 1]
    branching = int(problem.config.branching)
    horizon = int(problem.config.horizon)
    sigma_levels = []
    consensus_max = 0.0
    for level in range(horizon):
        nodes = branching**level
        descendants = branching ** (horizon - level)
        grouped = sigma_pathwise[:, level].reshape(nodes, descendants)
        consensus_max = max(
            consensus_max,
            float((grouped - grouped[:, :1]).abs().max().item()),
        )
        sigma_levels.append(grouped[:, 0])
    candidate = problem.pack(tuple(sigma_levels)).detach()

    # This independent closed-form path is retained as a permanent audit that
    # direct injection really uses the same Hamiltonian map as the tree oracle.
    analytic = problem.hamiltonian_best_response(current)
    map_error = float((candidate - analytic).abs().max().item())
    if consensus_max > 1e-12 or map_error > 1e-10:
        raise RuntimeError("exact conditional moments and control map are inconsistent")

    sigma = torch.cat(tuple(sigma_levels))
    return ExactGarchTreeMPResponse(
        variables=candidate,
        solution=solution,
        expected_adjoint_next=p.detach(),
        expected_adjoint_noise_next=r.detach(),
        controls=controls.detach(),
        metrics={
            "conditional_expectation_enumeration_exact": 1.0,
            "neural_regression_used": 0.0,
            "zero_drift_max_abs": float(drift.abs().max().item()),
            "nodewise_control_consensus_max_abs": consensus_max,
            "production_vs_closed_form_map_max_abs": map_error,
            "p_rms": float(p.double().pow(2).mean().sqrt().item()),
            "r_rms": float(r.double().pow(2).mean().sqrt().item()),
            "response_log_sigma_rms": float(
                (candidate - current).double().pow(2).mean().sqrt().item()
            ),
            "response_sigma_min": float(sigma.min().item()),
            "response_sigma_max": float(sigma.max().item()),
            "response_lower_bound_activity": float(
                (
                    sigma
                    <= float(problem.config.sigma_min) * (1.0 + 1e-6)
                )
                .double()
                .mean()
                .item()
            ),
            "response_upper_bound_activity": float(
                (
                    sigma
                    >= float(problem.config.sigma_max) * (1.0 - 1e-6)
                )
                .double()
                .mean()
                .item()
            ),
        },
    )


def enumerated_garch_noise(
    config: GarchTreeConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return every innovation sequence in target-tree leaf order."""

    leaves = int(config.leaf_count)
    branching = int(config.branching)
    innovations = torch.tensor(config.innovations, device=device, dtype=dtype)
    indices = torch.arange(leaves, device=device, dtype=torch.long)
    columns = []
    for step in range(int(config.horizon)):
        divisor = branching ** (int(config.horizon) - step - 1)
        digit = torch.div(indices, divisor, rounding_mode="floor") % branching
        columns.append(innovations.index_select(0, digit))
    return torch.stack(columns, dim=1).unsqueeze(-1)


def production_model_on_exact_tree(
    model: DiscreteMPModel,
    problem: ProductionGarchTreeProblem,
) -> tuple[GarchTreeSolution, dict[str, float]]:
    """Evaluate a fitted production checkpoint on every exact tree branch."""

    noise = enumerated_garch_noise(
        problem.config, device=model.device, dtype=model.dtype
    )
    x0 = torch.zeros(
        int(problem.config.leaf_count), 1, device=model.device, dtype=model.dtype
    )
    with torch.no_grad():
        rollout = model.rollout(x0=x0, noise=noise)
    sigma_pathwise = rollout.controls[..., 1]
    drift = rollout.controls[..., 0]
    sigma_levels = []
    consensus_max = 0.0
    branching = int(problem.config.branching)
    horizon = int(problem.config.horizon)
    for level in range(horizon):
        nodes = branching**level
        descendants = branching ** (horizon - level)
        grouped = sigma_pathwise[:, level].reshape(nodes, descendants)
        consensus_max = max(
            consensus_max,
            float((grouped - grouped[:, :1]).abs().max().item()),
        )
        sigma_levels.append(grouped[:, 0].to(dtype=problem.dtype))
    variables = problem.pack(tuple(sigma_levels))
    solution = problem.solution(variables)
    rollout_paths = rollout.paths[..., 0].to(dtype=problem.dtype)
    path_error = float((solution.leaf_paths - rollout_paths).abs().max().item())
    return solution, {
        "zero_drift_max_abs": float(drift.abs().max().item()),
        "nodewise_sigma_consensus_max_abs": consensus_max,
        "tree_reconstruction_max_abs": path_error,
    }


@dataclass(frozen=True)
class ExactGarchTreeConditionalTargetProvider:
    """Exact nodewise P/R labels for a complete production GARCH-tree rollout."""

    problem: ProductionGarchTreeProblem
    path_tolerance: float = 5e-6

    def __call__(
        self,
        model: DiscreteMPModel,
        rollout,
        target_paths: torch.Tensor,
    ) -> ConditionalAdjointTargets:
        config = self.problem.config
        leaves = int(config.leaf_count)
        horizon = int(config.horizon)
        branching = int(config.branching)
        if int(rollout.paths.shape[0]) != leaves:
            raise ValueError(
                "the exact GARCH-tree target provider requires every tree leaf"
            )
        expected_target_shape = (leaves, horizon + 1, 1)
        if tuple(target_paths.shape) != expected_target_shape:
            raise ValueError(
                "the exact GARCH-tree target provider requires the complete target tree"
            )
        exact_target = self.problem.target.leaf_paths.unsqueeze(-1).to(
            device=target_paths.device, dtype=target_paths.dtype
        )
        target_error = float((target_paths - exact_target).abs().max().item())
        if target_error > float(self.path_tolerance):
            raise ValueError("target_paths do not match the locked GARCH tree")
        drift = rollout.controls[..., 0]
        if not bool(torch.all(drift == 0.0).item()):
            raise ValueError("oracle-CE GARCH training requires exactly zero drift")

        sigma_pathwise = rollout.controls[..., 1]
        sigma_levels = []
        consensus_error = 0.0
        for level in range(horizon):
            nodes = branching**level
            descendants = branching ** (horizon - level)
            grouped = sigma_pathwise[:, level].reshape(nodes, descendants)
            consensus_error = max(
                consensus_error,
                float((grouped - grouped[:, :1]).abs().max().item()),
            )
            sigma_levels.append(
                grouped[:, 0].to(
                    device=self.problem.device, dtype=self.problem.dtype
                )
            )
        solution = self.problem.solution(
            self.problem.pack(tuple(sigma_levels))
        )
        rollout_error = float(
            (
                solution.leaf_paths
                - rollout.paths[..., 0].to(
                    device=self.problem.device, dtype=self.problem.dtype
                )
            )
            .abs()
            .max()
            .item()
        )
        if max(consensus_error, rollout_error) > float(self.path_tolerance):
            raise RuntimeError("production rollout is inconsistent with its tree law")

        p_levels, r_levels = self.problem.conditional_adjoint_targets_for_solution(
            solution
        )
        p_columns = []
        r_columns = []
        for level, (p_target, r_target) in enumerate(zip(p_levels, r_levels)):
            descendants = branching ** (horizon - level)
            p_columns.append(p_target.repeat_interleave(descendants))
            r_columns.append(r_target.repeat_interleave(descendants))
        p = torch.stack(p_columns, dim=1).unsqueeze(-1).to(
            device=model.device, dtype=model.dtype
        )
        r = torch.stack(r_columns, dim=1).unsqueeze(-1).to(
            device=model.device, dtype=model.dtype
        )
        return ConditionalAdjointTargets(
            expected_adjoint_next=p,
            expected_adjoint_noise_next=r,
            metrics={
                "oracle_ce_targets": 1.0,
                "oracle_ce_p_rms": float(torch.sqrt(torch.mean(p.pow(2))).item()),
                "oracle_ce_r_rms": float(torch.sqrt(torch.mean(r.pow(2))).item()),
                "oracle_ce_node_consensus_max_abs": consensus_error,
                "oracle_ce_tree_reconstruction_max_abs": rollout_error,
                "oracle_ce_target_tree_max_abs": target_error,
            },
        )


__all__ = [
    "ExactGarchTreeConditionalTargetProvider",
    "ExactGarchTreeMPResponse",
    "ProductionGarchTreeProblem",
    "build_production_garch_components",
    "constant_zero_drift_reference",
    "enumerated_garch_noise",
    "exact_garch_tree_mp_response",
    "production_model_on_exact_tree",
]
