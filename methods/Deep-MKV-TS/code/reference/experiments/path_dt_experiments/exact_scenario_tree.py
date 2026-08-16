from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import torch

from deep_mkv_gen_path_dt.discrepancies.mmd import rbf_mmd2


ControlMode = Literal["full", "volatility_only"]


@dataclass(frozen=True)
class ExactTreeConfig:
    horizon: int = 8
    dt: float = 0.125
    sigma_low: float = 0.18
    sigma_high: float = 0.45
    sigma_min: float = 0.03
    sigma_max: float = 0.80
    drawdown_threshold: float = 0.08
    history_cutoff: int = 2
    response_delay: int = 2
    eta: float = 4.0
    lambda_scale: float = 50.0
    bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0)

    def __post_init__(self) -> None:
        if int(self.horizon) < 2:
            raise ValueError("horizon must be at least two")
        if not math.isfinite(float(self.dt)) or float(self.dt) <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not 0 <= int(self.history_cutoff) < int(self.horizon):
            raise ValueError("history_cutoff must lie inside the horizon")
        if int(self.response_delay) < 1:
            raise ValueError("response_delay must be positive")
        if int(self.history_cutoff) + int(self.response_delay) >= int(self.horizon):
            raise ValueError("the delayed response must begin before the horizon")
        if not 0.0 < float(self.sigma_min) < float(self.sigma_low):
            raise ValueError("sigma_min must lie below sigma_low")
        if not float(self.sigma_low) < float(self.sigma_high) < float(self.sigma_max):
            raise ValueError("volatility levels must lie strictly inside the bounds")
        if float(self.eta) <= 0.0 or float(self.lambda_scale) <= 0.0:
            raise ValueError("eta and lambda_scale must be positive")

    @property
    def response_start(self) -> int:
        return int(self.history_cutoff) + int(self.response_delay)

    @property
    def decision_nodes(self) -> int:
        return 2 ** int(self.horizon) - 1

    @property
    def leaf_count(self) -> int:
        return 2 ** int(self.horizon)


@dataclass(frozen=True)
class ExactTargetTree:
    state_levels: tuple[torch.Tensor, ...]
    sigma_levels: tuple[torch.Tensor, ...]
    trigger_levels: tuple[torch.Tensor, ...]
    leaf_paths: torch.Tensor
    reference_sigma: torch.Tensor


@dataclass(frozen=True)
class ExactTreeSolution:
    mode: ControlMode
    alpha_levels: tuple[torch.Tensor, ...]
    sigma_levels: tuple[torch.Tensor, ...]
    state_levels: tuple[torch.Tensor, ...]
    leaf_paths: torch.Tensor


def _children(parent: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor, dt: float) -> torch.Tensor:
    center = parent + alpha * float(dt)
    displacement = sigma * math.sqrt(float(dt))
    return torch.stack((center - displacement, center + displacement), dim=1).reshape(-1)


def _leaf_paths(state_levels: tuple[torch.Tensor, ...]) -> torch.Tensor:
    horizon = len(state_levels) - 1
    leaves = 2**horizon
    columns = []
    for level, states in enumerate(state_levels):
        repeat = 2 ** (horizon - level)
        columns.append(states.repeat_interleave(repeat))
    return torch.stack(columns, dim=1).reshape(leaves, horizon + 1)


def build_delayed_memory_target(
    config: ExactTreeConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ExactTargetTree:
    device = torch.device(device)
    states = [torch.zeros(1, device=device, dtype=dtype)]
    running_max = [states[0].clone()]
    triggers = [torch.zeros(1, device=device, dtype=torch.bool)]
    sigmas = []
    for step in range(int(config.horizon)):
        active = triggers[step] & (step >= int(config.response_start))
        sigma = torch.where(
            active,
            torch.full_like(states[step], float(config.sigma_high)),
            torch.full_like(states[step], float(config.sigma_low)),
        )
        sigmas.append(sigma)
        alpha = torch.zeros_like(sigma)
        child = _children(states[step], alpha, sigma, float(config.dt))
        child_max = torch.maximum(
            running_max[step].repeat_interleave(2), child
        )
        inherited = triggers[step].repeat_interleave(2)
        if step + 1 <= int(config.history_cutoff):
            new_hit = child_max - child >= float(config.drawdown_threshold)
            child_trigger = inherited | new_hit
        else:
            child_trigger = inherited
        states.append(child)
        running_max.append(child_max)
        triggers.append(child_trigger)
    sigma_matrix = torch.stack(
        [
            torch.sqrt(torch.mean(level.pow(2)))
            for level in sigmas
        ]
    )
    state_tuple = tuple(states)
    return ExactTargetTree(
        state_levels=state_tuple,
        sigma_levels=tuple(sigmas),
        trigger_levels=tuple(triggers),
        leaf_paths=_leaf_paths(state_tuple),
        reference_sigma=sigma_matrix,
    )


class ExactBinaryTreeProblem:
    """Reduced simultaneous control problem on a complete binary tree."""

    def __init__(
        self,
        config: ExactTreeConfig,
        *,
        mode: ControlMode,
        fixed_drift_levels: tuple[torch.Tensor, ...] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if mode not in {"full", "volatility_only"}:
            raise ValueError("mode must be full or volatility_only")
        self.config = config
        self.mode = mode
        self.device = torch.device(device)
        self.dtype = dtype
        if fixed_drift_levels is not None and mode != "volatility_only":
            raise ValueError("fixed_drift_levels is only valid in volatility_only mode")
        if fixed_drift_levels is None:
            self.fixed_drift_levels = None
        else:
            if len(fixed_drift_levels) != int(config.horizon):
                raise ValueError("fixed_drift_levels must contain one tensor per step")
            converted = tuple(
                value.detach().to(device=self.device, dtype=self.dtype).reshape(-1)
                for value in fixed_drift_levels
            )
            expected = tuple(2**level for level in range(int(config.horizon)))
            observed = tuple(int(value.numel()) for value in converted)
            if observed != expected:
                raise ValueError(
                    "fixed_drift_levels must follow the complete-tree node counts"
                )
            self.fixed_drift_levels = converted
        self.target = build_delayed_memory_target(
            config, device=self.device, dtype=self.dtype
        )
        target = self.target.leaf_paths[:, 1:]
        center = target.mean(dim=0, keepdim=True)
        raw_scale = target.std(dim=0, keepdim=True, unbiased=False)
        positive = raw_scale[raw_scale > 0.0]
        floor = 0.1 * float(positive.median().item()) if int(positive.numel()) else 1.0
        self.feature_center = center.detach()
        self.feature_scale = raw_scale.clamp_min(max(floor, 1e-8)).detach()
        self.target_features = self._features(self.target.leaf_paths).detach()
        self._level_offsets = tuple(
            sum(2**previous for previous in range(level))
            for level in range(int(config.horizon))
        )

    @property
    def variable_count(self) -> int:
        multiplier = 2 if self.mode == "full" else 1
        return multiplier * int(self.config.decision_nodes)

    def _features(self, leaf_paths: torch.Tensor) -> torch.Tensor:
        return (leaf_paths[:, 1:] - self.feature_center) / self.feature_scale

    def unpack(self, variables: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if variables.ndim != 1 or int(variables.numel()) != self.variable_count:
            raise ValueError("variables have the wrong dimension")
        node_count = int(self.config.decision_nodes)
        if self.mode == "full":
            alpha_flat = variables[:node_count]
            log_sigma_flat = variables[node_count:]
        else:
            alpha_flat = (
                torch.zeros(
                    node_count, device=variables.device, dtype=variables.dtype
                )
                if self.fixed_drift_levels is None
                else torch.cat(self.fixed_drift_levels).to(variables)
            )
            log_sigma_flat = variables
        alpha_levels = []
        sigma_levels = []
        for level, offset in enumerate(self._level_offsets):
            count = 2**level
            alpha_levels.append(alpha_flat[offset : offset + count])
            sigma_levels.append(torch.exp(log_sigma_flat[offset : offset + count]))
        return tuple(alpha_levels), tuple(sigma_levels)

    def pack(
        self,
        alpha_levels: tuple[torch.Tensor, ...],
        sigma_levels: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        log_sigma = torch.cat([torch.log(value) for value in sigma_levels])
        if self.mode == "full":
            return torch.cat((torch.cat(alpha_levels), log_sigma))
        return log_sigma

    def reference_start(self) -> torch.Tensor:
        alpha = tuple(
            torch.zeros(2**level, device=self.device, dtype=self.dtype)
            for level in range(int(self.config.horizon))
        )
        sigma = tuple(
            torch.full(
                (2**level,),
                float(self.target.reference_sigma[level].item()),
                device=self.device,
                dtype=self.dtype,
            )
            for level in range(int(self.config.horizon))
        )
        return self.pack(alpha, sigma)

    def oracle_start(self) -> torch.Tensor:
        alpha = tuple(torch.zeros_like(value) for value in self.target.sigma_levels)
        return self.pack(alpha, self.target.sigma_levels)

    def target_solution(self) -> ExactTreeSolution:
        """Return the data-generating tree, independently of the drift constraint."""

        alpha = tuple(torch.zeros_like(value) for value in self.target.sigma_levels)
        return ExactTreeSolution(
            mode=self.mode,
            alpha_levels=alpha,
            sigma_levels=self.target.sigma_levels,
            state_levels=self.target.state_levels,
            leaf_paths=self.target.leaf_paths,
        )

    def bounds(self) -> list[tuple[float | None, float | None]]:
        sigma_bounds = [
            (math.log(float(self.config.sigma_min)), math.log(float(self.config.sigma_max)))
            for _ in range(int(self.config.decision_nodes))
        ]
        if self.mode == "full":
            return [(None, None)] * int(self.config.decision_nodes) + sigma_bounds
        return sigma_bounds

    def solution(self, variables: torch.Tensor) -> ExactTreeSolution:
        alpha, sigma = self.unpack(variables)
        states = [torch.zeros(1, device=variables.device, dtype=variables.dtype)]
        for level in range(int(self.config.horizon)):
            states.append(
                _children(
                    states[level], alpha[level], sigma[level], float(self.config.dt)
                )
            )
        state_tuple = tuple(states)
        return ExactTreeSolution(
            mode=self.mode,
            alpha_levels=alpha,
            sigma_levels=sigma,
            state_levels=state_tuple,
            leaf_paths=_leaf_paths(state_tuple),
        )

    def objective_components_for_solution(
        self,
        solution: ExactTreeSolution,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate the exact objective for any admissible tree solution."""

        if solution.mode != self.mode:
            raise ValueError("solution control mode does not match the problem")
        mmd = rbf_mmd2(
            self._features(solution.leaf_paths),
            self.target_features,
            bandwidths=self.config.bandwidths,
            unbiased=False,
        )
        drift_cost = solution.leaf_paths.new_tensor(0.0)
        volatility_cost = solution.leaf_paths.new_tensor(0.0)
        for level, (alpha, sigma) in enumerate(
            zip(solution.alpha_levels, solution.sigma_levels)
        ):
            probability = 2.0 ** (-level)
            sigma_ref = self.target.reference_sigma[level].to(sigma)
            ratio = sigma / sigma_ref
            # In volatility-only problems the physical drift is fixed rather
            # than controlled.  The drift running cost is therefore zero: it
            # penalizes an admissible drift correction, not the fixed baseline.
            if self.mode == "full":
                drift_cost = drift_cost + probability * float(self.config.dt) * 0.5 * alpha.pow(2).sum()
            volatility_cost = volatility_cost + probability * float(self.config.dt) * float(
                self.config.eta
            ) * (ratio - 1.0 - torch.log(ratio)).sum()
        discrepancy = float(self.config.lambda_scale) * mmd
        total = drift_cost + volatility_cost + discrepancy
        return total, {
            "total": total,
            "drift_cost": drift_cost,
            "volatility_cost": volatility_cost,
            "mmd2": mmd,
            "scaled_discrepancy": discrepancy,
        }

    def objective_components(self, variables: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.objective_components_for_solution(self.solution(variables))

    def objective(self, variables: torch.Tensor) -> torch.Tensor:
        return self.objective_components(variables)[0]

    def conditional_adjoint_targets_for_solution(
        self,
        solution: ExactTreeSolution,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Reconstruct exact nodewise P/R targets with controls held fixed."""

        if solution.mode != self.mode:
            raise ValueError("solution control mode does not match the problem")
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
        leaf_paths = _leaf_paths(state_levels)
        discrepancy = float(self.config.lambda_scale) * rbf_mmd2(
            self._features(leaf_paths),
            self.target_features,
            bandwidths=self.config.bandwidths,
            unbiased=False,
        )
        sources_tail = torch.autograd.grad(discrepancy, independent_states)
        sources = (torch.zeros_like(state_levels[0]), *sources_tail)
        costate: list[torch.Tensor] = [
            torch.empty(0, device=leaf_paths.device, dtype=leaf_paths.dtype)
        ] * (int(self.config.horizon) + 1)
        costate[-1] = sources[-1]
        for level in range(int(self.config.horizon) - 1, -1, -1):
            descendants = costate[level + 1].reshape(-1, 2).sum(dim=1)
            costate[level] = sources[level] + descendants

        p_levels = []
        r_levels = []
        for level in range(int(self.config.horizon)):
            child = costate[level + 1].reshape(-1, 2)
            node_probability = 2.0 ** (-level)
            p_levels.append((child.sum(dim=1) / node_probability).detach())
            r_levels.append(
                ((child[:, 1] - child[:, 0]) / node_probability).detach()
            )
        return tuple(p_levels), tuple(r_levels)

    def hamiltonian_best_response(self, variables: torch.Tensor) -> torch.Tensor:
        """Return the bounded nodewise Hamiltonian best response.

        The adjoints are reconstructed exactly under ``variables``.  This is
        the frozen-law MP proposal used by the one-stage correction audit; it
        does not refresh the law or solve a fixed point.
        """

        audit = self.exact_adjoint_audit(variables)
        solution = self.solution(variables.detach())
        alpha_levels = []
        sigma_levels = []
        for level, (p, r) in enumerate(
            zip(audit["P_levels"], audit["R_levels"])
        ):
            if self.mode == "full":
                alpha_levels.append(-p.to(solution.alpha_levels[level]))
            else:
                alpha_levels.append(torch.zeros_like(solution.alpha_levels[level]))
            sigma_ref = self.target.reference_sigma[level].to(
                solution.sigma_levels[level]
            )
            denominator = float(self.config.eta) / sigma_ref + r.to(
                sigma_ref
            ) / math.sqrt(float(self.config.dt))
            unconstrained = float(self.config.eta) / denominator.clamp_min(1e-12)
            sigma_levels.append(
                torch.where(
                    denominator > 1e-12,
                    unconstrained,
                    torch.full_like(unconstrained, float(self.config.sigma_max)),
                ).clamp(
                    min=float(self.config.sigma_min),
                    max=float(self.config.sigma_max),
                )
            )
        return self.pack(tuple(alpha_levels), tuple(sigma_levels)).detach()

    def exact_adjoint_audit(self, variables: torch.Tensor) -> dict[str, object]:
        detached = variables.detach().clone().requires_grad_(True)
        solution = self.solution(detached)
        p_levels, r_levels = self.conditional_adjoint_targets_for_solution(
            solution
        )
        drift_residuals = []
        sigma_residuals = []
        gradient_formula = []
        for level, (p, r) in enumerate(zip(p_levels, r_levels)):
            node_probability = 2.0 ** (-level)
            alpha = solution.alpha_levels[level]
            sigma = solution.sigma_levels[level]
            sigma_ref = self.target.reference_sigma[level].to(sigma)
            drift_stationarity = alpha + p
            denominator = float(self.config.eta) / sigma_ref + r / math.sqrt(
                float(self.config.dt)
            )
            unconstrained = float(self.config.eta) / denominator.clamp_min(1e-12)
            best_sigma = torch.where(
                denominator > 1e-12,
                unconstrained,
                torch.full_like(unconstrained, float(self.config.sigma_max)),
            ).clamp(
                min=float(self.config.sigma_min), max=float(self.config.sigma_max)
            )
            sigma_stationarity = torch.log(sigma) - torch.log(best_sigma)
            drift_residuals.append(drift_stationarity)
            sigma_residuals.append(sigma_stationarity)
            if self.mode == "full":
                gradient_formula.append(
                    node_probability * float(self.config.dt) * drift_stationarity
                )
            physical_sigma_gradient = node_probability * float(self.config.dt) * (
                float(self.config.eta) * (1.0 / sigma_ref - 1.0 / sigma)
                + r / math.sqrt(float(self.config.dt))
            )
            gradient_formula.append(physical_sigma_gradient * sigma)

        direct_gradient = torch.autograd.grad(self.objective(detached), detached)[0]
        if self.mode == "full":
            expected_gradient = torch.cat(
                (
                    torch.cat(gradient_formula[: int(self.config.horizon)]),
                    torch.cat(gradient_formula[int(self.config.horizon) :]),
                )
            )
        else:
            expected_gradient = torch.cat(gradient_formula)
        # The formula list is interleaved only in full mode; rebuild it by
        # direct level partitions for a robust independent comparison.
        if self.mode == "full":
            drift_formula = []
            sigma_formula = []
            for level, (p, r) in enumerate(zip(p_levels, r_levels)):
                probability = 2.0 ** (-level)
                alpha = solution.alpha_levels[level]
                sigma = solution.sigma_levels[level]
                sigma_ref = self.target.reference_sigma[level].to(sigma)
                drift_formula.append(
                    probability * float(self.config.dt) * (alpha + p)
                )
                physical = probability * float(self.config.dt) * (
                    float(self.config.eta) * (1.0 / sigma_ref - 1.0 / sigma)
                    + r / math.sqrt(float(self.config.dt))
                )
                sigma_formula.append(physical * sigma)
            expected_gradient = torch.cat((torch.cat(drift_formula), torch.cat(sigma_formula)))
        gradient_relative_error = float(
            torch.linalg.vector_norm((direct_gradient - expected_gradient).double())
            / torch.linalg.vector_norm(direct_gradient.double()).clamp_min(
                torch.finfo(torch.float64).eps
            )
        )
        sigma_flat = torch.cat(solution.sigma_levels)
        lower = float(self.config.sigma_min)
        upper = float(self.config.sigma_max)
        active_tolerance = 1e-6
        return {
            "P_levels": tuple(value.detach() for value in p_levels),
            "R_levels": tuple(value.detach() for value in r_levels),
            "drift_stationarity_rms": float(
                torch.mean(torch.cat(drift_residuals).double().pow(2)).sqrt().item()
            ),
            "volatility_log_best_response_rms": float(
                torch.mean(torch.cat(sigma_residuals).double().pow(2)).sqrt().item()
            ),
            "direct_gradient_rms": float(
                torch.mean(direct_gradient.double().pow(2)).sqrt().item()
            ),
            "independent_adjoint_gradient_relative_error": gradient_relative_error,
            "sigma_lower_activity": float(
                torch.mean((sigma_flat <= lower * (1.0 + active_tolerance)).double()).item()
            ),
            "sigma_upper_activity": float(
                torch.mean((sigma_flat >= upper * (1.0 - active_tolerance)).double()).item()
            ),
            "minimum_sigma_bound_margin": float(
                torch.minimum(sigma_flat - lower, upper - sigma_flat).min().item()
            ),
        }

    def delayed_memory_metrics(self, solution: ExactTreeSolution) -> dict[str, float]:
        paths = solution.leaf_paths.detach()
        returns = paths[:, 1:] - paths[:, :-1]
        prefix = paths[:, : int(self.config.history_cutoff) + 1]
        running_max = torch.cummax(prefix, dim=1).values
        hit = (running_max - prefix).max(dim=1).values >= float(
            self.config.drawdown_threshold
        )
        future = torch.sqrt(
            returns[:, int(self.config.response_start) :].pow(2).mean(dim=1)
            / float(self.config.dt)
        )
        if bool(hit.all().item()) or bool((~hit).all().item()):
            gap = float("nan")
            correlation = float("nan")
        else:
            gap = float((future[hit].mean() - future[~hit].mean()).item())
            correlation = float(
                torch.corrcoef(torch.stack((hit.to(future), future)))[0, 1].item()
            )
        return {
            "early_hit_rate": float(hit.double().mean().item()),
            "future_rv_mean": float(future.mean().item()),
            "future_rv_hit_gap": gap,
            "early_hit_future_rv_correlation": correlation,
        }

    def full_kkt_conditioning(
        self, variables: torch.Tensor, *, active_tolerance: float = 1e-6
    ) -> dict[str, float]:
        solution = self.solution(variables.detach())
        horizon = int(self.config.horizon)
        state_counts = [2**level for level in range(1, horizon + 1)]
        state_total = sum(state_counts)
        control_nodes = int(self.config.decision_nodes)
        alpha_count = control_nodes if self.mode == "full" else 0
        variable_count = state_total + alpha_count + control_nodes

        def state_discrepancy(flat: torch.Tensor) -> torch.Tensor:
            levels = [torch.zeros(1, device=flat.device, dtype=flat.dtype)]
            offset = 0
            for count in state_counts:
                levels.append(flat[offset : offset + count])
                offset += count
            return float(self.config.lambda_scale) * rbf_mmd2(
                self._features(_leaf_paths(tuple(levels))),
                self.target_features,
                bandwidths=self.config.bandwidths,
                unbiased=False,
            )

        flat_states = torch.cat(solution.state_levels[1:]).detach().requires_grad_(True)
        state_hessian = torch.autograd.functional.hessian(
            state_discrepancy, flat_states, vectorize=True
        ).detach().cpu().double().numpy()
        hessian = np.zeros((variable_count, variable_count), dtype=np.float64)
        hessian[:state_total, :state_total] = state_hessian
        control_offset = state_total
        if self.mode == "full":
            diagonal = []
            for level in range(horizon):
                diagonal.extend(
                    [2.0 ** (-level) * float(self.config.dt)] * (2**level)
                )
            hessian[
                control_offset : control_offset + alpha_count,
                control_offset : control_offset + alpha_count,
            ] = np.diag(diagonal)
            sigma_offset = control_offset + alpha_count
        else:
            sigma_offset = control_offset
        sigma_diagonal = []
        for level, sigma in enumerate(solution.sigma_levels):
            sigma_diagonal.extend(
                (
                    2.0 ** (-level)
                    * float(self.config.dt)
                    * float(self.config.eta)
                    / sigma.detach().cpu().double().numpy() ** 2
                ).tolist()
            )
        hessian[sigma_offset:, sigma_offset:] = np.diag(sigma_diagonal)

        state_offsets = {}
        offset = 0
        for level in range(1, horizon + 1):
            state_offsets[level] = offset
            offset += 2**level
        constraint_count = state_total
        jacobian = np.zeros((constraint_count, variable_count), dtype=np.float64)
        row = 0
        for level in range(horizon):
            parent_count = 2**level
            control_level_offset = self._level_offsets[level]
            for parent in range(parent_count):
                for branch, epsilon in enumerate((-1.0, 1.0)):
                    child = 2 * parent + branch
                    jacobian[row, state_offsets[level + 1] + child] = 1.0
                    if level > 0:
                        jacobian[row, state_offsets[level] + parent] = -1.0
                    if self.mode == "full":
                        jacobian[
                            row, control_offset + control_level_offset + parent
                        ] = -float(self.config.dt)
                    jacobian[
                        row, sigma_offset + control_level_offset + parent
                    ] = -epsilon * math.sqrt(float(self.config.dt))
                    row += 1
        sigma = torch.cat(solution.sigma_levels).detach().cpu().numpy()
        active = np.flatnonzero(
            (sigma <= float(self.config.sigma_min) * (1.0 + active_tolerance))
            | (sigma >= float(self.config.sigma_max) * (1.0 - active_tolerance))
        )
        if active.size:
            active_jacobian = np.zeros((active.size, variable_count), dtype=np.float64)
            active_jacobian[np.arange(active.size), sigma_offset + active] = 1.0
            jacobian = np.vstack((jacobian, active_jacobian))
        zeros = np.zeros((jacobian.shape[0], jacobian.shape[0]), dtype=np.float64)
        kkt = np.block([[hessian, jacobian.T], [jacobian, zeros]])
        singular = np.linalg.svd(kkt, compute_uv=False)
        minimum = max(float(singular[-1]), np.finfo(np.float64).tiny)
        return {
            "full_kkt_dimension": float(kkt.shape[0]),
            "active_bound_constraints": float(active.size),
            "largest_singular_value": float(singular[0]),
            "smallest_singular_value": float(singular[-1]),
            "condition_number": float(singular[0] / minimum),
            "state_hessian_minimum_eigenvalue": float(
                np.linalg.eigvalsh(state_hessian)[0]
            ),
            "state_hessian_maximum_eigenvalue": float(
                np.linalg.eigvalsh(state_hessian)[-1]
            ),
        }

    def reduced_hessian_conditioning(self, variables: torch.Tensor) -> dict[str, float]:
        point = variables.detach().clone().requires_grad_(True)
        hessian = torch.autograd.functional.hessian(
            self.objective, point, vectorize=True
        ).detach().cpu().double()
        eigenvalues = torch.linalg.eigvalsh(hessian)
        absolute = eigenvalues.abs()
        nonzero = absolute[absolute > 1e-12]
        smallest_absolute = (
            float(nonzero.min().item()) if int(nonzero.numel()) else 0.0
        )
        condition = (
            float(absolute.max().item()) / smallest_absolute
            if smallest_absolute > 0.0
            else float("inf")
        )
        return {
            "dimension": float(hessian.shape[0]),
            "minimum_eigenvalue": float(eigenvalues[0].item()),
            "maximum_eigenvalue": float(eigenvalues[-1].item()),
            "negative_eigenvalue_count_below_minus_1e_8": float(
                torch.sum(eigenvalues < -1e-8).item()
            ),
            "near_zero_eigenvalue_count_absolute_1e_8": float(
                torch.sum(absolute <= 1e-8).item()
            ),
            "absolute_spectral_condition_number": condition,
        }


__all__ = [
    "ControlMode",
    "ExactBinaryTreeProblem",
    "ExactTargetTree",
    "ExactTreeConfig",
    "ExactTreeSolution",
    "build_delayed_memory_target",
]
