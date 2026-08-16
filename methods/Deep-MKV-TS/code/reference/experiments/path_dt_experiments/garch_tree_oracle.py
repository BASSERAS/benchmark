from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from deep_mkv_gen_path_dt.discrepancies.mmd import rbf_mmd2


@dataclass(frozen=True)
class GarchTreeConfig:
    """Configuration for a finite, exactly enumerated GARCH-style tree.

    The three equally likely innovations have mean zero and variance one.  Equal
    probabilities keep every exact-population calculation transparent while the
    zero branch and two non-zero branches generate genuinely different squared
    returns (unlike a symmetric binary tree).
    """

    horizon: int = 6
    dt: float = 1.0 / 6.0
    long_run_sigma: float = 0.25
    arch: float = 0.35
    garch: float = 0.60
    sigma_min: float = 0.05
    sigma_max: float = 0.70
    eta: float = 1.0
    lambda_scale: float = 50.0
    bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0)
    innovations: tuple[float, ...] = (
        -math.sqrt(1.5),
        0.0,
        math.sqrt(1.5),
    )

    def __post_init__(self) -> None:
        if int(self.horizon) < 2:
            raise ValueError("horizon must be at least two")
        if not math.isfinite(float(self.dt)) or float(self.dt) <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not 0.0 < float(self.sigma_min) < float(self.long_run_sigma):
            raise ValueError("sigma_min must be below long_run_sigma")
        if not float(self.long_run_sigma) < float(self.sigma_max):
            raise ValueError("sigma_max must be above long_run_sigma")
        if float(self.arch) < 0.0 or float(self.garch) < 0.0:
            raise ValueError("GARCH coefficients must be nonnegative")
        if float(self.arch) + float(self.garch) >= 1.0:
            raise ValueError("the GARCH recursion must be covariance stationary")
        if float(self.eta) <= 0.0 or float(self.lambda_scale) <= 0.0:
            raise ValueError("eta and lambda_scale must be positive")
        if len(self.innovations) < 2:
            raise ValueError("at least two innovations are required")
        values = torch.tensor(self.innovations, dtype=torch.float64)
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("innovations must be finite")
        if abs(float(values.mean().item())) > 1e-12:
            raise ValueError("innovations must have mean zero")
        if abs(float(values.pow(2).mean().item()) - 1.0) > 1e-12:
            raise ValueError("innovations must have unit variance")
        for bandwidth in self.bandwidths:
            if not math.isfinite(float(bandwidth)) or float(bandwidth) <= 0.0:
                raise ValueError("bandwidths must be finite and positive")

    @property
    def branching(self) -> int:
        return len(self.innovations)

    @property
    def leaf_count(self) -> int:
        return int(self.branching) ** int(self.horizon)

    @property
    def decision_nodes(self) -> int:
        branching = int(self.branching)
        return (branching ** int(self.horizon) - 1) // (branching - 1)

    @property
    def omega(self) -> float:
        return (
            1.0 - float(self.arch) - float(self.garch)
        ) * float(self.long_run_sigma) ** 2


@dataclass(frozen=True)
class GarchTargetTree:
    state_levels: tuple[torch.Tensor, ...]
    sigma_levels: tuple[torch.Tensor, ...]
    leaf_paths: torch.Tensor
    reference_sigma: torch.Tensor


@dataclass(frozen=True)
class GarchTreeSolution:
    sigma_levels: tuple[torch.Tensor, ...]
    state_levels: tuple[torch.Tensor, ...]
    leaf_paths: torch.Tensor


def _children(
    parent: torch.Tensor,
    sigma: torch.Tensor,
    *,
    dt: float,
    innovations: torch.Tensor,
) -> torch.Tensor:
    return (
        parent[:, None]
        + sigma[:, None] * math.sqrt(float(dt)) * innovations[None, :]
    ).reshape(-1)


def _leaf_paths(
    state_levels: tuple[torch.Tensor, ...], *, branching: int
) -> torch.Tensor:
    horizon = len(state_levels) - 1
    leaves = int(branching) ** horizon
    columns = []
    for level, states in enumerate(state_levels):
        repeat = int(branching) ** (horizon - level)
        columns.append(states.repeat_interleave(repeat))
    return torch.stack(columns, dim=1).reshape(leaves, horizon + 1)


def build_garch_target(
    config: GarchTreeConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> GarchTargetTree:
    device = torch.device(device)
    innovations = torch.tensor(
        config.innovations, device=device, dtype=dtype
    )
    states = [torch.zeros(1, device=device, dtype=dtype)]
    sigmas = [
        torch.full(
            (1,), float(config.long_run_sigma), device=device, dtype=dtype
        )
    ]
    for step in range(int(config.horizon)):
        child = _children(
            states[step],
            sigmas[step],
            dt=float(config.dt),
            innovations=innovations,
        )
        states.append(child)
        if step + 1 < int(config.horizon):
            repeated_sigma = sigmas[step].repeat_interleave(
                int(config.branching)
            )
            repeated_innovation = innovations.repeat(int(sigmas[step].numel()))
            next_variance = (
                float(config.omega)
                + float(config.garch) * repeated_sigma.pow(2)
                + float(config.arch)
                * repeated_sigma.pow(2)
                * repeated_innovation.pow(2)
            )
            sigmas.append(torch.sqrt(next_variance.clamp_min(1e-15)))

    reference_sigma = torch.stack(
        [level.pow(2).mean().sqrt() for level in sigmas]
    )
    state_tuple = tuple(states)
    return GarchTargetTree(
        state_levels=state_tuple,
        sigma_levels=tuple(sigmas),
        leaf_paths=_leaf_paths(
            state_tuple, branching=int(config.branching)
        ),
        reference_sigma=reference_sigma,
    )


class ExactGarchTreeProblem:
    """Volatility-only finite control problem on a complete GARCH tree."""

    def __init__(
        self,
        config: GarchTreeConfig,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.innovations = torch.tensor(
            config.innovations, device=self.device, dtype=self.dtype
        )
        self.target = build_garch_target(
            config, device=self.device, dtype=self.dtype
        )
        target_returns = torch.diff(self.target.leaf_paths, dim=1)
        target_squares = target_returns.pow(2)
        target_states = self.target.leaf_paths[:, 1:]
        self.return_scale = target_returns.std(
            dim=0, unbiased=False
        ).clamp_min(1e-8).detach()
        self.square_center = target_squares.mean(dim=0).detach()
        self.square_scale = target_squares.std(
            dim=0, unbiased=False
        ).clamp_min(1e-8).detach()
        self.state_scale = target_states.std(
            dim=0, unbiased=False
        ).clamp_min(1e-8).detach()
        self.target_features = self._features(
            self.target.leaf_paths
        ).detach()
        branching = int(config.branching)
        self._level_offsets = tuple(
            sum(branching**previous for previous in range(level))
            for level in range(int(config.horizon))
        )

    @property
    def variable_count(self) -> int:
        return int(self.config.decision_nodes)

    def level_node_probability(self, level: int) -> float:
        return float(int(self.config.branching) ** (-int(level)))

    def _features(self, leaf_paths: torch.Tensor) -> torch.Tensor:
        returns = torch.diff(leaf_paths, dim=1)
        squares = returns.pow(2)
        states = leaf_paths[:, 1:]
        return torch.cat(
            (
                returns / self.return_scale.to(returns),
                (squares - self.square_center.to(squares))
                / self.square_scale.to(squares),
                states / self.state_scale.to(states),
            ),
            dim=1,
        )

    def unpack(self, variables: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if variables.ndim != 1 or int(variables.numel()) != self.variable_count:
            raise ValueError("variables have the wrong dimension")
        levels = []
        branching = int(self.config.branching)
        for level, offset in enumerate(self._level_offsets):
            count = branching**level
            levels.append(torch.exp(variables[offset : offset + count]))
        return tuple(levels)

    def pack(self, sigma_levels: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(sigma_levels) != int(self.config.horizon):
            raise ValueError("one volatility tensor is required per level")
        return torch.cat([torch.log(value) for value in sigma_levels])

    def reference_start(self) -> torch.Tensor:
        branching = int(self.config.branching)
        return self.pack(
            tuple(
                torch.full(
                    (branching**level,),
                    float(self.target.reference_sigma[level].item()),
                    device=self.device,
                    dtype=self.dtype,
                )
                for level in range(int(self.config.horizon))
            )
        )

    def target_start(self) -> torch.Tensor:
        return self.pack(self.target.sigma_levels)

    def bounds(self) -> list[tuple[float, float]]:
        return [
            (
                math.log(float(self.config.sigma_min)),
                math.log(float(self.config.sigma_max)),
            )
            for _ in range(self.variable_count)
        ]

    def solution(self, variables: torch.Tensor) -> GarchTreeSolution:
        sigma_levels = self.unpack(variables)
        states = [
            torch.zeros(1, device=variables.device, dtype=variables.dtype)
        ]
        for level, sigma in enumerate(sigma_levels):
            states.append(
                _children(
                    states[level],
                    sigma,
                    dt=float(self.config.dt),
                    innovations=self.innovations.to(variables),
                )
            )
        state_tuple = tuple(states)
        return GarchTreeSolution(
            sigma_levels=sigma_levels,
            state_levels=state_tuple,
            leaf_paths=_leaf_paths(
                state_tuple, branching=int(self.config.branching)
            ),
        )

    def target_solution(self) -> GarchTreeSolution:
        return GarchTreeSolution(
            sigma_levels=self.target.sigma_levels,
            state_levels=self.target.state_levels,
            leaf_paths=self.target.leaf_paths,
        )

    def objective_components_for_solution(
        self, solution: GarchTreeSolution
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mmd = rbf_mmd2(
            self._features(solution.leaf_paths),
            self.target_features,
            bandwidths=self.config.bandwidths,
            unbiased=False,
        )
        volatility_cost = solution.leaf_paths.new_tensor(0.0)
        for level, sigma in enumerate(solution.sigma_levels):
            probability = self.level_node_probability(level)
            sigma_ref = self.target.reference_sigma[level].to(sigma)
            ratio = sigma / sigma_ref
            volatility_cost = volatility_cost + (
                probability
                * float(self.config.dt)
                * float(self.config.eta)
                * (ratio - 1.0 - torch.log(ratio)).sum()
            )
        discrepancy = float(self.config.lambda_scale) * mmd
        total = volatility_cost + discrepancy
        return total, {
            "total": total,
            "volatility_cost": volatility_cost,
            "mmd2": mmd,
            "scaled_discrepancy": discrepancy,
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
            node_probability = self.level_node_probability(level)
            p_levels.append((child.sum(dim=1) / node_probability).detach())
            r_levels.append(
                (
                    (child * innovations[None, :]).sum(dim=1)
                    / node_probability
                ).detach()
            )
        return tuple(p_levels), tuple(r_levels)

    def sigma_from_r(self, r: torch.Tensor, *, level: int) -> torch.Tensor:
        sigma_ref = self.target.reference_sigma[int(level)].to(r)
        denominator = float(self.config.eta) / sigma_ref + r / math.sqrt(
            float(self.config.dt)
        )
        unconstrained = float(self.config.eta) / denominator.clamp_min(1e-12)
        return torch.where(
            denominator > 1e-12,
            unconstrained,
            torch.full_like(unconstrained, float(self.config.sigma_max)),
        ).clamp(
            min=float(self.config.sigma_min),
            max=float(self.config.sigma_max),
        )

    def hamiltonian_best_response(self, variables: torch.Tensor) -> torch.Tensor:
        solution = self.solution(variables.detach())
        _, r_levels = self.conditional_adjoint_targets_for_solution(solution)
        sigma = tuple(
            self.sigma_from_r(r, level=level)
            for level, r in enumerate(r_levels)
        )
        return self.pack(sigma).detach()

    def exact_adjoint_audit(self, variables: torch.Tensor) -> dict[str, object]:
        point = variables.detach().clone().requires_grad_(True)
        solution = self.solution(point)
        p_levels, r_levels = self.conditional_adjoint_targets_for_solution(
            solution
        )
        formula = []
        log_best_response = []
        for level, (sigma, r) in enumerate(
            zip(solution.sigma_levels, r_levels)
        ):
            probability = self.level_node_probability(level)
            sigma_ref = self.target.reference_sigma[level].to(sigma)
            physical_gradient = probability * float(self.config.dt) * (
                float(self.config.eta) * (1.0 / sigma_ref - 1.0 / sigma)
                + r / math.sqrt(float(self.config.dt))
            )
            formula.append(physical_gradient * sigma)
            best = self.sigma_from_r(r, level=level)
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


__all__ = [
    "ExactGarchTreeProblem",
    "GarchTargetTree",
    "GarchTreeConfig",
    "GarchTreeSolution",
    "build_garch_target",
]
