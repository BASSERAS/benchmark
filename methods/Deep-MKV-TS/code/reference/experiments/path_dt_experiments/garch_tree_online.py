from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import time

import torch
import torch.nn as nn

from path_dt_experiments.garch_tree_oracle import (
    ExactGarchTreeProblem,
    GarchTreeSolution,
    _children,
    _leaf_paths,
)


class CausalGarchAdjointPolicy(nn.Module):
    """Causal GRU with the P/R heads used by the one-stage MP solver."""

    def __init__(self, *, hidden_dim: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        if int(hidden_dim) < 1 or int(num_layers) < 1:
            raise ValueError("network dimensions must be positive")
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.p_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.r_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        for head in (self.p_head, self.r_head):
            final = head[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("the final adjoint-head layer must be linear")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def step(
        self,
        states: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, next_hidden = self.gru(states.reshape(-1, 1, 1), hidden)
        features = output[:, -1, :]
        return (
            self.p_head(features)[:, 0],
            self.r_head(features)[:, 0],
            next_hidden,
        )


@dataclass(frozen=True)
class GarchPolicyFitResult:
    state_dict: dict[str, torch.Tensor]
    history: tuple[dict[str, float], ...]
    resources: dict[str, float]
    stable: bool


def _clone_state_dict(policy: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in policy.state_dict().items()
    }


def rollout_garch_policy(
    problem: ExactGarchTreeProblem,
    policy: CausalGarchAdjointPolicy,
) -> tuple[
    GarchTreeSolution,
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]:
    states = [
        torch.zeros(1, device=problem.device, dtype=problem.dtype)
    ]
    sigma_levels = []
    p_levels = []
    r_levels = []
    hidden = None
    branching = int(problem.config.branching)
    for level in range(int(problem.config.horizon)):
        p, r, next_hidden = policy.step(states[level], hidden)
        sigma = problem.sigma_from_r(r, level=level)
        states.append(
            _children(
                states[level],
                sigma,
                dt=float(problem.config.dt),
                innovations=problem.innovations.to(sigma),
            )
        )
        sigma_levels.append(sigma)
        p_levels.append(p)
        r_levels.append(r)
        hidden = next_hidden.repeat_interleave(branching, dim=1)
    state_tuple = tuple(states)
    return (
        GarchTreeSolution(
            sigma_levels=tuple(sigma_levels),
            state_levels=state_tuple,
            leaf_paths=_leaf_paths(
                state_tuple, branching=int(problem.config.branching)
            ),
        ),
        tuple(p_levels),
        tuple(r_levels),
    )


def predict_garch_adjoint_on_fixed_tree(
    problem: ExactGarchTreeProblem,
    policy: CausalGarchAdjointPolicy,
    state_levels: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    p_levels = []
    r_levels = []
    hidden = None
    branching = int(problem.config.branching)
    for states in state_levels[:-1]:
        p, r, next_hidden = policy.step(states.detach(), hidden)
        p_levels.append(p)
        r_levels.append(r)
        hidden = next_hidden.repeat_interleave(branching, dim=1)
    return tuple(p_levels), tuple(r_levels)


def _population_mse(
    problem: ExactGarchTreeProblem,
    predictions: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if len(predictions) != len(targets):
        raise ValueError("prediction and target levels must align")
    loss = predictions[0].new_tensor(0.0)
    for level, (prediction, target) in enumerate(zip(predictions, targets)):
        loss = loss + problem.level_node_probability(level) * torch.sum(
            (prediction - target.to(prediction)).pow(2)
        )
    return loss / float(len(predictions))


def _relative_population_rmse(
    problem: ExactGarchTreeProblem,
    predictions: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
) -> float:
    error = 0.0
    energy = 0.0
    for level, (prediction, target) in enumerate(zip(predictions, targets)):
        probability = problem.level_node_probability(level)
        error += probability * float(
            torch.sum((prediction.detach() - target).double().pow(2)).item()
        )
        energy += probability * float(
            torch.sum(target.double().pow(2)).item()
        )
    return math.sqrt(error / max(energy, 1e-24))


def _resource_start(device: torch.device) -> tuple[float, int]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = int(torch.cuda.memory_allocated(device))
    else:
        baseline = 0
    return time.perf_counter(), baseline


def _resource_end(
    device: torch.device, started: float, baseline: int
) -> dict[str, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    return {
        "wall_seconds": float(time.perf_counter() - started),
        "cuda_baseline_allocated_bytes": float(baseline),
        "cuda_peak_allocated_bytes": float(peak),
        "cuda_incremental_peak_allocated_bytes": float(max(0, peak - baseline)),
    }


def fit_online_garch_mp(
    problem: ExactGarchTreeProblem,
    initial_policy: CausalGarchAdjointPolicy,
    *,
    steps: int = 1500,
    lr: float = 2e-3,
    weight_decay: float = 1e-5,
    grad_clip_norm: float | None = 5.0,
    log_every: int = 25,
) -> GarchPolicyFitResult:
    """Fit P/R online while refreshing the exact law and targets each step."""

    if int(steps) < 1 or int(log_every) < 1:
        raise ValueError("step counts must be positive")
    if float(lr) <= 0.0 or float(weight_decay) < 0.0:
        raise ValueError("optimizer settings are invalid")
    if grad_clip_norm is not None and float(grad_clip_norm) <= 0.0:
        raise ValueError("grad_clip_norm must be positive when provided")

    policy = deepcopy(initial_policy).to(
        device=problem.device, dtype=problem.dtype
    )
    policy.gru.flatten_parameters()
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    reference = float(problem.objective(problem.reference_start()).item())
    best_objective = reference
    best_step = 0
    best_state = _clone_state_dict(policy)
    history = []
    stable = True
    started, baseline = _resource_start(problem.device)
    objective_evaluations = 1

    for step in range(1, int(steps) + 1):
        solution, p_prediction, r_prediction = rollout_garch_policy(
            problem, policy
        )
        p_target, r_target = problem.conditional_adjoint_targets_for_solution(
            solution
        )
        p_loss = _population_mse(problem, p_prediction, p_target)
        r_loss = _population_mse(problem, r_prediction, r_target)
        loss = p_loss + r_loss
        if not bool(torch.isfinite(loss.detach()).item()):
            stable = False
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.sqrt(
            sum(
                parameter.grad.detach().double().pow(2).sum()
                for parameter in policy.parameters()
                if parameter.grad is not None
            )
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            stable = False
            break
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float(grad_clip_norm)
            )
        optimizer.step()

        if step == 1 or step % int(log_every) == 0 or step == int(steps):
            with torch.no_grad():
                evaluated, evaluated_p, evaluated_r = rollout_garch_policy(
                    problem, policy
                )
                objective = float(
                    problem.objective_components_for_solution(evaluated)[0]
                    .detach()
                    .cpu()
                    .item()
                )
            objective_evaluations += 1
            evaluated_p_target, evaluated_r_target = (
                problem.conditional_adjoint_targets_for_solution(evaluated)
            )
            if math.isfinite(objective) and objective < best_objective:
                best_objective = objective
                best_step = step
                best_state = _clone_state_dict(policy)
            history.append(
                {
                    "step": float(step),
                    "total_loss": float(loss.detach().cpu().item()),
                    "p_loss": float(p_loss.detach().cpu().item()),
                    "r_loss": float(r_loss.detach().cpu().item()),
                    "gradient_norm": float(gradient_norm.detach().cpu().item()),
                    "objective": objective,
                    "best_objective": best_objective,
                    "best_step": float(best_step),
                    "relative_p_residual": _relative_population_rmse(
                        problem, evaluated_p, evaluated_p_target
                    ),
                    "relative_r_residual": _relative_population_rmse(
                        problem, evaluated_r, evaluated_r_target
                    ),
                }
            )

    resources = _resource_end(problem.device, started, baseline)
    resources.update(
        {
            "optimizer_steps": float(step if stable else max(0, step - 1)),
            "objective_evaluations": float(objective_evaluations),
            "best_step": float(best_step),
        }
    )
    return GarchPolicyFitResult(
        state_dict=best_state,
        history=tuple(history),
        resources=resources,
        stable=bool(stable),
    )


def fit_direct_garch_objective(
    problem: ExactGarchTreeProblem,
    initial_policy: CausalGarchAdjointPolicy,
    *,
    steps: int = 1500,
    lr: float = 2e-3,
    weight_decay: float = 1e-5,
    grad_clip_norm: float | None = 5.0,
    log_every: int = 25,
) -> GarchPolicyFitResult:
    """Directly backpropagate the exact objective through the same policy."""

    if int(steps) < 1 or int(log_every) < 1:
        raise ValueError("step counts must be positive")
    policy = deepcopy(initial_policy).to(
        device=problem.device, dtype=problem.dtype
    )
    policy.gru.flatten_parameters()
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    reference = float(problem.objective(problem.reference_start()).item())
    best_objective = reference
    best_step = 0
    best_state = _clone_state_dict(policy)
    history = []
    stable = True
    started, baseline = _resource_start(problem.device)

    for step in range(1, int(steps) + 1):
        solution, _, _ = rollout_garch_policy(problem, policy)
        objective = problem.objective_components_for_solution(solution)[0]
        if not bool(torch.isfinite(objective.detach()).item()):
            stable = False
            break
        objective_value = float(objective.detach().cpu().item())
        if objective_value < best_objective:
            best_objective = objective_value
            best_step = step - 1
            best_state = _clone_state_dict(policy)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = torch.sqrt(
            sum(
                parameter.grad.detach().double().pow(2).sum()
                for parameter in policy.parameters()
                if parameter.grad is not None
            )
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            stable = False
            break
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float(grad_clip_norm)
            )
        optimizer.step()
        if step == 1 or step % int(log_every) == 0 or step == int(steps):
            with torch.no_grad():
                evaluated, _, _ = rollout_garch_policy(problem, policy)
                evaluated_objective = float(
                    problem.objective_components_for_solution(evaluated)[0]
                    .detach()
                    .cpu()
                    .item()
                )
            if evaluated_objective < best_objective:
                best_objective = evaluated_objective
                best_step = step
                best_state = _clone_state_dict(policy)
            history.append(
                {
                    "step": float(step),
                    "objective_before_update": objective_value,
                    "objective_after_update": evaluated_objective,
                    "best_objective": best_objective,
                    "best_step": float(best_step),
                    "gradient_norm": float(gradient_norm.detach().cpu().item()),
                }
            )

    resources = _resource_end(problem.device, started, baseline)
    resources.update(
        {
            "optimizer_steps": float(step if stable else max(0, step - 1)),
            "objective_evaluations": float(step + len(history)),
            "best_step": float(best_step),
        }
    )
    return GarchPolicyFitResult(
        state_dict=best_state,
        history=tuple(history),
        resources=resources,
        stable=bool(stable),
    )


def policy_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    hidden_dim: int,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> CausalGarchAdjointPolicy:
    policy = CausalGarchAdjointPolicy(
        hidden_dim=int(hidden_dim), num_layers=int(num_layers)
    ).to(device=device, dtype=dtype)
    policy.load_state_dict(state_dict)
    policy.gru.flatten_parameters()
    return policy


__all__ = [
    "CausalGarchAdjointPolicy",
    "GarchPolicyFitResult",
    "fit_direct_garch_objective",
    "fit_online_garch_mp",
    "policy_from_state_dict",
    "predict_garch_adjoint_on_fixed_tree",
    "rollout_garch_policy",
]
