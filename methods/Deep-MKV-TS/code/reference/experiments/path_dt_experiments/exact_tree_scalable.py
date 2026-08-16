from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import time

import torch
import torch.nn as nn

from path_dt_experiments.exact_scenario_tree import (
    ExactBinaryTreeProblem,
    ExactTreeSolution,
    _children,
    _leaf_paths,
)


class CausalVolatilityPolicy(nn.Module):
    """Causal GRU coordinate mapped to volatility by the MP Hamiltonian."""

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
        self.r_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        final = self.r_head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("final R head layer must be linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def step(
        self,
        states: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, next_hidden = self.gru(states.reshape(-1, 1, 1), hidden)
        return self.r_head(output[:, -1, :])[:, 0], next_hidden


@dataclass(frozen=True)
class ScalableFitResult:
    state_dict: dict[str, torch.Tensor]
    history: tuple[dict[str, float], ...]
    resources: dict[str, float]
    stable: bool


def _sigma_from_r(
    problem: ExactBinaryTreeProblem,
    r: torch.Tensor,
    *,
    level: int,
) -> torch.Tensor:
    sigma_ref = problem.target.reference_sigma[int(level)].to(r)
    denominator = float(problem.config.eta) / sigma_ref + r / math.sqrt(
        float(problem.config.dt)
    )
    unconstrained = float(problem.config.eta) / denominator.clamp_min(1e-12)
    return torch.where(
        denominator > 1e-12,
        unconstrained,
        torch.full_like(unconstrained, float(problem.config.sigma_max)),
    ).clamp(
        min=float(problem.config.sigma_min),
        max=float(problem.config.sigma_max),
    )


def rollout_causal_policy(
    problem: ExactBinaryTreeProblem,
    policy: CausalVolatilityPolicy,
) -> tuple[ExactTreeSolution, tuple[torch.Tensor, ...]]:
    if problem.mode != "volatility_only":
        raise ValueError("the scalable policy audit is volatility-only")
    states = [
        torch.zeros(1, device=problem.device, dtype=problem.dtype)
    ]
    alpha_levels = []
    sigma_levels = []
    r_levels = []
    hidden = None
    for level in range(int(problem.config.horizon)):
        r, next_hidden = policy.step(states[level], hidden)
        sigma = _sigma_from_r(problem, r, level=level)
        alpha = torch.zeros_like(sigma)
        states.append(
            _children(
                states[level],
                alpha,
                sigma,
                float(problem.config.dt),
            )
        )
        alpha_levels.append(alpha)
        sigma_levels.append(sigma)
        r_levels.append(r)
        hidden = next_hidden.repeat_interleave(2, dim=1)
    state_tuple = tuple(states)
    return (
        ExactTreeSolution(
            mode="volatility_only",
            alpha_levels=tuple(alpha_levels),
            sigma_levels=tuple(sigma_levels),
            state_levels=state_tuple,
            leaf_paths=_leaf_paths(state_tuple),
        ),
        tuple(r_levels),
    )


def predict_r_on_fixed_tree(
    policy: CausalVolatilityPolicy,
    state_levels: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    predictions = []
    hidden = None
    for states in state_levels[:-1]:
        r, next_hidden = policy.step(states.detach(), hidden)
        predictions.append(r)
        hidden = next_hidden.repeat_interleave(2, dim=1)
    return tuple(predictions)


def _normalized_r_loss(
    predictions: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    scales: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if not (len(predictions) == len(targets) == len(scales)):
        raise ValueError("prediction, target, and scale levels must align")
    loss = predictions[0].new_tensor(0.0)
    for level, (prediction, target, scale) in enumerate(
        zip(predictions, targets, scales)
    ):
        probability = 2.0 ** (-level)
        loss = loss + probability * torch.sum(
            ((prediction - target.to(prediction)) / scale.to(prediction)).pow(2)
        )
    return loss / float(len(predictions))


def _target_scales(
    targets: tuple[torch.Tensor, ...],
    *,
    floor: float = 1e-6,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        target.detach().double().pow(2).mean().sqrt().clamp_min(float(floor))
        for target in targets
    )


def _clone_state_dict(policy: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in policy.state_dict().items()
    }


def _interpolated_state_dict(
    accepted: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    fraction: float,
) -> dict[str, torch.Tensor]:
    return {
        name: accepted[name] + float(fraction) * (candidate[name] - accepted[name])
        for name in accepted
    }


def _resource_start(device: torch.device) -> tuple[float, int]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = int(torch.cuda.memory_allocated(device))
    else:
        baseline = 0
    return time.perf_counter(), baseline


def _resource_end(device: torch.device, started: float, baseline: int) -> dict[str, float]:
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


def fit_nested_exact_mp(
    problem: ExactBinaryTreeProblem,
    initial_policy: CausalVolatilityPolicy,
    *,
    outer_steps: int = 8,
    inner_steps: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    line_fractions: tuple[float, ...] = (
        0.0,
        1.0 / 512.0,
        1.0 / 256.0,
        1.0 / 128.0,
        1.0 / 64.0,
        1.0 / 32.0,
        1.0 / 16.0,
        1.0 / 8.0,
        1.0 / 4.0,
        1.0 / 2.0,
        1.0,
    ),
) -> ScalableFitResult:
    if min(int(outer_steps), int(inner_steps)) < 1:
        raise ValueError("nested step counts must be positive")
    if float(lr) <= 0.0 or float(weight_decay) < 0.0:
        raise ValueError("optimizer settings are invalid")
    fractions = tuple(sorted(set(float(value) for value in line_fractions)))
    if not fractions or fractions[0] != 0.0 or fractions[-1] != 1.0:
        raise ValueError("line fractions must include zero and one")

    accepted = deepcopy(initial_policy).to(device=problem.device, dtype=problem.dtype)
    accepted.gru.flatten_parameters()
    history = []
    stable = True
    started, baseline = _resource_start(problem.device)
    for outer in range(int(outer_steps)):
        accepted_solution, _ = rollout_causal_policy(problem, accepted)
        objective_before = float(
            problem.objective_components_for_solution(accepted_solution)[0]
            .detach()
            .cpu()
            .item()
        )
        _, targets = problem.conditional_adjoint_targets_for_solution(
            accepted_solution
        )
        scales = _target_scales(targets)
        candidate = deepcopy(accepted)
        candidate.gru.flatten_parameters()
        optimizer = torch.optim.AdamW(
            candidate.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )
        loss_before = None
        loss_after = None
        for inner in range(int(inner_steps)):
            predictions = predict_r_on_fixed_tree(
                candidate, accepted_solution.state_levels
            )
            loss = _normalized_r_loss(predictions, targets, scales)
            if not bool(torch.isfinite(loss.detach()).item()):
                stable = False
                break
            if loss_before is None:
                loss_before = float(loss.detach().cpu().item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_after = float(loss.detach().cpu().item())
        if not stable:
            break
        with torch.no_grad():
            loss_after = float(
                _normalized_r_loss(
                    predict_r_on_fixed_tree(
                        candidate, accepted_solution.state_levels
                    ),
                    targets,
                    scales,
                )
                .detach()
                .cpu()
                .item()
            )

        accepted_state = _clone_state_dict(accepted)
        candidate_state = _clone_state_dict(candidate)
        line_rows = []
        best = None
        trial = deepcopy(accepted)
        trial.gru.flatten_parameters()
        for fraction in fractions:
            trial.load_state_dict(
                _interpolated_state_dict(
                    accepted_state, candidate_state, float(fraction)
                )
            )
            trial_solution, _ = rollout_causal_policy(problem, trial)
            value = float(
                problem.objective_components_for_solution(trial_solution)[0]
                .detach()
                .cpu()
                .item()
            )
            row = {"fraction": float(fraction), "objective": value}
            line_rows.append(row)
            if math.isfinite(value) and (
                best is None or (value, float(fraction)) < best
            ):
                best = (value, float(fraction))
        if best is None:
            stable = False
            break
        objective_after, selected_fraction = best
        accepted.load_state_dict(
            _interpolated_state_dict(
                accepted_state, candidate_state, selected_fraction
            )
        )
        history.append(
            {
                "outer_step": float(outer + 1),
                "optimizer_steps": float((outer + 1) * int(inner_steps)),
                "objective_before": objective_before,
                "objective_after": objective_after,
                "accepted_fraction": selected_fraction,
                "adjoint_loss_before": float(loss_before),
                "adjoint_loss_after": float(loss_after),
                "accepted": float(selected_fraction > 0.0),
                "line_search_evaluations": float(len(line_rows)),
            }
        )
    resources = _resource_end(problem.device, started, baseline)
    resources["optimizer_steps"] = float(len(history) * int(inner_steps))
    resources["objective_evaluations"] = float(len(history) * len(fractions))
    return ScalableFitResult(
        state_dict=_clone_state_dict(accepted),
        history=tuple(history),
        resources=resources,
        stable=bool(stable and len(history) == int(outer_steps)),
    )


def fit_direct_exact_objective(
    problem: ExactBinaryTreeProblem,
    initial_policy: CausalVolatilityPolicy,
    *,
    steps: int = 1600,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    log_every: int = 100,
) -> ScalableFitResult:
    if int(steps) < 1 or int(log_every) < 1:
        raise ValueError("step counts must be positive")
    policy = deepcopy(initial_policy).to(device=problem.device, dtype=problem.dtype)
    policy.gru.flatten_parameters()
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    best_state = _clone_state_dict(policy)
    initial_solution, _ = rollout_causal_policy(problem, policy)
    best_objective = float(
        problem.objective_components_for_solution(initial_solution)[0]
        .detach()
        .cpu()
        .item()
    )
    history = []
    stable = True
    started, baseline = _resource_start(problem.device)
    for step in range(int(steps)):
        solution, _ = rollout_causal_policy(problem, policy)
        objective = problem.objective_components_for_solution(solution)[0]
        if not bool(torch.isfinite(objective.detach()).item()):
            stable = False
            break
        objective_value = float(objective.detach().cpu().item())
        if objective_value < best_objective:
            best_objective = objective_value
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
        optimizer.step()
        if step == 0 or (step + 1) % int(log_every) == 0 or step + 1 == int(steps):
            history.append(
                {
                    "step": float(step + 1),
                    "objective_before_update": objective_value,
                    "best_objective": best_objective,
                    "gradient_norm": float(gradient_norm.cpu().item()),
                }
            )
    if stable:
        with torch.no_grad():
            final_solution, _ = rollout_causal_policy(problem, policy)
            final_objective = float(
                problem.objective_components_for_solution(final_solution)[0]
                .detach()
                .cpu()
                .item()
            )
        if math.isfinite(final_objective) and final_objective < best_objective:
            best_objective = final_objective
            best_state = _clone_state_dict(policy)
    resources = _resource_end(problem.device, started, baseline)
    resources["optimizer_steps"] = float(
        int(steps) if stable else int(history[-1]["step"] if history else 0)
    )
    resources["objective_evaluations"] = float(
        resources["optimizer_steps"] + int(stable)
    )
    return ScalableFitResult(
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
) -> CausalVolatilityPolicy:
    policy = CausalVolatilityPolicy(
        hidden_dim=int(hidden_dim), num_layers=int(num_layers)
    ).to(device=device, dtype=dtype)
    policy.load_state_dict(state_dict)
    policy.gru.flatten_parameters()
    return policy


__all__ = [
    "CausalVolatilityPolicy",
    "ScalableFitResult",
    "fit_direct_exact_objective",
    "fit_nested_exact_mp",
    "policy_from_state_dict",
    "predict_r_on_fixed_tree",
    "rollout_causal_policy",
]
