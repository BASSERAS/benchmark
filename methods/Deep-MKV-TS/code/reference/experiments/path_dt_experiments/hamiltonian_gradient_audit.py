from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from deep_mkv_gen_path_dt.controls import (
    SpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from path_dt_experiments.lifted_collocation_solver import (
    CausalControlPolicy,
    LawEvaluation,
    PolicyStepResult,
    ScenarioCollocationBank,
    ScenarioDataContext,
    _control_safety,
    policy_moments_on_paths,
    projected_log_volatility_kkt,
    rollout_causal_policy,
    time_block_bounds,
)


@dataclass(frozen=True)
class EqualKLBudgetConfig:
    mean_budgets: tuple[float, ...] = (
        3.592864459278644e-6,
        1.4310129699879326e-5,
        5.6768189097056165e-5,
    )
    q95_multiplier: float = 8.0
    maximum_multiplier: float = 100.0
    bisection_steps: int = 36
    mean_budget_tolerance: float = 0.05
    maximum_scalar_step: float = 1.0
    maximum_proximal_rho: float = 1e4

    def __post_init__(self) -> None:
        budgets = tuple(float(value) for value in self.mean_budgets)
        if not budgets or any(
            not math.isfinite(value) or value <= 0.0 for value in budgets
        ):
            raise ValueError("mean_budgets must be positive and finite")
        if tuple(sorted(budgets)) != budgets:
            raise ValueError("mean_budgets must be increasing")
        object.__setattr__(self, "mean_budgets", budgets)
        for name in (
            "q95_multiplier",
            "maximum_multiplier",
            "maximum_scalar_step",
            "maximum_proximal_rho",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if float(self.maximum_multiplier) < float(self.q95_multiplier):
            raise ValueError("maximum_multiplier cannot be smaller than q95_multiplier")
        steps = int(self.bisection_steps)
        if steps < 8:
            raise ValueError("bisection_steps must be at least eight")
        object.__setattr__(self, "bisection_steps", steps)
        tolerance = float(self.mean_budget_tolerance)
        if not 0.0 <= tolerance < 1.0:
            raise ValueError("mean_budget_tolerance must lie in [0, 1)")
        object.__setattr__(self, "mean_budget_tolerance", tolerance)


@dataclass(frozen=True)
class HamiltonianDirectionPolicy:
    """Causal actor proposal driven by a fixed fitted MP critic."""

    model: DiscreteMPModel
    actor_policy: CausalControlPolicy
    critic_policy: CausalControlPolicy
    method: str
    signal: str
    block_index: int
    blocks: tuple[tuple[int, int], ...]
    parameter: float

    def __post_init__(self) -> None:
        if self.method not in {"full", "raw", "kl"}:
            raise ValueError("method must be full, raw, or kl")
        if self.signal not in {"alpha", "log_sigma"}:
            raise ValueError("signal must be alpha or log_sigma")
        if not 0 <= int(self.block_index) < len(self.blocks):
            raise ValueError("block_index is outside blocks")
        parameter = float(self.parameter)
        if not math.isfinite(parameter) or parameter <= 0.0:
            raise ValueError("proposal parameter must be positive and finite")
        object.__setattr__(self, "parameter", parameter)
        if not isinstance(self.model.control_map, SpecificEntropyDiagonalControl):
            raise TypeError("Hamiltonian directions require specific entropy")
        if self.model.control_map.sigma_max is None:
            raise ValueError("Hamiltonian directions require bounded volatility")

    def initial_state(self, *, batch_size, device, dtype):
        return (
            self.actor_policy.initial_state(
                batch_size=batch_size, device=device, dtype=dtype
            ),
            self.critic_policy.initial_state(
                batch_size=batch_size, device=device, dtype=dtype
            ),
        )

    def control_step(self, *, x_prefix, step_index, state):
        actor = self.actor_policy.control_step(
            x_prefix=x_prefix, step_index=step_index, state=state[0]
        )
        critic = self.critic_policy.control_step(
            x_prefix=x_prefix, step_index=step_index, state=state[1]
        )
        current = actor.control
        if (
            critic.expected_adjoint_next is None
            or critic.expected_adjoint_noise_next is None
        ):
            raise TypeError("critic policy must expose denormalized P and R")
        left, right = self.blocks[int(self.block_index)]
        if not left <= int(step_index) < right:
            return PolicyStepResult(
                control=current,
                state=(actor.state, critic.state),
            )
        state_dim = int(self.model.architecture.state_dim)
        current_alpha = current[..., :state_dim]
        current_sigma = current[..., state_dim:]
        control_map = self.model.control_map
        assert isinstance(control_map, SpecificEntropyDiagonalControl)
        if self.method == "full":
            target = critic.control
            target_alpha = target[..., :state_dim]
            target_log_sigma = torch.log(target[..., state_dim:])
            alpha = current_alpha + float(self.parameter) * (
                target_alpha - current_alpha
            )
            log_sigma = torch.log(current_sigma) + float(self.parameter) * (
                target_log_sigma - torch.log(current_sigma)
            )
        elif self.method == "raw":
            p = critic.expected_adjoint_next
            r = critic.expected_adjoint_noise_next
            sigma_ref = control_map._reference_sigma(
                x_prefix, step_index=int(step_index)
            )
            g_alpha = current_alpha + p
            denominator = float(control_map.eta) / sigma_ref + r / math.sqrt(
                float(control_map.dt)
            )
            g_log_sigma = current_sigma * denominator - float(control_map.eta)
            direction_log_sigma = -g_log_sigma
            lower = float(control_map.sigma_min)
            upper = float(control_map.sigma_max)
            direction_log_sigma = torch.where(
                (current_sigma <= lower * (1.0 + 1e-6))
                & (direction_log_sigma < 0.0),
                torch.zeros_like(direction_log_sigma),
                direction_log_sigma,
            )
            direction_log_sigma = torch.where(
                (current_sigma >= upper * (1.0 - 1e-6))
                & (direction_log_sigma > 0.0),
                torch.zeros_like(direction_log_sigma),
                direction_log_sigma,
            )
            alpha = current_alpha - float(self.parameter) * g_alpha
            log_sigma = torch.log(current_sigma) + float(
                self.parameter
            ) * direction_log_sigma
        else:
            # Exact mirror-gradient step for
            #   <grad H, d> + KL(p_{u+d} || p_u) / rho.
            # This is deliberately not the existing full-Hamiltonian proximal
            # control, because the audit isolates local MP-gradient geometry.
            p = critic.expected_adjoint_next
            r = critic.expected_adjoint_noise_next
            sigma_ref = control_map._reference_sigma(
                x_prefix, step_index=int(step_index)
            )
            g_alpha = current_alpha + p
            denominator = float(control_map.eta) / sigma_ref + r / math.sqrt(
                float(control_map.dt)
            )
            g_log_sigma = current_sigma * denominator - float(control_map.eta)
            rho = float(self.parameter)
            alpha = current_alpha - (
                rho
                * current_sigma.pow(2)
                / float(control_map.dt)
                * g_alpha
            )
            variance_ratio = 1.0 - rho * g_log_sigma
            lower_log_sigma = math.log(float(control_map.sigma_min))
            upper_log_sigma = math.log(float(control_map.sigma_max))
            unconstrained_log_sigma = torch.log(current_sigma) + 0.5 * torch.log(
                variance_ratio.clamp_min(torch.finfo(current_sigma.dtype).tiny)
            )
            log_sigma = torch.where(
                variance_ratio > 0.0,
                unconstrained_log_sigma,
                torch.full_like(unconstrained_log_sigma, lower_log_sigma),
            ).clamp(min=lower_log_sigma, max=upper_log_sigma)
        log_sigma = log_sigma.clamp(
            min=math.log(float(control_map.sigma_min)),
            max=math.log(float(control_map.sigma_max)),
        )
        if self.signal == "alpha":
            control = torch.cat((alpha, current_sigma), dim=-1)
        else:
            control = torch.cat((current_alpha, torch.exp(log_sigma)), dim=-1)
        return PolicyStepResult(
            control=control,
            state=(actor.state, critic.state),
        )


def _candidate_rollout_on_bank(
    *,
    model: DiscreteMPModel,
    policy: CausalControlPolicy,
    data_context: ScenarioDataContext,
    bank: ScenarioCollocationBank,
):
    bank.verify(data_context)
    initial = data_context.initial_state_pool.index_select(
        0, bank.initial_indices.to(data_context.initial_state_pool.device)
    )
    x0 = initial[:, 0, :].to(device=model.device, dtype=model.dtype)
    noise = bank.base_noise.to(device=model.device, dtype=model.dtype)
    with torch.no_grad():
        return rollout_causal_policy(model=model, policy=policy, x0=x0, noise=noise)


def proposal_kl_metrics(
    *,
    model: DiscreteMPModel,
    actor_policy: CausalControlPolicy,
    candidate_rollout,
) -> dict[str, float]:
    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise TypeError("KL metrics require specific entropy")
    with torch.no_grad():
        _, _, current_controls = policy_moments_on_paths(
            model=model,
            policy=actor_policy,
            paths=candidate_rollout.paths,
        )
        kl = model.control_map.transition_kernel_kl(
            controls=candidate_rollout.controls,
            current_controls=current_controls,
        ).detach()
    state_dim = int(model.architecture.state_dim)
    sigma = candidate_rollout.controls[..., state_dim:]
    return {
        "mean": float(kl.mean().item()),
        "q95": float(torch.quantile(kl.reshape(-1), 0.95).item()),
        "maximum": float(kl.max().item()),
        "sigma_min": float(sigma.min().item()),
        "sigma_max": float(sigma.max().item()),
        "sigma_lower_activity": float(
            torch.mean(
                (
                    sigma
                    <= float(model.control_map.sigma_min) * (1.0 + 1e-6)
                ).float()
            ).item()
        ),
        "sigma_upper_activity": float(
            torch.mean(
                (
                    sigma
                    >= float(model.control_map.sigma_max) * (1.0 - 1e-6)
                ).float()
            ).item()
        ),
    }


def calibrate_direction_to_kl_budget(
    *,
    model: DiscreteMPModel,
    actor_policy: CausalControlPolicy,
    critic_policy: CausalControlPolicy,
    data_context: ScenarioDataContext,
    fit_bank: ScenarioCollocationBank,
    method: str,
    signal: str,
    block_index: int,
    blocks: tuple[tuple[int, int], ...],
    budget: float,
    config: EqualKLBudgetConfig,
) -> tuple[float, dict[str, object]]:
    """Fit one proposal parameter using KL information from the fit bank only."""

    target = float(budget)
    q95_ceiling = target * float(config.q95_multiplier)
    maximum_ceiling = target * float(config.maximum_multiplier)
    evaluations = []

    def evaluate(parameter: float) -> tuple[bool, dict[str, float]]:
        policy = HamiltonianDirectionPolicy(
            model=model,
            actor_policy=actor_policy,
            critic_policy=critic_policy,
            method=method,
            signal=signal,
            block_index=int(block_index),
            blocks=blocks,
            parameter=float(parameter),
        )
        rollout = _candidate_rollout_on_bank(
            model=model,
            policy=policy,
            data_context=data_context,
            bank=fit_bank,
        )
        metrics = proposal_kl_metrics(
            model=model,
            actor_policy=actor_policy,
            candidate_rollout=rollout,
        )
        admissible = bool(
            math.isfinite(metrics["mean"])
            and metrics["mean"] <= target
            and metrics["q95"] <= q95_ceiling
            and metrics["maximum"] <= maximum_ceiling
            and metrics["sigma_min"]
            >= float(model.control_map.sigma_min) * (1.0 - 1e-6)
            and metrics["sigma_max"]
            <= float(model.control_map.sigma_max) * (1.0 + 1e-6)
        )
        evaluations.append(
            {"parameter": float(parameter), "admissible": admissible, **metrics}
        )
        return admissible, metrics

    low = 0.0
    if method == "kl":
        high = 1.0
        while high < float(config.maximum_proximal_rho):
            admissible, metrics = evaluate(high)
            if not admissible or float(metrics["mean"]) >= target * (
                1.0 - float(config.mean_budget_tolerance)
            ):
                break
            low = high
            high = min(high * 10.0, float(config.maximum_proximal_rho))
        if high == low:
            high = float(config.maximum_proximal_rho)
    else:
        high = float(config.maximum_scalar_step)
    best_parameter = 0.0
    best_metrics = {
        "mean": 0.0,
        "q95": 0.0,
        "maximum": 0.0,
        "sigma_min": float("nan"),
        "sigma_max": float("nan"),
        "sigma_lower_activity": 0.0,
        "sigma_upper_activity": 0.0,
    }
    if low > 0.0:
        admissible, metrics = evaluate(low)
        if admissible:
            best_parameter = low
            best_metrics = metrics
    for _ in range(int(config.bisection_steps)):
        middle = 0.5 * (low + high)
        if middle <= 0.0:
            break
        admissible, metrics = evaluate(middle)
        if admissible:
            low = middle
            best_parameter = middle
            best_metrics = metrics
        else:
            high = middle
    if best_parameter <= 0.0:
        raise RuntimeError("no positive proposal parameter satisfies the KL ceilings")
    exhausted = bool(
        float(best_metrics["mean"])
        >= target * (1.0 - float(config.mean_budget_tolerance))
    )
    limiting = []
    if not exhausted:
        if float(best_metrics["q95"]) >= q95_ceiling * 0.95:
            limiting.append("q95_ceiling")
        if float(best_metrics["maximum"]) >= maximum_ceiling * 0.95:
            limiting.append("maximum_ceiling")
        if method != "kl" and best_parameter >= float(
            config.maximum_scalar_step
        ) * 0.999:
            limiting.append("maximum_scalar_step")
        if method == "kl" and best_parameter >= float(
            config.maximum_proximal_rho
        ) * 0.999:
            limiting.append("maximum_proximal_rho")
    return best_parameter, {
        "method": method,
        "signal": signal,
        "block_index": int(block_index),
        "target_mean_kl": target,
        "q95_ceiling": q95_ceiling,
        "maximum_ceiling": maximum_ceiling,
        "selected_parameter": best_parameter,
        "selected_fit_kl": best_metrics,
        "mean_budget_exhausted": exhausted,
        "limiting_constraints": limiting,
        "fit_bank_only": True,
        "evaluation_count": len(evaluations),
        "trace": evaluations,
    }


def audit_direction_evaluation(
    *,
    model: DiscreteMPModel,
    evaluation: LawEvaluation,
    actor_policy: CausalControlPolicy,
    critic_policy: CausalControlPolicy,
    baseline_objective: float,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Audit deployed-critic and refreshed-target stationarity separately."""

    with torch.no_grad():
        critic_p, critic_r, _ = policy_moments_on_paths(
            model=model,
            policy=critic_policy,
            paths=evaluation.paths,
        )
        _, _, actor_controls = policy_moments_on_paths(
            model=model,
            policy=actor_policy,
            paths=evaluation.paths,
        )
    state_dim = int(model.architecture.state_dim)
    alpha = evaluation.controls[..., :state_dim]
    sigma = evaluation.controls[..., state_dim:]
    deployed_vol = projected_log_volatility_kkt(
        model=model,
        paths=evaluation.paths,
        sigma=sigma,
        moment_r=critic_r,
        gamma=1.0,
    )
    refreshed_vol = projected_log_volatility_kkt(
        model=model,
        paths=evaluation.paths,
        sigma=sigma,
        moment_r=evaluation.target_r,
        gamma=1.0,
    )
    deployed_drift = alpha + critic_p
    refreshed_drift = alpha + evaluation.target_p
    p_ce = critic_p - evaluation.target_p
    r_ce = critic_r - evaluation.target_r
    control = model.control_map
    assert isinstance(control, SpecificEntropyDiagonalControl)
    kl = control.transition_kernel_kl(
        controls=evaluation.controls,
        current_controls=actor_controls,
    ).detach()
    alpha_move = alpha - actor_controls[..., :state_dim]
    log_sigma_move = torch.log(sigma) - torch.log(actor_controls[..., state_dim:])
    objective = float(evaluation.objective["complete_objective_value"])
    objective_improvement = float(baseline_objective) - objective
    mean_kl = float(kl.mean().item())
    safety = _control_safety(
        model=model,
        paths=evaluation.paths,
        controls=evaluation.controls,
    )

    def rms(value: torch.Tensor) -> float:
        return float(torch.mean(value.detach().double().pow(2)).sqrt().item())

    report = {
        "complete_objective_value": objective,
        "objective_change": objective - float(baseline_objective),
        "objective_improvement": objective_improvement,
        "objective_improvement_per_mean_kl": objective_improvement
        / max(mean_kl, torch.finfo(torch.float64).eps),
        "p_ce_residual_rms": rms(p_ce),
        "r_ce_residual_rms": rms(r_ce),
        "deployed_critic_drift_kkt_rms": rms(deployed_drift),
        "deployed_critic_volatility_kkt_rms": rms(deployed_vol.residual),
        "refreshed_target_drift_kkt_rms": rms(refreshed_drift),
        "refreshed_target_volatility_kkt_rms": rms(refreshed_vol.residual),
        "drift_move_rms": rms(alpha_move),
        "log_volatility_move_rms": rms(log_sigma_move),
        "transition_kl_mean": mean_kl,
        "transition_kl_q95": float(
            torch.quantile(kl.reshape(-1), 0.95).item()
        ),
        "transition_kl_max": float(kl.max().item()),
        "candidate_control_safety": safety,
        "deployed_critic_denominator_min": float(
            deployed_vol.denominator.min().item()
        ),
        "refreshed_target_denominator_min": float(
            refreshed_vol.denominator.min().item()
        ),
        "candidate_sigma_lower_activity": float(
            torch.mean(
                (
                    sigma <= float(control.sigma_min) * (1.0 + 1e-6)
                ).float()
            ).item()
        ),
        "candidate_sigma_upper_activity": float(
            torch.mean(
                (
                    sigma >= float(control.sigma_max) * (1.0 - 1e-6)
                ).float()
            ).item()
        ),
    }
    return report, {
        "p_ce_residual": p_ce.detach().cpu(),
        "r_ce_residual": r_ce.detach().cpu(),
        "deployed_critic_drift_kkt": deployed_drift.detach().cpu(),
        "deployed_critic_volatility_kkt": deployed_vol.residual.cpu(),
        "refreshed_target_drift_kkt": refreshed_drift.detach().cpu(),
        "refreshed_target_volatility_kkt": refreshed_vol.residual.cpu(),
        "transition_kl": kl.cpu(),
        "alpha_move": alpha_move.detach().cpu(),
        "log_sigma_move": log_sigma_move.detach().cpu(),
    }


__all__ = [
    "EqualKLBudgetConfig",
    "HamiltonianDirectionPolicy",
    "audit_direction_evaluation",
    "calibrate_direction_to_kl_budget",
    "proposal_kl_metrics",
]
