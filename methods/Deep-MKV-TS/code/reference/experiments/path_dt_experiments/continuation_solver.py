from __future__ import annotations

from dataclasses import dataclass, replace
import math
import operator
from typing import Callable, Mapping

import torch
import torch.nn as nn

from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from deep_mkv_gen_path_dt.noise import derive_stream_seed
from path_dt_experiments.extragradient_solver import (
    SafeguardedExtragradientConfig,
    _build_operator_problem,
    _controls_for_values,
    _load_network_values,
    _moments_for_values,
    _network_values,
    _objective_bank_summary,
    _objective_replay_inputs,
    _objective_rows,
    _standard_law_bank,
    _stationarity_residual,
    fit_safeguarded_extragradient,
)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class DiscrepancyContinuationConfig:
    beta_schedule: tuple[float, ...] = (
        1.0 / 64.0,
        1.0 / 32.0,
        1.0 / 16.0,
        1.0 / 8.0,
        1.0 / 4.0,
        1.0 / 2.0,
        1.0,
    )
    stage_outer_steps: int = 2
    minimum_beta_increment: float = 1.0 / 4096.0
    maximum_subdivisions: int = 6
    minimum_stage_residual_improvement: float = 1e-4
    stationary_relative_residual_tolerance: float = 0.05
    minimum_stage_objective_improvement_fraction: float = 0.75
    maximum_stage_objective_relative_increase: float = 5e-3
    seed: int = 991_027

    def __post_init__(self) -> None:
        schedule = tuple(float(value) for value in self.beta_schedule)
        if (
            not schedule
            or any(not math.isfinite(value) for value in schedule)
            or tuple(sorted(set(schedule))) != schedule
            or schedule[0] <= 0.0
            or schedule[-1] != 1.0
            or schedule[-1] > 1.0
        ):
            raise ValueError(
                "beta_schedule must be unique, strictly increasing in (0, 1], "
                "and end at one"
            )
        object.__setattr__(self, "beta_schedule", schedule)
        for name in ("stage_outer_steps", "maximum_subdivisions"):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        for name in (
            "minimum_beta_increment",
            "minimum_stage_residual_improvement",
            "stationary_relative_residual_tolerance",
        ):
            value = _finite_float(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if float(self.minimum_beta_increment) > float(schedule[0]):
            raise ValueError("minimum_beta_increment cannot exceed the first beta")
        fraction = _finite_float(
            "minimum_stage_objective_improvement_fraction",
            self.minimum_stage_objective_improvement_fraction,
        )
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                "minimum_stage_objective_improvement_fraction must lie in [0, 1]"
            )
        object.__setattr__(
            self,
            "minimum_stage_objective_improvement_fraction",
            fraction,
        )
        tolerance = _finite_float(
            "maximum_stage_objective_relative_increase",
            self.maximum_stage_objective_relative_increase,
        )
        if tolerance < 0.0:
            raise ValueError(
                "maximum_stage_objective_relative_increase must be nonnegative"
            )
        object.__setattr__(
            self,
            "maximum_stage_objective_relative_increase",
            tolerance,
        )
        if isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        try:
            seed = int(operator.index(self.seed))
        except TypeError as exc:
            raise ValueError("seed must be an integer") from exc
        object.__setattr__(self, "seed", seed)


@dataclass(frozen=True)
class DiscrepancyContinuationResult:
    reached_beta_one: bool
    accepted_beta: float
    report: dict[str, object]
    history: list[dict[str, float]]
    artifacts: dict[str, object]


def _last_linear(module: nn.Module) -> nn.Linear:
    linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if not linears:
        raise TypeError("an adjoint output head must contain a linear layer")
    return linears[-1]


def initialize_zero_adjoint_policy(model: DiscreteMPModel) -> dict[str, float]:
    """Reset only final output layers so the policy equals the exact reference."""

    head_names = (
        "expected_adjoint_next_head",
        "expected_adjoint_noise_next_head",
    )
    with torch.no_grad():
        for name in head_names:
            head = getattr(model.network, name, None)
            if not isinstance(head, nn.Module):
                raise TypeError(
                    "continuation requires explicit P/R output heads; "
                    f"missing {name}"
                )
            final = _last_linear(head)
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()
    model._reset_target_preconditioner()
    return {
        "zeroed_p_head": 1.0,
        "zeroed_r_head": 1.0,
        "target_preconditioner_reset": 1.0,
    }


def _reference_initialization_audit(
    *,
    model: DiscreteMPModel,
    paths: torch.Tensor,
) -> dict[str, float]:
    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise ValueError("continuation currently requires specific entropy")
    probe = paths[: min(32, int(paths.shape[0]))].to(
        device=model.device, dtype=model.dtype
    )
    with torch.no_grad():
        moments = model._denormalize_moments(model.network(probe))
        p = moments.expected_adjoint_next
        r = moments.expected_adjoint_noise_next
        controls = model.control_map.controls_from_moments(
            paths=probe,
            expected_adjoint_next=p,
            expected_adjoint_noise_next=r,
        )
        state_dim = int(model.architecture.state_dim)
        reference_sigma = model.control_map._reference_sigmas_all(probe)
        alpha_error = controls[..., :state_dim].abs().amax()
        sigma_error = (
            controls[..., state_dim:] - reference_sigma
        ).abs().amax()
        p_max = p.abs().amax()
        r_max = r.abs().amax()
    audit = {
        "p_abs_max": float(p_max.item()),
        "r_abs_max": float(r_max.item()),
        "alpha_reference_max_abs_error": float(alpha_error.item()),
        "sigma_reference_max_abs_error": float(sigma_error.item()),
    }
    control_tolerance = 32.0 * torch.finfo(model.dtype).eps * max(
        1.0, float(reference_sigma.detach().abs().amax().item())
    )
    audit["control_machine_tolerance"] = float(control_tolerance)
    if (
        audit["p_abs_max"] != 0.0
        or audit["r_abs_max"] != 0.0
        or audit["alpha_reference_max_abs_error"] > control_tolerance
        or audit["sigma_reference_max_abs_error"] > control_tolerance
    ):
        raise RuntimeError(
            "beta=0 initialization must have zero adjoints and exact reference controls"
        )
    return audit


def _residual_probe(
    *,
    model: DiscreteMPModel,
    values: Mapping[str, torch.Tensor],
    fit_pool: torch.Tensor,
    validation_pool: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    extragradient: SafeguardedExtragradientConfig,
    seed: int,
) -> tuple[dict[str, object], object]:
    _load_network_values(model, dict(values))

    def law_factory(pool: torch.Tensor, count: int, law_seed: int):
        return _standard_law_bank(
            model=model,
            initial_state_pool=pool,
            count=count,
            seed=law_seed,
        )

    problem = _build_operator_problem(
        model=model,
        fit_pool=fit_pool,
        validation_pool=validation_pool,
        fit_count=int(extragradient.residual_fit_paths),
        validation_count=int(extragradient.residual_validation_paths),
        target_batch_size=int(extragradient.operator_target_batch_size),
        seed=int(seed),
        training=training,
        r_reliability_floor=float(extragradient.r_reliability_floor),
        law_factory=law_factory,
    )
    residual = _stationarity_residual(
        model=model,
        problem=problem,
        values=values,
        alpha_radius=float(extragradient.alpha_radius),
        log_sigma_radius=float(extragradient.log_sigma_radius),
    )
    return residual, problem


def _stage_control_trace(
    *,
    model: DiscreteMPModel,
    paths: torch.Tensor,
    before_values: Mapping[str, torch.Tensor],
    after_values: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    before = _controls_for_values(
        model=model, values=before_values, paths=paths
    )
    after = _controls_for_values(model=model, values=after_values, paths=paths)
    state_dim = int(model.architecture.state_dim)
    alpha_move = torch.sqrt(
        torch.mean(
            (after[..., :state_dim] - before[..., :state_dim])
            .detach()
            .double()
            .pow(2)
        )
    )
    log_sigma_move = torch.sqrt(
        torch.mean(
            (
                torch.log(after[..., state_dim:])
                - torch.log(before[..., state_dim:])
            )
            .detach()
            .double()
            .pow(2)
        )
    )
    sigma = after[..., state_dim:]
    control = model.control_map
    assert isinstance(control, SpecificEntropyDiagonalControl)
    lower = float(control.sigma_min)
    lower_activity = torch.mean((sigma <= lower * (1.0 + 1e-6)).float())
    if control.sigma_max is None:
        upper_activity = sigma.new_tensor(0.0)
    else:
        upper_activity = torch.mean(
            (sigma >= float(control.sigma_max) * (1.0 - 1e-6)).float()
        )
    _, after_r = _moments_for_values(
        model=model, values=after_values, paths=paths
    )
    sigma_ref = control._reference_sigmas_all(paths)
    denominator = float(control.eta) / sigma_ref + after_r / math.sqrt(
        float(control.dt)
    )
    transition_kl = control.transition_kernel_kl(
        controls=after,
        current_controls=before,
    ).detach().double()
    return {
        "alpha_move_rms": float(alpha_move.item()),
        "log_sigma_move_rms": float(log_sigma_move.item()),
        "sigma_lower_bound_activity": float(lower_activity.item()),
        "sigma_upper_bound_activity": float(upper_activity.item()),
        "specific_entropy_denominator_min": float(denominator.min().item()),
        "conditional_transition_kl_mean": float(transition_kl.mean().item()),
        "conditional_transition_kl_q95": float(
            torch.quantile(transition_kl.reshape(-1), 0.95).item()
        ),
        "conditional_transition_kl_max": float(transition_kl.max().item()),
    }


def _mean_objective(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(sum(row[key] for row in rows) / len(rows))
        for key in (
            "complete_objective_value",
            "discrepancy_objective_value",
            "running_cost_value",
        )
    }


def fit_discrepancy_continuation(
    *,
    model: DiscreteMPModel,
    backward_target_paths: torch.Tensor,
    validation_target_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    extragradient: SafeguardedExtragradientConfig,
    continuation: DiscrepancyContinuationConfig,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> DiscrepancyContinuationResult:
    """Follow the MP stationarity branch from beta=0 to the full objective."""

    if float(training.lambda_scale) <= 0.0:
        raise ValueError("continuation requires a positive final lambda_scale")
    if str(training.target_preconditioner) != "none":
        raise ValueError("continuation preserves the raw P/R regression loss")
    if str(training.noise_target_control_variate) != "adjoint":
        raise ValueError("continuation requires the unbiased adjoint R control variate")
    model._validate_target_paths(backward_target_paths)
    model._validate_target_paths(validation_target_paths)
    fit_pool = backward_target_paths.to(device=model.device, dtype=model.dtype)
    validation_pool = validation_target_paths.to(
        device=model.device, dtype=model.dtype
    )
    initialization = initialize_zero_adjoint_policy(model)
    initialization.update(
        _reference_initialization_audit(model=model, paths=validation_pool)
    )
    initial_checkpoint = model.checkpoint_state()
    initial_values = _network_values(model)
    beta_zero_training = replace(training, lambda_scale=0.0)
    beta_zero_banks = _objective_replay_inputs(
        model=model,
        pool=validation_pool,
        config=extragradient,
        seed=int(continuation.seed) + 1_000,
    )
    beta_zero_rows = _objective_rows(
        model=model,
        values=initial_values,
        banks=beta_zero_banks,
        training=beta_zero_training,
    )
    beta_zero_objective = _mean_objective(beta_zero_rows)

    pending: list[tuple[float, int]] = [
        (float(beta), 0) for beta in continuation.beta_schedule
    ]
    accepted_beta = 0.0
    stage_reports: list[dict[str, object]] = []
    history: list[dict[str, float]] = []
    accepted_checkpoints: list[dict[str, object]] = []
    attempt = 0
    terminal_reason = "beta_one_reached"
    while pending:
        target_beta, subdivision_depth = pending.pop(0)
        if target_beta <= accepted_beta:
            continue
        beta_before = float(accepted_beta)
        delta_beta = float(target_beta - accepted_beta)
        attempt += 1
        attempt_seed = derive_stream_seed(
            base_seed=int(continuation.seed),
            stream_offset=attempt * 1_000_000,
        )
        if attempt_seed is None:
            raise RuntimeError("continuation seed derivation failed")
        stage_checkpoint = model.checkpoint_state()
        before_values = _network_values(model)
        stage_training = replace(
            training,
            lambda_scale=float(training.lambda_scale) * float(target_beta),
            seed=int(attempt_seed) + 1,
        )
        stage_extragradient = replace(
            extragradient,
            outer_steps=int(continuation.stage_outer_steps),
            seed=int(attempt_seed) + 2,
        )
        stage_banks = _objective_replay_inputs(
            model=model,
            pool=validation_pool,
            config=stage_extragradient,
            seed=int(attempt_seed) + 100_000,
        )
        objective_before_rows = _objective_rows(
            model=model,
            values=before_values,
            banks=stage_banks,
            training=stage_training,
        )
        probe_seed = int(attempt_seed) + 200_000
        residual_before, before_problem = _residual_probe(
            model=model,
            values=before_values,
            fit_pool=fit_pool,
            validation_pool=validation_pool,
            training=stage_training,
            extragradient=stage_extragradient,
            seed=probe_seed,
        )
        _load_network_values(model, before_values)
        solve = fit_safeguarded_extragradient(
            model=model,
            backward_target_paths=fit_pool,
            validation_target_paths=validation_pool,
            training=stage_training,
            config=stage_extragradient,
        )
        after_values = _network_values(model)
        residual_after, _ = _residual_probe(
            model=model,
            values=after_values,
            fit_pool=fit_pool,
            validation_pool=validation_pool,
            training=stage_training,
            extragradient=stage_extragradient,
            seed=probe_seed,
        )
        objective_after_rows = _objective_rows(
            model=model,
            values=after_values,
            banks=stage_banks,
            training=stage_training,
        )
        objective_summary = _objective_bank_summary(
            baseline_rows=objective_before_rows,
            candidate_rows=objective_after_rows,
        )
        before_relative = float(residual_before["relative_moment_residual"])
        after_relative = float(residual_after["relative_moment_residual"])
        residual_improvement = (before_relative - after_relative) / max(
            abs(before_relative), torch.finfo(torch.float64).eps
        )
        accepted_updates = int(solve.report["accepted_outer_updates"])
        stationary_without_update = bool(
            accepted_updates == 0
            and before_relative
            <= float(continuation.stationary_relative_residual_tolerance)
        )
        update_gate = bool(accepted_updates > 0 or stationary_without_update)
        residual_gate = bool(
            stationary_without_update
            or (
                math.isfinite(after_relative)
                and residual_improvement
                >= float(continuation.minimum_stage_residual_improvement)
            )
        )
        objective_relative_increase = -float(
            objective_summary["mean_relative_improvement"]
        )
        objective_gate = bool(
            objective_summary["all_finite"]
            and objective_relative_increase
            <= float(continuation.maximum_stage_objective_relative_increase)
            and float(objective_summary["improvement_fraction"])
            >= float(
                continuation.minimum_stage_objective_improvement_fraction
            )
        )
        control_trace = _stage_control_trace(
            model=model,
            paths=before_problem.validation.paths,
            before_values=before_values,
            after_values=after_values,
        )
        denominator_gate = bool(
            control_trace["specific_entropy_denominator_min"]
            >= float(stage_extragradient.minimum_denominator_margin)
        )
        stage_accepted = bool(
            update_gate and residual_gate and objective_gate and denominator_gate
        )
        if not stage_accepted:
            model.load_checkpoint_state(stage_checkpoint)
        else:
            accepted_beta = float(target_beta)
            model.training = stage_training
            accepted_checkpoints.append(model.checkpoint_state())
        objective_before = _mean_objective(objective_before_rows)
        objective_after = _mean_objective(objective_after_rows)
        accepted_trials = [
            trial
            for outer in solve.report["outer_iterations"]
            for trial in outer["trials"]
            if bool(trial["accepted"])
        ]
        backtracking_trials = int(
            sum(len(outer["trials"]) for outer in solve.report["outer_iterations"])
        )
        report = {
            "attempt": int(attempt),
            "beta_before": beta_before,
            "beta_target": float(target_beta),
            "delta_beta": float(delta_beta),
            "subdivision_depth": int(subdivision_depth),
            "lambda_scale": float(stage_training.lambda_scale),
            "optimizer_reset_at_stage": True,
            "discrepancy_scaled_once": True,
            "running_cost_scaled": False,
            "proximal_kl_in_objective": False,
            "transition_kl_proximal": (
                stage_extragradient.transition_kl_rho is not None
            ),
            "transition_kl_rho": stage_extragradient.transition_kl_rho,
            "accepted_outer_updates": accepted_updates,
            "backtracking_trials": backtracking_trials,
            "accepted_backtrack_scales": [
                float(trial["backtrack_scale"]) for trial in accepted_trials
            ],
            "objective_before": objective_before,
            "objective_after": objective_after,
            "objective_confirmation": objective_summary,
            "residual_before": residual_before,
            "residual_after": residual_after,
            "relative_residual_improvement": float(residual_improvement),
            "control_transition": control_trace,
            "gates": {
                "update_or_stationary": update_gate,
                "relative_residual": residual_gate,
                "objective": objective_gate,
                "denominator": denominator_gate,
            },
            "stationary_without_update": stationary_without_update,
            "accepted": stage_accepted,
            "extragradient": solve.report,
        }
        stage_reports.append(report)
        history.append(
            {
                "attempt": float(attempt),
                "beta": float(target_beta),
                "delta_beta": float(delta_beta),
                "accepted": float(stage_accepted),
                "accepted_outer_updates": float(accepted_updates),
                "objective_before": float(
                    objective_before["complete_objective_value"]
                ),
                "objective_after": float(
                    objective_after["complete_objective_value"]
                ),
                "relative_residual_before": before_relative,
                "relative_residual_after": after_relative,
                "alpha_move_rms": float(control_trace["alpha_move_rms"]),
                "log_sigma_move_rms": float(
                    control_trace["log_sigma_move_rms"]
                ),
            }
        )
        if stage_accepted:
            if progress_callback is not None:
                progress_callback(report)
            continue

        can_subdivide = bool(
            subdivision_depth < int(continuation.maximum_subdivisions)
            and 0.5 * delta_beta
            >= float(continuation.minimum_beta_increment) - 1e-15
        )
        if not can_subdivide:
            terminal_reason = "minimum_increment_or_subdivision_limit"
            if progress_callback is not None:
                progress_callback(report)
            break
        midpoint = float(accepted_beta + 0.5 * delta_beta)
        pending.insert(0, (float(target_beta), subdivision_depth + 1))
        pending.insert(0, (midpoint, subdivision_depth + 1))
        if progress_callback is not None:
            progress_callback(report)

    reached_beta_one = bool(abs(accepted_beta - 1.0) <= 1e-12)
    if reached_beta_one:
        model.training = replace(training, lambda_scale=float(training.lambda_scale))
    report = {
        "protocol": {
            "homotopy": "J_beta=running_cost+beta*lambda*discrepancy",
            "exact_beta_zero_reference": True,
            "only_accepted_iterates_carried": True,
            "optimizer_moments_reset_between_stages": True,
            "stage_objective_is_J_beta": True,
            "proximal_kl_in_financial_objective": False,
            "transition_kl_is_conditional_on_current_accepted_law": True,
            "intermediate_output_metrics_evaluated": False,
            "final_objective_unchanged_at_beta_one": True,
            "neural_loss_is_adjoint_mse_only": True,
        },
        "config": dict(continuation.__dict__),
        "extragradient_config": dict(extragradient.__dict__),
        "final_lambda_scale": float(training.lambda_scale),
        "initialization": initialization,
        "beta_zero_objective": beta_zero_objective,
        "stages": stage_reports,
        "accepted_beta": float(accepted_beta),
        "reached_beta_one": reached_beta_one,
        "terminal_reason": terminal_reason,
    }
    return DiscrepancyContinuationResult(
        reached_beta_one=reached_beta_one,
        accepted_beta=float(accepted_beta),
        report=report,
        history=history,
        artifacts={
            "initial_reference_checkpoint": initial_checkpoint,
            "accepted_stage_checkpoints": accepted_checkpoints,
            "final_checkpoint": model.checkpoint_state(),
        },
    )


__all__ = [
    "DiscrepancyContinuationConfig",
    "DiscrepancyContinuationResult",
    "fit_discrepancy_continuation",
    "initialize_zero_adjoint_policy",
]
