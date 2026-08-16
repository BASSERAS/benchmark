from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import operator
from typing import Callable, Mapping

import torch

from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.controls import (
    DynamicsStepInputs,
    HamiltonianControlInputs,
    SpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.model import DiscreteMPModel, RolloutResult
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals
from path_dt_experiments.validated_nested_solver import (
    _control_move_metrics,
    _finite_parameters,
    _load_network_values,
    _network_values,
    _objective_bank_summary,
    _prediction_metrics,
    _project_train_to_eval,
    _sample_target_batch,
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
class SafeguardedExtragradientConfig:
    outer_steps: int = 4
    operator_fit_paths: int = 512
    operator_validation_paths: int = 256
    operator_target_batch_size: int = 1_024
    inner_batch_size: int = 256
    inner_max_steps: int = 400
    inner_eval_every: int = 25
    inner_patience: int = 6
    minimum_inner_relative_improvement: float = 1e-3
    alpha_radius: float = 0.1
    log_sigma_radius: float = 0.005
    r_reliability_floor: float = 0.02
    backtrack_scales: tuple[float, ...] = (1.0, 0.5, 0.25)
    residual_fit_paths: int = 512
    residual_validation_paths: int = 256
    objective_paths: int = 256
    objective_target_batch_size: int = 1_024
    objective_banks: int = 4
    minimum_residual_relative_improvement: float = 1e-3
    maximum_objective_relative_increase: float = 5e-3
    minimum_objective_nonworsening_fraction: float = 0.5
    minimum_denominator_margin: float = 1e-4
    transition_kl_rho: float | None = None
    seed: int = 731_009

    def __post_init__(self) -> None:
        for name in (
            "outer_steps",
            "operator_fit_paths",
            "operator_validation_paths",
            "operator_target_batch_size",
            "inner_batch_size",
            "inner_max_steps",
            "inner_eval_every",
            "inner_patience",
            "residual_fit_paths",
            "residual_validation_paths",
            "objective_paths",
            "objective_target_batch_size",
            "objective_banks",
        ):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        if int(self.inner_batch_size) > int(self.operator_fit_paths):
            raise ValueError("inner_batch_size cannot exceed operator_fit_paths")
        if int(self.inner_eval_every) > int(self.inner_max_steps):
            raise ValueError("inner_eval_every cannot exceed inner_max_steps")
        for name in (
            "minimum_inner_relative_improvement",
            "alpha_radius",
            "log_sigma_radius",
            "minimum_residual_relative_improvement",
            "minimum_denominator_margin",
        ):
            value = _finite_float(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in (
            "r_reliability_floor",
            "minimum_objective_nonworsening_fraction",
        ):
            value = _finite_float(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        objective_tolerance = _finite_float(
            "maximum_objective_relative_increase",
            self.maximum_objective_relative_increase,
        )
        if objective_tolerance < 0.0:
            raise ValueError("maximum_objective_relative_increase must be nonnegative")
        object.__setattr__(
            self,
            "maximum_objective_relative_increase",
            objective_tolerance,
        )
        if self.transition_kl_rho is not None:
            rho = _finite_float("transition_kl_rho", self.transition_kl_rho)
            if rho <= 0.0:
                raise ValueError("transition_kl_rho must be positive")
            object.__setattr__(self, "transition_kl_rho", rho)
        scales = tuple(float(value) for value in self.backtrack_scales)
        if (
            not scales
            or len(scales) != len(set(scales))
            or any(not math.isfinite(value) or value <= 0.0 for value in scales)
            or tuple(sorted(scales, reverse=True)) != scales
            or scales[0] != 1.0
        ):
            raise ValueError(
                "backtrack_scales must be unique, decreasing, positive, and start at one"
            )
        object.__setattr__(self, "backtrack_scales", scales)
        if isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        try:
            seed = int(operator.index(self.seed))
        except TypeError as exc:
            raise ValueError("seed must be an integer") from exc
        object.__setattr__(self, "seed", seed)


@dataclass(frozen=True)
class SafeguardedExtragradientResult:
    report: dict[str, object]
    history: list[dict[str, float]]
    artifacts: dict[str, object]


@dataclass(frozen=True)
class _LawBank:
    paths: torch.Tensor
    controls: torch.Tensor
    noise: torch.Tensor
    moments_p: torch.Tensor
    moments_r: torch.Tensor


@dataclass(frozen=True)
class _OperatorProblem:
    fit: _LawBank
    validation: _LawBank
    fit_teacher_p: torch.Tensor
    fit_teacher_r: torch.Tensor
    validation_teacher_p: torch.Tensor
    validation_teacher_r: torch.Tensor
    r_reliability: torch.Tensor
    metadata: dict[str, object]


def _sample_inputs(
    *,
    model: DiscreteMPModel,
    initial_state_pool: torch.Tensor,
    count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = model._sample_indices(
        size=int(initial_state_pool.shape[0]),
        count=int(count),
        generator=generator,
    )
    x0 = initial_state_pool.index_select(
        0, indices.to(initial_state_pool.device)
    )[:, 0, :]
    noise = sample_standard_normals(
        grid=model.grid,
        batch_size=int(count),
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=derive_stream_seed(base_seed=int(seed), stream_offset=1),
    )
    return x0, noise


def _standard_law_bank(
    *,
    model: DiscreteMPModel,
    initial_state_pool: torch.Tensor,
    count: int,
    seed: int,
) -> _LawBank:
    x0, noise = _sample_inputs(
        model=model,
        initial_state_pool=initial_state_pool,
        count=count,
        seed=seed,
    )
    with torch.no_grad():
        rollout = model.rollout(x0=x0, noise=noise)
    return _LawBank(
        paths=rollout.paths.detach(),
        controls=rollout.controls.detach(),
        noise=noise.detach(),
        moments_p=rollout.expected_adjoint_next.detach(),
        moments_r=rollout.expected_adjoint_noise_next.detach(),
    )


def _network_copy(
    model: DiscreteMPModel,
    values: Mapping[str, torch.Tensor],
) -> torch.nn.Module:
    network = copy.deepcopy(model.network)
    with torch.no_grad():
        for name, parameter in network.named_parameters():
            parameter.copy_(
                values[name].to(device=parameter.device, dtype=parameter.dtype)
            )
    for module in network.modules():
        if isinstance(module, torch.nn.RNNBase):
            module.flatten_parameters()
    network.eval()
    return network


def _bounded_impact_rates(
    *,
    current_controls: torch.Tensor,
    target_controls: torch.Tensor,
    state_dim: int,
    alpha_radius: float,
    log_sigma_radius: float,
    r_reliability: torch.Tensor,
) -> tuple[float, torch.Tensor, dict[str, float]]:
    if tuple(current_controls.shape) != tuple(target_controls.shape):
        raise ValueError("impact controls must have matching shapes")
    if int(current_controls.shape[-1]) != 2 * int(state_dim):
        raise ValueError("impact controls must pack alpha and sigma")
    if tuple(r_reliability.shape) != (int(current_controls.shape[1]),):
        raise ValueError("r_reliability must contain one value per time step")
    current_alpha = current_controls[..., :state_dim]
    target_alpha = target_controls[..., :state_dim]
    current_sigma = current_controls[..., state_dim:]
    target_sigma = target_controls[..., state_dim:]
    if bool(torch.any(current_sigma <= 0.0).item()) or bool(
        torch.any(target_sigma <= 0.0).item()
    ):
        raise ValueError("impact volatility must be positive")
    alpha_direction = target_alpha - current_alpha
    log_sigma_direction = torch.log(target_sigma) - torch.log(current_sigma)
    reliability = r_reliability.to(log_sigma_direction).view(1, -1, 1)
    weighted_log_sigma_direction = reliability * log_sigma_direction
    alpha_rms = torch.sqrt(torch.mean(alpha_direction.double().pow(2)))
    sigma_rms = torch.sqrt(
        torch.mean(weighted_log_sigma_direction.double().pow(2))
    )
    eps = torch.finfo(torch.float64).eps
    alpha_rate = min(1.0, float(alpha_radius) / max(float(alpha_rms), eps))
    sigma_scale = min(1.0, float(log_sigma_radius) / max(float(sigma_rms), eps))
    sigma_rates = (sigma_scale * r_reliability).clamp(0.0, 1.0)
    return alpha_rate, sigma_rates, {
        "unscaled_alpha_move_rms": float(alpha_rms.item()),
        "unscaled_reliability_weighted_log_sigma_move_rms": float(
            sigma_rms.item()
        ),
        "alpha_rate": alpha_rate,
        "log_sigma_global_scale": sigma_scale,
        "r_reliability_mean": float(r_reliability.mean().item()),
        "r_reliability_min": float(r_reliability.min().item()),
        "r_reliability_max": float(r_reliability.max().item()),
    }


def _interpolated_law_bank(
    *,
    model: DiscreteMPModel,
    initial_state_pool: torch.Tensor,
    count: int,
    seed: int,
    base_values: Mapping[str, torch.Tensor],
    target_values: Mapping[str, torch.Tensor],
    alpha_rate: float,
    sigma_rates: torch.Tensor,
    transition_kl_rho: float | None = None,
) -> _LawBank:
    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise ValueError("control interpolation requires specific entropy")
    x0, noise = _sample_inputs(
        model=model,
        initial_state_pool=initial_state_pool,
        count=count,
        seed=seed,
    )
    base_network = _network_copy(model, base_values)
    target_network = _network_copy(model, target_values)
    state_dim = int(model.architecture.state_dim)
    if tuple(sigma_rates.shape) != (int(model.grid.num_steps),):
        raise ValueError("sigma_rates must contain one value per time step")
    x_points = [x0]
    controls = []
    base_hidden = None
    target_hidden = None
    with torch.no_grad():
        for step in range(int(model.grid.num_steps)):
            x_prefix = torch.stack(x_points, dim=1)
            base_p_norm, base_r_norm, base_hidden = base_network.forward_step(
                x_t=x_points[-1], hidden=base_hidden
            )
            target_p_norm, target_r_norm, target_hidden = (
                target_network.forward_step(
                    x_t=x_points[-1], hidden=target_hidden
                )
            )
            base_p = model._denormalize_adjoint_step(
                base_p_norm, step_index=step, noise_adjoint=False
            )
            base_r = model._denormalize_adjoint_step(
                base_r_norm, step_index=step, noise_adjoint=True
            )
            target_p = model._denormalize_adjoint_step(
                target_p_norm, step_index=step, noise_adjoint=False
            )
            target_r = model._denormalize_adjoint_step(
                target_r_norm, step_index=step, noise_adjoint=True
            )
            base_control = model.control_map(
                HamiltonianControlInputs(
                    step_index=step,
                    x_prefix=x_prefix,
                    expected_adjoint_next=base_p,
                    expected_adjoint_noise_next=base_r,
                    parameters=model.parameters,
                )
            )
            target_inputs = HamiltonianControlInputs(
                step_index=step,
                x_prefix=x_prefix,
                expected_adjoint_next=target_p,
                expected_adjoint_noise_next=target_r,
                parameters=model.parameters,
            )
            target_control = (
                model.control_map(target_inputs)
                if transition_kl_rho is None
                else model.control_map.proximal_transition_kl_control(
                    inputs=target_inputs,
                    current_control=base_control,
                    rho=float(transition_kl_rho),
                )
            )
            base_alpha = base_control[..., :state_dim]
            base_sigma = base_control[..., state_dim:]
            target_alpha = target_control[..., :state_dim]
            target_sigma = target_control[..., state_dim:]
            alpha = base_alpha + float(alpha_rate) * (
                target_alpha - base_alpha
            )
            sigma_rate = sigma_rates[step].to(base_sigma)
            log_sigma = torch.log(base_sigma) + sigma_rate * (
                torch.log(target_sigma) - torch.log(base_sigma)
            )
            control = torch.cat((alpha, torch.exp(log_sigma)), dim=-1)
            x_next = model.dynamics_step(
                DynamicsStepInputs(
                    step_index=step,
                    x_prefix=x_prefix,
                    control=control,
                    noise_next=noise[:, step, :],
                    parameters=model.parameters,
                )
            )
            if x_next.ndim == 1:
                x_next = x_next.unsqueeze(-1)
            x_points.append(x_next)
            controls.append(control)
    paths = torch.stack(x_points, dim=1)
    packed_controls = torch.stack(controls, dim=1)
    p, r = model.control_map.moments_from_controls(
        paths=paths,
        controls=packed_controls,
    )
    return _LawBank(
        paths=paths.detach(),
        controls=packed_controls.detach(),
        noise=noise.detach(),
        moments_p=p.detach(),
        moments_r=r.detach(),
    )


def _datewise_reliability(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    floor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if tuple(left.shape) != tuple(right.shape) or left.ndim != 3:
        raise ValueError("reliability inputs must match with shape (B, N, d)")
    left = left.detach().double()
    right = right.detach().double()
    left_centered = left - left.mean(dim=0, keepdim=True)
    right_centered = right - right.mean(dim=0, keepdim=True)
    signal = torch.mean(left_centered * right_centered, dim=(0, 2)).clamp_min(0.0)
    mc_variance = 0.5 * torch.mean((left - right).pow(2), dim=(0, 2))
    eps = torch.finfo(torch.float64).eps
    reliability = signal / (signal + mc_variance).clamp_min(eps)
    reliability = reliability.clamp(min=float(floor), max=1.0)
    return reliability.to(device=left.device, dtype=torch.float32), {
        "mean": float(reliability.mean().item()),
        "min": float(reliability.min().item()),
        "max": float(reliability.max().item()),
        "floor": float(floor),
        "signal_variance_mean": float(signal.mean().item()),
        "mc_variance_mean": float(mc_variance.mean().item()),
    }


def _build_operator_problem(
    *,
    model: DiscreteMPModel,
    fit_pool: torch.Tensor,
    validation_pool: torch.Tensor,
    fit_count: int,
    validation_count: int,
    target_batch_size: int,
    seed: int,
    training: DiscreteMPTrainingConfig,
    r_reliability_floor: float,
    law_factory: Callable[[torch.Tensor, int, int], _LawBank],
) -> _OperatorProblem:
    fit = law_factory(fit_pool, int(fit_count), int(seed) + 1)
    validation = law_factory(
        validation_pool, int(validation_count), int(seed) + 2
    )
    fit_target = _sample_target_batch(
        model=model,
        pool=fit_pool,
        count=int(target_batch_size),
        seed=int(seed) + 3,
    )
    validation_target = _sample_target_batch(
        model=model,
        pool=validation_pool,
        count=int(target_batch_size),
        seed=int(seed) + 4,
    )
    fit_pathwise_p, fit_target_metadata = (
        model._path_functional_adjoint_targets(
            generated_path=fit.paths,
            controls=fit.controls,
            target_path=fit_target,
            training=training,
        )
    )
    validation_pathwise_p, validation_target_metadata = (
        model._path_functional_adjoint_targets(
            generated_path=validation.paths,
            controls=validation.controls,
            target_path=validation_target,
            training=training,
        )
    )
    fit_pathwise_r = model._adjoint_noise_target(
        pathwise_adjoint_target=fit_pathwise_p,
        moments_noise_target_dim=int(fit.moments_r.shape[-1]),
        noise=fit.noise,
        control_variate=fit.moments_p,
    )
    validation_pathwise_r = model._adjoint_noise_target(
        pathwise_adjoint_target=validation_pathwise_p,
        moments_noise_target_dim=int(validation.moments_r.shape[-1]),
        noise=validation.noise,
        control_variate=validation.moments_p,
    )
    with torch.no_grad():
        fit_teacher_p, fit_ce_metadata = model._project_pathwise_target(
            frozen_generated_path=fit.paths,
            pathwise_target=fit_pathwise_p,
            training=training,
        )
        fit_teacher_r, fit_r_ce_metadata = model._project_pathwise_target(
            frozen_generated_path=fit.paths,
            pathwise_target=fit_pathwise_r,
            training=training,
            noise_adjoint=True,
        )
        validation_teacher_p, validation_ce_metadata = (
            model._project_pathwise_target(
                frozen_generated_path=validation.paths,
                pathwise_target=validation_pathwise_p,
                training=training,
            )
        )
        validation_teacher_r, validation_r_ce_metadata = (
            model._project_pathwise_target(
                frozen_generated_path=validation.paths,
                pathwise_target=validation_pathwise_r,
                training=training,
                noise_adjoint=True,
            )
        )

    half = int(fit.paths.shape[0]) // 2
    if half < 2:
        raise ValueError("operator_fit_paths must provide two nontrivial halves")
    left_r = _project_train_to_eval(
        model=model,
        training_paths=fit.paths[:half],
        training_targets=fit_pathwise_r[:half],
        evaluation_paths=validation.paths,
        training=training,
    )
    right_r = _project_train_to_eval(
        model=model,
        training_paths=fit.paths[half:],
        training_targets=fit_pathwise_r[half:],
        evaluation_paths=validation.paths,
        training=training,
    )
    reliability, reliability_metadata = _datewise_reliability(
        left_r,
        right_r,
        floor=float(r_reliability_floor),
    )
    return _OperatorProblem(
        fit=fit,
        validation=validation,
        fit_teacher_p=fit_teacher_p.detach(),
        fit_teacher_r=fit_teacher_r.detach(),
        validation_teacher_p=validation_teacher_p.detach(),
        validation_teacher_r=validation_teacher_r.detach(),
        r_reliability=reliability.to(
            device=model.device, dtype=model.dtype
        ),
        metadata={
            "fit_target": fit_target_metadata,
            "validation_target": validation_target_metadata,
            "fit_ce": fit_ce_metadata,
            "fit_r_ce": fit_r_ce_metadata,
            "validation_ce": validation_ce_metadata,
            "validation_r_ce": validation_r_ce_metadata,
            "r_reliability": reliability_metadata,
        },
    )


def _fit_operator_network(
    *,
    model: DiscreteMPModel,
    problem: _OperatorProblem,
    initial_values: Mapping[str, torch.Tensor],
    training: DiscreteMPTrainingConfig,
    config: SafeguardedExtragradientConfig,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    _load_network_values(model, dict(initial_values))
    normalized_fit_p = model._normalize_adjoint_target(
        problem.fit_teacher_p, noise_adjoint=False
    ).detach()
    normalized_fit_r = model._normalize_adjoint_target(
        problem.fit_teacher_r, noise_adjoint=True
    ).detach()
    normalized_validation_p = model._normalize_adjoint_target(
        problem.validation_teacher_p, noise_adjoint=False
    ).detach()
    normalized_validation_r = model._normalize_adjoint_target(
        problem.validation_teacher_r, noise_adjoint=True
    ).detach()

    def losses(
        paths: torch.Tensor,
        target_p: torch.Tensor,
        target_r: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if indices is not None:
            paths = paths.index_select(0, indices)
            target_p = target_p.index_select(0, indices)
            target_r = target_r.index_select(0, indices)
        moments = model.network(paths)
        p_loss = torch.mean(
            (moments.expected_adjoint_next - target_p).pow(2)
        )
        r_loss = torch.mean(
            (moments.expected_adjoint_noise_next - target_r).pow(2)
        )
        total = (
            float(training.adjoint_weight) * p_loss
            + float(training.adjoint_noise_weight) * r_loss
        )
        return total, p_loss, r_loss

    def evaluate(step: int) -> dict[str, object]:
        with torch.no_grad():
            train_total, train_p, train_r = losses(
                problem.fit.paths,
                normalized_fit_p,
                normalized_fit_r,
            )
            validation_total, validation_p, validation_r = losses(
                problem.validation.paths,
                normalized_validation_p,
                normalized_validation_r,
            )
            validation_prediction = model._denormalize_moments(
                model.network(problem.validation.paths)
            )
        return {
            "step": int(step),
            "training_loss": {
                "total": float(train_total.item()),
                "p": float(train_p.item()),
                "r": float(train_r.item()),
            },
            "validation_loss": {
                "total": float(validation_total.item()),
                "p": float(validation_p.item()),
                "r": float(validation_r.item()),
            },
            "validation_target_fit": {
                "p": _prediction_metrics(
                    validation_prediction.expected_adjoint_next,
                    problem.validation_teacher_p,
                    training_target=problem.fit_teacher_p,
                ),
                "r": _prediction_metrics(
                    validation_prediction.expected_adjoint_noise_next,
                    problem.validation_teacher_r,
                    training_target=problem.fit_teacher_r,
                ),
            },
        }

    optimizer = torch.optim.AdamW(
        model.network.parameters(),
        lr=float(training.lr),
        weight_decay=float(training.weight_decay),
    )
    initial = evaluate(0)
    trajectory = [initial]
    best_values = _network_values(model)
    best_step = 0
    best_loss = float(initial["validation_loss"]["total"])
    meaningful_best = best_loss
    patience = 0
    stopped_early = False
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    gradient_norms: list[float] = []
    parameters = list(model.network.parameters())
    for step in range(1, int(config.inner_max_steps) + 1):
        indices = model._sample_indices(
            size=int(problem.fit.paths.shape[0]),
            count=int(config.inner_batch_size),
            generator=generator,
        ).to(model.device)
        loss, _, _ = losses(
            problem.fit.paths,
            normalized_fit_p,
            normalized_fit_r,
            indices,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = model._grad_norm(parameters)
        if not math.isfinite(gradient_norm):
            raise RuntimeError("extragradient operator-fit gradient is nonfinite")
        gradient_norms.append(float(gradient_norm))
        optimizer.step()
        if not _finite_parameters(model):
            raise RuntimeError("extragradient operator-fit parameter is nonfinite")
        if (
            step % int(config.inner_eval_every) == 0
            or step == int(config.inner_max_steps)
        ):
            row = evaluate(step)
            trajectory.append(row)
            current = float(row["validation_loss"]["total"])
            if current < best_loss:
                best_loss = current
                best_step = step
                best_values = _network_values(model)
            if current < meaningful_best * (
                1.0 - float(config.minimum_inner_relative_improvement)
            ):
                meaningful_best = current
                patience = 0
            else:
                patience += 1
            if patience >= int(config.inner_patience):
                stopped_early = True
                break
    _load_network_values(model, best_values)
    selected = evaluate(best_step)
    norms = torch.tensor(gradient_norms, dtype=torch.float64)
    return best_values, {
        "best_step": int(best_step),
        "terminal_step": int(trajectory[-1]["step"]),
        "stopped_early": stopped_early,
        "initial": initial,
        "selected": selected,
        "trajectory": trajectory,
        "gradient_norm_mean": float(norms.mean().item()),
        "gradient_norm_max": float(norms.max().item()),
    }


def _moments_for_values(
    *,
    model: DiscreteMPModel,
    values: Mapping[str, torch.Tensor],
    paths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    network = _network_copy(model, values)
    with torch.no_grad():
        normalized = network(paths)
        moments = model._denormalize_moments(normalized)
    return (
        moments.expected_adjoint_next.detach(),
        moments.expected_adjoint_noise_next.detach(),
    )


def _controls_for_values(
    *,
    model: DiscreteMPModel,
    values: Mapping[str, torch.Tensor],
    paths: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise ValueError("extragradient requires specific entropy")
    p, r = _moments_for_values(model=model, values=values, paths=paths)
    with torch.no_grad():
        return model.control_map.controls_from_moments(
            paths=paths,
            expected_adjoint_next=p,
            expected_adjoint_noise_next=r,
        ).detach()


def _proposal_controls_for_values(
    *,
    model: DiscreteMPModel,
    values: Mapping[str, torch.Tensor],
    paths: torch.Tensor,
    current_controls: torch.Tensor,
    transition_kl_rho: float | None,
) -> torch.Tensor:
    """Hamiltonian proposal, optionally proximal to the current transition."""

    if transition_kl_rho is None:
        return _controls_for_values(model=model, values=values, paths=paths)
    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise ValueError("transition-KL proposal requires specific entropy")
    p, r = _moments_for_values(model=model, values=values, paths=paths)
    with torch.no_grad():
        return model.control_map.proximal_transition_kl_controls(
            paths=paths,
            current_controls=current_controls,
            expected_adjoint_next=p,
            expected_adjoint_noise_next=r,
            rho=float(transition_kl_rho),
        ).detach()


def _corrector_targets(
    *,
    model: DiscreteMPModel,
    problem: _OperatorProblem,
    base_values: Mapping[str, torch.Tensor],
    target_values: Mapping[str, torch.Tensor],
    alpha_radius: float,
    log_sigma_radius: float,
    transition_kl_rho: float | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
]:
    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise ValueError("corrector targets require specific entropy")
    base_validation = _controls_for_values(
        model=model,
        values=base_values,
        paths=problem.validation.paths,
    )
    target_validation = _proposal_controls_for_values(
        model=model,
        values=target_values,
        paths=problem.validation.paths,
        current_controls=base_validation,
        transition_kl_rho=transition_kl_rho,
    )
    impact_origin = (
        base_validation
        if transition_kl_rho is not None
        else problem.validation.controls
    )
    alpha_rate, sigma_rates, impact = _bounded_impact_rates(
        current_controls=impact_origin,
        target_controls=target_validation,
        state_dim=int(model.architecture.state_dim),
        alpha_radius=float(alpha_radius),
        log_sigma_radius=float(log_sigma_radius),
        r_reliability=problem.r_reliability,
    )

    def targets_for(bank: _LawBank) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_controls = _controls_for_values(
            model=model,
            values=base_values,
            paths=bank.paths,
        )
        target_controls = _proposal_controls_for_values(
            model=model,
            values=target_values,
            paths=bank.paths,
            current_controls=base_controls,
            transition_kl_rho=transition_kl_rho,
        )
        state_dim = int(model.architecture.state_dim)
        base_alpha = base_controls[..., :state_dim]
        base_sigma = base_controls[..., state_dim:]
        target_alpha = target_controls[..., :state_dim]
        target_sigma = target_controls[..., state_dim:]
        direction_alpha = (
            target_alpha - base_alpha
            if transition_kl_rho is not None
            else target_alpha - bank.controls[..., :state_dim]
        )
        alpha = base_alpha + float(alpha_rate) * direction_alpha
        rate = sigma_rates.to(base_sigma).view(1, -1, 1)
        direction_log_sigma = (
            torch.log(target_sigma) - torch.log(base_sigma)
            if transition_kl_rho is not None
            else torch.log(target_sigma)
            - torch.log(bank.controls[..., state_dim:])
        )
        log_sigma = torch.log(base_sigma) + rate * (
            direction_log_sigma
        )
        controls = torch.cat((alpha, torch.exp(log_sigma)), dim=-1)
        p, r = model.control_map.moments_from_controls(
            paths=bank.paths,
            controls=controls,
        )
        return p.detach(), r.detach(), controls.detach()

    fit_p, fit_r, fit_controls = targets_for(problem.fit)
    validation_p, validation_r, validation_controls = targets_for(
        problem.validation
    )
    base_fit_controls = _controls_for_values(
        model=model,
        values=base_values,
        paths=problem.fit.paths,
    )
    target_fit_controls = _proposal_controls_for_values(
        model=model,
        values=target_values,
        paths=problem.fit.paths,
        current_controls=base_fit_controls,
        transition_kl_rho=transition_kl_rho,
    )
    denominator = float(model.control_map.eta) / validation_controls[
        ..., int(model.architecture.state_dim) :
    ]
    transition_kl = model.control_map.transition_kernel_kl(
        controls=validation_controls,
        current_controls=base_validation,
    ).detach().double()
    return fit_p, fit_r, validation_p, validation_r, {
        "impact": impact,
        "fit_control_move": _control_move_metrics(
            current=base_fit_controls,
            target=target_fit_controls,
            relaxed=fit_controls,
            state_dim=int(model.architecture.state_dim),
        ),
        "validation_control_move": _control_move_metrics(
            current=base_validation,
            target=target_validation,
            relaxed=validation_controls,
            state_dim=int(model.architecture.state_dim),
        ),
        "minimum_specific_entropy_denominator": float(
            denominator.min().item()
        ),
        "transition_kl_rho": transition_kl_rho,
        "conditional_transition_kl_mean": float(transition_kl.mean().item()),
        "conditional_transition_kl_q95": float(
            torch.quantile(transition_kl.reshape(-1), 0.95).item()
        ),
        "conditional_transition_kl_max": float(transition_kl.max().item()),
    }


def _problem_with_targets(
    problem: _OperatorProblem,
    *,
    fit_p: torch.Tensor,
    fit_r: torch.Tensor,
    validation_p: torch.Tensor,
    validation_r: torch.Tensor,
) -> _OperatorProblem:
    return _OperatorProblem(
        fit=problem.fit,
        validation=problem.validation,
        fit_teacher_p=fit_p,
        fit_teacher_r=fit_r,
        validation_teacher_p=validation_p,
        validation_teacher_r=validation_r,
        r_reliability=problem.r_reliability,
        metadata=problem.metadata,
    )


def _stationarity_residual(
    *,
    model: DiscreteMPModel,
    problem: _OperatorProblem,
    values: Mapping[str, torch.Tensor],
    alpha_radius: float,
    log_sigma_radius: float,
) -> dict[str, object]:
    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise ValueError("stationarity residual requires specific entropy")
    prediction_p, prediction_r = _moments_for_values(
        model=model,
        values=values,
        paths=problem.validation.paths,
    )
    current_controls = model.control_map.controls_from_moments(
        paths=problem.validation.paths,
        expected_adjoint_next=prediction_p,
        expected_adjoint_noise_next=prediction_r,
    )
    target_controls = model.control_map.controls_from_moments(
        paths=problem.validation.paths,
        expected_adjoint_next=problem.validation_teacher_p,
        expected_adjoint_noise_next=problem.validation_teacher_r,
    )
    state_dim = int(model.architecture.state_dim)
    alpha_gap = target_controls[..., :state_dim] - current_controls[..., :state_dim]
    log_sigma_gap = torch.log(target_controls[..., state_dim:]) - torch.log(
        current_controls[..., state_dim:]
    )
    reliability = problem.r_reliability.to(log_sigma_gap).view(1, -1, 1)
    alpha_rms = torch.sqrt(torch.mean(alpha_gap.double().pow(2)))
    log_sigma_rms = torch.sqrt(
        torch.mean((reliability * log_sigma_gap).double().pow(2))
    )
    normalized = torch.sqrt(
        (alpha_rms / float(alpha_radius)).pow(2)
        + (log_sigma_rms / float(log_sigma_radius)).pow(2)
    )
    prediction_p64 = prediction_p.detach().double()
    target_p64 = problem.validation_teacher_p.detach().double()
    prediction_r64 = prediction_r.detach().double()
    target_r64 = problem.validation_teacher_r.detach().double()
    reliability64 = reliability.detach().double()
    p_error_rms = torch.sqrt(torch.mean((prediction_p64 - target_p64).pow(2)))
    p_target_rms = torch.sqrt(torch.mean(target_p64.pow(2)))
    r_error_rms = torch.sqrt(
        torch.mean(
            (reliability64 * (prediction_r64 - target_r64)).pow(2)
        )
    )
    r_target_rms = torch.sqrt(
        torch.mean((reliability64 * target_r64).pow(2))
    )
    moment_error = torch.sqrt(p_error_rms.pow(2) + r_error_rms.pow(2))
    moment_target = torch.sqrt(p_target_rms.pow(2) + r_target_rms.pow(2))
    relative_eps = torch.finfo(torch.float64).eps
    relative_moment_residual = moment_error / (moment_target + relative_eps)
    return {
        "normalized_control_impact_norm": float(normalized.item()),
        "relative_moment_residual": float(relative_moment_residual.item()),
        "moment_error_rms": float(moment_error.item()),
        "moment_target_rms": float(moment_target.item()),
        "p_error_rms": float(p_error_rms.item()),
        "p_target_rms": float(p_target_rms.item()),
        "r_reliability_weighted_error_rms": float(r_error_rms.item()),
        "r_reliability_weighted_target_rms": float(r_target_rms.item()),
        "alpha_residual_rms": float(alpha_rms.item()),
        "reliability_weighted_log_sigma_residual_rms": float(
            log_sigma_rms.item()
        ),
        "p": _prediction_metrics(
            prediction_p,
            problem.validation_teacher_p,
            training_target=problem.fit_teacher_p,
        ),
        "r": _prediction_metrics(
            prediction_r,
            problem.validation_teacher_r,
            training_target=problem.fit_teacher_r,
        ),
        "r_reliability": problem.metadata["r_reliability"],
    }


def _objective_replay_inputs(
    *,
    model: DiscreteMPModel,
    pool: torch.Tensor,
    config: SafeguardedExtragradientConfig,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    banks = []
    for bank_index in range(int(config.objective_banks)):
        bank_seed = int(seed) + 100 * bank_index
        x0, noise = _sample_inputs(
            model=model,
            initial_state_pool=pool,
            count=int(config.objective_paths),
            seed=bank_seed,
        )
        target = _sample_target_batch(
            model=model,
            pool=pool,
            count=int(config.objective_target_batch_size),
            seed=bank_seed + 2,
        )
        banks.append({"x0": x0, "noise": noise, "target": target})
    return banks


def _objective_rows(
    *,
    model: DiscreteMPModel,
    values: Mapping[str, torch.Tensor],
    banks: list[dict[str, torch.Tensor]],
    training: DiscreteMPTrainingConfig,
) -> list[dict[str, float]]:
    _load_network_values(model, dict(values))
    rows = []
    with torch.no_grad():
        for bank in banks:
            rollout = model.rollout(x0=bank["x0"], noise=bank["noise"])
            finite = bool(
                torch.isfinite(rollout.paths).all().item()
                and torch.isfinite(rollout.controls).all().item()
            )
            if not finite:
                rows.append(
                    {
                        "complete_objective_value": float("inf"),
                        "discrepancy_objective_value": float("inf"),
                        "running_cost_value": float("inf"),
                    }
                )
                continue
            _, metrics = model._complete_objective_metrics(
                generated_path=rollout.paths,
                controls=rollout.controls,
                target_path=bank["target"],
                training=training,
            )
            rows.append(
                {
                    "complete_objective_value": float(
                        metrics["complete_objective_value"]
                    ),
                    "discrepancy_objective_value": float(
                        metrics["discrepancy_objective_value"]
                    ),
                    "running_cost_value": float(metrics["running_cost_value"]),
                }
            )
    return rows


def _predictor_corrector_alignment(
    *,
    model: DiscreteMPModel,
    predictor_problem: _OperatorProblem,
    base_values: Mapping[str, torch.Tensor],
    first_target_values: Mapping[str, torch.Tensor],
    second_target_values: Mapping[str, torch.Tensor],
    transition_kl_rho: float | None,
) -> dict[str, float | None]:
    paths = predictor_problem.validation.paths
    state_dim = int(model.architecture.state_dim)
    base = _controls_for_values(model=model, values=base_values, paths=paths)
    first = _proposal_controls_for_values(
        model=model,
        values=first_target_values,
        paths=paths,
        current_controls=base,
        transition_kl_rho=transition_kl_rho,
    )
    second = _proposal_controls_for_values(
        model=model,
        values=second_target_values,
        paths=paths,
        current_controls=base,
        transition_kl_rho=transition_kl_rho,
    )
    predictor = predictor_problem.validation.controls
    first_alpha = first[..., :state_dim] - base[..., :state_dim]
    second_origin = base if transition_kl_rho is not None else predictor
    second_alpha = (
        second[..., :state_dim] - second_origin[..., :state_dim]
    )
    first_log_sigma = torch.log(first[..., state_dim:]) - torch.log(
        base[..., state_dim:]
    )
    second_log_sigma = torch.log(second[..., state_dim:]) - torch.log(
        second_origin[..., state_dim:]
    )
    reliability = predictor_problem.r_reliability.to(first_log_sigma).view(
        1, -1, 1
    )

    def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
        left = left.detach().double().reshape(-1)
        right = right.detach().double().reshape(-1)
        denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(
            right
        )
        if float(denominator.item()) <= torch.finfo(torch.float64).eps:
            return None
        return float((torch.dot(left, right) / denominator).item())

    combined_first = torch.cat(
        (first_alpha.reshape(-1), (reliability * first_log_sigma).reshape(-1))
    )
    combined_second = torch.cat(
        (
            second_alpha.reshape(-1),
            (reliability * second_log_sigma).reshape(-1),
        )
    )
    return {
        "alpha_cosine": cosine(first_alpha, second_alpha),
        "reliability_weighted_log_sigma_cosine": cosine(
            reliability * first_log_sigma,
            reliability * second_log_sigma,
        ),
        "combined_control_impact_cosine": cosine(
            combined_first, combined_second
        ),
    }


def fit_safeguarded_extragradient(
    *,
    model: DiscreteMPModel,
    backward_target_paths: torch.Tensor,
    validation_target_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    config: SafeguardedExtragradientConfig,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> SafeguardedExtragradientResult:
    """Solve the MP moment fixed point with a safeguarded extragradient step."""

    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise ValueError(
            "safeguarded extragradient currently requires "
            "SpecificEntropyDiagonalControl"
        )
    if training.grad_clip_norm is not None:
        raise ValueError("safeguarded extragradient requires uncapped training")
    if str(training.target_preconditioner) != "none":
        raise ValueError("safeguarded extragradient preserves the raw P/R loss")
    if str(training.ce_target_mode) != "ridge":
        raise ValueError("safeguarded extragradient requires ridge projection")
    if str(training.noise_target_control_variate) != "adjoint":
        raise ValueError(
            "safeguarded extragradient requires the unbiased adjoint R control variate"
        )
    model._validate_target_paths(backward_target_paths)
    model._validate_target_paths(validation_target_paths)
    fit_pool = backward_target_paths.to(device=model.device, dtype=model.dtype)
    validation_pool = validation_target_paths.to(
        device=model.device, dtype=model.dtype
    )
    model.training = training
    initial_checkpoint = model.checkpoint_state()
    history: list[dict[str, float]] = []
    outer_reports: list[dict[str, object]] = []
    accepted_states: list[dict[str, torch.Tensor]] = []

    def standard_factory(
        pool: torch.Tensor, count: int, seed: int
    ) -> _LawBank:
        return _standard_law_bank(
            model=model,
            initial_state_pool=pool,
            count=count,
            seed=seed,
        )

    for outer_index in range(int(config.outer_steps)):
        outer_seed = derive_stream_seed(
            base_seed=int(config.seed),
            stream_offset=1_000_000 * outer_index,
        )
        if outer_seed is None:
            raise RuntimeError("extragradient seed derivation failed")
        base_values = _network_values(model)
        _load_network_values(model, base_values)
        current_problem = _build_operator_problem(
            model=model,
            fit_pool=fit_pool,
            validation_pool=validation_pool,
            fit_count=int(config.operator_fit_paths),
            validation_count=int(config.operator_validation_paths),
            target_batch_size=int(config.operator_target_batch_size),
            seed=int(outer_seed) + 10_000,
            training=training,
            r_reliability_floor=float(config.r_reliability_floor),
            law_factory=standard_factory,
        )
        first_target_values, first_fit_report = _fit_operator_network(
            model=model,
            problem=current_problem,
            initial_values=base_values,
            training=training,
            config=config,
            seed=int(outer_seed) + 20_000,
        )

        _load_network_values(model, base_values)
        residual_before_problem = _build_operator_problem(
            model=model,
            fit_pool=fit_pool,
            validation_pool=validation_pool,
            fit_count=int(config.residual_fit_paths),
            validation_count=int(config.residual_validation_paths),
            target_batch_size=int(config.operator_target_batch_size),
            seed=int(outer_seed) + 500_000,
            training=training,
            r_reliability_floor=float(config.r_reliability_floor),
            law_factory=standard_factory,
        )
        residual_before = _stationarity_residual(
            model=model,
            problem=residual_before_problem,
            values=base_values,
            alpha_radius=float(config.alpha_radius),
            log_sigma_radius=float(config.log_sigma_radius),
        )
        objective_banks = _objective_replay_inputs(
            model=model,
            pool=validation_pool,
            config=config,
            seed=int(outer_seed) + 700_000,
        )
        objective_before_rows = _objective_rows(
            model=model,
            values=base_values,
            banks=objective_banks,
            training=training,
        )

        first_target_validation_controls = _proposal_controls_for_values(
            model=model,
            values=first_target_values,
            paths=current_problem.validation.paths,
            current_controls=current_problem.validation.controls,
            transition_kl_rho=config.transition_kl_rho,
        )
        trial_reports: list[dict[str, object]] = []
        accepted = False
        accepted_values = base_values
        accepted_scale = 0.0
        for trial_index, scale in enumerate(config.backtrack_scales):
            alpha_rate, sigma_rates, predictor_impact = _bounded_impact_rates(
                current_controls=current_problem.validation.controls,
                target_controls=first_target_validation_controls,
                state_dim=int(model.architecture.state_dim),
                alpha_radius=float(config.alpha_radius) * float(scale),
                log_sigma_radius=float(config.log_sigma_radius) * float(scale),
                r_reliability=current_problem.r_reliability,
            )

            def predictor_factory(
                pool: torch.Tensor, count: int, seed: int
            ) -> _LawBank:
                return _interpolated_law_bank(
                    model=model,
                    initial_state_pool=pool,
                    count=count,
                    seed=seed,
                    base_values=base_values,
                    target_values=first_target_values,
                    alpha_rate=alpha_rate,
                    sigma_rates=sigma_rates,
                    transition_kl_rho=config.transition_kl_rho,
                )

            predictor_problem = _build_operator_problem(
                model=model,
                fit_pool=fit_pool,
                validation_pool=validation_pool,
                fit_count=int(config.operator_fit_paths),
                validation_count=int(config.operator_validation_paths),
                target_batch_size=int(config.operator_target_batch_size),
                seed=int(outer_seed) + 30_000,
                training=training,
                r_reliability_floor=float(config.r_reliability_floor),
                law_factory=predictor_factory,
            )
            second_target_values, second_fit_report = _fit_operator_network(
                model=model,
                problem=predictor_problem,
                initial_values=first_target_values,
                training=training,
                config=config,
                seed=int(outer_seed) + 40_000,
            )
            (
                corrector_fit_p,
                corrector_fit_r,
                corrector_validation_p,
                corrector_validation_r,
                corrector_impact,
            ) = _corrector_targets(
                model=model,
                problem=predictor_problem,
                base_values=base_values,
                target_values=second_target_values,
                alpha_radius=float(config.alpha_radius) * float(scale),
                log_sigma_radius=float(config.log_sigma_radius) * float(scale),
                transition_kl_rho=config.transition_kl_rho,
            )
            corrector_problem = _problem_with_targets(
                predictor_problem,
                fit_p=corrector_fit_p,
                fit_r=corrector_fit_r,
                validation_p=corrector_validation_p,
                validation_r=corrector_validation_r,
            )
            candidate_values, corrector_fit_report = _fit_operator_network(
                model=model,
                problem=corrector_problem,
                initial_values=base_values,
                training=training,
                config=config,
                seed=int(outer_seed) + 50_000,
            )
            _load_network_values(model, candidate_values)
            residual_after_problem = _build_operator_problem(
                model=model,
                fit_pool=fit_pool,
                validation_pool=validation_pool,
                fit_count=int(config.residual_fit_paths),
                validation_count=int(config.residual_validation_paths),
                target_batch_size=int(config.operator_target_batch_size),
                seed=int(outer_seed) + 500_000,
                training=training,
                r_reliability_floor=float(config.r_reliability_floor),
                law_factory=standard_factory,
            )
            residual_after = _stationarity_residual(
                model=model,
                problem=residual_after_problem,
                values=candidate_values,
                alpha_radius=float(config.alpha_radius),
                log_sigma_radius=float(config.log_sigma_radius),
            )
            before_norm = float(
                residual_before["normalized_control_impact_norm"]
            )
            after_norm = float(
                residual_after["normalized_control_impact_norm"]
            )
            residual_relative_improvement = (
                before_norm - after_norm
            ) / max(abs(before_norm), torch.finfo(torch.float64).eps)
            objective_after_rows = _objective_rows(
                model=model,
                values=candidate_values,
                banks=objective_banks,
                training=training,
            )
            objective_summary = _objective_bank_summary(
                baseline_rows=objective_before_rows,
                candidate_rows=objective_after_rows,
            )
            objective_relative_increase = -float(
                objective_summary["mean_relative_improvement"]
            )
            denominator_ok = bool(
                float(corrector_impact["minimum_specific_entropy_denominator"])
                >= float(config.minimum_denominator_margin)
            )
            residual_ok = bool(
                math.isfinite(after_norm)
                and residual_relative_improvement
                >= float(config.minimum_residual_relative_improvement)
            )
            objective_ok = bool(
                bool(objective_summary["all_finite"])
                and objective_relative_increase
                <= float(config.maximum_objective_relative_increase)
                and float(objective_summary["improvement_fraction"])
                >= float(config.minimum_objective_nonworsening_fraction)
            )
            trial_accepted = bool(
                denominator_ok and residual_ok and objective_ok
            )
            trial_report: dict[str, object] = {
                "trial": int(trial_index + 1),
                "backtrack_scale": float(scale),
                "predictor_impact": predictor_impact,
                "corrector_impact": corrector_impact,
                "predictor_corrector_alignment": (
                    _predictor_corrector_alignment(
                        model=model,
                        predictor_problem=predictor_problem,
                        base_values=base_values,
                        first_target_values=first_target_values,
                        second_target_values=second_target_values,
                        transition_kl_rho=config.transition_kl_rho,
                    )
                ),
                "second_operator_fit": second_fit_report,
                "corrector_fit": corrector_fit_report,
                "residual_before": residual_before,
                "residual_after": residual_after,
                "residual_relative_improvement": float(
                    residual_relative_improvement
                ),
                "objective": objective_summary,
                "objective_relative_increase": float(
                    objective_relative_increase
                ),
                "gate": {
                    "denominator_ok": denominator_ok,
                    "residual_ok": residual_ok,
                    "objective_ok": objective_ok,
                },
                "accepted": trial_accepted,
            }
            trial_reports.append(trial_report)
            if trial_accepted:
                accepted = True
                accepted_values = candidate_values
                accepted_scale = float(scale)
                break

        _load_network_values(
            model, accepted_values if accepted else base_values
        )
        if accepted:
            accepted_states.append(accepted_values)
        outer_report: dict[str, object] = {
            "outer_step": int(outer_index + 1),
            "first_operator_fit": first_fit_report,
            "current_operator_metadata": current_problem.metadata,
            "residual_before": residual_before,
            "trials": trial_reports,
            "accepted": accepted,
            "accepted_backtrack_scale": accepted_scale,
        }
        outer_reports.append(outer_report)
        accepted_trial = next(
            (row for row in trial_reports if bool(row["accepted"])), None
        )
        final_residual = (
            residual_before
            if accepted_trial is None
            else accepted_trial["residual_after"]
        )
        flat = {
            "outer_step": float(outer_index + 1),
            "accepted": float(accepted),
            "accepted_backtrack_scale": float(accepted_scale),
            "residual_before": float(
                residual_before["normalized_control_impact_norm"]
            ),
            "residual_after": float(
                final_residual["normalized_control_impact_norm"]
            ),
            "residual_relative_improvement": float(
                0.0
                if accepted_trial is None
                else accepted_trial["residual_relative_improvement"]
            ),
            "objective_relative_increase": float(
                0.0
                if accepted_trial is None
                else accepted_trial["objective_relative_increase"]
            ),
            "r_reliability_mean": float(
                current_problem.r_reliability.mean().item()
            ),
        }
        history.append(flat)
        if progress_callback is not None:
            progress_callback(outer_report)

    report: dict[str, object] = {
        "protocol": {
            "operator": "F(z)=z-T(z)",
            "predictor_target_evaluated_at_current_law": True,
            "corrector_target_evaluated_at_predictor_law": True,
            "corrector_applied_from_original_iterate": True,
            "control_impact_coordinates": "alpha_and_log_sigma",
            "r_step_reliability_weighted": True,
            "stationarity_residual_primary_safeguard": True,
            "complete_objective_used_only_as_safety_check": True,
            "independent_common_random_number_objective_replay": True,
            "running_cost_and_discrepancy_unchanged": True,
            "neural_loss_is_adjoint_mse_only": True,
            "fixed_points_match_original_mp_stationarity_system": True,
            "transition_kl_proximal": config.transition_kl_rho is not None,
            "transition_kl_rho": config.transition_kl_rho,
            "transition_kl_excluded_from_objective": True,
            "gradient_clipping": False,
            "test_paths_used_for_training_or_selection": False,
        },
        "config": dict(config.__dict__),
        "training": dict(training.__dict__),
        "outer_iterations": outer_reports,
        "accepted_outer_updates": int(sum(row["accepted"] for row in history)),
    }
    return SafeguardedExtragradientResult(
        report=report,
        history=history,
        artifacts={
            "initial_checkpoint": initial_checkpoint,
            "final_checkpoint": model.checkpoint_state(),
            "accepted_network_states": accepted_states,
        },
    )


__all__ = [
    "SafeguardedExtragradientConfig",
    "SafeguardedExtragradientResult",
    "fit_safeguarded_extragradient",
]
