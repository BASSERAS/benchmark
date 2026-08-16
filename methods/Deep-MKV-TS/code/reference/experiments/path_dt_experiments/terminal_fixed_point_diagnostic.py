from __future__ import annotations

from dataclasses import dataclass, replace
import math
import operator
from typing import Any, Callable, Mapping

import torch

from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.controls import (
    DynamicsStepInputs,
    HamiltonianControlInputs,
    SpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from deep_mkv_gen_path_dt.noise import derive_stream_seed
from path_dt_experiments.extragradient_solver import (
    SafeguardedExtragradientConfig,
    _LawBank,
    _build_operator_problem,
    _fit_operator_network,
    _load_network_values,
    _moments_for_values,
    _network_copy,
    _network_values,
    _sample_inputs,
    _standard_law_bank,
)
from path_dt_experiments.validated_nested_solver import (
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


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive and finite")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True)
class TerminalFixedPointDiagnosticConfig:
    residual_replicates: int = 12
    common_evaluation_paths: int = 1_024
    residual_fit_paths: int = 1_024
    residual_target_batch_size: int = 2_048
    projection_fit_paths: int = 2_048
    projection_validation_paths: int = 1_024
    projection_target_batch_size: int = 2_048
    projection_restarts: int = 3
    projection_inner_batch_size: int = 256
    projection_max_steps: int = 2_000
    projection_eval_every: int = 50
    projection_patience: int = 12
    time_blocks: int = 3
    jacobian_replicates: int = 3
    jacobian_fit_paths: int = 512
    jacobian_validation_paths: int = 512
    jacobian_target_batch_size: int = 1_024
    finite_difference_step: float = 0.05
    secondary_finite_difference_step: float = 0.025
    r_reliability_floor: float = 0.02
    seed: int = 2_608_031

    def __post_init__(self) -> None:
        for name in (
            "residual_replicates",
            "common_evaluation_paths",
            "residual_fit_paths",
            "residual_target_batch_size",
            "projection_fit_paths",
            "projection_validation_paths",
            "projection_target_batch_size",
            "projection_restarts",
            "projection_inner_batch_size",
            "projection_max_steps",
            "projection_eval_every",
            "projection_patience",
            "time_blocks",
            "jacobian_replicates",
            "jacobian_fit_paths",
            "jacobian_validation_paths",
            "jacobian_target_batch_size",
        ):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        if int(self.residual_replicates) < 2:
            raise ValueError("residual_replicates must be at least two")
        if int(self.jacobian_replicates) < 2:
            raise ValueError("jacobian_replicates must be at least two")
        if int(self.projection_inner_batch_size) > int(self.projection_fit_paths):
            raise ValueError("projection_inner_batch_size cannot exceed projection_fit_paths")
        if int(self.projection_eval_every) > int(self.projection_max_steps):
            raise ValueError("projection_eval_every cannot exceed projection_max_steps")
        for name in (
            "finite_difference_step",
            "secondary_finite_difference_step",
        ):
            object.__setattr__(self, name, _positive_float(name, getattr(self, name)))
        if float(self.secondary_finite_difference_step) >= float(
            self.finite_difference_step
        ):
            raise ValueError("secondary_finite_difference_step must be smaller")
        floor = float(self.r_reliability_floor)
        if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError("r_reliability_floor must lie in [0, 1]")
        object.__setattr__(self, "r_reliability_floor", floor)
        if isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        object.__setattr__(self, "seed", int(operator.index(self.seed)))


@dataclass(frozen=True)
class TerminalFixedPointDiagnosticResult:
    report: dict[str, object]
    artifacts: dict[str, Any]


def _time_blocks(num_steps: int, count: int) -> tuple[tuple[int, int], ...]:
    if count > num_steps:
        raise ValueError("time_blocks cannot exceed the number of steps")
    edges = [int(round(index * num_steps / count)) for index in range(count + 1)]
    edges[0] = 0
    edges[-1] = int(num_steps)
    if any(right <= left for left, right in zip(edges[:-1], edges[1:])):
        raise ValueError("time block construction produced an empty block")
    return tuple(zip(edges[:-1], edges[1:]))


def _teacher_on_common_evaluation(
    *,
    model: DiscreteMPModel,
    fit_pool: torch.Tensor,
    common_evaluation: _LawBank,
    training: DiscreteMPTrainingConfig,
    fit_count: int,
    target_batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    fit = _standard_law_bank(
        model=model,
        initial_state_pool=fit_pool,
        count=int(fit_count),
        seed=int(seed) + 1,
    )
    target = _sample_target_batch(
        model=model,
        pool=fit_pool,
        count=int(target_batch_size),
        seed=int(seed) + 2,
    )
    pathwise_p, target_metadata = model._path_functional_adjoint_targets(
        generated_path=fit.paths,
        controls=fit.controls,
        target_path=target,
        training=training,
    )
    pathwise_r = model._adjoint_noise_target(
        pathwise_adjoint_target=pathwise_p,
        moments_noise_target_dim=int(fit.moments_r.shape[-1]),
        noise=fit.noise,
        control_variate=fit.moments_p,
    )
    with torch.no_grad():
        teacher_p = _project_train_to_eval(
            model=model,
            training_paths=fit.paths,
            training_targets=pathwise_p,
            evaluation_paths=common_evaluation.paths,
            training=training,
        )
        teacher_r = _project_train_to_eval(
            model=model,
            training_paths=fit.paths,
            training_targets=pathwise_r,
            evaluation_paths=common_evaluation.paths,
            training=training,
        )
    return teacher_p.detach(), teacher_r.detach(), {
        "target": target_metadata,
        "pathwise_p_rms": float(torch.sqrt(torch.mean(pathwise_p.double().pow(2))).item()),
        "pathwise_r_rms": float(torch.sqrt(torch.mean(pathwise_r.double().pow(2))).item()),
    }


def _datewise_reliability_from_replicates(
    teachers: torch.Tensor,
    *,
    floor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if teachers.ndim != 4 or int(teachers.shape[0]) < 2:
        raise ValueError("teachers must have shape (K, B, N, d), K >= 2")
    values = teachers.detach().double()
    mean = values.mean(dim=0)
    noise = torch.mean((values - mean.unsqueeze(0)).pow(2), dim=(0, 1, 3))
    centered = mean - mean.mean(dim=0, keepdim=True)
    signal = torch.mean(centered.pow(2), dim=(0, 2))
    eps = torch.finfo(torch.float64).eps
    reliability = signal / (signal + noise).clamp_min(eps)
    reliability = reliability.clamp(min=float(floor), max=1.0)
    return reliability, {
        "mean": float(reliability.mean().item()),
        "min": float(reliability.min().item()),
        "max": float(reliability.max().item()),
        "signal_variance_mean": float(signal.mean().item()),
        "estimator_noise_variance_mean": float(noise.mean().item()),
    }


def _weighted_mean_square(values: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    squared = values.detach().double().pow(2)
    if weight is not None:
        shaped = weight.detach().double().view(
            *((1,) * (squared.ndim - 2)), int(weight.numel()), 1
        )
        squared = squared * shaped
    return torch.mean(squared)


def residual_variance_decomposition(
    prediction: torch.Tensor,
    teachers: torch.Tensor,
    *,
    date_weight: torch.Tensor | None = None,
) -> dict[str, object]:
    """Decompose repeated target residuals into policy bias and estimator noise."""

    if teachers.ndim != prediction.ndim + 1:
        raise ValueError("teachers must add one replicate axis to prediction")
    if tuple(teachers.shape[1:]) != tuple(prediction.shape):
        raise ValueError("prediction and teacher shapes must agree")
    mean = teachers.detach().double().mean(dim=0)
    prediction64 = prediction.detach().double()
    bias_mse = _weighted_mean_square(prediction64 - mean, date_weight)
    noise_mse = torch.mean(
        torch.stack(
            [
                _weighted_mean_square(row.double() - mean, date_weight)
                for row in teachers
            ]
        )
    )
    target_mse = _weighted_mean_square(mean, date_weight)
    expected_mse = bias_mse + noise_mse
    eps = torch.finfo(torch.float64).eps
    replicate_relative = []
    for row in teachers:
        error = _weighted_mean_square(prediction64 - row.double(), date_weight)
        scale = _weighted_mean_square(row.double(), date_weight)
        replicate_relative.append(float(torch.sqrt(error / scale.clamp_min(eps)).item()))
    replicate_tensor = torch.tensor(replicate_relative, dtype=torch.float64)
    return {
        "policy_bias_rms": float(torch.sqrt(bias_mse).item()),
        "estimator_noise_rms": float(torch.sqrt(noise_mse).item()),
        "ensemble_target_rms": float(torch.sqrt(target_mse).item()),
        "relative_policy_bias": float(torch.sqrt(bias_mse / target_mse.clamp_min(eps)).item()),
        "relative_estimator_noise": float(torch.sqrt(noise_mse / target_mse.clamp_min(eps)).item()),
        "estimator_noise_fraction_of_expected_residual_mse": float(
            (noise_mse / expected_mse.clamp_min(eps)).item()
        ),
        "replicate_relative_residuals": replicate_relative,
        "replicate_relative_residual_mean": float(replicate_tensor.mean().item()),
        "replicate_relative_residual_sd": float(
            replicate_tensor.std(unbiased=True).item()
            if int(replicate_tensor.numel()) > 1
            else 0.0
        ),
        "replicate_relative_residual_cv": float(
            (
                replicate_tensor.std(unbiased=True)
                / replicate_tensor.mean().abs().clamp_min(eps)
            ).item()
            if int(replicate_tensor.numel()) > 1
            else 0.0
        ),
    }


def _combined_decomposition(
    prediction_p: torch.Tensor,
    prediction_r: torch.Tensor,
    teachers_p: torch.Tensor,
    teachers_r: torch.Tensor,
    reliability: torch.Tensor,
) -> dict[str, float]:
    mean_p = teachers_p.detach().double().mean(dim=0)
    mean_r = teachers_r.detach().double().mean(dim=0)
    eps = torch.finfo(torch.float64).eps
    p_scale = torch.sqrt(torch.mean(mean_p.pow(2))).clamp_min(eps)
    weight = reliability.detach().double().view(1, -1, 1)
    r_scale = torch.sqrt(torch.mean(weight * mean_r.pow(2))).clamp_min(eps)

    def error_mse(p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            torch.mean(((p.double() - mean_p) / p_scale).pow(2))
            + torch.mean(weight * ((r.double() - mean_r) / r_scale).pow(2))
        )

    bias = error_mse(prediction_p, prediction_r)
    noises = torch.stack(
        [error_mse(row_p, row_r) for row_p, row_r in zip(teachers_p, teachers_r)]
    )
    noise = noises.mean()
    return {
        "dimensionless_policy_bias_norm": float(torch.sqrt(bias).item()),
        "dimensionless_estimator_noise_norm": float(torch.sqrt(noise).item()),
        "estimator_noise_fraction_of_expected_residual_mse": float(
            (noise / (bias + noise).clamp_min(eps)).item()
        ),
    }


def _projection_metrics(
    *,
    prediction_p: torch.Tensor,
    prediction_r: torch.Tensor,
    teachers_p: torch.Tensor,
    teachers_r: torch.Tensor,
    reliability: torch.Tensor,
) -> dict[str, object]:
    return {
        "p": residual_variance_decomposition(prediction_p, teachers_p),
        "r": residual_variance_decomposition(
            prediction_r,
            teachers_r,
            date_weight=reliability,
        ),
        "combined": _combined_decomposition(
            prediction_p,
            prediction_r,
            teachers_p,
            teachers_r,
            reliability,
        ),
    }


def _directional_law_factory(
    *,
    model: DiscreteMPModel,
    base_values: Mapping[str, torch.Tensor],
    candidate_values: Mapping[str, torch.Tensor],
    theta: torch.Tensor,
    blocks: tuple[tuple[int, int], ...],
) -> Callable[[torch.Tensor, int, int], _LawBank]:
    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise ValueError("terminal fixed-point diagnostic requires specific entropy")
    block_count = len(blocks)
    if tuple(theta.shape) != (2 * block_count,):
        raise ValueError("theta must contain P and R coordinates for every block")
    base_network = _network_copy(model, base_values)
    candidate_network = _network_copy(model, candidate_values)
    block_for_step = [
        next(index for index, (left, right) in enumerate(blocks) if left <= step < right)
        for step in range(int(model.grid.num_steps))
    ]

    def factory(pool: torch.Tensor, count: int, seed: int) -> _LawBank:
        x0, noise = _sample_inputs(
            model=model,
            initial_state_pool=pool,
            count=int(count),
            seed=int(seed),
        )
        points = [x0]
        controls = []
        moments_p = []
        moments_r = []
        base_hidden = None
        candidate_hidden = None
        with torch.no_grad():
            for step in range(int(model.grid.num_steps)):
                prefix = torch.stack(points, dim=1)
                base_p_norm, base_r_norm, base_hidden = base_network.forward_step(
                    x_t=points[-1], hidden=base_hidden
                )
                candidate_p_norm, candidate_r_norm, candidate_hidden = (
                    candidate_network.forward_step(
                        x_t=points[-1], hidden=candidate_hidden
                    )
                )
                base_p = model._denormalize_adjoint_step(
                    base_p_norm, step_index=step, noise_adjoint=False
                )
                base_r = model._denormalize_adjoint_step(
                    base_r_norm, step_index=step, noise_adjoint=True
                )
                candidate_p = model._denormalize_adjoint_step(
                    candidate_p_norm, step_index=step, noise_adjoint=False
                )
                candidate_r = model._denormalize_adjoint_step(
                    candidate_r_norm, step_index=step, noise_adjoint=True
                )
                block = block_for_step[step]
                p = base_p + theta[block].to(base_p) * (candidate_p - base_p)
                r = base_r + theta[block_count + block].to(base_r) * (
                    candidate_r - base_r
                )
                control = model.control_map(
                    HamiltonianControlInputs(
                        step_index=step,
                        x_prefix=prefix,
                        expected_adjoint_next=p,
                        expected_adjoint_noise_next=r,
                        parameters=model.parameters,
                    )
                )
                next_state = model.dynamics_step(
                    DynamicsStepInputs(
                        step_index=step,
                        x_prefix=prefix,
                        control=control,
                        noise_next=noise[:, step, :],
                        parameters=model.parameters,
                    )
                )
                if next_state.ndim == 1:
                    next_state = next_state.unsqueeze(-1)
                points.append(next_state)
                controls.append(control)
                moments_p.append(p)
                moments_r.append(r)
        return _LawBank(
            paths=torch.stack(points, dim=1).detach(),
            controls=torch.stack(controls, dim=1).detach(),
            noise=noise.detach(),
            moments_p=torch.stack(moments_p, dim=1).detach(),
            moments_r=torch.stack(moments_r, dim=1).detach(),
        )

    return factory


def _operator_coordinates(
    *,
    model: DiscreteMPModel,
    problem: Any,
    base_values: Mapping[str, torch.Tensor],
    candidate_values: Mapping[str, torch.Tensor],
    blocks: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    base_p, base_r = _moments_for_values(
        model=model, values=base_values, paths=problem.validation.paths
    )
    candidate_p, candidate_r = _moments_for_values(
        model=model, values=candidate_values, paths=problem.validation.paths
    )
    coordinates = []
    basis_report = []
    eps = torch.finfo(torch.float64).eps
    for signal, base, candidate, teacher in (
        ("P", base_p, candidate_p, problem.validation_teacher_p),
        ("R", base_r, candidate_r, problem.validation_teacher_r),
    ):
        for block_index, (left, right) in enumerate(blocks):
            direction = (candidate[:, left:right] - base[:, left:right]).double()
            target_delta = (teacher[:, left:right] - base[:, left:right]).double()
            denominator = torch.sum(direction.pow(2))
            if float(denominator.item()) <= eps:
                raise RuntimeError(
                    f"{signal} direction is numerically zero in block {block_index}"
                )
            coordinate = torch.sum(direction * target_delta) / denominator
            coordinates.append(coordinate)
            basis_report.append(
                {
                    "signal": signal,
                    "block": float(block_index),
                    "left_step": float(left),
                    "right_step_exclusive": float(right),
                    "direction_rms": float(torch.sqrt(torch.mean(direction.pow(2))).item()),
                    "target_projection_coordinate": float(coordinate.item()),
                    "target_projection_explained_fraction": float(
                        (
                            coordinate.pow(2) * denominator
                            / torch.sum(target_delta.pow(2)).clamp_min(eps)
                        ).item()
                    ),
                }
            )
    return torch.stack(coordinates).detach().cpu(), basis_report


def jacobian_spectrum(matrix: torch.Tensor) -> dict[str, object]:
    matrix64 = matrix.detach().double()
    if matrix64.ndim != 2 or int(matrix64.shape[0]) != int(matrix64.shape[1]):
        raise ValueError("matrix must be square")
    eigenvalues = torch.linalg.eigvals(matrix64)
    magnitudes = eigenvalues.abs()
    singular_values = torch.linalg.svdvals(matrix64)
    fixed_point = torch.eye(
        int(matrix64.shape[0]), dtype=matrix64.dtype
    ) - matrix64
    fixed_singular = torch.linalg.svdvals(fixed_point)
    eps = torch.finfo(torch.float64).eps
    return {
        "eigenvalues": [
            {"real": float(value.real.item()), "imag": float(value.imag.item())}
            for value in eigenvalues
        ],
        "spectral_radius_T": float(magnitudes.max().item()),
        "largest_singular_value_T": float(singular_values.max().item()),
        "minimum_singular_value_F": float(fixed_singular.min().item()),
        "condition_number_F": float(
            (fixed_singular.max() / fixed_singular.min().clamp_min(eps)).item()
        ),
    }


def _estimate_reduced_jacobian(
    *,
    model: DiscreteMPModel,
    fit_pool: torch.Tensor,
    validation_pool: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    base_values: Mapping[str, torch.Tensor],
    candidate_values: Mapping[str, torch.Tensor],
    blocks: tuple[tuple[int, int], ...],
    fit_count: int,
    validation_count: int,
    target_batch_size: int,
    step: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    dimension = 2 * len(blocks)
    matrix = torch.empty(dimension, dimension, dtype=torch.float64)
    columns = []
    for column in range(dimension):
        plus = torch.zeros(dimension, dtype=torch.float64)
        minus = torch.zeros(dimension, dtype=torch.float64)
        plus[column] = float(step)
        minus[column] = -float(step)
        problems = []
        coordinate_rows = []
        basis_rows = []
        for theta in (plus, minus):
            law_factory = _directional_law_factory(
                model=model,
                base_values=base_values,
                candidate_values=candidate_values,
                theta=theta,
                blocks=blocks,
            )
            problem = _build_operator_problem(
                model=model,
                fit_pool=fit_pool,
                validation_pool=validation_pool,
                fit_count=int(fit_count),
                validation_count=int(validation_count),
                target_batch_size=int(target_batch_size),
                seed=int(seed),
                training=training,
                r_reliability_floor=0.0,
                law_factory=law_factory,
            )
            coordinates, basis = _operator_coordinates(
                model=model,
                problem=problem,
                base_values=base_values,
                candidate_values=candidate_values,
                blocks=blocks,
            )
            problems.append(problem)
            coordinate_rows.append(coordinates)
            basis_rows.append(basis)
        derivative = (coordinate_rows[0] - coordinate_rows[1]) / (2.0 * float(step))
        matrix[:, column] = derivative
        columns.append(
            {
                "column": int(column),
                "plus_coordinates": coordinate_rows[0].tolist(),
                "minus_coordinates": coordinate_rows[1].tolist(),
                "derivative": derivative.tolist(),
                "plus_basis": basis_rows[0],
                "minus_basis": basis_rows[1],
                "common_random_numbers": True,
            }
        )
        del problems
    return matrix, {"columns": columns, "spectrum": jacobian_spectrum(matrix)}


def audit_terminal_fixed_point(
    *,
    model: DiscreteMPModel,
    target_paths: torch.Tensor,
    checkpoint_training: DiscreteMPTrainingConfig,
    operator_training: DiscreteMPTrainingConfig,
    checkpoint_beta: float,
    operator_beta: float,
    config: TerminalFixedPointDiagnosticConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> TerminalFixedPointDiagnosticResult:
    """Diagnose noise, local contraction, and neural projection at a frozen policy."""

    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise ValueError("terminal fixed-point diagnostic requires specific entropy")
    if float(operator_beta) <= float(checkpoint_beta):
        raise ValueError("operator_beta must be the first rejected beta above checkpoint_beta")
    if str(operator_training.ce_target_mode) != "ridge":
        raise ValueError("terminal fixed-point diagnostic requires ridge projection")
    if str(operator_training.noise_target_control_variate) != "adjoint":
        raise ValueError("terminal fixed-point diagnostic requires the adjoint R control variate")
    model._validate_target_paths(target_paths)
    target_paths = target_paths.to(device=model.device, dtype=model.dtype)
    if int(target_paths.shape[0]) <= int(config.common_evaluation_paths):
        raise ValueError("target_paths must leave a nonempty fit pool")
    validation_pool = target_paths[-int(config.common_evaluation_paths) :]
    fit_pool = target_paths[: -int(config.common_evaluation_paths)]
    base_values = _network_values(model)
    common_evaluation = _standard_law_bank(
        model=model,
        initial_state_pool=validation_pool,
        count=int(config.common_evaluation_paths),
        seed=int(config.seed) + 1_000,
    )

    teachers_p = []
    teachers_r = []
    teacher_metadata = []
    for replicate in range(int(config.residual_replicates)):
        seed = derive_stream_seed(
            base_seed=int(config.seed), stream_offset=10_000 + replicate * 10_000
        )
        if seed is None:
            raise RuntimeError("residual replicate seed derivation failed")
        p, r, metadata = _teacher_on_common_evaluation(
            model=model,
            fit_pool=fit_pool,
            common_evaluation=common_evaluation,
            training=operator_training,
            fit_count=int(config.residual_fit_paths),
            target_batch_size=int(config.residual_target_batch_size),
            seed=int(seed),
        )
        teachers_p.append(p.cpu())
        teachers_r.append(r.cpu())
        teacher_metadata.append(metadata)
        if progress_callback is not None:
            progress_callback(
                f"residual replicate {replicate + 1}/{config.residual_replicates}"
            )
    teacher_stack_p = torch.stack(teachers_p).to(model.device)
    teacher_stack_r = torch.stack(teachers_r).to(model.device)
    r_reliability, r_reliability_report = _datewise_reliability_from_replicates(
        teacher_stack_r,
        floor=float(config.r_reliability_floor),
    )
    base_residual = _projection_metrics(
        prediction_p=common_evaluation.moments_p,
        prediction_r=common_evaluation.moments_r,
        teachers_p=teacher_stack_p,
        teachers_r=teacher_stack_r,
        reliability=r_reliability,
    )

    def standard_factory(pool: torch.Tensor, count: int, seed: int) -> _LawBank:
        return _standard_law_bank(
            model=model,
            initial_state_pool=pool,
            count=count,
            seed=seed,
        )

    projection_problem = _build_operator_problem(
        model=model,
        fit_pool=fit_pool,
        validation_pool=validation_pool,
        fit_count=int(config.projection_fit_paths),
        validation_count=int(config.projection_validation_paths),
        target_batch_size=int(config.projection_target_batch_size),
        seed=int(config.seed) + 500_000,
        training=operator_training,
        r_reliability_floor=float(config.r_reliability_floor),
        law_factory=standard_factory,
    )
    projection_config = SafeguardedExtragradientConfig(
        outer_steps=1,
        operator_fit_paths=int(config.projection_fit_paths),
        operator_validation_paths=int(config.projection_validation_paths),
        operator_target_batch_size=int(config.projection_target_batch_size),
        inner_batch_size=int(config.projection_inner_batch_size),
        inner_max_steps=int(config.projection_max_steps),
        inner_eval_every=int(config.projection_eval_every),
        inner_patience=int(config.projection_patience),
        alpha_radius=0.05,
        log_sigma_radius=0.0025,
        residual_fit_paths=2,
        residual_validation_paths=2,
        objective_paths=2,
        objective_target_batch_size=2,
        objective_banks=1,
    )
    projection_fits = []
    candidate_states = []
    for restart in range(int(config.projection_restarts)):
        candidate, fit_report = _fit_operator_network(
            model=model,
            problem=projection_problem,
            initial_values=base_values,
            training=operator_training,
            config=projection_config,
            seed=int(config.seed) + 600_000 + restart * 10_000,
        )
        projection_fits.append(fit_report)
        candidate_states.append(candidate)
        if progress_callback is not None:
            progress_callback(
                f"projection restart {restart + 1}/{config.projection_restarts}"
            )
    best_index = min(
        range(len(projection_fits)),
        key=lambda index: float(projection_fits[index]["selected"]["validation_loss"]["total"]),
    )
    candidate_values = candidate_states[best_index]
    candidate_p, candidate_r = _moments_for_values(
        model=model,
        values=candidate_values,
        paths=common_evaluation.paths,
    )
    candidate_residual = _projection_metrics(
        prediction_p=candidate_p,
        prediction_r=candidate_r,
        teachers_p=teacher_stack_p,
        teachers_r=teacher_stack_r,
        reliability=r_reliability,
    )
    _load_network_values(model, base_values)

    blocks = _time_blocks(int(model.grid.num_steps), int(config.time_blocks))
    jacobians = []
    jacobian_reports = []
    for replicate in range(int(config.jacobian_replicates)):
        matrix, detail = _estimate_reduced_jacobian(
            model=model,
            fit_pool=fit_pool,
            validation_pool=validation_pool,
            training=operator_training,
            base_values=base_values,
            candidate_values=candidate_values,
            blocks=blocks,
            fit_count=int(config.jacobian_fit_paths),
            validation_count=int(config.jacobian_validation_paths),
            target_batch_size=int(config.jacobian_target_batch_size),
            step=float(config.finite_difference_step),
            seed=int(config.seed) + 1_000_000 + replicate * 10_000,
        )
        jacobians.append(matrix)
        jacobian_reports.append(detail)
        if progress_callback is not None:
            progress_callback(
                f"Jacobian replicate {replicate + 1}/{config.jacobian_replicates}"
            )
    secondary_matrix, secondary_report = _estimate_reduced_jacobian(
        model=model,
        fit_pool=fit_pool,
        validation_pool=validation_pool,
        training=operator_training,
        base_values=base_values,
        candidate_values=candidate_values,
        blocks=blocks,
        fit_count=int(config.jacobian_fit_paths),
        validation_count=int(config.jacobian_validation_paths),
        target_batch_size=int(config.jacobian_target_batch_size),
        step=float(config.secondary_finite_difference_step),
        seed=int(config.seed) + 1_000_000,
    )
    jacobian_stack = torch.stack(jacobians)
    mean_jacobian = jacobian_stack.mean(dim=0)
    jacobian_sd = jacobian_stack.std(dim=0, unbiased=True)
    spectral_radii = torch.tensor(
        [float(row["spectrum"]["spectral_radius_T"]) for row in jacobian_reports],
        dtype=torch.float64,
    )
    eps = torch.finfo(torch.float64).eps
    finite_difference_relative_change = float(
        (
            torch.linalg.vector_norm(jacobians[0] - secondary_matrix)
            / torch.linalg.vector_norm(jacobians[0]).clamp_min(eps)
        ).item()
    )
    jacobian_noise_ratio = float(
        (
            torch.linalg.vector_norm(jacobian_sd)
            / torch.linalg.vector_norm(mean_jacobian).clamp_min(eps)
        ).item()
    )
    _load_network_values(model, base_values)

    report: dict[str, object] = {
        "protocol": {
            "diagnostic_only": True,
            "checkpoint_frozen": True,
            "frozen_checkpoint_updated": False,
            "diagnostic_projection_network_fit": True,
            "operator": "F_beta(z)=z-T_beta(z)",
            "contraction_assessed_from_J_T_not_J_F": True,
            "operator_beta_is_first_rejected_continuation_stage": True,
            "common_random_central_finite_differences": True,
            "jacobian_subspace": (
                f"{2 * len(blocks)}_dimensional_residual_informed_"
                "P_R_time_blocks"
            ),
            "test_paths_used": False,
        },
        "config": dict(config.__dict__),
        "checkpoint_beta": float(checkpoint_beta),
        "operator_beta": float(operator_beta),
        "checkpoint_lambda_scale": float(checkpoint_training.lambda_scale),
        "operator_lambda_scale": float(operator_training.lambda_scale),
        "residual_estimator": {
            "r_reliability": r_reliability_report,
            "base_policy": base_residual,
            "teacher_replicates": teacher_metadata,
        },
        "projection_floor": {
            "best_restart": int(best_index),
            "fits": projection_fits,
            "base_policy": base_residual,
            "best_frozen_target_gru": candidate_residual,
        },
        "local_jacobian": {
            "coordinate_order": [
                f"{signal}_block_{block}"
                for signal in ("P", "R")
                for block in range(len(blocks))
            ],
            "time_blocks": [list(row) for row in blocks],
            "replicates": jacobian_reports,
            "mean_matrix": mean_jacobian.tolist(),
            "elementwise_sd": jacobian_sd.tolist(),
            "mean_matrix_spectrum": jacobian_spectrum(mean_jacobian),
            "replicate_spectral_radius_mean": float(spectral_radii.mean().item()),
            "replicate_spectral_radius_sd": float(spectral_radii.std(unbiased=True).item()),
            "replicate_spectral_radius_min": float(spectral_radii.min().item()),
            "replicate_spectral_radius_max": float(spectral_radii.max().item()),
            "jacobian_replication_noise_frobenius_ratio": jacobian_noise_ratio,
            "secondary_step": float(config.secondary_finite_difference_step),
            "secondary_same_seed_report": secondary_report,
            "finite_difference_step_relative_change": finite_difference_relative_change,
        },
    }
    artifacts = {
        "common_evaluation_paths": common_evaluation.paths.detach().cpu(),
        "common_evaluation_moments_p": common_evaluation.moments_p.detach().cpu(),
        "common_evaluation_moments_r": common_evaluation.moments_r.detach().cpu(),
        "teacher_replicates_p": teacher_stack_p.detach().cpu(),
        "teacher_replicates_r": teacher_stack_r.detach().cpu(),
        "r_reliability": r_reliability.detach().cpu(),
        "best_candidate_network_values": {
            name: value.detach().cpu() for name, value in candidate_values.items()
        },
        "jacobians": jacobian_stack,
        "secondary_jacobian": secondary_matrix,
    }
    return TerminalFixedPointDiagnosticResult(report=report, artifacts=artifacts)


__all__ = [
    "TerminalFixedPointDiagnosticConfig",
    "TerminalFixedPointDiagnosticResult",
    "audit_terminal_fixed_point",
    "jacobian_spectrum",
    "residual_variance_decomposition",
]
