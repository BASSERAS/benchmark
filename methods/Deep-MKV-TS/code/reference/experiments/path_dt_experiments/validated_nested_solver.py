from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Callable

import torch

from deep_mkv_gen_path_dt.ce import RidgeConditionalExpectation
from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals
from path_dt_experiments.nested_generalization import (
    PARAMETER_BLOCKS,
    apply_parameter_block_interpolation,
    select_block_coordinate_rates,
)


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _positive_float(name: str, value: object) -> float:
    result = float(value)
    if isinstance(value, bool) or not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return result


@dataclass(frozen=True)
class ValidatedNestedSolverConfig:
    outer_steps: int = 4
    backward_fit_paths: int = 1_024
    backward_validation_paths: int = 512
    outer_objective_paths: int = 512
    outer_selection_banks: int = 4
    outer_confirmation_banks: int = 4
    target_batch_size: int = 2_048
    inner_batch_size: int = 256
    inner_max_steps: int = 1_500
    inner_eval_every: int = 50
    inner_patience: int = 8
    minimum_relative_improvement: float = 1e-3
    minimum_outer_relative_improvement: float = 1e-3
    minimum_confirmation_improvement_fraction: float = 0.75
    antithetic_noise: bool = True
    gate_step_indices: tuple[int, ...] = (64, 96)
    minimum_r_teacher_correlation: float = 0.25
    maximum_r_teacher_relative_rmse: float = 1.25
    minimum_validation_p_r2: float = 0.0
    minimum_validation_r_r2: float = 0.0
    outer_relaxations: tuple[float, ...] = (
        0.0,
        0.0001220703125,
        0.000244140625,
        0.00048828125,
        0.0009765625,
        0.001953125,
        0.00390625,
        0.0078125,
        0.015625,
        0.03125,
        0.0625,
        0.125,
    )
    outer_coordinate_passes: int = 2
    proximal_alpha_fraction: float | None = None
    proximal_log_sigma_fraction: float | None = None
    seed: int = 92_031

    def __post_init__(self) -> None:
        for name in (
            "outer_steps",
            "backward_fit_paths",
            "backward_validation_paths",
            "outer_objective_paths",
            "outer_selection_banks",
            "outer_confirmation_banks",
            "target_batch_size",
            "inner_batch_size",
            "inner_max_steps",
            "inner_eval_every",
            "inner_patience",
            "outer_coordinate_passes",
        ):
            value = _require_int(name, getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
            object.__setattr__(self, name, value)
        if int(self.inner_batch_size) > int(self.backward_fit_paths):
            raise ValueError("inner_batch_size cannot exceed backward_fit_paths")
        if int(self.inner_eval_every) > int(self.inner_max_steps):
            raise ValueError("inner_eval_every cannot exceed inner_max_steps")
        if bool(self.antithetic_noise) and (
            int(self.backward_fit_paths) % 2
            or int(self.backward_validation_paths) % 2
            or int(self.outer_objective_paths) % 2
        ):
            raise ValueError(
                "antithetic backward path counts must both be even"
            )
        if not isinstance(self.antithetic_noise, bool):
            raise ValueError("antithetic_noise must be a bool")
        steps = tuple(
            _require_int("gate_step_index", value)
            for value in self.gate_step_indices
        )
        if (
            not steps
            or tuple(sorted(set(steps))) != steps
            or any(value < 0 for value in steps)
        ):
            raise ValueError(
                "gate_step_indices must be strictly increasing nonnegative integers"
            )
        object.__setattr__(self, "gate_step_indices", steps)
        object.__setattr__(
            self,
            "minimum_relative_improvement",
            _positive_float(
                "minimum_relative_improvement",
                self.minimum_relative_improvement,
            ),
        )
        object.__setattr__(
            self,
            "minimum_outer_relative_improvement",
            _positive_float(
                "minimum_outer_relative_improvement",
                self.minimum_outer_relative_improvement,
            ),
        )
        confirmation_fraction = float(
            self.minimum_confirmation_improvement_fraction
        )
        if (
            isinstance(self.minimum_confirmation_improvement_fraction, bool)
            or not math.isfinite(confirmation_fraction)
            or not 0.0 < confirmation_fraction <= 1.0
        ):
            raise ValueError(
                "minimum_confirmation_improvement_fraction must lie in (0, 1]"
            )
        object.__setattr__(
            self,
            "minimum_confirmation_improvement_fraction",
            confirmation_fraction,
        )
        for name in (
            "minimum_r_teacher_correlation",
            "maximum_r_teacher_relative_rmse",
            "minimum_validation_p_r2",
            "minimum_validation_r_r2",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not -1.0 <= float(self.minimum_r_teacher_correlation) <= 1.0:
            raise ValueError(
                "minimum_r_teacher_correlation must lie in [-1, 1]"
            )
        if float(self.maximum_r_teacher_relative_rmse) <= 0.0:
            raise ValueError(
                "maximum_r_teacher_relative_rmse must be positive"
            )
        relaxations = tuple(float(value) for value in self.outer_relaxations)
        if (
            not relaxations
            or len(set(relaxations)) != len(relaxations)
            or tuple(sorted(relaxations)) != relaxations
            or relaxations[0] != 0.0
            or any(not math.isfinite(value) or value < 0.0 for value in relaxations)
        ):
            raise ValueError(
                "outer_relaxations must be unique, increasing, finite, "
                "nonnegative, and start at zero"
            )
        object.__setattr__(self, "outer_relaxations", relaxations)
        proximal_values = (
            self.proximal_alpha_fraction,
            self.proximal_log_sigma_fraction,
        )
        if any(value is not None for value in proximal_values):
            if any(value is None for value in proximal_values):
                raise ValueError(
                    "proximal_alpha_fraction and "
                    "proximal_log_sigma_fraction must be set together"
                )
            alpha_fraction = float(self.proximal_alpha_fraction)
            log_sigma_fraction = float(self.proximal_log_sigma_fraction)
            if (
                not math.isfinite(alpha_fraction)
                or not 0.0 <= alpha_fraction <= 1.0
            ):
                raise ValueError(
                    "proximal_alpha_fraction must lie in [0, 1]"
                )
            if (
                not math.isfinite(log_sigma_fraction)
                or not 0.0 <= log_sigma_fraction <= 1.0
            ):
                raise ValueError(
                    "proximal_log_sigma_fraction must lie in [0, 1]"
                )
            if alpha_fraction == 0.0 and log_sigma_fraction == 0.0:
                raise ValueError(
                    "at least one proximal control fraction must be positive"
                )
            object.__setattr__(
                self, "proximal_alpha_fraction", alpha_fraction
            )
            object.__setattr__(
                self, "proximal_log_sigma_fraction", log_sigma_fraction
            )
        object.__setattr__(self, "seed", _require_int("seed", self.seed))


@dataclass(frozen=True)
class ValidatedNestedSolverResult:
    report: dict[str, object]
    history: list[dict[str, float]]
    artifacts: dict[str, object]


def _network_values(model: DiscreteMPModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.network.named_parameters()
    }


def _load_network_values(
    model: DiscreteMPModel,
    values: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, parameter in model.network.named_parameters():
            parameter.copy_(
                values[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().double().reshape(-1)
    right = right.detach().double().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return None
    return float((torch.dot(left, right) / denominator).item())


def _pair_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
) -> dict[str, float | None]:
    left = left.detach().double()
    right = right.detach().double()
    denominator = torch.sqrt(
        torch.mean(0.5 * (left.pow(2) + right.pow(2)))
    ).clamp_min(torch.finfo(torch.float64).eps)
    return {
        "correlation": _correlation(left, right),
        "relative_rmse": float(
            (
                torch.sqrt(torch.mean((left - right).pow(2)))
                / denominator
            ).item()
        ),
    }


def _prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    training_target: torch.Tensor,
) -> dict[str, float | None]:
    prediction = prediction.detach().double()
    target = target.detach().double()
    training_target = training_target.detach().double()
    baseline = training_target.mean(dim=0, keepdim=True)
    mse = torch.mean((prediction - target).pow(2))
    baseline_mse = torch.mean((baseline - target).pow(2))
    target_rms = torch.sqrt(torch.mean(target.pow(2))).clamp_min(
        torch.finfo(torch.float64).eps
    )
    return {
        "relative_rmse": float((torch.sqrt(mse) / target_rms).item()),
        "r2_against_fit_timewise_mean": float(
            (
                1.0
                - mse
                / baseline_mse.clamp_min(torch.finfo(torch.float64).eps)
            ).item()
        ),
        "correlation": _correlation(prediction, target),
        "prediction_target_rms_ratio": float(
            (
                torch.sqrt(torch.mean(prediction.pow(2)))
                / target_rms
            ).item()
        ),
    }


def _control_move_metrics(
    *,
    current: torch.Tensor,
    target: torch.Tensor,
    relaxed: torch.Tensor,
    state_dim: int,
) -> dict[str, float]:
    if tuple(current.shape) != tuple(target.shape) or tuple(current.shape) != tuple(
        relaxed.shape
    ):
        raise ValueError("control move tensors must have matching shapes")
    if int(current.shape[-1]) != 2 * int(state_dim):
        raise ValueError("control move tensors must pack alpha and sigma")
    eps = torch.finfo(current.dtype).eps
    current_alpha = current[..., :state_dim]
    target_alpha = target[..., :state_dim]
    relaxed_alpha = relaxed[..., :state_dim]
    current_log_sigma = torch.log(current[..., state_dim:].clamp_min(eps))
    target_log_sigma = torch.log(target[..., state_dim:].clamp_min(eps))
    relaxed_log_sigma = torch.log(relaxed[..., state_dim:].clamp_min(eps))

    def rms(values: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(values.detach().double().pow(2)))

    target_alpha_move = rms(target_alpha - current_alpha)
    relaxed_alpha_move = rms(relaxed_alpha - current_alpha)
    target_log_sigma_move = rms(target_log_sigma - current_log_sigma)
    relaxed_log_sigma_move = rms(relaxed_log_sigma - current_log_sigma)
    denominator_eps = torch.finfo(torch.float64).eps
    return {
        "target_alpha_move_rms": float(target_alpha_move.item()),
        "relaxed_alpha_move_rms": float(relaxed_alpha_move.item()),
        "realized_alpha_fraction": float(
            (
                relaxed_alpha_move
                / target_alpha_move.clamp_min(denominator_eps)
            ).item()
        ),
        "target_log_sigma_move_rms": float(target_log_sigma_move.item()),
        "relaxed_log_sigma_move_rms": float(relaxed_log_sigma_move.item()),
        "realized_log_sigma_fraction": float(
            (
                relaxed_log_sigma_move
                / target_log_sigma_move.clamp_min(denominator_eps)
            ).item()
        ),
        "relaxed_sigma_min": float(
            relaxed[..., state_dim:].detach().min().item()
        ),
        "relaxed_sigma_max": float(
            relaxed[..., state_dim:].detach().max().item()
        ),
    }


def _frozen_rollout(
    *,
    model: DiscreteMPModel,
    initial_state_pool: torch.Tensor,
    count: int,
    seed: int,
    antithetic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(count) < 1:
        raise ValueError("rollout count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if bool(antithetic) and int(count) % 2:
        raise ValueError("antithetic rollout count must be even")
    source_count = int(count) // 2 if bool(antithetic) else int(count)
    indices = model._sample_indices(
        size=int(initial_state_pool.shape[0]),
        count=source_count,
        generator=generator,
    )
    source_x0 = initial_state_pool.index_select(
        0,
        indices.to(initial_state_pool.device),
    )[:, 0, :]
    source_noise = sample_standard_normals(
        grid=model.grid,
        batch_size=source_count,
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=derive_stream_seed(base_seed=int(seed), stream_offset=1),
    )
    if bool(antithetic):
        x0 = torch.cat((source_x0, source_x0), dim=0)
        noise = torch.cat((source_noise, -source_noise), dim=0)
    else:
        x0 = source_x0
        noise = source_noise
    with torch.no_grad():
        rollout = model.rollout(x0=x0, noise=noise)
    return x0, noise, rollout.paths.detach(), rollout.controls.detach()


def _sample_target_batch(
    *,
    model: DiscreteMPModel,
    pool: torch.Tensor,
    count: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = model._sample_indices(
        size=int(pool.shape[0]),
        count=int(count),
        generator=generator,
    )
    return pool.index_select(0, indices.to(pool.device))


def _ce_for_training(
    model: DiscreteMPModel,
    training: DiscreteMPTrainingConfig,
):
    estimator = model.ce_estimator
    if isinstance(estimator, RidgeConditionalExpectation):
        return RidgeConditionalExpectation(
            ridge=float(training.ridge_lambda),
            eps=float(estimator.eps),
        )
    return estimator


def _project_train_to_eval(
    *,
    model: DiscreteMPModel,
    training_paths: torch.Tensor,
    training_targets: torch.Tensor,
    evaluation_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig,
) -> torch.Tensor:
    if str(training.ce_target_mode) != "ridge":
        raise ValueError("validated nested solver requires ridge CE projection")
    estimator = _ce_for_training(model, training)
    projected = []
    with torch.no_grad():
        for step in range(int(model.grid.num_steps)):
            result = estimator.fit_predict(
                prefixes_train=training_paths[:, : step + 1, :],
                targets_train=training_targets[:, step, :],
                prefixes_eval=evaluation_paths[:, : step + 1, :],
            )
            projected.append(result.prediction)
    return torch.stack(projected, dim=1)


def _gate_view(
    values: torch.Tensor,
    steps: tuple[int, ...],
) -> torch.Tensor:
    indices = torch.tensor(
        steps,
        device=values.device,
        dtype=torch.long,
    )
    return values.index_select(1, indices)


def _finite_parameters(model: DiscreteMPModel) -> bool:
    return all(
        bool(torch.isfinite(parameter).all().item())
        for parameter in model.network.parameters()
    )


def _objective_bank_summary(
    *,
    baseline_rows: list[dict[str, float]],
    candidate_rows: list[dict[str, float]],
) -> dict[str, object]:
    if not baseline_rows or len(baseline_rows) != len(candidate_rows):
        raise ValueError(
            "objective bank summaries require equally sized nonempty banks"
        )
    rows: list[dict[str, float | bool | int]] = []
    for index, (baseline, candidate) in enumerate(
        zip(baseline_rows, candidate_rows, strict=True)
    ):
        before = float(baseline["complete_objective_value"])
        after = float(candidate["complete_objective_value"])
        finite = bool(math.isfinite(before) and math.isfinite(after))
        scale = max(abs(before), torch.finfo(torch.float64).eps)
        relative_improvement = (
            (before - after) / scale if finite else float("-inf")
        )
        rows.append(
            {
                "bank": int(index + 1),
                "objective_before": before,
                "objective_after": after,
                "objective_change": after - before,
                "relative_improvement": relative_improvement,
                "improved": bool(finite and after < before),
                "finite": finite,
            }
        )
    finite = all(bool(row["finite"]) for row in rows)
    before_mean = float(
        sum(float(row["objective_before"]) for row in rows) / len(rows)
    )
    after_mean = float(
        sum(float(row["objective_after"]) for row in rows) / len(rows)
    )
    return {
        "bank_count": len(rows),
        "all_finite": finite,
        "objective_before_mean": before_mean,
        "objective_after_mean": after_mean,
        "objective_change_mean": after_mean - before_mean,
        "mean_relative_improvement": float(
            sum(float(row["relative_improvement"]) for row in rows)
            / len(rows)
        ),
        "improvement_fraction": float(
            sum(bool(row["improved"]) for row in rows) / len(rows)
        ),
        "banks": rows,
    }


def fit_validated_nested(
    *,
    model: DiscreteMPModel,
    backward_target_paths: torch.Tensor,
    validation_target_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    config: ValidatedNestedSolverConfig,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> ValidatedNestedSolverResult:
    """Run validation-selected backward solves and gated outer MP updates."""

    if training.grad_clip_norm is not None:
        raise ValueError("validated nested solver requires uncapped training")
    if str(training.target_preconditioner) != "none":
        raise ValueError("validated nested solver preserves the raw P/R loss")
    if str(training.noise_target_control_variate) != "adjoint":
        raise ValueError(
            "validated nested solver requires the unbiased adjoint R control variate"
        )
    proximal_enabled = config.proximal_alpha_fraction is not None
    if proximal_enabled and not isinstance(
        model.control_map, SpecificEntropyDiagonalControl
    ):
        raise ValueError(
            "proximal control-space updates currently require "
            "SpecificEntropyDiagonalControl"
        )
    if max(config.gate_step_indices) >= int(model.grid.num_steps):
        raise ValueError("gate_step_indices must precede the terminal point")
    model._validate_target_paths(backward_target_paths)
    model._validate_target_paths(validation_target_paths)
    backward_pool = backward_target_paths.to(
        device=model.device,
        dtype=model.dtype,
    )
    validation_pool = validation_target_paths.to(
        device=model.device,
        dtype=model.dtype,
    )
    model.training = training
    parameters = list(model.network.parameters())
    initial_state = model.checkpoint_state()
    history: list[dict[str, float]] = []
    outer_reports: list[dict[str, object]] = []
    candidate_states: list[dict[str, torch.Tensor]] = []

    for outer_index in range(int(config.outer_steps)):
        outer_seed = derive_stream_seed(
            base_seed=int(config.seed),
            stream_offset=100_000 * outer_index,
        )
        if outer_seed is None:
            raise RuntimeError("validated nested seed derivation failed")
        old_values = _network_values(model)
        (
            _fit_x0,
            fit_noise,
            fit_paths,
            fit_controls,
        ) = _frozen_rollout(
            model=model,
            initial_state_pool=backward_pool,
            count=int(config.backward_fit_paths),
            seed=int(outer_seed) + 1,
            antithetic=bool(config.antithetic_noise),
        )
        (
            validation_x0,
            validation_noise,
            validation_paths,
            validation_controls,
        ) = _frozen_rollout(
            model=model,
            initial_state_pool=validation_pool,
            count=int(config.backward_validation_paths),
            seed=int(outer_seed) + 2,
            antithetic=bool(config.antithetic_noise),
        )
        objective_banks: dict[
            str,
            list[dict[str, torch.Tensor | int]],
        ] = {
            "selection": [],
            "confirmation": [],
        }
        for bank_group, bank_count, seed_offset in (
            (
                "selection",
                int(config.outer_selection_banks),
                10_000,
            ),
            (
                "confirmation",
                int(config.outer_confirmation_banks),
                20_000,
            ),
        ):
            for bank_index in range(bank_count):
                bank_seed = int(outer_seed) + seed_offset + 10 * bank_index
                (
                    bank_x0,
                    bank_noise,
                    _bank_paths,
                    _bank_controls,
                ) = _frozen_rollout(
                    model=model,
                    initial_state_pool=validation_pool,
                    count=int(config.outer_objective_paths),
                    seed=bank_seed,
                    antithetic=bool(config.antithetic_noise),
                )
                bank_target = _sample_target_batch(
                    model=model,
                    pool=validation_pool,
                    count=int(config.target_batch_size),
                    seed=bank_seed + 1,
                )
                objective_banks[bank_group].append(
                    {
                        "x0": bank_x0,
                        "noise": bank_noise,
                        "target": bank_target,
                        "seed": bank_seed,
                        "target_seed": bank_seed + 1,
                    }
                )
        fit_target_batch = _sample_target_batch(
            model=model,
            pool=backward_pool,
            count=int(config.target_batch_size),
            seed=int(outer_seed) + 3,
        )
        validation_target_batch = _sample_target_batch(
            model=model,
            pool=validation_pool,
            count=int(config.target_batch_size),
            seed=int(outer_seed) + 4,
        )
        fit_pathwise_p, fit_target_metadata = (
            model._path_functional_adjoint_targets(
                generated_path=fit_paths,
                controls=fit_controls,
                target_path=fit_target_batch,
                training=training,
            )
        )
        validation_pathwise_p, validation_target_metadata = (
            model._path_functional_adjoint_targets(
                generated_path=validation_paths,
                controls=validation_controls,
                target_path=validation_target_batch,
                training=training,
            )
        )
        with torch.no_grad():
            fit_old_moments = model._denormalize_moments(
                model.network(fit_paths)
            )
            validation_old_moments = model._denormalize_moments(
                model.network(validation_paths)
            )
        fit_pathwise_r = model._adjoint_noise_target(
            pathwise_adjoint_target=fit_pathwise_p,
            moments_noise_target_dim=int(
                fit_old_moments.expected_adjoint_noise_next.shape[-1]
            ),
            noise=fit_noise,
            control_variate=fit_old_moments.expected_adjoint_next,
        )
        validation_pathwise_r = model._adjoint_noise_target(
            pathwise_adjoint_target=validation_pathwise_p,
            moments_noise_target_dim=int(
                validation_old_moments.expected_adjoint_noise_next.shape[-1]
            ),
            noise=validation_noise,
            control_variate=validation_old_moments.expected_adjoint_next,
        )
        with torch.no_grad():
            fit_teacher_p, fit_ce_metadata = model._project_pathwise_target(
                frozen_generated_path=fit_paths,
                pathwise_target=fit_pathwise_p,
                training=training,
            )
            fit_teacher_r, fit_r_ce_metadata = model._project_pathwise_target(
                frozen_generated_path=fit_paths,
                pathwise_target=fit_pathwise_r,
                training=training,
                noise_adjoint=True,
            )
            (
                validation_teacher_p,
                validation_ce_metadata,
            ) = model._project_pathwise_target(
                frozen_generated_path=validation_paths,
                pathwise_target=validation_pathwise_p,
                training=training,
            )
            (
                validation_teacher_r,
                validation_r_ce_metadata,
            ) = model._project_pathwise_target(
                frozen_generated_path=validation_paths,
                pathwise_target=validation_pathwise_r,
                training=training,
                noise_adjoint=True,
            )

        raw_fit_teacher_p = fit_teacher_p
        raw_fit_teacher_r = fit_teacher_r
        raw_validation_teacher_p = validation_teacher_p
        raw_validation_teacher_r = validation_teacher_r
        proximal_target_report: dict[str, object] | None = None
        if proximal_enabled:
            control_map = model.control_map
            if not isinstance(control_map, SpecificEntropyDiagonalControl):
                raise RuntimeError("validated proximal control type changed")
            with torch.no_grad():
                (
                    fit_teacher_p,
                    fit_teacher_r,
                    fit_relaxed_controls,
                    fit_hamiltonian_controls,
                ) = control_map.proximal_moment_targets(
                    paths=fit_paths,
                    current_controls=fit_controls,
                    target_adjoint_next=raw_fit_teacher_p,
                    target_adjoint_noise_next=raw_fit_teacher_r,
                    alpha_fraction=float(config.proximal_alpha_fraction),
                    log_sigma_fraction=float(
                        config.proximal_log_sigma_fraction
                    ),
                )
                (
                    validation_teacher_p,
                    validation_teacher_r,
                    validation_relaxed_controls,
                    validation_hamiltonian_controls,
                ) = control_map.proximal_moment_targets(
                    paths=validation_paths,
                    current_controls=validation_controls,
                    target_adjoint_next=raw_validation_teacher_p,
                    target_adjoint_noise_next=raw_validation_teacher_r,
                    alpha_fraction=float(config.proximal_alpha_fraction),
                    log_sigma_fraction=float(
                        config.proximal_log_sigma_fraction
                    ),
                )
            proximal_target_report = {
                "coordinate_system": "alpha_and_log_sigma",
                "alpha_fraction": float(config.proximal_alpha_fraction),
                "log_sigma_fraction": float(
                    config.proximal_log_sigma_fraction
                ),
                "fit": _control_move_metrics(
                    current=fit_controls,
                    target=fit_hamiltonian_controls,
                    relaxed=fit_relaxed_controls,
                    state_dim=int(model.architecture.state_dim),
                ),
                "validation": _control_move_metrics(
                    current=validation_controls,
                    target=validation_hamiltonian_controls,
                    relaxed=validation_relaxed_controls,
                    state_dim=int(model.architecture.state_dim),
                ),
            }

        raw_stationarity_before = {
            "p": _prediction_metrics(
                validation_old_moments.expected_adjoint_next,
                raw_validation_teacher_p,
                training_target=raw_fit_teacher_p,
            ),
            "r": _prediction_metrics(
                validation_old_moments.expected_adjoint_noise_next,
                raw_validation_teacher_r,
                training_target=raw_fit_teacher_r,
            ),
        }

        half = int(fit_paths.shape[0]) // 2
        half_predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for axis, pathwise in (
            ("p", fit_pathwise_p),
            ("r", fit_pathwise_r),
        ):
            left = _project_train_to_eval(
                model=model,
                training_paths=fit_paths[:half],
                training_targets=pathwise[:half],
                evaluation_paths=validation_paths,
                training=training,
            )
            right = _project_train_to_eval(
                model=model,
                training_paths=fit_paths[half:],
                training_targets=pathwise[half:],
                evaluation_paths=validation_paths,
                training=training,
            )
            half_predictions[axis] = (left, right)
        reliability = {
            axis: _pair_metrics(
                _gate_view(values[0], config.gate_step_indices),
                _gate_view(values[1], config.gate_step_indices),
            )
            for axis, values in half_predictions.items()
        }

        normalized_fit_p = model._normalize_adjoint_target(
            fit_teacher_p,
            noise_adjoint=False,
        ).detach()
        normalized_fit_r = model._normalize_adjoint_target(
            fit_teacher_r,
            noise_adjoint=True,
        ).detach()
        normalized_validation_p = model._normalize_adjoint_target(
            validation_teacher_p,
            noise_adjoint=False,
        ).detach()
        normalized_validation_r = model._normalize_adjoint_target(
            validation_teacher_r,
            noise_adjoint=True,
        ).detach()

        def losses(
            *,
            paths: torch.Tensor,
            target_p: torch.Tensor,
            target_r: torch.Tensor,
            indices: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            selected_paths = (
                paths if indices is None else paths.index_select(0, indices)
            )
            selected_p = (
                target_p
                if indices is None
                else target_p.index_select(0, indices)
            )
            selected_r = (
                target_r
                if indices is None
                else target_r.index_select(0, indices)
            )
            moments = model.network(selected_paths)
            p_loss = torch.mean(
                (moments.expected_adjoint_next - selected_p).pow(2)
            )
            r_loss = torch.mean(
                (moments.expected_adjoint_noise_next - selected_r).pow(2)
            )
            total = (
                float(training.adjoint_weight) * p_loss
                + float(training.adjoint_noise_weight) * r_loss
            )
            return total, p_loss, r_loss

        def evaluate_inner(step: int) -> dict[str, object]:
            with torch.no_grad():
                train_total, train_p_loss, train_r_loss = losses(
                    paths=fit_paths,
                    target_p=normalized_fit_p,
                    target_r=normalized_fit_r,
                )
                val_total, val_p_loss, val_r_loss = losses(
                    paths=validation_paths,
                    target_p=normalized_validation_p,
                    target_r=normalized_validation_r,
                )
                fit_prediction = model._denormalize_moments(
                    model.network(fit_paths)
                )
                validation_prediction = model._denormalize_moments(
                    model.network(validation_paths)
                )
            gate_fit_p = _gate_view(
                fit_teacher_p,
                config.gate_step_indices,
            )
            gate_fit_r = _gate_view(
                fit_teacher_r,
                config.gate_step_indices,
            )
            return {
                "step": int(step),
                "training_loss": {
                    "total": float(train_total.item()),
                    "p": float(train_p_loss.item()),
                    "r": float(train_r_loss.item()),
                },
                "validation_loss": {
                    "total": float(val_total.item()),
                    "p": float(val_p_loss.item()),
                    "r": float(val_r_loss.item()),
                },
                "validation_gate": {
                    "p": _prediction_metrics(
                        _gate_view(
                            validation_prediction.expected_adjoint_next,
                            config.gate_step_indices,
                        ),
                        _gate_view(
                            validation_teacher_p,
                            config.gate_step_indices,
                        ),
                        training_target=gate_fit_p,
                    ),
                    "r": _prediction_metrics(
                        _gate_view(
                            validation_prediction.expected_adjoint_noise_next,
                            config.gate_step_indices,
                        ),
                        _gate_view(
                            validation_teacher_r,
                            config.gate_step_indices,
                        ),
                        training_target=gate_fit_r,
                    ),
                },
            }

        optimizer = torch.optim.AdamW(
            model.network.parameters(),
            lr=float(training.lr),
            weight_decay=float(training.weight_decay),
        )
        initial_evaluation = evaluate_inner(0)
        trajectory = [initial_evaluation]
        best_step = 0
        best_validation_loss = float(
            initial_evaluation["validation_loss"]["total"]
        )
        meaningful_best = best_validation_loss
        best_values = _network_values(model)
        patience = 0
        stopped_early = False
        inner_generator = torch.Generator(device="cpu").manual_seed(
            int(outer_seed) + 5
        )
        gradient_norms = []
        for inner_step in range(1, int(config.inner_max_steps) + 1):
            indices = model._sample_indices(
                size=int(config.backward_fit_paths),
                count=int(config.inner_batch_size),
                generator=inner_generator,
            ).to(model.device)
            total_loss, _, _ = losses(
                paths=fit_paths,
                target_p=normalized_fit_p,
                target_r=normalized_fit_r,
                indices=indices,
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = model._grad_norm(parameters)
            if not math.isfinite(gradient_norm):
                raise RuntimeError("validated nested backward gradient is nonfinite")
            gradient_norms.append(float(gradient_norm))
            optimizer.step()
            if not _finite_parameters(model):
                raise RuntimeError("validated nested backward parameters are nonfinite")
            if (
                inner_step % int(config.inner_eval_every) == 0
                or inner_step == int(config.inner_max_steps)
            ):
                evaluation = evaluate_inner(inner_step)
                trajectory.append(evaluation)
                current = float(evaluation["validation_loss"]["total"])
                if current < best_validation_loss:
                    best_validation_loss = current
                    best_step = inner_step
                    best_values = _network_values(model)
                if current < meaningful_best * (
                    1.0 - float(config.minimum_relative_improvement)
                ):
                    meaningful_best = current
                    patience = 0
                else:
                    patience += 1
                if patience >= int(config.inner_patience):
                    stopped_early = True
                    break
        _load_network_values(model, best_values)
        best_evaluation = evaluate_inner(best_step)
        candidate_values = _network_values(model)
        candidate_states.append(candidate_values)
        with torch.no_grad():
            candidate_validation_moments = model._denormalize_moments(
                model.network(validation_paths)
            )
        raw_stationarity_candidate = {
            "p": _prediction_metrics(
                candidate_validation_moments.expected_adjoint_next,
                raw_validation_teacher_p,
                training_target=raw_fit_teacher_p,
            ),
            "r": _prediction_metrics(
                candidate_validation_moments.expected_adjoint_noise_next,
                raw_validation_teacher_r,
                training_target=raw_fit_teacher_r,
            ),
        }
        candidate_control_move: dict[str, float] | None = None
        if proximal_enabled:
            control_map = model.control_map
            if not isinstance(control_map, SpecificEntropyDiagonalControl):
                raise RuntimeError("validated proximal control type changed")
            with torch.no_grad():
                candidate_validation_controls = (
                    control_map.controls_from_moments(
                        paths=validation_paths,
                        expected_adjoint_next=(
                            candidate_validation_moments.expected_adjoint_next
                        ),
                        expected_adjoint_noise_next=(
                            candidate_validation_moments.expected_adjoint_noise_next
                        ),
                    )
                )
            candidate_control_move = _control_move_metrics(
                current=validation_controls,
                target=validation_hamiltonian_controls,
                relaxed=candidate_validation_controls,
                state_dim=int(model.architecture.state_dim),
            )
        candidate_validation_p_r2 = float(
            best_evaluation["validation_gate"]["p"][
                "r2_against_fit_timewise_mean"
            ]
        )
        candidate_validation_r_r2 = float(
            best_evaluation["validation_gate"]["r"][
                "r2_against_fit_timewise_mean"
            ]
        )
        r_correlation = reliability["r"]["correlation"]
        r_reliability_pass = bool(
            r_correlation is not None
            and float(r_correlation)
            >= float(config.minimum_r_teacher_correlation)
            and float(reliability["r"]["relative_rmse"])
            <= float(config.maximum_r_teacher_relative_rmse)
        )
        validation_pass = bool(
            candidate_validation_p_r2
            >= float(config.minimum_validation_p_r2)
            and candidate_validation_r_r2
            >= float(config.minimum_validation_r_r2)
        )
        backward_gate_pass = bool(r_reliability_pass and validation_pass)

        objective_cache: dict[
            tuple[str, int, float, float, float],
            dict[str, float],
        ] = {}

        def objective_bank_at(
            group: str,
            bank_index: int,
            rates: dict[str, float],
        ) -> dict[str, float]:
            rate_key = tuple(
                float(rates[name]) for name in PARAMETER_BLOCKS
            )
            key = (group, int(bank_index), *rate_key)
            if key in objective_cache:
                return dict(objective_cache[key])
            apply_parameter_block_interpolation(
                model,
                old_values=old_values,
                candidate_values=candidate_values,
                rates=rates,
            )
            bank = objective_banks[group][bank_index]
            with torch.no_grad():
                rollout = model.rollout(
                    x0=bank["x0"],
                    noise=bank["noise"],
                )
                finite = bool(
                    torch.isfinite(rollout.paths).all().item()
                    and torch.isfinite(rollout.controls).all().item()
                )
                if finite:
                    _, metrics = model._complete_objective_metrics(
                        generated_path=rollout.paths,
                        controls=rollout.controls,
                        target_path=bank["target"],
                        training=training,
                    )
                    row = {
                        "complete_objective_value": float(
                            metrics["complete_objective_value"]
                        ),
                        "discrepancy_objective_value": float(
                            metrics["discrepancy_objective_value"]
                        ),
                        "running_cost_value": float(
                            metrics["running_cost_value"]
                        ),
                    }
                else:
                    row = {
                        "complete_objective_value": float("inf"),
                        "discrepancy_objective_value": float("inf"),
                        "running_cost_value": float("inf"),
                    }
            objective_cache[key] = row
            return dict(row)

        def objective_group_rows(
            group: str,
            rates: dict[str, float],
        ) -> list[dict[str, float]]:
            return [
                objective_bank_at(group, bank_index, rates)
                for bank_index in range(len(objective_banks[group]))
            ]

        def selection_objective_at(rates: dict[str, float]) -> float:
            rows = objective_group_rows("selection", rates)
            return float(
                sum(
                    float(row["complete_objective_value"])
                    for row in rows
                )
                / len(rows)
            )

        zero_rates = {name: 0.0 for name in PARAMETER_BLOCKS}
        full_rates = {name: 1.0 for name in PARAMETER_BLOCKS}
        selection_baseline_rows = objective_group_rows(
            "selection",
            zero_rates,
        )
        objective_before = selection_objective_at(zero_rates)
        candidate_objective = selection_objective_at(full_rates)
        if backward_gate_pass:
            selected_rates, selected_objective, coordinate_passes = (
                select_block_coordinate_rates(
                    relaxations=config.outer_relaxations,
                    evaluate=selection_objective_at,
                    passes=int(config.outer_coordinate_passes),
                )
            )
        else:
            selected_rates = dict(zero_rates)
            selected_objective = objective_before
            coordinate_passes = 0
        proposed_rates = dict(selected_rates)
        proposed_selection_objective = float(selected_objective)
        selection_selected_rows = objective_group_rows(
            "selection",
            proposed_rates,
        )
        confirmation_baseline_rows = objective_group_rows(
            "confirmation",
            zero_rates,
        )
        confirmation_selected_rows = objective_group_rows(
            "confirmation",
            proposed_rates,
        )
        selection_summary = _objective_bank_summary(
            baseline_rows=selection_baseline_rows,
            candidate_rows=selection_selected_rows,
        )
        confirmation_summary = _objective_bank_summary(
            baseline_rows=confirmation_baseline_rows,
            candidate_rows=confirmation_selected_rows,
        )
        apply_parameter_block_interpolation(
            model,
            old_values=old_values,
            candidate_values=candidate_values,
            rates=proposed_rates,
        )
        objective_accepted = bool(
            any(float(value) > 0.0 for value in proposed_rates.values())
            and bool(selection_summary["all_finite"])
            and bool(confirmation_summary["all_finite"])
            and float(selection_summary["mean_relative_improvement"])
            >= float(config.minimum_outer_relative_improvement)
            and float(confirmation_summary["mean_relative_improvement"])
            >= float(config.minimum_outer_relative_improvement)
            and float(selection_summary["improvement_fraction"])
            >= float(config.minimum_confirmation_improvement_fraction)
            and float(confirmation_summary["improvement_fraction"])
            >= float(config.minimum_confirmation_improvement_fraction)
        )
        if not objective_accepted:
            selected_rates = dict(zero_rates)
            selected_objective = objective_before
            _load_network_values(model, old_values)

        with torch.no_grad():
            accepted_validation_moments = model._denormalize_moments(
                model.network(validation_paths)
            )
        raw_stationarity_accepted = {
            "p": _prediction_metrics(
                accepted_validation_moments.expected_adjoint_next,
                raw_validation_teacher_p,
                training_target=raw_fit_teacher_p,
            ),
            "r": _prediction_metrics(
                accepted_validation_moments.expected_adjoint_noise_next,
                raw_validation_teacher_r,
                training_target=raw_fit_teacher_r,
            ),
        }
        accepted_control_move: dict[str, float] | None = None
        if proximal_enabled:
            control_map = model.control_map
            if not isinstance(control_map, SpecificEntropyDiagonalControl):
                raise RuntimeError("validated proximal control type changed")
            with torch.no_grad():
                accepted_validation_controls = (
                    control_map.controls_from_moments(
                        paths=validation_paths,
                        expected_adjoint_next=(
                            accepted_validation_moments.expected_adjoint_next
                        ),
                        expected_adjoint_noise_next=(
                            accepted_validation_moments.expected_adjoint_noise_next
                        ),
                    )
                )
            accepted_control_move = _control_move_metrics(
                current=validation_controls,
                target=validation_hamiltonian_controls,
                relaxed=accepted_validation_controls,
                state_dim=int(model.architecture.state_dim),
            )

        selection_rate_keys = sorted(
            {
                key[2:]
                for key in objective_cache
                if key[0] == "selection"
            }
        )
        objective_curve = []
        for rate_key in selection_rate_keys:
            rows = [
                objective_cache[
                    ("selection", bank_index, *rate_key)
                ]
                for bank_index in range(int(config.outer_selection_banks))
            ]
            objective_curve.append(
                {
                    "rates": {
                        name: float(rate_key[position])
                        for position, name in enumerate(PARAMETER_BLOCKS)
                    },
                    "complete_objective_value": float(
                        sum(
                            row["complete_objective_value"]
                            for row in rows
                        )
                        / len(rows)
                    ),
                    "discrepancy_objective_value": float(
                        sum(
                            row["discrepancy_objective_value"]
                            for row in rows
                        )
                        / len(rows)
                    ),
                    "running_cost_value": float(
                        sum(row["running_cost_value"] for row in rows)
                        / len(rows)
                    ),
                    "bank_objective_values": [
                        float(row["complete_objective_value"])
                        for row in rows
                    ],
                }
            )

        gradient_tensor = torch.tensor(
            gradient_norms,
            dtype=torch.float64,
        )
        row: dict[str, object] = {
            "outer_step": int(outer_index + 1),
            "teacher_reliability": reliability,
            "initial_backward_evaluation": initial_evaluation,
            "best_backward_evaluation": best_evaluation,
            "backward_trajectory": trajectory,
            "backward_optimization": {
                "best_step": int(best_step),
                "terminal_step": int(trajectory[-1]["step"]),
                "stopped_early": stopped_early,
                "gradient_norm_mean": float(gradient_tensor.mean().item()),
                "gradient_norm_q95": float(
                    torch.quantile(gradient_tensor, 0.95).item()
                ),
                "gradient_norm_max": float(gradient_tensor.max().item()),
            },
            "gate": {
                "r_teacher_reliability_pass": r_reliability_pass,
                "validation_generalization_pass": validation_pass,
                "backward_gate_pass": backward_gate_pass,
            },
            "raw_stationarity": {
                "before_backward_solve": raw_stationarity_before,
                "full_backward_candidate": raw_stationarity_candidate,
                "accepted_update": raw_stationarity_accepted,
            },
            "proximal_control_step": {
                "enabled": proximal_enabled,
                "target": proximal_target_report,
                "full_backward_candidate_validation_move": (
                    candidate_control_move
                ),
                "accepted_validation_move": accepted_control_move,
                "fixed_points_match_original_stationarity_system": True,
            },
            "outer_update": {
                "proposed_rates": proposed_rates,
                "selected_rates": selected_rates,
                "coordinate_passes": int(coordinate_passes),
                "objective_before": objective_before,
                "full_candidate_objective": candidate_objective,
                "proposed_selection_objective": (
                    proposed_selection_objective
                ),
                "selected_objective": selected_objective,
                "objective_change": selected_objective - objective_before,
                "accepted": objective_accepted,
                "bank_evaluations": len(objective_cache),
                "selection_rate_evaluations": len(selection_rate_keys),
                "selection_banks": selection_summary,
                "confirmation_banks": confirmation_summary,
                "acceptance_rule": {
                    "minimum_mean_relative_improvement": float(
                        config.minimum_outer_relative_improvement
                    ),
                    "minimum_improvement_fraction": float(
                        config.minimum_confirmation_improvement_fraction
                    ),
                    "required_on_selection_and_confirmation_banks": True,
                },
                "objective_curve": objective_curve,
            },
            "target_metadata": {
                "fit": fit_target_metadata,
                "validation": validation_target_metadata,
                "fit_ce": fit_ce_metadata,
                "fit_r_ce": fit_r_ce_metadata,
                "validation_ce": validation_ce_metadata,
                "validation_r_ce": validation_r_ce_metadata,
            },
        }
        outer_reports.append(row)
        flat_history = {
            "outer_step": float(outer_index + 1),
            "backward_best_step": float(best_step),
            "backward_validation_loss": float(
                best_evaluation["validation_loss"]["total"]
            ),
            "backward_validation_p_r2": candidate_validation_p_r2,
            "backward_validation_r_r2": candidate_validation_r_r2,
            "r_teacher_split_correlation": (
                float("nan")
                if r_correlation is None
                else float(r_correlation)
            ),
            "r_teacher_split_relative_rmse": float(
                reliability["r"]["relative_rmse"]
            ),
            "backward_gate_pass": float(backward_gate_pass),
            "outer_update_accepted": float(objective_accepted),
            "outer_relaxation_p": float(selected_rates["p"]),
            "outer_relaxation_r": float(selected_rates["r"]),
            "outer_relaxation_shared": float(selected_rates["shared"]),
            "outer_objective_before": float(objective_before),
            "outer_objective_after": float(selected_objective),
            "raw_stationarity_p_relative_rmse_before": float(
                raw_stationarity_before["p"]["relative_rmse"]
            ),
            "raw_stationarity_p_relative_rmse_after": float(
                raw_stationarity_accepted["p"]["relative_rmse"]
            ),
            "raw_stationarity_r_relative_rmse_before": float(
                raw_stationarity_before["r"]["relative_rmse"]
            ),
            "raw_stationarity_r_relative_rmse_after": float(
                raw_stationarity_accepted["r"]["relative_rmse"]
            ),
        }
        history.append(flat_history)
        if progress_callback is not None:
            progress_callback(row)

    report: dict[str, object] = {
        "protocol": {
            "forward_law_frozen_during_each_backward_solve": True,
            "backward_checkpoint_selected_on_independent_validation_bank": True,
            "outer_update_selected_on_independent_objective_banks": True,
            "outer_update_confirmed_on_separate_independent_banks": True,
            "common_random_numbers_within_each_objective_bank": True,
            "antithetic_full_path_noise": bool(config.antithetic_noise),
            "unbiased_adjoint_r_control_variate": True,
            "raw_p_r_regression_loss_unchanged": True,
            "p_r_regression_loss_form_unchanged": True,
            "regression_target_is_proximal_control_step": proximal_enabled,
            "proximal_coordinates": (
                "alpha_and_log_sigma" if proximal_enabled else None
            ),
            "forward_law_refreshed_after_each_accepted_update": True,
            "proximal_fixed_points_match_original_stationarity_system": True,
            "running_cost_and_discrepancy_unchanged": True,
            "gradient_clipping": False,
            "test_paths_used_for_training_or_selection": False,
            "heston_parameters_or_latent_variance_used": False,
        },
        "config": dict(config.__dict__),
        "training": dict(training.__dict__),
        "outer_iterations": outer_reports,
        "accepted_outer_updates": int(
            sum(
                bool(row["outer_update"]["accepted"])
                for row in outer_reports
            )
        ),
    }
    artifacts: dict[str, object] = {
        "initial_checkpoint": initial_state,
        "final_checkpoint": model.checkpoint_state(),
        "candidate_network_states": candidate_states,
    }
    return ValidatedNestedSolverResult(
        report=report,
        history=history,
        artifacts=artifacts,
    )


__all__ = [
    "ValidatedNestedSolverConfig",
    "ValidatedNestedSolverResult",
    "fit_validated_nested",
]
