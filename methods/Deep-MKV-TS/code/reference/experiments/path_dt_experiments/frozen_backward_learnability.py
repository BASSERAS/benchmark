from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Any, Callable

import torch

from deep_mkv_gen_path_dt.ce import fit_ridge
from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.model import DiscreteMPModel, RolloutResult
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals
from path_dt_experiments.hamiltonian_regime import (
    REGIME_NAMES,
    prefix_volatility_regimes,
)


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _require_positive_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite float")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return result


@dataclass(frozen=True)
class FrozenBackwardLearnabilityConfig:
    fit_bank_size: int = 1024
    heldout_bank_size: int = 1024
    inner_batch_size: int = 256
    max_steps: int = 5000
    eval_every: int = 100
    plateau_patience: int = 10
    minimum_relative_improvement: float = 1e-4
    split_step: int = 64
    prefix_window: int = 16
    future_horizon: int = 32
    seed: int = 1234

    def __post_init__(self) -> None:
        for name in (
            "fit_bank_size",
            "heldout_bank_size",
            "inner_batch_size",
            "max_steps",
            "eval_every",
            "plateau_patience",
            "split_step",
            "prefix_window",
            "future_horizon",
        ):
            value = _require_int(name, getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
            object.__setattr__(self, name, value)
        if int(self.inner_batch_size) > int(self.fit_bank_size):
            raise ValueError("inner_batch_size cannot exceed fit_bank_size")
        if int(self.eval_every) > int(self.max_steps):
            raise ValueError("eval_every cannot exceed max_steps")
        if int(self.prefix_window) > int(self.split_step):
            raise ValueError("prefix_window must fit before split_step")
        object.__setattr__(
            self,
            "minimum_relative_improvement",
            _require_positive_float(
                "minimum_relative_improvement", self.minimum_relative_improvement
            ),
        )
        object.__setattr__(self, "seed", _require_int("seed", self.seed))


@dataclass(frozen=True)
class FrozenBackwardLearnabilityResult:
    report: dict[str, object]
    artifacts: dict[str, Any]


@dataclass(frozen=True)
class _FrozenBank:
    rollout: RolloutResult
    pathwise_p: torch.Tensor
    pathwise_r: torch.Tensor
    target_paths: torch.Tensor
    target_metadata: dict[str, float]


def _design(
    prefixes: torch.Tensor,
    *,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = prefixes.reshape(int(prefixes.shape[0]), -1)
    if mean is None:
        mean = features.mean(dim=0, keepdim=True)
    if std is None:
        std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(
            float(eps)
        )
    normalized = (features - mean) / std
    ones = torch.ones(
        int(features.shape[0]),
        1,
        device=features.device,
        dtype=features.dtype,
    )
    return torch.cat([ones, normalized], dim=1), mean, std


def fit_ridge_teacher_transfer(
    *,
    fit_paths: torch.Tensor,
    fit_p: torch.Tensor,
    fit_r: torch.Tensor,
    eval_paths: torch.Tensor,
    ridge: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Fit the exact timewise ridge teacher and transfer it to new prefixes."""

    if fit_paths.ndim != 3 or eval_paths.ndim != 3:
        raise ValueError("fit_paths and eval_paths must have shape (B, N + 1, d)")
    if fit_p.ndim != 3 or fit_r.ndim != 3:
        raise ValueError("fit_p and fit_r must have shape (B, N, m)")
    if tuple(fit_p.shape[:2]) != tuple(fit_r.shape[:2]):
        raise ValueError("fit P/R targets must share batch and time dimensions")
    if int(fit_paths.shape[0]) != int(fit_p.shape[0]):
        raise ValueError("fit paths and targets must share a batch size")
    if int(fit_paths.shape[1]) != int(fit_p.shape[1]) + 1:
        raise ValueError("fit targets must have one fewer time point than paths")
    if tuple(eval_paths.shape[1:]) != tuple(fit_paths.shape[1:]):
        raise ValueError("fit and evaluation path shapes must match")
    ridge = float(ridge)
    eps = float(eps)
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and >= 0")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and > 0")

    p_dim = int(fit_p.shape[-1])
    train_p_values = []
    train_r_values = []
    eval_p_values = []
    eval_r_values = []
    fit_residual_sum = fit_p.new_tensor(0.0)
    fit_target_sum = fit_p.new_tensor(0.0)
    for step_index in range(int(fit_p.shape[1])):
        train_design, mean, std = _design(
            fit_paths[:, : step_index + 1, :], eps=eps
        )
        eval_design, _, _ = _design(
            eval_paths[:, : step_index + 1, :], mean=mean, std=std, eps=eps
        )
        combined_target = torch.cat(
            [fit_p[:, step_index, :], fit_r[:, step_index, :]], dim=-1
        )
        coefficients = fit_ridge(
            design=train_design,
            target=combined_target,
            ridge=ridge,
        )
        train_prediction = train_design @ coefficients
        eval_prediction = eval_design @ coefficients
        train_p_values.append(train_prediction[:, :p_dim])
        train_r_values.append(train_prediction[:, p_dim:])
        eval_p_values.append(eval_prediction[:, :p_dim])
        eval_r_values.append(eval_prediction[:, p_dim:])
        fit_residual_sum = fit_residual_sum + torch.sum(
            (train_prediction - combined_target).pow(2)
        )
        centered = combined_target - combined_target.mean(dim=0, keepdim=True)
        fit_target_sum = fit_target_sum + torch.sum(centered.pow(2))
    count = int(fit_p.shape[0]) * int(fit_p.shape[1]) * (
        int(fit_p.shape[-1]) + int(fit_r.shape[-1])
    )
    mse = fit_residual_sum / float(count)
    baseline_mse = fit_target_sum / float(count)
    metadata = {
        "ridge": ridge,
        "eps": eps,
        "fit_rms": float(torch.sqrt(mse).item()),
        "fit_r2": float(
            (1.0 - mse / baseline_mse.clamp_min(torch.finfo(mse.dtype).eps)).item()
        ),
        "maximum_feature_dim": float(int(fit_paths.shape[1])),
    }
    return (
        torch.stack(train_p_values, dim=1).detach(),
        torch.stack(train_r_values, dim=1).detach(),
        torch.stack(eval_p_values, dim=1).detach(),
        torch.stack(eval_r_values, dim=1).detach(),
        metadata,
    )


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_flat = left.detach().double().reshape(-1)
    right_flat = right.detach().double().reshape(-1)
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(
        right_centered
    )
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return None
    return float((torch.dot(left_centered, right_centered) / denominator).item())


def _prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | None]:
    prediction64 = prediction.detach().double()
    target64 = target.detach().double()
    error = prediction64 - target64
    prediction_rms = torch.sqrt(torch.mean(prediction64.pow(2)))
    target_rms = torch.sqrt(torch.mean(target64.pow(2)))
    error_rms = torch.sqrt(torch.mean(error.pow(2)))
    eps = torch.finfo(torch.float64).eps
    baseline = target64.mean(dim=0, keepdim=True)
    baseline_mse = torch.mean((target64 - baseline).pow(2))
    return {
        "prediction_rms": float(prediction_rms.item()),
        "target_rms": float(target_rms.item()),
        "error_rms": float(error_rms.item()),
        "relative_rmse": float((error_rms / target_rms.clamp_min(eps)).item()),
        "prediction_target_rms_ratio": float(
            (prediction_rms / target_rms.clamp_min(eps)).item()
        ),
        "correlation": _correlation(prediction64, target64),
        "r2_against_timewise_mean": float(
            (1.0 - torch.mean(error.pow(2)) / baseline_mse.clamp_min(eps)).item()
        ),
    }


def _regime_prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    regimes: torch.Tensor,
    *,
    split_step: int,
    future_horizon: int,
) -> dict[str, object]:
    report: dict[str, object] = {
        "global": _prediction_metrics(prediction, target),
        "regimes": {},
    }
    at_split = slice(int(split_step), int(split_step) + 1)
    window = slice(int(split_step), int(split_step) + int(future_horizon))
    regime_report: dict[str, object] = {}
    for regime_index, name in enumerate(REGIME_NAMES):
        mask = regimes == regime_index
        if int(mask.sum().item()) == 0:
            raise ValueError("every prefix regime must contain at least one path")
        regime_report[name] = {
            "count": int(mask.sum().item()),
            "at_split": _prediction_metrics(
                prediction[mask, at_split, :], target[mask, at_split, :]
            ),
            "future_control_window": _prediction_metrics(
                prediction[mask, window, :], target[mask, window, :]
            ),
        }
    report["regimes"] = regime_report
    return report


def _paired_signal_report(
    *,
    prediction_p: torch.Tensor,
    prediction_r: torch.Tensor,
    target_p: torch.Tensor,
    target_r: torch.Tensor,
    regimes: torch.Tensor,
    split_step: int,
    future_horizon: int,
) -> dict[str, object]:
    return {
        "p": _regime_prediction_metrics(
            prediction_p,
            target_p,
            regimes,
            split_step=split_step,
            future_horizon=future_horizon,
        ),
        "r": _regime_prediction_metrics(
            prediction_r,
            target_r,
            regimes,
            split_step=split_step,
            future_horizon=future_horizon,
        ),
    }


def prediction_regime_report(
    prediction: torch.Tensor,
    target: torch.Tensor,
    regimes: torch.Tensor,
    *,
    split_step: int,
    future_horizon: int,
) -> dict[str, object]:
    """Public tensor-only wrapper used by follow-up stability diagnostics."""

    return _regime_prediction_metrics(
        prediction,
        target,
        regimes,
        split_step=split_step,
        future_horizon=future_horizon,
    )


def paired_signal_regime_report(
    *,
    prediction_p: torch.Tensor,
    prediction_r: torch.Tensor,
    target_p: torch.Tensor,
    target_r: torch.Tensor,
    regimes: torch.Tensor,
    split_step: int,
    future_horizon: int,
) -> dict[str, object]:
    """Public P/R comparison wrapper used by teacher-stability sweeps."""

    return _paired_signal_report(
        prediction_p=prediction_p,
        prediction_r=prediction_r,
        target_p=target_p,
        target_r=target_r,
        regimes=regimes,
        split_step=split_step,
        future_horizon=future_horizon,
    )


def _build_frozen_bank(
    *,
    model: DiscreteMPModel,
    path_pool: torch.Tensor,
    bank_size: int,
    training: DiscreteMPTrainingConfig,
    index_seed: int,
    noise_seed: int,
) -> _FrozenBank:
    pool = path_pool.to(device=model.device, dtype=model.dtype)
    path_count, _ = model._validate_target_paths(pool)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(index_seed))
    source_indices = model._sample_indices(
        size=path_count, count=int(bank_size), generator=generator
    )
    target_indices = model._sample_indices(
        size=path_count, count=int(bank_size), generator=generator
    )
    x0 = pool.index_select(0, source_indices.to(model.device))[:, 0, :]
    target_batch = pool.index_select(0, target_indices.to(model.device))
    noise = sample_standard_normals(
        grid=model.grid,
        batch_size=int(bank_size),
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=int(noise_seed),
    )
    with torch.no_grad():
        rollout = model.rollout(x0=x0, noise=noise)
    pathwise_p, metadata = model._path_functional_adjoint_targets(
        generated_path=rollout.paths,
        controls=rollout.controls,
        target_path=target_batch,
        training=training,
    )
    control_variate = model._noise_target_control_variate(
        training=training,
        raw_adjoint_prediction=rollout.expected_adjoint_next,
    )
    pathwise_r = model._adjoint_noise_target(
        pathwise_adjoint_target=pathwise_p.detach(),
        moments_noise_target_dim=int(rollout.expected_adjoint_noise_next.shape[-1]),
        noise=noise,
        control_variate=control_variate,
    )
    return _FrozenBank(
        rollout=rollout,
        pathwise_p=pathwise_p.detach(),
        pathwise_r=pathwise_r.detach(),
        target_paths=target_batch.detach(),
        target_metadata={str(name): float(value) for name, value in metadata.items()},
    )


def _state_dict_cpu(network: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }


def audit_frozen_backward_learnability(
    *,
    model: DiscreteMPModel,
    target_paths: torch.Tensor,
    heldout_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    config: FrozenBackwardLearnabilityConfig,
    log_callback: Callable[[dict[str, object]], None] | None = None,
) -> FrozenBackwardLearnabilityResult:
    """Fit one frozen backward problem without updating the forward law."""

    if str(training.ce_target_mode) != "ridge":
        raise ValueError("learnability audit currently requires the ridge CE teacher")
    if str(training.noise_target_control_variate) != "none":
        raise ValueError("learnability audit currently requires no noise control variate")
    if int(config.split_step) + int(config.future_horizon) > int(model.grid.num_steps):
        raise ValueError("future_horizon must fit after split_step")
    if training.grad_clip_norm is not None:
        raise ValueError("this selected-checkpoint audit expects unclipped regression training")

    initial_network_state = _state_dict_cpu(model.network)
    fit_seed = derive_stream_seed(base_seed=config.seed, stream_offset=1_000_000)
    heldout_seed = derive_stream_seed(base_seed=config.seed, stream_offset=1_100_000)
    assert fit_seed is not None and heldout_seed is not None
    fit_bank = _build_frozen_bank(
        model=model,
        path_pool=target_paths,
        bank_size=int(config.fit_bank_size),
        training=training,
        index_seed=int(fit_seed),
        noise_seed=int(fit_seed) + 1,
    )
    heldout_bank = _build_frozen_bank(
        model=model,
        path_pool=heldout_paths,
        bank_size=int(config.heldout_bank_size),
        training=training,
        index_seed=int(heldout_seed),
        noise_seed=int(heldout_seed) + 1,
    )

    (
        fit_teacher_p,
        fit_teacher_r,
        heldout_transferred_p,
        heldout_transferred_r,
        fit_teacher_metadata,
    ) = fit_ridge_teacher_transfer(
        fit_paths=fit_bank.rollout.paths,
        fit_p=fit_bank.pathwise_p,
        fit_r=fit_bank.pathwise_r,
        eval_paths=heldout_bank.rollout.paths,
        ridge=float(training.ridge_lambda),
    )
    (
        heldout_independent_p,
        heldout_independent_r,
        _heldout_self_p,
        _heldout_self_r,
        heldout_teacher_metadata,
    ) = fit_ridge_teacher_transfer(
        fit_paths=heldout_bank.rollout.paths,
        fit_p=heldout_bank.pathwise_p,
        fit_r=heldout_bank.pathwise_r,
        eval_paths=heldout_bank.rollout.paths,
        ridge=float(training.ridge_lambda),
    )

    target_reference = target_paths.to(device=model.device, dtype=model.dtype)
    fit_regimes, regime_thresholds, fit_prefix_rv = prefix_volatility_regimes(
        fit_bank.rollout.paths,
        target_reference,
        split_step=int(config.split_step),
        prefix_window=int(config.prefix_window),
    )
    heldout_regimes, _, heldout_prefix_rv = prefix_volatility_regimes(
        heldout_bank.rollout.paths,
        target_reference,
        split_step=int(config.split_step),
        prefix_window=int(config.prefix_window),
    )

    teacher_report = {
        "fit_projection_of_raw_pathwise_targets": _paired_signal_report(
            prediction_p=fit_teacher_p,
            prediction_r=fit_teacher_r,
            target_p=fit_bank.pathwise_p,
            target_r=fit_bank.pathwise_r,
            regimes=fit_regimes,
            split_step=int(config.split_step),
            future_horizon=int(config.future_horizon),
        ),
        "heldout_projection_of_raw_pathwise_targets": _paired_signal_report(
            prediction_p=heldout_independent_p,
            prediction_r=heldout_independent_r,
            target_p=heldout_bank.pathwise_p,
            target_r=heldout_bank.pathwise_r,
            regimes=heldout_regimes,
            split_step=int(config.split_step),
            future_horizon=int(config.future_horizon),
        ),
        "transferred_fit_teacher_vs_independent_heldout_teacher": (
            _paired_signal_report(
                prediction_p=heldout_transferred_p,
                prediction_r=heldout_transferred_r,
                target_p=heldout_independent_p,
                target_r=heldout_independent_r,
                regimes=heldout_regimes,
                split_step=int(config.split_step),
                future_horizon=int(config.future_horizon),
            )
        ),
        "fit_teacher_metadata": fit_teacher_metadata,
        "heldout_teacher_metadata": heldout_teacher_metadata,
    }

    normalized_fit_p = model._normalize_adjoint_target(
        fit_teacher_p, noise_adjoint=False
    ).detach()
    normalized_fit_r = model._normalize_adjoint_target(
        fit_teacher_r, noise_adjoint=True
    ).detach()
    parameters = list(model.network.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training.lr),
        weight_decay=float(training.weight_decay),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed) + 1_200_000)

    def evaluate(step: int) -> dict[str, object]:
        with torch.no_grad():
            normalized_fit_prediction = model.network(fit_bank.rollout.paths)
            normalized_heldout_prediction = model.network(
                heldout_bank.rollout.paths
            )
            fit_moments = model._denormalize_moments(
                normalized_fit_prediction
            )
            heldout_moments = model._denormalize_moments(
                normalized_heldout_prediction
            )
            p_loss = torch.mean(
                (
                    normalized_fit_prediction.expected_adjoint_next
                    - normalized_fit_p
                ).pow(2)
            )
            r_loss = torch.mean(
                (
                    normalized_fit_prediction.expected_adjoint_noise_next
                    - normalized_fit_r
                ).pow(2)
            )
            total_loss = (
                float(training.adjoint_weight) * p_loss
                + float(training.adjoint_noise_weight) * r_loss
            )
        return {
            "step": int(step),
            "normalized_training_loss": {
                "total": float(total_loss.item()),
                "p": float(p_loss.item()),
                "r": float(r_loss.item()),
            },
            "fit_bank_vs_fit_teacher": _paired_signal_report(
                prediction_p=fit_moments.expected_adjoint_next,
                prediction_r=fit_moments.expected_adjoint_noise_next,
                target_p=fit_teacher_p,
                target_r=fit_teacher_r,
                regimes=fit_regimes,
                split_step=int(config.split_step),
                future_horizon=int(config.future_horizon),
            ),
            "heldout_network_vs_transferred_fit_teacher": _paired_signal_report(
                prediction_p=heldout_moments.expected_adjoint_next,
                prediction_r=heldout_moments.expected_adjoint_noise_next,
                target_p=heldout_transferred_p,
                target_r=heldout_transferred_r,
                regimes=heldout_regimes,
                split_step=int(config.split_step),
                future_horizon=int(config.future_horizon),
            ),
            "heldout_network_vs_independent_teacher": _paired_signal_report(
                prediction_p=heldout_moments.expected_adjoint_next,
                prediction_r=heldout_moments.expected_adjoint_noise_next,
                target_p=heldout_independent_p,
                target_r=heldout_independent_r,
                regimes=heldout_regimes,
                split_step=int(config.split_step),
                future_horizon=int(config.future_horizon),
            ),
        }

    trajectory = [evaluate(0)]
    if log_callback is not None:
        log_callback(dict(trajectory[0]))
    best_loss = float(trajectory[0]["normalized_training_loss"]["total"])
    best_step = 0
    best_network_state = _state_dict_cpu(model.network)
    meaningful_best = best_loss
    plateau_evaluations = 0
    grad_norms: list[float] = []
    stopped_early = False
    for step in range(1, int(config.max_steps) + 1):
        indices_cpu = model._sample_indices(
            size=int(config.fit_bank_size),
            count=int(config.inner_batch_size),
            generator=generator,
        )
        indices = indices_cpu.to(model.device)
        normalized_moments = model.network(
            fit_bank.rollout.paths.index_select(0, indices)
        )
        p_loss = torch.mean(
            (
                normalized_moments.expected_adjoint_next
                - normalized_fit_p.index_select(0, indices)
            ).pow(2)
        )
        r_loss = torch.mean(
            (
                normalized_moments.expected_adjoint_noise_next
                - normalized_fit_r.index_select(0, indices)
            ).pow(2)
        )
        total_loss = (
            float(training.adjoint_weight) * p_loss
            + float(training.adjoint_noise_weight) * r_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = model._grad_norm(parameters)
        grad_norms.append(float(grad_norm))
        if not math.isfinite(grad_norm):
            raise RuntimeError("frozen backward regression produced a non-finite gradient")
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all().item()) for parameter in parameters):
            raise RuntimeError("frozen backward regression produced a non-finite parameter")

        if step % int(config.eval_every) == 0 or step == int(config.max_steps):
            row = evaluate(step)
            trajectory.append(row)
            if log_callback is not None:
                log_callback(dict(row))
            loss = float(row["normalized_training_loss"]["total"])
            if loss < best_loss:
                best_loss = loss
                best_step = step
                best_network_state = _state_dict_cpu(model.network)
            if loss < meaningful_best * (
                1.0 - float(config.minimum_relative_improvement)
            ):
                meaningful_best = loss
                plateau_evaluations = 0
            else:
                plateau_evaluations += 1
            if plateau_evaluations >= int(config.plateau_patience):
                stopped_early = True
                break

    terminal_step = int(trajectory[-1]["step"])
    terminal_network_state = _state_dict_cpu(model.network)
    model.network.load_state_dict(best_network_state)
    best_evaluation = evaluate(best_step)

    grad_tensor = torch.tensor(grad_norms, dtype=torch.float64)
    report: dict[str, object] = {
        "protocol": {
            "forward_law_frozen": True,
            "complete_path_dependent_adjoint": True,
            "neural_loss_unchanged": True,
            "optimizer": "AdamW with the selected training learning rate and weight decay",
            "learning_rate_schedule": False,
            "gradient_clipping": False,
            "heldout_bank_used_for_optimization_or_stopping": False,
            "best_checkpoint_selected_by": "full frozen fit-bank normalized P/R loss",
            "trained_network_is_a_diagnostic_backward_fit_not_a_generator_checkpoint": True,
        },
        "config": {
            **config.__dict__,
            "learning_rate": float(training.lr),
            "weight_decay": float(training.weight_decay),
            "adjoint_weight": float(training.adjoint_weight),
            "adjoint_noise_weight": float(training.adjoint_noise_weight),
            "ridge_lambda": float(training.ridge_lambda),
        },
        "regime_thresholds": [float(value.item()) for value in regime_thresholds],
        "fit_regime_counts": {
            name: int((fit_regimes == index).sum().item())
            for index, name in enumerate(REGIME_NAMES)
        },
        "heldout_regime_counts": {
            name: int((heldout_regimes == index).sum().item())
            for index, name in enumerate(REGIME_NAMES)
        },
        "target_construction": {
            "fit": fit_bank.target_metadata,
            "heldout": heldout_bank.target_metadata,
        },
        "teacher_diagnostics": teacher_report,
        "trajectory": trajectory,
        "optimization": {
            "best_step": int(best_step),
            "best_training_loss": float(best_loss),
            "terminal_step": terminal_step,
            "stopped_early_on_training_plateau": bool(stopped_early),
            "plateau_evaluations": int(plateau_evaluations),
            "gradient_norm_mean": float(grad_tensor.mean().item()),
            "gradient_norm_q95": float(torch.quantile(grad_tensor, 0.95).item()),
            "gradient_norm_max": float(grad_tensor.max().item()),
        },
        "initial_evaluation": trajectory[0],
        "terminal_evaluation": trajectory[-1],
        "best_training_evaluation": best_evaluation,
    }
    artifacts: dict[str, Any] = {
        "initial_network_state_dict": initial_network_state,
        "best_frozen_regression_network_state_dict": best_network_state,
        "terminal_frozen_regression_network_state_dict": terminal_network_state,
        "fit_paths": fit_bank.rollout.paths.detach().cpu(),
        "heldout_paths": heldout_bank.rollout.paths.detach().cpu(),
        "fit_noise": fit_bank.rollout.noise.detach().cpu(),
        "heldout_noise": heldout_bank.rollout.noise.detach().cpu(),
        "fit_pathwise_p": fit_bank.pathwise_p.detach().cpu(),
        "fit_pathwise_r": fit_bank.pathwise_r.detach().cpu(),
        "heldout_pathwise_p": heldout_bank.pathwise_p.detach().cpu(),
        "heldout_pathwise_r": heldout_bank.pathwise_r.detach().cpu(),
        "fit_teacher_p": fit_teacher_p.detach().cpu(),
        "fit_teacher_r": fit_teacher_r.detach().cpu(),
        "heldout_transferred_teacher_p": heldout_transferred_p.detach().cpu(),
        "heldout_transferred_teacher_r": heldout_transferred_r.detach().cpu(),
        "heldout_independent_teacher_p": heldout_independent_p.detach().cpu(),
        "heldout_independent_teacher_r": heldout_independent_r.detach().cpu(),
        "fit_regime_indices": fit_regimes.detach().cpu(),
        "heldout_regime_indices": heldout_regimes.detach().cpu(),
        "regime_thresholds": regime_thresholds.detach().cpu(),
        "fit_prefix_realized_volatility": fit_prefix_rv.detach().cpu(),
        "heldout_prefix_realized_volatility": heldout_prefix_rv.detach().cpu(),
    }
    return FrozenBackwardLearnabilityResult(report=report, artifacts=artifacts)


__all__ = [
    "FrozenBackwardLearnabilityConfig",
    "FrozenBackwardLearnabilityResult",
    "audit_frozen_backward_learnability",
    "fit_ridge_teacher_transfer",
    "paired_signal_regime_report",
    "prediction_regime_report",
]
