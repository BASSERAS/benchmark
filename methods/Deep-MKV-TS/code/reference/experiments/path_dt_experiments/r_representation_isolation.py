from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Callable

import torch

from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from path_dt_experiments.direct_backward_learnability import (
    fixed_baseline_r2_report,
)
from path_dt_experiments.frozen_backward_learnability import (
    FrozenBackwardLearnabilityConfig,
    prediction_regime_report,
)


R_REPRESENTATION_ARMS = ("r_head_only", "separate_r_encoder")


@dataclass(frozen=True)
class RRepresentationIsolationResult:
    report: dict[str, object]
    artifacts: dict[str, object]


def _state_dict_cpu(network: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(torch.sum(parameter.grad.detach().pow(2)).item())
    return squared**0.5


def _scale_summary(scale: torch.Tensor) -> dict[str, float]:
    flat = scale.detach().double().reshape(-1)
    quantiles = torch.quantile(
        flat,
        torch.tensor((0.01, 0.5, 0.99), device=flat.device, dtype=flat.dtype),
    )
    return {
        "minimum": float(flat.min().item()),
        "q01": float(quantiles[0].item()),
        "q50": float(quantiles[1].item()),
        "q99": float(quantiles[2].item()),
        "maximum": float(flat.max().item()),
    }


def _validate_problem(
    *,
    model: DiscreteMPModel,
    fit_paths: torch.Tensor,
    heldout_paths: torch.Tensor,
    fit_r: torch.Tensor,
    heldout_r: torch.Tensor,
    fit_regimes: torch.Tensor,
    heldout_regimes: torch.Tensor,
    config: FrozenBackwardLearnabilityConfig,
) -> None:
    num_steps = int(model.grid.num_steps)
    state_dim = int(model.architecture.state_dim)
    noise_dim = int(model.architecture.noise_dim)
    noise_adjoint_dim = model.architecture.noise_adjoint_dim
    if noise_adjoint_dim is None:
        output_dim = state_dim if state_dim == noise_dim else state_dim * noise_dim
    else:
        output_dim = int(noise_adjoint_dim)
    expected = (
        (
            "fit_paths",
            fit_paths,
            (int(config.fit_bank_size), num_steps + 1, state_dim),
        ),
        (
            "heldout_paths",
            heldout_paths,
            (int(config.heldout_bank_size), num_steps + 1, state_dim),
        ),
        (
            "fit_r",
            fit_r,
            (int(config.fit_bank_size), num_steps, output_dim),
        ),
        (
            "heldout_r",
            heldout_r,
            (int(config.heldout_bank_size), num_steps, output_dim),
        ),
        ("fit_regimes", fit_regimes, (int(config.fit_bank_size),)),
        ("heldout_regimes", heldout_regimes, (int(config.heldout_bank_size),)),
    )
    for name, tensor, shape in expected:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def _configure_arm(
    network: torch.nn.Module,
    *,
    arm: str,
) -> tuple[list[torch.nn.Parameter], list[str]]:
    trainable_names = []
    parameters = []
    for name, parameter in network.named_parameters():
        if arm == "r_head_only":
            trainable = name.startswith("expected_adjoint_noise_next_head.")
        elif arm == "separate_r_encoder":
            trainable = name.startswith("gru.") or name.startswith(
                "expected_adjoint_noise_next_head."
            )
        else:
            raise ValueError(f"unsupported R representation arm {arm}")
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
            parameters.append(parameter)
    if not parameters:
        raise RuntimeError(f"R representation arm {arm} has no trainable parameters")
    return parameters, trainable_names


def _train_arm(
    *,
    model: DiscreteMPModel,
    arm: str,
    fit_paths: torch.Tensor,
    heldout_paths: torch.Tensor,
    fit_r: torch.Tensor,
    heldout_r: torch.Tensor,
    fit_regimes: torch.Tensor,
    heldout_regimes: torch.Tensor,
    selected_fit_p: torch.Tensor,
    selected_heldout_p: torch.Tensor,
    r_scale: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    config: FrozenBackwardLearnabilityConfig,
    log_callback: Callable[[str, dict[str, object]], None] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    network = copy.deepcopy(model.network).to(device=model.device, dtype=model.dtype)
    parameters, parameter_names = _configure_arm(network, arm=arm)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training.lr),
        weight_decay=float(training.weight_decay),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed) + 1_500_000)

    def raw_moments(paths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = network(paths)
        return (
            model._denormalize_adjoint_prediction(
                normalized.expected_adjoint_next,
                noise_adjoint=False,
            ),
            model._denormalize_adjoint_prediction(
                normalized.expected_adjoint_noise_next,
                noise_adjoint=True,
            ),
        )

    def evaluate(step: int) -> dict[str, object]:
        with torch.no_grad():
            fit_p, fit_prediction = raw_moments(fit_paths)
            heldout_p, heldout_prediction = raw_moments(heldout_paths)
            loss = torch.mean(((fit_prediction - fit_r) / r_scale).pow(2))
            fit_p_difference = fit_p - selected_fit_p
            heldout_p_difference = heldout_p - selected_heldout_p
        return {
            "step": int(step),
            "normalized_fit_r_loss": float(loss.item()),
            "fit_vs_direct_r": prediction_regime_report(
                fit_prediction,
                fit_r,
                fit_regimes,
                split_step=int(config.split_step),
                future_horizon=int(config.future_horizon),
            ),
            "heldout_vs_direct_r": prediction_regime_report(
                heldout_prediction,
                heldout_r,
                heldout_regimes,
                split_step=int(config.split_step),
                future_horizon=int(config.future_horizon),
            ),
            "heldout_r2_against_fixed_fit_baseline": fixed_baseline_r2_report(
                prediction=heldout_prediction,
                target=heldout_r,
                fit_target=fit_r,
                eval_regimes=heldout_regimes,
                fit_regimes=fit_regimes,
                split_step=int(config.split_step),
                future_horizon=int(config.future_horizon),
            ),
            "p_output_drift_from_selected": {
                "fit_rms": float(
                    torch.sqrt(torch.mean(fit_p_difference.pow(2))).item()
                ),
                "fit_max_abs": float(fit_p_difference.abs().max().item()),
                "heldout_rms": float(
                    torch.sqrt(torch.mean(heldout_p_difference.pow(2))).item()
                ),
                "heldout_max_abs": float(
                    heldout_p_difference.abs().max().item()
                ),
            },
        }

    trajectory = [evaluate(0)]
    if log_callback is not None:
        log_callback(arm, dict(trajectory[0]))
    best_loss = float(trajectory[0]["normalized_fit_r_loss"])
    meaningful_best = best_loss
    best_step = 0
    best_state = _state_dict_cpu(network)
    plateau_evaluations = 0
    stopped_early = False
    gradient_norms = []
    for step in range(1, int(config.max_steps) + 1):
        indices_cpu = model._sample_indices(
            size=int(config.fit_bank_size),
            count=int(config.inner_batch_size),
            generator=generator,
        )
        indices = indices_cpu.to(model.device)
        _, prediction = raw_moments(fit_paths.index_select(0, indices))
        loss = torch.mean(
            (
                (prediction - fit_r.index_select(0, indices))
                / r_scale
            ).pow(2)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = _gradient_norm(parameters)
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"{arm} produced a non-finite gradient")
        gradient_norms.append(float(gradient_norm))
        optimizer.step()
        if any(
            not bool(torch.isfinite(parameter).all().item())
            for parameter in parameters
        ):
            raise RuntimeError(f"{arm} produced a non-finite parameter")
        if step % int(config.eval_every) == 0 or step == int(config.max_steps):
            row = evaluate(step)
            trajectory.append(row)
            if log_callback is not None:
                log_callback(arm, dict(row))
            current_loss = float(row["normalized_fit_r_loss"])
            if current_loss < best_loss:
                best_loss = current_loss
                best_step = step
                best_state = _state_dict_cpu(network)
            if current_loss < meaningful_best * (
                1.0 - float(config.minimum_relative_improvement)
            ):
                meaningful_best = current_loss
                plateau_evaluations = 0
            else:
                plateau_evaluations += 1
            if plateau_evaluations >= int(config.plateau_patience):
                stopped_early = True
                break

    terminal_state = _state_dict_cpu(network)
    network.load_state_dict(best_state)
    best_evaluation = evaluate(best_step)
    gradients = torch.tensor(gradient_norms, dtype=torch.float64)
    report: dict[str, object] = {
        "arm": arm,
        "trainable_parameter_names": parameter_names,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "trajectory": trajectory,
        "initial_evaluation": trajectory[0],
        "terminal_evaluation": trajectory[-1],
        "best_training_evaluation": best_evaluation,
        "optimization": {
            "best_step": int(best_step),
            "best_training_loss": float(best_loss),
            "terminal_step": int(trajectory[-1]["step"]),
            "stopped_early_on_training_plateau": bool(stopped_early),
            "plateau_evaluations": int(plateau_evaluations),
            "gradient_norm_mean": float(gradients.mean().item()),
            "gradient_norm_q95": float(torch.quantile(gradients, 0.95).item()),
            "gradient_norm_max": float(gradients.max().item()),
        },
    }
    artifacts = {
        "best_network_state_dict": best_state,
        "terminal_network_state_dict": terminal_state,
    }
    return report, artifacts


def audit_frozen_r_representation_isolation(
    *,
    model: DiscreteMPModel,
    fit_paths: torch.Tensor,
    heldout_paths: torch.Tensor,
    fit_r: torch.Tensor,
    heldout_r: torch.Tensor,
    fit_regimes: torch.Tensor,
    heldout_regimes: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    config: FrozenBackwardLearnabilityConfig,
    minimum_scale: float = 1e-3,
    arms: tuple[str, ...] = R_REPRESENTATION_ARMS,
    log_callback: Callable[[str, dict[str, object]], None] | None = None,
) -> RRepresentationIsolationResult:
    """Separate R-head optimization from R-encoder representation capacity."""

    arm_names = tuple(str(arm) for arm in arms)
    if not arm_names or len(set(arm_names)) != len(arm_names):
        raise ValueError("arms must be non-empty and unique")
    if any(arm not in R_REPRESENTATION_ARMS for arm in arm_names):
        raise ValueError(f"arms must be chosen from {R_REPRESENTATION_ARMS}")
    minimum_scale = float(minimum_scale)
    if not math.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be finite and > 0")
    if training.grad_clip_norm is not None:
        raise ValueError("R representation audit expects unclipped selected training")
    _validate_problem(
        model=model,
        fit_paths=fit_paths,
        heldout_paths=heldout_paths,
        fit_r=fit_r,
        heldout_r=heldout_r,
        fit_regimes=fit_regimes,
        heldout_regimes=heldout_regimes,
        config=config,
    )

    initial_state = _state_dict_cpu(model.network)
    device = model.device
    dtype = model.dtype
    fit_paths = fit_paths.to(device=device, dtype=dtype)
    heldout_paths = heldout_paths.to(device=device, dtype=dtype)
    fit_r = fit_r.to(device=device, dtype=dtype)
    heldout_r = heldout_r.to(device=device, dtype=dtype)
    fit_regimes = fit_regimes.to(device=device)
    heldout_regimes = heldout_regimes.to(device=device)
    r_scale = torch.sqrt(torch.mean(fit_r.detach().pow(2), dim=0)).clamp_min(
        minimum_scale
    )
    with torch.no_grad():
        fit_selected = model._denormalize_moments(model.network(fit_paths))
        heldout_selected = model._denormalize_moments(
            model.network(heldout_paths)
        )
        selected_fit_p = fit_selected.expected_adjoint_next.detach()
        selected_heldout_p = heldout_selected.expected_adjoint_next.detach()

    arm_reports: dict[str, object] = {}
    arm_artifacts: dict[str, object] = {}
    for arm in arm_names:
        arm_report, artifacts = _train_arm(
            model=model,
            arm=arm,
            fit_paths=fit_paths,
            heldout_paths=heldout_paths,
            fit_r=fit_r,
            heldout_r=heldout_r,
            fit_regimes=fit_regimes,
            heldout_regimes=heldout_regimes,
            selected_fit_p=selected_fit_p,
            selected_heldout_p=selected_heldout_p,
            r_scale=r_scale,
            training=training,
            config=config,
            log_callback=log_callback,
        )
        arm_reports[arm] = arm_report
        arm_artifacts[arm] = artifacts

    for name, value in model.network.state_dict().items():
        if not torch.equal(value.detach().cpu(), initial_state[name]):
            raise RuntimeError("R representation audit mutated the selected generator")
    report: dict[str, object] = {
        "protocol": {
            "selected_generator_frozen": True,
            "forward_paths_and_crossfit_r_targets_frozen": True,
            "timewise_rms_is_fixed": True,
            "r_population_target_unchanged": True,
            "p_loss_used": False,
            "optimizer": "fixed-rate AdamW",
            "learning_rate_schedule": False,
            "gradient_clipping": False,
            "heldout_bank_used_for_optimization_or_stopping": False,
            "r_head_only_freezes_selected_gru_and_p_head": True,
            "separate_r_encoder_is_an_independent_diagnostic_copy": True,
            "trained_networks_are_not_generator_checkpoints": True,
        },
        "config": {
            **config.__dict__,
            "minimum_scale": minimum_scale,
            "learning_rate": float(training.lr),
            "weight_decay": float(training.weight_decay),
            "arms": list(arm_names),
        },
        "r_scale_summary": _scale_summary(r_scale),
        "arms": arm_reports,
    }
    artifacts: dict[str, object] = {
        "selected_network_state_dict": initial_state,
        "r_scale": r_scale.detach().cpu(),
        "arms": arm_artifacts,
    }
    return RRepresentationIsolationResult(report=report, artifacts=artifacts)


__all__ = [
    "R_REPRESENTATION_ARMS",
    "RRepresentationIsolationResult",
    "audit_frozen_r_representation_isolation",
]
