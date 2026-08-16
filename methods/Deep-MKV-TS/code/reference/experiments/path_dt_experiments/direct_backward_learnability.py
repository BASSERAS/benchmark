from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import torch

from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from path_dt_experiments.frozen_backward_learnability import (
    FrozenBackwardLearnabilityConfig,
    paired_signal_regime_report,
)
from path_dt_experiments.crossfit_r_control_variate import (
    CrossfitRControlVariateConfig,
    fit_crossfit_variance_minimizing_r_baseline,
)
from path_dt_experiments.hamiltonian_regime import REGIME_NAMES


DIRECT_SCALING_MODES = ("none", "timewise_rms")


@dataclass(frozen=True)
class DirectBackwardLearnabilityResult:
    report: dict[str, object]
    artifacts: dict[str, object]


def _state_dict_cpu(network: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }


def _scale_summary(scale: torch.Tensor) -> dict[str, float]:
    values = scale.detach().double().reshape(-1)
    quantiles = torch.quantile(
        values,
        torch.tensor((0.01, 0.5, 0.99), dtype=torch.float64, device=values.device),
    )
    return {
        "minimum": float(values.min().item()),
        "q01": float(quantiles[0].item()),
        "q50": float(quantiles[1].item()),
        "q99": float(quantiles[2].item()),
        "maximum": float(values.max().item()),
    }


def _target_scales(
    p: torch.Tensor,
    r: torch.Tensor,
    *,
    mode: str,
    minimum_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "none":
        return torch.ones_like(p[0]), torch.ones_like(r[0])
    if mode != "timewise_rms":
        raise ValueError(f"unsupported direct scaling mode {mode}")
    return (
        torch.sqrt(torch.mean(p.detach().pow(2), dim=0)).clamp_min(
            float(minimum_scale)
        ),
        torch.sqrt(torch.mean(r.detach().pow(2), dim=0)).clamp_min(
            float(minimum_scale)
        ),
    )


def _fixed_baseline_r2(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    fit_target: torch.Tensor,
    eval_regimes: torch.Tensor,
    fit_regimes: torch.Tensor,
    split_step: int,
    future_horizon: int,
) -> dict[str, object]:
    prediction64 = prediction.detach().double()
    target64 = target.detach().double()
    fit64 = fit_target.detach().double()
    global_baseline = fit64.mean(dim=0, keepdim=True)
    eps = torch.finfo(torch.float64).eps

    def metrics(
        pred: torch.Tensor,
        actual: torch.Tensor,
        baseline: torch.Tensor,
    ) -> dict[str, float]:
        mse = torch.mean((pred - actual).pow(2))
        baseline_mse = torch.mean((baseline - actual).pow(2))
        return {
            "mse": float(mse.item()),
            "baseline_mse": float(baseline_mse.item()),
            "r2": float(
                (1.0 - mse / baseline_mse.clamp_min(eps)).item()
            ),
        }

    window = slice(int(split_step), int(split_step) + int(future_horizon))
    regimes: dict[str, object] = {}
    for index, name in enumerate(REGIME_NAMES):
        fit_mask = fit_regimes == index
        eval_mask = eval_regimes == index
        if int(fit_mask.sum().item()) == 0 or int(eval_mask.sum().item()) == 0:
            raise ValueError("every direct-target prefix regime must be populated")
        regime_baseline = fit64[fit_mask].mean(dim=0, keepdim=True)
        regimes[name] = {
            "count": int(eval_mask.sum().item()),
            "future_control_window": metrics(
                prediction64[eval_mask, window, :],
                target64[eval_mask, window, :],
                regime_baseline[:, window, :],
            ),
        }
    return {
        "global_timewise_fit_mean": metrics(
            prediction64, target64, global_baseline
        ),
        "regimes_vs_fit_timewise_regime_mean": regimes,
    }


def fixed_baseline_r2_report(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    fit_target: torch.Tensor,
    eval_regimes: torch.Tensor,
    fit_regimes: torch.Tensor,
    split_step: int,
    future_horizon: int,
) -> dict[str, object]:
    """Score predictions against fit-bank-only timewise baselines."""

    return _fixed_baseline_r2(
        prediction=prediction,
        target=target,
        fit_target=fit_target,
        eval_regimes=eval_regimes,
        fit_regimes=fit_regimes,
        split_step=split_step,
        future_horizon=future_horizon,
    )


def _second_moment_ratio(centered: torch.Tensor, uncentered: torch.Tensor) -> float:
    denominator = torch.mean(uncentered.detach().double().pow(2))
    return float(
        (
            torch.mean(centered.detach().double().pow(2))
            / denominator.clamp_min(torch.finfo(torch.float64).eps)
        ).item()
    )


def _noise_adjoint_dim(model: DiscreteMPModel) -> int:
    configured = model.architecture.noise_adjoint_dim
    if configured is not None:
        return int(configured)
    state_dim = int(model.architecture.state_dim)
    noise_dim = int(model.architecture.noise_dim)
    return state_dim if state_dim == noise_dim else state_dim * noise_dim


def _validate_frozen_problem(
    *,
    model: DiscreteMPModel,
    fit_paths: torch.Tensor,
    heldout_paths: torch.Tensor,
    fit_noise: torch.Tensor,
    heldout_noise: torch.Tensor,
    fit_pathwise_p: torch.Tensor,
    heldout_pathwise_p: torch.Tensor,
    fit_regimes: torch.Tensor,
    heldout_regimes: torch.Tensor,
    config: FrozenBackwardLearnabilityConfig,
) -> None:
    num_steps = int(model.grid.num_steps)
    state_dim = int(model.architecture.state_dim)
    noise_dim = int(model.architecture.noise_dim)
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
            "fit_noise",
            fit_noise,
            (int(config.fit_bank_size), num_steps, noise_dim),
        ),
        (
            "heldout_noise",
            heldout_noise,
            (int(config.heldout_bank_size), num_steps, noise_dim),
        ),
        (
            "fit_pathwise_p",
            fit_pathwise_p,
            (int(config.fit_bank_size), num_steps, state_dim),
        ),
        (
            "heldout_pathwise_p",
            heldout_pathwise_p,
            (int(config.heldout_bank_size), num_steps, state_dim),
        ),
        (
            "fit_regimes",
            fit_regimes,
            (int(config.fit_bank_size),),
        ),
        (
            "heldout_regimes",
            heldout_regimes,
            (int(config.heldout_bank_size),),
        ),
    )
    for name, value, shape in expected:
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


def _train_arm(
    *,
    model: DiscreteMPModel,
    initial_state: Mapping[str, torch.Tensor],
    fit_paths: torch.Tensor,
    heldout_paths: torch.Tensor,
    fit_p: torch.Tensor,
    fit_r: torch.Tensor,
    heldout_p: torch.Tensor,
    heldout_r: torch.Tensor,
    fit_regimes: torch.Tensor,
    heldout_regimes: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    config: FrozenBackwardLearnabilityConfig,
    scaling_mode: str,
    minimum_scale: float,
    log_callback: Callable[[str, dict[str, object]], None] | None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    model.network.load_state_dict(initial_state)
    p_scale, r_scale = _target_scales(
        fit_p,
        fit_r,
        mode=scaling_mode,
        minimum_scale=float(minimum_scale),
    )
    parameters = list(model.network.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training.lr),
        weight_decay=float(training.weight_decay),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed) + 1_300_000)

    def evaluate(step: int) -> dict[str, object]:
        with torch.no_grad():
            fit_moments = model._denormalize_moments(model.network(fit_paths))
            heldout_moments = model._denormalize_moments(
                model.network(heldout_paths)
            )
            fit_p_loss = torch.mean(
                ((fit_moments.expected_adjoint_next - fit_p) / p_scale).pow(2)
            )
            fit_r_loss = torch.mean(
                (
                    (fit_moments.expected_adjoint_noise_next - fit_r) / r_scale
                ).pow(2)
            )
            fit_total = (
                float(training.adjoint_weight) * fit_p_loss
                + float(training.adjoint_noise_weight) * fit_r_loss
            )
        fit_report = paired_signal_regime_report(
            prediction_p=fit_moments.expected_adjoint_next,
            prediction_r=fit_moments.expected_adjoint_noise_next,
            target_p=fit_p,
            target_r=fit_r,
            regimes=fit_regimes,
            split_step=int(config.split_step),
            future_horizon=int(config.future_horizon),
        )
        heldout_report = paired_signal_regime_report(
            prediction_p=heldout_moments.expected_adjoint_next,
            prediction_r=heldout_moments.expected_adjoint_noise_next,
            target_p=heldout_p,
            target_r=heldout_r,
            regimes=heldout_regimes,
            split_step=int(config.split_step),
            future_horizon=int(config.future_horizon),
        )
        return {
            "step": int(step),
            "normalized_fit_loss": {
                "total": float(fit_total.item()),
                "p": float(fit_p_loss.item()),
                "r": float(fit_r_loss.item()),
            },
            "fit_vs_direct_targets": fit_report,
            "heldout_vs_direct_targets": heldout_report,
            "heldout_r2_against_fixed_fit_baselines": {
                "p": _fixed_baseline_r2(
                    prediction=heldout_moments.expected_adjoint_next,
                    target=heldout_p,
                    fit_target=fit_p,
                    eval_regimes=heldout_regimes,
                    fit_regimes=fit_regimes,
                    split_step=int(config.split_step),
                    future_horizon=int(config.future_horizon),
                ),
                "r": _fixed_baseline_r2(
                    prediction=heldout_moments.expected_adjoint_noise_next,
                    target=heldout_r,
                    fit_target=fit_r,
                    eval_regimes=heldout_regimes,
                    fit_regimes=fit_regimes,
                    split_step=int(config.split_step),
                    future_horizon=int(config.future_horizon),
                ),
            },
        }

    trajectory = [evaluate(0)]
    if log_callback is not None:
        log_callback(scaling_mode, dict(trajectory[0]))
    best_loss = float(trajectory[0]["normalized_fit_loss"]["total"])
    meaningful_best = best_loss
    best_step = 0
    best_state = _state_dict_cpu(model.network)
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
        moments = model._denormalize_moments(
            model.network(fit_paths.index_select(0, indices))
        )
        p_loss = torch.mean(
            (
                (
                    moments.expected_adjoint_next
                    - fit_p.index_select(0, indices)
                )
                / p_scale
            ).pow(2)
        )
        r_loss = torch.mean(
            (
                (
                    moments.expected_adjoint_noise_next
                    - fit_r.index_select(0, indices)
                )
                / r_scale
            ).pow(2)
        )
        loss = (
            float(training.adjoint_weight) * p_loss
            + float(training.adjoint_noise_weight) * r_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = model._grad_norm(parameters)
        if not math.isfinite(grad_norm):
            raise RuntimeError("direct-target regression produced a non-finite gradient")
        grad_norms.append(float(grad_norm))
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all().item()) for parameter in parameters):
            raise RuntimeError("direct-target regression produced a non-finite parameter")
        if step % int(config.eval_every) == 0 or step == int(config.max_steps):
            row = evaluate(step)
            trajectory.append(row)
            if log_callback is not None:
                log_callback(scaling_mode, dict(row))
            current_loss = float(row["normalized_fit_loss"]["total"])
            if current_loss < best_loss:
                best_loss = current_loss
                best_step = step
                best_state = _state_dict_cpu(model.network)
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

    terminal_step = int(trajectory[-1]["step"])
    terminal_state = _state_dict_cpu(model.network)
    model.network.load_state_dict(best_state)
    best_evaluation = evaluate(best_step)
    gradients = torch.tensor(grad_norms, dtype=torch.float64)
    report: dict[str, object] = {
        "scaling_mode": scaling_mode,
        "scale_summary": {
            "p": _scale_summary(p_scale),
            "r": _scale_summary(r_scale),
        },
        "trajectory": trajectory,
        "initial_evaluation": trajectory[0],
        "terminal_evaluation": trajectory[-1],
        "best_training_evaluation": best_evaluation,
        "optimization": {
            "best_step": int(best_step),
            "best_training_loss": float(best_loss),
            "terminal_step": terminal_step,
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
        "p_scale": p_scale.detach().cpu(),
        "r_scale": r_scale.detach().cpu(),
    }
    return report, artifacts


def audit_frozen_direct_backward_learnability(
    *,
    model: DiscreteMPModel,
    fit_paths: torch.Tensor,
    heldout_paths: torch.Tensor,
    fit_noise: torch.Tensor,
    heldout_noise: torch.Tensor,
    fit_pathwise_p: torch.Tensor,
    heldout_pathwise_p: torch.Tensor,
    fit_regimes: torch.Tensor,
    heldout_regimes: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    config: FrozenBackwardLearnabilityConfig,
    scaling_modes: tuple[str, ...] = DIRECT_SCALING_MODES,
    minimum_scale: float = 1e-3,
    r_control_variate_mode: str = "selected_p",
    crossfit_control_variate: CrossfitRControlVariateConfig | None = None,
    log_callback: Callable[[str, dict[str, object]], None] | None = None,
) -> DirectBackwardLearnabilityResult:
    """Fit direct pathwise P/R samples on a frozen selected-model law."""

    modes = tuple(str(mode) for mode in scaling_modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("scaling_modes must be non-empty and unique")
    if any(mode not in DIRECT_SCALING_MODES for mode in modes):
        raise ValueError(f"scaling_modes must be chosen from {DIRECT_SCALING_MODES}")
    if not math.isfinite(float(minimum_scale)) or float(minimum_scale) <= 0.0:
        raise ValueError("minimum_scale must be finite and > 0")
    control_variate_mode = str(r_control_variate_mode).strip().lower()
    if control_variate_mode not in {"selected_p", "crossfit_knn"}:
        raise ValueError(
            "r_control_variate_mode must be selected_p or crossfit_knn"
        )
    if training.grad_clip_norm is not None:
        raise ValueError("direct frozen audit expects unclipped selected training")
    _validate_frozen_problem(
        model=model,
        fit_paths=fit_paths,
        heldout_paths=heldout_paths,
        fit_noise=fit_noise,
        heldout_noise=heldout_noise,
        fit_pathwise_p=fit_pathwise_p,
        heldout_pathwise_p=heldout_pathwise_p,
        fit_regimes=fit_regimes,
        heldout_regimes=heldout_regimes,
        config=config,
    )
    if int(config.split_step) + int(config.future_horizon) > int(
        fit_pathwise_p.shape[1]
    ):
        raise ValueError("future_horizon must fit after split_step")

    device = model.device
    dtype = model.dtype
    fit_paths = fit_paths.to(device=device, dtype=dtype)
    heldout_paths = heldout_paths.to(device=device, dtype=dtype)
    fit_noise = fit_noise.to(device=device, dtype=dtype)
    heldout_noise = heldout_noise.to(device=device, dtype=dtype)
    fit_p = fit_pathwise_p.to(device=device, dtype=dtype)
    heldout_p = heldout_pathwise_p.to(device=device, dtype=dtype)
    fit_regimes = fit_regimes.to(device=device)
    heldout_regimes = heldout_regimes.to(device=device)
    initial_state = _state_dict_cpu(model.network)
    fit_uncentered_r = model._adjoint_noise_target(
        pathwise_adjoint_target=fit_p,
        moments_noise_target_dim=_noise_adjoint_dim(model),
        noise=fit_noise,
        control_variate=None,
    ).detach()
    heldout_uncentered_r = model._adjoint_noise_target(
        pathwise_adjoint_target=heldout_p,
        moments_noise_target_dim=_noise_adjoint_dim(model),
        noise=heldout_noise,
        control_variate=None,
    ).detach()
    if control_variate_mode == "selected_p":
        with torch.no_grad():
            fit_baseline = model._denormalize_moments(
                model.network(fit_paths)
            ).expected_adjoint_next
            heldout_baseline = model._denormalize_moments(
                model.network(heldout_paths)
            ).expected_adjoint_next
        control_variate_details: dict[str, object] = {
            "method": "selected_checkpoint_causal_p",
            "features_are_prefix_measurable": True,
            "conditional_expectation_is_unchanged": True,
        }
    else:
        crossfit_result = fit_crossfit_variance_minimizing_r_baseline(
            fit_paths=fit_paths,
            fit_pathwise_p=fit_p,
            fit_noise=fit_noise,
            eval_paths=heldout_paths,
            eval_pathwise_p=heldout_p,
            eval_noise=heldout_noise,
            config=crossfit_control_variate,
        )
        fit_baseline = crossfit_result.fit_baseline
        heldout_baseline = crossfit_result.eval_baseline
        control_variate_details = crossfit_result.report
    fit_r = model._adjoint_noise_target(
        pathwise_adjoint_target=fit_p,
        moments_noise_target_dim=_noise_adjoint_dim(model),
        noise=fit_noise,
        control_variate=fit_baseline,
    ).detach()
    heldout_r = model._adjoint_noise_target(
        pathwise_adjoint_target=heldout_p,
        moments_noise_target_dim=_noise_adjoint_dim(model),
        noise=heldout_noise,
        control_variate=heldout_baseline,
    ).detach()

    arm_reports: dict[str, object] = {}
    arm_artifacts: dict[str, object] = {}
    for mode in modes:
        report, artifacts = _train_arm(
            model=model,
            initial_state=initial_state,
            fit_paths=fit_paths,
            heldout_paths=heldout_paths,
            fit_p=fit_p,
            fit_r=fit_r,
            heldout_p=heldout_p,
            heldout_r=heldout_r,
            fit_regimes=fit_regimes,
            heldout_regimes=heldout_regimes,
            training=training,
            config=config,
            scaling_mode=mode,
            minimum_scale=float(minimum_scale),
            log_callback=log_callback,
        )
        arm_reports[mode] = report
        arm_artifacts[mode] = artifacts
    model.network.load_state_dict(initial_state)

    report = {
        "protocol": {
            "forward_paths_and_complete_pathwise_targets_frozen": True,
            "direct_squared_error_population_minimizer_is_conditional_expectation": True,
            "r_control_variate": control_variate_mode,
            "r_control_variate_unbiased": True,
            "neural_architecture_and_loss_family_unchanged": True,
            "running_cost_discrepancy_and_hamiltonian_unchanged": True,
            "optimizer": "fixed-rate AdamW",
            "learning_rate_schedule": False,
            "gradient_clipping": False,
            "heldout_bank_used_for_optimization_or_stopping": False,
            "trained_networks_are_diagnostic_backward_fits_not_generator_checkpoints": True,
        },
        "config": {
            **config.__dict__,
            "scaling_modes": list(modes),
            "minimum_scale": float(minimum_scale),
            "learning_rate": float(training.lr),
            "weight_decay": float(training.weight_decay),
        },
        "control_variate": {
            "mode": control_variate_mode,
            "fit_second_moment_ratio": _second_moment_ratio(
                fit_r, fit_uncentered_r
            ),
            "heldout_second_moment_ratio": _second_moment_ratio(
                heldout_r, heldout_uncentered_r
            ),
            "details": control_variate_details,
        },
        "arms": arm_reports,
    }
    artifacts: dict[str, object] = {
        "initial_network_state_dict": initial_state,
        "fit_direct_p": fit_p.detach().cpu(),
        "fit_direct_r_control_variate": fit_r.detach().cpu(),
        "heldout_direct_p": heldout_p.detach().cpu(),
        "heldout_direct_r_control_variate": heldout_r.detach().cpu(),
        "fit_uncentered_r": fit_uncentered_r.detach().cpu(),
        "heldout_uncentered_r": heldout_uncentered_r.detach().cpu(),
        "fit_r_control_variate_baseline": fit_baseline.detach().cpu(),
        "heldout_r_control_variate_baseline": heldout_baseline.detach().cpu(),
        "arms": arm_artifacts,
    }
    return DirectBackwardLearnabilityResult(report=report, artifacts=artifacts)


__all__ = [
    "DIRECT_SCALING_MODES",
    "DirectBackwardLearnabilityResult",
    "audit_frozen_direct_backward_learnability",
    "fixed_baseline_r2_report",
]
