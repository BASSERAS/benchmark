from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Callable, Mapping

import torch

from deep_mkv_gen_path_dt.networks import AdjointMomentNetwork


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _require_positive_float(name: str, value: object) -> float:
    result = float(value)
    if isinstance(value, bool) or not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return result


@dataclass(frozen=True)
class FrozenSparseTeacherConfig:
    step_indices: tuple[int, ...] = (32, 64, 96)
    fit_prefixes: int = 48
    heldout_prefixes: int | None = None
    validation_prefixes: int = 0
    batch_size: int = 32
    max_steps: int = 5_000
    eval_every: int = 100
    plateau_patience: int = 10
    minimum_relative_improvement: float = 1e-4
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    p_weight: float = 1.0
    r_weight: float = 1.0
    seed: int = 91_031

    def __post_init__(self) -> None:
        steps = tuple(_require_int("step_index", value) for value in self.step_indices)
        if not steps or tuple(sorted(set(steps))) != steps or any(
            value < 1 for value in steps
        ):
            raise ValueError(
                "step_indices must be strictly increasing positive integers"
            )
        object.__setattr__(self, "step_indices", steps)
        for name in (
            "fit_prefixes",
            "batch_size",
            "max_steps",
            "eval_every",
            "plateau_patience",
        ):
            value = _require_int(name, getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
            object.__setattr__(self, name, value)
        heldout_prefixes = self.heldout_prefixes
        if heldout_prefixes is not None:
            heldout_prefixes = _require_int(
                "heldout_prefixes",
                heldout_prefixes,
            )
            if heldout_prefixes < 1:
                raise ValueError("heldout_prefixes must be >= 1 when provided")
            object.__setattr__(
                self,
                "heldout_prefixes",
                heldout_prefixes,
            )
        validation_prefixes = _require_int(
            "validation_prefixes",
            self.validation_prefixes,
        )
        if validation_prefixes < 0:
            raise ValueError("validation_prefixes must be >= 0")
        object.__setattr__(
            self,
            "validation_prefixes",
            validation_prefixes,
        )
        if int(self.batch_size) > int(self.fit_prefixes):
            raise ValueError("batch_size cannot exceed fit_prefixes")
        if int(self.eval_every) > int(self.max_steps):
            raise ValueError("eval_every cannot exceed max_steps")
        for name in (
            "minimum_relative_improvement",
            "learning_rate",
            "p_weight",
            "r_weight",
        ):
            object.__setattr__(
                self,
                name,
                _require_positive_float(name, getattr(self, name)),
            )
        weight_decay = float(self.weight_decay)
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight_decay must be a nonnegative finite float")
        object.__setattr__(self, "weight_decay", weight_decay)
        object.__setattr__(self, "seed", _require_int("seed", self.seed))


@dataclass(frozen=True)
class FrozenSparseTeacherResult:
    report: dict[str, object]
    artifacts: dict[str, object]


def _state_dict_cpu(network: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().double().reshape(-1)
    right = right.detach().double().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return None
    return float((torch.dot(left, right) / denominator).item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().double().reshape(-1)
    right = right.detach().double().reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return None
    return float((torch.dot(left, right) / denominator).item())


def _prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    baseline: torch.Tensor,
) -> dict[str, float | None]:
    prediction = prediction.detach().double()
    target = target.detach().double()
    baseline = baseline.detach().double()
    error_rms = torch.sqrt(torch.mean((prediction - target).pow(2)))
    target_rms = torch.sqrt(torch.mean(target.pow(2)))
    prediction_rms = torch.sqrt(torch.mean(prediction.pow(2)))
    baseline_mse = torch.mean((baseline - target).pow(2))
    mse = torch.mean((prediction - target).pow(2))
    eps = torch.finfo(torch.float64).eps
    return {
        "prediction_rms": float(prediction_rms.item()),
        "target_rms": float(target_rms.item()),
        "prediction_target_rms_ratio": float(
            (prediction_rms / target_rms.clamp_min(eps)).item()
        ),
        "error_rms": float(error_rms.item()),
        "relative_rmse": float(
            (error_rms / target_rms.clamp_min(eps)).item()
        ),
        "correlation": _correlation(prediction, target),
        "cosine": _cosine(prediction, target),
        "r2_against_fit_timewise_mean": float(
            (1.0 - mse / baseline_mse.clamp_min(eps)).item()
        ),
    }


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
        "cosine": _cosine(left, right),
        "relative_rmse": float(
            (
                torch.sqrt(torch.mean((left - right).pow(2)))
                / denominator
            ).item()
        ),
    }


def _predict_selected(
    network: AdjointMomentNetwork,
    prefixes: Mapping[int, torch.Tensor],
    *,
    step_indices: tuple[int, ...],
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    p_values = []
    r_values = []
    for step in step_indices:
        prefix = prefixes[step].index_select(0, indices)
        # The network consumes x_0,...,x_t as x_path[:, :-1].  Appending a
        # duplicate terminal point exposes x_t without changing the prefix.
        extended = torch.cat((prefix, prefix[:, -1:, :]), dim=1)
        moments = network(extended)
        p_values.append(moments.expected_adjoint_next[:, -1, :])
        r_values.append(moments.expected_adjoint_noise_next[:, -1, :])
    return torch.stack(p_values, dim=1), torch.stack(r_values, dim=1)


def _validate_inputs(
    *,
    network: AdjointMomentNetwork,
    prefixes: Mapping[int, torch.Tensor],
    teacher_p: Mapping[int, torch.Tensor],
    teacher_r: Mapping[int, torch.Tensor],
    samples_p: Mapping[int, torch.Tensor],
    samples_r: Mapping[int, torch.Tensor],
    config: FrozenSparseTeacherConfig,
) -> int:
    path_count: int | None = None
    for step in config.step_indices:
        for name, values in (
            ("prefixes", prefixes[step]),
            ("teacher_p", teacher_p[step]),
            ("teacher_r", teacher_r[step]),
            ("samples_p", samples_p[step]),
            ("samples_r", samples_r[step]),
        ):
            if not torch.isfinite(values).all():
                raise ValueError(f"{name}[{step}] must be finite")
        current = int(prefixes[step].shape[0])
        path_count = current if path_count is None else path_count
        if current != path_count:
            raise ValueError("every step must share the same prefix count")
        if tuple(prefixes[step].shape[1:]) != (
            int(step) + 1,
            int(network.state_dim),
        ):
            raise ValueError("prefix shape must align with step and network")
        if tuple(teacher_p[step].shape) != (
            current,
            int(network.adjoint_dim),
        ):
            raise ValueError("teacher P shape does not match the network")
        if tuple(teacher_r[step].shape) != (
            current,
            int(network.noise_adjoint_dim),
        ):
            raise ValueError("teacher R shape does not match the network")
        common_p_shape = tuple(samples_p[step].shape[:3])
        common_r_shape = tuple(samples_r[step].shape[:3])
        if (
            samples_p[step].ndim != 4
            or samples_r[step].ndim != 4
            or common_p_shape != common_r_shape
            or common_p_shape[0] != current
            or common_p_shape[1] < 2
            or common_p_shape[2] < 2
            or int(samples_p[step].shape[3]) != int(network.adjoint_dim)
            or int(samples_r[step].shape[3])
            != int(network.noise_adjoint_dim)
        ):
            raise ValueError(
                "teacher samples must have shape (paths, futures, targets, outputs)"
            )
        reconstructed_p = samples_p[step].mean(dim=(1, 2))
        reconstructed_r = samples_r[step].mean(dim=(1, 2))
        if not torch.allclose(
            reconstructed_p,
            teacher_p[step],
            rtol=1e-5,
            atol=1e-5,
        ) or not torch.allclose(
            reconstructed_r,
            teacher_r[step],
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError("stored conditional teacher must equal sample mean")
    assert path_count is not None
    validation_count = int(config.validation_prefixes)
    if config.heldout_prefixes is None:
        if not 1 <= int(config.fit_prefixes) + validation_count < path_count:
            raise ValueError(
                "fit_prefixes and validation_prefixes must leave at least "
                "one heldout prefix"
            )
    elif (
        int(config.fit_prefixes)
        + validation_count
        + int(config.heldout_prefixes)
        > path_count
    ):
        raise ValueError(
            "fit, validation, and heldout prefixes cannot exceed the "
            "prefix count"
        )
    return path_count


def audit_frozen_sparse_teacher_learnability(
    *,
    network: AdjointMomentNetwork,
    prefixes: Mapping[int, torch.Tensor],
    teacher_p: Mapping[int, torch.Tensor],
    teacher_r: Mapping[int, torch.Tensor],
    samples_p: Mapping[int, torch.Tensor],
    samples_r: Mapping[int, torch.Tensor],
    config: FrozenSparseTeacherConfig,
    log_callback: Callable[[dict[str, object]], None] | None = None,
) -> FrozenSparseTeacherResult:
    """Fit the unchanged P/R loss to fixed branch-averaged causal teachers."""

    path_count = _validate_inputs(
        network=network,
        prefixes=prefixes,
        teacher_p=teacher_p,
        teacher_r=teacher_r,
        samples_p=samples_p,
        samples_r=samples_r,
        config=config,
    )
    device = next(network.parameters()).device
    steps = config.step_indices
    prefix_values = {
        step: prefixes[step].detach().to(
            device=device,
            dtype=next(network.parameters()).dtype,
        )
        for step in steps
    }
    p_values = torch.stack(
        [teacher_p[step] for step in steps],
        dim=1,
    ).to(device=device, dtype=next(network.parameters()).dtype)
    r_values = torch.stack(
        [teacher_r[step] for step in steps],
        dim=1,
    ).to(device=device, dtype=next(network.parameters()).dtype)
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    permutation = torch.randperm(path_count, generator=generator)
    fit_indices = permutation[: int(config.fit_prefixes)].to(device)
    if config.heldout_prefixes is None:
        validation_start = int(config.fit_prefixes)
        validation_stop = validation_start + int(
            config.validation_prefixes
        )
        validation_indices = permutation[
            validation_start:validation_stop
        ].to(device)
        heldout_indices = permutation[validation_stop:].to(device)
    else:
        heldout_start = path_count - int(config.heldout_prefixes)
        validation_start = (
            heldout_start - int(config.validation_prefixes)
        )
        validation_indices = permutation[
            validation_start:heldout_start
        ].to(device)
        heldout_indices = permutation[
            heldout_start:
        ].to(device)
    fit_p = p_values.index_select(0, fit_indices)
    fit_r = r_values.index_select(0, fit_indices)
    heldout_p = p_values.index_select(0, heldout_indices)
    heldout_r = r_values.index_select(0, heldout_indices)
    validation_p = p_values.index_select(0, validation_indices)
    validation_r = r_values.index_select(0, validation_indices)
    baseline_p = fit_p.mean(dim=0, keepdim=True)
    baseline_r = fit_r.mean(dim=0, keepdim=True)
    initial_state = _state_dict_cpu(network)

    def evaluate(step: int) -> dict[str, object]:
        with torch.no_grad():
            fit_prediction_p, fit_prediction_r = _predict_selected(
                network,
                prefix_values,
                step_indices=steps,
                indices=fit_indices,
            )
            heldout_prediction_p, heldout_prediction_r = _predict_selected(
                network,
                prefix_values,
                step_indices=steps,
                indices=heldout_indices,
            )
            if int(validation_indices.numel()) > 0:
                (
                    validation_prediction_p,
                    validation_prediction_r,
                ) = _predict_selected(
                    network,
                    prefix_values,
                    step_indices=steps,
                    indices=validation_indices,
                )
                validation_p_loss = torch.mean(
                    (validation_prediction_p - validation_p).pow(2)
                )
                validation_r_loss = torch.mean(
                    (validation_prediction_r - validation_r).pow(2)
                )
            else:
                validation_prediction_p = None
                validation_prediction_r = None
                validation_p_loss = None
                validation_r_loss = None
            p_loss = torch.mean((fit_prediction_p - fit_p).pow(2))
            r_loss = torch.mean((fit_prediction_r - fit_r).pow(2))
            total = (
                float(config.p_weight) * p_loss
                + float(config.r_weight) * r_loss
            )
        result: dict[str, object] = {
            "step": int(step),
            "training_loss": {
                "total": float(total.item()),
                "p": float(p_loss.item()),
                "r": float(r_loss.item()),
            },
            "fit": {
                "p": _prediction_metrics(
                    fit_prediction_p,
                    fit_p,
                    baseline=baseline_p,
                ),
                "r": _prediction_metrics(
                    fit_prediction_r,
                    fit_r,
                    baseline=baseline_r,
                ),
            },
            "heldout": {
                "p": _prediction_metrics(
                    heldout_prediction_p,
                    heldout_p,
                    baseline=baseline_p,
                ),
                "r": _prediction_metrics(
                    heldout_prediction_r,
                    heldout_r,
                    baseline=baseline_r,
                ),
            },
            "by_step": {},
        }
        if (
            validation_prediction_p is not None
            and validation_prediction_r is not None
            and validation_p_loss is not None
            and validation_r_loss is not None
        ):
            validation_total = (
                float(config.p_weight) * validation_p_loss
                + float(config.r_weight) * validation_r_loss
            )
            result["validation_loss"] = {
                "total": float(validation_total.item()),
                "p": float(validation_p_loss.item()),
                "r": float(validation_r_loss.item()),
            }
            result["validation"] = {
                "p": _prediction_metrics(
                    validation_prediction_p,
                    validation_p,
                    baseline=baseline_p,
                ),
                "r": _prediction_metrics(
                    validation_prediction_r,
                    validation_r,
                    baseline=baseline_r,
                ),
            }
        by_step: dict[str, object] = {}
        for position, step_index in enumerate(steps):
            by_step[str(step_index)] = {
                "fit": {
                    "p": _prediction_metrics(
                        fit_prediction_p[:, position],
                        fit_p[:, position],
                        baseline=baseline_p[:, position],
                    ),
                    "r": _prediction_metrics(
                        fit_prediction_r[:, position],
                        fit_r[:, position],
                        baseline=baseline_r[:, position],
                    ),
                },
                "heldout": {
                    "p": _prediction_metrics(
                        heldout_prediction_p[:, position],
                        heldout_p[:, position],
                        baseline=baseline_p[:, position],
                    ),
                    "r": _prediction_metrics(
                        heldout_prediction_r[:, position],
                        heldout_r[:, position],
                        baseline=baseline_r[:, position],
                    ),
                },
            }
            if (
                validation_prediction_p is not None
                and validation_prediction_r is not None
            ):
                by_step[str(step_index)]["validation"] = {
                    "p": _prediction_metrics(
                        validation_prediction_p[:, position],
                        validation_p[:, position],
                        baseline=baseline_p[:, position],
                    ),
                    "r": _prediction_metrics(
                        validation_prediction_r[:, position],
                        validation_r[:, position],
                        baseline=baseline_r[:, position],
                    ),
                }
        result["by_step"] = by_step
        return result

    initial = evaluate(0)
    trajectory = [initial]
    if log_callback is not None:
        log_callback(initial)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    selection_split = (
        "validation"
        if int(validation_indices.numel()) > 0
        else "fit"
    )
    selection_key = (
        "validation_loss"
        if selection_split == "validation"
        else "training_loss"
    )
    best_loss = float(initial[selection_key]["total"])
    meaningful_best = best_loss
    best_step = 0
    best_state = _state_dict_cpu(network)
    plateau_evaluations = 0
    gradient_norms = []
    stopped_early = False
    batch_generator = torch.Generator(device="cpu").manual_seed(
        int(config.seed) + 1
    )
    parameters = list(network.parameters())
    for step in range(1, int(config.max_steps) + 1):
        selected_cpu = torch.randint(
            0,
            int(config.fit_prefixes),
            (int(config.batch_size),),
            generator=batch_generator,
        )
        selected = fit_indices.index_select(0, selected_cpu.to(device))
        prediction_p, prediction_r = _predict_selected(
            network,
            prefix_values,
            step_indices=steps,
            indices=selected,
        )
        target_p = p_values.index_select(0, selected)
        target_r = r_values.index_select(0, selected)
        p_loss = torch.mean((prediction_p - target_p).pow(2))
        r_loss = torch.mean((prediction_r - target_r).pow(2))
        loss = float(config.p_weight) * p_loss + float(config.r_weight) * r_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.sqrt(
            sum(
                parameter.grad.detach().double().pow(2).sum()
                for parameter in parameters
                if parameter.grad is not None
            )
        )
        gradient_norms.append(float(gradient_norm.item()))
        if not math.isfinite(gradient_norms[-1]):
            raise RuntimeError("frozen sparse teacher produced nonfinite gradients")
        optimizer.step()
        if any(
            not bool(torch.isfinite(parameter).all().item())
            for parameter in parameters
        ):
            raise RuntimeError("frozen sparse teacher produced nonfinite parameters")
        if step % int(config.eval_every) == 0 or step == int(config.max_steps):
            row = evaluate(step)
            trajectory.append(row)
            if log_callback is not None:
                log_callback(row)
            current_loss = float(row[selection_key]["total"])
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
    best = evaluate(best_step)
    gradient_tensor = torch.tensor(gradient_norms, dtype=torch.float64)
    heldout_cpu = heldout_indices.detach().cpu()
    reliability: dict[str, object] = {}
    for position, step in enumerate(steps):
        reliability[str(step)] = {}
        for axis, source in (("p", samples_p), ("r", samples_r)):
            values = source[step].index_select(0, heldout_cpu)
            future_half = int(values.shape[1]) // 2
            target_half = int(values.shape[2]) // 2
            reliability[str(step)][axis] = {
                "future_halves": _pair_metrics(
                    values[:, :future_half].mean(dim=(1, 2)),
                    values[:, future_half : 2 * future_half].mean(
                        dim=(1, 2)
                    ),
                ),
                "target_halves": _pair_metrics(
                    values[:, :, :target_half].mean(dim=(1, 2)),
                    values[:, :, target_half : 2 * target_half].mean(
                        dim=(1, 2)
                    ),
                ),
            }
    fit_interpolated = bool(
        float(best["fit"]["p"]["relative_rmse"]) <= 0.10
        and float(best["fit"]["r"]["relative_rmse"]) <= 0.10
    )
    heldout_generalizes = bool(
        float(best["heldout"]["p"]["r2_against_fit_timewise_mean"]) > 0.0
        and float(best["heldout"]["r"]["r2_against_fit_timewise_mean"]) > 0.0
    )
    future_reliability_sufficient = all(
        metrics[axis]["future_halves"]["correlation"] is not None
        and float(metrics[axis]["future_halves"]["correlation"]) >= 0.8
        and float(metrics[axis]["future_halves"]["relative_rmse"]) <= 0.5
        for metrics in reliability.values()
        for axis in ("p", "r")
    )
    target_reliability_sufficient = all(
        metrics[axis]["target_halves"]["correlation"] is not None
        and float(metrics[axis]["target_halves"]["correlation"]) >= 0.8
        and float(metrics[axis]["target_halves"]["relative_rmse"]) <= 0.5
        for metrics in reliability.values()
        for axis in ("p", "r")
    )
    heldout_interpretable = bool(
        future_reliability_sufficient and target_reliability_sufficient
    )
    by_step_decision: dict[str, object] = {}
    reliable_pair_fit = []
    reliable_pair_generalization = []
    for step in steps:
        step_name = str(step)
        by_step_decision[step_name] = {}
        for axis in ("p", "r"):
            future = reliability[step_name][axis]["future_halves"]
            target = reliability[step_name][axis]["target_halves"]
            teacher_sufficient = bool(
                future["correlation"] is not None
                and float(future["correlation"]) >= 0.8
                and float(future["relative_rmse"]) <= 0.5
                and target["correlation"] is not None
                and float(target["correlation"]) >= 0.8
                and float(target["relative_rmse"]) <= 0.5
            )
            fit_relative_rmse = float(
                best["by_step"][step_name]["fit"][axis]["relative_rmse"]
            )
            heldout_r2 = float(
                best["by_step"][step_name]["heldout"][axis][
                    "r2_against_fit_timewise_mean"
                ]
            )
            fit_sufficient = bool(fit_relative_rmse <= 0.1)
            generalizes = bool(heldout_r2 > 0.0)
            if teacher_sufficient:
                reliable_pair_fit.append(fit_sufficient)
                reliable_pair_generalization.append(generalizes)
            by_step_decision[step_name][axis] = {
                "teacher_reliability_sufficient": teacher_sufficient,
                "same_network_fits_fixed_target": fit_sufficient,
                "heldout_mapping_generalizes": generalizes,
                "heldout_mapping_conclusion_interpretable": (
                    teacher_sufficient
                ),
            }
    every_reliable_pair_fit = bool(
        reliable_pair_fit and all(reliable_pair_fit)
    )
    every_reliable_pair_generalizes = bool(
        reliable_pair_generalization
        and all(reliable_pair_generalization)
    )
    reliable_mapping_failure = bool(
        reliable_pair_generalization
        and not all(reliable_pair_generalization)
    )
    report: dict[str, object] = {
        "protocol": {
            "forward_generator_frozen": True,
            "teacher_fixed_before_optimization": True,
            "teacher_is_branch_and_target_batch_average": True,
            "neural_architecture_unchanged": True,
            "raw_p_r_regression_loss_unchanged": True,
            "gradient_clipping": False,
            "learning_rate_schedule": False,
            "heldout_prefixes_used_for_optimization_or_selection": False,
            "validation_prefixes_used_for_checkpoint_selection": bool(
                validation_indices.numel() > 0
            ),
            "trained_network_is_diagnostic_only": True,
        },
        "config": dict(config.__dict__),
        "fit_prefix_count": int(fit_indices.numel()),
        "validation_prefix_count": int(validation_indices.numel()),
        "heldout_prefix_count": int(heldout_indices.numel()),
        "unused_prefix_count": int(
            path_count
            - fit_indices.numel()
            - validation_indices.numel()
            - heldout_indices.numel()
        ),
        "teacher_reliability": reliability,
        "initial_evaluation": initial,
        "best_evaluation": best,
        "terminal_evaluation": trajectory[-1],
        "trajectory": trajectory,
        "optimization": {
            "best_step": int(best_step),
            "selection_split": selection_split,
            "best_selection_loss": best_loss,
            "best_training_loss": float(
                best["training_loss"]["total"]
            ),
            "terminal_step": int(trajectory[-1]["step"]),
            "stopped_early": stopped_early,
            "gradient_norm_mean": float(gradient_tensor.mean().item()),
            "gradient_norm_q95": float(
                torch.quantile(gradient_tensor, 0.95).item()
            ),
            "gradient_norm_max": float(gradient_tensor.max().item()),
        },
        "decision": {
            "same_network_can_interpolate_frozen_teacher": fit_interpolated,
            "finite_sample_representation_capacity_failure_rejected": (
                fit_interpolated
            ),
            "learned_mapping_generalizes_to_heldout_prefixes": (
                heldout_generalizes
            ),
            "future_branch_teacher_reliability_sufficient": (
                future_reliability_sufficient
            ),
            "target_batch_teacher_reliability_sufficient": (
                target_reliability_sufficient
            ),
            "heldout_generalization_is_interpretable": heldout_interpretable,
            "more_future_branches_needed_before_representation_conclusion": (
                not future_reliability_sufficient
            ),
            "moving_target_or_online_optimization_remains_a_candidate": (
                fit_interpolated or every_reliable_pair_fit
            ),
            "same_network_fits_every_reliable_step_axis_target": (
                every_reliable_pair_fit
            ),
            "every_reliable_step_axis_mapping_generalizes": (
                every_reliable_pair_generalizes
            ),
            "heldout_mapping_or_training_sample_limitation_supported": (
                reliable_mapping_failure
            ),
            "by_step_and_axis": by_step_decision,
        },
    }
    artifacts: dict[str, object] = {
        "initial_network_state_dict": initial_state,
        "best_network_state_dict": best_state,
        "terminal_network_state_dict": terminal_state,
        "fit_indices": fit_indices.detach().cpu(),
        "validation_indices": validation_indices.detach().cpu(),
        "heldout_indices": heldout_indices.detach().cpu(),
    }
    return FrozenSparseTeacherResult(report=report, artifacts=artifacts)


__all__ = [
    "FrozenSparseTeacherConfig",
    "FrozenSparseTeacherResult",
    "audit_frozen_sparse_teacher_learnability",
]
