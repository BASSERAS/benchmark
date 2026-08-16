from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Callable, Mapping

import torch
import torch.nn as nn


REPRESENTATIONS = (
    "log_price",
    "standardized_log_price",
    "standardized_log_return",
    "hybrid",
)
REGIME_NAMES = ("low", "middle", "high")


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite float")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return result


@dataclass(frozen=True)
class ConditionalRepresentationConfig:
    step_indices: tuple[int, ...] = (16, 32, 48, 64)
    primary_step: int | None = None
    representations: tuple[str, ...] = REPRESENTATIONS
    hidden_dim: int = 96
    num_layers: int = 1
    batch_size: int = 128
    max_steps: int = 3000
    eval_every: int = 100
    plateau_patience: int = 10
    minimum_relative_improvement: float = 1e-4
    lr: float = 2e-3
    weight_decay: float = 1e-5
    minimum_scale: float = 1e-3
    minimum_r2_improvement: float = 0.01
    prefix_volatility_window: int = 16
    bootstrap_replicates: int = 2000
    seed: int = 1234

    def __post_init__(self) -> None:
        step_indices = tuple(_require_int("step_index", value) for value in self.step_indices)
        if not step_indices or len(set(step_indices)) != len(step_indices):
            raise ValueError("step_indices must be non-empty and unique")
        if tuple(sorted(step_indices)) != step_indices or any(value < 1 for value in step_indices):
            raise ValueError("step_indices must be strictly increasing positive integers")
        primary_step = (
            max(step_indices)
            if self.primary_step is None
            else _require_int("primary_step", self.primary_step)
        )
        if primary_step not in step_indices:
            raise ValueError("primary_step must be one of step_indices")
        representations = tuple(str(value) for value in self.representations)
        if not representations or len(set(representations)) != len(representations):
            raise ValueError("representations must be non-empty and unique")
        if any(value not in REPRESENTATIONS for value in representations):
            raise ValueError(f"representations must be chosen from {REPRESENTATIONS}")
        for name in (
            "hidden_dim",
            "num_layers",
            "batch_size",
            "max_steps",
            "eval_every",
            "plateau_patience",
            "prefix_volatility_window",
            "bootstrap_replicates",
        ):
            value = _require_int(name, getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
            object.__setattr__(self, name, value)
        if int(self.eval_every) > int(self.max_steps):
            raise ValueError("eval_every cannot exceed max_steps")
        if int(self.bootstrap_replicates) < 2:
            raise ValueError("bootstrap_replicates must be >= 2")
        for name in (
            "minimum_relative_improvement",
            "lr",
            "weight_decay",
            "minimum_scale",
            "minimum_r2_improvement",
        ):
            value = float(getattr(self, name))
            if name == "weight_decay":
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError("weight_decay must be finite and >= 0")
            else:
                value = _positive_float(name, value)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "step_indices", step_indices)
        object.__setattr__(self, "primary_step", primary_step)
        object.__setattr__(self, "representations", representations)
        object.__setattr__(self, "seed", _require_int("seed", self.seed))


@dataclass(frozen=True)
class ConditionalRepresentationResult:
    report: dict[str, object]
    artifacts: dict[str, object]


@dataclass(frozen=True)
class _StepData:
    paths: torch.Tensor
    p: torch.Tensor
    r: torch.Tensor


class _CausalMomentRegressor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        p_dim: int,
        r_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
        )
        self.p_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(p_dim)),
        )
        self.r_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(r_dim)),
        )
        for head in (self.p_head, self.r_head):
            final = head[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("moment head must end in a linear layer")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.gru(inputs)
        features = output[:, -1, :]
        return self.p_head(features), self.r_head(features)


def _load_nested_data(
    tensors: Mapping[str, torch.Tensor],
    *,
    step_indices: tuple[int, ...],
) -> dict[int, _StepData]:
    result = {}
    for step_index in step_indices:
        prefix_key = f"step_{step_index}_prefix_paths"
        p_key = f"step_{step_index}_adjoint_branch_mean"
        r_key = f"step_{step_index}_noise_adjoint_raw_branch_mean"
        missing = [key for key in (prefix_key, p_key, r_key) if key not in tensors]
        if missing:
            raise ValueError(f"nested diagnostic tensors are missing {missing}")
        paths = tensors[prefix_key]
        p = tensors[p_key]
        r = tensors[r_key]
        if paths.ndim != 3 or int(paths.shape[1]) != int(step_index) + 1:
            raise ValueError(f"{prefix_key} has an invalid shape")
        if p.ndim != 2 or r.ndim != 2 or int(p.shape[0]) != int(paths.shape[0]):
            raise ValueError(f"nested targets for step {step_index} have invalid shapes")
        if int(r.shape[0]) != int(paths.shape[0]):
            raise ValueError(f"nested R targets for step {step_index} have an invalid shape")
        result[int(step_index)] = _StepData(paths=paths, p=p, r=r)
    return result


def _input_scales(
    fit_data: Mapping[int, _StepData],
    *,
    minimum_scale: float,
) -> dict[str, torch.Tensor]:
    longest = fit_data[max(fit_data)]
    paths = longest.paths
    returns = paths[:, 1:, :] - paths[:, :-1, :]
    price_scale = torch.sqrt(torch.mean(paths.pow(2), dim=(0, 1))).clamp_min(
        float(minimum_scale)
    )
    return_scale = torch.sqrt(torch.mean(returns.pow(2), dim=(0, 1))).clamp_min(
        float(minimum_scale)
    )
    return {
        "log_price": price_scale,
        "log_return": return_scale,
    }


def _representation_inputs(
    paths: torch.Tensor,
    *,
    representation: str,
    scales: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    price_scale = scales["log_price"].to(device=paths.device, dtype=paths.dtype)
    return_scale = scales["log_return"].to(device=paths.device, dtype=paths.dtype)
    returns = torch.cat(
        (
            torch.zeros_like(paths[:, :1, :]),
            paths[:, 1:, :] - paths[:, :-1, :],
        ),
        dim=1,
    )
    if representation == "log_price":
        return paths
    if representation == "standardized_log_price":
        return paths / price_scale
    if representation == "standardized_log_return":
        return returns / return_scale
    if representation == "hybrid":
        return torch.cat((paths, returns / return_scale), dim=-1)
    raise ValueError(f"unsupported representation {representation}")


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().double().reshape(-1)
    right = right.detach().double().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return None
    return float((torch.dot(left, right) / denominator).item())


def _prediction_report(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    fit_baseline: torch.Tensor,
) -> dict[str, float | None]:
    prediction = prediction.detach().double()
    target = target.detach().double()
    fit_baseline = fit_baseline.detach().double()
    error = prediction - target
    denominator = torch.sum((target - fit_baseline).pow(2)).clamp_min(
        torch.finfo(torch.float64).eps
    )
    return {
        "r2_against_fixed_fit_mean": float(
            (1.0 - torch.sum(error.pow(2)) / denominator).item()
        ),
        "correlation": _correlation(prediction, target),
        "relative_rmse": float(
            (
                torch.sqrt(torch.mean(error.pow(2)))
                / torch.sqrt(torch.mean(target.pow(2))).clamp_min(
                    torch.finfo(torch.float64).eps
                )
            ).item()
        ),
        "prediction_rms": float(torch.sqrt(torch.mean(prediction.pow(2))).item()),
        "target_rms": float(torch.sqrt(torch.mean(target.pow(2))).item()),
    }


def _prefix_volatility(
    paths: torch.Tensor,
    *,
    window: int,
) -> torch.Tensor:
    returns = paths[:, 1:, :] - paths[:, :-1, :]
    recent = returns[:, -min(int(window), int(returns.shape[1])) :, :]
    return torch.sqrt(torch.mean(recent.pow(2), dim=(1, 2)))


def _regime_indices(
    fit_paths: torch.Tensor,
    eval_paths: torch.Tensor,
    *,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float]]:
    fit_volatility = _prefix_volatility(fit_paths, window=window)
    eval_volatility = _prefix_volatility(eval_paths, window=window)
    thresholds = torch.quantile(
        fit_volatility,
        torch.tensor(
            (1.0 / 3.0, 2.0 / 3.0),
            device=fit_volatility.device,
            dtype=fit_volatility.dtype,
        ),
    )
    fit_regimes = torch.bucketize(fit_volatility, thresholds)
    eval_regimes = torch.bucketize(eval_volatility, thresholds)
    return (
        fit_regimes,
        eval_regimes,
        (float(thresholds[0].item()), float(thresholds[1].item())),
    )


def _state_dict_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _gradient_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach().pow(2)).item())
    return total**0.5


def _evaluate(
    *,
    network: _CausalMomentRegressor,
    fit_data: Mapping[int, _StepData],
    heldout_data: Mapping[int, _StepData],
    representation: str,
    input_scales: Mapping[str, torch.Tensor],
    p_scales: Mapping[int, torch.Tensor],
    r_scales: Mapping[int, torch.Tensor],
    prefix_volatility_window: int,
) -> tuple[dict[str, object], dict[str, dict[int, torch.Tensor]]]:
    predictions: dict[str, dict[int, torch.Tensor]] = {
        "fit_p": {},
        "fit_r": {},
        "heldout_p": {},
        "heldout_r": {},
    }
    normalized_fit_loss = 0.0
    step_reports: dict[str, object] = {}
    global_sums = {
        "p_error": 0.0,
        "p_baseline": 0.0,
        "r_error": 0.0,
        "r_baseline": 0.0,
    }
    with torch.no_grad():
        for step_index in sorted(fit_data):
            fit = fit_data[step_index]
            heldout = heldout_data[step_index]
            fit_p_prediction, fit_r_prediction = network(
                _representation_inputs(
                    fit.paths,
                    representation=representation,
                    scales=input_scales,
                )
            )
            heldout_p_prediction, heldout_r_prediction = network(
                _representation_inputs(
                    heldout.paths,
                    representation=representation,
                    scales=input_scales,
                )
            )
            predictions["fit_p"][step_index] = fit_p_prediction.detach().cpu()
            predictions["fit_r"][step_index] = fit_r_prediction.detach().cpu()
            predictions["heldout_p"][step_index] = heldout_p_prediction.detach().cpu()
            predictions["heldout_r"][step_index] = heldout_r_prediction.detach().cpu()
            normalized_fit_loss += float(
                (
                    torch.mean(
                        ((fit_p_prediction - fit.p) / p_scales[step_index]).pow(2)
                    )
                    + torch.mean(
                        ((fit_r_prediction - fit.r) / r_scales[step_index]).pow(2)
                    )
                ).item()
            )
            p_baseline = fit.p.mean(dim=0, keepdim=True)
            r_baseline = fit.r.mean(dim=0, keepdim=True)
            fit_regimes, heldout_regimes, thresholds = _regime_indices(
                fit.paths,
                heldout.paths,
                window=int(prefix_volatility_window),
            )
            regimes = {}
            for regime_index, regime_name in enumerate(REGIME_NAMES):
                fit_mask = fit_regimes == regime_index
                heldout_mask = heldout_regimes == regime_index
                if not bool(fit_mask.any().item()) or not bool(heldout_mask.any().item()):
                    regimes[regime_name] = {
                        "fit_count": int(fit_mask.sum().item()),
                        "heldout_count": int(heldout_mask.sum().item()),
                        "p": None,
                        "r": None,
                    }
                    continue
                regime_p_baseline = fit.p[fit_mask].mean(dim=0, keepdim=True)
                regime_r_baseline = fit.r[fit_mask].mean(dim=0, keepdim=True)
                regimes[regime_name] = {
                    "fit_count": int(fit_mask.sum().item()),
                    "heldout_count": int(heldout_mask.sum().item()),
                    "p": _prediction_report(
                        heldout_p_prediction[heldout_mask],
                        heldout.p[heldout_mask],
                        fit_baseline=regime_p_baseline,
                    ),
                    "r": _prediction_report(
                        heldout_r_prediction[heldout_mask],
                        heldout.r[heldout_mask],
                        fit_baseline=regime_r_baseline,
                    ),
                }
            p_error = torch.sum((heldout_p_prediction - heldout.p).pow(2))
            p_baseline_error = torch.sum((heldout.p - p_baseline).pow(2))
            r_error = torch.sum((heldout_r_prediction - heldout.r).pow(2))
            r_baseline_error = torch.sum((heldout.r - r_baseline).pow(2))
            global_sums["p_error"] += float(p_error.item())
            global_sums["p_baseline"] += float(p_baseline_error.item())
            global_sums["r_error"] += float(r_error.item())
            global_sums["r_baseline"] += float(r_baseline_error.item())
            step_reports[str(step_index)] = {
                "prefix_length": int(step_index) + 1,
                "regime_thresholds": list(thresholds),
                "p": _prediction_report(
                    heldout_p_prediction,
                    heldout.p,
                    fit_baseline=p_baseline,
                ),
                "r": _prediction_report(
                    heldout_r_prediction,
                    heldout.r,
                    fit_baseline=r_baseline,
                ),
                "regimes": regimes,
            }
    eps = torch.finfo(torch.float64).eps
    report: dict[str, object] = {
        "normalized_fit_loss": normalized_fit_loss / float(len(fit_data)),
        "global": {
            "p_r2_against_fixed_timewise_fit_mean": 1.0
            - global_sums["p_error"] / max(global_sums["p_baseline"], eps),
            "r_r2_against_fixed_timewise_fit_mean": 1.0
            - global_sums["r_error"] / max(global_sums["r_baseline"], eps),
        },
        "steps": step_reports,
    }
    return report, predictions


def _train_representation(
    *,
    representation: str,
    fit_data: Mapping[int, _StepData],
    heldout_data: Mapping[int, _StepData],
    input_scales: Mapping[str, torch.Tensor],
    config: ConditionalRepresentationConfig,
    device: torch.device,
    dtype: torch.dtype,
    log_callback: Callable[[str, dict[str, object]], None] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    input_dim = int(next(iter(fit_data.values())).paths.shape[-1])
    if representation == "hybrid":
        input_dim *= 2
    p_dim = int(next(iter(fit_data.values())).p.shape[-1])
    r_dim = int(next(iter(fit_data.values())).r.shape[-1])
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(config.seed) + 1000)
        network = _CausalMomentRegressor(
            input_dim=input_dim,
            p_dim=p_dim,
            r_dim=r_dim,
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
        ).to(device=device, dtype=dtype)
    fit_device = {
        step: _StepData(
            paths=data.paths.to(device=device, dtype=dtype),
            p=data.p.to(device=device, dtype=dtype),
            r=data.r.to(device=device, dtype=dtype),
        )
        for step, data in fit_data.items()
    }
    heldout_device = {
        step: _StepData(
            paths=data.paths.to(device=device, dtype=dtype),
            p=data.p.to(device=device, dtype=dtype),
            r=data.r.to(device=device, dtype=dtype),
        )
        for step, data in heldout_data.items()
    }
    p_scales = {
        step: torch.sqrt(torch.mean(data.p.pow(2), dim=0, keepdim=True)).clamp_min(
            float(config.minimum_scale)
        )
        for step, data in fit_device.items()
    }
    r_scales = {
        step: torch.sqrt(torch.mean(data.r.pow(2), dim=0, keepdim=True)).clamp_min(
            float(config.minimum_scale)
        )
        for step, data in fit_device.items()
    }
    parameters = list(network.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed) + 2000)
    step_indices = tuple(sorted(fit_device))

    def evaluate(step: int) -> tuple[dict[str, object], dict[str, dict[int, torch.Tensor]]]:
        report, predictions = _evaluate(
            network=network,
            fit_data=fit_device,
            heldout_data=heldout_device,
            representation=representation,
            input_scales=input_scales,
            p_scales=p_scales,
            r_scales=r_scales,
            prefix_volatility_window=int(config.prefix_volatility_window),
        )
        report["step"] = int(step)
        return report, predictions

    initial, _ = evaluate(0)
    trajectory = [initial]
    if log_callback is not None:
        log_callback(representation, dict(initial))
    best_loss = float(initial["normalized_fit_loss"])
    meaningful_best = best_loss
    best_step = 0
    best_state = _state_dict_cpu(network)
    plateau_evaluations = 0
    stopped_early = False
    gradient_norms = []
    for optimization_step in range(1, int(config.max_steps) + 1):
        step_index = step_indices[(optimization_step - 1) % len(step_indices)]
        data = fit_device[step_index]
        indices = torch.randint(
            int(data.paths.shape[0]),
            (min(int(config.batch_size), int(data.paths.shape[0])),),
            generator=generator,
            device="cpu",
        ).to(device)
        paths = data.paths.index_select(0, indices)
        target_p = data.p.index_select(0, indices)
        target_r = data.r.index_select(0, indices)
        prediction_p, prediction_r = network(
            _representation_inputs(
                paths,
                representation=representation,
                scales=input_scales,
            )
        )
        loss = torch.mean(
            ((prediction_p - target_p) / p_scales[step_index]).pow(2)
        ) + torch.mean(((prediction_r - target_r) / r_scales[step_index]).pow(2))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = _gradient_norm(parameters)
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"{representation} produced a non-finite gradient")
        gradient_norms.append(gradient_norm)
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all().item()) for parameter in parameters):
            raise RuntimeError(f"{representation} produced a non-finite parameter")
        if (
            optimization_step % int(config.eval_every) == 0
            or optimization_step == int(config.max_steps)
        ):
            row, predictions = evaluate(optimization_step)
            trajectory.append(row)
            if log_callback is not None:
                log_callback(representation, dict(row))
            current_loss = float(row["normalized_fit_loss"])
            if current_loss < best_loss:
                best_loss = current_loss
                best_step = optimization_step
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
    network.load_state_dict(best_state)
    selected, selected_predictions = evaluate(best_step)
    gradients = torch.tensor(gradient_norms, dtype=torch.float64)
    report: dict[str, object] = {
        "representation": representation,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "initial_evaluation": initial,
        "trajectory": trajectory,
        "fit_selected_evaluation": selected,
        "optimization": {
            "best_step": int(best_step),
            "best_training_loss": best_loss,
            "terminal_step": int(trajectory[-1]["step"]),
            "stopped_early_on_training_plateau": bool(stopped_early),
            "gradient_norm_mean": float(gradients.mean().item()),
            "gradient_norm_q95": float(torch.quantile(gradients, 0.95).item()),
            "gradient_norm_max": float(gradients.max().item()),
        },
    }
    artifacts: dict[str, object] = {
        "best_network_state_dict": best_state,
        "fit_selected_predictions": selected_predictions,
        "p_scales": {step: value.detach().cpu() for step, value in p_scales.items()},
        "r_scales": {step: value.detach().cpu() for step, value in r_scales.items()},
    }
    return report, artifacts


def _bootstrap_r2_difference(
    *,
    baseline_squared_error: torch.Tensor,
    candidate_squared_error: torch.Tensor,
    target_baseline_squared_error: torch.Tensor,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if (
        baseline_squared_error.ndim != 1
        or tuple(candidate_squared_error.shape) != tuple(baseline_squared_error.shape)
        or tuple(target_baseline_squared_error.shape) != tuple(baseline_squared_error.shape)
    ):
        raise ValueError("bootstrap inputs must be matching one-dimensional tensors")
    count = int(baseline_squared_error.shape[0])
    if count < 2:
        raise ValueError("bootstrap comparison requires at least two paths")
    baseline_squared_error = baseline_squared_error.double()
    candidate_squared_error = candidate_squared_error.double()
    target_baseline_squared_error = target_baseline_squared_error.double()
    denominator = target_baseline_squared_error.sum().clamp_min(
        torch.finfo(torch.float64).eps
    )
    estimate = float(
        (
            (baseline_squared_error.sum() - candidate_squared_error.sum())
            / denominator
        ).item()
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = []
    remaining = int(replicates)
    while remaining > 0:
        batch = min(remaining, 256)
        indices = torch.randint(
            count,
            (batch, count),
            generator=generator,
            device="cpu",
        )
        base = baseline_squared_error[indices].sum(dim=1)
        candidate = candidate_squared_error[indices].sum(dim=1)
        scale = target_baseline_squared_error[indices].sum(dim=1).clamp_min(
            torch.finfo(torch.float64).eps
        )
        values.append((base - candidate) / scale)
        remaining -= batch
    samples = torch.cat(values)
    quantiles = torch.quantile(samples, torch.tensor((0.025, 0.975), dtype=samples.dtype))
    return {
        "estimate": estimate,
        "ci_lower": float(quantiles[0].item()),
        "ci_upper": float(quantiles[1].item()),
        "replicates": float(replicates),
        "num_paths": float(count),
    }


def compare_conditional_representations(
    *,
    fit_tensors: Mapping[str, torch.Tensor],
    heldout_tensors: Mapping[str, torch.Tensor],
    config: ConditionalRepresentationConfig | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    log_callback: Callable[[str, dict[str, object]], None] | None = None,
) -> ConditionalRepresentationResult:
    """Fit matched causal P/R regressors to independent nested branch means."""

    diagnostic = config or ConditionalRepresentationConfig()
    device = torch.device(device)
    fit_data = _load_nested_data(fit_tensors, step_indices=diagnostic.step_indices)
    heldout_data = _load_nested_data(
        heldout_tensors,
        step_indices=diagnostic.step_indices,
    )
    fit_count = int(next(iter(fit_data.values())).paths.shape[0])
    heldout_count = int(next(iter(heldout_data.values())).paths.shape[0])
    if any(int(data.paths.shape[0]) != fit_count for data in fit_data.values()):
        raise ValueError("all fitting steps must contain the same prefixes")
    if any(int(data.paths.shape[0]) != heldout_count for data in heldout_data.values()):
        raise ValueError("all heldout steps must contain the same prefixes")
    if int(diagnostic.batch_size) > fit_count:
        raise ValueError("batch_size cannot exceed the number of fitting prefixes")
    input_scales_cpu = _input_scales(
        fit_data,
        minimum_scale=float(diagnostic.minimum_scale),
    )
    input_scales = {
        name: value.to(device=device, dtype=dtype)
        for name, value in input_scales_cpu.items()
    }
    arms: dict[str, object] = {}
    arm_artifacts: dict[str, object] = {}
    for representation in diagnostic.representations:
        report, artifacts = _train_representation(
            representation=representation,
            fit_data=fit_data,
            heldout_data=heldout_data,
            input_scales=input_scales,
            config=diagnostic,
            device=device,
            dtype=dtype,
            log_callback=log_callback,
        )
        arms[representation] = report
        arm_artifacts[representation] = artifacts

    baseline_name = "log_price"
    comparisons: dict[str, object] = {}
    if baseline_name in arm_artifacts:
        baseline_predictions = arm_artifacts[baseline_name]["fit_selected_predictions"][
            "heldout_r"
        ]
        primary_step = int(diagnostic.primary_step)
        fit_primary = fit_data[primary_step]
        heldout_primary = heldout_data[primary_step]
        fit_regimes, heldout_regimes, _ = _regime_indices(
            fit_primary.paths,
            heldout_primary.paths,
            window=int(diagnostic.prefix_volatility_window),
        )
        del fit_regimes
        low_mask = heldout_regimes == 0
        for arm_index, representation in enumerate(diagnostic.representations):
            if representation == baseline_name:
                continue
            candidate_predictions = arm_artifacts[representation][
                "fit_selected_predictions"
            ]["heldout_r"]
            global_baseline_error = torch.zeros(heldout_count, dtype=torch.float64)
            global_candidate_error = torch.zeros(heldout_count, dtype=torch.float64)
            global_target_baseline_error = torch.zeros(heldout_count, dtype=torch.float64)
            for step_index in diagnostic.step_indices:
                target = heldout_data[step_index].r.double()
                fit_mean = fit_data[step_index].r.double().mean(dim=0, keepdim=True)
                baseline_prediction = baseline_predictions[step_index].double()
                candidate_prediction = candidate_predictions[step_index].double()
                global_baseline_error += (baseline_prediction - target).pow(2).sum(dim=1)
                global_candidate_error += (candidate_prediction - target).pow(2).sum(dim=1)
                global_target_baseline_error += (target - fit_mean).pow(2).sum(dim=1)
            primary_target = heldout_primary.r.double()
            primary_fit_mean = fit_primary.r.double().mean(dim=0, keepdim=True)
            primary_baseline_error = (
                baseline_predictions[primary_step].double() - primary_target
            ).pow(2).sum(dim=1)
            primary_candidate_error = (
                candidate_predictions[primary_step].double() - primary_target
            ).pow(2).sum(dim=1)
            primary_target_baseline_error = (
                primary_target - primary_fit_mean
            ).pow(2).sum(dim=1)
            comparisons[representation] = {
                "candidate_minus_log_price_r2": {
                    "global": _bootstrap_r2_difference(
                        baseline_squared_error=global_baseline_error,
                        candidate_squared_error=global_candidate_error,
                        target_baseline_squared_error=global_target_baseline_error,
                        replicates=int(diagnostic.bootstrap_replicates),
                        seed=int(diagnostic.seed) + 20_000 + arm_index,
                    ),
                    f"step_{primary_step}": _bootstrap_r2_difference(
                        baseline_squared_error=primary_baseline_error,
                        candidate_squared_error=primary_candidate_error,
                        target_baseline_squared_error=primary_target_baseline_error,
                        replicates=int(diagnostic.bootstrap_replicates),
                        seed=int(diagnostic.seed) + 30_000 + arm_index,
                    ),
                    f"step_{primary_step}_low_regime": (
                        _bootstrap_r2_difference(
                            baseline_squared_error=primary_baseline_error[low_mask],
                            candidate_squared_error=primary_candidate_error[low_mask],
                            target_baseline_squared_error=primary_target_baseline_error[
                                low_mask
                            ],
                            replicates=int(diagnostic.bootstrap_replicates),
                            seed=int(diagnostic.seed) + 40_000 + arm_index,
                        )
                        if int(low_mask.sum().item()) >= 2
                        else None
                    ),
                }
            }

    primary_step = int(diagnostic.primary_step)
    eligible = []
    for name, comparison in comparisons.items():
        changes = comparison["candidate_minus_log_price_r2"]
        global_change = changes["global"]
        low_change = changes[f"step_{primary_step}_low_regime"]
        if (
            float(global_change["ci_lower"]) > 0.0
            and float(global_change["estimate"]) >= float(
                diagnostic.minimum_r2_improvement
            )
            and low_change is not None
            and float(low_change["estimate"]) >= float(
                diagnostic.minimum_r2_improvement
            )
        ):
            eligible.append(name)
    selected_representation = None
    if eligible:
        selected_representation = max(
            eligible,
            key=lambda name: float(
                arms[name]["fit_selected_evaluation"]["global"][
                    "r_r2_against_fixed_timewise_fit_mean"
                ]
            ),
        )
    report: dict[str, object] = {
        "protocol": {
            "fit_and_heldout_prefixes_are_independent": True,
            "all_representations_share_nested_targets_and_seeds": True,
            "generator_checkpoint_is_frozen": True,
            "generated_state_remains_log_price": True,
            "running_cost_and_discrepancy_are_unchanged": True,
            "selection_uses_only_fitting_loss": True,
            "gradient_clipping": False,
            "standardized_log_price_is_a_scaling_control": True,
            "promotion_requires_global_r2_ci_above_zero_and_material_global_and_low_regime_changes": True,
        },
        "config": {
            **diagnostic.__dict__,
            "device": str(device),
            "dtype": str(dtype),
            "fit_prefixes": fit_count,
            "heldout_prefixes": heldout_count,
        },
        "input_scales": {
            name: [float(value) for value in tensor.reshape(-1).tolist()]
            for name, tensor in input_scales_cpu.items()
        },
        "arms": arms,
        "paired_comparisons": comparisons,
        "decision": {
            "eligible_representations": eligible,
            "selected_representation_for_full_training": selected_representation,
            "full_nested_training_justified": selected_representation is not None,
        },
    }
    artifacts: dict[str, object] = {
        "input_scales": input_scales_cpu,
        "arms": arm_artifacts,
    }
    return ConditionalRepresentationResult(report=report, artifacts=artifacts)


__all__ = [
    "ConditionalRepresentationConfig",
    "ConditionalRepresentationResult",
    "REPRESENTATIONS",
    "compare_conditional_representations",
]
