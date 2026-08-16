from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import torch


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class CrossfitRControlVariateConfig:
    folds: int = 5
    neighbors: int = 128
    volatility_windows: tuple[int, ...] = (4, 8, 16, 32)
    minimum_feature_scale: float = 1e-4
    minimum_noise_weight: float = 1e-8

    def __post_init__(self) -> None:
        folds = _require_int("folds", self.folds)
        neighbors = _require_int("neighbors", self.neighbors)
        if folds < 2:
            raise ValueError("folds must be >= 2")
        if neighbors < 1:
            raise ValueError("neighbors must be >= 1")
        windows = tuple(
            _require_int("volatility_windows", value)
            for value in self.volatility_windows
        )
        if not windows or any(value < 1 for value in windows):
            raise ValueError("volatility_windows must contain positive integers")
        if len(set(windows)) != len(windows):
            raise ValueError("volatility_windows must not contain duplicates")
        minimum_feature_scale = float(self.minimum_feature_scale)
        minimum_noise_weight = float(self.minimum_noise_weight)
        if not math.isfinite(minimum_feature_scale) or minimum_feature_scale <= 0.0:
            raise ValueError("minimum_feature_scale must be finite and > 0")
        if not math.isfinite(minimum_noise_weight) or minimum_noise_weight <= 0.0:
            raise ValueError("minimum_noise_weight must be finite and > 0")
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "neighbors", neighbors)
        object.__setattr__(self, "volatility_windows", windows)
        object.__setattr__(self, "minimum_feature_scale", minimum_feature_scale)
        object.__setattr__(self, "minimum_noise_weight", minimum_noise_weight)


@dataclass(frozen=True)
class CrossfitRControlVariateResult:
    fit_baseline: torch.Tensor
    eval_baseline: torch.Tensor
    report: dict[str, object]


def _validate_problem(
    *,
    fit_paths: torch.Tensor,
    fit_pathwise_p: torch.Tensor,
    fit_noise: torch.Tensor,
    eval_paths: torch.Tensor,
    eval_pathwise_p: torch.Tensor,
    eval_noise: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if fit_paths.ndim != 3 or eval_paths.ndim != 3:
        raise ValueError("paths must have shape (B, N + 1, d)")
    if fit_pathwise_p.ndim != 3 or eval_pathwise_p.ndim != 3:
        raise ValueError("pathwise P must have shape (B, N, d)")
    if fit_noise.ndim != 3 or eval_noise.ndim != 3:
        raise ValueError("noise must have shape (B, N, q)")
    fit_count, point_count, state_dim = map(int, fit_paths.shape)
    eval_count = int(eval_paths.shape[0])
    num_steps = point_count - 1
    noise_dim = int(fit_noise.shape[-1])
    expected = (
        ("fit_pathwise_p", fit_pathwise_p, (fit_count, num_steps, state_dim)),
        ("fit_noise", fit_noise, (fit_count, num_steps, noise_dim)),
        ("eval_paths", eval_paths, (eval_count, point_count, state_dim)),
        ("eval_pathwise_p", eval_pathwise_p, (eval_count, num_steps, state_dim)),
        ("eval_noise", eval_noise, (eval_count, num_steps, noise_dim)),
    )
    for name, value, shape in expected:
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if fit_count < 2 or eval_count < 1 or num_steps < 1:
        raise ValueError("control-variate banks and time grid must be non-empty")
    return fit_count, eval_count, num_steps, state_dim, noise_dim


def causal_prefix_features(
    paths: torch.Tensor,
    *,
    step_index: int,
    volatility_windows: tuple[int, ...],
) -> torch.Tensor:
    """Return a fixed model-free feature vector measurable from X_0,...,X_t."""

    if paths.ndim != 3:
        raise ValueError("paths must have shape (B, N + 1, d)")
    step = _require_int("step_index", step_index)
    if not 0 <= step < int(paths.shape[1]) - 1:
        raise ValueError("step_index must identify a non-terminal path point")
    windows = tuple(_require_int("volatility_windows", value) for value in volatility_windows)
    if not windows or any(value < 1 for value in windows):
        raise ValueError("volatility_windows must contain positive integers")

    current = paths[:, step, :]
    values = [current, current - paths[:, 0, :]]
    if step == 0:
        zeros = torch.zeros_like(current)
        values.append(zeros)
        for _ in windows:
            values.extend((zeros, zeros, torch.log(zeros + 1e-4)))
        values.extend((zeros, zeros))
    else:
        returns = paths[:, 1 : step + 1, :] - paths[:, :step, :]
        values.append(returns[:, -1, :])
        for window in windows:
            recent = returns[:, -min(int(window), step) :, :]
            rms = torch.sqrt(torch.mean(recent.pow(2), dim=1) + 1e-8)
            values.extend((torch.mean(recent, dim=1), rms, torch.log(rms + 1e-4)))
        prefix = paths[:, : step + 1, :]
        values.extend(
            (
                torch.max(prefix, dim=1).values - current,
                current - torch.min(prefix, dim=1).values,
            )
        )
    return torch.cat(values, dim=-1)


def _second_moment_ratio(controlled: torch.Tensor, raw: torch.Tensor) -> float:
    numerator = torch.mean(controlled.detach().double().pow(2))
    denominator = torch.mean(raw.detach().double().pow(2))
    return float(
        (numerator / denominator.clamp_min(torch.finfo(torch.float64).eps)).item()
    )


def fit_crossfit_variance_minimizing_r_baseline(
    *,
    fit_paths: torch.Tensor,
    fit_pathwise_p: torch.Tensor,
    fit_noise: torch.Tensor,
    eval_paths: torch.Tensor,
    eval_pathwise_p: torch.Tensor,
    eval_noise: torch.Tensor,
    config: CrossfitRControlVariateConfig | None = None,
) -> CrossfitRControlVariateResult:
    """Estimate c(F_t)=E[P eps^2|F_t]/E[eps^2|F_t] without query leakage.

    Each fit query uses neighbors outside its deterministic fold. Evaluation
    queries use the complete fit bank. For a fixed baseline measurable from the
    prefix, subtracting ``c(F_t) * eps`` leaves the R conditional expectation
    unchanged.
    """

    settings = config or CrossfitRControlVariateConfig()
    fit_count, eval_count, num_steps, state_dim, noise_dim = _validate_problem(
        fit_paths=fit_paths,
        fit_pathwise_p=fit_pathwise_p,
        fit_noise=fit_noise,
        eval_paths=eval_paths,
        eval_pathwise_p=eval_pathwise_p,
        eval_noise=eval_noise,
    )
    if int(settings.folds) > fit_count:
        raise ValueError("folds cannot exceed the fit bank size")
    fold_indices = torch.arange(fit_count, device=fit_paths.device).remainder(
        int(settings.folds)
    )
    smallest_training_fold = min(
        fit_count - int((fold_indices == fold).sum().item())
        for fold in range(int(settings.folds))
    )
    if int(settings.neighbors) > smallest_training_fold:
        raise ValueError(
            "neighbors cannot exceed the smallest other-fold training bank"
        )
    if int(settings.neighbors) > fit_count:
        raise ValueError("neighbors cannot exceed the fit bank size")

    fit_baseline = torch.empty(
        fit_count,
        num_steps,
        state_dim,
        noise_dim,
        device=fit_paths.device,
        dtype=fit_paths.dtype,
    )
    eval_baseline = torch.empty(
        eval_count,
        num_steps,
        state_dim,
        noise_dim,
        device=eval_paths.device,
        dtype=eval_paths.dtype,
    )
    feature_dims = []
    fit_step_ratios = []
    eval_step_ratios = []
    eps = float(settings.minimum_noise_weight)
    with torch.no_grad():
        weighted_p = fit_pathwise_p.unsqueeze(-1) * fit_noise.pow(2).unsqueeze(-2)
        weights = fit_noise.pow(2).unsqueeze(-2).expand(
            -1, -1, state_dim, -1
        )
        for step in range(num_steps):
            fit_features = causal_prefix_features(
                fit_paths,
                step_index=step,
                volatility_windows=settings.volatility_windows,
            )
            eval_features = causal_prefix_features(
                eval_paths,
                step_index=step,
                volatility_windows=settings.volatility_windows,
            )
            mean = fit_features.mean(dim=0, keepdim=True)
            scale = fit_features.std(dim=0, keepdim=True, unbiased=False).clamp_min(
                float(settings.minimum_feature_scale)
            )
            normalized_fit = (fit_features - mean) / scale
            normalized_eval = (eval_features - mean) / scale
            feature_dims.append(int(normalized_fit.shape[-1]))

            fit_distances = torch.cdist(normalized_fit, normalized_fit)
            fit_distances.masked_fill_(
                fold_indices[:, None] == fold_indices[None, :],
                float("inf"),
            )
            fit_neighbors = torch.topk(
                fit_distances,
                k=int(settings.neighbors),
                largest=False,
            ).indices
            eval_neighbors = torch.topk(
                torch.cdist(normalized_eval, normalized_fit),
                k=int(settings.neighbors),
                largest=False,
            ).indices

            fit_numerator = weighted_p[:, step, :, :][fit_neighbors].sum(dim=1)
            fit_denominator = weights[:, step, :, :][fit_neighbors].sum(dim=1)
            eval_numerator = weighted_p[:, step, :, :][eval_neighbors].sum(dim=1)
            eval_denominator = weights[:, step, :, :][eval_neighbors].sum(dim=1)
            fit_step_baseline = fit_numerator / fit_denominator.clamp_min(eps)
            eval_step_baseline = eval_numerator / eval_denominator.clamp_min(eps)
            fit_baseline[:, step, :, :] = fit_step_baseline
            eval_baseline[:, step, :, :] = eval_step_baseline

            fit_raw = fit_pathwise_p[:, step, :].unsqueeze(-1) * fit_noise[
                :, step, :
            ].unsqueeze(-2)
            fit_controlled = (
                fit_pathwise_p[:, step, :].unsqueeze(-1) - fit_step_baseline
            ) * fit_noise[:, step, :].unsqueeze(-2)
            eval_raw = eval_pathwise_p[:, step, :].unsqueeze(-1) * eval_noise[
                :, step, :
            ].unsqueeze(-2)
            eval_controlled = (
                eval_pathwise_p[:, step, :].unsqueeze(-1) - eval_step_baseline
            ) * eval_noise[:, step, :].unsqueeze(-2)
            fit_step_ratios.append(_second_moment_ratio(fit_controlled, fit_raw))
            eval_step_ratios.append(_second_moment_ratio(eval_controlled, eval_raw))

        fit_raw = fit_pathwise_p.unsqueeze(-1) * fit_noise.unsqueeze(-2)
        fit_controlled = (fit_pathwise_p.unsqueeze(-1) - fit_baseline) * fit_noise.unsqueeze(-2)
        eval_raw = eval_pathwise_p.unsqueeze(-1) * eval_noise.unsqueeze(-2)
        eval_controlled = (eval_pathwise_p.unsqueeze(-1) - eval_baseline) * eval_noise.unsqueeze(-2)

    report: dict[str, object] = {
        "method": "crossfit_knn_conditional_weighted_least_squares",
        "formula": "E[P * epsilon^2 | prefix] / E[epsilon^2 | prefix]",
        "fit_queries_exclude_entire_query_fold": True,
        "eval_queries_use_fit_bank_only": True,
        "features_are_prefix_measurable": True,
        "conditional_expectation_is_unchanged": True,
        "config": {
            "folds": int(settings.folds),
            "neighbors": int(settings.neighbors),
            "volatility_windows": list(settings.volatility_windows),
            "minimum_feature_scale": float(settings.minimum_feature_scale),
            "minimum_noise_weight": float(settings.minimum_noise_weight),
        },
        "feature_dimension": int(feature_dims[0]),
        "fit_second_moment_ratio": _second_moment_ratio(fit_controlled, fit_raw),
        "eval_second_moment_ratio": _second_moment_ratio(eval_controlled, eval_raw),
        "fit_step_second_moment_ratios": fit_step_ratios,
        "eval_step_second_moment_ratios": eval_step_ratios,
    }
    return CrossfitRControlVariateResult(
        fit_baseline=fit_baseline.detach(),
        eval_baseline=eval_baseline.detach(),
        report=report,
    )


__all__ = [
    "CrossfitRControlVariateConfig",
    "CrossfitRControlVariateResult",
    "causal_prefix_features",
    "fit_crossfit_variance_minimizing_r_baseline",
]
