from __future__ import annotations

from dataclasses import dataclass, field
import math
import operator
from typing import Any, Sequence

import torch

from path_dt_experiments.metrics import path_returns


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _validate_paths(paths: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(paths, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if paths.ndim != 3:
        raise ValueError(f"{name} must have shape (B, N + 1, d)")
    if int(paths.shape[0]) < 1:
        raise ValueError(f"{name} must contain at least one path")
    if int(paths.shape[1]) < 2:
        raise ValueError(f"{name} must contain at least two time points")
    if int(paths.shape[2]) < 1:
        raise ValueError(f"{name} must contain at least one state dimension")
    if not torch.is_floating_point(paths):
        paths = paths.float()
    return paths


def _validate_prefix_length(prefix_length: int, *, num_points: int) -> int:
    prefix_length = _require_int("prefix_length", prefix_length)
    if prefix_length < 2:
        raise ValueError("prefix_length must be >= 2")
    if prefix_length >= int(num_points):
        raise ValueError("prefix_length must be smaller than the full path length")
    return prefix_length


def _realized_vol_summary(prefix_returns: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.mean(prefix_returns.pow(2), dim=1).clamp_min(float(eps)))


def prefix_feature_matrix(
    paths: torch.Tensor,
    *,
    prefix_length: int,
    include_returns: bool = True,
    include_terminal_return: bool = True,
    include_realized_volatility: bool = True,
) -> torch.Tensor:
    """
    Build features used to compare one prefix with another.

    ``prefix_length`` is the number of path points in the prefix.  If
    ``prefix_length=8``, the prefix contains points ``0`` through ``7`` and the
    continuation starts from point ``7``.
    """
    paths = _validate_paths(paths, name="paths")
    prefix_length = _validate_prefix_length(prefix_length, num_points=int(paths.shape[1]))
    if not any([include_returns, include_terminal_return, include_realized_volatility]):
        raise ValueError("at least one prefix feature must be enabled")

    prefix = paths[:, :prefix_length, :]
    returns = path_returns(prefix)
    features = []
    if include_returns:
        features.append(returns.reshape(int(paths.shape[0]), -1))
    if include_terminal_return:
        features.append(prefix[:, -1, :] - prefix[:, 0, :])
    if include_realized_volatility:
        features.append(_realized_vol_summary(returns))
    return torch.cat(features, dim=1)


@dataclass(frozen=True)
class ShadowingResult:
    selected_indices: torch.Tensor
    prefix_distances: torch.Tensor
    shadow_mean: torch.Tensor
    lower_50: torch.Tensor
    upper_50: torch.Tensor
    lower_90: torch.Tensor
    upper_90: torch.Tensor
    metrics: dict[str, float | str]
    path_metrics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class ShadowingFeatureConfig:
    """
    Feature blocks used to compare prefixes in path-shadowing diagnostics.

    Inputs are return/increment prefixes.  Each block is standardized on the
    candidate bank and multiplied by ``sqrt(weight)`` before the L2 distance is
    computed.
    """

    last_returns: int = 32
    max_path_points: int = 24
    rolling_windows: tuple[int, ...] = (5, 10, 20)
    acf_lags: tuple[int, ...] = (1, 2, 5, 10)
    return_weight: float = 1.0
    path_weight: float = 0.5
    vol_weight: float = 2.0
    acf_weight: float = 1.0


@dataclass(frozen=True)
class ContextBankShadowingConfig:
    """
    Path-shadowing protocol used by the benchmark presentation.

    The top-k selection is performed over a bank of forecast contexts.  For
    each selected neighbor context, all model continuations attached to that
    context are pooled into the shadow fan.
    """

    top_k: int = 16
    exclude_self: bool = True
    last_returns: int = 32
    max_path_points: int = 24
    rolling_windows: tuple[int, ...] = (5, 10, 20)
    acf_lags: tuple[int, ...] = (1, 2, 5, 10)
    return_weight: float = 1.0
    path_weight: float = 0.5
    vol_weight: float = 2.0
    acf_weight: float = 1.0
    energy_max_scenarios: int = 64


@dataclass(frozen=True)
class ContextBankShadowingResult:
    summary: dict[str, float | str]
    windows: list[dict[str, float | int | str | None]]


def _require_positive_int(name: str, value: object) -> int:
    result = _require_int(name, value)
    if result < 1:
        raise ValueError(f"{name} must be >= 1")
    return result


def _require_finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite(values: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _validate_return_contexts(contexts: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(contexts, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if contexts.ndim != 3:
        raise ValueError(f"{name} must have shape (C, context_length, d)")
    if int(contexts.shape[0]) < 1:
        raise ValueError(f"{name} must contain at least one context")
    if int(contexts.shape[1]) < 1:
        raise ValueError(f"{name} must have positive context length")
    if int(contexts.shape[2]) < 1:
        raise ValueError(f"{name} must contain at least one asset")
    if not torch.is_floating_point(contexts):
        contexts = contexts.to(dtype=torch.float64)
    else:
        contexts = contexts.to(dtype=torch.float64)
    return _finite(contexts)


def _validate_generated_scenarios(
    generated_scenarios: torch.Tensor,
    *,
    n_contexts: int,
    horizon: int,
    n_assets: int,
) -> torch.Tensor:
    if not isinstance(generated_scenarios, torch.Tensor):
        raise ValueError("generated_scenarios must be a torch.Tensor")
    if generated_scenarios.ndim != 4:
        raise ValueError("generated_scenarios must have shape (C, S, horizon, d)")
    if int(generated_scenarios.shape[0]) != int(n_contexts):
        raise ValueError("generated_scenarios context dimension must match contexts")
    if int(generated_scenarios.shape[1]) < 1:
        raise ValueError("generated_scenarios must contain at least one scenario per context")
    if int(generated_scenarios.shape[2]) != int(horizon):
        raise ValueError("generated_scenarios horizon dimension must match realized_futures")
    if int(generated_scenarios.shape[3]) != int(n_assets):
        raise ValueError("generated_scenarios asset dimension must match contexts")
    if not torch.is_floating_point(generated_scenarios):
        generated_scenarios = generated_scenarios.to(dtype=torch.float64)
    else:
        generated_scenarios = generated_scenarios.to(dtype=torch.float64)
    return _finite(generated_scenarios)


def _downsample_time(values: torch.Tensor, max_points: int) -> torch.Tensor:
    max_points = _require_positive_int("max_path_points", max_points)
    if int(values.shape[1]) <= max_points:
        return values
    indices = torch.linspace(
        0,
        int(values.shape[1]) - 1,
        max_points,
        device=values.device,
        dtype=torch.float64,
    ).round().long().unique(sorted=True)
    return values.index_select(1, indices.to(device=values.device))


def _flatten_context_block(values: torch.Tensor) -> torch.Tensor:
    return values.reshape(int(values.shape[0]), -1)


def _standardize_block(values: torch.Tensor, *, weight: float, eps: float = 1e-12) -> torch.Tensor:
    weight = _require_finite_float("weight", weight)
    if values.numel() == 0 or weight <= 0.0:
        return values.new_empty((int(values.shape[0]), 0))
    block = _finite(values)
    mean = block.mean(dim=0, keepdim=True)
    std = block.std(dim=0, keepdim=True, unbiased=False)
    std = torch.where(std > float(eps), std, torch.ones_like(std))
    return (weight**0.5) * ((block - mean) / std)


def _validate_shadowing_feature_config(config: ShadowingFeatureConfig) -> ShadowingFeatureConfig:
    last_returns = _require_positive_int("last_returns", config.last_returns)
    max_path_points = _require_positive_int("max_path_points", config.max_path_points)
    rolling_windows = tuple(_require_positive_int("rolling_windows", value) for value in config.rolling_windows)
    acf_lags = tuple(_require_positive_int("acf_lags", value) for value in config.acf_lags)
    return_weight = _require_finite_float("return_weight", config.return_weight)
    path_weight = _require_finite_float("path_weight", config.path_weight)
    vol_weight = _require_finite_float("vol_weight", config.vol_weight)
    acf_weight = _require_finite_float("acf_weight", config.acf_weight)
    if min(return_weight, path_weight, vol_weight, acf_weight) < 0.0:
        raise ValueError("feature weights must be >= 0")
    return ShadowingFeatureConfig(
        last_returns=last_returns,
        max_path_points=max_path_points,
        rolling_windows=rolling_windows,
        acf_lags=acf_lags,
        return_weight=return_weight,
        path_weight=path_weight,
        vol_weight=vol_weight,
        acf_weight=acf_weight,
    )


def _feature_config_from(config: ShadowingFeatureConfig | ContextBankShadowingConfig | None) -> ShadowingFeatureConfig:
    if config is None:
        return _validate_shadowing_feature_config(ShadowingFeatureConfig())
    return _validate_shadowing_feature_config(
        ShadowingFeatureConfig(
            last_returns=config.last_returns,
            max_path_points=config.max_path_points,
            rolling_windows=tuple(config.rolling_windows),
            acf_lags=tuple(config.acf_lags),
            return_weight=config.return_weight,
            path_weight=config.path_weight,
            vol_weight=config.vol_weight,
            acf_weight=config.acf_weight,
        )
    )


def _feature_weights(config: ShadowingFeatureConfig) -> dict[str, float]:
    return {
        "returns": float(config.return_weight),
        "path": float(config.path_weight),
        "vol": float(config.vol_weight),
        "acf": float(config.acf_weight),
    }


def _raw_return_context_feature_blocks(
    contexts: torch.Tensor,
    *,
    config: ShadowingFeatureConfig,
) -> dict[str, torch.Tensor]:
    context_length = int(contexts.shape[1])
    last = min(int(config.last_returns), context_length)
    cumulative = torch.cumsum(contexts, dim=1)
    return {
        "returns": _flatten_context_block(contexts[:, -last:, :]),
        "path": _flatten_context_block(_downsample_time(cumulative, int(config.max_path_points))),
        "vol": _rolling_vol_features(contexts, windows=config.rolling_windows),
        "acf": _acf_features(contexts, lags=config.acf_lags),
    }


def _weighted_standardized_feature_matrix(
    raw_blocks: dict[str, torch.Tensor],
    *,
    weights: dict[str, float],
    reference_blocks: dict[str, torch.Tensor] | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, int]]:
    blocks = []
    dims = {}
    for name, raw in raw_blocks.items():
        weight = _require_finite_float(f"{name}_weight", weights[name])
        if raw.numel() == 0 or weight <= 0.0:
            block = raw.new_empty((int(raw.shape[0]), 0))
        else:
            reference = raw if reference_blocks is None else reference_blocks[name].to(device=raw.device, dtype=raw.dtype)
            reference = _finite(reference)
            values = _finite(raw)
            mean = reference.mean(dim=0, keepdim=True)
            std = reference.std(dim=0, keepdim=True, unbiased=False)
            std = torch.where(std > float(eps), std, torch.ones_like(std))
            block = (weight**0.5) * ((values - mean) / std)
        dims[name] = int(block.shape[1])
        if int(block.shape[1]) > 0:
            blocks.append(block)
    if not blocks:
        raise ValueError("Path shadowing requires at least one positive feature weight")
    return torch.cat(blocks, dim=1), dims


def _rolling_vol_features(contexts: torch.Tensor, *, windows: Sequence[int], eps: float = 1e-12) -> torch.Tensor:
    context_length = int(contexts.shape[1])
    features = []
    squared = contexts.pow(2)
    for window in sorted({_require_positive_int("rolling_windows", value) for value in windows}):
        if window > context_length:
            continue
        rolled = torch.stack(
            [
                torch.sqrt(squared[:, start : start + window, :].mean(dim=1) + float(eps))
                for start in range(context_length - window + 1)
            ],
            dim=1,
        )
        features.extend([rolled[:, -1, :], rolled.mean(dim=1), rolled.std(dim=1, unbiased=False)])
    if not features:
        return contexts.new_empty((int(contexts.shape[0]), 0))
    return torch.cat(features, dim=1).reshape(int(contexts.shape[0]), -1)


def _acf_features(contexts: torch.Tensor, *, lags: Sequence[int], eps: float = 1e-12) -> torch.Tensor:
    context_length = int(contexts.shape[1])
    features = []
    for transformed in (contexts.abs(), contexts.pow(2)):
        for lag in sorted({_require_positive_int("acf_lags", value) for value in lags}):
            if lag >= context_length:
                continue
            left = transformed[:, :-lag, :]
            right = transformed[:, lag:, :]
            left_centered = left - left.mean(dim=1, keepdim=True)
            right_centered = right - right.mean(dim=1, keepdim=True)
            covariance = (left_centered * right_centered).mean(dim=1)
            scale = torch.sqrt(left_centered.pow(2).mean(dim=1) * right_centered.pow(2).mean(dim=1))
            features.append(covariance / scale.clamp_min(float(eps)))
    if not features:
        return contexts.new_empty((int(contexts.shape[0]), 0))
    return torch.cat(features, dim=1).reshape(int(contexts.shape[0]), -1)


def context_bank_feature_matrix(
    contexts: torch.Tensor,
    *,
    config: ContextBankShadowingConfig | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """
    Build the weighted standardized feature matrix used by benchmark shadowing.

    ``contexts`` are return/increment prefixes with shape ``(C, L, d)``.  The
    returned feature matrix is standardized block by block and each block is
    multiplied by ``sqrt(weight)`` before the final L2 distance is computed.
    """
    cfg = _validate_context_bank_config(config or ContextBankShadowingConfig())
    feature_cfg = _feature_config_from(cfg)
    contexts = _validate_return_contexts(contexts, name="contexts")
    raw_blocks = _raw_return_context_feature_blocks(contexts, config=feature_cfg)
    return _weighted_standardized_feature_matrix(raw_blocks, weights=_feature_weights(feature_cfg))


def path_shadowing_feature_matrix(
    paths: torch.Tensor,
    *,
    prefix_length: int,
    config: ShadowingFeatureConfig | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """
    Build rich weighted prefix features for unconditional path shadowing.

    ``paths`` are full paths with shape ``(B, N + 1, d)``.  The prefix is first
    converted to returns, then represented by recent returns, cumulative prefix
    shape, rolling realized volatility, and ACF blocks.
    """
    paths = _validate_paths(paths, name="paths")
    prefix_length = _validate_prefix_length(prefix_length, num_points=int(paths.shape[1]))
    cfg = _feature_config_from(config)
    contexts = path_returns(paths[:, :prefix_length, :]).to(dtype=torch.float64)
    raw_blocks = _raw_return_context_feature_blocks(contexts, config=cfg)
    return _weighted_standardized_feature_matrix(raw_blocks, weights=_feature_weights(cfg))


def _path_shadowing_standardized_features(
    *,
    real_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    prefix_length: int,
    feature_config: ShadowingFeatureConfig | None,
    include_returns: bool,
    include_terminal_return: bool,
    include_realized_volatility: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if feature_config is None and not all([include_returns, include_terminal_return, include_realized_volatility]):
        generated_features = prefix_feature_matrix(
            generated_paths,
            prefix_length=prefix_length,
            include_returns=include_returns,
            include_terminal_return=include_terminal_return,
            include_realized_volatility=include_realized_volatility,
        )
        real_features = prefix_feature_matrix(
            real_paths,
            prefix_length=prefix_length,
            include_returns=include_returns,
            include_terminal_return=include_terminal_return,
            include_realized_volatility=include_realized_volatility,
        )
        mean = generated_features.mean(dim=0, keepdim=True)
        std = generated_features.std(dim=0, keepdim=True, unbiased=False).clamp_min(float(eps))
        generated_standard = (generated_features - mean) / std
        real_standard = (real_features - mean) / std
        dims = {"returns": int(generated_standard.shape[1]), "path": 0, "vol": 0, "acf": 0}
        return real_standard, generated_standard, dims

    cfg = _feature_config_from(feature_config)
    generated_contexts = path_returns(generated_paths[:, :prefix_length, :]).to(dtype=torch.float64)
    real_contexts = path_returns(real_paths[:, :prefix_length, :]).to(
        device=generated_contexts.device,
        dtype=generated_contexts.dtype,
    )
    generated_raw = _raw_return_context_feature_blocks(generated_contexts, config=cfg)
    real_raw = _raw_return_context_feature_blocks(real_contexts, config=cfg)
    weights = _feature_weights(cfg)
    generated_standard, dims = _weighted_standardized_feature_matrix(
        generated_raw,
        weights=weights,
        eps=eps,
    )
    real_standard, _ = _weighted_standardized_feature_matrix(
        real_raw,
        weights=weights,
        reference_blocks=generated_raw,
        eps=eps,
    )
    return real_standard, generated_standard, dims


def _exact_topk(
    *,
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    top_k: int,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = _require_positive_int("batch_size", batch_size)
    selected_parts = []
    distance_parts = []
    for start in range(0, int(real_features.shape[0]), batch_size):
        stop = min(start + batch_size, int(real_features.shape[0]))
        distances = torch.cdist(real_features[start:stop], generated_features, p=2.0)
        prefix_distances, selected_indices = torch.topk(distances, k=top_k, dim=1, largest=False, sorted=True)
        selected_parts.append(selected_indices.cpu())
        distance_parts.append(prefix_distances.cpu())
    return torch.cat(selected_parts, dim=0), torch.cat(distance_parts, dim=0)


def _faiss_topk(
    *,
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        import faiss  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("FAISS search requested but faiss is not installed. Install the optional faiss-cpu package.") from exc

    generated_np = generated_features.detach().cpu().to(dtype=torch.float32).contiguous().numpy()
    real_np = real_features.detach().cpu().to(dtype=torch.float32).contiguous().numpy()
    index = faiss.IndexFlatL2(int(generated_np.shape[1]))
    index.add(generated_np)
    squared_distances, indices = index.search(real_np, int(top_k))
    distances = torch.from_numpy(squared_distances).clamp_min(0.0).sqrt().to(dtype=real_features.dtype)
    return torch.from_numpy(indices).long(), distances


def _resolve_search_backend(search_backend: str) -> str:
    backend = str(search_backend).lower()
    if backend not in {"auto", "exact", "faiss"}:
        raise ValueError("search_backend must be one of {'auto', 'exact', 'faiss'}")
    if backend != "auto":
        return backend
    try:
        import faiss  # noqa: F401  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return "exact"
    return "faiss"


def _search_topk(
    *,
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    top_k: int,
    search_backend: str,
    exact_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    resolved_backend = _resolve_search_backend(search_backend)
    if resolved_backend == "faiss":
        selected_indices, prefix_distances = _faiss_topk(
            real_features=real_features,
            generated_features=generated_features,
            top_k=top_k,
        )
    else:
        selected_indices, prefix_distances = _exact_topk(
            real_features=real_features,
            generated_features=generated_features,
            top_k=top_k,
            batch_size=exact_batch_size,
        )
    return selected_indices, prefix_distances, resolved_backend


def _build_path_shadowing_result(
    *,
    real_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    selected_indices: torch.Tensor,
    prefix_distances: torch.Tensor,
    prefix_length: int,
    horizon_steps: int,
    block_dims: dict[str, int],
    search_backend: str,
    quantile_levels: Sequence[float],
) -> ShadowingResult:
    levels = tuple(float(level) for level in quantile_levels)
    continuation_start = int(prefix_length) - 1
    continuation_stop = int(prefix_length) + int(horizon_steps)
    selected_on_device = selected_indices.to(device=generated_paths.device)
    raw_shadows = generated_paths[selected_on_device, continuation_start:continuation_stop, :]
    raw_real_continuation = real_paths[:, continuation_start:continuation_stop, :]

    # A shadow continuation is conditional on the observed prefix endpoint.  Its
    # own generated endpoint is only a nearby match, not a forecast draw.  Rebase
    # every continuation to the observed endpoint for absolute fan-chart output,
    # and score only the genuinely future relative values and increments.
    shadow_relative = raw_shadows - raw_shadows[:, :, :1, :]
    real_relative = raw_real_continuation - raw_real_continuation[:, :1, :]
    observed_anchor = raw_real_continuation[:, None, :1, :]
    shadows = observed_anchor + shadow_relative
    shadow_mean = shadows.mean(dim=1)
    quantiles = torch.quantile(
        shadows,
        torch.tensor(levels, device=shadows.device, dtype=shadows.dtype),
        dim=1,
    )
    lower_90, lower_50, upper_50, upper_90 = quantiles[0], quantiles[1], quantiles[2], quantiles[3]

    cumulative_return_samples = shadow_relative[:, :, 1:, :]
    cumulative_return_realized = real_relative[:, 1:, :]
    increment_samples = raw_shadows[:, :, 1:, :] - raw_shadows[:, :, :-1, :]
    increment_realized = raw_real_continuation[:, 1:, :] - raw_real_continuation[:, :-1, :]
    realized_volatility_samples = torch.sqrt(torch.sum(increment_samples.pow(2), dim=2).clamp_min(0.0))
    realized_volatility_realized = torch.sqrt(torch.sum(increment_realized.pow(2), dim=1).clamp_min(0.0))

    path_metrics: dict[str, torch.Tensor] = {}
    path_metrics.update(
        _forecast_metrics_per_path(
            cumulative_return_samples,
            cumulative_return_realized,
            prefix="cumulative_return",
        )
    )
    path_metrics.update(_forecast_metrics_per_path(increment_samples, increment_realized, prefix="increment"))
    path_metrics.update(
        _forecast_metrics_per_path(
            realized_volatility_samples,
            realized_volatility_realized,
            prefix="realized_volatility",
        )
    )
    terminal_errors = cumulative_return_samples[:, :, -1, :].mean(dim=1) - cumulative_return_realized[:, -1, :]
    path_metrics["terminal_return_mean_rmse"] = torch.sqrt(torch.mean(terminal_errors.pow(2), dim=1))
    aggregate_path_metrics = _mean_path_metric_values(path_metrics)
    unique_selected = torch.unique(selected_indices)
    top_k = int(selected_indices.shape[1])
    metrics: dict[str, float | str] = {
        "top_k": float(top_k),
        "prefix_length": float(prefix_length),
        "future_horizon": float(horizon_steps),
        "continuation_points": float(horizon_steps),
        "fan_points": float(horizon_steps + 1),
        "endpoint_aligned": 1.0,
        "endpoint_scored": 0.0,
        "search_backend": str(search_backend),
        "feature_dim": float(sum(block_dims.values())),
        "return_feature_dim": float(block_dims["returns"]),
        "path_feature_dim": float(block_dims["path"]),
        "vol_feature_dim": float(block_dims["vol"]),
        "acf_feature_dim": float(block_dims["acf"]),
        "prefix_distance_mean": float(prefix_distances.mean().item()),
        "prefix_distance_q50": float(torch.quantile(prefix_distances.reshape(-1), 0.5).item()),
        "prefix_distance_q95": float(torch.quantile(prefix_distances.reshape(-1), 0.95).item()),
        "unique_candidate_fraction": float(int(unique_selected.numel()) / int(generated_paths.shape[0])),
    }
    metrics.update(aggregate_path_metrics)
    # Compatibility aliases now refer to endpoint-aligned cumulative-return
    # forecasts and exclude the observed prefix endpoint from scoring.
    metrics.update(
        {
            "continuation_mean_rmse": aggregate_path_metrics["cumulative_return_mean_rmse"],
            "terminal_mean_rmse": aggregate_path_metrics["terminal_return_mean_rmse"],
            "coverage_50": aggregate_path_metrics["cumulative_return_coverage_50"],
            "coverage_90": aggregate_path_metrics["cumulative_return_coverage_90"],
            "band_width_50": aggregate_path_metrics["cumulative_return_band_width_50"],
            "band_width_90": aggregate_path_metrics["cumulative_return_band_width_90"],
        }
    )
    return ShadowingResult(
        selected_indices=selected_indices,
        prefix_distances=prefix_distances,
        shadow_mean=shadow_mean,
        lower_50=lower_50,
        upper_50=upper_50,
        lower_90=lower_90,
        upper_90=upper_90,
        metrics=metrics,
        path_metrics={name: values.detach().cpu() for name, values in path_metrics.items()},
    )


def _validate_context_bank_config(config: ContextBankShadowingConfig) -> ContextBankShadowingConfig:
    top_k = _require_positive_int("top_k", config.top_k)
    last_returns = _require_positive_int("last_returns", config.last_returns)
    max_path_points = _require_positive_int("max_path_points", config.max_path_points)
    rolling_windows = tuple(_require_positive_int("rolling_windows", value) for value in config.rolling_windows)
    acf_lags = tuple(_require_positive_int("acf_lags", value) for value in config.acf_lags)
    return_weight = _require_finite_float("return_weight", config.return_weight)
    path_weight = _require_finite_float("path_weight", config.path_weight)
    vol_weight = _require_finite_float("vol_weight", config.vol_weight)
    acf_weight = _require_finite_float("acf_weight", config.acf_weight)
    if min(return_weight, path_weight, vol_weight, acf_weight) < 0.0:
        raise ValueError("feature weights must be >= 0")
    energy_max_scenarios = _require_positive_int("energy_max_scenarios", config.energy_max_scenarios)
    return ContextBankShadowingConfig(
        top_k=top_k,
        exclude_self=bool(config.exclude_self),
        last_returns=last_returns,
        max_path_points=max_path_points,
        rolling_windows=rolling_windows,
        acf_lags=acf_lags,
        return_weight=return_weight,
        path_weight=path_weight,
        vol_weight=vol_weight,
        acf_weight=acf_weight,
        energy_max_scenarios=energy_max_scenarios,
    )


def _sample_crps(samples: torch.Tensor, realized: torch.Tensor) -> float:
    values = samples.reshape(int(samples.shape[0]), -1).transpose(0, 1)
    targets = realized.reshape(-1)
    mean_abs = torch.mean(torch.abs(values - targets[:, None]), dim=1)
    sorted_values, _ = torch.sort(values, dim=1)
    scenario_count = int(values.shape[1])
    coeff = (
        2.0
        * torch.arange(1, scenario_count + 1, device=values.device, dtype=values.dtype)
        - float(scenario_count)
        - 1.0
    )
    pairwise_mean = 2.0 * torch.sum(sorted_values * coeff[None, :], dim=1) / float(scenario_count**2)
    return float(torch.mean(mean_abs - 0.5 * pairwise_mean).item())


def _energy_score(samples: torch.Tensor, realized: torch.Tensor, *, max_scenarios: int) -> float:
    used = samples[: min(_require_positive_int("max_scenarios", max_scenarios), int(samples.shape[0]))]
    paths = used.reshape(int(used.shape[0]), -1)
    target = realized.reshape(1, -1)
    d_xy = torch.linalg.vector_norm(paths - target, dim=1)
    pairwise = torch.linalg.vector_norm(paths[:, None, :] - paths[None, :, :], dim=-1)
    return float((d_xy.mean() - 0.5 * pairwise.mean()).item())


def _predictive_mean_mse(samples: torch.Tensor, realized: torch.Tensor) -> float:
    mean_forecast = samples.mean(dim=0)
    return float(torch.mean((mean_forecast - realized).pow(2)).item())


def _coverage_error(samples: torch.Tensor, realized: torch.Tensor, *, alpha: float = 0.10) -> float:
    lower = torch.quantile(samples, float(alpha) / 2.0, dim=0)
    upper = torch.quantile(samples, 1.0 - float(alpha) / 2.0, dim=0)
    empirical_coverage = ((realized >= lower) & (realized <= upper)).to(samples.dtype).mean()
    return float(abs(float(empirical_coverage.item()) - (1.0 - float(alpha))))


def _shadow_sample_metrics(samples: torch.Tensor, realized: torch.Tensor, *, prefix: str, max_scenarios: int) -> dict[str, float]:
    return {
        f"{prefix}_crps": _sample_crps(samples, realized),
        f"{prefix}_energy_score": _energy_score(samples, realized, max_scenarios=max_scenarios),
        f"{prefix}_predictive_mean_mse": _predictive_mean_mse(samples, realized),
        f"{prefix}_coverage_90_error": _coverage_error(samples, realized, alpha=0.10),
    }


def _mean_path_errors(samples: torch.Tensor, realized: torch.Tensor, *, prefix: str) -> dict[str, float]:
    mean_path = samples.mean(dim=0)
    errors = mean_path - realized
    return {
        f"{prefix}_mean_mse": float(torch.mean(errors.pow(2)).item()),
        f"{prefix}_mean_mae": float(torch.mean(errors.abs()).item()),
    }


def _numeric_mean_std(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    values = torch.tensor(
        [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))],
        dtype=torch.float64,
    )
    values = values[torch.isfinite(values)]
    if int(values.numel()) == 0:
        return float("nan"), float("nan")
    return float(values.mean().item()), float(values.std(unbiased=False).item())


def evaluate_context_bank_path_shadowing(
    *,
    contexts: torch.Tensor,
    realized_futures: torch.Tensor,
    generated_scenarios: torch.Tensor,
    config: ContextBankShadowingConfig | None = None,
    model_name: str = "model",
) -> ContextBankShadowingResult:
    """
    Exact context-bank path-shadowing evaluator used by the presentation.

    ``contexts`` are prefix returns/increments with shape ``(C, L, d)``.
    ``realized_futures`` are realized continuations with shape ``(C, H, d)``.
    ``generated_scenarios`` are model continuations aligned with each context,
    with shape ``(C, S, H, d)``.  For each query context, the function selects
    the closest ``top_k`` contexts and pools all generated continuations
    attached to those neighbors.
    """
    cfg = _validate_context_bank_config(config or ContextBankShadowingConfig())
    contexts = _validate_return_contexts(contexts, name="contexts")
    realized_futures = _validate_return_contexts(realized_futures, name="realized_futures").to(
        device=contexts.device,
        dtype=contexts.dtype,
    )
    if int(realized_futures.shape[0]) != int(contexts.shape[0]):
        raise ValueError("realized_futures context dimension must match contexts")
    if int(realized_futures.shape[2]) != int(contexts.shape[2]):
        raise ValueError("realized_futures asset dimension must match contexts")
    generated_scenarios = _validate_generated_scenarios(
        generated_scenarios,
        n_contexts=int(contexts.shape[0]),
        horizon=int(realized_futures.shape[1]),
        n_assets=int(contexts.shape[2]),
    ).to(device=contexts.device, dtype=contexts.dtype)
    features, block_dims = context_bank_feature_matrix(contexts, config=cfg)
    top_k_requested = int(cfg.top_k)
    rows: list[dict[str, Any]] = []
    for context_index in range(int(contexts.shape[0])):
        distances = torch.linalg.vector_norm(features - features[context_index : context_index + 1], dim=1)
        if bool(cfg.exclude_self):
            distances[context_index] = torch.inf
        candidate_indices = torch.nonzero(torch.isfinite(distances), as_tuple=False).reshape(-1)
        row: dict[str, Any] = {
            "model": str(model_name),
            "context_index": int(context_index),
            "candidate_pool": "forecast_context_bank",
            "distance": "weighted_standardized_feature_l2",
            "top_k_requested": top_k_requested,
            "exclude_self": bool(cfg.exclude_self),
            "feature_dim": int(features.shape[1]),
            "return_feature_dim": int(block_dims["returns"]),
            "path_feature_dim": int(block_dims["path"]),
            "vol_feature_dim": int(block_dims["vol"]),
            "acf_feature_dim": int(block_dims["acf"]),
            "n_model_scenarios": int(generated_scenarios.shape[1]),
        }
        if int(candidate_indices.numel()) == 0:
            row.update(
                {
                    "n_shadow_neighbors": 0,
                    "n_shadow_samples": 0,
                    "top1_context_index": None,
                    "prefix_distance_top1": float("nan"),
                    "prefix_distance_mean": float("nan"),
                    "prefix_distance_max": float("nan"),
                }
            )
            rows.append(row)
            continue
        ordered = candidate_indices[torch.argsort(distances[candidate_indices])]
        neighbor_indices = ordered[: min(top_k_requested, int(ordered.numel()))]
        neighbor_distances = distances[neighbor_indices]
        shadow_samples = generated_scenarios.index_select(0, neighbor_indices).reshape(
            -1,
            int(realized_futures.shape[1]),
            int(contexts.shape[2]),
        )
        historical_shadow_samples = realized_futures.index_select(0, neighbor_indices)
        target = realized_futures[context_index]
        row.update(
            {
                "n_shadow_neighbors": int(neighbor_indices.numel()),
                "n_shadow_samples": int(shadow_samples.shape[0]),
                "top1_context_index": int(neighbor_indices[0].item()),
                "prefix_distance_top1": float(neighbor_distances[0].item()),
                "prefix_distance_mean": float(neighbor_distances.mean().item()),
                "prefix_distance_max": float(neighbor_distances.max().item()),
            }
        )
        row.update(_shadow_sample_metrics(shadow_samples, target, prefix="shadow", max_scenarios=cfg.energy_max_scenarios))
        row.update(_mean_path_errors(shadow_samples, target, prefix="shadow"))
        row.update(
            _shadow_sample_metrics(
                historical_shadow_samples,
                target,
                prefix="historical_shadow",
                max_scenarios=cfg.energy_max_scenarios,
            )
        )
        row.update(_mean_path_errors(historical_shadow_samples, target, prefix="historical_shadow"))
        rows.append(row)

    summary: dict[str, float | str] = {
        "model": str(model_name),
        "candidate_pool": "forecast_context_bank",
        "distance": "weighted_standardized_feature_l2",
        "n_contexts": float(len(rows)),
        "n_scored_contexts": float(sum(1 for row in rows if int(row.get("n_shadow_neighbors", 0)) > 0)),
        "top_k_requested": float(top_k_requested),
        "context_length": float(contexts.shape[1]),
        "horizon": float(realized_futures.shape[1]),
        "n_assets": float(contexts.shape[2]),
        "n_model_scenarios": float(generated_scenarios.shape[1]),
        "feature_dim": float(features.shape[1]),
        "return_feature_dim": float(block_dims["returns"]),
        "path_feature_dim": float(block_dims["path"]),
        "vol_feature_dim": float(block_dims["vol"]),
        "acf_feature_dim": float(block_dims["acf"]),
    }
    excluded = {
        "model",
        "context_index",
        "candidate_pool",
        "distance",
        "exclude_self",
        "top1_context_index",
        "top_k_requested",
        "feature_dim",
        "return_feature_dim",
        "path_feature_dim",
        "vol_feature_dim",
        "acf_feature_dim",
        "n_model_scenarios",
    }
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in excluded and isinstance(value, (int, float))
        }
    )
    for key in numeric_keys:
        mean, std = _numeric_mean_std(rows, key)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    return ContextBankShadowingResult(summary=summary, windows=rows)


def select_shadow_paths(
    *,
    real_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    prefix_length: int,
    top_k: int,
    feature_config: ShadowingFeatureConfig | None = None,
    search_backend: str = "auto",
    exact_batch_size: int = 64,
    include_returns: bool = True,
    include_terminal_return: bool = True,
    include_realized_volatility: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select generated paths whose prefixes are closest to each real prefix.

    By default, distances are weighted standardized L2 distances over return,
    cumulative-path, rolling-volatility, and ACF feature blocks.  The
    standardization is fitted on the unconditional generated candidate bank,
    then applied to both generated and real prefixes.
    """
    real_paths = _validate_paths(real_paths, name="real_paths")
    generated_paths = _validate_paths(generated_paths, name="generated_paths").to(
        device=real_paths.device,
        dtype=real_paths.dtype,
    )
    if tuple(real_paths.shape[1:]) != tuple(generated_paths.shape[1:]):
        raise ValueError("real and generated paths must have matching time/state dimensions")
    prefix_length = _validate_prefix_length(prefix_length, num_points=int(real_paths.shape[1]))
    top_k = _require_int("top_k", top_k)
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if top_k > int(generated_paths.shape[0]):
        raise ValueError("top_k cannot exceed the number of generated paths")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    real_standard, generated_standard, _ = _path_shadowing_standardized_features(
        real_paths=real_paths,
        generated_paths=generated_paths,
        prefix_length=prefix_length,
        feature_config=feature_config,
        include_returns=include_returns,
        include_terminal_return=include_terminal_return,
        include_realized_volatility=include_realized_volatility,
        eps=eps,
    )
    selected_indices, prefix_distances, _ = _search_topk(
        real_features=real_standard,
        generated_features=generated_standard,
        top_k=top_k,
        search_backend=search_backend,
        exact_batch_size=exact_batch_size,
    )
    return selected_indices, prefix_distances


def _empirical_crps_per_path(samples: torch.Tensor, realized: torch.Tensor) -> torch.Tensor:
    """Return empirical CRPS averaged over each path's non-scenario axes."""
    if samples.ndim < 3:
        raise ValueError("samples must have shape (B, scenarios, ...)")
    if tuple(samples.shape[:1]) + tuple(samples.shape[2:]) != tuple(realized.shape):
        raise ValueError("realized must match samples with the scenario axis removed")
    scenario_count = int(samples.shape[1])
    flattened = samples.reshape(int(samples.shape[0]), scenario_count, -1)
    targets = realized.reshape(int(realized.shape[0]), -1)
    mean_abs = torch.mean(torch.abs(flattened - targets[:, None, :]), dim=1)
    sorted_values, _ = torch.sort(flattened, dim=1)
    coefficients = (
        2.0
        * torch.arange(1, scenario_count + 1, device=samples.device, dtype=samples.dtype)
        - float(scenario_count)
        - 1.0
    )
    pairwise_abs_mean = (
        2.0 * torch.sum(sorted_values * coefficients[None, :, None], dim=1) / float(scenario_count**2)
    )
    return torch.mean(mean_abs - 0.5 * pairwise_abs_mean, dim=1)


def _forecast_metrics_per_path(
    samples: torch.Tensor,
    realized: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Compute proper scores and interval diagnostics independently per path."""
    if samples.ndim < 3:
        raise ValueError("samples must have shape (B, scenarios, ...)")
    if tuple(samples.shape[:1]) + tuple(samples.shape[2:]) != tuple(realized.shape):
        raise ValueError("realized must match samples with the scenario axis removed")
    flattened_realized = realized.reshape(int(realized.shape[0]), -1)
    mean_forecast = samples.mean(dim=1).reshape(int(realized.shape[0]), -1)
    errors = mean_forecast - flattened_realized
    quantiles = torch.quantile(
        samples,
        torch.tensor((0.05, 0.25, 0.75, 0.95), device=samples.device, dtype=samples.dtype),
        dim=1,
    )
    lower_90, lower_50, upper_50, upper_90 = quantiles[0], quantiles[1], quantiles[2], quantiles[3]
    reduce_dims = tuple(range(1, realized.ndim))
    return {
        f"{prefix}_mean_rmse": torch.sqrt(torch.mean(errors.pow(2), dim=1)),
        f"{prefix}_crps": _empirical_crps_per_path(samples, realized),
        f"{prefix}_coverage_50": ((realized >= lower_50) & (realized <= upper_50))
        .to(realized.dtype)
        .mean(dim=reduce_dims),
        f"{prefix}_coverage_90": ((realized >= lower_90) & (realized <= upper_90))
        .to(realized.dtype)
        .mean(dim=reduce_dims),
        f"{prefix}_band_width_50": (upper_50 - lower_50).mean(dim=reduce_dims),
        f"{prefix}_band_width_90": (upper_90 - lower_90).mean(dim=reduce_dims),
    }


def _mean_path_metric_values(path_metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(values.to(dtype=torch.float64).mean().item()) for name, values in path_metrics.items()}


def evaluate_path_shadowing(
    *,
    real_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    prefix_length: int,
    top_k: int = 32,
    future_horizon: int | None = None,
    feature_config: ShadowingFeatureConfig | None = None,
    search_backend: str = "auto",
    exact_batch_size: int = 64,
    include_returns: bool = True,
    include_terminal_return: bool = True,
    include_realized_volatility: bool = True,
    quantile_levels: Sequence[float] = (0.05, 0.25, 0.75, 0.95),
) -> ShadowingResult:
    """
    Evaluate shadow continuations selected by prefix similarity.

    For each held-out real path, the function selects the ``top_k`` closest
    generated prefixes and treats their continuations as conditional scenarios.
    Continuations are anchored at the observed prefix endpoint.  Metrics exclude
    that observed endpoint and separately score relative cumulative returns,
    future increments, and horizon realized volatility.
    """
    real_paths = _validate_paths(real_paths, name="real_paths")
    generated_paths = _validate_paths(generated_paths, name="generated_paths").to(
        device=real_paths.device,
        dtype=real_paths.dtype,
    )
    if tuple(real_paths.shape[1:]) != tuple(generated_paths.shape[1:]):
        raise ValueError("real and generated paths must have matching time/state dimensions")
    prefix_length = _validate_prefix_length(prefix_length, num_points=int(real_paths.shape[1]))
    levels = tuple(float(level) for level in quantile_levels)
    if levels != (0.05, 0.25, 0.75, 0.95):
        raise ValueError("quantile_levels currently must be (0.05, 0.25, 0.75, 0.95)")
    max_future_horizon = int(real_paths.shape[1]) - int(prefix_length)
    if future_horizon is None:
        horizon_steps = max_future_horizon
    else:
        horizon_steps = _require_positive_int("future_horizon", future_horizon)
        if horizon_steps > max_future_horizon:
            raise ValueError("future_horizon cannot exceed the number of available future increments")

    real_standard, generated_standard, block_dims = _path_shadowing_standardized_features(
        real_paths=real_paths,
        generated_paths=generated_paths,
        prefix_length=prefix_length,
        feature_config=feature_config,
        include_returns=include_returns,
        include_terminal_return=include_terminal_return,
        include_realized_volatility=include_realized_volatility,
        eps=1e-8,
    )
    selected_indices, prefix_distances, resolved_backend = _search_topk(
        real_features=real_standard,
        generated_features=generated_standard,
        top_k=min(int(top_k), int(generated_paths.shape[0])),
        search_backend=search_backend,
        exact_batch_size=exact_batch_size,
    )
    return _build_path_shadowing_result(
        real_paths=real_paths,
        generated_paths=generated_paths,
        selected_indices=selected_indices,
        prefix_distances=prefix_distances,
        prefix_length=prefix_length,
        horizon_steps=horizon_steps,
        block_dims=block_dims,
        search_backend=resolved_backend,
        quantile_levels=levels,
    )


def evaluate_path_shadowing_sweep(
    *,
    real_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    prefix_length: int,
    top_k_values: Sequence[int],
    future_horizon: int | None = None,
    feature_config: ShadowingFeatureConfig | None = None,
    search_backend: str = "auto",
    exact_batch_size: int = 64,
    quantile_levels: Sequence[float] = (0.05, 0.25, 0.75, 0.95),
) -> dict[int, ShadowingResult]:
    """
    Evaluate several top-k values from one shared nearest-neighbor search.

    The search is run once with ``max(top_k_values)``.  Results for smaller
    values use prefixes of that same sorted neighbor list.  Every result uses
    endpoint-aligned, future-only scoring as in :func:`evaluate_path_shadowing`.
    """
    real_paths = _validate_paths(real_paths, name="real_paths")
    generated_paths = _validate_paths(generated_paths, name="generated_paths").to(
        device=real_paths.device,
        dtype=real_paths.dtype,
    )
    if tuple(real_paths.shape[1:]) != tuple(generated_paths.shape[1:]):
        raise ValueError("real and generated paths must have matching time/state dimensions")
    prefix_length = _validate_prefix_length(prefix_length, num_points=int(real_paths.shape[1]))
    levels = tuple(float(level) for level in quantile_levels)
    if levels != (0.05, 0.25, 0.75, 0.95):
        raise ValueError("quantile_levels currently must be (0.05, 0.25, 0.75, 0.95)")
    requested = tuple(sorted({_require_positive_int("top_k_values", value) for value in top_k_values}))
    if len(requested) == 0:
        raise ValueError("top_k_values must be non-empty")
    max_top_k = int(requested[-1])
    if max_top_k > int(generated_paths.shape[0]):
        raise ValueError("top_k_values cannot exceed the number of generated paths")

    max_future_horizon = int(real_paths.shape[1]) - int(prefix_length)
    if future_horizon is None:
        horizon_steps = max_future_horizon
    else:
        horizon_steps = _require_positive_int("future_horizon", future_horizon)
        if horizon_steps > max_future_horizon:
            raise ValueError("future_horizon cannot exceed the number of available future increments")

    real_standard, generated_standard, block_dims = _path_shadowing_standardized_features(
        real_paths=real_paths,
        generated_paths=generated_paths,
        prefix_length=prefix_length,
        feature_config=feature_config,
        include_returns=True,
        include_terminal_return=True,
        include_realized_volatility=True,
        eps=1e-8,
    )
    selected_indices, prefix_distances, resolved_backend = _search_topk(
        real_features=real_standard,
        generated_features=generated_standard,
        top_k=max_top_k,
        search_backend=search_backend,
        exact_batch_size=exact_batch_size,
    )
    results: dict[int, ShadowingResult] = {}
    for top_k in requested:
        results[int(top_k)] = _build_path_shadowing_result(
            real_paths=real_paths,
            generated_paths=generated_paths,
            selected_indices=selected_indices[:, : int(top_k)],
            prefix_distances=prefix_distances[:, : int(top_k)],
            prefix_length=prefix_length,
            horizon_steps=horizon_steps,
            block_dims=block_dims,
            search_backend=resolved_backend,
            quantile_levels=levels,
        )
    return results


def _validate_bootstrap_inputs(
    *,
    num_replicates: int,
    confidence_level: float,
    seed: int,
) -> tuple[int, float, int]:
    num_replicates = _require_positive_int("num_replicates", num_replicates)
    confidence_level = _require_finite_float("confidence_level", confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    seed = _require_int("seed", seed)
    return num_replicates, confidence_level, seed


def _bootstrap_summary(
    values: torch.Tensor,
    *,
    bootstrap_indices: torch.Tensor,
    confidence_level: float,
) -> dict[str, float]:
    values = values.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if int(values.numel()) < 1:
        raise ValueError("path metric values must be non-empty")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("path metric values must be finite")
    bootstrap_means = values[bootstrap_indices].mean(dim=1)
    alpha = 0.5 * (1.0 - float(confidence_level))
    return {
        "estimate": float(values.mean().item()),
        "standard_error": float(bootstrap_means.std(unbiased=False).item()),
        "ci_lower": float(torch.quantile(bootstrap_means, alpha).item()),
        "ci_upper": float(torch.quantile(bootstrap_means, 1.0 - alpha).item()),
        "confidence_level": float(confidence_level),
        "num_paths": float(values.numel()),
        "num_replicates": float(bootstrap_indices.shape[0]),
    }


def bootstrap_shadowing_metrics(
    result: ShadowingResult,
    *,
    num_replicates: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Bootstrap aggregate shadowing metrics over independent held-out paths."""
    num_replicates, confidence_level, seed = _validate_bootstrap_inputs(
        num_replicates=num_replicates,
        confidence_level=confidence_level,
        seed=seed,
    )
    if not result.path_metrics:
        raise ValueError("result.path_metrics must be non-empty")
    path_counts = {int(values.numel()) for values in result.path_metrics.values()}
    if len(path_counts) != 1:
        raise ValueError("all path metrics must contain the same number of held-out paths")
    num_paths = path_counts.pop()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randint(0, num_paths, (num_replicates, num_paths), generator=generator)
    return {
        name: _bootstrap_summary(values, bootstrap_indices=indices, confidence_level=confidence_level)
        for name, values in sorted(result.path_metrics.items())
    }


def paired_bootstrap_shadowing_difference(
    primary: ShadowingResult,
    comparison: ShadowingResult,
    *,
    num_replicates: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """
    Bootstrap paired ``primary - comparison`` differences over held-out paths.

    The two results must have been evaluated on the same held-out paths in the
    same order.  Pairing preserves the common realized-path difficulty and is
    more informative than comparing two independent confidence intervals.
    """
    num_replicates, confidence_level, seed = _validate_bootstrap_inputs(
        num_replicates=num_replicates,
        confidence_level=confidence_level,
        seed=seed,
    )
    primary_keys = set(primary.path_metrics)
    comparison_keys = set(comparison.path_metrics)
    if primary_keys != comparison_keys:
        raise ValueError("paired results must expose identical path metric keys")
    if not primary_keys:
        raise ValueError("paired results must expose non-empty path metrics")
    path_counts = {
        (int(primary.path_metrics[name].numel()), int(comparison.path_metrics[name].numel()))
        for name in primary_keys
    }
    if len(path_counts) != 1:
        raise ValueError("all paired path metrics must have consistent lengths")
    primary_count, comparison_count = path_counts.pop()
    if primary_count != comparison_count:
        raise ValueError("paired results must contain the same number of held-out paths")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randint(
        0,
        primary_count,
        (num_replicates, primary_count),
        generator=generator,
    )
    summaries = {}
    for name in sorted(primary_keys):
        difference = primary.path_metrics[name].detach().cpu() - comparison.path_metrics[name].detach().cpu()
        summaries[name] = _bootstrap_summary(
            difference,
            bootstrap_indices=indices,
            confidence_level=confidence_level,
        )
    return summaries


def select_top_k_by_validation_crps(
    sweep: dict[int, ShadowingResult],
    *,
    metric: str = "cumulative_return_crps",
) -> int:
    """Select the smallest top-k attaining the lowest finite validation CRPS."""
    if not sweep:
        raise ValueError("validation sweep must be non-empty")
    candidates = []
    for top_k, result in sweep.items():
        if metric not in result.metrics:
            raise ValueError(f"validation metric {metric!r} is unavailable for top_k={top_k}")
        value = float(result.metrics[metric])
        if not math.isfinite(value):
            raise ValueError(f"validation metric {metric!r} must be finite for top_k={top_k}")
        candidates.append((value, int(top_k)))
    return min(candidates)[1]
