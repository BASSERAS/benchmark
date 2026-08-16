from __future__ import annotations

"""Closed-form vector--Jacobian products for path discrepancy features.

The routines in this module intentionally run outside PyTorch autograd.  They
implement the first derivatives of the feature maps used by the production
MMD objective and are used by the analytical path-source backend.
"""

import torch

from deep_mkv_gen_path_dt.discrepancies.features import (
    ACFLagProductFeature,
    AbsReturnsFeature,
    CompositeFeatureMap,
    CrossReturnProductFeature,
    IdentityPathFeature,
    ObservedPathFeature,
    ObservedReturnsFeature,
    RealizedVolatilityFeature,
    ReturnsFeature,
    RollingVolatilityACFLagProductFeature,
    SquaredReturnsFeature,
    TerminalFeature,
    TerminalReturnFeature,
)
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


def _validate_paths(paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> None:
    if paths.ndim != 3:
        raise ValueError("paths must have shape (B, N + 1, d)")
    if int(paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("paths time dimension must align with grid")


def returns_path_vjp(
    paths: torch.Tensor,
    returns_cotangent: torch.Tensor,
) -> torch.Tensor:
    """VJP of ``paths[:, 1:] - paths[:, :-1]``."""

    if paths.ndim != 3 or returns_cotangent.ndim != 3:
        raise ValueError("paths and return cotangent must be rank-three tensors")
    if tuple(returns_cotangent.shape) != (
        int(paths.shape[0]),
        int(paths.shape[1]) - 1,
        int(paths.shape[2]),
    ):
        raise ValueError("return cotangent does not align with paths")
    result = torch.zeros_like(paths)
    result[:, :-1, :].sub_(returns_cotangent)
    result[:, 1:, :].add_(returns_cotangent)
    return result


def _smooth_abs_vjp(
    values: torch.Tensor,
    cotangent: torch.Tensor,
    *,
    smoothing_epsilon: float,
) -> torch.Tensor:
    smoothing = float(smoothing_epsilon)
    if smoothing == 0.0:
        return cotangent * torch.sign(values)
    return cotangent * values / torch.sqrt(values.pow(2) + smoothing * smoothing)


def _pooled_standardize_vjp(
    values: torch.Tensor,
    cotangent: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Exact VJP of the pooled standardization in ``features.py``."""

    centered = values - values.mean()
    raw_scale = torch.sqrt(torch.mean(centered.pow(2)))
    floor = float(eps)
    if float(raw_scale.item()) > floor:
        standardized = centered / raw_scale
        return (
            cotangent
            - cotangent.mean()
            - standardized * torch.mean(cotangent * standardized)
        ) / raw_scale
    return (cotangent - cotangent.mean()) / floor


def _lag_product_base_vjp(
    base: torch.Tensor,
    flattened_cotangent: torch.Tensor,
    *,
    lags: tuple[int, ...],
) -> torch.Tensor:
    batch, steps, dimension = base.shape
    expected_rows = sum(steps - int(lag) for lag in lags)
    if tuple(flattened_cotangent.shape) != (batch, expected_rows * dimension):
        raise ValueError("lag-product cotangent has the wrong shape")
    cotangent = flattened_cotangent.reshape(batch, expected_rows, dimension)
    base_bar = torch.zeros_like(base)
    offset = 0
    for lag in lags:
        width = steps - int(lag)
        block = cotangent[:, offset : offset + width, :]
        base_bar[:, :-int(lag), :].add_(block * base[:, int(lag) :, :])
        base_bar[:, int(lag) :, :].add_(block * base[:, :-int(lag), :])
        offset += width
    return base_bar


def _rolling_volatility(
    returns: torch.Tensor,
    *,
    window: int,
    eps: float,
    smoothing_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    windows = returns.unfold(dimension=1, size=int(window), step=1)
    variance = torch.mean(windows.pow(2), dim=-1)
    smoothing = float(smoothing_epsilon)
    if smoothing == 0.0:
        series = torch.sqrt(variance.clamp_min(float(eps)))
        active = variance > float(eps)
    else:
        series = torch.sqrt(variance + smoothing * smoothing)
        active = torch.ones_like(variance, dtype=torch.bool)
    return series, active


def _rolling_volatility_vjp(
    returns: torch.Tensor,
    series: torch.Tensor,
    active: torch.Tensor,
    series_cotangent: torch.Tensor,
    *,
    window: int,
) -> torch.Tensor:
    output_steps = int(series.shape[1])
    result = torch.zeros_like(returns)
    common = (
        series_cotangent
        * active.to(dtype=returns.dtype)
        / (float(window) * series.clamp_min(torch.finfo(series.dtype).tiny))
    )
    for offset in range(int(window)):
        result[:, offset : offset + output_steps, :].add_(
            common * returns[:, offset : offset + output_steps, :]
        )
    return result


def feature_path_vjp(
    feature_map: object,
    paths: torch.Tensor,
    feature_cotangent: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
) -> torch.Tensor:
    """Return ``D feature_map(paths)^T feature_cotangent`` analytically."""

    paths = paths.detach()
    feature_cotangent = feature_cotangent.detach().to(
        device=paths.device,
        dtype=paths.dtype,
    )
    _validate_paths(paths, grid=grid)
    batch, point_count, dimension = paths.shape

    custom = getattr(feature_map, "analytic_path_vjp", None)
    if callable(custom):
        result = custom(paths, feature_cotangent, grid=grid)
        if tuple(result.shape) != tuple(paths.shape):
            raise ValueError("custom analytical feature VJP returned the wrong shape")
        return result

    if isinstance(feature_map, IdentityPathFeature):
        return feature_cotangent.reshape_as(paths)

    if isinstance(feature_map, ObservedPathFeature):
        indices = grid.observed_point_indices(device=paths.device)
        block = feature_cotangent.reshape(batch, int(indices.numel()), dimension)
        result = torch.zeros_like(paths)
        result.index_add_(1, indices, block)
        return result

    if isinstance(feature_map, TerminalFeature):
        result = torch.zeros_like(paths)
        result[:, -1, :].add_(feature_cotangent.reshape(batch, dimension))
        return result

    if isinstance(feature_map, TerminalReturnFeature):
        block = feature_cotangent.reshape(batch, dimension)
        result = torch.zeros_like(paths)
        result[:, 0, :].sub_(block)
        result[:, -1, :].add_(block)
        return result

    if isinstance(feature_map, ReturnsFeature):
        return returns_path_vjp(
            paths,
            feature_cotangent.reshape(batch, point_count - 1, dimension),
        )

    if isinstance(feature_map, ObservedReturnsFeature):
        indices = grid.observed_point_indices(device=paths.device)
        observed_count = int(indices.numel())
        return_bar = feature_cotangent.reshape(
            batch,
            observed_count - 1,
            dimension,
        )
        observed_bar = torch.zeros(
            batch,
            observed_count,
            dimension,
            device=paths.device,
            dtype=paths.dtype,
        )
        observed_bar[:, :-1, :].sub_(return_bar)
        observed_bar[:, 1:, :].add_(return_bar)
        result = torch.zeros_like(paths)
        result.index_add_(1, indices, observed_bar)
        return result

    returns = paths[:, 1:, :] - paths[:, :-1, :]

    if isinstance(feature_map, AbsReturnsFeature):
        cotangent = feature_cotangent.reshape_as(returns)
        return returns_path_vjp(
            paths,
            _smooth_abs_vjp(
                returns,
                cotangent,
                smoothing_epsilon=float(feature_map.smoothing_epsilon),
            ),
        )

    if isinstance(feature_map, SquaredReturnsFeature):
        cotangent = feature_cotangent.reshape_as(returns)
        return returns_path_vjp(paths, 2.0 * returns * cotangent)

    if isinstance(feature_map, RealizedVolatilityFeature):
        series, active = _rolling_volatility(
            returns,
            window=int(feature_map.window),
            eps=float(feature_map.eps),
            smoothing_epsilon=float(feature_map.smoothing_epsilon),
        )
        series_bar = feature_cotangent.reshape_as(series)
        return returns_path_vjp(
            paths,
            _rolling_volatility_vjp(
                returns,
                series,
                active,
                series_bar,
                window=int(feature_map.window),
            ),
        )

    if isinstance(feature_map, ACFLagProductFeature):
        if bool(feature_map.squared):
            base_values = returns.pow(2)
        else:
            smoothing = float(feature_map.smoothing_epsilon)
            base_values = (
                returns.abs()
                if smoothing == 0.0
                else torch.sqrt(returns.pow(2) + smoothing * smoothing) - smoothing
            )
        base = (base_values - base_values.mean()) / torch.sqrt(
            torch.mean((base_values - base_values.mean()).pow(2)).clamp_min(
                float(feature_map.eps) ** 2
            )
        )
        base_bar = _lag_product_base_vjp(
            base,
            feature_cotangent,
            lags=tuple(int(value) for value in feature_map.lags),
        )
        values_bar = _pooled_standardize_vjp(
            base_values,
            base_bar,
            eps=float(feature_map.eps),
        )
        returns_bar = (
            2.0 * returns * values_bar
            if bool(feature_map.squared)
            else _smooth_abs_vjp(
                returns,
                values_bar,
                smoothing_epsilon=float(feature_map.smoothing_epsilon),
            )
        )
        return returns_path_vjp(paths, returns_bar)

    if isinstance(feature_map, RollingVolatilityACFLagProductFeature):
        series, active = _rolling_volatility(
            returns,
            window=int(feature_map.window),
            eps=float(feature_map.eps),
            smoothing_epsilon=float(feature_map.smoothing_epsilon),
        )
        centered = series - series.mean()
        base = centered / torch.sqrt(
            torch.mean(centered.pow(2)).clamp_min(float(feature_map.eps) ** 2)
        )
        base_bar = _lag_product_base_vjp(
            base,
            feature_cotangent,
            lags=tuple(int(value) for value in feature_map.lags),
        )
        series_bar = _pooled_standardize_vjp(
            series,
            base_bar,
            eps=float(feature_map.eps),
        )
        returns_bar = _rolling_volatility_vjp(
            returns,
            series,
            active,
            series_bar,
            window=int(feature_map.window),
        )
        return returns_path_vjp(paths, returns_bar)

    if isinstance(feature_map, CrossReturnProductFeature):
        if int(dimension) < 2:
            return torch.zeros_like(paths)
        if feature_map.mode == "raw":
            base_values = returns
        elif feature_map.mode == "abs":
            smoothing = float(feature_map.smoothing_epsilon)
            base_values = (
                returns.abs()
                if smoothing == 0.0
                else torch.sqrt(returns.pow(2) + smoothing * smoothing) - smoothing
            )
        else:
            base_values = returns.pow(2)
        centered = base_values - base_values.mean()
        base = centered / torch.sqrt(
            torch.mean(centered.pow(2)).clamp_min(float(feature_map.eps) ** 2)
        )
        pair_count = dimension * (dimension - 1) // 2
        pair_bar = feature_cotangent.reshape(batch, point_count - 1, pair_count)
        base_bar = torch.zeros_like(base)
        pair = 0
        for left in range(dimension):
            for right in range(left + 1, dimension):
                block = pair_bar[..., pair]
                base_bar[..., left].add_(block * base[..., right])
                base_bar[..., right].add_(block * base[..., left])
                pair += 1
        values_bar = _pooled_standardize_vjp(
            base_values,
            base_bar,
            eps=float(feature_map.eps),
        )
        if feature_map.mode == "raw":
            returns_bar = values_bar
        elif feature_map.mode == "abs":
            returns_bar = _smooth_abs_vjp(
                returns,
                values_bar,
                smoothing_epsilon=float(feature_map.smoothing_epsilon),
            )
        else:
            returns_bar = 2.0 * returns * values_bar
        return returns_path_vjp(paths, returns_bar)

    if isinstance(feature_map, CompositeFeatureMap):
        result = torch.zeros_like(paths)
        offset = 0
        for child in tuple(feature_map.features):
            child_value = child(paths, grid=grid)
            width = int(child_value.shape[1])
            result.add_(
                feature_path_vjp(
                    child,
                    paths,
                    feature_cotangent[:, offset : offset + width],
                    grid=grid,
                )
            )
            offset += width
        if offset != int(feature_cotangent.shape[1]):
            raise ValueError("composite feature cotangent has the wrong width")
        return result

    raise TypeError(
        "no analytical path VJP is registered for feature map "
        f"{type(feature_map).__name__}"
    )

