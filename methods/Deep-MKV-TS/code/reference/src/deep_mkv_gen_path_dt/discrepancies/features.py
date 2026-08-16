from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from deep_mkv_gen_path_dt.config import _require_bool, _require_finite_float, _require_int
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid
from deep_mkv_gen_path_dt.paths import gather_observed_path


class PathFeatureMap(Protocol):
    name: str

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class IdentityPathFeature:
    name: str = "path"

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        if int(paths.shape[1]) != int(grid.num_steps) + 1:
            raise ValueError("paths time dimension must align with grid")
        return paths.reshape(int(paths.shape[0]), -1)


@dataclass(frozen=True)
class ObservedPathFeature:
    name: str = "observed_path"

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        observed = gather_observed_path(paths, grid=grid)
        return observed.reshape(int(paths.shape[0]), -1)


@dataclass(frozen=True)
class TerminalFeature:
    name: str = "terminal"

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        if int(paths.shape[1]) != int(grid.num_steps) + 1:
            raise ValueError("paths time dimension must align with grid")
        return paths[:, -1, :]


@dataclass(frozen=True)
class TerminalReturnFeature:
    name: str = "terminal_return"

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        if int(paths.shape[1]) != int(grid.num_steps) + 1:
            raise ValueError("paths time dimension must align with grid")
        return paths[:, -1, :] - paths[:, 0, :]


@dataclass(frozen=True)
class ReturnsFeature:
    name: str = "returns"

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        if int(paths.shape[1]) != int(grid.num_steps) + 1:
            raise ValueError("paths time dimension must align with grid")
        returns = paths[:, 1:, :] - paths[:, :-1, :]
        return returns.reshape(int(paths.shape[0]), -1)


def _returns(paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
    if paths.ndim != 3:
        raise ValueError("paths must have shape (B, N + 1, d)")
    if int(paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("paths time dimension must align with grid")
    return paths[:, 1:, :] - paths[:, :-1, :]


def _flatten_features(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(int(value.shape[0]), -1)


def _smooth_abs(value: torch.Tensor, *, smoothing_epsilon: float) -> torch.Tensor:
    smoothing = float(smoothing_epsilon)
    if smoothing == 0.0:
        return value.abs()
    return torch.sqrt(value.pow(2) + smoothing * smoothing) - smoothing


def _pooled_standardize(value: torch.Tensor, *, eps: float) -> torch.Tensor:
    centered = value - value.mean()
    scale = torch.sqrt(torch.mean(centered.pow(2)).clamp_min(float(eps) * float(eps)))
    return centered / scale


@dataclass(frozen=True)
class AbsReturnsFeature:
    smoothing_epsilon: float = 0.0
    name: str = "abs_returns"

    def __post_init__(self) -> None:
        smoothing_epsilon = _require_finite_float("smoothing_epsilon", self.smoothing_epsilon)
        if smoothing_epsilon < 0.0:
            raise ValueError("smoothing_epsilon must be >= 0")
        object.__setattr__(self, "smoothing_epsilon", smoothing_epsilon)

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        return _flatten_features(
            _smooth_abs(_returns(paths, grid=grid), smoothing_epsilon=float(self.smoothing_epsilon))
        )


@dataclass(frozen=True)
class SquaredReturnsFeature:
    name: str = "squared_returns"

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        return _flatten_features(_returns(paths, grid=grid).pow(2))


@dataclass(frozen=True)
class ObservedReturnsFeature:
    name: str = "observed_returns"

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        observed = gather_observed_path(paths, grid=grid)
        returns = observed[:, 1:, :] - observed[:, :-1, :]
        return returns.reshape(int(paths.shape[0]), -1)


@dataclass(frozen=True)
class RealizedVolatilityFeature:
    window: int = 5
    name: str = "realized_volatility"
    eps: float = 1e-8
    smoothing_epsilon: float = 0.0

    def __post_init__(self) -> None:
        window = _require_int("window", self.window)
        eps = _require_finite_float("eps", self.eps)
        smoothing_epsilon = _require_finite_float("smoothing_epsilon", self.smoothing_epsilon)
        if window < 1:
            raise ValueError("window must be >= 1")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        if smoothing_epsilon < 0.0:
            raise ValueError("smoothing_epsilon must be >= 0")
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "smoothing_epsilon", smoothing_epsilon)

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        returns = _returns(paths, grid=grid)
        if int(self.window) > int(returns.shape[1]):
            raise ValueError("window must be <= number of returns")
        windows = returns.unfold(dimension=1, size=int(self.window), step=1)
        variance = torch.mean(windows.pow(2), dim=-1)
        if float(self.smoothing_epsilon) == 0.0:
            realized = torch.sqrt(variance.clamp_min(float(self.eps)))
        else:
            realized = torch.sqrt(variance + float(self.smoothing_epsilon) * float(self.smoothing_epsilon))
        return _flatten_features(realized)


@dataclass(frozen=True)
class ACFLagProductFeature:
    lags: Sequence[int] = (1, 2, 5, 10)
    squared: bool = False
    eps: float = 1e-8
    smoothing_epsilon: float = 1e-6
    name: str = "acf_lag_product"

    def __post_init__(self) -> None:
        squared = _require_bool("squared", self.squared)
        lags = tuple(_require_int("lags", value) for value in self.lags)
        eps = _require_finite_float("eps", self.eps)
        smoothing_epsilon = _require_finite_float("smoothing_epsilon", self.smoothing_epsilon)
        if len(lags) == 0:
            raise ValueError("lags must be non-empty")
        if any(value < 1 for value in lags):
            raise ValueError("lags must be >= 1")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        if smoothing_epsilon < 0.0:
            raise ValueError("smoothing_epsilon must be >= 0")
        object.__setattr__(self, "squared", squared)
        object.__setattr__(self, "lags", lags)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "smoothing_epsilon", smoothing_epsilon)

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        returns = _returns(paths, grid=grid)
        base_values = (
            returns.pow(2)
            if bool(self.squared)
            else _smooth_abs(returns, smoothing_epsilon=float(self.smoothing_epsilon))
        )
        base = _pooled_standardize(base_values, eps=float(self.eps))
        features = []
        num_returns = int(base.shape[1])
        for lag in tuple(int(value) for value in self.lags):
            if lag < 1 or lag >= num_returns:
                raise ValueError("lags must satisfy 1 <= lag < number of returns")
            features.append(base[:, :-lag, :] * base[:, lag:, :])
        return _flatten_features(torch.cat(features, dim=1))


@dataclass(frozen=True)
class RollingVolatilityACFLagProductFeature:
    lags: Sequence[int] = (1, 2, 5, 10)
    window: int = 5
    eps: float = 1e-8
    smoothing_epsilon: float = 1e-6
    name: str = "rolling_volatility_acf_lag_product"

    def __post_init__(self) -> None:
        lags = tuple(_require_int("lags", value) for value in self.lags)
        window = _require_int("window", self.window)
        eps = _require_finite_float("eps", self.eps)
        smoothing_epsilon = _require_finite_float("smoothing_epsilon", self.smoothing_epsilon)
        if len(lags) == 0:
            raise ValueError("lags must be non-empty")
        if any(value < 1 for value in lags):
            raise ValueError("lags must be >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        if smoothing_epsilon < 0.0:
            raise ValueError("smoothing_epsilon must be >= 0")
        object.__setattr__(self, "lags", lags)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "smoothing_epsilon", smoothing_epsilon)

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        returns = _returns(paths, grid=grid)
        if int(self.window) > int(returns.shape[1]):
            raise ValueError("window must be <= number of returns")
        windows = returns.unfold(dimension=1, size=int(self.window), step=1)
        variance = torch.mean(windows.pow(2), dim=-1)
        if float(self.smoothing_epsilon) == 0.0:
            series = torch.sqrt(variance.clamp_min(float(self.eps)))
        else:
            series = torch.sqrt(variance + float(self.smoothing_epsilon) * float(self.smoothing_epsilon))
        base = _pooled_standardize(series, eps=float(self.eps))
        features = []
        num_steps = int(base.shape[1])
        for lag in tuple(int(value) for value in self.lags):
            if lag < 1 or lag >= num_steps:
                raise ValueError("lags must satisfy 1 <= lag < number of feature time points")
            features.append(base[:, :-lag, :] * base[:, lag:, :])
        return _flatten_features(torch.cat(features, dim=1))


@dataclass(frozen=True)
class CrossReturnProductFeature:
    mode: str = "raw"
    eps: float = 1e-8
    smoothing_epsilon: float = 0.0
    name: str = "cross_return_product"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise ValueError("mode must be raw, abs, or squared")
        eps = _require_finite_float("eps", self.eps)
        smoothing_epsilon = _require_finite_float("smoothing_epsilon", self.smoothing_epsilon)
        if self.mode not in {"raw", "abs", "squared"}:
            raise ValueError("mode must be raw, abs, or squared")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        if smoothing_epsilon < 0.0:
            raise ValueError("smoothing_epsilon must be >= 0")
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "smoothing_epsilon", smoothing_epsilon)

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        returns = _returns(paths, grid=grid)
        if int(returns.shape[-1]) < 2:
            return returns.new_zeros(int(paths.shape[0]), 0)
        if self.mode == "raw":
            base_values = returns
        elif self.mode == "abs":
            base_values = _smooth_abs(returns, smoothing_epsilon=float(self.smoothing_epsilon))
        else:
            base_values = returns.pow(2)
        base = _pooled_standardize(base_values, eps=float(self.eps))
        products = []
        state_dim = int(base.shape[-1])
        for left in range(state_dim):
            for right in range(left + 1, state_dim):
                products.append(base[..., left] * base[..., right])
        return _flatten_features(torch.stack(products, dim=-1))


@dataclass(frozen=True)
class CompositeFeatureMap:
    features: Sequence[PathFeatureMap]
    name: str = "composite"

    def __post_init__(self) -> None:
        if len(tuple(self.features)) == 0:
            raise ValueError("features must be non-empty")

    def __call__(self, paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> torch.Tensor:
        values = [feature(paths, grid=grid) for feature in tuple(self.features)]
        return torch.cat(values, dim=1)
