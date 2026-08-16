from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from deep_mkv_gen_path_dt.ce import ConditionalExpectationResult, fit_ridge
from deep_mkv_gen_path_dt.discrepancies import (
    PathFunctionalDiscrepancyResult,
    rbf_mmd2,
)
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


def _increasing_positive_ints(
    name: str,
    values: Sequence[int],
) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if (
        not result
        or any(value < 1 for value in result)
        or tuple(sorted(set(result))) != result
    ):
        raise ValueError(f"{name} must be strictly increasing positive integers")
    return result


def _validate_paths(paths: torch.Tensor, *, grid: DiscreteTimeGrid) -> None:
    if paths.ndim != 3 or int(paths.shape[-1]) != 1:
        raise ValueError("Heston-mixture summaries require paths of shape (B, N + 1, 1)")
    if int(paths.shape[1]) != int(grid.num_steps) + 1:
        raise ValueError("paths time dimension must align with grid")
    if int(paths.shape[0]) < 2:
        raise ValueError("Heston-mixture summaries require at least two paths")


def _pathwise_correlation(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    if left.ndim != 2 or tuple(left.shape) != tuple(right.shape):
        raise ValueError("correlation inputs must have matching shape (B, T)")
    left_centered = left - left.mean(dim=1, keepdim=True)
    right_centered = right - right.mean(dim=1, keepdim=True)
    denominator = torch.sqrt(
        left_centered.pow(2).mean(dim=1)
        * right_centered.pow(2).mean(dim=1)
        + float(eps) ** 2
    )
    return (left_centered * right_centered).mean(dim=1) / denominator


def _stable_row_moments(
    values: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, ...]:
    mean = values.mean(dim=1)
    centered = values - mean.unsqueeze(1)
    std = torch.sqrt(centered.pow(2).mean(dim=1) + float(eps) ** 2)
    standardized = centered / std.unsqueeze(1)
    skew = standardized.pow(3).mean(dim=1)
    fourth = standardized.pow(4).mean(dim=1)
    return (
        mean,
        torch.log(std),
        torch.tanh(skew / 5.0),
        torch.log(fourth.clamp_min(float(eps))),
    )


@dataclass(frozen=True)
class HestonMixtureSummaryConfig:
    """Generic price-path observables associated with Heston heterogeneity."""

    quantile_levels: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
    persistence_lags: tuple[int, ...] = (1, 5, 10, 20)
    leverage_lags: tuple[int, ...] = (0, 1, 2, 5, 10, 20)
    block_counts: tuple[int, ...] = (4, 8, 16)
    eps: float = 1e-6

    def __post_init__(self) -> None:
        quantiles = tuple(float(value) for value in self.quantile_levels)
        if (
            not quantiles
            or any(not 0.0 < value < 1.0 for value in quantiles)
            or tuple(sorted(set(quantiles))) != quantiles
        ):
            raise ValueError("quantile_levels must be strictly increasing in (0, 1)")
        persistence = _increasing_positive_ints(
            "persistence_lags",
            self.persistence_lags,
        )
        leverage = tuple(int(value) for value in self.leverage_lags)
        if (
            not leverage
            or any(value < 0 for value in leverage)
            or tuple(sorted(set(leverage))) != leverage
        ):
            raise ValueError("leverage_lags must be strictly increasing nonnegative integers")
        blocks = _increasing_positive_ints("block_counts", self.block_counts)
        eps = float(self.eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        object.__setattr__(self, "quantile_levels", quantiles)
        object.__setattr__(self, "persistence_lags", persistence)
        object.__setattr__(self, "leverage_lags", leverage)
        object.__setattr__(self, "block_counts", blocks)
        object.__setattr__(self, "eps", eps)

    @property
    def feature_names(self) -> tuple[str, ...]:
        marginal = [
            f"{series}_{moment}"
            for series in ("return", "absolute_return", "squared_return")
            for moment in ("mean", "log_std", "bounded_skew", "log_fourth_moment")
        ]
        marginal.extend(
            f"return_quantile_{level:g}" for level in self.quantile_levels
        )
        marginal.extend(("normalized_terminal_return", "normalized_max_drawdown"))
        volatility = [
            f"{series}_persistence_{lag}"
            for series in ("absolute_return", "squared_return")
            for lag in self.persistence_lags
        ]
        volatility.extend(
            f"log_block_volatility_{blocks}_{summary}"
            for blocks in self.block_counts
            for summary in ("mean", "std", "min", "max")
        )
        leverage = [f"leverage_{lag}" for lag in self.leverage_lags]
        return tuple((*marginal, *volatility, *leverage))

    @property
    def group_slices(self) -> dict[str, slice]:
        marginal_dim = 12 + len(self.quantile_levels) + 2
        volatility_dim = 2 * len(self.persistence_lags) + 4 * len(
            self.block_counts
        )
        return {
            "marginal_shape": slice(0, marginal_dim),
            "volatility_structure": slice(
                marginal_dim,
                marginal_dim + volatility_dim,
            ),
            "leverage": slice(
                marginal_dim + volatility_dim,
                len(self.feature_names),
            ),
        }


def heston_mixture_path_summary_features(
    paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    config: HestonMixtureSummaryConfig = HestonMixtureSummaryConfig(),
) -> torch.Tensor:
    """Extract differentiable, label-free path summaries from log-price paths."""

    _validate_paths(paths, grid=grid)
    values = paths[..., 0]
    returns = (values[:, 1:] - values[:, :-1]) / math.sqrt(float(grid.dt))
    num_returns = int(returns.shape[1])
    if max(config.persistence_lags) >= num_returns:
        raise ValueError("persistence_lags must be smaller than the path horizon")
    if max(config.leverage_lags) >= num_returns:
        raise ValueError("leverage_lags must be smaller than the path horizon")
    if max(config.block_counts) > num_returns:
        raise ValueError("block_counts must not exceed the number of returns")

    smooth_abs = torch.sqrt(returns.pow(2) + float(config.eps) ** 2)
    squared = returns.pow(2)
    features: list[torch.Tensor] = []
    for series in (returns, smooth_abs, squared):
        features.extend(_stable_row_moments(series, eps=float(config.eps)))
    features.extend(
        torch.quantile(
            returns,
            returns.new_tensor(config.quantile_levels),
            dim=1,
        ).unbind(dim=0)
    )
    horizon_scale = math.sqrt(float(grid.T))
    features.append((values[:, -1] - values[:, 0]) / horizon_scale)
    running_max = torch.cummax(values, dim=1).values
    features.append((running_max - values).amax(dim=1) / horizon_scale)

    for series in (smooth_abs, squared):
        for lag in config.persistence_lags:
            features.append(
                _pathwise_correlation(
                    series[:, :-lag],
                    series[:, lag:],
                    eps=float(config.eps),
                )
            )
    for block_count in config.block_counts:
        block_length = num_returns // int(block_count)
        blocked = returns[:, : int(block_count) * block_length].reshape(
            int(returns.shape[0]),
            int(block_count),
            block_length,
        )
        log_volatility = torch.log(
            torch.sqrt(
                blocked.pow(2).mean(dim=2) + float(config.eps) ** 2
            )
        )
        features.extend(
            (
                log_volatility.mean(dim=1),
                log_volatility.std(dim=1, unbiased=False),
                log_volatility.amin(dim=1),
                log_volatility.amax(dim=1),
            )
        )
    for lag in config.leverage_lags:
        left = returns if lag == 0 else returns[:, :-lag]
        right = squared if lag == 0 else squared[:, lag:]
        features.append(
            _pathwise_correlation(
                left,
                right,
                eps=float(config.eps),
            )
        )
    result = torch.stack(features, dim=1)
    if int(result.shape[1]) != len(config.feature_names):
        raise RuntimeError("Heston-mixture feature construction is inconsistent")
    if not torch.isfinite(result).all():
        raise RuntimeError("Heston-mixture feature construction produced nonfinite values")
    return result


@dataclass(frozen=True)
class FixedTargetHestonMixtureSummaryMMD:
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    config: HestonMixtureSummaryConfig = HestonMixtureSummaryConfig()
    bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    standardized_clip: float = 6.0
    joint_weight: float = 0.5
    name: str = "heston_mixture_path_summary"

    def __post_init__(self) -> None:
        if self.target_mean.ndim != 2 or int(self.target_mean.shape[0]) != 1:
            raise ValueError("target_mean must have shape (1, feature_dim)")
        if tuple(self.target_scale.shape) != tuple(self.target_mean.shape):
            raise ValueError("target_scale must match target_mean")
        if int(self.target_mean.shape[1]) != len(self.config.feature_names):
            raise ValueError("target statistics do not match the configured feature basis")
        if not torch.isfinite(self.target_mean).all() or not torch.isfinite(
            self.target_scale
        ).all():
            raise ValueError("target statistics must be finite")
        if not torch.all(self.target_scale > 0.0):
            raise ValueError("target_scale must be positive")
        bandwidths = tuple(float(value) for value in self.bandwidths)
        if not bandwidths or any(
            not math.isfinite(value) or value <= 0.0
            for value in bandwidths
        ):
            raise ValueError("bandwidths must contain positive finite floats")
        clip = float(self.standardized_clip)
        joint_weight = float(self.joint_weight)
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError("standardized_clip must be a positive finite float")
        if not math.isfinite(joint_weight) or not 0.0 <= joint_weight <= 1.0:
            raise ValueError("joint_weight must lie in [0, 1]")
        object.__setattr__(self, "target_mean", self.target_mean.detach().cpu().clone())
        object.__setattr__(self, "target_scale", self.target_scale.detach().cpu().clone())
        object.__setattr__(self, "bandwidths", bandwidths)
        object.__setattr__(self, "standardized_clip", clip)
        object.__setattr__(self, "joint_weight", joint_weight)

    def standardized_features(
        self,
        paths: torch.Tensor,
        *,
        grid: DiscreteTimeGrid,
    ) -> torch.Tensor:
        raw = heston_mixture_path_summary_features(
            paths,
            grid=grid,
            config=self.config,
        )
        mean = self.target_mean.to(device=paths.device, dtype=paths.dtype)
        scale = self.target_scale.to(device=paths.device, dtype=paths.dtype)
        standardized = (raw - mean) / scale
        clip = float(self.standardized_clip)
        return clip * torch.tanh(standardized / clip)

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        generated = self.standardized_features(generated_paths, grid=grid)
        target = self.standardized_features(
            target_paths.to(device=generated_paths.device, dtype=generated_paths.dtype),
            grid=grid,
        )
        group_values: dict[str, torch.Tensor] = {}
        for group, feature_slice in self.config.group_slices.items():
            group_values[group] = rbf_mmd2(
                generated[:, feature_slice],
                target[:, feature_slice],
                bandwidths=self.bandwidths,
            )
        joint = rbf_mmd2(
            generated,
            target,
            bandwidths=self.bandwidths,
        )
        grouped = torch.stack(tuple(group_values.values())).mean()
        value = float(self.joint_weight) * joint + (
            1.0 - float(self.joint_weight)
        ) * grouped
        metrics = {
            f"{self.name}_value": float(value.detach().item()),
            f"{self.name}_joint_mmd": float(joint.detach().item()),
            f"{self.name}_group_mean_mmd": float(grouped.detach().item()),
            f"{self.name}_feature_dim": float(generated.shape[1]),
            f"{self.name}_generated_standardized_mean_abs": float(
                generated.detach().mean(dim=0).abs().mean().item()
            ),
            f"{self.name}_generated_standardized_std_mean": float(
                generated.detach().std(dim=0, unbiased=False).mean().item()
            ),
        }
        metrics.update(
            {
                f"{self.name}_{group}_mmd": float(group_value.detach().item())
                for group, group_value in group_values.items()
            }
        )
        return PathFunctionalDiscrepancyResult(value=value, metrics=metrics)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "feature_names": list(self.config.feature_names),
            "groups": {
                name: [int(feature_slice.start), int(feature_slice.stop)]
                for name, feature_slice in self.config.group_slices.items()
            },
            "bandwidths": list(self.bandwidths),
            "standardized_clip": float(self.standardized_clip),
            "joint_weight": float(self.joint_weight),
            "target_fitted": True,
            "uses_oracle_labels_or_parameters": False,
        }


def fit_fixed_target_heston_mixture_summary_mmd(
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    config: HestonMixtureSummaryConfig = HestonMixtureSummaryConfig(),
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    standardized_clip: float = 6.0,
    joint_weight: float = 0.5,
    scale_floor_fraction: float = 1e-3,
) -> FixedTargetHestonMixtureSummaryMMD:
    _validate_paths(target_paths, grid=grid)
    scale_floor_fraction = float(scale_floor_fraction)
    if (
        not math.isfinite(scale_floor_fraction)
        or scale_floor_fraction <= 0.0
    ):
        raise ValueError("scale_floor_fraction must be a positive finite float")
    with torch.no_grad():
        features = heston_mixture_path_summary_features(
            target_paths,
            grid=grid,
            config=config,
        )
        target_mean = features.mean(dim=0, keepdim=True)
        raw_scale = features.std(dim=0, keepdim=True, unbiased=False)
        pooled_scale = raw_scale.median().clamp_min(float(config.eps))
        target_scale = raw_scale.clamp_min(
            scale_floor_fraction * pooled_scale
        )
    return FixedTargetHestonMixtureSummaryMMD(
        target_mean=target_mean,
        target_scale=target_scale,
        config=config,
        bandwidths=tuple(float(value) for value in bandwidths),
        standardized_clip=float(standardized_clip),
        joint_weight=float(joint_weight),
    )


@dataclass(frozen=True)
class HestonMixtureCausalRidgeConditionalExpectation:
    """Ridge CE on observable causal regime summaries.

    The estimator changes only the finite-dimensional conditional projection.
    Complete pathwise MP adjoint targets and the neural regression loss remain
    unchanged.
    """

    grid: DiscreteTimeGrid
    ridge: float = 1e-3
    eps: float = 1e-6
    rolling_windows: tuple[int, ...] = (5, 10, 20, 40)
    persistence_lags: tuple[int, ...] = (1, 5, 10, 20)
    leverage_lags: tuple[int, ...] = (0, 1, 2, 5, 10, 20)
    local_volatility_windows: tuple[int, ...] = (5, 10)

    def __post_init__(self) -> None:
        ridge = float(self.ridge)
        eps = float(self.eps)
        if not math.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be a nonnegative finite float")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite float")
        object.__setattr__(
            self,
            "rolling_windows",
            _increasing_positive_ints("rolling_windows", self.rolling_windows),
        )
        object.__setattr__(
            self,
            "persistence_lags",
            _increasing_positive_ints(
                "persistence_lags",
                self.persistence_lags,
            ),
        )
        leverage = tuple(int(value) for value in self.leverage_lags)
        if (
            not leverage
            or any(value < 0 for value in leverage)
            or tuple(sorted(set(leverage))) != leverage
        ):
            raise ValueError("leverage_lags must be strictly increasing nonnegative integers")
        object.__setattr__(self, "leverage_lags", leverage)
        object.__setattr__(
            self,
            "local_volatility_windows",
            _increasing_positive_ints(
                "local_volatility_windows",
                self.local_volatility_windows,
            ),
        )
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "eps", eps)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "normalized_time",
            "current_log_price",
            "current_drawdown",
            "maximum_drawdown",
            "last_return",
            "absolute_last_return",
            "last_realized_variance",
            *(
                f"rolling_mean_return_{window}"
                for window in self.rolling_windows
            ),
            *(
                f"rolling_log_volatility_{window}"
                for window in self.rolling_windows
            ),
            *(
                f"downside_semivariance_{window}"
                for window in self.rolling_windows
            ),
            *(
                f"{series}_persistence_{lag}"
                for series in ("absolute_return", "squared_return")
                for lag in self.persistence_lags
            ),
            *(f"leverage_{lag}" for lag in self.leverage_lags),
            *(
                f"local_log_volatility_{window}_{summary}"
                for window in self.local_volatility_windows
                for summary in ("mean", "std")
            ),
        )

    def _features(self, prefixes: torch.Tensor) -> torch.Tensor:
        if prefixes.ndim != 3 or int(prefixes.shape[-1]) != 1:
            raise ValueError("prefixes must have shape (B, k + 1, 1)")
        step = int(prefixes.shape[1]) - 1
        if step >= int(self.grid.num_steps):
            raise ValueError("prefix exceeds the conditional-expectation grid")
        values = prefixes[..., 0]
        returns = (
            values[:, 1:] - values[:, :-1]
        ) / math.sqrt(float(self.grid.dt))
        batch = int(prefixes.shape[0])
        zeros = values.new_zeros(batch)
        running_max = torch.cummax(values, dim=1).values
        drawdowns = running_max - values
        last = zeros if step == 0 else returns[:, -1]
        features: list[torch.Tensor] = [
            values.new_full(
                (batch,),
                float(step) / float(max(1, int(self.grid.num_steps) - 1)),
            ),
            values[:, -1],
            drawdowns[:, -1],
            drawdowns.amax(dim=1),
            last,
            last.abs(),
            last.pow(2),
        ]
        for window in self.rolling_windows:
            if step == 0:
                features.append(zeros)
            else:
                features.append(returns[:, max(0, step - window) :].mean(dim=1))
        for window in self.rolling_windows:
            if step == 0:
                features.append(zeros)
            else:
                recent = returns[:, max(0, step - window) :]
                features.append(
                    torch.log(
                        torch.sqrt(
                            recent.pow(2).mean(dim=1) + float(self.eps) ** 2
                        )
                    )
                )
        for window in self.rolling_windows:
            if step == 0:
                features.append(zeros)
            else:
                recent = returns[:, max(0, step - window) :]
                features.append(torch.relu(-recent).pow(2).mean(dim=1))

        smooth_abs = torch.sqrt(returns.pow(2) + float(self.eps) ** 2)
        squared = returns.pow(2)
        for series in (smooth_abs, squared):
            for lag in self.persistence_lags:
                if step <= lag:
                    features.append(zeros)
                else:
                    features.append(
                        _pathwise_correlation(
                            series[:, :-lag],
                            series[:, lag:],
                            eps=float(self.eps),
                        )
                    )
        for lag in self.leverage_lags:
            pair_count = step - lag
            if pair_count < 2:
                features.append(zeros)
            else:
                left = returns if lag == 0 else returns[:, :-lag]
                right = squared if lag == 0 else squared[:, lag:]
                features.append(
                    _pathwise_correlation(
                        left,
                        right,
                        eps=float(self.eps),
                    )
                )
        for window in self.local_volatility_windows:
            if step < window:
                features.extend((zeros, zeros))
            else:
                local = torch.sqrt(
                    returns.unfold(1, window, 1).pow(2).mean(dim=2)
                    + float(self.eps) ** 2
                )
                log_local = torch.log(local)
                features.extend(
                    (
                        log_local.mean(dim=1),
                        log_local.std(dim=1, unbiased=False),
                    )
                )
        result = torch.stack(features, dim=1)
        if int(result.shape[1]) != len(self.feature_names):
            raise RuntimeError("causal Heston-mixture feature construction is inconsistent")
        return result

    def fit_predict(
        self,
        *,
        prefixes_train: torch.Tensor,
        targets_train: torch.Tensor,
        prefixes_eval: torch.Tensor | None = None,
    ) -> ConditionalExpectationResult:
        if targets_train.ndim != 2:
            raise ValueError("targets_train must have shape (B, m)")
        train_features = self._features(prefixes_train)
        if int(train_features.shape[0]) != int(targets_train.shape[0]):
            raise ValueError("prefixes_train and targets_train batch sizes must match")
        mean = train_features.mean(dim=0, keepdim=True)
        scale = train_features.std(
            dim=0,
            keepdim=True,
            unbiased=False,
        ).clamp_min(float(self.eps))
        normalized = (train_features - mean) / scale
        design = torch.cat(
            [
                torch.ones(
                    int(normalized.shape[0]),
                    1,
                    device=normalized.device,
                    dtype=normalized.dtype,
                ),
                normalized,
            ],
            dim=1,
        )
        coefficients = fit_ridge(
            design=design,
            target=targets_train,
            ridge=float(self.ridge),
        )
        train_prediction = design @ coefficients
        if prefixes_eval is None:
            prediction = train_prediction
        else:
            if tuple(prefixes_eval.shape[1:]) != tuple(prefixes_train.shape[1:]):
                raise ValueError("prefix train/eval shapes must match after the batch dimension")
            eval_features = self._features(prefixes_eval)
            eval_normalized = (eval_features - mean) / scale
            eval_design = torch.cat(
                [
                    torch.ones(
                        int(eval_normalized.shape[0]),
                        1,
                        device=eval_normalized.device,
                        dtype=eval_normalized.dtype,
                    ),
                    eval_normalized,
                ],
                dim=1,
            )
            prediction = eval_design @ coefficients
        fit_rms = torch.sqrt(
            torch.mean((train_prediction - targets_train).pow(2))
        )
        return ConditionalExpectationResult(
            prediction=prediction,
            metrics={
                "ce_feature_dim": float(train_features.shape[1]),
                "ce_ridge": float(self.ridge),
                "ce_fit_rms": float(fit_rms.item()),
                "ce_heston_mixture_causal_basis": 1.0,
            },
        )


__all__ = [
    "FixedTargetHestonMixtureSummaryMMD",
    "HestonMixtureCausalRidgeConditionalExpectation",
    "HestonMixtureSummaryConfig",
    "fit_fixed_target_heston_mixture_summary_mmd",
    "heston_mixture_path_summary_features",
]
