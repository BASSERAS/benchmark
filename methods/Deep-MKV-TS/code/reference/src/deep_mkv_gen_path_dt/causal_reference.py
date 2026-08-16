from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from deep_mkv_gen_path_dt.ce import ConditionalExpectationResult, fit_ridge
from deep_mkv_gen_path_dt.config import _require_finite_float
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid
from deep_mkv_gen_path_dt.reference import LocalGaussianReferenceKernel


def _validate_positive_ints(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value < 1 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _validate_optional_positive_ints(
    name: str,
    values: Sequence[int],
) -> tuple[int, ...]:
    return () if not values else _validate_positive_ints(name, values)


def _validate_optional_nonnegative_ints(
    name: str,
    values: Sequence[int],
) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError(f"{name} must contain nonnegative integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _validate_optional_positive_float(
    name: str,
    value: float | None,
) -> float | None:
    if value is None:
        return None
    result = _require_finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0 when provided")
    return result


def _soft_clip_standardized(
    values: torch.Tensor,
    *,
    clip: float | None,
) -> torch.Tensor:
    if clip is None:
        return values
    return float(clip) * torch.tanh(values / float(clip))


def _feature_names(
    *,
    rolling_windows: tuple[int, ...],
    ewma_half_lives: tuple[int, ...],
    downside_windows: tuple[int, ...],
    drawdown_thresholds: tuple[float, ...],
    drawdown_memory_half_lives: tuple[int, ...],
    drawdown_memory_lags: tuple[int, ...],
    include_expanding_moments: bool,
    persistence_lags: tuple[int, ...],
    leverage_lags: tuple[int, ...],
    local_volatility_windows: tuple[int, ...],
) -> tuple[str, ...]:
    return (
        "normalized_time",
        "current_log_price",
        "drawdown",
        "maximum_drawdown",
        "last_return",
        "absolute_last_return",
        "last_realized_variance",
        "rolling_mean_return_5",
        "rolling_mean_return_20",
        *(f"rolling_realized_variance_{window}" for window in rolling_windows),
        *(f"ewma_realized_variance_{half_life}" for half_life in ewma_half_lives),
        *(f"downside_semivariance_{window}" for window in downside_windows),
        *(
            f"current_drawdown_gate_{threshold:g}"
            for threshold in drawdown_thresholds
        ),
        *(
            f"decayed_drawdown_memory_{threshold:g}_{half_life}"
            for threshold in drawdown_thresholds
            for half_life in drawdown_memory_half_lives
        ),
        *(
            f"lagged_drawdown_memory_{threshold:g}_{half_life}_lag_{lag}"
            for threshold in drawdown_thresholds
            for half_life in drawdown_memory_half_lives
            for lag in drawdown_memory_lags
        ),
        *(
            (
                "expanding_log_realized_variance",
                "expanding_log_fourth_moment",
                "expanding_downside_variance_share",
            )
            if include_expanding_moments
            else ()
        ),
        *(
            f"{series}_persistence_{lag}"
            for series in ("absolute_return", "squared_return")
            for lag in persistence_lags
        ),
        *(f"leverage_{lag}" for lag in leverage_lags),
        *(
            f"local_log_volatility_{window}_{summary}"
            for window in local_volatility_windows
            for summary in ("mean", "std", "min", "max")
        ),
    )


def _causal_features_at_step(
    x_prefix: torch.Tensor,
    *,
    step_index: int,
    grid: DiscreteTimeGrid,
    rolling_windows: tuple[int, ...],
    ewma_half_lives: tuple[int, ...],
    downside_windows: tuple[int, ...],
    drawdown_thresholds: tuple[float, ...] = (),
    drawdown_memory_half_lives: tuple[int, ...] = (),
    drawdown_memory_lags: tuple[int, ...] = (),
    include_expanding_moments: bool = False,
    persistence_lags: tuple[int, ...] = (),
    leverage_lags: tuple[int, ...] = (),
    local_volatility_windows: tuple[int, ...] = (),
    drawdown_gate_temperature: float = 0.005,
) -> torch.Tensor:
    if x_prefix.ndim != 3 or int(x_prefix.shape[-1]) != 1:
        raise ValueError("causal volatility reference requires paths of shape (B, n + 1, 1)")
    step = int(step_index)
    if int(x_prefix.shape[1]) != step + 1:
        raise ValueError("x_prefix must end at step_index")
    if not 0 <= step < int(grid.num_steps):
        raise ValueError("step_index is out of bounds")
    batch = int(x_prefix.shape[0])
    values = x_prefix[..., 0]
    returns = values[:, 1:] - values[:, :-1]
    zeros = values.new_zeros(batch)
    time_denominator = float(max(1, int(grid.num_steps) - 1))
    normalized_time = values.new_full((batch,), float(step) / time_denominator)
    current = values[:, -1]
    running_max = torch.cummax(values, dim=1).values
    drawdown_history = running_max - values
    current_drawdown = drawdown_history[:, -1]
    drawdown = -current_drawdown
    maximum_drawdown = drawdown_history.max(dim=1).values
    if step == 0:
        last = zeros
    else:
        last = returns[:, -1]
    dt = float(grid.dt)
    features = [
        normalized_time,
        current,
        drawdown,
        maximum_drawdown,
        last,
        last.abs(),
        last.pow(2) / dt,
    ]

    for window in (5, 20):
        if step == 0:
            features.append(zeros)
        else:
            recent = returns[:, max(0, step - int(window)) : step]
            features.append(recent.mean(dim=1))

    for window in rolling_windows:
        if step == 0:
            features.append(zeros)
        else:
            recent = returns[:, max(0, step - int(window)) : step]
            features.append(recent.pow(2).mean(dim=1) / dt)

    for half_life in ewma_half_lives:
        if step == 0:
            features.append(zeros)
        else:
            ages = torch.arange(
                step - 1,
                -1,
                -1,
                device=returns.device,
                dtype=returns.dtype,
            )
            weights = torch.exp(
                -math.log(2.0) * ages / float(half_life)
            )
            weights = weights / weights.sum()
            features.append((returns.pow(2) * weights.unsqueeze(0)).sum(dim=1) / dt)

    for window in downside_windows:
        if step == 0:
            features.append(zeros)
        else:
            recent = returns[:, max(0, step - int(window)) : step]
            features.append(torch.relu(-recent).pow(2).mean(dim=1) / dt)

    drawdown_gates = tuple(
        torch.sigmoid(
            (drawdown_history - float(threshold))
            / float(drawdown_gate_temperature)
        )
        for threshold in drawdown_thresholds
    )
    features.extend(gates[:, -1] for gates in drawdown_gates)
    for gates in drawdown_gates:
        for half_life in drawdown_memory_half_lives:
            ages = torch.arange(
                step,
                -1,
                -1,
                device=values.device,
                dtype=values.dtype,
            )
            decay = torch.exp(
                -math.log(2.0) * ages / float(half_life)
            )
            # This is the differentiable analogue of
            # M_t = max(decay * M_{t-1}, 1{DD_t >= threshold}).
            features.append((gates * decay.unsqueeze(0)).amax(dim=1))
    for gates in drawdown_gates:
        for half_life in drawdown_memory_half_lives:
            for lag in drawdown_memory_lags:
                history_end = step - int(lag)
                if history_end < 0:
                    features.append(zeros)
                    continue
                lagged_ages = torch.arange(
                    history_end,
                    -1,
                    -1,
                    device=values.device,
                    dtype=values.dtype,
                )
                lagged_decay = torch.exp(
                    -math.log(2.0) * lagged_ages / float(half_life)
                )
                # Observable delayed state M_{t-lag}.  No future path value is
                # used, and the state is zero before the requested lag exists.
                features.append(
                    (
                        gates[:, : history_end + 1]
                        * lagged_decay.unsqueeze(0)
                    ).amax(dim=1)
                )

    normalized_returns = returns / math.sqrt(dt)
    evidence_eps = 1e-6
    if include_expanding_moments:
        if step == 0:
            features.extend((zeros, zeros, zeros))
        else:
            squared_normalized = normalized_returns.pow(2)
            features.extend(
                (
                    torch.log(
                        squared_normalized.mean(dim=1)
                        + evidence_eps**2
                    ),
                    torch.log(
                        normalized_returns.pow(4).mean(dim=1)
                        + evidence_eps**4
                    ),
                    torch.relu(-normalized_returns).pow(2).sum(dim=1)
                    / squared_normalized.sum(dim=1).clamp_min(
                        evidence_eps**2
                    ),
                )
            )
    smooth_abs = torch.sqrt(
        normalized_returns.pow(2) + evidence_eps**2
    )
    squared_normalized = normalized_returns.pow(2)
    for series in (smooth_abs, squared_normalized):
        for lag in persistence_lags:
            if step <= int(lag):
                features.append(zeros)
            else:
                left = series[:, :-int(lag)]
                right = series[:, int(lag) :]
                left = left - left.mean(dim=1, keepdim=True)
                right = right - right.mean(dim=1, keepdim=True)
                features.append(
                    (left * right).mean(dim=1)
                    / torch.sqrt(
                        left.pow(2).mean(dim=1)
                        * right.pow(2).mean(dim=1)
                        + evidence_eps**2
                    )
                )
    for lag in leverage_lags:
        pair_count = step - int(lag)
        if pair_count < 2:
            features.append(zeros)
        else:
            left = (
                normalized_returns
                if int(lag) == 0
                else normalized_returns[:, :-int(lag)]
            )
            right = (
                squared_normalized
                if int(lag) == 0
                else squared_normalized[:, int(lag) :]
            )
            left = left - left.mean(dim=1, keepdim=True)
            right = right - right.mean(dim=1, keepdim=True)
            features.append(
                (left * right).mean(dim=1)
                / torch.sqrt(
                    left.pow(2).mean(dim=1)
                    * right.pow(2).mean(dim=1)
                    + evidence_eps**2
                )
            )
    for window in local_volatility_windows:
        if step < int(window):
            features.extend((zeros, zeros, zeros, zeros))
        else:
            local = torch.sqrt(
                normalized_returns.unfold(1, int(window), 1)
                .pow(2)
                .mean(dim=2)
                + evidence_eps**2
            )
            log_local = torch.log(local)
            features.extend(
                (
                    log_local.mean(dim=1),
                    log_local.std(dim=1, unbiased=False),
                    log_local.amin(dim=1),
                    log_local.amax(dim=1),
                )
            )

    return torch.stack(features, dim=1)


def _causal_features_all(
    paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    rolling_windows: tuple[int, ...],
    ewma_half_lives: tuple[int, ...],
    downside_windows: tuple[int, ...],
    drawdown_thresholds: tuple[float, ...] = (),
    drawdown_memory_half_lives: tuple[int, ...] = (),
    drawdown_memory_lags: tuple[int, ...] = (),
    include_expanding_moments: bool = False,
    persistence_lags: tuple[int, ...] = (),
    leverage_lags: tuple[int, ...] = (),
    local_volatility_windows: tuple[int, ...] = (),
    drawdown_gate_temperature: float = 0.005,
) -> torch.Tensor:
    if paths.ndim != 3 or tuple(paths.shape[1:]) != (
        int(grid.num_steps) + 1,
        1,
    ):
        raise ValueError("target_paths must have shape (B, N + 1, 1)")
    batch = int(paths.shape[0])
    steps = int(grid.num_steps)
    values = paths[..., 0]
    points = values[:, :-1]
    returns = values[:, 1:] - values[:, :-1]
    zeros = values.new_zeros((batch, steps))
    time = torch.arange(steps, device=values.device, dtype=values.dtype)
    normalized_time = time.unsqueeze(0).expand(batch, -1) / float(
        max(1, steps - 1)
    )
    running_max = torch.cummax(points, dim=1).values
    drawdown_history = running_max - points
    maximum_drawdown = torch.cummax(drawdown_history, dim=1).values
    last_return = torch.cat(
        [values.new_zeros((batch, 1)), returns[:, :-1]],
        dim=1,
    )
    dt = float(grid.dt)
    features = [
        normalized_time,
        points,
        -drawdown_history,
        maximum_drawdown,
        last_return,
        last_return.abs(),
        last_return.pow(2) / dt,
    ]

    def trailing_mean(history: torch.Tensor, window: int) -> torch.Tensor:
        cumulative = torch.cat(
            [
                history.new_zeros((batch, 1)),
                torch.cumsum(history, dim=1),
            ],
            dim=1,
        )
        end = torch.arange(steps, device=history.device)
        start = (end - int(window)).clamp_min(0)
        total = cumulative[:, end] - cumulative[:, start]
        count = end.clamp_max(int(window)).clamp_min(1).to(history.dtype)
        return total / count.unsqueeze(0)

    for window in (5, 20):
        features.append(trailing_mean(returns, int(window)))
    squared_returns = returns.pow(2)
    for window in rolling_windows:
        features.append(trailing_mean(squared_returns, int(window)) / dt)
    for half_life in ewma_half_lives:
        decay = math.exp(-math.log(2.0) / float(half_life))
        numerator = values.new_zeros(batch)
        denominator = 0.0
        timewise = []
        for step in range(steps):
            if step > 0:
                numerator = decay * numerator + squared_returns[:, step - 1]
                denominator = decay * denominator + 1.0
            timewise.append(
                numerator / max(denominator, torch.finfo(values.dtype).eps)
            )
        features.append(torch.stack(timewise, dim=1) / dt)
    downside = torch.relu(-returns).pow(2)
    for window in downside_windows:
        features.append(trailing_mean(downside, int(window)) / dt)

    drawdown_gates = tuple(
        torch.sigmoid(
            (drawdown_history - float(threshold))
            / float(drawdown_gate_temperature)
        )
        for threshold in drawdown_thresholds
    )
    features.extend(drawdown_gates)
    memories: list[list[torch.Tensor]] = []
    for gates in drawdown_gates:
        threshold_memories = []
        for half_life in drawdown_memory_half_lives:
            log_decay = -math.log(2.0) / float(half_life)
            scale = torch.exp((-log_decay) * time)
            memory = (
                torch.cummax(gates * scale.unsqueeze(0), dim=1).values
                * torch.exp(log_decay * time).unsqueeze(0)
            )
            features.append(memory)
            threshold_memories.append(memory)
        memories.append(threshold_memories)
    for threshold_memories in memories:
        for memory in threshold_memories:
            for lag in drawdown_memory_lags:
                lagged = zeros.clone()
                lagged[:, int(lag) :] = memory[:, : -int(lag)]
                features.append(lagged)

    normalized_returns = returns / math.sqrt(dt)
    evidence_eps = 1e-6
    squared_normalized = normalized_returns.pow(2)
    fourth_normalized = squared_normalized.pow(2)

    def expanding_past_sum(history: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                history.new_zeros((batch, 1)),
                torch.cumsum(history, dim=1)[:, :-1],
            ],
            dim=1,
        )

    count = time.clamp_min(1).to(values.dtype).unsqueeze(0)
    if include_expanding_moments:
        past_squared = expanding_past_sum(squared_normalized)
        past_fourth = expanding_past_sum(fourth_normalized)
        expanding_variance = past_squared / count
        expanding_fourth = past_fourth / count
        downside_sum = expanding_past_sum(
            torch.relu(-normalized_returns).pow(2)
        )
        active = (time > 0).unsqueeze(0)
        features.extend(
            (
                torch.where(
                    active,
                    torch.log(expanding_variance + evidence_eps**2),
                    zeros,
                ),
                torch.where(
                    active,
                    torch.log(expanding_fourth + evidence_eps**4),
                    zeros,
                ),
                torch.where(
                    active,
                    downside_sum / past_squared.clamp_min(evidence_eps**2),
                    zeros,
                ),
            )
        )

    def expanding_lagged_correlation(
        left_series: torch.Tensor,
        right_series: torch.Tensor,
        *,
        lag: int,
    ) -> torch.Tensor:
        lag_value = int(lag)
        left = (
            left_series
            if lag_value == 0
            else left_series[:, :-lag_value]
        )
        right = (
            right_series
            if lag_value == 0
            else right_series[:, lag_value:]
        )
        pair_steps = int(left.shape[1])
        sample_count = torch.arange(
            1,
            pair_steps + 1,
            device=values.device,
            dtype=values.dtype,
        ).unsqueeze(0)
        left_sum = torch.cumsum(left, dim=1)
        right_sum = torch.cumsum(right, dim=1)
        left_sq_sum = torch.cumsum(left.pow(2), dim=1)
        right_sq_sum = torch.cumsum(right.pow(2), dim=1)
        cross_sum = torch.cumsum(left * right, dim=1)
        left_mean = left_sum / sample_count
        right_mean = right_sum / sample_count
        covariance = cross_sum / sample_count - left_mean * right_mean
        left_variance = (
            left_sq_sum / sample_count - left_mean.pow(2)
        ).clamp_min(0.0)
        right_variance = (
            right_sq_sum / sample_count - right_mean.pow(2)
        ).clamp_min(0.0)
        correlation = covariance / torch.sqrt(
            left_variance * right_variance + evidence_eps**2
        )
        output = zeros.clone()
        start = lag_value + 1
        if start < steps:
            output[:, start:] = correlation[:, : steps - start]
        return output

    smooth_abs = torch.sqrt(squared_normalized + evidence_eps**2)
    for series in (smooth_abs, squared_normalized):
        for lag in persistence_lags:
            features.append(
                expanding_lagged_correlation(
                    series,
                    series,
                    lag=int(lag),
                )
            )
    for lag in leverage_lags:
        features.append(
            expanding_lagged_correlation(
                normalized_returns,
                squared_normalized,
                lag=int(lag),
            )
        )
    for window in local_volatility_windows:
        window_value = int(window)
        local = torch.sqrt(
            normalized_returns.unfold(1, window_value, 1)
            .pow(2)
            .mean(dim=2)
            + evidence_eps**2
        )
        log_local = torch.log(local)
        usable = max(0, steps - window_value)
        local_features = [zeros.clone() for _ in range(4)]
        if usable > 0:
            sequence = log_local[:, :usable]
            local_count = torch.arange(
                1,
                usable + 1,
                device=values.device,
                dtype=values.dtype,
            ).unsqueeze(0)
            cumulative = torch.cumsum(sequence, dim=1)
            cumulative_squared = torch.cumsum(sequence.pow(2), dim=1)
            local_mean = cumulative / local_count
            local_variance = (
                cumulative_squared / local_count - local_mean.pow(2)
            ).clamp_min(0.0)
            local_features[0][:, window_value:] = local_mean
            local_features[1][:, window_value:] = torch.sqrt(local_variance)
            local_features[2][:, window_value:] = torch.cummin(
                sequence,
                dim=1,
            ).values
            local_features[3][:, window_value:] = torch.cummax(
                sequence,
                dim=1,
            ).values
        features.extend(local_features)
    return torch.stack(features, dim=2)


def _gaussian_nll(
    design: torch.Tensor,
    realized_variance: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    log_variance = (design @ coefficients).clamp(min=-30.0, max=15.0)
    return 0.5 * (
        log_variance + realized_variance * torch.exp(-log_variance)
    ).mean()


def _fit_log_variance_newton(
    *,
    design: torch.Tensor,
    realized_variance: torch.Tensor,
    ridge: float,
    max_steps: int,
    tolerance: float,
) -> torch.Tensor:
    design64 = design.detach().to(dtype=torch.float64)
    variance64 = realized_variance.detach().to(dtype=torch.float64).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    width = int(design64.shape[1])
    coefficients = design64.new_zeros((width, 1))
    coefficients[0, 0] = torch.log(variance64.mean())
    penalty = design64.new_ones(width)
    penalty[0] = 0.0

    def penalized_objective(beta: torch.Tensor) -> torch.Tensor:
        return _gaussian_nll(design64, variance64, beta) + 0.5 * float(
            ridge
        ) * (penalty[:, None] * beta.pow(2)).sum()

    for _ in range(int(max_steps)):
        log_variance = (design64 @ coefficients).clamp(min=-30.0, max=15.0)
        weights = variance64 * torch.exp(-log_variance)
        gradient = (
            0.5
            * (design64.transpose(0, 1) @ (1.0 - weights))
            / float(design64.shape[0])
            + float(ridge) * penalty[:, None] * coefficients
        )
        hessian = (
            0.5
            * (
                design64.transpose(0, 1)
                @ (weights * design64)
            )
            / float(design64.shape[0])
        )
        hessian = hessian + float(ridge) * torch.diag(penalty)
        hessian = hessian + 1e-10 * torch.eye(
            width,
            device=hessian.device,
            dtype=hessian.dtype,
        )
        direction = torch.linalg.solve(hessian, gradient)
        old_objective = penalized_objective(coefficients)
        step_size = 1.0
        candidate = coefficients - direction
        for _ in range(20):
            candidate = coefficients - step_size * direction
            if bool((penalized_objective(candidate) <= old_objective).item()):
                break
            step_size *= 0.5
        change = float(
            torch.linalg.vector_norm(step_size * direction).item()
        )
        coefficients = candidate
        if change <= float(tolerance):
            break
    return coefficients.detach().to(device=design.device, dtype=design.dtype)


@dataclass(frozen=True)
class CausalPathFeatureRidgeConditionalExpectation:
    """Ridge CE on a fixed-dimensional, observable causal path basis.

    The basis is shared with :class:`CausalVolatilityReferenceKernel`.  In
    particular, optional decayed drawdown-threshold features preserve an
    earlier drawdown event even after the path has recovered at the current
    time.  The estimator remains a projection of the complete MP adjoint
    target; it does not change that target or the network regression loss.
    """

    grid: DiscreteTimeGrid
    ridge: float = 1e-3
    eps: float = 1e-6
    rolling_windows: tuple[int, ...] = (5, 10, 20, 32)
    ewma_half_lives: tuple[int, ...] = (5, 20, 60)
    downside_windows: tuple[int, ...] = (10, 32)
    drawdown_thresholds: tuple[float, ...] = ()
    drawdown_memory_half_lives: tuple[int, ...] = ()
    drawdown_memory_lags: tuple[int, ...] = ()
    include_expanding_moments: bool = False
    persistence_lags: tuple[int, ...] = ()
    leverage_lags: tuple[int, ...] = ()
    local_volatility_windows: tuple[int, ...] = ()
    drawdown_gate_temperature: float = 0.005
    standardized_feature_clip: float | None = None
    fixed_feature_center: torch.Tensor | None = None
    fixed_feature_scale: torch.Tensor | None = None
    pool_time: bool = False
    include_volatility_gates: bool = False
    volatility_gate_centers: tuple[float, ...] = (-1.0, 0.0, 1.0)
    volatility_gate_temperature: float = 0.5

    def __post_init__(self) -> None:
        ridge = _require_finite_float("ridge", self.ridge)
        eps = _require_finite_float("eps", self.eps)
        if ridge < 0.0:
            raise ValueError("ridge must be >= 0")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        if not isinstance(self.pool_time, bool):
            raise ValueError("pool_time must be bool")
        if not isinstance(self.include_volatility_gates, bool):
            raise ValueError("include_volatility_gates must be bool")
        gate_centers = tuple(float(value) for value in self.volatility_gate_centers)
        if not gate_centers or any(not math.isfinite(value) for value in gate_centers):
            raise ValueError("volatility_gate_centers must be finite and non-empty")
        gate_temperature = _require_finite_float(
            "volatility_gate_temperature",
            self.volatility_gate_temperature,
        )
        if gate_temperature <= 0.0:
            raise ValueError("volatility_gate_temperature must be > 0")
        rolling = _validate_positive_ints("rolling_windows", self.rolling_windows)
        ewma = _validate_positive_ints("ewma_half_lives", self.ewma_half_lives)
        downside = _validate_positive_ints("downside_windows", self.downside_windows)
        thresholds = tuple(float(value) for value in self.drawdown_thresholds)
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in thresholds
        ):
            raise ValueError("drawdown_thresholds must contain positive finite floats")
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("drawdown_thresholds must not contain duplicates")
        memory_half_lives = (
            ()
            if not thresholds
            else _validate_positive_ints(
                "drawdown_memory_half_lives",
                self.drawdown_memory_half_lives,
            )
        )
        memory_lags = (
            ()
            if not thresholds or not self.drawdown_memory_lags
            else _validate_positive_ints(
                "drawdown_memory_lags",
                self.drawdown_memory_lags,
            )
        )
        if any(value >= int(self.grid.num_steps) for value in memory_lags):
            raise ValueError("drawdown_memory_lags must be smaller than grid.num_steps")
        if not isinstance(self.include_expanding_moments, bool):
            raise ValueError("include_expanding_moments must be bool")
        persistence_lags = _validate_optional_positive_ints(
            "persistence_lags",
            self.persistence_lags,
        )
        leverage_lags = _validate_optional_nonnegative_ints(
            "leverage_lags",
            self.leverage_lags,
        )
        local_volatility_windows = _validate_optional_positive_ints(
            "local_volatility_windows",
            self.local_volatility_windows,
        )
        if any(value >= int(self.grid.num_steps) for value in persistence_lags):
            raise ValueError("persistence_lags must be smaller than grid.num_steps")
        if any(value >= int(self.grid.num_steps) for value in leverage_lags):
            raise ValueError("leverage_lags must be smaller than grid.num_steps")
        if any(
            value >= int(self.grid.num_steps)
            for value in local_volatility_windows
        ):
            raise ValueError(
                "local_volatility_windows must be smaller than grid.num_steps"
            )
        temperature = _require_finite_float(
            "drawdown_gate_temperature",
            self.drawdown_gate_temperature,
        )
        if temperature <= 0.0:
            raise ValueError("drawdown_gate_temperature must be > 0")
        feature_clip = _validate_optional_positive_float(
            "standardized_feature_clip",
            self.standardized_feature_clip,
        )
        fixed_center = self.fixed_feature_center
        fixed_scale = self.fixed_feature_scale
        if (fixed_center is None) != (fixed_scale is None):
            raise ValueError(
                "fixed_feature_center and fixed_feature_scale must be provided together"
            )
        if fixed_center is not None and fixed_scale is not None:
            expected_shape = (
                int(self.grid.num_steps),
                len(
                    _feature_names(
                        rolling_windows=rolling,
                        ewma_half_lives=ewma,
                        downside_windows=downside,
                        drawdown_thresholds=thresholds,
                        drawdown_memory_half_lives=memory_half_lives,
                        drawdown_memory_lags=memory_lags,
                        include_expanding_moments=self.include_expanding_moments,
                        persistence_lags=persistence_lags,
                        leverage_lags=leverage_lags,
                        local_volatility_windows=local_volatility_windows,
                    )
                ),
            )
            if tuple(fixed_center.shape) != expected_shape:
                raise ValueError(
                    "fixed_feature_center must have shape (N, feature_dim)"
                )
            if tuple(fixed_scale.shape) != expected_shape:
                raise ValueError(
                    "fixed_feature_scale must match fixed_feature_center"
                )
            if not bool(torch.isfinite(fixed_center).all()):
                raise ValueError("fixed_feature_center must be finite")
            if not bool(torch.isfinite(fixed_scale).all()) or bool(
                (fixed_scale <= 0.0).any()
            ):
                raise ValueError("fixed_feature_scale must be finite and positive")
            fixed_center = fixed_center.detach().clone()
            fixed_scale = fixed_scale.detach().clone()
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "rolling_windows", rolling)
        object.__setattr__(self, "ewma_half_lives", ewma)
        object.__setattr__(self, "downside_windows", downside)
        object.__setattr__(self, "drawdown_thresholds", thresholds)
        object.__setattr__(self, "drawdown_memory_half_lives", memory_half_lives)
        object.__setattr__(self, "drawdown_memory_lags", memory_lags)
        object.__setattr__(self, "persistence_lags", persistence_lags)
        object.__setattr__(self, "leverage_lags", leverage_lags)
        object.__setattr__(
            self,
            "local_volatility_windows",
            local_volatility_windows,
        )
        object.__setattr__(self, "drawdown_gate_temperature", temperature)
        object.__setattr__(self, "standardized_feature_clip", feature_clip)
        object.__setattr__(self, "fixed_feature_center", fixed_center)
        object.__setattr__(self, "fixed_feature_scale", fixed_scale)
        object.__setattr__(self, "volatility_gate_centers", gate_centers)
        object.__setattr__(self, "volatility_gate_temperature", gate_temperature)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return _feature_names(
            rolling_windows=self.rolling_windows,
            ewma_half_lives=self.ewma_half_lives,
            downside_windows=self.downside_windows,
            drawdown_thresholds=self.drawdown_thresholds,
            drawdown_memory_half_lives=self.drawdown_memory_half_lives,
            drawdown_memory_lags=self.drawdown_memory_lags,
            include_expanding_moments=self.include_expanding_moments,
            persistence_lags=self.persistence_lags,
            leverage_lags=self.leverage_lags,
            local_volatility_windows=self.local_volatility_windows,
        )

    def _features(self, prefixes: torch.Tensor) -> torch.Tensor:
        return _causal_features_at_step(
            prefixes,
            step_index=int(prefixes.shape[1]) - 1,
            grid=self.grid,
            rolling_windows=self.rolling_windows,
            ewma_half_lives=self.ewma_half_lives,
            downside_windows=self.downside_windows,
            drawdown_thresholds=self.drawdown_thresholds,
            drawdown_memory_half_lives=self.drawdown_memory_half_lives,
            drawdown_memory_lags=self.drawdown_memory_lags,
            include_expanding_moments=self.include_expanding_moments,
            persistence_lags=self.persistence_lags,
            leverage_lags=self.leverage_lags,
            local_volatility_windows=self.local_volatility_windows,
            drawdown_gate_temperature=float(self.drawdown_gate_temperature),
        )

    def features_at(self, prefixes: torch.Tensor) -> torch.Tensor:
        """Return the observable feature vector at the end of a prefix."""

        features = self._features(prefixes)
        if self.fixed_feature_center is None:
            return features
        step = int(prefixes.shape[1]) - 1
        center = self.fixed_feature_center[step].unsqueeze(0).to(features)
        scale = self.fixed_feature_scale[step].unsqueeze(0).to(features)
        return _soft_clip_standardized(
            (features - center) / scale,
            clip=self.standardized_feature_clip,
        )

    def features_all(self, paths: torch.Tensor) -> torch.Tensor:
        features = _causal_features_all(
            paths,
            grid=self.grid,
            rolling_windows=self.rolling_windows,
            ewma_half_lives=self.ewma_half_lives,
            downside_windows=self.downside_windows,
            drawdown_thresholds=self.drawdown_thresholds,
            drawdown_memory_half_lives=self.drawdown_memory_half_lives,
            drawdown_memory_lags=self.drawdown_memory_lags,
            include_expanding_moments=self.include_expanding_moments,
            persistence_lags=self.persistence_lags,
            leverage_lags=self.leverage_lags,
            local_volatility_windows=self.local_volatility_windows,
            drawdown_gate_temperature=float(self.drawdown_gate_temperature),
        )
        if self.fixed_feature_center is None:
            return features
        center = self.fixed_feature_center.unsqueeze(0).to(features)
        scale = self.fixed_feature_scale.unsqueeze(0).to(features)
        return _soft_clip_standardized(
            (features - center) / scale,
            clip=self.standardized_feature_clip,
        )

    def fit_predict_features(
        self,
        *,
        features_train: torch.Tensor,
        targets_train: torch.Tensor,
        features_eval: torch.Tensor | None = None,
    ) -> ConditionalExpectationResult:
        if features_train.ndim != 2:
            raise ValueError("features_train must have shape (B, f)")
        if targets_train.ndim != 2:
            raise ValueError("targets_train must have shape (B, m)")
        if int(features_train.shape[0]) != int(targets_train.shape[0]):
            raise ValueError("features_train and targets_train must share batch size")
        if (
            features_eval is not None
            and (
                features_eval.ndim != 2
                or int(features_eval.shape[1]) != int(features_train.shape[1])
            )
        ):
            raise ValueError("features_eval must have shape (B_eval, f)")
        train_features = features_train
        uses_fixed_normalization = self.fixed_feature_center is not None
        if uses_fixed_normalization:
            mean = None
            std = None
            normalized_train = train_features
        else:
            mean = train_features.mean(dim=0, keepdim=True)
            std = train_features.std(
                dim=0,
                keepdim=True,
                unbiased=False,
            ).clamp_min(float(self.eps))
            normalized_train = _soft_clip_standardized(
                (train_features - mean) / std,
                clip=self.standardized_feature_clip,
            )
        projection_train = self._projection_features(normalized_train)
        design_train = torch.cat(
            [
                torch.ones(
                    int(projection_train.shape[0]),
                    1,
                    device=projection_train.device,
                    dtype=projection_train.dtype,
                ),
                projection_train,
            ],
            dim=1,
        )
        # ``fit_ridge`` uses a sum-of-squares convention.  When paths and
        # times are pooled, scale the penalty by the number of time points so
        # that the regularization per observation matches the stepwise fit.
        effective_ridge = float(self.ridge) * (
            int(self.grid.num_steps) if self.pool_time else 1
        )
        coefficients = fit_ridge(
            design=design_train,
            target=targets_train,
            ridge=effective_ridge,
        )
        train_prediction = design_train @ coefficients
        if features_eval is None:
            prediction = train_prediction
        else:
            if uses_fixed_normalization:
                normalized_eval = features_eval
            else:
                assert mean is not None and std is not None
                normalized_eval = _soft_clip_standardized(
                    (features_eval - mean) / std,
                    clip=self.standardized_feature_clip,
                )
            projection_eval = self._projection_features(normalized_eval)
            design_eval = torch.cat(
                [
                    torch.ones(
                        int(projection_eval.shape[0]),
                        1,
                        device=projection_eval.device,
                        dtype=projection_eval.dtype,
                    ),
                    projection_eval,
                ],
                dim=1,
            )
            prediction = design_eval @ coefficients
        fit_rms = float(
            torch.sqrt(torch.mean((train_prediction - targets_train).pow(2))).item()
        )
        return ConditionalExpectationResult(
            prediction=prediction,
            metrics={
                "ce_feature_dim": float(projection_train.shape[1]),
                "ce_raw_feature_dim": float(train_features.shape[1]),
                "ce_volatility_gate_count": float(
                    projection_train.shape[1] - train_features.shape[1]
                ),
                "ce_ridge": float(self.ridge),
                "ce_effective_ridge": effective_ridge,
                "ce_fit_rms": fit_rms,
                "ce_drawdown_threshold_count": float(
                    len(self.drawdown_thresholds)
                ),
                "ce_drawdown_memory_lag_count": float(
                    len(self.drawdown_memory_lags)
                ),
                "ce_fixed_dimensional_causal_basis": 1.0,
                "ce_precomputed_timewise_features": 1.0,
                "ce_train_fixed_feature_normalization": float(
                    uses_fixed_normalization
                ),
                "ce_pooled_time_regression": float(self.pool_time),
                "ce_standardized_feature_clip": (
                    float("nan")
                    if self.standardized_feature_clip is None
                    else float(self.standardized_feature_clip)
                ),
            },
        )

    def _projection_features(self, normalized: torch.Tensor) -> torch.Tensor:
        """Add smooth generic volatility-region gates to the ridge design.

        The input is already standardized either with immutable train-law
        moments or with the current regression batch.  Fixed z-score knots
        therefore add no fitted labels or future information; they only let a
        linear conditional-expectation projection localize an adapted signal
        to calm or volatile observed prefixes.
        """

        if not bool(self.include_volatility_gates):
            return normalized
        names = self.feature_names
        indices = tuple(
            names.index(f"rolling_realized_variance_{window}")
            for window in (10, 20, 32)
            if f"rolling_realized_variance_{window}" in names
        )
        if not indices:
            raise ValueError("volatility gates require rolling variance features")
        values = normalized[:, indices]
        temperature = float(self.volatility_gate_temperature)
        gates = [
            torch.sigmoid((float(center) - values) / temperature)
            for center in self.volatility_gate_centers
        ]
        return torch.cat((normalized, *gates), dim=1)

    def fit_predict(
        self,
        *,
        prefixes_train: torch.Tensor,
        targets_train: torch.Tensor,
        prefixes_eval: torch.Tensor | None = None,
    ) -> ConditionalExpectationResult:
        if prefixes_train.ndim != 3:
            raise ValueError("prefixes_train must have shape (B, k + 1, d)")
        if targets_train.ndim != 2:
            raise ValueError("targets_train must have shape (B, m)")
        if int(prefixes_train.shape[0]) != int(targets_train.shape[0]):
            raise ValueError("prefixes_train and targets_train must have the same batch size")
        if prefixes_eval is not None:
            if prefixes_eval.ndim != 3:
                raise ValueError("prefixes_eval must have shape (B_eval, k + 1, d)")
            if tuple(prefixes_eval.shape[1:]) != tuple(prefixes_train.shape[1:]):
                raise ValueError("prefix train/eval shapes must match after the batch dimension")
        return self.fit_predict_features(
            features_train=self.features_at(prefixes_train),
            targets_train=targets_train,
            features_eval=(
                None
                if prefixes_eval is None
                else self.features_at(prefixes_eval)
            ),
        )


@dataclass(frozen=True)
class CausalVolatilityReferenceKernel:
    """Gaussian reference with a shared causal volatility-aware feature basis."""

    grid: DiscreteTimeGrid
    drift_coefficients: torch.Tensor
    log_variance_coefficients: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    rolling_windows: tuple[int, ...] = (5, 10, 20, 32)
    ewma_half_lives: tuple[int, ...] = (5, 20, 60)
    downside_windows: tuple[int, ...] = (10, 32)
    drawdown_thresholds: tuple[float, ...] = ()
    drawdown_memory_half_lives: tuple[int, ...] = ()
    drawdown_memory_lags: tuple[int, ...] = ()
    include_expanding_moments: bool = False
    persistence_lags: tuple[int, ...] = ()
    leverage_lags: tuple[int, ...] = ()
    local_volatility_windows: tuple[int, ...] = ()
    drawdown_gate_temperature: float = 0.005
    standardized_feature_clip: float | None = None
    selected_variance_ridge: float = 1e-4
    crossfit_folds: int = 3
    crossfit_nll: float = float("nan")
    sigma_min: float = 1e-3
    sigma_max: float | None = None

    def __post_init__(self) -> None:
        rolling = _validate_positive_ints("rolling_windows", self.rolling_windows)
        half_lives = _validate_positive_ints("ewma_half_lives", self.ewma_half_lives)
        downside = _validate_positive_ints("downside_windows", self.downside_windows)
        drawdown_thresholds = tuple(float(value) for value in self.drawdown_thresholds)
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in drawdown_thresholds
        ):
            raise ValueError("drawdown_thresholds must contain positive finite floats")
        if len(set(drawdown_thresholds)) != len(drawdown_thresholds):
            raise ValueError("drawdown_thresholds must not contain duplicates")
        drawdown_memory_half_lives = (
            ()
            if not drawdown_thresholds
            else _validate_positive_ints(
                "drawdown_memory_half_lives",
                self.drawdown_memory_half_lives,
            )
        )
        drawdown_memory_lags = (
            ()
            if not drawdown_thresholds or not self.drawdown_memory_lags
            else _validate_positive_ints(
                "drawdown_memory_lags",
                self.drawdown_memory_lags,
            )
        )
        if any(value >= int(self.grid.num_steps) for value in drawdown_memory_lags):
            raise ValueError("drawdown_memory_lags must be smaller than grid.num_steps")
        if not isinstance(self.include_expanding_moments, bool):
            raise ValueError("include_expanding_moments must be bool")
        persistence_lags = _validate_optional_positive_ints(
            "persistence_lags",
            self.persistence_lags,
        )
        leverage_lags = _validate_optional_nonnegative_ints(
            "leverage_lags",
            self.leverage_lags,
        )
        local_volatility_windows = _validate_optional_positive_ints(
            "local_volatility_windows",
            self.local_volatility_windows,
        )
        if any(value >= int(self.grid.num_steps) for value in persistence_lags):
            raise ValueError("persistence_lags must be smaller than grid.num_steps")
        if any(value >= int(self.grid.num_steps) for value in leverage_lags):
            raise ValueError("leverage_lags must be smaller than grid.num_steps")
        if any(
            value >= int(self.grid.num_steps)
            for value in local_volatility_windows
        ):
            raise ValueError(
                "local_volatility_windows must be smaller than grid.num_steps"
            )
        drawdown_gate_temperature = _require_finite_float(
            "drawdown_gate_temperature",
            self.drawdown_gate_temperature,
        )
        if drawdown_gate_temperature <= 0.0:
            raise ValueError("drawdown_gate_temperature must be > 0")
        feature_clip = _validate_optional_positive_float(
            "standardized_feature_clip",
            self.standardized_feature_clip,
        )
        feature_count = len(
            _feature_names(
                rolling_windows=rolling,
                ewma_half_lives=half_lives,
                downside_windows=downside,
                drawdown_thresholds=drawdown_thresholds,
                drawdown_memory_half_lives=drawdown_memory_half_lives,
                drawdown_memory_lags=drawdown_memory_lags,
                include_expanding_moments=self.include_expanding_moments,
                persistence_lags=persistence_lags,
                leverage_lags=leverage_lags,
                local_volatility_windows=local_volatility_windows,
            )
        )
        expected = (feature_count + 1, 1)
        if tuple(self.drift_coefficients.shape) != expected:
            raise ValueError("drift_coefficients do not match the causal feature basis")
        if tuple(self.log_variance_coefficients.shape) != expected:
            raise ValueError("log_variance_coefficients do not match the causal feature basis")
        if tuple(self.feature_mean.shape) != (1, feature_count):
            raise ValueError("feature_mean does not match the causal feature basis")
        if tuple(self.feature_std.shape) != (1, feature_count):
            raise ValueError("feature_std does not match the causal feature basis")
        sigma_min = _require_finite_float("sigma_min", self.sigma_min)
        sigma_max = (
            None
            if self.sigma_max is None
            else _require_finite_float("sigma_max", self.sigma_max)
        )
        if sigma_min <= 0.0:
            raise ValueError("sigma_min must be > 0")
        if sigma_max is not None and sigma_max <= sigma_min:
            raise ValueError("sigma_max must be > sigma_min")
        object.__setattr__(self, "rolling_windows", rolling)
        object.__setattr__(self, "ewma_half_lives", half_lives)
        object.__setattr__(self, "downside_windows", downside)
        object.__setattr__(self, "drawdown_thresholds", drawdown_thresholds)
        object.__setattr__(
            self,
            "drawdown_memory_half_lives",
            drawdown_memory_half_lives,
        )
        object.__setattr__(
            self,
            "drawdown_memory_lags",
            drawdown_memory_lags,
        )
        object.__setattr__(self, "persistence_lags", persistence_lags)
        object.__setattr__(self, "leverage_lags", leverage_lags)
        object.__setattr__(
            self,
            "local_volatility_windows",
            local_volatility_windows,
        )
        object.__setattr__(
            self,
            "drawdown_gate_temperature",
            drawdown_gate_temperature,
        )
        object.__setattr__(self, "standardized_feature_clip", feature_clip)
        object.__setattr__(self, "sigma_min", sigma_min)
        object.__setattr__(self, "sigma_max", sigma_max)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return _feature_names(
            rolling_windows=self.rolling_windows,
            ewma_half_lives=self.ewma_half_lives,
            downside_windows=self.downside_windows,
            drawdown_thresholds=self.drawdown_thresholds,
            drawdown_memory_half_lives=self.drawdown_memory_half_lives,
            drawdown_memory_lags=self.drawdown_memory_lags,
            include_expanding_moments=self.include_expanding_moments,
            persistence_lags=self.persistence_lags,
            leverage_lags=self.leverage_lags,
            local_volatility_windows=self.local_volatility_windows,
        )

    def _design(self, x_prefix: torch.Tensor, *, step_index: int) -> torch.Tensor:
        features = _causal_features_at_step(
            x_prefix,
            step_index=int(step_index),
            grid=self.grid,
            rolling_windows=self.rolling_windows,
            ewma_half_lives=self.ewma_half_lives,
            downside_windows=self.downside_windows,
            drawdown_thresholds=self.drawdown_thresholds,
            drawdown_memory_half_lives=self.drawdown_memory_half_lives,
            drawdown_memory_lags=self.drawdown_memory_lags,
            include_expanding_moments=self.include_expanding_moments,
            persistence_lags=self.persistence_lags,
            leverage_lags=self.leverage_lags,
            local_volatility_windows=self.local_volatility_windows,
            drawdown_gate_temperature=self.drawdown_gate_temperature,
        )
        mean = self.feature_mean.to(device=x_prefix.device, dtype=x_prefix.dtype)
        std = self.feature_std.to(device=x_prefix.device, dtype=x_prefix.dtype)
        normalized = _soft_clip_standardized(
            (features - mean) / std,
            clip=self.standardized_feature_clip,
        )
        return torch.cat(
            [
                torch.ones(
                    int(features.shape[0]),
                    1,
                    device=x_prefix.device,
                    dtype=x_prefix.dtype,
                ),
                normalized,
            ],
            dim=1,
        )

    def evaluate(
        self,
        x_prefix: torch.Tensor,
        *,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        design = self._design(x_prefix, step_index=int(step_index))
        drift = design @ self.drift_coefficients.to(
            device=x_prefix.device,
            dtype=x_prefix.dtype,
        )
        log_variance = design @ self.log_variance_coefficients.to(
            device=x_prefix.device,
            dtype=x_prefix.dtype,
        )
        log_variance = log_variance.clamp_min(
            2.0 * math.log(float(self.sigma_min))
        )
        if self.sigma_max is not None:
            log_variance = log_variance.clamp_max(
                2.0 * math.log(float(self.sigma_max))
            )
        return drift, torch.exp(0.5 * log_variance)

    def evaluate_all(
        self,
        paths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = _causal_features_all(
            paths,
            grid=self.grid,
            rolling_windows=self.rolling_windows,
            ewma_half_lives=self.ewma_half_lives,
            downside_windows=self.downside_windows,
            drawdown_thresholds=self.drawdown_thresholds,
            drawdown_memory_half_lives=self.drawdown_memory_half_lives,
            drawdown_memory_lags=self.drawdown_memory_lags,
            include_expanding_moments=self.include_expanding_moments,
            persistence_lags=self.persistence_lags,
            leverage_lags=self.leverage_lags,
            local_volatility_windows=self.local_volatility_windows,
            drawdown_gate_temperature=self.drawdown_gate_temperature,
        )
        mean = self.feature_mean.to(device=paths.device, dtype=paths.dtype)
        std = self.feature_std.to(device=paths.device, dtype=paths.dtype)
        normalized = _soft_clip_standardized(
            (features - mean.unsqueeze(1)) / std.unsqueeze(1),
            clip=self.standardized_feature_clip,
        )
        design = torch.cat(
            [
                torch.ones(
                    int(paths.shape[0]),
                    int(self.grid.num_steps),
                    1,
                    device=paths.device,
                    dtype=paths.dtype,
                ),
                normalized,
            ],
            dim=2,
        )
        drift = design @ self.drift_coefficients.to(
            device=paths.device,
            dtype=paths.dtype,
        )
        log_variance = design @ self.log_variance_coefficients.to(
            device=paths.device,
            dtype=paths.dtype,
        )
        log_variance = log_variance.clamp_min(
            2.0 * math.log(float(self.sigma_min))
        )
        if self.sigma_max is not None:
            log_variance = log_variance.clamp_max(
                2.0 * math.log(float(self.sigma_max))
            )
        return drift, torch.exp(0.5 * log_variance)

    def summary(self) -> dict[str, float]:
        return {
            "reference_sigma_min": float(self.sigma_min),
            "reference_sigma_max": (
                float("nan") if self.sigma_max is None else float(self.sigma_max)
            ),
            "reference_feature_dim_max": float(len(self.feature_names)),
            "reference_drift_coeff_l2_mean": float(
                torch.linalg.vector_norm(self.drift_coefficients).item()
            ),
            "reference_log_variance_coeff_l2_mean": float(
                torch.linalg.vector_norm(self.log_variance_coefficients).item()
            ),
            "reference_selected_variance_ridge": float(
                self.selected_variance_ridge
            ),
            "reference_crossfit_folds": float(self.crossfit_folds),
            "reference_crossfit_nll": float(self.crossfit_nll),
            "reference_drawdown_memory_lag_count": float(
                len(self.drawdown_memory_lags)
            ),
            "reference_standardized_feature_clip": (
                float("nan")
                if self.standardized_feature_clip is None
                else float(self.standardized_feature_clip)
            ),
        }


@dataclass(frozen=True)
class ShrunkCausalVolatilityReferenceKernel:
    """Log-space blend of a causal reference and a stable fitted anchor.

    The causal weight controls only the volatility reference.  The drift is
    taken from ``causal_reference`` for completeness, although the
    specific-entropy control uses only the reference volatility.
    """

    causal_reference: CausalVolatilityReferenceKernel
    anchor_reference: LocalGaussianReferenceKernel
    causal_log_variance_weight: float
    log_variance_offset: float = 0.0
    sigma_min: float = 1e-3
    sigma_max: float | None = None

    def __post_init__(self) -> None:
        weight = _require_finite_float(
            "causal_log_variance_weight",
            self.causal_log_variance_weight,
        )
        offset = _require_finite_float(
            "log_variance_offset",
            self.log_variance_offset,
        )
        sigma_min = _require_finite_float("sigma_min", self.sigma_min)
        sigma_max = (
            None
            if self.sigma_max is None
            else _require_finite_float("sigma_max", self.sigma_max)
        )
        if not 0.0 <= weight <= 1.0:
            raise ValueError("causal_log_variance_weight must lie in [0, 1]")
        if sigma_min <= 0.0:
            raise ValueError("sigma_min must be > 0")
        if sigma_max is not None and sigma_max <= sigma_min:
            raise ValueError("sigma_max must be > sigma_min")
        causal_grid = self.causal_reference.grid
        anchor_grid = self.anchor_reference.grid
        if int(causal_grid.num_steps) != int(anchor_grid.num_steps):
            raise ValueError("causal and anchor references must use the same grid")
        if abs(float(causal_grid.dt) - float(anchor_grid.dt)) > 1e-12:
            raise ValueError("causal and anchor references must use the same grid")
        object.__setattr__(self, "causal_log_variance_weight", weight)
        object.__setattr__(self, "log_variance_offset", offset)
        object.__setattr__(self, "sigma_min", sigma_min)
        object.__setattr__(self, "sigma_max", sigma_max)

    @property
    def grid(self) -> DiscreteTimeGrid:
        return self.causal_reference.grid

    def evaluate(
        self,
        x_prefix: torch.Tensor,
        *,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        drift, causal_sigma = self.causal_reference.evaluate(
            x_prefix,
            step_index=int(step_index),
        )
        _, anchor_sigma = self.anchor_reference.evaluate(
            x_prefix,
            step_index=int(step_index),
        )
        weight = float(self.causal_log_variance_weight)
        log_variance = (
            weight * (2.0 * torch.log(causal_sigma))
            + (1.0 - weight) * (2.0 * torch.log(anchor_sigma))
            + float(self.log_variance_offset)
        )
        log_variance = log_variance.clamp_min(
            2.0 * math.log(float(self.sigma_min))
        )
        if self.sigma_max is not None:
            log_variance = log_variance.clamp_max(
                2.0 * math.log(float(self.sigma_max))
            )
        return drift, torch.exp(0.5 * log_variance)

    def summary(self) -> dict[str, float]:
        summary = self.causal_reference.summary()
        summary.update(
            {
                "reference_causal_log_variance_weight": float(
                    self.causal_log_variance_weight
                ),
                "reference_log_variance_offset": float(
                    self.log_variance_offset
                ),
                "reference_sigma_min": float(self.sigma_min),
                "reference_sigma_max": (
                    float("nan")
                    if self.sigma_max is None
                    else float(self.sigma_max)
                ),
            }
        )
        return summary


def _reference_rollout(
    reference: ShrunkCausalVolatilityReferenceKernel,
    *,
    noise: torch.Tensor,
    x0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise.ndim != 3 or int(noise.shape[2]) != 1:
        raise ValueError("noise must have shape (B, N, 1)")
    if int(noise.shape[1]) != int(reference.grid.num_steps):
        raise ValueError("noise must align with the reference grid")
    if tuple(x0.shape) != (int(noise.shape[0]), 1):
        raise ValueError("x0 must have shape (B, 1)")
    points = [x0]
    sigmas = []
    sqrt_dt = math.sqrt(float(reference.grid.dt))
    with torch.no_grad():
        for step in range(int(reference.grid.num_steps)):
            prefix = torch.stack(points, dim=1)
            _, sigma = reference.evaluate(prefix, step_index=step)
            points.append(points[-1] + sigma * sqrt_dt * noise[:, step, :])
            sigmas.append(sigma)
    return torch.stack(points, dim=1), torch.stack(sigmas, dim=1)


def _rollout_statistics(
    paths: torch.Tensor,
    sigmas: torch.Tensor | None,
    *,
    dt: float,
    clustering_lags: tuple[int, ...],
) -> dict[str, object]:
    increments = paths[:, 1:, :] - paths[:, :-1, :]
    return_std = float(increments.std(unbiased=False).item())
    path_rv = torch.sqrt(increments.pow(2).mean(dim=1) / float(dt)).reshape(-1)
    rv_quantiles = torch.quantile(
        path_rv,
        path_rv.new_tensor((0.10, 0.50, 0.90)),
    )
    squared = increments[..., 0].pow(2)
    correlations: list[float] = []
    for lag in clustering_lags:
        if lag >= int(squared.shape[1]):
            raise ValueError("clustering_lags must be smaller than num_steps")
        # Squared intraday returns can have variance far below float32 eps.
        # Computing the correlation in float64 and guarding only a genuinely
        # zero denominator keeps this statistic scale invariant.  Clamping the
        # denominator at float32 eps silently drove real-data correlations to
        # almost zero and made reference selection blind to clustering.
        left = squared[:, :-lag].reshape(-1).to(dtype=torch.float64)
        right = squared[:, lag:].reshape(-1).to(dtype=torch.float64)
        left = left - left.mean()
        right = right - right.mean()
        denominator = torch.sqrt(left.pow(2).mean() * right.pow(2).mean())
        correlation = (
            left.new_tensor(0.0)
            if float(denominator.item()) <= torch.finfo(torch.float64).tiny
            else (left * right).mean() / denominator
        )
        correlations.append(float(correlation.item()))
    path_rv_mean = float(path_rv.mean().item())
    path_rv_std = float(path_rv.std(unbiased=False).item())
    path_rv_quantile_spread = float((rv_quantiles[2] - rv_quantiles[0]).item())
    result: dict[str, object] = {
        "return_std": return_std,
        "path_rv_quantiles": [float(value) for value in rv_quantiles.tolist()],
        "path_rv_mean": path_rv_mean,
        "path_rv_std": path_rv_std,
        "path_rv_cv": path_rv_std / max(path_rv_mean, torch.finfo(torch.float64).tiny),
        "path_rv_quantile_spread": path_rv_quantile_spread,
        "squared_return_correlations": correlations,
    }
    if sigmas is not None:
        flat_sigma = sigmas.reshape(-1)
        result.update(
            {
                "sigma_mean": float(flat_sigma.mean().item()),
                "sigma_std": float(flat_sigma.std(unbiased=False).item()),
                "sigma_max_active_fraction": float(
                    (
                        flat_sigma
                        >= float(flat_sigma.max().item()) - 1e-7
                    ).float().mean().item()
                ),
            }
        )
    return result


def calibrate_shrunk_causal_reference_kernel(
    *,
    target_paths: torch.Tensor,
    causal_reference: CausalVolatilityReferenceKernel,
    anchor_reference: LocalGaussianReferenceKernel,
    causal_weights: Sequence[float] = (
        0.0,
        0.25,
        0.4,
        0.55,
        0.7,
        0.75,
        0.8,
        0.85,
        1.0,
    ),
    calibration_paths: int = 4_096,
    calibration_seed: int = 271_828,
    offset_steps: int = 3,
    offset_relaxation: float = 1.0,
    return_std_tolerance: float = 0.02,
    sigma_std_floor: float = 0.045,
    sigma_cap_fraction_limit: float = 0.01,
    path_rv_std_ratio_min: float = 0.0,
    path_rv_std_ratio_max: float = math.inf,
    path_rv_quantile_spread_ratio_min: float = 0.0,
    path_rv_quantile_spread_ratio_max: float = math.inf,
    clustering_lags: Sequence[int] = (1, 5, 20),
    sigma_min: float = 1e-3,
    sigma_max: float | None = None,
) -> tuple[ShrunkCausalVolatilityReferenceKernel, dict[str, object]]:
    """Select a stable causal/anchor blend using training paths only.

    Each candidate is recursively rolled out with common Gaussian noise.  Its
    log-variance offset is iteratively calibrated to the training return
    standard deviation.  Among stable candidates, selection matches the
    training distribution of pathwise realized volatility and squared-return
    autocorrelations.  No validation or test path is used.
    """

    if target_paths.ndim != 3 or tuple(target_paths.shape[1:]) != (
        int(causal_reference.grid.num_steps) + 1,
        1,
    ):
        raise ValueError("target_paths must have shape (B, N + 1, 1)")
    weights = tuple(float(value) for value in causal_weights)
    if not weights or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in weights
    ):
        raise ValueError("causal_weights must be finite values in [0, 1]")
    if len(set(weights)) != len(weights):
        raise ValueError("causal_weights must not contain duplicates")
    path_count = int(calibration_paths)
    if path_count < 2:
        raise ValueError("calibration_paths must be >= 2")
    steps = int(offset_steps)
    if steps < 1:
        raise ValueError("offset_steps must be >= 1")
    relaxation = _require_finite_float(
        "offset_relaxation",
        offset_relaxation,
    )
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("offset_relaxation must lie in (0, 1]")
    tolerance = _require_finite_float(
        "return_std_tolerance",
        return_std_tolerance,
    )
    sigma_floor = _require_finite_float("sigma_std_floor", sigma_std_floor)
    cap_limit = _require_finite_float(
        "sigma_cap_fraction_limit",
        sigma_cap_fraction_limit,
    )
    rv_std_ratio_min = _require_finite_float(
        "path_rv_std_ratio_min",
        path_rv_std_ratio_min,
    )
    rv_std_ratio_max = float(path_rv_std_ratio_max)
    rv_spread_ratio_min = _require_finite_float(
        "path_rv_quantile_spread_ratio_min",
        path_rv_quantile_spread_ratio_min,
    )
    rv_spread_ratio_max = float(path_rv_quantile_spread_ratio_max)
    if math.isnan(rv_std_ratio_max) or math.isnan(rv_spread_ratio_max):
        raise ValueError("path-RV ratio upper bounds must not be NaN")
    if (
        tolerance < 0.0
        or sigma_floor < 0.0
        or not 0.0 <= cap_limit <= 1.0
        or rv_std_ratio_min < 0.0
        or rv_std_ratio_max < rv_std_ratio_min
        or rv_spread_ratio_min < 0.0
        or rv_spread_ratio_max < rv_spread_ratio_min
    ):
        raise ValueError("invalid calibration stability thresholds")
    lags = _validate_positive_ints("clustering_lags", clustering_lags)
    if any(lag >= int(causal_reference.grid.num_steps) for lag in lags):
        raise ValueError("clustering_lags must be smaller than num_steps")

    device = target_paths.device
    dtype = target_paths.dtype
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(calibration_seed))
    noise = torch.randn(
        path_count,
        int(causal_reference.grid.num_steps),
        1,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x0_value = target_paths[:, 0, :].mean(dim=0, keepdim=True)
    x0 = x0_value.expand(path_count, -1)
    target_stats = _rollout_statistics(
        target_paths,
        None,
        dt=float(causal_reference.grid.dt),
        clustering_lags=lags,
    )
    target_return_std = float(target_stats["return_std"])
    target_path_rv_std = float(target_stats["path_rv_std"])
    target_path_rv_quantile_spread = float(target_stats["path_rv_quantile_spread"])
    target_rv_quantiles = torch.tensor(
        target_stats["path_rv_quantiles"],
        dtype=torch.float64,
    )
    target_correlations = torch.tensor(
        target_stats["squared_return_correlations"],
        dtype=torch.float64,
    )

    candidates: list[dict[str, object]] = []
    candidate_references: list[ShrunkCausalVolatilityReferenceKernel] = []
    for weight in weights:
        offset = 0.0
        for _ in range(steps):
            reference = ShrunkCausalVolatilityReferenceKernel(
                causal_reference=causal_reference,
                anchor_reference=anchor_reference,
                causal_log_variance_weight=weight,
                log_variance_offset=offset,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
            paths, _ = _reference_rollout(reference, noise=noise, x0=x0)
            current_std = float(
                (paths[:, 1:, :] - paths[:, :-1, :])
                .std(unbiased=False)
                .item()
            )
            offset += float(relaxation) * 2.0 * math.log(
                target_return_std / max(current_std, torch.finfo(dtype).tiny)
            )
        reference = ShrunkCausalVolatilityReferenceKernel(
            causal_reference=causal_reference,
            anchor_reference=anchor_reference,
            causal_log_variance_weight=weight,
            log_variance_offset=offset,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        paths, sigmas = _reference_rollout(reference, noise=noise, x0=x0)
        stats = _rollout_statistics(
            paths,
            sigmas,
            dt=float(causal_reference.grid.dt),
            clustering_lags=lags,
        )
        rv_quantiles = torch.tensor(
            stats["path_rv_quantiles"],
            dtype=torch.float64,
        )
        correlations = torch.tensor(
            stats["squared_return_correlations"],
            dtype=torch.float64,
        )
        rv_score = float(
            (
                torch.log(rv_quantiles / target_rv_quantiles)
                .pow(2)
                .mean()
            ).item()
        )
        correlation_scale = target_correlations.abs().clamp_min(0.05)
        clustering_score = float(
            (
                ((correlations - target_correlations) / correlation_scale)
                .pow(2)
                .mean()
            ).item()
        )
        score = rv_score + 0.25 * clustering_score
        return_relative_error = abs(
            float(stats["return_std"]) / target_return_std - 1.0
        )
        cap_fraction = float(stats["sigma_max_active_fraction"])
        path_rv_std_ratio = float(stats["path_rv_std"]) / max(
            target_path_rv_std,
            torch.finfo(torch.float64).tiny,
        )
        path_rv_quantile_spread_ratio = float(
            stats["path_rv_quantile_spread"]
        ) / max(
            target_path_rv_quantile_spread,
            torch.finfo(torch.float64).tiny,
        )
        stable = bool(
            return_relative_error <= tolerance
            and float(stats["sigma_std"]) >= sigma_floor
            and cap_fraction <= cap_limit
            and rv_std_ratio_min <= path_rv_std_ratio <= rv_std_ratio_max
            and rv_spread_ratio_min
            <= path_rv_quantile_spread_ratio
            <= rv_spread_ratio_max
        )
        candidates.append(
            {
                "causal_log_variance_weight": weight,
                "log_variance_offset": offset,
                "return_std_relative_error": return_relative_error,
                "rv_distribution_score": rv_score,
                "clustering_score": clustering_score,
                "selection_score": score,
                "path_rv_std_ratio": path_rv_std_ratio,
                "path_rv_quantile_spread_ratio": path_rv_quantile_spread_ratio,
                "stable": stable,
                **stats,
            }
        )
        candidate_references.append(reference)

    eligible = [
        index
        for index, candidate in enumerate(candidates)
        if bool(candidate["stable"])
    ]
    if not eligible:
        raise RuntimeError(
            "no shrinkage candidate satisfies the train-only stability constraints"
        )
    selected_index = min(
        eligible,
        key=lambda index: float(candidates[index]["selection_score"]),
    )
    diagnostics: dict[str, object] = {
        "selection_uses_training_paths_only": True,
        "calibration_seed": int(calibration_seed),
        "calibration_paths": path_count,
        "offset_steps": steps,
        "offset_relaxation": relaxation,
        "clustering_lags": list(lags),
        "target": target_stats,
        "stability_thresholds": {
            "return_std_relative_tolerance": tolerance,
            "sigma_std_floor": sigma_floor,
            "sigma_cap_fraction_limit": cap_limit,
            "path_rv_std_ratio_min": rv_std_ratio_min,
            "path_rv_std_ratio_max": rv_std_ratio_max,
            "path_rv_quantile_spread_ratio_min": rv_spread_ratio_min,
            "path_rv_quantile_spread_ratio_max": rv_spread_ratio_max,
        },
        "candidates": candidates,
        "selected_index": selected_index,
        "selected": candidates[selected_index],
    }
    return candidate_references[selected_index], diagnostics


def fit_causal_volatility_reference_kernel(
    *,
    target_paths: torch.Tensor,
    grid: DiscreteTimeGrid,
    rolling_windows: Sequence[int] = (5, 10, 20, 32),
    ewma_half_lives: Sequence[int] = (5, 20, 60),
    downside_windows: Sequence[int] = (10, 32),
    drawdown_thresholds: Sequence[float] = (),
    drawdown_memory_half_lives: Sequence[int] = (20, 60),
    drawdown_memory_lags: Sequence[int] = (),
    include_expanding_moments: bool = False,
    persistence_lags: Sequence[int] = (),
    leverage_lags: Sequence[int] = (),
    local_volatility_windows: Sequence[int] = (),
    drawdown_gate_temperature: float = 0.005,
    standardized_feature_clip: float | None = None,
    drift_ridge: float = 1e-3,
    variance_ridges: Sequence[float] = (1e-5, 1e-4, 1e-3),
    crossfit_folds: int = 3,
    crossfit_seed: int = 1701,
    sigma_min: float = 1e-3,
    sigma_max: float | None = None,
    feature_eps: float = 1e-6,
    newton_steps: int = 30,
    newton_tolerance: float = 1e-7,
) -> CausalVolatilityReferenceKernel:
    """Fit a model-free causal reference by cross-fitted Gaussian likelihood."""

    if target_paths.ndim != 3 or tuple(target_paths.shape[1:]) != (
        int(grid.num_steps) + 1,
        1,
    ):
        raise ValueError("target_paths must have shape (B, N + 1, 1)")
    path_count = int(target_paths.shape[0])
    folds = int(crossfit_folds)
    if folds < 2 or folds > path_count:
        raise ValueError("crossfit_folds must lie between 2 and the number of paths")
    rolling = _validate_positive_ints("rolling_windows", rolling_windows)
    half_lives = _validate_positive_ints("ewma_half_lives", ewma_half_lives)
    downside = _validate_positive_ints("downside_windows", downside_windows)
    drawdown_threshold_values = tuple(float(value) for value in drawdown_thresholds)
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in drawdown_threshold_values
    ):
        raise ValueError("drawdown_thresholds must contain positive finite floats")
    if len(set(drawdown_threshold_values)) != len(drawdown_threshold_values):
        raise ValueError("drawdown_thresholds must not contain duplicates")
    drawdown_half_lives = (
        ()
        if not drawdown_threshold_values
        else _validate_positive_ints(
            "drawdown_memory_half_lives",
            drawdown_memory_half_lives,
        )
    )
    drawdown_lags = (
        ()
        if not drawdown_threshold_values or not drawdown_memory_lags
        else _validate_positive_ints(
            "drawdown_memory_lags",
            drawdown_memory_lags,
        )
    )
    if any(value >= int(grid.num_steps) for value in drawdown_lags):
        raise ValueError("drawdown_memory_lags must be smaller than grid.num_steps")
    if not isinstance(include_expanding_moments, bool):
        raise ValueError("include_expanding_moments must be bool")
    persistence = _validate_optional_positive_ints(
        "persistence_lags",
        persistence_lags,
    )
    leverage = _validate_optional_nonnegative_ints(
        "leverage_lags",
        leverage_lags,
    )
    local_windows = _validate_optional_positive_ints(
        "local_volatility_windows",
        local_volatility_windows,
    )
    if any(value >= int(grid.num_steps) for value in persistence):
        raise ValueError("persistence_lags must be smaller than grid.num_steps")
    if any(value >= int(grid.num_steps) for value in leverage):
        raise ValueError("leverage_lags must be smaller than grid.num_steps")
    if any(value >= int(grid.num_steps) for value in local_windows):
        raise ValueError(
            "local_volatility_windows must be smaller than grid.num_steps"
        )
    drawdown_temperature = _require_finite_float(
        "drawdown_gate_temperature",
        drawdown_gate_temperature,
    )
    if drawdown_temperature <= 0.0:
        raise ValueError("drawdown_gate_temperature must be > 0")
    feature_clip = _validate_optional_positive_float(
        "standardized_feature_clip",
        standardized_feature_clip,
    )
    ridges = tuple(float(value) for value in variance_ridges)
    if not ridges or any(not math.isfinite(value) or value < 0.0 for value in ridges):
        raise ValueError("variance_ridges must be non-empty nonnegative finite floats")
    drift_ridge = _require_finite_float("drift_ridge", drift_ridge)
    sigma_min = _require_finite_float("sigma_min", sigma_min)
    sigma_max = (
        None if sigma_max is None else _require_finite_float("sigma_max", sigma_max)
    )
    feature_eps = _require_finite_float("feature_eps", feature_eps)
    if drift_ridge < 0.0 or sigma_min <= 0.0 or feature_eps <= 0.0:
        raise ValueError("invalid causal reference fitting parameters")
    if sigma_max is not None and sigma_max <= sigma_min:
        raise ValueError("sigma_max must be greater than sigma_min")

    features = _causal_features_all(
        target_paths,
        grid=grid,
        rolling_windows=rolling,
        ewma_half_lives=half_lives,
        downside_windows=downside,
        drawdown_thresholds=drawdown_threshold_values,
        drawdown_memory_half_lives=drawdown_half_lives,
        drawdown_memory_lags=drawdown_lags,
        include_expanding_moments=include_expanding_moments,
        persistence_lags=persistence,
        leverage_lags=leverage,
        local_volatility_windows=local_windows,
        drawdown_gate_temperature=drawdown_temperature,
    )
    feature_mean = features.mean(dim=(0, 1), keepdim=False).unsqueeze(0)
    feature_std = features.std(
        dim=(0, 1),
        unbiased=False,
    ).clamp_min(float(feature_eps)).unsqueeze(0)
    normalized = _soft_clip_standardized(
        (features - feature_mean.unsqueeze(1)) / feature_std.unsqueeze(1),
        clip=feature_clip,
    )
    design = torch.cat(
        [
            torch.ones(
                path_count,
                int(grid.num_steps),
                1,
                device=target_paths.device,
                dtype=target_paths.dtype,
            ),
            normalized,
        ],
        dim=2,
    )
    increments = target_paths[:, 1:, :] - target_paths[:, :-1, :]
    drift_target = increments / float(grid.dt)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(crossfit_seed))
    permutation = torch.randperm(path_count, generator=generator)
    fold_id_cpu = torch.empty(path_count, dtype=torch.long)
    fold_id_cpu[permutation] = torch.arange(path_count) % folds
    fold_ids = fold_id_cpu.to(target_paths.device)
    residuals = torch.empty_like(increments)
    for fold in range(folds):
        train = fold_ids != fold
        valid = fold_ids == fold
        drift_coefficients = fit_ridge(
            design=design[train].reshape(-1, int(design.shape[-1])),
            target=drift_target[train].reshape(-1, 1),
            ridge=float(drift_ridge),
        )
        residuals[valid] = (
            increments[valid]
            - float(grid.dt)
            * (
                design[valid].reshape(-1, int(design.shape[-1]))
                @ drift_coefficients
            ).reshape(-1, int(grid.num_steps), 1)
        )

    realized_variance = residuals.pow(2) / float(grid.dt)
    fold_nll_by_ridge: dict[float, list[float]] = {ridge: [] for ridge in ridges}
    for ridge in ridges:
        for fold in range(folds):
            train = fold_ids != fold
            valid = fold_ids == fold
            coefficients = _fit_log_variance_newton(
                design=design[train].reshape(-1, int(design.shape[-1])),
                realized_variance=realized_variance[train].reshape(-1, 1),
                ridge=float(ridge),
                max_steps=int(newton_steps),
                tolerance=float(newton_tolerance),
            )
            validation_nll = _gaussian_nll(
                design[valid].reshape(-1, int(design.shape[-1])),
                realized_variance[valid].reshape(-1, 1),
                coefficients,
            )
            fold_nll_by_ridge[ridge].append(float(validation_nll.item()))
    selected_ridge = min(
        ridges,
        key=lambda value: sum(fold_nll_by_ridge[value])
        / len(fold_nll_by_ridge[value]),
    )
    crossfit_nll = sum(fold_nll_by_ridge[selected_ridge]) / len(
        fold_nll_by_ridge[selected_ridge]
    )
    final_drift = fit_ridge(
        design=design.reshape(-1, int(design.shape[-1])),
        target=drift_target.reshape(-1, 1),
        ridge=float(drift_ridge),
    )
    final_log_variance = _fit_log_variance_newton(
        design=design.reshape(-1, int(design.shape[-1])),
        realized_variance=realized_variance.reshape(-1, 1),
        ridge=float(selected_ridge),
        max_steps=int(newton_steps),
        tolerance=float(newton_tolerance),
    )
    return CausalVolatilityReferenceKernel(
        grid=grid,
        drift_coefficients=final_drift.detach(),
        log_variance_coefficients=final_log_variance.detach(),
        feature_mean=feature_mean.detach(),
        feature_std=feature_std.detach(),
        rolling_windows=rolling,
        ewma_half_lives=half_lives,
        downside_windows=downside,
        drawdown_thresholds=drawdown_threshold_values,
        drawdown_memory_half_lives=drawdown_half_lives,
        drawdown_memory_lags=drawdown_lags,
        include_expanding_moments=include_expanding_moments,
        persistence_lags=persistence,
        leverage_lags=leverage,
        local_volatility_windows=local_windows,
        drawdown_gate_temperature=drawdown_temperature,
        standardized_feature_clip=feature_clip,
        selected_variance_ridge=float(selected_ridge),
        crossfit_folds=folds,
        crossfit_nll=float(crossfit_nll),
        sigma_min=float(sigma_min),
        sigma_max=sigma_max,
    )


__all__ = [
    "CausalPathFeatureRidgeConditionalExpectation",
    "CausalVolatilityReferenceKernel",
    "ShrunkCausalVolatilityReferenceKernel",
    "calibrate_shrunk_causal_reference_kernel",
    "fit_causal_volatility_reference_kernel",
]
