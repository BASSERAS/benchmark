from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


StateSource = Literal["current", "delayed"]
ControlKind = Literal["return_only", "sigma_supervised", "exact"]


def observable_drawdown_memory(
    log_paths: np.ndarray,
    *,
    threshold: float,
    half_life: float,
) -> np.ndarray:
    """Recover the simulator's memory state from observed log-price paths."""

    values = np.asarray(log_paths, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("log_paths must have shape (paths, points >= 2)")
    if not np.isfinite(values).all():
        raise ValueError("log_paths must be finite")
    if not math.isfinite(float(threshold)) or float(threshold) <= 0.0:
        raise ValueError("threshold must be positive and finite")
    if not math.isfinite(float(half_life)) or float(half_life) <= 0.0:
        raise ValueError("half_life must be positive and finite")

    path_count, point_count = values.shape
    num_steps = point_count - 1
    running_max = np.zeros(path_count, dtype=np.float64)
    memory = np.zeros(path_count, dtype=np.float64)
    history = np.empty((path_count, num_steps), dtype=np.float64)
    decay = math.exp(-math.log(2.0) / float(half_life))
    for step in range(num_steps):
        drawdown = running_max - values[:, step]
        memory = np.maximum(
            decay * memory,
            (drawdown >= float(threshold)).astype(np.float64),
        )
        history[:, step] = memory
        running_max = np.maximum(running_max, values[:, step + 1])
    return history


def delayed_state(memory_history: np.ndarray, *, delay: int) -> np.ndarray:
    """Return M_{t-delay}, with zero state before the delay becomes available."""

    memory = np.asarray(memory_history, dtype=np.float64)
    if memory.ndim != 2:
        raise ValueError("memory_history must have shape (paths, steps)")
    lag = int(delay)
    if lag < 1 or lag >= memory.shape[1]:
        raise ValueError("delay must lie in [1, steps - 1]")
    result = np.zeros_like(memory)
    result[:, lag:] = memory[:, :-lag]
    return result


def _polynomial_design(state: np.ndarray) -> np.ndarray:
    z = np.asarray(state, dtype=np.float64).reshape(-1)
    return np.column_stack([np.ones_like(z), z, z * z])


def _linear_design(state: np.ndarray) -> np.ndarray:
    z = np.asarray(state, dtype=np.float64).reshape(-1)
    return np.column_stack([np.ones_like(z), z])


def _ridge_solve(
    design: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("design and target must share their sample dimension")
    penalty = np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        x.T @ x + float(ridge) * penalty,
        x.T @ y,
    )


@dataclass(frozen=True)
class DrawdownOracleControl:
    """A diagnostic control law evaluated inside the ordinary Euler dynamics."""

    kind: ControlKind
    state_source: StateSource
    drift_coefficients: np.ndarray | None = None
    variance_coefficients: np.ndarray | None = None
    sigma_coefficients: np.ndarray | None = None
    sigma_low: float | None = None
    sigma_high: float | None = None
    sigma_min: float = 1e-3
    sigma_max: float = 0.8

    def __post_init__(self) -> None:
        if self.kind not in {"return_only", "sigma_supervised", "exact"}:
            raise ValueError("unsupported control kind")
        if self.state_source not in {"current", "delayed"}:
            raise ValueError("state_source must be current or delayed")
        if not 0.0 < float(self.sigma_min) < float(self.sigma_max):
            raise ValueError("sigma bounds must satisfy 0 < min < max")
        if self.kind == "return_only":
            if self.drift_coefficients is None or self.variance_coefficients is None:
                raise ValueError("return-only controls require drift and variance fits")
        elif self.kind == "sigma_supervised":
            if self.sigma_coefficients is None:
                raise ValueError("sigma-supervised controls require a sigma fit")
        else:
            if self.sigma_low is None or self.sigma_high is None:
                raise ValueError("exact controls require sigma_low and sigma_high")

    def evaluate(
        self,
        *,
        current_memory: np.ndarray,
        delayed_memory: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = (
            np.asarray(current_memory, dtype=np.float64)
            if self.state_source == "current"
            else np.asarray(delayed_memory, dtype=np.float64)
        )
        if self.kind == "exact":
            sigma = float(self.sigma_low) + (
                float(self.sigma_high) - float(self.sigma_low)
            ) * state
            alpha = -0.5 * sigma * sigma
        elif self.kind == "sigma_supervised":
            sigma = (
                _linear_design(state)
                @ np.asarray(self.sigma_coefficients, dtype=np.float64)
            ).reshape(state.shape)
            sigma = np.clip(sigma, float(self.sigma_min), float(self.sigma_max))
            alpha = -0.5 * sigma * sigma
        else:
            design = _polynomial_design(state)
            alpha = (
                design @ np.asarray(self.drift_coefficients, dtype=np.float64)
            ).reshape(state.shape)
            variance = (
                design @ np.asarray(self.variance_coefficients, dtype=np.float64)
            ).reshape(state.shape)
            sigma = np.sqrt(
                np.clip(
                    variance,
                    float(self.sigma_min) ** 2,
                    float(self.sigma_max) ** 2,
                )
            )
        return alpha, sigma

    def summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "state_source": self.state_source,
            "drift_coefficients": (
                None
                if self.drift_coefficients is None
                else np.asarray(self.drift_coefficients).tolist()
            ),
            "variance_coefficients": (
                None
                if self.variance_coefficients is None
                else np.asarray(self.variance_coefficients).tolist()
            ),
            "sigma_coefficients": (
                None
                if self.sigma_coefficients is None
                else np.asarray(self.sigma_coefficients).tolist()
            ),
            "sigma_low": self.sigma_low,
            "sigma_high": self.sigma_high,
            "sigma_min": float(self.sigma_min),
            "sigma_max": float(self.sigma_max),
        }


def fit_return_only_oracle(
    *,
    log_paths: np.ndarray,
    state: np.ndarray,
    dt: float,
    state_source: StateSource,
    ridge: float = 1e-6,
    sigma_min: float = 1e-3,
    sigma_max: float = 0.8,
) -> DrawdownOracleControl:
    """Fit drift and conditional variance using returns and observable state only."""

    values = np.asarray(log_paths, dtype=np.float64)
    z = np.asarray(state, dtype=np.float64)
    if values.ndim != 2 or z.shape != (values.shape[0], values.shape[1] - 1):
        raise ValueError("state must align with the log-path increments")
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be positive and finite")
    design = _polynomial_design(z)
    increments = np.diff(values, axis=1).reshape(-1)
    drift = _ridge_solve(
        design,
        increments / float(dt),
        ridge=float(ridge),
    )
    residual = increments - float(dt) * (design @ drift)
    variance = _ridge_solve(
        design,
        residual * residual / float(dt),
        ridge=float(ridge),
    )
    return DrawdownOracleControl(
        kind="return_only",
        state_source=state_source,
        drift_coefficients=drift,
        variance_coefficients=variance,
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
    )


def fit_sigma_supervised_oracle(
    *,
    state: np.ndarray,
    sigma_history: np.ndarray,
    state_source: StateSource = "delayed",
    ridge: float = 1e-8,
    sigma_min: float = 1e-3,
    sigma_max: float = 0.8,
) -> DrawdownOracleControl:
    """Fit volatility from privileged train-sigma labels as an upper-bound test."""

    z = np.asarray(state, dtype=np.float64)
    sigma = np.asarray(sigma_history, dtype=np.float64)
    if z.ndim != 2 or sigma.shape != z.shape:
        raise ValueError("state and sigma_history must share shape (paths, steps)")
    coefficients = _ridge_solve(
        _linear_design(z),
        sigma.reshape(-1),
        ridge=float(ridge),
    )
    return DrawdownOracleControl(
        kind="sigma_supervised",
        state_source=state_source,
        sigma_coefficients=coefficients,
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
    )


def exact_observable_oracle(
    *,
    sigma_low: float,
    sigma_high: float,
    sigma_min: float = 1e-3,
    sigma_max: float = 0.8,
) -> DrawdownOracleControl:
    return DrawdownOracleControl(
        kind="exact",
        state_source="delayed",
        sigma_low=float(sigma_low),
        sigma_high=float(sigma_high),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
    )


def rollout_oracle(
    *,
    control: DrawdownOracleControl,
    num_paths: int,
    sequence_length: int,
    seed: int,
    dt: float,
    s0: float,
    drawdown_threshold: float,
    memory_half_life: float,
    response_delay: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out oracle controls through the benchmark's Euler state equation."""

    if int(num_paths) < 1 or int(sequence_length) < 3:
        raise ValueError("num_paths must be positive and sequence_length >= 3")
    num_steps = int(sequence_length) - 1
    lag = int(response_delay)
    if lag < 1 or lag >= num_steps:
        raise ValueError("response_delay must lie in [1, sequence_length - 2]")
    rng = np.random.default_rng(int(seed))
    log_paths = np.zeros((int(num_paths), int(sequence_length)), dtype=np.float64)
    memory = np.zeros(int(num_paths), dtype=np.float64)
    memory_history = np.zeros((int(num_paths), num_steps), dtype=np.float64)
    running_max = np.zeros(int(num_paths), dtype=np.float64)
    sigma_history = np.empty((int(num_paths), num_steps), dtype=np.float64)
    decay = math.exp(-math.log(2.0) / float(memory_half_life))

    for step in range(num_steps):
        drawdown = running_max - log_paths[:, step]
        memory = np.maximum(
            decay * memory,
            (drawdown >= float(drawdown_threshold)).astype(np.float64),
        )
        memory_history[:, step] = memory
        delayed_memory = (
            memory_history[:, step - lag]
            if step >= lag
            else np.zeros(int(num_paths), dtype=np.float64)
        )
        alpha, sigma = control.evaluate(
            current_memory=memory,
            delayed_memory=delayed_memory,
        )
        sigma_history[:, step] = sigma
        log_paths[:, step + 1] = (
            log_paths[:, step]
            + alpha * float(dt)
            + sigma * math.sqrt(float(dt)) * rng.standard_normal(int(num_paths))
        )
        running_max = np.maximum(running_max, log_paths[:, step + 1])

    return float(s0) * np.exp(log_paths), sigma_history


def control_prediction_metrics(
    *,
    control: DrawdownOracleControl,
    current_memory: np.ndarray,
    delayed_memory: np.ndarray,
    target_sigma: np.ndarray,
) -> dict[str, float]:
    alpha, sigma = control.evaluate(
        current_memory=current_memory,
        delayed_memory=delayed_memory,
    )
    truth = np.asarray(target_sigma, dtype=np.float64)
    if sigma.shape != truth.shape:
        raise ValueError("predicted and target sigma must share shape")
    sigma_flat = sigma.reshape(-1)
    truth_flat = truth.reshape(-1)
    correlation = float(np.corrcoef(sigma_flat, truth_flat)[0, 1])
    true_alpha = -0.5 * truth * truth
    return {
        "sigma_rmse": float(np.sqrt(np.mean((sigma - truth) ** 2))),
        "sigma_mae": float(np.mean(np.abs(sigma - truth))),
        "sigma_correlation": correlation,
        "drift_rmse_against_structural_truth": float(
            np.sqrt(np.mean((alpha - true_alpha) ** 2))
        ),
    }


__all__ = [
    "DrawdownOracleControl",
    "control_prediction_metrics",
    "delayed_state",
    "exact_observable_oracle",
    "fit_return_only_oracle",
    "fit_sigma_supervised_oracle",
    "observable_drawdown_memory",
    "rollout_oracle",
]
