from __future__ import annotations

import math
from typing import Sequence

import numpy as np


THETA_LEVELS = (0.01, 0.09)
XI_LEVELS = (0.35, 1.10)
RHO_LEVELS = (-0.99, 0.99)
KAPPA = 3.0
MU = 0.0
DT = 1.0 / 250.0
S0 = 100.0
SEQUENCE_LENGTH = 128


def regimes() -> tuple[dict[str, float | int], ...]:
    values = []
    for theta_index, theta in enumerate(THETA_LEVELS):
        for xi_index, xi in enumerate(XI_LEVELS):
            for rho_index, rho in enumerate(RHO_LEVELS):
                values.append(
                    {
                        "label": len(values),
                        "theta": float(theta),
                        "xi": float(xi),
                        "rho": float(rho),
                        "theta_index": theta_index,
                        "xi_index": xi_index,
                        "rho_index": rho_index,
                    }
                )
    return tuple(values)


def balanced_labels(num_paths: int, *, seed: int) -> np.ndarray:
    if int(num_paths) < 8:
        raise ValueError("num_paths must be at least 8")
    labels = np.arange(int(num_paths), dtype=np.int64) % 8
    np.random.default_rng(int(seed)).shuffle(labels)
    return labels


def simulate_heston_mixture(
    *,
    num_paths: int,
    seed: int,
    labels: np.ndarray | None = None,
    sequence_length: int = SEQUENCE_LENGTH,
    dt: float = DT,
    s0: float = S0,
) -> tuple[np.ndarray, np.ndarray]:
    if labels is None:
        labels = balanced_labels(num_paths, seed=seed + 10_000)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (int(num_paths),) or np.any((labels < 0) | (labels >= 8)):
        raise ValueError("labels must have shape (num_paths,) and values in [0, 7]")
    table = regimes()
    theta = np.asarray([table[int(label)]["theta"] for label in labels], dtype=np.float64)
    xi = np.asarray([table[int(label)]["xi"] for label in labels], dtype=np.float64)
    rho = np.asarray([table[int(label)]["rho"] for label in labels], dtype=np.float64)
    variance = theta.copy()
    log_paths = np.zeros((int(num_paths), int(sequence_length)), dtype=np.float64)
    rng = np.random.default_rng(int(seed))

    for step in range(int(sequence_length) - 1):
        price_noise = rng.standard_normal(int(num_paths))
        independent_noise = rng.standard_normal(int(num_paths))
        positive_variance = np.maximum(variance, 1e-8)
        log_paths[:, step + 1] = (
            log_paths[:, step]
            + (MU - 0.5 * positive_variance) * float(dt)
            + np.sqrt(positive_variance * float(dt)) * price_noise
        )
        variance_noise = (
            rho * price_noise
            + np.sqrt(np.maximum(1.0 - rho * rho, 0.0)) * independent_noise
        )
        variance = np.maximum(
            variance
            + KAPPA * (theta - positive_variance) * float(dt)
            + xi * np.sqrt(positive_variance * float(dt)) * variance_noise,
            1e-8,
        )
    return float(s0) * np.exp(log_paths), labels


def simulate_fixed_heston(
    *,
    num_paths: int,
    seed: int,
    theta: float,
    xi: float,
    rho: float,
    sequence_length: int = SEQUENCE_LENGTH,
    dt: float = DT,
    s0: float = S0,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    variance = np.full(int(num_paths), float(theta), dtype=np.float64)
    log_paths = np.zeros((int(num_paths), int(sequence_length)), dtype=np.float64)
    for step in range(int(sequence_length) - 1):
        price_noise = rng.standard_normal(int(num_paths))
        independent_noise = rng.standard_normal(int(num_paths))
        positive_variance = np.maximum(variance, 1e-8)
        log_paths[:, step + 1] = (
            log_paths[:, step]
            + (MU - 0.5 * positive_variance) * float(dt)
            + np.sqrt(positive_variance * float(dt)) * price_noise
        )
        variance_noise = (
            float(rho) * price_noise
            + math.sqrt(max(1.0 - float(rho) ** 2, 0.0)) * independent_noise
        )
        variance = np.maximum(
            variance
            + KAPPA * (float(theta) - positive_variance) * float(dt)
            + float(xi) * np.sqrt(positive_variance * float(dt)) * variance_noise,
            1e-8,
        )
    return float(s0) * np.exp(log_paths)


def _row_moments(values: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = values.mean(axis=1)
    centered = values - mean[:, None]
    std = np.sqrt(np.mean(centered * centered, axis=1) + 1e-12)
    skew = np.mean(centered**3, axis=1) / std**3
    kurtosis = np.mean(centered**4, axis=1) / std**4 - 3.0
    return mean, std, skew, kurtosis


def extract_price_path_features(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float64)
    if (
        prices.ndim != 2
        or int(prices.shape[1]) != SEQUENCE_LENGTH
        or not np.isfinite(prices).all()
        or np.any(prices <= 0.0)
    ):
        raise ValueError(f"prices must have shape (B, {SEQUENCE_LENGTH}) and be positive")
    log_paths = np.log(prices / prices[:, :1])
    returns = np.diff(log_paths, axis=1)
    columns: list[np.ndarray] = []
    for values in (returns, np.abs(returns), returns**2):
        columns.extend(_row_moments(values))
    for quantile in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        columns.append(np.quantile(returns, quantile, axis=1))
    for lag in (1, 2, 3, 5, 10, 15, 20, 30, 40, 60):
        for values in (returns, np.abs(returns), returns**2):
            centered = values - values.mean(axis=1, keepdims=True)
            columns.append(
                np.mean(centered[:, :-lag] * centered[:, lag:], axis=1)
                / (np.mean(centered * centered, axis=1) + 1e-12)
            )
    for lag in (0, 1, 2, 3, 5, 10, 20, 40):
        left = returns[:, : returns.shape[1] - lag] if lag else returns
        right = returns[:, lag:] ** 2 if lag else returns**2
        left = left - left.mean(axis=1, keepdims=True)
        right = right - right.mean(axis=1, keepdims=True)
        columns.append(
            np.mean(left * right, axis=1)
            / np.sqrt(
                (np.mean(left * left, axis=1) + 1e-12)
                * (np.mean(right * right, axis=1) + 1e-12)
            )
        )
    for num_blocks in (2, 4, 8, 16):
        block_length = returns.shape[1] // num_blocks
        blocked = returns[:, : num_blocks * block_length].reshape(
            len(returns), num_blocks, block_length
        )
        realized_volatility = np.sqrt(np.mean(blocked**2, axis=2) / DT)
        columns.extend(
            (
                realized_volatility.mean(axis=1),
                realized_volatility.std(axis=1),
                realized_volatility.min(axis=1),
                realized_volatility.max(axis=1),
            )
        )
    columns.extend(
        (
            log_paths[:, -1],
            np.max(np.maximum.accumulate(log_paths, axis=1) - log_paths, axis=1),
        )
    )
    features = np.stack(columns, axis=1)
    if not np.isfinite(features).all():
        raise RuntimeError("feature extraction produced nonfinite values")
    return features


def label_bits(labels: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    return {
        "theta": (labels >> 2) & 1,
        "xi": (labels >> 1) & 1,
        "rho": labels & 1,
    }


def parameter_values_by_label(name: str) -> np.ndarray:
    if name not in {"theta", "xi", "rho"}:
        raise ValueError("name must be theta, xi, or rho")
    return np.asarray([float(regime[name]) for regime in regimes()], dtype=np.float64)
