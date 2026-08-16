from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Sequence

import numpy as np
import torch

from path_dt_experiments.metrics import acf, excess_kurtosis


@dataclass(frozen=True)
class DateWindow:
    name: str
    start: date
    end: date
    role: str

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("window name must not be empty")
        if self.start > self.end:
            raise ValueError("window start must not follow its end")
        if self.role not in {
            "validation",
            "normal",
            "lead_in",
            "stress",
            "recovery",
            "test",
            "case_study",
        }:
            raise ValueError(f"unsupported window role: {self.role}")


@dataclass(frozen=True)
class RollingOriginProtocol:
    identifier: str
    label: str
    train_end: date
    validation: DateWindow
    evaluation_windows: tuple[DateWindow, ...]
    event_anchor: date | None = None
    scenario_bank_alias: str | None = None

    def __post_init__(self) -> None:
        if not str(self.identifier).strip() or not str(self.label).strip():
            raise ValueError("protocol identifier and label must not be empty")
        if self.validation.role != "validation":
            raise ValueError("the validation window must have role='validation'")
        if self.train_end >= self.validation.start:
            raise ValueError("training must end before validation begins")
        windows = (self.validation, *self.evaluation_windows)
        for previous, current in zip(windows, windows[1:]):
            if previous.end >= current.start:
                raise ValueError("protocol windows must be ordered and disjoint")
        if self.event_anchor is not None:
            stress = [
                window
                for window in self.evaluation_windows
                if window.role == "stress"
            ]
            if len(stress) != 1:
                raise ValueError("event protocols must have exactly one stress window")
            days_before_first_trading_session = (
                stress[0].start - self.event_anchor
            ).days
            if (
                self.event_anchor > stress[0].end
                or days_before_first_trading_session > 3
            ):
                raise ValueError(
                    "event anchor must lie in the stress window or no more "
                    "than three calendar days before its first trading session"
                )
        if self.scenario_bank_alias == self.identifier:
            raise ValueError("a protocol cannot alias its own scenario bank")

    @property
    def scenario_bank_identifier(self) -> str:
        return self.scenario_bank_alias or self.identifier


FROZEN_REAL_DATA_PROTOCOLS: tuple[RollingOriginProtocol, ...] = (
    RollingOriginProtocol(
        identifier="primary_2026_holdout",
        label="Primary chronological 2026 holdout",
        train_end=date(2025, 5, 31),
        validation=DateWindow(
            name="validation",
            start=date(2025, 6, 1),
            end=date(2025, 12, 31),
            role="validation",
        ),
        evaluation_windows=(
            DateWindow(
                name="test",
                start=date(2026, 1, 1),
                end=date(2026, 6, 30),
                role="test",
            ),
        ),
    ),
    RollingOriginProtocol(
        identifier="august_2024_carry_unwind",
        label="August 2024 global turbulence and carry-trade unwind",
        train_end=date(2024, 4, 30),
        validation=DateWindow(
            name="validation",
            start=date(2024, 5, 1),
            end=date(2024, 5, 31),
            role="validation",
        ),
        evaluation_windows=(
            DateWindow(
                name="matched_normal",
                start=date(2024, 6, 1),
                end=date(2024, 6, 30),
                role="normal",
            ),
            DateWindow(
                name="fragility_lead_in",
                start=date(2024, 7, 1),
                end=date(2024, 7, 30),
                role="lead_in",
            ),
            DateWindow(
                name="stress",
                start=date(2024, 7, 31),
                end=date(2024, 8, 9),
                role="stress",
            ),
            DateWindow(
                name="recovery",
                start=date(2024, 8, 12),
                end=date(2024, 8, 16),
                role="recovery",
            ),
        ),
        event_anchor=date(2024, 8, 5),
    ),
    RollingOriginProtocol(
        identifier="april_2025_tariff_shock",
        label="April 2025 tariff-announcement shock",
        train_end=date(2024, 11, 29),
        validation=DateWindow(
            name="validation",
            start=date(2024, 12, 1),
            end=date(2024, 12, 31),
            role="validation",
        ),
        evaluation_windows=(
            DateWindow(
                name="matched_normal",
                start=date(2025, 1, 1),
                end=date(2025, 1, 31),
                role="normal",
            ),
            DateWindow(
                name="policy_uncertainty_lead_in",
                start=date(2025, 2, 1),
                end=date(2025, 4, 1),
                role="lead_in",
            ),
            DateWindow(
                name="stress",
                start=date(2025, 4, 2),
                end=date(2025, 4, 11),
                role="stress",
            ),
            DateWindow(
                name="recovery",
                start=date(2025, 4, 14),
                end=date(2025, 4, 30),
                role="recovery",
            ),
        ),
        event_anchor=date(2025, 4, 2),
    ),
    RollingOriginProtocol(
        identifier="us_iran_2026_war",
        label="2026 U.S.–Iran war",
        train_end=date(2025, 5, 31),
        validation=DateWindow(
            name="validation",
            start=date(2025, 6, 1),
            end=date(2025, 12, 31),
            role="validation",
        ),
        evaluation_windows=(
            DateWindow(
                name="matched_prewar",
                start=date(2026, 1, 2),
                end=date(2026, 2, 27),
                role="normal",
            ),
            DateWindow(
                name="war",
                start=date(2026, 3, 2),
                end=date(2026, 4, 7),
                role="stress",
            ),
            DateWindow(
                name="initial_recovery",
                start=date(2026, 4, 8),
                end=date(2026, 4, 30),
                role="recovery",
            ),
            DateWindow(
                name="late_june_reescalation",
                start=date(2026, 6, 26),
                end=date(2026, 6, 30),
                role="case_study",
            ),
        ),
        event_anchor=date(2026, 2, 28),
        scenario_bank_alias="primary_2026_holdout",
    ),
)


EXTERNAL_EVENT_EVIDENCE: tuple[dict[str, str], ...] = (
    {
        "protocol": "august_2024_carry_unwind",
        "source": "Bank of Japan",
        "date": "2024-07-31",
        "evidence": (
            "The policy meeting raised the uncollateralized overnight call-rate "
            "guideline to around 0.25 percent, effective 1 August."
        ),
        "url": "https://www.boj.or.jp/en/mopo/mpmsche_minu/minu_2024/g240731.htm",
    },
    {
        "protocol": "august_2024_carry_unwind",
        "source": "Bank for International Settlements",
        "date": "2024-08-05",
        "evidence": (
            "The BIS identifies early-August global turbulence, a carry-trade "
            "unwind, and 5 August as the peak-stress date."
        ),
        "url": "https://www.bis.org/publ/bisbull90.htm",
    },
    {
        "protocol": "april_2025_tariff_shock",
        "source": "The White House",
        "date": "2025-04-02",
        "evidence": (
            "Executive Order 14257 announced broad reciprocal tariffs on "
            "2 April 2025."
        ),
        "url": (
            "https://www.whitehouse.gov/presidential-actions/2025/04/"
            "regulating-imports-with-a-reciprocal-tariff-to-rectify-trade-"
            "practices-that-contribute-to-large-and-persistent-annual-united-"
            "states-goods-trade-deficits/"
        ),
    },
    {
        "protocol": "april_2025_tariff_shock",
        "source": "International Monetary Fund",
        "date": "2025-04-09",
        "evidence": (
            "The IMF describes a broad sell-off and elevated cross-asset "
            "volatility after 2 April, followed by partial recovery after the "
            "9 April pause."
        ),
        "url": (
            "https://www.elibrary.imf.org/display/book/"
            "9798229023184/CH001.xml"
        ),
    },
    {
        "protocol": "us_iran_2026_war",
        "source": "The White House",
        "date": "2026-02-28",
        "evidence": (
            "The official operation chronology identifies 28 February as the "
            "start of U.S. military operations and 7 April as the date of the "
            "initial ceasefire."
        ),
        "url": (
            "https://www.whitehouse.gov/releases/2026/04/"
            "peace-through-strength-operation-epic-fury-crushes-iranian-"
            "threat-as-ceasefire-takes-hold/"
        ),
    },
    {
        "protocol": "us_iran_2026_war",
        "source": "International Monetary Fund",
        "date": "2026-04-01",
        "evidence": (
            "The IMF documents the associated equity correction, energy-price "
            "shock, and rise in market volatility."
        ),
        "url": (
            "https://www.imf.org/-/media/files/publications/gfsr/2026/"
            "april/english/ch1.pdf"
        ),
    },
    {
        "protocol": "us_iran_2026_war",
        "source": "United Nations DPPA",
        "date": "2026-06-26",
        "evidence": (
            "The UN records renewed U.S. strikes on 26 and 27 June; only three "
            "complete sessions remain in the dataset, so this is a case study."
        ),
        "url": (
            "https://dppa.un.org/en/news/un-calls-for-maximum-restraint-to-"
            "preserve-ceasefire-between-the-united-states-and-iran"
        ),
    },
)


def protocol_by_identifier(identifier: str) -> RollingOriginProtocol:
    matches = [
        protocol
        for protocol in FROZEN_REAL_DATA_PROTOCOLS
        if protocol.identifier == str(identifier)
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown real-data protocol: {identifier}")
    return matches[0]


def select_date_window(
    dates: Sequence[str],
    window: DateWindow,
) -> np.ndarray:
    parsed = np.asarray([date.fromisoformat(str(day)) for day in dates])
    return np.asarray(
        [
            index
            for index, day in enumerate(parsed)
            if window.start <= day <= window.end
        ],
        dtype=np.int64,
    )


def training_indices(
    dates: Sequence[str],
    protocol: RollingOriginProtocol,
) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, day in enumerate(dates)
            if date.fromisoformat(str(day)) <= protocol.train_end
        ],
        dtype=np.int64,
    )


def validate_protocol_on_calendar(
    dates: Sequence[str],
    protocol: RollingOriginProtocol,
) -> dict[str, int]:
    train = training_indices(dates, protocol)
    if train.size < 2:
        raise ValueError(f"{protocol.identifier} has fewer than two train sessions")
    counts = {"train": int(train.size)}
    for window in (protocol.validation, *protocol.evaluation_windows):
        indices = select_date_window(dates, window)
        if indices.size < 1:
            raise ValueError(
                f"{protocol.identifier}/{window.name} is empty on the calendar"
            )
        counts[window.name] = int(indices.size)
    return counts


def normalized_prices_to_log_paths(prices: np.ndarray) -> torch.Tensor:
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 3:
        raise ValueError("prices must have shape (sessions, at least 3 points)")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("prices must be finite and positive")
    logs = np.log(values / values[:, :1])
    return torch.from_numpy(logs).unsqueeze(-1).to(dtype=torch.float64)


def calendar_aligned_moving_block_bootstrap(
    training_prices: np.ndarray,
    *,
    bank_size: int,
    block_length: int = 8,
    seed: int = 1234,
) -> np.ndarray:
    """
    Bootstrap intraday log-return blocks at their original clock positions.

    Each output block chooses one source day, while its time indices remain
    fixed.  This preserves intraday seasonality and within-block dependence
    without using any post-training observation.
    """
    prices = np.asarray(training_prices, dtype=np.float64)
    if prices.ndim != 2 or prices.shape[0] < 2 or prices.shape[1] < 3:
        raise ValueError("training_prices must contain at least two paths")
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ValueError("training_prices must be finite and positive")
    if isinstance(bank_size, bool) or int(bank_size) < 1:
        raise ValueError("bank_size must be positive")
    if isinstance(block_length, bool) or int(block_length) < 1:
        raise ValueError("block_length must be positive")
    bank_size = int(bank_size)
    block_length = int(block_length)
    source_returns = np.diff(np.log(prices), axis=1)
    num_steps = int(source_returns.shape[1])
    if block_length > num_steps:
        raise ValueError("block_length cannot exceed the number of returns")
    generator = np.random.default_rng(int(seed))
    generated_returns = np.empty((bank_size, num_steps), dtype=np.float64)
    for start in range(0, num_steps, block_length):
        stop = min(start + block_length, num_steps)
        source_days = generator.integers(
            0,
            int(source_returns.shape[0]),
            size=bank_size,
        )
        generated_returns[:, start:stop] = source_returns[
            source_days,
            start:stop,
        ]
    generated_logs = np.concatenate(
        (
            np.zeros((bank_size, 1), dtype=np.float64),
            np.cumsum(generated_returns, axis=1),
        ),
        axis=1,
    )
    return (100.0 * np.exp(generated_logs)).astype(np.float32)


def whole_session_bootstrap(
    training_prices: np.ndarray,
    *,
    bank_size: int,
    seed: int = 1234,
) -> np.ndarray:
    prices = np.asarray(training_prices)
    if prices.ndim != 2 or prices.shape[0] < 2:
        raise ValueError("training_prices must contain at least two paths")
    if isinstance(bank_size, bool) or int(bank_size) < 1:
        raise ValueError("bank_size must be positive")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, int(prices.shape[0]), size=int(bank_size))
    return prices[indices].astype(np.float32, copy=True)


def _quantile_wasserstein(
    generated: torch.Tensor,
    target: torch.Tensor,
    *,
    quantile_count: int = 501,
) -> float:
    quantiles = torch.linspace(
        0.0,
        1.0,
        int(quantile_count),
        dtype=generated.dtype,
        device=generated.device,
    )
    left = torch.quantile(generated.reshape(-1), quantiles)
    right = torch.quantile(target.reshape(-1), quantiles)
    return float(torch.mean(torch.abs(left - right)).item())


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    value = float(denominator.item())
    if abs(value) < 1e-12:
        return math.nan
    return float(numerator.item() / value)


def _pooled_correlation(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    left = left.reshape(-1)
    right = right.reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(
        torch.mean(left.pow(2)) * torch.mean(right.pow(2))
    )
    if float(denominator.item()) < 1e-15:
        return left.new_tensor(0.0)
    return torch.mean(left * right) / denominator


def _maximum_drawdown(paths: torch.Tensor) -> torch.Tensor:
    running_max = torch.cummax(paths, dim=1).values
    return (running_max - paths).amax(dim=1)


def real_data_law_metrics(
    generated_log_paths: torch.Tensor,
    target_log_paths: torch.Tensor,
    *,
    lags: Sequence[int] = (1, 5, 10, 20),
    leverage_lags: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    if generated_log_paths.ndim != 3 or target_log_paths.ndim != 3:
        raise ValueError("paths must have shape (sessions, points, assets)")
    if tuple(generated_log_paths.shape[1:]) != tuple(
        target_log_paths.shape[1:]
    ):
        raise ValueError("generated and target path shapes must agree")
    if generated_log_paths.shape[2] != 1:
        raise ValueError("the frozen real-data law metrics are univariate")
    generated = generated_log_paths.to(dtype=torch.float64)
    target = target_log_paths.to(
        device=generated.device,
        dtype=generated.dtype,
    )
    generated_returns = torch.diff(generated, dim=1)
    target_returns = torch.diff(target, dim=1)
    generated_rv = torch.sqrt(torch.sum(generated_returns.pow(2), dim=1))
    target_rv = torch.sqrt(torch.sum(target_returns.pow(2), dim=1))
    generated_drawdown = _maximum_drawdown(generated)
    target_drawdown = _maximum_drawdown(target)
    generated_terminal = generated[:, -1, :]
    target_terminal = target[:, -1, :]
    target_return_std = target_returns.std(unbiased=False).clamp_min(1e-12)
    target_rv_mean = target_rv.mean().clamp_min(1e-12)
    target_drawdown_mean = target_drawdown.mean().clamp_min(1e-12)
    target_terminal_std = target_terminal.std(unbiased=False).clamp_min(1e-12)

    quantiles = torch.linspace(
        0.01,
        0.99,
        99,
        device=generated.device,
        dtype=generated.dtype,
    )
    generated_return_q = torch.quantile(generated_returns.reshape(-1), quantiles)
    target_return_q = torch.quantile(target_returns.reshape(-1), quantiles)
    abs_lag_values = tuple(int(lag) for lag in lags)
    generated_abs_acf = acf(generated_returns.abs(), lags=abs_lag_values)
    target_abs_acf = acf(target_returns.abs(), lags=abs_lag_values)
    generated_sq_acf = acf(generated_returns.pow(2), lags=abs_lag_values)
    target_sq_acf = acf(target_returns.pow(2), lags=abs_lag_values)

    leverage_error = []
    metrics: dict[str, float] = {}
    for lag in leverage_lags:
        lag = int(lag)
        generated_leverage = _pooled_correlation(
            generated_returns[:, :-lag, :],
            generated_returns[:, lag:, :].pow(2),
        )
        target_leverage = _pooled_correlation(
            target_returns[:, :-lag, :],
            target_returns[:, lag:, :].pow(2),
        )
        metrics[f"generated_leverage_lag_{lag}"] = float(
            generated_leverage.item()
        )
        metrics[f"target_leverage_lag_{lag}"] = float(target_leverage.item())
        leverage_error.append(generated_leverage - target_leverage)

    generated_kurtosis = excess_kurtosis(generated_returns)
    target_kurtosis = excess_kurtosis(target_returns)
    metrics.update(
        {
            "return_mean_error": float(
                (generated_returns.mean() - target_returns.mean()).item()
            ),
            "return_std_ratio": _ratio(
                generated_returns.std(unbiased=False),
                target_return_std,
            ),
            "return_qq_rmse_normalized": float(
                torch.sqrt(
                    torch.mean((generated_return_q - target_return_q).pow(2))
                ).item()
                / float(target_return_std.item())
            ),
            "return_wasserstein_normalized": (
                _quantile_wasserstein(generated_returns, target_returns)
                / float(target_return_std.item())
            ),
            "generated_excess_kurtosis": float(generated_kurtosis.item()),
            "target_excess_kurtosis": float(target_kurtosis.item()),
            "excess_kurtosis_error": float(
                (generated_kurtosis - target_kurtosis).item()
            ),
            "abs_return_acf_error_rms": float(
                torch.sqrt(
                    torch.mean((generated_abs_acf - target_abs_acf).pow(2))
                ).item()
            ),
            "squared_return_acf_error_rms": float(
                torch.sqrt(
                    torch.mean((generated_sq_acf - target_sq_acf).pow(2))
                ).item()
            ),
            "leverage_error_rms": float(
                torch.sqrt(torch.mean(torch.stack(leverage_error).pow(2))).item()
            ),
            "realized_volatility_mean_ratio": _ratio(
                generated_rv.mean(),
                target_rv.mean(),
            ),
            "realized_volatility_std_ratio": _ratio(
                generated_rv.std(unbiased=False),
                target_rv.std(unbiased=False),
            ),
            "realized_volatility_wasserstein_normalized": (
                _quantile_wasserstein(generated_rv, target_rv)
                / float(target_rv_mean.item())
            ),
            "realized_volatility_q90_ratio": _ratio(
                torch.quantile(generated_rv, 0.90),
                torch.quantile(target_rv, 0.90),
            ),
            "maximum_drawdown_mean_ratio": _ratio(
                generated_drawdown.mean(),
                target_drawdown.mean(),
            ),
            "maximum_drawdown_q90_ratio": _ratio(
                torch.quantile(generated_drawdown, 0.90),
                torch.quantile(target_drawdown, 0.90),
            ),
            "maximum_drawdown_wasserstein_normalized": (
                _quantile_wasserstein(generated_drawdown, target_drawdown)
                / float(target_drawdown_mean.item())
            ),
            "terminal_return_wasserstein_normalized": (
                _quantile_wasserstein(generated_terminal, target_terminal)
                / float(target_terminal_std.item())
            ),
        }
    )
    return metrics


def frozen_protocol_manifest() -> dict[str, object]:
    return {
        "version": "1.1",
        "amendment": {
            "date": "2026-07-31",
            "reason": (
                "Added the externally defined 2026 U.S.–Iran war stratification "
                "before any learned generator was evaluated on real data."
            ),
            "effect_on_training": (
                "None. The war protocol aliases the primary 2026 scenario bank "
                "and only stratifies held-out test sessions."
            ),
        },
        "coordinate_contract": {
            "shared_input": "normalized price, open=100",
            "evaluation_coordinate": "log(price / session_open)",
        },
        "scenario_bank_contract": {
            "paths_per_seed": 8192,
            "replication_seeds": [0, 1, 2, 3, 4],
            "shape_per_file": [8192, 128],
            "required_scale": "normalized price with every initial value equal to 100",
            "required_values": "finite and strictly positive",
            "trainable_model_rule": (
                "Each seed is an independent model fit followed by one scenario-bank draw."
            ),
            "nonparametric_model_rule": (
                "For a method with no fitted parameters, such as SBTS, each seed "
                "is an independent stochastic generation run from the same frozen "
                "training set. It must not be described as an independent fit."
            ),
        },
        "path_shadowing": {
            "prefix_points": 65,
            "future_increments": 32,
            "top_k": 256,
            "feature_weights": {
                "recent_returns": 1.0,
                "cumulative_path": 0.5,
                "rolling_volatility": 2.0,
                "absolute_and_squared_return_acf": 1.0,
            },
            "bootstrap_replicates": 2000,
            "confidence_level": 0.95,
        },
        "law_metrics": {
            "return": [
                "mean error",
                "standard-deviation ratio",
                "normalized QQ RMSE",
                "normalized Wasserstein distance",
                "excess-kurtosis error",
            ],
            "dependence": [
                "absolute-return ACF RMS error at lags 1, 5, 10, 20",
                "squared-return ACF RMS error at lags 1, 5, 10, 20",
                "leverage RMS error at lags 1, 5, 10",
            ],
            "path_functionals": [
                "realized-volatility mean/std/q90 ratios and Wasserstein error",
                "maximum-drawdown mean/q90 ratios and Wasserstein error",
                "terminal-return Wasserstein error",
            ],
        },
        "protocols": [
            {
                "identifier": protocol.identifier,
                "label": protocol.label,
                "train_end": protocol.train_end.isoformat(),
                "event_anchor": (
                    None
                    if protocol.event_anchor is None
                    else protocol.event_anchor.isoformat()
                ),
                "scenario_bank_identifier": (
                    protocol.scenario_bank_identifier
                ),
                "validation": {
                    "name": protocol.validation.name,
                    "role": protocol.validation.role,
                    "start": protocol.validation.start.isoformat(),
                    "end": protocol.validation.end.isoformat(),
                },
                "evaluation_windows": [
                    {
                        "name": window.name,
                        "role": window.role,
                        "start": window.start.isoformat(),
                        "end": window.end.isoformat(),
                    }
                    for window in protocol.evaluation_windows
                ],
            }
            for protocol in FROZEN_REAL_DATA_PROTOCOLS
        ],
        "external_event_evidence": list(EXTERNAL_EVENT_EVIDENCE),
        "interpretation": {
            "primary_2026_holdout": (
                "Headline held-out law comparison and path-shadowing evaluation."
            ),
            "event_protocols": (
                "Secondary rolling-origin tests. Report matched-normal, stress, "
                "and recovery phases separately; do not pool them."
            ),
            "small_episode_samples": (
                "Episode confidence intervals are descriptive because individual "
                "stress windows contain few trading sessions. Confirm conclusions "
                "across indices and seeds."
            ),
        },
    }


__all__ = [
    "DateWindow",
    "EXTERNAL_EVENT_EVIDENCE",
    "FROZEN_REAL_DATA_PROTOCOLS",
    "RollingOriginProtocol",
    "calendar_aligned_moving_block_bootstrap",
    "frozen_protocol_manifest",
    "normalized_prices_to_log_paths",
    "protocol_by_identifier",
    "real_data_law_metrics",
    "select_date_window",
    "training_indices",
    "validate_protocol_on_calendar",
    "whole_session_bootstrap",
]
