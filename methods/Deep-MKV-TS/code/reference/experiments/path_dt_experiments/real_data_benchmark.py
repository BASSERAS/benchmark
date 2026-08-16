from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq


INDEX_NAMES = ("ES", "NQ", "RTY", "YM")
SESSION_FILE_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<index>[A-Z]+)\..*\.parquet$"
)


@dataclass(frozen=True)
class RealDataPreprocessingConfig:
    timezone_name: str = "America/New_York"
    session_start: time = time(9, 30)
    session_end: time = time(16, 0)
    path_points: int = 128
    maximum_staleness_seconds: float = 60.0
    maximum_abs_log_return: float = 0.20
    train_end: date = date(2025, 5, 31)
    validation_end: date = date(2025, 12, 31)

    def __post_init__(self) -> None:
        if int(self.path_points) < 3:
            raise ValueError("path_points must be at least three")
        if not self.session_start < self.session_end:
            raise ValueError("session_start must precede session_end")
        for name in (
            "maximum_staleness_seconds",
            "maximum_abs_log_return",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if not self.train_end < self.validation_end:
            raise ValueError("train_end must precede validation_end")
        ZoneInfo(self.timezone_name)


@dataclass(frozen=True)
class SessionAuditResult:
    record: dict[str, object]
    normalized_prices: np.ndarray | None


def file_identity(path: Path) -> tuple[date, str]:
    match = SESSION_FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"unexpected session filename: {path.name}")
    return (
        date.fromisoformat(match.group("date")),
        match.group("index"),
    )


def session_grid_ns(
    session_date: date,
    config: RealDataPreprocessingConfig,
) -> np.ndarray:
    local_zone = ZoneInfo(config.timezone_name)
    start = datetime.combine(
        session_date,
        config.session_start,
        tzinfo=local_zone,
    ).astimezone(timezone.utc)
    end = datetime.combine(
        session_date,
        config.session_end,
        tzinfo=local_zone,
    ).astimezone(timezone.utc)
    start_ns = int(round(start.timestamp() * 1_000_000_000))
    end_ns = int(round(end.timestamp() * 1_000_000_000))
    offsets = np.rint(
        np.linspace(
            0.0,
            float(end_ns - start_ns),
            int(config.path_points),
        )
    ).astype(np.int64)
    return start_ns + offsets


def split_name(
    session_date: date,
    config: RealDataPreprocessingConfig,
) -> str:
    if session_date <= config.train_end:
        return "train"
    if session_date <= config.validation_end:
        return "validation"
    return "test"


def _timestamp_ns(column) -> np.ndarray:
    return (
        column.combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype("datetime64[ns]")
        .astype(np.int64)
    )


def _float_values(column) -> np.ndarray:
    return np.asarray(
        column.combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )


def audit_session_file(
    path: Path,
    *,
    expected_index: str,
    config: RealDataPreprocessingConfig,
) -> SessionAuditResult:
    path = Path(path)
    session_date, filename_index = file_identity(path)
    reasons: list[str] = []
    if filename_index != expected_index:
        reasons.append("filename_index_mismatch")

    base_record: dict[str, object] = {
        "index": expected_index,
        "date": session_date.isoformat(),
        "source_file": str(path.resolve()),
        "filename_fomc": "-fomc." in path.name,
        "split": split_name(session_date, config),
    }
    try:
        parquet = pq.ParquetFile(path)
        schema_names = parquet.schema_arrow.names
        required = {"ts_event", "mid_px"}
        if not required.issubset(schema_names):
            reasons.append("missing_required_columns")
            return SessionAuditResult(
                record={
                    **base_record,
                    "row_count": int(parquet.metadata.num_rows),
                    "schema_columns": list(schema_names),
                    "included_individually": False,
                    "exclusion_reasons": reasons,
                },
                normalized_prices=None,
            )
        table = pq.read_table(path, columns=["ts_event", "mid_px"])
        timestamps = _timestamp_ns(table["ts_event"])
        mid_prices = _float_values(table["mid_px"])
    except Exception as exc:  # pragma: no cover - exercised on bad input files
        return SessionAuditResult(
            record={
                **base_record,
                "included_individually": False,
                "exclusion_reasons": ["read_error"],
                "read_error": f"{type(exc).__name__}: {exc}",
            },
            normalized_prices=None,
        )

    grid = session_grid_ns(session_date, config)
    start_ns = int(grid[0])
    end_ns = int(grid[-1])
    finite_positive = np.isfinite(mid_prices) & (mid_prices > 0.0)
    nonpositive_count = int(
        np.sum(np.isfinite(mid_prices) & (mid_prices <= 0.0))
    )
    nonfinite_nonnull_count = int(
        np.sum(~np.isfinite(mid_prices) & ~np.isnan(mid_prices))
    )
    session_mask = (timestamps >= start_ns) & (timestamps <= end_ns)
    session_count = int(np.sum(session_mask))
    session_valid_count = int(np.sum(session_mask & finite_positive))
    nonincreasing_timestamps = int(np.sum(np.diff(timestamps) <= 0))
    if nonincreasing_timestamps:
        reasons.append("nonincreasing_timestamps")
    if nonpositive_count:
        reasons.append("nonpositive_midprice")
    if nonfinite_nonnull_count:
        reasons.append("nonfinite_midprice")

    valid_timestamps = timestamps[finite_positive]
    valid_prices = mid_prices[finite_positive]
    positions = np.searchsorted(valid_timestamps, grid, side="right") - 1
    missing_grid_sources = int(np.sum(positions < 0))
    if missing_grid_sources:
        reasons.append("missing_causal_grid_source")
        sampled = None
        staleness = np.full(len(grid), np.inf, dtype=np.float64)
    else:
        source_timestamps = valid_timestamps[positions]
        sampled = valid_prices[positions]
        staleness = (grid - source_timestamps).astype(np.float64) / 1e9
        if bool(np.any(staleness < 0.0)):
            reasons.append("future_observation_used")
        if float(np.max(staleness)) > float(
            config.maximum_staleness_seconds
        ):
            reasons.append("staleness_limit")

    normalized_prices: np.ndarray | None = None
    daily_metrics: dict[str, float | None] = {
        "maximum_staleness_seconds": (
            None if not np.isfinite(staleness).all() else float(staleness.max())
        ),
        "q95_staleness_seconds": (
            None
            if not np.isfinite(staleness).all()
            else float(np.quantile(staleness, 0.95))
        ),
        "mean_staleness_seconds": (
            None
            if not np.isfinite(staleness).all()
            else float(staleness.mean())
        ),
        "daily_realized_variation": None,
        "daily_terminal_log_return": None,
        "daily_maximum_drawdown": None,
        "maximum_abs_grid_log_return": None,
    }
    if sampled is not None:
        if not bool(np.isfinite(sampled).all()) or bool(np.any(sampled <= 0.0)):
            reasons.append("invalid_sampled_path")
        else:
            log_path = np.log(sampled / sampled[0])
            returns = np.diff(log_path)
            max_abs_return = float(np.max(np.abs(returns)))
            running_max = np.maximum.accumulate(log_path)
            daily_metrics = {
                **daily_metrics,
                "daily_realized_variation": float(
                    np.sqrt(np.sum(returns**2))
                ),
                "daily_terminal_log_return": float(log_path[-1]),
                "daily_maximum_drawdown": float(
                    np.max(running_max - log_path)
                ),
                "maximum_abs_grid_log_return": max_abs_return,
            }
            if max_abs_return > float(config.maximum_abs_log_return):
                reasons.append("grid_return_limit")
            normalized_prices = (
                100.0 * np.exp(log_path)
            ).astype(np.float32)

    included = not reasons
    if not included:
        normalized_prices = None
    record: dict[str, object] = {
        **base_record,
        "row_count": int(len(timestamps)),
        "schema_columns": list(parquet.schema_arrow.names),
        "source_start_utc": datetime.fromtimestamp(
            int(timestamps[0]) / 1e9,
            tz=timezone.utc,
        ).isoformat(),
        "source_end_utc": datetime.fromtimestamp(
            int(timestamps[-1]) / 1e9,
            tz=timezone.utc,
        ).isoformat(),
        "nonincreasing_timestamp_count": nonincreasing_timestamps,
        "midprice_null_count": int(np.sum(np.isnan(mid_prices))),
        "midprice_valid_count": int(np.sum(finite_positive)),
        "midprice_valid_fraction": float(np.mean(finite_positive)),
        "midprice_nonpositive_count": nonpositive_count,
        "midprice_nonfinite_nonnull_count": nonfinite_nonnull_count,
        "rth_row_count": session_count,
        "rth_valid_midprice_count": session_valid_count,
        "rth_valid_midprice_fraction": (
            float(session_valid_count / session_count)
            if session_count
            else 0.0
        ),
        "missing_grid_source_count": missing_grid_sources,
        **daily_metrics,
        "included_individually": included,
        "exclusion_reasons": reasons,
    }
    return SessionAuditResult(
        record=record,
        normalized_prices=normalized_prices,
    )


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def common_valid_dates(
    results_by_index: dict[str, Iterable[SessionAuditResult]],
) -> list[str]:
    valid_sets = {
        index: {
            str(result.record["date"])
            for result in results
            if bool(result.record["included_individually"])
        }
        for index, results in results_by_index.items()
    }
    if set(valid_sets) != set(INDEX_NAMES):
        raise ValueError("results must contain ES, NQ, RTY, and YM")
    return sorted(set.intersection(*valid_sets.values()))


__all__ = [
    "INDEX_NAMES",
    "RealDataPreprocessingConfig",
    "SessionAuditResult",
    "audit_session_file",
    "common_valid_dates",
    "file_identity",
    "session_grid_ns",
    "sha256_file",
    "split_name",
]
