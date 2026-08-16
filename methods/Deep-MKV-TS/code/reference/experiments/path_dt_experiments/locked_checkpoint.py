from __future__ import annotations

import math
from typing import Mapping, Sequence


_REQUIRED_SHADOW_METRICS = (
    "realized_volatility_coverage_90",
    "realized_volatility_crps",
    "cumulative_return_crps",
)


def coverage_focused_checkpoint_rule(
    *,
    candidate: Mapping[str, float | str],
    baseline: Mapping[str, float | str],
    nominal_coverage: float = 0.90,
) -> dict[str, float | bool]:
    """Evaluate the frozen coverage/CRPS rule without stylized-fact selection."""

    nominal = float(nominal_coverage)
    if not math.isfinite(nominal) or not 0.0 < nominal < 1.0:
        raise ValueError("nominal_coverage must lie in (0, 1)")
    values: dict[str, tuple[float, float]] = {}
    for name in _REQUIRED_SHADOW_METRICS:
        if name not in candidate or name not in baseline:
            raise ValueError(f"missing required shadow metric {name!r}")
        candidate_value = float(candidate[name])
        baseline_value = float(baseline[name])
        if not math.isfinite(candidate_value) or not math.isfinite(baseline_value):
            raise ValueError(f"shadow metric {name!r} must be finite")
        values[name] = candidate_value, baseline_value

    candidate_coverage, baseline_coverage = values[
        "realized_volatility_coverage_90"
    ]
    candidate_rv_crps, baseline_rv_crps = values["realized_volatility_crps"]
    candidate_return_crps, baseline_return_crps = values[
        "cumulative_return_crps"
    ]
    candidate_coverage_error = abs(candidate_coverage - nominal)
    baseline_coverage_error = abs(baseline_coverage - nominal)
    coverage_closer = candidate_coverage_error < baseline_coverage_error
    rv_crps_nonworsening = candidate_rv_crps <= baseline_rv_crps
    return_crps_nonworsening = candidate_return_crps <= baseline_return_crps
    return {
        "nominal_coverage": nominal,
        "candidate_coverage_error": candidate_coverage_error,
        "baseline_coverage_error": baseline_coverage_error,
        "coverage_error_change": candidate_coverage_error - baseline_coverage_error,
        "rv_crps_change": candidate_rv_crps - baseline_rv_crps,
        "return_crps_change": candidate_return_crps - baseline_return_crps,
        "coverage_closer": coverage_closer,
        "rv_crps_nonworsening": rv_crps_nonworsening,
        "return_crps_nonworsening": return_crps_nonworsening,
        "pass": (
            coverage_closer
            and rv_crps_nonworsening
            and return_crps_nonworsening
        ),
    }


def summarize_metric_rows(
    rows: Sequence[Mapping[str, float | str]],
) -> dict[str, dict[str, float]]:
    """Summarize common finite numeric metrics across independent seed rows."""

    if not rows:
        raise ValueError("rows must be non-empty")
    common = set(rows[0])
    for row in rows[1:]:
        common.intersection_update(row)
    summary: dict[str, dict[str, float]] = {}
    for name in sorted(common):
        try:
            values = [float(row[name]) for row in rows]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in values):
            continue
        count = len(values)
        mean = sum(values) / float(count)
        variance = sum((value - mean) ** 2 for value in values) / float(count)
        summary[name] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(values),
            "max": max(values),
            "num_seeds": float(count),
        }
    return summary


__all__ = [
    "coverage_focused_checkpoint_rule",
    "summarize_metric_rows",
]
