from __future__ import annotations

import math
from typing import Sequence

import torch

from path_dt_experiments.metrics import stylized_fact_metrics


STYLIZED_GUARD_METRICS = (
    "excess_kurtosis_error_rms",
    "tail_survival_error_rms",
    "abs_return_acf_error_rms",
    "squared_return_acf_error_rms",
)


def bootstrap_stylized_error_standard_errors(
    generated_paths: torch.Tensor,
    reference_paths: torch.Tensor,
    *,
    lags: Sequence[int],
    tail_quantiles: Sequence[float] = (0.90, 0.95, 0.99),
    num_replicates: int,
    seed: int,
    resample_paths: int | None = None,
) -> dict[str, float]:
    """Path-bootstrap uncertainty for the stylised-fact error guards.

    Generated and reference paths are resampled independently.  The target
    tail thresholds are re-estimated inside every replicate, matching the
    definition used by :func:`stylized_fact_metrics`.  SWD is deliberately
    excluded and controlled by a fixed multiplicative guard instead.
    """

    if generated_paths.ndim != 3 or reference_paths.ndim != 3:
        raise ValueError("generated_paths and reference_paths must have shape (B, N + 1, d)")
    if tuple(generated_paths.shape[1:]) != tuple(reference_paths.shape[1:]):
        raise ValueError("generated and reference path shapes must agree after the batch axis")
    if int(generated_paths.shape[0]) < 2 or int(reference_paths.shape[0]) < 2:
        raise ValueError("stylized bootstrap requires at least two paths per law")
    if int(num_replicates) < 2:
        raise ValueError("num_replicates must be >= 2")
    sample_count = (
        min(int(generated_paths.shape[0]), int(reference_paths.shape[0]))
        if resample_paths is None
        else int(resample_paths)
    )
    if sample_count < 2:
        raise ValueError("resample_paths must be >= 2")
    generated = generated_paths.detach()
    reference = reference_paths.detach().to(
        device=generated.device,
        dtype=generated.dtype,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    replicate_values = {name: [] for name in STYLIZED_GUARD_METRICS}
    for _ in range(int(num_replicates)):
        generated_indices = torch.randint(
            int(generated.shape[0]),
            (sample_count,),
            generator=generator,
            device="cpu",
        ).to(device=generated.device)
        reference_indices = torch.randint(
            int(reference.shape[0]),
            (sample_count,),
            generator=generator,
            device="cpu",
        ).to(device=reference.device)
        metrics = stylized_fact_metrics(
            generated.index_select(0, generated_indices),
            reference.index_select(0, reference_indices),
            lags=tuple(int(value) for value in lags),
            tail_quantiles=tuple(float(value) for value in tail_quantiles),
            include_swd=False,
        )
        for name in STYLIZED_GUARD_METRICS:
            replicate_values[name].append(float(metrics[name]))
    return {
        name: float(torch.tensor(values, dtype=torch.float64).std(unbiased=True).item())
        for name, values in replicate_values.items()
    }


def shadow_frontier_decision(
    candidate_shadowing: dict[str, float | str],
    baseline_shadowing: dict[str, float | str],
    candidate_stylized: dict[str, float],
    baseline_stylized: dict[str, float],
    *,
    shadowing_standard_errors: dict[str, float],
    stylized_standard_errors: dict[str, float],
    standard_error_budget: float = 0.5,
    swd_factor_limit: float = 1.10,
    nominal_coverage: float = 0.90,
) -> dict[str, object]:
    """Apply the predeclared validation gate for the shadowing frontier."""

    for name, value in (
        ("standard_error_budget", standard_error_budget),
        ("swd_factor_limit", swd_factor_limit),
        ("nominal_coverage", nominal_coverage),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(standard_error_budget) < 0.0:
        raise ValueError("standard_error_budget must be non-negative")
    if float(swd_factor_limit) <= 0.0:
        raise ValueError("swd_factor_limit must be positive")
    if not 0.0 < float(nominal_coverage) < 1.0:
        raise ValueError("nominal_coverage must lie in (0, 1)")

    def finite_metric(values: dict[str, object], name: str) -> float:
        if name not in values:
            raise ValueError(f"metrics are missing {name}")
        result = float(values[name])
        if not math.isfinite(result):
            raise ValueError(f"metric {name} must be finite")
        return result

    baseline_rv_crps = finite_metric(baseline_shadowing, "realized_volatility_crps")
    candidate_rv_crps = finite_metric(candidate_shadowing, "realized_volatility_crps")
    rv_crps_change = candidate_rv_crps - baseline_rv_crps
    rv_crps_check = rv_crps_change < 0.0

    baseline_coverage = finite_metric(
        baseline_shadowing,
        "realized_volatility_coverage_90",
    )
    candidate_coverage = finite_metric(
        candidate_shadowing,
        "realized_volatility_coverage_90",
    )
    baseline_coverage_error = abs(baseline_coverage - float(nominal_coverage))
    candidate_coverage_error = abs(candidate_coverage - float(nominal_coverage))
    coverage_check = candidate_coverage_error < baseline_coverage_error

    return_metric = "cumulative_return_crps"
    if return_metric not in shadowing_standard_errors:
        raise ValueError(f"shadowing_standard_errors is missing {return_metric}")
    return_standard_error = float(shadowing_standard_errors[return_metric])
    if not math.isfinite(return_standard_error) or return_standard_error < 0.0:
        raise ValueError("return CRPS standard error must be finite and non-negative")
    return_crps_change = finite_metric(candidate_shadowing, return_metric) - finite_metric(
        baseline_shadowing,
        return_metric,
    )
    return_crps_allowance = float(standard_error_budget) * return_standard_error
    return_crps_check = return_crps_change <= return_crps_allowance

    stylized_changes: dict[str, float] = {}
    stylized_allowances: dict[str, float] = {}
    stylized_checks: dict[str, bool] = {}
    for name in STYLIZED_GUARD_METRICS:
        if name not in stylized_standard_errors:
            raise ValueError(f"stylized_standard_errors is missing {name}")
        standard_error = float(stylized_standard_errors[name])
        if not math.isfinite(standard_error) or standard_error < 0.0:
            raise ValueError(f"stylized standard error for {name} must be finite and non-negative")
        change = finite_metric(candidate_stylized, name) - finite_metric(
            baseline_stylized,
            name,
        )
        allowance = float(standard_error_budget) * standard_error
        stylized_changes[name] = float(change)
        stylized_allowances[name] = float(allowance)
        stylized_checks[name] = bool(change <= allowance)

    baseline_swd = finite_metric(baseline_stylized, "path_swd")
    candidate_swd = finite_metric(candidate_stylized, "path_swd")
    swd_factor = candidate_swd / max(baseline_swd, 1e-12)
    swd_check = swd_factor <= float(swd_factor_limit)
    constraints_pass = bool(
        coverage_check
        and return_crps_check
        and all(stylized_checks.values())
        and swd_check
    )
    return {
        "pass": bool(rv_crps_check and constraints_pass),
        "constraints_pass": constraints_pass,
        "rv_crps_check": bool(rv_crps_check),
        "rv_crps_change": float(rv_crps_change),
        "coverage_check": bool(coverage_check),
        "baseline_coverage_error": float(baseline_coverage_error),
        "candidate_coverage_error": float(candidate_coverage_error),
        "return_crps_check": bool(return_crps_check),
        "return_crps_change": float(return_crps_change),
        "return_crps_allowance": float(return_crps_allowance),
        "stylized_checks": stylized_checks,
        "stylized_error_changes": stylized_changes,
        "stylized_error_allowances": stylized_allowances,
        "standard_error_budget": float(standard_error_budget),
        "swd_check": bool(swd_check),
        "swd_factor": float(swd_factor),
        "swd_factor_limit": float(swd_factor_limit),
    }


def select_shadow_frontier_checkpoint(
    records: Sequence[dict[str, object]],
    *,
    objective: str = "rv_crps",
    nominal_coverage: float = 0.90,
) -> int:
    """Select a validation checkpoint among eligible post-update records.

    Step zero denotes the already fitted source baseline and is returned when
    no candidate checkpoint passes the complete validation gate.

    ``rv_crps`` preserves the original rule. ``coverage_error`` ranks eligible
    checkpoints by absolute distance to nominal RV coverage, then RV CRPS, then
    the earlier training step. Eligibility itself still requires an RV-CRPS
    improvement and all predeclared guards.
    """

    if objective not in {"rv_crps", "coverage_error"}:
        raise ValueError("objective must be 'rv_crps' or 'coverage_error'")
    if not math.isfinite(float(nominal_coverage)) or not 0.0 < float(
        nominal_coverage
    ) < 1.0:
        raise ValueError("nominal_coverage must lie in (0, 1)")

    candidates: list[tuple[float, ...]] = []
    seen_steps: set[int] = set()
    for record in records:
        if "step" not in record or "decision" not in record or "shadowing" not in record:
            raise ValueError("every frontier record requires step, decision, and shadowing")
        step = int(record["step"])
        if step <= 0 or step in seen_steps:
            raise ValueError("frontier candidate steps must be unique positive integers")
        seen_steps.add(step)
        decision = record["decision"]
        shadowing = record["shadowing"]
        if not isinstance(decision, dict) or not isinstance(shadowing, dict):
            raise ValueError("frontier decision and shadowing entries must be dictionaries")
        if bool(decision.get("pass", False)):
            rv_crps = float(shadowing["realized_volatility_crps"])
            if not math.isfinite(rv_crps):
                raise ValueError("eligible RV CRPS scores must be finite")
            if objective == "rv_crps":
                candidates.append((rv_crps, step))
            else:
                coverage = float(shadowing["realized_volatility_coverage_90"])
                if not math.isfinite(coverage):
                    raise ValueError("eligible RV coverage scores must be finite")
                candidates.append(
                    (abs(coverage - float(nominal_coverage)), rv_crps, step)
                )
    return int(min(candidates)[-1]) if candidates else 0


__all__ = [
    "STYLIZED_GUARD_METRICS",
    "bootstrap_stylized_error_standard_errors",
    "select_shadow_frontier_checkpoint",
    "shadow_frontier_decision",
]
