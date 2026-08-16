from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class LockedEarlyVolatilityProposal:
    budget_index: int
    target_mean_kl: float
    coefficient: float
    block_index: int
    signal: str = "log_sigma"
    method: str = "full"


def locked_early_volatility_proposal(
    report: Mapping[str, object],
    *,
    budget_index: int = 1,
) -> LockedEarlyVolatilityProposal:
    """Select the prespecified early-volatility full-control proposal."""

    if list(report.get("methods", [])) != ["full"]:
        raise ValueError("proposal provenance must contain only the full method")
    results = report.get("results")
    if not isinstance(results, Mapping) or not isinstance(results.get("full"), list):
        raise ValueError("proposal provenance is missing full-method results")
    matches = [
        row
        for row in results["full"]
        if str(row.get("signal")) == "log_sigma"
        and int(row.get("block_index", -1)) == 0
        and int(row.get("budget_index", -1)) == int(budget_index)
    ]
    if len(matches) != 1:
        raise ValueError("proposal provenance does not identify exactly one locked row")
    row = matches[0]
    budget = float(row["target_mean_kl"])
    coefficient = float(row["selected_parameter"])
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("locked KL budget must be positive and finite")
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("locked proposal coefficient must be positive and finite")
    calibration = row.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("locked proposal is missing fit-bank calibration")
    if not bool(calibration.get("fit_bank_only")):
        raise ValueError("locked proposal was not calibrated on the fit bank only")
    if not bool(calibration.get("mean_budget_exhausted")):
        raise ValueError("locked proposal did not exhaust its mean-KL budget")
    baseline = report.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("proposal provenance is missing its baseline")
    for bank in ("fit", "confirmation"):
        candidate = row.get(bank)
        bank_baseline = baseline.get(bank)
        if not isinstance(candidate, Mapping) or not isinstance(bank_baseline, Mapping):
            raise ValueError(f"proposal provenance is missing {bank} results")
        checks = (
            float(candidate["objective_change"]) < 0.0,
            float(candidate["refreshed_target_volatility_kkt_rms"])
            < float(bank_baseline["refreshed_target_volatility_kkt_rms"]),
            bool(candidate["q95_ceiling_passed"]),
            bool(candidate["maximum_ceiling_passed"]),
        )
        if not all(checks):
            raise ValueError(f"locked proposal did not pass its {bank} alignment gates")
    return LockedEarlyVolatilityProposal(
        budget_index=int(budget_index),
        target_mean_kl=budget,
        coefficient=coefficient,
        block_index=0,
    )


def strict_complete_cycle_decision(
    *,
    precheck: Mapping[str, Mapping[str, object]],
    fresh_direction: Mapping[str, Mapping[str, object]],
    complete_cycle: Mapping[str, object],
    minimum_merit_relative_improvement: float = 1e-3,
) -> dict[str, object]:
    """Apply the locked two-bank success rule after adjoint refitting."""

    threshold = float(minimum_merit_relative_improvement)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum merit improvement must be finite and nonnegative")
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    for bank in ("fit", "confirmation"):
        before = precheck[bank]
        fresh = fresh_direction[bank]
        cycle_bank = complete_cycle[bank]
        cycle = cycle_bank["complete_cycle"]
        before_safety = before["candidate_control_safety"]
        fresh_safety = fresh["candidate_control_safety"]
        before_safe = bool(
            float(before_safety["controls_finite"]) == 1.0
            and float(before_safety["specific_entropy_denominator_margin_min"]) > 0.0
        )
        fresh_safe = bool(
            float(fresh_safety["controls_finite"]) == 1.0
            and float(fresh_safety["specific_entropy_denominator_margin_min"]) > 0.0
        )
        precheck_passed = bool(
            float(before["objective_change"]) < 0.0
            and float(before["refreshed_volatility_kkt_change"]) < 0.0
            and bool(before["q95_ceiling_passed"])
            and bool(before["maximum_ceiling_passed"])
            and before_safe
        )
        fresh_passed = bool(
            float(fresh["objective_change"]) < 0.0
            and float(fresh["refreshed_volatility_kkt_change"]) < 0.0
            and bool(fresh["q95_ceiling_passed"])
            and bool(fresh["maximum_ceiling_passed"])
            and fresh_safe
        )
        merit_improvement = float(cycle["merit_relative_improvement"])
        merit_passed = merit_improvement >= threshold
        checks[f"{bank}_precheck_alignment"] = precheck_passed
        checks[f"{bank}_fresh_alignment"] = fresh_passed
        checks[f"{bank}_post_refit_merit"] = merit_passed
        details[bank] = {
            "precheck_passed": precheck_passed,
            "fresh_alignment_passed": fresh_passed,
            "post_refit_merit_relative_improvement": merit_improvement,
            "post_refit_merit_passed": merit_passed,
            "precheck_control_safe": before_safe,
            "fresh_control_safe": fresh_safe,
            "component_relative_improvements": cycle[
                "component_relative_improvements"
            ],
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "details": details,
        "minimum_merit_relative_improvement": threshold,
        "interpretation": (
            "pursue_alignment_gated_block_coordinate_mp"
            if all(checks.values())
            else "stop_update_geometry_solver_family"
        ),
    }


__all__ = [
    "LockedEarlyVolatilityProposal",
    "locked_early_volatility_proposal",
    "strict_complete_cycle_decision",
]
