from __future__ import annotations

import math
from typing import Mapping

import torch

from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from path_dt_experiments.lifted_collocation_solver import LawEvaluation


def _rms(value: torch.Tensor) -> float:
    return float(torch.mean(value.detach().double().pow(2)).sqrt().item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.detach().double().reshape(-1)
    y = right.detach().double().reshape(-1)
    denominator = x.norm() * y.norm()
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return 1.0 if _rms(left - right) <= 1e-14 else 0.0
    return float((torch.dot(x, y) / denominator).item())


def first_operator_alignment(
    *,
    model: DiscreteMPModel,
    evaluation: LawEvaluation,
    fitted_controls: torch.Tensor,
) -> dict[str, float]:
    """Compare fitted and regenerated-target MP directions at one fixed law."""

    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise TypeError("first-operator alignment requires specific entropy")
    target_controls = model.control_map.controls_from_moments(
        paths=evaluation.paths,
        expected_adjoint_next=evaluation.target_p,
        expected_adjoint_noise_next=evaluation.target_r,
    )
    current = evaluation.controls
    state_dim = int(model.architecture.state_dim)
    fitted_drift = fitted_controls[..., :state_dim] - current[..., :state_dim]
    target_drift = target_controls[..., :state_dim] - current[..., :state_dim]
    fitted_log_vol = torch.log(fitted_controls[..., state_dim:]) - torch.log(
        current[..., state_dim:]
    )
    target_log_vol = torch.log(target_controls[..., state_dim:]) - torch.log(
        current[..., state_dim:]
    )
    fitted_target_drift = (
        fitted_controls[..., :state_dim] - target_controls[..., :state_dim]
    )
    fitted_target_log_vol = torch.log(fitted_controls[..., state_dim:]) - torch.log(
        target_controls[..., state_dim:]
    )
    return {
        "drift_direction_cosine": _cosine(fitted_drift, target_drift),
        "log_volatility_direction_cosine": _cosine(
            fitted_log_vol, target_log_vol
        ),
        "target_drift_move_rms": _rms(target_drift),
        "target_log_volatility_move_rms": _rms(target_log_vol),
        "fitted_target_drift_consensus_rms": _rms(fitted_target_drift),
        "fitted_target_log_volatility_consensus_rms": _rms(
            fitted_target_log_vol
        ),
    }


def _relative_gap(left: float, right: float) -> float:
    denominator = max(
        abs(float(left)), abs(float(right)), torch.finfo(torch.float64).eps
    )
    return abs(float(left) - float(right)) / denominator


def residual_phase_one_gate(
    results: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    """Locked gate for the residual-volatility objective variants.

    The thresholds are intentionally defined independently of candidate
    outcomes.  A candidate must reduce the pathological volatility response,
    preserve a positively aligned mathematical target, and reproduce its
    response on an independent bank.
    """

    control_name = "current_objective_eta1"
    candidates = (
        "residual_conditional_eta1",
        "residual_conditional_eta4",
    )
    thresholds = {
        "maximum_log_volatility_move_ratio": 0.80,
        "minimum_log_volatility_move_ratio": 0.25,
        "maximum_upper_activity_ratio": 0.80,
        "minimum_upper_activity_absolute_drop": 0.05,
        "minimum_log_volatility_alignment": 0.10,
        "maximum_relative_ce_ratio": 1.25,
        "minimum_r_ce_explained_energy": 0.0,
        "maximum_target_control_log_volatility_consensus_ratio": 1.0,
        "maximum_cross_bank_move_relative_gap": 0.35,
        "maximum_cross_bank_upper_activity_gap": 0.10,
    }
    reports: dict[str, object] = {}
    for candidate in candidates:
        checks: dict[str, bool] = {}
        ratios: dict[str, object] = {}
        for bank in ("fit", "confirmation"):
            baseline = results[control_name][bank]
            row = results[candidate][bank]
            move_ratio = float(row["log_volatility_move_rms"]) / max(
                float(baseline["log_volatility_move_rms"]),
                torch.finfo(torch.float64).eps,
            )
            base_upper = float(baseline["induced_sigma_upper_activity"])
            candidate_upper = float(row["induced_sigma_upper_activity"])
            upper_ratio = candidate_upper / max(
                base_upper, torch.finfo(torch.float64).eps
            )
            p_relative = float(row["p_ce_relative_residual"])
            r_relative = float(row["r_ce_relative_residual"])
            p_relative_ratio = p_relative / max(
                float(baseline["p_ce_relative_residual"]),
                torch.finfo(torch.float64).eps,
            )
            r_relative_ratio = r_relative / max(
                float(baseline["r_ce_relative_residual"]),
                torch.finfo(torch.float64).eps,
            )
            control_consensus_ratio = float(
                row["fitted_target_log_volatility_consensus_rms"]
            ) / max(
                float(
                    baseline["fitted_target_log_volatility_consensus_rms"]
                ),
                torch.finfo(torch.float64).eps,
            )
            ratios[bank] = {
                "log_volatility_move": move_ratio,
                "upper_activity": upper_ratio,
                "p_ce_relative_residual": p_relative_ratio,
                "r_ce_relative_residual": r_relative_ratio,
                "target_control_log_volatility_consensus": (
                    control_consensus_ratio
                ),
            }
            checks[f"{bank}_volatility_move_reduced"] = bool(
                move_ratio <= thresholds["maximum_log_volatility_move_ratio"]
            )
            checks[f"{bank}_volatility_move_not_suppressed"] = bool(
                move_ratio >= thresholds["minimum_log_volatility_move_ratio"]
            )
            checks[f"{bank}_upper_activity_reduced"] = bool(
                upper_ratio <= thresholds["maximum_upper_activity_ratio"]
                or candidate_upper
                <= max(
                    0.0,
                    base_upper
                    - thresholds["minimum_upper_activity_absolute_drop"],
                )
            )
            checks[f"{bank}_target_alignment_positive"] = bool(
                float(row["log_volatility_direction_cosine"])
                >= thresholds["minimum_log_volatility_alignment"]
            )
            checks[f"{bank}_relative_ce_not_materially_worse"] = bool(
                p_relative_ratio <= thresholds["maximum_relative_ce_ratio"]
                and r_relative_ratio <= thresholds["maximum_relative_ce_ratio"]
            )
            checks[f"{bank}_r_ce_has_explained_energy"] = bool(
                float(row["r_ce_explained_energy"])
                >= thresholds["minimum_r_ce_explained_energy"]
            )
            checks[f"{bank}_target_control_consensus_not_worse"] = bool(
                control_consensus_ratio
                <= thresholds[
                    "maximum_target_control_log_volatility_consensus_ratio"
                ]
            )
        fit = results[candidate]["fit"]
        confirmation = results[candidate]["confirmation"]
        move_gap = _relative_gap(
            float(fit["log_volatility_move_rms"]),
            float(confirmation["log_volatility_move_rms"]),
        )
        upper_gap = abs(
            float(fit["induced_sigma_upper_activity"])
            - float(confirmation["induced_sigma_upper_activity"])
        )
        checks["cross_bank_volatility_move_stable"] = bool(
            move_gap <= thresholds["maximum_cross_bank_move_relative_gap"]
        )
        checks["cross_bank_upper_activity_stable"] = bool(
            upper_gap <= thresholds["maximum_cross_bank_upper_activity_gap"]
        )
        reports[candidate] = {
            "passed": all(checks.values()),
            "checks": checks,
            "ratios_over_current": ratios,
            "cross_bank": {
                "log_volatility_move_relative_gap": move_gap,
                "upper_activity_absolute_gap": upper_gap,
            },
        }
    passing = [
        name for name in candidates if bool(reports[name]["passed"])  # type: ignore[index]
    ]
    return {
        "passed": bool(passing),
        "passing_variants": passing,
        "variants": reports,
        "thresholds": thresholds,
        "next_phase": (
            "reduced_spectral_audit" if passing else "stop_residual_objective_branch"
        ),
    }


__all__ = ["first_operator_alignment", "residual_phase_one_gate"]
