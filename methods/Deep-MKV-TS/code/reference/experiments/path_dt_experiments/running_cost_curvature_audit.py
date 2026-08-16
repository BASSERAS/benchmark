from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from deep_mkv_gen_path_dt.controls import (
    GaussianRelativeEntropyDiagonalControl,
    SpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from path_dt_experiments.lifted_collocation_solver import (
    CausalControlPolicy,
    LawEvaluation,
    policy_moments_on_paths,
)


CurvatureMatchedControl = (
    SpecificEntropyDiagonalControl | GaussianRelativeEntropyDiagonalControl
)


@dataclass(frozen=True)
class GenericProjectedVolatilityKKT:
    residual: torch.Tensor
    gradient: torch.Tensor
    sensitivity_abs: torch.Tensor
    active_set: torch.Tensor


def curvature_matched_gaussian_eta(specific_eta: float) -> float:
    value = float(specific_eta)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("specific_eta must be positive and finite")
    return 0.5 * value


def log_volatility_curvature_at_reference(
    control: CurvatureMatchedControl,
) -> float:
    if isinstance(control, SpecificEntropyDiagonalControl):
        return float(control.eta)
    if isinstance(control, GaussianRelativeEntropyDiagonalControl):
        return 2.0 * float(control.eta)
    raise TypeError("unsupported curvature-audit control")


def transition_kernel_kl(
    *,
    dt: float,
    controls: torch.Tensor,
    current_controls: torch.Tensor,
) -> torch.Tensor:
    if tuple(controls.shape) != tuple(current_controls.shape):
        raise ValueError("transition controls must have matching shapes")
    if controls.ndim != 3 or int(controls.shape[-1]) % 2 != 0:
        raise ValueError("controls must have shape (B, N, 2*d)")
    state_dim = int(controls.shape[-1]) // 2
    alpha = controls[..., :state_dim]
    sigma = controls[..., state_dim:]
    current_alpha = current_controls[..., :state_dim]
    current_sigma = current_controls[..., state_dim:]
    if bool(torch.any(sigma <= 0.0).item()) or bool(
        torch.any(current_sigma <= 0.0).item()
    ):
        raise ValueError("transition volatilities must be positive")
    variance_ratio = sigma.pow(2) / current_sigma.pow(2)
    mean_term = float(dt) * (alpha - current_alpha).pow(2) / current_sigma.pow(2)
    return 0.5 * (
        variance_ratio - 1.0 - torch.log(variance_ratio) + mean_term
    ).sum(dim=-1)


def projected_volatility_kkt(
    *,
    model: DiscreteMPModel,
    paths: torch.Tensor,
    sigma: torch.Tensor,
    moment_r: torch.Tensor,
    gamma: float = 1.0,
    activity_tolerance: float = 1e-6,
) -> GenericProjectedVolatilityKKT:
    control = model.control_map
    if not isinstance(
        control,
        (SpecificEntropyDiagonalControl, GaussianRelativeEntropyDiagonalControl),
    ):
        raise TypeError("unsupported curvature-audit control")
    if control.sigma_max is None:
        raise ValueError("curvature audit requires a finite upper volatility bound")
    expected = (
        int(paths.shape[0]),
        int(model.grid.num_steps),
        int(model.architecture.state_dim),
    )
    if tuple(sigma.shape) != expected or tuple(moment_r.shape) != expected:
        raise ValueError(f"sigma and moment_r must have shape {expected}")
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value <= 0.0:
        raise ValueError("gamma must be positive and finite")
    sigma_ref = control._reference_sigmas_all(paths)
    noise_term = moment_r / math.sqrt(float(control.dt))
    ratio = sigma / sigma_ref
    if isinstance(control, SpecificEntropyDiagonalControl):
        gradient = float(control.eta) * (ratio - 1.0) + noise_term * sigma
        interior_curvature = float(control.eta) * ratio + noise_term * sigma
    else:
        gradient = float(control.eta) * (ratio.pow(2) - 1.0) + noise_term * sigma
        interior_curvature = (
            2.0 * float(control.eta) * ratio.pow(2) + noise_term * sigma
        )
    lower = float(control.sigma_min)
    upper = float(control.sigma_max)
    log_sigma = torch.log(sigma)
    projected = (log_sigma - gamma_value * gradient).clamp(
        min=math.log(lower), max=math.log(upper)
    )
    residual = log_sigma - projected
    active = torch.zeros_like(sigma, dtype=torch.int8)
    active = torch.where(
        sigma <= lower * (1.0 + float(activity_tolerance)),
        -torch.ones_like(active),
        active,
    )
    active = torch.where(
        sigma >= upper * (1.0 - float(activity_tolerance)),
        torch.ones_like(active),
        active,
    )
    raw_sensitivity = sigma / (
        math.sqrt(float(control.dt))
        * interior_curvature.abs().clamp_min(torch.finfo(sigma.dtype).eps)
    )
    outward_at_lower = (active < 0) & (gradient > 0.0)
    outward_at_upper = (active > 0) & (gradient < 0.0)
    effective_sensitivity = torch.where(
        outward_at_lower | outward_at_upper,
        torch.zeros_like(raw_sensitivity),
        raw_sensitivity,
    )
    return GenericProjectedVolatilityKKT(
        residual=residual.detach(),
        gradient=gradient.detach(),
        sensitivity_abs=effective_sensitivity.detach(),
        active_set=active.detach(),
    )


def _rms(value: torch.Tensor) -> float:
    return float(torch.mean(value.detach().double().pow(2)).sqrt().item())


def _distribution_summary(value: torch.Tensor) -> dict[str, float]:
    flattened = value.detach().double().reshape(-1)
    return {
        "mean": float(flattened.mean().item()),
        "q95": float(torch.quantile(flattened, 0.95).item()),
        "maximum": float(flattened.max().item()),
    }


def target_agreement(
    specific: torch.Tensor,
    gaussian: torch.Tensor,
) -> dict[str, float]:
    """Compare curvature-matched mathematical targets on one frozen law."""

    if tuple(specific.shape) != tuple(gaussian.shape):
        raise ValueError("target tensors must have matching shapes")
    left = specific.detach().double().reshape(-1)
    right = gaussian.detach().double().reshape(-1)
    if left.numel() == 0:
        raise ValueError("target tensors cannot be empty")
    left_rms = torch.mean(left.pow(2)).sqrt()
    right_rms = torch.mean(right.pow(2)).sqrt()
    difference_rms = torch.mean((right - left).pow(2)).sqrt()
    epsilon = torch.finfo(torch.float64).eps
    cosine_denominator = left.norm() * right.norm()
    cosine = torch.where(
        cosine_denominator > epsilon,
        torch.dot(left, right) / cosine_denominator,
        torch.ones((), dtype=torch.float64),
    )
    return {
        "specific_rms": float(left_rms.item()),
        "gaussian_rms": float(right_rms.item()),
        "difference_rms": float(difference_rms.item()),
        "relative_difference": float(
            (difference_rms / left_rms.clamp_min(epsilon)).item()
        ),
        "cosine": float(cosine.item()),
    }


def fixed_law_cost_audit(
    *,
    model: DiscreteMPModel,
    evaluation: LawEvaluation,
    adjoint_policy: CausalControlPolicy,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    control = model.control_map
    if not isinstance(
        control,
        (SpecificEntropyDiagonalControl, GaussianRelativeEntropyDiagonalControl),
    ):
        raise TypeError("unsupported curvature-audit control")
    with torch.no_grad():
        p, r, induced_controls = policy_moments_on_paths(
            model=model,
            policy=adjoint_policy,
            paths=evaluation.paths,
        )
    state_dim = int(model.architecture.state_dim)
    current_alpha = evaluation.controls[..., :state_dim]
    current_sigma = evaluation.controls[..., state_dim:]
    induced_alpha = induced_controls[..., :state_dim]
    induced_sigma = induced_controls[..., state_dim:]
    p_residual = p - evaluation.target_p
    r_residual = r - evaluation.target_r
    current_fitted = projected_volatility_kkt(
        model=model,
        paths=evaluation.paths,
        sigma=current_sigma,
        moment_r=r,
    )
    current_target = projected_volatility_kkt(
        model=model,
        paths=evaluation.paths,
        sigma=current_sigma,
        moment_r=evaluation.target_r,
    )
    induced_fitted = projected_volatility_kkt(
        model=model,
        paths=evaluation.paths,
        sigma=induced_sigma,
        moment_r=r,
    )
    induced_target = projected_volatility_kkt(
        model=model,
        paths=evaluation.paths,
        sigma=induced_sigma,
        moment_r=evaluation.target_r,
    )
    kl = transition_kernel_kl(
        dt=float(model.grid.dt),
        controls=induced_controls,
        current_controls=evaluation.controls,
    )
    log_move = torch.log(induced_sigma) - torch.log(current_sigma)
    drift_move = induced_alpha - current_alpha
    p_target_centered = evaluation.target_p - evaluation.target_p.mean(dim=0, keepdim=True)
    r_target_centered = evaluation.target_r - evaluation.target_r.mean(dim=0, keepdim=True)

    def explained(residual: torch.Tensor, centered: torch.Tensor) -> float:
        denominator = float(torch.mean(centered.detach().double().pow(2)).item())
        numerator = float(torch.mean(residual.detach().double().pow(2)).item())
        return 1.0 - numerator / max(denominator, torch.finfo(torch.float64).eps)

    lower = float(control.sigma_min)
    upper = float(control.sigma_max)
    report = {
        "control_type": type(control).__name__,
        "volatility_eta": float(control.eta),
        "log_volatility_curvature_at_reference": (
            log_volatility_curvature_at_reference(control)
        ),
        "p_ce_residual_rms": _rms(p_residual),
        "r_ce_residual_rms": _rms(r_residual),
        "p_ce_explained_energy": explained(p_residual, p_target_centered),
        "r_ce_explained_energy": explained(r_residual, r_target_centered),
        "p_target_rms": _rms(evaluation.target_p),
        "r_target_rms": _rms(evaluation.target_r),
        "drift_move_rms": _rms(drift_move),
        "log_volatility_move_rms": _rms(log_move),
        "transition_kl": _distribution_summary(kl),
        "induced_sigma_lower_activity": float(
            torch.mean((induced_sigma <= lower * (1.0 + 1e-6)).float()).item()
        ),
        "induced_sigma_upper_activity": float(
            torch.mean((induced_sigma >= upper * (1.0 - 1e-6)).float()).item()
        ),
        "current_fitted_kkt_rms": _rms(current_fitted.residual),
        "current_target_kkt_rms": _rms(current_target.residual),
        "induced_fitted_kkt_rms": _rms(induced_fitted.residual),
        "induced_target_kkt_rms": _rms(induced_target.residual),
        "induced_fitted_sensitivity_abs": _distribution_summary(
            induced_fitted.sensitivity_abs
        ),
        "induced_target_sensitivity_abs": _distribution_summary(
            induced_target.sensitivity_abs
        ),
        "complete_objective_value": float(
            evaluation.objective["complete_objective_value"]
        ),
        "running_cost_value": float(evaluation.objective["running_cost_value"]),
        "discrepancy_objective_value": float(
            evaluation.objective["discrepancy_objective_value"]
        ),
    }
    return report, {
        "p": p.detach().cpu(),
        "r": r.detach().cpu(),
        "target_p": evaluation.target_p.detach().cpu(),
        "target_r": evaluation.target_r.detach().cpu(),
        "p_ce_residual": p_residual.detach().cpu(),
        "r_ce_residual": r_residual.detach().cpu(),
        "induced_controls": induced_controls.detach().cpu(),
        "current_controls": evaluation.controls.detach().cpu(),
        "log_volatility_move": log_move.detach().cpu(),
        "transition_kl": kl.detach().cpu(),
        "current_fitted_kkt": current_fitted.residual.cpu(),
        "current_target_kkt": current_target.residual.cpu(),
        "induced_fitted_kkt": induced_fitted.residual.cpu(),
        "induced_target_kkt": induced_target.residual.cpu(),
        "induced_fitted_sensitivity_abs": induced_fitted.sensitivity_abs.cpu(),
        "induced_target_sensitivity_abs": induced_target.sensitivity_abs.cpu(),
    }


def curvature_phase_one_gate(
    results: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    """Locked gate deciding whether Gaussian KL merits phase two."""

    checks: dict[str, bool] = {}
    ratios: dict[str, object] = {}
    for bank in ("fit", "confirmation"):
        specific = results["specific_entropy"][bank]
        gaussian = results["gaussian_relative_entropy"][bank]

        def ratio(path: tuple[str, ...]) -> float:
            left: object = gaussian
            right: object = specific
            for key in path:
                left = left[key]  # type: ignore[index]
                right = right[key]  # type: ignore[index]
            return float(left) / max(float(right), torch.finfo(torch.float64).eps)

        bank_ratios = {
            "sensitivity_q95": ratio(("induced_fitted_sensitivity_abs", "q95")),
            "transition_kl_q95": ratio(("transition_kl", "q95")),
            "transition_kl_max": ratio(("transition_kl", "maximum")),
            "upper_activity": ratio(("induced_sigma_upper_activity",)),
            "log_volatility_move_rms": ratio(("log_volatility_move_rms",)),
            "p_ce_residual_rms": ratio(("p_ce_residual_rms",)),
            "r_ce_residual_rms": ratio(("r_ce_residual_rms",)),
            "induced_target_kkt_rms": ratio(("induced_target_kkt_rms",)),
        }
        ratios[bank] = bank_ratios
        checks[f"{bank}_sensitivity_reduced"] = bool(
            bank_ratios["sensitivity_q95"] <= 0.75
        )
        checks[f"{bank}_tail_or_bound_reduced"] = bool(
            bank_ratios["transition_kl_q95"] <= 0.75
            or bank_ratios["transition_kl_max"] <= 0.75
            or bank_ratios["upper_activity"] <= 0.75
        )
        checks[f"{bank}_correction_not_suppressed"] = bool(
            bank_ratios["log_volatility_move_rms"] >= 0.50
        )
        checks[f"{bank}_ce_not_materially_worse"] = bool(
            bank_ratios["p_ce_residual_rms"] <= 1.25
            and bank_ratios["r_ce_residual_rms"] <= 1.25
        )
        checks[f"{bank}_target_kkt_not_worse"] = bool(
            bank_ratios["induced_target_kkt_rms"] <= 1.0
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "ratios_gaussian_over_specific": ratios,
        "thresholds": {
            "maximum_sensitivity_ratio": 0.75,
            "maximum_tail_or_bound_ratio": 0.75,
            "minimum_log_volatility_move_ratio": 0.50,
            "maximum_ce_residual_ratio": 1.25,
            "maximum_target_kkt_ratio": 1.0,
        },
        "next_phase": (
            "spectral_and_complete_cycle"
            if all(checks.values())
            else "stop_running_cost_solver_branch"
        ),
    }


__all__ = [
    "GenericProjectedVolatilityKKT",
    "curvature_matched_gaussian_eta",
    "curvature_phase_one_gate",
    "fixed_law_cost_audit",
    "log_volatility_curvature_at_reference",
    "projected_volatility_kkt",
    "target_agreement",
    "transition_kernel_kl",
]
