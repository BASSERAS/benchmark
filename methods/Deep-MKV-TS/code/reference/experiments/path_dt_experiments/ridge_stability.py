from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl
from path_dt_experiments.frozen_backward_learnability import (
    fit_ridge_teacher_transfer,
    paired_signal_regime_report,
    prediction_regime_report,
)
from path_dt_experiments.hamiltonian_regime import REGIME_NAMES


@dataclass(frozen=True)
class RidgeStabilitySweepResult:
    report: dict[str, object]
    artifacts: dict[str, object]


def _validate_values(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError("ridge_values must be non-empty")
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError("ridge_values must be finite and >= 0")
    if len(set(result)) != len(result):
        raise ValueError("ridge_values must not contain duplicates")
    return tuple(sorted(result))


def _conditional_variation_rms(values: torch.Tensor) -> float:
    centered = values.detach().double() - values.detach().double().mean(
        dim=0, keepdim=True
    )
    return float(torch.sqrt(torch.mean(centered.pow(2))).item())


def _structure_report(
    values: torch.Tensor,
    regimes: torch.Tensor,
    *,
    split_step: int,
    future_horizon: int,
) -> dict[str, object]:
    window = slice(int(split_step), int(split_step) + int(future_horizon))

    def summary(selected: torch.Tensor) -> dict[str, float]:
        values64 = selected.detach().double()
        return {
            "rms": float(torch.sqrt(torch.mean(values64.pow(2))).item()),
            "conditional_variation_rms": _conditional_variation_rms(values64),
        }

    return {
        "global": summary(values),
        "regimes": {
            name: {
                "count": int((regimes == index).sum().item()),
                "future_control_window": summary(
                    values[regimes == index, window, :]
                ),
            }
            for index, name in enumerate(REGIME_NAMES)
        },
    }


def _specific_entropy_controls(
    *,
    control: SpecificEntropyDiagonalControl,
    paths: torch.Tensor,
    p: torch.Tensor,
    r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma_ref = torch.stack(
        [
            control._reference_sigma(paths[:, : step + 1, :], step_index=step)
            for step in range(int(p.shape[1]))
        ],
        dim=1,
    )
    denominator = float(control.eta) / sigma_ref + r / math.sqrt(float(control.dt))
    if control.sigma_max is None:
        if bool(torch.any(denominator <= float(control.denominator_eps)).item()):
            raise ValueError("ridge teacher implies an unbounded specific-entropy sigma")
        sigma = float(control.eta) / denominator.clamp_min(
            float(control.denominator_eps)
        )
    else:
        unconstrained = float(control.eta) / denominator.clamp_min(
            float(control.denominator_eps)
        )
        sigma = torch.where(
            denominator > float(control.denominator_eps),
            unconstrained,
            torch.full_like(unconstrained, float(control.sigma_max)),
        ).clamp_max(float(control.sigma_max))
    sigma = sigma.clamp_min(float(control.sigma_min))
    return torch.cat([-p, sigma], dim=-1), sigma_ref


def _cap_fraction(
    sigma: torch.Tensor,
    *,
    sigma_max: float | None,
) -> float:
    if sigma_max is None:
        return 0.0
    tolerance = max(1e-7, 1e-5 * abs(float(sigma_max)))
    return float((sigma >= float(sigma_max) - tolerance).double().mean().item())


def _add_retention_ratios(
    rows: list[dict[str, object]],
    *,
    baseline_ridge: float,
) -> None:
    baseline = next(
        (row for row in rows if float(row["ridge"]) == float(baseline_ridge)),
        None,
    )
    if baseline is None:
        raise ValueError("baseline_ridge must be included in ridge_values")

    def ratio(value: float, reference: float) -> float:
        return value / max(reference, torch.finfo(torch.float64).eps)

    for row in rows:
        retention: dict[str, object] = {"global": {}, "regimes": {}}
        for signal in ("p", "r", "sigma"):
            current = row["heldout_independent_structure"][signal]
            reference = baseline["heldout_independent_structure"][signal]
            retention["global"][signal] = ratio(
                float(current["global"]["conditional_variation_rms"]),
                float(reference["global"]["conditional_variation_rms"]),
            )
        for name in REGIME_NAMES:
            retention["regimes"][name] = {}
            for signal in ("p", "r", "sigma"):
                current = row["heldout_independent_structure"][signal]["regimes"][
                    name
                ]["future_control_window"]
                reference = baseline["heldout_independent_structure"][signal][
                    "regimes"
                ][name]["future_control_window"]
                retention["regimes"][name][signal] = ratio(
                    float(current["conditional_variation_rms"]),
                    float(reference["conditional_variation_rms"]),
                )
        row["conditional_variation_retention_vs_baseline"] = retention


def sweep_frozen_ridge_stability(
    *,
    fit_paths: torch.Tensor,
    heldout_paths: torch.Tensor,
    fit_pathwise_p: torch.Tensor,
    fit_pathwise_r: torch.Tensor,
    heldout_pathwise_p: torch.Tensor,
    heldout_pathwise_r: torch.Tensor,
    fit_regimes: torch.Tensor,
    heldout_regimes: torch.Tensor,
    control: SpecificEntropyDiagonalControl,
    ridge_values: Sequence[float],
    baseline_ridge: float,
    split_step: int,
    future_horizon: int,
) -> RidgeStabilitySweepResult:
    """Compare independently fitted ridge CE teachers on fixed path banks."""

    values = _validate_values(ridge_values)
    baseline_ridge = float(baseline_ridge)
    if baseline_ridge not in values:
        raise ValueError("baseline_ridge must be included in ridge_values")
    if int(split_step) < 1 or int(future_horizon) < 1:
        raise ValueError("split_step and future_horizon must be >= 1")
    if int(split_step) + int(future_horizon) > int(fit_pathwise_p.shape[1]):
        raise ValueError("future_horizon must fit after split_step")
    device = fit_paths.device
    heldout_paths = heldout_paths.to(device=device, dtype=fit_paths.dtype)
    tensors = (
        fit_pathwise_p,
        fit_pathwise_r,
        heldout_pathwise_p,
        heldout_pathwise_r,
    )
    fit_pathwise_p, fit_pathwise_r, heldout_pathwise_p, heldout_pathwise_r = (
        tensor.to(device=device, dtype=fit_paths.dtype) for tensor in tensors
    )
    fit_regimes = fit_regimes.to(device=device)
    heldout_regimes = heldout_regimes.to(device=device)

    rows: list[dict[str, object]] = []
    tensor_artifacts: dict[str, object] = {}
    sigma_reference: torch.Tensor | None = None
    for ridge in values:
        (
            fit_teacher_p,
            fit_teacher_r,
            heldout_transferred_p,
            heldout_transferred_r,
            fit_metadata,
        ) = fit_ridge_teacher_transfer(
            fit_paths=fit_paths,
            fit_p=fit_pathwise_p,
            fit_r=fit_pathwise_r,
            eval_paths=heldout_paths,
            ridge=ridge,
        )
        (
            heldout_independent_p,
            heldout_independent_r,
            _heldout_duplicate_p,
            _heldout_duplicate_r,
            heldout_metadata,
        ) = fit_ridge_teacher_transfer(
            fit_paths=heldout_paths,
            fit_p=heldout_pathwise_p,
            fit_r=heldout_pathwise_r,
            eval_paths=heldout_paths,
            ridge=ridge,
        )
        transferred_controls, current_sigma_reference = _specific_entropy_controls(
            control=control,
            paths=heldout_paths,
            p=heldout_transferred_p,
            r=heldout_transferred_r,
        )
        independent_controls, _ = _specific_entropy_controls(
            control=control,
            paths=heldout_paths,
            p=heldout_independent_p,
            r=heldout_independent_r,
        )
        sigma_reference = current_sigma_reference
        state_dim = int(heldout_paths.shape[-1])
        transferred_sigma = transferred_controls[..., state_dim:]
        independent_sigma = independent_controls[..., state_dim:]
        teacher_agreement = paired_signal_regime_report(
            prediction_p=heldout_transferred_p,
            prediction_r=heldout_transferred_r,
            target_p=heldout_independent_p,
            target_r=heldout_independent_r,
            regimes=heldout_regimes,
            split_step=int(split_step),
            future_horizon=int(future_horizon),
        )
        sigma_agreement = prediction_regime_report(
            transferred_sigma,
            independent_sigma,
            heldout_regimes,
            split_step=int(split_step),
            future_horizon=int(future_horizon),
        )
        fit_projection = paired_signal_regime_report(
            prediction_p=fit_teacher_p,
            prediction_r=fit_teacher_r,
            target_p=fit_pathwise_p,
            target_r=fit_pathwise_r,
            regimes=fit_regimes,
            split_step=int(split_step),
            future_horizon=int(future_horizon),
        )
        heldout_projection = paired_signal_regime_report(
            prediction_p=heldout_independent_p,
            prediction_r=heldout_independent_r,
            target_p=heldout_pathwise_p,
            target_r=heldout_pathwise_r,
            regimes=heldout_regimes,
            split_step=int(split_step),
            future_horizon=int(future_horizon),
        )
        row: dict[str, object] = {
            "ridge": ridge,
            "ridge_per_fit_path": ridge / float(fit_paths.shape[0]),
            "fit_teacher_metadata": fit_metadata,
            "heldout_teacher_metadata": heldout_metadata,
            "teacher_agreement": teacher_agreement,
            "sigma_agreement": sigma_agreement,
            "fit_projection_of_raw_targets": fit_projection,
            "heldout_projection_of_raw_targets": heldout_projection,
            "heldout_independent_structure": {
                "p": _structure_report(
                    heldout_independent_p,
                    heldout_regimes,
                    split_step=int(split_step),
                    future_horizon=int(future_horizon),
                ),
                "r": _structure_report(
                    heldout_independent_r,
                    heldout_regimes,
                    split_step=int(split_step),
                    future_horizon=int(future_horizon),
                ),
                "sigma": _structure_report(
                    independent_sigma,
                    heldout_regimes,
                    split_step=int(split_step),
                    future_horizon=int(future_horizon),
                ),
            },
            "control_bounds": {
                "transferred_sigma_cap_fraction": _cap_fraction(
                    transferred_sigma, sigma_max=control.sigma_max
                ),
                "independent_sigma_cap_fraction": _cap_fraction(
                    independent_sigma, sigma_max=control.sigma_max
                ),
            },
        }
        rows.append(row)
        key = f"ridge_{ridge:g}".replace(".", "p")
        tensor_artifacts[key] = {
            "fit_teacher_p": fit_teacher_p.detach().cpu(),
            "fit_teacher_r": fit_teacher_r.detach().cpu(),
            "heldout_transferred_p": heldout_transferred_p.detach().cpu(),
            "heldout_transferred_r": heldout_transferred_r.detach().cpu(),
            "heldout_independent_p": heldout_independent_p.detach().cpu(),
            "heldout_independent_r": heldout_independent_r.detach().cpu(),
            "heldout_transferred_sigma": transferred_sigma.detach().cpu(),
            "heldout_independent_sigma": independent_sigma.detach().cpu(),
        }
    _add_retention_ratios(rows, baseline_ridge=baseline_ridge)
    report: dict[str, object] = {
        "protocol": {
            "forward_paths_and_pathwise_targets_frozen": True,
            "independent_fit_and_heldout_ridge_teachers": True,
            "same_banks_at_every_ridge": True,
            "neural_network_trained": False,
            "running_cost_or_discrepancy_changed": False,
            "conditional_variation_retention_reference": baseline_ridge,
        },
        "config": {
            "ridge_values": list(values),
            "baseline_ridge": baseline_ridge,
            "fit_bank_size": int(fit_paths.shape[0]),
            "heldout_bank_size": int(heldout_paths.shape[0]),
            "split_step": int(split_step),
            "future_horizon": int(future_horizon),
            "eta": float(control.eta),
            "sigma_min": float(control.sigma_min),
            "sigma_max": (
                None if control.sigma_max is None else float(control.sigma_max)
            ),
        },
        "rows": rows,
    }
    artifacts = {
        "ridge_teachers": tensor_artifacts,
        "sigma_reference": (
            None if sigma_reference is None else sigma_reference.detach().cpu()
        ),
    }
    return RidgeStabilitySweepResult(report=report, artifacts=artifacts)


__all__ = [
    "RidgeStabilitySweepResult",
    "sweep_frozen_ridge_stability",
]
