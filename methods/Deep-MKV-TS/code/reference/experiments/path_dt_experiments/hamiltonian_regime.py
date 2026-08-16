from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.controls import (
    HamiltonianControlInputs,
    SpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.discrepancies import PathFunctionalDiscrepancy
from deep_mkv_gen_path_dt.model import DiscreteMPModel, RolloutResult


REGIME_NAMES = ("low", "middle", "high")


@dataclass(frozen=True)
class HamiltonianRegimeAuditResult:
    """Frozen complete-MP Hamiltonian audit split by prefix-volatility regime."""

    summary: dict[str, object]
    artifacts: dict[str, torch.Tensor]
    rollout: RolloutResult


def _rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(values.detach().double().pow(2))).item())


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_flat = left.detach().double().reshape(-1)
    right_flat = right.detach().double().reshape(-1)
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(
        right_centered
    )
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return None
    return float((torch.dot(left_centered, right_centered) / denominator).item())


def _projection_fraction(actual: torch.Tensor, target: torch.Tensor) -> float | None:
    actual_flat = actual.detach().double().reshape(-1)
    target_flat = target.detach().double().reshape(-1)
    denominator = torch.dot(target_flat, target_flat)
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return None
    return float((torch.dot(actual_flat, target_flat) / denominator).item())


def _tensor_summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().double().reshape(-1)
    quantiles = torch.quantile(
        flat,
        torch.tensor((0.05, 0.50, 0.95), dtype=torch.float64, device=flat.device),
    )
    return {
        "mean": float(flat.mean().item()),
        "rms": float(torch.sqrt(torch.mean(flat.pow(2))).item()),
        "std": float(flat.std(unbiased=False).item()),
        "q05": float(quantiles[0].item()),
        "q50": float(quantiles[1].item()),
        "q95": float(quantiles[2].item()),
    }


def _signal_summary(
    fitted: torch.Tensor,
    target: torch.Tensor,
    components: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    target_rms = _rms(target)
    error_rms = _rms(fitted - target)
    report: dict[str, object] = {
        "fitted": _tensor_summary(fitted),
        "target": _tensor_summary(target),
        "fitted_to_target_rms_ratio": _rms(fitted)
        / max(target_rms, torch.finfo(torch.float64).eps),
        "relative_rmse": error_rms
        / max(target_rms, torch.finfo(torch.float64).eps),
        "fitted_target_correlation": _correlation(fitted, target),
        "fitted_target_projection_fraction": _projection_fraction(fitted, target),
        "components": {},
    }
    component_report: dict[str, object] = {}
    for name, values in components.items():
        component_report[name] = {
            **_tensor_summary(values),
            "to_full_target_rms_ratio": _rms(values)
            / max(target_rms, torch.finfo(torch.float64).eps),
            "cosine_with_full_target": _correlation(values, target),
        }
    report["components"] = component_report
    if "discrepancy" in components and "running_cost" in components:
        discrepancy = components["discrepancy"]
        running = components["running_cost"]
        report["running_to_discrepancy_rms_ratio"] = _rms(running) / max(
            _rms(discrepancy), torch.finfo(torch.float64).eps
        )
        report["running_discrepancy_correlation"] = _correlation(
            running, discrepancy
        )
    return report


def prefix_volatility_regimes(
    paths: torch.Tensor,
    target_paths: torch.Tensor,
    *,
    split_step: int,
    prefix_window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign paths to fixed target-prefix RMS-volatility tertiles."""

    if paths.ndim != 3 or target_paths.ndim != 3:
        raise ValueError("paths and target_paths must have shape (B, N + 1, d)")
    split_step = int(split_step)
    prefix_window = int(prefix_window)
    if split_step < 1 or split_step >= int(paths.shape[1]):
        raise ValueError("split_step must identify a non-initial path point")
    if prefix_window < 1 or prefix_window > split_step:
        raise ValueError("prefix_window must fit before split_step")
    if split_step >= int(target_paths.shape[1]):
        raise ValueError("target paths do not reach split_step")

    def prefix_rms(values: torch.Tensor) -> torch.Tensor:
        returns = values[:, 1 : split_step + 1, :] - values[:, :split_step, :]
        return torch.sqrt(
            torch.mean(returns[:, -prefix_window:, :].double().pow(2), dim=(1, 2))
        )

    target_prefix = prefix_rms(target_paths)
    thresholds = torch.quantile(
        target_prefix,
        torch.tensor(
            (1.0 / 3.0, 2.0 / 3.0),
            device=target_prefix.device,
            dtype=target_prefix.dtype,
        ),
    )
    path_prefix = prefix_rms(paths)
    regimes = torch.bucketize(path_prefix, thresholds)
    return regimes, thresholds, path_prefix


def _controls_from_signals(
    *,
    control: SpecificEntropyDiagonalControl,
    paths: torch.Tensor,
    p: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    values = []
    for step_index in range(int(p.shape[1])):
        values.append(
            control(
                HamiltonianControlInputs(
                    step_index=step_index,
                    x_prefix=paths[:, : step_index + 1, :],
                    expected_adjoint_next=p[:, step_index, :],
                    expected_adjoint_noise_next=r[:, step_index, :],
                )
            )
        )
    return torch.stack(values, dim=1)


def _projected_signals(
    *,
    model: DiscreteMPModel,
    rollout: RolloutResult,
    target_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    discrepancy: PathFunctionalDiscrepancy | None = None,
    include_running_cost: bool,
    apply_noise_control_variate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    pathwise, source_metrics = model._path_functional_adjoint_targets(
        generated_path=rollout.paths,
        controls=rollout.controls,
        target_path=target_paths,
        training=training,
        discrepancy=discrepancy,
        include_running_cost=include_running_cost,
    )
    with torch.no_grad():
        p, p_metadata = model._project_pathwise_target(
            frozen_generated_path=rollout.paths.detach(),
            pathwise_target=pathwise.detach(),
            training=training,
        )
    control_variate = (
        model._noise_target_control_variate(
            training=training,
            raw_adjoint_prediction=rollout.expected_adjoint_next,
        )
        if bool(apply_noise_control_variate)
        else None
    )
    pathwise_r = model._adjoint_noise_target(
        pathwise_adjoint_target=pathwise.detach(),
        moments_noise_target_dim=int(rollout.expected_adjoint_noise_next.shape[-1]),
        noise=rollout.noise,
        control_variate=control_variate,
    )
    with torch.no_grad():
        r, r_metadata = model._project_pathwise_target(
            frozen_generated_path=rollout.paths.detach(),
            pathwise_target=pathwise_r,
            training=training,
            noise_adjoint=True,
        )
    metadata = {
        **{f"source_{name}": float(value) for name, value in source_metrics.items()},
        **{f"p_{name}": float(value) for name, value in p_metadata.items()},
        **{f"r_{name}": float(value) for name, value in r_metadata.items()},
    }
    return p.detach(), r.detach(), metadata


def _future_rv(paths: torch.Tensor, *, split_step: int, future_horizon: int) -> torch.Tensor:
    returns = (
        paths[:, split_step + 1 : split_step + future_horizon + 1, :]
        - paths[:, split_step : split_step + future_horizon, :]
    )
    return torch.sqrt(torch.mean(returns.double().pow(2), dim=(1, 2)))


def _regime_law_summary(
    paths: torch.Tensor,
    regimes: torch.Tensor,
    *,
    split_step: int,
    future_horizon: int,
) -> dict[str, object]:
    future_rv = _future_rv(
        paths, split_step=split_step, future_horizon=future_horizon
    )
    result: dict[str, object] = {}
    for index, name in enumerate(REGIME_NAMES):
        mask = regimes == index
        values = future_rv[mask]
        result[name] = {
            "count": int(mask.sum().item()),
            "future_realized_volatility": _tensor_summary(values),
            "future_realized_volatility_iqr": float(
                (
                    torch.quantile(values, 0.75) - torch.quantile(values, 0.25)
                ).item()
            ),
        }
    return result


def _slice_regime_summary(
    *,
    mask: torch.Tensor,
    time_slice: slice,
    fitted_p: torch.Tensor,
    fitted_r: torch.Tensor,
    target_p: torch.Tensor,
    target_r: torch.Tensor,
    p_components: Mapping[str, torch.Tensor],
    r_components: Mapping[str, torch.Tensor],
    controls: torch.Tensor,
    oracle_controls: torch.Tensor,
    discrepancy_oracle_controls: torch.Tensor,
    target_eta_sensitivity_controls: Mapping[str, torch.Tensor],
    fitted_eta_sensitivity_controls: Mapping[str, torch.Tensor],
    reference_controls: torch.Tensor,
    sigma_refs: torch.Tensor,
    baseline_controls: torch.Tensor | None,
    control: SpecificEntropyDiagonalControl,
) -> dict[str, object]:
    state_dim = int(target_p.shape[-1])

    def select(values: torch.Tensor) -> torch.Tensor:
        return values[mask, time_slice, :]

    fitted_p_used = select(fitted_p)
    fitted_r_used = select(fitted_r)
    target_p_used = select(target_p)
    target_r_used = select(target_r)
    control_used = select(controls)
    oracle_used = select(oracle_controls)
    discrepancy_oracle_used = select(discrepancy_oracle_controls)
    reference_used = select(reference_controls)
    sigma_ref = select(sigma_refs)
    alpha = control_used[..., :state_dim]
    sigma = control_used[..., state_dim:]
    oracle_alpha = oracle_used[..., :state_dim]
    oracle_sigma = oracle_used[..., state_dim:]
    discrepancy_oracle_alpha = discrepancy_oracle_used[..., :state_dim]
    discrepancy_oracle_sigma = discrepancy_oracle_used[..., state_dim:]
    reference_alpha = reference_used[..., :state_dim]
    reference_sigma = reference_used[..., state_dim:]
    ratio = sigma / sigma_ref
    oracle_ratio = oracle_sigma / sigma_ref
    sqrt_dt = math.sqrt(float(control.dt))
    dt = float(control.dt)

    drift_cost = 0.5 * alpha.pow(2)
    volatility_cost = float(control.eta) * (ratio - 1.0 - torch.log(ratio))
    drift_work = target_p_used * alpha
    volatility_work = target_r_used * (sigma - sigma_ref) / sqrt_dt
    hamiltonian_delta = drift_cost + volatility_cost + drift_work + volatility_work

    oracle_drift_cost = 0.5 * oracle_alpha.pow(2)
    oracle_volatility_cost = float(control.eta) * (
        oracle_ratio - 1.0 - torch.log(oracle_ratio)
    )
    oracle_hamiltonian_delta = (
        oracle_drift_cost
        + oracle_volatility_cost
        + target_p_used * oracle_alpha
        + target_r_used * (oracle_sigma - sigma_ref) / sqrt_dt
    )
    target_alpha_residual = alpha + target_p_used
    target_sigma_residual = (
        float(control.eta) * (1.0 / sigma_ref - 1.0 / sigma)
        + target_r_used / sqrt_dt
    )
    fitted_alpha_residual = alpha + fitted_p_used
    fitted_sigma_residual = (
        float(control.eta) * (1.0 / sigma_ref - 1.0 / sigma)
        + fitted_r_used / sqrt_dt
    )

    report: dict[str, object] = {
        "p": _signal_summary(
            fitted_p_used,
            target_p_used,
            {name: select(value) for name, value in p_components.items()},
        ),
        "r": _signal_summary(
            fitted_r_used,
            target_r_used,
            {name: select(value) for name, value in r_components.items()},
        ),
        "controls": {
            "alpha": _tensor_summary(alpha),
            "sigma": _tensor_summary(sigma),
            "sigma_reference": _tensor_summary(reference_sigma),
            "sigma_to_reference_ratio": _tensor_summary(ratio),
            "oracle_alpha": _tensor_summary(oracle_alpha),
            "oracle_sigma": _tensor_summary(oracle_sigma),
            "oracle_sigma_to_reference_ratio": _tensor_summary(oracle_ratio),
            "oracle_alpha_change_from_current": _tensor_summary(
                oracle_alpha - alpha
            ),
            "oracle_sigma_change_from_current": _tensor_summary(
                oracle_sigma - sigma
            ),
            "running_source_effect_on_oracle_alpha": _tensor_summary(
                oracle_alpha - discrepancy_oracle_alpha
            ),
            "running_source_effect_on_oracle_sigma": _tensor_summary(
                oracle_sigma - discrepancy_oracle_sigma
            ),
            "alpha_signal_transmission_fraction": _projection_fraction(
                alpha - reference_alpha, oracle_alpha - reference_alpha
            ),
            "sigma_signal_transmission_fraction": _projection_fraction(
                sigma - reference_sigma, oracle_sigma - reference_sigma
            ),
            "alpha_current_oracle_correction_correlation": _correlation(
                alpha - reference_alpha, oracle_alpha - reference_alpha
            ),
            "sigma_current_oracle_correction_correlation": _correlation(
                sigma - reference_sigma, oracle_sigma - reference_sigma
            ),
            "target_r_requests_higher_sigma_fraction": float(
                (target_r_used < 0.0).double().mean().item()
            ),
            "oracle_sigma_above_reference_fraction": float(
                (oracle_sigma > reference_sigma).double().mean().item()
            ),
            "sigma_cap_fraction": (
                0.0
                if control.sigma_max is None
                else float(
                    (
                        sigma
                        >= float(control.sigma_max)
                        - max(1e-7, 1e-5 * float(control.sigma_max))
                    )
                    .double()
                    .mean()
                    .item()
                )
            ),
            "oracle_sigma_cap_fraction": (
                0.0
                if control.sigma_max is None
                else float(
                    (
                        oracle_sigma
                        >= float(control.sigma_max)
                        - max(1e-7, 1e-5 * float(control.sigma_max))
                    )
                    .double()
                    .mean()
                    .item()
                )
            ),
            "target_eta_sensitivity": {
                name: {
                    "sigma": _tensor_summary(select(values)[..., state_dim:]),
                    "sigma_change_from_eta_1": _tensor_summary(
                        select(values)[..., state_dim:] - oracle_sigma
                    ),
                }
                for name, values in target_eta_sensitivity_controls.items()
            },
            "fitted_policy_eta_sensitivity": {
                name: {
                    "sigma": _tensor_summary(select(values)[..., state_dim:]),
                    "sigma_change_from_eta_1": _tensor_summary(
                        select(values)[..., state_dim:] - sigma
                    ),
                }
                for name, values in fitted_eta_sensitivity_controls.items()
            },
        },
        "running_cost": {
            "drift_value": float((dt * drift_cost.sum(dim=(1, 2))).mean().item()),
            "volatility_value": float(
                (dt * volatility_cost.sum(dim=(1, 2))).mean().item()
            ),
            "volatility_to_drift_ratio": float(
                volatility_cost.sum().item()
                / max(drift_cost.sum().item(), torch.finfo(torch.float64).eps)
            ),
        },
        "hamiltonian_balance": {
            "drift_cost": float((dt * drift_cost.sum(dim=(1, 2))).mean().item()),
            "volatility_entropy_cost": float(
                (dt * volatility_cost.sum(dim=(1, 2))).mean().item()
            ),
            "p_alpha_work": float((dt * drift_work.sum(dim=(1, 2))).mean().item()),
            "r_sigma_increment_work": float(
                (dt * volatility_work.sum(dim=(1, 2))).mean().item()
            ),
            "current_delta_from_reference": float(
                (dt * hamiltonian_delta.sum(dim=(1, 2))).mean().item()
            ),
            "oracle_delta_from_reference": float(
                (dt * oracle_hamiltonian_delta.sum(dim=(1, 2))).mean().item()
            ),
            "current_minus_oracle_gap": float(
                (
                    dt
                    * (hamiltonian_delta - oracle_hamiltonian_delta).sum(
                        dim=(1, 2)
                    )
                )
                .mean()
                .item()
            ),
            "target_alpha_stationarity_rms": _rms(target_alpha_residual),
            "target_sigma_stationarity_rms": _rms(target_sigma_residual),
            "fitted_alpha_stationarity_rms": _rms(fitted_alpha_residual),
            "fitted_sigma_stationarity_rms": _rms(fitted_sigma_residual),
        },
    }
    if baseline_controls is not None:
        baseline = select(baseline_controls)
        report["controls"]["accepted_alpha_change_from_source"] = _tensor_summary(
            alpha - baseline[..., :state_dim]
        )
        report["controls"]["accepted_sigma_change_from_source"] = _tensor_summary(
            sigma - baseline[..., state_dim:]
        )
    return report


def audit_specific_entropy_hamiltonian_by_regime(
    *,
    model: DiscreteMPModel,
    target_paths: torch.Tensor,
    x0: torch.Tensor,
    noise: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    split_step: int,
    prefix_window: int,
    future_horizon: int,
    isolated_discrepancies: Mapping[str, PathFunctionalDiscrepancy] | None = None,
    baseline_rollout: RolloutResult | None = None,
) -> HamiltonianRegimeAuditResult:
    """Audit fitted versus projected complete-MP controls on one frozen bank.

    Regimes use fixed target-law prefix-volatility tertiles.  The ``oracle``
    controls below are not a Heston oracle: they are the pointwise Hamiltonian
    minimizers implied by ridge-projected complete-path P/R targets on this
    frozen empirical problem.
    """

    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise ValueError("audit requires SpecificEntropyDiagonalControl")
    split_step = int(split_step)
    prefix_window = int(prefix_window)
    future_horizon = int(future_horizon)
    if future_horizon < 1 or split_step + future_horizon > int(model.grid.num_steps):
        raise ValueError("future_horizon must fit after split_step")
    target = target_paths.to(device=model.device, dtype=model.dtype)
    x0_used = x0.to(device=model.device, dtype=model.dtype)
    noise_used = noise.to(device=model.device, dtype=model.dtype)
    with torch.no_grad():
        rollout = model.rollout(x0=x0_used, noise=noise_used)

    full_p, full_r, full_metadata = _projected_signals(
        model=model,
        rollout=rollout,
        target_paths=target,
        training=training,
        include_running_cost=True,
    )
    if str(training.noise_target_control_variate) == "none":
        full_without_cv_p = full_p
        full_without_cv_r = full_r
        full_without_cv_metadata = full_metadata
    else:
        (
            full_without_cv_p,
            full_without_cv_r,
            full_without_cv_metadata,
        ) = _projected_signals(
            model=model,
            rollout=rollout,
            target_paths=target,
            training=training,
            include_running_cost=True,
            apply_noise_control_variate=False,
        )
    discrepancy_p, discrepancy_r, discrepancy_metadata = _projected_signals(
        model=model,
        rollout=rollout,
        target_paths=target,
        training=training,
        include_running_cost=False,
        apply_noise_control_variate=False,
    )
    p_components: dict[str, torch.Tensor] = {
        "discrepancy": discrepancy_p,
        "running_cost": full_without_cv_p - discrepancy_p,
    }
    r_components: dict[str, torch.Tensor] = {
        "discrepancy": discrepancy_r,
        "running_cost": full_without_cv_r - discrepancy_r,
    }
    if str(training.noise_target_control_variate) != "none":
        r_components["noise_control_variate"] = full_r - full_without_cv_r
    component_metadata: dict[str, dict[str, float]] = {}
    for name, discrepancy in (isolated_discrepancies or {}).items():
        component_p, component_r, metadata = _projected_signals(
            model=model,
            rollout=rollout,
            target_paths=target,
            training=training,
            discrepancy=discrepancy,
            include_running_cost=False,
            apply_noise_control_variate=False,
        )
        p_components[str(name)] = component_p
        r_components[str(name)] = component_r
        component_metadata[str(name)] = metadata

    control = model.control_map
    with torch.no_grad():
        oracle_controls = _controls_from_signals(
            control=control,
            paths=rollout.paths,
            p=full_p,
            r=full_r,
        )
        discrepancy_oracle_controls = _controls_from_signals(
            control=control,
            paths=rollout.paths,
            p=discrepancy_p,
            r=discrepancy_r,
        )
        target_eta_sensitivity_controls = {}
        fitted_eta_sensitivity_controls = {}
        for eta_scale in (0.25, 0.5, 2.0, 4.0):
            scaled_control = SpecificEntropyDiagonalControl(
                dt=float(control.dt),
                reference_kernel=control.reference_kernel,
                eta=float(control.eta) * eta_scale,
                sigma_min=float(control.sigma_min),
                sigma_max=(
                    None if control.sigma_max is None else float(control.sigma_max)
                ),
                denominator_eps=float(control.denominator_eps),
            )
            target_eta_sensitivity_controls[
                f"eta_x{eta_scale:g}"
            ] = _controls_from_signals(
                control=scaled_control,
                paths=rollout.paths,
                p=full_p,
                r=full_r,
            )
            fitted_eta_sensitivity_controls[
                f"eta_x{eta_scale:g}"
            ] = _controls_from_signals(
                control=scaled_control,
                paths=rollout.paths,
                p=rollout.expected_adjoint_next,
                r=rollout.expected_adjoint_noise_next,
            )
        sigma_refs = torch.stack(
            [
                control._reference_sigma(
                    rollout.paths[:, : step + 1, :], step_index=step
                )
                for step in range(int(model.grid.num_steps))
            ],
            dim=1,
        )
    state_dim = int(model.architecture.state_dim)
    reference_controls = torch.cat(
        [torch.zeros_like(sigma_refs), sigma_refs], dim=-1
    )
    regimes, thresholds, prefix_rv = prefix_volatility_regimes(
        rollout.paths,
        target,
        split_step=split_step,
        prefix_window=prefix_window,
    )
    target_regimes, _, target_prefix_rv = prefix_volatility_regimes(
        target,
        target,
        split_step=split_step,
        prefix_window=prefix_window,
    )
    baseline_controls = None
    if baseline_rollout is not None:
        if tuple(baseline_rollout.controls.shape) != tuple(rollout.controls.shape):
            raise ValueError("baseline rollout must align with audited rollout")
        baseline_controls = baseline_rollout.controls.to(
            device=rollout.controls.device, dtype=rollout.controls.dtype
        )

    regimes_report: dict[str, object] = {}
    window = slice(split_step, split_step + future_horizon)
    at_split = slice(split_step, split_step + 1)
    for regime_index, regime_name in enumerate(REGIME_NAMES):
        mask = regimes == regime_index
        if int(mask.sum().item()) == 0:
            raise ValueError("every generated prefix-volatility regime must be populated")
        regimes_report[regime_name] = {
            "count": int(mask.sum().item()),
            "prefix_realized_volatility": _tensor_summary(prefix_rv[mask]),
            "at_split": _slice_regime_summary(
                mask=mask,
                time_slice=at_split,
                fitted_p=rollout.expected_adjoint_next,
                fitted_r=rollout.expected_adjoint_noise_next,
                target_p=full_p,
                target_r=full_r,
                p_components=p_components,
                r_components=r_components,
                controls=rollout.controls,
                oracle_controls=oracle_controls,
                discrepancy_oracle_controls=discrepancy_oracle_controls,
                target_eta_sensitivity_controls=target_eta_sensitivity_controls,
                fitted_eta_sensitivity_controls=fitted_eta_sensitivity_controls,
                reference_controls=reference_controls,
                sigma_refs=sigma_refs,
                baseline_controls=baseline_controls,
                control=control,
            ),
            "future_control_window": _slice_regime_summary(
                mask=mask,
                time_slice=window,
                fitted_p=rollout.expected_adjoint_next,
                fitted_r=rollout.expected_adjoint_noise_next,
                target_p=full_p,
                target_r=full_r,
                p_components=p_components,
                r_components=r_components,
                controls=rollout.controls,
                oracle_controls=oracle_controls,
                discrepancy_oracle_controls=discrepancy_oracle_controls,
                target_eta_sensitivity_controls=target_eta_sensitivity_controls,
                fitted_eta_sensitivity_controls=fitted_eta_sensitivity_controls,
                reference_controls=reference_controls,
                sigma_refs=sigma_refs,
                baseline_controls=baseline_controls,
                control=control,
            ),
        }

    generated_law = _regime_law_summary(
        rollout.paths,
        regimes,
        split_step=split_step,
        future_horizon=future_horizon,
    )
    target_law = _regime_law_summary(
        target,
        target_regimes,
        split_step=split_step,
        future_horizon=future_horizon,
    )
    for name in REGIME_NAMES:
        generated = generated_law[name]
        target_used = target_law[name]
        regimes_report[name]["future_law"] = {
            "generated": generated,
            "target": target_used,
            "generated_to_target_rv_std_ratio": float(
                generated["future_realized_volatility"]["std"]
                / max(
                    target_used["future_realized_volatility"]["std"],
                    torch.finfo(torch.float64).eps,
                )
            ),
            "generated_to_target_rv_iqr_ratio": float(
                generated["future_realized_volatility_iqr"]
                / max(
                    target_used["future_realized_volatility_iqr"],
                    torch.finfo(torch.float64).eps,
                )
            ),
        }

    summary: dict[str, object] = {
        "protocol": {
            "complete_path_dependent_adjoint": True,
            "running_cost": "specific_entropy",
            "regime_definition": "fixed target-prefix RMS-volatility tertiles",
            "hamiltonian_oracle_definition": (
                "specific-entropy pointwise minimizer from ridge-projected "
                "complete-path P/R targets; not a Heston-law oracle"
            ),
            "same_noise_as_baseline": baseline_rollout is not None,
            "neural_loss_changed": False,
            "model_parameters_changed": False,
        },
        "config": {
            "batch_size": int(rollout.paths.shape[0]),
            "split_step": split_step,
            "prefix_window": prefix_window,
            "future_horizon": future_horizon,
            "dt": float(control.dt),
            "eta": float(control.eta),
            "sigma_min": float(control.sigma_min),
            "sigma_max": (
                None if control.sigma_max is None else float(control.sigma_max)
            ),
            "state_dim": state_dim,
        },
        "regime_thresholds": [float(value.item()) for value in thresholds],
        "projection_metadata": {
            "complete": full_metadata,
            "complete_without_noise_control_variate": full_without_cv_metadata,
            "discrepancy_only": discrepancy_metadata,
            "isolated_discrepancies": component_metadata,
        },
        "regimes": regimes_report,
    }
    artifacts = {
        "paths": rollout.paths.detach().cpu(),
        "controls": rollout.controls.detach().cpu(),
        "noise": rollout.noise.detach().cpu(),
        "fitted_p": rollout.expected_adjoint_next.detach().cpu(),
        "fitted_r": rollout.expected_adjoint_noise_next.detach().cpu(),
        "target_p": full_p.detach().cpu(),
        "target_r": full_r.detach().cpu(),
        "discrepancy_p": discrepancy_p.detach().cpu(),
        "discrepancy_r": discrepancy_r.detach().cpu(),
        "running_cost_p": (full_p - discrepancy_p).detach().cpu(),
        "running_cost_r": (full_r - discrepancy_r).detach().cpu(),
        "oracle_controls": oracle_controls.detach().cpu(),
        "discrepancy_oracle_controls": discrepancy_oracle_controls.detach().cpu(),
        "sigma_reference": sigma_refs.detach().cpu(),
        "regime_indices": regimes.detach().cpu(),
        "regime_thresholds": thresholds.detach().cpu(),
        "prefix_realized_volatility": prefix_rv.detach().cpu(),
        "target_prefix_realized_volatility": target_prefix_rv.detach().cpu(),
    }
    for name, values in p_components.items():
        if name not in {"discrepancy", "running_cost"}:
            artifacts[f"{name}_p"] = values.detach().cpu()
    for name, values in r_components.items():
        if name not in {"discrepancy", "running_cost"}:
            artifacts[f"{name}_r"] = values.detach().cpu()
    return HamiltonianRegimeAuditResult(
        summary=summary,
        artifacts=artifacts,
        rollout=rollout,
    )


__all__ = [
    "HamiltonianRegimeAuditResult",
    "REGIME_NAMES",
    "audit_specific_entropy_hamiltonian_by_regime",
    "prefix_volatility_regimes",
]
