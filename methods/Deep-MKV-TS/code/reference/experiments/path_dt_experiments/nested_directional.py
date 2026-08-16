from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import math
from typing import Iterable

import torch
import torch.nn as nn

from deep_mkv_gen_path_dt import (
    DiscreteMPNestedTrainingConfig,
    DiscreteMPTrainingConfig,
)
from deep_mkv_gen_path_dt.controls import (
    DynamicsStepInputs,
    HamiltonianControlInputs,
    SpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from deep_mkv_gen_path_dt.networks import AdjointMoments
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals


DEFAULT_RELAXATIONS = (
    -0.125,
    -0.0625,
    -0.03125,
    -0.015625,
    -0.0078125,
    -0.00390625,
    -0.001953125,
    -0.0009765625,
    0.0,
    0.0009765625,
    0.001953125,
    0.00390625,
    0.0078125,
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
)

DEFAULT_MOMENT_RELAXATIONS = (
    0.0,
    0.0009765625,
    0.001953125,
    0.00390625,
    0.0078125,
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
)


class _BlendedAdjointMomentNetwork(nn.Module):
    """Blend old and candidate P/R outputs without mixing their parameters."""

    def __init__(
        self,
        *,
        old_network: nn.Module,
        candidate_network: nn.Module,
        p_relaxation: float,
        r_relaxation: float,
    ) -> None:
        super().__init__()
        self.old_network = old_network
        self.candidate_network = candidate_network
        self.p_relaxation = float(p_relaxation)
        self.r_relaxation = float(r_relaxation)

    def forward(self, paths: torch.Tensor) -> AdjointMoments:
        old = self.old_network(paths)
        candidate = self.candidate_network(paths)
        return AdjointMoments(
            expected_adjoint_next=(
                old.expected_adjoint_next
                + self.p_relaxation
                * (candidate.expected_adjoint_next - old.expected_adjoint_next)
            ),
            expected_adjoint_noise_next=(
                old.expected_adjoint_noise_next
                + self.r_relaxation
                * (
                    candidate.expected_adjoint_noise_next
                    - old.expected_adjoint_noise_next
                )
            ),
        )

    def forward_step(
        self,
        *,
        x_t: torch.Tensor,
        hidden: tuple[object | None, object | None] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[object, object]]:
        old_hidden, candidate_hidden = (None, None) if hidden is None else hidden
        old_p, old_r, next_old_hidden = self.old_network.forward_step(
            x_t=x_t,
            hidden=old_hidden,
        )
        candidate_p, candidate_r, next_candidate_hidden = (
            self.candidate_network.forward_step(
                x_t=x_t,
                hidden=candidate_hidden,
            )
        )
        return (
            old_p + self.p_relaxation * (candidate_p - old_p),
            old_r + self.r_relaxation * (candidate_r - old_r),
            (next_old_hidden, next_candidate_hidden),
        )


@dataclass(frozen=True)
class AdjointControlVariant:
    name: str
    p_scale: float
    r_scale: float
    interpretation: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("variant name must be non-empty")
        for name, value in (("p_scale", self.p_scale), ("r_scale", self.r_scale)):
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")


class _ScaledAdjointControl:
    def __init__(
        self,
        base: SpecificEntropyDiagonalControl,
        *,
        p_scale: float,
        r_scale: float,
    ) -> None:
        self.base = base
        self.p_scale = float(p_scale)
        self.r_scale = float(r_scale)
        self.sigma_min = base.sigma_min
        self.sigma_max = base.sigma_max

    def __call__(self, inputs: HamiltonianControlInputs) -> torch.Tensor:
        return self.base(
            HamiltonianControlInputs(
                step_index=int(inputs.step_index),
                x_prefix=inputs.x_prefix,
                expected_adjoint_next=self.p_scale * inputs.expected_adjoint_next,
                expected_adjoint_noise_next=(
                    self.r_scale * inputs.expected_adjoint_noise_next
                ),
                parameters=inputs.parameters,
            )
        )


def standard_sign_scaling_variants(*, dt: float) -> tuple[AdjointControlVariant, ...]:
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be a positive finite float")
    sqrt_dt = math.sqrt(float(dt))
    return (
        AdjointControlVariant("correct", 1.0, 1.0, "implemented maximum principle"),
        AdjointControlVariant("p_only", 1.0, 0.0, "suppress the fitted R correction"),
        AdjointControlVariant("r_only", 0.0, 1.0, "suppress the fitted P correction"),
        AdjointControlVariant("reverse_p", -1.0, 1.0, "reverse drift-adjoint sign"),
        AdjointControlVariant("reverse_r", 1.0, -1.0, "reverse noise-adjoint sign"),
        AdjointControlVariant("reverse_both", -1.0, -1.0, "reverse both signs"),
        AdjointControlVariant(
            "r_without_sqrt_dt_divisor",
            1.0,
            sqrt_dt,
            "replace R/sqrt(dt) by R",
        ),
        AdjointControlVariant(
            "r_with_dt_divisor",
            1.0,
            1.0 / sqrt_dt,
            "replace R/sqrt(dt) by R/dt",
        ),
        AdjointControlVariant(
            "p_times_dt",
            float(dt),
            1.0,
            "replace alpha=-P by alpha=-dt*P",
        ),
        AdjointControlVariant(
            "p_divided_by_dt",
            1.0 / float(dt),
            1.0,
            "replace alpha=-P by alpha=-P/dt",
        ),
    )


def _set_parameter_interpolation(
    parameters: list[torch.nn.Parameter],
    old_values: list[torch.Tensor],
    candidate_values: list[torch.Tensor],
    relaxation: float,
) -> None:
    with torch.no_grad():
        for parameter, old, candidate in zip(parameters, old_values, candidate_values):
            parameter.copy_(old + float(relaxation) * (candidate - old))


def summarize_objective_curve(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["relaxation"])
    by_rho = {float(row["relaxation"]): row for row in ordered}
    if 0.0 not in by_rho:
        raise ValueError("objective curve must contain relaxation zero")
    positive = sorted(value for value in by_rho if value > 0.0)
    negative = sorted((-value for value in by_rho if value < 0.0))
    shared = sorted(set(positive).intersection(negative))
    if not shared:
        raise ValueError("objective curve must contain a symmetric non-zero pair")
    h = float(shared[0])
    zero = by_rho[0.0]
    plus = by_rho[h]
    minus = by_rho[-h]
    summary: dict[str, float] = {"finite_difference_h": h}
    for metric in (
        "complete_objective_value",
        "discrepancy_objective_value",
        "running_cost_value",
    ):
        summary[f"{metric}_forward_derivative"] = (
            float(plus[metric]) - float(zero[metric])
        ) / h
        summary[f"{metric}_central_derivative"] = (
            float(plus[metric]) - float(minus[metric])
        ) / (2.0 * h)
        summary[f"{metric}_central_curvature"] = (
            float(plus[metric])
            - 2.0 * float(zero[metric])
            + float(minus[metric])
        ) / (h * h)
    finite_rows = [
        row
        for row in ordered
        if math.isfinite(float(row["complete_objective_value"]))
    ]
    best = min(finite_rows, key=lambda row: float(row["complete_objective_value"]))
    full = by_rho.get(1.0)
    summary.update(
        {
            "minimum_relaxation": float(best["relaxation"]),
            "minimum_complete_objective": float(best["complete_objective_value"]),
            "zero_complete_objective": float(zero["complete_objective_value"]),
            "full_complete_objective": (
                float("nan")
                if full is None
                else float(full["complete_objective_value"])
            ),
        }
    )
    return summary


def _controls_from_moments(
    *,
    control: SpecificEntropyDiagonalControl,
    paths: torch.Tensor,
    p: torch.Tensor,
    r: torch.Tensor,
    variant: AdjointControlVariant,
    parameters: object | None,
) -> torch.Tensor:
    values = []
    for step_index in range(int(p.shape[1])):
        values.append(
            control(
                HamiltonianControlInputs(
                    step_index=step_index,
                    x_prefix=paths[:, : step_index + 1, :],
                    expected_adjoint_next=float(variant.p_scale) * p[:, step_index, :],
                    expected_adjoint_noise_next=(
                        float(variant.r_scale) * r[:, step_index, :]
                    ),
                    parameters=parameters,
                )
            )
        )
    return torch.stack(values, dim=1)


def _control_impact_metrics(
    *,
    control: SpecificEntropyDiagonalControl,
    paths: torch.Tensor,
    controls: torch.Tensor,
    old_controls: torch.Tensor,
    r: torch.Tensor,
) -> dict[str, float]:
    """Measure the finite control displacement induced by a P/R proposal."""

    state_dim = int(paths.shape[-1])
    alpha = controls[..., :state_dim]
    sigma = controls[..., state_dim:]
    old_alpha = old_controls[..., :state_dim]
    old_sigma = old_controls[..., state_dim:]
    sigma_ref = control._reference_sigmas_all(paths)
    reference_denominator = float(control.eta) / sigma_ref
    denominator = reference_denominator + r / math.sqrt(float(control.dt))
    relative_denominator = denominator / reference_denominator
    alpha_delta = alpha - old_alpha
    log_sigma_delta = torch.log(sigma) - torch.log(old_sigma)
    sigma_delta = sigma - old_sigma
    denominator_flat = denominator.detach().double().reshape(-1)
    relative_flat = relative_denominator.detach().double().reshape(-1)
    quantile_levels = torch.tensor(
        (0.01, 0.05, 0.50, 0.95, 0.99),
        device=denominator_flat.device,
        dtype=denominator_flat.dtype,
    )
    denominator_quantiles = torch.quantile(denominator_flat, quantile_levels)
    relative_quantiles = torch.quantile(relative_flat, quantile_levels)
    lower_tolerance = max(1e-7, abs(float(control.sigma_min)) * 1e-5)
    lower_active = sigma <= float(control.sigma_min) + lower_tolerance
    upper_active = torch.zeros_like(lower_active)
    if control.sigma_max is not None:
        upper_tolerance = max(1e-7, abs(float(control.sigma_max)) * 1e-5)
        upper_active = sigma >= float(control.sigma_max) - upper_tolerance

    def rms(values: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean(values.detach().double().pow(2))).item())

    result = {
        "alpha_rms_change": rms(alpha_delta),
        "sigma_rms_change": rms(sigma_delta),
        "log_sigma_rms_change": rms(log_sigma_delta),
        "alpha_max_abs_change": float(alpha_delta.detach().abs().max().item()),
        "log_sigma_max_abs_change": float(
            log_sigma_delta.detach().abs().max().item()
        ),
        "sigma_floor_fraction": float(lower_active.float().mean().item()),
        "sigma_cap_fraction": float(upper_active.float().mean().item()),
        "denominator_nonpositive_fraction": float(
            (denominator <= float(control.denominator_eps)).float().mean().item()
        ),
        "denominator_min": float(denominator_flat.min().item()),
        "relative_denominator_min": float(relative_flat.min().item()),
    }
    for index, label in enumerate(("q01", "q05", "q50", "q95", "q99")):
        result[f"denominator_{label}"] = float(denominator_quantiles[index].item())
        result[f"relative_denominator_{label}"] = float(
            relative_quantiles[index].item()
        )
    return result


def _local_hamiltonian_metrics(
    *,
    control: SpecificEntropyDiagonalControl,
    paths: torch.Tensor,
    controls: torch.Tensor,
    old_controls: torch.Tensor,
    p_target: torch.Tensor,
    r_target: torch.Tensor,
) -> dict[str, float]:
    state_dim = int(paths.shape[-1])
    alpha = controls[..., :state_dim]
    sigma = controls[..., state_dim:]
    old_alpha = old_controls[..., :state_dim]
    old_sigma = old_controls[..., state_dim:]
    sigma_ref = torch.stack(
        [
            control._reference_sigma(paths[:, : step + 1, :], step_index=step)
            for step in range(int(p_target.shape[1]))
        ],
        dim=1,
    )
    sqrt_dt = math.sqrt(float(control.dt))
    ratio = sigma / sigma_ref
    integrand = (
        0.5 * alpha.pow(2)
        + float(control.eta) * (ratio - 1.0 - torch.log(ratio))
        + p_target * alpha
        + r_target * sigma / sqrt_dt
    )
    old_alpha_gradient = old_alpha + p_target
    old_sigma_gradient = (
        float(control.eta) * (1.0 / sigma_ref - 1.0 / old_sigma)
        + r_target / sqrt_dt
    )
    first_variation = float(control.dt) * (
        old_alpha_gradient * (alpha - old_alpha)
        + old_sigma_gradient * (sigma - old_sigma)
    ).sum(dim=(1, 2)).mean()
    stationarity_alpha = alpha + p_target
    stationarity_sigma = (
        float(control.eta) * (1.0 / sigma_ref - 1.0 / sigma)
        + r_target / sqrt_dt
    )
    lower_tolerance = max(1e-7, abs(float(control.sigma_min)) * 1e-5)
    lower_active = sigma <= float(control.sigma_min) + lower_tolerance
    upper_active = torch.zeros_like(lower_active)
    if control.sigma_max is not None:
        upper_tolerance = max(1e-7, abs(float(control.sigma_max)) * 1e-5)
        upper_active = sigma >= float(control.sigma_max) - upper_tolerance
    free = ~(lower_active | upper_active)
    sigma_kkt_violation = torch.where(
        lower_active,
        torch.clamp(-stationarity_sigma, min=0.0),
        torch.where(
            upper_active,
            torch.clamp(stationarity_sigma, min=0.0),
            stationarity_sigma.abs(),
        ),
    )
    free_sigma_residual = stationarity_sigma[free]
    return {
        "hamiltonian_value": float(
            (float(control.dt) * integrand.sum(dim=(1, 2)).mean()).item()
        ),
        "first_variation_from_old": float(first_variation.item()),
        "control_rms_change_from_old": float(
            torch.sqrt(torch.mean((controls - old_controls).pow(2))).item()
        ),
        "alpha_rms_change_from_old": float(
            torch.sqrt(torch.mean((alpha - old_alpha).pow(2))).item()
        ),
        "sigma_rms_change_from_old": float(
            torch.sqrt(torch.mean((sigma - old_sigma).pow(2))).item()
        ),
        "alpha_stationarity_rms": float(
            torch.sqrt(torch.mean(stationarity_alpha.pow(2))).item()
        ),
        "sigma_stationarity_rms": float(
            torch.sqrt(torch.mean(stationarity_sigma.pow(2))).item()
        ),
        "sigma_free_stationarity_rms": (
            0.0
            if int(free_sigma_residual.numel()) == 0
            else float(torch.sqrt(torch.mean(free_sigma_residual.pow(2))).item())
        ),
        "sigma_kkt_violation_rms": float(
            torch.sqrt(torch.mean(sigma_kkt_violation.pow(2))).item()
        ),
        "sigma_floor_fraction": float(lower_active.float().mean().item()),
        "sigma_cap_fraction": float(upper_active.float().mean().item()),
    }


def _comparison_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    error = actual.detach() - expected.detach()
    actual_flat = actual.detach().reshape(-1).double()
    expected_flat = expected.detach().reshape(-1).double()
    actual_norm = torch.linalg.vector_norm(actual_flat)
    expected_norm = torch.linalg.vector_norm(expected_flat)
    eps = torch.finfo(torch.float64).eps
    cosine = torch.dot(actual_flat, expected_flat) / (
        actual_norm.clamp_min(eps) * expected_norm.clamp_min(eps)
    )
    expected_rms = torch.sqrt(torch.mean(expected.detach().pow(2)))
    error_rms = torch.sqrt(torch.mean(error.pow(2)))
    return {
        f"{prefix}_actual_rms": float(torch.sqrt(torch.mean(actual.detach().pow(2))).item()),
        f"{prefix}_expected_rms": float(expected_rms.item()),
        f"{prefix}_error_rms": float(error_rms.item()),
        f"{prefix}_relative_rmse": float(
            (error_rms / expected_rms.clamp_min(torch.finfo(expected.dtype).eps)).item()
        ),
        f"{prefix}_cosine": float(cosine.item()),
        f"{prefix}_max_abs_error": float(error.abs().max().item()),
    }


def _discrete_chain_rule_audit(
    *,
    model: DiscreteMPModel,
    x0: torch.Tensor,
    noise: torch.Tensor,
    controls: torch.Tensor,
    expected_paths: torch.Tensor,
    target_batch: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    pathwise_target: torch.Tensor,
) -> dict[str, float]:
    controls_req = controls.detach().clone().requires_grad_(True)
    points = [x0.detach()]
    for step_index in range(int(model.grid.num_steps)):
        prefix = torch.stack(points, dim=1)
        next_state = model.dynamics_step(
            DynamicsStepInputs(
                step_index=step_index,
                x_prefix=prefix,
                control=controls_req[:, step_index, :],
                noise_next=noise[:, step_index, :],
                parameters=model.parameters,
            )
        )
        points.append(next_state)
    open_loop_paths = torch.stack(points, dim=1)
    objective, _ = model._complete_objective_metrics(
        generated_path=open_loop_paths,
        controls=controls_req,
        target_path=target_batch,
        training=training,
    )
    control_gradient = torch.autograd.grad(objective, controls_req)[0]
    state_dim = int(model.architecture.state_dim)
    batch_size = float(controls.shape[0])
    dt = float(model.grid.dt)
    sqrt_dt = math.sqrt(dt)
    alpha_chain = batch_size * control_gradient[..., :state_dim] / dt
    sigma_chain = batch_size * control_gradient[..., state_dim:] / sqrt_dt
    expected_sigma_chain = pathwise_target * noise
    control = model.control_map
    if not isinstance(control, SpecificEntropyDiagonalControl):
        raise ValueError("chain-rule audit requires SpecificEntropyDiagonalControl")
    with torch.no_grad():
        sigma_refs = torch.stack(
            [
                control._reference_sigma(
                    expected_paths[:, : step + 1, :],
                    step_index=step,
                )
                for step in range(int(model.grid.num_steps))
            ],
            dim=1,
        )
    metrics = {
        "open_loop_path_replay_max_abs_error": float(
            (open_loop_paths.detach() - expected_paths.detach())
            .abs()
            .max()
            .item()
        ),
        "old_alpha_rms": float(
            torch.sqrt(torch.mean(controls[..., :state_dim].detach().pow(2))).item()
        ),
        "old_sigma_reference_max_abs_error": float(
            (controls[..., state_dim:].detach() - sigma_refs).abs().max().item()
        ),
    }
    metrics.update(
        _comparison_metrics(
            alpha_chain,
            pathwise_target,
            prefix="alpha_chain_vs_pathwise_adjoint",
        )
    )
    metrics.update(
        _comparison_metrics(
            sigma_chain,
            expected_sigma_chain,
            prefix="sigma_chain_vs_adjoint_noise",
        )
    )
    return metrics


def diagnose_first_nested_direction(
    *,
    model: DiscreteMPModel,
    target_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    nested: DiscreteMPNestedTrainingConfig,
    relaxations: tuple[float, ...] = DEFAULT_RELAXATIONS,
    moment_relaxations: tuple[float, ...] = DEFAULT_MOMENT_RELAXATIONS,
    evaluation_batch_size: int | None = None,
    variants: tuple[AdjointControlVariant, ...] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Fit one frozen backward problem and audit its closed-loop direction."""

    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise ValueError("directional diagnostic requires SpecificEntropyDiagonalControl")
    if int(nested.outer_steps) != 1:
        raise ValueError("directional diagnostic requires nested.outer_steps=1")
    if 0.0 not in relaxations or 1.0 not in relaxations:
        raise ValueError("relaxations must contain zero and one")
    if len(set(float(value) for value in relaxations)) != len(relaxations):
        raise ValueError("relaxations must not contain duplicates")
    if any(not math.isfinite(float(value)) for value in relaxations):
        raise ValueError("relaxations must be finite")
    if 0.0 not in moment_relaxations or 1.0 not in moment_relaxations:
        raise ValueError("moment_relaxations must contain zero and one")
    if len(set(float(value) for value in moment_relaxations)) != len(
        moment_relaxations
    ):
        raise ValueError("moment_relaxations must not contain duplicates")
    if any(
        not math.isfinite(float(value))
        for value in moment_relaxations
    ):
        raise ValueError("moment_relaxations must be finite")
    evaluation_count = (
        int(nested.outer_batch_size)
        if evaluation_batch_size is None
        else int(evaluation_batch_size)
    )
    if evaluation_count < 2:
        raise ValueError("evaluation_batch_size must be >= 2")
    control = model.control_map
    variants_used = variants or standard_sign_scaling_variants(dt=float(control.dt))
    if not variants_used or variants_used[0].name != "correct":
        raise ValueError("the first control variant must be named correct")

    target_paths_device = target_paths.to(device=model.device, dtype=model.dtype)
    target_count, _ = model._validate_target_paths(target_paths_device)
    parameters = list(model.network.parameters())
    old_parameter_values = [parameter.detach().clone() for parameter in parameters]
    old_checkpoint = model.checkpoint_state()
    old_network = copy.deepcopy(model.network).eval()
    for module in old_network.modules():
        if isinstance(module, nn.GRU):
            module.flatten_parameters()

    nested_without_line_search = replace(
        nested,
        outer_objective_line_search=False,
    )
    fit_result = model.fit_nested(
        target_paths_device,
        training=training,
        nested=nested_without_line_search,
    )
    candidate_parameter_values = [
        parameter.detach().clone() for parameter in parameters
    ]
    candidate_checkpoint = model.checkpoint_state()
    candidate_network = copy.deepcopy(model.network).eval()
    for module in candidate_network.modules():
        if isinstance(module, nn.GRU):
            module.flatten_parameters()

    cpu_generator = torch.Generator(device="cpu")
    if nested.seed is None:
        cpu_generator.seed()
    else:
        cpu_generator.manual_seed(int(nested.seed))
    source_indices = model._sample_indices(
        size=target_count,
        count=evaluation_count,
        generator=cpu_generator,
    )
    target_indices = model._sample_indices(
        size=target_count,
        count=int(nested.outer_target_batch_size),
        generator=cpu_generator,
    )
    x0 = target_paths_device.index_select(0, source_indices.to(model.device))[:, 0, :]
    target_batch = target_paths_device.index_select(0, target_indices.to(model.device))
    outer_noise = sample_standard_normals(
        grid=model.grid,
        batch_size=evaluation_count,
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=derive_stream_seed(base_seed=nested.seed, stream_offset=200_000),
    )

    _set_parameter_interpolation(
        parameters,
        old_parameter_values,
        candidate_parameter_values,
        0.0,
    )
    with torch.no_grad():
        old_rollout = model.rollout(x0=x0, noise=outer_noise)
        old_normalized_moments = model.network(old_rollout.paths.detach())
        old_moments = model._denormalize_moments(old_normalized_moments)
    pathwise_target, target_metadata = model._path_functional_adjoint_targets(
        generated_path=old_rollout.paths,
        controls=old_rollout.controls,
        target_path=target_batch,
        training=training,
    )
    chain_rule_metrics = _discrete_chain_rule_audit(
        model=model,
        x0=x0,
        noise=outer_noise,
        controls=old_rollout.controls,
        expected_paths=old_rollout.paths,
        target_batch=target_batch,
        training=training,
        pathwise_target=pathwise_target.detach(),
    )
    frozen_paths = old_rollout.paths.detach()
    with torch.no_grad():
        p_target, p_projection_metadata = model._project_pathwise_target(
            frozen_generated_path=frozen_paths,
            pathwise_target=pathwise_target.detach(),
            training=training,
        )
    control_variate = model._noise_target_control_variate(
        training=training,
        raw_adjoint_prediction=old_moments.expected_adjoint_next,
    )
    pathwise_r_target = model._adjoint_noise_target(
        pathwise_adjoint_target=pathwise_target.detach(),
        moments_noise_target_dim=int(old_moments.expected_adjoint_noise_next.shape[-1]),
        noise=outer_noise,
        control_variate=control_variate,
    )
    with torch.no_grad():
        r_target, r_projection_metadata = model._project_pathwise_target(
            frozen_generated_path=frozen_paths,
            pathwise_target=pathwise_r_target,
            training=training,
            noise_adjoint=True,
        )
        _set_parameter_interpolation(
            parameters,
            old_parameter_values,
            candidate_parameter_values,
            1.0,
        )
        candidate_moments = model._denormalize_moments(model.network(frozen_paths))

    curve_rows: list[dict[str, object]] = []
    curve_summaries: dict[str, dict[str, float]] = {}
    local_rows: list[dict[str, object]] = []
    baseline_hamiltonian = _local_hamiltonian_metrics(
        control=control,
        paths=frozen_paths,
        controls=old_rollout.controls,
        old_controls=old_rollout.controls,
        p_target=p_target,
        r_target=r_target,
    )["hamiltonian_value"]

    original_control_map = model.control_map
    try:
        for variant in variants_used:
            model.control_map = _ScaledAdjointControl(
                control,
                p_scale=float(variant.p_scale),
                r_scale=float(variant.r_scale),
            )
            variant_rows: list[dict[str, float]] = []
            for relaxation in sorted(float(value) for value in relaxations):
                _set_parameter_interpolation(
                    parameters,
                    old_parameter_values,
                    candidate_parameter_values,
                    relaxation,
                )
                with torch.no_grad():
                    rollout = model.rollout(x0=x0, noise=outer_noise)
                    finite = bool(
                        torch.isfinite(rollout.paths).all().item()
                        and torch.isfinite(rollout.controls).all().item()
                    )
                    if finite:
                        _, objective_metrics = model._complete_objective_metrics(
                            generated_path=rollout.paths,
                            controls=rollout.controls,
                            target_path=target_batch,
                            training=training,
                        )
                        control_metrics = model._control_diagnostics(rollout.controls)
                        row = {
                            "relaxation": relaxation,
                            "complete_objective_value": float(
                                objective_metrics["complete_objective_value"]
                            ),
                            "discrepancy_objective_value": float(
                                objective_metrics["discrepancy_objective_value"]
                            ),
                            "running_cost_value": float(
                                objective_metrics["running_cost_value"]
                            ),
                            "running_cost_drift_value": float(
                                objective_metrics.get(
                                    "running_cost_drift_value", float("nan")
                                )
                            ),
                            "running_cost_volatility_value": float(
                                objective_metrics.get(
                                    "running_cost_volatility_value", float("nan")
                                )
                            ),
                            "path_rms_change": float(
                                torch.sqrt(
                                    torch.mean(
                                        (rollout.paths - old_rollout.paths).pow(2)
                                    )
                                ).item()
                            ),
                            "control_rms_change": float(
                                torch.sqrt(
                                    torch.mean(
                                        (rollout.controls - old_rollout.controls).pow(2)
                                    )
                                ).item()
                            ),
                            "sigma_mean": float(
                                control_metrics.get("sigma_mean", float("nan"))
                            ),
                            "sigma_max_active_fraction": float(
                                control_metrics.get(
                                    "sigma_max_active_fraction", float("nan")
                                )
                            ),
                        }
                    else:
                        row = {
                            "relaxation": relaxation,
                            "complete_objective_value": float("inf"),
                            "discrepancy_objective_value": float("inf"),
                            "running_cost_value": float("inf"),
                            "running_cost_drift_value": float("inf"),
                            "running_cost_volatility_value": float("inf"),
                            "path_rms_change": float("inf"),
                            "control_rms_change": float("inf"),
                            "sigma_mean": float("nan"),
                            "sigma_max_active_fraction": float("nan"),
                        }
                variant_rows.append(row)
                curve_rows.append({"variant": variant.name, **row})
            curve_summaries[variant.name] = summarize_objective_curve(variant_rows)

            for prediction_name, p_values, r_values in (
                ("oracle_target", p_target, r_target),
                (
                    "fitted_candidate",
                    candidate_moments.expected_adjoint_next,
                    candidate_moments.expected_adjoint_noise_next,
                ),
            ):
                with torch.no_grad():
                    controls = _controls_from_moments(
                        control=control,
                        paths=frozen_paths,
                        p=p_values,
                        r=r_values,
                        variant=variant,
                        parameters=model.parameters,
                    )
                    local_metrics = _local_hamiltonian_metrics(
                        control=control,
                        paths=frozen_paths,
                        controls=controls,
                        old_controls=old_rollout.controls,
                        p_target=p_target,
                        r_target=r_target,
                    )
                local_rows.append(
                    {
                        "variant": variant.name,
                        "prediction": prediction_name,
                        "hamiltonian_change_from_old": float(
                            local_metrics["hamiltonian_value"]
                            - baseline_hamiltonian
                        ),
                        **local_metrics,
                    }
                )
    finally:
        model.control_map = original_control_map
        _set_parameter_interpolation(
            parameters,
            old_parameter_values,
            candidate_parameter_values,
            1.0,
        )

    moment_surface_rows: list[dict[str, object]] = []
    original_network = model.network
    model.control_map = control
    try:
        for p_relaxation in sorted(float(value) for value in moment_relaxations):
            for r_relaxation in sorted(float(value) for value in moment_relaxations):
                model.network = _BlendedAdjointMomentNetwork(
                    old_network=old_network,
                    candidate_network=candidate_network,
                    p_relaxation=p_relaxation,
                    r_relaxation=r_relaxation,
                )
                with torch.no_grad():
                    rollout = model.rollout(x0=x0, noise=outer_noise)
                    finite = bool(
                        torch.isfinite(rollout.paths).all().item()
                        and torch.isfinite(rollout.controls).all().item()
                    )
                    if finite:
                        _, objective_metrics = model._complete_objective_metrics(
                            generated_path=rollout.paths,
                            controls=rollout.controls,
                            target_path=target_batch,
                            training=training,
                        )
                        impact = _control_impact_metrics(
                            control=control,
                            paths=rollout.paths,
                            controls=rollout.controls,
                            old_controls=old_rollout.controls,
                            r=rollout.expected_adjoint_noise_next,
                        )
                        row: dict[str, object] = {
                            "p_relaxation": p_relaxation,
                            "r_relaxation": r_relaxation,
                            "finite": True,
                            "complete_objective_value": float(
                                objective_metrics["complete_objective_value"]
                            ),
                            "discrepancy_objective_value": float(
                                objective_metrics["discrepancy_objective_value"]
                            ),
                            "running_cost_value": float(
                                objective_metrics["running_cost_value"]
                            ),
                            "running_cost_drift_value": float(
                                objective_metrics.get(
                                    "running_cost_drift_value", float("nan")
                                )
                            ),
                            "running_cost_volatility_value": float(
                                objective_metrics.get(
                                    "running_cost_volatility_value", float("nan")
                                )
                            ),
                            "path_rms_change": float(
                                torch.sqrt(
                                    torch.mean(
                                        (rollout.paths - old_rollout.paths).pow(2)
                                    )
                                ).item()
                            ),
                            **impact,
                        }
                    else:
                        row = {
                            "p_relaxation": p_relaxation,
                            "r_relaxation": r_relaxation,
                            "finite": False,
                            "complete_objective_value": float("inf"),
                            "discrepancy_objective_value": float("inf"),
                            "running_cost_value": float("inf"),
                            "running_cost_drift_value": float("inf"),
                            "running_cost_volatility_value": float("inf"),
                            "path_rms_change": float("inf"),
                        }
                moment_surface_rows.append(row)
    finally:
        model.network = original_network
        model.control_map = original_control_map

    finite_surface_rows = [
        row
        for row in moment_surface_rows
        if bool(row["finite"])
        and math.isfinite(float(row["complete_objective_value"]))
    ]
    best_surface_row = min(
        finite_surface_rows,
        key=lambda row: float(row["complete_objective_value"]),
    )

    projected_teacher_surface: list[dict[str, object]] = []
    identity_variant = AdjointControlVariant(
        name="correct",
        p_scale=1.0,
        r_scale=1.0,
        interpretation="implemented maximum principle",
    )
    for p_relaxation in sorted(float(value) for value in moment_relaxations):
        for r_relaxation in sorted(float(value) for value in moment_relaxations):
            mixed_p = (
                old_moments.expected_adjoint_next
                + p_relaxation
                * (p_target - old_moments.expected_adjoint_next)
            )
            mixed_r = (
                old_moments.expected_adjoint_noise_next
                + r_relaxation
                * (r_target - old_moments.expected_adjoint_noise_next)
            )
            with torch.no_grad():
                mixed_controls = _controls_from_moments(
                    control=control,
                    paths=frozen_paths,
                    p=mixed_p,
                    r=mixed_r,
                    variant=identity_variant,
                    parameters=model.parameters,
                )
                local_metrics = _local_hamiltonian_metrics(
                    control=control,
                    paths=frozen_paths,
                    controls=mixed_controls,
                    old_controls=old_rollout.controls,
                    p_target=p_target,
                    r_target=r_target,
                )
                impact_metrics = _control_impact_metrics(
                    control=control,
                    paths=frozen_paths,
                    controls=mixed_controls,
                    old_controls=old_rollout.controls,
                    r=mixed_r,
                )
            projected_teacher_surface.append(
                {
                    "p_relaxation": p_relaxation,
                    "r_relaxation": r_relaxation,
                    "hamiltonian_change_from_old": float(
                        local_metrics["hamiltonian_value"]
                        - baseline_hamiltonian
                    ),
                    **local_metrics,
                    **impact_metrics,
                }
            )

    report: dict[str, object] = {
        "protocol": {
            "common_initial_states_noise_and_target_batch": True,
            "candidate_fits_one_frozen_complete_mp_backward_problem": True,
            "neural_loss_unchanged": True,
            "running_cost_and_discrepancy_unchanged": True,
            "noise_is_standard_normal": True,
            "implemented_r_hamiltonian_coefficient": "R/sqrt(dt)",
        },
        "config": {
            "training": asdict(training),
            "nested": asdict(nested_without_line_search),
            "relaxations": [float(value) for value in sorted(relaxations)],
            "moment_relaxations": [
                float(value) for value in sorted(moment_relaxations)
            ],
            "evaluation_batch_size": evaluation_count,
            "variants": [asdict(variant) for variant in variants_used],
            "dt": float(model.grid.dt),
            "sqrt_dt": math.sqrt(float(model.grid.dt)),
        },
        "fit_final": dict(fit_result.final_metrics),
        "target_construction": {
            **{str(name): float(value) for name, value in target_metadata.items()},
            **{
                f"p_{name}": float(value)
                for name, value in p_projection_metadata.items()
            },
            **{
                f"r_{name}": float(value)
                for name, value in r_projection_metadata.items()
            },
        },
        "candidate_moment_diagnostics": {
            **model._moment_diagnostics(
                candidate_moments.expected_adjoint_next,
                p_target,
                prefix="p",
            ),
            **model._moment_diagnostics(
                candidate_moments.expected_adjoint_noise_next,
                r_target,
                prefix="r",
            ),
        },
        "discrete_chain_rule": chain_rule_metrics,
        "objective_curves": curve_rows,
        "curve_summaries": curve_summaries,
        "local_hamiltonian": {
            "old_hamiltonian_value": float(baseline_hamiltonian),
            "rows": local_rows,
        },
        "closed_loop_moment_surface": {
            "interpretation": (
                "independent convex relaxation of old-to-candidate normalized "
                "P/R outputs at every recursive rollout step"
            ),
            "best_row": best_surface_row,
            "rows": moment_surface_rows,
        },
        "frozen_projected_teacher_surface": {
            "interpretation": (
                "direct old-to-projected-conditional-target P/R relaxation on "
                "the frozen old-prefix bank; no recursive rollout"
            ),
            "rows": projected_teacher_surface,
        },
    }
    artifacts: dict[str, object] = {
        "old_checkpoint": old_checkpoint,
        "candidate_checkpoint": candidate_checkpoint,
        "old_paths": old_rollout.paths.detach().cpu(),
        "old_controls": old_rollout.controls.detach().cpu(),
        "noise": outer_noise.detach().cpu(),
        "p_target": p_target.detach().cpu(),
        "r_target": r_target.detach().cpu(),
    }
    return report, artifacts
