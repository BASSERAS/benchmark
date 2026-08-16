from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig, _require_int
from deep_mkv_gen_path_dt.controls import HamiltonianControlInputs
from deep_mkv_gen_path_dt.discrepancies import CompositePathFunctionalDiscrepancy
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals

if TYPE_CHECKING:
    from deep_mkv_gen_path_dt.model import DiscreteMPModel


@dataclass(frozen=True)
class ConditionalSignalDiagnosticConfig:
    num_prefixes: int = 64
    num_branches: int = 8
    population_batch_size: int | None = None
    target_batch_size: int = 256
    step_indices: tuple[int, ...] | None = None
    discrepancy_block_names: tuple[str, ...] = ()
    antithetic: bool = False
    seed: int = 1234

    def __post_init__(self) -> None:
        num_prefixes = _require_int("num_prefixes", self.num_prefixes)
        num_branches = _require_int("num_branches", self.num_branches)
        target_batch_size = _require_int("target_batch_size", self.target_batch_size)
        population_batch_size = (
            None
            if self.population_batch_size is None
            else _require_int(
                "population_batch_size", self.population_batch_size
            )
        )
        seed = _require_int("seed", self.seed)
        if num_prefixes < 2:
            raise ValueError("num_prefixes must be >= 2")
        if num_branches < 2:
            raise ValueError("num_branches must be >= 2")
        if not isinstance(self.antithetic, bool):
            raise ValueError("antithetic must be a bool")
        if bool(self.antithetic) and num_branches % 2 != 0:
            raise ValueError("antithetic diagnostics require an even num_branches")
        if target_batch_size < 1:
            raise ValueError("target_batch_size must be >= 1")
        if population_batch_size is not None and population_batch_size < 2:
            raise ValueError("population_batch_size must be >= 2")
        step_indices = self.step_indices
        if step_indices is not None:
            step_indices = tuple(_require_int("step_index", value) for value in step_indices)
            if len(step_indices) == 0:
                raise ValueError("step_indices must be non-empty when provided")
            if len(set(step_indices)) != len(step_indices):
                raise ValueError("step_indices must not contain duplicates")
        discrepancy_block_names = tuple(self.discrepancy_block_names)
        if any(not isinstance(value, str) or not value.strip() for value in discrepancy_block_names):
            raise ValueError("discrepancy_block_names must contain non-empty strings")
        if len(set(discrepancy_block_names)) != len(discrepancy_block_names):
            raise ValueError("discrepancy_block_names must not contain duplicates")
        object.__setattr__(self, "num_prefixes", num_prefixes)
        object.__setattr__(self, "num_branches", num_branches)
        object.__setattr__(self, "population_batch_size", population_batch_size)
        object.__setattr__(self, "target_batch_size", target_batch_size)
        object.__setattr__(self, "step_indices", step_indices)
        object.__setattr__(self, "discrepancy_block_names", discrepancy_block_names)
        object.__setattr__(self, "antithetic", bool(self.antithetic))
        object.__setattr__(self, "seed", seed)


@dataclass(frozen=True)
class ConditionalSignalDiagnosticResult:
    metrics: dict[str, object]
    tensors: dict[str, torch.Tensor]


def _default_step_indices(num_steps: int) -> tuple[int, ...]:
    candidates = (num_steps // 4, num_steps // 2, (3 * num_steps) // 4, num_steps - 1)
    return tuple(dict.fromkeys(max(0, min(num_steps - 1, value)) for value in candidates))


def _prediction_r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    residual = torch.sum((prediction - target).pow(2))
    baseline = target.mean(dim=0, keepdim=True)
    denominator = torch.sum((target - baseline).pow(2))
    eps = torch.finfo(target.dtype).eps
    return float((1.0 - residual / denominator.clamp_min(eps)).item())


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_flat = left.reshape(-1).to(dtype=torch.float64)
    right_flat = right.reshape(-1).to(dtype=torch.float64)
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    denominator = torch.sqrt(torch.sum(left_centered.pow(2)) * torch.sum(right_centered.pow(2)))
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return None
    return float((torch.sum(left_centered * right_centered) / denominator).item())


def _nested_signal_metrics(
    values: torch.Tensor,
    prediction: torch.Tensor | None,
) -> tuple[dict[str, float | None], dict[str, torch.Tensor]]:
    if values.ndim != 3:
        raise ValueError("nested values must have shape (prefixes, branches, outputs)")
    if prediction is not None and (
        prediction.ndim != 2
        or tuple(prediction.shape) != (int(values.shape[0]), int(values.shape[2]))
    ):
        raise ValueError("prediction must have shape (prefixes, outputs)")
    num_branches = int(values.shape[1])
    branch_mean = values.mean(dim=1)
    within_variance_by_output = values.var(dim=1, unbiased=True).mean(dim=0)
    branch_mean_variance_by_output = branch_mean.var(dim=0, unbiased=True)
    signal_variance_by_output = (
        branch_mean_variance_by_output - within_variance_by_output / float(num_branches)
    ).clamp_min(0.0)
    signal_variance = signal_variance_by_output.mean()
    within_variance = within_variance_by_output.mean()
    nested_mean_noise_variance = within_variance / float(num_branches)
    reliability_denominator = signal_variance + within_variance
    eps = torch.finfo(values.dtype).eps
    reliability = signal_variance / reliability_denominator.clamp_min(eps)
    conditional_mean_rms = torch.sqrt(torch.mean(branch_mean.pow(2)))
    sample_rms = torch.sqrt(torch.mean(values.pow(2)))
    signal_rms = torch.sqrt(signal_variance.clamp_min(0.0))
    within_noise_rms = torch.sqrt(within_variance.clamp_min(0.0))
    nested_mean_noise_standard_error = torch.sqrt(nested_mean_noise_variance.clamp_min(0.0))
    if float(signal_variance.item()) <= eps:
        branches_required_for_signal_to_noise_one = None
    else:
        branches_required_for_signal_to_noise_one = float((within_variance / signal_variance).item())

    half_count = num_branches // 2
    first_half = values[:, :half_count, :].mean(dim=1)
    second_half = values[:, half_count : 2 * half_count, :].mean(dim=1)
    split_half_error = torch.sqrt(torch.mean((first_half - second_half).pow(2)))
    split_half_scale = torch.sqrt(torch.mean(0.5 * (first_half.pow(2) + second_half.pow(2))))

    metrics: dict[str, float | None] = {
        "sample_rms": float(sample_rms.item()),
        "nested_conditional_mean_rms": float(conditional_mean_rms.item()),
        "within_prefix_noise_rms": float(within_noise_rms.item()),
        "between_prefix_signal_rms": float(signal_rms.item()),
        "nested_mean_noise_standard_error_rms": float(nested_mean_noise_standard_error.item()),
        "nested_mean_signal_to_noise_ratio": float(
            (signal_rms / nested_mean_noise_standard_error.clamp_min(eps)).item()
        ),
        "estimated_branches_for_signal_to_noise_one": branches_required_for_signal_to_noise_one,
        "estimated_reliability_ratio": float(reliability.item()),
        "split_half_conditional_mean_correlation": _correlation(first_half, second_half),
        "split_half_relative_rmse": float((split_half_error / split_half_scale.clamp_min(eps)).item()),
    }
    tensors = {
        "samples": values.detach().cpu(),
        "branch_mean": branch_mean.detach().cpu(),
        "within_variance_by_output": within_variance_by_output.detach().cpu(),
        "signal_variance_by_output": signal_variance_by_output.detach().cpu(),
    }
    if prediction is not None:
        prediction_rms = torch.sqrt(torch.mean(prediction.pow(2)))
        prediction_error_rms = torch.sqrt(torch.mean((prediction - branch_mean).pow(2)))
        metrics.update(
            {
                "prediction_rms": float(prediction_rms.item()),
                "prediction_r2_vs_nested_mean": _prediction_r2(prediction, branch_mean),
                "prediction_relative_rmse_vs_nested_mean": float(
                    (prediction_error_rms / conditional_mean_rms.clamp_min(eps)).item()
                ),
                "prediction_to_nested_mean_rms_ratio": float(
                    (prediction_rms / conditional_mean_rms.clamp_min(eps)).item()
                ),
            }
        )
        tensors["prediction"] = prediction.detach().cpu()
    return metrics, tensors


def _direction_alignment_metrics(
    component: torch.Tensor,
    full: torch.Tensor,
) -> dict[str, float | None]:
    if tuple(component.shape) != tuple(full.shape):
        raise ValueError("component and full conditional means must have the same shape")
    component_flat = component.reshape(-1).to(dtype=torch.float64)
    full_flat = full.reshape(-1).to(dtype=torch.float64)
    remainder_flat = full_flat - component_flat
    eps = torch.finfo(torch.float64).eps

    def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
        denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        if float(denominator.item()) <= eps:
            return None
        return float((torch.dot(left, right) / denominator).item())

    full_squared_norm = torch.dot(full_flat, full_flat)
    projection = None
    if float(full_squared_norm.item()) > eps:
        projection = float((torch.dot(component_flat, full_flat) / full_squared_norm).item())
    return {
        "component_rms": float(torch.sqrt(torch.mean(component_flat.pow(2))).item()),
        "full_rms": float(torch.sqrt(torch.mean(full_flat.pow(2))).item()),
        "remainder_rms": float(torch.sqrt(torch.mean(remainder_flat.pow(2))).item()),
        "cosine_with_full": cosine(component_flat, full_flat),
        "fraction_of_full_projection": projection,
        "cosine_with_remainder": cosine(component_flat, remainder_flat),
    }


def _cross_fitted_direction_alignment_metrics(
    component_values: torch.Tensor,
    full_values: torch.Tensor,
) -> dict[str, float | None]:
    if tuple(component_values.shape) != tuple(full_values.shape) or component_values.ndim != 3:
        raise ValueError("component and full samples must share shape (prefixes, branches, outputs)")
    half_count = int(component_values.shape[1]) // 2
    if half_count < 1:
        raise ValueError("cross-fitted alignment requires at least two branches")
    component_first = component_values[:, :half_count, :].mean(dim=1).to(dtype=torch.float64)
    component_second = component_values[:, half_count : 2 * half_count, :].mean(dim=1).to(
        dtype=torch.float64
    )
    full_first = full_values[:, :half_count, :].mean(dim=1).to(dtype=torch.float64)
    full_second = full_values[:, half_count : 2 * half_count, :].mean(dim=1).to(dtype=torch.float64)
    remainder_first = full_first - component_first
    remainder_second = full_second - component_second
    eps = torch.finfo(torch.float64).eps

    def cross_second_moment(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.mean(left * right)

    def symmetric_cross_inner(
        left_first: torch.Tensor,
        left_second: torch.Tensor,
        right_first: torch.Tensor,
        right_second: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (
            cross_second_moment(left_first, right_second)
            + cross_second_moment(left_second, right_first)
        )

    component_moment = cross_second_moment(component_first, component_second)
    full_moment = cross_second_moment(full_first, full_second)
    remainder_moment = cross_second_moment(remainder_first, remainder_second)
    component_full_inner = symmetric_cross_inner(
        component_first,
        component_second,
        full_first,
        full_second,
    )
    component_remainder_inner = symmetric_cross_inner(
        component_first,
        component_second,
        remainder_first,
        remainder_second,
    )

    def cross_rms(moment: torch.Tensor) -> float:
        return float(torch.sqrt(moment.clamp_min(0.0)).item())

    def cosine(inner: torch.Tensor, left_moment: torch.Tensor, right_moment: torch.Tensor) -> float | None:
        if float(left_moment.item()) <= eps or float(right_moment.item()) <= eps:
            return None
        denominator = torch.sqrt(left_moment * right_moment)
        value = float((inner / denominator).item())
        if not -1.0 - 1e-6 <= value <= 1.0 + 1e-6:
            return None
        return max(-1.0, min(1.0, value))

    return {
        "component_signal_rms": cross_rms(component_moment),
        "full_signal_rms": cross_rms(full_moment),
        "remainder_signal_rms": cross_rms(remainder_moment),
        "cosine_with_full": cosine(component_full_inner, component_moment, full_moment),
        "fraction_of_full_projection": (
            float((component_full_inner / full_moment).item())
            if float(full_moment.item()) > eps
            else None
        ),
        "cosine_with_remainder": cosine(
            component_remainder_inner,
            component_moment,
            remainder_moment,
        ),
    }


def _control_directional_effects(
    *,
    model: DiscreteMPModel,
    step_index: int,
    x_prefix: torch.Tensor,
    baseline_p: torch.Tensor,
    baseline_r: torch.Tensor,
    p_direction: torch.Tensor,
    r_direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(baseline_p.shape) != tuple(p_direction.shape):
        raise ValueError("P direction must have the same shape as the baseline P")
    if tuple(baseline_r.shape) != tuple(r_direction.shape):
        raise ValueError("R direction must have the same shape as the baseline R")

    def control(p_value: torch.Tensor, r_value: torch.Tensor) -> torch.Tensor:
        return model.control_map(
            HamiltonianControlInputs(
                step_index=int(step_index),
                x_prefix=x_prefix,
                expected_adjoint_next=p_value,
                expected_adjoint_noise_next=r_value,
            )
        )

    p0 = baseline_p.detach()
    r0 = baseline_r.detach()
    zero_p = torch.zeros_like(p_direction)
    zero_r = torch.zeros_like(r_direction)
    _, p_effect = torch.autograd.functional.jvp(
        control,
        (p0, r0),
        (p_direction.detach(), zero_r),
        create_graph=False,
        strict=False,
    )
    _, r_effect = torch.autograd.functional.jvp(
        control,
        (p0, r0),
        (zero_p, r_direction.detach()),
        create_graph=False,
        strict=False,
    )
    return p_effect, r_effect


def _control_effect_metrics(
    *,
    model: DiscreteMPModel,
    step_index: int,
    x_prefix: torch.Tensor,
    baseline_p: torch.Tensor,
    baseline_r: torch.Tensor,
    p_direction: torch.Tensor,
    r_direction: torch.Tensor,
) -> dict[str, float]:
    p_effect, r_effect = _control_directional_effects(
        model=model,
        step_index=step_index,
        x_prefix=x_prefix,
        baseline_p=baseline_p,
        baseline_r=baseline_r,
        p_direction=p_direction,
        r_direction=r_direction,
    )
    combined_effect = p_effect + r_effect

    def rms(value: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean(value.detach().pow(2))).item())

    metrics = {
        "p_direction_rms": rms(p_direction),
        "r_direction_rms": rms(r_direction),
        "p_control_effect_rms": rms(p_effect),
        "r_control_effect_rms": rms(r_effect),
        "combined_control_effect_rms": rms(combined_effect),
    }
    state_dim = int(model.architecture.state_dim)
    if int(p_effect.shape[-1]) == 2 * state_dim:
        p_to_drift_effect_rms = rms(p_effect[..., :state_dim])
        r_to_volatility_effect_rms = rms(r_effect[..., state_dim:])
        p_direction_rms = rms(p_direction)
        r_direction_rms = rms(r_direction)
        eps = torch.finfo(p_direction.dtype).eps
        metrics.update(
            {
                "p_to_drift_effect_rms": p_to_drift_effect_rms,
                "p_to_volatility_effect_rms": rms(p_effect[..., state_dim:]),
                "r_to_drift_effect_rms": rms(r_effect[..., :state_dim]),
                "r_to_volatility_effect_rms": r_to_volatility_effect_rms,
                "combined_drift_effect_rms": rms(combined_effect[..., :state_dim]),
                "combined_volatility_effect_rms": rms(combined_effect[..., state_dim:]),
                "p_to_drift_local_gain": p_to_drift_effect_rms / max(p_direction_rms, eps),
                "r_to_volatility_local_gain": r_to_volatility_effect_rms
                / max(r_direction_rms, eps),
                "r_volatility_to_p_drift_effect_ratio": r_to_volatility_effect_rms
                / max(p_to_drift_effect_rms, eps),
            }
        )
    return metrics


def _cross_fitted_control_effect_metrics(
    *,
    model: DiscreteMPModel,
    step_index: int,
    x_prefix: torch.Tensor,
    baseline_p: torch.Tensor,
    baseline_r: torch.Tensor,
    p_values: torch.Tensor,
    r_values: torch.Tensor,
) -> dict[str, float | None]:
    if p_values.ndim != 3 or r_values.ndim != 3:
        raise ValueError("cross-fitted control directions must have shape (prefixes, branches, outputs)")
    half_count = min(int(p_values.shape[1]), int(r_values.shape[1])) // 2
    if half_count < 1:
        raise ValueError("cross-fitted control effects require at least two branches")
    p_first = p_values[:, :half_count, :].mean(dim=1)
    p_second = p_values[:, half_count : 2 * half_count, :].mean(dim=1)
    r_first = r_values[:, :half_count, :].mean(dim=1)
    r_second = r_values[:, half_count : 2 * half_count, :].mean(dim=1)
    p_effect_first, r_effect_first = _control_directional_effects(
        model=model,
        step_index=step_index,
        x_prefix=x_prefix,
        baseline_p=baseline_p,
        baseline_r=baseline_r,
        p_direction=p_first,
        r_direction=r_first,
    )
    p_effect_second, r_effect_second = _control_directional_effects(
        model=model,
        step_index=step_index,
        x_prefix=x_prefix,
        baseline_p=baseline_p,
        baseline_r=baseline_r,
        p_direction=p_second,
        r_direction=r_second,
    )

    def cross_rms(left: torch.Tensor, right: torch.Tensor) -> float:
        cross_second_moment = torch.mean(left.detach() * right.detach()).clamp_min(0.0)
        return float(torch.sqrt(cross_second_moment).item())

    metrics = {
        "p_control_effect_signal_rms": cross_rms(p_effect_first, p_effect_second),
        "r_control_effect_signal_rms": cross_rms(r_effect_first, r_effect_second),
    }
    state_dim = int(model.architecture.state_dim)
    if int(p_effect_first.shape[-1]) == 2 * state_dim:
        p_drift = cross_rms(
            p_effect_first[..., :state_dim],
            p_effect_second[..., :state_dim],
        )
        r_volatility = cross_rms(
            r_effect_first[..., state_dim:],
            r_effect_second[..., state_dim:],
        )
        eps = torch.finfo(p_effect_first.dtype).eps
        metrics.update(
            {
                "p_to_drift_effect_signal_rms": p_drift,
                "p_to_volatility_effect_signal_rms": cross_rms(
                    p_effect_first[..., state_dim:],
                    p_effect_second[..., state_dim:],
                ),
                "r_to_drift_effect_signal_rms": cross_rms(
                    r_effect_first[..., :state_dim],
                    r_effect_second[..., :state_dim],
                ),
                "r_to_volatility_effect_signal_rms": r_volatility,
                "r_volatility_to_p_drift_effect_signal_ratio": (
                    r_volatility / p_drift if p_drift > eps else None
                ),
            }
        )
    return metrics


def diagnose_conditional_adjoint_signal(
    *,
    model: DiscreteMPModel,
    target_paths: torch.Tensor,
    config: ConditionalSignalDiagnosticConfig | None = None,
    training: DiscreteMPTrainingConfig | None = None,
) -> ConditionalSignalDiagnosticResult:
    """Estimate conditional-adjoint signal by branching independent futures from fixed prefixes."""
    diagnostic = config or ConditionalSignalDiagnosticConfig()
    training_config = training or model.training
    target_count, _ = model._validate_target_paths(target_paths)
    target_paths_device = target_paths.to(device=model.device, dtype=model.dtype)
    num_steps = int(model.grid.num_steps)
    step_indices = diagnostic.step_indices or _default_step_indices(num_steps)
    if any(not 0 <= int(step) < num_steps for step in step_indices):
        raise ValueError("every diagnostic step_index must lie in [0, num_steps)")

    block_discrepancies: dict[str, CompositePathFunctionalDiscrepancy] = {}
    if len(diagnostic.discrepancy_block_names) > 0:
        if not isinstance(model.discrepancy, CompositePathFunctionalDiscrepancy):
            raise ValueError("discrepancy_block_names require a CompositePathFunctionalDiscrepancy")
        blocks_by_name = {block.name: block for block in tuple(model.discrepancy.blocks)}
        if len(blocks_by_name) != len(tuple(model.discrepancy.blocks)):
            raise ValueError("composite discrepancy block names must be unique for block diagnostics")
        missing = [name for name in diagnostic.discrepancy_block_names if name not in blocks_by_name]
        if len(missing) > 0:
            raise ValueError(f"unknown discrepancy block names: {missing}")
        block_discrepancies = {
            name: CompositePathFunctionalDiscrepancy(blocks=(blocks_by_name[name],))
            for name in diagnostic.discrepancy_block_names
        }

    cpu_generator = torch.Generator(device="cpu").manual_seed(int(diagnostic.seed))
    source_indices = model._sample_indices(
        size=target_count,
        count=int(diagnostic.num_prefixes),
        generator=cpu_generator,
    )
    target_indices = model._sample_indices(
        size=target_count,
        count=int(diagnostic.target_batch_size),
        generator=cpu_generator,
    )
    x0 = target_paths_device.index_select(0, source_indices.to(model.device))[:, 0, :]
    target_batch = target_paths_device.index_select(0, target_indices.to(model.device))
    base_noise = sample_standard_normals(
        grid=model.grid,
        batch_size=int(diagnostic.num_prefixes),
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=derive_stream_seed(base_seed=int(diagnostic.seed), stream_offset=1),
    )
    with torch.no_grad():
        base_paths = model.rollout(x0=x0, noise=base_noise).paths.detach()
    population_batch_size = (
        int(diagnostic.num_prefixes)
        if diagnostic.population_batch_size is None
        else int(diagnostic.population_batch_size)
    )
    if population_batch_size == int(diagnostic.num_prefixes):
        population_paths = base_paths
    else:
        population_indices = model._sample_indices(
            size=target_count,
            count=population_batch_size,
            generator=cpu_generator,
        )
        population_x0 = target_paths_device.index_select(
            0, population_indices.to(model.device)
        )[:, 0, :]
        population_noise = sample_standard_normals(
            grid=model.grid,
            batch_size=population_batch_size,
            noise_dim=int(model.architecture.noise_dim),
            device=model.device,
            dtype=model.dtype,
            seed=derive_stream_seed(
                base_seed=int(diagnostic.seed), stream_offset=2
            ),
        )
        with torch.no_grad():
            population_paths = model.rollout(
                x0=population_x0,
                noise=population_noise,
            ).paths.detach()

    steps_metrics: dict[str, object] = {}
    tensors: dict[str, torch.Tensor] = {}
    reliability_values: dict[str, list[float]] = {
        "adjoint": [],
        "noise_adjoint_raw": [],
        "noise_adjoint_controlled": [],
    }
    prediction_r2_values: dict[str, list[float]] = {
        "adjoint": [],
        "noise_adjoint_raw": [],
        "noise_adjoint_controlled": [],
    }
    variance_reduction_values: list[float] = []
    block_summary_values: dict[str, dict[str, list[float]]] = {
        name: {
            "adjoint_reliability": [],
            "noise_adjoint_reliability": [],
            "adjoint_cosine_with_full": [],
            "noise_adjoint_cosine_with_full": [],
            "adjoint_cosine_with_remainder": [],
            "noise_adjoint_cosine_with_remainder": [],
            "cross_fitted_adjoint_cosine_with_full": [],
            "cross_fitted_noise_adjoint_cosine_with_full": [],
            "cross_fitted_adjoint_cosine_with_remainder": [],
            "cross_fitted_noise_adjoint_cosine_with_remainder": [],
            "p_to_drift_effect_rms": [],
            "r_to_volatility_effect_rms": [],
            "r_volatility_to_p_drift_effect_ratio": [],
            "cross_fitted_p_to_drift_effect_signal_rms": [],
            "cross_fitted_r_to_volatility_effect_signal_rms": [],
            "cross_fitted_r_volatility_to_p_drift_effect_signal_ratio": [],
        }
        for name in block_discrepancies
    }

    prefix_count = int(diagnostic.num_prefixes)
    branch_count = int(diagnostic.num_branches)
    for step_number, step_index in enumerate(step_indices):
        independent_branch_count = (
            branch_count // 2 if bool(diagnostic.antithetic) else branch_count
        )
        independent_noise = sample_standard_normals(
            grid=model.grid,
            batch_size=prefix_count * independent_branch_count,
            noise_dim=int(model.architecture.noise_dim),
            device=model.device,
            dtype=model.dtype,
            seed=derive_stream_seed(base_seed=int(diagnostic.seed), stream_offset=100 + step_number),
        ).reshape(
            prefix_count,
            independent_branch_count,
            num_steps,
            int(model.architecture.noise_dim),
        )
        branch_noise = (
            torch.cat((independent_noise, -independent_noise), dim=1)
            if bool(diagnostic.antithetic)
            else independent_noise
        )
        if int(step_index) > 0:
            branch_noise[:, :, : int(step_index), :] = base_noise[:, None, : int(step_index), :]
        flat_noise = branch_noise.reshape(
            prefix_count * branch_count,
            num_steps,
            int(model.architecture.noise_dim),
        )
        flat_x0 = x0[:, None, :].expand(prefix_count, branch_count, int(x0.shape[-1])).reshape(
            prefix_count * branch_count,
            int(x0.shape[-1]),
        )
        with torch.no_grad():
            rollout = model.rollout(x0=flat_x0, noise=flat_noise)

        paths_grouped = rollout.paths.reshape(
            prefix_count,
            branch_count,
            num_steps + 1,
            int(model.architecture.state_dim),
        )
        controls_grouped = rollout.controls.reshape(
            prefix_count,
            branch_count,
            num_steps,
            int(rollout.controls.shape[-1]),
        )
        prefix_reference = paths_grouped[:, :1, : int(step_index) + 1, :]
        prefix_max_abs_difference = float(
            (paths_grouped[:, :, : int(step_index) + 1, :] - prefix_reference).abs().max().item()
        )
        tensors[f"step_{int(step_index)}_prefix_paths"] = (
            prefix_reference[:, 0, :, :].detach().cpu()
        )

        flat_controls = controls_grouped.reshape(
            prefix_count * branch_count,
            num_steps,
            int(rollout.controls.shape[-1]),
        )
        flat_pathwise_targets, target_metrics = (
            model._frozen_law_path_functional_adjoint_targets(
                query_path=rollout.paths,
                population_path=population_paths,
                controls=flat_controls,
                target_path=target_batch,
                training=training_config,
            )
        )
        pathwise_targets = flat_pathwise_targets.reshape(
            prefix_count,
            branch_count,
            num_steps,
            int(model.architecture.state_dim),
        )
        block_pathwise_targets: dict[str, torch.Tensor] = {}
        for block_name, block_discrepancy in block_discrepancies.items():
            block_target, _ = model._frozen_law_path_functional_adjoint_targets(
                query_path=rollout.paths,
                population_path=population_paths,
                controls=flat_controls,
                target_path=target_batch,
                training=training_config,
                discrepancy=block_discrepancy,
                include_running_cost=False,
            )
            block_pathwise_targets[block_name] = block_target.reshape(
                prefix_count,
                branch_count,
                num_steps,
                int(model.architecture.state_dim),
            )
        flat_pathwise_targets = pathwise_targets.reshape(
            prefix_count * branch_count,
            num_steps,
            int(model.architecture.state_dim),
        )
        noise_output_dim = int(rollout.expected_adjoint_noise_next.shape[-1])
        raw_noise_targets = model._adjoint_noise_target(
            pathwise_adjoint_target=flat_pathwise_targets,
            moments_noise_target_dim=noise_output_dim,
            noise=flat_noise,
        )
        control_variate = model._noise_target_control_variate(
            training=training_config,
            raw_adjoint_prediction=rollout.expected_adjoint_next,
        )
        if control_variate is None:
            controlled_noise_targets = raw_noise_targets
        else:
            controlled_noise_targets = model._adjoint_noise_target(
                pathwise_adjoint_target=flat_pathwise_targets,
                moments_noise_target_dim=noise_output_dim,
                noise=flat_noise,
                control_variate=control_variate,
            )

        adjoint_values = pathwise_targets[:, :, int(step_index), :]
        raw_noise_values = raw_noise_targets.reshape(
            prefix_count,
            branch_count,
            num_steps,
            noise_output_dim,
        )[:, :, int(step_index), :]
        controlled_noise_values = controlled_noise_targets.reshape(
            prefix_count,
            branch_count,
            num_steps,
            noise_output_dim,
        )[:, :, int(step_index), :]

        def estimator_values(values: torch.Tensor) -> torch.Tensor:
            if not bool(diagnostic.antithetic):
                return values
            half = int(values.shape[1]) // 2
            return 0.5 * (values[:, :half, :] + values[:, half:, :])

        adjoint_values = estimator_values(adjoint_values)
        raw_noise_values = estimator_values(raw_noise_values)
        controlled_noise_values = estimator_values(controlled_noise_values)
        adjoint_prediction = rollout.expected_adjoint_next.reshape(
            prefix_count,
            branch_count,
            num_steps,
            int(model.architecture.state_dim),
        )[:, :, int(step_index), :].mean(dim=1)
        noise_prediction = rollout.expected_adjoint_noise_next.reshape(
            prefix_count,
            branch_count,
            num_steps,
            noise_output_dim,
        )[:, :, int(step_index), :].mean(dim=1)

        step_metrics: dict[str, object] = {
            "step_index": int(step_index),
            "prefix_length": int(step_index) + 1,
            "prefix_max_abs_difference": prefix_max_abs_difference,
            "mean_target_rms": float(target_metrics["target_rms"]),
            "frozen_law_lions_derivative": float(
                target_metrics["frozen_law_lions_derivative"]
            ),
        }
        for name, values, prediction in (
            ("adjoint", adjoint_values, adjoint_prediction),
            ("noise_adjoint_raw", raw_noise_values, noise_prediction),
            ("noise_adjoint_controlled", controlled_noise_values, noise_prediction),
        ):
            nested_metrics, nested_tensors = _nested_signal_metrics(values, prediction)
            step_metrics[name] = nested_metrics
            reliability_values[name].append(float(nested_metrics["estimated_reliability_ratio"]))
            prediction_r2_values[name].append(float(nested_metrics["prediction_r2_vs_nested_mean"]))
            for tensor_name, tensor in nested_tensors.items():
                tensors[f"step_{int(step_index)}_{name}_{tensor_name}"] = tensor

        full_adjoint_mean = adjoint_values.mean(dim=1)
        full_noise_mean = raw_noise_values.mean(dim=1)
        step_metrics["full_control_effect"] = _control_effect_metrics(
            model=model,
            step_index=int(step_index),
            x_prefix=prefix_reference[:, 0, :, :],
            baseline_p=adjoint_prediction,
            baseline_r=noise_prediction,
            p_direction=full_adjoint_mean,
            r_direction=full_noise_mean,
        )
        step_metrics["full_cross_fitted_control_effect"] = _cross_fitted_control_effect_metrics(
            model=model,
            step_index=int(step_index),
            x_prefix=prefix_reference[:, 0, :, :],
            baseline_p=adjoint_prediction,
            baseline_r=noise_prediction,
            p_values=adjoint_values,
            r_values=raw_noise_values,
        )

        block_step_metrics: dict[str, object] = {}
        for block_name, block_targets in block_pathwise_targets.items():
            flat_block_targets = block_targets.reshape(
                prefix_count * branch_count,
                num_steps,
                int(model.architecture.state_dim),
            )
            block_raw_noise_targets = model._adjoint_noise_target(
                pathwise_adjoint_target=flat_block_targets,
                moments_noise_target_dim=noise_output_dim,
                noise=flat_noise,
            )
            if control_variate is None:
                block_controlled_noise_targets = block_raw_noise_targets
            else:
                block_controlled_noise_targets = model._adjoint_noise_target(
                    pathwise_adjoint_target=flat_block_targets,
                    moments_noise_target_dim=noise_output_dim,
                    noise=flat_noise,
                    control_variate=control_variate,
                )

            block_adjoint_values = block_targets[:, :, int(step_index), :]
            block_raw_noise_values = block_raw_noise_targets.reshape(
                prefix_count,
                branch_count,
                num_steps,
                noise_output_dim,
            )[:, :, int(step_index), :]
            block_controlled_noise_values = block_controlled_noise_targets.reshape(
                prefix_count,
                branch_count,
                num_steps,
                noise_output_dim,
            )[:, :, int(step_index), :]
            block_adjoint_values = estimator_values(block_adjoint_values)
            block_raw_noise_values = estimator_values(block_raw_noise_values)
            block_controlled_noise_values = estimator_values(
                block_controlled_noise_values
            )

            block_metrics: dict[str, object] = {}
            for signal_name, signal_values in (
                ("adjoint", block_adjoint_values),
                ("noise_adjoint_raw", block_raw_noise_values),
                ("noise_adjoint_controlled", block_controlled_noise_values),
            ):
                nested_metrics, nested_tensors = _nested_signal_metrics(signal_values, None)
                block_metrics[signal_name] = nested_metrics
                for tensor_name, tensor in nested_tensors.items():
                    tensors[
                        f"step_{int(step_index)}_block_{block_name}_{signal_name}_{tensor_name}"
                    ] = tensor

            block_adjoint_mean = block_adjoint_values.mean(dim=1)
            block_noise_mean = block_raw_noise_values.mean(dim=1)
            adjoint_alignment = _direction_alignment_metrics(
                block_adjoint_mean,
                full_adjoint_mean,
            )
            noise_alignment = _direction_alignment_metrics(
                block_noise_mean,
                full_noise_mean,
            )
            cross_fitted_adjoint_alignment = _cross_fitted_direction_alignment_metrics(
                block_adjoint_values,
                adjoint_values,
            )
            cross_fitted_noise_alignment = _cross_fitted_direction_alignment_metrics(
                block_raw_noise_values,
                raw_noise_values,
            )
            control_effect = _control_effect_metrics(
                model=model,
                step_index=int(step_index),
                x_prefix=prefix_reference[:, 0, :, :],
                baseline_p=adjoint_prediction,
                baseline_r=noise_prediction,
                p_direction=block_adjoint_mean,
                r_direction=block_noise_mean,
            )
            cross_fitted_control_effect = _cross_fitted_control_effect_metrics(
                model=model,
                step_index=int(step_index),
                x_prefix=prefix_reference[:, 0, :, :],
                baseline_p=adjoint_prediction,
                baseline_r=noise_prediction,
                p_values=block_adjoint_values,
                r_values=block_raw_noise_values,
            )
            block_metrics["alignment_with_full"] = {
                "adjoint": adjoint_alignment,
                "noise_adjoint_raw": noise_alignment,
            }
            block_metrics["cross_fitted_alignment_with_full"] = {
                "adjoint": cross_fitted_adjoint_alignment,
                "noise_adjoint_raw": cross_fitted_noise_alignment,
            }
            block_metrics["control_effect"] = control_effect
            block_metrics["cross_fitted_control_effect"] = cross_fitted_control_effect
            block_metrics["control_effect_noise_estimator"] = "raw_nested_branch_mean"
            block_step_metrics[block_name] = block_metrics

            summary_values = block_summary_values[block_name]
            summary_values["adjoint_reliability"].append(
                float(block_metrics["adjoint"]["estimated_reliability_ratio"])
            )
            summary_values["noise_adjoint_reliability"].append(
                float(block_metrics["noise_adjoint_raw"]["estimated_reliability_ratio"])
            )
            for source, destination in (
                (adjoint_alignment["cosine_with_full"], "adjoint_cosine_with_full"),
                (noise_alignment["cosine_with_full"], "noise_adjoint_cosine_with_full"),
                (adjoint_alignment["cosine_with_remainder"], "adjoint_cosine_with_remainder"),
                (noise_alignment["cosine_with_remainder"], "noise_adjoint_cosine_with_remainder"),
                (
                    cross_fitted_adjoint_alignment["cosine_with_full"],
                    "cross_fitted_adjoint_cosine_with_full",
                ),
                (
                    cross_fitted_noise_alignment["cosine_with_full"],
                    "cross_fitted_noise_adjoint_cosine_with_full",
                ),
                (
                    cross_fitted_adjoint_alignment["cosine_with_remainder"],
                    "cross_fitted_adjoint_cosine_with_remainder",
                ),
                (
                    cross_fitted_noise_alignment["cosine_with_remainder"],
                    "cross_fitted_noise_adjoint_cosine_with_remainder",
                ),
            ):
                if source is not None:
                    summary_values[destination].append(float(source))
            if "p_to_drift_effect_rms" in control_effect:
                summary_values["p_to_drift_effect_rms"].append(
                    float(control_effect["p_to_drift_effect_rms"])
                )
                summary_values["r_to_volatility_effect_rms"].append(
                    float(control_effect["r_to_volatility_effect_rms"])
                )
                summary_values["r_volatility_to_p_drift_effect_ratio"].append(
                    float(control_effect["r_volatility_to_p_drift_effect_ratio"])
                )
                summary_values["cross_fitted_p_to_drift_effect_signal_rms"].append(
                    float(cross_fitted_control_effect["p_to_drift_effect_signal_rms"])
                )
                summary_values["cross_fitted_r_to_volatility_effect_signal_rms"].append(
                    float(cross_fitted_control_effect["r_to_volatility_effect_signal_rms"])
                )
                cross_fitted_effect_ratio = cross_fitted_control_effect[
                    "r_volatility_to_p_drift_effect_signal_ratio"
                ]
                if cross_fitted_effect_ratio is not None:
                    summary_values[
                        "cross_fitted_r_volatility_to_p_drift_effect_signal_ratio"
                    ].append(float(cross_fitted_effect_ratio))
        if len(block_step_metrics) > 0:
            step_metrics["discrepancy_blocks"] = block_step_metrics

        raw_second_moment = torch.mean(raw_noise_values.pow(2)).clamp_min(
            torch.finfo(raw_noise_values.dtype).eps
        )
        variance_reduction = float((torch.mean(controlled_noise_values.pow(2)) / raw_second_moment).item())
        raw_nested_mean = raw_noise_values.mean(dim=1)
        controlled_nested_mean = controlled_noise_values.mean(dim=1)
        conditional_mean_difference_rms = torch.sqrt(
            torch.mean((raw_nested_mean - controlled_nested_mean).pow(2))
        )
        conditional_mean_scale = torch.sqrt(torch.mean(raw_nested_mean.pow(2))).clamp_min(
            torch.finfo(raw_nested_mean.dtype).eps
        )
        step_metrics["noise_control_variate"] = {
            "mode": str(training_config.noise_target_control_variate),
            "second_moment_ratio": variance_reduction,
            "conditional_mean_difference_rms": float(conditional_mean_difference_rms.item()),
            "conditional_mean_relative_difference": float(
                (conditional_mean_difference_rms / conditional_mean_scale).item()
            ),
        }
        variance_reduction_values.append(variance_reduction)
        steps_metrics[str(int(step_index))] = step_metrics

    summary: dict[str, object] = {
        "mean_estimated_reliability_ratio": {
            name: float(sum(values) / len(values)) for name, values in reliability_values.items()
        },
        "mean_prediction_r2_vs_nested_mean": {
            name: float(sum(values) / len(values)) for name, values in prediction_r2_values.items()
        },
        "mean_noise_control_variate_second_moment_ratio": float(
            sum(variance_reduction_values) / len(variance_reduction_values)
        ),
    }
    if len(block_summary_values) > 0:
        summary["discrepancy_blocks"] = {
            block_name: {
                f"mean_{metric_name}": (
                    float(sum(values) / len(values)) if len(values) > 0 else None
                )
                for metric_name, values in metric_values.items()
            }
            for block_name, metric_values in block_summary_values.items()
        }
    metrics: dict[str, object] = {
        "config": {
            "num_prefixes": prefix_count,
            "num_branches": branch_count,
            "target_batch_size": int(diagnostic.target_batch_size),
            "step_indices": [int(value) for value in step_indices],
            "discrepancy_block_names": list(diagnostic.discrepancy_block_names),
            "seed": int(diagnostic.seed),
            "noise_target_control_variate": str(training_config.noise_target_control_variate),
            "branch_batches_are_separate_discrepancy_batches": True,
        },
        "steps": steps_metrics,
        "summary": summary,
    }
    return ConditionalSignalDiagnosticResult(metrics=metrics, tensors=tensors)


__all__ = [
    "ConditionalSignalDiagnosticConfig",
    "ConditionalSignalDiagnosticResult",
    "diagnose_conditional_adjoint_signal",
]
