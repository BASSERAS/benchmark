from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Any, Sequence

import torch

from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.discrepancies import (
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals
from path_dt_experiments.metrics import path_returns, stylized_fact_metrics
from path_dt_experiments.volatility_diagnostics import volatility_path_statistics


CONTROL_VARIATE_COMPONENT = "__noise_control_variate__"
PREDICTION_SHRINKAGE_COMPONENT = "__prediction_shrinkage__"
FULL_DIRECTION = "__full__"


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _require_positive_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite float")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return result


@dataclass(frozen=True)
class FrozenResponseAuditConfig:
    direction_batch_size: int = 256
    path_gradient_batch_size: int = 128
    target_batch_size: int = 256
    metric_num_paths: int = 512
    metric_batch_size: int = 512
    parameter_step_fraction: float = 1e-5
    path_return_step_fraction: float = 1e-2
    prefix_length: int | None = None
    prefix_window: int | None = None
    horizon: int | None = None
    lags: tuple[int, ...] = (1, 2, 5, 10, 20)
    swd_projections: int = 32
    seed: int = 1234

    def __post_init__(self) -> None:
        for name in (
            "direction_batch_size",
            "path_gradient_batch_size",
            "target_batch_size",
            "metric_num_paths",
            "metric_batch_size",
            "swd_projections",
        ):
            value = _require_int(name, getattr(self, name))
            if value < 2 and name not in {"metric_batch_size"}:
                raise ValueError(f"{name} must be >= 2")
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "parameter_step_fraction",
            _require_positive_float("parameter_step_fraction", self.parameter_step_fraction),
        )
        object.__setattr__(
            self,
            "path_return_step_fraction",
            _require_positive_float("path_return_step_fraction", self.path_return_step_fraction),
        )
        for name in ("prefix_length", "prefix_window", "horizon"):
            value = getattr(self, name)
            if value is not None:
                value = _require_int(name, value)
                if value < 1:
                    raise ValueError(f"{name} must be >= 1 when provided")
                object.__setattr__(self, name, value)
        lags = tuple(_require_int("lag", value) for value in self.lags)
        if len(lags) == 0 or any(value < 1 for value in lags):
            raise ValueError("lags must contain positive integers")
        if len(set(lags)) != len(lags):
            raise ValueError("lags must not contain duplicates")
        object.__setattr__(self, "lags", lags)
        object.__setattr__(self, "seed", _require_int("seed", self.seed))


@dataclass(frozen=True)
class FrozenResponseAuditResult:
    metrics: dict[str, Any]
    tensors: dict[str, torch.Tensor]


def _isolated_discrepancy(block: WeightedDiscrepancy) -> CompositePathFunctionalDiscrepancy:
    return CompositePathFunctionalDiscrepancy(blocks=(block,))


def _observed_problem(
    generated_paths: torch.Tensor,
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    observed_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, DiscreteTimeGrid]:
    if not observed_only:
        return generated_paths, target_paths, grid
    indices = grid.observed_point_indices(device=generated_paths.device)
    times = grid.time_points(device=generated_paths.device, dtype=torch.float64).index_select(0, indices)
    observed_grid = DiscreteTimeGrid(
        T=float(grid.T),
        num_steps=int(grid.observed_count) - 1,
        point_times=tuple(float(value) for value in times.detach().cpu().tolist()),
        allow_nonuniform=True,
    )
    return (
        generated_paths.index_select(1, indices),
        target_paths.index_select(1, indices.to(target_paths.device)),
        observed_grid,
    )


def _discrepancy_value(
    discrepancy: CompositePathFunctionalDiscrepancy,
    generated_paths: torch.Tensor,
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    observed_only: bool,
) -> float:
    generated_used, target_used, grid_used = _observed_problem(
        generated_paths,
        target_paths.to(device=generated_paths.device, dtype=generated_paths.dtype),
        grid=grid,
        observed_only=observed_only,
    )
    with torch.no_grad():
        result = discrepancy(
            generated_paths=generated_used,
            target_paths=target_used,
            grid=grid_used,
        )
    return float(result.value.detach().item())


def _path_gradient(
    discrepancy: CompositePathFunctionalDiscrepancy,
    generated_paths: torch.Tensor,
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    observed_only: bool,
    lambda_scale: float,
) -> tuple[torch.Tensor, float]:
    generated = generated_paths.detach().clone().requires_grad_(True)
    target = target_paths.to(device=generated.device, dtype=generated.dtype)
    generated_used, target_used, grid_used = _observed_problem(
        generated,
        target,
        grid=grid,
        observed_only=observed_only,
    )
    result = discrepancy(
        generated_paths=generated_used,
        target_paths=target_used,
        grid=grid_used,
    )
    gradient = torch.autograd.grad(float(lambda_scale) * result.value, generated)[0].detach()
    descent = -gradient
    descent[:, 0, :] = 0.0
    return descent, float(result.value.detach().item())


def _flatten_gradients(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    values = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1)
        for gradient, parameter in zip(gradients, parameters)
    ]
    return torch.cat(values)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left64 = left.detach().reshape(-1).to(dtype=torch.float64)
    right64 = right.detach().reshape(-1).to(dtype=torch.float64)
    denominator = torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64)
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return None
    return float((torch.dot(left64, right64) / denominator).item())


def _alignment(component: torch.Tensor, full: torch.Tensor) -> dict[str, float | None]:
    component64 = component.detach().to(dtype=torch.float64)
    full64 = full.detach().to(dtype=torch.float64)
    remainder = full64 - component64
    full_norm_squared = torch.sum(full64.pow(2))
    eps = torch.finfo(torch.float64).eps
    fraction = None
    if float(full_norm_squared.item()) > eps:
        fraction = float((torch.dot(component64, full64) / full_norm_squared).item())
    return {
        "cosine_with_full": _cosine(component64, full64),
        "cosine_with_remainder": _cosine(component64, remainder),
        "fraction_of_full_projection": fraction,
    }


def _direction_report(
    directions: dict[str, torch.Tensor],
    *,
    full_name: str = FULL_DIRECTION,
) -> tuple[dict[str, Any], torch.Tensor, list[str]]:
    order = list(directions)
    matrix = torch.stack([directions[name].detach().cpu() for name in order], dim=0)
    pairwise = [
        [_cosine(directions[left], directions[right]) for right in order]
        for left in order
    ]
    full = directions[full_name]
    component_order = [name for name in order if name != full_name]
    component_matrix = torch.stack(
        [directions[name].detach().reshape(-1).to(dtype=torch.float64) for name in component_order],
        dim=0,
    )
    def spectrum(values: torch.Tensor) -> dict[str, Any]:
        gram = values @ values.transpose(0, 1)
        singular_values = torch.sqrt(torch.linalg.eigvalsh(gram).clamp_min(0.0)).flip(0)
        largest = float(singular_values[0].item()) if int(singular_values.numel()) > 0 else 0.0
        tolerance = largest * 1e-8
        retained = singular_values[singular_values > tolerance]
        numerical_rank = int(retained.numel())
        return {
            "singular_values": [float(value.item()) for value in singular_values],
            "relative_rank_tolerance": 1e-8,
            "numerical_rank": numerical_rank,
            "condition_number_on_numerical_rank": (
                None
                if numerical_rank == 0
                else float((retained[0] / retained[-1]).item())
            ),
            "stable_rank": (
                0.0
                if largest == 0.0
                else float((torch.sum(singular_values.pow(2)) / singular_values[0].pow(2)).item())
            ),
        }

    raw_spectrum = spectrum(component_matrix)
    row_norms = torch.linalg.vector_norm(component_matrix, dim=1, keepdim=True)
    normalized_spectrum = spectrum(
        component_matrix / row_norms.clamp_min(torch.finfo(torch.float64).eps)
    )
    alignments = {
        name: _alignment(direction, full)
        for name, direction in directions.items()
        if name != full_name
    }
    component_norm_sum = sum(
        float(torch.linalg.vector_norm(direction.detach().double()).item())
        for name, direction in directions.items()
        if name != full_name
    )
    full_norm = float(torch.linalg.vector_norm(full.detach().double()).item())
    return (
        {
            "order": order,
            "norms": {
                name: float(torch.linalg.vector_norm(direction.detach().double()).item())
                for name, direction in directions.items()
            },
            "pairwise_cosines": pairwise,
            "alignments_with_full": alignments,
            "component_matrix_spectrum": {
                "component_order": component_order,
                "raw": raw_spectrum,
                "row_normalized": normalized_spectrum,
            },
            "full_to_component_norm_sum_ratio": (
                None if component_norm_sum == 0.0 else full_norm / component_norm_sum
            ),
        },
        matrix,
        order,
    )


def _sample_common_paths(
    model: DiscreteMPModel,
    *,
    x0: torch.Tensor,
    num_paths: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    chunks = []
    for batch_index, start in enumerate(range(0, int(num_paths), int(batch_size))):
        current = min(int(batch_size), int(num_paths) - start)
        x0_batch = x0.to(device=model.device, dtype=model.dtype)
        if int(x0_batch.shape[0]) == 1:
            x0_batch = x0_batch.repeat(current, 1)
        elif int(x0_batch.shape[0]) != current:
            raise ValueError("x0 must have one row or match every sampling batch")
        noise = sample_standard_normals(
            grid=model.grid,
            batch_size=current,
            noise_dim=int(model.architecture.noise_dim),
            device=model.device,
            dtype=model.dtype,
            seed=derive_stream_seed(base_seed=seed, stream_offset=10_000 + batch_index),
        )
        with torch.no_grad():
            chunks.append(model.rollout(x0=x0_batch, noise=noise).paths.detach().cpu())
    return torch.cat(chunks, dim=0)


def _metric_configuration(
    *,
    num_steps: int,
    config: FrozenResponseAuditConfig,
) -> tuple[int, int, int, tuple[int, ...]]:
    prefix_length = int(config.prefix_length) if config.prefix_length is not None else num_steps // 2 + 1
    prefix_window = int(config.prefix_window) if config.prefix_window is not None else max(1, num_steps // 4)
    horizon = int(config.horizon) if config.horizon is not None else max(1, num_steps // 4)
    future_start = prefix_length - 1
    if future_start < prefix_window or future_start + horizon > num_steps:
        raise ValueError("prefix_length, prefix_window, and horizon do not fit the path")
    lags = tuple(value for value in config.lags if value < num_steps)
    if not lags:
        raise ValueError("at least one configured lag must be smaller than num_steps")
    return prefix_length, prefix_window, horizon, lags


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-15:
        return 0.0
    return numerator / denominator


def _iqr(values: torch.Tensor) -> float:
    values64 = values.detach().double().reshape(-1)
    return float((torch.quantile(values64, 0.75) - torch.quantile(values64, 0.25)).item())


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.detach().double().reshape(-1)
    right64 = right.detach().double().reshape(-1)
    left_centered = left64 - left64.mean()
    right_centered = right64 - right64.mean()
    denominator = torch.sqrt(torch.mean(left_centered.pow(2)) * torch.mean(right_centered.pow(2)))
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return 0.0
    return float((torch.mean(left_centered * right_centered) / denominator).item())


def _response_metrics(
    generated_paths: torch.Tensor,
    reference_paths: torch.Tensor,
    *,
    config: FrozenResponseAuditConfig,
) -> dict[str, float]:
    generated = generated_paths.detach().cpu()
    reference = reference_paths.detach().cpu()
    num_steps = int(generated.shape[1]) - 1
    prefix_length, prefix_window, horizon, lags = _metric_configuration(
        num_steps=num_steps,
        config=config,
    )
    generated_stats = volatility_path_statistics(
        generated,
        prefix_length=prefix_length,
        prefix_window=prefix_window,
        horizons=(horizon,),
        lags=lags,
    )
    reference_stats = volatility_path_statistics(
        reference,
        prefix_length=prefix_length,
        prefix_window=prefix_window,
        horizons=(horizon,),
        lags=lags,
    )
    generated_rv = generated_stats.future_realized_volatility[horizon]
    reference_rv = reference_stats.future_realized_volatility[horizon]
    generated_mean = float(generated_rv.mean().item())
    reference_mean = float(reference_rv.mean().item())
    generated_std = float(generated_rv.std(unbiased=False).item())
    reference_std = float(reference_rv.std(unbiased=False).item())
    generated_iqr = _iqr(generated_rv)
    reference_iqr = _iqr(reference_rv)
    generated_correlation = _correlation(generated_stats.prefix_rms_volatility, generated_rv)
    reference_correlation = _correlation(reference_stats.prefix_rms_volatility, reference_rv)
    stylized = stylized_fact_metrics(
        generated,
        reference,
        lags=lags,
        include_swd=True,
        swd_projections=int(config.swd_projections),
    )
    generated_abs_mean = float(path_returns(generated).abs().mean().item())
    reference_abs_mean = float(path_returns(reference).abs().mean().item())
    mean_ratio = _safe_ratio(generated_mean, reference_mean)
    std_ratio = _safe_ratio(generated_std, reference_std)
    iqr_ratio = _safe_ratio(generated_iqr, reference_iqr)
    abs_mean_ratio = _safe_ratio(generated_abs_mean, reference_abs_mean)
    return {
        "future_rv_mean_ratio": mean_ratio,
        "future_rv_mean_abs_error": abs(mean_ratio - 1.0),
        "future_rv_std_ratio": std_ratio,
        "future_rv_std_abs_error": abs(std_ratio - 1.0),
        "future_rv_iqr_ratio": iqr_ratio,
        "future_rv_iqr_abs_error": abs(iqr_ratio - 1.0),
        "prefix_future_correlation": generated_correlation,
        "reference_prefix_future_correlation": reference_correlation,
        "prefix_future_correlation_abs_error": abs(generated_correlation - reference_correlation),
        "return_abs_mean_ratio": abs_mean_ratio,
        "return_abs_mean_abs_error": abs(abs_mean_ratio - 1.0),
        "abs_return_acf_error_rms": float(stylized["abs_return_acf_error_rms"]),
        "excess_kurtosis_error_rms": float(stylized["excess_kurtosis_error_rms"]),
        "return_std_error_rms": float(stylized["return_std_error_rms"]),
        "path_swd": float(stylized["path_swd"]),
    }


def _response_record(
    baseline: dict[str, float],
    plus: dict[str, float],
    minus: dict[str, float],
    *,
    step: float,
) -> dict[str, dict[str, float]]:
    return {
        "baseline": baseline,
        "plus": plus,
        "minus": minus,
        "plus_change": {name: plus[name] - baseline[name] for name in baseline},
        "minus_change": {name: minus[name] - baseline[name] for name in baseline},
        "central_derivative": {
            name: (plus[name] - minus[name]) / (2.0 * float(step))
            for name in baseline
        },
    }


def _parameter_target_directions(
    *,
    model: DiscreteMPModel,
    discrepancy: CompositePathFunctionalDiscrepancy,
    target_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    config: FrozenResponseAuditConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor]:
    target_device = target_paths.to(device=model.device, dtype=model.dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed) + 101)
    source_indices = torch.randint(
        int(target_device.shape[0]),
        (int(config.direction_batch_size),),
        generator=generator,
    )
    target_indices = torch.randint(
        int(target_device.shape[0]),
        (int(config.target_batch_size),),
        generator=generator,
    )
    x0 = target_device.index_select(0, source_indices.to(model.device))[:, 0, :]
    target_batch = target_device.index_select(0, target_indices.to(model.device))
    noise = sample_standard_normals(
        grid=model.grid,
        batch_size=int(config.direction_batch_size),
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=derive_stream_seed(base_seed=int(config.seed), stream_offset=102),
    )
    with torch.no_grad():
        frozen_path = model.rollout(x0=x0, noise=noise).paths.detach()
    normalized_moments = model.network(frozen_path)
    moments = model._denormalize_moments(normalized_moments)
    parameters = tuple(model.network.parameters())
    directions: dict[str, torch.Tensor] = {}
    component_metrics: dict[str, Any] = {}

    def gradient_of(scalar: torch.Tensor) -> torch.Tensor:
        gradients = torch.autograd.grad(
            scalar,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        return _flatten_gradients(gradients, parameters)

    for block in tuple(discrepancy.blocks):
        isolated = _isolated_discrepancy(block)
        pathwise_target, target_metadata = model._path_functional_adjoint_targets(
            generated_path=frozen_path,
            target_path=target_batch,
            training=training,
            discrepancy=isolated,
            include_running_cost=False,
        )
        with torch.no_grad():
            projected_p, _ = model._project_pathwise_target(
                frozen_generated_path=frozen_path,
                pathwise_target=pathwise_target,
                training=training,
            )
            normalized_p = model._normalize_adjoint_target(projected_p, noise_adjoint=False)
            raw_r = model._adjoint_noise_target(
                pathwise_adjoint_target=pathwise_target,
                moments_noise_target_dim=int(normalized_moments.expected_adjoint_noise_next.shape[-1]),
                noise=noise,
            )
            projected_r, _ = model._project_pathwise_target(
                frozen_generated_path=frozen_path,
                pathwise_target=raw_r,
                training=training,
                noise_adjoint=True,
            )
            normalized_r = model._normalize_adjoint_target(projected_r, noise_adjoint=True)
        scalar_p = float(training.adjoint_weight) * torch.mean(
            normalized_moments.expected_adjoint_next * normalized_p
        )
        scalar_r = float(training.adjoint_noise_weight) * torch.mean(
            normalized_moments.expected_adjoint_noise_next * normalized_r
        )
        p_direction = gradient_of(scalar_p)
        r_direction = gradient_of(scalar_r)
        combined = p_direction + r_direction
        name = str(block.name)
        directions[name] = combined
        component_metrics[name] = {
            "kind": "weighted_discrepancy_block",
            "weight": float(block.weight),
            "source_rms": float(target_metadata["source_rms"]),
            "pathwise_p_target_rms": float(target_metadata["target_rms"]),
            "projected_p_target_rms": float(torch.sqrt(torch.mean(projected_p.pow(2))).item()),
            "projected_r_target_rms": float(torch.sqrt(torch.mean(projected_r.pow(2))).item()),
            "p_parameter_direction_norm": float(torch.linalg.vector_norm(p_direction.double()).item()),
            "r_parameter_direction_norm": float(torch.linalg.vector_norm(r_direction.double()).item()),
            "combined_parameter_direction_norm": float(torch.linalg.vector_norm(combined.double()).item()),
            "p_r_parameter_direction_cosine": _cosine(p_direction, r_direction),
        }

    control_variate = model._noise_target_control_variate(
        training=training,
        raw_adjoint_prediction=moments.expected_adjoint_next,
    )
    if control_variate is not None and float(training.adjoint_noise_weight) > 0.0:
        with torch.no_grad():
            zero_p = torch.zeros_like(frozen_path[:, 1:, :])
            raw_cv_r = model._adjoint_noise_target(
                pathwise_adjoint_target=zero_p,
                moments_noise_target_dim=int(normalized_moments.expected_adjoint_noise_next.shape[-1]),
                noise=noise,
                control_variate=control_variate,
            )
            projected_cv_r, _ = model._project_pathwise_target(
                frozen_generated_path=frozen_path,
                pathwise_target=raw_cv_r,
                training=training,
                noise_adjoint=True,
            )
            normalized_cv_r = model._normalize_adjoint_target(projected_cv_r, noise_adjoint=True)
        scalar_cv = float(training.adjoint_noise_weight) * torch.mean(
            normalized_moments.expected_adjoint_noise_next * normalized_cv_r
        )
        cv_direction = gradient_of(scalar_cv)
        directions[CONTROL_VARIATE_COMPONENT] = cv_direction
        component_metrics[CONTROL_VARIATE_COMPONENT] = {
            "kind": "non_discrepancy_noise_control_variate",
            "projected_r_target_rms": float(torch.sqrt(torch.mean(projected_cv_r.pow(2))).item()),
            "combined_parameter_direction_norm": float(torch.linalg.vector_norm(cv_direction.double()).item()),
        }

    shrinkage_scalar = -0.5 * float(training.adjoint_weight) * torch.mean(
        normalized_moments.expected_adjoint_next.pow(2)
    )
    if float(training.adjoint_noise_weight) > 0.0:
        shrinkage_scalar = shrinkage_scalar - 0.5 * float(training.adjoint_noise_weight) * torch.mean(
            normalized_moments.expected_adjoint_noise_next.pow(2)
        )
    shrinkage_direction = gradient_of(shrinkage_scalar)
    directions[PREDICTION_SHRINKAGE_COMPONENT] = shrinkage_direction
    component_metrics[PREDICTION_SHRINKAGE_COMPONENT] = {
        "kind": "shared_prediction_shrinkage_from_adjoint_mse",
        "combined_parameter_direction_norm": float(
            torch.linalg.vector_norm(shrinkage_direction.double()).item()
        ),
    }

    directions[FULL_DIRECTION] = torch.stack(tuple(directions.values()), dim=0).sum(dim=0)
    component_metrics[FULL_DIRECTION] = {
        "kind": "instantaneous_gradient_descent_direction_up_to_common_factor_two",
        "combined_parameter_direction_norm": float(
            torch.linalg.vector_norm(directions[FULL_DIRECTION].double()).item()
        ),
    }
    return directions, component_metrics, frozen_path


def audit_frozen_block_responses(
    *,
    model: DiscreteMPModel,
    discrepancy: CompositePathFunctionalDiscrepancy,
    target_paths: torch.Tensor,
    reference_paths: torch.Tensor,
    x0: torch.Tensor,
    training: DiscreteMPTrainingConfig | None = None,
    config: FrozenResponseAuditConfig | None = None,
) -> FrozenResponseAuditResult:
    """Audit path-space and learned responses of each discrepancy block.

    The neural direction is the additive target-forcing contribution
    ``J_theta^T(P_b, R_b)``.  It intentionally excludes the shared prediction
    shrinkage term in the adjoint MSE, which cannot be attributed to an
    individual discrepancy block.  Symmetric parameter perturbations use the
    same Gaussian noise, so their central differences are not contaminated by
    Monte Carlo resampling noise.
    """

    if not isinstance(discrepancy, CompositePathFunctionalDiscrepancy):
        raise ValueError("discrepancy must be a CompositePathFunctionalDiscrepancy")
    names = [str(block.name) for block in tuple(discrepancy.blocks)]
    if len(set(names)) != len(names):
        raise ValueError("discrepancy block names must be unique")
    if target_paths.ndim != 3 or reference_paths.ndim != 3:
        raise ValueError("target_paths and reference_paths must have shape (B, N + 1, d)")
    if tuple(target_paths.shape[1:]) != tuple(reference_paths.shape[1:]):
        raise ValueError("target_paths and reference_paths must have matching path shapes")
    if int(target_paths.shape[1]) != int(model.grid.num_steps) + 1:
        raise ValueError("path time dimensions must align with the model grid")
    training_used = training or model.training
    config_used = config or FrozenResponseAuditConfig()

    parameter_directions, component_metrics, direction_paths = _parameter_target_directions(
        model=model,
        discrepancy=discrepancy,
        target_paths=target_paths,
        training=training_used,
        config=config_used,
    )
    parameter_report, parameter_matrix, parameter_order = _direction_report(parameter_directions)

    path_count = min(int(config_used.path_gradient_batch_size), int(direction_paths.shape[0]))
    target_count = min(int(config_used.target_batch_size), int(target_paths.shape[0]))
    path_generated = direction_paths[:path_count].detach()
    path_target = target_paths[:target_count].to(device=model.device, dtype=model.dtype)
    path_directions: dict[str, torch.Tensor] = {}
    path_values: dict[str, float] = {}
    for block in tuple(discrepancy.blocks):
        name = str(block.name)
        direction, value = _path_gradient(
            _isolated_discrepancy(block),
            path_generated,
            path_target,
            grid=model.grid,
            observed_only=bool(training_used.observed_only),
            lambda_scale=float(training_used.lambda_scale),
        )
        path_directions[name] = direction.reshape(-1)
        path_values[name] = value
    path_directions[FULL_DIRECTION] = torch.stack(tuple(path_directions.values()), dim=0).sum(dim=0)
    path_report, path_matrix, path_order = _direction_report(path_directions)

    metric_reference = reference_paths[: int(config_used.metric_num_paths)].detach().cpu()
    parameter_seed = derive_stream_seed(base_seed=int(config_used.seed), stream_offset=500) or int(config_used.seed)
    baseline_parameter_paths = _sample_common_paths(
        model,
        x0=x0,
        num_paths=int(config_used.metric_num_paths),
        batch_size=int(config_used.metric_batch_size),
        seed=int(parameter_seed),
    )
    baseline_parameter_metrics = _response_metrics(
        baseline_parameter_paths,
        metric_reference,
        config=config_used,
    )

    parameters = tuple(model.network.parameters())
    original_vector = torch.nn.utils.parameters_to_vector(parameters).detach().clone()
    parameter_norm = float(torch.linalg.vector_norm(original_vector.double()).item())
    absolute_parameter_step = float(config_used.parameter_step_fraction) * parameter_norm
    parameter_responses: dict[str, Any] = {}
    driver_target = target_paths[:target_count].to(device=model.device, dtype=model.dtype)
    driver_count = min(path_count, int(baseline_parameter_paths.shape[0]))
    discrepancy_by_name = {
        str(block.name): _isolated_discrepancy(block)
        for block in tuple(discrepancy.blocks)
    }
    discrepancy_by_name[FULL_DIRECTION] = discrepancy
    try:
        for name in parameter_order:
            direction = parameter_directions[name]
            direction_norm = torch.linalg.vector_norm(direction)
            if float(direction_norm.item()) <= torch.finfo(direction.dtype).eps:
                parameter_responses[name] = {"status": "zero_direction"}
                continue
            unit = direction / direction_norm
            variants: dict[str, torch.Tensor] = {}
            variant_metrics: dict[str, dict[str, float]] = {}
            driver_values: dict[str, float] = {}
            for label, sign in (("plus", 1.0), ("minus", -1.0)):
                torch.nn.utils.vector_to_parameters(
                    original_vector + sign * absolute_parameter_step * unit,
                    parameters,
                )
                variants[label] = _sample_common_paths(
                    model,
                    x0=x0,
                    num_paths=int(config_used.metric_num_paths),
                    batch_size=int(config_used.metric_batch_size),
                    seed=int(parameter_seed),
                )
                variant_metrics[label] = _response_metrics(
                    variants[label],
                    metric_reference,
                    config=config_used,
                )
                if name in discrepancy_by_name:
                    driver_values[label] = _discrepancy_value(
                        discrepancy_by_name[name],
                        variants[label][:driver_count].to(model.device),
                        driver_target,
                        grid=model.grid,
                        observed_only=bool(training_used.observed_only),
                    )
            record = _response_record(
                baseline_parameter_metrics,
                variant_metrics["plus"],
                variant_metrics["minus"],
                step=float(config_used.parameter_step_fraction),
            )
            record["direction_norm"] = float(direction_norm.item())
            if name in discrepancy_by_name:
                baseline_driver = _discrepancy_value(
                    discrepancy_by_name[name],
                    baseline_parameter_paths[:driver_count].to(model.device),
                    driver_target,
                    grid=model.grid,
                    observed_only=bool(training_used.observed_only),
                )
                record["driving_discrepancy"] = {
                    "baseline": baseline_driver,
                    "plus": driver_values["plus"],
                    "minus": driver_values["minus"],
                    "central_derivative": (
                        driver_values["plus"] - driver_values["minus"]
                    ) / (2.0 * float(config_used.parameter_step_fraction)),
                }
            parameter_responses[name] = record
    finally:
        torch.nn.utils.vector_to_parameters(original_vector, parameters)

    path_reference = reference_paths[:path_count].detach().cpu()
    baseline_path_metrics = _response_metrics(
        path_generated.detach().cpu(),
        path_reference,
        config=config_used,
    )
    base_return_rms = torch.sqrt(torch.mean(path_returns(path_generated).pow(2))).clamp_min(
        torch.finfo(path_generated.dtype).eps
    )
    path_responses: dict[str, Any] = {}
    for name in path_order:
        flat_direction = path_directions[name]
        direction = flat_direction.reshape_as(path_generated)
        direction_return_rms = torch.sqrt(torch.mean(path_returns(direction).pow(2)))
        if float(direction_return_rms.item()) <= torch.finfo(direction.dtype).eps:
            path_responses[name] = {"status": "zero_direction"}
            continue
        scaled_direction = direction * (base_return_rms / direction_return_rms)
        plus_paths = path_generated + float(config_used.path_return_step_fraction) * scaled_direction
        minus_paths = path_generated - float(config_used.path_return_step_fraction) * scaled_direction
        plus_metrics = _response_metrics(plus_paths.cpu(), path_reference, config=config_used)
        minus_metrics = _response_metrics(minus_paths.cpu(), path_reference, config=config_used)
        driver = discrepancy_by_name[name]
        baseline_driver = _discrepancy_value(
            driver,
            path_generated,
            path_target,
            grid=model.grid,
            observed_only=bool(training_used.observed_only),
        )
        plus_driver = _discrepancy_value(
            driver,
            plus_paths,
            path_target,
            grid=model.grid,
            observed_only=bool(training_used.observed_only),
        )
        minus_driver = _discrepancy_value(
            driver,
            minus_paths,
            path_target,
            grid=model.grid,
            observed_only=bool(training_used.observed_only),
        )
        record = _response_record(
            baseline_path_metrics,
            plus_metrics,
            minus_metrics,
            step=float(config_used.path_return_step_fraction),
        )
        record["direction_return_rms_before_normalization"] = float(direction_return_rms.item())
        record["driving_discrepancy"] = {
            "baseline": baseline_driver,
            "plus": plus_driver,
            "minus": minus_driver,
            "central_derivative": (plus_driver - minus_driver)
            / (2.0 * float(config_used.path_return_step_fraction)),
        }
        path_responses[name] = record

    error_metrics = [
        "future_rv_mean_abs_error",
        "future_rv_std_abs_error",
        "future_rv_iqr_abs_error",
        "prefix_future_correlation_abs_error",
        "return_abs_mean_abs_error",
        "abs_return_acf_error_rms",
        "excess_kurtosis_error_rms",
        "return_std_error_rms",
        "path_swd",
    ]
    metrics: dict[str, Any] = {
        "interpretation": {
            "path_plus": "direct empirical-discrepancy descent in path space",
            "parameter_plus": "block rows are additive J_theta^T(P_b, R_b) target forcing; full also includes shared prediction shrinkage",
            "parameter_full_direction": "negative gradient of the frozen adjoint MSE up to a common positive factor two",
            "parameter_randomness": "plus, minus, and baseline use identical Gaussian noise",
            "central_derivative": "change per unit configured perturbation fraction; negative improves error metrics",
            "error_metrics": error_metrics,
        },
        "config": {
            "direction_batch_size": int(config_used.direction_batch_size),
            "path_gradient_batch_size": path_count,
            "target_batch_size": target_count,
            "metric_num_paths": int(config_used.metric_num_paths),
            "metric_batch_size": int(config_used.metric_batch_size),
            "parameter_step_fraction": float(config_used.parameter_step_fraction),
            "absolute_parameter_step": absolute_parameter_step,
            "parameter_norm": parameter_norm,
            "path_return_step_fraction": float(config_used.path_return_step_fraction),
            "prefix_length": _metric_configuration(num_steps=int(model.grid.num_steps), config=config_used)[0],
            "prefix_window": _metric_configuration(num_steps=int(model.grid.num_steps), config=config_used)[1],
            "horizon": _metric_configuration(num_steps=int(model.grid.num_steps), config=config_used)[2],
            "lags": list(_metric_configuration(num_steps=int(model.grid.num_steps), config=config_used)[3]),
            "swd_projections": int(config_used.swd_projections),
            "seed": int(config_used.seed),
        },
        "block_components": component_metrics,
        "parameter_directions": parameter_report,
        "path_directions": path_report,
        "parameter_responses": parameter_responses,
        "path_responses": path_responses,
        "path_discrepancy_values": path_values,
    }
    tensors = {
        "parameter_direction_matrix": parameter_matrix,
        "path_direction_matrix": path_matrix,
        "parameter_direction_order_utf8": torch.tensor(
            [byte for name in parameter_order for byte in (name + "\0").encode("utf-8")],
            dtype=torch.uint8,
        ),
        "path_direction_order_utf8": torch.tensor(
            [byte for name in path_order for byte in (name + "\0").encode("utf-8")],
            dtype=torch.uint8,
        ),
    }
    return FrozenResponseAuditResult(metrics=metrics, tensors=tensors)


__all__ = [
    "CONTROL_VARIATE_COMPONENT",
    "FULL_DIRECTION",
    "PREDICTION_SHRINKAGE_COMPONENT",
    "FrozenResponseAuditConfig",
    "FrozenResponseAuditResult",
    "audit_frozen_block_responses",
]
