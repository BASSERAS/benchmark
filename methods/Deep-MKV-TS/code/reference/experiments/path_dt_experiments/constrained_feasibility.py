from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Sequence

import torch

from deep_mkv_gen_path_dt import DiscreteMPModel
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals


CORRELATION_ERROR = "prefix_future_correlation_abs_error"
MARGINAL_CONSTRAINTS = (
    "future_rv_mean_abs_error",
    "future_rv_std_abs_error",
    "future_rv_iqr_abs_error",
)


@dataclass(frozen=True)
class ConstrainedFeasibilityConfig:
    num_paths: int = 2048
    horizon: int = 32
    prefix_length: int = 65
    prefix_window: int = 32
    parameter_step_fractions: Sequence[float] = (1e-6, 3e-6, 1e-5)
    parameter_step_scale: float | None = None
    constraint_change_tolerance: float = 1e-5
    minimum_retained_direction_fraction: float = 0.10
    bootstrap_replicates: int = 0
    bootstrap_budget_multipliers: Sequence[float] = ()
    pareto_target_correlation_reduction_fraction: float = 0.10
    seed: int = 1234

    def __post_init__(self) -> None:
        for name in ("num_paths", "horizon", "prefix_length", "prefix_window"):
            value = int(getattr(self, name))
            minimum = 2 if name == "num_paths" else 1
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
            object.__setattr__(self, name, value)
        fractions = tuple(float(value) for value in self.parameter_step_fractions)
        if len(fractions) == 0 or any(
            not math.isfinite(value) or value <= 0.0 for value in fractions
        ):
            raise ValueError("parameter_step_fractions must be finite and positive")
        if tuple(sorted(fractions)) != fractions or len(set(fractions)) != len(fractions):
            raise ValueError("parameter_step_fractions must be strictly increasing")
        object.__setattr__(self, "parameter_step_fractions", fractions)
        step_scale = (
            None
            if self.parameter_step_scale is None
            else float(self.parameter_step_scale)
        )
        if step_scale is not None and (
            not math.isfinite(step_scale) or step_scale <= 0.0
        ):
            raise ValueError("parameter_step_scale must be positive and finite")
        object.__setattr__(self, "parameter_step_scale", step_scale)
        tolerance = float(self.constraint_change_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("constraint_change_tolerance must be finite and non-negative")
        object.__setattr__(self, "constraint_change_tolerance", tolerance)
        retained = float(self.minimum_retained_direction_fraction)
        if not math.isfinite(retained) or not 0.0 <= retained <= 1.0:
            raise ValueError("minimum_retained_direction_fraction must be in [0, 1]")
        object.__setattr__(self, "minimum_retained_direction_fraction", retained)
        bootstrap_replicates = int(self.bootstrap_replicates)
        if bootstrap_replicates < 0 or bootstrap_replicates == 1:
            raise ValueError("bootstrap_replicates must be zero or at least two")
        object.__setattr__(self, "bootstrap_replicates", bootstrap_replicates)
        budget_multipliers = tuple(
            float(value) for value in self.bootstrap_budget_multipliers
        )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in budget_multipliers
        ):
            raise ValueError(
                "bootstrap_budget_multipliers must be finite and non-negative"
            )
        object.__setattr__(
            self,
            "bootstrap_budget_multipliers",
            budget_multipliers,
        )
        target_reduction = float(
            self.pareto_target_correlation_reduction_fraction
        )
        if not math.isfinite(target_reduction) or not 0.0 < target_reduction <= 1.0:
            raise ValueError(
                "pareto_target_correlation_reduction_fraction must be in (0, 1]"
            )
        object.__setattr__(
            self,
            "pareto_target_correlation_reduction_fraction",
            target_reduction,
        )
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class ConstrainedFeasibilityResult:
    metrics: dict[str, Any]
    tensors: dict[str, torch.Tensor]


def _correlation(left: torch.Tensor, right: torch.Tensor, *, eps: float) -> torch.Tensor:
    left_centered = left - left.mean(dim=0, keepdim=True)
    right_centered = right - right.mean(dim=0, keepdim=True)
    denominator = torch.sqrt(
        left_centered.pow(2).mean(dim=0) * right_centered.pow(2).mean(dim=0)
    ).clamp_min(float(eps))
    return (left_centered * right_centered).mean(dim=0) / denominator


def differentiable_volatility_gate_metrics(
    generated_paths: torch.Tensor,
    reference_paths: torch.Tensor,
    *,
    dt: float,
    prefix_length: int,
    prefix_window: int,
    horizon: int,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    if generated_paths.ndim != 3 or reference_paths.ndim != 3:
        raise ValueError("generated_paths and reference_paths must have shape (B, N + 1, d)")
    if int(generated_paths.shape[0]) < 2 or int(reference_paths.shape[0]) < 2:
        raise ValueError("gate metrics require at least two paths per law")
    if int(generated_paths.shape[1]) != int(reference_paths.shape[1]):
        raise ValueError("generated and reference time dimensions must match")
    if int(generated_paths.shape[2]) != int(reference_paths.shape[2]):
        raise ValueError("generated and reference state dimensions must match")
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be a positive finite float")
    split_index = int(prefix_length) - 1
    if (
        int(prefix_window) < 1
        or int(horizon) < 1
        or split_index < int(prefix_window)
        or split_index + int(horizon) > int(generated_paths.shape[1]) - 1
    ):
        raise ValueError("prefix and future windows must fit the path")

    def volatility_pair(paths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        returns = (paths[:, 1:, :] - paths[:, :-1, :]) / math.sqrt(float(dt))
        prefix = torch.sqrt(
            returns[:, split_index - int(prefix_window) : split_index, :]
            .pow(2)
            .mean(dim=1)
            + float(eps) ** 2
        )
        future = torch.sqrt(
            returns[:, split_index : split_index + int(horizon), :]
            .pow(2)
            .mean(dim=1)
            + float(eps) ** 2
        )
        return prefix, future

    generated_prefix, generated_future = volatility_pair(generated_paths)
    reference = reference_paths.to(
        device=generated_paths.device,
        dtype=generated_paths.dtype,
    )
    reference_prefix, reference_future = volatility_pair(reference)

    generated_mean = generated_future.mean(dim=0)
    target_mean = reference_future.mean(dim=0).detach().clamp_min(float(eps))
    generated_std = generated_future.std(dim=0, unbiased=False)
    target_std = reference_future.std(dim=0, unbiased=False).detach().clamp_min(float(eps))
    quartile_levels = torch.tensor(
        [0.25, 0.75],
        device=generated_paths.device,
        dtype=generated_paths.dtype,
    )
    generated_quartiles = torch.quantile(generated_future, quartile_levels, dim=0)
    reference_quartiles = torch.quantile(reference_future, quartile_levels, dim=0).detach()
    generated_iqr = generated_quartiles[1] - generated_quartiles[0]
    target_iqr = (reference_quartiles[1] - reference_quartiles[0]).clamp_min(float(eps))
    generated_correlation = _correlation(
        generated_prefix,
        generated_future,
        eps=float(eps),
    )
    target_correlation = _correlation(
        reference_prefix,
        reference_future,
        eps=float(eps),
    ).detach()

    mean_ratio_by_asset = generated_mean / target_mean
    std_ratio_by_asset = generated_std / target_std
    iqr_ratio_by_asset = generated_iqr / target_iqr
    correlation_error_by_asset = torch.abs(generated_correlation - target_correlation)
    return {
        "future_rv_mean_ratio": mean_ratio_by_asset.mean(),
        "future_rv_mean_abs_error": torch.abs(mean_ratio_by_asset - 1.0).mean(),
        "future_rv_std_ratio": std_ratio_by_asset.mean(),
        "future_rv_std_abs_error": torch.abs(std_ratio_by_asset - 1.0).mean(),
        "future_rv_iqr_ratio": iqr_ratio_by_asset.mean(),
        "future_rv_iqr_abs_error": torch.abs(iqr_ratio_by_asset - 1.0).mean(),
        "prefix_future_correlation": generated_correlation.mean(),
        "reference_prefix_future_correlation": target_correlation.mean(),
        "prefix_future_correlation_abs_error": correlation_error_by_asset.mean(),
    }


def bootstrap_volatility_gate_standard_errors(
    generated_paths: torch.Tensor,
    reference_paths: torch.Tensor,
    *,
    dt: float,
    prefix_length: int,
    prefix_window: int,
    horizon: int,
    num_replicates: int,
    seed: int,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Bootstrap uncertainty for the four scalar feasibility-gate errors."""

    if int(num_replicates) < 2:
        raise ValueError("num_replicates must be >= 2")
    generated = generated_paths.detach().cpu().double()
    reference = reference_paths.detach().cpu().double()
    split_index = int(prefix_length) - 1

    def volatility_pair(paths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        returns = (paths[:, 1:, :] - paths[:, :-1, :]) / math.sqrt(float(dt))
        prefix = torch.sqrt(
            returns[:, split_index - int(prefix_window) : split_index, :]
            .pow(2)
            .mean(dim=1)
            + float(eps) ** 2
        )
        future = torch.sqrt(
            returns[:, split_index : split_index + int(horizon), :]
            .pow(2)
            .mean(dim=1)
            + float(eps) ** 2
        )
        return prefix, future

    generated_prefix, generated_future = volatility_pair(generated)
    reference_prefix, reference_future = volatility_pair(reference)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    generated_indices = torch.randint(
        int(generated.shape[0]),
        (int(num_replicates), int(generated.shape[0])),
        generator=generator,
    )
    reference_indices = torch.randint(
        int(reference.shape[0]),
        (int(num_replicates), int(reference.shape[0])),
        generator=generator,
    )
    generated_prefix_boot = generated_prefix[generated_indices]
    generated_future_boot = generated_future[generated_indices]
    reference_prefix_boot = reference_prefix[reference_indices]
    reference_future_boot = reference_future[reference_indices]

    def batched_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_centered = left - left.mean(dim=1, keepdim=True)
        right_centered = right - right.mean(dim=1, keepdim=True)
        denominator = torch.sqrt(
            left_centered.pow(2).mean(dim=1)
            * right_centered.pow(2).mean(dim=1)
        ).clamp_min(float(eps))
        return (left_centered * right_centered).mean(dim=1) / denominator

    generated_mean = generated_future_boot.mean(dim=1)
    reference_mean = reference_future_boot.mean(dim=1).clamp_min(float(eps))
    generated_std = generated_future_boot.std(dim=1, unbiased=False)
    reference_std = reference_future_boot.std(dim=1, unbiased=False).clamp_min(
        float(eps)
    )
    generated_quartiles = torch.quantile(
        generated_future_boot,
        torch.tensor([0.25, 0.75], dtype=torch.float64),
        dim=1,
    )
    reference_quartiles = torch.quantile(
        reference_future_boot,
        torch.tensor([0.25, 0.75], dtype=torch.float64),
        dim=1,
    )
    generated_iqr = generated_quartiles[1] - generated_quartiles[0]
    reference_iqr = (reference_quartiles[1] - reference_quartiles[0]).clamp_min(
        float(eps)
    )
    generated_correlation = batched_correlation(
        generated_prefix_boot,
        generated_future_boot,
    )
    reference_correlation = batched_correlation(
        reference_prefix_boot,
        reference_future_boot,
    )
    replicate_errors = {
        CORRELATION_ERROR: torch.abs(
            generated_correlation - reference_correlation
        ).mean(dim=1),
        "future_rv_mean_abs_error": torch.abs(
            generated_mean / reference_mean - 1.0
        ).mean(dim=1),
        "future_rv_std_abs_error": torch.abs(
            generated_std / reference_std - 1.0
        ).mean(dim=1),
        "future_rv_iqr_abs_error": torch.abs(
            generated_iqr / reference_iqr - 1.0
        ).mean(dim=1),
    }
    return {
        name: float(values.std(unbiased=True).item())
        for name, values in replicate_errors.items()
    }


def _flatten_gradients(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    return torch.cat(
        [
            (
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.detach().reshape(-1)
            )
            for parameter, gradient in zip(parameters, gradients)
        ]
    )


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left64 = left.detach().double().reshape(-1)
    right64 = right.detach().double().reshape(-1)
    denominator = torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64)
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return None
    return float(torch.dot(left64, right64).div(denominator).item())


def project_direction_to_linearized_constraints(
    proposed_direction: torch.Tensor,
    constraint_gradients: torch.Tensor,
    *,
    upper_bounds: torch.Tensor | None = None,
    tolerance: float = 1e-10,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Project a direction onto ``A d <= b`` by active-set enumeration."""

    proposed = proposed_direction.detach().double().reshape(-1)
    constraints = constraint_gradients.detach().double()
    if constraints.ndim != 2 or int(constraints.shape[1]) != int(proposed.numel()):
        raise ValueError("constraint_gradients must have shape (m, p)")
    if int(constraints.shape[0]) == 0:
        return proposed.to(proposed_direction), ()
    bounds = (
        torch.zeros(
            int(constraints.shape[0]),
            device=constraints.device,
            dtype=constraints.dtype,
        )
        if upper_bounds is None
        else upper_bounds.detach().double().reshape(-1)
    )
    if int(bounds.numel()) != int(constraints.shape[0]):
        raise ValueError("upper_bounds must have one value per constraint")
    if bool((constraints @ proposed - bounds <= float(tolerance)).all()):
        return proposed.to(proposed_direction), ()

    best_direction: torch.Tensor | None = None
    best_active: tuple[int, ...] | None = None
    best_distance = math.inf
    constraint_count = int(constraints.shape[0])
    feasibility_scale = max(
        1.0,
        float(torch.linalg.vector_norm(constraints, dim=1).max().item())
        * max(1.0, float(torch.linalg.vector_norm(proposed).item())),
    )
    feasibility_tolerance = max(float(tolerance), 1e-9 * feasibility_scale)
    for active_count in range(1, constraint_count + 1):
        for active in itertools.combinations(range(constraint_count), active_count):
            active_matrix = constraints[list(active)]
            gram = active_matrix @ active_matrix.transpose(0, 1)
            active_bounds = bounds[list(active)]
            rhs = active_matrix @ proposed - active_bounds
            multipliers = torch.linalg.lstsq(gram, rhs).solution
            if bool((multipliers < -feasibility_tolerance).any()):
                continue
            candidate = proposed - active_matrix.transpose(0, 1) @ multipliers
            if bool(
                (constraints @ candidate - bounds > feasibility_tolerance).any()
            ):
                continue
            distance = float(torch.sum((candidate - proposed).pow(2)).item())
            if distance < best_distance:
                best_direction = candidate
                best_active = tuple(int(value) for value in active)
                best_distance = distance
    if best_direction is None or best_active is None:
        raise RuntimeError("failed to project onto the linearized constraint cone")
    return best_direction.to(proposed_direction), best_active


def _metric_floats(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().item()) for name, value in metrics.items()}


def _response_record(
    baseline: dict[str, float],
    plus: dict[str, float],
    minus: dict[str, float],
    *,
    absolute_step: float,
    constraint_tolerance: float,
    constraint_error_ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    plus_change = {name: plus[name] - baseline[name] for name in baseline}
    minus_change = {name: minus[name] - baseline[name] for name in baseline}
    central_derivative = {
        name: (plus[name] - minus[name]) / (2.0 * float(absolute_step))
        for name in baseline
    }
    ratios = constraint_error_ratios or {
        name: 0.0 for name in MARGINAL_CONSTRAINTS
    }
    correlation_improvement = max(0.0, -plus_change[CORRELATION_ERROR])
    constraint_allowances = {
        name: float(constraint_tolerance)
        + float(ratios.get(name, 0.0)) * correlation_improvement
        for name in MARGINAL_CONSTRAINTS
    }
    checks = {
        "correlation_improves": plus_change[CORRELATION_ERROR] < 0.0,
        "constraints_do_not_worsen": all(
            plus_change[name] <= constraint_allowances[name]
            for name in MARGINAL_CONSTRAINTS
        ),
    }
    checks["pass"] = bool(
        checks["correlation_improves"] and checks["constraints_do_not_worsen"]
    )
    return {
        "baseline": baseline,
        "plus": plus,
        "minus": minus,
        "plus_change": plus_change,
        "minus_change": minus_change,
        "central_derivative": central_derivative,
        "constraint_error_ratios": {
            name: float(ratios.get(name, 0.0)) for name in MARGINAL_CONSTRAINTS
        },
        "constraint_allowances": constraint_allowances,
        "checks": checks,
    }


def audit_residual_constrained_feasibility(
    *,
    model: DiscreteMPModel,
    training_reference_paths: torch.Tensor,
    validation_reference_paths: torch.Tensor,
    x0: torch.Tensor,
    config: ConstrainedFeasibilityConfig | None = None,
) -> ConstrainedFeasibilityResult:
    """Test whether a local persistence direction can preserve h32 marginals."""

    config_used = config or ConstrainedFeasibilityConfig()
    residual_network = getattr(model.network, "residual_network", None)
    if not isinstance(residual_network, torch.nn.Module):
        raise ValueError("model.network must expose a residual_network")
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in residual_network.named_parameters()
        if parameter.requires_grad
    )
    parameters = tuple(parameter for _, parameter in named_parameters)
    if len(parameters) == 0:
        raise ValueError("residual_network must have trainable parameters")
    expected_points = int(model.grid.num_steps) + 1
    for name, paths in (
        ("training_reference_paths", training_reference_paths),
        ("validation_reference_paths", validation_reference_paths),
    ):
        if paths.ndim != 3 or int(paths.shape[1]) != expected_points:
            raise ValueError(f"{name} must align with the model grid")

    x0_device = x0.to(device=model.device, dtype=model.dtype)
    if int(x0_device.shape[0]) == 1:
        x0_device = x0_device.repeat(int(config_used.num_paths), 1)
    if int(x0_device.shape[0]) != int(config_used.num_paths):
        raise ValueError("x0 must have one row or num_paths rows")
    noise_seed = derive_stream_seed(
        base_seed=int(config_used.seed),
        stream_offset=120_000,
    )
    noise = sample_standard_normals(
        grid=model.grid,
        batch_size=int(config_used.num_paths),
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        # Draw in a fixed dtype and cast afterwards so changing the audit
        # precision does not silently change the Monte Carlo realization.
        dtype=torch.float32,
        seed=int(noise_seed if noise_seed is not None else config_used.seed),
    ).to(dtype=model.dtype)
    training_reference = training_reference_paths.to(
        device=model.device,
        dtype=model.dtype,
    )
    validation_reference = validation_reference_paths.to(
        device=model.device,
        dtype=model.dtype,
    )
    generated = model.rollout(x0=x0_device, noise=noise).paths
    objective_metrics = differentiable_volatility_gate_metrics(
        generated,
        training_reference,
        dt=float(model.grid.dt),
        prefix_length=int(config_used.prefix_length),
        prefix_window=int(config_used.prefix_window),
        horizon=int(config_used.horizon),
    )

    gradient_names = (CORRELATION_ERROR,) + tuple(MARGINAL_CONSTRAINTS)
    gradients: dict[str, torch.Tensor] = {}
    gradient_support: dict[str, dict[str, Any]] = {}
    for position, name in enumerate(gradient_names):
        values = torch.autograd.grad(
            objective_metrics[name],
            parameters,
            retain_graph=position + 1 < len(gradient_names),
            allow_unused=True,
        )
        gradients[name] = _flatten_gradients(values, parameters)
        active_parameters = []
        nonzero_elements = 0
        for (parameter_name, parameter), value in zip(named_parameters, values):
            gradient = torch.zeros_like(parameter) if value is None else value.detach()
            parameter_nonzero = int(torch.count_nonzero(gradient).item())
            nonzero_elements += parameter_nonzero
            if parameter_nonzero > 0:
                active_parameters.append(
                    {
                        "name": parameter_name,
                        "numel": int(parameter.numel()),
                        "nonzero_elements": parameter_nonzero,
                        "gradient_norm": float(
                            torch.linalg.vector_norm(gradient.double()).item()
                        ),
                    }
                )
        gradient_support[name] = {
            "active_parameters": active_parameters,
            "nonzero_elements": nonzero_elements,
            "nonzero_fraction": nonzero_elements
            / float(sum(parameter.numel() for parameter in parameters)),
        }

    correlation_gradient = gradients[CORRELATION_ERROR]
    constraint_matrix = torch.stack(
        [gradients[name] for name in MARGINAL_CONSTRAINTS],
        dim=0,
    )
    unconstrained_direction = -correlation_gradient
    projected_direction, active_indices = project_direction_to_linearized_constraints(
        unconstrained_direction,
        constraint_matrix,
    )
    active_constraints = tuple(MARGINAL_CONSTRAINTS[index] for index in active_indices)
    unconstrained_norm = float(
        torch.linalg.vector_norm(unconstrained_direction.detach().double()).item()
    )
    projected_norm = float(
        torch.linalg.vector_norm(projected_direction.detach().double()).item()
    )
    retained_fraction = projected_norm / max(
        unconstrained_norm,
        torch.finfo(torch.float64).eps,
    )
    linearized_derivatives = {
        name: float(
            torch.dot(
                gradients[name].detach().double(),
                projected_direction.detach().double(),
            ).item()
        )
        for name in gradient_names
    }

    gradient_order = list(gradient_names)
    gradient_matrix = torch.stack(
        [gradients[name].detach().double() for name in gradient_order],
        dim=0,
    )
    pairwise_cosines = [
        [_cosine(gradients[left], gradients[right]) for right in gradient_order]
        for left in gradient_order
    ]
    singular_values = torch.linalg.svdvals(gradient_matrix)
    largest_singular = float(singular_values[0].item())
    rank_tolerance = largest_singular * 1e-8
    numerical_rank = int((singular_values > rank_tolerance).sum().item())

    original_vector = torch.nn.utils.parameters_to_vector(parameters).detach().clone()
    parameter_norm = float(torch.linalg.vector_norm(original_vector.double()).item())
    baseline_paths = generated.detach()

    def metrics_for(
        paths: torch.Tensor,
        reference: torch.Tensor,
    ) -> dict[str, float]:
        return _metric_floats(
            differentiable_volatility_gate_metrics(
                paths,
                reference,
                dt=float(model.grid.dt),
                prefix_length=int(config_used.prefix_length),
                prefix_window=int(config_used.prefix_window),
                horizon=int(config_used.horizon),
            )
        )

    baseline_metrics = {
        "training": metrics_for(baseline_paths, training_reference),
        "validation": metrics_for(baseline_paths, validation_reference),
    }
    parameter_step_scale = (
        float(config_used.parameter_step_scale)
        if config_used.parameter_step_scale is not None
        else parameter_norm
    )
    if parameter_step_scale <= torch.finfo(torch.float64).eps:
        raise ValueError(
            "parameter_step_scale is required when the initial parameter norm is zero"
        )

    bootstrap_payload: dict[str, Any] = {"enabled": False}
    pareto_geometry: dict[str, Any] = {}
    direction_specs: dict[str, tuple[torch.Tensor, dict[str, float]]] = {
        "unconstrained_correlation_descent": (
            unconstrained_direction,
            {name: 0.0 for name in MARGINAL_CONSTRAINTS},
        ),
        "projected_feasible_descent": (
            projected_direction,
            {name: 0.0 for name in MARGINAL_CONSTRAINTS},
        ),
    }
    if int(config_used.bootstrap_replicates) > 0:
        bootstrap_seed = derive_stream_seed(
            base_seed=int(config_used.seed),
            stream_offset=122_000,
        )
        standard_errors = bootstrap_volatility_gate_standard_errors(
            baseline_paths,
            training_reference,
            dt=float(model.grid.dt),
            prefix_length=int(config_used.prefix_length),
            prefix_window=int(config_used.prefix_window),
            horizon=int(config_used.horizon),
            num_replicates=int(config_used.bootstrap_replicates),
            seed=int(
                bootstrap_seed
                if bootstrap_seed is not None
                else config_used.seed
            ),
        )
        target_correlation_improvement = (
            float(config_used.pareto_target_correlation_reduction_fraction)
            * baseline_metrics["training"][CORRELATION_ERROR]
        )
        bootstrap_payload = {
            "enabled": True,
            "replicates": int(config_used.bootstrap_replicates),
            "seed": int(
                bootstrap_seed
                if bootstrap_seed is not None
                else config_used.seed
            ),
            "standard_errors": standard_errors,
            "target_correlation_reduction_fraction": float(
                config_used.pareto_target_correlation_reduction_fraction
            ),
            "target_correlation_improvement": target_correlation_improvement,
        }
        for multiplier in tuple(config_used.bootstrap_budget_multipliers):
            error_ratios = {
                name: float(multiplier) * standard_errors[name]
                / max(target_correlation_improvement, torch.finfo(torch.float64).eps)
                for name in MARGINAL_CONSTRAINTS
            }
            effective_constraints = torch.stack(
                [
                    gradients[name]
                    + float(error_ratios[name]) * correlation_gradient
                    for name in MARGINAL_CONSTRAINTS
                ],
                dim=0,
            )
            relaxed_direction, relaxed_active_indices = (
                project_direction_to_linearized_constraints(
                    unconstrained_direction,
                    effective_constraints,
                )
            )
            relaxed_norm = float(
                torch.linalg.vector_norm(relaxed_direction.double()).item()
            )
            relaxed_retained = relaxed_norm / max(
                unconstrained_norm,
                torch.finfo(torch.float64).eps,
            )
            name = f"bootstrap_{float(multiplier):g}se_budget"
            direction_specs[name] = (relaxed_direction, error_ratios)
            if relaxed_norm > torch.finfo(torch.float64).eps:
                relaxed_unit = relaxed_direction / relaxed_norm
                unit_derivatives = {
                    metric_name: float(
                        torch.dot(
                            gradients[metric_name].double(),
                            relaxed_unit.double(),
                        ).item()
                    )
                    for metric_name in gradient_names
                }
                correlation_rate = -unit_derivatives[CORRELATION_ERROR]
                target_step = (
                    target_correlation_improvement / correlation_rate
                    if correlation_rate > torch.finfo(torch.float64).eps
                    else None
                )
            else:
                unit_derivatives = {metric_name: 0.0 for metric_name in gradient_names}
                target_step = None
            pareto_geometry[name] = {
                "bootstrap_budget_multiplier": float(multiplier),
                "constraint_error_ratios": error_ratios,
                "active_constraints": [
                    MARGINAL_CONSTRAINTS[index]
                    for index in relaxed_active_indices
                ],
                "direction_norm": relaxed_norm,
                "retained_direction_fraction": relaxed_retained,
                "unit_directional_derivatives": unit_derivatives,
                "parameter_step_for_target_correlation_improvement": target_step,
                "predicted_metric_changes_at_target": (
                    None
                    if target_step is None
                    else {
                        metric_name: float(target_step)
                        * unit_derivatives[metric_name]
                        for metric_name in gradient_names
                    }
                ),
            }

    direction_payload: dict[str, Any] = {}
    try:
        for direction_name, (direction, constraint_error_ratios) in direction_specs.items():
            direction_norm = float(torch.linalg.vector_norm(direction.double()).item())
            step_payload: dict[str, Any] = {}
            if direction_norm <= torch.finfo(torch.float64).eps:
                direction_payload[direction_name] = {
                    "direction_norm": direction_norm,
                    "steps": step_payload,
                }
                continue
            unit_direction = direction / float(direction_norm)
            for fraction in tuple(config_used.parameter_step_fractions):
                absolute_step = float(fraction) * parameter_step_scale
                response_by_reference: dict[str, Any] = {}
                path_by_sign: dict[int, torch.Tensor] = {}
                for sign in (1, -1):
                    with torch.no_grad():
                        torch.nn.utils.vector_to_parameters(
                            original_vector + float(sign) * absolute_step * unit_direction,
                            parameters,
                        )
                        path_by_sign[sign] = model.rollout(
                            x0=x0_device,
                            noise=noise,
                        ).paths.detach()
                for reference_name, reference in (
                    ("training", training_reference),
                    ("validation", validation_reference),
                ):
                    response_by_reference[reference_name] = _response_record(
                        baseline_metrics[reference_name],
                        metrics_for(path_by_sign[1], reference),
                        metrics_for(path_by_sign[-1], reference),
                        absolute_step=absolute_step,
                        constraint_tolerance=float(
                            config_used.constraint_change_tolerance
                        ),
                        constraint_error_ratios=constraint_error_ratios,
                    )
                step_payload[str(fraction)] = {
                    "parameter_step_fraction": float(fraction),
                    "absolute_parameter_step": absolute_step,
                    "responses": response_by_reference,
                }
            direction_payload[direction_name] = {
                "direction_norm": direction_norm,
                "constraint_error_ratios": constraint_error_ratios,
                "steps": step_payload,
            }
    finally:
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(original_vector, parameters)

    projected_validation_checks = [
        record["responses"]["validation"]["checks"]
        for record in direction_payload["projected_feasible_descent"]["steps"].values()
    ]
    finite_difference_pass = bool(
        len(projected_validation_checks) > 0
        and all(bool(record["pass"]) for record in projected_validation_checks)
    )
    meaningful_direction = retained_fraction >= float(
        config_used.minimum_retained_direction_fraction
    )
    local_feasibility_pass = bool(meaningful_direction and finite_difference_pass)
    pareto_decisions: dict[str, Any] = {}
    for name, geometry in pareto_geometry.items():
        validation_checks = [
            record["responses"]["validation"]["checks"]
            for record in direction_payload[name]["steps"].values()
        ]
        response_pass = bool(
            len(validation_checks) > 0
            and all(bool(record["pass"]) for record in validation_checks)
        )
        material = float(geometry["retained_direction_fraction"]) >= float(
            config_used.minimum_retained_direction_fraction
        )
        pareto_decisions[name] = {
            "meaningful_projected_direction": material,
            "finite_difference_validation_pass": response_pass,
            "local_feasibility_pass": bool(material and response_pass),
        }

    metrics: dict[str, Any] = {
        "config": {
            "num_paths": int(config_used.num_paths),
            "horizon": int(config_used.horizon),
            "prefix_length": int(config_used.prefix_length),
            "prefix_window": int(config_used.prefix_window),
            "parameter_step_fractions": list(config_used.parameter_step_fractions),
            "parameter_step_scale": float(parameter_step_scale),
            "parameter_step_scale_source": (
                "config" if config_used.parameter_step_scale is not None else "parameter_norm"
            ),
            "constraint_change_tolerance": float(
                config_used.constraint_change_tolerance
            ),
            "minimum_retained_direction_fraction": float(
                config_used.minimum_retained_direction_fraction
            ),
            "bootstrap_replicates": int(config_used.bootstrap_replicates),
            "bootstrap_budget_multipliers": list(
                config_used.bootstrap_budget_multipliers
            ),
            "pareto_target_correlation_reduction_fraction": float(
                config_used.pareto_target_correlation_reduction_fraction
            ),
            "seed": int(config_used.seed),
            "noise_seed": int(noise_seed if noise_seed is not None else config_used.seed),
            "noise_draw_dtype": "torch.float32",
        },
        "baseline_metrics": baseline_metrics,
        "gradient_geometry": {
            "order": gradient_order,
            "total_parameter_count": int(
                sum(parameter.numel() for parameter in parameters)
            ),
            "support_by_objective": gradient_support,
            "norms": {
                name: float(torch.linalg.vector_norm(gradients[name].double()).item())
                for name in gradient_order
            },
            "pairwise_cosines": pairwise_cosines,
            "singular_values": [float(value.item()) for value in singular_values],
            "numerical_rank": numerical_rank,
            "relative_rank_tolerance": 1e-8,
        },
        "projection": {
            "active_constraints": list(active_constraints),
            "unconstrained_direction_norm": unconstrained_norm,
            "projected_direction_norm": projected_norm,
            "retained_direction_fraction": retained_fraction,
            "cosine_with_unconstrained_direction": _cosine(
                projected_direction,
                unconstrained_direction,
            ),
            "linearized_directional_derivatives": linearized_derivatives,
        },
        "bootstrap_uncertainty": bootstrap_payload,
        "bootstrap_pareto_geometry": pareto_geometry,
        "finite_difference_responses": direction_payload,
        "decision": {
            "meaningful_projected_direction": meaningful_direction,
            "finite_difference_validation_pass": finite_difference_pass,
            "local_feasibility_pass": local_feasibility_pass,
            "bootstrap_pareto": pareto_decisions,
        },
    }
    tensors = {
        "gradient_matrix": gradient_matrix.detach().cpu(),
        "constraint_gradient_matrix": constraint_matrix.detach().cpu(),
        "unconstrained_direction": unconstrained_direction.detach().cpu(),
        "projected_direction": projected_direction.detach().cpu(),
        "original_parameter_vector": original_vector.detach().cpu(),
        "noise": noise.detach().cpu(),
    }
    for name, (direction, _) in direction_specs.items():
        if name.startswith("bootstrap_"):
            tensors[f"direction_{name}"] = direction.detach().cpu()
    return ConstrainedFeasibilityResult(metrics=metrics, tensors=tensors)
