from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Callable, Mapping

import torch

from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.controls import HamiltonianControlInputs
from deep_mkv_gen_path_dt.discrepancies import (
    PathFunctionalDiscrepancy,
    PathFunctionalDiscrepancyResult,
)
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals


COMPONENT_NAMES = ("running_cost", "base_discrepancy", "joint_discrepancy", "full")
REGIME_NAMES = ("low", "middle", "high")


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class ObjectiveEstimatorAuditConfig:
    num_prefixes: int = 128
    num_future_branches: int = 32
    num_target_batches: int = 16
    target_batch_size: int = 256
    step_indices: tuple[int, ...] = (64, 80, 95)
    regime_split_step: int = 64
    regime_prefix_window: int = 16
    seed: int = 1234

    def __post_init__(self) -> None:
        for name in (
            "num_prefixes",
            "num_future_branches",
            "num_target_batches",
            "target_batch_size",
            "regime_split_step",
            "regime_prefix_window",
        ):
            value = _require_int(name, getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
            object.__setattr__(self, name, value)
        if int(self.num_prefixes) < 2:
            raise ValueError("num_prefixes must be >= 2")
        if int(self.num_future_branches) < 2:
            raise ValueError("num_future_branches must be >= 2")
        if int(self.num_target_batches) < 2:
            raise ValueError("num_target_batches must be >= 2")
        if int(self.regime_prefix_window) > int(self.regime_split_step):
            raise ValueError("regime_prefix_window must fit before regime_split_step")
        step_indices = tuple(_require_int("step_index", value) for value in self.step_indices)
        if not step_indices or len(set(step_indices)) != len(step_indices):
            raise ValueError("step_indices must be non-empty and unique")
        if tuple(sorted(step_indices)) != step_indices:
            raise ValueError("step_indices must be strictly increasing")
        if any(value < int(self.regime_split_step) for value in step_indices):
            raise ValueError("every step_index must be at or after regime_split_step")
        if int(self.regime_split_step) not in step_indices:
            raise ValueError("step_indices must contain regime_split_step")
        object.__setattr__(self, "step_indices", step_indices)
        object.__setattr__(self, "seed", _require_int("seed", self.seed))


@dataclass(frozen=True)
class ObjectiveEstimatorAuditResult:
    report: dict[str, object]
    artifacts: dict[str, object]


class _ZeroDiscrepancy:
    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid,
    ) -> PathFunctionalDiscrepancyResult:
        del target_paths, grid
        value = generated_paths.sum() * 0.0
        return PathFunctionalDiscrepancyResult(
            value=value,
            metrics={"zero_discrepancy_value": 0.0},
        )


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().double().reshape(-1)
    right = right.detach().double().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return None
    return float((torch.dot(left, right) / denominator).item())


def _relative_rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().double()
    right = right.detach().double()
    error = torch.sqrt(torch.mean((left - right).pow(2)))
    scale = torch.sqrt(torch.mean(0.5 * (left.pow(2) + right.pow(2)))).clamp_min(
        torch.finfo(torch.float64).eps
    )
    return float((error / scale).item())


def _rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(values.detach().double().pow(2))).item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().double().reshape(-1)
    right = right.detach().double().reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return None
    return float((torch.dot(left, right) / denominator).item())


def _aggregation_curve(
    values: torch.Tensor,
    *,
    axis: int,
    unit_size: int,
) -> list[dict[str, float | None]]:
    if values.ndim != 4:
        raise ValueError("values must have shape (prefixes, futures, target_batches, outputs)")
    count = int(values.shape[axis])
    result = []
    group_size = 1
    while 2 * group_size <= count:
        first_indices = torch.arange(group_size, device=values.device)
        second_indices = torch.arange(group_size, 2 * group_size, device=values.device)
        first = values.index_select(axis, first_indices).mean(dim=(1, 2))
        second = values.index_select(axis, second_indices).mean(dim=(1, 2))
        result.append(
            {
                "averaged_units_per_half": float(group_size),
                "effective_samples_per_half": float(group_size * int(unit_size)),
                "correlation": _correlation(first, second),
                "relative_rmse": _relative_rmse(first, second),
                "first_half_rms": _rms(first),
                "second_half_rms": _rms(second),
            }
        )
        group_size *= 2
    return result


def _two_factor_metrics(
    values: torch.Tensor,
    *,
    target_batch_size: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    if values.ndim != 4:
        raise ValueError("values must have shape (prefixes, futures, target_batches, outputs)")
    prefix_count, future_count, target_count, _ = values.shape
    grand_by_prefix = values.mean(dim=(1, 2))
    future_means = values.mean(dim=2)
    target_means = values.mean(dim=1)
    future_effect = future_means - grand_by_prefix[:, None, :]
    target_effect = target_means - grand_by_prefix[:, None, :]
    interaction = (
        values
        - future_means[:, :, None, :]
        - target_means[:, None, :, :]
        + grand_by_prefix[:, None, None, :]
    )
    future_variance = torch.mean(future_effect.double().pow(2))
    target_variance = torch.mean(target_effect.double().pow(2))
    interaction_variance = torch.mean(interaction.double().pow(2))
    total_conditional_variance = future_variance + target_variance + interaction_variance
    eps = torch.finfo(torch.float64).eps
    between_prefix_variance = torch.var(
        grand_by_prefix.double(),
        dim=0,
        unbiased=True,
    ).mean()
    conditional_mean_noise_variance = (
        future_variance / float(future_count)
        + target_variance / float(target_count)
        + interaction_variance / float(future_count * target_count)
    )
    corrected_signal_variance = (
        between_prefix_variance - conditional_mean_noise_variance
    ).clamp_min(0.0)
    target_half = target_count // 2
    future_half = future_count // 2
    target_first = values[:, :, :target_half, :].mean(dim=(1, 2))
    target_second = values[:, :, target_half : 2 * target_half, :].mean(dim=(1, 2))
    future_first = values[:, :future_half, :, :].mean(dim=(1, 2))
    future_second = values[:, future_half : 2 * future_half, :, :].mean(dim=(1, 2))
    pathwise_rms = _rms(values)
    conditional_mean_rms = _rms(grand_by_prefix)
    estimated_signal_rms = float(torch.sqrt(corrected_signal_variance).item())
    metrics: dict[str, object] = {
        "pathwise_rms": pathwise_rms,
        "conditional_mean_rms": conditional_mean_rms,
        "conditional_mean_to_pathwise_rms_ratio": (
            conditional_mean_rms
            / max(pathwise_rms, torch.finfo(torch.float64).eps)
        ),
        "between_prefix_variance": float(between_prefix_variance.item()),
        "estimated_signal_variance": float(corrected_signal_variance.item()),
        "estimated_signal_rms": estimated_signal_rms,
        "estimated_signal_to_pathwise_rms_ratio": (
            estimated_signal_rms
            / max(pathwise_rms, torch.finfo(torch.float64).eps)
        ),
        "conditional_mean_noise_variance": float(conditional_mean_noise_variance.item()),
        "conditional_mean_signal_to_noise_ratio": float(
            torch.sqrt(
                corrected_signal_variance
                / conditional_mean_noise_variance.clamp_min(eps)
            ).item()
        ),
        "variance_components": {
            "future_branch": float(future_variance.item()),
            "target_batch": float(target_variance.item()),
            "interaction": float(interaction_variance.item()),
            "future_branch_share": float(
                (future_variance / total_conditional_variance.clamp_min(eps)).item()
            ),
            "target_batch_share": float(
                (target_variance / total_conditional_variance.clamp_min(eps)).item()
            ),
            "interaction_share": float(
                (interaction_variance / total_conditional_variance.clamp_min(eps)).item()
            ),
        },
        "target_half_teacher_agreement": {
            "correlation": _correlation(target_first, target_second),
            "relative_rmse": _relative_rmse(target_first, target_second),
            "target_paths_per_half": float(target_half * int(target_batch_size)),
        },
        "future_half_teacher_agreement": {
            "correlation": _correlation(future_first, future_second),
            "relative_rmse": _relative_rmse(future_first, future_second),
            "branches_per_half": float(future_half),
        },
        "target_batch_aggregation": _aggregation_curve(
            values,
            axis=2,
            unit_size=int(target_batch_size),
        ),
        "future_branch_aggregation": _aggregation_curve(
            values,
            axis=1,
            unit_size=1,
        ),
    }
    tensors = {
        "conditional_mean": grand_by_prefix.detach().cpu(),
        "future_effect": future_effect.detach().cpu(),
        "target_effect": target_effect.detach().cpu(),
    }
    return metrics, tensors


def _regime_indices(
    *,
    prefixes: torch.Tensor,
    target_paths: torch.Tensor,
    split_step: int,
    prefix_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    def volatility(paths: torch.Tensor) -> torch.Tensor:
        returns = paths[:, 1 : split_step + 1, :] - paths[:, :split_step, :]
        return torch.sqrt(
            torch.mean(returns[:, -prefix_window:, :].double().pow(2), dim=(1, 2))
        )

    target_volatility = volatility(target_paths)
    thresholds = torch.quantile(
        target_volatility,
        torch.tensor(
            (1.0 / 3.0, 2.0 / 3.0),
            device=target_volatility.device,
            dtype=target_volatility.dtype,
        ),
    )
    return torch.bucketize(volatility(prefixes), thresholds), thresholds


def _control(
    *,
    model: DiscreteMPModel,
    step_index: int,
    prefixes: torch.Tensor,
    p: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    return model.control_map(
        HamiltonianControlInputs(
            step_index=int(step_index),
            x_prefix=prefixes,
            expected_adjoint_next=p,
            expected_adjoint_noise_next=r,
        )
    )


def _regime_component_report(
    *,
    model: DiscreteMPModel,
    step_index: int,
    prefixes: torch.Tensor,
    regimes: torch.Tensor,
    conditional_p: Mapping[str, torch.Tensor],
    conditional_r: Mapping[str, torch.Tensor],
    current_p: torch.Tensor,
    current_r: torch.Tensor,
) -> dict[str, object]:
    full_p = conditional_p["full"]
    full_r = conditional_r["full"]
    full_control = _control(
        model=model,
        step_index=step_index,
        prefixes=prefixes,
        p=full_p,
        r=full_r,
    )
    current_control = _control(
        model=model,
        step_index=step_index,
        prefixes=prefixes,
        p=current_p,
        r=current_r,
    )
    state_dim = int(model.architecture.state_dim)
    full_alpha = full_control[:, :state_dim]
    current_alpha = current_control[:, :state_dim]
    full_sigma = full_control[:, state_dim:]
    current_sigma = current_control[:, state_dim:]
    component_alpha_effects = {}
    component_sigma_effects = {}
    for component in COMPONENT_NAMES[:-1]:
        without_control = _control(
            model=model,
            step_index=step_index,
            prefixes=prefixes,
            p=full_p - conditional_p[component],
            r=full_r - conditional_r[component],
        )
        component_alpha_effects[component] = (
            full_alpha - without_control[:, :state_dim]
        )
        component_sigma_effects[component] = full_sigma - without_control[
            :, state_dim:
        ]

    report = {}
    for regime_index, regime_name in enumerate(REGIME_NAMES):
        mask = regimes == regime_index
        if not bool(mask.any().item()):
            report[regime_name] = {"count": 0}
            continue
        full_alpha_change = full_alpha[mask] - current_alpha[mask]
        full_sigma_change = full_sigma[mask] - current_sigma[mask]
        components = {}
        for component in COMPONENT_NAMES[:-1]:
            p_values = conditional_p[component][mask]
            r_values = conditional_r[component][mask]
            alpha_effect = component_alpha_effects[component][mask]
            sigma_effect = component_sigma_effects[component][mask]
            components[component] = {
                "conditional_p_mean": float(p_values.double().mean().item()),
                "conditional_p_rms": _rms(p_values),
                "p_cosine_with_full": _cosine(p_values, full_p[mask]),
                "conditional_r_mean": float(r_values.double().mean().item()),
                "conditional_r_rms": _rms(r_values),
                "r_cosine_with_full": _cosine(r_values, full_r[mask]),
                "marginal_alpha_effect_mean": float(
                    alpha_effect.double().mean().item()
                ),
                "marginal_alpha_effect_rms": _rms(alpha_effect),
                "marginal_alpha_effect_positive_fraction": float(
                    (alpha_effect > 0.0).double().mean().item()
                ),
                "marginal_sigma_effect_mean": float(
                    sigma_effect.double().mean().item()
                ),
                "marginal_sigma_effect_rms": _rms(sigma_effect),
                "marginal_sigma_effect_positive_fraction": float(
                    (sigma_effect > 0.0).double().mean().item()
                ),
            }
        report[regime_name] = {
            "count": int(mask.sum().item()),
            "current_alpha_mean": float(current_alpha[mask].double().mean().item()),
            "full_target_alpha_mean": float(full_alpha[mask].double().mean().item()),
            "full_target_alpha_change_mean": float(
                full_alpha_change.double().mean().item()
            ),
            "full_target_alpha_change_rms": _rms(full_alpha_change),
            "current_sigma_mean": float(current_sigma[mask].double().mean().item()),
            "full_target_sigma_mean": float(full_sigma[mask].double().mean().item()),
            "full_target_sigma_change_mean": float(
                full_sigma_change.double().mean().item()
            ),
            "full_target_sigma_change_rms": _rms(full_sigma_change),
            "full_target_sigma_increase_fraction": float(
                (full_sigma_change > 0.0).double().mean().item()
            ),
            "components": components,
        }
    return report


def _component_alignment(
    component: torch.Tensor,
    full: torch.Tensor,
) -> dict[str, float | None]:
    return {
        "component_rms": _rms(component),
        "full_rms": _rms(full),
        "component_to_full_rms_ratio": _rms(component)
        / max(_rms(full), torch.finfo(torch.float64).eps),
        "cosine_with_full": _cosine(component, full),
    }


def _network_projection_metrics(
    prediction: torch.Tensor,
    conditional_target: torch.Tensor,
) -> dict[str, float | None]:
    return {
        "prediction_rms": _rms(prediction),
        "conditional_target_rms": _rms(conditional_target),
        "correlation": _correlation(prediction, conditional_target),
        "cosine": _cosine(prediction, conditional_target),
        "relative_rmse": _relative_rmse(prediction, conditional_target),
    }


def audit_objective_estimator_variance(
    *,
    model: DiscreteMPModel,
    target_paths: torch.Tensor,
    base_discrepancy: PathFunctionalDiscrepancy,
    joint_discrepancy: PathFunctionalDiscrepancy,
    training: DiscreteMPTrainingConfig | None = None,
    config: ObjectiveEstimatorAuditConfig | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> ObjectiveEstimatorAuditResult:
    """Cross future noise and target minibatches for complete-path objective components."""

    diagnostic = config or ObjectiveEstimatorAuditConfig()
    training_config = training or model.training
    target_count, _ = model._validate_target_paths(target_paths)
    if int(diagnostic.num_target_batches) * int(diagnostic.target_batch_size) > target_count:
        raise ValueError("disjoint target batches cannot exceed the target path pool")
    if max(diagnostic.step_indices) >= int(model.grid.num_steps):
        raise ValueError("step_indices must lie before the final grid point")
    target_device = target_paths.to(device=model.device, dtype=model.dtype)
    cpu_generator = torch.Generator(device="cpu").manual_seed(int(diagnostic.seed))
    permutation = torch.randperm(target_count, generator=cpu_generator)[
        : int(diagnostic.num_target_batches) * int(diagnostic.target_batch_size)
    ]
    target_batches = [
        target_device.index_select(
            0,
            permutation[
                batch_index
                * int(diagnostic.target_batch_size) : (batch_index + 1)
                * int(diagnostic.target_batch_size)
            ].to(model.device),
        )
        for batch_index in range(int(diagnostic.num_target_batches))
    ]
    source_indices = torch.randperm(target_count, generator=cpu_generator)[
        : int(diagnostic.num_prefixes)
    ]
    x0 = target_device.index_select(0, source_indices.to(model.device))[:, 0, :]
    base_noise = sample_standard_normals(
        grid=model.grid,
        batch_size=int(diagnostic.num_prefixes),
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=derive_stream_seed(base_seed=int(diagnostic.seed), stream_offset=1),
    )
    zero_discrepancy = _ZeroDiscrepancy()
    step_reports: dict[str, object] = {}
    artifacts: dict[str, object] = {}
    reconstruction_errors = []
    reconstruction_relative_errors = []
    target_batch_uses_full_pool = (
        int(diagnostic.num_target_batches) * int(diagnostic.target_batch_size)
        == target_count
    )

    for step_number, step_index in enumerate(diagnostic.step_indices):
        prefix_count = int(diagnostic.num_prefixes)
        future_count = int(diagnostic.num_future_branches)
        target_batch_count = int(diagnostic.num_target_batches)
        branch_noise = sample_standard_normals(
            grid=model.grid,
            batch_size=prefix_count * future_count,
            noise_dim=int(model.architecture.noise_dim),
            device=model.device,
            dtype=model.dtype,
            seed=derive_stream_seed(
                base_seed=int(diagnostic.seed),
                stream_offset=100 + step_number,
            ),
        ).reshape(
            prefix_count,
            future_count,
            int(model.grid.num_steps),
            int(model.architecture.noise_dim),
        )
        if int(step_index) > 0:
            branch_noise[:, :, : int(step_index), :] = base_noise[
                :, None, : int(step_index), :
            ]
        flat_noise = branch_noise.reshape(
            prefix_count * future_count,
            int(model.grid.num_steps),
            int(model.architecture.noise_dim),
        )
        flat_x0 = x0[:, None, :].expand(
            prefix_count,
            future_count,
            int(x0.shape[-1]),
        ).reshape(prefix_count * future_count, int(x0.shape[-1]))
        with torch.no_grad():
            rollout = model.rollout(x0=flat_x0, noise=flat_noise)
        paths = rollout.paths.reshape(
            prefix_count,
            future_count,
            int(model.grid.num_steps) + 1,
            int(model.architecture.state_dim),
        )
        controls = rollout.controls.reshape(
            prefix_count,
            future_count,
            int(model.grid.num_steps),
            int(rollout.controls.shape[-1]),
        )
        predicted_p = rollout.expected_adjoint_next.reshape(
            prefix_count,
            future_count,
            int(model.grid.num_steps),
            -1,
        )
        predicted_r = rollout.expected_adjoint_noise_next.reshape(
            prefix_count,
            future_count,
            int(model.grid.num_steps),
            -1,
        )
        prefix = paths[:, 0, : int(step_index) + 1, :]
        prefix_difference = float(
            (
                paths[:, :, : int(step_index) + 1, :]
                - prefix[:, None, :, :]
            )
            .abs()
            .max()
            .item()
        )
        component_p_lists: dict[str, list[torch.Tensor]] = {
            "running_cost": [],
            "base_discrepancy": [],
            "joint_discrepancy": [],
        }
        component_r_lists: dict[str, list[torch.Tensor]] = {
            "running_cost": [],
            "base_discrepancy": [],
            "joint_discrepancy": [],
        }
        for branch_index in range(future_count):
            branch_paths = paths[:, branch_index, :, :]
            branch_controls = controls[:, branch_index, :, :]
            branch_noise_values = branch_noise[:, branch_index, :, :]
            running_p_all, _ = model._path_functional_adjoint_targets(
                generated_path=branch_paths,
                controls=branch_controls,
                target_path=target_batches[0],
                training=training_config,
                discrepancy=zero_discrepancy,
                include_running_cost=True,
            )
            running_r_all = model._adjoint_noise_target(
                pathwise_adjoint_target=running_p_all,
                moments_noise_target_dim=int(predicted_r.shape[-1]),
                noise=branch_noise_values,
            )
            component_p_lists["running_cost"].append(
                running_p_all[:, int(step_index), :]
            )
            component_r_lists["running_cost"].append(
                running_r_all[:, int(step_index), :]
            )
            branch_base_p = []
            branch_base_r = []
            branch_joint_p = []
            branch_joint_r = []
            for target_batch in target_batches:
                base_p_all, _ = model._path_functional_adjoint_targets(
                    generated_path=branch_paths,
                    target_path=target_batch,
                    training=training_config,
                    discrepancy=base_discrepancy,
                    include_running_cost=False,
                )
                joint_p_all, _ = model._path_functional_adjoint_targets(
                    generated_path=branch_paths,
                    target_path=target_batch,
                    training=training_config,
                    discrepancy=joint_discrepancy,
                    include_running_cost=False,
                )
                base_r_all = model._adjoint_noise_target(
                    pathwise_adjoint_target=base_p_all,
                    moments_noise_target_dim=int(predicted_r.shape[-1]),
                    noise=branch_noise_values,
                )
                joint_r_all = model._adjoint_noise_target(
                    pathwise_adjoint_target=joint_p_all,
                    moments_noise_target_dim=int(predicted_r.shape[-1]),
                    noise=branch_noise_values,
                )
                branch_base_p.append(base_p_all[:, int(step_index), :])
                branch_base_r.append(base_r_all[:, int(step_index), :])
                branch_joint_p.append(joint_p_all[:, int(step_index), :])
                branch_joint_r.append(joint_r_all[:, int(step_index), :])
            if progress_callback is not None and (
                branch_index == 0
                or (branch_index + 1) % max(1, future_count // 8) == 0
                or branch_index + 1 == future_count
            ):
                progress_callback(
                    int(step_index),
                    int(branch_index) + 1,
                    future_count,
                )
            component_p_lists["base_discrepancy"].append(
                torch.stack(branch_base_p, dim=1)
            )
            component_r_lists["base_discrepancy"].append(
                torch.stack(branch_base_r, dim=1)
            )
            component_p_lists["joint_discrepancy"].append(
                torch.stack(branch_joint_p, dim=1)
            )
            component_r_lists["joint_discrepancy"].append(
                torch.stack(branch_joint_r, dim=1)
            )

        component_p = {
            "running_cost": torch.stack(
                component_p_lists["running_cost"],
                dim=1,
            )[:, :, None, :].expand(-1, -1, target_batch_count, -1),
            "base_discrepancy": torch.stack(
                component_p_lists["base_discrepancy"],
                dim=1,
            ),
            "joint_discrepancy": torch.stack(
                component_p_lists["joint_discrepancy"],
                dim=1,
            ),
        }
        component_r = {
            "running_cost": torch.stack(
                component_r_lists["running_cost"],
                dim=1,
            )[:, :, None, :].expand(-1, -1, target_batch_count, -1),
            "base_discrepancy": torch.stack(
                component_r_lists["base_discrepancy"],
                dim=1,
            ),
            "joint_discrepancy": torch.stack(
                component_r_lists["joint_discrepancy"],
                dim=1,
            ),
        }
        component_p["full"] = (
            component_p["running_cost"]
            + component_p["base_discrepancy"]
            + component_p["joint_discrepancy"]
        )
        component_r["full"] = (
            component_r["running_cost"]
            + component_r["base_discrepancy"]
            + component_r["joint_discrepancy"]
        )

        direct_p, _ = model._path_functional_adjoint_targets(
            generated_path=paths[:, 0, :, :],
            controls=controls[:, 0, :, :],
            target_path=target_batches[0],
            training=training_config,
            include_running_cost=True,
        )
        direct_r = model._adjoint_noise_target(
            pathwise_adjoint_target=direct_p,
            moments_noise_target_dim=int(predicted_r.shape[-1]),
            noise=branch_noise[:, 0, :, :],
        )
        p_reconstruction_error = float(
            (
                direct_p[:, int(step_index), :]
                - component_p["full"][:, 0, 0, :]
            )
            .abs()
            .max()
            .item()
        )
        r_reconstruction_error = float(
            (
                direct_r[:, int(step_index), :]
                - component_r["full"][:, 0, 0, :]
            )
            .abs()
            .max()
            .item()
        )
        p_reconstruction_scale = max(
            float(direct_p[:, int(step_index), :].abs().max().item()),
            float(component_p["full"][:, 0, 0, :].abs().max().item()),
            torch.finfo(torch.float64).eps,
        )
        r_reconstruction_scale = max(
            float(direct_r[:, int(step_index), :].abs().max().item()),
            float(component_r["full"][:, 0, 0, :].abs().max().item()),
            torch.finfo(torch.float64).eps,
        )
        p_reconstruction_relative_error = (
            p_reconstruction_error / p_reconstruction_scale
        )
        r_reconstruction_relative_error = (
            r_reconstruction_error / r_reconstruction_scale
        )
        reconstruction_errors.extend((p_reconstruction_error, r_reconstruction_error))
        reconstruction_relative_errors.extend(
            (p_reconstruction_relative_error, r_reconstruction_relative_error)
        )
        component_metrics = {}
        conditional_p = {}
        conditional_r = {}
        for component in COMPONENT_NAMES:
            p_metrics, p_tensors = _two_factor_metrics(
                component_p[component],
                target_batch_size=int(diagnostic.target_batch_size),
            )
            r_metrics, r_tensors = _two_factor_metrics(
                component_r[component],
                target_batch_size=int(diagnostic.target_batch_size),
            )
            conditional_p[component] = component_p[component].mean(dim=(1, 2))
            conditional_r[component] = component_r[component].mean(dim=(1, 2))
            component_metrics[component] = {
                "p": p_metrics,
                "r": r_metrics,
                "p_alignment_with_full": _component_alignment(
                    conditional_p[component],
                    conditional_p["full"]
                    if component == "full"
                    else component_p["full"].mean(dim=(1, 2)),
                ),
                "r_alignment_with_full": _component_alignment(
                    conditional_r[component],
                    conditional_r["full"]
                    if component == "full"
                    else component_r["full"].mean(dim=(1, 2)),
                ),
            }
            for name, tensor in p_tensors.items():
                artifacts[f"step_{step_index}_{component}_p_{name}"] = tensor
            for name, tensor in r_tensors.items():
                artifacts[f"step_{step_index}_{component}_r_{name}"] = tensor

        regimes, thresholds = _regime_indices(
            prefixes=prefix,
            target_paths=target_device,
            split_step=int(diagnostic.regime_split_step),
            prefix_window=int(diagnostic.regime_prefix_window),
        )
        current_p = predicted_p[:, 0, int(step_index), :]
        current_r = predicted_r[:, 0, int(step_index), :]
        network_projection = {
            "p": _network_projection_metrics(
                current_p,
                conditional_p["full"],
            ),
            "r": _network_projection_metrics(
                current_r,
                conditional_r["full"],
            ),
        }
        regime_report = _regime_component_report(
            model=model,
            step_index=int(step_index),
            prefixes=prefix,
            regimes=regimes,
            conditional_p=conditional_p,
            conditional_r=conditional_r,
            current_p=current_p,
            current_r=current_r,
        )
        step_reports[str(int(step_index))] = {
            "step_index": int(step_index),
            "prefix_length": int(step_index) + 1,
            "prefix_max_abs_difference": prefix_difference,
            "regime_thresholds": [
                float(thresholds[0].item()),
                float(thresholds[1].item()),
            ],
            "component_reconstruction_max_abs_error": {
                "p": p_reconstruction_error,
                "r": r_reconstruction_error,
            },
            "component_reconstruction_max_relative_error": {
                "p": p_reconstruction_relative_error,
                "r": r_reconstruction_relative_error,
            },
            "components": component_metrics,
            "network_projection_to_complete_conditional_target": (
                network_projection
            ),
            "regimes": regime_report,
        }
        artifacts[f"step_{step_index}_prefix_paths"] = prefix.detach().cpu()
        artifacts[f"step_{step_index}_regime_indices"] = regimes.detach().cpu()
        for component in COMPONENT_NAMES:
            artifacts[
                f"step_{step_index}_{component}_p_samples"
            ] = component_p[component].detach().cpu()
            artifacts[
                f"step_{step_index}_{component}_r_samples"
            ] = component_r[component].detach().cpu()

    primary = step_reports[str(int(diagnostic.regime_split_step))]
    primary_full_r = primary["components"]["full"]["r"]
    primary_low = primary["regimes"]["low"]
    target_agreement = primary_full_r["target_half_teacher_agreement"]
    future_agreement = primary_full_r["future_half_teacher_agreement"]
    low_sigma_change_value = primary_low.get("full_target_sigma_change_mean")
    low_sigma_change = (
        None
        if low_sigma_change_value is None
        else float(low_sigma_change_value)
    )
    full_pool_stable = (
        target_agreement["correlation"] is not None
        and float(target_agreement["correlation"]) >= 0.5
        and float(target_agreement["relative_rmse"]) <= 1.0
    )
    future_average_stable = (
        future_agreement["correlation"] is not None
        and float(future_agreement["correlation"]) >= 0.5
        and float(future_agreement["relative_rmse"]) <= 1.0
    )
    report: dict[str, object] = {
        "protocol": {
            "selected_generator_frozen": True,
            "same_prefixes_and_future_shocks_crossed_with_target_batches": True,
            "target_batches_are_disjoint": True,
            "target_partition_covers_full_pool": target_batch_uses_full_pool,
            "target_estimator_is_partition_average_not_direct_pooled_gradient": True,
            "running_cost_target_is_target_batch_independent": True,
            "component_sum_reconstructs_complete_target": bool(
                max(reconstruction_relative_errors) <= 1e-5
            ),
            "maximum_principle_running_cost_discrepancy_and_neural_loss_unchanged": True,
            "no_network_training": True,
        },
        "config": {
            **diagnostic.__dict__,
            "target_pool_size": target_count,
            "target_paths_used": int(diagnostic.num_target_batches)
            * int(diagnostic.target_batch_size),
            "device": str(model.device),
            "dtype": str(model.dtype),
        },
        "steps": step_reports,
        "decision": {
            "primary_step": int(diagnostic.regime_split_step),
            "full_target_half_teacher_stable": bool(full_pool_stable),
            "future_half_teacher_stable": bool(future_average_stable),
            "low_regime_partition_average_sigma_change_mean": low_sigma_change,
            "partition_averaged_target_estimator_supported": bool(
                full_pool_stable
                and low_sigma_change is not None
                and low_sigma_change > 0.0
            ),
            "running_cost_only_candidate": bool(
                primary["components"]["running_cost"]["r"]["estimated_signal_rms"]
                > primary["components"]["base_discrepancy"]["r"]["estimated_signal_rms"]
                and low_sigma_change is not None
                and low_sigma_change > 0.0
            ),
        },
        "maximum_component_reconstruction_abs_error": max(reconstruction_errors),
        "maximum_component_reconstruction_relative_error": max(
            reconstruction_relative_errors
        ),
    }
    artifacts["target_permutation"] = permutation.detach().cpu()
    return ObjectiveEstimatorAuditResult(report=report, artifacts=artifacts)


__all__ = [
    "COMPONENT_NAMES",
    "ObjectiveEstimatorAuditConfig",
    "ObjectiveEstimatorAuditResult",
    "audit_objective_estimator_variance",
]
