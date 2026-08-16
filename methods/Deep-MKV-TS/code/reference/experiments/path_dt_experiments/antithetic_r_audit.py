from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Callable

import torch

from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.controls import HamiltonianControlInputs
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals
from path_dt_experiments.hamiltonian_regime import (
    REGIME_NAMES,
    prefix_volatility_regimes,
)


ESTIMATOR_NAMES = ("antithetic", "iid")
BRANCH_NAMES = ("plus", "minus", "iid_first", "iid_second")


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class AntitheticRAuditConfig:
    num_prefixes: int = 128
    num_pairs: int = 64
    num_target_batches: int = 4
    target_batch_size: int = 256
    step_indices: tuple[int, ...] = (64, 80, 95)
    regime_split_step: int = 64
    regime_prefix_window: int = 16
    bootstrap_replicates: int = 2_000
    bootstrap_confidence: float = 0.95
    maximum_material_variance_ratio: float = 0.75
    minimum_agreement_improvement: float = 0.10
    minimum_conditional_mean_correlation: float = 0.50
    seed: int = 1234

    def __post_init__(self) -> None:
        for name in (
            "num_prefixes",
            "num_pairs",
            "num_target_batches",
            "target_batch_size",
            "regime_split_step",
            "regime_prefix_window",
            "bootstrap_replicates",
        ):
            value = _require_int(name, getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
            object.__setattr__(self, name, value)
        if int(self.num_prefixes) < 2:
            raise ValueError("num_prefixes must be >= 2")
        if int(self.num_pairs) < 4 or int(self.num_pairs) % 2 != 0:
            raise ValueError("num_pairs must be an even integer >= 4")
        if int(self.num_target_batches) < 2 or int(self.num_target_batches) % 2 != 0:
            raise ValueError("num_target_batches must be an even integer >= 2")
        if int(self.regime_prefix_window) > int(self.regime_split_step):
            raise ValueError("regime_prefix_window must fit before regime_split_step")
        step_indices = tuple(_require_int("step_index", value) for value in self.step_indices)
        if not step_indices or tuple(sorted(set(step_indices))) != step_indices:
            raise ValueError("step_indices must be strictly increasing and unique")
        if any(value < int(self.regime_split_step) for value in step_indices):
            raise ValueError("every step_index must be at or after regime_split_step")
        if int(self.regime_split_step) not in step_indices:
            raise ValueError("step_indices must contain regime_split_step")
        confidence = float(self.bootstrap_confidence)
        if not 0.0 < confidence < 1.0:
            raise ValueError("bootstrap_confidence must lie strictly between zero and one")
        variance_ratio = float(self.maximum_material_variance_ratio)
        if not 0.0 < variance_ratio < 1.0:
            raise ValueError("maximum_material_variance_ratio must lie between zero and one")
        agreement = float(self.minimum_agreement_improvement)
        if not 0.0 <= agreement <= 2.0:
            raise ValueError("minimum_agreement_improvement must lie between zero and two")
        mean_correlation = float(self.minimum_conditional_mean_correlation)
        if not -1.0 <= mean_correlation <= 1.0:
            raise ValueError("minimum_conditional_mean_correlation must lie between -1 and one")
        object.__setattr__(self, "step_indices", step_indices)
        object.__setattr__(self, "bootstrap_confidence", confidence)
        object.__setattr__(self, "maximum_material_variance_ratio", variance_ratio)
        object.__setattr__(self, "minimum_agreement_improvement", agreement)
        object.__setattr__(
            self,
            "minimum_conditional_mean_correlation",
            mean_correlation,
        )
        object.__setattr__(self, "seed", _require_int("seed", self.seed))


@dataclass(frozen=True)
class AntitheticRAuditResult:
    report: dict[str, object]
    artifacts: dict[str, object]


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


def _relative_rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    left_values = left.detach().double()
    right_values = right.detach().double()
    error = torch.sqrt(torch.mean((left_values - right_values).pow(2)))
    scale = torch.sqrt(
        torch.mean(0.5 * (left_values.pow(2) + right_values.pow(2)))
    ).clamp_min(torch.finfo(torch.float64).eps)
    return float((error / scale).item())


def _center_over_pairs(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 4:
        raise ValueError("values must have shape (prefixes, pairs, target_batches, outputs)")
    return values - values.mean(dim=1, keepdim=True)


def _pair_estimator_metrics(
    values: torch.Tensor,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    if values.ndim != 4:
        raise ValueError("values must have shape (prefixes, pairs, target_batches, outputs)")
    pair_count = int(values.shape[1])
    target_count = int(values.shape[2])
    pair_half = pair_count // 2
    target_half = target_count // 2
    centered = _center_over_pairs(values)
    per_prefix_future_variance = centered.double().pow(2).mean(dim=(1, 2, 3))
    conditional_mean = values.mean(dim=(1, 2))
    target_means = values.mean(dim=1)
    target_centered = target_means - target_means.mean(dim=1, keepdim=True)
    first_pair_half = values[:, :pair_half, :, :].mean(dim=(1, 2))
    second_pair_half = values[:, pair_half:, :, :].mean(dim=(1, 2))
    first_target_half = values[:, :, :target_half, :].mean(dim=(1, 2))
    second_target_half = values[:, :, target_half:, :].mean(dim=(1, 2))
    return (
        {
            "conditional_mean_rms": _rms(conditional_mean),
            "future_pair_variance": float(per_prefix_future_variance.mean().item()),
            "future_pair_standard_deviation": float(
                torch.sqrt(per_prefix_future_variance.mean()).item()
            ),
            "target_batch_variance_of_pair_mean": float(
                target_centered.double().pow(2).mean().item()
            ),
            "pair_half_teacher_agreement": {
                "correlation": _correlation(first_pair_half, second_pair_half),
                "relative_rmse": _relative_rmse(first_pair_half, second_pair_half),
                "pairs_per_half": pair_half,
                "paths_per_half": 2 * pair_half,
            },
            "target_half_teacher_agreement": {
                "correlation": _correlation(first_target_half, second_target_half),
                "relative_rmse": _relative_rmse(
                    first_target_half,
                    second_target_half,
                ),
                "target_batches_per_half": target_half,
            },
        },
        {
            "conditional_mean": conditional_mean.detach().cpu(),
            "per_prefix_future_variance": per_prefix_future_variance.detach().cpu(),
            "first_pair_half_mean": first_pair_half.detach().cpu(),
            "second_pair_half_mean": second_pair_half.detach().cpu(),
        },
    )


def _bootstrap_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    numerator_values = numerator.detach().double().reshape(-1)
    denominator_values = denominator.detach().double().reshape(-1)
    if tuple(numerator_values.shape) != tuple(denominator_values.shape):
        raise ValueError("bootstrap ratio inputs must have matching shapes")
    count = int(numerator_values.numel())
    if count < 2:
        raise ValueError("bootstrap ratio requires at least two units")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        count,
        (int(replicates), count),
        generator=generator,
        device="cpu",
    )
    numerator_cpu = numerator_values.cpu()
    denominator_cpu = denominator_values.cpu()
    ratios = numerator_cpu[indices].mean(dim=1) / denominator_cpu[
        indices
    ].mean(dim=1).clamp_min(torch.finfo(torch.float64).eps)
    tail = 0.5 * (1.0 - float(confidence))
    quantiles = torch.quantile(
        ratios,
        torch.tensor((tail, 1.0 - tail), dtype=torch.float64),
    )
    estimate = numerator_cpu.mean() / denominator_cpu.mean().clamp_min(
        torch.finfo(torch.float64).eps
    )
    return {
        "estimate": float(estimate.item()),
        "ci_lower": float(quantiles[0].item()),
        "ci_upper": float(quantiles[1].item()),
        "replicates": int(replicates),
        "confidence": float(confidence),
        "num_prefixes": count,
    }


def _bootstrap_mean(
    values: torch.Tensor,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    per_prefix = values.detach().double()
    if per_prefix.ndim == 1:
        per_prefix = per_prefix[:, None]
    per_prefix = per_prefix.mean(dim=tuple(range(1, per_prefix.ndim)))
    count = int(per_prefix.numel())
    if count < 2:
        raise ValueError("bootstrap mean requires at least two prefixes")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        count,
        (int(replicates), count),
        generator=generator,
        device="cpu",
    )
    estimates = per_prefix.cpu()[indices].mean(dim=1)
    tail = 0.5 * (1.0 - float(confidence))
    quantiles = torch.quantile(
        estimates,
        torch.tensor((tail, 1.0 - tail), dtype=torch.float64),
    )
    return {
        "estimate": float(per_prefix.mean().item()),
        "ci_lower": float(quantiles[0].item()),
        "ci_upper": float(quantiles[1].item()),
        "replicates": int(replicates),
        "confidence": float(confidence),
        "num_prefixes": count,
    }


def _residual_correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    return _correlation(_center_over_pairs(left), _center_over_pairs(right))


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
            parameters=model.parameters,
        )
    )


def _method_sigma(
    *,
    model: DiscreteMPModel,
    step_index: int,
    prefixes: torch.Tensor,
    p_values: torch.Tensor,
    r_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    pair_count = int(p_values.shape[1])
    pair_half = pair_count // 2

    def sigma(p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        control = _control(
            model=model,
            step_index=step_index,
            prefixes=prefixes,
            p=p,
            r=r,
        )
        state_dim = int(model.architecture.state_dim)
        if int(control.shape[-1]) != 2 * state_dim:
            raise ValueError("antithetic R audit requires controls packed as [alpha, sigma]")
        return control[:, state_dim:]

    full = sigma(
        p_values.mean(dim=(1, 2)),
        r_values.mean(dim=(1, 2)),
    )
    first_half = sigma(
        p_values[:, :pair_half, :, :].mean(dim=(1, 2)),
        r_values[:, :pair_half, :, :].mean(dim=(1, 2)),
    )
    second_half = sigma(
        p_values[:, pair_half:, :, :].mean(dim=(1, 2)),
        r_values[:, pair_half:, :, :].mean(dim=(1, 2)),
    )
    target_batch_sigmas = [
        sigma(
            p_values[:, :, target_index, :].mean(dim=1),
            r_values[:, :, target_index, :].mean(dim=1),
        )
        for target_index in range(int(p_values.shape[2]))
    ]
    return full, first_half, second_half, target_batch_sigmas


def _regime_report(
    *,
    model: DiscreteMPModel,
    step_index: int,
    prefixes: torch.Tensor,
    regimes: torch.Tensor,
    current_p: torch.Tensor,
    current_r: torch.Tensor,
    estimator_p: dict[str, torch.Tensor],
    estimator_r: dict[str, torch.Tensor],
    per_prefix_variances: dict[str, torch.Tensor],
    config: AntitheticRAuditConfig,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    state_dim = int(model.architecture.state_dim)
    current_sigma = _control(
        model=model,
        step_index=step_index,
        prefixes=prefixes,
        p=current_p,
        r=current_r,
    )[:, state_dim:]
    sigma_values = {}
    for method in ESTIMATOR_NAMES:
        sigma_values[method] = _method_sigma(
            model=model,
            step_index=step_index,
            prefixes=prefixes,
            p_values=estimator_p[method],
            r_values=estimator_r[method],
        )

    report: dict[str, object] = {}
    artifacts: dict[str, torch.Tensor] = {"current_sigma": current_sigma.detach().cpu()}
    for regime_index, regime_name in enumerate(REGIME_NAMES):
        mask = regimes == regime_index
        count = int(mask.sum().item())
        if count < 2:
            report[regime_name] = {"count": count}
            continue
        ant_variance = per_prefix_variances["antithetic"][mask]
        iid_variance = per_prefix_variances["iid"][mask]
        variance_ratio = _bootstrap_ratio(
            ant_variance,
            iid_variance,
            replicates=int(config.bootstrap_replicates),
            confidence=float(config.bootstrap_confidence),
            seed=int(config.seed) + 10_000 + 100 * int(step_index) + regime_index,
        )
        methods = {}
        for method_index, method in enumerate(ESTIMATOR_NAMES):
            full_sigma, first_sigma, second_sigma, target_sigmas = sigma_values[method]
            full_delta = full_sigma[mask] - current_sigma[mask]
            first_delta = first_sigma[mask] - current_sigma[mask]
            second_delta = second_sigma[mask] - current_sigma[mask]
            target_delta_means = [
                float((value[mask] - current_sigma[mask]).double().mean().item())
                for value in target_sigmas
            ]
            direction_product = float(first_delta.double().mean().item()) * float(
                second_delta.double().mean().item()
            )
            methods[method] = {
                "teacher_sigma_mean": float(full_sigma[mask].double().mean().item()),
                "sigma_change": _bootstrap_mean(
                    full_delta,
                    replicates=int(config.bootstrap_replicates),
                    confidence=float(config.bootstrap_confidence),
                    seed=(
                        int(config.seed)
                        + 20_000
                        + 100 * int(step_index)
                        + 10 * regime_index
                        + method_index
                    ),
                ),
                "sigma_change_rms": _rms(full_delta),
                "sigma_increase_fraction": float(
                    (full_delta > 0.0).double().mean().item()
                ),
                "first_pair_half_sigma_change_mean": float(
                    first_delta.double().mean().item()
                ),
                "second_pair_half_sigma_change_mean": float(
                    second_delta.double().mean().item()
                ),
                "pair_half_direction_agreement": bool(direction_product > 0.0),
                "target_batch_sigma_change_means": target_delta_means,
                "target_batch_direction_agreement": bool(
                    min(target_delta_means) * max(target_delta_means) > 0.0
                ),
            }
            artifacts[f"{regime_name}_{method}_sigma"] = full_sigma.detach().cpu()
        anti_change = float(methods["antithetic"]["sigma_change"]["estimate"])
        iid_change = float(methods["iid"]["sigma_change"]["estimate"])
        report[regime_name] = {
            "count": count,
            "current_sigma_mean": float(current_sigma[mask].double().mean().item()),
            "antithetic_to_iid_future_variance_ratio": variance_ratio,
            "methods": methods,
            "estimator_sigma_change_difference": _bootstrap_mean(
                (
                    sigma_values["antithetic"][0][mask]
                    - sigma_values["iid"][0][mask]
                ),
                replicates=int(config.bootstrap_replicates),
                confidence=float(config.bootstrap_confidence),
                seed=int(config.seed) + 30_000 + 100 * int(step_index) + regime_index,
            ),
            "estimator_direction_agreement": bool(anti_change * iid_change > 0.0),
        }
    return report, artifacts


def _future_noise(
    *,
    model: DiscreteMPModel,
    base_noise: torch.Tensor,
    prefix_count: int,
    pair_count: int,
    step_index: int,
    seed: int,
) -> torch.Tensor:
    values = sample_standard_normals(
        grid=model.grid,
        batch_size=prefix_count * pair_count,
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=int(seed),
    ).reshape(
        prefix_count,
        pair_count,
        int(model.grid.num_steps),
        int(model.architecture.noise_dim),
    )
    if int(step_index) > 0:
        values[:, :, : int(step_index), :] = base_noise[
            :, None, : int(step_index), :
        ]
    return values


def audit_antithetic_r_estimator(
    *,
    model: DiscreteMPModel,
    target_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig | None = None,
    config: AntitheticRAuditConfig | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> AntitheticRAuditResult:
    """Compare equal-cost antithetic and IID complete-MP P/R estimators."""

    diagnostic = config or AntitheticRAuditConfig()
    training_config = training or model.training
    target_count, _ = model._validate_target_paths(target_paths)
    target_paths_used = int(diagnostic.num_target_batches) * int(
        diagnostic.target_batch_size
    )
    if target_paths_used > target_count:
        raise ValueError("disjoint target batches cannot exceed the target path pool")
    if max(diagnostic.step_indices) >= int(model.grid.num_steps):
        raise ValueError("step_indices must lie before the final grid point")
    target_device = target_paths.to(device=model.device, dtype=model.dtype)
    cpu_generator = torch.Generator(device="cpu").manual_seed(int(diagnostic.seed))
    target_permutation = torch.randperm(target_count, generator=cpu_generator)[
        :target_paths_used
    ]
    target_batches = [
        target_device.index_select(
            0,
            target_permutation[
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
        seed=derive_stream_seed(
            base_seed=int(diagnostic.seed),
            stream_offset=1,
        ),
    )
    artifacts: dict[str, object] = {
        "target_permutation": target_permutation.detach().cpu(),
        "source_indices": source_indices.detach().cpu(),
    }
    step_reports: dict[str, object] = {}

    for step_number, step_index in enumerate(diagnostic.step_indices):
        prefix_count = int(diagnostic.num_prefixes)
        pair_count = int(diagnostic.num_pairs)
        plus_noise = _future_noise(
            model=model,
            base_noise=base_noise,
            prefix_count=prefix_count,
            pair_count=pair_count,
            step_index=int(step_index),
            seed=int(
                derive_stream_seed(
                    base_seed=int(diagnostic.seed),
                    stream_offset=100 + 10 * step_number,
                )
            ),
        )
        minus_noise = plus_noise.clone()
        minus_noise[:, :, int(step_index) :, :] = -minus_noise[
            :, :, int(step_index) :, :
        ]
        iid_first_noise = _future_noise(
            model=model,
            base_noise=base_noise,
            prefix_count=prefix_count,
            pair_count=pair_count,
            step_index=int(step_index),
            seed=int(
                derive_stream_seed(
                    base_seed=int(diagnostic.seed),
                    stream_offset=101 + 10 * step_number,
                )
            ),
        )
        iid_second_noise = _future_noise(
            model=model,
            base_noise=base_noise,
            prefix_count=prefix_count,
            pair_count=pair_count,
            step_index=int(step_index),
            seed=int(
                derive_stream_seed(
                    base_seed=int(diagnostic.seed),
                    stream_offset=102 + 10 * step_number,
                )
            ),
        )
        noise_values = {
            "plus": plus_noise,
            "minus": minus_noise,
            "iid_first": iid_first_noise,
            "iid_second": iid_second_noise,
        }
        branch_rollouts = {}
        flat_x0 = x0[:, None, :].expand(
            prefix_count,
            pair_count,
            int(x0.shape[-1]),
        ).reshape(prefix_count * pair_count, int(x0.shape[-1]))
        for name in BRANCH_NAMES:
            flat_noise = noise_values[name].reshape(
                prefix_count * pair_count,
                int(model.grid.num_steps),
                int(model.architecture.noise_dim),
            )
            with torch.no_grad():
                rollout = model.rollout(x0=flat_x0, noise=flat_noise)
            branch_rollouts[name] = {
                "paths": rollout.paths.reshape(
                    prefix_count,
                    pair_count,
                    int(model.grid.num_steps) + 1,
                    int(model.architecture.state_dim),
                ),
                "controls": rollout.controls.reshape(
                    prefix_count,
                    pair_count,
                    int(model.grid.num_steps),
                    -1,
                ),
                "predicted_p": rollout.expected_adjoint_next.reshape(
                    prefix_count,
                    pair_count,
                    int(model.grid.num_steps),
                    -1,
                ),
                "predicted_r": rollout.expected_adjoint_noise_next.reshape(
                    prefix_count,
                    pair_count,
                    int(model.grid.num_steps),
                    -1,
                ),
            }
        prefix = branch_rollouts["plus"]["paths"][
            :, 0, : int(step_index) + 1, :
        ]
        prefix_difference = 0.0
        for name in BRANCH_NAMES:
            prefix_difference = max(
                prefix_difference,
                float(
                    (
                        branch_rollouts[name]["paths"][
                            :, :, : int(step_index) + 1, :
                        ]
                        - prefix[:, None, :, :]
                    )
                    .abs()
                    .max()
                    .item()
                ),
            )
        branch_p_lists: dict[str, list[torch.Tensor]] = {
            name: [] for name in BRANCH_NAMES
        }
        branch_r_lists: dict[str, list[torch.Tensor]] = {
            name: [] for name in BRANCH_NAMES
        }
        for pair_index in range(pair_count):
            for name in BRANCH_NAMES:
                target_p = []
                target_r = []
                branch_paths = branch_rollouts[name]["paths"][:, pair_index, :, :]
                branch_controls = branch_rollouts[name]["controls"][
                    :, pair_index, :, :
                ]
                branch_noise = noise_values[name][:, pair_index, :, :]
                for target_batch in target_batches:
                    p_all, _ = model._path_functional_adjoint_targets(
                        generated_path=branch_paths,
                        controls=branch_controls,
                        target_path=target_batch,
                        training=training_config,
                        include_running_cost=True,
                    )
                    r_all = model._adjoint_noise_target(
                        pathwise_adjoint_target=p_all,
                        moments_noise_target_dim=int(
                            branch_rollouts[name]["predicted_r"].shape[-1]
                        ),
                        noise=branch_noise,
                    )
                    target_p.append(p_all[:, int(step_index), :])
                    target_r.append(r_all[:, int(step_index), :])
                branch_p_lists[name].append(torch.stack(target_p, dim=1))
                branch_r_lists[name].append(torch.stack(target_r, dim=1))
            if progress_callback is not None and (
                pair_index == 0
                or (pair_index + 1) % max(1, pair_count // 8) == 0
                or pair_index + 1 == pair_count
            ):
                progress_callback(int(step_index), pair_index + 1, pair_count)
        branch_p = {
            name: torch.stack(branch_p_lists[name], dim=1)
            for name in BRANCH_NAMES
        }
        branch_r = {
            name: torch.stack(branch_r_lists[name], dim=1)
            for name in BRANCH_NAMES
        }
        estimator_p = {
            "antithetic": 0.5 * (branch_p["plus"] + branch_p["minus"]),
            "iid": 0.5 * (branch_p["iid_first"] + branch_p["iid_second"]),
        }
        estimator_r = {
            "antithetic": 0.5 * (branch_r["plus"] + branch_r["minus"]),
            "iid": 0.5 * (branch_r["iid_first"] + branch_r["iid_second"]),
        }
        estimator_metrics = {}
        estimator_tensors = {}
        for method in ESTIMATOR_NAMES:
            p_metrics, p_tensors = _pair_estimator_metrics(estimator_p[method])
            r_metrics, r_tensors = _pair_estimator_metrics(estimator_r[method])
            estimator_metrics[method] = {"p": p_metrics, "r": r_metrics}
            estimator_tensors[method] = {"p": p_tensors, "r": r_tensors}

        ant_variance = estimator_tensors["antithetic"]["r"][
            "per_prefix_future_variance"
        ]
        iid_variance = estimator_tensors["iid"]["r"][
            "per_prefix_future_variance"
        ]
        variance_ratio = _bootstrap_ratio(
            ant_variance,
            iid_variance,
            replicates=int(diagnostic.bootstrap_replicates),
            confidence=float(diagnostic.bootstrap_confidence),
            seed=int(diagnostic.seed) + 40_000 + int(step_index),
        )
        ant_mean = estimator_tensors["antithetic"]["r"]["conditional_mean"]
        iid_mean = estimator_tensors["iid"]["r"]["conditional_mean"]
        pair_comparison = {
            "antithetic_to_iid_future_variance_ratio": variance_ratio,
            "iid_equivalent_path_multiplier": (
                1.0 / max(float(variance_ratio["estimate"]), torch.finfo(torch.float64).eps)
            ),
            "conditional_mean_agreement": {
                "correlation": _correlation(ant_mean, iid_mean),
                "relative_rmse": _relative_rmse(ant_mean, iid_mean),
                "difference_rms": _rms(ant_mean - iid_mean),
            },
            "branch_residual_correlations": {
                "plus_vs_minus": _residual_correlation(
                    branch_r["plus"],
                    branch_r["minus"],
                ),
                "iid_first_vs_second": _residual_correlation(
                    branch_r["iid_first"],
                    branch_r["iid_second"],
                ),
            },
            "pair_half_correlation_improvement": (
                float(
                    estimator_metrics["antithetic"]["r"][
                        "pair_half_teacher_agreement"
                    ]["correlation"]
                    or 0.0
                )
                - float(
                    estimator_metrics["iid"]["r"]["pair_half_teacher_agreement"][
                        "correlation"
                    ]
                    or 0.0
                )
            ),
        }
        regimes, thresholds, prefix_volatility = prefix_volatility_regimes(
            prefix,
            target_device,
            split_step=int(diagnostic.regime_split_step),
            prefix_window=int(diagnostic.regime_prefix_window),
        )
        current_p = branch_rollouts["plus"]["predicted_p"][
            :, 0, int(step_index), :
        ]
        current_r = branch_rollouts["plus"]["predicted_r"][
            :, 0, int(step_index), :
        ]
        regime_report, sigma_artifacts = _regime_report(
            model=model,
            step_index=int(step_index),
            prefixes=prefix,
            regimes=regimes,
            current_p=current_p,
            current_r=current_r,
            estimator_p=estimator_p,
            estimator_r=estimator_r,
            per_prefix_variances={
                "antithetic": ant_variance.to(model.device),
                "iid": iid_variance.to(model.device),
            },
            config=diagnostic,
        )
        step_reports[str(int(step_index))] = {
            "step_index": int(step_index),
            "prefix_length": int(step_index) + 1,
            "prefix_max_abs_difference": prefix_difference,
            "regime_thresholds": [
                float(thresholds[0].item()),
                float(thresholds[1].item()),
            ],
            "estimators": estimator_metrics,
            "comparison": pair_comparison,
            "regimes": regime_report,
        }
        artifacts[f"step_{step_index}_prefix_paths"] = prefix.detach().cpu()
        artifacts[f"step_{step_index}_prefix_volatility"] = (
            prefix_volatility.detach().cpu()
        )
        artifacts[f"step_{step_index}_regime_indices"] = regimes.detach().cpu()
        for name in BRANCH_NAMES:
            artifacts[f"step_{step_index}_{name}_p"] = branch_p[name].detach().cpu()
            artifacts[f"step_{step_index}_{name}_r"] = branch_r[name].detach().cpu()
        for method in ESTIMATOR_NAMES:
            artifacts[f"step_{step_index}_{method}_p"] = estimator_p[method].detach().cpu()
            artifacts[f"step_{step_index}_{method}_r"] = estimator_r[method].detach().cpu()
        for name, tensor in sigma_artifacts.items():
            artifacts[f"step_{step_index}_{name}"] = tensor

    primary = step_reports[str(int(diagnostic.regime_split_step))]
    primary_comparison = primary["comparison"]
    primary_antithetic = primary["estimators"]["antithetic"]["r"]
    primary_iid = primary["estimators"]["iid"]["r"]
    primary_low = primary["regimes"]["low"]
    variance_upper = float(
        primary_comparison["antithetic_to_iid_future_variance_ratio"]["ci_upper"]
    )
    agreement_improvement = float(
        primary_comparison["pair_half_correlation_improvement"]
    )
    mean_correlation = primary_comparison["conditional_mean_agreement"]["correlation"]
    low_direction_preserved = bool(
        int(primary_low.get("count", 0)) >= 2
        and primary_low["estimator_direction_agreement"]
        and primary_low["methods"]["antithetic"]["pair_half_direction_agreement"]
    )
    report: dict[str, object] = {
        "protocol": {
            "selected_generator_frozen": True,
            "equal_compute_two_path_estimators": True,
            "antithetic_future_noise_is_exact_negative_after_prefix": True,
            "iid_comparator_uses_two_independent_future_paths": True,
            "common_prefixes_and_target_batches": True,
            "complete_path_dependent_adjoint_includes_running_cost": True,
            "antithetic_estimator_unbiased_by_gaussian_symmetry": True,
            "maximum_principle_running_cost_discrepancy_and_neural_loss_unchanged": True,
            "no_network_training": True,
        },
        "config": {
            **diagnostic.__dict__,
            "target_pool_size": target_count,
            "target_paths_used": target_paths_used,
            "paths_per_estimator_pair": 2,
            "future_paths_per_method_and_prefix": 2 * int(diagnostic.num_pairs),
            "device": str(model.device),
            "dtype": str(model.dtype),
        },
        "steps": step_reports,
        "decision": {
            "primary_step": int(diagnostic.regime_split_step),
            "material_variance_reduction": bool(
                variance_upper
                < float(diagnostic.maximum_material_variance_ratio)
            ),
            "pair_half_agreement_improved": bool(
                agreement_improvement
                >= float(diagnostic.minimum_agreement_improvement)
            ),
            "conditional_mean_empirically_compatible": bool(
                mean_correlation is not None
                and float(mean_correlation)
                >= float(diagnostic.minimum_conditional_mean_correlation)
            ),
            "low_regime_sigma_direction_preserved": low_direction_preserved,
            "antithetic_training_pilot_supported": bool(
                variance_upper
                < float(diagnostic.maximum_material_variance_ratio)
                and agreement_improvement
                >= float(diagnostic.minimum_agreement_improvement)
                and mean_correlation is not None
                and float(mean_correlation)
                >= float(diagnostic.minimum_conditional_mean_correlation)
                and low_direction_preserved
            ),
            "primary_antithetic_pair_half_correlation": primary_antithetic[
                "pair_half_teacher_agreement"
            ]["correlation"],
            "primary_iid_pair_half_correlation": primary_iid[
                "pair_half_teacher_agreement"
            ]["correlation"],
        },
    }
    return AntitheticRAuditResult(report=report, artifacts=artifacts)


__all__ = [
    "AntitheticRAuditConfig",
    "AntitheticRAuditResult",
    "BRANCH_NAMES",
    "ESTIMATOR_NAMES",
    "audit_antithetic_r_estimator",
]
