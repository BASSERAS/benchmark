from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.discrepancies import (
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals


def _reference_rollout(
    *,
    model: DiscreteMPModel,
    x0: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    reference = getattr(model.control_map, "reference_kernel", None)
    if reference is None or not hasattr(reference, "evaluate"):
        raise ValueError("control_map must expose a reference_kernel with evaluate")
    points = [x0]
    sqrt_dt = math.sqrt(float(model.grid.dt))
    with torch.no_grad():
        for step in range(int(model.grid.num_steps)):
            prefix = torch.stack(points, dim=1)
            drift, sigma = reference.evaluate(prefix, step_index=step)
            points.append(
                points[-1]
                + float(model.grid.dt) * drift
                + sqrt_dt * sigma * noise[:, step, :]
            )
    return torch.stack(points, dim=1)


def adjoint_rms_balanced_weights(
    *,
    blocks: Sequence[WeightedDiscrepancy],
    combined_projected_target_rms: Sequence[float],
    total_target_rms: float = 1.0,
    inactive_tolerance: float = 1e-10,
) -> tuple[tuple[WeightedDiscrepancy, ...], dict[str, object]]:
    """Rescale blocks so their projected MP targets have equal RMS budget.

    The total budget is divided by ``sqrt(K)`` over active blocks.  Thus
    orthogonal block signals have total RMS ``total_target_rms`` while fully
    aligned blocks are conservatively bounded by ``sqrt(K)`` times that value.
    A block with zero local derivative is disabled rather than assigned an
    unbounded inverse scale.
    """

    block_values = tuple(blocks)
    scales = tuple(float(value) for value in combined_projected_target_rms)
    if len(block_values) == 0 or len(block_values) != len(scales):
        raise ValueError("blocks and combined_projected_target_rms must be non-empty and aligned")
    total = float(total_target_rms)
    tolerance = float(inactive_tolerance)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("total_target_rms must be a positive finite float")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("inactive_tolerance must be a nonnegative finite float")
    if any(not math.isfinite(value) or value < 0.0 for value in scales):
        raise ValueError("combined projected target RMS values must be finite and nonnegative")
    active_count = sum(value > tolerance for value in scales)
    if active_count == 0:
        raise ValueError("at least one discrepancy block must have nonzero adjoint signal")
    per_block_target = total / math.sqrt(float(active_count))
    normalized: list[WeightedDiscrepancy] = []
    diagnostics: dict[str, object] = {
        "rule": "equal projected P/R RMS with fixed total train-only budget",
        "total_target_rms": total,
        "active_block_count": active_count,
        "per_block_target_rms": per_block_target,
        "blocks": {},
    }
    block_diagnostics: dict[str, object] = {}
    for block, scale in zip(block_values, scales):
        active = scale > tolerance
        multiplier = per_block_target / scale if active else 0.0
        normalized_weight = float(block.weight) * multiplier
        normalized.append(
            WeightedDiscrepancy(
                name=str(block.name),
                weight=normalized_weight,
                discrepancy=block.discrepancy,
            )
        )
        block_diagnostics[str(block.name)] = {
            "active": active,
            "original_weight": float(block.weight),
            "combined_projected_target_rms": scale,
            "normalization_multiplier": multiplier,
            "normalized_weight": normalized_weight,
        }
    diagnostics["blocks"] = block_diagnostics
    return tuple(normalized), diagnostics


def fit_train_only_adjoint_rms_normalized_discrepancy(
    *,
    model: DiscreteMPModel,
    discrepancy: CompositePathFunctionalDiscrepancy,
    target_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    calibration_batches: int = 4,
    calibration_batch_size: int = 256,
    calibration_seed: int = 811_903,
    total_target_rms: float = 1.0,
    inactive_tolerance: float = 1e-10,
) -> tuple[CompositePathFunctionalDiscrepancy, dict[str, object]]:
    """Fit immutable discrepancy weights using training paths and reference rollouts only."""

    if int(calibration_batches) < 1 or int(calibration_batch_size) < 2:
        raise ValueError("calibration_batches must be positive and batch size at least two")
    if target_paths.ndim != 3 or int(target_paths.shape[1]) != int(model.grid.num_steps) + 1:
        raise ValueError("target_paths must align with model.grid")
    target = target_paths.to(device=model.device, dtype=model.dtype)
    blocks = tuple(discrepancy.blocks)
    p_second_moments = torch.zeros(len(blocks), dtype=torch.float64)
    r_second_moments = torch.zeros(len(blocks), dtype=torch.float64)
    cpu_generator = torch.Generator(device="cpu").manual_seed(int(calibration_seed))
    for batch_index in range(int(calibration_batches)):
        source_indices = model._sample_indices(
            size=int(target.shape[0]),
            count=int(calibration_batch_size),
            generator=cpu_generator,
        )
        target_indices = model._sample_indices(
            size=int(target.shape[0]),
            count=int(calibration_batch_size),
            generator=cpu_generator,
        )
        x0 = target.index_select(0, source_indices.to(model.device))[:, 0, :]
        target_batch = target.index_select(0, target_indices.to(model.device))
        noise = sample_standard_normals(
            grid=model.grid,
            batch_size=int(calibration_batch_size),
            noise_dim=int(model.architecture.noise_dim),
            device=model.device,
            dtype=model.dtype,
            seed=derive_stream_seed(
                base_seed=int(calibration_seed),
                stream_offset=batch_index + 1,
            ),
        )
        generated = _reference_rollout(model=model, x0=x0, noise=noise)
        for block_index, block in enumerate(blocks):
            single = CompositePathFunctionalDiscrepancy(blocks=(block,))
            raw_p, _ = model._path_functional_adjoint_targets(
                generated_path=generated,
                target_path=target_batch,
                training=training,
                discrepancy=single,
                include_running_cost=False,
            )
            raw_r = model._adjoint_noise_target(
                pathwise_adjoint_target=raw_p,
                moments_noise_target_dim=int(
                    model._default_noise_adjoint_dim(model.architecture)
                ),
                noise=noise,
            )
            with torch.no_grad():
                projected_p, _ = model._project_pathwise_target(
                    frozen_generated_path=generated,
                    pathwise_target=raw_p,
                    training=training,
                )
                projected_r, _ = model._project_pathwise_target(
                    frozen_generated_path=generated,
                    pathwise_target=raw_r,
                    training=training,
                    noise_adjoint=True,
                )
            p_second_moments[block_index] += float(
                projected_p.detach().double().pow(2).mean().item()
            )
            r_second_moments[block_index] += float(
                projected_r.detach().double().pow(2).mean().item()
            )
    denominator = float(calibration_batches)
    combined_scales = torch.sqrt(
        p_second_moments / denominator + r_second_moments / denominator
    )
    normalized_blocks, diagnostics = adjoint_rms_balanced_weights(
        blocks=blocks,
        combined_projected_target_rms=tuple(float(value) for value in combined_scales.tolist()),
        total_target_rms=float(total_target_rms),
        inactive_tolerance=float(inactive_tolerance),
    )
    diagnostics.update(
        {
            "selection_uses_training_paths_only": True,
            "rollout_law": "fitted_reference",
            "calibration_batches": int(calibration_batches),
            "calibration_batch_size": int(calibration_batch_size),
            "calibration_seed": int(calibration_seed),
            "lambda_scale_included": float(training.lambda_scale),
        }
    )
    return CompositePathFunctionalDiscrepancy(blocks=normalized_blocks), diagnostics


__all__ = [
    "adjoint_rms_balanced_weights",
    "fit_train_only_adjoint_rms_normalized_discrepancy",
    "fit_train_only_family_adjoint_rms_normalized_discrepancy",
    "restore_family_adjoint_rms_normalized_discrepancy",
]


def restore_family_adjoint_rms_normalized_discrepancy(
    *,
    discrepancy: CompositePathFunctionalDiscrepancy,
    normalization: Mapping[str, Any],
) -> tuple[CompositePathFunctionalDiscrepancy, dict[str, object]]:
    """Restore frozen family-normalized weights from a prior manifest.

    This is used for controlled estimator ablations where recomputing weights
    with a different conditional-expectation estimator would also change the
    objective.  Block implementations and ordering must match exactly.
    """

    families = normalization.get("families")
    if not isinstance(families, Mapping):
        raise ValueError("normalization is missing family diagnostics")
    weights: dict[str, float] = {}
    for family_name, family in families.items():
        if not isinstance(family, Mapping):
            raise ValueError(f"normalization family {family_name} must be a mapping")
        blocks = family.get("blocks")
        if not isinstance(blocks, Mapping):
            raise ValueError(f"normalization family {family_name} is missing blocks")
        for block_name, block in blocks.items():
            if not isinstance(block, Mapping) or "final_family_scaled_weight" not in block:
                raise ValueError(f"normalization block {block_name} is missing its final weight")
            weight = float(block["final_family_scaled_weight"])
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"normalization block {block_name} has an invalid final weight")
            if str(block_name) in weights:
                raise ValueError(f"normalization block {block_name} appears in multiple families")
            weights[str(block_name)] = weight

    original = {str(block.name): block for block in discrepancy.blocks}
    if set(original) != set(weights):
        missing = sorted(set(original) - set(weights))
        extra = sorted(set(weights) - set(original))
        raise ValueError(
            f"normalization blocks do not match discrepancy; missing={missing}, extra={extra}"
        )
    restored = CompositePathFunctionalDiscrepancy(
        blocks=tuple(
            WeightedDiscrepancy(
                name=name,
                weight=weights[name],
                discrepancy=original[name].discrepancy,
            )
            for name in original
        )
    )
    diagnostics = dict(normalization)
    diagnostics.update(
        {
            "restored_from_frozen_manifest": True,
            "restored_block_count": len(weights),
        }
    )
    return restored, diagnostics


def fit_train_only_family_adjoint_rms_normalized_discrepancy(
    *,
    model: DiscreteMPModel,
    discrepancy: CompositePathFunctionalDiscrepancy,
    target_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    family_by_block: dict[str, str],
    family_coefficients: dict[str, float],
    calibration_batches: int = 4,
    calibration_batch_size: int = 256,
    calibration_seed: int = 811_903,
    family_target_rms: float = 1.0,
    inactive_tolerance: float = 1e-10,
) -> tuple[CompositePathFunctionalDiscrepancy, dict[str, object]]:
    """Normalize blocks within named families, then apply family coefficients.

    Relative weights are fitted with training paths and reference rollouts only.
    ``family_coefficients`` therefore control the price-law and volatility-law
    adjoint budgets without changing their internally balanced block ratios.
    """

    blocks = tuple(discrepancy.blocks)
    block_names = tuple(str(block.name) for block in blocks)
    missing = [name for name in block_names if name not in family_by_block]
    if missing:
        raise ValueError(f"family_by_block is missing discrepancy blocks: {missing}")
    extra = [name for name in family_by_block if name not in set(block_names)]
    if extra:
        raise ValueError(f"family_by_block contains unknown discrepancy blocks: {extra}")
    families = tuple(dict.fromkeys(family_by_block[name] for name in block_names))
    if set(families) != set(family_coefficients):
        raise ValueError("family_coefficients must define exactly the named block families")
    coefficients = {name: float(value) for name, value in family_coefficients.items()}
    if any(not math.isfinite(value) or value <= 0.0 for value in coefficients.values()):
        raise ValueError("family coefficients must be positive finite floats")

    _, pilot = fit_train_only_adjoint_rms_normalized_discrepancy(
        model=model,
        discrepancy=discrepancy,
        target_paths=target_paths,
        training=training,
        calibration_batches=int(calibration_batches),
        calibration_batch_size=int(calibration_batch_size),
        calibration_seed=int(calibration_seed),
        total_target_rms=1.0,
        inactive_tolerance=float(inactive_tolerance),
    )
    pilot_blocks = pilot["blocks"]
    normalized_by_name: dict[str, WeightedDiscrepancy] = {}
    family_diagnostics: dict[str, object] = {}
    for family in families:
        family_blocks = tuple(
            block for block in blocks if family_by_block[str(block.name)] == family
        )
        family_scales = tuple(
            float(pilot_blocks[str(block.name)]["combined_projected_target_rms"])
            for block in family_blocks
        )
        normalized_family, diagnostics = adjoint_rms_balanced_weights(
            blocks=family_blocks,
            combined_projected_target_rms=family_scales,
            total_target_rms=float(family_target_rms),
            inactive_tolerance=float(inactive_tolerance),
        )
        coefficient = coefficients[family]
        final_family = tuple(
            WeightedDiscrepancy(
                name=str(block.name),
                weight=coefficient * float(block.weight),
                discrepancy=block.discrepancy,
            )
            for block in normalized_family
        )
        normalized_by_name.update({str(block.name): block for block in final_family})
        diagnostics["family_coefficient"] = coefficient
        diagnostics["family_target_rms_before_coefficient"] = float(family_target_rms)
        diagnostics["family_target_rms_after_coefficient"] = coefficient * float(
            family_target_rms
        )
        for block in final_family:
            diagnostics["blocks"][str(block.name)]["final_family_scaled_weight"] = float(
                block.weight
            )
        family_diagnostics[family] = diagnostics

    ordered = tuple(normalized_by_name[name] for name in block_names)
    output_diagnostics: dict[str, object] = {
        "rule": "separate train-only projected-adjoint RMS normalization within each family",
        "selection_uses_training_paths_only": True,
        "rollout_law": "fitted_reference",
        "calibration_batches": int(calibration_batches),
        "calibration_batch_size": int(calibration_batch_size),
        "calibration_seed": int(calibration_seed),
        "lambda_scale_included": float(training.lambda_scale),
        "family_by_block": dict(family_by_block),
        "family_coefficients": coefficients,
        "families": family_diagnostics,
    }
    return CompositePathFunctionalDiscrepancy(blocks=ordered), output_diagnostics
