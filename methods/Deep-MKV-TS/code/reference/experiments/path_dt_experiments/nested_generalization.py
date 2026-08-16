from __future__ import annotations

from dataclasses import asdict, replace
import math
from typing import Callable, Mapping

import torch

from deep_mkv_gen_path_dt import (
    DiscreteMPNestedTrainingConfig,
    DiscreteMPTrainingConfig,
)
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals


DEFAULT_BLOCK_RELAXATIONS = tuple(
    sorted({0.0, *(0.5**index for index in range(11))})
)
PARAMETER_BLOCKS = ("p", "r", "shared")
COORDINATE_ORDER = ("r", "p", "shared")


def parameter_block(name: str) -> str:
    """Map a recurrent-network parameter name to its outer relaxation block."""

    if name.startswith("expected_adjoint_next_head."):
        return "p"
    if name.startswith("expected_adjoint_noise_next_head."):
        return "r"
    return "shared"


def apply_parameter_block_interpolation(
    model: DiscreteMPModel,
    *,
    old_values: Mapping[str, torch.Tensor],
    candidate_values: Mapping[str, torch.Tensor],
    rates: Mapping[str, float],
) -> None:
    """Apply groupwise interpolation between two network parameter states."""

    if set(rates) != set(PARAMETER_BLOCKS):
        raise ValueError("rates must contain exactly p, r, and shared")
    with torch.no_grad():
        for name, parameter in model.network.named_parameters():
            if name not in old_values or name not in candidate_values:
                raise ValueError(f"missing interpolation value for parameter {name}")
            rate = float(rates[parameter_block(name)])
            if not math.isfinite(rate):
                raise ValueError("parameter interpolation rates must be finite")
            old = old_values[name].to(device=parameter.device, dtype=parameter.dtype)
            candidate = candidate_values[name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.copy_(old + rate * (candidate - old))


def select_block_coordinate_rates(
    *,
    relaxations: tuple[float, ...],
    evaluate: Callable[[dict[str, float]], float],
    passes: int = 2,
) -> tuple[dict[str, float], float, int]:
    """Minimize an objective over P/R/shared rates by coordinate enumeration."""

    if isinstance(passes, bool) or int(passes) != passes or int(passes) < 1:
        raise ValueError("passes must be a positive integer")
    candidates = tuple(sorted(set(float(value) for value in relaxations)))
    if len(candidates) != len(relaxations):
        raise ValueError("relaxations must not contain duplicates")
    if 0.0 not in candidates:
        raise ValueError("relaxations must contain zero")
    if any(not math.isfinite(value) or value < 0.0 for value in candidates):
        raise ValueError("relaxations must be finite and non-negative")

    rates = {block: 0.0 for block in PARAMETER_BLOCKS}
    objective = float(evaluate(dict(rates)))
    evaluations = 1
    if not math.isfinite(objective):
        raise ValueError("the zero-relaxation objective must be finite")
    epsilon = max(torch.finfo(torch.float64).eps, 1e-12 * max(1.0, abs(objective)))
    completed_passes = 0
    for coordinate_pass in range(int(passes)):
        changed = False
        for block in COORDINATE_ORDER:
            best_rate = float(rates[block])
            best_objective = objective
            for rate in candidates:
                if rate == float(rates[block]):
                    continue
                trial = dict(rates)
                trial[block] = float(rate)
                trial_objective = float(evaluate(trial))
                evaluations += 1
                if (
                    math.isfinite(trial_objective)
                    and trial_objective < best_objective - epsilon
                ):
                    best_rate = float(rate)
                    best_objective = trial_objective
            if best_rate != float(rates[block]):
                changed = True
            rates[block] = best_rate
            objective = best_objective
        completed_passes = coordinate_pass + 1
        if not changed:
            break
    return rates, objective, completed_passes


def _network_parameter_values(model: DiscreteMPModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.network.named_parameters()
    }


def _objective_batch(
    *,
    model: DiscreteMPModel,
    paths: torch.Tensor,
    source_count: int,
    target_count: int,
    index_seed: int | None,
    noise_seed: int | None,
) -> dict[str, torch.Tensor | int]:
    paths_device = paths.to(device=model.device, dtype=model.dtype)
    total, _ = model._validate_target_paths(paths_device)
    generator = torch.Generator(device="cpu")
    if index_seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(index_seed))
    source_indices = model._sample_indices(
        size=total,
        count=int(source_count),
        generator=generator,
    )
    target_indices = model._sample_indices(
        size=total,
        count=int(target_count),
        generator=generator,
    )
    x0 = paths_device.index_select(
        0,
        source_indices.to(paths_device.device),
    )[:, 0, :]
    target_batch = paths_device.index_select(
        0,
        target_indices.to(paths_device.device),
    )
    noise = sample_standard_normals(
        grid=model.grid,
        batch_size=int(source_count),
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=noise_seed,
    )
    return {
        "x0": x0,
        "target": target_batch,
        "noise": noise,
        "index_seed": -1 if index_seed is None else int(index_seed),
        "noise_seed": -1 if noise_seed is None else int(noise_seed),
    }


def diagnose_nested_objective_generalization(
    *,
    model: DiscreteMPModel,
    target_paths: torch.Tensor,
    heldout_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    nested: DiscreteMPNestedTrainingConfig,
    relaxations: tuple[float, ...] = DEFAULT_BLOCK_RELAXATIONS,
    coordinate_passes: int = 2,
    heldout_seed_offset: int = 700_000,
    selection_scales: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75, 1.0),
) -> tuple[dict[str, object], dict[str, object]]:
    """Compare one frozen backward candidate on fitting and independent banks.

    The candidate is fitted once on ``target_paths``.  Its joint and blockwise
    parameter relaxations are then evaluated with common random numbers within
    each of two objective banks.  The held-out bank uses an independent path
    pool, initial-state/target indices, and noise stream.
    """

    if int(nested.outer_steps) != 1:
        raise ValueError("generalization diagnostic requires nested.outer_steps=1")
    rates_used = tuple(sorted(set(float(value) for value in relaxations)))
    if len(rates_used) != len(relaxations):
        raise ValueError("relaxations must not contain duplicates")
    if 0.0 not in rates_used or 1.0 not in rates_used:
        raise ValueError("relaxations must contain zero and one")
    if any(not math.isfinite(value) or value < 0.0 for value in rates_used):
        raise ValueError("relaxations must be finite and non-negative")
    if isinstance(heldout_seed_offset, bool) or int(heldout_seed_offset) != heldout_seed_offset:
        raise ValueError("heldout_seed_offset must be an integer")
    scales_used = tuple(sorted(set(float(value) for value in selection_scales)))
    if len(scales_used) != len(selection_scales):
        raise ValueError("selection_scales must not contain duplicates")
    if any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in scales_used
    ):
        raise ValueError("selection_scales must lie in (0, 1]")

    old_checkpoint = model.checkpoint_state()
    old_values = _network_parameter_values(model)
    fit_result = model.fit_nested(
        target_paths,
        training=training,
        nested=replace(nested, outer_objective_line_search=False),
    )
    candidate_checkpoint = model.checkpoint_state()
    candidate_values = _network_parameter_values(model)

    training_bank = _objective_batch(
        model=model,
        paths=target_paths,
        source_count=int(nested.outer_batch_size),
        target_count=int(nested.outer_target_batch_size),
        index_seed=nested.seed,
        noise_seed=derive_stream_seed(
            base_seed=nested.seed,
            stream_offset=200_000,
        ),
    )
    heldout_seed = derive_stream_seed(
        base_seed=nested.seed,
        stream_offset=int(heldout_seed_offset),
    )
    heldout_bank = _objective_batch(
        model=model,
        paths=heldout_paths,
        source_count=int(nested.outer_batch_size),
        target_count=int(nested.outer_target_batch_size),
        index_seed=heldout_seed,
        noise_seed=derive_stream_seed(base_seed=heldout_seed, stream_offset=1),
    )
    banks = {"training": training_bank, "heldout": heldout_bank}

    baseline_paths: dict[str, torch.Tensor] = {}
    zero_rates = {block: 0.0 for block in PARAMETER_BLOCKS}
    apply_parameter_block_interpolation(
        model,
        old_values=old_values,
        candidate_values=candidate_values,
        rates=zero_rates,
    )
    with torch.no_grad():
        for bank_name, bank in banks.items():
            baseline_paths[bank_name] = model.rollout(
                x0=bank["x0"],
                noise=bank["noise"],
            ).paths.detach().clone()

    cache: dict[tuple[str, float, float, float], dict[str, float]] = {}

    def evaluate(bank_name: str, rates: Mapping[str, float]) -> dict[str, float]:
        key = (
            bank_name,
            float(rates["p"]),
            float(rates["r"]),
            float(rates["shared"]),
        )
        if key in cache:
            return dict(cache[key])
        apply_parameter_block_interpolation(
            model,
            old_values=old_values,
            candidate_values=candidate_values,
            rates=rates,
        )
        bank = banks[bank_name]
        with torch.no_grad():
            rollout = model.rollout(x0=bank["x0"], noise=bank["noise"])
            finite = bool(
                torch.isfinite(rollout.paths).all().item()
                and torch.isfinite(rollout.controls).all().item()
            )
            if finite:
                _, objective_metrics = model._complete_objective_metrics(
                    generated_path=rollout.paths,
                    controls=rollout.controls,
                    target_path=bank["target"],
                    training=training,
                )
                control_metrics = model._control_diagnostics(rollout.controls)
                row = {
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
                        objective_metrics.get("running_cost_drift_value", float("nan"))
                    ),
                    "running_cost_volatility_value": float(
                        objective_metrics.get(
                            "running_cost_volatility_value",
                            float("nan"),
                        )
                    ),
                    "path_rms_change": float(
                        torch.sqrt(
                            torch.mean(
                                (rollout.paths - baseline_paths[bank_name]).pow(2)
                            )
                        ).item()
                    ),
                    "sigma_max_active_fraction": float(
                        control_metrics.get("sigma_max_active_fraction", float("nan"))
                    ),
                }
            else:
                row = {
                    "complete_objective_value": float("inf"),
                    "discrepancy_objective_value": float("inf"),
                    "running_cost_value": float("inf"),
                    "running_cost_drift_value": float("inf"),
                    "running_cost_volatility_value": float("inf"),
                    "path_rms_change": float("inf"),
                    "sigma_max_active_fraction": float("nan"),
                }
        cache[key] = dict(row)
        return row

    curve_rows: list[dict[str, object]] = []
    directions = {
        "joint": ("p", "r", "shared"),
        "p_only": ("p",),
        "r_only": ("r",),
        "shared_only": ("shared",),
    }
    for bank_name in banks:
        for direction, active_blocks in directions.items():
            for rate in rates_used:
                trial_rates = {
                    block: (float(rate) if block in active_blocks else 0.0)
                    for block in PARAMETER_BLOCKS
                }
                curve_rows.append(
                    {
                        "bank": bank_name,
                        "direction": direction,
                        "relaxation": float(rate),
                        "rates": dict(trial_rates),
                        **evaluate(bank_name, trial_rates),
                    }
                )

    selections: dict[str, dict[str, object]] = {}
    for selector_bank in banks:
        selected_rates, selected_objective, completed_passes = (
            select_block_coordinate_rates(
                relaxations=rates_used,
                passes=int(coordinate_passes),
                evaluate=lambda trial, bank_name=selector_bank: float(
                    evaluate(bank_name, trial)["complete_objective_value"]
                ),
            )
        )
        other_bank = "heldout" if selector_bank == "training" else "training"
        selections[selector_bank] = {
            "rates": dict(selected_rates),
            "selector_objective": float(selected_objective),
            "coordinate_passes": int(completed_passes),
            "training": evaluate("training", selected_rates),
            "heldout": evaluate("heldout", selected_rates),
            "other_bank": other_bank,
        }

    baseline = {
        bank_name: evaluate(bank_name, zero_rates)
        for bank_name in banks
    }
    summaries: dict[str, object] = {"baseline": baseline}
    for selector_bank, selection in selections.items():
        summaries[f"{selector_bank}_selected"] = {
            "rates": dict(selection["rates"]),
            "training_objective_change": float(
                selection["training"]["complete_objective_value"]
                - baseline["training"]["complete_objective_value"]
            ),
            "heldout_objective_change": float(
                selection["heldout"]["complete_objective_value"]
                - baseline["heldout"]["complete_objective_value"]
            ),
            "training": dict(selection["training"]),
            "heldout": dict(selection["heldout"]),
        }

    training_selected_rates = dict(selections["training"]["rates"])
    scale_rows: list[dict[str, object]] = []
    for scale in scales_used:
        scaled_rates = {
            block: float(scale) * float(training_selected_rates[block])
            for block in PARAMETER_BLOCKS
        }
        training_row = evaluate("training", scaled_rates)
        heldout_row = evaluate("heldout", scaled_rates)
        scale_rows.append(
            {
                "scale": float(scale),
                "rates": scaled_rates,
                "training_objective_change": float(
                    training_row["complete_objective_value"]
                    - baseline["training"]["complete_objective_value"]
                ),
                "heldout_objective_change": float(
                    heldout_row["complete_objective_value"]
                    - baseline["heldout"]["complete_objective_value"]
                ),
                "training": training_row,
                "heldout": heldout_row,
            }
        )

    report: dict[str, object] = {
        "protocol": {
            "candidate_fits_one_frozen_backward_problem": True,
            "training_objective_replays_fitting_bank": True,
            "heldout_path_pool_indices_and_noise_are_independent": True,
            "common_random_numbers_within_each_bank_across_relaxations": True,
            "complete_mp_objective_unchanged": True,
            "neural_loss_unchanged": True,
        },
        "config": {
            "training": asdict(training),
            "nested": asdict(replace(nested, outer_objective_line_search=False)),
            "relaxations": [float(value) for value in rates_used],
            "coordinate_passes": int(coordinate_passes),
            "heldout_seed_offset": int(heldout_seed_offset),
            "selection_scales": [float(value) for value in scales_used],
            "training_index_seed": int(training_bank["index_seed"]),
            "training_noise_seed": int(training_bank["noise_seed"]),
            "heldout_index_seed": int(heldout_bank["index_seed"]),
            "heldout_noise_seed": int(heldout_bank["noise_seed"]),
        },
        "fit_final": dict(fit_result.final_metrics),
        "objective_curves": curve_rows,
        "selections": selections,
        "training_selection_scale_evaluations": scale_rows,
        "summary": summaries,
    }
    artifacts: dict[str, object] = {
        "old_checkpoint": old_checkpoint,
        "candidate_checkpoint": candidate_checkpoint,
        "old_parameter_values": old_values,
        "candidate_parameter_values": candidate_values,
        "training_noise": training_bank["noise"].detach().cpu(),
        "heldout_noise": heldout_bank["noise"].detach().cpu(),
    }
    apply_parameter_block_interpolation(
        model,
        old_values=old_values,
        candidate_values=candidate_values,
        rates={block: 1.0 for block in PARAMETER_BLOCKS},
    )
    return report, artifacts
