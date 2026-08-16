#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    CausalPathFeatureRidgeConditionalExpectation,
    DiscreteMPTrainingConfig,
)
from deep_mkv_gen_path_dt.ce import RidgeConditionalExpectation  # noqa: E402
from deep_mkv_gen_path_dt.controls import HamiltonianControlInputs  # noqa: E402
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
)
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals  # noqa: E402
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.discrepancies import (  # noqa: E402
    StabilizedPrefixFutureVolatilityCorrelationDiscrepancy,
    fit_fixed_target_conditional_residual_volatility_mmd,
    fit_fixed_target_prefix_regime_future_volatility_claims,
)
from run_basseras_heston_path_shadowing import (  # noqa: E402
    NUM_STEPS,
    _build_components,
    _build_model,
    _load_training_data,
)
from run_real_data_r_transmission_diagnostic import (  # noqa: E402
    _restore_normalized_discrepancy,
)
from run_real_data_tuning_candidate import (  # noqa: E402
    _component_args,
    _configure_adjoint_network_stack,
    _configure_r_ce_estimator,
    _replace_named_discrepancy_block,
)


BASIS_NAMES = ("flat_prefix", "causal_compact", "causal_rich")
CONTROL_VARIATES = ("none", "timewise", "adjoint")
TARGET_ESTIMATORS = ("score", "stein")
TIME_MODES = ("stepwise", "pooled")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen held-out sweep for the real-data conditional R estimator. "
            "No model parameter, objective, discrepancy, or running cost is changed."
        )
    )
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--eta-override",
        type=float,
        help=(
            "Diagnostic-only running-cost coefficient. The checkpoint stays "
            "frozen; targets and the Hamiltonian control map are recomputed "
            "under this eta."
        ),
    )
    parser.add_argument("--audit-paths", type=int, default=384)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--projection-folds", type=int, default=4)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=921_103)
    parser.add_argument(
        "--policy",
        choices=("source", "candidate"),
        default="candidate",
        help="Audit the frozen source policy or the completed candidate policy.",
    )
    parser.add_argument(
        "--basis-values",
        choices=BASIS_NAMES,
        nargs="+",
        default=BASIS_NAMES,
    )
    parser.add_argument(
        "--control-variates",
        choices=CONTROL_VARIATES,
        nargs="+",
        default=CONTROL_VARIATES,
    )
    parser.add_argument(
        "--target-estimators",
        choices=TARGET_ESTIMATORS,
        nargs="+",
        default=("score",),
    )
    parser.add_argument(
        "--stein-probes",
        type=int,
        default=8,
        help="Hutchinson probes for the Gaussian-Stein R estimator.",
    )
    parser.add_argument(
        "--time-modes",
        choices=TIME_MODES,
        nargs="+",
        default=TIME_MODES,
        help="Fit separate time-indexed projections or one projection pooled over time.",
    )
    parser.add_argument(
        "--ridge-values",
        type=float,
        nargs="+",
        default=(0.01, 0.1, 1.0, 10.0, 100.0),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _quantile(value: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(value.detach().double().reshape(-1), probability).item())


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.detach().double().pow(2))).item())


def _make_estimator(*, basis: str, ridge: float, grid, time_mode: str):
    pooled = str(time_mode) == "pooled"
    if basis == "flat_prefix":
        return RidgeConditionalExpectation(ridge=float(ridge))
    common = {
        "grid": grid,
        "ridge": float(ridge),
        "rolling_windows": (5, 10, 20, 32),
        "ewma_half_lives": (5, 20, 60),
        "downside_windows": (10, 32),
        "standardized_feature_clip": 4.0,
    }
    if basis == "causal_compact":
        return CausalPathFeatureRidgeConditionalExpectation(
            **common,
            pool_time=pooled,
        )
    if basis == "causal_rich":
        return CausalPathFeatureRidgeConditionalExpectation(
            **common,
            include_expanding_moments=True,
            persistence_lags=(1, 5, 20),
            leverage_lags=(0, 1, 5),
            local_volatility_windows=(5, 20, 32),
            pool_time=pooled,
        )
    raise ValueError(f"unsupported basis {basis}")


def _build_frozen_bank(
    *,
    model,
    target_paths: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    count: int,
    seed: int,
    stein_probes: int = 0,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    source_indices = model._sample_indices(
        size=int(target_paths.shape[0]), count=int(count), generator=generator
    )
    target_indices = model._sample_indices(
        size=int(target_paths.shape[0]), count=int(count), generator=generator
    )
    population_indices = model._sample_indices(
        size=int(target_paths.shape[0]), count=int(count), generator=generator
    )
    x0 = target_paths.index_select(0, source_indices.to(model.device))[:, 0, :]
    target_batch = target_paths.index_select(0, target_indices.to(model.device))
    population_x0 = target_paths.index_select(
        0, population_indices.to(model.device)
    )[:, 0, :]
    noise = sample_standard_normals(
        grid=model.grid,
        batch_size=int(count),
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=int(seed) + 1,
    )
    population_noise = sample_standard_normals(
        grid=model.grid,
        batch_size=int(count),
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=int(seed) + 3,
    )
    with torch.no_grad():
        population_paths = model.rollout(
            x0=population_x0,
            noise=population_noise,
        ).paths.detach()
    stein_r = None
    if int(stein_probes) > 0:
        noise = noise.detach().requires_grad_(True)
        rollout = model.rollout(x0=x0, noise=noise)
        differentiable_raw_p, _ = model._frozen_law_path_functional_adjoint_targets(
            query_path=rollout.paths,
            population_path=population_paths,
            controls=rollout.controls,
            target_path=target_batch,
            training=training,
            create_graph=True,
            preserve_query_path_graph=True,
        )
        stein_r = model._stein_adjoint_noise_target(
            differentiable_adjoint_target=differentiable_raw_p,
            noise=noise,
            moments_noise_target_dim=int(
                model._default_noise_adjoint_dim(model.architecture)
            ),
            num_probes=int(stein_probes),
            seed=int(seed) + 2,
        )
        raw_p = differentiable_raw_p.detach()
    else:
        with torch.no_grad():
            rollout = model.rollout(x0=x0, noise=noise)
        raw_p, _ = model._frozen_law_path_functional_adjoint_targets(
            query_path=rollout.paths,
            population_path=population_paths,
            controls=rollout.controls,
            target_path=target_batch,
            training=training,
        )
    with torch.no_grad():
        sigma_ref = model.control_map._reference_sigmas_all(rollout.paths.detach())
    bank = {
        "paths": rollout.paths.detach(),
        "noise": noise.detach(),
        "raw_p": raw_p.detach(),
        "fitted_p": rollout.expected_adjoint_next.detach(),
        "sigma_ref": sigma_ref.detach(),
    }
    if stein_r is not None:
        bank["stein_r"] = stein_r.detach()
    return bank


def _raw_r(
    *,
    model,
    bank: Mapping[str, torch.Tensor],
    training: DiscreteMPTrainingConfig,
    target_estimator: str = "score",
) -> torch.Tensor:
    if str(target_estimator) == "stein":
        if "stein_r" not in bank:
            raise ValueError("frozen bank does not contain a Stein R target")
        return bank["stein_r"].detach()
    if str(target_estimator) != "score":
        raise ValueError(f"unsupported target estimator {target_estimator}")
    baseline = model._noise_target_control_variate(
        training=training,
        raw_adjoint_prediction=bank["fitted_p"],
    )
    return model._adjoint_noise_target(
        pathwise_adjoint_target=bank["raw_p"],
        moments_noise_target_dim=int(model._default_noise_adjoint_dim(model.architecture)),
        noise=bank["noise"],
        control_variate=baseline,
    ).detach()


def _evaluate(
    *,
    model,
    bank: Mapping[str, torch.Tensor],
    raw_r: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    basis: str,
    ridge: float,
    time_mode: str,
) -> dict[str, float]:
    model.noise_ce_estimator = _make_estimator(
        basis=basis,
        ridge=ridge,
        grid=model.grid,
        time_mode=time_mode,
    )
    configured = replace(training, ridge_lambda=float(ridge))
    with torch.no_grad():
        projected, metadata = model._project_pathwise_target(
            frozen_generated_path=bank["paths"],
            pathwise_target=raw_r,
            training=configured,
            noise_adjoint=True,
        )
        batch_size = int(projected.shape[0])
        state_dim = int(model.architecture.state_dim)
        zero_p = projected.new_zeros((batch_size, state_dim))
        zero_r = projected.new_zeros(
            (batch_size, int(projected.shape[-1]))
        )
        sigma_rows = []
        baseline_rows = []
        for step_index in range(int(model.grid.num_steps)):
            prefix = bank["paths"][:, : step_index + 1, :]
            sigma_rows.append(
                model.control_map(
                    HamiltonianControlInputs(
                        step_index=step_index,
                        x_prefix=prefix,
                        expected_adjoint_next=zero_p,
                        expected_adjoint_noise_next=projected[:, step_index, :],
                    )
                )[..., state_dim:]
            )
            baseline_rows.append(
                model.control_map(
                    HamiltonianControlInputs(
                        step_index=step_index,
                        x_prefix=prefix,
                        expected_adjoint_next=zero_p,
                        expected_adjoint_noise_next=zero_r,
                    )
                )[..., state_dim:]
            )
        sigma = torch.stack(sigma_rows, dim=1).double()
        baseline_sigma = torch.stack(baseline_rows, dim=1).double()
    eps = torch.finfo(torch.float64).eps
    sigma_ratio = sigma / baseline_sigma.clamp_min(eps)
    absolute_relative_change = (sigma_ratio - 1.0).abs()
    sigma_min = float(model.control_map.sigma_min)
    sigma_max = getattr(model.control_map, "sigma_max", None)
    lower_active = sigma <= sigma_min * (1.0 + 1e-6)
    upper_active = (
        torch.zeros_like(lower_active)
        if sigma_max is None
        else sigma >= float(sigma_max) * (1.0 - 1e-6)
    )
    bound_active = lower_active | upper_active
    projection_residual = torch.sum(
        (projected.double() - raw_r.double()).pow(2)
    )
    timewise_baseline = raw_r.double().mean(dim=0, keepdim=True)
    time_centered_total = torch.sum(
        (raw_r.double() - timewise_baseline).pow(2)
    )
    time_centered_projection_r2 = float(
        (
            1.0
            - projection_residual
            / time_centered_total.clamp_min(torch.finfo(torch.float64).eps)
        ).item()
    )
    return {
        "projection_r2": float(metadata["ce_projection_r2"]),
        "time_centered_projection_r2": time_centered_projection_r2,
        "projection_error_rms": float(metadata["ce_projection_rms"]),
        "projected_r_rms": _rms(projected),
        "raw_r_rms": _rms(raw_r),
        "projected_to_raw_rms_ratio": _rms(projected)
        / max(_rms(raw_r), torch.finfo(torch.float64).eps),
        "feature_dim_max": float(metadata.get("ce_feature_dim_max", math.nan)),
        "control_sigma_ratio_q01": _quantile(sigma_ratio, 0.01),
        "control_sigma_ratio_q50": _quantile(sigma_ratio, 0.50),
        "control_sigma_ratio_q99": _quantile(sigma_ratio, 0.99),
        "control_sigma_absolute_relative_change_q50": _quantile(
            absolute_relative_change, 0.50
        ),
        "control_sigma_absolute_relative_change_q95": _quantile(
            absolute_relative_change, 0.95
        ),
        "control_requests_higher_sigma_fraction": float(
            (sigma > baseline_sigma).double().mean().item()
        ),
        "control_sigma_lower_bound_active_fraction": float(
            lower_active.double().mean().item()
        ),
        "control_sigma_upper_bound_active_fraction": float(
            upper_active.double().mean().item()
        ),
        "control_sigma_bound_active_fraction": float(
            bound_active.double().mean().item()
        ),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["target_estimator"]),
            str(row["basis"]),
            str(row["control_variate"]),
            str(row["time_mode"]),
            float(row["ridge"]),
        )
        grouped.setdefault(key, []).append(row)
    result = []
    metric_names = tuple(
        name
        for name in rows[0]
        if name not in {
            "target_estimator",
            "basis",
            "control_variate",
            "time_mode",
            "ridge",
            "repetition",
        }
    )
    for (
        target_estimator,
        basis,
        control_variate,
        time_mode,
        ridge,
    ), values in grouped.items():
        aggregate: dict[str, Any] = {
            "target_estimator": target_estimator,
            "basis": basis,
            "control_variate": control_variate,
            "time_mode": time_mode,
            "ridge": ridge,
            "repetitions": len(values),
        }
        for name in metric_names:
            items = [float(row[name]) for row in values]
            aggregate[f"{name}_mean"] = float(statistics.mean(items))
            aggregate[f"{name}_max"] = float(max(items))
            aggregate[f"{name}_min"] = float(min(items))
        result.append(aggregate)
    return result


def _select(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    base_eligible = [
        row
        for row in rows
        if math.isfinite(float(row["time_centered_projection_r2_mean"]))
        and float(row["time_centered_projection_r2_mean"]) > 0.0
        and float(row["projected_to_raw_rms_ratio_mean"]) >= 0.02
    ]
    strict = [
        row
        for row in base_eligible
        if float(row["control_sigma_bound_active_fraction_max"]) <= 0.01
    ]
    fallback = [
        row
        for row in base_eligible
        if float(row["control_sigma_bound_active_fraction_max"]) <= 0.10
    ]
    eligible = strict if strict else fallback
    gate = (
        "strict_max_control_bound_active_fraction_0.01"
        if strict
        else (
            "fallback_max_control_bound_active_fraction_0.10"
            if fallback
            else "no_candidate_passed_heldout_signal_and_control_bound_gates"
        )
    )
    for row in rows:
        row["eligible"] = row in eligible
        row["passes_strict_control_bound_gate"] = row in strict
        row["selection_score"] = (
            float(row["time_centered_projection_r2_mean"])
            - 0.25
            * float(
                row["time_centered_projection_r2_max"]
                - row["time_centered_projection_r2_min"]
            )
            - 2.0 * float(row["control_sigma_bound_active_fraction_mean"])
        )
    pool = eligible if eligible else rows
    return max(pool, key=lambda row: float(row["selection_score"])), gate


def main() -> None:
    args = _parse_args()
    if int(args.audit_paths) < 24 or int(args.repetitions) < 1:
        raise ValueError("audit-paths must be >=24 and repetitions must be positive")
    if int(args.projection_folds) < 2:
        raise ValueError("projection-folds must be at least two")
    if int(args.stein_probes) < 1:
        raise ValueError("stein-probes must be positive")
    ridge_values = tuple(float(value) for value in args.ridge_values)
    if any(not math.isfinite(value) or value < 0.0 for value in ridge_values):
        raise ValueError("ridge values must be finite and nonnegative")

    candidate_dir = args.candidate_run_dir.resolve()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    manifest = _read_json(candidate_dir / "candidate_manifest.json")
    adapter_dataset = candidate_dir / "adapter_dataset"
    device = torch.device(args.device)
    target_paths, transform, preprocessing = _load_training_data(
        REPO_ROOT,
        device=device,
        dtype=torch.float32,
        state_representation="log_price",
        custom_dataset_root=adapter_dataset,
        model_dt=1.0 / NUM_STEPS,
    )
    train_prices = np.load(adapter_dataset / "train.npy").astype(np.float64)
    reference_sigma_std_floor = (
        0.25
        * float(np.diff(np.log(train_prices), axis=1).std(ddof=0))
        / math.sqrt(1.0 / NUM_STEPS)
    )
    component_args = _component_args(
        argparse.Namespace(
            objective_protocol=str(manifest["objective_protocol"]),
            eta=(
                float(manifest["eta"])
                if args.eta_override is None
                else float(args.eta_override)
            ),
            running_cost=str(manifest.get("running_cost", "specific_entropy")),
            joint_weight=float(manifest["joint_weight"]),
            reference_feature_basis=str(
                manifest.get("reference_feature_basis", "compact")
            ),
            reference_causal_weight=manifest.get("reference_causal_weight"),
            discrepancy_acf_lags=tuple(
                int(value)
                for value in (
                    manifest.get("discrepancy_acf_lags")
                    or (1, 2, 5, 10)
                )
            ),
        ),
        reference_sigma_std_floor=reference_sigma_std_floor,
        admissible_sigma_max=float(manifest.get("admissible_sigma_max", 0.6)),
    )
    grid, architecture, control, _, joint = _build_components(
        component_args,
        target_paths=target_paths,
        path_scale=float(transform.sigma) / math.sqrt(float(transform.dt)),
    )
    conditional_claim = manifest.get("conditional_claim")
    if isinstance(conditional_claim, Mapping):
        mode = str(conditional_claim.get("mode", "replace_joint"))
        if mode == "replace_joint":
            claim = fit_fixed_target_prefix_regime_future_volatility_claims(
                target_paths,
                grid=grid,
                split_index=int(conditional_claim["split_index"]),
                prefix_window=int(conditional_claim["prefix_window"]),
                future_horizon=int(conditional_claim["future_horizon"]),
                gate_temperature_fraction=float(
                    conditional_claim["gate_temperature_fraction"]
                ),
                tail_temperature=float(conditional_claim["tail_temperature"]),
                conditional_moment_weight=float(
                    conditional_claim["conditional_moment_weight"]
                ),
            )
        elif mode == "residual_mmd":
            claim = fit_fixed_target_conditional_residual_volatility_mmd(
                target_paths,
                grid=grid,
                split_index=int(conditional_claim["split_index"]),
                future_horizon=int(conditional_claim["future_horizon"]),
                prefix_windows=tuple(
                    int(value) for value in conditional_claim["prefix_windows"]
                ),
                recent_return_window=int(
                    conditional_claim["recent_return_window"]
                ),
                ridge=float(conditional_claim["predictor_ridge"]),
                bandwidths=(0.5, 1.0, 2.0),
                moment_weight=float(conditional_claim.get("moment_weight", 0.0)),
            )
        elif mode == "correlation":
            claim = StabilizedPrefixFutureVolatilityCorrelationDiscrepancy(
                windows=tuple(int(value) for value in conditional_claim["windows"]),
                full_matrix=bool(conditional_claim["full_matrix"]),
                split_index=int(conditional_claim["split_index"]),
                scale_floor_fraction=float(
                    conditional_claim["scale_floor_fraction"]
                ),
            )
        else:
            raise ValueError(f"unsupported conditional claim mode {mode}")
        joint = _replace_named_discrepancy_block(
            joint,
            name="volatility_law_joint_prefix_future_rv",
            replacement=claim,
        )
    if str(manifest.get("discrepancy_stack", "full")) == "compact":
        retained_names = {
            str(value)
            for value in manifest.get("retained_discrepancy_blocks", ())
        }
        retained = tuple(
            block for block in joint.blocks if str(block.name) in retained_names
        )
        if {str(block.name) for block in retained} != retained_names:
            raise ValueError("compact discrepancy blocks do not match the manifest")
        joint = CompositePathFunctionalDiscrepancy(blocks=retained)
    discrepancy = _restore_normalized_discrepancy(
        joint, manifest["discrepancy_normalization"]
    )
    model = _build_model(
        architecture=architecture,
        grid=grid,
        control=control,
        discrepancy=discrepancy,
        device=device,
        dtype=torch.float32,
        seed=int(manifest["seed"]),
    )
    estimator_args = argparse.Namespace(
        p_ce_basis=str(manifest.get("p_ce_basis", "flat_prefix")),
        r_ce_basis=str(manifest.get("r_ce_basis", "shared_flat_prefix")),
        r_ce_normalization=str(manifest.get("r_ce_normalization", "batch")),
        r_ce_time_mode=str(manifest.get("r_ce_time_mode", "stepwise")),
        ridge_lambda=float(manifest["training"]["ridge_lambda"]),
    )
    _configure_r_ce_estimator(model, estimator_args, target_paths=target_paths)
    adjoint_network = manifest.get("adjoint_network", {"kind": "plain"})
    if not isinstance(adjoint_network, Mapping):
        raise ValueError("candidate manifest has an invalid adjoint_network summary")
    adjoint_kind = str(adjoint_network.get("kind", "plain"))
    default_stack_depth = 0 if adjoint_kind == "plain" else 1
    requested_stack_depth = int(
        adjoint_network.get("stack_depth", default_stack_depth)
    )
    if str(args.policy) == "source":
        source = manifest.get("source_checkpoint")
        if not isinstance(source, Mapping):
            raise ValueError("candidate manifest is missing source checkpoint provenance")
        requested_stack_depth = int(
            source.get(
                "stack_depth",
                1 if str(source.get("source_topology", "plain")) != "plain" else 0,
            )
        )
    _configure_adjoint_network_stack(
        model,
        kind=adjoint_kind,
        stack_depth=requested_stack_depth,
        seed=int(manifest["seed"]),
        target_paths=target_paths,
    )
    if str(args.policy) == "source":
        policy_checkpoint = Path(str(source["path"]))
    else:
        policy_checkpoint = Path(str(manifest["checkpoint"]))
    model.load_checkpoint_state(
        torch.load(policy_checkpoint, map_location="cpu", weights_only=True)
    )
    # The audit differentiates only with respect to Gaussian innovations.  The
    # frozen policy parameters are constants in the conditional expectation.
    for parameter in model.network.parameters():
        parameter.requires_grad_(False)
    base_training = replace(
        DiscreteMPTrainingConfig(**manifest["training"]),
        batch_size=min(int(args.audit_paths), int(target_paths.shape[0])),
        target_batch_size=min(int(args.audit_paths), int(target_paths.shape[0])),
        num_steps=1,
        ce_crossfit_folds=int(args.projection_folds),
        preconditioner_batches=int(args.calibration_batches),
        grad_clip_norm=None,
        log_every=1,
        seed=int(args.seed),
    )
    print("calibrating the independent timewise unbiased R control variate", flush=True)
    model._fit_target_preconditioner(
        target_paths=target_paths,
        training=replace(
            base_training,
            ce_crossfit_folds=1,
            noise_target_control_variate="timewise",
        ),
    )
    count = min(int(args.audit_paths), int(target_paths.shape[0]))
    banks = []
    for repetition in range(int(args.repetitions)):
        print(f"building frozen audit bank {repetition + 1}/{args.repetitions}", flush=True)
        seed = derive_stream_seed(
            base_seed=int(args.seed), stream_offset=100_000 + repetition
        )
        assert seed is not None
        banks.append(
            _build_frozen_bank(
                model=model,
                target_paths=target_paths,
                training=base_training,
                count=count,
                seed=int(seed),
                stein_probes=(
                    int(args.stein_probes)
                    if "stein" in tuple(str(value) for value in args.target_estimators)
                    else 0
                ),
            )
        )

    rows: list[dict[str, Any]] = []
    raw_rms_by_repetition: dict[int, dict[str, float]] = {}
    for repetition, bank in enumerate(banks):
        raw_rms_by_repetition[repetition] = {}
        estimator_control_variates = []
        if "score" in tuple(str(value) for value in args.target_estimators):
            estimator_control_variates.extend(
                ("score", str(value)) for value in args.control_variates
            )
        if "stein" in tuple(str(value) for value in args.target_estimators):
            estimator_control_variates.append(("stein", "not_applicable"))
        for target_estimator, control_variate in estimator_control_variates:
            training = replace(
                base_training,
                noise_target_control_variate=(
                    "none" if control_variate == "not_applicable" else control_variate
                ),
            )
            raw_r = _raw_r(
                model=model,
                bank=bank,
                training=training,
                target_estimator=target_estimator,
            )
            variance_key = f"{target_estimator}:{control_variate}"
            raw_rms_by_repetition[repetition][variance_key] = _rms(raw_r)
            for basis in tuple(str(value) for value in args.basis_values):
                for time_mode in tuple(str(value) for value in args.time_modes):
                    if basis == "flat_prefix" and time_mode == "pooled":
                        continue
                    for ridge in ridge_values:
                        metrics = _evaluate(
                            model=model,
                            bank=bank,
                            raw_r=raw_r,
                            training=training,
                            basis=basis,
                            ridge=ridge,
                            time_mode=time_mode,
                        )
                        rows.append(
                            {
                                "repetition": repetition,
                                "target_estimator": target_estimator,
                                "basis": basis,
                                "control_variate": control_variate,
                                "time_mode": time_mode,
                                "ridge": ridge,
                                **metrics,
                            }
                        )
            print(
                f"completed repetition={repetition + 1} "
                f"target_estimator={target_estimator} control_variate={control_variate}",
                flush=True,
            )
    aggregates = _aggregate(rows)
    selected, selection_gate = _select(aggregates)
    audited_control_variates = tuple(raw_rms_by_repetition[0])
    variance_reduction = {
        control_variate: {
            "raw_r_rms_by_repetition": [
                raw_rms_by_repetition[index][control_variate]
                for index in range(len(banks))
            ],
            "raw_r_rms_ratio_to_none_by_repetition": (
                [
                    raw_rms_by_repetition[index][control_variate]
                    / max(
                        raw_rms_by_repetition[index]["score:none"],
                        torch.finfo(torch.float64).eps,
                    )
                    for index in range(len(banks))
                ]
                if "score:none" in audited_control_variates
                else None
            ),
        }
        for control_variate in audited_control_variates
    }
    for payload in variance_reduction.values():
        values = payload["raw_r_rms_ratio_to_none_by_repetition"]
        payload["raw_r_rms_ratio_to_none_mean"] = (
            None if values is None else float(statistics.mean(values))
        )
    output = {
        "protocol": {
            "frozen_checkpoint_no_retraining": True,
            "four_fold_projection_is_held_out": int(args.projection_folds) == 4,
            "control_variates_are_f_n_measurable_and_unbiased_after_multiplication_by_epsilon": True,
            "selection_uses_only_frozen_training_law_banks": True,
            "objective_discrepancy_running_cost_and_neural_loss_unchanged": True,
        },
        "candidate_run_dir": str(candidate_dir),
        "policy": str(args.policy),
        "policy_checkpoint": str(policy_checkpoint.resolve()),
        "diagnostic_eta": float(component_args.eta),
        "checkpoint_eta": float(manifest["eta"]),
        "config": {
            **vars(args),
            "candidate_run_dir": str(candidate_dir),
            "run_dir": str(run_dir),
            "audit_paths_used": count,
            "ridge_values": list(ridge_values),
            "preprocessing": preprocessing,
        },
        "variance_reduction": variance_reduction,
        "selected": {
            "target_estimator": selected["target_estimator"],
            "basis": selected["basis"],
            "control_variate": selected["control_variate"],
            "time_mode": selected["time_mode"],
            "ridge": selected["ridge"],
            "selection_score": selected["selection_score"],
            "selection_gate": selection_gate,
            "passes_strict_control_bound_gate": bool(
                selected["passes_strict_control_bound_gate"]
            ),
            "eligible": bool(selected["eligible"]),
        },
        "aggregate_candidates": sorted(
            aggregates,
            key=lambda row: float(row["selection_score"]),
            reverse=True,
        ),
        "per_repetition_candidates": rows,
    }
    write_json(run_dir / "summary.json", output)
    print(
            "selected target_estimator={target_estimator} basis={basis} "
            "control_variate={cv} time_mode={time_mode} ridge={ridge:g} "
            "time_centered_R2={r2:.4f} control_bound_active={bound:.4f}".format(
            target_estimator=selected["target_estimator"],
            basis=selected["basis"],
            cv=selected["control_variate"],
            time_mode=selected["time_mode"],
            ridge=float(selected["ridge"]),
            r2=float(selected["time_centered_projection_r2_mean"]),
            bound=float(selected["control_sigma_bound_active_fraction_mean"]),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
