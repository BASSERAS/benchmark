#!/usr/bin/env python3
"""Resample a frozen tuning checkpoint and score a matched validation bank.

The tuning runner deliberately uses smaller banks for fast screening.  This
utility changes only Monte Carlo bank size: it reconstructs the recorded model
topology, loads an immutable checkpoint, and evaluates it on the same frozen
validation split and path-shadowing protocol.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from path_dt_experiments.real_data_generation import (  # noqa: E402
    load_frozen_real_data_fit,
)
from path_dt_experiments.real_data_protocol import (  # noqa: E402
    normalized_prices_to_log_paths,
    real_data_law_metrics,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from path_dt_experiments.shadowing import (  # noqa: E402
    ShadowingFeatureConfig,
    evaluate_path_shadowing,
)
from run_basseras_heston_path_shadowing import (  # noqa: E402
    NUM_STEPS,
    _build_components,
    _build_model,
    _load_training_data,
)
from run_real_data_tuning_candidate import (  # noqa: E402
    _component_args,
    _configure_adjoint_network_stack,
    _configure_r_ce_estimator,
    _sample_price_bank,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Defaults to the checkpoint recorded in candidate_manifest.json.",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help=(
            "Generate from the fitted reference by setting every learned "
            "adjoint-network parameter to zero instead of loading a checkpoint."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--sample-batch-size", type=int, default=8192)
    parser.add_argument(
        "--bank-seed",
        type=int,
        help="Defaults to the tuning screen's seed + 70000.",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=REPO_ROOT / "data" / "ICAIF_3Y" / "canonical_rth_128",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("/home/samer/scenarios/BASSERAS-benchmark"),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _excess_kurtosis(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    variance = float(np.mean(centered**2))
    return float(np.mean(centered**4) / max(variance * variance, 1e-30) - 3.0)


def _law_scale_diagnostics(
    generated_prices: np.ndarray,
    train_prices: np.ndarray,
) -> dict[str, float]:
    generated_returns = np.diff(np.log(generated_prices.astype(np.float64)), axis=1)
    train_returns = np.diff(np.log(train_prices.astype(np.float64)), axis=1)
    generated_rv = np.sqrt(np.sum(generated_returns**2, axis=1))
    train_rv = np.sqrt(np.sum(train_returns**2, axis=1))
    generated_abs = np.abs(generated_returns).reshape(-1)
    train_abs = np.abs(train_returns).reshape(-1)
    return {
        "generated_return_std": float(generated_returns.std(ddof=0)),
        "train_return_std": float(train_returns.std(ddof=0)),
        "return_std_ratio_to_training": float(
            generated_returns.std(ddof=0) / max(train_returns.std(ddof=0), 1e-15)
        ),
        "generated_rv_mean": float(generated_rv.mean()),
        "train_rv_mean": float(train_rv.mean()),
        "rv_mean_ratio_to_training": float(generated_rv.mean() / max(train_rv.mean(), 1e-15)),
        "generated_rv_std": float(generated_rv.std(ddof=0)),
        "train_rv_std": float(train_rv.std(ddof=0)),
        "rv_std_ratio_to_training": float(generated_rv.std(ddof=0) / max(train_rv.std(ddof=0), 1e-15)),
        "generated_rv_q90": float(np.quantile(generated_rv, 0.90)),
        "train_rv_q90": float(np.quantile(train_rv, 0.90)),
        "rv_q90_ratio_to_training": float(np.quantile(generated_rv, 0.90) / max(float(np.quantile(train_rv, 0.90)), 1e-15)),
        "abs_return_q99_ratio_to_training": float(np.quantile(generated_abs, 0.99) / max(float(np.quantile(train_abs, 0.99)), 1e-15)),
        "abs_return_q999_ratio_to_training": float(np.quantile(generated_abs, 0.999) / max(float(np.quantile(train_abs, 0.999)), 1e-15)),
        "generated_excess_kurtosis": _excess_kurtosis(generated_returns),
        "train_excess_kurtosis": _excess_kurtosis(train_returns),
        "nonfinite_path_fraction": float(1.0 - np.mean(np.isfinite(generated_prices))),
    }


def _reference_drifts_all(control_map, paths: torch.Tensor) -> torch.Tensor:
    reference_drifts_all = getattr(control_map, "_reference_drifts_all", None)
    if callable(reference_drifts_all):
        return reference_drifts_all(paths)
    reference_kernel = control_map.reference_kernel
    evaluate_all = getattr(reference_kernel, "evaluate_all", None)
    if callable(evaluate_all):
        reference_drift, _ = evaluate_all(paths)
        return reference_drift
    evaluate = getattr(reference_kernel, "evaluate", None)
    if not callable(evaluate):
        raise TypeError("the control's reference kernel must provide evaluate")
    return torch.stack(
        [
            evaluate(paths[:, : step + 1, :], step_index=step)[0]
            for step in range(int(paths.shape[1]) - 1)
        ],
        dim=1,
    )


def _control_diagnostics(
    model,
    *,
    bank_size: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    state_unit_scale: float,
) -> dict[str, float]:
    sigma_parts: list[np.ndarray] = []
    drift_parts: list[np.ndarray] = []
    correction_parts: list[np.ndarray] = []
    root_parts: list[np.ndarray] = []
    x0 = torch.zeros((1, 1), device=device, dtype=dtype)
    was_training = bool(model.network.training)
    model.network.eval()
    try:
        with torch.no_grad():
            for batch_index, start in enumerate(range(0, int(bank_size), int(batch_size))):
                count = min(int(batch_size), int(bank_size) - start)
                sample = model.sample(
                    num_paths=count,
                    x0=x0,
                    seed=derive_stream_seed(
                        base_seed=int(seed), stream_offset=50_000 + batch_index
                    ),
                )
                reference_drift = _reference_drifts_all(
                    model.control_map, sample.paths
                )
                reference_sigma = model.control_map._reference_sigmas_all(sample.paths)
                controls = sample.controls
                drift_parts.append(
                    (controls[..., :1] * float(state_unit_scale)).cpu().numpy()
                )
                sigma_parts.append(
                    (controls[..., 1:] * float(state_unit_scale)).cpu().numpy()
                )
                correction_parts.append(
                    (
                        (controls[..., :1] - reference_drift)
                        * float(state_unit_scale)
                    ).cpu().numpy()
                )
                linear = sample.expected_adjoint_noise_next / math.sqrt(
                    float(model.grid.dt)
                )
                root_parts.append(
                    torch.hypot(
                        linear,
                        2.0 * float(model.control_map.eta) / reference_sigma,
                    ).cpu().numpy()
                )
    finally:
        model.network.train(was_training)
    sigma = np.concatenate(sigma_parts).reshape(-1)
    drift = np.concatenate(drift_parts).reshape(-1)
    correction = np.concatenate(correction_parts).reshape(-1)
    roots = np.concatenate(root_parts).reshape(-1)
    sigma_min = float(model.control_map.sigma_min) * float(state_unit_scale)
    sigma_max = float(model.control_map.sigma_max) * float(state_unit_scale)
    return {
        "sigma_min_physical": sigma_min,
        "sigma_max_physical": sigma_max,
        "sigma_mean_physical": float(sigma.mean()),
        "sigma_std_physical": float(sigma.std()),
        "sigma_q99_physical": float(np.quantile(sigma, 0.99)),
        "sigma_min_active_fraction": float(np.mean(sigma <= sigma_min * (1.0 + 1e-5))),
        "sigma_max_active_fraction": float(np.mean(sigma >= sigma_max * (1.0 - 1e-5))),
        "physical_drift_rms": float(np.sqrt(np.mean(drift**2))),
        "physical_drift_max_abs": float(np.max(np.abs(drift))),
        "reference_drift_correction_rms_physical": float(np.sqrt(np.mean(correction**2))),
        "reference_drift_max_abs_error_physical": float(np.max(np.abs(correction))),
        "gaussian_discriminant_root_min": float(np.min(roots)),
        "nonfinite_control_fraction": float(
            1.0
            - np.mean(
                np.isfinite(np.concatenate((drift, sigma, correction, roots)))
            )
        ),
    }


def _guardrails(
    law: dict[str, float],
    controls: dict[str, float],
    *,
    volatility_only: bool,
    use_reference_drift: bool,
) -> dict[str, object]:
    checks = {
        "return_std_ratio": 0.5 <= law["return_std_ratio_to_training"] <= 2.0,
        "rv_mean_ratio": 0.5 <= law["rv_mean_ratio_to_training"] <= 2.0,
        "rv_std_ratio": 0.25 <= law["rv_std_ratio_to_training"] <= 3.0,
        "rv_q90_ratio": 0.5 <= law["rv_q90_ratio_to_training"] <= 2.0,
        "absolute_return_q99": law["abs_return_q99_ratio_to_training"] <= 2.5,
        "absolute_return_q999": law["abs_return_q999_ratio_to_training"] <= 4.0,
        "excess_kurtosis": abs(law["generated_excess_kurtosis"]) <= max(
            100.0, 10.0 * (1.0 + abs(law["train_excess_kurtosis"]))
        ),
        "sigma_min_activity": controls["sigma_min_active_fraction"] <= 0.01,
        "sigma_max_activity": controls["sigma_max_active_fraction"] <= 0.01,
        "finite_paths": law["nonfinite_path_fraction"] == 0.0,
        "finite_controls": controls["nonfinite_control_fraction"] == 0.0,
        "positive_gaussian_root": controls["gaussian_discriminant_root_min"] > 0.0,
    }
    if volatility_only:
        if use_reference_drift:
            checks["reference_drift_retained"] = (
                controls["reference_drift_max_abs_error_physical"] <= 1e-6
            )
        else:
            checks["zero_physical_drift"] = (
                controls["physical_drift_max_abs"] <= 1e-6
            )
    return {"checks": checks, "pass": bool(all(checks.values()))}


def main() -> None:
    args = _parse_args()
    if min(int(args.bank_size), int(args.sample_batch_size)) < 1:
        raise ValueError("bank sizes must be positive")
    candidate_dir = args.candidate_run_dir.resolve()
    manifest = _read_json(candidate_dir / "candidate_manifest.json")
    checkpoint = (
        Path(str(manifest["checkpoint"]))
        if args.checkpoint is None
        else args.checkpoint.resolve()
    )
    if not args.reference_only and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run_dir = ensure_run_dir(args.run_dir.resolve())
    device = torch.device(args.device)
    dtype = torch.float32
    fit = load_frozen_real_data_fit(
        args.canonical_root,
        protocol_identifier=str(manifest["protocol"]),
        index=str(manifest["index"]),
    )
    adapter_dataset = candidate_dir / "adapter_dataset"
    target_paths, transform, preprocessing = _load_training_data(
        args.benchmark_root,
        device=device,
        dtype=dtype,
        state_representation="log_price",
        custom_dataset_root=adapter_dataset,
        model_dt=1.0 / NUM_STEPS,
    )
    state_unit_scale = float(manifest.get("state_unit_scale", 1.0))
    if not math.isfinite(state_unit_scale) or state_unit_scale <= 0.0:
        raise ValueError("candidate manifest has an invalid state_unit_scale")
    target_paths = target_paths / state_unit_scale
    train_returns = np.diff(np.log(fit.train_prices.astype(np.float64)), axis=1)
    reference_sigma_std_floor = (
        0.25
        * float(train_returns.std(ddof=0))
        / math.sqrt(1.0 / NUM_STEPS)
    )
    component_namespace = argparse.Namespace(
        objective_protocol=str(manifest["objective_protocol"]),
        eta=float(manifest["eta"]),
        running_cost=str(manifest.get("running_cost", "specific_entropy")),
        reference_kind=str(manifest.get("reference_kind", "calibrated_causal")),
        guyon_realized_volatility_window=int(
            manifest.get("guyon_realized_volatility_window", 20)
        ),
        use_reference_drift=bool(manifest.get("use_reference_drift", False)),
        drift_control_mode=str(manifest.get("drift_control_mode", "full")),
        joint_weight=float(manifest["joint_weight"]),
        reference_feature_basis=str(
            manifest.get("reference_feature_basis", "compact")
        ),
        reference_causal_weight=manifest.get("reference_causal_weight"),
        reference_rv_spread_ratio_min=float(
            manifest.get("reference_rv_spread_ratio_min", 0.4)
        ),
        discrepancy_acf_lags=tuple(
            int(value)
            for value in manifest.get(
                "discrepancy_acf_lags",
                (1, 2, 5, 10),
            )
        ),
    )
    component_args = _component_args(
        component_namespace,
        reference_sigma_std_floor=reference_sigma_std_floor,
        admissible_sigma_max=float(manifest["admissible_sigma_max"]),
        state_unit_scale=state_unit_scale,
    )
    grid, architecture, control, base, joint = _build_components(
        component_args,
        target_paths=target_paths,
        path_scale=float(transform.sigma) / math.sqrt(float(transform.dt)),
    )
    model = _build_model(
        architecture=architecture,
        grid=grid,
        control=control,
        discrepancy=(base if str(manifest["phase"]) == "source" else joint),
        device=device,
        dtype=dtype,
        seed=int(manifest["seed"]),
    )
    estimator_args = argparse.Namespace(
        p_ce_basis=str(manifest.get("p_ce_basis", "flat_prefix")),
        r_ce_basis=str(manifest.get("r_ce_basis", "shared_flat_prefix")),
        r_ce_normalization=str(manifest.get("r_ce_normalization", "batch")),
        r_ce_time_mode=str(manifest.get("r_ce_time_mode", "stepwise")),
        ridge_lambda=float(manifest["training"]["ridge_lambda"]),
    )
    adjoint_network = manifest.get("adjoint_network", {"kind": "plain"})
    if not isinstance(adjoint_network, dict):
        raise ValueError("candidate manifest has an invalid adjoint network")
    adjoint_kind = str(adjoint_network.get("kind", "plain"))
    default_stack_depth = 0 if adjoint_kind == "plain" else 1
    stack_depth = int(adjoint_network.get("stack_depth", default_stack_depth))
    source_r_ce_basis = str(
        manifest.get("source_checkpoint_r_ce_basis", estimator_args.r_ce_basis)
    )
    regime_gate_temperature_fraction = float(
        adjoint_network.get("regime_gate_temperature_fraction", 0.10)
    )
    regime_gate_feature = str(
        adjoint_network.get(
            "regime_gate_feature", "rolling_realized_variance_20"
        )
    )
    source_checkpoint = manifest.get("source_checkpoint")
    source_stack_depth = 0
    if isinstance(source_checkpoint, dict):
        source_stack_depth = int(source_checkpoint.get("stack_depth", 0))
    if adjoint_kind == "causal_regime_residual":
        if source_stack_depth < 1 or source_stack_depth >= stack_depth:
            raise ValueError(
                "a regime-gated candidate must record its frozen causal source stack"
            )
        source_estimator_args = argparse.Namespace(**vars(estimator_args))
        source_estimator_args.r_ce_basis = source_r_ce_basis
        _configure_r_ce_estimator(
            model,
            source_estimator_args,
            target_paths=target_paths,
        )
        _configure_adjoint_network_stack(
            model,
            kind="causal_residual",
            stack_depth=source_stack_depth,
            seed=int(manifest["seed"]),
            target_paths=target_paths,
        )
        _configure_r_ce_estimator(model, estimator_args, target_paths=target_paths)
    elif stack_depth > 1 and source_r_ce_basis != str(estimator_args.r_ce_basis):
        source_estimator_args = argparse.Namespace(**vars(estimator_args))
        source_estimator_args.r_ce_basis = source_r_ce_basis
        _configure_r_ce_estimator(
            model,
            source_estimator_args,
            target_paths=target_paths,
        )
        _configure_adjoint_network_stack(
            model,
            kind=adjoint_kind,
            stack_depth=1,
            seed=int(manifest["seed"]),
            target_paths=target_paths,
        )
    _configure_r_ce_estimator(model, estimator_args, target_paths=target_paths)
    _configure_adjoint_network_stack(
        model,
        kind=adjoint_kind,
        stack_depth=stack_depth,
        seed=int(manifest["seed"]),
        target_paths=target_paths,
        regime_gate_temperature_fraction=regime_gate_temperature_fraction,
        regime_gate_feature=regime_gate_feature,
    )
    if args.reference_only:
        with torch.no_grad():
            for parameter in model.network.parameters():
                parameter.zero_()
    else:
        model.load_checkpoint_state(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )

    bank_seed = (
        int(manifest["seed"]) + 70_000
        if args.bank_seed is None
        else int(args.bank_seed)
    )
    bank = _sample_price_bank(
        model,
        transform=transform,
        bank_size=int(args.bank_size),
        batch_size=int(args.sample_batch_size),
        seed=bank_seed,
        device=device,
        dtype=dtype,
        state_unit_scale=state_unit_scale,
    )
    bank_path = run_dir / "validation_bank.npy"
    np.save(bank_path, bank)
    generated_log = normalized_prices_to_log_paths(bank)
    validation_log = normalized_prices_to_log_paths(fit.validation_prices)
    law = real_data_law_metrics(generated_log, validation_log)
    law_scale = _law_scale_diagnostics(bank, fit.train_prices)
    control_diagnostics = _control_diagnostics(
        model,
        bank_size=int(args.bank_size),
        batch_size=int(args.sample_batch_size),
        seed=bank_seed,
        device=device,
        dtype=dtype,
        state_unit_scale=state_unit_scale,
    )
    guardrails = _guardrails(
        law_scale,
        control_diagnostics,
        volatility_only=(
            str(manifest.get("drift_control_mode", "full")) == "volatility_only"
        ),
        use_reference_drift=bool(manifest.get("use_reference_drift", False)),
    )
    shadow = evaluate_path_shadowing(
        real_paths=validation_log,
        generated_paths=generated_log,
        prefix_length=65,
        future_horizon=32,
        top_k=256,
        feature_config=ShadowingFeatureConfig(),
        search_backend="auto",
    )
    write_json(
        run_dir / "metrics.json",
        {
            "selection_scope": "validation_only",
            "checkpoint_frozen": not bool(args.reference_only),
            "reference_only": bool(args.reference_only),
            "retraining": False,
            "candidate_run_dir": str(candidate_dir),
            "checkpoint": (
                None if args.reference_only else str(checkpoint.resolve())
            ),
            "bank": str(bank_path.resolve()),
            "bank_size": int(args.bank_size),
            "bank_seed": bank_seed,
            "preprocessing": preprocessing,
            "state_unit_scaling": str(manifest.get("state_unit_scaling", "none")),
            "state_unit_scale": state_unit_scale,
            "law_metrics": law,
            "shadowing_metrics": shadow.metrics,
            "full_bank_law_scale_diagnostics": law_scale,
            "full_bank_control_diagnostics": control_diagnostics,
            "hard_guardrails": guardrails,
        },
    )
    print(
        "bank={bank} return_w={rw:.6g} rv_w={rvw:.6g} rv_crps={crps:.6g} "
        "rv_cov90={coverage:.3f}".format(
            bank=int(args.bank_size),
            rw=float(law["return_wasserstein_normalized"]),
            rvw=float(law["realized_volatility_wasserstein_normalized"]),
            crps=float(shadow.metrics["realized_volatility_crps"]),
            coverage=float(shadow.metrics["realized_volatility_coverage_90"]),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
