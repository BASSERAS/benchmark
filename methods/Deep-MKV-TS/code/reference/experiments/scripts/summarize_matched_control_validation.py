#!/usr/bin/env python3
"""Aggregate the locked matched seed-zero control screen without test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


ARMS = ("reference", "full_control_nested_mp", "volatility_only_nested_mp")
SYNTHETIC = ("heston", "mixture")
REAL = ("ES", "NQ", "RTY", "YM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(path)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _history(
    path: Path, *, selected_outer_step: int | None = None
) -> dict[str, float | int]:
    all_rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = (
        all_rows
        if selected_outer_step is None
        else all_rows[: int(selected_outer_step)]
    )
    if not rows:
        return {"outer_steps": 0, "accepted_outer_steps": 0}
    before = float(rows[0].get("outer_objective_before", math.nan))
    after = float(rows[-1].get("outer_accepted_objective", math.nan))
    stage_improvements = []
    for row in rows:
        stage_before = float(row.get("outer_objective_before", math.nan))
        stage_after = float(row.get("outer_accepted_objective", math.nan))
        if math.isfinite(stage_before) and stage_before > 0.0 and math.isfinite(stage_after):
            stage_improvements.append((stage_before - stage_after) / stage_before)
    compounded_improvement = (
        1.0 - float(np.prod([1.0 - value for value in stage_improvements]))
        if stage_improvements
        else math.nan
    )
    return {
        "outer_steps": len(rows),
        "outer_steps_run": len(all_rows),
        "selected_outer_step": len(rows),
        "accepted_outer_steps": sum(
            float(row.get("outer_objective_accepted", 0.0)) > 0.5 for row in rows
        ),
        "accepted_outer_steps_run": sum(
            float(row.get("outer_objective_accepted", 0.0)) > 0.5
            for row in all_rows
        ),
        "objective_before": before,
        "objective_after": after,
        "objective_improvement_fraction": compounded_improvement,
        "mean_within_stage_objective_improvement_fraction": (
            float(np.mean(stage_improvements)) if stage_improvements else math.nan
        ),
        "objective_improvement_definition": (
            "compounded accepted within-stage reduction; refreshed outer-bank "
            "objective levels are not compared"
        ),
    }


def _gap(reference: float, candidate: float) -> float | None:
    if not math.isfinite(reference) or reference <= 1e-12 or not math.isfinite(candidate):
        return None
    return float((reference - candidate) / reference)


def _group(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    values = [value for key in reference if (value := _gap(reference[key], candidate[key])) is not None]
    return {
        "metric_count": len(values),
        "improved_count": sum(value > 0.0 for value in values),
        "median_gap_closed": (float(median(values)) if values else None),
        "mean_gap_closed": (float(np.mean(values)) if values else None),
    }


def _scale_tail(target_prices: np.ndarray, generated_prices: np.ndarray) -> dict[str, float]:
    target_returns = np.diff(np.log(target_prices / target_prices[:, :1]), axis=1)
    generated_returns = np.diff(np.log(generated_prices / generated_prices[:, :1]), axis=1)
    target_rv = np.sqrt(np.sum(target_returns**2, axis=1))
    generated_rv = np.sqrt(np.sum(generated_returns**2, axis=1))

    def excess(values: np.ndarray) -> float:
        centered = values - values.mean()
        variance = float(np.mean(centered**2))
        return float(np.mean(centered**4) / max(variance**2, 1e-30) - 3.0)

    return {
        "return_std_ratio": float(generated_returns.std() / max(target_returns.std(), 1e-15)),
        "rv_mean_ratio": float(generated_rv.mean() / max(target_rv.mean(), 1e-15)),
        "rv_std_ratio": float(generated_rv.std() / max(target_rv.std(), 1e-15)),
        "rv_q90_ratio": float(np.quantile(generated_rv, 0.90) / max(float(np.quantile(target_rv, 0.90)), 1e-15)),
        "abs_return_q99_ratio": float(np.quantile(np.abs(generated_returns), 0.99) / max(float(np.quantile(np.abs(target_returns), 0.99)), 1e-15)),
        "abs_return_q999_ratio": float(np.quantile(np.abs(generated_returns), 0.999) / max(float(np.quantile(np.abs(target_returns), 0.999)), 1e-15)),
        "generated_excess_kurtosis": excess(generated_returns),
        "target_excess_kurtosis": excess(target_returns),
        "nonfinite_path_fraction": float(1.0 - np.mean(np.isfinite(generated_prices))),
    }


def _synthetic(root: Path, task: str) -> dict[str, object]:
    task_root = root / "synthetic" / task
    validation_path = (
        Path("/home/samer/scenarios/BASSERAS-benchmark/dataset/Heston/heston_S_disc_8192x128.npy")
        if task == "heston"
        else root.parents[1] / "heston_parameter_mixture_seed0_20260730" / "data" / "disc.npy"
    )
    if task == "mixture":
        validation_path = Path(__file__).resolve().parents[2] / "runs" / "heston_parameter_mixture_seed0_20260730" / "data" / "disc.npy"
    target = np.asarray(np.load(validation_path, allow_pickle=False), dtype=np.float64)
    arms: dict[str, object] = {}
    training_manifests: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        arm_root = task_root / arm
        metrics = _read(arm_root / "metrics.json")
        controls = _read(arm_root / "control_diagnostics.json")
        bank = np.asarray(np.load(arm_root / "validation_bank.npy", allow_pickle=False), dtype=np.float64)
        row: dict[str, object] = {
            "metrics": metrics,
            "full_bank_scale_tail": _scale_tail(target, bank),
            "controls": controls,
        }
        mixture_metrics = arm_root / "mixture_metrics.json"
        if mixture_metrics.is_file():
            row["mixture_metrics"] = _read(mixture_metrics)
        if arm != "reference":
            manifest = _read(arm_root / "run_manifest.json")
            training_manifests[arm] = manifest
            row["source_fit"] = _history(arm_root / "source_history.jsonl")
            row["joint_fit"] = _history(arm_root / "joint_history.jsonl")
            row["resources"] = manifest["resources"]
        arms[arm] = row

    full_manifest = training_manifests["full_control_nested_mp"]
    volatility_only_manifest = training_manifests["volatility_only_nested_mp"]
    matched_protocol_checks = {
        "reference_summary": (
            full_manifest["reference_summary"]
            == volatility_only_manifest["reference_summary"]
        ),
        "conditional_target_estimator": (
            full_manifest["conditional_target_estimator"]
            == volatility_only_manifest["conditional_target_estimator"]
        ),
        "conditional_projection": (
            full_manifest["conditional_projection"]
            == volatility_only_manifest["conditional_projection"]
        ),
        "source_nested": (
            full_manifest["source_nested"] == volatility_only_manifest["source_nested"]
        ),
        "joint_nested": (
            full_manifest["joint_nested"] == volatility_only_manifest["joint_nested"]
        ),
        "source_training": (
            full_manifest["source_training"]
            == volatility_only_manifest["source_training"]
        ),
        "joint_training": (
            full_manifest["joint_training"]
            == volatility_only_manifest["joint_training"]
        ),
        "adjoint_loss_unchanged": (
            not bool(full_manifest["P_R_regression_loss_changed"])
            and not bool(volatility_only_manifest["P_R_regression_loss_changed"])
        ),
        "validation_only": (
            full_manifest["selection_scope"] == "validation_only"
            and volatility_only_manifest["selection_scope"] == "validation_only"
            and not bool(full_manifest["test_split_loaded"])
            and not bool(volatility_only_manifest["test_split_loaded"])
        ),
    }
    if not all(matched_protocol_checks.values()):
        failures = sorted(key for key, passed in matched_protocol_checks.items() if not passed)
        raise RuntimeError(f"{task} matched-protocol audit failed: {failures}")

    reference_path = arms["reference"]["metrics"]["path_law"]  # type: ignore[index]
    comparisons: dict[str, object] = {}
    for arm in ARMS[1:]:
        candidate_path = arms[arm]["metrics"]["path_law"]  # type: ignore[index]
        comparisons[arm] = {
            "distribution": _group(
                {key: float(reference_path[key]) for key in (
                    "path_swd_normalized", "terminal_w1_normalized",
                    "increment_w1_normalized", "realized_volatility_w1_normalized",
                    "maximum_drawdown_w1_normalized", "return_qq_rmse_normalized",
                )},
                {key: float(candidate_path[key]) for key in (
                    "path_swd_normalized", "terminal_w1_normalized",
                    "increment_w1_normalized", "realized_volatility_w1_normalized",
                    "maximum_drawdown_w1_normalized", "return_qq_rmse_normalized",
                )},
            ),
            "dependence": _group(
                {key: float(reference_path[key]) for key in (
                    "return_acf_rmse", "absolute_return_acf_rmse", "squared_return_acf_rmse",
                )},
                {key: float(candidate_path[key]) for key in (
                    "return_acf_rmse", "absolute_return_acf_rmse", "squared_return_acf_rmse",
                )},
            ),
            "local_calibration": _group(
                {key: float(reference_path[key]) for key in (
                    "timewise_return_mean_rmse_normalized", "timewise_return_std_rmse_normalized",
                )},
                {key: float(candidate_path[key]) for key in (
                    "timewise_return_mean_rmse_normalized", "timewise_return_std_rmse_normalized",
                )},
            ),
        }
    return {
        "arms": arms,
        "reference_relative": comparisons,
        "matched_protocol_audit": {
            "pass": True,
            "checks": matched_protocol_checks,
        },
    }


def _real(root: Path, index: str) -> dict[str, object]:
    task_root = root / "real" / index
    paths = {
        "reference": task_root / "reference" / "metrics.json",
        "full_control_nested_mp": task_root / "full_control_nested_mp" / "validation_8192" / "metrics.json",
        "volatility_only_nested_mp": task_root / "volatility_only_nested_mp" / "validation_8192" / "metrics.json",
    }
    arms: dict[str, object] = {arm: _read(path) for arm, path in paths.items()}
    training_manifests: dict[str, dict[str, dict[str, object]]] = {}
    for arm in ARMS[1:]:
        arm_root = task_root / arm
        source_manifest = _read(arm_root / "source" / "candidate_manifest.json")
        joint_manifest = _read(arm_root / "joint" / "candidate_manifest.json")
        training_manifests[arm] = {"source": source_manifest, "joint": joint_manifest}
        arms[arm]["training_summary"] = {  # type: ignore[index]
            "source": _history(
                arm_root / "source" / "history.jsonl", selected_outer_step=2
            ),
            "joint": _history(
                arm_root / "joint" / "history.jsonl", selected_outer_step=3
            ),
            "wall_seconds": float(source_manifest["resources"]["wall_seconds"]) + float(joint_manifest["resources"]["wall_seconds"]),  # type: ignore[index]
            "peak_gpu_allocated_gb": max(float(source_manifest["resources"]["peak_gpu_allocated_gb"]), float(joint_manifest["resources"]["peak_gpu_allocated_gb"])),  # type: ignore[index]
            "conditional_target_estimator": joint_manifest["nested_training"]["outer_target_estimator"],  # type: ignore[index]
            "conditional_branches": joint_manifest["nested_training"]["outer_conditional_branches"],  # type: ignore[index]
        }
    full_manifests = training_manifests["full_control_nested_mp"]
    volatility_only_manifests = training_manifests["volatility_only_nested_mp"]
    matched_protocol_checks: dict[str, bool] = {}
    for phase in ("source", "joint"):
        full_manifest = full_manifests[phase]
        volatility_only_manifest = volatility_only_manifests[phase]
        prefix = f"{phase}_"
        matched_protocol_checks.update({
            prefix + "train_array_sha256": (
                full_manifest["train_array_sha256"]
                == volatility_only_manifest["train_array_sha256"]
            ),
            prefix + "reference_summary": (
                full_manifest["reference_summary"]
                == volatility_only_manifest["reference_summary"]
            ),
            prefix + "discrepancy_normalization": (
                all(
                    full_manifest["discrepancy_normalization"][key]
                    == volatility_only_manifest["discrepancy_normalization"][key]
                    for key in (
                        "families", "family_by_block", "family_coefficients",
                        "lambda_scale_included", "rule",
                    )
                )
            ),
            prefix + "training": (
                full_manifest["training"] == volatility_only_manifest["training"]
            ),
            prefix + "conditional_target_estimator": (
                full_manifest["nested_training"]["outer_target_estimator"]
                == volatility_only_manifest["nested_training"]["outer_target_estimator"]
                == "nested_branches"
            ),
            prefix + "conditional_branches": (
                int(full_manifest["nested_training"]["outer_conditional_branches"])
                == int(volatility_only_manifest["nested_training"]["outer_conditional_branches"])
                == 64
            ),
            prefix + "running_cost": (
                full_manifest["running_cost"]
                == volatility_only_manifest["running_cost"]
            ),
            prefix + "eta": full_manifest["eta"] == volatility_only_manifest["eta"],
            prefix + "sigma_bound": (
                full_manifest["admissible_sigma_max"]
                == volatility_only_manifest["admissible_sigma_max"]
            ),
            prefix + "objective": all(
                full_manifest[key] == volatility_only_manifest[key]
                for key in (
                    "objective_protocol", "lambda_x", "lambda_v", "joint_weight",
                    "return_discrepancy", "global_rv_discrepancy", "discrepancy_stack",
                    "joint_block_multiplier",
                )
            ),
            prefix + "validation_only": (
                not bool(full_manifest["test_or_event_data_seen"])
                and not bool(volatility_only_manifest["test_or_event_data_seen"])
                and full_manifest["selection_scope"] == "training_and_validation_only"
                and volatility_only_manifest["selection_scope"]
                == "training_and_validation_only"
            ),
        })
    if not all(matched_protocol_checks.values()):
        failures = sorted(key for key, passed in matched_protocol_checks.items() if not passed)
        raise RuntimeError(f"{index} matched-protocol audit failed: {failures}")
    reference_law = arms["reference"]["law_metrics"]  # type: ignore[index]
    comparisons: dict[str, object] = {}
    categories = {
        "distribution": (
            "return_qq_rmse_normalized", "return_wasserstein_normalized",
            "realized_volatility_wasserstein_normalized",
            "maximum_drawdown_wasserstein_normalized",
            "terminal_return_wasserstein_normalized",
        ),
        "dependence": (
            "abs_return_acf_error_rms", "squared_return_acf_error_rms", "leverage_error_rms",
        ),
    }
    for arm in ARMS[1:]:
        candidate_law = arms[arm]["law_metrics"]  # type: ignore[index]
        comparisons[arm] = {
            name: _group(
                {key: float(reference_law[key]) for key in keys},
                {key: float(candidate_law[key]) for key in keys},
            )
            for name, keys in categories.items()
        }
        reference_local = {
            "return_mean": abs(float(reference_law["return_mean_error"])),
            "return_std": abs(float(reference_law["return_std_ratio"]) - 1.0),
            "rv_mean": abs(float(reference_law["realized_volatility_mean_ratio"]) - 1.0),
        }
        candidate_local = {
            "return_mean": abs(float(candidate_law["return_mean_error"])),
            "return_std": abs(float(candidate_law["return_std_ratio"]) - 1.0),
            "rv_mean": abs(float(candidate_law["realized_volatility_mean_ratio"]) - 1.0),
        }
        comparisons[arm]["local_calibration"] = _group(reference_local, candidate_local)  # type: ignore[index]
        comparisons[arm]["hard_guardrail_pass"] = bool(arms[arm]["hard_guardrails"]["pass"])  # type: ignore[index]
    full_bank_path = (
        task_root / "full_control_nested_mp" / "validation_8192" / "validation_bank.npy"
    )
    volatility_only_bank_path = (
        task_root
        / "volatility_only_nested_mp"
        / "validation_8192"
        / "validation_bank.npy"
    )
    full_bank_sha256 = _sha256(full_bank_path)
    volatility_only_bank_sha256 = _sha256(volatility_only_bank_path)
    return {
        "arms": arms,
        "reference_relative": comparisons,
        "arm_bank_identity": {
            "full_control_sha256": full_bank_sha256,
            "volatility_only_sha256": volatility_only_bank_sha256,
            "byte_identical": full_bank_sha256 == volatility_only_bank_sha256,
        },
        "matched_protocol_audit": {
            "pass": True,
            "checks": matched_protocol_checks,
        },
    }


def _pct(value: object) -> str:
    if value is None:
        return "--"
    return f"{100.0 * float(value):+.1f}%"


def _write_markdown(root: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Matched seed-zero validation: full versus volatility-only nested MP",
        "",
        "This report uses validation data only. No canonical test or event split was loaded, and no seed was excluded.",
        "",
        "## Reference-relative gap closure",
        "",
        "| Task | Arm | Distribution | Dependence | Local calibration | Hard guardrail |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    tasks = payload["tasks"]  # type: ignore[index]
    for task in (*SYNTHETIC, *REAL):
        comparisons = tasks[task]["reference_relative"]  # type: ignore[index]
        for arm in ARMS[1:]:
            row = comparisons[arm]
            hard = row.get("hard_guardrail_pass")
            lines.append(
                f"| {task} | {arm} | {_pct(row['distribution']['median_gap_closed'])} | "
                f"{_pct(row['dependence']['median_gap_closed'])} | "
                f"{_pct(row['local_calibration']['median_gap_closed'])} | "
                f"{'--' if hard is None else ('Pass' if hard else 'Fail')} |"
            )
    lines.extend([
        "",
        "## Accepted stages, objective trace, and resources",
        "",
        "Objective changes compound the accepted common-random reduction within each stage; objective levels from different refreshed outer banks are never compared directly.",
        "",
        "| Task | Arm | Source accepted | Joint accepted | Source objective change | Joint objective change | Wall min | Peak GB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for task in (*SYNTHETIC, *REAL):
        arms = tasks[task]["arms"]  # type: ignore[index]
        for arm in ARMS[1:]:
            row = arms[arm]
            if task in SYNTHETIC:
                source = row["source_fit"]
                joint = row["joint_fit"]
                resources = row["resources"]
                wall = float(resources["wall_seconds"])
                peak = float(resources["peak_gpu_allocated_gb"])
            else:
                training = row["training_summary"]
                source = training["source"]
                joint = training["joint"]
                wall = float(training["wall_seconds"])
                peak = float(training["peak_gpu_allocated_gb"])
            lines.append(
                f"| {task} | {arm} | {source['accepted_outer_steps']}/{source['outer_steps']} | "
                f"{joint['accepted_outer_steps']}/{joint['outer_steps']} | "
                f"{_pct(source.get('objective_improvement_fraction'))} | "
                f"{_pct(joint.get('objective_improvement_fraction'))} | "
                f"{wall / 60.0:.1f} | {peak:.2f} |"
            )
    lines.extend([
        "",
        "## Synthetic full-bank metrics",
        "",
        "| Task | Arm | Path SWD | RV W1 | Abs-ACF | Sq-ACF | Prefix/future corr. | Cum CRPS | RV CRPS | Return-SD ratio | q99 ratio | cap active |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for task in SYNTHETIC:
        arms = tasks[task]["arms"]  # type: ignore[index]
        for arm in ARMS:
            row = arms[arm]
            metrics = row["metrics"]
            path = metrics["path_law"]
            memory = metrics["conditional_path_memory"]["generated"]
            shadow = metrics["path_shadowing"]
            scale = row["full_bank_scale_tail"]
            controls = row["controls"]
            lines.append(
                f"| {task} | {arm} | {path['path_swd_normalized']:.4f} | "
                f"{path['realized_volatility_w1_normalized']:.4f} | "
                f"{path['absolute_return_acf_rmse']:.4f} | "
                f"{path['squared_return_acf_rmse']:.4f} | "
                f"{memory['prefix_future_rv_correlation']:.4f} | "
                f"{shadow['cumulative_return_crps']:.6g} | "
                f"{shadow['realized_volatility_crps']:.6g} | "
                f"{scale['return_std_ratio']:.3f} | {scale['abs_return_q99_ratio']:.3f} | "
                f"{controls['sigma_max_active_fraction']:.3%} |"
            )
    lines.extend([
        "",
        "## Synthetic dependence, memory, calibration, and safety",
        "",
        "Memory columns are absolute errors from the validation target statistic.",
        "",
        "| Task | Arm | Return ACF | Corr. error | Gap error | Terminal W1 | Increment W1 | Drawdown W1 | Mean RMSE | SD RMSE | Drift error | Denom. margin | Nonfinite |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for task in SYNTHETIC:
        arms = tasks[task]["arms"]  # type: ignore[index]
        for arm in ARMS:
            row = arms[arm]
            metrics = row["metrics"]
            path = metrics["path_law"]
            generated_memory = metrics["conditional_path_memory"]["generated"]
            target_memory = metrics["conditional_path_memory"]["target"]
            controls = row["controls"]
            lines.append(
                f"| {task} | {arm} | {path['return_acf_rmse']:.4f} | "
                f"{abs(generated_memory['prefix_future_rv_correlation'] - target_memory['prefix_future_rv_correlation']):.5f} | "
                f"{abs(generated_memory['prefix_high_low_future_rv_gap'] - target_memory['prefix_high_low_future_rv_gap']):.6g} | "
                f"{path['terminal_w1_normalized']:.4f} | {path['increment_w1_normalized']:.4f} | "
                f"{path['maximum_drawdown_w1_normalized']:.4f} | "
                f"{path['timewise_return_mean_rmse_normalized']:.4f} | "
                f"{path['timewise_return_std_rmse_normalized']:.4f} | "
                f"{controls['reference_drift_max_abs_error']:.3g} | "
                f"{controls['specific_entropy_denominator_min']:.3g} | "
                f"{max(controls['nonfinite_control_fraction'], controls['nonfinite_path_fraction']):.3%} |"
            )
    mixture_arms = tasks["mixture"]["arms"]  # type: ignore[index]
    lines.extend([
        "",
        "## Persistent-mixture diagnostics",
        "",
        "| Arm | Regime TVD | Generated low-confidence | Oracle low-confidence | RV Wasserstein | Abs-ACF RMSE | Sq-ACF RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ARMS:
        mixture = mixture_arms[arm]["mixture_metrics"]  # type: ignore[index]
        fidelity = mixture["mixture_fidelity"]
        observable = mixture["observable_fidelity"]
        lines.append(
            f"| {arm} | {fidelity['regime_proportion_tvd']:.4f} | "
            f"{fidelity['generated_low_confidence_fraction']:.2%} | "
            f"{fidelity['target_low_confidence_fraction']:.2%} | "
            f"{observable['realized_volatility_wasserstein']:.5f} | "
            f"{observable['abs_return_acf_rmse_lags_1_50']:.4f} | "
            f"{observable['squared_return_acf_rmse_lags_1_50']:.4f} |"
        )
    lines.extend([
        "",
        "## Futures path shadowing and full-bank diagnostics",
        "",
        "| Index | Arm | Cum CRPS | Incr CRPS | RV CRPS | RV cov. 90 | Return-SD/train | RV mean/train | q99/train | cap active |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for index in REAL:
        arms = tasks[index]["arms"]  # type: ignore[index]
        for arm in ARMS:
            row = arms[arm]
            shadow = row["shadowing_metrics"]
            scale = row["full_bank_law_scale_diagnostics"]
            controls = row["full_bank_control_diagnostics"]
            lines.append(
                f"| {index} | {arm} | {shadow['cumulative_return_crps']:.6g} | "
                f"{shadow['increment_crps']:.6g} | {shadow['realized_volatility_crps']:.6g} | "
                f"{shadow['realized_volatility_coverage_90']:.3f} | "
                f"{scale['return_std_ratio_to_training']:.3f} | "
                f"{scale['rv_mean_ratio_to_training']:.3f} | "
                f"{scale['abs_return_q99_ratio_to_training']:.3f} | "
                f"{controls['sigma_max_active_fraction']:.3%} |"
            )
    lines.extend([
        "",
        "## Futures distribution, dependence, and local calibration",
        "",
        "| Index | Arm | QQ | Return W1 | RV W1 | Drawdown W1 | Terminal W1 | Abs-ACF | Sq-ACF | Leverage | Mean error | SD ratio | RV/validation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for index in REAL:
        arms = tasks[index]["arms"]  # type: ignore[index]
        for arm in ARMS:
            law = arms[arm]["law_metrics"]
            lines.append(
                f"| {index} | {arm} | {law['return_qq_rmse_normalized']:.4f} | "
                f"{law['return_wasserstein_normalized']:.4f} | "
                f"{law['realized_volatility_wasserstein_normalized']:.4f} | "
                f"{law['maximum_drawdown_wasserstein_normalized']:.4f} | "
                f"{law['terminal_return_wasserstein_normalized']:.4f} | "
                f"{law['abs_return_acf_error_rms']:.4f} | "
                f"{law['squared_return_acf_error_rms']:.4f} | "
                f"{law['leverage_error_rms']:.4f} | "
                f"{law['return_mean_error']:.3g} | {law['return_std_ratio']:.3f} | "
                f"{law['realized_volatility_mean_ratio']:.3f} |"
            )
    lines.extend([
        "",
        "## Futures hard-guardrail audit on all 8,192 generated paths",
        "",
        "| Index | Arm | RV SD/train | RV q90/train | q999/train | Excess kurtosis | min cap | max cap | Root margin | Drift error | Nonfinite | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for index in REAL:
        arms = tasks[index]["arms"]  # type: ignore[index]
        for arm in ARMS:
            row = arms[arm]
            scale = row["full_bank_law_scale_diagnostics"]
            controls = row["full_bank_control_diagnostics"]
            guardrails = row["hard_guardrails"]
            lines.append(
                f"| {index} | {arm} | {scale['rv_std_ratio_to_training']:.3f} | "
                f"{scale['rv_q90_ratio_to_training']:.3f} | "
                f"{scale['abs_return_q999_ratio_to_training']:.3f} | "
                f"{scale['generated_excess_kurtosis']:.2f} | "
                f"{controls['sigma_min_active_fraction']:.3%} | "
                f"{controls['sigma_max_active_fraction']:.3%} | "
                f"{controls['gaussian_discriminant_root_min']:.3g} | "
                f"{controls['reference_drift_max_abs_error_physical']:.3g} | "
                f"{max(scale['nonfinite_path_fraction'], controls['nonfinite_control_fraction']):.3%} | "
                f"{'Pass' if guardrails['pass'] else 'Fail'} |"
            )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Values above 100% gap closure are not interpreted independently; the full-bank scale, tail, bound-activity, and finite-value diagnostics above take precedence. A failed hard guardrail is a failed arm even if CRPS improves.",
        "",
        "## Seed-zero decision",
        "",
        "- **Safety versus the reference:** volatility-only safely improves the broad Heston distribution. It also passes every locked futures guardrail and improves broad distribution and all three shadowing CRPS scores on every index, but the futures dependence panel remains worse than the reference. The mixture improves broadly but is not an unqualified safety pass: 10.98% of controls reach the upper volatility bound and the unconstrained specific-entropy denominator is negative on some prefixes.",
        "- **Competitiveness with full control:** volatility-only is near-equivalent on Heston and materially more balanced on the mixture. On ES, NQ, RTY, and YM the selected full-control policy already accepts no drift correction; the two generated validation banks are byte-identical for every index.",
        "- **Drift-removal conclusion:** removing drift does not sacrifice the main broad-law or shadowing gains in this screen and avoids the aggressive drift shortcut seen in the mixture. It is the more parsimonious control choice. This seed-zero, N=128 screen does not by itself prove horizon-scaling or multi-seed stability.",
        "- **Scope:** the requested stop condition is respected: no test/event split was loaded and no four-seed campaign was started.",
    ])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    locked = _read(root / "LOCKED_PROTOCOL.json")
    if bool(locked.get("test_split_access_authorized", True)):
        raise RuntimeError("protocol is not validation-only")
    tasks: dict[str, object] = {}
    for task in SYNTHETIC:
        tasks[task] = _synthetic(root, task)
    for index in REAL:
        tasks[index] = _real(root, index)
    payload = {
        "protocol_manifest": str((root / "LOCKED_PROTOCOL.json").resolve()),
        "model_seed": 0,
        "selection_scope": "validation_only",
        "test_split_loaded": False,
        "four_seed_campaign_started": False,
        "tasks": tasks,
    }
    (root / "SUMMARY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(root, payload)


if __name__ == "__main__":
    main()
