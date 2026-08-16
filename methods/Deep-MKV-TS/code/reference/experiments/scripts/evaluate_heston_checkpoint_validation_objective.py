#!/usr/bin/env python3
"""Evaluate one Heston checkpoint trajectory on a paired validation objective."""

from __future__ import annotations

import argparse
import hashlib
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

from deep_mkv_gen_path_dt import DiscreteTimeGrid, fit_local_gaussian_reference_kernel  # noqa: E402
from deep_mkv_gen_path_dt.ce import RidgeConditionalExpectation  # noqa: E402
from deep_mkv_gen_path_dt.controls import ReferenceDriftSpecificEntropyDiagonalControl  # noqa: E402
from deep_mkv_gen_path_dt.controls.base import RunningCostInputs  # noqa: E402
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_heston_mixture_deep_mkv_ablation import _load_log_paths  # noqa: E402
from run_matched_control_synthetic_validation import _model  # noqa: E402


DEFAULT_RUN_DIR = REPO_ROOT / "runs" / (
    "heston_online_fitted_drift_volatility_only_50_100_trajectory_seed0_20260805"
)
TRAIN_PATH = Path(
    "/home/samer/scenarios/BASSERAS-benchmark/dataset/Heston/heston_S_8192x128.npy"
)
VALIDATION_PATH = Path(
    "/home/samer/scenarios/BASSERAS-benchmark/dataset/Heston/heston_S_disc_8192x128.npy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--paths-per-replicate", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--lambda-scale", type=float, default=50.0)
    parser.add_argument("--kappa-scale", type=float, default=100.0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_se(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "standard_deviation": float(tensor.std(unbiased=True).item()),
        "standard_error": float(
            (tensor.std(unbiased=True) / math.sqrt(float(len(values)))).item()
        ),
    }


def _external_errors(metrics: dict[str, object]) -> dict[str, float]:
    path = metrics["path_law"]
    memory = metrics["conditional_path_memory"]
    if not isinstance(path, dict) or not isinstance(memory, dict):
        raise ValueError("invalid path-law metrics")
    target = memory["target"]
    generated = memory["generated"]
    if not isinstance(target, dict) or not isinstance(generated, dict):
        raise ValueError("invalid conditional-memory metrics")
    return {
        "path_swd": float(path["path_swd_normalized"]),
        "terminal": float(path["terminal_w1_normalized"]),
        "increment": float(path["increment_w1_normalized"]),
        "rv": float(path["realized_volatility_w1_normalized"]),
        "mdd": float(path["maximum_drawdown_w1_normalized"]),
        "qq": float(path["return_qq_rmse_normalized"]),
        "abs_acf": float(path["absolute_return_acf_rmse"]),
        "squared_acf": float(path["squared_return_acf_rmse"]),
        "conditional_correlation": abs(
            float(generated["prefix_future_rv_correlation"])
            - float(target["prefix_future_rv_correlation"])
        ),
        "delayed_gap": abs(
            float(generated["prefix_high_low_future_rv_gap"])
            - float(target["prefix_high_low_future_rv_gap"])
        ),
        "timewise_mean": float(path["timewise_return_mean_rmse_normalized"]),
        "timewise_std": float(path["timewise_return_std_rmse_normalized"]),
    }


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _eligibility(
    *,
    reference_errors: dict[str, float],
    metrics: dict[str, object],
    controls: dict[str, object],
) -> dict[str, object]:
    errors = _external_errors(metrics)
    closure = {
        name: (float(reference_errors[name]) - value) / float(reference_errors[name])
        for name, value in errors.items()
    }
    distribution = _median(
        [
            closure[name]
            for name in ("path_swd", "terminal", "increment", "rv", "mdd", "qq")
        ]
    )
    dependence = _median(
        [
            closure[name]
            for name in (
                "abs_acf",
                "squared_acf",
                "conditional_correlation",
                "delayed_gap",
            )
        ]
    )
    local = _median([closure["timewise_mean"], closure["timewise_std"]])
    scale_tail = metrics["full_bank_scale_tail"]
    if not isinstance(scale_tail, dict):
        raise ValueError("invalid scale-tail metrics")
    checks = {
        "finite": (
            float(controls["nonfinite_control_fraction"]) == 0.0
            and float(controls["nonfinite_path_fraction"]) == 0.0
        ),
        "upper_cap_at_most_one_percent": float(controls["sigma_max_active_fraction"])
        <= 0.01,
        "positive_entropy_denominator": float(
            controls["specific_entropy_denominator_min"]
        )
        > 0.0,
        "distribution_improves": distribution > 0.0,
        "dependence_improves": dependence > 0.0,
        "local_degradation_at_most_ten_percent": local >= -0.10,
        "return_scale_within_ten_percent": 0.90
        <= float(scale_tail["return_std_ratio"])
        <= 1.10,
        "rv_mean_within_ten_percent": 0.90
        <= float(scale_tail["realized_volatility_mean_ratio"])
        <= 1.10,
        "q999_tail_within_twenty_percent": 0.80
        <= float(scale_tail["absolute_return_q999_ratio"])
        <= 1.20,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "distribution_median_gap_closed": distribution,
        "dependence_median_gap_closed": dependence,
        "local_median_gap_closed": local,
    }


def main() -> None:
    args = parse_args()
    if int(args.replicates) < 2:
        raise ValueError("replicates must be at least two")
    if int(args.paths_per_replicate) < 2:
        raise ValueError("paths-per-replicate must be at least two")
    if float(args.lambda_scale) <= 0.0 or float(args.kappa_scale) < 0.0:
        raise ValueError("objective weights are invalid")

    run_dir = args.run_dir.resolve()
    arm_dir = run_dir / "volatility_only_online_mp"
    checkpoint_root = arm_dir / "training_checkpoints"
    checkpoint_paths = sorted(checkpoint_root.glob("step_*.pt"))
    if not checkpoint_paths:
        raise RuntimeError("no source checkpoints were found")
    output_dir = ensure_run_dir(
        (args.output_dir or (arm_dir / "validation_objective_audit")).resolve()
    )
    checkpoint_hashes_before = {
        path.name: _sha256(path) for path in checkpoint_paths
    }

    device = torch.device(args.device)
    train_paths = _load_log_paths(TRAIN_PATH, device=device, dtype=torch.float32)
    validation_prices = np.asarray(
        np.load(VALIDATION_PATH, allow_pickle=False), dtype=np.float64
    )
    validation_log = torch.as_tensor(
        np.log(validation_prices / validation_prices[:, :1])[..., None],
        device=device,
        dtype=torch.float32,
    )
    num_steps = int(train_paths.shape[1]) - 1
    grid = DiscreteTimeGrid(T=num_steps * 0.004, num_steps=num_steps)
    reference = fit_local_gaussian_reference_kernel(
        target_paths=train_paths,
        grid=grid,
        ridge=1e-3,
        variance_ridge=1e-3,
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    discrepancy = build_heston_discrepancy(
        num_steps=num_steps,
        include_acf=True,
        preset="old_fullv_w0p25",
        lambda_scale=float(args.lambda_scale),
        kappa_scale=float(args.kappa_scale),
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )
    control = ReferenceDriftSpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=1.0,
        reference_kernel=reference,
        sigma_min=1e-3,
        sigma_max=0.6,
        allow_drift_correction=False,
    )
    model = _model(
        grid=grid,
        control=control,
        discrepancy=discrepancy,
        estimator=RidgeConditionalExpectation(ridge=1e-3),
        device=device,
        seed=0,
        adjoint_network="gru",
        transformer_layers=2,
        transformer_heads=4,
    )
    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.zero_()
    reference_checkpoint = model.checkpoint_state()

    candidates: list[tuple[str, int | None, dict[str, object]]] = [
        ("reference", None, reference_checkpoint)
    ]
    for checkpoint_path in checkpoint_paths:
        step = int(checkpoint_path.stem.split("_")[1])
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        candidates.append((f"step_{step:04d}", step, checkpoint))

    generator = np.random.default_rng(int(args.seed))
    target_indices = [
        generator.choice(
            len(validation_log),
            size=int(args.paths_per_replicate),
            replace=False,
        )
        for _ in range(int(args.replicates))
    ]
    rollout_seeds = [int(args.seed) + 10_000 + index for index in range(int(args.replicates))]
    values: dict[str, dict[str, list[float]]] = {
        name: {"discrepancy": [], "running_cost": [], "total": []}
        for name, _, _ in candidates
    }
    x0 = torch.zeros((1, 1), device=device, dtype=torch.float32)
    for replicate, (indices, rollout_seed) in enumerate(
        zip(target_indices, rollout_seeds, strict=True), start=1
    ):
        target = validation_log.index_select(
            0, torch.as_tensor(indices, device=device, dtype=torch.long)
        )
        for name, step, checkpoint in candidates:
            model.load_checkpoint_state(checkpoint)
            with torch.no_grad():
                sample = model.sample(
                    num_paths=int(args.paths_per_replicate),
                    x0=x0,
                    seed=int(rollout_seed),
                )
                discrepancy_result = discrepancy(
                    generated_paths=sample.paths,
                    target_paths=target,
                    grid=grid,
                )
                running_result = model.running_cost(
                    RunningCostInputs(
                        paths=sample.paths,
                        controls=sample.controls,
                        grid=grid,
                        parameters=model.parameters,
                    )
                )
                discrepancy_value = float(args.lambda_scale) * float(
                    discrepancy_result.value.item()
                )
                running_value = float(running_result.value.item())
                total = discrepancy_value + running_value
            values[name]["discrepancy"].append(discrepancy_value)
            values[name]["running_cost"].append(running_value)
            values[name]["total"].append(total)
            print(
                f"replicate={replicate}/{args.replicates} candidate={name} "
                f"total={total:.8f}",
                flush=True,
            )

    reference_metrics = json.loads(
        (run_dir / "reference" / "metrics.json").read_text(encoding="utf-8")
    )
    reference_errors = _external_errors(reference_metrics)
    rows: dict[str, dict[str, object]] = {}
    for name, step, _ in candidates:
        objective = {key: _mean_se(component) for key, component in values[name].items()}
        row: dict[str, object] = {
            "step": step,
            "objective": objective,
            "replicate_values": values[name],
        }
        if step is not None:
            evaluation_dir = arm_dir / "checkpoint_evaluations" / f"step_{step:04d}"
            metrics = json.loads(
                (evaluation_dir / "metrics.json").read_text(encoding="utf-8")
            )
            controls = json.loads(
                (evaluation_dir / "control_diagnostics.json").read_text(
                    encoding="utf-8"
                )
            )
            row["eligibility"] = _eligibility(
                reference_errors=reference_errors,
                metrics=metrics,
                controls=controls,
            )
        rows[name] = row

    eligible_names = [
        name
        for name, row in rows.items()
        if row.get("eligibility", {}).get("pass", False)  # type: ignore[union-attr]
    ]
    if not eligible_names:
        selected = "reference"
        best = "reference"
        paired_differences: dict[str, object] = {}
    else:
        best = min(
            eligible_names,
            key=lambda name: float(rows[name]["objective"]["total"]["mean"]),  # type: ignore[index]
        )
        best_values = values[best]["total"]
        paired_differences = {}
        within_one_se = []
        for name in eligible_names:
            differences = [
                candidate - best_value
                for candidate, best_value in zip(
                    values[name]["total"], best_values, strict=True
                )
            ]
            summary = _mean_se(differences)
            paired_differences[name] = summary
            if float(summary["mean"]) <= float(summary["standard_error"]):
                within_one_se.append(name)
        selected = min(within_one_se, key=lambda name: int(rows[name]["step"]))

    checkpoint_hashes_after = {
        path.name: _sha256(path) for path in checkpoint_paths
    }
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise RuntimeError("a source checkpoint changed during the objective audit")

    report = {
        "protocol": {
            "selection_scope": "validation_only",
            "test_split_loaded": False,
            "training_path": str(TRAIN_PATH),
            "validation_path": str(VALIDATION_PATH),
            "lambda_scale": float(args.lambda_scale),
            "kappa_scale": float(args.kappa_scale),
            "replicates": int(args.replicates),
            "paths_per_replicate": int(args.paths_per_replicate),
            "common_rollout_seeds": rollout_seeds,
            "common_target_indices": True,
            "checkpoint_nonmutation_verified": True,
        },
        "rows": rows,
        "decision": {
            "eligible": eligible_names,
            "minimum_mean_objective": best,
            "paired_difference_from_minimum": paired_differences,
            "one_standard_error_selected": selected,
        },
        "checkpoint_sha256": checkpoint_hashes_after,
    }
    write_json(output_dir / "SUMMARY.json", report)

    lines = [
        "# Heston checkpoint validation-objective audit",
        "",
        "All values use validation-only target subsets and common random numbers.",
        "",
        "| Checkpoint | Discrepancy | Running cost | Total objective | SE | Eligible |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for name, _, _ in candidates:
        row = rows[name]
        objective = row["objective"]
        eligible = (
            "—"
            if name == "reference"
            else ("yes" if row["eligibility"]["pass"] else "no")
        )
        lines.append(
            "| {name} | {disc:.6f} | {cost:.6f} | {total:.6f} | {se:.6f} | {eligible} |".format(
                name=name,
                disc=float(objective["discrepancy"]["mean"]),
                cost=float(objective["running_cost"]["mean"]),
                total=float(objective["total"]["mean"]),
                se=float(objective["total"]["standard_error"]),
                eligible=eligible,
            )
        )
    lines.extend(
        [
            "",
            f"Minimum mean objective: **{best}**.",
            f"One-standard-error selection: **{selected}**.",
            "",
            "No test split was loaded and no checkpoint was modified.",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote objective audit to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
