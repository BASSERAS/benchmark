#!/usr/bin/env python3
"""Held-out decomposition of Monte Carlo, ridge-CE, and GRU P/R errors."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    CausalPathFeatureRidgeConditionalExpectation,
    ConditionalSignalDiagnosticConfig,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
    diagnose_conditional_adjoint_signal,
    fit_causal_volatility_reference_kernel,
)
from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl  # noqa: E402
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from path_dt_experiments.discrepancies import (  # noqa: E402
    build_heston_discrepancy,
    fit_fixed_target_joint_prefix_future_volatility_mmd,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_drawdown_memory_deep_mkv_ablation import (  # noqa: E402
    _build_model,
    _load_dataset,
)


DEFAULT_DATA = (
    REPO_ROOT / "runs" / "drawdown_memory_seed0_comparison_20260730" / "data"
)
DEFAULT_MODEL = (
    REPO_ROOT
    / "runs"
    / "paper_drawdown_fully_generic_4seed_20260801"
    / "seed_0"
    / "generic_ce_nested"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-run-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=73021)
    parser.add_argument("--fit-prefixes", type=int, default=64)
    parser.add_argument("--heldout-prefixes", type=int, default=64)
    parser.add_argument("--branches", type=int, default=128)
    parser.add_argument("--population-batch-size", type=int, default=1024)
    parser.add_argument("--target-batch-size", type=int, default=1024)
    parser.add_argument("--step-indices", type=int, nargs="+", default=(32, 64, 96, 120))
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--wait-for-pid",
        type=int,
        default=None,
        help="Wait for this process to exit before allocating the requested device.",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def prediction_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | None]:
    prediction = prediction.detach().double().reshape(-1)
    target = target.detach().double().reshape(-1)
    residual = prediction - target
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    denominator = torch.sum(target_centered.pow(2))
    prediction_variance = torch.sum(prediction_centered.pow(2))
    eps = torch.finfo(torch.float64).eps
    r2 = None
    if float(denominator.item()) > eps:
        r2 = float((1.0 - torch.sum(residual.pow(2)) / denominator).item())
    correlation = None
    correlation_denominator = torch.sqrt(denominator * prediction_variance)
    if float(correlation_denominator.item()) > eps:
        correlation = float(
            (torch.sum(target_centered * prediction_centered) / correlation_denominator).item()
        )
    slope = None
    intercept = None
    if float(prediction_variance.item()) > eps:
        slope_tensor = torch.sum(prediction_centered * target_centered) / prediction_variance
        slope = float(slope_tensor.item())
        intercept = float((target.mean() - slope_tensor * prediction.mean()).item())
    target_rms = torch.sqrt(torch.mean(target.pow(2))).clamp_min(eps)
    prediction_rms = torch.sqrt(torch.mean(prediction.pow(2)))
    rmse = torch.sqrt(torch.mean(residual.pow(2)))
    return {
        "r2": r2,
        "correlation": correlation,
        "rmse": float(rmse.item()),
        "relative_rmse": float((rmse / target_rms).item()),
        "target_rms": float(target_rms.item()),
        "prediction_rms": float(prediction_rms.item()),
        "prediction_to_target_rms_ratio": float((prediction_rms / target_rms).item()),
        "mean_bias": float(residual.mean().item()),
        "calibration_slope_target_on_prediction": slope,
        "calibration_intercept_target_on_prediction": intercept,
    }


def compact_nested(metrics: dict[str, object], step: int, signal: str) -> dict[str, object]:
    row = metrics["steps"][str(step)][signal]  # type: ignore[index]
    return {
        key: row[key]  # type: ignore[index]
        for key in (
            "estimated_reliability_ratio",
            "nested_mean_signal_to_noise_ratio",
            "nested_mean_noise_standard_error_rms",
            "estimated_branches_for_signal_to_noise_one",
            "split_half_conditional_mean_correlation",
            "split_half_relative_rmse",
        )
    }


def main() -> None:
    args = parse_args()
    if int(args.branches) < 4 or int(args.branches) % 2 != 0:
        raise ValueError("branches must be an even integer >= 4 for the antithetic audit")
    if args.wait_for_pid is not None:
        wait_pid = int(args.wait_for_pid)
        if wait_pid <= 0:
            raise ValueError("wait-for-pid must be positive")
        poll_seconds = float(args.poll_seconds)
        if poll_seconds <= 0.0:
            raise ValueError("poll-seconds must be > 0")
        print(f"waiting for process {wait_pid} to release the GPU", flush=True)
        while True:
            try:
                os.kill(wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(poll_seconds)
        print(f"process {wait_pid} exited; starting the adjoint audit", flush=True)
    run_dir = ensure_run_dir(args.run_dir.resolve())
    device = torch.device(args.device)
    dtype = torch.float32
    target_paths, dataset_manifest, dataset_paths = _load_dataset(
        args.dataset_root.resolve(),
        device=device,
        dtype=dtype,
    )
    configuration = dataset_manifest["configuration"]
    sequence_length = int(configuration["sequence_length"])
    if sequence_length != 128:
        raise ValueError("this diagnostic currently targets the frozen 128-point protocol")
    num_steps = sequence_length - 1
    grid = DiscreteTimeGrid(
        T=num_steps * float(configuration["dt"]),
        num_steps=num_steps,
    )
    reference = fit_causal_volatility_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        rolling_windows=(5, 10, 20, 32),
        ewma_half_lives=(5, 20, 60),
        downside_windows=(10, 32),
        drawdown_thresholds=(),
        drawdown_memory_half_lives=(),
        drawdown_memory_lags=(),
        drawdown_gate_temperature=0.002,
        drift_ridge=1e-3,
        variance_ridges=(1e-5, 1e-4, 1e-3),
        crossfit_folds=3,
        crossfit_seed=1701,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    control = SpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=1.0,
        reference_kernel=reference,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    ce_estimator = CausalPathFeatureRidgeConditionalExpectation(
        grid=grid,
        ridge=float(args.ridge),
        rolling_windows=(5, 10, 20, 32),
        ewma_half_lives=(5, 20, 60),
        downside_windows=(10, 32),
        drawdown_thresholds=(),
        drawdown_memory_half_lives=(),
        drawdown_memory_lags=(),
        drawdown_gate_temperature=0.002,
    )
    base = build_heston_discrepancy(
        num_steps=num_steps,
        include_acf=True,
        preset="old_fullv_w0p25",
        lambda_scale=50.0,
        kappa_scale=50.0,
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )
    joint = fit_fixed_target_joint_prefix_future_volatility_mmd(
        target_paths,
        grid=grid,
        split_index=int(configuration["split_index"]),
        future_horizon=int(configuration["future_horizon"]),
        prefix_windows=(16, 32, 64),
        recent_return_window=16,
        bandwidths=(0.5, 1.0, 2.0),
    )
    discrepancy = CompositePathFunctionalDiscrepancy(
        blocks=tuple(base.blocks)
        + (
            WeightedDiscrepancy(
                name="volatility_law_joint_prefix_future_rv",
                weight=1.0,
                discrepancy=joint,
            ),
        )
    )
    build_args = argparse.Namespace(hidden_dim=96, num_layers=1)
    model = _build_model(
        args=build_args,
        grid=grid,
        control=control,
        discrepancy=discrepancy,
        ce_estimator=ce_estimator,
        device=device,
        dtype=dtype,
        seed=0,
    )
    model_run_dir = args.model_run_dir.resolve()
    checkpoint = torch.load(
        model_run_dir / "model_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_checkpoint_state(checkpoint)
    manifest = json.loads(
        (model_run_dir / "training_manifest.json").read_text(encoding="utf-8")
    )
    training = DiscreteMPTrainingConfig(**manifest["joint_training"])
    diagnostic_training = replace(
        training,
        noise_target_control_variate="adjoint",
    )
    model.training = diagnostic_training

    midpoint = int(target_paths.shape[0]) // 2
    fit_paths = target_paths[:midpoint]
    heldout_paths = target_paths[midpoint:]
    step_indices = tuple(int(value) for value in args.step_indices)
    diagnostic_kwargs = {
        "num_prefixes": int(args.fit_prefixes),
        "num_branches": int(args.branches),
        "population_batch_size": int(args.population_batch_size),
        "target_batch_size": int(args.target_batch_size),
        "step_indices": step_indices,
        "antithetic": True,
    }
    print("estimating high-branch fit conditional targets", flush=True)
    fit_result = diagnose_conditional_adjoint_signal(
        model=model,
        target_paths=fit_paths,
        training=diagnostic_training,
        config=ConditionalSignalDiagnosticConfig(
            **diagnostic_kwargs,
            seed=int(args.seed),
        ),
    )
    print("estimating disjoint held-out conditional targets", flush=True)
    heldout_result = diagnose_conditional_adjoint_signal(
        model=model,
        target_paths=heldout_paths,
        training=diagnostic_training,
        config=ConditionalSignalDiagnosticConfig(
            **{
                **diagnostic_kwargs,
                "num_prefixes": int(args.heldout_prefixes),
            },
            seed=int(args.seed) + 1_000_003,
        ),
    )

    steps: dict[str, object] = {}
    for step in step_indices:
        fit_prefix = fit_result.tensors[f"step_{step}_prefix_paths"].to(device)
        heldout_prefix = heldout_result.tensors[f"step_{step}_prefix_paths"].to(device)
        step_rows: dict[str, object] = {}
        for label, tensor_stem, nested_signal in (
            ("P", "adjoint", "adjoint"),
            ("R", "noise_adjoint_controlled", "noise_adjoint_controlled"),
        ):
            fit_target = fit_result.tensors[
                f"step_{step}_{tensor_stem}_branch_mean"
            ].to(device)
            heldout_target = heldout_result.tensors[
                f"step_{step}_{tensor_stem}_branch_mean"
            ].to(device)
            gru_prediction = heldout_result.tensors[
                f"step_{step}_{tensor_stem}_prediction"
            ].to(device)
            ridge_result = ce_estimator.fit_predict(
                prefixes_train=fit_prefix,
                targets_train=fit_target,
                prefixes_eval=heldout_prefix,
            )
            ridge_prediction = ridge_result.prediction
            step_rows[label] = {
                "monte_carlo": compact_nested(
                    heldout_result.metrics,
                    step,
                    nested_signal,
                ),
                "ridge_ce_vs_nested_mean": prediction_metrics(
                    ridge_prediction,
                    heldout_target,
                ),
                "gru_vs_nested_mean": prediction_metrics(
                    gru_prediction,
                    heldout_target,
                ),
                "gru_vs_ridge_ce": prediction_metrics(
                    gru_prediction,
                    ridge_prediction,
                ),
                "ridge_fit_metrics": ridge_result.metrics,
            }
        step_rows["R_control_variate"] = heldout_result.metrics["steps"][str(step)][  # type: ignore[index]
            "noise_control_variate"
        ]
        steps[str(step)] = step_rows
        print(
            "step={step} P ridge/gru R2={pr}/{pg}; R ridge/gru R2={rr}/{rg}".format(
                step=step,
                pr=step_rows["P"]["ridge_ce_vs_nested_mean"]["r2"],  # type: ignore[index]
                pg=step_rows["P"]["gru_vs_nested_mean"]["r2"],  # type: ignore[index]
                rr=step_rows["R"]["ridge_ce_vs_nested_mean"]["r2"],  # type: ignore[index]
                rg=step_rows["R"]["gru_vs_nested_mean"]["r2"],  # type: ignore[index]
            ),
            flush=True,
        )

    payload = {
        "protocol": {
            "dataset_root": str(args.dataset_root.resolve()),
            "model_run_dir": str(model_run_dir),
            "checkpoint": str((model_run_dir / "model_checkpoint.pt").resolve()),
            "fit_prefix_pool": [0, midpoint],
            "heldout_prefix_pool": [midpoint, int(target_paths.shape[0])],
            "fit_prefixes": int(args.fit_prefixes),
            "heldout_prefixes": int(args.heldout_prefixes),
            "branches": int(args.branches),
            "antithetic": True,
            "effective_independent_branch_pairs": int(args.branches) // 2,
            "population_batch_size": int(args.population_batch_size),
            "target_batch_size": int(args.target_batch_size),
            "step_indices": list(step_indices),
            "ridge": float(args.ridge),
            "noise_target_control_variate": "adjoint",
            "network_parameters_changed": False,
            "test_split_used": False,
        },
        "steps": steps,
        "fit_nested_summary": fit_result.metrics["summary"],
        "heldout_nested_summary": heldout_result.metrics["summary"],
        "sources": {
            key: str(value.resolve()) for key, value in dataset_paths.items()
        },
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(
        {
            "fit": fit_result.tensors,
            "heldout": heldout_result.tensors,
        },
        run_dir / "diagnostic_tensors.pt",
    )
    print(f"wrote conditional-expectation decomposition to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
