#!/usr/bin/env python3
"""Evaluate saved Heston checkpoints under their matched reference/network."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteTimeGrid,
    calibrate_shrunk_causal_reference_kernel,
    fit_causal_volatility_reference_kernel,
    fit_local_gaussian_reference_kernel,
)
from deep_mkv_gen_path_dt.ce import RidgeConditionalExpectation  # noqa: E402
from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    ReferenceDriftSpecificEntropyDiagonalControl,
)
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_heston_mixture_deep_mkv_ablation import _load_log_paths  # noqa: E402
from run_matched_control_synthetic_validation import (  # noqa: E402
    _evaluate_common,
    _model,
    _sample,
)


HESTON_ROOT = Path("/home/samer/scenarios/BASSERAS-benchmark/dataset/Heston")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    parser.add_argument("--lambda-scale", type=float, default=50.0)
    parser.add_argument("--kappa-scale", type=float, default=100.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument(
        "--reference-kind",
        choices=("local_gaussian", "calibrated_causal"),
        default="calibrated_causal",
    )
    parser.add_argument(
        "--adjoint-network",
        choices=("gru", "causal_transformer"),
        default="gru",
    )
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    arm_dir = run_dir / "volatility_only_online_mp"
    checkpoint_paths = {
        int(step): arm_dir / "training_checkpoints" / f"step_{int(step):04d}.pt"
        for step in args.steps
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoints: " + ", ".join(missing))
    hashes_before = {step: sha256(path) for step, path in checkpoint_paths.items()}

    device = torch.device(args.device)
    target_paths = _load_log_paths(
        HESTON_ROOT / "heston_S_8192x128.npy",
        device=device,
        dtype=torch.float32,
    )
    validation_prices = np.asarray(
        np.load(HESTON_ROOT / "heston_S_disc_8192x128.npy", allow_pickle=False),
        dtype=np.float64,
    )
    num_steps = int(target_paths.shape[1]) - 1
    grid = DiscreteTimeGrid(T=num_steps * 0.004, num_steps=num_steps)
    local_reference = fit_local_gaussian_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        ridge=1e-3,
        variance_ridge=1e-3,
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    calibration = None
    if str(args.reference_kind) == "calibrated_causal":
        causal_reference = fit_causal_volatility_reference_kernel(
            target_paths=target_paths,
            grid=grid,
            rolling_windows=(5, 10, 20, 32),
            ewma_half_lives=(5, 20, 60),
            downside_windows=(10, 32),
            drift_ridge=1e-3,
            variance_ridges=(1e-5, 1e-4, 1e-3),
            crossfit_folds=3,
            crossfit_seed=1701,
            sigma_min=1e-3,
            sigma_max=0.6,
        )
        reference, calibration = calibrate_shrunk_causal_reference_kernel(
            target_paths=target_paths,
            causal_reference=causal_reference,
            anchor_reference=local_reference,
            causal_weights=(0.0, 0.25, 0.4, 0.55, 0.7, 0.75, 0.8, 0.85, 1.0),
            calibration_paths=4096,
            calibration_seed=271828,
            offset_steps=3,
            return_std_tolerance=0.02,
            sigma_std_floor=0.045,
            sigma_cap_fraction_limit=0.01,
            clustering_lags=(1, 5, 20),
            sigma_min=1e-3,
            sigma_max=0.6,
        )
    else:
        reference = local_reference
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
        eta=float(args.eta),
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
        seed=int(args.seed),
        adjoint_network=str(args.adjoint_network),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
    )

    evaluations: list[dict[str, object]] = []
    for step, checkpoint_path in checkpoint_paths.items():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_checkpoint_state(checkpoint)
        bank, controls = _sample(
            model,
            num_paths=int(args.bank_size),
            batch_size=int(args.sample_batch_size),
            seed=70_000 + int(args.seed),
        )
        output_dir = ensure_run_dir(
            arm_dir / "checkpoint_evaluations" / f"step_{step:04d}"
        )
        bank_path = output_dir / "validation_bank.npy"
        np.save(bank_path, bank)
        metrics = _evaluate_common(
            target_prices=validation_prices,
            generated=bank,
            device=device,
            seed=int(args.seed),
        )
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "control_diagnostics.json", controls)
        evaluations.append(
            {
                "step": int(step),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": hashes_before[step],
                "validation_bank": str(bank_path),
                "validation_bank_sha256": sha256(bank_path),
                "metrics": str(output_dir / "metrics.json"),
                "control_diagnostics": str(
                    output_dir / "control_diagnostics.json"
                ),
            }
        )
        print(f"evaluated step {step}", flush=True)

    for step, checkpoint_path in checkpoint_paths.items():
        if sha256(checkpoint_path) != hashes_before[step]:
            raise RuntimeError(f"checkpoint {step} changed during evaluation")
    write_json(
        arm_dir / "interim_checkpoint_evaluation_manifest.json",
        {
            "training_performed": False,
            "model_update_performed": False,
            "selection_scope": "validation_only",
            "test_split_loaded": False,
            "reference_kind": str(args.reference_kind),
            "reference_calibration": calibration,
            "adjoint_network": str(args.adjoint_network),
            "transformer_layers": int(args.transformer_layers),
            "transformer_heads": int(args.transformer_heads),
            "common_random_numbers": True,
            "validation_bank_seed": 70_000 + int(args.seed),
            "validation_bank_paths": int(args.bank_size),
            "evaluations": evaluations,
        },
    )


if __name__ == "__main__":
    main()
