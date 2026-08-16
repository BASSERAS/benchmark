#!/usr/bin/env python3
"""Replay a saved Heston source-stage checkpoint on the locked validation bank."""

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
    fit_local_gaussian_reference_kernel,
)
from deep_mkv_gen_path_dt.ce import RidgeConditionalExpectation  # noqa: E402
from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    ReferenceDriftSpecificEntropyDiagonalControl,
)
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402
from path_dt_experiments.runners import write_json  # noqa: E402
from run_heston_mixture_deep_mkv_ablation import _load_log_paths  # noqa: E402
from run_matched_control_synthetic_validation import (  # noqa: E402
    _evaluate_common,
    _model,
    _sample,
)


DEFAULT_RUN_DIR = (
    REPO_ROOT / "runs" / "heston_online_fitted_drift_volatility_only_seed0_20260805"
)
HESTON_ROOT = Path("/home/samer/scenarios/BASSERAS-benchmark/dataset/Heston")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    protocol_path = run_dir / "LOCKED_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if bool(protocol.get("test_split_access_authorized", True)):
        raise RuntimeError("locked protocol does not certify validation-only use")

    arm_dir = run_dir / "volatility_only_online_mp"
    checkpoint_path = arm_dir / "source_model_checkpoint.pt"
    checkpoint_hash_before = _sha256(checkpoint_path)
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
    reference = fit_local_gaussian_reference_kernel(
        target_paths=target_paths,
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
        lambda_scale=50.0,
        kappa_scale=50.0,
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
        seed=int(args.seed),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_checkpoint_state(checkpoint)
    bank, controls = _sample(
        model,
        num_paths=int(args.bank_size),
        batch_size=int(args.sample_batch_size),
        seed=70_000 + int(args.seed),
    )
    metrics = _evaluate_common(
        target_prices=validation_prices,
        generated=bank,
        device=device,
        seed=int(args.seed),
    )

    bank_path = arm_dir / "source_validation_bank.npy"
    metrics_path = arm_dir / "source_metrics.json"
    controls_path = arm_dir / "source_control_diagnostics.json"
    np.save(bank_path, bank)
    write_json(metrics_path, metrics)
    write_json(controls_path, controls)
    checkpoint_hash_after = _sha256(checkpoint_path)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("source checkpoint changed during evaluation")
    write_json(
        arm_dir / "source_evaluation_manifest.json",
        {
            "selection_scope": "validation_only",
            "test_split_loaded": False,
            "training_performed": False,
            "model_update_performed": False,
            "physical_drift": "fitted_reference_fixed",
            "drift_correction_admissible": False,
            "discrepancy_stage": "broad_volatility_features_only",
            "seed": int(args.seed),
            "validation_bank_seed": 70_000 + int(args.seed),
            "validation_bank_paths": int(args.bank_size),
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": checkpoint_hash_before,
            "checkpoint_nonmutation": True,
            "validation_bank_sha256": _sha256(bank_path),
            "metrics": str(metrics_path),
            "control_diagnostics": str(controls_path),
        },
    )
    print(f"wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
