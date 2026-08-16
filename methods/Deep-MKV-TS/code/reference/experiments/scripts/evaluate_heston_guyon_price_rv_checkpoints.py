#!/usr/bin/env python3
"""Read-only validation of price-calibrated Guyon Heston checkpoints."""

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
    fit_guyon_lekeufack_price_realized_volatility_reference_kernel,
)
from deep_mkv_gen_path_dt.ce import RidgeConditionalExpectation  # noqa: E402
from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    ReferenceDriftSpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from path_dt_experiments.discrepancies import (  # noqa: E402
    build_heston_discrepancy,
    fit_fixed_target_joint_prefix_future_volatility_mmd,
)
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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if bool(protocol.get("test_split_access_authorized", True)):
        raise RuntimeError("the evaluation must remain validation-only")
    configuration = protocol["configuration"]
    reference_configuration = protocol["reference"]
    run_root = args.run_root.resolve()
    arm_dir = run_root / "volatility_only_online_mp"
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
    reference = fit_guyon_lekeufack_price_realized_volatility_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        realized_volatility_window=int(
            reference_configuration["forward_realized_volatility_window"]
        ),
        ridge=1e-3,
        variance_ridge=1e-3,
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=1e-3,
        sigma_max=0.6,
        calibration_fraction=float(reference_configuration["calibration_fraction"]),
        optimization_steps=int(reference_configuration["optimization_steps"]),
    )
    base = build_heston_discrepancy(
        num_steps=num_steps,
        include_acf=True,
        preset="old_fullv_w0p25",
        lambda_scale=float(configuration["lambda_scale"]),
        kappa_scale=float(configuration["kappa_scale"]),
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
        abs_return_acf_weight=float(configuration["abs_return_acf_weight"]),
        squared_return_acf_weight=float(
            configuration["squared_return_acf_weight"]
        ),
    )
    blocks = list(base.blocks)
    source_joint_weight = float(configuration["source_joint_volatility_weight"])
    if source_joint_weight > 0.0:
        joint = fit_fixed_target_joint_prefix_future_volatility_mmd(
            target_paths,
            grid=grid,
            split_index=64,
            future_horizon=32,
            prefix_windows=(16, 32, 64),
            recent_return_window=16,
            bandwidths=(0.5, 1.0, 2.0),
        )
        blocks.append(
            WeightedDiscrepancy(
                name="volatility_law_joint_prefix_future_rv",
                weight=source_joint_weight,
                discrepancy=joint,
            )
        )
    discrepancy = CompositePathFunctionalDiscrepancy(blocks=tuple(blocks))
    control = ReferenceDriftSpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=float(configuration["eta"]),
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
        seed=int(protocol["seed"]),
        adjoint_network="gru",
        transformer_layers=2,
        transformer_heads=4,
    )

    for step in args.steps:
        checkpoint = arm_dir / "training_checkpoints" / f"step_{int(step):04d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hash = sha256(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_checkpoint_state(state)
        bank, controls = _sample(
            model,
            num_paths=int(args.bank_size),
            batch_size=int(args.sample_batch_size),
            seed=int(configuration["validation_bank_seed"]),
        )
        metrics = _evaluate_common(
            target_prices=validation_prices,
            generated=bank,
            device=device,
            seed=int(protocol["seed"]),
        )
        output = ensure_run_dir(
            arm_dir / "interim_evaluations" / f"step_{int(step):04d}"
        )
        np.save(output / "validation_bank.npy", bank)
        write_json(output / "metrics.json", metrics)
        write_json(output / "control_diagnostics.json", controls)
        if sha256(checkpoint) != checkpoint_hash:
            raise RuntimeError("checkpoint changed during read-only evaluation")
        write_json(
            output / "manifest.json",
            {
                "step": int(step),
                "reference_kind": str(reference_configuration["kind"]),
                "validation_bank_seed": int(configuration["validation_bank_seed"]),
                "validation_bank_paths": int(args.bank_size),
                "common_random_numbers": True,
                "selection_scope": "validation_only",
                "test_split_loaded": False,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_nonmutation_verified": True,
            },
        )
        print(f"evaluated step {int(step)}", flush=True)


if __name__ == "__main__":
    main()
