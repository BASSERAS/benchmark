#!/usr/bin/env python3
"""Nested Monte Carlo audit of the real-data complete-MP P/R signal.

This diagnostic holds each simulated prefix fixed, branches independent
Gaussian futures, and estimates the conditional means in the stochastic
maximum principle directly from those branches.  It is intentionally a
read-only checkpoint audit: no parameter is optimized and no validation or
test observation enters the backward target.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

from deep_mkv_gen_path_dt import (  # noqa: E402
    ConditionalSignalDiagnosticConfig,
    DiscreteMPTrainingConfig,
    diagnose_conditional_adjoint_signal,
)
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
)
from path_dt_experiments.discrepancies import (  # noqa: E402
    StabilizedPrefixFutureVolatilityCorrelationDiscrepancy,
    fit_fixed_target_conditional_residual_volatility_mmd,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_basseras_heston_path_shadowing import (  # noqa: E402
    NUM_STEPS,
    _build_components,
    _build_model,
    _load_training_data,
)
from run_real_data_r_transmission_diagnostic import (  # noqa: E402
    _read_json,
    _restore_normalized_discrepancy,
)
from run_real_data_tuning_candidate import (  # noqa: E402
    _component_args,
    _configure_adjoint_network_stack,
    _configure_r_ce_estimator,
    _replace_named_discrepancy_block,
)


DEFAULT_BLOCKS = (
    "path_law_returns",
    "volatility_law_squared_returns",
    "volatility_law_rolling_rv",
    "volatility_law_state_rv",
    "volatility_law_global_rv",
    "volatility_law_abs_return_acf",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Audit this topology-compatible checkpoint instead of the manifest checkpoint.",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help=(
            "Audit the zero-adjoint fitted reference before the first MP stage. "
            "The residual adjoint stack remains at its zero initialization."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefixes", type=int, default=48)
    parser.add_argument("--branches", type=int, default=16)
    parser.add_argument("--population-batch-size", type=int, default=None)
    parser.add_argument(
        "--antithetic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--target-batch-size", type=int, default=496)
    parser.add_argument(
        "--step-indices",
        type=int,
        nargs="+",
        default=(16, 32, 64, 96),
    )
    parser.add_argument("--blocks", nargs="+", default=DEFAULT_BLOCKS)
    parser.add_argument("--seed", type=int, default=941_731)
    return parser.parse_args()


def _jsonable_metrics(metrics: dict[str, object]) -> dict[str, object]:
    """Round-trip through JSON to reject accidental tensors/non-finite data."""

    return json.loads(json.dumps(metrics, allow_nan=False))


def main() -> None:
    args = _parse_args()
    candidate_dir = args.candidate_run_dir.resolve()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    manifest = _read_json(candidate_dir / "candidate_manifest.json")
    if manifest.get("objective_protocol") != "real_scale_corrected_v3_family_adjoint_rms":
        raise ValueError("diagnostic requires a v3 family-normalized candidate")

    device = torch.device(args.device)
    dtype = torch.float32
    adapter_dataset = candidate_dir / "adapter_dataset"
    target_paths, transform, preprocessing = _load_training_data(
        REPO_ROOT,
        device=device,
        dtype=dtype,
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
    component_namespace = argparse.Namespace(
        objective_protocol=str(manifest["objective_protocol"]),
        eta=float(manifest["eta"]),
        running_cost=str(manifest.get("running_cost", "specific_entropy")),
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
            for value in (
                manifest.get("discrepancy_acf_lags")
                or (1, 2, 5, 10)
            )
        ),
    )
    component_args = _component_args(
        component_namespace,
        reference_sigma_std_floor=reference_sigma_std_floor,
        admissible_sigma_max=float(manifest.get("admissible_sigma_max", 0.6)),
    )
    grid, architecture, control, base_discrepancy, joint_discrepancy = _build_components(
        component_args,
        target_paths=target_paths,
        path_scale=float(transform.sigma) / math.sqrt(float(transform.dt)),
    )
    conditional_summary = manifest.get("conditional_claim")
    if (
        str(manifest.get("phase")) == "joint"
        and isinstance(conditional_summary, dict)
        and str(conditional_summary.get("mode")) == "residual_mmd"
    ):
        joint_discrepancy = _replace_named_discrepancy_block(
            joint_discrepancy,
            name="volatility_law_joint_prefix_future_rv",
            replacement=fit_fixed_target_conditional_residual_volatility_mmd(
                target_paths,
                grid=grid,
                split_index=int(conditional_summary["split_index"]),
                future_horizon=int(conditional_summary["future_horizon"]),
                prefix_windows=tuple(
                    int(value) for value in conditional_summary["prefix_windows"]
                ),
                recent_return_window=int(
                    conditional_summary["recent_return_window"]
                ),
                ridge=float(conditional_summary["predictor_ridge"]),
                bandwidths=(0.5, 1.0, 2.0),
                moment_weight=float(conditional_summary.get("moment_weight", 0.0)),
            ),
        )
    elif (
        str(manifest.get("phase")) == "joint"
        and isinstance(conditional_summary, dict)
        and str(conditional_summary.get("mode")) == "correlation"
    ):
        joint_discrepancy = _replace_named_discrepancy_block(
            joint_discrepancy,
            name="volatility_law_joint_prefix_future_rv",
            replacement=StabilizedPrefixFutureVolatilityCorrelationDiscrepancy(
                windows=tuple(
                    int(value) for value in conditional_summary["windows"]
                ),
                full_matrix=bool(conditional_summary["full_matrix"]),
                split_index=int(conditional_summary["split_index"]),
                scale_floor_fraction=float(
                    conditional_summary["scale_floor_fraction"]
                ),
            ),
        )
    if (
        str(manifest.get("phase")) == "joint"
        and str(manifest.get("discrepancy_stack", "full")) == "compact"
    ):
        retained_names = {
            str(value)
            for value in manifest.get("retained_discrepancy_blocks", ())
        }
        retained = tuple(
            block
            for block in joint_discrepancy.blocks
            if str(block.name) in retained_names
        )
        if {str(block.name) for block in retained} != retained_names:
            raise ValueError("compact discrepancy blocks do not match the manifest")
        joint_discrepancy = CompositePathFunctionalDiscrepancy(blocks=retained)
    discrepancy = _restore_normalized_discrepancy(
        (
            base_discrepancy
            if str(manifest.get("phase")) == "source"
            else joint_discrepancy
        ),
        manifest["discrepancy_normalization"],
    )
    model = _build_model(
        architecture=architecture,
        grid=grid,
        control=control,
        discrepancy=discrepancy,
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
    _configure_r_ce_estimator(model, estimator_args, target_paths=target_paths)
    adjoint_network = manifest.get("adjoint_network", {"kind": "plain"})
    if not isinstance(adjoint_network, dict):
        raise ValueError("candidate manifest has an invalid adjoint_network summary")
    adjoint_kind = str(adjoint_network.get("kind", "plain"))
    default_stack_depth = 0 if adjoint_kind == "plain" else 1
    _configure_adjoint_network_stack(
        model,
        kind=adjoint_kind,
        stack_depth=int(adjoint_network.get("stack_depth", default_stack_depth)),
        seed=int(manifest["seed"]),
        target_paths=target_paths,
    )
    checkpoint = None
    if not bool(args.reference_only):
        checkpoint = (
            Path(str(manifest["checkpoint"]))
            if args.checkpoint is None
            else args.checkpoint.resolve()
        )
        model.load_checkpoint_state(
            torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        )
    training = replace(
        DiscreteMPTrainingConfig(**manifest["training"]),
        batch_size=int(args.prefixes) * int(args.branches),
        target_batch_size=min(int(args.target_batch_size), int(target_paths.shape[0])),
        num_steps=1,
        noise_target_control_variate="adjoint",
        grad_clip_norm=None,
        log_every=1,
        seed=int(args.seed),
    )
    result = diagnose_conditional_adjoint_signal(
        model=model,
        target_paths=target_paths,
        training=training,
        config=ConditionalSignalDiagnosticConfig(
            num_prefixes=int(args.prefixes),
            num_branches=int(args.branches),
            population_batch_size=args.population_batch_size,
            target_batch_size=min(
                int(args.target_batch_size), int(target_paths.shape[0])
            ),
            step_indices=tuple(int(value) for value in args.step_indices),
            discrepancy_block_names=tuple(str(value) for value in args.blocks),
            antithetic=bool(args.antithetic),
            seed=int(args.seed),
        ),
    )
    payload = {
        "protocol": {
            "checkpoint_frozen": True,
            "stage": "reference" if bool(args.reference_only) else "mp",
            "reference_only_zero_adjoint": bool(args.reference_only),
            "train_split_only": True,
            "complete_path_dependent_maximum_principle": True,
            "independent_futures_from_fixed_prefixes": True,
            "antithetic_future_pairs": bool(args.antithetic),
            "frozen_population_batch_size": (
                int(args.prefixes)
                if args.population_batch_size is None
                else int(args.population_batch_size)
            ),
        },
        "candidate_run_dir": str(candidate_dir),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "preprocessing": preprocessing,
        "metrics": _jsonable_metrics(result.metrics),
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(result.tensors, run_dir / "diagnostic_artifacts.pt")
    summary = result.metrics["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote nested signal diagnostic to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
