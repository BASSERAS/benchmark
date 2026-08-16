#!/usr/bin/env python3
"""Estimate one reduced MP-operator matrix with a frozen dual witness."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from path_dt_experiments.direct_ridge_adjoint import (  # noqa: E402
    BlockwiseAdjointInterpolationPolicy,
    DirectRidgeRAdjointPolicy,
)
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    AdjointNetworkControlPolicy,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.terminal_fixed_point_diagnostic import (  # noqa: E402
    _time_blocks,
    jacobian_spectrum,
)
from run_direct_ridge_r_spectral_matrix import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DIRECT_RUN,
    _operator_coordinates,
)
from run_residual_volatility_reduced_spectral_audit import (  # noqa: E402
    _construct_model,
)
from run_residual_volatility_first_operator_audit import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_SIGNAL_RUN,
)


DEFAULT_WITNESS_RUN = REPO_ROOT / "runs" / "residual_dual_witness_audit_20260803"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--witness-run-dir", type=Path, default=DEFAULT_WITNESS_RUN)
    parser.add_argument("--direct-run-dir", type=Path, default=DEFAULT_DIRECT_RUN)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--signal-run-dir", type=Path, default=DEFAULT_SIGNAL_RUN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--fit-pool-paths", type=int, default=3072)
    parser.add_argument("--fit-paths", type=int, default=512)
    parser.add_argument("--validation-paths", type=int, default=512)
    parser.add_argument("--target-paths", type=int, default=1024)
    parser.add_argument("--finite-difference-step", type=float, default=0.05)
    parser.add_argument("--matrix-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2_608_601)
    parser.add_argument("--network-init-seed", type=int, default=2_608_401)
    return parser.parse_args()


def _model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device,
        dataset_root=args.dataset_root,
        signal_run_dir=args.signal_run_dir,
        calibration_paths=int(args.calibration_paths),
        variant="residual_conditional_eta4",
        network_init_seed=int(args.network_init_seed),
    )


def _replace_residual_block(model, witness: object) -> None:
    replacement_count = 0
    blocks = []
    for block in model.discrepancy.blocks:
        if block.name == "volatility_law_reference_residual_conditional_rv":
            blocks.append(
                WeightedDiscrepancy(
                    name=block.name,
                    weight=float(block.weight),
                    discrepancy=witness,
                )
            )
            replacement_count += 1
        else:
            blocks.append(block)
    if replacement_count != 1:
        raise RuntimeError("expected to replace exactly one residual MMD block")
    model.discrepancy = CompositePathFunctionalDiscrepancy(blocks=tuple(blocks))


def main() -> None:
    args = parse_args()
    if int(args.matrix_index) < 0:
        raise ValueError("matrix-index must be nonnegative")
    if not 0 < int(args.fit_pool_paths) < int(args.calibration_paths):
        raise ValueError("fit-pool-paths must split the calibration law")
    run_dir = ensure_run_dir(args.run_dir.resolve())
    witness_report = json.loads(
        (args.witness_run_dir.resolve() / "report.json").read_text(encoding="utf-8")
    )
    if not bool(witness_report["gate"]["passed"]):
        raise RuntimeError("the exact dual-witness identity gate did not pass")
    witness_artifact = torch.load(
        args.witness_run_dir.resolve() / "fixed_residual_witness.pt",
        map_location="cpu",
        weights_only=False,
    )
    witness = witness_artifact["witness"]
    model, paths = _construct_model(_model_args(args))
    _replace_residual_block(model, witness)

    checkpoint_path = (
        args.checkpoint_root.resolve()
        / "residual_eta4"
        / "fresh_adjoint_residual_conditional_eta4.pt"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=model.device,
        weights_only=False,
    )
    p_network = copy.deepcopy(model.network)
    p_network.load_state_dict(checkpoint["state_dict"])
    p_network.eval()
    for module in p_network.modules():
        if isinstance(module, torch.nn.GRU):
            module.flatten_parameters()
    for module in model.network.modules():
        if isinstance(module, torch.nn.GRU):
            module.flatten_parameters()
    projection_path = args.direct_run_dir.resolve() / "residual_aware_ridge.pt"
    projection = torch.load(projection_path, map_location="cpu", weights_only=False)
    candidate_policy = DirectRidgeRAdjointPolicy(
        model=model,
        p_network=p_network,
        r_projection=projection,
    )
    base_policy = AdjointNetworkControlPolicy(model=model, network=model.network)
    fit_pool = paths[: int(args.fit_pool_paths)]
    validation_pool = paths[int(args.fit_pool_paths) :]
    blocks = _time_blocks(int(model.grid.num_steps), 3)
    dimension = 2 * len(blocks)
    seed = int(args.seed) + int(args.matrix_index) * 10_000
    matrix = torch.empty(dimension, dimension, dtype=torch.float64)
    columns = []
    for column in range(dimension):
        print(f"matrix {args.matrix_index}: column {column + 1}/{dimension}", flush=True)
        rows = []
        bases = []
        for sign in (1.0, -1.0):
            theta = torch.zeros(dimension, dtype=torch.float64)
            theta[column] = sign * float(args.finite_difference_step)
            coordinates, basis = _operator_coordinates(
                model=model,
                fit_pool=fit_pool,
                validation_pool=validation_pool,
                base_policy=base_policy,
                candidate_policy=candidate_policy,
                theta=theta,
                blocks=blocks,
                fit_count=int(args.fit_paths),
                validation_count=int(args.validation_paths),
                target_count=int(args.target_paths),
                seed=seed,
            )
            rows.append(coordinates)
            bases.append(basis)
        derivative = (rows[0] - rows[1]) / (2.0 * float(args.finite_difference_step))
        matrix[:, column] = derivative
        columns.append(
            {
                "column": int(column),
                "plus_coordinates": rows[0].tolist(),
                "minus_coordinates": rows[1].tolist(),
                "derivative": derivative.tolist(),
                "plus_basis": bases[0],
                "minus_basis": bases[1],
            }
        )
    report = {
        "protocol": {
            "matrix_index": int(args.matrix_index),
            "seed": seed,
            "finite_difference_step": float(args.finite_difference_step),
            "three_time_blocks": True,
            "direct_ridge_R_refitted_under_each_candidate_law": True,
            "P_projected_with_existing_generic_causal_estimator": True,
            "residual_mmd_dual_witness_fixed_at_reference_target_mismatch": True,
            "residual_population_law_source_regeneration_disabled": True,
            "all_other_discrepancy_blocks_unchanged": True,
            "common_random_central_finite_differences": True,
            "test_paths_used": False,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "artifacts": {
            "witness": str(
                (args.witness_run_dir.resolve() / "fixed_residual_witness.pt")
            ),
            "p_checkpoint": str(checkpoint_path),
            "r_projection": str(projection_path),
        },
        "time_blocks": [list(block) for block in blocks],
        "matrix": matrix.tolist(),
        "columns": columns,
        "spectrum": jacobian_spectrum(matrix),
    }
    write_json(run_dir / "report.json", report)
    torch.save({"matrix": matrix}, run_dir / "artifacts.pt")
    print(json.dumps(report["spectrum"], indent=2), flush=True)


if __name__ == "__main__":
    main()
