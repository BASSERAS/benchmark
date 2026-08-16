#!/usr/bin/env python3
"""Estimate one reduced Jacobian matrix for the direct-ridge R operator."""

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

from path_dt_experiments.direct_ridge_adjoint import (  # noqa: E402
    BlockwiseAdjointInterpolationPolicy,
    DirectRidgeRAdjointPolicy,
    TimeSmoothedRidgeProjection,
)
from path_dt_experiments.extragradient_solver import _sample_inputs  # noqa: E402
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    AdjointNetworkControlPolicy,
    policy_moments_on_paths,
    rollout_causal_policy,
)
from path_dt_experiments.terminal_fixed_point_diagnostic import (  # noqa: E402
    _time_blocks,
    jacobian_spectrum,
)
from path_dt_experiments.validated_nested_solver import (  # noqa: E402
    _project_train_to_eval,
    _sample_target_batch,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_residual_volatility_reduced_spectral_audit import (  # noqa: E402
    _construct_model,
)
from run_residual_volatility_first_operator_audit import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_SIGNAL_RUN,
)


DEFAULT_DIRECT_RUN = REPO_ROOT / "runs" / "direct_ridge_r_audit_seed0_20260803"
DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT / "runs" / "residual_volatility_first_operator_corrected_parts_20260803"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--direct-run-dir", type=Path, default=DEFAULT_DIRECT_RUN)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--signal-run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--fit-pool-paths", type=int, default=3072)
    parser.add_argument("--fit-paths", type=int, default=512)
    parser.add_argument("--validation-paths", type=int, default=512)
    parser.add_argument("--target-paths", type=int, default=1024)
    parser.add_argument("--finite-difference-step", type=float, default=0.05)
    parser.add_argument("--matrix-index", type=int, default=0)
    parser.add_argument("--zero-only", action="store_true")
    parser.add_argument("--seed", type=int, default=2_608_601)
    parser.add_argument("--network-init-seed", type=int, default=2_608_401)
    return parser.parse_args()


def _model_args(args: argparse.Namespace) -> argparse.Namespace:
    # Reuse the locked residual-objective model constructor without duplicating
    # its mathematical configuration.
    return argparse.Namespace(
        device=args.device,
        dataset_root=DEFAULT_DATASET if args.dataset_root is None else args.dataset_root,
        signal_run_dir=(
            DEFAULT_SIGNAL_RUN if args.signal_run_dir is None else args.signal_run_dir
        ),
        calibration_paths=int(args.calibration_paths),
        variant="residual_conditional_eta4",
        network_init_seed=int(args.network_init_seed),
    )


def _operator_coordinates(
    *,
    model,
    fit_pool: torch.Tensor,
    validation_pool: torch.Tensor,
    base_policy: object,
    candidate_policy: DirectRidgeRAdjointPolicy,
    theta: torch.Tensor,
    blocks: tuple[tuple[int, int], ...],
    fit_count: int,
    validation_count: int,
    target_count: int,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    directional = BlockwiseAdjointInterpolationPolicy(
        model=model,
        base_policy=base_policy,
        candidate_policy=candidate_policy,
        theta=theta,
        blocks=blocks,
    )
    fit_x0, fit_noise = _sample_inputs(
        model=model, initial_state_pool=fit_pool, count=int(fit_count), seed=int(seed) + 1
    )
    validation_x0, validation_noise = _sample_inputs(
        model=model,
        initial_state_pool=validation_pool,
        count=int(validation_count),
        seed=int(seed) + 2,
    )
    fit_law = rollout_causal_policy(
        model=model, policy=directional, x0=fit_x0, noise=fit_noise
    )
    validation_law = rollout_causal_policy(
        model=model,
        policy=directional,
        x0=validation_x0,
        noise=validation_noise,
    )
    target = _sample_target_batch(
        model=model,
        pool=fit_pool,
        count=int(target_count),
        seed=int(seed) + 3,
    )
    pathwise_p, _ = model._path_functional_adjoint_targets(
        generated_path=fit_law.paths,
        controls=fit_law.controls,
        target_path=target,
        training=model.training,
    )
    pathwise_r = model._adjoint_noise_target(
        pathwise_adjoint_target=pathwise_p,
        moments_noise_target_dim=int(fit_law.expected_adjoint_noise_next.shape[-1]),
        noise=fit_law.noise,
        control_variate=fit_law.expected_adjoint_next,
    )
    with torch.no_grad():
        teacher_p = _project_train_to_eval(
            model=model,
            training_paths=fit_law.paths,
            training_targets=pathwise_p,
            evaluation_paths=validation_law.paths,
            training=model.training,
        )
        refreshed_ridge = TimeSmoothedRidgeProjection.fit(
            feature_map=candidate_policy.r_projection.feature_map,
            paths=fit_law.paths,
            targets=pathwise_r,
            ridge=float(candidate_policy.r_projection.ridge),
        )
        teacher_r = refreshed_ridge.predict_all(validation_law.paths)
        base_p, base_r, _ = policy_moments_on_paths(
            model=model, policy=base_policy, paths=validation_law.paths
        )
        candidate_p, candidate_r, _ = policy_moments_on_paths(
            model=model, policy=candidate_policy, paths=validation_law.paths
        )
    coordinates = []
    basis_report = []
    eps = torch.finfo(torch.float64).eps
    for signal, base, candidate, teacher in (
        ("P", base_p, candidate_p, teacher_p),
        ("R", base_r, candidate_r, teacher_r),
    ):
        for block_index, (left, right) in enumerate(blocks):
            direction = (candidate[:, left:right] - base[:, left:right]).double()
            target_delta = (teacher[:, left:right] - base[:, left:right]).double()
            denominator = torch.sum(direction.pow(2))
            if float(denominator.item()) <= eps:
                raise RuntimeError(f"{signal} direction is zero in block {block_index}")
            coordinate = torch.sum(direction * target_delta) / denominator
            coordinates.append(coordinate)
            basis_report.append(
                {
                    "signal": signal,
                    "block": float(block_index),
                    "left_step": float(left),
                    "right_step_exclusive": float(right),
                    "direction_rms": float(torch.mean(direction.pow(2)).sqrt().item()),
                    "teacher_rms": float(torch.mean(target_delta.pow(2)).sqrt().item()),
                    "coordinate": float(coordinate.item()),
                    "explained_fraction": float(
                        (
                            coordinate.pow(2)
                            * denominator
                            / torch.sum(target_delta.pow(2)).clamp_min(eps)
                        ).item()
                    ),
                }
            )
    return torch.stack(coordinates).detach().cpu(), basis_report


def main() -> None:
    args = parse_args()
    if int(args.matrix_index) < 0:
        raise ValueError("matrix-index must be nonnegative")
    if not 0 < int(args.fit_pool_paths) < int(args.calibration_paths):
        raise ValueError("fit-pool-paths must split the calibration law")
    run_dir = ensure_run_dir(args.run_dir.resolve())
    model, paths = _construct_model(_model_args(args))
    checkpoint = torch.load(
        args.checkpoint_root.resolve()
        / "residual_eta4"
        / "fresh_adjoint_residual_conditional_eta4.pt",
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
    projection = torch.load(
        args.direct_run_dir.resolve() / "residual_aware_ridge.pt",
        map_location="cpu",
        weights_only=False,
    )
    candidate_policy = DirectRidgeRAdjointPolicy(
        model=model, p_network=p_network, r_projection=projection
    )
    base_policy = AdjointNetworkControlPolicy(model=model, network=model.network)
    fit_pool = paths[: int(args.fit_pool_paths)]
    validation_pool = paths[int(args.fit_pool_paths) :]
    blocks = _time_blocks(int(model.grid.num_steps), 3)
    dimension = 2 * len(blocks)
    seed = int(args.seed) + int(args.matrix_index) * 10_000
    if bool(args.zero_only):
        print(f"zero operator replicate {args.matrix_index}", flush=True)
        coordinates, basis = _operator_coordinates(
            model=model,
            fit_pool=fit_pool,
            validation_pool=validation_pool,
            base_policy=base_policy,
            candidate_policy=candidate_policy,
            theta=torch.zeros(dimension, dtype=torch.float64),
            blocks=blocks,
            fit_count=int(args.fit_paths),
            validation_count=int(args.validation_paths),
            target_count=int(args.target_paths),
            seed=seed,
        )
        report = {
            "protocol": {
                "zero_operator_replicate": True,
                "matrix_index": int(args.matrix_index),
                "seed": seed,
                "direct_ridge_R_refitted_under_the_zero_coordinate_law": True,
                "P_projected_with_existing_generic_causal_estimator": True,
                "test_paths_used": False,
            },
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "time_blocks": [list(block) for block in blocks],
            "coordinates": coordinates.tolist(),
            "basis": basis,
        }
        write_json(run_dir / "report.json", report)
        torch.save({"coordinates": coordinates}, run_dir / "artifacts.pt")
        print(json.dumps(report, indent=2), flush=True)
        return
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
            "common_random_central_finite_differences": True,
            "test_paths_used": False,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
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
