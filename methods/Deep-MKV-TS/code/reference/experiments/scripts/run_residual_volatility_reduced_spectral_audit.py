#!/usr/bin/env python3
"""Estimate the locked six-dimensional MP operator spectrum for one variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    CausalPathFeatureRidgeConditionalExpectation,
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
)
from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl  # noqa: E402
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.terminal_fixed_point_diagnostic import (  # noqa: E402
    _estimate_reduced_jacobian,
    _time_blocks,
    jacobian_spectrum,
)
from path_dt_experiments.validated_nested_solver import _network_values  # noqa: E402
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_residual_volatility_first_operator_audit import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_SIGNAL_RUN,
    VARIANTS,
    _build_discrepancies,
    _load_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--signal-run-dir", type=Path, default=DEFAULT_SIGNAL_RUN)
    parser.add_argument("--first-operator-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--fit-pool-paths", type=int, default=3072)
    parser.add_argument("--jacobian-fit-paths", type=int, default=512)
    parser.add_argument("--jacobian-validation-paths", type=int, default=512)
    parser.add_argument("--jacobian-target-paths", type=int, default=1024)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--finite-difference-step", type=float, default=0.05)
    parser.add_argument("--secondary-step", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=2_608_501)
    parser.add_argument("--network-init-seed", type=int, default=2_608_401)
    return parser.parse_args()


def _construct_model(args: argparse.Namespace) -> tuple[DiscreteMPModel, torch.Tensor]:
    device = torch.device(args.device)
    target_paths, manifest = _load_paths(
        args.dataset_root.resolve(),
        count=int(args.calibration_paths),
        device=device,
    )
    configuration = manifest["configuration"]
    num_steps = int(configuration["sequence_length"]) - 1
    grid = DiscreteTimeGrid(
        T=float(num_steps) * float(configuration["dt"]), num_steps=num_steps
    )
    signal = torch.load(
        args.signal_run_dir.resolve() / "selected_residual_objective.pt",
        map_location="cpu",
        weights_only=False,
    )
    residual = signal["selected_discrepancy"]
    reference = signal["generic_reference"]
    discrepancies = _build_discrepancies(
        target_paths=target_paths,
        grid=grid,
        configuration=configuration,
        residual_discrepancy=residual,
        requested_variants=(str(args.variant),),
    )
    architecture = DiscreteMPArchitectureConfig(
        state_dim=1,
        noise_dim=1,
        hidden_dim=96,
        num_layers=1,
        adjoint_dim=1,
        noise_adjoint_dim=1,
    )
    ce = CausalPathFeatureRidgeConditionalExpectation(
        grid=grid,
        ridge=1e-3,
        rolling_windows=(5, 10, 20, 32),
        ewma_half_lives=(5, 20, 60),
        downside_windows=(10, 32),
        drawdown_thresholds=(),
        drawdown_memory_half_lives=(),
        drawdown_memory_lags=(),
        drawdown_gate_temperature=0.005,
    )
    eta = 4.0 if str(args.variant).endswith("eta4") else 1.0
    control = SpecificEntropyDiagonalControl(
        dt=float(grid.dt),
        eta=eta,
        reference_kernel=reference,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    training = DiscreteMPTrainingConfig(
        batch_size=256,
        target_batch_size=256,
        num_steps=2000,
        lr=2e-3,
        weight_decay=1e-5,
        lambda_scale=50.0,
        ce_target_mode="ridge",
        ridge_lambda=1e-3,
        ce_crossfit_folds=1,
        target_preconditioner="none",
        noise_target_control_variate="adjoint",
        grad_clip_norm=None,
        log_every=100,
        seed=0,
        observed_only=False,
    )
    model = DiscreteMPModel(
        architecture=architecture,
        grid=grid,
        dynamics_step=DriftVolatilityEulerStep(dt=float(grid.dt)),
        control_map=control,
        discrepancy=discrepancies[str(args.variant)],
        running_cost=control.running_cost,
        ce_estimator=ce,
        noise_ce_estimator=ce,
        training=training,
        init_seed=int(args.network_init_seed),
    )
    model.network.to(device=device, dtype=torch.float32)
    return model, target_paths


def main() -> None:
    args = parse_args()
    if not 0 < int(args.fit_pool_paths) < int(args.calibration_paths):
        raise ValueError("fit-pool-paths must split the calibration law")
    if int(args.replicates) < 2:
        raise ValueError("at least two spectral replicates are required")
    run_dir = ensure_run_dir(args.run_dir.resolve())
    model, paths = _construct_model(args)
    checkpoint_path = (
        args.first_operator_run_dir.resolve()
        / f"fresh_adjoint_{args.variant}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    base_values = _network_values(model)
    candidate_values = {
        name: checkpoint["state_dict"][name].detach().clone()
        for name, _ in model.network.named_parameters()
    }
    fit_pool = paths[: int(args.fit_pool_paths)]
    validation_pool = paths[int(args.fit_pool_paths) :]
    blocks = _time_blocks(int(model.grid.num_steps), 3)
    matrices = []
    details = []
    for replicate in range(int(args.replicates)):
        print(
            f"{args.variant}: Jacobian replicate {replicate + 1}/{args.replicates}",
            flush=True,
        )
        matrix, detail = _estimate_reduced_jacobian(
            model=model,
            fit_pool=fit_pool,
            validation_pool=validation_pool,
            training=model.training,
            base_values=base_values,
            candidate_values=candidate_values,
            blocks=blocks,
            fit_count=int(args.jacobian_fit_paths),
            validation_count=int(args.jacobian_validation_paths),
            target_batch_size=int(args.jacobian_target_paths),
            step=float(args.finite_difference_step),
            seed=int(args.seed) + replicate * 10_000,
        )
        matrices.append(matrix)
        details.append(detail)
    print(f"{args.variant}: secondary-radius audit", flush=True)
    secondary, secondary_detail = _estimate_reduced_jacobian(
        model=model,
        fit_pool=fit_pool,
        validation_pool=validation_pool,
        training=model.training,
        base_values=base_values,
        candidate_values=candidate_values,
        blocks=blocks,
        fit_count=int(args.jacobian_fit_paths),
        validation_count=int(args.jacobian_validation_paths),
        target_batch_size=int(args.jacobian_target_paths),
        step=float(args.secondary_step),
        seed=int(args.seed),
    )
    stack = torch.stack(matrices)
    mean = stack.mean(dim=0)
    radii = torch.tensor(
        [row["spectrum"]["spectral_radius_T"] for row in details],
        dtype=torch.float64,
    )
    epsilon = torch.finfo(torch.float64).eps
    report = {
        "protocol": {
            "variant": str(args.variant),
            "beta": 1.0,
            "lambda_scale": 50.0,
            "three_time_blocks": True,
            "p_r_coordinate_dimension": 6,
            "common_random_central_finite_differences": True,
            "test_paths_used": False,
            "first_operator_checkpoint": str(checkpoint_path),
        },
        "config": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "time_blocks": [list(value) for value in blocks],
        "replicates": details,
        "matrices": [value.tolist() for value in matrices],
        "mean_matrix": mean.tolist(),
        "mean_spectrum": jacobian_spectrum(mean),
        "replicate_spectral_radius": {
            "values": radii.tolist(),
            "mean": float(radii.mean().item()),
            "sd": float(radii.std(unbiased=True).item()),
            "minimum": float(radii.min().item()),
            "maximum": float(radii.max().item()),
        },
        "secondary_matrix": secondary.tolist(),
        "secondary_detail": secondary_detail,
        "finite_difference_relative_change": float(
            (
                torch.linalg.vector_norm(matrices[0] - secondary)
                / torch.linalg.vector_norm(matrices[0]).clamp_min(epsilon)
            ).item()
        ),
    }
    write_json(run_dir / "report.json", report)
    torch.save(
        {"jacobians": stack, "secondary_jacobian": secondary},
        run_dir / "artifacts.pt",
    )
    (run_dir / "SUMMARY.md").write_text(
        "# Reduced spectral audit\n\n"
        f"- Variant: `{args.variant}`\n"
        f"- Mean spectral radius: `{report['replicate_spectral_radius']['mean']:.6g}`\n"
        f"- Replicate range: `[{report['replicate_spectral_radius']['minimum']:.6g}, "
        f"{report['replicate_spectral_radius']['maximum']:.6g}]`\n"
        f"- Mean-matrix spectral radius: `{report['mean_spectrum']['spectral_radius_T']:.6g}`\n"
        f"- Radius sensitivity: `{report['finite_difference_relative_change']:.6g}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report["replicate_spectral_radius"], indent=2))


if __name__ == "__main__":
    main()
