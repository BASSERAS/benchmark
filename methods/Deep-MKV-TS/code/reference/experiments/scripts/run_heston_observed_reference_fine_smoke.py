"""End-to-end Heston smoke test for fine-step MP volatility decisions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
    fit_local_gaussian_reference_kernel,
)
from path_dt_experiments.data import HestonConfig, simulate_heston_log_paths  # noqa: E402
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.observed_reference_fine_grid import (  # noqa: E402
    FineGridVolatilityOnlySpecificEntropyControl,
    ObservedReferenceFineGridLift,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "heston_observed_reference_true_fine_smoke_20260807"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--observed-steps", type=int, default=12)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--num-target-paths", type=int, default=64)
    parser.add_argument("--training-steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float32
    observed_steps = int(args.observed_steps)
    substeps = int(args.substeps)
    fine_steps = observed_steps * substeps
    observed_grid = DiscreteTimeGrid(T=0.12, num_steps=observed_steps)
    fine_grid = DiscreteTimeGrid(
        T=0.12,
        num_steps=fine_steps,
        observed_indices=tuple(range(0, fine_steps + 1, substeps)),
    )
    target_config = HestonConfig(
        T=float(fine_grid.T),
        num_steps=fine_steps,
        mu=0.05,
        kappa=2.0,
        theta=0.04,
        vol_of_vol=0.3,
        rho=-0.7,
        v0=0.04,
        x0=0.0,
    )
    target_fine = simulate_heston_log_paths(
        num_paths=int(args.num_target_paths),
        config=target_config,
        seed=int(args.seed),
        device=device,
        dtype=dtype,
    )
    target_observed = target_fine[:, ::substeps, :].contiguous()
    if int(target_observed.shape[1]) != observed_steps + 1:
        raise RuntimeError("target downsampling did not land on the observed grid")

    observed_reference = fit_local_gaussian_reference_kernel(
        target_paths=target_observed,
        grid=observed_grid,
        ridge=1e-3,
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=1e-4,
        sigma_max=0.6,
    )
    reference_lift = ObservedReferenceFineGridLift(
        observed_reference=observed_reference,
        fine_grid=fine_grid,
        substeps=substeps,
    )
    control = FineGridVolatilityOnlySpecificEntropyControl(
        dt=float(fine_grid.dt),
        reference_lift=reference_lift,
        eta=1.0,
        sigma_min=1e-4,
        sigma_max=2.0,
    )
    discrepancy = build_heston_discrepancy(
        num_steps=observed_steps,
        include_acf=True,
        preset="simple",
        lambda_scale=50.0,
        kappa_scale=100.0,
    )
    training = DiscreteMPTrainingConfig(
        batch_size=min(32, int(args.num_target_paths)),
        target_batch_size=min(32, int(args.num_target_paths)),
        num_steps=int(args.training_steps),
        lr=2.5e-4,
        weight_decay=1e-5,
        lambda_scale=50.0,
        ce_target_mode="ridge",
        ridge_lambda=1e-3,
        grad_clip_norm=5.0,
        log_every=1,
        seed=int(args.seed),
        observed_only=True,
    )
    model = DiscreteMPModel(
        architecture=DiscreteMPArchitectureConfig(
            state_dim=1,
            noise_dim=1,
            hidden_dim=16,
        ),
        grid=fine_grid,
        dynamics_step=DriftVolatilityEulerStep(dt=float(fine_grid.dt)),
        control_map=control,
        discrepancy=discrepancy,
        training=training,
        init_seed=int(args.seed),
    )
    model.network.to(device=device, dtype=dtype)
    fit = model.fit(target_fine, training=training)
    sample = model.sample(
        num_paths=16,
        x0=torch.tensor([float(target_config.x0)], device=device, dtype=dtype),
        seed=int(args.seed) + 1,
    )

    losses = [float(row["train_total_loss"]) for row in fit.history]
    sigma = sample.controls[..., 1]
    sigma_blocks = sigma.reshape(16, observed_steps, substeps)
    within_block_ranges = sigma_blocks.max(dim=-1).values - sigma_blocks.min(
        dim=-1
    ).values
    within_block_change = float(within_block_ranges.max().item())
    reference_feature_width = max(
        int(mean.shape[1]) for mean in observed_reference.feature_means
    )
    finite = bool(
        all(math.isfinite(value) for value in losses)
        and torch.isfinite(sample.paths).all().item()
        and torch.isfinite(sample.controls).all().item()
    )
    summary = {
        "status": (
            "passed" if finite and within_block_change > 1e-8 else "failed"
        ),
        "fine_grid_steps": fine_steps,
        "observed_steps": observed_steps,
        "substeps_per_observation": substeps,
        "reference_fitted_on_observed_points": observed_steps + 1,
        "reference_max_prefix_width": reference_feature_width,
        "discrepancy_points": observed_steps + 1,
        "running_cost_steps": fine_steps,
        "mp_control_decisions": fine_steps,
        "training_steps": int(args.training_steps),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "losses_finite": bool(all(math.isfinite(value) for value in losses)),
        "rollout_finite": finite,
        "sigma_min": float(sigma.min().item()),
        "sigma_max": float(sigma.max().item()),
        "max_within_block_sigma_range": within_block_change,
        "volatility_changes_inside_observation_intervals": bool(
            within_block_change > 1e-8
        ),
        "production_solver_modified": False,
    }
    run_dir = ensure_run_dir(args.run_dir)
    write_json(run_dir / "summary.json", summary)
    print(summary)
    if summary["status"] != "passed":
        raise RuntimeError("observed-reference fine-grid Heston smoke test failed")


if __name__ == "__main__":
    main()
