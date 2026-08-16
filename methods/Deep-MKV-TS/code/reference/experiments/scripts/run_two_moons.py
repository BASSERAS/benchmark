from __future__ import annotations

import argparse
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
from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl  # noqa: E402
from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from path_dt_experiments.data import RoutedTwoMoonsConfig, sample_routed_two_moons_paths  # noqa: E402
from path_dt_experiments.discrepancies import build_path_volatility_discrepancy  # noqa: E402
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.metrics import sliced_wasserstein_distance  # noqa: E402
from path_dt_experiments.runners import default_run_dir, ensure_run_dir, write_history_jsonl, write_json  # noqa: E402
from path_dt_experiments.scores import BenchmarkScoreConfig, benchmark_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrated routed two-moons discrete MP experiment.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-target-paths", type=int, default=8192)
    parser.add_argument("--num-heldout-paths", type=int, default=2048)
    parser.add_argument("--num-sample-paths", type=int, default=2048)
    parser.add_argument("--num-grid-steps", type=int, default=32)
    parser.add_argument("--source-std", type=float, default=0.35)
    parser.add_argument("--terminal-noise-std", type=float, default=0.05)
    parser.add_argument("--bridge-noise-std", type=float, default=0.03)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--training-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-scale", type=float, default=50.0)
    parser.add_argument("--kappa-scale", type=float, default=0.0)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.8)
    parser.add_argument("--reference-ridge", type=float, default=1e-3)
    parser.add_argument("--reference-variance-ridge", type=float, default=None)
    parser.add_argument("--reference-variance-shrinkage", type=float, default=0.1)
    parser.add_argument("--reference-log-variance-bias-correction", type=float, default=0.6)
    parser.add_argument("--reference-sigma-max", type=float, default=0.8)
    parser.add_argument("--scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-steps", type=int, default=300)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--score-hidden-dim", type=int, default=32)
    parser.add_argument("--score-lr", type=float, default=1e-3)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-tensors", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _terminal_swd(generated_paths: torch.Tensor, target_paths: torch.Tensor) -> float:
    generated_terminal = generated_paths[:, -1:, :].repeat(1, 2, 1)
    target_terminal = target_paths[:, -1:, :].repeat(1, 2, 1)
    return sliced_wasserstein_distance(
        generated_terminal,
        target_terminal,
        num_projections=128,
        seed=1234,
    )


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    run_dir = ensure_run_dir(args.run_dir or default_run_dir(prefix="two_moons_dt"))

    config = RoutedTwoMoonsConfig(
        num_steps=int(args.num_grid_steps),
        source_std=float(args.source_std),
        terminal_noise_std=float(args.terminal_noise_std),
        bridge_noise_std=float(args.bridge_noise_std),
    )
    grid = DiscreteTimeGrid(T=float(config.T), num_steps=int(config.num_steps))
    target = sample_routed_two_moons_paths(
        num_paths=int(args.num_target_paths),
        config=config,
        seed=int(args.seed),
        device=device,
        dtype=dtype,
    )
    evaluation_seed = derive_stream_seed(base_seed=int(args.seed), stream_offset=20) or int(args.seed)
    heldout_seed = derive_stream_seed(base_seed=evaluation_seed, stream_offset=1)
    sample_seed = derive_stream_seed(base_seed=evaluation_seed, stream_offset=2)
    heldout = sample_routed_two_moons_paths(
        num_paths=int(args.num_heldout_paths),
        config=config,
        seed=heldout_seed,
        device=device,
        dtype=dtype,
    )
    reference_kernel = fit_local_gaussian_reference_kernel(
        target_paths=target.x_path,
        grid=grid,
        ridge=float(args.reference_ridge),
        variance_ridge=args.reference_variance_ridge,
        variance_shrinkage=float(args.reference_variance_shrinkage),
        log_variance_bias_correction=float(args.reference_log_variance_bias_correction),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.reference_sigma_max),
    )
    model = DiscreteMPModel(
        architecture=DiscreteMPArchitectureConfig(
            state_dim=2,
            noise_dim=2,
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
        ),
        grid=grid,
        dynamics_step=DriftVolatilityEulerStep(dt=grid.dt),
        control_map=SpecificEntropyDiagonalControl(
            dt=grid.dt,
            eta=float(args.eta),
            reference_kernel=reference_kernel,
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
        ),
        discrepancy=build_path_volatility_discrepancy(
            observed_path_weight=2.0,
            terminal_weight=2.0,
            increments_weight=0.5,
            lambda_scale=float(args.lambda_scale),
            kappa_scale=float(args.kappa_scale),
        ),
        init_seed=int(args.seed),
    )
    model.network.to(device=device, dtype=dtype)
    fit_result = model.fit(
        target.x_path,
        training=DiscreteMPTrainingConfig(
            batch_size=int(args.batch_size),
            target_batch_size=int(args.target_batch_size),
            num_steps=int(args.training_steps),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            lambda_scale=float(args.lambda_scale),
            ridge_lambda=float(args.ridge_lambda),
            grad_clip_norm=float(args.grad_clip_norm) if args.grad_clip_norm > 0.0 else None,
            log_every=int(args.log_every),
            seed=int(args.seed),
            observed_only=True,
        ),
    )
    x0 = heldout.x0[: int(args.num_sample_paths)]
    sample = model.sample(num_paths=int(x0.shape[0]), x0=x0, seed=sample_seed)
    generated_paths = sample.paths.detach()
    target_eval = heldout.x_path[: int(generated_paths.shape[0])]
    metrics = {
        "run_dir": str(run_dir),
        "config": {
            "two_moons": config.to_dict(),
            "seed": int(args.seed),
            "evaluation_seed": int(evaluation_seed),
            "heldout_seed": int(heldout_seed) if heldout_seed is not None else None,
            "sample_seed": int(sample_seed) if sample_seed is not None else None,
            "lambda_scale": float(args.lambda_scale),
            "kappa_scale": float(args.kappa_scale),
            "reference": reference_kernel.summary(),
        },
        "fit_final": fit_result.final_metrics,
        "path_swd": sliced_wasserstein_distance(generated_paths, target_eval, num_projections=128, seed=1234),
        "terminal_swd": _terminal_swd(generated_paths, target_eval),
        "target_upper_route_fraction": float((target_eval[:, int(grid.num_steps) // 2, 1] > 0.0).float().mean().item()),
        "generated_upper_route_fraction": float((generated_paths[:, int(grid.num_steps) // 2, 1] > 0.0).float().mean().item()),
    }
    if bool(args.scores):
        metrics["benchmark_scores"] = benchmark_scores(
            generated_paths,
            target_eval,
            config=BenchmarkScoreConfig(
                train_steps=int(args.score_steps),
                batch_size=int(args.score_batch_size),
                hidden_dim=int(args.score_hidden_dim),
                lr=float(args.score_lr),
                seed=int(evaluation_seed) + 1000,
            ),
        )
    write_json(run_dir / "metrics.json", metrics)
    write_history_jsonl(run_dir / "history.jsonl", fit_result.history)
    if bool(args.save_tensors):
        torch.save(target.x_path.detach().cpu(), run_dir / "target_paths.pt")
        torch.save(target_eval.detach().cpu(), run_dir / "heldout_paths.pt")
        torch.save(generated_paths.detach().cpu(), run_dir / "generated_paths.pt")
    if not bool(args.no_plots):
        try:
            from path_dt_experiments.plots import plot_two_moons_animation, plot_two_moons_summary
            import matplotlib.pyplot as plt

            fig = plot_two_moons_summary(
                target_paths=target_eval.detach().cpu(),
                generated_paths=generated_paths.detach().cpu(),
                output_path=run_dir / "two_moons_summary.png",
            )
            plt.close(fig)
            plot_two_moons_animation(
                target_paths=target_eval.detach().cpu(),
                generated_paths=generated_paths.detach().cpu(),
                output_path=run_dir / "two_moons.gif",
            )
        except ImportError as exc:
            write_json(run_dir / "plot_error.json", {"error": str(exc)})
    print(f"wrote two-moons run to {run_dir}")
    print(f"metrics: {metrics}")


if __name__ == "__main__":
    main()
