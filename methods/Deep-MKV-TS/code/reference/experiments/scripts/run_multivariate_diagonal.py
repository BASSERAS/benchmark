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
from deep_mkv_gen_path_dt.controls import HamiltonianControlInputs, SpecificEntropyDiagonalControl  # noqa: E402
from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from path_dt_experiments.data import (  # noqa: E402
    MultivariateDiagonalConfig,
    sample_multivariate_diagonal_paths,
)
from path_dt_experiments.discrepancies import build_path_volatility_discrepancy  # noqa: E402
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.metrics import sliced_wasserstein_distance, stylized_fact_metrics  # noqa: E402
from path_dt_experiments.runners import default_run_dir, ensure_run_dir, write_history_jsonl, write_json  # noqa: E402
from path_dt_experiments.scores import BenchmarkScoreConfig, benchmark_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrated multivariate diagonal discrete MP validation.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--case", choices=("constant", "state_dependent"), default="constant")
    parser.add_argument("--state-dim", type=int, default=3)
    parser.add_argument("--num-grid-steps", type=int, default=32)
    parser.add_argument("--observed-every", type=int, default=1)
    parser.add_argument("--num-target-paths", type=int, default=4096)
    parser.add_argument("--num-heldout-paths", type=int, default=4096)
    parser.add_argument("--num-sample-paths", type=int, default=4096)
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-scale", type=float, default=50.0)
    parser.add_argument("--kappa-scale", type=float, default=50.0)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.8)
    parser.add_argument("--reference-variance-source", choices=("residual", "oracle"), default="oracle")
    parser.add_argument("--reference-ridge", type=float, default=1e-3)
    parser.add_argument("--reference-variance-ridge", type=float, default=None)
    parser.add_argument("--reference-variance-shrinkage", type=float, default=0.1)
    parser.add_argument("--reference-log-variance-bias-correction", type=float, default=0.0)
    parser.add_argument("--reference-sigma-max", type=float, default=0.8)
    parser.add_argument("--drift", type=str, default=None)
    parser.add_argument("--base-sigma", type=str, default=None)
    parser.add_argument("--mean-reversion", type=float, default=None)
    parser.add_argument("--state-sensitivity", type=float, default=None)
    parser.add_argument("--cross-sensitivity", type=float, default=None)
    parser.add_argument("--initial-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-steps", type=int, default=300)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--score-hidden-dim", type=int, default=32)
    parser.add_argument("--score-lr", type=float, default=1e-3)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-tensors", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _parse_float_tuple(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    items = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not items:
        raise ValueError("expected at least one comma-separated float")
    return items


def observed_indices_from_every(*, num_steps: int, observed_every: int) -> tuple[int, ...]:
    if int(observed_every) < 1:
        raise ValueError("observed_every must be >= 1")
    if int(num_steps) % int(observed_every) != 0:
        raise ValueError("num_grid_steps must be divisible by observed_every")
    return tuple(range(0, int(num_steps) + 1, int(observed_every)))


def _default_vector(*, state_dim: int, start: float, step: float) -> tuple[float, ...]:
    return tuple(float(start) + float(step) * float(index) for index in range(int(state_dim)))


def _make_config(args: argparse.Namespace) -> MultivariateDiagonalConfig:
    state_dim = int(args.state_dim)
    if str(args.case) == "constant":
        mean_reversion = 0.0 if args.mean_reversion is None else float(args.mean_reversion)
        state_sensitivity = 0.0 if args.state_sensitivity is None else float(args.state_sensitivity)
        cross_sensitivity = 0.0 if args.cross_sensitivity is None else float(args.cross_sensitivity)
        drift = _parse_float_tuple(args.drift) or _default_vector(state_dim=state_dim, start=0.02, step=-0.01)
        base_sigma = _parse_float_tuple(args.base_sigma) or _default_vector(state_dim=state_dim, start=0.16, step=0.035)
    else:
        mean_reversion = 0.6 if args.mean_reversion is None else float(args.mean_reversion)
        state_sensitivity = 0.7 if args.state_sensitivity is None else float(args.state_sensitivity)
        cross_sensitivity = 0.35 if args.cross_sensitivity is None else float(args.cross_sensitivity)
        drift = _parse_float_tuple(args.drift) or _default_vector(state_dim=state_dim, start=0.04, step=-0.015)
        base_sigma = _parse_float_tuple(args.base_sigma) or _default_vector(state_dim=state_dim, start=0.13, step=0.03)
    return MultivariateDiagonalConfig(
        num_steps=int(args.num_grid_steps),
        state_dim=state_dim,
        drift=drift,
        base_sigma=base_sigma,
        mean_reversion=mean_reversion,
        state_sensitivity=state_sensitivity,
        cross_sensitivity=cross_sensitivity,
        sigma_min=max(float(args.sigma_min), 1e-4),
        sigma_max=float(args.sigma_max),
        initial_std=float(args.initial_std),
    )


def _corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.detach().reshape(-1)
    y_flat = y.detach().reshape(-1).to(device=x_flat.device, dtype=x_flat.dtype)
    x_centered = x_flat - x_flat.mean()
    y_centered = y_flat - y_flat.mean()
    denom = torch.sqrt(torch.mean(x_centered.pow(2)) * torch.mean(y_centered.pow(2))).clamp_min(1e-12)
    return float(torch.mean(x_centered * y_centered).div(denom).item())


def _controls_on_path(model: DiscreteMPModel, x_path: torch.Tensor) -> torch.Tensor:
    moments = model.network(x_path)
    controls = []
    for step_index in range(int(x_path.shape[1]) - 1):
        controls.append(
            model.control_map(
                HamiltonianControlInputs(
                    step_index=int(step_index),
                    x_prefix=x_path[:, : step_index + 1, :],
                    expected_adjoint_next=moments.expected_adjoint_next[:, step_index, :],
                    expected_adjoint_noise_next=moments.expected_adjoint_noise_next[:, step_index, :],
                    parameters=model.parameters,
                )
            )
        )
    return torch.stack(controls, dim=1)


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    run_dir = ensure_run_dir(args.run_dir or default_run_dir(prefix="multivariate_diagonal_dt"))
    config = _make_config(args)
    grid = DiscreteTimeGrid(
        T=float(config.T),
        num_steps=int(config.num_steps),
        observed_indices=observed_indices_from_every(
            num_steps=int(config.num_steps),
            observed_every=int(args.observed_every),
        ),
    )
    target = sample_multivariate_diagonal_paths(
        num_paths=int(args.num_target_paths),
        config=config,
        seed=int(args.seed),
        device=device,
        dtype=dtype,
    )
    evaluation_seed = derive_stream_seed(base_seed=int(args.seed), stream_offset=20) or int(args.seed)
    heldout_seed = derive_stream_seed(base_seed=evaluation_seed, stream_offset=1)
    sample_seed = derive_stream_seed(base_seed=evaluation_seed, stream_offset=2)
    heldout = sample_multivariate_diagonal_paths(
        num_paths=int(args.num_heldout_paths),
        config=config,
        seed=heldout_seed,
        device=device,
        dtype=dtype,
    )
    variance_target = target.effective_sigma_path.pow(2) if str(args.reference_variance_source) == "oracle" else None
    reference_kernel = fit_local_gaussian_reference_kernel(
        target_paths=target.x_path,
        grid=grid,
        variance_target_path=variance_target,
        ridge=float(args.reference_ridge),
        variance_ridge=args.reference_variance_ridge,
        variance_shrinkage=float(args.reference_variance_shrinkage),
        log_variance_bias_correction=float(args.reference_log_variance_bias_correction),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.reference_sigma_max),
    )
    model = DiscreteMPModel(
        architecture=DiscreteMPArchitectureConfig(
            state_dim=int(config.state_dim),
            noise_dim=int(config.state_dim),
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
            terminal_weight=1.0,
            increments_weight=2.0,
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
    sigma_eval = heldout.effective_sigma_path[: int(generated_paths.shape[0])]
    drift_eval = heldout.drift_path[: int(generated_paths.shape[0])]
    teacher_control = _controls_on_path(model, target_eval)
    teacher_alpha, teacher_sigma = teacher_control.split(int(config.state_dim), dim=-1)
    metrics = {
        "run_dir": str(run_dir),
        "config": {
            "case": str(args.case),
            "multivariate_diagonal": config.to_dict(),
            "seed": int(args.seed),
            "evaluation_seed": int(evaluation_seed),
            "heldout_seed": int(heldout_seed) if heldout_seed is not None else None,
            "sample_seed": int(sample_seed) if sample_seed is not None else None,
            "lambda_scale": float(args.lambda_scale),
            "kappa_scale": float(args.kappa_scale),
            "reference_variance_source": str(args.reference_variance_source),
            "reference": reference_kernel.summary(),
        },
        "fit_final": fit_result.final_metrics,
        "stylized": stylized_fact_metrics(
            generated_paths,
            target_eval,
            lags=tuple(lag for lag in (1, 2, 5, 10) if lag < int(config.num_steps)) or (1,),
            include_swd=True,
            swd_projections=128,
        ),
        "terminal_swd": sliced_wasserstein_distance(
            generated_paths[:, -1:, :].repeat(1, 2, 1),
            target_eval[:, -1:, :].repeat(1, 2, 1),
            num_projections=128,
            seed=1234,
        ),
        "teacher_sigma_rmse": float(torch.sqrt(torch.mean((teacher_sigma - sigma_eval).pow(2))).item()),
        "teacher_sigma_corr": _corrcoef(teacher_sigma, sigma_eval),
        "teacher_alpha_rmse": float(torch.sqrt(torch.mean((teacher_alpha - drift_eval).pow(2))).item()),
        "teacher_alpha_corr": _corrcoef(teacher_alpha, drift_eval),
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
            from path_dt_experiments.plots import plot_correlation_heatmaps
            import matplotlib.pyplot as plt

            fig = plot_correlation_heatmaps(
                target_paths=target_eval.detach().cpu(),
                generated_paths=generated_paths.detach().cpu(),
                output_path=run_dir / "return_correlation_heatmaps.png",
            )
            plt.close(fig)
        except ImportError as exc:
            write_json(run_dir / "plot_error.json", {"error": str(exc)})
    print(f"wrote multivariate diagonal run to {run_dir}")
    print(f"metrics: {metrics}")


if __name__ == "__main__":
    main()
