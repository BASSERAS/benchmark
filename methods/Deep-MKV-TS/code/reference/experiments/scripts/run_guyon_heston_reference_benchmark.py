from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteTimeGrid,
    fit_guyon_lekeufack_reference_kernel,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_centered_log_paths(
    path: Path,
    *,
    num_paths: int | None,
    num_time_points: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float, tuple[int, int]]:
    prices = np.load(path, mmap_mode="r")
    if prices.ndim != 2:
        raise ValueError("official Heston training data must have shape (B, N + 1)")
    if int(num_time_points) < 2 or int(num_time_points) > int(prices.shape[1]):
        raise ValueError("num_time_points must lie between 2 and the training path width")
    count = int(prices.shape[0]) if num_paths is None else min(int(num_paths), int(prices.shape[0]))
    if count < 8:
        raise ValueError("at least eight fitting paths are required")
    selected = np.asarray(prices[:count, : int(num_time_points)], dtype=np.float64)
    if not np.isfinite(selected).all() or np.any(selected <= 0.0):
        raise ValueError("official Heston training paths must be finite and strictly positive")
    initial = selected[:, :1]
    if not np.allclose(initial, initial[0, 0], rtol=0.0, atol=1e-12):
        raise ValueError("all official Heston paths must share the same initial price")
    centered = np.log(selected / initial)
    tensor = torch.from_numpy(centered).to(device=device, dtype=dtype).unsqueeze(-1)
    return tensor, float(initial[0, 0]), (int(prices.shape[0]), int(prices.shape[1]))


@torch.no_grad()
def generate_reference_paths(
    reference,
    *,
    num_paths: int,
    seed: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    if int(num_paths) < 1 or int(batch_size) < 1:
        raise ValueError("num_paths and batch_size must be positive")
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed))
    noise = torch.randn(
        int(num_paths),
        int(reference.grid.num_steps),
        1,
        generator=cpu_generator,
        dtype=dtype,
        device="cpu",
    )
    path_chunks: list[torch.Tensor] = []
    sigma_chunks: list[torch.Tensor] = []
    sqrt_dt = math.sqrt(float(reference.grid.dt))
    for start in range(0, int(num_paths), int(batch_size)):
        stop = min(start + int(batch_size), int(num_paths))
        current_noise = noise[start:stop].to(device=device)
        paths = torch.zeros(
            stop - start,
            int(reference.grid.num_steps) + 1,
            1,
            device=device,
            dtype=dtype,
        )
        sigmas: list[torch.Tensor] = []
        for step in range(int(reference.grid.num_steps)):
            drift, sigma = reference.evaluate(paths[:, : step + 1, :], step_index=step)
            paths[:, step + 1, :] = (
                paths[:, step, :]
                + drift * float(reference.grid.dt)
                + sigma * sqrt_dt * current_noise[:, step, :]
            )
            sigmas.append(sigma)
        path_chunks.append(paths.cpu())
        sigma_chunks.append(torch.stack(sigmas, dim=1).cpu())
    paths = torch.cat(path_chunks, dim=0)
    sigmas = torch.cat(sigma_chunks, dim=0)
    sigma_max = reference.sigma_max
    diagnostics = {
        "sigma_mean": float(sigmas.mean().item()),
        "sigma_std": float(sigmas.std(unbiased=False).item()),
        "sigma_q01": float(torch.quantile(sigmas, 0.01).item()),
        "sigma_q50": float(torch.quantile(sigmas, 0.50).item()),
        "sigma_q99": float(torch.quantile(sigmas, 0.99).item()),
        "sigma_lower_bound_fraction": float((sigmas <= float(reference.sigma_min) + 1e-7).float().mean().item()),
        "sigma_upper_bound_fraction": (
            0.0
            if sigma_max is None
            else float((sigmas >= float(sigma_max) - 1e-7).float().mean().item())
        ),
    }
    return paths, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and sample the standalone Guyon--Lekeufack Heston reference."
    )
    parser.add_argument("--train-paths", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-fit-paths", type=int, default=None)
    parser.add_argument("--num-generate-paths", type=int, default=8192)
    parser.add_argument("--num-time-points", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps-per-year", type=float, default=250.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--variance-shrinkage", type=float, default=0.1)
    parser.add_argument("--log-variance-bias-correction", type=float, default=0.6)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    num_steps = int(args.num_time_points) - 1
    grid = DiscreteTimeGrid(
        T=float(num_steps) / float(args.steps_per_year),
        num_steps=num_steps,
    )

    started = time.perf_counter()
    training_paths, initial_price, source_shape = load_centered_log_paths(
        args.train_paths,
        num_paths=args.num_fit_paths,
        num_time_points=int(args.num_time_points),
        device=device,
        dtype=dtype,
    )
    fit_started = time.perf_counter()
    reference = fit_guyon_lekeufack_reference_kernel(
        target_paths=training_paths,
        grid=grid,
        ridge=float(args.ridge),
        variance_shrinkage=float(args.variance_shrinkage),
        log_variance_bias_correction=float(args.log_variance_bias_correction),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    fit_seconds = time.perf_counter() - fit_started

    generation_started = time.perf_counter()
    centered_paths, generation_diagnostics = generate_reference_paths(
        reference,
        num_paths=int(args.num_generate_paths),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        device=device,
        dtype=dtype,
    )
    generation_seconds = time.perf_counter() - generation_started
    prices = float(initial_price) * torch.exp(centered_paths[..., 0].double())
    prices = prices.clamp_min(1e-6)
    prices[:, 0] = float(initial_price)
    output_path = run_dir / (
        f"generated_paths_{int(args.num_generate_paths)}x{int(args.num_time_points)}.npy"
    )
    np.save(output_path, prices.numpy())

    checkpoint = {
        "grid": {"T": float(grid.T), "num_steps": int(grid.num_steps)},
        "reference_summary": reference.summary(),
        "trend_lambdas": reference.trend_lambdas,
        "activity_lambdas": reference.activity_lambdas,
        "trend_weight": reference.trend_weight,
        "activity_weight": reference.activity_weight,
        "beta_0": reference.beta_0.detach().cpu(),
        "beta_1": reference.beta_1.detach().cpu(),
        "beta_2": reference.beta_2.detach().cpu(),
        "initial_trend": reference.initial_trend.detach().cpu(),
        "initial_activity": reference.initial_activity.detach().cpu(),
        "drift_kernel": {
            "drift_coefficients": tuple(value.detach().cpu() for value in reference.drift_kernel.drift_coefficients),
            "feature_means": tuple(value.detach().cpu() for value in reference.drift_kernel.feature_means),
            "feature_stds": tuple(value.detach().cpu() for value in reference.drift_kernel.feature_stds),
        },
    }
    torch.save(checkpoint, run_dir / "reference_checkpoint.pt")
    report = {
        "method": "standalone_guyon_lekeufack_reference_with_fitted_causal_drift",
        "seed": int(args.seed),
        "train_path": str(args.train_paths.resolve()),
        "train_path_sha256": _sha256(args.train_paths.resolve()),
        "source_shape": list(source_shape),
        "fit_shape": list(training_paths.shape),
        "output_path": str(output_path),
        "output_shape": list(prices.shape),
        "output_dtype": str(prices.numpy().dtype),
        "initial_price": float(initial_price),
        "fit_seconds": float(fit_seconds),
        "generation_seconds": float(generation_seconds),
        "total_seconds": float(time.perf_counter() - started),
        "reference": reference.summary(),
        "generation": generation_diagnostics,
        "protocol": {
            "uses_train_split_only": True,
            "uses_latent_heston_variance": False,
            "deep_mkv_correction": False,
            "fitted_causal_drift_retained": True,
            "num_fit_paths": int(training_paths.shape[0]),
            "num_generated_paths": int(prices.shape[0]),
            "num_time_points": int(prices.shape[1]),
            "steps_per_year": float(args.steps_per_year),
        },
    }
    _write_json(run_dir / "manifest.json", report)
    print(f"generated={output_path}")
    print(f"reference={reference.summary()}")
    print(f"generation={generation_diagnostics}")


if __name__ == "__main__":
    main()
