from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteTimeGrid,
    fit_causal_volatility_reference_kernel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a train-only lag-aware causal Gaussian reference and require "
            "a reference-only rollout to recover delayed drawdown memory before "
            "any maximum-principle retraining."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "drawdown_memory_seed0_comparison_20260730"
            / "data"
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    parser.add_argument(
        "--drawdown-threshold-quantiles",
        type=float,
        nargs="+",
        default=(0.20, 0.30, 0.40, 0.60, 0.80),
    )
    parser.add_argument(
        "--drawdown-memory-half-lives",
        type=int,
        nargs="+",
        default=(20, 60),
    )
    parser.add_argument(
        "--drawdown-memory-lags",
        type=int,
        nargs="+",
        default=(8, 16, 24, 32, 40, 48),
    )
    parser.add_argument("--drawdown-gate-temperature", type=float, default=0.005)
    parser.add_argument(
        "--reference-variance-ridges",
        type=float,
        nargs="+",
        default=(1e-5, 1e-4, 1e-3),
    )
    parser.add_argument("--reference-crossfit-folds", type=int, default=3)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.6)
    parser.add_argument(
        "--generic-basis",
        action="store_true",
        help=(
            "Fit the generic causal reference used by the headline control: "
            "omit thresholded and lagged drawdown-memory features."
        ),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also score the frozen test split after validation scoring.",
    )
    return parser.parse_args()


def default_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "runs" / f"drawdown_memory_lagged_reference_gate_{timestamp}"


def load_prices(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(f"{path} must contain finite positive price paths")
    return values


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fitted_thresholds(
    target_paths: torch.Tensor,
    *,
    history_cutoff: int,
    quantiles: tuple[float, ...],
) -> tuple[float, ...]:
    if (
        not quantiles
        or any(not 0.0 < value < 1.0 for value in quantiles)
        or tuple(sorted(set(quantiles))) != quantiles
    ):
        raise ValueError("threshold quantiles must be strictly increasing in (0, 1)")
    prefix = target_paths[:, : int(history_cutoff) + 1, 0]
    drawdown = torch.cummax(prefix, dim=1).values - prefix
    maximum = drawdown.amax(dim=1)
    values = torch.quantile(maximum, maximum.new_tensor(quantiles))
    return tuple(float(value) for value in values.detach().cpu().tolist())


def reference_rollout(
    *,
    reference,
    num_paths: int,
    batch_size: int,
    seed: int,
    s0: float,
) -> tuple[np.ndarray, np.ndarray]:
    sequence_length = int(reference.grid.num_steps) + 1
    prices = np.empty((int(num_paths), sequence_length), dtype=np.float32)
    sigmas = np.empty((int(num_paths), sequence_length - 1), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(
        (int(num_paths), sequence_length - 1),
        dtype=np.float32,
    )
    sqrt_dt = math.sqrt(float(reference.grid.dt))
    device = reference.drift_coefficients.device
    dtype = reference.drift_coefficients.dtype
    with torch.no_grad():
        for start in range(0, int(num_paths), int(batch_size)):
            end = min(start + int(batch_size), int(num_paths))
            batch_noise = torch.from_numpy(noise[start:end]).to(
                device=device,
                dtype=dtype,
            )
            points = [
                torch.zeros((end - start, 1), device=device, dtype=dtype)
            ]
            batch_sigmas = []
            for step in range(int(reference.grid.num_steps)):
                prefix = torch.stack(points, dim=1)
                drift, sigma = reference.evaluate(prefix, step_index=step)
                points.append(
                    points[-1]
                    + drift * float(reference.grid.dt)
                    + sigma * sqrt_dt * batch_noise[:, step : step + 1]
                )
                batch_sigmas.append(sigma)
            log_paths = torch.stack(points, dim=1)[..., 0]
            prices[start:end] = (
                float(s0) * torch.exp(log_paths)
            ).cpu().numpy().astype(np.float32)
            sigmas[start:end] = (
                torch.stack(batch_sigmas, dim=1)[..., 0]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return prices, sigmas


def evaluate_bank(
    *,
    dataset_root: Path,
    bank_path: Path,
    output_path: Path,
    split_filename: str = "disc.npy",
) -> dict[str, object]:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "scripts" / "evaluate_drawdown_memory.py"),
            "--train-data",
            str(dataset_root / "train.npy"),
            "--test-data",
            str(dataset_root / split_filename),
            "--generated-data",
            str(bank_path),
            "--dataset-manifest",
            str(dataset_root / "manifest.json"),
            "--output",
            str(output_path),
        ],
        check=True,
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def lag_coefficient_ranking(reference) -> list[dict[str, float]]:
    coefficients = (
        reference.log_variance_coefficients[1:, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    grouped: dict[int, list[float]] = {}
    for name, coefficient in zip(reference.feature_names, coefficients, strict=True):
        marker = "_lag_"
        if marker not in name:
            continue
        lag = int(name.rsplit(marker, 1)[1])
        grouped.setdefault(lag, []).append(float(coefficient))
    return sorted(
        (
            {
                "lag": float(lag),
                "coefficient_rms": float(
                    np.sqrt(np.mean(np.square(values)))
                ),
                "coefficient_abs_max": float(np.max(np.abs(values))),
            }
            for lag, values in grouped.items()
        ),
        key=lambda item: item["coefficient_rms"],
        reverse=True,
    )


def lag_feature_coefficient_ranking(reference) -> list[dict[str, object]]:
    coefficients = (
        reference.log_variance_coefficients[1:, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    return sorted(
        (
            {
                "feature": name,
                "coefficient": float(coefficient),
                "absolute_coefficient": abs(float(coefficient)),
            }
            for name, coefficient in zip(
                reference.feature_names,
                coefficients,
                strict=True,
            )
            if "_lag_" in name
        ),
        key=lambda item: item["absolute_coefficient"],
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    run_dir = Path(args.run_dir) if args.run_dir is not None else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (dataset_root / "manifest.json").read_text(encoding="utf-8")
    )
    config = manifest["configuration"]
    train_prices = load_prices(dataset_root / "train.npy")
    validation_prices = load_prices(dataset_root / "disc.npy")
    validation_sigma = np.asarray(
        np.load(dataset_root / "disc_sigma.npy"),
        dtype=np.float64,
    )
    train_log = np.log(train_prices / train_prices[:, :1])
    validation_log = np.log(validation_prices / validation_prices[:, :1])
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    target = torch.from_numpy(train_log).unsqueeze(-1).to(
        device=device,
        dtype=dtype,
    )
    validation = torch.from_numpy(validation_log).unsqueeze(-1).to(
        device=device,
        dtype=dtype,
    )
    num_steps = int(config["sequence_length"]) - 1
    grid = DiscreteTimeGrid(T=num_steps * float(config["dt"]), num_steps=num_steps)
    quantiles = tuple(float(value) for value in args.drawdown_threshold_quantiles)
    thresholds = (
        ()
        if args.generic_basis
        else fitted_thresholds(
            target,
            history_cutoff=48,
            quantiles=quantiles,
        )
    )
    lags = (
        ()
        if args.generic_basis
        else tuple(int(value) for value in args.drawdown_memory_lags)
    )
    half_lives = (
        ()
        if args.generic_basis
        else tuple(int(value) for value in args.drawdown_memory_half_lives)
    )
    print(
        f"fitting {'generic' if args.generic_basis else 'lag-aware'} reference "
        "with thresholds="
        f"{thresholds}, half_lives={half_lives}, lags={lags}",
        flush=True,
    )
    reference = fit_causal_volatility_reference_kernel(
        target_paths=target,
        grid=grid,
        rolling_windows=(5, 10, 20, 32),
        ewma_half_lives=(5, 20, 60),
        downside_windows=(10, 32),
        drawdown_thresholds=thresholds,
        drawdown_memory_half_lives=half_lives,
        drawdown_memory_lags=lags,
        drawdown_gate_temperature=float(args.drawdown_gate_temperature),
        drift_ridge=float(args.ridge_lambda),
        variance_ridges=tuple(float(value) for value in args.reference_variance_ridges),
        crossfit_folds=int(args.reference_crossfit_folds),
        crossfit_seed=1701 + int(args.seed),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    split = int(config["split_index"])
    with torch.no_grad():
        _, predicted_split_sigma = reference.evaluate(
            validation[:, : split + 1],
            step_index=split,
        )
    predicted = predicted_split_sigma[:, 0].cpu().numpy().astype(np.float64)
    truth = validation_sigma[:, split]
    prediction_metrics = {
        "split_index": split,
        "sigma_rmse": float(np.sqrt(np.mean((predicted - truth) ** 2))),
        "sigma_mae": float(np.mean(np.abs(predicted - truth))),
        "sigma_correlation": float(np.corrcoef(predicted, truth)[0, 1]),
        "predicted_sigma_mean": float(predicted.mean()),
        "target_sigma_mean": float(truth.mean()),
    }
    del validation
    if device.type == "cuda":
        torch.cuda.empty_cache()
    prices, generated_sigma = reference_rollout(
        reference=reference,
        num_paths=int(args.bank_size),
        batch_size=int(args.sample_batch_size),
        seed=730_101 + int(args.seed),
        s0=float(config["s0"]),
    )
    bank_path = run_dir / (
        f"generated_reference_paths_{int(args.bank_size)}x"
        f"{int(config['sequence_length'])}.npy"
    )
    np.save(bank_path, prices)
    np.save(run_dir / "generated_reference_sigma.npy", generated_sigma)
    validation_metrics = evaluate_bank(
        dataset_root=dataset_root,
        bank_path=bank_path,
        output_path=run_dir / "validation_metrics.json",
    )
    test_metrics = (
        evaluate_bank(
            dataset_root=dataset_root,
            bank_path=bank_path,
            output_path=run_dir / "test_metrics.json",
            split_filename="test.npy",
        )
        if args.evaluate_test
        else None
    )
    target_memory = validation_metrics["target_memory"]
    generated_memory = validation_metrics["generated_memory"]
    gate_errors = {
        "incremental_r2": abs(
            generated_memory["early_history_incremental_r2"]
            - target_memory["early_history_incremental_r2"]
        ),
        "hit_future_rv_correlation": abs(
            generated_memory["early_hit_future_rv_correlation"]
            - target_memory["early_hit_future_rv_correlation"]
        ),
        "future_rv_hit_gap": abs(
            generated_memory["future_rv_hit_gap"]
            - target_memory["future_rv_hit_gap"]
        ),
    }
    tolerances = {
        "incremental_r2": 0.05,
        "hit_future_rv_correlation": 0.05,
        "future_rv_hit_gap": 0.03,
    }
    gate_passed = all(
        gate_errors[name] <= tolerances[name] for name in tolerances
    )
    payload = {
        "design": {
            "basis": "generic_causal" if args.generic_basis else "drawdown_memory_aware",
            "dataset_root": str(dataset_root.resolve()),
            "fit_split": "train.npy",
            "validation_split": "disc.npy",
            "test_split_used_for_selection": False,
            "test_split_scored": bool(args.evaluate_test),
            "sigma_labels_used_for_fit": False,
            "threshold_quantiles": list(quantiles),
            "fitted_thresholds": list(thresholds),
            "drawdown_memory_half_lives": list(half_lives),
            "drawdown_memory_lags": list(lags),
            "feature_names": list(reference.feature_names),
        },
        "reference": reference.summary(),
        "train_fitted_lag_coefficient_ranking": lag_coefficient_ranking(reference),
        "train_fitted_lag_feature_coefficient_ranking": (
            lag_feature_coefficient_ranking(reference)
        ),
        "heldout_split_sigma_prediction": prediction_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "gate": {
            "passed": gate_passed,
            "errors": gate_errors,
            "tolerances": tolerances,
        },
    }
    write_json(run_dir / "summary.json", payload)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "reference": reference.summary(),
                "train_fitted_lag_coefficient_ranking": payload[
                    "train_fitted_lag_coefficient_ranking"
                ],
                "heldout_split_sigma_prediction": prediction_metrics,
                "target_memory": target_memory,
                "generated_memory": generated_memory,
                "gate": payload["gate"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
