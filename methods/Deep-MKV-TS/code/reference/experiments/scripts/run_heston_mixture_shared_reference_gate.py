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
    calibrate_shrunk_causal_reference_kernel,
    fit_causal_volatility_reference_kernel,
    fit_local_gaussian_reference_kernel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and gate a label-free Heston-mixture causal reference whose "
            "observable prefix basis can also be reused by the MP conditional "
            "expectation estimator."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "heston_parameter_mixture_seed0_20260730"
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    parser.add_argument(
        "--rolling-windows",
        type=int,
        nargs="+",
        default=(5, 10, 20, 40, 64),
    )
    parser.add_argument(
        "--ewma-half-lives",
        type=int,
        nargs="+",
        default=(5, 10, 20, 40, 80),
    )
    parser.add_argument(
        "--downside-windows",
        type=int,
        nargs="+",
        default=(10, 20, 40),
    )
    parser.add_argument(
        "--persistence-lags",
        type=int,
        nargs="+",
        default=(1, 2, 5, 10, 20, 40),
    )
    parser.add_argument(
        "--leverage-lags",
        type=int,
        nargs="+",
        default=(0, 1, 2, 5, 10, 20, 40),
    )
    parser.add_argument(
        "--local-volatility-windows",
        type=int,
        nargs="+",
        default=(5, 10, 20),
    )
    parser.add_argument(
        "--reference-variance-ridges",
        type=float,
        nargs="+",
        default=(1e-5, 1e-4, 1e-3),
    )
    parser.add_argument("--reference-crossfit-folds", type=int, default=3)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--standardized-feature-clip", type=float, default=None)
    parser.add_argument("--anchor-variance-shrinkage", type=float, default=0.1)
    parser.add_argument(
        "--anchor-log-variance-bias-correction",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--causal-weights",
        type=float,
        nargs="+",
        default=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0),
    )
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--calibration-offset-steps", type=int, default=8)
    parser.add_argument("--calibration-offset-relaxation", type=float, default=0.5)
    parser.add_argument(
        "--calibration-return-std-tolerance",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--calibration-sigma-std-floor",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--calibration-sigma-cap-fraction-limit",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--calibration-clustering-lags",
        type=int,
        nargs="+",
        default=(1, 5, 20),
    )
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.8)
    return parser.parse_args()


def default_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "runs" / f"heston_mixture_shared_reference_gate_{timestamp}"


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


def evaluate(
    *,
    root: Path,
    generated: Path,
    output: Path,
) -> dict[str, object]:
    subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "experiments"
                / "scripts"
                / "evaluate_heston_parameter_mixture.py"
            ),
            "--train-data",
            str(root / "data" / "train.npy"),
            "--test-data",
            str(root / "data" / "disc.npy"),
            "--generated-data",
            str(generated),
            "--oracle",
            str(root / "oracle" / "oracle.joblib"),
            "--oracle-gate-report",
            str(root / "oracle" / "gate_report.json"),
            "--output",
            str(output),
        ],
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def reference_rollout(
    *,
    reference,
    num_paths: int,
    batch_size: int,
    seed: int,
    s0: float,
) -> np.ndarray:
    sequence_length = int(reference.grid.num_steps) + 1
    prices = np.empty((int(num_paths), sequence_length), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(
        (int(num_paths), sequence_length - 1),
        dtype=np.float32,
    )
    sqrt_dt = math.sqrt(float(reference.grid.dt))
    fitted_reference = getattr(reference, "causal_reference", reference)
    device = fitted_reference.drift_coefficients.device
    dtype = fitted_reference.drift_coefficients.dtype
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
            for step in range(int(reference.grid.num_steps)):
                prefix = torch.stack(points, dim=1)
                _, sigma = reference.evaluate(prefix, step_index=step)
                points.append(
                    points[-1]
                    + sigma * sqrt_dt * batch_noise[:, step : step + 1]
                )
            log_paths = torch.stack(points, dim=1)[..., 0]
            prices[start:end] = (
                float(s0) * torch.exp(log_paths)
            ).cpu().numpy().astype(np.float32)
    return prices


def primary_metrics(payload: dict[str, object]) -> dict[str, float]:
    mixture = payload["mixture_fidelity"]
    parameters = mixture["parameters"]
    observable = payload["observable_fidelity"]
    return {
        "regime_tvd": float(mixture["regime_proportion_tvd"]),
        "average_parameter_normalized_wasserstein": float(
            np.mean(
                [
                    parameters[name]["support_normalized_wasserstein"]
                    for name in ("theta", "xi", "rho")
                ]
            )
        ),
        "theta_normalized_wasserstein": float(
            parameters["theta"]["support_normalized_wasserstein"]
        ),
        "xi_normalized_wasserstein": float(
            parameters["xi"]["support_normalized_wasserstein"]
        ),
        "rho_normalized_wasserstein": float(
            parameters["rho"]["support_normalized_wasserstein"]
        ),
        "realized_volatility_wasserstein": float(
            observable["realized_volatility_wasserstein"]
        ),
        "leverage_curve_rmse": float(observable["leverage_curve_rmse_lags_0_20"]),
        "abs_return_acf_rmse": float(
            observable["abs_return_acf_rmse_lags_1_50"]
        ),
        "squared_return_acf_rmse": float(
            observable["squared_return_acf_rmse_lags_1_50"]
        ),
        "terminal_log_price_ks": float(observable["terminal_log_price_ks"]),
    }


def coefficient_ranking(reference) -> list[dict[str, object]]:
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
                "coefficient": float(value),
                "absolute_coefficient": abs(float(value)),
            }
            for name, value in zip(
                reference.feature_names,
                coefficients,
                strict=True,
            )
        ),
        key=lambda item: item["absolute_coefficient"],
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    run_dir = Path(args.run_dir) if args.run_dir is not None else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (root / "data" / "manifest.json").read_text(encoding="utf-8")
    )
    train_prices = load_prices(root / "data" / "train.npy")
    train_log = np.log(train_prices / train_prices[:, :1])
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    target = torch.from_numpy(train_log).unsqueeze(-1).to(
        device=device,
        dtype=dtype,
    )
    num_steps = int(manifest["sequence_length"]) - 1
    grid = DiscreteTimeGrid(
        T=num_steps * float(manifest["dt"]),
        num_steps=num_steps,
    )
    feature_config = {
        "rolling_windows": tuple(int(value) for value in args.rolling_windows),
        "ewma_half_lives": tuple(int(value) for value in args.ewma_half_lives),
        "downside_windows": tuple(int(value) for value in args.downside_windows),
        "include_expanding_moments": True,
        "persistence_lags": tuple(int(value) for value in args.persistence_lags),
        "leverage_lags": tuple(int(value) for value in args.leverage_lags),
        "local_volatility_windows": tuple(
            int(value) for value in args.local_volatility_windows
        ),
        "standardized_feature_clip": (
            None
            if args.standardized_feature_clip is None
            else float(args.standardized_feature_clip)
        ),
    }
    print(f"fitting shared Heston evidence basis {feature_config}", flush=True)
    causal_reference = fit_causal_volatility_reference_kernel(
        target_paths=target,
        grid=grid,
        **feature_config,
        drift_ridge=float(args.ridge_lambda),
        variance_ridges=tuple(float(value) for value in args.reference_variance_ridges),
        crossfit_folds=int(args.reference_crossfit_folds),
        crossfit_seed=2701 + int(args.seed),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    print("fitting stable local Gaussian anchor", flush=True)
    anchor_reference = fit_local_gaussian_reference_kernel(
        target_paths=target,
        grid=grid,
        ridge=float(args.ridge_lambda),
        variance_ridge=float(args.ridge_lambda),
        variance_shrinkage=float(args.anchor_variance_shrinkage),
        log_variance_bias_correction=float(
            args.anchor_log_variance_bias_correction
        ),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    print(
        "selecting causal/anchor volatility blend on train-only recursive rollouts",
        flush=True,
    )
    reference, calibration = calibrate_shrunk_causal_reference_kernel(
        target_paths=target,
        causal_reference=causal_reference,
        anchor_reference=anchor_reference,
        causal_weights=tuple(float(value) for value in args.causal_weights),
        calibration_paths=int(args.calibration_paths),
        calibration_seed=410_003 + int(args.seed),
        offset_steps=int(args.calibration_offset_steps),
        offset_relaxation=float(args.calibration_offset_relaxation),
        return_std_tolerance=float(args.calibration_return_std_tolerance),
        sigma_std_floor=float(args.calibration_sigma_std_floor),
        sigma_cap_fraction_limit=float(
            args.calibration_sigma_cap_fraction_limit
        ),
        clustering_lags=tuple(
            int(value) for value in args.calibration_clustering_lags
        ),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    selected = calibration["selected"]
    print(
        "selected train-only calibration: "
        f"weight={selected['causal_log_variance_weight']:.3f} "
        f"offset={selected['log_variance_offset']:.6f} "
        f"return_std={selected['return_std']:.6f} "
        f"sigma_std={selected['sigma_std']:.6f} "
        f"score={selected['selection_score']:.6f}",
        flush=True,
    )
    bank = reference_rollout(
        reference=reference,
        num_paths=int(args.bank_size),
        batch_size=int(args.sample_batch_size),
        seed=731_001 + int(args.seed),
        s0=float(manifest["s0"]),
    )
    bank_path = run_dir / (
        f"generated_reference_paths_{int(args.bank_size)}x"
        f"{int(manifest['sequence_length'])}.npy"
    )
    np.save(bank_path, bank)
    reference_metrics = evaluate(
        root=root,
        generated=bank_path,
        output=run_dir / "reference_validation_metrics.json",
    )
    floor_metrics = evaluate(
        root=root,
        generated=root / "data" / "oracle_validation.npy",
        output=run_dir / "independent_floor_validation_metrics.json",
    )
    primary = primary_metrics(reference_metrics)
    floor_primary = primary_metrics(floor_metrics)
    tolerances = {
        "regime_tvd": 0.30,
        "average_parameter_normalized_wasserstein": 0.13,
        "realized_volatility_wasserstein": 0.012,
        "leverage_curve_rmse": 0.025,
        "abs_return_acf_rmse": 0.030,
        "squared_return_acf_rmse": 0.030,
        "terminal_log_price_ks": 0.060,
    }
    gate_passed = all(primary[name] <= limit for name, limit in tolerances.items())
    payload = {
        "design": {
            "fit_split": "train.npy",
            "validation_split": "disc.npy",
            "test_split_used": False,
            "oracle_labels_used_for_fit": False,
            "latent_variance_used_for_fit": False,
            "recursive_calibration_uses_training_paths_only": True,
            "reference_rollout": "P=R=0; zero drift and fitted sigma_ref",
            "feature_config": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in feature_config.items()
            },
            "feature_names": list(causal_reference.feature_names),
        },
        "reference": reference.summary(),
        "anchor_reference": anchor_reference.summary(),
        "train_only_calibration": calibration,
        "train_fitted_coefficient_ranking": coefficient_ranking(
            causal_reference
        ),
        "independent_floor_primary": floor_primary,
        "reference_primary": primary,
        "gate": {
            "passed": gate_passed,
            "tolerances": tolerances,
        },
        "sources": {
            "root": str(root.resolve()),
            "generated_bank": str(bank_path.resolve()),
        },
    }
    write_json(run_dir / "summary.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
