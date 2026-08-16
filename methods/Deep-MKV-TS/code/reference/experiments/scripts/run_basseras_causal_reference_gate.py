from __future__ import annotations

import argparse
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
    calibrate_shrunk_causal_reference_kernel,
    fit_causal_volatility_reference_kernel,
)
from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl  # noqa: E402
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_basseras_heston_path_shadowing import (  # noqa: E402
    _build_components,
    _build_model,
    _load_pdf_evaluator,
    _load_training_data,
)
from run_basseras_pr_coverage_diagnostic import (  # noqa: E402
    _evaluate_bank,
    _paired_difference,
    _sample_price_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gate a cross-fitted volatility-aware sigma_ref before any MP retraining."
        )
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("/home/samer/scenarios/BASSERAS-benchmark"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-representation", default="log_price")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--reference-ridge", type=float, default=1e-3)
    parser.add_argument("--reference-variance-shrinkage", type=float, default=0.1)
    parser.add_argument(
        "--reference-log-variance-bias-correction",
        type=float,
        default=0.6,
    )
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.6)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--rolling-windows", type=int, nargs="+", default=(5, 10, 20, 32))
    parser.add_argument("--ewma-half-lives", type=int, nargs="+", default=(5, 20, 60))
    parser.add_argument("--downside-windows", type=int, nargs="+", default=(10, 32))
    parser.add_argument("--variance-ridges", type=float, nargs="+", default=(1e-5, 1e-4, 1e-3))
    parser.add_argument("--crossfit-folds", type=int, default=3)
    parser.add_argument("--crossfit-seed", type=int, default=1701)
    parser.add_argument(
        "--causal-weights",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.4, 0.55, 0.7, 0.75, 0.8, 0.85, 1.0),
    )
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--calibration-seed", type=int, default=271828)
    parser.add_argument("--calibration-offset-steps", type=int, default=3)
    parser.add_argument(
        "--calibration-return-std-tolerance",
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
    parser.add_argument("--validation-query-seed", type=int, default=4)
    parser.add_argument("--validation-reference-seed", type=int, default=5)
    parser.add_argument("--model-bank-seed", type=int, default=6001)
    parser.add_argument("--num-query-paths", type=int, default=512)
    parser.add_argument("--num-reference-paths", type=int, default=4096)
    parser.add_argument("--num-bank-paths", type=int, default=65_536)
    parser.add_argument("--sample-batch-size", type=int, default=32_768)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20230814)
    parser.add_argument("--coverage-gate", type=float, default=0.80)
    parser.add_argument("--sigma-std-gate", type=float, default=0.045)
    parser.add_argument("--return-std-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--cum-crps-tolerance", type=float, default=5e-4)
    return parser.parse_args()


def _return_std(prices: np.ndarray) -> float:
    returns = np.log(
        np.asarray(prices, dtype=np.float64)[:, 1:]
        / np.asarray(prices, dtype=np.float64)[:, :-1]
    )
    return float(returns.std(ddof=0))


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir(Path(args.run_dir))
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    target_paths, transform, preprocessing = _load_training_data(
        Path(args.benchmark_root),
        device=device,
        dtype=dtype,
        state_representation="log_price",
    )
    grid, architecture, old_control, base, _joint = _build_components(
        args,
        target_paths=target_paths,
        path_scale=float(transform.sigma) / math.sqrt(float(transform.dt)),
    )
    print("fitting cross-fitted causal volatility reference", flush=True)
    causal_reference = fit_causal_volatility_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        rolling_windows=tuple(int(value) for value in args.rolling_windows),
        ewma_half_lives=tuple(int(value) for value in args.ewma_half_lives),
        downside_windows=tuple(int(value) for value in args.downside_windows),
        drift_ridge=float(args.reference_ridge),
        variance_ridges=tuple(float(value) for value in args.variance_ridges),
        crossfit_folds=int(args.crossfit_folds),
        crossfit_seed=int(args.crossfit_seed),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    print(
        "calibrating causal log-variance shrinkage on train-only recursive rollouts",
        flush=True,
    )
    calibrated_reference, calibration = calibrate_shrunk_causal_reference_kernel(
        target_paths=target_paths,
        causal_reference=causal_reference,
        anchor_reference=old_control.reference_kernel,
        causal_weights=tuple(float(value) for value in args.causal_weights),
        calibration_paths=int(args.calibration_paths),
        calibration_seed=int(args.calibration_seed),
        offset_steps=int(args.calibration_offset_steps),
        return_std_tolerance=float(args.calibration_return_std_tolerance),
        sigma_std_floor=float(args.sigma_std_gate),
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
    new_control = SpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=float(args.eta),
        reference_kernel=calibrated_reference,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    old_model = _build_model(
        architecture=architecture,
        grid=grid,
        control=old_control,
        discrepancy=base,
        device=device,
        dtype=dtype,
        seed=int(args.seed),
    )
    new_model = _build_model(
        architecture=architecture,
        grid=grid,
        control=new_control,
        discrepancy=base,
        device=device,
        dtype=dtype,
        seed=int(args.seed),
    )

    evaluator, _ = _load_pdf_evaluator(Path(args.benchmark_root))
    query_prices = evaluator.generate_heston_bank(
        int(args.num_query_paths),
        int(args.validation_query_seed),
    )
    reference_prices = evaluator.generate_heston_bank(
        int(args.num_reference_paths),
        int(args.validation_reference_seed),
    )
    boot_idx = np.random.default_rng(int(args.bootstrap_seed)).integers(
        0,
        int(args.num_query_paths),
        size=(int(args.bootstrap_replicates), int(args.num_query_paths)),
    )

    results = {}
    per_path = {}
    for name, model in (
        ("old_reference", old_model),
        ("calibrated_causal_reference", new_model),
    ):
        print(f"sampling {name} with P=R=0 and common noise", flush=True)
        bank, controls = _sample_price_bank(
            model,
            transform=transform,
            num_paths=int(args.num_bank_paths),
            batch_size=int(args.sample_batch_size),
            seed=int(args.model_bank_seed),
            zero_p=True,
            zero_r=True,
        )
        metrics, current_per_path = _evaluate_bank(
            evaluator,
            bank=bank,
            query_prices=query_prices,
            reference_prices=reference_prices,
            top_k=int(args.top_k),
            boot_idx=boot_idx,
        )
        metrics["controls"] = controls
        metrics["unconditional_return_std"] = _return_std(bank)
        results[name] = metrics
        per_path[name] = current_per_path
        rv = metrics["quantities"]["rv"]
        print(
            f"{name}: rv_crps={rv['crps']['value']:.6f} "
            f"rv_cov90={rv['coverage90']['value']:.4f} "
            f"rv_width90={rv['width90']['value']:.6f} "
            f"return_std={metrics['unconditional_return_std']:.6f} "
            f"sigma_std={controls['sigma_std']:.6f}",
            flush=True,
        )

    old = results["old_reference"]
    new = results["calibrated_causal_reference"]
    train_return_std = float(preprocessing["pooled_log_return_sigma"])
    return_relative_error = abs(
        float(new["unconditional_return_std"]) / train_return_std - 1.0
    )
    gate = {
        "coverage_at_least_threshold": bool(
            float(new["quantities"]["rv"]["coverage90"]["value"])
            >= float(args.coverage_gate)
        ),
        "rv_crps_improved": bool(
            float(new["quantities"]["rv"]["crps"]["value"])
            < float(old["quantities"]["rv"]["crps"]["value"])
        ),
        "return_std_within_tolerance": bool(
            return_relative_error <= float(args.return_std_relative_tolerance)
        ),
        "sigma_std_at_least_threshold": bool(
            float(new["controls"]["sigma_std"]) >= float(args.sigma_std_gate)
        ),
        "cum_crps_within_tolerance": bool(
            float(new["quantities"]["cum"]["crps"]["value"])
            <= float(old["quantities"]["cum"]["crps"]["value"])
            + float(args.cum_crps_tolerance)
        ),
    }
    gate["passes_all"] = bool(all(gate.values()))
    payload = {
        "protocol": {
            "benchmark_seed3_queries_used": False,
            "benchmark_test_seed1_reference_used": False,
            "validation_query_seed": int(args.validation_query_seed),
            "validation_reference_seed": int(args.validation_reference_seed),
            "model_bank_seed": int(args.model_bank_seed),
            "common_model_noise": True,
            "P": 0,
            "R": 0,
            "num_query_paths": int(args.num_query_paths),
            "num_reference_paths": int(args.num_reference_paths),
            "num_bank_paths": int(args.num_bank_paths),
            "K": int(args.top_k),
        },
        "causal_reference": {
            "summary": causal_reference.summary(),
            "feature_names": list(causal_reference.feature_names),
            "rolling_windows": list(causal_reference.rolling_windows),
            "ewma_half_lives": list(causal_reference.ewma_half_lives),
            "downside_windows": list(causal_reference.downside_windows),
        },
        "calibrated_reference": {
            "summary": calibrated_reference.summary(),
            "train_only_calibration": calibration,
        },
        "gate_thresholds": {
            "coverage90": float(args.coverage_gate),
            "sigma_std": float(args.sigma_std_gate),
            "return_std_relative_tolerance": float(
                args.return_std_relative_tolerance
            ),
            "cum_crps_tolerance": float(args.cum_crps_tolerance),
        },
        "return_std_relative_error": float(return_relative_error),
        "results": results,
        "paired_calibrated_causal_minus_old": _paired_difference(
            evaluator,
            candidate=per_path["calibrated_causal_reference"],
            baseline=per_path["old_reference"],
            boot_idx=boot_idx,
        ),
        "gate": gate,
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(
        {
            "drift_coefficients": causal_reference.drift_coefficients.detach().cpu(),
            "log_variance_coefficients": (
                causal_reference.log_variance_coefficients.detach().cpu()
            ),
            "feature_mean": causal_reference.feature_mean.detach().cpu(),
            "feature_std": causal_reference.feature_std.detach().cpu(),
            "metadata": {
                "grid_T": float(grid.T),
                "grid_num_steps": int(grid.num_steps),
                "rolling_windows": causal_reference.rolling_windows,
                "ewma_half_lives": causal_reference.ewma_half_lives,
                "downside_windows": causal_reference.downside_windows,
                "selected_variance_ridge": float(
                    causal_reference.selected_variance_ridge
                ),
                "crossfit_folds": int(causal_reference.crossfit_folds),
                "crossfit_nll": float(causal_reference.crossfit_nll),
                "causal_log_variance_weight": float(
                    calibrated_reference.causal_log_variance_weight
                ),
                "log_variance_offset": float(
                    calibrated_reference.log_variance_offset
                ),
                "calibration_seed": int(args.calibration_seed),
                "calibration_paths": int(args.calibration_paths),
                "sigma_min": float(causal_reference.sigma_min),
                "sigma_max": float(causal_reference.sigma_max),
            },
        },
        run_dir / "causal_reference.pt",
    )
    np.save(run_dir / "validation_query_prices.npy", query_prices)
    np.save(run_dir / "validation_reference_prices.npy", reference_prices)
    print(f"reference gate passes_all={gate['passes_all']}", flush=True)
    print(f"wrote reference gate to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
