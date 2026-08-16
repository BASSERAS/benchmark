#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt.causal_reference import (  # noqa: E402
    ShrunkCausalVolatilityReferenceKernel,
    _reference_rollout,
    calibrate_shrunk_causal_reference_kernel,
    fit_causal_volatility_reference_kernel,
)
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402
from deep_mkv_gen_path_dt.reference import (  # noqa: E402
    fit_local_gaussian_reference_kernel,
)
from path_dt_experiments.real_data_generation import (  # noqa: E402
    load_frozen_real_data_fit,
    materialize_deep_mkv_adapter_dataset,
)
from path_dt_experiments.real_data_protocol import (  # noqa: E402
    normalized_prices_to_log_paths,
    real_data_law_metrics,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.shadowing import (  # noqa: E402
    ShadowingFeatureConfig,
    evaluate_path_shadowing,
)
from run_basseras_heston_path_shadowing import (  # noqa: E402
    NUM_STEPS,
    _load_training_data,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the train-fitted causal-reference shrinkage frontier "
            "on the frozen real-data validation split."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=REPO_ROOT / "data" / "ICAIF_3Y" / "canonical_rth_128",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("/home/samer/scenarios/BASSERAS-benchmark"),
    )
    parser.add_argument("--protocol", default="primary_2026_holdout")
    parser.add_argument("--index", choices=("ES", "NQ", "RTY", "YM"), default="ES")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=(0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _states_to_prices(states: np.ndarray, transform) -> np.ndarray:
    return_states = np.zeros_like(states)
    return_states[:, 1:] = (
        np.diff(states, axis=1)
        * math.sqrt(float(transform.dt))
        / float(transform.sigma)
    )
    return transform.inverse(return_states, dtype=np.float32)


def main() -> None:
    args = _parse_args()
    run_dir = ensure_run_dir(args.run_dir)
    output = run_dir / "reference_frontier.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    fit = load_frozen_real_data_fit(
        args.canonical_root,
        protocol_identifier=args.protocol,
        index=args.index,
    )
    dataset_root = run_dir / "adapter_dataset"
    materialize_deep_mkv_adapter_dataset(fit, dataset_root)
    device = torch.device(args.device)
    model_dt = 1.0 / NUM_STEPS
    target_paths, transform, _ = _load_training_data(
        args.benchmark_root,
        device=device,
        dtype=torch.float32,
        state_representation="log_price",
        custom_dataset_root=dataset_root,
        model_dt=model_dt,
    )
    grid = DiscreteTimeGrid(T=1.0, num_steps=NUM_STEPS)
    anchor = fit_local_gaussian_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        ridge=1e-3,
        variance_ridge=1e-3,
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    causal = fit_causal_volatility_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        rolling_windows=(5, 10, 20, 32),
        ewma_half_lives=(5, 20, 60),
        downside_windows=(10, 32),
        standardized_feature_clip=4.0,
        drift_ridge=1e-3,
        variance_ridges=(1e-5, 1e-4, 1e-3),
        crossfit_folds=3,
        crossfit_seed=1701,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    train_returns = np.diff(np.log(fit.train_prices.astype(np.float64)), axis=1)
    sigma_std_floor = 0.25 * float(train_returns.std(ddof=0)) / math.sqrt(model_dt)
    _, calibration = calibrate_shrunk_causal_reference_kernel(
        target_paths=target_paths,
        causal_reference=causal,
        anchor_reference=anchor,
        causal_weights=tuple(args.weights),
        calibration_paths=int(args.bank_size),
        calibration_seed=271828,
        offset_steps=8,
        offset_relaxation=0.5,
        return_std_tolerance=0.03,
        sigma_std_floor=sigma_std_floor,
        sigma_cap_fraction_limit=0.01,
        path_rv_std_ratio_min=0.0,
        path_rv_std_ratio_max=math.inf,
        path_rv_quantile_spread_ratio_min=0.0,
        path_rv_quantile_spread_ratio_max=math.inf,
        clustering_lags=(1, 5, 20),
        sigma_min=1e-3,
        sigma_max=0.6,
    )

    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(args.seed) + 70_000)
    noise = torch.randn(
        int(args.bank_size),
        NUM_STEPS,
        1,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    x0 = target_paths[:, 0].mean(dim=0, keepdim=True).expand(
        int(args.bank_size), -1
    )
    validation_log = normalized_prices_to_log_paths(fit.validation_prices)
    rows = []
    for candidate in calibration["candidates"]:
        reference = ShrunkCausalVolatilityReferenceKernel(
            causal_reference=causal,
            anchor_reference=anchor,
            causal_log_variance_weight=float(
                candidate["causal_log_variance_weight"]
            ),
            log_variance_offset=float(candidate["log_variance_offset"]),
            sigma_min=1e-3,
            sigma_max=0.6,
        )
        states, _ = _reference_rollout(reference, noise=noise, x0=x0)
        prices = _states_to_prices(states[..., 0].cpu().numpy(), transform)
        generated_log = normalized_prices_to_log_paths(prices)
        law = real_data_law_metrics(generated_log, validation_log)
        shadow = evaluate_path_shadowing(
            real_paths=validation_log,
            generated_paths=generated_log,
            prefix_length=65,
            future_horizon=32,
            top_k=256,
            feature_config=ShadowingFeatureConfig(),
            search_backend="auto",
        )
        row = {
            "weight": float(candidate["causal_log_variance_weight"]),
            "offset": float(candidate["log_variance_offset"]),
            "train_calibration": candidate,
            "validation_law_metrics": law,
            "validation_shadowing_metrics": shadow.metrics,
        }
        rows.append(row)
        print(
            f"weight={row['weight']:.3f} "
            f"rv_crps={shadow.metrics['realized_volatility_crps']:.6g} "
            f"rv_cov90={shadow.metrics['realized_volatility_coverage_90']:.3f} "
            f"rv_width90={shadow.metrics['realized_volatility_band_width_90']:.6g}",
            flush=True,
        )
    payload = {
        "selection_scope": "training_and_validation_only",
        "test_or_event_data_seen": False,
        "index": args.index,
        "seed": int(args.seed),
        "bank_size": int(args.bank_size),
        "common_random_numbers": True,
        "calibration": calibration,
        "rows": rows,
    }
    write_json(output, payload)


if __name__ == "__main__":
    main()
