#!/usr/bin/env python3
"""Write the locked validation-only SBTS long-horizon comparison manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (0, 1, 3, 4)
DATASET_ROOTS = {
    128: REPO_ROOT / "runs" / "drawdown_memory_seed0_comparison_20260730" / "data",
    256: REPO_ROOT / "data" / "drawdown_memory_scaling" / "n256",
    512: REPO_ROOT / "data" / "drawdown_memory_scaling" / "n512",
}
METHOD_ROOTS = {
    horizon: REPO_ROOT / "runs" / f"volatility_only_nested_n{horizon}_4seed_20260804"
    for horizon in DATASET_ROOTS
}
ARCHIVED_PAPER_SEED0 = (
    REPO_ROOT
    / "runs"
    / "paper_synthetic_4seed_20260801"
    / "drawdown"
    / "sbts"
    / "seed_0"
    / "generated_paths_8192x128.npy"
)
OFFICIAL_ADAPTER = Path(
    "/home/samer/scenarios/BASSERAS-benchmark/methods/SBTS/code/sbts_generate.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    horizons: dict[str, object] = {}
    for horizon, dataset_root in DATASET_ROOTS.items():
        method_root = METHOD_ROOTS[horizon]
        summary_path = method_root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        protocol = summary["protocol"]
        if bool(protocol.get("test_split_used", True)):
            raise RuntimeError(f"existing N={horizon} comparison is not validation-only")
        if tuple(int(seed) for seed in protocol["seeds"]) != SEEDS:
            raise RuntimeError(f"existing N={horizon} seed set does not match {SEEDS}")
        for seed in SEEDS:
            scaling = json.loads(
                (method_root / f"seed_{seed}" / "scaling_protocol.json").read_text(
                    encoding="utf-8"
                )
            )
            if bool(scaling.get("test_split_used", True)):
                raise RuntimeError(f"N={horizon}, seed={seed} used a final test split")
        horizons[str(horizon)] = {
            "dataset_root": str(dataset_root.resolve()),
            "train_sha256": _sha256(dataset_root / "train.npy"),
            "validation_file": "disc.npy",
            "validation_sha256": _sha256(dataset_root / "disc.npy"),
            "final_test_file_access": "forbidden and not hashed",
            "existing_method_run_root": str(method_root.resolve()),
            "existing_summary": str(summary_path.resolve()),
            "existing_summary_sha256": _sha256(summary_path),
            "existing_arms": {
                "reference": "generic_reference_only",
                "volatility_only_nested_mp": "generic_ce_nested_volatility_only",
                "direct_pathwise": "direct_bptt_volatility_only",
            },
        }

    payload = {
        "protocol_version": 2,
        "created_utc": "2026-08-04T00:00:00Z",
        "amendment": {
            "predecessor": str(
                (args.output.parent / "LOCKED_PROTOCOL.json").resolve()
            ),
            "reason": (
                "The v1 reproduction audit found that the official Python RNG "
                "seed does not initialize the Numba RNG used for Brownian draws."
            ),
            "change": (
                "Seed the Numba RNG once inside each worker; retain the official "
                "SBTS formula, parameters, worker count, paper-row tolerances, "
                "data, and evaluation protocol."
            ),
            "v1_failed_artifacts_retained": True,
        },
        "question": (
            "Does SBTS preserve delayed path dependence as the horizon grows, "
            "relative to the same reference used by volatility-only nested MP?"
        ),
        "decision_scope": "validation-only matched long-horizon SBTS comparison",
        "test_split_access_authorized": False,
        "seeds": list(SEEDS),
        "seed_zero_first": True,
        "failure_retention": "every missing, failed, or invalid seed remains in completion rates",
        "comparison_arms": [
            "generic_reference_only",
            "SBTS",
            "generic_ce_nested_volatility_only",
            "direct_bptt_volatility_only",
        ],
        "sbts": {
            "official_adapter": str(OFFICIAL_ADAPTER),
            "official_adapter_sha256": _sha256(OFFICIAL_ADAPTER),
            "official_formula_unchanged": True,
            "deterministic_numba_seed": True,
            "seed_patch": (
                "path_dt_experiments.sbts_reproducible_worker: Numba RNG "
                "initialization only"
            ),
            "h": 0.05,
            "K": 20,
            "N_pi": 50,
            "workers": 16,
            "num_paths": 8192,
            "numba_cache_dir": "/tmp/deep_mkv_sbts_long_horizon_numba_cache",
            "seed_interpretation": "independent SBTS sampler randomness; SBTS has no fitted parameters",
            "gpu_use": "none; peak GPU allocation must be zero",
        },
        "evaluation": {
            "decision_split": "disc.npy",
            "generated_paths": 8192,
            "subsample_size": 1024,
            "swd_projections": 256,
            "common_index_rng": "numpy default_rng(20260804 + 1000*N + seed), target draw then generated draw",
            "path_mmd": {
                "estimator": "square root of biased RBF MMD2",
                "features": "open-centered log paths excluding initial point",
                "normalization": "train-path coordinate mean and scalar standard deviation",
                "bandwidths": [0.5, 1.0, 2.0, 4.0],
            },
            "gap_closure": "(reference_error - candidate_error) / reference_error",
            "metric_groups": {
                "distribution": [
                    "path_mmd", "path_swd_normalized", "terminal_w1_normalized",
                    "increment_w1_normalized", "realized_volatility_w1_normalized",
                    "maximum_drawdown_w1_normalized", "return_qq_rmse_normalized",
                ],
                "dependence": [
                    "return_acf_rmse", "absolute_return_acf_rmse",
                    "squared_return_acf_rmse",
                ],
                "path_memory": [
                    "future_rv_hit_gap_error",
                    "early_hit_future_rv_correlation_error",
                    "early_history_incremental_r2_error", "early_hit_rate_error",
                ],
                "local_calibration": [
                    "timewise_return_mean_rmse_normalized",
                    "timewise_return_std_rmse_normalized",
                ],
            },
            "resources": [
                "training_wall_seconds", "cuda_incremental_peak_allocated_bytes",
                "peak process/child RSS for SBTS", "completion and failure reason",
            ],
        },
        "n128_reproduction_gate": {
            "archived_paper_seed0_bank": str(ARCHIVED_PAPER_SEED0.resolve()),
            "archived_paper_seed0_bank_sha256": _sha256(ARCHIVED_PAPER_SEED0),
            "archived_bank_validation_metrics": {
                "future_rv_hit_gap_error": 0.1635555155952023,
                "early_history_incremental_r2_error": 0.2955311410207405,
            },
            "recomputed_metric_abs_tolerance": 1e-10,
            "require_archived_bank_sha256": False,
            "require_archived_validation_metrics": False,
            "archived_hash_status": (
                "diagnostic only because the legacy adapter did not seed its Numba RNG"
            ),
            "paper_four_seed_mean": {
                "future_rv_hit_gap_error": 0.16345,
                "early_history_incremental_r2_error": 0.28508,
            },
            "paper_row_abs_tolerance": 0.015,
            "failure_action": "stop and fix comparability before N=256 or N=512",
        },
        "seed_zero_long_horizon_soundness_gate": {
            "required_horizons": [256, 512],
            "checks": [
                "official adapter/configuration unchanged",
                "finite positive 8192-path bank with exact horizon shape",
                "all reported metrics finite",
                "validation-only certificate true",
                "zero peak GPU allocation",
            ],
            "failure_action": "retain failure and do not launch remaining seeds",
        },
        "horizons": horizons,
        "stop_condition": "aggregate four locked seeds, report, and do not load final tests",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(_sha256(args.output))


if __name__ == "__main__":
    main()
