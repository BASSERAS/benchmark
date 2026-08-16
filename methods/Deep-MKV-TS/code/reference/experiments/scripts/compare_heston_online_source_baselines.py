#!/usr/bin/env python3
"""Rescore Heston seed-zero baselines against the locked validation split."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_matched_control_synthetic_validation import _evaluate_common  # noqa: E402


BENCHMARK_ROOT = Path("/home/samer/scenarios/BASSERAS-benchmark")
DEFAULT_OUTPUT = (
    REPO_ROOT / "runs" / "heston_online_source_baseline_comparison_seed0_20260805"
)
INTERNAL_ROOT = (
    REPO_ROOT / "runs" / "heston_online_fitted_drift_volatility_only_seed0_20260805"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _conditional_errors(metrics: dict[str, object]) -> dict[str, float]:
    memory = metrics["conditional_path_memory"]
    target = memory["target"]
    generated = memory["generated"]
    return {
        "prefix_future_rv_correlation_error": abs(
            float(target["prefix_future_rv_correlation"])
            - float(generated["prefix_future_rv_correlation"])
        ),
        "prefix_high_low_future_rv_gap_error": abs(
            float(target["prefix_high_low_future_rv_gap"])
            - float(generated["prefix_high_low_future_rv_gap"])
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_run_dir(args.output_dir.resolve())
    target_path = BENCHMARK_ROOT / "dataset" / "Heston" / "heston_S_disc_8192x128.npy"
    paths = {
        "Fitted reference": INTERNAL_ROOT / "reference" / "validation_bank.npy",
        "Nested MP": (
            REPO_ROOT
            / "runs"
            / "heston_fixed_drift_seed0_ablation_20260805"
            / "fitted_reference_drift"
            / "volatility_only_nested_mp"
            / "validation_bank.npy"
        ),
        "Online MP source": (
            INTERNAL_ROOT
            / "volatility_only_online_mp"
            / "source_validation_bank.npy"
        ),
        "SBTS": (
            BENCHMARK_ROOT
            / "methods"
            / "SBTS"
            / "generated_paths"
            / "seed_0"
            / "generated_paths_8192x128.npy"
        ),
        "CSDI": (
            BENCHMARK_ROOT
            / "methods"
            / "CSDI"
            / "generated_paths"
            / "seed_0"
            / "generated_paths_8192x128.npy"
        ),
        "LS4": (
            BENCHMARK_ROOT
            / "methods"
            / "LS4"
            / "generated_paths"
            / "seed_0"
            / "generated_paths_8192x128.npy"
        ),
    }
    target = np.asarray(np.load(target_path, allow_pickle=False), dtype=np.float64)
    device = torch.device(args.device)
    summary: dict[str, object] = {
        "selection_scope": "validation_only",
        "test_split_loaded": False,
        "seed": 0,
        "target": str(target_path),
        "target_sha256": _sha256(target_path),
        "common_evaluator": "run_matched_control_synthetic_validation._evaluate_common",
        "methods": {},
    }
    for label, path in paths.items():
        generated = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if generated.shape != target.shape:
            raise ValueError(f"shape mismatch for {label}: {generated.shape} != {target.shape}")
        metrics = _evaluate_common(
            target_prices=target,
            generated=generated,
            device=device,
            seed=0,
        )
        method_dir = ensure_run_dir(output_dir / label.lower().replace(" ", "_"))
        metrics_path = method_dir / "metrics.json"
        write_json(metrics_path, metrics)
        path_law = metrics["path_law"]
        summary["methods"][label] = {
            "generated": str(path),
            "generated_sha256": _sha256(path),
            "metrics": str(metrics_path),
            "headline": {
                key: float(path_law[key])
                for key in (
                    "path_swd_normalized",
                    "terminal_w1_normalized",
                    "increment_w1_normalized",
                    "realized_volatility_w1_normalized",
                    "maximum_drawdown_w1_normalized",
                    "return_qq_rmse_normalized",
                    "absolute_return_acf_rmse",
                    "squared_return_acf_rmse",
                    "timewise_return_std_rmse_normalized",
                )
            },
            "conditional": _conditional_errors(metrics),
        }
    write_json(output_dir / "SUMMARY.json", summary)
    print(f"wrote {output_dir / 'SUMMARY.json'}", flush=True)


if __name__ == "__main__":
    main()
