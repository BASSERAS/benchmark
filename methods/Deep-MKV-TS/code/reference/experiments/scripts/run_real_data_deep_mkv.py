#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from path_dt_experiments.real_data_generation import (  # noqa: E402
    load_frozen_real_data_fit,
    materialize_deep_mkv_adapter_dataset,
)
from path_dt_experiments.real_data_protocol import (  # noqa: E402
    FROZEN_REAL_DATA_PROTOCOLS,
)


BACKEND = REPO_ROOT / "experiments" / "scripts" / "run_basseras_heston_path_shadowing.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the frozen specific-entropy Deep-MKV-Gen specification on one "
            "real-data training slice and generate its scenario bank."
        )
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=REPO_ROOT / "data" / "ICAIF_3Y" / "canonical_rth_128",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=REPO_ROOT / "runs" / "real_data_protocol" / "generated" / "deep_mkv",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=REPO_ROOT / "runs" / "real_data_protocol" / "training" / "deep_mkv",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("/home/samer/scenarios/BASSERAS-benchmark"),
    )
    parser.add_argument(
        "--protocol",
        choices=tuple(item.identifier for item in FROZEN_REAL_DATA_PROTOCOLS),
        default="primary_2026_holdout",
    )
    parser.add_argument("--index", choices=("ES", "NQ", "RTY", "YM"), default="ES")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2, 3, 4), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--stage", choices=("all", "train", "bank"), default="all")
    parser.add_argument("--source-training-steps", type=int, default=2000)
    parser.add_argument("--joint-training-steps", type=int, default=1600)
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--sample-batch-size", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _backend_command(
    args: argparse.Namespace,
    *,
    stage: str,
    dataset_root: Path,
    run_dir: Path,
    bank_output: Path,
    reference_sigma_std_floor: float,
) -> list[str]:
    return [
        sys.executable,
        str(BACKEND),
        "--benchmark-root",
        str(args.benchmark_root),
        "--custom-dataset-root",
        str(dataset_root),
        "--run-dir",
        str(run_dir),
        "--stage",
        stage,
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--data-layout",
        "path_shadowing_4096",
        "--state-representation",
        "log_price",
        "--reference-kind",
        "calibrated_causal",
        "--time-horizon",
        "1.0",
        "--causal-calibration-sigma-std-floor",
        str(reference_sigma_std_floor),
        "--source-training-steps",
        str(args.source_training_steps),
        "--joint-training-steps",
        str(args.joint_training_steps),
        "--bank-size",
        str(args.bank_size),
        "--bank-seed",
        str(args.seed),
        "--sample-batch-size",
        str(args.sample_batch_size),
        "--bank-output",
        str(bank_output),
    ]


def main() -> None:
    args = _parse_args()
    if args.source_training_steps < 1 or args.joint_training_steps < 1:
        raise ValueError("training step counts must be positive")
    if args.bank_size < 1 or args.sample_batch_size < 1:
        raise ValueError("bank sizes must be positive")
    fit = load_frozen_real_data_fit(
        args.canonical_root,
        protocol_identifier=args.protocol,
        index=args.index,
    )
    run_dir = (
        args.training_root
        / fit.scenario_bank_protocol
        / fit.index
        / f"seed_{args.seed}"
    )
    dataset_root = run_dir / "adapter_dataset"
    bank_output = (
        args.generated_root
        / fit.scenario_bank_protocol
        / fit.index
        / f"seed_{args.seed}.npy"
    )
    manifest_path = run_dir / "real_data_adapter_manifest.json"
    if args.stage in {"all", "train"}:
        checkpoint = run_dir / "model_checkpoint.pt"
        if checkpoint.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint}")
    if args.stage in {"all", "bank"} and bank_output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite bank: {bank_output}")
    run_dir.mkdir(parents=True, exist_ok=True)
    materialize_deep_mkv_adapter_dataset(fit, dataset_root)
    train_returns = np.diff(np.log(fit.train_prices.astype(np.float64)), axis=1)
    train_return_std = float(train_returns.std(ddof=0))
    model_dt = 1.0 / (fit.train_prices.shape[1] - 1)
    reference_sigma_std_floor = 0.25 * train_return_std / math.sqrt(model_dt)
    started = time.perf_counter()
    commands: list[list[str]] = []
    if args.stage in {"all", "train"}:
        commands.append(
            _backend_command(
                args,
                stage="train",
                dataset_root=dataset_root,
                run_dir=run_dir,
                bank_output=bank_output,
                reference_sigma_std_floor=reference_sigma_std_floor,
            )
        )
    if args.stage in {"all", "bank"}:
        commands.append(
            _backend_command(
                args,
                stage="bank",
                dataset_root=dataset_root,
                run_dir=run_dir,
                bank_output=bank_output,
                reference_sigma_std_floor=reference_sigma_std_floor,
            )
        )
    for command in commands:
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    payload = {
        "model": "Deep-MKV-Gen",
        "specification": (
            "specific entropy with calibrated causal reference and joint "
            "prefix/future realized-volatility discrepancy"
        ),
        "replication_kind": "independent_model_fit",
        "seed": int(args.seed),
        "requested_protocol": fit.requested_protocol,
        "scenario_bank_protocol": fit.scenario_bank_protocol,
        "index": fit.index,
        "train_sessions": int(fit.train_prices.shape[0]),
        "train_first_date": str(fit.train_dates[0]),
        "train_last_date": str(fit.train_dates[-1]),
        "train_array_sha256": fit.train_sha256,
        "validation_or_test_seen_by_fit": False,
        "time_horizon": 1.0,
        "train_one_step_return_std": train_return_std,
        "reference_sigma_std_floor": reference_sigma_std_floor,
        "reference_sigma_std_floor_rule": (
            "0.25 times train return std divided by sqrt(model dt)"
        ),
        "source_training_steps": int(args.source_training_steps),
        "joint_training_steps": int(args.joint_training_steps),
        "bank_size": int(args.bank_size),
        "bank_output": str(bank_output.resolve()),
        "backend": str(BACKEND.resolve()),
        "backend_is_legacy_adapter_only": True,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"completed Deep-MKV-Gen real-data stage {args.stage}: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
