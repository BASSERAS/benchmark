#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_RUNNER = (
    REPO_ROOT / "experiments" / "scripts" / "run_real_data_tuning_candidate.py"
)


SOURCE_CANDIDATES: tuple[dict[str, float | str], ...] = (
    {
        "identifier": "lambda1_lr5e-4_ridge1e-2_eta2",
        "lambda_scale": 1.0,
        "lr": 5e-4,
        "ridge_lambda": 1e-2,
        "eta": 2.0,
    },
    {
        "identifier": "lambda5_lr5e-4_ridge1e-2_eta2",
        "lambda_scale": 5.0,
        "lr": 5e-4,
        "ridge_lambda": 1e-2,
        "eta": 2.0,
    },
    {
        "identifier": "lambda10_lr5e-4_ridge1e-2_eta2",
        "lambda_scale": 10.0,
        "lr": 5e-4,
        "ridge_lambda": 1e-2,
        "eta": 2.0,
    },
    {
        "identifier": "lambda5_lr2.5e-4_ridge1e-2_eta2",
        "lambda_scale": 5.0,
        "lr": 2.5e-4,
        "ridge_lambda": 1e-2,
        "eta": 2.0,
    },
    {
        "identifier": "lambda5_lr1e-3_ridge1e-2_eta2",
        "lambda_scale": 5.0,
        "lr": 1e-3,
        "ridge_lambda": 1e-2,
        "eta": 2.0,
    },
    {
        "identifier": "lambda5_lr5e-4_ridge1e-3_eta2",
        "lambda_scale": 5.0,
        "lr": 5e-4,
        "ridge_lambda": 1e-3,
        "eta": 2.0,
    },
    {
        "identifier": "lambda5_lr5e-4_ridge1e-1_eta2",
        "lambda_scale": 5.0,
        "lr": 5e-4,
        "ridge_lambda": 1e-1,
        "eta": 2.0,
    },
    {
        "identifier": "lambda5_lr5e-4_ridge1e-2_eta1",
        "lambda_scale": 5.0,
        "lr": 5e-4,
        "ridge_lambda": 1e-2,
        "eta": 1.0,
    },
    {
        "identifier": "lambda5_lr5e-4_ridge1e-2_eta4",
        "lambda_scale": 5.0,
        "lr": 5e-4,
        "ridge_lambda": 1e-2,
        "eta": 4.0,
    },
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one deterministic shard of the real-data source stability sweep."
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs" / "real_data_tuning" / "source",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("shard-index must lie in [0, num-shards)")
    jobs = [
        (candidate, seed)
        for candidate in SOURCE_CANDIDATES
        for seed in (1, 2)
    ]
    selected = [
        job
        for index, job in enumerate(jobs)
        if index % int(args.num_shards) == int(args.shard_index)
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    sweep_manifest = {
        "selection_scope": "training_and_validation_only",
        "test_or_event_data_seen": False,
        "phase": "source",
        "seeds": [1, 2],
        "steps": int(args.steps),
        "bank_size": int(args.bank_size),
        "grad_clip_norm": 50.0,
        "candidates": list(SOURCE_CANDIDATES),
    }
    (args.output_root / "sweep_manifest.json").write_text(
        json.dumps(sweep_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for candidate, seed in selected:
        identifier = str(candidate["identifier"])
        run_dir = args.output_root / identifier / f"seed_{seed}"
        manifest = run_dir / "candidate_manifest.json"
        if manifest.is_file():
            print(f"skip completed {identifier}/seed_{seed}", flush=True)
            continue
        command = [
            sys.executable,
            str(CANDIDATE_RUNNER),
            "--phase",
            "source",
            "--run-dir",
            str(run_dir),
            "--seed",
            str(seed),
            "--device",
            str(args.device),
            "--steps",
            str(args.steps),
            "--lr",
            str(candidate["lr"]),
            "--ridge-lambda",
            str(candidate["ridge_lambda"]),
            "--lambda-scale",
            str(candidate["lambda_scale"]),
            "--eta",
            str(candidate["eta"]),
            "--joint-weight",
            "0.25",
            "--grad-clip-norm",
            "50",
            "--bank-size",
            str(args.bank_size),
            "--sample-batch-size",
            str(args.bank_size),
            "--log-every",
            "50",
        ]
        print(f"start {identifier}/seed_{seed} on {args.device}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    print(f"complete shard {args.shard_index}/{args.num_shards}", flush=True)


if __name__ == "__main__":
    main()
