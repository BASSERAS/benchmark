#!/usr/bin/env python3
"""Run one locked validation-only SBTS delayed-memory horizon/seed job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, choices=(128, 256, 512), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 3, 4), required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, environment: dict[str, str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if bool(protocol.get("test_split_access_authorized", True)):
        raise RuntimeError("the locked protocol must forbid final-test access")
    sequence_length = int(args.sequence_length)
    seed = int(args.seed)
    horizon = protocol["horizons"][str(sequence_length)]
    dataset_root = Path(horizon["dataset_root"])
    train_path = dataset_root / "train.npy"
    if _sha256(train_path) != horizon["train_sha256"]:
        raise RuntimeError("training dataset hash differs from the locked protocol")
    if _sha256(dataset_root / "disc.npy") != horizon["validation_sha256"]:
        raise RuntimeError("validation dataset hash differs from the locked protocol")

    output_dir = args.run_root.resolve() / f"n{sequence_length}" / f"seed_{seed}" / "sbts"
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = output_dir / f"generated_paths_8192x{sequence_length}.npy"
    metrics_path = output_dir / "validation_metrics.json"
    if (output_dir / "COMPLETE.json").exists():
        raise FileExistsError(f"completed output already exists: {output_dir}")

    sbts = protocol["sbts"]
    environment = os.environ.copy()
    environment.update(
        {
            "NUMBA_CACHE_DIR": str(Path(sbts["numba_cache_dir"])),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    try:
        generation_command = [
                sys.executable,
                str(REPO_ROOT / "experiments" / "scripts" / "run_drawdown_memory_sbts.py"),
                "--train-data", str(train_path),
                "--output", str(bank_path),
                "--seed", str(seed),
                "--num-paths", str(sbts["num_paths"]),
                "--h", str(sbts["h"]),
                "--K", str(sbts["K"]),
                "--N-pi", str(sbts["N_pi"]),
                "--workers", str(sbts["workers"]),
            ]
        if bool(sbts.get("deterministic_numba_seed", False)):
            generation_command.append("--deterministic-numba-seed")
        _run(generation_command, environment=environment)
        generation_manifest_path = output_dir / "generation_manifest.json"
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "experiments" / "scripts" / "evaluate_long_horizon_sbts.py"),
                "--dataset-root", str(dataset_root),
                "--generated-data", str(bank_path),
                "--generation-manifest", str(generation_manifest_path),
                "--sequence-length", str(sequence_length),
                "--seed", str(seed),
                "--output", str(metrics_path),
                "--subsample-size", str(protocol["evaluation"]["subsample_size"]),
                "--swd-projections", str(protocol["evaluation"]["swd_projections"]),
                "--device", "cpu",
            ],
            environment=environment,
        )

        gate: dict[str, object] = {"required": sequence_length == 128 and seed == 0}
        if bool(gate["required"]):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
            expected = protocol["n128_reproduction_gate"]
            checks = {
                "paper_delay_gap_row_agreement": abs(
                    float(metrics["future_rv_hit_gap_error"])
                    - float(expected["paper_four_seed_mean"]["future_rv_hit_gap_error"])
                )
                <= float(expected["paper_row_abs_tolerance"]),
                "paper_early_r2_row_agreement": abs(
                    float(metrics["early_history_incremental_r2_error"])
                    - float(expected["paper_four_seed_mean"]["early_history_incremental_r2_error"])
                )
                <= float(expected["paper_row_abs_tolerance"]),
            }
            if bool(expected.get("require_archived_bank_sha256", False)):
                checks["generated_bank_sha256"] = (
                    _sha256(bank_path)
                    == expected["archived_paper_seed0_bank_sha256"]
                )
            if bool(expected.get("require_archived_validation_metrics", False)):
                checks["future_rv_hit_gap_validation"] = abs(
                    float(metrics["future_rv_hit_gap_error"])
                    - float(
                        expected["archived_bank_validation_metrics"][
                            "future_rv_hit_gap_error"
                        ]
                    )
                ) <= float(expected["recomputed_metric_abs_tolerance"])
                checks["early_history_r2_validation"] = abs(
                    float(metrics["early_history_incremental_r2_error"])
                    - float(
                        expected["archived_bank_validation_metrics"][
                            "early_history_incremental_r2_error"
                        ]
                    )
                ) <= float(expected["recomputed_metric_abs_tolerance"])
            gate.update({"checks": checks, "pass": bool(all(checks.values()))})
            (output_dir / "REPRODUCTION_GATE.json").write_text(
                json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if not bool(gate["pass"]):
                raise RuntimeError(f"N=128 SBTS reproduction gate failed: {checks}")

        generation = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
        complete = {
            "status": "complete",
            "sequence_length": sequence_length,
            "seed": seed,
            "decision_split": "disc.npy",
            "test_split_used": False,
            "official_formula_unchanged": True,
            "generated_bank_sha256": generation["generated_bank_sha256"],
            "reproduction_gate": gate,
        }
        (output_dir / "COMPLETE.json").write_text(
            json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception as error:
        failure = {
            "status": "failed",
            "sequence_length": sequence_length,
            "seed": seed,
            "decision_split": "disc.npy",
            "test_split_used": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
