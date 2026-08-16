#!/usr/bin/env python3
"""Evaluate the frozen Y-free, fitted-drift Deep MKV real-data checkpoints.

This queue performs no fitting or selection.  It draws an independent scenario
bank from each checkpoint selected on training/validation data, then evaluates
the locked chronological-test and stress windows.  ES, NQ, and YM form the
large-capitalization main panel; RTY is deliberately outside this queue.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/samer/venvs/mfc/bin/python")
SOURCE_ROOT = (
    REPO_ROOT / "runs" / "real_yweight0_fitted_drift_3index_4seed_20260808"
)
RUN_ROOT = REPO_ROOT / "runs" / "final_yfree_fitted_drift_evaluation_20260808"
BANK_ROOT = RUN_ROOT / "generated"
EVALUATION_ROOT = RUN_ROOT / "protocol"
MODEL_NAME = "Deep-MKV-final-yfree-fitted-drift"
INDICES = ("ES", "NQ", "YM")
SEEDS = (0, 1, 3, 4)


LANES: dict[int, tuple[tuple[str, int], ...]] = {
    0: (("ES", 0), ("NQ", 0), ("YM", 0)),
    1: (("ES", 1), ("NQ", 1), ("YM", 1)),
    2: (("ES", 3), ("NQ", 3), ("YM", 3)),
    3: (("ES", 4), ("NQ", 4), ("YM", 4)),
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate(index: str, seed: int) -> Path:
    return SOURCE_ROOT / index / f"seed_{seed}"


def _frozen_protocol() -> dict[str, object]:
    checkpoints: dict[str, object] = {}
    for index in INDICES:
        checkpoints[index] = {}
        for seed in SEEDS:
            candidate = _candidate(index, seed)
            checkpoint = candidate / "source_model_checkpoint.pt"
            manifest = candidate / "candidate_manifest.json"
            if not checkpoint.is_file() or not manifest.is_file():
                raise FileNotFoundError(candidate)
            checkpoints[index][str(seed)] = {
                "candidate_directory": str(candidate.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "candidate_manifest": str(manifest.resolve()),
                "candidate_manifest_sha256": _sha256(manifest),
            }
    return {
        "experiment": "final_yfree_fitted_drift_real_evaluation_20260808",
        "created_utc": _timestamp(),
        "selection_scope": "training_and_validation_only",
        "protected_test_and_stress_access_authorized": True,
        "test_or_stress_used_for_selection": False,
        "method": {
            "name": MODEL_NAME,
            "controlled_channel": "volatility_only",
            "forward_drift": "fitted_path_dependent_reference_drift",
            "adjoint_Y_loss_weight": 0.0,
            "adjoint_Z_loss_weight": 1.0,
            "solver": "online_source_stage",
            "lambda_price": 50.0,
            "lambda_volatility": 100.0,
            "eta": 2.0,
            "training_steps": 400,
        },
        "scope": {
            "indices": list(INDICES),
            "seeds": list(SEEDS),
            "rty_excluded_from_main_panel": True,
        },
        "scenario_bank": {
            "paths_per_seed": 8192,
            "bank_seed_rule": "70000 + model seed",
            "fresh_bank_after_checkpoint_freeze": True,
        },
        "evaluation_protocols": [
            "primary_2026_holdout",
            "us_iran_2026_war",
        ],
        "checkpoints": checkpoints,
    }


def _freeze_protocol() -> Path:
    path = RUN_ROOT / "LOCKED_PROTOCOL.json"
    payload = _frozen_protocol()
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous.pop("created_utc", None)
        candidate = dict(payload)
        candidate.pop("created_utc", None)
        if previous != candidate:
            raise RuntimeError(f"existing frozen protocol differs: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _run(command: list[str], *, dry_run: bool) -> None:
    print(f"[{_timestamp()}] RUN {' '.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _link(*, target: Path, link: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[{_timestamp()}] LINK {link} -> {target}", flush=True)
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"refusing to replace {link}")
    link.symlink_to(os.path.relpath(target, start=link.parent))


def _run_task(index: str, seed: int, *, device: str, dry_run: bool) -> None:
    candidate = _candidate(index, seed)
    checkpoint = candidate / "source_model_checkpoint.pt"
    candidate_manifest = candidate / "candidate_manifest.json"
    if not checkpoint.is_file() or not candidate_manifest.is_file():
        raise FileNotFoundError(candidate)

    resample_root = RUN_ROOT / "banks" / index / f"seed_{seed}"
    bank = resample_root / "validation_bank.npy"
    metrics = resample_root / "metrics.json"
    if not bank.is_file() or not metrics.is_file():
        _run(
            [
                str(PYTHON),
                str(REPO_ROOT / "experiments/scripts/evaluate_real_data_candidate_bank.py"),
                "--candidate-run-dir", str(candidate),
                "--checkpoint", str(checkpoint),
                "--run-dir", str(resample_root),
                "--device", device,
                "--bank-size", "8192",
                "--sample-batch-size", "8192",
                "--bank-seed", str(70_000 + seed),
            ],
            dry_run=dry_run,
        )
    else:
        print(f"[{_timestamp()}] REUSE {bank}", flush=True)

    generated_link = BANK_ROOT / "primary_2026_holdout" / index / f"seed_{seed}.npy"
    _link(target=bank, link=generated_link, dry_run=dry_run)

    output = EVALUATION_ROOT / MODEL_NAME / index / f"seed_{seed}"
    expected = [
        output / protocol / "metrics.json"
        for protocol in ("primary_2026_holdout", "us_iran_2026_war")
    ]
    if not all(path.is_file() for path in expected):
        _run(
            [
                str(PYTHON),
                str(REPO_ROOT / "experiments/scripts/run_real_data_evaluation.py"),
                "--generated-root", str(BANK_ROOT),
                "--output-root", str(EVALUATION_ROOT),
                "--model-name", MODEL_NAME,
                "--index", index,
                "--seed", str(seed),
                "--protocol", "primary_2026_holdout",
                "--protocol", "us_iran_2026_war",
            ],
            dry_run=dry_run,
        )
    else:
        print(f"[{_timestamp()}] REUSE {output}", flush=True)

    if not dry_run:
        provenance = {
            "method": MODEL_NAME,
            "index": index,
            "seed": seed,
            "selection_scope": "training_and_validation_only",
            "protected_data_used_for_selection": False,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "candidate_manifest": str(candidate_manifest.resolve()),
            "candidate_manifest_sha256": _sha256(candidate_manifest),
            "bank": str(bank.resolve()),
            "bank_sha256": _sha256(bank),
            "bank_seed": 70_000 + seed,
            "bank_size": 8192,
            "test_and_stress_role": "final_evaluation_only",
        }
        path = resample_root / "FINAL_EVALUATION_PROVENANCE.json"
        path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=int, choices=tuple(LANES))
    parser.add_argument("--device")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = _freeze_protocol()
    print(f"[{_timestamp()}] FROZEN {protocol}", flush=True)
    if bool(args.freeze_only):
        return
    if args.lane is None or args.device is None:
        parser.error("--lane and --device are required unless --freeze-only is used")
    for index, seed in LANES[int(args.lane)]:
        _run_task(index, seed, device=str(args.device), dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
