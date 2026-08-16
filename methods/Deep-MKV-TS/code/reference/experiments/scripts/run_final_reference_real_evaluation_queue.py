#!/usr/bin/env python3
"""Generate and score reference banks matched to final Deep MKV checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

from run_final_selected_real_evaluation_queue import (
    BANK_ROOT as DEEP_BANK_ROOT,
    CANDIDATES,
    LANES,
    REPO_ROOT,
    RUN_ROOT,
    PYTHON,
)


BANK_ROOT = RUN_ROOT / "reference_generated"
EVALUATION_ROOT = RUN_ROOT / "reference_protocol"
MODEL_NAME = "Fitted-reference-final"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, dry_run: bool) -> None:
    print(f"[{_timestamp()}] RUN {' '.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _run_task(index: str, seed: int, *, device: str, dry_run: bool) -> None:
    candidate = CANDIDATES[(index, seed)]
    manifest = candidate / "candidate_manifest.json"
    checkpoint = candidate / "source_model_checkpoint.pt"
    root = RUN_ROOT / "reference_banks" / index / f"seed_{seed}"
    bank = root / "validation_bank.npy"
    metrics = root / "metrics.json"
    if not bank.is_file() or not metrics.is_file():
        _run(
            [
                str(PYTHON),
                str(REPO_ROOT / "experiments/scripts/evaluate_real_data_candidate_bank.py"),
                "--candidate-run-dir", str(candidate),
                "--checkpoint", str(checkpoint),
                "--reference-only",
                "--run-dir", str(root),
                "--device", device,
                "--bank-size", "8192",
                "--sample-batch-size", "8192",
                "--bank-seed", str(80_000 + seed),
            ],
            dry_run=dry_run,
        )
    else:
        print(f"[{_timestamp()}] REUSE {bank}", flush=True)

    link = BANK_ROOT / "primary_2026_holdout" / index / f"seed_{seed}.npy"
    if dry_run:
        print(f"[{_timestamp()}] LINK {link} -> {bank}", flush=True)
    elif not link.exists() and not link.is_symlink():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(bank, start=link.parent))
    elif not (link.is_symlink() and link.resolve() == bank.resolve()):
        raise FileExistsError(f"refusing to replace {link}")

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
        payload = {
            "method": MODEL_NAME,
            "index": index,
            "seed": seed,
            "source_candidate_manifest": str(manifest.resolve()),
            "source_candidate_manifest_sha256": _sha256(manifest),
            "reference_construction": "same fitted reference with all learned adjoint parameters set to zero",
            "bank": str(bank.resolve()),
            "bank_sha256": _sha256(bank),
            "bank_seed": 80_000 + seed,
            "bank_size": 8192,
            "selection_scope": "training_and_validation_only",
            "test_and_crisis_role": "final_evaluation_only",
            "matched_deep_bank_root": str(DEEP_BANK_ROOT),
        }
        (root / "FINAL_EVALUATION_PROVENANCE.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=int, choices=tuple(LANES), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for index, seed in LANES[args.lane]:
        _run_task(index, seed, device=args.device, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
