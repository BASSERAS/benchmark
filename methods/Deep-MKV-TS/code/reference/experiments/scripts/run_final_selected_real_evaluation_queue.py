#!/usr/bin/env python3
"""Generate and evaluate frozen final Deep MKV real-data scenario banks.

The checkpoints in this queue were selected using training and validation data
only.  The script first draws an independent 8,192-path bank from each frozen
checkpoint and only then invokes the protected holdout/event evaluator.  It is
resumable and never trains or updates a model.
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
SOURCE_ROOT = REPO_ROOT / "runs" / "selected_deep_mkv_online_completion_20260805" / "real"
RUN_ROOT = REPO_ROOT / "runs" / "final_selected_real_evaluation_20260806"
BANK_ROOT = RUN_ROOT / "generated"
EVALUATION_ROOT = RUN_ROOT / "protocol"
MODEL_NAME = "Deep-MKV-final-online-volatility"
SEEDS = (0, 1, 3, 4)


CANDIDATES: dict[tuple[str, int], Path] = {
    ("ES", 0): SOURCE_ROOT / "ES_early_eta2_capq99",
    ("ES", 1): SOURCE_ROOT / "ES_4seed" / "seed_1",
    ("ES", 3): SOURCE_ROOT / "ES_4seed" / "seed_3",
    ("ES", 4): SOURCE_ROOT / "ES_4seed" / "seed_4",
    ("NQ", 0): SOURCE_ROOT / "transfer_NQ_seed0",
    ("NQ", 1): SOURCE_ROOT / "NQ_4seed" / "seed_1",
    ("NQ", 3): SOURCE_ROOT / "NQ_4seed" / "seed_3",
    ("NQ", 4): SOURCE_ROOT / "NQ_4seed" / "seed_4",
    ("RTY", 0): SOURCE_ROOT / "transfer_RTY_seed0",
    ("RTY", 1): SOURCE_ROOT / "RTY_4seed" / "seed_1",
    ("RTY", 3): SOURCE_ROOT / "RTY_4seed" / "seed_3",
    ("RTY", 4): SOURCE_ROOT / "RTY_4seed" / "seed_4",
    ("YM", 0): SOURCE_ROOT / "transfer_YM_seed0",
    ("YM", 1): SOURCE_ROOT / "YM_4seed" / "seed_1",
    ("YM", 3): SOURCE_ROOT / "YM_4seed" / "seed_3",
    ("YM", 4): SOURCE_ROOT / "YM_4seed" / "seed_4",
}


LANES: dict[int, tuple[tuple[str, int], ...]] = {
    0: (("ES", 0), ("NQ", 0), ("RTY", 0), ("YM", 0), ("YM", 4)),
    1: (("ES", 1), ("NQ", 1), ("RTY", 1), ("YM", 1), ("ES", 4), ("NQ", 4)),
    2: (("ES", 3), ("NQ", 3), ("RTY", 3), ("YM", 3), ("RTY", 4)),
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    candidate = CANDIDATES[(index, seed)]
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
            "test_and_crisis_role": "final_evaluation_only",
        }
        path = resample_root / "FINAL_EVALUATION_PROVENANCE.json"
        path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


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
