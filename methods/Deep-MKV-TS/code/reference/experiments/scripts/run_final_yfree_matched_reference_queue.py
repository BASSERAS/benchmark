#!/usr/bin/env python3
"""Generate matched fitted-drift reference banks for the final futures model.

The reference reuses every frozen component of the final Y-free Deep MKV
candidate, including its fitted path-dependent Gaussian drift and volatility,
while setting all learned correction-network parameters to zero.  No fitting or
selection is performed here.
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
RUN_ROOT = (
    REPO_ROOT / "runs" / "final_yfree_matched_reference_evaluation_20260810"
)
BANK_ROOT = RUN_ROOT / "generated"
EVALUATION_ROOT = RUN_ROOT / "protocol"
MODEL_NAME = "Fitted-reference-yfree-fitted-drift"
INDICES = ("ES", "NQ", "YM")
SEEDS = (0, 1, 3, 4)
LANES: dict[int, tuple[tuple[str, int], ...]] = {
    0: (("ES", 0), ("NQ", 0), ("YM", 0)),
    1: (("ES", 1), ("NQ", 1), ("YM", 1)),
    2: (("ES", 3), ("NQ", 3), ("YM", 3)),
    3: (("ES", 4), ("NQ", 4), ("YM", 4)),
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate(index: str, seed: int) -> Path:
    return SOURCE_ROOT / index / f"seed_{seed}"


def _audit_candidate(index: str, seed: int) -> dict[str, object]:
    candidate = _candidate(index, seed)
    manifest_path = candidate / "candidate_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "index": str(index),
        "seed": int(seed),
        "reference_kind": "calibrated_causal",
        "drift_control_mode": "volatility_only",
        "use_reference_drift": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"{manifest_path}: expected {key}={expected!r}, "
                f"found {manifest.get(key)!r}"
            )
    training = manifest.get("training", {})
    if float(training.get("adjoint_weight", -1.0)) != 0.0:
        raise RuntimeError(f"{manifest_path}: final candidate is not Y-free")
    return manifest


def _freeze_protocol() -> Path:
    candidates: dict[str, object] = {}
    for index in INDICES:
        candidates[index] = {}
        for seed in SEEDS:
            _audit_candidate(index, seed)
            manifest = _candidate(index, seed) / "candidate_manifest.json"
            candidates[index][str(seed)] = {
                "candidate_manifest": str(manifest.resolve()),
                "candidate_manifest_sha256": _sha256(manifest),
            }
    payload = {
        "experiment": "final_yfree_matched_fitted_drift_reference_20260810",
        "created_utc": _timestamp(),
        "selection_scope": "none_evaluation_only",
        "test_or_stress_used_for_selection": False,
        "reference": {
            "kind": "path_dependent_gaussian_ridge",
            "forward_drift": "same_fitted_path_dependent_drift_as_final_deep_mkv",
            "forward_volatility": "same_fitted_path_dependent_volatility_as_final_deep_mkv",
            "learned_correction": "identically_zero",
        },
        "indices": list(INDICES),
        "seeds": list(SEEDS),
        "bank_seed_rule": "70000 + seed, matched to final Deep MKV bank",
        "bank_size": 8192,
        "protocols": ["primary_2026_holdout", "us_iran_2026_war"],
        "candidates": candidates,
    }
    path = RUN_ROOT / "LOCKED_PROTOCOL.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous.pop("created_utc", None)
        candidate_payload = dict(payload)
        candidate_payload.pop("created_utc", None)
        if previous != candidate_payload:
            raise RuntimeError(f"existing protocol differs: {path}")
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
    manifest_data = _audit_candidate(index, seed)
    candidate = _candidate(index, seed)
    manifest = candidate / "candidate_manifest.json"
    root = RUN_ROOT / "reference_banks" / index / f"seed_{seed}"
    bank = root / "validation_bank.npy"
    metrics = root / "metrics.json"
    if not bank.is_file() or not metrics.is_file():
        _run(
            [
                str(PYTHON),
                str(REPO_ROOT / "experiments/scripts/evaluate_real_data_candidate_bank.py"),
                "--candidate-run-dir", str(candidate),
                "--reference-only",
                "--run-dir", str(root),
                "--device", device,
                "--bank-size", "8192",
                "--sample-batch-size", "8192",
                "--bank-seed", str(70_000 + seed),
            ],
            dry_run=dry_run,
        )
    else:
        print(f"[{_timestamp()}] REUSE {bank}", flush=True)

    link = BANK_ROOT / "primary_2026_holdout" / index / f"seed_{seed}.npy"
    _link(target=bank, link=link, dry_run=dry_run)

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
            "candidate_manifest": str(manifest.resolve()),
            "candidate_manifest_sha256": _sha256(manifest),
            "reference_kind": manifest_data["reference_kind"],
            "reference_feature_basis": manifest_data["reference_feature_basis"],
            "use_reference_drift": manifest_data["use_reference_drift"],
            "drift_control_mode": manifest_data["drift_control_mode"],
            "reference_construction": (
                "final fitted-drift Gaussian reference with correction network zeroed"
            ),
            "bank": str(bank.resolve()),
            "bank_sha256": _sha256(bank),
            "bank_seed": 70_000 + seed,
            "bank_size": 8192,
            "test_and_stress_role": "evaluation_only",
        }
        (root / "FINAL_EVALUATION_PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=int, choices=tuple(LANES))
    parser.add_argument("--device")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = _freeze_protocol()
    print(f"[{_timestamp()}] FROZEN {protocol}", flush=True)
    if args.freeze_only:
        return
    if args.lane is None or args.device is None:
        parser.error("--lane and --device are required unless --freeze-only is used")
    for index, seed in LANES[int(args.lane)]:
        _run_task(index, seed, device=str(args.device), dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
