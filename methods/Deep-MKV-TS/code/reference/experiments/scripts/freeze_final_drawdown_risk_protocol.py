#!/usr/bin/env python3
"""Freeze the final ES drawdown-risk protocol after validation calibration."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = (
    REPO_ROOT
    / "experiments"
    / "protocols"
    / "drawdown_risk_budgeting_external_test_v1.json"
)
DEFAULT_VALIDATION = (
    REPO_ROOT / "runs" / "es_drawdown_risk_budgeting_final_validation_v2"
)
DEFAULT_BANK_ROOT = (
    REPO_ROOT / "runs" / "final_selected_real_evaluation_20260806"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments"
    / "protocols"
    / "drawdown_risk_budgeting_final_test_v2.json"
)
SEEDS = (0, 1, 3, 4)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--validation-run", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK_ROOT)
    parser.add_argument(
        "--reference-bank-root",
        type=Path,
        help=(
            "Optional root containing reference_banks/ES/seed_<seed>.  "
            "Defaults to --bank-root."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--experiment-name",
        default="drawdown_constrained_position_sizing_final_test_v2",
    )
    args = parser.parse_args()

    base = json.loads(args.base_protocol.read_text(encoding="utf-8"))
    validation_report = args.validation_run / "report.json"
    if not validation_report.is_file():
        raise FileNotFoundError(validation_report)

    method_paths: dict[str, Path] = {
        "sbts": Path(base["methods"]["sbts"]["bank_path"]),
        "moving_block": Path(base["methods"]["moving_block"]["bank_path"]),
        "whole_session": Path(base["methods"]["whole_session"]["bank_path"]),
        "historical_neighbor": Path(
            base["methods"]["historical_neighbor"]["bank_path"]
        ),
    }
    for seed in SEEDS:
        method_paths[f"deep_seed{seed}"] = (
            args.bank_root / "banks" / "ES" / f"seed_{seed}" / "validation_bank.npy"
        )
        method_paths[f"reference_seed{seed}"] = (
            (args.reference_bank_root or args.bank_root)
            / "reference_banks"
            / "ES"
            / f"seed_{seed}"
            / "validation_bank.npy"
        )

    methods = {}
    validation_paths = {}
    for name, bank in method_paths.items():
        if not bank.is_file():
            raise FileNotFoundError(bank)
        metrics = args.validation_run / name / "path_metrics.npz"
        if not metrics.is_file():
            raise FileNotFoundError(metrics)
        methods[name] = {
            "bank_kind": (
                "training_sessions"
                if name == "historical_neighbor"
                else "generated_scenario_bank"
            ),
            "bank_path": str(bank.resolve()),
            "bank_sha256": _sha256(bank),
        }
        validation_paths[name] = {
            "path": str(metrics.resolve()),
            "sha256": _sha256(metrics),
        }

    payload = dict(base)
    payload.update(
        {
            "experiment": str(args.experiment_name),
            "external_methods_only": False,
            "ours_authorized": True,
            "primary_method": "deep_seed0",
            "methods": methods,
            "validation_path_metrics": validation_paths,
        }
    )
    payload["calibration"] = {
        **base["calibration"],
        "validation_report": str(validation_report.resolve()),
        "validation_report_sha256": _sha256(validation_report),
        "selection_note": (
            "All safety factors are fit separately on the locked 2025 "
            "validation sessions and frozen before protected test scoring."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
