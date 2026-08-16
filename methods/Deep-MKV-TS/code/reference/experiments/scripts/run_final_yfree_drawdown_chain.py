#!/usr/bin/env python3
"""Run validation calibration and frozen ES drawdown-risk test for final Deep MKV."""

from __future__ import annotations

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/samer/venvs/mfc/bin/python")
FINAL_ROOT = REPO_ROOT / "runs" / "final_yfree_fitted_drift_evaluation_20260808"
OLD_FINAL_ROOT = REPO_ROOT / "runs" / "final_selected_real_evaluation_20260806"
VALIDATION_RUN = REPO_ROOT / "runs" / "es_drawdown_risk_yfree_validation_20260808"
TEST_RUN = REPO_ROOT / "runs" / "es_drawdown_risk_yfree_test_20260808"
TEST_PROTOCOL = FINAL_ROOT / "DRAWDOWN_LOCKED_TEST_PROTOCOL.json"
SEEDS = (0, 1, 3, 4)


def _run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    validation_report = VALIDATION_RUN / "report.json"
    if not validation_report.is_file():
        command = [
            str(PYTHON),
            str(REPO_ROOT / "experiments/scripts/run_drawdown_risk_budgeting.py"),
            "--run-dir", str(VALIDATION_RUN),
            "--primary-method", "deep_seed0",
        ]
        for seed in SEEDS:
            command.extend(
                [
                    "--bank",
                    (
                        f"deep_seed{seed}="
                        f"{FINAL_ROOT / 'banks' / 'ES' / f'seed_{seed}' / 'validation_bank.npy'}"
                    ),
                ]
            )
        for seed in SEEDS:
            command.extend(
                [
                    "--bank",
                    (
                        f"reference_seed{seed}="
                        f"{OLD_FINAL_ROOT / 'reference_banks' / 'ES' / f'seed_{seed}' / 'validation_bank.npy'}"
                    ),
                ]
            )
        command.extend(
            [
                "--bank",
                (
                    "sbts="
                    f"{REPO_ROOT / 'runs/real_data_protocol/SBTS/ES/seed_0/primary_2026_holdout/generated_bank.npy'}"
                ),
                "--bank",
                (
                    "moving_block="
                    f"{REPO_ROOT / 'runs/real_data_protocol/moving_block_bootstrap/ES/seed_0/primary_2026_holdout/generated_bank.npy'}"
                ),
                "--bank",
                (
                    "whole_session="
                    f"{REPO_ROOT / 'runs/real_data_protocol/whole_session_bootstrap/ES/seed_0/primary_2026_holdout/generated_bank.npy'}"
                ),
                "--include-historical-neighbor",
            ]
        )
        _run(command)
    else:
        print(f"REUSE {validation_report}", flush=True)

    if not TEST_PROTOCOL.is_file():
        _run(
            [
                str(PYTHON),
                str(REPO_ROOT / "experiments/scripts/freeze_final_drawdown_risk_protocol.py"),
                "--validation-run", str(VALIDATION_RUN),
                "--bank-root", str(FINAL_ROOT),
                "--reference-bank-root", str(OLD_FINAL_ROOT),
                "--output", str(TEST_PROTOCOL),
                "--experiment-name",
                "drawdown_constrained_position_sizing_yfree_final_test_20260808",
            ]
        )
    else:
        print(f"REUSE {TEST_PROTOCOL}", flush=True)

    test_report = TEST_RUN / "report.json"
    if not test_report.is_file():
        _run(
            [
                str(PYTHON),
                str(REPO_ROOT / "experiments/scripts/run_drawdown_risk_budgeting_external_test.py"),
                "--run-dir", str(TEST_RUN),
                "--protocol-manifest", str(TEST_PROTOCOL),
            ]
        )
    else:
        print(f"REUSE {test_report}", flush=True)


if __name__ == "__main__":
    main()
