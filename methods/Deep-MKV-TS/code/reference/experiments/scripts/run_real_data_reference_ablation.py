#!/usr/bin/env python3
"""Generate and evaluate one real-data reference/source/full ablation lane."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/samer/venvs/mfc/bin/python")
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "real_reference_ablation_4seed_20260802"
MAIN_ROOT = REPO_ROOT / "runs" / "real_dimensionless_4seed_20260801"
RTY_ROOT = REPO_ROOT / "runs" / "rty_dimensionless_4seed_20260801"
STAGES = ("reference", "source", "complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=("ES", "NQ", "RTY", "YM"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 3, 4), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def source_seed_root(index: str, seed: int) -> Path:
    if index == "RTY":
        return RTY_ROOT / f"seed_{seed}"
    return MAIN_ROOT / index / f"seed_{seed}"


def ensure_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        return
    link.symlink_to(os.path.relpath(target.resolve(), start=link.parent.resolve()))


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    source_root = source_seed_root(args.index, args.seed)
    source_dir = source_root / "source"
    source_checkpoint = source_dir / "outer_002_checkpoint.pt"
    complete_bank = source_root / "matched8192" / "validation_bank.npy"
    for required in (
        source_dir / "candidate_manifest.json",
        source_checkpoint,
        complete_bank,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    stage_banks: dict[str, Path] = {"complete": complete_bank}
    generator = REPO_ROOT / "experiments" / "scripts" / "evaluate_real_data_candidate_bank.py"
    for stage in ("reference", "source"):
        if stage not in args.stages:
            continue
        bank_root = run_root / "banks" / args.index / f"seed_{args.seed}" / stage
        bank = bank_root / "validation_bank.npy"
        if not bank.is_file():
            command = [
                str(PYTHON),
                str(generator),
                "--candidate-run-dir", str(source_dir),
                "--run-dir", str(bank_root),
                "--device", str(args.device),
                "--bank-size", "8192",
                "--sample-batch-size", "8192",
                "--bank-seed", str(70_000 + int(args.seed)),
            ]
            if stage == "reference":
                command.append("--reference-only")
            else:
                command.extend(["--checkpoint", str(source_checkpoint)])
            run(command)
        stage_banks[stage] = bank

    evaluator = REPO_ROOT / "experiments" / "scripts" / "run_real_data_evaluation.py"
    model_names = {
        "reference": "Deep-MKV-reference-only",
        "source": "Deep-MKV-source-stage",
        "complete": "Deep-MKV-complete",
    }
    for stage in args.stages:
        bank = stage_banks[stage]
        generated_root = run_root / "generated" / stage
        link = (
            generated_root
            / "primary_2026_holdout"
            / args.index
            / f"seed_{args.seed}.npy"
        )
        ensure_link(link, bank)
        summary = (
            run_root
            / "protocol"
            / model_names[stage]
            / args.index
            / f"seed_{args.seed}"
            / "summary.json"
        )
        if summary.is_file():
            continue
        run(
            [
                str(PYTHON),
                str(evaluator),
                "--generated-root", str(generated_root),
                "--output-root", str(run_root / "protocol"),
                "--model-name", model_names[stage],
                "--index", args.index,
                "--seed", str(args.seed),
                "--protocol", "primary_2026_holdout",
                "--protocol", "us_iran_2026_war",
            ]
        )

    print(
        f"complete reference ablation lane: index={args.index} seed={args.seed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
