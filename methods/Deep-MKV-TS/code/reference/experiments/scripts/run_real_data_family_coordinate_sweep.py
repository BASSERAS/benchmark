#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = REPO_ROOT / "experiments" / "scripts" / "run_real_data_tuning_candidate.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially sweep lambda_X, lambda_V, and eta one at a time "
            "under fixed train-only family adjoint normalization."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs" / "real_data_family_coordinate_sweep",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "runs" / "real_data_scale_corrected_v1" / "source",
    )
    parser.add_argument("--devices", nargs="+", default=("cuda:0", "cuda:3"))
    parser.add_argument("--screen-steps", type=int, default=400)
    parser.add_argument("--confirmation-steps", type=int, default=700)
    parser.add_argument("--screen-bank-size", type=int, default=1024)
    parser.add_argument("--confirmation-bank-size", type=int, default=4096)
    parser.add_argument(
        "--extended-grid",
        action="store_true",
        help=(
            "Use post-normalization grids lambda_X={2,4,8,16} and "
            "lambda_V={1,2,4,8,16}; otherwise use the conservative pilot grids."
        ),
    )
    parser.add_argument(
        "--high-range-pass",
        action="store_true",
        help=(
            "Run the second coordinate pass lambda_X={16,32,64} at lambda_V=16, "
            "then lambda_V={16,32,64}, retaining eta=2."
        ),
    )
    return parser.parse_args()


def _identifier(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "candidate_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_command(
    *,
    run_dir: Path,
    source_checkpoint: Path,
    seed: int,
    device: str,
    steps: int,
    bank_size: int,
    lambda_x: float,
    lambda_v: float,
    eta: float,
) -> list[str]:
    return [
        sys.executable,
        str(CANDIDATE),
        "--phase",
        "joint",
        "--objective-protocol",
        "real_scale_corrected_v3_family_adjoint_rms",
        "--run-dir",
        str(run_dir),
        "--source-checkpoint",
        str(source_checkpoint),
        "--seed",
        str(int(seed)),
        "--device",
        str(device),
        "--steps",
        str(int(steps)),
        "--lr",
        "2.5e-4",
        "--ridge-lambda",
        "1e-2",
        "--lambda-scale",
        "1",
        "--lambda-x",
        str(float(lambda_x)),
        "--lambda-v",
        str(float(lambda_v)),
        "--eta",
        str(float(eta)),
        "--joint-weight",
        "0.25",
        "--grad-clip-norm",
        "0",
        "--bank-size",
        str(int(bank_size)),
        "--sample-batch-size",
        str(int(bank_size)),
    ]


def _run_stage(
    *,
    jobs: Sequence[dict[str, Any]],
    devices: Sequence[str],
) -> list[dict[str, Any]]:
    pending = list(jobs)
    completed: list[dict[str, Any]] = []
    while pending:
        wave = pending[: len(devices)]
        del pending[: len(devices)]
        processes: list[tuple[dict[str, Any], subprocess.Popen[bytes], Any]] = []
        for job, device in zip(wave, devices):
            run_dir = Path(job["run_dir"])
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = run_dir / "candidate_manifest.json"
            if manifest_path.is_file():
                print(f"reusing completed {run_dir}", flush=True)
                completed.append(job | {"manifest": _read_manifest(run_dir)})
                continue
            log_handle = (run_dir / "run.log").open("wb")
            command = _candidate_command(
                run_dir=Path(job["run_dir"]),
                source_checkpoint=Path(job["source_checkpoint"]),
                seed=int(job["seed"]),
                device=str(device),
                steps=int(job["steps"]),
                bank_size=int(job["bank_size"]),
                lambda_x=float(job["lambda_x"]),
                lambda_v=float(job["lambda_v"]),
                eta=float(job["eta"]),
            )
            print(
                f"launch {job['identifier']} seed={job['seed']} device={device}",
                flush=True,
            )
            processes.append(
                (job, subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT), log_handle)
            )
        for job, process, log_handle in processes:
            return_code = process.wait()
            log_handle.close()
            if return_code != 0:
                raise RuntimeError(
                    f"candidate {job['identifier']} failed with exit code {return_code}; "
                    f"see {Path(job['run_dir']) / 'run.log'}"
                )
            completed.append(job | {"manifest": _read_manifest(Path(job["run_dir"]))})
            print(f"complete {job['identifier']} seed={job['seed']}", flush=True)
    return completed


def _stable(manifest: dict[str, Any]) -> bool:
    stability = manifest["stability"]
    return bool(
        stability["all_logged_quantities_finite"]
        and float(stability["clip_active_fraction"]) == 0.0
        and float(stability["gradient_nonfinite_fraction_max"]) == 0.0
        and float(stability["grad_norm_max"]) <= 1_000.0
        and float(stability["sigma_cap_fraction_max"]) == 0.0
        and float(stability["sigma_q99_max"]) < 0.10
    )


def _metric(manifest: dict[str, Any], group: str, name: str) -> float:
    return float(manifest[group][name])


PRICE_METRICS: tuple[tuple[str, str], ...] = (
    ("validation_law_metrics", "return_qq_rmse_normalized"),
    ("validation_law_metrics", "return_wasserstein_normalized"),
    ("validation_law_metrics", "terminal_return_wasserstein_normalized"),
    ("validation_shadowing_metrics", "cumulative_return_crps"),
    ("validation_shadowing_metrics", "increment_crps"),
)

VOLATILITY_METRICS: tuple[tuple[str, str], ...] = (
    ("validation_shadowing_metrics", "realized_volatility_crps"),
    ("validation_law_metrics", "realized_volatility_wasserstein_normalized"),
    ("validation_law_metrics", "abs_return_acf_error_rms"),
    ("validation_law_metrics", "squared_return_acf_error_rms"),
)


def _normalized_scores(
    rows: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[tuple[str, str]],
    include_coverage_gap: bool,
) -> dict[str, float]:
    eligible = [row for row in rows if _stable(row["manifest"])]
    if not eligible:
        raise RuntimeError("no coordinate candidate passes the uncapped stability gate")
    denominators: dict[tuple[str, str], float] = {}
    for metric in metrics:
        values = [_metric(row["manifest"], *metric) for row in eligible]
        denominators[metric] = max(statistics.median(values), 1e-12)
    coverage_gaps = [
        abs(
            _metric(
                row["manifest"],
                "validation_shadowing_metrics",
                "realized_volatility_coverage_90",
            )
            - 0.90
        )
        for row in eligible
    ]
    coverage_denominator = max(statistics.median(coverage_gaps), 1e-12)
    result: dict[str, float] = {}
    for row in eligible:
        score_terms = [
            _metric(row["manifest"], *metric) / denominators[metric]
            for metric in metrics
        ]
        if include_coverage_gap:
            gap = abs(
                _metric(
                    row["manifest"],
                    "validation_shadowing_metrics",
                    "realized_volatility_coverage_90",
                )
                - 0.90
            )
            score_terms.append(gap / coverage_denominator)
        result[str(row["identifier"])] = sum(score_terms) / len(score_terms)
    return result


def _price_guard(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    return all(
        _metric(candidate, *metric) <= 1.25 * _metric(reference, *metric)
        for metric in PRICE_METRICS
    )


def _select(
    rows: Sequence[dict[str, Any]],
    *,
    kind: str,
    price_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = PRICE_METRICS if kind == "price" else VOLATILITY_METRICS
    scores = _normalized_scores(
        rows,
        metrics=metrics,
        include_coverage_gap=kind == "volatility",
    )
    eligible = [
        row
        for row in rows
        if str(row["identifier"]) in scores
        and (
            price_reference is None
            or _price_guard(row["manifest"], price_reference)
        )
    ]
    if not eligible:
        raise RuntimeError("no stable candidate passes the frozen price-law guard")
    selected = min(eligible, key=lambda row: scores[str(row["identifier"])])
    for row in rows:
        row["stable"] = _stable(row["manifest"])
        row["selection_score"] = scores.get(str(row["identifier"]), math.inf)
        row["passes_price_guard"] = (
            True if price_reference is None else _price_guard(row["manifest"], price_reference)
        )
    return selected


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    manifest = row["manifest"]
    shadow = manifest["validation_shadowing_metrics"]
    law = manifest["validation_law_metrics"]
    return {
        "identifier": row["identifier"],
        "lambda_x": float(row["lambda_x"]),
        "lambda_v": float(row["lambda_v"]),
        "eta": float(row["eta"]),
        "stable": bool(row.get("stable", _stable(manifest))),
        "passes_price_guard": bool(row.get("passes_price_guard", True)),
        "selection_score": float(row.get("selection_score", math.nan)),
        "return_qq_rmse_normalized": float(law["return_qq_rmse_normalized"]),
        "return_wasserstein_normalized": float(law["return_wasserstein_normalized"]),
        "abs_return_acf_error_rms": float(law["abs_return_acf_error_rms"]),
        "squared_return_acf_error_rms": float(law["squared_return_acf_error_rms"]),
        "rv_wasserstein_normalized": float(law["realized_volatility_wasserstein_normalized"]),
        "rv_std_ratio": float(law["realized_volatility_std_ratio"]),
        "rv_crps": float(shadow["realized_volatility_crps"]),
        "rv_coverage_90": float(shadow["realized_volatility_coverage_90"]),
        "rv_width_90": float(shadow["realized_volatility_band_width_90"]),
        "cumulative_return_crps": float(shadow["cumulative_return_crps"]),
        "increment_crps": float(shadow["increment_crps"]),
        "grad_norm_max": float(manifest["stability"]["grad_norm_max"]),
        "sigma_mean_max": float(manifest["stability"]["sigma_mean_max"]),
        "run_dir": str(Path(row["run_dir"]).resolve()),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _screen_jobs(
    *,
    root: Path,
    stage: str,
    values: Sequence[float],
    coordinate: str,
    lambda_x: float,
    lambda_v: float,
    eta: float,
    source_checkpoint: Path,
    steps: int,
    bank_size: int,
) -> list[dict[str, Any]]:
    jobs = []
    for value in values:
        configuration = {
            "lambda_x": float(lambda_x),
            "lambda_v": float(lambda_v),
            "eta": float(eta),
        }
        configuration[coordinate] = float(value)
        identifier = f"{coordinate}_{_identifier(value)}"
        jobs.append(
            {
                "identifier": identifier,
                "run_dir": root / stage / identifier / "seed_1",
                "source_checkpoint": source_checkpoint,
                "seed": 1,
                "steps": int(steps),
                "bank_size": int(bank_size),
                **configuration,
            }
        )
    return jobs


def main() -> None:
    args = _parse_args()
    if bool(args.extended_grid) and bool(args.high_range_pass):
        raise ValueError("--extended-grid and --high-range-pass are mutually exclusive")
    if not args.devices:
        raise ValueError("at least one device is required")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_seed1 = args.source_root / "seed_1" / "source_model_checkpoint.pt"
    if not source_seed1.is_file():
        raise FileNotFoundError(source_seed1)

    if args.high_range_pass:
        lambda_x_values = (16.0, 32.0, 64.0)
        lambda_v_values = (32.0, 64.0)
        baseline_lambda_v = 16.0
    elif args.extended_grid:
        lambda_x_values = (2.0, 4.0, 8.0, 16.0)
        lambda_v_values = (2.0, 4.0, 8.0, 16.0)
        baseline_lambda_v = 1.0
    else:
        lambda_x_values = (0.5, 1.0, 2.0)
        lambda_v_values = (0.5, 2.0, 4.0)
        baseline_lambda_v = 1.0
    lambda_x_rows = _run_stage(
        jobs=_screen_jobs(
            root=root,
            stage="01_lambda_x",
            values=lambda_x_values,
            coordinate="lambda_x",
            lambda_x=1.0,
            lambda_v=baseline_lambda_v,
            eta=2.0,
            source_checkpoint=source_seed1,
            steps=int(args.screen_steps),
            bank_size=int(args.screen_bank_size),
        ),
        devices=args.devices,
    )
    selected_x = _select(lambda_x_rows, kind="price")
    price_reference = selected_x["manifest"]
    _write_json(
        root / "01_lambda_x" / "summary.json",
        {
            "coordinate": "lambda_x",
            "held_fixed": {"lambda_v": baseline_lambda_v, "eta": 2.0},
            "selected_identifier": selected_x["identifier"],
            "candidates": [_summary_row(row) for row in lambda_x_rows],
        },
    )

    selected_lambda_x = float(selected_x["lambda_x"])
    lambda_v_rows = _run_stage(
        jobs=_screen_jobs(
            root=root,
            stage="02_lambda_v",
            values=lambda_v_values,
            coordinate="lambda_v",
            lambda_x=selected_lambda_x,
            lambda_v=1.0,
            eta=2.0,
            source_checkpoint=source_seed1,
            steps=int(args.screen_steps),
            bank_size=int(args.screen_bank_size),
        ),
        devices=args.devices,
    )
    lambda_v_rows.append(
        {
            **selected_x,
            "identifier": f"lambda_v_{_identifier(baseline_lambda_v)}",
            "lambda_v": baseline_lambda_v,
        }
    )
    selected_v = _select(
        lambda_v_rows,
        kind="volatility",
        price_reference=price_reference,
    )
    _write_json(
        root / "02_lambda_v" / "summary.json",
        {
            "coordinate": "lambda_v",
            "held_fixed": {"lambda_x": selected_lambda_x, "eta": 2.0},
            "selected_identifier": selected_v["identifier"],
            "candidates": [_summary_row(row) for row in lambda_v_rows],
        },
    )

    selected_lambda_v = float(selected_v["lambda_v"])
    if args.high_range_pass:
        selected_eta_value = 2.0
        _write_json(
            root / "03_eta" / "summary.json",
            {
                "coordinate": "eta",
                "selection": "retained from preceding coordinate pass",
                "selected_identifier": "eta_2",
                "selected_eta": selected_eta_value,
            },
        )
    else:
        eta_rows = _run_stage(
            jobs=_screen_jobs(
                root=root,
                stage="03_eta",
                values=(1.0, 4.0),
                coordinate="eta",
                lambda_x=selected_lambda_x,
                lambda_v=selected_lambda_v,
                eta=2.0,
                source_checkpoint=source_seed1,
                steps=int(args.screen_steps),
                bank_size=int(args.screen_bank_size),
            ),
            devices=args.devices,
        )
        eta_rows.append(
            {
                **selected_v,
                "identifier": "eta_2",
                "eta": 2.0,
            }
        )
        selected_eta = _select(
            eta_rows,
            kind="volatility",
            price_reference=price_reference,
        )
        _write_json(
            root / "03_eta" / "summary.json",
            {
                "coordinate": "eta",
                "held_fixed": {
                    "lambda_x": selected_lambda_x,
                    "lambda_v": selected_lambda_v,
                },
                "selected_identifier": selected_eta["identifier"],
                "candidates": [_summary_row(row) for row in eta_rows],
            },
        )
        selected_eta_value = float(selected_eta["eta"])
    confirmation_jobs = []
    for seed in (1, 2):
        source = args.source_root / f"seed_{seed}" / "source_model_checkpoint.pt"
        if not source.is_file():
            raise FileNotFoundError(source)
        confirmation_jobs.append(
            {
                "identifier": f"selected_seed_{seed}",
                "run_dir": root / "04_confirmation" / f"seed_{seed}",
                "source_checkpoint": source,
                "seed": seed,
                "steps": int(args.confirmation_steps),
                "bank_size": int(args.confirmation_bank_size),
                "lambda_x": selected_lambda_x,
                "lambda_v": selected_lambda_v,
                "eta": selected_eta_value,
            }
        )
    confirmation = _run_stage(jobs=confirmation_jobs, devices=args.devices)
    _write_json(
        root / "summary.json",
        {
            "protocol": "sequential_coordinate_sweep; one coefficient changes per stage",
            "extended_grid": bool(args.extended_grid),
            "high_range_pass": bool(args.high_range_pass),
            "selected": {
                "lambda_x": selected_lambda_x,
                "lambda_v": selected_lambda_v,
                "eta": selected_eta_value,
            },
            "screen_steps": int(args.screen_steps),
            "confirmation_steps": int(args.confirmation_steps),
            "confirmation": [_summary_row(row) for row in confirmation],
        },
    )
    print(
        "coordinate sweep complete "
        f"lambda_x={selected_lambda_x:g} lambda_v={selected_lambda_v:g} "
        f"eta={selected_eta_value:g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
