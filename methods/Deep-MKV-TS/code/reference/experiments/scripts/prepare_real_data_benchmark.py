#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
from datetime import date, time
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from path_dt_experiments.real_data_benchmark import (  # noqa: E402
    INDEX_NAMES,
    RealDataPreprocessingConfig,
    SessionAuditResult,
    audit_session_file,
    common_valid_dates,
    sha256_file,
)
from path_dt_experiments.real_data_protocol import (  # noqa: E402
    frozen_protocol_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the four ICAIF index archives and construct the frozen, "
            "model-neutral real-data benchmark arrays."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=REPO_ROOT / "data" / "ICAIF_3Y" / "raw",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=REPO_ROOT / "data" / "ICAIF_3Y" / "archives" / "ICAIF",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "ICAIF_3Y" / "canonical_rth_128",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--path-points", type=int, default=128)
    parser.add_argument(
        "--maximum-staleness-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--maximum-abs-grid-log-return",
        type=float,
        default=0.20,
    )
    parser.add_argument("--train-end", type=date.fromisoformat, default=date(2025, 5, 31))
    parser.add_argument(
        "--validation-end",
        type=date.fromisoformat,
        default=date(2025, 12, 31),
    )
    return parser.parse_args()


def _audit_task(
    payload: tuple[Path, str, RealDataPreprocessingConfig],
) -> SessionAuditResult:
    path, index, config = payload
    return audit_session_file(path, expected_index=index, config=config)


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_audit_csv(path: Path, records: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in record.items()
                }
            )


def _robust_standardize(
    values: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    logged = np.log(np.maximum(values, np.finfo(np.float64).tiny))
    fit_values = logged[fit_mask]
    centre = float(np.median(fit_values))
    scale = float(1.4826 * np.median(np.abs(fit_values - centre)))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.std(fit_values))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return (logged - centre) / scale


def _stress_diagnostics(
    output_root: Path,
    dates: list[str],
    arrays: dict[str, np.ndarray],
    train_end: date,
) -> dict[str, object]:
    train_mask = np.asarray(
        [date.fromisoformat(day) <= train_end for day in dates],
        dtype=bool,
    )
    realized_variations: dict[str, np.ndarray] = {}
    standardized: list[np.ndarray] = []
    for index in INDEX_NAMES:
        log_paths = np.log(arrays[index].astype(np.float64) / 100.0)
        realized_variation = np.sqrt(np.sum(np.diff(log_paths, axis=1) ** 2, axis=1))
        realized_variations[index] = realized_variation
        standardized.append(_robust_standardize(realized_variation, train_mask))
    composite = np.median(np.stack(standardized, axis=1), axis=1)
    smooth = np.convolve(composite, np.ones(5) / 5.0, mode="same")
    candidate_threshold = float(np.quantile(smooth[train_mask], 0.90))
    candidate_positions = np.flatnonzero(smooth >= candidate_threshold)

    rows: list[dict[str, object]] = []
    for position, day in enumerate(dates):
        row: dict[str, object] = {
            "date": day,
            "composite_log_rv_robust_z": float(composite[position]),
            "five_session_smoothed_score": float(smooth[position]),
            "stress_candidate": bool(smooth[position] >= candidate_threshold),
        }
        for index in INDEX_NAMES:
            row[f"{index}_realized_variation"] = float(
                realized_variations[index][position]
            )
            row[f"{index}_log_rv_robust_z"] = float(
                standardized[INDEX_NAMES.index(index)][position]
            )
        rows.append(row)
    _write_audit_csv(output_root / "daily_stress_scores.csv", rows)

    clusters: list[list[int]] = []
    for position in candidate_positions.tolist():
        if not clusters or position - clusters[-1][-1] > 3:
            clusters.append([position])
        else:
            clusters[-1].append(position)
    window_rows: list[dict[str, object]] = []
    for number, positions in enumerate(clusters, start=1):
        first = positions[0]
        last = positions[-1]
        peak = max(positions, key=lambda position: smooth[position])
        window_rows.append(
            {
                "candidate_window": number,
                "start_date": dates[first],
                "end_date": dates[last],
                "candidate_sessions": len(positions),
                "peak_date": dates[peak],
                "peak_five_session_score": float(smooth[peak]),
            }
        )
    _write_audit_csv(output_root / "stress_candidate_windows.csv", window_rows)

    top_positions = np.argsort(smooth)[::-1][:20]
    top_dates = [
        {
            "rank": rank,
            "date": dates[int(position)],
            "five_session_smoothed_score": float(smooth[int(position)]),
            "composite_log_rv_robust_z": float(composite[int(position)]),
        }
        for rank, position in enumerate(top_positions, start=1)
    ]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(11.5, 4.2))
        x = np.arange(len(dates))
        axis.plot(x, smooth, color="#2457a6", linewidth=1.2)
        axis.axhline(
            candidate_threshold,
            color="#b94735",
            linestyle="--",
            linewidth=1.0,
            label="90th percentile candidate threshold",
        )
        tick_positions = np.linspace(0, len(dates) - 1, 9).astype(int)
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(
            [dates[position] for position in tick_positions],
            rotation=30,
            ha="right",
        )
        axis.set_ylabel("Five-session robust volatility score")
        axis.set_title("Data-derived candidate stress periods (not event labels)")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper left", frameon=False)
        figure.tight_layout()
        figure.savefig(output_root / "stress_candidate_timeline.pdf")
        figure.savefig(output_root / "stress_candidate_timeline.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass

    return {
        "definition": (
            "Median across indices of robust-standardized log daily realized "
            "variation, averaged over five sessions."
        ),
        "candidate_threshold_quantile": 0.90,
        "threshold_fit_period_end": train_end.isoformat(),
        "candidate_threshold": candidate_threshold,
        "candidate_window_max_gap_sessions": 3,
        "candidate_window_count": len(window_rows),
        "top_dates": top_dates,
        "warning": (
            "These are data-derived candidates for a preregistered stress "
            "analysis. Economic or crisis labels require external confirmation."
        ),
    }


def main() -> None:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    config = RealDataPreprocessingConfig(
        path_points=args.path_points,
        maximum_staleness_seconds=args.maximum_staleness_seconds,
        maximum_abs_log_return=args.maximum_abs_grid_log_return,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    paths_by_index: dict[str, list[Path]] = {}
    tasks: list[tuple[Path, str, RealDataPreprocessingConfig]] = []
    for index in INDEX_NAMES:
        paths = sorted((args.raw_root / index).glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no Parquet sessions found for {index}")
        paths_by_index[index] = paths
        tasks.extend((path, index, config) for path in paths)
        print(f"{index}: discovered {len(paths)} sessions", flush=True)

    print(
        f"Auditing {len(tasks)} sessions with {args.workers} workers...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            executor.map(
                _audit_task,
                tasks,
                chunksize=4,
            )
        )
    results_by_index: dict[str, list[SessionAuditResult]] = {
        index: [] for index in INDEX_NAMES
    }
    for result in results:
        results_by_index[str(result.record["index"])].append(result)
    for index in INDEX_NAMES:
        results_by_index[index].sort(key=lambda result: str(result.record["date"]))

    dates = common_valid_dates(results_by_index)
    common_set = set(dates)
    if not dates:
        raise RuntimeError("the four indices have no common valid sessions")
    records: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for index in INDEX_NAMES:
        by_date = {
            str(result.record["date"]): result
            for result in results_by_index[index]
        }
        index_paths = []
        for day in dates:
            path = by_date[day].normalized_prices
            if path is None:
                raise RuntimeError(f"missing canonical path for {index} {day}")
            index_paths.append(path)
        arrays[index] = np.stack(index_paths).astype(np.float32)
        for result in results_by_index[index]:
            record = result.record
            day = str(record["date"])
            individually_valid = bool(record["included_individually"])
            record["included_common_calendar"] = individually_valid and day in common_set
            record["common_calendar_exclusion"] = (
                "not_valid_for_another_index"
                if individually_valid and day not in common_set
                else ""
            )
            records.append(record)

    _write_audit_csv(output_root / "session_audit.csv", records)
    (output_root / "common_dates.txt").write_text(
        "\n".join(dates) + "\n",
        encoding="utf-8",
    )

    split_dates: dict[str, list[str]] = {
        split: [
            day
            for day in dates
            if (
                (split == "train" and date.fromisoformat(day) <= config.train_end)
                or (
                    split == "validation"
                    and config.train_end < date.fromisoformat(day) <= config.validation_end
                )
                or (
                    split == "test"
                    and date.fromisoformat(day) > config.validation_end
                )
            )
        ]
        for split in ("train", "validation", "test")
    }
    date_positions = {day: position for position, day in enumerate(dates)}
    output_files: list[Path] = []
    for index in INDEX_NAMES:
        index_root = output_root / index
        index_root.mkdir(exist_ok=True)
        all_prices_path = index_root / "prices_all.npy"
        all_dates_path = index_root / "dates_all.npy"
        np.save(all_prices_path, arrays[index], allow_pickle=False)
        np.save(all_dates_path, np.asarray(dates, dtype="U10"), allow_pickle=False)
        output_files.extend((all_prices_path, all_dates_path))
        for split, days in split_dates.items():
            positions = np.asarray([date_positions[day] for day in days], dtype=int)
            prices_path = index_root / f"prices_{split}.npy"
            dates_path = index_root / f"dates_{split}.npy"
            np.save(prices_path, arrays[index][positions], allow_pickle=False)
            np.save(dates_path, np.asarray(days, dtype="U10"), allow_pickle=False)
            output_files.extend((prices_path, dates_path))

    stress = _stress_diagnostics(output_root, dates, arrays, config.train_end)
    archive_hashes = {
        path.name: sha256_file(path)
        for path in sorted(args.archive_root.glob("*.zip"))
    }
    reason_counts: dict[str, int] = {}
    for record in records:
        for reason in record.get("exclusion_reasons", []):
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    per_index: dict[str, object] = {}
    for index in INDEX_NAMES:
        index_records = [record for record in records if record["index"] == index]
        per_index[index] = {
            "discovered_sessions": len(index_records),
            "individually_valid_sessions": sum(
                bool(record["included_individually"]) for record in index_records
            ),
            "common_sessions": len(dates),
            "first_source_date": min(str(record["date"]) for record in index_records),
            "last_source_date": max(str(record["date"]) for record in index_records),
        }
    manifest = {
        "benchmark_name": "ICAIF_3Y_four_index_RTH_128",
        "purpose": (
            "Frozen model-neutral preprocessing contract for subsequent "
            "real-data scenario-generator evaluation."
        ),
        "created_with_script": str(Path(__file__).resolve()),
        "configuration": asdict(config),
        "source": {
            "raw_root": args.raw_root.resolve(),
            "archive_root": args.archive_root.resolve(),
            "archive_sha256": archive_hashes,
            "price_column": "mid_px",
            "excluded_columns": ["micro_px", "bid_imb", "ask_imb", "qimb"],
        },
        "sampling": {
            "causal": True,
            "rule": "last valid mid-price at or before each target time",
            "normalization": "100 * mid_price(t) / mid_price(session_open)",
            "output_dtype": "float32",
        },
        "calendar": {
            "common_session_count": len(dates),
            "first_common_date": dates[0],
            "last_common_date": dates[-1],
            "split_counts": {
                split: len(days) for split, days in split_dates.items()
            },
            "split_date_ranges": {
                split: (
                    {"first": days[0], "last": days[-1]}
                    if days
                    else {"first": None, "last": None}
                )
                for split, days in split_dates.items()
            },
        },
        "audit": {
            "per_index": per_index,
            "exclusion_reason_counts_across_index_files": reason_counts,
        },
        "stress_candidates": stress,
        "canonical_file_sha256": {
            str(path.relative_to(output_root)): sha256_file(path)
            for path in output_files
        },
        "leakage_controls": [
            "Strictly chronological train/validation/test splits.",
            "No future observation is used for grid sampling.",
            "No model is trained or selected during this audit.",
            "Stress candidates use only realized variation and remain unlabeled.",
        ],
        "evaluation_protocol": {
            "status": "frozen before learned-model real-data evaluation",
            "manifest_file": "frozen_real_data_protocol.json",
            "human_readable_report": "real_data_experimental_protocol.pdf",
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    _write_json(
        output_root / "frozen_real_data_protocol.json",
        frozen_protocol_manifest(),
    )
    _write_json(output_root / "stress_summary.json", stress)
    print(
        "Complete: "
        f"{len(dates)} common sessions; "
        + ", ".join(
            f"{split}={len(days)}" for split, days in split_dates.items()
        ),
        flush=True,
    )
    print(f"Outputs: {output_root}", flush=True)


if __name__ == "__main__":
    main()
