#!/usr/bin/env python3
"""Create the frozen ES/NQ/YM report and paired CRPS uncertainty audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "runs" / "final_selected_real_evaluation_20260806"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "final_large_cap_evaluation_20260806"
INDICES = ("ES", "NQ", "YM")
SEEDS = (0, 1, 3, 4)
CRPS_METRICS = (
    "cumulative_return_crps",
    "increment_crps",
    "realized_volatility_crps",
)
CRPS_LABELS = {
    "cumulative_return_crps": "Cumulative CRPS",
    "increment_crps": "Increment CRPS",
    "realized_volatility_crps": "RV CRPS",
}
LAW_METRICS = (
    "return_qq_rmse_normalized",
    "abs_return_acf_error_rms",
    "realized_volatility_wasserstein_normalized",
    "maximum_drawdown_wasserstein_normalized",
)
LAW_LABELS = {
    "return_qq_rmse_normalized": "Return QQ",
    "abs_return_acf_error_rms": "Abs-ACF",
    "realized_volatility_wasserstein_normalized": "RV W1",
    "maximum_drawdown_wasserstein_normalized": "Drawdown W1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--deep-root",
        type=Path,
        help="Optional explicit Deep MKV protocol root.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        help="Optional explicit fitted-reference protocol root.",
    )
    parser.add_argument(
        "--sbts-root",
        type=Path,
        help="Optional explicit SBTS protocol root.",
    )
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_806)
    return parser.parse_args()


def method_roots(
    source_root: Path,
    *,
    deep_root: Path | None = None,
    reference_root: Path | None = None,
    sbts_root: Path | None = None,
) -> dict[str, Path]:
    return {
        "Deep MKV": (
            deep_root
            if deep_root is not None
            else source_root / "protocol" / "Deep-MKV-final-online-volatility"
        ),
        "Fitted reference": (
            reference_root
            if reference_root is not None
            else source_root / "reference_protocol" / "Fitted-reference-final"
        ),
        "SBTS": (
            sbts_root
            if sbts_root is not None
            else REPO_ROOT / "runs" / "real_data_protocol" / "SBTS"
        ),
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def phase_dir(root: Path, index: str, seed: int, protocol: str) -> Path:
    return root / index / f"seed_{seed}" / protocol


def phase_rows(root: Path, index: str, protocol: str, phase: str) -> list[dict]:
    return [
        read_json(phase_dir(root, index, seed, protocol) / "metrics.json")["phases"][phase]
        for seed in SEEDS
    ]


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def aggregate_phase(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        "law_metrics": {
            metric: stats([float(row["law_metrics"][metric]) for row in rows])
            for metric in LAW_METRICS
        },
        "path_shadowing_metrics": {
            metric: stats(
                [float(row["path_shadowing_metrics"][metric]) for row in rows]
            )
            for metric in CRPS_METRICS
        },
    }


def metric_mean(phase: dict, metric: str) -> float:
    family = "law_metrics" if metric in LAW_METRICS else "path_shadowing_metrics"
    return float(phase[family][metric]["mean"])


def relative_gain(primary: float, comparison: float) -> float:
    return float((comparison - primary) / comparison)


def load_per_date(
    root: Path,
    index: str,
    seed: int,
    protocol: str,
    filename: str,
    metric: str,
) -> np.ndarray:
    path = phase_dir(root, index, seed, protocol) / filename
    with np.load(path, allow_pickle=False) as payload:
        if metric not in payload.files:
            raise KeyError(f"{metric!r} is unavailable in {path}")
        values = np.asarray(payload[metric], dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"{path}:{metric} must contain at least two finite values")
    return values


def moving_block_means(
    values: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    generator: np.random.Generator,
) -> np.ndarray:
    num_dates = int(values.size)
    if block_length < 1 or block_length > num_dates:
        raise ValueError("block length must lie between one and the phase length")
    if replicates < 2:
        raise ValueError("bootstrap replicates must be at least two")
    num_blocks = (num_dates + block_length - 1) // block_length
    starts = generator.integers(
        0,
        num_dates - block_length + 1,
        size=(replicates, num_blocks),
    )
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[..., None] + offsets).reshape(replicates, -1)
    return values[indices[:, :num_dates]].mean(axis=1)


def paired_interval(
    differences: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict:
    """Summarize primary-minus-comparison differences.

    ``differences`` has shape ``(indices, seeds, dates)``.  Dates are resampled
    in moving blocks after averaging matched banks.  The reported seed spread
    first averages the same seed across indices.
    """

    if differences.ndim != 3:
        raise ValueError("differences must have shape (indices, seeds, dates)")
    if differences.shape[1] != len(SEEDS):
        raise ValueError("the seed dimension does not match the locked protocol")
    date_differences = differences.mean(axis=(0, 1))
    seed_estimates = differences.mean(axis=(0, 2))
    bootstrap = moving_block_means(
        date_differences,
        block_length=block_length,
        replicates=replicates,
        generator=np.random.default_rng(seed),
    )
    alpha = 0.5 * (1.0 - confidence_level)
    lower = float(np.quantile(bootstrap, alpha))
    upper = float(np.quantile(bootstrap, 1.0 - alpha))
    if upper < 0.0:
        decision = "Deep MKV lower"
    elif lower > 0.0:
        decision = "Deep MKV higher"
    else:
        decision = "interval includes zero"
    return {
        "difference_direction": "Deep MKV minus comparison",
        "estimate": float(date_differences.mean()),
        "bootstrap_standard_error": float(bootstrap.std(ddof=0)),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": confidence_level,
        "decision": decision,
        "num_indices": int(differences.shape[0]),
        "num_dates": int(differences.shape[2]),
        "num_seed_pairs": int(differences.shape[0] * differences.shape[1]),
        "block_length_dates": block_length,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "seed_estimates": [float(value) for value in seed_estimates],
        "seed_sample_standard_deviation": float(seed_estimates.std(ddof=1)),
    }


def collect_differences(
    roots: dict[str, Path],
    indices: tuple[str, ...],
    comparison: str,
    metric: str,
) -> np.ndarray:
    per_index = []
    for index in indices:
        per_seed = []
        for seed in SEEDS:
            deep = load_per_date(
                roots["Deep MKV"],
                index,
                seed,
                "primary_2026_holdout",
                "test_per_path_metrics.npz",
                metric,
            )
            other = load_per_date(
                roots[comparison],
                index,
                seed,
                "primary_2026_holdout",
                "test_per_path_metrics.npz",
                metric,
            )
            if deep.shape != other.shape:
                raise ValueError(
                    f"unaligned per-date arrays for {index}, seed {seed}, {comparison}"
                )
            per_seed.append(deep - other)
        per_index.append(np.stack(per_seed))
    shapes = {array.shape for array in per_index}
    if len(shapes) != 1:
        raise ValueError(f"indices have different seed/date shapes: {sorted(shapes)}")
    return np.stack(per_index)


def make_payload(args: argparse.Namespace) -> dict:
    source_root = args.source_root.resolve()
    roots = method_roots(
        source_root,
        deep_root=args.deep_root.resolve() if args.deep_root is not None else None,
        reference_root=(
            args.reference_root.resolve()
            if args.reference_root is not None
            else None
        ),
        sbts_root=args.sbts_root.resolve() if args.sbts_root is not None else None,
    )
    methods: dict[str, dict] = {}
    for index in INDICES:
        methods[index] = {}
        for method, root in roots.items():
            methods[index][method] = {
                "test": aggregate_phase(
                    phase_rows(root, index, "primary_2026_holdout", "test")
                ),
                "war": aggregate_phase(
                    phase_rows(root, index, "us_iran_2026_war", "war")
                ),
            }

    intervals: dict[str, dict] = {index: {} for index in (*INDICES, "pooled")}
    counter = 0
    for scope in (*INDICES, "pooled"):
        selected_indices = INDICES if scope == "pooled" else (scope,)
        for comparison in ("Fitted reference", "SBTS"):
            intervals[scope][comparison] = {}
            for metric in CRPS_METRICS:
                differences = collect_differences(
                    roots, selected_indices, comparison, metric
                )
                intervals[scope][comparison][metric] = paired_interval(
                    differences,
                    block_length=int(args.block_length),
                    replicates=int(args.bootstrap_replicates),
                    confidence_level=float(args.confidence_level),
                    seed=int(args.bootstrap_seed) + counter,
                )
                counter += 1

    primary_wins = 0
    crisis_wins = 0
    for index in INDICES:
        for metric in CRPS_METRICS:
            if metric_mean(methods[index]["Deep MKV"]["test"], metric) < metric_mean(
                methods[index]["SBTS"]["test"], metric
            ):
                primary_wins += 1
            if metric_mean(methods[index]["Deep MKV"]["war"], metric) < metric_mean(
                methods[index]["SBTS"]["war"], metric
            ):
                crisis_wins += 1

    return {
        "scope": {
            "indices": list(INDICES),
            "description": (
                "Highly liquid futures on large-capitalization U.S. equity indices"
            ),
            "seeds": list(SEEDS),
            "completed_banks_per_multiseed_method": len(INDICES) * len(SEEDS),
            "selection_scope": "training_and_validation_only",
            "test_and_crisis_role": "final_evaluation_only",
        },
        "bootstrap_protocol": {
            "difference_direction": "Deep MKV minus comparison; negative favors Deep MKV",
            "block_length_dates": int(args.block_length),
            "replicates": int(args.bootstrap_replicates),
            "confidence_level": float(args.confidence_level),
            "seed": int(args.bootstrap_seed),
            "uncertainty_axis": (
                "moving-date-block bootstrap after averaging matched generated-bank seeds"
            ),
        },
        "indices": methods,
        "paired_crps_intervals": intervals,
        "headline": {
            "primary_deep_vs_sbts_crps_wins": primary_wins,
            "primary_crps_comparisons": len(INDICES) * len(CRPS_METRICS),
            "war_deep_vs_sbts_crps_wins": crisis_wins,
            "war_crps_comparisons": len(INDICES) * len(CRPS_METRICS),
        },
        "sources": {name: str(path.resolve()) for name, path in roots.items()},
    }


def format_interval(interval: dict) -> str:
    return (
        f"{1000.0 * interval['estimate']:.3f} "
        f"[{1000.0 * interval['ci_lower']:.3f}, "
        f"{1000.0 * interval['ci_upper']:.3f}]"
    )


def markdown(payload: dict) -> str:
    lines = [
        "# Final large-cap futures evaluation",
        "",
        "This report contains only ES, NQ, and YM: highly liquid futures on "
        "large-capitalization U.S. equity indices. All checkpoints and settings "
        "were selected before the chronological test and crisis windows were opened.",
        "",
        "## Completion",
        "",
        "Deep MKV, the fitted reference, and SBTS each have 12/12 completed "
        "index-seed evaluations (three indices, four seeds). No bank is excluded.",
        "",
        "## Chronological test",
        "",
        "Scores are four-seed means and lower is better. CRPS values are multiplied by 1,000.",
        "",
        "| Index | Method | Cum. CRPS | Incr. CRPS | RV CRPS | Return QQ | Abs-ACF | RV W1 | Drawdown W1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index in INDICES:
        for method in ("Deep MKV", "Fitted reference", "SBTS"):
            phase = payload["indices"][index][method]["test"]
            shadow = phase["path_shadowing_metrics"]
            law = phase["law_metrics"]
            lines.append(
                f"| {index} | {method} | "
                f"{1000*shadow['cumulative_return_crps']['mean']:.3f} | "
                f"{1000*shadow['increment_crps']['mean']:.3f} | "
                f"{1000*shadow['realized_volatility_crps']['mean']:.3f} | "
                f"{law['return_qq_rmse_normalized']['mean']:.4f} | "
                f"{law['abs_return_acf_error_rms']['mean']:.4f} | "
                f"{law['realized_volatility_wasserstein_normalized']['mean']:.4f} | "
                f"{law['maximum_drawdown_wasserstein_normalized']['mean']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Deep MKV correction relative to the fitted reference",
            "",
            "Positive percentages mean that Deep MKV removes error.",
            "",
            "| Index | Cum. CRPS | Incr. CRPS | RV CRPS | Return QQ | Abs-ACF | RV W1 | Drawdown W1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index in INDICES:
        deep = payload["indices"][index]["Deep MKV"]["test"]
        reference = payload["indices"][index]["Fitted reference"]["test"]
        gains = {
            metric: relative_gain(metric_mean(deep, metric), metric_mean(reference, metric))
            for metric in (*CRPS_METRICS, *LAW_METRICS)
        }
        lines.append(
            f"| {index} | {100*gains['cumulative_return_crps']:.1f}% | "
            f"{100*gains['increment_crps']:.1f}% | "
            f"{100*gains['realized_volatility_crps']:.1f}% | "
            f"{100*gains['return_qq_rmse_normalized']:.1f}% | "
            f"{100*gains['abs_return_acf_error_rms']:.1f}% | "
            f"{100*gains['realized_volatility_wasserstein_normalized']:.1f}% | "
            f"{100*gains['maximum_drawdown_wasserstein_normalized']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Paired uncertainty for chronological-test CRPS",
            "",
            "Entries are Deep MKV minus comparison, multiplied by 1,000, with "
            "95% moving-date-block intervals. Negative values favor Deep MKV. "
            "The pooled row averages the three indices and four matched seeds "
            "before resampling dates.",
            "",
            "| Scope | Metric | vs reference | vs SBTS |",
            "|---|---|---:|---:|",
        ]
    )
    for scope in (*INDICES, "pooled"):
        for metric in CRPS_METRICS:
            reference = payload["paired_crps_intervals"][scope]["Fitted reference"][metric]
            sbts = payload["paired_crps_intervals"][scope]["SBTS"][metric]
            lines.append(
                f"| {scope.upper() if scope != 'pooled' else 'Pooled'} | "
                f"{CRPS_LABELS[metric]} | "
                f"{format_interval(reference)} | {format_interval(sbts)} |"
            )

    lines.extend(
        [
            "",
            "## War-window comparison with SBTS",
            "",
            "CRPS values are multiplied by 1,000. Each cell reports Deep MKV / SBTS; lower is better.",
            "",
            "| Index | Cumulative CRPS | Increment CRPS | RV CRPS |",
            "|---|---:|---:|---:|",
        ]
    )
    for index in INDICES:
        deep = payload["indices"][index]["Deep MKV"]["war"]
        sbts = payload["indices"][index]["SBTS"]["war"]
        cells = []
        for metric in CRPS_METRICS:
            cells.append(
                f"{1000*metric_mean(deep, metric):.3f} / {1000*metric_mean(sbts, metric):.3f}"
            )
        lines.append(f"| {index} | {' | '.join(cells)} |")

    headline = payload["headline"]
    lines.extend(
        [
            "",
            f"Deep MKV beats SBTS in {headline['primary_deep_vs_sbts_crps_wins']}/"
            f"{headline['primary_crps_comparisons']} chronological-test CRPS comparisons "
            f"and {headline['war_deep_vs_sbts_crps_wins']}/"
            f"{headline['war_crps_comparisons']} war-window comparisons.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not 0.0 < float(args.confidence_level) < 1.0:
        raise ValueError("confidence level must lie between zero and one")
    payload = make_payload(args)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_json = output_root / "FINAL_LARGE_CAP_EVALUATION.json"
    output_md = output_root / "FINAL_LARGE_CAP_EVALUATION.md"
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = markdown(payload)
    output_md.write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
