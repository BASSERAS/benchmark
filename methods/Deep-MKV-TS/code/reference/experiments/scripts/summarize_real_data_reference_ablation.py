#!/usr/bin/env python3
"""Aggregate the four-index, four-seed real-data reference ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "real_reference_ablation_4seed_20260802"
INDICES = ("ES", "NQ", "RTY", "YM")
SEEDS = (0, 1, 3, 4)
STAGES = {
    "reference": "Deep-MKV-reference-only",
    "source": "Deep-MKV-source-stage",
    "complete": "Deep-MKV-complete",
}
METRICS = (
    "realized_volatility_crps",
    "cumulative_return_crps",
    "increment_crps",
    "realized_volatility_coverage_90",
    "realized_volatility_band_width_90",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def read_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    shadow = payload["phases"]["test"]["path_shadowing_metrics"]
    return {metric: float(shadow[metric]) for metric in METRICS}


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": float(statistics.stdev(values)),
    }


def fmt(metric: str, value: float) -> str:
    if metric == "realized_volatility_coverage_90":
        return f"{value:.3f}"
    return f"{1_000.0 * value:.3f}"


def fmt_difference(metric: str, value: float) -> str:
    if metric == "realized_volatility_coverage_90":
        return f"{value:.4f}"
    return f"{1_000.0 * value:.4f}"


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    per_seed: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for index in INDICES:
        per_seed[index] = {}
        for stage, model_name in STAGES.items():
            per_seed[index][stage] = {}
            for seed in SEEDS:
                path = (
                    root
                    / "protocol"
                    / model_name
                    / index
                    / f"seed_{seed}"
                    / "primary_2026_holdout"
                    / "metrics.json"
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                per_seed[index][stage][str(seed)] = read_metrics(path)

    by_index: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for index in INDICES:
        by_index[index] = {}
        for stage in STAGES:
            by_index[index][stage] = {
                metric: aggregate(
                    [per_seed[index][stage][str(seed)][metric] for seed in SEEDS]
                )
                for metric in METRICS
            }

    panel_by_seed: dict[str, dict[str, dict[str, float]]] = {}
    for stage in STAGES:
        panel_by_seed[stage] = {}
        for seed in SEEDS:
            panel_by_seed[stage][str(seed)] = {
                metric: float(
                    statistics.mean(
                        per_seed[index][stage][str(seed)][metric]
                        for index in INDICES
                    )
                )
                for metric in METRICS
            }
    panel = {
        stage: {
            metric: aggregate(
                [panel_by_seed[stage][str(seed)][metric] for seed in SEEDS]
            )
            for metric in METRICS
        }
        for stage in STAGES
    }

    interval_dir = root / "paired_date_block"
    paired_date_block: dict[str, dict[str, float | int]] = {}
    for metric in (
        "realized_volatility_crps",
        "cumulative_return_crps",
        "increment_crps",
    ):
        interval_path = interval_dir / f"panel_{metric}.json"
        if interval_path.is_file():
            interval = json.loads(interval_path.read_text(encoding="utf-8"))
            paired_date_block[metric] = {
                "estimate": float(interval["estimate"]),
                "ci_lower": float(interval["ci_lower"]),
                "ci_upper": float(interval["ci_upper"]),
                "block_length_dates": int(interval["block_length_dates"]),
                "bootstrap_replicates": int(interval["bootstrap_replicates"]),
            }

    paired: dict[str, object] = {}
    comparisons = {
        "source_minus_reference": ("reference", "source"),
        "complete_minus_reference": ("reference", "complete"),
        "complete_minus_source": ("source", "complete"),
    }
    for comparison, (baseline_stage, candidate_stage) in comparisons.items():
        paired[comparison] = {}
        for metric in METRICS:
            differences = []
            wins = 0
            for index in INDICES:
                for seed in SEEDS:
                    baseline = per_seed[index][baseline_stage][str(seed)][metric]
                    candidate = per_seed[index][candidate_stage][str(seed)][metric]
                    differences.append(candidate - baseline)
                    if metric == "realized_volatility_coverage_90":
                        wins += int(abs(candidate - 0.9) < abs(baseline - 0.9))
                    else:
                        wins += int(candidate < baseline)
            paired[comparison][metric] = {
                **aggregate(differences),
                "wins_out_of_16": int(wins),
            }

    payload = {
        "indices": list(INDICES),
        "seeds": list(SEEDS),
        "stage_model_names": STAGES,
        "metric_units": {
            "realized_volatility_crps": "raw; multiplied by 1000 in markdown",
            "cumulative_return_crps": "raw; multiplied by 1000 in markdown",
            "increment_crps": "raw; multiplied by 1000 in markdown",
            "realized_volatility_coverage_90": "fraction",
            "realized_volatility_band_width_90": "raw; multiplied by 1000 in markdown",
        },
        "per_seed": per_seed,
        "aggregate_by_index": by_index,
        "panel_by_seed": panel_by_seed,
        "panel_aggregate": panel,
        "paired_comparisons": paired,
        "paired_date_block_first_stage_minus_reference": paired_date_block,
    }
    (root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Real-data reference ablation",
        "",
        "Four-seed means (sample SDs). CRPS and width are multiplied by 1,000.",
        "",
        "| Index | Stage | RV CRPS | Cum. CRPS | Inc. CRPS | RV cov. | RV width |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "reference": "Reference only",
        "source": "First MP stage",
        "complete": "Complete Deep MKV",
    }
    for index in (*INDICES, "Panel"):
        source = panel if index == "Panel" else by_index[index]
        for stage in STAGES:
            values = source[stage]
            cells = []
            for metric in METRICS:
                mean = fmt(metric, values[metric]["mean"])
                sd = fmt(metric, values[metric]["sample_sd"])
                cells.append(f"{mean} ({sd})")
            lines.append(
                f"| {index} | {labels[stage]} | " + " | ".join(cells) + " |"
            )
    lines.extend(["", "## Paired stage comparisons", ""])
    comparison_labels = {
        "source_minus_reference": "First MP stage minus reference",
        "complete_minus_reference": "Complete Deep MKV minus reference",
        "complete_minus_source": "Complete Deep MKV minus first MP stage",
    }
    for comparison, comparison_label in comparison_labels.items():
        lines.append(f"### {comparison_label}")
        lines.append("")
        lines.append("| Metric | Mean difference | Wins |")
        lines.append("|---|---:|---:|")
        for metric in METRICS:
            values = paired[comparison][metric]
            difference = fmt_difference(metric, values["mean"])
            lines.append(
                f"| {metric} | {difference} | {values['wins_out_of_16']}/16 |"
            )
        lines.append("")
    if paired_date_block:
        lines.extend(
            [
                "## Paired date-block intervals",
                "",
                (
                    "First MP stage minus reference, averaged across the four "
                    "indices and four fitted seeds. Five-session moving blocks; "
                    "10,000 draws. Values are multiplied by 1,000."
                ),
                "",
                "| Metric | Estimate [95% interval] |",
                "|---|---:|",
            ]
        )
        for metric, values in paired_date_block.items():
            estimate = 1_000.0 * float(values["estimate"])
            lower = 1_000.0 * float(values["ci_lower"])
            upper = 1_000.0 * float(values["ci_upper"])
            lines.append(
                f"| {metric} | {estimate:.3f} [{lower:.3f}, {upper:.3f}] |"
            )
        lines.append("")
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "SUMMARY.md")


if __name__ == "__main__":
    main()
