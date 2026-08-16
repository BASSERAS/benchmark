#!/usr/bin/env python3
"""Create the real-futures forecasting and drawdown-decision paper figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

COLORS = {
    "Deep MKV": "#1464A5",
    "Fitted reference": "#7A7A7A",
    "SBTS": "#D97706",
    "Historical neighbor": "#7B61A8",
    "Moving-block bootstrap": "#3A9D5D",
    "Whole-session bootstrap": "#A8557A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "paper_figures_guyon_seed0_20260809",
    )
    return parser.parse_args()


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def futures_figure(output_dir: Path) -> dict[str, object]:
    source = (
        REPO_ROOT
        / "runs/final_yfree_large_cap_evaluation_20260808/"
        "FINAL_LARGE_CAP_EVALUATION.json"
    )
    report = json.loads(source.read_text(encoding="utf-8"))
    indices = ("ES", "NQ", "YM")
    metrics = (
        ("cumulative_return_crps", "Cum."),
        ("increment_crps", "Incr."),
        ("realized_volatility_crps", "RV"),
    )
    rows = (
        ("test", "Fitted reference", "Test: vs reference"),
        ("test", "SBTS", "Test: vs SBTS"),
        ("war", "Fitted reference", "Stress: vs reference"),
        ("war", "SBTS", "Stress: vs SBTS"),
    )

    values = np.empty((len(rows), len(indices) * len(metrics)), dtype=np.float64)
    for row, (split, comparison, _) in enumerate(rows):
        for column, (index, metric, _) in enumerate(
            (index, metric, short)
            for index in indices
            for metric, short in metrics
        ):
            baseline = float(
                report["indices"][index][comparison][split]
                ["path_shadowing_metrics"][metric]["mean"]
            )
            deep = float(
                report["indices"][index]["Deep MKV"][split]
                ["path_shadowing_metrics"][metric]["mean"]
            )
            values[row, column] = 100.0 * (baseline - deep) / baseline

    fig, ax = plt.subplots(figsize=(7.15, 2.95), constrained_layout=True)
    limit = max(35.0, float(np.ceil(np.max(np.abs(values)) / 5.0) * 5.0))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    labels = [f"{index}\n{short}" for index in indices for _, short in metrics]
    image = ax.imshow(values, cmap="RdBu", norm=norm, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:+.0f}%",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.52 * limit else "#222222",
                fontsize=7.4,
                fontweight="bold" if value > 0.0 else "normal",
            )
    ax.set_title("Deep MKV conditional-forecast improvement")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([label for _, _, label in rows])
    for boundary in (2.5, 5.5):
        ax.axvline(boundary, color="white", linewidth=1.6)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.025)
    colorbar.set_label("CRPS reduction; positive favors Deep MKV")
    stem = output_dir / "futures_path_shadowing_crps_test_stress"
    save_figure(fig, stem)
    return {
        "source": str(source.resolve()),
        "deep_mkv_improvement_percent": {
            label: {
                labels[column].replace("\n", "_"): float(values[row, column])
                for column in range(values.shape[1])
            }
            for row, (_, _, label) in enumerate(rows)
        },
    }


def drawdown_figure(output_dir: Path) -> dict[str, object]:
    source = REPO_ROOT / "runs/es_drawdown_risk_yfree_aggregate_20260808/AGGREGATE.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    names = (
        "Deep MKV",
        "Fitted reference",
        "SBTS",
        "Historical neighbor",
        "Moving-block bootstrap",
        "Whole-session bootstrap",
    )
    short_names = (
        "Deep MKV",
        "Reference",
        "SBTS",
        "Historical",
        "Block boot.",
        "Session boot.",
    )
    x = np.arange(len(names), dtype=np.float64)
    leverage = np.asarray(
        [report["methods"][name]["metrics"]["leverage"]["estimate"] for name in names]
    )
    leverage_low = np.asarray(
        [report["methods"][name]["metrics"]["leverage"]["ci_lower"] for name in names]
    )
    leverage_high = np.asarray(
        [report["methods"][name]["metrics"]["leverage"]["ci_upper"] for name in names]
    )
    violation = 100.0 * np.asarray(
        [report["methods"][name]["metrics"]["violation"]["estimate"] for name in names]
    )
    violation_low = 100.0 * np.asarray(
        [report["methods"][name]["metrics"]["violation"]["ci_lower"] for name in names]
    )
    violation_high = 100.0 * np.asarray(
        [report["methods"][name]["metrics"]["violation"]["ci_upper"] for name in names]
    )
    colors = [COLORS[name] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    axes[0].bar(x, leverage, color=colors, width=0.72)
    axes[0].errorbar(
        x,
        leverage,
        yerr=np.vstack((leverage - leverage_low, leverage_high - leverage)),
        fmt="none",
        ecolor="#222222",
        elinewidth=0.9,
        capsize=2.5,
    )
    axes[0].set_title("Permitted ES futures exposure")
    axes[0].set_ylabel("Exposure multiple")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=31, ha="right")
    axes[0].set_ylim(1.65, 2.12)
    axes[0].grid(axis="y", alpha=0.17)

    axes[1].bar(x, violation, color=colors, width=0.72)
    axes[1].errorbar(
        x,
        violation,
        yerr=np.vstack((violation - violation_low, violation_high - violation)),
        fmt="none",
        ecolor="#222222",
        elinewidth=0.9,
        capsize=2.5,
    )
    axes[1].axhline(10.0, color="#B33A3A", linestyle="--", linewidth=1.2, label="10% target")
    axes[1].set_title("Observed drawdown-limit violations")
    axes[1].set_ylabel("Test sessions (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=31, ha="right")
    axes[1].set_ylim(0.0, max(15.0, 1.08 * violation_high.max()))
    axes[1].grid(axis="y", alpha=0.17)
    axes[1].legend(frameon=False, loc="upper left")
    fig.subplots_adjust(bottom=0.32, wspace=0.30)
    stem = output_dir / "es_drawdown_risk_exposure_and_violations"
    save_figure(fig, stem)
    return {
        "source": str(source.resolve()),
        "methods": {
            name: {
                "leverage": float(leverage[row]),
                "leverage_ci": [float(leverage_low[row]), float(leverage_high[row])],
                "violation_percent": float(violation[row]),
                "violation_ci_percent": [
                    float(violation_low[row]),
                    float(violation_high[row]),
                ],
            }
            for row, name in enumerate(names)
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    publication_style()
    payload = {
        "futures": futures_figure(args.output_dir),
        "drawdown_risk": drawdown_figure(args.output_dir),
    }
    output = args.output_dir / "real_futures_and_drawdown_plot_data.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
