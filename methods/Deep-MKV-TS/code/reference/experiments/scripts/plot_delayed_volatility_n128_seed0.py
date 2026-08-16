#!/usr/bin/env python3
"""Plot the best-seed N=128 delayed-volatility comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

COLORS = {
    "Target": "#111111",
    "Reference": "#7A7A7A",
    "Deep MKV": "#1464A5",
    "SBTS": "#D97706",
}

LINESTYLES = {
    "Target": "-",
    "Reference": "--",
    "Deep MKV": "-",
    "SBTS": "-.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "paper_figures_guyon_seed0_20260809",
    )
    parser.add_argument("--rolling-window", type=int, default=8)
    return parser.parse_args()


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def load_prices(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(f"{path} must contain finite positive price paths")
    return values


def delayed_features(
    prices: np.ndarray,
    *,
    threshold: float,
    history_cutoff: int,
    split: int,
    future_horizon: int,
    annualization: float,
    rolling_window: int,
) -> dict[str, np.ndarray | float]:
    log_prices = np.log(prices / prices[:, :1])
    returns = np.diff(log_prices, axis=1)
    drawdown = np.maximum.accumulate(log_prices, axis=1) - log_prices
    early_hit = drawdown[:, : history_cutoff + 1].max(axis=1) >= threshold
    if early_hit.all() or (~early_hit).all():
        raise RuntimeError("both early-drawdown groups must be non-empty")

    rolling_variance = np.lib.stride_tricks.sliding_window_view(
        returns**2, rolling_window, axis=1
    ).mean(axis=-1)
    rolling_volatility = np.sqrt(annualization * rolling_variance)
    hit_curve = rolling_volatility[early_hit].mean(axis=0)
    no_hit_curve = rolling_volatility[~early_hit].mean(axis=0)
    gap_curve = hit_curve - no_hit_curve
    standard_error = np.sqrt(
        rolling_volatility[early_hit].var(axis=0, ddof=1) / early_hit.sum()
        + rolling_volatility[~early_hit].var(axis=0, ddof=1) / (~early_hit).sum()
    )

    future_returns = returns[:, split : split + future_horizon]
    future_rv = np.sqrt(annualization * np.mean(future_returns**2, axis=1))
    future_gap = float(future_rv[early_hit].mean() - future_rv[~early_hit].mean())
    correlation = float(
        np.corrcoef(early_hit.astype(np.float64), future_rv)[0, 1]
    )
    return {
        "early_hit": early_hit,
        "early_hit_rate": float(early_hit.mean()),
        "rolling_gap": gap_curve,
        "rolling_gap_se": standard_error,
        "future_rv_gap": future_gap,
        "early_hit_future_rv_correlation": correlation,
    }


def main() -> None:
    args = parse_args()
    if args.rolling_window < 2:
        raise ValueError("rolling-window must be at least two")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    publication_style()

    dataset_root = REPO_ROOT / "runs" / "drawdown_memory_seed0_comparison_20260730" / "data"
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["configuration"]
    paths = {
        "Target": dataset_root / "disc.npy",
        "Reference": REPO_ROOT
        / "runs/delayed_guyon_yweight0_n128_checkpoint_seed0_20260809/"
        "confirmation_selection/bank_0/reference_bank.npy",
        "Deep MKV": REPO_ROOT
        / "runs/delayed_guyon_price_rv_deepmkv_4seed_20260809/seed_1/"
        "generic_ce_online_source_volatility_only/generated_paths_8192x128.npy",
        "SBTS": REPO_ROOT
        / "runs/paper_synthetic_4seed_20260801/drawdown/sbts/seed_1/"
        "generated_paths_8192x128.npy",
    }
    banks = {name: load_prices(path) for name, path in paths.items()}
    features = {
        name: delayed_features(
            bank,
            threshold=float(config["drawdown_threshold"]),
            history_cutoff=int(config["history_cutoff"]),
            split=int(config["split_index"]),
            future_horizon=int(config["future_horizon"]),
            annualization=1.0 / float(config["dt"]),
            rolling_window=int(args.rolling_window),
        )
        for name, bank in banks.items()
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), gridspec_kw={"width_ratios": [1.55, 1.0]})
    time = np.arange(args.rolling_window, int(config["sequence_length"]), dtype=np.int64)
    for name in ("Target", "Reference", "Deep MKV", "SBTS"):
        values = np.asarray(features[name]["rolling_gap"], dtype=np.float64)
        se = np.asarray(features[name]["rolling_gap_se"], dtype=np.float64)
        color = COLORS[name]
        width = 2.0 if name == "Target" else 1.65
        axes[0].plot(
            time,
            values,
            color=color,
            linestyle=LINESTYLES[name],
            linewidth=width,
            label=name,
        )
        axes[0].fill_between(
            time,
            values - 1.96 * se,
            values + 1.96 * se,
            color=color,
            alpha=0.07,
            linewidth=0,
        )
    cutoff = int(config["history_cutoff"])
    split = int(config["split_index"])
    delay = int(config["response_delay"])
    axes[0].axvline(cutoff, color="#999999", linewidth=0.9, linestyle=":")
    axes[0].axvline(split, color="#666666", linewidth=0.9, linestyle="--")
    axes[0].annotate(
        "32-step delay",
        xy=(cutoff + delay, 0.015),
        xytext=(cutoff, 0.015),
        arrowprops={"arrowstyle": "<->", "color": "#777777", "linewidth": 0.8},
        color="#666666",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    axes[0].text(cutoff - 1, 0.307, "event cutoff", color="#777777", ha="right", va="top", fontsize=7.2)
    axes[0].text(split + 1, 0.307, "forecast split", color="#555555", ha="left", va="top", fontsize=7.2)
    axes[0].set_title("Delayed volatility response")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Early-drawdown minus no-drawdown\nannualized rolling volatility")
    axes[0].set_xlim(args.rolling_window, int(config["sequence_length"]) - 1)
    axes[0].set_ylim(-0.02, 0.32)
    axes[0].grid(alpha=0.17)

    target_gap = float(features["Target"]["future_rv_gap"])
    target_corr = float(features["Target"]["early_hit_future_rv_correlation"])
    names = ("Target", "Reference", "Deep MKV", "SBTS")
    ratios = np.asarray(
        [
            [float(features[name]["future_rv_gap"]) / target_gap for name in names],
            [
                float(features[name]["early_hit_future_rv_correlation"]) / target_corr
                for name in names
            ],
        ]
    )
    x = np.arange(2, dtype=np.float64)
    width = 0.19
    offsets = (-1.5, -0.5, 0.5, 1.5)
    for offset, name, values in zip(offsets, names, ratios.T):
        axes[1].bar(
            x + offset * width,
            100.0 * values,
            width,
            color=COLORS[name],
            label=name,
        )
    axes[1].axhline(100.0, color="#AAAAAA", linewidth=0.8, linestyle=":")
    axes[1].set_title("Target dependence recovered")
    axes[1].set_ylabel("Percent of target")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(("Future-RV\ngap", "Hit--future-RV\ncorrelation"))
    axes[1].set_ylim(0.0, 112.0)
    axes[1].grid(axis="y", alpha=0.17)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.25, wspace=0.34)
    stem = args.output_dir / "delayed_volatility_n128_reference_deepmkv_sbts_seed1"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    payload = {
        "scope": "validation_only",
        "test_split_loaded": False,
        "sequence_length": int(config["sequence_length"]),
        "selected_seed": 1,
        "selected_deep_mkv_checkpoint": 3000,
        "rolling_window": int(args.rolling_window),
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "summary": {
            name: {
                "early_hit_rate": float(features[name]["early_hit_rate"]),
                "future_rv_gap": float(features[name]["future_rv_gap"]),
                "early_hit_future_rv_correlation": float(
                    features[name]["early_hit_future_rv_correlation"]
                ),
                "future_rv_gap_target_fraction": float(
                    features[name]["future_rv_gap"] / target_gap
                ),
                "correlation_target_fraction": float(
                    features[name]["early_hit_future_rv_correlation"] / target_corr
                ),
            }
            for name in names
        },
    }
    (stem.with_suffix(".json")).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
