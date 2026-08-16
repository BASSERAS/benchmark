#!/usr/bin/env python3
"""Create the matched seed-zero Guyon comparison figures for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.heston_mixture import regimes  # noqa: E402


COLORS = {
    "target": "#111111",
    "reference": "#7A7A7A",
    "deep_mkv": "#1464A5",
    "sbts": "#D97706",
    "exact": "#2E8B57",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "paper_figures_guyon_seed0_20260809",
    )
    return parser.parse_args()


def load_prices(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(f"{path} must contain finite positive price paths")
    return values


def log_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices / prices[:, :1]), axis=1)


def pooled_acf(values: np.ndarray, lags: np.ndarray) -> np.ndarray:
    centered = values - values.mean()
    variance = max(float(np.mean(centered * centered)), 1e-18)
    return np.asarray(
        [np.mean(centered[:, :-lag] * centered[:, lag:]) / variance for lag in lags]
    )


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    probability = np.arange(1, len(ordered) + 1, dtype=np.float64) / len(ordered)
    return ordered, probability


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


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def heston_figure(output_dir: Path) -> dict[str, object]:
    paths = {
        "Target": Path(
            "/home/samer/scenarios/BASSERAS-benchmark/dataset/Heston/"
            "heston_S_disc_8192x128.npy"
        ),
        "Reference": REPO_ROOT
        / "runs/heston_guyon_price_rv_mkv_seed0_20260809/"
        "reference/validation_bank.npy",
        "Deep MKV": REPO_ROOT
        / "runs/heston_guyon_price_rv_mkv_seed0_20260809/"
        "volatility_only_online_mp/checkpoint_evaluations/step_2500/validation_bank.npy",
        "SBTS": Path(
            "/home/samer/scenarios/BASSERAS-benchmark/methods/SBTS/"
            "generated_paths/seed_0/generated_paths_8192x128.npy"
        ),
    }
    banks = {name: load_prices(path) for name, path in paths.items()}
    returns = {name: log_returns(bank) for name, bank in banks.items()}
    annualization = 250.0
    realized_volatility = {
        name: np.sqrt(annualization * np.mean(values**2, axis=1))
        for name, values in returns.items()
    }
    lags = np.arange(1, 21, dtype=np.int64)
    absolute_acf = {
        name: pooled_acf(np.abs(values), lags) for name, values in returns.items()
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65))
    style = {
        "Target": (COLORS["target"], "-", 2.0),
        "Reference": (COLORS["reference"], "--", 1.6),
        "Deep MKV": (COLORS["deep_mkv"], "-", 1.8),
        "SBTS": (COLORS["sbts"], "-.", 1.5),
    }
    for name in ("Target", "Reference", "Deep MKV", "SBTS"):
        color, line_style, width = style[name]
        x, y = empirical_cdf(realized_volatility[name])
        axes[0].plot(x, y, color=color, linestyle=line_style, linewidth=width, label=name)
        axes[1].plot(
            lags,
            absolute_acf[name],
            color=color,
            linestyle=line_style,
            linewidth=width,
            label=name,
        )
    axes[0].set_title("Realized-volatility distribution")
    axes[0].set_xlabel("Annualized realized volatility")
    axes[0].set_ylabel("Empirical probability")
    axes[0].grid(alpha=0.18)
    axes[1].set_title("Volatility persistence")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("ACF of absolute returns")
    axes[1].set_xticks((1, 5, 10, 15, 20))
    axes[1].grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.25, wspace=0.30)
    save_figure(fig, output_dir / "heston_deepmkv_sbts_seed0")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "checkpoint": 2500,
        "realized_volatility_mean": {
            name: float(values.mean()) for name, values in realized_volatility.items()
        },
        "absolute_return_acf": {
            name: values.tolist() for name, values in absolute_acf.items()
        },
    }


def read_mixture_proportions(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    values = np.asarray(
        report["mixture_fidelity"]["generated_regime_proportions"],
        dtype=np.float64,
    )
    if values.shape != (8,):
        raise ValueError(f"{path} does not contain eight regime proportions")
    return values


def mixture_figure(output_dir: Path) -> dict[str, object]:
    root = REPO_ROOT / "runs/mixture_guyon_price_rv_mkv_4seed_20260809"
    metric_paths = {
        "Reference": root / "seed_1/reference/mixture_metrics.json",
        "Deep MKV": root
        / "seed_1/volatility_only_online_mp/checkpoint_evaluations/"
        "step_2500/mixture_metrics.json",
        "SBTS": REPO_ROOT
        / "runs/final_mixture_yfree_step3000_sbts_comparison_20260808/"
        "seed_1/sbts/mixture_metrics.json",
    }
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in metric_paths.items()
    }
    target = np.asarray(
        reports["Deep MKV"]["mixture_fidelity"]["target_regime_proportions"],
        dtype=np.float64,
    )
    proportions = {
        "Target": target,
        **{
            name: np.asarray(
                report["mixture_fidelity"]["generated_regime_proportions"],
                dtype=np.float64,
            )
            for name, report in reports.items()
        },
    }
    labels = [
        rf"$({'L' if row['theta'] < 0.05 else 'H'},"
        rf"{'L' if row['xi'] < 0.7 else 'H'},"
        rf"{'-' if row['rho'] < 0 else '+'})$"
        for row in regimes()
    ]
    x = np.arange(8, dtype=np.float64)
    width = 0.19
    offsets = (-1.5, -0.5, 0.5, 1.5)
    names = ("Target", "Reference", "Deep MKV", "SBTS")
    colors = (
        COLORS["target"],
        COLORS["reference"],
        COLORS["deep_mkv"],
        COLORS["sbts"],
    )
    fig, ax = plt.subplots(figsize=(7.15, 3.15))
    for offset, name, color in zip(offsets, names, colors):
        ax.bar(x + offset * width, proportions[name], width, label=name, color=color)
    ax.axhline(1.0 / 8.0, color="#BBBBBB", linewidth=0.8, zorder=0)
    ax.set_title("Persistent-mixture regime proportions inferred from generated paths")
    ax.set_ylabel("Share of paths")
    ax.set_xlabel(r"Regime $(\theta,\xi,\rho)$: L = low, H = high")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.7)
    ax.set_ylim(0.0, max(0.46, 1.08 * max(value.max() for value in proportions.values())))
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.20)
    save_figure(fig, output_dir / "mixture_deepmkv_sbts_seed1")

    bank_paths = {
        "Target": REPO_ROOT
        / "runs/heston_parameter_mixture_seed0_20260730/data/disc.npy",
        "Reference": root / "seed_1/reference/validation_bank.npy",
        "Deep MKV": root
        / "seed_1/volatility_only_online_mp/checkpoint_evaluations/"
        "step_2500/validation_bank.npy",
        "SBTS": REPO_ROOT
        / "runs/paper_synthetic_4seed_20260801/heston_mixture/sbts/"
        "seed_1/generated_paths_8192x128.npy",
    }
    banks = {name: load_prices(path) for name, path in bank_paths.items()}
    realized_volatility = {
        name: np.sqrt(250.0 * np.mean(log_returns(bank) ** 2, axis=1))
        for name, bank in banks.items()
    }
    probabilities = np.linspace(0.01, 0.99, 99)
    quantiles = {
        name: np.quantile(values, probabilities)
        for name, values in realized_volatility.items()
    }
    target_quantiles = quantiles["Target"]
    wasserstein = {
        name: float(
            np.mean(
                np.abs(
                    np.sort(values)
                    - np.sort(realized_volatility["Target"])
                )
            )
        )
        for name, values in realized_volatility.items()
        if name != "Target"
    }

    combined, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))
    lower = min(float(value.min()) for value in quantiles.values())
    upper = max(float(value.max()) for value in quantiles.values())
    axes[0].plot(
        (lower, upper),
        (lower, upper),
        color=COLORS["target"],
        linestyle=":",
        linewidth=1.5,
        label="Target (ideal)",
    )
    qq_styles = {
        "Reference": (COLORS["reference"], "--", 1.5),
        "Deep MKV": (COLORS["deep_mkv"], "-", 1.9),
        "SBTS": (COLORS["sbts"], "-.", 1.5),
    }
    for name, (color, line_style, line_width) in qq_styles.items():
        axes[0].plot(
            target_quantiles,
            quantiles[name],
            color=color,
            linestyle=line_style,
            linewidth=line_width,
            label=f"{name} ($W_1$={wasserstein[name]:.3f})",
        )
    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Realized-volatility quantiles")
    axes[0].set_xlabel("Target quantile")
    axes[0].set_ylabel("Generated quantile")
    axes[0].grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=6.8, loc="upper left")

    for offset, name, color in zip(offsets, names, colors):
        axes[1].bar(
            x + offset * width,
            proportions[name],
            width,
            label=name,
            color=color,
        )
    axes[1].axhline(1.0 / 8.0, color="#BBBBBB", linewidth=0.8, zorder=0)
    axes[1].set_title("Inferred regime proportions")
    axes[1].set_ylabel("Share of paths")
    axes[1].set_xlabel(r"Regime: $\theta$ level, $\xi$ level, sign of $\rho$")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        ("LL$-$", "LL$+$", "LH$-$", "LH$+$", "HL$-$", "HL$+$", "HH$-$", "HH$+$"),
        fontsize=6.8,
    )
    axes[1].set_ylim(
        0.0,
        max(0.46, 1.08 * max(value.max() for value in proportions.values())),
    )
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].legend(frameon=False, fontsize=6.8, loc="upper right")
    combined.subplots_adjust(wspace=0.34, bottom=0.18)
    save_figure(
        combined,
        output_dir / "mixture_rv_qq_and_regimes_seed1",
    )
    return {
        "metrics": {name: str(path) for name, path in metric_paths.items()},
        "banks": {name: str(path) for name, path in bank_paths.items()},
        "checkpoint": 2500,
        "regime_labels": labels,
        "regime_proportions": {
            name: value.tolist() for name, value in proportions.items()
        },
        "regime_tvd": {
            name: float(0.5 * np.abs(value - target).sum())
            for name, value in proportions.items()
        },
        "realized_volatility_quantile_probabilities": probabilities.tolist(),
        "realized_volatility_quantiles": {
            name: value.tolist() for name, value in quantiles.items()
        },
        "realized_volatility_w1": wasserstein,
    }


def tree_figure(output_dir: Path) -> dict[str, object] | None:
    run_dir = REPO_ROOT / "runs/garch_tree_guyon_yfree_seed0_20260809"
    exact_path = run_dir / "exact_report.json"
    report_path = run_dir / "seed_0/report.json"
    comparison_path = run_dir / "seed_0/tree_comparison.pt"
    history_path = run_dir / "seed_0/tree_validation_history.jsonl"
    required = (exact_path, report_path, comparison_path, history_path)
    if not all(path.is_file() for path in required):
        return None

    exact_report = json.loads(exact_path.read_text(encoding="utf-8"))
    deep_report = json.loads(report_path.read_text(encoding="utf-8"))
    comparison = torch.load(comparison_path, map_location="cpu", weights_only=True)
    history = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = np.asarray([row["step"] for row in history], dtype=np.float64)
    objective = np.asarray(
        [row["exact_tree_objective"] for row in history], dtype=np.float64
    )
    reference_objective = float(exact_report["reference"]["objective"]["total"])
    exact_objective = float(
        exact_report["exact_tree_optimum"]["objective"]["total"]
    )
    selected_step = int(deep_report["selection"]["selected_step"])
    selected_objective = float(deep_report["selection"]["selected_objective"])
    exact_sigma = np.exp(comparison["exact_variables"].double().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7))
    axes[0].plot(steps, objective, color=COLORS["deep_mkv"], linewidth=1.5)
    axes[0].axhline(
        reference_objective,
        color=COLORS["reference"],
        linestyle="--",
        linewidth=1.4,
        label="Guyon reference",
    )
    axes[0].axhline(
        exact_objective,
        color=COLORS["exact"],
        linestyle=":",
        linewidth=1.8,
        label="Exact tree solution",
    )
    axes[0].scatter(
        [selected_step],
        [selected_objective],
        color=COLORS["deep_mkv"],
        s=28,
        zorder=4,
        label="Selected Deep MKV",
    )
    axes[0].set_title("Objective during Deep MKV learning")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Exact tree objective")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.18)
    axes[0].legend(frameon=False, loc="best")

    deep_summary = deep_report["selected_deep_mkv"]["summary"]
    exact_summary = exact_report["exact_tree_optimum"]["summary"]
    categories = ("Objective", "Distribution", "Dependence")
    deep_values = np.asarray(
        [
            deep_report["selected_deep_mkv"]["objective_gain_fraction_of_exact"],
            deep_summary["distribution"]["median_gap_closed"],
            deep_summary["dependence"]["median_gap_closed"],
        ],
        dtype=np.float64,
    )
    exact_values = np.asarray(
        [
            1.0,
            exact_summary["distribution"]["median_gap_closed"],
            exact_summary["dependence"]["median_gap_closed"],
        ],
        dtype=np.float64,
    )
    x = np.arange(len(categories), dtype=np.float64)
    width = 0.34
    axes[1].bar(
        x - width / 2,
        deep_values,
        width,
        color=COLORS["deep_mkv"],
        label="Guyon--Deep MKV",
    )
    axes[1].bar(
        x + width / 2,
        exact_values,
        width,
        color=COLORS["exact"],
        label="Exact tree solution",
    )
    for offset, values in ((-width / 2, deep_values), (width / 2, exact_values)):
        for index, value in enumerate(values):
            axes[1].text(
                index + offset,
                min(float(value) + 0.025, 1.055),
                f"{100.0 * value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[1].set_ylim(0.0, 1.12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_title("Share of the reference gap removed")
    axes[1].set_ylabel("Fraction")
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )
    fig.subplots_adjust(wspace=0.30, bottom=0.24)
    save_figure(fig, output_dir / "tree_guyon_deepmkv_exact_seed0")
    return {
        "run_dir": str(run_dir),
        "reference_objective": reference_objective,
        "exact_objective": exact_objective,
        "selected_step": selected_step,
        "selected_objective": selected_objective,
        "objective_gain_fraction_of_exact": float(
            deep_report["selected_deep_mkv"]["objective_gain_fraction_of_exact"]
        ),
        "decision_nodes": int(exact_sigma.size),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    publication_style()
    report = {
        "selection": {
            "heston": {
                "seed": 0,
                "checkpoint": 2500,
                "scope": "validation only"
            },
            "persistent_mixture": {
                "seed": 1,
                "checkpoint": 2500,
                "scope": "validation only"
            },
        },
        "heston": heston_figure(output_dir),
        "persistent_mixture": mixture_figure(output_dir),
    }
    tree = tree_figure(output_dir)
    if tree is not None:
        report["exact_tree"] = tree
    (output_dir / "plot_data.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "files": sorted(p.name for p in output_dir.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
