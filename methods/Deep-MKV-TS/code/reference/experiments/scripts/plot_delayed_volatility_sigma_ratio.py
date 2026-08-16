#!/usr/bin/env python3
"""Plot the pathwise Deep-MKV/reference volatility ratio for delayed memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    CausalPathFeatureRidgeConditionalExpectation,
    DiscreteTimeGrid,
    fit_guyon_lekeufack_price_realized_volatility_reference_kernel,
)
from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    VolatilityOnlySpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402
from run_drawdown_memory_deep_mkv_ablation import (  # noqa: E402
    _build_model,
    _load_dataset,
)


COLORS = {
    "Threshold not crossed": "#1464A5",
    "Threshold crossed": "#C4473A",
}

RESPONSE_COLORS = {
    "Target": "#111111",
    "Reference": "#7A7A7A",
    "Deep MKV": "#1464A5",
    "SBTS": "#D97706",
}

RESPONSE_LINESTYLES = {
    "Target": "-",
    "Reference": "--",
    "Deep MKV": "-",
    "SBTS": "-.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "delayed_guyon_price_rv_deepmkv_4seed_20260809"
            / "seed_1"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "drawdown_memory_seed0_comparison_20260730"
            / "data"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "paper_figures_final_20260810",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bank-size", type=int, default=8192)
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


def early_drawdown_group(
    prices: np.ndarray,
    *,
    cutoff: int,
    threshold: float,
) -> np.ndarray:
    log_prices = np.log(prices / prices[:, :1])
    drawdown = np.maximum.accumulate(log_prices, axis=1) - log_prices
    group = drawdown[:, : cutoff + 1].max(axis=1) >= float(threshold)
    if group.all() or (~group).all():
        raise RuntimeError("both early-drawdown groups must be nonempty")
    return group


def quantile_curves(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "q10": np.quantile(values, 0.10, axis=0),
        "q25": np.quantile(values, 0.25, axis=0),
        "q50": np.quantile(values, 0.50, axis=0),
        "q75": np.quantile(values, 0.75, axis=0),
        "q90": np.quantile(values, 0.90, axis=0),
    }


def load_prices(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(f"{path} must contain finite positive price paths")
    return values


def delayed_response(
    prices: np.ndarray,
    *,
    threshold: float,
    history_cutoff: int,
    dt: float,
    rolling_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    log_prices = np.log(prices / prices[:, :1])
    returns = np.diff(log_prices, axis=1)
    drawdown = np.maximum.accumulate(log_prices, axis=1) - log_prices
    early_hit = drawdown[:, : history_cutoff + 1].max(axis=1) >= threshold
    if early_hit.all() or (~early_hit).all():
        raise RuntimeError("both early-drawdown groups must be nonempty")
    rolling_variance = np.lib.stride_tricks.sliding_window_view(
        returns**2,
        rolling_window,
        axis=1,
    ).mean(axis=-1)
    rolling_volatility = np.sqrt(rolling_variance / float(dt))
    hit_curve = rolling_volatility[early_hit].mean(axis=0)
    no_hit_curve = rolling_volatility[~early_hit].mean(axis=0)
    gap = hit_curve - no_hit_curve
    standard_error = np.sqrt(
        rolling_volatility[early_hit].var(axis=0, ddof=1) / early_hit.sum()
        + rolling_volatility[~early_hit].var(axis=0, ddof=1) / (~early_hit).sum()
    )
    return gap, standard_error


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    publication_style()

    target_paths, manifest, _ = _load_dataset(
        args.dataset_root,
        device=device,
        dtype=torch.float32,
    )
    config = manifest["configuration"]
    sequence_length = int(config["sequence_length"])
    num_steps = sequence_length - 1
    grid = DiscreteTimeGrid(
        T=num_steps * float(config["dt"]),
        num_steps=num_steps,
    )
    reference = fit_guyon_lekeufack_price_realized_volatility_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        ridge=1e-3,
        variance_ridge=1e-3,
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=1e-3,
        sigma_max=0.6,
        realized_volatility_window=20,
    )
    control = VolatilityOnlySpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=1.0,
        reference_kernel=reference,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    discrepancy = build_heston_discrepancy(
        num_steps=num_steps,
        include_acf=True,
        acf_lags=(1, 2, 5, 10),
        preset="old_fullv_w0p25",
        lambda_scale=50.0,
        kappa_scale=100.0,
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )
    estimator = CausalPathFeatureRidgeConditionalExpectation(
        grid=grid,
        ridge=1e-3,
        rolling_windows=(5, 10, 20, 32),
        ewma_half_lives=(5, 20, 60),
        downside_windows=(10, 32),
        drawdown_thresholds=(),
        drawdown_memory_half_lives=(),
        drawdown_memory_lags=(),
        drawdown_gate_temperature=0.005,
    )
    architecture_args = SimpleNamespace(
        hidden_dim=96,
        num_layers=1,
        adjoint_network="gru",
        transformer_layers=2,
        transformer_heads=4,
    )
    model = _build_model(
        args=architecture_args,
        grid=grid,
        control=control,
        discrepancy=discrepancy,
        ce_estimator=estimator,
        device=device,
        dtype=torch.float32,
        seed=int(args.seed),
    )
    checkpoint_path = (
        args.run_dir
        / "generic_ce_online_source_volatility_only"
        / "model_checkpoint.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_checkpoint_state(checkpoint)

    x0 = torch.zeros((1, 1), device=device, dtype=torch.float32)
    rollout_seed = derive_stream_seed(
        base_seed=int(args.seed),
        stream_offset=10_000,
    )
    with torch.no_grad():
        sample = model.sample(
            num_paths=int(args.bank_size),
            x0=x0,
            seed=rollout_seed,
        )
        sigma = sample.controls[..., 1].detach()
        sigma_reference = control._reference_sigmas_all(sample.paths).squeeze(-1)
        ratio = (sigma / sigma_reference).cpu().numpy().astype(np.float64)
        generated_prices = (
            float(config["s0"])
            * torch.exp(sample.paths[..., 0])
        ).cpu().numpy().astype(np.float64)

    saved_bank_path = (
        args.run_dir
        / "generic_ce_online_source_volatility_only"
        / f"generated_paths_{int(args.bank_size)}x{sequence_length}.npy"
    )
    saved_bank = np.asarray(np.load(saved_bank_path, allow_pickle=False), dtype=np.float64)
    if saved_bank.shape != generated_prices.shape:
        raise RuntimeError("replayed and saved path banks have different shapes")
    replay_max_abs_error = float(np.max(np.abs(saved_bank - generated_prices)))
    if replay_max_abs_error > 1e-3:
        raise RuntimeError(
            f"checkpoint replay does not match the saved bank: {replay_max_abs_error:.6g}"
        )

    cutoff = int(config["history_cutoff"])
    split = int(config["split_index"])
    response_start = cutoff + int(config["response_delay"])
    response_end = split + int(config["future_horizon"])
    early_hit = early_drawdown_group(
        generated_prices,
        cutoff=cutoff,
        threshold=float(config["drawdown_threshold"]),
    )
    groups = {
        "Threshold not crossed": ~early_hit,
        "Threshold crossed": early_hit,
    }
    curves = {name: quantile_curves(ratio[mask]) for name, mask in groups.items()}
    response_ratio = {
        name: ratio[mask, response_start:response_end].mean(axis=1)
        for name, mask in groups.items()
    }

    response_paths = {
        "Target": args.dataset_root / "disc.npy",
        "Reference": (
            REPO_ROOT
            / "runs"
            / "delayed_guyon_yweight0_n128_checkpoint_seed0_20260809"
            / "confirmation_selection"
            / "bank_0"
            / "reference_bank.npy"
        ),
        "SBTS": (
            REPO_ROOT
            / "runs"
            / "paper_synthetic_4seed_20260801"
            / "drawdown"
            / "sbts"
            / "seed_1"
            / "generated_paths_8192x128.npy"
        ),
    }
    response_banks = {
        "Target": load_prices(response_paths["Target"]),
        "Reference": load_prices(response_paths["Reference"]),
        "Deep MKV": generated_prices,
        "SBTS": load_prices(response_paths["SBTS"]),
    }
    rolling_window = 8
    response_curves = {
        name: delayed_response(
            bank,
            threshold=float(config["drawdown_threshold"]),
            history_cutoff=cutoff,
            dt=float(config["dt"]),
            rolling_window=rolling_window,
        )
        for name, bank in response_banks.items()
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.85),
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )
    response_time = np.arange(rolling_window, sequence_length, dtype=np.int64)
    for name in ("Target", "Reference", "Deep MKV", "SBTS"):
        values, standard_error = response_curves[name]
        axes[0].plot(
            response_time,
            values,
            color=RESPONSE_COLORS[name],
            linestyle=RESPONSE_LINESTYLES[name],
            linewidth=2.0 if name == "Target" else 1.6,
            label=name,
        )
        axes[0].fill_between(
            response_time,
            values - 1.96 * standard_error,
            values + 1.96 * standard_error,
            color=RESPONSE_COLORS[name],
            alpha=0.07,
            linewidth=0,
        )
    axes[0].axvline(cutoff, color="#777777", linewidth=0.9, linestyle=":")
    axes[0].axvline(response_start, color="#777777", linewidth=0.9, linestyle=":")
    axes[0].axvspan(
        response_start,
        response_end,
        color="#D6A84A",
        alpha=0.08,
        linewidth=0,
    )
    axes[0].annotate(
        "32-step delay",
        xy=(response_start, 0.018),
        xytext=(cutoff, 0.018),
        arrowprops={
            "arrowstyle": "<->",
            "color": "#666666",
            "linewidth": 0.8,
        },
        color="#555555",
        fontsize=7.2,
        ha="center",
        va="bottom",
    )
    axes[0].set_title("Delayed volatility response")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Rolling-volatility gap between groups")
    axes[0].set_xlim(rolling_window, sequence_length - 1)
    axes[0].set_ylim(-0.02, 0.32)
    axes[0].grid(alpha=0.16)
    axes[0].legend(loc="upper left", ncol=2, frameon=False, fontsize=7.2)

    time = np.arange(num_steps, dtype=np.int64)
    for name in ("Threshold not crossed", "Threshold crossed"):
        values = curves[name]
        color = COLORS[name]
        axes[1].fill_between(
            time,
            values["q10"],
            values["q90"],
            color=color,
            alpha=0.09,
            linewidth=0,
        )
        axes[1].fill_between(
            time,
            values["q25"],
            values["q75"],
            color=color,
            alpha=0.19,
            linewidth=0,
        )
        axes[1].plot(
            time,
            values["q50"],
            color=color,
            linewidth=1.8,
            label=(
                (
                    "4% drawdown not crossed"
                    if name == "Threshold not crossed"
                    else "4% drawdown crossed"
                )
                + f" by step 40 ($n={int(groups[name].sum())}$)"
            ),
        )
    axes[1].axhline(1.0, color="#222222", linewidth=0.9, linestyle="--")
    axes[1].axvline(cutoff, color="#777777", linewidth=0.9, linestyle=":")
    axes[1].axvline(response_start, color="#777777", linewidth=0.9, linestyle=":")
    axes[1].axvspan(0, cutoff, color="#777777", alpha=0.035, linewidth=0)
    axes[1].axvspan(
        response_start,
        response_end,
        color="#D6A84A",
        alpha=0.08,
        linewidth=0,
    )
    axes[1].set_ylim(0.28, 1.95)
    axes[1].text(
        cutoff / 2,
        1.91,
        "Earlier-path window",
        color="#555555",
        fontsize=7.4,
        ha="center",
        va="top",
    )
    axes[1].text(
        0.5 * (response_start + response_end),
        1.91,
        "Delayed-response window",
        color="#8A651F",
        fontsize=7.4,
        ha="center",
        va="top",
    )
    axes[1].annotate(
        "32-step delay",
        xy=(response_start, 1.76),
        xytext=(cutoff, 1.76),
        arrowprops={
            "arrowstyle": "<->",
            "color": "#666666",
            "linewidth": 0.8,
        },
        color="#555555",
        fontsize=7.4,
        ha="center",
        va="bottom",
    )
    axes[1].set_title("Volatility correction")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Volatility ratio (Deep MKV / reference)")
    axes[1].set_xlim(0, num_steps - 1)
    axes[1].grid(alpha=0.16)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(wspace=0.34, bottom=0.24)
    stem = args.output_dir / "delayed_response_and_sigma_ratio_seed1"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "scope": "validation_only",
        "test_split_loaded": False,
        "seed": int(args.seed),
        "checkpoint": str(checkpoint_path.resolve()),
        "saved_bank": str(saved_bank_path.resolve()),
        "bank_size": int(args.bank_size),
        "replay_max_abs_price_error": replay_max_abs_error,
        "ratio_definition": (
            "Deep-MKV sigma divided by reference sigma evaluated on the same "
            "Deep-MKV-generated prefix"
        ),
        "conditioning": {
            "history_cutoff": cutoff,
            "drawdown_threshold": float(config["drawdown_threshold"]),
            "early_drawdown_fraction": float(early_hit.mean()),
        },
        "windows": {
            "forecast_split": split,
            "delayed_response_start": response_start,
            "delayed_response_end_exclusive": response_end,
        },
        "overall_ratio": {
            "mean": float(ratio.mean()),
            "median": float(np.median(ratio)),
            "q10": float(np.quantile(ratio, 0.10)),
            "q90": float(np.quantile(ratio, 0.90)),
        },
        "response_window_pathwise_mean_ratio": {
            name: {
                "count": int(values.size),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }
            for name, values in response_ratio.items()
        },
        "outputs": {
            "pdf": str(stem.with_suffix(".pdf").resolve()),
            "png": str(stem.with_suffix(".png").resolve()),
        },
    }
    stem.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
