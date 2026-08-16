#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from path_dt_experiments.real_data_protocol import (  # noqa: E402
    FROZEN_REAL_DATA_PROTOCOLS,
    calendar_aligned_moving_block_bootstrap,
    frozen_protocol_manifest,
    normalized_prices_to_log_paths,
    protocol_by_identifier,
    real_data_law_metrics,
    select_date_window,
    training_indices,
    validate_protocol_on_calendar,
    whole_session_bootstrap,
)
from path_dt_experiments.shadowing import (  # noqa: E402
    ShadowingFeatureConfig,
    bootstrap_shadowing_metrics,
    evaluate_path_shadowing,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen real-data law and path-shadowing protocol on a "
            "scenario bank or on the nonparametric bootstrap baseline."
        )
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=REPO_ROOT / "data" / "ICAIF_3Y" / "canonical_rth_128",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs" / "real_data_protocol",
    )
    parser.add_argument(
        "--index",
        choices=("ES", "NQ", "RTY", "YM"),
        default="ES",
    )
    parser.add_argument(
        "--protocol",
        action="append",
        choices=tuple(
            protocol.identifier for protocol in FROZEN_REAL_DATA_PROTOCOLS
        ),
        help="May be repeated. The default runs all frozen protocols.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--generated-root",
        type=Path,
        help=(
            "Load <root>/<protocol>/<index>/seed_<seed>.npy normalized-price "
            "banks supplied by a model."
        ),
    )
    source.add_argument(
        "--baseline",
        choices=("moving_block", "whole_session"),
        default="moving_block",
    )
    parser.add_argument("--model-name", default="moving_block_bootstrap")
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2, 3, 4), default=0)
    parser.add_argument("--prefix-points", type=int, default=65)
    parser.add_argument("--future-increments", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--search-backend", choices=("auto", "exact", "faiss"), default="auto")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Evaluate only the frozen validation window. This is the only mode "
            "permitted for hyperparameter screening."
        ),
    )
    return parser.parse_args()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _bank_path(
    generated_root: Path,
    protocol: str,
    index: str,
    seed: int,
) -> Path:
    return generated_root / protocol / index / f"seed_{seed}.npy"


def _load_or_generate_bank(
    *,
    args: argparse.Namespace,
    protocol_identifier: str,
    training_prices: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    if args.generated_root is not None:
        path = _bank_path(
            args.generated_root,
            protocol_identifier,
            args.index,
            args.seed,
        )
        if not path.is_file():
            raise FileNotFoundError(
                "missing generated bank required by the frozen input contract: "
                f"{path}"
            )
        values = np.load(path, allow_pickle=False)
        source = {
            "kind": "external_model",
            "model_name": args.model_name,
            "path": path.resolve(),
        }
    elif args.baseline == "whole_session":
        values = whole_session_bootstrap(
            training_prices,
            bank_size=args.bank_size,
            seed=args.seed,
        )
        source = {
            "kind": "whole_session_bootstrap",
            "model_name": args.model_name,
        }
    else:
        values = calendar_aligned_moving_block_bootstrap(
            training_prices,
            bank_size=args.bank_size,
            block_length=args.block_length,
            seed=args.seed,
        )
        source = {
            "kind": "calendar_aligned_moving_block_bootstrap",
            "model_name": args.model_name,
            "block_length_grid_increments": int(args.block_length),
        }
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[2] == 1:
        values = values[:, :, 0]
    expected_points = int(training_prices.shape[1])
    if values.ndim != 2 or int(values.shape[1]) != expected_points:
        raise ValueError(
            f"generated bank must have shape (bank, {expected_points})"
        )
    if int(values.shape[0]) < int(args.top_k):
        raise ValueError("generated bank must contain at least top_k paths")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("generated normalized prices must be finite and positive")
    if not np.allclose(values[:, 0], 100.0, rtol=0.0, atol=1e-5):
        raise ValueError("every generated path must start at normalized price 100")
    return values.astype(np.float32), source


def _phase_metrics(
    *,
    generated_prices: np.ndarray,
    target_prices: np.ndarray,
    args: argparse.Namespace,
    bootstrap_seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    generated = normalized_prices_to_log_paths(generated_prices)
    target = normalized_prices_to_log_paths(target_prices)
    law_metrics = real_data_law_metrics(generated, target)
    shadowing = evaluate_path_shadowing(
        real_paths=target,
        generated_paths=generated,
        prefix_length=args.prefix_points,
        future_horizon=args.future_increments,
        top_k=args.top_k,
        feature_config=ShadowingFeatureConfig(),
        search_backend=args.search_backend,
    )
    shadow_bootstrap = bootstrap_shadowing_metrics(
        shadowing,
        num_replicates=args.bootstrap_replicates,
        confidence_level=0.95,
        seed=bootstrap_seed,
    )
    payload: dict[str, object] = {
        "target_session_count": int(target_prices.shape[0]),
        "law_metrics": law_metrics,
        "path_shadowing_metrics": shadowing.metrics,
        "path_shadowing_bootstrap": shadow_bootstrap,
    }
    per_path = {
        name: values.detach().cpu().numpy()
        for name, values in shadowing.path_metrics.items()
    }
    return payload, per_path


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    probabilities = np.arange(1, len(ordered) + 1, dtype=np.float64) / len(
        ordered
    )
    return ordered, probabilities


def _plot_phase_diagnostics(
    *,
    generated_prices: np.ndarray,
    target_prices: np.ndarray,
    phase_label: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generated = np.log(
        generated_prices.astype(np.float64) / generated_prices[:, :1]
    )
    target = np.log(target_prices.astype(np.float64) / target_prices[:, :1])
    generated_returns = np.diff(generated, axis=1)
    target_returns = np.diff(target, axis=1)
    generated_flat = generated_returns.reshape(-1)
    target_flat = target_returns.reshape(-1)
    generated_abs = np.abs(generated_flat)
    target_abs = np.abs(target_flat)
    generated_rv = np.sqrt(np.sum(generated_returns**2, axis=1))
    target_rv = np.sqrt(np.sum(target_returns**2, axis=1))
    generated_drawdown = np.max(
        np.maximum.accumulate(generated, axis=1) - generated,
        axis=1,
    )
    target_drawdown = np.max(
        np.maximum.accumulate(target, axis=1) - target,
        axis=1,
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.7))
    low = float(np.quantile(target_flat, 0.001))
    high = float(np.quantile(target_flat, 0.999))
    axes[0, 0].hist(
        target_flat,
        bins=80,
        range=(low, high),
        density=True,
        histtype="step",
        linewidth=1.8,
        color="black",
        label="real",
    )
    axes[0, 0].hist(
        generated_flat,
        bins=80,
        range=(low, high),
        density=True,
        histtype="step",
        linewidth=1.2,
        color="#2457a6",
        label="scenario bank",
    )
    axes[0, 0].set_title("Intraday log-return density")
    axes[0, 0].legend(frameon=False)

    quantiles = np.linspace(0.01, 0.99, 99)
    target_q = np.quantile(target_flat, quantiles)
    generated_q = np.quantile(generated_flat, quantiles)
    axes[0, 1].scatter(target_q, generated_q, s=9, color="#2457a6")
    qq_low = min(float(target_q.min()), float(generated_q.min()))
    qq_high = max(float(target_q.max()), float(generated_q.max()))
    axes[0, 1].plot(
        [qq_low, qq_high],
        [qq_low, qq_high],
        color="black",
        linewidth=1,
    )
    axes[0, 1].set_title("Return QQ")
    axes[0, 1].set_xlabel("real quantile")
    axes[0, 1].set_ylabel("scenario quantile")

    thresholds = np.quantile(target_abs, np.linspace(0.50, 0.995, 80))
    target_survival = np.asarray(
        [np.mean(target_abs > threshold) for threshold in thresholds]
    )
    generated_survival = np.asarray(
        [np.mean(generated_abs > threshold) for threshold in thresholds]
    )
    axes[0, 2].semilogy(
        thresholds,
        target_survival,
        color="black",
        linewidth=1.8,
        label="real",
    )
    axes[0, 2].semilogy(
        thresholds,
        generated_survival,
        color="#2457a6",
        linewidth=1.2,
        label="scenario bank",
    )
    axes[0, 2].set_title("Absolute-return tail")
    axes[0, 2].legend(frameon=False)

    lag_values = (1, 5, 10, 20)
    generated_tensor = normalized_prices_to_log_paths(generated_prices)
    target_tensor = normalized_prices_to_log_paths(target_prices)
    generated_return_tensor = torch.diff(generated_tensor, dim=1)
    target_return_tensor = torch.diff(target_tensor, dim=1)
    from path_dt_experiments.metrics import acf

    for values, label, color, width in (
        (target_return_tensor, "real", "black", 1.8),
        (generated_return_tensor, "scenario bank", "#2457a6", 1.2),
    ):
        abs_acf = acf(values.abs(), lags=lag_values)[:, 0].numpy()
        squared_acf = acf(values.pow(2), lags=lag_values)[:, 0].numpy()
        axes[1, 0].plot(
            lag_values,
            abs_acf,
            marker="o",
            color=color,
            linewidth=width,
            label=f"{label} |r|",
        )
        axes[1, 0].plot(
            lag_values,
            squared_acf,
            marker="s",
            linestyle="--",
            color=color,
            linewidth=width,
            label=f"{label} r²",
        )
    axes[1, 0].set_title("Volatility clustering")
    axes[1, 0].set_xlabel("intraday lag")
    axes[1, 0].legend(frameon=False, fontsize="small", ncol=2)

    for values, label, color, width in (
        (target_rv, "real", "black", 1.8),
        (generated_rv, "scenario bank", "#2457a6", 1.2),
    ):
        x, y = _ecdf(values)
        axes[1, 1].plot(x, y, color=color, linewidth=width, label=label)
    axes[1, 1].set_title("Daily realized-volatility CDF")
    axes[1, 1].legend(frameon=False)

    for values, label, color, width in (
        (target_drawdown, "real", "black", 1.8),
        (generated_drawdown, "scenario bank", "#2457a6", 1.2),
    ):
        x, y = _ecdf(values)
        axes[1, 2].plot(x, y, color=color, linewidth=width, label=label)
    axes[1, 2].set_title("Maximum-drawdown CDF")
    axes[1, 2].legend(frameon=False)

    figure.suptitle(phase_label)
    for axis in axes.reshape(-1):
        axis.grid(alpha=0.16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return None
    if abs(denominator) < 1e-15:
        return None
    return float(numerator / denominator)


def _stress_degradation(
    phases: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    normal_names = [
        name
        for name, phase in phases.items()
        if phase["window"]["role"] == "normal"
    ]
    stress_names = [
        name
        for name, phase in phases.items()
        if phase["window"]["role"] == "stress"
    ]
    if len(normal_names) != 1 or len(stress_names) != 1:
        return None
    normal_name = normal_names[0]
    stress_name = stress_names[0]
    normal = phases[normal_name]
    stress = phases[stress_name]
    normal_law = normal["law_metrics"]
    stress_law = stress["law_metrics"]
    normal_shadow = normal["path_shadowing_metrics"]
    stress_shadow = stress["path_shadowing_metrics"]
    lower_is_better_law = (
        "return_qq_rmse_normalized",
        "abs_return_acf_error_rms",
        "squared_return_acf_error_rms",
        "leverage_error_rms",
        "realized_volatility_wasserstein_normalized",
        "maximum_drawdown_wasserstein_normalized",
    )
    lower_is_better_shadow = (
        "cumulative_return_crps",
        "realized_volatility_crps",
        "realized_volatility_mean_rmse",
    )
    result: dict[str, object] = {
        "normal_phase": normal_name,
        "stress_phase": stress_name,
        "interpretation": (
            "Ratios above one mean that an error or proper score is worse in "
            "the stress window than in the matched-normal window."
        )
    }
    for name in lower_is_better_law:
        result[f"{name}_stress_to_normal_ratio"] = _safe_ratio(
            float(stress_law[name]),
            float(normal_law[name]),
        )
    for name in lower_is_better_shadow:
        result[f"{name}_stress_to_normal_ratio"] = _safe_ratio(
            float(stress_shadow[name]),
            float(normal_shadow[name]),
        )
    result["realized_volatility_coverage_90_stress_minus_normal"] = float(
        stress_shadow["realized_volatility_coverage_90"]
    ) - float(normal_shadow["realized_volatility_coverage_90"])
    return result


def main() -> None:
    args = _parse_args()
    if args.bank_size < args.top_k:
        raise ValueError("--bank-size must be at least --top-k")
    frozen = frozen_protocol_manifest()
    expected_bank_size = int(
        frozen["scenario_bank_contract"]["paths_per_seed"]
    )
    if int(args.bank_size) != expected_bank_size:
        raise ValueError(
            f"--bank-size must equal the frozen value {expected_bank_size}"
        )
    frozen_shadow = frozen["path_shadowing"]
    requested_shadow = {
        "prefix_points": int(args.prefix_points),
        "future_increments": int(args.future_increments),
        "top_k": int(args.top_k),
        "bootstrap_replicates": int(args.bootstrap_replicates),
    }
    expected_shadow = {
        key: int(frozen_shadow[key]) for key in requested_shadow
    }
    if requested_shadow != expected_shadow:
        raise ValueError(
            "shadowing arguments differ from the frozen protocol: "
            f"requested={requested_shadow}, expected={expected_shadow}"
        )

    canonical_root = args.canonical_root.resolve()
    prices = np.load(
        canonical_root / args.index / "prices_all.npy",
        allow_pickle=False,
    )
    dates = np.load(
        canonical_root / args.index / "dates_all.npy",
        allow_pickle=False,
    ).astype(str).tolist()
    protocol_ids = args.protocol or [
        protocol.identifier for protocol in FROZEN_REAL_DATA_PROTOCOLS
    ]
    run_root = (
        args.output_root.resolve()
        / args.model_name
        / args.index
        / f"seed_{args.seed}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "frozen_protocol.json", frozen)

    summary: dict[str, object] = {
        "model_name": args.model_name,
        "index": args.index,
        "seed": int(args.seed),
        "canonical_root": canonical_root,
        "validation_only": bool(args.validation_only),
        "protocols": {},
    }
    for protocol_offset, protocol_id in enumerate(protocol_ids):
        protocol = protocol_by_identifier(protocol_id)
        counts = validate_protocol_on_calendar(dates, protocol)
        train_positions = training_indices(dates, protocol)
        training_prices = prices[train_positions]
        generated, source = _load_or_generate_bank(
            args=args,
            protocol_identifier=protocol.scenario_bank_identifier,
            training_prices=training_prices,
        )
        source["evaluation_protocol"] = protocol.identifier
        source["scenario_bank_identifier"] = (
            protocol.scenario_bank_identifier
        )
        if int(generated.shape[0]) != expected_bank_size:
            raise ValueError(
                "generated bank differs from the frozen path count: "
                f"{generated.shape[0]} != {expected_bank_size}"
            )
        protocol_root = run_root / protocol.identifier
        protocol_root.mkdir(parents=True, exist_ok=True)
        bank_path = protocol_root / "generated_bank.npy"
        np.save(bank_path, generated, allow_pickle=False)
        windows = (
            (protocol.validation,)
            if bool(args.validation_only)
            else (protocol.validation, *protocol.evaluation_windows)
        )
        phases: dict[str, dict[str, object]] = {}
        for phase_offset, window in enumerate(windows):
            positions = select_date_window(dates, window)
            phase, per_path = _phase_metrics(
                generated_prices=generated,
                target_prices=prices[positions],
                args=args,
                bootstrap_seed=(
                    int(args.seed)
                    + 10_000 * protocol_offset
                    + 100 * phase_offset
                ),
            )
            phase["window"] = {
                **asdict(window),
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
            }
            phase["target_dates"] = [dates[int(position)] for position in positions]
            phases[window.name] = phase
            np.savez_compressed(
                protocol_root / f"{window.name}_per_path_metrics.npz",
                **per_path,
            )
            _plot_phase_diagnostics(
                generated_prices=generated,
                target_prices=prices[positions],
                phase_label=(
                    f"{args.model_name}: {protocol.label} / {window.name}"
                ),
                output_path=protocol_root / f"{window.name}_diagnostics.pdf",
            )
            print(
                f"{protocol.identifier}/{window.name}: "
                f"n={len(positions)} "
                "RV-CRPS="
                f"{phase['path_shadowing_metrics']['realized_volatility_crps']:.6f} "
                "RV-coverage90="
                f"{phase['path_shadowing_metrics']['realized_volatility_coverage_90']:.3f}",
                flush=True,
            )
        protocol_payload = {
            "label": protocol.label,
            "calendar_counts": counts,
            "training_first_date": dates[int(train_positions[0])],
            "training_last_date": dates[int(train_positions[-1])],
            "training_session_count": int(train_positions.size),
            "generated_bank_size": int(generated.shape[0]),
            "scenario_source": source,
            "phases": phases,
            "stress_degradation_vs_matched_normal": _stress_degradation(phases),
        }
        summary["protocols"][protocol.identifier] = protocol_payload
        _write_json(protocol_root / "metrics.json", protocol_payload)
    _write_json(run_root / "summary.json", summary)
    print(f"Complete: {run_root}", flush=True)


if __name__ == "__main__":
    main()
