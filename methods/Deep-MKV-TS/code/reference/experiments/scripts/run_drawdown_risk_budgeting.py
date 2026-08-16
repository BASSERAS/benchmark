#!/usr/bin/env python3
"""Evaluate scenario banks as drawdown-constrained ES position-sizing engines."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import re
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from path_dt_experiments.real_data_generation import (  # noqa: E402
    load_frozen_real_data_fit,
)
from path_dt_experiments.real_data_protocol import (  # noqa: E402
    normalized_prices_to_log_paths,
)
from path_dt_experiments.risk_budgeting import (  # noqa: E402
    DrawdownRiskBudgetConfig,
    bootstrap_drawdown_risk_budget,
    drawdown_policy_metrics,
    evaluate_drawdown_risk_budget,
    paired_bootstrap_drawdown_difference,
    summarize_drawdown_policy,
)
from path_dt_experiments.shadowing import ShadowingFeatureConfig  # noqa: E402


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "experiments"
    / "protocols"
    / "drawdown_risk_budgeting_validation_v1.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--bank",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat for every reference/model scenario bank to compare.",
    )
    parser.add_argument(
        "--include-historical-neighbor",
        action="store_true",
        help="Also treat the training sessions as a historical-neighbor bank.",
    )
    parser.add_argument(
        "--primary-method",
        help="Method used as primary-minus-comparison in paired intervals.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
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


def _parse_banks(specifications: list[str]) -> dict[str, Path]:
    banks: dict[str, Path] = {}
    for specification in specifications:
        name, separator, raw_path = str(specification).partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("every --bank must have the form SAFE_NAME=PATH")
        if name in banks:
            raise ValueError(f"duplicate bank name: {name}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        banks[name] = path
    return banks


def _load_bank(path: Path, *, expected_points: int, top_k: int) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False))
    if values.ndim == 3 and int(values.shape[2]) == 1:
        values = values[:, :, 0]
    if values.ndim != 2 or int(values.shape[1]) != int(expected_points):
        raise ValueError(
            f"{path} must have shape (bank, {expected_points})"
        )
    if int(values.shape[0]) < int(top_k):
        raise ValueError(f"{path} contains fewer than top_k paths")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(f"{path} must contain finite positive prices")
    if not np.allclose(values[:, 0], 100.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"{path} paths must start at normalized price 100")
    return values.astype(np.float64, copy=False)


def _config_from_protocol(protocol: dict[str, Any]) -> DrawdownRiskBudgetConfig:
    decision = protocol["decision_rule"]
    shadow = protocol["prefix_matching"]
    return DrawdownRiskBudgetConfig(
        prefix_length=int(shadow["prefix_length"]),
        future_horizon=int(shadow["future_horizon"]),
        top_k=int(shadow["top_k"]),
        drawdown_quantile=float(decision["drawdown_quantile"]),
        risk_budget=float(decision["risk_budget"]),
        safety_factor=float(decision["safety_factor"]),
        max_leverage=float(decision["max_leverage"]),
        min_leverage=float(decision["min_leverage"]),
        feature_config=ShadowingFeatureConfig(
            last_returns=int(shadow["features"]["last_returns"]),
            max_path_points=int(shadow["features"]["max_path_points"]),
            rolling_windows=tuple(
                int(value) for value in shadow["features"]["rolling_windows"]
            ),
            acf_lags=tuple(int(value) for value in shadow["features"]["acf_lags"]),
            return_weight=float(shadow["features"]["return_weight"]),
            path_weight=float(shadow["features"]["path_weight"]),
            vol_weight=float(shadow["features"]["vol_weight"]),
            acf_weight=float(shadow["features"]["acf_weight"]),
        ),
        search_backend=str(shadow["search_backend"]),
        exact_batch_size=int(shadow["exact_batch_size"]),
    )


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("selection_scope") != "validation_only":
        raise RuntimeError("the protocol must be validation-only")
    if bool(protocol.get("protected_test_or_event_access_authorized", True)):
        raise RuntimeError("the protocol must forbid protected test/event access")
    if protocol.get("economic_interpretation") != "constant_futures_notional":
        raise RuntimeError("unexpected portfolio P&L convention")
    if protocol.get("index") != "ES":
        raise RuntimeError("the locked first experiment is restricted to ES")


def _frontier(
    result,
    *,
    safety_factors: list[float],
) -> dict[str, dict[str, float]]:
    paths = result.path_metrics
    output = {}
    for safety_factor in safety_factors:
        metrics = drawdown_policy_metrics(
            predicted_drawdown_quantile=paths[
                "predicted_unlevered_drawdown_quantile"
            ],
            realized_unlevered_drawdown=paths["realized_unlevered_drawdown"],
            realized_unlevered_terminal_return=paths[
                "realized_unlevered_terminal_return"
            ],
            risk_budget=float(result.config.risk_budget),
            max_leverage=float(result.config.max_leverage),
            min_leverage=float(result.config.min_leverage),
            safety_factor=float(safety_factor),
        )
        output[f"{float(safety_factor):.8g}"] = summarize_drawdown_policy(
            metrics,
            drawdown_quantile=float(result.config.drawdown_quantile),
            risk_budget=float(result.config.risk_budget),
            max_leverage=float(result.config.max_leverage),
            safety_factor=float(safety_factor),
        )
    return output


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ES drawdown-constrained position sizing",
        "",
        "Validation-only diagnostic. No protected test or event split was loaded.",
        "",
        "| Method | Violation rate | Average leverage | Mean excess | Cap fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, payload in report["methods"].items():
        summary = payload["summary"]
        lines.append(
            "| {name} | {violation:.3f} | {leverage:.3f} | {excess:.6f} | {cap:.3f} |".format(
                name=name,
                violation=float(summary["observed_violation_rate"]),
                leverage=float(summary["average_leverage"]),
                excess=float(summary["mean_violation_excess_unconditional"]),
                cap=float(summary["upper_leverage_cap_fraction"]),
            )
        )
    lines.extend(
        [
            "",
            "A model is economically preferable when it supports more average exposure without a material increase in violation frequency or severity.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    protocol_path = args.protocol_manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol(protocol)
    config = _config_from_protocol(protocol)
    banks = _parse_banks(list(args.bank))
    if not banks and not bool(args.include_historical_neighbor):
        raise ValueError("provide at least one --bank or the historical baseline")
    if bool(args.include_historical_neighbor) and "historical_neighbor" in banks:
        raise ValueError("historical_neighbor is a reserved method name")

    run_dir = args.run_dir.resolve()
    report_path = run_dir / "report.json"
    if report_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"refusing to overwrite {report_path}")
    run_dir.mkdir(parents=True, exist_ok=True)

    fit = load_frozen_real_data_fit(
        Path(protocol["canonical_root"]),
        protocol_identifier=str(protocol["scenario_bank_protocol"]),
        index=str(protocol["index"]),
    )
    real_paths = normalized_prices_to_log_paths(fit.validation_prices)
    method_banks: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for name, path in banks.items():
        values = _load_bank(
            path,
            expected_points=int(fit.validation_prices.shape[1]),
            top_k=int(config.top_k),
        )
        method_banks[name] = (
            values,
            {
                "kind": "generated_scenario_bank",
                "path": str(path),
                "sha256": _file_sha256(path),
                "num_paths": int(values.shape[0]),
            },
        )
    if bool(args.include_historical_neighbor):
        if int(fit.train_prices.shape[0]) < int(config.top_k):
            raise ValueError("training split is smaller than top_k")
        method_banks["historical_neighbor"] = (
            fit.train_prices.astype(np.float64),
            {
                "kind": "training_session_neighbor_bank",
                "train_array_sha256": fit.train_sha256,
                "num_paths": int(fit.train_prices.shape[0]),
            },
        )

    primary = args.primary_method
    if primary is not None and str(primary) not in method_banks:
        raise ValueError("--primary-method must name one supplied bank")
    bootstrap = protocol["bootstrap"]
    frontier_factors = [
        float(value) for value in protocol["safety_factor_frontier"]
    ]
    results = {}
    method_payload = {}
    for method_index, (name, (prices, source)) in enumerate(method_banks.items()):
        print(f"evaluating {name} with {prices.shape[0]:,} scenario paths", flush=True)
        result = evaluate_drawdown_risk_budget(
            real_paths=real_paths,
            generated_paths=normalized_prices_to_log_paths(prices),
            model_name=name,
            config=config,
        )
        results[name] = result
        confidence = bootstrap_drawdown_risk_budget(
            result,
            num_replicates=int(bootstrap["num_replicates"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=int(bootstrap["seed"]) + method_index,
        )
        method_dir = run_dir / name
        method_dir.mkdir(parents=True, exist_ok=True)
        np.save(method_dir / "selected_indices.npy", result.selected_indices.numpy())
        np.save(method_dir / "prefix_distances.npy", result.prefix_distances.numpy())
        np.savez_compressed(
            method_dir / "path_metrics.npz",
            **{key: value.numpy() for key, value in result.path_metrics.items()},
        )
        payload = {
            "source": source,
            "summary": result.summary,
            "bootstrap": confidence,
            "safety_factor_frontier": _frontier(
                result,
                safety_factors=frontier_factors,
            ),
        }
        _write_json(method_dir / "metrics.json", payload)
        method_payload[name] = payload

    paired = {}
    if primary is not None:
        for comparison_name, comparison in results.items():
            if comparison_name == str(primary):
                continue
            paired[comparison_name] = paired_bootstrap_drawdown_difference(
                results[str(primary)],
                comparison,
                num_replicates=int(bootstrap["num_replicates"]),
                confidence_level=float(bootstrap["confidence_level"]),
                seed=int(bootstrap["seed"]) + 10_000,
            )

    report = {
        "experiment": str(protocol["experiment"]),
        "selection_scope": "validation_only",
        "protected_test_or_event_data_seen": False,
        "index": fit.index,
        "scenario_bank_protocol": fit.scenario_bank_protocol,
        "validation_sessions": int(fit.validation_prices.shape[0]),
        "validation_first_date": str(fit.validation_dates[0]),
        "validation_last_date": str(fit.validation_dates[-1]),
        "training_array_sha256": fit.train_sha256,
        "protocol_manifest": str(protocol_path),
        "protocol_manifest_sha256": _file_sha256(protocol_path),
        "config": asdict(config),
        "methods": method_payload,
        "paired_primary_method": primary,
        "paired_primary_minus_comparison": paired,
        "interpretation": {
            "success_rule": protocol["success_rule"],
            "drawdown": protocol["drawdown_definition"],
            "not_derivative_pricing": True,
            "not_exchange_margin_estimation": True,
        },
    }
    _write_json(report_path, report)
    (run_dir / "SUMMARY.md").write_text(
        _summary_markdown(report),
        encoding="utf-8",
    )
    print(_summary_markdown(report), flush=True)


if __name__ == "__main__":
    main()
