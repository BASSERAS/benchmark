#!/usr/bin/env python3
"""Calibrate external drawdown-risk rules on validation and score frozen test."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
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
    normalized_prices_to_log_paths,
    protocol_by_identifier,
    select_date_window,
)
from path_dt_experiments.risk_budgeting import (  # noqa: E402
    bootstrap_drawdown_risk_budget,
    calibrate_drawdown_safety_factor,
    evaluate_drawdown_risk_budget,
    paired_bootstrap_drawdown_difference,
)
from run_drawdown_risk_budgeting import (  # noqa: E402
    _config_from_protocol,
    _load_bank,
)


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "experiments"
    / "protocols"
    / "drawdown_risk_budgeting_external_test_v1.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _file_sha256(path)
    if actual != str(expected):
        raise RuntimeError(f"frozen input hash mismatch for {path}: {actual}")


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


def _load_validation_metrics(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "predicted_unlevered_drawdown_quantile",
            "realized_unlevered_drawdown",
        }
        if not required.issubset(payload.files):
            raise ValueError(f"{path} is missing calibration arrays")
        predicted = torch.from_numpy(
            np.asarray(payload["predicted_unlevered_drawdown_quantile"])
        )
        realized = torch.from_numpy(
            np.asarray(payload["realized_unlevered_drawdown"])
        )
    return predicted, realized


def _result_payload(
    result,
    *,
    bootstrap: dict[str, Any],
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "summary": result.summary,
        "bootstrap": bootstrap_drawdown_risk_budget(
            result,
            num_replicates=int(bootstrap["num_replicates"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=int(seed),
            block_length_sessions=int(block_length),
        ),
    }


def _save_path_artifacts(output: Path, result) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "selected_indices.npy", result.selected_indices.numpy())
    np.save(output / "prefix_distances.npy", result.prefix_distances.numpy())
    np.savez_compressed(
        output / "path_metrics.npz",
        **{name: values.numpy() for name, values in result.path_metrics.items()},
    )


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen drawdown-risk test",
        "",
        "Safety factors were calibrated on validation and frozen before test access.",
        "",
        "| Method | Safety factor | Test violations | Average leverage | Mean excess |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, payload in report["methods"].items():
        test = payload["test"]["summary"]
        lines.append(
            "| {name} | {factor:.3f} | {violation:.3f} | {leverage:.3f} | {excess:.6f} |".format(
                name=name,
                factor=float(payload["calibration"]["safety_factor"]),
                violation=float(test["observed_violation_rate"]),
                leverage=float(test["average_leverage"]),
                excess=float(test["mean_violation_excess_unconditional"]),
            )
        )
    lines.extend(
        [
            "",
            (
                "Crisis-window results are reported separately and were never "
                "used for recalibration."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    protocol_path = args.protocol_manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not bool(protocol.get("protected_test_access_authorized", False)):
        raise RuntimeError("the locked protocol does not authorize test access")
    external_only = bool(protocol.get("external_methods_only", False))
    ours_authorized = bool(protocol.get("ours_authorized", False))
    if external_only and ours_authorized:
        raise RuntimeError("an external-only protocol cannot authorize our models")
    if not external_only and not ours_authorized:
        raise RuntimeError(
            "a non-external protocol must explicitly authorize our frozen models"
        )
    methods = dict(protocol["methods"])
    if external_only:
        if set(methods) != {
            "sbts",
            "moving_block",
            "whole_session",
            "historical_neighbor",
        }:
            raise RuntimeError("unexpected external method set")
    elif len(methods) < 2:
        raise RuntimeError("the frozen comparison must contain at least two methods")

    run_dir = args.run_dir.resolve()
    report_path = run_dir / "report.json"
    if report_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"refusing to overwrite {report_path}")

    # Verify every frozen validation/calibration input before reading test data.
    validation_protocol_spec = protocol["validation_protocol"]
    validation_protocol_path = Path(validation_protocol_spec["path"])
    _require_hash(validation_protocol_path, validation_protocol_spec["sha256"])
    validation_protocol = json.loads(
        validation_protocol_path.read_text(encoding="utf-8")
    )
    base_config = _config_from_protocol(validation_protocol)
    validation_report_spec = protocol["calibration"]
    _require_hash(
        Path(validation_report_spec["validation_report"]),
        validation_report_spec["validation_report_sha256"],
    )
    calibrations = {}
    for name, specification in protocol["validation_path_metrics"].items():
        path = Path(specification["path"])
        _require_hash(path, specification["sha256"])
        predicted, realized = _load_validation_metrics(path)
        calibrations[name] = calibrate_drawdown_safety_factor(
            predicted_drawdown_quantile=predicted,
            realized_unlevered_drawdown=realized,
            coverage_level=float(protocol["calibration"]["coverage_level"]),
        )
    banks = {}
    bank_provenance = {}
    for name, specification in methods.items():
        path = Path(specification["bank_path"])
        _require_hash(path, specification["bank_sha256"])
        banks[name] = _load_bank(
            path,
            expected_points=128,
            top_k=int(base_config.top_k),
        )
        bank_provenance[name] = {
            **specification,
            "num_paths": int(banks[name].shape[0]),
        }

    # This is the first point at which the protected chronological test is read.
    test_spec = protocol["test_data"]
    test_prices_path = Path(test_spec["prices_path"])
    test_dates_path = Path(test_spec["dates_path"])
    _require_hash(test_prices_path, test_spec["prices_sha256"])
    _require_hash(test_dates_path, test_spec["dates_sha256"])
    test_prices = np.asarray(
        np.load(test_prices_path, allow_pickle=False),
        dtype=np.float64,
    )
    test_dates = np.asarray(
        np.load(test_dates_path, allow_pickle=False),
    ).astype(str)
    if test_prices.shape != (123, 128) or test_dates.shape != (123,):
        raise RuntimeError("protected ES test arrays violate the frozen shape")
    if not np.allclose(test_prices[:, 0], 100.0, rtol=0.0, atol=1e-5):
        raise RuntimeError("protected ES paths must start at normalized price 100")

    bootstrap = protocol["bootstrap"]
    method_payload = {}
    test_results = {}
    stress_protocol = protocol_by_identifier(
        str(protocol["crisis_stress_test"]["protocol"])
    )
    requested_stress = set(protocol["crisis_stress_test"]["windows"])
    available_stress = {
        window.name: window for window in stress_protocol.evaluation_windows
    }
    if requested_stress != set(available_stress):
        raise RuntimeError("the frozen crisis-window set is incomplete")

    for method_index, name in enumerate(methods):
        calibration = calibrations[name]
        config = replace(
            base_config,
            safety_factor=float(calibration.safety_factor),
        )
        print(
            f"testing {name} with frozen safety factor {calibration.safety_factor:.6f}",
            flush=True,
        )
        test_result = evaluate_drawdown_risk_budget(
            real_paths=normalized_prices_to_log_paths(test_prices),
            generated_paths=normalized_prices_to_log_paths(banks[name]),
            model_name=name,
            config=config,
        )
        test_results[name] = test_result
        method_root = run_dir / name
        _save_path_artifacts(method_root / "test", test_result)
        test_payload = _result_payload(
            test_result,
            bootstrap=bootstrap,
            block_length=int(bootstrap["session_block_length"]),
            seed=int(bootstrap["seed"]) + method_index,
        )
        _write_json(method_root / "test" / "metrics.json", test_payload)

        stress_payload = {}
        for window_index, (window_name, window) in enumerate(
            available_stress.items()
        ):
            indices = select_date_window(test_dates, window)
            window_result = evaluate_drawdown_risk_budget(
                real_paths=normalized_prices_to_log_paths(test_prices[indices]),
                generated_paths=normalized_prices_to_log_paths(banks[name]),
                model_name=name,
                config=config,
            )
            output = method_root / "crisis" / window_name
            _save_path_artifacts(output, window_result)
            payload = _result_payload(
                window_result,
                bootstrap=bootstrap,
                block_length=min(
                    int(bootstrap["session_block_length"]),
                    int(indices.size),
                ),
                seed=int(bootstrap["seed"]) + 1_000 + 10 * method_index + window_index,
            )
            payload["dates"] = {
                "first": str(test_dates[indices[0]]),
                "last": str(test_dates[indices[-1]]),
                "num_sessions": int(indices.size),
                "role": str(window.role),
            }
            _write_json(output / "metrics.json", payload)
            stress_payload[window_name] = payload

        method_payload[name] = {
            "bank": bank_provenance[name],
            "calibration": asdict(calibration),
            "test": test_payload,
            "crisis_stress_test": stress_payload,
        }

    primary_name = str(protocol["primary_method"])
    paired = {}
    for comparison_name, comparison in test_results.items():
        if comparison_name == primary_name:
            continue
        paired[comparison_name] = paired_bootstrap_drawdown_difference(
            test_results[primary_name],
            comparison,
            num_replicates=int(bootstrap["num_replicates"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=int(bootstrap["seed"]) + 20_000,
            block_length_sessions=int(bootstrap["session_block_length"]),
        )

    report = {
        "experiment": str(protocol["experiment"]),
        "external_methods_only": external_only,
        "ours_evaluated": ours_authorized,
        "test_access_authorized_by_locked_protocol": True,
        "calibration_used_validation_only": True,
        "test_used_for_selection_or_recalibration": False,
        "crisis_used_for_selection_or_recalibration": False,
        "test_dates": {
            "first": str(test_dates[0]),
            "last": str(test_dates[-1]),
            "num_sessions": int(test_dates.size),
        },
        "protocol_manifest": str(protocol_path),
        "protocol_manifest_sha256": _file_sha256(protocol_path),
        "base_config": asdict(base_config),
        "methods": method_payload,
        "paired_primary_method": primary_name,
        "paired_test_primary_minus_comparison": paired,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    summary = _summary_markdown(report)
    (run_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
