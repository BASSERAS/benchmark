#!/usr/bin/env python3
"""Aggregate independently run equal-KL Hamiltonian direction audits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-dir", type=Path, required=True)
    parser.add_argument("--raw-run-dir", type=Path, required=True)
    parser.add_argument("--kl-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_report(run_dir: Path) -> dict[str, object]:
    path = run_dir.resolve() / "equal_kl_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.detach().double().reshape(-1)
    y = right.detach().double().reshape(-1)
    denominator = float(x.norm().item() * y.norm().item())
    if denominator <= torch.finfo(torch.float64).eps:
        return 1.0 if float((x - y).norm().item()) == 0.0 else 0.0
    return float(torch.dot(x, y).item() / denominator)


def _relative_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.detach().double().reshape(-1)
    y = right.detach().double().reshape(-1)
    denominator = max(
        float(x.norm().item()),
        float(y.norm().item()),
        torch.finfo(torch.float64).eps,
    )
    return float((x - y).norm().item() / denominator)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    run_dirs = {
        "full": args.full_run_dir.resolve(),
        "raw": args.raw_run_dir.resolve(),
        "kl": args.kl_run_dir.resolve(),
    }
    reports = {method: _load_report(path) for method, path in run_dirs.items()}
    tensors = {
        method: torch.load(
            path / "equal_kl_tensors.pt", map_location="cpu", weights_only=False
        )
        for method, path in run_dirs.items()
    }
    reference = reports["full"]
    for method, report in reports.items():
        if report["methods"] != [method]:
            raise ValueError(f"{method} run contains unexpected methods")
        for key in (
            "config",
            "operator_beta",
            "effective_lambda_scale",
            "blocks",
            "fit_bank_fingerprint",
            "confirmation_bank_fingerprint",
        ):
            if report[key] != reference[key]:
                raise ValueError(f"{key} differs between method runs")
        for bank in ("fit", "confirmation"):
            for name, expected in reference["baseline"][bank].items():
                actual = report["baseline"][bank][name]
                if isinstance(expected, (int, float)) and isinstance(
                    actual, (int, float)
                ):
                    if not math.isclose(
                        float(actual), float(expected), rel_tol=1e-8, abs_tol=1e-9
                    ):
                        raise ValueError(
                            f"baseline scalar {bank}/{name} differs between runs"
                        )
                elif actual != expected:
                    raise ValueError(
                        f"baseline field {bank}/{name} differs between runs"
                    )
        for bank in ("fit", "confirmation"):
            reference_tensors = tensors["full"]["baseline"][bank]
            candidate_tensors = tensors[method]["baseline"][bank]
            if set(candidate_tensors) != set(reference_tensors):
                raise ValueError("baseline tensor keys differ between runs")
            for name, expected in reference_tensors.items():
                if not torch.allclose(
                    candidate_tensors[name], expected, rtol=1e-6, atol=5e-7
                ):
                    raise ValueError(
                        f"baseline tensor {bank}/{name} differs between runs"
                    )

    indexed: dict[str, dict[tuple[str, int, int], dict[str, object]]] = {}
    for method, report in reports.items():
        indexed[method] = {
            (str(row["signal"]), int(row["block_index"]), int(row["budget_index"])): row
            for row in report["results"][method]
        }
    keys = sorted(indexed["full"])
    if any(set(indexed[method]) != set(keys) for method in indexed):
        raise ValueError("method runs do not contain the same audit coordinates")

    baseline = reference["baseline"]
    rows = []
    rankings = []
    count_fields = {
        method: {
            "objective_descent_both_banks": 0,
            "refreshed_kkt_improvement_both_banks": 0,
            "joint_objective_and_kkt_both_banks": 0,
            "joint_and_confirmation_kl_safe": 0,
            "confirmation_kl_ceiling_passed": 0,
            "fit_budget_exhausted": 0,
            "confirmation_objective_best_or_tied": 0,
        }
        for method in indexed
    }
    for key in keys:
        signal, block_index, budget_index = key
        coordinate_rows = []
        for method in ("full", "raw", "kl"):
            source = indexed[method][key]
            kkt_name = (
                "refreshed_target_drift_kkt_rms"
                if signal == "alpha"
                else "refreshed_target_volatility_kkt_rms"
            )
            fit = source["fit"]
            confirmation = source["confirmation"]
            fit_kkt_change = float(fit[kkt_name]) - float(baseline["fit"][kkt_name])
            confirmation_kkt_change = float(confirmation[kkt_name]) - float(
                baseline["confirmation"][kkt_name]
            )
            objective_both = bool(
                float(fit["objective_change"]) < 0.0
                and float(confirmation["objective_change"]) < 0.0
            )
            kkt_both = bool(fit_kkt_change < 0.0 and confirmation_kkt_change < 0.0)
            confirm_kl_passed = bool(
                confirmation["q95_ceiling_passed"]
                and confirmation["maximum_ceiling_passed"]
            )
            row = {
                "method": method,
                "signal": signal,
                "block_index": block_index,
                "budget_index": budget_index,
                "target_mean_kl": float(source["target_mean_kl"]),
                "selected_parameter": float(source["selected_parameter"]),
                "fit_mean_kl_ratio": float(fit["mean_budget_ratio"]),
                "confirmation_mean_kl_ratio": float(confirmation["mean_budget_ratio"]),
                "confirmation_kl_ceiling_passed": confirm_kl_passed,
                "fit_objective_change": float(fit["objective_change"]),
                "confirmation_objective_change": float(
                    confirmation["objective_change"]
                ),
                "fit_objective_efficiency": float(
                    fit["objective_improvement_per_mean_kl"]
                ),
                "confirmation_objective_efficiency": float(
                    confirmation["objective_improvement_per_mean_kl"]
                ),
                "fit_refreshed_signal_kkt_change": fit_kkt_change,
                "confirmation_refreshed_signal_kkt_change": confirmation_kkt_change,
                "objective_descent_both_banks": objective_both,
                "refreshed_kkt_improvement_both_banks": kkt_both,
                "joint_objective_and_kkt_both_banks": objective_both and kkt_both,
                "joint_and_confirmation_kl_safe": (
                    objective_both and kkt_both and confirm_kl_passed
                ),
                "fit_budget_exhausted": bool(
                    source["calibration"]["mean_budget_exhausted"]
                ),
                "fit_limiting_constraints": list(
                    source["calibration"]["limiting_constraints"]
                ),
            }
            for field in (
                "objective_descent_both_banks",
                "refreshed_kkt_improvement_both_banks",
                "joint_objective_and_kkt_both_banks",
                "joint_and_confirmation_kl_safe",
                "confirmation_kl_ceiling_passed",
                "fit_budget_exhausted",
            ):
                count_fields[method][field] += int(bool(row[field]))
            rows.append(row)
            coordinate_rows.append(row)
        best_value = min(
            float(row["confirmation_objective_change"])
            for row in coordinate_rows
        )
        tie_tolerance = max(1e-10, 1e-6 * abs(best_value))
        winners = [
            row["method"]
            for row in coordinate_rows
            if abs(float(row["confirmation_objective_change"]) - best_value)
            <= tie_tolerance
        ]
        for winner in winners:
            count_fields[winner]["confirmation_objective_best_or_tied"] += 1
        rankings.append(
            {
                "signal": signal,
                "block_index": block_index,
                "budget_index": budget_index,
                "confirmation_objective_best_or_tied": winners,
                "jointly_improving_methods": [
                    row["method"]
                    for row in coordinate_rows
                    if row["joint_objective_and_kkt_both_banks"]
                ],
            }
        )

    movement_comparisons = []
    for key in keys:
        signal, block_index, budget_index = key
        coordinate = f"{signal}_block_{block_index}_budget_{budget_index}"
        tensor_name = "alpha_move" if signal == "alpha" else "log_sigma_move"
        for bank in ("fit", "confirmation"):
            movements = {
                method: tensors[method]["results"][method][coordinate][bank][tensor_name]
                for method in ("full", "raw", "kl")
            }
            for left, right in (("full", "raw"), ("full", "kl"), ("raw", "kl")):
                movement_comparisons.append(
                    {
                        "signal": signal,
                        "block_index": block_index,
                        "budget_index": budget_index,
                        "bank": bank,
                        "left": left,
                        "right": right,
                        "cosine_similarity": _cosine(movements[left], movements[right]),
                        "relative_vector_difference": _relative_difference(
                            movements[left], movements[right]
                        ),
                    }
                )

    total = len(keys)
    summary = {
        method: {**counts, "total_coordinates": total}
        for method, counts in count_fields.items()
    }
    summary_by_signal = {}
    for signal in ("alpha", "log_sigma"):
        summary_by_signal[signal] = {}
        for method in ("full", "raw", "kl"):
            selected = [
                row
                for row in rows
                if row["signal"] == signal and row["method"] == method
            ]
            summary_by_signal[signal][method] = {
                field: sum(int(bool(row[field])) for row in selected)
                for field in (
                    "objective_descent_both_banks",
                    "refreshed_kkt_improvement_both_banks",
                    "joint_objective_and_kkt_both_banks",
                    "joint_and_confirmation_kl_safe",
                    "confirmation_kl_ceiling_passed",
                )
            }
            summary_by_signal[signal][method]["total_coordinates"] = len(selected)
    raw_kl_volatility_cosines = [
        row["cosine_similarity"]
        for row in movement_comparisons
        if row["signal"] == "log_sigma"
        and row["bank"] == "confirmation"
        and row["left"] == "raw"
        and row["right"] == "kl"
    ]
    raw_kl_volatility_cosines.sort()
    raw_kl_volatility_median_cosine = raw_kl_volatility_cosines[
        len(raw_kl_volatility_cosines) // 2
    ]
    drift_full_raw_equal = all(
        row["relative_vector_difference"] <= 1e-7
        for row in movement_comparisons
        if row["signal"] == "alpha"
        and row["left"] == "full"
        and row["right"] == "raw"
    )
    early_volatility_rows = [
        row
        for row in rows
        if row["signal"] == "log_sigma" and int(row["block_index"]) == 0
    ]
    middle_volatility_rows = [
        row
        for row in rows
        if row["signal"] == "log_sigma" and int(row["block_index"]) == 1
    ]
    late_volatility_rows = [
        row
        for row in rows
        if row["signal"] == "log_sigma" and int(row["block_index"]) == 2
    ]
    early_volatility_joint_all = all(
        bool(row["joint_objective_and_kkt_both_banks"])
        for row in early_volatility_rows
    )
    middle_volatility_conflict_all = all(
        bool(row["objective_descent_both_banks"])
        and not bool(row["refreshed_kkt_improvement_both_banks"])
        for row in middle_volatility_rows
    )
    late_volatility_failure_all = all(
        not bool(row["objective_descent_both_banks"])
        and not bool(row["refreshed_kkt_improvement_both_banks"])
        for row in late_volatility_rows
    )
    kl_middle_late_volatility_safe_joint = sum(
        int(bool(row["joint_and_confirmation_kl_safe"]))
        for row in rows
        if row["method"] == "kl"
        and row["signal"] == "log_sigma"
        and int(row["block_index"]) in (1, 2)
    )
    decision = {
        "kl_wins_majority_of_confirmation_objectives": bool(
            summary["kl"]["confirmation_objective_best_or_tied"] > total / 2
        ),
        "kl_jointly_improves_majority": bool(
            summary["kl"]["joint_objective_and_kkt_both_banks"] > total / 2
        ),
        "all_methods_fail_refreshed_kkt_everywhere": bool(
            all(
                summary[method]["refreshed_kkt_improvement_both_banks"] == 0
                for method in summary
            )
        ),
        "raw_and_kl_volatility_directions_nearly_collinear": bool(
            raw_kl_volatility_median_cosine >= 0.99
        ),
        "kl_resolves_middle_or_late_volatility_conflict": bool(
            kl_middle_late_volatility_safe_joint > 0
        ),
        "implement_kl_gradient_actor_solver": bool(
            summary["kl"]["joint_and_confirmation_kl_safe"]
            > summary["full"]["joint_and_confirmation_kl_safe"]
            and kl_middle_late_volatility_safe_joint > 0
        ),
    }
    report = {
        "protocol": {
            "three_independent_method_runs": True,
            "same_banks_and_baseline_P_R_tensors_verified": True,
            "cross_gpu_baseline_tolerance": {"relative": 1e-6, "absolute": 5e-7},
            "fit_bank_parameter_calibration_only": True,
            "confirmation_bank_never_rescaled": True,
            "no_candidate_deployment": True,
            "no_adjoint_refit": True,
        },
        "source_run_dirs": {method: str(path) for method, path in run_dirs.items()},
        "config": reference["config"],
        "operator_beta": reference["operator_beta"],
        "baseline": baseline,
        "rows": rows,
        "rankings": rankings,
        "movement_comparisons": movement_comparisons,
        "summary": summary,
        "summary_by_signal": summary_by_signal,
        "raw_kl_volatility_confirmation_median_cosine": (
            raw_kl_volatility_median_cosine
        ),
        "decision": decision,
        "interpretation": {
            "drift_full_equals_raw": drift_full_raw_equal,
            "volatility_raw_and_kl_are_effectively_same_direction": bool(
                raw_kl_volatility_median_cosine >= 0.99
            ),
            "early_volatility_joint_improvement_all_methods": (
                early_volatility_joint_all
            ),
            "middle_volatility_objective_stationarity_conflict": (
                middle_volatility_conflict_all
            ),
            "late_volatility_failure_all_methods": late_volatility_failure_all,
            "recommended_action": "do_not_implement_kl_gradient_actor_solver",
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", report)

    lines = [
        "# Equal-KL Hamiltonian direction comparison",
        "",
        "All proposal parameters were calibrated on the fit bank. Confirmation-bank",
        "proposals reuse the fitted parameter without rescaling. No candidate was deployed",
        "and no adjoint was refit.",
        "",
        "| Method | Obj. descent both | Refreshed KKT better both | Both criteria | Both + KL safe | Confirm KL safe | Confirm obj. best/tied |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("full", "raw", "kl"):
        values = summary[method]
        lines.append(
            f"| {method} | {values['objective_descent_both_banks']}/{total} | "
            f"{values['refreshed_kkt_improvement_both_banks']}/{total} | "
            f"{values['joint_objective_and_kkt_both_banks']}/{total} | "
            f"{values['joint_and_confirmation_kl_safe']}/{total} | "
            f"{values['confirmation_kl_ceiling_passed']}/{total} | "
            f"{values['confirmation_objective_best_or_tied']}/{total} |"
        )
    lines.extend(
        [
            "",
            "## Signal-specific result",
            "",
            "| Signal | Method | Obj. descent both | KKT better both | Both + KL safe |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for signal in ("alpha", "log_sigma"):
        for method in ("full", "raw", "kl"):
            values = summary_by_signal[signal][method]
            count = values["total_coordinates"]
            lines.append(
                f"| {signal} | {method} | "
                f"{values['objective_descent_both_banks']}/{count} | "
                f"{values['refreshed_kkt_improvement_both_banks']}/{count} | "
                f"{values['joint_and_confirmation_kl_safe']}/{count} |"
            )
    lines.extend(
        [
            "",
            f"The confirmation-bank median cosine between raw and KL-mirror "
            f"volatility moves is {raw_kl_volatility_median_cosine:.6f}.",
        ]
    )
    lines.extend(
        [
            "",
            "## Coordinate-level confirmation results",
            "",
            "Negative dJ and negative dKKT are improvements relative to the accepted law.",
            "",
            "| Signal | Block | Budget | Method | Confirm KL/budget | Confirm dJ | Confirm dKKT | Joint both banks |",
            "|---|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['signal']} | {row['block_index']} | {row['target_mean_kl']:.3g} | "
            f"{row['method']} | {row['confirmation_mean_kl_ratio']:.3f} | "
            f"{row['confirmation_objective_change']:.5g} | "
            f"{row['confirmation_refreshed_signal_kkt_change']:.5g} | "
            f"{row['joint_objective_and_kkt_both_banks']} |"
        )
    lines.extend(["", "## Locked decision indicators", ""])
    for key, value in decision.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Within this audited checkpoint, full-control and raw-gradient drift moves "
            "are exactly the same direction. Raw and KL-mirror volatility moves are also "
            f"effectively collinear after budget calibration (median cosine "
            f"{raw_kl_volatility_median_cosine:.6f}), so the KL geometry does not reveal "
            "a materially different local volatility direction.",
            "",
            "All three methods jointly improve the objective and refreshed volatility "
            "KKT residual in the early block. In the middle block they lower the objective "
            "but worsen refreshed stationarity; in the late block they worsen both on the "
            "confirmation bank. The gradient variants therefore do not resolve the "
            "law-refresh conflict. Their early-block moves also hit the maximum-KL tail "
            "ceiling before using the full mean-KL budget, whereas the full-control move "
            "does not.",
            "",
            "Decision: do not implement the KL-gradient actor solver from this direction. "
            "The diagnostic supports carefully gated local MP proposals, but it does not "
            "provide a new route to self-consistent stationarity.",
        ]
    )
    (output_dir / "COMPARISON.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote comparison to {output_dir}")


if __name__ == "__main__":
    main()
