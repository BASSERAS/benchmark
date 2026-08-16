from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate locked nested-trust fresh-test results across seeds."
    )
    parser.add_argument("--one-step-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--multi-step-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def _load_metrics(run_dir: Path) -> dict[str, object]:
    path = Path(run_dir) / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _row(run_dir: Path, *, horizon: str) -> dict[str, object]:
    payload = _load_metrics(run_dir)
    config = dict(payload["config"])
    fresh = dict(payload["fresh_test"])
    baseline = dict(fresh["baseline"])
    selected = dict(fresh["selected"])
    baseline_shadow = dict(baseline["shadowing"])
    selected_shadow = dict(selected["shadowing"])
    baseline_stylized = dict(baseline["stylized"])
    selected_stylized = dict(selected["stylized"])
    paired = dict(fresh["paired_bootstrap"])
    rv_pair = dict(paired["realized_volatility_crps"])
    coverage_pair = dict(paired["realized_volatility_coverage_90"])
    return {
        "horizon": horizon,
        "seed": int(config["seed"]),
        "run_dir": str(run_dir),
        "selected_step": int(payload["selected_step"]),
        "fresh_gate": bool(dict(fresh["decision"])["pass"]),
        "rv_crps_baseline": float(baseline_shadow["realized_volatility_crps"]),
        "rv_crps_selected": float(selected_shadow["realized_volatility_crps"]),
        "rv_crps_change": float(rv_pair["estimate"]),
        "rv_crps_ci_lower": float(rv_pair["ci_lower"]),
        "rv_crps_ci_upper": float(rv_pair["ci_upper"]),
        "coverage_baseline": float(
            baseline_shadow["realized_volatility_coverage_90"]
        ),
        "coverage_selected": float(
            selected_shadow["realized_volatility_coverage_90"]
        ),
        "coverage_change": float(coverage_pair["estimate"]),
        "coverage_ci_lower": float(coverage_pair["ci_lower"]),
        "coverage_ci_upper": float(coverage_pair["ci_upper"]),
        "return_crps_change": float(
            selected_shadow["cumulative_return_crps"]
            - baseline_shadow["cumulative_return_crps"]
        ),
        "kurtosis_error_baseline": float(
            baseline_stylized["excess_kurtosis_error_rms"]
        ),
        "kurtosis_error_selected": float(
            selected_stylized["excess_kurtosis_error_rms"]
        ),
        "kurtosis_error_change": float(
            selected_stylized["excess_kurtosis_error_rms"]
            - baseline_stylized["excess_kurtosis_error_rms"]
        ),
        "path_swd_change": float(
            selected_stylized["path_swd"] - baseline_stylized["path_swd"]
        ),
    }


def _metric_summary(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("summary values must be finite and non-empty")
    return {
        "mean": float(statistics.fmean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = (
        "selected_step",
        "rv_crps_change",
        "coverage_change",
        "return_crps_change",
        "kurtosis_error_change",
        "path_swd_change",
    )
    result: dict[str, object] = {
        name: _metric_summary([float(row[name]) for row in rows])
        for name in metrics
    }
    result.update(
        {
            "num_seeds": len(rows),
            "fresh_gate_passes": sum(bool(row["fresh_gate"]) for row in rows),
            "rv_crps_improvement_count": sum(
                float(row["rv_crps_change"]) < 0.0 for row in rows
            ),
            "rv_crps_ci_excludes_zero_count": sum(
                float(row["rv_crps_ci_upper"]) < 0.0 for row in rows
            ),
            "coverage_improvement_count": sum(
                float(row["coverage_change"]) > 0.0 for row in rows
            ),
            "coverage_ci_nonnegative_count": sum(
                float(row["coverage_ci_lower"]) >= 0.0 for row in rows
            ),
            "kurtosis_improvement_count": sum(
                float(row["kurtosis_error_change"]) < 0.0 for row in rows
            ),
        }
    )
    return result


def _plot(rows: list[dict[str, object]], *, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    horizons = ("one_step", "multi_step")
    colors = {"one_step": "#1f77b4", "multi_step": "#ff7f0e"}
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = (
        ("rv_crps_change", "Fresh RV CRPS change", 1e4),
        ("coverage_change", "Fresh 90% coverage change", 100.0),
        ("kurtosis_error_change", "Fresh kurtosis-error change", 1.0),
        ("path_swd_change", "Fresh path-SWD change", 1.0),
    )
    for axis, (metric, title, scale) in zip(axes.reshape(-1), panels):
        for x, horizon in enumerate(horizons):
            selected = sorted(
                (row for row in rows if row["horizon"] == horizon),
                key=lambda row: int(row["seed"]),
            )
            values = [scale * float(row[metric]) for row in selected]
            axis.scatter(
                [x] * len(values),
                values,
                color=colors[horizon],
                s=45,
                zorder=3,
            )
            axis.plot(
                [x - 0.15, x + 0.15],
                [statistics.fmean(values)] * 2,
                color="black",
                linewidth=2,
            )
        axis.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
        axis.set_xticks((0, 1), ("one step", "selected multi-step"))
        axis.set_title(title + (" ×10⁴" if scale == 1e4 else (" (pp)" if scale == 100 else "")))
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Locked one-eighth nested MP correction across seeds")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if len(args.one_step_runs) != len(args.multi_step_runs):
        raise ValueError("one-step and multi-step run lists must have equal length")
    rows = [
        *(_row(path, horizon="one_step") for path in args.one_step_runs),
        *(_row(path, horizon="multi_step") for path in args.multi_step_runs),
    ]
    seeds_by_horizon = {
        horizon: sorted(int(row["seed"]) for row in rows if row["horizon"] == horizon)
        for horizon in ("one_step", "multi_step")
    }
    if seeds_by_horizon["one_step"] != seeds_by_horizon["multi_step"]:
        raise ValueError("one-step and multi-step runs must use the same seeds")
    run_dir = ensure_run_dir(args.run_dir)
    report = {
        "protocol": {
            "source_warm_start": True,
            "outer_block_trust_fraction": 0.125,
            "one_step_is_fixed_before_fresh_test": True,
            "multi_step_checkpoint_selected_on_validation_only": True,
        },
        "seeds": seeds_by_horizon["one_step"],
        "rows": rows,
        "one_step": _summary(
            [row for row in rows if row["horizon"] == "one_step"]
        ),
        "multi_step": _summary(
            [row for row in rows if row["horizon"] == "multi_step"]
        ),
    }
    write_json(run_dir / "metrics.json", report)
    _plot(rows, output_path=run_dir / "multiseed_summary.png")
    for horizon in ("one_step", "multi_step"):
        summary = dict(report[horizon])
        print(
            f"{horizon}: gates={summary['fresh_gate_passes']}/{summary['num_seeds']} "
            f"mean_rv_crps_change={dict(summary['rv_crps_change'])['mean']:+.6g} "
            f"mean_coverage_change={dict(summary['coverage_change'])['mean']:+.6g} "
            f"mean_kurtosis_change={dict(summary['kurtosis_error_change'])['mean']:+.6g}",
            flush=True,
        )
    print(f"wrote multi-seed summary to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
