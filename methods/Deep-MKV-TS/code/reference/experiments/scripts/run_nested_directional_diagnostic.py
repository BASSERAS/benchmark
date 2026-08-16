from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import DiscreteMPNestedTrainingConfig  # noqa: E402
from path_dt_experiments.nested_directional import (  # noqa: E402
    DEFAULT_RELAXATIONS,
    diagnose_first_nested_direction,
)
from path_dt_experiments.runners import (  # noqa: E402
    default_run_dir,
    ensure_run_dir,
    write_json,
)
from run_specific_entropy_joint_shadow import _build_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the first frozen complete-MP backward problem, then evaluate "
            "the common-random-number objective along parameter relaxation, "
            "adjoint-sign, and dt-scaling variants."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721"),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--split-step", type=int, default=64)
    parser.add_argument("--future-steps", type=int, default=32)
    parser.add_argument("--prefix-windows", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--recent-return-window", type=int, default=16)
    parser.add_argument("--bandwidths", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--inner-steps", type=int, default=200)
    parser.add_argument("--outer-batch-size", type=int, default=1024)
    parser.add_argument("--outer-target-batch-size", type=int, default=1024)
    parser.add_argument("--inner-batch-size", type=int, default=256)
    parser.add_argument(
        "--relaxations",
        type=float,
        nargs="+",
        default=DEFAULT_RELAXATIONS,
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "inner_steps",
        "outer_batch_size",
        "outer_target_batch_size",
        "inner_batch_size",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be >= 1")
    if int(args.inner_batch_size) > int(args.outer_batch_size):
        raise ValueError("inner-batch-size cannot exceed outer-batch-size")
    relaxations = tuple(float(value) for value in args.relaxations)
    if any(not math.isfinite(value) for value in relaxations):
        raise ValueError("relaxations must be finite")
    if 0.0 not in relaxations or 1.0 not in relaxations:
        raise ValueError("relaxations must include zero and one")


def _plot_curves(report: dict[str, object], *, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = list(report["objective_curves"])
    summaries = dict(report["curve_summaries"])
    variants = [str(item["name"]) for item in report["config"]["variants"]]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in variants:
        selected = sorted(
            (row for row in rows if str(row["variant"]) == name),
            key=lambda row: float(row["relaxation"]),
        )
        x = [float(row["relaxation"]) for row in selected]
        y = [float(row["complete_objective_value"]) for row in selected]
        axes[0].plot(x, y, marker="o", markersize=3, linewidth=1.2, label=name)
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_xlabel("parameter relaxation")
    axes[0].set_ylabel("complete objective")
    axes[0].set_title("Sign and scaling variants")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    correct = sorted(
        (row for row in rows if str(row["variant"]) == "correct"),
        key=lambda row: float(row["relaxation"]),
    )
    x = [float(row["relaxation"]) for row in correct]
    for metric, label in (
        ("complete_objective_value", "complete"),
        ("discrepancy_objective_value", "discrepancy"),
        ("running_cost_value", "running cost"),
    ):
        axes[1].plot(
            x,
            [float(row[metric]) for row in correct],
            marker="o",
            markersize=3,
            label=label,
        )
    minimum = float(summaries["correct"]["minimum_relaxation"])
    axes[1].axvline(minimum, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_xlabel("parameter relaxation")
    axes[1].set_ylabel("objective contribution")
    axes[1].set_title(f"Implemented direction; grid minimum at {minimum:g}")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    run_dir = ensure_run_dir(
        args.run_dir
        or default_run_dir(prefix="heston_dt_nested_directional_diagnostic")
    )
    (
        _source_model,
        candidate_model,
        training,
        _stored_config,
        _heston,
        target_paths,
        _heldout_paths,
        _joint_discrepancy,
        _mapped,
    ) = _build_models(
        source=Path(args.source_run_dir),
        device=torch.device(args.device),
        args=args,
    )
    training = replace(training, grad_clip_norm=None)
    nested = DiscreteMPNestedTrainingConfig(
        outer_steps=1,
        inner_steps=int(args.inner_steps),
        outer_batch_size=int(args.outer_batch_size),
        outer_target_batch_size=int(args.outer_target_batch_size),
        inner_batch_size=int(args.inner_batch_size),
        fixed_point_probe_paths=min(512, int(args.outer_batch_size)),
        outer_objective_line_search=False,
        seed=int(args.seed),
    )
    print("fitting the frozen backward candidate and tracing objective curves", flush=True)
    report, artifacts = diagnose_first_nested_direction(
        model=candidate_model,
        target_paths=target_paths,
        training=training,
        nested=nested,
        relaxations=tuple(float(value) for value in args.relaxations),
    )
    report["run_dir"] = str(run_dir)
    report["source_run_dir"] = str(args.source_run_dir)
    write_json(run_dir / "metrics.json", report)
    torch.save(artifacts, run_dir / "diagnostic_artifacts.pt")
    _plot_curves(report, output_path=run_dir / "objective_curves.png")

    summaries = dict(report["curve_summaries"])
    correct = dict(summaries["correct"])
    print(
        "correct direction: derivative={derivative:.6g}, grid minimum rho={rho:.6g}, "
        "objective {zero:.6g}->{minimum:.6g}->{full:.6g}".format(
            derivative=float(
                correct["complete_objective_value_central_derivative"]
            ),
            rho=float(correct["minimum_relaxation"]),
            zero=float(correct["zero_complete_objective"]),
            minimum=float(correct["minimum_complete_objective"]),
            full=float(correct["full_complete_objective"]),
        ),
        flush=True,
    )
    for name in (
        "p_only",
        "r_only",
        "reverse_p",
        "reverse_r",
        "reverse_both",
        "r_without_sqrt_dt_divisor",
        "r_with_dt_divisor",
        "p_times_dt",
        "p_divided_by_dt",
    ):
        summary = dict(summaries[name])
        print(
            f"{name}: derivative="
            f"{float(summary['complete_objective_value_central_derivative']):.6g}, "
            f"minimum rho={float(summary['minimum_relaxation']):.6g}, "
            f"minimum objective={float(summary['minimum_complete_objective']):.6g}",
            flush=True,
        )
    print(f"wrote directional diagnostic to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
