from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
)
from path_dt_experiments.objective_estimator_audit import (  # noqa: E402
    COMPONENT_NAMES,
    ObjectiveEstimatorAuditConfig,
    audit_objective_estimator_variance,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_specific_entropy_joint_shadow import _build_models  # noqa: E402


DEFAULT_SOURCE = Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721")
DEFAULT_FRONTIER = Path(
    "runs/heston_dt_specific_entropy_complete_mp_nested_trust100_10x200_seed1234_20260722"
)
JOINT_BLOCK_NAME = "volatility_law_joint_prefix_future_rv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the selected specific-entropy generator and cross common "
            "future shocks with disjoint target-law minibatches. Decompose the "
            "complete P/R target into running-cost, base-discrepancy, and joint "
            "prefix/future-discrepancy contributions."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--frontier-run-dir", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prefixes", type=int, default=128)
    parser.add_argument("--future-branches", type=int, default=32)
    parser.add_argument("--target-batches", type=int, default=16)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--step-indices", type=int, nargs="+", default=(64, 80, 95))
    parser.add_argument("--split-step", type=int, default=64)
    parser.add_argument("--regime-prefix-window", type=int, default=16)

    # Arguments used to reconstruct the selected model exactly.
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--future-steps", type=int, default=32)
    parser.add_argument("--prefix-windows", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--recent-return-window", type=int, default=16)
    parser.add_argument("--bandwidths", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--regime-weight", type=float, default=0.0)
    parser.add_argument("--regime-gate-temperature-fraction", type=float, default=0.25)
    parser.add_argument("--regime-tail-temperature", type=float, default=0.15)
    parser.add_argument("--regime-conditional-moment-weight", type=float, default=0.0)
    return parser.parse_args()


def _split_discrepancy(
    discrepancy: CompositePathFunctionalDiscrepancy,
) -> tuple[CompositePathFunctionalDiscrepancy, CompositePathFunctionalDiscrepancy]:
    joint_blocks = tuple(
        block for block in discrepancy.blocks if block.name == JOINT_BLOCK_NAME
    )
    base_blocks = tuple(
        block for block in discrepancy.blocks if block.name != JOINT_BLOCK_NAME
    )
    if len(joint_blocks) != 1:
        raise ValueError(
            f"expected exactly one {JOINT_BLOCK_NAME!r} block, found {len(joint_blocks)}"
        )
    if not base_blocks:
        raise ValueError("selected discrepancy has no base blocks")
    return (
        CompositePathFunctionalDiscrepancy(blocks=base_blocks),
        CompositePathFunctionalDiscrepancy(blocks=joint_blocks),
    )


def _optional_float(value: object) -> float:
    return float("nan") if value is None else float(value)


def _compact(report: dict[str, object]) -> dict[str, object]:
    rows = []
    for step_name, step in report["steps"].items():
        full_r = step["components"]["full"]["r"]
        variance = full_r["variance_components"]
        low = step["regimes"]["low"]
        component_effects = {
            name: (
                None
                if int(low["count"]) == 0
                else float(low["components"][name]["marginal_sigma_effect_mean"])
            )
            for name in COMPONENT_NAMES[:-1]
        }
        rows.append(
            {
                "step_index": int(step_name),
                "full_r_conditional_mean_rms": float(full_r["conditional_mean_rms"]),
                "full_r_estimated_signal_rms": float(full_r["estimated_signal_rms"]),
                "future_branch_variance_share": float(
                    variance["future_branch_share"]
                ),
                "target_batch_variance_share": float(
                    variance["target_batch_share"]
                ),
                "interaction_variance_share": float(variance["interaction_share"]),
                "target_half_correlation": full_r[
                    "target_half_teacher_agreement"
                ]["correlation"],
                "target_half_relative_rmse": float(
                    full_r["target_half_teacher_agreement"]["relative_rmse"]
                ),
                "future_half_correlation": full_r[
                    "future_half_teacher_agreement"
                ]["correlation"],
                "future_half_relative_rmse": float(
                    full_r["future_half_teacher_agreement"]["relative_rmse"]
                ),
                "low_regime_count": int(low["count"]),
                "low_regime_partition_average_sigma_change_mean": (
                    None
                    if int(low["count"]) == 0
                    else float(low["full_target_sigma_change_mean"])
                ),
                "low_regime_component_sigma_effects": component_effects,
            }
        )
    return {
        "rows": rows,
        "decision": report["decision"],
        "maximum_component_reconstruction_abs_error": float(
            report["maximum_component_reconstruction_abs_error"]
        ),
        "maximum_component_reconstruction_relative_error": float(
            report["maximum_component_reconstruction_relative_error"]
        ),
    }


def _plot(report: dict[str, object], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    steps = [int(value) for value in report["steps"]]
    full_rows = [
        report["steps"][str(step)]["components"]["full"]["r"] for step in steps
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    bottoms = [0.0] * len(steps)
    for name, label in (
        ("future_branch_share", "future branch"),
        ("target_batch_share", "target batch"),
        ("interaction_share", "interaction"),
    ):
        values = [
            float(row["variance_components"][name]) for row in full_rows
        ]
        axes[0, 0].bar(steps, values, bottom=bottoms, width=5.0, label=label)
        bottoms = [left + right for left, right in zip(bottoms, values)]
    axes[0, 0].set_title("Complete R conditional-variance shares")
    axes[0, 0].set_ylim(0.0, 1.0)

    for axis_name, label in (
        ("target_batch_aggregation", "target paths"),
        ("future_branch_aggregation", "future branches"),
    ):
        target_axis = (
            axes[0, 1]
            if axis_name == "target_batch_aggregation"
            else axes[1, 0]
        )
        for step, row in zip(steps, full_rows):
            curve = row[axis_name]
            x_key = (
                "effective_samples_per_half"
                if axis_name == "target_batch_aggregation"
                else "averaged_units_per_half"
            )
            target_axis.plot(
                [float(point[x_key]) for point in curve],
                [_optional_float(point["correlation"]) for point in curve],
                marker="o",
                label=f"step {step}",
            )
        target_axis.set_title(f"Half-teacher agreement vs {label}")
        target_axis.set_ylim(-1.0, 1.0)

    x = torch.arange(len(steps), dtype=torch.float64).numpy()
    width = 0.24
    for component_index, component in enumerate(COMPONENT_NAMES[:-1]):
        axes[1, 1].bar(
            x + (component_index - 1) * width,
            [
                float(
                    report["steps"][str(step)]["regimes"]["low"].get(
                        "components",
                        {},
                    ).get(component, {}).get("marginal_sigma_effect_mean", float("nan"))
                )
                for step in steps
            ],
            width=width,
            label=component.replace("_", " "),
        )
    axes[1, 1].set_xticks(x, [str(step) for step in steps])
    axes[1, 1].set_title("Low-regime marginal sigma effect")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)

    for axis in axes.reshape(-1):
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[1, 0].set_xlabel("averaged branches per half")
    axes[0, 1].set_xlabel("target paths per half")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_dir)
    frontier_dir = Path(args.frontier_run_dir)
    (
        _source_model,
        model,
        training,
        _stored_config,
        _heston,
        target_paths,
        _heldout_paths,
        _joint,
        mapped,
    ) = _build_models(
        source=Path(args.source_run_dir),
        device=torch.device(args.device),
        args=args,
    )
    checkpoint = torch.load(
        frontier_dir / "selected_model_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("selected checkpoint must contain a dictionary")
    model.load_checkpoint_state(checkpoint)
    if not isinstance(model.discrepancy, CompositePathFunctionalDiscrepancy):
        raise ValueError("selected model must use a composite discrepancy")
    base_discrepancy, joint_discrepancy = _split_discrepancy(model.discrepancy)
    with (frontier_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        frontier_metrics = json.load(handle)
    selected_step = int(frontier_metrics["selected_step"])

    print(
        "auditing frozen selected model: "
        f"step={selected_step} prefixes={args.prefixes} "
        f"future_branches={args.future_branches} "
        f"target_batches={args.target_batches}x{args.target_batch_size}",
        flush=True,
    )
    result = audit_objective_estimator_variance(
        model=model,
        target_paths=target_paths,
        base_discrepancy=base_discrepancy,
        joint_discrepancy=joint_discrepancy,
        training=training,
        config=ObjectiveEstimatorAuditConfig(
            num_prefixes=int(args.prefixes),
            num_future_branches=int(args.future_branches),
            num_target_batches=int(args.target_batches),
            target_batch_size=int(args.target_batch_size),
            step_indices=tuple(int(value) for value in args.step_indices),
            regime_split_step=int(args.split_step),
            regime_prefix_window=int(args.regime_prefix_window),
            seed=int(args.seed),
        ),
        progress_callback=lambda step, done, total: print(
            f"step={step} branches={done}/{total}",
            flush=True,
        ),
    )
    compact = _compact(result.report)
    payload = {
        "run_dir": str(run_dir),
        "source_run_dir": str(args.source_run_dir),
        "frontier_run_dir": str(frontier_dir),
        "selected_step": selected_step,
        "mapped_discrepancy_indices": mapped,
        "compact": compact,
        **result.report,
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(result.artifacts, run_dir / "diagnostic_artifacts.pt")
    _plot(payload, run_dir / "objective_estimator_variance_audit.png")
    for row in compact["rows"]:
        effects = row["low_regime_component_sigma_effects"]
        display_row = {
            **row,
            "low_regime_partition_average_sigma_change_mean": _optional_float(
                row["low_regime_partition_average_sigma_change_mean"]
            ),
        }
        print(
            "step={step_index} variance_share(future/target/interaction)="
            "{future_branch_variance_share:.3f}/{target_batch_variance_share:.3f}/"
            "{interaction_variance_share:.3f} target_half_corr={target_half_correlation} "
            "low_sigma_delta={low_regime_partition_average_sigma_change_mean:.6g} "
            "component_delta(run/base/joint)={run:.6g}/{base:.6g}/{joint:.6g}".format(
                **display_row,
                run=_optional_float(effects["running_cost"]),
                base=_optional_float(effects["base_discrepancy"]),
                joint=_optional_float(effects["joint_discrepancy"]),
            ),
            flush=True,
        )
    print(f"decision: {payload['decision']}", flush=True)
    print(f"wrote objective-estimator audit to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
