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

from path_dt_experiments.antithetic_r_audit import (  # noqa: E402
    AntitheticRAuditConfig,
    audit_antithetic_r_estimator,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_specific_entropy_joint_shadow import _build_models  # noqa: E402


DEFAULT_SOURCE = Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721")
DEFAULT_FRONTIER = Path(
    "runs/heston_dt_specific_entropy_complete_mp_nested_trust100_10x200_seed1234_20260722"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare equal-cost antithetic and IID estimators of the frozen "
            "complete-MP noise adjoint R."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--frontier-run-dir", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prefixes", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--target-batches", type=int, default=4)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--step-indices", type=int, nargs="+", default=(64, 80, 95))
    parser.add_argument("--split-step", type=int, default=64)
    parser.add_argument("--regime-prefix-window", type=int, default=16)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--maximum-material-variance-ratio", type=float, default=0.75)
    parser.add_argument("--minimum-agreement-improvement", type=float, default=0.10)
    parser.add_argument("--minimum-conditional-mean-correlation", type=float, default=0.50)

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


def _compact(report: dict[str, object]) -> dict[str, object]:
    rows = []
    for step_name, step in report["steps"].items():
        comparison = step["comparison"]
        anti = step["estimators"]["antithetic"]["r"]
        iid = step["estimators"]["iid"]["r"]
        low = step["regimes"]["low"]
        rows.append(
            {
                "step_index": int(step_name),
                "variance_ratio": comparison[
                    "antithetic_to_iid_future_variance_ratio"
                ],
                "iid_equivalent_path_multiplier": float(
                    comparison["iid_equivalent_path_multiplier"]
                ),
                "antithetic_pair_half_correlation": anti[
                    "pair_half_teacher_agreement"
                ]["correlation"],
                "iid_pair_half_correlation": iid[
                    "pair_half_teacher_agreement"
                ]["correlation"],
                "pair_half_correlation_improvement": float(
                    comparison["pair_half_correlation_improvement"]
                ),
                "conditional_mean_correlation": comparison[
                    "conditional_mean_agreement"
                ]["correlation"],
                "conditional_mean_relative_rmse": float(
                    comparison["conditional_mean_agreement"]["relative_rmse"]
                ),
                "plus_minus_residual_correlation": comparison[
                    "branch_residual_correlations"
                ]["plus_vs_minus"],
                "iid_residual_correlation": comparison[
                    "branch_residual_correlations"
                ]["iid_first_vs_second"],
                "low_regime_count": int(low["count"]),
                "low_antithetic_sigma_change": (
                    None
                    if int(low["count"]) < 2
                    else low["methods"]["antithetic"]["sigma_change"]
                ),
                "low_iid_sigma_change": (
                    None
                    if int(low["count"]) < 2
                    else low["methods"]["iid"]["sigma_change"]
                ),
                "low_estimator_direction_agreement": bool(
                    low.get("estimator_direction_agreement", False)
                ),
            }
        )
    return {"rows": rows, "decision": report["decision"]}


def _optional_float(value: object) -> float:
    return float("nan") if value is None else float(value)


def _plot(report: dict[str, object], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    steps = [int(value) for value in report["steps"]]
    rows = [report["steps"][str(step)] for step in steps]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    ratios = [
        row["comparison"]["antithetic_to_iid_future_variance_ratio"] for row in rows
    ]
    estimates = [float(value["estimate"]) for value in ratios]
    axes[0, 0].errorbar(
        steps,
        estimates,
        yerr=[
            [
                estimate - float(value["ci_lower"])
                for estimate, value in zip(estimates, ratios)
            ],
            [
                float(value["ci_upper"]) - estimate
                for estimate, value in zip(estimates, ratios)
            ],
        ],
        marker="o",
        capsize=4,
    )
    axes[0, 0].axhline(1.0, color="black", linewidth=0.8)
    axes[0, 0].axhline(
        float(report["config"]["maximum_material_variance_ratio"]),
        color="tab:red",
        linewidth=0.8,
        linestyle="--",
    )
    axes[0, 0].set_title("Antithetic / IID future variance")

    for method in ("antithetic", "iid"):
        axes[0, 1].plot(
            steps,
            [
                _optional_float(
                    row["estimators"][method]["r"]["pair_half_teacher_agreement"][
                        "correlation"
                    ]
                )
                for row in rows
            ],
            marker="o",
            label=method,
        )
    axes[0, 1].set_ylim(-1.0, 1.0)
    axes[0, 1].set_title("Pair-half conditional-mean agreement")

    axes[1, 0].plot(
        steps,
        [
            _optional_float(
                row["comparison"]["branch_residual_correlations"]["plus_vs_minus"]
            )
            for row in rows
        ],
        marker="o",
        label="plus vs minus",
    )
    axes[1, 0].plot(
        steps,
        [
            _optional_float(
                row["comparison"]["branch_residual_correlations"][
                    "iid_first_vs_second"
                ]
            )
            for row in rows
        ],
        marker="o",
        label="IID first vs second",
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_ylim(-1.0, 1.0)
    axes[1, 0].set_title("Within-prefix branch residual correlation")

    x = torch.arange(len(steps), dtype=torch.float64).numpy()
    width = 0.36
    for method_index, method in enumerate(("antithetic", "iid")):
        axes[1, 1].bar(
            x + (method_index - 0.5) * width,
            [
                (
                    float("nan")
                    if int(row["regimes"]["low"]["count"]) < 2
                    else float(
                        row["regimes"]["low"]["methods"][method]["sigma_change"][
                            "estimate"
                        ]
                    )
                )
                for row in rows
            ],
            width=width,
            label=method,
        )
    axes[1, 1].set_xticks(x, [str(step) for step in steps])
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Low-regime sigma change")

    for axis in axes.reshape(-1):
        axis.grid(alpha=0.25)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
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
    with (frontier_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        frontier_metrics = json.load(handle)
    selected_step = int(frontier_metrics["selected_step"])
    print(
        "auditing equal-cost frozen R estimators: "
        f"selected_step={selected_step} prefixes={args.prefixes} "
        f"pairs={args.pairs} target_batches={args.target_batches}x"
        f"{args.target_batch_size}",
        flush=True,
    )
    result = audit_antithetic_r_estimator(
        model=model,
        target_paths=target_paths,
        training=training,
        config=AntitheticRAuditConfig(
            num_prefixes=int(args.prefixes),
            num_pairs=int(args.pairs),
            num_target_batches=int(args.target_batches),
            target_batch_size=int(args.target_batch_size),
            step_indices=tuple(int(value) for value in args.step_indices),
            regime_split_step=int(args.split_step),
            regime_prefix_window=int(args.regime_prefix_window),
            bootstrap_replicates=int(args.bootstrap_replicates),
            bootstrap_confidence=float(args.bootstrap_confidence),
            maximum_material_variance_ratio=float(
                args.maximum_material_variance_ratio
            ),
            minimum_agreement_improvement=float(
                args.minimum_agreement_improvement
            ),
            minimum_conditional_mean_correlation=float(
                args.minimum_conditional_mean_correlation
            ),
            seed=int(args.seed),
        ),
        progress_callback=lambda step, done, total: print(
            f"step={step} pairs={done}/{total}",
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
    _plot(payload, run_dir / "antithetic_r_audit.png")
    for row in compact["rows"]:
        ratio = row["variance_ratio"]
        anti_sigma = row["low_antithetic_sigma_change"]
        iid_sigma = row["low_iid_sigma_change"]
        print(
            "step={step_index} variance_ratio={ratio:.3f} "
            "CI=[{lower:.3f},{upper:.3f}] iid_multiplier={multiplier:.2f} "
            "half_corr(anti/iid)={anti_corr:.3f}/{iid_corr:.3f} "
            "mean_corr={mean_corr:.3f} low_sigma(anti/iid)="
            "{anti_sigma:.6g}/{iid_sigma:.6g}".format(
                step_index=int(row["step_index"]),
                ratio=float(ratio["estimate"]),
                lower=float(ratio["ci_lower"]),
                upper=float(ratio["ci_upper"]),
                multiplier=float(row["iid_equivalent_path_multiplier"]),
                anti_corr=_optional_float(
                    row["antithetic_pair_half_correlation"]
                ),
                iid_corr=_optional_float(row["iid_pair_half_correlation"]),
                mean_corr=_optional_float(row["conditional_mean_correlation"]),
                anti_sigma=(
                    float("nan")
                    if anti_sigma is None
                    else float(anti_sigma["estimate"])
                ),
                iid_sigma=(
                    float("nan")
                    if iid_sigma is None
                    else float(iid_sigma["estimate"])
                ),
            ),
            flush=True,
        )
    print(f"decision: {payload['decision']}", flush=True)
    print(f"wrote antithetic R audit to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
