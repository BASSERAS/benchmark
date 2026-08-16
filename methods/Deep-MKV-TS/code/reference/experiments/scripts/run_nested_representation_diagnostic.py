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

from deep_mkv_gen_path_dt import (  # noqa: E402
    ConditionalSignalDiagnosticConfig,
    diagnose_conditional_adjoint_signal,
)
from path_dt_experiments.conditional_representation import (  # noqa: E402
    ConditionalRepresentationConfig,
    REPRESENTATIONS,
    compare_conditional_representations,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_specific_entropy_joint_shadow import _build_models  # noqa: E402


DEFAULT_FRONTIER = Path(
    "runs/heston_dt_specific_entropy_complete_mp_nested_trust100_10x200_seed1234_20260722"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare log-price, standardized-return, and hybrid causal inputs "
            "against independent nested branch-averaged complete-path P/R targets."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721"),
    )
    parser.add_argument("--frontier-run-dir", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fit-prefixes", type=int, default=256)
    parser.add_argument("--heldout-prefixes", type=int, default=256)
    parser.add_argument("--branches", type=int, default=16)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--step-indices", type=int, nargs="+", default=(16, 32, 48, 64))
    parser.add_argument("--primary-step", type=int, default=None)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=REPRESENTATIONS,
    )
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--inner-batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--representation-eval-every", type=int, default=100)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--minimum-scale", type=float, default=1e-3)
    parser.add_argument("--minimum-r2-improvement", type=float, default=0.01)
    parser.add_argument("--prefix-volatility-window", type=int, default=16)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)

    # These reproduce the selected complete-MP discrepancy when rebuilding it.
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--split-step", type=int, default=64)
    parser.add_argument("--future-steps", type=int, default=32)
    parser.add_argument("--prefix-windows", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--recent-return-window", type=int, default=16)
    parser.add_argument("--bandwidths", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--regime-weight", type=float, default=0.0)
    parser.add_argument("--regime-prefix-window", type=int, default=16)
    parser.add_argument("--regime-gate-temperature-fraction", type=float, default=0.25)
    parser.add_argument("--regime-tail-temperature", type=float, default=0.15)
    parser.add_argument("--regime-conditional-moment-weight", type=float, default=0.0)
    return parser.parse_args()


def _nested_metrics_subset(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "config": metrics["config"],
        "summary": metrics["summary"],
        "steps": {
            step: {
                "prefix_length": row["prefix_length"],
                "prefix_max_abs_difference": row["prefix_max_abs_difference"],
                "adjoint": row["adjoint"],
                "noise_adjoint_raw": row["noise_adjoint_raw"],
                "noise_adjoint_controlled": row["noise_adjoint_controlled"],
                "noise_control_variate": row["noise_control_variate"],
            }
            for step, row in metrics["steps"].items()
        },
    }


def _compact(report: dict[str, object]) -> dict[str, object]:
    primary_step = int(report["config"]["primary_step"])
    rows = []
    for name, arm in report["arms"].items():
        selected = arm["fit_selected_evaluation"]
        primary = selected["steps"][str(primary_step)]
        rows.append(
            {
                "representation": name,
                "selected_step": int(arm["optimization"]["best_step"]),
                "parameter_count": int(arm["trainable_parameter_count"]),
                "normalized_fit_loss": float(selected["normalized_fit_loss"]),
                "global_p_r2": float(
                    selected["global"]["p_r2_against_fixed_timewise_fit_mean"]
                ),
                "global_r_r2": float(
                    selected["global"]["r_r2_against_fixed_timewise_fit_mean"]
                ),
                "primary_step_r_r2": float(
                    primary["r"]["r2_against_fixed_fit_mean"]
                ),
                "primary_step_low_r_r2": float(
                    primary["regimes"]["low"]["r"]["r2_against_fixed_fit_mean"]
                ),
                "primary_step_middle_r_r2": float(
                    primary["regimes"]["middle"]["r"]["r2_against_fixed_fit_mean"]
                ),
                "primary_step_high_r_r2": float(
                    primary["regimes"]["high"]["r"]["r2_against_fixed_fit_mean"]
                ),
            }
        )
    return {
        "primary_step": primary_step,
        "rows": rows,
        "paired_comparisons": report["paired_comparisons"],
        "decision": report["decision"],
    }


def _plot(report: dict[str, object], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    primary_step = int(report["config"]["primary_step"])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name, arm in report["arms"].items():
        trajectory = list(arm["trajectory"])
        steps = [int(row["step"]) for row in trajectory]
        axes[0, 0].plot(
            steps,
            [float(row["normalized_fit_loss"]) for row in trajectory],
            label=name,
        )
        axes[0, 1].plot(
            steps,
            [
                float(row["global"]["r_r2_against_fixed_timewise_fit_mean"])
                for row in trajectory
            ],
            label=name,
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Nested-mean fitting loss")
    axes[0, 1].set_title("Independent global R²")

    names = list(report["arms"])
    x = torch.arange(len(names), dtype=torch.float64).numpy()
    selected = [report["arms"][name]["fit_selected_evaluation"] for name in names]
    axes[1, 0].bar(
        x,
        [
            float(row["steps"][str(primary_step)]["r"]["r2_against_fixed_fit_mean"])
            for row in selected
        ],
    )
    axes[1, 0].set_xticks(x, names, rotation=20, ha="right")
    axes[1, 0].set_title(f"Step {primary_step} independent R²")

    width = 0.25
    for regime_index, regime in enumerate(("low", "middle", "high")):
        axes[1, 1].bar(
            x + (regime_index - 1) * width,
            [
                float(
                    row["steps"][str(primary_step)]["regimes"][regime]["r"][
                        "r2_against_fixed_fit_mean"
                    ]
                )
                for row in selected
            ],
            width=width,
            label=regime,
        )
    axes[1, 1].set_xticks(x, names, rotation=20, ha="right")
    axes[1, 1].set_title(f"Step {primary_step} R² by prefix-volatility regime")
    for axis in axes.reshape(-1):
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
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
        _training,
        _stored_config,
        _heston,
        target_paths,
        _heldout_paths,
        _joint,
        _mapped,
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
    step_indices = tuple(int(value) for value in args.step_indices)
    fit_seed = int(args.seed) + 2_100_000
    heldout_seed = int(args.seed) + 2_200_000

    print(
        f"building fitting nested targets: prefixes={args.fit_prefixes} "
        f"branches={args.branches}",
        flush=True,
    )
    fit_nested = diagnose_conditional_adjoint_signal(
        model=model,
        target_paths=target_paths,
        training=model.training,
        config=ConditionalSignalDiagnosticConfig(
            num_prefixes=int(args.fit_prefixes),
            num_branches=int(args.branches),
            target_batch_size=int(args.target_batch_size),
            step_indices=step_indices,
            seed=fit_seed,
        ),
    )
    print(
        f"building independent nested targets: prefixes={args.heldout_prefixes} "
        f"branches={args.branches}",
        flush=True,
    )
    heldout_nested = diagnose_conditional_adjoint_signal(
        model=model,
        target_paths=target_paths,
        training=model.training,
        config=ConditionalSignalDiagnosticConfig(
            num_prefixes=int(args.heldout_prefixes),
            num_branches=int(args.branches),
            target_batch_size=int(args.target_batch_size),
            step_indices=step_indices,
            seed=heldout_seed,
        ),
    )
    print("fitting matched causal representations", flush=True)
    comparison = compare_conditional_representations(
        fit_tensors=fit_nested.tensors,
        heldout_tensors=heldout_nested.tensors,
        config=ConditionalRepresentationConfig(
            step_indices=step_indices,
            primary_step=args.primary_step,
            representations=tuple(str(value) for value in args.representations),
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            batch_size=int(args.inner_batch_size),
            max_steps=int(args.max_steps),
            eval_every=int(args.representation_eval_every),
            plateau_patience=int(args.plateau_patience),
            minimum_relative_improvement=float(args.minimum_relative_improvement),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            minimum_scale=float(args.minimum_scale),
            minimum_r2_improvement=float(args.minimum_r2_improvement),
            prefix_volatility_window=int(args.prefix_volatility_window),
            bootstrap_replicates=int(args.bootstrap_replicates),
            seed=int(args.seed),
        ),
        device=torch.device(args.device),
        dtype=model.dtype,
        log_callback=lambda name, row: print(
            "representation={name} step={step} fit={loss:.5f} "
            "independent_R2={r2:.4f}".format(
                name=name,
                step=int(row["step"]),
                loss=float(row["normalized_fit_loss"]),
                r2=float(
                    row["global"]["r_r2_against_fixed_timewise_fit_mean"]
                ),
            ),
            flush=True,
        ),
    )
    compact = _compact(comparison.report)
    payload = {
        "run_dir": str(run_dir),
        "source_run_dir": str(args.source_run_dir),
        "frontier_run_dir": str(frontier_dir),
        "selected_step": selected_step,
        "fit_nested_signal": _nested_metrics_subset(fit_nested.metrics),
        "heldout_nested_signal": _nested_metrics_subset(heldout_nested.metrics),
        "compact": compact,
        **comparison.report,
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(
        {
            "fit_nested_tensors": fit_nested.tensors,
            "heldout_nested_tensors": heldout_nested.tensors,
            "comparison": comparison.artifacts,
        },
        run_dir / "diagnostic_artifacts.pt",
    )
    _plot(payload, run_dir / "nested_representation_comparison.png")
    for row in compact["rows"]:
        print(
            "{representation}: global R2={global_r_r2:.4f}; step {primary} "
            "R2={primary_step_r_r2:.4f}; low/mid/high="
            "{primary_step_low_r_r2:.4f}/{primary_step_middle_r_r2:.4f}/"
            "{primary_step_high_r_r2:.4f}".format(
                primary=int(compact["primary_step"]),
                **row,
            ),
            flush=True,
        )
    print(f"decision: {comparison.report['decision']}", flush=True)
    print(f"wrote nested representation diagnostic to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
