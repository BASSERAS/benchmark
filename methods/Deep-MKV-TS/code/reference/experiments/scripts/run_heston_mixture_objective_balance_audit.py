from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteTimeGrid,
    fit_local_gaussian_reference_kernel,
)
from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl  # noqa: E402
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from path_dt_experiments.discrepancies import (  # noqa: E402
    fit_fixed_target_joint_prefix_future_volatility_mmd,
)
from path_dt_experiments.heston_mixture_discrepancy import (  # noqa: E402
    HestonMixtureCausalRidgeConditionalExpectation,
    fit_fixed_target_heston_mixture_summary_mmd,
)
from path_dt_experiments.objective_estimator_audit import (  # noqa: E402
    COMPONENT_NAMES,
    ObjectiveEstimatorAuditConfig,
    audit_objective_estimator_variance,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_heston_mixture_deep_mkv_ablation import (  # noqa: E402
    _base_discrepancy,
    _build_model,
    _load_log_paths,
    _load_manifest,
    _make_training,
)


SUMMARY_BLOCK_NAME = "mixture_law_joint_path_summaries"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the price-only Heston-mixture Deep-MKV generator and "
            "decompose its complete conditional P/R target into the "
            "specific-entropy running cost, generic path discrepancies, and "
            "the label-free joint path-summary discrepancy."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "heston_parameter_mixture_seed0_20260730"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "heston_mixture_deep_mkv_ablation_seed0_20260730"
            / "summary_online_uncapped_causal_ce"
            / "model_checkpoint.pt"
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=81031)
    parser.add_argument("--prefixes", type=int, default=64)
    parser.add_argument("--future-branches", type=int, default=16)
    parser.add_argument("--target-batches", type=int, default=8)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--step-indices", type=int, nargs="+", default=(32, 64, 96))
    parser.add_argument("--split-step", type=int, default=32)
    parser.add_argument("--regime-prefix-window", type=int, default=16)

    # Frozen model reconstruction.
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.6)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--summary-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-batch-size-training", type=int, default=256)
    parser.add_argument("--training-steps", type=int, default=1600)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clipped-grad-norm", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def default_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        REPO_ROOT
        / "runs"
        / f"heston_mixture_objective_balance_audit_{timestamp}"
    )


def split_discrepancy(
    discrepancy: CompositePathFunctionalDiscrepancy,
) -> tuple[
    CompositePathFunctionalDiscrepancy,
    CompositePathFunctionalDiscrepancy,
]:
    summary = tuple(
        block
        for block in discrepancy.blocks
        if block.name == SUMMARY_BLOCK_NAME
    )
    remainder = tuple(
        block
        for block in discrepancy.blocks
        if block.name != SUMMARY_BLOCK_NAME
    )
    if len(summary) != 1:
        raise ValueError(
            f"expected one {SUMMARY_BLOCK_NAME!r} block, found {len(summary)}"
        )
    if not remainder:
        raise ValueError("generic path discrepancy group must not be empty")
    return (
        CompositePathFunctionalDiscrepancy(blocks=remainder),
        CompositePathFunctionalDiscrepancy(blocks=summary),
    )


def compact_report(report: dict[str, object]) -> dict[str, object]:
    rows = []
    for step_name, step in report["steps"].items():
        components = step["components"]
        projection = step["network_projection_to_complete_conditional_target"]
        component_rows = {}
        for name in COMPONENT_NAMES[:-1]:
            component_rows[name] = {
                axis: {
                    "pathwise_rms": float(components[name][axis]["pathwise_rms"]),
                    "conditional_signal_rms": float(
                        components[name][axis]["estimated_signal_rms"]
                    ),
                    "conditional_survival_ratio": float(
                        components[name][axis][
                            "estimated_signal_to_pathwise_rms_ratio"
                        ]
                    ),
                    "cosine_with_full": components[name][
                        f"{axis}_alignment_with_full"
                    ]["cosine_with_full"]
                    if f"{axis}_alignment_with_full" in components[name]
                    else None,
                }
                for axis in ("p", "r")
            }
        regimes = {}
        for regime_name, regime in step["regimes"].items():
            if int(regime["count"]) == 0:
                regimes[regime_name] = {"count": 0}
                continue
            regimes[regime_name] = {
                "count": int(regime["count"]),
                "current_sigma_mean": float(regime["current_sigma_mean"]),
                "complete_target_sigma_change_mean": float(
                    regime["full_target_sigma_change_mean"]
                ),
                "component_sigma_effect_mean": {
                    name: float(
                        regime["components"][name][
                            "marginal_sigma_effect_mean"
                        ]
                    )
                    for name in COMPONENT_NAMES[:-1]
                },
                "complete_target_alpha_change_mean": float(
                    regime["full_target_alpha_change_mean"]
                ),
                "component_alpha_effect_mean": {
                    name: float(
                        regime["components"][name][
                            "marginal_alpha_effect_mean"
                        ]
                    )
                    for name in COMPONENT_NAMES[:-1]
                },
            }
        rows.append(
            {
                "step_index": int(step_name),
                "components": component_rows,
                "network_projection": projection,
                "observable_prefix_volatility_terciles": regimes,
            }
        )
    return {
        "component_labels": {
            "running_cost": "specific-entropy running cost",
            "base_discrepancy": (
                "generic path discrepancies including joint prefix/future RV"
            ),
            "joint_discrepancy": (
                "label-free joint path-summary distribution discrepancy"
            ),
        },
        "rows": rows,
        "decision": report["decision"],
        "maximum_component_reconstruction_relative_error": float(
            report["maximum_component_reconstruction_relative_error"]
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.experiment_root)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    run_dir = ensure_run_dir(
        Path(args.run_dir) if args.run_dir is not None else default_run_dir()
    )
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    manifest = _load_manifest(root)
    num_steps = int(manifest["sequence_length"]) - 1
    dt = float(manifest["dt"])
    grid = DiscreteTimeGrid(T=num_steps * dt, num_steps=num_steps)
    target_paths = _load_log_paths(
        root / "data" / "train.npy",
        device=device,
        dtype=dtype,
    )
    reference = fit_local_gaussian_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        ridge=float(args.ridge_lambda),
        variance_ridge=float(args.ridge_lambda),
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    control = SpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=float(args.eta),
        reference_kernel=reference,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    base = _base_discrepancy(num_steps)
    joint = fit_fixed_target_joint_prefix_future_volatility_mmd(
        target_paths,
        grid=grid,
        split_index=64,
        future_horizon=32,
        prefix_windows=(16, 32, 64),
        recent_return_window=16,
        bandwidths=(0.5, 1.0, 2.0),
    )
    summary = fit_fixed_target_heston_mixture_summary_mmd(
        target_paths,
        grid=grid,
    )
    discrepancy = CompositePathFunctionalDiscrepancy(
        blocks=tuple(base.blocks)
        + (
            WeightedDiscrepancy(
                name="volatility_law_joint_prefix_future_rv",
                weight=1.0,
                discrepancy=joint,
            ),
            WeightedDiscrepancy(
                name=SUMMARY_BLOCK_NAME,
                weight=float(args.summary_weight),
                discrepancy=summary,
            ),
        )
    )
    generic_discrepancy, summary_discrepancy = split_discrepancy(discrepancy)
    ce_estimator = HestonMixtureCausalRidgeConditionalExpectation(
        grid=grid,
        ridge=float(args.ridge_lambda),
    )
    # The frozen arm was trained without gradient clipping.
    audit_target_batch_size = int(args.target_batch_size)
    setattr(args, "target_batch_size", int(args.target_batch_size_training))
    training = _make_training(
        args,
        clipped=False,
        seed=1,
    )
    model = _build_model(
        args=args,
        grid=grid,
        control=control,
        discrepancy=discrepancy,
        ce_estimator=ce_estimator,
        device=device,
        dtype=dtype,
        seed=0,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_checkpoint_state(checkpoint)
    print(
        "auditing frozen price-only generator "
        f"prefixes={args.prefixes} branches={args.future_branches} "
        f"target_batches={args.target_batches}x{audit_target_batch_size}",
        flush=True,
    )
    result = audit_objective_estimator_variance(
        model=model,
        target_paths=target_paths,
        base_discrepancy=generic_discrepancy,
        joint_discrepancy=summary_discrepancy,
        training=training,
        config=ObjectiveEstimatorAuditConfig(
            num_prefixes=int(args.prefixes),
            num_future_branches=int(args.future_branches),
            num_target_batches=int(args.target_batches),
            target_batch_size=audit_target_batch_size,
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
    compact = compact_report(result.report)
    payload = {
        "protocol": {
            "price_paths_only": True,
            "heston_labels_or_parameters_used": False,
            "latent_variance_used": False,
            "generator_frozen": True,
            "network_training": False,
            "running_cost_discrepancy_and_loss_changed": False,
            "prefix_groups": "observable recent-volatility terciles",
        },
        "sources": {
            "experiment_root": str(root.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
        },
        "compact": compact,
        "report": result.report,
    }
    write_json(run_dir / "objective_balance_audit.json", payload)
    torch.save(result.artifacts, run_dir / "objective_balance_artifacts.pt")
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
