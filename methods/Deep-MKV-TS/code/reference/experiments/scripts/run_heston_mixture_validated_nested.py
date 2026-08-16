from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteMPTrainingConfig,
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
from path_dt_experiments.runners import (  # noqa: E402
    ensure_run_dir,
    write_history_jsonl,
    write_json,
)
from path_dt_experiments.validated_nested_solver import (  # noqa: E402
    ValidatedNestedSolverConfig,
    fit_validated_nested,
)
from run_heston_mixture_deep_mkv_ablation import (  # noqa: E402
    _base_discrepancy,
    _build_model,
    _evaluate,
    _load_log_paths,
    _load_manifest,
    _primary_metrics,
    _sample_prices,
)


DEFAULT_EXPERIMENT = (
    REPO_ROOT / "runs" / "heston_parameter_mixture_seed0_20260730"
)
DEFAULT_SOURCE = (
    REPO_ROOT
    / "runs"
    / "heston_mixture_deep_mkv_ablation_seed0_20260730"
    / "summary_online_uncapped_causal_ce"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a price-only Heston-mixture Deep-MKV generator with "
            "validation-selected frozen backward solves and gated controlled "
            "forward updates."
        )
    )
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_SOURCE / "model_checkpoint.pt",
    )
    parser.add_argument(
        "--source-metrics",
        type=Path,
        default=DEFAULT_SOURCE / "validation_metrics.json",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=92_031)
    parser.add_argument("--validation-pool-paths", type=int, default=2_048)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.6)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--summary-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--outer-steps", type=int, default=4)
    parser.add_argument("--backward-fit-paths", type=int, default=1_024)
    parser.add_argument("--backward-validation-paths", type=int, default=512)
    parser.add_argument("--outer-objective-paths", type=int, default=512)
    parser.add_argument("--outer-selection-banks", type=int, default=4)
    parser.add_argument("--outer-confirmation-banks", type=int, default=4)
    parser.add_argument("--target-batch-size", type=int, default=2_048)
    parser.add_argument("--inner-batch-size", type=int, default=256)
    parser.add_argument("--inner-max-steps", type=int, default=1_500)
    parser.add_argument("--inner-eval-every", type=int, default=50)
    parser.add_argument("--inner-patience", type=int, default=8)
    parser.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--minimum-outer-relative-improvement",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--minimum-confirmation-improvement-fraction",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--gate-step-indices",
        type=int,
        nargs="+",
        default=(64, 96),
    )
    parser.add_argument(
        "--minimum-r-teacher-correlation",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--maximum-r-teacher-relative-rmse",
        type=float,
        default=1.25,
    )
    parser.add_argument("--minimum-validation-p-r2", type=float, default=0.0)
    parser.add_argument("--minimum-validation-r-r2", type=float, default=0.0)
    parser.add_argument(
        "--outer-relaxations",
        type=float,
        nargs="+",
        default=(
            0.0,
            0.0001220703125,
            0.000244140625,
            0.00048828125,
            0.0009765625,
            0.001953125,
            0.00390625,
            0.0078125,
            0.015625,
            0.03125,
            0.0625,
            0.125,
        ),
    )
    parser.add_argument("--outer-coordinate-passes", type=int, default=2)
    parser.add_argument("--bank-size", type=int, default=8_192)
    parser.add_argument("--sample-batch-size", type=int, default=8_192)
    return parser.parse_args()


def default_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        REPO_ROOT
        / "runs"
        / f"heston_mixture_validated_nested_{timestamp}"
    )


def _paired_primary_comparison(
    source_metrics: dict[str, object],
    candidate_metrics: dict[str, object],
) -> dict[str, object]:
    source_primary = _primary_metrics(source_metrics)
    candidate_primary = _primary_metrics(candidate_metrics)
    changes: dict[str, dict[str, float | None]] = {}
    for name, source_value in source_primary.items():
        candidate_value = candidate_primary[name]
        absolute = candidate_value - source_value
        relative = None if source_value == 0.0 else absolute / abs(source_value)
        changes[name] = {
            "source": source_value,
            "candidate": candidate_value,
            "absolute_change": absolute,
            "relative_change": relative,
        }
    return {
        "source": source_primary,
        "validated_nested": candidate_primary,
        "candidate_minus_source": changes,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.experiment_root)
    checkpoint_path = Path(args.source_checkpoint)
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
    s0 = float(manifest["s0"])
    grid = DiscreteTimeGrid(T=num_steps * dt, num_steps=num_steps)
    all_target_paths = _load_log_paths(
        root / "data" / "train.npy",
        device=device,
        dtype=dtype,
    )
    validation_count = int(args.validation_pool_paths)
    if not 1 <= validation_count < int(all_target_paths.shape[0]):
        raise ValueError(
            "validation-pool-paths must leave a nonempty backward fitting pool"
        )
    backward_pool = all_target_paths[:-validation_count]
    validation_pool = all_target_paths[-validation_count:]

    reference = fit_local_gaussian_reference_kernel(
        target_paths=all_target_paths,
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
        all_target_paths,
        grid=grid,
        split_index=64,
        future_horizon=32,
        prefix_windows=(16, 32, 64),
        recent_return_window=16,
        bandwidths=(0.5, 1.0, 2.0),
    )
    summary = fit_fixed_target_heston_mixture_summary_mmd(
        all_target_paths,
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
                name="mixture_law_joint_path_summaries",
                weight=float(args.summary_weight),
                discrepancy=summary,
            ),
        )
    )
    ce_estimator = HestonMixtureCausalRidgeConditionalExpectation(
        grid=grid,
        ridge=float(args.ridge_lambda),
    )
    training = DiscreteMPTrainingConfig(
        batch_size=int(args.backward_fit_paths),
        target_batch_size=int(args.target_batch_size),
        num_steps=int(args.outer_steps) * int(args.inner_max_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        lambda_scale=50.0,
        adjoint_weight=1.0,
        adjoint_noise_weight=1.0,
        ce_target_mode="ridge",
        ridge_lambda=float(args.ridge_lambda),
        ce_crossfit_folds=1,
        target_preconditioner="none",
        noise_target_control_variate="adjoint",
        grad_clip_norm=None,
        log_every=int(args.inner_eval_every),
        seed=int(args.seed),
        observed_only=False,
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
    source_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_checkpoint_state(source_checkpoint)
    source_common_bank_output = (
        run_dir
        / (
            f"source_generated_paths_common_seed_{int(args.bank_size)}x"
            f"{num_steps + 1}.npy"
        )
    )
    source_common_metrics_output = (
        run_dir / "source_validation_metrics_common_seed.json"
    )
    if (
        source_common_bank_output.exists()
        and source_common_metrics_output.exists()
    ):
        source_common_metrics = json.loads(
            source_common_metrics_output.read_text(encoding="utf-8")
        )
    else:
        source_common_bank = _sample_prices(
            model,
            num_paths=int(args.bank_size),
            batch_size=int(args.sample_batch_size),
            seed=int(args.seed),
            s0=s0,
        )
        np.save(source_common_bank_output, source_common_bank)
        _evaluate(
            root=root,
            generated=source_common_bank_output,
            output=source_common_metrics_output,
        )
        source_common_metrics = json.loads(
            source_common_metrics_output.read_text(encoding="utf-8")
        )
    config = ValidatedNestedSolverConfig(
        outer_steps=int(args.outer_steps),
        backward_fit_paths=int(args.backward_fit_paths),
        backward_validation_paths=int(args.backward_validation_paths),
        outer_objective_paths=int(args.outer_objective_paths),
        outer_selection_banks=int(args.outer_selection_banks),
        outer_confirmation_banks=int(args.outer_confirmation_banks),
        target_batch_size=int(args.target_batch_size),
        inner_batch_size=int(args.inner_batch_size),
        inner_max_steps=int(args.inner_max_steps),
        inner_eval_every=int(args.inner_eval_every),
        inner_patience=int(args.inner_patience),
        minimum_relative_improvement=float(
            args.minimum_relative_improvement
        ),
        minimum_outer_relative_improvement=float(
            args.minimum_outer_relative_improvement
        ),
        minimum_confirmation_improvement_fraction=float(
            args.minimum_confirmation_improvement_fraction
        ),
        antithetic_noise=True,
        gate_step_indices=tuple(
            int(value) for value in args.gate_step_indices
        ),
        minimum_r_teacher_correlation=float(
            args.minimum_r_teacher_correlation
        ),
        maximum_r_teacher_relative_rmse=float(
            args.maximum_r_teacher_relative_rmse
        ),
        minimum_validation_p_r2=float(args.minimum_validation_p_r2),
        minimum_validation_r_r2=float(args.minimum_validation_r_r2),
        outer_relaxations=tuple(
            float(value) for value in args.outer_relaxations
        ),
        outer_coordinate_passes=int(args.outer_coordinate_passes),
        seed=int(args.seed),
    )

    def progress(row: dict[str, object]) -> None:
        best = row["best_backward_evaluation"]
        update = row["outer_update"]
        reliability = row["teacher_reliability"]["r"]
        confirmation = update["confirmation_banks"]
        print(
            f"outer={int(row['outer_step'])} "
            f"best_inner={int(row['backward_optimization']['best_step'])} "
            f"validation(P/R R2)="
            f"{float(best['validation_gate']['p']['r2_against_fit_timewise_mean']):.3f}/"
            f"{float(best['validation_gate']['r']['r2_against_fit_timewise_mean']):.3f} "
            f"R_reliability={reliability['correlation']}/"
            f"{float(reliability['relative_rmse']):.3f} "
            f"gate={row['gate']['backward_gate_pass']} "
            f"proposed={update['proposed_rates']} "
            f"confirmed={update['accepted']} "
            f"confirm(mean/fraction)="
            f"{float(confirmation['mean_relative_improvement']):.3%}/"
            f"{float(confirmation['improvement_fraction']):.2f} "
            f"applied={update['selected_rates']} "
            f"objective_change={float(update['objective_change']):.5g}",
            flush=True,
        )

    result = fit_validated_nested(
        model=model,
        backward_target_paths=backward_pool,
        validation_target_paths=validation_pool,
        training=training,
        config=config,
        progress_callback=progress,
    )
    checkpoint_output = run_dir / "model_checkpoint.pt"
    report_output = run_dir / "validated_nested_report.json"
    artifact_output = run_dir / "validated_nested_artifacts.pt"
    torch.save(model.checkpoint_state(), checkpoint_output)
    torch.save(result.artifacts, artifact_output)
    write_history_jsonl(run_dir / "training_history.jsonl", result.history)
    write_json(report_output, result.report)

    bank_output = (
        run_dir
        / f"generated_paths_{int(args.bank_size)}x{num_steps + 1}.npy"
    )
    bank = _sample_prices(
        model,
        num_paths=int(args.bank_size),
        batch_size=int(args.sample_batch_size),
        seed=int(args.seed),
        s0=s0,
    )
    np.save(bank_output, bank)
    metrics_output = run_dir / "validation_metrics.json"
    _evaluate(
        root=root,
        generated=bank_output,
        output=metrics_output,
    )
    metrics = json.loads(metrics_output.read_text(encoding="utf-8"))
    comparison = _paired_primary_comparison(source_common_metrics, metrics)
    historical_source_metrics_path = Path(args.source_metrics)
    write_json(
        run_dir / "comparison.json",
        {
            "protocol": result.report["protocol"],
            "data_partition": {
                "complete_training_pool": int(all_target_paths.shape[0]),
                "backward_fitting_pool": int(backward_pool.shape[0]),
                "solver_validation_pool": int(validation_pool.shape[0]),
                "disc_paths_used_only_for_final_evaluation": True,
            },
            "source_checkpoint": str(checkpoint_path.resolve()),
            "source_common_random_numbers": {
                "seed": int(args.seed),
                "generated_paths": str(source_common_bank_output.resolve()),
                "metrics": str(source_common_metrics_output.resolve()),
            },
            "historical_source_metrics": (
                str(historical_source_metrics_path.resolve())
                if historical_source_metrics_path.exists()
                else None
            ),
            "training": asdict(training),
            "validated_nested": asdict(config),
            "primary_metrics": comparison,
        },
    )
    print(json.dumps(comparison, indent=2), flush=True)
    print(f"saved validated nested run to {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
