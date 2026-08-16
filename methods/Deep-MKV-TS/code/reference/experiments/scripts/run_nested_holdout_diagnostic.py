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
from path_dt_experiments.data import simulate_heston_log_paths  # noqa: E402
from path_dt_experiments.nested_generalization import (  # noqa: E402
    DEFAULT_BLOCK_RELAXATIONS,
    PARAMETER_BLOCKS,
    apply_parameter_block_interpolation,
    diagnose_nested_objective_generalization,
)
from path_dt_experiments.runners import (  # noqa: E402
    default_run_dir,
    ensure_run_dir,
    write_json,
)
from path_dt_experiments.shadowing import (  # noqa: E402
    paired_bootstrap_shadowing_difference,
)
from run_specific_entropy_joint_shadow import (  # noqa: E402
    _build_models,
    _joint_metrics,
    _load_tensor,
    _metric_subset,
    _sample_bank,
    _shadow,
    _stream_seed,
)
from run_specific_entropy_shadow_frontier import (  # noqa: E402
    _decision,
    _shadow_standard_errors,
    _stylized,
    _stylized_standard_errors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one frozen complete-MP backward problem, compare its outer "
            "relaxation objective on fitting and independent banks, and "
            "measure common-noise path-shadowing consequences."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721"),
    )
    parser.add_argument("--validation-paths", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--initialization",
        choices=("source", "fresh"),
        default="source",
        help=(
            "Start the candidate from the saved specific-entropy checkpoint "
            "or from the existing fresh-network experimental initialization."
        ),
    )
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
    parser.add_argument("--coordinate-passes", type=int, default=2)
    parser.add_argument(
        "--relaxations",
        type=float,
        nargs="+",
        default=DEFAULT_BLOCK_RELAXATIONS,
    )
    parser.add_argument(
        "--selection-scales",
        type=float,
        nargs="+",
        default=(0.125, 0.25, 0.5, 0.75, 1.0),
        help="Fractions of the training-selected block update to audit on the path-law gate.",
    )
    parser.add_argument("--validation-real-paths", type=int, default=512)
    parser.add_argument("--validation-bank-paths", type=int, default=4096)
    parser.add_argument("--fresh-test-real-paths", type=int, default=512)
    parser.add_argument("--fresh-test-reference-paths", type=int, default=4096)
    parser.add_argument("--fresh-test-bank-paths", type=int, default=4096)
    parser.add_argument("--locked-scale", type=float, default=0.125)
    parser.add_argument("--sample-batch-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--shadow-search", choices=("exact", "faiss"), default="exact")
    parser.add_argument("--shadow-exact-batch-size", type=int, default=64)
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--shadow-bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--stylized-bootstrap-replicates", type=int, default=200)
    parser.add_argument("--stylized-bootstrap-paths", type=int, default=1024)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=24001307)
    parser.add_argument("--standard-error-budget", type=float, default=0.5)
    parser.add_argument("--swd-factor-limit", type=float, default=1.10)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "inner_steps",
        "outer_batch_size",
        "outer_target_batch_size",
        "inner_batch_size",
        "coordinate_passes",
        "validation_real_paths",
        "validation_bank_paths",
        "fresh_test_real_paths",
        "fresh_test_reference_paths",
        "fresh_test_bank_paths",
        "sample_batch_size",
        "top_k",
        "shadow_exact_batch_size",
        "swd_projections",
        "stylized_bootstrap_paths",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be >= 1")
    if int(args.inner_batch_size) > int(args.outer_batch_size):
        raise ValueError("inner-batch-size cannot exceed outer-batch-size")
    if int(args.validation_bank_paths) < int(args.top_k):
        raise ValueError("validation-bank-paths must be at least top-k")
    if int(args.fresh_test_bank_paths) < int(args.top_k):
        raise ValueError("fresh-test-bank-paths must be at least top-k")
    if int(args.fresh_test_reference_paths) < int(args.fresh_test_real_paths):
        raise ValueError("fresh-test-reference-paths must be at least fresh-test-real-paths")
    if int(args.shadow_bootstrap_replicates) < 2:
        raise ValueError("shadow-bootstrap-replicates must be >= 2")
    if int(args.stylized_bootstrap_replicates) < 2:
        raise ValueError("stylized-bootstrap-replicates must be >= 2")
    if not 0.0 < float(args.bootstrap_confidence) < 1.0:
        raise ValueError("bootstrap-confidence must be in (0, 1)")
    if float(args.standard_error_budget) < 0.0:
        raise ValueError("standard-error-budget must be >= 0")
    if float(args.swd_factor_limit) < 1.0:
        raise ValueError("swd-factor-limit must be >= 1")
    relaxations = tuple(float(value) for value in args.relaxations)
    if len(set(relaxations)) != len(relaxations):
        raise ValueError("relaxations must not contain duplicates")
    if 0.0 not in relaxations or 1.0 not in relaxations:
        raise ValueError("relaxations must contain zero and one")
    if any(not math.isfinite(value) or value < 0.0 for value in relaxations):
        raise ValueError("relaxations must be finite and non-negative")
    selection_scales = tuple(float(value) for value in args.selection_scales)
    if len(set(selection_scales)) != len(selection_scales):
        raise ValueError("selection-scales must not contain duplicates")
    if any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in selection_scales
    ):
        raise ValueError("selection-scales must lie in (0, 1]")
    if not math.isfinite(float(args.locked_scale)) or not 0.0 < float(
        args.locked_scale
    ) <= 1.0:
        raise ValueError("locked-scale must lie in (0, 1]")


def _plot_report(report: dict[str, object], *, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = list(report["objective_curves"])
    baseline = dict(dict(report["summary"])["baseline"])
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, direction in zip(
        axes.reshape(-1),
        ("joint", "p_only", "r_only", "shared_only"),
    ):
        for bank_name, color in (("training", "#1f77b4"), ("heldout", "#d62728")):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["bank"] == bank_name and row["direction"] == direction
                ),
                key=lambda row: float(row["relaxation"]),
            )
            zero = float(dict(baseline[bank_name])["complete_objective_value"])
            axis.plot(
                [float(row["relaxation"]) for row in selected],
                [float(row["complete_objective_value"]) - zero for row in selected],
                marker="o",
                markersize=3,
                label=bank_name,
                color=color,
            )
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        axis.set_xscale("symlog", linthresh=1e-3)
        axis.set_yscale("symlog", linthresh=0.1)
        axis.set_title(direction.replace("_", " "))
        axis.set_xlabel("parameter relaxation")
        axis.set_ylabel("complete objective change")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Frozen backward candidate: fitting versus independent objective bank")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    run_dir = ensure_run_dir(
        args.run_dir or default_run_dir(prefix="heston_dt_nested_holdout_diagnostic")
    )
    (
        source_model,
        candidate_model,
        training,
        _stored_config,
        heston,
        target_paths,
        source_heldout_paths,
        joint_discrepancy,
        mapped,
    ) = _build_models(
        source=Path(args.source_run_dir),
        device=torch.device(args.device),
        args=args,
    )
    if str(args.initialization) == "source":
        candidate_model.load_checkpoint_state(source_model.checkpoint_state())
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

    print("fitting one frozen backward candidate and tracing both objective banks", flush=True)
    report, artifacts = diagnose_nested_objective_generalization(
        model=candidate_model,
        target_paths=target_paths,
        heldout_paths=source_heldout_paths,
        training=training,
        nested=nested,
        relaxations=tuple(float(value) for value in args.relaxations),
        coordinate_passes=int(args.coordinate_passes),
        selection_scales=tuple(float(value) for value in args.selection_scales),
    )

    validation_reference = (
        source_heldout_paths
        if args.validation_paths is None
        else _load_tensor(Path(args.validation_paths))
    )
    if tuple(validation_reference.shape[1:]) != tuple(target_paths.shape[1:]):
        raise ValueError("validation paths must have the source target-law shape")
    if int(validation_reference.shape[0]) < int(args.validation_real_paths):
        raise ValueError("validation artifact has fewer paths than validation-real-paths")
    validation_real = validation_reference[: int(args.validation_real_paths)]
    lags = tuple(value for value in (1, 2, 5, 10, 20) if value < int(heston.num_steps))
    bank_seed = _stream_seed(int(args.seed), 80_000)
    old_values = dict(artifacts["old_parameter_values"])
    candidate_values = dict(artifacts["candidate_parameter_values"])
    selections = dict(report["selections"])
    states: dict[str, dict[str, float] | None] = {
        "source_baseline": None,
        "candidate_initial": {block: 0.0 for block in PARAMETER_BLOCKS},
        "training_selected": dict(dict(selections["training"])["rates"]),
        "heldout_selected": dict(dict(selections["heldout"])["rates"]),
        "full_candidate": {block: 1.0 for block in PARAMETER_BLOCKS},
    }
    training_selected_rates = dict(states["training_selected"])
    for scale in sorted(float(value) for value in args.selection_scales):
        if scale == 1.0:
            continue
        scale_name = str(scale).replace(".", "p")
        states[f"training_selected_scale_{scale_name}"] = {
            block: scale * float(training_selected_rates[block])
            for block in PARAMETER_BLOCKS
        }
    path_law_evaluation: dict[str, object] = {}
    generated_banks: dict[str, torch.Tensor] = {}
    shadow_results: dict[str, object] = {}
    for name, rates in states.items():
        model = source_model if rates is None else candidate_model
        if rates is not None:
            apply_parameter_block_interpolation(
                candidate_model,
                old_values=old_values,
                candidate_values=candidate_values,
                rates=rates,
            )
        bank = _sample_bank(
            model,
            num_paths=int(args.validation_bank_paths),
            seed=int(bank_seed),
            x0=float(heston.x0),
            batch_size=int(args.sample_batch_size),
        )
        generated_banks[name] = bank
        shadow_result = _shadow(
            real_paths=validation_real,
            generated_paths=bank,
            args=args,
        )
        shadow_results[name] = shadow_result
        shadow = _metric_subset(shadow_result)
        stylized = _stylized(
            bank,
            validation_reference,
            lags=lags,
            swd_projections=int(args.swd_projections),
        )
        path_law_evaluation[name] = {
            "rates": rates,
            "shadowing": shadow,
            "stylized": stylized,
            "joint_discrepancy": _joint_metrics(
                joint_discrepancy,
                generated_paths=bank,
                target_paths=validation_reference,
                grid=model.grid,
            ),
        }

    baseline_evaluation = dict(path_law_evaluation["source_baseline"])
    baseline_shadow = dict(baseline_evaluation["shadowing"])
    baseline_stylized = dict(baseline_evaluation["stylized"])
    baseline_shadow_bootstrap, shadow_standard_errors = _shadow_standard_errors(
        shadow_results["source_baseline"],
        args=args,
        seed=int(args.bootstrap_seed),
    )
    stylized_standard_errors = _stylized_standard_errors(
        generated_banks["source_baseline"],
        validation_reference,
        lags=lags,
        args=args,
        seed=_stream_seed(int(args.bootstrap_seed), 1),
    )
    for name, evaluation in path_law_evaluation.items():
        evaluation["decision"] = _decision(
            candidate_shadowing=dict(evaluation["shadowing"]),
            baseline_shadowing=baseline_shadow,
            candidate_stylized=dict(evaluation["stylized"]),
            baseline_stylized=baseline_stylized,
            shadowing_standard_errors=shadow_standard_errors,
            stylized_standard_errors=stylized_standard_errors,
            args=args,
        )

    locked_rates = {
        block: float(args.locked_scale) * float(training_selected_rates[block])
        for block in PARAMETER_BLOCKS
    }
    apply_parameter_block_interpolation(
        candidate_model,
        old_values=old_values,
        candidate_values=candidate_values,
        rates=locked_rates,
    )
    fresh_reference_seed = _stream_seed(int(args.seed), 180_000)
    fresh_bank_seed = _stream_seed(int(args.seed), 180_001)
    fresh_reference = simulate_heston_log_paths(
        num_paths=int(args.fresh_test_reference_paths),
        config=heston,
        seed=int(fresh_reference_seed),
        device="cpu",
        dtype=target_paths.dtype,
    )
    fresh_source_bank = _sample_bank(
        source_model,
        num_paths=int(args.fresh_test_bank_paths),
        seed=int(fresh_bank_seed),
        x0=float(heston.x0),
        batch_size=int(args.sample_batch_size),
    )
    fresh_locked_bank = _sample_bank(
        candidate_model,
        num_paths=int(args.fresh_test_bank_paths),
        seed=int(fresh_bank_seed),
        x0=float(heston.x0),
        batch_size=int(args.sample_batch_size),
    )
    fresh_real = fresh_reference[: int(args.fresh_test_real_paths)]
    fresh_source_result = _shadow(
        real_paths=fresh_real,
        generated_paths=fresh_source_bank,
        args=args,
    )
    fresh_locked_result = _shadow(
        real_paths=fresh_real,
        generated_paths=fresh_locked_bank,
        args=args,
    )
    fresh_source_shadow = _metric_subset(fresh_source_result)
    fresh_locked_shadow = _metric_subset(fresh_locked_result)
    fresh_source_stylized = _stylized(
        fresh_source_bank,
        fresh_reference,
        lags=lags,
        swd_projections=int(args.swd_projections),
    )
    fresh_locked_stylized = _stylized(
        fresh_locked_bank,
        fresh_reference,
        lags=lags,
        swd_projections=int(args.swd_projections),
    )
    fresh_bootstrap, fresh_shadow_standard_errors = _shadow_standard_errors(
        fresh_source_result,
        args=args,
        seed=_stream_seed(int(args.bootstrap_seed), 2),
    )
    fresh_stylized_standard_errors = _stylized_standard_errors(
        fresh_source_bank,
        fresh_reference,
        lags=lags,
        args=args,
        seed=_stream_seed(int(args.bootstrap_seed), 3),
    )
    fresh_decision = _decision(
        candidate_shadowing=fresh_locked_shadow,
        baseline_shadowing=fresh_source_shadow,
        candidate_stylized=fresh_locked_stylized,
        baseline_stylized=fresh_source_stylized,
        shadowing_standard_errors=fresh_shadow_standard_errors,
        stylized_standard_errors=fresh_stylized_standard_errors,
        args=args,
    )
    fresh_paired_shadowing = paired_bootstrap_shadowing_difference(
        fresh_locked_result,
        fresh_source_result,
        num_replicates=int(args.shadow_bootstrap_replicates),
        confidence_level=float(args.bootstrap_confidence),
        seed=_stream_seed(int(args.bootstrap_seed), 4),
    )
    fresh_test = {
        "locked_scale": float(args.locked_scale),
        "locked_rates": locked_rates,
        "reference_seed": int(fresh_reference_seed),
        "bank_seed": int(fresh_bank_seed),
        "source": {
            "shadowing": fresh_source_shadow,
            "stylized": fresh_source_stylized,
        },
        "locked_candidate": {
            "shadowing": fresh_locked_shadow,
            "stylized": fresh_locked_stylized,
        },
        "decision": fresh_decision,
        "baseline_shadowing_bootstrap": fresh_bootstrap,
        "baseline_shadowing_standard_errors": fresh_shadow_standard_errors,
        "baseline_stylized_standard_errors": fresh_stylized_standard_errors,
        "paired_shadowing_difference": fresh_paired_shadowing,
    }

    report["run_dir"] = str(run_dir)
    report["source_run_dir"] = str(args.source_run_dir)
    report["initialization"] = str(args.initialization)
    report["discrepancy_grid_mapping"] = mapped
    report["path_law_evaluation"] = path_law_evaluation
    report["fresh_test"] = fresh_test
    report["path_law_protocol"] = {
        "validation_reference": (
            "source_heldout_paths.pt"
            if args.validation_paths is None
            else str(args.validation_paths)
        ),
        "validation_real_paths": int(args.validation_real_paths),
        "validation_bank_paths": int(args.validation_bank_paths),
        "bank_seed": int(bank_seed),
        "top_k": int(args.top_k),
        "future_horizon": int(args.future_steps),
        "common_noise_across_model_states": True,
        "training_selection_scales": [
            float(value) for value in sorted(args.selection_scales)
        ],
        "shadowing_bootstrap": baseline_shadow_bootstrap,
        "shadowing_standard_errors": shadow_standard_errors,
        "stylized_standard_errors": stylized_standard_errors,
        "standard_error_budget": float(args.standard_error_budget),
        "swd_factor_limit": float(args.swd_factor_limit),
    }
    write_json(run_dir / "metrics.json", report)
    torch.save(artifacts, run_dir / "diagnostic_artifacts.pt")
    torch.save(generated_banks, run_dir / "evaluation_generated_paths.pt")
    torch.save(
        {
            "reference": fresh_reference,
            "source": fresh_source_bank,
            "locked_candidate": fresh_locked_bank,
        },
        run_dir / "fresh_test_paths.pt",
    )
    _plot_report(report, output_path=run_dir / "objective_generalization.png")

    for selector in ("training", "heldout"):
        summary = dict(dict(report["summary"])[f"{selector}_selected"])
        rates = dict(summary["rates"])
        print(
            f"{selector}-selected rates "
            f"P={float(rates['p']):.6g} R={float(rates['r']):.6g} "
            f"shared={float(rates['shared']):.6g}; "
            f"objective changes train={float(summary['training_objective_change']):+.6g} "
            f"heldout={float(summary['heldout_objective_change']):+.6g}",
            flush=True,
        )
    for name in states:
        metrics = dict(path_law_evaluation[name])
        shadow = dict(metrics["shadowing"])
        stylized = dict(metrics["stylized"])
        decision = dict(metrics["decision"])
        print(
            f"{name}: rv_crps={float(shadow['realized_volatility_crps']):.6f} "
            f"rv_cov90={float(shadow['realized_volatility_coverage_90']):.4f} "
            f"kurtosis_error={float(stylized['excess_kurtosis_error_rms']):.4f} "
            f"gate={bool(decision['pass'])}",
            flush=True,
        )
    print(
        "fresh locked scale={scale:.6g}: rv_crps={base_crps:.6f}->{candidate_crps:.6f} "
        "rv_cov90={base_cov:.4f}->{candidate_cov:.4f} "
        "kurtosis_error={base_kurt:.4f}->{candidate_kurt:.4f} gate={gate}".format(
            scale=float(args.locked_scale),
            base_crps=float(fresh_source_shadow["realized_volatility_crps"]),
            candidate_crps=float(fresh_locked_shadow["realized_volatility_crps"]),
            base_cov=float(fresh_source_shadow["realized_volatility_coverage_90"]),
            candidate_cov=float(fresh_locked_shadow["realized_volatility_coverage_90"]),
            base_kurt=float(fresh_source_stylized["excess_kurtosis_error_rms"]),
            candidate_kurt=float(fresh_locked_stylized["excess_kurtosis_error_rms"]),
            gate=bool(fresh_decision["pass"]),
        ),
        flush=True,
    )
    print(f"wrote held-out nested diagnostic to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
