from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import DiscreteMPNestedTrainingConfig  # noqa: E402
from path_dt_experiments.data import simulate_heston_log_paths  # noqa: E402
from path_dt_experiments.dated_marginals import (  # noqa: E402
    build_dated_volatility_marginals,
)
from path_dt_experiments.metrics import stylized_fact_metrics  # noqa: E402
from path_dt_experiments.runners import (  # noqa: E402
    default_run_dir,
    ensure_run_dir,
    write_history_jsonl,
    write_json,
)
from path_dt_experiments.shadow_frontier import (  # noqa: E402
    bootstrap_stylized_error_standard_errors,
    select_shadow_frontier_checkpoint,
    shadow_frontier_decision,
)
from path_dt_experiments.shadowing import (  # noqa: E402
    ShadowingResult,
    bootstrap_shadowing_metrics,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every fixed-interval checkpoint of the specific-entropy "
            "joint-volatility run on a validation-only shadow/stylized frontier, "
            "then evaluate the selected model on a newly simulated Heston law."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721"),
    )
    parser.add_argument(
        "--validation-paths",
        type=Path,
        default=Path(
            "runs/heston_dt_entropy_barrier_k25_baseline_seed4321_20260717/heldout_paths.pt"
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--initialization",
        choices=("fresh", "source", "frontier"),
        default="fresh",
        help=(
            "Initialize a fresh candidate, warm-start from the source checkpoint, "
            "or continue from a selected joint-frontier checkpoint."
        ),
    )
    parser.add_argument(
        "--frontier-run-dir",
        type=Path,
        default=Path(
            "runs/heston_dt_specific_entropy_complete_mp_nested_trust100_"
            "10x200_seed1234_20260722"
        ),
    )
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--split-step", type=int, default=64)
    parser.add_argument("--future-steps", type=int, default=32)
    parser.add_argument("--prefix-windows", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--recent-return-window", type=int, default=16)
    parser.add_argument("--bandwidths", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument(
        "--regime-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of fixed empirical prefix-regime/future-RV moment and tail "
            "claims; zero preserves the existing joint-MMD objective."
        ),
    )
    parser.add_argument("--regime-prefix-window", type=int, default=16)
    parser.add_argument("--regime-gate-temperature-fraction", type=float, default=0.25)
    parser.add_argument("--regime-tail-temperature", type=float, default=0.15)
    parser.add_argument("--regime-conditional-moment-weight", type=float, default=0.0)
    parser.add_argument(
        "--dated-marginal-weight",
        type=float,
        default=0.0,
        help=(
            "Total weight of formal dated local-volatility marginal costs. "
            "Zero preserves the existing joint objective."
        ),
    )
    parser.add_argument(
        "--dated-marginal-endpoint-spacing",
        type=int,
        default=4,
        help="Spacing of dated marginal endpoints in full-grid steps.",
    )
    parser.add_argument(
        "--dated-marginal-windows",
        type=int,
        nargs="+",
        default=(4, 8, 16, 32),
        help="Local realized-volatility windows in full-grid steps.",
    )
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--validation-real-paths", type=int, default=512)
    parser.add_argument("--validation-bank-paths", type=int, default=4096)
    parser.add_argument("--fresh-test-real-paths", type=int, default=512)
    parser.add_argument("--fresh-test-reference-paths", type=int, default=4096)
    parser.add_argument("--fresh-test-bank-paths", type=int, default=4096)
    parser.add_argument("--sample-batch-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--shadow-search", choices=("exact", "faiss"), default="exact")
    parser.add_argument("--shadow-exact-batch-size", type=int, default=64)
    parser.add_argument("--shadow-bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--stylized-bootstrap-replicates", type=int, default=200)
    parser.add_argument("--stylized-bootstrap-paths", type=int, default=1024)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=24001307)
    parser.add_argument("--standard-error-budget", type=float, default=0.5)
    parser.add_argument("--swd-factor-limit", type=float, default=1.10)
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument(
        "--solver",
        choices=("online", "nested"),
        default="online",
        help="Use one moving-target update per step or a frozen-target nested forward/backward solver.",
    )
    parser.add_argument("--nested-outer-steps", type=int, default=10)
    parser.add_argument("--nested-inner-steps", type=int, default=200)
    parser.add_argument("--nested-outer-batch-size", type=int, default=1024)
    parser.add_argument("--nested-outer-target-batch-size", type=int, default=1024)
    parser.add_argument("--nested-inner-batch-size", type=int, default=256)
    parser.add_argument("--nested-fixed-point-probe-paths", type=int, default=512)
    parser.add_argument(
        "--nested-outer-objective-line-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Backtrack the outer Picard displacement until the complete objective does not increase.",
    )
    parser.add_argument(
        "--nested-outer-relaxation-mode",
        choices=("joint", "adjoint_blocks"),
        default="joint",
        help=(
            "Relax every network parameter jointly or select separate GRU, "
            "P-head, and R-head outer rates."
        ),
    )
    parser.add_argument("--nested-outer-max-backtracks", type=int, default=10)
    parser.add_argument("--nested-outer-backtrack-factor", type=float, default=0.5)
    parser.add_argument("--nested-outer-objective-tolerance", type=float, default=0.0)
    parser.add_argument("--nested-outer-block-coordinate-passes", type=int, default=2)
    parser.add_argument(
        "--nested-outer-block-trust-fraction",
        type=float,
        default=1.0,
        help="Multiply objective-selected P/R/shared rates by this fixed trust fraction.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        default="source",
        metavar="SOURCE|NONE|FLOAT",
        help=(
            "Override the source training cap with a positive value, disable "
            "clipping with 'none', or preserve it with 'source'."
        ),
    )
    parser.add_argument(
        "--selection-objective",
        choices=("rv_crps", "coverage_error"),
        default="rv_crps",
        help=(
            "Rank eligible checkpoints by RV CRPS (legacy) or by distance to "
            "90%% RV coverage, with RV CRPS and earlier step as tie-breakers."
        ),
    )
    return parser.parse_args()


def _resolve_grad_clip_norm(
    value: str,
    *,
    source_value: float | None,
) -> float | None:
    normalized = str(value).strip().lower()
    if normalized == "source":
        return source_value
    if normalized == "none":
        return None
    try:
        result = float(normalized)
    except ValueError as exc:
        raise ValueError("grad-clip-norm must be source, none, or a positive float") from exc
    if not torch.isfinite(torch.tensor(result)) or result <= 0.0:
        raise ValueError("numeric grad-clip-norm must be a positive finite float")
    return result


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "validation_real_paths",
        "validation_bank_paths",
        "fresh_test_real_paths",
        "fresh_test_reference_paths",
        "fresh_test_bank_paths",
        "sample_batch_size",
        "top_k",
        "shadow_exact_batch_size",
        "swd_projections",
    )
    for name in positive:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be >= 1")
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
    if int(args.stylized_bootstrap_paths) < 2:
        raise ValueError("stylized-bootstrap-paths must be >= 2")
    if not 0.0 < float(args.bootstrap_confidence) < 1.0:
        raise ValueError("bootstrap-confidence must be in (0, 1)")
    _resolve_grad_clip_norm(str(args.grad_clip_norm), source_value=1.0)
    for name in (
        "regime_weight",
        "regime_gate_temperature_fraction",
        "regime_tail_temperature",
        "regime_conditional_moment_weight",
    ):
        value = float(getattr(args, name))
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"{name.replace('_', '-')} must be finite")
    if float(args.regime_weight) < 0.0:
        raise ValueError("regime-weight must be >= 0")
    if int(args.regime_prefix_window) < 1:
        raise ValueError("regime-prefix-window must be >= 1")
    if float(args.regime_gate_temperature_fraction) <= 0.0:
        raise ValueError("regime-gate-temperature-fraction must be > 0")
    if float(args.regime_tail_temperature) <= 0.0:
        raise ValueError("regime-tail-temperature must be > 0")
    if float(args.regime_conditional_moment_weight) < 0.0:
        raise ValueError("regime-conditional-moment-weight must be >= 0")
    if not torch.isfinite(torch.tensor(float(args.dated_marginal_weight))):
        raise ValueError("dated-marginal-weight must be finite")
    if float(args.dated_marginal_weight) < 0.0:
        raise ValueError("dated-marginal-weight must be >= 0")
    if int(args.dated_marginal_endpoint_spacing) < 1:
        raise ValueError("dated-marginal-endpoint-spacing must be >= 1")
    marginal_windows = tuple(int(value) for value in args.dated_marginal_windows)
    if (
        not marginal_windows
        or tuple(sorted(set(marginal_windows))) != marginal_windows
        or any(value < 1 for value in marginal_windows)
    ):
        raise ValueError(
            "dated-marginal-windows must be strictly increasing positive integers"
        )
    for name in (
        "nested_outer_steps",
        "nested_inner_steps",
        "nested_outer_batch_size",
        "nested_outer_target_batch_size",
        "nested_inner_batch_size",
        "nested_fixed_point_probe_paths",
        "nested_outer_block_coordinate_passes",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be >= 1")
    if int(args.nested_inner_batch_size) > int(args.nested_outer_batch_size):
        raise ValueError("nested-inner-batch-size cannot exceed nested-outer-batch-size")
    if int(args.nested_outer_max_backtracks) < 0:
        raise ValueError("nested-outer-max-backtracks must be >= 0")
    if not 0.0 < float(args.nested_outer_backtrack_factor) < 1.0:
        raise ValueError("nested-outer-backtrack-factor must be in (0, 1)")
    if float(args.nested_outer_objective_tolerance) < 0.0:
        raise ValueError("nested-outer-objective-tolerance must be >= 0")
    if not 0.0 < float(args.nested_outer_block_trust_fraction) <= 1.0:
        raise ValueError("nested-outer-block-trust-fraction must lie in (0, 1]")
    if float(args.nested_outer_block_trust_fraction) < 1.0 and (
        str(args.nested_outer_relaxation_mode) != "adjoint_blocks"
        or not bool(args.nested_outer_objective_line_search)
    ):
        raise ValueError(
            "a trust fraction below one requires the adjoint-blocks outer line search"
        )


def _stylized(
    generated_paths: torch.Tensor,
    reference_paths: torch.Tensor,
    *,
    lags: tuple[int, ...],
    swd_projections: int,
) -> dict[str, float]:
    return stylized_fact_metrics(
        generated_paths,
        reference_paths,
        lags=lags,
        tail_quantiles=(0.90, 0.95, 0.99),
        include_swd=True,
        swd_projections=int(swd_projections),
    )


def _shadow_standard_errors(
    result: ShadowingResult,
    *,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    bootstrap = bootstrap_shadowing_metrics(
        result,
        num_replicates=int(args.shadow_bootstrap_replicates),
        confidence_level=float(args.bootstrap_confidence),
        seed=int(seed),
    )
    return bootstrap, {
        name: float(summary["standard_error"]) for name, summary in bootstrap.items()
    }


def _stylized_standard_errors(
    generated_paths: torch.Tensor,
    reference_paths: torch.Tensor,
    *,
    lags: tuple[int, ...],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float]:
    return bootstrap_stylized_error_standard_errors(
        generated_paths,
        reference_paths,
        lags=lags,
        tail_quantiles=(0.90, 0.95, 0.99),
        num_replicates=int(args.stylized_bootstrap_replicates),
        seed=int(seed),
        resample_paths=int(args.stylized_bootstrap_paths),
    )


def _decision(
    *,
    candidate_shadowing: dict[str, float | str],
    baseline_shadowing: dict[str, float | str],
    candidate_stylized: dict[str, float],
    baseline_stylized: dict[str, float],
    shadowing_standard_errors: dict[str, float],
    stylized_standard_errors: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, object]:
    return shadow_frontier_decision(
        candidate_shadowing,
        baseline_shadowing,
        candidate_stylized,
        baseline_stylized,
        shadowing_standard_errors=shadowing_standard_errors,
        stylized_standard_errors=stylized_standard_errors,
        standard_error_budget=float(args.standard_error_budget),
        swd_factor_limit=float(args.swd_factor_limit),
        nominal_coverage=0.90,
    )


def _plot_frontier(
    records: list[dict[str, object]],
    *,
    baseline_shadowing: dict[str, float | str],
    baseline_stylized: dict[str, float],
    stylized_standard_errors: dict[str, float],
    selected_step: int,
    standard_error_budget: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    steps = [int(record["step"]) for record in records]
    eligible = [bool(dict(record["decision"])["pass"]) for record in records]
    rv_crps = [
        float(dict(record["shadowing"])["realized_volatility_crps"])
        for record in records
    ]
    coverage = [
        float(dict(record["shadowing"])["realized_volatility_coverage_90"])
        for record in records
    ]
    return_changes = [
        float(dict(record["decision"])["return_crps_change"]) for record in records
    ]
    kurtosis_errors = [
        float(dict(record["stylized"])["excess_kurtosis_error_rms"])
        for record in records
    ]
    colors = ["#2ca02c" if value else "#b0b0b0" for value in eligible]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    panels = (
        (
            axes[0, 0],
            rv_crps,
            float(baseline_shadowing["realized_volatility_crps"]),
            "RV CRPS",
        ),
        (
            axes[0, 1],
            coverage,
            float(baseline_shadowing["realized_volatility_coverage_90"]),
            "RV coverage 90%",
        ),
        (axes[1, 0], return_changes, 0.0, "Return CRPS change"),
        (
            axes[1, 1],
            kurtosis_errors,
            float(baseline_stylized["excess_kurtosis_error_rms"]),
            "Excess-kurtosis error",
        ),
    )
    for axis, values, baseline, title in panels:
        axis.plot(steps, values, color="#1f77b4", linewidth=1.5, zorder=1)
        axis.scatter(steps, values, c=colors, edgecolors="black", linewidths=0.4, zorder=2)
        axis.axhline(baseline, color="#d62728", linestyle="--", linewidth=1.2)
        if int(selected_step) > 0:
            index = steps.index(int(selected_step))
            axis.scatter(
                [steps[index]],
                [values[index]],
                marker="*",
                s=150,
                color="#ffbf00",
                edgecolors="black",
                linewidths=0.7,
                zorder=3,
            )
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0, 1].axhline(0.90, color="black", linestyle=":", linewidth=1.0)
    return_allowance = float(
        dict(records[0]["decision"])["return_crps_allowance"]
    )
    axes[1, 0].axhline(return_allowance, color="black", linestyle=":", linewidth=1.0)
    kurtosis_allowance = float(baseline_stylized["excess_kurtosis_error_rms"]) + float(
        standard_error_budget
    ) * float(stylized_standard_errors["excess_kurtosis_error_rms"])
    axes[1, 1].axhline(kurtosis_allowance, color="black", linestyle=":", linewidth=1.0)
    axes[1, 0].set_xlabel("Training step")
    axes[1, 1].set_xlabel("Training step")
    figure.suptitle(
        "Validation-only checkpoint frontier\n"
        "green = eligible, grey = rejected, star = selected; red dashed = source baseline"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    source = Path(args.source_run_dir)
    run_dir = ensure_run_dir(
        args.run_dir or default_run_dir(prefix="heston_dt_specific_entropy_shadow_frontier")
    )
    checkpoint_dir = ensure_run_dir(run_dir / "validation_checkpoints")

    (
        source_model,
        candidate_model,
        training,
        stored_config,
        heston,
        target_paths,
        _source_heldout_paths,
        joint_discrepancy,
        mapped,
    ) = _build_models(source=source, device=torch.device(args.device), args=args)
    regime_discrepancy = next(
        (
            block.discrepancy
            for block in tuple(candidate_model.discrepancy.blocks)
            if block.name == "volatility_law_prefix_regime_future_rv"
        ),
        None,
    )
    initialization = str(args.initialization)
    if initialization == "source":
        candidate_model.load_checkpoint_state(source_model.checkpoint_state())
    elif initialization == "frontier":
        frontier_checkpoint = torch.load(
            Path(args.frontier_run_dir) / "selected_model_checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(frontier_checkpoint, dict):
            raise ValueError("frontier selected checkpoint must contain a dictionary")
        candidate_model.load_checkpoint_state(frontier_checkpoint)
        source_model = copy.deepcopy(candidate_model)
    marginal_build = None
    if float(args.dated_marginal_weight) > 0.0:
        marginal_build = build_dated_volatility_marginals(
            base_discrepancy=candidate_model.discrepancy,
            target_paths=target_paths.to(
                device=candidate_model.device,
                dtype=candidate_model.dtype,
            ),
            grid=candidate_model.grid,
            observed_only=bool(training.observed_only),
            total_weight=float(args.dated_marginal_weight),
            endpoint_spacing_full_grid=int(
                args.dated_marginal_endpoint_spacing
            ),
            windows_full_grid=tuple(
                int(value) for value in args.dated_marginal_windows
            ),
        )
        candidate_model.discrepancy = marginal_build.augmented_discrepancy
    source_grad_clip_norm = training.grad_clip_norm
    training = replace(
        training,
        grad_clip_norm=_resolve_grad_clip_norm(
            str(args.grad_clip_norm),
            source_value=source_grad_clip_norm,
        ),
    )
    nested_training = DiscreteMPNestedTrainingConfig(
        outer_steps=int(args.nested_outer_steps),
        inner_steps=int(args.nested_inner_steps),
        outer_batch_size=int(args.nested_outer_batch_size),
        outer_target_batch_size=int(args.nested_outer_target_batch_size),
        inner_batch_size=int(args.nested_inner_batch_size),
        fixed_point_probe_paths=int(args.nested_fixed_point_probe_paths),
        log_every_outer=1,
        outer_objective_line_search=bool(
            args.nested_outer_objective_line_search
        ),
        outer_relaxation_mode=str(args.nested_outer_relaxation_mode),
        outer_max_backtracks=int(args.nested_outer_max_backtracks),
        outer_backtrack_factor=float(args.nested_outer_backtrack_factor),
        outer_objective_tolerance=float(args.nested_outer_objective_tolerance),
        outer_block_coordinate_passes=int(
            args.nested_outer_block_coordinate_passes
        ),
        outer_block_trust_fraction=float(
            args.nested_outer_block_trust_fraction
        ),
        seed=int(args.seed),
    )
    validation_reference = _load_tensor(Path(args.validation_paths))
    if tuple(validation_reference.shape[1:]) != tuple(target_paths.shape[1:]):
        raise ValueError("validation paths must have the source target-law shape")
    if int(validation_reference.shape[0]) < int(args.validation_real_paths):
        raise ValueError("validation artifact has fewer paths than validation-real-paths")
    validation_real = validation_reference[: int(args.validation_real_paths)]
    lags = tuple(value for value in (1, 2, 5, 10, 20) if value < int(heston.num_steps))
    x0 = float(heston.x0)
    validation_bank_seed = _stream_seed(int(args.seed), 80_000)

    print("sampling and calibrating the frozen source validation baseline", flush=True)
    baseline_validation_bank = _sample_bank(
        source_model,
        num_paths=int(args.validation_bank_paths),
        seed=int(validation_bank_seed),
        x0=x0,
        batch_size=int(args.sample_batch_size),
    )
    baseline_validation_result = _shadow(
        real_paths=validation_real,
        generated_paths=baseline_validation_bank,
        args=args,
    )
    baseline_validation_shadow = _metric_subset(baseline_validation_result)
    baseline_validation_stylized = _stylized(
        baseline_validation_bank,
        validation_reference,
        lags=lags,
        swd_projections=int(args.swd_projections),
    )
    baseline_validation_bootstrap, validation_shadow_standard_errors = (
        _shadow_standard_errors(
            baseline_validation_result,
            args=args,
            seed=int(args.bootstrap_seed),
        )
    )
    validation_stylized_standard_errors = _stylized_standard_errors(
        baseline_validation_bank,
        validation_reference,
        lags=lags,
        args=args,
        seed=_stream_seed(int(args.bootstrap_seed), 1),
    )
    torch.save(
        baseline_validation_bank,
        run_dir / "baseline_validation_generated_paths.pt",
    )

    selection_history: list[dict[str, object]] = []

    def validate_checkpoint(model, train_metrics: dict[str, float]) -> None:
        step = int(train_metrics["step"])
        if (
            str(args.solver) == "online"
            and step % int(args.eval_every) != 0
            and step != int(training.num_steps)
        ):
            return
        bank = _sample_bank(
            model,
            num_paths=int(args.validation_bank_paths),
            seed=int(validation_bank_seed),
            x0=x0,
            batch_size=int(args.sample_batch_size),
        )
        shadow_result = _shadow(
            real_paths=validation_real,
            generated_paths=bank,
            args=args,
        )
        shadow_metrics = _metric_subset(shadow_result)
        stylized_metrics = _stylized(
            bank,
            validation_reference,
            lags=lags,
            swd_projections=int(args.swd_projections),
        )
        decision = _decision(
            candidate_shadowing=shadow_metrics,
            baseline_shadowing=baseline_validation_shadow,
            candidate_stylized=stylized_metrics,
            baseline_stylized=baseline_validation_stylized,
            shadowing_standard_errors=validation_shadow_standard_errors,
            stylized_standard_errors=validation_stylized_standard_errors,
            args=args,
        )
        stem = f"step_{step:04d}"
        checkpoint_path = checkpoint_dir / f"{stem}_checkpoint.pt"
        bank_path = checkpoint_dir / f"{stem}_validation_paths.pt"
        torch.save(model.checkpoint_state(), checkpoint_path)
        torch.save(bank, bank_path)
        row: dict[str, object] = {
            "step": step,
            "shadowing": shadow_metrics,
            "stylized": stylized_metrics,
            "decision": decision,
            "checkpoint": str(checkpoint_path),
            "validation_generated_paths": str(bank_path),
            "training_joint_weighted_value": float(
                train_metrics.get(
                    "volatility_law_joint_prefix_future_rv_weighted_value",
                    float("nan"),
                )
            ),
            "training_joint_future_rv_std_ratio": float(
                train_metrics.get(
                    "volatility_law_joint_prefix_future_rv_"
                    "joint_prefix_future_volatility_future_rv_std_ratio",
                    float("nan"),
                )
            ),
            "training_regime_weighted_value": float(
                train_metrics.get(
                    "volatility_law_prefix_regime_future_rv_weighted_value",
                    float("nan"),
                )
            ),
            "training_regime_low_future_z_std": float(
                train_metrics.get(
                    "volatility_law_prefix_regime_future_rv_"
                    "prefix_regime_future_volatility_claims_low_future_z_std",
                    float("nan"),
                )
            ),
            "training_regime_middle_future_z_std": float(
                train_metrics.get(
                    "volatility_law_prefix_regime_future_rv_"
                    "prefix_regime_future_volatility_claims_middle_future_z_std",
                    float("nan"),
                )
            ),
            "training_regime_high_future_z_std": float(
                train_metrics.get(
                    "volatility_law_prefix_regime_future_rv_"
                    "prefix_regime_future_volatility_claims_high_future_z_std",
                    float("nan"),
                )
            ),
            "training_fixed_adjoint_prediction_target_rms_ratio": float(
                train_metrics.get(
                    "fixed_adjoint_prediction_target_rms_ratio",
                    float("nan"),
                )
            ),
            "training_fixed_noise_adjoint_prediction_target_rms_ratio": float(
                train_metrics.get(
                    "fixed_noise_adjoint_prediction_target_rms_ratio",
                    float("nan"),
                )
            ),
            "training_inner_loss_ratio": float(
                train_metrics.get("inner_loss_ratio", float("nan"))
            ),
            "training_fixed_point_path_rms_change": float(
                train_metrics.get("fixed_point_path_rms_change", float("nan"))
            ),
            "training_candidate_fixed_adjoint_prediction_target_rms_ratio": float(
                train_metrics.get(
                    "candidate_fixed_adjoint_prediction_target_rms_ratio",
                    float("nan"),
                )
            ),
            "training_candidate_fixed_noise_adjoint_prediction_target_rms_ratio": float(
                train_metrics.get(
                    "candidate_fixed_noise_adjoint_prediction_target_rms_ratio",
                    float("nan"),
                )
            ),
            "training_accepted_fixed_loss_ratio": float(
                train_metrics.get("accepted_fixed_loss_ratio", float("nan"))
            ),
            "training_outer_objective_before": float(
                train_metrics.get("outer_objective_before", float("nan"))
            ),
            "training_outer_candidate_objective": float(
                train_metrics.get("outer_candidate_objective", float("nan"))
            ),
            "training_outer_accepted_objective": float(
                train_metrics.get("outer_accepted_objective", float("nan"))
            ),
            "training_outer_candidate_discrepancy_objective": float(
                train_metrics.get(
                    "outer_candidate_discrepancy_objective",
                    float("nan"),
                )
            ),
            "training_outer_candidate_running_cost": float(
                train_metrics.get("outer_candidate_running_cost", float("nan"))
            ),
            "training_outer_relaxation": float(
                train_metrics.get("outer_relaxation", float("nan"))
            ),
            "training_outer_relaxation_p": float(
                train_metrics.get("outer_relaxation_p", float("nan"))
            ),
            "training_outer_relaxation_r": float(
                train_metrics.get("outer_relaxation_r", float("nan"))
            ),
            "training_outer_relaxation_shared": float(
                train_metrics.get("outer_relaxation_shared", float("nan"))
            ),
            "training_outer_selected_relaxation_p": float(
                train_metrics.get("outer_selected_relaxation_p", float("nan"))
            ),
            "training_outer_selected_relaxation_r": float(
                train_metrics.get("outer_selected_relaxation_r", float("nan"))
            ),
            "training_outer_selected_relaxation_shared": float(
                train_metrics.get(
                    "outer_selected_relaxation_shared",
                    float("nan"),
                )
            ),
            "training_outer_block_trust_fraction": float(
                train_metrics.get("outer_block_trust_fraction", float("nan"))
            ),
            "training_outer_block_search_evaluations": float(
                train_metrics.get(
                    "outer_block_search_evaluations",
                    float("nan"),
                )
            ),
            "training_outer_backtracks": float(
                train_metrics.get("outer_backtracks", float("nan"))
            ),
            "training_multi_marginal_total": float(
                train_metrics.get("multi_marginal_total", float("nan"))
            ),
        }
        selection_history.append(row)
        failed = [
            name
            for name, passed in (
                ("rv_crps", decision["rv_crps_check"]),
                ("coverage", decision["coverage_check"]),
                ("return_crps", decision["return_crps_check"]),
                ("swd", decision["swd_check"]),
            )
            if not bool(passed)
        ]
        failed.extend(
            name
            for name, passed in dict(decision["stylized_checks"]).items()
            if not bool(passed)
        )
        print(
            "validation step={step} rv_crps={crps:.6f} rv_cov90={coverage:.4f} "
            "return_crps={return_crps:.6f} kurtosis_error={kurtosis:.4f} "
            "outer_relax={outer_relax:.6f} "
            "blocks=({outer_p:.6f},{outer_r:.6f},{outer_shared:.6f}) "
            "eligible={eligible} failed={failed}".format(
                step=step,
                crps=float(shadow_metrics["realized_volatility_crps"]),
                coverage=float(shadow_metrics["realized_volatility_coverage_90"]),
                return_crps=float(shadow_metrics["cumulative_return_crps"]),
                kurtosis=float(stylized_metrics["excess_kurtosis_error_rms"]),
                outer_relax=float(train_metrics.get("outer_relaxation", float("nan"))),
                outer_p=float(
                    train_metrics.get("outer_relaxation_p", float("nan"))
                ),
                outer_r=float(
                    train_metrics.get("outer_relaxation_r", float("nan"))
                ),
                outer_shared=float(
                    train_metrics.get("outer_relaxation_shared", float("nan"))
                ),
                eligible=bool(decision["pass"]),
                failed=",".join(failed) if failed else "none",
            ),
            flush=True,
        )

    print(
        "training the {objective} trajectory with the {solver} solver and "
        "auditing every checkpoint".format(
            objective=(
                "joint-MMD plus dated-marginal"
                if marginal_build is not None
                else "joint-MMD"
            ),
            solver=args.solver,
        ),
        flush=True,
    )
    if str(args.solver) == "nested":
        fit_result = candidate_model.fit_nested(
            target_paths,
            training=training,
            nested=nested_training,
            log_callback=validate_checkpoint,
        )
    else:
        fit_result = candidate_model.fit(
            target_paths,
            training=training,
            log_callback=validate_checkpoint,
        )
    write_history_jsonl(run_dir / "training_history.jsonl", fit_result.history)
    torch.save(candidate_model.checkpoint_state(), run_dir / "terminal_model_checkpoint.pt")
    if not selection_history:
        raise RuntimeError("the validation callback produced no checkpoint records")

    selected_step = select_shadow_frontier_checkpoint(
        selection_history,
        objective=str(args.selection_objective),
        nominal_coverage=0.90,
    )
    selected_record = next(
        (record for record in selection_history if int(record["step"]) == selected_step),
        None,
    )
    if selected_step == 0:
        selected_model = source_model
        selected_validation_bank = baseline_validation_bank
        selected_validation_shadow = baseline_validation_shadow
        selected_validation_stylized = baseline_validation_stylized
        selected_validation_decision: dict[str, object] = {
            "pass": False,
            "reason": "no post-update checkpoint satisfied every validation guard",
        }
        selected_checkpoint = source_model.checkpoint_state()
    else:
        if selected_record is None:
            raise RuntimeError("selected validation record is missing")
        selected_model = candidate_model
        checkpoint = torch.load(
            Path(str(selected_record["checkpoint"])),
            map_location="cpu",
            weights_only=True,
        )
        selected_model.load_checkpoint_state(checkpoint)
        selected_validation_bank = _load_tensor(
            Path(str(selected_record["validation_generated_paths"]))
        )
        selected_validation_shadow = dict(selected_record["shadowing"])
        selected_validation_stylized = dict(selected_record["stylized"])
        selected_validation_decision = dict(selected_record["decision"])
        selected_checkpoint = checkpoint
    torch.save(selected_checkpoint, run_dir / "selected_model_checkpoint.pt")
    torch.save(
        selected_validation_bank,
        run_dir / "selected_validation_generated_paths.pt",
    )
    write_json(
        run_dir / "selection_history.json",
        {
            "records": selection_history,
            "selected_step": int(selected_step),
            "selection_objective": str(args.selection_objective),
            "nominal_coverage": 0.90,
        },
    )
    _plot_frontier(
        selection_history,
        baseline_shadowing=baseline_validation_shadow,
        baseline_stylized=baseline_validation_stylized,
        stylized_standard_errors=validation_stylized_standard_errors,
        selected_step=int(selected_step),
        standard_error_budget=float(args.standard_error_budget),
        output_path=run_dir / "selection_frontier.png",
    )

    # These seeds and Heston parameters are fixed above, but the law is not
    # simulated until selection is irreversibly complete.
    fresh_reference_seed = _stream_seed(int(args.seed), 180_000)
    fresh_bank_seed = _stream_seed(int(args.seed), 180_001)
    print(
        f"selected step={selected_step}; now opening newly simulated Heston test law",
        flush=True,
    )
    fresh_reference = simulate_heston_log_paths(
        num_paths=int(args.fresh_test_reference_paths),
        config=heston,
        seed=int(fresh_reference_seed),
        device="cpu",
        dtype=target_paths.dtype,
    )
    baseline_fresh_bank = _sample_bank(
        source_model,
        num_paths=int(args.fresh_test_bank_paths),
        seed=int(fresh_bank_seed),
        x0=x0,
        batch_size=int(args.sample_batch_size),
    )
    selected_fresh_bank = (
        baseline_fresh_bank.clone()
        if selected_step == 0
        else _sample_bank(
            selected_model,
            num_paths=int(args.fresh_test_bank_paths),
            seed=int(fresh_bank_seed),
            x0=x0,
            batch_size=int(args.sample_batch_size),
        )
    )
    fresh_real = fresh_reference[: int(args.fresh_test_real_paths)]
    baseline_fresh_result = _shadow(
        real_paths=fresh_real,
        generated_paths=baseline_fresh_bank,
        args=args,
    )
    selected_fresh_result = _shadow(
        real_paths=fresh_real,
        generated_paths=selected_fresh_bank,
        args=args,
    )
    baseline_fresh_shadow = _metric_subset(baseline_fresh_result)
    selected_fresh_shadow = _metric_subset(selected_fresh_result)
    baseline_fresh_stylized = _stylized(
        baseline_fresh_bank,
        fresh_reference,
        lags=lags,
        swd_projections=int(args.swd_projections),
    )
    selected_fresh_stylized = _stylized(
        selected_fresh_bank,
        fresh_reference,
        lags=lags,
        swd_projections=int(args.swd_projections),
    )
    baseline_fresh_bootstrap, fresh_shadow_standard_errors = _shadow_standard_errors(
        baseline_fresh_result,
        args=args,
        seed=_stream_seed(int(args.bootstrap_seed), 2),
    )
    fresh_stylized_standard_errors = _stylized_standard_errors(
        baseline_fresh_bank,
        fresh_reference,
        lags=lags,
        args=args,
        seed=_stream_seed(int(args.bootstrap_seed), 3),
    )
    fresh_decision = _decision(
        candidate_shadowing=selected_fresh_shadow,
        baseline_shadowing=baseline_fresh_shadow,
        candidate_stylized=selected_fresh_stylized,
        baseline_stylized=baseline_fresh_stylized,
        shadowing_standard_errors=fresh_shadow_standard_errors,
        stylized_standard_errors=fresh_stylized_standard_errors,
        args=args,
    )
    paired_fresh = paired_bootstrap_shadowing_difference(
        selected_fresh_result,
        baseline_fresh_result,
        num_replicates=int(args.shadow_bootstrap_replicates),
        confidence_level=float(args.bootstrap_confidence),
        seed=_stream_seed(int(args.bootstrap_seed), 4),
    )

    selection_rule = (
        "minimum validation realized-volatility CRPS among checkpoints passing "
        "every predeclared guard; otherwise source baseline (step 0)"
        if str(args.selection_objective) == "rv_crps"
        else (
            "minimum absolute validation realized-volatility coverage error from "
            "90% among checkpoints passing every predeclared guard, tie-broken by "
            "lower realized-volatility CRPS then earlier step; otherwise source "
            "baseline (step 0)"
        )
    )

    payload: dict[str, object] = {
        "run_dir": str(run_dir),
        "source_run_dir": str(source),
        "validation_paths": str(args.validation_paths),
        "protocol": {
            "selection_uses_validation_only": True,
            "source_heldout_artifact_used": False,
            "fresh_test_simulated_only_after_selection": True,
            "selection_rule": selection_rule,
            "selection_objective": str(args.selection_objective),
            "nominal_coverage": 0.90,
            "candidate_objective_unchanged": bool(marginal_build is None),
            "neural_adjoint_loss_unchanged": True,
            "specific_entropy_running_cost_unchanged": True,
            "preexisting_terminal_objective_unchanged": True,
            "dated_marginal_costs_added": bool(marginal_build is not None),
            "training_solver": str(args.solver),
            "online_training_trajectory_unchanged": bool(str(args.solver) == "online"),
        },
        "selected_step": int(selected_step),
        "selected_source_baseline": bool(selected_step == 0),
        "config": {
            "seed": int(args.seed),
            "initialization": str(args.initialization),
            "frontier_run_dir": (
                str(args.frontier_run_dir)
                if str(args.initialization) == "frontier"
                else None
            ),
            "heston": asdict(heston),
            "training": asdict(training),
            "nested_training": asdict(nested_training),
            "source_grad_clip_norm": source_grad_clip_norm,
            "grad_clip_override": str(args.grad_clip_norm),
            "joint_weight": float(args.joint_weight),
            "regime_weight": float(args.regime_weight),
            "regime_prefix_window_full_grid": int(args.regime_prefix_window),
            "regime_gate_temperature_fraction": float(
                args.regime_gate_temperature_fraction
            ),
            "regime_tail_temperature": float(args.regime_tail_temperature),
            "regime_conditional_moment_weight": float(
                args.regime_conditional_moment_weight
            ),
            "dated_marginals": (
                None if marginal_build is None else marginal_build.metadata
            ),
            "split_step_full_grid": int(args.split_step),
            "future_steps_full_grid": int(args.future_steps),
            "prefix_windows_full_grid": [int(value) for value in args.prefix_windows],
            "recent_return_window_full_grid": int(args.recent_return_window),
            "discrepancy_grid_mapping": {
                "split_index": int(mapped["split_index"]),
                "future_horizon": int(mapped["future_horizon"]),
                "prefix_windows": [int(value) for value in mapped["prefix_windows"]],
                "recent_return_window": int(mapped["recent_return_window"]),
                "regime_prefix_window": (
                    None
                    if mapped["regime_prefix_window"] is None
                    else int(mapped["regime_prefix_window"])
                ),
            },
            "bandwidths": [float(value) for value in args.bandwidths],
            "eval_every": int(args.eval_every),
            "validation_real_paths": int(validation_real.shape[0]),
            "validation_reference_paths": int(validation_reference.shape[0]),
            "validation_bank_paths": int(args.validation_bank_paths),
            "fresh_test_real_paths": int(fresh_real.shape[0]),
            "fresh_test_reference_paths": int(fresh_reference.shape[0]),
            "fresh_test_bank_paths": int(selected_fresh_bank.shape[0]),
            "top_k": int(args.top_k),
            "shadow_search": str(args.shadow_search),
            "standard_error_budget": float(args.standard_error_budget),
            "swd_factor_limit": float(args.swd_factor_limit),
            "selection_objective": str(args.selection_objective),
            "shadow_bootstrap_replicates": int(args.shadow_bootstrap_replicates),
            "stylized_bootstrap_replicates": int(args.stylized_bootstrap_replicates),
            "stylized_bootstrap_paths": int(args.stylized_bootstrap_paths),
            "bootstrap_confidence": float(args.bootstrap_confidence),
            "validation_bank_seed": int(validation_bank_seed),
            "fresh_reference_seed": int(fresh_reference_seed),
            "fresh_bank_seed": int(fresh_bank_seed),
        },
        "validation_baseline_uncertainty": {
            "shadowing_bootstrap": baseline_validation_bootstrap,
            "stylized_error_standard_errors": validation_stylized_standard_errors,
        },
        "validation": {
            "baseline": {
                "shadowing": baseline_validation_shadow,
                "stylized": baseline_validation_stylized,
            },
            "selected": {
                "shadowing": selected_validation_shadow,
                "stylized": selected_validation_stylized,
                "decision": selected_validation_decision,
            },
        },
        "selection_history": selection_history,
        "fresh_test_baseline_uncertainty": {
            "shadowing_bootstrap": baseline_fresh_bootstrap,
            "stylized_error_standard_errors": fresh_stylized_standard_errors,
        },
        "fresh_test": {
            "baseline": {
                "shadowing": baseline_fresh_shadow,
                "stylized": baseline_fresh_stylized,
            },
            "selected": {
                "shadowing": selected_fresh_shadow,
                "stylized": selected_fresh_stylized,
            },
            "decision": fresh_decision,
            "paired_difference_definition": "selected - source baseline",
            "paired_bootstrap": paired_fresh,
            "joint_discrepancy": {
                "baseline": _joint_metrics(
                    joint_discrepancy,
                    generated_paths=baseline_fresh_bank,
                    target_paths=fresh_reference,
                    grid=source_model.grid,
                ),
                "selected": _joint_metrics(
                    joint_discrepancy,
                    generated_paths=selected_fresh_bank,
                    target_paths=fresh_reference,
                    grid=selected_model.grid,
                ),
            },
            "regime_discrepancy": (
                None
                if regime_discrepancy is None
                else {
                    "baseline": _joint_metrics(
                        regime_discrepancy,
                        generated_paths=baseline_fresh_bank,
                        target_paths=fresh_reference,
                        grid=source_model.grid,
                    ),
                    "selected": _joint_metrics(
                        regime_discrepancy,
                        generated_paths=selected_fresh_bank,
                        target_paths=fresh_reference,
                        grid=selected_model.grid,
                    ),
                }
            ),
            "dated_marginal_discrepancy": (
                None
                if marginal_build is None
                else {
                    "baseline": _joint_metrics(
                        marginal_build.marginal_only_discrepancy,
                        generated_paths=baseline_fresh_bank,
                        target_paths=fresh_reference,
                        grid=source_model.grid,
                    ),
                    "selected": _joint_metrics(
                        marginal_build.marginal_only_discrepancy,
                        generated_paths=selected_fresh_bank,
                        target_paths=fresh_reference,
                        grid=selected_model.grid,
                    ),
                }
            ),
        },
        "fit_final": fit_result.final_metrics,
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(fresh_reference, run_dir / "fresh_test_reference_paths.pt")
    torch.save(baseline_fresh_bank, run_dir / "baseline_fresh_test_generated_paths.pt")
    torch.save(selected_fresh_bank, run_dir / "selected_fresh_test_generated_paths.pt")
    torch.save(
        {
            "baseline_validation": baseline_validation_result.path_metrics,
            "baseline_fresh_test": baseline_fresh_result.path_metrics,
            "selected_fresh_test": selected_fresh_result.path_metrics,
        },
        run_dir / "shadowing_per_path_metrics.pt",
    )
    print(f"wrote shadow-frontier audit to {run_dir}", flush=True)
    print(
        "fresh test rv_crps {base_crps:.6f}->{selected_crps:.6f}; "
        "rv_cov90 {base_cov:.4f}->{selected_cov:.4f}; return_crps "
        "{base_return:.6f}->{selected_return:.6f}; full_gate={gate}".format(
            base_crps=float(baseline_fresh_shadow["realized_volatility_crps"]),
            selected_crps=float(selected_fresh_shadow["realized_volatility_crps"]),
            base_cov=float(baseline_fresh_shadow["realized_volatility_coverage_90"]),
            selected_cov=float(selected_fresh_shadow["realized_volatility_coverage_90"]),
            base_return=float(baseline_fresh_shadow["cumulative_return_crps"]),
            selected_return=float(selected_fresh_shadow["cumulative_return_crps"]),
            gate=bool(fresh_decision["pass"]),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
