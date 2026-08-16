from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    CausalPathFeatureRidgeConditionalExpectation,
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteMPNestedTrainingConfig,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
    calibrate_shrunk_causal_reference_kernel,
    fit_causal_volatility_reference_kernel,
    fit_guyon_lekeufack_likelihood_reference_kernel,
    fit_guyon_lekeufack_price_realized_volatility_reference_kernel,
    fit_local_gaussian_reference_kernel,
)
from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    SpecificEntropyDiagonalControl,
    VolatilityOnlySpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from deep_mkv_gen_path_dt.noise import sample_standard_normals  # noqa: E402
from deep_mkv_gen_path_dt.model import FitResult  # noqa: E402
from path_dt_experiments.discrepancies import (  # noqa: E402
    build_heston_discrepancy,
    fit_fixed_target_joint_prefix_future_volatility_mmd,
)
from path_dt_experiments.drawdown_memory import (  # noqa: E402
    FixedTargetResidualDrawdownMemoryMoments,
    fit_fixed_target_residual_drawdown_memory_moments,
)
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.causal_transformer_adjoint import (  # noqa: E402
    CachedCausalTransformerAdjointNetwork,
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
from path_dt_experiments.extragradient_solver import (  # noqa: E402
    SafeguardedExtragradientConfig,
    fit_safeguarded_extragradient,
)
from path_dt_experiments.continuation_solver import (  # noqa: E402
    DiscrepancyContinuationConfig,
    fit_discrepancy_continuation,
)
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    network_sha256,
    stable_sha256,
)


ARM_REFERENCE_CE = "reference_ce"
ARM_REFERENCE_CE_NESTED = "reference_ce_nested"
ARM_RESIDUAL_ONLINE = "reference_ce_residual_online"
ARM_RESIDUAL_NESTED = "reference_ce_residual_nested"
ARM_GENERIC_CE_ONLINE = "generic_ce_online"
ARM_GENERIC_CE_ONLINE_VOLATILITY_ONLY = "generic_ce_online_volatility_only"
ARM_GENERIC_CE_ONLINE_SOURCE_VOLATILITY_ONLY = (
    "generic_ce_online_source_volatility_only"
)
ARM_GENERIC_CE_NESTED = "generic_ce_nested"
ARM_DIRECT_BPTT = "direct_bptt"
ARM_GENERIC_CE_PROXIMAL = "generic_ce_proximal_fixed_point"
ARM_GENERIC_CE_EXTRAGRADIENT = "generic_ce_safeguarded_extragradient"
ARM_GENERIC_CE_CONTINUATION = "generic_ce_continuation_extragradient"
ARM_GENERIC_REFERENCE_ONLY = "generic_reference_only"
ARM_GENERIC_CE_NESTED_REFERENCE = "generic_ce_nested_reference"
ARM_GENERIC_CE_NESTED_VOLATILITY_ONLY = "generic_ce_nested_volatility_only"
ARM_DIRECT_BPTT_VOLATILITY_ONLY = "direct_bptt_volatility_only"
ARMS = (
    ARM_REFERENCE_CE,
    ARM_REFERENCE_CE_NESTED,
    ARM_RESIDUAL_ONLINE,
    ARM_RESIDUAL_NESTED,
    ARM_GENERIC_CE_ONLINE,
    ARM_GENERIC_CE_NESTED,
    ARM_DIRECT_BPTT,
)
AVAILABLE_ARMS = (
    *ARMS,
    ARM_GENERIC_CE_ONLINE_VOLATILITY_ONLY,
    ARM_GENERIC_CE_ONLINE_SOURCE_VOLATILITY_ONLY,
    ARM_GENERIC_CE_PROXIMAL,
    ARM_GENERIC_CE_EXTRAGRADIENT,
    ARM_GENERIC_CE_CONTINUATION,
    ARM_GENERIC_REFERENCE_ONLY,
    ARM_GENERIC_CE_NESTED_REFERENCE,
    ARM_GENERIC_CE_NESTED_VOLATILITY_ONLY,
    ARM_DIRECT_BPTT_VOLATILITY_ONLY,
)

EXACT_REFERENCE_ARMS = {
    ARM_GENERIC_CE_ONLINE_VOLATILITY_ONLY,
    ARM_GENERIC_CE_ONLINE_SOURCE_VOLATILITY_ONLY,
    ARM_GENERIC_REFERENCE_ONLY,
    ARM_GENERIC_CE_NESTED_REFERENCE,
    ARM_GENERIC_CE_NESTED_VOLATILITY_ONLY,
    ARM_DIRECT_BPTT_VOLATILITY_ONLY,
}

VOLATILITY_ONLY_ARMS = {
    ARM_GENERIC_CE_ONLINE_VOLATILITY_ONLY,
    ARM_GENERIC_CE_ONLINE_SOURCE_VOLATILITY_ONLY,
    ARM_GENERIC_CE_NESTED_VOLATILITY_ONLY,
    ARM_DIRECT_BPTT_VOLATILITY_ONLY,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled Deep-MKV-Gen ablation for delayed drawdown-memory "
            "volatility: observable causal reference/CE basis, cross-fitted "
            "residual moment, and nested forward-backward solver."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "drawdown_memory_seed0_comparison_20260730"
            / "data"
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help=(
            "Optional frozen source run containing source_model_checkpoint.pt; "
            "when omitted, fit the source stage in run-dir."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--arms", nargs="+", choices=AVAILABLE_ARMS, default=ARMS
    )
    parser.add_argument(
        "--max-training-paths",
        type=int,
        default=0,
        help="Optional deterministic prefix for smoke tests; zero uses all paths.",
    )
    parser.add_argument("--source-training-steps", type=int, default=2_000)
    parser.add_argument(
        "--online-checkpoint-steps",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Validation-only intermediate checkpoints for the online "
            "source-stage volatility-only arm. Values must be increasing, "
            "positive, and smaller than source-training-steps."
        ),
    )
    parser.add_argument("--joint-training-steps", type=int, default=1_600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--bank-size", type=int, default=8_192)
    parser.add_argument("--sample-batch-size", type=int, default=8_192)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument(
        "--adjoint-network",
        choices=("gru", "causal_transformer"),
        default="gru",
        help="Sequential network used to store conditional adjoint signals.",
    )
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--adjoint-weight", type=float, default=1.0)
    parser.add_argument("--adjoint-noise-weight", type=float, default=1.0)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help=(
            "Use a positive value to enable optimizer gradient clipping. "
            "The ablation defaults to the previously verified finite uncapped dynamics."
        ),
    )
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.6)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--lambda-scale", type=float, default=50.0)
    parser.add_argument("--kappa-scale", type=float, default=50.0)
    parser.add_argument(
        "--reference-kind",
        choices=(
            "causal",
            "guyon_lekeufack_structural_likelihood",
            "guyon_lekeufack_price_realized_volatility",
        ),
        default="causal",
        help=(
            "Reference volatility family. The structural-likelihood Guyon option "
            "fits conditional return likelihood. The price-realized-volatility "
            "option fits the published empirical factor regression to forward "
            "realized volatility constructed from prices alone."
        ),
    )
    parser.add_argument(
        "--guyon-realized-volatility-window",
        type=int,
        default=20,
        help="Forward return window used by the price-only empirical Guyon calibration.",
    )
    parser.add_argument(
        "--generic-reference",
        action="store_true",
        help=(
            "Remove thresholded and lagged drawdown-memory coordinates from "
            "both the fitted local reference and the source-stage projection."
        ),
    )
    parser.add_argument(
        "--calibrate-reference",
        action="store_true",
        help=(
            "Apply the predeclared train-only causal/anchor calibration grid "
            "used by the real-data protocol. No validation or test path is used."
        ),
    )
    parser.add_argument(
        "--scale-temporal-features",
        action="store_true",
        help=(
            "Scale every time-indexed reference, projection, and discrepancy "
            "window by sequence_length/128. Used by the resolution study so "
            "all arms see the same physical-time information."
        ),
    )
    parser.add_argument("--reference-crossfit-folds", type=int, default=3)
    parser.add_argument(
        "--reference-variance-ridges",
        type=float,
        nargs="+",
        default=(1e-5, 1e-4, 1e-3),
    )
    parser.add_argument(
        "--drawdown-threshold-quantiles",
        type=float,
        nargs="+",
        default=(0.35, 0.60, 0.80),
    )
    parser.add_argument(
        "--drawdown-memory-half-lives",
        type=int,
        nargs="+",
        default=(20, 60),
    )
    parser.add_argument(
        "--drawdown-memory-lags",
        type=int,
        nargs="+",
        default=(8, 16, 24, 32, 40, 48),
        help=(
            "Train-only candidate lags for observable drawdown-memory states. "
            "All candidates enter the shared reference/CE ridge basis."
        ),
    )
    parser.add_argument("--drawdown-gate-temperature", type=float, default=0.005)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--residual-crossfit-folds", type=int, default=5)
    parser.add_argument("--nested-outer-steps", type=int, default=8)
    parser.add_argument("--nested-inner-steps", type=int, default=200)
    parser.add_argument("--nested-outer-batch-size", type=int, default=1024)
    parser.add_argument("--nested-outer-target-batch-size", type=int, default=1024)
    parser.add_argument(
        "--nested-target-estimator",
        choices=("projected_score", "nested_branches"),
        default="projected_score",
    )
    parser.add_argument("--nested-conditional-branches", type=int, default=16)
    parser.add_argument(
        "--nested-population-batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--nested-line-search-batch-size",
        type=int,
        default=None,
    )
    parser.add_argument("--nested-inner-batch-size", type=int, default=256)
    parser.add_argument("--nested-trust-fraction", type=float, default=1.0)
    parser.add_argument("--nested-max-backtracks", type=int, default=10)
    parser.add_argument("--proximal-validation-paths", type=int, default=2_048)
    parser.add_argument("--proximal-backward-fit-paths", type=int, default=1_024)
    parser.add_argument(
        "--proximal-backward-validation-paths", type=int, default=512
    )
    parser.add_argument("--proximal-objective-paths", type=int, default=512)
    parser.add_argument("--proximal-selection-banks", type=int, default=4)
    parser.add_argument("--proximal-confirmation-banks", type=int, default=4)
    parser.add_argument("--proximal-inner-max-steps", type=int, default=600)
    parser.add_argument("--proximal-inner-eval-every", type=int, default=25)
    parser.add_argument("--proximal-inner-patience", type=int, default=6)
    parser.add_argument(
        "--proximal-alpha-fraction", type=float, default=0.03125
    )
    parser.add_argument(
        "--proximal-log-sigma-fraction", type=float, default=0.001953125
    )
    parser.add_argument("--extragradient-alpha-radius", type=float, default=0.1)
    parser.add_argument(
        "--extragradient-log-sigma-radius", type=float, default=0.005
    )
    parser.add_argument(
        "--extragradient-r-reliability-floor", type=float, default=0.02
    )
    parser.add_argument(
        "--extragradient-backtrack-scales",
        type=float,
        nargs="+",
        default=(1.0, 0.5, 0.25),
    )
    parser.add_argument(
        "--extragradient-minimum-residual-improvement",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--extragradient-maximum-objective-increase",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        "--extragradient-transition-kl-rho",
        type=float,
        default=0.0,
        help=(
            "Positive rho enables the algorithmic conditional transition-KL "
            "proximal proposal; zero disables it."
        ),
    )
    parser.add_argument(
        "--continuation-beta-schedule",
        type=float,
        nargs="+",
        default=(1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0),
    )
    parser.add_argument("--continuation-stage-outer-steps", type=int, default=2)
    parser.add_argument(
        "--continuation-minimum-beta-increment",
        type=float,
        default=1 / 4096,
    )
    parser.add_argument("--continuation-maximum-subdivisions", type=int, default=6)
    parser.add_argument(
        "--continuation-minimum-stage-residual-improvement",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--continuation-stationary-residual-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--continuation-minimum-objective-bank-fraction",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--noise-target-control-variate",
        choices=("none", "adjoint"),
        default="none",
    )
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _default_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "runs" / f"drawdown_memory_deep_mkv_ablation_{timestamp}"


def _load_prices(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(f"{path} must contain finite positive price paths")
    return values


def _load_dataset(
    dataset_root: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, object], dict[str, Path]]:
    paths = {
        name: dataset_root / f"{name}.npy"
        for name in ("train", "disc", "test")
    }
    manifest_path = dataset_root / "manifest.json"
    for path in (*paths.values(), manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_prices = _load_prices(paths["train"])
    train_log_paths = np.log(train_prices / train_prices[:, :1])
    target = torch.from_numpy(train_log_paths).unsqueeze(-1).to(
        device=device,
        dtype=dtype,
    )
    return target, manifest, {**paths, "manifest": manifest_path}


def _thresholds_from_training(
    target_paths: torch.Tensor,
    *,
    history_cutoff: int,
    quantiles: tuple[float, ...],
) -> tuple[float, ...]:
    if (
        not quantiles
        or any(not 0.0 < value < 1.0 for value in quantiles)
        or tuple(sorted(set(quantiles))) != quantiles
    ):
        raise ValueError("drawdown threshold quantiles must be strictly increasing in (0, 1)")
    values = target_paths[..., 0]
    prefix = values[:, : int(history_cutoff) + 1]
    drawdown = torch.cummax(prefix, dim=1).values - prefix
    maximum = drawdown.amax(dim=1)
    fitted = torch.quantile(
        maximum,
        maximum.new_tensor(quantiles),
    )
    return tuple(float(value) for value in fitted.detach().cpu().tolist())


def _make_training(
    args: argparse.Namespace,
    *,
    num_steps: int,
    seed: int,
) -> DiscreteMPTrainingConfig:
    clip = (
        None
        if float(args.grad_clip_norm) <= 0.0
        else float(args.grad_clip_norm)
    )
    return DiscreteMPTrainingConfig(
        batch_size=int(args.batch_size),
        target_batch_size=int(args.target_batch_size),
        num_steps=int(num_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        lambda_scale=float(args.lambda_scale),
        adjoint_weight=float(args.adjoint_weight),
        adjoint_noise_weight=float(args.adjoint_noise_weight),
        ce_target_mode="ridge",
        ridge_lambda=float(args.ridge_lambda),
        ce_crossfit_folds=1,
        target_preconditioner="none",
        noise_target_control_variate=str(args.noise_target_control_variate),
        grad_clip_norm=clip,
        log_every=int(args.log_every),
        seed=int(seed),
        observed_only=False,
    )


def _make_nested(
    args: argparse.Namespace,
    *,
    seed: int,
) -> DiscreteMPNestedTrainingConfig:
    return DiscreteMPNestedTrainingConfig(
        outer_steps=int(args.nested_outer_steps),
        inner_steps=int(args.nested_inner_steps),
        outer_batch_size=int(args.nested_outer_batch_size),
        outer_target_batch_size=int(args.nested_outer_target_batch_size),
        outer_target_estimator=str(args.nested_target_estimator),
        outer_conditional_branches=int(args.nested_conditional_branches),
        outer_conditional_antithetic=True,
        outer_population_batch_size=args.nested_population_batch_size,
        outer_line_search_batch_size=args.nested_line_search_batch_size,
        inner_batch_size=int(args.nested_inner_batch_size),
        fixed_point_probe_paths=min(512, int(args.nested_outer_batch_size)),
        log_every_outer=1,
        outer_objective_line_search=True,
        outer_relaxation_mode="adjoint_blocks",
        outer_max_backtracks=int(args.nested_max_backtracks),
        outer_backtrack_factor=0.5,
        outer_objective_tolerance=0.0,
        outer_block_coordinate_passes=2,
        outer_block_trust_fraction=float(args.nested_trust_fraction),
        seed=int(seed),
    )


def _make_proximal_solver(
    args: argparse.Namespace,
    *,
    seed: int,
    num_steps: int,
) -> ValidatedNestedSolverConfig:
    gate_steps = tuple(
        sorted(
            {
                max(0, min(int(num_steps) - 1, int(num_steps) // 2)),
                max(0, min(int(num_steps) - 1, 3 * int(num_steps) // 4)),
            }
        )
    )
    return ValidatedNestedSolverConfig(
        outer_steps=int(args.nested_outer_steps),
        backward_fit_paths=int(args.proximal_backward_fit_paths),
        backward_validation_paths=int(
            args.proximal_backward_validation_paths
        ),
        outer_objective_paths=int(args.proximal_objective_paths),
        outer_selection_banks=int(args.proximal_selection_banks),
        outer_confirmation_banks=int(args.proximal_confirmation_banks),
        target_batch_size=int(args.nested_outer_target_batch_size),
        inner_batch_size=int(args.nested_inner_batch_size),
        inner_max_steps=int(args.proximal_inner_max_steps),
        inner_eval_every=int(args.proximal_inner_eval_every),
        inner_patience=int(args.proximal_inner_patience),
        minimum_relative_improvement=1e-3,
        minimum_outer_relative_improvement=1e-4,
        minimum_confirmation_improvement_fraction=0.75,
        antithetic_noise=True,
        gate_step_indices=gate_steps,
        minimum_r_teacher_correlation=-1.0,
        maximum_r_teacher_relative_rmse=100.0,
        minimum_validation_p_r2=-100.0,
        minimum_validation_r_r2=-100.0,
        outer_relaxations=(0.0, 0.125, 0.25, 0.5, 1.0),
        outer_coordinate_passes=2,
        proximal_alpha_fraction=float(args.proximal_alpha_fraction),
        proximal_log_sigma_fraction=float(
            args.proximal_log_sigma_fraction
        ),
        seed=int(seed),
    )


def _make_extragradient_solver(
    args: argparse.Namespace,
    *,
    seed: int,
) -> SafeguardedExtragradientConfig:
    return SafeguardedExtragradientConfig(
        outer_steps=int(args.nested_outer_steps),
        operator_fit_paths=int(args.proximal_backward_fit_paths),
        operator_validation_paths=int(
            args.proximal_backward_validation_paths
        ),
        operator_target_batch_size=int(args.nested_outer_target_batch_size),
        inner_batch_size=int(args.nested_inner_batch_size),
        inner_max_steps=int(args.proximal_inner_max_steps),
        inner_eval_every=int(args.proximal_inner_eval_every),
        inner_patience=int(args.proximal_inner_patience),
        alpha_radius=float(args.extragradient_alpha_radius),
        log_sigma_radius=float(args.extragradient_log_sigma_radius),
        r_reliability_floor=float(
            args.extragradient_r_reliability_floor
        ),
        backtrack_scales=tuple(
            float(value) for value in args.extragradient_backtrack_scales
        ),
        residual_fit_paths=int(args.proximal_backward_fit_paths),
        residual_validation_paths=int(
            args.proximal_backward_validation_paths
        ),
        objective_paths=int(args.proximal_objective_paths),
        objective_target_batch_size=int(args.nested_outer_target_batch_size),
        objective_banks=int(args.proximal_confirmation_banks),
        minimum_residual_relative_improvement=float(
            args.extragradient_minimum_residual_improvement
        ),
        maximum_objective_relative_increase=float(
            args.extragradient_maximum_objective_increase
        ),
        minimum_objective_nonworsening_fraction=0.5,
        transition_kl_rho=(
            None
            if float(args.extragradient_transition_kl_rho) <= 0.0
            else float(args.extragradient_transition_kl_rho)
        ),
        seed=int(seed),
    )


def _make_continuation_solver(
    args: argparse.Namespace,
    *,
    seed: int,
) -> DiscrepancyContinuationConfig:
    return DiscrepancyContinuationConfig(
        beta_schedule=tuple(
            float(value) for value in args.continuation_beta_schedule
        ),
        stage_outer_steps=int(args.continuation_stage_outer_steps),
        minimum_beta_increment=float(
            args.continuation_minimum_beta_increment
        ),
        maximum_subdivisions=int(args.continuation_maximum_subdivisions),
        minimum_stage_residual_improvement=float(
            args.continuation_minimum_stage_residual_improvement
        ),
        stationary_relative_residual_tolerance=float(
            args.continuation_stationary_residual_tolerance
        ),
        minimum_stage_objective_improvement_fraction=float(
            args.continuation_minimum_objective_bank_fraction
        ),
        maximum_stage_objective_relative_increase=float(
            args.extragradient_maximum_objective_increase
        ),
        seed=int(seed),
    )


def _progress(stage: str):
    def report(_model: DiscreteMPModel, metrics: dict[str, float]) -> None:
        loss = metrics.get(
            "train_total_loss",
            metrics.get(
                "outer_accepted_objective",
                metrics.get(
                    "complete_objective_value",
                    metrics.get("inner_final_total_loss", float("nan")),
                ),
            ),
        )
        grad = metrics.get(
            "grad_norm", metrics.get("inner_grad_norm_mean", float("nan"))
        )
        clip = metrics.get(
            "grad_clip_scale",
            1.0 - float(metrics.get("inner_clip_fraction", float("nan"))),
        )
        print(
            "{stage} step={step:>4.0f} loss={loss:.5g} sigma={sigma:.4f} "
            "grad={grad:.3g} clip={clip:.3f} outer={outer:.0f}".format(
                stage=stage,
                step=float(metrics.get("step", metrics.get("inner_step", 0.0))),
                loss=float(loss),
                sigma=float(metrics.get("sigma_mean", float("nan"))),
                grad=float(grad),
                clip=float(clip),
                outer=float(metrics.get("outer_step", 0.0)),
            ),
            flush=True,
        )

    return report


def _build_model(
    *,
    args: argparse.Namespace,
    grid: DiscreteTimeGrid,
    control: SpecificEntropyDiagonalControl,
    discrepancy: CompositePathFunctionalDiscrepancy,
    ce_estimator: CausalPathFeatureRidgeConditionalExpectation,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> DiscreteMPModel:
    architecture = DiscreteMPArchitectureConfig(
        state_dim=1,
        noise_dim=1,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
    )
    network = None
    init_seed: int | None = int(seed)
    if str(args.adjoint_network) == "causal_transformer":
        if int(args.hidden_dim) % int(args.transformer_heads) != 0:
            raise ValueError("hidden-dim must be divisible by transformer-heads")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            network = CachedCausalTransformerAdjointNetwork(
                state_dim=1,
                adjoint_dim=1,
                noise_adjoint_dim=1,
                hidden_dim=int(args.hidden_dim),
                num_layers=int(args.transformer_layers),
                num_heads=int(args.transformer_heads),
                max_steps=int(grid.num_steps),
                input_mode=str(architecture.adjoint_input_mode),
            )
        init_seed = None
    model = DiscreteMPModel(
        architecture=architecture,
        grid=grid,
        dynamics_step=DriftVolatilityEulerStep(dt=grid.dt),
        control_map=control,
        discrepancy=discrepancy,
        ce_estimator=ce_estimator,
        network=network,
        init_seed=init_seed,
    )
    model.network.to(device=device, dtype=dtype)
    return model


def _fit_direct_objective_control(
    model: DiscreteMPModel,
    target_paths: torch.Tensor,
    *,
    training: DiscreteMPTrainingConfig,
    log_callback,
) -> FitResult:
    """Optimize the unchanged complete objective through the rollout.

    This is the matched non-MP control used to attribute any gain to the
    projected-adjoint solver.  Architecture, initialization, reference,
    admissible control map, running cost, discrepancy, minibatches, noise
    streams, optimizer, and update budget remain fixed.
    """

    model.training = training
    target_paths = target_paths.to(device=model.device, dtype=model.dtype)
    target_count, _ = model._validate_target_paths(target_paths)
    cpu_generator = torch.Generator(device="cpu")
    if training.seed is not None:
        cpu_generator.manual_seed(int(training.seed))
    optimizer = torch.optim.AdamW(
        model.network.parameters(),
        lr=float(training.lr),
        weight_decay=float(training.weight_decay),
    )
    history: list[dict[str, float]] = []
    for step in range(int(training.num_steps)):
        source_indices = model._sample_indices(
            size=target_count,
            count=int(training.batch_size),
            generator=cpu_generator,
        )
        target_indices = model._sample_indices(
            size=target_count,
            count=int(training.target_batch_size),
            generator=cpu_generator,
        )
        source_batch = target_paths.index_select(
            0, source_indices.to(target_paths.device)
        )
        target_batch = target_paths.index_select(
            0, target_indices.to(target_paths.device)
        )
        noise = sample_standard_normals(
            grid=model.grid,
            batch_size=int(training.batch_size),
            noise_dim=int(model.architecture.noise_dim),
            device=model.device,
            dtype=model.dtype,
            seed=derive_stream_seed(
                base_seed=training.seed,
                stream_offset=step + 1,
            ),
        )
        rollout = model.rollout(x0=source_batch[:, 0, :], noise=noise)
        objective, objective_metrics = model._complete_objective_metrics(
            generated_path=rollout.paths,
            controls=rollout.controls,
            target_path=target_batch,
            training=training,
            differentiate_controls=True,
        )
        if not bool(torch.isfinite(objective.detach()).item()):
            raise RuntimeError("direct-BPTT objective became nonfinite")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        parameters = list(model.network.parameters())
        grad_norm = model._grad_norm(parameters)
        clip_active = float(
            training.grad_clip_norm is not None
            and grad_norm > float(training.grad_clip_norm)
        )
        if training.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                float(training.grad_clip_norm),
            )
        clipped_grad_norm = model._grad_norm(parameters)
        optimizer.step()
        should_log = (
            step == 0
            or (step + 1) % int(training.log_every) == 0
            or step + 1 == int(training.num_steps)
        )
        if should_log:
            metrics = {
                **objective_metrics,
                **model._control_diagnostics(rollout.controls),
                "step": float(step + 1),
                "train_total_loss": float(objective.detach().item()),
                "grad_norm": float(grad_norm),
                "grad_norm_clipped": float(clipped_grad_norm),
                "grad_clip_active": clip_active,
                "grad_clip_scale": (
                    float(clipped_grad_norm / grad_norm)
                    if grad_norm > 0.0
                    else 1.0
                ),
                "gradient_nonfinite_fraction": model._nonfinite_fraction(
                    [
                        parameter.grad
                        for parameter in parameters
                        if parameter.grad is not None
                    ]
                ),
                "parameter_nonfinite_fraction": model._nonfinite_fraction(
                    parameters
                ),
                "direct_bptt_control": 1.0,
            }
            history.append(metrics)
            if log_callback is not None:
                log_callback(model, metrics)
    return FitResult(
        history=history,
        final_metrics=(dict(history[-1]) if history else {}),
    )


def _sample_prices(
    model: DiscreteMPModel,
    *,
    num_paths: int,
    batch_size: int,
    seed: int,
    s0: float,
) -> np.ndarray:
    output = np.empty(
        (int(num_paths), int(model.grid.num_steps) + 1),
        dtype=np.float32,
    )
    x0 = torch.zeros((1, 1), device=model.device, dtype=model.dtype)
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, int(num_paths), int(batch_size))):
            count = min(int(batch_size), int(num_paths) - start)
            batch_seed = derive_stream_seed(
                base_seed=int(seed),
                stream_offset=10_000 + batch_index,
            )
            sample = model.sample(num_paths=count, x0=x0, seed=batch_seed)
            log_paths = sample.paths[..., 0].detach().cpu().numpy()
            output[start : start + count] = (
                float(s0) * np.exp(log_paths)
            ).astype(np.float32)
    return output


def _evaluate_validation(
    *,
    dataset_paths: dict[str, Path],
    bank_path: Path,
    output_path: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "scripts" / "evaluate_drawdown_memory.py"),
            "--train-data",
            str(dataset_paths["train"]),
            "--test-data",
            str(dataset_paths["disc"]),
            "--generated-data",
            str(bank_path),
            "--dataset-manifest",
            str(dataset_paths["manifest"]),
            "--output",
            str(output_path),
        ],
        check=True,
    )


def _reference_diagnostic(
    *,
    target_paths: torch.Tensor,
    reference,
    split_index: int,
    history_cutoff: int,
    threshold: float,
) -> dict[str, float]:
    values = target_paths[..., 0]
    prefix = values[:, : int(history_cutoff) + 1]
    drawdown = torch.cummax(prefix, dim=1).values - prefix
    hit = drawdown.amax(dim=1) >= float(threshold)
    with torch.no_grad():
        _, sigma = reference.evaluate(
            target_paths[:, : int(split_index) + 1],
            step_index=int(split_index),
        )
    flat_sigma = sigma[:, 0]
    centered_hit = hit.to(flat_sigma.dtype) - hit.to(flat_sigma.dtype).mean()
    centered_sigma = flat_sigma - flat_sigma.mean()
    correlation = torch.mean(centered_hit * centered_sigma) / torch.sqrt(
        torch.mean(centered_hit.pow(2)) * torch.mean(centered_sigma.pow(2))
    ).clamp_min(torch.finfo(flat_sigma.dtype).eps)
    return {
        "history_cutoff": float(history_cutoff),
        "drawdown_threshold": float(threshold),
        "hit_rate": float(hit.to(flat_sigma.dtype).mean().item()),
        "sigma_at_split_hit_mean": float(flat_sigma[hit].mean().item()),
        "sigma_at_split_no_hit_mean": float(flat_sigma[~hit].mean().item()),
        "early_hit_sigma_at_split_correlation": float(correlation.item()),
    }


def main() -> None:
    args = _parse_args()
    online_checkpoint_steps = tuple(int(value) for value in args.online_checkpoint_steps)
    if online_checkpoint_steps != tuple(sorted(set(online_checkpoint_steps))):
        raise ValueError("online-checkpoint-steps must be unique and increasing")
    if any(
        value < 1 or value >= int(args.source_training_steps)
        for value in online_checkpoint_steps
    ):
        raise ValueError(
            "online-checkpoint-steps must lie in [1, source-training-steps)"
        )
    if online_checkpoint_steps and ARM_GENERIC_CE_ONLINE_SOURCE_VOLATILITY_ONLY not in args.arms:
        raise ValueError(
            "online-checkpoint-steps require the online source-stage "
            "volatility-only arm"
        )
    run_dir = ensure_run_dir(args.run_dir or _default_run_dir())
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    target_paths, dataset_manifest, dataset_paths = _load_dataset(
        Path(args.dataset_root),
        device=device,
        dtype=dtype,
    )
    if int(args.max_training_paths) > 0:
        if int(args.max_training_paths) < 2:
            raise ValueError("max-training-paths must be zero or at least two")
        target_paths = target_paths[: int(args.max_training_paths)]
    configuration = dataset_manifest["configuration"]
    dt = float(configuration["dt"])
    sequence_length = int(configuration["sequence_length"])
    num_steps = sequence_length - 1
    split_index = int(configuration["split_index"])
    history_cutoff = int(configuration["history_cutoff"])
    s0 = float(configuration["s0"])
    grid = DiscreteTimeGrid(T=num_steps * dt, num_steps=num_steps)
    temporal_scale = (
        float(sequence_length) / 128.0
        if bool(args.scale_temporal_features)
        else 1.0
    )

    def scaled(values: tuple[int, ...]) -> tuple[int, ...]:
        result = tuple(max(1, int(round(temporal_scale * value))) for value in values)
        if len(set(result)) != len(result):
            raise ValueError("temporal scaling produced duplicate window lengths")
        return result

    rolling_windows = scaled((5, 10, 20, 32))
    ewma_half_lives = scaled((5, 20, 60))
    downside_windows = scaled((10, 32))
    acf_lags = scaled((1, 2, 5, 10))
    joint_prefix_windows = scaled((16, 32, 64))
    joint_recent_return_window = scaled((16,))[0]
    memory_half_lives = scaled(
        tuple(int(value) for value in args.drawdown_memory_half_lives)
    )
    memory_lags = scaled(tuple(int(value) for value in args.drawdown_memory_lags))
    threshold_fit_cutoff = min(split_index, scaled((48,))[0])
    quantiles = tuple(float(value) for value in args.drawdown_threshold_quantiles)
    thresholds = _thresholds_from_training(
        target_paths,
        history_cutoff=threshold_fit_cutoff,
        quantiles=quantiles,
    )
    print(
        "train-only drawdown thresholds "
        + ", ".join(f"{value:.6f}" for value in thresholds),
        flush=True,
    )

    reference_thresholds = () if args.generic_reference else thresholds
    reference_half_lives = (
        ()
        if args.generic_reference
        else memory_half_lives
    )
    reference_lags = (
        ()
        if args.generic_reference
        else memory_lags
    )
    reference_calibration: dict[str, object] | None = None
    if str(args.reference_kind) in {
        "guyon_lekeufack_structural_likelihood",
        "guyon_lekeufack_price_realized_volatility",
    }:
        if bool(args.calibrate_reference):
            raise ValueError(
                "--calibrate-reference is not defined for the Guyon reference"
            )
        guyon_kwargs = {
            "target_paths": target_paths,
            "grid": grid,
            "ridge": float(args.ridge_lambda),
            "variance_ridge": float(args.ridge_lambda),
            "variance_shrinkage": 0.1,
            "log_variance_bias_correction": 0.6,
            "sigma_min": float(args.sigma_min),
            "sigma_max": float(args.sigma_max),
        }
        if str(args.reference_kind) == "guyon_lekeufack_price_realized_volatility":
            reference = fit_guyon_lekeufack_price_realized_volatility_reference_kernel(
                **guyon_kwargs,
                realized_volatility_window=int(args.guyon_realized_volatility_window),
            )
        else:
            reference = fit_guyon_lekeufack_likelihood_reference_kernel(
                **guyon_kwargs,
                activity_update="structural_variance",
            )
    else:
        causal_reference = fit_causal_volatility_reference_kernel(
            target_paths=target_paths,
            grid=grid,
            rolling_windows=rolling_windows,
            ewma_half_lives=ewma_half_lives,
            downside_windows=downside_windows,
            drawdown_thresholds=reference_thresholds,
            drawdown_memory_half_lives=reference_half_lives,
            drawdown_memory_lags=reference_lags,
            drawdown_gate_temperature=float(args.drawdown_gate_temperature),
            drift_ridge=float(args.ridge_lambda),
            variance_ridges=tuple(
                float(value) for value in args.reference_variance_ridges
            ),
            crossfit_folds=int(args.reference_crossfit_folds),
            crossfit_seed=1701 + int(args.seed),
            standardized_feature_clip=(4.0 if args.calibrate_reference else None),
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
        )
        if args.calibrate_reference:
            anchor_reference = fit_local_gaussian_reference_kernel(
                target_paths=target_paths,
                grid=grid,
                ridge=float(args.ridge_lambda),
                variance_ridge=float(args.ridge_lambda),
                variance_shrinkage=0.1,
                log_variance_bias_correction=0.6,
                sigma_min=float(args.sigma_min),
                sigma_max=float(args.sigma_max),
            )
            training_return_std = float(
                (target_paths[:, 1:, :] - target_paths[:, :-1, :])
                .std(unbiased=False)
                .item()
            )
            reference, reference_calibration = (
                calibrate_shrunk_causal_reference_kernel(
                    target_paths=target_paths,
                    causal_reference=causal_reference,
                    anchor_reference=anchor_reference,
                    causal_weights=(
                        0.0,
                        0.4,
                        0.5,
                        0.55,
                        0.6,
                        0.65,
                        0.7,
                        0.75,
                        0.8,
                        1.0,
                    ),
                    calibration_paths=4096,
                    calibration_seed=271828,
                    offset_steps=8,
                    offset_relaxation=0.5,
                    return_std_tolerance=0.03,
                    sigma_std_floor=(
                        0.25 * training_return_std / math.sqrt(float(grid.dt))
                    ),
                    sigma_cap_fraction_limit=0.01,
                    path_rv_std_ratio_min=0.5,
                    path_rv_std_ratio_max=1.5,
                    path_rv_quantile_spread_ratio_min=0.4,
                    path_rv_quantile_spread_ratio_max=1.6,
                    clustering_lags=(1, 5, 20),
                    sigma_min=float(args.sigma_min),
                    sigma_max=float(args.sigma_max),
                )
            )
        else:
            reference = causal_reference
    control = SpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=float(args.eta),
        reference_kernel=reference,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    volatility_only_control = VolatilityOnlySpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=float(args.eta),
        reference_kernel=reference,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    memory_ce_estimator = CausalPathFeatureRidgeConditionalExpectation(
        grid=grid,
        ridge=float(args.ridge_lambda),
        rolling_windows=rolling_windows,
        ewma_half_lives=ewma_half_lives,
        downside_windows=downside_windows,
        drawdown_thresholds=thresholds,
        drawdown_memory_half_lives=memory_half_lives,
        drawdown_memory_lags=memory_lags,
        drawdown_gate_temperature=float(args.drawdown_gate_temperature),
    )
    generic_ce_estimator = CausalPathFeatureRidgeConditionalExpectation(
        grid=grid,
        ridge=float(args.ridge_lambda),
        rolling_windows=rolling_windows,
        ewma_half_lives=ewma_half_lives,
        downside_windows=downside_windows,
        drawdown_thresholds=(),
        drawdown_memory_half_lives=(),
        drawdown_memory_lags=(),
        drawdown_gate_temperature=float(args.drawdown_gate_temperature),
    )
    base = build_heston_discrepancy(
        num_steps=num_steps,
        include_acf=True,
        acf_lags=acf_lags,
        preset="old_fullv_w0p25",
        lambda_scale=float(args.lambda_scale),
        kappa_scale=float(args.kappa_scale),
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )
    joint = fit_fixed_target_joint_prefix_future_volatility_mmd(
        target_paths,
        grid=grid,
        split_index=split_index,
        future_horizon=int(configuration["future_horizon"]),
        prefix_windows=joint_prefix_windows,
        recent_return_window=joint_recent_return_window,
        bandwidths=(0.5, 1.0, 2.0),
    )
    residual_arms = {ARM_RESIDUAL_ONLINE, ARM_RESIDUAL_NESTED}
    needs_residual = bool(residual_arms.intersection(str(arm) for arm in args.arms))
    residual: FixedTargetResidualDrawdownMemoryMoments | None = (
        fit_fixed_target_residual_drawdown_memory_moments(
            target_paths,
            grid=grid,
            split_index=split_index,
            history_cutoffs=scaled((32, 40, 48)),
            future_horizons=scaled((16, 32, 48)),
            recent_volatility_windows=scaled((10, 20, 32)),
            drawdown_thresholds=thresholds,
            gate_temperature=float(args.drawdown_gate_temperature),
            ridge=float(args.ridge_lambda),
            crossfit_folds=int(args.residual_crossfit_folds),
            crossfit_seed=1907 + int(args.seed),
        )
        if needs_residual
        else None
    )
    joint_discrepancy = CompositePathFunctionalDiscrepancy(
        blocks=tuple(base.blocks)
        + (
            WeightedDiscrepancy(
                name="volatility_law_joint_prefix_future_rv",
                weight=1.0,
                discrepancy=joint,
            ),
        )
    )
    residual_discrepancy = (
        CompositePathFunctionalDiscrepancy(
            blocks=tuple(joint_discrepancy.blocks)
            + (
                WeightedDiscrepancy(
                    name="volatility_law_residual_drawdown_memory",
                    weight=float(args.residual_weight),
                    discrepancy=residual,
                ),
            )
        )
        if residual is not None
        else joint_discrepancy
    )

    reference_report = _reference_diagnostic(
        target_paths=target_paths,
        reference=reference,
        split_index=split_index,
        history_cutoff=history_cutoff,
        threshold=float(configuration["drawdown_threshold"]),
    )
    write_json(
        run_dir / "design.json",
        {
            "dataset": {
                name: str(path.resolve())
                for name, path in dataset_paths.items()
            },
            "test_split_used": False,
            "validation_split": "disc.npy",
            "source_run_dir": (
                str(Path(args.source_run_dir).resolve())
                if args.source_run_dir is not None
                else str(run_dir.resolve())
            ),
            "drawdown_threshold_quantiles": list(quantiles),
            "fitted_drawdown_thresholds": list(thresholds),
            "reference": reference.summary(),
            "reference_calibration": reference_calibration,
            "reference_basis": (
                str(args.reference_kind)
                if str(args.reference_kind).startswith("guyon_lekeufack")
                else (
                    "generic_causal"
                    if args.generic_reference
                    else "drawdown_memory_aware"
                )
            ),
            "temporal_scaling": {
                "enabled": bool(args.scale_temporal_features),
                "scale": float(temporal_scale),
                "rolling_windows": list(rolling_windows),
                "ewma_half_lives": list(ewma_half_lives),
                "downside_windows": list(downside_windows),
                "acf_lags": list(acf_lags),
                "joint_prefix_windows": list(joint_prefix_windows),
                "joint_recent_return_window": int(joint_recent_return_window),
            },
            "reference_memory_diagnostic": reference_report,
            "conditional_expectation_feature_names": list(
                memory_ce_estimator.feature_names
            ),
            "generic_conditional_expectation_feature_names": list(
                generic_ce_estimator.feature_names
            ),
            "residual_discrepancy": (
                residual.summary() if residual is not None else None
            ),
        },
    )
    print(
        "reference early-hit diagnostic "
        f"sigma(hit)={reference_report['sigma_at_split_hit_mean']:.4f} "
        f"sigma(no-hit)={reference_report['sigma_at_split_no_hit_mean']:.4f} "
        f"corr={reference_report['early_hit_sigma_at_split_correlation']:.3f}",
        flush=True,
    )
    if device.type == "cuda":
        # The train-only full-law fits above use large temporary design
        # matrices.  They are no longer needed during MP training.
        torch.cuda.empty_cache()

    source_training = _make_training(
        args,
        num_steps=int(args.source_training_steps),
        seed=int(args.seed),
    )
    source_run_dir = (
        Path(args.source_run_dir)
        if args.source_run_dir is not None
        else run_dir
    )
    source_checkpoint_path = source_run_dir / "source_model_checkpoint.pt"
    source_history_path = source_run_dir / "source_history.jsonl"
    source_metrics_path = source_run_dir / "source_metrics.json"
    exact_reference_protocol = bool(
        args.arms
        and all(str(arm) in EXACT_REFERENCE_ARMS for arm in args.arms)
    )
    if source_checkpoint_path.exists():
        source_checkpoint = torch.load(
            source_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        source_final = (
            json.loads(source_metrics_path.read_text(encoding="utf-8"))
            if source_metrics_path.exists()
            else {}
        )
        print("reusing completed causal-basis source checkpoint", flush=True)
    elif args.source_run_dir is not None:
        raise FileNotFoundError(source_checkpoint_path)
    else:
        source = _build_model(
            args=args,
            grid=grid,
            control=control,
            discrepancy=base,
            ce_estimator=(
                generic_ce_estimator
                if args.generic_reference
                else memory_ce_estimator
            ),
            device=device,
            dtype=dtype,
            seed=int(args.seed),
        )
        if exact_reference_protocol:
            source_checkpoint = source.checkpoint_state()
            source_final = {
                "exact_zero_adjoint_reference": 1.0,
                "source_optimizer_steps": 0.0,
            }
            source_history = []
        else:
            source_fit = source.fit(
                target_paths,
                training=source_training,
                log_callback=_progress("source"),
            )
            source_checkpoint = source.checkpoint_state()
            source_final = source_fit.final_metrics
            source_history = source_fit.history
        torch.save(source_checkpoint, source_checkpoint_path)
        write_history_jsonl(source_history_path, source_history)
        write_json(source_metrics_path, source_final)
        del source
        if device.type == "cuda":
            torch.cuda.empty_cache()

    joint_training = _make_training(
        args,
        num_steps=int(args.joint_training_steps),
        seed=int(args.seed) + 1,
    )
    nested = _make_nested(args, seed=int(args.seed) + 1)
    proximal_config = _make_proximal_solver(
        args,
        seed=int(args.seed) + 1,
        num_steps=num_steps,
    )
    extragradient_config = _make_extragradient_solver(
        args,
        seed=int(args.seed) + 1,
    )
    continuation_config = _make_continuation_solver(
        args,
        seed=int(args.seed) + 1,
    )
    proximal_validation_count = int(args.proximal_validation_paths)
    if not 1 <= proximal_validation_count < int(target_paths.shape[0]):
        raise ValueError(
            "proximal-validation-paths must leave a nonempty backward pool"
        )
    proximal_backward_pool = target_paths[:-proximal_validation_count]
    proximal_validation_pool = target_paths[-proximal_validation_count:]
    completed: dict[str, object] = {}
    for arm in tuple(str(value) for value in args.arms):
        arm_dir = ensure_run_dir(run_dir / arm)
        checkpoint_path = arm_dir / "model_checkpoint.pt"
        bank_path = arm_dir / f"generated_paths_{int(args.bank_size)}x{sequence_length}.npy"
        metrics_path = arm_dir / "validation_metrics.json"
        if metrics_path.exists():
            completed[arm] = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"reusing completed arm {arm}", flush=True)
            continue
        nested_arms = {
            ARM_REFERENCE_CE_NESTED,
            ARM_RESIDUAL_NESTED,
            ARM_GENERIC_CE_NESTED,
            ARM_GENERIC_CE_NESTED_REFERENCE,
            ARM_GENERIC_CE_NESTED_VOLATILITY_ONLY,
        }
        discrepancy = (
            base
            if arm == ARM_GENERIC_CE_ONLINE_SOURCE_VOLATILITY_ONLY
            else (residual_discrepancy if arm in residual_arms else joint_discrepancy)
        )
        arm_ce_estimator = (
            generic_ce_estimator
            if arm
            in {
                ARM_GENERIC_CE_ONLINE,
                ARM_GENERIC_CE_ONLINE_VOLATILITY_ONLY,
                ARM_GENERIC_CE_ONLINE_SOURCE_VOLATILITY_ONLY,
                ARM_GENERIC_CE_NESTED,
                ARM_GENERIC_CE_PROXIMAL,
                ARM_GENERIC_CE_EXTRAGRADIENT,
                ARM_GENERIC_CE_CONTINUATION,
                ARM_GENERIC_REFERENCE_ONLY,
                ARM_GENERIC_CE_NESTED_REFERENCE,
                ARM_GENERIC_CE_NESTED_VOLATILITY_ONLY,
                ARM_DIRECT_BPTT_VOLATILITY_ONLY,
            }
            else memory_ce_estimator
        )
        arm_control = (
            volatility_only_control if arm in VOLATILITY_ONLY_ARMS else control
        )
        model = _build_model(
            args=args,
            grid=grid,
            control=arm_control,
            discrepancy=discrepancy,
            ce_estimator=arm_ce_estimator,
            device=device,
            dtype=dtype,
            seed=int(args.seed),
        )
        arm_training = (
            source_training
            if arm == ARM_GENERIC_CE_ONLINE_SOURCE_VOLATILITY_ONLY
            else joint_training
        )
        initial_network_sha256 = network_sha256(model.network)
        source_network_sha256 = stable_sha256(
            source_checkpoint["network_state_dict"]
        )
        exact_reference_initialization = bool(
            arm == ARM_GENERIC_CE_CONTINUATION
            or arm in EXACT_REFERENCE_ARMS
        )
        if not exact_reference_initialization:
            model.load_checkpoint_state(source_checkpoint)
            initial_network_sha256 = network_sha256(model.network)
        elif initial_network_sha256 != source_network_sha256:
            raise RuntimeError(
                "the exact-reference comparison arms do not share the same "
                "initial network state"
            )
        training_resources: dict[str, float] = {}
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_allocated = 0
            baseline_reserved = 0
        training_started = time.perf_counter()
        checkpoint_dir = arm_dir / "training_checkpoints"
        if (
            arm == ARM_GENERIC_CE_ONLINE_SOURCE_VOLATILITY_ONLY
            and online_checkpoint_steps
        ):
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        validated_report: dict[str, object] | None = None
        validated_artifacts: dict[str, object] | None = None
        continuation_reached_beta_one = True
        if arm == ARM_GENERIC_REFERENCE_ONLY:
            fit = FitResult(
                history=[],
                final_metrics={
                    "reference_only": 1.0,
                    "optimizer_steps": 0.0,
                },
            )
            solver = "none_reference_only"
        elif arm == ARM_GENERIC_CE_CONTINUATION:
            if str(joint_training.noise_target_control_variate) != "adjoint":
                raise ValueError(
                    "the continuation arm requires "
                    "--noise-target-control-variate adjoint"
                )

            def continuation_progress(row: dict[str, object]) -> None:
                print(
                    f"{arm} beta={float(row['beta_before']):.6g}->"
                    f"{float(row['beta_target']):.6g} "
                    f"accepted={bool(row['accepted'])} "
                    f"updates={int(row['accepted_outer_updates'])} "
                    f"residual="
                    f"{float(row['residual_before']['relative_moment_residual']):.4g}->"
                    f"{float(row['residual_after']['relative_moment_residual']):.4g} "
                    f"objective_gain="
                    f"{float(row['objective_confirmation']['mean_relative_improvement']):.3%}",
                    flush=True,
                )

            continued = fit_discrepancy_continuation(
                model=model,
                backward_target_paths=proximal_backward_pool,
                validation_target_paths=proximal_validation_pool,
                training=joint_training,
                extragradient=extragradient_config,
                continuation=continuation_config,
                progress_callback=continuation_progress,
            )
            fit = FitResult(
                history=continued.history,
                final_metrics=(
                    dict(continued.history[-1])
                    if continued.history
                    else {}
                ),
            )
            validated_report = continued.report
            validated_artifacts = continued.artifacts
            continuation_reached_beta_one = bool(
                continued.reached_beta_one
            )
            solver = "discrepancy_continuation_extragradient"
        elif arm == ARM_GENERIC_CE_EXTRAGRADIENT:
            if str(joint_training.noise_target_control_variate) != "adjoint":
                raise ValueError(
                    "the safeguarded extragradient arm requires "
                    "--noise-target-control-variate adjoint"
                )

            def extragradient_progress(row: dict[str, object]) -> None:
                accepted_trial = next(
                    (
                        trial
                        for trial in row["trials"]
                        if bool(trial["accepted"])
                    ),
                    None,
                )
                residual_after = (
                    row["residual_before"]
                    if accepted_trial is None
                    else accepted_trial["residual_after"]
                )
                print(
                    f"{arm} outer={int(row['outer_step'])} "
                    f"accepted={bool(row['accepted'])} "
                    f"scale={float(row['accepted_backtrack_scale']):.4g} "
                    f"residual="
                    f"{float(row['residual_before']['normalized_control_impact_norm']):.4g}->"
                    f"{float(residual_after['normalized_control_impact_norm']):.4g}",
                    flush=True,
                )

            extragradient = fit_safeguarded_extragradient(
                model=model,
                backward_target_paths=proximal_backward_pool,
                validation_target_paths=proximal_validation_pool,
                training=joint_training,
                config=extragradient_config,
                progress_callback=extragradient_progress,
            )
            fit = FitResult(
                history=extragradient.history,
                final_metrics=(
                    dict(extragradient.history[-1])
                    if extragradient.history
                    else {}
                ),
            )
            validated_report = extragradient.report
            validated_artifacts = extragradient.artifacts
            solver = "safeguarded_proximal_extragradient"
        elif arm == ARM_GENERIC_CE_PROXIMAL:
            if str(joint_training.noise_target_control_variate) != "adjoint":
                raise ValueError(
                    "the proximal fixed-point arm requires "
                    "--noise-target-control-variate adjoint"
                )

            def proximal_progress(row: dict[str, object]) -> None:
                update = row["outer_update"]
                stationarity = row["raw_stationarity"]["accepted_update"]
                print(
                    f"{arm} outer={int(row['outer_step'])} "
                    f"accepted={bool(update['accepted'])} "
                    f"rates={update['selected_rates']} "
                    f"objective_change={float(update['objective_change']):.5g} "
                    f"raw_P/R_rel_rmse="
                    f"{float(stationarity['p']['relative_rmse']):.3f}/"
                    f"{float(stationarity['r']['relative_rmse']):.3f}",
                    flush=True,
                )

            validated = fit_validated_nested(
                model=model,
                backward_target_paths=proximal_backward_pool,
                validation_target_paths=proximal_validation_pool,
                training=joint_training,
                config=proximal_config,
                progress_callback=proximal_progress,
            )
            fit = FitResult(
                history=validated.history,
                final_metrics=(
                    dict(validated.history[-1])
                    if validated.history
                    else {}
                ),
            )
            validated_report = validated.report
            validated_artifacts = validated.artifacts
            solver = "refreshed_proximal_forward_backward"
        elif arm in nested_arms:
            fit = model.fit_nested(
                target_paths,
                training=joint_training,
                nested=nested,
                log_callback=_progress(arm),
            )
            solver = "nested_forward_backward"
        elif arm in {ARM_DIRECT_BPTT, ARM_DIRECT_BPTT_VOLATILITY_ONLY}:
            fit = _fit_direct_objective_control(
                model,
                target_paths,
                training=joint_training,
                log_callback=_progress(arm),
            )
            solver = "direct_bptt"
        else:
            progress = _progress(arm)

            def online_progress_with_checkpoints(
                current_model: DiscreteMPModel,
                metrics: dict[str, float],
            ) -> None:
                progress(current_model, metrics)
                step = int(metrics.get("step", 0.0))
                if step in online_checkpoint_steps:
                    torch.save(
                        current_model.checkpoint_state(),
                        checkpoint_dir / f"step_{step:04d}.pt",
                    )

            fit = model.fit(
                target_paths,
                training=arm_training,
                log_callback=(
                    online_progress_with_checkpoints
                    if arm == ARM_GENERIC_CE_ONLINE_SOURCE_VOLATILITY_ONLY
                    and online_checkpoint_steps
                    else progress
                ),
            )
            solver = "online"
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_allocated = 0
            peak_reserved = 0
        training_resources = {
            "training_wall_seconds": float(time.perf_counter() - training_started),
            "cuda_baseline_allocated_bytes": float(baseline_allocated),
            "cuda_baseline_reserved_bytes": float(baseline_reserved),
            "cuda_peak_allocated_bytes": float(peak_allocated),
            "cuda_peak_reserved_bytes": float(peak_reserved),
            "cuda_incremental_peak_allocated_bytes": float(
                max(0, peak_allocated - baseline_allocated)
            ),
            "cuda_incremental_peak_reserved_bytes": float(
                max(0, peak_reserved - baseline_reserved)
            ),
        }
        final_checkpoint = model.checkpoint_state()
        torch.save(final_checkpoint, checkpoint_path)
        if validated_report is not None:
            write_json(arm_dir / "validated_nested_report.json", validated_report)
        if validated_artifacts is not None:
            torch.save(
                validated_artifacts,
                arm_dir / "validated_nested_artifacts.pt",
            )
        write_history_jsonl(arm_dir / "training_history.jsonl", fit.history)
        if not continuation_reached_beta_one:
            raise RuntimeError(
                "continuation did not reach beta=1; downstream metrics were "
                "not evaluated. Inspect validated_nested_report.json"
            )
        checkpoint_evaluations: list[dict[str, object]] = []
        if (
            arm == ARM_GENERIC_CE_ONLINE_SOURCE_VOLATILITY_ONLY
            and online_checkpoint_steps
        ):
            for step in online_checkpoint_steps:
                intermediate_checkpoint_path = checkpoint_dir / f"step_{step:04d}.pt"
                if not intermediate_checkpoint_path.is_file():
                    raise RuntimeError(
                        f"missing requested online checkpoint {intermediate_checkpoint_path}"
                    )
                intermediate = torch.load(
                    intermediate_checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_checkpoint_state(intermediate)
                evaluation_dir = ensure_run_dir(
                    arm_dir / "checkpoint_evaluations" / f"step_{step:04d}"
                )
                intermediate_bank_path = (
                    evaluation_dir
                    / f"generated_paths_{int(args.bank_size)}x{sequence_length}.npy"
                )
                intermediate_metrics_path = evaluation_dir / "validation_metrics.json"
                intermediate_bank = _sample_prices(
                    model,
                    num_paths=int(args.bank_size),
                    batch_size=int(args.sample_batch_size),
                    seed=int(args.seed),
                    s0=s0,
                )
                np.save(intermediate_bank_path, intermediate_bank)
                _evaluate_validation(
                    dataset_paths=dataset_paths,
                    bank_path=intermediate_bank_path,
                    output_path=intermediate_metrics_path,
                )
                checkpoint_evaluations.append(
                    {
                        "step": int(step),
                        "checkpoint": str(intermediate_checkpoint_path.resolve()),
                        "bank": str(intermediate_bank_path.resolve()),
                        "metrics": str(intermediate_metrics_path.resolve()),
                        "common_bank_seed": int(args.seed),
                    }
                )
                del intermediate_bank
            model.load_checkpoint_state(final_checkpoint)

        bank = _sample_prices(
            model,
            num_paths=int(args.bank_size),
            batch_size=int(args.sample_batch_size),
            seed=int(args.seed),
            s0=s0,
        )
        np.save(bank_path, bank)
        _evaluate_validation(
            dataset_paths=dataset_paths,
            bank_path=bank_path,
            output_path=metrics_path,
        )
        validation_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        completed[arm] = validation_metrics
        write_json(
            arm_dir / "training_manifest.json",
            {
                "arm": arm,
                "solver": solver,
                "running_cost": "specific_entropy",
                "admissible_control": (
                    "volatility_only_alpha_fixed_zero"
                    if arm in VOLATILITY_ONLY_ARMS
                    else "drift_and_volatility"
                ),
                "network_regression_loss_changed": False,
                "adjoint_network": str(args.adjoint_network),
                "transformer_layers": int(args.transformer_layers),
                "transformer_heads": int(args.transformer_heads),
                "source_training": asdict(source_training),
                "source_checkpoint": str(source_checkpoint_path.resolve()),
                "joint_training": asdict(joint_training),
                "fit_training": asdict(arm_training),
                "training_stage": (
                    "source"
                    if arm == ARM_GENERIC_CE_ONLINE_SOURCE_VOLATILITY_ONLY
                    else "joint"
                ),
                "online_checkpoint_evaluations": checkpoint_evaluations,
                "nested_training": (
                    asdict(nested)
                    if arm in nested_arms
                    else (
                        asdict(proximal_config)
                        if arm == ARM_GENERIC_CE_PROXIMAL
                        else (
                            asdict(extragradient_config)
                            if arm == ARM_GENERIC_CE_EXTRAGRADIENT
                            else (
                                {
                                    "continuation": asdict(
                                        continuation_config
                                    ),
                                    "extragradient": asdict(
                                        extragradient_config
                                    ),
                                }
                                if arm == ARM_GENERIC_CE_CONTINUATION
                                else None
                            )
                        )
                    )
                ),
                "reference": reference.summary(),
                "conditional_expectation_feature_names": list(
                    arm_ce_estimator.feature_names
                ),
                "projection_basis": (
                    "unused_by_direct_bptt"
                    if arm in {ARM_DIRECT_BPTT, ARM_DIRECT_BPTT_VOLATILITY_ONLY}
                    else (
                        "generic_causal"
                        if arm
                        in {
                            ARM_GENERIC_CE_ONLINE,
                            ARM_GENERIC_CE_ONLINE_VOLATILITY_ONLY,
                            ARM_GENERIC_CE_ONLINE_SOURCE_VOLATILITY_ONLY,
                            ARM_GENERIC_CE_NESTED,
                            ARM_GENERIC_CE_PROXIMAL,
                            ARM_GENERIC_CE_EXTRAGRADIENT,
                            ARM_GENERIC_CE_CONTINUATION,
                            ARM_GENERIC_REFERENCE_ONLY,
                            ARM_GENERIC_CE_NESTED_REFERENCE,
                            ARM_GENERIC_CE_NESTED_VOLATILITY_ONLY,
                        }
                        else "drawdown_memory_aware"
                    )
                ),
                "residual_discrepancy": (
                    residual.summary()
                    if arm in residual_arms and residual is not None
                    else None
                ),
                "source_fit_final": source_final,
                "source_checkpoint_loaded": bool(
                    not exact_reference_initialization
                ),
                "initial_network_sha256": initial_network_sha256,
                "source_network_sha256": source_network_sha256,
                "initial_network_matches_source": bool(
                    initial_network_sha256 == source_network_sha256
                ),
                "initialization": (
                    "exact_zero_adjoint_reference"
                    if exact_reference_initialization
                    else "source_checkpoint"
                ),
                "fit_final": fit.final_metrics,
                "training_resources": training_resources,
                "validation_split": str(dataset_paths["disc"].resolve()),
                "test_split_used": False,
            },
        )
        del model, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        arm: {
            "errors": payload["errors"],
            "target_memory": payload["target_memory"],
            "generated_memory": payload["generated_memory"],
        }
        for arm, payload in completed.items()
    }
    write_json(run_dir / "validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
