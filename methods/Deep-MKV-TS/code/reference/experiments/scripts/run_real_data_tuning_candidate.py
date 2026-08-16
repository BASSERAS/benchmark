#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    CausalPathFeatureRidgeConditionalExpectation,
    DiscreteMPNestedTrainingConfig,
    DiscreteMPTrainingConfig,
)
from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from deep_mkv_gen_path_dt.noise import sample_standard_normals  # noqa: E402
from deep_mkv_gen_path_dt.model import FitResult  # noqa: E402
from deep_mkv_gen_path_dt.networks import (  # noqa: E402
    CausalFeatureAdjointResidualNetwork,
    CausalRegimeGatedAdjointResidualNetwork,
    CompactVolatilityCausalFeatureMap,
    FrozenBaseCausalResidualAdjointNetwork,
    IncrementCausalFeatureMap,
    TimewiseLinearCausalAdjointResidualNetwork,
    fit_timewise_causal_feature_normalization,
    fit_timewise_causal_regime_gates,
)
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from path_dt_experiments.real_data_generation import (  # noqa: E402
    load_frozen_real_data_fit,
    materialize_deep_mkv_adapter_dataset,
)
from path_dt_experiments.adjoint_gradient_normalization import (  # noqa: E402
    fit_train_only_adjoint_rms_normalized_discrepancy,
    fit_train_only_family_adjoint_rms_normalized_discrepancy,
    restore_family_adjoint_rms_normalized_discrepancy,
)
from path_dt_experiments.discrepancies import (  # noqa: E402
    StabilizedPrefixFutureVolatilityCorrelationDiscrepancy,
    fit_fixed_target_acf_moments,
    fit_fixed_target_conditional_residual_volatility_mmd,
    fit_fixed_target_pooled_return_wasserstein,
    fit_fixed_target_prefix_future_volatility_kernel_moments,
    fit_fixed_target_prefix_regime_future_volatility_claims,
)
from path_dt_experiments.residual_persistence import (  # noqa: E402
    fit_fixed_target_time_marginal_volatility,
)
from path_dt_experiments.real_data_protocol import (  # noqa: E402
    normalized_prices_to_log_paths,
    real_data_law_metrics,
)
from path_dt_experiments.runners import (  # noqa: E402
    ensure_run_dir,
    write_history_jsonl,
    write_json,
)
from path_dt_experiments.shadowing import (  # noqa: E402
    ShadowingFeatureConfig,
    evaluate_path_shadowing,
)
from run_basseras_heston_path_shadowing import (  # noqa: E402
    NUM_STEPS,
    _build_components,
    _build_model,
    _load_training_data,
    _write_scaler,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one validation-only real-data tuning candidate while "
            "preserving the maximum-principle model and adjoint loss."
        )
    )
    parser.add_argument("--phase", choices=("source", "joint"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument(
        "--source-checkpoint-topology",
        choices=("plain", "causal_residual"),
        default="plain",
        help=(
            "Topology stored in --source-checkpoint before adding the new "
            "joint-phase residual. Use causal_residual to freeze the complete "
            "fitted source correction and learn a second additive residual."
        ),
    )
    parser.add_argument(
        "--source-checkpoint-r-ce-basis",
        choices=("causal_compact", "causal_rich"),
        default="causal_compact",
        help=(
            "Causal feature topology stored in a causal-residual source "
            "checkpoint. The new joint residual may use a different --r-ce-basis."
        ),
    )
    parser.add_argument(
        "--source-checkpoint-stack-depth",
        type=int,
        default=None,
        help=(
            "Number of frozen causal-residual layers stored in a causal "
            "source checkpoint. The default is one for backward compatibility."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help=(
            "Resume a checkpoint whose network topology exactly matches "
            "--adjoint-network. Unlike --source-checkpoint, this does not "
            "wrap a plain source policy in a new residual network."
        ),
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=REPO_ROOT / "data" / "ICAIF_3Y" / "canonical_rth_128",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("/home/samer/scenarios/BASSERAS-benchmark"),
    )
    parser.add_argument("--protocol", default="primary_2026_holdout")
    parser.add_argument(
        "--objective-protocol",
        choices=(
            "legacy_heston_scale",
            "real_scale_corrected_v1",
            "real_scale_corrected_v2_adjoint_rms",
            "real_scale_corrected_v3_family_adjoint_rms",
        ),
        default="legacy_heston_scale",
        help="Select the legacy Heston-unit objective or the train-standardized real-data objective.",
    )
    parser.add_argument("--index", choices=("ES", "NQ", "RTY", "YM"), default="ES")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2, 3, 4), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--state-unit-scaling",
        choices=("none", "train_return_volatility"),
        default="none",
        help=(
            "Optionally express log-price states in train-only volatility "
            "units. The reference, admissible volatility set, drift cost, "
            "and generated-bank inverse transform are changed together, so "
            "this is a coordinate change rather than a different objective."
        ),
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument(
        "--adjoint-weight",
        type=float,
        default=1.0,
        help=(
            "Loss weight for the conditional level-adjoint (Y/P) target. "
            "Set to zero for the selected volatility-only Y-free policy."
        ),
    )
    parser.add_argument(
        "--adjoint-noise-weight",
        type=float,
        default=1.0,
        help=(
            "Loss weight for the conditional noise-adjoint (Z/R) target "
            "that enters the volatility control map."
        ),
    )
    parser.add_argument(
        "--online-checkpoint-steps",
        type=int,
        nargs="*",
        default=(),
        help=(
            "For online training, save and evaluate these intermediate steps "
            "on the common validation bank."
        ),
    )
    parser.add_argument(
        "--solver",
        choices=("online", "nested", "direct_bptt"),
        default="online",
        help=(
            "Online refreshes the empirical backward target every update; "
            "nested freezes one forward/backward problem for each inner loop; "
            "direct_bptt is a matched, non-MP optimization diagnostic for "
            "testing attainability of the unchanged complete objective."
        ),
    )
    parser.add_argument("--nested-outer-steps", type=int, default=4)
    parser.add_argument("--nested-inner-steps", type=int, default=175)
    parser.add_argument("--nested-outer-batch-size", type=int, default=512)
    parser.add_argument("--nested-outer-target-batch-size", type=int, default=512)
    parser.add_argument("--nested-outer-backward-replicates", type=int, default=1)
    parser.add_argument(
        "--nested-outer-target-estimator",
        choices=("projected_score", "nested_branches"),
        default="projected_score",
        help=(
            "Estimate P/R by a one-path score plus causal projection, or by "
            "averaging independent futures from each fixed prefix."
        ),
    )
    parser.add_argument("--nested-conditional-branches", type=int, default=16)
    parser.add_argument(
        "--nested-conditional-antithetic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--nested-conditional-query-batch-size",
        type=int,
        default=16384,
        help=(
            "Maximum number of time-indexed conditional query paths in one "
            "frozen-law Lions-derivative evaluation. This changes batching "
            "only, not the conditional estimator."
        ),
    )
    parser.add_argument(
        "--nested-population-batch-size",
        type=int,
        default=None,
        help=(
            "Optional independent unconditional generated-law bank used in "
            "the frozen Lions derivative. Prefix regression count remains "
            "--nested-outer-batch-size."
        ),
    )
    parser.add_argument(
        "--nested-line-search-batch-size",
        type=int,
        default=None,
        help=(
            "Optional independent common-random-number bank size used only "
            "for the complete-objective outer line search."
        ),
    )
    parser.add_argument("--nested-inner-batch-size", type=int, default=256)
    parser.add_argument(
        "--nested-inner-solver",
        choices=("adam", "timewise_linear_lstsq"),
        default="adam",
        help=(
            "Solve each frozen backward regression with Adam or, for the "
            "timewise-linear residual, its exact minimum-norm least-squares fit."
        ),
    )
    parser.add_argument("--nested-probe-paths", type=int, default=512)
    parser.add_argument(
        "--nested-relaxation-mode",
        choices=("joint", "adjoint_blocks"),
        default="adjoint_blocks",
    )
    parser.add_argument("--nested-max-backtracks", type=int, default=6)
    parser.add_argument("--nested-backtrack-factor", type=float, default=0.5)
    parser.add_argument("--nested-objective-tolerance", type=float, default=0.0)
    parser.add_argument("--nested-block-coordinate-passes", type=int, default=2)
    parser.add_argument("--nested-block-trust-fraction", type=float, default=1.0)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--ridge-lambda", type=float, required=True)
    parser.add_argument(
        "--ce-target-mode",
        choices=("ridge", "direct"),
        default="ridge",
        help=(
            "Use a causal ridge projection, or train directly on nested "
            "branch means that already estimate the two conditional MP "
            "moments. Direct mode is only valid with nested_branches."
        ),
    )
    parser.add_argument(
        "--r-ce-basis",
        choices=(
            "shared_flat_prefix",
            "causal_compact",
            "causal_compact_gated",
            "causal_rich",
        ),
        default="shared_flat_prefix",
        help="Conditional basis for the noise adjoint R.",
    )
    parser.add_argument(
        "--p-ce-basis",
        choices=("flat_prefix", "causal_compact"),
        default="flat_prefix",
        help="Conditional basis for the ordinary adjoint P.",
    )
    parser.add_argument(
        "--r-ce-normalization",
        choices=("batch", "train_fixed"),
        default="batch",
        help=(
            "Normalize causal R features separately in each outer regression "
            "or with immutable moments fitted on the training paths."
        ),
    )
    parser.add_argument(
        "--r-ce-time-mode",
        choices=("stepwise", "pooled"),
        default="stepwise",
        help=(
            "Fit a separate R projection at every time or one causal ridge "
            "projection pooled across paths and time."
        ),
    )
    parser.add_argument(
        "--adjoint-network",
        choices=(
            "plain",
            "increment_residual",
            "compact_residual",
            "timewise_linear_residual",
            "causal_residual",
            "causal_regime_residual",
        ),
        default="plain",
        help=(
            "Residual variants freeze the source network and add a zero-"
            "initialized adapted P/R network on observable causal features."
        ),
    )
    parser.add_argument(
        "--adjoint-regime-gate-temperature-fraction",
        type=float,
        default=0.10,
        help=(
            "Train-law rolling-volatility tercile gate temperature for the "
            "generic causal-regime residual network."
        ),
    )
    parser.add_argument(
        "--adjoint-regime-gate-feature",
        choices=(
            "rolling_realized_variance_20",
            "rolling_realized_variance_32",
            "ewma_realized_variance_20",
            "ewma_realized_variance_60",
        ),
        default="rolling_realized_variance_20",
        help="Generic causal volatility-memory statistic used by the regime experts.",
    )
    parser.add_argument("--ce-crossfit-folds", type=int, default=1)
    parser.add_argument(
        "--noise-target-control-variate",
        choices=("none", "timewise", "adjoint"),
        default="none",
    )
    parser.add_argument(
        "--noise-target-estimator",
        choices=("score", "stein"),
        default="score",
        help=(
            "Equivalent Gaussian estimators of R: the score product A*epsilon "
            "or the Stein path derivative dA/depsilon."
        ),
    )
    parser.add_argument("--stein-probes", type=int, default=16)
    parser.add_argument("--preconditioner-batches", type=int, default=8)
    parser.add_argument(
        "--reference-kind",
        choices=(
            "calibrated_causal",
            "guyon_lekeufack_price_realized_volatility",
        ),
        default="calibrated_causal",
        help="Frozen reference family fitted on the training sessions only.",
    )
    parser.add_argument(
        "--guyon-realized-volatility-window",
        type=int,
        default=20,
        help="Forward return window used by the price-only Guyon calibration.",
    )
    parser.add_argument(
        "--reference-rv-spread-ratio-min",
        type=float,
        default=0.4,
        help=(
            "Train-only lower stability gate for the calibrated reference's "
            "path-RV 10--90 percent quantile spread relative to training data."
        ),
    )
    parser.add_argument(
        "--reference-feature-basis",
        choices=("compact", "rich_generic"),
        default="compact",
        help=(
            "Use the compact causal volatility reference or add only generic "
            "causal persistence features. Both variants retain the same "
            "train-only cross-fitting and recursive calibration protocol."
        ),
    )
    parser.add_argument(
        "--reference-causal-weight",
        type=float,
        default=None,
        help=(
            "Optionally fix the causal/anchor log-variance blend weight. The "
            "log-variance offset is still calibrated recursively on training "
            "paths; validation is used only to compare the frozen candidates."
        ),
    )
    parser.add_argument(
        "--frozen-discrepancy-normalization-manifest",
        type=Path,
        help="Reuse exact family-normalized block weights for controlled estimator ablations.",
    )
    parser.add_argument("--lambda-scale", type=float, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument(
        "--running-cost",
        choices=("specific_entropy", "gaussian_relative_entropy"),
        default="specific_entropy",
    )
    parser.add_argument(
        "--use-reference-drift",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Express drift controls as the fitted physical reference drift "
            "plus an MP correction. This must be enabled for both arms of a "
            "matched reference-drift control comparison."
        ),
    )
    parser.add_argument(
        "--drift-control-mode",
        choices=("full", "volatility_only"),
        default="full",
        help=(
            "Allow the MP P head to correct fitted drift, or retain fitted "
            "drift exactly while continuing to train the unchanged P/R loss."
        ),
    )
    parser.add_argument("--lambda-x", type=float, default=1.0)
    parser.add_argument("--lambda-v", type=float, default=1.0)
    parser.add_argument("--joint-weight", type=float, default=0.25)
    parser.add_argument(
        "--return-discrepancy",
        choices=("mmd", "wasserstein"),
        default="mmd",
        help=(
            "Use the existing path-vector return MMD or a train-fitted "
            "pooled-return empirical Wasserstein-2 block."
        ),
    )
    parser.add_argument(
        "--global-rv-discrepancy",
        choices=(
            "mmd",
            "wasserstein",
            "log_wasserstein",
            "dated_wasserstein",
            "dated_log_wasserstein",
            "block_wasserstein",
            "block_log_wasserstein",
        ),
        default="mmd",
        help=(
            "Use the existing RBF-MMD global RV block or a train-fitted "
            "one-dimensional W2 quantile transport block in RV levels or log RV. "
            "The block variants use non-overlapping 32-step intraday marginals."
        ),
    )
    parser.add_argument(
        "--acf-discrepancy",
        choices=("mmd", "moment"),
        default="mmd",
        help=(
            "Use the legacy lag-product feature MMD or directly match the "
            "train-law pooled absolute/squared-return autocorrelations."
        ),
    )
    parser.add_argument(
        "--conditional-claim-mode",
        choices=(
            "none",
            "replace_joint",
            "kernel_moments",
            "residual_mmd",
            "correlation",
        ),
        default="none",
        help=(
            "Replace the joint prefix/future MMD with smooth train-fitted "
            "prefix-regime claims, with a train-fitted causal future-"
            "volatility residual-law MMD, or with a scale-invariant direct "
            "prefix/future realized-volatility correlation functional."
        ),
    )
    parser.add_argument(
        "--discrepancy-stack",
        choices=("full", "compact"),
        default="full",
        help=(
            "Use every legacy stylized-fact MMD block or retain only path, "
            "return, global-RV, and prefix/future conditional-law blocks."
        ),
    )
    parser.add_argument(
        "--conditional-moment-weight",
        type=float,
        default=1.0,
        help="Weight of the explicit within-prefix-regime mean/std term.",
    )
    parser.add_argument(
        "--conditional-mean-weight",
        type=float,
        default=1.0,
        help="Relative weight of conditional future log-RV mean matching.",
    )
    parser.add_argument(
        "--conditional-std-weight",
        type=float,
        default=1.0,
        help="Relative weight of conditional future log-RV spread matching.",
    )
    parser.add_argument(
        "--conditional-claim-weight",
        type=float,
        default=1.0,
        help=(
            "Weight of the smooth regime-mass/moment/tail claim term. Set "
            "to zero to retain only conditional regime mean/std matching."
        ),
    )
    parser.add_argument(
        "--conditional-claim-component-weights",
        type=float,
        nargs=6,
        default=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        metavar=("MASS", "MEAN", "SECOND", "LOWER", "MEDIAN", "UPPER"),
        help="Relative weights of the six smooth conditional claim components.",
    )
    parser.add_argument(
        "--conditional-future-quantile-levels",
        type=float,
        nargs=3,
        default=(0.10, 0.50, 0.90),
        metavar=("LOWER", "MEDIAN", "UPPER"),
        help="Train-law future log-RV quantiles used by the smooth tail claims.",
    )
    parser.add_argument(
        "--conditional-prefix-window",
        type=int,
        default=64,
        help=(
            "Number of returns immediately before the split used to define "
            "the smooth prefix-volatility regimes."
        ),
    )
    parser.add_argument(
        "--conditional-gate-temperature-fraction",
        type=float,
        default=0.25,
        help=(
            "Smooth prefix-regime gate temperature as a fraction of the "
            "train-law inter-tercile log-volatility gap. Smaller positive "
            "values localize the otherwise unchanged differentiable claims."
        ),
    )
    parser.add_argument(
        "--conditional-regime-weights",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("LOW", "MIDDLE", "HIGH"),
        help=(
            "Nonnegative train-law weights for the low, middle, and high "
            "smooth prefix-volatility regime claims."
        ),
    )
    parser.add_argument(
        "--joint-block-multiplier",
        type=float,
        default=1.0,
        help=(
            "Post-normalization multiplier for the prefix/future RV block; "
            "one preserves the frozen family normalization."
        ),
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=50.0,
        help="Nonpositive disables clipping; screening uses a high safety cap.",
    )
    parser.add_argument(
        "--sigma-cap-quantile",
        type=float,
        default=0.999,
        help=(
            "Train-only quantile of |return|/sqrt(dt) used to define the "
            "admissible volatility cap."
        ),
    )
    parser.add_argument(
        "--sigma-cap-std-multiple",
        type=float,
        default=5.0,
        help=(
            "Train-only unconditional-volatility multiple used as the second "
            "candidate for the admissible volatility cap."
        ),
    )
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument("--sample-batch-size", type=int, default=4096)
    parser.add_argument(
        "--screen-each-outer",
        action="store_true",
        help=(
            "For nested validation runs, save and score every accepted outer "
            "iterate using a common-random-number validation bank."
        ),
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _component_args(
    args: argparse.Namespace,
    *,
    reference_sigma_std_floor: float,
    admissible_sigma_max: float = 0.6,
    state_unit_scale: float = 1.0,
) -> argparse.Namespace:
    if not math.isfinite(float(state_unit_scale)) or float(state_unit_scale) <= 0.0:
        raise ValueError("state_unit_scale must be positive and finite")
    scale_corrected = str(args.objective_protocol) != "legacy_heston_scale"
    rich_reference = str(
        getattr(args, "reference_feature_basis", "compact")
    ) == "rich_generic"
    requested_causal_weight = getattr(args, "reference_causal_weight", None)
    calibration_weights = (
        (float(requested_causal_weight),)
        if requested_causal_weight is not None
        else (
            (0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0)
            if scale_corrected
            else (0.0, 0.25, 0.4, 0.55, 0.7, 0.75, 0.8, 0.85, 1.0)
        )
    )
    return argparse.Namespace(
        hidden_dim=96,
        num_layers=1,
        reference_ridge=1e-3,
        reference_variance_shrinkage=0.1,
        reference_log_variance_bias_correction=0.6,
        reference_kind=str(args.reference_kind),
        guyon_realized_volatility_window=int(
            args.guyon_realized_volatility_window
        ),
        causal_rolling_windows=(5, 10, 20, 32),
        causal_ewma_half_lives=(5, 20, 60),
        causal_downside_windows=(10, 32),
        causal_include_expanding_moments=rich_reference,
        causal_persistence_lags=((1, 5, 20) if rich_reference else ()),
        causal_leverage_lags=((0, 1, 5) if rich_reference else ()),
        causal_local_volatility_windows=((5, 20, 32) if rich_reference else ()),
        causal_variance_ridges=(1e-5, 1e-4, 1e-3),
        causal_crossfit_folds=3,
        causal_crossfit_seed=1701,
        causal_standardized_feature_clip=(4.0 if scale_corrected else None),
        causal_calibration_weights=calibration_weights,
        causal_calibration_paths=4096,
        causal_calibration_seed=271828,
        causal_calibration_offset_steps=(8 if scale_corrected else 3),
        causal_calibration_offset_relaxation=(0.5 if scale_corrected else 1.0),
        causal_calibration_return_std_tolerance=(0.03 if scale_corrected else 0.02),
        causal_calibration_sigma_std_floor=(
            float(reference_sigma_std_floor) / float(state_unit_scale)
        ),
        causal_calibration_sigma_cap_fraction_limit=0.01,
        causal_calibration_path_rv_std_ratio_min=(0.5 if scale_corrected else 0.0),
        causal_calibration_path_rv_std_ratio_max=(1.5 if scale_corrected else math.inf),
        causal_calibration_path_rv_quantile_spread_ratio_min=(
            float(getattr(args, "reference_rv_spread_ratio_min", 0.4))
            if scale_corrected
            else 0.0
        ),
        causal_calibration_path_rv_quantile_spread_ratio_max=(
            1.6 if scale_corrected else math.inf
        ),
        causal_calibration_clustering_lags=(1, 5, 20),
        eta=float(args.eta),
        running_cost=str(getattr(args, "running_cost", "specific_entropy")),
        use_reference_drift=bool(getattr(args, "use_reference_drift", False)),
        drift_control_mode=str(getattr(args, "drift_control_mode", "full")),
        drift_penalty=float(state_unit_scale) ** 2,
        sigma_min=1e-3 / float(state_unit_scale),
        reference_sigma_max=0.6 / float(state_unit_scale),
        sigma_max=float(admissible_sigma_max) / float(state_unit_scale),
        state_representation="log_price",
        time_horizon=1.0,
        joint_weight=float(args.joint_weight),
        discrepancy_preset=(
            "real_train_standardized_v1"
            if scale_corrected
            else "heston_old_fullv_w0p25"
        ),
        discrepancy_acf_lags=tuple(
            int(value)
            for value in getattr(
                args,
                "discrepancy_acf_lags",
                ((1, 2, 5, 10, 20, 32) if scale_corrected else (1, 2, 5, 10)),
            )
        ),
    )


def _train_only_admissible_sigma_max(
    train_prices: np.ndarray,
    *,
    model_dt: float,
    tail_quantile: float = 0.999,
    std_multiple: float = 5.0,
) -> float:
    """Set the bounded control set from robust train-only return scales."""

    prices = np.asarray(train_prices, dtype=np.float64)
    if prices.ndim != 2 or prices.shape[0] < 2 or prices.shape[1] < 2:
        raise ValueError("train_prices must have shape (sessions, points)")
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ValueError("train_prices must be finite and positive")
    returns = np.diff(np.log(prices), axis=1)
    instantaneous = np.abs(returns) / math.sqrt(float(model_dt))
    if not 0.0 < float(tail_quantile) < 1.0:
        raise ValueError("tail_quantile must lie strictly between zero and one")
    if not math.isfinite(float(std_multiple)) or float(std_multiple) <= 0.0:
        raise ValueError("std_multiple must be positive and finite")
    robust_tail = float(np.quantile(instantaneous, float(tail_quantile)))
    unconditional_scale = float(returns.std(ddof=0)) / math.sqrt(
        float(model_dt)
    )
    return max(0.01, robust_tail, float(std_multiple) * unconditional_scale)


def _training_config(args: argparse.Namespace) -> DiscreteMPTrainingConfig:
    clip = (
        None
        if float(args.grad_clip_norm) <= 0.0
        else float(args.grad_clip_norm)
    )
    phase_offset = 0 if str(args.phase) == "source" else 10_000
    return DiscreteMPTrainingConfig(
        batch_size=256,
        target_batch_size=256,
        num_steps=int(args.steps),
        lr=float(args.lr),
        weight_decay=1e-5,
        lambda_scale=float(args.lambda_scale),
        adjoint_weight=float(args.adjoint_weight),
        adjoint_noise_weight=float(args.adjoint_noise_weight),
        ce_target_mode=str(args.ce_target_mode),
        ridge_lambda=float(args.ridge_lambda),
        ce_crossfit_folds=int(args.ce_crossfit_folds),
        target_preconditioner="none",
        preconditioner_batches=int(args.preconditioner_batches),
        noise_target_control_variate=str(args.noise_target_control_variate),
        noise_target_estimator=str(args.noise_target_estimator),
        stein_num_probes=int(args.stein_probes),
        grad_clip_norm=clip,
        log_every=int(args.log_every),
        seed=int(args.seed) + phase_offset,
        observed_only=False,
    )


def _configure_r_ce_estimator(
    model,
    args: argparse.Namespace,
    *,
    target_paths: torch.Tensor,
) -> None:
    common = {
        "grid": model.grid,
        "ridge": float(args.ridge_lambda),
        "rolling_windows": (5, 10, 20, 32),
        "ewma_half_lives": (5, 20, 60),
        "downside_windows": (10, 32),
        "standardized_feature_clip": 4.0,
    }

    def causal_estimator(
        *, rich: bool, volatility_gates: bool = False
    ) -> CausalPathFeatureRidgeConditionalExpectation:
        extra = (
            {
                "include_expanding_moments": True,
                "persistence_lags": (1, 5, 20),
                "leverage_lags": (0, 1, 5),
                "local_volatility_windows": (5, 20, 32),
            }
            if rich
            else {}
        )
        estimator = CausalPathFeatureRidgeConditionalExpectation(
            **common,
            **extra,
            include_volatility_gates=bool(volatility_gates),
            pool_time=str(args.r_ce_time_mode) == "pooled",
        )
        if str(args.r_ce_normalization) == "train_fixed":
            center, scale = fit_timewise_causal_feature_normalization(
                estimator,
                target_paths,
                minimum_scale=1e-6,
            )
            estimator = CausalPathFeatureRidgeConditionalExpectation(
                **common,
                **extra,
                include_volatility_gates=bool(volatility_gates),
                fixed_feature_center=center,
                fixed_feature_scale=scale,
                pool_time=str(args.r_ce_time_mode) == "pooled",
            )
        return estimator

    if str(args.p_ce_basis) == "causal_compact":
        model.ce_estimator = causal_estimator(rich=False)

    basis = str(args.r_ce_basis)
    if basis == "shared_flat_prefix":
        model.noise_ce_estimator = None
    elif basis == "causal_compact":
        model.noise_ce_estimator = causal_estimator(rich=False)
    elif basis == "causal_compact_gated":
        model.noise_ce_estimator = causal_estimator(
            rich=False,
            volatility_gates=True,
        )
    elif basis == "causal_rich":
        model.noise_ce_estimator = causal_estimator(rich=True)
    else:
        raise ValueError(f"unsupported R conditional basis {basis}")


def _configure_adjoint_network(
    model,
    args: argparse.Namespace,
    *,
    target_paths: torch.Tensor,
) -> dict[str, Any]:
    kind = str(args.adjoint_network)
    if kind == "plain":
        return {"kind": "plain", "stack_depth": 0}
    if kind not in {
        "increment_residual",
        "compact_residual",
        "timewise_linear_residual",
        "causal_residual",
        "causal_regime_residual",
    }:
        raise ValueError(f"unsupported adjoint network {kind}")
    source_phase = str(args.phase) == "source"
    if kind == "increment_residual":
        feature_map = IncrementCausalFeatureMap(
            num_steps=int(model.grid.num_steps),
            dt=float(model.grid.dt),
        )
    elif kind in {"compact_residual", "timewise_linear_residual"}:
        feature_map = CompactVolatilityCausalFeatureMap(grid=model.grid)
    else:
        feature_map = model.noise_ce_estimator
        if not isinstance(
            feature_map, CausalPathFeatureRidgeConditionalExpectation
        ):
            raise ValueError(
                "causal_residual requires a causal compact/rich R feature map"
            )
    center, scale = fit_timewise_causal_feature_normalization(
        feature_map,
        target_paths,
        minimum_scale=1e-6,
    )
    residual_init_seed = int(args.seed) + 310_000
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(residual_init_seed)
        common = {
            "state_dim": int(model.architecture.state_dim),
            "adjoint_dim": int(model.architecture.state_dim),
            "noise_adjoint_dim": int(
                model._default_noise_adjoint_dim(model.architecture)
            ),
            "feature_map": feature_map,
            "feature_center": center,
            "feature_scale": scale,
            "standardized_feature_clip": 4.0,
        }
        if kind == "timewise_linear_residual":
            residual = TimewiseLinearCausalAdjointResidualNetwork(**common)
        elif kind == "causal_regime_residual":
            gate_feature_index, gate_thresholds, gate_temperature = (
                fit_timewise_causal_regime_gates(
                    feature_map,
                    target_paths,
                    feature_name=str(args.adjoint_regime_gate_feature),
                    temperature_fraction=float(
                        args.adjoint_regime_gate_temperature_fraction
                    ),
                )
            )
            residual = CausalRegimeGatedAdjointResidualNetwork(
                **common,
                gate_feature_index=gate_feature_index,
                gate_thresholds=gate_thresholds,
                gate_temperature=gate_temperature,
                hidden_dim=int(model.architecture.hidden_dim),
                num_layers=int(model.architecture.num_layers),
            )
        else:
            residual = CausalFeatureAdjointResidualNetwork(
                **common,
                hidden_dim=int(model.architecture.hidden_dim),
                num_layers=int(model.architecture.num_layers),
            )
    residual = residual.to(device=model.device, dtype=model.dtype)
    if source_phase:
        with torch.no_grad():
            base_probe = model.network(target_paths[:2])
        if not (
            bool((base_probe.expected_adjoint_next == 0.0).all().item())
            and bool(
                (base_probe.expected_adjoint_noise_next == 0.0).all().item()
            )
        ):
            raise ValueError(
                "a source-phase residual requires the exact zero-adjoint "
                "calibrated reference as its frozen base"
            )
    base_stack_depth = _adjoint_network_stack_depth(model.network)
    combined = FrozenBaseCausalResidualAdjointNetwork(
        base_network=model.network,
        residual_network=residual,
    ).to(device=model.device, dtype=model.dtype)
    model.network = combined
    model._validate_network_contract(
        network=combined,
        architecture=model.architecture,
    )
    summary = {
        "kind": kind,
        "stack_depth": base_stack_depth + 1,
        "source_policy_frozen": True,
        "frozen_base_role": (
            "zero-adjoint calibrated reference"
            if source_phase
            else "fitted source checkpoint"
        ),
        "residual_zero_initialized": True,
        "residual_init_seed": residual_init_seed,
        "feature_names": list(residual.feature_names),
        "feature_count": len(residual.feature_names),
        "feature_normalization": "timewise train-law mean/std",
        "standardized_feature_clip": 4.0,
    }
    if kind == "causal_regime_residual":
        summary.update(
            {
                "regime_gate_feature": str(residual.gate_feature_name),
                "regime_gate_count": 3,
                "regime_gate_temperature_fraction": float(
                    args.adjoint_regime_gate_temperature_fraction
                ),
                "regime_gate_fit_scope": "complete training split only",
            }
        )
    return summary


def _adjoint_network_stack_depth(network: torch.nn.Module) -> int:
    """Count additive frozen-base residual layers in an adjoint network."""

    depth = 0
    current = network
    while isinstance(current, FrozenBaseCausalResidualAdjointNetwork):
        depth += 1
        current = current.base_network
    return depth


def _configure_adjoint_network_stack(
    model,
    *,
    kind: str,
    stack_depth: int,
    seed: int,
    target_paths: torch.Tensor,
    regime_gate_temperature_fraction: float = 0.10,
    regime_gate_feature: str = "rolling_realized_variance_20",
) -> list[dict[str, Any]]:
    """Reconstruct a recorded additive residual topology before loading state."""

    depth = int(stack_depth)
    if depth < 0:
        raise ValueError("adjoint residual stack depth must be nonnegative")
    if str(kind) == "plain":
        if depth != 0:
            raise ValueError("a plain adjoint network must have stack depth zero")
        return []
    if depth < 1:
        raise ValueError("a residual adjoint network must have positive stack depth")
    existing_depth = _adjoint_network_stack_depth(model.network)
    if existing_depth > depth:
        raise ValueError("existing adjoint residual stack exceeds requested depth")
    summaries = []
    for layer_index in range(existing_depth, depth):
        summaries.append(
            _configure_adjoint_network(
                model,
                argparse.Namespace(
                    adjoint_network=str(kind),
                    phase=("source" if layer_index == 0 else "joint"),
                    seed=int(seed),
                    adjoint_regime_gate_temperature_fraction=float(
                        regime_gate_temperature_fraction
                    ),
                    adjoint_regime_gate_feature=str(regime_gate_feature),
                ),
                target_paths=target_paths,
            )
        )
    return summaries


def _sample_price_bank(
    model,
    *,
    transform,
    bank_size: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    state_unit_scale: float = 1.0,
) -> np.ndarray:
    if not math.isfinite(float(state_unit_scale)) or float(state_unit_scale) <= 0.0:
        raise ValueError("state_unit_scale must be positive and finite")
    bank = np.empty((int(bank_size), NUM_STEPS + 1), dtype=np.float32)
    x0 = torch.zeros((1, 1), device=device, dtype=dtype)
    was_training = bool(model.network.training)
    model.network.eval()
    try:
        with torch.no_grad():
            for batch_index, start in enumerate(
                range(0, int(bank_size), int(batch_size))
            ):
                count = min(int(batch_size), int(bank_size) - start)
                sample = model.sample(
                    num_paths=count,
                    x0=x0,
                    seed=derive_stream_seed(
                        base_seed=int(seed),
                        stream_offset=50_000 + batch_index,
                    ),
                )
                states = (
                    sample.paths[..., 0].detach().cpu().numpy()
                    * float(state_unit_scale)
                )
                return_states = np.zeros_like(states)
                return_states[:, 1:] = (
                    np.diff(states, axis=1)
                    * math.sqrt(float(transform.dt))
                    / float(transform.sigma)
                )
                bank[start : start + count] = transform.inverse(
                    return_states,
                    dtype=np.float32,
                )
    finally:
        model.network.train(was_training)
    return bank


def _history_summary(history: list[dict[str, float]]) -> dict[str, float | bool]:
    def history_value(record: dict[str, float], name: str) -> float:
        nested_fallbacks = {
            "train_total_loss": "accepted_fixed_total_loss",
            "grad_norm": "inner_grad_norm_max",
            "grad_clip_active": "inner_clip_fraction",
        }
        if name in nested_fallbacks and name not in record:
            return float(record.get(nested_fallbacks[name], math.nan))
        if name == "grad_clip_scale" and name not in record:
            # Nested training records the inner clipping fraction.  When it is
            # zero, the effective scale was identically one.
            clip_fraction = float(record.get("inner_clip_fraction", math.nan))
            return 1.0 if clip_fraction == 0.0 else math.nan
        if name == "gradient_nonfinite_fraction":
            return float(
                record.get(
                    name,
                    record.get("rolling_gradient_nonfinite_fraction_max", math.nan),
                )
            )
        return float(record.get(name, math.nan))

    tracked = {
        name: np.asarray(
            [history_value(record, name) for record in history],
            dtype=np.float64,
        )
        for name in (
            "train_total_loss",
            "sigma_mean",
            "sigma_q99",
            "sigma_max_active_fraction",
            "grad_norm",
            "grad_clip_scale",
            "grad_clip_active",
            "gradient_nonfinite_fraction",
        )
    }
    finite = bool(
        all(np.isfinite(values).all() for values in tracked.values())
    )

    def finite_stat(name: str, operation, default: float = math.nan) -> float:
        values = tracked[name]
        values = values[np.isfinite(values)]
        return default if values.size == 0 else float(operation(values))

    return {
        "all_logged_quantities_finite": finite,
        "logged_records": int(len(history)),
        "loss_max": finite_stat("train_total_loss", np.max),
        "sigma_mean_min": finite_stat("sigma_mean", np.min),
        "sigma_mean_max": finite_stat("sigma_mean", np.max),
        "sigma_q99_max": finite_stat("sigma_q99", np.max),
        "sigma_cap_fraction_max": finite_stat(
            "sigma_max_active_fraction", np.max
        ),
        "grad_norm_max": finite_stat("grad_norm", np.max),
        "clip_active_fraction": finite_stat("grad_clip_active", np.mean),
        "clip_scale_min": finite_stat("grad_clip_scale", np.min),
        "clip_scale_median": finite_stat("grad_clip_scale", np.median),
        "gradient_nonfinite_fraction_max": finite_stat(
            "gradient_nonfinite_fraction", np.max
        ),
    }


def _bank_diagnostics(
    generated: np.ndarray,
    validation: np.ndarray,
) -> dict[str, Any]:
    generated_returns = np.diff(np.log(generated.astype(np.float64)), axis=1)
    validation_returns = np.diff(np.log(validation.astype(np.float64)), axis=1)
    generated_rv = np.sqrt(np.sum(generated_returns**2, axis=1))
    validation_rv = np.sqrt(np.sum(validation_returns**2, axis=1))
    validation_q90 = float(np.quantile(validation_rv, 0.90))
    return {
        "generated_price_min": float(generated.min()),
        "generated_price_max": float(generated.max()),
        "generated_return_std": float(generated_returns.std(ddof=0)),
        "validation_return_std": float(validation_returns.std(ddof=0)),
        "generated_rv_max": float(generated_rv.max()),
        "validation_rv_max": float(validation_rv.max()),
        "generated_rv_max_to_validation_max": float(
            generated_rv.max() / validation_rv.max()
        ),
        "generated_fraction_above_validation_rv_q90": float(
            np.mean(generated_rv > validation_q90)
        ),
        "generated_paths_outside_95_105": int(
            np.sum(
                (generated.min(axis=1) < 95.0)
                | (generated.max(axis=1) > 105.0)
            )
        ),
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scale_named_discrepancy_block(
    discrepancy: CompositePathFunctionalDiscrepancy,
    *,
    name: str,
    multiplier: float,
) -> CompositePathFunctionalDiscrepancy:
    """Scale one already-normalized block without changing the others."""

    found = False
    scaled_blocks = []
    for block in discrepancy.blocks:
        if str(block.name) == str(name):
            found = True
            scaled_blocks.append(
                WeightedDiscrepancy(
                    name=str(block.name),
                    weight=float(block.weight) * float(multiplier),
                    discrepancy=block.discrepancy,
                )
            )
        else:
            scaled_blocks.append(block)
    if not found:
        raise ValueError(f"discrepancy is missing {name}")
    return CompositePathFunctionalDiscrepancy(blocks=tuple(scaled_blocks))


def _replace_named_discrepancy_block(
    discrepancy: CompositePathFunctionalDiscrepancy,
    *,
    name: str,
    replacement: object,
) -> CompositePathFunctionalDiscrepancy:
    """Replace one block's path functional while preserving its base weight."""

    found = False
    blocks = []
    for block in discrepancy.blocks:
        if str(block.name) == str(name):
            found = True
            blocks.append(
                WeightedDiscrepancy(
                    name=str(block.name),
                    weight=float(block.weight),
                    discrepancy=replacement,
                )
            )
        else:
            blocks.append(block)
    if not found:
        raise ValueError(f"discrepancy is missing {name}")
    return CompositePathFunctionalDiscrepancy(blocks=tuple(blocks))


def _fit_direct_objective_diagnostic(
    model,
    target_paths: torch.Tensor,
    *,
    training: DiscreteMPTrainingConfig,
    log_callback,
) -> FitResult:
    """Optimize the unchanged complete objective through the rollout.

    This deliberately is not an alternative MP solver.  It is the matched
    direct-backpropagation control requested by the paper protocol: identical
    forward architecture, reference, admissible controls, running cost, and
    path discrepancy.  Its sole purpose here is to distinguish an
    unattainable objective from a failure of the projected-adjoint solver.
    """

    model.training = training
    target_paths = target_paths.to(device=model.device, dtype=model.dtype)
    target_count, _ = model._validate_target_paths(target_paths)
    cpu_generator = torch.Generator(device="cpu")
    if training.seed is not None:
        cpu_generator.manual_seed(int(training.seed) + 3_000_000)
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
                stream_offset=3_000_001 + step,
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
            raise RuntimeError("direct-objective diagnostic became nonfinite")
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
                parameters, float(training.grad_clip_norm)
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
                "direct_objective_diagnostic": 1.0,
            }
            history.append(metrics)
            if log_callback is not None:
                log_callback(model, metrics)
    return FitResult(
        history=history,
        final_metrics=(dict(history[-1]) if history else {}),
    )


def main() -> None:
    wall_started = time.perf_counter()
    args = _parse_args()
    if int(args.steps) < 1 or int(args.bank_size) < 256:
        raise ValueError("steps must be positive and bank-size must be at least 256")
    online_checkpoint_steps = tuple(int(value) for value in args.online_checkpoint_steps)
    if online_checkpoint_steps != tuple(sorted(set(online_checkpoint_steps))):
        raise ValueError("online-checkpoint-steps must be unique and increasing")
    if any(value < 1 or value >= int(args.steps) for value in online_checkpoint_steps):
        raise ValueError("online-checkpoint-steps must lie in [1, steps)")
    if online_checkpoint_steps and str(args.solver) != "online":
        raise ValueError("online-checkpoint-steps require --solver online")
    if min(
        float(args.lr),
        float(args.ridge_lambda),
        float(args.lambda_scale),
        float(args.eta),
        float(args.lambda_x),
        float(args.lambda_v),
        float(args.joint_weight),
        float(args.joint_block_multiplier),
    ) <= 0.0:
        raise ValueError("optimization and objective hyperparameters must be positive")
    if float(args.conditional_moment_weight) < 0.0:
        raise ValueError("conditional-moment-weight must be nonnegative")
    if float(args.conditional_mean_weight) < 0.0:
        raise ValueError("conditional-mean-weight must be nonnegative")
    if float(args.conditional_std_weight) < 0.0:
        raise ValueError("conditional-std-weight must be nonnegative")
    if (
        float(args.conditional_mean_weight) == 0.0
        and float(args.conditional_std_weight) == 0.0
    ):
        raise ValueError(
            "conditional mean and std weights cannot both be zero"
        )
    if float(args.conditional_claim_weight) < 0.0:
        raise ValueError("conditional-claim-weight must be nonnegative")
    if (
        any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in args.conditional_claim_component_weights
        )
        or sum(float(value) for value in args.conditional_claim_component_weights)
        <= 0.0
    ):
        raise ValueError(
            "conditional claim component weights must be nonnegative with positive sum"
        )
    conditional_quantiles = tuple(
        float(value) for value in args.conditional_future_quantile_levels
    )
    if not (
        0.0
        < conditional_quantiles[0]
        < conditional_quantiles[1]
        < conditional_quantiles[2]
        < 1.0
    ):
        raise ValueError(
            "conditional future quantile levels must be increasing in (0, 1)"
        )
    if (
        not math.isfinite(float(args.adjoint_regime_gate_temperature_fraction))
        or float(args.adjoint_regime_gate_temperature_fraction) <= 0.0
    ):
        raise ValueError(
            "adjoint-regime-gate-temperature-fraction must be positive and finite"
        )
    if (
        args.source_checkpoint_stack_depth is not None
        and int(args.source_checkpoint_stack_depth) < 1
    ):
        raise ValueError("source-checkpoint-stack-depth must be positive")
    if (
        not math.isfinite(float(args.conditional_gate_temperature_fraction))
        or float(args.conditional_gate_temperature_fraction) <= 0.0
    ):
        raise ValueError(
            "conditional-gate-temperature-fraction must be positive and finite"
        )
    if (
        float(args.conditional_claim_weight) == 0.0
        and float(args.conditional_moment_weight) == 0.0
    ):
        raise ValueError(
            "conditional claim and moment weights cannot both be zero"
        )
    if int(args.ce_crossfit_folds) < 1:
        raise ValueError("ce-crossfit-folds must be positive")
    if str(args.ce_target_mode) == "direct" and not (
        str(args.solver) == "nested"
        and str(args.nested_outer_target_estimator) == "nested_branches"
    ):
        raise ValueError(
            "direct CE targets require the nested solver with nested_branches"
        )
    if (
        str(args.ce_target_mode) == "direct"
        and int(args.ce_crossfit_folds) != 1
    ):
        raise ValueError("direct CE targets require ce-crossfit-folds=1")
    if int(args.preconditioner_batches) < 1:
        raise ValueError("preconditioner-batches must be positive")
    if not 0.0 <= float(args.reference_rv_spread_ratio_min) <= 1.0:
        raise ValueError("reference-rv-spread-ratio-min must lie in [0, 1]")
    if args.reference_causal_weight is not None and not 0.0 <= float(
        args.reference_causal_weight
    ) <= 1.0:
        raise ValueError("reference-causal-weight must lie in [0, 1]")
    if str(args.solver) == "nested":
        if int(args.steps) != int(args.nested_outer_steps) * int(
            args.nested_inner_steps
        ):
            raise ValueError(
                "for nested training, --steps must equal "
                "--nested-outer-steps times --nested-inner-steps"
            )
    if (
        args.phase == "joint"
        and args.source_checkpoint is None
        and args.resume_checkpoint is None
    ):
        raise ValueError(
            "--source-checkpoint or --resume-checkpoint is required for a joint candidate"
        )

    run_dir = ensure_run_dir(args.run_dir)
    manifest_path = run_dir / "candidate_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    fit = load_frozen_real_data_fit(
        args.canonical_root,
        protocol_identifier=args.protocol,
        index=args.index,
    )
    dataset_root = run_dir / "adapter_dataset"
    materialize_deep_mkv_adapter_dataset(fit, dataset_root)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    model_dt = 1.0 / NUM_STEPS
    target_paths, transform, preprocessing = _load_training_data(
        args.benchmark_root,
        device=device,
        dtype=dtype,
        state_representation="log_price",
        custom_dataset_root=dataset_root,
        model_dt=model_dt,
    )
    state_unit_scale = 1.0
    if str(args.state_unit_scaling) == "train_return_volatility":
        state_unit_scale = float(transform.sigma) / math.sqrt(model_dt)
        target_paths = target_paths / state_unit_scale
    train_returns = np.diff(np.log(fit.train_prices.astype(np.float64)), axis=1)
    reference_sigma_std_floor = (
        0.25 * float(train_returns.std(ddof=0)) / math.sqrt(model_dt)
    )
    admissible_sigma_max = _train_only_admissible_sigma_max(
        fit.train_prices,
        model_dt=model_dt,
        tail_quantile=float(args.sigma_cap_quantile),
        std_multiple=float(args.sigma_cap_std_multiple),
    )
    component_args = _component_args(
        args,
        reference_sigma_std_floor=reference_sigma_std_floor,
        admissible_sigma_max=admissible_sigma_max,
        state_unit_scale=state_unit_scale,
    )
    grid, architecture, control, base, joint = _build_components(
        component_args,
        target_paths=target_paths,
        path_scale=float(transform.sigma) / math.sqrt(float(transform.dt)),
    )
    discrepancy = base if args.phase == "source" else joint
    return_summary: dict[str, Any] = {"mode": str(args.return_discrepancy)}
    if str(args.return_discrepancy) == "wasserstein":
        if str(args.phase) != "joint":
            raise ValueError("the return Wasserstein block is only defined for the joint phase")
        if args.frozen_discrepancy_normalization_manifest is not None:
            raise ValueError(
                "a changed return-law functional requires fresh train-only normalization"
            )
        pooled_return = fit_fixed_target_pooled_return_wasserstein(
            target_paths,
            grid=grid,
        )
        discrepancy = _replace_named_discrepancy_block(
            discrepancy,
            name="path_law_returns",
            replacement=pooled_return,
        )
        return_summary.update(
            {
                "pooling": "paths and intraday time, separately by asset",
                "distance": "standardized empirical one-dimensional Wasserstein-2",
                "target_fit_scope": "complete training split only",
            }
        )
    global_rv_summary: dict[str, Any] = {"mode": str(args.global_rv_discrepancy)}
    if str(args.global_rv_discrepancy) in {
        "wasserstein",
        "log_wasserstein",
        "dated_wasserstein",
        "dated_log_wasserstein",
        "block_wasserstein",
        "block_log_wasserstein",
    }:
        if str(args.phase) != "joint":
            raise ValueError("the global RV Wasserstein block is only defined for the joint phase")
        if args.frozen_discrepancy_normalization_manifest is not None:
            raise ValueError(
                "a changed global RV functional requires fresh train-only normalization"
            )
        dated = str(args.global_rv_discrepancy).startswith("dated_")
        block = str(args.global_rv_discrepancy).startswith("block_")
        log_transform = "log" in str(args.global_rv_discrepancy)
        pairs = (
            ((32, 32), (32, 64), (32, 96), (32, int(grid.num_steps)))
            if block
            else (
                ((32, 32), (64, 64), (96, 96), (int(grid.num_steps), int(grid.num_steps)))
                if dated
                else ((int(grid.num_steps), int(grid.num_steps)),)
            )
        )
        global_rv = fit_fixed_target_time_marginal_volatility(
            target_paths,
            grid=grid,
            window_endpoint_pairs=pairs,
            value_transform=("log" if log_transform else "identity"),
        )
        discrepancy = _replace_named_discrepancy_block(
            discrepancy,
            name="volatility_law_global_rv",
            replacement=global_rv,
        )
        global_rv_summary.update(
            {
                "window_endpoint_pairs": [list(pair) for pair in pairs],
                "distance": "standardized empirical one-dimensional Wasserstein-2",
                "value_transform": ("log" if log_transform else "identity"),
                "intermediate_marginals": bool(dated or block),
                "marginal_semantics": (
                    "nonoverlapping_32_step_blocks"
                    if block
                    else ("expanding_from_open" if dated else "full_session")
                ),
                "target_fit_scope": "complete training split only",
            }
        )
    acf_summary: dict[str, Any] = {"mode": str(args.acf_discrepancy)}
    if str(args.acf_discrepancy) == "moment":
        if args.frozen_discrepancy_normalization_manifest is not None:
            raise ValueError(
                "a changed ACF law functional requires fresh train-only normalization"
            )
        acf_lags = tuple(
            int(value) for value in component_args.discrepancy_acf_lags
        )
        absolute_acf = fit_fixed_target_acf_moments(
            target_paths,
            grid=grid,
            lags=acf_lags,
            transform="absolute",
            name="absolute_return_acf_moments",
        )
        squared_acf = fit_fixed_target_acf_moments(
            target_paths,
            grid=grid,
            lags=acf_lags,
            transform="squared",
            name="squared_return_acf_moments",
        )
        discrepancy = _replace_named_discrepancy_block(
            discrepancy,
            name="volatility_law_abs_return_acf",
            replacement=absolute_acf,
        )
        discrepancy = _replace_named_discrepancy_block(
            discrepancy,
            name="volatility_law_squared_return_acf",
            replacement=squared_acf,
        )
        acf_summary.update(
            {
                "lags": list(acf_lags),
                "statistics": [
                    "pooled absolute-return autocorrelation",
                    "pooled squared-return autocorrelation",
                ],
                "target_fit_scope": "complete training split only",
                "reference_dependent": False,
            }
        )
    conditional_claim_summary: dict[str, Any] | None = None
    if str(args.conditional_claim_mode) == "replace_joint":
        if str(args.phase) != "joint":
            raise ValueError("conditional claims are only defined for the joint phase")
        if not isinstance(discrepancy, CompositePathFunctionalDiscrepancy):
            raise ValueError("conditional claims require a composite discrepancy")
        conditional_prefix_window = int(args.conditional_prefix_window)
        conditional_gate_temperature_fraction = float(
            args.conditional_gate_temperature_fraction
        )
        if conditional_prefix_window < 1 or conditional_prefix_window > 64:
            raise ValueError("conditional-prefix-window must lie in [1, 64]")
        conditional_claim = (
            fit_fixed_target_prefix_regime_future_volatility_claims(
                target_paths,
                grid=grid,
                split_index=64,
                prefix_window=conditional_prefix_window,
                future_horizon=32,
                gate_temperature_fraction=conditional_gate_temperature_fraction,
                tail_temperature=0.15,
                conditional_moment_weight=float(
                    args.conditional_moment_weight
                ),
                conditional_mean_weight=float(args.conditional_mean_weight),
                conditional_std_weight=float(args.conditional_std_weight),
                claim_component_weights=tuple(
                    float(value)
                    for value in args.conditional_claim_component_weights
                ),
                future_quantile_levels=conditional_quantiles,
                claim_weight=float(args.conditional_claim_weight),
                regime_weights=tuple(
                    float(value) for value in args.conditional_regime_weights
                ),
            )
        )
        discrepancy = _replace_named_discrepancy_block(
            discrepancy,
            name="volatility_law_joint_prefix_future_rv",
            replacement=conditional_claim,
        )
        conditional_claim_summary = {
            "mode": "replace_joint",
            "split_index": 64,
            "prefix_window": conditional_prefix_window,
            "future_horizon": 32,
            "prefix_regimes": 3,
            "gate_temperature_fraction": conditional_gate_temperature_fraction,
            "tail_temperature": 0.15,
            "conditional_moment_weight": float(
                args.conditional_moment_weight
            ),
            "conditional_mean_weight": float(args.conditional_mean_weight),
            "conditional_std_weight": float(args.conditional_std_weight),
            "claim_component_weights": [
                float(value)
                for value in args.conditional_claim_component_weights
            ],
            "future_quantile_levels": list(conditional_quantiles),
            "claim_weight": float(args.conditional_claim_weight),
            "regime_weights": [
                float(value) for value in args.conditional_regime_weights
            ],
            "target_fit_scope": "complete training split only",
        }
    elif str(args.conditional_claim_mode) == "kernel_moments":
        if str(args.phase) != "joint":
            raise ValueError("conditional kernel moments require the joint phase")
        if not isinstance(discrepancy, CompositePathFunctionalDiscrepancy):
            raise ValueError("conditional kernel moments require a composite discrepancy")
        conditional_prefix_window = int(args.conditional_prefix_window)
        conditional_claim = fit_fixed_target_prefix_future_volatility_kernel_moments(
            target_paths,
            grid=grid,
            split_index=64,
            prefix_window=conditional_prefix_window,
            future_horizon=32,
            kernel_quantiles=(0.10, 0.30, 0.50, 0.70, 0.90),
            bandwidth_multiplier=1.0,
            mean_weight=float(args.conditional_mean_weight),
            std_weight=float(args.conditional_std_weight),
        )
        discrepancy = _replace_named_discrepancy_block(
            discrepancy,
            name="volatility_law_joint_prefix_future_rv",
            replacement=conditional_claim,
        )
        conditional_claim_summary = {
            "mode": "kernel_moments",
            "split_index": 64,
            "prefix_window": conditional_prefix_window,
            "future_horizon": 32,
            "kernel_quantiles": [0.10, 0.30, 0.50, 0.70, 0.90],
            "bandwidth_multiplier": 1.0,
            "conditional_mean_weight": float(args.conditional_mean_weight),
            "conditional_std_weight": float(args.conditional_std_weight),
            "target_fit_scope": "complete training split only",
            "reference_dependent": False,
        }
    elif str(args.conditional_claim_mode) == "residual_mmd":
        if str(args.phase) != "joint":
            raise ValueError(
                "conditional residual MMD is only defined for the joint phase"
            )
        if not isinstance(discrepancy, CompositePathFunctionalDiscrepancy):
            raise ValueError(
                "conditional residual MMD requires a composite discrepancy"
            )
        conditional_residual = (
            fit_fixed_target_conditional_residual_volatility_mmd(
                target_paths,
                grid=grid,
                split_index=64,
                future_horizon=32,
                prefix_windows=(16, 32, 64),
                recent_return_window=16,
                ridge=1e-2,
                bandwidths=(0.5, 1.0, 2.0),
                moment_weight=1.0,
            )
        )
        discrepancy = _replace_named_discrepancy_block(
            discrepancy,
            name="volatility_law_joint_prefix_future_rv",
            replacement=conditional_residual,
        )
        conditional_claim_summary = {
            "mode": "residual_mmd",
            "split_index": 64,
            "future_horizon": 32,
            "prefix_windows": [16, 32, 64],
            "recent_return_window": 16,
            "predictor": "train-fitted standardized causal ridge",
            "predictor_ridge": 1e-2,
            "distance": (
                "RBF MMD plus first/second-moment error on standardized "
                "future-log-RV residuals"
            ),
            "moment_weight": 1.0,
            "target_fit_scope": "complete training split only",
            "reference_dependent": False,
        }
    elif str(args.conditional_claim_mode) == "correlation":
        if str(args.phase) != "joint":
            raise ValueError(
                "conditional correlation is only defined for the joint phase"
            )
        if not isinstance(discrepancy, CompositePathFunctionalDiscrepancy):
            raise ValueError(
                "conditional correlation requires a composite discrepancy"
            )
        conditional_correlation = (
            StabilizedPrefixFutureVolatilityCorrelationDiscrepancy(
                windows=(4, 8, 16, 32),
                full_matrix=False,
                split_index=64,
                scale_floor_fraction=0.25,
            )
        )
        discrepancy = _replace_named_discrepancy_block(
            discrepancy,
            name="volatility_law_joint_prefix_future_rv",
            replacement=conditional_correlation,
        )
        conditional_claim_summary = {
            "mode": "correlation",
            "split_index": 64,
            "windows": [4, 8, 16, 32],
            "full_matrix": False,
            "scale_floor_fraction": 0.25,
            "marginal_invariant_above_scale_floor": True,
            "target_fit_scope": "complete training split only",
            "reference_dependent": False,
        }
    retained_discrepancy_blocks: list[str] | None = None
    if str(args.discrepancy_stack) == "compact":
        if not isinstance(discrepancy, CompositePathFunctionalDiscrepancy):
            raise ValueError("compact discrepancy stack requires a composite discrepancy")
        retained_names = {
            "path_law_observed_path",
            "path_law_returns",
            "volatility_law_global_rv",
            "volatility_law_joint_prefix_future_rv",
        }
        retained = tuple(
            block for block in discrepancy.blocks if str(block.name) in retained_names
        )
        retained_discrepancy_blocks = [str(block.name) for block in retained]
        if set(retained_discrepancy_blocks) != retained_names:
            missing = sorted(retained_names.difference(retained_discrepancy_blocks))
            raise ValueError(f"compact discrepancy stack is missing blocks: {missing}")
        discrepancy = CompositePathFunctionalDiscrepancy(blocks=retained)
    model = _build_model(
        architecture=architecture,
        grid=grid,
        control=control,
        discrepancy=discrepancy,
        device=device,
        dtype=dtype,
        seed=int(args.seed),
    )
    source_estimator_args = args
    if (
        args.phase == "joint"
        and args.source_checkpoint is not None
        and str(args.source_checkpoint_topology) == "causal_residual"
        and str(args.source_checkpoint_r_ce_basis) != str(args.r_ce_basis)
    ):
        source_estimator_args = argparse.Namespace(**vars(args))
        source_estimator_args.r_ce_basis = str(args.source_checkpoint_r_ce_basis)
    _configure_r_ce_estimator(
        model,
        source_estimator_args,
        target_paths=target_paths,
    )
    source_provenance: dict[str, Any] | None = None
    if args.phase == "joint" and args.source_checkpoint is not None:
        source_checkpoint = args.source_checkpoint.resolve()
        if not source_checkpoint.is_file():
            raise FileNotFoundError(source_checkpoint)
        source_topology = str(args.source_checkpoint_topology)
        if source_topology == "causal_residual":
            source_stack_depth = (
                1
                if args.source_checkpoint_stack_depth is None
                else int(args.source_checkpoint_stack_depth)
            )
            _configure_adjoint_network_stack(
                model,
                kind="causal_residual",
                stack_depth=source_stack_depth,
                seed=int(args.seed),
                target_paths=target_paths,
            )
        checkpoint = torch.load(
            source_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        model.load_checkpoint_state(checkpoint)
        source_provenance = {
            "path": str(source_checkpoint),
            "sha256": _file_sha256(source_checkpoint),
            "load_mode": f"{source_topology}_source_then_new_residual",
            "source_topology": source_topology,
            "source_r_ce_basis": str(args.source_checkpoint_r_ce_basis),
            "stack_depth": _adjoint_network_stack_depth(model.network),
        }
        if str(args.source_checkpoint_r_ce_basis) != str(args.r_ce_basis):
            _configure_r_ce_estimator(model, args, target_paths=target_paths)
    adjoint_network = _configure_adjoint_network(
        model,
        args,
        target_paths=target_paths,
    )
    if args.resume_checkpoint is not None:
        resume_checkpoint = args.resume_checkpoint.resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(resume_checkpoint)
        checkpoint = torch.load(
            resume_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        model.load_checkpoint_state(checkpoint)
        source_provenance = {
            "path": str(resume_checkpoint),
            "sha256": _file_sha256(resume_checkpoint),
            "load_mode": "matching_network_topology_resume",
        }
    training = _training_config(args)
    discrepancy_normalization: dict[str, Any] | None = None
    if str(args.objective_protocol) == "real_scale_corrected_v2_adjoint_rms":
        if not hasattr(discrepancy, "blocks"):
            raise ValueError("adjoint-RMS normalization requires a composite discrepancy")
        discrepancy, discrepancy_normalization = (
            fit_train_only_adjoint_rms_normalized_discrepancy(
                model=model,
                discrepancy=discrepancy,
                target_paths=target_paths,
                training=training,
                calibration_batches=4,
                calibration_batch_size=256,
                calibration_seed=811_903,
                total_target_rms=1.0,
            )
        )
        model.discrepancy = discrepancy
        print(
            "fitted train-only adjoint-RMS discrepancy normalization "
            f"active_blocks={discrepancy_normalization['active_block_count']} "
            f"per_block_target={discrepancy_normalization['per_block_target_rms']:.6g}",
            flush=True,
        )

    if str(args.objective_protocol) == "real_scale_corrected_v3_family_adjoint_rms":
        if not hasattr(discrepancy, "blocks"):
            raise ValueError("family adjoint-RMS normalization requires a composite discrepancy")
        frozen_manifest_path = args.frozen_discrepancy_normalization_manifest
        if frozen_manifest_path is not None:
            frozen_manifest = json.loads(
                frozen_manifest_path.read_text(encoding="utf-8")
            )
            frozen_normalization = frozen_manifest.get("discrepancy_normalization")
            if not isinstance(frozen_normalization, dict):
                raise ValueError(
                    "frozen normalization manifest is missing discrepancy_normalization"
                )
            frozen_coefficients = frozen_normalization.get("family_coefficients")
            expected_coefficients = {
                "price_law": float(args.lambda_x),
                "volatility_law": float(args.lambda_v),
            }
            if frozen_coefficients != expected_coefficients:
                raise ValueError(
                    "frozen normalization family coefficients do not match lambda-X/lambda-V"
                )
            discrepancy, discrepancy_normalization = (
                restore_family_adjoint_rms_normalized_discrepancy(
                    discrepancy=discrepancy,
                    normalization=frozen_normalization,
                )
            )
            discrepancy_normalization["restored_manifest"] = str(
                frozen_manifest_path.resolve()
            )
        else:
            price_blocks = {"path_law_observed_path", "path_law_returns"}
            family_by_block = {
                str(block.name): (
                    "price_law" if str(block.name) in price_blocks else "volatility_law"
                )
                for block in discrepancy.blocks
            }
            discrepancy, discrepancy_normalization = (
                fit_train_only_family_adjoint_rms_normalized_discrepancy(
                    model=model,
                    discrepancy=discrepancy,
                    target_paths=target_paths,
                    training=training,
                    family_by_block=family_by_block,
                    family_coefficients={
                        "price_law": float(args.lambda_x),
                        "volatility_law": float(args.lambda_v),
                    },
                    calibration_batches=4,
                    calibration_batch_size=256,
                    calibration_seed=811_903,
                    family_target_rms=1.0,
                )
            )
        model.discrepancy = discrepancy
        normalization_action = (
            "restored frozen"
            if bool(discrepancy_normalization.get("restored_from_frozen_manifest"))
            else "fitted train-only"
        )
        print(
            f"{normalization_action} family adjoint-RMS normalization "
            f"lambda_x={args.lambda_x:.6g} lambda_v={args.lambda_v:.6g}",
            flush=True,
        )

    # This multiplier is intentionally applied *after* any train-only/frozen
    # normalization.  It therefore changes exactly the relative coefficient of
    # the conditional prefix/future volatility functional, rather than silently
    # refitting every discrepancy block around the requested intervention.
    if float(args.joint_block_multiplier) != 1.0:
        if not isinstance(discrepancy, CompositePathFunctionalDiscrepancy):
            raise ValueError("joint block scaling requires a composite discrepancy")
        joint_name = "volatility_law_joint_prefix_future_rv"
        discrepancy = _scale_named_discrepancy_block(
            discrepancy,
            name=joint_name,
            multiplier=float(args.joint_block_multiplier),
        )
        model.discrepancy = discrepancy
        print(
            "scaled prefix/future RV discrepancy after frozen normalization "
            f"multiplier={float(args.joint_block_multiplier):.6g}",
            flush=True,
        )

    validation_log = normalized_prices_to_log_paths(fit.validation_prices)

    def progress(_model, metrics: dict[str, float]) -> None:
        if str(args.solver) in {"online", "direct_bptt"}:
            message = (
                "phase={phase} solver={solver} seed={seed} step={step:.0f} "
                "loss={loss:.6g} sigma={sigma:.6g} grad={grad:.6g} "
                "clip={clip:.4g}"
            ).format(
                phase=args.phase,
                solver=args.solver,
                seed=args.seed,
                step=float(metrics["step"]),
                loss=float(metrics["train_total_loss"]),
                sigma=float(metrics.get("sigma_mean", math.nan)),
                grad=float(metrics.get("grad_norm", math.nan)),
                clip=float(metrics.get("grad_clip_scale", math.nan)),
            )
        else:
            message = (
                "phase={phase} solver=nested seed={seed} outer={outer:.0f} "
                "step={step:.0f} fixed_loss={loss:.6g} objective={objective:.6g} "
                "accepted={accepted:.0f} rates=({p:.4g},{r:.4g},{shared:.4g}) "
                "sigma={sigma:.6g} grad_max={grad:.6g}"
            ).format(
                phase=args.phase,
                seed=args.seed,
                outer=float(metrics["outer_step"]),
                step=float(metrics["step"]),
                loss=float(metrics["accepted_fixed_total_loss"]),
                objective=float(metrics["outer_accepted_objective"]),
                accepted=float(metrics["outer_objective_accepted"]),
                p=float(metrics["outer_relaxation_p"]),
                r=float(metrics["outer_relaxation_r"]),
                shared=float(metrics["outer_relaxation_shared"]),
                sigma=float(metrics.get("sigma_mean", math.nan)),
                grad=float(metrics.get("inner_grad_norm_max", math.nan)),
            )
        print(message, flush=True)
        if (
            str(args.solver) == "online"
            and int(metrics["step"]) in online_checkpoint_steps
        ):
            step = int(metrics["step"])
            checkpoint_path = run_dir / f"step_{step:04d}_checkpoint.pt"
            torch.save(_model.checkpoint_state(), checkpoint_path)
            checkpoint_bank = _sample_price_bank(
                _model,
                transform=transform,
                bank_size=int(args.bank_size),
                batch_size=int(args.sample_batch_size),
                seed=int(args.seed) + 70_000,
                device=device,
                dtype=dtype,
                state_unit_scale=state_unit_scale,
            )
            checkpoint_bank_path = run_dir / f"step_{step:04d}_validation_bank.npy"
            np.save(checkpoint_bank_path, checkpoint_bank)
            checkpoint_generated_log = normalized_prices_to_log_paths(
                checkpoint_bank
            )
            checkpoint_law = real_data_law_metrics(
                checkpoint_generated_log,
                validation_log,
            )
            checkpoint_shadow = evaluate_path_shadowing(
                real_paths=validation_log,
                generated_paths=checkpoint_generated_log,
                prefix_length=65,
                future_horizon=32,
                top_k=256,
                feature_config=ShadowingFeatureConfig(),
                search_backend="auto",
            )
            write_json(
                run_dir / f"step_{step:04d}_validation.json",
                {
                    "selection_scope": "validation_only",
                    "common_bank_seed": int(args.seed) + 70_000,
                    "step": step,
                    "law_metrics": checkpoint_law,
                    "shadowing_metrics": checkpoint_shadow.metrics,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "bank": str(checkpoint_bank_path.resolve()),
                },
            )
            print(
                "online checkpoint validation "
                f"step={step} "
                f"rv_crps={checkpoint_shadow.metrics['realized_volatility_crps']:.6g} "
                f"rv_cov90={checkpoint_shadow.metrics['realized_volatility_coverage_90']:.3f}",
                flush=True,
            )
        if str(args.solver) == "nested" and bool(args.screen_each_outer):
            outer = int(metrics["outer_step"])
            checkpoint_path = run_dir / f"outer_{outer:03d}_checkpoint.pt"
            torch.save(_model.checkpoint_state(), checkpoint_path)
            outer_bank = _sample_price_bank(
                _model,
                transform=transform,
                bank_size=int(args.bank_size),
                batch_size=int(args.sample_batch_size),
                seed=int(args.seed) + 70_000,
                device=device,
                dtype=dtype,
                state_unit_scale=state_unit_scale,
            )
            outer_bank_path = run_dir / f"outer_{outer:03d}_validation_bank.npy"
            np.save(outer_bank_path, outer_bank)
            outer_generated_log = normalized_prices_to_log_paths(outer_bank)
            outer_law = real_data_law_metrics(
                outer_generated_log,
                validation_log,
            )
            outer_shadow = evaluate_path_shadowing(
                real_paths=validation_log,
                generated_paths=outer_generated_log,
                prefix_length=65,
                future_horizon=32,
                top_k=256,
                feature_config=ShadowingFeatureConfig(),
                search_backend="auto",
            )
            write_json(
                run_dir / f"outer_{outer:03d}_validation.json",
                {
                    "selection_scope": "validation_only",
                    "common_bank_seed": int(args.seed) + 70_000,
                    "outer_step": outer,
                    "law_metrics": outer_law,
                    "shadowing_metrics": outer_shadow.metrics,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "bank": str(outer_bank_path.resolve()),
                },
            )
            print(
                "outer validation "
                f"outer={outer} "
                f"rv_crps={outer_shadow.metrics['realized_volatility_crps']:.6g} "
                f"rv_cov90={outer_shadow.metrics['realized_volatility_coverage_90']:.3f} "
                f"rv_width90={outer_shadow.metrics['realized_volatility_band_width_90']:.6g}",
                flush=True,
            )

    nested_config = None
    if str(args.solver) == "online":
        result = model.fit(
            target_paths,
            training=training,
            log_callback=progress,
        )
    elif str(args.solver) == "direct_bptt":
        result = _fit_direct_objective_diagnostic(
            model,
            target_paths,
            training=training,
            log_callback=progress,
        )
    else:
        nested_config = DiscreteMPNestedTrainingConfig(
            outer_steps=int(args.nested_outer_steps),
            inner_steps=int(args.nested_inner_steps),
            outer_batch_size=int(args.nested_outer_batch_size),
            outer_target_batch_size=int(args.nested_outer_target_batch_size),
            outer_backward_replicates=int(
                args.nested_outer_backward_replicates
            ),
            outer_target_estimator=str(args.nested_outer_target_estimator),
            outer_conditional_branches=int(args.nested_conditional_branches),
            outer_conditional_antithetic=bool(
                args.nested_conditional_antithetic
            ),
            outer_conditional_query_batch_size=int(
                args.nested_conditional_query_batch_size
            ),
            outer_population_batch_size=args.nested_population_batch_size,
            outer_line_search_batch_size=args.nested_line_search_batch_size,
            inner_batch_size=int(args.nested_inner_batch_size),
            fixed_point_probe_paths=int(args.nested_probe_paths),
            log_every_outer=1,
            outer_objective_line_search=True,
            outer_relaxation_mode=str(args.nested_relaxation_mode),
            outer_max_backtracks=int(args.nested_max_backtracks),
            outer_backtrack_factor=float(args.nested_backtrack_factor),
            outer_objective_tolerance=float(args.nested_objective_tolerance),
            outer_block_coordinate_passes=int(
                args.nested_block_coordinate_passes
            ),
            outer_block_trust_fraction=float(args.nested_block_trust_fraction),
            inner_solver=str(args.nested_inner_solver),
            seed=int(args.seed) + 10_000,
        )
        result = model.fit_nested(
            target_paths,
            training=training,
            nested=nested_config,
            log_callback=progress,
        )
    checkpoint_name = (
        "source_model_checkpoint.pt"
        if args.phase == "source"
        else "model_checkpoint.pt"
    )
    checkpoint_path = run_dir / checkpoint_name
    torch.save(model.checkpoint_state(), checkpoint_path)
    write_history_jsonl(run_dir / "history.jsonl", result.history)
    _write_scaler(run_dir / "preprocessing.json", transform)
    bank = _sample_price_bank(
        model,
        transform=transform,
        bank_size=int(args.bank_size),
        batch_size=int(args.sample_batch_size),
        seed=int(args.seed) + 70_000,
        device=device,
        dtype=dtype,
        state_unit_scale=state_unit_scale,
    )
    bank_path = run_dir / "validation_screen_bank.npy"
    np.save(bank_path, bank)
    generated_log = normalized_prices_to_log_paths(bank)
    law = real_data_law_metrics(generated_log, validation_log)
    shadow = evaluate_path_shadowing(
        real_paths=validation_log,
        generated_paths=generated_log,
        prefix_length=65,
        future_horizon=32,
        top_k=256,
        feature_config=ShadowingFeatureConfig(),
        search_backend="auto",
    )
    manifest = {
        "selection_scope": "training_and_validation_only",
        "test_or_event_data_seen": False,
        "phase": str(args.phase),
        "objective_protocol": str(args.objective_protocol),
        "seed": int(args.seed),
        "protocol": fit.scenario_bank_protocol,
        "index": fit.index,
        "train_sessions": int(fit.train_prices.shape[0]),
        "validation_sessions": int(fit.validation_prices.shape[0]),
        "train_first_date": str(fit.train_dates[0]),
        "train_last_date": str(fit.train_dates[-1]),
        "train_array_sha256": fit.train_sha256,
        "source_checkpoint": source_provenance,
        "architecture": asdict(architecture),
        "training": asdict(training),
        "solver": str(args.solver),
        "nested_training": (
            None if nested_config is None else asdict(nested_config)
        ),
        "eta": float(args.eta),
        "running_cost": str(args.running_cost),
        "use_reference_drift": bool(args.use_reference_drift),
        "drift_control_mode": str(args.drift_control_mode),
        "state_unit_scaling": str(args.state_unit_scaling),
        "state_unit_scale": float(state_unit_scale),
        "drift_penalty": float(component_args.drift_penalty),
        "lambda_x": float(args.lambda_x),
        "lambda_v": float(args.lambda_v),
        "joint_weight": float(args.joint_weight),
        "return_discrepancy": return_summary,
        "global_rv_discrepancy": global_rv_summary,
        "acf_discrepancy": acf_summary,
        "joint_block_multiplier": float(args.joint_block_multiplier),
        "conditional_claim": conditional_claim_summary,
        "discrepancy_stack": str(args.discrepancy_stack),
        "discrepancy_acf_lags": [
            int(value) for value in component_args.discrepancy_acf_lags
        ],
        "retained_discrepancy_blocks": retained_discrepancy_blocks,
        "screen_each_outer": bool(args.screen_each_outer),
        "online_checkpoint_steps": list(online_checkpoint_steps),
        "r_ce_basis": str(args.r_ce_basis),
        "source_checkpoint_r_ce_basis": str(args.source_checkpoint_r_ce_basis),
        "p_ce_basis": str(args.p_ce_basis),
        "r_ce_normalization": str(args.r_ce_normalization),
        "r_ce_time_mode": str(args.r_ce_time_mode),
        "adjoint_network": adjoint_network,
        "ce_crossfit_folds": int(args.ce_crossfit_folds),
        "ce_target_mode": str(args.ce_target_mode),
        "noise_target_control_variate": str(args.noise_target_control_variate),
        "noise_target_estimator": str(args.noise_target_estimator),
        "stein_probes": int(args.stein_probes),
        "discrepancy_normalization": discrepancy_normalization,
        "reference_kind": str(args.reference_kind),
        "guyon_realized_volatility_window": int(
            args.guyon_realized_volatility_window
        ),
        "reference_feature_basis": str(args.reference_feature_basis),
        "reference_causal_weight": (
            None
            if args.reference_causal_weight is None
            else float(args.reference_causal_weight)
        ),
        "reference_summary": control.reference_kernel.summary(),
        "reference_sigma_std_floor": reference_sigma_std_floor,
        "reference_rv_spread_ratio_min": float(
            args.reference_rv_spread_ratio_min
        ),
        "admissible_sigma_max": admissible_sigma_max,
        "admissible_sigma_max_rule": {
            "formula": (
                "max(0.01, train |return|/sqrt(dt) quantile, "
                "multiple * train return std/sqrt(dt))"
            ),
            "tail_quantile": float(args.sigma_cap_quantile),
            "std_multiple": float(args.sigma_cap_std_multiple),
        },
        "preprocessing": preprocessing,
        "fit_final": result.final_metrics,
        "stability": _history_summary(result.history),
        "bank_diagnostics": _bank_diagnostics(bank, fit.validation_prices),
        "resources": {
            "wall_seconds": float(time.perf_counter() - wall_started),
            "peak_gpu_allocated_gb": (
                float(torch.cuda.max_memory_allocated()) / 2**30
                if device.type == "cuda"
                else 0.0
            ),
        },
        "validation_law_metrics": law,
        "validation_shadowing_metrics": shadow.metrics,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "screen_bank": str(bank_path.resolve()),
    }
    write_json(manifest_path, manifest)
    print(
        "complete "
        f"phase={args.phase} seed={args.seed} "
        f"rv_crps={shadow.metrics['realized_volatility_crps']:.6g} "
        f"rv_cov90={shadow.metrics['realized_volatility_coverage_90']:.3f} "
        f"clip_fraction={manifest['stability']['clip_active_fraction']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
