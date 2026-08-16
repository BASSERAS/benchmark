#!/usr/bin/env python3
"""Validation-only nested-MP control screen and final-method runner.

This runner is deliberately limited to the two synthetic paper tasks.  It
starts both arms from the same fitted physical-drift reference and changes
only whether the P-induced drift correction is admissible.  Single-arm modes
also support matched zero-drift and fixed fitted-reference-drift ablations in
which MP controls volatility alone.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    calibrate_shrunk_causal_reference_kernel,
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteMPNestedTrainingConfig,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
    fit_causal_volatility_reference_kernel,
    fit_guyon_lekeufack_likelihood_reference_kernel,
    fit_guyon_lekeufack_price_realized_volatility_reference_kernel,
    fit_local_gaussian_reference_kernel,
)
from deep_mkv_gen_path_dt.ce import RidgeConditionalExpectation  # noqa: E402
from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    ReferenceDriftSpecificEntropyDiagonalControl,
    VolatilityOnlySpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from path_dt_experiments.discrepancies import (  # noqa: E402
    build_heston_discrepancy,
    fit_fixed_target_joint_prefix_future_volatility_mmd,
)
from path_dt_experiments.causal_transformer_adjoint import (  # noqa: E402
    CachedCausalTransformerAdjointNetwork,
)
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.heston_mixture_discrepancy import (  # noqa: E402
    HestonMixtureCausalRidgeConditionalExpectation,
    fit_fixed_target_heston_mixture_summary_mmd,
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
from run_heston_mixture_deep_mkv_ablation import (  # noqa: E402
    _load_log_paths,
)
from summarize_volatility_only_nested_scaling import _path_metrics  # noqa: E402


ARMS = ("full_control_nested_mp", "volatility_only_nested_mp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("heston", "mixture"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    parser.add_argument("--lambda-scale", type=float, default=50.0)
    parser.add_argument("--kappa-scale", type=float, default=50.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument(
        "--path-derivative-backend",
        choices=("autograd", "analytical"),
        default="autograd",
        help=(
            "Compute discrepancy and running-cost path sources with PyTorch "
            "autograd or with the closed-form first-derivative backend."
        ),
    )
    parser.add_argument(
        "--drift-adjoint-backend",
        choices=("none", "autograd_replay", "analytical_reference"),
        default="none",
        help=(
            "Propagate path sources through the frozen path-dependent reference "
            "drift. The default preserves the historical cumulative-sum labels."
        ),
    )
    parser.add_argument(
        "--adjoint-weight",
        type=float,
        default=1.0,
        help="Weight of the Y/expected-adjoint regression loss.",
    )
    parser.add_argument(
        "--adjoint-noise-weight",
        type=float,
        default=1.0,
        help="Weight of the Z/noise-adjoint regression loss.",
    )
    parser.add_argument(
        "--adjoint-network",
        choices=("gru", "causal_transformer"),
        default="gru",
        help="Sequential network used to store the learned conditional adjoint signals.",
    )
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help=(
            "Optional optimizer gradient clipping norm. When omitted, the "
            "established default is 5 for Heston and no clipping for mixture."
        ),
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help=(
            "Specific-entropy running-cost coefficient. Larger values penalize "
            "volatility movement away from the fitted reference more strongly."
        ),
    )
    parser.add_argument(
        "--reference-kind",
        choices=(
            "local_gaussian",
            "calibrated_causal",
            "guyon_lekeufack_structural_likelihood",
            "guyon_lekeufack_price_realized_volatility",
        ),
        default="local_gaussian",
        help=(
            "Causal reference fitted on the locked training paths. The default "
            "preserves existing campaigns; calibrated_causal uses the pooled "
            "causal-feature volatility reference used by the long-horizon and "
            "real-data protocols; the structural Guyon option is available for "
            "matched Heston experiments."
        ),
    )
    parser.add_argument(
        "--guyon-realized-volatility-window",
        type=int,
        default=20,
        help=(
            "Forward return window used as the price-derived volatility label "
            "for empirical Guyon calibration."
        ),
    )
    parser.add_argument("--joint-volatility-weight", type=float, default=1.0)
    parser.add_argument(
        "--source-joint-volatility-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the joint prefix--future volatility block during the "
            "broad source stage. The historical source-only configuration is zero."
        ),
    )
    parser.add_argument(
        "--abs-return-acf-weight",
        type=float,
        default=None,
        help="Optional override of the absolute-return lag-product block weight.",
    )
    parser.add_argument(
        "--squared-return-acf-weight",
        type=float,
        default=None,
        help="Optional override of the squared-return lag-product block weight.",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Stop after the broad-discrepancy source stage and evaluate it.",
    )
    parser.add_argument("--source-steps", type=int, default=2000)
    parser.add_argument(
        "--source-checkpoint-steps",
        type=int,
        nargs="*",
        default=(),
        help=(
            "For an online source-only run, save these intermediate steps during "
            "one continuous optimizer trajectory and evaluate every saved checkpoint "
            "on the common validation bank after training."
        ),
    )
    parser.add_argument("--joint-steps", type=int, default=1600)
    parser.add_argument(
        "--solver",
        choices=("nested", "online"),
        default="nested",
        help="MP target-refresh/training procedure; the control problem is unchanged.",
    )
    drift_group = parser.add_mutually_exclusive_group()
    drift_group.add_argument(
        "--final-zero-drift-only",
        action="store_true",
        help=(
            "Run only the locked final nested volatility-only method with an "
            "identically zero physical drift."
        ),
    )
    drift_group.add_argument(
        "--fitted-reference-drift-only",
        action="store_true",
        help=(
            "Run only nested volatility-only MP while retaining the fitted "
            "reference drift as a fixed physical drift."
        ),
    )
    return parser.parse_args()


def _benchmark_heston_root() -> Path:
    """Locate <benchmark>/dataset/Heston by walking up from this vendored file."""
    override = os.environ.get("BENCHMARK_HESTON_DIR")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "dataset" / "Heston"
        if (candidate / "heston_S_8192x128.npy").is_file():
            return candidate
    raise FileNotFoundError(
        "could not locate <benchmark>/dataset/Heston; set BENCHMARK_HESTON_DIR"
    )


def _task_paths(task: str) -> tuple[Path, Path, Path | None]:
    if task == "heston":
        root = _benchmark_heston_root()
        # Hyperparameter search scores on the val splits, never on `disc`
        # (GUIDELINE 11, selection-on-eval). The locked reproduction leaves
        # BENCHMARK_HESTON_EVAL unset and therefore scores on `disc`.
        eval_name = os.environ.get("BENCHMARK_HESTON_EVAL", "heston_S_disc_8192x128.npy")
        return (
            root / "heston_S_8192x128.npy",
            root / eval_name,
            None,
        )
    root = REPO_ROOT / "runs" / "heston_parameter_mixture_seed0_20260730"
    return root / "data" / "train.npy", root / "data" / "disc.npy", root


def _nested(*, outer_steps: int, seed: int) -> DiscreteMPNestedTrainingConfig:
    return DiscreteMPNestedTrainingConfig(
        outer_steps=int(outer_steps),
        inner_steps=200,
        outer_batch_size=1024,
        outer_target_batch_size=1024,
        outer_target_estimator="projected_score",
        inner_batch_size=256,
        fixed_point_probe_paths=512,
        log_every_outer=1,
        outer_objective_line_search=True,
        outer_relaxation_mode="adjoint_blocks",
        outer_max_backtracks=10,
        outer_backtrack_factor=0.5,
        outer_objective_tolerance=0.0,
        outer_block_coordinate_passes=2,
        outer_block_trust_fraction=1.0,
        seed=int(seed),
    )


def _training(
    *,
    task: str,
    steps: int,
    seed: int,
    lambda_scale: float,
    lr: float,
    grad_clip_norm: float | None,
    adjoint_weight: float,
    adjoint_noise_weight: float,
    path_derivative_backend: str,
    drift_adjoint_backend: str,
) -> DiscreteMPTrainingConfig:
    return DiscreteMPTrainingConfig(
        batch_size=256,
        target_batch_size=256,
        num_steps=int(steps),
        lr=float(lr),
        weight_decay=1e-5,
        lambda_scale=float(lambda_scale),
        adjoint_weight=float(adjoint_weight),
        adjoint_noise_weight=float(adjoint_noise_weight),
        ce_target_mode="ridge",
        ridge_lambda=1e-3,
        ce_crossfit_folds=1,
        target_preconditioner="none",
        noise_target_control_variate="none",
        grad_clip_norm=(
            float(grad_clip_norm)
            if grad_clip_norm is not None
            else (5.0 if task == "heston" else None)
        ),
        log_every=100,
        seed=int(seed),
        observed_only=False,
        path_derivative_backend=str(path_derivative_backend),
        drift_adjoint_backend=str(drift_adjoint_backend),
    )


def _model(
    *,
    grid: DiscreteTimeGrid,
    control,
    discrepancy,
    estimator,
    device: torch.device,
    seed: int,
    adjoint_network: str,
    transformer_layers: int,
    transformer_heads: int,
) -> DiscreteMPModel:
    architecture = DiscreteMPArchitectureConfig(
        state_dim=1, noise_dim=1, hidden_dim=96, num_layers=1
    )
    network = None
    init_seed: int | None = int(seed)
    if str(adjoint_network) == "causal_transformer":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            network = CachedCausalTransformerAdjointNetwork(
                state_dim=1,
                adjoint_dim=1,
                noise_adjoint_dim=1,
                hidden_dim=96,
                num_layers=int(transformer_layers),
                num_heads=int(transformer_heads),
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
        ce_estimator=estimator,
        network=network,
        init_seed=init_seed,
    )
    model.network.to(device=device, dtype=torch.float32)
    return model


def _progress(arm: str, phase: str):
    def report(_model: DiscreteMPModel, metrics: dict[str, float]) -> None:
        print(
            f"{arm}/{phase} outer={metrics.get('outer_step', 0):.0f} "
            f"accepted={metrics.get('outer_objective_accepted', 0):.0f} "
            f"objective={metrics.get('outer_accepted_objective', float('nan')):.6g}",
            flush=True,
        )

    return report


def _online_progress(arm: str, phase: str):
    def report(_model: DiscreteMPModel, metrics: dict[str, float]) -> None:
        print(
            f"{arm}/{phase} step={metrics.get('step', 0):.0f} "
            f"loss={metrics.get('train_total_loss', float('nan')):.6g} "
            f"objective={metrics.get('complete_objective_value', float('nan')):.6g}",
            flush=True,
        )

    return report


def _online_progress_with_checkpoints(
    arm: str,
    phase: str,
    *,
    checkpoint_dir: Path,
    checkpoint_steps: tuple[int, ...],
):
    progress = _online_progress(arm, phase)
    selected = set(int(step) for step in checkpoint_steps)

    def report(model: DiscreteMPModel, metrics: dict[str, float]) -> None:
        progress(model, metrics)
        step = int(metrics.get("step", 0.0))
        if step in selected:
            torch.save(
                model.checkpoint_state(),
                checkpoint_dir / f"step_{step:04d}.pt",
            )

    return report


def _sample(
    model: DiscreteMPModel,
    *,
    num_paths: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    prices = np.empty((num_paths, model.grid.num_steps + 1), dtype=np.float32)
    sigma_values: list[np.ndarray] = []
    drift_errors: list[np.ndarray] = []
    correction_values: list[np.ndarray] = []
    denominator_values: list[np.ndarray] = []
    x0 = torch.zeros((1, 1), device=model.device, dtype=model.dtype)
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, num_paths, batch_size)):
            count = min(batch_size, num_paths - start)
            result = model.sample(
                num_paths=count,
                x0=x0,
                seed=derive_stream_seed(
                    base_seed=int(seed), stream_offset=10_000 + batch_index
                ),
            )
            log_paths = result.paths[..., 0]
            controls = result.controls
            reference_drift_fn = getattr(
                model.control_map, "_reference_drifts_all", None
            )
            reference_drift = (
                reference_drift_fn(result.paths)
                if callable(reference_drift_fn)
                else torch.zeros_like(controls[..., :1])
            )
            reference_sigma = model.control_map._reference_sigmas_all(result.paths)
            prices[start : start + count] = (
                100.0 * torch.exp(log_paths)
            ).cpu().numpy().astype(np.float32)
            sigma_values.append(controls[..., 1:].cpu().numpy())
            correction_values.append(
                (controls[..., :1] - reference_drift).cpu().numpy()
            )
            drift_errors.append(
                torch.abs(controls[..., :1] - reference_drift).cpu().numpy()
            )
            denominator_values.append(
                (
                    float(model.control_map.eta) / reference_sigma
                    + result.expected_adjoint_noise_next
                    / math.sqrt(float(model.grid.dt))
                ).cpu().numpy()
            )
    sigma = np.concatenate(sigma_values).reshape(-1)
    correction = np.concatenate(correction_values).reshape(-1)
    drift_error = np.concatenate(drift_errors).reshape(-1)
    denominator = np.concatenate(denominator_values).reshape(-1)
    sigma_min = float(model.control_map.sigma_min)
    sigma_max = float(model.control_map.sigma_max)
    diagnostics = {
        "sigma_mean": float(sigma.mean()),
        "sigma_std": float(sigma.std()),
        "sigma_q99": float(np.quantile(sigma, 0.99)),
        "sigma_min_active_fraction": float(np.mean(sigma <= sigma_min * (1.0 + 1e-5))),
        "sigma_max_active_fraction": float(np.mean(sigma >= sigma_max * (1.0 - 1e-5))),
        "drift_correction_rms": float(np.sqrt(np.mean(correction**2))),
        "reference_drift_max_abs_error": float(drift_error.max()),
        "specific_entropy_denominator_min": float(denominator.min()),
        "specific_entropy_denominator_q001": float(
            np.quantile(denominator, 0.001)
        ),
        "specific_entropy_denominator_q01": float(
            np.quantile(denominator, 0.01)
        ),
        "specific_entropy_denominator_q05": float(
            np.quantile(denominator, 0.05)
        ),
        "specific_entropy_denominator_nonpositive_fraction": float(
            np.mean(denominator <= 0.0)
        ),
        "nonfinite_control_fraction": float(
            1.0 - np.mean(np.isfinite(np.concatenate((sigma, correction))))
        ),
        "nonfinite_path_fraction": float(1.0 - np.mean(np.isfinite(prices))),
    }
    return prices, diagnostics


def _history_summary(history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {"outer_steps": 0, "accepted_outer_steps": 0}
    accepted = [row for row in history if float(row.get("outer_objective_accepted", 0.0)) > 0.5]
    before = float(history[0].get("outer_objective_before", float("nan")))
    after = float(history[-1].get("outer_accepted_objective", float("nan")))
    return {
        "outer_steps": len(history),
        "accepted_outer_steps": len(accepted),
        "objective_before": before,
        "objective_after": after,
        "objective_improvement_fraction": (
            (before - after) / before if math.isfinite(before) and before != 0.0 else float("nan")
        ),
    }


def _conditional_metrics(prices: np.ndarray) -> dict[str, float]:
    log_paths = np.log(prices / prices[:, :1])
    returns = np.diff(log_paths, axis=1)
    prefix = np.sqrt(np.sum(returns[:, :64] ** 2, axis=1))
    future = np.sqrt(np.sum(returns[:, 64:96] ** 2, axis=1))
    threshold = float(np.median(prefix))
    correlation = float(np.corrcoef(prefix, future)[0, 1])
    gap = float(future[prefix >= threshold].mean() - future[prefix < threshold].mean())
    return {
        "prefix_future_rv_correlation": correlation,
        "prefix_high_low_future_rv_gap": gap,
    }


def _full_bank_scale_tail_metrics(
    target_prices: np.ndarray,
    generated_prices: np.ndarray,
) -> dict[str, float]:
    target_returns = np.diff(np.log(target_prices / target_prices[:, :1]), axis=1)
    generated_returns = np.diff(
        np.log(generated_prices / generated_prices[:, :1]), axis=1
    )
    target_rv = np.sqrt(np.sum(target_returns**2, axis=1))
    generated_rv = np.sqrt(np.sum(generated_returns**2, axis=1))

    def kurtosis(values: np.ndarray) -> float:
        centered = values - values.mean()
        variance = np.mean(centered**2)
        return float(np.mean(centered**4) / max(float(variance**2), 1e-30) - 3.0)

    return {
        "return_std_ratio": float(generated_returns.std() / max(target_returns.std(), 1e-15)),
        "realized_volatility_mean_ratio": float(generated_rv.mean() / max(target_rv.mean(), 1e-15)),
        "realized_volatility_std_ratio": float(generated_rv.std() / max(target_rv.std(), 1e-15)),
        "realized_volatility_q90_ratio": float(np.quantile(generated_rv, 0.9) / max(float(np.quantile(target_rv, 0.9)), 1e-15)),
        "absolute_return_q99_ratio": float(np.quantile(np.abs(generated_returns), 0.99) / max(float(np.quantile(np.abs(target_returns), 0.99)), 1e-15)),
        "absolute_return_q999_ratio": float(np.quantile(np.abs(generated_returns), 0.999) / max(float(np.quantile(np.abs(target_returns), 0.999)), 1e-15)),
        "generated_excess_kurtosis": kurtosis(generated_returns),
        "target_excess_kurtosis": kurtosis(target_returns),
        "nonfinite_path_fraction": float(1.0 - np.mean(np.isfinite(generated_prices))),
    }


def _evaluate_common(
    *, target_prices: np.ndarray, generated: np.ndarray, device: torch.device, seed: int
) -> dict[str, object]:
    path = _path_metrics(
        target_prices,
        generated,
        device=device,
        swd_projections=256,
        swd_seed=20260804 + int(seed),
    )
    target_log = torch.as_tensor(
        np.log(target_prices / target_prices[:, :1])[..., None], dtype=torch.float32
    )
    generated_log = torch.as_tensor(
        np.log(generated / generated[:, :1])[..., None], dtype=torch.float32
    )
    shadow = evaluate_path_shadowing(
        real_paths=target_log,
        generated_paths=generated_log,
        prefix_length=65,
        future_horizon=32,
        top_k=256,
        feature_config=ShadowingFeatureConfig(),
        search_backend="auto",
    )
    return {
        "path_law": path,
        "full_bank_scale_tail": _full_bank_scale_tail_metrics(
            target_prices, generated
        ),
        "conditional_path_memory": {
            "target": _conditional_metrics(target_prices),
            "generated": _conditional_metrics(generated),
        },
        "path_shadowing": shadow.metrics,
    }


def _write_mixture_metrics(root: Path, bank: Path, output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "scripts" / "evaluate_heston_parameter_mixture.py"),
            "--train-data", str(root / "data" / "train.npy"),
            "--test-data", str(root / "data" / "disc.npy"),
            "--generated-data", str(bank),
            "--oracle", str(root / "oracle" / "oracle.joblib"),
            "--oracle-gate-report", str(root / "oracle" / "gate_report.json"),
            "--output", str(output),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    if (
        str(args.solver) == "online"
        and not bool(args.final_zero_drift_only)
        and not bool(args.fitted_reference_drift_only)
    ):
        raise ValueError(
            "the matched online arm requires a volatility-only drift mode"
        )
    if not math.isfinite(float(args.lambda_scale)) or float(args.lambda_scale) <= 0.0:
        raise ValueError("lambda-scale must be positive and finite")
    if not math.isfinite(float(args.kappa_scale)) or float(args.kappa_scale) < 0.0:
        raise ValueError("kappa-scale must be nonnegative and finite")
    if not math.isfinite(float(args.eta)) or float(args.eta) <= 0.0:
        raise ValueError("eta must be positive and finite")
    if not math.isfinite(float(args.lr)) or float(args.lr) <= 0.0:
        raise ValueError("lr must be positive and finite")
    if args.grad_clip_norm is not None and (
        not math.isfinite(float(args.grad_clip_norm))
        or float(args.grad_clip_norm) <= 0.0
    ):
        raise ValueError("grad-clip-norm must be positive and finite when supplied")
    if (
        not math.isfinite(float(args.joint_volatility_weight))
        or float(args.joint_volatility_weight) < 0.0
    ):
        raise ValueError("joint-volatility-weight must be nonnegative and finite")
    if (
        not math.isfinite(float(args.source_joint_volatility_weight))
        or float(args.source_joint_volatility_weight) < 0.0
    ):
        raise ValueError(
            "source-joint-volatility-weight must be nonnegative and finite"
        )
    for name, value in (
        ("abs-return-acf-weight", args.abs_return_acf_weight),
        ("squared-return-acf-weight", args.squared_return_acf_weight),
    ):
        if value is not None and (
            not math.isfinite(float(value)) or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be nonnegative and finite when supplied")
    if int(args.source_steps) < 1:
        raise ValueError("source-steps must be positive")
    if int(args.joint_steps) < 1:
        raise ValueError("joint-steps must be positive")
    if int(args.transformer_layers) < 1:
        raise ValueError("transformer-layers must be positive")
    if int(args.transformer_heads) < 1 or 96 % int(args.transformer_heads) != 0:
        raise ValueError("transformer-heads must be a positive divisor of 96")
    source_checkpoint_steps = tuple(int(step) for step in args.source_checkpoint_steps)
    if len(set(source_checkpoint_steps)) != len(source_checkpoint_steps):
        raise ValueError("source-checkpoint-steps must be unique")
    if source_checkpoint_steps != tuple(sorted(source_checkpoint_steps)):
        raise ValueError("source-checkpoint-steps must be strictly increasing")
    if any(step < 1 or step > int(args.source_steps) for step in source_checkpoint_steps):
        raise ValueError("source-checkpoint-steps must lie within the source training budget")
    if any(
        step != int(args.source_steps) and step % 100 != 0
        for step in source_checkpoint_steps
    ):
        raise ValueError(
            "source-checkpoint-steps must align with the 100-step online logging interval"
        )
    if source_checkpoint_steps and (
        str(args.solver) != "online" or not bool(args.source_only)
    ):
        raise ValueError(
            "source-checkpoint-steps require an online source-only run"
        )
    protocol = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))
    if bool(protocol.get("test_split_access_authorized", True)):
        raise RuntimeError("locked protocol does not certify validation-only execution")
    if (
        not bool(args.final_zero_drift_only)
        and int(args.seed) != 0
        and not bool(protocol.get("multiseed_expansion_authorized", False))
    ):
        raise ValueError(
            "nonzero fitted-reference-drift seeds require a protocol with "
            "multiseed_expansion_authorized=true"
        )
    run_dir = ensure_run_dir(args.run_dir.resolve())
    device = torch.device(args.device)
    train_path, validation_path, mixture_root = _task_paths(args.task)
    target_paths = _load_log_paths(train_path, device=device, dtype=torch.float32)
    validation_prices = np.asarray(np.load(validation_path, allow_pickle=False), dtype=np.float64)
    num_steps = int(target_paths.shape[1]) - 1
    grid = DiscreteTimeGrid(T=num_steps * 0.004, num_steps=num_steps)
    reference_kwargs = dict(
        target_paths=target_paths,
        grid=grid,
        ridge=1e-3,
        variance_ridge=1e-3,
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    reference_calibration: dict[str, object] | None = None
    if str(args.reference_kind) == "guyon_lekeufack_structural_likelihood":
        reference = fit_guyon_lekeufack_likelihood_reference_kernel(
            **reference_kwargs,
            activity_update="structural_variance",
        )
    elif str(args.reference_kind) == "guyon_lekeufack_price_realized_volatility":
        reference = fit_guyon_lekeufack_price_realized_volatility_reference_kernel(
            **reference_kwargs,
            realized_volatility_window=int(args.guyon_realized_volatility_window),
        )
    elif str(args.reference_kind) == "calibrated_causal":
        local_reference = fit_local_gaussian_reference_kernel(**reference_kwargs)
        causal_reference = fit_causal_volatility_reference_kernel(
            target_paths=target_paths,
            grid=grid,
            rolling_windows=(5, 10, 20, 32),
            ewma_half_lives=(5, 20, 60),
            downside_windows=(10, 32),
            drift_ridge=1e-3,
            variance_ridges=(1e-5, 1e-4, 1e-3),
            crossfit_folds=3,
            crossfit_seed=1701,
            sigma_min=1e-3,
            sigma_max=0.6,
        )
        reference, reference_calibration = calibrate_shrunk_causal_reference_kernel(
            target_paths=target_paths,
            causal_reference=causal_reference,
            anchor_reference=local_reference,
            causal_weights=(0.0, 0.25, 0.4, 0.55, 0.7, 0.75, 0.8, 0.85, 1.0),
            calibration_paths=4096,
            calibration_seed=271828,
            offset_steps=3,
            return_std_tolerance=0.02,
            sigma_std_floor=0.045,
            sigma_cap_fraction_limit=0.01,
            clustering_lags=(1, 5, 20),
            sigma_min=1e-3,
            sigma_max=0.6,
        )
        selected = reference_calibration["selected"]
        print(
            "calibrated causal reference "
            f"weight={selected['causal_log_variance_weight']:.3f} "
            f"offset={selected['log_variance_offset']:.6f} "
            f"return_std={selected['return_std']:.6f} "
            f"sigma_std={selected['sigma_std']:.6f}",
            flush=True,
        )
    else:
        reference = fit_local_gaussian_reference_kernel(**reference_kwargs)
    base = build_heston_discrepancy(
        num_steps=num_steps,
        include_acf=True,
        preset="old_fullv_w0p25",
        lambda_scale=float(args.lambda_scale),
        kappa_scale=float(args.kappa_scale),
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
        abs_return_acf_weight=args.abs_return_acf_weight,
        squared_return_acf_weight=args.squared_return_acf_weight,
    )
    joint = fit_fixed_target_joint_prefix_future_volatility_mmd(
        target_paths,
        grid=grid,
        split_index=64,
        future_horizon=32,
        prefix_windows=(16, 32, 64),
        recent_return_window=16,
        bandwidths=(0.5, 1.0, 2.0),
    )
    source_blocks = list(base.blocks)
    if float(args.source_joint_volatility_weight) > 0.0:
        source_blocks.append(
            WeightedDiscrepancy(
                name="volatility_law_joint_prefix_future_rv",
                weight=float(args.source_joint_volatility_weight),
                discrepancy=joint,
            )
        )
    source_discrepancy = CompositePathFunctionalDiscrepancy(
        blocks=tuple(source_blocks)
    )
    blocks = list(base.blocks)
    blocks.append(
        WeightedDiscrepancy(
            name="volatility_law_joint_prefix_future_rv",
            weight=float(args.joint_volatility_weight),
            discrepancy=joint,
        )
    )
    if args.task == "mixture":
        blocks.append(
            WeightedDiscrepancy(
                name="mixture_law_joint_path_summaries",
                weight=1.0,
                discrepancy=fit_fixed_target_heston_mixture_summary_mmd(
                    target_paths, grid=grid
                ),
            )
        )
    complete = CompositePathFunctionalDiscrepancy(blocks=tuple(blocks))
    estimator = (
        HestonMixtureCausalRidgeConditionalExpectation(grid=grid, ridge=1e-3)
        if args.task == "mixture"
        else RidgeConditionalExpectation(ridge=1e-3)
    )

    # The exact reference is shared and generated once with zero adjoint heads.
    control_common = dict(
        dt=grid.dt,
        eta=float(args.eta),
        reference_kernel=reference,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    reference_control = (
        VolatilityOnlySpecificEntropyDiagonalControl(**control_common)
        if bool(args.final_zero_drift_only)
        else ReferenceDriftSpecificEntropyDiagonalControl(
            **control_common, allow_drift_correction=False
        )
    )
    reference_model = _model(
        grid=grid,
        control=reference_control,
        discrepancy=complete,
        estimator=estimator,
        device=device,
        seed=int(args.seed),
        adjoint_network=str(args.adjoint_network),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
    )
    with torch.no_grad():
        for parameter in reference_model.network.parameters():
            parameter.zero_()
    reference_dir = ensure_run_dir(run_dir / "reference")
    reference_bank, reference_controls = _sample(
        reference_model,
        num_paths=int(args.bank_size),
        batch_size=int(args.sample_batch_size),
        seed=70000 + int(args.seed),
    )
    np.save(reference_dir / "validation_bank.npy", reference_bank)
    write_json(reference_dir / "control_diagnostics.json", reference_controls)
    write_json(
        reference_dir / "metrics.json",
        _evaluate_common(
            target_prices=validation_prices,
            generated=reference_bank,
            device=device,
            seed=int(args.seed),
        ),
    )
    if args.task == "mixture" and mixture_root is not None:
        _write_mixture_metrics(
            mixture_root,
            reference_dir / "validation_bank.npy",
            reference_dir / "mixture_metrics.json",
        )
    del reference_model, reference_bank
    if device.type == "cuda":
        torch.cuda.empty_cache()

    selected_arms = (
        (
            "volatility_only_online_mp"
            if str(args.solver) == "online"
            else "volatility_only_nested_mp",
        )
        if bool(args.final_zero_drift_only)
        or bool(args.fitted_reference_drift_only)
        else ARMS
    )
    for arm in selected_arms:
        arm_dir = ensure_run_dir(run_dir / arm)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        control = (
            VolatilityOnlySpecificEntropyDiagonalControl(**control_common)
            if bool(args.final_zero_drift_only)
            else ReferenceDriftSpecificEntropyDiagonalControl(
                **control_common,
                allow_drift_correction=(
                    arm == "full_control_nested_mp"
                    and not bool(args.fitted_reference_drift_only)
                ),
            )
        )
        source = _model(
            grid=grid,
            control=control,
            discrepancy=source_discrepancy,
            estimator=estimator,
            device=device,
            seed=int(args.seed),
            adjoint_network=str(args.adjoint_network),
            transformer_layers=int(args.transformer_layers),
            transformer_heads=int(args.transformer_heads),
        )
        source_training = _training(
            task=args.task,
            steps=int(args.source_steps),
            seed=int(args.seed),
            lambda_scale=float(args.lambda_scale),
            lr=float(args.lr),
            grad_clip_norm=args.grad_clip_norm,
            adjoint_weight=float(args.adjoint_weight),
            adjoint_noise_weight=float(args.adjoint_noise_weight),
            path_derivative_backend=str(args.path_derivative_backend),
            drift_adjoint_backend=str(args.drift_adjoint_backend),
        )
        source_checkpoint_dir = arm_dir / "training_checkpoints"
        if source_checkpoint_steps:
            source_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if str(args.solver) == "nested":
            source_fit = source.fit_nested(
                target_paths,
                training=source_training,
                nested=_nested(outer_steps=10, seed=int(args.seed)),
                log_callback=_progress(arm, "source"),
            )
        else:
            source_fit = source.fit(
                target_paths,
                training=source_training,
                log_callback=(
                    _online_progress_with_checkpoints(
                        arm,
                        "source",
                        checkpoint_dir=source_checkpoint_dir,
                        checkpoint_steps=source_checkpoint_steps,
                    )
                    if source_checkpoint_steps
                    else _online_progress(arm, "source")
                ),
            )
        source_checkpoint = source.checkpoint_state()
        torch.save(source_checkpoint, arm_dir / "source_model_checkpoint.pt")
        write_history_jsonl(arm_dir / "source_history.jsonl", source_fit.history)
        if bool(args.source_only):
            checkpoint_evaluations: list[dict[str, object]] = []
            for checkpoint_step in source_checkpoint_steps:
                checkpoint_path = (
                    source_checkpoint_dir / f"step_{checkpoint_step:04d}.pt"
                )
                if not checkpoint_path.is_file():
                    raise RuntimeError(
                        f"missing requested source checkpoint {checkpoint_path}"
                    )
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                source.load_checkpoint_state(checkpoint)
                checkpoint_bank, checkpoint_controls = _sample(
                    source,
                    num_paths=int(args.bank_size),
                    batch_size=int(args.sample_batch_size),
                    seed=70000 + int(args.seed),
                )
                evaluation_dir = ensure_run_dir(
                    arm_dir / "checkpoint_evaluations" / f"step_{checkpoint_step:04d}"
                )
                checkpoint_bank_path = evaluation_dir / "validation_bank.npy"
                np.save(checkpoint_bank_path, checkpoint_bank)
                checkpoint_metrics = _evaluate_common(
                    target_prices=validation_prices,
                    generated=checkpoint_bank,
                    device=device,
                    seed=int(args.seed),
                )
                write_json(evaluation_dir / "metrics.json", checkpoint_metrics)
                write_json(
                    evaluation_dir / "control_diagnostics.json",
                    checkpoint_controls,
                )
                if args.task == "mixture" and mixture_root is not None:
                    _write_mixture_metrics(
                        mixture_root,
                        checkpoint_bank_path,
                        evaluation_dir / "mixture_metrics.json",
                    )
                checkpoint_evaluations.append(
                    {
                        "step": int(checkpoint_step),
                        "checkpoint": str(checkpoint_path),
                        "evaluation_dir": str(evaluation_dir),
                        "bank_seed": 70000 + int(args.seed),
                        "bank_size": int(args.bank_size),
                    }
                )
                del checkpoint_bank
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            source.load_checkpoint_state(source_checkpoint)
            if checkpoint_evaluations:
                write_json(
                    arm_dir / "checkpoint_trajectory_manifest.json",
                    {
                        "continuous_optimizer_trajectory": True,
                        "checkpoint_steps": list(source_checkpoint_steps),
                        "common_random_numbers": True,
                        "selection_scope": "validation_only",
                        "test_split_loaded": False,
                        "evaluations": checkpoint_evaluations,
                    },
                )
            bank, control_diagnostics = _sample(
                source,
                num_paths=int(args.bank_size),
                batch_size=int(args.sample_batch_size),
                seed=70000 + int(args.seed),
            )
            bank_path = arm_dir / "validation_bank.npy"
            np.save(bank_path, bank)
            metrics = _evaluate_common(
                target_prices=validation_prices,
                generated=bank,
                device=device,
                seed=int(args.seed),
            )
            write_json(arm_dir / "metrics.json", metrics)
            write_json(arm_dir / "control_diagnostics.json", control_diagnostics)
            if args.task == "mixture" and mixture_root is not None:
                _write_mixture_metrics(
                    mixture_root, bank_path, arm_dir / "mixture_metrics.json"
                )
            resources = {
                "wall_seconds": float(time.perf_counter() - started),
                "peak_gpu_allocated_gb": (
                    float(torch.cuda.max_memory_allocated()) / 2**30
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            write_json(
                arm_dir / "run_manifest.json",
                {
                    "task": args.task,
                    "arm": arm,
                    "stage": "source_only",
                    "seed": int(args.seed),
                    "selection_scope": "validation_only",
                    "test_split_loaded": False,
                    "reference_kind": str(args.reference_kind),
                    "reference_calibration": reference_calibration,
                    "reference_summary": reference.summary(),
                    "physical_drift": (
                        "zero_controlled_drift"
                        if bool(args.final_zero_drift_only)
                        else "fitted_reference_fixed"
                    ),
                    "drift_correction_admissible": False,
                    "conditional_target_estimator": "projected_score",
                    "adjoint_network": str(args.adjoint_network),
                    "transformer_layers": int(args.transformer_layers),
                    "transformer_heads": int(args.transformer_heads),
                    "conditional_projection": (
                        "heston_mixture_causal_observable_ridge"
                        if args.task == "mixture"
                        else "raw_prefix_ridge"
                    ),
                    "P_R_regression_loss_changed": bool(
                        float(args.adjoint_weight) != 1.0
                        or float(args.adjoint_noise_weight) != 1.0
                    ),
                    "Y_affects_learning": bool(float(args.adjoint_weight) > 0.0),
                    "Y_used_as_Z_control_variate": False,
                    "solver": str(args.solver),
                    "running_cost": {
                        "name": "specific_entropy",
                        "eta": float(args.eta),
                    },
                    "objective_weights": {
                        "lambda_scale": float(args.lambda_scale),
                        "kappa_scale": float(args.kappa_scale),
                        "joint_volatility_weight": 0.0,
                        "source_joint_volatility_weight": float(
                            args.source_joint_volatility_weight
                        ),
                        "abs_return_acf_weight": (
                            None
                            if args.abs_return_acf_weight is None
                            else float(args.abs_return_acf_weight)
                        ),
                        "squared_return_acf_weight": (
                            None
                            if args.squared_return_acf_weight is None
                            else float(args.squared_return_acf_weight)
                        ),
                    },
                    "source_training": asdict(source_training),
                    "source_nested": (
                        asdict(_nested(outer_steps=10, seed=int(args.seed)))
                        if str(args.solver) == "nested"
                        else None
                    ),
                    "source_fit": (
                        _history_summary(source_fit.history)
                        if str(args.solver) == "nested"
                        else {
                            "logged_steps": len(source_fit.history),
                            "final_step": float(
                                source_fit.history[-1].get("step", 0.0)
                            ),
                            "final_loss": float(
                                source_fit.history[-1].get(
                                    "train_total_loss", float("nan")
                                )
                            ),
                        }
                    ),
                    "joint_training": None,
                    "joint_nested": None,
                    "joint_fit": None,
                    "resources": resources,
                },
            )
            del source, bank
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        del source
        if device.type == "cuda":
            torch.cuda.empty_cache()

        candidate = _model(
            grid=grid,
            control=control,
            discrepancy=complete,
            estimator=estimator,
            device=device,
            seed=int(args.seed),
            adjoint_network=str(args.adjoint_network),
            transformer_layers=int(args.transformer_layers),
            transformer_heads=int(args.transformer_heads),
        )
        candidate.load_checkpoint_state(source_checkpoint)
        joint_seed = int(args.seed) + 1
        joint_training = _training(
            task=args.task,
            steps=int(args.joint_steps),
            seed=joint_seed,
            lambda_scale=float(args.lambda_scale),
            lr=float(args.lr),
            grad_clip_norm=args.grad_clip_norm,
            adjoint_weight=float(args.adjoint_weight),
            adjoint_noise_weight=float(args.adjoint_noise_weight),
            path_derivative_backend=str(args.path_derivative_backend),
            drift_adjoint_backend=str(args.drift_adjoint_backend),
        )
        if str(args.solver) == "nested":
            joint_fit = candidate.fit_nested(
                target_paths,
                training=joint_training,
                nested=_nested(outer_steps=8, seed=joint_seed),
                log_callback=_progress(arm, "joint"),
            )
        else:
            joint_fit = candidate.fit(
                target_paths,
                training=joint_training,
                log_callback=_online_progress(arm, "joint"),
            )
        torch.save(candidate.checkpoint_state(), arm_dir / "model_checkpoint.pt")
        write_history_jsonl(arm_dir / "joint_history.jsonl", joint_fit.history)
        bank, control_diagnostics = _sample(
            candidate,
            num_paths=int(args.bank_size),
            batch_size=int(args.sample_batch_size),
            seed=70000 + int(args.seed),
        )
        bank_path = arm_dir / "validation_bank.npy"
        np.save(bank_path, bank)
        metrics = _evaluate_common(
            target_prices=validation_prices,
            generated=bank,
            device=device,
            seed=int(args.seed),
        )
        write_json(arm_dir / "metrics.json", metrics)
        write_json(arm_dir / "control_diagnostics.json", control_diagnostics)
        if args.task == "mixture" and mixture_root is not None:
            _write_mixture_metrics(
                mixture_root, bank_path, arm_dir / "mixture_metrics.json"
            )
        resources = {
            "wall_seconds": float(time.perf_counter() - started),
            "peak_gpu_allocated_gb": (
                float(torch.cuda.max_memory_allocated()) / 2**30
                if device.type == "cuda"
                else 0.0
            ),
        }
        write_json(
            arm_dir / "run_manifest.json",
            {
                "task": args.task,
                "arm": arm,
                "seed": int(args.seed),
                "selection_scope": "validation_only",
                "test_split_loaded": False,
                "reference_kind": str(args.reference_kind),
                "reference_calibration": reference_calibration,
                "reference_summary": reference.summary(),
                "physical_drift": (
                    "zero_controlled_drift"
                    if bool(args.final_zero_drift_only)
                    else (
                        "fitted_reference_fixed"
                        if arm
                        in {
                            "volatility_only_nested_mp",
                            "volatility_only_online_mp",
                        }
                        else "fitted_reference_plus_correction"
                    )
                ),
                "drift_correction_admissible": (
                    False
                    if bool(args.final_zero_drift_only)
                    or bool(args.fitted_reference_drift_only)
                    else arm == "full_control_nested_mp"
                ),
                "conditional_target_estimator": "projected_score",
                "adjoint_network": str(args.adjoint_network),
                "transformer_layers": int(args.transformer_layers),
                "transformer_heads": int(args.transformer_heads),
                "conditional_projection": (
                    "heston_mixture_causal_observable_ridge"
                    if args.task == "mixture"
                    else "raw_prefix_ridge"
                ),
                "P_R_regression_loss_changed": bool(
                    float(args.adjoint_weight) != 1.0
                    or float(args.adjoint_noise_weight) != 1.0
                ),
                "Y_affects_learning": bool(float(args.adjoint_weight) > 0.0),
                "Y_used_as_Z_control_variate": False,
                "solver": str(args.solver),
                "running_cost": {
                    "name": "specific_entropy",
                    "eta": float(args.eta),
                },
                "objective_weights": {
                    "lambda_scale": float(args.lambda_scale),
                    "kappa_scale": float(args.kappa_scale),
                    "joint_volatility_weight": float(
                        args.joint_volatility_weight
                    ),
                    "source_joint_volatility_weight": float(
                        args.source_joint_volatility_weight
                    ),
                    "abs_return_acf_weight": (
                        None
                        if args.abs_return_acf_weight is None
                        else float(args.abs_return_acf_weight)
                    ),
                    "squared_return_acf_weight": (
                        None
                        if args.squared_return_acf_weight is None
                        else float(args.squared_return_acf_weight)
                    ),
                },
                "source_training": asdict(source_training),
                "source_nested": (
                    asdict(_nested(outer_steps=10, seed=int(args.seed)))
                    if str(args.solver) == "nested"
                    else None
                ),
                "joint_training": asdict(joint_training),
                "joint_nested": (
                    asdict(_nested(outer_steps=8, seed=joint_seed))
                    if str(args.solver) == "nested"
                    else None
                ),
                "source_fit": (
                    _history_summary(source_fit.history)
                    if str(args.solver) == "nested"
                    else {
                        "logged_steps": len(source_fit.history),
                        "final_step": float(source_fit.history[-1].get("step", 0.0)),
                        "final_loss": float(source_fit.history[-1].get("train_total_loss", float("nan"))),
                    }
                ),
                "joint_fit": (
                    _history_summary(joint_fit.history)
                    if str(args.solver) == "nested"
                    else {
                        "logged_steps": len(joint_fit.history),
                        "final_step": float(joint_fit.history[-1].get("step", 0.0)),
                        "final_loss": float(joint_fit.history[-1].get("train_total_loss", float("nan"))),
                    }
                ),
                "resources": resources,
            },
        )
        del candidate, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
