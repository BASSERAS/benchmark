#!/usr/bin/env python3
"""Run the accuracy-gated, damped Monte-Carlo-CE solver on Heston data.

This is a validation-only comparison against the established 3,000-step
online MP arm.  It keeps the Heston control problem fixed: fitted reference
drift, volatility-only MP correction, the source discrepancy with
``lambda=50`` and ``kappa=100``, and specific-entropy weight ``eta=1``.
Only the numerical solver changes.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
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
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
    fit_local_gaussian_reference_kernel,
)
from deep_mkv_gen_path_dt.ce import RidgeConditionalExpectation  # noqa: E402
from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    ReferenceDriftSpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.noise import (  # noqa: E402
    derive_stream_seed,
    sample_standard_normals,
)
from path_dt_experiments.accuracy_gated_nested import (  # noqa: E402
    FrozenCETrainingConfig,
    ReferenceControlPolicy,
    RelaxedLogVolatilityPolicy,
    ScaledAdjointControlPolicy,
    cpu_network_state,
    fit_frozen_sampled_ce,
)
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    CausalControlPolicy,
    ScenarioCollocationBank,
    ScenarioDataContext,
    _control_safety,
    build_scenario_collocation_bank,
    policy_moments_on_paths,
    rollout_causal_policy,
    rollout_policy_on_bank,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_heston_mixture_deep_mkv_ablation import _load_log_paths  # noqa: E402
from run_matched_control_synthetic_validation import _evaluate_common  # noqa: E402


DEFAULT_TRAIN = Path(
    "/home/samer/scenarios/BASSERAS-benchmark/dataset/Heston/"
    "heston_S_8192x128.npy"
)
DEFAULT_VALIDATION = Path(
    "/home/samer/scenarios/BASSERAS-benchmark/dataset/Heston/"
    "heston_S_disc_8192x128.npy"
)
DEFAULT_COMPARATOR = REPO_ROOT / (
    "runs/heston_online_fitted_drift_volatility_only_50_100_trajectory_"
    "seed0_20260805"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-data", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--comparator-run", type=Path, default=DEFAULT_COMPARATOR)
    parser.add_argument("--lambda-scale", type=float, default=50.0)
    parser.add_argument("--kappa-scale", type=float, default=100.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--outer-steps", type=int, default=10)
    parser.add_argument("--target-paths", type=int, default=2048)
    parser.add_argument("--population-paths", type=int, default=1024)
    parser.add_argument("--fit-paths", type=int, default=256)
    parser.add_argument("--validation-paths", type=int, default=128)
    parser.add_argument("--fit-branches", type=int, default=64)
    parser.add_argument("--validation-branches", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=16384)
    parser.add_argument("--inner-max-steps", type=int, default=2000)
    parser.add_argument("--inner-min-steps", type=int, default=400)
    parser.add_argument("--inner-batch-size", type=int, default=256)
    parser.add_argument("--inner-eval-every", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--ce-relative-tolerance", type=float, default=0.20)
    parser.add_argument(
        "--ce-gate-mode",
        choices=("accuracy", "directional", "advisory"),
        default="accuracy",
        help=(
            "accuracy also enforces the noise-adjusted relative-error ceiling; "
            "directional requires positive explained energy, correlation, and "
            "a nondegenerate amplitude; advisory records both diagnostics but "
            "always delegates acceptance to the independent objective and "
            "safety banks."
        ),
    )
    parser.add_argument(
        "--conditional-target-source",
        choices=("nested_mc", "projected_score"),
        default="nested_mc",
        help=(
            "nested_mc uses conditional future branching; projected_score uses "
            "the previous online MP score target followed by ridge projection."
        ),
    )
    parser.add_argument("--projected-target-paths", type=int, default=256)
    parser.add_argument("--objective-paths", type=int, default=4096)
    parser.add_argument("--selection-banks", type=int, default=3)
    parser.add_argument("--confirmation-banks", type=int, default=3)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1e-4)
    parser.add_argument("--maximum-upper-cap-activity", type=float, default=0.01)
    parser.add_argument("--final-bank-size", type=int, default=8192)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    parser.add_argument("--final-bank-seed", type=int, default=70000)
    parser.add_argument(
        "--resume-accepted-chain",
        type=Path,
        default=None,
        help=(
            "Resume after the last compressed single-GRU policy stored in an "
            "accepted_policy_chain.pt file. Outer-step seeds retain their "
            "original indices."
        ),
    )
    parser.add_argument(
        "--compress-accepted-policy",
        action="store_true",
        help=(
            "After each accepted law move, distill its implied effective R into "
            "one GRU and deploy it only after independent law-preservation gates."
        ),
    )
    parser.add_argument("--compression-fit-paths", type=int, default=2048)
    parser.add_argument("--compression-validation-paths", type=int, default=1024)
    parser.add_argument("--compression-max-steps", type=int, default=1500)
    parser.add_argument("--compression-min-steps", type=int, default=300)
    parser.add_argument("--compression-eval-every", type=int, default=100)
    parser.add_argument("--compression-r-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--compression-objective-relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--compression-mean-kl-tolerance", type=float, default=1e-4)
    parser.add_argument("--compression-q95-kl-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--skip-final-evaluation",
        action="store_true",
        help="For a reduced resource smoke test, stop after the outer loop.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        "outer_steps",
        "target_paths",
        "population_paths",
        "fit_paths",
        "validation_paths",
        "fit_branches",
        "validation_branches",
        "query_batch_size",
        "inner_max_steps",
        "inner_min_steps",
        "inner_batch_size",
        "inner_eval_every",
        "hidden_dim",
        "num_layers",
        "objective_paths",
        "selection_banks",
        "confirmation_banks",
        "final_bank_size",
        "sample_batch_size",
        "compression_fit_paths",
        "compression_validation_paths",
        "compression_max_steps",
        "compression_min_steps",
        "compression_eval_every",
        "projected_target_paths",
    )
    for name in positive_ints:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if int(args.inner_min_steps) > int(args.inner_max_steps):
        raise ValueError("inner-min-steps cannot exceed inner-max-steps")
    if int(args.compression_min_steps) > int(args.compression_max_steps):
        raise ValueError("compression-min-steps cannot exceed compression-max-steps")
    for name in ("fit_branches", "validation_branches", "population_paths", "objective_paths"):
        if int(getattr(args, name)) % 2:
            raise ValueError(f"{name} must be even for antithetic sampling")
    for name in ("lambda_scale", "kappa_scale", "eta"):
        if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0.0 < float(args.ce_relative_tolerance) < 1.0:
        raise ValueError("ce-relative-tolerance must lie in (0, 1)")
    if float(args.minimum_relative_improvement) < 0.0:
        raise ValueError("minimum-relative-improvement must be nonnegative")
    if not 0.0 <= float(args.maximum_upper_cap_activity) <= 1.0:
        raise ValueError("maximum-upper-cap-activity must lie in [0, 1]")
    for name in (
        "compression_r_relative_tolerance",
        "compression_objective_relative_tolerance",
        "compression_mean_kl_tolerance",
        "compression_q95_kl_tolerance",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")


def _model(
    *,
    grid: DiscreteTimeGrid,
    reference,
    discrepancy,
    device: torch.device,
    seed: int,
    eta: float,
    hidden_dim: int,
    num_layers: int,
) -> DiscreteMPModel:
    control = ReferenceDriftSpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=float(eta),
        reference_kernel=reference,
        sigma_min=1e-3,
        sigma_max=0.6,
        allow_drift_correction=False,
    )
    model = DiscreteMPModel(
        architecture=DiscreteMPArchitectureConfig(
            state_dim=1,
            noise_dim=1,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
        ),
        grid=grid,
        dynamics_step=DriftVolatilityEulerStep(dt=grid.dt),
        control_map=control,
        discrepancy=discrepancy,
        ce_estimator=RidgeConditionalExpectation(ridge=1e-3),
        init_seed=int(seed),
    )
    model.network.to(device=device, dtype=torch.float32)
    return model


def _training(args: argparse.Namespace) -> DiscreteMPTrainingConfig:
    return DiscreteMPTrainingConfig(
        batch_size=256,
        target_batch_size=256,
        num_steps=int(args.outer_steps) * int(args.inner_max_steps),
        lr=2e-3,
        weight_decay=1e-5,
        lambda_scale=float(args.lambda_scale),
        adjoint_weight=1.0,
        adjoint_noise_weight=1.0,
        ce_target_mode=(
            "ridge"
            if str(args.conditional_target_source) == "projected_score"
            else "direct"
        ),
        ridge_lambda=1e-3,
        ce_crossfit_folds=1,
        target_preconditioner="none",
        noise_target_control_variate="none",
        grad_clip_norm=5.0,
        log_every=100,
        seed=int(args.seed),
        observed_only=False,
    )


def _antithetic_noise(
    *, model: DiscreteMPModel, count: int, seed: int
) -> torch.Tensor:
    if int(count) < 2 or int(count) % 2:
        raise ValueError("antithetic path count must be positive and even")
    half = sample_standard_normals(
        grid=model.grid,
        batch_size=int(count) // 2,
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=int(seed),
    )
    return torch.cat((half, -half), dim=0)


def _complete_target_bank(
    bank: ScenarioCollocationBank, *, target_count: int, label: str | None = None
) -> ScenarioCollocationBank:
    return ScenarioCollocationBank(
        initial_indices=bank.initial_indices,
        target_indices=torch.arange(int(target_count), dtype=torch.long),
        base_noise=bank.base_noise,
        branch_seed=int(bank.branch_seed),
        branch_stream_offset=int(bank.branch_stream_offset),
        num_branches=int(bank.num_branches),
        antithetic=bool(bank.antithetic),
        beta=float(bank.beta),
        dataset_fingerprint=bank.dataset_fingerprint,
        label=bank.label if label is None else str(label),
    )


def _replication_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left64 = left.detach().double()
    right64 = right.detach().double()
    scale = torch.sqrt(torch.mean(0.5 * (left64.pow(2) + right64.pow(2)))).clamp_min(1e-30)
    centered_left = left64 - left64.mean(dim=0, keepdim=True)
    centered_right = right64 - right64.mean(dim=0, keepdim=True)
    denominator = (
        torch.linalg.vector_norm(centered_left.reshape(-1))
        * torch.linalg.vector_norm(centered_right.reshape(-1))
    ).clamp_min(1e-30)
    disagreement = torch.mean((left64 - right64).pow(2)).sqrt() / scale
    return {
        "symmetric_relative_disagreement": float(disagreement.item()),
        "estimated_average_label_noise_floor": float(0.5 * disagreement.item()),
        "correlation": float(
            (torch.sum(centered_left * centered_right) / denominator).item()
        ),
        "amplitude_ratio": float(
            (
                right64.pow(2).mean().sqrt()
                / left64.pow(2).mean().sqrt().clamp_min(1e-30)
            ).item()
        ),
    }


@dataclass(frozen=True)
class _ProjectedCELaw:
    paths: torch.Tensor
    target_p: torch.Tensor
    target_r: torch.Tensor
    metadata: dict[str, Any]


def _projected_score_ce_on_bank(
    *,
    model: DiscreteMPModel,
    policy: CausalControlPolicy,
    bank: ScenarioCollocationBank,
    context: ScenarioDataContext,
    training: DiscreteMPTrainingConfig,
) -> _ProjectedCELaw:
    """Reproduce the previous online score/Ridge CE target on a frozen bank."""

    bank.verify(context)
    initial_paths = context.initial_state_pool.index_select(
        0, bank.initial_indices.to(context.initial_state_pool.device)
    )
    target_paths = context.target_path_pool.index_select(
        0, bank.target_indices.to(context.target_path_pool.device)
    ).to(device=model.device, dtype=model.dtype)
    base_noise = bank.base_noise.to(device=model.device, dtype=model.dtype)
    x0 = initial_paths[:, 0, :].to(device=model.device, dtype=model.dtype)
    with torch.no_grad():
        rollout = rollout_causal_policy(
            model=model,
            policy=policy,
            x0=x0,
            noise=base_noise,
        )
    pathwise_p, source_metrics = model._path_functional_adjoint_targets(
        generated_path=rollout.paths,
        controls=rollout.controls,
        target_path=target_paths,
        training=training,
    )
    projected_p, p_metrics = model._project_pathwise_target(
        frozen_generated_path=rollout.paths.detach(),
        pathwise_target=pathwise_p.detach(),
        training=training,
        noise_adjoint=False,
    )
    raw_r = model._adjoint_noise_target(
        pathwise_adjoint_target=pathwise_p.detach(),
        moments_noise_target_dim=int(model.architecture.state_dim),
        noise=base_noise,
    )
    projected_r, r_metrics = model._project_pathwise_target(
        frozen_generated_path=rollout.paths.detach(),
        pathwise_target=raw_r.detach(),
        training=training,
        noise_adjoint=True,
    )
    return _ProjectedCELaw(
        paths=rollout.paths.detach(),
        target_p=projected_p.detach(),
        target_r=projected_r.detach(),
        metadata={
            "source": source_metrics,
            "p_projection": p_metrics,
            "r_projection": r_metrics,
        },
    )


def _objective_banks(
    *,
    model: DiscreteMPModel,
    target_paths: torch.Tensor,
    count: int,
    paths_per_bank: int,
    seed: int,
) -> list[dict[str, Any]]:
    x0_value = target_paths[0, 0, :].detach()
    banks: list[dict[str, Any]] = []
    for index in range(int(count)):
        bank_seed = derive_stream_seed(
            base_seed=int(seed), stream_offset=10_000 * index
        )
        assert bank_seed is not None
        banks.append(
            {
                "seed": int(bank_seed),
                "x0": x0_value.unsqueeze(0).expand(int(paths_per_bank), -1).clone(),
                "noise": _antithetic_noise(
                    model=model, count=int(paths_per_bank), seed=int(bank_seed)
                ),
                "target": target_paths,
            }
        )
    return banks


def _evaluate_objective_banks(
    *,
    model: DiscreteMPModel,
    policy: CausalControlPolicy,
    banks: list[dict[str, Any]],
    training: DiscreteMPTrainingConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for bank in banks:
            rollout = rollout_causal_policy(
                model=model,
                policy=policy,
                x0=bank["x0"],
                noise=bank["noise"],
            )
            value, metrics = model._complete_objective_metrics(
                generated_path=rollout.paths,
                controls=rollout.controls,
                target_path=bank["target"],
                training=training,
            )
            rows.append(
                {
                    "seed": int(bank["seed"]),
                    "value": float(value.item()),
                    "metrics": metrics,
                    "safety": _control_safety(
                        model=model,
                        paths=rollout.paths,
                        controls=rollout.controls,
                        moment_r=rollout.expected_adjoint_noise_next,
                    ),
                }
            )
    return rows


def _objective_comparison(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> dict[str, Any]:
    if not before or len(before) != len(after):
        raise ValueError("objective rows must be nonempty and aligned")
    rows = []
    for index, (left, right) in enumerate(zip(before, after, strict=True)):
        left_value = float(left["value"])
        right_value = float(right["value"])
        rows.append(
            {
                "bank": index,
                "seed": int(left["seed"]),
                "before": left_value,
                "after": right_value,
                "improvement": left_value - right_value,
                "relative_improvement": (
                    (left_value - right_value) / max(abs(left_value), 1e-30)
                ),
                "improved": bool(right_value < left_value),
                "safety": right["safety"],
            }
        )
    return {
        "before_mean": float(sum(float(row["value"]) for row in before) / len(before)),
        "after_mean": float(sum(float(row["value"]) for row in after) / len(after)),
        "relative_improvement_mean": float(
            sum(float(row["relative_improvement"]) for row in rows) / len(rows)
        ),
        "all_improved": bool(all(bool(row["improved"]) for row in rows)),
        "rows": rows,
    }


def _safety_passed(report: dict[str, Any], *, maximum_upper_activity: float) -> bool:
    for row in report["rows"]:
        safety = row["safety"]
        if float(safety.get("controls_finite", 0.0)) < 0.5:
            return False
        if float(safety.get("specific_entropy_denominator_margin_min", -math.inf)) <= 0.0:
            return False
        if float(safety.get("sigma_upper_bound_activity", 1.0)) > float(maximum_upper_activity):
            return False
    return True


def _sample_policy(
    *,
    model: DiscreteMPModel,
    policy: CausalControlPolicy,
    num_paths: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    prices = np.empty((int(num_paths), int(model.grid.num_steps) + 1), dtype=np.float32)
    sigma_values: list[np.ndarray] = []
    drift_errors: list[np.ndarray] = []
    denominator_values: list[np.ndarray] = []
    x0_value = torch.zeros(1, device=model.device, dtype=model.dtype)
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, int(num_paths), int(batch_size))):
            count = min(int(batch_size), int(num_paths) - start)
            noise_seed = derive_stream_seed(
                base_seed=int(seed), stream_offset=10_000 + batch_index
            )
            assert noise_seed is not None
            noise = sample_standard_normals(
                grid=model.grid,
                batch_size=count,
                noise_dim=int(model.architecture.noise_dim),
                device=model.device,
                dtype=model.dtype,
                seed=noise_seed,
            )
            rollout = rollout_causal_policy(
                model=model,
                policy=policy,
                x0=x0_value.unsqueeze(0).expand(count, -1).clone(),
                noise=noise,
            )
            log_paths = rollout.paths[..., 0]
            controls = rollout.controls
            reference_drift = model.control_map._reference_drifts_all(rollout.paths)
            reference_sigma = model.control_map._reference_sigmas_all(rollout.paths)
            denominator = (
                float(model.control_map.eta) / reference_sigma
                + rollout.expected_adjoint_noise_next / math.sqrt(float(model.grid.dt))
            )
            prices[start : start + count] = (
                100.0 * torch.exp(log_paths)
            ).cpu().numpy().astype(np.float32)
            sigma_values.append(controls[..., 1:].cpu().numpy())
            drift_errors.append(torch.abs(controls[..., :1] - reference_drift).cpu().numpy())
            denominator_values.append(denominator.cpu().numpy())
    sigma = np.concatenate(sigma_values).reshape(-1)
    drift_error = np.concatenate(drift_errors).reshape(-1)
    denominator = np.concatenate(denominator_values).reshape(-1)
    return prices, {
        "sigma_mean": float(sigma.mean()),
        "sigma_std": float(sigma.std()),
        "sigma_q99": float(np.quantile(sigma, 0.99)),
        "sigma_min_active_fraction": float(np.mean(sigma <= 1e-3 * (1.0 + 1e-5))),
        "sigma_max_active_fraction": float(np.mean(sigma >= 0.6 * (1.0 - 1e-5))),
        "reference_drift_max_abs_error": float(drift_error.max()),
        "specific_entropy_denominator_min": float(denominator.min()),
        "specific_entropy_denominator_nonpositive_fraction": float(np.mean(denominator <= 0.0)),
        "nonfinite_control_fraction": float(1.0 - np.mean(np.isfinite(np.concatenate((sigma, drift_error, denominator))))),
        "nonfinite_path_fraction": float(1.0 - np.mean(np.isfinite(prices))),
    }


def _compress_accepted_policy(
    *,
    args: argparse.Namespace,
    model: DiscreteMPModel,
    composite_policy: CausalControlPolicy,
    fixed_target: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    outer_seed: int,
    initial_network_state: dict[str, torch.Tensor],
) -> tuple[CausalControlPolicy | None, dict[str, Any]]:
    """Project an accepted policy chain into one effective-R GRU.

    This is representation-only: the regression targets are the exact implied
    R values of the already accepted bounded volatility controls.  Fresh MP
    adjoint targets are not substituted or modified here.
    """

    fit_count = int(args.compression_fit_paths)
    validation_count = int(args.compression_validation_paths)
    if fit_count % 2 or validation_count % 2:
        raise ValueError("compression path counts must be even")
    x0_value = fixed_target[0, 0, :].detach()

    def rollout(count: int, seed: int):
        return rollout_causal_policy(
            model=model,
            policy=composite_policy,
            x0=x0_value.unsqueeze(0).expand(int(count), -1).clone(),
            noise=_antithetic_noise(model=model, count=int(count), seed=int(seed)),
        )

    with torch.no_grad():
        fit_law = rollout(fit_count, outer_seed + 30_001)
        validation_law = rollout(validation_count, outer_seed + 30_002)
    zero_fit_p = torch.zeros_like(fit_law.expected_adjoint_next)
    zero_validation_p = torch.zeros_like(validation_law.expected_adjoint_next)
    compression_fit = fit_frozen_sampled_ce(
        model=model,
        fit_paths=fit_law.paths,
        fit_target_p=zero_fit_p,
        fit_target_r=fit_law.expected_adjoint_noise_next,
        validation_paths=validation_law.paths,
        validation_target_p=zero_validation_p,
        validation_target_r=validation_law.expected_adjoint_noise_next,
        config=FrozenCETrainingConfig(
            max_steps=int(args.compression_max_steps),
            minimum_steps=int(args.compression_min_steps),
            batch_size=int(args.inner_batch_size),
            eval_every=int(args.compression_eval_every),
            lr=2e-3,
            weight_decay=1e-5,
            grad_clip_norm=5.0,
            minimum_scale=1e-3,
            validation_relative_tolerance=float(
                args.compression_r_relative_tolerance
            ),
            p_loss_weight=0.0,
            r_loss_weight=1.0,
            seed=outer_seed + 30_003,
        ),
        initial_network_state=initial_network_state,
    )
    compressed_policy = ScaledAdjointControlPolicy(
        model=model,
        network=compression_fit.network,
        p_scale=compression_fit.p_scale,
        r_scale=compression_fit.r_scale,
    )
    verification_banks = _objective_banks(
        model=model,
        target_paths=fixed_target,
        count=int(args.confirmation_banks),
        paths_per_bank=int(args.objective_paths),
        seed=outer_seed + 40_000,
    )
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for bank in verification_banks:
            composite_rollout = rollout_causal_policy(
                model=model,
                policy=composite_policy,
                x0=bank["x0"],
                noise=bank["noise"],
            )
            compressed_rollout = rollout_causal_policy(
                model=model,
                policy=compressed_policy,
                x0=bank["x0"],
                noise=bank["noise"],
            )
            composite_objective, _ = model._complete_objective_metrics(
                generated_path=composite_rollout.paths,
                controls=composite_rollout.controls,
                target_path=bank["target"],
                training=training,
            )
            compressed_objective, _ = model._complete_objective_metrics(
                generated_path=compressed_rollout.paths,
                controls=compressed_rollout.controls,
                target_path=bank["target"],
                training=training,
            )
            _, _, compressed_controls_on_composite_paths = policy_moments_on_paths(
                model=model,
                policy=compressed_policy,
                paths=composite_rollout.paths,
            )
            state_dim = int(model.architecture.state_dim)
            old_alpha = composite_rollout.controls[..., :state_dim]
            new_alpha = compressed_controls_on_composite_paths[..., :state_dim]
            old_sigma = composite_rollout.controls[..., state_dim:]
            new_sigma = compressed_controls_on_composite_paths[..., state_dim:]
            variance_ratio = (new_sigma / old_sigma).pow(2)
            transition_kl = 0.5 * (
                variance_ratio
                - 1.0
                - torch.log(variance_ratio)
                + float(model.grid.dt)
                * (new_alpha - old_alpha).pow(2)
                / old_sigma.pow(2)
            )
            relative_objective_change = float(
                (
                    (compressed_objective - composite_objective)
                    / composite_objective.abs().clamp_min(1e-30)
                ).item()
            )
            safety = _control_safety(
                model=model,
                paths=compressed_rollout.paths,
                controls=compressed_rollout.controls,
                moment_r=compressed_rollout.expected_adjoint_noise_next,
            )
            rows.append(
                {
                    "seed": int(bank["seed"]),
                    "composite_objective": float(composite_objective.item()),
                    "compressed_objective": float(compressed_objective.item()),
                    "relative_objective_change": relative_objective_change,
                    "transition_kl_mean": float(transition_kl.mean().item()),
                    "transition_kl_q95": float(
                        torch.quantile(transition_kl.reshape(-1), 0.95).item()
                    ),
                    "transition_kl_max": float(transition_kl.max().item()),
                    "log_sigma_rmse": float(
                        torch.mean(
                            (torch.log(new_sigma) - torch.log(old_sigma)).pow(2)
                        )
                        .sqrt()
                        .item()
                    ),
                    "safety": safety,
                }
            )
    r_metrics = compression_fit.validation_metrics
    r_gate = bool(
        float(r_metrics["r_relative_error"])
        <= float(args.compression_r_relative_tolerance)
        and float(r_metrics["r_correlation"]) > 0.99
        and 0.95 <= float(r_metrics["r_amplitude_ratio"]) <= 1.05
    )
    verification_gate = bool(
        all(
            float(row["relative_objective_change"])
            <= float(args.compression_objective_relative_tolerance)
            and float(row["transition_kl_mean"])
            <= float(args.compression_mean_kl_tolerance)
            and float(row["transition_kl_q95"])
            <= float(args.compression_q95_kl_tolerance)
            and float(row["safety"].get("controls_finite", 0.0)) > 0.5
            and float(
                row["safety"].get(
                    "specific_entropy_denominator_margin_min", -math.inf
                )
            )
            > 0.0
            and float(row["safety"].get("sigma_upper_bound_activity", 1.0))
            <= float(args.maximum_upper_cap_activity)
            for row in rows
        )
    )
    report = {
        "fit_selected_step": int(compression_fit.selected_step),
        "fit_metrics": compression_fit.fit_metrics,
        "validation_metrics": compression_fit.validation_metrics,
        "r_gate_passed": r_gate,
        "verification_gate_passed": verification_gate,
        "passed": bool(r_gate and verification_gate),
        "verification_rows": rows,
        "thresholds": {
            "r_relative_error": float(args.compression_r_relative_tolerance),
            "objective_relative_deterioration": float(
                args.compression_objective_relative_tolerance
            ),
            "transition_kl_mean": float(args.compression_mean_kl_tolerance),
            "transition_kl_q95": float(args.compression_q95_kl_tolerance),
        },
    }
    return (compressed_policy if bool(report["passed"]) else None), report


def _metric_comparison(
    *, reference: dict[str, Any], online: dict[str, Any], damped: dict[str, Any]
) -> dict[str, Any]:
    path_keys = (
        "path_swd_normalized",
        "terminal_w1_normalized",
        "increment_w1_normalized",
        "realized_volatility_w1_normalized",
        "maximum_drawdown_w1_normalized",
        "return_qq_rmse_normalized",
        "timewise_return_std_rmse_normalized",
        "absolute_return_acf_rmse",
        "squared_return_acf_rmse",
    )
    rows: dict[str, Any] = {}
    for key in path_keys:
        ref = float(reference["path_law"][key])
        old = float(online["path_law"][key])
        new = float(damped["path_law"][key])
        rows[key] = {
            "reference": ref,
            "online_3000": old,
            "damped_mc_ce": new,
            "damped_gap_closed_vs_reference": (ref - new) / max(abs(ref), 1e-30),
            "damped_relative_change_vs_online": (new - old) / max(abs(old), 1e-30),
        }
    target = damped["conditional_path_memory"]["target"]
    for key in ("prefix_future_rv_correlation", "prefix_high_low_future_rv_gap"):
        truth = float(target[key])
        ref_value = float(reference["conditional_path_memory"]["generated"][key])
        old_value = float(online["conditional_path_memory"]["generated"][key])
        new_value = float(damped["conditional_path_memory"]["generated"][key])
        ref_error = abs(ref_value - truth)
        rows[f"{key}_error"] = {
            "reference": ref_error,
            "online_3000": abs(old_value - truth),
            "damped_mc_ce": abs(new_value - truth),
            "damped_gap_closed_vs_reference": (
                (ref_error - abs(new_value - truth)) / max(ref_error, 1e-30)
            ),
            "damped_relative_change_vs_online": (
                (abs(new_value - truth) - abs(old_value - truth))
                / max(abs(old_value - truth), 1e-30)
            ),
        }
    return rows


def main() -> None:
    args = parse_args()
    _validate_args(args)
    run_dir = ensure_run_dir(args.run_dir.resolve())
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    target_all = _load_log_paths(
        args.train_data.resolve(), device=device, dtype=torch.float32
    )
    validation_prices = np.asarray(
        np.load(args.validation_data.resolve(), allow_pickle=False), dtype=np.float64
    )
    num_steps = int(target_all.shape[1]) - 1
    grid = DiscreteTimeGrid(T=float(num_steps) * 0.004, num_steps=num_steps)
    reference = fit_local_gaussian_reference_kernel(
        target_paths=target_all,
        grid=grid,
        ridge=1e-3,
        variance_ridge=1e-3,
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    discrepancy = build_heston_discrepancy(
        num_steps=num_steps,
        include_acf=True,
        preset="old_fullv_w0p25",
        lambda_scale=float(args.lambda_scale),
        kappa_scale=float(args.kappa_scale),
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )
    model = _model(
        grid=grid,
        reference=reference,
        discrepancy=discrepancy,
        device=device,
        seed=int(args.seed),
        eta=float(args.eta),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
    )
    training = _training(args)

    subset_generator = torch.Generator(device="cpu").manual_seed(61_000 + int(args.seed))
    subset_indices = torch.randperm(
        int(target_all.shape[0]), generator=subset_generator
    )[: int(args.target_paths)]
    fixed_target = target_all.index_select(0, subset_indices.to(target_all.device)).detach()
    context = ScenarioDataContext(
        initial_state_pool=fixed_target,
        target_path_pool=fixed_target,
        dataset_id="heston-train-fixed-target-subset",
    )
    protocol = {
        "selection_scope": "validation_only",
        "canonical_test_split_loaded": False,
        "control_problem": {
            "physical_drift": "fitted_reference_fixed",
            "drift_correction_admissible": False,
            "controlled_coordinate": "volatility_only",
            "objective_stage": "source_only",
            "lambda_scale": float(args.lambda_scale),
            "kappa_scale": float(args.kappa_scale),
            "eta": float(args.eta),
            "sigma_bounds": [0.001, 0.6],
        },
        "solver": {
            "outer_steps": int(args.outer_steps),
            "fixed_target_paths": int(args.target_paths),
            "population_paths": int(args.population_paths),
            "fit_prefix_paths": int(args.fit_paths),
            "fit_future_branches": int(args.fit_branches),
            "validation_prefix_paths": int(args.validation_paths),
            "validation_future_branches": int(args.validation_branches),
            "validation_target_replicates": 2,
            "operative_adjoint_head": "R_only",
            "ce_gate_mode": str(args.ce_gate_mode),
            "conditional_target_source": str(args.conditional_target_source),
            "projected_target_paths": int(args.projected_target_paths),
            "relaxations": [2.0 ** (-index) for index in range(11)],
            "selection_banks": int(args.selection_banks),
            "confirmation_banks": int(args.confirmation_banks),
            "objective_paths_per_bank": int(args.objective_paths),
            "minimum_relative_improvement": float(args.minimum_relative_improvement),
            "maximum_upper_cap_activity": float(args.maximum_upper_cap_activity),
            "compress_accepted_policy": bool(args.compress_accepted_policy),
            "compression": {
                "fit_paths": int(args.compression_fit_paths),
                "validation_paths": int(args.compression_validation_paths),
                "max_steps": int(args.compression_max_steps),
                "minimum_steps": int(args.compression_min_steps),
                "r_relative_tolerance": float(
                    args.compression_r_relative_tolerance
                ),
                "objective_relative_tolerance": float(
                    args.compression_objective_relative_tolerance
                ),
                "mean_transition_kl_tolerance": float(
                    args.compression_mean_kl_tolerance
                ),
                "q95_transition_kl_tolerance": float(
                    args.compression_q95_kl_tolerance
                ),
            },
            "resume_accepted_chain": (
                None
                if args.resume_accepted_chain is None
                else str(args.resume_accepted_chain.resolve())
            ),
        },
        "ce_training": asdict(
            FrozenCETrainingConfig(
                max_steps=int(args.inner_max_steps),
                minimum_steps=int(args.inner_min_steps),
                batch_size=int(args.inner_batch_size),
                eval_every=int(args.inner_eval_every),
                lr=2e-3,
                weight_decay=1e-5,
                grad_clip_norm=5.0,
                minimum_scale=1e-3,
                validation_relative_tolerance=float(args.ce_relative_tolerance),
                p_loss_weight=0.0,
                r_loss_weight=1.0,
                seed=int(args.seed),
            )
        ),
        "adjoint_network": {
            "kind": "GRU",
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
        },
        "reference_summary": reference.summary(),
        "scenario_context_fingerprint": context.fingerprint,
    }
    write_json(run_dir / "LOCKED_PROTOCOL.json", protocol)

    current_policy: CausalControlPolicy = ReferenceControlPolicy(model)
    warm_state: dict[str, torch.Tensor] | None = None
    accepted_states: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    start_outer = 0
    if args.resume_accepted_chain is not None:
        resume_path = args.resume_accepted_chain.resolve()
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        accepted_states = list(payload.get("accepted_states", ()))
        history = list(payload.get("history", ()))
        if not accepted_states:
            raise ValueError("resume chain contains no accepted policy")
        last_state = accepted_states[-1]
        representations = [state.get("representation") for state in accepted_states]
        if str(last_state.get("representation")) == "compressed_single_gru":
            states_to_restore = (last_state,)
            restore_recursively = False
        elif all(
            representation in (None, "recursive_relaxed_policy")
            for representation in representations
        ):
            # Older checkpoints predate the explicit representation field. In
            # an uncompressed run each stored network is the stage candidate;
            # replay the accepted relaxations to reconstruct the exact law.
            states_to_restore = tuple(accepted_states)
            restore_recursively = True
        else:
            raise ValueError("resume chain has an unsupported mixed representation")
        resumed_network: torch.nn.Module | None = None
        for restored_state in states_to_restore:
            resumed_network = copy.deepcopy(model.network)
            resumed_network.load_state_dict(
                restored_state["network_state"], strict=True
            )
            resumed_network.to(device=device, dtype=model.dtype).eval()
            restored_candidate = ScaledAdjointControlPolicy(
                model=model,
                network=resumed_network,
                p_scale=restored_state["p_scale"].detach().clone(),
                r_scale=restored_state["r_scale"].detach().clone(),
            )
            if restore_recursively:
                current_policy = RelaxedLogVolatilityPolicy(
                    model=model,
                    current_policy=current_policy,
                    candidate_policy=restored_candidate,
                    relaxation=float(restored_state["relaxation"]),
                )
            else:
                current_policy = restored_candidate
        assert resumed_network is not None
        warm_state = cpu_network_state(resumed_network)
        start_outer = int(last_state["outer_step"]) + 1
        if start_outer >= int(args.outer_steps):
            raise ValueError("resume chain has already reached outer-steps")
    status = "maximum_outer_steps"
    relaxations = tuple(2.0 ** (-index) for index in range(11))

    for outer in range(start_outer, int(args.outer_steps)):
        outer_seed = derive_stream_seed(
            base_seed=1000 + int(args.seed), stream_offset=1_000_000 * outer
        )
        assert outer_seed is not None
        nested_targets = str(args.conditional_target_source) == "nested_mc"
        ce_target_count = (
            int(args.target_paths)
            if nested_targets
            else int(args.projected_target_paths)
        )
        fit_bank = build_scenario_collocation_bank(
            model=model,
            context=context,
            path_count=int(args.fit_paths),
            target_count=ce_target_count,
            num_branches=int(args.fit_branches),
            antithetic=True,
            beta=1.0,
            seed=outer_seed + 1,
            label=f"fit_{outer}",
        )
        validation_bank = build_scenario_collocation_bank(
            model=model,
            context=context,
            path_count=int(args.validation_paths),
            target_count=ce_target_count,
            num_branches=int(args.validation_branches),
            antithetic=True,
            beta=1.0,
            seed=outer_seed + 2,
            label=f"validation_{outer}",
        )
        if nested_targets:
            fit_bank = _complete_target_bank(
                fit_bank, target_count=int(args.target_paths)
            )
            validation_bank = _complete_target_bank(
                validation_bank, target_count=int(args.target_paths)
            )
            replication_target_indices = validation_bank.target_indices
        else:
            replication_index_bank = build_scenario_collocation_bank(
                model=model,
                context=context,
                path_count=int(args.validation_paths),
                target_count=ce_target_count,
                num_branches=int(args.validation_branches),
                antithetic=True,
                beta=1.0,
                seed=outer_seed + 500_002,
                label=f"validation_replication_indices_{outer}",
            )
            replication_target_indices = replication_index_bank.target_indices
        validation_replication_bank = ScenarioCollocationBank(
            initial_indices=validation_bank.initial_indices,
            target_indices=replication_target_indices,
            base_noise=validation_bank.base_noise,
            branch_seed=int(validation_bank.branch_seed) + 500_000,
            branch_stream_offset=int(validation_bank.branch_stream_offset),
            num_branches=int(validation_bank.num_branches),
            antithetic=True,
            beta=1.0,
            dataset_fingerprint=validation_bank.dataset_fingerprint,
            label=f"validation_replication_{outer}",
        )
        if nested_targets:
            population_noise = _antithetic_noise(
                model=model,
                count=int(args.population_paths),
                seed=outer_seed + 500,
            )
            population_x0 = fixed_target[0, 0, :].unsqueeze(0).expand(
                int(args.population_paths), -1
            ).clone()
            with torch.no_grad():
                population_paths = rollout_causal_policy(
                    model=model,
                    policy=current_policy,
                    x0=population_x0,
                    noise=population_noise,
                ).paths.detach()
            fit_law = rollout_policy_on_bank(
                model=model,
                policy=current_policy,
                bank=fit_bank,
                data_context=context,
                full_training=training,
                query_batch_size=int(args.query_batch_size),
                population_paths=population_paths,
            )
            validation_law = rollout_policy_on_bank(
                model=model,
                policy=current_policy,
                bank=validation_bank,
                data_context=context,
                full_training=training,
                query_batch_size=int(args.query_batch_size),
                population_paths=population_paths,
            )
            validation_replication_law = rollout_policy_on_bank(
                model=model,
                policy=current_policy,
                bank=validation_replication_bank,
                data_context=context,
                full_training=training,
                query_batch_size=int(args.query_batch_size),
                population_paths=population_paths,
            )
        else:
            fit_law = _projected_score_ce_on_bank(
                model=model,
                policy=current_policy,
                bank=fit_bank,
                context=context,
                training=training,
            )
            validation_law = _projected_score_ce_on_bank(
                model=model,
                policy=current_policy,
                bank=validation_bank,
                context=context,
                training=training,
            )
            validation_replication_law = _projected_score_ce_on_bank(
                model=model,
                policy=current_policy,
                bank=validation_replication_bank,
                context=context,
                training=training,
            )
        validation_target_p = 0.5 * (
            validation_law.target_p + validation_replication_law.target_p
        )
        validation_target_r = 0.5 * (
            validation_law.target_r + validation_replication_law.target_r
        )
        p_replication = _replication_metrics(
            validation_law.target_p, validation_replication_law.target_p
        )
        r_replication = _replication_metrics(
            validation_law.target_r, validation_replication_law.target_r
        )
        fit_config = FrozenCETrainingConfig(
            max_steps=int(args.inner_max_steps),
            minimum_steps=int(args.inner_min_steps),
            batch_size=int(args.inner_batch_size),
            eval_every=int(args.inner_eval_every),
            lr=2e-3,
            weight_decay=1e-5,
            grad_clip_norm=5.0,
            minimum_scale=1e-3,
            validation_relative_tolerance=float(args.ce_relative_tolerance),
            p_loss_weight=0.0,
            r_loss_weight=1.0,
            seed=outer_seed + 3,
        )
        fit = fit_frozen_sampled_ce(
            model=model,
            fit_paths=fit_law.paths,
            fit_target_p=fit_law.target_p,
            fit_target_r=fit_law.target_r,
            validation_paths=validation_law.paths,
            validation_target_p=validation_target_p,
            validation_target_r=validation_target_r,
            config=fit_config,
            initial_network_state=warm_state,
        )
        warm_state = cpu_network_state(fit.network)
        r_allowed_error = max(
            float(args.ce_relative_tolerance),
            1.5 * float(r_replication["estimated_average_label_noise_floor"]),
        )
        r_metrics = fit.validation_metrics
        directional_ce_gate = bool(
            float(r_metrics["r_explained_energy"]) > 0.0
            and float(r_metrics["r_correlation"]) > 0.5
            and 0.5 <= float(r_metrics["r_amplitude_ratio"]) <= 1.5
        )
        accuracy_ce_gate = bool(
            directional_ce_gate
            and float(r_metrics["r_relative_error"]) <= r_allowed_error
        )
        if str(args.ce_gate_mode) == "advisory":
            ce_gate = True
        elif str(args.ce_gate_mode) == "directional":
            ce_gate = directional_ce_gate
        else:
            ce_gate = accuracy_ce_gate
        row: dict[str, Any] = {
            "outer_step": outer,
            "ce_gate_mode": str(args.ce_gate_mode),
            "ce_gate_passed": ce_gate,
            "ce_accuracy_gate_passed": accuracy_ce_gate,
            "ce_directional_gate_passed": directional_ce_gate,
            "ce_relative_error_gate_passed": bool(
                float(r_metrics["r_relative_error"]) <= r_allowed_error
            ),
            "fit_selected_step": int(fit.selected_step),
            "fit_metrics": fit.fit_metrics,
            "validation_metrics": fit.validation_metrics,
            "p_target_replication": p_replication,
            "r_target_replication": r_replication,
            "r_allowed_relative_error": r_allowed_error,
            "fit_bank_fingerprint": fit_bank.fingerprint,
            "validation_bank_fingerprint": validation_bank.fingerprint,
        }
        if not ce_gate:
            row["update_accepted"] = False
            history.append(row)
            status = "ce_accuracy_gate_failed"
            print(json.dumps(row, sort_keys=True), flush=True)
            write_json(run_dir / "progress.json", {"status": status, "history": history})
            break

        candidate_policy = ScaledAdjointControlPolicy(
            model=model,
            network=fit.network,
            p_scale=fit.p_scale,
            r_scale=fit.r_scale,
        )
        selection_banks = _objective_banks(
            model=model,
            target_paths=fixed_target,
            count=int(args.selection_banks),
            paths_per_bank=int(args.objective_paths),
            seed=outer_seed + 10_000,
        )
        confirmation_banks = _objective_banks(
            model=model,
            target_paths=fixed_target,
            count=int(args.confirmation_banks),
            paths_per_bank=int(args.objective_paths),
            seed=outer_seed + 20_000,
        )
        selection_before = _evaluate_objective_banks(
            model=model,
            policy=current_policy,
            banks=selection_banks,
            training=training,
        )
        candidates: list[tuple[float, float, CausalControlPolicy, dict[str, Any]]] = []
        for rate in relaxations:
            policy = RelaxedLogVolatilityPolicy(
                model=model,
                current_policy=current_policy,
                candidate_policy=candidate_policy,
                relaxation=rate,
            )
            after = _evaluate_objective_banks(
                model=model, policy=policy, banks=selection_banks, training=training
            )
            report = _objective_comparison(selection_before, after)
            if (
                bool(report["all_improved"])
                and float(report["relative_improvement_mean"])
                >= float(args.minimum_relative_improvement)
                and _safety_passed(
                    report,
                    maximum_upper_activity=float(args.maximum_upper_cap_activity),
                )
            ):
                candidates.append(
                    (float(report["after_mean"]), float(rate), policy, report)
                )
        candidates.sort(key=lambda item: item[0])
        confirmation_before = _evaluate_objective_banks(
            model=model,
            policy=current_policy,
            banks=confirmation_banks,
            training=training,
        )
        accepted = None
        last_confirmation = None
        for _, rate, policy, selection_report in candidates:
            after = _evaluate_objective_banks(
                model=model, policy=policy, banks=confirmation_banks, training=training
            )
            confirmation_report = _objective_comparison(confirmation_before, after)
            last_confirmation = confirmation_report
            if (
                bool(confirmation_report["all_improved"])
                and float(confirmation_report["relative_improvement_mean"])
                >= float(args.minimum_relative_improvement)
                and _safety_passed(
                    confirmation_report,
                    maximum_upper_activity=float(args.maximum_upper_cap_activity),
                )
            ):
                accepted = (rate, policy, selection_report, confirmation_report)
                break
        if accepted is None:
            row.update(
                {
                    "update_accepted": False,
                    "selection_candidate_count": len(candidates),
                    "last_confirmation": last_confirmation,
                }
            )
            history.append(row)
            status = "objective_confirmation_rejected"
            print(json.dumps(row, sort_keys=True), flush=True)
            write_json(run_dir / "progress.json", {"status": status, "history": history})
            break
        rate, accepted_composite, selection_report, confirmation_report = accepted
        compression_report = None
        if bool(args.compress_accepted_policy):
            compressed_policy, compression_report = _compress_accepted_policy(
                args=args,
                model=model,
                composite_policy=accepted_composite,
                fixed_target=fixed_target,
                training=training,
                outer_seed=outer_seed,
                initial_network_state=cpu_network_state(fit.network),
            )
            if compressed_policy is None:
                row.update(
                    {
                        "mp_update_accepted_before_compression": True,
                        "update_accepted": False,
                        "relaxation": float(rate),
                        "selection_candidate_count": len(candidates),
                        "selection": selection_report,
                        "confirmation": confirmation_report,
                        "compression": compression_report,
                    }
                )
                history.append(row)
                status = "accepted_policy_compression_failed"
                print(json.dumps(row, sort_keys=True), flush=True)
                write_json(
                    run_dir / "progress.json",
                    {
                        "status": status,
                        "elapsed_seconds": time.perf_counter() - started,
                        "history": history,
                    },
                )
                break
            current_policy = compressed_policy
        else:
            current_policy = accepted_composite
        deployed_network = getattr(current_policy, "network", fit.network)
        deployed_p_scale = getattr(current_policy, "p_scale", fit.p_scale)
        deployed_r_scale = getattr(current_policy, "r_scale", fit.r_scale)
        accepted_states.append(
            {
                "outer_step": outer,
                "relaxation": float(rate),
                "representation": (
                    "compressed_single_gru"
                    if bool(args.compress_accepted_policy)
                    else "recursive_relaxed_policy"
                ),
                "network_state": cpu_network_state(deployed_network),
                "p_scale": deployed_p_scale.detach().cpu(),
                "r_scale": deployed_r_scale.detach().cpu(),
            }
        )
        row.update(
            {
                "update_accepted": True,
                "relaxation": float(rate),
                "selection_candidate_count": len(candidates),
                "selection": selection_report,
                "confirmation": confirmation_report,
                "compression": compression_report,
            }
        )
        history.append(row)
        torch.save(
            {"accepted_states": accepted_states, "history": history},
            run_dir / "accepted_policy_chain.pt",
        )
        write_json(
            run_dir / "progress.json",
            {
                "status": "running",
                "elapsed_seconds": time.perf_counter() - started,
                "history": history,
            },
        )
        print(json.dumps(row, sort_keys=True), flush=True)

    final: dict[str, Any] = {
        "status": status,
        "accepted_outer_steps": int(sum(bool(row.get("update_accepted")) for row in history)),
        "history": history,
        "resources": {
            "wall_seconds_before_final_evaluation": float(time.perf_counter() - started),
            "peak_gpu_allocated_gb": (
                float(torch.cuda.max_memory_allocated(device)) / 2**30
                if device.type == "cuda"
                else 0.0
            ),
        },
    }
    if not bool(args.skip_final_evaluation):
        generated, controls = _sample_policy(
            model=model,
            policy=current_policy,
            num_paths=int(args.final_bank_size),
            batch_size=int(args.sample_batch_size),
            seed=int(args.final_bank_seed) + int(args.seed),
        )
        np.save(run_dir / "validation_bank.npy", generated)
        metrics = _evaluate_common(
            target_prices=validation_prices,
            generated=generated,
            device=device,
            seed=int(args.seed),
        )
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "control_diagnostics.json", controls)
        reference_metrics = json.loads(
            (args.comparator_run.resolve() / "reference" / "metrics.json").read_text()
        )
        online_metrics = json.loads(
            (
                args.comparator_run.resolve()
                / "volatility_only_online_mp"
                / "metrics.json"
            ).read_text()
        )
        comparison = _metric_comparison(
            reference=reference_metrics, online=online_metrics, damped=metrics
        )
        write_json(run_dir / "COMPARISON.json", comparison)
        final.update(
            {
                "metrics": metrics,
                "control_diagnostics": controls,
                "comparison": comparison,
                "common_final_bank_seed": int(args.final_bank_seed) + int(args.seed),
                "common_final_bank_size": int(args.final_bank_size),
            }
        )
    final["resources"]["wall_seconds_total"] = float(time.perf_counter() - started)
    write_json(run_dir / "SUMMARY.json", final)
    print(json.dumps({"status": status, "run_dir": str(run_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
