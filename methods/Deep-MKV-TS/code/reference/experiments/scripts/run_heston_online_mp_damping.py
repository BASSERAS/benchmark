#!/usr/bin/env python3
"""Select a scalar damping of the established Heston online-MP policy.

The online checkpoint and its projected-score/ridge conditional-expectation
training are not changed.  Damping acts only at deployment, by interpolating
the online MP volatility toward the fitted reference in log-volatility space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteTimeGrid,
    fit_local_gaussian_reference_kernel,
)
from deep_mkv_gen_path_dt.ce import RidgeConditionalExpectation  # noqa: E402
from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    HamiltonianControlInputs,
    ReferenceDriftSpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.controls.base import RunningCostInputs  # noqa: E402
from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_heston_mixture_deep_mkv_ablation import _load_log_paths  # noqa: E402
from run_matched_control_synthetic_validation import (  # noqa: E402
    _evaluate_common,
    _model,
)


DEFAULT_SOURCE_RUN = REPO_ROOT / "runs" / (
    "heston_online_fitted_drift_volatility_only_50_100_trajectory_"
    "seed0_20260805"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / (
    "heston_online_mp_logvol_damping_seed0_20260805"
)
HESTON_ROOT = Path("/home/samer/scenarios/BASSERAS-benchmark/dataset/Heston")


class DampedReferenceVolatilityControl:
    """Apply one scalar relaxation to a fitted-reference volatility correction."""

    def __init__(
        self,
        *,
        base: ReferenceDriftSpecificEntropyDiagonalControl,
        damping: float,
    ) -> None:
        value = float(damping)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("damping must lie in [0, 1]")
        self.base = base
        self.damping = value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def __call__(self, inputs: HamiltonianControlInputs) -> torch.Tensor:
        full = self.base(inputs)
        state_dim = int(inputs.expected_adjoint_next.shape[-1])
        _, reference_sigma = self.base.reference_kernel.evaluate(
            inputs.x_prefix,
            step_index=int(inputs.step_index),
        )
        reference_sigma = reference_sigma.clamp_min(float(self.base.sigma_min))
        if self.base.sigma_max is not None:
            reference_sigma = reference_sigma.clamp_max(float(self.base.sigma_max))
        full_sigma = full[..., state_dim:]
        log_sigma = torch.log(reference_sigma) + self.damping * (
            torch.log(full_sigma) - torch.log(reference_sigma)
        )
        log_sigma = log_sigma.clamp_min(math.log(float(self.base.sigma_min)))
        if self.base.sigma_max is not None:
            log_sigma = log_sigma.clamp_max(math.log(float(self.base.sigma_max)))
        return torch.cat((full[..., :state_dim], torch.exp(log_sigma)), dim=-1)

    def moments_from_controls(
        self, *, paths: torch.Tensor, controls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.base.moments_from_controls(paths=paths, controls=controls)

    def running_cost(self, inputs: RunningCostInputs):
        return self.base.running_cost(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-scale", type=float, default=50.0)
    parser.add_argument("--kappa-scale", type=float, default=100.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--objective-paths", type=int, default=4096)
    parser.add_argument("--objective-targets", type=int, default=2048)
    parser.add_argument("--selection-banks", type=int, default=3)
    parser.add_argument("--confirmation-banks", type=int, default=3)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1e-4)
    parser.add_argument("--maximum-upper-cap-activity", type=float, default=0.01)
    parser.add_argument("--final-bank-size", type=int, default=8192)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    parser.add_argument("--final-bank-seed", type=int, default=70000)
    parser.add_argument(
        "--evaluate-all-dampings",
        action="store_true",
        help=(
            "Evaluate and retain the full common metric panel for every "
            "damping value, not only the validation-selected value."
        ),
    )
    parser.add_argument(
        "--damping-grid",
        type=float,
        nargs="+",
        default=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _antithetic_noise(
    *, model, count: int, seed: int
) -> torch.Tensor:
    if int(count) < 2 or int(count) % 2:
        raise ValueError("objective-paths must be an even integer >= 2")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    half = torch.randn(
        (int(count) // 2, int(model.grid.num_steps), int(model.architecture.noise_dim)),
        generator=generator,
        dtype=model.dtype,
    ).to(model.device)
    return torch.cat((half, -half), dim=0)


def _make_banks(
    *,
    model,
    validation_log: torch.Tensor,
    count: int,
    paths_per_bank: int,
    targets_per_bank: int,
    seed: int,
) -> list[dict[str, Any]]:
    banks: list[dict[str, Any]] = []
    for index in range(int(count)):
        bank_seed = int(seed) + 100_003 * index
        generator = torch.Generator(device="cpu").manual_seed(bank_seed + 17)
        indices = torch.randperm(
            int(validation_log.shape[0]), generator=generator
        )[: int(targets_per_bank)]
        banks.append(
            {
                "seed": bank_seed,
                "noise": _antithetic_noise(
                    model=model, count=int(paths_per_bank), seed=bank_seed + 29
                ),
                "target": validation_log.index_select(
                    0, indices.to(validation_log.device)
                ),
            }
        )
    return banks


def _law_safety(
    *, base: ReferenceDriftSpecificEntropyDiagonalControl, rollout
) -> dict[str, float]:
    controls = rollout.controls
    paths = rollout.paths
    state_dim = int(paths.shape[-1])
    sigma = controls[..., state_dim:]
    reference_drift = base._reference_drifts_all(paths)
    reference_sigma = base._reference_sigmas_all(paths)
    ratio2 = (sigma / reference_sigma).pow(2)
    transition_kl = 0.5 * (ratio2 - 1.0 - torch.log(ratio2))
    denominator = float(base.eta) / sigma
    upper = float(base.sigma_max) if base.sigma_max is not None else float("inf")
    return {
        "controls_finite": float(torch.isfinite(controls).all().item()),
        "paths_finite": float(torch.isfinite(paths).all().item()),
        "reference_drift_max_abs_error": float(
            torch.max(torch.abs(controls[..., :state_dim] - reference_drift)).item()
        ),
        "sigma_lower_bound_activity": float(
            torch.mean((sigma <= float(base.sigma_min) * (1.0 + 1e-5)).float()).item()
        ),
        "sigma_upper_bound_activity": float(
            torch.mean((sigma >= upper * (1.0 - 1e-5)).float()).item()
            if math.isfinite(upper)
            else 0.0
        ),
        "effective_entropy_denominator_min": float(denominator.min().item()),
        "transition_kl_mean": float(transition_kl.mean().item()),
        "transition_kl_q95": float(torch.quantile(transition_kl, 0.95).item()),
        "transition_kl_max": float(transition_kl.max().item()),
    }


def _evaluate_objective(
    *, model, base_control, damping: float, banks: list[dict[str, Any]], lambda_scale: float
) -> dict[str, Any]:
    model.control_map = DampedReferenceVolatilityControl(
        base=base_control, damping=float(damping)
    )
    x0 = torch.zeros((1, 1), device=model.device, dtype=model.dtype)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for bank in banks:
            noise = bank["noise"]
            rollout = model.rollout(
                x0=x0.expand(int(noise.shape[0]), -1), noise=noise
            )
            discrepancy = model.discrepancy(
                generated_paths=rollout.paths,
                target_paths=bank["target"],
                grid=model.grid,
            )
            running = base_control.running_cost(
                RunningCostInputs(
                    paths=rollout.paths,
                    controls=rollout.controls,
                    grid=model.grid,
                    parameters=model.parameters,
                )
            )
            discrepancy_value = float(lambda_scale) * float(discrepancy.value.item())
            running_value = float(running.value.item())
            rows.append(
                {
                    "seed": int(bank["seed"]),
                    "discrepancy": discrepancy_value,
                    "running_cost": running_value,
                    "total": discrepancy_value + running_value,
                    "safety": _law_safety(base=base_control, rollout=rollout),
                }
            )
    return {
        "damping": float(damping),
        "mean_total": float(np.mean([row["total"] for row in rows])),
        "rows": rows,
    }


def _comparison(
    *, reference: dict[str, Any], candidate: dict[str, Any], minimum: float,
    maximum_upper_activity: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for before, after in zip(reference["rows"], candidate["rows"], strict=True):
        relative = (float(before["total"]) - float(after["total"])) / max(
            abs(float(before["total"])), 1e-30
        )
        safety = after["safety"]
        safe = bool(
            float(safety["controls_finite"]) > 0.5
            and float(safety["paths_finite"]) > 0.5
            and float(safety["reference_drift_max_abs_error"]) <= 1e-6
            and float(safety["effective_entropy_denominator_min"]) > 0.0
            and float(safety["sigma_upper_bound_activity"])
            <= float(maximum_upper_activity)
        )
        rows.append(
            {
                "seed": int(after["seed"]),
                "before": float(before["total"]),
                "after": float(after["total"]),
                "relative_improvement": relative,
                "improved": relative >= float(minimum),
                "safe": safe,
                "safety": safety,
            }
        )
    return {
        "all_improved": all(bool(row["improved"]) for row in rows),
        "all_safe": all(bool(row["safe"]) for row in rows),
        "relative_improvement_mean": float(
            np.mean([row["relative_improvement"] for row in rows])
        ),
        "rows": rows,
    }


def _sample_final(
    *, model, base_control, damping: float, num_paths: int, batch_size: int, seed: int
) -> tuple[np.ndarray, dict[str, float]]:
    model.control_map = DampedReferenceVolatilityControl(
        base=base_control, damping=float(damping)
    )
    prices = np.empty((int(num_paths), int(model.grid.num_steps) + 1), dtype=np.float32)
    safety_rows: list[dict[str, float]] = []
    x0 = torch.zeros((1, 1), device=model.device, dtype=model.dtype)
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, int(num_paths), int(batch_size))):
            count = min(int(batch_size), int(num_paths) - start)
            rollout = model.sample(
                num_paths=count,
                x0=x0,
                seed=derive_stream_seed(
                    base_seed=int(seed), stream_offset=10_000 + batch_index
                ),
            )
            prices[start : start + count] = (
                100.0 * torch.exp(rollout.paths[..., 0])
            ).cpu().numpy().astype(np.float32)
            safety_rows.append(_law_safety(base=base_control, rollout=rollout))
    diagnostics: dict[str, float] = {
        "damping": float(damping),
        "nonfinite_path_fraction": float(1.0 - np.mean(np.isfinite(prices))),
    }
    for key in safety_rows[0]:
        values = [float(row[key]) for row in safety_rows]
        if key.endswith("_min"):
            diagnostics[key] = min(values)
        elif key.endswith("_max") or key.endswith("_error"):
            diagnostics[key] = max(values)
        else:
            diagnostics[key] = float(np.mean(values))
    return prices, diagnostics


def main() -> None:
    args = parse_args()
    damping_grid = tuple(dict.fromkeys(float(value) for value in args.damping_grid))
    if not damping_grid or any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in damping_grid
    ):
        raise ValueError("damping-grid values must lie in (0, 1]")
    if min(int(args.selection_banks), int(args.confirmation_banks)) < 1:
        raise ValueError("selection and confirmation banks must be positive")

    source_run = args.source_run.resolve()
    output_dir = ensure_run_dir(args.output_dir.resolve())
    checkpoint_path = (
        source_run / "volatility_only_online_mp" / "source_model_checkpoint.pt"
    )
    checkpoint_hash_before = _sha256(checkpoint_path)
    device = torch.device(args.device)
    train_paths = _load_log_paths(
        HESTON_ROOT / "heston_S_8192x128.npy",
        device=device,
        dtype=torch.float32,
    )
    validation_prices = np.asarray(
        np.load(HESTON_ROOT / "heston_S_disc_8192x128.npy", allow_pickle=False),
        dtype=np.float64,
    )
    validation_log = torch.as_tensor(
        np.log(validation_prices / validation_prices[:, :1])[..., None],
        device=device,
        dtype=torch.float32,
    )
    num_steps = int(train_paths.shape[1]) - 1
    grid = DiscreteTimeGrid(T=num_steps * 0.004, num_steps=num_steps)
    reference = fit_local_gaussian_reference_kernel(
        target_paths=train_paths,
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
    base_control = ReferenceDriftSpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=float(args.eta),
        reference_kernel=reference,
        sigma_min=1e-3,
        sigma_max=0.6,
        allow_drift_correction=False,
    )
    model = _model(
        grid=grid,
        control=base_control,
        discrepancy=discrepancy,
        estimator=RidgeConditionalExpectation(ridge=1e-3),
        device=device,
        seed=int(args.seed),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_checkpoint_state(checkpoint)
    # The running cost must always evaluate the physical damped controls under
    # the original control problem.
    model.running_cost = base_control.running_cost

    selection = _make_banks(
        model=model,
        validation_log=validation_log,
        count=int(args.selection_banks),
        paths_per_bank=int(args.objective_paths),
        targets_per_bank=int(args.objective_targets),
        seed=81_000 + int(args.seed),
    )
    confirmation = _make_banks(
        model=model,
        validation_log=validation_log,
        count=int(args.confirmation_banks),
        paths_per_bank=int(args.objective_paths),
        targets_per_bank=int(args.objective_targets),
        seed=91_000 + int(args.seed),
    )
    selection_reference = _evaluate_objective(
        model=model,
        base_control=base_control,
        damping=0.0,
        banks=selection,
        lambda_scale=float(args.lambda_scale),
    )
    confirmation_reference = _evaluate_objective(
        model=model,
        base_control=base_control,
        damping=0.0,
        banks=confirmation,
        lambda_scale=float(args.lambda_scale),
    )
    candidates: list[dict[str, Any]] = []
    for damping in damping_grid:
        objective = _evaluate_objective(
            model=model,
            base_control=base_control,
            damping=damping,
            banks=selection,
            lambda_scale=float(args.lambda_scale),
        )
        comparison = _comparison(
            reference=selection_reference,
            candidate=objective,
            minimum=float(args.minimum_relative_improvement),
            maximum_upper_activity=float(args.maximum_upper_cap_activity),
        )
        candidates.append(
            {
                "damping": damping,
                "selection_objective": objective,
                "selection": comparison,
            }
        )
        print(
            f"damping={damping:.8g} selection_gain="
            f"{comparison['relative_improvement_mean']:.6%} "
            f"pass={comparison['all_improved'] and comparison['all_safe']}",
            flush=True,
        )

    eligible = [
        row
        for row in candidates
        if bool(row["selection"]["all_improved"])
        and bool(row["selection"]["all_safe"])
    ]
    eligible.sort(key=lambda row: float(row["selection_objective"]["mean_total"]))
    selected: dict[str, Any] | None = None
    for row in eligible:
        objective = _evaluate_objective(
            model=model,
            base_control=base_control,
            damping=float(row["damping"]),
            banks=confirmation,
            lambda_scale=float(args.lambda_scale),
        )
        comparison = _comparison(
            reference=confirmation_reference,
            candidate=objective,
            minimum=float(args.minimum_relative_improvement),
            maximum_upper_activity=float(args.maximum_upper_cap_activity),
        )
        row["confirmation_objective"] = objective
        row["confirmation"] = comparison
        print(
            f"damping={row['damping']:.8g} confirmation_gain="
            f"{comparison['relative_improvement_mean']:.6%} "
            f"pass={comparison['all_improved'] and comparison['all_safe']}",
            flush=True,
        )
        if bool(comparison["all_improved"]) and bool(comparison["all_safe"]):
            selected = row
            break
    if selected is None:
        write_json(
            output_dir / "SUMMARY.json",
            {
                "status": "no_damping_passed_independent_objective_and_safety",
                "candidates": candidates,
            },
        )
        raise RuntimeError("no damping passed independent objective and safety")

    selected_damping = float(selected["damping"])
    final_bank, final_controls = _sample_final(
        model=model,
        base_control=base_control,
        damping=selected_damping,
        num_paths=int(args.final_bank_size),
        batch_size=int(args.sample_batch_size),
        seed=int(args.final_bank_seed) + int(args.seed),
    )
    np.save(output_dir / "validation_bank.npy", final_bank)
    metrics = _evaluate_common(
        target_prices=validation_prices,
        generated=final_bank,
        device=device,
        seed=int(args.seed),
    )
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "control_diagnostics.json", final_controls)
    checkpoint_hash_after = _sha256(checkpoint_path)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("the online MP checkpoint changed during damping selection")
    candidate_evaluations: dict[str, dict[str, Any]] = {}
    if bool(args.evaluate_all_dampings):
        for damping in damping_grid:
            label = f"damping_{damping:.10f}".rstrip("0").rstrip(".").replace(".", "p")
            candidate_dir = ensure_run_dir(output_dir / label)
            if damping == selected_damping:
                candidate_bank = final_bank
                candidate_controls = final_controls
                candidate_metrics = metrics
            else:
                candidate_bank, candidate_controls = _sample_final(
                    model=model,
                    base_control=base_control,
                    damping=damping,
                    num_paths=int(args.final_bank_size),
                    batch_size=int(args.sample_batch_size),
                    seed=int(args.final_bank_seed) + int(args.seed),
                )
                candidate_metrics = _evaluate_common(
                    target_prices=validation_prices,
                    generated=candidate_bank,
                    device=device,
                    seed=int(args.seed),
                )
            np.save(candidate_dir / "validation_bank.npy", candidate_bank)
            write_json(candidate_dir / "metrics.json", candidate_metrics)
            write_json(candidate_dir / "control_diagnostics.json", candidate_controls)
            candidate_evaluations[f"{damping:.10g}"] = {
                "damping": damping,
                "directory": str(candidate_dir),
                "metrics": candidate_metrics,
                "control_diagnostics": candidate_controls,
            }
            print(f"evaluated full metric panel for damping={damping:.8g}", flush=True)

    protocol = {
        "method": "original_online_mp_with_scalar_log_volatility_damping",
        "training_performed": False,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_hash_before,
        "conditional_target_estimator": "projected_score_then_raw_prefix_ridge",
        "adjoint_loss_changed": False,
        "online_training_trajectory_changed": False,
        "physical_drift": "fitted_reference_fixed",
        "controlled_coordinate": "volatility_only",
        "compression_used": False,
        "recursive_policy_used": False,
        "damping_coordinate": "log_volatility_between_reference_and_online_mp",
        "damping_grid": list(damping_grid),
        "selection_banks": int(args.selection_banks),
        "confirmation_banks": int(args.confirmation_banks),
        "objective_paths_per_bank": int(args.objective_paths),
        "objective_targets_per_bank": int(args.objective_targets),
        "lambda_scale": float(args.lambda_scale),
        "kappa_scale": float(args.kappa_scale),
        "eta": float(args.eta),
        "test_split_loaded": False,
        "checkpoint_nonmutation": True,
        "all_dampings_evaluated": bool(args.evaluate_all_dampings),
    }
    write_json(output_dir / "LOCKED_PROTOCOL.json", protocol)
    write_json(
        output_dir / "SUMMARY.json",
        {
            "status": "completed",
            "selected_damping": selected_damping,
            "selection": selected["selection"],
            "confirmation": selected["confirmation"],
            "candidates": candidates,
            "candidate_evaluations": candidate_evaluations,
            "control_diagnostics": final_controls,
            "metrics_path": str(output_dir / "metrics.json"),
            "protocol": protocol,
        },
    )
    print(
        f"selected damping={selected_damping:.8g}; wrote {output_dir / 'SUMMARY.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
