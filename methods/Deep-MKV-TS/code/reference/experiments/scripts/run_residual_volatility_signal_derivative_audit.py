#!/usr/bin/env python3
"""Run the locked signal and derivative gate for residual conditional volatility."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteTimeGrid,
    fit_causal_volatility_reference_kernel,
)
from deep_mkv_gen_path_dt.discrepancies.mmd import rbf_mmd2  # noqa: E402
from deep_mkv_gen_path_dt.noise import sample_standard_normals  # noqa: E402
from path_dt_experiments.residual_volatility_objective import (  # noqa: E402
    FixedReferenceResidualConditionalVolatilityMMD,
    fit_fixed_reference_residual_conditional_volatility_mmd,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402


DEFAULT_DATASET = (
    REPO_ROOT / "runs" / "drawdown_memory_seed0_comparison_20260730" / "data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--audit-paths", type=int, default=4096)
    parser.add_argument("--mmd-partition-size", type=int, default=512)
    parser.add_argument("--floor-fraction", type=float, default=1e-3)
    parser.add_argument("--signal-ratio", type=float, default=2.0)
    parser.add_argument("--trigger-effect", type=float, default=0.25)
    parser.add_argument("--fd-median-tolerance", type=float, default=1e-3)
    parser.add_argument("--fd-maximum-tolerance", type=float, default=1e-2)
    parser.add_argument("--lions-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def _load_training_paths(
    root: Path, *, device: torch.device
) -> tuple[torch.Tensor, dict[str, object]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    prices = np.asarray(np.load(root / "train.npy"), dtype=np.float64)
    log_paths = np.log(prices / prices[:, :1])
    return (
        torch.from_numpy(log_paths).unsqueeze(-1).to(
            device=device, dtype=torch.float32
        ),
        manifest,
    )


def _drawdown_thresholds(
    paths: torch.Tensor, *, cutoff: int
) -> tuple[float, ...]:
    prefix = paths[:, : int(cutoff) + 1, 0]
    drawdown = torch.cummax(prefix, dim=1).values - prefix
    values = drawdown.amax(dim=1)
    quantiles = values.new_tensor((0.35, 0.60, 0.80))
    return tuple(float(value) for value in torch.quantile(values, quantiles).cpu())


def _fit_reference(
    paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    task_aware: bool,
    thresholds: tuple[float, ...],
    seed: int,
):
    return fit_causal_volatility_reference_kernel(
        target_paths=paths,
        grid=grid,
        rolling_windows=(5, 10, 20, 32),
        ewma_half_lives=(5, 20, 60),
        downside_windows=(10, 32),
        drawdown_thresholds=thresholds if task_aware else (),
        drawdown_memory_half_lives=(20, 60) if task_aware else (),
        drawdown_memory_lags=(8, 16, 24, 32, 40, 48) if task_aware else (),
        drawdown_gate_temperature=0.005,
        drift_ridge=1e-3,
        variance_ridges=(1e-5, 1e-4, 1e-3),
        crossfit_folds=3,
        crossfit_seed=int(seed),
        sigma_min=1e-3,
        sigma_max=0.6,
    )


def _simulate_reference(
    *,
    reference: object,
    grid: DiscreteTimeGrid,
    num_paths: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    noise = sample_standard_normals(
        grid=grid,
        batch_size=int(num_paths),
        noise_dim=1,
        device=device,
        dtype=torch.float32,
        seed=int(seed),
    )
    points = [torch.zeros(int(num_paths), 1, device=device)]
    for step in range(int(grid.num_steps)):
        prefix = torch.stack(points, dim=1)
        _, sigma = reference.evaluate(prefix, step_index=step)
        points.append(
            points[-1]
            + math.sqrt(float(grid.dt)) * sigma * noise[:, step, :]
        )
    return torch.stack(points, dim=1).detach()


def _partition_mmd(
    discrepancy: FixedReferenceResidualConditionalVolatilityMMD,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    size: int,
) -> dict[str, object]:
    count = min(int(left.shape[0]), int(right.shape[0])) // int(size)
    if count < 2:
        raise ValueError("signal audit requires at least two MMD partitions")
    values = []
    for index in range(count):
        l = left[index * size : (index + 1) * size]
        r = right[index * size : (index + 1) * size]
        values.append(
            float(
                discrepancy(
                    generated_paths=l,
                    target_paths=r,
                    grid=grid,
                ).value.detach().item()
            )
        )
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "values": values,
        "median": float(torch.median(tensor).item()),
        "mean": float(tensor.mean().item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
        "partitions": count,
        "partition_size": int(size),
    }


def _trigger_report(
    discrepancy: FixedReferenceResidualConditionalVolatilityMMD,
    paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    history_cutoff: int,
    trigger_threshold: float,
) -> dict[str, float]:
    prefix = paths[:, : int(history_cutoff) + 1, 0]
    drawdown = torch.cummax(prefix, dim=1).values - prefix
    triggered = drawdown.amax(dim=1) >= float(trigger_threshold)
    if int(triggered.sum().item()) < 2 or int((~triggered).sum().item()) < 2:
        raise RuntimeError("trigger audit requires both triggered and control paths")
    features = discrepancy.features(paths, grid=grid).detach().double()
    prefix_features = features[:, :-1]

    def effect(values: torch.Tensor) -> torch.Tensor:
        difference = values[triggered].mean(dim=0) - values[~triggered].mean(dim=0)
        scale = values.std(dim=0, unbiased=False).clamp_min(
            torch.finfo(values.dtype).eps
        )
        return difference / scale

    prefix_effect = effect(prefix_features).abs()
    future_effect = effect(features[:, -1:]).abs()
    return {
        "triggered_fraction": float(triggered.double().mean().item()),
        "maximum_prefix_trigger_effect": float(prefix_effect.max().item()),
        "future_trigger_effect": float(future_effect.max().item()),
        "triggered_future_mean": float(features[triggered, -1].mean().item()),
        "control_future_mean": float(features[~triggered, -1].mean().item()),
    }


def _reference_parameters_frozen(reference: object) -> bool:
    if not is_dataclass(reference):
        return False
    for field in fields(reference):
        value = getattr(reference, field.name)
        if isinstance(value, torch.Tensor) and bool(value.requires_grad):
            return False
    return True


def _derivative_audit(
    discrepancy: FixedReferenceResidualConditionalVolatilityMMD,
    paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sample = paths[:8].detach().double()
    direction = torch.randn(sample.shape, generator=generator, dtype=torch.float64).to(
        sample.device
    )
    direction = direction / direction.pow(2).mean().sqrt()
    full = sample.clone().requires_grad_(True)
    features = discrepancy.raw_features(full, grid=grid)
    fd_rows = []
    step = 1e-5
    for coordinate in range(int(features.shape[1])):
        scalar = features[:, coordinate].mean()
        gradient = torch.autograd.grad(
            scalar, full, retain_graph=coordinate + 1 < int(features.shape[1])
        )[0]
        autodiff = torch.sum(gradient * direction)
        plus = discrepancy.raw_features(
            sample + step * direction, grid=grid
        )[:, coordinate].mean()
        minus = discrepancy.raw_features(
            sample - step * direction, grid=grid
        )[:, coordinate].mean()
        finite = (plus - minus) / (2.0 * step)
        relative = float(
            (autodiff - finite).abs().item()
            / max(
                float(autodiff.abs().item()),
                float(finite.abs().item()),
                1e-8,
            )
        )
        fd_rows.append(
            {
                "coordinate": coordinate,
                "autodiff": float(autodiff.item()),
                "finite_difference": float(finite.item()),
                "relative_error": relative,
            }
        )
    fd_errors = torch.tensor(
        [row["relative_error"] for row in fd_rows], dtype=torch.float64
    )

    full_reference = sample.clone().requires_grad_(True)
    full_scalar = discrepancy.raw_features(full_reference, grid=grid).sum()
    full_gradient = torch.autograd.grad(full_scalar, full_reference)[0]
    detached_reference = sample.clone().requires_grad_(True)
    detached_scalar = discrepancy.raw_features(
        detached_reference, grid=grid, detach_reference_outputs=True
    ).sum()
    detached_gradient = torch.autograd.grad(
        detached_scalar, detached_reference
    )[0]
    reference_derivative_fraction = float(
        (full_gradient - detached_gradient).norm().item()
        / max(float(full_gradient.norm().item()), torch.finfo(torch.float64).eps)
    )

    generated = paths[8:72].detach().clone().requires_grad_(True)
    target = paths[72:136].detach().clone().requires_grad_(True)
    direct_value = discrepancy(
        generated_paths=generated, target_paths=target, grid=grid
    ).value
    direct_source = float(generated.shape[0]) * torch.autograd.grad(
        direct_value, generated
    )[0]
    query = generated.detach().clone().requires_grad_(True)
    target_potential = target.detach().clone().requires_grad_(True)
    potential = discrepancy.lions_potential(
        query_paths=query,
        population_paths=generated.detach(),
        target_paths=target_potential,
        grid=grid,
    )
    query_gradient, target_gradient = torch.autograd.grad(
        potential, (query, target_potential), allow_unused=True
    )
    lions_source = float(query.shape[0]) * query_gradient
    lions_relative = float(
        (direct_source - lions_source).norm().item()
        / max(float(direct_source.norm().item()), torch.finfo(torch.float64).eps)
    )

    identical = paths[136:200].detach().clone().requires_grad_(True)
    identical_value = discrepancy(
        generated_paths=identical,
        target_paths=identical.detach().clone(),
        grid=grid,
    ).value
    identical_source = float(identical.shape[0]) * torch.autograd.grad(
        identical_value, identical
    )[0]
    return {
        "finite_difference": {
            "rows": fd_rows,
            "median_relative_error": float(torch.median(fd_errors).item()),
            "maximum_relative_error": float(fd_errors.max().item()),
            "step": step,
        },
        "generated_reference_derivative": {
            "included": True,
            "relative_gradient_contribution": reference_derivative_fraction,
        },
        "target_paths_detached": target_gradient is None,
        "reference_parameters_frozen": _reference_parameters_frozen(
            discrepancy.reference_kernel
        ),
        "lions_vs_direct": {
            "relative_source_error": lions_relative,
            "direct_source_rms": float(direct_source.pow(2).mean().sqrt().item()),
            "lions_source_rms": float(lions_source.pow(2).mean().sqrt().item()),
        },
        "identical_law": {
            "mmd_value": float(identical_value.detach().item()),
            "source_rms": float(identical_source.pow(2).mean().sqrt().item()),
        },
    }


def _signal_for_reference(
    *,
    name: str,
    reference: object,
    calibration: torch.Tensor,
    audit: torch.Tensor,
    generated: torch.Tensor,
    grid: DiscreteTimeGrid,
    split_index: int,
    prefix_window: int,
    future_horizon: int,
    floor_fraction: float,
    partition_size: int,
    history_cutoff: int,
    trigger_threshold: float,
) -> tuple[dict[str, object], dict[bool, FixedReferenceResidualConditionalVolatilityMMD]]:
    discrepancies = {
        include_drawdown: fit_fixed_reference_residual_conditional_volatility_mmd(
            calibration,
            reference_kernel=reference,
            grid=grid,
            split_index=split_index,
            prefix_window=prefix_window,
            future_horizon=future_horizon,
            include_drawdown=include_drawdown,
            floor_fraction=floor_fraction,
        )
        for include_drawdown in (False, True)
    }
    report = {"reference": name, "specifications": {}}
    for include_drawdown, discrepancy in discrepancies.items():
        label = "drawdown_augmented" if include_drawdown else "two_dimensional"
        floor = _partition_mmd(
            discrepancy, calibration, audit, grid=grid, size=partition_size
        )
        reference_target = _partition_mmd(
            discrepancy, generated, audit, grid=grid, size=partition_size
        )
        ratio = float(reference_target["median"]) / max(
            float(floor["median"]), torch.finfo(torch.float64).eps
        )
        report["specifications"][label] = {
            "design": discrepancy.summary(),
            "independent_target_floor": floor,
            "reference_target": reference_target,
            "reference_to_floor_ratio": ratio,
            "trigger": _trigger_report(
                discrepancy,
                audit,
                grid=grid,
                history_cutoff=history_cutoff,
                trigger_threshold=trigger_threshold,
            ),
        }
    return report, discrepancies


def _summary_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Residual conditional-volatility signal and derivative audit",
        "",
        f"- Selected specification: `{report['selection']['selected_specification']}`",
        f"- Signal gate passed: `{report['selection']['signal_gate_passed']}`",
        f"- Derivative gate passed: `{report['derivative_gate']['passed']}`",
        f"- Next stage: `{report['next_stage']}`",
        "",
        "## Frozen residual-feature mismatch and trigger association",
        "",
        "| Reference | Specification | Target floor | Reference-target | Ratio | Prefix trigger effect | Future trigger effect |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for reference in ("generic", "task_aware"):
        for spec, row in report["signal"][reference]["specifications"].items():
            lines.append(
                f"| {reference} | {spec} | "
                f"{row['independent_target_floor']['median']:.5g} | "
                f"{row['reference_target']['median']:.5g} | "
                f"{row['reference_to_floor_ratio']:.2f} | "
                f"{row['trigger']['maximum_prefix_trigger_effect']:.3f} | "
                f"{row['trigger']['future_trigger_effect']:.3f} |"
            )
    derivative = report["derivative"]
    lines.extend(
        [
            "",
            "## Complete derivative",
            "",
            f"- Median/max finite-difference relative error: "
            f"`{derivative['finite_difference']['median_relative_error']:.3e}` / "
            f"`{derivative['finite_difference']['maximum_relative_error']:.3e}`.",
            f"- Lions/direct relative source error: "
            f"`{derivative['lions_vs_direct']['relative_source_error']:.3e}`.",
            f"- Reference-output gradient contribution: "
            f"`{derivative['generated_reference_derivative']['relative_gradient_contribution']:.3e}`.",
            f"- Identical-law MMD/source RMS: "
            f"`{derivative['identical_law']['mmd_value']:.3e}` / "
            f"`{derivative['identical_law']['source_rms']:.3e}`.",
            f"- Target detached / reference frozen: "
            f"`{derivative['target_paths_detached']}` / "
            f"`{derivative['reference_parameters_frozen']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    device = torch.device(args.device)
    paths, manifest = _load_training_paths(args.dataset_root.resolve(), device=device)
    configuration = manifest["configuration"]
    calibration_count = int(args.calibration_paths)
    audit_count = int(args.audit_paths)
    if calibration_count + audit_count > int(paths.shape[0]):
        raise ValueError("calibration and audit partitions exceed the training law")
    calibration = paths[:calibration_count]
    audit = paths[calibration_count : calibration_count + audit_count]
    grid = DiscreteTimeGrid(
        T=(int(configuration["sequence_length"]) - 1) * float(configuration["dt"]),
        num_steps=int(configuration["sequence_length"]) - 1,
    )
    thresholds = _drawdown_thresholds(calibration, cutoff=48)
    print("fitting generic causal reference", flush=True)
    generic = _fit_reference(
        calibration,
        grid=grid,
        task_aware=False,
        thresholds=thresholds,
        seed=1701 + int(args.seed),
    )
    print("fitting task-aware oracle reference", flush=True)
    task_aware = _fit_reference(
        calibration,
        grid=grid,
        task_aware=True,
        thresholds=thresholds,
        seed=1801 + int(args.seed),
    )
    with torch.no_grad():
        print("simulating reference laws", flush=True)
        generic_paths = _simulate_reference(
            reference=generic,
            grid=grid,
            num_paths=audit_count,
            seed=2608201 + int(args.seed),
            device=device,
        )
        task_aware_paths = _simulate_reference(
            reference=task_aware,
            grid=grid,
            num_paths=audit_count,
            seed=2608201 + int(args.seed),
            device=device,
        )
    signal = {}
    fitted = {}
    for name, reference, generated in (
        ("generic", generic, generic_paths),
        ("task_aware", task_aware, task_aware_paths),
    ):
        print(f"auditing {name} residual signal", flush=True)
        signal[name], fitted[name] = _signal_for_reference(
            name=name,
            reference=reference,
            calibration=calibration,
            audit=audit,
            generated=generated,
            grid=grid,
            split_index=int(configuration["split_index"]),
            prefix_window=int(configuration["future_horizon"]),
            future_horizon=int(configuration["future_horizon"]),
            floor_fraction=float(args.floor_fraction),
            partition_size=int(args.mmd_partition_size),
            history_cutoff=int(configuration["history_cutoff"]),
            trigger_threshold=float(configuration["drawdown_threshold"]),
        )
    primary = signal["generic"]["specifications"]["two_dimensional"]
    augmented = signal["generic"]["specifications"]["drawdown_augmented"]

    def trigger_pass(row: dict[str, object]) -> bool:
        return bool(
            float(row["trigger"]["maximum_prefix_trigger_effect"])
            >= float(args.trigger_effect)
            and float(row["trigger"]["future_trigger_effect"])
            >= float(args.trigger_effect)
        )

    if trigger_pass(primary):
        selected_label = "two_dimensional"
        selected = fitted["generic"][False]
        selected_signal = primary
        fallback_used = False
    elif trigger_pass(augmented):
        selected_label = "drawdown_augmented"
        selected = fitted["generic"][True]
        selected_signal = augmented
        fallback_used = True
    else:
        selected_label = None
        selected = fitted["generic"][True]
        selected_signal = augmented
        fallback_used = True
    signal_gate = bool(
        selected_label is not None
        and float(selected_signal["reference_to_floor_ratio"])
        >= float(args.signal_ratio)
    )
    print("running complete derivative audit", flush=True)
    derivative = _derivative_audit(
        selected,
        audit,
        grid=grid,
        seed=2_608_501 + int(args.seed),
    )
    derivative_checks = {
        "finite_difference_median": float(
            derivative["finite_difference"]["median_relative_error"]
        ) <= float(args.fd_median_tolerance),
        "finite_difference_maximum": float(
            derivative["finite_difference"]["maximum_relative_error"]
        ) <= float(args.fd_maximum_tolerance),
        "lions_direct": float(
            derivative["lions_vs_direct"]["relative_source_error"]
        ) <= float(args.lions_tolerance),
        "target_detached": bool(derivative["target_paths_detached"]),
        "reference_parameters_frozen": bool(
            derivative["reference_parameters_frozen"]
        ),
        "reference_derivative_included": bool(
            derivative["generated_reference_derivative"][
                "relative_gradient_contribution"
            ]
            > 0.0
        ),
        "identical_mmd_near_zero": abs(
            float(derivative["identical_law"]["mmd_value"])
        ) <= 1e-6,
        "identical_source_near_zero": float(
            derivative["identical_law"]["source_rms"]
        ) <= 1e-5,
    }
    derivative_gate = {"passed": all(derivative_checks.values()), "checks": derivative_checks}
    next_stage = (
        "first_operator_control_active_set_audit"
        if signal_gate and derivative_gate["passed"]
        else "stop_residual_objective_experiment"
    )
    report = {
        "protocol": {
            "train_only": True,
            "calibration_paths": calibration_count,
            "audit_paths": audit_count,
            "mmd_partition_size": int(args.mmd_partition_size),
            "floor_fraction": float(args.floor_fraction),
            "bandwidths_train_fitted": True,
            "normalization_train_fitted": True,
            "task_aware_reference_oracle_diagnostic_only": True,
            "zero_adjoint_reference_drift": 0.0,
            "feature_innovations_remove_fitted_reference_drift": True,
            "signal_interpretation": (
                "residual-feature mismatch and trigger association; "
                "joint dependence is not isolated from marginal mismatch"
            ),
            "fallback_locked_before_results": True,
        },
        "dataset": str(args.dataset_root.resolve()),
        "thresholds": list(thresholds),
        "signal": signal,
        "selection": {
            "selected_specification": selected_label,
            "fallback_used": fallback_used,
            "signal_gate_passed": signal_gate,
            "required_reference_to_floor_ratio": float(args.signal_ratio),
            "required_trigger_effect": float(args.trigger_effect),
        },
        "derivative": derivative,
        "derivative_gate": derivative_gate,
        "next_stage": next_stage,
    }
    write_json(run_dir / "signal_derivative_report.json", report)
    torch.save(
        {
            "selected_discrepancy": selected,
            "generic_reference": generic,
            "task_aware_reference": task_aware,
            "calibration_indices": torch.arange(calibration_count),
            "audit_indices": torch.arange(calibration_count, calibration_count + audit_count),
            "protocol": report["protocol"],
            "selection": report["selection"],
        },
        run_dir / "selected_residual_objective.pt",
    )
    (run_dir / "SUMMARY.md").write_text(_summary_markdown(report), encoding="utf-8")
    print(f"wrote residual signal/derivative audit to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
