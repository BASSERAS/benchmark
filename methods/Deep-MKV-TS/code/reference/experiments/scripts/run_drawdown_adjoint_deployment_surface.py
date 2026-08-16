#!/usr/bin/env python3
"""Audit independent P/R deployment for a frozen drawdown-memory MP solve."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    CausalPathFeatureRidgeConditionalExpectation,
    DiscreteMPNestedTrainingConfig,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
    fit_causal_volatility_reference_kernel,
)
from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl  # noqa: E402
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from path_dt_experiments.discrepancies import (  # noqa: E402
    build_heston_discrepancy,
    fit_fixed_target_joint_prefix_future_volatility_mmd,
)
from path_dt_experiments.nested_directional import (  # noqa: E402
    AdjointControlVariant,
    DEFAULT_MOMENT_RELAXATIONS,
    diagnose_first_nested_direction,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_drawdown_memory_deep_mkv_ablation import (  # noqa: E402
    _build_model,
    _load_dataset,
)


DEFAULT_DATA = (
    REPO_ROOT / "runs" / "drawdown_memory_seed0_comparison_20260730" / "data"
)
DEFAULT_MODEL = (
    REPO_ROOT
    / "runs"
    / "paper_drawdown_fully_generic_4seed_20260801"
    / "seed_0"
    / "generic_ce_nested"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-run-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=73031)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--inner-steps", type=int, default=200)
    parser.add_argument("--outer-batch-size", type=int, default=512)
    parser.add_argument("--outer-target-batch-size", type=int, default=512)
    parser.add_argument("--inner-batch-size", type=int, default=256)
    parser.add_argument(
        "--target-estimator",
        choices=("projected_score", "nested_branches"),
        default="projected_score",
    )
    parser.add_argument("--conditional-branches", type=int, default=64)
    parser.add_argument("--population-batch-size", type=int, default=1024)
    parser.add_argument("--evaluation-batch-size", type=int, default=None)
    parser.add_argument(
        "--moment-relaxations",
        type=float,
        nargs="+",
        default=DEFAULT_MOMENT_RELAXATIONS,
    )
    return parser.parse_args()


def _plot_surface(report: dict[str, object], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = list(report["closed_loop_moment_surface"]["rows"])  # type: ignore[index]
    rates = sorted({float(row["p_relaxation"]) for row in rows})
    lookup = {
        (float(row["p_relaxation"]), float(row["r_relaxation"])): row
        for row in rows
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (metric, title) in zip(
        axes,
        (
            ("complete_objective_value", "Complete objective"),
            ("log_sigma_rms_change", "RMS change in log volatility"),
            ("sigma_cap_fraction", "Volatility-cap activity"),
        ),
    ):
        values = np.asarray(
            [
                [float(lookup[(p_rate, r_rate)].get(metric, np.nan)) for r_rate in rates]
                for p_rate in rates
            ],
            dtype=float,
        )
        image = axis.imshow(values, origin="lower", aspect="auto")
        axis.set_xticks(range(len(rates)), [f"{value:g}" for value in rates], rotation=60)
        axis.set_yticks(range(len(rates)), [f"{value:g}" for value in rates])
        axis.set_xlabel(r"$\tau_R$")
        axis.set_ylabel(r"$\tau_P$")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.85)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    device = torch.device(args.device)
    dtype = torch.float32
    target_paths, dataset_manifest, dataset_paths = _load_dataset(
        args.dataset_root.resolve(),
        device=device,
        dtype=dtype,
    )
    configuration = dataset_manifest["configuration"]
    sequence_length = int(configuration["sequence_length"])
    num_steps = sequence_length - 1
    grid = DiscreteTimeGrid(
        T=num_steps * float(configuration["dt"]),
        num_steps=num_steps,
    )
    reference = fit_causal_volatility_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        rolling_windows=(5, 10, 20, 32),
        ewma_half_lives=(5, 20, 60),
        downside_windows=(10, 32),
        drawdown_thresholds=(),
        drawdown_memory_half_lives=(),
        drawdown_memory_lags=(),
        drawdown_gate_temperature=0.005,
        drift_ridge=1e-3,
        variance_ridges=(1e-5, 1e-4, 1e-3),
        crossfit_folds=3,
        crossfit_seed=1701 + int(args.model_seed),
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    control = SpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=1.0,
        reference_kernel=reference,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    ce_estimator = CausalPathFeatureRidgeConditionalExpectation(
        grid=grid,
        ridge=1e-3,
        rolling_windows=(5, 10, 20, 32),
        ewma_half_lives=(5, 20, 60),
        downside_windows=(10, 32),
        drawdown_thresholds=(),
        drawdown_memory_half_lives=(),
        drawdown_memory_lags=(),
        drawdown_gate_temperature=0.005,
    )
    base = build_heston_discrepancy(
        num_steps=num_steps,
        include_acf=True,
        preset="old_fullv_w0p25",
        lambda_scale=50.0,
        kappa_scale=50.0,
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )
    joint = fit_fixed_target_joint_prefix_future_volatility_mmd(
        target_paths,
        grid=grid,
        split_index=int(configuration["split_index"]),
        future_horizon=int(configuration["future_horizon"]),
        prefix_windows=(16, 32, 64),
        recent_return_window=16,
        bandwidths=(0.5, 1.0, 2.0),
    )
    discrepancy = CompositePathFunctionalDiscrepancy(
        blocks=tuple(base.blocks)
        + (
            WeightedDiscrepancy(
                name="volatility_law_joint_prefix_future_rv",
                weight=1.0,
                discrepancy=joint,
            ),
        )
    )
    model = _build_model(
        args=argparse.Namespace(hidden_dim=96, num_layers=1),
        grid=grid,
        control=control,
        discrepancy=discrepancy,
        ce_estimator=ce_estimator,
        device=device,
        dtype=dtype,
        seed=int(args.model_seed),
    )
    model_run_dir = args.model_run_dir.resolve()
    model.load_checkpoint_state(
        torch.load(
            model_run_dir / "model_checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    manifest = json.loads(
        (model_run_dir / "training_manifest.json").read_text(encoding="utf-8")
    )
    training = replace(
        DiscreteMPTrainingConfig(**manifest["joint_training"]),
        grad_clip_norm=None,
        noise_target_control_variate="adjoint",
    )
    nested = DiscreteMPNestedTrainingConfig(
        outer_steps=1,
        inner_steps=int(args.inner_steps),
        outer_batch_size=int(args.outer_batch_size),
        outer_target_batch_size=int(args.outer_target_batch_size),
        outer_target_estimator=str(args.target_estimator),
        outer_conditional_branches=int(args.conditional_branches),
        outer_conditional_antithetic=True,
        outer_population_batch_size=int(args.population_batch_size),
        inner_batch_size=int(args.inner_batch_size),
        fixed_point_probe_paths=min(512, int(args.outer_batch_size)),
        outer_objective_line_search=False,
        seed=int(args.seed),
    )
    variants = (
        AdjointControlVariant("correct", 1.0, 1.0, "implemented MP"),
        AdjointControlVariant("p_only", 1.0, 0.0, "suppress R"),
        AdjointControlVariant("r_only", 0.0, 1.0, "suppress P"),
    )
    print("fitting one frozen candidate and evaluating the P/R surface", flush=True)
    report, artifacts = diagnose_first_nested_direction(
        model=model,
        target_paths=target_paths,
        training=training,
        nested=nested,
        relaxations=(
            -0.015625,
            -0.0078125,
            -0.00390625,
            -0.001953125,
            -0.0009765625,
            0.0,
            0.0009765625,
            0.001953125,
            0.00390625,
            0.0078125,
            0.015625,
            0.03125,
            0.0625,
            0.125,
            0.25,
            0.5,
            1.0,
        ),
        moment_relaxations=tuple(float(value) for value in args.moment_relaxations),
        evaluation_batch_size=args.evaluation_batch_size,
        variants=variants,
    )
    report["run_dir"] = str(run_dir)
    report["model_run_dir"] = str(model_run_dir)
    report["dataset_sources"] = {
        name: str(path.resolve()) for name, path in dataset_paths.items()
    }
    write_json(run_dir / "metrics.json", report)
    torch.save(artifacts, run_dir / "diagnostic_artifacts.pt")
    _plot_surface(report, run_dir / "closed_loop_surface.png")
    best = report["closed_loop_moment_surface"]["best_row"]  # type: ignore[index]
    print(
        "best closed-loop moment relaxation "
        f"P={best['p_relaxation']:g} R={best['r_relaxation']:g} "
        f"objective={best['complete_objective_value']:.6g} "
        f"log-sigma RMS={best['log_sigma_rms_change']:.6g}",
        flush=True,
    )
    print(f"wrote deployment audit to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
