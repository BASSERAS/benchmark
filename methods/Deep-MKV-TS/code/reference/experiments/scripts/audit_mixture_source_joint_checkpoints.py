#!/usr/bin/env python3
"""Validation-only audit of saved mixture source and joint MP checkpoints.

The audit performs no training and never mutates the source campaign.  Each
checkpoint is replayed with the same validation noise used by the final
campaign so that source-versus-joint differences reflect the saved policies,
not different Monte Carlo draws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

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
from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    VolatilityOnlySpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from path_dt_experiments.discrepancies import (  # noqa: E402
    build_heston_discrepancy,
    fit_fixed_target_joint_prefix_future_volatility_mmd,
)
from path_dt_experiments.heston_mixture_discrepancy import (  # noqa: E402
    HestonMixtureCausalRidgeConditionalExpectation,
    fit_fixed_target_heston_mixture_summary_mmd,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_heston_mixture_deep_mkv_ablation import _load_log_paths  # noqa: E402
from run_matched_control_synthetic_validation import (  # noqa: E402
    _evaluate_common,
    _model,
    _sample,
    _write_mixture_metrics,
)


DEFAULT_CAMPAIGN = (
    REPO_ROOT
    / "runs"
    / "final_nested_volatility_only_validation_20260804"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "mixture_source_joint_checkpoint_audit_20260804"
MIXTURE_ROOT = REPO_ROOT / "runs" / "heston_parameter_mixture_seed0_20260730"
SEEDS = (0, 1, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected an object in {path}")
    return payload


def _error_groups(metrics: dict[str, object]) -> dict[str, dict[str, float]]:
    path = metrics["path_law"]
    memory = metrics["conditional_path_memory"]
    generated = memory["generated"]
    target = memory["target"]
    return {
        "distribution": {
            key: float(path[key])
            for key in (
                "path_swd_normalized",
                "terminal_w1_normalized",
                "increment_w1_normalized",
                "realized_volatility_w1_normalized",
                "maximum_drawdown_w1_normalized",
                "return_qq_rmse_normalized",
            )
        },
        "dependence": {
            key: float(path[key])
            for key in (
                "return_acf_rmse",
                "absolute_return_acf_rmse",
                "squared_return_acf_rmse",
            )
        },
        "conditional_memory": {
            "prefix_future_rv_correlation_error": abs(
                float(generated["prefix_future_rv_correlation"])
                - float(target["prefix_future_rv_correlation"])
            ),
            "prefix_high_low_future_rv_gap_error": abs(
                float(generated["prefix_high_low_future_rv_gap"])
                - float(target["prefix_high_low_future_rv_gap"])
            ),
        },
    }


def _gap_summary(
    reference: dict[str, float], candidate: dict[str, float]
) -> dict[str, object]:
    gaps = {
        key: (float(reference[key]) - float(candidate[key])) / float(reference[key])
        for key in reference
        if math.isfinite(float(reference[key]))
        and float(reference[key]) > 1e-12
        and math.isfinite(float(candidate[key]))
    }
    values = list(gaps.values())
    return {
        "metric_count": len(values),
        "improved_count": sum(value > 0.0 for value in values),
        "median_gap_closed": float(np.median(values)) if values else None,
        "per_metric_gap_closed": gaps,
    }


def _safety_gate(
    metrics: dict[str, object], controls: dict[str, object]
) -> dict[str, object]:
    scale = metrics["full_bank_scale_tail"]
    target_kurtosis = float(scale["target_excess_kurtosis"])
    checks = {
        "finite_paths_and_controls": max(
            float(scale["nonfinite_path_fraction"]),
            float(controls["nonfinite_path_fraction"]),
            float(controls["nonfinite_control_fraction"]),
        )
        == 0.0,
        "upper_cap_activity_at_most_1pct": (
            float(controls["sigma_max_active_fraction"]) <= 0.01
        ),
        "return_std_ratio_0p8_to_1p2": (
            0.8 <= float(scale["return_std_ratio"]) <= 1.2
        ),
        "rv_mean_ratio_0p8_to_1p2": (
            0.8 <= float(scale["realized_volatility_mean_ratio"]) <= 1.2
        ),
        "rv_std_ratio_0p5_to_1p5": (
            0.5 <= float(scale["realized_volatility_std_ratio"]) <= 1.5
        ),
        "absolute_return_q99_ratio_0p75_to_1p25": (
            0.75 <= float(scale["absolute_return_q99_ratio"]) <= 1.25
        ),
        "absolute_return_q999_ratio_0p75_to_1p25": (
            0.75 <= float(scale["absolute_return_q999_ratio"]) <= 1.25
        ),
        "excess_kurtosis_at_most_1p5x_target": (
            float(scale["generated_excess_kurtosis"])
            <= 1.5 * max(target_kurtosis, 0.0)
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "values": {
            "sigma_max_active_fraction": float(
                controls["sigma_max_active_fraction"]
            ),
            "specific_entropy_denominator_min": float(
                controls["specific_entropy_denominator_min"]
            ),
            **{key: float(value) for key, value in scale.items()},
        },
    }


def _build_problem(device: torch.device):
    target_paths = _load_log_paths(
        MIXTURE_ROOT / "data" / "train.npy",
        device=device,
        dtype=torch.float32,
    )
    num_steps = int(target_paths.shape[1]) - 1
    grid = DiscreteTimeGrid(T=num_steps * 0.004, num_steps=num_steps)
    reference = fit_local_gaussian_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        ridge=1e-3,
        variance_ridge=1e-3,
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=1e-3,
        sigma_max=0.6,
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
        split_index=64,
        future_horizon=32,
        prefix_windows=(16, 32, 64),
        recent_return_window=16,
        bandwidths=(0.5, 1.0, 2.0),
    )
    complete = CompositePathFunctionalDiscrepancy(
        blocks=tuple(base.blocks)
        + (
            WeightedDiscrepancy(
                name="volatility_law_joint_prefix_future_rv",
                weight=1.0,
                discrepancy=joint,
            ),
            WeightedDiscrepancy(
                name="mixture_law_joint_path_summaries",
                weight=1.0,
                discrepancy=fit_fixed_target_heston_mixture_summary_mmd(
                    target_paths, grid=grid
                ),
            ),
        )
    )
    return target_paths, grid, reference, base, complete


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    output_dir = ensure_run_dir(args.output_dir.resolve())
    protocol_path = campaign_root / "LOCKED_PROTOCOL.json"
    protocol = _read_json(protocol_path)
    if bool(protocol.get("test_split_access_authorized", True)):
        raise RuntimeError("campaign protocol does not certify validation-only use")

    seeds = tuple(int(seed) for seed in args.seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    device = torch.device(args.device)
    target_paths, grid, reference, base, complete = _build_problem(device)
    validation_prices = np.asarray(
        np.load(MIXTURE_ROOT / "data" / "disc.npy", allow_pickle=False),
        dtype=np.float64,
    )
    control = VolatilityOnlySpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=1.0,
        reference_kernel=reference,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    estimator = HestonMixtureCausalRidgeConditionalExpectation(
        grid=grid, ridge=1e-3
    )

    summary: dict[str, object] = {
        "selection_scope": "validation_only",
        "test_split_loaded": False,
        "training_performed": False,
        "model_update_performed": False,
        "same_replay_seed_for_source_and_joint": True,
        "campaign_protocol": str(protocol_path),
        "campaign_protocol_sha256": _sha256(protocol_path),
        "safety_gate_locked_before_source_evaluation": True,
        "seeds": {},
    }
    checkpoint_hashes_before: dict[str, str] = {}
    for seed in seeds:
        campaign_arm = (
            campaign_root
            / "synthetic"
            / "mixture"
            / f"seed_{seed}"
            / "volatility_only_nested_mp"
        )
        reference_metrics = _read_json(campaign_arm.parent / "reference" / "metrics.json")
        reference_groups = _error_groups(reference_metrics)
        seed_root = ensure_run_dir(output_dir / f"seed_{seed}")
        stage_rows: dict[str, object] = {}
        for stage, checkpoint_name, discrepancy in (
            ("source", "source_model_checkpoint.pt", base),
            ("joint", "model_checkpoint.pt", complete),
        ):
            checkpoint_path = campaign_arm / checkpoint_name
            checkpoint_hashes_before[str(checkpoint_path)] = _sha256(checkpoint_path)
            model = _model(
                grid=grid,
                control=control,
                discrepancy=discrepancy,
                estimator=estimator,
                device=device,
                seed=seed,
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            model.load_checkpoint_state(checkpoint)
            bank, controls = _sample(
                model,
                num_paths=int(args.bank_size),
                batch_size=int(args.sample_batch_size),
                seed=70_000 + seed,
            )
            metrics = _evaluate_common(
                target_prices=validation_prices,
                generated=bank,
                device=device,
                seed=seed,
            )
            stage_root = ensure_run_dir(seed_root / stage)
            bank_path = stage_root / "validation_bank.npy"
            np.save(bank_path, bank)
            write_json(stage_root / "metrics.json", metrics)
            write_json(stage_root / "control_diagnostics.json", controls)
            _write_mixture_metrics(
                MIXTURE_ROOT,
                bank_path,
                stage_root / "mixture_metrics.json",
            )
            groups = _error_groups(metrics)
            stage_rows[stage] = {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hashes_before[str(checkpoint_path)],
                "validation_bank_sha256": _sha256(bank_path),
                "safety": _safety_gate(metrics, controls),
                "reference_relative": {
                    name: _gap_summary(reference_groups[name], groups[name])
                    for name in reference_groups
                },
                "metrics_path": str(stage_root / "metrics.json"),
                "control_diagnostics_path": str(
                    stage_root / "control_diagnostics.json"
                ),
                "mixture_metrics_path": str(stage_root / "mixture_metrics.json"),
            }
            del model, checkpoint, bank
            if device.type == "cuda":
                torch.cuda.empty_cache()

        summary["seeds"][str(seed)] = {"stages": stage_rows}

    checkpoint_mutation_checks = {
        path: _sha256(Path(path)) == digest
        for path, digest in checkpoint_hashes_before.items()
    }
    if not all(checkpoint_mutation_checks.values()):
        raise RuntimeError("one or more source campaign checkpoints changed during audit")
    summary["checkpoint_nonmutation"] = {
        "pass": True,
        "checks": checkpoint_mutation_checks,
    }
    write_json(output_dir / "SUMMARY.json", summary)
    print(f"wrote {output_dir / 'SUMMARY.json'}", flush=True)


if __name__ == "__main__":
    main()
