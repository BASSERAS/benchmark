from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.uncapped_audit import (  # noqa: E402
    common_noise_path_transition,
    finite_pearson_correlation,
)
from run_specific_entropy_joint_shadow import _build_models  # noqa: E402


DEFAULT_FRONTIER = Path(
    "runs/heston_dt_specific_entropy_complete_mp_dt_clipnone_frontier_seed1234_20260721"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit an unclipped complete-MP checkpoint trajectory without "
            "retraining, using logged regression/control diagnostics and "
            "saved common-noise validation banks."
        )
    )
    parser.add_argument("--frontier-run-dir", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--control-paths", type=int, default=1024)
    parser.add_argument("--control-seed", type=int, default=24001331)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_history(path: Path) -> list[dict[str, float]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("training history must contain JSON objects")
    return [{str(key): float(value) for key, value in row.items()} for row in rows]


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} must contain a tensor")
    return value.detach().cpu()


def _training_value(row: dict[str, float], name: str) -> float:
    if name not in row:
        raise ValueError(f"training history is missing {name}")
    return float(row[name])


def _common_control_profile(
    model,
    *,
    num_paths: int,
    seed: int,
    x0: float,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    with torch.no_grad():
        sample = model.sample(
            num_paths=int(num_paths),
            x0=torch.tensor([[float(x0)]], device=model.device, dtype=model.dtype),
            seed=int(seed),
        )
    state_dim = int(model.architecture.state_dim)
    alpha = sample.controls[..., :state_dim].detach()
    sigma = sample.controls[..., state_dim:].detach()
    p_values = sample.expected_adjoint_next.detach()
    r_values = sample.expected_adjoint_noise_next.detach()
    sigma_max = getattr(model.control_map, "sigma_max", None)
    if sigma_max is None:
        upper_active = torch.zeros_like(sigma, dtype=torch.bool)
    else:
        tolerance = max(1e-7, abs(float(sigma_max)) * 1e-5)
        upper_active = sigma >= float(sigma_max) - tolerance
    upper_by_time = upper_active.to(dtype=sigma.dtype).mean(dim=(0, 2))
    sigma_mean_by_time = sigma.mean(dim=(0, 2))
    sigma_std_by_time = sigma.std(dim=(0, 2), unbiased=False)
    alpha_rms_by_time = torch.sqrt(torch.mean(alpha.pow(2), dim=(0, 2)))
    p_rms_by_time = torch.sqrt(torch.mean(p_values.pow(2), dim=(0, 2)))
    r_rms_by_time = torch.sqrt(torch.mean(r_values.pow(2), dim=(0, 2)))
    third = max(1, int(sigma.shape[1]) // 3)
    maximum_index = int(torch.argmax(upper_by_time).item())
    summary = {
        "common_alpha_rms": float(torch.sqrt(torch.mean(alpha.pow(2))).item()),
        "common_p_rms": float(torch.sqrt(torch.mean(p_values.pow(2))).item()),
        "common_r_rms": float(torch.sqrt(torch.mean(r_values.pow(2))).item()),
        "common_sigma_mean": float(sigma.mean().item()),
        "common_sigma_std": float(sigma.std(unbiased=False).item()),
        "common_sigma_max_active_fraction": float(upper_active.to(dtype=sigma.dtype).mean().item()),
        "common_sigma_max_active_fraction_time_max": float(upper_by_time.max().item()),
        "common_sigma_max_active_peak_step": float(maximum_index),
        "common_sigma_max_active_early": float(upper_by_time[:third].mean().item()),
        "common_sigma_max_active_middle": float(upper_by_time[third : 2 * third].mean().item()),
        "common_sigma_max_active_late": float(upper_by_time[2 * third :].mean().item()),
    }
    profiles = {
        "sigma_mean": [float(value) for value in sigma_mean_by_time.cpu().tolist()],
        "sigma_std": [float(value) for value in sigma_std_by_time.cpu().tolist()],
        "sigma_max_active_fraction": [float(value) for value in upper_by_time.cpu().tolist()],
        "alpha_rms": [float(value) for value in alpha_rms_by_time.cpu().tolist()],
        "p_rms": [float(value) for value in p_rms_by_time.cpu().tolist()],
        "r_rms": [float(value) for value in r_rms_by_time.cpu().tolist()],
    }
    return summary, profiles


def _plot_audit(
    records: list[dict[str, float]],
    *,
    baseline_shadowing: dict[str, object],
    baseline_stylized: dict[str, object],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    steps = [int(record["step"]) for record in records]
    figure, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)

    def panel(
        axis,
        names: tuple[tuple[str, str], ...],
        title: str,
        *,
        horizontal: tuple[float, str] | None = None,
        log_y: bool = False,
    ) -> None:
        for name, label in names:
            axis.plot(steps, [record[name] for record in records], marker="o", label=label)
        if horizontal is not None:
            axis.axhline(horizontal[0], color="#d62728", linestyle="--", label=horizontal[1])
        if log_y:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    panel(
        axes[0, 0],
        (("realized_volatility_crps", "uncapped"),),
        "RV CRPS",
        horizontal=(float(baseline_shadowing["realized_volatility_crps"]), "source"),
    )
    panel(
        axes[0, 1],
        (("realized_volatility_coverage_90", "uncapped"),),
        "RV 90% coverage",
        horizontal=(0.90, "nominal"),
    )
    panel(
        axes[0, 2],
        (("excess_kurtosis_error_rms", "uncapped"),),
        "Excess-kurtosis error",
        horizontal=(float(baseline_stylized["excess_kurtosis_error_rms"]), "source"),
    )
    panel(
        axes[1, 0],
        (("rolling_train_total_loss_mean", "mean"), ("rolling_train_total_loss_q95", "q95")),
        "Regression loss (rolling)",
        log_y=True,
    )
    panel(
        axes[1, 1],
        (
            ("adjoint_prediction_target_rms_ratio", "P prediction/target"),
            ("noise_adjoint_prediction_target_rms_ratio", "R prediction/target"),
        ),
        "Learned adjoint signal fraction",
        log_y=True,
    )
    panel(
        axes[1, 2],
        (("adjoint_relative_rmse", "P"), ("noise_adjoint_relative_rmse", "R")),
        "Adjoint relative RMSE",
        horizontal=(1.0, "zero-prediction level"),
    )
    panel(
        axes[2, 0],
        (
            ("common_sigma_mean", "mean"),
            ("common_sigma_std", "std"),
            ("common_sigma_max_active_fraction", "at upper cap"),
        ),
        "Volatility-control diagnostics",
    )
    panel(
        axes[2, 1],
        (("parameter_norm_before_update", "parameter norm"), ("rolling_parameter_update_norm_mean", "update norm")),
        "Parameters and AdamW updates",
        log_y=True,
    )
    panel(
        axes[2, 2],
        (
            ("running_to_discrepancy_source_rms_ratio", "running/discrepancy"),
            ("path_rms_change", "common-noise path change"),
        ),
        "Source balance and law movement",
        log_y=True,
    )
    for axis in axes[2, :]:
        axis.set_xlabel("Training step")
    figure.suptitle("Unclipped complete-MP trajectory audit")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    frontier = Path(args.frontier_run_dir)
    run_dir = ensure_run_dir(args.run_dir)
    metrics = _load_json(frontier / "metrics.json")
    config = metrics.get("config")
    if not isinstance(config, dict):
        raise ValueError("frontier metrics are missing config")
    training_config = config.get("training")
    if not isinstance(training_config, dict) or training_config.get("grad_clip_norm") is not None:
        raise ValueError("trajectory audit requires an unclipped frontier run")
    selection_history = metrics.get("selection_history")
    if not isinstance(selection_history, list) or not selection_history:
        raise ValueError("frontier metrics are missing selection_history")
    training_history = _load_history(frontier / "training_history.jsonl")
    training_by_step = {int(round(row["step"])): row for row in training_history}
    if len(training_by_step) != len(training_history):
        raise ValueError("training-history steps must be unique")

    baseline_validation = metrics.get("validation")
    if not isinstance(baseline_validation, dict) or not isinstance(
        baseline_validation.get("baseline"), dict
    ):
        raise ValueError("frontier metrics are missing validation baseline")
    baseline = dict(baseline_validation["baseline"])
    baseline_shadowing = dict(baseline["shadowing"])
    baseline_stylized = dict(baseline["stylized"])
    previous_paths = _load_tensor(frontier / "baseline_validation_generated_paths.pt")

    training_fields = (
        "train_total_loss",
        "rolling_train_total_loss_mean",
        "rolling_train_total_loss_q95",
        "adjoint_relative_rmse",
        "noise_adjoint_relative_rmse",
        "normalized_adjoint_relative_rmse",
        "normalized_noise_adjoint_relative_rmse",
        "adjoint_prediction_target_rms_ratio",
        "noise_adjoint_prediction_target_rms_ratio",
        "ce_projection_r2",
        "noise_ce_projection_r2",
        "expected_adjoint_rms",
        "expected_adjoint_noise_rms",
        "projected_adjoint_target_abs_rms",
        "projected_adjoint_noise_target_abs_rms",
        "sigma_mean",
        "sigma_std",
        "sigma_q99",
        "sigma_min_active_fraction",
        "sigma_max_active_fraction",
        "running_source_rms",
        "discrepancy_source_rms",
        "running_to_discrepancy_source_rms_ratio",
        "running_discrepancy_source_cosine",
        "rolling_grad_norm_mean",
        "rolling_grad_norm_q95",
        "rolling_parameter_update_norm_mean",
        "rolling_parameter_update_norm_q95",
        "update_to_parameter_norm",
        "parameter_norm_before_update",
        "rolling_gradient_nonfinite_fraction_max",
        "rolling_parameter_nonfinite_fraction_max",
        "volatility_law_joint_prefix_future_rv_joint_prefix_future_volatility_future_rv_mean_ratio",
        "volatility_law_joint_prefix_future_rv_joint_prefix_future_volatility_future_rv_std_ratio",
    )
    records: list[dict[str, float]] = []
    for item in selection_history:
        if not isinstance(item, dict):
            raise ValueError("selection-history rows must be dictionaries")
        step = int(item["step"])
        if step not in training_by_step:
            raise ValueError(f"training history is missing checkpoint step {step}")
        shadowing = dict(item["shadowing"])
        stylized = dict(item["stylized"])
        current_paths = _load_tensor(
            frontier / "validation_checkpoints" / f"step_{step:04d}_validation_paths.pt"
        )
        transition = common_noise_path_transition(previous_paths, current_paths)
        training_row = training_by_step[step]
        record = {
            "step": float(step),
            **{name: float(value) for name, value in shadowing.items() if not isinstance(value, str)},
            **{name: float(value) for name, value in stylized.items()},
            **{name: _training_value(training_row, name) for name in training_fields},
            **transition,
        }
        record["alpha_rms"] = record["expected_adjoint_rms"]
        records.append(record)
        previous_paths = current_paths

    if int(args.control_paths) < 1:
        raise ValueError("control-paths must be >= 1")
    build_args = SimpleNamespace(
        eval_every=int(config["eval_every"]),
        joint_weight=float(config["joint_weight"]),
        prefix_windows=tuple(int(value) for value in config["prefix_windows_full_grid"]),
        split_step=int(config["split_step_full_grid"]),
        future_steps=int(config["future_steps_full_grid"]),
        recent_return_window=int(config["recent_return_window_full_grid"]),
        bandwidths=tuple(float(value) for value in config["bandwidths"]),
        seed=int(config["seed"]),
    )
    source_run_dir = metrics.get("source_run_dir")
    if not isinstance(source_run_dir, str):
        raise ValueError("frontier metrics are missing source_run_dir")
    (
        _source_model,
        checkpoint_model,
        _training,
        _stored_config,
        heston,
        _target_paths,
        _heldout_paths,
        _joint,
        _mapped,
    ) = _build_models(
        source=Path(source_run_dir),
        device=torch.device(args.device),
        args=build_args,
    )
    control_profiles: dict[str, dict[str, list[float]]] = {}
    for record, item in zip(records, selection_history):
        step = int(record["step"])
        checkpoint = torch.load(
            Path(str(item["checkpoint"])),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError(f"step {step} checkpoint must be a dictionary")
        checkpoint_model.load_checkpoint_state(checkpoint)
        control_summary, profile = _common_control_profile(
            checkpoint_model,
            num_paths=int(args.control_paths),
            seed=int(args.control_seed),
            x0=float(heston.x0),
        )
        record.update(control_summary)
        control_profiles[str(step)] = profile
        print(f"replayed common controls at step {step}", flush=True)

    correlation_pairs = (
        ("sigma_max_active_fraction", "excess_kurtosis_error_rms"),
        ("common_sigma_max_active_fraction", "excess_kurtosis_error_rms"),
        ("common_sigma_max_active_late", "excess_kurtosis_error_rms"),
        ("sigma_std", "excess_kurtosis_error_rms"),
        ("rolling_parameter_update_norm_mean", "excess_kurtosis_error_rms"),
        ("expected_adjoint_rms", "realized_volatility_crps"),
        ("expected_adjoint_noise_rms", "realized_volatility_coverage_90"),
        ("path_rms_change", "excess_kurtosis_error_rms"),
    )
    correlations = {
        f"{left}__vs__{right}": finite_pearson_correlation(
            records,
            left=left,
            right=right,
        )
        for left, right in correlation_pairs
    }
    parameter_norms = [record["parameter_norm_before_update"] for record in records]
    p_ratios = [record["adjoint_prediction_target_rms_ratio"] for record in records]
    r_ratios = [record["noise_adjoint_prediction_target_rms_ratio"] for record in records]
    p_errors = [record["adjoint_relative_rmse"] for record in records]
    r_errors = [record["noise_adjoint_relative_rmse"] for record in records]
    losses = [record["train_total_loss"] for record in records]
    source_ratios = [record["running_to_discrepancy_source_rms_ratio"] for record in records]
    sigma_q99 = [record["sigma_q99"] for record in records]
    best_rv = min(records, key=lambda record: record["realized_volatility_crps"])
    best_kurtosis = min(records, key=lambda record: record["excess_kurtosis_error_rms"])
    best_coverage = min(
        records,
        key=lambda record: abs(record["realized_volatility_coverage_90"] - 0.90),
    )
    summary = {
        "all_gradients_and_parameters_finite": bool(
            max(record["rolling_gradient_nonfinite_fraction_max"] for record in records) == 0.0
            and max(record["rolling_parameter_nonfinite_fraction_max"] for record in records) == 0.0
        ),
        "p_prediction_target_rms_ratio_range": [min(p_ratios), max(p_ratios)],
        "r_prediction_target_rms_ratio_range": [min(r_ratios), max(r_ratios)],
        "p_relative_rmse_range": [min(p_errors), max(p_errors)],
        "r_relative_rmse_range": [min(r_errors), max(r_errors)],
        "training_loss_range": [min(losses), max(losses)],
        "parameter_norm_growth_factor": parameter_norms[-1] / parameter_norms[0],
        "running_to_discrepancy_source_rms_ratio_range": [
            min(source_ratios),
            max(source_ratios),
        ],
        "sigma_q99_equals_configured_upper_bound_at_every_checkpoint": bool(
            max(sigma_q99) - min(sigma_q99) <= 1e-6
            and all(record["sigma_max_active_fraction"] > 0.0 for record in records)
        ),
        "best_rv_crps_step": int(best_rv["step"]),
        "best_kurtosis_step": int(best_kurtosis["step"]),
        "closest_coverage_step": int(best_coverage["step"]),
    }

    payload: dict[str, object] = {
        "frontier_run_dir": str(frontier),
        "protocol": {
            "retraining_performed": False,
            "gradient_clipping_disabled": True,
            "validation_banks_share_noise_seed_across_checkpoints": True,
            "path_transition_definition": "current checkpoint minus preceding checkpoint; source precedes step 200",
            "training_control_diagnostics_use_the_logged_training_rollout": True,
            "correlations_are_exploratory_across_ten_checkpoints": True,
        },
        "summary": summary,
        "correlations": correlations,
        "baseline": baseline,
        "records": records,
        "common_control_profiles": control_profiles,
        "common_control_config": {
            "num_paths": int(args.control_paths),
            "seed": int(args.control_seed),
            "device": str(args.device),
        },
    }
    write_json(run_dir / "metrics.json", payload)
    with (run_dir / "checkpoint_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _plot_audit(
        records,
        baseline_shadowing=baseline_shadowing,
        baseline_stylized=baseline_stylized,
        output_path=run_dir / "uncapped_trajectory_audit.png",
    )
    print(f"wrote unclipped trajectory audit to {run_dir}", flush=True)
    print(
        "P signal ratio {p0:.4%}-{p1:.4%}; R signal ratio {r0:.4%}-{r1:.4%}; "
        "sigma-cap/kurtosis correlation={correlation:.3f}".format(
            p0=min(p_ratios),
            p1=max(p_ratios),
            r0=min(r_ratios),
            r1=max(r_ratios),
            correlation=correlations[
                "sigma_max_active_fraction__vs__excess_kurtosis_error_rms"
            ],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
