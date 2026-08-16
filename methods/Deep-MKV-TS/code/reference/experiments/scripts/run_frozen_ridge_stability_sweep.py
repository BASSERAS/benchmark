from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from path_dt_experiments.ridge_stability import (  # noqa: E402
    sweep_frozen_ridge_stability,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_specific_entropy_joint_shadow import _build_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep ridge regularization on fixed complete-path P/R samples and "
            "compare independently fitted conditional teachers."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721"),
    )
    parser.add_argument("--frontier-run-dir", type=Path, required=True)
    parser.add_argument("--learnability-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--ridge-values",
        type=float,
        nargs="+",
        default=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    )
    parser.add_argument("--baseline-ridge", type=float, default=1e-3)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--split-step", type=int, default=64)
    parser.add_argument("--future-steps", type=int, default=32)
    parser.add_argument("--prefix-windows", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--recent-return-window", type=int, default=16)
    parser.add_argument("--bandwidths", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--regime-weight", type=float, default=0.0)
    parser.add_argument("--regime-prefix-window", type=int, default=16)
    parser.add_argument("--regime-gate-temperature-fraction", type=float, default=0.25)
    parser.add_argument("--regime-tail-temperature", type=float, default=0.15)
    parser.add_argument("--regime-conditional-moment-weight", type=float, default=0.0)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _compact_row(row: dict[str, object]) -> dict[str, float]:
    def finite_or_nan(value: object) -> float:
        return float("nan") if value is None else float(value)

    agreement = row["teacher_agreement"]
    sigma = row["sigma_agreement"]
    projection = row["heldout_projection_of_raw_targets"]
    retention = row["conditional_variation_retention_vs_baseline"]
    low_p = agreement["p"]["regimes"]["low"]["future_control_window"]
    low_r = agreement["r"]["regimes"]["low"]["future_control_window"]
    low_sigma = sigma["regimes"]["low"]["future_control_window"]
    return {
        "ridge": float(row["ridge"]),
        "ridge_per_fit_path": float(row["ridge_per_fit_path"]),
        "p_agreement_relative_rmse": float(
            agreement["p"]["global"]["relative_rmse"]
        ),
        "p_agreement_correlation": finite_or_nan(
            agreement["p"]["global"]["correlation"]
        ),
        "r_agreement_relative_rmse": float(
            agreement["r"]["global"]["relative_rmse"]
        ),
        "r_agreement_correlation": finite_or_nan(
            agreement["r"]["global"]["correlation"]
        ),
        "sigma_agreement_relative_rmse": float(sigma["global"]["relative_rmse"]),
        "sigma_agreement_correlation": finite_or_nan(
            sigma["global"]["correlation"]
        ),
        "heldout_raw_p_projection_r2": float(
            projection["p"]["global"]["r2_against_timewise_mean"]
        ),
        "heldout_raw_r_projection_r2": float(
            projection["r"]["global"]["r2_against_timewise_mean"]
        ),
        "p_conditional_variation_retention": float(retention["global"]["p"]),
        "r_conditional_variation_retention": float(retention["global"]["r"]),
        "sigma_conditional_variation_retention": float(
            retention["global"]["sigma"]
        ),
        "low_p_agreement_relative_rmse": float(low_p["relative_rmse"]),
        "low_p_agreement_correlation": finite_or_nan(low_p["correlation"]),
        "low_r_agreement_relative_rmse": float(low_r["relative_rmse"]),
        "low_r_agreement_correlation": finite_or_nan(low_r["correlation"]),
        "low_sigma_agreement_relative_rmse": float(low_sigma["relative_rmse"]),
        "low_sigma_agreement_correlation": finite_or_nan(
            low_sigma["correlation"]
        ),
        "low_r_conditional_variation_retention": float(
            retention["regimes"]["low"]["r"]
        ),
        "low_sigma_conditional_variation_retention": float(
            retention["regimes"]["low"]["sigma"]
        ),
        "transferred_sigma_cap_fraction": float(
            row["control_bounds"]["transferred_sigma_cap_fraction"]
        ),
        "independent_sigma_cap_fraction": float(
            row["control_bounds"]["independent_sigma_cap_fraction"]
        ),
    }


def _plot(rows: list[dict[str, float]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    x = [row["ridge"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(x, [row["p_agreement_correlation"] for row in rows], marker="o", label="P")
    axes[0, 0].plot(x, [row["r_agreement_correlation"] for row in rows], marker="o", label="R")
    axes[0, 0].plot(x, [row["sigma_agreement_correlation"] for row in rows], marker="o", label="sigma")
    axes[0, 0].set_title("Independent-teacher agreement correlation")

    axes[0, 1].plot(x, [row["p_agreement_relative_rmse"] for row in rows], marker="o", label="P")
    axes[0, 1].plot(x, [row["r_agreement_relative_rmse"] for row in rows], marker="o", label="R")
    axes[0, 1].plot(x, [row["sigma_agreement_relative_rmse"] for row in rows], marker="o", label="sigma")
    axes[0, 1].set_title("Independent-teacher agreement relative RMSE")

    axes[1, 0].plot(x, [row["p_conditional_variation_retention"] for row in rows], marker="o", label="P")
    axes[1, 0].plot(x, [row["r_conditional_variation_retention"] for row in rows], marker="o", label="R")
    axes[1, 0].plot(x, [row["sigma_conditional_variation_retention"] for row in rows], marker="o", label="sigma")
    axes[1, 0].set_title("Conditional variation retained vs ridge 0.001")

    axes[1, 1].plot(x, [row["heldout_raw_p_projection_r2"] for row in rows], marker="o", label="P")
    axes[1, 1].plot(x, [row["heldout_raw_r_projection_r2"] for row in rows], marker="o", label="R")
    axes[1, 1].set_title("Heldout raw-target projection R²")
    for axis in axes.reshape(-1):
        axis.set_xscale("log")
        axis.set_xlabel("ridge lambda")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_dir)
    learnability_dir = Path(args.learnability_run_dir)
    learnability_metrics = _load_json(learnability_dir / "metrics.json")
    artifacts = torch.load(
        learnability_dir / "diagnostic_artifacts.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(artifacts, dict):
        raise ValueError("learnability artifacts must be a dictionary")
    model_args = argparse.Namespace(**vars(args))
    model_args.eval_every = 200
    (
        _source_model,
        model,
        _training,
        _stored_config,
        _heston,
        _target_paths,
        _heldout_paths,
        _joint,
        _mapped,
    ) = _build_models(
        source=Path(args.source_run_dir),
        device=torch.device(args.device),
        args=model_args,
    )
    checkpoint = torch.load(
        Path(args.frontier_run_dir) / "selected_model_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("selected checkpoint must be a dictionary")
    model.load_checkpoint_state(checkpoint)
    device = model.device
    dtype = model.dtype
    print(
        "sweeping ridge values on the saved frozen pathwise P/R banks",
        flush=True,
    )
    result = sweep_frozen_ridge_stability(
        fit_paths=artifacts["fit_paths"].to(device=device, dtype=dtype),
        heldout_paths=artifacts["heldout_paths"].to(device=device, dtype=dtype),
        fit_pathwise_p=artifacts["fit_pathwise_p"].to(device=device, dtype=dtype),
        fit_pathwise_r=artifacts["fit_pathwise_r"].to(device=device, dtype=dtype),
        heldout_pathwise_p=artifacts["heldout_pathwise_p"].to(
            device=device, dtype=dtype
        ),
        heldout_pathwise_r=artifacts["heldout_pathwise_r"].to(
            device=device, dtype=dtype
        ),
        fit_regimes=artifacts["fit_regime_indices"].to(device=device),
        heldout_regimes=artifacts["heldout_regime_indices"].to(device=device),
        control=model.control_map,
        ridge_values=tuple(float(value) for value in args.ridge_values),
        baseline_ridge=float(args.baseline_ridge),
        split_step=int(args.split_step),
        future_horizon=int(args.future_steps),
    )
    compact_rows = [_compact_row(row) for row in result.report["rows"]]
    payload = {
        "run_dir": str(run_dir),
        "source_run_dir": str(args.source_run_dir),
        "frontier_run_dir": str(args.frontier_run_dir),
        "learnability_run_dir": str(learnability_dir),
        "selected_step": int(learnability_metrics["selected_step"]),
        "compact_rows": compact_rows,
        **result.report,
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(result.artifacts, run_dir / "diagnostic_artifacts.pt")
    _plot(compact_rows, run_dir / "ridge_stability_sweep.png")
    for row in compact_rows:
        print(
            "ridge={ridge:g}: P corr/rmse={pc:.3f}/{pe:.3f}, "
            "R corr/rmse={rc:.3f}/{re:.3f}, sigma corr/rmse={sc:.3f}/{se:.3f}, "
            "R/sigma variation={rv:.3f}/{sv:.3f}, raw R2={r2:.3f}".format(
                ridge=row["ridge"],
                pc=row["p_agreement_correlation"],
                pe=row["p_agreement_relative_rmse"],
                rc=row["r_agreement_correlation"],
                re=row["r_agreement_relative_rmse"],
                sc=row["sigma_agreement_correlation"],
                se=row["sigma_agreement_relative_rmse"],
                rv=row["r_conditional_variation_retention"],
                sv=row["sigma_conditional_variation_retention"],
                r2=row["heldout_raw_r_projection_r2"],
            ),
            flush=True,
        )
    print(f"wrote frozen ridge stability sweep to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
