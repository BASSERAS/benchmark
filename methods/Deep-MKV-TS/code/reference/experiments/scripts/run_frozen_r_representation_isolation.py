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

from path_dt_experiments.frozen_backward_learnability import (  # noqa: E402
    FrozenBackwardLearnabilityConfig,
)
from path_dt_experiments.hamiltonian_regime import REGIME_NAMES  # noqa: E402
from path_dt_experiments.r_representation_isolation import (  # noqa: E402
    audit_frozen_r_representation_isolation,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_specific_entropy_joint_shadow import _build_models  # noqa: E402


DEFAULT_FRONTIER = Path(
    "runs/heston_dt_specific_entropy_complete_mp_nested_trust100_10x200_seed1234_20260722"
)
DEFAULT_FROZEN_BANK = Path(
    "runs/heston_dt_specific_entropy_complete_mp_nested_trust100_"
    "frozen_backward_learnability_seed1234_20260722"
)
DEFAULT_DIRECT_TARGET = Path(
    "runs/heston_dt_specific_entropy_complete_mp_nested_trust100_"
    "frozen_direct_crossfit_r_seed1234_20260722"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare R-head-only fitting with an independent R encoder on the "
            "same frozen cross-fitted direct R targets."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721"),
    )
    parser.add_argument("--frontier-run-dir", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--frozen-bank-run-dir", type=Path, default=DEFAULT_FROZEN_BANK)
    parser.add_argument("--direct-target-run-dir", type=Path, default=DEFAULT_DIRECT_TARGET)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--inner-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1e-4)
    parser.add_argument("--minimum-scale", type=float, default=1e-3)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--split-step", type=int, default=64)
    parser.add_argument("--future-steps", type=int, default=32)
    parser.add_argument("--prefix-window", type=int, default=16)
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


def _load_artifacts(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a dictionary")
    return value


def _fixed_r2(row: dict[str, object], regime: str | None = None) -> float:
    report = row["heldout_r2_against_fixed_fit_baseline"]
    metrics = (
        report["global_timewise_fit_mean"]
        if regime is None
        else report["regimes_vs_fit_timewise_regime_mean"][regime][
            "future_control_window"
        ]
    )
    return float(metrics["r2"])


def _raw_metric(
    row: dict[str, object], comparison: str, name: str, regime: str | None = None
) -> float | None:
    report = row[comparison]
    metrics = (
        report["global"]
        if regime is None
        else report["regimes"][regime]["future_control_window"]
    )
    value = metrics[name]
    return None if value is None else float(value)


def _compact(report: dict[str, object]) -> dict[str, object]:
    compact_arms = {}
    for name, arm in report["arms"].items():
        initial = arm["initial_evaluation"]
        selected = arm["best_training_evaluation"]
        trajectory = list(arm["trajectory"])
        compact_arms[name] = {
            "optimization": arm["optimization"],
            "trainable_parameter_count": int(arm["trainable_parameter_count"]),
            "initial_heldout_fixed_r2": _fixed_r2(initial),
            "fit_selected_heldout_fixed_r2": _fixed_r2(selected),
            "maximum_diagnostic_heldout_fixed_r2": max(
                _fixed_r2(row) for row in trajectory
            ),
            "maximum_diagnostic_heldout_step": max(
                trajectory, key=_fixed_r2
            )["step"],
            "fit_selected_fit_relative_rmse": _raw_metric(
                selected, "fit_vs_direct_r", "relative_rmse"
            ),
            "fit_selected_heldout_relative_rmse": _raw_metric(
                selected, "heldout_vs_direct_r", "relative_rmse"
            ),
            "fit_selected_p_output_drift": selected[
                "p_output_drift_from_selected"
            ],
            "regimes": [
                {
                    "regime": regime,
                    "initial_heldout_fixed_r2": _fixed_r2(initial, regime),
                    "fit_selected_heldout_fixed_r2": _fixed_r2(selected, regime),
                    "maximum_diagnostic_heldout_fixed_r2": max(
                        _fixed_r2(row, regime) for row in trajectory
                    ),
                    "fit_selected_correlation": _raw_metric(
                        selected,
                        "heldout_vs_direct_r",
                        "correlation",
                        regime,
                    ),
                }
                for regime in REGIME_NAMES
            ],
        }
    return {"r_scale_summary": report["r_scale_summary"], "arms": compact_arms}


def _plot(report: dict[str, object], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name, arm in report["arms"].items():
        rows = list(arm["trajectory"])
        steps = [int(row["step"]) for row in rows]
        axes[0, 0].plot(
            steps,
            [float(row["normalized_fit_r_loss"]) for row in rows],
            label=name,
        )
        axes[0, 1].plot(steps, [_fixed_r2(row) for row in rows], label=name)
        axes[1, 0].plot(
            steps,
            [
                float(row["p_output_drift_from_selected"]["heldout_rms"])
                for row in rows
            ],
            label=name,
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Frozen normalized R fit loss")
    axes[0, 1].set_title("Heldout R R2 vs fixed fit baseline")
    axes[1, 0].set_title("Diagnostic-copy P-output drift")

    x = torch.arange(len(REGIME_NAMES), dtype=torch.float64).numpy()
    width = 0.35
    for index, (name, arm) in enumerate(report["arms"].items()):
        selected = arm["best_training_evaluation"]
        axes[1, 1].bar(
            x + (index - 0.5) * width,
            [_fixed_r2(selected, regime) for regime in REGIME_NAMES],
            width=width,
            label=name,
        )
    axes[1, 1].set_xticks(x, REGIME_NAMES)
    axes[1, 1].set_title("Fit-selected heldout future R R2")
    for axis in axes.reshape(-1):
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_dir)
    frontier_dir = Path(args.frontier_run_dir)
    frozen_bank_dir = Path(args.frozen_bank_run_dir)
    direct_target_dir = Path(args.direct_target_run_dir)
    selected_step = int(_load_json(frontier_dir / "metrics.json")["selected_step"])
    direct_metrics = _load_json(direct_target_dir / "metrics.json")
    if direct_metrics["control_variate"]["mode"] != "crossfit_knn":
        raise ValueError("direct target run must use the crossfit_knn R control variate")

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
        frontier_dir / "selected_model_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("selected checkpoint must be a dictionary")
    model.load_checkpoint_state(checkpoint)

    bank = _load_artifacts(frozen_bank_dir / "diagnostic_artifacts.pt")
    direct = _load_artifacts(direct_target_dir / "diagnostic_artifacts.pt")
    fit_paths = bank["fit_paths"]
    heldout_paths = bank["heldout_paths"]
    config = FrozenBackwardLearnabilityConfig(
        fit_bank_size=int(fit_paths.shape[0]),
        heldout_bank_size=int(heldout_paths.shape[0]),
        inner_batch_size=int(args.inner_batch_size),
        max_steps=int(args.max_steps),
        eval_every=int(args.eval_every),
        plateau_patience=int(args.plateau_patience),
        minimum_relative_improvement=float(args.minimum_relative_improvement),
        split_step=int(args.split_step),
        prefix_window=int(args.prefix_window),
        future_horizon=int(args.future_steps),
        seed=int(args.seed),
    )
    print(
        f"isolating R representation from selected step {selected_step}",
        flush=True,
    )
    result = audit_frozen_r_representation_isolation(
        model=model,
        fit_paths=fit_paths,
        heldout_paths=heldout_paths,
        fit_r=direct["fit_direct_r_control_variate"],
        heldout_r=direct["heldout_direct_r_control_variate"],
        fit_regimes=bank["fit_regime_indices"],
        heldout_regimes=bank["heldout_regime_indices"],
        training=model.training,
        config=config,
        minimum_scale=float(args.minimum_scale),
        log_callback=lambda arm, row: print(
            "arm={arm} step={step} loss={loss:.6g} heldout fixed R2={r2:.4f}".format(
                arm=arm,
                step=int(row["step"]),
                loss=float(row["normalized_fit_r_loss"]),
                r2=_fixed_r2(row),
            ),
            flush=True,
        ),
    )
    compact = _compact(result.report)
    payload = {
        "run_dir": str(run_dir),
        "source_run_dir": str(args.source_run_dir),
        "frontier_run_dir": str(frontier_dir),
        "frozen_bank_run_dir": str(frozen_bank_dir),
        "direct_target_run_dir": str(direct_target_dir),
        "selected_step": selected_step,
        "crossfit_control_variate": direct_metrics["control_variate"],
        "compact": compact,
        **result.report,
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(result.artifacts, run_dir / "diagnostic_artifacts.pt")
    _plot(payload, run_dir / "frozen_r_representation_isolation.png")
    for name, arm in compact["arms"].items():
        print(
            "arm={name}: step={step}; heldout R2 {initial:.4f}->{selected:.4f}; "
            "diagnostic max={maximum:.4f} at {max_step}; heldout relRMSE={rmse:.3f}".format(
                name=name,
                step=int(arm["optimization"]["best_step"]),
                initial=float(arm["initial_heldout_fixed_r2"]),
                selected=float(arm["fit_selected_heldout_fixed_r2"]),
                maximum=float(arm["maximum_diagnostic_heldout_fixed_r2"]),
                max_step=int(arm["maximum_diagnostic_heldout_step"]),
                rmse=float(arm["fit_selected_heldout_relative_rmse"]),
            ),
            flush=True,
        )
    print(f"wrote frozen R representation audit to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
