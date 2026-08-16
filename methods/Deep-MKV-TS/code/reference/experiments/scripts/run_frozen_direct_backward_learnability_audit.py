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

from path_dt_experiments.direct_backward_learnability import (  # noqa: E402
    DIRECT_SCALING_MODES,
    audit_frozen_direct_backward_learnability,
)
from path_dt_experiments.crossfit_r_control_variate import (  # noqa: E402
    CrossfitRControlVariateConfig,
)
from path_dt_experiments.frozen_backward_learnability import (  # noqa: E402
    FrozenBackwardLearnabilityConfig,
)
from path_dt_experiments.hamiltonian_regime import REGIME_NAMES  # noqa: E402
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_specific_entropy_joint_shadow import _build_models  # noqa: E402


DEFAULT_FRONTIER = Path(
    "runs/heston_dt_specific_entropy_complete_mp_nested_trust100_10x200_seed1234_20260722"
)
DEFAULT_FROZEN_BANK = Path(
    "runs/heston_dt_specific_entropy_complete_mp_nested_trust100_"
    "frozen_backward_learnability_seed1234_20260722"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit direct complete-path P/R samples on the already-frozen selected "
            "law, with and without fixed timewise-RMS loss scaling."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721"),
    )
    parser.add_argument("--frontier-run-dir", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--frozen-bank-run-dir", type=Path, default=DEFAULT_FROZEN_BANK)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--inner-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1e-4)
    parser.add_argument("--minimum-scale", type=float, default=1e-3)
    parser.add_argument(
        "--r-control-variate-mode",
        choices=("selected_p", "crossfit_knn"),
        default="selected_p",
    )
    parser.add_argument("--control-variate-folds", type=int, default=5)
    parser.add_argument("--control-variate-neighbors", type=int, default=128)
    parser.add_argument(
        "--control-variate-volatility-windows",
        type=int,
        nargs="+",
        default=(4, 8, 16, 32),
    )
    parser.add_argument(
        "--scaling-modes",
        nargs="+",
        choices=DIRECT_SCALING_MODES,
        default=DIRECT_SCALING_MODES,
    )
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


def _raw_metric(
    evaluation: dict[str, object],
    comparison: str,
    signal: str,
    name: str,
    *,
    regime: str | None = None,
) -> float | None:
    report = evaluation[comparison][signal]
    metrics = (
        report["global"]
        if regime is None
        else report["regimes"][regime]["future_control_window"]
    )
    value = metrics[name]
    return None if value is None else float(value)


def _fixed_r2(
    evaluation: dict[str, object], signal: str, *, regime: str | None = None
) -> float:
    report = evaluation["heldout_r2_against_fixed_fit_baselines"][signal]
    metrics = (
        report["global_timewise_fit_mean"]
        if regime is None
        else report["regimes_vs_fit_timewise_regime_mean"][regime][
            "future_control_window"
        ]
    )
    return float(metrics["r2"])


def _compact(report: dict[str, object]) -> dict[str, object]:
    arms = {}
    for mode, arm in report["arms"].items():
        initial = arm["initial_evaluation"]
        best = arm["best_training_evaluation"]
        arms[mode] = {
            "optimization": arm["optimization"],
            "scale_summary": arm["scale_summary"],
            "global": {
                "initial_fit_p_relative_rmse": _raw_metric(
                    initial, "fit_vs_direct_targets", "p", "relative_rmse"
                ),
                "best_fit_p_relative_rmse": _raw_metric(
                    best, "fit_vs_direct_targets", "p", "relative_rmse"
                ),
                "initial_fit_r_relative_rmse": _raw_metric(
                    initial, "fit_vs_direct_targets", "r", "relative_rmse"
                ),
                "best_fit_r_relative_rmse": _raw_metric(
                    best, "fit_vs_direct_targets", "r", "relative_rmse"
                ),
                "heldout_p_relative_rmse": _raw_metric(
                    best, "heldout_vs_direct_targets", "p", "relative_rmse"
                ),
                "heldout_r_relative_rmse": _raw_metric(
                    best, "heldout_vs_direct_targets", "r", "relative_rmse"
                ),
                "heldout_p_fixed_baseline_r2": _fixed_r2(best, "p"),
                "heldout_r_fixed_baseline_r2": _fixed_r2(best, "r"),
            },
            "regimes": [
                {
                    "regime": name,
                    "heldout_p_relative_rmse": _raw_metric(
                        best,
                        "heldout_vs_direct_targets",
                        "p",
                        "relative_rmse",
                        regime=name,
                    ),
                    "heldout_r_relative_rmse": _raw_metric(
                        best,
                        "heldout_vs_direct_targets",
                        "r",
                        "relative_rmse",
                        regime=name,
                    ),
                    "heldout_p_correlation": _raw_metric(
                        best,
                        "heldout_vs_direct_targets",
                        "p",
                        "correlation",
                        regime=name,
                    ),
                    "heldout_r_correlation": _raw_metric(
                        best,
                        "heldout_vs_direct_targets",
                        "r",
                        "correlation",
                        regime=name,
                    ),
                    "heldout_p_fixed_regime_baseline_r2": _fixed_r2(
                        best, "p", regime=name
                    ),
                    "heldout_r_fixed_regime_baseline_r2": _fixed_r2(
                        best, "r", regime=name
                    ),
                }
                for name in REGIME_NAMES
            ],
        }
    return {"control_variate": report["control_variate"], "arms": arms}


def _plot(report: dict[str, object], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for mode, arm in report["arms"].items():
        rows = list(arm["trajectory"])
        steps = [int(row["step"]) for row in rows]
        axes[0, 0].plot(
            steps,
            [float(row["normalized_fit_loss"]["total"]) for row in rows],
            label=mode,
        )
        axes[0, 1].plot(
            steps, [_fixed_r2(row, "p") for row in rows], label=mode
        )
        axes[1, 0].plot(
            steps, [_fixed_r2(row, "r") for row in rows], label=mode
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Frozen direct-target fit loss")
    axes[0, 1].set_title("Heldout P R2 vs fixed fit baseline")
    axes[1, 0].set_title("Heldout R R2 vs fixed fit baseline")

    x = torch.arange(len(REGIME_NAMES), dtype=torch.float64).numpy()
    width = 0.35
    for index, (mode, arm) in enumerate(report["arms"].items()):
        best = arm["best_training_evaluation"]
        axes[1, 1].bar(
            x + (index - 0.5) * width,
            [_fixed_r2(best, "r", regime=name) for name in REGIME_NAMES],
            width=width,
            label=mode,
        )
    axes[1, 1].set_xticks(x, REGIME_NAMES)
    axes[1, 1].set_title("Heldout future R R2 by volatility regime")
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
    selected_step = int(_load_json(frontier_dir / "metrics.json")["selected_step"])

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
        f"fitting frozen direct pathwise targets from selected step {selected_step}",
        flush=True,
    )
    result = audit_frozen_direct_backward_learnability(
        model=model,
        fit_paths=fit_paths,
        heldout_paths=heldout_paths,
        fit_noise=bank["fit_noise"],
        heldout_noise=bank["heldout_noise"],
        fit_pathwise_p=bank["fit_pathwise_p"],
        heldout_pathwise_p=bank["heldout_pathwise_p"],
        fit_regimes=bank["fit_regime_indices"],
        heldout_regimes=bank["heldout_regime_indices"],
        training=model.training,
        config=config,
        scaling_modes=tuple(args.scaling_modes),
        minimum_scale=float(args.minimum_scale),
        r_control_variate_mode=str(args.r_control_variate_mode),
        crossfit_control_variate=CrossfitRControlVariateConfig(
            folds=int(args.control_variate_folds),
            neighbors=int(args.control_variate_neighbors),
            volatility_windows=tuple(
                int(value) for value in args.control_variate_volatility_windows
            ),
        ),
        log_callback=lambda mode, row: print(
            "mode={mode} step={step} loss={loss:.6g} heldout fixed P/R R2="
            "{p:.4f}/{r:.4f}".format(
                mode=mode,
                step=int(row["step"]),
                loss=float(row["normalized_fit_loss"]["total"]),
                p=_fixed_r2(row, "p"),
                r=_fixed_r2(row, "r"),
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
        "selected_step": selected_step,
        "compact": compact,
        **result.report,
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(result.artifacts, run_dir / "diagnostic_artifacts.pt")
    _plot(payload, run_dir / "frozen_direct_backward_learnability.png")

    print(
        "R control-variate second-moment ratio fit/heldout={fit:.4f}/{held:.4f}".format(
            fit=float(compact["control_variate"]["fit_second_moment_ratio"]),
            held=float(compact["control_variate"]["heldout_second_moment_ratio"]),
        ),
        flush=True,
    )
    for mode, arm in compact["arms"].items():
        global_metrics = arm["global"]
        print(
            "mode={mode}: step={step}; fit P/R={fp:.3f}/{fr:.3f}; "
            "heldout P/R={hp:.3f}/{hr:.3f}; fixed-baseline P/R R2={p2:.4f}/{r2:.4f}".format(
                mode=mode,
                step=int(arm["optimization"]["best_step"]),
                fp=float(global_metrics["best_fit_p_relative_rmse"]),
                fr=float(global_metrics["best_fit_r_relative_rmse"]),
                hp=float(global_metrics["heldout_p_relative_rmse"]),
                hr=float(global_metrics["heldout_r_relative_rmse"]),
                p2=float(global_metrics["heldout_p_fixed_baseline_r2"]),
                r2=float(global_metrics["heldout_r_fixed_baseline_r2"]),
            ),
            flush=True,
        )
    print(f"wrote frozen direct-target audit to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
