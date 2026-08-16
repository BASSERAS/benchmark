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
    audit_frozen_backward_learnability,
)
from path_dt_experiments.hamiltonian_regime import REGIME_NAMES  # noqa: E402
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_specific_entropy_joint_shadow import _build_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the unchanged P/R regression loss to convergence on one "
            "frozen selected-model bank and test it on an independent bank."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721"),
    )
    parser.add_argument("--frontier-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fit-bank-size", type=int, default=1024)
    parser.add_argument("--heldout-bank-size", type=int, default=1024)
    parser.add_argument("--inner-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1e-4)
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
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return result


def _metric(
    evaluation: dict[str, object],
    comparison: str,
    signal: str,
    *,
    regime: str | None = None,
) -> float:
    report = evaluation[comparison][signal]
    if regime is None:
        metrics = report["global"]
    else:
        metrics = report["regimes"][regime]["future_control_window"]
    return float(metrics["relative_rmse"])


def _plot(report: dict[str, object], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    trajectory = list(report["trajectory"])
    steps = [int(row["step"]) for row in trajectory]
    comparisons = (
        ("fit_bank_vs_fit_teacher", "fit"),
        ("heldout_network_vs_transferred_fit_teacher", "heldout transferred"),
        ("heldout_network_vs_independent_teacher", "heldout independent"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(
        steps,
        [float(row["normalized_training_loss"]["total"]) for row in trajectory],
        color="black",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Frozen normalized P/R training loss")
    axes[0, 0].set_xlabel("optimizer step")

    for comparison, label in comparisons:
        axes[0, 1].plot(
            steps,
            [_metric(row, comparison, "p") for row in trajectory],
            label=label,
        )
        axes[1, 0].plot(
            steps,
            [_metric(row, comparison, "r") for row in trajectory],
            label=label,
        )
    axes[0, 1].set_title("P relative RMSE, full grid")
    axes[1, 0].set_title("R relative RMSE, full grid")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)

    best = report["best_training_evaluation"]
    x = torch.arange(len(REGIME_NAMES), dtype=torch.float64).numpy()
    width = 0.25
    for offset, (comparison, label) in enumerate(comparisons):
        axes[1, 1].bar(
            x + (offset - 1) * width,
            [_metric(best, comparison, "r", regime=name) for name in REGIME_NAMES],
            width=width,
            label=label,
        )
    axes[1, 1].set_xticks(x, REGIME_NAMES)
    axes[1, 1].set_title("Best-step R relative RMSE by regime")
    axes[1, 1].legend(fontsize=7)
    for axis in axes.reshape(-1):
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _compact(report: dict[str, object]) -> dict[str, object]:
    best = report["best_training_evaluation"]
    initial = report["initial_evaluation"]
    teacher = report["teacher_diagnostics"][
        "transferred_fit_teacher_vs_independent_heldout_teacher"
    ]
    rows = []
    for name in REGIME_NAMES:
        rows.append(
            {
                "regime": name,
                "initial_fit_p_relative_rmse": _metric(
                    initial, "fit_bank_vs_fit_teacher", "p", regime=name
                ),
                "best_fit_p_relative_rmse": _metric(
                    best, "fit_bank_vs_fit_teacher", "p", regime=name
                ),
                "initial_fit_r_relative_rmse": _metric(
                    initial, "fit_bank_vs_fit_teacher", "r", regime=name
                ),
                "best_fit_r_relative_rmse": _metric(
                    best, "fit_bank_vs_fit_teacher", "r", regime=name
                ),
                "best_fit_p_correlation": best["fit_bank_vs_fit_teacher"]["p"][
                    "regimes"
                ][name]["future_control_window"]["correlation"],
                "best_fit_r_correlation": best["fit_bank_vs_fit_teacher"]["r"][
                    "regimes"
                ][name]["future_control_window"]["correlation"],
                "best_fit_p_r2": float(
                    best["fit_bank_vs_fit_teacher"]["p"]["regimes"][name][
                        "future_control_window"
                    ]["r2_against_timewise_mean"]
                ),
                "best_fit_r_r2": float(
                    best["fit_bank_vs_fit_teacher"]["r"]["regimes"][name][
                        "future_control_window"
                    ]["r2_against_timewise_mean"]
                ),
                "best_heldout_transferred_p_relative_rmse": _metric(
                    best,
                    "heldout_network_vs_transferred_fit_teacher",
                    "p",
                    regime=name,
                ),
                "best_heldout_transferred_r_relative_rmse": _metric(
                    best,
                    "heldout_network_vs_transferred_fit_teacher",
                    "r",
                    regime=name,
                ),
                "teacher_p_relative_rmse": float(
                    teacher["p"]["regimes"][name]["future_control_window"][
                        "relative_rmse"
                    ]
                ),
                "teacher_r_relative_rmse": float(
                    teacher["r"]["regimes"][name]["future_control_window"][
                        "relative_rmse"
                    ]
                ),
                "teacher_p_correlation": teacher["p"]["regimes"][name][
                    "future_control_window"
                ]["correlation"],
                "teacher_r_correlation": teacher["r"]["regimes"][name][
                    "future_control_window"
                ]["correlation"],
            }
        )
    return {
        "optimization": report["optimization"],
        "regimes": rows,
        "global": {
            "initial_fit_p_relative_rmse": _metric(
                initial, "fit_bank_vs_fit_teacher", "p"
            ),
            "best_fit_p_relative_rmse": _metric(
                best, "fit_bank_vs_fit_teacher", "p"
            ),
            "initial_fit_r_relative_rmse": _metric(
                initial, "fit_bank_vs_fit_teacher", "r"
            ),
            "best_fit_r_relative_rmse": _metric(
                best, "fit_bank_vs_fit_teacher", "r"
            ),
            "best_heldout_transferred_p_relative_rmse": _metric(
                best, "heldout_network_vs_transferred_fit_teacher", "p"
            ),
            "best_heldout_transferred_r_relative_rmse": _metric(
                best, "heldout_network_vs_transferred_fit_teacher", "r"
            ),
            "teacher_p_relative_rmse": float(teacher["p"]["global"]["relative_rmse"]),
            "teacher_r_relative_rmse": float(teacher["r"]["global"]["relative_rmse"]),
            "teacher_p_correlation": teacher["p"]["global"]["correlation"],
            "teacher_r_correlation": teacher["r"]["global"]["correlation"],
        },
    }


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_dir)
    frontier_dir = Path(args.frontier_run_dir)
    frontier_metrics = _load_json(frontier_dir / "metrics.json")
    selected_step = int(frontier_metrics["selected_step"])
    model_args = argparse.Namespace(**vars(args))
    # _build_models only uses this value to validate compatibility with the
    # source training logger; the audit has its own independent eval cadence.
    model_args.eval_every = 200
    (
        _source_model,
        model,
        _training,
        _stored_config,
        _heston,
        target_paths,
        heldout_paths,
        _joint,
        _mapped,
    ) = _build_models(
        source=Path(args.source_run_dir),
        device=torch.device(args.device),
        args=model_args,
    )
    selected_checkpoint = torch.load(
        frontier_dir / "selected_model_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(selected_checkpoint, dict):
        raise ValueError("selected checkpoint must be a dictionary")
    model.load_checkpoint_state(selected_checkpoint)
    config = FrozenBackwardLearnabilityConfig(
        fit_bank_size=int(args.fit_bank_size),
        heldout_bank_size=int(args.heldout_bank_size),
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
        f"building frozen fit and heldout backward problems from selected step {selected_step}",
        flush=True,
    )
    result = audit_frozen_backward_learnability(
        model=model,
        target_paths=target_paths,
        heldout_paths=heldout_paths,
        training=model.training,
        config=config,
        log_callback=lambda row: print(
            "frozen step={step} loss={loss:.6g} fit P/R={p:.3f}/{r:.3f} ".format(
                step=int(row["step"]),
                loss=float(row["normalized_training_loss"]["total"]),
                p=float(
                    row["fit_bank_vs_fit_teacher"]["p"]["global"][
                        "relative_rmse"
                    ]
                ),
                r=float(
                    row["fit_bank_vs_fit_teacher"]["r"]["global"][
                        "relative_rmse"
                    ]
                ),
            ),
            flush=True,
        ),
    )
    compact = _compact(result.report)
    payload = {
        "run_dir": str(run_dir),
        "source_run_dir": str(args.source_run_dir),
        "frontier_run_dir": str(frontier_dir),
        "selected_step": selected_step,
        "compact": compact,
        **result.report,
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(result.artifacts, run_dir / "diagnostic_artifacts.pt")
    _plot(payload, run_dir / "frozen_backward_learnability.png")

    optimization = compact["optimization"]
    print(
        "best frozen fit step={step} loss={loss:.6g}; terminal={terminal}; "
        "plateau_stop={stopped}".format(
            step=int(optimization["best_step"]),
            loss=float(optimization["best_training_loss"]),
            terminal=int(optimization["terminal_step"]),
            stopped=bool(optimization["stopped_early_on_training_plateau"]),
        ),
        flush=True,
    )
    for row in compact["regimes"]:
        print(
            "regime={regime}: fit P {p0:.3f}->{p1:.3f}, "
            "fit R {r0:.3f}->{r1:.3f}, heldout-transfer P/R={hp:.3f}/{hr:.3f}, "
            "teacher agreement P/R rmse={tp:.3f}/{tr:.3f}".format(
                regime=row["regime"],
                p0=float(row["initial_fit_p_relative_rmse"]),
                p1=float(row["best_fit_p_relative_rmse"]),
                r0=float(row["initial_fit_r_relative_rmse"]),
                r1=float(row["best_fit_r_relative_rmse"]),
                hp=float(row["best_heldout_transferred_p_relative_rmse"]),
                hr=float(row["best_heldout_transferred_r_relative_rmse"]),
                tp=float(row["teacher_p_relative_rmse"]),
                tr=float(row["teacher_r_relative_rmse"]),
            ),
            flush=True,
        )
    print(f"wrote frozen backward learnability audit to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
