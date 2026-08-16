#!/usr/bin/env python3
"""Select Guyon Deep-MKV checkpoints on independent validation banks.

This is a replay-only evaluator.  It reconstructs the locked task, loads
saved checkpoints without mutation, and uses common random numbers across
all checkpoint candidates on each validation bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    CausalPathFeatureRidgeConditionalExpectation,
    DiscreteTimeGrid,
    fit_guyon_lekeufack_likelihood_reference_kernel,
)
from deep_mkv_gen_path_dt.ce import RidgeConditionalExpectation  # noqa: E402
from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    ReferenceDriftSpecificEntropyDiagonalControl,
    VolatilityOnlySpecificEntropyDiagonalControl,
)
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_drawdown_memory_deep_mkv_ablation import (  # noqa: E402
    _build_model as _build_delayed_model,
    _load_dataset,
    _sample_prices,
)
from run_heston_mixture_deep_mkv_ablation import _load_log_paths  # noqa: E402
from run_matched_control_synthetic_validation import (  # noqa: E402
    _evaluate_common,
    _model as _build_heston_model,
    _sample as _sample_heston,
)
from evaluate_long_horizon_sbts import evaluate_bank as evaluate_delayed_bank  # noqa: E402


HESTON_ROOT = Path("/home/samer/scenarios/BASSERAS-benchmark/dataset/Heston")
DELAYED_ROOT = REPO_ROOT / "runs" / "drawdown_memory_seed0_comparison_20260730" / "data"

HESTON_SELECTION_METRICS = (
    "path_swd_normalized",
    "realized_volatility_w1_normalized",
    "absolute_return_acf_rmse",
    "prefix_future_rv_correlation_error",
)
DELAYED_DISTRIBUTION_METRICS = (
    "path_swd_normalized",
    "terminal_w1_normalized",
    "increment_w1_normalized",
    "realized_volatility_w1_normalized",
    "maximum_drawdown_w1_normalized",
    "return_qq_rmse_normalized",
)
DELAYED_DEPENDENCE_METRICS = (
    "return_acf_rmse",
    "absolute_return_acf_rmse",
    "squared_return_acf_rmse",
    "future_rv_hit_gap_error",
    "early_hit_future_rv_correlation_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("heston", "delayed128"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--bank-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_paths(task: str, run_dir: Path, steps: tuple[int, ...]) -> dict[int, Path]:
    if task == "heston":
        arm_dir = run_dir / "volatility_only_online_mp"
    else:
        arm_dir = run_dir / "generic_ce_online_source_volatility_only"
    result: dict[int, Path] = {}
    for step in steps:
        intermediate = arm_dir / "training_checkpoints" / f"step_{step:04d}.pt"
        result[step] = intermediate if intermediate.is_file() else arm_dir / "model_checkpoint.pt"
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoints: " + ", ".join(missing))
    return result


def zero_network(model) -> None:
    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.zero_()


def heston_metric_vector(metrics: dict[str, object]) -> dict[str, float]:
    path = metrics["path_law"]
    memory = metrics["conditional_path_memory"]
    generated = memory["generated"]
    target = memory["target"]
    return {
        "path_swd_normalized": float(path["path_swd_normalized"]),
        "realized_volatility_w1_normalized": float(
            path["realized_volatility_w1_normalized"]
        ),
        "absolute_return_acf_rmse": float(path["absolute_return_acf_rmse"]),
        "prefix_future_rv_correlation_error": abs(
            float(generated["prefix_future_rv_correlation"])
            - float(target["prefix_future_rv_correlation"])
        ),
    }


def relative_gain(reference: float, candidate: float) -> float:
    return (float(reference) - float(candidate)) / max(abs(float(reference)), 1e-12)


def heston_score(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    gains = {
        key: relative_gain(reference[key], candidate[key])
        for key in HESTON_SELECTION_METRICS
    }
    return {"metric_gains": gains, "score": float(np.mean(list(gains.values())))}


def delayed_score(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    gains = {
        key: relative_gain(reference[key], candidate[key])
        for key in (*DELAYED_DISTRIBUTION_METRICS, *DELAYED_DEPENDENCE_METRICS)
    }
    distribution = float(np.median([gains[key] for key in DELAYED_DISTRIBUTION_METRICS]))
    dependence = float(np.median([gains[key] for key in DELAYED_DEPENDENCE_METRICS]))
    return {
        "metric_gains": gains,
        "distribution_median_gain": distribution,
        "dependence_median_gain": dependence,
        "score": 0.5 * (distribution + dependence),
    }


def build_heston(device: torch.device):
    target = _load_log_paths(
        HESTON_ROOT / "heston_S_8192x128.npy",
        device=device,
        dtype=torch.float32,
    )
    validation = np.asarray(
        np.load(HESTON_ROOT / "heston_S_disc_8192x128.npy", allow_pickle=False),
        dtype=np.float64,
    )
    num_steps = int(target.shape[1]) - 1
    grid = DiscreteTimeGrid(T=num_steps * 0.004, num_steps=num_steps)
    reference = fit_guyon_lekeufack_likelihood_reference_kernel(
        target_paths=target,
        grid=grid,
        ridge=1e-3,
        variance_ridge=1e-3,
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=1e-3,
        sigma_max=0.6,
        activity_update="structural_variance",
    )
    discrepancy = build_heston_discrepancy(
        num_steps=num_steps,
        include_acf=True,
        preset="old_fullv_w0p25",
        lambda_scale=50.0,
        kappa_scale=100.0,
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )
    control = ReferenceDriftSpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=1.0,
        reference_kernel=reference,
        sigma_min=1e-3,
        sigma_max=0.6,
        allow_drift_correction=False,
    )
    model = _build_heston_model(
        grid=grid,
        control=control,
        discrepancy=discrepancy,
        estimator=RidgeConditionalExpectation(ridge=1e-3),
        device=device,
        seed=0,
        adjoint_network="gru",
        transformer_layers=2,
        transformer_heads=4,
    )
    return model, validation


def build_delayed(device: torch.device):
    target, manifest, dataset_paths = _load_dataset(
        DELAYED_ROOT,
        device=device,
        dtype=torch.float32,
    )
    config = manifest["configuration"]
    num_steps = int(config["sequence_length"]) - 1
    grid = DiscreteTimeGrid(T=num_steps * float(config["dt"]), num_steps=num_steps)
    reference = fit_guyon_lekeufack_likelihood_reference_kernel(
        target_paths=target,
        grid=grid,
        ridge=1e-3,
        variance_ridge=1e-3,
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=1e-3,
        sigma_max=0.6,
        activity_update="structural_variance",
    )
    discrepancy = build_heston_discrepancy(
        num_steps=num_steps,
        include_acf=True,
        acf_lags=(1, 2, 5, 10),
        preset="old_fullv_w0p25",
        lambda_scale=50.0,
        kappa_scale=100.0,
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )
    control = VolatilityOnlySpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=2.0,
        reference_kernel=reference,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    estimator = CausalPathFeatureRidgeConditionalExpectation(
        grid=grid,
        ridge=1e-3,
        rolling_windows=(5, 10, 20, 32),
        ewma_half_lives=(5, 20, 60),
        downside_windows=(10, 32),
        drawdown_thresholds=(),
        drawdown_memory_half_lives=(),
        drawdown_memory_lags=(),
        drawdown_gate_temperature=0.002,
    )
    model_args = SimpleNamespace(
        hidden_dim=96,
        num_layers=1,
        adjoint_network="gru",
        transformer_layers=2,
        transformer_heads=4,
    )
    model = _build_delayed_model(
        args=model_args,
        grid=grid,
        control=control,
        discrepancy=discrepancy,
        ce_estimator=estimator,
        device=device,
        dtype=torch.float32,
        seed=0,
    )
    return model, dataset_paths, float(config["s0"])


def main() -> None:
    args = parse_args()
    steps = tuple(sorted(set(int(value) for value in args.steps)))
    bank_seeds = tuple(int(value) for value in args.bank_seeds)
    if len(bank_seeds) < 2 or len(set(bank_seeds)) != len(bank_seeds):
        raise ValueError("bank-seeds must contain at least two distinct values")
    run_dir = args.run_dir.resolve()
    output_dir = ensure_run_dir(args.output_dir.resolve())
    checkpoints = checkpoint_paths(args.task, run_dir, steps)
    hashes_before = {step: sha256(path) for step, path in checkpoints.items()}
    device = torch.device(args.device)

    if args.task == "heston":
        model, validation_prices = build_heston(device)
        dataset_paths = None
        s0 = None
    else:
        model, dataset_paths, s0 = build_delayed(device)
        validation_prices = None

    zero_network(model)
    reference_checkpoint = model.checkpoint_state()
    records: dict[str, object] = {}
    scores_by_step: dict[int, list[float]] = {step: [] for step in steps}
    for bank_seed in bank_seeds:
        bank_dir = ensure_run_dir(output_dir / f"bank_{bank_seed}")
        model.load_checkpoint_state(reference_checkpoint)
        if args.task == "heston":
            reference_bank, reference_controls = _sample_heston(
                model,
                num_paths=int(args.bank_size),
                batch_size=int(args.sample_batch_size),
                seed=bank_seed,
            )
            reference_metrics = _evaluate_common(
                target_prices=validation_prices,
                generated=reference_bank,
                device=device,
                seed=bank_seed,
            )
            reference_vector = heston_metric_vector(reference_metrics)
            write_json(bank_dir / "reference_metrics.json", reference_metrics)
            write_json(bank_dir / "reference_controls.json", reference_controls)
        else:
            reference_bank = _sample_prices(
                model,
                num_paths=int(args.bank_size),
                batch_size=int(args.sample_batch_size),
                seed=bank_seed,
                s0=s0,
            )
            reference_path = bank_dir / "reference_bank.npy"
            np.save(reference_path, reference_bank)
            reference_metrics = evaluate_delayed_bank(
                dataset_root=DELAYED_ROOT,
                generated_path=reference_path,
                sequence_length=128,
                seed=bank_seed,
                subsample_size=1024,
                swd_projections=256,
                device=device,
            )
            reference_vector = reference_metrics["metrics"]
            write_json(bank_dir / "reference_metrics.json", reference_metrics)

        bank_record: dict[str, object] = {
            "reference_selection_metrics": reference_vector,
            "steps": {},
        }
        for step in steps:
            checkpoint = torch.load(
                checkpoints[step], map_location=device, weights_only=True
            )
            model.load_checkpoint_state(checkpoint)
            if args.task == "heston":
                bank, controls = _sample_heston(
                    model,
                    num_paths=int(args.bank_size),
                    batch_size=int(args.sample_batch_size),
                    seed=bank_seed,
                )
                metrics = _evaluate_common(
                    target_prices=validation_prices,
                    generated=bank,
                    device=device,
                    seed=bank_seed,
                )
                vector = heston_metric_vector(metrics)
                score = heston_score(reference_vector, vector)
                write_json(bank_dir / f"step_{step:04d}_controls.json", controls)
            else:
                bank = _sample_prices(
                    model,
                    num_paths=int(args.bank_size),
                    batch_size=int(args.sample_batch_size),
                    seed=bank_seed,
                    s0=s0,
                )
                bank_path = bank_dir / f"step_{step:04d}_bank.npy"
                np.save(bank_path, bank)
                metrics = evaluate_delayed_bank(
                    dataset_root=DELAYED_ROOT,
                    generated_path=bank_path,
                    sequence_length=128,
                    seed=bank_seed,
                    subsample_size=1024,
                    swd_projections=256,
                    device=device,
                )
                vector = metrics["metrics"]
                score = delayed_score(reference_vector, vector)
            write_json(bank_dir / f"step_{step:04d}_metrics.json", metrics)
            bank_record["steps"][str(step)] = {
                "selection_metrics": vector,
                **score,
            }
            scores_by_step[step].append(float(score["score"]))
            print(
                f"{args.task} bank={bank_seed} step={step} "
                f"score={float(score['score']):.6f}",
                flush=True,
            )
        records[str(bank_seed)] = bank_record

    aggregates = {
        str(step): {
            "bank_scores": scores_by_step[step],
            "mean_score": float(np.mean(scores_by_step[step])),
            "median_score": float(np.median(scores_by_step[step])),
            "minimum_score": float(np.min(scores_by_step[step])),
        }
        for step in steps
    }
    selected_step = max(
        steps,
        key=lambda step: (float(aggregates[str(step)]["median_score"]), -step),
    )
    for step, path in checkpoints.items():
        if sha256(path) != hashes_before[step]:
            raise RuntimeError(f"checkpoint {step} changed during replay")
    write_json(
        output_dir / "SELECTION.json",
        {
            "task": args.task,
            "selection_scope": "validation_only",
            "test_split_loaded": False,
            "training_performed": False,
            "model_update_performed": False,
            "reference_kind": "guyon_lekeufack_structural_likelihood",
            "common_random_numbers_across_checkpoints": True,
            "bank_seeds": list(bank_seeds),
            "steps": list(steps),
            "selection_rule": (
                "largest median four-metric reference-error-removal score"
                if args.task == "heston"
                else "largest median across banks of the unweighted mean of "
                "distribution- and dependence-median reference-error removal"
            ),
            "aggregates": aggregates,
            "selected_step": int(selected_step),
            "checkpoint_sha256": {
                str(step): hashes_before[step] for step in steps
            },
            "checkpoint_nonmutation": True,
            "banks": records,
        },
    )
    print(f"selected step {selected_step}", flush=True)


if __name__ == "__main__":
    main()
