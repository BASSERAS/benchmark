#!/usr/bin/env python3
"""Select a mixture Guyon Deep-MKV checkpoint on replay-only validation banks."""

from __future__ import annotations

import argparse
import json
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
    fit_guyon_lekeufack_likelihood_reference_kernel,
)
from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    VolatilityOnlySpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from evaluate_guyon_checkpoint_confirmation_banks import (  # noqa: E402
    checkpoint_paths,
    relative_gain,
    sha256,
    zero_network,
)
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402
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


MIXTURE_ROOT = REPO_ROOT / "runs" / "heston_parameter_mixture_seed0_20260730"
SELECTION_METRICS = (
    "path_swd_normalized",
    "realized_volatility_w1_normalized",
    "absolute_return_acf_rmse",
    "regime_proportion_tvd",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--bank-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    return parser.parse_args()


def build_model(device: torch.device):
    target = _load_log_paths(
        MIXTURE_ROOT / "data" / "train.npy",
        device=device,
        dtype=torch.float32,
    )
    validation = np.asarray(
        np.load(MIXTURE_ROOT / "data" / "disc.npy", allow_pickle=False),
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
    base = build_heston_discrepancy(
        num_steps=num_steps,
        include_acf=True,
        preset="old_fullv_w0p25",
        lambda_scale=50.0,
        kappa_scale=100.0,
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )
    mixture_block = WeightedDiscrepancy(
        name="mixture_law_joint_path_summaries",
        weight=1.0,
        discrepancy=fit_fixed_target_heston_mixture_summary_mmd(target, grid=grid),
    )
    discrepancy = CompositePathFunctionalDiscrepancy(
        blocks=tuple((*base.blocks, mixture_block))
    )
    control = VolatilityOnlySpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=2.0,
        reference_kernel=reference,
        sigma_min=1e-3,
        sigma_max=0.6,
    )
    estimator = HestonMixtureCausalRidgeConditionalExpectation(
        grid=grid,
        ridge=1e-3,
    )
    model = _model(
        grid=grid,
        control=control,
        discrepancy=discrepancy,
        estimator=estimator,
        device=device,
        seed=0,
        adjoint_network="gru",
        transformer_layers=2,
        transformer_heads=4,
    )
    return model, validation


def metric_vector(common: dict[str, object], mixture: dict[str, object]) -> dict[str, float]:
    path = common["path_law"]
    return {
        "path_swd_normalized": float(path["path_swd_normalized"]),
        "realized_volatility_w1_normalized": float(
            path["realized_volatility_w1_normalized"]
        ),
        "absolute_return_acf_rmse": float(path["absolute_return_acf_rmse"]),
        "regime_proportion_tvd": float(
            mixture["mixture_fidelity"]["regime_proportion_tvd"]
        ),
    }


def score(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    gains = {
        key: relative_gain(reference[key], candidate[key])
        for key in SELECTION_METRICS
    }
    return {"metric_gains": gains, "score": float(np.mean(list(gains.values())))}


def evaluate(
    *,
    model,
    validation: np.ndarray,
    output_dir: Path,
    label: str,
    bank_seed: int,
    bank_size: int,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, object]]:
    bank, controls = _sample(
        model,
        num_paths=bank_size,
        batch_size=batch_size,
        seed=bank_seed,
    )
    bank_path = output_dir / f"{label}_bank.npy"
    np.save(bank_path, bank)
    common = _evaluate_common(
        target_prices=validation,
        generated=bank,
        device=model.device,
        seed=bank_seed,
    )
    mixture_path = output_dir / f"{label}_mixture_metrics.json"
    _write_mixture_metrics(MIXTURE_ROOT, bank_path, mixture_path)
    mixture = json.loads(mixture_path.read_text(encoding="utf-8"))
    write_json(output_dir / f"{label}_metrics.json", common)
    write_json(output_dir / f"{label}_controls.json", controls)
    return metric_vector(common, mixture), {
        "common_metrics": str(output_dir / f"{label}_metrics.json"),
        "mixture_metrics": str(mixture_path),
        "bank": str(bank_path),
    }


def main() -> None:
    args = parse_args()
    steps = tuple(sorted(set(int(value) for value in args.steps)))
    bank_seeds = tuple(int(value) for value in args.bank_seeds)
    if len(bank_seeds) < 2 or len(set(bank_seeds)) != len(bank_seeds):
        raise ValueError("bank-seeds must contain at least two distinct values")
    run_dir = args.run_dir.resolve()
    output_dir = ensure_run_dir(args.output_dir.resolve())
    checkpoints = checkpoint_paths("heston", run_dir, steps)
    hashes_before = {step: sha256(path) for step, path in checkpoints.items()}
    device = torch.device(args.device)
    model, validation = build_model(device)
    zero_network(model)
    reference_checkpoint = model.checkpoint_state()

    records: dict[str, object] = {}
    scores_by_step: dict[int, list[float]] = {step: [] for step in steps}
    for bank_seed in bank_seeds:
        bank_dir = ensure_run_dir(output_dir / f"bank_{bank_seed}")
        model.load_checkpoint_state(reference_checkpoint)
        reference_vector, reference_files = evaluate(
            model=model,
            validation=validation,
            output_dir=bank_dir,
            label="reference",
            bank_seed=bank_seed,
            bank_size=int(args.bank_size),
            batch_size=int(args.sample_batch_size),
        )
        bank_record: dict[str, object] = {
            "reference_selection_metrics": reference_vector,
            "reference_files": reference_files,
            "steps": {},
        }
        for step in steps:
            checkpoint = torch.load(
                checkpoints[step], map_location=device, weights_only=True
            )
            model.load_checkpoint_state(checkpoint)
            vector, files = evaluate(
                model=model,
                validation=validation,
                output_dir=bank_dir,
                label=f"step_{step:04d}",
                bank_seed=bank_seed,
                bank_size=int(args.bank_size),
                batch_size=int(args.sample_batch_size),
            )
            scored = score(reference_vector, vector)
            bank_record["steps"][str(step)] = {
                "selection_metrics": vector,
                "files": files,
                **scored,
            }
            scores_by_step[step].append(float(scored["score"]))
            print(
                f"mixture bank={bank_seed} step={step} "
                f"score={float(scored['score']):.6f}",
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
            "task": "mixture",
            "selection_scope": "validation_only",
            "canonical_test_split_loaded": False,
            "training_performed": False,
            "model_update_performed": False,
            "reference_kind": "guyon_lekeufack_structural_likelihood",
            "common_random_numbers_across_checkpoints": True,
            "bank_seeds": list(bank_seeds),
            "steps": list(steps),
            "selection_metrics": list(SELECTION_METRICS),
            "selection_rule": (
                "largest median across banks of the equal-weight mean "
                "reference-error removal over the four paper-facing metrics"
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
