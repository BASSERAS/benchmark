from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteMPNestedTrainingConfig,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
    fit_local_gaussian_reference_kernel,
)
from deep_mkv_gen_path_dt.ce import RidgeConditionalExpectation  # noqa: E402
from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl  # noqa: E402
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from path_dt_experiments.discrepancies import (  # noqa: E402
    build_heston_discrepancy,
    fit_fixed_target_joint_prefix_future_volatility_mmd,
)
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.heston_mixture_discrepancy import (  # noqa: E402
    HestonMixtureCausalRidgeConditionalExpectation,
    fit_fixed_target_heston_mixture_summary_mmd,
)
from path_dt_experiments.runners import (  # noqa: E402
    ensure_run_dir,
    write_history_jsonl,
    write_json,
)


ARM_SUMMARY_CLIPPED = "summary_online_clipped_default_ce"
ARM_SUMMARY_UNCAPPED = "summary_online_uncapped_default_ce"
ARM_CAUSAL_CE = "summary_online_uncapped_causal_ce"
ARM_NESTED = "summary_nested_uncapped_causal_ce"
ARMS = (
    ARM_SUMMARY_CLIPPED,
    ARM_SUMMARY_UNCAPPED,
    ARM_CAUSAL_CE,
    ARM_NESTED,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled Deep-MKV-Gen retraining arms for the variable-parameter "
            "Heston mixture using label-free joint path summaries."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "heston_parameter_mixture_seed0_20260730"
        ),
    )
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument(
        "--max-training-paths",
        type=int,
        default=0,
        help="Optional deterministic prefix for smoke tests; zero uses the full training law.",
    )
    parser.add_argument("--training-steps", type=int, default=1_600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--bank-size", type=int, default=8_192)
    parser.add_argument("--sample-batch-size", type=int, default=8_192)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--path-derivative-backend",
        choices=("autograd", "analytical"),
        default="autograd",
        help=(
            "Compute discrepancy and running-cost path sources with PyTorch "
            "autograd or with the closed-form first-derivative backend."
        ),
    )
    parser.add_argument("--summary-weight", type=float, default=1.0)
    parser.add_argument("--clipped-grad-norm", type=float, default=5.0)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.6)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--nested-outer-steps", type=int, default=8)
    parser.add_argument("--nested-inner-steps", type=int, default=200)
    parser.add_argument("--nested-outer-batch-size", type=int, default=1024)
    parser.add_argument("--nested-outer-target-batch-size", type=int, default=1024)
    parser.add_argument("--nested-inner-batch-size", type=int, default=256)
    parser.add_argument("--nested-trust-fraction", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _default_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "runs" / f"heston_mixture_deep_mkv_ablation_{timestamp}"


def _load_manifest(root: Path) -> dict[str, object]:
    path = root / "data" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_log_paths(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    prices = np.asarray(np.load(path), dtype=np.float64)
    if prices.ndim != 2 or not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ValueError(f"{path} must contain finite positive price paths")
    log_paths = np.log(prices / prices[:, :1])
    return torch.from_numpy(log_paths).unsqueeze(-1).to(
        device=device,
        dtype=dtype,
    )


def _base_discrepancy(num_steps: int) -> CompositePathFunctionalDiscrepancy:
    return build_heston_discrepancy(
        num_steps=int(num_steps),
        include_acf=True,
        preset="old_fullv_w0p25",
        lambda_scale=50.0,
        kappa_scale=50.0,
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )


def _make_training(
    args: argparse.Namespace,
    *,
    clipped: bool,
    seed: int,
) -> DiscreteMPTrainingConfig:
    return DiscreteMPTrainingConfig(
        batch_size=int(args.batch_size),
        target_batch_size=int(args.target_batch_size),
        num_steps=int(args.training_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        lambda_scale=50.0,
        ce_target_mode="ridge",
        ridge_lambda=float(args.ridge_lambda),
        ce_crossfit_folds=1,
        target_preconditioner="none",
        noise_target_control_variate="none",
        grad_clip_norm=(
            float(args.clipped_grad_norm)
            if bool(clipped)
            else None
        ),
        log_every=int(args.log_every),
        seed=int(seed),
        observed_only=False,
        path_derivative_backend=str(args.path_derivative_backend),
    )


def _make_nested(
    args: argparse.Namespace,
    *,
    seed: int,
) -> DiscreteMPNestedTrainingConfig:
    return DiscreteMPNestedTrainingConfig(
        outer_steps=int(args.nested_outer_steps),
        inner_steps=int(args.nested_inner_steps),
        outer_batch_size=int(args.nested_outer_batch_size),
        outer_target_batch_size=int(args.nested_outer_target_batch_size),
        inner_batch_size=int(args.nested_inner_batch_size),
        fixed_point_probe_paths=min(512, int(args.nested_outer_batch_size)),
        log_every_outer=1,
        outer_objective_line_search=True,
        outer_relaxation_mode="adjoint_blocks",
        outer_max_backtracks=10,
        outer_backtrack_factor=0.5,
        outer_objective_tolerance=0.0,
        outer_block_coordinate_passes=2,
        outer_block_trust_fraction=float(args.nested_trust_fraction),
        seed=int(seed),
    )


def _build_model(
    *,
    args: argparse.Namespace,
    grid: DiscreteTimeGrid,
    control: SpecificEntropyDiagonalControl,
    discrepancy: CompositePathFunctionalDiscrepancy,
    ce_estimator,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> DiscreteMPModel:
    model = DiscreteMPModel(
        architecture=DiscreteMPArchitectureConfig(
            state_dim=1,
            noise_dim=1,
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
        ),
        grid=grid,
        dynamics_step=DriftVolatilityEulerStep(dt=grid.dt),
        control_map=control,
        discrepancy=discrepancy,
        ce_estimator=ce_estimator,
        init_seed=int(seed),
    )
    model.network.to(device=device, dtype=dtype)
    return model


def _progress(stage: str):
    def report(_model: DiscreteMPModel, metrics: dict[str, float]) -> None:
        print(
            "{stage} step={step:>4.0f} outer={outer:>2.0f} loss={loss:.5g} "
            "sigma={sigma:.4f} grad={grad:.3g} clip_scale={clip:.3f}".format(
                stage=stage,
                step=float(metrics.get("step", metrics.get("inner_step", 0.0))),
                outer=float(metrics.get("outer_step", 0.0)),
                loss=float(metrics.get("train_total_loss", float("nan"))),
                sigma=float(metrics.get("sigma_mean", float("nan"))),
                grad=float(metrics.get("grad_norm", float("nan"))),
                clip=float(metrics.get("grad_clip_scale", 1.0)),
            ),
            flush=True,
        )

    return report


def _sample_prices(
    model: DiscreteMPModel,
    *,
    num_paths: int,
    batch_size: int,
    seed: int,
    s0: float,
) -> np.ndarray:
    bank = np.empty(
        (int(num_paths), int(model.grid.num_steps) + 1),
        dtype=np.float32,
    )
    x0 = torch.zeros((1, 1), device=model.device, dtype=model.dtype)
    with torch.no_grad():
        for batch_index, start in enumerate(
            range(0, int(num_paths), int(batch_size))
        ):
            count = min(int(batch_size), int(num_paths) - start)
            sample = model.sample(
                num_paths=count,
                x0=x0,
                seed=derive_stream_seed(
                    base_seed=int(seed),
                    stream_offset=10_000 + batch_index,
                ),
            )
            log_paths = sample.paths[..., 0].detach().cpu().numpy()
            bank[start : start + count] = (
                float(s0) * np.exp(log_paths)
            ).astype(np.float32)
    return bank


def _evaluate(
    *,
    root: Path,
    generated: Path,
    output: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "experiments"
                / "scripts"
                / "evaluate_heston_parameter_mixture.py"
            ),
            "--train-data",
            str(root / "data" / "train.npy"),
            "--test-data",
            str(root / "data" / "disc.npy"),
            "--generated-data",
            str(generated),
            "--oracle",
            str(root / "oracle" / "oracle.joblib"),
            "--oracle-gate-report",
            str(root / "oracle" / "gate_report.json"),
            "--output",
            str(output),
        ],
        check=True,
    )


def _primary_metrics(payload: dict[str, object]) -> dict[str, float]:
    mixture = payload["mixture_fidelity"]
    parameters = mixture["parameters"]
    normalized_wasserstein = [
        float(parameters[name]["support_normalized_wasserstein"])
        for name in ("theta", "xi", "rho")
    ]
    observable = payload["observable_fidelity"]
    return {
        "regime_tvd": float(mixture["regime_proportion_tvd"]),
        "average_parameter_normalized_wasserstein": float(
            sum(normalized_wasserstein) / len(normalized_wasserstein)
        ),
        "theta_normalized_wasserstein": normalized_wasserstein[0],
        "xi_normalized_wasserstein": normalized_wasserstein[1],
        "rho_normalized_wasserstein": normalized_wasserstein[2],
        "realized_volatility_wasserstein": float(
            observable["realized_volatility_wasserstein"]
        ),
        "leverage_curve_rmse": float(
            observable["leverage_curve_rmse_lags_0_20"]
        ),
        "terminal_log_price_ks": float(observable["terminal_log_price_ks"]),
        "abs_return_acf_rmse": float(
            observable["abs_return_acf_rmse_lags_1_50"]
        ),
        "squared_return_acf_rmse": float(
            observable["squared_return_acf_rmse_lags_1_50"]
        ),
    }


def main() -> None:
    args = _parse_args()
    root = Path(args.experiment_root)
    source_run = (
        Path(args.source_run_dir)
        if args.source_run_dir is not None
        else root / "deep_mkv"
    )
    source_checkpoint_path = source_run / "source_model_checkpoint.pt"
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(source_checkpoint_path)
    run_dir = ensure_run_dir(args.run_dir or _default_run_dir())
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    manifest = _load_manifest(root)
    num_steps = int(manifest["sequence_length"]) - 1
    dt = float(manifest["dt"])
    s0 = float(manifest["s0"])
    grid = DiscreteTimeGrid(T=num_steps * dt, num_steps=num_steps)
    target_paths = _load_log_paths(
        root / "data" / "train.npy",
        device=device,
        dtype=dtype,
    )
    if int(args.max_training_paths) > 0:
        if int(args.max_training_paths) < 2:
            raise ValueError("max-training-paths must be zero or at least two")
        target_paths = target_paths[: int(args.max_training_paths)]

    reference = fit_local_gaussian_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        ridge=float(args.ridge_lambda),
        variance_ridge=float(args.ridge_lambda),
        variance_shrinkage=0.1,
        log_variance_bias_correction=0.6,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    control = SpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=float(args.eta),
        reference_kernel=reference,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    base = _base_discrepancy(num_steps)
    joint = fit_fixed_target_joint_prefix_future_volatility_mmd(
        target_paths,
        grid=grid,
        split_index=64,
        future_horizon=32,
        prefix_windows=(16, 32, 64),
        recent_return_window=16,
        bandwidths=(0.5, 1.0, 2.0),
    )
    summary_discrepancy = fit_fixed_target_heston_mixture_summary_mmd(
        target_paths,
        grid=grid,
    )
    discrepancy = CompositePathFunctionalDiscrepancy(
        blocks=tuple(base.blocks)
        + (
            WeightedDiscrepancy(
                name="volatility_law_joint_prefix_future_rv",
                weight=1.0,
                discrepancy=joint,
            ),
            WeightedDiscrepancy(
                name="mixture_law_joint_path_summaries",
                weight=float(args.summary_weight),
                discrepancy=summary_discrepancy,
            ),
        )
    )
    default_ce = RidgeConditionalExpectation(ridge=float(args.ridge_lambda))
    causal_ce = HestonMixtureCausalRidgeConditionalExpectation(
        grid=grid,
        ridge=float(args.ridge_lambda),
    )
    source_checkpoint = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    baseline_validation_path = run_dir / "baseline_validation_metrics.json"
    if not baseline_validation_path.exists():
        _evaluate(
            root=root,
            generated=root / "deep_mkv" / "generated_paths_8192x128.npy",
            output=baseline_validation_path,
        )
    baseline_validation = json.loads(
        baseline_validation_path.read_text(encoding="utf-8")
    )
    write_json(
        run_dir / "design.json",
        {
            "source_checkpoint": str(source_checkpoint_path.resolve()),
            "target_training_paths": str(
                (root / "data" / "train.npy").resolve()
            ),
            "target_training_path_count": int(target_paths.shape[0]),
            "validation_paths": str((root / "data" / "disc.npy").resolve()),
            "test_paths_used_for_selection": False,
            "running_cost": "specific_entropy",
            "reference": reference.summary(),
            "network_regression_loss_changed": False,
            "summary_discrepancy_weight": float(args.summary_weight),
            "summary_discrepancy": summary_discrepancy.summary(),
            "causal_ce_feature_names": list(causal_ce.feature_names),
            "baseline_validation_primary": _primary_metrics(
                baseline_validation
            ),
        },
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    nested = _make_nested(args, seed=int(args.seed) + 1)
    results: dict[str, dict[str, object]] = {}
    for arm in tuple(str(value) for value in args.arms):
        arm_dir = ensure_run_dir(run_dir / arm)
        metrics_path = arm_dir / "validation_metrics.json"
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            results[arm] = payload
            print(f"reusing completed arm {arm}", flush=True)
            continue
        clipped = arm == ARM_SUMMARY_CLIPPED
        use_causal_ce = arm in {ARM_CAUSAL_CE, ARM_NESTED}
        training = _make_training(
            args,
            clipped=clipped,
            seed=int(args.seed) + 1,
        )
        model = _build_model(
            args=args,
            grid=grid,
            control=control,
            discrepancy=discrepancy,
            ce_estimator=(causal_ce if use_causal_ce else default_ce),
            device=device,
            dtype=dtype,
            seed=int(args.seed),
        )
        model.load_checkpoint_state(source_checkpoint)
        if arm == ARM_NESTED:
            fit = model.fit_nested(
                target_paths,
                training=training,
                nested=nested,
                log_callback=_progress(arm),
            )
            solver = "nested_forward_backward"
        else:
            fit = model.fit(
                target_paths,
                training=training,
                log_callback=_progress(arm),
            )
            solver = "online"
        checkpoint_path = arm_dir / "model_checkpoint.pt"
        torch.save(model.checkpoint_state(), checkpoint_path)
        write_history_jsonl(arm_dir / "training_history.jsonl", fit.history)
        bank_path = (
            arm_dir
            / f"generated_paths_{int(args.bank_size)}x{num_steps + 1}.npy"
        )
        bank = _sample_prices(
            model,
            num_paths=int(args.bank_size),
            batch_size=int(args.sample_batch_size),
            seed=int(args.seed),
            s0=s0,
        )
        np.save(bank_path, bank)
        _evaluate(root=root, generated=bank_path, output=metrics_path)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        results[arm] = payload
        write_json(
            arm_dir / "training_manifest.json",
            {
                "arm": arm,
                "solver": solver,
                "gradient_clipping": training.grad_clip_norm,
                "conditional_expectation": (
                    "heston_mixture_causal_observable_basis"
                    if use_causal_ce
                    else "raw_prefix_ridge"
                ),
                "running_cost": "specific_entropy",
                "network_regression_loss_changed": False,
                "training": asdict(training),
                "nested_training": (
                    asdict(nested) if arm == ARM_NESTED else None
                ),
                "summary_discrepancy": summary_discrepancy.summary(),
                "fit_final": fit.final_metrics,
                "validation_primary": _primary_metrics(payload),
                "test_paths_used_for_selection": False,
            },
        )
        del model, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison = {
        "baseline": _primary_metrics(baseline_validation),
        **{
            arm: _primary_metrics(payload)
            for arm, payload in results.items()
        },
    }
    write_json(run_dir / "validation_summary.json", comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
