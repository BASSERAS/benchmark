from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
    fit_local_gaussian_reference_kernel,
)
from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl  # noqa: E402
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from path_dt_experiments.data import HestonConfig  # noqa: E402
from path_dt_experiments.discrepancies import (  # noqa: E402
    FixedTargetJointPrefixFutureVolatilityMMD,
    build_heston_discrepancy,
    fit_fixed_target_joint_prefix_future_volatility_mmd,
    fit_fixed_target_prefix_regime_future_volatility_claims,
)
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.metrics import stylized_fact_metrics  # noqa: E402
from path_dt_experiments.runners import (  # noqa: E402
    default_run_dir,
    ensure_run_dir,
    write_history_jsonl,
    write_json,
)
from path_dt_experiments.shadowing import (  # noqa: E402
    ShadowingResult,
    evaluate_path_shadowing,
    paired_bootstrap_shadowing_difference,
)
from run_heston import sample_model_paths_in_batches  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a specific-entropy Heston generator with one fixed, "
            "target-standardized joint prefix/future realized-volatility MMD "
            "and select its checkpoint on validation shadow RV CRPS."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721"),
    )
    parser.add_argument(
        "--validation-paths",
        type=Path,
        default=Path("runs/heston_dt_entropy_barrier_k25_baseline_seed4321_20260717/heldout_paths.pt"),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--split-step", type=int, default=64)
    parser.add_argument("--future-steps", type=int, default=32)
    parser.add_argument("--prefix-windows", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--recent-return-window", type=int, default=16)
    parser.add_argument("--bandwidths", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--validation-real-paths", type=int, default=512)
    parser.add_argument("--validation-bank-paths", type=int, default=4096)
    parser.add_argument("--test-real-paths", type=int, default=512)
    parser.add_argument("--test-bank-paths", type=int, default=4096)
    parser.add_argument("--sample-batch-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--shadow-search", choices=("exact", "faiss"), default="exact")
    parser.add_argument("--shadow-exact-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=24001307)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} must contain a tensor")
    return value.detach().cpu()


def _stream_seed(base_seed: int, offset: int) -> int:
    value = derive_stream_seed(base_seed=int(base_seed), stream_offset=int(offset))
    return int(value if value is not None else int(base_seed) + int(offset))


def _observed_indices(*, num_steps: int, spacing: int) -> tuple[int, ...]:
    if int(spacing) < 1 or int(num_steps) % int(spacing) != 0:
        raise ValueError("stored observed_every must divide the number of Heston steps")
    return tuple(range(0, int(num_steps) + 1, int(spacing)))


def _observed_discrepancy_law(
    paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
) -> tuple[torch.Tensor, DiscreteTimeGrid]:
    indices = grid.observed_point_indices(device=torch.device("cpu"))
    observed_times = grid.time_points(device=torch.device("cpu"), dtype=torch.float64).index_select(
        0,
        indices,
    )
    observed_grid = DiscreteTimeGrid(
        T=float(grid.T),
        num_steps=int(grid.observed_count) - 1,
        point_times=tuple(float(value) for value in observed_times.tolist()),
        allow_nonuniform=True,
    )
    return paths.index_select(1, indices), observed_grid


def _mapped_steps(values: tuple[int, ...], *, observed_every: int, name: str) -> tuple[int, ...]:
    if len(values) == 0 or any(int(value) < 1 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if tuple(sorted(set(int(value) for value in values))) != values:
        raise ValueError(f"{name} must be strictly increasing without duplicates")
    if any(int(value) % int(observed_every) != 0 for value in values):
        raise ValueError(f"every {name} value must be divisible by observed_every")
    return tuple(int(value) // int(observed_every) for value in values)


def _base_discrepancy(
    *,
    stored_config: dict[str, object],
    training: DiscreteMPTrainingConfig,
    grid: DiscreteTimeGrid,
) -> CompositePathFunctionalDiscrepancy:
    if bool(stored_config.get("conditional_volatility_discrepancy", False)) or bool(
        stored_config.get("conditional_volatility_correlation_discrepancy", False)
    ):
        raise ValueError("the pilot requires a source without an existing conditional-volatility block")
    stored_endpoints = stored_config.get("time_marginal_volatility_endpoints", ())
    if not isinstance(stored_endpoints, (list, tuple)):
        raise ValueError("stored time-marginal endpoints must be a sequence")
    discrepancy_steps = (
        int(grid.observed_count) - 1 if bool(training.observed_only) else int(grid.num_steps)
    )
    return build_heston_discrepancy(
        num_steps=discrepancy_steps,
        include_acf=bool(stored_config["include_acf"]),
        preset=str(stored_config["preset"]),
        lambda_scale=float(stored_config["lambda_scale"]),
        kappa_scale=float(stored_config["kappa_scale"]),
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
        time_marginal_volatility_window=(
            None
            if stored_config.get("time_marginal_volatility_window") is None
            else int(stored_config["time_marginal_volatility_window"])
        ),
        time_marginal_volatility_endpoints=tuple(int(value) for value in stored_endpoints),
        time_marginal_volatility_weight=float(
            stored_config.get("time_marginal_volatility_weight", 1.0)
        ),
    )


def _build_models(
    *,
    source: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[
    DiscreteMPModel,
    DiscreteMPModel,
    DiscreteMPTrainingConfig,
    dict[str, object],
    HestonConfig,
    torch.Tensor,
    torch.Tensor,
    FixedTargetJointPrefixFutureVolatilityMMD,
    dict[str, int | tuple[int, ...]],
]:
    metrics = _load_json(source / "metrics.json")
    stored_config = metrics.get("config")
    if not isinstance(stored_config, dict):
        raise ValueError("source metrics.json is missing config")
    control_payload = stored_config.get("control_map")
    if not isinstance(control_payload, dict) or control_payload.get("name") != "specific_entropy":
        raise ValueError("source checkpoint must use the specific-entropy control")
    checkpoint = torch.load(source / "model_checkpoint.pt", map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("source checkpoint must be a dictionary")
    architecture_payload = checkpoint.get("architecture")
    training_payload = checkpoint.get("training")
    if not isinstance(architecture_payload, dict) or not isinstance(training_payload, dict):
        raise ValueError("source checkpoint is missing architecture or training")
    architecture = DiscreteMPArchitectureConfig(**architecture_payload)
    training = DiscreteMPTrainingConfig(**training_payload)
    if not bool(training.observed_only):
        raise ValueError("this controlled pilot expects the source's observed-only discrepancy grid")
    if int(args.eval_every) < 1 or int(args.eval_every) % int(training.log_every) != 0:
        raise ValueError("eval-every must be a positive multiple of the source log_every")
    if not math.isfinite(float(args.joint_weight)) or float(args.joint_weight) <= 0.0:
        raise ValueError("joint-weight must be a positive finite float")

    heston_payload = stored_config.get("heston")
    if not isinstance(heston_payload, dict):
        raise ValueError("source config is missing Heston parameters")
    heston = HestonConfig(**heston_payload)
    observed_every = int(stored_config["observed_every"])
    grid = DiscreteTimeGrid(
        T=float(heston.T),
        num_steps=int(heston.num_steps),
        observed_indices=_observed_indices(
            num_steps=int(heston.num_steps),
            spacing=observed_every,
        ),
    )
    target_paths = _load_tensor(source / "target_paths.pt")
    heldout_paths = _load_tensor(source / "heldout_paths.pt")
    dtype = torch.float64 if str(stored_config["dtype"]) == "float64" else torch.float32
    target_device = target_paths.to(device=device, dtype=dtype)
    reference_payload = control_payload.get("reference")
    if not isinstance(reference_payload, dict):
        raise ValueError("source specific-entropy control is missing reference metadata")
    reference = fit_local_gaussian_reference_kernel(
        target_paths=target_device,
        grid=grid,
        ridge=float(reference_payload["reference_ridge"]),
        variance_ridge=float(reference_payload["reference_variance_ridge"]),
        variance_shrinkage=float(reference_payload["reference_variance_shrinkage"]),
        log_variance_bias_correction=float(
            reference_payload["reference_log_variance_bias_correction"]
        ),
        sigma_min=float(stored_config["sigma_min"]),
        sigma_max=(
            None
            if reference_payload.get("reference_sigma_max") is None
            else float(reference_payload["reference_sigma_max"])
        ),
    )
    control = SpecificEntropyDiagonalControl(
        dt=grid.dt,
        eta=float(control_payload["eta"]),
        reference_kernel=reference,
        sigma_min=float(stored_config["sigma_min"]),
        sigma_max=float(stored_config["sigma_max"]),
    )
    base = _base_discrepancy(stored_config=stored_config, training=training, grid=grid)
    target_observed, discrepancy_grid = _observed_discrepancy_law(target_paths, grid=grid)
    mapped_prefix_windows = _mapped_steps(
        tuple(int(value) for value in args.prefix_windows),
        observed_every=observed_every,
        name="prefix-windows",
    )
    mapped_split = _mapped_steps(
        (int(args.split_step),),
        observed_every=observed_every,
        name="split-step",
    )[0]
    mapped_future = _mapped_steps(
        (int(args.future_steps),),
        observed_every=observed_every,
        name="future-steps",
    )[0]
    mapped_recent = _mapped_steps(
        (int(args.recent_return_window),),
        observed_every=observed_every,
        name="recent-return-window",
    )[0]
    joint = fit_fixed_target_joint_prefix_future_volatility_mmd(
        target_observed,
        grid=discrepancy_grid,
        split_index=int(mapped_split),
        future_horizon=int(mapped_future),
        prefix_windows=mapped_prefix_windows,
        recent_return_window=int(mapped_recent),
        bandwidths=tuple(float(value) for value in args.bandwidths),
    )
    regime_weight = float(getattr(args, "regime_weight", 0.0))
    if not math.isfinite(regime_weight) or regime_weight < 0.0:
        raise ValueError("regime-weight must be a nonnegative finite float")
    additional_blocks = []
    if regime_weight > 0.0:
        regime_prefix_window_full = int(getattr(args, "regime_prefix_window", 16))
        mapped_regime_prefix_window = _mapped_steps(
            (regime_prefix_window_full,),
            observed_every=observed_every,
            name="regime-prefix-window",
        )[0]
        regime_claims = fit_fixed_target_prefix_regime_future_volatility_claims(
            target_observed,
            grid=discrepancy_grid,
            split_index=int(mapped_split),
            prefix_window=int(mapped_regime_prefix_window),
            future_horizon=int(mapped_future),
            gate_temperature_fraction=float(
                getattr(args, "regime_gate_temperature_fraction", 0.25)
            ),
            tail_temperature=float(getattr(args, "regime_tail_temperature", 0.15)),
            conditional_moment_weight=float(
                getattr(args, "regime_conditional_moment_weight", 0.0)
            ),
        )
        additional_blocks.append(
            WeightedDiscrepancy(
                name="volatility_law_prefix_regime_future_rv",
                weight=regime_weight,
                discrepancy=regime_claims,
            )
        )
    else:
        mapped_regime_prefix_window = None
    candidate_discrepancy = CompositePathFunctionalDiscrepancy(
        blocks=tuple(base.blocks)
        + (
            WeightedDiscrepancy(
                name="volatility_law_joint_prefix_future_rv",
                weight=float(args.joint_weight),
                discrepancy=joint,
            ),
        )
        + tuple(additional_blocks)
    )
    source_model = DiscreteMPModel(
        architecture=architecture,
        grid=grid,
        dynamics_step=DriftVolatilityEulerStep(dt=grid.dt),
        control_map=control,
        discrepancy=base,
    )
    source_model.network.to(device=device, dtype=dtype)
    source_model.load_checkpoint_state(checkpoint)
    candidate_model = DiscreteMPModel(
        architecture=architecture,
        grid=grid,
        dynamics_step=DriftVolatilityEulerStep(dt=grid.dt),
        control_map=control,
        discrepancy=candidate_discrepancy,
        init_seed=int(args.seed),
    )
    candidate_model.network.to(device=device, dtype=dtype)
    mapped = {
        "split_index": int(mapped_split),
        "future_horizon": int(mapped_future),
        "prefix_windows": mapped_prefix_windows,
        "recent_return_window": int(mapped_recent),
        "regime_prefix_window": mapped_regime_prefix_window,
    }
    return (
        source_model,
        candidate_model,
        training,
        stored_config,
        heston,
        target_paths,
        heldout_paths,
        joint,
        mapped,
    )


def _sample_bank(
    model: DiscreteMPModel,
    *,
    num_paths: int,
    seed: int,
    x0: float,
    batch_size: int,
) -> torch.Tensor:
    with torch.no_grad():
        return sample_model_paths_in_batches(
            model,
            num_paths=int(num_paths),
            x0=torch.tensor([[float(x0)]], device=model.device, dtype=model.dtype),
            seed=int(seed),
            batch_size=min(int(batch_size), int(num_paths)),
        )


def _shadow(
    *,
    real_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    args: argparse.Namespace,
) -> ShadowingResult:
    return evaluate_path_shadowing(
        real_paths=real_paths,
        generated_paths=generated_paths,
        prefix_length=int(args.split_step) + 1,
        top_k=min(int(args.top_k), int(generated_paths.shape[0])),
        future_horizon=int(args.future_steps),
        search_backend=str(args.shadow_search),
        exact_batch_size=int(args.shadow_exact_batch_size),
    )


def _metric_subset(result: ShadowingResult) -> dict[str, float | str]:
    names = (
        "prefix_distance_mean",
        "cumulative_return_crps",
        "cumulative_return_mean_rmse",
        "cumulative_return_coverage_90",
        "cumulative_return_band_width_90",
        "realized_volatility_crps",
        "realized_volatility_mean_rmse",
        "realized_volatility_coverage_90",
        "realized_volatility_band_width_90",
    )
    return {name: result.metrics[name] for name in names}


def _joint_metrics(
    discrepancy: FixedTargetJointPrefixFutureVolatilityMMD,
    *,
    generated_paths: torch.Tensor,
    target_paths: torch.Tensor,
    grid: DiscreteTimeGrid,
    max_paths: int = 1024,
) -> dict[str, float]:
    generated_observed, discrepancy_grid = _observed_discrepancy_law(generated_paths, grid=grid)
    target_observed, _ = _observed_discrepancy_law(target_paths, grid=grid)
    count = min(int(max_paths), int(generated_observed.shape[0]), int(target_observed.shape[0]))
    return discrepancy(
        generated_paths=generated_observed[:count],
        target_paths=target_observed[:count],
        grid=discrepancy_grid,
    ).metrics


def main() -> None:
    args = parse_args()
    source = Path(args.source_run_dir)
    run_dir = ensure_run_dir(
        args.run_dir or default_run_dir(prefix="heston_dt_specific_entropy_joint_shadow")
    )
    if int(args.validation_real_paths) < 1 or int(args.test_real_paths) < 1:
        raise ValueError("validation-real-paths and test-real-paths must be >= 1")
    if int(args.validation_bank_paths) < int(args.top_k) or int(args.test_bank_paths) < int(
        args.top_k
    ):
        raise ValueError("each generated bank must contain at least top-k paths")
    if int(args.bootstrap_replicates) < 2:
        raise ValueError("bootstrap-replicates must be >= 2")
    if not 0.0 < float(args.bootstrap_confidence) < 1.0:
        raise ValueError("bootstrap-confidence must be in (0, 1)")

    (
        source_model,
        candidate_model,
        training,
        stored_config,
        heston,
        target_paths,
        heldout_paths,
        joint_discrepancy,
        mapped,
    ) = _build_models(source=source, device=torch.device(args.device), args=args)
    validation_paths_all = _load_tensor(Path(args.validation_paths))
    if tuple(validation_paths_all.shape[1:]) != tuple(target_paths.shape[1:]):
        raise ValueError("validation path shape must match the source target law")
    validation_paths = validation_paths_all[: int(args.validation_real_paths)]
    test_paths = heldout_paths[: int(args.test_real_paths)]
    x0 = float(heston.x0)
    validation_bank_seed = _stream_seed(int(args.seed), 80_000)
    test_bank_seed = int(stored_config.get("sample_seed", _stream_seed(int(args.seed), 80_001)))

    print("sampling fixed validation bank for the source baseline", flush=True)
    baseline_validation_bank = _sample_bank(
        source_model,
        num_paths=int(args.validation_bank_paths),
        seed=int(validation_bank_seed),
        x0=x0,
        batch_size=int(args.sample_batch_size),
    )
    baseline_validation = _shadow(
        real_paths=validation_paths,
        generated_paths=baseline_validation_bank,
        args=args,
    )

    selection_history: list[dict[str, object]] = []
    best_score = float("inf")
    best_step: int | None = None
    best_checkpoint: dict[str, object] | None = None
    best_validation_result: ShadowingResult | None = None
    best_validation_bank: torch.Tensor | None = None

    def validate_checkpoint(model: DiscreteMPModel, train_metrics: dict[str, float]) -> None:
        nonlocal best_score, best_step, best_checkpoint, best_validation_result, best_validation_bank
        step = int(train_metrics["step"])
        if step % int(args.eval_every) != 0 and step != int(training.num_steps):
            return
        bank = _sample_bank(
            model,
            num_paths=int(args.validation_bank_paths),
            seed=int(validation_bank_seed),
            x0=x0,
            batch_size=int(args.sample_batch_size),
        )
        result = _shadow(real_paths=validation_paths, generated_paths=bank, args=args)
        score = float(result.metrics["realized_volatility_crps"])
        row: dict[str, object] = {
            "step": step,
            "selection_metric": "realized_volatility_crps",
            "selection_score": score,
            "shadowing": _metric_subset(result),
            "training_joint_weighted_value": float(
                train_metrics.get(
                    "volatility_law_joint_prefix_future_rv_weighted_value",
                    float("nan"),
                )
            ),
            "training_joint_future_rv_std_ratio": float(
                train_metrics.get(
                    "volatility_law_joint_prefix_future_rv_"
                    "joint_prefix_future_volatility_future_rv_std_ratio",
                    float("nan"),
                )
            ),
        }
        selection_history.append(row)
        selected = score < best_score
        if selected:
            best_score = score
            best_step = step
            best_checkpoint = model.checkpoint_state()
            best_validation_result = result
            best_validation_bank = bank.clone()
        print(
            "validation step={step} rv_crps={crps:.6f} rv_cov90={coverage:.4f} "
            "return_crps={return_crps:.6f}{marker}".format(
                step=step,
                crps=score,
                coverage=float(result.metrics["realized_volatility_coverage_90"]),
                return_crps=float(result.metrics["cumulative_return_crps"]),
                marker=" selected" if selected else "",
            ),
            flush=True,
        )

    print("training the joint-MMD candidate from the source initialization", flush=True)
    fit_result = candidate_model.fit(
        target_paths,
        training=training,
        log_callback=validate_checkpoint,
    )
    terminal_checkpoint = candidate_model.checkpoint_state()
    torch.save(terminal_checkpoint, run_dir / "terminal_model_checkpoint.pt")
    if best_checkpoint is None or best_step is None or best_validation_result is None:
        raise RuntimeError("validation selection produced no candidate checkpoint")
    candidate_model.load_checkpoint_state(best_checkpoint)
    torch.save(best_checkpoint, run_dir / "selected_model_checkpoint.pt")

    if best_validation_bank is None:
        best_validation_bank = _sample_bank(
            candidate_model,
            num_paths=int(args.validation_bank_paths),
            seed=int(validation_bank_seed),
            x0=x0,
            batch_size=int(args.sample_batch_size),
        )
    baseline_test_artifact = source / "generated_paths.pt"
    if baseline_test_artifact.exists():
        baseline_test_bank = _load_tensor(baseline_test_artifact)[: int(args.test_bank_paths)]
    else:
        baseline_test_bank = _sample_bank(
            source_model,
            num_paths=int(args.test_bank_paths),
            seed=int(test_bank_seed),
            x0=x0,
            batch_size=int(args.sample_batch_size),
        )
    if int(baseline_test_bank.shape[0]) < int(args.test_bank_paths):
        raise ValueError("source generated-path artifact is smaller than test-bank-paths")
    candidate_test_bank = _sample_bank(
        candidate_model,
        num_paths=int(args.test_bank_paths),
        seed=int(test_bank_seed),
        x0=x0,
        batch_size=int(args.sample_batch_size),
    )
    baseline_test = _shadow(real_paths=test_paths, generated_paths=baseline_test_bank, args=args)
    candidate_test = _shadow(real_paths=test_paths, generated_paths=candidate_test_bank, args=args)
    paired = paired_bootstrap_shadowing_difference(
        candidate_test,
        baseline_test,
        num_replicates=int(args.bootstrap_replicates),
        confidence_level=float(args.bootstrap_confidence),
        seed=int(args.bootstrap_seed),
    )

    baseline_metrics = _metric_subset(baseline_test)
    candidate_metrics = _metric_subset(candidate_test)
    rv_crps_improved = float(candidate_metrics["realized_volatility_crps"]) < float(
        baseline_metrics["realized_volatility_crps"]
    )
    rv_coverage_improved = float(candidate_metrics["realized_volatility_coverage_90"]) > float(
        baseline_metrics["realized_volatility_coverage_90"]
    )
    return_crps_not_worse = float(candidate_metrics["cumulative_return_crps"]) <= float(
        baseline_metrics["cumulative_return_crps"]
    )
    lags = tuple(value for value in (1, 2, 5, 10, 20) if value < int(heston.num_steps))
    baseline_stylized = stylized_fact_metrics(
        baseline_test_bank,
        heldout_paths,
        lags=lags,
        tail_quantiles=(0.90, 0.95, 0.99),
        include_swd=True,
        swd_projections=128,
    )
    candidate_stylized = stylized_fact_metrics(
        candidate_test_bank,
        heldout_paths,
        lags=lags,
        tail_quantiles=(0.90, 0.95, 0.99),
        include_swd=True,
        swd_projections=128,
    )
    shadowing_gate = {
        "rv_crps_improved": bool(rv_crps_improved),
        "rv_coverage_improved": bool(rv_coverage_improved),
        "return_crps_not_worse": bool(return_crps_not_worse),
        "passes_all": bool(rv_crps_improved and rv_coverage_improved and return_crps_not_worse),
    }
    stylized_gate = {
        "path_swd_within_10_percent": bool(
            float(candidate_stylized["path_swd"])
            <= 1.10 * float(baseline_stylized["path_swd"])
        ),
        "abs_return_acf_error_not_worse": bool(
            float(candidate_stylized["abs_return_acf_error_rms"])
            <= float(baseline_stylized["abs_return_acf_error_rms"])
        ),
        "squared_return_acf_error_not_worse": bool(
            float(candidate_stylized["squared_return_acf_error_rms"])
            <= float(baseline_stylized["squared_return_acf_error_rms"])
        ),
        "tail_survival_error_not_worse": bool(
            float(candidate_stylized["tail_survival_error_rms"])
            <= float(baseline_stylized["tail_survival_error_rms"])
        ),
        "excess_kurtosis_error_not_worse": bool(
            float(candidate_stylized["excess_kurtosis_error_rms"])
            <= float(baseline_stylized["excess_kurtosis_error_rms"])
        ),
    }
    stylized_gate["passes_all"] = bool(all(stylized_gate.values()))
    payload: dict[str, object] = {
        "run_dir": str(run_dir),
        "source_run_dir": str(source),
        "validation_paths": str(args.validation_paths),
        "selection_uses_validation_only": True,
        "heldout_test_used_during_selection": False,
        "selection_metric": "realized_volatility_crps",
        "selected_step": int(best_step),
        "config": {
            "seed": int(args.seed),
            "heston": asdict(heston),
            "training": asdict(training),
            "joint_weight": float(args.joint_weight),
            "split_step_full_grid": int(args.split_step),
            "future_steps_full_grid": int(args.future_steps),
            "prefix_windows_full_grid": [int(value) for value in args.prefix_windows],
            "recent_return_window_full_grid": int(args.recent_return_window),
            "discrepancy_grid_mapping": {
                "split_index": int(mapped["split_index"]),
                "future_horizon": int(mapped["future_horizon"]),
                "prefix_windows": [int(value) for value in mapped["prefix_windows"]],
                "recent_return_window": int(mapped["recent_return_window"]),
            },
            "bandwidths": [float(value) for value in args.bandwidths],
            "eval_every": int(args.eval_every),
            "validation_real_paths": int(validation_paths.shape[0]),
            "validation_bank_paths": int(best_validation_bank.shape[0]),
            "test_real_paths": int(test_paths.shape[0]),
            "test_bank_paths": int(candidate_test_bank.shape[0]),
            "top_k": int(args.top_k),
            "shadow_search": str(args.shadow_search),
            "validation_bank_seed": int(validation_bank_seed),
            "test_bank_seed": int(test_bank_seed),
        },
        "selection_history": selection_history,
        "validation": {
            "baseline": _metric_subset(baseline_validation),
            "selected_candidate": _metric_subset(best_validation_result),
        },
        "test": {
            "baseline": baseline_metrics,
            "selected_candidate": candidate_metrics,
            "paired_difference_definition": "selected_candidate - baseline",
            "paired_bootstrap": paired,
        },
        "shadowing_gate": shadowing_gate,
        "stylized_gate": stylized_gate,
        "overall_gate": {
            "passes_all": bool(shadowing_gate["passes_all"] and stylized_gate["passes_all"]),
            "definition": "shadowing_gate and stylized_gate",
        },
        "joint_discrepancy_test": {
            "baseline": _joint_metrics(
                joint_discrepancy,
                generated_paths=baseline_test_bank,
                target_paths=heldout_paths,
                grid=source_model.grid,
            ),
            "selected_candidate": _joint_metrics(
                joint_discrepancy,
                generated_paths=candidate_test_bank,
                target_paths=heldout_paths,
                grid=candidate_model.grid,
            ),
        },
        "stylized": {
            "baseline": baseline_stylized,
            "selected_candidate": candidate_stylized,
        },
        "fit_final": fit_result.final_metrics,
    }
    write_json(run_dir / "metrics.json", payload)
    write_history_jsonl(run_dir / "history.jsonl", fit_result.history)
    torch.save(validation_paths_all, run_dir / "validation_paths.pt")
    torch.save(baseline_validation_bank, run_dir / "baseline_validation_generated_paths.pt")
    torch.save(best_validation_bank, run_dir / "selected_validation_generated_paths.pt")
    torch.save(candidate_test_bank, run_dir / "selected_test_generated_paths.pt")
    torch.save(
        {
            "baseline_validation": baseline_validation.path_metrics,
            "selected_validation": best_validation_result.path_metrics,
            "baseline_test": baseline_test.path_metrics,
            "selected_test": candidate_test.path_metrics,
        },
        run_dir / "shadowing_per_path_metrics.pt",
    )
    print(f"wrote joint shadowing pilot to {run_dir}", flush=True)
    print(
        "selected step={step}; test rv_crps {base_crps:.6f}->{candidate_crps:.6f}; "
        "rv_cov90 {base_cov:.4f}->{candidate_cov:.4f}; return_crps "
        "{base_return:.6f}->{candidate_return:.6f}; shadow_gate={shadow_gate}; "
        "overall_gate={overall_gate}".format(
            step=int(best_step),
            base_crps=float(baseline_metrics["realized_volatility_crps"]),
            candidate_crps=float(candidate_metrics["realized_volatility_crps"]),
            base_cov=float(baseline_metrics["realized_volatility_coverage_90"]),
            candidate_cov=float(candidate_metrics["realized_volatility_coverage_90"]),
            base_return=float(baseline_metrics["cumulative_return_crps"]),
            candidate_return=float(candidate_metrics["cumulative_return_crps"]),
            shadow_gate=bool(shadowing_gate["passes_all"]),
            overall_gate=bool(payload["overall_gate"]["passes_all"]),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
