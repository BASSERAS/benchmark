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

from deep_mkv_gen_path_dt.discrepancies import PathFunctionalDiscrepancy  # noqa: E402
from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from path_dt_experiments.control_ablation import (  # noqa: E402
    AdjointComponentAblationControl,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_basseras_heston_path_shadowing import (  # noqa: E402
    SEQ_LEN,
    _build_components,
    _build_model,
    _load_pdf_evaluator,
    _load_training_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use fresh Heston validation laws and common model noise to identify "
            "whether P or R contracts BASSERAS realized-volatility coverage."
        )
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("/home/samer/scenarios/BASSERAS-benchmark"),
    )
    parser.add_argument(
        "--model-run-dir",
        type=Path,
        default=Path(
            "runs/basseras_heston_specific_entropy_joint_logprice_seed0_"
            "20260729_133000"
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-representation", default="log_price")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--reference-ridge", type=float, default=1e-3)
    parser.add_argument("--reference-variance-shrinkage", type=float, default=0.1)
    parser.add_argument(
        "--reference-log-variance-bias-correction",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--reference-kind",
        choices=("local", "calibrated_causal"),
        default="local",
    )
    parser.add_argument(
        "--causal-rolling-windows",
        type=int,
        nargs="+",
        default=(5, 10, 20, 32),
    )
    parser.add_argument(
        "--causal-ewma-half-lives",
        type=int,
        nargs="+",
        default=(5, 20, 60),
    )
    parser.add_argument(
        "--causal-downside-windows",
        type=int,
        nargs="+",
        default=(10, 32),
    )
    parser.add_argument(
        "--causal-variance-ridges",
        type=float,
        nargs="+",
        default=(1e-5, 1e-4, 1e-3),
    )
    parser.add_argument("--causal-crossfit-folds", type=int, default=3)
    parser.add_argument("--causal-crossfit-seed", type=int, default=1701)
    parser.add_argument(
        "--causal-calibration-weights",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.4, 0.55, 0.7, 0.75, 0.8, 0.85, 1.0),
    )
    parser.add_argument("--causal-calibration-paths", type=int, default=4096)
    parser.add_argument("--causal-calibration-seed", type=int, default=271828)
    parser.add_argument("--causal-calibration-offset-steps", type=int, default=3)
    parser.add_argument(
        "--causal-calibration-return-std-tolerance",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--causal-calibration-sigma-std-floor",
        type=float,
        default=0.045,
    )
    parser.add_argument(
        "--causal-calibration-sigma-cap-fraction-limit",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--causal-calibration-clustering-lags",
        type=int,
        nargs="+",
        default=(1, 5, 20),
    )
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.6)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--validation-query-seed", type=int, default=4)
    parser.add_argument("--validation-reference-seed", type=int, default=5)
    parser.add_argument("--model-bank-seed", type=int, default=6001)
    parser.add_argument("--num-query-paths", type=int, default=512)
    parser.add_argument("--num-reference-paths", type=int, default=4096)
    parser.add_argument("--num-bank-paths", type=int, default=65_536)
    parser.add_argument("--sample-batch-size", type=int, default=32_768)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20230814)
    return parser.parse_args()


def _load_models(args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype):
    target_paths, transform, _ = _load_training_data(
        Path(args.benchmark_root),
        device=device,
        dtype=dtype,
        state_representation="log_price",
    )
    grid, architecture, control, base, joint = _build_components(
        args,
        target_paths=target_paths,
        path_scale=float(transform.sigma) / math.sqrt(float(transform.dt)),
    )

    def make(discrepancy: PathFunctionalDiscrepancy, checkpoint_name: str):
        model = _build_model(
            architecture=architecture,
            grid=grid,
            control=control,
            discrepancy=discrepancy,
            device=device,
            dtype=dtype,
            seed=int(args.seed),
        )
        checkpoint = torch.load(
            Path(args.model_run_dir) / checkpoint_name,
            map_location="cpu",
            weights_only=True,
        )
        model.load_checkpoint_state(checkpoint)
        model.network.eval()
        return model

    return make(base, "source_model_checkpoint.pt"), make(
        joint,
        "model_checkpoint.pt",
    ), transform


def _sample_price_bank(
    model,
    *,
    transform,
    num_paths: int,
    batch_size: int,
    seed: int,
    zero_p: bool,
    zero_r: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    base_control = model.control_map
    if bool(zero_p) or bool(zero_r):
        model.control_map = AdjointComponentAblationControl(
            base_control=base_control,
            zero_p=bool(zero_p),
            zero_r=bool(zero_r),
        )
    prices = np.empty((int(num_paths), SEQ_LEN), dtype=np.float32)
    sigma_values: list[torch.Tensor] = []
    alpha_sum_squares = 0.0
    p_sum_squares = 0.0
    r_sum_squares = 0.0
    control_count = 0
    moment_count = 0
    x0 = torch.zeros((1, 1), device=model.device, dtype=model.dtype)
    try:
        with torch.no_grad():
            for batch_index, start in enumerate(range(0, int(num_paths), int(batch_size))):
                count = min(int(batch_size), int(num_paths) - start)
                batch_seed = derive_stream_seed(
                    base_seed=int(seed),
                    stream_offset=10_000 + int(batch_index),
                )
                sample = model.sample(
                    num_paths=count,
                    x0=x0,
                    seed=batch_seed,
                )
                log_paths = sample.paths[..., 0].detach().cpu().numpy()
                scaled_returns = np.zeros_like(log_paths)
                scaled_returns[:, 1:] = (
                    np.diff(log_paths, axis=1)
                    * math.sqrt(float(transform.dt))
                    / float(transform.sigma)
                )
                prices[start : start + count] = transform.inverse(
                    scaled_returns,
                    dtype=np.float32,
                )
                alpha = sample.controls[..., :1].detach().double()
                sigma = sample.controls[..., 1:].detach().float().cpu().reshape(-1)
                p = sample.expected_adjoint_next.detach().double()
                r = sample.expected_adjoint_noise_next.detach().double()
                sigma_values.append(sigma)
                alpha_sum_squares += float(alpha.pow(2).sum().item())
                p_sum_squares += float(p.pow(2).sum().item())
                r_sum_squares += float(r.pow(2).sum().item())
                control_count += int(alpha.numel())
                moment_count += int(p.numel())
                print(
                    f"  sampled {start + count:>6}/{num_paths}",
                    flush=True,
                )
    finally:
        model.control_map = base_control
    sigma_all = torch.cat(sigma_values).double()
    quantiles = torch.quantile(
        sigma_all,
        torch.tensor((0.05, 0.50, 0.95, 0.99), dtype=torch.float64),
    )
    return prices, {
        "alpha_rms": math.sqrt(alpha_sum_squares / float(control_count)),
        "network_p_rms": math.sqrt(p_sum_squares / float(moment_count)),
        "network_r_rms": math.sqrt(r_sum_squares / float(moment_count)),
        "effective_p_zero": bool(zero_p),
        "effective_r_zero": bool(zero_r),
        "sigma_mean": float(sigma_all.mean().item()),
        "sigma_std": float(sigma_all.std(unbiased=False).item()),
        "sigma_q05": float(quantiles[0].item()),
        "sigma_q50": float(quantiles[1].item()),
        "sigma_q95": float(quantiles[2].item()),
        "sigma_q99": float(quantiles[3].item()),
        "sigma_max_active_fraction": float(
            (sigma_all >= float(getattr(base_control, "sigma_max")) - 1e-5)
            .double()
            .mean()
            .item()
        ),
    }


def _evaluate_bank(
    evaluator,
    *,
    bank: np.ndarray,
    query_prices: np.ndarray,
    reference_prices: np.ndarray,
    top_k: int,
    boot_idx: np.ndarray,
) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
    qlog = np.log(np.asarray(query_prices, dtype=np.float64))
    rlog = np.log(np.asarray(reference_prices, dtype=np.float64))
    q_quantities = evaluator.forecast_quantities(qlog)
    q_features, sqrtw = evaluator.build_features(qlog[:, : evaluator.S_IDX + 1])
    reference_features, _ = evaluator.build_features(
        rlog[:, : evaluator.S_IDX + 1]
    )
    mu = reference_features.mean(axis=0)
    sd = reference_features.std(axis=0) + 1e-12
    q_standardized = (sqrtw * (q_features - mu) / sd).astype(np.float32)

    log_bank = np.log(np.asarray(bank, dtype=np.float64))
    bank_features, _ = evaluator.build_features(
        log_bank[:, : evaluator.S_IDX + 1]
    )
    bank_standardized = (sqrtw * (bank_features - mu) / sd).astype(np.float32)
    indices, distances = evaluator.retrieve(
        q_standardized,
        bank_standardized,
        int(top_k),
    )
    bank_quantities = evaluator.forecast_quantities(log_bank)
    quantities: dict[str, object] = {}
    per_path: dict[str, dict[str, np.ndarray]] = {}
    for name in ("cum", "step", "rv"):
        ensemble = bank_quantities[name][indices]
        quantities[name] = evaluator.metrics_with_ci(
            ensemble,
            q_quantities[name],
            boot_idx,
        )
        per_path[name] = evaluator.per_path_metrics(
            ensemble,
            q_quantities[name],
        )
    return {
        "bank_size": int(bank.shape[0]),
        "quantities": quantities,
        "diagnostics": {
            "prefix_dist_mean": float(distances.mean()),
            "prefix_dist_median": float(np.median(distances)),
            "prefix_dist_p95": float(np.percentile(distances, 95)),
            "unique_candidate_frac": float(np.unique(indices).size / bank.shape[0]),
            "rv_mean_bias": float(
                bank_quantities["rv"][indices].mean()
                - q_quantities["rv"].mean()
            ),
        },
    }, per_path


def _paired_difference(
    evaluator,
    *,
    candidate: dict[str, dict[str, np.ndarray]],
    baseline: dict[str, dict[str, np.ndarray]],
    boot_idx: np.ndarray,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for quantity in ("cum", "step", "rv"):
        output[quantity] = {}
        for metric, (path_key, kind) in evaluator.METRIC_MAP.items():
            candidate_values = candidate[quantity][path_key]
            baseline_values = baseline[quantity][path_key]
            if kind == "rmse":
                point = float(
                    np.sqrt(candidate_values.mean())
                    - np.sqrt(baseline_values.mean())
                )
                boot = np.sqrt(candidate_values[boot_idx].mean(axis=1)) - np.sqrt(
                    baseline_values[boot_idx].mean(axis=1)
                )
            else:
                differences = candidate_values - baseline_values
                point = float(differences.mean())
                boot = differences[boot_idx].mean(axis=1)
            output[quantity][metric] = {
                "value": point,
                "ci": [
                    float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5)),
                ],
            }
    return output


def main() -> None:
    args = parse_args()
    if int(args.num_bank_paths) < int(args.top_k):
        raise ValueError("num-bank-paths must be at least top-k")
    if int(args.num_query_paths) < 2 or int(args.num_reference_paths) < 2:
        raise ValueError("validation laws must contain at least two paths")
    run_dir = ensure_run_dir(Path(args.run_dir))
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    source_model, final_model, transform = _load_models(
        args,
        device=device,
        dtype=dtype,
    )
    evaluator, evaluator_path = _load_pdf_evaluator(Path(args.benchmark_root))

    query_prices = evaluator.generate_heston_bank(
        int(args.num_query_paths),
        int(args.validation_query_seed),
    )
    reference_prices = evaluator.generate_heston_bank(
        int(args.num_reference_paths),
        int(args.validation_reference_seed),
    )
    np.save(run_dir / "validation_query_prices.npy", query_prices)
    np.save(run_dir / "validation_reference_prices.npy", reference_prices)
    boot_idx = np.random.default_rng(int(args.bootstrap_seed)).integers(
        0,
        int(args.num_query_paths),
        size=(int(args.bootstrap_replicates), int(args.num_query_paths)),
    )

    variants = (
        ("reference_p0_r0", source_model, True, True),
        ("source_full", source_model, False, False),
        ("final_full", final_model, False, False),
        ("final_p_only_r0", final_model, False, True),
        ("final_r_only_p0", final_model, True, False),
    )
    results: dict[str, object] = {}
    per_path: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name, model, zero_p, zero_r in variants:
        print(f"sampling {name} with common noise", flush=True)
        bank, controls = _sample_price_bank(
            model,
            transform=transform,
            num_paths=int(args.num_bank_paths),
            batch_size=int(args.sample_batch_size),
            seed=int(args.model_bank_seed),
            zero_p=bool(zero_p),
            zero_r=bool(zero_r),
        )
        metrics, current_per_path = _evaluate_bank(
            evaluator,
            bank=bank,
            query_prices=query_prices,
            reference_prices=reference_prices,
            top_k=int(args.top_k),
            boot_idx=boot_idx,
        )
        metrics["controls"] = controls
        results[name] = metrics
        per_path[name] = current_per_path
        rv = metrics["quantities"]["rv"]
        print(
            f"{name}: rv_crps={rv['crps']['value']:.6f} "
            f"rv_cov90={rv['coverage90']['value']:.4f} "
            f"rv_width90={rv['width90']['value']:.6f} "
            f"sigma_mean={controls['sigma_mean']:.4f}",
            flush=True,
        )

    comparisons = {}
    for candidate, baseline in (
        ("source_full", "reference_p0_r0"),
        ("final_full", "source_full"),
        ("final_full", "reference_p0_r0"),
        ("final_p_only_r0", "final_full"),
        ("final_r_only_p0", "final_full"),
    ):
        comparisons[f"{candidate}_minus_{baseline}"] = _paired_difference(
            evaluator,
            candidate=per_path[candidate],
            baseline=per_path[baseline],
            boot_idx=boot_idx,
        )

    payload = {
        "protocol": {
            "purpose": "fresh-law source/final P/R coverage diagnostic",
            "benchmark_seed3_queries_used": False,
            "benchmark_test_seed1_reference_used": False,
            "validation_query_seed": int(args.validation_query_seed),
            "validation_reference_seed": int(args.validation_reference_seed),
            "model_bank_seed": int(args.model_bank_seed),
            "common_model_noise": True,
            "num_query_paths": int(args.num_query_paths),
            "num_reference_paths": int(args.num_reference_paths),
            "num_bank_paths": int(args.num_bank_paths),
            "K": int(args.top_k),
            "n_boot": int(args.bootstrap_replicates),
            "bootstrap_seed": int(args.bootstrap_seed),
            "evaluator_path": str(evaluator_path.resolve()),
            "evaluator_sha256": hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
            "feature_and_metric_functions_modified": False,
        },
        "variants": {
            "reference_p0_r0": "calibrated sigma_ref with P=R=0",
            "source_full": "source checkpoint with learned P and R",
            "final_full": "joint-discrepancy terminal checkpoint with learned P and R",
            "final_p_only_r0": "final checkpoint with learned P and R forced to zero",
            "final_r_only_p0": "final checkpoint with P forced to zero and learned R",
        },
        "results": results,
        "paired_differences": comparisons,
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(per_path, run_dir / "per_path_metrics.pt")
    print(f"wrote diagnostic to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
