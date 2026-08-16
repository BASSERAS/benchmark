from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from path_dt_experiments.data import HestonConfig, simulate_heston_log_paths  # noqa: E402
from path_dt_experiments.runners import default_run_dir, ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.shadowing import (  # noqa: E402
    ShadowingResult,
    bootstrap_shadowing_metrics,
    evaluate_path_shadowing_sweep,
    paired_bootstrap_shadowing_difference,
    select_top_k_by_validation_crps,
)


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} did not contain a torch.Tensor")
    if value.ndim != 3:
        raise ValueError(f"{path} must have shape (B, N + 1, d)")
    return value


def _heston_config_from_source(path: Path | None) -> HestonConfig:
    if path is None:
        return HestonConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("config", {}).get("heston")
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain config.heston")
    return HestonConfig(**raw)


def _simulate_heston_bank(
    *,
    num_paths: int,
    batch_size: int,
    config: HestonConfig,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if int(num_paths) < 1:
        raise ValueError("num_paths must be >= 1")
    if int(batch_size) < 1:
        raise ValueError("bank_batch_size must be >= 1")
    bank = torch.empty(int(num_paths), int(config.num_steps) + 1, 1, dtype=dtype, device="cpu")
    for batch_index, start in enumerate(range(0, int(num_paths), int(batch_size))):
        stop = min(start + int(batch_size), int(num_paths))
        batch_seed = derive_stream_seed(base_seed=int(seed), stream_offset=30_000 + int(batch_index))
        chunk = simulate_heston_log_paths(
            num_paths=stop - start,
            config=config,
            seed=batch_seed,
            device=device,
            dtype=dtype,
        )
        bank[start:stop].copy_(chunk.detach().cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure path-shadowing sweeps on saved or synthetic path banks.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--heldout-paths", type=Path, required=True)
    parser.add_argument("--validation-paths", type=Path, default=None)
    parser.add_argument("--generated-paths", type=Path, default=None)
    parser.add_argument("--comparison-generated-paths", type=Path, default=None)
    parser.add_argument("--comparison-label", default="comparison")
    parser.add_argument("--source-metrics", type=Path, default=None)
    parser.add_argument("--generate-heston-bank", action="store_true")
    parser.add_argument("--num-bank-paths", type=int, default=1_000_000)
    parser.add_argument("--bank-batch-size", type=int, default=50_000)
    parser.add_argument("--bank-seed", type=int, default=23001303)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--save-bank", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--shadow-real-paths", type=int, default=512)
    parser.add_argument("--shadow-validation-paths", type=int, default=None)
    parser.add_argument("--prefix-length", type=int, required=True)
    parser.add_argument("--future-steps", type=int, default=None)
    parser.add_argument("--top-k", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--shadow-search", choices=("auto", "exact", "faiss"), default="faiss")
    parser.add_argument("--shadow-exact-batch-size", type=int, default=64)
    parser.add_argument(
        "--selection-metric",
        choices=("cumulative_return_crps", "increment_crps", "realized_volatility_crps"),
        default="cumulative_return_crps",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=23001304)
    parser.add_argument("--label", default="shadowing_sweep")
    return parser.parse_args()


def _evaluate_sweep(
    *,
    real_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    args: argparse.Namespace,
) -> dict[int, ShadowingResult]:
    return evaluate_path_shadowing_sweep(
        real_paths=real_paths,
        generated_paths=generated_paths,
        prefix_length=int(args.prefix_length),
        future_horizon=None if args.future_steps is None else int(args.future_steps),
        top_k_values=tuple(int(value) for value in args.top_k),
        search_backend=str(args.shadow_search),
        exact_batch_size=int(args.shadow_exact_batch_size),
    )


def _metrics_payload(sweep: dict[int, ShadowingResult]) -> dict[str, dict[str, float | str]]:
    return {str(top_k): dict(result.metrics) for top_k, result in sweep.items()}


def _bootstrap_payload(
    sweep: dict[int, ShadowingResult],
    *,
    args: argparse.Namespace,
    seed_offset: int = 0,
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        str(top_k): bootstrap_shadowing_metrics(
            result,
            num_replicates=int(args.bootstrap_replicates),
            confidence_level=float(args.bootstrap_confidence),
            seed=int(args.bootstrap_seed) + int(seed_offset) + int(top_k),
        )
        for top_k, result in sweep.items()
    }


def _per_path_payload(sweep: dict[int, ShadowingResult]) -> dict[int, dict[str, torch.Tensor]]:
    return {
        int(top_k): {name: values.detach().cpu() for name, values in result.path_metrics.items()}
        for top_k, result in sweep.items()
    }


def main() -> None:
    args = parse_args()
    if bool(args.generate_heston_bank) == (args.generated_paths is not None):
        raise ValueError("Provide exactly one of --generated-paths or --generate-heston-bank")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    run_dir = ensure_run_dir(args.run_dir or default_run_dir(prefix="shadowing_sweep"))

    heldout_paths = _load_tensor(args.heldout_paths).to(dtype=dtype)
    if int(args.shadow_real_paths) < 1:
        raise ValueError("--shadow-real-paths must be >= 1")
    real_count = min(int(args.shadow_real_paths), int(heldout_paths.shape[0]))
    real_paths = heldout_paths[:real_count]
    if args.validation_paths is not None:
        if args.validation_paths.resolve() == args.heldout_paths.resolve():
            raise ValueError("--validation-paths must be distinct from --heldout-paths")
        validation_paths_all = _load_tensor(args.validation_paths).to(dtype=dtype)
        validation_limit = (
            int(args.shadow_real_paths)
            if args.shadow_validation_paths is None
            else int(args.shadow_validation_paths)
        )
        if validation_limit < 1:
            raise ValueError("--shadow-validation-paths must be >= 1")
        validation_count = min(validation_limit, int(validation_paths_all.shape[0]))
        validation_paths = validation_paths_all[:validation_count]
    else:
        validation_paths = None

    heston_config = None
    if args.generate_heston_bank:
        heston_config = _heston_config_from_source(args.source_metrics)
        generated_paths = _simulate_heston_bank(
            num_paths=int(args.num_bank_paths),
            batch_size=int(args.bank_batch_size),
            config=heston_config,
            seed=int(args.bank_seed),
            device=device,
            dtype=dtype,
        )
        if bool(args.save_bank):
            torch.save(generated_paths, run_dir / "generated_heston_bank.pt")
    else:
        assert args.generated_paths is not None
        generated_paths = _load_tensor(args.generated_paths).to(dtype=dtype)

    comparison_paths = (
        None
        if args.comparison_generated_paths is None
        else _load_tensor(args.comparison_generated_paths).to(dtype=dtype)
    )

    sweep = _evaluate_sweep(real_paths=real_paths, generated_paths=generated_paths, args=args)
    bootstrap_by_top_k = _bootstrap_payload(sweep, args=args)
    validation_sweep = (
        None
        if validation_paths is None
        else _evaluate_sweep(real_paths=validation_paths, generated_paths=generated_paths, args=args)
    )
    selected_top_k = (
        None
        if validation_sweep is None
        else select_top_k_by_validation_crps(validation_sweep, metric=str(args.selection_metric))
    )

    comparison_sweep = (
        None
        if comparison_paths is None
        else _evaluate_sweep(real_paths=real_paths, generated_paths=comparison_paths, args=args)
    )
    comparison_validation_sweep = (
        None
        if comparison_paths is None or validation_paths is None
        else _evaluate_sweep(real_paths=validation_paths, generated_paths=comparison_paths, args=args)
    )
    comparison_selected_top_k = (
        None
        if comparison_validation_sweep is None
        else select_top_k_by_validation_crps(
            comparison_validation_sweep,
            metric=str(args.selection_metric),
        )
    )
    comparison_bootstrap_by_top_k = (
        None
        if comparison_sweep is None
        else _bootstrap_payload(comparison_sweep, args=args, seed_offset=10_000)
    )
    paired_differences_by_top_k = None
    paired_selected_difference = None
    if comparison_sweep is not None:
        paired_differences_by_top_k = {
            str(top_k): paired_bootstrap_shadowing_difference(
                sweep[top_k],
                comparison_sweep[top_k],
                num_replicates=int(args.bootstrap_replicates),
                confidence_level=float(args.bootstrap_confidence),
                seed=int(args.bootstrap_seed) + 20_000 + int(top_k),
            )
            for top_k in sorted(set(sweep) & set(comparison_sweep))
        }
        if selected_top_k is not None and comparison_selected_top_k is not None:
            paired_selected_difference = paired_bootstrap_shadowing_difference(
                sweep[selected_top_k],
                comparison_sweep[comparison_selected_top_k],
                num_replicates=int(args.bootstrap_replicates),
                confidence_level=float(args.bootstrap_confidence),
                seed=int(args.bootstrap_seed) + 30_000,
            )

    per_path_artifact = run_dir / "shadowing_per_path_metrics.pt"
    per_path_payload: dict[str, object] = {"primary": _per_path_payload(sweep)}
    if validation_sweep is not None:
        per_path_payload["primary_validation"] = _per_path_payload(validation_sweep)
    if comparison_sweep is not None:
        per_path_payload["comparison"] = _per_path_payload(comparison_sweep)
    if comparison_validation_sweep is not None:
        per_path_payload["comparison_validation"] = _per_path_payload(comparison_validation_sweep)
    torch.save(per_path_payload, per_path_artifact)

    payload = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "heldout_paths": str(args.heldout_paths),
        "validation_paths": None if args.validation_paths is None else str(args.validation_paths),
        "generated_paths": None if args.generated_paths is None else str(args.generated_paths),
        "generated_bank_kind": "heston" if args.generate_heston_bank else "saved",
        "num_generated_paths": int(generated_paths.shape[0]),
        "num_real_paths": int(real_paths.shape[0]),
        "num_validation_paths": 0 if validation_paths is None else int(validation_paths.shape[0]),
        "path_shape": list(generated_paths.shape),
        "prefix_length": int(args.prefix_length),
        "future_steps": None if args.future_steps is None else int(args.future_steps),
        "top_k": [int(value) for value in args.top_k],
        "shadow_search": str(args.shadow_search),
        "selection_metric": str(args.selection_metric),
        "selected_top_k": selected_top_k,
        "heston": None if heston_config is None else heston_config.to_dict(),
        "bank_seed": None if not args.generate_heston_bank else int(args.bank_seed),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_confidence": float(args.bootstrap_confidence),
        "bootstrap_seed": int(args.bootstrap_seed),
        "per_path_metrics": str(per_path_artifact),
        "metrics_by_top_k": _metrics_payload(sweep),
        "bootstrap_by_top_k": bootstrap_by_top_k,
        "validation_metrics_by_top_k": None if validation_sweep is None else _metrics_payload(validation_sweep),
        "selected_test_metrics": None if selected_top_k is None else dict(sweep[selected_top_k].metrics),
        "comparison": None
        if comparison_sweep is None
        else {
            "label": str(args.comparison_label),
            "generated_paths": str(args.comparison_generated_paths),
            "num_generated_paths": int(comparison_paths.shape[0]),
            "path_shape": list(comparison_paths.shape),
            "selected_top_k": comparison_selected_top_k,
            "metrics_by_top_k": _metrics_payload(comparison_sweep),
            "bootstrap_by_top_k": comparison_bootstrap_by_top_k,
            "validation_metrics_by_top_k": None
            if comparison_validation_sweep is None
            else _metrics_payload(comparison_validation_sweep),
            "selected_test_metrics": None
            if comparison_selected_top_k is None
            else dict(comparison_sweep[comparison_selected_top_k].metrics),
        },
        "paired_difference_definition": None
        if comparison_sweep is None
        else f"{args.label} - {args.comparison_label}",
        "paired_differences_by_top_k": paired_differences_by_top_k,
        "paired_selected_difference": paired_selected_difference,
    }
    write_json(run_dir / "shadowing_sweep_metrics.json", payload)
    print(f"wrote shadowing sweep to {run_dir}")
    for top_k in sorted(sweep):
        metrics = sweep[top_k].metrics
        print(
            "top_k={top_k} prefix_distance={prefix:.4f} crps={crps:.4f} rmse={rmse:.4f} coverage90={coverage:.4f}".format(
                top_k=top_k,
                prefix=float(metrics["prefix_distance_mean"]),
                crps=float(metrics["cumulative_return_crps"]),
                rmse=float(metrics["continuation_mean_rmse"]),
                coverage=float(metrics["coverage_90"]),
            )
        )
    if selected_top_k is not None:
        print(
            f"selected primary top_k={selected_top_k} on validation {args.selection_metric}="
            f"{validation_sweep[selected_top_k].metrics[str(args.selection_metric)]:.6f}"
        )
    if comparison_selected_top_k is not None:
        assert comparison_validation_sweep is not None
        print(
            f"selected comparison top_k={comparison_selected_top_k} on validation {args.selection_metric}="
            f"{comparison_validation_sweep[comparison_selected_top_k].metrics[str(args.selection_metric)]:.6f}"
        )


if __name__ == "__main__":
    main()
