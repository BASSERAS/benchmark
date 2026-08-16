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

from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.shadowing import (  # noqa: E402
    ShadowingResult,
    bootstrap_shadowing_metrics,
    evaluate_path_shadowing,
    paired_bootstrap_shadowing_difference,
)
from run_shadowing_sweep import _simulate_heston_bank  # noqa: E402
from run_specific_entropy_joint_shadow import (  # noqa: E402
    _build_models,
    _load_tensor,
    _metric_subset,
    _sample_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a nested scenario-bank convergence sweep for the fitted "
            "specific-entropy source, frontier-selected checkpoint, and a "
            "matched Heston oracle on one untouched test law."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721"),
    )
    parser.add_argument(
        "--frontier-run-dir",
        type=Path,
        default=Path("runs/heston_dt_specific_entropy_shadow_frontier_seed1234_20260721"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--split-step", type=int, default=64)
    parser.add_argument("--future-steps", type=int, default=32)
    parser.add_argument("--prefix-windows", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--recent-return-window", type=int, default=16)
    parser.add_argument("--bandwidths", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument(
        "--bank-sizes",
        type=int,
        nargs="+",
        default=(4096, 16384, 65536, 262144, 1_000_000),
    )
    parser.add_argument("--model-bank-batch-size", type=int, default=50_000)
    parser.add_argument("--heston-bank-batch-size", type=int, default=100_000)
    parser.add_argument("--shadow-real-paths", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--shadow-search", choices=("exact", "faiss"), default="faiss")
    parser.add_argument("--shadow-exact-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=24001320)
    parser.add_argument("--oracle-bank-seed", type=int, default=1761398539)
    parser.add_argument("--save-banks", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    sizes = tuple(int(value) for value in args.bank_sizes)
    if not sizes or tuple(sorted(set(sizes))) != sizes or any(value < 1 for value in sizes):
        raise ValueError("bank-sizes must be strictly increasing positive integers")
    if int(args.top_k) < 1 or int(args.top_k) > min(sizes):
        raise ValueError("top-k must lie between one and the smallest bank size")
    for name in (
        "model_bank_batch_size",
        "heston_bank_batch_size",
        "shadow_real_paths",
        "bootstrap_replicates",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be >= 1")
    if int(args.bootstrap_replicates) < 2:
        raise ValueError("bootstrap-replicates must be >= 2")
    if not 0.0 < float(args.bootstrap_confidence) < 1.0:
        raise ValueError("bootstrap-confidence must lie in (0, 1)")
    return sizes


def _evaluate(
    real_paths: torch.Tensor,
    bank: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> ShadowingResult:
    return evaluate_path_shadowing(
        real_paths=real_paths,
        generated_paths=bank,
        prefix_length=int(args.split_step) + 1,
        top_k=int(args.top_k),
        future_horizon=int(args.future_steps),
        search_backend=str(args.shadow_search),
        exact_batch_size=int(args.shadow_exact_batch_size),
    )


def _bootstrap(
    result: ShadowingResult,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, dict[str, float]]:
    return bootstrap_shadowing_metrics(
        result,
        num_replicates=int(args.bootstrap_replicates),
        confidence_level=float(args.bootstrap_confidence),
        seed=int(seed),
    )


def _paired(
    primary: ShadowingResult,
    comparison: ShadowingResult,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, dict[str, float]]:
    return paired_bootstrap_shadowing_difference(
        primary,
        comparison,
        num_replicates=int(args.bootstrap_replicates),
        confidence_level=float(args.bootstrap_confidence),
        seed=int(seed),
    )


def main() -> None:
    args = parse_args()
    bank_sizes = _validate_args(args)
    run_dir = ensure_run_dir(args.run_dir)
    frontier_dir = Path(args.frontier_run_dir)
    frontier_metrics = _load_json(frontier_dir / "metrics.json")
    config = frontier_metrics.get("config")
    if not isinstance(config, dict):
        raise ValueError("frontier metrics are missing config")
    selected_step = int(frontier_metrics["selected_step"])
    if selected_step <= 0:
        raise ValueError("frontier run selected the source; no distinct candidate is available")
    test_reference = _load_tensor(frontier_dir / "fresh_test_reference_paths.pt")
    if int(test_reference.shape[0]) < int(args.shadow_real_paths):
        raise ValueError("fresh test artifact has fewer paths than shadow-real-paths")
    real_paths = test_reference[: int(args.shadow_real_paths)]

    (
        source_model,
        selected_model,
        _training,
        _stored_config,
        heston,
        _target_paths,
        _source_heldout,
        _joint,
        _mapped,
    ) = _build_models(
        source=Path(args.source_run_dir),
        device=torch.device(args.device),
        args=args,
    )
    selected_checkpoint = torch.load(
        frontier_dir / "selected_model_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(selected_checkpoint, dict):
        raise ValueError("selected checkpoint must be a dictionary")
    selected_model.load_checkpoint_state(selected_checkpoint)

    maximum_size = max(bank_sizes)
    source_seed = int(config["fresh_bank_seed"])
    selected_seed = int(config["fresh_bank_seed"])
    print(f"sampling {maximum_size:,} source paths", flush=True)
    source_bank = _sample_bank(
        source_model,
        num_paths=maximum_size,
        seed=source_seed,
        x0=float(heston.x0),
        batch_size=int(args.model_bank_batch_size),
    )
    print(f"sampling {maximum_size:,} frontier-selected paths", flush=True)
    selected_bank = _sample_bank(
        selected_model,
        num_paths=maximum_size,
        seed=selected_seed,
        x0=float(heston.x0),
        batch_size=int(args.model_bank_batch_size),
    )
    print(f"sampling {maximum_size:,} independent Heston oracle paths", flush=True)
    oracle_bank = _simulate_heston_bank(
        num_paths=maximum_size,
        batch_size=int(args.heston_bank_batch_size),
        config=heston,
        seed=int(args.oracle_bank_seed),
        device=torch.device(args.device),
        dtype=test_reference.dtype,
    )
    if bool(args.save_banks):
        torch.save(source_bank, run_dir / "source_bank.pt")
        torch.save(selected_bank, run_dir / "frontier_selected_bank.pt")
        torch.save(oracle_bank, run_dir / "heston_oracle_bank.pt")

    laws = {
        "specific_entropy_source": source_bank,
        "frontier_selected": selected_bank,
        "heston_oracle": oracle_bank,
    }
    results: dict[int, dict[str, ShadowingResult]] = {}
    metrics_by_size: dict[str, object] = {}
    per_path: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for size in bank_sizes:
        print(f"evaluating nested bank size {size:,}", flush=True)
        current = {
            name: _evaluate(real_paths, bank[:size], args=args) for name, bank in laws.items()
        }
        results[size] = current
        per_path[size] = {
            name: {key: value.detach().cpu() for key, value in result.path_metrics.items()}
            for name, result in current.items()
        }
        bootstrap = {
            name: _bootstrap(
                result,
                args=args,
                seed=int(args.bootstrap_seed) + size + 1000 * index,
            )
            for index, (name, result) in enumerate(current.items())
        }
        paired = {
            "frontier_selected_minus_source": _paired(
                current["frontier_selected"],
                current["specific_entropy_source"],
                args=args,
                seed=int(args.bootstrap_seed) + size + 10_000,
            ),
            "frontier_selected_minus_oracle": _paired(
                current["frontier_selected"],
                current["heston_oracle"],
                args=args,
                seed=int(args.bootstrap_seed) + size + 20_000,
            ),
            "source_minus_oracle": _paired(
                current["specific_entropy_source"],
                current["heston_oracle"],
                args=args,
                seed=int(args.bootstrap_seed) + size + 30_000,
            ),
        }
        metrics_by_size[str(size)] = {
            "metrics": {name: _metric_subset(result) for name, result in current.items()},
            "bootstrap": bootstrap,
            "paired_bootstrap": paired,
        }
        print(
            "size={size:,} RV coverage source={source:.4f} selected={selected:.4f} "
            "oracle={oracle:.4f}".format(
                size=size,
                source=float(current["specific_entropy_source"].metrics["realized_volatility_coverage_90"]),
                selected=float(current["frontier_selected"].metrics["realized_volatility_coverage_90"]),
                oracle=float(current["heston_oracle"].metrics["realized_volatility_coverage_90"]),
            ),
            flush=True,
        )

    torch.save(per_path, run_dir / "shadowing_per_path_metrics.pt")
    payload: dict[str, object] = {
        "run_dir": str(run_dir),
        "source_run_dir": str(args.source_run_dir),
        "frontier_run_dir": str(frontier_dir),
        "fresh_test_reference": str(frontier_dir / "fresh_test_reference_paths.pt"),
        "selected_step": selected_step,
        "config": {
            "bank_sizes": list(bank_sizes),
            "maximum_bank_size": maximum_size,
            "shadow_real_paths": int(real_paths.shape[0]),
            "prefix_length": int(args.split_step) + 1,
            "future_steps": int(args.future_steps),
            "top_k": int(args.top_k),
            "shadow_search": str(args.shadow_search),
            "source_bank_seed": source_seed,
            "selected_bank_seed": selected_seed,
            "oracle_bank_seed": int(args.oracle_bank_seed),
            "common_random_numbers_source_selected": True,
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_confidence": float(args.bootstrap_confidence),
        },
        "metrics_by_bank_size": metrics_by_size,
    }
    write_json(run_dir / "metrics.json", payload)
    print(f"wrote matched bank-size sweep to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
