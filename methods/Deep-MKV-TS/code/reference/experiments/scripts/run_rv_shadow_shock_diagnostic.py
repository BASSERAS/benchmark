from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.rv_shadow_diagnostics import (  # noqa: E402
    diagnose_shadow_realized_variation,
    paired_bootstrap_diagnostic_difference,
)
from path_dt_experiments.shadowing import evaluate_path_shadowing  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether realized-volatility shadowing misses are caused "
            "by isolated large returns or persistent future variation."
        )
    )
    parser.add_argument(
        "--bank-sweep-dir",
        type=Path,
        default=Path(
            "runs/heston_dt_specific_entropy_frontier_bank_sweep_seed1234_20260721"
        ),
    )
    parser.add_argument(
        "--real-paths",
        type=Path,
        default=Path(
            "runs/heston_dt_specific_entropy_shadow_frontier_seed1234_20260721/"
            "fresh_test_reference_paths.pt"
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-bank-paths", type=int, default=1_000_000)
    parser.add_argument("--real-path-count", type=int, default=512)
    parser.add_argument("--prefix-length", type=int, default=65)
    parser.add_argument("--future-horizon", type=int, default=32)
    parser.add_argument("--diagnostic-horizons", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--shadow-search", choices=("exact", "faiss"), default="faiss")
    parser.add_argument("--shadow-exact-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=24001330)
    return parser.parse_args()


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(f"{path} must contain a path tensor with shape (B, N + 1, d)")
    return value.detach().cpu()


def main() -> None:
    args = parse_args()
    if int(args.num_bank_paths) < int(args.top_k):
        raise ValueError("num-bank-paths must be at least top-k")
    if int(args.real_path_count) < 1:
        raise ValueError("real-path-count must be >= 1")
    if int(args.bootstrap_replicates) < 2:
        raise ValueError("bootstrap-replicates must be >= 2")
    if not 0.0 < float(args.bootstrap_confidence) < 1.0:
        raise ValueError("bootstrap-confidence must lie in (0, 1)")
    run_dir = ensure_run_dir(args.run_dir)
    bank_dir = Path(args.bank_sweep_dir)
    real_paths = _load_tensor(Path(args.real_paths))[: int(args.real_path_count)]
    specific_entropy_bank = _load_tensor(bank_dir / "frontier_selected_bank.pt")[
        : int(args.num_bank_paths)
    ]
    oracle_bank = _load_tensor(bank_dir / "heston_oracle_bank.pt")[: int(args.num_bank_paths)]
    if int(specific_entropy_bank.shape[0]) < int(args.num_bank_paths) or int(
        oracle_bank.shape[0]
    ) < int(args.num_bank_paths):
        raise ValueError("saved bank is smaller than num-bank-paths")

    results = {}
    diagnostics = {}
    for name, bank in (
        ("specific_entropy", specific_entropy_bank),
        ("heston_oracle", oracle_bank),
    ):
        print(f"selecting {args.top_k} shadows from {args.num_bank_paths:,} {name} paths", flush=True)
        result = evaluate_path_shadowing(
            real_paths=real_paths,
            generated_paths=bank,
            prefix_length=int(args.prefix_length),
            top_k=int(args.top_k),
            future_horizon=int(args.future_horizon),
            search_backend=str(args.shadow_search),
            exact_batch_size=int(args.shadow_exact_batch_size),
        )
        diagnostic = diagnose_shadow_realized_variation(
            real_paths=real_paths,
            generated_paths=bank,
            selected_indices=result.selected_indices,
            prefix_length=int(args.prefix_length),
            horizons=tuple(int(value) for value in args.diagnostic_horizons),
        )
        results[name] = result
        diagnostics[name] = diagnostic

    paired = paired_bootstrap_diagnostic_difference(
        diagnostics["specific_entropy"],
        diagnostics["heston_oracle"],
        num_replicates=int(args.bootstrap_replicates),
        confidence_level=float(args.bootstrap_confidence),
        seed=int(args.bootstrap_seed),
    )
    payload: dict[str, object] = {
        "run_dir": str(run_dir),
        "bank_sweep_dir": str(bank_dir),
        "real_paths": str(args.real_paths),
        "config": {
            "num_bank_paths": int(args.num_bank_paths),
            "num_real_paths": int(real_paths.shape[0]),
            "prefix_length": int(args.prefix_length),
            "future_horizon": int(args.future_horizon),
            "diagnostic_horizons": [int(value) for value in args.diagnostic_horizons],
            "top_k": int(args.top_k),
            "shadow_search": str(args.shadow_search),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_confidence": float(args.bootstrap_confidence),
        },
        "specific_entropy": diagnostics["specific_entropy"].summary,
        "heston_oracle": diagnostics["heston_oracle"].summary,
        "paired_difference_definition": "specific_entropy - heston_oracle",
        "paired_bootstrap": paired,
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(
        {
            "specific_entropy_selected_indices": results["specific_entropy"].selected_indices,
            "heston_oracle_selected_indices": results["heston_oracle"].selected_indices,
            "specific_entropy_path_metrics": diagnostics["specific_entropy"].path_metrics,
            "heston_oracle_path_metrics": diagnostics["heston_oracle"].path_metrics,
        },
        run_dir / "diagnostic_artifacts.pt",
    )
    for horizon in args.diagnostic_horizons:
        print(f"horizon={horizon}", flush=True)
        for name in ("specific_entropy", "heston_oracle"):
            summary = dict(diagnostics[name].summary["horizons"])[str(int(horizon))]
            rv = dict(summary["realized_volatility"])
            leave_out = dict(summary["leave_largest_return_out_volatility"])
            bipower = dict(summary["bipower_volatility"])
            max_return = dict(summary["maximum_absolute_return"])
            print(
                "  {name}: RV={rv:.4f} leave-max={leave:.4f} bipower={bipower:.4f} "
                "max-return={max_return:.4f} upper-miss={upper:.4f}".format(
                    name=name,
                    rv=float(rv["coverage_90"]),
                    leave=float(leave_out["coverage_90"]),
                    bipower=float(bipower["coverage_90"]),
                    max_return=float(max_return["coverage_90"]),
                    upper=float(rv["upper_miss_rate"]),
                ),
                flush=True,
            )
    print(f"wrote RV shock diagnostic to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
