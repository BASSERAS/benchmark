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
    diagnose_shadow_by_prefix_volatility_regime,
    paired_bootstrap_regime_difference,
)
from path_dt_experiments.shadowing import evaluate_path_shadowing  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose selected-model and oracle RV shadowing calibration by "
            "observed prefix-volatility regime."
        )
    )
    parser.add_argument("--bank-sweep-dir", type=Path, required=True)
    parser.add_argument("--real-paths", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-bank-paths", type=int, default=1_000_000)
    parser.add_argument("--real-path-count", type=int, default=512)
    parser.add_argument("--prefix-length", type=int, default=65)
    parser.add_argument("--prefix-window", type=int, default=16)
    parser.add_argument("--future-horizon", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--shadow-search", choices=("exact", "faiss"), default="faiss")
    parser.add_argument("--shadow-exact-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=24001340)
    return parser.parse_args()


def _load_paths(path: Path) -> torch.Tensor:
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
    real_paths = _load_paths(Path(args.real_paths))[: int(args.real_path_count)]
    laws = {
        "specific_entropy_source": _load_paths(bank_dir / "source_bank.pt"),
        "full_trust_selected": _load_paths(bank_dir / "frontier_selected_bank.pt"),
        "heston_oracle": _load_paths(bank_dir / "heston_oracle_bank.pt"),
    }
    for name, bank in laws.items():
        if int(bank.shape[0]) < int(args.num_bank_paths):
            raise ValueError(f"{name} bank is smaller than num-bank-paths")
        laws[name] = bank[: int(args.num_bank_paths)]

    results = {}
    diagnostics = {}
    for name, bank in laws.items():
        print(
            f"selecting {args.top_k} shadows from {args.num_bank_paths:,} {name} paths",
            flush=True,
        )
        result = evaluate_path_shadowing(
            real_paths=real_paths,
            generated_paths=bank,
            prefix_length=int(args.prefix_length),
            top_k=int(args.top_k),
            future_horizon=int(args.future_horizon),
            search_backend=str(args.shadow_search),
            exact_batch_size=int(args.shadow_exact_batch_size),
        )
        diagnostic = diagnose_shadow_by_prefix_volatility_regime(
            real_paths=real_paths,
            generated_paths=bank,
            selected_indices=result.selected_indices,
            prefix_length=int(args.prefix_length),
            prefix_window=int(args.prefix_window),
            future_horizon=int(args.future_horizon),
        )
        results[name] = result
        diagnostics[name] = diagnostic

    comparisons = {
        "full_trust_selected_minus_source": paired_bootstrap_regime_difference(
            diagnostics["full_trust_selected"],
            diagnostics["specific_entropy_source"],
            num_replicates=int(args.bootstrap_replicates),
            confidence_level=float(args.bootstrap_confidence),
            seed=int(args.bootstrap_seed),
        ),
        "full_trust_selected_minus_oracle": paired_bootstrap_regime_difference(
            diagnostics["full_trust_selected"],
            diagnostics["heston_oracle"],
            num_replicates=int(args.bootstrap_replicates),
            confidence_level=float(args.bootstrap_confidence),
            seed=int(args.bootstrap_seed) + 1,
        ),
        "source_minus_oracle": paired_bootstrap_regime_difference(
            diagnostics["specific_entropy_source"],
            diagnostics["heston_oracle"],
            num_replicates=int(args.bootstrap_replicates),
            confidence_level=float(args.bootstrap_confidence),
            seed=int(args.bootstrap_seed) + 2,
        ),
    }
    payload: dict[str, object] = {
        "run_dir": str(run_dir),
        "bank_sweep_dir": str(bank_dir),
        "real_paths": str(args.real_paths),
        "config": {
            "num_bank_paths": int(args.num_bank_paths),
            "num_real_paths": int(real_paths.shape[0]),
            "prefix_length": int(args.prefix_length),
            "prefix_window": int(args.prefix_window),
            "future_horizon": int(args.future_horizon),
            "top_k": int(args.top_k),
            "shadow_search": str(args.shadow_search),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_confidence": float(args.bootstrap_confidence),
        },
        "laws": {name: diagnostic.summary for name, diagnostic in diagnostics.items()},
        "paired_difference_definition": "first law minus second law within the same real-prefix regime",
        "paired_bootstrap": comparisons,
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(
        {
            name: {
                "selected_indices": results[name].selected_indices,
                "path_metrics": diagnostic.path_metrics,
                "regime_indices": diagnostic.regime_indices,
                "regime_thresholds": diagnostic.regime_thresholds,
            }
            for name, diagnostic in diagnostics.items()
        },
        run_dir / "diagnostic_artifacts.pt",
    )

    for regime in ("overall", "low", "middle", "high"):
        print(f"regime={regime}", flush=True)
        for name in laws:
            summary = diagnostics[name].summary
            metrics = summary["overall"] if regime == "overall" else summary["regimes"][regime]
            print(
                "  {name}: coverage={coverage:.4f} lower={lower:.4f} "
                "upper={upper:.4f} width={width:.5f} crps={crps:.5f} "
                "bias={bias:+.5f}".format(
                    name=name,
                    coverage=float(metrics["coverage_90"]),
                    lower=float(metrics["lower_miss_rate"]),
                    upper=float(metrics["upper_miss_rate"]),
                    width=float(metrics["band_width_90"]),
                    crps=float(metrics["crps"]),
                    bias=float(metrics["mean_bias"]),
                ),
                flush=True,
            )
    print(f"wrote prefix-regime RV shadow diagnostic to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
