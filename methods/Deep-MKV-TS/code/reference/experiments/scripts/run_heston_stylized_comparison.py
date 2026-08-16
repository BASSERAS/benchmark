from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.metrics import stylized_fact_metrics  # noqa: E402
from path_dt_experiments.plots import plot_heston_diagnostics_comparison  # noqa: E402
from path_dt_experiments.runners import default_run_dir, ensure_run_dir, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare several generated path banks with one Heston target using "
            "return density, tail survival, QQ, and volatility-clustering plots."
        )
    )
    parser.add_argument("--target-paths", type=Path, required=True)
    parser.add_argument("--target-label", type=str, default="Heston target")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Named candidate tensor; repeat for multiple generated path banks.",
    )
    parser.add_argument("--asset", type=int, default=0)
    parser.add_argument("--lags", type=int, nargs="+", default=tuple(range(1, 21)))
    parser.add_argument("--tail-quantiles", type=float, nargs="+", default=(0.90, 0.95, 0.99))
    parser.add_argument("--max-target-paths", type=int, default=None)
    parser.add_argument("--max-candidate-paths", type=int, default=8192)
    parser.add_argument("--subsample-seed", type=int, default=25001309)
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"candidate must have the form LABEL=PATH, got {value!r}")
    label, raw_path = (part.strip() for part in value.split("=", maxsplit=1))
    if not label or not raw_path:
        raise ValueError(f"candidate must have the form LABEL=PATH, got {value!r}")
    return label, Path(raw_path)


def _load_paths(path: Path, *, max_paths: int | None, seed: int) -> tuple[torch.Tensor, int]:
    if max_paths is not None and int(max_paths) < 1:
        raise ValueError("path subsample limits must be >= 1")
    if path.suffix.lower() == ".npy":
        prices = np.load(path, mmap_mode="r")
        if prices.ndim != 2:
            raise ValueError(f"{path} must contain a price array with shape (B, N + 1)")
        original_count = int(prices.shape[0])
        indices: np.ndarray | slice = slice(None)
        if max_paths is not None and original_count > int(max_paths):
            indices = np.random.default_rng(int(seed)).choice(
                original_count,
                size=int(max_paths),
                replace=False,
            )
        selected = np.asarray(prices[indices], dtype=np.float64)
        if not np.isfinite(selected).all() or np.any(selected <= 0.0):
            raise ValueError(f"{path} contains invalid prices")
        centered_log_prices = np.log(selected / selected[:, :1])
        value = torch.from_numpy(centered_log_prices).unsqueeze(-1)
    else:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError(f"{path} must contain a tensor with shape (B, N + 1, d)")
        original_count = int(value.shape[0])
    if max_paths is not None:
        if path.suffix.lower() != ".npy" and original_count > int(max_paths):
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            indices = torch.randperm(original_count, generator=generator)[: int(max_paths)]
            value = value.index_select(0, indices)
    if not torch.is_floating_point(value):
        value = value.float()
    return value, original_count


def main() -> None:
    args = parse_args()
    parsed_candidates = [_parse_candidate(value) for value in args.candidate]
    all_labels = [str(args.target_label), *(label for label, _ in parsed_candidates)]
    if any(not label.strip() for label in all_labels) or len(set(all_labels)) != len(all_labels):
        raise ValueError("target and candidate labels must be non-empty and unique")

    run_dir = ensure_run_dir(args.run_dir or default_run_dir(prefix="heston_stylized_comparison"))
    target_paths, target_original_count = _load_paths(
        args.target_paths,
        max_paths=args.max_target_paths,
        seed=int(args.subsample_seed),
    )
    candidates: dict[str, torch.Tensor] = {}
    sources: dict[str, object] = {
        str(args.target_label): {
            "path": str(args.target_paths),
            "original_num_paths": target_original_count,
            "evaluated_num_paths": int(target_paths.shape[0]),
        }
    }
    for index, (label, path) in enumerate(parsed_candidates):
        paths, original_count = _load_paths(
            path,
            max_paths=args.max_candidate_paths,
            seed=int(args.subsample_seed) + index + 1,
        )
        candidates[label] = paths
        sources[label] = {
            "path": str(path),
            "original_num_paths": original_count,
            "evaluated_num_paths": int(paths.shape[0]),
        }

    lags = tuple(int(value) for value in args.lags)
    tail_quantiles = tuple(float(value) for value in args.tail_quantiles)
    comparisons = {
        label: stylized_fact_metrics(
            paths,
            target_paths,
            lags=lags,
            tail_quantiles=tail_quantiles,
            include_swd=True,
            swd_projections=int(args.swd_projections),
        )
        for label, paths in candidates.items()
    }
    plot_heston_diagnostics_comparison(
        target_paths=target_paths,
        candidate_paths=candidates,
        output_path=run_dir / "heston_stylized_comparison.png",
        target_label=str(args.target_label),
        asset=int(args.asset),
        lags=lags,
    )
    report = {
        "run_dir": str(run_dir),
        "configuration": {
            "asset": int(args.asset),
            "lags": list(lags),
            "tail_quantiles": list(tail_quantiles),
            "swd_projections": int(args.swd_projections),
            "subsample_seed": int(args.subsample_seed),
        },
        "sources": sources,
        "comparisons_to_target": comparisons,
    }
    write_json(run_dir / "metrics.json", report)

    print(f"run_dir={run_dir}")
    for label, metrics in comparisons.items():
        print(
            f"{label}: abs_acf={metrics['abs_return_acf_error_rms']:.4f} "
            f"sq_acf={metrics['squared_return_acf_error_rms']:.4f} "
            f"kurtosis={metrics['excess_kurtosis_error_rms']:.4f} "
            f"tail={metrics['tail_survival_error_rms']:.4f} "
            f"swd={metrics['path_swd']:.4f}"
        )


if __name__ == "__main__":
    main()
