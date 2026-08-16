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

from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402
from path_dt_experiments.heston_mixture_discrepancy import (  # noqa: E402
    fit_fixed_target_heston_mixture_summary_mmd,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the label-free Heston-mixture summary discrepancy."
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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--subsample-paths", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=73001)
    return parser.parse_args()


def _load_log_paths(path: Path, *, device: torch.device) -> torch.Tensor:
    prices = np.asarray(np.load(path), dtype=np.float32)
    if prices.ndim != 2 or not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ValueError(f"{path} must contain finite positive price paths")
    values = torch.from_numpy(prices).to(device=device)
    return torch.log(values / values[:, :1]).unsqueeze(-1)


def main() -> None:
    args = _parse_args()
    root = Path(args.experiment_root)
    output = args.output or root / "discrepancy_audit.json"
    manifest = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))
    grid = DiscreteTimeGrid(
        T=(int(manifest["sequence_length"]) - 1) * float(manifest["dt"]),
        num_steps=int(manifest["sequence_length"]) - 1,
    )
    device = torch.device(args.device)
    paths = {
        "oracle_floor": root / "data" / "disc.npy",
        "mean_heston": root / "data" / "mean_heston.npy",
        "bootstrap": root / "data" / "whole_path_bootstrap.npy",
        "sbts": root / "sbts" / "generated_paths_8192x128.npy",
        "deep_mkv": root / "deep_mkv" / "generated_paths_8192x128.npy",
    }
    target = _load_log_paths(root / "data" / "train.npy", device=device)
    candidates = {
        name: _load_log_paths(path, device=device)
        for name, path in paths.items()
    }
    discrepancy = fit_fixed_target_heston_mixture_summary_mmd(
        target,
        grid=grid,
    )
    sample_count = int(args.subsample_paths)
    if sample_count < 2 or sample_count > int(target.shape[0]):
        raise ValueError("subsample_paths must lie between 2 and the target path count")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    rows: dict[str, list[dict[str, float]]] = {
        name: [] for name in candidates
    }
    for _ in range(int(args.repeats)):
        target_indices = torch.randperm(
            int(target.shape[0]),
            generator=generator,
        )[:sample_count].to(device)
        target_batch = target.index_select(0, target_indices)
        for name, candidate in candidates.items():
            candidate_indices = torch.randperm(
                int(candidate.shape[0]),
                generator=generator,
            )[:sample_count].to(device)
            result = discrepancy(
                generated_paths=candidate.index_select(0, candidate_indices),
                target_paths=target_batch,
                grid=grid,
            )
            rows[name].append(
                {
                    "value": float(result.value.detach().item()),
                    **{
                        key: float(value)
                        for key, value in result.metrics.items()
                        if key.endswith("_mmd")
                    },
                }
            )

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for name, candidate_rows in rows.items():
        summaries[name] = {}
        for metric in candidate_rows[0]:
            values = np.asarray(
                [row[metric] for row in candidate_rows],
                dtype=np.float64,
            )
            summaries[name][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }

    gradient_paths = candidates["deep_mkv"][: min(256, sample_count)].detach().clone()
    gradient_paths.requires_grad_(True)
    gradient_target = target[: int(gradient_paths.shape[0])]
    gradient_result = discrepancy(
        generated_paths=gradient_paths,
        target_paths=gradient_target,
        grid=grid,
    )
    gradient = torch.autograd.grad(
        gradient_result.value,
        gradient_paths,
    )[0]
    payload = {
        "discrepancy": discrepancy.summary(),
        "subsample_paths": sample_count,
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "candidates": summaries,
        "gradient_audit": {
            "value": float(gradient_result.value.detach().item()),
            "rms": float(torch.sqrt(torch.mean(gradient.pow(2))).item()),
            "max_abs": float(gradient.abs().max().item()),
            "finite": bool(torch.isfinite(gradient).all().item()),
        },
        "sources": {
            "train": str((root / "data" / "train.npy").resolve()),
            **{
                name: str(path.resolve())
                for name, path in paths.items()
            },
        },
        "uses_oracle_labels_or_parameters": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
