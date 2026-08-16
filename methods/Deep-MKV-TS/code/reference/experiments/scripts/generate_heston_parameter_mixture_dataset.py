from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.heston_mixture import (  # noqa: E402
    DT,
    KAPPA,
    MU,
    RHO_LEVELS,
    S0,
    SEQUENCE_LENGTH,
    THETA_LEVELS,
    XI_LEVELS,
    regimes,
    simulate_fixed_heston,
    simulate_heston_mixture,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a balanced eight-regime Heston mixture.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-paths", type=int, default=8192)
    parser.add_argument("--oracle-train-paths", type=int, default=32768)
    parser.add_argument("--oracle-validation-paths", type=int, default=8192)
    return parser.parse_args()


def save_split(output_dir: Path, name: str, *, num_paths: int, seed: int) -> dict[str, object]:
    prices, labels = simulate_heston_mixture(num_paths=num_paths, seed=seed)
    np.save(output_dir / f"{name}.npy", prices)
    np.save(output_dir / f"{name}_labels.npy", labels)
    returns = np.diff(np.log(prices), axis=1)
    return {
        "seed": seed,
        "shape": list(prices.shape),
        "regime_counts": np.bincount(labels, minlength=8).tolist(),
        "log_return_mean": float(returns.mean()),
        "log_return_std": float(returns.std()),
        "price_min": float(prices.min()),
        "price_max": float(prices.max()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specifications = {
        "train": (int(args.num_paths), 0),
        "test": (int(args.num_paths), 1),
        "disc": (int(args.num_paths), 2),
        "oracle_train": (int(args.oracle_train_paths), 100),
        "oracle_validation": (int(args.oracle_validation_paths), 101),
    }
    diagnostics = {
        name: save_split(args.output_dir, name, num_paths=count, seed=seed)
        for name, (count, seed) in specifications.items()
    }
    mean_heston = simulate_fixed_heston(
        num_paths=int(args.num_paths),
        seed=3,
        theta=float(np.mean(THETA_LEVELS)),
        xi=float(np.mean(XI_LEVELS)),
        rho=float(np.mean(RHO_LEVELS)),
    )
    np.save(args.output_dir / "mean_heston.npy", mean_heston)
    train = np.load(args.output_dir / "train.npy", mmap_mode="r")
    indices = np.random.default_rng(4).choice(len(train), size=int(args.num_paths), replace=True)
    np.save(args.output_dir / "whole_path_bootstrap.npy", np.asarray(train[indices]))
    manifest = {
        "name": "balanced_discrete_heston_parameter_mixture",
        "sequence_length": SEQUENCE_LENGTH,
        "dt": DT,
        "s0": S0,
        "kappa": KAPPA,
        "mu": MU,
        "theta_levels": list(THETA_LEVELS),
        "xi_levels": list(XI_LEVELS),
        "rho_levels": list(RHO_LEVELS),
        "regimes": list(regimes()),
        "splits": diagnostics,
        "mean_heston_parameters": {
            "theta": float(np.mean(THETA_LEVELS)),
            "xi": float(np.mean(XI_LEVELS)),
            "rho": float(np.mean(RHO_LEVELS)),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
