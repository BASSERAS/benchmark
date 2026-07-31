"""Independent true-DGP draws used only as the evaluator-side "perfect floor".

The floor answers "how well can anything score?": each bank is a fresh sample from
the exact data-generating process, scored against test.npy through the same code
path as a model's bank. Every metric has a non-zero floor because both sides are
finite samples.

Seeds follow the convention already used by metrics/compute_perfect_recovery.py:
draw i uses seed 1000 + i, which is disjoint from every protocol seed (0/1/2
ordinary splits, 100/101 oracle splits). These banks are evaluator-side only and
must never be shown to a generator.

Usage:
  python make_perfect_floor.py --experiment A
  python make_perfect_floor.py --experiment B --draws 5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol" / "experiments"
sys.path.insert(0, str(PROTOCOL))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate independent true-DGP perfect-floor banks.")
    parser.add_argument("--experiment", choices=("A", "B"), required=True)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--num-paths", type=int, default=8192)
    args = parser.parse_args()

    experiment_dir = HERE / f"experiment_{args.experiment}"
    output_dir = experiment_dir / "perfect_floor"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))

    if args.experiment == "A":
        generator = _load_module(
            "generate_drawdown_memory_dataset",
            PROTOCOL / "scripts" / "generate_drawdown_memory_dataset.py",
        )
        config = manifest["configuration"]
        simulate = lambda seed: generator.simulate(  # noqa: E731
            seed=seed,
            num_paths=args.num_paths,
            sequence_length=int(config["sequence_length"]),
            dt=float(config["dt"]),
            s0=float(config["s0"]),
            sigma_low=float(config["sigma_low"]),
            sigma_high=float(config["sigma_high"]),
            drawdown_threshold=float(config["drawdown_threshold"]),
            memory_half_life=float(config["memory_half_life"]),
            response_delay=int(config["response_delay"]),
        )
    else:
        from path_dt_experiments.heston_mixture import simulate_heston_mixture

        simulate = lambda seed: simulate_heston_mixture(  # noqa: E731
            num_paths=args.num_paths, seed=seed
        )

    seeds = [int(args.base_seed) + index for index in range(int(args.draws))]
    draws = []
    for seed in seeds:
        prices, side = simulate(seed)
        np.save(output_dir / f"floor_seed{seed}.npy", prices)
        if args.experiment == "A":
            np.save(output_dir / f"floor_seed{seed}_sigma.npy", side.astype(np.float32))
        else:
            np.save(output_dir / f"floor_seed{seed}_labels.npy", side)
        returns = np.diff(np.log(prices), axis=1)
        entry = {
            "seed": seed,
            "shape": list(prices.shape),
            "log_return_mean": float(returns.mean()),
            "log_return_std": float(returns.std()),
            "price_min": float(prices.min()),
            "price_max": float(prices.max()),
        }
        if args.experiment == "B":
            entry["regime_counts"] = np.bincount(side, minlength=8).tolist()
        draws.append(entry)

    summary = {
        "experiment": args.experiment,
        "role": "evaluator-side perfect floor, independent true-DGP draws",
        "seeds": seeds,
        "num_paths": int(args.num_paths),
        "draws": draws,
    }
    (output_dir / "perfect_floor_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
