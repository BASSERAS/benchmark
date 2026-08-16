from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired method differences and a moving-date-block "
            "bootstrap interval for a normal-to-stress interaction."
        )
    )
    parser.add_argument(
        "--primary-root",
        type=Path,
        action="append",
        required=True,
        help="Method root; repeat once per fitted seed.",
    )
    parser.add_argument(
        "--comparison-root",
        type=Path,
        action="append",
        required=True,
        help="Comparator root paired by seed; repeat once per fitted seed.",
    )
    parser.add_argument("--normal-phase", default="matched_prewar")
    parser.add_argument("--stress-phase", default="war")
    parser.add_argument("--metric", default="realized_volatility_crps")
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20_260_801)
    parser.add_argument(
        "--pairs-per-group",
        type=int,
        default=0,
        help=(
            "For a multi-index panel, number of matched seeds per index. "
            "Roots must be ordered index-major with the same seed order."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metric(root: Path, phase: str, metric: str) -> np.ndarray:
    path = root / f"{phase}_per_path_metrics.npz"
    with np.load(path, allow_pickle=False) as payload:
        if metric not in payload.files:
            raise KeyError(f"{metric!r} is unavailable in {path}")
        values = np.asarray(payload[metric], dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"{path}:{metric} must contain at least two finite values")
    return values


def moving_block_means(
    values: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    generator: np.random.Generator,
) -> np.ndarray:
    num_dates = int(values.size)
    if block_length < 1 or block_length > num_dates:
        raise ValueError("block_length must lie between one and the phase length")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    num_blocks = (num_dates + block_length - 1) // block_length
    starts = generator.integers(
        0,
        num_dates - block_length + 1,
        size=(replicates, num_blocks),
    )
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[..., None] + offsets).reshape(replicates, -1)
    return values[indices[:, :num_dates]].mean(axis=1)


def summarize(values: np.ndarray, bootstrap: np.ndarray, confidence: float) -> dict[str, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    alpha = 0.5 * (1.0 - confidence)
    return {
        "estimate": float(values.mean()),
        "bootstrap_standard_error": float(bootstrap.std(ddof=0)),
        "ci_lower": float(np.quantile(bootstrap, alpha)),
        "ci_upper": float(np.quantile(bootstrap, 1.0 - alpha)),
        "confidence_level": float(confidence),
        "num_dates": int(values.size),
    }


def main() -> None:
    args = parse_args()
    if len(args.primary_root) != len(args.comparison_root):
        raise ValueError(
            "--primary-root and --comparison-root must be repeated equally often"
        )
    phase_differences: dict[str, np.ndarray] = {}
    phase_seed_differences: dict[str, list[np.ndarray]] = {}
    for phase in (args.normal_phase, args.stress_phase):
        seed_differences = []
        for primary_root, comparison_root in zip(
            args.primary_root, args.comparison_root, strict=True
        ):
            primary = load_metric(primary_root, phase, args.metric)
            comparison = load_metric(comparison_root, phase, args.metric)
            if primary.shape != comparison.shape:
                raise ValueError(
                    f"paired arrays differ in {phase}: "
                    f"{primary.shape} != {comparison.shape}"
                )
            seed_differences.append(primary - comparison)
        phase_shapes = {values.shape for values in seed_differences}
        if len(phase_shapes) != 1:
            raise ValueError(f"seed arrays differ in {phase}: {sorted(phase_shapes)}")
        phase_seed_differences[phase] = seed_differences
        phase_differences[phase] = np.stack(seed_differences).mean(axis=0)

    generator = np.random.default_rng(int(args.seed))
    normal_bootstrap = moving_block_means(
        phase_differences[args.normal_phase],
        block_length=int(args.block_length),
        replicates=int(args.replicates),
        generator=generator,
    )
    stress_bootstrap = moving_block_means(
        phase_differences[args.stress_phase],
        block_length=int(args.block_length),
        replicates=int(args.replicates),
        generator=generator,
    )
    interaction = (
        phase_differences[args.stress_phase].mean()
        - phase_differences[args.normal_phase].mean()
    )
    interaction_bootstrap = stress_bootstrap - normal_bootstrap
    pair_interactions = np.asarray(
        [
            stress.mean() - normal.mean()
            for normal, stress in zip(
                phase_seed_differences[args.normal_phase],
                phase_seed_differences[args.stress_phase],
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    if args.pairs_per_group:
        if args.pairs_per_group < 1 or (
            pair_interactions.size % args.pairs_per_group
        ):
            raise ValueError("pairs_per_group must divide the number of root pairs")
        seed_interactions = pair_interactions.reshape(
            -1, args.pairs_per_group
        ).mean(axis=0)
        num_groups = pair_interactions.size // args.pairs_per_group
    else:
        seed_interactions = pair_interactions
        num_groups = 1
    alpha = 0.5 * (1.0 - float(args.confidence_level))
    payload = {
        "metric": args.metric,
        "difference_direction": "primary_minus_comparison",
        "normal_phase": args.normal_phase,
        "stress_phase": args.stress_phase,
        "block_length_dates": int(args.block_length),
        "bootstrap_replicates": int(args.replicates),
        "bootstrap_seed": int(args.seed),
        "normal_difference": summarize(
            phase_differences[args.normal_phase],
            normal_bootstrap,
            float(args.confidence_level),
        ),
        "stress_difference": summarize(
            phase_differences[args.stress_phase],
            stress_bootstrap,
            float(args.confidence_level),
        ),
        "interaction": {
            "estimate": float(interaction),
            "bootstrap_standard_error": float(interaction_bootstrap.std(ddof=0)),
            "ci_lower": float(np.quantile(interaction_bootstrap, alpha)),
            "ci_upper": float(np.quantile(interaction_bootstrap, 1.0 - alpha)),
            "confidence_level": float(args.confidence_level),
            "seed_estimates": [float(value) for value in seed_interactions],
            "seed_sample_standard_deviation": (
                float(seed_interactions.std(ddof=1))
                if seed_interactions.size > 1
                else None
            ),
        },
        "sources": {
            "primary_roots": [str(root.resolve()) for root in args.primary_root],
            "comparison_roots": [
                str(root.resolve()) for root in args.comparison_root
            ],
        },
        "num_seed_pairs": len(args.primary_root),
        "num_groups": int(num_groups),
        "paired_seeds_per_group": int(
            args.pairs_per_group or pair_interactions.size
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
