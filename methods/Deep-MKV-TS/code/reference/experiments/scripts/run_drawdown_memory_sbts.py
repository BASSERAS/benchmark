from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))


OFFICIAL_ADAPTER = Path(
    "/home/samer/scenarios/BASSERAS-benchmark/methods/SBTS/code/sbts_generate.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the existing official-code SBTS adapter on one synthetic dataset."
    )
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-paths", type=int, default=8192)
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--N-pi", type=int, default=50)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--deterministic-numba-seed",
        action="store_true",
        help=(
            "Seed the Numba RNG inside each official worker. This changes only "
            "randomness reproducibility, not the SBTS simulation formula."
        ),
    )
    return parser.parse_args()


def load_adapter():
    sys.path.insert(0, str(OFFICIAL_ADAPTER.parent))
    return importlib.import_module("sbts_generate")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    train = np.load(args.train_data)
    if train.ndim != 2 or not np.isfinite(train).all() or np.any(train <= 0.0):
        raise ValueError("train-data must contain finite positive price paths")
    adapter = load_adapter()
    if args.deterministic_numba_seed:
        from path_dt_experiments.sbts_reproducible_worker import enable_seeded_worker

        enable_seeded_worker(adapter)
    adapter.warmup_jit()
    started = time.perf_counter()
    generated, metadata = adapter.generate_paths(
        train,
        M_simu=int(args.num_paths),
        h=float(args.h),
        K=int(args.K),
        N_pi=int(args.N_pi),
        n_workers=int(args.workers),
        seed=int(args.seed),
    )
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, generated)
    payload = {
        **metadata,
        "official_formula_unchanged": True,
        "official_adapter": str(OFFICIAL_ADAPTER),
        "train_data": str(args.train_data.resolve()),
        "train_data_sha256": _sha256(args.train_data),
        "generated_bank_sha256": _sha256(args.output),
        "elapsed_total_sec": float(elapsed),
        "training_wall_seconds": float(elapsed),
        "cuda_incremental_peak_allocated_bytes": 0,
        "peak_gpu_allocated_gb": 0.0,
        "peak_process_rss_mb": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
        "peak_child_rss_mb": float(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
        ),
        "test_split_used": False,
        "deterministic_numba_seed": bool(args.deterministic_numba_seed),
        "seed_patch_scope": (
            "Numba RNG initialization only; official formula unchanged"
            if args.deterministic_numba_seed
            else "none"
        ),
    }
    (args.output.parent / "generation_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved {generated.shape} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
