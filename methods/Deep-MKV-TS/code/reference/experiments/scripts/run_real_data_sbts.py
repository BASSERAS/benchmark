#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from path_dt_experiments.real_data_generation import (  # noqa: E402
    load_frozen_real_data_fit,
)
from path_dt_experiments.real_data_protocol import (  # noqa: E402
    FROZEN_REAL_DATA_PROTOCOLS,
)


DEFAULT_OFFICIAL_ADAPTER = Path(
    "/home/samer/scenarios/BASSERAS-benchmark/methods/SBTS/code/sbts_generate.py"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one frozen real-data SBTS scenario bank with the official "
            "implementation and no test-window access."
        )
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=REPO_ROOT / "data" / "ICAIF_3Y" / "canonical_rth_128",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=REPO_ROOT / "runs" / "real_data_protocol" / "generated" / "sbts",
    )
    parser.add_argument(
        "--protocol",
        choices=tuple(item.identifier for item in FROZEN_REAL_DATA_PROTOCOLS),
        default="primary_2026_holdout",
    )
    parser.add_argument("--index", choices=("ES", "NQ", "RTY", "YM"), default="ES")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2, 3, 4), required=True)
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--N-pi", type=int, default=50)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--official-adapter", type=Path, default=DEFAULT_OFFICIAL_ADAPTER)
    parser.add_argument(
        "--numba-cache-dir",
        type=Path,
        default=Path("/tmp/deep_mkv_real_data_sbts_numba_cache"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_official_adapter(path: Path):
    sys.path.insert(0, str(path.parent))
    return importlib.import_module(path.stem)


def main() -> None:
    args = _parse_args()
    if args.bank_size < 1 or args.workers < 1:
        raise ValueError("bank-size and workers must be positive")
    fit = load_frozen_real_data_fit(
        args.canonical_root,
        protocol_identifier=args.protocol,
        index=args.index,
    )
    output_dir = args.generated_root / fit.scenario_bank_protocol / fit.index
    output_path = output_dir / f"seed_{args.seed}.npy"
    manifest_path = output_dir / f"seed_{args.seed}.manifest.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing bank: {output_path}")
    adapter_path = args.official_adapter.resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(args.numba_cache_dir.resolve()))
    adapter = _load_official_adapter(adapter_path)
    adapter.warmup_jit()
    started = time.perf_counter()
    generated, official_metadata = adapter.generate_paths(
        fit.train_prices,
        M_simu=int(args.bank_size),
        h=float(args.h),
        K=int(args.K),
        N_pi=int(args.N_pi),
        n_workers=int(args.workers),
        seed=int(args.seed),
    )
    elapsed = time.perf_counter() - started
    generated = np.asarray(generated, dtype=np.float32)
    expected_shape = (int(args.bank_size), int(fit.train_prices.shape[1]))
    if generated.shape != expected_shape:
        raise RuntimeError(
            f"official SBTS returned {generated.shape}, expected {expected_shape}"
        )
    if not np.isfinite(generated).all() or np.any(generated <= 0.0):
        raise RuntimeError("official SBTS returned invalid prices")
    if not np.allclose(generated[:, 0], 100.0, rtol=0.0, atol=1e-5):
        raise RuntimeError("official SBTS bank does not start at 100")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_path, generated)
    payload = {
        "model": "SBTS",
        "replication_kind": "independent_generation_seed_no_fitted_parameters",
        "seed": int(args.seed),
        "requested_protocol": fit.requested_protocol,
        "scenario_bank_protocol": fit.scenario_bank_protocol,
        "index": fit.index,
        "train_sessions": int(fit.train_prices.shape[0]),
        "train_first_date": str(fit.train_dates[0]),
        "train_last_date": str(fit.train_dates[-1]),
        "train_array_sha256": fit.train_sha256,
        "validation_or_test_seen": False,
        "bank_shape": list(generated.shape),
        "output": str(output_path.resolve()),
        "official_adapter": str(adapter_path),
        "official_formula_unchanged": True,
        "official_metadata": official_metadata,
        "elapsed_total_sec": float(elapsed),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved {generated.shape} to {output_path}", flush=True)


if __name__ == "__main__":
    main()
