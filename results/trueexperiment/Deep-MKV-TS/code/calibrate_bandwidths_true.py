#!/usr/bin/env python3
"""Measure, freeze, and apply the per-block MMD bandwidth gauge for TrueDataset.

WHY THIS FILE EXISTS
--------------------
The published Deep-MKV-TS discrepancy stack (preset ``old_fullv_w0p25``, 13
blocks) uses ABSOLUTE bandwidths.  ``MMDPathFunctionalDiscrepancy`` does not
standardise its features -- see
``src/deep_mkv_gen_path_dt/discrepancies/mmd.py:237`` -- and ``rbf_mmd2``
evaluates

    k = exp( - ||x - y||^2 / (2 h^2 p) ),      p = feature dimension

so the kernel compares the PER-COORDINATE RMS distance of the features against
the bandwidth ``h``.  Those bandwidths were tuned on Heston paths at
``dt = 1/252``.  TrueDataset is 30-second crypto bars at
``dt = 9.512937595129376e-07``, i.e. ``sqrt(dt)`` is 65x smaller, while the
realized-variance features are ``1/dt``-scaled and therefore blow UP.  Both
failure modes are present at once, measured on 512 paths per dataset:

    block                                    pcRMS(H)    pcRMS(T)     ratio
    path_law_observed_path                  2.0712e-01  7.6231e-03    0.0368
    path_law_terminal                       2.9731e-01  1.1238e-02    0.0378
    path_law_increments                     1.8107e-02  1.1008e-03    0.0608
    volatility_law_instantaneous_rv         1.0671e-01  4.4796e+00   41.9787
    volatility_law_global_rv                1.9657e-02  8.9923e-01   45.7460
    volatility_law_sigma_like_terminal_...  2.9731e-01  1.1232e-02    0.0378
    volatility_law_sigma_like_returns       1.8107e-02  1.1008e-03    0.0608
    volatility_law_sigma_like_realized_vol  7.3582e-03  4.3413e-04    0.0590
    volatility_law_sigma_like_abs_returns   1.1557e-02  5.7435e-04    0.0497
    volatility_law_sigma_like_squared_ret.  4.2346e-04  4.2614e-06    0.0101
    volatility_law_abs_return_acf           1.7138e+00  5.5683e+00    3.2490
    volatility_law_squared_return_acf       2.6224e+00  1.9164e+01    7.3078

With the Heston bandwidths left in place the price blocks give ``k ~ 0.9999``
(the MMD is then pure float32 cancellation noise, eps = 1.2e-7) and the
volatility / ACF blocks give ``k = 0`` exactly (the MMD is the constant ``2/n``
with IDENTICALLY ZERO gradient).  Nothing crashes.  Training would run for
hours, print a smooth loss curve, and learn almost nothing.

THE FIX (chosen by the user, verbatim)
--------------------------------------
    "Multiply each block's bandwidths by that block's measured pcRMS ratio, so
     every block sits at exactly the same kernel contrast it has at Heston.
     Pure gauge change: same features, same relative weights, same geometry,
     only the ruler changes.  Bandwidths get recorded in the config as
     derived-from-data, not hand-typed."

So for every block

    h_true = h_heston * pcRMS(True) / pcRMS(Heston)

which leaves the dimensionless ratio ``h / pcRMS`` -- the only thing the RBF
kernel can see -- invariant.  Feature maps, block names, block weights,
``lambda_scale``/``kappa_scale`` and the number of bandwidths per block are all
untouched.  This is a change of units, not a change of objective.

WHAT IS *NOT* RESCALED
----------------------
``volatility_law_state_rv`` is a ``TargetStandardizedStateRealizedVarianceMMD``,
not an ``MMDPathFunctionalDiscrepancy``: it standardises its features against
the target batch internally, so its bandwidths are already dimensionless and
already transfer.  It is recorded in the gauge with ``mode="standardized"`` and
``ratio=1.0`` so that the artefact still accounts for all 13 blocks -- a block
silently missing from the gauge is a bug, and the applier raises on it.

HESTON IS A CALIBRATION-TIME DEPENDENCY ONLY
--------------------------------------------
This script reads the Heston train split ONCE, writes
``losses/bandwidth_gauge.json``, and is then never run again.  ``train_true.py``
consumes only the JSON.  The frozen reference package under
``methods/Deep-MKV-TS/code/reference`` is not modified: the gauge is applied
with ``dataclasses.replace`` on the blocks that ``build_heston_discrepancy``
returns.

USAGE
-----
    calibrate  :  python calibrate_bandwidths_true.py --num-paths 512
    inspect    :  python calibrate_bandwidths_true.py --report-only
    consume    :  from calibrate_bandwidths_true import (
                      load_bandwidth_gauge, apply_bandwidth_gauge)
                  discrepancy = apply_bandwidth_gauge(
                      build_heston_discrepancy(num_steps=127, ...),
                      load_bandwidth_gauge())
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REFERENCE = Path("/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference")
for _path in (str(_REFERENCE / "src"), str(_REFERENCE / "experiments"), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from deep_mkv_gen_path_dt.discrepancies.composite import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from deep_mkv_gen_path_dt.discrepancies.mmd import (  # noqa: E402
    MMDPathFunctionalDiscrepancy,
)
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402

# ---------------------------------------------------------------------------
# The two panels.  Both are TRAIN splits; validation and test are never touched
# by the gauge, which is a property of the measurement units and not of the fit.
# ---------------------------------------------------------------------------

HESTON_DATASET = Path("/home/tbasseras/benchmark/dataset/HestonMultiAsset")
HESTON_TRAIN_FILE = "heston_ma_S_8192x252x8.npy"
HESTON_DT = 0.003968253968253968

TRUE_DATASET = Path(
    "/home/tbasseras/benchmark/dataset/TrueDataset/variants/om_2022-07_N6144"
)
TRUE_SEQ_TAG = "6144x128x8"
TRUE_TRAIN_FILE = f"true_S_{TRUE_SEQ_TAG}.npy"
TRUE_DT = 9.512937595129376e-07

STATE_DIM = 8

# Discrepancy arguments copied verbatim from the published Heston trainer
# (results/HestonMultiAsset/Deep-MKV-TS/code/train_multiasset.py:226-236).
# Everything except ``num_steps`` is identical between the two panels: the
# gauge changes the ruler, not the objective.
DISCREPANCY_KWARGS: dict[str, Any] = {
    "include_acf": True,
    "preset": "old_fullv_w0p25",
    "lambda_scale": 50.0,
    "kappa_scale": 100.0,
    "include_conditional_volatility": False,
    "include_conditional_volatility_correlation": False,
    "abs_return_acf_weight": 0.25,
    "squared_return_acf_weight": 0.125,
}

# Blocks whose features are standardised inside the discrepancy and therefore
# must NOT be rescaled.  Kept as an explicit set so that a future preset change
# that adds another standardised block fails loudly instead of being rescaled
# twice.
STANDARDIZED_BLOCKS = frozenset({"volatility_law_state_rv"})

EXPECTED_NUM_BLOCKS = 13

DEFAULT_GAUGE = _HERE.parent / "losses" / "bandwidth_gauge.json"
DEFAULT_NUM_PATHS = 512


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load_log_prices(
    dataset: Path, filename: str, *, num_paths: int, device: torch.device
) -> torch.Tensor:
    """Train-split LOG prices, ``(N, T + 1, d)`` float64.

    Both panels store PRICES on disk with ``S[:, 0, :] == 100.0``; every feature
    map in the stack consumes log prices, so the log is taken here exactly as
    the two trainers do.

    Both datasets keep their splits flat in one directory and mark them with a
    suffix (``_val``, ``_test``, ``_disc``, ``_valdisc``); the UNSUFFIXED file
    is the train split.  ``filename`` is therefore already the train file and
    must not be joined with a ``train/`` component.
    """
    path = dataset / filename
    if not path.exists():
        raise SystemExit(f"missing train split: {path}")
    prices = np.load(path, mmap_mode="r")
    if prices.ndim != 3 or int(prices.shape[2]) != STATE_DIM:
        raise SystemExit(f"{path}: expected (N, T+1, {STATE_DIM}), got {prices.shape}")
    subset = np.asarray(prices[: int(num_paths)], dtype=np.float64)
    if not np.isfinite(subset).all():
        raise SystemExit(f"{path}: non-finite prices in the first {num_paths} paths")
    if (subset <= 0.0).any():
        raise SystemExit(f"{path}: non-positive prices in the first {num_paths} paths")
    return torch.from_numpy(np.log(subset)).to(device=device, dtype=torch.float64)


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


def per_coordinate_rms(features: torch.Tensor) -> float:
    """RMS pairwise distance per feature coordinate -- the quantity ``h`` faces.

    ``rbf_mmd2`` divides the squared distance by ``2 h^2 p`` with ``p`` the
    feature dimension, so the kernel argument is ``(||x-y||^2 / p) / (2 h^2)``.
    The scale a bandwidth must match is therefore

        pcRMS = sqrt( E_{i != j} ||x_i - x_j||^2 / p )

    taken over the TARGET batch, because ``k_yy`` is the one kernel block whose
    scale is fixed by data rather than by the (initially arbitrary) generator.
    The diagonal is excluded: it is identically zero and would bias the mean
    down by ``1/n``.
    """
    flat = features.reshape(int(features.shape[0]), -1).double()
    n = int(flat.shape[0])
    p = float(max(1, int(flat.shape[1])))
    if n < 2:
        raise SystemExit("need at least 2 paths to measure a pairwise scale")
    squared = torch.cdist(flat, flat).pow(2)
    off_diagonal_sum = float(squared.sum()) - float(torch.diagonal(squared).sum())
    mean_squared = off_diagonal_sum / float(n * (n - 1))
    value = float(np.sqrt(max(mean_squared, 0.0) / p))
    if not np.isfinite(value):
        raise SystemExit("non-finite pairwise scale -- feature map produced inf/nan")
    return value


def _iterate_blocks(
    discrepancy: CompositePathFunctionalDiscrepancy,
) -> list[WeightedDiscrepancy]:
    blocks = list(discrepancy.blocks)
    if len(blocks) != EXPECTED_NUM_BLOCKS:
        raise SystemExit(
            f"expected {EXPECTED_NUM_BLOCKS} blocks in preset "
            f"{DISCREPANCY_KWARGS['preset']!r}, found {len(blocks)}; the gauge "
            "was derived for the published stack and must be re-derived if the "
            "preset changes"
        )
    return blocks


def measure_panel(
    *, log_prices: torch.Tensor, dt: float
) -> tuple[dict[str, float], dict[str, tuple[float, ...]], list[str]]:
    """pcRMS and bandwidths of every rescalable block on one panel."""
    num_steps = int(log_prices.shape[1]) - 1
    grid = DiscreteTimeGrid(T=num_steps * float(dt), num_steps=num_steps)
    discrepancy = build_heston_discrepancy(num_steps=num_steps, **DISCREPANCY_KWARGS)
    scales: dict[str, float] = {}
    bandwidths: dict[str, tuple[float, ...]] = {}
    names: list[str] = []
    paths = log_prices.to(torch.float32)
    with torch.no_grad():
        for block in _iterate_blocks(discrepancy):
            names.append(str(block.name))
            inner = block.discrepancy
            if str(block.name) in STANDARDIZED_BLOCKS:
                continue
            if not isinstance(inner, MMDPathFunctionalDiscrepancy):
                raise SystemExit(
                    f"block {block.name!r} is a {type(inner).__name__}, which is "
                    "neither a plain MMD block nor a declared standardised "
                    "block; refusing to guess whether it needs a gauge"
                )
            features = inner.feature_map(paths, grid=grid)
            scales[str(block.name)] = per_coordinate_rms(features)
            bandwidths[str(block.name)] = tuple(float(v) for v in inner.bandwidths)
    return scales, bandwidths, names


def build_gauge(*, num_paths: int, device: torch.device) -> dict[str, Any]:
    heston = _load_log_prices(
        HESTON_DATASET, HESTON_TRAIN_FILE, num_paths=num_paths, device=device
    )
    true_panel = _load_log_prices(
        TRUE_DATASET, TRUE_TRAIN_FILE, num_paths=num_paths, device=device
    )
    heston_scales, heston_bandwidths, heston_names = measure_panel(
        log_prices=heston, dt=HESTON_DT
    )
    true_scales, true_bandwidths, true_names = measure_panel(
        log_prices=true_panel, dt=TRUE_DT
    )
    if heston_names != true_names:
        raise SystemExit(
            "block names differ between panels; the two discrepancies are not "
            f"the same stack:\n  heston {heston_names}\n  true   {true_names}"
        )
    if heston_bandwidths != true_bandwidths:
        raise SystemExit(
            "bandwidths differ between panels before rescaling; "
            "DISCREPANCY_KWARGS is not being applied identically"
        )

    blocks: dict[str, dict[str, Any]] = {}
    for name in heston_names:
        if name in STANDARDIZED_BLOCKS:
            blocks[name] = {
                "mode": "standardized",
                "ratio": 1.0,
                "reason": (
                    "features are standardised against the target batch inside "
                    "the discrepancy; bandwidths are already dimensionless"
                ),
            }
            continue
        reference_scale = heston_scales[name]
        target_scale = true_scales[name]
        if reference_scale <= 0.0:
            raise SystemExit(
                f"block {name!r}: Heston pcRMS is {reference_scale}; cannot form "
                "a ratio from a degenerate reference scale"
            )
        ratio = target_scale / reference_scale
        source = heston_bandwidths[name]
        blocks[name] = {
            "mode": "rescaled",
            "pcrms_heston": reference_scale,
            "pcrms_true": target_scale,
            "ratio": ratio,
            "bandwidths_heston": list(source),
            "bandwidths_true": [float(v) * ratio for v in source],
        }

    if len(blocks) != EXPECTED_NUM_BLOCKS:
        raise SystemExit(
            f"gauge covers {len(blocks)} blocks, expected {EXPECTED_NUM_BLOCKS}"
        )

    return {
        "description": (
            "Per-block MMD bandwidth gauge. h_true = h_heston * pcRMS(true) / "
            "pcRMS(heston), so every block keeps the dimensionless ratio "
            "h / pcRMS -- the only quantity the RBF kernel can observe -- that "
            "it had on the published Heston run. Change of units, not of "
            "objective."
        ),
        "preset": DISCREPANCY_KWARGS["preset"],
        "discrepancy_kwargs": dict(DISCREPANCY_KWARGS),
        "num_paths": int(num_paths),
        "scale_estimator": (
            "sqrt(mean_{i != j} ||f_i - f_j||^2 / feature_dim) over the train "
            "split, matching the k_yy block of rbf_mmd2"
        ),
        "reference_panel": {
            "dataset": str(HESTON_DATASET),
            "train_file": HESTON_TRAIN_FILE,
            "dt": HESTON_DT,
            "num_steps": int(heston.shape[1]) - 1,
        },
        "target_panel": {
            "dataset": str(TRUE_DATASET),
            "train_file": TRUE_TRAIN_FILE,
            "dt": TRUE_DT,
            "num_steps": int(true_panel.shape[1]) - 1,
        },
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# Consumption -- the only part train_true.py imports
# ---------------------------------------------------------------------------


def load_bandwidth_gauge(path: Path | str | None = None) -> dict[str, Any]:
    """Read the frozen gauge.  No Heston dependency at training time."""
    source = Path(path) if path is not None else DEFAULT_GAUGE
    if not source.exists():
        raise SystemExit(
            f"missing bandwidth gauge {source}; run calibrate_bandwidths_true.py "
            "before training"
        )
    with source.open() as handle:
        gauge = json.load(handle)
    if str(gauge.get("preset")) != DISCREPANCY_KWARGS["preset"]:
        raise SystemExit(
            f"{source}: gauge was measured for preset {gauge.get('preset')!r}, "
            f"trainer builds {DISCREPANCY_KWARGS['preset']!r}"
        )
    if len(gauge.get("blocks", {})) != EXPECTED_NUM_BLOCKS:
        raise SystemExit(f"{source}: gauge does not cover {EXPECTED_NUM_BLOCKS} blocks")
    return gauge


def apply_bandwidth_gauge(
    discrepancy: CompositePathFunctionalDiscrepancy, gauge: dict[str, Any]
) -> CompositePathFunctionalDiscrepancy:
    """Return the same stack with every rescalable block's bandwidths regauged.

    Every block must appear in the gauge and every gauge entry must appear in
    the stack -- a mismatch means the gauge and the trainer disagree about what
    is being optimised, which is exactly the failure this file exists to
    prevent, so it raises rather than falling back to the Heston bandwidths.
    """
    entries = dict(gauge["blocks"])
    blocks = _iterate_blocks(discrepancy)
    rebuilt: list[WeightedDiscrepancy] = []
    for block in blocks:
        name = str(block.name)
        if name not in entries:
            raise SystemExit(f"block {name!r} has no entry in the bandwidth gauge")
        entry = entries.pop(name)
        if entry["mode"] == "standardized":
            if name not in STANDARDIZED_BLOCKS:
                raise SystemExit(
                    f"gauge marks {name!r} standardised but the trainer does not"
                )
            rebuilt.append(block)
            continue
        inner = block.discrepancy
        if not isinstance(inner, MMDPathFunctionalDiscrepancy):
            raise SystemExit(
                f"block {name!r} is a {type(inner).__name__}; cannot regauge"
            )
        current = tuple(float(v) for v in inner.bandwidths)
        expected = tuple(float(v) for v in entry["bandwidths_heston"])
        if current != expected:
            raise SystemExit(
                f"block {name!r}: trainer bandwidths {current} do not match the "
                f"gauge's reference bandwidths {expected}; the gauge is stale"
            )
        regauged = tuple(float(v) for v in entry["bandwidths_true"])
        rebuilt.append(
            dataclasses.replace(
                block,
                discrepancy=dataclasses.replace(inner, bandwidths=regauged),
            )
        )
    if entries:
        raise SystemExit(
            f"gauge has entries not present in the stack: {sorted(entries)}"
        )
    return dataclasses.replace(discrepancy, blocks=tuple(rebuilt))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _report(gauge: dict[str, Any]) -> None:
    print(f"{'block':<46}{'pcRMS(H)':>12}{'pcRMS(T)':>12}{'ratio':>14}")
    for name, entry in gauge["blocks"].items():
        if entry["mode"] == "standardized":
            print(f"{name:<46}{'-':>12}{'-':>12}{'standardized':>14}")
            continue
        print(
            f"{name:<46}{entry['pcrms_heston']:>12.4e}"
            f"{entry['pcrms_true']:>12.4e}{entry['ratio']:>14.4f}"
        )
    print(
        "\nbandwidths, Heston -> TrueDataset "
        "(identical dimensionless h / pcRMS on every block):"
    )
    for name, entry in gauge["blocks"].items():
        if entry["mode"] == "standardized":
            continue
        before = ", ".join(f"{v:g}" for v in entry["bandwidths_heston"])
        after = ", ".join(f"{v:.4g}" for v in entry["bandwidths_true"])
        print(f"  {name:<46} ({before})  ->  ({after})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-paths", type=int, default=DEFAULT_NUM_PATHS)
    parser.add_argument("--out", type=Path, default=DEFAULT_GAUGE)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="print the existing gauge and exit without measuring or writing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing gauge (it is meant to be frozen once)",
    )
    args = parser.parse_args()

    if args.report_only:
        _report(load_bandwidth_gauge(args.out))
        return

    if args.out.exists() and not args.force:
        raise SystemExit(
            f"{args.out} already exists; the gauge is a frozen artefact. "
            "Pass --force only if you intend to invalidate every run that used it."
        )

    gauge = build_gauge(num_paths=int(args.num_paths), device=torch.device(args.device))
    _report(gauge)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(gauge, handle, indent=2, sort_keys=False)
    os.replace(temporary, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
