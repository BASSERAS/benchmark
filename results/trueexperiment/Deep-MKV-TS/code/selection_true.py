#!/usr/bin/env python3
"""
Guideline-§7 selection criterion for Deep-MKV-TS on dataset/TrueDataset.

WHY THIS FILE EXISTS AT ALL
---------------------------
The Heston d=8 pipeline selects both `ridge_lambda` and the training checkpoint
by ARGMIN VALIDATION DISCREPANCY (see
results/HestonMultiAsset/Deep-MKV-TS/code/select_checkpoint_multiasset.py).
That is a goodness-of-fit rule.  On TrueDataset it is forbidden, and the
guideline proves why with a measurement rather than an argument (§7.1):

    h     K   vol_err %        NNratio   verdict
    0.05  3   6.06             0.197     best fit in the grid -- REJECTED
    0.07  1   15.61 +- 2.39    0.923     selected

The 6.06% row beats the real-vs-real MINIMUM of 8.72%.  Two honest slices of the
same market never agree that well, so a generator that does is not modelling the
market -- it is returning pieces of the training set.  Its NNratio of 0.197
confirms it.  Any criterion that maximises agreement with held-out real data
will pick that row every time.  So agreement is demoted to a CONSTRAINT and the
objective becomes a memorisation diagnostic.

THE RULE (pre-registered, zero free parameters)
-----------------------------------------------
    admissible  <=>  paths finite and > MIN_PRICE
                AND  vol_err  <= max real-vs-real vol_err
                AND  corr_err <= max real-vs-real corr_err
    objective   =    minimise |log NNratio|
    tie-break   =    within 0.02 of the best objective, lowest corr_err

Excess kurtosis is REPORTED AND GATES NOTHING: its real-vs-real spread on this
build is a factor of ~34, so at this sample size it separates noise from noise.

Every threshold is recomputed from --data-dir at call time.  It is never
hardcoded, because the dataset-variant sweep changes the splits and therefore
changes the envelope; a hardcoded ceiling would silently apply one variant's
envelope to another.

ONE SOURCE OF TRUTH FOR THE STATISTICS
--------------------------------------
`ann_vol`, `excess_kurt`, `cross_corr`, `min_nn`, `rel_err`, `logret` and
`real_vs_real_envelope` are NOT reimplemented here.  They are imported from the
pre-registered SBTS criterion file and its sha256 is recorded in every artefact
this module writes.  If SBTS's criterion is edited, the cached envelope's digest
stops matching and it is recomputed rather than silently reused.

THE NNratio DENOMINATOR IS `val`, NOT `test`
--------------------------------------------
median NN(val -> train) is the yardstick.  `test` is never touched by any
selection code.  train/val/valdisc all live in the 2022-07 -> 2024-12 era, so
val measures "how close does an honest same-era sample get to train" -- exactly
the quantity a memoriser beats.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
METHOD_ROOT = _HERE.parent

SBTS_CRITERION = (
    METHOD_ROOT.parent / "SBTS" / "code" / "calibrate_bandwidth_true.py"
)
DEFAULT_DATA_DIR = Path(
    "/home/tbasseras/benchmark/dataset/TrueDataset/variants/om_2022-07_N6144"
)
DEFAULT_SEQ_TAG = "6144x128x8"
DEFAULT_ENVELOPE = METHOD_ROOT / "losses" / "selection_envelope.json"

SCORE_SUFFIX = "_val"
TIE_BREAK_WINDOW = 0.02


# --------------------------------------------------------------------------
# the pre-registered criterion module
# --------------------------------------------------------------------------
def _sha256_16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_criterion(data_dir: Path | str = DEFAULT_DATA_DIR):
    """Import SBTS's criterion and give it the bars-per-year it needs.

    The module keeps ``BARS_PER_YEAR`` as a module global that its own ``main``
    fills from dataset_stats.json.  We are importing it as a library, so
    ``main`` never runs and we must fill it ourselves -- from the SAME file, so
    the annualisation factor cannot drift between the two entry points.
    """
    spec = importlib.util.spec_from_file_location(
        "sbts_true_criterion", str(SBTS_CRITERION)
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import criterion from {SBTS_CRITERION}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    stats_path = Path(data_dir) / "dataset_stats.json"
    if not stats_path.exists():
        raise SystemExit(f"missing {stats_path}: cannot annualise volatility")
    stats = json.loads(stats_path.read_text())
    if "bars_per_year" not in stats:
        raise SystemExit(
            f"{stats_path} has no 'bars_per_year'. Refusing to guess it: an "
            "annualisation factor wrong by 64x turns every vol row into fiction."
        )
    module.BARS_PER_YEAR = float(stats["bars_per_year"])
    return module


# --------------------------------------------------------------------------
# envelope
# --------------------------------------------------------------------------
def envelope(
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    seq_tag: str = DEFAULT_SEQ_TAG,
    cache: Path | str | None = DEFAULT_ENVELOPE,
    refresh: bool = False,
) -> dict:
    """Real-vs-real ceilings for this exact dataset variant.

    Cached only because it costs a few minutes of nearest-neighbour work and is
    consulted by every sweep point.  The cache key includes the data-dir, the
    seq tag and the criterion's sha256, so it is invalidated by anything that
    could change the answer.
    """
    data_dir = Path(data_dir)
    criterion_digest = _sha256_16(SBTS_CRITERION)
    key = {
        "data_dir": str(data_dir),
        "seq_tag": seq_tag,
        "score_suffix": SCORE_SUFFIX,
        "criterion_sha256_16": criterion_digest,
    }

    cache_path = Path(cache) if cache is not None else None
    if cache_path is not None and cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text())
        if cached.get("key") == key:
            return cached["envelope"]

    module = load_criterion(data_dir)
    env = module.real_vs_real_envelope(
        str(data_dir), score_suffix=SCORE_SUFFIX, seq_tag=seq_tag
    )
    env = {k: v for k, v in env.items()}
    env["min_price"] = float(module.MIN_PRICE)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"key": key, "envelope": env}, indent=2, sort_keys=True)
            + "\n"
        )
    return env


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def load_splits(
    *, data_dir: Path | str = DEFAULT_DATA_DIR, seq_tag: str = DEFAULT_SEQ_TAG
) -> tuple[np.ndarray, np.ndarray]:
    """train and val only.

    Selection code must never open test or disc.  val is both the reference for
    vol/corr/kurtosis AND the NNratio denominator, for the reason in the module
    docstring: it is a same-era honest sample, so it sets the distance a
    non-memorising generator should keep from train.
    """
    data_dir = Path(data_dir)
    train = np.load(data_dir / f"true_S_{seq_tag}.npy")
    val = np.load(data_dir / f"true_S{SCORE_SUFFIX}_{seq_tag}.npy")
    return train, val


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score_candidate(
    generated: np.ndarray,
    *,
    train: np.ndarray,
    val: np.ndarray,
    module,
    median_nn_heldout: float,
) -> dict:
    """All five numbers §7 asks for, for one generated bank.

    `median_nn_heldout` is passed in rather than recomputed because it is the
    denominator shared by every candidate in a sweep; recomputing it per
    candidate would let it drift with float non-determinism and make two rows
    incomparable.
    """
    generated = np.asarray(generated)
    finite = bool(np.all(np.isfinite(generated)) and np.all(generated > 0.0))
    min_price = float(generated.min()) if finite else 0.0

    if not finite:
        return dict(
            finite=False,
            min_price=min_price,
            vol_err_pct=float("inf"),
            vol_ratio=float("nan"),
            kurt_err_pct=float("inf"),
            corr_err=float("inf"),
            nn_ratio=float("nan"),
            abs_log_nn_ratio=float("inf"),
            m_probe=int(generated.shape[0]),
        )

    vol_ref = module.ann_vol(val)
    vol_gen = module.ann_vol(generated)
    vol_err = module.rel_err(vol_gen, vol_ref)
    vol_ratio = float(np.mean(vol_gen / vol_ref))
    kurt_err = module.rel_err(module.excess_kurt(generated), module.excess_kurt(val))
    corr_err = float(
        np.mean(np.abs(module.cross_corr(generated) - module.cross_corr(val)))
    )

    train_flat = module.logret(train).reshape(len(train), -1)
    gen_flat = module.logret(generated).reshape(len(generated), -1)
    nn_ratio = float(
        np.median(module.min_nn(gen_flat, train_flat)) / median_nn_heldout
    )
    log_nn = abs(math.log(nn_ratio)) if nn_ratio > 0.0 else float("inf")

    return dict(
        finite=True,
        min_price=min_price,
        vol_err_pct=float(vol_err),
        vol_ratio=vol_ratio,
        kurt_err_pct=float(kurt_err),
        corr_err=corr_err,
        nn_ratio=nn_ratio,
        abs_log_nn_ratio=log_nn,
        m_probe=int(generated.shape[0]),
    )


def admissibility(record: dict, env: dict) -> list[str]:
    """Reasons this candidate is disqualified. Empty list = admissible."""
    reasons: list[str] = []
    if not record.get("finite", False):
        reasons.append("non-finite or non-positive prices")
    if record.get("min_price", 0.0) <= env["min_price"]:
        reasons.append(
            f"path collapse: min price {record.get('min_price', 0.0):.3f} "
            f"<= {env['min_price']}"
        )
    if record["vol_err_pct"] > env["vol_err_pct_max"]:
        reasons.append(
            f"vol_err {record['vol_err_pct']:.2f}% > ceiling "
            f"{env['vol_err_pct_max']:.2f}%"
        )
    if record["corr_err"] > env["corr_err_max"]:
        reasons.append(
            f"corr_err {record['corr_err']:.4f} > ceiling {env['corr_err_max']:.4f}"
        )
    return reasons


def objective(record: dict) -> float:
    """|log NNratio|. Lower is better. 0 means the generator sits exactly as far
    from train as an honest held-out real sample does."""
    return float(record["abs_log_nn_ratio"])


def select(records: list[dict], env: dict) -> tuple[dict | None, list[dict]]:
    """Apply the §7 rule to a list of scored candidates.

    Returns (winner, annotated).  `winner` is None when NOTHING is admissible.
    That case must be reported as-is.  Relaxing the envelope to manufacture a
    winner would replace a measured ceiling with a chosen one and destroy the
    only property this criterion has: zero free parameters.
    """
    annotated = []
    for rec in records:
        rec = dict(rec)
        reasons = admissibility(rec, env)
        rec["admissible"] = not reasons
        rec["rejected_because"] = reasons
        annotated.append(rec)

    ok = [r for r in annotated if r["admissible"] and math.isfinite(objective(r))]
    if not ok:
        return None, annotated

    best = min(objective(r) for r in ok)
    tied = [r for r in ok if objective(r) - best <= TIE_BREAK_WINDOW]
    winner = min(tied, key=lambda r: r["corr_err"])
    winner["selected"] = True
    winner["n_tied"] = len(tied)
    return winner, annotated


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def format_table(annotated: list[dict], env: dict, *, label_key: str = "label") -> str:
    lines = []
    width = max([len(str(r.get(label_key, ""))) for r in annotated] + [5])
    header = (
        f"{'candidate'.ljust(width)}  {'vol%':>7}  {'corr':>7}  {'kurt%':>9}  "
        f"{'NNratio':>8}  {'|logNN|':>8}  verdict"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in annotated:
        verdict = "SELECTED" if r.get("selected") else (
            "ok" if r["admissible"] else "; ".join(r["rejected_because"])
        )
        lines.append(
            f"{str(r.get(label_key, '')).ljust(width)}  "
            f"{r['vol_err_pct']:7.2f}  {r['corr_err']:7.4f}  "
            f"{r['kurt_err_pct']:9.2f}  {r['nn_ratio']:8.4f}  "
            f"{r['abs_log_nn_ratio']:8.4f}  {verdict}"
        )
    lines.append("")
    lines.append(
        f"ceilings (recomputed, not hardcoded): vol_err <= "
        f"{env['vol_err_pct_max']:.2f}%   corr_err <= {env['corr_err_max']:.4f}"
    )
    return "\n".join(lines)


def report_envelope(env: dict) -> str:
    nn_lo, nn_hi = env["nn_ratio_band"]
    return "\n".join(
        [
            "--- real-vs-real envelope (derived from --data-dir, not hardcoded) ---",
            f"  splits found: {env['splits_found']}",
            f"  vol_err%   min {env['vol_err_pct_min']:7.2f}   "
            f"median {env['vol_err_pct_median']:7.2f}   "
            f"CEILING {env['vol_err_pct_max']:7.2f}",
            f"  corr_err   min {env['corr_err_min']:.4f}   "
            f"median {env['corr_err_median']:.4f}   "
            f"CEILING {env['corr_err_max']:.4f}",
            f"  kurt_err%  min {env['kurt_err_pct_min']:7.2f}   "
            f"median {env['kurt_err_pct_median']:7.2f}   "
            f"max {env['kurt_err_pct_max']:7.2f}   (REPORTED, NOT GATED)",
            f"  NNratio band {nn_lo:.4f} .. {nn_hi:.4f}   "
            f"{env['nn_ratio_by_split']}",
            f"  median NN(val -> train) = {env['median_nn_heldout']:.6f}"
            "   <- NNratio denominator",
            f"  a PERFECT generator (= train) would score "
            f"vol {env['train_vs_score']['vol_err_pct']:.2f}%  "
            f"corr {env['train_vs_score']['corr_err']:.4f}",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--seq-tag", default=DEFAULT_SEQ_TAG)
    ap.add_argument("--cache", default=str(DEFAULT_ENVELOPE))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    env = envelope(
        data_dir=args.data_dir,
        seq_tag=args.seq_tag,
        cache=args.cache,
        refresh=args.refresh,
    )
    print(report_envelope(env))
    print(f"\ncached to {args.cache}")


if __name__ == "__main__":
    main()
