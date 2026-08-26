#!/usr/bin/env python3
"""Re-select the Z-proxy ridge ``ridge_lambda`` on the VALIDATION bank at d = 8.

Why this file exists
--------------------
Algorithm 1's Z-proxy is a ridge regression of ``A_{k+1} eps_{k+1}`` on the
reference features ``u_{t_k} = Phi_k^ref(.)``.  The frozen package builds
``Phi_k^ref`` as the FLATTENED PATH PREFIX plus an intercept, so its width is

    p = 1 + (k + 1) * d

At d = 1 that is at most 252 columns against ``batch_size = 256`` rows -- an
honest, over-determined regression (measured ``ce_projection_r2 ~ 0.28`` on the
committed run).  At d = 8 the same basis reaches ``p = 2017`` columns, so from
step 31 onward the system is UNDER-determined and the ridge interpolates:
measured ``ce_projection_r2 = 0.9335``.  An interpolating projection does no
denoising, which is the entire point of the proxy, and it also makes
``cholesky_ex`` on ``D^T D + lambda I`` fail at every step, forcing the
``lstsq`` fallback that costs ~41% of wall-clock.

The basis is frozen package code and is NOT touched.  Instead ``ridge_lambda``
-- the single knob controlling how hard the projection is regularised -- is
re-selected on the validation discrepancy bank.  This is the ONE re-tuned
hyperparameter of the d = 8 run and is documented as such in the README.

Protocol
--------
For each candidate lambda:

  1. train seed 0 for ``--steps`` steps on the TRAIN split only,
  2. sample a bank of ``--score-paths`` paths from the resulting model,
  3. score the SAME ``build_heston_discrepancy`` used as the training objective
     against ``heston_ma_S_val_8192x252x8.npy``.

Lowest discrepancy on that file wins -- ``MULTIASSET_GUIDELINE.md`` section 0.2
names it as THE re-selection split.  ``heston_ma_S_valdisc_8192x252x8.npy`` is
the A18/A19 judge split; it is scored too and printed alongside, purely to show
the ranking is not an artefact of one draw, and it never decides the winner.
The TEST split is never opened.

Corroborating diagnostic: the committed d = 1 run sits at
``ce_projection_r2 ~ 0.28``.  A candidate whose r2 is near 1 is interpolating,
whatever its discrepancy says.

Everything except ``ridge_lambda`` is imported from ``train_multiasset.py``, so
the sweep cannot silently drift from the campaign configuration.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=3 \
    PYTHONPATH="$R/src:$R/experiments" taskset -c 0-2 \
    /home/tbasseras/gpu-venv/bin/python sweep_ridge_lambda.py \
        --ridge-lambda 1e-3 --steps 250 --device cuda:0

    # afterwards
    python sweep_ridge_lambda.py --report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REFERENCE = Path("/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference")
for _path in (HERE, REFERENCE / "src", REFERENCE / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402

import train_multiasset as tm  # noqa: E402
import eigh_fallback  # noqa: E402
from fit_reference_multiasset import DATASET, DT, STATE_DIM  # noqa: E402

# The paper's spectral clip calls batched cuSOLVER `eigh` on B*T symmetric 8x8
# matrices per outer iteration, and on 2026-08-26 it failed to converge 31 minutes
# into the lambda=1 run. The shim diagnoses the failure and retries the SAME
# decomposition on CPU LAPACK; it raises rather than retrying if Theta is
# non-finite. See eigh_fallback.py for why this is not an algorithmic change.
eigh_fallback.install()

# MULTIASSET_GUIDELINE.md section 0.2 is explicit: a re-selected hyperparameter
# is re-selected on the VALIDATION split, `heston_ma_S_val_8192x252x8.npy`.
# `_valdisc_` is the judge split for A18/A19 and is NOT the selection split; it
# is scored here only as a secondary robustness check and never decides.
VAL_FILE = "heston_ma_S_val_8192x252x8.npy"
VALDISC_FILE = "heston_ma_S_valdisc_8192x252x8.npy"
SWEEP_DIR = HERE / "sweep"
SWEEP_SEED = 0
SCORE_SEED = 20260826
S0 = 100.0


def _tag(value: float) -> str:
    """Filesystem-safe tag for a lambda: 1e-3 -> ``1e-3``, 1000 -> ``1e3``."""

    return f"{float(value):.0e}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")


def load_split(
    filename: str,
    *,
    num_paths: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """A price bank as LOG prices, shape ``(N, T + 1, d)``."""

    prices = np.load(DATASET / filename, mmap_mode="r")
    array = np.asarray(
        prices[: int(num_paths), : int(num_steps) + 1, :], dtype=np.float64
    )
    if array.ndim != 3 or int(array.shape[2]) != STATE_DIM:
        raise ValueError(f"expected (N, T + 1, {STATE_DIM}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{filename} contains non-finite prices")
    if float(array.min()) <= 0.0:
        raise ValueError(f"{filename} contains non-positive prices")
    return torch.log(torch.from_numpy(array)).to(device=device, dtype=dtype)


def _history_mean(history: list[dict], key: str, tail: int = 5) -> float:
    """Mean of ``key`` over the last ``tail`` records that carry a finite value."""

    values = [
        float(rec[key])
        for rec in history
        if key in rec and float(rec[key]) == float(rec[key])
    ]
    if not values:
        return float("nan")
    window = values[-int(tail) :]
    return float(sum(window) / len(window))


def run_one(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    ridge = float(args.ridge_lambda)

    target = tm.load_log_prices(num_paths=args.num_paths, device=device).to(tm.DTYPE)
    if args.num_steps is not None:
        target = target[:, : int(args.num_steps) + 1, :].contiguous()
    num_steps = int(target.shape[1]) - 1
    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)
    print(
        f"[data] train {tuple(target.shape)} lambda={ridge:g} steps={args.steps}",
        flush=True,
    )

    model = tm.build_model(
        grid=grid, device=device, seed=SWEEP_SEED, kernel_steps=num_steps
    )
    training = tm.build_training(
        seed=SWEEP_SEED, steps=int(args.steps), ridge_lambda=ridge
    )
    if float(training.ridge_lambda) != ridge:
        raise RuntimeError("ridge_lambda override did not reach the training config")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    try:
        fit = model.fit(target, training=training)
    except RuntimeError as exc:
        # A lambda whose control diverges is a RESULT, not a missing data point.
        # Left uncaught it would simply be absent from sweep/, and `report()`
        # would tabulate the survivors as though the sweep had been a clean
        # 5-way comparison. Record it, then re-raise so the exit code is honest.
        if "not finite" not in str(exc):
            raise
        elapsed = time.perf_counter() - started
        SWEEP_DIR.mkdir(parents=True, exist_ok=True)
        out = SWEEP_DIR / f"lambda_{_tag(ridge)}.json"
        out.write_text(
            json.dumps(
                {
                    "ridge_lambda": ridge,
                    "steps": int(args.steps),
                    "seed": SWEEP_SEED,
                    "diverged": True,
                    "diverged_after_sec": elapsed,
                    "diverged_detail": str(exc),
                    "selection_file": VAL_FILE,
                    "date": date.today().isoformat(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[diverged] lambda={ridge:g} after {elapsed / 60.0:.1f} min -> {out}",
              flush=True)
        raise
    elapsed = time.perf_counter() - started
    sec_per_step = elapsed / float(args.steps)
    print(f"[fit] {elapsed / 60.0:.1f} min = {sec_per_step:.3f} s/step", flush=True)

    del target
    if device.type == "cuda":
        torch.cuda.empty_cache()

    x0 = torch.full((1, STATE_DIM), math.log(S0), device=device, dtype=model.dtype)
    with torch.no_grad():
        generated = model.sample(
            num_paths=int(args.score_paths), x0=x0, seed=SCORE_SEED
        ).paths
    if not torch.isfinite(generated).all():
        raise RuntimeError("generated bank contains non-finite values")

    def _score(filename: str) -> float:
        bank = load_split(
            filename,
            num_paths=int(args.score_paths),
            num_steps=num_steps,
            device=device,
            dtype=model.dtype,
        )
        with torch.no_grad():
            scored = model.discrepancy(
                generated_paths=generated, target_paths=bank, grid=grid
            )
        value = float(scored.value if hasattr(scored, "value") else scored)
        del bank
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return value

    # Primary selection criterion (MULTIASSET_GUIDELINE.md section 0.2).
    val_discrepancy = _score(VAL_FILE)
    # Secondary, reported only: a second validation-family bank, to show the
    # ranking is not an artefact of one draw.  It never decides the winner.
    valdisc_discrepancy = _score(VALDISC_FILE)
    print(
        f"[val] {VAL_FILE} discrepancy={val_discrepancy:.8g}   "
        f"[check] {VALDISC_FILE} discrepancy={valdisc_discrepancy:.8g}",
        flush=True,
    )

    record = {
        "ridge_lambda": ridge,
        "steps": int(args.steps),
        "seed": SWEEP_SEED,
        "val_discrepancy": val_discrepancy,
        "valdisc_discrepancy": valdisc_discrepancy,
        "ce_projection_r2": _history_mean(fit.history, "ce_projection_r2"),
        "noise_ce_projection_r2": _history_mean(fit.history, "noise_ce_projection_r2"),
        "final_loss": _history_mean(fit.history, "train_total_loss", tail=1),
        "sec_per_step": sec_per_step,
        "elapsed_sec": elapsed,
        "eta_3000_hours": sec_per_step * 3000.0 / 3600.0,
        "peak_gpu_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else None
        ),
        "score_paths": int(args.score_paths),
        "score_seed": SCORE_SEED,
        "selection_file": VAL_FILE,
        "secondary_file": VALDISC_FILE,
        # Non-zero means this lambda drove Theta into a spectrum batched cuSOLVER
        # could not decompose. The run is still valid -- the retry is the same
        # decomposition on CPU LAPACK -- but a lambda that needs it often is a
        # lambda operating in a numerically hard region, and that is selection-
        # relevant information, not a footnote.
        "eigh_fallback": eigh_fallback.counters(),
        "date": date.today().isoformat(),
    }
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    out = SWEEP_DIR / f"lambda_{_tag(ridge)}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[out] {out}", flush=True)
    return record


def report() -> None:
    """Print the sweep table and name the winner."""

    files = sorted(SWEEP_DIR.glob("lambda_*.json"))
    if not files:
        raise SystemExit(f"ABORT: no sweep results in {SWEEP_DIR}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    records.sort(key=lambda rec: rec["ridge_lambda"])

    header = (
        f"{'lambda':>10} {'val_disc*':>14} {'valdisc':>14} {'ce_r2':>8} "
        f"{'noise_r2':>9} {'s/step':>8} {'eta_3000h':>10}"
    )
    print(f"* selection criterion: {VAL_FILE} (MULTIASSET_GUIDELINE.md 0.2)")
    print(f"  secondary, never decides: {VALDISC_FILE}")
    print(f"  d = 1 committed ce_projection_r2 reference: 0.28\n")
    print(header)
    print("-" * len(header))
    for rec in records:
        if rec.get("diverged"):
            # Printed on its own row so the table never silently shrinks to the
            # survivors. A reader must be able to see that this lambda was tried.
            print(
                f"{rec['ridge_lambda']:>10.0e} {'DIVERGED':>14} "
                f"{'--':>14} {'--':>8} {'--':>9} {'--':>8} {'--':>10}"
            )
            continue
        print(
            f"{rec['ridge_lambda']:>10.0e} {rec['val_discrepancy']:>14.8g} "
            f"{rec.get('valdisc_discrepancy', float('nan')):>14.8g} "
            f"{rec['ce_projection_r2']:>8.4f} {rec['noise_ce_projection_r2']:>9.4f} "
            f"{rec['sec_per_step']:>8.3f} {rec['eta_3000_hours']:>10.2f}"
        )
    diverged = [rec for rec in records if rec.get("diverged")]
    if diverged:
        lams = ", ".join(f"{rec['ridge_lambda']:g}" for rec in diverged)
        print(
            f"\n[warn] {len(diverged)} of {len(records)} candidates DIVERGED "
            f"(lambda = {lams}). The control produced non-finite values inside "
            f"{records[0]['steps']} steps. Divergence here is not monotone in lambda, "
            "so surviving a short screen does not bound the full-length run: see "
            "MULTIASSET_GUIDELINE.md section 12.8."
        )
    finite = [
        rec
        for rec in records
        if not rec.get("diverged")
        and float(rec["val_discrepancy"]) == float(rec["val_discrepancy"])
    ]
    if not finite:
        raise SystemExit("ABORT: every candidate produced a non-finite discrepancy")
    best = min(finite, key=lambda rec: rec["val_discrepancy"])
    print(
        f"\nWINNER  ridge_lambda = {best['ridge_lambda']:g}  "
        f"val_discrepancy = {best['val_discrepancy']:.8g}  "
        f"ce_projection_r2 = {best['ce_projection_r2']:.4f}"
    )
    # Stamp the winner with the coverage it was chosen from. Without this a
    # report run mid-sweep produces a winner.json indistinguishable from the
    # final one, and any downstream gate keyed on "winner.json exists" would
    # happily launch against a partial comparison.
    stamped = dict(best)
    stamped["candidates_compared"] = sorted(rec["ridge_lambda"] for rec in records)
    stamped["candidates_diverged"] = sorted(rec["ridge_lambda"] for rec in diverged)
    stamped["n_candidates"] = len(records)
    (SWEEP_DIR / "winner.json").write_text(
        json.dumps(stamped, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[out] {SWEEP_DIR / 'winner.json'}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ridge-lambda", type=float, default=None)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-paths", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--score-paths", type=int, default=8192)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.report:
        report()
        return
    if args.ridge_lambda is None:
        raise SystemExit("ABORT: pass --ridge-lambda, or --report to aggregate")
    run_one(args)


if __name__ == "__main__":
    main()
