#!/usr/bin/env python3
"""Re-select the Z-proxy ``ridge_lambda`` for Deep-MKV-TS on dataset/TrueDataset.

WHAT IS RE-SELECTED AND WHY
---------------------------
Algorithm 1's Z-proxy regresses ``A_{k+1} eps_{k+1}`` on the reference features
``u_{t_k} = Phi_k^ref(.)``, which the frozen package builds as the flattened
path prefix plus an intercept -- width ``p = 1 + (k + 1) * d``.  At d = 8 and
T = 127 that reaches ``p = 1025`` columns against ``batch_size = 256`` rows, so
from step 31 onward the system is under-determined and the ridge interpolates.
``ridge_lambda`` is the single knob controlling how hard that projection is
regularised.  The basis is frozen package code and is NOT touched.

HOW IT DIFFERS FROM THE HESTON SWEEP
------------------------------------
results/HestonMultiAsset/Deep-MKV-TS/code/sweep_ridge_lambda.py selects the
winner with ``min(finite, key=val_discrepancy)`` -- lowest discrepancy against
the validation bank.  That is a goodness-of-fit rule and it is CORRECT there,
because the five Heston splits are five independent draws from one known SDE,
so "closer to val" really does mean "closer to the law".

It is FORBIDDEN here.  There is one history of the real market; the five
TrueDataset splits are carved out of it.  The guideline measures what happens
when you select on fit anyway (section 7.1): the best-fitting SBTS row scored
vol_err 6.06%, beating the real-vs-real MINIMUM of 8.72%, with NNratio 0.197 --
a memoriser, selected unanimously by the fit rule.

So the discrepancy is still computed and still printed, because a ranking that
inverts between the two rules is worth seeing, but it NEVER decides.  The
winner is chosen by selection_true.select: minimise ``|log NNratio|`` subject
to the real-vs-real vol/corr envelope, tie-broken within 0.02 on ``corr_err``.

WHY THE SCORE BANK IS 2048 PATHS BY DEFAULT
-------------------------------------------
Guideline section 7.2: at ``m_probe = 512`` the sampling noise on ``vol_err`` is
1.67pp, which is wider than the gaps this sweep needs to resolve.  2048 is the
smallest size the guideline accepts for a decision rather than a screen.

Protocol, per candidate lambda:
  1. train seed 0 for ``--steps`` steps on the TRAIN split only,
  2. sample a bank of ``--score-paths`` paths,
  3. score it with the section-7 statistics against the VALIDATION split,
  4. additionally score the training discrepancy against val and valdisc,
     reported only.
The TEST split is never opened.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \\
    PYTHONPATH="$R/src:$R/experiments" taskset -c 0-7 \\
    /home/tbasseras/gpu-venv/bin/python sweep_ridge_lambda_true.py \\
        --ridge-lambda 1e-3 --steps 250 --device cuda:0

    # afterwards
    /home/tbasseras/gpu-venv/bin/python sweep_ridge_lambda_true.py --report
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

import eigh_fallback  # noqa: E402
import selection_true as sel  # noqa: E402
import train_true as tt  # noqa: E402
from fit_reference_true import DATASET, DT, SEQ_TAG, STATE_DIM  # noqa: E402

eigh_fallback.install()

VAL_FILE = f"true_S_val_{SEQ_TAG}.npy"
VALDISC_FILE = f"true_S_valdisc_{SEQ_TAG}.npy"
SWEEP_DIR = HERE / "sweep"
SWEEP_SEED = 0
SCORE_SEED = 20260826
# dataset/TrueDataset is built with every path starting at exactly 100.0
# (the guideline's ``S[:, 0, :] == 100.0`` contract), so this is the sampling x0.
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

    target = tt.load_log_prices(num_paths=args.num_paths, device=device).to(tt.DTYPE)
    if args.num_steps is not None:
        target = target[:, : int(args.num_steps) + 1, :].contiguous()
    num_steps = int(target.shape[1]) - 1
    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)
    print(
        f"[data] train {tuple(target.shape)} lambda={ridge:g} steps={args.steps}",
        flush=True,
    )

    model, sigma_max = tt.build_model(
        grid=grid, device=device, seed=SWEEP_SEED, kernel_steps=num_steps
    )
    training = tt.build_training(
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
        # would tabulate the survivors as though the comparison had been clean.
        # Record it, then re-raise so the exit code is honest.
        if "not finite" not in str(exc):
            raise
        elapsed = time.perf_counter() - started
        SWEEP_DIR.mkdir(parents=True, exist_ok=True)
        out = SWEEP_DIR / f"lambda_{_tag(ridge)}.json"
        out.write_text(
            json.dumps(
                {
                    "ridge_lambda": ridge,
                    "label": f"lambda={ridge:g}",
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
        print(
            f"[diverged] lambda={ridge:g} after {elapsed / 60.0:.1f} min -> {out}",
            flush=True,
        )
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

    # ---- section-7 statistics: the criterion that actually decides ---------
    # `generated` is in LOG price space; every statistic in selection_true is
    # defined on PRICES, so the exponential happens exactly once, here.
    module = sel.load_criterion(DATASET)
    env = sel.envelope(data_dir=DATASET, seq_tag=SEQ_TAG)
    train_prices, val_prices = sel.load_splits(data_dir=DATASET, seq_tag=SEQ_TAG)
    gen_prices = np.exp(generated.detach().to(torch.float64).cpu().numpy())
    record = sel.score_candidate(
        gen_prices,
        train=train_prices,
        val=val_prices,
        module=module,
        median_nn_heldout=env["median_nn_heldout"],
    )
    reasons = sel.admissibility(record, env)
    print(
        f"[sec7] vol={record['vol_err_pct']:.2f}% (ceiling {env['vol_err_pct_max']:.2f}) "
        f"corr={record['corr_err']:.4f} (ceiling {env['corr_err_max']:.4f}) "
        f"NN={record['nn_ratio']:.4f} |logNN|={record['abs_log_nn_ratio']:.4f} "
        f"{'ADMISSIBLE' if not reasons else 'REJECTED: ' + '; '.join(reasons)}",
        flush=True,
    )

    # ---- training discrepancy: reported, never decides ---------------------
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

    val_discrepancy = _score(VAL_FILE)
    valdisc_discrepancy = _score(VALDISC_FILE)
    print(
        f"[fit-only] val discrepancy={val_discrepancy:.8g}   "
        f"valdisc={valdisc_discrepancy:.8g}   (REPORTED, DOES NOT DECIDE)",
        flush=True,
    )

    record.update(
        {
            "ridge_lambda": ridge,
            "label": f"lambda={ridge:g}",
            "steps": int(args.steps),
            "seed": SWEEP_SEED,
            "sigma_max": float(sigma_max),
            "val_discrepancy": val_discrepancy,
            "valdisc_discrepancy": valdisc_discrepancy,
            "ce_projection_r2": _history_mean(fit.history, "ce_projection_r2"),
            "noise_ce_projection_r2": _history_mean(
                fit.history, "noise_ce_projection_r2"
            ),
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
            # Non-zero means this lambda drove Theta into a spectrum batched
            # cuSOLVER could not decompose. The run is still valid -- the retry
            # is the same decomposition on CPU LAPACK -- but a lambda that needs
            # it often is operating in a numerically hard region, and that is
            # selection-relevant information, not a footnote.
            "eigh_fallback": eigh_fallback.counters(),
            "date": date.today().isoformat(),
        }
    )
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    out = SWEEP_DIR / f"lambda_{_tag(ridge)}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[out] {out}", flush=True)
    return record


def report() -> None:
    """Print the sweep table and name the winner under the section-7 rule."""

    files = sorted(SWEEP_DIR.glob("lambda_*.json"))
    if not files:
        raise SystemExit(f"ABORT: no sweep results in {SWEEP_DIR}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    records.sort(key=lambda rec: rec["ridge_lambda"])

    diverged = [rec for rec in records if rec.get("diverged")]
    scored = [rec for rec in records if not rec.get("diverged")]

    env = sel.envelope(data_dir=DATASET, seq_tag=SEQ_TAG)
    print(sel.report_envelope(env))
    print()

    winner, annotated = sel.select(scored, env)
    print(sel.format_table(annotated, env, label_key="label"))

    print("\n--- fit-only diagnostics (REPORTED, DID NOT DECIDE) ---")
    header = (
        f"{'lambda':>10} {'val_disc':>14} {'valdisc':>14} {'ce_r2':>8} "
        f"{'noise_r2':>9} {'s/step':>8} {'eta_3000h':>10}"
    )
    print(header)
    print("-" * len(header))
    for rec in records:
        if rec.get("diverged"):
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

    if diverged:
        lams = ", ".join(f"{rec['ridge_lambda']:g}" for rec in diverged)
        print(
            f"\n[warn] {len(diverged)} of {len(records)} candidates DIVERGED "
            f"(lambda = {lams}). The control produced non-finite values inside "
            f"{records[0]['steps']} steps. Divergence is not monotone in lambda, "
            "so surviving a short screen does not bound the full-length run."
        )

    if winner is None:
        raise SystemExit(
            "ABORT: no candidate is admissible under the real-vs-real envelope. "
            "That is the result. Widening the envelope to produce a winner would "
            "replace a measured ceiling with a chosen one and destroy the only "
            "property this criterion has -- zero free parameters. Widen the "
            "lambda grid, or train longer, instead."
        )

    if scored:
        best_fit = min(scored, key=lambda rec: rec["val_discrepancy"])
        if best_fit["ridge_lambda"] != winner["ridge_lambda"]:
            print(
                f"\n[note] the fit rule would have chosen lambda = "
                f"{best_fit['ridge_lambda']:g} (val_discrepancy "
                f"{best_fit['val_discrepancy']:.8g}, NNratio "
                f"{best_fit['nn_ratio']:.4f}). Section 7 rejects that rule; the "
                "disagreement is recorded, not resolved in its favour."
            )

    print(
        f"\nWINNER  ridge_lambda = {winner['ridge_lambda']:g}  "
        f"NNratio = {winner['nn_ratio']:.4f}  |logNN| = "
        f"{winner['abs_log_nn_ratio']:.4f}  vol_err = "
        f"{winner['vol_err_pct']:.2f}%  corr_err = {winner['corr_err']:.4f}"
    )

    stamped = dict(winner)
    stamped["candidates_compared"] = sorted(rec["ridge_lambda"] for rec in records)
    stamped["candidates_diverged"] = sorted(rec["ridge_lambda"] for rec in diverged)
    stamped["n_candidates"] = len(records)
    stamped["criterion"] = (
        "minimise |log NNratio| subject to vol_err <= real-vs-real max and "
        "corr_err <= real-vs-real max; tie-break within 0.02 on corr_err "
        "(TrueDataset guideline section 7)"
    )
    stamped["envelope"] = env
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEP_DIR / "winner.json").write_text(
        json.dumps(stamped, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[out] {SWEEP_DIR / 'winner.json'}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ridge-lambda", type=float, default=None)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-paths", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--score-paths", type=int, default=2048)
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
