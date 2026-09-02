#!/usr/bin/env python3
"""Iterative one-factor hyperparameter search for Deep-MKV-TS with the SBTS reference.

Protocol (fixed by instruction: one seed, small subset, fast, hill-climb)
------------------------------------------------------------------------
The search is NOT a grid.  It is a sequence of STAGES.  Within a stage exactly
one factor varies and every other factor is held at the current incumbent; the
stage winner becomes the incumbent for the next stage.  Stages are run in this
order, chosen so the cheapest and most influential knob moves first:

    stage 1  lr                {5e-4, 1e-3, 2e-3, 4e-3}
    stage 2  ridge_lambda      {1e2, 1e3, 1e4}
    stage 3  h                 {0.20, 0.26, 0.31, 0.38, 0.46}
    stage 4  npi               {1, 50}          (both with grad=detached, so the
                                                 comparison is one-factor)
    stage 5  weight_grad_mode  {detached, analytic}
    stage 6  drift_correction  {off, on}

Each arm trains SEED 0 for ``--steps`` steps on a ``--num-paths`` subset of the
TRAIN split, samples a bank, and scores the SAME ``build_heston_discrepancy``
used as the training objective against ``heston_ma_S_val_8192x252x8.npy``.
Lowest validation discrepancy wins; MULTIASSET_GUIDELINE.md section 0.2 names
that file as THE selection split.  ``heston_ma_S_valdisc_8192x252x8.npy`` is
scored alongside purely to show the ranking is not an artefact of one draw and
never decides.  The TEST split is never opened.

The SBTS bank is always the FULL 8192-path train split even when
``--num-paths`` shrinks the discrepancy target: shrinking the bank would change
the reference model itself, so a subset sweep would be selecting for the wrong
reference.

Everything except the swept factor is imported from ``train_multiasset.py``, so
the sweep cannot silently drift from the campaign configuration.

Sampling is CHUNKED (``--score-chunk``) because the SBTS weight tensor is
``(K, B, M)``: at ``B = M = 8192, K = 20`` that is 5.4 GB per lag-stacked
intermediate and several are alive at once.  The chunk size and the per-chunk
seeds are fixed constants, so every arm is scored against the same noise draw.

Usage
-----
    # one arm
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python sweep_hyperparams.py \
        --stage lr --lr 1e-3 --device cuda:0

    # tabulate a stage, name its winner, promote it to the incumbent
    python sweep_hyperparams.py --report --stage lr --promote
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

eigh_fallback.install()

VAL_FILE = "heston_ma_S_val_8192x252x8.npy"
VALDISC_FILE = "heston_ma_S_valdisc_8192x252x8.npy"
SWEEP_DIR = HERE / "sweep"
SWEEP_SEED = 0
SCORE_SEED = 20260831
S0 = 100.0

FACTORS = (
    "lr",
    "ridge_lambda",
    "h",
    "markov_order",
    "npi",
    "weight_grad_mode",
    "jacobian_lags",
    "allow_drift_correction",
)


def start_incumbent() -> dict:
    """The STARTING incumbent; ``sweep/incumbent.json`` overrides it per stage."""

    return {
        "lr": tm.DEFAULT_LR,
        "ridge_lambda": tm.DEFAULT_RIDGE_LAMBDA,
        "h": tm.DEFAULT_H,
        "markov_order": tm.DEFAULT_MARKOV_ORDER,
        "npi": tm.DEFAULT_NPI,
        # Stages 1-4 vary things that have nothing to do with the drift gradient,
        # so they run with the CHEAP and unambiguous backward (zero db^ref/dx).
        # Stage 5 then decides whether the analytic Jacobian is worth its cost,
        # holding everything else at the stage-4 incumbent.
        "weight_grad_mode": "detached",
        # -1 = every one of the `markov_order` lags enters the adjoint, i.e. the
        # exact gradient.  The first 72-arm sweep ran before this factor existed,
        # against a hard-wired 1-lag truncation that was wrong by 100% relative
        # error, so its choice of `h` is not evidence about the corrected code.
        "jacobian_lags": -1,
        "allow_drift_correction": False,
    }


def load_incumbent() -> dict:
    values = start_incumbent()
    path = SWEEP_DIR / "incumbent.json"
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        for key in FACTORS:
            if key in stored:
                values[key] = stored[key]
    return values


def _tag(values: dict) -> str:
    """Filesystem-safe arm tag naming every factor, so no two arms collide."""

    tag = (
        f"lr{float(values['lr']):.3e}"
        f"_rl{float(values['ridge_lambda']):.0e}"
        f"_h{float(values['h']):.3f}"
        f"_K{int(values['markov_order'])}"
        f"_npi{int(values['npi'])}"
        f"_{values['weight_grad_mode']}"
        f"_dc{int(bool(values['allow_drift_correction']))}"
    ).replace("+", "")
    # `jacobian_lags` did not exist when the first 72 arms were written, and the
    # code they ran against was hard-wired to a single lag.  So "no suffix"
    # means jl1, and only other values take a suffix: old filenames stay
    # byte-for-byte comparable while a corrected-adjoint arm can never silently
    # overwrite the truncated arm it is supposed to be compared against.
    jl = int(values.get("jacobian_lags", 1))
    if jl != 1:
        tag = f"{tag}_jl{jl}"
    return tag


def load_split(
    filename: str,
    *,
    num_paths: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """A price bank as LOG prices, shape ``(N, T + 1, d)``."""

    prices = np.load(tm.DATASET / filename, mmap_mode="r")
    array = np.asarray(
        prices[: int(num_paths), : int(num_steps) + 1, :], dtype=np.float64
    )
    if array.ndim != 3 or int(array.shape[2]) != tm.STATE_DIM:
        raise ValueError(f"expected (N, T + 1, {tm.STATE_DIM}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{filename} contains non-finite prices")
    if float(array.min()) <= 0.0:
        raise ValueError(f"{filename} contains non-positive prices")
    return torch.log(torch.from_numpy(array)).to(device=device, dtype=dtype)


def _history_mean(history: list[dict], key: str, tail: int = 5) -> float:
    values = [
        float(rec[key])
        for rec in history
        if key in rec and float(rec[key]) == float(rec[key])
    ]
    if not values:
        return float("nan")
    window = values[-int(tail) :]
    return float(sum(window) / len(window))


def _sample_chunked(model, *, num_paths: int, chunk: int, device) -> torch.Tensor:
    """Sample ``num_paths`` paths in fixed-size chunks with fixed per-chunk seeds."""

    x0 = torch.full((1, tm.STATE_DIM), math.log(S0), device=device, dtype=model.dtype)
    banks = []
    remaining = int(num_paths)
    index = 0
    while remaining > 0:
        size = min(int(chunk), remaining)
        with torch.no_grad():
            banks.append(
                model.sample(num_paths=size, x0=x0, seed=SCORE_SEED + index).paths
            )
        remaining -= size
        index += 1
    return torch.cat(banks, dim=0)


def run_one(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    values = load_incumbent()
    for key in FACTORS:
        override = getattr(args, key, None)
        if override is not None:
            values[key] = override
    tag = _tag(values)
    # The tag is the only thing stopping two arms from overwriting each other,
    # and it encodes every hyperparameter but NOT the training seed.  A repeat
    # of one configuration under several seeds would therefore collide on a
    # single filename and silently leave only whichever run finished last --
    # destroying exactly the spread a noise calibration exists to measure.
    # Suffix the seed, but only when it differs from the default, so every arm
    # written before this flag existed keeps its filename byte-for-byte and
    # stays comparable.
    seed = int(args.seed)
    if seed != SWEEP_SEED:
        tag = f"{tag}_s{seed}"

    bank_prices = tm.load_prices()
    target_prices = (
        bank_prices if args.num_paths is None else bank_prices[: int(args.num_paths)]
    )
    target = torch.log(torch.from_numpy(np.ascontiguousarray(target_prices))).to(
        device=device, dtype=tm.DTYPE
    )
    if args.num_steps is not None:
        target = target[:, : int(args.num_steps) + 1, :].contiguous()
        bank_prices = bank_prices[:, : int(args.num_steps) + 1, :]
    num_steps = int(target.shape[1]) - 1
    grid = DiscreteTimeGrid(T=num_steps * tm.DT, num_steps=num_steps)
    print(f"[arm] stage={args.stage} {tag}", flush=True)
    print(
        f"[data] target {tuple(target.shape)} bank {tuple(bank_prices.shape)} "
        f"steps={args.steps}",
        flush=True,
    )

    model, kernel = tm.build_model(
        grid=grid,
        device=device,
        seed=seed,
        bank_prices=bank_prices,
        h=float(values["h"]),
        markov_order=int(values["markov_order"]),
        npi=int(values["npi"]),
        weight_grad_mode=str(values["weight_grad_mode"]),
        jacobian_lags=int(values["jacobian_lags"]),
        allow_drift_correction=bool(values["allow_drift_correction"]),
    )
    training = tm.build_training(
        seed=seed,
        steps=int(args.steps),
        lr=float(values["lr"]),
        ridge_lambda=float(values["ridge_lambda"]),
    )
    # An override that never reached the config would turn the whole sweep into a
    # relabelling of one identical run. Fail loudly instead of tabulating noise.
    if float(training.lr) != float(values["lr"]):
        raise RuntimeError("lr override did not reach the training config")
    if float(training.ridge_lambda) != float(values["ridge_lambda"]):
        raise RuntimeError("ridge_lambda override did not reach the training config")
    if float(kernel.h) != float(values["h"]) or int(kernel.npi) != int(values["npi"]):
        raise RuntimeError("reference override did not reach the SBTS kernel")
    # Without this the whole point of the re-sweep evaporates: every arm would
    # quietly run the truncated adjoint again and the table would be a relabelling
    # of the sweep it is meant to supersede.
    if int(kernel.jacobian_lags) != int(values["jacobian_lags"]):
        raise RuntimeError(
            f"jacobian_lags override did not reach the SBTS kernel: asked "
            f"{int(values['jacobian_lags'])}, kernel has {int(kernel.jacobian_lags)}"
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    out = SWEEP_DIR / f"{args.stage}__{tag}.json"

    started = time.perf_counter()
    try:
        fit = model.fit(target, training=training)
    except RuntimeError as exc:
        # An arm whose control diverges is a RESULT, not a missing data point.
        # Left uncaught it would simply be absent from sweep/ and `report()`
        # would tabulate the survivors as though the stage had been complete.
        if "not finite" not in str(exc):
            raise
        elapsed = time.perf_counter() - started
        out.write_text(
            json.dumps(
                {
                    "stage": args.stage,
                    "tag": tag,
                    **values,
                    "steps": int(args.steps),
                    "seed": seed,
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
        print(f"[diverged] {tag} after {elapsed / 60.0:.1f} min -> {out}", flush=True)
        raise
    elapsed = time.perf_counter() - started
    sec_per_step = elapsed / float(args.steps)
    print(f"[fit] {elapsed / 60.0:.1f} min = {sec_per_step:.3f} s/step", flush=True)

    del target
    if device.type == "cuda":
        torch.cuda.empty_cache()

    generated = _sample_chunked(
        model,
        num_paths=int(args.score_paths),
        chunk=int(args.score_chunk),
        device=device,
    )
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

    val_discrepancy = _score(VAL_FILE)
    valdisc_discrepancy = _score(VALDISC_FILE)
    print(
        f"[val] {VAL_FILE} discrepancy={val_discrepancy:.8g}   "
        f"[check] {VALDISC_FILE} discrepancy={valdisc_discrepancy:.8g}",
        flush=True,
    )

    record = {
        "stage": args.stage,
        "tag": tag,
        **values,
        "steps": int(args.steps),
        "seed": seed,
        "target_paths": int(args.num_paths or bank_prices.shape[0]),
        "bank_paths": int(bank_prices.shape[0]),
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
        "score_chunk": int(args.score_chunk),
        "score_seed": SCORE_SEED,
        "selection_file": VAL_FILE,
        "secondary_file": VALDISC_FILE,
        "eigh_fallback": eigh_fallback.counters(),
        "date": date.today().isoformat(),
    }
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[out] {out}", flush=True)
    return record


def _neighbour_worst_case(records: list[dict], stage: str) -> list[dict]:
    """Attach, to each numeric arm, the worst score in its 3-point neighbourhood.

    Why this exists
    ---------------
    The raw minimiser of a one-dimensional screen is the WRONG thing to promote
    into a 5-seed campaign when the curve has a cliff.  Measured here: lr = 5e-5
    scores 0.0806 and lr = 1e-5 -- one grid point below -- scores 0.1379, 71%
    worse.  A point whose neighbour falls off a cliff is a point where a seed
    change, a longer horizon or a slightly different bank draw can land on the
    bad side.

    So each arm is also scored by ``max(self, below, above)``: the worst it
    could do if the effective learning rate drifted one grid step in either
    direction.  Selecting on THAT picks the flattest safe point rather than the
    luckiest one.  Endpoints have only one neighbour and are marked, because an
    endpoint's neighbourhood is not comparable to an interior point's -- the
    grid simply has not looked at what lies beyond it.

    The rule is fixed in code before the remaining arms land, so the choice
    cannot be a post-hoc rationalisation of whichever number came out nicest.
    """

    numeric = []
    for rec in records:
        if rec.get("diverged"):
            continue
        try:
            value = float(rec[stage])
        except (KeyError, TypeError, ValueError):
            return []
        score = float(rec["val_discrepancy"])
        if score != score:
            continue
        numeric.append((value, score, rec))
    if len(numeric) < 3:
        return []
    numeric.sort(key=lambda item: item[0])

    annotated = []
    for index, (value, score, rec) in enumerate(numeric):
        below = numeric[index - 1][1] if index > 0 else None
        above = numeric[index + 1][1] if index + 1 < len(numeric) else None
        window = [score] + [v for v in (below, above) if v is not None]
        enriched = dict(rec)
        enriched["neighbour_worst"] = max(window)
        # The TRADE-OFF score: the arm's own value, robustified by half a grid
        # step of drift in each direction.  Weights (1/4, 1/2, 1/4) are the
        # standard 3-point smoothing kernel -- the arm still dominates its own
        # score, so a genuinely good point is not thrown away, but a cliff on
        # either side is priced in instead of being invisible (raw minimum) or
        # totally dominant (minimax).  Endpoints reuse their one neighbour on
        # both sides, which is the least favourable honest assumption available
        # without arms the grid has not run.
        low = below if below is not None else above
        high = above if above is not None else below
        enriched["neighbour_smoothed"] = 0.25 * low + 0.5 * score + 0.25 * high
        enriched["neighbour_below"] = below
        enriched["neighbour_above"] = above
        enriched["is_endpoint"] = below is None or above is None
        annotated.append(enriched)
    return annotated


def report(stage: str, *, promote: bool, select: str = "smoothed") -> None:
    """Tabulate one stage, name its winner, and optionally promote it.

    ``select`` chooses where to sit on the optimality/stability axis:

        "raw"       argmin of the arm's own validation discrepancy.  Maximum
                    optimality, zero robustness -- it will happily pick a point
                    whose neighbour is a cliff.
        "smoothed"  argmin of ``(below + 2*self + above) / 4``.  THE TRADE-OFF,
                    and the default: the arm's own score still carries half the
                    weight, so a genuinely better point wins, but a cliff on
                    either side is priced in.
        "minimax"   argmin of ``max(below, self, above)``.  Maximum robustness,
                    and it will refuse a clearly better point over one bad
                    neighbour.

    Under "smoothed" and "minimax" only INTERIOR arms are eligible: an endpoint
    has an unexplored side, so its neighbourhood statistic is optimistic by
    construction and not comparable.
    """

    files = sorted(SWEEP_DIR.glob(f"{stage}__*.json"))
    if not files:
        raise SystemExit(f"ABORT: no arms for stage {stage!r} in {SWEEP_DIR}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    records.sort(key=lambda rec: str(rec.get(stage, rec["tag"])))

    print(f"* selection criterion: {VAL_FILE} (MULTIASSET_GUIDELINE.md 0.2)")
    print(f"  secondary, never decides: {VALDISC_FILE}\n")
    header = (
        f"{stage:>18} {'val_disc*':>14} {'valdisc':>14} {'ce_r2':>8} "
        f"{'s/step':>8} {'eta_3000h':>10}"
    )
    print(header)
    print("-" * len(header))
    for rec in records:
        label = str(rec.get(stage, "?"))
        if rec.get("diverged"):
            # Printed on its own row so the table never silently shrinks to the
            # survivors: a reader must be able to see that this arm was tried.
            print(
                f"{label:>18} {'DIVERGED':>14} {'--':>14} {'--':>8} {'--':>8} {'--':>10}"
            )
            continue
        print(
            f"{label:>18} {rec['val_discrepancy']:>14.8g} "
            f"{rec.get('valdisc_discrepancy', float('nan')):>14.8g} "
            f"{rec['ce_projection_r2']:>8.4f} {rec['sec_per_step']:>8.3f} "
            f"{rec['eta_3000_hours']:>10.2f}"
        )

    diverged = [rec for rec in records if rec.get("diverged")]
    finite = [
        rec
        for rec in records
        if not rec.get("diverged")
        and float(rec["val_discrepancy"]) == float(rec["val_discrepancy"])
    ]
    if diverged:
        print(
            f"\n[warn] {len(diverged)} of {len(records)} arms DIVERGED inside "
            f"{records[0]['steps']} steps. Divergence is not monotone in these "
            "factors, so surviving a short screen does not bound a full-length run."
        )
    if not finite:
        raise SystemExit("ABORT: every arm produced a non-finite discrepancy")

    raw_best = min(finite, key=lambda rec: rec["val_discrepancy"])
    best = raw_best
    rule = "argmin val_discrepancy"

    key = {"smoothed": "neighbour_smoothed", "minimax": "neighbour_worst"}.get(select)
    annotated = _neighbour_worst_case(finite, stage) if key else []
    if key and annotated:
        print(
            f"\n{'':>18} {'val_disc':>14} {'smoothed':>14} {'nbr_worst':>14} "
            f"{'below':>12} {'above':>12}"
        )
        for rec in annotated:
            mark = " (endpoint)" if rec["is_endpoint"] else ""
            below = rec["neighbour_below"]
            above = rec["neighbour_above"]
            print(
                f"{str(rec.get(stage)):>18} {rec['val_discrepancy']:>14.8g} "
                f"{rec['neighbour_smoothed']:>14.8g} {rec['neighbour_worst']:>14.8g} "
                f"{('--' if below is None else f'{below:.6g}'):>12} "
                f"{('--' if above is None else f'{above:.6g}'):>12}{mark}"
            )
        interior = [rec for rec in annotated if not rec["is_endpoint"]]
        if interior:
            best = min(interior, key=lambda rec: rec[key])
            rule = f"argmin {key} over interior arms ({select})"
        else:
            print(
                "\n[warn] every arm is a grid endpoint; falling back to the raw "
                "minimum. Widen the grid before trusting this."
            )

    print(
        f"\nSTAGE WINNER  {stage} = {best.get(stage)}  "
        f"val_discrepancy = {best['val_discrepancy']:.8g}"
        + (
            f"  smoothed = {best['neighbour_smoothed']:.8g}"
            f"  neighbour_worst = {best['neighbour_worst']:.8g}"
            if "neighbour_worst" in best
            else ""
        )
    )
    print(f"  rule: {rule}")
    if key and best is not raw_best:
        # State the price of stability explicitly. A reader must be able to see
        # what was given up, not just what was chosen.
        cost = best["val_discrepancy"] / float(raw_best["val_discrepancy"]) - 1.0
        print(
            f"  the raw minimiser was {stage} = {raw_best.get(stage)} at "
            f"{raw_best['val_discrepancy']:.8g}; the stable choice costs "
            f"{100.0 * cost:+.1f}% on the screen and buys a flat neighbourhood."
        )

    stamped = dict(best)
    stamped["selection_rule"] = rule
    stamped["raw_minimiser"] = raw_best.get(stage)
    stamped["raw_minimum"] = float(raw_best["val_discrepancy"])
    stamped["candidates_compared"] = [rec.get(stage) for rec in records]
    stamped["candidates_diverged"] = [rec.get(stage) for rec in diverged]
    stamped["n_candidates"] = len(records)
    (SWEEP_DIR / f"winner_{stage}.json").write_text(
        json.dumps(stamped, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[out] {SWEEP_DIR / f'winner_{stage}.json'}")

    if promote:
        # The hill-climb step: this stage's winner becomes the incumbent every
        # later stage holds fixed. Written explicitly rather than inferred, so
        # the search path is reconstructible from the directory alone.
        values = load_incumbent()
        values[stage] = best.get(stage)
        values["promoted_from"] = f"winner_{stage}.json"
        values["promoted_on"] = date.today().isoformat()
        (SWEEP_DIR / "incumbent.json").write_text(
            json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[promote] incumbent {stage} = {best.get(stage)}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ridge-lambda", type=float, default=None, dest="ridge_lambda")
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--markov-order", type=int, default=None, dest="markov_order")
    parser.add_argument("--npi", type=int, default=None)
    parser.add_argument(
        "--weight-grad-mode",
        choices=("analytic", "detached"),
        default=None,
        dest="weight_grad_mode",
    )
    parser.add_argument(
        "--jacobian-lags",
        type=int,
        default=None,
        dest="jacobian_lags",
        help="trailing lags of d b^ref/dx carried by the adjoint; -1 = all "
        "markov_order of them (the exact gradient), 1 = the truncated ablation "
        "the first sweep unknowingly ran against.",
    )
    parser.add_argument(
        "--allow-drift-correction",
        type=int,
        choices=(0, 1),
        default=None,
        dest="allow_drift_correction",
    )
    parser.add_argument("--steps", type=int, default=250)
    # Training seed only.  SCORE_SEED is deliberately NOT exposed: scoring must
    # stay deterministic so that the spread across --seed measures run-to-run
    # variation of the optimiser, not of the sampler used to grade it.
    parser.add_argument("--seed", type=int, default=SWEEP_SEED)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-paths", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--score-paths", type=int, default=4096)
    parser.add_argument("--score-chunk", type=int, default=1024)
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="with --report: write the stage winner into sweep/incumbent.json",
    )
    parser.add_argument(
        "--select",
        choices=("raw", "smoothed", "minimax"),
        default="smoothed",
        help=(
            "where to sit on the optimality/stability axis: raw = pure minimum, "
            "smoothed = (below + 2*self + above)/4 trade-off (default), "
            "minimax = worst case over the 3-point neighbourhood"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.allow_drift_correction is not None:
        args.allow_drift_correction = bool(args.allow_drift_correction)
    if args.report:
        report(args.stage, promote=bool(args.promote), select=str(args.select))
        return
    run_one(args)


if __name__ == "__main__":
    main()
