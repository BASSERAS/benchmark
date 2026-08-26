#!/usr/bin/env python3
"""READ-ONLY: is training to 3000 steps useful at d = 8, or is it too much?

What this answers
-----------------
The paper trains 3000 steps and reports K = 2500 (Appendix B).  The d = 8
campaign copies that.  The question this script settles is whether the last
stretch of training buys anything on a split the model never trained on, or
whether it is wall-clock spent for nothing.

The evidence is already on disk: ``runs/seed_*/training_checkpoints/step_*.pt``
is a checkpoint every 500 steps.  Scoring each one on the validation bank with
the SAME discrepancy that trained the model gives the whole curve, so no new
training is needed.  Three shapes are distinguishable:

  * argmin at 3000            -> the curve was still improving; 3000 is not too
                                 much, and arguably not enough,
  * argmin in the interior    -> the tail is wasted for that seed,
  * flat after some step      -> the tail is wasted for every seed.

The answer is per seed and it need not agree across seeds.  Seed 2 selected
2500 and seed 0 selected 3000, so a single-seed answer is not an answer.

Why NOT ``select_checkpoint_multiasset.py``
-------------------------------------------
``select_one`` does not only score.  At lines 170-186 it PROMOTES the argmin
into ``weights/seed_{i}_model.pt`` and stamps ``selected_step`` into
``weights/seed_{i}_config.json``.  Pointing it at a seed that is still training
would promote a checkpoint from the middle of a run and write a selection record
that looks exactly like a final one.  Seeds 4, 5 and 6 are still training, so
that is not a hypothetical.

This script therefore reuses ``score_model`` -- the pure scoring function, no
side effects -- and writes ONLY to ``analysis/``.  It promotes nothing, and it
never touches ``weights/`` or ``selection/``.

Reading a checkpoint while the trainer runs is safe: ``train_multiasset`` writes
them through ``_save_atomic`` (line 367), so a partially written file is never
visible under its final name.

The TEST split is not opened.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 \
    PYTHONPATH="$R/src:$R/experiments" taskset -c 19 \
    /home/tbasseras/gpu-venv/bin/python scan_step_curve.py --seeds 4 5 6

    # afterwards, pooling in the finished seeds' own selection records
    python scan_step_curve.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
METHOD_ROOT = HERE.parent
REFERENCE = Path("/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference")
for _path in (HERE, REFERENCE / "src", REFERENCE / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402

import train_multiasset as tm  # noqa: E402
from fit_reference_multiasset import DT  # noqa: E402
from sweep_ridge_lambda import VAL_FILE, VALDISC_FILE  # noqa: E402

# Importing this module executes only constants and sys.path edits; its main()
# is guarded by __name__ == "__main__", so no selection or promotion happens.
from select_checkpoint_multiasset import SCORE_SEED, score_model  # noqa: E402

ANALYSIS_DIR = HERE / "analysis"
SELECTION_DIR = HERE / "selection"


def _pct(before: float, after: float) -> float:
    """Percent improvement going from ``before`` to ``after`` (positive = better)."""

    if before != before or after != after or before == 0.0:
        return float("nan")
    return 100.0 * (before - after) / before


def _summarise(seed: int, per_step: list[dict]) -> dict:
    """Turn a scored curve into the three numbers that answer the question."""

    finite = [
        r for r in per_step
        if float(r["val_discrepancy"]) == float(r["val_discrepancy"])
    ]
    if not finite:
        raise SystemExit(f"ABORT: seed {seed} produced no finite discrepancy")
    best = min(finite, key=lambda r: r["val_discrepancy"])
    last = max(finite, key=lambda r: r["step"])
    by_step = {int(r["step"]): float(r["val_discrepancy"]) for r in finite}

    return {
        "seed": int(seed),
        "per_step": per_step,
        "argmin_step": int(best["step"]),
        "last_step_scored": int(last["step"]),
        # The decisive flag. True means the curve had not turned over yet, so the
        # tail of training was NOT wasted for this seed.
        "argmin_is_last": int(best["step"]) == int(last["step"]),
        "best_val_discrepancy": float(best["val_discrepancy"]),
        # How much the whole run bought, relative to the earliest checkpoint.
        "gain_500_to_best_pct": (
            _pct(by_step[500], float(best["val_discrepancy"]))
            if 500 in by_step else float("nan")
        ),
        # The specific question: does the last 500 steps of the paper's schedule
        # help or hurt? Positive = 3000 beats 2500. None = not both scored yet.
        "gain_2500_to_3000_pct": (
            _pct(by_step[2500], by_step[3000])
            if 2500 in by_step and 3000 in by_step else None
        ),
        "selection_file": VAL_FILE,
        "secondary_file": VALDISC_FILE,
        "score_paths": None,
        "score_seed": SCORE_SEED,
        "read_only": True,
        "date": date.today().isoformat(),
    }


def scan_one(seed: int, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    ckpt_dir = Path(args.run_root) / f"seed_{int(seed)}" / "training_checkpoints"
    # `latest.pt` is deliberately excluded: it is not on the 500-step grid and
    # its step number is unknown from the filename, so it cannot be placed on a
    # curve. select_checkpoint_multiasset ignores it for the same reason.
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    if not checkpoints:
        raise SystemExit(f"ABORT: no checkpoints in {ckpt_dir}")

    probe = tm.load_log_prices(num_paths=1, device="cpu")
    num_steps = int(probe.shape[1]) - 1
    del probe
    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)
    model = tm.build_model(
        grid=grid, device=device, seed=int(seed), kernel_steps=num_steps
    )

    per_step: list[dict[str, float]] = []
    for path in checkpoints:
        step = int(path.stem.split("_")[1])
        model.load_checkpoint_state(
            torch.load(path, map_location=device, weights_only=True)
        )
        started = time.perf_counter()
        value = score_model(
            model, grid=grid, device=device,
            score_paths=int(args.score_paths), filename=VAL_FILE,
        )
        secondary = score_model(
            model, grid=grid, device=device,
            score_paths=int(args.score_paths), filename=VALDISC_FILE,
        )
        per_step.append(
            {"step": step, "val_discrepancy": value, "valdisc_discrepancy": secondary}
        )
        print(
            f"  seed={seed} step={step:5d} val={value:.8g} "
            f"valdisc={secondary:.8g}  ({time.perf_counter() - started:.1f}s)",
            flush=True,
        )

    record = _summarise(int(seed), per_step)
    record["score_paths"] = int(args.score_paths)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS_DIR / f"step_curve_seed_{int(seed)}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  [out] {out}", flush=True)
    return record


def _load_all() -> list[dict]:
    """Curves from analysis/, plus the finished seeds' own selection records.

    A seed that has already been selected has an identical curve sitting in
    ``selection/seed_{i}_selection.json`` under the same ``per_step`` key.
    Re-scoring it would burn GPU to reproduce a number already on disk, so it is
    read instead.  ``analysis/`` wins on conflict, being the more recent scan.
    """

    curves: dict[int, dict] = {}
    for path in sorted(SELECTION_DIR.glob("seed_*_selection.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if "per_step" not in rec:
            continue
        summary = _summarise(int(rec["seed"]), rec["per_step"])
        summary["source"] = "selection"
        curves[int(rec["seed"])] = summary
    for path in sorted(ANALYSIS_DIR.glob("step_curve_seed_*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["source"] = "analysis"
        curves[int(rec["seed"])] = rec
    return [curves[k] for k in sorted(curves)]


def report() -> None:
    curves = _load_all()
    if not curves:
        raise SystemExit(f"ABORT: no curves in {ANALYSIS_DIR} or {SELECTION_DIR}")

    steps = sorted({int(r["step"]) for c in curves for r in c["per_step"]})
    head = "seed  " + "".join(f"{s:>11}" for s in steps) + "   argmin  still_improving"
    print("val discrepancy per checkpoint (lower is better)\n")
    print(head)
    print("-" * len(head))
    for c in curves:
        by = {int(r["step"]): float(r["val_discrepancy"]) for r in c["per_step"]}
        best = c["argmin_step"]
        cells = ""
        for s in steps:
            if s not in by:
                cells += f"{'--':>11}"
            else:
                cells += f"{by[s]:>10.6f}" + ("*" if s == best else " ")
        flag = "YES" if c["argmin_is_last"] else "no"
        print(f"{c['seed']:>4}  {cells}   {best:>6}  {flag:>15}")
    print("\n* = best checkpoint for that seed")

    # Aggregate verdict.
    complete = [c for c in curves if c["last_step_scored"] >= 3000]
    print(
        f"\nseeds with the full 3000-step grid scored: {[c['seed'] for c in complete]}"
    )
    if complete:
        argmins = [c["argmin_step"] for c in complete]
        n_last = sum(1 for c in complete if c["argmin_step"] == 3000)
        print(f"  argmin steps: {argmins}")
        print(f"  best at the FINAL step: {n_last} of {len(complete)} seeds")
        gains = [
            c["gain_2500_to_3000_pct"]
            for c in complete
            if c["gain_2500_to_3000_pct"] is not None
        ]
        if gains:
            mean = sum(gains) / len(gains)
            print(
                f"  step 2500 -> 3000 change: "
                f"{', '.join(f'{g:+.1f}%' for g in gains)}  (mean {mean:+.1f}%)"
            )
            print("  (positive = 3000 is better than 2500)")
        early = [
            c["gain_500_to_best_pct"] for c in complete
            if c["gain_500_to_best_pct"] == c["gain_500_to_best_pct"]
        ]
        if early:
            print(
                f"  step 500 -> best improvement: "
                f"{', '.join(f'{g:+.1f}%' for g in early)}"
            )
        print()
        if n_last == len(complete):
            print("VERDICT: every seed's best checkpoint is the LAST one. Training "
                  "to 3000 is not too much -- the curve had not turned over.")
        elif n_last == 0:
            print("VERDICT: no seed's best checkpoint is the last one. The tail of "
                  "training is wasted; an earlier stop would lose nothing.")
        else:
            print("VERDICT: MIXED -- some seeds peak at 3000, others earlier. The "
                  "budget is not wasted, but the best step is seed-dependent, so "
                  "per-seed validation selection is doing real work and a fixed "
                  "stopping step would be wrong for some seeds.")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[4, 5, 6])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--score-paths", type=int, default=8192)
    p.add_argument("--run-root", type=Path, default=HERE / "runs")
    p.add_argument("--report", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.report:
        report()
        return
    for seed in args.seeds:
        scan_one(int(seed), args)
    print()
    report()


if __name__ == "__main__":
    main()
