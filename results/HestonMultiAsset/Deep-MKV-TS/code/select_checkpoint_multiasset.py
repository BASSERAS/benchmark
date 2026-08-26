#!/usr/bin/env python3
"""Pick the reported checkpoint per seed on the VALIDATION split (d = 8).

Why this file exists
--------------------
The committed d = 1 Deep-MKV-TS run does not report the last optimizer step; it
reports step 2500 of a 3000-step run, chosen on validation.  Reproducing the
method means reproducing that choice, so this script does at d = 8 exactly what
was done at d = 1: score every saved checkpoint on the validation split with the
SAME discrepancy that trained the model, and keep the argmin.

``MULTIASSET_GUIDELINE.md`` section 0.2 names ``heston_ma_S_val_8192x252x8.npy``
as the selection split.  ``_valdisc_`` is scored alongside as a reported-only
robustness check and never decides.  The TEST split is not opened by this file.

Why the full checkpoint and not ``network.state_dict()``
-------------------------------------------------------
``DiscreteMPModel.checkpoint_state()`` says it plainly: the timewise target
scales and the noise-target baseline feed the adjoint moments handed to the
control map, so a bare ``state_dict`` does NOT reproduce a fitted model's
rollouts.  This script therefore rewrites ``weights/seed_{i}_model.pt`` with the
selected checkpoint's full payload, replacing the provisional final-step payload
the trainer wrote.  ``run_all_multiasset.py`` restores it with
``load_checkpoint_state``.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
    PYTHONPATH="$R/src:$R/experiments" taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python select_checkpoint_multiasset.py --seeds 0 2 4 5 6
"""

from __future__ import annotations

import argparse
import json
import math
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
from fit_reference_multiasset import DT, STATE_DIM  # noqa: E402
from sweep_ridge_lambda import VAL_FILE, VALDISC_FILE, load_split  # noqa: E402

SELECTION_DIR = HERE / "selection"
SCORE_SEED = 20260826
S0 = 100.0


def score_model(
    model,
    *,
    grid: DiscreteTimeGrid,
    device: torch.device,
    score_paths: int,
    filename: str,
) -> float:
    """Discrepancy of ``score_paths`` sampled paths against ``filename``."""

    x0 = torch.full((1, STATE_DIM), math.log(S0), device=device, dtype=model.dtype)
    with torch.no_grad():
        generated = model.sample(
            num_paths=int(score_paths), x0=x0, seed=SCORE_SEED
        ).paths
    if not torch.isfinite(generated).all():
        return float("nan")
    bank = load_split(
        filename,
        num_paths=int(score_paths),
        num_steps=int(grid.num_steps),
        device=device,
        dtype=model.dtype,
    )
    with torch.no_grad():
        scored = model.discrepancy(
            generated_paths=generated, target_paths=bank, grid=grid
        )
    value = float(scored.value if hasattr(scored, "value") else scored)
    del generated, bank
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return value


def select_one(seed: int, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    run_dir = Path(args.run_root) / f"seed_{int(seed)}"
    ckpt_dir = run_dir / "training_checkpoints"
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    if not checkpoints:
        raise SystemExit(f"ABORT: no checkpoints in {ckpt_dir}")

    # The grid must match training exactly, so read its width off the train split.
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
        # weights_only=True: checkpoint_state() emits only tensors, plain dicts
        # and primitives -- verified by round-tripping a real payload through
        # save/load/load_checkpoint_state -- so there is no reason to allow
        # arbitrary unpickling here.
        model.load_checkpoint_state(
            torch.load(path, map_location=device, weights_only=True)
        )
        started = time.perf_counter()
        value = score_model(
            model,
            grid=grid,
            device=device,
            score_paths=int(args.score_paths),
            filename=VAL_FILE,
        )
        secondary = score_model(
            model,
            grid=grid,
            device=device,
            score_paths=int(args.score_paths),
            filename=VALDISC_FILE,
        )
        per_step.append(
            {
                "step": step,
                "val_discrepancy": value,
                "valdisc_discrepancy": secondary,
            }
        )
        print(
            f"  seed={seed} step={step:5d} val={value:.8g} "
            f"valdisc={secondary:.8g}  ({time.perf_counter() - started:.1f}s)",
            flush=True,
        )

    finite = [
        rec
        for rec in per_step
        if float(rec["val_discrepancy"]) == float(rec["val_discrepancy"])
    ]
    if not finite:
        raise SystemExit(f"ABORT: seed {seed} produced no finite discrepancy")
    best = min(finite, key=lambda rec: rec["val_discrepancy"])
    selected_step = int(best["step"])
    print(
        f"  seed={seed} SELECTED step={selected_step} "
        f"val={best['val_discrepancy']:.8g}",
        flush=True,
    )

    # Promote the selected checkpoint into the reported weights slot.
    selected_path = ckpt_dir / f"step_{selected_step:04d}.pt"
    payload = torch.load(selected_path, map_location="cpu", weights_only=True)
    weights_path = METHOD_ROOT / "weights" / f"seed_{int(seed)}_model.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, weights_path)

    # Record the choice inside the config the README reads.
    config_path = METHOD_ROOT / "weights" / f"seed_{int(seed)}_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["selected_step"] = selected_step
        config["selection_file"] = VAL_FILE
        config["selection_val_discrepancy"] = float(best["val_discrepancy"])
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    record = {
        "seed": int(seed),
        "selected_step": selected_step,
        "val_discrepancy": float(best["val_discrepancy"]),
        "valdisc_discrepancy": float(best["valdisc_discrepancy"]),
        "per_step": per_step,
        "selection_file": VAL_FILE,
        "secondary_file": VALDISC_FILE,
        "score_paths": int(args.score_paths),
        "score_seed": SCORE_SEED,
        "date": date.today().isoformat(),
    }
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    out = SELECTION_DIR / f"seed_{int(seed)}_selection.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  [out] {out}", flush=True)
    return record


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Not range(5): TWO seeds diverged with a non-finite control -- seed 3 at
    # step 2, seed 1 at step ~1450 -- and were replaced by seeds 5 and 6. See
    # "The caveat came true twice" in README.md. There is no seed-3 run directory
    # to select from at all; seed 1 has step_0500.pt and step_1000.pt but died
    # before the 3000-step grid completed, so it is not a selection candidate.
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 2, 4, 5, 6])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score-paths", type=int, default=8192)
    parser.add_argument("--run-root", type=Path, default=HERE / "runs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    records = [select_one(int(seed), args) for seed in args.seeds]
    print("\n seed  selected_step   val_discrepancy")
    print("-" * 40)
    for rec in records:
        print(
            f"{rec['seed']:>5}  {rec['selected_step']:>13}   "
            f"{rec['val_discrepancy']:.8g}"
        )
    steps = sorted({rec["selected_step"] for rec in records})
    print(f"\nselected steps across seeds: {steps}")


if __name__ == "__main__":
    main()
