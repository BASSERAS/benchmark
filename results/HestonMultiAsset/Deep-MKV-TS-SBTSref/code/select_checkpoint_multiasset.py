#!/usr/bin/env python3
"""Pick the reported checkpoint per seed on the VALIDATION split (SBTS reference).

Why this file exists
--------------------
The committed Deep-MKV-TS runs do not report the last optimizer step; they report
the checkpoint that scored best on validation.  Reproducing the method means
reproducing that choice, so this script scores every saved checkpoint with the
SAME discrepancy that trained the model and keeps the argmin.

``MULTIASSET_GUIDELINE.md`` section 0.2 names ``heston_ma_S_val_8192x252x8.npy``
as the selection split.  ``_valdisc_`` is scored alongside as a reported-only
robustness check and never decides.  The TEST split is not opened by this file.

Why this is not a copy of the sibling's version
-----------------------------------------------
The sibling does ``from fit_reference_multiasset import DT, STATE_DIM`` and calls
``tm.build_model(grid=..., device=..., seed=..., kernel_steps=...)``.  Neither
applies here: there is no ``fit_reference_multiasset`` module, and this method's
``build_model`` needs the SBTS bank plus ``(h, markov_order, npi,
weight_grad_mode, allow_drift_correction)`` and returns a ``(model, kernel)``
tuple.  Those knobs are read back from the run's own ``config.json`` rather than
re-defaulted, so the selection model is the model that was trained even if the
module defaults move later.

Why the full checkpoint and not ``network.state_dict()``
-------------------------------------------------------
``DiscreteMPModel.checkpoint_state()`` carries the timewise target scales and the
noise-target baseline, which feed the adjoint moments handed to the control map.
A bare ``state_dict`` does NOT reproduce a fitted model's rollouts.  This script
therefore rewrites ``weights/seed_{i}_model.pt`` with the selected checkpoint's
full payload, replacing the provisional final-step payload the trainer wrote.

Why the scoring path is imported, not re-implemented
----------------------------------------------------
``_sample_chunked``, ``load_split``, ``SCORE_SEED`` and the split filenames come
from ``sweep_hyperparams`` even though the first is private.  A second copy of
the scoring path is the failure mode worth avoiding: if selection scored
differently from the sweep, every number in the write-up would silently stop
being comparable.  One scoring path, one seed, one chunking rule.

Divergence is NOT laundered
---------------------------
The default seed list is the honest ``0 1 2 3 4``.  If a seed produced no finite
discrepancy the script ABORTS naming that seed; it does not quietly substitute a
replacement seed and it does not renumber.  A stability failure is a result.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
    PYTHONPATH="$R/src:$R/experiments" taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python select_checkpoint_multiasset.py --seeds 0 1 2 3 4
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
from sweep_hyperparams import (  # noqa: E402
    SCORE_SEED,
    VAL_FILE,
    VALDISC_FILE,
    _sample_chunked,
    load_split,
)

SELECTION_DIR = HERE / "selection"


def _run_config(seed: int, run_dir: Path) -> dict:
    """The knobs this seed was actually trained with.

    Prefers the published ``weights/seed_N_config.json`` and falls back to the
    run directory's own copy, because ``--no-artifacts`` runs write only the
    latter.  Failing loudly here is deliberate: rebuilding the kernel with
    guessed hyperparameters would produce a model that scores fine and is not
    the model that was trained.
    """

    candidates = (
        METHOD_ROOT / "weights" / f"seed_{int(seed)}_config.json",
        run_dir / "config.json",
    )
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise SystemExit(
        f"ABORT: seed {seed} has no config.json; looked in "
        + ", ".join(str(p) for p in candidates)
    )


def _rebuild(seed: int, config: dict, *, device: torch.device):
    """``(model, grid, num_steps)`` reconstructed from the recorded config."""

    # The grid width must match training exactly, so read it off the train split
    # rather than trusting seq_len, then cross-check the two against each other.
    bank_prices = tm.load_prices()
    num_steps = int(bank_prices.shape[1]) - 1
    recorded = int(config.get("seq_len", num_steps + 1)) - 1
    if recorded != num_steps:
        raise SystemExit(
            f"ABORT: seed {seed} config records seq_len-1 = {recorded} but the "
            f"train split has {num_steps} steps"
        )
    grid = DiscreteTimeGrid(T=num_steps * tm.DT, num_steps=num_steps)

    model, _kernel = tm.build_model(
        grid=grid,
        device=device,
        seed=int(seed),
        bank_prices=bank_prices,
        h=float(config["reference_h"]),
        markov_order=int(config["reference_markov_order"]),
        npi=int(config["reference_npi"]),
        weight_grad_mode=str(config["reference_weight_grad_mode"]),
        # Subscript, never ``.get``.  A config predating the full-lag Jacobian
        # carries no ``reference_jacobian_lags``, and defaulting it would
        # silently rebuild the model with the one-lag backward pass -- the exact
        # truncation the analytic gradient exists to remove.  Crash instead.
        jacobian_lags=int(config["reference_jacobian_lags"]),
        allow_drift_correction=bool(config["allow_drift_correction"]),
    )
    return model, grid, num_steps


def score_model(
    model,
    *,
    grid: DiscreteTimeGrid,
    device: torch.device,
    num_steps: int,
    score_paths: int,
    score_chunk: int,
) -> tuple[float, float]:
    """``(val, valdisc)`` discrepancies of one generated bank against both splits.

    The bank is sampled ONCE and scored twice, so the two numbers share their
    Monte-Carlo noise and their difference is a property of the splits rather
    than of the sampler.
    """

    generated = _sample_chunked(
        model, num_paths=int(score_paths), chunk=int(score_chunk), device=device
    )
    if not torch.isfinite(generated).all():
        del generated
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return float("nan"), float("nan")

    def _score(filename: str) -> float:
        bank = load_split(
            filename,
            num_paths=int(score_paths),
            num_steps=int(num_steps),
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

    value = _score(VAL_FILE)
    secondary = _score(VALDISC_FILE)
    del generated
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return value, secondary


def select_one(seed: int, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    run_dir = Path(args.run_root) / f"seed_{int(seed)}"
    ckpt_dir = run_dir / "training_checkpoints"
    # "latest.pt" is the rolling crash-survival file, NOT a selection candidate;
    # globbing "step_*.pt" is what keeps it out.
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    if not checkpoints:
        raise SystemExit(f"ABORT: no step_*.pt checkpoints in {ckpt_dir}")

    config = _run_config(int(seed), run_dir)
    model, grid, num_steps = _rebuild(int(seed), config, device=device)

    per_step: list[dict[str, float]] = []
    for path in checkpoints:
        step = int(path.stem.split("_")[1])
        # weights_only=True: checkpoint_state() emits only tensors, plain dicts
        # and primitives, so there is no reason to allow arbitrary unpickling.
        model.load_checkpoint_state(
            torch.load(path, map_location=device, weights_only=True)
        )
        started = time.perf_counter()
        value, secondary = score_model(
            model,
            grid=grid,
            device=device,
            num_steps=num_steps,
            score_paths=int(args.score_paths),
            score_chunk=int(args.score_chunk),
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
        raise SystemExit(
            f"ABORT: seed {seed} produced no finite discrepancy at any of the "
            f"{len(per_step)} checkpoints. This is a divergence, not a missing "
            "file: report it, do not substitute another seed."
        )
    if len(finite) < len(per_step):
        print(
            f"  seed={seed} WARNING: {len(per_step) - len(finite)} of "
            f"{len(per_step)} checkpoints were non-finite and are excluded",
            flush=True,
        )
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
        published = json.loads(config_path.read_text(encoding="utf-8"))
        published["selected_step"] = selected_step
        published["selection_file"] = VAL_FILE
        published["selection_val_discrepancy"] = float(best["val_discrepancy"])
        config_path.write_text(
            json.dumps(published, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
        "score_chunk": int(args.score_chunk),
        "score_seed": SCORE_SEED,
        "reference_h": float(config["reference_h"]),
        "reference_markov_order": int(config["reference_markov_order"]),
        "reference_npi": int(config["reference_npi"]),
        "reference_weight_grad_mode": str(config["reference_weight_grad_mode"]),
        "date": date.today().isoformat(),
    }
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    out = SELECTION_DIR / f"seed_{int(seed)}_selection.json"
    out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"  [out] {out}", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="cuda:0")
    # 8192 matches the split size, so selection is not scored on a subsample the
    # way the sweep arms were (the sweep used 4096 for speed).
    parser.add_argument("--score-paths", type=int, default=8192)
    parser.add_argument("--score-chunk", type=int, default=1024)
    parser.add_argument("--run-root", type=Path, default=HERE / "runs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    records = [select_one(int(seed), args) for seed in args.seeds]
    print("\n seed  selected_step   val_discrepancy   valdisc_discrepancy")
    print("-" * 60)
    for rec in records:
        print(
            f"{rec['seed']:>5}  {rec['selected_step']:>13}   "
            f"{rec['val_discrepancy']:.8g}   {rec['valdisc_discrepancy']:.8g}"
        )
    steps = sorted({rec["selected_step"] for rec in records})
    print(f"\nselected steps across seeds: {steps}")
    last_step = max(int(rec["step"]) for rec in records[0]["per_step"])
    if steps == [last_step]:
        print(
            "NOTE: every seed selected the LAST checkpoint. Validation never "
            "turned over, so the 3000-step budget may be short rather than the "
            "selection being informative."
        )


if __name__ == "__main__":
    main()
