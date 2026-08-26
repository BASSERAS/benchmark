#!/usr/bin/env python3
"""Score the ZERO-CONTROL model: what the discrepancy is when Algorithm 1 does nothing.

Why this file exists
--------------------
The ridge sweep selected ``ridge_lambda = 1000`` on ``val_discrepancy``, but that
candidate's ``ce_projection_r2`` is ``-0.3758`` -- worse than predicting the mean
of the target.  Across the surviving grid both diagnostics fall monotonically as
lambda rises:

    lambda      1e-3      10       100      1000
    ce_r2       0.928     0.783    0.456   -0.376
    final_loss  19.36     11.03    3.94     1.19

That is the signature of the ridge coefficient collapsing, ``beta -> 0``.  If
``beta -> 0`` then ``Z^proxy -> 0``; the adjoint loss ``|Zhat^theta - Z^proxy|^2``
is then minimised by ``Zhat -> 0``; and with ``Zhat = 0`` the control

    Theta = eta * (sigma^ref)^{-1} + Zhat / sqrt(dt)

reduces to ``eta * (sigma^ref)^{-1}``, i.e. ``sigma = sigma^ref``.  The trained
model would be the reference SDE with extra steps.

`sweep_ridge_lambda.py`'s docstring anticipated the OPPOSITE failure -- an r2 near
1 meaning the underdetermined ridge interpolates and denoises nothing -- and it
pinned the committed d = 1 run at ``ce_projection_r2 ~ 0.28`` as the healthy
reference.  It had no guard for r2 going negative.  A selection criterion that
cannot distinguish "learned something good" from "learned nothing" is not a
selection criterion, so this script measures the second option directly.

What it measures
----------------
Algorithm 1 zero-initialises the final GRU layer, so a model that has never been
fitted emits ``Zhat == 0`` EXACTLY.  Building the model and sampling it without
calling ``fit`` therefore draws from the pure reference SDE.  Scoring that bank
gives the NULL FLOOR: the discrepancy obtainable with zero learning.

Read the result as follows.

  * null floor MUCH WORSE than 0.023836 -> lambda = 1000 did learn something.
    The negative r2 says the linear proxy is a poor explainer of the target, not
    that the target vanished, and the sweep result stands.
  * null floor AT OR BELOW 0.023836 -> lambda = 1000 "wins" by switching the
    method off, the sweep ranked a null model first, and the selection criterion
    must be reconsidered BEFORE a 16 h campaign is spent reproducing it.

Everything -- seed, path count, score seed, both split files, the discrepancy
object -- is imported from the sweep so the number is comparable to the sweep
table by construction rather than by care.  The TEST split is never opened.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
    PYTHONPATH="$R/src:$R/experiments" taskset -c 0-3 \
    /home/tbasseras/gpu-venv/bin/python null_control_baseline.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
from sweep_ridge_lambda import (  # noqa: E402
    S0,
    SCORE_SEED,
    SWEEP_DIR,
    SWEEP_SEED,
    VAL_FILE,
    VALDISC_FILE,
    load_split,
)

eigh_fallback.install()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-paths", type=int, default=8192)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Derive the horizon from the split itself rather than hard-coding 252, so a
    # regenerated bank cannot silently desynchronise this from the sweep.
    header = np.load(DATASET / VAL_FILE, mmap_mode="r")
    num_steps = int(header.shape[1]) - 1
    if int(header.shape[2]) != STATE_DIM:
        raise ValueError(f"expected d = {STATE_DIM}, got {header.shape}")
    del header

    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)
    model = tm.build_model(
        grid=grid, device=device, seed=SWEEP_SEED, kernel_steps=num_steps
    )
    # NO fit() call. This is the whole point of the script.

    x0 = torch.full((1, STATE_DIM), math.log(S0), device=device, dtype=model.dtype)
    with torch.no_grad():
        generated = model.sample(
            num_paths=int(args.score_paths), x0=x0, seed=SCORE_SEED
        ).paths
    if not torch.isfinite(generated).all():
        raise RuntimeError("zero-control bank contains non-finite values")

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
        f"[null] steps=0 (Zhat == 0, pure reference SDE)\n"
        f"[null] {VAL_FILE} discrepancy={val_discrepancy:.8g}\n"
        f"[null] {VALDISC_FILE} discrepancy={valdisc_discrepancy:.8g}",
        flush=True,
    )

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    out = SWEEP_DIR / "null_control.json"
    out.write_text(
        json.dumps(
            {
                "steps": 0,
                "seed": SWEEP_SEED,
                "val_discrepancy": val_discrepancy,
                "valdisc_discrepancy": valdisc_discrepancy,
                "score_paths": int(args.score_paths),
                "score_seed": SCORE_SEED,
                "selection_file": VAL_FILE,
                "secondary_file": VALDISC_FILE,
                "note": (
                    "Zero-control floor. The model was BUILT but never FITTED; "
                    "Algorithm 1 zero-initialises the final GRU layer, so Zhat == 0 "
                    "exactly and the sampler draws from the reference SDE. Compare "
                    "against sweep/lambda_*.json: any candidate that does not beat "
                    "this number has not learned anything."
                ),
                "date": date.today().isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
