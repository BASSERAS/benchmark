#!/usr/bin/env python3
"""Score the ZERO-CONTROL model: the discrepancy when Algorithm 1 does nothing.

Why this file exists
--------------------
The sibling variant has ``sweep/null_control.json``; this arm never measured its
own.  That gap mattered the moment the two arms were compared: the SBTS-reference
campaign plateaus at objective ~14 while the sibling reaches ~2.4 on the *same*
discrepancy functional (identical preset, LAMBDA_SCALE, KAPPA_SCALE and ACF
weights), and its control_rms sits ~8x higher.  Two readings fit that evidence:

  * the reference itself is a poor starting point and the control cannot drag it
    far enough, or
  * the reference is fine and the fit is failing.

Only a zero-control measurement separates them, because it scores the reference
SDE alone with no learning at all.

What it measures
----------------
``networks/adjoint.py`` (and ``causal_adjoint.py``) call ``nn.init.zeros_`` on the
final layer's weight AND bias, so a model that has never been fitted emits
``Zhat == 0`` exactly.  The control then reduces to

    Theta = eta * (sigma^ref)^{-1} + Zhat / sqrt(dt)  ->  eta * (sigma^ref)^{-1}

i.e. ``sigma = sigma^ref``.  Building the model and sampling WITHOUT calling
``fit`` therefore draws from the pure reference SDE.  That is the null floor.

Read the result against the sibling's ``sweep/null_control.json`` (0.30009511 on
the same VAL_FILE).  The test was written to decide between the two readings
above: a null floor much worse than the sibling's would have meant the SBTS
reference -- Markovian b^ref plus a CONSTANT per-asset sigma^ref -- is a worse
starting point and the final gap is inherited rather than produced.

MEASURED: 0.22584774, i.e. BETTER than the sibling's 0.30009511.  That reading is
falsified.  The SBTS reference is the stronger of the two starting points on this
functional, so whatever costs this arm its final quality happens during fitting,
not before it.  The two runs used different ``score_seed`` values (20260831 here,
20260826 there) but the same 8192 paths and the same VAL_FILE, so the gap is far
wider than the sampling noise.

Do NOT compare either number to the ``objective`` column in ``losses/*.csv``.
Those are different quantities on different scales; only null-vs-null is a
like-for-like comparison.

Hyperparameters come from ``sweep/incumbent.json`` by SUBSCRIPT, never ``.get``.
An incumbent predating a field would otherwise be silently defaulted and this
script would measure a different model than the promoted one, making the number
a lie.  Crash instead.

The TEST split is never opened.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
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
from sweep_hyperparams import (  # noqa: E402
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

    with open(SWEEP_DIR / "incumbent.json", encoding="utf-8") as handle:
        incumbent = json.load(handle)

    # Derive the horizon from the split itself rather than hard-coding 252, so a
    # regenerated bank cannot silently desynchronise this from the sweep.
    header = np.load(tm.DATASET / VAL_FILE, mmap_mode="r")
    num_steps = int(header.shape[1]) - 1
    if int(header.shape[2]) != tm.STATE_DIM:
        raise ValueError(f"expected d = {tm.STATE_DIM}, got {header.shape}")
    del header

    grid = DiscreteTimeGrid(T=num_steps * tm.DT, num_steps=num_steps)

    # The kernel is fitted on the TRAIN bank, exactly as the campaign does; the
    # reference is data-dependent even when the control is switched off.
    bank_prices = tm.load_prices()

    model, _kernel = tm.build_model(
        grid=grid,
        device=device,
        seed=SWEEP_SEED,
        bank_prices=bank_prices,
        h=float(incumbent["h"]),
        markov_order=int(incumbent["markov_order"]),
        npi=int(incumbent["npi"]),
        weight_grad_mode=str(incumbent["weight_grad_mode"]),
        jacobian_lags=int(incumbent["jacobian_lags"]),
        allow_drift_correction=bool(incumbent["allow_drift_correction"]),
    )
    # NO fit() call. This is the whole point of the script.

    # Verify the premise rather than inherit it from the sibling's docstring: if
    # any adjoint output parameter were non-zero the sampler would NOT be drawing
    # from the pure reference SDE and the number below would not be a null floor.
    # ``DiscreteMPModel`` is not an ``nn.Module``; it OWNS the adjoint moment
    # network as ``model.network`` (train_multiasset.py:670 counts parameters
    # through exactly that attribute, and :773 checkpoints its state_dict).
    #
    # Do NOT filter on the substring "final": ``_zero_init_final_linear``
    # (causal_adjoint.py:254) zeroes ``module[-1]`` of an ``nn.Sequential``, so
    # the parameters are named ``<head>.<index>.weight`` and NOTHING matches
    # "final".  A name filter therefore inspects zero tensors and prints a
    # vacuous 0/0 pass -- a check that can never fail is worse than no check.
    # Enumerate the same objects the initialiser does instead: the last child of
    # every Sequential, when that child is Linear.
    heads = [
        (f"{name}[-1]", module[-1])
        for name, module in model.network.named_modules()
        if isinstance(module, torch.nn.Sequential)
        and len(module) > 0
        and isinstance(module[-1], torch.nn.Linear)
    ]
    if not heads:
        raise RuntimeError(
            "found no Sequential-with-Linear-tail output head on model.network; "
            "the zero-init premise cannot be verified, so this is not a null floor"
        )
    nonzero = [
        name
        for name, layer in heads
        if float(layer.weight.detach().abs().max()) != 0.0
        or (layer.bias is not None and float(layer.bias.detach().abs().max()) != 0.0)
    ]
    print(
        f"[null] zero-init check: {len(heads) - len(nonzero)}/{len(heads)} "
        f"output heads are all-zero; non-zero = {nonzero}",
        flush=True,
    )
    if nonzero:
        raise RuntimeError(
            f"output head(s) {nonzero} are not zero-initialised, so Zhat != 0 and "
            "the sampler is not drawing from the pure reference SDE"
        )

    x0 = torch.full((1, tm.STATE_DIM), math.log(S0), device=device, dtype=model.dtype)
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
        f"[null] steps=0 (Zhat == 0, pure SBTS reference SDE)\n"
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
                "incumbent": {
                    k: incumbent[k]
                    for k in (
                        "h",
                        "markov_order",
                        "npi",
                        "weight_grad_mode",
                        "jacobian_lags",
                        "allow_drift_correction",
                    )
                },
                "note": (
                    "Zero-control floor for the SBTS-reference arm. The model was "
                    "BUILT but never FITTED; the adjoint network's final layer is "
                    "zero-initialised, so Zhat == 0 exactly and the sampler draws "
                    "from the pure reference SDE (Markovian b^ref, CONSTANT "
                    "per-asset sigma^ref). Comparable to the sibling variant's "
                    "Deep-MKV-TS/code/sweep/null_control.json, which uses the same "
                    "discrepancy functional and the same VAL_FILE."
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
