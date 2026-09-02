"""Diagnostic probe: is ``allow_drift_correction`` a no-op, and if so, why?

Context.  Three sweep arms were run with ``--allow-drift-correction 1`` and
returned val_discrepancy identical to ``dc = 0`` in all 17 significant digits.
A direct unit probe of
``MultivariateReferenceDriftSpecificEntropyMatrixControl`` showed the class DOES
honour the flag (the drift block moved by 2.71 on random adjoints), and
``train_multiasset.build_model`` does pass ``control_map=control``.  That leaves
exactly one candidate: the correction term itself,

    correction = -inputs.expected_adjoint_next

is identically zero during a real rollout, which would make the flag a genuine
mathematical no-op for this configuration rather than a plumbing bug.

This script wraps the model's ``control_map`` in a transparent spy, runs one
short rollout at the promoted incumbent, and reports the largest ``|p|`` seen.
It writes nothing and trains nothing.
"""

from __future__ import annotations

import json

import numpy as np
import torch

import train_multiasset as tm
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("sweep/incumbent.json", encoding="utf-8") as handle:
        incumbent = json.load(handle)

    # A short, narrow slice: the question is whether p is zero, which does not
    # depend on the horizon or on the bank being the full 8192 paths.
    bank = tm.load_prices()[:256, :52, :]
    grid = DiscreteTimeGrid(T=51 * tm.DT, num_steps=51)

    model, _kernel = tm.build_model(
        grid=grid,
        device=device,
        seed=0,
        bank_prices=bank,
        h=float(incumbent["h"]),
        markov_order=int(incumbent["markov_order"]),
        npi=int(incumbent["npi"]),
        weight_grad_mode=str(incumbent["weight_grad_mode"]),
        # Subscript, never ``.get``.  An incumbent predating the full-lag
        # Jacobian carries no ``jacobian_lags``, and defaulting it would
        # silently probe the one-lag backward pass -- a different model from
        # the promoted one, which would make the answer below a lie.  Crash
        # instead.
        jacobian_lags=int(incumbent["jacobian_lags"]),
        allow_drift_correction=True,
    )

    seen: list[tuple[float, float]] = []

    class Spy:
        """Forwards everything, records ``|p|`` on the way through."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __call__(self, inputs):
            p = inputs.expected_adjoint_next
            seen.append((float(p.abs().max()), float(p.norm())))
            return self._inner(inputs)

    model.control_map = Spy(model.control_map)

    x0 = torch.full(
        (64, tm.STATE_DIM), float(np.log(100.0)), device=device, dtype=tm.DTYPE
    )
    noise = torch.randn(64, 51, tm.STATE_DIM, device=device, dtype=tm.DTYPE)

    method = None
    for candidate in ("simulate", "rollout", "sample_paths", "sample"):
        if hasattr(model, candidate):
            method = candidate
            break
    if method is None:
        raise SystemExit(f"no rollout entry point found on {type(model).__name__}")
    print(f"[probe] rollout entry point = {method}", flush=True)

    with torch.no_grad():
        getattr(model, method)(x0=x0, noise=noise)

    if not seen:
        raise SystemExit("control_map was never called -- the spy saw nothing")

    max_abs = max(item[0] for item in seen)
    print(f"[probe] control calls          = {len(seen)}")
    print(f"[probe] max |p| over all       = {max_abs!r}")
    print(f"[probe] first three (max,norm) = {seen[:3]}")
    print(f"[probe] p identically zero     = {max_abs == 0.0}")


if __name__ == "__main__":
    main()
