#!/usr/bin/env python
"""Apply — and, more importantly, *audit* — the declared S0 repair
``S <- 100 * S / S[:, :1]`` on a method's five generated banks.

Why this file exists
--------------------
PDF §1.4 and §7 checklist item 5 require every returned bank to start at
``S0 = 100``. LS4 generates in standardized price space with no ``t = 0`` anchor,
so its raw banks miss that by up to ~1.4e-2 relative. The repair is declared
under PDF §1.3 in both READMEs and recorded per seed in
``generation_manifest.json -> numerical_repair``.

It was originally applied ad hoc. That is the defect this file closes: a
transformation that stands between the generator and every scored number must be
**committed code**, not a shell line someone remembers typing. Otherwise the
committed pipeline does not reproduce the committed banks, and no reader can
check the claim "the path law is untouched".

What the repair does and does not do
------------------------------------
``S'_t = 100 * S_t / S_0``. Every ratio ``S_t / S_u`` is preserved exactly in
exact arithmetic, so every log return is preserved and every metric that
normalises by ``S[:, :1]`` -- which both PDF evaluators do -- is invariant. It is
a change of numeraire, not a change of path law.

It is **not** free in floating point: the division is one rounding per entry. The
``--check`` mode measures that drift rather than asserting it is zero.

Modes
-----
``--check`` (default, non-destructive)
    Re-measures the §1.4 contract on each bank and reports whether it is anchored.
    **This is a weak check and the script says so.** It cannot prove the repair
    was applied correctly, because it does not have the input.

``--check --raw-dir DIR`` (the strong check)
    With the pre-repair banks available, verifies ``repair(raw) == scored``
    bit-for-bit and reports the induced log-return drift. This is the only
    complete audit of the transformation, and it is the one a method porting in
    **now** should be able to produce. Keep your raw banks.

``--apply`` (destructive, writes in place)
    Refuses to run unless the bank is finite and strictly positive.

Status of the two LS4 banks — measured, and asymmetric
------------------------------------------------------
* **Experiment A: fully re-derived.** The raw pre-repair banks are in git at commit
  ``ba7c748``. Extract them and this script reproduces all five scored banks
  **bit-for-bit**::

      git show ba7c748:results/new_experiments/experiment_A/LS4/generated_paths/\\
          seed_0/generated_paths_8192x128.npy > raw/generated_paths/seed_0/...
      python apply_s0_repair.py --model-dir ../experiment_A/LS4 --raw-dir raw

* **Experiment B: NOT re-derivable.** B's banks were first committed (``f54d87f``)
  already repaired, so no pre-repair copy exists in history, and generation is not
  replayable from the checkpoint alone -- the torch RNG state at generation time
  depends on the whole training run. What is verifiable for B is only that
  ``S[:, 0] == 100.0`` exactly on all five banks and that the map is a per-path
  rescaling. That is weaker than A, and this file will not pretend otherwise.

**Idempotence is not evidence, so this script does not test it.** Once
``S[:, 0] == 100`` exactly, a divide-first repair multiplies by exactly ``1.0`` and
returns its input; a "fixed point" check on an already-anchored bank is vacuous.
The multiply-first forms are not even idempotent, differing by ~2.8e-14. Either
way the result says nothing about whether the repair was applied correctly. Only
``--raw-dir`` does.

Usage
-----
    python apply_s0_repair.py --model-dir ../experiment_A/LS4                 # weak
    python apply_s0_repair.py --model-dir ../experiment_A/LS4 --raw-dir /raw  # strong
    python apply_s0_repair.py --model-dir ../experiment_A/LS4 --apply         # repair

Exit codes: 0 all seeds OK, 1 at least one seed failed (all are reported).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

N_PATHS, N_STEPS = 8192, 128
S0 = 100.0
BANK = f"generated_paths/seed_{{s}}/generated_paths_{N_PATHS}x{N_STEPS}.npy"


def repair(a: np.ndarray) -> np.ndarray:
    """The declared transformation, verbatim, in one place so it cannot drift.

    ``S <- 100 * S / S[:, 0]`` -- exactly as written in every
    ``generation_manifest.json -> numerical_repair.operation``, and exactly as
    Python associates it (multiply first, then divide). **This form reproduces the
    committed Experiment A banks bit-for-bit from the raw banks in commit
    ``ba7c748``**; see the ``--raw-dir`` mode.

    **Operator order is load-bearing and was measured, not reasoned about.** The
    four algebraically identical forms are *not* identical in float64; they differ
    by up to 2.8e-14 absolute, so only one of them can re-derive a given bank:

        100.0 * S / S[:, :1]     <-- declared, used here, re-derives Experiment A
        100.0 * (S / S[:, :1])   -> differs by 2.8e-14
        (S / S[:, :1]) * 100.0   -> differs by 2.8e-14
        S * (100.0 / S[:, :1])   -> differs by 2.8e-14

    All four perturb log returns by at most 1.8e-15, so the choice does not touch
    the path law -- it only decides which bank is bit-reproducible.

    **Anchoring is data-dependent, not guaranteed.** The declared form landed on
    ``S[:, 0] == 100.0`` exactly for all ten LS4 banks, but it need not in general:
    on synthetic lognormal paths it leaves a residual of 1.4e-14, which fails PDF
    §1.4's *"starts at 100"* -- stated with no tolerance. The divide-first forms
    are exact by construction (``x / x`` is exactly ``1.0``, and ``100.0 * 1.0`` is
    exactly ``100.0``). So this function applies the declared form and the caller
    **verifies** the result; ``--anchor-exact`` switches to the divide-first form
    for a new method that trips the check. Never assume; the check is cheap.
    """
    return S0 * a / a[:, :1]


def repair_anchor_exact(a: np.ndarray) -> np.ndarray:
    """Divide-first variant: anchors at exactly ``S0`` for any finite positive input.

    Algebraically the same map; differs from :func:`repair` at the 1e-14 level.
    Use for a NEW method whose bank fails the §1.4 anchor under the declared form.
    """
    return S0 * (a / a[:, :1])


def log_return_drift(a: np.ndarray, b: np.ndarray) -> float:
    """Max absolute change in log returns induced by the map. Should be ~1e-16."""
    ra = np.diff(np.log(a), axis=1)
    rb = np.diff(np.log(b), axis=1)
    return float(np.abs(ra - rb).max())


def check_contract(a: np.ndarray) -> list[str]:
    """PDF §1.4, re-measured. Returns a list of violations (empty = conforming)."""
    bad: list[str] = []
    if a.shape != (N_PATHS, N_STEPS):
        bad.append(f"shape {a.shape} != ({N_PATHS}, {N_STEPS})")
    if a.dtype not in (np.float32, np.float64):
        bad.append(f"dtype {a.dtype} not float32/float64")
    if not np.isfinite(a).all():
        bad.append(f"{int((~np.isfinite(a)).sum())} non-finite entries")
        return bad  # everything below would be meaningless
    if not (a > 0).all():
        bad.append(f"non-positive: min={a.min()}")
    if not np.all(a[:, 0] == S0):
        bad.append(f"S0 != {S0}: max deviation {np.abs(a[:, 0] - S0).max():.6e}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-dir", required=True,
                    help="results/new_experiments/experiment_<X>/<Method>")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--raw-dir", default=None,
                    help="directory holding the PRE-repair banks, same seed_<q>/ layout; "
                         "enables the only complete audit of the transformation")
    ap.add_argument("--apply", action="store_true",
                    help="DESTRUCTIVE: rewrite each bank in place")
    ap.add_argument("--anchor-exact", action="store_true",
                    help="use the divide-first variant, exact by construction; only for a "
                         "NEW method whose bank fails the 1.4 anchor under the declared form")
    a = ap.parse_args()

    xform = repair_anchor_exact if a.anchor_exact else repair

    root = os.path.abspath(a.model_dir)
    raw_root = os.path.abspath(a.raw_dir) if a.raw_dir else None
    seeds = [int(s) for s in a.seeds.split(",")]
    fails: list[str] = []
    strong = 0

    for s in seeds:
        rel = BANK.format(s=s)
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            fails.append(f"seed {s}: MISSING {rel}")
            continue

        arr = np.load(path)
        if not np.isfinite(arr).all() or not (arr > 0).all():
            fails.append(f"seed {s}: bank is not finite-and-positive; refusing to touch it")
            continue

        if a.apply:
            dev = float(np.abs(arr[:, 0] - S0).max())
            if dev == 0.0:
                print(f"seed {s}: already anchored at {S0:g}, nothing written")
            else:
                fixed = xform(arr)
                drift = log_return_drift(arr, fixed)
                np.save(path, fixed)
                print(f"seed {s}: repaired (pre-repair max |S0-{S0:g}| = {dev:.6e}, "
                      f"log-return drift {drift:.3e}) -> {rel}")
            arr = np.load(path)

        # --- the strong check: repair(raw) must equal the scored bank, bit for bit
        elif raw_root is not None:
            rpath = os.path.join(raw_root, rel)
            if not os.path.isfile(rpath):
                fails.append(f"seed {s}: --raw-dir given but MISSING {rpath}")
            else:
                raw = np.load(rpath)
                got = xform(raw)
                if np.array_equal(got, arr):
                    strong += 1
                    print(f"seed {s}: STRONG OK  repair(raw) == scored bit-for-bit; "
                          f"log-return drift {log_return_drift(raw, got):.3e}")
                else:
                    fails.append(f"seed {s}: repair(raw) != scored, max |diff| "
                                 f"{float(np.abs(got - arr).max()):.6e}")

        # --- the weak check, and it announces its own weakness
        else:
            dev = float(np.abs(arr[:, 0] - S0).max())
            print(f"seed {s}: anchored (max |S0-{S0:g}| = {dev:.1e}) — WEAK: without "
                  f"--raw-dir this cannot verify the repair, only that a bank is anchored")

        for v in check_contract(arr):
            fails.append(f"seed {s}: PDF 1.4 violation: {v}")

    mode_label = "apply" if a.apply else ("check+raw" if raw_root else "check")
    label = f"{os.path.basename(root)} [{mode_label}]"
    if fails:
        print(f"\nFAIL  {label}: {len(fails)} problem(s)\n")
        for f in fails:
            print(f"  {f}")
        return 1
    if raw_root and not a.apply:
        form = (f"S <- {S0:g} * (S / S[:, :1])" if a.anchor_exact
                else f"S <- {S0:g} * S / S[:, :1]")
        print(f"\nPASS  {label}: {strong}/{len(seeds)} seeds re-derived from raw "
              f"via {form}; all satisfy PDF 1.4")
    else:
        print(f"\nPASS  {label}: {len(seeds)} seeds satisfy PDF 1.4 "
              f"(S0 == {S0:g} exactly). This does NOT verify the repair — "
              f"pass --raw-dir for that.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
