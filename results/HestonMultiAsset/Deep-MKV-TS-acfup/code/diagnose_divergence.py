"""Why do seeds 1 and 3 die with a non-finite Theta despite the sigma clip?

This script does NOT modify any source file. It monkeypatches one method at
runtime and then calls the ordinary trainer, so the frozen reference package and
``train_multiasset.py`` are byte-identical before and after.

The question it answers
-----------------------
``Theta = eta * sigma_ref^{-1} + Z / sqrt(dt)`` is assembled in
``theta_from_moments``, and the spectral clip that ``sigma_min``/``sigma_max``
implement lives in ``project_theta_to_sigma``, which runs AFTER an ``eigh`` of
that Theta.  So the clip can only bound the OUTPUT sigma given a finite Theta --
it is structurally incapable of preventing a non-finite Theta.  The crash the
seeds hit is that eigh, one call before the clip ever executes.

That leaves exactly two candidate sources for the non-finiteness, and they are
separable because they are the two additive terms:

    sigma_ref^{-1}   -- the reference kernel's volatility, inverted
    Z / sqrt(dt)     -- the GRU head output, amplified by 1/sqrt(dt) = 15.87

This wrapper records both, per call, and trips on the FIRST call where anything
is non-finite, printing which term went first plus a short run-up history.

It also measures the quantity the cuSOLVER message actually complains about --
"too many repeated eigenvalues" -- by counting how many eigenvalues of sigma_ref
are pinned to a clip boundary.  Clipping is a many-to-one map: every eigenvalue
above sigma_max becomes exactly sigma_max.  Repeated eigenvalues make eigh's
backward, which carries 1/(mu_i - mu_j), singular.  ``multivariate_reference.py``
guards its own eigh against exactly this with a custom degeneracy-safe backward
(``_ClipSqrtEigh``, lines 203-262); the frozen ``project_theta_to_sigma`` does
not.

Seed 3 dies at step 2, so this reproduces in seconds.

Usage:
    cd results/HestonMultiAsset/Deep-MKV-TS/code
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 \
    PYTHONPATH="$R/src:$R/experiments" taskset -c 15-16 \
    /home/tbasseras/gpu-venv/bin/python diagnose_divergence.py \
        --seed 3 --steps 2 --device cuda:0 --run-root /tmp/mkv_diag
"""
from __future__ import annotations

import sys
from collections import deque

import torch

HISTORY = deque(maxlen=25)
STATE = {"calls": 0, "tripped": False}


def _spec(m: torch.Tensor) -> dict:
    """Eigen-summary of a batch of symmetric matrices, NaN-safe."""
    out = {"finite": bool(torch.isfinite(m).all())}
    with torch.no_grad():
        if not out["finite"]:
            out["n_bad"] = int((~torch.isfinite(m)).sum())
            out["n_tot"] = int(m.numel())
            return out
        sym = 0.5 * (m + m.transpose(-1, -2))
        ev = torch.linalg.eigvalsh(sym.double())
        out["ev_min"] = float(ev.min())
        out["ev_max"] = float(ev.max())
        gaps = (ev[..., 1:] - ev[..., :-1]).abs()
        out["min_gap"] = float(gaps.min())
        out["ev"] = ev
    return out


def _boundary_counts(ev: torch.Tensor, lo: float, hi: float) -> tuple[int, int, int]:
    """How many eigenvalues sit ON a clip boundary (i.e. were clipped)."""
    tol = 1e-9
    at_lo = int((ev <= lo * (1 + tol)).sum())
    at_hi = int((ev >= hi * (1 - tol)).sum())
    return at_lo, at_hi, int(ev.numel())


def install() -> None:
    import matrix_control_multiasset as mcm

    base = mcm.MultivariateSpecificEntropyMatrixControl
    original = base.theta_from_moments

    def patched(self, *, r, sigma_ref, state_dim):
        STATE["calls"] += 1
        idx = STATE["calls"]
        sig_min = float(self.sigma_min)
        sig_max = float(self.sigma_max)

        with torch.no_grad():
            r_finite = bool(torch.isfinite(r).all())
            r_absmax = float(r.abs().max()) if r_finite else float("inf")
            s_finite = bool(torch.isfinite(sigma_ref).all())

        rec = {"call": idx, "r_finite": r_finite, "r_absmax": r_absmax,
               "sigma_ref_finite": s_finite}

        if s_finite and sigma_ref.ndim >= 2 and sigma_ref.shape[-1] == sigma_ref.shape[-2]:
            sp = _spec(sigma_ref)
            rec["sref_ev"] = (sp.get("ev_min"), sp.get("ev_max"))
            rec["sref_min_gap"] = sp.get("min_gap")
            if "ev" in sp:
                lo, hi, tot = _boundary_counts(sp["ev"], sig_min, sig_max)
                rec["clipped_at_min"] = lo
                rec["clipped_at_max"] = hi
                rec["n_eigenvalues"] = tot

        theta = original(self, r=r, sigma_ref=sigma_ref, state_dim=state_dim)

        with torch.no_grad():
            t_finite = bool(torch.isfinite(theta).all())
        rec["theta_finite"] = t_finite
        HISTORY.append(rec)

        if not (t_finite and r_finite and s_finite) and not STATE["tripped"]:
            STATE["tripped"] = True
            _report(rec, sig_min, sig_max)
        return theta

    base.theta_from_moments = patched
    print("[diag] patched theta_from_moments", flush=True)


def _fmt(rec: dict) -> str:
    def g(k, f="{:.4g}"):
        v = rec.get(k)
        return "n/a" if v is None else (f.format(v) if isinstance(v, float) else str(v))
    return (f"call={rec['call']:>6} theta_finite={str(rec['theta_finite']):>5} "
            f"r_finite={str(rec['r_finite']):>5} |Z|max={g('r_absmax')} "
            f"sref_finite={str(rec['sigma_ref_finite']):>5} "
            f"sref_ev={rec.get('sref_ev')} min_gap={g('sref_min_gap')} "
            f"clipped[min/max]={rec.get('clipped_at_min')}/{rec.get('clipped_at_max')}"
            f" of {rec.get('n_eigenvalues')}")


def _report(rec: dict, sig_min: float, sig_max: float) -> None:
    print("\n" + "=" * 78, file=sys.stderr)
    print("[diag] FIRST NON-FINITE DETECTED", file=sys.stderr)
    print(f"[diag] sigma_min={sig_min:g} sigma_max={sig_max:g}", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    print("[diag] run-up (oldest first):", file=sys.stderr)
    for h in HISTORY:
        print("   " + _fmt(h), file=sys.stderr)
    print("\n[diag] VERDICT INPUTS:", file=sys.stderr)
    print(f"   sigma_ref finite : {rec['sigma_ref_finite']}", file=sys.stderr)
    print(f"   Z (r)     finite : {rec['r_finite']}   |Z|max={rec['r_absmax']:.6g}",
          file=sys.stderr)
    print(f"   Theta     finite : {rec['theta_finite']}", file=sys.stderr)
    print("   -> if sigma_ref is finite and Z is NOT, the clip is irrelevant: the",
          file=sys.stderr)
    print("      network output exploded and enters Theta additively, before eigh.",
          file=sys.stderr)
    print("=" * 78 + "\n", file=sys.stderr, flush=True)


def main() -> None:
    install()
    import train_multiasset as tm
    argv = list(sys.argv[1:])
    try:
        tm.main(argv)
    except BaseException as exc:  # noqa: BLE001 - we want the post-mortem either way
        print(f"\n[diag] trainer raised: {type(exc).__name__}: {exc}", file=sys.stderr)
        if not STATE["tripped"]:
            print("[diag] NOTE: died WITHOUT theta_from_moments ever seeing a "
                  "non-finite input -- the corruption is downstream of Theta "
                  "assembly.", file=sys.stderr)
            print("[diag] last calls (oldest first):", file=sys.stderr)
            for h in HISTORY:
                print("   " + _fmt(h), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
