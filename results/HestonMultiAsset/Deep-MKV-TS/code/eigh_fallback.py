"""Method-local robustness shim for the paper's spectral clip at d = 8.

Why this file exists
--------------------
On 2026-08-26 the ridge-lambda sweep at ``lambda = 1`` died 31 minutes in::

    torch._C._LinAlgError: linalg.eigh: (Batch element 0): The algorithm failed
    to converge because the input matrix is ill-conditioned or has too many
    repeated eigenvalues (error code: 1)

raised at ``specific_entropy_matrix.py:106``, inside

    sigma = U diag( eta / clip_[eta/smax, eta/smin](theta_i) ) U^T

which is Algorithm 1's spectral clip. This never happens at d = 1, where ``Theta``
is 1x1 and ``eigh`` is arithmetic. At d = 8 it is a real batched cuSOLVER call on
``B * T`` symmetric 8x8 matrices per outer iteration, and cuSOLVER's ``syevj``/
``syevd`` can fail to converge on a clustered spectrum.

The failure is fatal to the campaign, not just the sweep: the crash occurred inside
250 steps, and the 5-seed campaign is 3000 steps per seed -- roughly 60x more
decompositions.

What this shim does, and what it deliberately does NOT do
---------------------------------------------------------
It wraps ``project_theta_to_sigma`` so that a cuSOLVER convergence failure is
*diagnosed* and *retried on a different backend*, never papered over:

  1. On ``_LinAlgError``, report the state of ``Theta``: how many entries are
     non-finite, which batch elements are responsible, the finite entries' range,
     and the symmetry residual. This is printed, not swallowed -- a silent retry
     would hide an exploding control behind a "successful" run.
  2. **If ``Theta`` is not finite, raise immediately.** A NaN in ``Theta`` means the
     control blew up upstream; retrying on another backend would either fail again
     or, worse, return garbage. That is a bug to fix, not a backend to swap.
  3. If ``Theta`` is finite, retry the SAME decomposition on CPU LAPACK, first in
     the input dtype, then in float64. LAPACK's ``syevd`` is markedly more robust
     than the batched cuSOLVER path on clustered spectra.

**The mathematics is unchanged.** No jitter is added, no eigenvalue is clamped
beyond the clip the paper already specifies, and the returned ``(sigma, clipped)``
is the same object the GPU path would have produced had it converged. The only
difference is which LAPACK implementation computes the eigendecomposition. This is
why it is a robustness shim and not a fourth d = 8 deviation: adding ``eps * I`` to
break degenerate eigenvalues WOULD change the clipped spectrum, and was considered
and rejected for exactly that reason.

Gradients survive the detour: ``.cpu()`` and ``.to(device)`` are autograd
operations, so the graph stays continuous through the retry. The float64 promotion
changes precision on that one call only, and is counted separately from the
same-dtype retry so the two are never conflated in the audit record.

Zero edits to the frozen package
--------------------------------
``methods/Deep-MKV-TS/code/reference`` is the package the d = 1 reproduction matched
Table 1 with, and it is not forked. This shim rebinds the module attribute
``deep_mkv_gen_path_dt.controls.specific_entropy_matrix.project_theta_to_sigma`` at
runtime. That is sufficient because the sole caller
(``_control_from_reference_sigma``, line 211) lives in the same module and resolves
the name as a global at call time. ``controls/__init__.py`` re-exports the symbol
but nothing calls it from there.

Audit record
------------
``counters()`` returns the tallies the drivers write into
``weights/seed_*_config.json`` and ``sweep/lambda_*.json``. A run whose
``n_fallback_cpu`` is non-zero is still a valid run, but it is a run that hit a
numerically hard region, and the reader is entitled to know how often.

Usage:
    import eigh_fallback
    eigh_fallback.install()          # once, before the model is built
    ...
    cfg["eigh_fallback"] = eigh_fallback.counters()
"""
from __future__ import annotations

import sys

import torch

_COUNTERS = {
    "n_calls_failed": 0,          # cuSOLVER raised
    "n_fallback_cpu": 0,          # recovered on CPU in the input dtype
    "n_fallback_cpu_float64": 0,  # recovered on CPU only after promoting to float64
}

_INSTALLED = False


def counters() -> dict:
    """Copy of the retry tallies, for the config JSON."""
    return dict(_COUNTERS)


def reset() -> None:
    for key in _COUNTERS:
        _COUNTERS[key] = 0


def _describe(theta: torch.Tensor) -> str:
    """Human-readable state of Theta at the moment eigh gave up."""
    with torch.no_grad():
        flat = theta.reshape(-1, theta.shape[-2], theta.shape[-1])
        finite = torch.isfinite(flat)
        n_bad = int((~finite).sum())
        parts = [f"shape={tuple(theta.shape)}", f"dtype={theta.dtype}",
                 f"non_finite_entries={n_bad}"]
        if n_bad:
            bad = torch.nonzero(~finite.all(dim=-1).all(dim=-1)).flatten()
            parts.append(f"bad_batch_elements={bad[:8].tolist()}"
                         f"{'...' if bad.numel() > 8 else ''} of {flat.shape[0]}")
        good = flat[finite]
        if good.numel():
            parts.append(f"finite_range=[{float(good.min()):.6e}, "
                         f"{float(good.max()):.6e}]")
        # Symmetry is the precondition eigh relies on, so report the residual:
        # it distinguishes a broken symmetrisation from a genuinely hard spectrum.
        sym = (flat - flat.transpose(-1, -2)).abs()
        sym_finite = sym[torch.isfinite(sym)]
        if sym_finite.numel():
            parts.append(f"max_asymmetry={float(sym_finite.max()):.3e}")
    return ", ".join(parts)


def install(*, verbose: bool = True) -> None:
    """Rebind ``project_theta_to_sigma`` with the diagnostic CPU-retry wrapper.

    Idempotent: a second call is a no-op, so drivers may call it unconditionally.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    from deep_mkv_gen_path_dt.controls import specific_entropy_matrix as sem

    original = sem.project_theta_to_sigma

    def patched(theta, *, eta, sigma_min, sigma_max):
        try:
            return original(theta, eta=eta, sigma_min=sigma_min, sigma_max=sigma_max)
        except torch._C._LinAlgError as exc:
            _COUNTERS["n_calls_failed"] += 1
            desc = _describe(theta)
            print(f"[eigh_fallback] cuSOLVER eigh failed: {exc}\n"
                  f"[eigh_fallback] Theta: {desc}", file=sys.stderr, flush=True)

            if not bool(torch.isfinite(theta).all()):
                # An exploding control, not a backend problem. Fail loudly:
                # retrying elsewhere would conceal the actual defect.
                raise RuntimeError(
                    "eigh failed AND Theta is not finite -- this is an exploding "
                    "control, not a cuSOLVER convergence problem. Do not retry on "
                    f"another backend. Theta: {desc}"
                ) from exc

            # Same decomposition, LAPACK instead of cuSOLVER. Autograd follows the
            # device round-trip, so the graph is preserved.
            try:
                out = original(theta.cpu(), eta=eta,
                               sigma_min=sigma_min, sigma_max=sigma_max)
                _COUNTERS["n_fallback_cpu"] += 1
                if verbose:
                    print(f"[eigh_fallback] recovered on CPU LAPACK ({theta.dtype})",
                          file=sys.stderr, flush=True)
            except torch._C._LinAlgError:
                out = original(theta.cpu().to(torch.float64), eta=eta,
                               sigma_min=sigma_min, sigma_max=sigma_max)
                _COUNTERS["n_fallback_cpu_float64"] += 1
                if verbose:
                    print("[eigh_fallback] recovered on CPU LAPACK after promoting "
                          "to float64", file=sys.stderr, flush=True)
            sigma, clipped = out
            return (sigma.to(device=theta.device, dtype=theta.dtype),
                    clipped.to(device=theta.device, dtype=theta.dtype))

    patched.__name__ = original.__name__
    patched.__doc__ = original.__doc__
    sem.project_theta_to_sigma = patched
    sem._cpu_eigh_fallback_installed = True
    _INSTALLED = True
    if verbose:
        print("[eigh_fallback] installed: diagnostic CPU retry around "
              "project_theta_to_sigma (frozen package unmodified)",
              file=sys.stderr, flush=True)
