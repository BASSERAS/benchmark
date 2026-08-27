"""Promote the three differentiated eigendecompositions of the matrix
specific-entropy control to float64.  The frozen package is not edited.

Why this file exists
====================
On TrueDataset, ``train_true.py`` died at optimizer step 3 with an all-NaN
``Theta`` (16384/16384 entries), surfacing as a cuSOLVER convergence failure
inside ``project_theta_to_sigma``.  ``torch.autograd.set_detect_anomaly(True)``
named the actual producer::

    Function 'LinalgEighBackward0' returned nan values in its 0th output.
      model.py:2448  fit -> _path_functional_adjoint_targets
      model.py:1466       -> _running_cost_result_and_path_gradient
      model.py:1158       -> running_cost(inputs)
      matrix_control_multiasset.py:244 -> super().running_cost(inputs)
      specific_entropy_matrix.py:259   -> specific_entropy_matrix_cost(...)
      specific_entropy_matrix.py:133   -> torch.linalg.eigh(_symmetrise(sigma_ref))

The eigendecomposition's BACKWARD carries ``1 / (lambda_i - lambda_j)`` terms.
The forward is perfectly healthy -- it is only the gradient that divides by an
eigenvalue gap.

The gap is a float32 artefact, and that is the whole point
=========================================================
Measured on the frozen kernel (sha256 1826647cc692c74c, sigma_max = 5.0) over
130048 real ``sigma_ref`` matrices, 8x8, from the training split:

    float64 spectrum:  gap <= 1e-9   in       0 / 910336 pairs  (0.00%)
                       gap <= 1e-6   in    6407 / 910336 pairs  (0.94%)
    exact ties at the sigma_max ceiling:  2 / 130048 matrices   (0.00%)

So in exact arithmetic the eigenvalues are DISTINCT and the gradient is finite.
But ``DTYPE = torch.float32`` has eps = 1.19e-07, and 0.891% of these matrices
have a relative min-gap at or below that.  Rounded to float32 the two
eigenvalues become bit-identical, ``lambda_i - lambda_j`` evaluates to exactly
0.0, and the backward returns inf/NaN.  Measured directly, one autograd.grad
call per site on 32512 matrices::

    SITE1 eigh f32: nan-matrices  270/32512   max|grad| 3.698e+00
    SITE1 eigh f64: nan-matrices    0/32512   max|grad| 3.698e+00
    SITE2 eigh f32: nan-matrices  270/32512   max|grad| 1.510e+01
    SITE2 eigh f64: nan-matrices    0/32512   max|grad| 1.510e+01
    SITE3 eigh f32: nan-matrices  269/32512   max|grad| 6.525e+01
    SITE3 eigh f64: nan-matrices    0/32512   max|grad| 6.525e+01

``max|grad|`` is IDENTICAL in the two precisions.  The true gradient is bounded
and O(1..60); there is no genuine blow-up to regularise away.  Promoting the
decomposition to float64 does not damp, clip, jitter or otherwise alter the
objective -- it just stops float32 from inventing a tie that is not there.  The
cost value itself agrees with a float64 Cholesky reference to 1.5e-14.

Consequently this is NOT a hyperparameter change and NOT a modelling change.
Nothing in the reported configuration moves.

Why it killed the run at step 3 rather than step 1
==================================================
A rollout starts every path at ``x0 = log(100)``, so the first rollouts visit a
narrow region of state space where the degenerate matrices are rare.  As the
control disperses the paths they reach the 0.8% degenerate set, one NaN enters
the summed running cost, every gradient becomes NaN, ``clip_grad_norm_`` cannot
repair it (it rescales finite gradients, it does not filter), Adam writes NaN
into all 56136 weights, and the next rollout emits an entirely NaN ``Theta``.
The observed growth curve matches exactly::

    rollout 0: max|Theta| 6.66     non-finite 0
    rollout 1: max|Theta| 803.78   non-finite 0
    rollout 2: max|Theta| 68.48    non-finite 0
    rollout 3: max|Theta| nan      non-finite 34816

The three sites
===============
All three differentiate an ``eigh`` whose input inherits ``sigma_ref``'s
spectrum, so all three are affected and all three are patched:

1. ``specific_entropy_matrix_cost``  -- ``eigh(sigma_ref)`` for ``sigma_ref^-1/2``
   plus ``eigvalsh(similar)``; reached from the running cost.
2. ``SpecificEntropyMatrixControl.theta_from_moments`` -- ``eigh(sigma_ref)``
   for ``eta * sigma_ref^-1``; reached from the control map every date.
3. ``project_theta_to_sigma`` -- ``eigh(Theta)`` for the spectral clip.  At
   initialisation ``Z = 0`` exactly, so ``Theta = eta * sigma_ref^-1`` and it
   inherits the degeneracy wholesale (269/32512 NaN).  Once ``Z`` is nonzero the
   ``1/sqrt(dt) = 1025.28`` term breaks the ties and it measures clean, but it
   is patched anyway: "clean once training starts" is not a guarantee.

Sites 1 and 2 also admit an exactly equivalent Cholesky form with no eigengap
in its backward, which was measured to be equally clean.  Float64 is preferred
because it is the single mechanism that also covers site 3 -- a spectral clamp
has no Cholesky equivalent -- so the whole control runs under one rule instead
of two, and because it leaves the frozen expressions themselves untouched.

Cost
====
Only the 8x8 batched decompositions move to float64; the network, the rollout
and the discrepancy stay in float32.  Everything is cast straight back, so no
float64 tensor escapes into the model.

Interaction with ``eigh_fallback``
==================================
``eigh_fallback.install()`` wraps the same ``project_theta_to_sigma`` with a
diagnostic CPU retry.  Install THIS module second so float64 is the outer
layer: the fallback then still guards a genuine cuSOLVER failure, and it now
sees float64 input, where cuSOLVER is in any case better behaved.  Both are
idempotent, so drivers may call both unconditionally.

Usage
-----
    import eigh_fallback, eigh_float64
    eigh_fallback.install()
    eigh_float64.install()           # AFTER the fallback, before the model
    ...
    cfg["eigh_float64"] = eigh_float64.counters()
"""

from __future__ import annotations

import torch

__all__ = ["install", "counters", "installed"]

_INSTALLED = False

# float64 is the target, so anything strictly narrower gets promoted. float64
# and complex inputs are passed through untouched.
_NARROW = (torch.float32, torch.float16, torch.bfloat16)

_COUNTERS = {
    "n_promoted_project_theta_to_sigma": 0,
    "n_promoted_specific_entropy_cost": 0,
    "n_promoted_theta_from_moments": 0,
}


def counters() -> dict[str, int]:
    """How many calls each site promoted, for the run's config record."""
    return dict(_COUNTERS)


def installed() -> bool:
    return _INSTALLED


def install(*, verbose: bool = True) -> None:
    """Rebind the three sites with float64-promoting wrappers.

    Idempotent: a second call is a no-op, so drivers may call it
    unconditionally.  Rebinds module globals and one class attribute; no file in
    the frozen package is modified.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    from deep_mkv_gen_path_dt.controls import specific_entropy_matrix as sem

    # ---- site 3: eigh(Theta) for the spectral clip -----------------------
    original_project = sem.project_theta_to_sigma

    def project_theta_to_sigma(theta, *, eta, sigma_min, sigma_max):
        if theta.dtype not in _NARROW:
            return original_project(
                theta, eta=eta, sigma_min=sigma_min, sigma_max=sigma_max
            )
        _COUNTERS["n_promoted_project_theta_to_sigma"] += 1
        sigma, clipped = original_project(
            theta.double(), eta=eta, sigma_min=sigma_min, sigma_max=sigma_max
        )
        # Autograd follows both casts, so the graph is preserved and the
        # backward runs in float64 where the eigenvalue gaps are resolvable.
        return sigma.to(theta.dtype), clipped.to(theta.dtype)

    project_theta_to_sigma.__doc__ = original_project.__doc__
    sem.project_theta_to_sigma = project_theta_to_sigma

    # ---- site 1: eigh(sigma_ref) + eigvalsh(similar) in the running cost --
    original_cost = sem.specific_entropy_matrix_cost

    def specific_entropy_matrix_cost(sigma, sigma_ref, *, eta):
        if sigma.dtype not in _NARROW and sigma_ref.dtype not in _NARROW:
            return original_cost(sigma, sigma_ref, eta=eta)
        _COUNTERS["n_promoted_specific_entropy_cost"] += 1
        value = original_cost(sigma.double(), sigma_ref.double(), eta=eta)
        return value.to(sigma.dtype)

    specific_entropy_matrix_cost.__doc__ = original_cost.__doc__
    sem.specific_entropy_matrix_cost = specific_entropy_matrix_cost

    # ---- site 2: eigh(sigma_ref) for eta * sigma_ref^-1 -------------------
    # Patched on the BASE class. matrix_control_multiasset overrides
    # theta_from_moments only to unflatten r from (B, d*d) to (B, d, d) and then
    # calls super(), which resolves here at call time -- so the subclass is
    # covered without being touched.
    control_cls = sem.SpecificEntropyMatrixControl
    original_theta = control_cls.theta_from_moments

    def theta_from_moments(self, *, r, sigma_ref, state_dim):
        if r.dtype not in _NARROW and sigma_ref.dtype not in _NARROW:
            return original_theta(
                self, r=r, sigma_ref=sigma_ref, state_dim=state_dim
            )
        _COUNTERS["n_promoted_theta_from_moments"] += 1
        theta = original_theta(
            self, r=r.double(), sigma_ref=sigma_ref.double(), state_dim=state_dim
        )
        return theta.to(r.dtype)

    theta_from_moments.__doc__ = original_theta.__doc__
    control_cls.theta_from_moments = theta_from_moments

    sem._float64_eigh_installed = True
    _INSTALLED = True
    if verbose:
        print(
            "[eigh_float64] installed: eigh promoted to float64 at 3 sites "
            "(project_theta_to_sigma, specific_entropy_matrix_cost, "
            "theta_from_moments); frozen package unmodified",
            flush=True,
        )
