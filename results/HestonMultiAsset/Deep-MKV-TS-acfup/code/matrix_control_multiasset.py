"""Matrix specific-entropy control wired to the multivariate reference kernel.

``SpecificEntropyMatrixControl`` already implements the projection the paper
specifies -- clip the *eigenvalues* of ``Theta = eta sigma_ref^{-1} + Z/sqrt(dt)``
rather than its diagonal entries -- and that part is reused verbatim.  Two
inherited pieces are unusable with a matrix-valued reference and are overridden
here.  Neither the package nor the frozen ``d = 1`` reproduction is touched.

1. ``__post_init__`` (``specific_entropy.py:72-82``) whitelists four univariate
   kernel classes and rejects anything else.  All four hard-code ``(1, d)``
   coefficient shapes, i.e. exactly the diagonal assumption the matrix control
   exists to remove, so the whitelist cannot simply be extended -- it has to be
   bypassed.  We re-implement the *validation* it also performs (dt, eta,
   sigma bounds, grid agreement) so nothing is silently skipped, and add the
   checks the matrix case needs on top.

2. ``_reference_sigma`` / ``_reference_sigmas_all``
   (``specific_entropy.py:93-131``) clamp with ``.clamp_min(sigma_min)`` and
   ``.clamp_max(sigma_max)``.  Those are ELEMENTWISE.  On a ``(B, d)`` diagonal
   volatility that is the intended projection; on a ``(B, d, d)`` matrix it is
   silently wrong, because every negative off-diagonal entry -- the entire
   representation of negative cross-asset covariance -- is forced up to
   ``+sigma_min``.  The correct projection acts on the spectrum, and the
   reference kernel has already applied it (``spectral_sigma``), so the honest
   override is a passthrough guarded by a construction-time agreement check
   rather than a second clip.

Why passthrough and not a re-clip: a spectral clip here would mean an ``eigh``
on every date of every path, ``8192 x 251 = 2.06e+06`` matrices per call, purely
to redo work the kernel already did.  Instead we require the control's
``[sigma_min, sigma_max]`` to equal the kernel's, which makes the clip provably
idempotent, and offer ``verify_spectrum`` to assert that once in the smoke test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from deep_mkv_gen_path_dt.controls.specific_entropy_matrix import (
    SpecificEntropyMatrixControl,
)
from deep_mkv_gen_path_dt.controls import (
    DynamicsStepInputs,
    HamiltonianControlInputs,
    RunningCostInputs,
    RunningCostResult,
)
from deep_mkv_gen_path_dt.controls.reference_drift import _ReferenceDriftMixin

_TOLERANCE = 1e-12


def _require_finite_float(name: str, value: object) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not torch.isfinite(torch.tensor(number)).item():
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class MultivariateSpecificEntropyMatrixControl(SpecificEntropyMatrixControl):
    """Matrix control accepting a ``MultivariateCovarianceReferenceKernel``.

    ``verify_spectrum`` turns on an assertion that the reference volatility
    really does have its spectrum inside ``[sigma_min, sigma_max]``.  It costs
    an ``eigh`` per date, so it is off by default and meant for smoke tests.
    """

    verify_spectrum: bool = False
    max_eigh_batch: int = 32768

    def __post_init__(self) -> None:
        # Deliberately NOT calling super().__post_init__(): it would run the
        # univariate-kernel whitelist.  Everything else it validates is
        # re-implemented below so no check is lost.
        dt = _require_finite_float("dt", self.dt)
        eta = _require_finite_float("eta", self.eta)
        sigma_min = _require_finite_float("sigma_min", self.sigma_min)
        denominator_eps = _require_finite_float(
            "denominator_eps", self.denominator_eps
        )
        if self.sigma_max is None:
            raise ValueError(
                "the matrix specific-entropy control requires sigma_max: without an "
                "upper bound the projection of an eigenvalue at or below zero is "
                "unbounded"
            )
        sigma_max = _require_finite_float("sigma_max", self.sigma_max)
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        if eta <= 0.0:
            raise ValueError("eta must be > 0")
        if sigma_min <= 0.0:
            raise ValueError("sigma_min must be > 0")
        if sigma_max <= sigma_min:
            raise ValueError("sigma_max must be > sigma_min when provided")
        if denominator_eps <= 0.0:
            raise ValueError("denominator_eps must be > 0")

        kernel = self.reference_kernel
        for attribute in ("grid", "state_dim", "evaluate", "evaluate_sigmas_all"):
            if not hasattr(kernel, attribute):
                raise ValueError(
                    f"reference_kernel must expose {attribute!r} to drive the "
                    "matrix control"
                )
        if int(kernel.grid.num_steps) <= 0:
            raise ValueError("reference_kernel must be fitted on a non-empty grid")
        if abs(float(kernel.grid.dt) - dt) > 1e-12:
            raise ValueError("dt must match reference_kernel.grid.dt")
        if int(kernel.state_dim) < 1:
            raise ValueError("reference_kernel.state_dim must be >= 1")

        # Makes the inherited elementwise clip provably idempotent, which is the
        # whole justification for overriding it with a passthrough.
        for name, expected in (("sigma_min", sigma_min), ("sigma_max", sigma_max)):
            actual = getattr(kernel, name, None)
            if actual is None:
                raise ValueError(f"reference_kernel must expose {name!r}")
            if abs(float(actual) - float(expected)) > _TOLERANCE:
                raise ValueError(
                    f"control {name}={expected!r} must equal reference_kernel."
                    f"{name}={float(actual)!r}: the matrix control passes the "
                    "reference volatility through unclipped and relies on the "
                    "kernel having already projected its spectrum"
                )

        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "sigma_min", sigma_min)
        object.__setattr__(self, "sigma_max", sigma_max)
        object.__setattr__(self, "denominator_eps", denominator_eps)
        object.__setattr__(self, "verify_spectrum", bool(self.verify_spectrum))

        max_eigh_batch = int(self.max_eigh_batch)
        if max_eigh_batch < 1:
            raise ValueError("max_eigh_batch must be >= 1")
        object.__setattr__(self, "max_eigh_batch", max_eigh_batch)

    # ------------------------------------------------------------------ #
    # flat (B, d*d) noise adjoint -> (B, d, d)
    # ------------------------------------------------------------------ #
    def theta_from_moments(
        self,
        *,
        r: torch.Tensor,
        sigma_ref: torch.Tensor,
        state_dim: int,
    ) -> torch.Tensor:
        """Unflatten ``Z`` before handing it to the base assembly of ``Theta``.

        In Algorithm 1 the GRU head emits ``Z`` as a ``d x d`` matrix -- it sits
        inside ``Theta = eta sigma_ref^{-1} + Z / sqrt(dt)``, which is a matrix
        equation.  So the architecture is configured with
        ``noise_adjoint_dim = d * d`` and the network hands back a flat
        ``(B, d*d)`` row.

        The base ``theta_from_moments`` (``specific_entropy_matrix.py:169-197``)
        pushes that straight into ``_as_matrix``, which accepts only
        ``(..., d, d)`` or ``(..., d)`` -- the latter meaning "diagonal, promote
        via ``diag_embed``".  A flat ``(B, d*d)`` row is neither, so it raises.
        At ``d = 1`` the ambiguity is invisible because ``d*d == d == 1``; from
        ``d = 2`` on the two readings differ and the base class rightly refuses
        to guess.  Unflattening here is that disambiguation, and it is the only
        insertion point needed: both controls in this module reach
        ``theta_from_moments`` through ``_control_from_reference_sigma``.

        ``(..., d, d)`` and ``(..., d)`` are passed through untouched, so the
        base contract still holds for anyone building ``R`` directly.
        """

        state_dim = int(state_dim)
        if r.ndim == 2 and int(r.shape[-1]) == state_dim * state_dim != state_dim:
            r = r.reshape(-1, state_dim, state_dim)
        return super().theta_from_moments(
            r=r, sigma_ref=sigma_ref, state_dim=state_dim
        )

    # ------------------------------------------------------------------ #
    # chunked evaluation: cuSOLVER cannot batch this many eigh calls
    # ------------------------------------------------------------------ #
    def running_cost(self, inputs: RunningCostInputs) -> RunningCostResult:
        """Batch-chunked wrapper around the base specific-entropy running cost.

        The base implementation (``specific_entropy_matrix.py:231-273``) is
        mathematically what we want, but it evaluates every date of every path
        in one shot: ``specific_entropy_matrix_cost`` calls
        ``torch.linalg.eigh`` on a ``(B, N, d, d)`` tensor, i.e. ``B * N``
        matrices at once.  Measured on an A100 with torch 2.13 / CUDA 13:

        * ``B * N = 64256`` -> works, 0.27 s, but **17.3 GiB** peak;
        * ``B * N = 65536`` -> ``CUSOLVER_STATUS_INTERNAL_ERROR``.

        cuSOLVER's batched ``syev`` reserves roughly 280 kB of workspace per
        ``8 x 8`` matrix, so the cost is in the workspace, not the data (the
        matrices themselves are 5 MB).  MAGMA is not an escape: torch 2.13
        deprecates it and silently redirects ``eigh`` back to cuSOLVER, so both
        backends fail at the same threshold.

        The frozen ``d = 1`` run used ``batch_size = 256`` and we match it, which
        puts us at ``256 * 251 = 64256`` -- inside the limit by 2%.  Running a
        five-seed, three-thousand-step campaign two percent from a hard crash is
        not a plan, so this override splits the work along the batch axis.

        ``max_eigh_batch = 32768`` is the measured knee.  Each chunk repeats the
        251-step Python recursion in ``evaluate_sigmas_all``, so smaller chunks
        trade memory for wall clock; on an A100 at ``B = 256``, ``N = 251``,
        float64:

        =============  ==========  ========
        max_eigh_batch  peak         time
        =============  ==========  ========
        unchunked       34.88 GiB    0.74 s
        49152           26.60 GiB    1.37 s
        32768           17.76 GiB    1.32 s
        16384            9.11 GiB    2.67 s
        =============  ==========  ========

        32768 gives two chunks, a 2x margin under the cliff, half the memory of
        unchunked and no measurable time penalty against 49152.  Every row above
        reproduces the unchunked value, gradient and metrics BIT-IDENTICALLY.

        Paths are independent given the parameters, so the split is exact rather
        than an approximation: the base value is a mean over the batch, and a
        size-weighted mean of per-chunk means is the same number.  Autograd is
        preserved because the weighted sum is built as a tensor expression --
        each chunk keeps its own graph and they are added, so ``d/dpaths`` still
        flows, which is what ``differentiable_sigma=True`` needs.
        """

        paths = inputs.paths
        if paths.ndim != 3:
            raise ValueError("running-cost paths must have shape (B, N + 1, d)")
        batch_size = int(paths.shape[0])
        num_steps = int(paths.shape[1]) - 1
        if num_steps < 1:
            raise ValueError("running-cost paths must contain at least one step")

        chunk = max(1, int(self.max_eigh_batch) // num_steps)
        if chunk >= batch_size:
            return super().running_cost(inputs)

        total: torch.Tensor | None = None
        metrics: dict[str, float] = {}
        for start in range(0, batch_size, chunk):
            stop = min(start + chunk, batch_size)
            weight = float(stop - start) / float(batch_size)
            piece = super().running_cost(
                RunningCostInputs(
                    paths=paths[start:stop],
                    controls=inputs.controls[start:stop],
                    grid=inputs.grid,
                    parameters=inputs.parameters,
                    differentiate_controls=bool(inputs.differentiate_controls),
                )
            )
            contribution = piece.value * weight
            total = contribution if total is None else total + contribution
            for key, value in piece.metrics.items():
                metrics[key] = metrics.get(key, 0.0) + float(value) * weight

        assert total is not None  # batch_size >= 1, so the loop ran
        return RunningCostResult(value=total, metrics=metrics)

    # ------------------------------------------------------------------ #
    # spectral, not elementwise
    # ------------------------------------------------------------------ #
    def _check_spectrum(self, sigma: torch.Tensor) -> torch.Tensor:
        if not self.verify_spectrum:
            return sigma
        with torch.no_grad():
            state_dim = int(self.reference_kernel.state_dim)
            spectrum = torch.linalg.eigvalsh(
                sigma.reshape(-1, state_dim, state_dim).detach()
            )
            low = float(spectrum.min())
            high = float(spectrum.max())
        if low < float(self.sigma_min) - 1e-9 or high > float(self.sigma_max) + 1e-9:
            raise ValueError(
                f"reference sigma spectrum [{low:.6e}, {high:.6e}] escapes "
                f"[{self.sigma_min:.6e}, {self.sigma_max:.6e}]"
            )
        return sigma

    def _reference_sigma(
        self, x_prefix: torch.Tensor, *, step_index: int
    ) -> torch.Tensor:
        """Reference volatility at one date, shape ``(B, d, d)``.

        Overrides the base method's elementwise clamp, which would force every
        negative off-diagonal to ``+sigma_min``.
        """

        _, sigma_ref = self.reference_kernel.evaluate(
            x_prefix, step_index=int(step_index)
        )
        return self._check_spectrum(sigma_ref)

    def _reference_sigmas_all(self, paths: torch.Tensor) -> torch.Tensor:
        """Reference volatility at every date, shape ``(B, N, d, d)``.

        Same override, and it also pins the dispatch: the base method probes
        ``evaluate_sigmas_all``, then ``evaluate_all``, then falls back to a
        per-step Python loop.  The fallback would be ``O(T^2)`` here, so we
        require the vectorised entry point instead of letting it degrade
        silently.
        """

        return self._check_spectrum(self.reference_kernel.evaluate_sigmas_all(paths))


@dataclass(frozen=True)
class MultivariateReferenceDriftSpecificEntropyMatrixControl(
    _ReferenceDriftMixin,
    MultivariateSpecificEntropyMatrixControl,
):
    """Matrix control in drift-correction coordinates around the fitted drift.

    This is the matrix analogue of the control the frozen ``d = 1`` run actually
    used.  Its manifest records ``physical_drift = fitted_reference_fixed`` and
    ``drift_correction_admissible = False``, i.e. the physical drift is pinned to
    the reference drift ``b_ref`` and only the volatility is controlled -- so the
    default here is ``allow_drift_correction = False``, matching that run rather
    than the package default of ``True``.

    Only ``__call__`` needs overriding.  The diagonal version
    (``reference_drift.py:76-93``) re-applies the elementwise
    ``clamp_min``/``clamp_max`` to ``sigma_ref`` inline, which would again crush
    the negative off-diagonals; everything else it does is packing-agnostic,
    because it slices the volatility block as ``[..., state_dim:]`` and never
    assumes that block has width ``d``.

    The inherited ``running_cost`` is correct as-is: the specific-entropy
    running cost is a function of ``sigma`` and ``sigma_ref`` only, and with
    ``allow_drift_correction = False`` the drift correction is identically zero,
    so it contributes nothing to the relative entropy.
    """

    allow_drift_correction: bool = False

    def __call__(self, inputs: HamiltonianControlInputs) -> torch.Tensor:
        state_dim = int(inputs.expected_adjoint_next.shape[-1])
        reference_drift, sigma_ref = self.reference_kernel.evaluate(
            inputs.x_prefix, step_index=int(inputs.step_index)
        )
        sigma_ref = self._check_spectrum(sigma_ref)
        correction_control = self._control_from_reference_sigma(
            p=inputs.expected_adjoint_next,
            r=inputs.expected_adjoint_noise_next,
            sigma_ref=sigma_ref,
        )
        correction = correction_control[..., :state_dim]
        physical_drift = reference_drift + (
            correction
            if bool(self.allow_drift_correction)
            else torch.zeros_like(correction)
        )
        return torch.cat(
            (physical_drift, correction_control[..., state_dim:]), dim=-1
        )


@dataclass(frozen=True)
class MatrixDriftVolatilityEulerStep:
    """Euler step for controls packed as ``[alpha, sigma.flatten(-2)]``.

    The package's ``DriftVolatilityEulerStep`` requires shape ``(B, 2 d)`` and
    multiplies the noise ELEMENTWISE, which encodes the diagonal assumption in
    the dynamics themselves: it can only ever produce independent per-asset
    shocks.  The matrix control emits ``(B, d + d*d)``, packed by
    ``specific_entropy_matrix.py:218`` as
    ``torch.cat([alpha, sigma.flatten(start_dim=-2)], dim=-1)``, so the noise
    must be applied as a matrix-vector product:

        x_{k+1} = x_k + alpha dt + sqrt(dt) * sigma @ dW.

    That product is what actually correlates the assets; with ``sigma`` diagonal
    it reduces to the package's elementwise form exactly.
    """

    dt: float

    def __post_init__(self) -> None:
        if float(self.dt) <= 0.0:
            raise ValueError("dt must be > 0")

    def __call__(self, inputs: DynamicsStepInputs) -> torch.Tensor:
        x_t = inputs.x_prefix[:, -1, :]
        state_dim = int(x_t.shape[-1])
        if int(inputs.noise_next.shape[-1]) != state_dim:
            raise ValueError(
                "MatrixDriftVolatilityEulerStep expects noise_dim == state_dim"
            )
        expected = state_dim + state_dim * state_dim
        if inputs.control.ndim != 2 or int(inputs.control.shape[-1]) != expected:
            raise ValueError(
                "control must have shape (B, state_dim + state_dim ** 2), got "
                f"{tuple(inputs.control.shape)} for state_dim={state_dim}"
            )
        alpha = inputs.control[:, :state_dim]
        sigma = inputs.control[:, state_dim:].reshape(-1, state_dim, state_dim)
        diffusion = torch.einsum("bij,bj->bi", sigma, inputs.noise_next)
        return x_t + alpha * float(self.dt) + math.sqrt(float(self.dt)) * diffusion
