"""Multivariate covariance reference model for Deep-MKV-TS at ``d > 1``.

WHY THIS FILE EXISTS
--------------------
The shipped package (``methods/Deep-MKV-TS/code/reference``) contains four
reference-kernel fitters and every one of them is univariate in the sense that
matters here: ``GuyonLekeufackReferenceKernel.__post_init__`` demands
coefficients of shape ``(1, d)``, so its ``sigma^ref`` is a *vector* of
per-asset volatilities, i.e. a diagonal matrix.  A diagonal reference cannot
carry the ``d(d-1)/2`` spot correlations of ``dataset/HestonMultiAsset``
(off-diagonal ``sigma_s`` entries: min 0.2937, mean 0.4495, max 0.7024).

The prior smoke test ``paper_reimplementation/metric/
smoke_matrix_control_multivariate.py`` quantified what a diagonal reference
costs at exactly this correlation level, with an *oracle* Z handed to the
diagonal control (i.e. the strongest possible diagonal input):

    rho_market  rel. Frobenius error to optimal sigma   Hamiltonian gap
      0.0                    3.27e-17                     -2.9e-18
      0.6                    0.3438                        0.2741
      0.9                    0.6044                        1.982

So at the dataset's mean correlation the diagonal map is structurally wrong,
not merely suboptimal.  This module supplies the matrix-valued reference the
matrix control needs.

THE MODEL
---------
Two-scale causal EWMA memories, per asset ``j``:

    T^(a)_i = lam^T_a T^(a)_{i-1} + (1 - lam^T_a) dX_{i-1}
    V^(a)_i = lam^V_a V^(a)_{i-1} + (1 - lam^V_a) s_{i-1}
    lam^*_a = 2^{-1/h^*_a},   V^(a)_0 = v0 * 1_d,   T^(a)_0 = 0
    T_i = (1 - theta_T) T^(1)_i + theta_T T^(2)_i
    V_i = (1 - theta_V) V^(1)_i + theta_V V^(2)_i

``s_{i-1}`` is the activity innovation.  With ``activity_update =
"structural_variance"`` (the setting the frozen d = 1 Heston run used, and the
one selected here) it is the model's OWN instantaneous variance
``diag(sigma^ref_{i-1} sigma^ref_{i-1})``, not the realised squared return.  At
``d = 1`` that is ``clip(C)`` and coincides with the package's
``GuyonLekeufackReferenceKernel`` structural branch.

Symmetric pair features, ``phi_i^{jk} in R^7``:

    (1, sqrt((V_ij V_ik)_+), (V_ij + V_ik)/2, mean_k V_ik,
        T_ij T_ik, (T_ij + T_ik)/2, mean_k T_ik)

Covariance and volatility:

    (C_i)_{jk} = gamma_d . phi_i^{jj}          if j == k
                 gamma_o . phi_i^{jk}          if j != k
    sigma^ref_i = Pi_[smin,smax](C_i^{1/2})
                = U_i diag( clip( sqrt((mu_i)_+) ) ) U_i^T

The spectral clip is the same projection the matrix control uses, so the
reference is by construction a feasible point of
``S^d_[smin,smax] = {U diag(mu) U^T : mu in [smin,smax]^d}``.  It is symmetric
positive definite, and at ``d = 1`` it collapses to the scalar clip.

Drift features ``psi_i^j = (1, T_ij, mean_k T_ik, V_ij, mean_k V_ik) in R^5``
and ``(b_i)_j = gamma_b . psi_i^j``.

CALIBRATION
-----------
Exact Gaussian negative log-likelihood of the Euler increment, penalised:

    L(gamma) = (1/N) sum_i [ log det sigma_i + (1/(2 dt)) || sigma_i^{-1} e_i ||^2 ]
               + rho_c ||(gamma_d, gamma_o)||^2 + rho_b ||gamma_b||^2
    e_i = dX_i - b_i dt

minimised on 80% of the training paths; ``rho_c, rho_b`` selected on the
held-out 20%; then frozen.  This is a genuine MLE, NOT the ridge least squares
on realised products ``r_i r_j / dt`` used by the smoke-test prototype -- that
prototype was a feasibility probe, not a calibration protocol.  The ridge LS
survives here only as the MLE's initialiser.

The four half-lives and the two blend weights are fitted here as well (6 extra
degrees of freedom), on the d = 8 train split.  The article's protocol fixes
them; refitting was chosen deliberately because nothing guarantees that a
timescale calibrated at d = 1 is the right one for an 8-asset covariance.

DIFFERENTIABILITY (disclosed)
-----------------------------
``evaluate_sigmas_all`` returns ``sigma^ref`` **detached** from the path unless
``differentiable_sigma`` is set.  The reason is not convenience.  At the
dataset's near-equicorrelated structure the spectrum of ``C`` is one large
market eigenvalue plus ``d - 1`` tightly clustered idiosyncratic ones;
``eigh``'s backward pass carries ``1 / (mu_i - mu_j)`` factors and is therefore
ill-conditioned exactly here.  Consequence: the running cost's derivative with
respect to the path through the reference volatility (the ``d_x f`` branch) is
dropped.  The reference's DRIFT path-dependence is kept and is differentiated
exactly, via the ``autograd_replay`` drift-adjoint backend, which this kernel
supports in O(T) through the same one-step state cache the package's Guyon
kernel uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

__all__ = [
    "COVARIANCE_FEATURE_NAMES",
    "DRIFT_FEATURE_NAMES",
    "MultivariateCovarianceReferenceKernel",
    "MultivariateReferenceFitResult",
    "covariance_features",
    "covariance_from_features",
    "decay_from_half_life",
    "drift_features",
    "fit_multivariate_covariance_reference_kernel",
    "spectral_sigma",
]

COVARIANCE_FEATURE_NAMES: tuple[str, ...] = (
    "1",
    "sqrt(V_j V_k)",
    "(V_j+V_k)/2",
    "mean_k V_k",
    "T_j T_k",
    "(T_j+T_k)/2",
    "mean_k T_k",
)
DRIFT_FEATURE_NAMES: tuple[str, ...] = ("1", "T_j", "mean_k T_k", "V_j", "mean_k V_k")


def decay_from_half_life(half_life: float) -> float:
    """``lam = 2^{-1/h}``, the spec's discrete-step convention."""

    if not (float(half_life) > 0.0):
        raise ValueError("half_life must be > 0")
    return float(2.0 ** (-1.0 / float(half_life)))


def _symmetrise(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (value + value.transpose(-1, -2))


def covariance_features(trend: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
    """Return ``phi`` with shape ``(..., d, d, 7)``, symmetric in the last two axes.

    Every entry is a causal statistic of the observed past.  Nothing about the
    generator (Heston parameters, the true correlation matrix) is supplied.
    """

    if trend.shape != activity.shape:
        raise ValueError("trend and activity must have the same shape")
    v_j = activity.unsqueeze(-1)
    v_k = activity.unsqueeze(-2)
    t_j = trend.unsqueeze(-1)
    t_k = trend.unsqueeze(-2)
    v_bar = activity.mean(dim=-1, keepdim=True).unsqueeze(-1)
    t_bar = trend.mean(dim=-1, keepdim=True).unsqueeze(-1)
    ones = torch.ones_like(v_j * v_k)
    return torch.stack(
        [
            ones,
            torch.sqrt((v_j * v_k).clamp_min(0.0)),
            0.5 * (v_j + v_k),
            v_bar.expand_as(ones),
            t_j * t_k,
            0.5 * (t_j + t_k),
            t_bar.expand_as(ones),
        ],
        dim=-1,
    )


def drift_features(trend: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
    """Return ``psi`` with shape ``(..., d, 5)``."""

    v_bar = activity.mean(dim=-1, keepdim=True).expand_as(activity)
    t_bar = trend.mean(dim=-1, keepdim=True).expand_as(trend)
    return torch.stack([torch.ones_like(trend), trend, t_bar, activity, v_bar], dim=-1)


def covariance_from_features(
    features: torch.Tensor,
    gamma_diagonal: torch.Tensor,
    gamma_offdiagonal: torch.Tensor,
) -> torch.Tensor:
    """Assemble ``C`` from ``phi`` and the two coefficient families."""

    state_dim = int(features.shape[-2])
    diagonal_value = features @ gamma_diagonal
    offdiag_value = features @ gamma_offdiagonal
    mask = torch.eye(state_dim, dtype=torch.bool, device=features.device).expand_as(
        diagonal_value
    )
    return _symmetrise(torch.where(mask, diagonal_value, offdiag_value))


def _clip_sqrt(
    eigenvalues: torch.Tensor, *, sigma_min: float, sigma_max: float
) -> torch.Tensor:
    return eigenvalues.clamp_min(0.0).sqrt().clamp(float(sigma_min), float(sigma_max))


class _SpectralSigma(torch.autograd.Function):
    """``C -> U diag(clip(sqrt(mu_+))) U^T`` with a degeneracy-safe backward.

    Autograd's own ``eigh`` backward differentiates the eigenvectors and so
    carries a ``1 / (mu_i - mu_j)`` factor.  That factor is fatal here: at step
    ``0`` the memories are identical across assets, every pair feature collapses
    to the same value and ``C = a I + b (1 1^T - I)`` is exactly equicorrelated,
    giving one simple eigenvalue and one of multiplicity ``d - 1``.  The
    observed gaps are ``~3e-18``, so the factor overflows and every upstream
    gradient becomes NaN.

    The singularity is an artefact of the chain rule through ``U``, not of the
    map itself: a symmetric matrix function is differentiable at a repeated
    eigenvalue, with the Daleckii-Krein differential

        dF = U [ Lambda o (U^T dC U) ] U^T,
        Lambda_ij = (f(mu_i) - f(mu_j)) / (mu_i - mu_j)   -> f'(mu_i) as j -> i.

    Implementing that adjoint directly removes the singularity exactly rather
    than damping it, and never invokes ``eigh`` backward at all.

    ``eigenvectors`` and ``values`` are returned for convenience but are marked
    NON-DIFFERENTIABLE: any gradient a caller needs must be taken through
    ``sigma``.  Callers in this module comply (``diag(sigma^2)`` for the
    structural variance, a Cholesky of ``sigma`` in the likelihood).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, covariance: torch.Tensor, sigma_min: float, sigma_max: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        values = _clip_sqrt(eigenvalues, sigma_min=sigma_min, sigma_max=sigma_max)
        sigma = _symmetrise(
            eigenvectors @ torch.diag_embed(values) @ eigenvectors.transpose(-1, -2)
        )
        ctx.save_for_backward(eigenvalues, eigenvectors, values)
        ctx.sigma_min = float(sigma_min)
        ctx.sigma_max = float(sigma_max)
        ctx.mark_non_differentiable(eigenvectors, values)
        return sigma, eigenvectors, values

    @staticmethod
    def backward(ctx, grad_sigma, _grad_eigenvectors, _grad_values):  # type: ignore[override]
        eigenvalues, eigenvectors, values = ctx.saved_tensors
        if grad_sigma is None:
            return None, None, None
        low = ctx.sigma_min**2
        high = ctx.sigma_max**2

        gap = eigenvalues.unsqueeze(-1) - eigenvalues.unsqueeze(-2)
        difference = values.unsqueeze(-1) - values.unsqueeze(-2)
        # Below `tolerance` the divided difference loses too many digits to
        # cancellation; the derivative at the midpoint is its exact limit.
        scale = eigenvalues.abs().amax(dim=-1, keepdim=True).unsqueeze(-1) + 1.0
        tolerance = 1e-7 * scale
        midpoint = 0.5 * (eigenvalues.unsqueeze(-1) + eigenvalues.unsqueeze(-2))
        inside = (midpoint > low) & (midpoint < high)
        derivative = torch.where(
            inside,
            0.5 / midpoint.clamp_min(low).sqrt(),
            torch.zeros_like(midpoint),
        )
        loewner = torch.where(
            gap.abs() > tolerance,
            difference / torch.where(gap.abs() > tolerance, gap, torch.ones_like(gap)),
            derivative,
        )

        symmetric = _symmetrise(grad_sigma)
        transposed = eigenvectors.transpose(-1, -2)
        projected = transposed @ symmetric @ eigenvectors
        grad_covariance = _symmetrise(
            eigenvectors @ (loewner * projected) @ transposed
        )
        return grad_covariance, None, None


def spectral_sigma(
    covariance: torch.Tensor, *, sigma_min: float, sigma_max: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(sigma, eigenvectors, clipped_singular_values)``.

    ``sigma = U diag(clip(sqrt(mu_+))) U^T``.  Only ``sigma`` carries a
    gradient; see :class:`_SpectralSigma` for why, and for the exact adjoint
    used in its place.
    """

    if sigma_max <= sigma_min:
        raise ValueError("sigma_max must be > sigma_min")
    return _SpectralSigma.apply(
        _symmetrise(covariance), float(sigma_min), float(sigma_max)
    )


@dataclass(frozen=True)
class MultivariateCovarianceReferenceKernel:
    """Matrix-valued causal reference volatility and drift for ``d >= 1``.

    Deliberately NOT a subclass of any package kernel: the package's
    ``__post_init__`` validators enforce ``(1, d)`` coefficient shapes, which is
    the diagonal assumption this class exists to remove.  The public surface
    (``grid``, ``evaluate``, ``evaluate_all``, ``evaluate_sigmas_all``) is the
    one the controls and the model actually call.
    """

    grid: object
    state_dim: int
    trend_half_lives: tuple[float, float]
    activity_half_lives: tuple[float, float]
    trend_weight: float
    activity_weight: float
    gamma_diagonal: torch.Tensor
    gamma_offdiagonal: torch.Tensor
    gamma_drift: torch.Tensor
    initial_activity: float
    sigma_min: float = 1e-3
    sigma_max: float = 0.6
    activity_update: str = "structural_variance"
    ridge_covariance: float = 0.0
    ridge_drift: float = 0.0
    calibration_nll: float = float("nan")
    validation_nll: float = float("nan")
    differentiable_sigma: bool = False
    _cache_step: int = field(default=-1, init=False, repr=False, compare=False)
    _cache_trend: tuple[torch.Tensor, torch.Tensor] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _cache_activity: tuple[torch.Tensor, torch.Tensor] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if int(self.state_dim) < 1:
            raise ValueError("state_dim must be >= 1")
        if self.activity_update not in {"structural_variance", "realized_returns"}:
            raise ValueError(
                "activity_update must be 'structural_variance' or 'realized_returns'"
            )
        for name in ("trend_half_lives", "activity_half_lives"):
            values = tuple(float(v) for v in getattr(self, name))
            if len(values) != 2 or any(not math.isfinite(v) or v <= 0.0 for v in values):
                raise ValueError(f"{name} must be two positive finite half-lives")
            object.__setattr__(self, name, values)
        for name in ("trend_weight", "activity_weight"):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        if tuple(self.gamma_diagonal.shape) != (7,):
            raise ValueError("gamma_diagonal must have shape (7,)")
        if tuple(self.gamma_offdiagonal.shape) != (7,):
            raise ValueError("gamma_offdiagonal must have shape (7,)")
        if tuple(self.gamma_drift.shape) != (5,):
            raise ValueError("gamma_drift must have shape (5,)")
        if not (float(self.sigma_min) > 0.0):
            raise ValueError("sigma_min must be > 0")
        if not (float(self.sigma_max) > float(self.sigma_min)):
            raise ValueError("sigma_max must be > sigma_min")
        if not (float(self.initial_activity) > 0.0):
            raise ValueError("initial_activity must be > 0")
        object.__setattr__(self, "state_dim", int(self.state_dim))
        object.__setattr__(self, "sigma_min", float(self.sigma_min))
        object.__setattr__(self, "sigma_max", float(self.sigma_max))
        object.__setattr__(self, "initial_activity", float(self.initial_activity))

    # ------------------------------------------------------------------ #
    # recursion primitives
    # ------------------------------------------------------------------ #
    def _initial_states(
        self, *, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        shape = (int(batch_size), int(self.state_dim))
        trend = torch.zeros(shape, device=device, dtype=dtype)
        activity = torch.full(
            shape, float(self.initial_activity), device=device, dtype=dtype
        )
        return [trend, trend], [activity, activity]

    def _blend(self, states: list[torch.Tensor], weight: float) -> torch.Tensor:
        return (1.0 - float(weight)) * states[0] + float(weight) * states[1]

    def _coefficients(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.gamma_diagonal.to(device=device, dtype=dtype),
            self.gamma_offdiagonal.to(device=device, dtype=dtype),
            self.gamma_drift.to(device=device, dtype=dtype),
        )

    def _step_quantities(
        self,
        trend_states: list[torch.Tensor],
        activity_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(drift, sigma, structural_variance)`` at one date."""

        trend = self._blend(trend_states, self.trend_weight)
        activity = self._blend(activity_states, self.activity_weight)
        gamma_d, gamma_o, gamma_b = self._coefficients(
            device=trend.device, dtype=trend.dtype
        )
        covariance = covariance_from_features(
            covariance_features(trend, activity), gamma_d, gamma_o
        )
        sigma, _eigenvectors, _values = spectral_sigma(
            covariance, sigma_min=float(self.sigma_min), sigma_max=float(self.sigma_max)
        )
        drift = drift_features(trend, activity) @ gamma_b
        # diag(sigma sigma), the model's own instantaneous variance.  Taken from
        # sigma rather than from (U, values) because those are marked
        # non-differentiable; sigma is symmetric, so (sigma sigma)_jj = sum_k
        # sigma_jk^2 exactly.
        structural_variance = sigma.pow(2).sum(dim=-1)
        return drift, sigma, structural_variance

    def _advance(
        self,
        trend_states: list[torch.Tensor],
        activity_states: list[torch.Tensor],
        *,
        increment: torch.Tensor,
        structural_variance: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        activity_source = (
            structural_variance
            if self.activity_update == "structural_variance"
            else increment.pow(2) / float(self.grid.dt)
        )
        trend_next = [
            decay_from_half_life(h) * state + (1.0 - decay_from_half_life(h)) * increment
            for state, h in zip(trend_states, self.trend_half_lives)
        ]
        activity_next = [
            decay_from_half_life(h) * state
            + (1.0 - decay_from_half_life(h)) * activity_source
            for state, h in zip(activity_states, self.activity_half_lives)
        ]
        return trend_next, activity_next

    def _states_for_prefix(
        self, x_prefix: torch.Tensor, *, step_index: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Advance the cached memory once, with a full-replay fallback.

        The cache is what makes the ``autograd_replay`` drift adjoint O(T)
        rather than O(T^2): that backend walks ``step = 0, 1, 2, ...`` with
        growing prefixes, so every call after the first hits the fast branch and
        keeps the autograd graph alive through the one-step update.
        """

        batch_size = int(x_prefix.shape[0])
        device, dtype = x_prefix.device, x_prefix.dtype
        step = int(step_index)
        if step == 0:
            return self._initial_states(
                batch_size=batch_size, device=device, dtype=dtype
            )
        cached_trend = self._cache_trend
        cached_activity = self._cache_activity
        if (
            int(self._cache_step) == step - 1
            and cached_trend is not None
            and cached_activity is not None
            and tuple(cached_trend[0].shape) == (batch_size, int(self.state_dim))
            and cached_trend[0].device == device
            and cached_trend[0].dtype == dtype
        ):
            trend_states = list(cached_trend)
            activity_states = list(cached_activity)
            _, _, structural = self._step_quantities(trend_states, activity_states)
            return self._advance(
                trend_states,
                activity_states,
                increment=x_prefix[:, -1, :] - x_prefix[:, -2, :],
                structural_variance=structural,
            )
        trend_states, activity_states = self._initial_states(
            batch_size=batch_size, device=device, dtype=dtype
        )
        for increment in (x_prefix[:, 1:, :] - x_prefix[:, :-1, :]).unbind(dim=1):
            _, _, structural = self._step_quantities(trend_states, activity_states)
            trend_states, activity_states = self._advance(
                trend_states,
                activity_states,
                increment=increment,
                structural_variance=structural,
            )
        return trend_states, activity_states

    # ------------------------------------------------------------------ #
    # public surface used by the controls and by the model
    # ------------------------------------------------------------------ #
    def evaluate(
        self, x_prefix: torch.Tensor, *, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step = int(step_index)
        if x_prefix.ndim != 3:
            raise ValueError("x_prefix must have shape (B, n + 1, d)")
        if int(x_prefix.shape[1]) != step + 1:
            raise ValueError("x_prefix time dimension must equal step_index + 1")
        if int(x_prefix.shape[2]) != int(self.state_dim):
            raise ValueError("x_prefix state dimension must match state_dim")
        trend_states, activity_states = self._states_for_prefix(
            x_prefix, step_index=step
        )
        object.__setattr__(self, "_cache_step", step)
        object.__setattr__(self, "_cache_trend", tuple(trend_states))
        object.__setattr__(self, "_cache_activity", tuple(activity_states))
        drift, sigma, _ = self._step_quantities(trend_states, activity_states)
        return drift, sigma

    def _scan(
        self, paths: torch.Tensor, *, want_drift: bool, want_sigma: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        if int(paths.shape[1]) != int(self.grid.num_steps) + 1:
            raise ValueError("path time dimension must align with grid")
        trend_states, activity_states = self._initial_states(
            batch_size=int(paths.shape[0]), device=paths.device, dtype=paths.dtype
        )
        drifts: list[torch.Tensor] = []
        sigmas: list[torch.Tensor] = []
        for step in range(int(self.grid.num_steps)):
            drift, sigma, structural = self._step_quantities(
                trend_states, activity_states
            )
            if want_drift:
                drifts.append(drift)
            if want_sigma:
                sigmas.append(sigma)
            trend_states, activity_states = self._advance(
                trend_states,
                activity_states,
                increment=paths[:, step + 1, :] - paths[:, step, :],
                structural_variance=structural,
            )
        return (
            torch.stack(drifts, dim=1) if want_drift else None,
            torch.stack(sigmas, dim=1) if want_sigma else None,
        )

    def evaluate_all(self, paths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.differentiable_sigma:
            drifts, sigmas = self._scan(paths, want_drift=True, want_sigma=True)
        else:
            with torch.no_grad():
                drifts, sigmas = self._scan(
                    paths.detach(), want_drift=True, want_sigma=True
                )
        assert drifts is not None and sigmas is not None
        return drifts, sigmas

    def evaluate_sigmas_all(self, paths: torch.Tensor) -> torch.Tensor:
        """Reference volatility at every date, shape ``(B, N, d, d)``.

        Detached unless ``differentiable_sigma`` is set; see the module
        docstring for why (``eigh`` backward is ill-conditioned on this
        dataset's near-degenerate idiosyncratic spectrum).
        """

        if self.differentiable_sigma:
            _, sigmas = self._scan(paths, want_drift=False, want_sigma=True)
        else:
            with torch.no_grad():
                _, sigmas = self._scan(
                    paths.detach(), want_drift=False, want_sigma=True
                )
        assert sigmas is not None
        return sigmas

    def evaluate_drifts_all(self, paths: torch.Tensor) -> torch.Tensor:
        drifts, _ = self._scan(paths, want_drift=True, want_sigma=False)
        assert drifts is not None
        return drifts

    def parameter_summary(self) -> dict[str, float]:
        summary: dict[str, float] = {
            "state_dim": float(self.state_dim),
            "trend_half_life_fast_steps": float(self.trend_half_lives[0]),
            "trend_half_life_slow_steps": float(self.trend_half_lives[1]),
            "activity_half_life_fast_steps": float(self.activity_half_lives[0]),
            "activity_half_life_slow_steps": float(self.activity_half_lives[1]),
            "trend_weight": float(self.trend_weight),
            "activity_weight": float(self.activity_weight),
            "initial_activity": float(self.initial_activity),
            "sigma_min": float(self.sigma_min),
            "sigma_max": float(self.sigma_max),
            "ridge_covariance": float(self.ridge_covariance),
            "ridge_drift": float(self.ridge_drift),
            "calibration_nll": float(self.calibration_nll),
            "validation_nll": float(self.validation_nll),
        }
        for index, name in enumerate(COVARIANCE_FEATURE_NAMES):
            summary[f"gamma_diagonal[{name}]"] = float(self.gamma_diagonal[index])
            summary[f"gamma_offdiagonal[{name}]"] = float(self.gamma_offdiagonal[index])
        for index, name in enumerate(DRIFT_FEATURE_NAMES):
            summary[f"gamma_drift[{name}]"] = float(self.gamma_drift[index])
        return summary


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MultivariateReferenceFitResult:
    kernel: MultivariateCovarianceReferenceKernel
    history: list[dict[str, float]]
    calibration_nll: float
    validation_nll: float


def _softplus_inverse(value: float) -> float:
    return float(math.log(math.expm1(max(float(value), 1e-6))))


def _logit(value: float) -> float:
    clipped = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return float(math.log(clipped / (1.0 - clipped)))


class _Parameterisation(torch.nn.Module):
    """Unconstrained parameterisation of the reference model.

    Half-lives are ``1 + softplus(raw)`` so they stay above one step; the blend
    weights are ``sigmoid(raw)`` so they stay in ``[0, 1]``; the gammas are
    free.  In particular ``C`` is NOT forced to be positive definite: the
    spectral clip already handles an indefinite ``C``, and constraining it would
    silently change the model rather than the optimiser.

    The gammas are carried in STANDARDISED coordinates: the optimiser sees
    ``w``, the model uses ``gamma = w / scale`` where ``scale`` is the frozen
    root-mean-square of each feature.  This is a pure diagonal reparameterisation
    -- it changes nothing about which models are representable -- and it is
    forced on us by the data: measured on the ``d = 8`` train split the feature
    RMS runs from ``9.1e-07`` (``T_j T_k``) to ``1.0`` (the constant), a spread
    of ``1.1e+06``.  A single Adam step size cannot serve both ends of that
    range, and in practice the unstandardised fit oscillated over four orders of
    magnitude in the NLL.  ``scale`` is computed once, at the initial memory
    parameters, and then held fixed so the reparameterisation does not drift as
    the half-lives move.
    """

    def __init__(
        self,
        *,
        trend_half_lives: tuple[float, float],
        activity_half_lives: tuple[float, float],
        trend_weight: float,
        activity_weight: float,
        gamma_diagonal: torch.Tensor,
        gamma_offdiagonal: torch.Tensor,
        gamma_drift: torch.Tensor,
        covariance_scale: torch.Tensor,
        drift_scale: torch.Tensor,
        fit_memory: bool,
    ) -> None:
        super().__init__()
        self.register_buffer("covariance_scale", covariance_scale.clone())
        self.register_buffer("drift_scale", drift_scale.clone())
        self.raw_trend_half_lives = torch.nn.Parameter(
            torch.tensor([_softplus_inverse(h - 1.0) for h in trend_half_lives]),
            requires_grad=bool(fit_memory),
        )
        self.raw_activity_half_lives = torch.nn.Parameter(
            torch.tensor([_softplus_inverse(h - 1.0) for h in activity_half_lives]),
            requires_grad=bool(fit_memory),
        )
        self.raw_trend_weight = torch.nn.Parameter(
            torch.tensor(_logit(trend_weight)), requires_grad=bool(fit_memory)
        )
        self.raw_activity_weight = torch.nn.Parameter(
            torch.tensor(_logit(activity_weight)), requires_grad=bool(fit_memory)
        )
        self.raw_gamma_diagonal = torch.nn.Parameter(
            gamma_diagonal.clone() * covariance_scale
        )
        self.raw_gamma_offdiagonal = torch.nn.Parameter(
            gamma_offdiagonal.clone() * covariance_scale
        )
        self.raw_gamma_drift = torch.nn.Parameter(gamma_drift.clone() * drift_scale)

    @property
    def gamma_diagonal(self) -> torch.Tensor:
        return self.raw_gamma_diagonal / self.covariance_scale

    @property
    def gamma_offdiagonal(self) -> torch.Tensor:
        return self.raw_gamma_offdiagonal / self.covariance_scale

    @property
    def gamma_drift(self) -> torch.Tensor:
        return self.raw_gamma_drift / self.drift_scale

    def half_lives(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            1.0 + torch.nn.functional.softplus(self.raw_trend_half_lives),
            1.0 + torch.nn.functional.softplus(self.raw_activity_half_lives),
        )

    def weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.sigmoid(self.raw_trend_weight),
            torch.sigmoid(self.raw_activity_weight),
        )


def _negative_log_likelihood(
    parameterisation: _Parameterisation,
    log_paths: torch.Tensor,
    *,
    dt: float,
    initial_activity: float,
    sigma_min: float,
    sigma_max: float,
    activity_update: str,
) -> torch.Tensor:
    """Exact Gaussian NLL of the Euler increment under the reference model.

    ``log det sigma`` and ``sigma^{-1} e`` both come from a single Cholesky
    factor of ``sigma``, so no inverse or determinant is ever formed.  The
    Cholesky always succeeds: ``sigma``'s spectrum is clipped into
    ``[sigma_min, sigma_max]``, so it is positive definite with condition number
    at most ``sigma_max / sigma_min``.  Going through ``sigma`` rather than
    through the eigenpair is what keeps the gradient finite -- see
    :class:`_SpectralSigma`.
    """

    trend_half_lives, activity_half_lives = parameterisation.half_lives()
    trend_weight, activity_weight = parameterisation.weights()
    trend_decay = torch.pow(
        torch.tensor(2.0, device=log_paths.device, dtype=log_paths.dtype),
        -1.0 / trend_half_lives,
    )
    activity_decay = torch.pow(
        torch.tensor(2.0, device=log_paths.device, dtype=log_paths.dtype),
        -1.0 / activity_half_lives,
    )
    gamma_d = parameterisation.gamma_diagonal
    gamma_o = parameterisation.gamma_offdiagonal
    gamma_b = parameterisation.gamma_drift

    batch_size, point_count, state_dim = log_paths.shape
    device, dtype = log_paths.device, log_paths.dtype
    zero = torch.zeros((batch_size, state_dim), device=device, dtype=dtype)
    start = torch.full(
        (batch_size, state_dim), float(initial_activity), device=device, dtype=dtype
    )
    trend_states = [zero, zero]
    activity_states = [start, start]

    total = log_paths.new_zeros(())
    for step in range(point_count - 1):
        trend = (1.0 - trend_weight) * trend_states[0] + trend_weight * trend_states[1]
        activity = (1.0 - activity_weight) * activity_states[
            0
        ] + activity_weight * activity_states[1]
        covariance = covariance_from_features(
            covariance_features(trend, activity), gamma_d, gamma_o
        )
        sigma, _eigenvectors, _values = spectral_sigma(
            covariance, sigma_min=sigma_min, sigma_max=sigma_max
        )
        drift = drift_features(trend, activity) @ gamma_b
        increment = log_paths[:, step + 1, :] - log_paths[:, step, :]
        residual = increment - drift * float(dt)
        factor = torch.linalg.cholesky(sigma)
        solved = torch.cholesky_solve(residual.unsqueeze(-1), factor).squeeze(-1)
        log_determinant = 2.0 * torch.log(
            torch.diagonal(factor, dim1=-2, dim2=-1)
        ).sum(dim=-1)
        total = total + (
            log_determinant + 0.5 * solved.pow(2).sum(dim=-1) / float(dt)
        ).mean()
        structural = sigma.pow(2).sum(dim=-1)
        activity_source = (
            structural
            if activity_update == "structural_variance"
            else increment.pow(2) / float(dt)
        )
        trend_states = [
            trend_decay[index] * trend_states[index]
            + (1.0 - trend_decay[index]) * increment
            for index in range(2)
        ]
        activity_states = [
            activity_decay[index] * activity_states[index]
            + (1.0 - activity_decay[index]) * activity_source
            for index in range(2)
        ]
    return total / float(point_count - 1)


def _feature_scales(
    log_paths: torch.Tensor,
    *,
    dt: float,
    trend_half_lives: tuple[float, float],
    activity_half_lives: tuple[float, float],
    trend_weight: float,
    activity_weight: float,
    initial_activity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen per-feature RMS of the covariance and drift features.

    Run once, at the initial memory parameters, with the realised-return
    activity feed (the structural feed depends on the gammas we are about to
    fit, which would make the scales circular).  Returns ``((7,), (5,))``,
    floored away from zero so a dead feature cannot divide by zero.
    """

    device, dtype = log_paths.device, log_paths.dtype
    batch_size, point_count, state_dim = log_paths.shape
    trend_decay = [decay_from_half_life(h) for h in trend_half_lives]
    activity_decay = [decay_from_half_life(h) for h in activity_half_lives]
    zero = torch.zeros((batch_size, state_dim), device=device, dtype=dtype)
    start = torch.full(
        (batch_size, state_dim), float(initial_activity), device=device, dtype=dtype
    )
    trend_states = [zero, zero]
    activity_states = [start, start]

    covariance_total = torch.zeros(7, device=device, dtype=dtype)
    drift_total = torch.zeros(5, device=device, dtype=dtype)
    covariance_count = drift_count = 0
    with torch.no_grad():
        for step in range(point_count - 1):
            trend = (1.0 - trend_weight) * trend_states[0] + trend_weight * trend_states[1]
            activity = (1.0 - activity_weight) * activity_states[
                0
            ] + activity_weight * activity_states[1]
            pair = covariance_features(trend, activity)
            single = drift_features(trend, activity)
            covariance_total += pair.pow(2).sum(dim=(0, 1, 2))
            covariance_count += pair[..., 0].numel()
            drift_total += single.pow(2).sum(dim=(0, 1))
            drift_count += single[..., 0].numel()
            increment = log_paths[:, step + 1, :] - log_paths[:, step, :]
            source = increment.pow(2) / float(dt)
            trend_states = [
                trend_decay[i] * trend_states[i] + (1.0 - trend_decay[i]) * increment
                for i in range(2)
            ]
            activity_states = [
                activity_decay[i] * activity_states[i]
                + (1.0 - activity_decay[i]) * source
                for i in range(2)
            ]
    covariance_scale = (covariance_total / max(covariance_count, 1)).sqrt()
    drift_scale = (drift_total / max(drift_count, 1)).sqrt()
    floor = torch.finfo(dtype).eps ** 0.5
    return covariance_scale.clamp_min(floor), drift_scale.clamp_min(floor)


def _warm_start_gammas(
    log_paths: torch.Tensor,
    *,
    dt: float,
    trend_half_lives: tuple[float, float],
    activity_half_lives: tuple[float, float],
    trend_weight: float,
    activity_weight: float,
    initial_activity: float,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ridge least squares of ``r_j r_k / dt`` on ``phi``, realised-return memories.

    This is only the MLE's initialiser.  It is fast, convex and gets the scale
    right; the likelihood then moves the coefficients to where the
    structural-variance recursion actually wants them.
    """

    with torch.no_grad():
        _, point_count, state_dim = log_paths.shape
        increments = log_paths[:, 1:, :] - log_paths[:, :-1, :]
        trend_decay = [decay_from_half_life(h) for h in trend_half_lives]
        activity_decay = [decay_from_half_life(h) for h in activity_half_lives]
        zero = torch.zeros_like(increments[:, 0, :])
        start = torch.full_like(increments[:, 0, :], float(initial_activity))
        trend_states = [zero, zero]
        activity_states = [start, start]
        features: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for step in range(point_count - 1):
            trend = (
                1.0 - trend_weight
            ) * trend_states[0] + trend_weight * trend_states[1]
            activity = (1.0 - activity_weight) * activity_states[
                0
            ] + activity_weight * activity_states[1]
            features.append(covariance_features(trend, activity))
            increment = increments[:, step, :]
            targets.append(increment.unsqueeze(-1) * increment.unsqueeze(-2) / float(dt))
            trend_states = [
                trend_decay[i] * trend_states[i] + (1.0 - trend_decay[i]) * increment
                for i in range(2)
            ]
            activity_states = [
                activity_decay[i] * activity_states[i]
                + (1.0 - activity_decay[i]) * increment.pow(2) / float(dt)
                for i in range(2)
            ]
        design = torch.stack(features, dim=1)
        target = torch.stack(targets, dim=1)
        mask = torch.eye(state_dim, dtype=torch.bool, device=log_paths.device)
        solutions = []
        for selector in (mask, ~mask):
            rows = design[:, :, selector].reshape(-1, 7).double()
            values = target[:, :, selector].reshape(-1).double()
            scale = rows.pow(2).mean(dim=0).sqrt()
            scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
            scaled = rows / scale
            gram = scaled.T @ scaled
            gram = gram + float(ridge) * torch.trace(gram) / 7.0 * torch.eye(
                7, dtype=gram.dtype, device=gram.device
            )
            solution = torch.linalg.solve(gram, scaled.T @ values)
            solutions.append((solution / scale).to(log_paths.dtype))
        return solutions[0], solutions[1]


def fit_multivariate_covariance_reference_kernel(
    *,
    target_paths: torch.Tensor,
    grid: object,
    ridge_covariance: float,
    ridge_drift: float,
    initial_activity: float,
    sigma_min: float = 1e-3,
    sigma_max: float = 0.6,
    activity_update: str = "structural_variance",
    fit_memory: bool = True,
    initial_trend_half_lives: tuple[float, float] = (
        12.883786821834656,
        74.25954988259295,
    ),
    initial_activity_half_lives: tuple[float, float] = (
        11.777077698433718,
        50.337696867760634,
    ),
    initial_trend_weight: float = 0.8225691318511963,
    initial_activity_weight: float = 0.254260390996933,
    warm_start_ridge: float = 1e-3,
    steps: int = 200,
    lr: float = 5e-2,
    validation_fraction: float = 0.2,
    seed: int = 0,
    log_every: int = 20,
    verbose: bool = False,
) -> MultivariateReferenceFitResult:
    """Fit the model by penalised MLE on an 80/20 split of ``target_paths``.

    ``target_paths`` are LOG prices, shape ``(N, T + 1, d)``.  The defaults for
    the memory parameters are the values the frozen ``d = 1`` Heston run
    produced; they are only a starting point when ``fit_memory`` is true.
    """

    if target_paths.ndim != 3:
        raise ValueError("target_paths must have shape (N, T + 1, d)")
    if not (0.0 < float(validation_fraction) < 1.0):
        raise ValueError("validation_fraction must lie in (0, 1)")
    state_dim = int(target_paths.shape[2])
    dt = float(grid.dt)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(int(target_paths.shape[0]), generator=generator)
    split = int(round((1.0 - float(validation_fraction)) * len(permutation)))
    if split < 1 or split >= len(permutation):
        raise ValueError("validation_fraction leaves an empty split")
    calibration_paths = target_paths[permutation[:split]]
    validation_paths = target_paths[permutation[split:]]

    gamma_diagonal, gamma_offdiagonal = _warm_start_gammas(
        calibration_paths,
        dt=dt,
        trend_half_lives=initial_trend_half_lives,
        activity_half_lives=initial_activity_half_lives,
        trend_weight=initial_trend_weight,
        activity_weight=initial_activity_weight,
        initial_activity=initial_activity,
        ridge=warm_start_ridge,
    )
    gamma_drift = torch.zeros(5, device=target_paths.device, dtype=target_paths.dtype)

    covariance_scale, drift_scale = _feature_scales(
        calibration_paths,
        dt=dt,
        trend_half_lives=initial_trend_half_lives,
        activity_half_lives=initial_activity_half_lives,
        trend_weight=initial_trend_weight,
        activity_weight=initial_activity_weight,
        initial_activity=initial_activity,
    )

    parameterisation = _Parameterisation(
        trend_half_lives=initial_trend_half_lives,
        activity_half_lives=initial_activity_half_lives,
        trend_weight=initial_trend_weight,
        activity_weight=initial_activity_weight,
        gamma_diagonal=gamma_diagonal,
        gamma_offdiagonal=gamma_offdiagonal,
        gamma_drift=gamma_drift,
        covariance_scale=covariance_scale,
        drift_scale=drift_scale,
        fit_memory=bool(fit_memory),
    ).to(device=target_paths.device, dtype=target_paths.dtype)

    optimiser = torch.optim.Adam(
        [p for p in parameterisation.parameters() if p.requires_grad], lr=float(lr)
    )
    nll_kwargs = dict(
        dt=dt,
        initial_activity=float(initial_activity),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        activity_update=str(activity_update),
    )
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")

    for step in range(1, int(steps) + 1):
        optimiser.zero_grad(set_to_none=True)
        loss = _negative_log_likelihood(
            parameterisation, calibration_paths, **nll_kwargs
        )
        # Ridge in STANDARDISED coordinates.  Penalising the raw gammas would be
        # meaningless here: with a 1.1e+06 spread in feature scale it would crush
        # the T_j T_k coefficient (which must reach ~1e+03 to matter) while
        # leaving the constant term essentially unpenalised.
        penalty = float(ridge_covariance) * (
            parameterisation.raw_gamma_diagonal.pow(2).sum()
            + parameterisation.raw_gamma_offdiagonal.pow(2).sum()
        ) + float(ridge_drift) * parameterisation.raw_gamma_drift.pow(2).sum()
        total = loss + penalty
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite reference objective at step {step}")
        total.backward()
        # clip_grad_norm_ computes one global norm, so a single non-finite
        # gradient silently poisons EVERY parameter and only surfaces later as
        # an opaque eigh convergence failure.  Fail here instead, naming the
        # parameter.
        offenders = [
            name
            for name, parameter in parameterisation.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if offenders:
            raise RuntimeError(
                f"non-finite reference gradient at step {step}: {', '.join(offenders)}"
            )
        torch.nn.utils.clip_grad_norm_(
            [p for p in parameterisation.parameters() if p.requires_grad], 10.0
        )
        optimiser.step()
        if step % int(log_every) == 0 or step == int(steps) or step == 1:
            with torch.no_grad():
                validation = float(
                    _negative_log_likelihood(
                        parameterisation, validation_paths, **nll_kwargs
                    )
                )
            history.append(
                {
                    "step": float(step),
                    "calibration_nll": float(loss.detach()),
                    "penalty": float(penalty.detach()),
                    "validation_nll": validation,
                }
            )
            if validation < best_validation:
                best_validation = validation
                best_state = {
                    name: tensor.detach().clone()
                    for name, tensor in parameterisation.state_dict().items()
                }
            if verbose:
                print(
                    f"  reference step={step:4d} cal={float(loss.detach()):+.6f} "
                    f"val={validation:+.6f}",
                    flush=True,
                )

    if best_state is not None:
        parameterisation.load_state_dict(best_state)
    with torch.no_grad():
        calibration_nll = float(
            _negative_log_likelihood(parameterisation, calibration_paths, **nll_kwargs)
        )
        validation_nll = float(
            _negative_log_likelihood(parameterisation, validation_paths, **nll_kwargs)
        )
        trend_half_lives, activity_half_lives = parameterisation.half_lives()
        trend_weight, activity_weight = parameterisation.weights()

    kernel = MultivariateCovarianceReferenceKernel(
        grid=grid,
        state_dim=state_dim,
        trend_half_lives=(float(trend_half_lives[0]), float(trend_half_lives[1])),
        activity_half_lives=(
            float(activity_half_lives[0]),
            float(activity_half_lives[1]),
        ),
        trend_weight=float(trend_weight),
        activity_weight=float(activity_weight),
        gamma_diagonal=parameterisation.gamma_diagonal.detach().clone(),
        gamma_offdiagonal=parameterisation.gamma_offdiagonal.detach().clone(),
        gamma_drift=parameterisation.gamma_drift.detach().clone(),
        initial_activity=float(initial_activity),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        activity_update=str(activity_update),
        ridge_covariance=float(ridge_covariance),
        ridge_drift=float(ridge_drift),
        calibration_nll=calibration_nll,
        validation_nll=validation_nll,
    )
    return MultivariateReferenceFitResult(
        kernel=kernel,
        history=history,
        calibration_nll=calibration_nll,
        validation_nll=validation_nll,
    )
