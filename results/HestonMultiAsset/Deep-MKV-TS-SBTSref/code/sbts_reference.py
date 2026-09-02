"""SBTS reference kernel for Deep-MKV-TS (multi-asset Heston, d = 8).

WHAT THIS REPLACES
------------------
The published Deep-MKV-TS run on ``dataset/HestonMultiAsset`` drives its matrix
control with ``multivariate_reference.MultivariateCovarianceReferenceKernel``
(the Guyon-Lekeufack path-dependent memory kernel).  This module supplies a
drop-in alternative whose reference drift and reference volatility are the ones
the SBTS baseline actually uses, so that Deep-MKV-TS runs Algorithm 1 on top of
SBTS instead of on top of the memory kernel.

It exposes exactly the four attributes ``matrix_control_multiasset`` requires
(``grid``, ``state_dim``, ``evaluate``, ``evaluate_sigmas_all``) plus the
``sigma_min`` / ``sigma_max`` pair the control cross-checks.


WHAT SBTS ACTUALLY MODELS  (read the source before changing anything here)
-------------------------------------------------------------------------
``results/HestonMultiAsset/SBTS/code/sbts_generate_multiasset.py`` builds its
training tensor as

    R[m, t] = log S[m, t+1] / S[m, t]            raw log-returns, (M, T-1, d)
    sigma_j = std(R[:, :, j])                    one scalar per asset
    Rt[m, t] = R[m, t] * sqrt(dt) / sigma_j      scaled log-returns
    X = concat([zeros(M, 1, d), Rt], axis=1)     (M, T, d)

and then diffuses **the return itself**, not the log price:

    X_ <- X_ + drift * timestep + Brownian * sqrt(timestep)

with ``timeSeries[i+1] = X_``.  So the SBTS state at date ``i`` is the scaled
log-return ``rt_i``, the diffusion coefficient in that coordinate is the
identity by construction (step 2 above is exactly what makes it 1), and the
generated series is read back as ``R_gen = Rt_gen * sigma_j / sqrt(dt)``.

CONSEQUENCE FOR THE LOG-PRICE COORDINATE THAT DEEP-MKV-TS USES.  Write
``D = diag(sigma_j / sqrt(dt))`` (this is the "corrected" volatility: constant
in time and across paths, and equal to the per-asset realised log-return
standard deviation divided by sqrt(dt)).  One SBTS interval is

    rt_{i+1} = rt_i + bt_i dt + sqrt(dt) Z ,     x_{i+1} - x_i = D rt_{i+1}

so, matching against the Deep-MKV-TS Euler step
``x_{i+1} - x_i = b^ref dt + sigma^ref sqrt(dt) Z``,

    sigma^ref = D                                                    (exact)
    b^ref     = D (rt_i / dt + bt_i)                                 (exact)

and, because the k = 0 SBTS drift is ``bt_i = (E_w[X^m_{i+1}] - rt_i) / dt``,
the ``rt_i`` carry term cancels identically and

    b^ref_i = D E_w[ Xtilde^m_{i+1} ] / dt
            = E_w[ R^m_{i+1} ] / dt                                  (*)

i.e. **the kernel-weighted average RAW log-return of the training bank at the
date being generated, divided by dt**.  That is the drift implemented below.
The cancellation is not a simplification: it is why this reference is a genuine
log-price drift rather than a momentum artefact.

The weights are the SBTS Markovian ones, over the last ``K`` realised returns of
the Deep-MKV-TS prefix:

    w_m = prod_{l = i-K+1}^{i} Kh( Xtilde^m_l - rt_l ),
    Kh(u) = (h^2 - ||u||^2)^2 1{||u|| < h}       (radial, NOT a product kernel)

with ``Xtilde^m_l = Rt[m, l-1]`` (the dummy zero row shifts the index by one)
and ``rt_l = (x_l - x_{l-1}) sqrt(dt) / sigma``.  Fewer than ``K`` lags are used
while ``i < K``, exactly as the reference implementation's ``index_queue`` does.
If every weight in a batch row underflows to zero the drift is set to zero,
which is the reference's ``expec_den > 0`` fallback.


NUMERICS: WHY LOG-WEIGHTS ARE NOT OPTIONAL
------------------------------------------
At ``h = 0.31`` the kernel is bounded by ``h^4 = 9.2e-3``.  A product of K = 20
such factors is at most ``6e-42``, below the smallest normal float32
(``1.2e-38``), so the naive product underflows to zero for *every* bank path and
the drift silently becomes zero -- the same silent failure the SBTS module
docstring warns about for h too small.  This module therefore accumulates
``log Kh`` and normalises by the per-row maximum before exponentiating, so only
weight *ratios* are ever materialised.  Ratios are all the drift needs.


COST: WHY THE DISTANCES ARE COMPUTED BY EXPANSION
-------------------------------------------------
The naive form materialises ``bank[None] - rt[:, None]`` of shape
``(K, B, M, d) = (20, 256, 8192, 8)``, i.e. 1.3 GB per step in float32, 252
times per forward.  Instead we use

    ||a - b||^2 = ||a||^2 - 2 <a, b> + ||b||^2

which turns each lag into a ``(B, d) x (d, M)`` matmul: ``K B M d = 3.4e8``
multiply-adds per step, with only ``(K, B, M)`` scalars alive.  The bank norms
``||Xtilde^m_l||^2`` are precomputed once at construction.


DIFFERENTIABILITY (disclosed, and deliberately not autograd)
------------------------------------------------------------
The drift is path-dependent through ``w``, so ``d b^ref / d x`` is not zero and
the ``autograd_replay`` drift-adjoint backend would happily differentiate it --
but the graph of a single call holds several ``(B, M)`` tensors (~32 MB), which
over 252 steps is ~8 GB on top of the existing 17.7 GiB peak.  So the Jacobian
is instead computed **in the same pass as the drift** and returned through a
custom ``autograd.Function`` that saves only a ``(B, d, d)`` tensor (64 KB per
step).  With ``p_m = w_m / sum w``, ``Y_m = R[m, i]`` and

    g^{(j)}_m = d log w_m / d rt_{l_j} = 4 delta^{(j)}_m / (h^2 - ||delta||^2),
    delta^{(j)}_m = Xtilde^m_{l_j} - rt_{l_j}

the closed form is

    d b^ref / d rt_{l_j} = ( E_p[Y g^T] - E_p[Y] E_p[g]^T ) / dt
                         = Cov_p(Y, g^{(j)}) / dt

and then ``d rt_l / d x_l = D^{-1}``, ``d rt_l / d x_{l-1} = -D^{-1}``.

ALL K lags are carried (``jacobian_lags = -1``, the default).  This matters and
was not always so: ``b^ref_i`` reads the last ``K = 20`` returns, so perturbing
one state changes the drift at 20 different future steps, and an adjoint that
keeps only the newest lag hands the optimiser 1 of those 20 pathways.  An
earlier version of this file did exactly that, on a memory argument that does
not survive contact with the numbers -- the ``(B, M)`` blow-up quoted below is
the AUTOGRAD route through the weights, whereas the analytic Jacobian is
``(B, J, d, d)`` = 1.3 MB per step at ``B = 256, J = 20, d = 8`` in float32,
about 330 MB over the 252-step rollout.  The blocks share ``p`` and ``Y`` and
differ only in ``g``, so all J come out of one batched pass.

Set ``jacobian_lags = 1`` to recover the truncated behaviour as an ablation, or
``weight_grad_mode="detached"`` to drop the drift path-derivative entirely --
that is the arm to use when comparing ``N_pi`` variants, so the comparison
stays one-factor.

``evaluate_sigmas_all`` is exactly constant, so its derivative is zero and the
``differentiable_sigma`` question that the memory kernel has to answer does not
arise here.


N_pi
----
SBTS sub-integrates each interval with ``N_pi = 50`` Euler substeps.
Deep-MKV-TS takes exactly one step per interval, so "N_pi = 50" has no literal
meaning as a drift function.  Two well-defined readings are implemented:

* ``npi = 1``   -- the k = 0 branch only, equation (*) above.  This already
  reproduces the SBTS conditional mean endpoint exactly, because the k = 0
  drift is built to move the state onto ``E_w[X^m_{i+1}]`` in one step of
  length dt.
* ``npi > 1``   -- run the reference's own inner loop (both branches, including
  the ``weights_tilde`` bridge reweighting) with the Brownian increments set to
  zero, then report the *effective* drift ``D * X_end / dt``.  Deterministic,
  hence a genuine drift function, and it retains the bridge correction that the
  k = 0 branch does not see.  Cost is ~N_pi times the inner reduction.

There is a third reading -- refine the whole Deep-MKV-TS grid to 251 * 50 steps
-- which is a 50x training cost and is not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

__all__ = [
    "SBTSReferenceKernel",
    "build_sbts_reference_kernel",
    "scaled_returns_from_prices",
]

_LOG_ZERO = float("-inf")


def scaled_returns_from_prices(
    prices: torch.Tensor, *, dt: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce ``sbts_generate_multiasset.scale_log_returns`` in torch.

    Parameters
    ----------
    prices : ``(M, T, d)`` price paths (float64 strongly preferred).
    dt : calendar step.

    Returns
    -------
    ``(raw_returns, scaled_returns, sigma_assets)`` with shapes
    ``(M, T-1, d)``, ``(M, T-1, d)`` and ``(d,)``.

    ``sigma_assets`` is the population std (``correction=0``), matching numpy's
    ``ndarray.std`` default used by the SBTS generator; the biased/unbiased
    difference is 2e-7 relative at M*(T-1) = 2e6 samples, but "matching" beats
    "negligible".
    """

    if prices.ndim != 3:
        raise ValueError("prices must have shape (M, T, d)")
    raw = torch.log(prices[:, 1:, :] / prices[:, :-1, :])
    sigma_assets = raw.reshape(-1, raw.shape[-1]).std(dim=0, correction=0)
    scaled = raw * math.sqrt(float(dt)) / sigma_assets
    return raw, scaled, sigma_assets


class _SBTSDrift(torch.autograd.Function):
    """Carry ``b^ref`` forward and the analytic ``d b / d x`` backward.

    ``b^ref_i`` reads the ``K`` most recent scaled returns, so a perturbation of
    one state feeds the drift through EVERY lagged return that contains it.  A
    faithful adjoint therefore carries one Jacobian block per differentiated lag,
    not just the newest one.

    ``jacobian`` is ``(B, J, d, d)``, OLDEST LAG FIRST, so ``[:, -1]`` is
    ``d b / d rt_i``.  Each block is already chained by ``d rt / d x = D^{-1}``.
    ``x_window`` is ``(B, J + 1, d)``, the states those returns are built from:

        rt_{l_j} = (x_window[:, j + 1] - x_window[:, j]) sqrt(dt) / sigma

    so return ``j`` sends ``+g_j`` to the state on its right and ``-g_j`` to the
    state on its left.  Interior states sit on the right of one return and the
    left of the next, and correctly receive both contributions.
    """

    @staticmethod
    def forward(ctx, x_window, drift, jacobian):  # noqa: D401
        ctx.save_for_backward(jacobian)
        return drift.clone()

    @staticmethod
    def backward(ctx, grad_drift):
        (jacobian,) = ctx.saved_tensors
        # (B, d) x (B, J, d, d) -> (B, J, d): gradient w.r.t. each lagged return.
        g = torch.einsum("ba,bjac->bjc", grad_drift, jacobian)
        grad_window = torch.zeros(
            (g.shape[0], g.shape[1] + 1, g.shape[2]),
            device=g.device,
            dtype=g.dtype,
        )
        grad_window[:, 1:, :] += g
        grad_window[:, :-1, :] -= g
        return grad_window, None, None


@dataclass(frozen=True)
class SBTSReferenceKernel:
    """SBTS drift + SBTS corrected volatility, in Deep-MKV-TS log-price coordinates.

    Attributes
    ----------
    grid : object exposing ``dt`` and ``num_steps`` (a ``DiscreteTimeGrid``).
    state_dim : ``d``.
    bank_scaled : ``(M, N, d)`` scaled log-returns of the TRAIN split.
    bank_raw : ``(M, N, d)`` raw log-returns of the TRAIN split.
    sigma_assets : ``(d,)`` per-asset log-return std.
    h, markov_order, npi : SBTS hyperparameters (``h``, ``K``, ``N_pi``).
    sigma_min, sigma_max : cross-checked by the matrix control; the constant
        reference spectrum must lie inside ``[sigma_min, sigma_max]`` or
        construction fails loudly rather than being silently clipped.
    weight_grad_mode : ``"analytic"`` or ``"detached"``.
    jacobian_lags : how many trailing lags carry a path-derivative.  ``-1``
        (default) means all ``markov_order`` of them, which is the exact
        adjoint.  Any value in ``1..markov_order`` is the truncated ablation;
        ``1`` reproduces the pre-fix behaviour.
    """

    grid: object
    state_dim: int
    bank_scaled: torch.Tensor
    bank_raw: torch.Tensor
    sigma_assets: torch.Tensor
    h: float
    markov_order: int = 20
    npi: int = 1
    sigma_min: float = 1e-3
    sigma_max: float = 0.6
    weight_grad_mode: str = "analytic"
    jacobian_lags: int = -1

    _bank_sq: torch.Tensor = field(init=False, repr=False)
    _sigma_ref: torch.Tensor = field(init=False, repr=False)
    _sigma_ref_inv_diag: torch.Tensor = field(init=False, repr=False)

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        d = int(self.state_dim)
        if self.bank_scaled.ndim != 3 or self.bank_raw.ndim != 3:
            raise ValueError("bank tensors must have shape (M, N, d)")
        if self.bank_scaled.shape != self.bank_raw.shape:
            raise ValueError("bank_scaled and bank_raw must have the same shape")
        if int(self.bank_scaled.shape[-1]) != d:
            raise ValueError("bank state dimension must equal state_dim")
        if int(self.bank_scaled.shape[1]) < int(self.grid.num_steps):
            raise ValueError(
                "bank must supply one target return per grid step: got "
                f"{int(self.bank_scaled.shape[1])} returns for "
                f"{int(self.grid.num_steps)} steps"
            )
        if not (float(self.h) > 0.0):
            raise ValueError("h must be > 0")
        if int(self.markov_order) < 1:
            raise ValueError("markov_order must be >= 1")
        if int(self.npi) < 1:
            raise ValueError("npi must be >= 1")
        if self.weight_grad_mode not in ("analytic", "detached"):
            raise ValueError("weight_grad_mode must be 'analytic' or 'detached'")
        jac_lags = int(self.jacobian_lags)
        if jac_lags == 0 or jac_lags < -1:
            raise ValueError("jacobian_lags must be -1 (all lags) or >= 1")
        if jac_lags > int(self.markov_order):
            raise ValueError(
                f"jacobian_lags={jac_lags} exceeds markov_order="
                f"{int(self.markov_order)}: the drift cannot depend on a lag it "
                "never reads"
            )

        # ||Xtilde^m_l||^2 for the expansion, precomputed once.
        object.__setattr__(
            self, "_bank_sq", self.bank_scaled.pow(2).sum(dim=-1).contiguous()
        )

        # sigma^ref = diag(sigma_j / sqrt(dt)), constant.  Verify it is inside
        # the control's clip box: the control passes it through UNCLIPPED and
        # relies on the kernel having already projected the spectrum.
        scale = self.sigma_assets / math.sqrt(float(self.grid.dt))
        low, high = float(scale.min()), float(scale.max())
        if low < float(self.sigma_min) or high > float(self.sigma_max):
            raise ValueError(
                f"constant reference spectrum [{low:.6f}, {high:.6f}] escapes "
                f"[sigma_min={self.sigma_min}, sigma_max={self.sigma_max}]"
            )
        object.__setattr__(self, "_sigma_ref", torch.diag(scale))
        object.__setattr__(self, "_sigma_ref_inv_diag", 1.0 / scale)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _scaled_returns_of_prefix(
        self, x_prefix: torch.Tensor, *, lags: int
    ) -> torch.Tensor:
        """Last ``lags`` scaled returns of the prefix, newest last.

        Returns ``(B, lags, d)``.  ``rt_l = (x_l - x_{l-1}) sqrt(dt) / sigma``.
        """

        window = x_prefix[:, -(lags + 1) :, :]
        increments = window[:, 1:, :] - window[:, :-1, :]
        return increments * (
            math.sqrt(float(self.grid.dt))
            / self.sigma_assets.to(device=x_prefix.device, dtype=x_prefix.dtype)
        )

    def _log_weights(
        self, returns: torch.Tensor, *, step_index: int, lags: int
    ) -> torch.Tensor:
        """``log w`` up to a per-row constant, shape ``(B, M)``.

        ``returns`` is ``(B, lags, d)`` with the newest lag last.  Lag ``j`` of
        that window is date ``l = step_index - lags + 1 + j``, whose bank slice
        is ``bank_scaled[:, l - 1]``.
        """

        device, dtype = returns.device, returns.dtype
        bank_lo = int(step_index) - lags  # = (l - 1) for j = 0
        bank = self.bank_scaled[:, bank_lo : bank_lo + lags, :].to(
            device=device, dtype=dtype
        )  # (M, lags, d)
        bank_sq = self._bank_sq[:, bank_lo : bank_lo + lags].to(
            device=device, dtype=dtype
        )  # (M, lags)

        # (lags, B, d) @ (lags, d, M) -> (lags, B, M)
        query = returns.transpose(0, 1).contiguous()  # (lags, B, d)
        cross = torch.bmm(query, bank.permute(1, 2, 0).contiguous())
        sq = (
            bank_sq.transpose(0, 1).unsqueeze(1)  # (lags, 1, M)
            - 2.0 * cross
            + query.pow(2).sum(dim=-1, keepdim=True)  # (lags, B, 1)
        )

        h_sq = float(self.h) ** 2
        support = h_sq - sq
        inside = support > 0.0
        # log Kh = 2 log(h^2 - ||u||^2); the constant -4 log h that would
        # normalise the kernel to a max of 1 is a per-lag additive constant and
        # cancels in the weight ratio, so it is not applied.
        log_kernel = torch.where(
            inside,
            2.0 * torch.log(support.clamp_min(torch.finfo(dtype).tiny)),
            torch.full_like(support, _LOG_ZERO),
        )
        return log_kernel.sum(dim=0)  # (B, M)

    @staticmethod
    def _normalised_weights(
        log_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(p, alive)``: row-normalised weights and a row-validity mask."""

        row_max = log_weights.max(dim=1, keepdim=True).values
        alive = torch.isfinite(row_max)
        shifted = torch.where(
            alive,
            log_weights - torch.where(alive, row_max, torch.zeros_like(row_max)),
            torch.full_like(log_weights, _LOG_ZERO),
        )
        weights = torch.exp(shifted)
        total = weights.sum(dim=1, keepdim=True)
        safe = total > 0.0
        p = torch.where(
            safe, weights / total.clamp_min(1e-300), torch.zeros_like(weights)
        )
        return p, (safe & alive).squeeze(1)

    # ------------------------------------------------------------------ #
    # public surface required by matrix_control_multiasset
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

        batch = int(x_prefix.shape[0])
        device, dtype = x_prefix.device, x_prefix.dtype
        sigma = self._sigma_ref.to(device=device, dtype=dtype).expand(
            batch, int(self.state_dim), int(self.state_dim)
        )

        target_raw = self.bank_raw[:, step, :].to(device=device, dtype=dtype)  # (M, d)
        lags = min(step, int(self.markov_order))

        if lags == 0:
            # SBTS i = 0: weights are uniform (the dummy zero row).
            drift = target_raw.mean(dim=0, keepdim=True).expand(batch, -1) / float(
                self.grid.dt
            )
            return drift.contiguous(), sigma

        with torch.no_grad():
            returns = self._scaled_returns_of_prefix(x_prefix.detach(), lags=lags)
            log_weights = self._log_weights(returns, step_index=step, lags=lags)
            p, alive = self._normalised_weights(log_weights)
            mean_target = p @ target_raw  # (B, d)
            drift = torch.where(
                alive.unsqueeze(1), mean_target, torch.zeros_like(mean_target)
            ) / float(self.grid.dt)

            if int(self.npi) > 1:
                drift = self._effective_drift_npi(
                    returns=returns,
                    log_weights=log_weights,
                    alive=alive,
                    step_index=step,
                )

        if self.weight_grad_mode == "detached" or not x_prefix.requires_grad:
            return drift.contiguous(), sigma

        if int(self.npi) > 1:
            # The sub-integrated drift has no closed-form Jacobian here; the
            # N_pi sweep is run with weight_grad_mode="detached" so that the
            # comparison stays one-factor.  Refuse rather than silently
            # returning a gradient that belongs to a different quantity.
            raise ValueError(
                "weight_grad_mode='analytic' is only defined for npi=1; use "
                "'detached' for the sub-integrated arm"
            )

        # How many of the `lags` returns actually carry a derivative.  -1 (the
        # default) means all of them, i.e. the exact adjoint of a drift that
        # reads `markov_order` lags.  A smaller value is the truncated ablation.
        jac_lags = int(self.jacobian_lags)
        n_jac = lags if jac_lags < 0 else min(lags, jac_lags)

        with torch.no_grad():
            jacobian = self._drift_jacobian(
                returns=returns[:, -n_jac:, :],
                p=p,
                alive=alive,
                target_raw=target_raw,
                step_index=step,
            )
        drift = _SBTSDrift.apply(x_prefix[:, -(n_jac + 1):, :], drift, jacobian)
        return drift, sigma

    def _drift_jacobian(
        self,
        *,
        returns: torch.Tensor,
        p: torch.Tensor,
        alive: torch.Tensor,
        target_raw: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        """``d b^ref / d x`` for every differentiated lag, shape ``(B, J, d, d)``.

        ``returns`` is ``(B, J, d)``, already restricted to the ``J`` lags that
        carry a derivative, oldest first.  Lag ``j`` is bank date
        ``step_index - J + j``.

        Lag ``j`` contributes ``2 log(h^2 - ||Xtilde^m_{l_j} - rt_{l_j}||^2)`` to
        ``log w_m``, so

            g^{(j)}_m = d log w_m / d rt_{l_j}
                      = 4 (Xtilde^m_{l_j} - rt_{l_j}) / (h^2 - ||delta||^2)
            d b / d rt_{l_j} = Cov_p(Y, g^{(j)}) / dt

        The normalised weights ``p`` and the targets ``Y`` are SHARED across
        lags -- only ``g`` changes -- which is why all ``J`` blocks come out of
        one batched pass instead of a Python loop over lags.
        """

        n_lags = int(returns.shape[1])
        device, dtype = returns.device, returns.dtype
        lo = int(step_index) - n_lags

        query = returns.transpose(0, 1).contiguous()  # (J, B, d)
        bank = (
            self.bank_scaled[:, lo:lo + n_lags, :]
            .to(device=device, dtype=dtype)
            .permute(1, 0, 2)
            .contiguous()
        )  # (J, M, d), Xtilde^m_{l_j}
        bank_sq = (
            self._bank_sq[:, lo:lo + n_lags]
            .to(device=device, dtype=dtype)
            .transpose(0, 1)
            .contiguous()
        )  # (J, M)

        sq = (
            bank_sq.unsqueeze(1)  # (J, 1, M)
            - 2.0 * torch.bmm(query, bank.transpose(1, 2))  # (J, B, M)
            + query.pow(2).sum(dim=-1, keepdim=True)  # (J, B, 1)
        )
        # g_m = 4 (Xtilde^m - rt) / (h^2 - ||.||^2) is defined only strictly
        # inside the support.  Outside it the kernel factor is 0, so p_m = 0 and
        # the term drops -- but 4/(h^2 - ||.||^2) is then +-inf or a division by
        # a clamped zero, and 0 * inf is NaN.  The mask therefore has to be
        # applied BEFORE the division, not after it.  ``p`` is the FULL-window
        # weight and is shared by every lag, hence the broadcast over ``J``.
        support = float(self.h) ** 2 - sq
        contributing = (p.unsqueeze(0) > 0.0) & (support > 0.0)
        support_safe = torch.where(contributing, support, torch.ones_like(support))
        inv = torch.where(
            contributing, 4.0 / support_safe, torch.zeros_like(support)
        )  # (J, B, M)

        pi = p.unsqueeze(0) * inv  # (J, B, M)
        e_yg = torch.einsum("jbm,ma,jmc->jbac", pi, target_raw, bank)
        e_yg = e_yg - torch.einsum("jbm,ma,jbc->jbac", pi, target_raw, query)
        e_y = p @ target_raw  # (B, d), lag-independent
        e_g = (
            torch.einsum("jbm,jmc->jbc", pi, bank)
            - pi.sum(dim=-1, keepdim=True) * query
        )  # (J, B, d)
        cov = e_yg - torch.einsum("ba,jbc->jbac", e_y, e_g)

        # d rt / d x = sqrt(dt) / sigma = 1 / (sigma / sqrt(dt)), same for all lags
        chain = self._sigma_ref_inv_diag.to(device=device, dtype=dtype)
        jac = (cov / float(self.grid.dt)) * chain.view(1, 1, 1, -1)
        jac = jac.permute(1, 0, 2, 3).contiguous()  # (B, J, d, d)
        return torch.where(alive.view(-1, 1, 1, 1), jac, torch.zeros_like(jac))

    def _effective_drift_npi(
        self,
        *,
        returns: torch.Tensor,
        log_weights: torch.Tensor,
        alive: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        """Noise-free SBTS sub-integration; returns ``D * X_end / dt``.

        Mirrors the ``k == 0`` / ``k > 0`` branches of
        ``sbts_generate_multiasset._simulate_one`` with the Brownian increments
        set to zero.  Weights are frozen over the interval, as in the reference.
        """

        dt = float(self.grid.dt)
        npi = int(self.npi)
        device, dtype = returns.device, returns.dtype
        target_scaled = self.bank_scaled[:, step_index, :].to(
            device=device, dtype=dtype
        )  # (M, d)

        state = returns[:, -1, :].clone()  # rt_i, (B, d)

        # weights_tilde = w * exp(+||Xtilde_{i+1} - rt_i||^2 / (2 dt)), with the
        # difference taken at the START of the interval, as in the reference.
        diff0 = target_scaled.unsqueeze(0) - state.unsqueeze(1)  # (B, M, d)
        log_tilde = log_weights + diff0.pow(2).sum(dim=-1) / (2.0 * dt)
        del diff0

        for k in range(npi):
            time_prev = dt * k / npi
            time_step = dt / npi
            remaining = dt - time_prev
            if k == 0:
                logw = log_weights
            else:
                diff = target_scaled.unsqueeze(0) - state.unsqueeze(1)
                logw = log_tilde - diff.pow(2).sum(dim=-1) / (2.0 * remaining)
                del diff
            row_max = logw.max(dim=1, keepdim=True).values
            weights = torch.exp(
                logw - torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
            )
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            total = weights.sum(dim=1, keepdim=True)
            numerator = weights @ target_scaled - total * state
            step_drift = numerator / (total.clamp_min(1e-300) * remaining)
            step_drift = torch.where(
                total > 0.0, step_drift, torch.zeros_like(step_drift)
            )
            state = state + step_drift * time_step

        scale = self.sigma_assets.to(device=device, dtype=dtype) / math.sqrt(dt)
        effective = state * scale / dt
        return torch.where(alive.unsqueeze(1), effective, torch.zeros_like(effective))

    def evaluate_sigmas_all(self, paths: torch.Tensor) -> torch.Tensor:
        """Constant ``sigma^ref`` at every date, shape ``(B, N, d, d)``."""

        if paths.ndim != 3:
            raise ValueError("paths must have shape (B, N + 1, d)")
        if int(paths.shape[1]) != int(self.grid.num_steps) + 1:
            raise ValueError("path time dimension must align with grid")
        d = int(self.state_dim)
        sigma = self._sigma_ref.to(device=paths.device, dtype=paths.dtype)
        return sigma.view(1, 1, d, d).expand(
            int(paths.shape[0]), int(self.grid.num_steps), d, d
        )

    def evaluate_drifts_all(self, paths: torch.Tensor) -> torch.Tensor:
        """Reference drift at every date, shape ``(B, N, d)``.  Diagnostics only."""

        drifts = []
        for step in range(int(self.grid.num_steps)):
            drift, _ = self.evaluate(paths[:, : step + 1, :], step_index=step)
            drifts.append(drift)
        return torch.stack(drifts, dim=1)


def build_sbts_reference_kernel(
    *,
    train_prices: torch.Tensor,
    grid: object,
    h: float,
    markov_order: int = 20,
    npi: int = 1,
    sigma_min: float = 1e-3,
    sigma_max: float = 0.6,
    weight_grad_mode: str = "analytic",
    jacobian_lags: int = -1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SBTSReferenceKernel:
    """Build the kernel from TRAIN-split price paths.

    ``train_prices`` must be the training split and nothing else: the bank is
    the conditioning set the reference drift averages over, so feeding it
    validation or test paths is training on them.
    """

    raw, scaled, sigma_assets = scaled_returns_from_prices(
        train_prices.double(), dt=float(grid.dt)
    )
    return SBTSReferenceKernel(
        grid=grid,
        state_dim=int(train_prices.shape[-1]),
        bank_scaled=scaled.to(device=device, dtype=dtype).contiguous(),
        bank_raw=raw.to(device=device, dtype=dtype).contiguous(),
        sigma_assets=sigma_assets.to(device=device, dtype=dtype).contiguous(),
        h=float(h),
        markov_order=int(markov_order),
        npi=int(npi),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        weight_grad_mode=str(weight_grad_mode),
        jacobian_lags=int(jacobian_lags),
    )
