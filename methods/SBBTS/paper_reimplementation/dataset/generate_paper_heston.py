"""
Paper's own Heston dataset (arXiv:2604.07159 §5.1 + Table 3).

This is NOT the benchmark's Heston dataset. The paper uses a *heterogeneous*
dataset: every trajectory draws its own (r, kappa, theta, rho, xi) from Table 3,
and both the price and the variance channel are kept (d = 2), because the
evaluation metric is a bivariate MLE that needs v_t.

    dX_t = r X_t dt + sqrt(v_t) X_t dW^X_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t,   corr(W^X, W^v) = rho

Simulation scheme is the authors' exact-variance / inverse-Gaussian sampler,
vendored verbatim at ``../../code/reference/utils/data_generation.py``.

Table 3 ranges: kappa [0.5, 4], theta [0.5, 1.5], xi [0.1, 0.9],
                rho [-0.9, 0.9], r [0.01, 0.1]
Paper §5.1:     M = 5000 trajectories, N = 252 steps, dt = 1/252, S0 = 1, v0 = 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REF = HERE.parent.parent / "code" / "reference"
sys.path.insert(0, str(REF / "utils"))

from data_generation import DataGenerator  # noqa: E402  (vendored verbatim)

# Table 3 (paper appendix C.1)
R_RANGE = [0.01, 0.10]
KAPPA_RANGE = [0.5, 4.0]
THETA_RANGE = [0.5, 1.5]
RHO_RANGE = [-0.9, 0.9]
XI_RANGE = [0.1, 0.9]

# Paper section 5.1
M_DEFAULT = 5000
N_DEFAULT = 252
DT = 1.0 / 252.0


def generate(M=M_DEFAULT, N=N_DEFAULT, seed=0):
    """Return (X, params): X (M, N+1, 2) = [price, variance], params (M, 5).

    ``params`` columns are [r, kappa, theta, rho, xi], matching the draw order
    inside ``DataGenerator.generate_heston``. The reference does not persist the
    ground-truth parameters, so we replay the same seeded draw sequence to
    recover them, then restore the RNG state before calling the generator.
    """
    outer_state = np.random.get_state()
    try:
        np.random.seed(seed)
        gen = DataGenerator(M)

        state_before = np.random.get_state()
        r = np.random.uniform(*R_RANGE, M)
        kappa = np.random.uniform(*KAPPA_RANGE, M)
        theta = np.random.uniform(*THETA_RANGE, M)
        rho = np.random.uniform(*RHO_RANGE, M)
        xi = np.random.uniform(*XI_RANGE, M)
        params = np.stack([r, kappa, theta, rho, xi], axis=1)

        np.random.set_state(state_before)
        X = gen.generate_heston(
            r_range=R_RANGE, kappa_range=KAPPA_RANGE, theta_range=THETA_RANGE,
            rho_range=RHO_RANGE, xi_range=XI_RANGE, N=N, dt=DT, S0=1.0, v0=1.0,
        )
    finally:
        np.random.set_state(outer_state)

    return X, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=M_DEFAULT)
    ap.add_argument("--N", type=int, default=N_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(HERE))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    X, params = generate(args.M, args.N, args.seed)
    elapsed = time.perf_counter() - t0

    x_name = f"X_heston_paper_{args.M}x{args.N + 1}x2.npy"
    p_name = f"params_true_{args.M}x5.npy"
    np.save(out / x_name, X)
    np.save(out / p_name, params)

    meta = dict(
        source="arXiv:2604.07159 section 5.1 + Table 3",
        M=args.M, N=args.N, dt=DT, S0=1.0, v0=1.0, seed=args.seed,
        shape=list(X.shape), channels=["price", "variance"],
        param_columns=["r", "kappa", "theta", "rho", "xi"],
        ranges=dict(r=R_RANGE, kappa=KAPPA_RANGE, theta=THETA_RANGE,
                    rho=RHO_RANGE, xi=XI_RANGE),
        price_min=float(X[:, :, 0].min()), price_max=float(X[:, :, 0].max()),
        var_min=float(X[:, :, 1].min()), var_max=float(X[:, :, 1].max()),
        n_nonfinite=int((~np.isfinite(X)).sum()),
        n_nonpositive_price=int((X[:, :, 0] <= 0).sum()),
        elapsed_sec=round(elapsed, 1),
        files=[x_name, p_name],
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
