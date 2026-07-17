"""
generate_rw.py — Calibrated Random Walk (GBM) baseline for Heston benchmark.

Generates 8 192 price paths of length 128 by drawing i.i.d. Gaussian log-returns
calibrated to the empirical mean and std of the real Heston training data.

This is the simplest possible baseline: no temporal structure, no stochastic
volatility, no autocorrelation — just matched marginals.

Usage:
    python generate_rw.py   # generates all 5 seeds
"""

import numpy as np
import os
import json

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset", "Heston")
OUT_BASE    = os.path.join(REPO_ROOT, "methods", "RW", "generated_paths")

N, T = 8192, 128
N_SEEDS = 5

def main():
    # ── Calibrate from real Heston data ──────────────────────────────────────
    S_real = np.load(os.path.join(DATASET_DIR, "heston_S_8192x128.npy"))
    R_real = np.diff(np.log(S_real), axis=1)   # (8192, 127) log-returns
    mu_dt  = float(R_real.mean())              # per-step drift
    sig_dt = float(R_real.std())              # per-step vol (≈ √(θ·dt) for Heston)
    print(f"Calibrated from Heston data:")
    print(f"  mu_dt  = {mu_dt:.6f}  (= μ·dt ≈ 0.05/250)")
    print(f"  sig_dt = {sig_dt:.6f}  (= σ·√dt ≈ √(θ)/√250 ≈ 0.0126)")

    for seed in range(N_SEEDS):
        rng   = np.random.default_rng(seed)
        R_gen = rng.normal(mu_dt, sig_dt, size=(N, T - 1))   # i.i.d. Gaussian
        S_gen = np.empty((N, T), dtype=np.float64)
        S_gen[:, 0] = 100.0                                    # anchor at S₀=100
        for t in range(T - 1):
            S_gen[:, t + 1] = S_gen[:, t] * np.exp(R_gen[:, t])

        out_dir = os.path.join(OUT_BASE, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "generated_paths_8192x128.npy"), S_gen)

        meta = dict(
            seed=seed, method="RW", shape=list(S_gen.shape),
            mu_dt=mu_dt, sig_dt=sig_dt,
            description="Calibrated GBM: R_t ~ N(mu_dt, sig_dt^2) i.i.d.",
        )
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  seed {seed}: S_T mean={S_gen[:,-1].mean():.2f}  std={S_gen[:,-1].std():.2f}")

    print(f"\nSaved {N_SEEDS} seeds to {OUT_BASE}")

if __name__ == "__main__":
    main()
