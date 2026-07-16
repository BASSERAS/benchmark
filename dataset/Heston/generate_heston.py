"""
Heston stochastic volatility dataset generator.
Euler-Maruyama full-truncation scheme, canonical parameters from tsg-benchmark.

Saves:
  heston_S_8192x128.npy  — price paths  (8192, 128)  float64
  heston_v_8192x128.npy  — variance paths (8192, 128) float64
"""
import os
import numpy as np

MU=0.05; KAPPA=2.0; THETA=0.04; XI=0.3; RHO=-0.7; S0=100.0; V0=0.04; DT=1.0/250.0
N_SAMPLES=8192; SEQ_LEN=128; SEED=0


def generate_heston(n_samples=N_SAMPLES, seq_len=SEQ_LEN, seed=SEED):
    rng = np.random.default_rng(seed)
    T = seq_len
    z1 = rng.normal(size=(n_samples, T-1))
    z2 = rng.normal(size=(n_samples, T-1))
    z_s = z1
    z_v = RHO*z1 + np.sqrt(1.0-RHO**2)*z2
    sqrt_dt = np.sqrt(DT)
    S = np.empty((n_samples, T), dtype=np.float64)
    v = np.empty((n_samples, T), dtype=np.float64)
    S[:,0]=S0; v[:,0]=V0
    for t in range(1, T):
        v_plus = np.maximum(v[:,t-1], 0.0)
        v[:,t] = np.maximum(v[:,t-1] + KAPPA*(THETA-v_plus)*DT + XI*np.sqrt(v_plus)*sqrt_dt*z_v[:,t-1], 0.0)
        S[:,t] = S[:,t-1] + MU*S[:,t-1]*DT + np.sqrt(v_plus)*S[:,t-1]*sqrt_dt*z_s[:,t-1]
    return S, v


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    S, v = generate_heston(N_SAMPLES, SEQ_LEN, SEED)
    np.save(os.path.join(out_dir, f"heston_S_{N_SAMPLES}x{SEQ_LEN}.npy"), S)
    np.save(os.path.join(out_dir, f"heston_v_{N_SAMPLES}x{SEQ_LEN}.npy"), v)
    print(f"S shape={S.shape} S[:,0].mean={S[:,0].mean():.2f}")
    print(f"v shape={v.shape} v[:,0].mean={v[:,0].mean():.5f}")
