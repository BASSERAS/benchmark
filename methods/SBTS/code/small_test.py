"""
STEP 2 — SBTS Small Sanity Test.

Runs SBTS on a SMALL subset of the Heston data before launching the full
8192-path generation. Verifies:

  1. Numba JIT compiles without error
  2. Output shape and price range are correct
  3. Generated log-return statistics match training data (mean, std, skew)
  4. No NaN / inf in output
  5. Visual plot saved for manual inspection

Paper reference (Appendix C):  h=0.4, K=1, N_pi=200, dt=1/252 for Heston T=100.
We use dt=1/250 and T=128; small test uses N_train=200, M_simu=20, T_small=32.

Run:
    /home/tbasseras/sbts-venv/bin/python small_test.py
"""

import os, sys, time
import numpy as np
from scipy import stats as sp_stats

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CODE  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)

from sbts_generate import generate_paths, warmup_jit, DT, S0

# ── Small-test parameters ─────────────────────────────────────────────────────
N_TRAIN = 200    # training paths (subset of 8192)
M_SIMU  = 20     # paths to generate
T_SMALL = 32     # timesteps (truncated from 128)
H       = 0.4    # paper default for Heston (Appendix C)
K       = 1      # Markovian order (paper: 1 for Heston)
N_PI    = 200    # Euler substeps (paper: 200)
N_WORK  = 4      # workers for small test (well within 16-core limit)

print("=" * 60)
print("SBTS Small Sanity Test")
print("=" * 60)

# ── Load and truncate ─────────────────────────────────────────────────────────
data_path = os.path.join(BENCH, "dataset/Heston/heston_S_8192x128.npy")
S_all     = np.load(data_path)
print(f"Loaded: {S_all.shape}  dtype={S_all.dtype}  "
      f"S[:,0] mean={S_all[:,0].mean():.2f}")

S_train = S_all[:N_TRAIN, :T_SMALL].copy()
print(f"Small training set: {S_train.shape}  "
      f"prices in [{S_train.min():.1f}, {S_train.max():.1f}]")

# ── [1] Numba warm-up ─────────────────────────────────────────────────────────
print("\n[1] Numba JIT warm-up")
warmup_jit()

# ── [2] Generate paths ────────────────────────────────────────────────────────
print(f"\n[2] Generating {M_SIMU} paths  "
      f"(N_train={N_TRAIN}, T={T_SMALL}, h={H}, K={K}, N_pi={N_PI})")
t0 = time.perf_counter()
S_gen, meta = generate_paths(
    S_train, M_simu=M_SIMU, h=H,
    K=K, N_pi=N_PI, n_workers=N_WORK, seed=0,
)
elapsed = time.perf_counter() - t0
print(f"  Wall-clock: {elapsed:.2f}s")

# ── [3] Hard checks ───────────────────────────────────────────────────────────
print("\n[3] Sanity checks")

assert S_gen.shape == (M_SIMU, T_SMALL), \
    f"FAIL shape: expected ({M_SIMU},{T_SMALL}), got {S_gen.shape}"
print(f"  ✓ Shape: {S_gen.shape}")

assert np.isfinite(S_gen).all(), "FAIL: NaN or inf in generated paths"
print("  ✓ No NaN / inf")

assert np.allclose(S_gen[:, 0], S0, atol=1e-6), \
    f"FAIL: initial prices not {S0}"
print(f"  ✓ All paths start at S0={S0}")

lo, hi = float(S_gen.min()), float(S_gen.max())
assert lo > 0.0, f"FAIL: negative prices (min={lo:.4f})"
assert hi < 1e6, f"FAIL: prices exploding (max={hi:.2f})"
print(f"  ✓ Price range: [{lo:.2f}, {hi:.2f}]")

# ── [4] Log-return statistics ─────────────────────────────────────────────────
print("\n[4] Log-return statistics (generated vs training)")

R_train = np.log(S_train[:, 1:] / S_train[:, :-1]).flatten()
R_gen   = np.log(S_gen[:, 1:]   / S_gen[:, :-1]  ).flatten()

for name, fn in [
    ("mean", np.mean),
    ("std",  np.std),
    ("skew", sp_stats.skew),
    ("kurt", sp_stats.kurtosis),
]:
    v_tr, v_ge = fn(R_train), fn(R_gen)
    ok = "✓" if abs(v_tr - v_ge) < max(3 * abs(v_tr), 0.01) else "⚠"
    print(f"  {ok} {name:6s}: train={v_tr:+.5f}   gen={v_ge:+.5f}")

# ── [5] Timing extrapolation ──────────────────────────────────────────────────
print("\n[5] Timing extrapolation → full 8192-path run")
sec_per_path = elapsed / M_SIMU
# Kernel cost: O(M_train × T × N_pi) per generated path
m_scale  = 8192.0 / N_TRAIN   # more training paths → proportionally slower
t_scale  = 128.0  / T_SMALL   # longer series
est_ser  = sec_per_path * 8192 * m_scale * t_scale
est_16w  = est_ser / 16
print(f"  Per path (now):         {sec_per_path:.3f}s")
print(f"  Expected serial:        {est_ser/60:.0f} min")
print(f"  Expected with 16 workers:{est_16w/60:.0f} min per seed")
print(f"  5 seeds (sequential):   {5*est_16w/3600:.1f} h total")

# ── [6] Diagnostic plot ───────────────────────────────────────────────────────
print("\n[6] Saving diagnostic plot")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = os.path.join(BENCH, "results/Heston/SBTS/plots")
os.makedirs(PLOT_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(
    f"SBTS Small Test  (N_train={N_TRAIN}, M_simu={M_SIMU}, "
    f"T={T_SMALL}, h={H}, K={K})", fontsize=10
)

t_ax = np.arange(T_SMALL)
ax   = axes[0]
for i in range(min(8, N_TRAIN)):
    ax.plot(t_ax, S_train[i], color="#2563EB", alpha=0.4, linewidth=0.8)
for i in range(M_SIMU):
    ax.plot(t_ax, S_gen[i],   color="#DC2626", alpha=0.6, linewidth=0.8)
ax.set_title("Price paths (blue=train, red=gen)", fontsize=8)
ax.set_xlabel("Step"); ax.set_ylabel("Price")

ax = axes[1]
ax.hist(R_train, bins=40, density=True, alpha=0.6,
        color="#2563EB", label=f"train n={len(R_train)}")
ax.hist(R_gen,   bins=40, density=True, alpha=0.6,
        color="#DC2626", label=f"gen   n={len(R_gen)}")
ax.set_title("Log-return distribution", fontsize=8)
ax.set_xlabel("Log-return"); ax.legend(fontsize=7)

ax  = axes[2]
q_tr = np.quantile(R_train, np.linspace(0.01, 0.99, 100))
q_ge = np.quantile(R_gen,   np.linspace(0.01, 0.99, 100))
ax.scatter(q_tr, q_ge, s=10, color="#7C3AED", alpha=0.7)
lim = max(abs(q_tr).max(), abs(q_ge).max()) * 1.05
ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8)
ax.set_title("QQ: generated vs train", fontsize=8)
ax.set_xlabel("Train quantile"); ax.set_ylabel("Generated quantile")

plt.tight_layout()
out = os.path.join(PLOT_DIR, "small_test.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

print("\n" + "=" * 60)
print("SMALL TEST PASSED ✓")
print(f"  {M_SIMU} paths, shape {S_gen.shape}, {elapsed:.2f}s")
print(f"  Prices positive, anchored at S0={S0}, no NaN")
print("=" * 60)
