"""
SBTS full run — 5 seeds, 8192 paths each, length 128.

Only uses dataset/Heston/heston_S_8192x128.npy as training data.
Seeds run SEQUENTIALLY (each seed uses 16 workers → ~8 min/seed).
Total expected: ~40 min for all 5 seeds.

Hardware (hard limits):
  max 16 physical cores, max 2 GPUs (GPU used only for metrics, not here)

Saved per seed:
  methods/SBTS/generated_paths/seed_{i}/generated_paths_8192x128.npy
  methods/SBTS/generated_paths/seed_{i}/metadata.json
  methods/SBTS/losses/seed_{i}_bandwidth.json   (h selection note)
  methods/SBTS/losses/generation_time.csv

Run:
    /home/tbasseras/sbts-venv/bin/python run_all.py
"""

import os, sys, time, json, csv
import numpy as np

BENCH   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CODE    = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)

from sbts_generate import generate_paths, warmup_jit, DT, S0

# ── Hyper-parameters (paper Appendix C, Table 4 — Heston) ────────────────────
H        = 0.4    # bandwidth (paper: h=0.4 for Heston T=100, Δt=1/252)
K        = 1      # Markovian order (paper: k=1 for Heston)
N_PI     = 200    # Euler substeps  (paper: N^π=200 for Heston)
M_SIMU   = 8192   # paths to generate per seed
# Override via env: SBTS_NWORK=64 SBTS_SEEDS=1,2,3,4 python run_all.py
N_WORK   = int(os.environ.get("SBTS_NWORK",  "16"))
_seeds_env = os.environ.get("SBTS_SEEDS", "0,1,2,3,4")
SEEDS    = [int(s) for s in _seeds_env.split(",")]

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(BENCH, "dataset/Heston/heston_S_8192x128.npy")
GEN_ROOT   = os.path.join(BENCH, "methods/SBTS/generated_paths")
LOSSES_DIR = os.path.join(BENCH, "methods/SBTS/losses")

os.makedirs(LOSSES_DIR, exist_ok=True)

# ── Load training data once ───────────────────────────────────────────────────
print("=" * 65)
print("SBTS Full Run  —  5 seeds x 8192 paths x 128 steps")
print("=" * 65)
print(f"h={H}  K={K}  N_pi={N_PI}  dt={DT:.5f}  workers={N_WORK}")
print()

S_train = np.load(DATA_PATH)
print(f"Training data: {S_train.shape}  "
      f"prices in [{S_train.min():.1f}, {S_train.max():.1f}]")
print()

# ── Warm up Numba JIT (once, in parent process) ───────────────────────────────
warmup_jit()
print()

# ── CSV timing log ────────────────────────────────────────────────────────────
timing_csv = os.path.join(LOSSES_DIR, "generation_time.csv")
csv_rows   = []

# ── Seed loop ─────────────────────────────────────────────────────────────────
t_global = time.perf_counter()

for seed in SEEDS:
    print(f"{'─'*65}")
    print(f"  Seed {seed}", flush=True)
    print(f"{'─'*65}")

    out_dir = os.path.join(GEN_ROOT, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    t_seed = time.perf_counter()
    S_gen, meta = generate_paths(
        S_train,
        M_simu=M_SIMU,
        h=H,
        K=K,
        N_pi=N_PI,
        n_workers=N_WORK,
        seed=seed,
    )
    elapsed = time.perf_counter() - t_seed

    # ── Save generated paths ────────────────────────────────────────────────
    npy_path = os.path.join(out_dir, "generated_paths_8192x128.npy")
    np.save(npy_path, S_gen)
    print(f"  Saved: {npy_path}  shape={S_gen.shape}", flush=True)

    # ── Save metadata.json ──────────────────────────────────────────────────
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ── Save bandwidth note (no loss — kernel method) ───────────────────────
    bw_path = os.path.join(LOSSES_DIR, f"seed_{seed}_bandwidth.json")
    bw_info = dict(
        seed=seed, h=H, K=K, N_pi=N_PI, dt=DT,
        method="paper_default",
        note=(
            "h=0.4 from SBTS paper Appendix C Table 4 "
            "(Heston T=100, dt=1/252). "
            "SBTS is kernel-based — no training loss to log."
        ),
    )
    with open(bw_path, "w") as f:
        json.dump(bw_info, f, indent=2)

    csv_rows.append({
        "seed": seed, "n_samples": M_SIMU,
        "T": S_gen.shape[1], "h": H, "K": K, "N_pi": N_PI,
        "n_workers": N_WORK, "elapsed_sec": round(elapsed, 2),
        "elapsed_min": round(elapsed / 60, 2),
    })
    print(f"  Seed {seed} done in {elapsed/60:.1f} min", flush=True)
    print()

total = time.perf_counter() - t_global
print(f"{'='*65}")
print(f"All 5 seeds done in {total/60:.1f} min total")

# ── Write timing CSV ──────────────────────────────────────────────────────────
with open(timing_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    w.writeheader()
    w.writerows(csv_rows)
print(f"Saved: {timing_csv}")

# ── Final verification ────────────────────────────────────────────────────────
print("\nFinal verification:")
for seed in SEEDS:
    p   = os.path.join(GEN_ROOT, f"seed_{seed}", "generated_paths_8192x128.npy")
    arr = np.load(p)
    print(f"  seed {seed}: {arr.shape}  "
          f"price range [{arr.min():.1f}, {arr.max():.1f}]  "
          f"S[:,0] all==100: {np.allclose(arr[:,0], 100.0)}")

print("\nDone. Next step: run metrics/compute_all.py --method SBTS")
