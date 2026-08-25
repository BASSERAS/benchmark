"""
SBTS full run, MULTI-ASSET — 5 seeds, 8192 paths each, length 252, d = 8 assets.

Trains on dataset/HestonMultiAsset/heston_ma_S_8192x252x8.npy only.
Seeds run SEQUENTIALLY (each seed uses 16 workers → ~82 min/seed).
Total expected: ~6.9 h for all 5 seeds.

Hardware (hard limits):
  max 16 physical cores, max 2 GPUs (GPU is used only for metrics, never here —
  SBTS is CPU-only)

Saved per seed (all inside results/HestonMultiAsset/SBTS/):
  generated_paths/seed_{i}/generated_paths_8192x252x8.npy
  generated_paths/seed_{i}/metadata.json
  losses/seed_{i}_bandwidth.json
  losses/generation_time.csv

There is no weights/ checkpoint: SBTS is non-parametric, the "model" IS the
training array plus (h, K, N_pi). See ../weights/README.md.

Run (must be detached — this is a multi-hour job):
    setsid taskset -c 0-15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
        /home/tbasseras/sbts-venv/bin/python run_all_multiasset.py \
        > /tmp/sbts_ma_run.log 2>&1 < /dev/null & disown
"""

import os, sys, time, json, csv
import numpy as np

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
CODE  = os.path.dirname(os.path.abspath(__file__))
SBTS  = os.path.dirname(CODE)          # results/HestonMultiAsset/SBTS
sys.path.insert(0, CODE)

from sbts_generate_multiasset import generate_paths, warmup_jit, DT, S0

# ── Hyper-parameters ─────────────────────────────────────────────────────────
# K and N_pi are the AUTHOR's values (A. Alouadi, confirmed 2026-07-27 for the
# d = 1, length-128 Heston benchmark) and are carried over unchanged.
#
# h is OURS. It is the one hyperparameter that cannot cross dimension: the SBTS
# kernel is radial, K_h(x) = (h² − ‖x‖²)²·1{‖x‖₂ < h}, so its support is a ball
# whose radius must scale with the typical distance between d-dimensional
# increments — and that distance grows like √d (median pairwise distance 0.0372
# at d = 1 vs 0.2643 at d = 8). The author's h = 0.05 is catastrophic here:
# 40.8 % volatility error, generated excess kurtosis 45.66 against a real 1.12,
# and a minimum price of 0.0 (outright path collapse).
#
# h = 0.31 was selected on the VALIDATION split (seed 3), never on test, as the
# largest bandwidth still satisfying all three hard constraints and therefore
# the least-memorising admissible choice. h = 0.35 is the cliff: volatility
# error jumps 1.20 → 6.45 %. Full sweep, criterion and caveats:
#   ../losses/bandwidth_selection.json
#   ../losses/selection_criterion.md
H      = 0.31   # bandwidth — RECALIBRATED FOR d = 8, not the author's 0.05
K      = 20     # Markovian order (author)
N_PI   = 50     # Euler substeps  (author)
M_SIMU = 8192   # paths to generate per seed

# Override via env: SBTS_NWORK=8 SBTS_SEEDS=1,2 python run_all_multiasset.py
N_WORK = int(os.environ.get("SBTS_NWORK", "16"))
SEEDS  = [int(s) for s in os.environ.get("SBTS_SEEDS", "0,1,2,3,4").split(",")]

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(BENCH, "dataset/HestonMultiAsset/heston_ma_S_8192x252x8.npy")
GEN_ROOT   = os.path.join(SBTS, "generated_paths")
LOSSES_DIR = os.path.join(SBTS, "losses")

os.makedirs(LOSSES_DIR, exist_ok=True)

# ── Load training data once ──────────────────────────────────────────────────
print("=" * 72)
print("SBTS Multi-Asset Full Run  —  5 seeds x 8192 paths x 252 steps x 8 assets")
print("=" * 72)
print(f"h={H}  K={K}  N_pi={N_PI}  dt={DT:.6f}  workers={N_WORK}")
print()

S_train = np.load(DATA_PATH)
D = S_train.shape[2]
print(f"Training data: {S_train.shape}  "
      f"prices in [{S_train.min():.1f}, {S_train.max():.1f}]")
print()

# ── Warm up Numba JIT (once, in the parent process) ──────────────────────────
warmup_jit(D)
print()

timing_csv = os.path.join(LOSSES_DIR, "generation_time.csv")
csv_rows   = []

t_global = time.perf_counter()

for seed in SEEDS:
    print("─" * 72)
    print(f"  Seed {seed}", flush=True)
    print("─" * 72)

    out_dir = os.path.join(GEN_ROOT, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    t_seed = time.perf_counter()
    S_gen, meta = generate_paths(
        S_train, M_simu=M_SIMU, h=H, K=K, N_pi=N_PI,
        n_workers=N_WORK, seed=seed,
    )
    elapsed = time.perf_counter() - t_seed

    # ── Save generated paths ────────────────────────────────────────────────
    tag = f"{M_SIMU}x{S_gen.shape[1]}x{D}"
    npy_path = os.path.join(out_dir, f"generated_paths_{tag}.npy")
    np.save(npy_path, S_gen)
    print(f"  Saved: {npy_path}  shape={S_gen.shape}", flush=True)

    # ── Save metadata.json ──────────────────────────────────────────────────
    meta = dict(meta)
    meta.update({
        "seed": seed, "h": H, "K": K, "N_pi": N_PI, "dt": DT, "S0": S0,
        "d": D, "shape": list(S_gen.shape), "dtype": str(S_gen.dtype),
        "S_min": float(S_gen.min()), "S_max": float(S_gen.max()),
        "S0_exact": bool(np.all(S_gen[:, 0, :] == S0)),
        "elapsed_sec": round(elapsed, 1),
    })
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Save bandwidth note (no loss — kernel method) ───────────────────────
    bw_info = dict(
        seed=seed, h=H, K=K, N_pi=N_PI, dt=DT,
        method="K and N_pi author-specified; h recalibrated for d = 8",
        h_selection_split="validation (seed 3), never test",
        note=(
            "K=20, N_pi=50 confirmed by A. Alouadi (SBTS author) 2026-07-27 for "
            "the d=1 length-128 Heston benchmark and carried over unchanged. "
            "h=0.05 is a d=1 value and CANNOT be carried: the radial kernel's "
            "support radius must scale with the typical increment distance, "
            "which grows like sqrt(d). At d=8, h=0.05 gives 40.8% volatility "
            "error, excess kurtosis 45.66 vs a real 1.12, and minimum price 0.0 "
            "(path collapse). h=0.31 was selected on the VALIDATION split as the "
            "largest bandwidth meeting all hard constraints, hence the least "
            "memorising admissible choice. Even so its NN ratio is 0.215: "
            "generated paths sit ~4.6x closer to the training set than held-out "
            "real data does. See losses/bandwidth_selection.json and "
            "losses/selection_criterion.md. SBTS is kernel-based - there is no "
            "training loss to log and no weights to serialise."
        ),
    )
    with open(os.path.join(LOSSES_DIR, f"seed_{seed}_bandwidth.json"), "w") as f:
        json.dump(bw_info, f, indent=2)

    csv_rows.append({
        "seed": seed, "n_samples": M_SIMU, "T": S_gen.shape[1], "d": D,
        "h": H, "K": K, "N_pi": N_PI, "n_workers": N_WORK,
        "elapsed_sec": round(elapsed, 2), "elapsed_min": round(elapsed / 60, 2),
    })
    print(f"  Seed {seed} done in {elapsed/60:.1f} min", flush=True)
    print()

    # rewrite the timing CSV after every seed so a crash still leaves a record
    with open(timing_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)

total = time.perf_counter() - t_global
print("=" * 72)
print(f"All {len(SEEDS)} seeds done in {total/60:.1f} min total")
print(f"Saved: {timing_csv}")

# ── Final verification ───────────────────────────────────────────────────────
print("\nFinal verification:")
ok = True
for seed in SEEDS:
    d = os.path.join(GEN_ROOT, f"seed_{seed}")
    hits = [f for f in os.listdir(d) if f.startswith("generated_paths") and f.endswith(".npy")]
    arr = np.load(os.path.join(d, hits[0]))
    good = (arr.shape == (M_SIMU, S_train.shape[1], D)
            and np.all(np.isfinite(arr)) and np.all(arr > 0.0)
            and np.all(arr[:, 0, :] == S0))
    ok &= good
    print(f"  seed {seed}: {arr.shape}  price [{arr.min():.1f}, {arr.max():.1f}]  "
          f"S[:,0]==100: {np.all(arr[:, 0, :] == S0)}  finite&positive: "
          f"{np.all(np.isfinite(arr)) and np.all(arr > 0.0)}  -> {'OK' if good else 'FAIL'}")

print("\nAll seeds OK." if ok else "\nSOME SEEDS FAILED VERIFICATION.")
print("Next: metrics/compute_all_multiasset.py --method SBTS")
