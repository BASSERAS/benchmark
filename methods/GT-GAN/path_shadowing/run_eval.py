"""
Run Path Shadowing Monte-Carlo evaluation for all 5 GT-GAN seeds in parallel.

Seeds run concurrently via multiprocessing.Pool(5).
Each worker uses OMP_NUM_THREADS=32 for numpy (set before fork).

Saves per-seed JSON + summary JSON to:
  results/Heston/GT-GAN/path_shadowing/

Generates figures:
  results/Heston/GT-GAN/path_shadowing/plots/ps_mc_example.png
  results/Heston/GT-GAN/path_shadowing/plots/crps_per_step.png
"""

import os, sys, json, time
import numpy as np
from multiprocessing import Pool

# ── Set thread counts before any import that reads them ──────────────────────
os.environ.setdefault("OMP_NUM_THREADS",      "32")
os.environ.setdefault("MKL_NUM_THREADS",      "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "32")

BENCH      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PS_DIR     = os.path.dirname(__file__)
OUT_DIR    = os.path.join(BENCH, "results/Heston/GT-GAN/path_shadowing")
PLOT_DIR   = os.path.join(OUT_DIR, "plots")
PREFIX_LEN = 64
K          = 77
HORIZONS   = {"h32": (0, 32), "h64": (0, 64)}
BATCH_N    = 512     # N-batch size for vectorised CRPS


# ── Worker function (must be top-level for multiprocessing) ──────────────────

def eval_seed(seed):
    """Process one seed: retrieve + CRPS/MAE/RMSE at two horizons."""
    sys.path.insert(0, PS_DIR)
    from path_shadowing import (
        ps_mc_retrieve, uniform_weights, gaussian_weights,
        crps, evaluate_horizon,
    )

    t0 = time.time()

    X_real = np.load(os.path.join(BENCH, "dataset/Heston/heston_S_test_8192x128.npy"))
    X_fake = np.load(os.path.join(
        BENCH, f"methods/GT-GAN/generated_paths/seed_{seed}"
               f"/generated_paths_8192x128.npy"
    ))

    N, T   = X_real.shape
    y_fut  = X_real[:, PREFIX_LEN:]   # (N, 64)

    ensemble, distances, _, real_emb_norms = ps_mc_retrieve(
        X_real, X_fake, prefix_len=PREFIX_LEN, K=K
    )
    w_unif           = uniform_weights(N, K)
    # Adaptive η̃: calibrate from data so η_i = η̃_adapt × ‖h_i‖ matches the
    # median nearest-neighbour distance.  The paper's η̃=0.075 is calibrated on
    # S&P data; our Heston embedding norms differ, so we re-derive η̃ here.
    median_dist = float(np.median(distances))
    median_norm = float(np.median(real_emb_norms)) + 1e-30
    eta_tilde_adapt = median_dist / median_norm
    w_gauss, eta_val = gaussian_weights(
        distances, real_emb_norms=real_emb_norms, eta_tilde=eta_tilde_adapt
    )

    seed_res = {"seed": seed, "eta": float(eta_val)}
    for h_name, (h_start, h_end) in HORIZONS.items():
        metrics = evaluate_horizon(
            ensemble, y_fut, w_unif, w_gauss,
            h_start=h_start, h_end=h_end, batch_n=BATCH_N
        )
        for mk, mv in metrics.items():
            seed_res[f"{h_name}_{mk}"] = mv

    # Per-step CRPS for plot
    crps_u = crps(ensemble, y_fut, weights=w_unif,  batch_n=BATCH_N).mean(axis=0)
    crps_g = crps(ensemble, y_fut, weights=w_gauss, batch_n=BATCH_N).mean(axis=0)

    elapsed = time.time() - t0
    print(f"  [seed {seed}] done in {elapsed:.1f}s  "
          f"CRPS-h32 uniform={seed_res['h32_CRPS_uniform']:.4f}  "
          f"gaussian={seed_res['h32_CRPS_gaussian']:.4f}",
          flush=True)

    return seed_res, crps_u.tolist(), crps_g.tolist()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR,  exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    sys.path.insert(0, PS_DIR)
    from path_shadowing import naive_baseline

    X_real = np.load(os.path.join(BENCH, "dataset/Heston/heston_S_test_8192x128.npy"))
    print(f"Real data: {X_real.shape}  prefix={PREFIX_LEN}  K={K}")

    baseline = naive_baseline(X_real, prefix_len=PREFIX_LEN)
    print(f"Naive baseline: {baseline}", flush=True)

    # ── Run 5 seeds in parallel ───────────────────────────────────────────────
    print(f"\nLaunching 5 seeds in parallel (OMP_NUM_THREADS=32 each)...", flush=True)
    t_global = time.time()

    with Pool(5) as pool:
        raw = pool.map(eval_seed, range(5))

    total_elapsed = time.time() - t_global
    print(f"\nAll seeds done in {total_elapsed:.1f}s total", flush=True)

    all_results          = [r[0] for r in raw]
    crps_per_step_u      = np.array([r[1] for r in raw])   # (5, 64)
    crps_per_step_g      = np.array([r[2] for r in raw])   # (5, 64)

    # ── Save per-seed JSON ────────────────────────────────────────────────────
    for res in all_results:
        with open(os.path.join(OUT_DIR, f"seed_{res['seed']}_results.json"), "w") as f:
            json.dump(res, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n══ Summary (mean ± std across 5 seeds) ══════════════════════════")
    scalar_keys = [k for k in all_results[0] if k not in ("seed", "eta")]
    summary = {"baseline": baseline}
    for k in scalar_keys:
        vals = np.array([r[k] for r in all_results])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
        print(f"  {k}: {vals.mean():.4f} ± {vals.std():.4f}")

    summary["per_seed"] = all_results
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {OUT_DIR}/summary.json", flush=True)

    # ── Figure 1 — example path with ensemble (seed 0) ───────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from path_shadowing import ps_mc_retrieve, uniform_weights

    N, T = X_real.shape
    np.random.seed(0)
    X_fake0 = np.load(os.path.join(
        BENCH, "methods/GT-GAN/generated_paths/seed_0/generated_paths_8192x128.npy"
    ))
    ens0, _, _, _ = ps_mc_retrieve(X_real, X_fake0, prefix_len=PREFIX_LEN, K=K)
    t_axis     = np.arange(T)
    idx        = np.random.choice(N, 4, replace=False)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=False)
    fig.suptitle(
        f"Path Shadowing MC — prefix (blue) + K={K} futures (red), seed 0",
        fontsize=10
    )
    for ax, i in zip(axes, idx):
        ax.plot(t_axis[:PREFIX_LEN], X_real[i, :PREFIX_LEN],
                color="#2563EB", linewidth=1.8, label="Real prefix")
        ax.plot(t_axis[PREFIX_LEN:], X_real[i, PREFIX_LEN:],
                color="#2563EB", linewidth=1.8, linestyle="--", label="Real future")
        for k_idx in range(K):
            ax.plot(t_axis[PREFIX_LEN:], ens0[i, k_idx],
                    color="#DC2626", alpha=0.06, linewidth=0.5)
        ax.plot(t_axis[PREFIX_LEN:], ens0[i].mean(axis=0),
                color="#DC2626", linewidth=1.8, label="PS-MC mean")
        ax.axvline(PREFIX_LEN, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Time step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"Path {i}", fontsize=8)
    axes[0].set_ylabel("Price", fontsize=8)
    seen, h2, l2 = set(), [], []
    for h, l in zip(*axes[0].get_legend_handles_labels()):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    out1 = os.path.join(PLOT_DIR, "ps_mc_example.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")

    # ── Figure 2 — CRPS per forecast step ────────────────────────────────────
    h_axis = np.arange(1, 65)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CRPS per forecast step — GT-GAN on Heston (5 seeds)", fontsize=11)

    for ax, (label, arr) in zip(
        axes,
        [("Uniform (flat average)",        crps_per_step_u),
         ("Gaussian (distance-weighted)",  crps_per_step_g)]
    ):
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(h_axis, mean, color="#DC2626", linewidth=1.8,
                label="Mean CRPS (5 seeds)")
        ax.fill_between(h_axis, mean - std, mean + std,
                        color="#DC2626", alpha=0.2, label="±1 std")
        ax.axhline(baseline["CRPS_h32"], color="gray", linewidth=1.0,
                   linestyle="--", label=f"RW h=32: {baseline['CRPS_h32']:.2f}")
        ax.axhline(baseline["CRPS_h64"], color="gray", linewidth=1.0,
                   linestyle=":",  label=f"RW h=64: {baseline['CRPS_h64']:.2f}")
        ax.axvline(32, color="black", linewidth=0.7, linestyle=":")
        ax.set_xlabel("Forecast horizon (steps)", fontsize=9)
        ax.set_ylabel("Mean CRPS", fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    out2 = os.path.join(PLOT_DIR, "crps_per_step.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    print("\nDone.")
