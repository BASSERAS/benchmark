"""
Path Shadowing Monte-Carlo — full PDF protocol (arXiv:2308.01486) for
CSDI + SBTS log-return preprocessing.

Mirrors LS4/path_shadowing/path_shadowing_mc.py exactly (same protocol from
GUIDELINE.md §9):

    bank size = 1,000,000 generated paths (per seed)
    K         = 256 nearest neighbours
    prefix    = 64   horizon = 32 (also reports 64)
    queries   = 512  held-out real paths from the *ps* split (seed 3),
                strictly independent of train and of the bank
    alignment = additive-endpoint in log == multiplicative price anchoring
    score     = CRPS (energy score), lower better

The PS math (65D murex embedding, retrieval, Gaussian bandwidth, CRPS) is
REUSED verbatim from methods/LS4/path_shadowing/path_shadowing.py — the SAME
method-agnostic core used by the LS4 experiment (NOT the murex K=77 eval under
methods/CSDI). This file only swaps the 1M-bank generator to the CSDI DDPM
(unconditional diffusion + SBTS inverse). Nothing under methods/ is modified.

Bank build:
  load results/.../CSDI/weights/seed_{i}_model.pt  (CSDI has no EMA),
  unconditional-sample bank_size paths in R~ space, invert with the SAME frozen
  (sigma, x_mu, x_sd) from the checkpoint, cum-exp anchored at S0=100.

Usage:
  CUDA_VISIBLE_DEVICES=0 python path_shadowing_mc.py --seed 0 --bank_size 1000000
  CUDA_VISIBLE_DEVICES=0 python path_shadowing_mc.py --seeds 0,1,2,3,4 --bank_size 1000000
  # --keep_bank_seed picks which 1M npy to persist (the deliverable); default seed 0
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))                 # .../CSDI/path_shadowing
CSDI_DIR = os.path.dirname(HERE)                                  # .../CSDI
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CSDI_DIR))))

# untouched reference CSDI model (subclass + config)
REF_TRAIN = os.path.join(BENCH_ROOT, "methods", "CSDI", "code")
sys.path.insert(0, REF_TRAIN)
from train_heston import CSDI_Heston, BASE_CONFIG                 # noqa: E402

# reuse the validated, METHOD-AGNOSTIC PS-MC core (embedding / retrieve / CRPS)
ORIG_PS = os.path.join(BENCH_ROOT, "methods", "LS4", "path_shadowing")
sys.path.insert(0, ORIG_PS)
from path_shadowing import (                                      # noqa: E402
    ps_mc_retrieve, uniform_weights, gaussian_weights,
    crps, evaluate_horizon, naive_baseline,
)

DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns")
PS_QUERY = os.path.join(DATA_DIR, "heston_S_ps_512x128.npy")     # 512 independent real paths

OUT_DIR = os.path.join(CSDI_DIR, "path_shadowing")
BANK_DIR = os.path.join(OUT_DIR, "bank")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

DT = 1.0 / 250.0
S0 = 100.0
SEQ_LEN = 128
PREFIX_LEN = 64
K = 256
BATCH_N = 32                                                     # CRPS N-batch (caps K^2 RAM)


def load_gen_model(seed, device):
    """Rebuild the CSDI DDPM from the saved checkpoint (no EMA in CSDI recipe)."""
    ckpt = torch.load(
        os.path.join(CSDI_DIR, "weights", f"seed_{seed}_model.pt"),
        map_location=device,
    )
    cfg = json.loads(json.dumps(BASE_CONFIG))                     # deep copy of paper config
    model = CSDI_Heston(cfg, device, target_dim=1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, float(ckpt["sbts_sigma"]), float(ckpt["x_mu"]), float(ckpt["x_sd"])


def build_bank(seed, bank_size, gen_batch, device):
    """Unconditional-sample `bank_size` CSDI paths, invert with frozen (sigma,x_mu,x_sd)."""
    gen_model, sigma, x_mu, x_sd = load_gen_model(seed, device)
    torch.manual_seed(1000 + seed)          # bank RNG distinct from train RNG
    np.random.seed(1000 + seed)
    t0 = time.time()
    chunks = []
    done = 0
    with torch.no_grad():
        while done < bank_size:
            b = min(gen_batch, bank_size - done)
            # CSDI generate returns a (b, L) 2D numpy array in unit-var (R~) scale
            g = gen_model.generate(b, SEQ_LEN, gen_batch=gen_batch)
            chunks.append(np.asarray(g, dtype=np.float32))
            done += b
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            eta = (bank_size - done) / max(rate, 1e-9)
            print(f"  [seed {seed} bank] {done}/{bank_size} "
                  f"{rate:.0f} paths/s ETA {eta:.0f}s", flush=True)
    gen = np.concatenate(chunks, axis=0).astype(np.float64)       # (bank, L) unit-var scale
    # inverse: destandardize -> drop dummy col -> undo SBTS scale -> cum-exp @ S0
    X_sbts = gen * x_sd + x_mu                                    # (bank, 128) back to R~ space
    R_tilde = X_sbts[:, 1:]
    R_gen = R_tilde * sigma / np.sqrt(DT)
    S = np.empty((bank_size, SEQ_LEN), dtype=np.float64)
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(np.cumsum(R_gen, axis=1))
    build_sec = time.time() - t0
    return S.astype(np.float32), build_sec, sigma


def eval_bank(X_query, X_bank):
    """Full-protocol PS-MC: retrieve K=256, CRPS at H=32 and H=64."""
    N = X_query.shape[0]
    ensemble, distances, _, real_norms = ps_mc_retrieve(
        X_query, X_bank, prefix_len=PREFIX_LEN, K=K
    )
    y_fut = X_query[:, PREFIX_LEN:]
    w_unif = uniform_weights(N, K)
    median_dist = float(np.median(distances))
    median_norm = float(np.median(real_norms)) + 1e-30
    eta_tilde = median_dist / median_norm
    w_gauss, eta_val = gaussian_weights(distances, real_emb_norms=real_norms,
                                        eta_tilde=eta_tilde)
    res = {"eta": float(eta_val), "K": K, "prefix": PREFIX_LEN,
           "bank_size": int(X_bank.shape[0]), "n_query": int(N)}
    for h_name, (h0, h1) in {"h32": (0, 32), "h64": (0, 64)}.items():
        m = evaluate_horizon(ensemble, y_fut, w_unif, w_gauss,
                             h_start=h0, h_end=h1, batch_n=BATCH_N)
        for mk, mv in m.items():
            res[f"{h_name}_{mk}"] = mv
    # per-step CRPS (uniform + gaussian) for the plot
    crps_u = crps(ensemble, y_fut, weights=w_unif, batch_n=BATCH_N).mean(axis=0)
    crps_g = crps(ensemble, y_fut, weights=w_gauss, batch_n=BATCH_N).mean(axis=0)
    return res, ensemble, crps_u, crps_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=str, default=None, help="comma list, e.g. 0,1,2,3,4")
    ap.add_argument("--bank_size", type=int, default=1_000_000)
    ap.add_argument("--gen_batch", type=int, default=8192)
    ap.add_argument("--keep_bank_seed", type=int, default=0,
                    help="which seed's 1M bank npy to persist (the deliverable)")
    ap.add_argument("--keep_all_banks", action="store_true")
    args = ap.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [args.seed if args.seed is not None else 0])

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(BANK_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_query = np.load(PS_QUERY).astype(np.float64)               # (512,128)
    print(f"[ps] query {X_query.shape} from {os.path.basename(PS_QUERY)}  "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          flush=True)
    baseline = naive_baseline(X_query, prefix_len=PREFIX_LEN)
    print(f"[ps] random-walk baseline: {baseline}", flush=True)

    all_res = []
    keep_ens0 = None
    per_step = []
    for seed in seeds:
        Xb, build_sec, sigma = build_bank(seed, args.bank_size, args.gen_batch, device)
        print(f"[seed {seed}] bank {Xb.shape} built in {build_sec:.0f}s  "
              f"price[min={Xb.min():.2f},max={Xb.max():.2f}] "
              f"nan={not np.isfinite(Xb).all()} sigma={sigma:.8f}", flush=True)
        if args.keep_all_banks or seed == args.keep_bank_seed:
            bank_path = os.path.join(
                BANK_DIR, f"generated_bank_seed{seed}_{args.bank_size}x{SEQ_LEN}.npy")
            np.save(bank_path, Xb)
            print(f"[seed {seed}] saved bank -> {bank_path} "
                  f"({os.path.getsize(bank_path)/1e9:.2f} GB)", flush=True)

        t0 = time.time()
        res, ensemble, crps_u, crps_g = eval_bank(X_query, Xb)
        res["seed"] = seed
        res["build_sec"] = round(build_sec, 1)
        res["eval_sec"] = round(time.time() - t0, 1)
        all_res.append(res)
        per_step.append((crps_u, crps_g))
        if seed == args.keep_bank_seed:
            keep_ens0 = ensemble
        with open(os.path.join(OUT_DIR, f"ps_results_seed{seed}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"[seed {seed}] CRPS h32 uniform={res['h32_CRPS_uniform']:.4f} "
              f"gaussian={res['h32_CRPS_gaussian']:.4f}  "
              f"h64 uniform={res['h64_CRPS_uniform']:.4f} "
              f"gaussian={res['h64_CRPS_gaussian']:.4f}  eval={res['eval_sec']}s", flush=True)
        del Xb
        if seed != args.keep_bank_seed and not args.keep_all_banks:
            del ensemble

    # summary mean+/-std
    summary = {"baseline": baseline, "protocol": {
        "bank_size": args.bank_size, "K": K, "prefix": PREFIX_LEN,
        "horizons": [32, 64], "n_query": int(X_query.shape[0]),
        "query_file": os.path.basename(PS_QUERY)}}
    keys = [k for k in all_res[0]
            if k not in ("seed", "eta", "build_sec", "eval_sec", "K",
                         "prefix", "bank_size", "n_query")]
    for k in keys:
        v = np.array([r[k] for r in all_res], dtype=np.float64)
        summary[k] = {"mean": float(v.mean()), "std": float(v.std())}
    summary["per_seed"] = all_res
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n== PS-MC summary (mean +/- std across seeds) ==", flush=True)
    for k in keys:
        print(f"  {k}: {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}", flush=True)

    # ── plots ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: example shadowed forecast fans (keep_bank seed)
    if keep_ens0 is not None:
        np.random.seed(0)
        idx = np.random.choice(X_query.shape[0], 4, replace=False)
        t_axis = np.arange(SEQ_LEN)
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
        fig.suptitle(f"PS-MC shadowed forecasts — CSDI+logret, bank={args.bank_size:,}, K={K}",
                     fontsize=10)
        for ax, i in zip(axes, idx):
            ax.plot(t_axis[:PREFIX_LEN], X_query[i, :PREFIX_LEN],
                    color="#2563EB", lw=1.8, label="Real prefix")
            ax.plot(t_axis[PREFIX_LEN:], X_query[i, PREFIX_LEN:],
                    color="#2563EB", lw=1.8, ls="--", label="Real future")
            step = max(1, K // 64)
            for kk in range(0, K, step):
                ax.plot(t_axis[PREFIX_LEN:], keep_ens0[i, kk],
                        color="#DC2626", alpha=0.05, lw=0.5)
            ax.plot(t_axis[PREFIX_LEN:], keep_ens0[i].mean(axis=0),
                    color="#DC2626", lw=1.8, label="PS-MC mean")
            ax.axvline(PREFIX_LEN, color="black", lw=0.8, ls=":")
            ax.set_xlabel("Time step", fontsize=8); ax.tick_params(labelsize=7)
            ax.set_title(f"Query {i}", fontsize=8)
        axes[0].set_ylabel("Price", fontsize=8)
        seen, h2, l2 = set(), [], []
        for h, l in zip(*axes[0].get_legend_handles_labels()):
            if l not in seen:
                seen.add(l); h2.append(h); l2.append(l)
        fig.legend(h2, l2, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.08))
        plt.tight_layout()
        p1 = os.path.join(PLOT_DIR, "ps_mc_example.png")
        plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved: {p1}", flush=True)

    # Fig 2: CRPS per forecast step (mean +/- std across seeds)
    arr_u = np.array([p[0] for p in per_step])   # (S, 64)
    arr_g = np.array([p[1] for p in per_step])
    h_axis = np.arange(1, SEQ_LEN - PREFIX_LEN + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"CRPS per forecast step — CSDI+logret on Heston ({len(seeds)} seed(s))",
                 fontsize=11)
    for ax, (label, arr) in zip(axes, [("Uniform", arr_u), ("Gaussian", arr_g)]):
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(h_axis, mean, color="#DC2626", lw=1.8, label="Mean CRPS")
        ax.fill_between(h_axis, mean - std, mean + std, color="#DC2626", alpha=0.2, label="±1 std")
        ax.axhline(baseline["CRPS_h32"], color="gray", lw=1.0, ls="--",
                   label=f"RW h32={baseline['CRPS_h32']:.2f}")
        ax.axvline(32, color="black", lw=0.7, ls=":")
        ax.set_xlabel("Forecast horizon (steps)", fontsize=9)
        ax.set_ylabel("Mean CRPS", fontsize=9); ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7); ax.tick_params(labelsize=8)
    plt.tight_layout()
    p2 = os.path.join(PLOT_DIR, "crps_per_step.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {p2}\nDone.", flush=True)


if __name__ == "__main__":
    main()
