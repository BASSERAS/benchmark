"""
TimesFM-1.0-200m as a FORECASTER REFERENCE on the path-shadowing task.

Role in the benchmark
---------------------
TimesFM is a decoder-only probabilistic *forecaster*, not an unconditional
generator.  Exactly like the Chronos-2 forecaster reference, we use it for what
it is built for: given the 64-step real prefix of each held-out Heston test
path, forecast the next 64 steps in a single shot.  This yields a "best
forecaster" reference score that the path-shadowing (PS-MC) rows of the true
*generators* are measured against — same prefix length (64), same horizons
(H=32, H=64), same CRPS / MAE / RMSE definitions and the same naive random-walk
baseline, so the numbers are directly comparable.

Two rows are produced:
    * zero_shot : released google/timesfm-1.0-200m-pytorch weights (one model).
    * finetune  : the 5 full-finetuned seed checkpoints in ../weights/ ,
                  reported as mean +/- std across the 5 seeds.

Ensemble construction
---------------------
`tfm.forecast()` returns, per future step, 9 predictive quantiles at levels
0.1 .. 0.9 (column 0 is the mean, columns 1..9 are the quantiles).  We turn that
into a K=77-member ensemble per (series, step) by piecewise-linear inverse-CDF
sampling over the 9 quantile levels, giving an ensemble of shape (N, K, H).
CRPS is then the exact energy score used by the generators' PS-MC harness
(path_shadowing.crps), so a forecaster row and a generator row are
apples-to-apples.

Forecasting is done in PRICE space (the same scale the fine-tune checkpoints
were trained on); TimesFM applies reversible instance-norm on the context
internally, and a single-shot 64-step forecast has no autoregressive feedback so
it is stable.

Usage
-----
  CUDA_VISIBLE_DEVICES=0 /home/tbasseras/timesfm-v1-venv/bin/python run_forecaster_ref.py
Outputs -> results/Heston/TimesFM/  (forecaster_summary.json, per-seed JSON,
plots/forecaster_*.png).
"""
import os
import sys
import json
import time
import numpy as np
import torch

PS_DIR = os.path.dirname(os.path.abspath(__file__))                # methods/TimesFM/path_shadowing
METHOD_DIR = os.path.dirname(PS_DIR)                               # methods/TimesFM
BENCH_ROOT = os.path.dirname(os.path.dirname(METHOD_DIR))          # benchmark/
WEIGHTS_DIR = os.path.join(METHOD_DIR, "weights")
OUT_DIR = os.path.join(BENCH_ROOT, "results", "Heston", "TimesFM")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

sys.path.insert(0, PS_DIR)
from path_shadowing import crps, evaluate_horizon, naive_baseline  # noqa: E402

TEST_DATA = os.path.join(BENCH_ROOT, "dataset", "Heston", "heston_S_test_8192x128.npy")
DEFAULT_MODEL_ID = "google/timesfm-1.0-200m-pytorch"
PREFIX_LEN = 64
HORIZON = 64                       # forecast steps 64..127
K = 77                             # ensemble members (matches PS-MC K)
HORIZONS = {"h32": (0, 32), "h64": (0, 64)}
FC_BATCH = 512
CRPS_BATCH_N = 512
Q_LEVELS = np.round(np.arange(1, 10) * 0.1, 1)   # 0.1 .. 0.9 (TimesFM native)


def model_cfg(model_id):
    """(num_layers, use_positional_embedding) per released checkpoint."""
    if "2.0-500m" in model_id:
        return 50, False      # 2.0-500m
    return 20, True           # 1.0-200m


# --------------------------------------------------------------------------- #
#  Vectorised inverse-CDF sampling over shared quantile levels                 #
# --------------------------------------------------------------------------- #
def sample_inverse_cdf(q_vals, q_levels, K, rng):
    """
    q_vals   : (M, Q) predictive quantile values, ascending along Q
    q_levels : (Q,)   quantile levels in (0,1), ascending, shared across rows
    Returns  : (M, K) piecewise-linear inverse-CDF samples.
    """
    M, Q = q_vals.shape
    u = rng.random((M, K)).astype(np.float64)                      # (M, K)
    idx = np.searchsorted(q_levels, u, side="right")               # (M, K)
    idx = np.clip(idx, 1, Q - 1)
    lo = idx - 1
    x0 = q_levels[lo]
    x1 = q_levels[idx]
    y0 = np.take_along_axis(q_vals, lo, axis=1)
    y1 = np.take_along_axis(q_vals, idx, axis=1)
    t = (u - x0) / (x1 - x0 + 1e-30)
    return y0 + t * (y1 - y0)                                      # (M, K)


# --------------------------------------------------------------------------- #
#  Forecast an ensemble for every series                                       #
# --------------------------------------------------------------------------- #
def forecast_ensemble(tfm, prefixes, horizon, K, rng, batch_size=512):
    """
    prefixes : (N, PREFIX_LEN) price-scale context.
    Returns  : ensemble (N, K, horizon) price-scale.
    """
    q_levels = np.asarray(Q_LEVELS, dtype=np.float64)              # (Q,)
    N = prefixes.shape[0]
    Q = q_levels.shape[0]
    q_all = np.empty((N, horizon, Q), dtype=np.float64)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        ctx = [prefixes[i] for i in range(s, e)]
        _, quant = tfm.forecast(ctx, freq=[0] * (e - s))          # quant (b, H, 10)
        quant = np.asarray(quant)[:, :horizon, 1:1 + Q]           # drop mean col -> (b, H, Q)
        q_all[s:e] = quant
        print(f"    forecast {e}/{N}", flush=True)

    # per-step inverse-CDF sampling: (N*horizon, Q) -> (N*horizon, K)
    flat = q_all.reshape(N * horizon, Q)
    flat = np.sort(flat, axis=1)                                   # enforce monotone quantiles
    samp = sample_inverse_cdf(flat, q_levels, K, rng)             # (N*horizon, K)
    ens = samp.reshape(N, horizon, K).transpose(0, 2, 1)         # (N, K, horizon)
    return ens.astype(np.float64)


def score_ensemble(ensemble, y_fut):
    """CRPS / MAE / RMSE (uniform weights) at h32 and h64 + per-step CRPS."""
    N, Kk, H = ensemble.shape
    w_unif = np.full((N, Kk), 1.0 / Kk, dtype=np.float32)
    res = {}
    for h_name, (h0, h1) in HORIZONS.items():
        m = evaluate_horizon(ensemble, y_fut, w_unif, w_unif,
                             h_start=h0, h_end=h1, batch_n=CRPS_BATCH_N)
        res[f"{h_name}_CRPS"] = m["CRPS_uniform"]
        res[f"{h_name}_MAE"] = m["MAE_uniform"]
        res[f"{h_name}_RMSE"] = m["RMSE_uniform"]
    crps_step = crps(ensemble, y_fut, weights=w_unif, batch_n=CRPS_BATCH_N).mean(axis=0)
    return res, crps_step


def load_tfm(device, model_id, num_layers, use_pos_emb, state_dict_path=None):
    import timesfm
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu" if device == "cuda" else "cpu",
            per_core_batch_size=32,
            horizon_len=HORIZON,
            num_layers=num_layers,
            context_len=512,
            use_positional_embedding=use_pos_emb,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_id),
    )
    if state_dict_path is not None:
        sd = torch.load(state_dict_path, map_location=device, weights_only=True)
        tfm._model.load_state_dict(sd)
        tfm._model.to(device).eval()
        print(f"  loaded fine-tuned weights: {os.path.basename(state_dict_path)}", flush=True)
    return tfm


# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_ID,
                    help="released TimesFM checkpoint (1.0-200m or 2.0-500m)")
    args = ap.parse_args()

    num_layers, use_pos_emb = model_cfg(args.model)
    tag = args.model.split("/")[-1]                                # timesfm-2.0-500m-pytorch
    is_default = "1.0-200m" in args.model
    # 1.0-200m keeps the committed flat weights/ dir and unsuffixed output filenames.
    weights_dir = WEIGHTS_DIR if is_default else os.path.join(WEIGHTS_DIR, tag)
    suffix = "" if is_default else f"_{tag}"

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    X_real = np.load(TEST_DATA).astype(np.float64)                 # (8192, 128)
    N, T = X_real.shape
    prefixes = X_real[:, :PREFIX_LEN].astype(np.float32)           # (N, 64)
    y_fut = X_real[:, PREFIX_LEN:]                                 # (N, 64)
    print(f"=== {tag} forecaster reference  device={dev_name} ===", flush=True)
    print(f"[data] test{X_real.shape}  prefix={PREFIX_LEN}  horizon={HORIZON}  "
          f"K={K}", flush=True)

    baseline = naive_baseline(X_real, prefix_len=PREFIX_LEN)
    print(f"[baseline RW] {baseline}", flush=True)

    summary = {"prefix_len": PREFIX_LEN, "horizon": HORIZON, "K": K,
               "model_id": args.model, "baseline": baseline}
    per_step = {}

    # -- zero-shot (single model) --------------------------------------------
    print("\n-- zero_shot ------------------------------------------------", flush=True)
    t0 = time.time()
    tfm = load_tfm(device, args.model, num_layers, use_pos_emb, state_dict_path=None)
    rng = np.random.default_rng(0)
    ens_zs = forecast_ensemble(tfm, prefixes, HORIZON, K, rng, batch_size=FC_BATCH)
    zs_res, zs_step = score_ensemble(ens_zs, y_fut)
    zs_res["time_sec"] = round(time.time() - t0, 1)
    summary["zero_shot"] = zs_res
    per_step["zero_shot"] = zs_step.tolist()
    print(f"[zero_shot] h32_CRPS={zs_res['h32_CRPS']:.4f}  "
          f"h64_CRPS={zs_res['h64_CRPS']:.4f}  ({zs_res['time_sec']}s)", flush=True)
    del tfm, ens_zs
    if device == "cuda":
        torch.cuda.empty_cache()

    # -- fine-tune (5 seeds) --------------------------------------------------
    print("\n-- finetune (5 seeds) ---------------------------------------", flush=True)
    ft_seeds = []
    ft_step_stack = []
    ens_ft0 = None
    for seed in range(5):
        t0 = time.time()
        sd_path = os.path.join(weights_dir, f"seed_{seed}_model.pt")
        tfm = load_tfm(device, args.model, num_layers, use_pos_emb, state_dict_path=sd_path)
        rng = np.random.default_rng(seed)
        ens = forecast_ensemble(tfm, prefixes, HORIZON, K, rng, batch_size=FC_BATCH)
        res, step = score_ensemble(ens, y_fut)
        res["seed"] = seed
        res["time_sec"] = round(time.time() - t0, 1)
        ft_seeds.append(res)
        ft_step_stack.append(step)
        if seed == 0:
            ens_ft0 = ens.copy()
        print(f"[finetune seed {seed}] h32_CRPS={res['h32_CRPS']:.4f}  "
              f"h64_CRPS={res['h64_CRPS']:.4f}  ({res['time_sec']}s)", flush=True)
        del tfm, ens
        if device == "cuda":
            torch.cuda.empty_cache()

    # aggregate fine-tune mean +/- std
    ft_agg = {"per_seed": ft_seeds}
    scalar_keys = [k for k in ft_seeds[0] if k not in ("seed", "time_sec")]
    for k in scalar_keys:
        vals = np.array([r[k] for r in ft_seeds])
        ft_agg[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
        print(f"  finetune {k}: {vals.mean():.4f} +/- {vals.std():.4f}", flush=True)
    summary["finetune"] = ft_agg
    per_step["finetune_mean"] = np.array(ft_step_stack).mean(axis=0).tolist()
    per_step["finetune_std"] = np.array(ft_step_stack).std(axis=0).tolist()

    # -- save JSON ------------------------------------------------------------
    summary["per_step_crps"] = per_step
    with open(os.path.join(OUT_DIR, f"forecaster_summary{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for r in ft_seeds:
        with open(os.path.join(OUT_DIR, f"forecaster_seed_{r['seed']}{suffix}.json"), "w") as f:
            json.dump(r, f, indent=2)
    print(f"\nSaved -> {OUT_DIR}/forecaster_summary{suffix}.json", flush=True)

    # -- Figure 1 -- example forecasts (fine-tune seed 0) --------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(0)
    idx = np.random.choice(N, 4, replace=False)
    t_axis = np.arange(T)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    fig.suptitle(
        f"{tag} forecaster reference -- 64-step prefix (blue) + K={K} "
        f"forecast members (red), fine-tune seed 0", fontsize=10)
    for ax, i in zip(axes, idx):
        ax.plot(t_axis[:PREFIX_LEN], X_real[i, :PREFIX_LEN],
                color="#2563EB", linewidth=1.8, label="Real prefix")
        ax.plot(t_axis[PREFIX_LEN:], X_real[i, PREFIX_LEN:],
                color="#2563EB", linewidth=1.8, linestyle="--", label="Real future")
        for k_idx in range(K):
            ax.plot(t_axis[PREFIX_LEN:], ens_ft0[i, k_idx],
                    color="#DC2626", alpha=0.06, linewidth=0.5)
        ax.plot(t_axis[PREFIX_LEN:], ens_ft0[i].mean(axis=0),
                color="#DC2626", linewidth=1.8, label="Forecast mean")
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
    out1 = os.path.join(PLOT_DIR, f"forecaster_example{suffix}.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}", flush=True)

    # -- Figure 2 -- CRPS per forecast step (zero-shot vs fine-tune) ---------
    h_axis = np.arange(1, HORIZON + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(h_axis, per_step["zero_shot"], color="#7C3AED", linewidth=1.8,
            label="Zero-shot")
    ft_mean = np.array(per_step["finetune_mean"])
    ft_std = np.array(per_step["finetune_std"])
    ax.plot(h_axis, ft_mean, color="#DC2626", linewidth=1.8,
            label="Fine-tune (5 seeds)")
    ax.fill_between(h_axis, ft_mean - ft_std, ft_mean + ft_std,
                    color="#DC2626", alpha=0.2, label="+/-1 std")
    ax.axhline(baseline["CRPS_h32"], color="gray", linewidth=1.0,
               linestyle="--", label=f"RW h=32: {baseline['CRPS_h32']:.2f}")
    ax.axhline(baseline["CRPS_h64"], color="gray", linewidth=1.0,
               linestyle=":", label=f"RW h=64: {baseline['CRPS_h64']:.2f}")
    ax.axvline(32, color="black", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Forecast horizon (steps ahead)", fontsize=9)
    ax.set_ylabel("Mean CRPS", fontsize=9)
    ax.set_title(f"{tag} forecaster reference -- CRPS per step (Heston test set)",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    out2 = os.path.join(PLOT_DIR, f"forecaster_crps_per_step{suffix}.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
