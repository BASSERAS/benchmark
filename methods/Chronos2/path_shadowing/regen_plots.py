#!/usr/bin/env python
"""Regenerate the two forecaster-reference plots from a completed eval.

The full eval (run_forecaster_ref.py) already wrote forecaster_summary.json but
its plots/ directory was moved away by a concurrent reorg before the figures were
saved.  This script rebuilds both figures WITHOUT re-running the whole 6-model
sweep:

  * Figure 2 (CRPS per step) is reconstructed purely from forecaster_summary.json.
  * Figure 1 (example forecasts) needs the fine-tune seed-0 ensemble, so it
    reloads only that one pipeline and forecasts 4 example paths (~15 s on GPU).

Reuses the exact helpers/constants from run_forecaster_ref.py so numbers match.
"""
import os
import json
import numpy as np

import run_forecaster_ref as R  # module-level: sets sys.path, constants, helpers

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    os.makedirs(R.PLOT_DIR, exist_ok=True)

    X_real = np.load(R.TEST_DATA).astype(np.float64)          # (N, T)
    N, T = X_real.shape
    prefixes = X_real[:, :R.PREFIX_LEN]

    with open(os.path.join(R.OUT_DIR, "forecaster_summary.json")) as f:
        summary = json.load(f)
    baseline = summary["baseline"]
    per_step = summary["per_step_crps"]

    # ── Figure 1 — example forecasts (fine-tune seed 0) ──────────────────────
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd_path = os.path.join(R.WEIGHTS_DIR, "seed_0_model.pt")
    pipe = R.load_pipeline(device, state_dict_path=sd_path)
    rng = np.random.default_rng(0)
    ens_ft0 = R.forecast_ensemble(pipe, prefixes, R.HORIZON, R.K, rng,
                                  batch_size=R.FC_BATCH)         # (N, K, HORIZON)

    np.random.seed(0)
    idx = np.random.choice(N, 4, replace=False)
    t_axis = np.arange(T)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    fig.suptitle(
        f"Chronos-2 forecaster reference — 64-step prefix (blue) + K={R.K} "
        f"forecast members (red), fine-tune seed 0", fontsize=10)
    for ax, i in zip(axes, idx):
        ax.plot(t_axis[:R.PREFIX_LEN], X_real[i, :R.PREFIX_LEN],
                color="#2563EB", linewidth=1.8, label="Real prefix")
        ax.plot(t_axis[R.PREFIX_LEN:], X_real[i, R.PREFIX_LEN:],
                color="#2563EB", linewidth=1.8, linestyle="--", label="Real future")
        for k_idx in range(R.K):
            ax.plot(t_axis[R.PREFIX_LEN:], ens_ft0[i, k_idx],
                    color="#DC2626", alpha=0.06, linewidth=0.5)
        ax.plot(t_axis[R.PREFIX_LEN:], ens_ft0[i].mean(axis=0),
                color="#DC2626", linewidth=1.8, label="Forecast mean")
        ax.axvline(R.PREFIX_LEN, color="black", linewidth=0.8, linestyle=":")
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
    out1 = os.path.join(R.PLOT_DIR, "forecaster_example.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}", flush=True)

    # ── Figure 2 — CRPS per forecast step (zero-shot vs fine-tune) ───────────
    h_axis = np.arange(1, R.HORIZON + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(h_axis, per_step["zero_shot"], color="#7C3AED", linewidth=1.8,
            label="Zero-shot")
    ft_mean = np.array(per_step["finetune_mean"])
    ft_std = np.array(per_step["finetune_std"])
    ax.plot(h_axis, ft_mean, color="#DC2626", linewidth=1.8,
            label="Fine-tune (5 seeds)")
    ax.fill_between(h_axis, ft_mean - ft_std, ft_mean + ft_std,
                    color="#DC2626", alpha=0.2, label="±1 std")
    ax.axhline(baseline["CRPS_h32"], color="gray", linewidth=1.0,
               linestyle="--", label=f"RW h=32: {baseline['CRPS_h32']:.2f}")
    ax.axhline(baseline["CRPS_h64"], color="gray", linewidth=1.0,
               linestyle=":", label=f"RW h=64: {baseline['CRPS_h64']:.2f}")
    ax.axvline(32, color="black", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Forecast horizon (steps ahead)", fontsize=9)
    ax.set_ylabel("Mean CRPS", fontsize=9)
    ax.set_title("Chronos-2 forecaster reference — CRPS per step (Heston test set)",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    out2 = os.path.join(R.PLOT_DIR, "forecaster_crps_per_step.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
