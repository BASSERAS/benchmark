"""CSDI — Heston loss convergence plot (5 seeds overlaid).

CSDI trains a conditional score-based diffusion model; in our unconditional
Heston adaptation the loss is the standard DDPM noise-prediction MSE
``E_t ||eps - eps_theta(x_t, t)||^2`` averaged over the batch, logged once per
optimiser step (see ../train_heston.py). This mirrors
methods/FourierFlow/code/plot_losses.py, but CSDI logs per-STEP (columns
``step,loss``) rather than per-epoch, so we plot against step and overlay a
trailing moving average to expose the trend under per-step noise.

Reads   ../losses/seed_{0..4}_losses.csv   (columns: step,loss)
Writes  ../losses/loss_convergence.png

Usage:
    /home/tbasseras/gpu-venv/bin/python methods/CSDI/code/plot_losses.py
"""
from __future__ import annotations
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)                      # methods/CSDI
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]


def _read(seed: int):
    path = os.path.join(LOSSES_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None, None
    st, ls = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            st.append(int(r["step"]))
            ls.append(float(r["loss"]))
    return st, ls


def _moving_avg(vals, w=200):
    """Trailing moving average to expose the trend under step noise."""
    csum = [0.0]
    for v in vals:
        csum.append(csum[-1] + v)
    out = []
    for i in range(len(vals)):
        lo = max(0, i - w + 1)
        out.append((csum[i + 1] - csum[lo]) / (i + 1 - lo))
    return out


def main():
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 6))
    last_vals = []
    ZOOM_FRAC = 0.5                                    # zoom into 2nd half of training
    for seed in range(5):
        st, ls = _read(seed)
        if st is None:
            continue
        trend = _moving_avg(ls, w=200)
        ax_full.plot(st, ls, color=COLORS[seed], alpha=0.15, linewidth=0.5)
        ax_full.plot(st, trend, color=COLORS[seed], alpha=0.95,
                     linewidth=1.3, label=f"seed {seed}")
        zi = [i for i, s in enumerate(st) if s >= ZOOM_FRAC * st[-1]]
        if zi:
            ax_zoom.plot([st[i] for i in zi], [trend[i] for i in zi],
                         color=COLORS[seed], alpha=0.95, linewidth=1.3,
                         label=f"seed {seed} (last≈{trend[-1]:.4f})")
        last_vals.append(trend[-1])

    ax_full.set_title("Full DDPM noise-MSE trajectory (all steps)", fontsize=12)
    ax_zoom.set_title("Plateau zoom (2nd half of training)", fontsize=12)
    for ax in (ax_full, ax_zoom):
        ax.set_xlabel("Optimiser step")
        ax.set_ylabel(r"Noise-prediction MSE  $\mathbb{E}_t\,\|\epsilon-\epsilon_\theta(x_t,t)\|^2$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    band = (f"loss_trend_last ∈ [{min(last_vals):.4f}, {max(last_vals):.4f}]"
            if last_vals else "")
    plt.suptitle(
        "CSDI Loss Convergence — Heston seq_len=128  "
        f"(z-score norm; num_steps=50 quad β-schedule; no NaN; {band})",
        fontsize=13)
    plt.tight_layout()
    out = os.path.join(LOSSES_DIR, "loss_convergence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}  ({band})")


if __name__ == "__main__":
    main()
