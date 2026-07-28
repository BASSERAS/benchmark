"""TimeMoDE — Heston loss convergence plot (5 seeds overlaid, one panel per term).

TimeMoDE optimises a *composite* objective (see ../train_seed.py):

    loss_total = simple + 0.01 * proto + 0.01 * aux

  - ``simple`` : standard DDPM epsilon-prediction MSE
                 ``E_t || eps - eps_theta(x_t, t) ||^2`` (T=250, learn_sigma=False).
                 This is the term that actually drives sample quality; it should
                 converge toward ~0.006-0.008.
  - ``proto``  : prototype orthonormality penalty ``|| P P^T - I ||_F^2`` on the
                 K=8 domain-expert prototype bank. Collapses quickly toward 0 as
                 the prototypes de-correlate.
  - ``aux``    : Switch-transformer top-2 load-balancing auxiliary loss on the
                 MoDE router. Plateaus around ~12 (near the uniform-routing
                 floor for K=8, top-2) once the router stops re-allocating.

Each term is logged once per optimiser step. We overlay all 5 training seeds in
each panel and draw a trailing moving average (w=200) so the per-step noise does
not hide the trend. Mirrors methods/CSDI/code/plot_losses.py (same COLORS,
same _moving_avg helper) but uses a 2x2 grid because TimeMoDE logs four terms.

Reads   ../losses/seed_{0..4}_losses.csv   (columns: step,loss_total,simple,proto,aux)
Writes  ../losses/loss_convergence.png

Usage:
    /home/tbasseras/gpu-venv/bin/python methods/TimeMoDE/code/plot_losses.py
"""
from __future__ import annotations
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)                      # methods/TimeMoDE
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

# (csv column, panel title, y-axis label)
COMPONENTS = [
    ("loss_total", "Total objective  (simple + 0.01·proto + 0.01·aux)",
     r"$\mathcal{L}_{\mathrm{total}}$"),
    ("simple", "DDPM noise-prediction MSE  (drives sample quality)",
     r"$\mathbb{E}_t\,\|\epsilon-\epsilon_\theta(x_t,t)\|^2$"),
    ("proto", "Prototype orthonormality penalty",
     r"$\|P P^\top - I\|_F^2$"),
    ("aux", "MoDE Switch top-2 load-balancing aux loss",
     r"$\mathcal{L}_{\mathrm{aux}}$"),
]


def _read(seed: int):
    path = os.path.join(LOSSES_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None
    rows = {"step": [], "loss_total": [], "simple": [], "proto": [], "aux": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows["step"].append(int(r["step"]))
            for k in ("loss_total", "simple", "proto", "aux"):
                rows[k].append(float(r[k]))
    return rows


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
    data = {seed: _read(seed) for seed in range(5)}
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for ax, (col, title, ylab) in zip(axes.ravel(), COMPONENTS):
        last_vals = []
        for seed in range(5):
            d = data[seed]
            if d is None:
                continue
            st, vals = d["step"], d[col]
            trend = _moving_avg(vals, w=200)
            ax.plot(st, vals, color=COLORS[seed], alpha=0.15, linewidth=0.5)
            ax.plot(st, trend, color=COLORS[seed], alpha=0.95, linewidth=1.3,
                    label=f"seed {seed} (last≈{trend[-1]:.4f})")
            last_vals.append(trend[-1])
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Optimiser step")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        # log-scale helps proto/aux which span orders of magnitude at the start
        if col in ("proto", "aux"):
            ax.set_yscale("log")

    plt.suptitle(
        "TimeMoDE Loss Convergence — Heston seq_len=128  "
        "(global min-max norm; DiT+MoDE, K=8 top-2; T=250; EMA 0.9999; "
        "batch 2048 = 256×8 grad-accum; 1000 epochs = 4000 steps)",
        fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(LOSSES_DIR, "loss_convergence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
