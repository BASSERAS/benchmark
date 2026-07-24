"""GT-GAN — Heston loss convergence plot (5 seeds overlaid).

GT-GAN (gtgan mode) trains in two phases (see ../train_heston.py):

  Phase 1 — embed pretrain (``--first_epoch 10000`` steps): only the
            autoencoder reconstruction loss ``loss_e`` = MSE(recovery(embed(x)), x)
            is optimised (NeuralCDE embedder + Multi_Layer_ODENetwork recovery).
  Phase 2 — joint adversarial (``--max_steps 3000`` steps): the CNF generator,
            ODE discriminator, recovery and embedder are updated together.
            Logged losses:
              loss_e    — reconstruction (kept small during adversarial training)
              loss_d    — discriminator BCE (real vs CNF-generated latents)
              loss_g_u  — generator unsupervised adversarial loss
              loss_g_v  — generator moment-matching loss (mean + var of features)
              loss_s    — supervised one-step-ahead latent prediction loss

There is **no supervisor network** in gtgan mode — ``loss_s`` is a supervised
term on the shared latent, not a separate sub-network.  This mirrors
methods/COSCI-GAN/code/plot_losses.py but with GT-GAN's phase layout: a
continuous ``loss_e`` panel spanning both phases (global step, dashed line marks
the embed->joint transition at step 10000) plus one panel per joint-phase loss.

Reads   ../losses/seed_{0..4}_losses.csv
        (columns: step,phase,loss_e,loss_d,loss_g_u,loss_g_v,loss_s)
Writes  ../losses/loss_convergence.png

Usage:
    /home/tbasseras/gpu-venv/bin/python methods/GT-GAN/code/plot_losses.py
"""
from __future__ import annotations
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DIR = os.path.dirname(HERE)                     # methods/GT-GAN
LOSSES_DIR = os.path.join(METHOD_DIR, "losses")
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

EMBED_STEPS = 10000   # --first_epoch (offset for the global-step axis)

# joint-phase-only loss panels
JOINT_FIELDS = ["loss_d", "loss_g_u", "loss_g_v", "loss_s"]
JOINT_TITLES = ["Discriminator (loss_d)",
                "Generator adversarial (loss_g_u)",
                "Generator moment-match (loss_g_v)",
                "Supervised latent (loss_s)"]


def _read(seed: int):
    path = os.path.join(LOSSES_DIR, f"seed_{seed}_losses.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(x):
    """Parse a possibly-empty CSV cell to float or None."""
    if x is None or x == "":
        return None
    return float(x)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    last_e = []

    for seed in range(5):
        rows = _read(seed)
        if rows is None:
            continue

        # ── loss_e over full training (embed + joint), global step axis ──
        gstep, e_vals = [], []
        for r in rows:
            e = _f(r["loss_e"])
            if e is None:
                continue
            s = int(r["step"])
            g = s if r["phase"] == "embed" else EMBED_STEPS + s
            gstep.append(g)
            e_vals.append(e)
        if e_vals:
            axes[0].plot(gstep, e_vals, color=COLORS[seed], alpha=0.85,
                         linewidth=1.2, label=f"seed {seed}")
            last_e.append(e_vals[-1])

        # ── joint-phase-only losses over joint step ──
        joint = [r for r in rows if r["phase"] == "joint"]
        jstep = [int(r["step"]) for r in joint]
        for ax, fld in zip(axes[1:5], JOINT_FIELDS):
            vals = [_f(r[fld]) for r in joint]
            xs = [x for x, v in zip(jstep, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            ax.plot(xs, ys, color=COLORS[seed], alpha=0.85,
                    linewidth=1.2, label=f"seed {seed}")

    # loss_e panel styling
    axes[0].axvline(EMBED_STEPS, color="black", linestyle="--", linewidth=1.0,
                    label="embed->joint")
    axes[0].set_title("Reconstruction (loss_e) — full training", fontsize=12)
    axes[0].set_xlabel("Global step (embed 0-10000, then joint)")
    axes[0].set_ylabel("loss_e")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # joint-loss panels styling
    for ax, title in zip(axes[1:5], JOINT_TITLES):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Joint step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[5].axis("off")   # 6th panel unused (loss_e + 4 joint losses = 5 panels)

    band = (f"loss_e_last ∈ [{min(last_e):.2e}, {max(last_e):.2e}]"
            if last_e else "")
    fig.suptitle(
        "GT-GAN Loss Convergence — Heston seq_len=128  "
        f"(embed 10000 steps + joint 3000 steps; {band})",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(LOSSES_DIR, "loss_convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}  ({band})")


if __name__ == "__main__":
    main()
