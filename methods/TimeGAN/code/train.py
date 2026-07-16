"""
TimeGAN training orchestrator — 5 seeds on 2 GPUs in pairs.
Sets CUDA_VISIBLE_DEVICES in subprocess env so torch picks up the correct GPU.

Usage:
  python train.py               # uses GPU 0 and GPU 3
  python train.py --gpu0 0 --gpu1 3
"""
import argparse, csv, os, subprocess, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # methods/TimeGAN/code
METHOD_DIR = os.path.dirname(SCRIPT_DIR)                    # methods/TimeGAN
PYTHON     = sys.executable


def run_seed(seed: int, gpu: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    core_start = int(gpu) * 8
    cmd = [
        "taskset", "-c", f"{core_start}-{core_start+7}",
        PYTHON, os.path.join(SCRIPT_DIR, "train_seed.py"),
        "--seed", str(seed),
    ]
    os.makedirs(os.path.join(METHOD_DIR, "losses"), exist_ok=True)
    log = open(os.path.join(METHOD_DIR, "losses", f"seed_{seed}.log"), "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"  Launched seed {seed} on GPU {gpu}  PID={p.pid}", flush=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu0", default="0")
    ap.add_argument("--gpu1", default="3")
    args = ap.parse_args()

    seeds   = [0, 1, 2, 3, 4]
    batches = [
        [(seeds[0], args.gpu0), (seeds[1], args.gpu1)],
        [(seeds[2], args.gpu0), (seeds[3], args.gpu1)],
        [(seeds[4], args.gpu0)],
    ]

    t_total = time.perf_counter()
    for batch in batches:
        desc = " + ".join(f"seed {s} GPU {g}" for s, g in batch)
        print(f"\n--- {desc} ---", flush=True)
        procs = [run_seed(s, g) for s, g in batch]
        for p in procs:
            p.wait()
        codes = [p.returncode for p in procs]
        print(f"Batch done  return codes: {codes}", flush=True)

    elapsed = (time.perf_counter() - t_total) / 60
    print(f"\nAll 5 seeds done in {elapsed:.1f} min", flush=True)
    plot_losses()


def plot_losses():
    losses_dir = os.path.join(METHOD_DIR, "losses")
    colors = ["#2196F3","#FF9800","#4CAF50","#9C27B0","#F44336"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.flatten()
    fields = ["e_loss","s_loss","g_loss","d_loss"]
    titles = ["Embedding Loss","Supervised Loss","Generator Loss","Discriminator Loss"]

    for seed in range(5):
        path = os.path.join(losses_dir, f"seed_{seed}_losses.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for ax, fld in zip(axes, fields):
            steps = [int(r["step"]) for r in rows if r.get(fld) not in ("","None","")]
            vals  = [float(r[fld])  for r in rows if r.get(fld) not in ("","None","")]
            if steps:
                ax.plot(steps, vals, color=colors[seed], alpha=0.8,
                        label=f"seed {seed}", linewidth=1.2)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle("TimeGAN Loss Convergence — Heston seq_len=128", fontsize=14)
    plt.tight_layout()
    out = os.path.join(losses_dir, "loss_convergence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
