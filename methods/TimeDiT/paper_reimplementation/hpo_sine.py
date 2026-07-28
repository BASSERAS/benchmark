"""TimeDiT synthetic-generation HPO harness (target: discriminative score on Sine).

Motivation
----------
Every faithful-reproduction attempt (scratch, decomp, pretrain->finetune) landed the
Yoon *discriminative* score in 0.26-0.42, while the paper reports ~0.0086 and SOTA
TS-diffusion (Diffusion-TS) reaches ~0.006 on Sine.  The predictive score is already
fine (~0.10), so the model learns the dynamics; the marginal/joint fidelity of the
*samples* is what the classifier trivially separates.  This harness grid-searches the
knobs most likely to move sample fidelity, on a small Sine subset with short training,
so we can find the recipe that actually targets the disc score before committing to a
full run.

Swept axes (built-in grid, sharded across GPUs):
  * norm       : how the [0,1] data is mapped before diffusion  {znorm, minmax11}
  * sampler    : reverse process                                {ddpm_learned, ddpm_fixed,
                                                                  ddim_eta0, ddim_eta50}
  * schedule   : beta schedule                                  {linear, cosine}
  * lr         : Adam learning rate                             {1e-4, 3e-4}
  * learn_sigma: learned variance (hybrid L) vs simple L        {True, False}
(only sampler/learn_sigma-compatible combos are enumerated)

Each trial: train from scratch on C=5 Sine (no K_MAX padding), sample n_gen windows,
denormalise to [0,1], score disc (mean of disc_seeds) + pred (1 seed).  Results are
appended to hpo_results_shard{idx}.jsonl so the two GPU shards never collide.

Usage (per GPU, launched by the orchestrator):
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
      python hpo_sine.py --shard_idx 0 --n_shards 2 --steps 4000
"""
import argparse
import itertools
import json
import time

import numpy as np
import torch

from data_paper import load_dataset
from gaussian_diffusion import GaussianDiffusion
from timedit_model import build_timedit
from yoon_metrics import discriminative_score, predictive_score


def build_grid():
    """Enumerate the full grid; only valid (sampler, learn_sigma) pairs are kept."""
    norms = ["znorm", "minmax11"]
    schedules = ["linear", "cosine"]
    lrs = [1e-4, 3e-4]
    # (sampler, learn_sigma) — ddpm_learned requires learned variance
    sampler_sigma = [
        ("ddpm_learned", True),
        ("ddim_eta0", True),
        ("ddim_eta0", False),
        ("ddpm_fixed", False),
    ]
    grid = []
    for norm, sched, lr, (samp, ls) in itertools.product(
            norms, schedules, lrs, sampler_sigma):
        grid.append({"norm": norm, "schedule": sched, "lr": lr,
                     "sampler": samp, "learn_sigma": ls})
    return grid


def normalize(data01, cut, norm):
    """Map [0,1] data to the diffusion space; return (dataN, denorm_fn)."""
    if norm == "znorm":
        mu = data01[:cut].mean(axis=(0, 1), keepdims=True)
        sd = data01[:cut].std(axis=(0, 1), keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        dataN = ((data01 - mu) / sd).astype(np.float32)

        def denorm(fz):
            return np.clip(fz * sd + mu, 0.0, 1.0).astype(np.float32)
        return dataN, denorm
    if norm == "minmax11":
        # data already in [0,1] -> [-1,1]
        dataN = (2.0 * data01 - 1.0).astype(np.float32)

        def denorm(fz):
            return np.clip((fz + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        return dataN, denorm
    raise ValueError(norm)


def sample(diff, model, n_gen, L, C, device, cfg):
    fakes = []
    remaining = n_gen
    while remaining > 0:
        b = min(1000, remaining)
        samp = cfg["sampler"]
        if samp == "ddpm_learned":
            x = diff.p_sample_loop(model, (b, L, C), device,
                                   learn_sigma=cfg["learn_sigma"], sample_var="learned")
        elif samp == "ddpm_fixed":
            x = diff.p_sample_loop(model, (b, L, C), device,
                                   learn_sigma=cfg["learn_sigma"], sample_var="fixed")
        elif samp == "ddim_eta0":
            x = diff.ddim_sample_loop(model, (b, L, C), device,
                                      learn_sigma=cfg["learn_sigma"], steps=100, eta=0.0)
        elif samp == "ddim_eta50":
            x = diff.ddim_sample_loop(model, (b, L, C), device,
                                      learn_sigma=cfg["learn_sigma"], steps=100, eta=0.5)
        else:
            raise ValueError(samp)
        fakes.append(x.cpu().numpy())
        remaining -= b
    return np.concatenate(fakes, 0)[:n_gen]


def run_trial(cfg, data01, device, args):
    N, L, C = data01.shape
    cut = int(N * 0.8)
    dataN, denorm = normalize(data01, cut, cfg["norm"])
    train = torch.tensor(dataN[:cut], device=device)

    torch.manual_seed(0)
    model = build_timedit(L, C, model_size="S", learn_sigma=cfg["learn_sigma"]).to(device)
    diff = GaussianDiffusion(T=1000, schedule=cfg["schedule"], loss_mode="hybrid")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0.0)

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, cut, (args.batch,), device=device)
        x0 = train[idx]
        loss, _ = diff.training_loss(model, x0, learn_sigma=cfg["learn_sigma"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not np.isfinite(loss.item()):
            return {"disc_mean": float("nan"), "pred_mean": float("nan"),
                    "note": f"NaN at step {step}"}
    train_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        fz = sample(diff, model, args.n_gen, L, C, device, cfg)
    fake01 = denorm(fz)

    disc = [discriminative_score(data01, fake01, device=device, seed=s)
            for s in range(args.disc_seeds)]
    pred = predictive_score(data01, fake01, device=device, seed=0)
    return {"disc_mean": float(np.mean(disc)), "disc_std": float(np.std(disc)),
            "disc_seeds": [float(x) for x in disc], "pred_mean": float(pred),
            "fake_min": float(fake01.min()), "fake_max": float(fake01.max()),
            "train_s": train_s}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--subset", type=int, default=4000)
    p.add_argument("--n_gen", type=int, default=1000)
    p.add_argument("--disc_seeds", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    grid = build_grid()
    mine = [c for i, c in enumerate(grid) if i % args.n_shards == args.shard_idx]
    out = f"hpo_results_shard{args.shard_idx}.jsonl"
    print(f"[hpo] shard {args.shard_idx}/{args.n_shards}: {len(mine)}/{len(grid)} trials "
          f"-> {out}  steps={args.steps} subset={args.subset} n_gen={args.n_gen}",
          flush=True)

    data01 = load_dataset("sine", seq_len=args.seq_len, seed=0)
    if args.subset and len(data01) > args.subset:
        data01 = data01[:args.subset]
    print(f"[hpo] sine data {data01.shape}", flush=True)

    for k, cfg in enumerate(mine):
        t0 = time.time()
        res = run_trial(cfg, data01, device, args)
        row = {**cfg, **res, "wall_s": time.time() - t0}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[hpo {args.shard_idx}] {k + 1}/{len(mine)} "
              f"norm={cfg['norm']} samp={cfg['sampler']} ls={cfg['learn_sigma']} "
              f"sched={cfg['schedule']} lr={cfg['lr']:.0e} -> "
              f"disc={res['disc_mean']:.4f} pred={res['pred_mean']:.4f} "
              f"({row['wall_s']:.0f}s)", flush=True)
    print(f"[hpo {args.shard_idx}] DONE", flush=True)


if __name__ == "__main__":
    main()
