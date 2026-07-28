"""TimeDiT Sine-disc HPO — STAGE 2 (refined grid in the winner's direction).

Stage-1 (hpo_sine.py) swept the coarse axes and found the discriminative-score
gap is driven by the *loss + sampler* configuration, not by training length.
The stage-1 leader was

    znorm · linear · learn_sigma=False · ddpm_fixed · lr=3e-4  ->  disc 0.0115

(paper target 0.0086).  Coordinate analysis (analyze_hpo.py) confirmed the
dominant directions: znorm >> minmax11, linear >> cosine, learn_sigma=False >>
True.  Those three axes are therefore **locked** here, and stage-2 concentrates
the search budget on the knobs that plausibly close the remaining 0.0115->0.0086
gap:

  * lr        : dense around the 3e-4 knife-edge winner   {3e-4, 4e-4, 5e-4}
                (2e-4 dropped: stage-1 showed it collapses to ~0.13-0.15 for
                 ddpm_fixed; below 3e-4 is the wrong side of the edge)
  * sampler   : the fixed-variance DDPM winner + one high-step deterministic
                probe                                   {ddpm_fixed, ddim_eta0(1000)}
                (ddim(250) dropped: at 3k steps it lagged at ~0.39)
  * ema       : EMA of the weights during training       {off, 0.999+warmup}
                (0.9999 removed: its ~10k-step averaging window never converges
                 in a 3k-step budget -> disc ~0.46; 0.999 (~1k window) + a
                 warmup ramp tracks the short run correctly)

Training length is raised (default 3000 steps) since more fidelity, not more
dynamics, is what the classifier is separating on.

Grid = 3 lr x 2 sampler x 2 ema = 12 trials, single GPU.

Usage (one GPU, per the standing 1-GPU rule):
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
      python hpo_stage2.py --steps 3000
"""
import argparse
import copy
import itertools
import json
import time

import numpy as np
import torch

from data_paper import load_dataset
from gaussian_diffusion import GaussianDiffusion
from timedit_model import build_timedit
from yoon_metrics import discriminative_score, predictive_score
from hpo_sine import normalize   # reuse the exact stage-1 normalisers

# Locked axes (stage-1 winners)
NORM = "znorm"
SCHEDULE = "linear"
LEARN_SIGMA = False


def build_grid():
    """Refined grid centred on the stage-1 winner. Each sampler entry is
    (name, ddim_steps) — ddim_steps is ignored for ddpm_fixed."""
    lrs = [3e-4, 4e-4, 5e-4]
    samplers = [("ddpm_fixed", None), ("ddim_eta0", 1000)]
    emas = [0.0, 0.999]
    grid = []
    for lr, (samp, dstep), ema in itertools.product(lrs, samplers, emas):
        grid.append({"norm": NORM, "schedule": SCHEDULE, "learn_sigma": LEARN_SIGMA,
                     "lr": lr, "sampler": samp, "ddim_steps": dstep, "ema": ema})
    return grid


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay):
        self.decay = decay
        self.step = 0
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        # Warmup ramp: effective decay grows from ~0.09 -> self.decay so the
        # shadow tracks the fast-moving early weights and converges within a
        # short (few-k step) HPO budget instead of lagging ~1/(1-decay) steps.
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v, alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def sample(diff, model, n_gen, L, C, device, cfg):
    fakes = []
    remaining = n_gen
    while remaining > 0:
        b = min(1000, remaining)
        if cfg["sampler"] == "ddpm_fixed":
            x = diff.p_sample_loop(model, (b, L, C), device,
                                   learn_sigma=cfg["learn_sigma"], sample_var="fixed")
        elif cfg["sampler"] == "ddim_eta0":
            x = diff.ddim_sample_loop(model, (b, L, C), device,
                                      learn_sigma=cfg["learn_sigma"],
                                      steps=cfg["ddim_steps"], eta=0.0)
        else:
            raise ValueError(cfg["sampler"])
        fakes.append(x.cpu().numpy())
        remaining -= b
    return np.concatenate(fakes, 0)[:n_gen]


def run_trial(cfg, data01, device, args, train_seed=0):
    N, L, C = data01.shape
    cut = int(N * 0.8)
    dataN, denorm = normalize(data01, cut, cfg["norm"])
    train = torch.tensor(dataN[:cut], device=device)

    torch.manual_seed(train_seed)
    model = build_timedit(L, C, model_size="S", learn_sigma=cfg["learn_sigma"]).to(device)
    diff = GaussianDiffusion(T=1000, schedule=cfg["schedule"], loss_mode="hybrid")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg.get("weight_decay", 0.0))
    ema = EMA(model, cfg["ema"]) if cfg["ema"] > 0 else None

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
        if ema is not None:
            ema.update(model)
        if not np.isfinite(loss.item()):
            return {"disc_mean": float("nan"), "pred_mean": float("nan"),
                    "note": f"NaN at step {step}"}
    train_s = time.time() - t0

    if ema is not None:
        ema.copy_to(model)
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
    p.add_argument("--steps", type=int, default=3000)
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
    out = "hpo_stage2_shard0.jsonl"
    print(f"[hpo2] {len(grid)} trials -> {out}  steps={args.steps} "
          f"locked: norm={NORM} sched={SCHEDULE} ls={LEARN_SIGMA}", flush=True)

    data01 = load_dataset("sine", seq_len=args.seq_len, seed=0)
    if args.subset and len(data01) > args.subset:
        data01 = data01[:args.subset]
    print(f"[hpo2] sine data {data01.shape}", flush=True)

    for k, cfg in enumerate(grid):
        t0 = time.time()
        res = run_trial(cfg, data01, device, args)
        row = {**cfg, **res, "wall_s": time.time() - t0}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        ds = cfg["ddim_steps"]
        print(f"[hpo2] {k + 1}/{len(grid)} lr={cfg['lr']:.0e} "
              f"samp={cfg['sampler']}{'' if ds is None else f'({ds})'} "
              f"ema={cfg['ema']} -> disc={res['disc_mean']:.4f} "
              f"pred={res['pred_mean']:.4f} ({row['wall_s']:.0f}s)", flush=True)
    print("[hpo2] DONE", flush=True)
    with open("hpo_status.txt", "a") as f:
        f.write("STAGE2_DONE\n")


if __name__ == "__main__":
    main()
