"""TimeMoDE paper reproduction driver — "From Scratch" synthetic generation.

Trains the from-scratch TimeMoDE (DiT-MoDE, no pretraining) on the StarLightCurves
10% few-shot windows and evaluates the three metrics TimeMoDE reports (Table 19
"From Scratch" column via the Appendix B.4 protocol):

  context-FID (c-FID)  = TS2Vec Frechet distance   (lower = better)
  discriminative score = |accuracy - 0.5|          (lower = better)
  predictive score     = TSTR MAE                  (lower = better)

Paper protocol (Table 7 finetune / From-Scratch, Table 8 shared arch):
  arch: hidden 256, depth 6, heads 4, experts 8 (top-2), expansion 4, T=250.
  optim: AdamW lr 1e-4, wd 1e-5, batch min(2048, N_train), 1000 epochs,
         EMA decay 0.9999.

REPRO notes (paper leaves unspecified):
  - w_aux / w_proto: Eq 15 sums L_DDPM + L_proto + L_aux without stated weights.
    We use small weights (0.01 each) so the regularisers do not dominate L_DDPM.
  - learn_sigma=False: the written loss (Eq 3/15) has no VLB term.
  - DP exemplars: each step/sample builds the Domain Prompt from a random batch
    of clean few-shot training windows (single-domain setting).

Usage:
  Smoke (sanity — 5% data, few epochs, NaN / loss-decrease check):
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
      /home/tbasseras/gpu-venv/bin/python run_paper.py --smoke

  Full From-Scratch reproduction:
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
      /home/tbasseras/gpu-venv/bin/python run_paper.py \
        --epochs 1000 --batch 2048 --out results_slc.json
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from diffusion import Diffusion
from timemode_model import build_timemode
import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")


def ema_update(ema_model, model, decay):
    with torch.no_grad():
        for e, p in zip(ema_model.parameters(), model.parameters()):
            e.mul_(decay).add_(p, alpha=1 - decay)
        for eb, b in zip(ema_model.buffers(), model.buffers()):
            eb.copy_(b)


def parse_args():
    p = argparse.ArgumentParser(description="TimeMoDE From-Scratch reproduction (StarLightCurves)")
    p.add_argument("--train_npy", default=os.path.join(DATA, "slc_train10_windows_24x1.npy"))
    p.add_argument("--real_npy", default=os.path.join(DATA, "slc_full_windows_24x1.npy"))
    p.add_argument("--T", type=int, default=250)
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--w_aux", type=float, default=0.01)
    p.add_argument("--w_proto", type=float, default=0.01)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--n_gen", type=int, default=0, help="#samples to generate (0 = match #real eval)")
    p.add_argument("--n_real_eval", type=int, default=3000, help="real windows subsampled for scoring")
    p.add_argument("--eval_seeds", type=int, default=3)
    p.add_argument("--out", default="results_slc.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def dp_batch(pool_t, n):
    idx = torch.randint(0, len(pool_t), (n,), device=pool_t.device)
    return pool_t[idx]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    train = np.load(args.train_npy).astype(np.float32)
    real = np.load(args.real_npy).astype(np.float32)
    epochs = args.epochs
    if args.smoke:
        train = train[: max(int(len(train) * 0.05), 64)]
        epochs = 30
    N, L, C = train.shape
    batch = min(args.batch, N)
    steps_per_epoch = max(N // batch, 1)
    print(f"[data] train {train.shape} real {real.shape} "
          f"range=[{train.min():.3f},{train.max():.3f}] batch={batch} epochs={epochs}", flush=True)

    train_t = torch.tensor(train, device=device)

    model = build_timemode(L, C, learn_sigma=False).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    diff = Diffusion(T=args.T, schedule=args.schedule)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] TimeMoDE params={n_params/1e6:.2f}M", flush=True)

    loss_log = []
    model.train()
    t0 = time.time()
    gstep = 0
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            idx = np.random.randint(0, N, batch)
            x0 = train_t[idx]
            dpx = dp_batch(train_t, batch)
            loss, parts = diff.training_loss(model, x0, dpx,
                                             w_proto=args.w_proto, w_aux=args.w_aux)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            decay = 0.0 if gstep < 100 else args.ema_decay
            ema_update(ema, model, decay)
            if not np.isfinite(loss.item()):
                raise RuntimeError(f"NaN/Inf loss at step {gstep}: {parts}")
            gstep += 1
        if epoch % max(epochs // 20, 1) == 0 or epoch == epochs - 1:
            loss_log.append({"epoch": epoch, "step": gstep, "loss": loss.item(), **parts})
            print(f"[ep {epoch:4d} step {gstep:6d}] loss={loss.item():.4f} "
                  f"simple={parts['simple']:.4f} proto={parts['proto']:.2f} "
                  f"aux={parts['aux']:.2f} ({time.time()-t0:.1f}s)", flush=True)
    train_time = time.time() - t0

    # ---- sample ----
    n_real_eval = min(args.n_real_eval, len(real))
    ridx = np.random.permutation(len(real))[:n_real_eval]
    real_eval = real[ridx]
    n_gen = args.n_gen if args.n_gen > 0 else n_real_eval

    fakes = []
    ema.eval()
    with torch.no_grad():
        remaining = n_gen
        while remaining > 0:
            b = min(1024, remaining)
            dpx = dp_batch(train_t, b)
            x = diff.p_sample_loop(ema, (b, L, C), device, dpx)
            fakes.append(x.cpu().numpy())
            remaining -= b
    fake = np.concatenate(fakes, 0)[:n_gen]
    fake = np.clip(fake, 0.0, 1.0).astype(np.float32)
    print(f"[gen] fake {fake.shape} range=[{fake.min():.3f},{fake.max():.3f}]", flush=True)

    # ---- score (average over eval seeds) ----
    cfid, disc, pred = [], [], []
    for s in range(args.eval_seeds):
        np.random.seed(s); torch.manual_seed(s)
        cfid.append(metrics.context_fid(real_eval, fake))
        disc.append(metrics.discriminative_score(real_eval, fake, device=device, seed=s))
        pred.append(metrics.predictive_score(real_eval, fake, device=device, seed=s))
        print(f"[eval seed {s}] cfid={cfid[-1]:.4f} disc={disc[-1]:.4f} pred={pred[-1]:.4f}", flush=True)

    result = {
        "dataset": "StarLightCurves",
        "setting": "From Scratch, 10% few-shot",
        "seq_len": L, "channels": C,
        "n_train": int(N), "n_gen": int(n_gen), "n_real_eval": int(n_real_eval),
        "epochs": epochs, "batch": batch, "smoke": args.smoke,
        "params_M": n_params / 1e6, "train_time_sec": train_time,
        "context_fid_mean": float(np.mean(cfid)), "context_fid_std": float(np.std(cfid)),
        "discriminative_mean": float(np.mean(disc)), "discriminative_std": float(np.std(disc)),
        "predictive_mean": float(np.mean(pred)), "predictive_std": float(np.std(pred)),
        "context_fid_per_seed": [float(x) for x in cfid],
        "discriminative_per_seed": [float(x) for x in disc],
        "predictive_per_seed": [float(x) for x in pred],
        "loss_log": loss_log,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] cfid={result['context_fid_mean']:.4f} "
          f"disc={result['discriminative_mean']:.4f}±{result['discriminative_std']:.4f} "
          f"pred={result['predictive_mean']:.4f}±{result['predictive_std']:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
