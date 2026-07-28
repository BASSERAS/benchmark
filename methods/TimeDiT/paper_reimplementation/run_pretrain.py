"""TimeDiT pretraining driver (paper §5.6, faithful reproduction).

Pretrains DiT-S self-supervised on the multi-dataset corpus (ETTx4 + optional
electricity/weather) using the Time Series Mask Unit: each training instance gets
one of {M^R random, M^S stride, M^B block, M^Rec reconstruction} and the model
denoises the target region conditioned on the observed region.  Channels are padded
to K_MAX=40 with a validity vector so padded channels are excluded from the loss.

Output: pretrain_dit_s.pt (EMA state_dict), the checkpoint finetuned by run_finetune.py.

Usage
-----
Smoke:  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 python run_pretrain.py --smoke
Full:   CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 python run_pretrain.py \
            --steps 30000 --out pretrain_dit_s.pt
"""
import argparse
import copy
import json
import time

import numpy as np
import torch

from data_pretrain import K_MAX, load_pretrain_corpus
from gaussian_diffusion import GaussianDiffusion
from timedit_model import build_timedit


def ema_update(ema_model, model, decay):
    with torch.no_grad():
        for e, p in zip(ema_model.parameters(), model.parameters()):
            e.mul_(decay).add_(p, alpha=1 - decay)
        for eb, b in zip(ema_model.buffers(), model.buffers()):
            eb.copy_(b)


def build_masks(B, L, C, device, rng):
    """Per-instance conditioning mask (1 = observed). Mixture of M^R/M^S/M^B/M^Rec.

    Vectorised: each of the B rows is assigned one mask type; all four candidate
    masks are built batch-wide and selected per row by type.
    """
    types = torch.tensor(rng.integers(0, 4, size=B), device=device)   # 0=rec,1=rnd,2=stride,3=block
    t_idx = torch.arange(L, device=device).view(1, L, 1)              # (1,L,1)

    # M^R random points: per-row observation probability in [0.3,0.7]
    p_obs = torch.tensor(rng.uniform(0.3, 0.7, size=B), device=device,
                         dtype=torch.float32).view(B, 1, 1)
    m_rnd = (torch.rand(B, L, C, device=device) < p_obs).float()

    # M^S stride: observe every k-th timestep, k in {2,3,4}
    k = torch.tensor(rng.integers(2, 5, size=B), device=device).view(B, 1, 1)
    m_stride = ((t_idx % k) == 0).float().expand(B, L, C)

    # M^B block: observe everything except a contiguous target block
    blk = torch.tensor(rng.integers(L // 4, L // 2 + 1, size=B), device=device).view(B, 1, 1)
    start = torch.tensor([rng.integers(0, L - int(b) + 1) for b in blk.flatten().tolist()],
                         device=device).view(B, 1, 1)
    in_block = (t_idx >= start) & (t_idx < start + blk)              # (B,L,1)
    m_block = (~in_block).float().expand(B, L, C)

    ty = types.view(B, 1, 1)
    cond = torch.zeros(B, L, C, device=device)                       # type 0 = M^Rec
    cond = torch.where(ty == 1, m_rnd, cond)
    cond = torch.where(ty == 2, m_stride, cond)
    cond = torch.where(ty == 3, m_block, cond)
    return cond


def parse_args():
    p = argparse.ArgumentParser(description="TimeDiT SSL pretraining (mask unit)")
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--model_size", choices=["S", "B", "L"], default="S")
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--max_windows_per_ds", type=int, default=20000)
    p.add_argument("--out", default="pretrain_dit_s.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    steps = args.steps
    max_w = args.max_windows_per_ds
    if args.smoke:
        steps = 300
        max_w = 2000

    # ---- corpus: pad every block to K_MAX, stack ----
    corpus = load_pretrain_corpus(seq_len=args.seq_len, max_windows_per_ds=max_w,
                                  seed=args.seed)
    all_x, all_valid = [], []
    for d in corpus:
        w = d["windows"]                                    # (N, L, C)
        C = d["C"]
        pad = np.zeros((w.shape[0], w.shape[1], K_MAX - C), dtype=np.float32)
        all_x.append(np.concatenate([w, pad], axis=-1))
        v = np.zeros(K_MAX, dtype=np.float32); v[:C] = 1.0
        all_valid.append(np.broadcast_to(v, (w.shape[0], K_MAX)))
    X = torch.tensor(np.concatenate(all_x, 0), device=device)          # (Ntot,L,K_MAX)
    V = torch.tensor(np.concatenate(all_valid, 0), device=device)      # (Ntot,K_MAX)
    Ntot, L, _ = X.shape
    print(f"[corpus] windows={Ntot} L={L} K_MAX={K_MAX} blocks={len(corpus)} steps={steps}")

    # ---- model + diffusion ----
    model = build_timedit(L, K_MAX, model_size=args.model_size, learn_sigma=True).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    diff = GaussianDiffusion(T=args.T, schedule=args.schedule, loss_mode="hybrid")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] DiT-{args.model_size} params={n_params/1e6:.2f}M learn_sigma=True")

    # ---- train ----
    model.train()
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, Ntot, (args.batch,), device=device)
        x0 = X[idx]                                         # (B,L,K_MAX)
        valid = V[idx].unsqueeze(1)                         # (B,1,K_MAX)
        cond = build_masks(args.batch, L, K_MAX, device, rng)
        cond = cond * valid                                 # never "observe" padding
        loss, parts = diff.masked_training_loss(model, x0, cond, valid, learn_sigma=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        decay = 0.0 if step < 100 else args.ema_decay
        ema_update(ema, model, decay)
        if not np.isfinite(loss.item()):
            raise RuntimeError(f"NaN/Inf at step {step}: {parts}")
        if step % 200 == 0 or step == steps - 1:
            print(f"[{step:6d}] loss={loss.item():.4f} simple={parts['simple']:.4f} "
                  f"vlb={parts['vlb']:.4f} ({time.time()-t0:.1f}s)")
    train_time = time.time() - t0

    torch.save(ema.state_dict(), args.out)
    cfg = {"steps": steps, "k_max": K_MAX, "seq_len": L, "T": args.T,
           "schedule": args.schedule, "ema_decay": args.ema_decay,
           "model_size": args.model_size, "n_blocks": len(corpus),
           "n_windows": int(Ntot), "params_M": n_params / 1e6,
           "train_time_sec": train_time, "corpus": [d["name"] for d in corpus]}
    with open(args.out.rsplit(".", 1)[0] + "_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[done] EMA -> {args.out}  ({train_time:.1f}s)")


if __name__ == "__main__":
    main()
