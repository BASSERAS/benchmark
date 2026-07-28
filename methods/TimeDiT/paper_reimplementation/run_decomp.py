"""TimeDiT-Decomp driver — x0-prediction diffusion with the smoothness stack.

Improvement variant of run_paper.py aimed at closing the discriminative-score gap.
Differences from the plain reproduction, all on the training side, plus the
decomposition head in timedit_decomp.py:

  * data centred to [-1, 1]        (x = 2*minmax - 1) instead of [0, 1]
  * x0-parameterisation            (model predicts x0, not eps)
  * cosine beta-schedule           (Nichol & Dhariwal) instead of linear
  * Min-SNR-gamma weighting        (Hang et al. 2023): x0-loss weight = min(SNR_t, g)
  * x0 clamp to [-1, 1]            at both training-target range and sampling
  * ancestral fixed-variance       reverse process (posterior sigma_t)

Everything else (DiT-S backbone, EMA 0.9999, Adam lr 1e-4 no wd, grad-clip 1.0,
batch 512, 15000 steps, Yoon discriminative/predictive metrics, 80% train split)
matches run_paper.py so the comparison is apples-to-apples.

Metrics are computed in the original [0, 1] scale (generated samples are mapped back
via (x+1)/2) so discriminative/predictive numbers are directly comparable to the
plain TimeDiT results and to the paper's Table 6.

Usage
-----
Smoke (5% data, 300 steps):
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python run_decomp.py --dataset sine --smoke
Full:
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python run_decomp.py --dataset sine \
        --steps 15000 --out results_sine_decomp.json --save_model ema_sine_decomp.pt
"""
import argparse
import copy
import json
import time

import numpy as np
import torch

from data_paper import load_dataset
from gaussian_diffusion import make_beta_schedule
from timedit_decomp import build_timedit_decomp
from yoon_metrics import discriminative_score, predictive_score


def _extract(arr, t, shape):
    out = torch.as_tensor(arr, device=t.device, dtype=torch.float32)[t]
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


class X0Diffusion:
    """Minimal x0-prediction DDPM with cosine schedule + Min-SNR-gamma weighting."""

    def __init__(self, T=1000, schedule="cosine", min_snr_gamma=5.0):
        self.T = T
        self.gamma = min_snr_gamma
        betas = make_beta_schedule(T, schedule)
        alphas = 1.0 - betas
        acp = np.cumprod(alphas)
        acp_prev = np.append(1.0, acp[:-1])
        self.sqrt_acp = np.sqrt(acp)
        self.sqrt_one_minus_acp = np.sqrt(1.0 - acp)
        self.snr = acp / (1.0 - acp)                       # signal-to-noise ratio
        # posterior q(x_{t-1}|x_t,x_0)
        post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        self.posterior_log_var = np.log(np.append(post_var[1], post_var[1:]))
        self.posterior_c1 = betas * np.sqrt(acp_prev) / (1.0 - acp)
        self.posterior_c2 = (1.0 - acp_prev) * np.sqrt(alphas) / (1.0 - acp)

    def q_sample(self, x0, t, noise):
        return _extract(self.sqrt_acp, t, x0.shape) * x0 + \
            _extract(self.sqrt_one_minus_acp, t, x0.shape) * noise

    def training_loss(self, model, x0):
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        x0_hat = model(xt, t)
        # Min-SNR-gamma: x0-MSE weighted by min(SNR_t, gamma)
        snr_t = _extract(self.snr, t, (B, 1, 1)).reshape(B)
        w = torch.clamp(snr_t, max=self.gamma)
        se = ((x0_hat - x0) ** 2).mean(dim=[1, 2])          # per-sample MSE
        loss = (w * se).mean()
        return loss, {"mse": se.mean().item()}

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device):
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x0_hat = model(x, t).clamp(-1.0, 1.0)
            mean = _extract(self.posterior_c1, t, x.shape) * x0_hat + \
                _extract(self.posterior_c2, t, x.shape) * x
            if i > 0:
                log_var = _extract(self.posterior_log_var, t, x.shape)
                x = mean + torch.exp(0.5 * log_var) * torch.randn_like(x)
            else:
                x = mean
        return x


def ema_update(ema_model, model, decay):
    with torch.no_grad():
        for e, p in zip(ema_model.parameters(), model.parameters()):
            e.mul_(decay).add_(p, alpha=1 - decay)
        for eb, b in zip(ema_model.buffers(), model.buffers()):
            eb.copy_(b)


def parse_args():
    p = argparse.ArgumentParser(description="TimeDiT-Decomp (x0 + smoothness stack)")
    p.add_argument("--dataset", choices=["sine", "stock"], default="sine")
    p.add_argument("--stock_csv", default=None)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--model_size", choices=["S", "B", "L"], default="S")
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--min_snr_gamma", type=float, default=5.0)
    p.add_argument("--n_poly", type=int, default=4)
    p.add_argument("--n_freq", type=int, default=8)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--n_gen", type=int, default=0)
    p.add_argument("--eval_seeds", type=int, default=3)
    p.add_argument("--out", default="results_decomp.json")
    p.add_argument("--save_model", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- data (loaded in [0,1], centred to [-1,1] for training) ----
    data01 = load_dataset(args.dataset, seq_len=args.seq_len,
                          stock_csv=args.stock_csv, seed=args.seed)
    steps = args.steps
    if args.smoke:
        n_sub = max(int(len(data01) * 0.05), args.batch)
        data01 = data01[:n_sub]
        steps = 300
    N, L, C = data01.shape
    data = (data01 * 2.0 - 1.0).astype(np.float32)          # [-1, 1]
    print(f"[data] {args.dataset}: {data.shape} centred range=[{data.min():.3f},{data.max():.3f}] steps={steps}")

    cut = int(N * 0.8)
    train_t = torch.tensor(data[:cut], dtype=torch.float32, device=device)

    # ---- model + diffusion ----
    model = build_timedit_decomp(L, C, model_size=args.model_size,
                                 n_poly=args.n_poly, n_freq=args.n_freq).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    diff = X0Diffusion(T=args.T, schedule=args.schedule, min_snr_gamma=args.min_snr_gamma)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] TimeDiT-Decomp-{args.model_size} params={n_params/1e6:.2f}M "
          f"n_poly={args.n_poly} n_freq={args.n_freq} schedule={args.schedule} gamma={args.min_snr_gamma}")

    # ---- train ----
    loss_log = []
    model.train()
    t0 = time.time()
    for step in range(steps):
        idx = np.random.randint(0, len(train_t), args.batch)
        x0 = train_t[idx]
        loss, parts = diff.training_loss(model, x0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        decay = 0.0 if step < 100 else args.ema_decay
        ema_update(ema, model, decay)
        if not np.isfinite(loss.item()):
            raise RuntimeError(f"NaN/Inf loss at step {step}: {parts}")
        if step % 100 == 0 or step == steps - 1:
            loss_log.append({"step": step, "loss": loss.item(), "mse": parts["mse"]})
            print(f"[{step:5d}] loss={loss.item():.4f} mse={parts['mse']:.4f} "
                  f"({time.time()-t0:.1f}s)")
    train_time = time.time() - t0

    # ---- sample + score (map back to [0,1] for metrics) ----
    n_gen = args.n_gen if args.n_gen > 0 else N
    ema.eval()
    fakes = []
    with torch.no_grad():
        remaining = n_gen
        while remaining > 0:
            b = min(1024, remaining)
            x = diff.p_sample_loop(ema, (b, L, C), device)
            fakes.append(x.cpu().numpy())
            remaining -= b
    fake = np.concatenate(fakes, 0)[:n_gen]
    fake01 = np.clip((fake + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    print(f"[gen] fake {fake01.shape} range=[{fake01.min():.3f},{fake01.max():.3f}]")

    # roughness diagnostic (mean abs 2nd-difference): the metric we set out to fix
    def roughness(a):
        return float(np.mean(np.abs(np.diff(a, n=2, axis=1))))
    rough_real, rough_fake = roughness(data01), roughness(fake01)

    disc, pred = [], []
    for s in range(args.eval_seeds):
        disc.append(discriminative_score(data01, fake01, device=device, seed=s))
        pred.append(predictive_score(data01, fake01, device=device, seed=s))
        print(f"[eval seed {s}] disc={disc[-1]:.4f} pred={pred[-1]:.4f}")

    result = {
        "dataset": args.dataset,
        "variant": "decomp",
        "seq_len": L, "channels": C, "n_train": int(cut), "n_gen": int(n_gen),
        "steps": steps, "smoke": args.smoke, "model_size": args.model_size,
        "schedule": args.schedule, "min_snr_gamma": args.min_snr_gamma,
        "n_poly": args.n_poly, "n_freq": args.n_freq,
        "params_M": n_params / 1e6, "train_time_sec": train_time,
        "roughness_real": rough_real, "roughness_fake": rough_fake,
        "roughness_ratio": rough_fake / (rough_real + 1e-12),
        "discriminative_mean": float(np.mean(disc)),
        "discriminative_std": float(np.std(disc)),
        "predictive_mean": float(np.mean(pred)),
        "predictive_std": float(np.std(pred)),
        "discriminative_per_seed": [float(x) for x in disc],
        "predictive_per_seed": [float(x) for x in pred],
        "loss_log": loss_log,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    if args.save_model:
        torch.save(ema.state_dict(), args.save_model)
        npy_path = args.save_model.rsplit(".", 1)[0] + "_fake.npy"
        np.save(npy_path, fake01)
        print(f"[save] EMA -> {args.save_model}  fake -> {npy_path}")

    print(f"\n[done] {args.dataset} -> {args.out}")
    print(f"   disc={np.mean(disc):.4f}±{np.std(disc):.4f} "
          f"pred={np.mean(pred):.4f}±{np.std(pred):.4f}  "
          f"roughness fake/real={rough_fake:.5f}/{rough_real:.5f} ({rough_fake/(rough_real+1e-12):.2f}x)")


if __name__ == "__main__":
    main()
