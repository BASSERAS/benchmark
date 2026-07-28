"""TimeDiT paper reproduction driver — synthetic generation (Yoon 2019 benchmark).

Trains the from-scratch TimeDiT (DiT-S + DDPM learned variance + Time Series Mask
Unit) unconditionally (M^Rec = 0) on a length-24 dataset, then evaluates the
generated samples with the two metrics TimeDiT reports in Table 6:

  discriminative score = |accuracy - 0.5|   (lower = better; ideal 0)
  predictive score     = TSTR MAE           (lower = better)

Paper protocol (Table 9 "S" + Appendix C/E): model size S (hidden 384, depth 12,
heads 6), full loss "L" (learn_sigma = hybrid vlb), Adam lr 1e-4, no weight decay,
batch 512, MinMax scaling, A100.  Train on 80% split, evaluate on the same set.

Usage
-----
Smoke test (5% data, 300 steps, quick NaN / loss-decrease check):
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python run_paper.py --dataset sine --smoke

Full reproduction:
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python run_paper.py --dataset sine \
        --steps 15000 --batch 512 --lr 1e-4 --out results_sine.json
    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 python run_paper.py --dataset stock \
        --stock_csv /path/to/stock_data.csv --steps 15000 --out results_stock.json
"""
import argparse
import copy
import json
import time

import numpy as np
import torch

from data_paper import load_dataset
from gaussian_diffusion import GaussianDiffusion
from timedit_model import build_timedit
from yoon_metrics import discriminative_score, predictive_score


def ema_update(ema_model, model, decay):
    with torch.no_grad():
        for e, p in zip(ema_model.parameters(), model.parameters()):
            e.mul_(decay).add_(p, alpha=1 - decay)
        for eb, b in zip(ema_model.buffers(), model.buffers()):
            eb.copy_(b)


def parse_args():
    p = argparse.ArgumentParser(description="TimeDiT paper reproduction (synthetic generation)")
    p.add_argument("--dataset", choices=["sine", "stock"], default="sine")
    p.add_argument("--stock_csv", default=None, help="path to stock_data.csv (required for --dataset stock)")
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--model_size", choices=["S", "B", "L"], default="S")
    p.add_argument("--T", type=int, default=1000, help="diffusion steps")
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--learn_sigma", type=int, default=1, help="1 = full loss L (learned variance)")
    p.add_argument("--steps", type=int, default=15000, help="training iterations")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--n_gen", type=int, default=0, help="#samples to generate (0 = match train N)")
    p.add_argument("--eval_seeds", type=int, default=3, help="#metric seeds to average")
    p.add_argument("--out", default="results.json")
    p.add_argument("--save_model", default=None,
                   help="path to save EMA state_dict (.pt); also saves fake samples .npy alongside")
    p.add_argument("--sample_var", choices=["learned", "fixed", "both"], default="both",
                   help="reverse-process variance: learned (Improved-DDPM), fixed (DDPM Alg.1 sigma_t), "
                        "or both (score under each to compare)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true", help="5%% data + 300 steps sanity run")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    learn_sigma = bool(args.learn_sigma)

    # ---- data ----
    data = load_dataset(args.dataset, seq_len=args.seq_len, stock_csv=args.stock_csv, seed=args.seed)
    steps = args.steps
    if args.smoke:
        n_sub = max(int(len(data) * 0.05), args.batch)
        data = data[:n_sub]
        steps = 300
    N, L, C = data.shape
    print(f"[data] {args.dataset}: {data.shape}  range=[{data.min():.3f},{data.max():.3f}]  steps={steps}")

    # 80% training split (paper protocol)
    cut = int(N * 0.8)
    train = data[:cut]
    train_t = torch.tensor(train, dtype=torch.float32, device=device)

    # ---- model + diffusion ----
    model = build_timedit(L, C, model_size=args.model_size, learn_sigma=learn_sigma).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    diff = GaussianDiffusion(T=args.T, schedule=args.schedule,
                             loss_mode="hybrid" if learn_sigma else "simple")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] TimeDiT-{args.model_size} params={n_params/1e6:.2f}M learn_sigma={learn_sigma}")

    # ---- train ----
    loss_log = []
    model.train()
    t0 = time.time()
    for step in range(steps):
        idx = np.random.randint(0, len(train_t), args.batch)
        x0 = train_t[idx]
        loss, parts = diff.training_loss(model, x0, learn_sigma=learn_sigma)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        decay = 0.0 if step < 100 else args.ema_decay
        ema_update(ema, model, decay)
        if not np.isfinite(loss.item()):
            raise RuntimeError(f"NaN/Inf loss at step {step}: {parts}")
        if step % 100 == 0 or step == steps - 1:
            loss_log.append({"step": step, "loss": loss.item(),
                             "simple": parts["simple"], "vlb": parts["vlb"]})
            print(f"[{step:5d}] loss={loss.item():.4f} simple={parts['simple']:.4f} "
                  f"vlb={parts['vlb']:.4f} ({time.time()-t0:.1f}s)")
    train_time = time.time() - t0

    # ---- sample + score, per variance mode ----
    n_gen = args.n_gen if args.n_gen > 0 else N
    var_modes = ["learned", "fixed"] if args.sample_var == "both" else [args.sample_var]
    ema.eval()

    def sample(sample_var):
        fakes = []
        with torch.no_grad():
            remaining = n_gen
            while remaining > 0:
                b = min(1024, remaining)
                x = diff.p_sample_loop(ema, (b, L, C), device,
                                       learn_sigma=learn_sigma, sample_var=sample_var)
                fakes.append(x.cpu().numpy())
                remaining -= b
        fk = np.concatenate(fakes, 0)[:n_gen]
        return np.clip(fk, 0.0, 1.0).astype(np.float32)

    by_var = {}
    saved_fake = None
    for vm in var_modes:
        fake = sample(vm)
        print(f"[gen:{vm}] fake {fake.shape} range=[{fake.min():.3f},{fake.max():.3f}]")
        disc, pred = [], []
        for s in range(args.eval_seeds):
            disc.append(discriminative_score(data, fake, device=device, seed=s))
            pred.append(predictive_score(data, fake, device=device, seed=s))
            print(f"[eval:{vm} seed {s}] disc={disc[-1]:.4f} pred={pred[-1]:.4f}")
        by_var[vm] = {
            "discriminative_mean": float(np.mean(disc)),
            "discriminative_std": float(np.std(disc)),
            "predictive_mean": float(np.mean(pred)),
            "predictive_std": float(np.std(pred)),
            "discriminative_per_seed": [float(x) for x in disc],
            "predictive_per_seed": [float(x) for x in pred],
        }
        if saved_fake is None:
            saved_fake = fake  # save the first (learned by default)

    # primary mode for top-level fields (learned if present, else the single mode)
    primary = "learned" if "learned" in by_var else var_modes[0]
    pv = by_var[primary]
    result = {
        "dataset": args.dataset,
        "seq_len": L,
        "channels": C,
        "n_train": int(cut),
        "n_gen": int(n_gen),
        "steps": steps,
        "smoke": args.smoke,
        "model_size": args.model_size,
        "learn_sigma": learn_sigma,
        "params_M": n_params / 1e6,
        "train_time_sec": train_time,
        "primary_sample_var": primary,
        "by_sample_var": by_var,
        "discriminative_mean": pv["discriminative_mean"],
        "discriminative_std": pv["discriminative_std"],
        "predictive_mean": pv["predictive_mean"],
        "predictive_std": pv["predictive_std"],
        "discriminative_per_seed": pv["discriminative_per_seed"],
        "predictive_per_seed": pv["predictive_per_seed"],
        "loss_log": loss_log,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    if args.save_model:
        torch.save(ema.state_dict(), args.save_model)
        npy_path = args.save_model.rsplit(".", 1)[0] + "_fake.npy"
        np.save(npy_path, saved_fake)
        print(f"[save] EMA -> {args.save_model}  fake -> {npy_path}")

    print(f"\n[done] {args.dataset}  -> {args.out}")
    for vm in var_modes:
        b = by_var[vm]
        print(f"   [{vm:7s}] disc={b['discriminative_mean']:.4f}±{b['discriminative_std']:.4f} "
              f"pred={b['predictive_mean']:.4f}±{b['predictive_std']:.4f}")


if __name__ == "__main__":
    main()
