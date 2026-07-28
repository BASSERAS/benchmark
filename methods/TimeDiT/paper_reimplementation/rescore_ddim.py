"""Quick ablation: does deterministic DDIM sampling raise temporal autocorrelation
(and lower the discriminative score) vs ancestral DDPM sampling — from the SAME
already-trained EMA weights (no retraining)?

Loads a saved EMA state_dict, regenerates samples under several samplers, and
prints lag-1 autocorrelation + Yoon discriminative/predictive scores for each.
"""
import argparse
import numpy as np
import torch

from data_paper import load_dataset
from gaussian_diffusion import GaussianDiffusion
from timedit_model import build_timedit
from yoon_metrics import discriminative_score, predictive_score


def lag1_autocorr(x):
    x = x - x.mean(1, keepdims=True)
    num = (x[:, 1:] * x[:, :-1]).sum(1)
    den = (x * x).sum(1) + 1e-9
    return float((num / den).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="stock")
    p.add_argument("--stock_csv", default="/home/tbasseras/benchmark/methods/DiffusionTS/code/reference/Data/datasets/stock_data.csv")
    p.add_argument("--weights", default="ema_stock.pt")
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval_seeds", type=int, default=3)
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    data = load_dataset(args.dataset, seq_len=args.seq_len, stock_csv=args.stock_csv)
    N, L, C = data.shape
    print(f"[data] {args.dataset} {data.shape} real lag1={lag1_autocorr(data):.4f}")

    model = build_timedit(L, C, model_size="S", learn_sigma=True).to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    diff = GaussianDiffusion(T=1000, schedule="linear", loss_mode="hybrid")

    def gen(fn):
        outs = []
        rem = N
        with torch.no_grad():
            while rem > 0:
                b = min(1024, rem)
                outs.append(fn(b).cpu().numpy())
                rem -= b
        f = np.concatenate(outs, 0)[:N]
        return np.clip(f, 0.0, 1.0).astype(np.float32)

    samplers = {
        "ancestral_learned": lambda b: diff.p_sample_loop(model, (b, L, C), device, learn_sigma=True, sample_var="learned"),
        "ancestral_fixed":   lambda b: diff.p_sample_loop(model, (b, L, C), device, learn_sigma=True, sample_var="fixed"),
        "ddim_eta0_1000":    lambda b: diff.ddim_sample_loop(model, (b, L, C), device, learn_sigma=True, steps=1000, eta=0.0),
        "ddim_eta0_100":     lambda b: diff.ddim_sample_loop(model, (b, L, C), device, learn_sigma=True, steps=100, eta=0.0),
    }

    for name, fn in samplers.items():
        fake = gen(fn)
        disc = [discriminative_score(data, fake, device=device, seed=s) for s in range(args.eval_seeds)]
        pred = [predictive_score(data, fake, device=device, seed=s) for s in range(args.eval_seeds)]
        print(f"[{name:18s}] lag1={lag1_autocorr(fake):.4f} "
              f"disc={np.mean(disc):.4f}±{np.std(disc):.4f} "
              f"pred={np.mean(pred):.4f}±{np.std(pred):.4f}")


if __name__ == "__main__":
    main()
