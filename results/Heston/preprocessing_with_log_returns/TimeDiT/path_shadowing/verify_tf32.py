"""TF32-vs-fp32 equivalence check for the TimeDiT DDPM sampler.

WHY: the 1,000,000-path scenario banks (§9.5 of ../../GUIDELINE.md) need exact
DDPM (``p_sample_loop``, T=1000) because DDIM was measured 4.7x worse at the
winning HPO config and collapses lag-1 autocorrelation (0.344 -> 0.043 vs real
0.689), which is precisely the statistic the PS embedding weights most. Exact
DDPM at fp32 costs ~98 h per 1M bank. Enabling TF32 matmuls costs ~33 h for the
*same* algorithm and the *same* RNG stream -- only the matmul mantissa changes.

This script proves that is lossless BEFORE committing ~66 GPU-hours: it samples
4096 paths from the SAME seed-0 checkpoint with the SAME torch.manual_seed and
the SAME batch schedule, once with TF32 off and once with TF32 on, and saves the
two price panels for a statistical comparison against the seed-0/seed-1 spread.

Nothing under methods/ is touched: build_timedit / GaussianDiffusion are imported
verbatim, and the SBTS inverse is copied from ../code/train_timedit_logret.py.

Usage (both modes in parallel, one GPU each):
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7  python -u verify_tf32.py --mode fp32
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 8-15 python -u verify_tf32.py --mode tf32
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))            # .../TimeDiT/path_shadowing
TIMEDIT_DIR = os.path.dirname(HERE)                          # .../TimeDiT
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(TIMEDIT_DIR))))                          # .../benchmark

REFERENCE = os.path.join(BENCH_ROOT, "methods", "TimeDiT", "code")
sys.path.insert(0, REFERENCE)
from timedit_model import build_timedit             # noqa: E402
from gaussian_diffusion import GaussianDiffusion    # noqa: E402

SEQ_LEN = 128
FEAT = 1
DT = 1.0 / 250.0
S0 = 100.0
SAMPLE_SEED = 12345          # identical RNG stream for both modes
OUT_DIR = os.path.join(HERE, "verify")


def load_ckpt(ckpt_path, device):
    # checkpoint holds only tensors + python scalars -> safe strict load
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_timedit(SEQ_LEN, FEAT, model_size="S", learn_sigma=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def sbts_inverse(fz, ck):
    """model space -> price. Verbatim from ../code/train_timedit_logret.py L188-196."""
    mu, sd = ck["znorm_mu"], ck["znorm_sd"]
    lo, hi = ck["minmax_lo"], ck["minmax_hi"]
    sigma = ck["sbts_sigma"]
    X01_g = np.clip(fz[:, :, 0] * sd + mu, 0.0, 1.0)
    X_sbts_gen = X01_g * (hi - lo) + lo
    R_tilde_gen = X_sbts_gen[:, 1:]
    R_gen = R_tilde_gen * sigma / np.sqrt(DT)
    n = fz.shape[0]
    S = np.empty((n, SEQ_LEN), dtype=np.float64)
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(np.cumsum(R_gen, axis=1))
    return np.clip(S, 1e-6, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp32", "tf32"], required=True)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--gen_batch", type=int, default=1024)
    ap.add_argument("--ckpt", default=os.path.join(TIMEDIT_DIR, "weights", "seed_0_model.pt"))
    a = ap.parse_args()

    use_tf32 = (a.mode == "tf32")
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== verify_tf32 mode={a.mode} n={a.n} gen_batch={a.gen_batch} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')} "
          f"device={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'} "
          f"allow_tf32={torch.backends.cuda.matmul.allow_tf32} ===", flush=True)

    model, ck = load_ckpt(a.ckpt, device)
    diff = GaussianDiffusion(T=1000, schedule="linear", loss_mode="hybrid")
    print(f"[ckpt] seed={ck['seed']} sigma={ck['sbts_sigma']:.10f} "
          f"minmax=[{ck['minmax_lo']:.8f},{ck['minmax_hi']:.8f}] "
          f"znorm mu={ck['znorm_mu']:.8f} sd={ck['znorm_sd']:.8f}", flush=True)

    torch.manual_seed(SAMPLE_SEED)
    torch.cuda.manual_seed_all(SAMPLE_SEED)

    chunks = []
    done = 0
    t0 = time.time()
    with torch.no_grad():
        while done < a.n:
            b = min(a.gen_batch, a.n - done)
            x = diff.p_sample_loop(model, (b, SEQ_LEN, FEAT), device,
                                   learn_sigma=False, sample_var="fixed")
            chunks.append(x.float().cpu().numpy())
            done += b
            el = time.time() - t0
            print(f"[gen] {done}/{a.n} elapsed={el:.0f}s rate={done/el:.2f} paths/s", flush=True)
    gen_sec = time.time() - t0

    fz = np.concatenate(chunks, 0)[:a.n]
    S = sbts_inverse(fz, ck)

    out_npy = os.path.join(OUT_DIR, f"gen_{a.mode}_{a.n}x{SEQ_LEN}.npy")
    np.save(out_npy, S)
    meta = {"mode": a.mode, "n": a.n, "gen_batch": a.gen_batch,
            "sample_seed": SAMPLE_SEED, "gen_sec": round(gen_sec, 1),
            "paths_per_sec": round(a.n / gen_sec, 4),
            "eta_1M_hours": round(1_000_000 / (a.n / gen_sec) / 3600, 2),
            "price_min": float(S.min()), "price_max": float(S.max()),
            "price_mean": float(S.mean()), "price_std": float(S.std()),
            "model_space_mean": float(fz.mean()), "model_space_std": float(fz.std()),
            "has_nan": bool(not np.isfinite(S).all()),
            "ckpt": os.path.relpath(a.ckpt, BENCH_ROOT)}
    with open(os.path.join(OUT_DIR, f"meta_{a.mode}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()
