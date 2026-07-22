"""
Large empirical Heston reference dataset — GPU generator.

Generates a very large bank of Heston paths (same SDE, same parameters,
same dt=1/250, same seq_len=128 as the benchmark training/test sets) to serve
as a high-resolution *empirical* reference for the diagnostic "theory" curves
(the panels with no clean closed form: ACF|r|, rolling-vol histogram, and any
curve validated by high-res Monte Carlo).

Scheme: full-truncation Euler-Maruyama, identical to generate_heston.py, run in
float32 on GPU and vectorised over millions of paths per step.

Each worker writes ONE shard:
    large_S_shard{k}.npy   (n, 128) float32   price paths
    large_v_shard{k}.npy   (n, 128) float32   variance paths

Usage (one process per GPU, launched by generate_large_launch.sh):
    CUDA_VISIBLE_DEVICES=0 python generate_heston_large.py \
        --n 25000000 --chunk 2000000 --seed 100 --shard 0 --outdir large_dataset
"""
import argparse
import os
import time

import numpy as np
import torch

MU = 0.05; KAPPA = 2.0; THETA = 0.04; XI = 0.3; RHO = -0.7
S0 = 100.0; V0 = 0.04; DT = 1.0 / 250.0
SEQ_LEN = 128


def generate_chunk(n: int, seed: int, device: str) -> tuple:
    """Generate `n` Heston paths of length SEQ_LEN on `device`, float32.

    Returns (S, v) as float32 numpy arrays of shape (n, SEQ_LEN).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    T = SEQ_LEN
    sqrt_dt = float(np.sqrt(DT))
    rho = RHO
    sqrt_1mrho2 = float(np.sqrt(1.0 - RHO ** 2))

    S = torch.empty((n, T), dtype=torch.float32, device=device)
    v = torch.empty((n, T), dtype=torch.float32, device=device)
    S[:, 0] = S0
    v[:, 0] = V0

    S_prev = S[:, 0].clone()
    v_prev = v[:, 0].clone()
    for t in range(1, T):
        z1 = torch.randn(n, generator=g, device=device, dtype=torch.float32)
        z2 = torch.randn(n, generator=g, device=device, dtype=torch.float32)
        z_s = z1
        z_v = rho * z1 + sqrt_1mrho2 * z2

        v_plus = torch.clamp(v_prev, min=0.0)
        sqrt_vp = torch.sqrt(v_plus)
        v_new = torch.clamp(
            v_prev + KAPPA * (THETA - v_plus) * DT + XI * sqrt_vp * sqrt_dt * z_v,
            min=0.0,
        )
        S_new = S_prev + MU * S_prev * DT + sqrt_vp * S_prev * sqrt_dt * z_s

        S[:, t] = S_new
        v[:, t] = v_new
        S_prev = S_new
        v_prev = v_new

    return S.cpu().numpy(), v.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="total paths for this worker")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, required=True, help="base seed for this worker")
    ap.add_argument("--shard", type=int, required=True, help="shard index k")
    ap.add_argument("--outdir", default="large_dataset")
    ap.add_argument("--benchmark", action="store_true",
                    help="generate one chunk, print throughput, do not save")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    if args.benchmark:
        n = min(args.chunk, args.n)
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        S, v = generate_chunk(n, args.seed, device)
        torch.cuda.synchronize() if device == "cuda" else None
        dt = time.perf_counter() - t0
        print(f"[benchmark] {n:,} paths in {dt:.2f}s -> {n/dt:,.0f} paths/s on {device}")
        print(f"[benchmark] S mean(S0)={S[:,0].mean():.2f}  v mean(v0)={v[:,0].mean():.5f}")
        print(f"[benchmark] bytes/path (S+v f32) = {2*SEQ_LEN*4} B")
        return

    S_path = os.path.join(outdir, f"large_S_shard{args.shard}.npy")
    v_path = os.path.join(outdir, f"large_v_shard{args.shard}.npy")

    S_mm = np.lib.format.open_memmap(
        S_path, mode="w+", dtype=np.float32, shape=(args.n, SEQ_LEN))
    v_mm = np.lib.format.open_memmap(
        v_path, mode="w+", dtype=np.float32, shape=(args.n, SEQ_LEN))

    done = 0
    ci = 0
    t_start = time.perf_counter()
    while done < args.n:
        n = min(args.chunk, args.n - done)
        S, v = generate_chunk(n, args.seed + ci, device)
        S_mm[done:done + n] = S
        v_mm[done:done + n] = v
        done += n
        ci += 1
        el = time.perf_counter() - t_start
        print(f"[shard {args.shard}] {done:,}/{args.n:,} paths  "
              f"({done/el:,.0f}/s, {el:.0f}s elapsed)", flush=True)
    S_mm.flush(); v_mm.flush()
    print(f"[shard {args.shard}] DONE {args.n:,} paths -> {S_path}", flush=True)


if __name__ == "__main__":
    main()
