"""
Generate + PERSIST the 1,000,000-path scenario bank for each LS4+logret seed.

The full PDF path-shadowing protocol needs a bank-size sweep
(4096; 16384; 65536; 262144; 1,000,000) taken as *nested prefixes of the same
one-million-path bank* (PDF §5.2). That requires all five seed banks on disk, so
this script materialises them once. Bank RNG = manual_seed(1000+seed), identical
to path_shadowing_mc.build_bank, so banks are reproducible and seed-distinct.

Reuses build_bank() from path_shadowing_mc.py verbatim (LS4 prior sampling +
frozen-sigma SBTS inverse). Nothing under methods/ is touched.

Usage (one seed per process, pin cores, pick GPU):
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
      python gen_banks.py --seed 1
  # skips if bank/generated_bank_seed{i}_1000000x128.npy already exists
"""
import os
import sys
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from path_shadowing_mc import build_bank, BANK_DIR, SEQ_LEN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--bank_size", type=int, default=1_000_000)
    ap.add_argument("--gen_batch", type=int, default=8192)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    os.makedirs(BANK_DIR, exist_ok=True)
    out = os.path.join(
        BANK_DIR, f"generated_bank_seed{a.seed}_{a.bank_size}x{SEQ_LEN}.npy"
    )
    if os.path.exists(out) and not a.overwrite:
        arr = np.load(out, mmap_mode="r")
        print(f"[seed {a.seed}] bank exists {arr.shape} -> skip ({out})", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"=== gen bank seed={a.seed} size={a.bank_size} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} "
          f"device={dev_name} ===", flush=True)

    S, build_sec, sigma = build_bank(a.seed, a.bank_size, a.gen_batch, device)
    assert S.shape == (a.bank_size, SEQ_LEN), S.shape
    assert not np.isnan(S).any(), "NaN in bank"
    np.save(out, S)
    gb = os.path.getsize(out) / 1e9
    print(f"[seed {a.seed}] saved bank {S.shape} in {build_sec:.0f}s "
          f"price[min={S.min():.2f},max={S.max():.2f}] sigma={sigma:.8f} "
          f"-> {out} ({gb:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
