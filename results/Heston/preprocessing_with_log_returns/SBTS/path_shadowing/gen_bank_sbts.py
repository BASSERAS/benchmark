"""
Generate + PERSIST the 1,000,000-path scenario bank for SBTS+logret (seed 0).

The full PDF path-shadowing protocol (arXiv:2308.01486) needs a bank-size sweep
(4096; 16384; 65536; 262144; 1,000,000) taken as *nested prefixes of the same
one-million-path bank* (PDF §5.2). That requires the seed-0 bank on disk, so
this script materialises it once.

SBTS is non-parametric — the bank is just generate_paths(M_simu=1e6, seed=0)
with the author hyperparameters (h=0.05, K=20, N_pi=50). Reuses the untouched
methods/SBTS/code/sbts_generate.generate_paths; nothing under methods/ is edited.

Usage (sbts-venv; SBTS is CPU-only):
  SBTS_NWORK=100 python gen_bank_sbts.py --seed 0
  # skips if bank/generated_bank_seed{i}_1000000x128.npy already exists
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                 # .../SBTS/path_shadowing
METHOD_DIR = os.path.dirname(HERE)                                # .../SBTS
BENCH_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns")
BANK_DIR = os.path.join(HERE, "bank")
SEQ_LEN = 128

SBTS_SRC = os.path.join(BENCH_ROOT, "methods", "SBTS", "code")
sys.path.insert(0, SBTS_SRC)
from sbts_generate import generate_paths, warmup_jit   # noqa: E402

H_BW, K_MARKOV, N_PI = 0.05, 20, 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bank_size", type=int, default=1_000_000)
    ap.add_argument("--train", default=os.path.join(DATA_DIR, "heston_S_4096x128.npy"))
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SBTS_NWORK", "100")))
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

    S_train = np.load(a.train).astype(np.float64)
    print(f"=== gen SBTS bank seed={a.seed} size={a.bank_size} "
          f"train={S_train.shape} workers={a.workers} ===", flush=True)

    warmup_jit()
    S_gen, meta = generate_paths(
        S_train, a.bank_size, H_BW,
        K=K_MARKOV, N_pi=N_PI, n_workers=a.workers, seed=a.seed)

    np.save(out, S_gen.astype(np.float64))
    print(f"[seed {a.seed}] saved bank {S_gen.shape} -> {out} "
          f"({meta['elapsed_sec']}s, min={meta['min_val']:.2f} "
          f"max={meta['max_val']:.2f})", flush=True)


if __name__ == "__main__":
    main()
