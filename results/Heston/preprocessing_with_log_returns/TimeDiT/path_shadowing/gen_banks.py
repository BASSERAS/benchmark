"""
Generate + PERSIST the 1,000,000-path scenario bank for TimeDiT -- BOTH variants.

The full PDF path-shadowing protocol needs a bank-size sweep
(4096; 16384; 65536; 262144; 1,000,000) taken as *nested prefixes of the same
one-million-path bank* (PDF §5.2 / ../../GUIDELINE.md §9.5). That requires the
seed-0 bank on disk, so this script materialises it once per variant:

  --variant preproc : ../weights/seed_0_model.pt
                      SBTS log-return front-end -> inverse is cumsum-exp @ S0=100
                      bank -> ./bank/
  --variant raw     : ../baseline_no_preproc/weights/seed_0_model.pt
                      raw-price front-end       -> inverse is minmax^-1 only
                      bank -> ../baseline_no_preproc/path_shadowing/bank/

Theo's instruction, verbatim: "for the metrics A and B pls use the SBTS prepro
variant but for the path shadowing pls use do both with no propoc and with
propoc so u have to save two files of 1M files inside the dsik pls".

Output dtype is float32 (0.51 GB), identical to the CSDI and LS4 banks, so
path_shadowing_pdf.py loads it unchanged.

Unlike CSDI/LS4 -- whose banks took ~23 min -- exact DDPM at T=1000 needs ~33 h
per bank on one A100, so this script writes straight into a .npy memmap and
checkpoints after every chunk: a run that dies at hour 30 resumes at hour 30.

SHARDING (Theo, verbatim: "generate it on two GPUs in parrallel so it shoukld
take about 17h"). --n_shards N / --shard S splits the work across N processes,
one GPU each, all writing disjoint row ranges of the SAME memmap:

  * the bank is cut into ceil(bank_size/gen_batch) chunks; chunk c always owns
    rows [c*gen_batch, (c+1)*gen_batch) and is always seeded with the *global*
    chunk index c, exactly as in the 1-shard run;
  * shard S takes the chunks with c % N == S (round-robin, so both shards get
    the same count even when the last chunk is ragged).

Consequence: the finished bank is BIT-IDENTICAL to a single-process run with
the same --gen_batch and TF32 setting. Sharding buys wall-clock, it does not
change a single sampled path. Each shard keeps its own progress file and
resumes independently; whichever shard finishes last writes the combined
meta_seed{N}.json.

Sampler: EXACT DDPM (p_sample_loop, T=1000, sample_var="fixed"), never DDIM --
the HPO study measured DDIM-100 at 4.7x worse loss at the winning config and it
collapses lag-1 autocorrelation (0.344 -> 0.043 vs real 0.689), precisely the
statistic the PS embedding weights most (rolling-vol w=2.0, ACF w=1.0). TF32
matmuls are on (3.0x: 98h -> 33h) only because verify_tf32.py + compare_tf32.py
showed the distribution shift stays under the seed-to-seed noise floor.

Nothing under methods/ is touched: build_timedit / GaussianDiffusion are
imported verbatim.

Usage (ONE variant across two GPUs -- Theo's order: raw first, then preproc):
  CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 taskset -c 24-31 \
      python -u gen_banks.py --variant raw --n_shards 2 --shard 0
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 8-15 \
      python -u gen_banks.py --variant raw --n_shards 2 --shard 1
  # shard 0 creates the memmap; other shards wait for it, then open r+
  # resumes automatically; --overwrite (on shard 0) forces a clean rebuild
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

VARIANTS = {
    # variant -> (checkpoint, bank dir, rng offset)
    "preproc": (os.path.join(TIMEDIT_DIR, "weights", "seed_0_model.pt"),
                os.path.join(HERE, "bank"), 0),
    "raw": (os.path.join(TIMEDIT_DIR, "baseline_no_preproc", "weights", "seed_0_model.pt"),
            os.path.join(TIMEDIT_DIR, "baseline_no_preproc", "path_shadowing", "bank"), 500),
}


def inverse_preproc(fz, ck):
    """model space -> price, SBTS variant. Verbatim from ../code/train_timedit_logret.py L188-196."""
    X01 = np.clip(fz[:, :, 0] * ck["znorm_sd"] + ck["znorm_mu"], 0.0, 1.0)
    X_sbts = X01 * (ck["minmax_hi"] - ck["minmax_lo"]) + ck["minmax_lo"]
    R = X_sbts[:, 1:] * ck["sbts_sigma"] / np.sqrt(DT)       # drop dummy col, unscale
    S = np.empty((fz.shape[0], SEQ_LEN), dtype=np.float64)
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(np.cumsum(R, axis=1))
    return np.clip(S, 1e-6, None)


def inverse_raw(fz, ck):
    """model space -> price, no-preproc variant. Verbatim from baseline_no_preproc/code/train_timedit_raw.py L196-198."""
    X01 = np.clip(fz[:, :, 0] * ck["znorm_sd"] + ck["znorm_mu"], 0.0, 1.0)
    S = X01 * (ck["minmax_hi"] - ck["minmax_lo"]) + ck["minmax_lo"]
    return np.clip(S, 1e-6, None).astype(np.float64)


def progress_path(bank_dir, seed, shard, n_shards):
    """1-shard keeps the original filename so existing runs stay resumable."""
    if n_shards == 1:
        return os.path.join(bank_dir, f"progress_seed{seed}.json")
    return os.path.join(bank_dir, f"progress_seed{seed}_shard{shard}of{n_shards}.json")


def shard_chunks(bank_size, gen_batch, shard, n_shards):
    """Global chunk indices owned by this shard (round-robin => balanced)."""
    n_chunks = (bank_size + gen_batch - 1) // gen_batch
    return [c for c in range(n_chunks) if c % n_shards == shard]


def all_shards_complete(bank_dir, seed, n_shards):
    secs = []
    for s in range(n_shards):
        p = progress_path(bank_dir, seed, s, n_shards)
        if not os.path.exists(p):
            return False, {}
        with open(p) as f:
            pr = json.load(f)
        if pr.get("chunks_done", 0) < pr.get("chunks_total", 1):
            return False, {}
        secs.append(float(pr.get("elapsed_sec", 0.0)))
    return True, {"per_shard_sec": [round(x, 1) for x in secs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bank_size", type=int, default=1_000_000)
    ap.add_argument("--gen_batch", type=int, default=8192)
    ap.add_argument("--n_shards", type=int, default=1,
                    help="split the bank across N processes / GPUs")
    ap.add_argument("--shard", type=int, default=0,
                    help="which shard this process owns (0..N-1)")
    ap.add_argument("--no_tf32", action="store_true",
                    help="disable TF32 matmuls (3x slower; audit rerun only)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if not (0 <= a.shard < a.n_shards):
        raise SystemExit(f"--shard {a.shard} out of range for --n_shards {a.n_shards}")

    use_tf32 = not a.no_tf32
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    ckpt_path, bank_dir, rng_off = VARIANTS[a.variant]
    os.makedirs(bank_dir, exist_ok=True)
    out = os.path.join(bank_dir,
                       f"generated_bank_seed{a.seed}_{a.bank_size}x{SEQ_LEN}.npy")
    prog_path = progress_path(bank_dir, a.seed, a.shard, a.n_shards)

    my_chunks = shard_chunks(a.bank_size, a.gen_batch, a.shard, a.n_shards)
    my_rows = sum(min(a.gen_batch, a.bank_size - c * a.gen_batch) for c in my_chunks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== gen bank variant={a.variant} seed={a.seed} size={a.bank_size} "
          f"shard={a.shard}/{a.n_shards} chunks={len(my_chunks)} rows={my_rows} "
          f"gen_batch={a.gen_batch} tf32={use_tf32} sampler=ddpm_fixed_T1000 "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')} "
          f"device={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'} ===",
          flush=True)
    print(f"[ckpt] {os.path.relpath(ckpt_path, BENCH_ROOT)}", flush=True)
    print(f"[out ] {os.path.relpath(out, BENCH_ROOT)}", flush=True)

    # ---- resume state (per shard) ----
    chunks_done, prev_sec = 0, 0.0
    if a.overwrite:
        if a.shard == 0 and os.path.exists(out):
            os.remove(out)
        if os.path.exists(prog_path):
            os.remove(prog_path)
    elif os.path.exists(prog_path):
        with open(prog_path) as f:
            pr = json.load(f)
        if (pr.get("total") == a.bank_size and pr.get("gen_batch") == a.gen_batch
                and pr.get("n_shards", 1) == a.n_shards):
            chunks_done = int(pr.get("chunks_done", 0))
            prev_sec = float(pr.get("elapsed_sec", 0.0))
            print(f"[resume] shard {a.shard}: {chunks_done}/{len(my_chunks)} chunks "
                  f"({prev_sec/3600:.2f} h spent)", flush=True)
        else:
            print("[resume] progress file incompatible (size/batch/shards changed) "
                  "-- restarting this shard from scratch", flush=True)

    # ---- memmap: shard 0 creates it, the others wait for it ----
    nbytes = a.bank_size * SEQ_LEN * 4
    if a.shard == 0:
        if not os.path.exists(out):
            m = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32,
                                          shape=(a.bank_size, SEQ_LEN))
            m.flush()
            del m
            print(f"[init] allocated {nbytes/1e9:.3f} GB memmap", flush=True)
    else:
        waited = 0
        while not (os.path.exists(out) and os.path.getsize(out) >= nbytes):
            time.sleep(5)
            waited += 5
            if waited > 3600:
                raise SystemExit("shard 0 never created the bank file (waited 1 h)")
        if waited:
            print(f"[wait] shard 0 created the memmap after {waited}s", flush=True)
    bank = np.lib.format.open_memmap(out, mode="r+")
    assert bank.shape == (a.bank_size, SEQ_LEN), bank.shape

    if chunks_done >= len(my_chunks):
        print(f"[skip] shard {a.shard} already complete", flush=True)
    else:
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = build_timedit(SEQ_LEN, FEAT, model_size="S", learn_sigma=False).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        diff = GaussianDiffusion(T=1000, schedule="linear", loss_mode="hybrid")
        inverse = inverse_preproc if a.variant == "preproc" else inverse_raw

        base_seed = 1000 + a.seed + rng_off      # distinct per seed AND per variant
        rows_done = sum(min(a.gen_batch, a.bank_size - c * a.gen_batch)
                        for c in my_chunks[:chunks_done])
        t0 = time.time()
        with torch.no_grad():
            for i in range(chunks_done, len(my_chunks)):
                c = my_chunks[i]                 # GLOBAL chunk index -> same seed as 1-shard
                row0 = c * a.gen_batch
                b = min(a.gen_batch, a.bank_size - row0)
                torch.manual_seed(base_seed * 1_000_003 + c)
                torch.cuda.manual_seed_all(base_seed * 1_000_003 + c)
                x = diff.p_sample_loop(model, (b, SEQ_LEN, FEAT), device,
                                       learn_sigma=False, sample_var="fixed")
                bank[row0:row0 + b] = inverse(x.float().cpu().numpy(), ck).astype(np.float32)
                bank.flush()
                rows_done += b
                tot = prev_sec + (time.time() - t0)
                rate = rows_done / tot
                with open(prog_path, "w") as f:
                    json.dump({"shard": a.shard, "n_shards": a.n_shards,
                               "chunks_done": i + 1, "chunks_total": len(my_chunks),
                               "rows_done": rows_done, "rows_total": my_rows,
                               "total": a.bank_size, "gen_batch": a.gen_batch,
                               "tf32": use_tf32, "elapsed_sec": round(tot, 1)}, f)
                print(f"[bank s{a.shard}] chunk {i+1}/{len(my_chunks)} "
                      f"rows {rows_done}/{my_rows} ({100*rows_done/my_rows:.2f}%) "
                      f"rate={rate:.1f} paths/s "
                      f"eta={(my_rows-rows_done)/rate/3600:.2f} h", flush=True)

    # ---- finalise once EVERY shard reports complete ----
    complete, extra = all_shards_complete(bank_dir, a.seed, a.n_shards)
    if not complete:
        print(f"[wait] shard {a.shard} done; other shards still running -- "
              f"meta_seed{a.seed}.json will be written by the last one", flush=True)
        return

    wall = max(extra["per_shard_sec"]) if extra["per_shard_sec"] else 0.0
    sample = np.asarray(bank[::997], dtype=np.float64)   # coprime stride, ~1004 rows
    meta = {"variant": a.variant, "seed": a.seed, "bank_size": a.bank_size,
            "gen_batch": a.gen_batch, "n_shards": a.n_shards,
            "per_shard_sec": extra["per_shard_sec"],
            "tf32": use_tf32, "sampler": "ddpm_fixed_T1000",
            "dtype": "float32", "total_sec": round(wall, 1),
            "hours": round(wall / 3600, 2),
            "paths_per_sec": round(a.bank_size / wall, 3) if wall else None,
            "gb_on_disk": round(os.path.getsize(out) / 1e9, 3),
            "sample_price_min": float(sample.min()), "sample_price_max": float(sample.max()),
            "sample_price_mean": float(sample.mean()), "sample_price_std": float(sample.std()),
            "sample_has_nan": bool(not np.isfinite(sample).all()),
            "ckpt": os.path.relpath(ckpt_path, BENCH_ROOT)}
    with open(os.path.join(bank_dir, f"meta_seed{a.seed}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()
