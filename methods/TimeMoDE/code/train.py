"""TimeMoDE Heston -- 5-seed orchestrator.

Launches `train_seed.py` once per seed, two seeds at a time on two GPUs, each
pinned to 8 physical cores (respects the machine's hard limits: max 2 GPUs,
16 cores). Every run uses the DEFAULT hyper-parameters of train_seed.py, which
ARE the SLC seed-0 values -- so all five Heston models are the exact same
architecture that passed the reproduction gate; only --seed differs.

Usage:
  /home/tbasseras/gpu-venv/bin/python train.py                 # seeds 0..4, GPUs 1,2
  /home/tbasseras/gpu-venv/bin/python train.py --gpus 1 2 --seeds 5
  /home/tbasseras/gpu-venv/bin/python train.py --smoke         # 30-epoch sanity, all seeds
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def parse_args():
    p = argparse.ArgumentParser(description="TimeMoDE Heston 5-seed orchestrator")
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (0..seeds-1)")
    p.add_argument("--gpus", type=int, nargs="+", default=[1, 2],
                   help="GPU ids to use (max 2, machine is shared)")
    p.add_argument("--cores_per_gpu", type=int, default=8)
    p.add_argument("--core_base", type=int, default=8,
                   help="first physical core to pin (cores 8..8+2*cpg-1)")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def launch(seed, gpu, core_lo, core_hi, smoke):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = str(core_hi - core_lo + 1)
    cmd = ["taskset", "-c", f"{core_lo}-{core_hi}", PY,
           os.path.join(HERE, "train_seed.py"), "--seed", str(seed)]
    if smoke:
        cmd.append("--smoke")
    log = os.path.join(HERE, "..", "losses", f"train_seed{seed}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    print(f"[launch] seed={seed} gpu={gpu} cores={core_lo}-{core_hi} -> {log}", flush=True)
    f = open(log, "w")
    return subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT), f


def main():
    a = parse_args()
    gpus = a.gpus[:2]                    # hard cap: 2 GPUs
    cpg = a.cores_per_gpu
    core_ranges = [(a.core_base + i * cpg, a.core_base + i * cpg + cpg - 1)
                   for i in range(len(gpus))]

    seeds = list(range(a.seeds))
    # process in waves of len(gpus): seed i -> gpu i%len(gpus)
    for wave_start in range(0, len(seeds), len(gpus)):
        wave = seeds[wave_start:wave_start + len(gpus)]
        procs = []
        for j, seed in enumerate(wave):
            gpu = gpus[j]
            lo, hi = core_ranges[j]
            procs.append((seed, *launch(seed, gpu, lo, hi, a.smoke)))
        for seed, p, f in procs:
            rc = p.wait()
            f.close()
            print(f"[done] seed={seed} rc={rc}", flush=True)
            if rc != 0:
                print(f"[ERROR] seed {seed} exited {rc}; see log", flush=True)


if __name__ == "__main__":
    main()
