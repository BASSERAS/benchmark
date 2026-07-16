"""
Single-seed TimeGAN training script.
CUDA_VISIBLE_DEVICES must be set in the environment before launching this script
(done by train.py via subprocess env). Do NOT import torch at module level.
"""
import argparse, csv, json, os, sys, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",              type=int, required=True)
    ap.add_argument("--n-samples",         type=int, default=8192)
    ap.add_argument("--hidden-dim",        type=int, default=24)
    ap.add_argument("--num-layers",        type=int, default=3)
    ap.add_argument("--batch-size",        type=int, default=128)
    ap.add_argument("--embedding-steps",   type=int, default=5000)
    ap.add_argument("--supervised-steps",  type=int, default=5000)
    ap.add_argument("--joint-steps",       type=int, default=10000)
    ap.add_argument("--log-every",         type=int, default=100)
    args = ap.parse_args()

    # torch import AFTER env is set (CUDA_VISIBLE_DEVICES comes from parent env)
    import torch
    sys.path.insert(0, SCRIPT_DIR)
    from timegan_torch import TimeGAN

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"=== Seed {args.seed}  CUDA_VISIBLE_DEVICES={cvd}  device={gpu_info} ===", flush=True)

    # ── Load dataset ──────────────────────────────────────────────────────
    ds_path = os.path.join(REPO_ROOT, "dataset", "heston_S_8192x128.npy")
    S = np.load(ds_path)   # (8192, 128)
    print(f"Dataset {S.shape}  S0_mean={S[:,0].mean():.2f}", flush=True)

    # ── Train ─────────────────────────────────────────────────────────────
    model = TimeGAN(
        n_features=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        embedding_steps=args.embedding_steps,
        supervised_steps=args.supervised_steps,
        joint_steps=args.joint_steps,
        log_every=args.log_every,
    )
    t0 = time.perf_counter()
    model.fit(S)
    elapsed = time.perf_counter() - t0
    print(f"Training done in {elapsed:.1f}s", flush=True)

    # ── Save results ──────────────────────────────────────────────────────
    results  = os.path.join(SCRIPT_DIR, "results")
    seed_tag = f"seed_{args.seed}"

    # 1. Generated paths
    gen_dir  = os.path.join(results, "generated_paths", seed_tag)
    os.makedirs(gen_dir, exist_ok=True)
    paths = model.sample(args.n_samples)            # (8192, 128)
    npy_path = os.path.join(gen_dir, f"generated_paths_{args.n_samples}x128.npy")
    np.save(npy_path, paths)
    meta = {
        "seed": args.seed, "n_samples": args.n_samples, "seq_len": 128,
        "min_val": float(model.min_val[0]), "max_val": float(model.max_val[0]),
        "train_time_sec": round(elapsed, 2),
        "generated_mean": float(paths.mean()), "generated_std": float(paths.std()),
        "real_mean": float(S.mean()), "real_std": float(S.std()),
    }
    with open(os.path.join(gen_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Paths saved {paths.shape}  mean={paths.mean():.2f}", flush=True)

    # 2. Model params
    params_dir = os.path.join(results, "params")
    os.makedirs(params_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(params_dir, f"{seed_tag}_model.pt"))
    cfg = model.config(); cfg["seed"] = args.seed; cfg["train_time_sec"] = round(elapsed, 2)
    with open(os.path.join(params_dir, f"{seed_tag}_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # 3. Loss history
    losses_dir = os.path.join(results, "losses")
    os.makedirs(losses_dir, exist_ok=True)
    loss_path = os.path.join(losses_dir, f"{seed_tag}_losses.csv")
    fields = ["step","phase","e_loss","s_loss","g_loss","d_loss"]
    with open(loss_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in model.loss_history:
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"Losses saved {len(model.loss_history)} rows -> {loss_path}", flush=True)
    print(f"=== Seed {args.seed} DONE ===", flush=True)


if __name__ == "__main__":
    main()
