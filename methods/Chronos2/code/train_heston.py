"""
Chronos-2 (Ansari et al., "Chronos-2: From Univariate to Universal Forecasting",
arXiv:2510.15821) -- Heston generation via autoregressive rollout.

Chronos-2 is a *pretrained, inference-only* probabilistic forecaster: given a
context it emits 21 quantile forecasts per future step. It is NOT an
unconditional generator.  To fit the benchmark's `generated_paths_8192x128.npy`
contract we use the user-approved scheme:

    AUTOREGRESSIVE ROLLOUT FROM REAL TEST-SET PREFIXES
    --------------------------------------------------
    - Take the first  `prefix`  steps of each of the 8192 real TEST paths
      (dataset/Heston/heston_S_test_8192x128.npy) as context.
    - Repeatedly forecast the 1-step-ahead predictive quantiles for every
      series, draw ONE sample per series by inverse-CDF sampling over the 21
      trained quantile levels, append it to the context, and continue until the
      path reaches length 128.
    - The generated path is [ real prefix | sampled tail ].  Different seeds =
      different sampling RNG => different tails (the model's own stochasticity).

Two variants are produced (the benchmark keeps the better-scoring one as the
headline method; the other's full artefacts are archived under
methods/Chronos2/<variant>_variant/):

    * zero-shot : the released `amazon/chronos-2` weights, no training.
    * finetune  : `pipeline.fit(...)` a full-finetuned copy on the 8192 Heston
                  TRAIN paths, then roll out with the fine-tuned weights.

Usage:
  # zero-shot, one seed
  CUDA_VISIBLE_DEVICES=0 python train_heston.py --variant zero_shot --seed 0
  # fine-tuned, one seed
  CUDA_VISIBLE_DEVICES=0 python train_heston.py --variant finetune --seed 0
  # strategic smoke test on 5% of data, few finetune steps
  CUDA_VISIBLE_DEVICES=0 python train_heston.py --variant finetune --seed 0 \
        --frac 0.05 --ft_steps 60 --tag smoke
"""
import os
import sys
import csv
import json
import time
import argparse
import numpy as np
import torch

METHOD_CODE = os.path.dirname(os.path.abspath(__file__))            # methods/Chronos2/code
METHOD_DIR = os.path.dirname(METHOD_CODE)                           # methods/Chronos2
BENCH_ROOT = os.path.dirname(os.path.dirname(METHOD_DIR))           # benchmark/
REF_SRC = os.path.join(METHOD_CODE, "reference", "src")
sys.path.insert(0, REF_SRC)                                         # chronos via PYTHONPATH=src
from chronos import BaseChronosPipeline                             # noqa: E402

TRAIN_DATA = os.path.join(BENCH_ROOT, "dataset", "Heston", "heston_S_8192x128.npy")
TEST_DATA = os.path.join(BENCH_ROOT, "dataset", "Heston", "heston_S_test_8192x128.npy")
SEQ_LEN = 128
MODEL_ID = "amazon/chronos-2"


# --------------------------------------------------------------------------- #
#  Fine-tune loss recorder                                                     #
# --------------------------------------------------------------------------- #
def make_loss_recorder():
    from transformers.trainer_callback import TrainerCallback

    class LossRecorder(TrainerCallback):
        def __init__(self):
            self.rows = []

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            if "loss" in logs:                       # periodic training-loss log
                self.rows.append({
                    "step": int(state.global_step),
                    "train_loss": float(logs["loss"]),
                    "lr": float(logs.get("learning_rate", 0.0)),
                })
            elif "train_loss" in logs:               # final summary log (end of training)
                self.rows.append({
                    "step": int(state.global_step),
                    "train_loss": float(logs["train_loss"]),
                    "lr": float(logs.get("learning_rate", 0.0)),
                })
    return LossRecorder()


# --------------------------------------------------------------------------- #
#  Autoregressive rollout from real prefixes                                   #
# --------------------------------------------------------------------------- #
def rollout_from_prefixes(pipeline, prefixes, target_len, seed,
                          batch_size=512, hstep=1, ctx_cap=None, space="log",
                          verbose=True):
    """
    prefixes : (N, prefix_len) float array -- real test-set warm-up context
               (PRICE scale; the transform to `space` is applied internally).
    Returns  : (N, target_len) float array in PRICE scale -- [prefix | sampled tail].

    At each iteration we ask Chronos-2 for `hstep`-step-ahead quantile forecasts
    over the 21 trained quantile levels, then draw one Monte-Carlo sample per
    series/step by inverse-CDF (piecewise-linear) sampling and append it.

    space="log": roll out on log(price).  Heston's S is geometric, so log(S) is a
    near-random-walk with roughly constant increments; this keeps Chronos-2's
    level-proportional predictive scale stable and guarantees positive prices.
    A price-space rollout diverges (multiplicative runaway + negative prices).
    """
    rng = np.random.default_rng(seed)
    q_levels = list(pipeline.quantiles)               # 21 trained levels, ascending
    ql = np.asarray(q_levels)
    N, prefix_len = prefixes.shape
    fwd = np.log if space == "log" else (lambda x: x)
    inv = np.exp if space == "log" else (lambda x: x)
    paths = np.empty((N, target_len), dtype=np.float64)
    paths[:, :prefix_len] = fwd(prefixes)
    t = prefix_len
    while t < target_len:
        h = min(hstep, target_len - t)
        lo = 0 if ctx_cap is None else max(0, t - ctx_cap)
        ctx = paths[:, lo:t]
        q_all = np.empty((N, h, len(q_levels)), dtype=np.float64)
        for s in range(0, N, batch_size):
            sub = [torch.tensor(ctx[i], dtype=torch.float32) for i in range(s, min(s + batch_size, N))]
            quant, _ = pipeline.predict_quantiles(
                sub, prediction_length=h, quantile_levels=q_levels,
            )
            for j, qt in enumerate(quant):
                q_all[s + j] = qt.detach().cpu().numpy()[0]         # (h, 21)
        for k in range(h):
            u = rng.random(N)
            qk = q_all[:, k, :]                                     # (N, 21) ascending
            samp = np.array([np.interp(u[i], ql, qk[i]) for i in range(N)])
            paths[:, t + k] = samp
        t += h
        if verbose:
            print(f"  [rollout seed={seed}] t={t}/{target_len}", flush=True)
    return inv(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["zero_shot", "finetune"], default="zero_shot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", type=int, default=16,
                    help="length of real test-set prefix used as warm-up context")
    ap.add_argument("--hstep", type=int, default=1, help="rollout horizon per step")
    ap.add_argument("--space", choices=["log", "price"], default="log",
                    help="rollout space; log(price) is stable for geometric Heston")
    ap.add_argument("--ctx_cap", type=int, default=0, help="cap rolling context length (0=full)")
    ap.add_argument("--gen_batch", type=int, default=512)
    ap.add_argument("--gen_num", type=int, default=8192)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of data (smoke test)")
    ap.add_argument("--ft_pred_len", type=int, default=16)
    ap.add_argument("--ft_steps", type=int, default=1000)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--ft_batch", type=int, default=256)
    ap.add_argument("--tag", default="", help="run tag (e.g. 'smoke'); skips canonical weights")
    a = ap.parse_args()
    ctx_cap = a.ctx_cap if a.ctx_cap > 0 else None

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    tagp = (a.tag + "_") if a.tag else ""
    variant = a.variant

    # --- data ---
    S_train = np.load(TRAIN_DATA).astype(np.float64)               # (8192,128)
    S_test = np.load(TEST_DATA).astype(np.float64)                 # (8192,128)
    if a.frac < 1.0:
        n = int(round(S_test.shape[0] * a.frac))
        S_train = S_train[:n]
        S_test = S_test[:n]
    N = S_test.shape[0]
    gen_n = min(a.gen_num, N)
    prefixes = S_test[:gen_n, :a.prefix]
    print(f"=== Chronos-2 Heston  variant={variant}  seed={a.seed}  "
          f"CUDA_VISIBLE_DEVICES={cvd}  device={dev_name} ===", flush=True)
    print(f"[data] train{S_train.shape} test{S_test.shape}  prefix={a.prefix}  "
          f"gen_n={gen_n}  hstep={a.hstep}  ctx_cap={ctx_cap}", flush=True)

    # --- load pipeline ---
    t_load = time.time()
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID, device_map=device, torch_dtype=torch.float32)
    print(f"[model] loaded {MODEL_ID} in {time.time()-t_load:.1f}s  "
          f"quantiles={len(pipeline.quantiles)}", flush=True)

    ft_rows = []
    train_time = 0.0
    ft_min_loss = None
    ft_first_loss = None
    if variant == "finetune":
        t0 = time.time()
        rec = make_loss_recorder()
        inputs = [torch.tensor(S_train[i], dtype=torch.float32) for i in range(S_train.shape[0])]
        print(f"[finetune] {len(inputs)} series  pred_len={a.ft_pred_len}  "
              f"steps={a.ft_steps}  lr={a.ft_lr}  batch={a.ft_batch}", flush=True)
        pipeline = pipeline.fit(
            inputs, prediction_length=a.ft_pred_len, finetune_mode="full",
            learning_rate=a.ft_lr, num_steps=a.ft_steps, batch_size=a.ft_batch,
            callbacks=[rec], remove_printer_callback=False,
            logging_steps=max(1, min(10, a.ft_steps // 5)),
            output_dir=os.path.join("/tmp", f"c2ft_{variant}_{a.seed}_{int(time.time())}"),
        )
        train_time = time.time() - t0
        ft_rows = rec.rows
        if ft_rows:
            losses = [r["train_loss"] for r in ft_rows]
            ft_first_loss, ft_min_loss = float(losses[0]), float(min(losses))
        print(f"[finetune] done in {train_time:.1f}s  "
              f"first_loss={ft_first_loss}  min_loss={ft_min_loss}", flush=True)

    # --- generate ---
    g0 = time.time()
    Xg = rollout_from_prefixes(
        pipeline, prefixes, target_len=SEQ_LEN, seed=a.seed,
        batch_size=a.gen_batch, hstep=a.hstep, ctx_cap=ctx_cap, space=a.space,
        verbose=True)
    gen_time = time.time() - g0
    gen_has_nan = bool(not np.isfinite(Xg).all())

    # --- output dirs ---
    # finetune is the sole shipped (headline) Chronos2 method: it writes to the
    # methods/Chronos2/ root, matching the layout of every other benchmark method.
    # zero_shot (dropped: autoregressive rollout is unstable for low-vol geometric
    # Heston) writes to a variant subfolder if ever regenerated for the archive.
    base = METHOD_DIR if variant == "finetune" else os.path.join(METHOD_DIR, f"{variant}_variant")
    weights_dir = os.path.join(base, "weights")
    losses_dir = os.path.join(base, "losses")
    gen_dir = os.path.join(base, "generated_paths", f"seed_{a.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    out_npy = os.path.join(gen_dir, f"{tagp}generated_paths_8192x128.npy")
    np.save(out_npy, Xg.astype(np.float64))

    if variant == "finetune":
        loss_name = f"{tagp}seed_{a.seed}_losses.csv"
        with open(os.path.join(losses_dir, loss_name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "train_loss", "lr"])
            w.writeheader()
            w.writerows(ft_rows)

    if not a.tag:
        if variant == "finetune":
            torch.save({"model": pipeline.model.state_dict(), "seed": a.seed,
                        "variant": variant}, os.path.join(weights_dir, f"seed_{a.seed}_model.pt"))
        cfg = {"method": "Chronos2", "variant": variant, "seed": a.seed,
               "model_id": MODEL_ID, "feat_dim": 1, "seq_len": SEQ_LEN,
               "gen_scheme": "autoregressive_rollout_from_real_test_prefix",
               "prefix": a.prefix, "hstep": a.hstep, "ctx_cap": ctx_cap,
               "n_quantiles": len(pipeline.quantiles)}
        if variant == "finetune":
            cfg.update({"ft_pred_len": a.ft_pred_len, "ft_steps": a.ft_steps,
                        "ft_lr": a.ft_lr, "ft_batch": a.ft_batch, "finetune_mode": "full"})
        with open(os.path.join(weights_dir, f"seed_{a.seed}_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    meta = {"method": "Chronos2", "variant": variant, "seed": a.seed,
            "shape": list(Xg.shape), "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
            "real_mean": float(S_test[:gen_n].mean()), "real_std": float(S_test[:gen_n].std()),
            "prefix": a.prefix, "gen_sec": round(gen_time, 1),
            "train_time_sec": round(train_time, 1), "gpu": "A100-SXM4-80GB",
            "date": time.strftime("%Y-%m-%d"),
            "ft_first_loss": ft_first_loss, "ft_min_loss": ft_min_loss,
            "gen_has_nan": gen_has_nan}
    with open(os.path.join(gen_dir, f"{tagp}metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] variant={variant} seed={a.seed} gen={Xg.shape} "
          f"price=[{Xg.min():.2f},{Xg.max():.2f}] nan={gen_has_nan} "
          f"gen={gen_time:.1f}s train={train_time:.1f}s", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
