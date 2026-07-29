"""
TimesFM forecaster @ this experiment's scale — fine-tune on the 4096 TRAIN paths,
forecast the 512 path-shadowing queries (H=32, K=256), score with the strict PDF
metrics (via pdf_bridge) so the row is directly comparable to LS4 / CSDI generator
PS columns, the Heston oracle and the RW floor.

Fine-tune recipe is byte-identical to the main-benchmark TimesFM finetune
(methods/TimesFM/path_shadowing/finetune_heston.py): full fine-tune, Adam lr=1e-4
wd=0.01, masked MSE(mean head) + pinball over the 9 quantile heads on the first 64
output steps, 1000 steps, batch 256. Only the TRAIN set (4096 vs 8192) changes.
Forecast: 65-point prefix (points 0..64) -> next H=32 steps, K=256 inverse-CDF
ensemble members, exactly the object the strict scorer consumes.

Usage:
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 8-15 \
      /home/tbasseras/timesfm-v1-venv/bin/python run_timesfm_forecaster.py
"""
import os
import sys
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HERE)
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(EXP_DIR)))
DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns")
TRAIN_DATA = os.path.join(DATA_DIR, "heston_S_4096x128.npy")

TFM_PS = os.path.join(BENCH_ROOT, "methods", "TimesFM", "path_shadowing")
sys.path.insert(0, TFM_PS)
import run_forecaster_ref as RF                 # noqa: E402  (load_tfm, forecast_ensemble, model_cfg)
import finetune_heston as FT                    # noqa: E402  (pinball, CONTEXT_LEN, HORIZON, QUANTILES)

sys.path.insert(0, HERE)
import pdf_bridge as B                          # noqa: E402  (score_forecaster, P)

MODEL_ID = "google/timesfm-1.0-200m-pytorch"
SEED = 0
K = 256                     # ensemble members == generator retrieval K (comparable)
FT_STEPS = 1000
FT_BATCH = 256
FT_LR = 1e-4
FT_WD = 0.01


def finetune(model, S_train, device):
    """Replicates finetune_heston.py: masked MSE + pinball(9q) on first-64 output steps."""
    ctx_np = S_train[:, :FT.CONTEXT_LEN].astype(np.float32)          # (N,64) points 0..63
    fut_np = S_train[:, FT.CONTEXT_LEN:FT.CONTEXT_LEN + FT.HORIZON].astype(np.float32)  # (N,64)
    N = ctx_np.shape[0]
    ctx = torch.tensor(ctx_np, device=device)
    fut = torch.tensor(fut_np, device=device)
    pad = torch.zeros_like(ctx)
    freq = torch.zeros((FT_BATCH, 1), dtype=torch.long, device=device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=FT_LR, weight_decay=FT_WD)
    rng = np.random.default_rng(SEED)
    t0 = time.time(); first = last = float("nan")
    for step in range(1, FT_STEPS + 1):
        bidx = rng.integers(0, N, size=FT_BATCH)
        xb, yb, pb = ctx[bidx], fut[bidx], pad[bidx]
        preds = model(xb, pb.float(), freq)                         # (B,n_patch,opl,1+Q)
        mean = preds[..., 0]
        loss = torch.mean((mean[:, -1, :FT.HORIZON] - yb) ** 2)
        for qi, q in enumerate(FT.QUANTILES):
            qp = preds[:, -1, :FT.HORIZON, qi + 1]
            loss = loss + torch.mean(FT.pinball(qp, yb, q))
        opt.zero_grad(); loss.backward(); opt.step()
        lv = float(loss.detach().cpu())
        if step == 1:
            first = lv
        last = lv
        if step % max(1, FT_STEPS // 10) == 0 or step == 1:
            print(f"  [ft step {step:4d}/{FT_STEPS}] loss={lv:.6f} ({time.time()-t0:.1f}s)", flush=True)
    model.eval()
    print(f"[timesfm] finetune done first={first:.6f} last={last:.6f} "
          f"({time.time()-t0:.1f}s)", flush=True)
    return round(time.time() - t0, 1), first, last


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    S_IDX, H = B.P.S_IDX, B.P.H
    num_layers, use_pos = RF.model_cfg(MODEL_ID)

    S_train = np.load(TRAIN_DATA).astype(np.float64)               # (4096,128) price
    qS = np.load(B.P.PS_QUERY).astype(np.float64)                 # (512,128) price
    prefixes = qS[:, :S_IDX + 1].astype(np.float32)              # (512,65) points 0..64
    print(f"[timesfm] train{S_train.shape} query{qS.shape} prefix={prefixes.shape[1]} "
          f"H={H} K={K} dev={device}", flush=True)

    tfm = RF.load_tfm(device, MODEL_ID, num_layers, use_pos, state_dict_path=None)
    model = tfm._model.to(device)
    ft_time, first, last = finetune(model, S_train, device)

    wdir = os.path.join(HERE, "weights"); os.makedirs(wdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(wdir, "timesfm_seed0_4096.pt"))

    t2 = time.time()
    rng = np.random.default_rng(SEED)
    ens = RF.forecast_ensemble(tfm, prefixes, H, K, rng, batch_size=512)   # (512,K,H) price
    print(f"[timesfm] forecast {ens.shape} in {time.time()-t2:.1f}s "
          f"nan={not np.isfinite(ens).all()}", flush=True)

    B.score_forecaster(
        ens, "timesfm", os.path.join(HERE, "timesfm_pdf.json"),
        extra_meta={"model_id": MODEL_ID, "n_train": int(S_train.shape[0]),
                    "ft_steps": FT_STEPS, "ft_lr": FT_LR, "ft_batch": FT_BATCH,
                    "ft_weight_decay": FT_WD, "ft_first_loss": first, "ft_last_loss": last,
                    "ft_time_sec": ft_time, "seed": SEED})
    print("[timesfm] DONE.", flush=True)


if __name__ == "__main__":
    main()
