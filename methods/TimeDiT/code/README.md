# TimeDiT, Heston training code

**Paper:** Cao, Ye, Zhang, Liu. *TimeDiT: General-purpose Diffusion Transformers for
Time Series Foundation Model.* arXiv:2409.02322v1 (Sept 2024).
**Official code:** none released. The paper (Appendix C) states the codebase is
*"modified from https://github.com/facebookresearch/DiT"* (Peebles & Xie, 2022).

Because there is no upstream repo to clone, `reference/` holds the **from-scratch
faithful reimplementation** that was validated against the paper's own Sine + Stocks
synthetic-generation benchmark in `../paper_reimplementation/` (see its README for the
paper-vs-ours table and GATE verdict). The same two modules are used verbatim here.

## Files

| File | Role |
|------|------|
| `timedit_model.py` | DiT-S backbone (hidden=384, depth=12, heads=6) + Time Series Mask Unit, copy of the reimplementation |
| `gaussian_diffusion.py` | DDPM forward/reverse process (fixed + learned variance, DDIM), copy of the reimplementation |
| `train_heston.py` | per-seed Heston trainer + generator (the worker) |
| `reference/` | archival copies of the two modules above (no upstream repo exists) |

## Answers to GUIDELINE §0 (Q1-Q8)

| # | Answer |
|---|--------|
| Q1 | **Price paths (levels).** Heston `S_t`. Min-max to [0,1] then z-norm; denormalise back to price before saving. |
| Q2 | **Univariate** (`d = 1`). TimeDiT tokenises over time, so `in_channels = 1`. |
| Q3 | One backbone; **DiT-S** (paper Table 9 "S": hidden=384, depth=12, heads=6). Synthetic generation uses the reconstruction mask `M^Rec = 0` → unconditional. |
| Q4 | Paper hyperparameters carried verbatim from the reproduction GATE winner (no Heston tuning): znorm · linear · `learn_sigma=False` · `ddpm_fixed` · lr 3e-4 · no-EMA · batch 256 · T=1000 · 15000 steps. |
| Q5 | **PyTorch, GPU.** Env: `/home/tbasseras/gpu-venv/bin/python` (torch cu13). |
| Q6 | ~one A100-hour per seed at 15000 steps, seq_len=128 (see `metadata.json` `train_time_sec`). |
| Q7 | Weights saved natively, `torch.save(model.state_dict())` → `weights/seed_{i}_model.pt`. |
| Q8 | Clean generation: `GaussianDiffusion.p_sample_loop(sample_var="fixed")` inside `train_heston.py`; no retraining needed. |

## Fixes / adaptations from the paper reproduction to Heston

- **Fix 1, length + channels.** Paper repro was `seq_len=24`, `C ∈ {5, 6}`. Heston is
  `seq_len=128`, `C=1`. The DiT backbone is length/channel-agnostic (positional embed is
  built for `seq_len`, `x_embedder` for `in_channels·3`), so only the constructor args
  change; no architecture edit.
- **Fix 2, price wrapper.** Paper data was already in [0,1]; Heston prices are ~40-155.
  A min-max→[0,1] wrapper is applied *before* the paper's z-norm, and inverted after
  sampling, so the model still trains in the exact z-normed space the recipe was tuned in.
- **Fix 3, train on all 8192 paths.** The paper repro used an 80/20 split (its metric
  needed held-out data); here metrics score against a **separate** Heston test seed
  (GUIDELINE §5.0), so the generator trains on the full seed-0 training draw.
- Generated paths are clipped to `>= 1e-6` (§4.3), the diffusion output is unbounded in
  principle, though in practice the [0,1] clip before price-inversion already prevents
  non-positive prices.

## Run

```bash
# smoke (short) on one GPU
python train_heston.py --seed 0 --gpu 0 --steps 300 --gen_num 1000 --tag smoke

# one full seed
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 python train_heston.py --seed 0 --gpu 0

# 5 seeds in 2-GPU pairs (GPU 0 + GPU 3) — see GUIDELINE §4.2
```
