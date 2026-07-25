# Chronos-2 Code — Sources & Implementation

## Original work

| Field | Details |
|-------|---------|
| **Paper** | *Chronos-2: From Univariate to Universal Forecasting* |
| **Authors** | Abdul Fatir Ansari, Caner Turkmen, Oleksandr Shchur, et al. (Amazon Science) |
| **Venue** | Preprint, 2025 — [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) |
| **Original code** | https://github.com/amazon-science/chronos-forecasting |
| **Original framework** | **PyTorch** (Hugging Face `transformers`) — no framework port needed |
| **Checkpoint used** | [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2) — ~120M params, the **largest model in the benchmark** |
| **Method type** | **Pretrained encoder-decoder (T5-style) probabilistic forecaster** with *group attention* for universal (uni-/multivariate, covariate-aware) forecasting. Outputs a set of predictive **quantiles** per future step; here we use the 21 trained quantile levels for one-step-ahead inference. |

The verbatim reference implementation is kept under [`reference/`](reference/) for transparency
(`.git` stripped). Chronos-2 is imported **unchanged** — [`train_heston.py`](train_heston.py) puts
`reference/src` on `sys.path` and imports `from chronos import BaseChronosPipeline`. **No model code is
modified**: the network, tokeniser-free patch embedding, group-attention encoder-decoder, and quantile
head are all the released `amazon/chronos-2` checkpoint. Our code is a **fine-tune + autoregressive
rollout driver** around that frozen implementation.

---

## Our implementation

Chronos-2 is a **forecaster**, not a generator: given a context it predicts the next
`prediction_length` steps as quantiles. To turn it into a path *generator* for the benchmark,
[`train_heston.py`](train_heston.py):

1. **Full fine-tunes** `amazon/chronos-2` on the 8192 Heston training paths (`pipeline.fit(...)`,
   `finetune_mode="full"`), so the model's scale/shape prior matches low-vol geometric Heston.
2. **Autoregressively rolls out** each of 8192 paths from a **real length-16 prefix taken from the
   held-out TEST set**, in **log-space**, drawing **one Monte-Carlo sample per step** by inverse-CDF
   over the 21 predicted quantiles, until each path reaches length 128.
3. Writes weights / losses / generated paths / metadata in the benchmark's standard layout.

### Why fine-tune only (no zero-shot variant ships)

> Zero-shot Chronos-2 rolled out autoregressively is **fundamentally unstable** on low-volatility
> geometric Heston. The model's robust (level-proportional) scaling means a small per-step positive
> bias compounds **multiplicatively**: even in log-space the rollout runs away over 112 steps,
> producing terminal prices orders of magnitude off. Fine-tuning re-calibrates the one-step
> conditional so the rollout stays on-manifold. Per the benchmark decision, **only the fine-tuned
> variant ships** — no zero-shot generated-path artifacts are archived. The zero-shot *paper*
> reproduction (MASE/WQL on the Chronos datasets, where forecasts are short-horizon, not rolled out)
> lives separately in [`../paper_reimplementation/`](../paper_reimplementation/).

### The rollout mechanism (`rollout_from_prefixes`)

```
space = "log":  fwd = np.log,  inv = np.exp        # rollout in log-price
q_levels = list(pipeline.quantiles)                # 21 trained levels, ascending
for each step until length == 128:
    quant = pipeline.predict_quantiles(context, prediction_length=hstep, quantile_levels=q_levels)
    for each series i:
        u  = uniform(0,1)                            # one MC draw
        x  = np.interp(u, q_levels, quant[i])        # inverse-CDF over the 21 quantiles
    append inv(x) to the path, feed back as context  # autoregressive
```

`hstep = 1` (one step forecast per iteration), `prefix = 16` (real TEST-set context seed), batched over
`gen_batch = 512` series. The prefix is real data, so **only the 112 post-prefix steps are model-
generated**; the metrics score the full 128-length path.

### Fine-tune loss capture

`make_loss_recorder()` returns a Hugging Face `TrainerCallback` whose `on_log` appends
`{step, train_loss, lr}` rows; these are written to `losses/seed_{i}_losses.csv`. `pipeline.fit(...)`
returns a **new** fine-tuned pipeline object (the call does not mutate in place).

---

## Adaptations vs the reference

Chronos-2's model code is used **unchanged**; all logic is in the driver:

1. **Generation-by-rollout wrapper.** *Our addition:* `rollout_from_prefixes` — the reference has no
   path-generation entry point (it forecasts fixed horizons). We wrap `predict_quantiles` in a
   log-space autoregressive loop with inverse-CDF MC sampling.
2. **Fine-tune plumbing.** *Our addition:* load the Heston `.npy`, build the `inputs` list Chronos-2's
   `.fit` expects, capture the loss via a callback, and persist the fine-tuned `state_dict`.
3. **Data I/O + seeding.** Load `heston_S_8192x128.npy` (train) / `heston_S_test_8192x128.npy` (prefix
   source); seed torch + numpy from `--seed`; export the generated `.npy` in price scale via `np.exp`.

No hyperparameter or objective change to the model itself — the checkpoint, group-attention
architecture, and quantile head are the released `amazon/chronos-2` defaults.

---

## Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `model_id` | `amazon/chronos-2` | released checkpoint (~120M params) |
| `finetune_mode` | `full` | full fine-tune (all weights) |
| `ft_steps` | 1000 | fine-tune training steps |
| `ft_lr` | 1e-4 | fine-tune learning rate (cosine → 0) |
| `ft_batch` | 256 | fine-tune batch size |
| `ft_pred_len` | 16 | fine-tune forecast horizon |
| `torch_dtype` | `float32` | inference/fine-tune precision |
| `prefix` | **16** | real TEST-set context length seeding each rollout |
| `hstep` | 1 | steps forecast per rollout iteration |
| `n_quantiles` | 21 | trained quantile levels used for inverse-CDF sampling |
| `space` | **log** | rollout performed in log-price |
| `ctx_cap` | none | full growing context fed back each step |
| `gen_batch` | 512 | series per rollout batch |
| `gen_num` | 8192 | paths generated |
| `feat_dim` | **1** | **Heston-specific** — univariate price series |
| `seq_len` | **128** | **Heston-specific** — the Heston sequence length |

Only `feat_dim` / `seq_len` are dataset-specific; the fine-tune preset (`full`, 1000 steps, lr 1e-4,
batch 256, pred_len 16) was chosen to stabilise the rollout at minimal compute (~85 s/seed) — larger
step counts gave no further loss improvement past the ~7.57 floor.

---

## How to change hyperparameters

All knobs are CLI flags on [`train_heston.py`](train_heston.py):

| Flag | Default | Effect |
|------|---------|--------|
| `--variant` | `zero_shot` | `finetune` = the shipped variant; `zero_shot` = no fine-tune (unstable, not archived). |
| `--seed` | 0 | Re-seeds torch + numpy (fine-tune init + MC sampling). |
| `--prefix` | 16 | Real TEST-set context length seeding each rollout. |
| `--hstep` | 1 | Steps forecast per rollout iteration. |
| `--space` | `log` | `log` or `price` rollout space. |
| `--ctx_cap` | 0 | Cap on fed-back context length (0 = unbounded). |
| `--ft_pred_len` | 16 | Fine-tune forecast horizon. |
| `--ft_steps` | 1000 | Fine-tune training steps. |
| `--ft_lr` | 1e-4 | Fine-tune learning rate. |
| `--ft_batch` | 256 | Fine-tune batch size. |
| `--gen_batch` | 512 | Series per rollout batch. |
| `--gen_num` | 8192 | Paths to generate. |
| `--frac` | 1.0 | Fraction of training paths (smoke runs). |
| `--tag` | "" | Run tag — prefixes outputs, skips canonical weights. |

---

## How to use a different dataset

`train_heston.py` reads `dataset/Heston/heston_S_8192x128.npy` (fine-tune) and
`heston_S_test_8192x128.npy` (prefix source), each a `.npy` of shape `(N, T)` in price/level space.
Point `TRAIN_DATA` / `TEST_DATA` at other `(N, T)` arrays to fine-tune and roll out on a different
process. Because the rollout is log-space and level-proportional, any strictly-positive series works;
for a series that can go non-positive, switch `--space price`.

---

## How to produce new seeds

Each seed re-seeds torch + numpy, so fine-tune init and MC sampling differ:

```bash
cd methods/Chronos2/code
# one extra seed (fine-tuned variant)
CUDA_VISIBLE_DEVICES=0 /home/tbasseras/gpu-venv/bin/python \
  train_heston.py --variant finetune --seed 5

# all 5 canonical seeds, 2 GPUs in parallel (~85 s fine-tune + ~148 s rollout per seed)
for s in 0 1 2 3 4; do
  g=$((s % 2 == 0 ? 0 : 3)); c=$(((s % 2)*8))
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 taskset -c $c-$((c+7)) \
    /home/tbasseras/gpu-venv/bin/python train_heston.py --variant finetune --seed $s &
done
wait
```

Each seed writes:
- `weights/seed_{s}_model.pt` — `{"model": fine-tuned state_dict, "seed", "variant"}` (~478 MB, git-excluded)
- `weights/seed_{s}_config.json` — full hyperparameters + `gen_scheme` + `prefix` + `ft_*`
- `losses/seed_{s}_losses.csv` — `step, train_loss, lr`
- `generated_paths/seed_{s}/generated_paths_8192x128.npy` — (8192, 128) price scale
- `generated_paths/seed_{s}/metadata.json` — seed, min/max, gen/train time, ft loss, gen_has_nan

---

## Sanity check

The canonical runs (5 seeds) confirm a healthy fine-tune and NaN-free generation:

```
=== Chronos-2 Heston  variant=finetune  seed=0  device=A100-SXM4-80GB ===
[model] loaded amazon/chronos-2
[fine-tune] step 10 loss=8.590 ... min loss=7.573  (1000 steps, ~85 s)
[rollout] prefix=16 log-space 8192×128  nan=False  gen≈148 s
[done] seed=0  gen_mean=103.30  gen_std=15.23  gen_has_nan=False
```

Expected sane signals (all five seeds, verified from metadata):
- fine-tune loss **8.590 → ~7.918** (min **7.573**) on every seed; `gen_has_nan = false` throughout;
- **fine-tune ≈ 81–85 s/seed, rollout ≈ 145–148 s/seed** (largest model, but rollout-bound not
  train-bound);
- generated mean ≈ 103.1–103.9 (real ≈ 101.25); generated std ≈ 14.2–15.2 on seeds 0/1/2/4 — **seed 3
  std = 110.3**, inflated by a single runaway rollout path (kept unclipped; it drives the seed-3 tails
  in A1/A20/A26/A30).

---

## Reproduce (exact run path)

```bash
# Environment: gpu-venv (torch, CUDA, transformers). Chronos-2 fine-tunes + rolls out on GPU.
cd methods/Chronos2/code

# All 5 Heston seeds (2 GPUs in parallel — see "How to produce new seeds")

# Compute metrics
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method Chronos2 --dataset Heston
```

**Exact run path — which file produced which committed number:**

| Committed number | Interpreter + env | Command | Input file(s) scored | Output file |
|------------------|-------------------|---------|----------------------|-------------|
| Chronos-2 synthetic paths, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0/3 taskset -c … OMP_NUM_THREADS=8` | `train_heston.py --variant finetune --seed i` | fine-tune: real `dataset/Heston/heston_S_8192x128.npy`; prefixes: `heston_S_test_8192x128.npy` | `generated_paths/seed_i/generated_paths_8192x128.npy` + `metadata.json` |
| Heston A1–A34 + B, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0` | `metrics/compute_all.py --method Chronos2 --dataset Heston` | `methods/Chronos2/generated_paths/seed_i/generated_paths_8192x128.npy` (8192,128) vs real test set | `results/Heston/Chronos2/seed_i_metrics.json` (A) + `curve_b_aggregate.json` (B) |
| A18/A19 disc/pred loss curves, per seed `i` | `gpu-venv` | `metrics/compute_all.py --method Chronos2 --dataset Heston` | same generated `.npy` vs real | `results/Heston/Chronos2/seed_i_{disc,pred}_{gru,mlp}_loss.csv` → `plots/{disc_classifier,pred_score}_loss.png` |
| Path-Shadowing MC (CRPS) | `gpu-venv` | `results/Heston/Chronos2/path_shadowing/run_eval.py` | generated `.npy` vs real test set | `results/Heston/Chronos2/path_shadowing/summary.json` |

Each `results/Heston/Chronos2/seed_i_metrics.json` is the sole source for that seed's column in every
README A-table; the mean±std rows aggregate the 5 files.

The zero-shot paper reproduction (MASE / WQL on the Chronos datasets — the paper's own metrics) lives
separately in [`../paper_reimplementation/`](../paper_reimplementation/).
