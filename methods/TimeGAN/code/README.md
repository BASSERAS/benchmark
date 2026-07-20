# TimeGAN Code — Sources & Modifications

## Original work

| Field | Details |
|-------|---------|
| **Paper** | *Time-series Generative Adversarial Networks* |
| **Authors** | Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar |
| **Venue** | NeurIPS 2019 |
| **arXiv** | https://arxiv.org/abs/2010.00782 |
| **Original code** | https://github.com/jsyoon0823/TimeGAN |
| **Original framework** | TensorFlow 1.x (CPU-only) |

The verbatim TF1 reference is kept under [`reference/`](reference/) for transparency.

## Our implementation

`timegan_torch.py` is a **PyTorch GPU re-implementation** of the same 5-component
architecture described in the paper.

### Architecture (unchanged from paper)

| Component | Layers | Output |
|-----------|--------|--------|
| Embedder (E) | 3× GRU(24) | sigmoid → (T, 24) |
| Recovery (R) | 3× GRU(24) + Linear | sigmoid → (T, d) |
| Generator (G) | 3× GRU(24) | sigmoid → (T, 24) |
| Supervisor (S) | 2× GRU(24) | sigmoid → (T, 24) |
| Discriminator (D) | 3× GRU(24) + Linear | logit → (T, 1) |

Hidden dim = 24, batch size = 128, total steps = 20 000 (5 k embed + 5 k sup + 10 k joint).

### Fixes applied vs the TF1 reference

The TF1 reference code contains several bugs that degrade generation quality.
We fixed all five:

| # | Location | TF1 (buggy) | Our fix |
|---|----------|-------------|---------|
| 1 | Recovery output | linear activation | **sigmoid** (matches paper eq.) |
| 2 | Phase-1 embedding loss | `10 * MSE` | **`10 * √MSE`** (matches paper) |
| 3 | Generator supervised coeff | `10 * √G_sup` | **`100 * √G_sup`** (matches paper) |
| 4 | Embedder joint loss | missing supervision term | **`10*√MSE + 0.1*G_sup_detached`** |
| 5 | Moment-matching gradient | inside `no_grad` block | **gradients flow through Recovery** |

Fix 5 was the most impactful: the moment-matching loss (mean + std alignment)
must propagate gradients back through the Generator to be effective.

### Other changes vs reference

- **Framework**: TensorFlow 1.x → PyTorch 2.x (CUDA 13)
- **Hardware**: single CPU → 2× A100-SXM4-80GB GPUs (seeds run in parallel)
- **Data**: Sine waves / Stocks (reference) → Heston price paths (1D, T=128)
- **Discriminator hidden_dim**: fixed `int(1/2) = 0` crash for 1D data → `max(8, d*8)`
- **Discriminator**: added MLP variant alongside GRU for metric A13

## Training

```bash
cd methods/TimeGAN/code
# Single seed on GPU 0:
CUDA_VISIBLE_DEVICES=0 python train_seed.py --seed 0

# All 5 seeds on 2 GPUs (recommended):
python train.py --gpu0 0 --gpu1 3
```

Outputs written to `methods/TimeGAN/{generated_paths,weights,losses}/`.

---

## Changing hyperparameters

Edit `timegan_torch.py` or pass overrides via `train_seed.py` arguments.
The canonical paper hyperparameters are in the table below; do NOT tune on Heston
without documenting the change and why.

| Hyperparameter | Default | Where to change | Effect |
|----------------|---------|-----------------|--------|
| `hidden_dim` | 24 | `timegan_torch.py` line ~10 | GRU hidden state size for all 5 components |
| `num_layers` | 3 | `timegan_torch.py` line ~11 | GRU depth (Embedder, Recovery, Generator, Discriminator); Supervisor = num_layers − 1 |
| `batch_size` | 128 | `timegan_torch.py` line ~12 | Mini-batch size (affects memory and noise level) |
| `embed_steps` | 5 000 | `train_seed.py --embed_steps` | Phase-1 embedding pre-training steps |
| `sup_steps` | 5 000 | `train_seed.py --sup_steps` | Phase-2 supervisor pre-training steps |
| `joint_steps` | 10 000 | `train_seed.py --joint_steps` | Phase-3 adversarial joint training steps |
| `lr` | 1e-3 | `timegan_torch.py` | Adam learning rate for all components |

**Example — double the joint training steps:**
```bash
CUDA_VISIBLE_DEVICES=0 python train_seed.py --seed 0 --joint_steps 20000
```

---

## Using a different dataset

`train_seed.py` expects a path to a `.npy` file containing price paths of shape `(N, T)`,
dtype `float64`, in the original price scale (e.g. S₀ ≈ 100).

```bash
# Train on a custom dataset
CUDA_VISIBLE_DEVICES=0 python train_seed.py \
    --seed 0 \
    --data_path /path/to/my_paths_Nx128.npy \
    --out_dir   /path/to/my_method_output/seed_0/
```

Inside `train_seed.py`, the data is min-max normalised to [0, 1] before training
and denormalised back to the original scale before saving generated paths.
If your data has a different time dimension, update `seq_len` in the config.

---

## Producing new seeds

Each seed initialises a new random state (PyTorch, NumPy, CUDA) via `torch.manual_seed(seed)`.
To run a single extra seed without modifying the orchestrator:

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 OMP_NUM_THREADS=8 \
    python train_seed.py --seed 5
```

To run all seeds (0–4) in the standard parallel configuration:

```bash
python train.py --gpu0 0 --gpu1 3
```

Outputs land in:
- `generated_paths/seed_{i}/generated_paths_8192x128.npy`
- `weights/seed_{i}_model.pt`
- `weights/seed_{i}_config.json`
- `losses/seed_{i}_losses.csv`

---

## Reproduce the Heston metrics (exact run path)

After the 5 seeds are trained, the committed A1–A34 + B numbers come from the shared scorer:

```bash
cd /home/tbasseras/benchmark
CUDA_VISIBLE_DEVICES=0 \
    /home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method TimeGAN --dataset Heston
```

**Which file produced which committed number:**

| Committed number | Interpreter + env | Command | Input file(s) scored | Output file |
|------------------|-------------------|---------|----------------------|-------------|
| Heston A1–A34 + B, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0` | `metrics/compute_all.py --method TimeGAN --dataset Heston` | `methods/TimeGAN/generated_paths/seed_i/generated_paths_8192x128.npy` (8192,128) vs real Heston `dataset/Heston/heston_S_8192x128.npy` | `results/Heston/TimeGAN/seed_i_metrics.json` (A) + `curve_b_aggregate.json` (B) |
| TimeGAN synthetic paths, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0/3` | `train_seed.py --seed i` (or `train.py --gpu0 0 --gpu1 3`) | real Heston `dataset/Heston/heston_S_8192x128.npy`, min-max→[0,1] internally | `generated_paths/seed_i/generated_paths_8192x128.npy` + `weights/seed_i_model.pt` |

Each `results/Heston/TimeGAN/seed_i_metrics.json` is the sole source for that seed's column in every README A-table; mean±std rows aggregate the 5 files.
