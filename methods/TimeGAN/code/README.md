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
