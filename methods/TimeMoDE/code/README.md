# TimeMoDE — Heston training code

Trains **TimeMoDE** (arXiv:2606.15172, *Towards a Unified Generative Model for
Scarce Time Series with Domain Experts*, Yao/Zheng/Zuo/Zhang, ICML 2026) on the
benchmark's 8192×128 Heston price paths and writes generated paths back in price
scale for the shared metric harness.

## The model is the reproduction-gate model — imported, not re-written

There is **no official TimeMoDE code**. This benchmark entry is a from-the-paper
reimplementation that first reproduced the paper's own headline numbers on the
paper's simplest dataset (StarLightCurves, "From Scratch"), documented in
[`../paper_reimplementation/README.md`](../paper_reimplementation/README.md).

To make it *impossible* for the Heston model to drift from the model that passed
that gate, [`train_seed.py`](train_seed.py) **imports the architecture directly**:

```python
sys.path.insert(0, "../paper_reimplementation")
from timemode_model import build_timemode   # the exact DiT-MoDE
from diffusion import Diffusion             # the exact DDPM
```

The only thing that changes between SLC and Heston is the **input geometry**
(SLC `L=24, C=1` → Heston `L=128, C=1`). Everything else is byte-for-byte the
seed-0 configuration:

| Group | Value | Same as SLC seed 0? |
|-------|-------|:-------------------:|
| hidden / depth / heads | 256 / 6 / 4 | ✅ |
| experts K / top-k / shared E₀ | 8 / 2 / 1 | ✅ |
| expert expansion n | 4 | ✅ |
| diffusion steps T / schedule | 250 / linear | ✅ |
| patch size | 1 | ✅ |
| params | 53.9 M (input proj scales with L) | arch identical |
| optimiser | AdamW lr 1e-4, wd 1e-5 | ✅ |
| batch / epochs | 2048 / 1000 | ✅ |
| EMA decay (0 for first 100 steps) | 0.9999 | ✅ |
| grad-clip | 1.0 | ✅ |
| loss weights w_proto / w_aux | 0.01 / 0.01 | ✅ |

## Architecture recap (full detail in the paper-reimpl README)

TimeMoDE is a **Diffusion Transformer (DiT)** whose token-mixing MLPs are
replaced by a **Mixture of Domain Experts (MoDE)**. ε-prediction DDPM, fixed
variance. `model(x, t, dp_exemplar, y) → (eps, aux_loss, proto_loss)`:

1. **Patch embed + timestep embed** → conditioning vector `c`.
2. **Domain Prompt (Eq 6):** `DP = Avg(Linear(Conv1d(x)) + PosEnc)` from a batch
   of clean windows (`dp_exemplar`), used only for expert routing.
3. **6 × DiT-MoDE block (Eq 5):** `x += gate₁·MHSA(AdaLN(x))` then
   `x += gate₂·MoDE(AdaLN(x))`, adaLN-Zero (`SiLU→Linear(d,6d)`).
   - **MoDE (Eq 10–13):** router `sᵢ ∝ ‖Protoᵢ·DP‖² + (Wz)ᵢ + τ`, Top-2→Softmax
     gate, output `Σ G(sᵢ)·Eᵢ(z) + E₀(z)`.
   - **Expert (Eq 11):** stage-aware AdaLN + SwiGLU + residual.
4. **Final AdaLN + linear** → ε.

Auxiliary losses summed into the objective (Eq 15,
`L = L_DDPM + w_proto·L_proto + w_aux·L_aux`): prototype orthonormality
`‖PPᵀ − I‖²_F` and Switch-transformer load balancing.

## Normalisation (identical to SLC)

The model trains in **[0,1]** (not [-1,1] — matches `run_paper.py`, which trains
on [0,1] windows and clips fakes to [0,1]):

```
S(price) --global min-max--> [0,1]        (model trains here)
sample --clip[0,1]--> [0,1] --invert--> price   (clip min 1e-6, strictly positive)
```

`(lo, hi)` are the global min/max of `heston_S_8192x128.npy` and are saved to each
seed's config/metadata so the price inversion is fully reproducible.

## Files

| File | Purpose |
|------|---------|
| [`train_seed.py`](train_seed.py) | One seed: train → generate 8192 paths → save weights/losses/paths/metadata. |
| [`train.py`](train.py) | 5-seed orchestrator: 2 seeds at a time on 2 GPUs, 8 cores each (hard limits). |

Model/diffusion live in [`../paper_reimplementation/`](../paper_reimplementation/)
and are imported — they are **not** duplicated here.

## Reproduce

```bash
# from methods/TimeMoDE/code
# smoke (30 epochs, 5% data — sanity only, does not overwrite canonical outputs)
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 8-15 \
  /home/tbasseras/gpu-venv/bin/python train_seed.py --seed 0 --smoke

# full 5-seed run (GPUs 1,2; cores 8-15 and 16-23)
/home/tbasseras/gpu-venv/bin/python train.py --gpus 1 2 --seeds 5

# or a single canonical seed
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 8-15 \
  /home/tbasseras/gpu-venv/bin/python train_seed.py --seed 0
```

## Outputs (GUIDELINE §4.3 schema)

| Path | Content |
|------|---------|
| `../generated_paths/seed_i/generated_paths_8192x128.npy` | (8192,128) float64 **price** paths, clip min 1e-6. |
| `../generated_paths/seed_i/metadata.json` | shape, price range, scale_min/max, params, timings, loss log. |
| `../weights/seed_i_model.pt` | `{model, ema, arch, seed, minmax:[lo,hi]}`. |
| `../weights/seed_i_config.json` | full architecture + optimiser config. |
| `../losses/seed_i_losses.csv` | `step, loss_total, simple, proto, aux`. |

## Reproduction decisions (unchanged from SLC)

These are the paper-unspecified choices carried over verbatim from the gate:
`learn_sigma=False` (loss Eq 3/15 has no VLB term); `w_proto=w_aux=0.01` (Eq 15
sums the regularisers with no stated weights — small weights keep them
subordinate to L_DDPM); Domain-Prompt exemplars = random batch of clean training
windows per step/sample (single-domain setting); proto loss = `‖PPᵀ − I‖²_F`
(orthonormality is the intended separability objective). Rationale in
[`../paper_reimplementation/README.md §3`](../paper_reimplementation/README.md).
