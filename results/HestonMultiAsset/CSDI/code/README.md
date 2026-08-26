# CSDI — source and implementation notes (d = 8)

## Source

**CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation**
Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon. NeurIPS 2021.
Paper: [arXiv:2107.03502](https://arxiv.org/abs/2107.03502) ·
Code: <https://github.com/ermongroup/CSDI> (MIT, `LICENSE` vendored under `reference/`)

CSDI is a **DDPM with a 2-D Transformer denoiser**. Each of the 4 residual blocks runs one
`nn.TransformerEncoderLayer` along the *time* axis and one along the *feature* axis, so
cross-asset structure is modelled by the same mechanism that models autocorrelation — this is
what makes the method a genuine multi-asset candidate rather than eight univariate fits stapled
together.

The paper's headline task is **imputation** (conditional generation). The unconditional regime
used here is the one the authors describe in Sec 4.1 and Appendix C and expose as a flag:

```
is_unconditional = 1        observed_mask ≡ 1        cond_mask ≡ 0
```

The mechanics are worth stating precisely, because "unconditional CSDI" sounds like a
modification and is not. `CSDI_base.set_input_to_diffmodel` branches on `is_unconditional`; in
the unconditional branch it feeds the network **only** the noisy sequence (`input_dim = 1`
instead of 2). `cond_mask` never gates the network input at all — it only selects which points
enter the loss, through `target_mask = observed_mask − cond_mask`. With `observed_mask ≡ 1` and
`cond_mask ≡ 0` that target is 1 everywhere, and the objective collapses to the standard DDPM
loss `E_t ‖ε − ε_θ(x_t, t)‖²`. On the sampling side, `impute` with `cond_mask = 0` has nothing
to condition on and reduces to plain ancestral sampling from pure noise. No line of
`reference/main_model.py` was changed to achieve any of this.

The d = 1 entry for the same method is [`methods/CSDI/`](../../../../methods/CSDI/).

---

## Files

| File | Role |
|------|------|
| `train_multiasset.py` | trains one seed **and** generates its 8 192 paths |
| `collect_artifacts.py` | rebuilds `losses/generation_time.csv`, checks the §4 contract, exits non-zero on violation |
| `plot_losses.py` | `plots/loss_convergence.png`, 2 panels × 5 seeds |
| `plot_diagnostics_multiasset.py` | the 8-panel stylised-facts figure |
| `measure_memorisation.py` | nearest-neighbour memorisation diagnostic → `losses/memorisation.json` |
| `render_readme.py` | regenerates `../README.md` from the artefacts on disk |
| `reference/` | the vendored CSDI release: `main_model.py`, `diff_models.py`, `config/base.yaml` |

`reference/` is byte-identical to `methods/CSDI/code/reference/` **except for one line**, the
lazy import documented below. The upstream `data/` and `save/` directories (PhysioNet and PM2.5
downloads, 1.5 GB) are *not* vendored — nothing in this port imports `exe_physio.py`,
`exe_pm25.py` or `download.py`.

---

## Provenance check — was the right CSDI actually run?

Copying a model directory proves nothing. The check is whether this tree, driven by this
port's code path, reproduces the **already-committed d = 1 run** exactly:

```bash
cd methods/CSDI/code
CUDA_VISIBLE_DEVICES=2 /home/tbasseras/gpu-venv/bin/python \
    train_heston.py --seed 0 --epochs 4 --tag provcheck
```

| Quantity | Committed d = 1 run | Re-run | |
|----------|---------------------|--------|--|
| parameter count | 412 945 | 412 945 | ✅ |
| `zscore_mean` | 101.32547381502401 | 101.32547381502401 | ✅ |
| `zscore_std` | 9.971659995159825 | 9.971659995159825 | ✅ |
| loss, step 0 | 0.9960838556289673 | 0.9960838556289673 | ✅ |
| loss, step 1 | 0.99347984790802 | 0.99347984790802 | ✅ |
| loss, step 2 | 0.9883894920349121 | 0.9883894920349121 | ✅ |
| **all 512 steps of epoch 0** | — | **bit-identical** | ✅ |

Not "close" — the same float64 digits, all 512 of them. That rules out a silently different
noise schedule, a different mask convention, a different RNG ordering and a different
optimiser state, which is the whole point of running it.

> **The `--epochs 1` trap.** The first attempt at this check used `--epochs 1` and matched only
> step 0. The cause is `MultiStepLR(milestones=[int(0.75·E), int(0.9·E)], gamma=0.1)`: at
> `E = 1` both milestones evaluate to **0**, so both fire at construction and the run trains at
> `1e-3 × 0.1 × 0.1 = 1e-5`. Any short probe of this codebase must use **`--epochs ≥ 4`** to keep
> epoch 0 at the real learning rate. This is a property of the reference schedule, not a defect
> in either implementation, and it will bite anyone who repeats the check.

At d = 8 the parameter count is **413 057**, exactly **112** more. That difference is the
feature embedding `nn.Embedding(target_dim, 16)` growing from 1 × 16 to 8 × 16 = +112. Nothing
else in the network depends on `d`, which is the numerical statement of "CSDI is
feature-agnostic".

---

## The four questions (MULTIASSET_GUIDELINE §0)

### 1. Does the method natively accept `(N, T, d)` input?

**Yes, and the fit is joint, not per-asset.** `target_dim = K` flows into two places: the
feature embedding `nn.Embedding(K, featureemb)` and the feature-axis Transformer inside every
residual block. `T` is arbitrary. The model is instantiated once with `target_dim = 8` and sees
all 8 channels of every sample simultaneously; `process_data` permutes the loader's `(B, L, K)`
into the `(B, K, L)` the reference expects and hands over `observed_mask ≡ 1`.

This matters for **A20**, the terminal-covariance-error row, which is the only metric that
tests whether the d(d−1)/2 spot correlations Σˢ survived generation. A method fitted per asset
cannot pass it except by luck; CSDI at least has a mechanism.

`config/base.yaml` contains **no** dimension-dependent entry — no `input_dim`, no `in_channels`.
The only thing the data forces is `target_dim`, passed at construction.

### 2. Does any hyperparameter fail to cross dimension?

**None.** `retuned_for_d8` is empty in every `weights/seed_*_config.json`, and that field is
read from disk by `render_readme.py` rather than asserted in prose. Layers, channels, heads,
diffusion embedding dim, `num_steps`, the quadratic β schedule, `beta_start`/`beta_end`, the
time and feature embedding widths, the optimiser, the LR schedule, the batch size and the epoch
budget are all the released `config/base.yaml`, unchanged. The d = 8 run is the authors'
configuration pointed at a wider tensor, not a search that happened to land somewhere.

That is a stronger claim than it may look, and it was **not** assumed — see question 4. It only
holds because the measured cost turned out to be affordable; had 200 epochs at batch 16 been
unaffordable, the honest move would have been to cut and declare the cut here.

**The scaler did change, from a global standardise to a per-channel one — and that is a scoping
consequence, not a retune.** Three reasons:

1. At K = 1 the d = 1 code's `S.mean()` **is** the per-feature statistic. Per-channel is the
   faithful extension of the same rule to K = 8, not a new rule.
2. CSDI's own PhysioNet loader (`reference/dataset_physio.py`) standardises **per feature**.
   Global standardisation is the d = 1 special case, not the upstream convention.
3. The eight assets are deliberately given different volatilities — per-asset σ is
   `[11.826, 15.741, 14.048, 18.550, 13.311, 16.412, 12.684, 11.142]` against a global σ of
   14.408 — so a single global divisor would hand the network channels ranging over ±29 % in
   scale for no reason.

Crucially, **a per-channel affine map leaves the cross-asset correlation matrix exactly
unchanged**, so Σˢ — the thing A20 scores — survives both the standardisation and its inverse.
The choice cannot flatter the covariance row.

**Two operational traps worth recording**, neither a hyperparameter but both capable of
silently ruining a run:

- `calc_loss_valid` loops `for t in range(self.num_steps)` — **all 50** diffusion steps — so a
  full 8 192-path validation pass costs ~50× a training epoch and would have dominated the
  wall-clock. Capped at `--val_n 256` on a `val_every = max(1, epochs // 20)` cadence, which
  keeps validation overhead near 2 %.
- The validation `DataLoader` uses `drop_last=False`. With `drop_last=True` a `--val_n` smaller
  than `--batch_size` yields an **empty** loader and a silent `nan` validation curve instead of
  an error. That happened once during a batch-64 probe.

### 3. What is the memorisation risk?

CSDI never touches a training path at generation time: sampling starts from `torch.randn` and
runs 50 reverse steps through the network. There is no retrieval step, unlike a kernel method.
On capacity grounds verbatim memorisation is implausible — 413 057 parameters against
8 192 × 252 × 8 ≈ 16.5 M training values, a ratio of ~1:40.

That argument is suggestive, not proof, so the diagnostic is **measured**:
`measure_memorisation.py` computes the nearest-neighbour ratio on the final 8 192-path output
and writes `losses/memorisation.json`. The number is reported on the dataset-level page
alongside every other method's, on byte-identical code, so it is comparable rather than
self-graded. For reference on this dataset, SBTS scores 0.2189 ± 0.0003 and LS4 0.8381 ± 0.0166.

### 4. Compute budget

**Measured before launching, not estimated.** My own first estimate from naive K·T² Transformer
scaling was ~295 s/epoch. The measurement is **28.6 s/epoch** (27.6 s steady state, 41.5 s for
epoch 0 including CUDA warm-up) at batch 16 on the full 8 192 paths, T = 252, K = 8. The
estimate was wrong by roughly 10× because at d = 1 — batch 16, T = 128, K = 1 — the A100 sat
nearly idle, so the d = 1 timings carry no information about the d = 8 cost.

| | |
|---|---|
| training | 200 epochs × 27.6 s ≈ **1.55 h/seed** |
| generation | 8 192 paths × 50 reverse steps ≈ **403 s** |
| per seed | ≈ **1.85 h** |
| 5 seeds, 3 waves on 2 GPUs | ≈ **5.2 h** |

That fits inside the budget already approved for LS4, which is why nothing had to be cut and
`retuned_for_d8` could stay empty. Had it not fit, this section would record the cut instead.

---

## Deviations from the released d = 1 implementation

Four, all of them narrow and all of them here rather than buried.

**1. Lazy import in `reference/diff_models.py`.** Upstream has a *top-level*
`from linear_attention_transformer import LinearAttentionTransformer`. That dependency is only
reachable through the forecasting path (`exe_forecasting.py`), which this benchmark never runs,
but the top-level import makes the whole module unimportable without it. Moved inside the
function that uses it. This is the **only** edit to any vendored file, it is marked in-place
with a `NOTE (CSDI benchmark fix 1)` comment, and it changes no computation.

**2. `generate()` had to be generalised past asset 0.** The d = 1 wrapper
(`methods/CSDI/code/train_heston.py`) ends its sampler with `samples[:, 0, 0, :]` — correct and
invisible at K = 1, but at K = 8 it would have silently kept **asset 0 only** and returned a
degenerate tensor. Replaced with `samples[:, 0].permute(0, 2, 1)`, `(B, 1, K, L) → (B, L, K)`.
This is a bug in the d = 1 wrapper's shortcut, not in the reference model, and it is exactly the
class of error that a copy-paste port ships without noticing.

**3. Per-channel z-score** instead of global. Justified under question 2 above.

**4. S0 rescaling to the §4 contract.** The guideline requires `S[:, 0, :] == 100.0` exactly.
`rescale_to_s0` does:

```python
Xg = Xg * (S0 / Xg[:, 0:1, :])
Xg[:, 0, :] = S0
```

A constant per-path-per-asset multiplier is exactly a shift of the log-price *level*, so **every
log-return is bit-identical** and A1–A25 / A27–A34 are unaffected by construction. `s0_rescaled`
is recorded as `true` in every `metadata.json`.

The clipping that precedes it is **not** so innocent, and is therefore counted rather than
hidden. A model can emit a non-positive price; `np.where(Xg <= 0, 1e-6, Xg)` fixes that, but
unlike the multiplier it does **not** preserve log-returns — and a *first* price clipped to
1e-6 rescales that entire path by 1e8. So the two counts are reported separately in every
`metadata.json`:

```
n_nonpositive_s0_before_rescale      non-positive first prices  (catastrophic)
n_nonpositive_total_before_rescale   non-positive prices anywhere
```

`collect_artifacts.py` prints both per seed. A short undertrained probe (2 epochs) produced
52–498 non-positive first prices and an apparent `S_max` of 1.2e10; at 2 *full* epochs the raw
range was already `[−28.09, 247.58]` with **zero** non-positive first prices, against a real
Heston range of roughly `[28, 235]`. The explosion was an undertraining artefact, not a design
fault — but the counters stay, because the only way to know that is to look at them.

---

## Deviation from the §1 layout, stated rather than papered over

There is **no `run_all_multiasset.py`**, although MULTIASSET_GUIDELINE.md §1 lists one. CSDI
generates inside `train_multiasset.py`, from the in-memory model immediately after training.
Splitting generation into a second script would mean reloading a checkpoint and duplicating the
per-channel scaler inversion and the S0 rescaling in a second place where the two copies could
silently drift — and a drifted inverse-scaler is invisible in every artefact except the metrics
it corrupts. `collect_artifacts.py` fills the slot for post-generation bookkeeping and contract
verification.

No seed process writes a shared file. The 5 seeds train **concurrently across 2 GPUs**, so an
in-run writer for `generation_time.csv` would not merely lose rows for seeds it did not re-run
(the failure §4 warns about) — live processes would race on the same path. The timing table is
rebuilt once, afterwards, from the per-seed `metadata.json` files.

---

## Reproduce

```bash
cd /home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python

# 0. provenance check against the committed d = 1 run — --epochs 4, NOT 1 (see above)
cd methods/CSDI/code && CUDA_VISIBLE_DEVICES=2 $PY train_heston.py \
    --seed 0 --epochs 4 --tag provcheck && cd -

# 1. final 5 seeds, 200 epochs, 2 GPUs, 3 waves (~5.2 h wall-clock)
cd results/HestonMultiAsset/CSDI/code
GPUS=(2 3); CORES=("16-23" "24-31")
for wave in "0 1" "2 3" "4"; do
  i=0
  for s in $wave; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} OMP_NUM_THREADS=8 taskset -c ${CORES[$i]} \
        $PY train_multiasset.py --seed $s & i=$((i+1))
  done
  wait
done

# 2. bookkeeping — must print "5 rows" and exit 0 before anything downstream
$PY collect_artifacts.py

# 3. metrics, figures, memorisation, README
cd /home/tbasseras/benchmark
CUDA_VISIBLE_DEVICES=2 $PY metrics/compute_all_multiasset.py --method CSDI \
    --dataset HestonMultiAsset --seeds 5
$PY results/HestonMultiAsset/CSDI/code/plot_losses.py
$PY results/HestonMultiAsset/CSDI/code/plot_diagnostics_multiasset.py --method CSDI
$PY metrics/plot_score_losses.py --method CSDI --dataset HestonMultiAsset
$PY results/HestonMultiAsset/CSDI/code/measure_memorisation.py --seeds 0,1,2,3,4
$PY results/HestonMultiAsset/CSDI/code/render_readme.py
```

`collect_artifacts.py` is a **gate**, not a formality: it exits non-zero on a contract
violation, and `compute_all_multiasset.py` costs ~55 minutes and will produce normal-looking
numbers on a broken array.
