# LS4 — source and implementation notes (d = 8)

## Source

**Paper.** Zhou, Kang, Molina-Salgado, Wu, Ermon, Grover — *Deep Latent State Space
Models for Time-Series Generation*, ICML 2023.

**Code.** The released LS4 implementation, vendored unmodified into
`reference_models/` from `methods/LS4/code/reference/models/` in this repository:

```
reference_models/ls4.py        the VAE module (prior, posterior, decoder)
reference_models/s4models.py   S4Model / Model backbones
reference_models/s4.py         S4 kernel
reference_models/s4d.py        S4D / S4DJoint
reference_models/seq_unet.py   SequentialUnet
reference_configs/vae_solarweekly_released.yaml   the released solar_weekly preset
```

The preset is the released `solar_weekly` configuration — the one that reproduced
the paper's Solar Weekly marginal score (paper 0.0459, ours ≈ 0.045): `sigma = 0.1`,
`d_state = 64`, `d_model = 64`, `n_layers = 4`, `backbone = autoreg`, `s4_type = s4`,
`latent_type = split`; optimiser `AdamW(lr = 1e-3, wd = 0)` +
`ReduceLROnPlateau(patience = 20, factor = 0.5)` + `EMA(lamb = 0.99, start_step = 200)`,
batch 128.

## Files

| file | role |
|------|------|
| `train_multiasset.py` | trains one seed and generates its 8192 paths in the same process |
| `collect_artifacts.py` | rebuilds `../losses/generation_time.csv`, normalises `metadata.json`, checks the §4 contract |
| `plot_losses.py` | `../plots/loss_convergence.png`, 5 seeds overlaid, x-axis = step |
| `measure_memorisation.py` | NN-ratio diagnostic; estimator byte-identical to SBTS's |
| `plot_diagnostics_multiasset.py` | copied from `SBTS/code/`, run with `--method LS4` |
| `render_readme.py` | copied from `SBTS/code/`, adapted per §8; generates `../README.md` |
| `reference_models/`, `reference_configs/` | vendored LS4 release, unmodified apart from the documented Cauchy patch |

## Provenance check — was the right LS4 actually run?

Before any d = 8 work, `methods/LS4/code/train_heston.py --seed 0 --epochs 5` was
re-run and compared against the committed `methods/LS4/losses/seed_0_losses.csv`.
It reproduced **bit-exactly**:

| epoch | total | kld | nll | mse |
|------:|------:|----:|----:|----:|
| 0 | 24.563837595283985 | 0.4643734924029559 | 24.09946396201849 | 0.5096622176934034 |
| 4 | 0.03708194966020528 | 0.35022108210250735 | −0.3131391337956302 | 0.02141012719948776 |

together with `params = 2146857`, `scaler_mu = 101.3255`, `scaler_sigma = 9.9717`,
all matching `methods/LS4/weights/seed_0_config.json`. The vendored
`reference_models/` is therefore the code that produced the committed d = 1
results, not a lookalike.

## The four questions (MULTIASSET_GUIDELINE §0)

### 1. Does the method natively accept `(N, T, d)` input?

**Yes — one joint model over all 8 channels.** `weights/seed_{i}_config.json`
records `"joint_or_per_asset": "joint"`. LS4's encoder and decoder are
`d_input`/`d_output` parameterised, so `d = 8` is a native configuration, not an
adaptation.

One trap worth spelling out, because getting it wrong is silent. The released YAML
ties three fields to a single anchor:

```yaml
data:  { channel: &channel 1 }
model:
  in_channels: *channel
  decoder:  { decoder:   { d_output: *channel } }
  encoder:  { posterior: { d_input:  *channel } }
```

and four more to `&z_dim`. OmegaConf resolves anchors at load time, so setting
`config.model.in_channels = 8` alone leaves a **1-channel decoder head** and a
1-channel posterior input: the model trains happily and is wrong. The d = 1
trainer sets only `in_channels`, and is correct there purely because 1 == 1.
`build_config()` in `train_multiasset.py` sets **all seven** anchored fields
explicitly.

### 2. Does any hyperparameter fail to cross dimension?

**Yes — `z_dim`, and it fails badly.** `weights/seed_{i}_config.json` records
`"retuned_for_d8": ["in_channels", "z_dim", "scaler"]` and
`"paper_hyperparams": false`.

- **`in_channels` 1 → 8.** Forced by the data; not a tuning choice.
- **`z_dim` 5 → 40.** Selected on the **validation split**
  (`heston_ma_S_val_8192x252x8.npy`), never on test. The criterion was written to
  [`../losses/selection_criterion.md`](../losses/selection_criterion.md) *before*
  the sweep ran; the numbers are in `../losses/zdim_selection.json`. The released
  `z_dim = 5` scores a validation ELBO of **+71.62** against **−5.37** for
  `z_dim = 40`, and its train ELBO (66.93) essentially equals its validation ELBO
  (66.99) — under-capacity, not overfitting. A 5-dimensional latent cannot carry 8
  correlated channels.
  The ordering was monotone (40 < 32 < 16 < 5) with no interior optimum inside the
  tested range, so `z_dim = 40` is the *largest tested* candidate rather than a
  located maximum. Larger latents were not tried; that boundary is disclosed here
  rather than dressed up as convergence.
- **`scaler` global → per-channel `(μⱼ, σⱼ)`.** The eight marginals have different
  price scales (σ from 11.14 to 18.55); one global σ would let the widest asset
  dominate the reconstruction term. A per-channel **affine** map leaves the
  cross-asset correlation matrix exactly unchanged, so the target coupling Σˢ —
  the point of the dataset, and what A20 scores — survives both the
  standardisation and its inverse.

Everything else is the released preset unchanged, including the **epoch budget of
100**, which is exactly what the committed d = 1 run used.

### 3. What is the memorisation risk?

**Structurally low, and measured anyway.** LS4 never touches a training path at
generation time: it draws `z` from the prior and decodes. It can only memorise by
having encoded the training set into 2.2 M weights — which 100 epochs over 8192
paths can certainly do in principle.

`measure_memorisation.py` is run regardless, and its estimator is **byte-identical
to `SBTS/code/measure_memorisation.py`** (same log-return space, same
median-nearest-neighbour statistic, same train/test splits), so the two columns are
directly comparable. Result in `../losses/memorisation.json`.

One asymmetry to keep in view when reading the two numbers. For SBTS, memorisation
is the *default* failure mode — it is an explicit kernel average over training
paths, and it scores 0.2189 ± 0.0003 at d = 8. For LS4 the exact-duplicate count is
close to uninformative: the decoder output is a continuous function of a Gaussian
sample, so bitwise equality has probability zero by construction. **Only the ratio
carries information here.**

### 4. Compute budget

Measured before launching, not estimated after:

| | value |
|---|---|
| params | 2 205 832 |
| sec/epoch (d = 8, T = 252, 8192 paths, one A100) | 7.6 train; ≈ 9.5 with per-epoch validation |
| train, 100 epochs | ≈ 16 min/seed |
| 5 seeds on 2 GPUs (3 + 2) | ≈ 50 min |
| z_dim screening sweep (4 configs, 20 epochs, 25 % subset) | ≈ 4 min |
| metrics | ≈ 11 min/seed per the guideline |

Hardware limits respected throughout: **2 GPUs**, `taskset -c 0-7` / `8-15`
(16 cores), `OMP_NUM_THREADS=8`. `nvidia-smi` was checked before each launch; the
final run was moved to GPUs 2/3 because another user took GPU 0 mid-session.

## Deviations from the released d = 1 implementation

**Cauchy kernel, `reference_models/s4.py` line 795.** The naive (pure-PyTorch)
Cauchy kernel sums over conjugate pole *pairs*, `cauchy_naive(_conj(v), z, _conj(w))`,
matching the keops/CUDA path. Not cosmetic: `model.generate` rolls the prior through
`latent.step` (STEP-mode), where the unpatched kernel disagrees with conv-mode — so
generation and training would otherwise use different dynamics. This patch is
inherited from the d = 1 port, not introduced here.

**Invertible scaler.** The released `normalize_per_seq` preset has an identity
decode that cannot map prior samples back to price scale. The d = 1 port replaced it
with a global standardise `(X − μ)/σ`; this port uses the per-channel version, for
the reason given under question 2.

**S0 rescaling.** GUIDELINE §4 requires `S[:, 0, :] == 100.0` exactly. LS4 prior
samples do not satisfy this — at d = 1 the generated `S0` spread over
`[99.30, 100.48]`, std 0.055 — so each generated path is rescaled per asset,
`S ← S · 100 / S[:, 0, :]`, and `metadata.json` records `"s0_rescaled": true`.
A per-path per-asset **constant** multiplier is exactly a shift of the log-price
level, so every log-return is bit-identical and metrics A1–A25 / A27–A34, all of
which are log-return based, are unaffected. This is a contract fix, not a
performance fix, and it cannot flatter the results.

## Deviation from the §1 layout, stated rather than papered over

§1 lists `run_all_multiasset.py` (generation, 5 seeds) as a separate file. **There
is no such file here.** LS4 generates inside `train_multiasset.py`, from the
in-memory EMA model immediately after training. Splitting it out would mean
reloading a checkpoint and duplicating the scaler inversion and S0 rescaling in a
second place where the two copies could silently drift — the exact class of bug the
guideline warns about elsewhere. `collect_artifacts.py` fills the slot for
post-generation bookkeeping and contract verification. This is the only deviation
from §1.

## Reproduce

```bash
cd /home/tbasseras/benchmark/results/HestonMultiAsset/LS4/code
PY=/home/tbasseras/gpu-venv/bin/python

# 1. z_dim screening (regenerates ../losses/zdim_selection.json)
for z in 5 16 32 40; do
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 taskset -c 0-7 $PY train_multiasset.py \
      --seed 0 --z_dim $z --epochs 20 --frac 0.25 --val --tag zdim$z --no_generate
done

# 2. final 5 seeds, 2 GPUs
for s in 0 2 4; do CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 taskset -c 0-7  $PY \
    train_multiasset.py --seed $s --z_dim 40 --epochs 100 --val; done &
for s in 1 3;   do CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 taskset -c 8-15 $PY \
    train_multiasset.py --seed $s --z_dim 40 --epochs 100 --val; done &
wait

# 3. bookkeeping, figures, metrics
$PY collect_artifacts.py                       # must print "5 rows"
$PY plot_losses.py
$PY measure_memorisation.py --seeds 0,1,2,3,4
$PY plot_diagnostics_multiasset.py --method LS4
cd /home/tbasseras/benchmark && CUDA_VISIBLE_DEVICES=2 $PY \
    metrics/compute_all_multiasset.py --method LS4 --dataset HestonMultiAsset --seeds 5
$PY results/HestonMultiAsset/LS4/code/render_readme.py
```

`train_multiasset.py` refuses to run unless the per-asset parameter digest is
`231da80bdedf22e9`. That guard exists because `np.random.Generator` is **not**
version-stable: a different digest means the dataset was regenerated under a
different numpy, and nothing would be comparable to the SBTS column. Do not bypass it.
