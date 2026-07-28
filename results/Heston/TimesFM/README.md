# TimesFM, Forecaster Reference on Heston

**TimesFM is a *forecaster reference*, not a generative method.** Every other entry in this benchmark
(except Chronos-2) is an **unconditional generator** whose forecasting ability is measured *indirectly*
through Path-Shadowing Monte-Carlo (PS-MC). TimesFM, like Chronos-2, answers *how good is a purpose-built
conditional forecaster on the same task?* by forecasting the Heston future **directly**. It sits in the
**forecaster category** alongside Chronos-2 and is one of the "best forecaster" yardsticks the generator
PS-MC rows are compared against.

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.
Forecasts are made on the **test set** (`dataset/Heston/heston_S_test_8192x128.npy`).

**Model.** TimesFM, a pretrained decoder-only patched probabilistic **conditional forecaster** (Das, Kong,
Sen, Zhou, Google Research, **ICML 2024**, *A decoder-only foundation model for time-series forecasting*,
arXiv:2310.10688v4, `github.com/google-research/timesfm`). **Both released checkpoints are run through the
same protocol:** `google/timesfm-1.0-200m-pytorch` (the **paper model** and the **headline reference**,
~200M params; 20 layers, model_dims 1280, input-patch 32, output-patch 128, positional embeddings on) and
`google/timesfm-2.0-500m-pytorch` (newer, **~500M params**; 50 layers, positional embeddings off, not in
the paper). Given a context window each emits a mean plus **9 predictive quantiles** (levels 0.1…0.9) per
future step. See [`../../../methods/TimesFM/README.md`](../../../methods/TimesFM/README.md).

> **There is no `old/` folder.** Unlike Chronos-2, TimesFM was never run as an unconditional generator in
> this benchmark, it enters purely as a forecaster reference, so there is no retired generator experiment
> to archive. TimesFM appears in the forecaster category only and is **excluded from every A/B/PS-MC
> generator table and win-count** in the root and results READMEs.

---

## Forecaster-reference protocol

1. **Prefix = 64 steps** (0-63) of each real test path, matches the PS-MC query prefix length.
2. **Single-shot forecast** of the next **64 steps** (64-127) in **price space** with one
   `tfm.forecast(context, freq=[0]*N)` call (TimesFM applies reversible instance-norm on the context
   internally). Single-shot has no autoregressive feedback, so it is stable.
3. **Ensemble K = 77** per path by piecewise-linear inverse-CDF sampling over the **9** quantile levels
   TimesFM emits (the mean column is dropped; only the 9 quantile heads feed the ensemble). K = 77 matches
   the PS-MC ensemble size.
4. **Scored with the identical harness** as the generators, the same `crps` / `evaluate_horizon` /
   `naive_baseline` from `path_shadowing.py`, so CRPS and the naive random-walk (RW) baseline are directly
   comparable to the generator PS-MC rows.

Two reference variants **per checkpoint** (→ four rows total): **zero-shot** (pretrained checkpoint, a
single model → one point value) and **fine-tuned** (5 per-seed full fine-tunes on the 8 192 Heston training
paths, 1 000 Adam steps, lr 1e-4, wd 0.01, batch 256, masked MSE + 9-quantile pinball loss, mean ± std
across seeds). The **1.0-200m** and **2.0-500m** checkpoints use the identical recipe (only num_layers and
the positional-embedding flag differ). Horizons **H = 32** and **H = 64** are cut from the same 64-step
forecast.

---

## Reference scores, CRPS / MAE / RMSE (price space, 64-step prefix)

Lower is better. CRPS is the energy-score CRPS; the RW baseline repeats the last prefix value (its
CRPS = MAE). **Both released checkpoints** are run through the identical forecaster protocol:
**1.0-200m** (the paper model, ~200M params, the headline reference) and **2.0-500m** (newer, ~500M,
not in the paper). 1.0-200m read from [`forecaster_summary.json`](forecaster_summary.json); 2.0-500m from
[`forecaster_summary_timesfm-2.0-500m-pytorch.json`](forecaster_summary_timesfm-2.0-500m-pytorch.json).

| Forecaster | CRPS H=32 ↓ | CRPS H=64 ↓ | MAE H=32 | MAE H=64 | RMSE H=32 | RMSE H=64 |
|------------|:-----------:|:-----------:|:--------:|:--------:|:---------:|:---------:|
| **1.0-200m zero-shot** | 3.065 | 4.347 | 4.147 | 5.770 | 5.639 | 7.811 |
| **1.0-200m fine-tuned** *(5 seeds)* | **2.976 ± 0.140** | **4.046 ± 0.139** | 4.039 ± 0.169 | 5.470 ± 0.153 | 5.365 ± 0.174 | 7.338 ± 0.173 |
| 2.0-500m zero-shot | 3.103 | 4.549 | 4.240 | 6.052 | 5.737 | 8.163 |
| 2.0-500m fine-tuned *(5 seeds)* | 3.169 ± 0.312 | 4.440 ± 0.539 | 4.075 ± 0.142 | 5.621 ± 0.237 | 5.479 ± 0.207 | 7.590 ± 0.327 |
| *RW baseline* | *3.738* | *5.246* | *3.738* | *5.246* | *5.040* | *7.066* |

**All four variants beat the naive random walk on CRPS at both horizons**, a real conditional forecaster
adds value over "tomorrow = today". For **1.0-200m**, fine-tuning sharpens the per-step scale (CRPS
3.065 → 2.976 at H=32, 4.347 → 4.046 at H=64) and the 5 seeds show **genuine spread** (std ≈ 0.14): the
fine-tuner draws random minibatches (`rng.integers`) seeded per run, so each seed sees a different data
order, the std is real seed variance, not numerical noise.

**The smaller 1.0-200m is the stronger checkpoint on Heston**, it beats 2.0-500m at every cell, zero-shot
and fine-tuned. On 2.0-500m, full fine-tuning of a ~500M model on only 8 192 short Heston paths is
**unstable**: the 5 seeds' H=32 CRPS ranges 2.875 → 3.643 (std 0.312), and the fine-tuned mean (3.169)
does **not** improve on its own zero-shot (3.103) at H=32, the larger model overfits some seeds and
under-fits others rather than converging tightly. This is why the **1.0-200m checkpoint is used as the
headline TimesFM forecaster reference** everywhere else in the benchmark; 2.0-500m is reported here for
completeness alongside its ETT paper-reproduction numbers.

### Forecaster reference vs generator Path-Shadowing MC

Same task, prefix length, CRPS and RW baseline, directly comparable:

| CRPS ↓ (price space) | H=32 | H=64 |
|----------------------|:----:|:----:|
| **Forecaster reference** | | |
| TimesFM-1.0-200m zero-shot | 3.065 | 4.347 |
| TimesFM-1.0-200m fine-tuned *(5 seeds)* | 2.976 ± 0.140 | 4.046 ± 0.139 |
| TimesFM-2.0-500m zero-shot | 3.103 | 4.549 |
| TimesFM-2.0-500m fine-tuned *(5 seeds)* | 3.169 ± 0.312 | 4.440 ± 0.539 |
| Chronos-2 zero-shot | 2.996 | 4.234 |
| Chronos-2 fine-tuned *(5 seeds)* | 2.760 ± 0.0001944 | 3.980 ± 0.0004099 |
| **Best generator PS-MC** | | |
| LS4 (path-shadowing, best of 10 generators) | **2.704 ± 0.002510** | **3.763 ± 0.005851** |
| **Baselines** | | |
| Random walk (naive) | 3.738 | 5.246 |
| Perfect (oracle Heston pool) | 2.721 ± 0.004183 | 3.788 ± 0.006463 |

**Headline finding holds with a second forecaster.** TimesFM (best checkpoint, 1.0-200m) is **comparable to
but slightly behind Chronos-2** at both horizons and both variants (fine-tuned 2.976 / 4.046 vs Chronos-2
2.760 / 3.980); the larger 2.0-500m checkpoint is weaker still on this task (3.169 / 4.440 fine-tuned). And,
crucially, **neither purpose-built foundation forecaster, at either size, beats the best generator's
path-shadowing**: LS4 PS-MC (2.704 / 3.763) edges out all of them and reaches the Perfect oracle floor
(2.721 / 3.788) while the forecasters do not. Path-Shadowing MC over a well-trained unconditional generator
is a **competitive conditional forecaster in its own right**, the generator route is not merely a fidelity
check.

---

## Plots

**Example forecasts (fine-tune seed 0)**, 64-step real prefix (blue solid) + real future (blue dashed) +
K = 77 inverse-CDF forecast members (red) + forecast mean, 4 random test paths (**1.0-200m**):

![Forecaster example](plots/forecaster_example.png)

**CRPS per forecast step** (zero-shot vs fine-tune, ±1 std band, RW baseline lines), **1.0-200m**:

![CRPS per step](plots/forecaster_crps_per_step.png)

The same two plots for the **2.0-500m** checkpoint (note the wider fine-tune ±1 std band, reflecting the
unstable 5-seed fine-tune discussed above):

![Forecaster example 2.0-500m](plots/forecaster_example_timesfm-2.0-500m-pytorch.png)

![CRPS per step 2.0-500m](plots/forecaster_crps_per_step_timesfm-2.0-500m-pytorch.png)

---

## Validating the port, paper zero-shot ETT reproduction

> ⚠️ **TimesFM's paper metric is zero-shot forecasting accuracy on standard benchmarks, not Heston.**
> The arXiv:2310.10688v4 headline is **zero-shot MAE** on the ETT long-horizon benchmark (paper **Table 5**).
> We validated our port by reproducing that zero-shot MAE (last-test-window protocol, context 512,
> train-statistic z-normalisation, horizons 96 and 192), for **both** released checkpoints.

| Checkpoint | ETT avg MAE (ours) | Paper `TimesFM(ZS)` avg |
|------------|:------------------:|:-----------------------:|
| `google/timesfm-1.0-200m-pytorch` *(paper model)* | **0.3409** | 0.36 |
| `google/timesfm-2.0-500m-pytorch` *(newer)* | 0.3415 |, (not in paper) |

Our 1.0-200m average (0.3409) matches the paper's `TimesFM(ZS)` average (0.36) closely, confirming the port
loads and runs the pretrained checkpoint faithfully. Full 8-task per-dataset table (both checkpoints vs the
paper's Table 5 `TimesFM(ZS)` column):
[`../../../methods/TimesFM/paper_reimplementation/README.md`](../../../methods/TimesFM/paper_reimplementation/README.md).

---

## Files

| File | Description |
|------|-------------|
| `forecaster_summary.json` | **1.0-200m** zero-shot + fine-tune (mean ± std, 5 seeds) CRPS/MAE/RMSE at H=32/H=64 + RW baseline + per-step CRPS |
| `forecaster_seed_{i}.json` | **1.0-200m** per-seed fine-tune forecaster scores |
| `forecaster_summary_timesfm-2.0-500m-pytorch.json` | **2.0-500m** counterpart of `forecaster_summary.json` |
| `forecaster_seed_{i}_timesfm-2.0-500m-pytorch.json` | **2.0-500m** per-seed fine-tune forecaster scores |
| `plots/forecaster_example.png` | 4 example paths: prefix + K=77 forecast members + mean (1.0-200m, fine-tune seed 0) |
| `plots/forecaster_crps_per_step.png` | CRPS per forecast step, zero-shot vs fine-tune, RW baseline lines (1.0-200m) |
| `plots/forecaster_example_timesfm-2.0-500m-pytorch.png` | 2.0-500m counterpart example-forecasts plot |
| `plots/forecaster_crps_per_step_timesfm-2.0-500m-pytorch.png` | 2.0-500m counterpart CRPS-per-step plot |

→ Cross-method comparison with the 10 generators: [`results/README.md`](../../README.md)
