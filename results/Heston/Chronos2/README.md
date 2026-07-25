# Chronos-2 — Forecaster Reference on Heston

**Chronos-2 is a *forecaster reference*, not a generative method.** Every other entry in this benchmark
is an **unconditional generator** whose forecasting ability is measured *indirectly* through
Path-Shadowing Monte-Carlo (PS-MC). Chronos-2 answers *how good is a purpose-built conditional forecaster
on the same task?* by forecasting the Heston future **directly** — it is the "best forecaster" yardstick
the generator PS-MC rows are compared against.

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.
Forecasts are made on the **test set** (`dataset/Heston/heston_S_test_8192x128.npy`).

**Model.** Chronos-2 — a pretrained encoder-decoder T5-style probabilistic **conditional forecaster** with
group attention (Ansari, Turkmen, Shchur, et al., Amazon Science, **2025**, *Chronos-2: From Univariate to
Universal Forecasting*, arXiv:2510.15821, `github.com/amazon-science/chronos-forecasting`). Checkpoint
`amazon/chronos-2` (**~120M params — the largest model in the benchmark**). Given a context window it emits
21 predictive quantiles per future step via `predict_quantiles`. See
[`../../../methods/Chronos2/README.md`](../../../methods/Chronos2/README.md).

> **The retired unconditional-generator experiment is archived under [`old/`](old/)** (generator metrics,
> curve-B, generator-pool PS-MC, diagnostic + PCA/t-SNE plots) and
> [`../../../methods/Chronos2/old/`](../../../methods/Chronos2/old/) (generator code, paths, losses). It is
> **excluded from every A/B/PS-MC table and win-count** in the root and results READMEs.

---

## Forecaster-reference protocol

1. **Prefix = 64 steps** (0–63) of each real test path — matches the PS-MC query prefix length.
2. **Single-shot forecast** of the next **64 steps** (64–127) in **price space** with one
   `predict_quantiles(prediction_length=64, quantile_levels=<21 levels>)` call. Single-shot has no
   autoregressive feedback, so it is stable.
3. **Ensemble K = 77** per path by piecewise-linear inverse-CDF sampling over the 21 quantile levels
   (matches the PS-MC ensemble size).
4. **Scored with the identical harness** as the generators — the same `crps` / `evaluate_horizon` /
   `naive_baseline` from `path_shadowing.py`, so CRPS and the naive random-walk (RW) baseline are directly
   comparable to the generator PS-MC rows.

Two reference variants, reported as **two rows**: **zero-shot** (pretrained checkpoint, a single model →
one point value) and **fine-tuned** (5 per-seed checkpoints from the retired experiment, mean ± std).
Horizons **H = 32** and **H = 64** are cut from the same 64-step forecast.

---

## Reference scores — CRPS / MAE / RMSE (price space, 64-step prefix)

Lower is better. CRPS is the energy-score CRPS; the RW baseline repeats the last prefix value (its
CRPS = MAE). Read from [`forecaster_summary.json`](forecaster_summary.json).

| Forecaster | CRPS H=32 ↓ | CRPS H=64 ↓ | MAE H=32 | MAE H=64 | RMSE H=32 | RMSE H=64 |
|------------|:-----------:|:-----------:|:--------:|:--------:|:---------:|:---------:|
| **Chronos-2 zero-shot** | 2.996 | 4.234 | 4.074 | 5.692 | 5.507 | 7.700 |
| **Chronos-2 fine-tuned** *(5 seeds)* | **2.760 ± 0.0001944** | **3.980 ± 0.0004099** | 3.740 ± 0.0001595 | 5.256 ± 0.0004472 | 5.049 ± 0.0004853 | 7.088 ± 0.0006329 |
| *RW baseline* | *3.738* | *5.246* | *3.738* | *5.246* | *5.040* | *7.066* |

**Both variants beat the naive random walk on CRPS at both horizons** — a real conditional forecaster adds
value over "tomorrow = today". Fine-tuning sharpens the per-step scale (CRPS 2.996 → 2.760 at H=32,
4.234 → 3.980 at H=64); its 5 seeds are near-identical (std ≈ 2e-04, deterministic data order). The
fine-tune checkpoints were trained at `prediction_length = 16`, so their single-shot 64-step forecast shows
mild 16-step **block structure** in the per-step CRPS curve, yet still improve on zero-shot overall.

### Forecaster reference vs generator Path-Shadowing MC

Same task, prefix length, CRPS and RW baseline — directly comparable:

| CRPS ↓ (price space) | H=32 | H=64 |
|----------------------|:----:|:----:|
| **Forecaster reference** | | |
| Chronos-2 zero-shot | 2.996 | 4.234 |
| Chronos-2 fine-tuned *(5 seeds)* | 2.760 ± 0.0001944 | 3.980 ± 0.0004099 |
| **Best generator PS-MC** | | |
| LS4 (path-shadowing, best of 10 generators) | **2.704 ± 0.002510** | **3.763 ± 0.005851** |
| **Baselines** | | |
| Random walk (naive) | 3.738 | 5.246 |
| Perfect (oracle Heston pool) | 2.721 ± 0.004183 | 3.788 ± 0.006463 |

**Headline finding.** A purpose-built, fine-tuned foundation forecaster (~120M params) **does not beat the
best generator's path-shadowing** at either horizon: LS4 PS-MC (2.704 / 3.763) edges out fine-tuned
Chronos-2 (2.760 / 3.980) and reaches the Perfect oracle floor (2.721 / 3.788) while the forecaster does
not. Path-Shadowing MC over a well-trained unconditional generator is a **competitive conditional
forecaster in its own right** — the generator route is not merely a fidelity check.

---

## Plots

**Example forecasts (fine-tune seed 0)** — 64-step real prefix (blue solid) + real future (blue dashed) +
K = 77 inverse-CDF forecast members (red) + forecast mean, 4 random test paths:

![Forecaster example](plots/forecaster_example.png)

**CRPS per forecast step** (zero-shot vs fine-tune, ±1 std band, RW baseline lines). The mild 16-step steps
in the fine-tune curve are the `prediction_length = 16` training footprint:

![CRPS per step](plots/forecaster_crps_per_step.png)

---

## Validating the port — paper zero-shot reproduction

> ⚠️ **Chronos-2's paper metrics are zero-shot forecasting accuracy on standard benchmarks, not Heston.**
> The arXiv:2510.15821 paper reports **MASE** and **WQL** on forecasting datasets (Chronos-datasets), not a
> generative fidelity score. We validated our port by reproducing its **zero-shot** MASE/WQL on five
> held-out datasets, against the paper's `chronos-bolt-base` reference numbers.

| Dataset | MASE ours | MASE ref | WQL ours | WQL ref |
|---------|:---------:|:--------:|:--------:|:-------:|
| ercot | 0.7765 | 0.6933 | 0.02457 | 0.02142 |
| exchange_rate | 1.8788 | 1.7095 | 0.01214 | 0.01201 |
| monash_australian_electricity | 0.6135 | 0.7403 | 0.02878 | 0.03584 |
| monash_traffic | 0.8308 | 0.7843 | 0.24254 | 0.23149 |
| nn5 | 0.5561 | 0.5764 | 0.14489 | 0.15005 |

The zero-shot MASE/WQL land in the **same regime** as the `chronos-bolt-base` reference across all five
datasets, confirming the port loads and runs the pretrained checkpoint faithfully. Full write-up:
[`../../../methods/Chronos2/paper_reimplementation/README.md`](../../../methods/Chronos2/paper_reimplementation/README.md).

---

## Files

| File | Description |
|------|-------------|
| `forecaster_summary.json` | Zero-shot + fine-tune (mean ± std, 5 seeds) CRPS/MAE/RMSE at H=32/H=64 + RW baseline + per-step CRPS |
| `forecaster_seed_{i}.json` | Per-seed fine-tune forecaster scores |
| `plots/forecaster_example.png` | 4 example paths: prefix + K=77 forecast members + mean (fine-tune seed 0) |
| `plots/forecaster_crps_per_step.png` | CRPS per forecast step, zero-shot vs fine-tune, RW baseline lines |
| `old/` | RETIRED unconditional-generator artifacts (metrics, curve-B, generator-pool PS-MC, plots) |

→ Cross-method comparison with the 10 generators: [`results/README.md`](../../README.md)
