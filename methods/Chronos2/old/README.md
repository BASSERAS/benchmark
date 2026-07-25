# RETIRED — Chronos-2 as an unconditional generator

**This folder archives a retired experiment.** Chronos-2 was originally fine-tuned into an
**unconditional Heston generator** and scored like every other generative method (A1–A34 + B curves +
Path-Shadowing MC, 5 seeds). That framing was **retired**: Chronos-2 is a *conditional forecaster*, not
a generator, and forcing it to emit unconditional paths is both conceptually wrong and empirically weak.

Chronos-2 now serves as a **forecaster reference** — a direct conditional forecast of the Heston future
that the generator Path-Shadowing-MC rows are measured against. See the current
[`../README.md`](../README.md) for that role.

Nothing here is referenced by the root README, the results README, or any A/B/PS-MC table or win-count.
It is kept only for provenance and reproducibility.

---

## What the retired experiment did

**Generation scheme — autoregressive rollout in log-space from a real prefix.** Each generated series
started from the **first 16 steps of a real test path** (`dataset/Heston/heston_S_test_8192x128.npy`).
At every step the fine-tuned model forecast the next-step quantiles (21 trained levels); one Monte-Carlo
sample was drawn per series by inverse-CDF on those quantiles, appended, and the window advanced —
repeated to length 128. The rollout ran in **log-price space** and the output was exponentiated back to
price scale. Feature dim 1, sequence length 128.

**Fine-tune only.** Chronos-2's robust scaling is level-proportional, so a **zero-shot** autoregressive
rollout on a low-vol geometric process like Heston runs away multiplicatively even in log-space.
Fine-tuning (`finetune_mode="full"`, 1 000 steps, lr 1e-4, batch 256, prediction length 16) fixed the
per-step scale and was required for a stable 112-step rollout. Only the fine-tuned variant was archived.

## Why it was retired

- **A forecaster is not a generator.** Chronos-2 is trained to predict a *conditional* next-step
  distribution given a context. Sampling that conditional autoregressively to fabricate an
  *unconditional* path is off-label use.
- **Stacking one-step forecasts over-disperses the marginal law.** Compounding a slightly-too-wide
  per-step quantile over 112 autoregressive steps produced the **worst log-return-histogram fit in the
  benchmark** (curve-B MSE ≈ 20704, ~10× the next-worst) and a kurtosis ratio ≈ 0.099 (target 1.0) — the
  generated returns were ≈ 10× too leptokurtic and the price std ≈ 1.5× the real. One seed-3 path even
  ran away to a terminal price ≈ 1323.
- **The honest fix is to change the question.** Instead of asking Chronos-2 to *generate*, we ask it to
  *forecast* the real future directly and use that as the reference score for the best forecaster — which
  is what Path-Shadowing MC on the generators approximates. That is the current
  [`../README.md`](../README.md).

For the record, the retired generator's own Path-Shadowing MC (retrieval over its *generated pool*, not a
direct forecast) beat the naive RW slightly — CRPS H=32 **3.719** (RW 3.738), H=64 **5.218** (RW 5.246) —
but well behind the best generator LS4 (2.704 / 3.763) and behind the *direct* fine-tuned Chronos-2
forecast now in [`../README.md`](../README.md) (2.760 / 3.980).

---

## Archived contents

```
methods/Chronos2/old/
├── code/
│   ├── train_heston.py     fine-tune + log-space autoregressive rollout generator
│   ├── plot_losses.py      loss-convergence plot generator
│   └── README.md           original generator code notes (architecture, fine-tune, rollout)
├── generated_paths/seed_{0..4}/   generated_paths_8192x128.npy + metadata.json
├── losses/                 seed_{i}_losses.csv + loss_convergence.png
└── path_shadowing/run_eval.py     generator-pool PS-MC runner (model-agnostic retrieval)
```

Scored generator outputs (metrics, curve-B, PS-MC results, diagnostic + PCA/t-SNE plots) are archived
under [`../../../results/Heston/Chronos2/old/`](../../../results/Heston/Chronos2/old/).
