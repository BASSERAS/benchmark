# Fourier Flow — Paper Reimplementation (Stocks)

Reproduction of the **Fourier Flow** paper result on the **Stocks** dataset using
the *official* code, verbatim from the authors' repository.

- **Paper:** *Generative Time-series Modeling with Fourier Flows* — Alaa, Chan,
  van der Schaar, ICLR 2021.
- **Official code:** `github.com/ahmedmalaa/Fourier-flows` (mirrored here under
  `../code/reference/`).
- **This run:** `metric/reproduce_stock.py` — train → sample → score, CPU only
  (Fourier Flow uses `numpy.fft`; no GPU), 5 independent experiments.

---

## ⚠️ Reproduction caveat — released MLP, not the paper's BiRNN

The paper's Section 3 describes the spectral coupling networks as **bidirectional
RNNs**. The **released** `FourierFlow` class we reproduce with actually wires the
couplings as **MLPs** (`SpectralFilter`: `Linear → Sigmoid → Linear → Sigmoid →
Linear`); the BiRNN lives in a separate, *unused* `AttentionFilter`/`TimeFlow`
path. Per the locked task decision we run the **released code as-is** — the MLP
version is what produced Table 2. This is a paper-vs-code discrepancy in the
authors' own release, not a change we made.

One additional compatibility shim: the paper used torch 1.3.1 / numpy 1.18.5;
`gpu-venv` has torch 2.13 / numpy 2.4. We call
`torch.distributions.Distribution.set_default_validate_args(False)` to match the
old-torch default. No model math is touched.

---

## 1. Paper metrics (as defined in the paper, Table 2)

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **F-score** | Sajjadi precision/recall support-overlap F1: per-timestep min/max intervals of real vs synthetic, harmonic mean of coverage both ways (`metrics/PRcurve.computeF1`). | higher = better |
| **MAE** | TSTR: train a 2-layer LSTM (100 units) on **synthetic** to predict the next step, evaluate **MAE** on **real** (`metrics/MAE.computeMAE`). | lower = better |

Reported as **mean ± 95 % CI over 5 experiments** (`scipy.stats.t`). The dataset
`X` is loaded **once** with a fixed seed (identical across experiments, as in the
released driver); each experiment re-seeds torch so model init + sampling vary.

Target (paper Table 2, Fourier flow / Stocks): **F-score 0.984, MAE 0.009**.

---

## 2. Method hyperparameters (from the released `run_experiment_2.py`)

Taken **verbatim** from `FF_model_params["stock"]` / `FF_train_params["stock"]`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `hidden` | 200 | width of the coupling MLPs |
| `n_flows` | 3 | number of spectral coupling layers |
| `normalize` | True | per-bin spectral standardisation |
| `fft_size` | T + 1 = 101 | DFT length (odd; see below) |
| `epochs` | 1000 | full-batch max-likelihood epochs |
| `batch_size` | 500 | **ignored** — `fit()` is full-batch |
| `learning_rate` | 1e-3 | Adam, ExponentialLR γ=0.999 |

**Why `fft_size` is odd (T+1).** `real_data_loading` prepends a 0 to each length-T
window (`np.hstack((0, series))`), giving length T+1 = 101. This is **required**,
not cosmetic: `FourierFlow.forward` reshapes the cropped DFT output to size
`d+1`, which is only consistent for **odd** input length. It also gives the
imaginary-DC bin float jitter so `normalize=True` does not divide by an exactly
zero std. (The same fact drives the Heston adaptation — see `../code/README.md`.)

Training wall-clock: **~256 s per experiment** (CPU, single process).

---

## 3. Dataset

**Stocks** (Google daily prices — the standard TimeGAN/Fourier-Flow benchmark),
loaded and windowed **verbatim** from the released `run_experiment_2.py`
(`real_data_loading("stock", 100)`): `loadtxt(stock_data.csv)` → reverse to
chronological → per-feature min-max → sliding windows of length 100 → random
permutation → keep the GOOG open column → prepend 0 → **length-101 univariate
sequences**.

- Loader: `../code/reference/data/stock_data.csv` (shipped with the release).
- The reproduction script regenerates `X` on the fly with `--data-seed 42` for a
  fixed permutation; no separate `.npy` is cached.

---

## 4. Results — ours vs paper

| Dataset | Metric | **Ours (released FF code, 5 exps)** | **Paper (Table 2)** | Verdict |
|---------|--------|-------------------------------------|---------------------|---------|
| Stocks | F-score ↑ | **0.9920 ± 0.0017** | 0.984 | **matches** ✓ (ours slightly higher) |
| Stocks | MAE ↓ | **0.0084 ± 0.0007** | 0.009 | **matches** ✓ |

Per-experiment (`results/stock_repro_summary.json`):

| exp | F-score | MAE | loss_last | sec |
|-----|---------|-----|-----------|-----|
| 0 | 0.9934 | 0.0082 | 22.52 | 255.2 |
| 1 | 0.9906 | 0.0078 | 22.81 | 257.7 |
| 2 | 0.9921 | 0.0084 | 22.94 | 260.2 |
| 3 | 0.9908 | 0.0093 | 23.61 | 256.5 |
| 4 | 0.9933 | 0.0083 | 22.92 | 256.7 |

**Reproduced.** Both metrics land on the paper's Table 2 values: the F-score is
0.9920 ± 0.0017 vs the paper's 0.984 (ours marginally *higher*, i.e. slightly
better support overlap), and the MAE is 0.0084 ± 0.0007 vs the paper's 0.009 —
within the 95 % CI. Running the released MLP `FourierFlow` with the released
Stocks hyperparameters therefore matches the published result; the paper-vs-code
BiRNN/MLP discrepancy (§ caveat) does **not** prevent reproduction because Table 2
was itself produced by the released MLP path.

---

## 5. How to reproduce

```bash
cd metric
# reference dir must be importable (SequentialFlows, run_experiment_2, metrics/)
PYTHONPATH=../../code/reference OMP_NUM_THREADS=3 \
  taskset -c 0-2 /home/tbasseras/gpu-venv/bin/python reproduce_stock.py \
    --T 100 --n-exps 5 --n-samples 10000 --epochs 1000 \
    --data-seed 42 --out ../results/stock_repro.json
```

To run the 5 experiments in parallel (one core-pinned process each), launch five
copies with `--n-exps 1 --exp-base k` for k = 0..4 and aggregate afterward
(the driver saves incrementally, so per-`exp_k.json` files can be pooled).

---

## 6. Files

| Path | Content |
|------|---------|
| `metric/reproduce_stock.py` | end-to-end script (load → train → sample → score F1 + MAE) |
| `results/exp_0.json` … `exp_4.json` | per-experiment scores |
| `results/stock_repro_summary.json` | aggregated F-score / MAE mean ± 95 % CI + paper reference |
| `results/exp_{k}.log` | per-experiment stdout logs |
