# Benchmark — Time Series Generation on Heston

Public benchmark repo for time-series generation quality evaluation.

## Structure

```
benchmark/
├── dataset/          8192 Heston paths, seq_len=128 (canonical parameters)
├── TimeGan/          PyTorch TimeGAN (5 seeds) + reference TF1 code
│   └── results/      generated_paths/, params/, losses/
└── metrics/          14 evaluation metrics (A1-A14) + Heston A15
    └── results/      per-seed JSON, summary CSV, PCA/t-SNE plots
```

## Dataset

Heston stochastic volatility (Euler-Maruyama, full-truncation):
- mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, S0=100, v0=0.04, dt=1/250
- 8192 paths x 128 time steps, seed=0

## TimeGAN

PyTorch reimplementation (GPU-enabled) matching Yoon et al. NeurIPS 2019:
- Embedder/Recovery/Generator/Supervisor/Discriminator — 3-layer GRU, hidden_dim=24
- 3 training phases: embedding (5k) + supervised (5k) + joint adversarial (10k)
- Reference TF1 code in `TimeGan/reference/`

## Metrics (A1-A14 + A15)

| ID | Metric | Type |
|----|--------|------|
| A1 | Full path MMD² | distributional |
| A2 | Terminal MMD² | distributional |
| A3 | Increment MMD² | distributional |
| A4 | Volatility MMD | distributional |
| A5 | Terminal SWD | distributional |
| A6 | Path SWD | distributional |
| A7 | Terminal Cov Error | statistical |
| A8 | Terminal Mean RMSE | statistical |
| A9 | Return Std Error | statistical |
| A10 | Return Kurtosis Error | statistical |
| A11 | ACF Error (abs returns) | temporal |
| A12 | ACF Error (sq returns) | temporal |
| A13 | Discriminative Score (GRU + MLP) | downstream |
| A14 | Predictive Score TSTR (GRU + MLP) | downstream |
| A15 | Teacher-Sigma Corr/RMSE | Heston-specific |

## Usage

```bash
# Generate dataset
python dataset/generate_heston.py

# Train TimeGAN (5 seeds, 2 A100 GPUs in parallel)
python TimeGan/train.py

# Compute all metrics
python metrics/compute_all.py
```

## Reference
- Yoon et al., "Time-series Generative Adversarial Networks", NeurIPS 2019
- tsg-benchmark: https://github.com/BASSERAS/tsg-benchmark
