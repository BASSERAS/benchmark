# TimeGAN on Heston

PyTorch reimplementation of TimeGAN (Yoon et al., NeurIPS 2019) trained on 8192 Heston paths.

## Results structure

```
results/
├── generated_paths/seed_{0..4}/
│   ├── generated_paths_8192x128.npy   shape (8192, 128), original price scale
│   └── metadata.json
├── params/
│   ├── seed_{i}_model.pt              full state_dict
│   └── seed_{i}_config.json
└── losses/
    ├── seed_{i}_losses.csv            step,phase,e_loss,s_loss,g_loss,d_loss
    └── loss_convergence.png
```

## Training

Seeds 0-4 trained on 2 A100 80GB GPUs:
- Phase 1: Embedding pre-training (5000 steps)
- Phase 2: Supervisor pre-training (5000 steps)
- Phase 3: Joint adversarial (10000 steps)

Reference TF1 implementation (original paper code) is in `reference/`.
