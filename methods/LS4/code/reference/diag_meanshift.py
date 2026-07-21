"""Root-cause diagnostic for the LS4 Solar Weekly generation mean-shift.
Loads the trained EMA checkpoint and compares:
  - real val data mean/std
  - reconstruction (encode->decode) output mean/std
  - generation (prior sample -> decode) output mean/std
  - posterior latent z stats vs prior-rollout latent z stats
Localizes whether the ~ -1.3 mean shift enters in latent space or the decoder.
"""
import torch, numpy as np
from omegaconf import OmegaConf
from models.ls4 import VAE
from datasets import parse_datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = OmegaConf.load('configs/monash/vae_solarweekly_repro.yaml')

data_objs = parse_datasets(cfg.data, batch_size=cfg.optim.batch_size, device=torch.device('cpu'))
valloader = data_objs['test_dataloader']
cfg.model.n_labels = data_objs.get('n_labels', 1)

def load_vae(key):
    m = VAE(cfg.model).to(device)
    ck = torch.load('outputs_repro/solar_weekly/solar_weekly_repro/checkpoints/ckpt.pth', map_location=device)
    sd = ck[key]
    new = {}
    for k, v in sd.items():
        if k == 'n_averaged':
            continue
        new[k[len('module.'):] if k.startswith('module.') else k] = v
    missing, unexpected = m.load_state_dict(new, strict=False)
    print(f'[{key}] loaded. missing={len(missing)} unexpected={len(unexpected)} epoch={ck.get("epoch")} step={ck.get("step")}')
    m.eval(); m.setup_rnn()
    return m

for key in ['ema_model', 'model']:
    try:
        m = load_vae(key)
    except Exception as e:
        print(f'[{key}] load failed: {e}')
        continue
    with torch.no_grad():
        data, masks = next(iter(valloader))
        data = data.to(device); masks = masks.to(device)
        B, L, C = data.shape

        recon = m.reconstruct(data, None, masks=masks, get_full_nll=False)
        gen = m.generate(B, L, device=device)
        z_post, z_mean, z_std = m.encoder.encode(data, None, use_forward=True)

        dec = m.decoder
        zs = [dec.z_prior[None].expand(B, -1)]
        hs = dec.latent.default_state(B, device=device)
        for t in range(L):
            z_t, hs = dec.latent.step(zs[-1], None, t=None, state=hs, sample=True)
            zs.append(z_t)
        z_prior_roll = torch.stack(zs[1:], dim=1)

    def stat(name, x):
        x = x.detach().float().cpu().numpy()
        print(f'  {name:28s} shape={str(x.shape):18s} mean={x.mean():+.4f} std={x.std():.4f} min={x.min():+.3f} max={x.max():+.3f}')

    print(f'--- {key} ---')
    stat('data (real)', data)
    stat('reconstruction', recon)
    stat('generation (prior sample)', gen)
    stat('z_post (encoder)', z_post)
    stat('z_prior_roll (prior sample)', z_prior_roll)
    zp = z_post.reshape(-1, z_post.shape[-1]).float().cpu().numpy()
    zr = z_prior_roll.reshape(-1, z_prior_roll.shape[-1]).float().cpu().numpy()
    print('  z_post  per-dim mean:', np.round(zp.mean(0), 3))
    print('  z_prior per-dim mean:', np.round(zr.mean(0), 3))
    print('  z_post  per-dim std :', np.round(zp.std(0), 3))
    print('  z_prior per-dim std :', np.round(zr.std(0), 3))
