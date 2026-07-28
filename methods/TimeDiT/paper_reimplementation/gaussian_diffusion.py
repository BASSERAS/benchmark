"""Gaussian diffusion (DDPM) with optional learned variance.

Compact, self-contained reimplementation of the diffusion process DiT/TimeDiT use,
following Ho et al. (2020, DDPM) for the forward/reverse process and Nichol &
Dhariwal (2021, improved DDPM) for the learned-variance (hybrid) objective that the
DiT codebase and TimeDiT's full loss "L" employ.

Loss modes:
  - "simple"  : L_simple = E_t || eps - eps_theta ||^2   (fixed variance sampling)
                Used by TimeDiT for imputation / anomaly / forecasting (Table 9, L_simple).
  - "hybrid"  : L = L_simple + lambda * L_vlb  (learned variance).
                Used by TimeDiT for the synthetic-generation task (Table 9, "L").

The model predicts eps (and, if learn_sigma, an interpolation coefficient v for the
per-step variance).  Shapes are (B, L, C).
"""
import numpy as np
import torch
import torch.nn.functional as F


def make_beta_schedule(T, kind="linear"):
    if kind == "linear":
        # DDPM linear schedule (scaled for arbitrary T as in Ho et al.)
        scale = 1000.0 / T
        return np.linspace(scale * 1e-4, scale * 2e-2, T, dtype=np.float64)
    if kind == "cosine":
        steps = T + 1
        s = 0.008
        x = np.linspace(0, T, steps, dtype=np.float64)
        ac = np.cos(((x / T) + s) / (1 + s) * np.pi / 2) ** 2
        ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1])
        return np.clip(betas, 0, 0.999)
    raise ValueError(kind)


def _extract(arr, t, shape):
    out = torch.as_tensor(arr, device=t.device, dtype=torch.float32)[t]
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0 + logvar2 - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_gaussian_ll(x, means, log_scales):
    """Approx Gaussian log-likelihood for continuous data (nats)."""
    centered = x - means
    inv_std = torch.exp(-log_scales)
    return -0.5 * (np.log(2 * np.pi) + 2 * log_scales + (centered * inv_std) ** 2)


class GaussianDiffusion:
    def __init__(self, T=1000, schedule="linear", loss_mode="hybrid"):
        self.T = T
        self.loss_mode = loss_mode
        betas = make_beta_schedule(T, schedule)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_acp = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_acp = np.sqrt(1.0 - self.alphas_cumprod)
        # posterior q(x_{t-1}|x_t,x_0)
        self.posterior_var = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_var_clipped = np.log(np.append(self.posterior_var[1], self.posterior_var[1:]))
        self.posterior_mean_c1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_c2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    # ---- forward process ----
    def q_sample(self, x0, t, noise):
        return _extract(self.sqrt_acp, t, x0.shape) * x0 + \
            _extract(self.sqrt_one_minus_acp, t, x0.shape) * noise

    def q_posterior_mean_logvar(self, x0, xt, t):
        mean = _extract(self.posterior_mean_c1, t, xt.shape) * x0 + \
            _extract(self.posterior_mean_c2, t, xt.shape) * xt
        logvar = _extract(self.posterior_log_var_clipped, t, xt.shape)
        return mean, logvar

    def predict_x0_from_eps(self, xt, t, eps):
        return (xt - _extract(self.sqrt_one_minus_acp, t, xt.shape) * eps) / \
            _extract(self.sqrt_acp, t, xt.shape)

    def _split_model_out(self, model_out, learn_sigma, C):
        if learn_sigma:
            eps, v = model_out[..., :C], model_out[..., C:]
            return eps, v
        return model_out, None

    def p_mean_logvar(self, model, xt, t, obs, mask, learn_sigma, sample_var="learned"):
        C = xt.shape[-1]
        out = model(xt, t, obs, mask)
        eps, v = self._split_model_out(out, learn_sigma, C)
        if learn_sigma and sample_var == "learned":
            min_log = _extract(self.posterior_log_var_clipped, t, xt.shape)
            max_log = _extract(np.log(self.betas), t, xt.shape)
            frac = (v + 1) / 2  # v in [-1,1] -> [0,1]
            model_log_var = frac * max_log + (1 - frac) * min_log
        else:
            # fixed variance = posterior variance (DDPM Algorithm 1 sigma_t)
            model_log_var = _extract(self.posterior_log_var_clipped, t, xt.shape)
        x0 = self.predict_x0_from_eps(xt, t, eps).clamp(-3, 3)
        mean, _ = self.q_posterior_mean_logvar(x0, xt, t)
        return mean, model_log_var, x0

    # ---- training loss ----
    def training_loss(self, model, x0, obs=None, mask=None, learn_sigma=True):
        B, L, C = x0.shape
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        out = model(xt, t, obs, mask)
        eps, v = self._split_model_out(out, learn_sigma, C)
        loss_simple = F.mse_loss(eps, noise)
        if not learn_sigma:
            return loss_simple, {"simple": loss_simple.item(), "vlb": 0.0}
        # vlb term: freeze eps (use its detached value), learn only the variance
        frozen = torch.cat([eps.detach(), v], dim=-1)
        vlb = self._vlb_term(frozen, x0, xt, t).mean()
        loss = loss_simple + (self.T / 1000.0) * vlb  # lambda scaling (Nichol & Dhariwal)
        return loss, {"simple": loss_simple.item(), "vlb": vlb.item()}

    def masked_training_loss(self, model, x0, cond_mask, valid, learn_sigma=True):
        """Time-Series-Mask-Unit training loss (TimeDiT pretraining).

        x0        : (B, L, C) padded to K_MAX; padding channels are 0.
        cond_mask : (B, L, C) 1 = observed (fed clean as condition), 0 = target.
        valid     : (B, 1, C) 1 = real channel, 0 = padding channel.
        Loss is computed only over the target region intersected with valid channels.
        M^Rec reduces to cond_mask == 0 everywhere (unconditional generation).
        """
        B, L, C = x0.shape
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        # composite input: observed region clean, target region noised, padding zeroed
        x_input = valid * (cond_mask * x0 + (1.0 - cond_mask) * xt)
        obs = valid * cond_mask * x0
        out = model(x_input, t, obs, cond_mask)
        eps, v = self._split_model_out(out, learn_sigma, C)

        tgt = (1.0 - cond_mask) * valid                       # (B, L, C) loss support
        denom = tgt.sum().clamp(min=1.0)
        loss_simple = (((eps - noise) ** 2) * tgt).sum() / denom
        if not learn_sigma:
            return loss_simple, {"simple": loss_simple.item(), "vlb": 0.0}
        frozen = torch.cat([eps.detach(), v], dim=-1)
        vlb_el = self._vlb_elementwise(frozen, x0, xt, t)     # (B, L, C)
        vlb = (vlb_el * tgt).sum() / denom
        loss = loss_simple + (self.T / 1000.0) * vlb
        return loss, {"simple": loss_simple.item(), "vlb": vlb.item()}

    def _vlb_elementwise(self, model_out, x0, xt, t):
        """Per-element VLB (nats->bits), same math as _vlb_term without reduction."""
        C = x0.shape[-1]
        eps, v = model_out[..., :C], model_out[..., C:]
        min_log = _extract(self.posterior_log_var_clipped, t, xt.shape)
        max_log = _extract(np.log(self.betas), t, xt.shape)
        frac = (v + 1) / 2
        model_log_var = frac * max_log + (1 - frac) * min_log
        x0_pred = self.predict_x0_from_eps(xt, t, eps).clamp(-3, 3)
        model_mean, _ = self.q_posterior_mean_logvar(x0_pred, xt, t)
        true_mean, true_log_var = self.q_posterior_mean_logvar(x0, xt, t)
        kl = normal_kl(true_mean, true_log_var, model_mean, model_log_var) / np.log(2.0)
        nll = -approx_gaussian_ll(x0, model_mean, 0.5 * model_log_var) / np.log(2.0)
        t_b = t.reshape(-1, *([1] * (x0.dim() - 1)))
        return torch.where(t_b == 0, nll, kl)

    def _vlb_term(self, model_out, x0, xt, t):
        C = x0.shape[-1]
        eps, v = model_out[..., :C], model_out[..., C:]
        min_log = _extract(self.posterior_log_var_clipped, t, xt.shape)
        max_log = _extract(np.log(self.betas), t, xt.shape)
        frac = (v + 1) / 2
        model_log_var = frac * max_log + (1 - frac) * min_log
        x0_pred = self.predict_x0_from_eps(xt, t, eps).clamp(-3, 3)
        model_mean, _ = self.q_posterior_mean_logvar(x0_pred, xt, t)
        true_mean, true_log_var = self.q_posterior_mean_logvar(x0, xt, t)
        kl = normal_kl(true_mean, true_log_var, model_mean, model_log_var).mean(dim=[1, 2]) / np.log(2.0)
        nll = -approx_gaussian_ll(x0, model_mean, 0.5 * model_log_var).mean(dim=[1, 2]) / np.log(2.0)
        return torch.where(t == 0, nll, kl)

    # ---- sampling ----
    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, obs=None, mask=None, learn_sigma=True,
                      sample_var="learned"):
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            mean, log_var, _ = self.p_mean_logvar(model, x, t, obs, mask, learn_sigma,
                                                  sample_var=sample_var)
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = mean + torch.exp(0.5 * log_var) * noise
        return x

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, device, obs=None, mask=None, learn_sigma=True,
                         steps=None, eta=0.0):
        """Deterministic (eta=0) DDIM sampler. Reuses the same eps-prediction model;
        eta controls injected noise (0 = fully deterministic, smoother samples)."""
        steps = steps or self.T
        # evenly spaced subsequence of timesteps
        ts = np.linspace(0, self.T - 1, steps, dtype=int)[::-1]
        C = shape[-1]
        x = torch.randn(shape, device=device)
        for k, i in enumerate(ts):
            t = torch.full((shape[0],), int(i), device=device, dtype=torch.long)
            out = model(x, t, obs, mask)
            eps, _ = self._split_model_out(out, learn_sigma, C)
            acp_t = _extract(self.alphas_cumprod, t, x.shape)
            x0 = ((x - torch.sqrt(1 - acp_t) * eps) / torch.sqrt(acp_t)).clamp(-3, 3)
            if k == len(ts) - 1:
                x = x0
                break
            i_prev = int(ts[k + 1])
            t_prev = torch.full((shape[0],), i_prev, device=device, dtype=torch.long)
            acp_prev = _extract(self.alphas_cumprod, t_prev, x.shape)
            sigma = eta * torch.sqrt((1 - acp_prev) / (1 - acp_t)) * torch.sqrt(1 - acp_t / acp_prev)
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = torch.sqrt(acp_prev) * x0 + torch.sqrt(1 - acp_prev - sigma ** 2) * eps + sigma * noise
        return x
