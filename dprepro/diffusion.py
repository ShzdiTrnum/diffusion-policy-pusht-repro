"""Minimal DDPM training / DDPM + DDIM sampling, matching the settings used for Diffusion Policy:
squared-cosine beta schedule ("squaredcos_cap_v2"), epsilon prediction, 100 training steps,
sample clipping to [-1, 1].

Written from the equations in Ho et al. 2020 (DDPM) and Song et al. 2021 (DDIM) so the repo
has no dependency on a particular `diffusers` version.
"""
import math

import torch


def squaredcos_cap_v2_betas(num_steps: int, max_beta: float = 0.999) -> torch.Tensor:
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = [min(1 - alpha_bar((i + 1) / num_steps) / alpha_bar(i / num_steps), max_beta)
             for i in range(num_steps)]
    return torch.tensor(betas, dtype=torch.float32)


class NoiseScheduler:
    def __init__(self, num_train_steps: int = 100, clip_sample: bool = True):
        self.T = num_train_steps
        self.clip_sample = clip_sample
        self.betas = squaredcos_cap_v2_betas(num_train_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    # ---------------- training ----------------
    def add_noise(self, x0, noise, t):
        ac = self.alphas_cumprod.to(x0.device)[t].view(-1, *([1] * (x0.ndim - 1)))
        return ac.sqrt() * x0 + (1 - ac).sqrt() * noise

    # ---------------- sampling ----------------
    def _x0_from_eps(self, x_t, eps, ac_t):
        x0 = (x_t - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()
        return x0.clamp(-1, 1) if self.clip_sample else x0

    @torch.no_grad()
    def sample(self, model, shape, global_cond, method: str = "ddpm", num_inference_steps: int = None,
               generator: torch.Generator = None):
        """Generate action sequences of `shape` = (B, T, action_dim) conditioned on global_cond."""
        device = global_cond.device
        x = torch.randn(shape, device=device, generator=generator)
        ac_all = self.alphas_cumprod.to(device)
        if method == "ddpm":
            # Full ancestral sampling over all training steps (posterior variance "fixed_small").
            for t in reversed(range(self.T)):
                eps = model(x, torch.full((shape[0],), t, device=device, dtype=torch.long), global_cond)
                ac_t = ac_all[t]
                ac_prev = ac_all[t - 1] if t > 0 else torch.tensor(1.0, device=device)
                beta_t = 1 - ac_t / ac_prev
                x0 = self._x0_from_eps(x, eps, ac_t)
                mean = (ac_prev.sqrt() * beta_t / (1 - ac_t)) * x0 + \
                       ((1 - beta_t).sqrt() * (1 - ac_prev) / (1 - ac_t)) * x
                if t > 0:
                    var = beta_t * (1 - ac_prev) / (1 - ac_t)
                    noise = torch.randn(shape, device=device, generator=generator)
                    x = mean + var.clamp(min=1e-20).sqrt() * noise
                else:
                    x = mean
            return x
        if method == "ddim":
            k = num_inference_steps or 10
            # "leading" timestep spacing, as in diffusers' DDIMScheduler default
            step = self.T // k
            timesteps = list(range(0, k * step, step))[::-1]
            for i, t in enumerate(timesteps):
                eps = model(x, torch.full((shape[0],), t, device=device, dtype=torch.long), global_cond)
                ac_t = ac_all[t]
                t_prev = t - step
                ac_prev = ac_all[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
                x0 = self._x0_from_eps(x, eps, ac_t)
                if self.clip_sample:  # recompute eps consistent with the clipped x0
                    eps = (x - ac_t.sqrt() * x0) / (1 - ac_t).sqrt()
                x = ac_prev.sqrt() * x0 + (1 - ac_prev).sqrt() * eps  # eta = 0 (deterministic)
            return x
        raise ValueError(f"unknown method {method}")
