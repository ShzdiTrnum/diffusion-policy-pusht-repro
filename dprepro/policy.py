"""Diffusion Policy wrapper: normalization, EMA, action-chunk prediction and checkpoint I/O."""
import copy

import numpy as np
import torch

from .dataset import MinMaxNormalizer
from .diffusion import NoiseScheduler
from .model import ConditionalUnet1D


class EMA:
    """EMA with warmup as in the reference code (inv_gamma=1, power=0.75, max_value=0.9999)."""

    def __init__(self, model, power=0.75, inv_gamma=1.0, min_value=0.0, max_value=0.9999):
        self.averaged_model = copy.deepcopy(model).eval().requires_grad_(False)
        self.power, self.inv_gamma = power, inv_gamma
        self.min_value, self.max_value = min_value, max_value
        self.optimization_step = 0

    def decay(self):
        step = max(0, self.optimization_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return 0.0 if step <= 0 else max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, model):
        self.optimization_step += 1
        d = self.decay()
        for p_ema, p in zip(self.averaged_model.parameters(), model.parameters()):
            p_ema.mul_(d).add_(p.detach(), alpha=1 - d)
        for b_ema, b in zip(self.averaged_model.buffers(), model.buffers()):
            b_ema.copy_(b)


class DiffusionPolicy:
    def __init__(self, cfg: dict, normalizer: MinMaxNormalizer, device="cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.normalizer = normalizer
        self.net = ConditionalUnet1D(
            input_dim=cfg["action_dim"],
            global_cond_dim=cfg["obs_dim"] * cfg["obs_horizon"],
            diffusion_step_embed_dim=cfg.get("diffusion_step_embed_dim", 256),
            down_dims=tuple(cfg.get("down_dims", (256, 512, 1024))),
            kernel_size=cfg.get("kernel_size", 5),
        ).to(self.device)
        self.scheduler = NoiseScheduler(cfg.get("num_diffusion_iters", 100))

    # ------------------------------------------------------------------ training
    def compute_loss(self, batch, net=None):
        net = net or self.net
        obs = batch["obs"].to(self.device)
        action = batch["action"].to(self.device)
        B = obs.shape[0]
        cond = obs.flatten(start_dim=1)
        noise = torch.randn_like(action)
        t = torch.randint(0, self.scheduler.T, (B,), device=self.device).long()
        noisy = self.scheduler.add_noise(action, noise, t)
        pred = net(noisy, t, cond)
        return torch.nn.functional.mse_loss(pred, noise)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_action_samples(self, obs_seq: np.ndarray, num_samples=1, method="ddpm",
                               num_inference_steps=None, net=None, generator=None) -> np.ndarray:
        """obs_seq: (obs_horizon, obs_dim) raw observations. Returns (num_samples, pred_horizon, action_dim)
        un-normalized actions. Each sample uses independent initial noise."""
        net = (net or self.net).eval()
        nobs = self.normalizer.normalize("obs", obs_seq.astype(np.float32))
        cond = torch.from_numpy(nobs).to(self.device).flatten()[None].repeat(num_samples, 1)
        shape = (num_samples, self.cfg["pred_horizon"], self.cfg["action_dim"])
        naction = self.scheduler.sample(net, shape, cond, method=method,
                                        num_inference_steps=num_inference_steps, generator=generator)
        return self.normalizer.unnormalize("action", naction.cpu().numpy())

    # ------------------------------------------------------------------ I/O
    def save(self, path, ema_net=None, extra=None, include_raw=True):
        """include_raw=False stores only the EMA weights (used for evaluation), halving file size."""
        torch.save({"cfg": self.cfg, "normalizer": self.normalizer.state_dict(),
                    "net": self.net.state_dict() if (include_raw or ema_net is None) else None,
                    "ema_net": ema_net.state_dict() if ema_net is not None else None,
                    "extra": extra or {}}, path)

    @classmethod
    def load(cls, path, device="cpu", use_ema=True):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        pol = cls(ckpt["cfg"], MinMaxNormalizer(ckpt["normalizer"]), device=device)
        state = ckpt["ema_net"] if (use_ema and ckpt["ema_net"] is not None) else ckpt["net"]
        pol.net.load_state_dict(state)
        pol.net.eval()
        return pol, ckpt.get("extra", {})
