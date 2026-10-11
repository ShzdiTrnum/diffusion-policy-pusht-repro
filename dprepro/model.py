"""1D temporal U-Net noise-prediction network (CNN-based Diffusion Policy).

Re-implementation of the ConditionalUnet1D architecture described in
Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023):
FiLM conditioning on the observation features + diffusion-step embedding,
Conv1d residual blocks with GroupNorm and Mish.
"""
import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=x.device) * -scale)
        emb = x.float()[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block whose features are modulated by FiLM (per-channel scale and bias)."""

    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=5, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_ch, out_ch, kernel_size, n_groups),
            Conv1dBlock(out_ch, out_ch, kernel_size, n_groups),
        ])
        self.out_ch = out_ch
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_ch * 2))
        self.residual_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        film = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_ch, 1)
        scale, bias = film[:, 0], film[:, 1]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim: int, global_cond_dim: int,
                 diffusion_step_embed_dim: int = 256,
                 down_dims=(256, 512, 1024), kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid, mid, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid, mid, cond_dim, kernel_size, n_groups),
        ])
        self.down_modules = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(d_in, d_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(d_out, d_out, cond_dim, kernel_size, n_groups),
                nn.Conv1d(d_out, d_out, 3, 2, 1) if not is_last else nn.Identity(),
            ]))
        self.up_modules = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(reversed(in_out[1:])):
            is_last = i == len(in_out) - 1
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(d_out * 2, d_in, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(d_in, d_in, cond_dim, kernel_size, n_groups),
                nn.ConvTranspose1d(d_in, d_in, 4, 2, 1) if not is_last else nn.Identity(),
            ]))
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor):
        """sample: (B, T, action_dim); timestep: (B,) or scalar; global_cond: (B, cond_dim)."""
        x = sample.moveaxis(-1, -2)  # (B, C, T)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])
        cond = torch.cat([self.diffusion_step_encoder(timestep), global_cond], dim=-1)

        skips = []
        for res1, res2, down in self.down_modules:
            x = res2(res1(x, cond), cond)
            skips.append(x)
            x = down(x)
        for m in self.mid_modules:
            x = m(x, cond)
        for res1, res2, up in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = res2(res1(x, cond), cond)
            x = up(x)
        x = self.final_conv(x)
        return x.moveaxis(-2, -1)
