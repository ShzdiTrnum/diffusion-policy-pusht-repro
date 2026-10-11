import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dprepro.dataset import MinMaxNormalizer, create_sample_indices, sample_sequence  # noqa: E402
from dprepro.diffusion import NoiseScheduler  # noqa: E402
from dprepro.model import ConditionalUnet1D  # noqa: E402
from scripts.uncertainty_failure import auroc  # noqa: E402


def test_unet_shapes():
    net = ConditionalUnet1D(input_dim=2, global_cond_dim=10, diffusion_step_embed_dim=32, down_dims=(32, 64, 128))
    x = torch.randn(3, 16, 2)
    out = net(x, torch.tensor([0, 50, 99]), torch.randn(3, 10))
    assert out.shape == x.shape


def test_full_size_param_count():
    net = ConditionalUnet1D(input_dim=2, global_cond_dim=10)
    n = sum(p.numel() for p in net.parameters())
    assert 50e6 < n < 80e6, n  # reference CNN model is ~65M parameters for this config


def test_add_noise_and_ddim_recovers_x0_with_oracle():
    """If the network predicts the true noise, deterministic DDIM must return x0."""
    sch = NoiseScheduler(100)
    x0 = torch.rand(4, 16, 2) * 1.6 - 0.8

    class Oracle(torch.nn.Module):
        def forward(self, x, t, cond):
            ac = sch.alphas_cumprod[t].view(-1, 1, 1)
            return (x - ac.sqrt() * x0) / (1 - ac).sqrt()

    for k in (1, 10, 100):
        out = sch.sample(Oracle(), x0.shape, torch.zeros(4, 1), method="ddim", num_inference_steps=k)
        assert torch.allclose(out, x0, atol=1e-4), k
    out = sch.sample(Oracle(), x0.shape, torch.zeros(4, 1), method="ddpm")
    assert torch.allclose(out, x0, atol=1e-3)


def test_sample_indices_padding():
    ends = np.array([5, 9])
    idx = create_sample_indices(ends, sequence_length=4, pad_before=1, pad_after=2)
    data = {"a": np.arange(9, dtype=np.float32)[:, None]}
    for b0, b1, s0, s1 in idx:
        seq = sample_sequence(data, 4, b0, b1, s0, s1)["a"][:, 0]
        assert len(seq) == 4
        ep = 0 if b0 < 5 else 1
        lo, hi = (0, 4) if ep == 0 else (5, 8)
        assert seq.min() >= lo and seq.max() <= hi  # never crosses episode boundary


def test_normalizer_roundtrip():
    x = np.random.rand(100, 5).astype(np.float32) * 500
    n = MinMaxNormalizer.fit({"obs": x})
    y = n.normalize("obs", x)
    assert y.min() >= -1 - 1e-6 and y.max() <= 1 + 1e-6
    assert np.allclose(n.unnormalize("obs", y), x, atol=1e-3)


def test_auroc():
    assert auroc([3, 4], [False, True]) == 1.0
    assert auroc([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert auroc([1, 1], [True, False]) == 0.5


def test_official_split_masks():
    from dprepro.dataset import downsample_mask, get_val_mask
    val = get_val_mask(206, 0.02, seed=42)
    assert val.sum() == 4  # round(206 * 0.02)
    train = downsample_mask(~val, 90, seed=42)
    assert train.sum() == 90 and not (train & val).any()
    assert (get_val_mask(206, 0.02, seed=42) == val).all()  # deterministic
