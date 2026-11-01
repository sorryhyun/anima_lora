"""MOSS DAVE invariants on a tiny random DiT (CPU): the hook seam, the edit, the window."""

import numpy as np
import pytest
import torch

from library.moss.dave import MossDAVE
from library.moss.generate import denoise
from library.moss.model import MOSSImageTransformer

H, W = 4, 6


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return MOSSImageTransformer(
        dim=96, ffn_dim=128, num_heads=4, num_layers=4, text_dim=32, latent_channels=8
    ).eval()


def _inputs(batch=2, s=7):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(batch, 8, H, W, generator=g)
    e = torch.randn(batch, s, 32, generator=g)
    mask = torch.ones(batch, s, dtype=torch.bool)
    mask[-1, 3:] = False
    return x, e, mask


def _dave(weight, strength=0.5, tau=1.0, exclude=()):
    return MossDAVE(np.asarray(weight, dtype=np.float64), strength, tau, exclude)


def test_edit_touches_image_tokens_only():
    x = torch.randn(3, H * W + 5, 16)
    out = _dave([1.0], strength=0.4)(0, x, H * W)
    assert torch.equal(out[:, H * W :], x[:, H * W :])
    mu, mu_out = x[:, : H * W].mean(1), out[:, : H * W].mean(1)
    assert torch.allclose(mu_out, 0.6 * mu, atol=1e-6)
    # AC residual unchanged.
    assert torch.allclose(
        out[:, : H * W] - mu_out[:, None], x[:, : H * W] - mu[:, None], atol=1e-6
    )


def test_excluded_channels_keep_their_mean():
    x = torch.randn(2, H * W + 5, 16) + 50.0 * (torch.arange(16) == 3)
    out = _dave([1.0], strength=0.4, exclude=[3, 7])(0, x, H * W)
    assert torch.equal(out[..., [3, 7]], x[..., [3, 7]])
    keep = [c for c in range(16) if c not in (3, 7)]
    mu = x[:, : H * W, keep].mean(1)
    assert torch.allclose(out[:, : H * W, keep].mean(1), 0.6 * mu, atol=1e-5)


@torch.no_grad()
def test_batched_matches_reference_with_dave(tiny):
    x, e, mask = _inputs()
    tiny.layer_hook = _dave([0, 1, 1, 0])
    try:
        ref = tiny(x, 500.0, e, mask).sample
        out = tiny.forward_batched(x, 500.0, e, mask)
    finally:
        tiny.layer_hook = None
    assert torch.allclose(out, ref, atol=1e-5)
    assert not torch.allclose(out, tiny.forward_batched(x, 500.0, e, mask), atol=1e-3)


@torch.no_grad()
def test_hook_sees_only_rows_that_ran_the_block(tiny):
    x, e, mask = _inputs()
    seen = {}

    def record(index, h, n_img):
        seen[index] = (h.shape[0], n_img)
        return h

    tiny.layer_hook = record
    try:
        tiny.forward_batched(x, 500.0, e, mask, skip_layers=[(), (2,)])
    finally:
        tiny.layer_hook = None
    assert seen == {0: (2, H * W), 1: (2, H * W), 2: (1, H * W), 3: (2, H * W)}


def _denoise(tiny, dave):
    x, e, mask = _inputs(batch=1)
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25])
    return denoise(
        tiny, x, e, mask, e.flip(1), mask, sigmas, cfg=3.0, uncond_skip=(2,), dave=dave
    )


@torch.no_grad()
def test_denoise_window_and_cleanup(tiny):
    ref = _denoise(tiny, None)
    assert torch.equal(_denoise(tiny, _dave([0, 0, 0, 0])), ref)
    one = _dave([0, 1, 1, 0], tau=0.25)
    out = _denoise(tiny, one)
    assert one.active_steps == 1
    assert not torch.allclose(out, ref, atol=1e-4)
    assert tiny.layer_hook is None
