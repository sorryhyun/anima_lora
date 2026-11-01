"""Invariant tests for the 2D-folded Qwen-Image VAE (``--qwen_image_vae_2d``).

The fold rewrites every causal ``QwenImageCausalConv3d`` into a plain 2D
``Conv2d`` using the last temporal slice of the kernel — exact for single-image
(``T=1``) input because the causal temporal padding zero-fills with the real
frame in the last slot. See ``_archive/bench/qwen_vae_2d/`` and
``AutoencoderKLQwenImage.convert_to_2d``.

Contracts pinned here:
  1. per-layer fold is bit-exact in fp64,
  2. (gated on the VAE weights being present) the whole VAE's bf16 latents
     match the 3D path within bf16 quantization noise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from library.models.qwen_vae import (
    Folded2DConv,
    QwenImageCausalConv3d,
    load_vae,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
VAE_PATH = REPO_ROOT / "models" / "vae" / "qwen_image_vae.safetensors"


# kernel, stride, padding combos that actually occur in the VAE (temporal
# kernel 3 always ships with temporal pad 1; 1x1x1 and spatial-only also occur)
_LAYER_CONFIGS = [
    (3, 3, 3, 1, 1),  # main feature conv
    (3, 1, 1, 1, 1),  # temporal-only (pad 1 keeps T=1 valid)
    (1, 1, 1, 0, 1),  # pointwise
    (3, 3, 3, 1, (2, 1, 1)),  # temporal-downsample stride
]


@pytest.mark.parametrize("kt,kh,kw,pad,stride", _LAYER_CONFIGS)
def test_folded_conv_bit_exact_fp64(kt, kh, kw, pad, stride):
    """Per-layer: folded 2D == 3D causal conv for T=1 (bit-exact in fp64)."""
    torch.manual_seed(0)
    c3d = QwenImageCausalConv3d(
        8, 16, (kt, kh, kw), stride=stride, padding=pad
    ).double()
    x = torch.randn(2, 8, 1, 17, 17, dtype=torch.float64)
    y3d = c3d(x)
    y2d = Folded2DConv(c3d).double()(x)
    assert y3d.shape == y2d.shape
    assert torch.equal(y3d, y2d), (y3d - y2d).abs().max().item()


@pytest.mark.core
@pytest.mark.skipif(
    not VAE_PATH.exists(), reason="VAE weights not present (skip full-model check)"
)
def test_full_vae_2d_matches_3d_latents():
    """Whole-VAE encode: 2D fold matches 3D within bf16 noise (CPU, fp32)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "" or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda:0"

    torch.manual_seed(0)
    px = torch.rand(1, 3, 128, 128, device=device) * 2 - 1

    vae = load_vae(
        str(VAE_PATH),
        device=device,
        dtype=torch.float32,
        eval=True,
        disable_cache=True,
    )
    with torch.no_grad():
        lat3d = vae.encode_pixels_to_latents(px.float())
        n = vae.convert_to_2d()
        lat2d = vae.encode_pixels_to_latents(px.float())

    assert n == 61
    assert vae.is_2d is True
    # fp32 over 61 layers + differing cuDNN/oneDNN algos: small rounding only.
    max_abs = (lat3d - lat2d).abs().max().item()
    assert max_abs < 5e-3, f"latent max|Δ| = {max_abs:.3e}"


def test_full_vae_2d_matches_3d_decode():
    """Whole-VAE decode (the inference path): 2D fold matches 3D within bf16 noise."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "" or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda:0"

    torch.manual_seed(0)
    # z_dim latent channels at 1/8 spatial scale of a 128px image.
    lat = torch.randn(1, 16, 16, 16, device=device)

    vae = load_vae(
        str(VAE_PATH),
        device=device,
        dtype=torch.float32,
        eval=True,
        disable_cache=True,
    )
    with torch.no_grad():
        px3d = vae.decode_to_pixels(lat.float())
        vae.convert_to_2d()
        px2d = vae.decode_to_pixels(lat.float())

    assert vae.is_2d is True
    max_abs = (px3d - px2d).abs().max().item()
    assert max_abs < 5e-3, f"pixel max|Δ| = {max_abs:.3e}"
