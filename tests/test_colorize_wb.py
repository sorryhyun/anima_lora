"""Colorize target white-balance (i2i_tasks/colorization/wb.py).

The colorize target pool is corpus-wide warm/desaturated, so the prep step
white-balances each target before VAE-encoding it into the colorize-specific
target-latent cache. These tests pin the correction math (warm cast
neutralized, gain clamp) and the ``cache_latents``
hooks the stage rides on (``keep_stems`` scoping + ``image_transform``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

COLORIZE_DIR = Path(__file__).resolve().parent.parent / ("i2i_tasks/colorization")
if str(COLORIZE_DIR) not in sys.path:
    sys.path.insert(0, str(COLORIZE_DIR))

from wb import (  # noqa: E402
    apply_whitebalance,
    whitepoint_gains,
)


def _warm_page(cast_r: float = 1.12, cast_b: float = 0.94) -> np.ndarray:
    """Synthetic manga-ish page: warm paper background + darker line art."""
    rng = np.random.default_rng(0)
    img = np.full((128, 128, 3), 235.0, dtype=np.float32)
    # darker "line art"/screentone region (bottom half)
    img[64:] = 110.0
    img += rng.normal(0.0, 3.0, img.shape)
    img[..., 0] *= cast_r  # warm cast: R up, B down
    img[..., 2] *= cast_b
    return np.clip(img, 0, 255).astype(np.uint8)


def test_warm_cast_neutralized():
    img = _warm_page()
    out = apply_whitebalance(img)
    luma = out.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722])
    bright = out[luma >= np.quantile(luma, 0.92)].astype(np.float32)
    means = bright.mean(axis=0)
    # bright-region R−B collapses from ~+40 (cast) to near zero
    assert abs(means[0] - means[2]) < 4.0
    assert abs(means[0] - means[1]) < 4.0


def test_gain_clamp_bounds_extreme_cast():
    img = _warm_page(cast_r=1.9, cast_b=0.5)  # heavily stylized palette
    gains = whitepoint_gains(img, max_gain=1.35)
    assert np.all(gains <= 1.35 + 1e-6)
    assert np.all(gains >= 1.0 / 1.35 - 1e-6)


def test_dark_image_grayworld_fallback_is_finite():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 60, (64, 64, 3), dtype=np.uint8)  # no bright region
    gains = whitepoint_gains(img)
    assert np.all(np.isfinite(gains))
    out = apply_whitebalance(img)
    assert out.dtype == np.uint8 and out.shape == img.shape


class _FakeVae:
    """Stands in for the Qwen VAE: 8x avg-pool 'encoder' with the real
    device/dtype surface ``cache_latents`` reads."""

    device = torch.device("cpu")
    dtype = torch.float32

    def encode_pixels_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(x, 8)


def _write_png(path: Path, value: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def test_cache_latents_keep_stems_and_transform(tmp_path):
    from library.preprocess.latents import cache_latents

    src = tmp_path / "src"
    cache = tmp_path / "cache"
    _write_png(src / "keep.png", 100)
    _write_png(src / "drop.png", 100)

    stats = cache_latents(
        src,
        _FakeVae(),
        cache_dir=cache,
        keep_stems=frozenset({"keep"}),
        image_transform=lambda img: np.zeros_like(img),
        batch_size=2,
    )
    assert stats.written == 1
    files = sorted(p.name for p in cache.glob("*.npz"))
    assert files == ["keep_0032x0032_anima.npz"]
    # the transform (zeros) reached the encoder: latent is the transform of
    # black in [-1, 1] space, i.e. exactly -1 everywhere
    lat = np.load(cache / "keep_0032x0032_anima.npz")["latents_4x4"]
    assert np.allclose(lat, -1.0)

    # idempotence: second run skips the cached resolution
    stats2 = cache_latents(
        src,
        _FakeVae(),
        cache_dir=cache,
        keep_stems=frozenset({"keep"}),
        batch_size=2,
    )
    assert stats2.written == 0 and stats2.skipped == 1
