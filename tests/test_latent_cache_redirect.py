"""``latent_cache_dir`` — target-latent-only cache redirect (colorize WB targets).

The subset knob redirects ONLY the target VAE-latent cache; TE stays on
``text_cache_dir``/``cache_dir`` and the REPA PE sidecars stay on ``cache_dir``.
This pins the read-only completeness probe in ``BaseDataset``, so the redirect
can't silently regress to the shared cache — which would re-train colorize on
the uncorrected (sepia) targets.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from library.anima.strategy import AnimaLatentsCachingStrategy
from library.anima.text_strategies import LatentsCachingStrategy
from library.datasets.base import BaseDataset


@pytest.fixture()
def anima_latents_strategy():
    prev = LatentsCachingStrategy._strategy
    LatentsCachingStrategy._strategy = AnimaLatentsCachingStrategy(
        cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False
    )
    try:
        yield LatentsCachingStrategy._strategy
    finally:
        LatentsCachingStrategy._strategy = prev


class _StubDataset:
    """Minimal stand-in borrowing the real completeness probe; it only reads
    ``image_data`` / ``image_to_subset``."""

    is_latents_cache_complete = BaseDataset.is_latents_cache_complete

    def __init__(self, infos: dict, image_to_subset: dict):
        self.image_data = infos
        self.image_to_subset = image_to_subset


def _write_latent_npz(path: Path, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = f"_{h // 8}x{w // 8}"
    np.savez(
        path,
        **{
            f"latents{key}": np.zeros((16, h // 8, w // 8), dtype=np.float32),
            f"original_size{key}": np.array([w, h]),
            f"crop_ltrb{key}": np.array([0, 0, w, h]),
        },
    )


def _layout(tmp_path: Path, *, latent_redirect: bool):
    image_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    latent_dir = tmp_path / "target" if latent_redirect else None
    image_dir.mkdir(parents=True, exist_ok=True)
    subset = SimpleNamespace(
        image_dir=str(image_dir),
        cache_dir=str(cache_dir),
        latent_cache_dir=str(latent_dir) if latent_dir else None,
        flip_aug=False,
        alpha_mask=False,
    )
    info = SimpleNamespace(
        image_key="img",
        absolute_path=str(image_dir / "img.png"),
        image_size=(32, 32),
        bucket_reso=(32, 32),
        latents_npz=None,
    )
    return subset, info, cache_dir, latent_dir


def test_probe_reads_redirected_latent_cache(tmp_path, anima_latents_strategy):
    subset, info, cache_dir, latent_dir = _layout(tmp_path, latent_redirect=True)
    ds = _StubDataset({"img": info}, {"img": subset})

    # latent only in the shared cache → NOT complete (redirect must win)
    _write_latent_npz(cache_dir / "img_0032x0032_anima.npz", 32, 32)
    assert not ds.is_latents_cache_complete()

    # latent in the redirected target cache → complete
    _write_latent_npz(latent_dir / "img_0032x0032_anima.npz", 32, 32)
    assert ds.is_latents_cache_complete()
