"""Utilities for loading cached latents and text encoder outputs from preprocessed datasets.

Provides shared helpers for discovering and loading the disk-cached
``*_anima.npz`` (VAE latents) and ``*_anima_te.safetensors`` (text encoder
outputs) files produced by the preprocessing pipeline.
"""

import glob
import logging
import os
import random
from typing import NamedTuple

import numpy as np
import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

LATENT_CACHE_SUFFIX = "_anima.npz"
TE_CACHE_SUFFIX = "_anima_te.safetensors"


def resolve_cache_path(
    image_abs_path: str | os.PathLike,
    suffix: str,
    cache_dir: str | os.PathLike | None = None,
) -> str:
    """Build a cache file path from a source image path + suffix.

    Sidecar default (``cache_dir=None``) preserves the legacy behavior of
    writing the cache next to the image. With ``cache_dir`` set, the cache
    is redirected into that directory using a stem-mirrored filename — the
    pattern IP-Adapter / EasyControl use to keep ``ip-adapter-dataset/`` and
    ``easycontrol-dataset/`` purely user-facing source dirs while caches
    live under ``post_image_dataset/``.
    """
    src = str(image_abs_path)
    stem = os.path.splitext(os.path.basename(src))[0]
    if cache_dir is None:
        return os.path.splitext(src)[0] + suffix
    cache_dir_str = str(cache_dir)
    os.makedirs(cache_dir_str, exist_ok=True)
    return os.path.join(cache_dir_str, stem + suffix)


class CachedImage(NamedTuple):
    """A preprocessed image with its cached latent and optional text encoder output."""

    stem: str
    image_path: str | None
    npz_path: str
    te_path: str | None


def discover_cached_images(data_dir: str) -> list[CachedImage]:
    """Find all images in a preprocessed dataset directory that have cached latents.

    Returns a sorted list of :class:`CachedImage` tuples.
    """
    images = []
    for png_path in sorted(glob.glob(os.path.join(data_dir, "*.png"))):
        stem = os.path.splitext(png_path)[0]
        npz_files = glob.glob(f"{stem}_*{LATENT_CACHE_SUFFIX}")
        if not npz_files:
            continue
        te_path = f"{stem}{TE_CACHE_SUFFIX}"
        if not os.path.exists(te_path):
            te_path = None
        images.append(
            CachedImage(
                stem=os.path.basename(stem),
                image_path=png_path,
                npz_path=npz_files[0],
                te_path=te_path,
            )
        )
    return images


def discover_cached_pairs(cache_dir: str) -> list[CachedImage]:
    """Find latent+TE cache pairs in a directory without requiring source PNGs.

    Use this when the cache directory is decoupled from the source images
    (e.g. ``post_image_dataset/lora/`` holds caches written from
    ``post_image_dataset/resized/`` via the subset-level ``cache_dir`` knob).
    """
    images = []
    te_paths = sorted(glob.glob(os.path.join(cache_dir, f"*{TE_CACHE_SUFFIX}")))
    for te_path in te_paths:
        stem = os.path.basename(te_path).removesuffix(TE_CACHE_SUFFIX)
        npz_files = glob.glob(os.path.join(cache_dir, f"{stem}_*{LATENT_CACHE_SUFFIX}"))
        if not npz_files:
            continue
        images.append(
            CachedImage(
                stem=stem,
                image_path=None,
                npz_path=npz_files[0],
                te_path=te_path,
            )
        )
    return images


def get_latent_resolution(npz_path: str) -> str:
    """Extract the resolution string (e.g. ``"64x64"``) from a cached latent NPZ."""
    npz_keys = np.load(npz_path).files
    latent_key = next(k for k in npz_keys if k.startswith("latents_"))
    return latent_key.split("_", 1)[1]


def load_cached_latents(npz_path: str) -> tuple[torch.Tensor, str, int, int]:
    """Load cached latents from a preprocessed NPZ file.

    Returns:
        latents: ``(C, H, W)`` float32 tensor (no batch dim).
        resolution: Latent resolution string, e.g. ``"64x64"``.
        orig_h, orig_w: Original pixel dimensions.
    """
    data = np.load(npz_path)
    latent_key = next(k for k in data.keys() if k.startswith("latents_"))
    latents = torch.from_numpy(data[latent_key].copy()).float()

    resolution = latent_key.split("_", 1)[1]
    size_key = f"original_size_{resolution}"
    if size_key in data:
        orig_w, orig_h = int(data[size_key][0]), int(data[size_key][1])
    else:
        orig_h = latents.shape[-2] * 8
        orig_w = latents.shape[-1] * 8

    return latents, resolution, orig_h, orig_w


def load_cached_crossattn_emb(
    te_path: str, *, variant: int | str = 0
) -> torch.Tensor | None:
    """Load ``crossattn_emb`` from a cached TE safetensors file.

    Args:
        te_path: Path to the ``*_anima_te.safetensors`` file.
        variant: Variant index (``int``), or ``"random"`` for random selection.
                 Falls back to ``crossattn_emb`` if no variants exist.

    Returns:
        ``(S, D)`` float32 tensor, or ``None`` if not found.
    """
    sd = load_file(te_path)

    if "num_variants" in sd:
        n = int(sd["num_variants"])
        vi = random.randint(0, n - 1) if variant == "random" else min(int(variant), n - 1)
        key = f"crossattn_emb_v{vi}"
        if key in sd:
            return sd[key].float()

    if "crossattn_emb_v0" in sd:
        return sd["crossattn_emb_v0"].float()
    if "crossattn_emb" in sd:
        return sd["crossattn_emb"].float()

    return None


def stem_from_cache_path(path: str | os.PathLike) -> str | None:
    """Extract the image stem from a cache file path.

    Handles both latent NPZ (``{stem}_{WxH}_anima.npz``) and
    TE safetensors (``{stem}_anima_te.safetensors``) patterns.

    Returns ``None`` if the path doesn't match a known cache pattern.
    """
    name = os.path.basename(str(path))
    if name.endswith(TE_CACHE_SUFFIX):
        return name.removesuffix(TE_CACHE_SUFFIX)
    if name.endswith(LATENT_CACHE_SUFFIX):
        # {stem}_{WxH}_anima.npz -> strip _anima.npz, then rsplit to remove _{WxH}
        without_suffix = name.removesuffix(LATENT_CACHE_SUFFIX)
        parts = without_suffix.rsplit("_", 1)
        return parts[0] if len(parts) >= 2 else without_suffix
    return None
