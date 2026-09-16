"""Utilities for loading cached latents and text encoder outputs from preprocessed datasets.

Provides shared helpers for discovering and loading the disk-cached
``*_anima.npz`` (VAE latents) and ``*_anima_te.safetensors`` (text encoder
outputs) files produced by the preprocessing pipeline.
"""

import glob
import logging
import os
import random
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from safetensors.torch import load_file

# Suffix conventions live in the torch-free leaf so torch-free consumers (the
# GUI) share one definition. Re-exported for call sites importing them from
# ``library.io.cache``.
from library.io.cache_names import (  # noqa: F401
    DEFAULT_PE_ENCODER,
    LATENT_CACHE_SUFFIX,
    TE_CACHE_SUFFIX,
    classify_cache_file,
    count_preprocess_caches,
    pe_cache_suffix,
)

logger = logging.getLogger(__name__)


def resolve_cache_path(
    image_abs_path: str | os.PathLike,
    suffix: str,
    cache_dir: str | os.PathLike | None = None,
    image_dir: str | os.PathLike | None = None,
) -> str:
    """Build a cache file path from a source image path + suffix.

    ``cache_dir=None`` writes the cache next to the image (legacy sidecar
    layout). With ``cache_dir`` set, the cache is redirected there; if
    ``image_dir`` is also given, the relative subpath from ``image_dir`` to
    ``image_abs_path`` is mirrored under ``cache_dir`` so nested source
    layouts produce nested caches instead of flattening.
    """
    src = str(image_abs_path)
    stem = os.path.splitext(os.path.basename(src))[0]
    if cache_dir is None:
        return os.path.splitext(src)[0] + suffix
    cache_dir_path = os.fspath(cache_dir)
    rel_dir = ""
    if image_dir is not None:
        try:
            rel = os.path.relpath(os.path.dirname(src), os.fspath(image_dir))
        except ValueError:
            rel = ""
        # "." means directly under image_dir (flat); ".." means escaped the
        # root, so also fall back to flat rather than write outside cache_dir.
        if rel and rel != "." and not rel.startswith(".."):
            rel_dir = rel
    target_dir = os.path.join(cache_dir_path, rel_dir) if rel_dir else cache_dir_path
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, stem + suffix)


def caption_key(image_path: str | os.PathLike, image_dir: str | os.PathLike | None = None) -> str:
    """Stable, subdir-aware key for the caption index (``build_caption_index``).

    Keys by path **relative to the dataset root**, extension stripped,
    separators normalized to ``/`` — e.g. ``en/1.png`` under root
    ``image_dataset`` → ``en/1``. A bare basename key would conflate
    same-stem images in different subfolders (``en/1`` vs ``ew/1``).

    ``image_dir`` is the root to relativize against; when ``None`` or the
    path escapes it, falls back to the bare posix stem (identical to the
    relpath key for un-nested datasets, so old flat indexes still match).
    """
    stem_no_ext = os.path.splitext(os.fspath(image_path))[0]
    if image_dir is not None:
        try:
            rel = os.path.relpath(stem_no_ext, os.fspath(image_dir))
        except ValueError:
            rel = ""
        if rel and rel != "." and not rel.startswith(".."):
            return rel.replace(os.sep, "/")
    return os.path.basename(stem_no_ext)


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
    """Find latent+TE cache pairs anywhere under a cache directory.

    Walks ``cache_dir`` recursively so nested layouts (caches mirrored from
    a subfoldered ``image_dataset/`` source tree) are discovered. Each
    latent NPZ is looked up next to the TE sidecar (same subdir + same
    stem), which is where the writers place them.
    """
    images = []
    te_paths = sorted(
        glob.glob(os.path.join(cache_dir, "**", f"*{TE_CACHE_SUFFIX}"), recursive=True)
    )
    for te_path in te_paths:
        stem = os.path.basename(te_path).removesuffix(TE_CACHE_SUFFIX)
        parent = os.path.dirname(te_path)
        npz_files = glob.glob(os.path.join(parent, f"{stem}_*{LATENT_CACHE_SUFFIX}"))
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
        vi = (
            random.randint(0, n - 1)
            if variant == "random"
            else min(int(variant), n - 1)
        )
        key = f"crossattn_emb_v{vi}"
        if key in sd:
            return sd[key].float()

    if "crossattn_emb_v0" in sd:
        return sd["crossattn_emb_v0"].float()
    if "crossattn_emb" in sd:
        return sd["crossattn_emb"].float()

    return None


def load_cached_text_features(
    te_path: str, *, variant: int | str = 0
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Load ``(crossattn_emb, pooled_text)`` for one sample.

    The pooled tensor is computed at load time via ``crossattn_emb.amax(dim=0)``
    (cheaper than ``.max(dim=0).values``, which also computes argmax) over the
    same variant index as the cross-attn so the two can't desync.

    Returns ``(None, None)`` if no crossattn variant is found.
    """
    sd = load_file(te_path)

    vi = 0
    if "num_variants" in sd:
        n = int(sd["num_variants"])
        vi = (
            random.randint(0, n - 1)
            if variant == "random"
            else min(int(variant), n - 1)
        )

    crossattn = None
    if f"crossattn_emb_v{vi}" in sd:
        crossattn = sd[f"crossattn_emb_v{vi}"].float()
    elif "crossattn_emb_v0" in sd:
        crossattn = sd["crossattn_emb_v0"].float()
    elif "crossattn_emb" in sd:
        crossattn = sd["crossattn_emb"].float()

    if crossattn is None:
        return None, None

    pooled = crossattn.amax(dim=0)
    return crossattn, pooled


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


# `{stem}_{Wpix}x{Hpix}_anima.npz`. The `{3,5}` digit bound matches real image
# dimensions and avoids swallowing a numeric stem suffix into the resolution.
_LATENT_PIXEL_RE = re.compile(
    r"^(?P<stem>.+)_(?P<w>\d{3,5})x(?P<h>\d{3,5})"
    + re.escape(LATENT_CACHE_SUFFIX)
    + r"$"
)


class LatentCacheFile(NamedTuple):
    """A parsed bucketed-latent cache filename."""

    path: Path
    stem: str
    width: int  # source pixels (the W in the `{W}x{H}` filename tag)
    height: int


def parse_latent_cache_name(path: str | os.PathLike) -> LatentCacheFile | None:
    """Parse a ``{stem}_{W}x{H}_anima.npz`` path into its parts.

    Returns ``(path, stem, W_px, H_px)`` or ``None`` if the basename doesn't
    match the bucketed-latent convention. Bench probes should use this rather
    than their own filename regex.
    """
    p = Path(path)
    m = _LATENT_PIXEL_RE.match(p.name)
    if m is None:
        return None
    return LatentCacheFile(p, m.group("stem"), int(m.group("w")), int(m.group("h")))


def discover_latents_by_stem(
    cache_dir: str | os.PathLike, *, recursive: bool = True
) -> dict[str, list[LatentCacheFile]]:
    """Group every cached latent NPZ under ``cache_dir`` by image stem.

    Recursive by default — the lora cache is nested by artist. Each value is
    the stem's bucket file(s) (a stem re-cached at multiple aspect ratios has
    several). Unlike :func:`discover_cached_pairs` this isn't TE-gated and
    returns every bucket, not just the first.
    """
    root = Path(cache_dir)
    pattern = "*" + LATENT_CACHE_SUFFIX
    paths = root.rglob(pattern) if recursive else root.glob(pattern)
    out: dict[str, list[LatentCacheFile]] = {}
    for parsed in (parse_latent_cache_name(p) for p in sorted(paths)):
        if parsed is not None:
            out.setdefault(parsed.stem, []).append(parsed)
    return out


# Bucketed sample discovery.
def discover_bucketed_samples(
    data_dir: Path,
    bucket: str | None,
    num_samples: int,
    seed: int,
    *,
    allow_replace: bool = False,
) -> tuple[str, list[tuple[str, str, str, str]]]:
    """Scan ``data_dir`` for (latent npz, TE sidecar) pairs grouped by bucket.

    Filename convention: ``{stem}_{Wpix}x{Hpix}_anima.npz`` paired with
    ``{stem}_anima_te.safetensors``; items without a TE sidecar are skipped.
    ``latents_{WxH}`` keys inside the npz define the bucket string (latent
    dims, not pixel dims). Used by bench/probe harnesses that need a fixed
    same-shape batch. ``bucket=None`` picks the most populous bucket;
    ``allow_replace`` resamples with replacement (logged) when the pool is
    smaller than ``num_samples``, else raises ``SystemExit``.

    Returns ``(chosen_bucket, [(stem, latent_key, npz_path, te_path), ...])``.
    """
    npz_paths = sorted(glob.glob(str(data_dir / "*_anima.npz")))
    if not npz_paths:
        raise SystemExit(f"no `*_anima.npz` in {data_dir}")

    by_bucket: dict[str, list[tuple[str, str, str, str]]] = {}
    for p in npz_paths:
        parsed = parse_latent_cache_name(p)
        if parsed is None:
            continue
        stem = parsed.stem
        te = data_dir / f"{stem}{TE_CACHE_SUFFIX}"
        if not te.exists():
            continue
        with np.load(p) as z:
            for k in z.keys():
                if k.startswith("latents_"):
                    bk = k.removeprefix("latents_")
                    by_bucket.setdefault(bk, []).append((stem, k, p, str(te)))
                    break

    if not by_bucket:
        raise SystemExit("no paired (latent, TE) samples found")

    chosen = bucket or max(by_bucket, key=lambda k: len(by_bucket[k]))
    if chosen not in by_bucket:
        top = sorted(((k, len(v)) for k, v in by_bucket.items()), key=lambda x: -x[1])[
            :5
        ]
        raise SystemExit(f"bucket {chosen!r} not found. Top buckets: {top}")

    pool = by_bucket[chosen]
    if len(pool) < num_samples:
        if not allow_replace:
            raise SystemExit(
                f"bucket {chosen!r} has {len(pool)} samples; need {num_samples}. "
                f"Pass allow_replace=True to resample with replacement."
            )
        logger.warning(
            f"bucket {chosen!r} has {len(pool)} samples; resampling with "
            f"replacement to reach {num_samples}."
        )

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=num_samples, replace=(len(pool) < num_samples))
    return chosen, [pool[i] for i in idx]
