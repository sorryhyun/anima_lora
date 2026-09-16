"""Torch-free cache-filename conventions for preprocessed sidecars.

The suffixes the preprocess pipeline writes, plus the by-name classifier /
counter built on them (:func:`classify_cache_file`,
:func:`count_preprocess_caches`). Stdlib-only at import time so the PySide6 GUI
(which cannot import torch — see ``gui/CLAUDE.md``) shares these rules instead
of copying the literals.
"""

from __future__ import annotations

import os
from pathlib import Path

# VAE latents: ``{stem}_{WxH}_anima.npz`` (resolution infix added by the caller).
LATENT_CACHE_SUFFIX = "_anima.npz"

# Text-encoder cross-attention embeddings: ``{stem}_anima_te.safetensors``.
TE_CACHE_SUFFIX = "_anima_te.safetensors"


def demoted_latents_key(width: int, height: int) -> str:
    """NPZ key of the σ-demote sibling latent (sigma_lowres).

    Lives *inside* the image's native ``{stem}_{WxH}_anima.npz`` — no sibling
    file. Deliberately NOT prefixed ``latents_``: readers like
    ``library/io/cache.py::load_cached_latents`` grab the first ``latents_*``
    key and must never see the demoted entry. ``(width, height)`` is the
    demoted pixel bucket.
    """
    return f"demoted_{height // 8}x{width // 8}"


# Default REPA / PE vision encoder (configurable via the `repa_encoder` knob).
DEFAULT_PE_ENCODER = "pe_spatial"


def pe_cache_suffix(encoder: str | None = None) -> str:
    """Sidecar suffix for a PE/vision encoder: ``_anima_{encoder}.safetensors``.

    ``encoder=None`` (or blank) resolves to :data:`DEFAULT_PE_ENCODER`
    (``pe_spatial``), matching the default ``repa_encoder``.
    """
    name = (encoder or "").strip() or DEFAULT_PE_ENCODER
    return f"_anima_{name}.safetensors"


def classify_cache_file(name: str, pe_encoder: str | None = None) -> str | None:
    """Bucket a cache filename into ``"latents"`` / ``"te"`` / ``"pe"`` (or None).

    ``pe_encoder`` picks which PE variant counts as ``"pe"`` (defaults to
    ``pe_spatial``). ``te`` is tested before ``pe`` so the ``pe`` encoder can
    never shadow a ``_anima_te`` sidecar.
    """
    if name.endswith(TE_CACHE_SUFFIX):
        return "te"
    if name.endswith(pe_cache_suffix(pe_encoder)):
        return "pe"
    if name.endswith(LATENT_CACHE_SUFFIX):
        return "latents"
    return None


def count_preprocess_caches(
    cache_dir: str | os.PathLike,
    path_pattern: str | None = None,
    pe_encoder: str | None = None,
) -> dict[str, int]:
    """Count latent / TE / PE cache sidecars under ``cache_dir`` by filename.

    Returns ``{"latents", "te", "pe"}`` counts (zeros if the directory is
    missing). Walks recursively; ``path_pattern`` (a glob relative to
    ``cache_dir``) optionally narrows the walk the same way training's
    ``path_pattern`` filter does.
    """
    out = {"latents": 0, "te": 0, "pe": 0}
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return out
    paths = [p for p in cache_dir.rglob("*") if p.is_file()]
    if path_pattern and path_pattern != "*":
        # Lazy import keeps this module a stdlib-only leaf at import time.
        from library.datasets.path_filter import filter_paths_by_glob

        keep = filter_paths_by_glob(
            [str(p) for p in paths], str(cache_dir), path_pattern
        )
        paths = [p for p, k in zip(paths, keep) if k]
    for p in paths:
        kind = classify_cache_file(p.name, pe_encoder)
        if kind:
            out[kind] += 1
    return out
