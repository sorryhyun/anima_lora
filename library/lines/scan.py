"""The source / cache folder counts a line's GUI shows; torch-free.

File layout and staleness stay with each line (``library/moss/scan.py``,
``library/qwen21/scan.py``); this is what they report.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheScan:
    images: int  # images in the source folder
    pairs: int  # of those, with a .txt caption
    text_cached: int  # MOSS: current TE files; Qwen: every TE file
    latents_cached: int
    stale_text: int  # text cache behind its caption
    duplicates: int  # MOSS: stems prefixed with their folder; Qwen: refused by caching


@dataclass(frozen=True)
class CacheCounts:
    text: int
    latents: int
    samples: int  # stems with both files: what training sees


def cache_counts(
    cache: Path, text_suffix: str, latent_stems: Callable[[Path], Iterable[str]]
) -> CacheCounts:
    """Counts of ``cache``; ``latent_stems`` names the stems of its latent files."""
    if not cache.is_dir():
        return CacheCounts(0, 0, 0)
    text = {p.name[: -len(text_suffix)] for p in cache.glob(f"*{text_suffix}")}
    latents = set(latent_stems(cache))
    return CacheCounts(len(text), len(latents), len(text & latents))
