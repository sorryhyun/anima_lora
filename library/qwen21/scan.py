"""What a source folder and a cache folder hold — torch-free, for the GUI.

``cache.py`` skips a file that already exists, so an edited caption is ignored
until its text cache is rebuilt; ``stale_text`` counts those.
"""

from __future__ import annotations

from pathlib import Path

from library.lines.scan import CacheCounts, CacheScan
from library.lines.scan import cache_counts as _cache_counts

IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg")
_LATENT_SUFFIX = ".latent.safetensors"


def find_images(src: Path) -> list[Path]:
    """Every image under ``src``, subfolders included (``post_image_dataset/
    resized`` keeps one folder per artist)."""
    return sorted(
        p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )


def duplicate_stems(images: list[Path]) -> dict[str, list[Path]]:
    """Stems seen more than once — the cache is flat, keyed by stem."""
    seen: dict[str, list[Path]] = {}
    for p in images:
        seen.setdefault(p.stem, []).append(p)
    return {stem: paths for stem, paths in seen.items() if len(paths) > 1}


def scan(src: Path | None, out: Path | None) -> CacheScan:
    stems: list[tuple[str, Path]] = []
    images = 0
    dupes = 0
    if src is not None and src.is_dir():
        found = find_images(src)
        dupes = len(duplicate_stems(found))
        for path in found:
            images += 1
            caption = path.with_suffix(".txt")
            if caption.exists():
                stems.append((path.stem, caption))
    text = latents = stale = 0
    if out is not None and out.is_dir():
        for stem, caption in stems:
            te = out / f"{stem}.te.safetensors"
            if te.exists():
                text += 1
                if caption.stat().st_mtime > te.stat().st_mtime:
                    stale += 1
            if (out / f"{stem}.latent.safetensors").exists():
                latents += 1
    return CacheScan(images, len(stems), text, latents, stale, dupes)


def cache_counts(cache: Path) -> CacheCounts:
    return _cache_counts(
        cache,
        ".te.safetensors",
        lambda d: (
            p.name[: -len(_LATENT_SUFFIX)] for p in d.glob(f"*{_LATENT_SUFFIX}")
        ),
    )
