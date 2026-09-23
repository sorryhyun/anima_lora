"""What a source folder and a cache folder hold — torch-free, for the GUI.

``cache.py`` skips a file that already exists, so an edited caption is ignored
until its text cache is rebuilt; ``stale_text`` counts those.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class CacheScan:
    images: int  # images in the source folder
    pairs: int  # of those, with a .txt caption
    text_cached: int
    latents_cached: int
    stale_text: int  # caption newer than its text cache
    duplicates: int  # file names in more than one subfolder — caching refuses


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


@dataclass(frozen=True)
class CacheCounts:
    text: int
    latents: int
    samples: int  # stems with both files — what training will actually see


def cache_counts(cache: Path) -> CacheCounts:
    if not cache.is_dir():
        return CacheCounts(0, 0, 0)
    text = {p.name[: -len(".te.safetensors")] for p in cache.glob("*.te.safetensors")}
    latents = {
        p.name[: -len(".latent.safetensors")]
        for p in cache.glob("*.latent.safetensors")
    }
    return CacheCounts(len(text), len(latents), len(text & latents))
