"""What a source folder and a cache folder hold; torch-free, for the GUI.

The file layout constants live here so ``cache.py`` and the GUI name files the
same way. A TE file is current while its ``source_sha256`` (or legacy
``caption_sha256``) matches the caption on disk, read from the safetensors JSON
header. A ``caption_format`` switch is invisible here; ``cache.py`` sees it and
re-encodes.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Iterator
from pathlib import Path

from library.env import resolve_under_home
from library.lines.scan import CacheCounts, CacheScan
from library.lines.scan import cache_counts as _cache_counts

IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg")
TE_SUFFIX = ".moss_te.safetensors"
LATENT_SUFFIX = ".moss_latent.safetensors"
# Optional clean reference latent per stem (i2i probe), key ``latent``.
REF_SUFFIX = ".moss_ref.safetensors"
LATENT_RE = re.compile(
    rf"^(?P<stem>.+)_(?P<w>\d+)x(?P<h>\d+){re.escape(LATENT_SUFFIX)}$"
)


def glob_escape(text: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", text)


def latent_files(
    folder: Path, stem: str | None = None
) -> Iterator[tuple[str, int, int, Path]]:
    """``(stem, width, height, path)`` of each latent file in ``folder``, or of
    ``stem``'s only."""
    pattern = "*" if stem is None else f"{glob_escape(stem)}_*"
    for p in folder.glob(pattern + LATENT_SUFFIX):
        m = LATENT_RE.match(p.name)
        if m is not None and (stem is None or m["stem"] == stem):
            yield m["stem"], int(m["w"]), int(m["h"]), p


def original_tree(src: Path) -> tuple[Path | None, frozenset[Path]]:
    """The ``image_dataset/`` folder ``src`` mirrors when ``src`` sits in the Anima
    resized tree, and the originals the exclusion ledgers name; ``(None, ∅)``
    for any other ``src``.

    anime_tools' Export under ``sidecars_only`` (what the trainer GUI seeds)
    publishes captions into the resized tree and no pixels; Anima's resize stage
    fills the PNGs in. The MOSS line has no resize stage, so a caption there whose
    PNG was never written pairs with its original instead."""
    from library.datasets.curation_actions import (
        RESIZED_DIR,
        SOURCE_DIR,
        excluded_rels,
        workspace_excluded_rels,
    )

    resized = resolve_under_home(RESIZED_DIR)
    source = resolve_under_home(SOURCE_DIR)
    for a, b in ((src, resized), (src.resolve(), resized.resolve())):
        if a.is_relative_to(b):
            rel = a.relative_to(b)
            break
    else:
        return None, frozenset()
    rels = {*excluded_rels(), *workspace_excluded_rels()}
    return source / rel, frozenset(source / r for r in rels)


def source_pairs(
    src: Path, originals: Path | None = None, skip: frozenset[Path] = frozenset()
) -> tuple[list[Path], list[tuple[str, Path, Path]]]:
    """Every image under ``src`` and, for those with a ``.txt`` beside them,
    ``(cache stem, image, caption)``. A stem that appears in several subfolders
    is prefixed with its relative folder (``sub__dir__stem``).

    With ``originals`` (a tree mirroring ``src``), a caption under ``src`` with
    no image beside it pairs with the image at the same relative folder and stem
    there, unless that image is in ``skip``."""
    images = sorted(
        p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )
    roots = {p: src for p in images}
    if originals is not None and originals.is_dir():
        have = {p.relative_to(src).with_suffix("") for p in images}
        for p in sorted(originals.rglob("*")):
            if p.suffix.lower() not in IMAGE_EXTS or not p.is_file() or p in skip:
                continue
            rel = p.relative_to(originals).with_suffix("")
            if rel in have or not (src / rel.parent / f"{p.stem}.txt").exists():
                continue
            have.add(rel)
            images.append(p)
            roots[p] = originals
    counts: dict[str, int] = {}
    for p in images:
        counts[p.stem] = counts.get(p.stem, 0) + 1
    pairs = []
    for p in images:
        rel = p.relative_to(roots[p])
        caption = src / rel.parent / f"{p.stem}.txt"
        if not caption.exists():
            continue
        stem = p.stem
        if counts[stem] > 1:
            stem = "__".join((*rel.parent.parts, p.stem))
        pairs.append((stem, p, caption))
    return images, pairs


def layout_sidecar(caption: Path) -> Path:
    """The layout-JSON prompt anime_tools' Export publishes beside ``caption``
    (``{stem}.txt``) when its caption format is ``layout_json``."""
    from anime_tools.contract import LAYOUT_JSON_SUFFIX

    return caption.with_suffix(LAYOUT_JSON_SUFFIX)


def caption_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def header_metadata(path: Path) -> dict:
    """The ``__metadata__`` of a safetensors file, or ``{}`` if unreadable."""
    try:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            return json.loads(f.read(n)).get("__metadata__") or {}
    except (OSError, ValueError, struct.error):
        return {}


def scan(src: Path | None, out: Path | None) -> CacheScan:
    images: list[Path] = []
    pairs: list[tuple[str, Path, Path]] = []
    if src is not None and src.is_dir():
        images, pairs = source_pairs(src, *original_tree(src))
    dupes = sum(1 for stem, image, _ in pairs if stem != image.stem)
    text = latents = stale = 0
    if out is not None and out.is_dir():
        latent_stems = {stem for stem, *_ in latent_files(out)}
        for stem, _, caption in pairs:
            te = out / f"{stem}{TE_SUFFIX}"
            if te.exists():
                sha = caption_sha(caption.read_text(encoding="utf-8").strip())
                meta = header_metadata(te)
                if meta.get("source_sha256", meta.get("caption_sha256")) == sha:
                    text += 1
                else:
                    stale += 1
            latents += stem in latent_stems
    return CacheScan(len(images), len(pairs), text, latents, stale, dupes)


def cache_counts(cache: Path) -> CacheCounts:
    return _cache_counts(
        cache, TE_SUFFIX, lambda d: (stem for stem, *_ in latent_files(d))
    )
