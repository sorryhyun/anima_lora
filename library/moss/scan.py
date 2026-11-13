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
import os
import re
import struct
from collections.abc import Iterator
from pathlib import Path

from library.env import resolve_under_home
from library.lines.scan import CacheCounts, CacheScan

IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg")
TE_SUFFIX = ".moss_te.safetensors"
# Self-distillation teacher embeds per stem (library/moss/distill.py).
VIS_SUFFIX = ".moss_vis.safetensors"
LATENT_SUFFIX = ".moss_latent.safetensors"
# Optional clean reference latent per stem (i2i), key ``latent``.
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


def _tree_names(root: Path) -> dict[str, list[str]]:
    """``{folder: its file names}`` for every folder under ``root``: one
    ``os.walk`` instead of a ``Path`` and a stat per file (the GUI scans tens of
    thousands of cache files at launch)."""
    return {d: files for d, _, files in os.walk(root)}


def source_pairs(
    src: Path, originals: Path | None = None, skip: frozenset[Path] = frozenset()
) -> tuple[list[Path], list[tuple[str, Path, Path]]]:
    """Every image under ``src`` and, for those with a ``.txt`` beside them,
    ``(cache stem, image, caption)``. A stem that appears in several subfolders
    is prefixed with its relative folder (``sub__dir__stem``).

    With ``originals`` (a tree mirroring ``src``), a caption under ``src`` with
    no image beside it pairs with the image at the same relative folder and stem
    there, unless that image is in ``skip``."""
    names = _tree_names(src)
    images = sorted(
        Path(d, n)
        for d, ns in names.items()
        for n in ns
        if os.path.splitext(n)[1].lower() in IMAGE_EXTS
    )
    roots = {p: src for p in images}

    def has_caption(folder: Path, stem: str) -> bool:
        return f"{stem}.txt" in names.get(str(folder), ())

    if originals is not None and originals.is_dir():
        have = {p.relative_to(src).with_suffix("") for p in images}
        for p in sorted(
            Path(d, n)
            for d, ns in _tree_names(originals).items()
            for n in ns
            if os.path.splitext(n)[1].lower() in IMAGE_EXTS
        ):
            if p in skip:
                continue
            rel = p.relative_to(originals).with_suffix("")
            if rel in have or not has_caption(src / rel.parent, p.stem):
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
        if not has_caption(src / rel.parent, p.stem):
            continue
        caption = src / rel.parent / f"{p.stem}.txt"
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


def stale_teachers(cache: Path) -> int:
    """Samples (latent + TE file) whose teacher file is missing or older than their
    TE file. A TE file is rewritten when its caption, encoder or tail changes, so
    this tracks the teacher's stamp without reading it."""
    if not cache.is_dir():
        return 0
    n = 0
    for stem in {stem for stem, *_ in latent_files(cache)}:
        te, vis = cache / f"{stem}{TE_SUFFIX}", cache / f"{stem}{VIS_SUFFIX}"
        if te.is_file() and (
            not vis.is_file() or vis.stat().st_mtime < te.stat().st_mtime
        ):
            n += 1
    return n


def cache_text_encoder(cache: Path) -> str:
    """The TE weights a cache was encoded with (its ``encoder`` stamp), int8 when
    no TE file is there yet."""
    te = next(cache.glob(f"*{TE_SUFFIX}"), None) if cache.is_dir() else None
    if te is None:
        return "int8"
    stamp = header_metadata(te).get("encoder", "")
    return next((p for p in ("int8", "fp8") if stamp.endswith(f"{p} weights")), "bf16")


def cache_counts(cache: Path) -> CacheCounts:
    """``library.lines.scan.cache_counts`` over one ``os.scandir`` of ``cache``."""
    if not cache.is_dir():
        return CacheCounts(0, 0, 0)
    names = [e.name for e in os.scandir(cache)]
    text = {n[: -len(TE_SUFFIX)] for n in names if n.endswith(TE_SUFFIX)}
    latents = {
        m["stem"]
        for n in names
        if n.endswith(LATENT_SUFFIX) and (m := LATENT_RE.match(n)) is not None
    }
    return CacheCounts(len(text), len(latents), len(text & latents))
