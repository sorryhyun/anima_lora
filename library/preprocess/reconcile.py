"""Reconcile resized/latent/PE/mask caches against a target_res bucket layout.

Each image's *correct* bucket is recomputed from its native size + the active
``target_res`` tiers (the same ``choose_edge`` → nearest-aspect-bucket rule
``process_image`` uses). Any cache that disagrees is stale and can be removed so
the next resize / latent / PE / mask pass regenerates it cleanly:

  - latent  ``<lora>/<rel>/{stem}_{WxH}_anima.npz``         — WxH != correct bucket
  - resized ``<resized>/<rel>/{stem}.png``                  — on-disk size != correct bucket
  - PE      ``<lora>/<rel>/{stem}_anima_pe.safetensors``  } removed when the image's
  - mask    ``<masks>/<rel>/{stem}_mask.png``             } bucket changed (neither
                                                            filename carries a resolution)

TE caches (``{stem}_anima_te.safetensors``) are text-only and never touched.

The walk → native-size index → per-image bucket check lives here so the task
layer / GUI / tests can drive it without a CLI attached; ``scripts/preprocess/
reconcile_caches.py`` is a thin argparse shell over ``reconcile_caches``.
"""

from __future__ import annotations

import os
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from library.datasets.buckets import BucketManager, buckets_for_edges, choose_edge
from library.io.walk import safe_walk

NPZ_RE = re.compile(r"^(?P<stem>.+)_(?P<w>\d{4})x(?P<h>\d{4})_anima\.npz$")
TE_RE = re.compile(r"^(?P<stem>.+)_anima_te\.safetensors$")
PE_RE = re.compile(r"^(?P<stem>.+)_anima_pe\.safetensors$")
MASK_RE = re.compile(r"^(?P<stem>.+)_mask\.png$")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


@dataclass
class StaleCaches:
    """Stale cache paths grouped by kind, plus a bucket-change tally.

    ``changed`` maps ``(current, correct)`` → count, where ``current`` is the
    stale bucket ``(W, H)`` tuple (or the string ``"png"`` when only the resized
    image's size disagreed and no latent npz was present).
    """

    npz: list[Path] = field(default_factory=list)
    png: list[Path] = field(default_factory=list)
    pe: list[Path] = field(default_factory=list)
    mask: list[Path] = field(default_factory=list)
    changed: Counter = field(default_factory=Counter)

    @property
    def n_images(self) -> int:
        return sum(self.changed.values())

    def all_paths(self) -> list[Path]:
        return [*self.npz, *self.png, *self.pe, *self.mask]


def _correct_bucket(w: int, h: int, target_res: list[int]) -> tuple[int, int]:
    """Mirror ``process_image``: ``choose_edge`` → nearest-aspect bucket in tier."""
    edge = choose_edge(w, h, target_res)
    mgr = BucketManager()
    mgr.set_predefined_resos(buckets_for_edges([edge]))
    reso, _, _ = mgr.select_bucket(w, h)
    return reso


def _native_size_index(image_dir: Path) -> dict[tuple[str, str], tuple[int, int]]:
    """``(rel_subdir, stem) -> (W, H)`` for every image under ``image_dir``.

    Walks with ``followlinks=True`` because the dataset root is often a symlink
    to nested artist dirs (a plain walk would return nothing).
    """
    idx: dict[tuple[str, str], tuple[int, int]] = {}
    with warnings.catch_warnings():
        # Large source art legitimately trips PIL's decompression-bomb guard;
        # we only read the header (.size), never decode pixels.
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        for dirpath, _, files in safe_walk(image_dir, followlinks=True):
            rel = os.path.relpath(dirpath, image_dir)
            rel = "" if rel == "." else rel
            for fn in files:
                stem, ext = os.path.splitext(fn)
                if ext.lower() not in IMAGE_EXTS:
                    continue
                try:
                    with Image.open(os.path.join(dirpath, fn)) as im:
                        idx[(rel, stem)] = im.size
                except Exception:
                    continue
    return idx


@dataclass
class OrphanCaches:
    """Cache paths whose source image no longer exists, grouped by kind.

    Unlike :class:`StaleCaches`, the text-embedding cache (``te``) *is* listed:
    when the source image is gone its ``.txt`` caption is gone too, so the TE
    cache is dead weight (a bucket change, by contrast, leaves the text valid).
    """

    npz: list[Path] = field(default_factory=list)
    te: list[Path] = field(default_factory=list)
    pe: list[Path] = field(default_factory=list)
    png: list[Path] = field(default_factory=list)
    mask: list[Path] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return len(self.all_paths())

    def all_paths(self) -> list[Path]:
        return [*self.npz, *self.te, *self.pe, *self.png, *self.mask]


def _native_keys(image_dir: Path) -> set[tuple[str, str]]:
    """``{(rel_subdir, stem)}`` for every source image under ``image_dir``.

    Existence only — no header read (cheaper than ``_native_size_index``).
    Walks with ``followlinks=True`` since the dataset root is usually a symlink
    to nested artist dirs.
    """
    keys: set[tuple[str, str]] = set()
    for dirpath, _, files in safe_walk(image_dir, followlinks=True):
        rel = os.path.relpath(dirpath, image_dir)
        rel = "" if rel == "." else rel
        for fn in files:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in IMAGE_EXTS:
                keys.add((rel, stem))
    return keys


def find_orphan_caches(
    image_dir: Path,
    resized_dir: Path,
    lora_cache_dir: Path,
    mask_dir: Path,
) -> OrphanCaches:
    """Find every cache whose source image is missing from ``image_dir``.

    A cache is an orphan when no source image shares its ``(rel_subdir, stem)``
    — i.e. the image was deleted (or moved to another subdir, leaving the old
    location's cache dead). Matching mirrors the on-disk layout: caches live
    under ``<root>/<rel>/`` paralleling ``image_dir``.
    """
    native = _native_keys(image_dir)
    orphans = OrphanCaches()

    def _walk(root: Path):
        for dirpath, _, files in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            rel = "" if rel == "." else rel
            for fn in files:
                yield rel, dirpath, fn

    # Latent / text / PE caches all live under the lora cache dir.
    for rel, dirpath, fn in _walk(lora_cache_dir):
        for rx, bucket in (
            (NPZ_RE, orphans.npz),
            (TE_RE, orphans.te),
            (PE_RE, orphans.pe),
        ):
            m = rx.match(fn)
            if m:
                if (rel, m.group("stem")) not in native:
                    bucket.append(Path(dirpath) / fn)
                break

    for rel, dirpath, fn in _walk(resized_dir):
        stem, ext = os.path.splitext(fn)
        if ext.lower() == ".png" and (rel, stem) not in native:
            orphans.png.append(Path(dirpath) / fn)

    for rel, dirpath, fn in _walk(mask_dir):
        m = MASK_RE.match(fn)
        if m and (rel, m.group("stem")) not in native:
            orphans.mask.append(Path(dirpath) / fn)

    return orphans


def delete_orphans(orphans: OrphanCaches) -> Counter:
    """Unlink every orphan path; return a ``{kind: count}`` tally."""
    removed: Counter = Counter()
    for kind, paths in (
        ("npz", orphans.npz),
        ("te", orphans.te),
        ("pe", orphans.pe),
        ("png", orphans.png),
        ("mask", orphans.mask),
    ):
        for p in paths:
            if p.exists():
                p.unlink()
                removed[kind] += 1
    return removed


def find_stale_caches(
    image_dir: Path,
    resized_dir: Path,
    lora_cache_dir: Path,
    mask_dir: Path,
    target_res: list[int],
) -> StaleCaches:
    """Scan caches and return everything inconsistent with ``target_res``.

    Iterates native images (not cache files) so every artifact is reconciled
    regardless of which caches happen to exist — an image may be mid-pipeline
    with only some of its caches built.
    """
    native = _native_size_index(image_dir)
    stale = StaleCaches()

    for (rel, stem), (w, h) in native.items():
        correct = _correct_bucket(w, h, target_res)
        reldir = Path(rel) if rel else Path()

        # Latent npzs: a stem may carry several (multi-resolution); any whose
        # filename bucket != correct is an orphan from an old bucket assignment.
        wrong_npz = [
            p
            for p in (lora_cache_dir / reldir).glob(f"{stem}_*_anima.npz")
            if (m := NPZ_RE.match(p.name))
            and (int(m.group("w")), int(m.group("h"))) != correct
        ]

        png = resized_dir / reldir / f"{stem}.png"
        png_wrong = False
        if png.exists():
            try:
                with Image.open(png) as im:
                    png_wrong = im.size != correct
            except Exception:
                png_wrong = True

        if not wrong_npz and not png_wrong:
            continue  # consistent with the configured target_res

        cur: tuple[int, int] | str = "png"
        if wrong_npz and (m := NPZ_RE.match(wrong_npz[0].name)):
            cur = (int(m.group("w")), int(m.group("h")))
        stale.changed[(cur, correct)] += 1

        stale.npz.extend(wrong_npz)
        if png_wrong:
            stale.png.append(png)
        pe = lora_cache_dir / reldir / f"{stem}_anima_pe.safetensors"
        if pe.exists():
            stale.pe.append(pe)
        mask = mask_dir / reldir / f"{stem}_mask.png"
        if mask.exists():
            stale.mask.append(mask)

    return stale


def delete_stale(stale: StaleCaches) -> Counter:
    """Unlink every stale path; return a ``{kind: count}`` tally of what went."""
    removed: Counter = Counter()
    for kind, paths in (
        ("npz", stale.npz),
        ("png", stale.png),
        ("pe", stale.pe),
        ("mask", stale.mask),
    ):
        for p in paths:
            if p.exists():
                p.unlink()
                removed[kind] += 1
    return removed


def reconcile_caches(
    image_dir: Path,
    resized_dir: Path,
    lora_cache_dir: Path,
    mask_dir: Path,
    target_res: list[int],
    *,
    delete: bool = False,
) -> tuple[StaleCaches, Counter]:
    """Find stale caches and (when ``delete``) remove them.

    Returns ``(stale, removed)`` — ``removed`` is empty on a dry run.
    """
    stale = find_stale_caches(
        image_dir, resized_dir, lora_cache_dir, mask_dir, target_res
    )
    removed = delete_stale(stale) if delete else Counter()
    return stale, removed
