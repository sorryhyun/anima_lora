"""Shared walk / group / skip loop for the preprocess cache scripts.

Used by ``preprocess/cache_latents.py`` / ``cache_text_embeddings.py`` /
``cache_pe_encoder.py``. Holds only the pipeline-agnostic orchestration:

1. enumerate the dataset images under a directory (optionally recursive, with
   the per-subfolder stem-collision check the stem-keyed cache layout requires),
2. pre-skip entries whose sidecar already exists (idempotent re-runs),
3. group the remainder by pixel shape so one batched encoder forward serves a
   whole group (same ``(W, H)`` → same bucket → same output shape).

The encoder, suffix, and save format stay in each caller. The scan + stem check
reuse ``library.datasets.image_utils``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

from library.datasets.image_utils import _assert_unique_stems, glob_images_pathlib
from library.datasets.subsets import filter_paths_by_glob


@dataclass
class PreprocessStats:
    """Outcome counts for a cache pass.

    ``seen`` is every image enumerated under the source dir; ``written`` were
    encoded + saved this run; ``skipped`` already had a sidecar; ``failed``
    raised during encode/save (callers that don't track failures leave it 0).
    """

    seen: int = 0
    written: int = 0
    skipped: int = 0
    failed: int = 0


def walk_images(
    data_dir: Path,
    recursive: bool = False,
    pattern: str | None = None,
) -> list[Path]:
    """Enumerate dataset images under ``data_dir``, sorted and de-duplicated.

    With ``recursive`` set, walks subfolders; the same stem may repeat across
    folders (the nested cache layout disambiguates by subdir) but two images
    that share a stem *within one folder* would overwrite each other's
    stem-keyed sidecar, so that raises ``ValueError`` (via
    :func:`library.datasets.image_utils._assert_unique_stems`).

    ``pattern`` is an optional ``fnmatch`` glob (``|``-OR-combined) matched
    against each path relative to ``data_dir``, with the same semantics as the
    training subset ``path_pattern`` (:func:`filter_paths_by_glob`). ``None``,
    empty, or ``"*"`` keeps everything. The uniqueness check runs on the
    filtered set so a narrowed pattern can't trip a collision in files it
    excludes.
    """
    paths = glob_images_pathlib(data_dir, recursive)
    if pattern and pattern != "*":
        keep = filter_paths_by_glob([str(p) for p in paths], str(data_dir), pattern)
        paths = [p for p, k in zip(paths, keep) if k]
    _assert_unique_stems([str(p) for p in paths], source_label=str(data_dir))
    return paths


def group_by_shape(paths: Iterable[Path]) -> dict[tuple[int, int], list[Path]]:
    """Group image paths by their ``(W, H)`` pixel size.

    Reads each image header (cheap — no full decode) to get its size. Images
    of the same shape land in the same encoder bucket, so a single batched
    forward can serve the whole group without intra-batch padding.
    """
    groups: dict[tuple[int, int], list[Path]] = {}
    for p in paths:
        with Image.open(p) as img:
            size = img.size  # (W, H)
        groups.setdefault(size, []).append(p)
    return groups


def partition_cached(
    paths: Iterable[Path], cache_path_for: Callable[[Path], Path]
) -> tuple[list[Path], int]:
    """Split ``paths`` into ``(pending, skipped_count)`` by sidecar existence.

    ``cache_path_for(p)`` returns the sidecar path for image ``p``; entries
    whose sidecar already exists are counted as skipped (so workers never
    decode them) and the rest are returned as ``pending`` for encoding.
    """
    pending: list[Path] = []
    skipped = 0
    for p in paths:
        if cache_path_for(p).exists():
            skipped += 1
        else:
            pending.append(p)
    return pending, skipped
