"""Cycle-safe directory walking for the project's symlinked dataset trees."""

from __future__ import annotations

import os
from typing import Iterator, Union

__all__ = ["safe_walk"]


def safe_walk(
    top: Union[str, os.PathLike], *, followlinks: bool = True
) -> Iterator[tuple[str, list[str], list[str]]]:
    """``os.walk`` that follows symlinks but never revisits a directory.

    The dataset roots here (``image_dataset`` and the caption master) are
    symlinks to trees of (sometimes cross-linked) artist dirs, so callers must
    pass ``followlinks=True`` to descend into them at all — a plain walk yields
    nothing. But ``os.walk`` has **no** cycle detection: a symlink pointing back
    up the tree (or two dirs that link to each other) makes a ``followlinks``
    walk loop forever, which is the classic "preprocess/training just hangs with
    no error" failure.

    This tracks the real path of every directory entered and prunes any child
    already visited from ``dirnames`` in place (the documented ``os.walk``
    pruning contract), which both breaks cycles and de-dupes diamond joins.
    ``seen`` is seeded with the root's real path so a symlink pointing straight
    back at ``top`` is caught too. Yields the same
    ``(dirpath, dirnames, filenames)`` triples as ``os.walk``.
    """
    seen: set[str] = {os.path.realpath(top)}
    for dirpath, dirnames, filenames in os.walk(top, followlinks=followlinks):
        kept: list[str] = []
        for d in dirnames:
            real = os.path.realpath(os.path.join(dirpath, d))
            if real in seen:
                continue
            seen.add(real)
            kept.append(d)
        dirnames[:] = kept  # prune in place so os.walk won't re-descend a cycle
        yield dirpath, dirnames, filenames
