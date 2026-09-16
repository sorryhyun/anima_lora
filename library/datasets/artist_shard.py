"""Resolve a ``--artists_shard k_N`` knob into a subset ``path_pattern``.

Each artist is a top-level subdirectory under a subset's ``image_dir`` (the
training tree mirrors the artist folders, e.g.
``post_image_dataset/resized/<artist>/<file>.png``). ``--artists_shard k_N``
sorts the artist dirs, round-robin assigns artist ``i`` to shard ``i % N``, and
restricts training to shard ``k`` (1-indexed) by emitting a ``path_pattern`` of
``<artist>/*`` globs joined with ``|`` — the alternation syntax
:func:`library.datasets.path_filter.filter_paths_by_glob` already understands.

Round-robin (not contiguous blocks) keeps per-shard image counts balanced.

Pure stdlib — the GUI and config linters can call it without importing torch.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse ``"k_N"`` → ``(k, n)`` (1-indexed k), validating ``1 <= k <= n``."""
    raw = str(spec).strip()
    parts = raw.split("_")
    if len(parts) != 2:
        raise ValueError(
            f"--artists_shard must look like 'k_N' (e.g. '1_4'), got {spec!r}"
        )
    try:
        k, n = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"--artists_shard must look like 'k_N' with integers, got {spec!r}"
        ) from exc
    if n < 1:
        raise ValueError(f"--artists_shard N must be >= 1, got {n}")
    if not (1 <= k <= n):
        raise ValueError(f"--artists_shard k must be in [1, {n}], got {k}")
    return k, n


def list_artists(image_dir: str) -> list[str]:
    """Sorted names of the immediate subdirectories under ``image_dir``."""
    try:
        entries = os.scandir(image_dir)
    except (FileNotFoundError, NotADirectoryError):
        return []
    with entries:
        return sorted(e.name for e in entries if e.is_dir())


def shard_of(artists: list[str], k: int, n: int) -> list[str]:
    """Round-robin shard ``k`` (1-indexed) of ``n`` over a sorted artist list."""
    return artists[k - 1 :: n]


def _pattern_for(artists: list[str]) -> str:
    """``|``-joined ``<artist>/*`` globs, fnmatch-escaping special chars."""
    return "|".join(f"{glob.escape(a)}/*" for a in artists)


def apply_artist_shard(
    user_config: dict,
    spec: str,
    *,
    resolve: Optional[Callable[[str], object]] = None,
) -> dict:
    """Set ``path_pattern`` on every non-reg subset to its shard's artist globs.

    Mutates ``user_config`` in place. ``resolve`` maps a possibly-relative ``image_dir`` to an
    on-disk path for directory enumeration (pass ``resolve_under_home``); the
    written ``path_pattern`` stays relative to ``image_dir``, as the filter
    expects. Raises if a subset already carries an explicit ``path_pattern`` (a
    shard restriction can't be composed with an arbitrary OR-glob in one
    string). Returns ``{image_dir: {"n_artists", "n_shard", "artists"}}`` for
    logging.
    """
    k, n = parse_shard(spec)
    info: dict = {}
    for ds in user_config.get("datasets", []):
        for sub in ds.get("subsets", []):
            if sub.get("is_reg"):
                continue
            image_dir = sub.get("image_dir")
            if not image_dir:
                continue
            existing = sub.get("path_pattern")
            if existing and existing != "*":
                raise ValueError(
                    f"--artists_shard cannot compose with an explicit "
                    f"path_pattern={existing!r} on subset image_dir={image_dir!r}; "
                    f"remove one."
                )
            base = str(resolve(image_dir)) if resolve else image_dir
            artists = list_artists(base)
            if not artists:
                logger.warning(
                    "--artists_shard: no artist subdirectories under %s "
                    "(resolved %s) — path_pattern left unset",
                    image_dir,
                    base,
                )
                continue
            shard = shard_of(artists, k, n)
            sub["path_pattern"] = _pattern_for(shard)
            info[image_dir] = {
                "n_artists": len(artists),
                "n_shard": len(shard),
                "artists": shard,
            }
    return info
