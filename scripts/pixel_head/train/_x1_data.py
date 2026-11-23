"""The ×1 line's held-out images: P1's test artists' images, one text image per text
artist, and every other cached pair to train on.

The two artist lists are data, not code: ``output/pixel_head/x1_split.json`` (local, out of
the repo), ``{"test_artists": [...], "text_artists": [...]}``. ``test_artists`` is what
:func:`pick_samples` returns at its defaults on the cache it was made on (readout_1118_2219);
``text_artists`` are the artists with the most "Japanese text" captions in
``post_image_dataset/resized``.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from library.env import resolve_under_home
from library.io.cache import CachedImage, discover_cached_pairs
from scripts.pixel_head.train._common import npz_tokens

SPLIT = "output/pixel_head/x1_split.json"


def split_artists(key: str) -> tuple[str, ...]:
    """``key`` ("test_artists" | "text_artists") of the local split file."""
    path = Path(resolve_under_home(SPLIT))
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: the ×1 held-out split (local, not in the repo)"
        )
    return tuple(json.loads(path.read_text())[key])


def pick_samples(
    cache_root: Path,
    n_train: int = 256,
    n_test_artists: int = 8,
    per_artist: int = 4,
    max_tokens: int = 4608,
    seed: int = 0,
):
    """P1's picker → (its train images, test images, test artists)."""
    rng = random.Random(seed)
    artists = sorted(d.name for d in cache_root.iterdir() if d.is_dir())
    rng.shuffle(artists)
    test_artists, train_artists = artists[:n_test_artists], artists[n_test_artists:]

    def ok(ci):
        return npz_tokens(ci.npz_path) <= max_tokens

    test = []
    for a in test_artists:
        pairs = [p for p in discover_cached_pairs(str(cache_root / a)) if ok(p)]
        rng.shuffle(pairs)
        test += pairs[:per_artist]
    pool = []
    for a in train_artists:
        pool += [p for p in discover_cached_pairs(str(cache_root / a)) if ok(p)]
    rng.shuffle(pool)
    return pool[:n_train], test, test_artists


def caption_path(ci: CachedImage) -> Path:
    artist = Path(ci.npz_path).parent.name
    return Path(
        resolve_under_home(f"post_image_dataset/resized/{artist}/{ci.stem}.txt")
    )


def pick_text(cache_root: Path, exclude: set[str]) -> list[CachedImage]:
    """Per text artist, the image whose quoted caption text is longest."""
    out = []
    for a in split_artists("text_artists"):
        best, best_len = None, 0
        for ci in discover_cached_pairs(str(cache_root / a)):
            if ci.npz_path in exclude:
                continue
            p = caption_path(ci)
            if not p.exists():
                continue
            cap = p.read_text(encoding="utf-8")
            if "japanese text" not in cap.lower():
                continue
            n = sum(len(s) for s in re.findall(r'"([^"]*)"', cap))
            if n > best_len:
                best, best_len = ci, n
        if best is not None:
            out.append(best)
    return out


def held_out_split(cache_root: Path) -> dict[str, list[CachedImage]]:
    """{"train" | "test" | "text": cached pairs}: P1's test images, one text image per
    text artist, and every pair outside those images and the test artists."""
    p1_train, test, picked = pick_samples(cache_root)
    test_artists = split_artists("test_artists")
    assert tuple(picked) == test_artists, (
        f"held-out set did not reproduce: the picker shuffles {cache_root}'s artist "
        f"dirs, and they gave {picked}"
    )
    text = pick_text(cache_root, {ci.npz_path for ci in p1_train})
    held = {ci.npz_path for ci in text}
    train = [
        ci
        for a in sorted(d.name for d in cache_root.iterdir() if d.is_dir())
        if a not in test_artists
        for ci in discover_cached_pairs(str(cache_root / a))
        if ci.npz_path not in held
    ]
    return {"train": train, "test": test, "text": text}
