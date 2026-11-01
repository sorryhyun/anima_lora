"""The curation-side copies in ``anime_tools._walk`` must behave
exactly like the trainer originals they duplicate (both walk the same caption
master). If one side changes, change the other — or this fails."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from anime_tools import _walk
from library.datasets import image_utils
from library.io import cache as io_cache
from library.io import walk as io_walk
from library.preprocess import _dataset


def test_image_extensions_identical():
    assert _walk.IMAGE_EXTENSIONS == image_utils.IMAGE_EXTENSIONS


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for rel in ("a/1.png", "a/2.jpg", "b/1.PNG", "b/sub/3.webp", "c.txt"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return root


@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize("pattern", [None, "a/*|b/sub/*"])
def test_walk_images_parity(tmp_path: Path, recursive: bool, pattern):
    root = _tree(tmp_path)
    assert _walk.walk_images(root, recursive, pattern) == _dataset.walk_images(
        root, recursive, pattern
    )


def test_walk_images_collision_parity(tmp_path: Path):
    root = _tree(tmp_path)
    (root / "a" / "1.jpg").write_bytes(b"")
    with pytest.raises(ValueError):
        _walk.walk_images(root, recursive=True)
    with pytest.raises(ValueError):
        _dataset.walk_images(root, recursive=True)


def test_safe_walk_parity(tmp_path: Path):
    root = _tree(tmp_path)
    os.symlink(root / "a", root / "b" / "back")  # cycle-ish link
    assert list(_walk.safe_walk(root)) == list(io_walk.safe_walk(root))


@pytest.mark.parametrize(
    "image_path, image_dir",
    [("ds/en/1.png", "ds"), ("ds/1.png", "ds"), ("/x/y.png", "/z"), ("q.png", None)],
)
def test_caption_key_parity(image_path, image_dir):
    assert _walk.caption_key(image_path, image_dir) == io_cache.caption_key(
        image_path, image_dir
    )
