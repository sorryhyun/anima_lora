"""Tests for the image_dataset/ subdir-mirror behavior across the cache,
mask, and resize pipelines.

Covers:
- ``library.io.cache.resolve_cache_path`` — flat / nested / no-image_dir /
  escaping-image_dir branches.
- ``library.datasets.image_utils.load_mask_from_dir`` — nested-preferred,
  flat-fallback, missing.
- ``library.datasets.subsets._resolve_default_mask_dir`` — priority order
  across the new + legacy candidates.
- ``anime_tools.masking.cli.merge_masks`` end-to-end through ``main()`` — `(rel_dir,
  name)` keying, flat-to-flat passthrough, mixed nested+flat inputs.
- ``library.preprocess.images.process_image`` (the anime_tools resize worker) — writes under
  ``out_dir/<rel>/`` and mirrors the caption sidecar.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# resolve_cache_path
# ---------------------------------------------------------------------------


def test_resolve_cache_path_legacy_sidecar(tmp_path: Path) -> None:
    """No cache_dir → cache lives next to the image (legacy sidecar layout)."""
    from library.io.cache import resolve_cache_path

    img = tmp_path / "img1.png"
    out = resolve_cache_path(str(img), "_anima_te.safetensors")
    assert out == str(tmp_path / "img1_anima_te.safetensors")


def test_resolve_cache_path_flat_no_image_dir(tmp_path: Path) -> None:
    """cache_dir without image_dir falls back to flat layout (no <rel> prefix)."""
    from library.io.cache import resolve_cache_path

    img = tmp_path / "charA" / "img1.png"
    img.parent.mkdir()
    cache = tmp_path / "cache"
    out = resolve_cache_path(str(img), "_suf.bin", cache_dir=str(cache))
    assert out == str(cache / "img1_suf.bin")


def test_resolve_cache_path_nested(tmp_path: Path) -> None:
    """image_dir + nested source → <rel> mirrored under cache_dir."""
    from library.io.cache import resolve_cache_path

    src = tmp_path / "image_dataset"
    img = src / "charA" / "img1.png"
    img.parent.mkdir(parents=True)
    cache = tmp_path / "cache"
    out = resolve_cache_path(
        str(img), "_suf.bin", cache_dir=str(cache), image_dir=str(src)
    )
    assert out == str(cache / "charA" / "img1_suf.bin")
    # Side effect: the nested cache dir is created so writers can drop into it.
    assert (cache / "charA").is_dir()


def test_resolve_cache_path_flat_when_source_at_root(tmp_path: Path) -> None:
    """image_dir set but image sits directly under it → no <rel>, flat layout."""
    from library.io.cache import resolve_cache_path

    src = tmp_path / "image_dataset"
    img = src / "img1.png"
    img.parent.mkdir(parents=True)
    cache = tmp_path / "cache"
    out = resolve_cache_path(
        str(img), "_suf.bin", cache_dir=str(cache), image_dir=str(src)
    )
    assert out == str(cache / "img1_suf.bin")


def test_resolve_cache_path_escaping_image_dir(tmp_path: Path) -> None:
    """An image path outside image_dir falls back to flat (no `..` in cache)."""
    from library.io.cache import resolve_cache_path

    src = tmp_path / "image_dataset"
    src.mkdir()
    outside = tmp_path / "other" / "img1.png"
    outside.parent.mkdir(parents=True)
    cache = tmp_path / "cache"
    out = resolve_cache_path(
        str(outside), "_suf.bin", cache_dir=str(cache), image_dir=str(src)
    )
    # Bails to flat — no traversal escape under the cache root.
    assert out == str(cache / "img1_suf.bin")


# ---------------------------------------------------------------------------
# load_mask_from_dir
# ---------------------------------------------------------------------------


def _write_mask(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4), value, dtype=np.uint8), mode="L").save(path)


def test_load_mask_from_dir_nested_preferred(tmp_path: Path) -> None:
    """When both nested and flat masks exist, the nested one wins."""
    from library.datasets.image_utils import load_mask_from_dir

    image_dir = tmp_path / "image_dataset"
    img = image_dir / "charA" / "img1.png"
    img.parent.mkdir(parents=True)

    mask_dir = tmp_path / "post_image_dataset" / "masks"
    _write_mask(mask_dir / "charA" / "img1_mask.png", 200)
    _write_mask(mask_dir / "img1_mask.png", 50)

    mask = load_mask_from_dir(str(mask_dir), str(img), (4, 4), image_dir=str(image_dir))
    assert mask is not None
    # Nested mask had value 200 → 200/255 in float form.
    assert abs(float(mask.mean()) - 200.0 / 255.0) < 1e-5


def test_load_mask_from_dir_flat_fallback(tmp_path: Path) -> None:
    """Nested missing → flat path is consulted as a legacy-compat fallback."""
    from library.datasets.image_utils import load_mask_from_dir

    image_dir = tmp_path / "image_dataset"
    img = image_dir / "charA" / "img1.png"
    img.parent.mkdir(parents=True)

    mask_dir = tmp_path / "masks" / "merged"
    _write_mask(mask_dir / "img1_mask.png", 50)

    mask = load_mask_from_dir(str(mask_dir), str(img), (4, 4), image_dir=str(image_dir))
    assert mask is not None
    assert abs(float(mask.mean()) - 50.0 / 255.0) < 1e-5


def test_load_mask_from_dir_missing(tmp_path: Path) -> None:
    """No mask anywhere → None."""
    from library.datasets.image_utils import load_mask_from_dir

    image_dir = tmp_path / "image_dataset"
    img = image_dir / "charA" / "img1.png"
    img.parent.mkdir(parents=True)
    mask_dir = tmp_path / "post_image_dataset" / "masks"
    mask_dir.mkdir(parents=True)

    mask = load_mask_from_dir(str(mask_dir), str(img), (4, 4), image_dir=str(image_dir))
    assert mask is None


def test_load_mask_from_dir_legacy_no_image_dir(tmp_path: Path) -> None:
    """image_dir=None preserves the pre-refactor flat lookup."""
    from library.datasets.image_utils import load_mask_from_dir

    mask_dir = tmp_path / "masks"
    img = tmp_path / "img1.png"
    _write_mask(mask_dir / "img1_mask.png", 128)

    mask = load_mask_from_dir(str(mask_dir), str(img), (4, 4))
    assert mask is not None
    assert abs(float(mask.mean()) - 128.0 / 255.0) < 1e-5


# ---------------------------------------------------------------------------
# _resolve_default_mask_dir
# ---------------------------------------------------------------------------


def test_resolve_default_mask_dir_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """post_image_dataset/masks > masks/merged > masks/sam > masks/mit > None."""
    from library.datasets.subsets import _resolve_default_mask_dir

    monkeypatch.chdir(tmp_path)
    assert _resolve_default_mask_dir() is None

    (tmp_path / "masks" / "mit").mkdir(parents=True)
    assert _resolve_default_mask_dir() == "masks/mit"

    (tmp_path / "masks" / "sam").mkdir(parents=True)
    assert _resolve_default_mask_dir() == "masks/sam"

    (tmp_path / "masks" / "merged").mkdir(parents=True)
    assert _resolve_default_mask_dir() == "masks/merged"

    (tmp_path / "post_image_dataset" / "masks").mkdir(parents=True)
    assert _resolve_default_mask_dir() == "post_image_dataset/masks"


# ---------------------------------------------------------------------------
# merge_masks.py (driver-level)
# ---------------------------------------------------------------------------


def _run_merge(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    """Run ``anime_tools.masking.cli.merge_masks:main`` with the given argv."""
    monkeypatch.setattr(sys, "argv", ["merge_masks.py", *argv])
    # main() reads sys.argv at call time; the module-level imports run once.
    merge_masks = importlib.import_module("anime_tools.masking.cli.merge_masks")
    merge_masks.main()


def test_merge_masks_nested_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SAM + MIT nested trees union into the same nested output layout."""
    sam_dir = tmp_path / "sam"
    mit_dir = tmp_path / "mit"
    out_dir = tmp_path / "out"

    # Same (rel, name) in both inputs — must merge into one nested output.
    _write_mask(sam_dir / "charA" / "img1_mask.png", 200)
    _write_mask(mit_dir / "charA" / "img1_mask.png", 100)
    # rel=charB only in SAM — passthrough to the nested target.
    _write_mask(sam_dir / "charB" / "img2_mask.png", 240)

    _run_merge(
        monkeypatch,
        [str(sam_dir), str(mit_dir), "--output-dir", str(out_dir)],
    )

    merged_a = out_dir / "charA" / "img1_mask.png"
    merged_b = out_dir / "charB" / "img2_mask.png"
    assert merged_a.exists()
    assert merged_b.exists()

    arr_a = np.array(Image.open(merged_a))
    arr_b = np.array(Image.open(merged_b))
    # pixel-wise minimum = union of masked regions.
    assert int(arr_a.flat[0]) == 100
    assert int(arr_b.flat[0]) == 240
    # Nothing leaked into the flat root of out_dir.
    assert not (out_dir / "img1_mask.png").exists()


def test_merge_masks_flat_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flat inputs (no <rel>) collide on filename and produce flat output."""
    sam_dir = tmp_path / "sam"
    mit_dir = tmp_path / "mit"
    out_dir = tmp_path / "out"
    _write_mask(sam_dir / "img1_mask.png", 200)
    _write_mask(mit_dir / "img1_mask.png", 150)

    _run_merge(
        monkeypatch,
        [str(sam_dir), str(mit_dir), "--output-dir", str(out_dir)],
    )

    merged = out_dir / "img1_mask.png"
    assert merged.exists()
    assert int(np.array(Image.open(merged)).flat[0]) == 150


def test_merge_masks_mixed_rel_does_not_collide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same stem at flat root vs. nested subdir must NOT merge — different (rel, name)."""
    sam_dir = tmp_path / "sam"
    mit_dir = tmp_path / "mit"
    out_dir = tmp_path / "out"
    _write_mask(sam_dir / "img1_mask.png", 200)
    _write_mask(mit_dir / "charA" / "img1_mask.png", 100)

    _run_merge(
        monkeypatch,
        [str(sam_dir), str(mit_dir), "--output-dir", str(out_dir)],
    )

    # Both survive with their respective single-source values (no merge).
    flat_out = out_dir / "img1_mask.png"
    nested_out = out_dir / "charA" / "img1_mask.png"
    assert flat_out.exists()
    assert nested_out.exists()
    assert int(np.array(Image.open(flat_out)).flat[0]) == 200
    assert int(np.array(Image.open(nested_out)).flat[0]) == 100


# ---------------------------------------------------------------------------
# resize stage process_image
# ---------------------------------------------------------------------------


def _write_test_image(path: Path, size: tuple[int, int] = (1024, 1024)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(64, 128, 192)).save(path)


def test_resize_images_nested_output(tmp_path: Path) -> None:
    """process_image writes under out_dir/<rel>/ when rel_dir is set."""
    from library.preprocess.images import ResizeOptions, process_image

    src = tmp_path / "image_dataset" / "charA"
    img_path = src / "cover.png"
    _write_test_image(img_path)
    # Caption sidecar should follow the same nested layout.
    img_path.with_suffix(".txt").write_text("a test caption", encoding="utf-8")

    dst = tmp_path / "post_image_dataset" / "resized"
    # Default options → the canonical 1024 tier; free-fit resize.
    name, _reso, _skipped = process_image(
        img_path, dst, ResizeOptions(), rel_dir="charA", copy_captions=True
    )

    assert name == "cover.png"
    out_png = dst / "charA" / "cover.png"
    out_txt = dst / "charA" / "cover.txt"
    assert out_png.exists(), "resized PNG not written under nested layout"
    assert out_txt.exists(), "caption sidecar not mirrored into nested layout"
    # Flat layout must NOT be populated when rel_dir is set.
    assert not (dst / "cover.png").exists()


def test_resize_images_flat_output(tmp_path: Path) -> None:
    """Empty rel_dir collapses back to the legacy flat layout (no breakage)."""
    from library.preprocess.images import ResizeOptions, process_image

    img_path = tmp_path / "image_dataset" / "cover.png"
    _write_test_image(img_path)

    dst = tmp_path / "post_image_dataset" / "resized"
    process_image(img_path, dst, ResizeOptions(), rel_dir="", copy_captions=False)
    assert (dst / "cover.png").exists()
    # No phantom subdir was created.
    assert not any(p.is_dir() for p in dst.iterdir())
