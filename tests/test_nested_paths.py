"""Tests for the image_dataset/ subdir-mirror behavior across the cache,
mask, and resize pipelines.

Covers:
- ``library.io.cache.resolve_cache_path`` — nested / escaping-image_dir branches.
- ``library.datasets.image_utils.load_mask_from_dir`` — nested-preferred,
  flat-fallback.
- ``library.datasets.subsets.resolve_configured_mask_dir`` and the blueprint
  fallback that carries ``mask_dir`` to the subsets.
- ``anime_tools.masking.cli.merge_masks`` end-to-end through ``main()`` — `(rel_dir,
  name)` keying and the union.
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


@pytest.mark.core
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


def test_resolve_default_mask_dir_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """post_image_dataset/masks > masks/merged > masks/sam > None (masks/mit
    went with the v2 MIT removal)."""
    from library.datasets.subsets import _resolve_default_mask_dir

    monkeypatch.chdir(tmp_path)
    assert _resolve_default_mask_dir() is None

    (tmp_path / "masks" / "mit").mkdir(parents=True)
    assert _resolve_default_mask_dir() is None

    (tmp_path / "masks" / "sam").mkdir(parents=True)
    assert _resolve_default_mask_dir() == "masks/sam"

    (tmp_path / "masks" / "merged").mkdir(parents=True)
    assert _resolve_default_mask_dir() == "masks/merged"

    (tmp_path / "post_image_dataset" / "masks").mkdir(parents=True)
    assert _resolve_default_mask_dir() == "post_image_dataset/masks"


# ---------------------------------------------------------------------------
# config-level mask_dir
# ---------------------------------------------------------------------------


def test_resolve_configured_mask_dir_gates_on_existence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config-level mask_dir only reaches the subsets when the dir exists.

    ``configs/preprocess.toml`` ships ``mask_dir`` active, so a maskless
    checkout would otherwise hand every subset a nonexistent root (flipping
    ``alpha_mask`` on and suppressing the legacy auto-resolution).
    """
    from library.datasets.subsets import resolve_configured_mask_dir

    monkeypatch.chdir(tmp_path)
    assert resolve_configured_mask_dir(None) is None
    assert resolve_configured_mask_dir("post_image_dataset/masks") is None
    (tmp_path / "post_image_dataset" / "masks").mkdir(parents=True)
    assert (
        resolve_configured_mask_dir("post_image_dataset/masks")
        == "post_image_dataset/masks"
    )


def test_config_mask_dir_reaches_subsets_via_blueprint_fallback() -> None:
    """`mask_dir` rides preprocess.toml → args → the BlueprintGenerator
    fallback, and a subset's own `mask_dir` still wins over it."""
    import argparse

    from library.config.loader import BlueprintGenerator, ConfigSanitizer

    gen = BlueprintGenerator(ConfigSanitizer(support_dropout=True))
    user_config = {
        "general": {},
        "datasets": [
            {
                "subsets": [
                    {"image_dir": "a"},
                    {"image_dir": "b", "mask_dir": "own/masks"},
                ]
            }
        ],
    }
    namespace = argparse.Namespace(
        mask_dir="cfg/masks", debug_dataset=False, prior_loss_weight=1.0
    )
    subsets = gen.generate(user_config, namespace).dataset_group.datasets[0].subsets
    assert subsets[0].params.mask_dir == "cfg/masks"
    assert subsets[1].params.mask_dir == "own/masks"

    namespace.mask_dir = None
    subsets = gen.generate(user_config, namespace).dataset_group.datasets[0].subsets
    assert subsets[0].params.mask_dir is None
    assert subsets[1].params.mask_dir == "own/masks"


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
