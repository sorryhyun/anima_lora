"""Tests for ``library.preprocess._dataset`` — the shared walk/group/skip loop
extracted from the ``preprocess/cache_*.py`` scripts (Phase 1 of
``docs/proposal/tooling_architecture.md``).

These exercise the orchestration helpers without any model/encoder, so they
run in the unit suite. End-to-end content parity for the PE cache is gated
separately on ``make preprocess-pe`` (needs the encoder weights).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from library.datasets.buckets import freefit_band_for_edge


def _tokens(reso: tuple[int, int]) -> int:
    return (reso[0] // 16) * (reso[1] // 16)


def _in_tier_band(reso: tuple[int, int], edge: int) -> bool:
    lo, hi = freefit_band_for_edge(edge)
    return lo <= _tokens(reso) <= hi


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_move_linked_files_preserves_layout_and_sidecars(tmp_path: Path) -> None:
    from library.datasets.curation_actions import move_linked_files

    source = tmp_path / "image_dataset"
    target = tmp_path / "post_image_dataset" / "moved"
    image = source / "charA" / "cover.png"
    _write_image(image, (8, 8))
    for suffix in (".txt", ".caption", ".json", ".txt.history.jsonl"):
        image.with_suffix(suffix).write_text(suffix, encoding="utf-8")

    moved = move_linked_files(image, source_root=source, target_root=target)

    expected = [
        target / "charA" / "cover.png",
        target / "charA" / "cover.txt",
        target / "charA" / "cover.caption",
        target / "charA" / "cover.json",
        target / "charA" / "cover.txt.history.jsonl",
    ]
    assert moved == expected
    assert all(path.exists() for path in expected)
    assert not image.exists()
    assert not image.with_suffix(".txt").exists()


def test_load_curation_decisions_rebases_to_source_subdir(tmp_path: Path) -> None:
    from library.datasets.curation_actions import (
        load_curation_decisions,
        save_curation_decisions,
    )

    source = tmp_path / "image_dataset"
    decisions_path = tmp_path / "post_image_dataset" / "curation_decisions.json"
    save_curation_decisions(
        decisions_path,
        source_dir=str(source),
        images={
            "fuzichoco/keep.png": {"action": "use"},
            "fuzichoco/skip.png": {"action": "skip"},
            "other/skip.png": {"action": "skip"},
        },
    )

    decisions = load_curation_decisions(
        decisions_path,
        source_dir=source / "fuzichoco",
    )

    assert decisions == {
        "keep.png": {"action": "use"},
        "skip.png": {"action": "skip"},
    }


def test_walk_images_flat(tmp_path: Path) -> None:
    from library.preprocess import walk_images

    _write_image(tmp_path / "b.png", (8, 8))
    _write_image(tmp_path / "a.png", (8, 8))
    (tmp_path / "caption.txt").write_text("not an image")

    paths = walk_images(tmp_path, recursive=False)
    assert [p.name for p in paths] == ["a.png", "b.png"]  # sorted, txt excluded


def test_walk_images_recursive_same_stem_across_folders_ok(tmp_path: Path) -> None:
    from library.preprocess import walk_images

    _write_image(tmp_path / "charA" / "cover.png", (8, 8))
    _write_image(tmp_path / "charB" / "cover.png", (8, 8))

    paths = walk_images(tmp_path, recursive=True)
    assert len(paths) == 2  # same stem in different folders is fine


def test_walk_images_path_pattern_filters_relative_paths(tmp_path: Path) -> None:
    from library.preprocess import walk_images

    _write_image(tmp_path / "charA" / "cover.png", (8, 8))
    _write_image(tmp_path / "charB" / "cover.png", (8, 8))

    paths = walk_images(tmp_path, recursive=True, pattern="charA/*")
    assert [p.relative_to(tmp_path).as_posix() for p in paths] == ["charA/cover.png"]


def test_walk_images_collision_within_folder_raises(tmp_path: Path) -> None:
    from library.preprocess import walk_images

    _write_image(tmp_path / "cover.png", (8, 8))
    _write_image(tmp_path / "cover.jpg", (8, 8))

    with pytest.raises(ValueError, match="Duplicate image stems"):
        walk_images(tmp_path, recursive=False)


def test_group_by_shape(tmp_path: Path) -> None:
    from library.preprocess import group_by_shape

    _write_image(tmp_path / "a.png", (8, 16))
    _write_image(tmp_path / "b.png", (8, 16))
    _write_image(tmp_path / "c.png", (16, 8))

    groups = group_by_shape(
        [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]
    )
    assert {k: sorted(p.name for p in v) for k, v in groups.items()} == {
        (8, 16): ["a.png", "b.png"],
        (16, 8): ["c.png"],
    }


def test_partition_cached(tmp_path: Path) -> None:
    from library.preprocess import partition_cached

    imgs = [tmp_path / f"img{i}.png" for i in range(3)]
    for p in imgs:
        _write_image(p, (8, 8))
    # Pretend img1 is already cached.
    (tmp_path / "img1.cached").touch()

    pending, skipped = partition_cached(imgs, lambda p: tmp_path / f"{p.stem}.cached")
    assert skipped == 1
    assert [p.name for p in pending] == ["img0.png", "img2.png"]


def test_count_preprocess_caches_path_pattern_filters_nested_caches(
    tmp_path: Path,
) -> None:
    from gui.dialogs import count_preprocess_caches

    (tmp_path / "charA").mkdir()
    (tmp_path / "charB").mkdir()
    (tmp_path / "charA" / "cover_1024x1024_anima.npz").touch()
    (tmp_path / "charA" / "cover_anima_te.safetensors").touch()
    (tmp_path / "charB" / "cover_1024x1024_anima.npz").touch()
    (tmp_path / "charB" / "cover_anima_te.safetensors").touch()

    assert count_preprocess_caches(tmp_path, "charA/*") == {
        "latents": 1,
        "te": 1,
        "pe": 0,
    }


# Minimal Danbooru-KB CSV (mirrors anime_tools/tests/test_caption_correction.py).
def _tag_csv(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "name,category,post_count,description",
                '1girl,0,10,"[인물 > 인원수] count"',
                'solo,0,10,"[인물 > 인원수] count"',
                'hatsune_miku,4,10,"[캐릭터 > vocaloid] character"',
                'vocaloid,3,10,"[작품 > series] copyright"',
                'sincos,1,10,"[작가 > illustrator] artist"',
                'best_quality,5,10,"[메타 > 화질] quality"',
                'highres,5,10,"[메타 > 화질] resolution meta"',
                'commentary,5,10,"[메타 > 정보_요청] artist commentary"',
                'long_hair,0,10,"[머리카락 > 머리 길이] general"',
                'copyright_notice,0,10,"[메타 > 정보_요청] misleading description"',
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_write_corrected_preprocess_captions_preserves_source(tmp_path: Path) -> None:
    from anime_tools.captions.correction import (
        CaptionCorrectionOptions,
        load_tag_knowledge_base,
    )
    from anime_tools.stages.captions import write_corrected_preprocess_captions

    source = tmp_path / "image_dataset"
    resized = tmp_path / "post_image_dataset" / "resized"
    _write_image(source / "charA" / "cover.jpg", (64, 64))
    _write_image(resized / "charA" / "cover.png", (64, 64))
    original = "long hair, vocaloid, hatsune miku, 1girl"
    (source / "charA" / "cover.txt").write_text(original, encoding="utf-8")

    stats = write_corrected_preprocess_captions(
        source,
        resized,
        load_tag_knowledge_base(_tag_csv(tmp_path / "tags.csv")),
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            trigger_word="@dataset-trigger",
        ),
        recursive=True,
    )

    assert stats.written == 1
    assert (source / "charA" / "cover.txt").read_text(encoding="utf-8") == original
    assert (resized / "charA" / "cover.txt").read_text(encoding="utf-8") == (
        "1girl, hatsune miku, vocaloid, @dataset-trigger, long hair"
    )


def test_write_corrected_preprocess_captions_keeps_a_revised_caption_without_master(
    tmp_path: Path,
) -> None:
    """A revised caption is *the* caption (anime_tools >= 0.4.0, revised-first):
    with no master beside it, it is corrected in place, not treated as stale."""
    from anime_tools.captions.correction import CaptionCorrectionOptions
    from anime_tools.captions.correction import load_tag_knowledge_base
    from anime_tools.stages.captions import write_corrected_preprocess_captions

    source = tmp_path / "image_dataset"
    resized = tmp_path / "post_image_dataset" / "resized"
    source.mkdir()
    _write_image(resized / "charA" / "cover.png", (64, 64))
    revised = resized / "charA" / "cover.txt"
    revised.write_text("smile, 1girl", encoding="utf-8")

    stats = write_corrected_preprocess_captions(
        source,
        resized,
        load_tag_knowledge_base(_tag_csv(tmp_path / "tags.csv")),
        options=CaptionCorrectionOptions(),
        recursive=True,
    )

    assert stats.no_caption == 0
    assert stats.from_master == 0
    assert revised.exists()


def test_confirm_train_using_cache_requires_pe_when_repa_on(tmp_path: Path) -> None:
    """use_repa Train gating: a built latent/TE cache that lacks PE sidecars
    must return None (→ auto-chain the PE-caching preprocess) rather than
    offering to reuse the cache and launching a silent-no-op REPA run.

    Only the ``None`` branches are exercised — they return before any
    QMessageBox is constructed, so no QApplication is needed.
    """
    from gui.dialogs import confirm_train_using_cache

    # Core caches present, PE absent.
    (tmp_path / "cover_1024x1024_anima.npz").touch()
    (tmp_path / "cover_anima_te.safetensors").touch()

    # REPA on → PE mandatory → treat as cache-missing (auto-preprocess).
    assert confirm_train_using_cache(None, tmp_path, require_pe=True) is None


def test_confirm_train_using_cache_empty_returns_none(tmp_path: Path) -> None:
    from gui.dialogs import confirm_train_using_cache

    assert confirm_train_using_cache(None, tmp_path) is None
    assert confirm_train_using_cache(None, tmp_path, require_pe=True) is None


# ---------------------------------------------------------------------------
# Pre-flight cache-coverage probes — let the entry points skip the (slow) model
# load when a dataset is already fully cached. Model-free.
# ---------------------------------------------------------------------------


def test_count_pending_latents_per_resolution(tmp_path: Path) -> None:
    from library.preprocess import count_pending_latents, get_latents_npz_path

    data = tmp_path / "imgs"
    cache = tmp_path / "cache"
    _write_image(data / "a.png", (64, 64))
    _write_image(data / "b.png", (64, 64))

    assert count_pending_latents(data, cache_dir=cache) == (2, 2)

    # Cache a's 64x64 latent (key latents_{H//8}x{W//8} = latents_8x8).
    npz = get_latents_npz_path(
        data / "a.png", (64, 64), cache_dir=cache, image_dir=data
    )
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **{"latents_8x8": np.zeros((16, 8, 8), dtype=np.float32)})
    assert count_pending_latents(data, cache_dir=cache) == (1, 2)


def test_count_pending_pe_existence(tmp_path: Path) -> None:
    from library.preprocess import count_pending_pe, pe_cache_path_for

    data = tmp_path / "imgs"
    cache = tmp_path / "cache"
    _write_image(data / "a.png", (64, 64))
    _write_image(data / "b.png", (64, 64))

    assert count_pending_pe(data, "pe_spatial", cache_dir=cache) == (2, 2)

    sidecar = pe_cache_path_for(
        data / "a.png", "pe_spatial", cache_dir=cache, image_dir=data
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.touch()
    assert count_pending_pe(data, "pe_spatial", cache_dir=cache) == (1, 2)


def test_count_pending_text_counts_uncaptioned(tmp_path: Path) -> None:
    # Uncaptioned images are candidates too (encoded with an empty caption), so
    # they count toward pending until their TE cache exists.
    from library.preprocess import count_pending_text
    from library.preprocess.text import _te_cache_path

    data = tmp_path / "imgs"
    cache = tmp_path / "cache"
    _write_image(data / "a.png", (64, 64))
    _write_image(data / "b.png", (64, 64))  # no .txt — still a candidate
    (data / "a.txt").write_text("hello", encoding="utf-8")

    assert count_pending_text(data, cache_dir=cache, min_pixels=0) == (2, 2)

    te = _te_cache_path(data / "a.png", cache, data)
    te.parent.mkdir(parents=True, exist_ok=True)
    te.touch()
    assert count_pending_text(data, cache_dir=cache, min_pixels=0) == (1, 2)


def test_count_pending_text_recaches_when_caption_is_newer(tmp_path: Path) -> None:
    from library.preprocess import count_pending_text
    from library.preprocess.text import _te_cache_path

    data = tmp_path / "imgs"
    cache = tmp_path / "cache"
    img = data / "a.png"
    caption = data / "a.txt"
    _write_image(img, (64, 64))
    caption.write_text("old", encoding="utf-8")
    te = _te_cache_path(img, cache, data)
    te.parent.mkdir(parents=True, exist_ok=True)
    te.touch()

    os.utime(caption, (100, 100))
    os.utime(te, (200, 200))
    assert count_pending_text(data, cache_dir=cache, min_pixels=0) == (0, 1)

    os.utime(caption, (300, 300))
    assert count_pending_text(data, cache_dir=cache, min_pixels=0) == (1, 1)


def test_count_pending_text_min_pixels_filter(tmp_path: Path) -> None:
    from library.preprocess import count_pending_text

    data = tmp_path / "imgs"
    _write_image(data / "small.png", (16, 16))  # 256 px — below threshold
    _write_image(data / "big.png", (64, 64))  # 4096 px

    # total reflects the post-min_pixels candidate set (small filtered out).
    assert count_pending_text(data, min_pixels=1000) == (1, 1)


def test_count_pending_text_keep_rel_stems_filters_nested_paths(tmp_path: Path) -> None:
    from library.preprocess import count_pending_text
    from library.preprocess.text import _te_cache_path

    data = tmp_path / "imgs"
    cache = tmp_path / "cache"
    _write_image(data / "charA" / "cover.png", (64, 64))
    _write_image(data / "charB" / "cover.png", (64, 64))

    assert count_pending_text(
        data,
        cache_dir=cache,
        recursive=True,
        keep_rel_stems={"charA/cover"},
        min_pixels=0,
    ) == (1, 1)

    te = _te_cache_path(data / "charA" / "cover.png", cache, data)
    te.parent.mkdir(parents=True, exist_ok=True)
    te.touch()
    assert count_pending_text(
        data,
        cache_dir=cache,
        recursive=True,
        keep_rel_stems={"charA/cover"},
        min_pixels=0,
    ) == (0, 1)


# ---------------------------------------------------------------------------
# Model-free end-to-end coverage for the loops moved into library/preprocess/
# (item A of the proposal). cache_pe_features / cache_latents /
# cache_text_embeddings need real encoders, so they stay gated on make
# preprocess-*; these two need no model.
# ---------------------------------------------------------------------------


def test_count_preprocess_caches_is_pe_encoder_aware(tmp_path: Path) -> None:
    # Regression: the PE sidecar suffix is ``_anima_{encoder}.safetensors`` and
    # the default REPA encoder is ``pe_spatial`` — counting must default to that,
    # not the PE-Core ``pe`` variant (a hardcoded ``_anima_pe`` once made the GUI
    # blind to a fully-cached PE-Spatial dataset and force-relaunch preprocess).
    from library.io.cache_names import (
        LATENT_CACHE_SUFFIX,
        TE_CACHE_SUFFIX,
        count_preprocess_caches,
        pe_cache_suffix,
    )

    (tmp_path / f"a{LATENT_CACHE_SUFFIX}").write_bytes(b"")
    (tmp_path / f"a{TE_CACHE_SUFFIX}").write_bytes(b"")
    (tmp_path / f"a{pe_cache_suffix('pe_spatial')}").write_bytes(b"")

    # Default encoder (pe_spatial) sees the spatial sidecar.
    counts = count_preprocess_caches(tmp_path)
    assert counts == {"latents": 1, "te": 1, "pe": 1}

    # Asking for the PE-Core encoder finds no matching sidecar.
    assert count_preprocess_caches(tmp_path, pe_encoder="pe")["pe"] == 0


def test_resize_to_buckets_writes_and_mirrors_layout(tmp_path: Path) -> None:
    from library.preprocess import resize_to_buckets

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    # Two images >= 0.5MP (so min_pixels keeps them); one nested.
    _write_image(src / "a.png", (900, 900))
    (src / "a.txt").write_text("caption a")
    _write_image(src / "charB" / "b.png", (900, 900))

    stats, bucket_counts = resize_to_buckets(
        src, dst, recursive=True, workers=1, verbose=False
    )
    assert stats.seen == 2
    assert stats.written == 2
    assert sum(bucket_counts.values()) == 2

    out_a = dst / "a.png"
    out_b = dst / "charB" / "b.png"
    assert out_a.exists() and out_b.exists()  # nested layout mirrored
    assert (dst / "a.txt").read_text() == "caption a"  # caption copied
    # Output matches a real bucket resolution.
    with Image.open(out_a) as im:
        assert (im.width, im.height) in bucket_counts


def test_resize_to_buckets_path_pattern_preserves_filtered_layout(
    tmp_path: Path,
) -> None:
    from library.preprocess import resize_to_buckets

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_image(src / "charA" / "a.png", (900, 900))
    _write_image(src / "charB" / "b.png", (900, 900))

    stats, bucket_counts = resize_to_buckets(
        src,
        dst,
        recursive=True,
        path_pattern="charA/*",
        min_pixels=0,
        workers=1,
        verbose=False,
    )
    assert stats.seen == 1
    assert stats.written == 1
    assert sum(bucket_counts.values()) == 1
    assert (dst / "charA" / "a.png").exists()
    assert not (dst / "charB" / "b.png").exists()


def test_resize_to_buckets_applies_curation_skip_decision(tmp_path: Path) -> None:
    from library.preprocess import resize_to_buckets

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_image(src / "keep.png", (900, 900))
    _write_image(src / "skip.png", (900, 900))
    _write_image(src / "move.png", (900, 900))

    stats, bucket_counts = resize_to_buckets(
        src,
        dst,
        min_pixels=0,
        workers=1,
        verbose=False,
        curation_decisions={
            "keep.png": {"action": "use"},
            "skip.png": {"action": "skip"},
            "move.png": {"action": "move"},
        },
    )

    assert stats.seen == 3
    assert stats.skipped == 2
    assert stats.written == 1
    assert sum(bucket_counts.values()) == 1
    assert (dst / "keep.png").exists()
    assert not (dst / "skip.png").exists()
    assert not (dst / "move.png").exists()
    assert (src / "skip.png").exists()
    assert (src / "move.png").exists()


def test_resize_to_buckets_accumulates_decision_and_min_pixel_skips(
    tmp_path: Path,
) -> None:
    from library.preprocess import resize_to_buckets

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_image(src / "keep.png", (900, 900))
    _write_image(src / "decision_skip.png", (900, 900))
    _write_image(src / "too_small.png", (64, 64))

    stats, bucket_counts = resize_to_buckets(
        src,
        dst,
        min_pixels=500_000,
        workers=1,
        verbose=False,
        curation_decisions={"decision_skip.png": {"action": "skip"}},
    )

    assert stats.seen == 3
    assert stats.skipped == 2
    assert stats.written == 1
    assert sum(bucket_counts.values()) == 1
    assert (dst / "keep.png").exists()
    assert not (dst / "decision_skip.png").exists()
    assert not (dst / "too_small.png").exists()


def test_resize_to_buckets_default_tier_does_not_upscale_to_multitier(
    tmp_path: Path,
) -> None:
    """Regression: target_res=None (no preprocess.toml / no flag, and the bare
    [1024] that tasks.py strips to None) must resize against the single 1024
    tier, NOT a larger tier. The old multi-tier catalog else-branch shoved a
    0.73MP portrait into the 1536-tier (1024, 2160) bucket — a 3x upscale. Under
    free-fit the image lands inside the 1024 tier's token band at its native
    aspect."""
    from library.preprocess import resize_to_buckets

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_image(src / "portrait.png", (589, 1233))  # 0.73MP, ar 0.478

    for target_res in (None, [1024]):
        stats, _ = resize_to_buckets(
            src,
            dst,
            target_res=target_res,
            min_pixels=0,
            workers=1,
            verbose=False,
            overwrite=True,
        )
        assert stats.written == 1
        with Image.open(dst / "portrait.png") as im:
            reso = (im.width, im.height)
        assert _in_tier_band(reso, 1024), f"{target_res}: {reso} escaped 1024 tier"
        assert reso != (1024, 2160), f"{target_res}: reproduced the upscale bug"
        # native aspect preserved to sub-patch (no AR-snap)
        assert abs(reso[0] / reso[1] - 589 / 1233) < (16 / min(reso))


def test_resize_to_buckets_skips_up_to_date_and_rebuckets_on_tier_change(
    tmp_path: Path,
) -> None:
    """Idempotent skip: a re-run touches nothing; adding a 768 tier re-resizes
    only the image whose target bucket actually moves."""
    from library.preprocess import resize_to_buckets

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_image(src / "small.png", (700, 860))  # ~0.6MP → flips to 768 tier
    _write_image(src / "big.png", (1400, 1050))  # ~1.5MP → stays 1024 tier

    # First pass at the single 1024 tier writes both.
    stats, _ = resize_to_buckets(
        src, dst, target_res=[1024], min_pixels=0, workers=1, verbose=False
    )
    assert (stats.written, stats.skipped) == (2, 0)

    # Re-run, same tiers: both already at their bucket → all skipped.
    stats, _ = resize_to_buckets(
        src, dst, target_res=[1024], min_pixels=0, workers=1, verbose=False
    )
    assert (stats.written, stats.skipped) == (0, 2)

    # Add the 768 tier: only `small` moves bucket → exactly one re-resize.
    stats, counts = resize_to_buckets(
        src, dst, target_res=[768, 1024], min_pixels=0, workers=1, verbose=False
    )
    assert (stats.written, stats.skipped) == (1, 1)
    with Image.open(dst / "small.png") as im:
        assert _in_tier_band((im.width, im.height), 768)

    # overwrite=True forces both even when up to date.
    stats, _ = resize_to_buckets(
        src,
        dst,
        target_res=[768, 1024],
        min_pixels=0,
        workers=1,
        verbose=False,
        overwrite=True,
    )
    assert (stats.written, stats.skipped) == (2, 0)


def test_resize_to_buckets_min_pixels_filter(tmp_path: Path) -> None:
    from library.preprocess import resize_to_buckets

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_image(src / "tiny.png", (64, 64))  # 4096 px, below default 0.5MP

    stats, _ = resize_to_buckets(src, dst, workers=1, verbose=False)
    assert stats.seen == 1
    assert stats.skipped == 1
    assert stats.written == 0
    assert not (dst / "tiny.png").exists()


def test_reconcile_caches_removes_only_wrong_bucket(tmp_path: Path) -> None:
    """Under [768, 1024], the small image's old 1024-tier caches are stale; the
    big image's correct caches and every TE sidecar are left untouched."""
    from library.preprocess import find_stale_caches, delete_stale

    image_dir = tmp_path / "image_dataset"
    resized = tmp_path / "post" / "resized"
    lora = tmp_path / "post" / "lora"
    masks = tmp_path / "post" / "masks"
    target_res = [768, 1024]

    # small: 0.6MP → flips to the 768 tier (640x864); caches still at 1024 (896x1152).
    _write_image(image_dir / "charA" / "small.png", (700, 860))
    _write_image(resized / "charA" / "small.png", (896, 1152))  # wrong size
    (lora / "charA").mkdir(parents=True)
    small_npz = lora / "charA" / "small_0896x1152_anima.npz"
    small_pe = lora / "charA" / "small_anima_pe.safetensors"
    small_te = lora / "charA" / "small_anima_te.safetensors"  # must survive
    small_npz.touch()
    small_pe.touch()
    small_te.touch()
    (masks / "charA").mkdir(parents=True)
    small_mask = masks / "charA" / "small_mask.png"
    small_mask.touch()

    # big: 1.47MP → stays 1024 tier (1200x896); caches already correct.
    _write_image(image_dir / "big.png", (1400, 1050))
    _write_image(resized / "big.png", (1200, 896))  # correct size
    big_npz = lora / "big_1200x0896_anima.npz"
    big_npz.touch()

    stale = find_stale_caches(image_dir, resized, lora, masks, target_res)
    assert stale.n_images == 1
    assert stale.npz == [small_npz]
    assert stale.png == [resized / "charA" / "small.png"]
    assert stale.pe == [small_pe]
    assert stale.mask == [small_mask]
    assert big_npz not in stale.npz  # consistent image untouched

    removed = delete_stale(stale)
    assert removed == {"npz": 1, "png": 1, "pe": 1, "mask": 1}
    assert not small_npz.exists() and not small_pe.exists() and not small_mask.exists()
    assert small_te.exists()  # TE is text-only — never reconciled
    assert big_npz.exists()
