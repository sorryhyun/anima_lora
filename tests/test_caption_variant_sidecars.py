"""Variant-sidecar pipeline: generation (caption step), encode (TE step),
round-trip I/O, and reconcile orphan cleanup.

The sidecar ({stem}.variants.txt) is the source of truth that makes the visible
train-time variants and the encoded TE cache the same thing — these tests pin
that contract end to end without loading the real text encoder (the encode batch
is stubbed).
"""

from __future__ import annotations

import random
from pathlib import Path
from unittest import mock

import torch
from safetensors import safe_open

from anime_tools.captions.correction import (
    CaptionCorrectionOptions,
    load_tag_knowledge_base,
)
from anime_tools.stages.captions import write_corrected_preprocess_captions
from library.preprocess import text as te
from anime_tools.captions.variants import (
    read_variants_sidecar,
    variants_sidecar_path,
    write_variants_sidecar,
)
from library.preprocess.reconcile import delete_orphans, find_orphan_caches


def _csv(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "name,category,post_count,description",
                '1girl,0,10,"[인물 > 인원수] count"',
                'solo,0,10,"[인물 > 인원수] count"',
                'blue_hair,0,10,"[머리카락] general"',
                'smile,0,10,"[표정] general"',
                'sincos,1,10,"[작가] artist"',
            ]
        ),
        encoding="utf-8",
    )
    return path


# ----- sidecar I/O round-trip ---------------------------------------------


def test_sidecar_path_multidot_stem():
    p = variants_sidecar_path(Path("sub/a.b.png"))
    assert p.name == "a.b.variants.txt"
    assert p.parent.name == "sub"


def test_sidecar_roundtrip_and_comment_tolerance(tmp_path):
    sp = tmp_path / "x.variants.txt"
    rows = [
        ("v0", "1girl, @sincos"),
        ("v1", "@sincos, 1girl"),
        ("r1", "swing, @sincos"),
    ]
    write_variants_sidecar(sp, rows)
    # Header comment is present; parser skips it and any blank line.
    raw = sp.read_text(encoding="utf-8")
    assert raw.splitlines()[0].startswith("#")
    assert read_variants_sidecar(sp) == rows


def test_sidecar_parser_skips_malformed_lines(tmp_path):
    sp = tmp_path / "x.variants.txt"
    sp.write_text("# header\nv0\t1girl\n\nnotabhere\nv1\t@sincos, 1girl\n", "utf-8")
    assert read_variants_sidecar(sp) == [("v0", "1girl"), ("v1", "@sincos, 1girl")]


# ----- caption step writes the sidecar ------------------------------------


def _kb(tmp_path):
    return load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))


def test_caption_step_writes_sidecar_v0_is_corrected(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("1girl, blue hair, @sincos, smile", encoding="utf-8")
    (dst / "a.png").write_bytes(b"")

    random.seed(0)
    stats = write_corrected_preprocess_captions(
        src,
        dst,
        _kb(tmp_path),
        options=CaptionCorrectionOptions(insert_no_artist=False),
        recursive=False,
        num_variants=4,
        tag_dropout_rate=0.3,
    )
    assert stats.variants_written == 1
    corrected = (dst / "a.txt").read_text(encoding="utf-8")
    rows = read_variants_sidecar(variants_sidecar_path(dst / "a.png"))
    labels = [label for label, _ in rows]
    assert labels == ["v0", "v1", "v2", "v3"]
    # v0 is the corrected caption that also lands in {stem}.txt.
    assert dict(rows)["v0"] == corrected
    # Every variant keeps the same multiset minus dropped tags; @sincos is
    # protected (prefix), so it survives in all.
    assert all("@sincos" in text for _, text in rows)


def test_caption_step_passthrough_v0_is_raw(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    raw = "1girl, blue hair, @sincos, smile"
    (src / "a.txt").write_text(raw, encoding="utf-8")
    (dst / "a.png").write_bytes(b"")

    random.seed(0)
    write_corrected_preprocess_captions(
        src,
        dst,
        _kb(tmp_path),
        options=CaptionCorrectionOptions(),
        recursive=False,
        correct=False,
        num_variants=2,
        tag_dropout_rate=0.0,
    )
    rows = dict(read_variants_sidecar(variants_sidecar_path(dst / "a.png")))
    assert rows["v0"] == raw  # passthrough: no reordering
    assert (dst / "a.txt").read_text(encoding="utf-8") == raw


def test_caption_step_idempotent_rerun_keeps_sidecar_stable(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("1girl, blue hair, @sincos", encoding="utf-8")
    (dst / "a.png").write_bytes(b"")
    opts = CaptionCorrectionOptions()

    random.seed(0)
    write_corrected_preprocess_captions(
        src, dst, _kb(tmp_path), options=opts, recursive=False, num_variants=4
    )
    before = variants_sidecar_path(dst / "a.png").read_text(encoding="utf-8")
    random.seed(99)  # different seed must NOT cause a rewrite
    stats = write_corrected_preprocess_captions(
        src, dst, _kb(tmp_path), options=opts, recursive=False, num_variants=4
    )
    after = variants_sidecar_path(dst / "a.png").read_text(encoding="utf-8")
    assert stats.variants_written == 0
    assert before == after


def test_caption_step_off_removes_stale_sidecar(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("1girl, @sincos", encoding="utf-8")
    (dst / "a.png").write_bytes(b"")
    opts = CaptionCorrectionOptions()

    write_corrected_preprocess_captions(
        src, dst, _kb(tmp_path), options=opts, recursive=False, num_variants=2
    )
    sidecar = variants_sidecar_path(dst / "a.png")
    assert sidecar.exists()
    stats = write_corrected_preprocess_captions(
        src, dst, _kb(tmp_path), options=opts, recursive=False, num_variants=0
    )
    assert not sidecar.exists()
    assert stats.variants_removed == 1


def test_caption_step_revised_caption_without_master_keeps_sidecar(tmp_path):
    """Revised-first (anime_tools >= 0.4.0): a revised caption with no master is
    the caption, so its sidecar is regenerated from it rather than removed."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (dst / "a.png").write_bytes(b"")
    (dst / "a.txt").write_text("1girl, smile", encoding="utf-8")
    write_variants_sidecar(variants_sidecar_path(dst / "a.png"), [("v0", "stale")])

    stats = write_corrected_preprocess_captions(
        src,
        dst,
        _kb(tmp_path),
        options=CaptionCorrectionOptions(),
        recursive=False,
        num_variants=2,
    )
    assert stats.no_caption == 0
    sidecar = variants_sidecar_path(dst / "a.png")
    assert sidecar.exists()
    assert "stale" not in sidecar.read_text(encoding="utf-8")


def test_caption_step_no_caption_at_all_removes_orphan_sidecar(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    # Resized image + leftover sidecar, but neither a revised nor a master caption.
    (dst / "a.png").write_bytes(b"")
    write_variants_sidecar(variants_sidecar_path(dst / "a.png"), [("v0", "stale")])

    stats = write_corrected_preprocess_captions(
        src,
        dst,
        _kb(tmp_path),
        options=CaptionCorrectionOptions(),
        recursive=False,
        num_variants=2,
    )
    assert stats.no_caption == 1
    assert not variants_sidecar_path(dst / "a.png").exists()


# ----- TE step encodes the sidecar verbatim -------------------------------


def _fake_encode(captions, *a, **k):
    n = len(captions)
    pe = torch.stack([torch.full((2, 3), float(i)) for i in range(n)])
    am = torch.ones(n, 2, dtype=torch.int32)
    ti = torch.zeros(n, 2, dtype=torch.long)
    ta = torch.ones(n, 2, dtype=torch.int32)
    return pe, am, ti, ta, None


def test_te_encodes_sidecar_into_variant_keys(tmp_path):
    d = tmp_path
    (d / "a.png").write_bytes(b"")
    (d / "a.txt").write_text("1girl, @sincos, blue hair", encoding="utf-8")
    write_variants_sidecar(
        variants_sidecar_path(d / "a.png"),
        [
            ("v0", "1girl, @sincos, blue hair"),
            ("v1", "1girl, @sincos"),
            ("r1", "swing"),
        ],
    )

    with mock.patch.object(te, "_encode_batch", _fake_encode):
        stats = te.cache_text_embeddings(
            d,
            object(),
            object(),
            object(),
            device=torch.device("cpu"),
            recursive=False,
            min_pixels=0,
            caption_shuffle_variants=0,  # sidecar present → variant mode regardless
        )
    assert stats.written == 1
    with safe_open(str(d / "a_anima_te.safetensors"), framework="pt") as f:
        keys = set(f.keys())
        assert {"prompt_embeds_v0", "prompt_embeds_v1", "prompt_embeds_r1"} <= keys
        assert "prompt_embeds_v2" not in keys
        assert f.get_tensor("num_variants").item() == 2
        assert f.get_tensor("num_randomized").item() == 1


def test_te_single_caption_when_no_sidecar_and_no_variants(tmp_path):
    d = tmp_path
    (d / "a.png").write_bytes(b"")
    (d / "a.txt").write_text("1girl, @sincos", encoding="utf-8")

    with mock.patch.object(te, "_encode_batch", _fake_encode):
        te.cache_text_embeddings(
            d,
            object(),
            object(),
            object(),
            device=torch.device("cpu"),
            recursive=False,
            min_pixels=0,
            caption_shuffle_variants=0,
        )
    with safe_open(str(d / "a_anima_te.safetensors"), framework="pt") as f:
        keys = set(f.keys())
        # Legacy flat layout — no suffixed variant keys, no markers.
        assert "prompt_embeds" in keys
        assert not any(k.startswith("prompt_embeds_v") for k in keys)
        assert "num_variants" not in keys


def test_te_recaches_when_sidecar_is_newer(tmp_path):
    d = tmp_path
    (d / "a.png").write_bytes(b"")
    (d / "a.txt").write_text("1girl", encoding="utf-8")
    sidecar = variants_sidecar_path(d / "a.png")
    write_variants_sidecar(sidecar, [("v0", "1girl"), ("v1", "1girl")])

    with mock.patch.object(te, "_encode_batch", _fake_encode):
        te.cache_text_embeddings(
            d,
            object(),
            object(),
            object(),
            device=torch.device("cpu"),
            recursive=False,
            min_pixels=0,
        )
        cache = d / "a_anima_te.safetensors"
        # Bump the sidecar past the cache → next pass must re-encode, not skip.
        future = cache.stat().st_mtime + 100
        import os

        os.utime(sidecar, (future, future))
        stats = te.cache_text_embeddings(
            d,
            object(),
            object(),
            object(),
            device=torch.device("cpu"),
            recursive=False,
            min_pixels=0,
        )
    assert stats.written == 1 and stats.skipped == 0


# ----- reconcile orphan cleanup -------------------------------------------


def test_reconcile_lists_and_deletes_orphan_variant_sidecars(tmp_path):
    image_dir = tmp_path / "image_dataset"
    resized = tmp_path / "resized"
    lora = tmp_path / "lora"
    masks = tmp_path / "masks"
    for d in (image_dir, resized, lora, masks):
        d.mkdir()

    # 'kept' has a live source image; 'gone' does not.
    (image_dir / "kept.png").write_bytes(b"")
    (resized / "kept.png").write_bytes(b"")
    (resized / "kept.txt").write_text("c", encoding="utf-8")
    write_variants_sidecar(variants_sidecar_path(resized / "kept.png"), [("v0", "c")])

    (resized / "gone.png").write_bytes(b"")
    (resized / "gone.txt").write_text("c", encoding="utf-8")
    write_variants_sidecar(variants_sidecar_path(resized / "gone.png"), [("v0", "c")])

    orphans = find_orphan_caches(image_dir, resized, lora, masks)
    orphan_names = {p.name for p in orphans.txt}
    assert orphan_names == {"gone.txt", "gone.variants.txt"}
    # kept sidecar/caption must NOT be flagged.
    assert all("kept" not in p.name for p in orphans.txt)

    removed = delete_orphans(orphans)
    assert removed["txt"] == 2
    assert not (resized / "gone.variants.txt").exists()
    assert (resized / "kept.variants.txt").exists()


# ----- position clauses survive the mirror --------------------------------


def _clause_corpus(tmp_path, master, derived):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text(master, encoding="utf-8")
    (dst / "a.png").write_bytes(b"")
    (dst / "a.txt").write_text(derived, encoding="utf-8")
    return src, dst


def test_mirror_keeps_clauses_the_master_does_not_have(tmp_path):
    """`caption-position` writes only ``resized/`` — a plain mirror would undo it."""
    src, dst = _clause_corpus(
        tmp_path,
        "1girl, blue hair, smile, @sincos",
        "1girl, smile, @sincos. On the left, blue hair.",
    )

    stats = write_corrected_preprocess_captions(
        src,
        dst,
        _kb(tmp_path),
        options=CaptionCorrectionOptions(),
        recursive=False,
    )

    assert stats.clauses_preserved == 1
    out = (dst / "a.txt").read_text(encoding="utf-8")
    assert out.endswith("On the left, blue hair.")
    # The moved tag stays moved: asserted in the clause, not back in the bag.
    assert "blue hair" not in out.split(". On the")[0]
    # …and the bag is still bucket-corrected around it.
    assert "@sincos" in out and "smile" in out


def test_mirror_reruns_are_stable_on_a_clause_caption(tmp_path):
    src, dst = _clause_corpus(
        tmp_path,
        "1girl, blue hair, smile, @sincos",
        "1girl, smile, @sincos. On the left, blue hair.",
    )
    opts = CaptionCorrectionOptions()
    write_corrected_preprocess_captions(
        src, dst, _kb(tmp_path), options=opts, recursive=False
    )
    first = (dst / "a.txt").read_text(encoding="utf-8")

    stats = write_corrected_preprocess_captions(
        src, dst, _kb(tmp_path), options=opts, recursive=False
    )

    assert (dst / "a.txt").read_text(encoding="utf-8") == first
    assert stats.unchanged == 1


def test_a_master_edit_does_not_reach_a_revised_caption(tmp_path):
    """Revised-first (anime_tools >= 0.4.0): the revised caption is the authority
    once it exists; deleting it is how a master edit gets re-mirrored."""
    src, dst = _clause_corpus(
        tmp_path,
        "1girl, blue hair, smile, @sincos, solo",
        "1girl, smile, @sincos. On the left, blue hair.",
    )

    stats = write_corrected_preprocess_captions(
        src, dst, _kb(tmp_path), options=CaptionCorrectionOptions(), recursive=False
    )
    out = (dst / "a.txt").read_text(encoding="utf-8")
    assert stats.from_master == 0
    assert "solo" not in out
    assert out.endswith("On the left, blue hair.")

    (dst / "a.txt").unlink()
    stats = write_corrected_preprocess_captions(
        src, dst, _kb(tmp_path), options=CaptionCorrectionOptions(), recursive=False
    )
    out = (dst / "a.txt").read_text(encoding="utf-8")
    assert stats.from_master == 1
    assert "solo" in out and "blue hair" in out


def test_mirror_regenerates_the_sidecar_around_restored_clauses(tmp_path):
    """v0 changed, so the sidecar the TE step encodes has to be redrawn."""
    src, dst = _clause_corpus(
        tmp_path,
        "1girl, blue hair, smile, @sincos",
        "1girl, smile, @sincos. On the left, blue hair.",
    )
    random.seed(0)

    write_corrected_preprocess_captions(
        src,
        dst,
        _kb(tmp_path),
        options=CaptionCorrectionOptions(),
        recursive=False,
        num_variants=4,
    )

    rows = dict(read_variants_sidecar(variants_sidecar_path(dst / "a.png")))
    assert rows["v0"] == (dst / "a.txt").read_text(encoding="utf-8")
    # Clauses are atomic in every variant, never dissolved into the bag.
    assert all("On the left, blue hair." in text for text in rows.values())


def test_the_revised_caption_wins_over_a_master_edit(tmp_path):
    """Revised-first (anime_tools >= 0.4.0): once an image has a revised caption,
    a hand-edit of its master no longer reaches it — edit the revised caption,
    or delete it to re-mirror."""
    src, dst = _clause_corpus(
        tmp_path,
        "1girl, smile, @sincos. On the right, blue hair.",
        "1girl, smile, @sincos. On the left, blue hair.",
    )

    stats = write_corrected_preprocess_captions(
        src, dst, _kb(tmp_path), options=CaptionCorrectionOptions(), recursive=False
    )

    assert stats.clauses_preserved == 1
    assert stats.from_master == 0
    assert "On the left, blue hair." in (dst / "a.txt").read_text(encoding="utf-8")
