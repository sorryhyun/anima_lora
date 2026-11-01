"""Tests for the identity pair sampler behind the soft-tokens contrastive negatives.

Covers candidate-pool restriction, the shuffled (unrelated) negative, the hard /
back-off tiers, draw() dispatch and the tag-Jaccard weight.
"""

from __future__ import annotations

import json
import random

import pytest

from library.datasets.identity_pairs import IdentityPairSampler


def _write_index(tmp_path, image_meta):
    """Build a caption_index.json (image_meta + inverted groups) from a
    {stem: {path, character, copyright, artist}} dict."""
    groups = {"character": {}, "copyright": {}, "artist": {}}
    for stem, meta in image_meta.items():
        for axis in groups:
            for tag in meta.get(axis, []):
                groups[axis].setdefault(tag, []).append(stem)
    for axis in groups:
        groups[axis] = {t: sorted(s) for t, s in sorted(groups[axis].items())}
    index = {
        "meta": {"n_images": len(image_meta)},
        "image_meta": image_meta,
        "groups": groups,
    }
    p = tmp_path / "caption_index.json"
    p.write_text(json.dumps(index), encoding="utf-8")
    return str(p)


@pytest.fixture
def index_path(tmp_path):
    meta = {
        # frieren: 3 imgs across 2 artists
        "a1": {
            "path": "art_x/a1.txt",
            "character": ["frieren"],
            "copyright": ["sousou no frieren"],
            "artist": ["@x"],
        },
        "a2": {
            "path": "art_x/a2.txt",
            "character": ["frieren"],
            "copyright": ["sousou no frieren"],
            "artist": ["@x"],
        },
        "a3": {
            "path": "art_y/a3.txt",
            "character": ["frieren"],
            "copyright": ["sousou no frieren"],
            "artist": ["@y"],
        },
        # fern: same franchise, different character, single image (no char positive)
        "b1": {
            "path": "art_y/b1.txt",
            "character": ["fern"],
            "copyright": ["sousou no frieren"],
            "artist": ["@y"],
        },
        # unrelated identity, shares artist @y only
        "c1": {
            "path": "art_y/c1.txt",
            "character": ["miku"],
            "copyright": ["vocaloid"],
            "artist": ["@y"],
        },
    }
    return _write_index(tmp_path, meta)


def test_restrict_stems(index_path):
    # Exclude a3 (validation split) → only a2 reachable for a1 at char tier.
    s = IdentityPairSampler(
        index_path, min_level="artist", restrict_stems={"a1", "a2", "b1", "c1"}
    )
    rng = random.Random(0)
    refs = {s.resolve("a1", rng)[0] for _ in range(20)}
    assert refs == {"a2"}


def test_shuffled_avoids_identity(index_path):
    s = IdentityPairSampler(index_path, min_level="artist")
    rng = random.Random(1)
    # a1's identity = {frieren, sousou no frieren}; shuffled should pick an
    # unrelated stem (c1 shares neither; b1 shares the franchise).
    picks = {s.shuffled("a1", rng)[0] for _ in range(40)}
    assert "c1" in picks
    assert not (picks & {"a1"})  # never the target


# ── Soft-tokens contrastive hard negatives + tag Jaccard ────────────────────


def test_hard_negative_same_artist_different_character(index_path):
    s = IdentityPairSampler(index_path, min_level="artist")
    rng = random.Random(0)
    # a3 (frieren, @y): same-artist siblings with a different, tagged character
    # are b1 (fern) and c1 (miku).
    refs = {s.hard_negative("a3", rng) for _ in range(20)}
    chosen = {stem for stem, _ in refs}
    assert chosen <= {"b1", "c1"}
    assert all(level == "hard" for _, level in refs)


def test_hard_negative_falls_back_when_no_disjoint_sibling(index_path):
    s = IdentityPairSampler(index_path, min_level="artist")
    rng = random.Random(0)
    # a1 (frieren, @x): the only same-artist sibling is a2, which shares the
    # character → no genuine hard negative → fall back to shuffled.
    stem, level = s.hard_negative("a1", rng)
    assert level == "shuffled"
    assert stem != "a1"


def test_hard_negative_backoff_prefers_artist_tier(index_path):
    s = IdentityPairSampler(index_path, min_level="artist")
    rng = random.Random(0)
    # a3 (frieren, @y) has same-artist disjoint-character siblings b1/c1 → the
    # artist tier fires before copyright is ever consulted.
    refs = {s.hard_negative_backoff("a3", rng) for _ in range(20)}
    assert {stem for stem, _ in refs} <= {"b1", "c1"}
    assert all(level == "hard_artist" for _, level in refs)


def test_hard_negative_backoff_copyright_tier_rescues(index_path):
    s = IdentityPairSampler(index_path, min_level="artist")
    rng = random.Random(0)
    # a1 (frieren, @x): the only same-artist sibling a2 shares the character, so
    # plain hard_negative falls back to shuffled — but the franchise tier finds
    # b1 (fern, same copyright, different character), the back-off's whole point.
    assert s.hard_negative("a1", rng)[1] == "shuffled"
    stem, level = s.hard_negative_backoff("a1", rng)
    assert (stem, level) == ("b1", "hard_copyright")


def test_hard_negative_backoff_original_tier(tmp_path):
    # Two original works by @x (no character tags), plus an unrelated franchise
    # image. The OC target o1 has no character to disjoint on → the back-off's
    # original tier returns the same-artist sibling o2, not shuffled.
    meta = {
        "o1": {
            "path": "x/o1.txt",
            "character": [],
            "copyright": ["original"],
            "artist": ["@x"],
        },
        "o2": {
            "path": "x/o2.txt",
            "character": [],
            "copyright": ["original"],
            "artist": ["@x"],
        },
        "f1": {
            "path": "y/f1.txt",
            "character": ["miku"],
            "copyright": ["vocaloid"],
            "artist": ["@y"],
        },
    }
    s = IdentityPairSampler(_write_index(tmp_path, meta), min_level="artist")
    rng = random.Random(0)
    # plain hard has no character → shuffled; backoff rescues via the original tier.
    assert s.hard_negative("o1", rng)[1] == "shuffled"
    stem, level = s.hard_negative_backoff("o1", rng)
    assert (stem, level) == ("o2", "hard_original")


def test_hard_negative_backoff_original_tier_only_for_original(tmp_path):
    # A franchise target with no diff-character sibling must NOT borrow an
    # original negative — the original tier is gated on the target being OC.
    meta = {
        "f1": {
            "path": "x/f1.txt",
            "character": ["miku"],
            "copyright": ["vocaloid"],
            "artist": ["@x"],
        },
        "o1": {
            "path": "x/o1.txt",
            "character": [],
            "copyright": ["original"],
            "artist": ["@x"],
        },
    }
    s = IdentityPairSampler(_write_index(tmp_path, meta), min_level="artist")
    stem, level = s.hard_negative_backoff("f1", random.Random(0))
    assert level == "shuffled"


def test_draw_dispatch(index_path):
    s = IdentityPairSampler(index_path, min_level="artist")
    rng = random.Random(0)
    assert s.draw("a1", "hard", rng)[1] == "shuffled"
    assert s.draw("a1", "hard_backoff", rng)[1] == "hard_copyright"
    assert s.draw("a1", "shuffled", rng)[1] in ("shuffled", "self")
    assert s.draw("a1", "jaccard", rng)[1] in ("shuffled", "self")


def test_tag_jaccard(index_path):
    s = IdentityPairSampler(index_path, min_level="artist")
    # identical tag sets (frieren/sousou/@x)
    assert s.tag_jaccard("a1", "a2") == pytest.approx(1.0)
    # disjoint identities + artists
    assert s.tag_jaccard("a1", "c1") == pytest.approx(0.0)
    # a3 {frieren, sousou, @y} vs b1 {fern, sousou, @y}: ∩=2, ∪=4 → 0.5
    assert s.tag_jaccard("a3", "b1") == pytest.approx(0.5)
