"""Caption-shuffle boundary tests.

Pins down the three corner cases the inline ``tag.startswith("@")`` predicate
got wrong:

1. ``@ @`` (booru ``@_@`` eye-shape, space-form) must not trigger the artist
   boundary.
2. Multi-artist captions (``@artist1, @artist2, …``) must protect the full
   leading handle run.
3. The ``@no-artist`` sentinel must participate in the boundary but be
   stripped from every cache variant (including v0).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library.anima.training import (  # noqa: E402
    NO_ARTIST_SENTINEL,
    anima_smart_shuffle_caption,
    find_anima_prefix_end,
    strip_no_artist_sentinel,
)
from anime_tools.captions.taxonomy import is_artist_tag as _is_artist_tag  # noqa: E402


# ----- predicate ----------------------------------------------------------


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("@sincos", True),
        ("@sumiyao (amam)", True),
        ("@no-artist", True),
        ("@", False),  # one char, no handle body
        ("@ @", False),  # booru @_@ eye-shape, space-form
        ("@ ", False),  # trailing space → not artist
        ("blue hair", False),
        ("1girl", False),
        ("", False),
    ],
)
def test_is_artist_tag(tag, expected):
    assert _is_artist_tag(tag) is expected


# ----- prefix-end walk ----------------------------------------------------


def test_prefix_end_single_artist_first():
    assert find_anima_prefix_end(["@sincos", "blue hair", "1girl"]) == 1


def test_prefix_end_no_artist():
    # The case @no-artist exists to fix: zero protection without a sentinel.
    assert find_anima_prefix_end(["blue hair", "1girl"]) == 0


def test_prefix_end_multi_artist_collab():
    assert find_anima_prefix_end(["@artist1", "@artist2", "@artist3", "blue hair"]) == 3


def test_prefix_end_leading_content_then_artist():
    # Old behavior preserved: leading non-@ tags extend into the prefix.
    assert find_anima_prefix_end(["solo", "1girl", "@sincos", "blue hair"]) == 3


def test_prefix_end_eye_shape_no_artist():
    # @ @ alone must NOT trigger the boundary.
    assert find_anima_prefix_end(["solo", "@ @", "blue hair"]) == 0


def test_prefix_end_eye_shape_before_real_artist():
    # @ @ falls through; @sincos is the real boundary, and @ @ rides along
    # in the prefix as a leading-content tag (same as any other non-@ tag).
    assert find_anima_prefix_end(["@ @", "solo", "@sincos", "blue hair"]) == 3


def test_prefix_end_sentinel_acts_as_artist():
    assert find_anima_prefix_end([NO_ARTIST_SENTINEL, "blue hair"]) == 1


# ----- strip helper -------------------------------------------------------


def test_strip_no_artist_sentinel_removes_all_occurrences():
    tags = ["a", NO_ARTIST_SENTINEL, "b", NO_ARTIST_SENTINEL, "c"]
    assert strip_no_artist_sentinel(tags) == ["a", "b", "c"]


def test_strip_no_artist_sentinel_no_op_when_absent():
    tags = ["@sincos", "blue hair"]
    assert strip_no_artist_sentinel(tags) == tags


# ----- shuffle integration -----------------------------------------------


def test_shuffle_preserves_prefix_order_with_multi_artist():
    random.seed(0)
    tags = ["@artist1", "@artist2", "@artist3", "a", "b", "c", "d"]
    out = anima_smart_shuffle_caption(tags.copy())
    # Prefix run preserved in order.
    assert out[:3] == ["@artist1", "@artist2", "@artist3"]
    # Suffix has same multiset, possibly reordered.
    assert sorted(out[3:]) == sorted(["a", "b", "c", "d"])


def test_shuffle_eye_shape_does_not_anchor_prefix():
    random.seed(0)
    tags = ["solo", "@ @", "blue hair", "red eyes"]
    out = anima_smart_shuffle_caption(tags.copy())
    # No real artist → split_idx=0 → everything is shuffleable.
    # We can't assert a specific order (random), only that the multiset is
    # preserved and the input wasn't accidentally locked into the prefix.
    assert sorted(out) == sorted(tags)


def test_shuffle_keeps_sentinel_in_output_for_caller_strip():
    # Contract: shuffle does NOT strip the sentinel (so split_idx stays
    # meaningful for the caller's dropout protection). The caller must strip
    # before tokenization.
    random.seed(0)
    tags = [NO_ARTIST_SENTINEL, "blue hair", "1girl"]
    out = anima_smart_shuffle_caption(tags.copy())
    assert NO_ARTIST_SENTINEL in out
    assert out[0] == NO_ARTIST_SENTINEL  # prefix order preserved


# ----- variant generator (TE cache path) ---------------------------------


def _gen_variants(*args, **kwargs):
    from library.preprocess import generate_caption_variants

    return generate_caption_variants(*args, **kwargs)


def test_variants_strip_sentinel_from_v0():
    random.seed(0)
    out = _gen_variants(
        f"{NO_ARTIST_SENTINEL}, blue hair, 1girl", num_variants=3, tag_dropout_rate=0.0
    )
    assert all(NO_ARTIST_SENTINEL not in v for v in out)
    # v0 retains the original tag order (sentinel removed).
    assert out[0] == "blue hair, 1girl"


def test_variants_v0_byte_identical_when_no_sentinel():
    # Existing datasets must not see whitespace renormalization in v0.
    raw = "@sincos,blue hair  ,1girl"
    out = _gen_variants(raw, num_variants=1, tag_dropout_rate=0.0)
    assert out[0] == raw


def test_direct_te_caption_strips_sentinel_without_normalizing_clean_caption():
    from library.preprocess.text import _strip_no_artist_sentinel_from_caption

    assert (
        _strip_no_artist_sentinel_from_caption(
            f"{NO_ARTIST_SENTINEL}, blue hair, 1girl"
        )
        == "blue hair, 1girl"
    )
    raw = "@sincos,blue hair  ,1girl"
    assert _strip_no_artist_sentinel_from_caption(raw) == raw


def test_variants_strip_sentinel_after_dropout():
    random.seed(0)
    # High dropout rate to exercise the kept-list path.
    out = _gen_variants(
        f"{NO_ARTIST_SENTINEL}, a, b, c, d, e",
        num_variants=8,
        tag_dropout_rate=0.5,
    )
    for v in out:
        assert NO_ARTIST_SENTINEL not in v


def test_variants_multi_artist_protected_from_dropout():
    random.seed(0)
    # Force every dropable tag to roll the dice; with rate=1.0 every
    # non-prefix tag is dropped. All three artist handles must survive.
    out = _gen_variants(
        "@artist1, @artist2, @artist3, a, b, c",
        num_variants=4,
        tag_dropout_rate=1.0,
    )
    # v0 untouched.
    assert out[0] == "@artist1, @artist2, @artist3, a, b, c"
    for v in out[1:]:
        toks = [t.strip() for t in v.split(",")]
        assert "@artist1" in toks
        assert "@artist2" in toks
        assert "@artist3" in toks


# ----- identity randomization (lexinvariant tag regularization) -----------


# Stub erasure pool — stands in for build_erasure_token_pool()'s dual-single words.
_POOL = ["swing", "sodium", "awards", "covering", "largest", "album"]


def test_randomize_keeps_v0_pristine():
    random.seed(0)
    raw = "@sincos, 1girl, blue hair"
    out = _gen_variants(
        raw,
        num_variants=4,
        tag_dropout_rate=0.0,
        tag_randomize_rate=1.0,
        erasure_pool=_POOL,
    )
    # v0 is never randomized.
    assert out[0] == raw


def test_randomize_preserves_slot_count_and_erases_identity():
    random.seed(0)
    # rate=1.0 with no dropout: every tag slot survives but every post-prefix
    # identity is replaced — the defining property vs dropout (removes slots).
    raw = "@artist, 1girl, blue hair, smile"
    out = _gen_variants(
        raw,
        num_variants=8,
        tag_dropout_rate=0.0,
        tag_randomize_rate=1.0,
        erasure_pool=_POOL,
    )
    post_prefix = {"1girl", "blue hair", "smile"}
    for v in out[1:]:
        toks = [t.strip() for t in v.split(",")]
        assert len(toks) == 4  # slot count preserved (nothing dropped)
        # The @artist prefix (the trigger) is now protected, not erased.
        assert "@artist" in toks
        # Every post-prefix identity is erased — replaced by a pool token.
        assert not (set(toks) & post_prefix)
        erased = [t for t in toks if t != "@artist"]
        assert all(t in _POOL for t in erased)


def test_randomize_respects_protect_fn_and_sentinel():
    random.seed(0)
    out = _gen_variants(
        f"{NO_ARTIST_SENTINEL}, keepme, a, b",
        num_variants=8,
        tag_dropout_rate=0.0,
        tag_randomize_rate=1.0,
        protect_fn=lambda t: t == "keepme",
        erasure_pool=_POOL,
    )
    for v in out[1:]:
        toks = [t.strip() for t in v.split(",")]
        assert NO_ARTIST_SENTINEL not in toks  # sentinel stripped, never emitted
        assert "keepme" in toks  # protected tag kept verbatim


def test_randomize_without_pool_raises():
    # No random-ASCII fallback: randomizing without a pool is a hard error.
    import pytest

    with pytest.raises(ValueError, match="erasure_pool"):
        _gen_variants(
            "@artist, 1girl, blue hair",
            num_variants=4,
            tag_dropout_rate=0.0,
            tag_randomize_rate=1.0,
        )


def test_build_erasure_token_pool_dual_single_and_exclude():
    from library.preprocess.text import build_erasure_token_pool

    class _Qwen:
        # Leading-space ("Ġ") lowercase-ascii alpha words are the Qwen3-single
        # candidates; everything else is filtered before T5 is consulted.
        def get_vocab(self):
            return {
                "Ġswing": 500,  # dual-single, kept
                "Ġsodium": 501,  # dual-single, kept
                "Ġsword": 502,  # dual-single but a REAL tag → excluded
                "Ġcat": 503,  # too short (<4) → dropped
                "ĠSwing": 504,  # not lowercase → dropped
                "swing": 505,  # no leading space → dropped
                "Ġxyzzy": 506,  # T5 fragments it → dropped
            }

    class _T5:
        # Simulate sentencepiece: ", " is one token; a word is "single" iff in
        # T5_SINGLE, else it fragments into 2.
        T5_SINGLE = {"swing", "sodium", "sword"}

        def __call__(self, text, add_special_tokens=False):
            if text == ", ":
                ids = [1]
            else:
                word = text[2:]  # strip ", "
                ids = [1, 9] if word in self.T5_SINGLE else [1, 9, 9]
            return {"input_ids": ids}

    pool = build_erasure_token_pool(_Qwen(), _T5(), exclude={"sword"})
    assert sorted(pool) == ["sodium", "swing"]  # xyzzy (T5-frag) + sword (tag) gone
    # Missing tokenizer API → empty (caller treats as hard error upstream).
    assert build_erasure_token_pool(object(), _T5()) == []


# ----- r-family loader contract (use_randomized_caption_variants) ---------


def _write_variant_cache(path, *, n_variants, n_randomized):
    """Synthetic TE cache mirroring the writer's key layout. Each variant's
    prompt_embeds is a constant tile encoding its own name so the draw is
    identifiable."""
    import torch
    from safetensors.torch import save_file

    names = [f"v{i}" for i in range(n_variants)] + [
        f"r{j}" for j in range(1, n_randomized + 1)
    ]

    def emb(name):
        return torch.full((2, 4), float(abs(hash(name)) % 1000))

    sd = {
        "num_variants": torch.tensor(n_variants),
        "v0_intact": torch.tensor(1, dtype=torch.int8),
        "caption_dropout_rate": torch.tensor(0.0),
    }
    if n_randomized:
        sd["num_randomized"] = torch.tensor(n_randomized)
    for name in names:
        sd[f"prompt_embeds_{name}"] = emb(name)
        sd[f"attn_mask_{name}"] = torch.ones(2, dtype=torch.int32)
        sd[f"t5_input_ids_{name}"] = torch.zeros(2, dtype=torch.long)
        sd[f"t5_attn_mask_{name}"] = torch.ones(2, dtype=torch.int32)
    save_file(sd, str(path))
    return {name: emb(name) for name in names}


def _draw_names(strat, path, embs, n=3000):
    import torch

    random.seed(0)
    seen = set()
    for _ in range(n):
        out = strat.load_outputs_npz(str(path))[0]
        for name, e in embs.items():
            if torch.equal(out, e):
                seen.add(name)
                break
    return seen


def test_loader_randomized_draws_r_family_and_v0(tmp_path):
    from library.anima.strategy import AnimaTextEncoderOutputsCachingStrategy

    p = tmp_path / "x_anima_te.safetensors"
    embs = _write_variant_cache(p, n_variants=4, n_randomized=3)
    strat = AnimaTextEncoderOutputsCachingStrategy(
        True, 1, True, use_randomized_caption_variants=True
    )
    drawn = _draw_names(strat, p, embs)
    # v0 (pristine anchor) + r-family only — never the shuffled v1..v3.
    assert drawn == {"v0", "r1", "r2", "r3"}


def test_loader_randomized_only_excludes_v0(tmp_path):
    from library.anima.strategy import AnimaTextEncoderOutputsCachingStrategy

    p = tmp_path / "x_anima_te.safetensors"
    embs = _write_variant_cache(p, n_variants=4, n_randomized=3)
    strat = AnimaTextEncoderOutputsCachingStrategy(
        True, 1, True, use_randomized_caption_variants_only=True
    )
    drawn = _draw_names(strat, p, embs)
    assert drawn == {"r1", "r2", "r3"}


def test_loader_randomized_falls_back_to_v_family_without_r(tmp_path):
    from library.anima.strategy import AnimaTextEncoderOutputsCachingStrategy

    p = tmp_path / "x_anima_te.safetensors"
    embs = _write_variant_cache(p, n_variants=3, n_randomized=0)
    strat = AnimaTextEncoderOutputsCachingStrategy(
        True, 1, True, use_randomized_caption_variants=True
    )
    drawn = _draw_names(strat, p, embs)
    # No r-family on disk → graceful fallback to the shuffled v-family.
    assert drawn == {"v0", "v1", "v2"}
