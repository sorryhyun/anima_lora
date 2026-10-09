"""Per-glyph routing (``mapping["glyph_route"]`` / ``ANIMA_VOCAB_GLYPH_ROUTE``).

Invariants (against the real tokenizers + pack, skipped when absent):

* **Off = bit-identical**: a pack without ``glyph_route`` (or ``False``)
  encodes exactly as before, EN and CJK alike.
* **On, a JA piece → its glyphs' single rows**: unspaced こんにちは encodes to
  the spelled ``こ ん に ち は``'s ids; a space-prefixed glyph (`` の``) to the
  glyph's own row, not its own token row.
* **EN, hangul and pure-punctuation tokens never split.**
* The mapping key reaches the digest only when set; the env var overrides
  the key through ``VocabPack.build_encoder``.
* **Hangul under its own key** (``glyph_route_ko`` /
  ``ANIMA_VOCAB_GLYPH_ROUTE_KO``): a Hangul piece → its syllables' single
  rows, space-prefixed forms included; JA and EN encode as without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from library.anima import ext_vocab as ev
from library.anima.ext_vocab import T5_TABLE_SIZE as T

from library.env import resolve_under_home

EXT_PREFIX = resolve_under_home("models/vocab_packs/anima_cjk_vocab_pack")


def _encs():
    if not EXT_PREFIX.with_suffix(".json").exists():
        pytest.skip("vocab pack not downloaded")
    from anima_lora import default_checkpoints
    from library.anima import strategy as strategy_anima

    ckpt = default_checkpoints()
    if not Path(ckpt.text_encoder).exists():
        pytest.skip("Qwen3 text encoder not downloaded")
    tok = strategy_anima.AnimaTokenizeStrategy(qwen3_path=ckpt.text_encoder)
    table, mapping = ev.load_ext_assets(EXT_PREFIX)
    t5, qw = tok.t5_tokenizer, tok.qwen3_tokenizer
    return (
        ev.HybridT5Encoder.from_mapping(t5, qw, mapping),
        ev.HybridT5Encoder.from_mapping(t5, qw, {**mapping, "glyph_route": True}),
        table,
        mapping,
    )


def _ext(enc, text):
    ids, mask = enc.encode(f'Japanese text reads as "{text}".', 128)
    return [i - T for i, m in zip(ids, mask) if m and i >= T]


def test_off_is_bit_identical():
    off, _, _, mapping = _encs()
    assert off.glyph_split is None
    for p in (
        "1girl, solo",
        'text reads as "こんにちは"',
        "漢字 ひらがな 한국어 中文。",
    ):
        explicit = ev.HybridT5Encoder.from_mapping(
            off.t5_tok, off.qwen_tok, {**mapping, "glyph_route": False}
        )
        assert off.encode(p) == explicit.encode(p)


def test_on_routes_pieces_to_single_rows():
    off, on, _, _ = _encs()
    for w in ("こんにちは", "ありがとう", "東京タワー"):
        assert _ext(on, w) == [_ext(off, c)[0] for c in w], w
    assert _ext(on, "こんにちは") == _ext(off, "こ ん に ち は")
    # " の" is its own Qwen token (its own row) when spelled; routed, it is の
    no = _ext(off, "の")
    assert _ext(on, "私の") == [*_ext(off, "私"), *no]
    assert _ext(on, "ありがとう の") == [*_ext(on, "ありがとう"), *no]


def test_on_leaves_en_hangul_and_punctuation():
    off, on, _, _ = _encs()
    for p in ("1girl, solo, smile", "안녕하세요", "……！？"):
        assert on.encode(p) == off.encode(p), p


def test_digest_and_env_override(monkeypatch):
    _, _, table, mapping = _encs()
    base = {k: v for k, v in mapping.items() if k != "glyph_route"}
    d0 = ev.pack_digest(table, base)
    assert ev.pack_digest(table, {**base, "glyph_route": False}) == d0
    assert ev.pack_digest(table, {**base, "glyph_route": True}) != d0

    from library.anima import vocab_pack as vp

    pack = vp.VocabPack(prefix=EXT_PREFIX, table=table, mapping=base, digest=d0)
    off, on, _, _ = _encs()
    monkeypatch.setenv("ANIMA_VOCAB_GLYPH_ROUTE", "1")
    assert pack.build_encoder(off.t5_tok, off.qwen_tok).glyph_split == on.glyph_split
    monkeypatch.setenv("ANIMA_VOCAB_GLYPH_ROUTE", "0")
    assert pack.build_encoder(off.t5_tok, off.qwen_tok).glyph_split is None


def test_ko_routes_hangul_pieces_only():
    off, on, table, mapping = _encs()
    ko = ev.HybridT5Encoder.from_mapping(
        off.t5_tok, off.qwen_tok, {**mapping, "glyph_route_ko": True}
    )
    both = ev.HybridT5Encoder.from_mapping(
        off.t5_tok,
        off.qwen_tok,
        {**mapping, "glyph_route": True, "glyph_route_ko": True},
    )
    # 하세요 is one Qwen piece with a row of its own
    assert len(_ext(off, "하세요")) == 1
    for w in ("하세요", "회사", "습니다"):
        assert _ext(ko, w) == [_ext(off, c)[0] for c in w], w
    singles = [_ext(off, c)[0] for c in "하세요"]
    assert _ext(ko, "안녕 하세요") == [*_ext(ko, "안녕"), *singles]
    # " 가" is its own Qwen token when spelled after a space; routed, it is 가
    assert _ext(ko, "그 가") == [*_ext(off, "그"), *_ext(off, "가")]
    for p in ("1girl, solo, smile", "こんにちは 漢字", "……！？"):
        assert ko.encode(p) == off.encode(p), p
        assert both.encode(p) == on.encode(p), p
    assert both.encode("안녕하세요") == ko.encode("안녕하세요")

    base = {k: v for k, v in mapping.items() if k != "glyph_route_ko"}
    d0 = ev.pack_digest(table, base)
    assert ev.pack_digest(table, {**base, "glyph_route_ko": False}) == d0
    assert ev.pack_digest(table, {**base, "glyph_route_ko": True}) != d0


def test_ko_env_override(monkeypatch):
    off, _, table, mapping = _encs()
    from library.anima import vocab_pack as vp

    pack = vp.VocabPack(prefix=EXT_PREFIX, table=table, mapping=mapping, digest="")
    monkeypatch.delenv("ANIMA_VOCAB_GLYPH_ROUTE", raising=False)
    monkeypatch.setenv("ANIMA_VOCAB_GLYPH_ROUTE_KO", "1")
    enc = pack.build_encoder(off.t5_tok, off.qwen_tok)
    assert _ext(enc, "하세요") == [_ext(off, c)[0] for c in "하세요"]
    assert _ext(enc, "こんにちは") == _ext(off, "こんにちは")
    monkeypatch.setenv("ANIMA_VOCAB_GLYPH_ROUTE_KO", "0")
    assert pack.build_encoder(off.t5_tok, off.qwen_tok).glyph_split is None
