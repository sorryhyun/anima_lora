"""Quote partition for the CJK ext vocab (DiT line D1, 2026-09-05).

A pack may carry a content-free isotropic block regenerated from a seed
(``mapping["iso"]``) plus a quote rule (``route.quotes``); routed spans
inside ``「…」`` / ``『…』`` / ``"…"`` land on that block, bare CJK keeps the
trained rows, the delimiters stay on their old path, and a pack without the
partition encodes bit-identically to before. ``pack_digest`` is what a LoRA
trained through the pack stamps (``ss_ext_pack_sha``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from library.anima import ext_vocab as ev

REPO = Path(__file__).resolve().parents[1]
EXT_PREFIX = REPO / "bench" / "cjk_adapter" / "assets" / "ext_embed"


# ---------------------------------------------------------------------------
# Pure-Python contract (no tokenizers, no assets)
# ---------------------------------------------------------------------------


def test_quote_spans_three_spellings_and_stray_openers():
    route = ev.Route(quotes=ev.DEFAULT_QUOTES)
    for text in ["a, 「大丈夫」, b", "a, 『大丈夫』, b", 'a, "大丈夫", b']:
        (span,) = route.quote_spans(text)
        assert text[span[0] : span[1]] == "大丈夫"
    # Left to right, non-nesting, stray opener inert, empty pair skipped.
    text = '「x」 "y, z" 『』 「open "w"'
    assert [text[a:b] for a, b in route.quote_spans(text)] == ["x", "y, z", "w"]
    assert ev.Route.default().quote_spans("「x」") == []


def test_route_json_round_trips_quotes_and_stays_backward_compatible():
    route = ev.Route(chars=frozenset("^"), quotes=ev.DEFAULT_QUOTES)
    js = route.to_json()
    assert js["quotes"] == [["「", "」"], ["『", "』"], ['"', '"']]
    assert ev.Route.from_mapping({"route": js}) == route
    # A pre-partition json has no ``quotes`` key at all and reads back empty.
    assert "quotes" not in ev.Route(chars=frozenset("^")).to_json()
    assert ev.Route.from_mapping({"route": {"chars": "^"}}).quotes == ()


def test_iso_block_is_byte_deterministic_and_at_norm():
    a = ev.iso_block(7, 300, 64, 212.0)
    b = ev.iso_block(7, 300, 64, 212.0, chunk=7)
    assert a.dtype == torch.float32 and a.shape == (300, 64)
    assert a.numpy().tobytes() == b.numpy().tobytes()
    assert torch.allclose(a.norm(dim=1), torch.full((300,), 212.0), atol=1e-3)
    assert not torch.equal(a, ev.iso_block(8, 300, 64, 212.0))
    # Near-orthogonal rows (content-free): |cos| small.
    an = a / a.norm(dim=1, keepdim=True)
    off = (an @ an.T - torch.eye(300)).abs().max()
    assert off < 0.6


def test_iso_spec_round_trip_and_materialize():
    spec = ev.IsoSpec(seed=3, n_rows=10, dim=16, norm=5.0, start=10)
    mapping = {"rows": 20, "iso": spec.to_json()}
    assert ev.IsoSpec.from_mapping(mapping) == spec
    assert ev.IsoSpec.from_mapping({"rows": 10}) is None
    trained = torch.randn(10, 16)
    full = ev.materialize_iso(trained, mapping)
    assert full.shape == (20, 16)
    assert torch.equal(full[:10], trained)
    assert torch.equal(full[10:], spec.build())
    # Already materialised → untouched; wrong size → loud.
    assert ev.materialize_iso(full, mapping) is full
    with pytest.raises(ValueError, match="mismatch"):
        ev.materialize_iso(torch.randn(11, 16), mapping)


def test_pack_digest_is_stable_and_regeneration_invariant():
    spec = ev.IsoSpec(seed=0, n_rows=4, dim=8, norm=1.0, start=4)
    trained = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    mapping = {
        "qwen": {"5": 0},
        "char": {"猫": 1},
        "rows": 8,
        "iso": spec.to_json(),
        "route": ev.Route(quotes=ev.DEFAULT_QUOTES).to_json(),
        "training": {"note": "provenance never changes the digest"},
    }
    seed_only = ev.pack_digest(trained, mapping)
    shipped = ev.pack_digest(ev.materialize_iso(trained, mapping), mapping)
    assert seed_only == shipped and len(seed_only) == 64
    assert ev.pack_digest(trained, {**mapping, "training": {}}) == seed_only
    # Rows, ids or the quote rule change → a different pack.
    assert ev.pack_digest(trained + 1, mapping) != seed_only
    assert ev.pack_digest(trained, {**mapping, "char": {"犬": 1}}) != seed_only
    no_quotes = {**mapping, "route": ev.Route().to_json()}
    assert ev.pack_digest(trained, no_quotes) != seed_only


def test_quote_spans_need_both_halves_of_the_partition_and_cut_runs():
    base = dict(t5_tok=None, qwen_tok=None, qwen_map={}, char_map={})
    text = 'a 「b」 "c"'
    off = ev.HybridT5Encoder(**base, route=ev.Route(quotes=ev.DEFAULT_QUOTES))
    assert not off.quote_routing and off.quote_spans(text) == []
    assert not ev.HybridT5Encoder(**base, iso_offset=10).quote_routing
    on = ev.HybridT5Encoder(
        **base, route=ev.Route(quotes=ev.DEFAULT_QUOTES), iso_offset=10
    )
    assert on.quote_routing and on.quote_spans(text) == [(3, 4), (7, 8)]
    # A routed run 「b」 at text offset 2 is cut into delimiter / content /
    # delimiter; a run with no quote overlap stays whole.
    assert on._cut_run("「b」", 2, on.quote_spans(text)) == [
        ("「", False, 0),
        ("b", True, 1),
        ("」", False, 2),
    ]
    assert on._cut_run("xyz", 20, on.quote_spans(text)) == [("xyz", False, 0)]


# ---------------------------------------------------------------------------
# Against the real tokenizers + asset json (skipped when not present)
# ---------------------------------------------------------------------------


def _tok_and_mappings():
    if not EXT_PREFIX.with_suffix(".json").exists():
        pytest.skip("ext_embed assets not built (bench/cjk_adapter/build_ext.py)")
    from anima_lora import default_checkpoints
    from library.anima import strategy as strategy_anima

    ckpt = default_checkpoints()
    if not Path(ckpt.text_encoder).exists():
        pytest.skip("Qwen3 text encoder not downloaded")
    tok = strategy_anima.AnimaTokenizeStrategy(qwen3_path=ckpt.text_encoder)
    plain = json.loads(EXT_PREFIX.with_suffix(".json").read_text(encoding="utf-8"))
    plain.pop("iso", None)
    n = int(plain["rows"])
    route = ev.Route.from_mapping(plain)
    part = {
        **plain,
        "rows": 2 * n,
        "iso": ev.IsoSpec(seed=0, n_rows=n, dim=1024, norm=212.0, start=n).to_json(),
        "route": ev.Route(
            ranges=route.ranges, chars=route.chars, quotes=ev.DEFAULT_QUOTES
        ).to_json(),
    }
    return tok, plain, part, n


def _live(enc, text):
    ids, mask = enc.encode(text, 128)
    return [i for i, m in zip(ids, mask) if m]


def test_partition_keeps_en_bit_exact_and_bare_cjk_on_trained_rows():
    tok, plain, part, n = _tok_and_mappings()
    t5, qw = tok.t5_tokenizer, tok.qwen3_tokenizer
    legacy = ev.HybridT5Encoder.from_mapping(t5, qw, plain)
    enc = ev.HybridT5Encoder.from_mapping(t5, qw, part)
    assert enc.quote_routing and enc.iso_offset == n
    en = '1girl, english text, "Are you okay? I\'m fine, really.", smile'
    stock = t5(en, padding="max_length", max_length=512, truncation=True)
    assert enc.encode(en, 512)[0] == stock["input_ids"]
    # Bare CJK / symbols: identical to the pre-partition encoder.
    for text in ["1girl, 黒髪, 猫耳, ^^^", "萊莎琳·斯托特", "レイム, smile"]:
        assert enc.encode(text, 512) == legacy.encode(text, 512), text


def test_quoted_content_touches_only_the_isotropic_block():
    tok, plain, part, n = _tok_and_mappings()
    t5, qw = tok.t5_tokenizer, tok.qwen3_tokenizer
    legacy = ev.HybridT5Encoder.from_mapping(t5, qw, plain)
    enc = ev.HybridT5Encoder.from_mapping(t5, qw, part)
    lo, hi = ev.T5_TABLE_SIZE, ev.T5_TABLE_SIZE + n
    cap = "japanese text, 黒髪, 「大丈夫、本当に」, smile"
    live = _live(enc, cap)
    trained = [i for i in live if lo <= i < hi]
    iso = [i for i in live if i >= hi]
    assert iso, "quoted content produced no isotropic rows"
    # The delimiters 「」 are CJK punctuation → trained rows; so is 黒髪.
    delim = {i - lo for i in _live(legacy, "「」") if i >= lo}
    assert delim and delim <= {i - lo for i in trained}
    kurokami = {i - lo for i in _live(legacy, "黒髪") if i >= lo}
    assert kurokami <= {i - lo for i in trained}
    # Everything the legacy encoder put on ext rows for the *content* is in
    # the mirror at +n, and nothing else of the content is on trained rows.
    content_legacy = [i - lo for i in _live(legacy, "大丈夫、本当に") if i >= lo]
    assert sorted(i - hi for i in iso) == sorted(content_legacy)
    assert not (set(i - lo for i in trained) - delim - kurokami)
    # Without the iso record the same json encodes exactly as legacy.
    assert ev.HybridT5Encoder.from_mapping(
        t5, qw, {k: v for k, v in part.items() if k != "iso"}
    ).encode(cap, 512) == legacy.encode(cap, 512)


def test_three_spellings_of_a_line_give_the_same_content_ids():
    tok, plain, part, n = _tok_and_mappings()
    enc = ev.HybridT5Encoder.from_mapping(tok.t5_tokenizer, tok.qwen3_tokenizer, part)
    hi = ev.T5_TABLE_SIZE + n
    got = []
    for text in [
        "japanese text, 「大丈夫、本当に」, smile",
        "japanese text, 『大丈夫、本当に』, smile",
        'japanese text, "大丈夫、本当に", smile',
    ]:
        got.append([i for i in _live(enc, text) if i >= hi])
    assert got[0] == got[1] == got[2] and got[0]
    # The order-format phrase (several ASCII-quoted lines) routes each line.
    order = 'Japanese text in following order: "大丈夫", "本当に"'
    assert [i for i in _live(enc, order) if i >= hi]
    # Symbol-tail chars inside a quote ride the mirror too; ASCII letters
    # inside a quote stay on spiece.
    assert [i for i in _live(enc, '"^^^"') if i >= hi]
    en_line = '"Are you okay?"'
    assert all(i < ev.T5_TABLE_SIZE for i in _live(enc, en_line))


# ---------------------------------------------------------------------------
# Caption grammar (anime_tools) — a quoted line is one tag
# ---------------------------------------------------------------------------


def test_caption_grammar_keeps_a_quoted_line_whole():
    from anime_tools.captions import position_clauses as pc

    if not hasattr(pc, "quoted_spans"):
        pytest.skip("installed anime_tools predates the quote-aware grammar")
    cap = 'safe, 1girl, english text, "Are you okay? I\'m fine, really.", speech bubble'
    parsed = pc.parse_caption(cap)
    assert parsed.flat_tags[3] == '"Are you okay? I\'m fine, really."'
    assert pc.compose_caption(parsed.flat_tags, parsed.clauses) == cap
    assert pc.parse_caption("a, 「大丈夫、本当に」, b").flat_tags == (
        "a",
        "「大丈夫、本当に」",
        "b",
    )


# ---------------------------------------------------------------------------
# LoRA stamp: the inference loader says when a pack-trained LoRA meets none
# ---------------------------------------------------------------------------


def test_inference_loader_warns_on_a_pack_stamped_lora(tmp_path, caplog):
    import logging

    from safetensors.torch import save_file

    from library.inference import models as inf_models

    stamped = tmp_path / "stamped.safetensors"
    save_file(
        {"lora_unet_x.weight": torch.zeros(1)},
        str(stamped),
        metadata={"ss_ext_pack": "pack_a", "ss_ext_pack_sha": "ab" * 32},
    )
    plain = tmp_path / "plain.safetensors"
    save_file({"lora_unet_x.weight": torch.zeros(1)}, str(plain), metadata={})
    with caplog.at_level(logging.WARNING, logger=inf_models.logger.name):
        inf_models._warn_ext_pack_stamp(str(plain))
        assert not caplog.records
        inf_models._warn_ext_pack_stamp(str(stamped))
    assert any("pack_a" in r.getMessage() for r in caplog.records)
