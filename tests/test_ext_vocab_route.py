"""Symbol routing for the CJK ext vocab (2026-09-03).

The rule deciding which characters leave the T5 spiece path ships inside the
pack json (``mapping["route"]``); a pack without it must behave exactly as the
legacy CJK-only encoder, and symbol rows are a strict *append* after the CJK
blocks so no existing row id, cache or trained pack moves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from library.anima import ext_vocab as ev

REPO = Path(__file__).resolve().parents[1]
EXT_PREFIX = REPO / "bench" / "cjk_adapter" / "assets" / "ext_embed"


# ---------------------------------------------------------------------------
# Pure-Python contract (no tokenizers, no assets)
# ---------------------------------------------------------------------------


def test_default_route_is_the_legacy_predicate():
    route = ev.Route.default()
    for o in range(0, 0x10000, 3):
        ch = chr(o)
        assert route(ch) == ev.is_cjk_char(ch)
    assert not route("^") and not route("~") and not route("·")


def test_route_round_trips_through_the_pack_json():
    route = ev.Route(chars=frozenset("^<~·×☆"))
    back = ev.Route.from_mapping({"route": route.to_json()})
    assert back == route
    assert back("^") and back("☆") and back("黒") and not back("a")
    # No `route` in the mapping → the legacy rule, not an error.
    assert ev.Route.from_mapping({"qwen": {}, "char": {}}) == ev.Route.default()
    assert ev.Route.from_mapping(None) == ev.Route.default()


def test_segment_runs_honours_the_route():
    text = "1girl, ^^^, 黒髪 :<"
    legacy = ev.segment_runs(text)
    assert legacy == [("t5", "1girl, ^^^, "), ("cjk", "黒髪"), ("t5", " :<")]
    routed = ev.segment_runs(text, ev.Route(chars=frozenset("^<")))
    assert routed == [
        ("t5", "1girl, "),
        ("cjk", "^^^"),
        ("t5", ", "),
        ("cjk", "黒髪"),
        ("t5", " :"),
        ("cjk", "<"),
    ]


def test_encoder_routes_uses_the_pack_rule_and_defaults_to_cjk():
    enc_legacy = ev.HybridT5Encoder(
        t5_tok=None, qwen_tok=None, qwen_map={}, char_map={}
    )
    enc_sym = ev.HybridT5Encoder(
        t5_tok=None,
        qwen_tok=None,
        qwen_map={},
        char_map={},
        route=ev.Route(chars=frozenset("^")),
    )
    assert not enc_legacy.routes("^^^") and enc_legacy.routes("黒")
    assert enc_sym.routes("^^^") and enc_sym.routes("黒") and not enc_sym.routes("abc")


# ---------------------------------------------------------------------------
# Against the real tokenizers + asset (skipped when not present)
# ---------------------------------------------------------------------------


def _tok_and_mapping():
    if not EXT_PREFIX.with_suffix(".json").exists():
        pytest.skip("ext_embed assets not built (bench/cjk_adapter/build_ext.py)")
    from anima_lora import default_checkpoints
    from library.anima import strategy as strategy_anima

    ckpt = default_checkpoints()
    if not Path(ckpt.text_encoder).exists():
        pytest.skip("Qwen3 text encoder not downloaded")
    tok = strategy_anima.AnimaTokenizeStrategy(qwen3_path=ckpt.text_encoder)
    _, mapping = ev.load_ext_assets(EXT_PREFIX)
    return tok, mapping


def test_symbol_route_chars_are_exactly_the_t5_gaps():
    tok, _ = _tok_and_mapping()
    chars = ev.symbol_route_chars(tok.t5_tokenizer, tok.qwen3_tokenizer)
    for ch in "^<~·×☆\\♪":
        assert ch in chars, ch
    # Letters and punctuation spiece can spell never route.
    for ch in "abcXYZ019,.:()'é":
        assert ch not in chars, ch
    for ch in chars:
        assert not ev.is_cjk_char(ch)
        assert (
            ev.T5_UNK_ID in tok.t5_tokenizer(ch, add_special_tokens=False)["input_ids"]
        )


def test_symbol_block_is_a_strict_append_after_the_cjk_blocks():
    _, mapping = _tok_and_mapping()
    if "sym" not in mapping:
        pytest.skip("asset built without the symbol block (--no-symbols)")
    n_cjk = len(mapping["qwen"]) + len(mapping["char"])
    assert max(mapping["qwen"].values()) < len(mapping["qwen"])
    assert max(mapping["char"].values()) < n_cjk
    sym_rows = list(mapping["sym"].values()) + list(mapping["sym_char"].values())
    assert min(sym_rows) == n_cjk
    assert max(sym_rows) == mapping["rows"] - 1
    assert mapping["sym_rows"] == [n_cjk, mapping["rows"]]
    assert set(mapping["route"]["chars"]) >= set("^<~·×☆")


def test_symbols_encode_to_rows_and_en_stays_bit_identical():
    tok, mapping = _tok_and_mapping()
    if "sym" not in mapping:
        pytest.skip("asset built without the symbol block (--no-symbols)")
    enc = ev.HybridT5Encoder.from_mapping(
        tok.t5_tokenizer, tok.qwen3_tokenizer, mapping
    )
    n_cjk = len(mapping["qwen"]) + len(mapping["char"])
    for text in ["1girl, ^^^, smile", ":<, お尻", "萊莎琳·斯托特", "(˘ω˘)", "🎉"]:
        ids, mask = enc.encode(text, 64)
        live = [i for i, m in zip(ids, mask) if m]
        assert ev.T5_UNK_ID not in live, text
        assert any(i >= ev.T5_TABLE_SIZE + n_cjk for i in live), text

    en = "1girl, black hair, school uniform, classroom, anime style"
    stock = tok.t5_tokenizer(en, padding="max_length", max_length=512, truncation=True)
    assert enc.encode(en, 512)[0] == stock["input_ids"]

    # A pack without `route` (the old JSON shape) keeps `^^^` on the spiece path.
    legacy_map = {
        k: v for k, v in mapping.items() if k not in ("route", "sym", "sym_char")
    }
    legacy = ev.HybridT5Encoder.from_mapping(
        tok.t5_tokenizer, tok.qwen3_tokenizer, legacy_map
    )
    ids, mask = legacy.encode("1girl, ^^^", 64)
    assert ev.T5_UNK_ID in [i for i, m in zip(ids, mask) if m]
    # …and CJK ids are the same under both packs (symbol rows never re-index).
    ja = "1girl, 黒髪, holding a sign reading 「おはよう」"
    assert enc.encode(ja, 512) == legacy.encode(ja, 512)
