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
EXT_PREFIX = REPO / "_archive" / "bench" / "cjk_adapter" / "assets" / "ext_embed"


# ---------------------------------------------------------------------------
# Pure-Python contract (no tokenizers, no assets)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Against the real tokenizers + asset (skipped when not present)
# ---------------------------------------------------------------------------


def _tok_and_mapping():
    if not EXT_PREFIX.with_suffix(".json").exists():
        pytest.skip("ext_embed assets not present (_archive/bench/cjk_adapter/assets)")
    from anima_lora import default_checkpoints
    from library.anima import strategy as strategy_anima

    ckpt = default_checkpoints()
    if not Path(ckpt.text_encoder).exists():
        pytest.skip("Qwen3 text encoder not downloaded")
    tok = strategy_anima.AnimaTokenizeStrategy(qwen3_path=ckpt.text_encoder)
    _, mapping = ev.load_ext_assets(EXT_PREFIX)
    return tok, mapping


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
