"""OCR lines -> arm-C caption (project/cjk_aware_anima/datasets/cache_te_ext.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    path = ROOT / "project" / "cjk_aware_anima" / "datasets" / "cache_te_ext.py"
    spec = importlib.util.spec_from_file_location("cache_te_ext", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_order_format_is_one_phrase_in_record_order_and_round_trips():
    from anime_tools.captions.position_clauses import compose_caption, parse_caption

    m = _mod()
    cap = "1girl, smile. On the left, akita neru, yellow eyes."
    out = m.append_tags(cap, ["温水くん、私と", "あっ…うっ…"], "order")
    assert out == (
        '1girl, smile, Japanese text in following order: "温水くん、私と", '
        '"あっ…うっ…". On the left, akita neru, yellow eyes.'
    )
    # Grammar-safe: parse/compose leaves it byte-identical, the clause intact.
    p = parse_caption(out)
    assert compose_caption(p.flat_tags, p.clauses) == out
    assert p.clauses[0].tags == ("akita neru", "yellow eyes")
    # No separate presence tag: the phrase opens with it.
    assert "japanese text," not in out


def test_order_format_escapes_the_grammar_delimiters():
    # D1 rev: the grammar is quote-aware, so a comma inside the pair stays;
    # only an inner ASCII quote (which would close the pair) is rewritten.
    m = _mod()
    out = m.append_tags("1girl", ['say "hi", now'], "order")
    assert out == '1girl, Japanese text in following order: "say ”hi”, now"'


def test_tags_format_is_the_c2_shape():
    m = _mod()
    assert (
        m.append_tags("1girl, smile", ["あっ"], "tags")
        == "1girl, smile, japanese text, 「あっ」"
    )
    # An existing text tag suppresses the presence tag.
    assert (
        m.append_tags("1girl, speech bubble", ["あっ"], "tags")
        == "1girl, speech bubble, 「あっ」"
    )


def _sfx():
    path = ROOT / "project" / "cjk_aware_anima" / "datasets" / "ocr_sfx.py"
    spec = importlib.util.spec_from_file_location("ocr_sfx", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_line_kind_splits_speech_from_sfx():
    k = _sfx().line_kind
    # Onomatopoeia: repeated unit, voiced initial, lexicon onset, sokuon.
    for sfx in (
        "ぱんぱん",
        "ばるん",
        "じゃぽ",
        "ブルン",
        "びくっ",
        "ちら。",
        "カリカリ・…",
        "Kッ4vv",
        "きゅっ",
    ):
        assert k(sfx) == "sfx", sfx
    # A mouth: vowel / h-row initial, kanji, or long → speech.
    for sp in (
        "あっ…うっ…",
        "はぁ",
        "おおおん",
        "いいから",
        "さわって",
        "嘘だ～",
        "ちんこでけえ…",
        "もー特别だよ?",
        "たおy",
        "",
    ):
        assert k(sp) == "speech", sp


def test_sentence_format_is_one_speech_only_text_clause():
    m = _mod()
    out = m.append_tags(
        "1girl, smile",
        ["温水くん、私と", "ぱんぱん", "あっ…うっ…", "ばるん"],
        "sentence",
    )
    # SFX-classified lines are skipped for now (decision 2 amended); reading
    # order survives among the speech lines.
    assert out == '1girl, smile. Japanese text reads as "温水くん、私と", "あっ…うっ…".'
    assert m.append_tags("1girl", ["ぱんぱん"], "sentence") == "1girl"
    assert m.append_tags("1girl", [], "sentence") == "1girl"


def test_sentence_format_is_grammar_native_and_round_trips():
    from anime_tools.captions.position_clauses import (
        compose_caption,
        parse_caption,
    )

    m = _mod()
    cap = "1girl, smile. On the left, akita neru, yellow eyes."
    out = m.append_tags(cap, ['say "hi"', "温水くん、私と"], "sentence")
    assert out == (
        "1girl, smile. On the left, akita neru, yellow eyes. "
        'Japanese text reads as "say ”hi”", "温水くん、私と".'
    )
    assert ".." not in out
    # B2: the C10 caption re-parses to the same string, the position clause
    # intact and the text clause its own kind, last.
    p = parse_caption(out)
    assert compose_caption(p.flat_tags, p.clauses) == out
    assert p.clauses[0].tags == ("akita neru", "yellow eyes")
    assert p.clauses[1].is_text and p.clauses[1].tags == (
        '"say ”hi”"',
        '"温水くん、私と"',
    )
    # The shuffled-variants pass leaves the tail whole.
    from anime_tools.captions.variants import generate_caption_variants

    for v in generate_caption_variants(out, 6, 0.5, clause_dropout_rate=1.0)[1:]:
        assert v.endswith(' Japanese text reads as "say ”hi”", "温水くん、私と".')


def test_line_kind_balloon_veto():
    k = _sfx().line_kind
    assert k("カリカリ", in_bubble=True) == "speech"
    assert k("カリカリ", in_bubble=False) == "sfx"
    # Outside a balloon is not SFX by itself.
    assert k("さわって", in_bubble=False) == "speech"


# ---- hybrid records (project/cjk_aware_anima/datasets/build_ocr_records.py, B0)
def _hybrid():
    path = ROOT / "project" / "cjk_aware_anima" / "datasets" / "build_ocr_records.py"
    spec = importlib.util.spec_from_file_location("build_ocr_records", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_ocr_records"] = mod  # dataclasses resolve annotations here
    spec.loader.exec_module(mod)
    return mod


def test_parse_spotting_scales_the_1000_grid_to_page_pixels():
    m = _hybrid()
    raw = (
        "前驚<|LOC_443|><|LOC_34|><|LOC_467|><|LOC_34|><|LOC_467|><|LOC_83|><|LOC_443|><|LOC_83|>\n"
        "garbage row without locs\n"
        "請隨<|LOC_470|><|LOC_36|><|LOC_494|><|LOC_36|><|LOC_494|><|LOC_248|><|LOC_470|><|LOC_248|>"
    )
    lines = m.parse_spotting(raw, 896, 1184)
    assert [ln.text for ln in lines] == ["前驚", "請隨"]
    assert lines[1].box == (421, 42, 443, 294)


def test_runaway_guard_keeps_real_onomatopoeia():
    m = _hybrid()
    assert m.is_runaway("ぉ" * 100)
    assert m.is_runaway("ふくっ" * 30)
    assert not m.is_runaway("ぱんぱん")
    assert not m.is_runaway("おおおん")
    assert not m.is_runaway("温水くん、私と一緒にいるとやっぱりムラムラしちゃうんだ～")


def test_merge_matches_offset_thin_columns_and_keeps_vl_only_sfx():
    m = _hybrid()
    pp = [
        {
            "stem": "s",
            "text": "あっ…うっ…",
            "score": 0.86,
            "box": [162, 27, 187, 169],
            "engine": "ppocr_v6",
        },
        {
            "stem": "s",
            "text": "でくv",
            "score": 0.6,
            "box": [500, 500, 540, 600],
            "engine": "ppocr_v6",
        },
    ]
    spotting = [
        m.OcrLine(0, (175, 29, 195, 171), -1.0, "あっ…うっ…"),  # same line, IoU ~0.36
        m.OcrLine(1, (854, 602, 893, 719), -1.0, "ぬぎっ"),  # PP never boxed it
        m.OcrLine(2, (10, 10, 40, 40), -1.0, "・・"),  # symbol-only garbage
        m.OcrLine(3, (700, 10, 730, 400), -1.0, "ふくっ" * 40),  # runaway
    ]
    crop_reads = ["あっ…うっ…", "びくっ"]
    recs, stats = m.merge_page(
        "s",
        pp,
        spotting,
        crop_reads,
        iou_thr=0.3,
        contain_thr=0.5,
        weak_score=0.85,
        min_chars=3,
    )
    by_engine = {r.engine: r for r in recs}
    assert stats["pp_matched"] == 1 and stats["vl_only"] == 1
    assert [r.text for r in recs if r.engine == "vl16_spotting"] == ["ぬぎっ"]
    weak = by_engine["ppocr_v6+vl16_crop"]
    assert weak.text == "びくっ" and weak.pp_text == "でくv" and weak.rule1b == "weak"
    assert weak.kind == "sfx"
    # right-to-left: the SFX at x~854 reads before the balloon at x~162
    assert [r.text for r in recs][0] == "ぬぎっ"


def test_second_read_guard_rejects_runaway_and_overlong():
    m = _hybrid()
    assert m.accept_second_read("でくv", "びくっ", 3) == "びくっ"
    assert m.accept_second_read("でくv", "ぉ" * 50, 3) is None
    assert m.accept_second_read("でくv", "とても長い読みになってしまった", 3) is None
    assert m.accept_second_read("あっ…うっ…", "あっ…\nうっ…", 3) == "あっ… うっ…"


def test_chrome_is_a_kind_and_the_mirror_builder_drops_it(tmp_path):
    m = _hybrid()
    assert m.record_kind("ツイート") == "chrome"
    assert m.record_kind("22:34") == "chrome"
    assert m.record_kind("ぱんぱん") == "sfx"
    recs = tmp_path / "r.jsonl"
    recs.write_text(
        '{"stem": "a", "text": "ツイート", "kind": "chrome"}\n'
        '{"stem": "a", "text": "ぱんぱん", "kind": "sfx"}\n'
        '{"stem": "a", "text": "おはよう"}\n',
        encoding="utf-8",
    )
    # sfx is excluded from captions for now (decision 2026-09-05); chrome always
    assert _mod().ocr_lines_by_stem(recs, 8) == {"a": ["おはよう"]}
    assert _mod().ocr_lines_by_stem(recs, 8, drop_kinds=frozenset({"chrome"})) == {
        "a": ["ぱんぱん", "おはよう"]
    }


def test_vl_reads_drop_latex_and_keep_row_boundaries_as_spaces():
    m = _hybrid()
    assert m._normalize_read("身長: \\( 156 \\, cm \\)") == "身長: 156 cm"
    assert (
        m.accept_second_read(
            "椎名真昼ち人身長：156cm", "椎名真昼ちゃん\n身長：156cm", 3
        )
        == "椎名真昼ちゃん 身長：156cm"
    )


def test_second_read_never_loses_a_heart_and_weak_needs_corroboration():
    m = _hybrid()
    # PP had the heart, the crop read dropped it -> keep PP
    assert m.accept_second_read("かわいい♡", "かわいい", 3) is None
    # the reverse (PP dropped it) is accepted, and the emoji heart is normalised
    assert m.accept_second_read("かわいい", "かわいい❤️", 3) == "かわいい♥"
    # weak: two VL readings must agree
    assert (
        m.accept_second_read(
            "おいしそう", "おぃ～う", 3, corroborate="はぁ～レおいしそう"
        )
        is None
    )
    assert (
        m.accept_second_read(
            "借てきたよ", "借りてきたよ", 3, corroborate="借りてさたよ"
        )
        == "借りてきたよ"
    )


def test_merge_drops_full_page_quads_and_duplicate_vl_lines():
    m = _hybrid()
    spotting = [
        m.OcrLine(0, (0, 0, 704, 1487), -1.0, "だなんか"),
        m.OcrLine(1, (1189, 184, 1311, 363), -1.0, "ごちゃく"),
        m.OcrLine(2, (1189, 180, 1311, 363), -1.0, "ごちゃく"),
    ]
    recs, stats = m.merge_page(
        "s",
        [],
        spotting,
        None,
        page_size=(1400, 1500),
        iou_thr=0.3,
        contain_thr=0.5,
        weak_score=0.85,
        min_chars=3,
    )
    assert [r.text for r in recs] == ["ごちゃく"]
    assert stats["vl_fullpage_dropped"] == 1 and stats["vl_only_dup"] == 1


def test_symbol_dispute_may_not_change_letters():
    m = _hybrid()
    assert m.accept_second_read("ご主人様♡", "ごー主人様♡", 3, reason="symbol") is None
    assert m.accept_second_read("かほ1♡", "かほー♡", 3, reason="symbol") is None
    assert m.accept_second_read("イく", "イく♡", 3, reason="symbol") == "イく♡"
