"""The OCR line's one glyph fold (project/cjk_aware_anima_dit/ocr/textnorm.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    path = ROOT / "project" / "cjk_aware_anima_dit" / "ocr" / "textnorm.py"
    spec = importlib.util.spec_from_file_location("textnorm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "raw, target",
    [
        # every spelling of a pause is one …
        ("わからなかったの......", "わからなかったの…"),
        ("これはますます‥‥", "これはますます…"),
        ("あ・・・っ", "あ…っ"),
        ("…………", "…"),
        ("はーい‥…", "はーい…"),
        ("これは‥", "これは…"),
        # a lone middle dot is a separator, not a pause
        ("ジョン・スミス", "ジョン・スミス"),
        # hearts → ♡, wave → ~, fullwidth → ASCII
        ("ムフッ♥!", "ムフッ♡!"),
        ("えらいぞ❤\ufe0fアイー", "えらいぞ♡アイー"),
        ("なっ...成島〜〜", "なっ…成島~~"),
        ("えっ！？　ほんと？", "えっ!? ほんと?"),
        ("あ‼", "あ!!"),
        # long dashes → U+2015, runs keep their length
        ("わ──っ", "わ――っ"),
        ("だから――――", "だから――――"),
        ("それは—人類", "それは―人類"),
        # spacing dakuten → combining, no space (the TARGET_NORM 2 bug)
        ("あ゛っ", "あ゙っ"),
        ("か゛", "が"),
        # whitespace: runs → one space, U+3000 and newlines included
        ("あ　麻美さん\nそれ", "あ 麻美さん それ"),
        ("  BUT YOU'RE  NOT ", "BUT YOU'RE NOT"),
    ],
)
def test_normalize_target(raw, target):
    assert _mod().normalize_target(raw) == target


def test_exact_key_is_the_target_without_whitespace():
    m = _mod()
    raw = "あ゛っ　……♥\n──"
    assert m.exact_key(raw) == m.normalize_target(raw).replace(" ", "")
    assert m.exact_key("あ゛っ") == m.exact_key("あ゙っ")


def test_dash_glyphs_are_the_ones_the_corpus_uses():
    # ― U+2015 (kept), ─ U+2500, — U+2014 (folded)
    m = _mod()
    assert ord("―") == 0x2015 and ord("─") == 0x2500 and ord("—") == 0x2014
    assert m.exact_key("─") == m.exact_key("—") == "―"
