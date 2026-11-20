"""The OCR line's one glyph fold (``ocr/textnorm.py``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HOME = Path(__file__).resolve().parents[1]


def _mod():
    path = HOME / "ocr" / "textnorm.py"
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
        # curly double quotes → ASCII (TARGET_NORM 6)
        ("”ここ”って", '"ここ"って'),
        ("“그래”", '"그래"'),
        # lone jamo stay the typed compatibility form; syllables untouched
        ("ㅋㅋㅋ 큭", "ㅋㅋㅋ 큭"),
        ("\u110f\u110f\u1112", "ㅋㅋㅎ"),
        ("ㅠㅠ", "ㅠㅠ"),
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


def test_training_target_spells_the_single_token_heart():
    # R5b: only the training target flips to ♥; key and record stay ♡
    m = _mod()
    for raw in ("ムフッ♥!", "えらいぞ❤\ufe0fアイー", "びく♡", "あ♡♥❤っ"):
        assert "♡" not in m.training_target(raw)
        assert m.training_target(raw) == m.normalize_target(raw).replace("♡", "♥")
        assert "♥" not in m.exact_key(raw)
    raw = "あ゛っ　……\n──"
    assert m.training_target(raw) == "あ゙っ ……\n――"
    assert m.exact_key(m.training_target(raw)) == m.exact_key(raw)


@pytest.mark.parametrize(
    "raw",
    ["わからなかったの......", "あ・・・っ", "…………", "なっ...成島", "え…?"],
)
def test_training_target_keeps_dot_spelling(raw):
    # TARGET_NORM 7: dots trained as labelled (… survives NFKC); keys still fold
    m = _mod()
    assert m.training_target(raw) == raw
    assert m.exact_key(m.training_target(raw)) == m.exact_key(raw)


def test_training_target_folds_dot_width():
    m = _mod()
    assert m.training_target("あ．．．･･･") == "あ...・・・"
    assert m.training_target("お‥‥お玉") == "お....お玉"
    assert m.training_target("はーい‥…") == "はーい..…"


@pytest.mark.parametrize(
    "raw, target",
    [
        ("나중에\n여자 친구가\n좋아할걸?", "나중에\n여자 친구가\n좋아할걸?"),
        ("あ　麻美さん\nそれ", "あ 麻美さん\nそれ"),
        ("  一行目 \n\n\n  二行目  ", "一行目\n二行目"),
        ("a\r\nb\rc", "a\nb\nc"),
        ("\n", ""),
    ],
)
def test_training_target_keeps_line_breaks(raw, target):
    assert _mod().training_target(raw) == target


def test_dash_glyphs_are_the_ones_the_corpus_uses():
    # ― U+2015 (kept), ─ U+2500, — U+2014 (folded)
    m = _mod()
    assert ord("―") == 0x2015 and ord("─") == 0x2500 and ord("—") == 0x2014
    assert m.exact_key("─") == m.exact_key("—") == "―"


@pytest.mark.parametrize(
    "raw, target",
    [
        ("짜안―\n손님용", "짜안-\n손님용"),
        ("촬영은 커녕——…", "촬영은 커녕--…"),
        ("유즈–", "유즈-"),
        ("스기야마-참모-총장", "스기야마-참모-총장"),
        ("だから――――", "だから――――"),
    ],
)
def test_korean_dashes_fold_to_hyphen(raw, target):
    # TARGET_NORM 8: a Hangul string's dashes → "-", run length kept; Japanese keeps ―
    assert _mod().training_target(raw) == target


@pytest.mark.parametrize(
    "raw, target",
    [
        ("○○바이러스", "○○바이러스"),
        ("オ〇ニー", "オ○ニー"),
        ("二〇三六年", "二○三六年"),
        ("◯太陽は小さく…", "○太陽は小さく…"),
        ("O자키군", "O자키군"),
    ],
)
def test_circles_fold_to_white_circle(raw, target):
    # TARGET_NORM 9: ◯ / 〇 → ○ in the target and the scoring key; Latin O stays
    assert _mod().training_target(raw) == target
    assert _mod().exact_key(raw) == target
