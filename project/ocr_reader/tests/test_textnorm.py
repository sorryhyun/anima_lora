"""The OCR line's one glyph fold (``core/textnorm.py``)."""

from __future__ import annotations

import pytest


def _mod():
    from project.ocr_reader.core import textnorm

    return textnorm


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
        ("なっ...成島〜〜", "なっ…成島~"),
        ("えっ！？　ほんと？", "えっ!? ほんと?"),
        ("あ‼", "あ!!"),
        # long dashes → "-", a run one "-" (TARGET_NORM 12)
        ("わ──っ", "わ-っ"),
        ("だから――――", "だから-"),
        ("それは—人類", "それは-人類"),
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


def test_training_target_keeps_the_two_hearts():
    # TARGET_NORM 10: the target keeps ♥ / ♡ apart (❤ is the filled one); keys stay ♡
    m = _mod()
    assert m.training_target("ムフッ♥!") == "ムフッ♥!"
    assert m.training_target("えらいぞ❤\ufe0fアイー") == "えらいぞ♥アイー"
    assert m.training_target("びく♡💢") == "びく♡💢"
    assert m.training_target("あ♡♥❤っ") == "あ♡♥♥っ"
    for raw in ("ムフッ♥!", "えらいぞ❤\ufe0fアイー", "びく♡", "あ♡♥❤っ"):
        assert "♥" not in m.exact_key(raw)
        assert m.exact_key(m.training_target(raw)) == m.exact_key(raw)
    raw = "あ゛っ　……\n──"
    assert m.training_target(raw) == "あ゙っ ……\n-"
    assert m.exact_key(m.training_target(raw)) == m.exact_key(raw)


@pytest.mark.parametrize(
    "raw",
    ["わからなかったの......", "…………", "なっ...成島", "え…?", "ジョン・スミス"],
)
def test_training_target_keeps_dot_spelling(raw):
    # TARGET_NORM 7: dots trained as labelled (… survives NFKC); keys still fold
    m = _mod()
    assert m.training_target(raw) == raw
    assert m.exact_key(m.training_target(raw)) == m.exact_key(raw)


@pytest.mark.parametrize(
    "raw, target",
    [
        ("あ・・・っ", "あ…っ"),
        ("・・", "…"),
        ("あ・・・・", "あ…"),
        ("あ・・・・・", "あ……"),
        ("そんな・・・・・・\nこと", "そんな……\nこと"),
        ("・・・・・・・・", "………"),
        ("あ・・…", "あ……"),
    ],
)
def test_training_target_folds_mid_dot_runs(raw, target):
    # TARGET_NORM 11: a ・ run → … per three dots, at least one; keys unchanged
    m = _mod()
    assert m.training_target(raw) == target
    assert m.exact_key(m.training_target(raw)) == m.exact_key(raw)


def test_training_target_folds_dot_width():
    m = _mod()
    assert m.training_target("あ．．．･･･") == "あ...…"
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
    # ― U+2015, ─ U+2500, — U+2014: all one "-" (TARGET_NORM 12)
    m = _mod()
    assert ord("―") == 0x2015 and ord("─") == 0x2500 and ord("—") == 0x2014
    assert m.exact_key("─") == m.exact_key("—") == m.exact_key("―") == "-"


@pytest.mark.parametrize(
    "raw, target",
    [
        ("짜안―\n손님용", "짜안-\n손님용"),
        ("촬영은 커녕——…", "촬영은 커녕-…"),
        ("유즈–", "유즈-"),
        ("스기야마-참모-총장", "스기야마-참모-총장"),
        ("だから――――", "だから-"),
    ],
)
def test_korean_dashes_fold_to_hyphen(raw, target):
    # TARGET_NORM 8: a Hangul string's dashes → "-"; since 12 a run is one "-"
    assert _mod().training_target(raw) == target


@pytest.mark.parametrize(
    "raw, target",
    [
        ("むうーーーーーーーー~~", "むう-~"),
        ("ケーキ", "ケーキ"),
        ("あれー?", "あれー?"),
        ("もう―――", "もう-"),
        ("あれ―?", "あれ-?"),
        ("――そう思って", "-そう思って"),
        ("ゴオーー…", "ゴオ-…"),
        ("ーー―~", "-~"),
        ("そうか〜〜〜", "そうか~"),
        ("ず~ん", "ず~ん"),
        ("747-400D", "747-400D"),
        ("---江戸---", "-江戸-"),
    ],
)
def test_drawn_out_lines_fold(raw, target):
    # TARGET_NORM 12: ― → "-", a ー / - run → one "-", a ~ run → one ~; a lone ー stays
    m = _mod()
    assert m.training_target(raw) == target
    assert m.exact_key(raw) == m.exact_key(target) == "".join(target.split())


@pytest.mark.parametrize(
    "raw, target",
    [
        ("○○바이러스", "○○바이러스"),
        ("オ〇ニー", "オ○ニー"),
        ("二〇三六年", "二○三六年"),
        ("◯太陽は小さく…", "○太陽は小さく…"),
        ("O자키군", "O자키군"),
        ("「☓☓☓と☓×☓」", "「×××と×××」"),
        ("✕ 프라우다고교", "× 프라우다고교"),
        ("X파일", "X파일"),
    ],
)
def test_shapes_fold(raw, target):
    # TARGET_NORM 9: ◯ / 〇 → ○, ☓ / ✕ … → × in the target and the scoring key;
    # Latin O / x / X stay
    assert _mod().training_target(raw) == target
    assert _mod().exact_key(raw) == "".join(target.split())


@pytest.mark.parametrize(
    "a, b, same",
    [
        # a break in either string matches a space or none there
        ("최근 제게\n보내는 관심이", "최근 제게 보내는 관심이", True),
        ("최근 제게\n보내는", "최근 제게보내는", True),
        ("啪\n!!", "啪!!", True),
        # away from a break, 띄어쓰기 counts
        ("최근 제게 보내는", "최근 제게보내는", False),
        ("남친 있는데", "남친있는데\n", False),
        # the exact_key folds still apply; a missing glyph is still a miss
        ("아앙♥", "아앙♡", True),
        ("아앙♥", "아앙", False),
        (" x ", "x", True),
    ],
)
def test_linebreak_blind(a, b, same):
    tn = _mod()
    assert tn.linebreak_blind(a, b) is same
    assert tn.linebreak_blind(b, a) is same
