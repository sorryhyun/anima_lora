"""Prompt templates, eval groups, the native scene prompts and clauses."""

from __future__ import annotations

from .paths import REPO

EN_WORDS = [
    "HELLO",
    "STOP",
    "YES",
    "NO WAY",
    "WAIT",
    "SORRY",
    "WHAT",
    "OK",
    "RUN",
    "HELP",
    "GO",
    "HEY",
]

TPL_BUBBLE = 'manga, speech bubble, japanese text. Japanese text reads as "{}".'
TPL_PLAIN = (
    'japanese text, white background, simple background. Japanese text reads as "{}".'
)
TPL_EN = 'manga, speech bubble, english text. English text reads as "{}".'
# cf_sense --cf_layout flat: the EN twin of TPL_PLAIN (no bubble drawn, none
# captioned)
TPL_PLAIN_EN = (
    'english text, white background, simple background. English text reads as "{}".'
)
# S-line composite: the scene's own tag caption (``english text`` →
# ``japanese text``) + the trained clause shape
TPL_SCENE_JA = '{tags}. Japanese text reads as "{text}".'

# grid items (2026-09-20): k single units drawn mechanically in a cols × rows
# grid, one position clause per cell. Both frames and the clause shape are the
# ones `probe/position_probe.py` read on the base model (clause → cell binds)
GRID_FRAMES = {
    "flat": "white background, simple background, no humans, text focus, {lang} text.",
    "bubble": "manga, multiple speech bubbles, {lang} text.",
}
# the plain grid caption (user, 2026-10-02 — grid_small's `recap` leg): no
# `manga`, no language tag, the clause without its language, and the clause's
# wording one of `GRID_CLAUSES` (`{u}` = the cell's unit)
# `tint`: a flat canvas that is not full white (reseed, user 10-03)
GRID_FRAMES_PLAIN = {
    "flat": "white background, simple background, no humans, text focus.",
    "tint": "simple background, no humans, text focus.",
    "bubble": "simple background, no humans, multiple speech bubbles.",
}
GRID_CLAUSES = {
    "reads_as": 'text reads as "{u}".',
    "says": 'the text says "{u}".',
    "written": '"{u}" is written.',
    "bare": '"{u}".',
    "bubble_reads": 'a speech bubble reads "{u}".',
}
GRID_CLAUSES_BUBBLE = {"bubble_reads"}  # true of a bubble cell only
_GRID_COLS = {1: ("",), 2: ("left", "right"), 3: ("left", "middle", "right")}
_GRID_ROWS = {1: ("",), 2: ("top", "bottom"), 3: ("top", "middle", "bottom")}


def grid_cell_header(cols: int, rows: int, cell: int) -> str:
    """Clause header of ``cell`` (row-major) in a ``cols`` × ``rows`` grid —
    anime_tools' position vocabulary (``On the top left`` … ``In the center``)."""
    r, c = divmod(cell, cols)
    if (cols, rows) == (3, 3) and (r, c) == (1, 1):
        return "In the center"
    return "On the " + " ".join(
        w for w in (_GRID_ROWS[rows][r], _GRID_COLS[cols][c]) if w
    )


def grid_caption(
    frame: str,
    cols: int,
    rows: int,
    units,
    order=None,
    lang: str = "japanese",
    horizontal=(),
    clause: str | None = None,
) -> str:
    """``units[i]`` sits in cell ``i``; ``order`` is the clause sequence
    (default reading order). Cells in ``horizontal`` are marked as a
    left-to-right line — the unmarked clause is the manga default, a column.

    ``clause`` (a `GRID_CLAUSES` name) asks for the plain caption:
    `GRID_FRAMES_PLAIN`'s frame and that wording in every cell, no language
    anywhere. It has no line marker, so it takes no ``horizontal`` cell. A
    1×1 is one unit with no position to name: one bubble, the bare clause."""
    cells = order if order is not None else range(len(units))
    if clause is not None:
        assert not horizontal, "the plain grid caption has no line marker"
        assert frame == "bubble" or clause not in GRID_CLAUSES_BUBBLE, (frame, clause)
        tpl = GRID_CLAUSES[clause]
        if cols * rows == 1:
            head = GRID_FRAMES_PLAIN[frame].replace(
                "multiple speech bubbles", "speech bubble"
            )
            said = tpl.format(u=units[0])
            return f"{head} {said[0].upper()}{said[1:]}"
        clauses = [
            f"{grid_cell_header(cols, rows, i)}, {tpl.format(u=units[i])}"
            for i in cells
        ]
        return f"{GRID_FRAMES_PLAIN[frame]} {' '.join(clauses)}"
    clauses = [
        f"{grid_cell_header(cols, rows, i)}, {'horizontal ' if i in horizontal else ''}"
        f'{lang.capitalize()} text reads as "{units[i]}".'
        for i in cells
    ]
    return f"{GRID_FRAMES[frame].format(lang=lang)} {' '.join(clauses)}"


# report / sheet order; ``word`` / ``word_held`` / ``line`` are the 2026-09-14
# word-address groups (``--words``)
EVAL_GROUPS = (
    "single",
    "single_held",
    "single_ext",
    "single_small",
    "single_kanji",
    "single_extra",
    "word",
    "word_held",
    "line",
    "flip",
    "str3",
    "phrase",
    "phrase_held",
    "short",
    "short_held",
    "gword",
    "gword_held",
    "combo",
    "corpus",
    "en",
)

NATIVE_PROMPTS = (
    REPO
    / "project"
    / "finished"
    / "cjk_aware_anima"
    / "assets"
    / "unmask_eval_prompts.txt"
)
# the user's own target captions (ComfyUI, 2026-09-17), rendered verbatim by
# ``--stage target`` (the line's copy: ``project/finished/cjk_anima_scale/assets/``)
TARGET_PROMPTS = (
    REPO / "project" / "finished" / "cjk_anima_scale" / "assets" / "target_prompts.txt"
)
NATIVE_CLAUSES = {
    # the trained clause shape, hung off a scene prompt instead of the template
    "en": '{p}, japanese text. Japanese text reads as "{k}".',
    # the user's phrasing: a Japanese-language clause (its own words route to
    # untrained pack rows; only the kana row carries the delta)
    "ja": "{p}. ひらがなの「{k}」という文字がある。",
    # the EN-reference swap (2026-09-15): the *same* caption as the EN
    # reference render with only the quoted word replaced — the pair differs
    # in nothing but the ext row (enref uses "hi"; see eval/enref.py)
    "swap": '{p}, english text. English text reads as "{k}".',
}
