"""The S-line scene readers the line's recipes use (``cjk_scale/recipes.py``):
the kept ``scenes_<tag>`` pools (``load_scenes``), the scene caption with the
anchor swapped for JA text (``scene_caption``), and the sentence / short
letter floors (``_letters``, ``_SENT_DISTINCT``, ``_SHORT_DISTINCT``). The
old data stage's S-line mix (``synth_recs``) is gone (pruned 2026-09-25).
"""

from __future__ import annotations

import json

from common.paths import OUT
from common.prompts import TPL_SCENE_JA
from common.text import KANJI_RE


def load_scenes(
    tags: str,
    min_ar: float = 0.0,
    min_tokens: int = 0,
    drop: str = "",
    one_bubble: str = "",
) -> list[dict]:
    """Kept scenes of every ``scenes_<tag>`` run in the comma list (s0 + a
    frame-mix run compose). ``min_ar`` (``--scene_tall_ar``) keeps only
    scenes whose headline region is at least that tall for its width —
    the sentence line's tategaki pool (user, 2026-09-16: tall bubbles
    first; regenerate when they run short). ``min_tokens``
    (``--scene_min_tokens``) drops canvases under that many DiT tokens
    (900: the 512² family only — user, 2026-09-16). ``drop`` (``--scene_drop``,
    ``tag:i,i;tag:i``) removes kept scenes by index — sl1w 332 / 957 are
    bubble-less tall regions (a hooded sketch's body, a box beside a
    figure) that the sentence quota reused 12–13 times per 400 items (user,
    2026-09-16). ``one_bubble`` (``--scene_one_bubble``, comma tags) keeps
    only scenes with one anchor box in those runs. Every scene is tagged
    ``pool`` = its run tag (indices ``i`` repeat across runs)."""
    one = {t for t in one_bubble.split(",") if t}
    unknown = one - set(tags.split(","))
    assert not unknown, f"--scene_one_bubble {sorted(unknown)}: not in --scenes {tags}"
    dropped = {}
    for part in [x for x in drop.split(";") if x]:
        tag, ids = part.split(":")
        dropped[tag] = {int(x) for x in ids.split(",") if x}
    scenes = []
    for tag in [t for t in tags.split(",") if t]:
        path = OUT / f"scenes_{tag}" / "scenes.jsonl"
        got = [json.loads(ln) for ln in path.read_text().splitlines() if ln]
        assert got, f"--scenes {tag}: no kept scenes in {path}"
        if dropped.get(tag):
            got = [s for s in got if s["i"] not in dropped[tag]]
            print(f"scenes {tag}: dropped {sorted(dropped[tag])}", flush=True)
        for s in got:
            s["pool"] = tag
        if tag in one:
            single = [s for s in got if len(s["boxes_anchor"]) == 1]
            print(
                f"scenes {tag}: {len(single)}/{len(got)} kept scenes with one anchor bubble",
                flush=True,
            )
            got = single
        if min_tokens > 0:
            big = [
                s
                for s in got
                if (s["shape"][0] // 16) * (s["shape"][1] // 16) >= min_tokens
            ]
            print(
                f"scenes {tag}: {len(big)}/{len(got)} kept scenes at >= {min_tokens} tokens",
                flush=True,
            )
            got = big
        if min_ar > 0:
            tall = [
                s
                for s in got
                if (s["region"][3] - s["region"][1])
                >= min_ar * (s["region"][2] - s["region"][0])
            ]
            print(
                f"scenes {tag}: {len(tall)}/{len(got)} kept scenes with region "
                f"AR >= {min_ar}",
                flush=True,
            )
            got = tall
        scenes += got
    assert scenes, f"--scenes {tags}: no scenes left (--scene_tall_ar {min_ar})"
    return scenes


HORIZONTAL_SUFFIX = ", written horizontally."


def scene_caption(scene: dict, text: str, horizontal: bool = False) -> str:
    """The scene prompt with the anchor swapped for the JA text *in the frame
    the scene was drawn under* (`clause_tpl`; s0 records predate it and are
    the `reads as` frame): `english text` → `japanese text` in the tags,
    `English text reads as` → `Japanese text reads as` in the clause, every
    other frame (`She is saying "…"`, `holding a sign that reads "…"`) keeps
    its words and only the quote changes.

    ``horizontal`` (a text drawn as left-to-right lines): a `reads as` frame
    becomes `horizontal Japanese text reads as "…"` — the grid cell's marker
    (`common.prompts.grid_caption`) — and every other frame takes the
    suffix `, written horizontally.` before its period. The unmarked
    caption is a column, the manga default."""
    generals = [
        "japanese text" if g == "english text" else g for g in scene["generals"]
    ]
    tags = ", ".join(scene["head"] + sorted(set(generals)))
    tpl = scene.get("clause_tpl") or TPL_SCENE_JA.replace("{tags}. ", "").replace(
        "{text}", "{a}"
    )
    clause = tpl.replace("English text reads as", "Japanese text reads as").replace(
        "English SFX reads as", "Japanese SFX reads as"
    )
    if horizontal:
        if "Japanese text reads as" in clause:
            clause = clause.replace(
                "Japanese text reads as", "horizontal Japanese text reads as"
            )
        else:
            clause = clause.rstrip().rstrip(".") + HORIZONTAL_SUFFIX
    return f"{tags}. {clause.format(a=text)}"


def _letters(t: str) -> list[str]:
    """Kana + kanji glyphs of ``t`` — the sentence floor's unit (punctuation,
    digits, Latin and the prolonged-sound mark do not count)."""
    return [c for c in t if "ぁ" <= c <= "ゖ" or "ァ" <= c <= "ヺ" or KANJI_RE.match(c)]


# the length floor alone let ハハハハハハ / おやおやおや / うわああああ through
# as sentences (smoke, 2026-09-16): a sentence has >= 4 distinct letters, a
# short item >= 2
_SENT_DISTINCT = 4
_SHORT_DISTINCT = 2
