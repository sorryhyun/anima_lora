"""**caption**: captions each target with the twin read's lines in its
pasted groups (CPU), or with ``--caption_ocr`` (GPU) with a second read of
the staged target itself (``<root>/ocr_staged.json``, keyed by page and
pasted regions); either way only lines holding no kana or CJK ideograph, so
JA lettering — left by the translator or by a partial swap — is never named
and stays the source; a target with no such line drops. The caption is layout JSON:
``{"summary": {"caption": "comic, english text"}, "items": [one text item
(text + bbox) per line]}``. Writes ``{stem}.txt``, ``pairs.jsonl`` and the
final ``stats.json``.
"""

from __future__ import annotations

import json

from i2i_tasks.swap_text.common import (
    JA_SCRIPT,
    read_json,
    read_jsonl,
    write_jsonl,
    write_stats,
    write_text,
)
from i2i_tasks.swap_text.ocr import kept_lines, read_into

LANG_TAG = {"en": "english text", "ko": "korean text"}


def layout_caption(lang: str, lines: list, size) -> str:
    from anime_tools.captions.layout_json import centered_box

    layout = {"summary": {"caption": f"comic, {LANG_TAG[lang]}"}}
    if lines:
        layout["items"] = [
            {
                "label": "text",
                "caption": "text",
                "text": text,
                "bbox": centered_box(box, size),
            }
            for text, box in lines
        ]
        layout["bbox_coord_space"] = "abs_centralized"
        layout["bbox_format"] = "yxyx"
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def _pair_row(r: dict, n_lines: int) -> dict:
    return {
        "stem": r["stem"],
        "target": f"{r['lang']}/{r['stem']}.tgt.webp",
        "cond": f"{r['lang']}/{r['stem']}.src.webp",
        "caption": f"{r['lang']}/{r['stem']}.txt",
        "work": r["work"],
        "lang": r["lang"],
        "mode": r["mode"],
        "groups": r["groups"],
        "clean_groups": r["clean_groups"],
        "swapped_groups": r["swapped_groups"],
        "lines": n_lines,
    }


def stage_caption(a) -> None:
    staging = a.root / "staging"
    plan = read_jsonl(staging / "plan.jsonl")

    def key(r):
        return f"{r['key']}:{','.join(map(str, r['chosen']))}"

    if a.caption_ocr:
        reads = read_into(
            a,
            a.root / "ocr_staged.json",
            [key(r) for r in plan],
            [staging / r["lang"] / f"{r['stem']}.tgt.webp" for r in plan],
            "caption ocr",
        )
        reads = {k: kept_lines(v) for k, v in reads.items()}
    rows, n_ja, n_empty = [], 0, 0
    for r in plan:
        base = staging / r["lang"]
        read = reads[key(r)] if a.caption_ocr else r["lines"]
        lines = [[t.replace('"', "”").strip(), b] for t, b in read]
        lines = [ln for ln in lines if ln[0]]
        keep = [ln for ln in lines if not JA_SCRIPT.search(ln[0])]
        n_ja += len(lines) - len(keep)
        if not keep:
            n_empty += 1
            for ext in (".tgt.webp", ".src.webp", ".txt"):
                (base / f"{r['stem']}{ext}").unlink(missing_ok=True)
            continue
        text = layout_caption(r["lang"], keep, r["size"])
        write_text(base / f"{r['stem']}.txt", text)
        rows.append(_pair_row(r, len(keep)))
    write_jsonl(staging / "pairs.jsonl", rows)
    stats = read_json(staging / "stats.json")
    stats.update(
        {
            "pairs": len(rows),
            "by_lang": {lang: sum(r["lang"] == lang for r in rows) for lang in a.langs},
            "partial": sum(r["mode"] == "partial" for r in rows),
            "works": len({r["work"] for r in rows}),
            "dropped_unread_target": n_empty,
            "ja_lines_dropped": n_ja,
        }
    )
    write_stats(staging / "stats.json", stats)
