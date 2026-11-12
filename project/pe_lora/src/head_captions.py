"""Head + prose MOSS captions: a head tag line, a line break, then ``rewrite_set.py``'s prose.

Head = rating → characters → series → @artists → count tags, the MOSS USERGUIDE tag order cut
after the artist and count; the general tags are left to the prose. Names come from the
``{stem}.layout.txt`` beside each resized image (anime_tools' layout JSON summary; converted from
the caption when absent, as the MOSS cache does), the rating
and count tags from the Anima ``{stem}.txt``. Images are copied beside the captions, so ``--out``
is a ready ``scripts/moss/cache.py --src``.

    .venv/bin/python project/pe_lora/src/head_captions.py \\
        --src post_image_dataset/resized/ama_mitsuki --prose post_image_dataset/pe_ab/ama_nl \\
        --out post_image_dataset/pe_ab/ama_head
"""

from __future__ import annotations

import argparse
import json
import re
import shutil

from anime_tools.captions.position_clauses import parse_caption
from library.env import resolve_under_home
from library.moss.cache import caption_converter

RATINGS = {"general", "safe", "sensitive", "questionable", "explicit", "nsfw", "sfw"}
COUNT = re.compile(
    r"^(\d\+?(girl|boy|other)s?|multiple (girls|boys|others)|no humans|solo)$"
)


def head_line(caption: str, summary: dict) -> str:
    flat = parse_caption(caption).flat_tags
    rating = [t for t in flat[:1] if t in RATINGS]
    count = [t for t in flat if COUNT.match(t)]
    named = [summary.get(k, []) for k in ("characters", "series", "artists")]
    return ", ".join(rating + [t for group in named for t in group] + count)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", required=True, help="resized folder: <stem>.png/.txt/.layout.txt"
    )
    ap.add_argument(
        "--prose", required=True, help="rewrite_set.py --out (<stem>.txt prose)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--strip_prefix", default="", help="drop this prefix from each prose"
    )
    args = ap.parse_args()
    src, prose, out = (resolve_under_home(p) for p in (args.src, args.prose, args.out))
    out.mkdir(parents=True, exist_ok=True)
    to_layout = caption_converter("layout_json")
    n, missing = 0, []
    for png in sorted(src.glob("*.png")):
        s = png.stem
        if not (prose / f"{s}.txt").exists():
            missing.append(s)
            continue
        caption = (src / f"{s}.txt").read_text().strip()
        lay = src / f"{s}.layout.txt"
        summary = json.loads(lay.read_text() if lay.exists() else to_layout(caption))[
            "summary"
        ]
        text = (prose / f"{s}.txt").read_text().strip()
        if args.strip_prefix and text.startswith(args.strip_prefix):
            text = text[len(args.strip_prefix) :]
        head = head_line(caption, summary)
        (out / f"{s}.txt").write_text(f"{head}\n{text}\n")
        shutil.copy(png, out / png.name)
        n += 1
    print(f"wrote {n} captions to {out}; no prose for {len(missing)}: {missing}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
