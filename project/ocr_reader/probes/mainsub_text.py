#!/usr/bin/env python3
"""The Manga109 ``<text>`` boxes the main / sub crops hold that ``manifest_sfxp_speech``
does not (60,810 on the train split, 11-21): their texts are the raw annotation, with no
ink-projection line breaks and no furigana pass. This cuts them as ``build_speech`` cuts
the speech crops (the box at the 4 px margin) into ``crops_sfxp/mainsub_text/<split>/`` and
writes ``manifest_mainsub_text.parquet`` (the speech manifest's columns, ``source =
mainsub_text``), for ``furigana_find.py --manifest mainsub_text`` and ``linebreak_auto.py
inkread --set mainsub_text``. Texts take ``ellipsis_fold`` unless they hold Hangul
(Korean keeps ``...`` and ``…`` apart). ``m109_mainsub.py`` reads the manifest after the
speech one.

    ANIMA_MANGA109S_ROOT=… .venv/bin/python project/ocr_reader/probes/mainsub_text.py \\
        --mainsub output/ocr/mainsub/m109_mainsub.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import pandas as pd

from project.ocr_reader.core import manga109 as m109
from project.ocr_reader.core.crops import margin_box
from project.ocr_reader.core.labels import ellipsis_fold

HANGUL = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mainsub", type=Path, required=True)
    a = ap.parse_args()
    d = m109.derived_root()
    sp = pd.read_parquet(d / "manifest_sfxp_speech.parquet")
    have = {(b, int(p), str(i)) for b, p, i in zip(sp.book, sp.page, sp.id)}
    want: dict[tuple, str] = {}
    for line in open(a.mainsub, encoding="utf-8"):
        x = json.loads(line)
        for t in x["sp"]:
            k = (x["book"], int(x["page"]), str(t))
            if k not in have:
                want[k] = x["split"]
    books = sorted({k[0] for k in want})
    rows = []
    for book in books:
        cur, img = None, None
        for ln in m109.iter_text(book):
            k = (book, int(ln.page), str(ln.id))
            if k not in want or not ln.text:
                continue
            if cur != ln.page:
                img, cur = cv2.imread(str(m109.page_path(book, ln.page))), ln.page
            q = ln.poly
            x0, y0, x1, y1 = min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2])
            H, W = img.shape[:2]
            c = margin_box(x0, y0, x1, y1, W, H)
            crop = img[c[1] : c[3], c[0] : c[2]]
            if crop.size == 0:
                continue
            split = want[k]
            rel = (
                f"crops_sfxp/mainsub_text/{split}/{book}_{int(ln.page):03d}_{ln.id}.png"
            )
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / rel), crop)
            h, w = crop.shape[:2]
            text = ln.text if HANGUL.search(ln.text) else ellipsis_fold(ln.text)
            rows.append(
                dict(
                    split=split,
                    kind="speech",
                    id=str(ln.id),
                    book=book,
                    page=int(ln.page),
                    text=text,
                    # crops.deskew_crop's rule, as the speech manifest has it
                    orient="vertical"
                    if h > w * 1.3
                    else ("horizontal" if w > h * 1.3 else "square"),
                    w=w,
                    h=h,
                    path=rel,
                    source="mainsub_text",
                )
            )
    df = pd.DataFrame(rows)
    df.to_parquet(d / "manifest_mainsub_text.parquet", index=False)
    print(
        f"[mainsub_text] {len(df)} of {len(want)} <text> boxes cut; "
        f"{df.orient.value_counts().to_dict()}; with a line break "
        f"{int(df.text.str.contains(chr(10)).sum())} → {d / 'manifest_mainsub_text.parquet'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
