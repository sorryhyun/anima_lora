#!/usr/bin/env python3
"""Random contact sheets of the ``OCR:`` speech rows of the two-prompt reader
(``build_sfx_prefix.py --speech`` / ``--kozh``): Manga109 speech, gelnote KO / ZH and
swaprelax_v5, one sheet per source.

    page context (red the row's box, blue the page's other speech boxes, green COO
    polygons on Manga109) | the crop as trained | the target

    ANIMA_MANGA109S_ROOT=... python project/finished/ocr_reader/probes/sfxp_text_sheet.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
import manga109 as m109  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
GELNOTE = Path("/media/sorryhyun/new/dataset/gelnote_crops")
SWAPRELAX = Path("/media/sorryhyun/new/dataset/hito_ocr/swap_relax_v5")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
PW, PH, CW = 330, 300, 300


def fit(img, mw, mh):
    h, w = img.shape[:2]
    s = min(mw / w, mh / h, 3.0)
    return cv2.resize(
        img,
        (max(1, int(w * s)), max(1, int(h * s))),
        interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
    )


def context(page, box, others=(), polys=()):
    H, W = page.shape[:2]
    x0, y0, x1, y1 = box
    m = int(0.6 * max(x1 - x0, y1 - y0)) + 30
    X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    ctx = page.copy()
    for b in others:
        cv2.rectangle(
            ctx, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 120, 0), 1
        )
    for p in polys:
        cv2.polylines(
            ctx, [np.asarray(p, np.int32).reshape(-1, 1, 2)], True, (0, 200, 0), 2
        )
    cv2.rectangle(ctx, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    return fit(ctx[int(Y0) : int(Y1), int(X0) : int(X1)], PW, PH)


def wrap(text, n=34, max_lines=5):
    out = []
    for i, para in enumerate(str(text).split("\n")):
        para = ("⏎ " if i else "") + para
        out += [para[k : k + n] for k in range(0, max(len(para), 1), n)]
    return out[:max_lines] + (["…"] if len(out) > max_lines else [])


def card(ctx, crop, head, text, font, small):
    lines = wrap(text)
    tw = PW + CW + 40
    canvas = Image.new("RGB", (tw, PH + 30 + 26 * 6), "white")
    for im, x in ((ctx, 0), (fit(crop, CW, PH), PW + 20)):
        canvas.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), (x, 0))
    d = ImageDraw.Draw(canvas)
    d.text((4, PH + 4), head, fill=(90, 90, 90), font=small)
    for i, ln in enumerate(lines):
        d.text((4, PH + 26 + 26 * i), ln, fill=(0, 70, 200), font=font)
    return canvas


def sheet(title, cards, path, big, small, cols=2):
    cw, ch = cards[0].size
    rows = (len(cards) + cols - 1) // cols
    hdr = 70
    S = Image.new(
        "RGB", (cols * (cw + 16) + 16, hdr + rows * (ch + 16) + 8), (238, 238, 238)
    )
    d = ImageDraw.Draw(S)
    d.text((16, 10), title, fill="black", font=big)
    d.text(
        (16, 44),
        "context: red = this row's box · blue = other speech boxes on the page · green = COO SFX polygons"
        "     |     crop as trained     |     target (OCR:)",
        fill=(80, 80, 80),
        font=small,
    )
    for i, c in enumerate(cards):
        S.paste(c, (16 + (i % cols) * (cw + 16), hdr + (i // cols) * (ch + 16)))
    S.save(path)
    print(f"[sheet] {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = m109.derived_root()
    font = ImageFont.truetype(FONT, 18)
    small = ImageFont.truetype(FONT, 14)
    big = ImageFont.truetype(FONT, 24)

    # Manga109 speech
    sp = pd.read_parquet(d / "manifest_sfxp_speech.parquet")
    sp = sp[sp.split == "train"].sample(a.n, random_state=a.seed)
    g = pd.read_parquet(d / "manifest.parquet")
    g = g[(g.source.fillna("grey") == "grey") & (g.split == "train")]
    by_page = defaultdict(list)
    for r in g[g.book.isin(sp.book) & g.page.isin(sp.page)].itertuples():
        by_page[(r.book, r.page)].append(r)
    cards = []
    for r in sp.itertuples():
        page = cv2.imread(str(m109.page_path(r.book, r.page)))
        rows = by_page[(r.book, r.page)]
        me = next(x for x in rows if x.id == r.id)
        q = json.loads(me.poly)
        box = (min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2]))
        others = [
            (min(p[0::2]), min(p[1::2]), max(p[0::2]), max(p[1::2]))
            for x in rows
            if x.kind == "speech" and x.id != r.id
            for p in [json.loads(x.poly)]
        ]
        polys = [json.loads(x.poly) for x in rows if x.kind == "sfx"]
        crop = cv2.imread(str(d / r.path))
        cards.append(
            card(
                context(page, box, others, polys),
                crop,
                f"{r.book} p{r.page}  {r.id}",
                r.text,
                font,
                small,
            )
        )
    sheet(
        "Manga109 speech (OCR: rows, <text> boxes re-cut at the 4 px margin)",
        cards,
        a.out / "12_speech_m109.png",
        big,
        small,
    )

    # KO / ZH
    kz = pd.read_parquet(d / "manifest_sfxp_kozh.parquet")
    gel = {}
    for line in open(GELNOTE / "manifest.jsonl", encoding="utf-8"):
        x = json.loads(line)
        gel[x["id"]] = x
    lab = {}
    for line in open(SWAPRELAX / "labels.jsonl", encoding="utf-8"):
        x = json.loads(line)
        lab[x["crop"]] = x
    sw = (
        pd.read_parquet(d / "manifest_pseudo_swaprelax_v5.parquet").set_index("id").path
    )
    gel_by_page = defaultdict(list)
    for x in gel.values():
        gel_by_page[x["page_path"]].append(x["box"])
    lab_by_page = defaultdict(list)
    for x in lab.values():
        lab_by_page[x["page"]].append(x["box"])
    for src, num in (("gelnote_ko", 13), ("gelnote_zh", 14), ("swaprelax_v5", 15)):
        sub = kz[kz.source == src].sample(a.n, random_state=a.seed)
        cards = []
        for r in sub.itertuples():
            if src.startswith("gelnote"):
                x = gel[r.id]
                page_p, box, others = (
                    GELNOTE / x["page_path"],
                    x["box"],
                    gel_by_page[x["page_path"]],
                )
            else:
                x = lab["crops/" + os.path.basename(sw[r.id])]
                page_p, box, others = REPO / x["page"], x["box"], lab_by_page[x["page"]]
            page = cv2.imread(str(page_p))
            others = [b for b in others if list(b) != list(box)]
            crop = cv2.imread(str(d / r.path))
            cards.append(
                card(
                    context(page, box, others),
                    crop,
                    f"{r.book} p{r.page}  {r.id}",
                    r.text,
                    font,
                    small,
                )
            )
        sheet(
            f"{src} (OCR: rows, re-cut at the 4 px margin, labels as v6)",
            cards,
            a.out / f"{num}_{src}.png",
            big,
            small,
        )


if __name__ == "__main__":
    main()
