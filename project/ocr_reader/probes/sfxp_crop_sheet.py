#!/usr/bin/env python3
"""Detector-box SFX crops (``build_sfx_prefix.py --det``) against the COO polygons, by type
(the classifier is ``core.det_types.classify``).

Every train COO polygon is matched to the AnimeText boxes of its page (``pf`` = share of the
polygon inside the crop, ``cf`` = share of the crop it covers, the build's rule) and gets one
type; neg_speech boxes written as ``SFX:`` negatives get a type of their own when a COO
polygon reaches into them. Writes ``types.parquet`` and one contact sheet per type:

    page context (green COO polygon, red the box(es) in question, orange other detector
    boxes, blue Manga109 speech boxes) | the detector crop as trained | v6's COO crop

    ANIMA_MANGA109S_ROOT=... python project/ocr_reader/probes/sfxp_crop_sheet.py \\
        --out <dir> [--n 12] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from project.ocr_reader.core import manga109 as m109
from project.ocr_reader.core.det_types import classify

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

TYPES = {
    "same": "same: one box holds the SFX alone, crop ≈ COO crop (IoU ≥ 0.6, pf ≥ 0.9)",
    "loose": "loose: one box holds the SFX alone, but the crop is much larger / offset (IoU < 0.6)",
    "cut": "cut: kept with the full COO text, but 10–20 % of the polygon lies outside the crop",
    "graze": "graze: label = this SFX only, another COO polygon partly visible in the crop (pf < 0.2)",
    "merged": "merged: one box holds 2+ SFX polygons, joined right→left, top→down",
    "mixed": "mixed: SFX and speech in one box (SFX: target the SFX, OCR: target the speech if whole)",
    "dup": "dup: the SFX is ≥ 80 % inside 2+ kept boxes (trained twice, different crops)",
    "drop_nb": "drop_nb: the SFX fits a box, but the box was dropped (a neighbour SFX is cut)",
    "fragment": "fragment: the detector split / cut the SFX (best pf 0.2–0.8) → dropped",
    "missed": "missed: no box covers ≥ 20 % of the SFX → never an SFX: row",
    "neg_graze": 'neg_graze: SFX: negative (target "") with a COO polygon partly inside',
}


def fit(img, mw, mh):
    h, w = img.shape[:2]
    s = min(mw / w, mh / h, 3.0)
    return cv2.resize(
        img,
        (max(1, int(w * s)), max(1, int(h * s))),
        interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
    )


def card(r, det, grey, font, small, PW=330, PH=300, CW=200):
    page = cv2.imread(str(m109.page_path(r.book, r.page)))
    H, W = page.shape[:2]
    jj = json.loads(r.det_rows)
    pts = np.asarray(json.loads(r.poly), np.int32).reshape(-1, 2)
    xs = [pts[:, 0].min(), pts[:, 0].max()]
    ys = [pts[:, 1].min(), pts[:, 1].max()]
    for j in jj:
        b = det.at[j, "box"]
        xs += [b[0], b[2]]
        ys += [b[1], b[3]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    m = int(0.35 * max(x1 - x0, y1 - y0)) + 30
    X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    ctx = page.copy()
    pg_rows = det[(det.book == r.book) & (det.page == r.page)]
    for b in pg_rows.box:
        if b[2] > X0 and b[0] < X1 and b[3] > Y0 and b[1] < Y1:
            cv2.rectangle(ctx, (b[0], b[1]), (b[2], b[3]), (0, 150, 255), 1)
    for sp in SPEECH.get((r.book, r.page), []):
        cv2.rectangle(ctx, sp[:2], sp[2:], (255, 120, 0), 1)
    cv2.polylines(ctx, [pts.reshape(-1, 1, 2)], True, (0, 200, 0), 2)
    for j in jj:
        b = det.at[j, "box"]
        cv2.rectangle(ctx, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    ctx = fit(ctx[Y0:Y1, X0:X1], PW, PH)
    # detector crop as trained
    j0 = jj[0] if jj else None
    if j0 is not None:
        p = det.at[j0, "path"]
        b = det.at[j0, "box"]
        dc = (
            cv2.imread(str(m109.derived_root() / p))
            if isinstance(p, str)
            else page[b[1] : b[3], b[0] : b[2]]  # not written: the box as cut
        )
    else:
        dc = np.full((40, 120, 3), 230, np.uint8)
    dc = fit(dc, CW, PH)
    g = grey.get((r.book, r.page, r.coo_id.split("|")[0]))
    gc = cv2.imread(str(m109.derived_root() / g)) if g else None
    gc = fit(gc, CW, PH) if gc is not None else np.full((40, 120, 3), 230, np.uint8)
    tw = PW + 2 * CW + 40
    th = PH + 112
    canvas = Image.new("RGB", (tw, th), "white")
    for im, x in ((ctx, 0), (dc, PW + 20), (gc, PW + CW + 40)):
        canvas.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), (x, 0))
    d = ImageDraw.Draw(canvas)
    y = PH + 4
    d.text(
        (4, y),
        f"{r.book} p{r.page}  coo {r.coo_id}" + ("  (joined)" if r.joined else ""),
        fill=(90, 90, 90),
        font=small,
    )
    d.text((4, y + 20), f"COO: {r.coo_text}", fill=(0, 130, 0), font=font)
    if j0 is not None:
        sfx = det.at[j0, "sfx_text"]
        sp = det.at[j0, "speech_text"]
        cat = det.at[j0, "cat"]
        tgt = "" if not isinstance(sfx, str) else str(sfx).replace("\n", " ⏎ ")
        if isinstance(det.at[j0, "path"], str):
            d.text(
                (4, y + 46),
                f'SFX: target [{cat}]: "{tgt}"',
                fill=(200, 0, 0),
                font=font,
            )
        else:
            d.text(
                (4, y + 46),
                f"[{cat}] not written: no training row",
                fill=(150, 150, 150),
                font=font,
            )
        if cat == "pos_mixed":
            d.text(
                (4, y + 72),
                "OCR: target: "
                + (
                    "(row skipped: speech cut)"
                    if not isinstance(sp, str)
                    else f'"{str(sp).replace(chr(10), " ⏎ ")}"'
                ),
                fill=(0, 70, 200),
                font=font,
            )
    d.text(
        (tw - 330, y),
        f"pf {r.pf:.2f}  cf {r.cf:.2f}  IoU {r.iou:.2f}",
        fill=(90, 90, 90),
        font=small,
    )
    return canvas


SPEECH: dict = {}


def sheet(t, sub, det, grey, out, font, small, big, cols=2):
    cards = [card(r, det, grey, font, small) for r in sub.itertuples()]
    if not cards:
        return None
    cw, ch = cards[0].size
    rows = (len(cards) + cols - 1) // cols
    hdr = 70
    S = Image.new(
        "RGB", (cols * (cw + 16) + 16, hdr + rows * (ch + 16) + 8), (238, 238, 238)
    )
    d = ImageDraw.Draw(S)
    d.text((16, 10), TYPES[t], fill="black", font=big)
    d.text(
        (16, 44),
        "context: green COO polygon · red box(es) in question · orange other detector boxes · blue speech boxes     |     detector crop (trained)     |     v6 COO crop",
        fill=(80, 80, 80),
        font=small,
    )
    for i, c in enumerate(cards):
        S.paste(c, (16 + (i % cols) * (cw + 16), hdr + (i // cols) * (ch + 16)))
    p = out / f"sheet_{t}.png"
    S.save(p)
    return p


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--types", nargs="+", default=list(TYPES))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tp = a.out / f"types_{a.split}.parquet"
    if tp.exists():
        det = pd.read_parquet(m109.derived_root() / "manifest_sfxp_det.parquet")
        det = det[det.split == a.split].reset_index(drop=True)
        df = pd.read_parquet(tp)
    else:
        det, df = classify(a.split)
        df.to_parquet(tp, index=False)
    print(df.type.value_counts().to_string())
    g = pd.read_parquet(m109.derived_root() / "manifest.parquet")
    g = g[(g.kind == "sfx") & (g.split == a.split)]
    grey = {(r.book, r.page, r.id): r.path for r in g.itertuples()}
    font = ImageFont.truetype(FONT, 19)
    small = ImageFont.truetype(FONT, 14)
    big = ImageFont.truetype(FONT, 24)
    rng = random.Random(a.seed)
    for t in a.types:
        sub = df[df.type == t]
        if not len(sub):
            continue
        idx = rng.sample(range(len(sub)), min(a.n, len(sub)))
        sub = sub.iloc[idx]
        for book in sub.book.unique():
            for line in m109.iter_text(book):
                q = [int(v) for v in line.poly]
                SPEECH.setdefault((book, line.page), []).append(
                    (q[0], q[1], q[2], q[5])
                )
        p = sheet(t, sub, det, grey, a.out, font, small, big)
        print(f"[sheet] {t}: {len(sub)} → {p}", flush=True)


if __name__ == "__main__":
    main()
