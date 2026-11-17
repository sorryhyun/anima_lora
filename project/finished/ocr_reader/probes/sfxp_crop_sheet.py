#!/usr/bin/env python3
"""Detector-box SFX crops (``build_sfx_prefix.py --det``) against the COO polygons, by type.

Every train COO polygon is matched to the AnimeText boxes of its page (``pf`` = share of the
polygon inside the crop, ``cf`` = share of the crop it covers, the build's rule) and gets one
type; neg_speech boxes written as ``SFX:`` negatives get a type of their own when a COO
polygon reaches into them. Writes ``types.parquet`` and one contact sheet per type:

    page context (green COO polygon, red the box(es) in question, orange other detector
    boxes, blue Manga109 speech boxes) | the detector crop as trained | v6's COO crop

    ANIMA_MANGA109S_ROOT=... python project/finished/ocr_reader/probes/sfxp_crop_sheet.py \\
        --out <dir> [--n 12] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
import manga109 as m109  # noqa: E402

INCLUDE, GRAZE = 0.8, 0.2
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


def bbox(pts):
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(u, 1)


def margin_box(x0, y0, x1, y1, W, H):
    w, h = x1 - x0, y1 - y0
    p = min(0.12 * max(w, h), 0.3 * min(w, h), 4)
    return (
        int(round(max(0, x0 - p))),
        int(round(max(0, y0 - p))),
        int(round(min(W, x1 + p))),
        int(round(min(H, y1 + p))),
    )


class Poly:
    """One COO line: its mask within its own bbox (clipped to the page)."""

    def __init__(self, line, W, H):
        a = np.asarray(line.poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        if line.joined:
            a = cv2.convexHull(a).reshape(-1, 2)
        self.line, self.pts = line, a
        x0, y0, x1, y1 = bbox(a)
        self.x0, self.y0 = max(0, x0), max(0, y0)
        self.x1, self.y1 = min(W, x1 + 1), min(H, y1 + 1)
        mk = np.zeros((max(self.y1 - self.y0, 1), max(self.x1 - self.x0, 1)), np.uint8)
        cv2.fillPoly(mk, [(a - [self.x0, self.y0]).reshape(-1, 1, 2)], 1)
        self.mk = mk.astype(bool)
        self.area = int(self.mk.sum())
        self.cbox = margin_box(x0, y0, x1, y1, W, H)

    def inside(self, c):
        x0, y0 = max(c[0], self.x0), max(c[1], self.y0)
        x1, y1 = min(c[2], self.x1), min(c[3], self.y1)
        if x1 <= x0 or y1 <= y0:
            return 0
        return int(
            self.mk[y0 - self.y0 : y1 - self.y0, x0 - self.x0 : x1 - self.x0].sum()
        )


def classify(split: str):
    det = pd.read_parquet(m109.derived_root() / "manifest_sfxp_det.parquet")
    det = det[det.split == split].reset_index(drop=True)
    boxes_by = defaultdict(list)
    for i, r in enumerate(det.itertuples()):
        boxes_by[(r.book, r.page)].append(i)
    rows, neg_rows = [], []
    books = sorted(det.book.unique())
    for bi, book in enumerate(books):
        coo_by, txt_by = defaultdict(list), defaultdict(list)
        for line in m109.iter_coo(book):
            coo_by[line.page].append(line)
        for line in m109.iter_text(book):
            txt_by[line.page].append(line)
        pages = sorted({p for (b, p) in boxes_by if b == book} | set(coo_by))
        for pg in pages:
            W, H = Image.open(m109.page_path(book, pg)).size
            polys = [Poly(line, W, H) for line in coo_by.get(pg, [])]
            polys = [p for p in polys if p.area]
            bidx = boxes_by.get((book, pg), [])
            # overlaps[pi] = [(box index, pf, cf)]
            ov = [[] for _ in polys]
            box_inc = defaultdict(list)  # box -> polys with pf >= INCLUDE
            box_graze = defaultdict(
                list
            )  # box -> polys with 0 < pf < GRAZE, cf < GRAZE
            for j in bidx:
                c = det.at[j, "box"]
                ca = max((c[2] - c[0]) * (c[3] - c[1]), 1)
                for pi, p in enumerate(polys):
                    ins = p.inside(c)
                    if not ins:
                        continue
                    pf, cf = ins / p.area, ins / ca
                    ov[pi].append((j, pf, cf))
                    if pf >= INCLUDE:
                        box_inc[j].append(pi)
                    elif pf < GRAZE and cf < GRAZE:
                        box_graze[j].append(pi)
            for pi, p in enumerate(polys):
                kept = [
                    o
                    for o in ov[pi]
                    if o[1] >= INCLUDE and det.at[o[0], "cat"] in ("pos", "pos_mixed")
                ]
                full = [o for o in ov[pi] if o[1] >= INCLUDE]
                best = max(ov[pi], key=lambda o: o[1], default=(None, 0.0, 0.0))
                j_show = None
                if not kept:
                    if full:
                        t, j_show = "drop_nb", full[0][0]
                    elif any(o[1] >= GRAZE or o[2] >= GRAZE for o in ov[pi]):
                        t, j_show = "fragment", best[0]
                    else:
                        t, j_show = "missed", best[0]
                    pf = best[1] if t != "drop_nb" else full[0][1]
                    cf = best[2] if t != "drop_nb" else full[0][2]
                    jj = [o[0] for o in ov[pi] if o[1] >= GRAZE or o[2] >= GRAZE] or (
                        [j_show] if j_show is not None else []
                    )
                else:
                    j, pf, cf = max(kept, key=lambda o: o[1])
                    j_show, jj = j, [o[0] for o in kept]
                    cat = det.at[j, "cat"]
                    if len(kept) > 1:
                        t = "dup"
                    elif cat == "pos_mixed":
                        t = "mixed"
                    elif len(box_inc[j]) > 1:
                        t = "merged"
                    elif pf < 0.9:
                        t = "cut"
                    elif box_graze[j]:
                        t = "graze"
                    elif iou(det.at[j, "box"], p.cbox) < 0.6:
                        t = "loose"
                    else:
                        t = "same"
                rows.append(
                    dict(
                        book=book,
                        page=pg,
                        coo_id=p.line.id,
                        coo_text=p.line.text,
                        joined=p.line.joined,
                        type=t,
                        pf=pf,
                        cf=cf,
                        iou=iou(det.at[j_show, "box"], p.cbox)
                        if j_show is not None
                        else 0.0,
                        det_rows=json.dumps([int(x) for x in jj]),
                        poly=json.dumps(p.pts.reshape(-1).tolist()),
                        cbox=list(p.cbox),
                    )
                )
            for j in bidx:
                if (
                    det.at[j, "cat"] == "neg_speech"
                    and isinstance(det.at[j, "path"], str)
                    and box_graze[j]
                ):
                    gp = [polys[pi] for pi in box_graze[j]]
                    c = det.at[j, "box"]
                    ca = max((c[2] - c[0]) * (c[3] - c[1]), 1)
                    neg_rows.append(
                        dict(
                            book=book,
                            page=pg,
                            coo_id="|".join(p.line.id for p in gp),
                            coo_text=" / ".join(p.line.text for p in gp),
                            joined=False,
                            type="neg_graze",
                            pf=max(p.inside(c) / p.area for p in gp),
                            cf=max(p.inside(c) / ca for p in gp),
                            iou=0.0,
                            det_rows=json.dumps([int(j)]),
                            poly=json.dumps(gp[0].pts.reshape(-1).tolist()),
                            cbox=list(gp[0].cbox),
                        )
                    )
        print(f"[classify] {bi + 1}/{len(books)} {book}: {len(rows)} polys", flush=True)
    return det, pd.DataFrame(rows + neg_rows)


# --------------------------------------------------------------------------- sheets


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
