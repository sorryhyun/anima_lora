"""Detector-box SFX crops (``build_sfx_prefix.py --det``) against the COO polygons, by type:
:func:`classify` gives every COO polygon of a split one type (``probes/sfxp_crop_sheet.py``
describes them and draws a sheet per type; ``build_sfx_prefix.py --clean`` drops by them)."""

from __future__ import annotations

import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from . import manga109 as m109
from .crops import margin_box

INCLUDE, GRAZE = 0.8, 0.2


def bbox(pts):
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(u, 1)


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
