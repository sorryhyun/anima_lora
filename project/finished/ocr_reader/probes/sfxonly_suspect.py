#!/usr/bin/env python3
"""Suspicion features for the SFX-only ``OCR2:`` negatives (``crop_dataset.sfx_only``): crops that
may hold text the annotation does not cover, so an empty sub would be wrong. CPU only.

Per crop of ``--list`` (a jsonl with an ``id`` per line: the unreviewed SFX-only train rows):

* ``det_ov``: the largest share of another AnimeText detector box (``manifest_sfxp_det``, any
  category) inside the crop; ``det_ov_other`` the same over ``neg_other`` boxes (no annotation).
* ``ann_gap``: px from the crop rect to the nearest annotated item (COO polygon or ``<text>``
  box) not inside it; ``ann_f`` that item's share inside (under 0.05 it is not in the row's ``part``).
* ``ink_*``: dark marks outside the SFX polygons (dilated by a quarter glyph). ``ink_n``
  counts connected components of glyph-stroke size (longest side 0.2–1.3 glyph, not a solid
  blob), ``ink_edge`` those touching the crop edge, ``ink_frac`` the dark share outside the
  mask, ``cover`` the polygons' share of the crop.

    ANIMA_MANGA109S_ROOT=... python project/finished/ocr_reader/probes/sfxonly_suspect.py \\
        --list output/ocr/eval/sfxonly_train.jsonl --out output/ocr/eval/sfxonly_suspect.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402
from m109_mainsub import INCLUDE, Item, rect_quad  # noqa: E402

DARK = 110


def rect_frac(q, c) -> float:
    iw = max(0.0, min(q[2], c[2]) - max(q[0], c[0]))
    ih = max(0.0, min(q[3], c[3]) - max(q[1], c[1]))
    return iw * ih / max((q[2] - q[0]) * (q[3] - q[1]), 1e-6)


def rect_gap(h: np.ndarray, c) -> float:
    x0, y0 = h.min(0)
    x1, y1 = h.max(0)
    dx = max(c[0] - x1, x0 - c[2], 0.0)
    dy = max(c[1] - y1, y0 - c[3], 0.0)
    return float(np.hypot(dx, dy))


def ink(crop: np.ndarray, polys: list[np.ndarray], off, g: float) -> dict:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    mask = np.zeros((H, W), np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [(p - off).astype(np.int32).reshape(-1, 1, 2)], 1)
    cover = float(mask.mean())
    r = max(2, int(0.25 * g))
    mask = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    )
    dark = (gray < DARK).astype(np.uint8)
    # screentone dots: an open by a small cross keeps strokes
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    out = dark & (1 - mask)
    n, lab, st, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    lo, hi = 0.2 * g, 1.3 * g
    n_g = n_e = 0
    for k in range(1, n):
        x, y, w, h, a = st[k]
        side = max(w, h)
        if not (lo <= side <= hi) or a < 6:
            continue
        if a / max(w * h, 1) > 0.65:  # a solid blob (hair, a fill), not a stroke
            continue
        n_g += 1
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            n_e += 1
    free = max(int((1 - mask).sum()), 1)
    return dict(
        cover=round(cover, 3),
        ink_frac=round(float(out.sum()) / free, 3),
        ink_n=n_g,
        ink_edge=n_e,
    )


def book_feats(args):
    book, rows, det_pages = args
    d = m109.derived_root()
    pages: dict[int, list[Item]] = {}
    for ln in m109.iter_coo(book):
        pages.setdefault(ln.page, []).append(Item(ln, ln.text))
    for ln in m109.iter_text(book):
        pages.setdefault(ln.page, []).append(Item(ln, ln.text))
    out = []
    for r in rows:
        c = r["box"]
        quad = rect_quad(*c)
        items = pages.get(r["page"], [])
        fr = {id(it): it.frac(quad) for it in items}
        inc = [it for it in items if fr[id(it)] >= INCLUDE and it.kind == "sfx"]
        rest = [it for it in items if fr[id(it)] < INCLUDE]
        g = max((it.g for it in inc), default=20.0)
        if rest:
            near = min(rest, key=lambda it: (rect_gap(it.hull, c), -fr[id(it)]))
            gap, af = rect_gap(near.hull, c), fr[id(near)]
        else:
            gap, af = 1e4, 0.0
        dov = dov_o = 0.0
        for did, cat, b in det_pages.get(r["page"], []):
            if did == r["det_id"]:
                continue
            f = rect_frac(b, c)
            dov = max(dov, f)
            if cat == "neg_other":
                dov_o = max(dov_o, f)
        crop = cv2.imread(str(d / r["path"]))
        f_ink = ink(crop, [it.hull for it in inc], np.array(c[:2], np.float32), g)
        out.append(
            dict(
                id=r["id"],
                path=r["path"],
                main=r["main"],
                g=round(g, 1),
                w=c[2] - c[0],
                h=c[3] - c[1],
                det_ov=round(dov, 3),
                det_ov_other=round(dov_o, 3),
                ann_gap=round(gap, 1),
                ann_f=round(af, 3),
                **f_ink,
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ids = {json.loads(line)["id"] for line in open(a.list, encoding="utf-8")}
    ms = []
    for line in open(
        m109.REPO / "output/ocr/mainsub/m109_mainsub.jsonl", encoding="utf-8"
    ):
        x = json.loads(line)
        if x["id"] in ids:
            ms.append(
                dict(
                    id=x["id"],
                    book=x["book"],
                    page=int(x["page"]),
                    path=x["path"],
                    box=x["box"],
                    main=x["main"],
                    det_id=x["id"][len(x["book"]) + 1 :],
                )
            )
    det = pd.read_parquet(m109.derived_root() / "manifest_sfxp_det.parquet")
    det = det[det.split == "train"]
    dp: dict[str, dict[int, list]] = {}
    for b, p, i, cat, box in zip(det.book, det.page, det.id, det.cat, det.box):
        dp.setdefault(b, {}).setdefault(int(p), []).append((i, cat, list(box)))
    by_book: dict[str, list] = {}
    for r in ms:
        by_book.setdefault(r["book"], []).append(r)
    jobs = [(b, rs, dp.get(b, {})) for b, rs in by_book.items()]
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for k, part in enumerate(pool.map(book_feats, jobs)):
            rows += part
            print(f"[{k + 1}/{len(jobs)}] {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(df.describe().T.to_string())
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
