#!/usr/bin/env python3
"""Manga109 crops as ``OCR:`` main / ``OCR2:`` sub proposals, for the review page
(``ko_label_review.py --mainsub``).

Every crop is a quad in page coordinates (an axis-aligned box for the detector and speech
crops, ``pilot.deskew_crop``'s rotated rect for the missed polygons). Each COO polygon and
``<text>`` box on the page is scored by the share of its convex hull inside that quad:

* ``≥ 0.8`` — in the crop (``items``): ``main`` = the item with the largest glyph
  (``sqrt(hull area / glyphs)``), with the same-kind items of ≥ 0.7 × that glyph or the
  same text (a repeat) joined in reading order; ``sub`` = the other items, one per line,
  reading order;
* ``0.05 … 0.8`` — partly visible (``part``): never written into a label, shown as a
  hint (cut text the reviewer types as far as it is visible, e.g. a speech edge in an SFX
  crop).

The crop's own label stays as the manifests hold it (line breaks, review edits): a
speech item's text comes from ``manifest_sfxp_speech`` when that row exists, a det crop
whose main is all its SFX keeps ``sfx_text``. Unannotated text (signs, titles, ruby
outside a ``<text>`` box) is invisible here; the page's eye catches it.

Sources: every Manga109 crop the two-prompt mix trains on (``crop_dataset.load_sfxp``),
train split by default:

* ``mainsub_det`` — ``manifest_sfxp_det_clean``'s written crops: ``pos`` / ``pos_mixed``
  (the ``SFX:`` rows) and ``neg_speech`` (the ``SFX:`` negative pool; its main is the
  speech, or empty when every speech box is cut);
* ``mainsub_speech`` — ``manifest_sfxp_speech`` (``sfxp_speech`` and ``m109_lines``: the
  ``OCR:`` rows);
* ``mainsub_missed`` — ``manifest_sfxp_missed``'s cut polygons (``--sfxp_missed`` rows).

    ANIMA_MANGA109S_ROOT=… python project/finished/ocr_reader/probes/m109_mainsub.py \\
        --out output/ocr/mainsub/m109_mainsub.jsonl
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
import manga109 as m109  # noqa: E402

INCLUDE, PART = 0.8, 0.05
SAME = 0.7
SOFT = set("!！?？・…‥~〜ー-―.．、。")
STRIP = "・…~ー〜！!？?"


def rot_rect(poly, pad_frac: float) -> np.ndarray:
    """``pilot.deskew_crop``'s crop region (its ``minAreaRect`` + margin) as a page quad."""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if angle > 45:
        angle -= 90
        w, h = h, w
    w, h = max(w, 1.0), max(h, 1.0)
    p = pad_frac * max(w, h)
    return cv2.boxPoints(((cx, cy), (w + 2 * p, h + 2 * p), angle)).astype(np.float32)


def cap_frac(poly) -> float:
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    _, (w, h), _ = cv2.minAreaRect(pts)
    L, S = max(w, h, 1.0), max(min(w, h), 1.0)
    return min(0.12 * L, 0.3 * S, 4) / L


def rect_quad(x0, y0, x1, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def glyphs(t: str) -> float:
    return max(1.0, sum(0.5 if c in SOFT else 1.0 for c in t if not c.isspace()))


class Item:
    __slots__ = ("kind", "id", "text", "pts", "hull", "area", "cx", "cy", "g")

    def __init__(self, line, text: str):
        pts = np.asarray(line.poly, dtype=np.float32).reshape(-1, 2)
        self.pts = pts
        self.hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)
        self.area = max(float(cv2.contourArea(self.hull)), 1.0)
        self.cx, self.cy = (float(v) for v in pts.mean(0))
        self.kind, self.id, self.text = line.kind, str(line.id), text
        self.g = float(np.sqrt(self.area / glyphs(text)))

    def frac(self, quad: np.ndarray) -> float:
        a, _ = cv2.intersectConvexConvex(self.hull, quad)
        return max(0.0, float(a)) / self.area


def order(items):
    return sorted(items, key=lambda it: (-it.cx, it.cy))


def propose(items: list[Item], fr: dict, own: set[str], own_text: str | None):
    """(main, sub, inc, part) for one crop; ``own`` ids count as inside whatever their
    share (the trim and the deskew make the quad approximate)."""
    inc = [it for it in items if it.id in own or fr[it.id] >= INCLUDE]
    part = [it for it in items if it not in inc and fr[it.id] >= PART and it.text]
    inc = [it for it in inc if it.text]
    if not inc:
        return own_text or "", "", inc, part
    top = max(inc, key=lambda it: it.g)
    main = [it for it in inc if it.kind == top.kind and it.g >= SAME * top.g]
    # a repeat of a main text (あ~ ×3, ザワ ×3) is the same lettering at any size
    reps = {it.text.strip(STRIP) for it in main}
    main += [
        it
        for it in inc
        if it not in main and it.kind == top.kind and it.text.strip(STRIP) in reps
    ]
    rest = [it for it in inc if it not in main]
    if own_text is not None and {it.id for it in main} == own:
        main_text = own_text
    else:
        main_text = "\n".join(it.text for it in order(main))
    sub = "\n".join(it.text for it in order(rest))
    return main_text, sub, inc, part


def book_rows(args) -> list[dict]:
    book, splits, tables = args
    sp_text = tables["sp_text"]
    pages: dict[int, list[Item]] = {}
    for ln in m109.iter_coo(book):
        pages.setdefault(ln.page, []).append(Item(ln, ln.text))
    for ln in m109.iter_text(book):
        t = sp_text.get((book, ln.page, str(ln.id)), ln.text)
        pages.setdefault(ln.page, []).append(Item(ln, t))
    out = []

    def emit(src, r, quad, own, own_text, cat, extra=None):
        items = pages.get(int(r["page"]), [])
        fr = {it.id: it.frac(quad) for it in items}
        main, sub, inc, part = propose(items, fr, own, own_text)
        stem = Path(r["path"]).stem
        out.append(
            dict(
                source=src,
                id=stem,
                split=r["split"],
                cat=cat,
                book=book,
                page=int(r["page"]),
                path=r["path"],
                main=main,
                sub=sub,
                label=own_text if own_text is not None else "",
                items=[
                    dict(k=it.kind, t=it.text, g=round(it.g, 1), f=round(fr[it.id], 2))
                    for it in order(inc)
                ],
                part=[
                    dict(k=it.kind, t=it.text, f=round(fr[it.id], 2))
                    for it in order(part)
                ],
                box=[int(v) for v in quad.min(0)] + [int(v) for v in quad.max(0)],
                **(extra or {}),
            )
        )

    def own_sfx(items, quad):
        return {
            it.id
            for it in items
            if it.kind == "sfx" and it.frac(quad) >= INCLUDE and it.text
        }

    for r in tables["det"]:
        if r["split"] not in splits:
            continue
        quad = rect_quad(*r["box"])
        items = pages.get(int(r["page"]), [])
        label = r["sfx_text"] if isinstance(r["sfx_text"], str) else None
        emit("mainsub_det", r, quad, own_sfx(items, quad), label, r["cat"])
    for r in tables["missed"]:
        items = pages.get(int(r["page"]), [])
        it = next((x for x in items if x.kind == "sfx" and x.id == str(r["id"])), None)
        if it is None:
            continue
        quad = rot_rect(it.pts, cap_frac(it.pts))
        emit("mainsub_missed", r, quad, {it.id}, r["text"], r["source"])
    for r in tables["speech"]:
        if r["split"] not in splits:
            continue
        items = pages.get(int(r["page"]), [])
        own_it = next(
            (it for it in items if it.kind == "speech" and it.id == str(r["id"])), None
        )
        if own_it is None:
            continue
        x0, y0 = own_it.hull.min(0)
        x1, y1 = own_it.hull.max(0)
        p = min(0.12 * max(x1 - x0, y1 - y0), 0.3 * min(x1 - x0, y1 - y0), 4)
        emit(
            "mainsub_speech",
            r,
            rect_quad(x0 - p, y0 - p, x1 + p, y1 + p),
            {own_it.id},
            r["text"],
            r["source"],
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    d = m109.derived_root()
    det = pd.read_parquet(d / "manifest_sfxp_det_clean.parquet")
    det = det[det.path.notna() & det.split.isin(a.splits)]
    missed = pd.read_parquet(d / "manifest_sfxp_missed.parquet")
    missed = missed[missed.path.notna() & missed.split.isin(a.splits)]
    det = det.assign(box=det.box.map(lambda b: [int(v) for v in b]))
    speech = pd.read_parquet(d / "manifest_sfxp_speech.parquet")
    sp_text = {
        (b, int(p), str(i)): t
        for b, p, i, t, s in zip(
            speech.book, speech.page, speech.id, speech.text, speech.source
        )
        if s in ("sfxp_speech", "m109_lines")
    }
    speech = speech[
        speech.source.isin(["sfxp_speech", "m109_lines"]) & speech.split.isin(a.splits)
    ]

    def by_book(df, cols):
        return {
            b: g[cols].to_dict("records") for b, g in df.groupby("book", sort=False)
        }

    base = ["split", "id", "page", "path", "text"]
    det_b = by_book(det, ["split", "id", "page", "path", "cat", "box", "sfx_text"])
    mi_b = by_book(missed, base + ["source"])
    sp_b = by_book(speech, base + ["source"])
    books = sorted(set(det_b) | set(mi_b) | set(sp_b))
    jobs = [
        (
            b,
            set(a.splits),
            dict(
                det=det_b.get(b, []),
                missed=mi_b.get(b, []),
                speech=sp_b.get(b, []),
                sp_text={k: v for k, v in sp_text.items() if k[0] == b},
            ),
        )
        for b in books
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n: dict[str, list[int]] = {}
    with (
        ProcessPoolExecutor(a.workers) as ex,
        open(a.out, "w", encoding="utf-8") as f,
    ):
        for rows in ex.map(book_rows, jobs):
            for r in rows:
                c = n.setdefault(r["source"], [0, 0, 0, 0])
                c[0] += 1
                c[1] += bool(r["sub"])
                c[2] += bool(r["part"])
                c[3] += r["main"] != r["label"]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s, (tot, sub, part, moved) in n.items():
        print(
            f"[mainsub] {s}: {tot} crops, sub {sub}, partly visible {part}, "
            f"main ≠ label {moved}",
            flush=True,
        )
    print(f"[mainsub] → {a.out}", flush=True)


if __name__ == "__main__":
    main()
