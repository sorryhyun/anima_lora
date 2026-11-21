#!/usr/bin/env python3
"""Manga109 crops as ``OCR:`` main / ``OCR2:`` sub proposals, for the review page
(``ko_label_review.py --mainsub``).

Every crop is an axis-aligned quad in page coordinates. Each COO polygon and
``<text>`` box on the page is scored by the share of its convex hull inside that quad:

* ``≥ 0.8`` — in the crop (``items``): ``main`` = the item with the largest glyph
  (``sqrt(hull area / glyphs)``), with the same-kind items of ≥ 0.7 × that glyph or the
  same text (a repeat) joined in reading order; ``sub`` = the other items, one per line,
  reading order;
* ``0.05 … 0.8`` — partly visible (``part``): shown as a hint (cut text the reviewer
  types as far as it is visible, e.g. a speech edge in an SFX crop). One exception: a
  crop with nothing whole inside takes the speech box ≥ 0.5 in as its main (``trim``: the
  detector split lettering the annotation joins), cut to the lines the crop shows
  (``trim_lines``, ``trim_cut`` = lines dropped); the reviewer checks the rest.

The crop's own label stays as the manifests hold it (line breaks, review edits): a
speech item's text comes from ``manifest_sfxp_speech`` when that row exists, a det crop
whose main is all its SFX keeps ``sfx_text``. Unannotated text (signs, titles, ruby
outside a ``<text>`` box) is invisible here; the page's eye catches it.

One source, ``mainsub_det``: every AnimeText box holding text
(``manifest_sfxp_det_all``, ``build_sfx_prefix.py --det_all``: ``pos`` / ``pos_mixed`` /
``neg_speech``; a ``neg_speech`` main is the speech, or empty when every speech box is
cut), train split by default. main / sub is read on the boxes the detector gives (the
user's call, 2026-11-21): the COO polygons it misses (``manifest_sfxp_missed``) and the
crops cut from the ``<text>`` boxes (``manifest_sfxp_speech``) are out; the speech texts
still come from the latter. Each row keeps the ``<text>`` ids in it (``sp``: the
furigana reads follow them) and, when it holds one ``<text>`` box and no SFX, ``gt_iou``:
the IoU of the detector box with that box at the 4 px margin.

    ANIMA_MANGA109S_ROOT=… python project/finished/ocr_reader/probes/m109_mainsub.py \\
        --out output/ocr/mainsub/m109_mainsub.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402
from build_sfx_prefix import ellipsis_fold, margin_box  # noqa: E402
from ko_label_review import agent_subs  # noqa: E402
from linebreak_auto import (  # noqa: E402
    glyph_weight,
    guess_vertical,
    ink_mask,
    segment,
)

INCLUDE, PART, TRIM = 0.8, 0.05, 0.5
SAME = 0.7
HANGUL = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
SOFT = set("!！?？・…‥~〜ー-―.．、。")
STRIP = "・…~ー〜！!？?"


def rect_quad(x0, y0, x1, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def glyphs(t: str) -> float:
    return max(1.0, sum(0.5 if c in SOFT else 1.0 for c in t if not c.isspace()))


class Item:
    __slots__ = ("kind", "id", "text", "hull", "area", "cx", "cy", "g")

    def __init__(self, line, text: str):
        pts = np.asarray(line.poly, dtype=np.float32).reshape(-1, 2)
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
    share (the page-edge clip makes the quad approximate)."""
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


def trim_lines(it: Item, quad: np.ndarray, page: np.ndarray) -> tuple[str, int]:
    """A ``trim`` main cut to what the crop shows. The ``<text>`` box is cut from the
    page at the 4 px margin and split into columns / rows by ink projection
    (``linebreak_auto.segment``, its eval settings). The text takes the columns by its
    ``\\n`` when the counts agree, or whole when the ink is one column; otherwise it is
    left whole (an unbroken label spread over ink lengths lands on the wrong columns:
    BakuretsuKungFuGirl_024_29, PlatinumJungle_021_04). A column stays when half its width is inside the crop; along it, the glyphs spread over
    the column's ink by ``glyph_weight`` and a glyph stays when its centre is inside.
    Returns (text, glyphs dropped), the text unchanged when it is left whole or nothing
    drops."""
    x0, y0 = it.hull.min(0)
    x1, y1 = it.hull.max(0)
    H, W = page.shape[:2]
    bx0, by0, bx1, by1 = margin_box(x0, y0, x1, y1, W, H)
    box = page[by0:by1, bx0:bx1]
    text = it.text.strip()
    if box.size == 0 or not text:
        return it.text, 0
    vert = guess_vertical(box, text)
    segs = segment(ink_mask(box), vert, 0.03, 0.55, 0.0)
    if not segs:
        return it.text, 0
    lines = [s.strip() for s in text.split("\n") if s.strip()]
    if len(segs) == 1:
        lines = ["".join(lines)]
    elif len(lines) != len(segs):
        return it.text, 0
    (cx0, cy0), (cx1, cy1) = quad.min(0), quad.max(0)
    # a = across the columns, b = along them, in page coordinates
    ao, bo, c0, c1, d0, d1 = (
        (bx0, by0, cx0, cx1, cy0, cy1) if vert else (by0, bx0, cy0, cy1, cx0, cx1)
    )
    keep, dropped = [], 0
    for s, (a0, a1, b0, b1) in zip(lines, segs):
        if min(a1 + ao, c1) - max(a0 + ao, c0) < 0.5 * (a1 - a0):
            dropped += sum(not c.isspace() for c in s)
            continue
        w = np.array([glyph_weight(c) for c in s])
        mid = b0 + bo + (np.cumsum(w) - w / 2) / w.sum() * (b1 - b0)
        vis = "".join(c for c, m in zip(s, mid) if d0 <= m <= d1).strip()
        dropped += sum(not c.isspace() for c, m in zip(s, mid) if not d0 <= m <= d1)
        if vis:
            keep.append(vis)
    if not keep or not dropped:
        return it.text, 0
    return "\n".join(keep), dropped


def ruby_text(groups: list) -> str:
    """The reads of a box's ruby groups, one line per main column in reading order, a
    column's groups top down joined by a space (as the review page's furigana source)."""
    cols: dict[int, list] = {}
    for g in sorted(groups, key=lambda g: (g["col"], g["y0"])):
        if g.get("read"):
            cols.setdefault(g["col"], []).append(g["read"])
    return "\n".join(" ".join(v) for _, v in sorted(cols.items()))


def visible_ruby(it: Item, quad: np.ndarray, page: np.ndarray, groups: list) -> str:
    """The ruby reads of the ``<text>`` box ``it`` whose group centre lies in the crop,
    one line per main column in reading order (as the furigana source joins them).
    Group boxes are in the box's 4 px cut, whose origin ``margin_box`` gives."""
    x0, y0 = it.hull.min(0)
    x1, y1 = it.hull.max(0)
    H, W = page.shape[:2]
    bx0, by0, _, _ = margin_box(x0, y0, x1, y1, W, H)
    (cx0, cy0), (cx1, cy1) = quad.min(0), quad.max(0)
    cols: dict[int, list] = {}
    for g in sorted(groups, key=lambda g: (g["col"], g["y0"])):
        mx = bx0 + (g["x0"] + g["x1"]) / 2
        my = by0 + (g["y0"] + g["y1"]) / 2
        if g.get("read") and cx0 <= mx <= cx1 and cy0 <= my <= cy1:
            cols.setdefault(g["col"], []).append(g["read"])
    return "\n".join(" ".join(v) for _, v in sorted(cols.items()))


def gt_iou(quad: np.ndarray, inc: list[Item]) -> float | None:
    """IoU of the crop's box with its one ``<text>`` box at the 4 px margin (the
    ``manifest_sfxp_speech`` cut), ``None`` unless the crop holds exactly that."""
    if len(inc) != 1 or inc[0].kind != "speech":
        return None
    x0, y0 = inc[0].hull.min(0)
    x1, y1 = inc[0].hull.max(0)
    p = min(0.12 * max(x1 - x0, y1 - y0), 0.3 * min(x1 - x0, y1 - y0), 4)
    g = (x0 - p, y0 - p, x1 + p, y1 + p)
    c = (*quad.min(0), *quad.max(0))
    iw = max(0.0, min(g[2], c[2]) - max(g[0], c[0]))
    ih = max(0.0, min(g[3], c[3]) - max(g[1], c[1]))
    i = iw * ih
    u = (g[2] - g[0]) * (g[3] - g[1]) + (c[2] - c[0]) * (c[3] - c[1]) - i
    return round(float(i / max(u, 1e-6)), 3)


def book_rows(args) -> list[dict]:
    book, splits, tables = args
    sp_text, ruby, fixed = tables["sp_text"], tables["ruby"], tables["fixed"]
    pages: dict[int, list[Item]] = {}
    for ln in m109.iter_coo(book):
        pages.setdefault(ln.page, []).append(Item(ln, ln.text))
    for ln in m109.iter_text(book):
        # dots as the JA speech labels spell them (``ellipsis_fold``, idempotent): the
        # <text> boxes outside manifest_sfxp_speech carry the raw annotation. Korean
        # keeps ``...`` and ``…`` apart (the user, 11-21): a text with Hangul is left as is
        t = sp_text.get((book, ln.page, str(ln.id)), ln.text)
        if not HANGUL.search(t):
            t = ellipsis_fold(t)
        pages.setdefault(ln.page, []).append(Item(ln, t))
    out = []
    imgs: dict[int, np.ndarray] = {}

    def page_img(p: int) -> np.ndarray:
        if p not in imgs:
            imgs[p] = cv2.imread(str(m109.page_path(book, p)))
        return imgs[p]

    def emit(src, r, quad, own, own_text, cat, extra=None):
        items = pages.get(int(r["page"]), [])
        fr = {it.id: it.frac(quad) for it in items}
        main, sub, inc, part = propose(items, fr, own, own_text)
        # nothing whole inside but a speech box half in (the detector split what the
        # annotation joined): its text as main, for the reviewer to cut down
        trim, cut = None, 0
        if not inc:
            half = [it for it in part if it.kind == "speech" and fr[it.id] >= TRIM]
            if half:
                trim = max(half, key=lambda it: fr[it.id])
                main, cut = trim_lines(trim, quad, page_img(int(r["page"])))
                rk = (book, int(r["page"]), trim.id)
                if rk in ruby:
                    extra = dict(extra or {})
                    extra["trim_ruby"] = visible_ruby(
                        trim, quad, page_img(int(r["page"])), ruby[rk]
                    )
        stem = Path(r["path"]).stem
        sp = [it.id for it in order(inc) if it.kind == "speech"] + (
            [trim.id] if trim else []
        )
        # the sub proposal with the ruby first: per <text> box the agents' fix where they
        # reviewed it, else the reads; a trim crop the ruby inside it
        heads = []
        for t in sp:
            rid = f"{book}_{int(r['page']):03d}_{t}"
            if trim is not None and t == trim.id:
                heads.append((extra or {}).get("trim_ruby", ""))
            elif rid in fixed:
                heads.append(fixed[rid])
            elif (book, int(r["page"]), t) in ruby:
                heads.append(ruby_text(ruby[(book, int(r["page"]), t)]))
        subp = "\n".join([h for h in heads if h] + ([sub] if sub else []))
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
                sp=sp,
                subp=subp,
                trim=int(trim is not None),
                trim_cut=cut,
                gt_iou=gt_iou(quad, inc),
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
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--ruby",
        type=Path,
        nargs="*",
        default=[],
        help="furigana_find.py read outputs: the ruby reads head each row's sub "
        "proposal (subp); a trim row takes the ruby inside its crop",
    )
    ap.add_argument(
        "--ruby_fix",
        type=Path,
        help="agent outputs (furigana_find.py batches, **/out*.tsv): their sub replaces "
        "the reads of the boxes they reviewed",
    )
    a = ap.parse_args()
    d = m109.derived_root()
    det = pd.read_parquet(d / "manifest_sfxp_det_all.parquet")
    det = det[det.split.isin(a.splits)]
    det = det.assign(box=det.box.map(lambda b: [int(v) for v in b]))
    speech = pd.read_parquet(d / "manifest_sfxp_speech.parquet")
    sp_text = {
        (b, int(p), str(i)): t
        for b, p, i, t, s in zip(
            speech.book, speech.page, speech.id, speech.text, speech.source
        )
        if s in ("sfxp_speech", "m109_lines")
    }
    # the <text> boxes outside the speech manifest (mainsub_text.py, line breaks from
    # linebreak_auto.py apply --set mainsub_text); the speech manifest wins
    mt = d / "manifest_mainsub_text.parquet"
    if mt.exists():
        extra = pd.read_parquet(mt)
        for b, p, i, t in zip(extra.book, extra.page, extra.id, extra.text):
            sp_text.setdefault((b, int(p), str(i)), t)
    # furigana_find.py read files: the ruby groups per <text> box, for trim rows
    fixed = {k: v[0] for k, v in agent_subs(a.ruby_fix).items()}
    ruby: dict[tuple, list] = {}
    for rp in a.ruby:
        for line in open(rp, encoding="utf-8"):
            x = json.loads(line)
            if x["groups"]:
                ruby[(x["book"], int(x["page"]), str(x["id"]))] = x["groups"]

    def by_book(df, cols):
        return {
            b: g[cols].to_dict("records") for b, g in df.groupby("book", sort=False)
        }

    det_b = by_book(det, ["split", "id", "page", "path", "cat", "box", "sfx_text"])
    books = sorted(det_b)
    jobs = [
        (
            b,
            set(a.splits),
            dict(
                det=det_b[b],
                sp_text={k: v for k, v in sp_text.items() if k[0] == b},
                ruby={k: v for k, v in ruby.items() if k[0] == b},
                fixed={k: v for k, v in fixed.items() if k.startswith(b + "_")},
            ),
        )
        for b in books
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n: dict[str, list[int]] = {}
    ious = []
    with (
        ProcessPoolExecutor(a.workers) as ex,
        open(a.out, "w", encoding="utf-8") as f,
    ):
        for rows in ex.map(book_rows, jobs):
            for r in rows:
                c = n.setdefault(r["source"], [0, 0, 0, 0, 0])
                c[0] += 1
                c[1] += bool(r["sub"])
                c[2] += bool(r["part"])
                c[3] += r["main"] != r["label"]
                c[4] += r["trim"]
                if r["gt_iou"] is not None:
                    ious.append(r["gt_iou"])
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s, (tot, sub, part, moved, trim) in n.items():
        print(
            f"[mainsub] {s}: {tot} crops, sub {sub}, partly visible {part}, "
            f"main ≠ label {moved}, main to trim {trim}",
            flush=True,
        )
    q = np.asarray(ious)
    if len(q):
        print(
            f"[mainsub] one <text> box, no SFX: {len(q)} crops; IoU with the 4 px "
            f"<text> cut: median {np.median(q):.3f}, ≥ 0.9 {np.mean(q >= 0.9):.1%}, "
            f"≥ 0.8 {np.mean(q >= 0.8):.1%}, < 0.5 {np.mean(q < 0.5):.1%}",
            flush=True,
        )
    print(f"[mainsub] → {a.out}", flush=True)


if __name__ == "__main__":
    main()
