#!/usr/bin/env python3
"""Automatic line breaks for one-line Manga109 speech labels, measured on the community
line annotations (``assets/manga109s_line_annotations.json``, 5,988 ``<text>`` boxes).

The crop is cut as ``build_sfx_prefix.build_speech`` cuts it (the ``<text>`` box at the
4 px margin). Columns (vertical text) or rows (horizontal) come from the ink projection:
binarise (Otsu), drop the ink components touching the crop edge, sum ink across the reading
axis, split at empty runs, drop slivers (furigana) narrower than ``--sliver`` × the widest
column. The label is cut
in proportion to each column's ink length, then the cut is moved off a character that
may not start a line (small kana, prolonged sound mark, closing punctuation) and onto a
nearby space.

    ANIMA_MANGA109S_ROOT=… .venv/bin/python project/ocr_reader/labeling/linebreak_auto.py eval
    … make daemon-run ARGS="… linebreak_auto.py read --name stock [--ckpt output/ocr/<run>/best]"
    … linebreak_auto.py transfer --name stock

``read`` runs a PaddleOCR-VL reader (``core.readers.Vl16Reader``, ``OCR:``) over the same
crops and stores its reads (``output/ocr/linebreak/<name>.jsonl``); ``transfer`` aligns
each read to the label (edit-distance DP on NFKC-folded characters) and carries the
read's line breaks onto the label, then scores against the annotation.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter

import cv2
import numpy as np

from project.ocr_reader.core import manga109 as m109
from project.ocr_reader.core.crops import (
    DOTS,
    SPACES,
    glyph_weight,
    guess_vertical,
    ink_mask,
    margin_box,
    segment,
)
from project.ocr_reader.core.labels import ellipsis_fold, line_broken
from project.ocr_reader.core.paths import LINE_ANNOTATIONS as LINES
from project.ocr_reader.core.paths import OUTPUT
from project.ocr_reader.core.readers import MAX_NEW_TOKENS, Vl16Reader

NO_START = set(
    "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶーｰ～~、。，．,.!！?？)）」』】〉》…‥・ﾞﾟ゛゜"
)


def assign(text: str, segs, vertical: bool, snap: int) -> str:
    flat = text.replace("\n", "")
    n = len(flat)
    if len(segs) <= 1 or n <= 1:
        return flat.strip()
    lens = np.array([s[3] - s[2] for s in segs], dtype=float)
    wts = np.array([glyph_weight(ch) for ch in flat])
    cw = np.concatenate([[0.0], np.cumsum(wts)])
    targets = np.cumsum(lens)[:-1] / lens.sum() * cw[-1]
    cuts, prev = [], 0
    for c in targets:
        # the cut whose preceding weight is nearest the column boundary
        k = int(np.argmin(np.abs(cw[1:-1] - c))) + 1 if n > 1 else 1
        k = min(max(k, prev + 1), n - 1)
        # a space within reach is the break itself
        near = [
            j for j in range(k - snap, k + snap + 1) if 0 < j < n and flat[j] in SPACES
        ]
        if near:
            k = min(near, key=lambda j: abs(j - k))
        else:
            while k < n - 1 and flat[k] in NO_START:
                k += 1  # the mark stays at the bottom of the column above
        if k <= prev:
            continue
        cuts.append(k)
        prev = k
    parts, i = [], 0
    for k in cuts + [n]:
        parts.append(flat[i:k].strip())
        i = k
    return "\n".join(p for p in parts if p)


def gt_set():
    ann = {}
    for pg in json.loads(LINES.read_text(encoding="utf-8")).values():
        for t in pg["texts"]:
            ann[(pg["book"], int(pg["page_index"]), t["id"])] = t
    by_page: dict[tuple[str, int], list] = {}
    for book in sorted({k[0] for k in ann}):
        root = ET.parse(m109.manga109s_root() / "annotations" / f"{book}.xml").getroot()
        for p in root.iter("page"):
            for e in p.findall("text"):
                k = (book, int(p.get("index")), e.get("id"))
                if k in ann:
                    by_page.setdefault(k[:2], []).append((k, e, ann[k]))
    return by_page


def evaluate(a):
    by_page = gt_set()
    tally = Counter()
    misses = []
    for (book, page), items in sorted(by_page.items()):
        img = cv2.imread(str(m109.page_path(book, page)))
        H, W = img.shape[:2]
        for k, e, t in items:
            x0, y0, x1, y1 = (int(e.get(c)) for c in ("xmin", "ymin", "xmax", "ymax"))
            c = margin_box(x0, y0, x1, y1, W, H)
            crop = img[c[1] : c[3], c[0] : c[2]]
            raw = e.text or ""
            gt = line_broken(raw, t["line_lengths"])
            ls = t["lines"]
            tall = sum(
                (ln["ymax"] - ln["ymin"]) > (ln["xmax"] - ln["xmin"]) for ln in ls
            )
            vertical_gt = 2 * tall >= len(ls)
            vertical = vertical_gt if a.oracle_orient else guess_vertical(crop, raw)
            segs = segment(ink_mask(crop), vertical, a.gap, a.sliver, a.merge)
            pred = assign(raw, segs, vertical, a.snap)
            n_gt = gt.count("\n") + 1
            n_pr = pred.count("\n") + 1
            tally["n"] += 1
            tally["single_gt"] += n_gt == 1
            tally["orient_ok"] += vertical == vertical_gt
            tally["count_ok"] += n_gt == n_pr
            tally["exact"] += pred == gt
            if n_gt > 1 and n_gt == n_pr:
                tally["alloc_n"] += 1
                tally["alloc_ok"] += pred == gt
            if n_gt > 1:
                tally["multi"] += 1
                tally["multi_count_ok"] += n_gt == n_pr
                tally["multi_exact"] += pred == gt
            tally[
                "count_diff_" + ("lt" if n_pr < n_gt else "gt" if n_pr > n_gt else "eq")
            ] += 1
            if pred != gt and len(misses) < a.show:
                misses.append((f"{book}_{page:03d}_{k[2]}", gt, pred, len(segs)))
    n = tally["n"]
    print(
        f"n {n} (single-line {tally['single_gt']}); orient {tally['orient_ok'] / n:.3f}; "
        f"line count {tally['count_ok'] / n:.3f} (fewer {tally['count_diff_lt']}, more "
        f"{tally['count_diff_gt']}); exact {tally['exact'] / n:.3f}"
    )
    mm = tally["multi"]
    print(
        f"multi-line GT {mm}: line count {tally['multi_count_ok'] / mm:.3f}, "
        f"exact {tally['multi_exact'] / mm:.3f}; exact when the count is right "
        f"{tally['alloc_ok'] / max(1, tally['alloc_n']):.3f} ({tally['alloc_n']})"
    )
    for mid, gt, pred, ns in misses:
        print(f"  {mid}  segs {ns}\n    gt   {gt!r}\n    pred {pred!r}")


OUT = OUTPUT / "linebreak"


def gt_crops():
    """(key, raw text, gt broken text, crop) for every annotated ``<text>``."""
    for (book, page), items in sorted(gt_set().items()):
        img = cv2.imread(str(m109.page_path(book, page)))
        H, W = img.shape[:2]
        for k, e, t in items:
            x0, y0, x1, y1 = (int(e.get(c)) for c in ("xmin", "ymin", "xmax", "ymax"))
            c = margin_box(x0, y0, x1, y1, W, H)
            raw = e.text or ""
            yield (
                k,
                raw,
                line_broken(raw, t["line_lengths"]),
                img[c[1] : c[3], c[0] : c[2]],
            )


def read(a):
    rd = Vl16Reader(a.ckpt, "cuda")
    rows = list(gt_crops())
    preds = rd.read([r[3] for r in rows], None, a.bs)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{a.name}.jsonl", "w", encoding="utf-8") as f:
        for (k, raw, gt, _), p in zip(rows, preds):
            f.write(
                json.dumps(
                    dict(book=k[0], page=k[1], id=k[2], raw=raw, gt=gt, pred=p),
                    ensure_ascii=False,
                )
                + "\n"
            )
    n_lf = sum("\n" in p for p in preds)
    print(
        f"wrote {OUT / (a.name + '.jsonl')}: {len(preds)} reads, {n_lf} with a line break"
    )


def boxes_in(boxes, x0, y0, x1, y1, pad: int, inside: float) -> list:
    """Detector boxes with ≥ ``inside`` of their area within the ``<text>`` box ± ``pad``."""
    out = []
    for b in boxes:
        ix = max(0, min(b[2], x1 + pad) - max(b[0], x0 - pad))
        iy = max(0, min(b[3], y1 + pad) - max(b[1], y0 - pad))
        if ix * iy / max(1, (b[2] - b[0]) * (b[3] - b[1])) >= inside:
            out.append(b)
    return out


def detcount(a):
    """Line boxes per ``<text>`` against the annotated line count, per detector
    input size (``--imgsz``; 0 = the page's native size)."""
    from anime_tools.ocr.animetext import AnimeTextDetector

    pages = sorted(gt_set().items())
    imgs = {k: cv2.imread(str(m109.page_path(*k))) for k, _ in pages}
    for sz in a.imgsz:
        det = AnimeTextDetector.load(device="cuda", imgsz=sz)
        t = Counter()
        for (book, page), items in pages:
            img = imgs[(book, page)]
            prep = det.prepare(img)
            boxes = [
                b[:4]
                for b in det._settle(det.forward_batch([prep])[0], prep, img.shape)
            ]
            for k, e, ann in items:
                x0, y0, x1, y1 = (
                    int(e.get(c)) for c in ("xmin", "ymin", "xmax", "ymax")
                )
                n = len(boxes_in(boxes, x0, y0, x1, y1, a.pad, a.inside))
                g = len(ann["line_lengths"])
                t["eq"] += n == g
                t["lt"] += n < g
                t["gt"] += n > g
                if g > 1:
                    t["multi"] += 1
                    t["multi_eq"] += n == g
        n = t["eq"] + t["lt"] + t["gt"]
        print(
            f"imgsz {sz}: count right {t['eq'] / n:.3f} (fewer {t['lt']}, more {t['gt']}); "
            f"multi-line {t['multi']}: right {t['multi_eq'] / t['multi']:.3f}",
            flush=True,
        )


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / max(1, u)


def detdiag(a):
    """Why the detector gives about one box per ``<text>``: every box it emits at a
    low score floor with no denesting, page-level (640) and crop-level (the ``<text>``
    box with ``--ctx`` × its size of context each side, letterboxed at 640), matched
    to the annotated line boxes (IoU ≥ ``--line_iou``). Writes the boxes per text to
    ``detdiag.jsonl``; ``detdiag_report`` summarises and draws overlays."""
    from anime_tools.ocr.animetext import AnimeTextDetector

    det = AnimeTextDetector.load(device="cuda", conf=a.conf_floor, nest="raw")
    OUT.mkdir(parents=True, exist_ok=True)
    pages = sorted(gt_set().items())
    with open(OUT / "detdiag.jsonl", "w", encoding="utf-8") as f:
        for pi, ((book, page), items) in enumerate(pages):
            img = cv2.imread(str(m109.page_path(book, page)))
            H, W = img.shape[:2]
            page_boxes = det.detect_scored(img)
            crops, offs = [], []
            for k, e, t in items:
                x0, y0, x1, y1 = (
                    int(e.get(c)) for c in ("xmin", "ymin", "xmax", "ymax")
                )
                cx, cy = int(a.ctx * (x1 - x0)), int(a.ctx * (y1 - y0))
                c = (max(0, x0 - cx), max(0, y0 - cy), min(W, x1 + cx), min(H, y1 + cy))
                crops.append(img[c[1] : c[3], c[0] : c[2]])
                offs.append(c[:2])
            prep = [det.prepare(c) for c in crops]
            crop_boxes = [
                det._settle(raw, p, c.shape)
                for raw, p, c in zip(det.forward_batch(prep), prep, crops)
            ]
            for (k, e, t), cb, (ox, oy) in zip(items, crop_boxes, offs):
                box = [int(e.get(c)) for c in ("xmin", "ymin", "xmax", "ymax")]
                lines = [
                    [ln["xmin"], ln["ymin"], ln["xmax"], ln["ymax"]]
                    for ln in t["lines"]
                ]
                near = [
                    list(b)
                    for b in page_boxes
                    if _iou(b, [box[0] - 64, box[1] - 64, box[2] + 64, box[3] + 64]) > 0
                ]
                crop_pg = [
                    [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, b[4]] for b in cb
                ]
                f.write(
                    json.dumps(
                        dict(
                            book=book,
                            page=page,
                            id=k[2],
                            box=box,
                            lines=lines,
                            page_boxes=near,
                            crop_boxes=crop_pg,
                        )
                    )
                    + "\n"
                )
            if pi % 100 == 0:
                print(f"page {pi}/{len(pages)}", flush=True)
    print(f"wrote {OUT / 'detdiag.jsonl'}")


def detdiag_report(a):
    from anime_tools.ocr.animetext import denest

    rows = [json.loads(x) for x in open(OUT / "detdiag.jsonl", encoding="utf-8")]

    def inside(bs, box):
        return boxes_in(bs, *box, a.pad, a.inside)

    def matched(bs, lines):
        return sum(any(_iou(b, ln) >= a.line_iou for b in bs) for ln in lines)

    variants = {
        "page inner@.25": lambda r: denest(
            [b for b in r["page_boxes"] if b[4] >= 0.25]
        ),
        "page raw@.25": lambda r: [b for b in r["page_boxes"] if b[4] >= 0.25],
        f"page raw@{a.conf_floor}": lambda r: r["page_boxes"],
        "crop inner@.25": lambda r: denest(
            [b for b in r["crop_boxes"] if b[4] >= 0.25]
        ),
        "crop raw@.25": lambda r: [b for b in r["crop_boxes"] if b[4] >= 0.25],
        f"crop raw@{a.conf_floor}": lambda r: r["crop_boxes"],
    }
    multi = [r for r in rows if len(r["lines"]) > 1]
    nl = sum(len(r["lines"]) for r in multi)
    print(f"{len(rows)} texts, {len(multi)} multi-line ({nl} lines)")
    for name, fn in variants.items():
        eq = lt = gt = hit = 0
        for r in multi:
            bs = inside(fn(r), r["box"])
            n, g = len(bs), len(r["lines"])
            eq += n == g
            lt += n < g
            gt += n > g
            hit += matched(bs, r["lines"])
        print(
            f"  {name:18s} count right {eq / len(multi):.3f} (fewer {lt}, more {gt}); "
            f"GT lines with a box at IoU≥{a.line_iou}: {hit / nl:.3f}"
        )
    # the score of the best-matching raw box per GT line (page and crop)
    for src in ("page_boxes", "crop_boxes"):
        best = []
        for r in multi:
            for ln in r["lines"]:
                s = [b[4] for b in r[src] if _iou(b, ln) >= a.line_iou]
                best.append(max(s) if s else 0.0)
        best = np.array(best)
        qs = " ".join(
            f"≥{t}:{(best >= t).mean():.3f}" for t in (0.05, 0.1, 0.25, 0.426)
        )
        print(f"  {src}: GT lines matched by a box scoring {qs}")
    # overlays: green = GT lines, red = page raw@.25, blue = crop raw@.25
    viz = OUT / "detdiag_viz"
    viz.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    pick = [multi[i] for i in rng.choice(len(multi), a.show, replace=False)]
    for r in pick:
        img = cv2.imread(str(m109.page_path(r["book"], r["page"])))
        H, W = img.shape[:2]
        x0, y0, x1, y1 = r["box"]
        m = int(0.6 * max(x1 - x0, y1 - y0))
        c = (max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m))
        panels = []
        for src, col in (("page_boxes", (0, 0, 255)), ("crop_boxes", (255, 0, 0))):
            p = img.copy()
            for ln in r["lines"]:
                cv2.rectangle(p, ln[:2], ln[2:4], (0, 180, 0), 2)
            for b in r[src]:
                if b[4] >= 0.25:
                    cv2.rectangle(p, (b[0], b[1]), (b[2], b[3]), col, 2)
                elif b[4] >= a.conf_floor:
                    cv2.rectangle(p, (b[0], b[1]), (b[2], b[3]), (0, 200, 255), 1)
            p = p[c[1] : c[3], c[0] : c[2]]
            s = 480 / max(p.shape[:2])
            panels.append(cv2.resize(p, None, fx=s, fy=s))
        h = max(p.shape[0] for p in panels)
        panels = [
            cv2.copyMakeBorder(
                p, 0, h - p.shape[0], 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            for p in panels
        ]
        name = f"{r['book']}_{r['page']:03d}_{r['id']}.png"
        cv2.imwrite(str(viz / name), np.hstack(panels))
    print(f"overlays in {viz}")


def column_crops(crop, raw: str, a) -> list:
    """The ink-projection segments of ``crop`` cut at the 4 px margin, reading order."""
    vertical = guess_vertical(crop, raw)
    segs = segment(ink_mask(crop), vertical, a.gap, a.sliver, a.merge)
    if len(segs) < 2:
        return []
    H, W = crop.shape[:2]
    out = []
    for a0, a1, b0, b1 in segs:
        x0, x1, y0, y1 = (a0, a1, b0, b1) if vertical else (b0, b1, a0, a1)
        c = margin_box(x0, y0, x1, y1, W, H)
        out.append(crop[c[1] : c[3], c[0] : c[2]])
    return out


def manifest_rows(name: str = "sfxp_speech"):
    """One-line ``name`` rows of ``manifest_<name>`` the line annotation does not cover:
    (key, label, None, crop). ``sfxp_speech``: the speech manifest; ``mainsub_text``:
    the ``<text>`` boxes outside it (``mainsub_text.py``)."""
    import pandas as pd

    d = m109.derived_root()
    df = pd.read_parquet(d / f"manifest_{name}.parquet")
    ann = {
        (pg["book"], int(pg["page_index"]), t["id"])
        for pg in json.loads(LINES.read_text(encoding="utf-8")).values()
        for t in pg["texts"]
    }
    df = df[(df.source == name) & ~df.text.str.contains("\n")]
    for r in df.itertuples():
        k = (r.book, int(r.page), str(r.id))
        if k in ann:
            continue
        yield k, r.text, None, cv2.imread(str(d / r.path))


def inkread(a):
    """Ink-projection columns (rows) of each crop, each read alone with a
    PaddleOCR-VL reader, the reads joined with ``\\n`` → ``<name>.jsonl`` (``pred``;
    ``out`` = the label with the read's breaks). A crop with one segment is not read
    (its pred is the label). ``--set eval``: the annotated ``<text>`` crops (score
    with ``transfer``); ``--set manifest``: the one-line speech rows to break."""
    import time

    t0 = time.time()
    rows, crops, caps = [], [], []
    src = (
        gt_crops()
        if a.set == "eval"
        else manifest_rows("sfxp_speech" if a.set == "manifest" else a.set)
    )
    for k, raw, gt, crop in src:
        cols = column_crops(crop, raw, a)
        rows.append(
            dict(
                book=k[0],
                page=k[1],
                id=k[2],
                raw=raw,
                gt=gt,
                n_segs=max(1, len(cols)),
                first=len(crops),
            )
        )
        crops += cols
        caps += [min(MAX_NEW_TOKENS, 2 * len(raw) + 8)] * len(cols)
    print(
        f"{len(rows)} texts, {len(crops)} segment crops ({time.time() - t0:.0f} s)",
        flush=True,
    )
    reads = Vl16Reader(a.ckpt, "cuda").read(
        crops,
        None,
        a.ink_bs,
        caps=caps,
        budget=27648,
        prefetch=a.prefetch,
        min_pixels=a.min_pixels,
        progress=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{a.name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if r["n_segs"] < 2:
                r["lines"], r["pred"] = [], r["raw"]
                r["out"] = r["raw"].strip()
            else:
                r["lines"] = reads[r["first"] : r["first"] + r["n_segs"]]
                r["pred"] = "\n".join(x.replace("\n", "") for x in r["lines"])
                r["out"] = transfer_breaks(r["pred"], r["raw"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT / (a.name + '.jsonl')}")


def detread(a):
    """AnimeText detector over the annotated pages (``nest inner``: a balloon's
    columns), the boxes inside each ``<text>`` box taken as its lines in reading
    order, each line read alone (``OCR:``), the reads joined with ``\\n``."""
    from anime_tools.ocr.animetext import AnimeTextDetector

    det = AnimeTextDetector.load(device="cuda")
    rows, crops = [], []
    for (book, page), items in sorted(gt_set().items()):
        img = cv2.imread(str(m109.page_path(book, page)))
        H, W = img.shape[:2]
        prep = det.prepare(img)
        boxes = [
            b[:4] for b in det._settle(det.forward_batch([prep])[0], prep, img.shape)
        ]
        for k, e, t in items:
            x0, y0, x1, y1 = (int(e.get(c)) for c in ("xmin", "ymin", "xmax", "ymax"))
            inside = []
            for b in boxes:
                bx0, by0, bx1, by1 = b
                ix = max(0, min(bx1, x1 + a.pad) - max(bx0, x0 - a.pad))
                iy = max(0, min(by1, y1 + a.pad) - max(by0, y0 - a.pad))
                area = max(1, (bx1 - bx0) * (by1 - by0))
                if ix * iy / area >= a.inside:
                    inside.append(b)
            tall = sum((b[3] - b[1]) > (b[2] - b[0]) for b in inside)
            vertical = 2 * tall >= len(inside) if inside else True
            inside.sort(key=lambda b: -(b[0] + b[2]) if vertical else (b[1] + b[3]))
            raw = e.text or ""
            rows.append(
                dict(
                    book=book,
                    page=page,
                    id=k[2],
                    raw=raw,
                    gt=line_broken(raw, t["line_lengths"]),
                    n_boxes=len(inside),
                    first=len(crops),
                )
            )
            for b in inside:
                c = margin_box(*b, W, H)
                crops.append(img[c[1] : c[3], c[0] : c[2]])
    print(f"{len(rows)} texts, {len(crops)} line boxes", flush=True)
    reads = Vl16Reader(a.ckpt, "cuda").read(crops, None, a.bs)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{a.name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            r["lines"] = reads[r["first"] : r["first"] + r["n_boxes"]]
            r["pred"] = "\n".join(x for x in r["lines"] if x)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT / (a.name + '.jsonl')}")


def _fold(c: str) -> str:
    """One character → one comparable character (dot glyphs → ``.``)."""
    import unicodedata

    if c in DOTS:
        return "."
    f = unicodedata.normalize("NFKC", c).lower()
    return f if len(f) == 1 else c


def transfer_breaks(pred: str, label: str) -> str:
    """The label with the pred's line breaks, placed by an edit-distance alignment."""
    flat = label.replace("\n", "")
    lines = [ln.replace("…", "...").replace("‥", "..") for ln in pred.split("\n")]
    if len(lines) <= 1 or not flat:
        return flat.strip()
    P = "".join(lines)
    starts, i = set(), 0
    for ln in lines[:-1]:
        i += len(ln)
        starts.add(i)  # index in P where a new line begins
    n, m = len(P), len(flat)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    pf = [_fold(c) for c in P]
    lf = [_fold(c) for c in flat]
    for x in range(1, n + 1):
        for y in range(1, m + 1):
            D[x, y] = min(
                D[x - 1, y] + 1,
                D[x, y - 1] + 1,
                D[x - 1, y - 1] + (pf[x - 1] != lf[y - 1]),
            )
    # backtrace to (pred index, label index) pairs; a pred index that starts a line
    # maps to the label index of the first pred character at or after it that
    # aligns to a label character (substitutions included)
    pairs = []
    x, y = n, m
    # ties skip a label character first, so a short read of a repeated run (… for
    # ......) aligns to the run's start, where its line begins
    while x > 0 and y > 0:
        if D[x, y] == D[x, y - 1] + 1:
            y -= 1
        elif D[x, y] == D[x - 1, y - 1] + (pf[x - 1] != lf[y - 1]):
            pairs.append((x - 1, y - 1))
            x, y = x - 1, y - 1
        else:
            x -= 1
    pairs.reverse()
    cuts = set()
    for i in starts:
        nxt = next((ly for px, ly in pairs if px >= i), None)
        if nxt is not None and 0 < nxt < m:
            cuts.add(nxt)
    parts, j = [], 0
    for k in sorted(cuts) + [m]:
        parts.append(flat[j:k].strip())
        j = k
    return "\n".join(p for p in parts if p)


def _gate_norm(s: str) -> str:
    import re
    import unicodedata

    s = re.sub(r"\s", "", unicodedata.normalize("NFKC", s))
    s = re.sub(r"[.．・…‥･]+", ".", s)
    s = re.sub(r"[~〜～ー\-]{2,}", "~", s)
    return re.sub(r"([!?！？])\1+", r"\1", s).lower()


def read_distance(pred: str, label: str) -> float:
    """Edit distance between the joined column reads and the label over the label's
    length, both NFKC, spaceless, with dot / wave / ``!?`` runs folded to one (a read's
    ``……`` against a label's ``......`` is no miss). The gate: on the annotated set,
    rows under 0.05 whose line count equals the segment count are 81.6 % of the
    segmented rows at 0.989 exact."""
    p, q = _gate_norm(pred), _gate_norm(label)
    D = np.arange(len(q) + 1)
    for x in range(1, len(p) + 1):
        prev, D[0] = D.copy(), x
        for y in range(1, len(q) + 1):
            D[y] = min(prev[y] + 1, D[y - 1] + 1, prev[y - 1] + (p[x - 1] != q[y - 1]))
    return float(D[-1]) / max(1, len(q))


GATE = 0.05


def _flat_key(t: str) -> str:
    """``t`` without whitespace or dot glyphs."""
    import re

    return re.sub(r"[\s.．・･…‥]", "", t)


def apply(a):
    """``inkread --set manifest``'s rows into ``manifest_sfxp_speech``: a segmented row
    passing the gate (:func:`read_distance` < ``GATE``, lines == segments) takes the
    broken label; the other segmented rows go to ``linebreak_suspicious.jsonl`` for
    ``review_page.py``. ``.pre_linebreak.parquet`` backup, once. ``--set
    mainsub_text``: into ``manifest_mainsub_text``, the others to
    ``linebreak_suspicious_mainsub_text.jsonl`` (not on the page)."""
    import shutil

    import pandas as pd

    rows = [json.loads(x) for x in open(OUT / f"{a.name}.jsonl", encoding="utf-8")]
    d = m109.derived_root()
    name = "mainsub_text" if a.set == "mainsub_text" else "sfxp_speech"
    path = d / f"manifest_{name}.parquet"
    df = pd.read_parquet(path)
    pos = {
        (r.book, int(r.page), str(r.id)): i
        for i, r in enumerate(df.itertuples())
        if r.source == name
    }
    applied, stale, sus = 0, 0, []
    col = df.columns.get_loc("text")
    for r in rows:
        if r["n_segs"] < 2:
            continue
        dist = read_distance(r["pred"], r["raw"])
        ok = dist < GATE and r["out"].count("\n") + 1 == r["n_segs"]
        if not ok:
            sus.append(dict(r, dist=round(dist, 3)))
            continue
        i = pos.get((r["book"], int(r["page"]), str(r["id"])))
        # the manifest may hold the label ellipsis-folded (build_sfx_prefix --ellipsis)
        # or already broken (a rerun), and a break placed on a space takes the space:
        # compare blind to whitespace and dot spelling
        if i is None or _flat_key(df.text.iat[i]) != _flat_key(r["raw"]):
            stale += 1
            continue
        if df.text.iat[i] != ellipsis_fold(r["out"]):
            df.iat[i, col] = ellipsis_fold(r["out"])
            applied += 1
    print(f"applied {applied}, stale {stale}, suspicious {len(sus)}")
    sus_name = (
        "linebreak_suspicious.jsonl"
        if name == "sfxp_speech"
        else f"linebreak_suspicious_{name}.jsonl"
    )
    with open(OUT / sus_name, "w", encoding="utf-8") as f:
        for r in sus:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.dry_run or not applied:
        return
    bak = path.with_suffix(".pre_linebreak.parquet")
    if not bak.exists():
        shutil.copy2(path, bak)
    df.to_parquet(path, index=False)
    n_lf = df.text.str.contains("\n").sum()
    print(f"wrote {path}: {len(df)} rows, {n_lf} with a line break")


DOT_ONLY = set(".．・･…‥")


def dot_columns(crop) -> list[int]:
    """Dots per printed column (row, for a horizontal leader), reading order: the ink
    components at ≥ 0.3 × the median component area, counted per ink-projection
    segment. The axis is the one that gives fewer segments — a dot-only crop's shape
    says nothing (three short columns make a wide box)."""
    # columns from the interior dots (edge components out, as ink_mask does); then a
    # dot-sized blob on the edge counts when it sits on one of those columns (a column's
    # last dot often touches the 4 px margin) — balloon-outline bits in the corners do not
    raw = ink_mask(crop, border=False)
    n, lab, st, cen = cv2.connectedComponentsWithStats(
        raw.astype(np.uint8), connectivity=8
    )
    if n < 2:
        return []
    H, W = raw.shape
    x, y, w, h, area = (st[1:, j] for j in range(5))
    edge = (x == 0) | (y == 0) | (x + w == W) | (y + h == H)
    if edge.all():
        return []
    med = np.median(area[~edge])
    inner = [i + 1 for i in range(n - 1) if not edge[i] and area[i] >= 0.3 * med]
    extra = [
        i + 1
        for i in range(n - 1)
        if edge[i]
        and 0.5 * med <= area[i] <= 2 * med
        and max(w[i], h[i]) <= 2 * min(w[i], h[i])
    ]
    ink = np.isin(lab, inner)
    best = None
    for vertical in (True, False):
        segs = segment(ink, vertical, 0.03, 0.55, 0.0)
        ax = 0 if vertical else 1
        cols = [
            sum(a0 <= cen[i][ax] < a1 for i in inner)
            + sum(a0 <= cen[i][ax] < a1 for i in extra)
            for a0, a1, *_ in segs
        ]
        cols = [c for c in cols if c]
        if best is None or len(cols) < len(best):
            best = cols
    return best


def dots(a):
    """Dot-only speech labels (``・・・・・・\\n・・・``) re-spelled from the page, as the
    user's review edits do: per printed column ``ceil(dots / 3)`` × ``…``, one
    line per column. A row passes when every column holds ≥ 2 dots, there are ≤ 6
    columns and the counted dots are within 0.6–1.6× the label's (``…`` = 3, ``‥`` = 2);
    passing rows are written to ``manifest_sfxp_speech`` (``.pre_dots.parquet`` backup),
    the rest go to ``dots_suspicious.jsonl`` for ``review_page.py``."""
    import math
    import shutil

    import pandas as pd

    d = m109.derived_root()
    path = d / "manifest_sfxp_speech.parquet"
    df = pd.read_parquet(path)
    col = df.columns.get_loc("text")
    changed, same, sus = 0, 0, []
    for i, r in enumerate(df.itertuples()):
        flat = "".join(r.text.split())
        if not flat or not set(flat) <= DOT_ONLY:
            continue
        cols = dot_columns(cv2.imread(str(d / r.path)))
        new = "\n".join("…" * math.ceil(c / 3) for c in cols)
        label = sum(3 if c == "…" else 2 if c == "‥" else 1 for c in flat)
        ok = (
            cols
            and min(cols) >= 2
            and len(cols) <= 6
            and 0.6 <= sum(cols) / label <= 1.6
        )
        if not ok:
            sus.append(
                dict(
                    book=r.book,
                    page=int(r.page),
                    id=str(r.id),
                    raw=r.text,
                    out=new or r.text,
                    cols=[int(c) for c in cols],
                    label_dots=label,
                )
            )
            continue
        if new == r.text:
            same += 1
            continue
        df.iat[i, col] = new
        changed += 1
    print(f"dot-only rows: changed {changed}, already so {same}, suspicious {len(sus)}")
    with open(OUT / "dots_suspicious.jsonl", "w", encoding="utf-8") as f:
        for x in sus:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    if a.dry_run or not changed:
        return
    bak = path.with_suffix(".pre_dots.parquet")
    if not bak.exists():
        shutil.copy2(path, bak)
    df.to_parquet(path, index=False)
    print(f"wrote {path}")


def transfer(a):
    rows = [json.loads(x) for x in open(OUT / f"{a.name}.jsonl", encoding="utf-8")]
    t = Counter()
    misses = []
    for r in rows:
        gt, pred = r["gt"], transfer_breaks(r["pred"], r["raw"])
        n_gt, n_pr = gt.count("\n") + 1, pred.count("\n") + 1
        t["n"] += 1
        t["read_lf"] += "\n" in r["pred"]
        t["exact"] += pred == gt
        t["count_ok"] += n_gt == n_pr
        if n_gt > 1:
            t["multi"] += 1
            t["multi_exact"] += pred == gt
            t["multi_count_ok"] += n_gt == n_pr
            if n_gt == n_pr:
                t["alloc_n"] += 1
                t["alloc_ok"] += pred == gt
        if pred != gt and len(misses) < a.show:
            misses.append(
                (f"{r['book']}_{r['page']:03d}_{r['id']}", gt, r["pred"], pred)
            )
    n, mm = t["n"], t["multi"]
    print(
        f"{a.name}: n {n}, reads with a break {t['read_lf']}; line count {t['count_ok'] / n:.3f}; "
        f"exact {t['exact'] / n:.3f}\nmulti-line GT {mm}: line count "
        f"{t['multi_count_ok'] / mm:.3f}, exact {t['multi_exact'] / mm:.3f}; exact when the "
        f"count is right {t['alloc_ok'] / max(1, t['alloc_n']):.3f} ({t['alloc_n']})"
    )
    for mid, gt, rd, pred in misses:
        print(f"  {mid}\n    gt   {gt!r}\n    read {rd!r}\n    out  {pred!r}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "mode",
        choices=[
            "eval",
            "read",
            "transfer",
            "detread",
            "detcount",
            "inkread",
            "apply",
            "dots",
            "detdiag",
            "detdiag_report",
        ],
    )
    ap.add_argument(
        "--set", choices=["eval", "manifest", "mainsub_text"], default="eval"
    )
    ap.add_argument("--dry_run", action="store_true", help="apply: count only")
    ap.add_argument(
        "--prefetch", type=int, default=2, help="inkread: batches prepped ahead"
    )
    ap.add_argument(
        "--min_pixels", type=int, default=25088, help="inkread: crop upscale floor"
    )
    ap.add_argument("--conf_floor", type=float, default=0.05)
    ap.add_argument("--ctx", type=float, default=0.25, help="detdiag: crop context")
    ap.add_argument("--line_iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1024, 1280, 0])
    ap.add_argument(
        "--pad", type=int, default=8, help="detread: px around the <text> box"
    )
    ap.add_argument(
        "--inside",
        type=float,
        default=0.6,
        help="detread: share of a detector box inside the <text> box",
    )
    ap.add_argument("--name", default="stock")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--ink_bs", type=int, default=96, help="inkread: crops per batch")
    ap.add_argument(
        "--gap", type=float, default=0.03, help="profile ≤ gap·max is empty"
    )
    ap.add_argument("--sliver", type=float, default=0.55)
    ap.add_argument("--snap", type=int, default=2)
    ap.add_argument("--merge", type=float, default=0.0)
    ap.add_argument("--oracle_orient", action="store_true")
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    modes = dict(
        eval=evaluate,
        read=read,
        transfer=transfer,
        detread=detread,
        detcount=detcount,
        inkread=inkread,
        apply=apply,
        dots=dots,
        detdiag=detdiag,
        detdiag_report=detdiag_report,
    )
    modes[a.mode](a)


if __name__ == "__main__":
    main()
