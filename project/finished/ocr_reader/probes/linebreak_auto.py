#!/usr/bin/env python3
"""Automatic line breaks for one-line Manga109 speech labels, measured on the community
line annotations (``assets/manga109s_line_annotations.json``, 5,988 ``<text>`` boxes).

The crop is cut as ``build_sfx_prefix.build_speech`` cuts it (the ``<text>`` box at the
4 px margin). Columns (vertical text) or rows (horizontal) come from the ink projection:
binarise (Otsu), sum ink across the reading axis, split at empty runs, drop slivers
(furigana, balloon edges) narrower than ``--sliver`` × the widest column. The label is cut
in proportion to each column's ink length, then the cut is moved off a character that
may not start a line (small kana, prolonged sound mark, closing punctuation) and onto a
nearby space.

    ANIMA_MANGA109S_ROOT=… .venv/bin/python project/finished/ocr_reader/probes/linebreak_auto.py eval
    … make daemon-run ARGS="… linebreak_auto.py read --name stock [--ckpt output/ocr/<run>/best]"
    … linebreak_auto.py transfer --name stock

``read`` runs a PaddleOCR-VL reader (``eval_manga109.Vl16Reader``, ``OCR:``) over the same
crops and stores its reads (``output/ocr/linebreak/<name>.jsonl``); ``transfer`` aligns
each read to the label (edit-distance DP on NFKC-folded characters) and carries the
read's line breaks onto the label, then scores against the annotation.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ocr"))
import manga109 as m109  # noqa: E402
from build_sfx_prefix import LINES, line_broken, margin_box  # noqa: E402

NO_START = set(
    "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶーｰ～~、。，．,.!！?？)）」』】〉》…‥・ﾞﾟ゛゜"
)
SPACES = set(" 　")


def ink_mask(crop: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b > 0


def runs(on: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def segment(
    ink: np.ndarray, vertical: bool, gap_frac: float, sliver: float, merge: float = 0.0
) -> list[tuple[int, int, int, int]]:
    """Columns (vertical) / rows (horizontal) in reading order as (a0, a1, b0, b1):
    ``a`` across the reading axis, ``b`` the ink extent along it."""
    m = ink if vertical else ink.T  # columns run along axis 0
    prof = m.sum(axis=0)
    if prof.max() == 0:
        return []
    on = prof > gap_frac * prof.max()
    segs = [(a, b) for a, b in runs(on)]
    if not segs:
        return []
    widest = max(b - a for a, b in segs)
    # a gap narrower than merge × the widest run sits inside a glyph (は, い, ?)
    merged = [segs[0]]
    for a, b in segs[1:]:
        if a - merged[-1][1] < merge * widest:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    widest = max(b - a for a, b in merged)
    segs = [(a, b) for a, b in merged if (b - a) >= sliver * widest]
    out = []
    for a, b in segs:
        rows = np.flatnonzero(m[:, a:b].any(axis=1))
        out.append((a, b, int(rows[0]), int(rows[-1]) + 1))
    if vertical:
        out.reverse()  # right to left
    return out


SMALL = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ、。，．,")
DOTS = set(".．・…‥")


def glyph_weight(ch: str) -> float:
    """Length a glyph takes along a column, in full-width cells."""
    if ch in SPACES:
        return 0.5
    if ch in DOTS:
        return 0.35
    if ch in SMALL:
        return 0.7
    if ch in "!！?？":
        return 0.6
    if ch.isascii():
        return 0.55
    return 1.0


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


OUT = m109.REPO / "output/ocr/linebreak"


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
    import eval_manga109 as ev

    rd = ev.Vl16Reader(a.ckpt, "cuda")
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


def guess_vertical(crop: np.ndarray, raw: str) -> bool:
    """Vertical unless the box is clearly wider than a single row of the label."""
    h, w = crop.shape[:2]
    return not (w > 1.6 * h)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["eval", "read", "transfer"])
    ap.add_argument("--name", default="stock")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--gap", type=float, default=0.0, help="profile ≤ gap·max is empty")
    ap.add_argument("--sliver", type=float, default=0.45)
    ap.add_argument("--snap", type=int, default=2)
    ap.add_argument("--merge", type=float, default=0.25)
    ap.add_argument("--oracle_orient", action="store_true")
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    {"eval": evaluate, "read": read, "transfer": transfer}[a.mode](a)


if __name__ == "__main__":
    main()
