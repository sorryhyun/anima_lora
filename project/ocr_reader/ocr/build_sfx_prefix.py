#!/usr/bin/env python3
"""Crops for the two-prompt reader: ``OCR:`` reads speech, ``SFX:`` reads sound effects,
each answers ``""`` on the other kind.

    ANIMA_MANGA109S_ROOT=~/manga109s/Manga109s_released_2026_05_21 \\
        python project/ocr_reader/ocr/build_sfx_prefix.py [--det] [--speech] [--kozh]

Every crop takes the margin ``min(0.12·long, 0.3·short, 4 px)`` around its box, clipped
to the page. Three parts, each its own manifest under ``derived/``:

* ``--det`` → ``manifest_sfxp_det.parquet``: the AnimeText detector (the OCR stage's,
  ``denest`` inner) over every train / val page. Each box is labelled against the COO
  polygons by pixel overlap, ``pf`` = share of the polygon inside the crop, ``cf`` = share
  of the crop the polygon covers:

  - ``pf ≥ 0.8`` → the SFX is in the target;
  - ``pf < 0.2`` and ``cf < 0.2`` → a graze, ignored;
  - anything else (a polygon cut partway, or a fragment of a large one) → box dropped.

  Speech: a ``<text>`` box with ≥ 30 % of its area in the crop marks the crop as holding
  speech. ``cat`` = ``pos`` (SFX, no speech), ``pos_mixed`` (both), ``neg_speech``
  (speech, no SFX), ``neg_other`` (neither; not written). ``pos_mixed`` also carries
  ``speech_text`` when every ``<text>`` box touching it (≥ 20 %) lies ≥ 80 % inside.
  ``neg_speech`` crops are written for a ``--neg_keep`` share only.
* ``--speech`` → ``manifest_sfxp_speech.parquet``: the grey manifest's Manga109 speech
  boxes, re-cut at the new margin, labels as they are; then ``--lines``.
* ``--lines`` → the same manifest, in place: line breaks from the community line
  annotations (``assets/manga109s_line_annotations.json``, ``build_lines``) on the rows
  they cover, and their other train / val ``<text>`` boxes cut and appended.
* ``--ellipsis`` → the same manifest, in place (also run at the end of ``--speech`` /
  ``--lines``): a dot run whose length is a multiple of three becomes that many ``…``
  thirds (``・・・・・・`` → ``……``; :func:`ellipsis_fold`), as the gelnote / swaprelax labels
  spell it; a run of four in a label with text becomes one ``…``.
  Other runs stay.
* ``--gt test`` → ``manifest_sfxp_gt.parquet``: the grey manifest's rows of those splits (COO
  and speech) re-cut through ``crops.deskew_crop`` at the capped margin, for the gate evals
  (``eval_manga109.py --manifest sfxp_gt``).
* ``--clean`` → ``manifest_sfxp_det_clean.parquet``: ``--det``'s boxes with the doubtful
  labels unwritten (``build_clean``).
* ``--det_all`` → ``manifest_sfxp_det_all.parquet``: ``--clean``'s kept boxes holding text
  (``pos`` / ``pos_mixed`` / ``neg_speech``), every one with a crop: the ``neg_speech``
  boxes ``--neg_keep`` left unwritten are cut now, same box and name (``build_det_all``;
  the main / sub review, ``labeling/m109_mainsub.py``). ``load_sfxp`` does not read it.
* ``--missed`` → ``manifest_sfxp_missed.parquet``: train COO polygons no detector box
  covers, typed (``missed_sub``); the ``--missed_keep`` types (``i_plain``, ``a_punct``,
  and ``g_tiny`` / ``e_dark_bg`` with no other SFX in the crop) cut from the polygon as
  ``SFX:`` rows (``build_missed``).
* ``--kozh`` → ``manifest_sfxp_kozh.parquet``: the gelnote KO / ZH and swaprelax_v5 pseudo
  rows, re-cut from their pages at the new margin, labels as they are, ``kind = speech``.

Several SFX (or speech boxes) in one crop join with ``\\n``, right to left, then top down.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from project.ocr_reader.core import manga109 as m109
from project.ocr_reader.core.crops import cap_frac, deskew_crop, margin_box
from project.ocr_reader.core.det_types import classify
from project.ocr_reader.core.labels import ellipsis_fold, line_broken
from project.ocr_reader.core.paths import GELNOTE, REPO, SWAPRELAX
from project.ocr_reader.core.paths import LINE_ANNOTATIONS as LINES

INCLUDE, GRAZE, SPEECH_IN = 0.8, 0.2, 0.3


def orient(w, h) -> str:
    return "vertical" if h > w else "horizontal"


def reading_order(items):
    """(x-centre, y-centre, text) → texts right to left, then top down."""
    return [t for _, _, t in sorted(items, key=lambda r: (-r[0], r[1]))]


def poly_mask(line, shape):
    a = np.asarray(line.poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
    if line.joined:  # two linked point sets: their hull
        a = cv2.convexHull(a)
    mk = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mk, [a.reshape(-1, 1, 2)], 1)
    return mk.astype(bool), a


def rect_frac(q, c) -> float:
    """Share of rect ``q`` (x0, y0, x1, y1) inside rect ``c``."""
    iw = max(0.0, min(q[2], c[2]) - max(q[0], c[0]))
    ih = max(0.0, min(q[3], c[3]) - max(q[1], c[1]))
    return iw * ih / max((q[2] - q[0]) * (q[3] - q[1]), 1e-6)


def label_box(c, coo, texts):
    """The ``--det`` labelling rule (module docstring) for one crop ``c``; returns (cat, sfx_text, speech_text) or None
    for a dropped box."""
    cx0, cy0, cx1, cy1 = c
    ca = max((cx1 - cx0) * (cy1 - cy0), 1)
    inc = []
    for line, mk, area, (mx, my) in coo:
        inside = int(mk[cy0:cy1, cx0:cx1].sum())
        if not inside:
            continue
        pf, cf = inside / area, inside / ca
        if pf >= INCLUDE:
            inc.append((mx, my, line.text))
        elif pf >= GRAZE or cf >= GRAZE:
            return None
    sp_hit, sp_inc, sp_partial = False, [], False
    for t in texts:
        q = (t.poly[0], t.poly[1], t.poly[2], t.poly[5])
        f = rect_frac(q, c)
        if f >= SPEECH_IN:
            sp_hit = True
        if f >= INCLUDE:
            sp_inc.append(((q[0] + q[2]) / 2, (q[1] + q[3]) / 2, t.text))
        elif f >= GRAZE:
            sp_partial = True
    sfx = "\n".join(reading_order(inc)) if inc else None
    if sfx is not None:
        if not sp_hit:
            return "pos", sfx, None
        speech = None if sp_partial or not sp_inc else "\n".join(reading_order(sp_inc))
        return "pos_mixed", sfx, speech
    return ("neg_speech" if sp_hit else "neg_other"), None, None


def build_det(a, derived: Path):
    from anime_tools.ocr.animetext import AnimeTextDetector

    det = AnimeTextDetector.load(device=a.device)
    split = m109.load_split()
    rng = random.Random(0)
    rows, t0, n_pages = [], time.time(), 0
    pool = ThreadPoolExecutor(8)
    for sp in a.splits:
        for book in split[sp]:
            coo_by, txt_by = {}, {}
            for line in m109.iter_coo(book):
                coo_by.setdefault(line.page, []).append(line)
            for line in m109.iter_text(book):
                txt_by.setdefault(line.page, []).append(line)
            pages = sorted(
                int(p.stem)
                for p in (m109.manga109s_root() / "images" / book).glob("*.jpg")
            )
            if a.limit_pages:
                pages = pages[: a.limit_pages]
            out_dir = derived / "crops_sfxp" / "det" / sp
            out_dir.mkdir(parents=True, exist_ok=True)
            for s in range(0, len(pages), det.max_batch):
                chunk = pages[s : s + det.max_batch]
                imgs = list(
                    pool.map(lambda p: cv2.imread(str(m109.page_path(book, p))), chunk)
                )
                prepared = [det.prepare(im) for im in imgs]
                raws = det.forward_batch(prepared)
                writes = []
                for pg, im, raw, prep in zip(chunk, imgs, raws, prepared):
                    H, W = im.shape[:2]
                    coo = []
                    for line in coo_by.get(pg, []):
                        mk, pts = poly_mask(line, im.shape)
                        area = int(mk.sum())
                        if area:
                            coo.append((line, mk, area, pts.reshape(-1, 2).mean(0)))
                    for k, b in enumerate(det._settle(raw, prep, im.shape)):
                        c = margin_box(*b[:4], W, H)
                        if c[2] - c[0] < 4 or c[3] - c[1] < 4:
                            continue
                        lab = label_box(c, coo, txt_by.get(pg, []))
                        if lab is None:
                            cat = "drop"
                            sfx = speech = None
                        else:
                            cat, sfx, speech = lab
                        keep = cat in ("pos", "pos_mixed") or (
                            cat == "neg_speech" and rng.random() < a.neg_keep
                        )
                        rel = f"crops_sfxp/det/{sp}/{book}_{pg:03d}_{k:02d}.png"
                        rows.append(
                            dict(
                                split=sp,
                                cat=cat,
                                book=book,
                                page=pg,
                                id=f"{pg:03d}_{k:02d}",
                                box=list(c),
                                det=float(b[4]),
                                sfx_text=sfx,
                                speech_text=speech,
                                w=c[2] - c[0],
                                h=c[3] - c[1],
                                orient=orient(c[2] - c[0], c[3] - c[1]),
                                path=rel if keep else None,
                            )
                        )
                        if keep:
                            writes.append((derived / rel, im[c[1] : c[3], c[0] : c[2]]))
                list(pool.map(lambda wr: cv2.imwrite(str(wr[0]), wr[1]), writes))
                n_pages += len(chunk)
                if n_pages % 400 < len(chunk):
                    print(
                        f"[det] {n_pages} pages, {len(rows)} boxes, "
                        f"{time.time() - t0:.0f} s",
                        flush=True,
                    )
    df = pd.DataFrame(rows)
    df.to_parquet(derived / "manifest_sfxp_det.parquet", index=False)
    print(f"[det] {n_pages} pages in {time.time() - t0:.0f} s", flush=True)
    print(df.groupby(["split", "cat"]).size().to_string(), flush=True)
    mixed = df[df.cat == "pos_mixed"]
    print(
        f"[det] pos_mixed with a speech label: {mixed.speech_text.notna().sum()} / {len(mixed)}"
    )


KEPT_TYPES = {"same", "loose", "cut", "graze", "merged", "mixed", "dup"}


def merged_ok(texts, polys) -> bool:
    """A box holding 2+ SFX keeps its joined target when the order is not in doubt:
    the texts repeat (``ザワ``×3), or vertical columns stand side by side (right to
    left is right). Stacks, horizontal rows and scattered SFX are cut: the build's
    ``(-x, y)`` order reverses rows and mis-sorts stacks, and scattered ones have none."""
    strip = "・…~ー〜！!？?"
    if len({t.strip(strip) for t in texts}) == 1:
        return True
    bb = [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in polys]

    def ov(a0, a1, b0, b1):
        return max(0, min(a1, b1) - max(a0, b0)) / max(1, min(a1 - a0, b1 - b0))

    for i in range(len(bb)):
        for k in range(i + 1, len(bb)):
            yo = ov(bb[i][1], bb[i][3], bb[k][1], bb[k][3])
            xo = ov(bb[i][0], bb[i][2], bb[k][0], bb[k][2])
            if not (yo > 0.5 and xo < 0.5):
                return False
    return np.mean([(b[3] - b[1]) > (b[2] - b[0]) for b in bb]) >= 0.5


def build_clean(a, derived: Path):
    """``--clean`` → ``manifest_sfxp_det_clean.parquet``: the det manifest with the boxes
    whose labels the crop sheets (``probes/sfxp_crop_sheet.py``) put in doubt
    unwritten (``path`` None, the reason in ``clean``):

    * ``dup_outer`` — an SFX ≥ 80 % inside 2+ kept boxes (nested detector boxes): every
      box but the smallest goes, its speech row with it;
    * ``loose`` — one SFX alone in a box much larger than its COO crop (IoU < 0.6):
      15 % read text the target leaves out (titles, signs, unannotated asides);
    * ``merged_order`` — 2+ SFX in one box whose joined order is in doubt (``merged_ok``);
    * ``neg_graze`` — an ``SFX:`` negative with part of a COO polygon inside.

    Fragments and misses (the detector cuts or skips the SFX) stay out.
    Per-polygon types land in ``sfxp_types_<split>.parquet``.
    """
    full = pd.read_parquet(derived / "manifest_sfxp_det.parquet")
    full["clean"] = None
    for sp in a.splits:
        det, ty = classify(sp)
        ty.to_parquet(derived / f"sfxp_types_{sp}.parquet", index=False)
        rows = ty.assign(js=ty.det_rows.map(json.loads))
        reason: dict[int, str] = {}

        def area(j):
            b = det.at[j, "box"]
            return (b[2] - b[0]) * (b[3] - b[1])

        for js in rows[rows.type == "dup"].js:
            for j in sorted(js, key=area)[1:]:
                reason[j] = "dup_outer"
        for t in ("loose", "neg_graze"):
            for js in rows[rows.type == t].js:
                reason.setdefault(js[0], t)
        box_polys: dict[int, list] = {}
        for r in rows[rows.type.isin(KEPT_TYPES)].itertuples():
            for j in r.js:
                box_polys.setdefault(j, []).append(r)
        for j, ps in box_polys.items():
            if len(ps) < 2 or j in reason:
                continue
            polys = [np.asarray(json.loads(p.poly)).reshape(-1, 2) for p in ps]
            if not merged_ok([p.coo_text for p in ps], polys):
                reason[j] = "merged_order"
        idx = full.index[full.split == sp]
        for j, why in reason.items():
            g = idx[j]  # det was reset_index'd from the same split filter, in order
            full.at[g, "clean"] = why
            full.at[g, "path"] = None
        got = pd.Series(list(reason.values())).value_counts().to_dict()
        print(f"[clean] {sp}: {got}", flush=True)
    full.to_parquet(derived / "manifest_sfxp_det_clean.parquet", index=False)
    print(
        full[full.path.notna()].groupby(["split", "cat"]).size().to_string(), flush=True
    )


def build_det_all(a, derived: Path):
    """``--det_all``: see the module docstring. Reads each page once, eight threads."""
    df = pd.read_parquet(derived / "manifest_sfxp_det_clean.parquet")
    df = df[
        df.split.isin(a.splits)
        & df.cat.isin(["pos", "pos_mixed", "neg_speech"])
        & df.clean.isna()
    ].copy()
    rel = "crops_sfxp/det/" + df.split + "/" + df.book + "_" + df.id + ".png"
    todo = df.path.isna()
    df["path"] = rel
    cut = df[todo & ~df.path.map(lambda p: (derived / p).exists())]
    t0 = time.time()

    def one(g):
        (book, pg), rows = g
        im = cv2.imread(str(m109.page_path(book, int(pg))))
        for b, p in zip(rows.box, rows.path):
            (derived / p).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(derived / p), im[b[1] : b[3], b[0] : b[2]])
        return len(rows)

    n = 0
    with ThreadPoolExecutor(8) as pool:
        for k in pool.map(one, cut.groupby(["book", "page"])):
            n += k
            if n % 10000 < k:
                print(
                    f"[det_all] {n} / {len(cut)} cut, {time.time() - t0:.0f} s",
                    flush=True,
                )
    df.to_parquet(derived / "manifest_sfxp_det_all.parquet", index=False)
    print(
        f"[det_all] {n} crops cut in {time.time() - t0:.0f} s; "
        f"{df.groupby(['split', 'cat']).size().to_dict()}",
        flush=True,
    )


def build_speech(a, derived: Path):
    src = pd.read_parquet(derived / "manifest.parquet")
    src = src[(src.kind == "speech") & (src.split.isin(a.splits))]
    if "source" in src.columns:
        src = src[src.source.fillna("grey") == "grey"]
    rows, cur, img = [], None, None
    for r in src.sort_values(["book", "page"]).itertuples():
        if cur != (r.book, r.page):
            img, cur = cv2.imread(str(m109.page_path(r.book, r.page))), (r.book, r.page)
        q = json.loads(r.poly)
        x0, y0, x1, y1 = min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2])
        H, W = img.shape[:2]
        c = margin_box(x0, y0, x1, y1, W, H)
        rel = f"crops_sfxp/speech/{r.split}/{r.book}_{r.page:03d}_{r.id}.png"
        (derived / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(derived / rel), img[c[1] : c[3], c[0] : c[2]])
        rows.append(
            dict(
                split=r.split,
                kind="speech",
                id=r.id,
                book=r.book,
                page=r.page,
                text=r.text,
                orient=r.orient,
                w=c[2] - c[0],
                h=c[3] - c[1],
                path=rel,
                source="sfxp_speech",
            )
        )
    df = pd.DataFrame(rows)
    df.to_parquet(derived / "manifest_sfxp_speech.parquet", index=False)
    print(f"[speech] {df.split.value_counts().to_dict()}", flush=True)
    if LINES.exists():
        build_lines(a, derived)


def build_ellipsis(a, derived: Path):
    """:func:`ellipsis_fold` over ``manifest_sfxp_speech`` in place;
    ``.pre_ellipsis.parquet`` backup, once."""
    path = derived / "manifest_sfxp_speech.parquet"
    df = pd.read_parquet(path)
    new = df.text.map(ellipsis_fold)
    n = int((new != df.text).sum())
    if n:
        bak = path.with_suffix(".pre_ellipsis.parquet")
        if not bak.exists():
            import shutil

            shutil.copy2(path, bak)
        df["text"] = new
        df.to_parquet(path, index=False)
    print(f"[ellipsis] {n} rows folded", flush=True)


def build_lines(a, derived: Path):
    """Line breaks from :data:`LINES` into ``manifest_sfxp_speech``: a speech row the
    annotation covers gets its text re-broken per line; an annotated ``<text>`` of a
    train / val book the manifest lacks (the grey manifest samples speech per book) is
    cut at the same margin and appended (``source = m109_lines``). Test books are left
    alone (``sfxp_gt`` is the gate set). Idempotent; ``.pre_lines.parquet`` backup."""
    import xml.etree.ElementTree as ET

    ann = {}
    for pg in json.loads(LINES.read_text(encoding="utf-8")).values():
        for t in pg["texts"]:
            ann[(pg["book"], int(pg["page_index"]), t["id"])] = t
    raw = {}
    for book in sorted({k[0] for k in ann}):
        root = ET.parse(m109.manga109s_root() / "annotations" / f"{book}.xml").getroot()
        for p in root.iter("page"):
            for e in p.findall("text"):
                raw[(book, int(p.get("index")), e.get("id"))] = e
    split_of = m109.split_of(m109.load_split())
    path = derived / "manifest_sfxp_speech.parquet"
    df = pd.read_parquet(path)
    have = {(r.book, int(r.page), str(r.id)): i for i, r in enumerate(df.itertuples())}
    rebroken = same = bad = 0
    for k, i in have.items():
        if k not in ann:
            continue
        t = line_broken(raw[k].text or "", ann[k]["line_lengths"])
        if t is not None:
            t = ellipsis_fold(t)
        if t is None or "".join(t.split()) != "".join(
            ellipsis_fold(df.text.iat[i]).split()
        ):
            bad += 1
            continue
        if t == df.text.iat[i]:
            same += 1
        else:
            df.iat[i, df.columns.get_loc("text")] = t
            rebroken += 1
    new, cur, img, skipped = [], None, None, {"test": 0, "bad": 0}
    for k in sorted(set(ann) - set(have)):
        book, page, tid = k
        sp = split_of[book]
        if sp == "test" or sp not in a.splits:
            skipped["test"] += 1
            continue
        e = raw[k]
        t = line_broken(e.text or "", ann[k]["line_lengths"])
        if not t:
            skipped["bad"] += 1
            continue
        t = ellipsis_fold(t)
        if cur != (book, page):
            img, cur = cv2.imread(str(m109.page_path(book, page))), (book, page)
        x0, y0, x1, y1 = (int(e.get(c)) for c in ("xmin", "ymin", "xmax", "ymax"))
        H, W = img.shape[:2]
        c = margin_box(x0, y0, x1, y1, W, H)
        rel = f"crops_sfxp/speech/{sp}/{book}_{page:03d}_{tid}.png"
        (derived / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(derived / rel), img[c[1] : c[3], c[0] : c[2]])
        ls = ann[k]["lines"]
        tall = sum((ln["ymax"] - ln["ymin"]) > (ln["xmax"] - ln["xmin"]) for ln in ls)
        new.append(
            dict(
                split=sp,
                kind="speech",
                id=tid,
                book=book,
                page=page,
                text=t,
                orient="vertical" if 2 * tall >= len(ls) else "horizontal",
                w=c[2] - c[0],
                h=c[3] - c[1],
                path=rel,
                source="m109_lines",
            )
        )
    if new:
        df = pd.concat([df, pd.DataFrame(new)], ignore_index=True)
    bak = path.with_suffix(".pre_lines.parquet")
    if (rebroken or new) and not bak.exists():
        import shutil

        shutil.copy2(path, bak)
    df.to_parquet(path, index=False)
    added = pd.Series([r["split"] for r in new]).value_counts().to_dict()
    print(
        f"[lines] re-broken {rebroken}, already broken {same}, text mismatch {bad}; "
        f"added {added}; skipped {skipped}; manifest {len(df)} rows",
        flush=True,
    )
    build_ellipsis(a, derived)


def build_gt(a, derived: Path):
    src = pd.read_parquet(derived / "manifest.parquet")
    if "source" in src.columns:
        src = src[src.source.fillna("grey") == "grey"]
    src = src[src.split.isin(a.gt)]
    rows, cur, img, lost = [], None, None, 0
    for r in src.sort_values(["book", "page"]).itertuples():
        if cur != (r.book, r.page):
            img, cur = cv2.imread(str(m109.page_path(r.book, r.page))), (r.book, r.page)
        poly = json.loads(r.poly)
        crop, ori = deskew_crop(img, poly, cap_frac(poly), 1)
        if crop is None:
            lost += 1
            continue
        rel = f"crops_sfxp/gt/{r.split}/{r.kind}/{r.book}_{r.page:03d}_{r.id}.png"
        (derived / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(derived / rel), crop)
        d = r._asdict()
        d.pop("Index")
        d.update(
            orient=ori, w=crop.shape[1], h=crop.shape[0], path=rel, source="sfxp_gt"
        )
        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_parquet(derived / "manifest_sfxp_gt.parquet", index=False)
    print(
        f"[gt] {df.groupby(['split', 'kind']).size().to_dict()}; lost {lost}",
        flush=True,
    )


PUNCT_ONLY = re.compile(r"^[・…‥.。、!！?？♡♥~〜ー―\-\s☆★♪]+$")


def missed_sub(gray, pts, text: str, joined: bool) -> str:
    """The sub-type of a ``missed`` COO polygon (no detector box covers 20 % of it),
    first rule that fits: ``a_punct`` (marks only), ``b_page_edge`` (touches the page
    edge), ``c_huge`` (long side ≥ 300 px), ``d_joined``, ``e_dark_bg`` (a 4 px ring
    round the polygon averages under 150 grey), ``f_rotated`` (≥ 25° off axis),
    ``g_tiny`` (long side < 40 px), ``h_one_char``, else ``i_plain``."""
    H, W = gray.shape
    p = np.asarray(pts, np.int32).reshape(-1, 2)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    long = max(x1 - x0 + 1, y1 - y0 + 1)
    if PUNCT_ONLY.match(text):
        return "a_punct"
    if x0 <= 3 or y0 <= 3 or x1 >= W - 4 or y1 >= H - 4:
        return "b_page_edge"
    if long >= 300:
        return "c_huge"
    if joined:
        return "d_joined"
    mk = np.zeros_like(gray, np.uint8)
    cv2.fillPoly(mk, [p.reshape(-1, 1, 2)], 1)
    ring = cv2.dilate(mk, np.ones((9, 9), np.uint8)).astype(bool) & ~mk.astype(bool)
    if ring.any() and gray[ring].mean() < 150:
        return "e_dark_bg"
    _, _, ang = cv2.minAreaRect(p.astype(np.float32))
    if abs(((ang + 45) % 90) - 45) >= 25:
        return "f_rotated"
    if long < 40:
        return "g_tiny"
    if len(re.sub(r"\s", "", text)) <= 1:
        return "h_one_char"
    return "i_plain"


def build_missed(a, derived: Path):
    """``--missed`` → ``manifest_sfxp_missed.parquet``: the train COO polygons no detector
    box covers (``sfxp_types_train.parquet`` type ``missed``, written by ``--clean``),
    each with its :func:`missed_sub` type and ``o_sfx``, the largest share of another COO
    polygon's bbox inside this polygon's capped-margin bbox. The ``--missed_keep`` types
    are cut from the polygon as the gate crops are (``crops.deskew_crop``, capped margin)
    and carry a ``path``, except ``--missed_alone`` types with ``o_sfx ≥ 0.1`` (another
    SFX in the crop); ``load_sfxp(missed_manifest=…)`` adds those as ``SFX:`` rows."""
    allty = pd.read_parquet(derived / "sfxp_types_train.parquet")
    page_polys = {
        k: [(str(c), json.loads(q)) for c, q in zip(g.coo_id, g.poly)]
        for k, g in allty.groupby(["book", "page"])
    }
    ty = allty[allty.type == "missed"]

    def bb(q):
        q = np.asarray(q).reshape(-1, 2)
        return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()

    def o_sfx(book, page, cid, poly):
        x0, y0, x1, y1 = bb(poly)
        m = min(4, 0.3 * min(x1 - x0, y1 - y0))
        best = 0.0
        for oid, q in page_polys[(book, page)]:
            if oid == cid:
                continue
            a = bb(q)
            iw = max(0, min(a[2], x1 + m) - max(a[0], x0 - m))
            ih = max(0, min(a[3], y1 + m) - max(a[1], y0 - m))
            best = max(best, iw * ih / max(1, (a[2] - a[0]) * (a[3] - a[1])))
        return best

    rows, cur, img, gray, lost = [], None, None, None, 0
    for r in ty.sort_values(["book", "page"]).itertuples():
        if cur != (r.book, r.page):
            img, cur = cv2.imread(str(m109.page_path(r.book, r.page))), (r.book, r.page)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        poly = json.loads(r.poly)
        sub = missed_sub(gray, poly, r.coo_text, bool(r.joined))
        ov = o_sfx(r.book, r.page, str(r.coo_id), poly)
        d = dict(
            split="train",
            kind="sfx",
            id=str(r.coo_id),
            book=r.book,
            page=int(r.page),
            text=r.coo_text,
            sub=sub,
            o_sfx=round(float(ov), 3),
            source=f"missed_{sub}",
            orient=None,
            w=0,
            h=0,
            path=None,
        )
        if sub in a.missed_keep and not (sub in a.missed_alone and ov >= 0.1):
            crop, ori = deskew_crop(img, poly, cap_frac(poly), 1)
            if crop is None:
                lost += 1
            else:
                rel = (
                    f"crops_sfxp/missed/train/{r.book}_{int(r.page):03d}_{r.coo_id}.png"
                )
                (derived / rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(derived / rel), crop)
                d.update(orient=ori, w=crop.shape[1], h=crop.shape[0], path=rel)
        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_parquet(derived / "manifest_sfxp_missed.parquet", index=False)
    print(
        f"[missed] {df['sub'].value_counts().sort_index().to_dict()}; "
        f"cut {df[df.path.notna()]['sub'].value_counts().to_dict()}; lost {lost}",
        flush=True,
    )


def build_kozh(a, derived: Path):
    gel = {}
    for line in open(GELNOTE / "manifest.jsonl", encoding="utf-8"):
        r = json.loads(line)
        gel[r["id"]] = r
    lab = {}
    for line in open(SWAPRELAX / "labels.jsonl", encoding="utf-8"):
        r = json.loads(line)
        lab[r["crop"]] = r
    jobs = []
    for name in ("pseudo_gelnote_ko", "pseudo_gelnote_zh", "pseudo_swaprelax_v5"):
        src = pd.read_parquet(derived / f"manifest_{name}.parquet")
        for r in src.itertuples():
            if name.startswith("pseudo_gelnote"):
                g = gel[r.id]
                page, box = GELNOTE / g["page_path"], g["box"]
            else:
                s = lab["crops/" + os.path.basename(r.path)]
                page, box = REPO / s["page"], s["box"]
            jobs.append((name, r, page, box))
    jobs.sort(key=lambda j: str(j[2]))
    rows, cur, img, miss = [], None, None, 0
    out = derived / "crops_sfxp" / "kozh"
    out.mkdir(parents=True, exist_ok=True)
    for name, r, page, box in jobs:
        if cur != page:
            img, cur = (cv2.imread(str(page)) if page.exists() else None), page
        if img is None:
            miss += 1
            continue
        H, W = img.shape[:2]
        c = margin_box(*box, W, H)
        rel = f"crops_sfxp/kozh/{name.removeprefix('pseudo_')}_{r.id}.png"
        cv2.imwrite(str(derived / rel), img[c[1] : c[3], c[0] : c[2]])
        rows.append(
            dict(
                split="train",
                kind="speech",
                id=r.id,
                book=r.book,
                page=r.page,
                text=r.text,
                orient=r.orient,
                w=c[2] - c[0],
                h=c[3] - c[1],
                path=rel,
                source=name.removeprefix("pseudo_"),
            )
        )
    df = pd.DataFrame(rows)
    df.to_parquet(derived / "manifest_sfxp_kozh.parquet", index=False)
    print(
        f"[kozh] {df.source.value_counts().to_dict()}; page missing {miss}", flush=True
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--det", action="store_true")
    ap.add_argument("--speech", action="store_true")
    ap.add_argument("--kozh", action="store_true")
    ap.add_argument(
        "--lines",
        action="store_true",
        help="line breaks + extra speech rows from assets/manga109s_line_annotations.json "
        "(also run at the end of --speech)",
    )
    ap.add_argument(
        "--ellipsis",
        action="store_true",
        help="dot runs of 3k → k × … in manifest_sfxp_speech (also run after --lines)",
    )
    ap.add_argument(
        "--clean", action="store_true", help="det manifest → manifest_sfxp_det_clean"
    )
    ap.add_argument(
        "--det_all",
        action="store_true",
        help="--clean's text boxes, every one cut → manifest_sfxp_det_all (main / sub review)",
    )
    ap.add_argument("--gt", nargs="+", help="splits to re-cut for the gate evals")
    ap.add_argument(
        "--missed",
        action="store_true",
        help="train COO polygons the detector misses → manifest_sfxp_missed (needs --clean's "
        "sfxp_types_train)",
    )
    ap.add_argument(
        "--missed_keep",
        nargs="+",
        default=["i_plain", "a_punct", "g_tiny", "e_dark_bg"],
        help="missed sub-types cut as SFX: rows (missed_sub)",
    )
    ap.add_argument(
        "--missed_alone",
        nargs="+",
        default=["g_tiny", "e_dark_bg"],
        help="of --missed_keep, the types cut only when no other COO SFX reaches the crop "
        "(o_sfx < 0.1)",
    )
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument(
        "--neg_keep", type=float, default=0.15, help="share of neg_speech crops written"
    )
    ap.add_argument("--limit_pages", type=int, help="pages per book (smoke)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    derived = m109.derived_root()
    if a.speech:
        build_speech(a, derived)
    elif a.lines:
        build_lines(a, derived)
    elif a.ellipsis:
        build_ellipsis(a, derived)
    if a.kozh:
        build_kozh(a, derived)
    if a.gt:
        build_gt(a, derived)
    if a.det:
        build_det(a, derived)
    if a.clean:
        build_clean(a, derived)
    if a.det_all:
        build_det_all(a, derived)
    if a.missed:
        build_missed(a, derived)


if __name__ == "__main__":
    main()
