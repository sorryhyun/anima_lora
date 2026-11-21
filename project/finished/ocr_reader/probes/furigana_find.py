#!/usr/bin/env python3
"""Furigana (ruby) candidates on the Manga109 speech crops, for the ``OCR2:`` sub prompt.

Manga109 labels carry no ruby, so a crop showing furigana has no label for it. This
probe finds the ruby by ink projection and reads it alone with a reader:

``find`` (CPU) — vertical crops whose label holds a kanji. Ink as ``linebreak_auto``
(Otsu, edge-touching components dropped); the x projection splits into runs. Main
columns are runs ≥ ``--main_frac`` × the median wide run (``wm``, the main glyph size);
a ruby run is narrower than ``--ruby_frac`` · wm, sits right of a main column within
``--ruby_gap`` · wm, and its components are no taller than ``--ruby_glyph`` · wm. A main
run with a valley in its right part (ruby touching the column in projection) is split
there. A ruby run splits along y at gaps ≥ ``--group_gap`` · wm into groups (one per
word). Each group is aligned to the label: main column j (right to left) ↔ label line j
when the counts agree; the group's y-span maps onto the line's characters by
``linebreak_auto.glyph_weight``; ``aligned`` = a kanji lies under it.

``read`` (GPU, daemon) — each group cut at the 4 px margin and read with ``OCR:``.

``sheet`` — contact sheet: main columns blue, ruby groups red, read and kanji under.

    .venv/bin/python project/finished/ocr_reader/probes/furigana_find.py find --n 3000
    make daemon-run ARGS="--stall-timeout 600 project/finished/ocr_reader/probes/furigana_find.py read --ckpt output/ocr/sfxpc_1121_0117/best"
    .venv/bin/python project/finished/ocr_reader/probes/furigana_find.py sheet --n 36
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ocr"))
sys.path.insert(0, str(HERE))
import manga109 as m109  # noqa: E402
from build_sfx_prefix import margin_box  # noqa: E402
from linebreak_auto import glyph_weight, ink_mask, runs  # noqa: E402

KANJI = re.compile(r"[一-鿿々〆ヶ]")
KANA = re.compile(r"^[ぁ-ゟ゠-ヿー・]+$")
OUT = m109.REPO / "output/ocr/furigana"


def split_touching(prof: np.ndarray, a: int, b: int, wm: float, ruby_frac: float):
    """A wide run whose right part holds ruby touching the column: cut at the
    projection valley in [a + 0.7 wm, b − 0.15 wm) when the valley is ≤ 15 % of the
    run's peak and the right piece is ruby-wide."""
    if b - a < 1.25 * wm:
        return [(a, b)]
    lo, hi = int(a + 0.7 * wm), int(b - 0.15 * wm)
    if hi <= lo:
        return [(a, b)]
    k = lo + int(np.argmin(prof[lo:hi]))
    if prof[k] <= 0.15 * prof[a:b].max() and (b - k) <= ruby_frac * wm:
        return [(a, k), (k + 1, b)]
    return [(a, b)]


def find_ruby(crop: np.ndarray, label: str, a) -> dict:
    ink = ink_mask(crop)
    H, W = ink.shape
    prof = ink.sum(axis=0)
    rs = [(int(x0), int(x1)) for x0, x1 in runs(prof > 0)]
    if not rs:
        return dict(wm=0, mains=[], groups=[])
    widths = np.array([x1 - x0 for x0, x1 in rs])
    wide = widths[widths >= 0.6 * widths.max()]
    wm = float(np.median(wide))
    pieces = []
    for x0, x1 in rs:
        pieces += split_touching(prof, x0, x1, wm, a.ruby_frac)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    mains, rubies = [], []
    for x0, x1 in pieces:
        w = x1 - x0
        if w >= a.main_frac * wm:
            mains.append((x0, x1))
        elif w <= a.ruby_frac * wm:
            rubies.append((x0, x1))
    mains.sort(key=lambda r: -r[0])  # right to left
    lines = label.split("\n")
    groups = []
    for rx0, rx1 in rubies:
        # the main column on the left, within the gap
        left = [m for m in mains if m[1] <= rx0 + 1 and rx0 - m[1] <= a.ruby_gap * wm]
        if not left:
            continue
        col = max(left, key=lambda m: m[1])
        j = mains.index(col)
        rows = np.flatnonzero(ink[:, rx0:rx1].any(axis=1))
        gs = [(int(y0), int(y1)) for y0, y1 in runs(np.isin(np.arange(H), rows))]
        merged = []
        for y0, y1 in gs:
            if merged and y0 - merged[-1][1] < a.group_gap * wm:
                merged[-1] = (merged[-1][0], y1)
            else:
                merged.append((y0, y1))
        cy = np.flatnonzero(ink[:, col[0] : col[1]].any(axis=1))
        cb0, cb1 = (int(cy[0]), int(cy[-1]) + 1) if len(cy) else (0, H)
        for y0, y1 in merged:
            sub = lab[y0:y1, rx0:rx1]
            ids = np.unique(sub[sub > 0])
            tall = max((st[i, 3] for i in ids), default=0)
            if tall > a.ruby_glyph * wm or (y1 - y0) < 0.25 * wm or not len(ids):
                continue
            # the whole components the strip touches: a projection cut or a split
            # touching run can clip a kana's left strokes
            gx0 = max(int(min(st[i, 0] for i in ids)), int(col[1] - 0.15 * wm))
            gx1 = int(max(st[i, 0] + st[i, 2] for i in ids))
            y0 = min(y0, int(min(st[i, 1] for i in ids)))
            y1 = max(y1, int(max(st[i, 1] + st[i, 3] for i in ids)))
            g = dict(x0=gx0, x1=gx1, y0=y0, y1=y1, col=j, span="", aligned=False)
            if len(lines) == len(mains):
                line = lines[j]
                wts = np.array([glyph_weight(c) for c in line])
                if wts.sum() > 0:
                    edges = cb0 + np.concatenate([[0], np.cumsum(wts)]) / wts.sum() * (
                        cb1 - cb0
                    )
                    mid = (edges[:-1] + edges[1:]) / 2
                    pad = 0.25 * wm
                    under = [i for i, m in enumerate(mid) if y0 - pad <= m <= y1 + pad]
                    span = "".join(line[i] for i in under)
                    g["span"] = span
                    g["aligned"] = bool(KANJI.search(span))
            groups.append(g)
    return dict(wm=wm, mains=mains, groups=groups)


def rows(a):
    import pandas as pd

    d = m109.derived_root()
    df = pd.read_parquet(d / f"manifest_{a.manifest}.parquet")
    df = df[(df.orient != "horizontal") & df.text.str.contains(KANJI)]
    if a.n:
        df = df.sample(min(a.n, len(df)), random_state=a.seed)
    return d, df


def find(a):
    d, df = rows(a)
    OUT.mkdir(parents=True, exist_ok=True)
    tot = dict(crops=0, with_groups=0, groups=0, aligned=0, count_ok=0)
    with open(OUT / f"{a.name}.jsonl", "w", encoding="utf-8") as f:
        for r in df.itertuples():
            crop = cv2.imread(str(d / r.path))
            res = find_ruby(crop, r.text, a)
            tot["crops"] += 1
            tot["count_ok"] += len(r.text.split("\n")) == len(res["mains"])
            tot["with_groups"] += bool(res["groups"])
            tot["groups"] += len(res["groups"])
            tot["aligned"] += sum(g["aligned"] for g in res["groups"])
            f.write(
                json.dumps(
                    dict(
                        split=r.split,
                        book=r.book,
                        page=int(r.page),
                        id=str(r.id),
                        path=r.path,
                        text=r.text,
                        **res,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps(tot))
    print(f"wrote {OUT / (a.name + '.jsonl')}")


def read(a):
    import eval_manga109 as ev
    from linebreak_auto import fast_read

    d = m109.derived_root()
    recs = [json.loads(x) for x in open(OUT / f"{a.name}.jsonl", encoding="utf-8")]
    crops, where = [], []
    for ri, r in enumerate(recs):
        if not r["groups"]:
            continue
        img = cv2.imread(str(d / r["path"]))
        H, W = img.shape[:2]
        for gi, g in enumerate(r["groups"]):
            c = margin_box(g["x0"], g["y0"], g["x1"], g["y1"], W, H)
            cr = img[c[1] : c[3], c[0] : c[2]]
            if (
                a.up > 1
            ):  # ruby glyphs are ~6 px wide: the reader answers "" at that size
                cr = cv2.resize(
                    cr, None, fx=a.up, fy=a.up, interpolation=cv2.INTER_CUBIC
                )
            if a.pad:
                p = int(a.pad * a.up)
                cr = cv2.copyMakeBorder(
                    cr, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
            crops.append(cr)
            where.append((ri, gi))
    print(f"{len(crops)} ruby groups to read", flush=True)
    reads = fast_read(ev.Vl16Reader(a.ckpt, "cuda"), crops, [16] * len(crops), a.bs, 2)
    for (ri, gi), t in zip(where, reads):
        recs[ri]["groups"][gi]["read"] = t
        recs[ri]["groups"][gi]["kana"] = bool(KANA.match(t))
    with open(OUT / f"{a.name}_read{a.tag}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gs = [g for r in recs for g in r["groups"]]
    print(
        json.dumps(
            dict(
                groups=len(gs),
                kana=sum(g["kana"] for g in gs),
                aligned_kana=sum(g["kana"] and g["aligned"] for g in gs),
                empty=sum(g["read"] == "" for g in gs),
            )
        )
    )


def sheet(a):
    from PIL import Image, ImageDraw, ImageFont

    d = m109.derived_root()
    src = OUT / f"{a.name}_read.jsonl"
    if not src.is_file():
        src = OUT / f"{a.name}.jsonl"
    recs = [json.loads(x) for x in open(src, encoding="utf-8")]
    rng = np.random.default_rng(a.seed)
    pick = [recs[i] for i in rng.permutation(len(recs))[: a.n]]
    font = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 14
    )
    T, TXT, COLS = 260, 110, 6
    out = Image.new("RGB", (COLS * T, -(-len(pick) // COLS) * (T + TXT)), "white")
    dr = ImageDraw.Draw(out)
    for i, r in enumerate(pick):
        img = Image.open(d / r["path"]).convert("RGB")
        o = ImageDraw.Draw(img)
        for x0, x1 in r["mains"]:
            o.rectangle((x0, 0, x1 - 1, img.height - 1), outline=(80, 80, 255))
        for g in r["groups"]:
            col = (230, 0, 0) if g["aligned"] else (255, 150, 0)
            o.rectangle(
                (g["x0"], g["y0"], g["x1"] - 1, g["y1"] - 1), outline=col, width=2
            )
        s = min((T - 8) / img.width, (T - 8) / img.height, 3.0)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))
        X, Y = (i % COLS) * T, (i // COLS) * (T + TXT)
        out.paste(img, (X + (T - img.width) // 2, Y + 4))
        dr.rectangle((X, Y, X + T - 1, Y + T + TXT - 1), outline="gray")
        lines = [
            f"{i:02d} {r['book']}_{r['page']}_{r['id']}",
            r["text"].replace("\n", "/")[:20],
        ]
        for g in r["groups"][:4]:
            lines.append(f"{g['span'] or '?'} → {g.get('read', '')}")
        for k, ln in enumerate(lines):
            dr.text(
                (X + 4, Y + T + 2 + k * 18),
                ln,
                font=font,
                fill="blue" if k == 0 else "black",
            )
    p = OUT / f"{a.name}_sheet_{a.seed}.png"
    out.save(p)
    print(p)


def doubt(r: dict) -> int:
    """Groups that read as something other than kana or sit over no kanji — the review
    page's ``sus`` for the furigana source."""
    return sum(not (g.get("kana") and g["aligned"]) for g in r["groups"])


def batches(a):
    """The ``--n`` most doubtful crops (the review page's order: doubt, then id) cut into
    batches of ``--per`` for agent review: ``agent/<tag>/batch_<k>/rows.tsv`` (n, id,
    label, proposal, groups; ``\\n`` escaped) and ``sheet_<s>.png``, ``--rows`` crops a
    sheet. A tile is the crop with each group boxed and numbered, then each group
    enlarged beside the column it annotates."""
    from PIL import Image, ImageDraw, ImageFont

    d = m109.derived_root()
    recs = [
        json.loads(x)
        for x in open(OUT / f"{a.name}_read{a.tag}.jsonl", encoding="utf-8")
    ]
    recs = [r for r in recs if r["groups"]]
    for r in recs:
        r["rid"] = f"{r['book']}_{int(r['page']):03d}_{r['id']}"
        r["groups"].sort(key=lambda g: (g["col"], g["y0"]))
    recs.sort(key=lambda r: (-doubt(r), r["rid"]))
    recs = recs[: a.n]
    font = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18
    )
    q = lambda t: t.replace("\\", "\\\\").replace("\t", " ").replace("\n", "\\n")  # noqa: E731
    root = OUT / "agent" / a.batch_tag
    RH, CW, GW, PERLINE = 320, 300, 150, 10
    for b0 in range(0, len(recs), a.per):
        bdir = root / f"batch_{b0 // a.per}"
        bdir.mkdir(parents=True, exist_ok=True)
        part = recs[b0 : b0 + a.per]
        with open(bdir / "rows.tsv", "w", encoding="utf-8") as f:
            f.write("n\tid\tlabel\tproposal\tgroups\n")
            for n, r in enumerate(part):
                cols = sorted({g["col"] for g in r["groups"] if g.get("read")})
                prop = "\n".join(
                    " ".join(
                        g["read"]
                        for g in r["groups"]
                        if g["col"] == c and g.get("read")
                    )
                    for c in cols
                )
                gs = " | ".join(
                    f"g{k + 1} col{g['col'] + 1} {g['span'] or '?'}→{g.get('read') or '∅'}"
                    for k, g in enumerate(r["groups"])
                )
                f.write(f"{n}\t{r['rid']}\t{q(r['text'])}\t{q(prop)}\t{q(gs)}\n")
        for s0 in range(0, len(part), a.rows):
            chunk = part[s0 : s0 + a.rows]
            # zoom cells wrap after PERLINE groups; a crop's block is as tall as it needs
            hts = [RH * max(1, -(-len(r["groups"]) // PERLINE)) for r in chunk]
            ng = min(PERLINE, max(len(r["groups"]) for r in chunk))
            sheet = Image.new("RGB", (CW + ng * GW, sum(hts)), "white")
            dr = ImageDraw.Draw(sheet)
            Y = 0
            for i, r in enumerate(chunk):
                clean = Image.open(d / r["path"]).convert("RGB")
                img = clean.copy()
                o = ImageDraw.Draw(img)
                for g in r["groups"]:
                    o.rectangle(
                        (g["x0"] - 1, g["y0"] - 1, g["x1"], g["y1"]),
                        outline=(230, 0, 0),
                    )
                s = min((CW - 40) / img.width, (RH - 30) / img.height, 4.0)
                im = img.resize(
                    (max(1, int(img.width * s)), max(1, int(img.height * s)))
                )
                sheet.paste(im, (36, Y + 26))
                for k, g in enumerate(r["groups"]):
                    dr.text(
                        (36 + g["x1"] * s + 1, Y + 26 + g["y0"] * s - 4),
                        str(k + 1),
                        font=font,
                        fill=(230, 0, 0),
                    )
                dr.text((4, Y + 2), f"#{s0 + i}", font=font, fill=(0, 0, 200))
                for k, g in enumerate(r["groups"]):
                    x0 = max(0, int(g["x0"] - r["wm"] - 3))
                    zc = clean.crop(
                        (
                            x0,
                            max(0, g["y0"] - 3),
                            min(clean.width, g["x1"] + 3),
                            min(clean.height, g["y1"] + 3),
                        )
                    )
                    z = min((GW - 8) / zc.width, (RH - 40) / zc.height, 10.0)
                    zc = zc.resize(
                        (max(1, int(zc.width * z)), max(1, int(zc.height * z))),
                        Image.LANCZOS,
                    )
                    X = CW + (k % PERLINE) * GW
                    ZY = Y + (k // PERLINE) * RH
                    sheet.paste(zc, (X + 4, ZY + 30))
                    dr.text((X + 4, ZY + 4), f"g{k + 1}", font=font, fill=(230, 0, 0))
                Y += hts[i]
                dr.line((0, Y - 1, sheet.width, Y - 1), fill=(150, 150, 150))
            sheet.save(bdir / f"sheet_{s0 // a.rows:02d}.png")
        print(f"{bdir}: {len(part)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["find", "read", "sheet", "batches"])
    ap.add_argument("--per", type=int, default=400, help="batches: rows per batch")
    ap.add_argument("--rows", type=int, default=6, help="batches: crops per sheet")
    ap.add_argument("--batch_tag", default="top2000", help="batches: output folder")
    ap.add_argument("--name", default="ruby_v1")
    ap.add_argument(
        "--manifest",
        default="sfxp_speech",
        help="find: manifest_<name>.parquet under derived/ (mainsub_text: the <text> "
        "boxes outside the speech manifest, mainsub_text.py)",
    )
    ap.add_argument(
        "--n", type=int, default=0, help="find: sample size (0 = all); sheet: tiles"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--main_frac", type=float, default=0.7)
    ap.add_argument("--ruby_frac", type=float, default=0.6)
    ap.add_argument("--ruby_gap", type=float, default=0.5)
    ap.add_argument("--ruby_glyph", type=float, default=0.7)
    ap.add_argument("--group_gap", type=float, default=0.6)
    ap.add_argument("--ckpt", default="output/ocr/sfxpc_1121_0117/best")
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--up", type=float, default=1.0, help="read: upscale factor")
    ap.add_argument(
        "--pad", type=int, default=0, help="read: white border, px before --up"
    )
    ap.add_argument("--tag", default="", help="read: output suffix")
    a = ap.parse_args()
    {"find": find, "read": read, "sheet": sheet, "batches": batches}[a.cmd](a)


if __name__ == "__main__":
    main()
