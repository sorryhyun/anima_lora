#!/usr/bin/env python3
"""Contact sheet for an OCR records file: page + mask + boxes + the lines, for an eyeball.

Promoted from the 2026-09-05 session script. One tile per page in
``masked ∪ has-records``; left the resized page with the text mask tinted red and
every record's box drawn — **green** for PP-OCRv6, **magenta** for VL-only
(``engine=vl16_spotting``), **orange** for a PP box whose text rule 1b replaced
with the VL crop read; right the lines in reading order with engine / kind /
score, the PP read beside a replacement, and (``--baseline``) the lines the
baseline records had that the new file lacks and vice versa. Pages with a mask
and no line are flagged (the floor).

    python project/cjk_aware_anima_dit/probes/ocr_contact_sheet.py \\
        --records post_image_dataset/cjk_unmask/ocr_records_sincos_hybrid.jsonl \\
        --baseline post_image_dataset/cjk_unmask/ocr_records_sincos_ppocr_v2.jsonl \\
        --name sincos_hybrid

Writes ``output/tests/ocr_contact_sheet/<name>_pNN.png`` + ``<name>.pdf``.
CPU only (PIL); needs a CJK font (``fc-match 'Noto Sans CJK JP'``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401  (PDF save needs the JPEG codec registered)

REPO = Path(__file__).resolve().parents[3]

ENGINE_COLOUR = {
    "ppocr_v6": (0, 200, 0),
    "ppocr_v6+vl16_crop": (255, 140, 0),
    "vl16_spotting": (220, 0, 200),
    "sfx_reader": (0, 120, 255),  # O4: a MIT-mask component the SFX reader read
}


def engine_colour(engine: str):
    """``<engine>+sfx_reader`` (O4 re-read) keeps its base engine's colour."""
    return ENGINE_COLOUR.get(engine) or ENGINE_COLOUR.get(
        engine.removesuffix("+sfx_reader"), (128, 128, 128)
    )
KIND_COLOUR = {"speech": "black", "sfx": (200, 0, 0), "chrome": (120, 120, 120)}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    fam = "Noto Sans CJK JP" + (":bold" if bold else "")
    path = subprocess.run(
        ["fc-match", "-f", "%{file}", fam], capture_output=True, text=True
    ).stdout.strip()
    return ImageFont.truetype(path, size)


def load(path: Path) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by[r["stem"]].append(r)
    return by


def wrap(text: str, fnt, width: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        if fnt.getlength(cur + ch) > width and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines or [""]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--shard", default="sincos")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", type=Path, default=REPO / "output/tests/ocr_contact_sheet")
    ap.add_argument(
        "--only",
        choices=["all", "floor", "vl", "changed"],
        default="all",
        help="floor: masked pages only; vl: pages with a VL-only line; changed: pages differing from the baseline",
    )
    opts = ap.parse_args()
    name = opts.name or opts.records.stem
    out = opts.out
    out.mkdir(parents=True, exist_ok=True)

    resized = REPO / "post_image_dataset/resized" / opts.shard
    masks = REPO / "post_image_dataset/masks" / opts.shard
    recs = load(opts.records)
    base = load(opts.baseline) if opts.baseline else {}
    masked = {p.name[: -len("_mask.png")] for p in masks.glob("*_mask.png")}
    stems = sorted(masked | set(recs))
    if opts.only == "floor":
        stems = [s for s in stems if s in masked]
    elif opts.only == "vl":
        stems = [
            s
            for s in stems
            if any(r["engine"] == "vl16_spotting" for r in recs.get(s, []))
        ]
    elif opts.only == "changed":
        stems = [
            s
            for s in stems
            if [r["text"] for r in recs.get(s, [])]
            != [r["text"] for r in base.get(s, [])]
        ]

    F, FB, FS = font(19), font(21, True), font(16)
    IMG_W, TXT_W, PAD = 520, 600, 12
    TILE_W = IMG_W + TXT_W + 3 * PAD

    def tile(stem: str) -> Image.Image:
        im = Image.open(resized / f"{stem}.png").convert("RGB")
        W, H = im.size
        s = min(IMG_W / W, 640 / H)
        mp = masks / f"{stem}_mask.png"
        if mp.exists():
            m = np.array(Image.open(mp).convert("L").resize(im.size)) <= 127
            a = np.zeros((*m.shape, 4), np.uint8)
            a[m] = (255, 40, 40, 90)
            im = Image.alpha_composite(
                im.convert("RGBA"), Image.fromarray(a, "RGBA")
            ).convert("RGB")
        d = ImageDraw.Draw(im)
        page = recs.get(stem, [])
        for r in page:
            d.rectangle(
                r["box"],
                outline=ENGINE_COLOUR.get(r["engine"], (0, 0, 255)),
                width=max(2, int(2 / s)),
            )
        im = im.resize((int(W * s), int(H * s)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        for i, r in enumerate(page, 1):
            x0, y0 = r["box"][0] * s, r["box"][1] * s
            col = ENGINE_COLOUR.get(r["engine"], (0, 0, 255))
            d.rectangle([x0, y0 - 18, x0 + 26, y0], fill=col)
            d.text((x0 + 3, y0 - 18), str(i), font=FS, fill="white")

        rows: list[tuple[str, object, object]] = []
        n_vl = sum(r["engine"] == "vl16_spotting" for r in page)
        hdr = f"{stem}  {W}x{H}  mask={'yes' if stem in masked else 'NO'}  lines={len(page)}  vl-only={n_vl}"
        rows.append((hdr, FB, "black"))
        if not page and stem in masked:
            rows.append(("MASKED, NO OCR LINE (floor)", FB, (200, 0, 0)))
        for i, r in enumerate(page, 1):
            eng = {
                "ppocr_v6": "PP ",
                "ppocr_v6+vl16_crop": "PP→VL",
                "vl16_spotting": "VL ",
            }.get(r["engine"], r["engine"])
            sc = f"{r['score']:.2f}" if r.get("score") is not None else "  – "
            kind = r.get("kind", "speech")
            tag = {"sfx": "SFX", "chrome": "CHR", "speech": "   "}[kind]
            col = KIND_COLOUR[kind]
            if r["engine"] != "ppocr_v6":
                col = engine_colour(r["engine"]) if kind == "speech" else col
            for j, ln in enumerate(
                wrap(f"{i:>2} {eng} {sc} {tag} {r['text']}", F, TXT_W)
            ):
                rows.append((ln if j == 0 else "        " + ln, F, col))
            if r.get("pp_text"):
                for ln in wrap(
                    f"        PP read: {r['pp_text']}  [{r.get('rule1b', '')}]",
                    FS,
                    TXT_W,
                ):
                    rows.append((ln, FS, (150, 90, 0)))
        if opts.baseline:
            bt = [r["text"] for r in base.get(stem, [])]
            nt = [r["text"] for r in page]
            rows.append(("", FS, "black"))
            same = bt == nt
            rows.append(
                (
                    f"baseline {opts.baseline.stem}: "
                    + ("identical" if same else "DIFFERS"),
                    FB,
                    "black" if same else (180, 0, 120),
                )
            )
            if not same:
                for t in bt:
                    if t not in nt:
                        for ln in wrap("  baseline only: " + t, FS, TXT_W):
                            rows.append((ln, FS, (180, 0, 120)))
                for t in nt:
                    if t not in bt:
                        for ln in wrap("  new only: " + t, FS, TXT_W):
                            rows.append((ln, FS, (0, 90, 180)))
        out_rows = [(ln, f, c) for t, f, c in rows for ln in wrap(t, f, TXT_W)]
        th = sum(f.size + 6 for _, f, _ in out_rows) + 2 * PAD
        h = max(im.height, th) + 2 * PAD
        t = Image.new("RGB", (TILE_W, h), "white")
        t.paste(im, (PAD, PAD))
        d = ImageDraw.Draw(t)
        y = PAD
        for ln, f, c in out_rows:
            d.text((IMG_W + 2 * PAD, y), ln, font=f, fill=c)
            y += f.size + 6
        d.rectangle([0, 0, TILE_W - 1, h - 1], outline=(200, 200, 200))
        return t

    tiles = [tile(s) for s in stems]
    COLS, MAXH = 2, 2300
    pages: list[tuple[list, int]] = []
    page: list = []
    y = 0
    row: list = []
    for t in tiles:
        row.append(t)
        if len(row) == COLS:
            rh = max(tt.height for tt in row)
            if y + rh > MAXH and page:
                pages.append((page, y))
                page, y = [], 0
            page.append((row, y, rh))
            y += rh
            row = []
    if row:
        rh = max(tt.height for tt in row)
        if y + rh > MAXH and page:
            pages.append((page, y))
            page, y = [], 0
        page.append((row, y, rh))
        y += rh
    if page:
        pages.append((page, y))
    legend = (
        f"{name} · boxes: green = PP-OCRv6, orange = PP box re-read by VL-1.6 (rule 1b), "
        "magenta = VL-only (Spotting) · red tint = text mask · SFX/CHR = kind · numbers = reading order"
    )
    outs = []
    for i, (pg, ph) in enumerate(pages, 1):
        canvas = Image.new("RGB", (COLS * TILE_W, ph + 40), "white")
        d = ImageDraw.Draw(canvas)
        d.text((10, 8), f"p{i}/{len(pages)}  " + legend, font=FS, fill=(80, 80, 80))
        for r, yy, rh in pg:
            for j, t in enumerate(r):
                canvas.paste(t, (j * TILE_W, yy + 40))
        canvas.save(out / f"{name}_p{i:02d}.png")
        outs.append(canvas)
    if outs:
        outs[0].save(
            out / f"{name}.pdf", save_all=True, append_images=outs[1:], resolution=100
        )
    print(f"{len(stems)} tiles, {len(pages)} pages -> {out / (name + '.pdf')}")


if __name__ == "__main__":
    main()
