#!/usr/bin/env python3
"""Plain contact sheet of the sincos hand labels: the crop, and what the label says.

No readers, no scoring — just ``crop → text_hand`` so the label file can be
eyeballed against the pixels. One tile per row of
``assets/sfx_labels_sincos.tsv``, in file order, cut with the reader's own
12 % pad (``anime_tools.ocr.sfx.crop_box``) so the tile is exactly what a
reader is asked to read.

Tile background carries ``status`` — **green** ``checked`` (user-certified),
white ``drafted``, **yellow** ``draft`` (uncertain, flagged by the drafter),
**grey** ``unchecked`` (never scored). The caption line is coloured by
``kind_hand``: **red** sfx, black speech, grey chrome; a ``≠`` marks a row
whose ``kind_rec`` disagrees, and ``·`` prefixes the note.

    python project/cjk_aware_anima_dit/probes/label_sheet.py                # all 975 rows
    … --kind sfx                                                            # SFX only
    … --status draft --status unchecked                                     # the uncertain ones
    … --cols 4 --rows 5                                                     # bigger crops

Writes ``output/tests/ocr_contact_sheet/<name>.pdf`` (+ ``<name>_pNN.png``
with ``--png``). CPU only (PIL + OpenCV); needs a CJK font.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401  (PDF save needs the JPEG codec registered)

from anime_tools.ocr.sfx import CROP_PAD, crop_box

REPO = Path(__file__).resolve().parents[3]
LINE = Path(__file__).resolve().parents[1]
LABELS = LINE / "assets/sfx_labels_sincos.tsv"
PAGES = REPO / "post_image_dataset/resized/sincos"
OUT = REPO / "output/tests/ocr_contact_sheet"

STATUS_BG = {
    "checked": (226, 244, 226),
    "drafted": (255, 255, 255),
    "draft": (252, 246, 214),
    "unchecked": (234, 234, 234),
}
KIND_COLOUR = {"sfx": (200, 0, 0), "speech": (20, 20, 20), "chrome": (130, 130, 130)}
INK = (20, 20, 20)
GREY = (120, 120, 120)
MARGIN = 24
CROP_H = 150
TEXT_H = 78
HEAD_H = 62


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    fam = "Noto Sans CJK JP" + (":bold" if bold else "")
    path = subprocess.run(
        ["fc-match", "-f", "%{file}", fam], capture_output=True, text=True
    ).stdout.strip()
    return ImageFont.truetype(path, size)


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    s = min(w / img.width, h / img.height, 3.0)
    return img.resize(
        (max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS
    )


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
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument(
        "--kind", action="append", choices=list(KIND_COLOUR), help="default: all"
    )
    ap.add_argument(
        "--status", action="append", choices=list(STATUS_BG), help="default: all"
    )
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--tile-w", type=int, default=210)
    ap.add_argument("--png", action="store_true", help="also write the per-page PNGs")
    ap.add_argument("--name")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    with args.labels.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if args.kind:
        rows = [r for r in rows if r["kind_hand"] in args.kind]
    if args.status:
        rows = [r for r in rows if r["status"] in args.status]
    if not rows:
        raise SystemExit("no rows matched")

    tally = Counter(f"{r['kind_hand']}/{r['status']}" for r in rows)
    name = args.name or "label_sheet" + (
        "_" + "-".join(args.kind) if args.kind else ""
    ) + ("_" + "-".join(args.status) if args.status else "")
    args.out.mkdir(parents=True, exist_ok=True)

    big, mid, small = font(24), font(17), font(13)
    tile_w, tile_h = args.tile_w, CROP_H + TEXT_H
    per_page = args.cols * args.rows
    page_w = MARGIN * 2 + args.cols * tile_w
    n_pages = (len(rows) - 1) // per_page + 1
    img_cache: dict[str, object] = {}
    pages: list[Image.Image] = []

    for p in range(0, len(rows), per_page):
        chunk = rows[p : p + per_page]
        n_rows = (len(chunk) - 1) // args.cols + 1
        page = Image.new(
            "RGB", (page_w, MARGIN * 2 + HEAD_H + n_rows * tile_h), "white"
        )
        d = ImageDraw.Draw(page)
        d.text(
            (MARGIN, MARGIN),
            f"sincos hand labels — {args.labels.name}   ({len(rows)} rows, page {p // per_page + 1} / {n_pages})",
            font=font(20, bold=True),
            fill=INK,
        )
        for j, ln in enumerate(
            (
                "bg: green = checked · white = drafted · yellow = draft (uncertain) · grey = unchecked",
                "caption colour = kind (red sfx / black speech / grey chrome), ≠ = kind_rec disagrees"
                f"   ·   crop = reader's {CROP_PAD:.0%} pad   ·   … = label longer than the tile",
            )
        ):
            d.text((MARGIN, MARGIN + 26 + 16 * j), ln, font=small, fill=GREY)
        for i, r in enumerate(chunk):
            cx = MARGIN + (i % args.cols) * tile_w
            cy = MARGIN + HEAD_H + (i // args.cols) * tile_h
            d.rectangle(
                [cx + 2, cy + 2, cx + tile_w - 4, cy + tile_h - 6],
                fill=STATUS_BG.get(r["status"], (255, 255, 255)),
                outline=(210, 210, 210),
            )
            if r["stem"] not in img_cache:
                img_cache[r["stem"]] = cv2.imread(str(PAGES / f"{r['stem']}.png"))
            bgr = img_cache[r["stem"]]
            crop = crop_box(bgr, json.loads(r["box"])) if bgr is not None else None
            if crop is not None:
                im = fit(Image.fromarray(crop[:, :, ::-1]), tile_w - 16, CROP_H - 12)
                page.paste(
                    im,
                    (
                        cx + 6 + (tile_w - 16 - im.width) // 2,
                        cy + 6 + (CROP_H - 12 - im.height) // 2,
                    ),
                )
            else:
                d.text((cx + 10, cy + 60), "∅ empty box", font=small, fill=GREY)

            colour = KIND_COLOUR.get(r["kind_hand"], GREY)
            head = f"#{r['row']}  {r['kind_hand']}"
            if r["kind_rec"] != r["kind_hand"]:
                head += f" ≠{r['kind_rec']}"
            head += f"  {r['stem']}"
            d.text((cx + 8, cy + CROP_H - 4), head, font=small, fill=colour)
            # long speech lines drop to the smaller face + a third line before they clip
            text = r["text_hand"] or "∅ (blank)"
            fnt, cap = big, 2
            lines = wrap(text, fnt, tile_w - 18)
            if len(lines) > cap:
                fnt, cap = mid, 3
                lines = wrap(text, fnt, tile_w - 18)
            if len(lines) > cap:
                lines = lines[:cap]
                lines[-1] = lines[-1][:-1] + "…"
            ty = cy + CROP_H + 10
            for ln in lines:
                d.text((cx + 8, ty), ln, font=fnt, fill=INK if r["text_hand"] else GREY)
                ty += fnt.size + 4
        pages.append(page)

    pdf = args.out / f"{name}.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=96)
    if args.png:
        for i, page in enumerate(pages, 1):
            page.save(args.out / f"{name}_p{i:02d}.png")
    print(f"{len(rows)} labels → {len(pages)} pages: {pdf}")
    print("   ".join(f"{k} {v}" for k, v in sorted(tally.items())))


if __name__ == "__main__":
    main()
