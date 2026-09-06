#!/usr/bin/env python3
"""Contact sheet for two readers on the sincos hand labels: crop | truth | A | B.

``eval_sfx.py`` writes one prediction row per label into
``output/ocr/eval/sfx_<name>.jsonl``; this joins two such files back to the
label boxes (``assets/sfx_labels_sincos.tsv``), cuts each crop with the
reader's own 12 % pad (``anime_tools.ocr.sfx.crop_box``, axis-aligned — the
eval's deskew of an axis-aligned box is the same crop) and lays the rows out
for an eyeball: the crop, the hand label, reader A's read, reader B's read,
each with sim and an exact tick, the row tinted **green** where B improved on
A (exact gained, or sim up by ≥ 0.05), **red** where it lost, plain where the
two agree. Disagreements come first, worst B-vs-A first.

    python project/cjk_aware_anima_dit/probes/ocr_eval_sheet.py \\
        --a vl16_tower_lr1e-5 --b vl16_tower_col100            # 99 SFX rows
    … --kind sfx --kind speech --kind chrome                    # every label
    … --only diff                                              # rows where the reads differ

Writes ``output/tests/ocr_contact_sheet/ab_<a>_vs_<b>[_<kinds>]_pNN.png`` +
``….pdf``. CPU only (PIL); needs a CJK font (``fc-match 'Noto Sans CJK JP'``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401  (PDF save needs the JPEG codec registered)

from anime_tools.ocr.sfx import CROP_PAD, crop_box

REPO = Path(__file__).resolve().parents[3]
LABELS = REPO / "project/cjk_aware_anima_dit/assets/sfx_labels_sincos.tsv"
PAGES = REPO / "post_image_dataset/resized/sincos"
EVAL = REPO / "output/ocr/eval"
OUT = REPO / "output/tests/ocr_contact_sheet"

GREEN = (222, 245, 222)
RED = (250, 222, 222)
INK = (20, 20, 20)
GREY = (120, 120, 120)
CROP_W, CROP_H = 260, 150
COL_TEXT = 330
ROWS_PER_PAGE = 12
MARGIN = 24
ROW_H = CROP_H + 28


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    fam = "Noto Sans CJK JP" + (":bold" if bold else "")
    path = subprocess.run(
        ["fc-match", "-f", "%{file}", fam], capture_output=True, text=True
    ).stdout.strip()
    return ImageFont.truetype(path, size)


def load_eval(name: str) -> dict[str, dict]:
    by = {}
    for line in (EVAL / f"sfx_{name}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by[str(r["row"])] = r
    return by


def load_boxes() -> dict[str, tuple[str, list[int]]]:
    import csv

    out = {}
    with LABELS.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[str(r["row"])] = (r["stem"], json.loads(r["box"]))
    return out


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    s = min(w / img.width, h / img.height, 2.5)
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


def verdict(a: dict, b: dict) -> tuple[str, float]:
    """``better`` / ``worse`` / ``same`` for B vs A, and the sim delta."""
    d = float(b["sim"]) - float(a["sim"])
    if b["exact"] and not a["exact"]:
        return "better", d
    if a["exact"] and not b["exact"]:
        return "worse", d
    if d >= 0.05:
        return "better", d
    if d <= -0.05:
        return "worse", d
    return "same", d


def draw_row(
    d: ImageDraw.ImageDraw,
    y: int,
    crop: Image.Image,
    row: dict,
    a: dict,
    b: dict,
    names,
    fonts,
    page_w,
):
    big, small, bold = fonts
    v, delta = verdict(a, b)
    if v != "same":
        d.rectangle(
            [MARGIN, y, page_w - MARGIN, y + ROW_H - 4],
            fill=GREEN if v == "better" else RED,
        )
    d.rectangle([MARGIN, y, MARGIN + CROP_W, y + CROP_H], outline=GREY)
    im = fit(crop, CROP_W - 4, CROP_H - 4)
    d._image.paste(
        im,
        (
            MARGIN + 2 + (CROP_W - 4 - im.width) // 2,
            y + 2 + (CROP_H - 4 - im.height) // 2,
        ),
    )
    d.text(
        (MARGIN, y + CROP_H + 1),
        f"#{row['row']}  {row['stem']}  {row['kind']} · {row['orient']}",
        font=small,
        fill=GREY,
    )

    x = MARGIN + CROP_W + 16
    d.text((x, y), "truth", font=small, fill=GREY)
    yy = y + 18
    for ln in wrap(row["text"], big, COL_TEXT - 8)[:4]:
        d.text((x, yy), ln, font=big, fill=INK)
        yy += 30

    for i, (name, r) in enumerate(((names[0], a), (names[1], b))):
        x = MARGIN + CROP_W + 16 + COL_TEXT * (i + 1)
        tick = "✓" if r["exact"] else ("♡✓" if r.get("exact_noheart") else "×")
        head = f"{name}   sim {float(r['sim']):.2f}  {tick}"
        if i == 1 and v != "same":
            head += f"  ({delta:+.2f})"
        d.text((x, y), head, font=small, fill=GREY)
        yy = y + 18
        pred = r["pred"] or "∅"
        if r.get("runaway"):
            pred = pred[:24] + "…(runaway)"
        for ln in wrap(pred, big, COL_TEXT - 8)[:4]:
            d.text((x, yy), ln, font=bold if r["exact"] else big, fill=INK)
            yy += 30


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--a", required=True, help="eval name (output/ocr/eval/sfx_<a>.jsonl)"
    )
    ap.add_argument("--b", required=True)
    ap.add_argument("--kind", action="append", help="default sfx")
    ap.add_argument("--only", choices=["diff", "all"], default="all")
    ap.add_argument("--name")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    kinds = args.kind or ["sfx"]

    A, B = load_eval(args.a), load_eval(args.b)
    boxes = load_boxes()
    rows = []
    for key, ra in A.items():
        rb = B.get(key)
        if rb is None or ra["kind"] not in kinds:
            continue
        if args.only == "diff" and ra["pred_norm"] == rb["pred_norm"]:
            continue
        v, delta = verdict(ra, rb)
        rows.append((0 if v != "same" else 1, -abs(delta), key, ra, rb))
    rows.sort(key=lambda t: (t[0], t[1], int(t[2])))
    if not rows:
        raise SystemExit("no rows")

    n_better = sum(1 for r in rows if verdict(r[3], r[4])[0] == "better")
    n_worse = sum(1 for r in rows if verdict(r[3], r[4])[0] == "worse")
    ex_a = sum(1 for r in rows if r[3]["exact"])
    ex_b = sum(1 for r in rows if r[4]["exact"])
    sim_a = float(np.mean([float(r[3]["sim"]) for r in rows]))
    sim_b = float(np.mean([float(r[4]["sim"]) for r in rows]))

    name = args.name or f"ab_{args.a}_vs_{args.b}" + (
        "" if kinds == ["sfx"] else "_" + "-".join(kinds)
    ) + ("_diff" if args.only == "diff" else "")
    args.out.mkdir(parents=True, exist_ok=True)
    fonts = (font(22), font(15), font(22, bold=True))
    page_w = MARGIN * 2 + CROP_W + 16 + COL_TEXT * 3
    page_h = MARGIN * 2 + 60 + ROWS_PER_PAGE * ROW_H
    pages: list[Image.Image] = []
    img_cache: dict[str, np.ndarray] = {}
    import cv2

    for p in range(0, len(rows), ROWS_PER_PAGE):
        page = Image.new("RGB", (page_w, page_h), "white")
        d = ImageDraw.Draw(page)
        head = f"A = {args.a}   B = {args.b}   ·  sincos hand labels, kind {'+'.join(kinds)}"
        stats = f"n {len(rows)}   exact {ex_a} → {ex_b}   sim {sim_a:.3f} → {sim_b:.3f}   B better {n_better} / worse {n_worse}"
        d.text((MARGIN, MARGIN), head, font=fonts[2], fill=INK)
        d.text(
            (page_w - MARGIN - fonts[1].getlength(stats), MARGIN + 6),
            stats,
            font=fonts[1],
            fill=INK,
        )
        d.text(
            (MARGIN, MARGIN + 30),
            f"green = B gained exact or sim +0.05 · red = B lost · disagreements first · crop = reader's {CROP_PAD:.0%} pad · page {p // ROWS_PER_PAGE + 1} / {(len(rows) - 1) // ROWS_PER_PAGE + 1}",
            font=fonts[1],
            fill=GREY,
        )
        y = MARGIN + 60
        for _, _, key, ra, rb in rows[p : p + ROWS_PER_PAGE]:
            stem, box = boxes[key]
            if stem not in img_cache:
                img_cache[stem] = cv2.imread(str(PAGES / f"{stem}.png"))
            crop = crop_box(img_cache[stem], box)
            pil = (
                Image.fromarray(crop[:, :, ::-1])
                if crop is not None
                else Image.new("RGB", (8, 8), "grey")
            )
            draw_row(d, y, pil, ra, ra, rb, (args.a, args.b), fonts, page_w)
            y += ROW_H
        pages.append(page)

    for i, page in enumerate(pages, 1):
        page.save(args.out / f"{name}_p{i:02d}.png")
    pages[0].save(
        args.out / f"{name}.pdf", save_all=True, append_images=pages[1:], resolution=96
    )
    print(f"{len(rows)} rows → {len(pages)} pages: {args.out / name}_pNN.png + .pdf")
    print(
        f"exact {ex_a} → {ex_b}, sim {sim_a:.3f} → {sim_b:.3f}, B better {n_better} / worse {n_worse}"
    )


if __name__ == "__main__":
    main()
