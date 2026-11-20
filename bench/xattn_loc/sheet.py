#!/usr/bin/env python
"""xattn_loc sheets — a probe run's kept maps over their pages (CPU).

One row per quoted string of the kept set: the page with the string's OCR box,
then the string's cross-attn map (the mean over ``--blocks``) at each σ, drawn
over the dimmed page with the same box. Each panel is scaled to its own max.

    .venv/bin/python bench/xattn_loc/sheet.py bench/xattn_loc/results/<run> --blocks 13,14,15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from library.env import resolve_under_home

PANEL_H = 420
ROWS_PER_SHEET = 6


def font(px: int):
    for p in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, px)
    return ImageFont.load_default()


def heat_panel(page: Image.Image, mp: np.ndarray, box, size) -> Image.Image:
    g = np.asarray(page.convert("L").resize(size), np.float32) / 255.0
    h = (
        np.asarray(
            Image.fromarray(
                (mp / max(float(mp.max()), 1e-12) * 255).astype(np.uint8)
            ).resize(size, Image.NEAREST),
            np.float32,
        )
        / 255.0
    )
    rgb = np.stack([0.35 * g + 0.65 * h, 0.35 * g * (1 - h), 0.35 * g * (1 - h)], -1)
    im = Image.fromarray((rgb.clip(0, 1) * 255).astype(np.uint8))
    sx, sy = size[0] / page.width, size[1] / page.height
    ImageDraw.Draw(im).rectangle(
        (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy),
        outline=(0, 255, 255),
        width=2,
    )
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--blocks", default="13,14,15")
    a = p.parse_args()
    run = Path(a.run)
    blocks = [int(b) for b in a.blocks.split(",")]
    maps = np.load(run / "caststation_maps.npz")
    items = json.loads((run / "items.json").read_text("utf-8"))
    sigmas = sorted({k.split("|")[2] for k in maps.files}, key=float, reverse=True)
    f, rows = font(20), []
    for m in items:
        page = Image.open(
            resolve_under_home("post_image_dataset/resized")
            / m["set"]
            / f"{m['stem']}.png"
        ).convert("RGB")
        size = (round(PANEL_H * page.width / page.height), PANEL_H)
        for i, s in enumerate(m["strings"]):
            if f"{m['stem']}|{i}|{sigmas[0]}" not in maps.files:
                continue
            first = page.resize(size)
            sx, sy = size[0] / page.width, size[1] / page.height
            b = s["box"]
            ImageDraw.Draw(first).rectangle(
                (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy),
                outline=(0, 255, 255),
                width=2,
            )
            panels = [first] + [
                heat_panel(
                    page,
                    maps[f"{m['stem']}|{i}|{sg}"].astype(np.float32)[blocks].mean(0),
                    b,
                    size,
                )
                for sg in sigmas
            ]
            rows.append((f"{m['stem']}  “{s['text']}”", panels))
    out = []
    for n in range(0, len(rows), ROWS_PER_SHEET):
        chunk = rows[n : n + ROWS_PER_SHEET]
        W = max(sum(im.width for im in ps) + 8 * len(ps) for _, ps in chunk)
        sheet = Image.new("RGB", (W, (PANEL_H + 34) * len(chunk) + 30), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        d.text((6, 4), "page | " + " | ".join(f"σ {s}" for s in sigmas)
               + f"   (blocks {a.blocks}, box = OCR line)", fill=(220, 220, 220), font=f)  # fmt: skip
        for r, (title, ps) in enumerate(chunk):
            y = 30 + r * (PANEL_H + 34)
            d.text((6, y + 4), title, fill=(255, 255, 255), font=f)
            x = 0
            for im in ps:
                sheet.paste(im, (x, y + 32))
                x += im.width + 8
        fn = run / f"sheet_b{a.blocks.replace(',', '_')}_{n // ROWS_PER_SHEET:02d}.png"
        sheet.save(fn)
        out.append(fn)
    print("\n".join(str(x) for x in out))


if __name__ == "__main__":
    main()
