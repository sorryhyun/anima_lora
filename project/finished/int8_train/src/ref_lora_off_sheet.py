#!/usr/bin/env python3
"""Sheets for ``ref_lora_off_render.py``: per page, each text bbox cropped from the
source, the v3 render (control), the ``blocks`` arm and the ``all`` arm, side by side
at 2x. CPU only. Writes ``<results>/v3_en400_ko1709_e2_reflora_compare/<page>.png``."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RESULTS = Path("output/moss_i2i_eval/swap_text/results")
INPUTS = Path("output/moss_i2i_eval/swap_text/inputs")
ARMS = {
    "src": None,
    "v3 (ref LoRA on)": RESULTS / "v3_en400_ko1709_e2_ema_bf16_fp32s",
    "ref blocks off": RESULTS / "v3_en400_ko1709_e2_ema_bf16_fp32s_reflora_blocks",
    "ref all off": RESULTS / "v3_en400_ko1709_e2_ema_bf16_fp32s_reflora_all",
    "ref k/v only": RESULTS / "v3_en400_ko1709_e2_ema_bf16_fp32s_reflora_kv",
    "ref k/v off": RESULTS / "v3_en400_ko1709_e2_ema_bf16_fp32s_reflora_nokv",
}
PAGES = ("1483610_0007_en", "3124789_0042_en", "589883_0004_ko", "737486_0019_ko")
ZOOM, PAD = 2, 24


def render_of(folder: Path, page: str) -> Image.Image | None:
    pngs = sorted((folder / page).glob("*.png")) if folder else []
    return Image.open(pngs[0]).convert("RGB") if pngs else None


def main() -> None:
    out = RESULTS / "v3_en400_ko1709_e2_reflora_compare"
    out.mkdir(exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 18)
    for page in PAGES:
        layout = json.loads((INPUTS / f"{page}.txt").read_text())
        src = Image.open(INPUTS / f"{page}.src.webp").convert("RGB")
        w, h = src.size
        images = {
            name: (src if folder is None else render_of(folder, page))
            for name, folder in ARMS.items()
        }
        images = {k: v for k, v in images.items() if v is not None}
        rows = []
        for item in layout["items"]:
            y0, x0, y1, x1 = item["bbox"]
            box = (
                max(0, x0 + w // 2 - PAD),
                max(0, y0 + h // 2 - PAD),
                min(w, x1 + w // 2 + PAD),
                min(h, y1 + h // 2 + PAD),
            )
            crops = [
                im.resize(src.size)
                .crop(box)
                .resize(
                    ((box[2] - box[0]) * ZOOM, (box[3] - box[1]) * ZOOM), Image.LANCZOS
                )
                for im in images.values()
            ]
            rows.append((item["text"], crops))
        cell_w = max(c.width for _, cs in rows for c in cs) + 8
        heights = [max(c.height for c in cs) + 8 for _, cs in rows]
        head = 56
        sheet = Image.new(
            "RGB", (cell_w * len(images), head + sum(heights) + 30 * len(rows)), "white"
        )
        d = ImageDraw.Draw(sheet)
        for i, name in enumerate(images):
            d.text((i * cell_w + 4, 4), name, fill="black", font=font)
        d.text((4, 28), page, fill="gray", font=font)
        y = head
        for (text, crops), hh in zip(rows, heights):
            d.text((4, y), text, fill="black", font=font)
            y += 30
            for i, c in enumerate(crops):
                sheet.paste(c, (i * cell_w + 4, y))
            y += hh
        sheet.save(out / f"{page}.png")
        print(out / f"{page}.png", sheet.size)


if __name__ == "__main__":
    main()
