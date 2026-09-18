#!/usr/bin/env python
"""vae_glyph_floor — does a JA glyph survive the VAE round trip at N px?

The sentence floor (20 px, plan_synth3) was a bubble-capacity number, never a
legibility one; the corpus draws dialogue at 15–20 px per glyph
(`channel_(caststation)` OCR boxes, user 2026-09-18). This renders test
strings at 12 / 16 / 20 / 24 / 28 px (tategaki and yokogaki, two fonts) on a
512² canvas, plus the real corpus crops at native scale, runs each through
``vae.encode_pixels_to_latents`` → ``decode_to_pixels`` and reads the text
region before and after with both readers. No training, no DiT: this is the
floor on the *target*, below which nothing can be learned.

    make daemon-run ARGS="project/cjk_renderable_anima/src/probe/vae_glyph_floor.py"

Writes ``output/wake_probe/vae_glyph_floor/{sheet.png,reads.json}`` and prints
a table (size × font × orientation: sfx / vl read, before → after, CER).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import OUT  # noqa: E402

SIZES = (8, 10, 12, 16)
STRINGS = ("くっそ鬱雑えわ", "どうしたんだよー")
# real corpus text at 15–20 px per glyph (resized tier, OCR sidecar boxes)
CORPUS = (
    ("channel_(caststation)/12248604.png", (466, 220, 489, 492), "発火しちゃうとかくっそだせえなw"),
    ("channel_(caststation)/12248603.png", (525, 86, 616, 110), "ふぁああい…"),
    ("channel_(caststation)/12248603.png", (447, 618, 470, 738), "くっそ鬱雑えわ"),
)


def draw(text, fs, font_path, vertical, canvas=512):
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", (canvas, canvas), "white")
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(font_path, fs, index=0)
    x0, y0 = canvas // 2 - fs, canvas // 4
    if vertical:
        for i, ch in enumerate(text):
            d.text((x0, y0 + i * int(fs * 1.1)), ch, fill="black", font=font)
        box = (x0 - 4, y0 - 4, x0 + fs + 6, y0 + len(text) * int(fs * 1.1) + 6)
    else:
        d.text((x0, y0), text, fill="black", font=font)
        w = d.textlength(text, font=font)
        box = (x0 - 4, y0 - 4, int(x0 + w) + 6, y0 + int(fs * 1.3) + 6)
    return im, box


def roundtrip(vae, im, device):
    import numpy as np
    import torch

    from library.inference.output import pixels_to_pil

    px = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].float().div(127.5).sub(1.0)
    with torch.no_grad():
        lat = vae.encode_pixels_to_latents(px.to(device, dtype=vae.dtype))
        out = vae.decode_to_pixels(lat)
    if out.ndim == 5:
        out = out.squeeze(2)
    return pixels_to_pil(out[0].float().cpu())


def main():
    import numpy as np
    from PIL import Image, ImageDraw

    from common.models import load_vae
    from common.readers import Readers
    from common.render.flat import find_fonts
    from common.text import cer

    device = "cuda"
    out = OUT / "vae_glyph_floor"
    out.mkdir(parents=True, exist_ok=True)
    fonts = find_fonts()
    picks = [fonts[0], fonts[-1]] if len(fonts) > 1 else fonts[:1]
    vae = load_vae(device)
    rd = Readers(device)

    cases = []
    for fs in SIZES:
        for fp in picks:
            for vertical in (True, False):
                text = STRINGS[0] if vertical else STRINGS[1]
                im, box = draw(text, fs, fp, vertical)
                cases.append(
                    {"kind": "synth", "size": fs, "font": Path(fp).stem, "vertical": vertical, "text": text, "im": im, "box": box}
                )
    for rel, (x0, y0, x1, y1), text in CORPUS:
        im = Image.open(REPO / "post_image_dataset" / "resized" / rel).convert("RGB")
        # a 256² window around the box, aligned to the 16-px VAE grid
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        wx0 = max(0, min(im.width - 256, cx - 128)) // 16 * 16
        wy0 = max(0, min(im.height - 256, cy - 128)) // 16 * 16
        win = im.crop((wx0, wy0, wx0 + 256, wy0 + 256))
        n = len(text)
        cases.append(
            {
                "kind": "corpus",
                "size": round(max(x1 - x0, y1 - y0) / n, 1),
                "font": rel.split("/")[-1].split(".")[0],
                "vertical": (y1 - y0) > (x1 - x0),
                "text": text,
                "im": win,
                "box": (x0 - wx0 - 4, y0 - wy0 - 4, x1 - wx0 + 4, y1 - wy0 + 4),
            }
        )

    rows = []
    for c in cases:
        c["rt"] = roundtrip(vae, c["im"], device)
        reads = {}
        for tag, im in (("before", c["im"]), ("after", c["rt"])):
            crop = np.ascontiguousarray(np.array(im.crop(c["box"]))[:, :, ::-1])
            s = rd.sfx.read_scored([crop])[0]
            v = rd.vl.read([crop])[0]
            reads[tag] = {
                "sfx": None if s is None else s[0],
                "sfx_conf": None if s is None else float(s[1]),
                "vl": v[0],
            }
        c["reads"] = reads
        rows.append({k: v for k, v in c.items() if k not in ("im", "rt")})

    # sheet: before | after crops at 3×, one row per case
    tiles = []
    for c in cases:
        b = c["im"].crop(c["box"])
        a = c["rt"].crop(c["box"])
        z = 3
        b, a = b.resize((b.width * z, b.height * z), Image.NEAREST), a.resize((a.width * z, a.height * z), Image.NEAREST)
        w = b.width + a.width + 30
        h = max(b.height, a.height) + 60
        t = Image.new("RGB", (max(w, 720), h), "white")
        t.paste(b, (0, 50))
        t.paste(a, (b.width + 30, 50))
        d = ImageDraw.Draw(t)
        r = c["reads"]
        d.text(
            (0, 2),
            f"{c['kind']} {c['size']}px {c['font']} {'tate' if c['vertical'] else 'yoko'} | {c['text']}",
            fill="black",
        )
        d.text(
            (0, 22),
            f"before sfx {r['before']['sfx']} / vl {r['before']['vl']}   →   after sfx {r['after']['sfx']} / vl {r['after']['vl']}",
            fill=(0, 90, 0),
        )
        tiles.append(t)
    W = max(t.width for t in tiles)
    sheet = Image.new("RGB", (W, sum(t.height for t in tiles)), "white")
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height
    sheet.save(out / "sheet.png")
    (out / "reads.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    print("\n| kind | px | font | dir | sfx before → after | vl before → after | CER sfx after |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        b, a = r["reads"]["before"], r["reads"]["after"]
        print(
            f"| {r['kind']} | {r['size']} | {r['font']} | {'tate' if r['vertical'] else 'yoko'} | "
            f"{b['sfx']} → {a['sfx']} | {b['vl']} → {a['vl']} | {cer(a['sfx'] or '', r['text']):.2f} |"
        )
    print(f"\nsheet: {out / 'sheet.png'}")


if __name__ == "__main__":
    main()
