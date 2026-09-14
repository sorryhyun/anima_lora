"""Reading renders back: detector + SFX reader + stock VL16, and contact sheets."""

from __future__ import annotations

import math
from glob import glob
from pathlib import Path

from .common import cer, norm


class Readers:
    def __init__(self, device: str):
        from anime_tools.ocr.animetext import AnimeTextDetector
        from anime_tools.ocr.sfx import SfxReader

        self.det = AnimeTextDetector.load(device=device)
        self.sfx = SfxReader.load(device=device, batch_size=16)
        import pseudo_label as pl

        self.vl = pl.SWEEPERS["stock"](None, device)

    def read_image(self, bgr, *, whole: bool = True):
        """Return list of dict(box, sfx, sfx_conf, vl) — one per detector box,
        plus the whole image as a pseudo-box when ``whole``."""
        import numpy as np

        H, W = bgr.shape[:2]
        boxes = [tuple(int(v) for v in b) for b in self.det.detect(bgr)]
        crops = []
        for b in boxes:
            x0, y0, x1, y1 = b
            pw, ph = int(0.12 * (x1 - x0)), int(0.12 * (y1 - y0))
            c = bgr[
                max(0, y0 - ph) : min(H, y1 + ph), max(0, x0 - pw) : min(W, x1 + pw)
            ]
            crops.append(np.ascontiguousarray(c))
        if whole:
            boxes.append((0, 0, W, H))
            crops.append(bgr)
        if not crops:
            return []
        sfx = self.sfx.read_scored(crops)
        vl = self.vl.read(crops)
        out = []
        for b, s, v in zip(boxes, sfx, vl):
            out.append(
                {
                    "box": list(b),
                    "whole": b == (0, 0, W, H),
                    "sfx": None if s is None else s[0],
                    "sfx_conf": None if s is None else float(s[1]),
                    "vl": v[0],
                }
            )
        return out


def read_scored(rd: Readers, m: dict) -> list:
    """Read ``m['file']`` (whole image included); attach ``reads`` and each
    reader's best CER against ``m['text']`` to ``m``."""
    reads = rd.read_image(load_bgr(Path(m["file"])), whole=True)
    m["reads"] = reads
    m["cer_sfx"] = min([cer(r["sfx"] or "", m["text"]) for r in reads] or [1.0])
    m["cer_vl"] = min([cer(r["vl"] or "", m["text"]) for r in reads] or [1.0])
    return reads


def hit(reads: list, text: str, reader: str) -> bool:
    """Any box where ``reader`` (``'sfx'`` / ``'vl'``) reads exactly ``text``."""
    return any(norm(r[reader] or "") == norm(text) for r in reads)


def load_bgr(path: Path):
    import numpy as np
    from PIL import Image

    return np.ascontiguousarray(np.array(Image.open(path).convert("RGB"))[:, :, ::-1])


def _label_font(size=22):
    from PIL import ImageFont

    for p in glob("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc") + glob(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
    ):
        return ImageFont.truetype(p, size, index=0)
    return ImageFont.load_default()


def contact_sheet(rows, path: Path, thumb=256, cols=4):
    """rows: list of (PIL image, [lines of text]). Grid contact sheet."""
    from PIL import Image, ImageDraw

    font = _label_font(18)
    cell_h = thumb + 22 * 4
    n = len(rows)
    r = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * (thumb + 8), r * cell_h + 8), "white")
    d = ImageDraw.Draw(sheet)
    for i, (im, lines) in enumerate(rows):
        x = (i % cols) * (thumb + 8) + 4
        y = (i // cols) * cell_h + 4
        t = im.copy()
        t.thumbnail((thumb, thumb))
        sheet.paste(t, (x, y))
        for j, ln in enumerate(lines[:4]):
            d.text((x, y + thumb + 2 + 22 * j), ln[:26], fill="black", font=font)
    sheet.save(path)
