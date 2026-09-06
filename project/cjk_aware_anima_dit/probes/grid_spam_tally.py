"""OCR spam tally over the 8-row unmask grids: images with any lenient PP-OCRv6 box.

    .venv/bin/python probes/grid_spam_tally.py <out.json> [grid_dir ...]

Grid dirs are names under output/tests/cjk_unmask_eval2/ (default: the C10 /
C9ISOQ set of plan_base1 B3).
"""

import json
import sys
from pathlib import Path

from PIL import Image

from anime_tools.ocr._onnx import load_ocr

EVAL = Path("output/tests/cjk_unmask_eval2")
DIRS = sys.argv[2:] or [
    f"armC10{t}_s{s}" for t in ("", "s7", "s1234") for s in (42, 7, 1234)
] + [f"armC9ISOQ_s{s}" for s in (42, 7, 1234)]
engine = load_ocr(device="cpu", min_score=0.3, min_chars=1, skip_en=False)
out = {}
for d in DIRS:
    pngs = sorted((EVAL / d).glob("*.png"))
    cells = []
    for i, png in enumerate(pngs, 1):
        w, h = Image.open(png).size
        lines = engine.read(png)
        area = sum(
            max(0, x1 - x0) * max(0, y1 - y0)
            for (x0, y0, x1, y1) in (ln.box for ln in lines)
        )
        cells.append(
            {
                "row": f"r{i}",
                "n_lines": len(lines),
                "glyph_frac": area / (w * h),
                "chars": sum(len(ln.text) for ln in lines),
                "texts": [ln.text for ln in lines],
            }
        )
    out[d] = cells
    print(
        d,
        "imgs_with_text",
        sum(1 for c in cells if c["n_lines"]),
        "lines",
        sum(c["n_lines"] for c in cells),
        "glyph%",
        round(100 * sum(c["glyph_frac"] for c in cells) / len(cells), 2),
        flush=True,
    )
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
