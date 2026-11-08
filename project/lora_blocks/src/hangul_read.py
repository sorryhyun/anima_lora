"""Plan step 3 reads on swap_text's Korean eval pages: per arm and seed, the stroke number
``grad_in`` (mean Sobel magnitude of luma inside the caption's text boxes, padded 4 px,
over the same on the Korean edition page; 1.0 = edges as strong as the edition's), and zoom
sheets per page and seed (rows = the edition, then each arm; columns = the text boxes at 2x).

    .venv/bin/python project/lora_blocks/src/hangul_read.py OUT_DIR [--ko10] arm1 arm2 ...

An arm is a results folder name under output/moss_i2i_eval/swap_text/results without the
seed suffix (seed 0 = no suffix, seed n = ``_s<n>``). ``--ko10`` reads the ten held-out
pages of ``inputs_ko10/`` (seed 0, folders ``ko10_<arm>``) instead of the two eval pages.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from library.env import resolve_under_home

IN = resolve_under_home("output/moss_i2i_eval/swap_text/inputs")
RES = resolve_under_home("output/moss_i2i_eval/swap_text/results")
PAGES = ("589883_0004_ko", "737486_0019_ko")
SEEDS = (0, 1, 2)
PRE = ""
PAD = 4


def boxes(page: str, size):
    w, h = size
    cap = json.loads((IN / f"{page}.txt").read_text())
    out = []
    for it in cap["items"]:
        y0, x0, y1, x1 = it["bbox"]
        out.append(
            (
                max(0, int(x0 + w / 2) - PAD),
                max(0, int(y0 + h / 2) - PAD),
                min(w, int(x1 + w / 2) + PAD),
                min(h, int(y1 + h / 2) + PAD),
            )
        )
    return out


def sobel(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("L"), dtype=np.float32)
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    return np.hypot(gx, gy)


def grad_in(img, bxs) -> float:
    g = sobel(img)
    return float(np.mean(np.concatenate([g[b:d, a:c].ravel() for a, b, c, d in bxs])))


def render(arm: str, seed: int, page: str) -> Path | None:
    sfx = "" if seed == 0 else f"_s{seed}"
    hits = sorted((RES / f"{PRE}{arm}{sfx}" / page).glob("*.png"))
    return hits[0] if hits else None


def main():
    global IN, PAGES, SEEDS, PRE
    out = Path(sys.argv[1])
    arms = sys.argv[2:]
    if arms and arms[0] == "--ko10":
        arms = arms[1:]
        IN = resolve_under_home("output/moss_i2i_eval/swap_text/inputs_ko10")
        PAGES = tuple(
            sorted(p.name[: -len(".src.webp")] for p in IN.glob("*.src.webp"))
        )
        SEEDS, PRE = (0,), "ko10_"
    out.mkdir(parents=True, exist_ok=True)
    table = {}
    for page in PAGES:
        src = Image.open(IN / f"{page}.src.webp")
        tgt = Image.open(IN / f"{page}.tgt.webp").convert("RGB")
        for seed in SEEDS:
            rows = []
            for arm in arms:
                p = render(arm, seed, page)
                if p is None:
                    continue
                img = Image.open(p).convert("RGB")
                ref = tgt.resize(img.size, Image.LANCZOS)
                bxs = boxes(page, img.size)
                r = grad_in(img, bxs) / grad_in(ref, bxs)
                table.setdefault(arm, {})[f"{page}_s{seed}"] = round(r, 3)
                rows.append((arm, img, bxs))
            if not rows:
                continue
            ref = tgt.resize(rows[0][1].size, Image.LANCZOS)
            bxs = rows[0][2]
            allrows = [("edition", ref, bxs)] + rows
            crops = [
                [
                    im.crop(b).resize(
                        ((b[2] - b[0]) * 2, (b[3] - b[1]) * 2), Image.LANCZOS
                    )
                    for b in bb
                ]
                for _, im, bb in allrows
            ]
            colw = [max(c[i].width for c in crops) for i in range(len(bxs))]
            rowh = [max(c.height for c in cs) for cs in crops]
            label = 150
            sheet = Image.new(
                "RGB",
                (label + sum(colw) + 8 * len(colw), sum(rowh) + 8 * len(rowh)),
                "white",
            )
            d = ImageDraw.Draw(sheet)
            y = 0
            for (name, _, _), cs, rh in zip(allrows, crops, rowh):
                d.text((4, y + 4), name[-24:], fill=(200, 0, 0))
                x = label
                for c, cw in zip(cs, colw):
                    sheet.paste(c, (x, y))
                    x += cw + 8
                y += rh + 8
            sheet.save(out / f"zoom_{page}_s{seed}.png")
        del src
    (out / "grad_in.json").write_text(json.dumps(table, indent=1))
    for arm, v in table.items():
        print(arm, v)


if __name__ == "__main__":
    main()
