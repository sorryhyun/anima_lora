"""P1 data for the copy / fill block C (CPU, symlinks only).

- ``post_image_dataset/moss/c_fill_illu/``: an identity cache over colorize v5's colour
  illustration targets (``i2i_colorize_control_1k``, the ``manga__`` pages left out: they
  carry text). Per stem the target latent, the same file as ``{stem}.moss_ref`` (a
  ``ref_scale 1`` reference is the target's own encode, same key and grid) and the
  colorize TE file, unread at ``caption_dropout 1``.
- ``--bw``: also a ``bw__{stem}`` identity pair per illustration on its B&W condition
  (the colorize ``*.moss_ref``, same grid as the target) as latent and reference, into
  ``--out`` (C v2: ``c_fill_illu_bw``). The ``manga__`` conditions stay out: their balloons
  would teach a hole to letter.
- ``hole_layouts.json`` beside it: the OCR text boxes of the cache's manga pages
  (``text_boxes.json``), as fractions of each condition page, one list per page.

    .venv/bin/python project/lora_blocks/src/build_c_data.py [--bw --out post_image_dataset/moss/c_fill_illu_bw]
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image
from safetensors import safe_open

from library.env import resolve_under_home

SRC = resolve_under_home("post_image_dataset/moss/i2i_colorize_control_1k")
OUT = resolve_under_home("post_image_dataset/moss/c_fill_illu")


def link(src: Path, dst: Path) -> None:
    if not dst.is_symlink():
        os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = resolve_under_home(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = n_bw = 0
    for lat in sorted(SRC.glob("*.moss_latent.safetensors")):
        stem, size = lat.name.split(".moss_latent")[0].rsplit("_", 1)
        if stem.startswith("manga__"):
            continue
        te = SRC / f"{stem}.moss_te.safetensors"
        if not te.exists():
            continue
        link(lat, out / lat.name)
        link(lat, out / f"{stem}.moss_ref.safetensors")
        link(te, out / te.name)
        n += 1
        bw = SRC / f"{stem}.moss_ref.safetensors"
        if args.bw and bw.exists():
            link(bw, out / f"bw__{stem}_{size}.moss_latent.safetensors")
            link(bw, out / f"bw__{stem}.moss_ref.safetensors")
            link(te, out / f"bw__{stem}.moss_te.safetensors")
            n_bw += 1
    print(f"{n} colour + {n_bw} B&W identity pairs in {out}")

    blocks = json.loads((SRC / "text_boxes.json").read_text())
    layouts = []
    for stem, lines in blocks.items():
        ref = SRC / f"{stem}.moss_ref.safetensors"
        if not lines or not ref.exists():
            continue
        with safe_open(str(ref), "pt") as f:
            cond = f.metadata()["condition"]
        w, h = Image.open(cond).size
        layouts.append([[b[0] / w, b[1] / h, b[2] / w, b[3] / h] for _, b in lines])
    (out / "hole_layouts.json").write_text(json.dumps(layouts))
    boxes = sum(len(ly) for ly in layouts)
    print(f"{len(layouts)} layouts, {boxes} boxes -> {out / 'hole_layouts.json'}")


if __name__ == "__main__":
    main()
