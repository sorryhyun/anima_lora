"""P1 data for the copy / fill block C (CPU, symlinks only).

- ``post_image_dataset/moss/c_fill_illu/``: an identity cache over colorize v5's colour
  illustration targets (``i2i_colorize_control_1k``, the ``manga__`` pages left out: they
  carry text). Per stem the target latent, the same file as ``{stem}.moss_ref`` (a
  ``ref_scale 1`` reference is the target's own encode, same key and grid) and the
  colorize TE file, unread at ``caption_dropout 1``.
- ``hole_layouts.json`` beside it: the OCR text boxes of the cache's manga pages
  (``text_boxes.json``), as fractions of each condition page, one list per page.

    .venv/bin/python project/lora_blocks/src/build_c_data.py
"""

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
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for lat in sorted(SRC.glob("*.moss_latent.safetensors")):
        stem = lat.name.split(".moss_latent")[0].rsplit("_", 1)[0]
        if stem.startswith("manga__"):
            continue
        te = SRC / f"{stem}.moss_te.safetensors"
        if not te.exists():
            continue
        link(lat, OUT / lat.name)
        link(lat, OUT / f"{stem}.moss_ref.safetensors")
        link(te, OUT / te.name)
        n += 1
    print(f"{n} identity pairs in {OUT}")

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
    (OUT / "hole_layouts.json").write_text(json.dumps(layouts))
    boxes = sum(len(ly) for ly in layouts)
    print(f"{len(layouts)} layouts, {boxes} boxes -> {OUT / 'hole_layouts.json'}")


if __name__ == "__main__":
    main()
