#!/usr/bin/env python3
"""Freeze the reader gates' images into ``project/ocr_reader/eval_data/`` (gitignored).

    ANIMA_MANGA109S_ROOT=… .venv/bin/python project/ocr_reader/gate/build_eval_data.py

The sincos gate used to cut its crops on the fly from
``post_image_dataset/resized/sincos/``, a training tree that lost 8 label pages
on 2026-10-10. This copies what each gate reads into one place:

* ``sincos/pages/<stem>.png`` — the 155 resized pages behind
  ``assets/sfx_labels_sincos_597.tsv`` (so ``eval_sfx.py --pad`` still works);
* ``sincos/crops/<row>.png`` — every label row cut exactly as ``eval_sfx.py``
  cuts it (``deskew_crop``, pad 0.12, lossless PNG), plus ``sincos/index.tsv``
  with the orientation ``deskew_crop`` returned;
* ``k3_gelnote/crops/<id>.webp`` and ``k3_gelnote/pages/`` — the K3 rows of
  ``assets/k3_gelnote.tsv``, byte copies of the gelnote files.

Labels stay in the tracked ``assets/`` TSVs; this tree holds images only.
Existing files are skipped, so a re-run only fills gaps.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
LINE = REPO / "project/ocr_reader"
OUT = LINE / "eval_data"
SINCOS_LABELS = LINE / "assets/sfx_labels_sincos_597.tsv"
K3_LABELS = LINE / "assets/k3_gelnote.tsv"
SINCOS_PAGES = REPO / "post_image_dataset/resized/sincos"
GELNOTE_DEFAULT = "/media/sorryhyun/new/dataset/gelnote_crops"
PAD = 0.12  # eval_sfx.py's --pad default

sys.path.insert(0, str(REPO / "project/finished/cjk_aware_anima_dit/ocr"))
import manga109 as m109  # noqa: E402


def sincos() -> None:
    df = pd.read_csv(SINCOS_LABELS, sep="\t", dtype=str, keep_default_na=False)
    pages, crops = OUT / "sincos/pages", OUT / "sincos/crops"
    pages.mkdir(parents=True, exist_ok=True)
    crops.mkdir(parents=True, exist_ok=True)
    for stem in sorted(df.stem.unique()):
        dst = pages / f"{stem}.png"
        if not dst.exists():
            shutil.copy2(SINCOS_PAGES / f"{stem}.png", dst)

    mt = m109.pilot_manga_text()
    rows = []
    for r in df.itertuples():
        img = cv2.imread(str(pages / f"{r.stem}.png"))
        x0, y0, x1, y1 = json.loads(r.box)
        crop, orient = mt.deskew_crop(img, [x0, y0, x1, y0, x1, y1, x0, y1], PAD, 8)
        dst = crops / f"{r.row}.png"
        if not dst.exists():
            cv2.imwrite(str(dst), crop)
        rows.append({"row": r.row, "stem": r.stem, "orient": orient, "pad": PAD})
    pd.DataFrame(rows).to_csv(OUT / "sincos/index.tsv", sep="\t", index=False)
    print(f"sincos: {df.stem.nunique()} pages, {len(rows)} crops → {OUT / 'sincos'}")


def k3_gelnote() -> None:
    root = Path(os.environ.get("ANIMA_GELNOTE_ROOT", GELNOTE_DEFAULT)).expanduser()
    df = pd.read_csv(K3_LABELS, sep="\t", dtype=str, keep_default_na=False)
    man = pd.read_json(root / "manifest.jsonl", lines=True)
    page_of = dict(zip(man.image_id.astype(str), man.page_path))
    crops, pages = OUT / "k3_gelnote/crops", OUT / "k3_gelnote/pages"
    crops.mkdir(parents=True, exist_ok=True)
    pages.mkdir(parents=True, exist_ok=True)
    for r in df.itertuples():
        dst = crops / Path(r.path).name
        if not dst.exists():
            shutil.copy2(root / r.path, dst)
        page = page_of[r.image_id]
        dst = pages / Path(page).name
        if not dst.exists():
            shutil.copy2(root / page, dst)
    print(
        f"k3_gelnote: {len(df)} crops, {df.image_id.nunique()} pages → {OUT / 'k3_gelnote'}"
    )


if __name__ == "__main__":
    sincos()
    k3_gelnote()
