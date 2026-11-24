#!/usr/bin/env python3
"""A main / sub reader under ``OCR2:`` on the 292 val crops that were ``sub_neg`` in
``mainsub_1121_1806`` (a main with no kanji, so no ruby): the share of empty answers.
The crop list is that run's re-read, ``output/ocr/eval/subneg292_mainsub_1121_1806.jsonl``.

    make daemon-run ARGS="--queue project/ocr_reader/probes/eval_subneg.py \\
        --ckpt output/ocr/<run>/best --name <run>"

Writes ``output/ocr/eval/subneg292_<name>.jsonl`` (path, pred, empty). ``--crops`` reads
another crop list (a jsonl with a ``path`` per line, relative to ``derived_root()``) and writes
``output/ocr/eval/<crops stem>_<name>.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
import eval_manga109 as ev  # noqa: E402
import manga109 as m109  # noqa: E402

CROPS = m109.REPO / "output/ocr/eval/subneg292_mainsub_1121_1806.jsonl"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crops", type=Path, default=CROPS)
    a = ap.parse_args()
    paths = [json.loads(line)["path"] for line in open(a.crops, encoding="utf-8")]
    d = m109.derived_root()
    rd = ev.Vl16Reader(a.ckpt, "cuda")
    ev.TASK = "OCR2:"  # Vl16Reader.read's prompt
    crops = [cv2.imread(str(d / p)) for p in paths]
    preds = rd.read(crops, [""] * len(crops), a.bs)
    stem = "subneg292" if a.crops == CROPS else a.crops.stem
    out = m109.REPO / "output/ocr/eval" / f"{stem}_{a.name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p, s in zip(paths, preds):
            row = dict(path=p, pred=s, empty=not s.strip())
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = sum(not s.strip() for s in preds)
    print(f"{a.name}: empty {n} / {len(preds)} ({100 * n / len(preds):.1f} %) -> {out}")


if __name__ == "__main__":
    main()
