#!/usr/bin/env python3
"""v6 (the shipped ``OCR:`` reader) over the written detector crops of ``build_sfx_prefix.py
--det``: every ``pos`` / ``pos_mixed`` box of a split and the ``neg_speech`` boxes
``sfxp_crop_sheet.py`` typed ``neg_graze``. The read shows text in a crop that its target
leaves out (unannotated in COO and in ``<text>``).

    ANIMA_MANGA109S_ROOT=... make daemon-run ARGS="--stall-timeout 600 \\
        project/finished/ocr_reader/probes/sfxp_det_read.py --types <types_train.parquet> --out <reads.parquet>"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
import manga109 as m109  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--types", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    d = m109.derived_root()
    det = pd.read_parquet(d / "manifest_sfxp_det.parquet")
    det = det[det.split == a.split].reset_index(drop=True)
    ty = pd.read_parquet(a.types)
    ng = {json.loads(s)[0] for s in ty[ty.type == "neg_graze"].det_rows}
    keep = det.cat.isin(["pos", "pos_mixed"]) | det.index.isin(ng)
    sel = det[keep & det.path.notna()].copy()
    sel["j"] = sel.index
    print(f"[read] {len(sel)} crops ({sel.cat.value_counts().to_dict()})", flush=True)
    crops = [cv2.imread(str(d / p)) for p in sel.path]
    from anime_tools.ocr.reader import OcrReader

    reader = OcrReader.load(device=a.device, batch_size=a.bs)  # the catalog's v6
    t0 = time.time()
    preds = [r[0] if r else "" for r in reader.read_scored(crops)]
    print(f"[read] {len(sel) / (time.time() - t0):.1f} crops/s", flush=True)
    sel["read"] = preds
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sel[
        ["j", "book", "page", "id", "cat", "sfx_text", "speech_text", "path", "read"]
    ].to_parquet(a.out, index=False)
    print(f"[read] → {a.out}", flush=True)


if __name__ == "__main__":
    main()
