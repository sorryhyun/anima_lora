#!/usr/bin/env python3
"""v6 (the shipped reader) over the KO ``OCR:`` crops of ``manifest_sfxp_kozh.parquet``
(gelnote_ko, swaprelax_v5) as trained (4 px margin), with its confidence. Input for
``ko_label_fix.py`` (line breaks from the read, text outside the crop).

    ANIMA_MANGA109S_ROOT=... make daemon-run ARGS="--stall-timeout 600 \\
        project/finished/ocr_reader/probes/ko_v6_read.py --out <reads.parquet>"
"""

from __future__ import annotations

import argparse
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
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", default=["gelnote_ko", "swaprelax_v5"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    d = m109.derived_root()
    kz = pd.read_parquet(d / "manifest_sfxp_kozh.parquet")
    kz = kz[kz.source.isin(a.sources)].reset_index(drop=True)
    print(f"[read] {len(kz)} crops {kz.source.value_counts().to_dict()}", flush=True)
    crops = [cv2.imread(str(d / p)) for p in kz.path]
    from anime_tools.ocr.reader import OcrReader

    reader = OcrReader.load(device=a.device, batch_size=a.bs)
    t0 = time.time()
    res = reader.read_scored(crops)
    print(f"[read] {len(kz) / (time.time() - t0):.1f} crops/s", flush=True)
    kz["read"] = [r[0] if r else "" for r in res]
    kz["conf"] = [r[1] if r else float("nan") for r in res]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    kz[["source", "id", "text", "read", "conf", "w", "h", "path"]].to_parquet(
        a.out, index=False
    )
    print(f"[read] → {a.out}", flush=True)


if __name__ == "__main__":
    main()
