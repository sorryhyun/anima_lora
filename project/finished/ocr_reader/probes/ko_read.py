#!/usr/bin/env python3
"""A reader over the review page's label rows, as trained (the manifest crops), with its
confidence: the KO ``OCR:`` crops of ``manifest_sfxp_kozh.parquet`` (gelnote_ko,
swaprelax_v5) and the hito v4 self-labels (``manifest_pseudo_hito_v4.parquet``, as
``hito_v4_ja`` / ``hito_v4_ko``). Input for ``ko_label_fix.py``, whose flags file the
review page (``ko_label_review.py --flags``) sorts on ("reader disagrees").

The reader is the shipped v6 by default; ``--adapter_dir output/ocr/<run>/best`` reads
with a local run (``--version v7_preview`` for a two-prompt run, which reads ``OCR:``).
A crop the reader runs away on, or (two-prompt) answers ``""`` for, is stored as
``read = ""``, ``conf = NaN``.

    ANIMA_MANGA109S_ROOT=... make daemon-run ARGS="--stall-timeout 600 \\
        project/finished/ocr_reader/probes/ko_read.py --out <reads.parquet> \\
        [--adapter_dir output/ocr/<run>/best --version v7_preview]"
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

KOZH = ("gelnote_ko", "swaprelax_v5")
HITO = ("hito_v4_ja", "hito_v4_ko")


def rows(d: Path, sources: list[str]) -> pd.DataFrame:
    kz = pd.read_parquet(d / "manifest_sfxp_kozh.parquet")
    parts = [kz[kz.source.isin(sources)]]
    if any(s in HITO for s in sources):
        h = pd.read_parquet(d / "manifest_pseudo_hito_v4.parquet")
        h = h.assign(source="hito_v4_" + h.id.str.rsplit("_", n=1).str[1])
        parts.append(h[h.source.isin(sources)])
    cols = ["source", "id", "text", "w", "h", "path"]
    return pd.concat([p[cols] for p in parts], ignore_index=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", default=[*KOZH, *HITO])
    ap.add_argument(
        "--adapter_dir",
        type=Path,
        help="a run's checkpoint dir (adapter + lm.safetensors); default: the catalog row",
    )
    ap.add_argument(
        "--version", default="v6", help="OcrReader version (v6 / v7_preview)"
    )
    ap.add_argument(
        "--name",
        help="reader name shown on the review page (default: the run dir's name, else --version)",
    )
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    name = a.name or (a.adapter_dir.parent.name if a.adapter_dir else a.version)
    d = m109.derived_root()
    df = rows(d, a.sources)
    print(f"[read] {len(df)} crops {df.source.value_counts().to_dict()}", flush=True)
    crops = [cv2.imread(str(d / p)) for p in df.path]
    from anime_tools.ocr.reader import OcrReader

    reader = OcrReader.load(
        device=a.device, batch_size=a.bs, adapter_dir=a.adapter_dir, version=a.version
    )
    t0 = time.time()
    res = reader.read_scored(crops)
    print(f"[read] {name}: {len(df) / (time.time() - t0):.1f} crops/s", flush=True)
    df["read"] = [r[0] if r else "" for r in res]
    df["conf"] = [r[1] if r else float("nan") for r in res]
    df["reader"] = name
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"[read] → {a.out}", flush=True)


if __name__ == "__main__":
    main()
