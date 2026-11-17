#!/usr/bin/env python3
"""swaprelax_v5 rows whose box holds smaller AnimeText boxes (``det.json`` of the
swap_text relax tree keeps every detector box; the swap_text read kept only the outer one).
Reads each inner box with v6 (4 px margin, as trained) so the inner read can be compared
with the row's label: the inner box holds the whole text (a loose outer box) or only part
of it (a line of the block, or one of several blocks the outer box merged).

    make daemon-run ARGS="--stall-timeout 600 \\
        project/finished/ocr_reader/probes/nested_inner_read.py --out <inner.parquet>"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
RELAX = REPO / "post_image_dataset/i2i/swap_text_relax"
LABELS = Path("/media/sorryhyun/new/dataset/hito_ocr/swap_relax_v5/labels.jsonl")
MARGIN = 4


def area(b) -> int:
    return (b[2] - b[0]) * (b[3] - b[1])


def inside(i, o, tol=3) -> bool:
    return (
        i[0] >= o[0] - tol
        and i[1] >= o[1] - tol
        and i[2] <= o[2] + tol
        and i[3] <= o[3] + tol
    )


def nested_rows() -> list[dict]:
    det = json.loads((RELAX / "det.json").read_text())
    out = []
    for line in open(LABELS, encoding="utf-8"):
        r = json.loads(line)
        work, page, lang = r["stem"].split("_")
        tgt = det.get(f"{lang}/{work}/{page}", {}).get("tgt")
        if not tgt:
            continue
        B = r["box"]
        inner = [b for b in tgt if inside(b, B) and area(b) < 0.9 * area(B)]
        if inner and (REPO / r["page"]).exists():
            out.append(dict(r, inner=inner))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rows = nested_rows()
    jobs, crops, img, cur = [], [], None, None
    for r in sorted(rows, key=lambda r: r["page"]):
        if cur != r["page"]:
            img, cur = cv2.imread(str(REPO / r["page"])), r["page"]
        H, W = img.shape[:2]
        for j, b in enumerate(r["inner"]):
            x0, y0 = max(b[0] - MARGIN, 0), max(b[1] - MARGIN, 0)
            x1, y1 = min(b[2] + MARGIN, W), min(b[3] + MARGIN, H)
            crops.append(img[y0:y1, x0:x1])
            jobs.append(
                dict(
                    id=r["id"],
                    stem=r["stem"],
                    line=r["line"],
                    status=r["status"],
                    box=r["box"],
                    label=r["label"],
                    n_inner=len(r["inner"]),
                    j=j,
                    inner=b,
                    frac=area(b) / area(r["box"]),
                )
            )
    print(f"[inner] {len(rows)} rows, {len(crops)} inner boxes", flush=True)
    from anime_tools.ocr.reader import OcrReader

    reader = OcrReader.load(device=a.device, batch_size=a.bs)
    t0 = time.time()
    res = reader.read_scored(crops)
    print(f"[inner] {len(crops) / (time.time() - t0):.1f} crops/s", flush=True)
    df = pd.DataFrame(jobs)
    df["read"] = [r[0] if r else "" for r in res]
    df["conf"] = [r[1] if r else float("nan") for r in res]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"[inner] → {a.out}", flush=True)


if __name__ == "__main__":
    main()
