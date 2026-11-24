#!/usr/bin/env python3
"""The v7 reads of the Anima training set (``post_image_dataset/resized``) whose mean token
confidence is at or under ``--max_score`` (0.85), as a label source for the review page
(``anima_lowconf``): the read is the first-pass main label.

Input: the OCR stage's dry-run report read with no score floor and pictographs kept (it
writes no sidecar)::

    make daemon-run ARGS="--queue --stall-timeout 900 -m anime_tools.stages.cli.ocr_captions \\
        --dst post_image_dataset/resized --min_score 0 --keep_symbols \\
        --report_dir output/ocr/anima_lowconf/read"

The stage's other floors stand: det >= 0.6, at least 2 characters, no ASCII-only line, a read
the decode guard rejected is gone. The ``sincos`` folder is left out: it is the doujin
gate's source (``eval_data/sincos``), so its pages never become training labels. Each line's crop is cut from the resized image with the
training margin (``core.crops.margin_box``, 4 px) into ``<ANIMA_LOWCONF>/crops/<id>.png``;
the id is ``<artist>__<stem>_<x0>_<y0>_<x1>_<y1>``, stable while the detector boxes are.
Output: ``<ANIMA_LOWCONF>/rows.jsonl``, one line per crop, ``{id, image, box, det, score,
read}``; ``review_page.py --anima_lowconf`` takes it, ``apply_review.py`` writes the
reviewed rows to ``assets/anima_mainsub.tsv``.

    python project/ocr_reader/labeling/anima_lowconf.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from project.ocr_reader.core.crops import margin_box
from project.ocr_reader.core.paths import ANIMA_LOWCONF, OUTPUT

EXCLUDE = ("sincos",)
"""Folders of the resized tree that feed an eval gate (sincos → ``doujin_sfx``)."""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=OUTPUT / "anima_lowconf/read/report.json",
        help="the OCR stage's dry-run report (no score floor)",
    )
    ap.add_argument("--max_score", type=float, default=0.85)
    ap.add_argument("--out", type=Path, default=ANIMA_LOWCONF)
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=list(EXCLUDE),
        help="top-level folders of the resized tree left out (default: the eval sources)",
    )
    a = ap.parse_args()

    rep = json.loads(a.report.read_text(encoding="utf-8"))
    res = Path(rep["dst"])
    crops = a.out / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    rows, n_cut = [], 0
    for r in rep["rows"]:
        lines = [ln for ln in r["lines"] if ln["score"] <= a.max_score]
        if not lines:
            continue
        rel = Path(r["image"])
        if rel.parts[0] in a.exclude:
            continue
        img = None
        for ln in lines:
            x0, y0, x1, y1 = (int(v) for v in ln["box"])
            rid = f"{'__'.join(rel.with_suffix('').parts)}_{x0}_{y0}_{x1}_{y1}"
            f = crops / f"{rid}.png"
            if not f.exists():
                if img is None:
                    img = cv2.imread(str(res / rel))
                if img is None:
                    break
                H, W = img.shape[:2]
                cx0, cy0, cx1, cy1 = margin_box(x0, y0, x1, y1, W, H)
                cv2.imwrite(str(f), img[cy0:cy1, cx0:cx1])
                n_cut += 1
            rows.append(
                dict(
                    id=rid,
                    image=str(rel),
                    box=[x0, y0, x1, y1],
                    det=ln["det"],
                    score=ln["score"],
                    read=ln["text"],
                )
            )
    with open(a.out / "rows.jsonl", "w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    pages = len({x["image"] for x in rows})
    print(
        f"[anima_lowconf] {len(rows)} crops at score <= {a.max_score} on {pages} images "
        f"({n_cut} cut) → {a.out / 'rows.jsonl'}"
    )


if __name__ == "__main__":
    main()
