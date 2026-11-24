#!/usr/bin/env python3
"""K3 holdout off the gelnote pool: N KO + N ZH crops, one per page, for hand labels.

    .venv/bin/python project/ocr_reader/gate/k3_gelnote.py [--n 50]

Only single-language pages (page tag exactly ``ko`` / ``zh``) are eligible, so
a gate row's language is not a guess. The **whole page** is held out:
``pseudo_label.py --pool gelnote`` drops every crop whose ``image_id`` is in the
TSV, so no gate page reaches a training manifest through its sibling notes.

Writes ``project/ocr_reader/assets/k3_gelnote.tsv`` with an empty ``text``
column for the hand label; ``note_body`` is the gelbooru translation (English),
kept only as context for the labeller. Refuses to overwrite: the draw is
built from this file, so a re-cut after a sweep would leak gate pages.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "project/ocr_reader/assets/k3_gelnote.tsv"
GELNOTE_DEFAULT = "/media/sorryhyun/new/dataset/gelnote_crops"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50, help="crops per language")
    ap.add_argument("--min_side", type=int, default=24)
    ap.add_argument("--max_side", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists — the gelnote draw is cut against it")

    root = Path(os.environ.get("ANIMA_GELNOTE_ROOT", GELNOTE_DEFAULT)).expanduser()
    m = pd.read_json(root / "manifest.jsonl", lines=True)
    side_ok = (m[["w", "h"]].min(axis=1) >= a.min_side) & (
        m[["w", "h"]].max(axis=1) <= a.max_side
    )
    rows = []
    for lang in ("ko", "zh"):
        pool = m[(m.lang == lang) & side_ok]
        pages = pool.image_id.drop_duplicates().sample(frac=1.0, random_state=a.seed)
        for pid in pages[: a.n]:
            rows.append(
                pool[pool.image_id == pid].sample(1, random_state=a.seed).iloc[0]
            )
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            # not df.id: read_json parses "9161673_680261" as the int 9161673680261
            "id": [f"{i}_{k}" for i, k in zip(df.image_id, df.note_id)],
            "lang": df.lang,
            "image_id": df.image_id,
            "note_id": df.note_id,
            "path": df.path,
            "w": df.w,
            "h": df.h,
            "orient": df.orient,
            "text": "",
            "note_body": df.note_body.str.replace(r"\s+", " ", regex=True),
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False)
    held = m[m.image_id.isin(set(out.image_id))]
    print(
        f"wrote {OUT}: {out.lang.value_counts().to_dict()} gate crops; "
        f"{len(held)} pool crops on the {out.image_id.nunique()} held-out pages"
    )


if __name__ == "__main__":
    main()
