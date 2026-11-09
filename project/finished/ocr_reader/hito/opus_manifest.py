#!/usr/bin/env python3
"""The Opus relabel rows as a training manifest (``reports/opus_relabel.md``).

    ANIMA_MANGA109S_ROOT=… .venv/bin/python project/finished/ocr_reader/hito/opus_manifest.py \\
        [--exclude ids.txt]

Reads every ``hito_ocr/ko/prelabel_opus*.jsonl`` round (``opus_batch.py merge`` adds
one per round) and ``gelnote_crops/relabel_opus.jsonl``, keeps ``status`` ∈ {ok,
ok_fixed}, and writes ``manifest_pseudo_opus_relabel.parquet`` next to the other
pseudo manifests (``kind = "sfx"``, ``source = "opus"``). Train with
``--extra_manifest pseudo_opus_relabel``.

Round 1 marked the galleries in ``hito_ocr/FLAGGED.md`` ``excluded`` in the jsonl.
Later rounds label every crop and carry the labeller's ``flag`` per row instead;
``--drop_flagged`` leaves out every gallery with a flagged row or listed in
``flagged_galleries_opus*.json`` (round 1's refusals included).
``--exclude`` drops more: a file with one hito gallery / gelnote image id per line
(``#`` comments allowed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
import manga109 as m109  # noqa: E402

HITO_DIR = Path("/media/sorryhyun/new/dataset/hito_ocr/ko")
GELNOTE = Path("/media/sorryhyun/new/dataset/gelnote_crops/relabel_opus.jsonl")
KEEP = {"ok", "ok_fixed"}


def gallery(uid: str) -> str:
    return uid.split(":")[1].split("_")[0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--hito_dir", type=Path, default=HITO_DIR)
    ap.add_argument("--gelnote", type=Path, default=GELNOTE)
    ap.add_argument("--exclude", type=Path)
    ap.add_argument("--drop_flagged", action="store_true")
    ap.add_argument("--out_name", default="opus_relabel")
    a = ap.parse_args()

    drop = set()
    if a.exclude:
        for line in a.exclude.open(encoding="utf-8"):
            line = line.split("#")[0].strip()
            if line:
                drop.add(line)

    files = sorted(a.hito_dir.glob("prelabel_opus*.jsonl")) + [a.gelnote]
    rows = [json.loads(line) for f in files for line in f.open(encoding="utf-8") if line.strip()]
    if a.drop_flagged:
        drop |= {gallery(r["uid"]) for r in rows if r.get("flag")}
        for p in a.hito_dir.glob("flagged_galleries_opus*.json"):
            drop |= set(json.loads(p.read_text()))
    kept = [r for r in rows if r["status"] in KEEP and gallery(r["uid"]) not in drop]

    out = []
    for r in kept:
        src, key = r["uid"].split(":", 1)
        h, w = cv2.imread(r["crop"]).shape[:2]
        if src == "hito":
            rid, side = key.split(":")
            book, page, cid = (
                f"hi{gallery(r['uid'])}",
                int(rid.split("_")[1]),
                f"{rid}_{side}",
            )
        else:
            book, page, cid = f"gn{gallery(r['uid'])}", 0, key
        out.append(
            {
                "split": "train",
                # sfx by design for every pseudo manifest: kind reaches neither the prompt nor
                # the loss, only load_split's speech draw (speech_ratio × the sfx count), where a
                # speech-tagged extra would be subsampled with grey speech. An unknown kind is dropped.
                "kind": "sfx",
                "id": cid,
                "book": book,
                "page": page,
                "text": r["label"],
                "joined": False,
                "orient": "vertical" if h > w else "horizontal",
                "w": w,
                "h": h,
                # the crop is the whole image; CropDataset reads ``path`` only
                "poly": json.dumps([0.0, 0.0, w, 0.0, w, h, 0.0, h]),
                "path": r["crop"],
                "source": "opus",
            }
        )
    df = pd.DataFrame(out)
    path = m109.derived_root() / f"manifest_pseudo_{a.out_name}.parquet"
    df.to_parquet(path, index=False)
    by = df.book.str[:2].map({"hi": "hito", "gn": "gelnote"})
    print(
        f"wrote {path} ({len(df)} rows; dropped by --exclude/--drop_flagged: "
        f"{sum(r['status'] in KEEP for r in rows) - len(kept)})"
    )
    print(pd.crosstab(by, [r["lang"] for r in kept]).to_string())


if __name__ == "__main__":
    main()
