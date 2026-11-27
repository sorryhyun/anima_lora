#!/usr/bin/env python3
"""Any reader over the Manga109-s crop manifest.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="project/ocr_reader/ocr/eval_manga109.py --reader manga_ocr"
    … --reader vl16         # PaddleOCR-VL-1.6 crop ``OCR:`` (batched, left-padded, use_cache)
    … --reader manga_ocr --ckpt output/ocr/<run>/best    # a tuned model, same report
    … --reader hayai        # the community hayai-ocr VLM (--ckpt <repo>[@<rev>])
    … --reader hunyuan      # tencent/HunyuanOCR 1.5, official per-task prompt

Metrics per ``kind`` (``sfx`` = COO test crops, ``speech`` = the matched
``<text>`` control) on ``--split`` (default ``test``):

* **exact** — NFKC + whitespace-stripped string equality (hearts, ``ー``,
  small kana all count);
* **sim** — ``scoring.sim`` (NFKC, punctuation / symbols / spaces
  gone, ``SequenceMatcher`` ratio) after ``scoring.normalize_ja`` (vendored)
  on the prediction (vertical if the crop is);
* **runaway** — ``is_runaway`` count (the VL failure class).

Writes ``reports/eval/ocr_eval_<name>.md`` (summary table + worst 25 SFX lines)
and ``output/ocr/eval/<name>_<split>.jsonl`` (every prediction).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import pandas as pd

from project.ocr_reader.core import manga109 as m109
from project.ocr_reader.core.paths import EVAL_OUT, EVAL_REPORTS
from project.ocr_reader.core.readers import MAX_NEW_TOKENS, READERS, VL16_TASKS
from project.ocr_reader.core.scoring import score, summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reader", choices=sorted(READERS), required=True)
    ap.add_argument(
        "--ckpt", help="model dir overriding the stock weights (same reader class)"
    )
    ap.add_argument(
        "--name", help="report name (default: reader, or reader-<ckpt stem>)"
    )
    ap.add_argument("--task", default="OCR:", choices=VL16_TASKS, help="vl16 prompt")
    ap.add_argument(
        "--manifest",
        help="read derived/manifest_<name>.parquet instead (sfxp_gt: the capped-margin "
        "re-cut, build_sfx_prefix.py --gt)",
    )
    ap.add_argument("--split", default="test", choices=m109.SPLITS)
    ap.add_argument("--kind", choices=["sfx", "speech"], action="append")
    ap.add_argument("--limit", type=int, help="first N crops per kind (smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"decode cap per crop (default {MAX_NEW_TOKENS}); a long speech bubble "
        "needs one token per kana on manga-ocr",
    )
    a = ap.parse_args()
    name = a.name or (f"{a.reader}-{Path(a.ckpt).stem}" if a.ckpt else a.reader)
    if a.task != "OCR:" and not a.name:
        name += "_" + a.task.rstrip(":").lower()

    derived = m109.derived_root()
    df = pd.read_parquet(
        derived
        / (f"manifest_{a.manifest}.parquet" if a.manifest else "manifest.parquet")
    )
    df = df[df.split == a.split]
    if a.kind:
        df = df[df.kind.isin(a.kind)]
    if a.limit:
        df = df.groupby("kind", group_keys=False).head(a.limit)
    df = df.sort_values(["kind", "book", "page", "id"]).reset_index(drop=True)
    crops = [cv2.imread(str(derived / p)) for p in df.path]
    assert all(c is not None for c in crops), (
        "missing crop png — rerun build_manga109_crops"
    )

    reader = READERS[a.reader](a.ckpt, a.device)
    reader.max_tokens = a.max_new_tokens
    reader.task = a.task
    t0 = time.time()
    preds = reader.read(crops, list(df.orient), a.bs)
    wall = time.time() - t0
    scored = score(df, preds)

    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    with (EVAL_OUT / f"{name}_{a.split}.jsonl").open("w", encoding="utf-8") as f:
        for _, r in scored.iterrows():
            f.write(
                json.dumps(
                    {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()},
                    ensure_ascii=False,
                )
                + "\n"
            )
    md = summary(scored, name, a.split, wall)
    if not a.limit:
        EVAL_REPORTS.mkdir(parents=True, exist_ok=True)
        (EVAL_REPORTS / f"ocr_eval_{name}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
