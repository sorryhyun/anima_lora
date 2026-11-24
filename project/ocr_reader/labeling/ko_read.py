#!/usr/bin/env python3
"""A reader over the review page's label rows, as trained (the manifest crops), with its
confidence: the KO ``OCR:`` crops of ``manifest_sfxp_kozh.parquet`` (gelnote_ko,
swaprelax_v5) and the hito v4 self-labels (``manifest_pseudo_hito_v4.parquet``, as
``hito_v4_ja`` / ``hito_v4_ko``). Input for ``ko_label_fix.py``, whose flags file the
review page (``review_page.py --flags``) sorts on ("reader disagrees").

The reader is the shipped v7 by default (``OCR:`` only); ``--adapter_dir
output/ocr/<run>/best`` reads with a local run (``--version v6`` for a run on 12 % crops,
``--version v7_preview`` for a two-prompt run).
Each crop is read under ``OCR:``; ``why`` says what the guard made of the decode:
``""`` (kept), ``length`` (longer than ``length_cap`` for the crop: the read is kept, the
cap's 16 px glyph is larger than much of the KO lettering), ``runaway`` / ``empty`` (``read
= ""``, ``conf = NaN``). A two-prompt reader reads the ``empty`` crops again under ``SFX:``;
a read there is stored with ``task = "SFX:"`` (the review page checks its SFX box).
``--fill <reads.parquet>`` does only that second pass (plus the ``OCR:`` re-read for the
reason) over an earlier file's empty rows.

    ANIMA_MANGA109S_ROOT=... make daemon-run ARGS="--stall-timeout 600 \\
        project/ocr_reader/labeling/ko_read.py --out <reads.parquet> \\
        [--adapter_dir output/ocr/<run>/best --version v7] [--fill <reads.parquet>]"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from project.ocr_reader.core import manga109 as m109

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
        "--version", default="v7", help="OcrReader version (v7 / v6 / v7_preview)"
    )
    ap.add_argument(
        "--name",
        help="reader name shown on the review page (default: the run dir's name, else --version)",
    )
    ap.add_argument(
        "--fill",
        type=Path,
        help="an earlier reads file: re-read only its empty rows (OCR: reason, then SFX:)",
    )
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    name = a.name or (a.adapter_dir.parent.name if a.adapter_dir else a.version)
    d = m109.derived_root()
    if a.fill:
        df = pd.read_parquet(a.fill)
        for c, v in (("task", "OCR:"), ("why", "")):
            if c not in df:
                df[c] = [v if r else "" for r in df.read.fillna("")]
        todo = df.index[df.read.fillna("") == ""]
    else:
        df = rows(d, a.sources)
        df["read"], df["conf"], df["task"], df["why"] = "", float("nan"), "", ""
        todo = df.index
    print(
        f"[read] {len(todo)} of {len(df)} crops "
        f"{df.loc[todo].source.value_counts().to_dict()}",
        flush=True,
    )
    from anime_tools.ocr.reader import (
        PROMPT,
        SFX_PROMPT,
        OcrReader,
        guard,
        is_runaway,
        normalize_read,
    )

    reader = OcrReader.load(
        device=a.device, batch_size=a.bs, adapter_dir=a.adapter_dir, version=a.version
    )

    def read(idx, task):
        crops = [cv2.imread(str(d / p)) for p in df.loc[idx, "path"]]
        t0 = time.time()
        raws = reader.read_raw_scored(crops, task)
        print(f"[read] {task} {len(idx) / (time.time() - t0):.1f} crops/s", flush=True)
        return [(raw, conf, c) for (raw, conf), c in zip(raws, crops, strict=True)]

    for i, (raw, conf, c) in zip(todo, read(todo, PROMPT), strict=True):
        text, norm = guard(raw, int(c.shape[1]), int(c.shape[0])), normalize_read(raw)
        why = (
            ""
            if text
            else "empty"
            if not norm
            else "runaway"
            if is_runaway(norm)
            else "length"
        )
        keep = text or (norm if why == "length" else "")
        df.loc[i, ["read", "conf", "task", "why"]] = [
            keep,
            conf if keep else float("nan"),
            PROMPT if keep else "",
            why,
        ]
    empty = df.index[df.why == "empty"]
    if reader.two_prompt and len(empty):
        n = 0
        for i, (raw, conf, c) in zip(empty, read(empty, SFX_PROMPT), strict=True):
            text = guard(raw, int(c.shape[1]), int(c.shape[0]))
            if text:
                df.loc[i, ["read", "conf", "task"]] = [text, conf, SFX_PROMPT]
                n += 1
        print(f"[read] SFX: filled {n} of {len(empty)} empty OCR: reads", flush=True)
    print(f"[read] why: {df.loc[todo].why.value_counts().to_dict()}", flush=True)
    df["reader"] = name
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"[read] → {a.out}", flush=True)


if __name__ == "__main__":
    main()
