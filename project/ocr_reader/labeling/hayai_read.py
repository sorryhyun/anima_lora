#!/usr/bin/env python3
"""hayai-ocr v2.7 Supernova over every AnimeText detector box of the Manga109-s train / val
pages (``manifest_sfxp_det.parquet``, 164,090 boxes): a first-pass read for the boxes that
have no label (``neg_other``: no annotation inside; ``drop``: an SFX cut partway) and a second
read beside the main / sub proposals of the rest (``m109_mainsub.py``), the mains waiting on a
line break among them.

A box of ``manifest_sfxp_det_all`` is read from its crop on disk; any other box is cut from
the page with the same 4 px margin (``build_sfx_prefix.margin_box``). hayai reads a crop as a
whole: it does not tell main from sub.

The reads are first-pass labels for the user's review. The weights are Apache-2.0; the
licence call is the user's (README, Rules: Labels).

Output: one JSON line per box, ``{id, split, cat, book, page, box, read}``, appended in
chunks; a rerun skips the ids already written.

    ANIMA_MANGA109S_ROOT=... make daemon-run ARGS="--queue --stall-timeout 900 \\
        project/ocr_reader/labeling/hayai_read.py --out output/ocr/hayai/hayai27_det.jsonl"
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

from project.ocr_reader.core import manga109 as m109
from project.ocr_reader.core.crops import margin_box

MODEL = "JustANormalTinkerer/hayai-ocr-v2.7-supernova"
REVISION = "90fc61482c33858ee9ea227cd797f6fdff33345c"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=2048, help="boxes per write")
    ap.add_argument("--limit", type=int, help="read only this many boxes (smoke)")
    ap.add_argument("--cats", nargs="+", help="only these cats (e.g. neg_other drop)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    d = m109.derived_root()
    det = pd.read_parquet(d / "manifest_sfxp_det.parquet")
    # the manifests' ``id`` is local to a book; the crop files and the main / sub
    # proposals name a box ``<book>_<id>``
    cut = pd.read_parquet(d / "manifest_sfxp_det_all.parquet")
    cut = pd.Series(cut.path.values, index=cut.book + "_" + cut.id)
    det["id"] = det.book + "_" + det.id
    assert det.id.is_unique and cut.index.is_unique
    det["path"] = det.id.map(cut)
    if a.cats:
        det = det[det.cat.isin(a.cats)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        done = {json.loads(ln)["id"] for ln in a.out.open() if ln.strip()}
    det = det[~det.id.isin(done)].sort_values(["book", "page"]).reset_index(drop=True)
    if a.limit:
        # a spread over the cats, not the first pages of one book
        det = det.groupby("cat", group_keys=False).head(
            max(1, a.limit // det.cat.nunique())
        )
        det = det.sort_values(["book", "page"]).reset_index(drop=True)
    print(
        f"[hayai] {len(det)} boxes to read ({len(done)} already in {a.out}); "
        f"{det.cat.value_counts().to_dict()}; from pages {int(det.path.isna().sum())}",
        flush=True,
    )

    import torch
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(
        a.model, revision=a.revision, trust_remote_code=True
    )
    model = model.to(a.device).eval()
    proc = AutoProcessor.from_pretrained(
        a.model, revision=a.revision, trust_remote_code=True
    )

    def load(r) -> Image.Image | None:
        if isinstance(r.path, str):
            img = cv2.imread(str(d / r.path))
        else:
            page = cv2.imread(str(m109.page_path(r.book, int(r.page))))
            if page is None:
                return None
            H, W = page.shape[:2]
            x0, y0, x1, y1 = margin_box(*(int(v) for v in r.box), W, H)
            img = page[y0:y1, x0:x1]
        if img is None or not img.size:
            return None
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    t0, n = time.time(), 0
    with ThreadPoolExecutor(8) as pool, a.out.open("a") as fh:
        for s in range(0, len(det), a.chunk):
            part = det.iloc[s : s + a.chunk]
            rows = list(part.itertuples())
            crops = list(pool.map(load, rows))
            keep = [i for i, c in enumerate(crops) if c is not None]
            with torch.inference_mode():
                texts = model.ocr(
                    [crops[i] for i in keep],
                    proc,
                    max_patches="auto",
                    batch_size=a.bs,
                    escalate=False,
                    max_new_tokens=128,
                )
            reads = dict(zip(keep, texts, strict=True))
            for i, r in enumerate(rows):
                rec = {
                    "id": r.id,
                    "split": r.split,
                    "cat": r.cat,
                    "book": r.book,
                    "page": int(r.page),
                    "box": [int(v) for v in r.box],
                    "read": reads.get(i),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n += len(rows)
            print(
                f"[hayai] {n}/{len(det)} {n / (time.time() - t0):.1f} boxes/s",
                flush=True,
            )
    print(f"[hayai] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
