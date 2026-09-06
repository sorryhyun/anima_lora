#!/usr/bin/env python3
"""O1 crop builder (``plan_ocr.md``): Manga109-s COO + speech crops by the official split.

    ANIMA_MANGA109S_ROOT=~/manga109s/Manga109s_released_2026_05_21 \\
    python project/cjk_aware_anima_dit/ocr/build_manga109_crops.py --write_split
    python project/cjk_aware_anima_dit/ocr/build_manga109_crops.py --split test   # O0's cut
    python project/cjk_aware_anima_dit/ocr/build_manga109_crops.py                # all splits

For every book in the requested split(s): (a) every COO polygon (truncation links
joined — ``manga109.iter_coo``), (b) a **matched draw** of ``<text>`` speech boxes,
``min(n_coo, n_text)`` per book with a fixed seed, so speech and SFX share the
book split and roughly the count. Crops via the pilot's ``deskew_crop``
(orientation preserved, ``--pad`` 12 %), dropped below ``--min_side`` px.

Output (never in-tree): ``$MANGA109S_ROOT/../derived/crops/<split>/<kind>/<book>_<page>_<id>.png``
+ ``derived/manifest.parquet`` (one row per kept crop; merged across runs by
``(split, kind, id)``) + a stats block on stdout for ``findings.md``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402


def build(
    split_names: list[str], *, pad: float, min_side: int, seed: int, overwrite: bool
):
    mt = m109.pilot_manga_text()
    split = m109.load_split()
    derived = m109.derived_root()
    rows: list[dict] = []
    dropped = Counter()
    for sname in split_names:
        for book in split[sname]:
            coo = list(m109.iter_coo(book))
            speech_all = list(m109.iter_text(book))
            rng = random.Random(f"{seed}:{book}")
            speech = rng.sample(speech_all, min(len(coo), len(speech_all)))
            by_page: dict[int, list[m109.Line]] = {}
            for ln in coo + speech:
                by_page.setdefault(ln.page, []).append(ln)
            for page, lines in sorted(by_page.items()):
                img = cv2.imread(str(m109.page_path(book, page)))
                if img is None:
                    dropped["missing_page"] += len(lines)
                    continue
                for ln in lines:
                    out = derived / "crops" / sname / ln.kind
                    out.mkdir(parents=True, exist_ok=True)
                    fn = out / f"{book}_{page:03d}_{ln.id}.png"
                    if not ln.text:
                        dropped["empty_text"] += 1
                        continue
                    crop, orient = mt.deskew_crop(img, list(ln.poly), pad, min_side)
                    if crop is None:
                        dropped["min_side"] += 1
                        continue
                    if overwrite or not fn.exists():
                        cv2.imwrite(str(fn), crop)
                    h, w = crop.shape[:2]
                    rows.append(
                        dict(
                            split=sname,
                            kind=ln.kind,
                            id=ln.id,
                            book=book,
                            page=page,
                            text=ln.text,
                            joined=ln.joined,
                            orient=orient,
                            w=w,
                            h=h,
                            poly=json.dumps(list(ln.poly)),
                            path=str(fn.relative_to(derived)),
                        )
                    )
            print(
                f"{sname:5s} {book:28s} sfx {len(coo):5d} speech {len(speech):5d}",
                flush=True,
            )
    return pd.DataFrame(rows), dropped


def merge_manifest(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.exists():
        old = pd.read_parquet(path)
        old = old[
            ~old.set_index(["split", "kind", "id"]).index.isin(
                df.set_index(["split", "kind", "id"]).index
            )
        ]
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(path, index=False)
    return df


def stats(df: pd.DataFrame, dropped: Counter) -> str:
    lines = [
        "| split | kind | crops | joined | len p50 / p90 / max | min side p10 / p50 | vertical |",
        "|---|---|---|---|---|---|---|",
    ]
    for (s, k), g in df.groupby(["split", "kind"]):
        L = g.text.str.len()
        ms = np.minimum(g.w, g.h)
        lines.append(
            f"| {s} | {k} | {len(g)} | {int(g.joined.sum())} | "
            f"{int(L.quantile(0.5))} / {int(L.quantile(0.9))} / {L.max()} | "
            f"{int(ms.quantile(0.1))} / {int(ms.quantile(0.5))} | "
            f"{(g.orient == 'vertical').mean():.2f} |"
        )
    chars = Counter("".join(df[df.kind == "sfx"].text))
    lines.append(f"\nSFX char set: {len(chars)}; dropped: {dict(dropped)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--write_split",
        action="store_true",
        help="regenerate assets/coo_split_manga109s.json and exit",
    )
    ap.add_argument(
        "--split",
        choices=m109.SPLITS,
        action="append",
        help="build only these (default all)",
    )
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--min_side", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.write_split:
        sp = m109.write_split()
        print({k: len(v) for k, v in sp.items()}, "->", m109.SPLIT_PATH)
        return
    df, dropped = build(
        a.split or list(m109.SPLITS),
        pad=a.pad,
        min_side=a.min_side,
        seed=a.seed,
        overwrite=a.overwrite,
    )
    path = m109.derived_root() / "manifest.parquet"
    merged = merge_manifest(df, path)
    print(f"\nmanifest {path}: {len(merged)} rows ({len(df)} this run)\n")
    print(stats(df, dropped))


if __name__ == "__main__":
    main()
