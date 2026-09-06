#!/usr/bin/env python3
"""O1 crop builder (``plan_ocr.md``): Manga109-s COO + speech crops by the official split.

    ANIMA_MANGA109S_ROOT=~/manga109s/Manga109s_released_2026_05_21 \\
    python project/cjk_aware_anima_dit/ocr/build_manga109_crops.py --write_split
    python project/cjk_aware_anima_dit/ocr/build_manga109_crops.py --split test   # O0's cut
    python project/cjk_aware_anima_dit/ocr/build_manga109_crops.py --workers 8   # all splits
    python project/cjk_aware_anima_dit/ocr/build_manga109_crops.py --stats_only  # findings block
    … --image_root ~/manga109s/derived/colorized/1024_half_comic --name colorized_1024_half_comic --split train
                                                                  # O3: the same polygons off the colorized pages

``--image_root`` re-cuts the **same** polygons from another page tree
(``<root>/<Book>/<page>.png``, native size — O3's colorized copies); pages
missing there are skipped, and the crops + manifest land under
``derived/crops_<name>/`` / ``derived/manifest_<name>.parquet`` with a
``source=<name>`` column, so ``crop_dataset.load_split(extra=…)`` can mix them
in without touching the grey manifest. Each colorized crop is checked against
the grey crop of the same polygon (stroke-mask IoU after a 1-px dilation, the
O3 pilot's clause (a)) and dropped under ``--min_iou`` (0.8) — a repainted
glyph must not train under the source's label.

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


def stroke_mask(crop: np.ndarray) -> np.ndarray:
    """Otsu-binarised dark pixels (the ink) of a BGR / grey crop."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return g < thr


def dilate(m: np.ndarray, px: int = 1) -> np.ndarray:
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(m.astype(np.uint8), k) > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0])) > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def stroke_iou_d1(a: np.ndarray, b: np.ndarray) -> float:
    """Stroke-mask IoU of two crops of the same polygon, both masks dilated 1 px."""
    return iou(dilate(stroke_mask(a)), dilate(stroke_mask(b)))


def _build_book(args) -> tuple[list[dict], Counter]:
    """One book (a worker unit): its COO lines + a count-matched speech draw."""
    sname, book, pad, min_side, seed, overwrite, image_root, name, min_iou = args
    mt = m109.pilot_manga_text()
    derived = m109.derived_root()
    rows: list[dict] = []
    dropped: Counter = Counter()
    coo = list(m109.iter_coo(book))
    speech_all = list(m109.iter_text(book))
    rng = random.Random(f"{seed}:{book}")
    speech = rng.sample(speech_all, min(len(coo), len(speech_all)))
    by_page: dict[int, list[m109.Line]] = {}
    for ln in coo + speech:
        by_page.setdefault(ln.page, []).append(ln)
    crops_dir = derived / (f"crops_{name}" if name else "crops")
    for page, lines in sorted(by_page.items()):
        src = (
            Path(image_root) / book / f"{page:03d}.png"
            if image_root
            else m109.page_path(book, page)
        )
        if image_root and not src.exists():  # a page the colorize job has not rendered
            dropped["missing_page"] += len(lines)
            continue
        img = cv2.imread(str(src))
        if img is None:
            dropped["missing_page"] += len(lines)
            continue
        grey = cv2.imread(str(m109.page_path(book, page))) if image_root else None
        for ln in lines:
            out = crops_dir / sname / ln.kind
            out.mkdir(parents=True, exist_ok=True)
            fn = out / f"{book}_{page:03d}_{ln.id}.png"
            if not ln.text:
                dropped["empty_text"] += 1
                continue
            crop, orient = mt.deskew_crop(img, list(ln.poly), pad, min_side)
            if crop is None:
                dropped["min_side"] += 1
                continue
            iou_d1 = None
            if grey is not None:
                ref, _ = mt.deskew_crop(grey, list(ln.poly), pad, min_side)
                iou_d1 = stroke_iou_d1(ref, crop) if ref is not None else 0.0
                if iou_d1 < min_iou:
                    dropped["min_iou"] += 1
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
                    source=name or "grey",
                    iou_d1=iou_d1,
                )
            )
    print(
        f"{sname:5s} {book:28s} sfx {len(coo):5d} speech {len(speech):5d}",
        flush=True,
    )
    return rows, dropped


def build(
    split_names: list[str],
    *,
    pad: float,
    min_side: int,
    seed: int,
    overwrite: bool,
    workers: int = 1,
    image_root: str | None = None,
    name: str | None = None,
    min_iou: float = 0.8,
):
    split = m109.load_split()
    m109.derived_root()
    jobs = [
        (sname, book, pad, min_side, seed, overwrite, image_root, name, min_iou)
        for sname in split_names
        for book in split[sname]
    ]
    rows: list[dict] = []
    dropped: Counter = Counter()
    if workers > 1:
        from multiprocessing import Pool

        with Pool(workers) as pool:
            results = pool.imap_unordered(_build_book, jobs)
            for r, d in results:
                rows += r
                dropped.update(d)
    else:
        for job in jobs:
            r, d = _build_book(job)
            rows += r
            dropped.update(d)
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


LEN_BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 8), (9, 12), (13, 10**6)]


def _vocab_chars() -> set[str] | None:
    """Character inventory of manga-ocr's WordPiece vocab (cached HF file)."""
    try:
        from huggingface_hub import hf_hub_download

        mt = m109.pilot_manga_text()
        vocab = Path(hf_hub_download(mt.OCR_MODEL, "vocab.txt")).read_text(
            encoding="utf-8"
        )
    except Exception as e:  # offline + uncached, or no HF at all
        print(f"(vocab coverage skipped: {e})")
        return None
    chars: set[str] = set()
    for tok in vocab.splitlines():
        if tok.startswith("[") and tok.endswith("]"):
            continue
        chars.update(tok[2:] if tok.startswith("##") else tok)
    return chars


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
    # length histogram, SFX vs speech (all splits pooled)
    lines.append(
        "\n| text len | "
        + " | ".join(
            f"{lo}" if lo == hi else (f"{lo}–{hi}" if hi < 10**6 else f"{lo}+")
            for lo, hi in LEN_BINS
        )
        + " |"
    )
    lines.append("|---|" + "---|" * len(LEN_BINS))
    for k, g in df.groupby("kind"):
        L = g.text.str.len()
        cells = [f"{((L >= lo) & (L <= hi)).mean() * 100:.1f} %" for lo, hi in LEN_BINS]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    # char coverage
    sfx = df[df.kind == "sfx"]
    chars = Counter("".join(sfx.text))
    kinds = Counter()
    for c, n in chars.items():
        o = ord(c)
        kinds[
            "hiragana"
            if 0x3040 <= o <= 0x309F
            else "katakana"
            if 0x30A0 <= o <= 0x30FF or 0xFF66 <= o <= 0xFF9F
            else "kanji"
            if 0x4E00 <= o <= 0x9FFF
            else "symbol"
        ] += n
    tot = sum(chars.values())
    lines.append(
        f"\nSFX char set: {len(chars)} chars / {tot} occurrences — "
        + ", ".join(f"{k} {v / tot * 100:.1f} %" for k, v in kinds.most_common())
    )
    vocab = _vocab_chars()
    if vocab is not None:
        for k, g in df.groupby("kind"):
            cc = Counter("".join(g.text))
            miss = {c: n for c, n in cc.items() if c not in vocab}
            n_miss = sum(miss.values())
            top = "".join(c for c, _ in sorted(miss.items(), key=lambda x: -x[1])[:20])
            lines.append(
                f"manga-ocr vocab coverage, {k}: {len(cc) - len(miss)} / {len(cc)} chars, "
                f"{(1 - n_miss / sum(cc.values())) * 100:.2f} % of occurrences; "
                f"missing top: {top!r}"
            )
    lines.append(f"\ndropped: {dict(dropped)}")
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
    ap.add_argument("--workers", type=int, default=1, help="books in parallel")
    ap.add_argument(
        "--image_root",
        help="another page tree (<root>/<Book>/<page>.png) to cut the same polygons from",
    )
    ap.add_argument(
        "--name",
        help="with --image_root: crops_<name>/ + manifest_<name>.parquet + source column",
    )
    ap.add_argument(
        "--min_iou",
        type=float,
        default=0.8,
        help="with --image_root: drop a crop whose stroke IoU (1-px dilated) vs the grey crop is below this",
    )
    ap.add_argument(
        "--stats_only",
        action="store_true",
        help="print the stats block over the existing manifest and exit",
    )
    a = ap.parse_args()
    if bool(a.image_root) != bool(a.name):
        ap.error("--image_root and --name go together")
    manifest_name = f"manifest_{a.name}.parquet" if a.name else "manifest.parquet"
    if a.stats_only:
        df = pd.read_parquet(m109.derived_root() / manifest_name)
        print(stats(df, Counter()))
        return
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
        workers=a.workers,
        image_root=a.image_root,
        name=a.name,
        min_iou=a.min_iou,
    )
    path = m109.derived_root() / manifest_name
    merged = merge_manifest(df, path)
    print(f"\nmanifest {path}: {len(merged)} rows ({len(df)} this run)\n")
    print(stats(df, dropped))


if __name__ == "__main__":
    main()
