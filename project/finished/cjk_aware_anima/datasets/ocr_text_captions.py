#!/usr/bin/env python3
"""OCR in-image text into arm-C captions for the unmask A/B (plan_ko3 sequel).

For every image with a text mask (train masks are white=trainable, so text
lives in the mask's *complement*): connected components -> padded crops ->
batched manga-ocr with mean-token logprobs (reusing ``manga_text.MangaOCR``)
-> gate (logprob / area / length) -> append the kept lines to the master
caption as quoted flat tags (the pack's quote register shape, ``「シード」``),
plus a ``japanese text`` presence tag when no ``* text`` tag exists.

Captions are written to a SIDECAR dir (masters untouched); images without a
mask are copied through verbatim so the output dir is a complete caption set
for the arm-C dataset. Grammar-safe: tags are appended to the flat bag via
``parse_caption`` / ``compose_caption``, never a hand split.

Usage (GPU -> daemon)::

    make daemon-run ARGS="project/finished/cjk_aware_anima/datasets/ocr_text_captions.py \
        --shard sincos"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

_spec = importlib.util.spec_from_file_location("manga_text", HERE / "manga_text.py")
_manga_text = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_manga_text)

IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg")
TEXT_TAG_RE = re.compile(r"\b(text|speech bubble|sound effects)\b")
# OCR lines that are pure punctuation / a single latin char are mask debris.
JUNK_RE = re.compile(r"^[\W_ｅeＥE．.。・…〜~♡♥!?！？]*$")


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def text_regions(
    mask: np.ndarray, min_area: int, dilate: int
) -> list[tuple[slice, slice, int]]:
    from scipy import ndimage

    text = mask <= 127
    if dilate:
        # Vertical manga columns within one bubble label as separate
        # components; dilating before labeling merges them to bubble level.
        merged = ndimage.binary_dilation(text, iterations=dilate)
        lab, n = ndimage.label(merged)
        lab = np.where(text, lab, 0)  # areas/boxes still from real text pixels
    else:
        lab, n = ndimage.label(text)
    if not n:
        return []
    areas = ndimage.sum(text, lab, range(1, n + 1))
    objs = ndimage.find_objects(lab)
    keep = [
        (objs[i], int(areas[i]))
        for i in np.argsort(areas)[::-1]
        if areas[i] >= min_area and objs[i] is not None
    ]
    return [(sl[0], sl[1], a) for sl, a in keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", default="sincos", help="artist dir under image_dataset/")
    ap.add_argument("--images", type=Path, default=None)
    ap.add_argument("--masks", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=REPO / "post_image_dataset" / "cjk_unmask")
    ap.add_argument("--min_area", type=int, default=400, help="mask-pixel floor per region")
    ap.add_argument("--dilate", type=int, default=9, help="pre-label dilation (mask px)")
    ap.add_argument("--logprob_gate", type=float, default=-0.3)
    ap.add_argument("--max_lines", type=int, default=8, help="kept OCR lines per image")
    ap.add_argument("--pad", type=int, default=8, help="crop padding in image pixels")
    ap.add_argument("--limit", type=int, default=None)
    opts = ap.parse_args()

    images_dir = opts.images or REPO / "image_dataset" / opts.shard
    masks_dir = opts.masks or REPO / "post_image_dataset" / "masks" / opts.shard
    cap_dir = opts.out / "captions" / opts.shard
    cap_dir.mkdir(parents=True, exist_ok=True)

    from anime_tools.captions.position_clauses import compose_caption, parse_caption

    ocr = _manga_text.MangaOCR()
    records, n_copy, n_ocr, n_lines = [], 0, 0, 0

    txts = sorted(images_dir.glob("*.txt"))[: opts.limit]
    for txt in txts:
        stem = txt.stem
        img_path = find_image(images_dir, stem)
        if img_path is None:
            continue
        caption = txt.read_text(encoding="utf-8").strip()
        mask_path = masks_dir / f"{stem}_mask.png"
        lines: list[tuple[str, float, int]] = []
        if mask_path.exists():
            img = Image.open(img_path).convert("RGB")
            mask = np.array(Image.open(mask_path).convert("L"))
            sx, sy = img.width / mask.shape[1], img.height / mask.shape[0]
            crops = []
            regions = text_regions(mask, opts.min_area, opts.dilate)
            for ys, xs, area in regions:
                box = (
                    max(0, int(xs.start * sx) - opts.pad),
                    max(0, int(ys.start * sy) - opts.pad),
                    min(img.width, int(xs.stop * sx) + opts.pad),
                    min(img.height, int(ys.stop * sy) + opts.pad),
                )
                if box[2] - box[0] < 12 or box[3] - box[1] < 12:
                    continue
                crops.append((np.array(img.crop(box))[:, :, ::-1], area, box))
            if crops:
                read = ocr.read([c for c, _, _ in crops])
                seen = set()
                for (text, lp), (_, area, box) in zip(read, crops):
                    text = text.strip().replace(",", "、")
                    if lp <= opts.logprob_gate or len(text) < 2 or JUNK_RE.match(text):
                        continue
                    if text in seen:
                        continue
                    seen.add(text)
                    lines.append((text, lp, area))
                    records.append(
                        {"stem": stem, "text": text, "logprob": round(lp, 3),
                         "area": area, "box": box}
                    )
                lines = lines[: opts.max_lines]
        if lines:
            parsed = parse_caption(caption)
            extra = [] if TEXT_TAG_RE.search(caption) else ["japanese text"]
            extra += [f"「{t}」" for t, _, _ in lines]
            caption = compose_caption(tuple(parsed.flat_tags) + tuple(extra), parsed.clauses)
            n_ocr += 1
            n_lines += len(lines)
        else:
            n_copy += 1
        (cap_dir / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")

    rec_path = opts.out / f"ocr_records_{opts.shard}.jsonl"
    with rec_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"{opts.shard}: {n_ocr} captions gained text ({n_lines} lines), "
        f"{n_copy} copied through -> {cap_dir}\nrecords: {rec_path}"
    )


if __name__ == "__main__":
    main()
