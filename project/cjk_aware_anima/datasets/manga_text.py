#!/usr/bin/env python3
"""OCR pilot over the danbooru text-detection corpus (polygon labels + parquet pages).

The corpus is a *detection* set -- 73,725 danbooru images with text regions, and
**no transcriptions**. Two label files ship for the same split:

  * ``test-00000-of-00001.parquet`` -- images + axis-aligned boxes, normalised
    ``cxcywh``, multi-line bubbles merged into one box (464,399 text boxes);
  * ``polys_test.json`` -- COCO, one **segmentation polygon per text line**
    (602,127 text + 16,446 ``hard_neg``), absolute pixels. Strictly richer: the
    image ids are a subset of the parquet's (the 1,417 missing ones are exactly
    the zero-annotation images), and vertical bubble columns are split per line
    instead of merged.

The polygons are why an OCR pass is worth trying at all: manga-ocr wants a
*single text line*, so the merged parquet boxes would feed it multi-column
blocks. This script answers the three questions that decide whether the OCR'd
text is worth adding to the distillation corpus at all (``../plan.md``, "Data
sources not yet in the mix"):

  a. **SFX share** -- what fraction of text regions are ``はみゅ`` / ``ぱ``
     rather than translatable dialogue;
  b. **OCR failure rate** -- and whether the model's own mean token logprob
     separates good reads from garbage, i.e. whether a confidence gate exists;
  c. **MT quality** -- Hy-MT2 on JA->EN, which is the *easy* direction (general
     Japanese, none of the danbooru-tag knowledge misses that
     ``mt.py --probe`` measured on EN->JA).

Nothing here trains anything. The output is a JSONL of per-region records plus a
``report.md`` with an eyeball table, so the mix decision is made on measured
numbers rather than on the guess that manga text is mostly SFX.

The ``hard_neg`` sample rides along for free: it is the natural negative control
for the confidence gate in (b).

GPU work goes through the daemon:

    make daemon-run ARGS="project/cjk_aware_anima/datasets/manga_text.py --sample 1000"

CPU smoke (crops only, no model):

    uv run python project/cjk_aware_anima/datasets/manga_text.py --sample 24 --no-ocr
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
sys.path.insert(0, str(HERE))

DEFAULT_ROOT = Path("/media/sorryhyun/새 볼륨/dataset")
DEFAULT_POLYS = DEFAULT_ROOT / "polys_test.json"
DEFAULT_PARQUET = DEFAULT_ROOT / "test-00000-of-00001.parquet"
DEFAULT_OUT = ASSETS / "manga_text_pilot"

OCR_MODEL = "kha-white/manga-ocr-base"
CAT_TEXT = 1
CAT_HARD_NEG = 2

KANJI = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
HIRAGANA = re.compile(r"[\u3040-\u309f]")
KATAKANA = re.compile(r"[\u30a0-\u30ff\u31f0-\u31ff]")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9０-９]")
# Sentence-ish material: particles and enders that SFX essentially never carry.
SENTENCE = re.compile(r"[はがをにへとでもの、。？！…ですますだねよかなっ]")
PUNCT_ONLY = re.compile(
    r"[\s。、．，…‥・！？!?~〜ー―－\-—+*/\\()（）「」『』:;\"'’”.,♪♥❤★☆]+"
)


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def load_polys(path: Path) -> tuple[dict, dict]:
    """Return ``({image_id: meta}, {image_id: [ann, ...]})`` from the COCO file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    images = {
        im["id"]: {
            "stem": Path(im["file_name"]).stem,
            "width": im["width"],
            "height": im["height"],
        }
        for im in raw["images"]
    }
    by_image: dict[int, list] = {}
    for ann in raw["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    return images, by_image


def pick_row_groups(n_groups: int, want: int, rng: random.Random) -> list[int]:
    """Evenly spaced groups with a random offset.

    The parquet is **sorted by danbooru post id**, so a contiguous or clustered
    draw would sample one era of the site (and one era of the art). Spreading
    the groups is the cheap way to keep the pilot's distribution honest without
    reading all 8.4 GB.
    """
    want = min(want, n_groups)
    step = n_groups / want
    off = rng.random()
    return sorted({min(n_groups - 1, int((i + off) * step)) for i in range(want)})


# --------------------------------------------------------------------------
# crops
# --------------------------------------------------------------------------


def deskew_crop(img, poly: list[float], pad_frac: float, min_side: int):
    """Rotate a polygon upright and return the tight crop.

    Two rules that matter for manga-ocr:

    * **Orientation is preserved.** ``minAreaRect`` reports an angle in
      ``[0, 90)``; taking it verbatim would transpose a vertical text column
      into a horizontal strip. manga-ocr reads vertical Japanese natively, so we
      pick whichever of ``angle`` / ``angle - 90`` is the smaller rotation and
      only correct the tilt.
    * **Padding.** The polygons hug the glyphs, and the recogniser was trained
      on crops with a margin; a zero-pad crop clips strokes on the outer
      characters.
    """
    import cv2
    import numpy as np

    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if angle > 45:
        # OpenCV >= 4.5 reports angle in (0, 90] with (w, h) measured along the
        # rotated axes: an axis-aligned 30x120 box comes back as (120, 30) @ 90°.
        # Taking angle-90 without swapping the extents cropped a transposed
        # rectangle (found 2026-09-06 on the plan_ocr sincos gate; every
        # axis-aligned box was hit) — swap so the extents follow the rotation.
        angle -= 90
        w, h = h, w
    w = max(w, 1.0)
    h = max(h, 1.0)
    pad = pad_frac * max(w, h)
    w, h = w + 2 * pad, h + 2 * pad

    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    ih, iw = img.shape[:2]
    rotated = cv2.warpAffine(
        img,
        rot,
        (iw, ih),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    crop = cv2.getRectSubPix(rotated, (int(round(w)), int(round(h))), (cx, cy))
    if crop is None or crop.size == 0:
        return None, None
    ch, cw = crop.shape[:2]
    if min(ch, cw) < min_side:
        return None, None
    orient = (
        "vertical" if ch > cw * 1.3 else ("horizontal" if cw > ch * 1.3 else "square")
    )
    return crop, orient


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------


class MangaOCR:
    """manga-ocr (kha-white) with a vocab-only decoder.

    The published tokenizer is ``BertJapaneseTokenizer``, whose *encoder* needs
    MeCab/fugashi -- but generation only ever **decodes**, and decoding is plain
    WordPiece over ``vocab.txt``. Reading the vocab directly keeps two native
    dependencies out of the project for a pilot that never tokenises Japanese
    input.
    """

    def __init__(self, model_id: str = OCR_MODEL, device: str = "cuda") -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoImageProcessor, VisionEncoderDecoderModel

        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.to(device).eval()

        local = Path(model_id) / "vocab.txt"  # a fine-tuned dir (plan_ocr O2)
        vp = local if local.is_file() else Path(hf_hub_download(model_id, "vocab.txt"))
        vocab = vp.read_text(encoding="utf-8")
        self.vocab = vocab.splitlines()
        self.specials = {
            i for i, t in enumerate(self.vocab) if t.startswith("[") and t.endswith("]")
        }

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i in self.specials or i >= len(self.vocab):
                continue
            tok = self.vocab[i]
            out.append(tok[2:] if tok.startswith("##") else tok)
        return "".join(out).replace(" ", "")

    def read(self, crops: list, batch_size: int = 32, max_new_tokens: int = 48):
        """Return ``[(text, mean_token_logprob), ...]``.

        The logprob is the whole point of running greedy with scores: it is the
        only per-region quality signal available without ground truth, and the
        ``hard_neg`` control tells us whether it actually separates.
        """
        import numpy as np
        from PIL import Image

        torch = self.torch
        results: list[tuple[str, float]] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            images = [Image.fromarray(c[:, :, ::-1]).convert("RGB") for c in batch]
            pixel = self.processor(images, return_tensors="pt").pixel_values.to(
                self.device
            )
            with torch.no_grad():
                out = self.model.generate(
                    pixel,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                scores = self.model.compute_transition_scores(
                    out.sequences, out.scores, normalize_logits=True
                )
            seqs = out.sequences[:, -scores.shape[-1] :]
            for row, sc, seq in zip(out.sequences, scores, seqs):
                text = self.decode(row.tolist())
                keep = [
                    float(s)
                    for s, t in zip(sc.tolist(), seq.tolist())
                    if int(t) not in self.specials and not math.isinf(s)
                ]
                results.append((text, float(np.mean(keep)) if keep else float("-inf")))
        return results


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def load_tag_lexicon(path: Path) -> dict[str, tuple[str, str]]:
    """``{japanese surface: (english tag, axis)}`` from the Phase-2a glossary.

    The pilot's fourth question, which only became visible once crops were on
    screen: hairstyle charts and reference sheets put **danbooru tag vocabulary
    in Japanese** directly in the image (``ポニーテール``, ``編み込みセミロング``).
    Those are precisely the ext rows users prompt with, so a hit here is worth
    more than a line of dialogue -- and it arrives with an EN side already
    attached, no MT required.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[str, str]] = {}
    for tag, rec in raw.get("tags", {}).items():
        for surface in [rec.get("ja"), *rec.get("alts", [])]:
            if not surface or len(surface) < 2:
                continue
            if not (
                KANJI.search(surface)
                or HIRAGANA.search(surface)
                or KATAKANA.search(surface)
            ):
                continue
            out.setdefault(surface, (tag, rec.get("axis", "general")))
    return out


def _substring_safe(surface: str, axis: str) -> bool:
    """Is this surface specific enough to be matched *inside* a longer string?

    Same trap the Wikidata lexicon hit: an unguarded flat match looks like high
    recall and is mostly wrong, and it fired twice here. First as ``たな``
    (=shelf) inside ``やがったなこのやろ`` -- short all-hiragana surfaces are
    ordinary grammar fragments. Then, once character names were in the lexicon,
    as ``カイ`` (=irida (pokemon)) and ``アイ`` (=hoshino ai) inside plain
    dialogue: a two-mora katakana *given name* matches everywhere and means
    nothing. Only kanji is dense enough to be trusted at length 2; every other
    script needs 4 characters, and non-``general`` axes need 4 regardless.

    Whole-region matches bypass this entirely -- if the region *is* the surface,
    there is no ambiguity to guard against.
    """
    if axis != "general":
        return len(surface) >= 4
    return len(surface) >= 2 if KANJI.search(surface) else len(surface) >= 4


def find_tags(
    text: str, lex: dict[str, tuple[str, str]], max_len: int = 12
) -> list[str]:
    """Glossary surfaces occurring in ``text``, longest first, nested hits dropped."""
    if not lex or not text:
        return []
    if text in lex:  # whole-region match needs no specificity guard
        return [f"{text}={lex[text][0]}"]
    hits: dict[str, str] = {}
    n = len(text)
    for i in range(n):
        for j in range(i + 2, min(n, i + max_len) + 1):
            sub = text[i:j]
            entry = lex.get(sub)
            if entry and _substring_safe(sub, entry[1]):
                hits[sub] = entry[0]
    keys = sorted(hits, key=len, reverse=True)
    kept = [k for idx, k in enumerate(keys) if not any(k in o for o in keys[:idx])]
    return [f"{k}={hits[k]}" for k in kept]


def script_stats(s: str) -> dict:
    return {
        "kanji": len(KANJI.findall(s)),
        "hiragana": len(HIRAGANA.findall(s)),
        "katakana": len(KATAKANA.findall(s)),
        "latin": len(LATIN.findall(s)),
        "digit": len(DIGIT.findall(s)),
    }


def classify(text: str, st: dict, tag_exact: bool = False) -> str:
    """Coarse register bucket.

    Deliberately crude and reported alongside the raw length histogram, so the
    cut can be re-drawn from the JSONL without re-running the GPU pass. The
    distinction that matters for the mix decision is *translatable dialogue* vs
    *onomatopoeia*, and length + kanji presence carries most of it: SFX are
    short, kana-only, and carry no particles.

    ``tag`` outranks ``sfx`` because the two collide by shape -- ``ポニーテール``
    and ``ドキドキ`` are both mid-length katakana with no particles, and only the
    glossary can tell them apart.
    """
    n = len(text)
    if n == 0:
        return "empty"
    if tag_exact:
        return "tag"
    cjk = st["kanji"] + st["hiragana"] + st["katakana"]
    if cjk == 0:
        if st["latin"] + st["digit"]:
            return "latin_or_digit"
        return "punct" if PUNCT_ONLY.fullmatch(text) else "garbage"
    if n <= 3 and st["kanji"] == 0:
        return "sfx"
    if n <= 6 and st["kanji"] == 0 and not SENTENCE.search(text):
        return "sfx"
    if n <= 5:
        return "short"
    return "line"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def sample_regions(args, rng: random.Random):
    """Draw ``(image, polygon)`` targets, reading only the row groups we need."""
    import pyarrow.parquet as pq

    images, by_image = load_polys(args.polys)
    print(
        f"[polys] {len(images)} images, "
        f"{sum(len(v) for v in by_image.values())} annotations",
        file=sys.stderr,
        flush=True,
    )
    stem_to_id = {m["stem"]: i for i, m in images.items()}

    pf = pq.ParquetFile(args.parquet)
    groups = pick_row_groups(pf.metadata.num_row_groups, args.row_groups, rng)
    per_group_text = max(1, math.ceil(args.sample / len(groups)))
    per_group_neg = math.ceil(args.hard_neg / len(groups)) if args.hard_neg else 0

    picked: list[dict] = []
    for gi, g in enumerate(groups):
        table = pf.read_row_group(g, columns=["image_id", "image"])
        rows = table.to_pylist()
        rng.shuffle(rows)
        want_text, want_neg = per_group_text, per_group_neg
        for row in rows:
            if want_text <= 0 and want_neg <= 0:
                break
            img_id = stem_to_id.get(str(row["image_id"]))
            if img_id is None:
                continue
            anns = by_image.get(img_id, [])
            meta = images[img_id]
            texts = [a for a in anns if int(a["category_id"]) == CAT_TEXT]
            negs = [a for a in anns if int(a["category_id"]) == CAT_HARD_NEG]
            rng.shuffle(texts)
            rng.shuffle(negs)
            take = []
            if want_text > 0 and texts:
                # At most 3 lines from one page: bubbles on a page share a
                # speaker and a scene, so a page-heavy draw would understate
                # the corpus's real variety.
                k = min(len(texts), 3, want_text)
                take += [(a, "text") for a in texts[:k]]
                want_text -= k
            if want_neg > 0 and negs:
                k = min(len(negs), 2, want_neg)
                take += [(a, "hard_neg") for a in negs[:k]]
                want_neg -= k
            if not take:
                continue
            for ann, kind in take:
                picked.append(
                    {
                        "kind": kind,
                        "post_id": int(row["image_id"]),
                        "ann_id": ann["id"],
                        "segmentation": ann["segmentation"][0],
                        "bbox": ann["bbox"],
                        "area": ann["area"],
                        "label_wh": (meta["width"], meta["height"]),
                        "_bytes": row["image"]["bytes"],
                    }
                )
        print(
            f"[sample] group {gi + 1}/{len(groups)} -> {len(picked)} regions",
            file=sys.stderr,
            flush=True,
        )
    rng.shuffle(picked)
    return picked


def build_crops(picked: list[dict], args, crop_dir: Path):
    import cv2
    import numpy as np

    crop_dir.mkdir(parents=True, exist_ok=True)
    records, crops = [], []
    for idx, p in enumerate(picked):
        arr = np.frombuffer(p["_bytes"], dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        lw, lh = p["label_wh"]
        # Guard: the two label files agreed on size in every image checked, but
        # a mismatch would silently shift every polygon.
        sx, sy = iw / lw, ih / lh
        poly = [
            (v * sx if i % 2 == 0 else v * sy) for i, v in enumerate(p["segmentation"])
        ]
        crop, orient = deskew_crop(img, poly, args.pad, args.min_side)
        if crop is None:
            continue
        name = f"{idx:05d}_{p['kind']}.png"
        cv2.imwrite(str(crop_dir / name), crop)
        ch, cw = crop.shape[:2]
        records.append(
            {
                "kind": p["kind"],
                "post_id": p["post_id"],
                "ann_id": p["ann_id"],
                "crop": name,
                "crop_wh": [cw, ch],
                "orientation": orient,
                "poly_points": len(p["segmentation"]) // 2,
                "area_frac": round(p["area"] / (lw * lh), 5),
            }
        )
        crops.append(crop)
    return records, crops


def write_report(out: Path, records: list[dict], args) -> None:
    text_rows = [r for r in records if r["kind"] == "text"]
    neg_rows = [r for r in records if r["kind"] == "hard_neg"]
    cls = Counter(r.get("class", "?") for r in text_rows)
    orient = Counter(r["orientation"] for r in text_rows)
    lens = Counter(min(len(r.get("ocr", "")), 30) for r in text_rows)

    def pct(k: int, tot: int) -> str:
        return f"{100 * k / tot:.1f}%" if tot else "-"

    lines = [
        "# manga text — OCR pilot",
        "",
        f"Sample: **{len(text_rows)} text regions** + {len(neg_rows)} `hard_neg` "
        f"controls, drawn from {args.row_groups} evenly-spaced parquet row groups "
        f"(seed {args.seed}).",
        "",
        "## Register split (question a)",
        "",
        "| class | n | share |",
        "|---|---:|---:|",
    ]
    for k, v in cls.most_common():
        lines.append(f"| `{k}` | {v} | {pct(v, len(text_rows))} |")
    usable = cls["line"] + cls["short"]
    tagged = [r for r in text_rows if r.get("tags")]
    tag_counter: Counter = Counter()
    for r in tagged:
        tag_counter.update(r["tags"])
    lines += [
        "",
        f"**Translatable share** (`line` + `short`): {usable} / {len(text_rows)} "
        f"= {pct(usable, len(text_rows))}.",
        "",
        "## Danbooru tag vocabulary appearing in-image (question d)",
        "",
        f"{len(tagged)} / {len(text_rows)} regions ({pct(len(tagged), len(text_rows))}) "
        f"contain at least one Japanese surface from the Phase-2a glossary; "
        f"{cls['tag']} are an exact whole-region match. These need **no MT** — the "
        "EN side is the tag itself.",
        "",
        "| surface=tag | n |",
        "|---|---:|",
    ]
    for k, v in tag_counter.most_common(30):
        lines.append(f"| `{k}` | {v} |")
    lines += [
        "",
        "## Geometry",
        "",
        "| orientation | n |",
        "|---|---:|",
    ]
    for k, v in orient.most_common():
        lines.append(f"| {k} | {v} |")

    if any("logprob" in r for r in records):
        lines += ["", "## Confidence gate (question b)", ""]
        lines += [
            "| bucket | n | mean logprob | p10 | p90 |",
            "|---|---:|---:|---:|---:|",
        ]
        groups = {
            "text/line": [r for r in text_rows if r.get("class") == "line"],
            "text/sfx": [r for r in text_rows if r.get("class") == "sfx"],
            "text/empty+garbage": [
                r for r in text_rows if r.get("class") in {"empty", "garbage"}
            ],
            "hard_neg": neg_rows,
        }
        for name, rows in groups.items():
            vals = sorted(
                r["logprob"] for r in rows if math.isfinite(r.get("logprob", -math.inf))
            )
            if not vals:
                lines.append(f"| {name} | 0 | - | - | - |")
                continue
            p10 = vals[int(0.1 * (len(vals) - 1))]
            p90 = vals[int(0.9 * (len(vals) - 1))]
            mean = sum(vals) / len(vals)
            lines.append(
                f"| {name} | {len(vals)} | {mean:.3f} | {p10:.3f} | {p90:.3f} |"
            )

        # The gate is only worth having if it keeps dialogue while dropping the
        # controls, so report the trade directly rather than leaving it to be
        # inferred from four means.
        line_rows = [r for r in text_rows if r.get("class") == "line"]
        sfx_rows = [r for r in text_rows if r.get("class") == "sfx"]
        lines += [
            "",
            "### Threshold sweep — what a logprob gate actually buys",
            "",
            "| gate | `line` kept | `sfx` admitted | `hard_neg` admitted (false text) |",
            "|---|---:|---:|---:|",
        ]
        for th in (-0.05, -0.1, -0.2, -0.3, -0.5):

            def kept(rows, th=th):
                n = sum(1 for r in rows if r.get("logprob", -math.inf) > th)
                return f"{n} ({pct(n, len(rows))})"

            lines.append(
                f"| > {th} | {kept(line_rows)} | {kept(sfx_rows)} | {kept(neg_rows)} |"
            )

    lines += ["", "## OCR length histogram", "", "| chars | n |", "|---|---:|"]
    for k in sorted(lens):
        lines.append(f"| {k}{'+' if k == 30 else ''} | {lens[k]} |")

    spot = [r for r in text_rows if r.get("en")][: args.spotcheck]
    if spot:
        lines += [
            "",
            "## Spotcheck (question c)",
            "",
            "| crop | OCR (ja) | Hy-MT2 (en) | logprob |",
            "|---|---|---|---:|",
        ]
        for r in spot:
            ja = r["ocr"].replace("|", "\\|")
            en = r["en"].replace("|", "\\|").replace("\n", " ")
            lp = r.get("logprob", float("nan"))
            lines.append(f"| ![](crops/{r['crop']}) | {ja} | {en} | {lp:.2f} |")

    tag_spot = [r for r in text_rows if r.get("tags")][: args.spotcheck]
    if tag_spot:
        lines += [
            "",
            "## Tag-hit spotcheck — eyeball the glossary matching",
            "",
            "| crop | OCR (ja) | matched | logprob |",
            "|---|---|---|---:|",
        ]
        for r in tag_spot:
            ja = r["ocr"].replace("|", "\\|")
            hits = ", ".join(f"`{t}`" for t in r["tags"][:4])
            lines.append(
                f"| ![](crops/{r['crop']}) | {ja} | {hits} | "
                f"{r.get('logprob', float('nan')):.2f} |"
            )

    negs = [r for r in neg_rows if r.get("ocr") is not None][:20]
    if negs:
        lines += [
            "",
            "## `hard_neg` control — what OCR says on non-text",
            "",
            "| crop | OCR | logprob |",
            "|---|---|---:|",
        ]
        for r in negs:
            lines.append(
                f"| ![](crops/{r['crop']}) | {r['ocr'].replace('|', '\\|')} | "
                f"{r.get('logprob', float('nan')):.2f} |"
            )

    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--polys", type=Path, default=DEFAULT_POLYS)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sample", type=int, default=1000, help="text regions to draw")
    ap.add_argument("--hard-neg", type=int, default=100, help="negative controls")
    ap.add_argument("--row-groups", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--pad", type=float, default=0.06, help="crop pad, fraction of long side"
    )
    ap.add_argument("--min-side", type=int, default=8)
    ap.add_argument("--no-ocr", action="store_true", help="crops + geometry only (CPU)")
    ap.add_argument("--no-mt", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ocr-batch", type=int, default=32)
    ap.add_argument("--mt-model", default="tencent/Hy-MT2-1.8B")
    ap.add_argument("--mt-limit", type=int, default=300)
    ap.add_argument("--mt-batch", type=int, default=32)
    ap.add_argument("--spotcheck", type=int, default=40)
    ap.add_argument("--glossary", type=Path, default=ASSETS / "tag_glossary_ja.json")
    ap.add_argument(
        "--recount",
        action="store_true",
        help="re-classify an existing pilot.jsonl and rewrite report.md (no GPU)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.recount:
        # Every bucket boundary here is a judgement call, and the OCR pass is
        # the expensive part; re-cutting must never cost a second GPU run.
        lex = load_tag_lexicon(args.glossary)
        records = [
            json.loads(line)
            for line in (out / "pilot.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for rec in records:
            text = rec.get("ocr")
            if text is None:
                continue
            st = script_stats(text)
            rec["script"] = st
            rec["tags"] = find_tags(text, lex)
            rec["class"] = classify(text, st, tag_exact=text in lex)
        with (out / "pilot.jsonl").open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        write_report(out, records, args)
        cls = Counter(r.get("class") for r in records if r["kind"] == "text")
        print(f"[recount] {len(records)} regions -> {out}", file=sys.stderr)
        for k, v in cls.most_common():
            print(f"  {k}: {v}", file=sys.stderr)
        return

    picked = sample_regions(args, rng)
    records, crops = build_crops(picked, args, out / "crops")
    print(f"[crops] {len(records)} written to {out / 'crops'}", file=sys.stderr)

    if not args.no_ocr:
        lex = load_tag_lexicon(args.glossary)
        print(f"[lex] {len(lex)} japanese tag surfaces", file=sys.stderr)
        ocr = MangaOCR(device=args.device)
        print(f"[ocr] reading {len(crops)} crops", file=sys.stderr, flush=True)
        for rec, (text, lp) in zip(records, ocr.read(crops, batch_size=args.ocr_batch)):
            st = script_stats(text)
            tags = find_tags(text, lex)
            rec["ocr"] = text
            rec["logprob"] = lp
            rec["script"] = st
            rec["tags"] = tags
            rec["class"] = classify(text, st, tag_exact=text in lex)
        del ocr
        import torch

        torch.cuda.empty_cache()

    if not (args.no_ocr or args.no_mt):
        from mt import MTEngine, Request  # noqa: PLC0415

        todo = [
            r
            for r in records
            if r["kind"] == "text" and r.get("class") in {"line", "short"}
        ][: args.mt_limit]
        if todo:
            print(
                f"[mt] translating {len(todo)} lines ja->en",
                file=sys.stderr,
                flush=True,
            )
            engine = MTEngine(args.mt_model, greedy=True, max_new_tokens=96)
            outs = engine.translate(
                [Request(text=r["ocr"]) for r in todo],
                target_lang="en",
                batch_size=args.mt_batch,
                max_new_tokens=96,
                cache=out / "mt_cache.jsonl",
                progress_every=64,
            )
            for rec, en in zip(todo, outs):
                rec["en"] = en

    with (out / "pilot.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_report(out, records, args)

    text_rows = [r for r in records if r["kind"] == "text"]
    cls = Counter(r.get("class", "?") for r in text_rows)
    print(f"[done] {len(records)} regions -> {out}", file=sys.stderr)
    for k, v in cls.most_common():
        print(f"  {k}: {v} ({100 * v / max(1, len(text_rows)):.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
