#!/usr/bin/env python
"""Rebuild people_count anchor artifacts.

Replaces girl_count + boy_count with a single 7-way mutex people_count group.
Prototypes are re-derived from real slot-level T5 embeddings at the constituent
count tags, NOT averaged from the existing per-tag prototypes.

Merges new keys (prefix ``people=``) into:
  - {output_dir}/phase2_class_prototypes.safetensors
  - {output_dir}/phase1_positions.json  (per_class_occurrences)

The original per-tag entries (``1girl``, ``2girls``, ``1boy``, ...) are left
untouched but are no longer referenced by ``anchors.yaml``.

Classes (disjoint partition of the dataset):
  1girl          — 1girl, no boy tag
  1girl, 1boy    — exactly 1girl + 1boy
  2girls         — 2girls, no boy tag
  2girls, 1boy   — exactly 2girls + 1boy
  1boy           — no girl tag, 1boy (solo male)
  multi          — 3+girls / 2+boys-with-any-girls / bare multiple_*
  no_people      — none of the count tags present (prototype = zero)

Usage:
    python scripts/img2emb/rebuild_people_prototypes.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from library.io.cache import discover_cached_images  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


PROTO_PREFIX = "people="
DEFAULT_SLOT = 2
D_EMB = 1024  # T5 hidden


# --------------------------------------------------------------------------- tag → T5 slot extraction
# Copied from bench/inversionv2/tag_slot_analysis.py so this script doesn't
# reach into bench/ for runtime code. The bench script remains the analysis
# source of truth; keep these helpers in sync if the round-trip / offset-mapping
# logic changes there.


def split_tags(caption: str) -> list[tuple[str, int, int]]:
    """Split caption into (tag, char_start, char_end) tuples."""
    out = []
    cursor = 0
    for raw in caption.split(","):
        tag = raw.strip()
        seg_start = cursor
        seg_end = cursor + len(raw)
        if tag:
            lws = len(raw) - len(raw.lstrip())
            char_start = seg_start + lws
            char_end = char_start + len(tag)
            out.append((tag, char_start, char_end))
        cursor = seg_end + 1
    return out


def tokens_in_range(offset_mapping, char_start: int, char_end: int) -> list[int]:
    """Token indices whose offset falls strictly inside [char_start, char_end)."""
    slots = []
    for i, (s, e) in enumerate(offset_mapping):
        if s == e == 0:  # special / padding
            continue
        if s >= char_start and e <= char_end:
            slots.append(i)
    return slots


def process_variant(
    stem: str,
    vi: int,
    sd: dict[str, torch.Tensor],
    tokenizer,
    zero_eps: float = 1e-6,
):
    """Decode + re-tokenize one (image, variant); extract tag → slot map."""
    ids_key = f"t5_input_ids_v{vi}"
    emb_key = f"crossattn_emb_v{vi}"
    if ids_key not in sd or emb_key not in sd:
        return None
    ids = sd[ids_key].tolist()
    while ids and ids[-1] == 0:
        ids.pop()
    if not ids:
        return None

    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    enc = tokenizer(
        decoded,
        return_offsets_mapping=True,
        truncation=True,
        max_length=len(ids) + 8,
        add_special_tokens=True,
    )
    if enc["input_ids"] != ids:
        # Cached ids contain <unk> (id 2) for chars SentencePiece can't encode
        # (e.g. Japanese subtitles). skip_special_tokens drops <unk>, so
        # round-trip shortens — data property, not tokenizer drift.
        return {"_skip": True, "has_unk": 2 in ids}

    emb = sd[emb_key].to(torch.float32)
    nonzero = emb.abs().amax(dim=-1) > zero_eps
    n_content = (
        int(nonzero.nonzero(as_tuple=False)[-1].item()) + 1 if nonzero.any() else 0
    )

    tags_with_slots = []
    for tag, cs, ce in split_tags(decoded):
        slots = tokens_in_range(enc["offset_mapping"], cs, ce)
        if not slots:
            continue
        tags_with_slots.append((tag, slots[0], slots[-1], len(slots)))

    return {
        "caption": decoded,
        "tags": tags_with_slots,
        "emb": emb,
        "n_content": n_content,
    }


# --------------------------------------------------------------------------- people_count partition

PEOPLE_CLASSES = [
    "1girl",
    "1girl, 1boy",
    "2girls",
    "2girls, 1boy",
    "1boy",
    "multi",
    "no_people",
]

# Check tags largest-first so "3girls" wins over "1girl" when both somehow appear
# (shouldn't happen in canonical booru, but be explicit).
GIRL_TAGS_DESC = ["6+girls", "5girls", "4girls", "3girls", "2girls", "1girl"]
BOY_TAGS_DESC = ["5boys", "4boys", "3boys", "2boys", "1boy"]
ALL_COUNT_TAGS = set(GIRL_TAGS_DESC) | set(BOY_TAGS_DESC)
MG_TAGS = {"multiple girls", "multiple_girls"}
MB_TAGS = {"multiple boys", "multiple_boys"}


def classify(tags: set[str]) -> str:
    girls = next((t for t in GIRL_TAGS_DESC if t in tags), None)
    boys = next((t for t in BOY_TAGS_DESC if t in tags), None)
    has_mg = bool(tags & MG_TAGS)
    has_mb = bool(tags & MB_TAGS)
    if girls == "1girl" and boys is None:
        return "1girl"
    if girls == "1girl" and boys == "1boy":
        return "1girl, 1boy"
    if girls == "2girls" and boys is None:
        return "2girls"
    if girls == "2girls" and boys == "1boy":
        return "2girls, 1boy"
    if girls is None and boys == "1boy":
        return "1boy"
    if girls is None and boys is None and not has_mg and not has_mb:
        return "no_people"
    return "multi"


def constituents(klass: str) -> set[str]:
    if klass == "1girl":
        return {"1girl"}
    if klass == "1girl, 1boy":
        return {"1girl", "1boy"}
    if klass == "2girls":
        return {"2girls"}
    if klass == "2girls, 1boy":
        return {"2girls", "1boy"}
    if klass == "1boy":
        return {"1boy"}
    if klass == "multi":
        return set(ALL_COUNT_TAGS)
    if klass == "no_people":
        return set()
    raise ValueError(klass)


def read_caption_tags(txt_path: Path) -> set[str]:
    return {t.strip() for t in txt_path.read_text().split(",")}


def iter_variants(spec: str) -> list[int]:
    if spec == "all":
        return list(range(8))
    return [int(spec)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image_dir", default="post_image_dataset")
    p.add_argument(
        "--tokenizer",
        default=str(REPO_ROOT / "library" / "anima" / "configs" / "t5_old"),
    )
    p.add_argument("--variants", default="all")
    p.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "bench" / "inversionv2" / "results" / "tag_slot"),
    )
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Report stats, don't touch the prototype/positions files.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)
    if not image_dir.is_absolute():
        image_dir = REPO_ROOT / image_dir

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    variants = iter_variants(args.variants)
    logger.info(f"tokenizer={args.tokenizer}  variants={variants}")

    images = discover_cached_images(str(image_dir))
    if args.max_images:
        images = images[: args.max_images]
    logger.info(f"found {len(images)} cached images in {image_dir}")

    # --- classify images by caption (order-invariant; shuffle doesn't matter)
    stem_to_class: dict[str, str] = {}
    class_counts: dict[str, int] = defaultdict(int)
    missing_caption = 0
    for img in images:
        txt = image_dir / f"{img.stem}.txt"
        if not txt.exists():
            missing_caption += 1
            continue
        klass = classify(read_caption_tags(txt))
        stem_to_class[img.stem] = klass
        class_counts[klass] += 1

    logger.info("image counts per class:")
    for c in PEOPLE_CLASSES:
        logger.info(f"  {c!r:>20s} : {class_counts[c]}")
    if missing_caption:
        logger.warning(f"{missing_caption} images had no .txt caption and were skipped")

    # --- walk cached te safetensors; collect slot-vectors + occurrences
    class_vectors: dict[str, list[torch.Tensor]] = defaultdict(list)
    class_occurrences: dict[str, list[list]] = defaultdict(list)
    rt_fail = 0
    ok = 0
    missing_te = 0

    for idx, img in enumerate(images):
        klass = stem_to_class.get(img.stem)
        if klass is None:
            continue
        cons = constituents(klass)

        if img.te_path is None:
            missing_te += 1
            continue
        try:
            sd = load_file(img.te_path)
        except Exception as e:
            logger.warning(f"te load failed {img.te_path}: {e}")
            continue

        for vi in variants:
            result = process_variant(img.stem, vi, sd, tokenizer)
            if result is None or result.get("_skip"):
                rt_fail += 1
                continue
            ok += 1
            emb = result["emb"]
            tags_list = result["tags"]  # list of (tag, s0, s1, n_tok)
            tag_index = {t[0]: t for t in tags_list}

            if klass == "no_people":
                # No constituent slot to anchor to — use default_slot as a
                # placeholder and contribute no vector (prototype stays zero).
                class_occurrences[klass].append(
                    [img.stem, vi, DEFAULT_SLOT, DEFAULT_SLOT, 1]
                )
                continue

            present = [tag_index[t] for t in cons if t in tag_index]
            if not present:
                # Caption classified to this class but the constituent tag
                # isn't visible in this variant (tokenizer drift edge case).
                continue

            # Canonical injection slot = earliest constituent.
            present.sort(key=lambda t: t[1])
            earliest = present[0]
            class_occurrences[klass].append(
                [img.stem, vi, int(earliest[1]), int(earliest[2]), int(earliest[3])]
            )

            # Prototype contribution: one slot-vector per constituent tag
            # present. For `multi`, this means the prototype averages over
            # however many count tags each image happens to have.
            for tag in present:
                s0 = int(tag[1])
                class_vectors[klass].append(emb[s0].clone().float().cpu())

        if (idx + 1) % 200 == 0:
            logger.info(
                f"  processed {idx + 1}/{len(images)}  ok={ok}  rt_fail={rt_fail}"
            )

    logger.info(
        f"extraction: ok={ok}  rt_fail={rt_fail}  missing_te={missing_te}"
    )

    # --- mean per class -> prototypes
    prototypes: dict[str, torch.Tensor] = {}
    for klass in PEOPLE_CLASSES:
        vecs = class_vectors[klass]
        if not vecs:
            prototypes[klass] = torch.zeros(D_EMB, dtype=torch.float32)
            logger.info(f"  proto[{klass!r}] zero (n_vecs=0)")
            continue
        X = torch.stack(vecs, dim=0)
        mean_vec = X.mean(dim=0)
        prototypes[klass] = mean_vec
        logger.info(
            f"  proto[{klass!r}] n_vecs={len(vecs):>5d}  "
            f"norm={mean_vec.norm().item():.2f}"
        )

    # --- merge into artifacts
    proto_path = out_dir / "phase2_class_prototypes.safetensors"
    pos_path = out_dir / "phase1_positions.json"

    if args.dry_run:
        logger.info("dry_run — skipping writes")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    existing_proto = load_file(str(proto_path)) if proto_path.exists() else {}
    for klass, v in prototypes.items():
        existing_proto[f"{PROTO_PREFIX}{klass}"] = v.to(torch.bfloat16).contiguous()
    save_file(existing_proto, str(proto_path))
    logger.info(f"updated -> {proto_path}")

    existing_pos = (
        json.loads(pos_path.read_text())
        if pos_path.exists()
        else {"per_class_occurrences": {}}
    )
    occ = existing_pos.setdefault("per_class_occurrences", {})
    for klass, entries in class_occurrences.items():
        occ[f"{PROTO_PREFIX}{klass}"] = entries
    pos_path.write_text(json.dumps(existing_pos))
    logger.info(f"updated -> {pos_path}")


if __name__ == "__main__":
    main()
