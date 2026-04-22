#!/usr/bin/env python
"""Rebuild anchor artifacts for img2emb training.

Produces the two files every anchor group in ``scripts/img2emb/anchors.yaml``
consumes:
  - {output_dir}/phase2_class_prototypes.safetensors  (class prototypes)
  - {output_dir}/phase1_positions.json                (per-image anchor slots)

The yaml is the source of truth for *which groups exist*, *their class lists*,
*prototype key prefix* and *default slot*. This script only owns the
raw-tag → class mapping — that's booru-semantic and can't live in data. Each
yaml group must have a classifier registered in ``CLASSIFIERS`` below; a new
group is added by (1) declaring it in anchors.yaml and (2) registering its
classifier here.

Prototypes are the mean of slot-level T5 embeddings at the anchor tag slot,
not averages of prior per-tag prototypes.

Usage:
    python scripts/img2emb/rebuild_anchor_artifacts.py
    python scripts/img2emb/rebuild_anchor_artifacts.py --anchors_yaml my.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import yaml
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from library.io.cache import discover_cached_images  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


DEFAULT_ANCHORS_YAML = Path(__file__).parent / "anchors.yaml"
D_EMB = 1024  # T5 hidden


# --------------------------------------------------------------------------- tag → T5 slot extraction


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

    # Cached ids may contain <unk> for characters SentencePiece can't encode
    # (e.g. '~', '^'). Dropping <unk> then re-tokenizing is lossy because
    # SentencePiece normalizes consecutive ▁ tokens, so strict round-trip can
    # drift around <unk> positions. Since anchor tags (rating, people_count)
    # always live near the start of the caption — always before the first
    # <unk> — we only need a clean round-trip on the pre-<unk> prefix.
    unk_id = tokenizer.unk_token_id
    eos_id = tokenizer.eos_token_id
    cutoff = len(ids)
    for i, t in enumerate(ids):
        if t == unk_id or t == eos_id:
            cutoff = i
            break
    prefix_ids = ids[:cutoff]
    # Trailing whitespace-only tokens (e.g. the SentencePiece '▁' that was
    # glued to the now-excluded <unk>/EOS) get dropped by the tokenizer's
    # whitespace normalization on re-encode. Trim them so the round-trip
    # matches.
    while prefix_ids and tokenizer.decode(
        [prefix_ids[-1]], skip_special_tokens=True
    ).strip() == "":
        prefix_ids = prefix_ids[:-1]
    if not prefix_ids:
        return {"_skip": True, "has_unk": unk_id in ids}

    decoded = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    enc = tokenizer(
        decoded,
        return_offsets_mapping=True,
        truncation=True,
        max_length=cutoff + 8,
        add_special_tokens=False,
    )
    re_ids = enc["input_ids"]
    if re_ids != prefix_ids:
        # Prefix has no <unk> yet still drifts — real tokenizer normalization
        # difference. Give up; these are rare.
        return {"_skip": True, "has_unk": unk_id in ids}

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
        # Slots index into prefix_ids, which is ids[:cutoff] — so they already
        # map directly into the cached ids/emb. No remap needed.
        tags_with_slots.append((tag, slots[0], slots[-1], len(slots)))

    return {
        "caption": decoded,
        "tags": tags_with_slots,
        "emb": emb,
        "n_content": n_content,
    }


# --------------------------------------------------------------------------- classifiers
#
# Each registered classifier is (classify, constituents):
#   classify(tags: set[str]) -> class_name | None
#       Returns the class this image falls under for this group, or None if
#       the image doesn't belong to any class (image is skipped for the group).
#   constituents(class_name: str) -> set[str]
#       The raw caption tags whose slot(s) identify this class in a caption.
#       If empty, the group's default_slot is used as a placeholder (e.g. the
#       people_count "no_people" class, which has no corresponding tag).

# --- rating ---

def classify_rating(tags: set[str], classes=("explicit", "sensitive", "general")) -> str | None:
    for r in classes:
        if r in tags:
            return r
    return None


def rating_constituents(klass: str) -> set[str]:
    # Rating classes ARE the tag names.
    return {klass}


# --- people_count ---
#
# Check tags largest-first so "3girls" wins over "1girl" if both somehow appear
# (shouldn't happen in canonical booru captions, but be explicit).
GIRL_TAGS_DESC = ["6+girls", "5girls", "4girls", "3girls", "2girls", "1girl"]
BOY_TAGS_DESC = ["5boys", "4boys", "3boys", "2boys", "1boy"]
ALL_COUNT_TAGS = set(GIRL_TAGS_DESC) | set(BOY_TAGS_DESC)
MG_TAGS = {"multiple girls", "multiple_girls"}
MB_TAGS = {"multiple boys", "multiple_boys"}


def classify_people(tags: set[str]) -> str:
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


def people_constituents(klass: str) -> set[str]:
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
        return set()  # -> use default_slot placeholder
    raise ValueError(klass)


CLASSIFIERS: dict[
    str,
    tuple[Callable[[set[str]], str | None], Callable[[str], set[str]]],
] = {
    "rating": (classify_rating, rating_constituents),
    "people_count": (classify_people, people_constituents),
}


# --------------------------------------------------------------------------- yaml spec


@dataclass
class AnchorGroupSpec:
    name: str
    classes: list[str]
    proto_key_prefix: str
    default_slot: int
    classify: Callable[[set[str]], str | None]
    constituents: Callable[[str], set[str]]


def load_group_specs(yaml_path: Path) -> list[AnchorGroupSpec]:
    doc = yaml.safe_load(yaml_path.read_text())
    if not isinstance(doc, dict) or not doc:
        raise ValueError(f"{yaml_path} is empty or not a mapping")

    specs: list[AnchorGroupSpec] = []
    for name, cfg in doc.items():
        if name not in CLASSIFIERS:
            raise ValueError(
                f"anchors.yaml group '{name}' has no classifier registered in "
                f"{Path(__file__).name}. Add an entry to CLASSIFIERS."
            )
        classes_cfg = cfg.get("classes")
        if classes_cfg == "auto" or not classes_cfg:
            raise ValueError(
                f"group '{name}': classes must be an explicit list "
                f"(rebuilder produces the proto file, so it can't resolve 'auto')"
            )
        classify, constituents = CLASSIFIERS[name]
        specs.append(
            AnchorGroupSpec(
                name=name,
                classes=list(classes_cfg),
                proto_key_prefix=str(cfg.get("proto_key_prefix", "")),
                default_slot=int(cfg.get("default_slot", 0)),
                classify=classify,
                constituents=constituents,
            )
        )
    return specs


def read_caption_tags(txt_path: Path) -> set[str]:
    return {t.strip() for t in txt_path.read_text().split(",")}


def iter_variants(spec: str) -> list[int]:
    if spec == "all":
        return list(range(8))
    return [int(spec)]


# --------------------------------------------------------------------------- main


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
        default=str(REPO_ROOT / "output" / "img2embs" / "anchors"),
    )
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Report stats, don't touch the prototype/positions files.",
    )
    p.add_argument(
        "--anchors_yaml",
        default=str(DEFAULT_ANCHORS_YAML),
        help="Anchor-group spec (group names, classes, proto_key_prefix, "
        "default_slot). Source of truth for everything except the "
        "tag-to-class mapping, which is code in this file.",
    )
    return p.parse_args()


def _slot_summary(entries: list[list]) -> dict:
    """Compute slot position histogram for a per-class occurrence list."""
    if not entries:
        return {"n": 0}
    slots = [int(e[2]) for e in entries]
    ntoks = [int(e[4]) for e in entries]
    slots_sorted = sorted(slots)

    def _pct(xs, q):
        if not xs:
            return 0.0
        k = (len(xs) - 1) * q
        lo = int(k)
        hi = min(lo + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    mean = sum(slots) / len(slots)
    return {
        "n": len(entries),
        "slot_p10": _pct(slots_sorted, 0.10),
        "slot_p50": _pct(slots_sorted, 0.50),
        "slot_p90": _pct(slots_sorted, 0.90),
        "slot_min": slots_sorted[0],
        "slot_max": slots_sorted[-1],
        "slot_mean": mean,
        "n_tokens_mean": sum(ntoks) / len(ntoks),
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)
    if not image_dir.is_absolute():
        image_dir = REPO_ROOT / image_dir

    specs = load_group_specs(Path(args.anchors_yaml))
    declared = {s.name: set(s.classes) for s in specs}

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    variants = iter_variants(args.variants)
    logger.info(f"anchors_yaml={args.anchors_yaml}")
    logger.info(f"groups={[s.name for s in specs]}")
    logger.info(f"tokenizer={args.tokenizer}  variants={variants}")
    logger.info(f"image_dir={image_dir}")
    logger.info(f"output_dir={out_dir}")

    images = discover_cached_images(str(image_dir))
    if args.max_images:
        images = images[: args.max_images]
    logger.info(f"found {len(images)} cached images in {image_dir}")

    # --- classify every image per group (captions are shuffle-order-invariant)
    image_class: dict[str, dict[str, str | None]] = {s.name: {} for s in specs}
    counts: dict[str, dict[str | None, int]] = {
        s.name: defaultdict(int) for s in specs
    }
    missing_caption = 0
    for img in images:
        txt = image_dir / f"{img.stem}.txt"
        if not txt.exists():
            missing_caption += 1
            continue
        tags = read_caption_tags(txt)
        for s in specs:
            klass = s.classify(tags)
            if klass is not None and klass not in declared[s.name]:
                raise ValueError(
                    f"group '{s.name}' classifier returned {klass!r}, which "
                    f"is not in anchors.yaml classes {sorted(declared[s.name])}"
                )
            image_class[s.name][img.stem] = klass
            counts[s.name][klass] += 1

    for s in specs:
        logger.info(f"{s.name} distribution:")
        for c in s.classes:
            logger.info(f"  {c!r:>20s} : {counts[s.name][c]}")
        logger.info(f"  {'(none)':>20s} : {counts[s.name][None]}")
    if missing_caption:
        logger.warning(f"{missing_caption} images had no .txt caption and were skipped")

    # --- walk cached te safetensors; collect slot vectors + occurrences per group/class
    vectors: dict[str, dict[str, list[torch.Tensor]]] = {
        s.name: defaultdict(list) for s in specs
    }
    occurrences: dict[str, dict[str, list[list]]] = {
        s.name: defaultdict(list) for s in specs
    }
    rt_fail = 0
    ok = 0
    missing_te = 0

    for idx, img in enumerate(images):
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
            tag_index = {t[0]: t for t in result["tags"]}

            for s in specs:
                klass = image_class[s.name].get(img.stem)
                if klass is None:
                    continue
                cons = s.constituents(klass)
                if not cons:
                    # Placeholder class with no tag (e.g. people_count "no_people").
                    occurrences[s.name][klass].append(
                        [img.stem, vi, s.default_slot, s.default_slot, 1]
                    )
                    continue
                present = [tag_index[t] for t in cons if t in tag_index]
                if not present:
                    continue
                present.sort(key=lambda t: t[1])
                earliest = present[0]
                occurrences[s.name][klass].append(
                    [img.stem, vi, int(earliest[1]), int(earliest[2]), int(earliest[3])]
                )
                for tag in present:
                    vectors[s.name][klass].append(
                        emb[int(tag[1])].clone().float().cpu()
                    )

        if (idx + 1) % 200 == 0:
            logger.info(
                f"  processed {idx + 1}/{len(images)}  ok={ok}  rt_fail={rt_fail}"
            )

    logger.info(f"extraction: ok={ok}  rt_fail={rt_fail}  missing_te={missing_te}")

    # --- mean per class -> prototypes; assemble positions
    prototypes: dict[str, torch.Tensor] = {}
    per_class_occ: dict[str, list[list]] = {}
    summary_per_class: dict[str, dict] = {}
    for s in specs:
        for klass in s.classes:
            key = f"{s.proto_key_prefix}{klass}"
            vecs = vectors[s.name][klass]
            if vecs:
                mean_vec = torch.stack(vecs, dim=0).mean(dim=0)
            else:
                mean_vec = torch.zeros(D_EMB, dtype=torch.float32)
            prototypes[key] = mean_vec
            entries = occurrences[s.name].get(klass, [])
            per_class_occ[key] = entries
            summary_per_class[key] = _slot_summary(entries)
            logger.info(
                f"  proto[{key!r}] n_vecs={len(vecs):>5d}  "
                f"norm={mean_vec.norm().item():.2f}"
            )

    positions = {
        "summary": {"per_class": summary_per_class},
        "per_class_occurrences": per_class_occ,
    }

    # --- write
    proto_path = out_dir / "phase2_class_prototypes.safetensors"
    pos_path = out_dir / "phase1_positions.json"

    if args.dry_run:
        logger.info("dry_run — skipping writes")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    proto_bf16 = {k: v.to(torch.bfloat16).contiguous() for k, v in prototypes.items()}
    save_file(proto_bf16, str(proto_path))
    logger.info(f"wrote -> {proto_path}")

    pos_path.write_text(json.dumps(positions))
    logger.info(f"wrote -> {pos_path}")


if __name__ == "__main__":
    main()
