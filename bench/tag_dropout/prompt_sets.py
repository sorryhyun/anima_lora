#!/usr/bin/env python
"""Build the tag-dropout evaluation prompt sets off ``caption_index.json``.

Phase 0 deliverable (``docs/proposal/tag_dropout_mechanism.md``). Pure data, no
GPU, no torch model — only the frozen tagger vocab (via :class:`TagReadback`) to
type tags into content vs identity. Emits a ``prompt_sets.json`` the render+score
harness (:mod:`bench.tag_dropout.run_eval`) consumes for every arm.

For one artist it produces the four prompt families the proposal pre-registers:

* **sparse** — ``@artist, <count>, [<character>,] solo`` stubs that carry NONE of
  the artist's characteristic *content* tags. The richness probe: does the
  characteristic content appear anyway?
* **detailed** — real captions sampled from the artist's images (the adherence
  probe). Each caption's own content tags are its ``prompted`` set.
* **omission** — the detailed captions with the top-m characteristic tags
  textually removed; the removed tags become the ``omitted`` set (the leakage /
  controllability probe: does removing the tag remove the content?).
* **no_trigger** — the sparse stubs with ``@artist`` stripped (the H4 locus
  probe: is the migration bound to the trigger or the global uncond?).

The **characteristic set** is the artist's general-content signature: general
tags (identity/character/copyright/count/rating/meta all masked out by the
tagger's own category typing — the readback content mask) present in at least
``--freq_floor`` of the artist's images and in at least ``--min_images`` of them,
ranked by frequency.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DEFAULT_INDEX = "post_image_dataset/captions/caption_index.json"
DEFAULT_SRC = "image_dataset"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--artist",
        required=True,
        help="Artist trigger, e.g. '@mignon' (the @ is optional).",
    )
    p.add_argument("--caption_index", default=DEFAULT_INDEX)
    p.add_argument(
        "--src",
        default=DEFAULT_SRC,
        help="Root the caption sidecar paths resolve under.",
    )
    p.add_argument(
        "--model_dir",
        default="models/captioners/anima-tagger-dbv4",
        help="Tagger dir — supplies the vocab/category typing (no forward run).",
    )
    p.add_argument(
        "--freq_floor",
        type=float,
        default=0.25,
        help="A tag joins the characteristic set if present in >= this fraction of the artist's images.",
    )
    p.add_argument(
        "--min_images",
        type=int,
        default=4,
        help="...and in at least this many images (tail-tag calibration guard).",
    )
    p.add_argument(
        "--min_distinct_ratio",
        type=float,
        default=2.0,
        help="...AND artist_freq/corpus_freq >= this (TF-IDF distinctiveness). Generic anime "
        "content base already draws (ratio ~1) has no richness headroom, so it is excluded. "
        "Set 0 to disable (rank by raw artist frequency — the old, generic behavior).",
    )
    p.add_argument(
        "--max_characteristic",
        type=int,
        default=20,
        help="Cap the characteristic set (ranked by distinctiveness, else frequency).",
    )
    p.add_argument(
        "--num_detailed",
        type=int,
        default=12,
        help="Real captions sampled for detailed/omission sets.",
    )
    p.add_argument(
        "--omit_top_m",
        type=int,
        default=4,
        help="Characteristic tags removed per omission prompt.",
    )
    p.add_argument(
        "--max_sparse_chars",
        type=int,
        default=6,
        help="Top characters used to seed sparse stubs.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Deterministic caption sampling."
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output json (default bench/tag_dropout/prompt_sets_<artist>.json).",
    )
    return p.parse_args()


def read_caption(src: Path, rel_txt: str) -> List[str]:
    """Read one caption sidecar → list of trimmed tag strings (empty on missing)."""
    fp = src / rel_txt
    try:
        raw = fp.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def main() -> None:
    args = parse_args()
    trigger = args.artist if args.artist.startswith("@") else "@" + args.artist
    artist_slug = trigger.lstrip("@")

    with open(args.caption_index) as f:
        index = json.load(f)
    stems: List[str] = index["groups"]["artist"].get(trigger, [])
    if not stems:
        raise SystemExit(f"artist {trigger!r} not found in {args.caption_index}")
    image_meta = index["image_meta"]
    src = Path(args.src)

    # Frozen tagger vocab → general-content typing. content_categories=("general",)
    # so caption_indices() keeps ONLY general tags: character/copyright/count/artist
    # (separate categories) and rating/metadata/deprecated all drop out. Same
    # masking the readback primitive uses for the "general_only" set.
    from anime_tools.tagger.tagger import AnimaTagger
    from anime_tools.tagger.readback import TagReadback

    rb_gen = TagReadback(
        AnimaTagger(args.model_dir, device="cpu"), content_categories=("general",)
    )

    def general_tags(tags: List[str]) -> List[str]:
        """Caption tag strings → the general-content tag names the tagger recognizes."""
        idxs = rb_gen.caption_indices(tags)
        return [rb_gen.name_by_idx[i].replace("_", " ") for i in idxs]

    # -- characteristic set: per-artist general-tag frequency ------------------
    n = len(stems)
    freq: Counter = Counter()
    captions: Dict[str, List[str]] = {}
    for stem in stems:
        meta = image_meta.get(stem, {})
        rel = meta.get("path")
        if not rel:
            continue
        tags = read_caption(src, rel)
        if not tags:
            continue
        captions[stem] = tags
        for g in set(general_tags(tags)):  # per-image presence, not multiplicity
            freq[g] += 1

    # Corpus-wide general-tag document frequency → TF-IDF distinctiveness. Generic
    # anime content (high corpus freq) is what base already draws from a bare
    # prompt, so it carries no richness headroom; distinctiveness = the artist's
    # own signature relative to the whole dataset.
    corpus_df: Counter = Counter()
    corpus_n = 0
    if args.min_distinct_ratio > 0:
        for stem, meta in image_meta.items():
            rel = meta.get("path")
            if not rel:
                continue
            tags = read_caption(src, rel)
            if not tags:
                continue
            corpus_n += 1
            for g in set(general_tags(tags)):
                corpus_df[g] += 1

    floor_count = max(args.min_images, int(round(args.freq_floor * n)))
    eligible = [t for t, c in freq.items() if c >= floor_count]
    distinct: Dict[str, float] = {}
    for t in eligible:
        af = freq[t] / n
        gf = (
            (corpus_df[t] / corpus_n)
            if corpus_n and corpus_df.get(t)
            else (1.0 / max(corpus_n, 1))
        )
        distinct[t] = round(af / gf, 2) if gf > 0 else float("inf")
    if args.min_distinct_ratio > 0:
        eligible = [t for t in eligible if distinct[t] >= args.min_distinct_ratio]
        eligible.sort(key=lambda t: distinct[t], reverse=True)
    else:
        eligible.sort(key=lambda t: freq[t], reverse=True)
    characteristic = eligible[: args.max_characteristic]
    char_freq = {t: round(freq[t] / n, 3) for t in characteristic}
    log.info(
        "artist %s: %d images, %d captioned, %d characteristic tags "
        "(floor=%d imgs, min_distinct=%.1fx, corpus_n=%d)",
        trigger,
        n,
        len(captions),
        len(characteristic),
        floor_count,
        args.min_distinct_ratio,
        corpus_n,
    )
    log.info(
        "characteristic: %s",
        ", ".join(f"{t}({char_freq[t]},{distinct[t]}x)" for t in characteristic[:16]),
    )

    # -- identity anchors for sparse stubs -------------------------------------
    count_freq: Counter = Counter()
    char_tag_freq: Counter = Counter()
    for stem in captions:
        meta = image_meta.get(stem, {})
        for c in meta.get("count", []):
            count_freq[c] += 1
        for c in meta.get("character", []):
            char_tag_freq[c] += 1
    top_count = count_freq.most_common(1)[0][0] if count_freq else "1girl"
    top_chars = [c for c, _ in char_tag_freq.most_common(args.max_sparse_chars)]

    # -- prompt families -------------------------------------------------------
    def prompted_general(prompt_tags: List[str]) -> List[str]:
        """The general-content subset of a prompt (what 'adherence' scores)."""
        return general_tags(prompt_tags)

    sparse: List[dict] = []
    # bare stub (identity-free content-wise) + one per top character.
    bare_tags = [trigger, top_count, "solo"]
    sparse.append(
        {
            "id": "sparse_bare",
            "text": ", ".join(bare_tags),
            "tags": bare_tags,
            "prompted": prompted_general(bare_tags),
        }
    )
    for i, ch in enumerate(top_chars):
        t = [trigger, top_count, ch, "solo"]
        sparse.append(
            {
                "id": f"sparse_char{i}",
                "text": ", ".join(t),
                "tags": t,
                "prompted": prompted_general(t),
            }
        )

    # no_trigger = sparse minus the @artist trigger.
    no_trigger: List[dict] = []
    for s in sparse:
        t = [x for x in s["tags"] if x != trigger]
        no_trigger.append(
            {
                "id": s["id"].replace("sparse", "notrig"),
                "text": ", ".join(t),
                "tags": t,
                "prompted": prompted_general(t),
            }
        )

    # detailed = deterministic sample of real captions.
    import random

    rng = random.Random(args.seed)
    stem_pool = sorted(captions.keys())
    rng.shuffle(stem_pool)
    detailed_stems = stem_pool[: args.num_detailed]
    detailed: List[dict] = []
    omission: List[dict] = []
    for i, stem in enumerate(detailed_stems):
        tags = captions[stem]
        text = ", ".join(tags)
        gen = prompted_general(tags)
        detailed.append(
            {"id": f"det{i}", "stem": stem, "text": text, "tags": tags, "prompted": gen}
        )
        # omission: remove up to omit_top_m characteristic tags this caption carries
        # (ranked by artist frequency), keeping identity/order otherwise.
        present_char = [t for t in characteristic if t in set(gen)][: args.omit_top_m]
        if not present_char:
            continue  # nothing to omit -> not a useful leakage probe
        omit_set = set(present_char)
        kept = [t for t in tags if t.strip() not in omit_set]
        omission.append(
            {
                "id": f"omit{i}",
                "stem": stem,
                "text": ", ".join(kept),
                "tags": kept,
                "prompted": prompted_general(kept),
                "omitted": present_char,
            }
        )

    out = {
        "artist": trigger,
        "artist_slug": artist_slug,
        "trigger": trigger,
        "n_images": n,
        "characteristic_set": characteristic,
        "characteristic_freq": char_freq,
        "characteristic_distinctiveness": {t: distinct[t] for t in characteristic},
        "top_count": top_count,
        "top_characters": top_chars,
        "params": {
            "freq_floor": args.freq_floor,
            "min_images": args.min_images,
            "floor_count": floor_count,
            "omit_top_m": args.omit_top_m,
            "num_detailed": args.num_detailed,
            "seed": args.seed,
        },
        "prompts": {
            "sparse": sparse,
            "detailed": detailed,
            "omission": omission,
            "no_trigger": no_trigger,
        },
    }
    out_path = (
        Path(args.out)
        if args.out
        else (REPO_ROOT / "bench" / "tag_dropout" / f"prompt_sets_{artist_slug}.json")
    )
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    log.info(
        "wrote %s  (sparse=%d detailed=%d omission=%d no_trigger=%d)",
        out_path,
        len(sparse),
        len(detailed),
        len(omission),
        len(no_trigger),
    )


if __name__ == "__main__":
    main()
