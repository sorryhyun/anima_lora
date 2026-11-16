"""MOSS-format captions for one image set (``CaptionsRequest``).

The Anima caption keeps a full shared tag bag beside its position clauses. The
MOSS format keeps only the head (rating, count, names, series, artist) and the
tags no subject owns in the bag: a tag any subject's crop carries goes into that
subject's clause, repeated across clauses when several carry it, and a tag a
more specific one already says is cut.

Built in a tree of its own, so the Anima captions stay untouched:

1. ``build_tree``: every captioned image of ``src`` symlinked into ``out`` beside
   its caption with the position clauses merged back into the bag
   (``flatten_caption``; text clauses kept), since the position stage only
   rewrites a caption without clauses.
2. anime_tools' position stage over ``out`` with :data:`POSITION_FORMAT` (SAM3 +
   the Anima Tagger: the GPU part, so a daemon job), each subject mask grown by
   ``mask_dilate`` px so a held object stays in the crop.
3. ``drop_implied_tags`` on every caption when ``implied``.
4. ``write_variants``: ``{stem}.variants.txt`` from the finished caption when
   ``variants`` (the Anima variants describe the full bag, so none are copied).

``out/moss_captions.json`` records the request and the anime_tools version.
"""

from __future__ import annotations

import json
import random
import shutil
from importlib.metadata import version
from pathlib import Path

from library.env import resolve_under_home
from library.moss.requests import CaptionsRequest
from library.moss.scan import IMAGE_EXTS

# The position-stage settings that make a caption MOSS-format, shared with
# Anima's caption stages.
from library.preprocess.position_format import POSITION_FORMAT

STAMP = "moss_captions.json"
REPORT_DIR = "_position_report"


def build_tree(src: Path, out: Path) -> int:
    """Symlink every captioned image of ``src`` into ``out`` (same relative
    folders) and write its caption with the position clauses flattened.
    Returns the number of captions written."""
    from anime_tools.captions.position_clauses import flatten_caption

    n = 0
    for image in sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        caption = image.with_suffix(".txt")
        if not caption.is_file():
            continue
        rel = image.relative_to(src)
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(image.resolve())
        text = caption.read_text(encoding="utf-8").strip()
        target.with_suffix(".txt").write_text(
            flatten_caption(text) + "\n", encoding="utf-8"
        )
        n += 1
    return n


def cut_implied(out: Path) -> int:
    """``drop_implied_tags`` on every caption under ``out``; returns tags cut."""
    from anime_tools.captions.implied_tags import drop_implied_tags

    cut = 0
    for caption in sorted(out.rglob("*.txt")):
        if caption.name.endswith((".history.txt", ".variants.txt")):
            continue
        text = caption.read_text(encoding="utf-8").strip()
        result = drop_implied_tags(text)
        if result.dropped_tags:
            caption.write_text(result.text + "\n", encoding="utf-8")
            cut += len(result.dropped_tags)
    return cut


def write_variants(out: Path, num_variants: int, tag_dropout: float) -> int:
    """``{stem}.variants.txt`` beside every caption under ``out`` (anime_tools'
    ``generate_caption_variants``, seeded by the caption's relative path).
    Returns the number of sidecars written."""
    from anime_tools.captions.variants import (
        generate_caption_variants,
        variants_sidecar_path,
        write_variants_sidecar,
    )

    n = 0
    for caption in sorted(out.rglob("*.txt")):
        if caption.name.endswith((".history.txt", ".variants.txt")):
            continue
        random.seed(caption.relative_to(out).as_posix())
        texts = generate_caption_variants(
            caption.read_text(encoding="utf-8").strip(), num_variants, tag_dropout
        )
        write_variants_sidecar(
            variants_sidecar_path(caption),
            [(f"v{i}", text) for i, text in enumerate(texts)],
        )
        n += 1
    return n


def run_captions(req: CaptionsRequest) -> Path:
    """Build ``req``'s MOSS caption tree; returns its folder."""
    if not req.src:
        raise SystemExit("--src is required: the image set with its Anima captions")
    src = resolve_under_home(req.src)
    if not src.is_dir():
        raise SystemExit(f"--src {src} is not a folder")
    out = req.out_dir()
    if out.exists():
        if not req.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to rebuild it")
        if not (out / STAMP).is_file():
            raise SystemExit(
                f"{out} has no {STAMP}: not a MOSS caption tree, not removing it"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True)

    n = build_tree(src, out)
    print(f"{n} captions from {src} → {out}", flush=True)

    from anime_tools.stages.position_captions import run_position
    from anime_tools.stages.requests import PositionRequest

    run_position(
        PositionRequest(
            src=str(out),
            dst=str(out),
            apply=True,
            report_dir=str(out / REPORT_DIR),
            max_clause_tags=req.max_clause_tags,
            max_novel_tags=req.max_novel_tags,
            mask_dilate=req.mask_dilate,
            **POSITION_FORMAT,
        )
    )
    if req.implied:
        print(f"implied: {cut_implied(out)} tags cut", flush=True)
    if req.variants > 0:
        n_var = write_variants(out, req.variants, req.variant_tag_dropout)
        print(f"variants: {n_var} sidecars x {req.variants}", flush=True)

    stamp = {
        "src": str(src),
        "anime_tools": version("anime_tools"),
        "request": {k: v for k, v in vars(req).items()},
        "position_format": POSITION_FORMAT,
    }
    (out / STAMP).write_text(json.dumps(stamp, indent=1, ensure_ascii=False) + "\n")
    print(f"MOSS captions at {out}; cache with --src {out}", flush=True)
    return out
