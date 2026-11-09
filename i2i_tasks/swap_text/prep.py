#!/usr/bin/env python3
"""Stage the swap_text i2i task from hitocrawl edition twins.

Every hitocrawl work is the same pages lettered in JA and in a translated
edition (``en/``, ``ko/``). The condition is the JA page; the target is the
same page with its translated lettering pasted where the OCR can read it, and
the caption lists the text read off that target. Six stages, each idempotent;
each module's docstring holds its stage's rules and outputs:

1. **build** (``build.py``, CPU) — register each page pair, diff the lettering
   into regions and groups, write the pair under ``<root>/work/``.
2. **ocr** (``ocr.py``, GPU) — read the twin pages into ``<root>/ocr.json``.
3. **det** (``ocr.py``, GPU) — text blocks on both sides into ``<root>/det.json``.
4. **pool** (``pool.py``, CPU) — gate pages and groups into ``<root>/pool/``.
5. **assemble** (``assemble.py``, CPU) — sample the pool and paste the targets
   into ``<root>/staging/``.
6. **caption** (``caption.py``) — write ``{stem}.txt`` and ``pairs.jsonl``.

``audit.py`` (outside the stages) proofreads the captions against crops of their
lettering and folds the fixes and drops back into ``ocr.json``.

Stems are ``{anchor}_{page}_{lang}``; ``pairs.jsonl`` rows carry ``work`` so a
holdout can split by work (the same work appears in both languages).

    make i2i-stage TASK=swap_text ARGS="--limit 4"
    make daemon-run ARGS="i2i_tasks/swap_text/prep.py --stages ocr det pool assemble caption"
    .venv/bin/python i2i_tasks/swap_text/prep.py --stages assemble caption --sample 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from i2i_tasks.swap_text.assemble import stage_assemble  # noqa: E402
from i2i_tasks.swap_text.build import stage_build  # noqa: E402
from i2i_tasks.swap_text.caption import LANG_TAG, stage_caption  # noqa: E402
from i2i_tasks.swap_text.ocr import stage_det, stage_ocr  # noqa: E402
from i2i_tasks.swap_text.pool import stage_pool  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

STAGES = {
    "build": stage_build,
    "ocr": stage_ocr,
    "det": stage_det,
    "pool": stage_pool,
    "assemble": stage_assemble,
    "caption": stage_caption,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--hito", type=Path, default=Path.home() / "hitocrawl/retrieved")
    ap.add_argument("--root", default="post_image_dataset/i2i/swap_text_v3")
    ap.add_argument("--langs", nargs="+", default=["en", "ko"], choices=list(LANG_TAG))
    ap.add_argument("--works", nargs="*", help="anchor ids; default every work")
    ap.add_argument(
        "--pages",
        type=Path,
        help="JSON list of <lang>/<anchor>/<page> keys: every stage handles only these "
        "(a work built this way is marked built with these pages alone)",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("build")
    g.add_argument("--limit", type=int, default=0, help="works per language (build)")
    g.add_argument("--force", action="store_true", help="rebuild built works")
    g.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="staged size: the MOSS cache's target_size at this resolution, so the "
        "cache encodes the pages as-is; the build diffs at this size too",
    )
    g.add_argument("--min_inliers", type=int, default=50)
    g.add_argument(
        "--scale_tol", type=float, default=0.02, help="max |scale / work median - 1|"
    )
    g.add_argument(
        "--max_chroma_gap",
        type=float,
        default=20.0,
        help="mean HSV saturation gap above which a colour / B&W edition pair drops",
    )
    g = ap.add_argument_group("ocr")
    g.add_argument(
        "--caption_ocr",
        action="store_true",
        help="caption from a second read of each staged target (GPU) instead of "
        "the twin read's lines in the pasted groups",
    )
    g.add_argument(
        "--ocr_procs",
        type=int,
        default=2,
        help="OCR processes sharing the GPU (each loads its own reader)",
    )
    g = ap.add_argument_group("page gates")
    g.add_argument(
        "--max_resid",
        type=float,
        default=8.0,
        help="page mean |src − tgt| outside the lettering above which it drops",
    )
    g.add_argument(
        "--max_uncovered",
        type=float,
        default=0.005,
        help="page share the warped twin leaves uncovered above which it drops",
    )
    g = ap.add_argument_group("group gates")
    g.add_argument(
        "--max_seam",
        type=float,
        default=8.0,
        help="group seam above which a partial pair never pastes it",
    )
    g.add_argument(
        "--max_seam_text",
        type=float,
        default=8.0,
        help="text-only paste: mean |src − twin| on the paper pixels of the ring "
        "around its mask above which the group falls back to the whole region",
    )
    g.add_argument(
        "--max_leftover",
        type=float,
        default=0.01,
        help="group leftover above which a partial pair never pastes it",
    )
    g.add_argument(
        "--det_pad", type=int, default=4, help="text block pad (px) in the pool's mask"
    )
    g.add_argument(
        "--max_bg_resid",
        type=float,
        default=1e9,
        help="page mean |ΔLab| outside every text block above which it leaves the pool",
    )
    g.add_argument(
        "--max_nontext",
        type=float,
        default=8.0,
        help="group median |ΔLab| in its paste mask outside the text blocks above "
        "which it is never pasted",
    )
    g.add_argument(
        "--max_ink_shift",
        type=float,
        default=30.0,
        help="group a/b distance between the replaced and the pasted ink above "
        "which it is never pasted",
    )
    g.add_argument(
        "--flip_contrast",
        type=float,
        default=40.0,
        help="ink lightness vs background on both sides at which a sign flip "
        "(dark ↔ light lettering) keeps a group from being pasted",
    )
    g.add_argument(
        "--paste",
        default="text",
        choices=("text", "region"),
        help="text: paste a group's diff region cut down to the text boxes (both "
        "sides' blocks and the OCR lines) when that passes seam / leftover, else "
        "the whole region; region: always the whole region",
    )
    g.add_argument(
        "--max_free",
        type=int,
        default=3000,
        help="non-text pixels the whole-region fallback of --paste text may carry "
        "(the twin's art) above which the group is never pasted",
    )
    g.add_argument(
        "--retained_ja",
        type=float,
        default=0.05,
        help="share of a kana / CJK line's box changed between the editions (under "
        "its best shift) below which it is JA lettering the twin kept, not a "
        "misread, and does not drop its group",
    )
    g.add_argument(
        "--pad_frac",
        type=float,
        default=0.15,
        help="region box pad (× its long side) an OCR line centre may fall in",
    )
    g = ap.add_argument_group("assemble")
    g.add_argument(
        "--partial_ratio",
        type=float,
        default=0.5,
        help="share of pairs that swap a proper subset of their captioned regions",
    )
    g.add_argument(
        "--sample",
        type=int,
        default=0,
        help="pool pages the assemble stage stages (0: all)",
    )
    g.add_argument(
        "--dilate_frac",
        type=float,
        default=0.003,
        help="paste mask dilation / short edge",
    )
    g.add_argument("--feather", type=float, default=1.5, help="paste edge blur σ (px)")
    a = ap.parse_args()
    a.root = resolve_under_home(a.root)
    a.works = set(a.works or [])
    a.pages = set(json.loads(a.pages.read_text())) if a.pages else set()
    return a


def main() -> None:
    a = parse_args()
    for stage in a.stages:
        STAGES[stage](a)


if __name__ == "__main__":
    main()
