#!/usr/bin/env python3
"""plan_render S1 — trainer prep for the bubble-fill EasyControl task.

A thin sibling of ``easycontrol_adapters/inpainting/prep.py`` over the tree
``cut.py`` wrote (``post_image_dataset/render/<ed>/``). Three idempotent
stages::

    mask    resized/ + boxes.jsonl → staging/    the holed cond (boxes gray-filled,
                                                 mask_image.GRAY; heldout/ → heldout_staging/)
    encode  resized/ → lora/  and  staging/ → cond/   VAE latents at native size,
                                                 library.preprocess.cache_latents
    text    resized/*.txt → text/                one verbatim caption per sample
                                                 (no shuffle / dropout variants — the
                                                 transcript is never dropped, decision 5),
                                                 through the inpaint prep's pack-aware
                                                 stage_text (``--vocab_pack``: "" = stock,
                                                 a path = that pack, omitted = config default)

Only ``encode`` and ``text`` touch the GPU — run those through the daemon::

    python project/cjk_aware_anima_dit/render/prep_render.py mask --edition en
    make daemon-run ARGS="project/cjk_aware_anima_dit/render/prep_render.py encode text --edition en --vocab_pack ''"

Via the task runner: ``make easycontrol-staging EASYADAPTER=render_en`` (cut +
mask) and ``make easycontrol-preprocess EASYADAPTER=render_en`` (encode + text).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
INPAINT = REPO / "easycontrol_adapters" / "inpainting"
sys.path.insert(0, str(INPAINT))  # mask_image, the inpaint prep's sibling import


def _load_inpaint_prep():
    spec = importlib.util.spec_from_file_location("inpaint_prep", INPAINT / "prep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hole(png: Path, out: Path, holes: list, gray: int) -> None:
    """Gray-fill the holes and write atomically (temp + os.replace, the inpaint
    prep's rule: a half-written PNG must never survive an interrupt)."""
    arr = np.array(Image.open(png).convert("RGB"))
    for x0, y0, x1, y1 in holes:
        arr[y0:y1, x0:x1] = gray
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.png")
    try:
        Image.fromarray(arr).save(tmp)
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _mask_one(job) -> int:
    png, out, holes, gray, overwrite = job
    out = Path(out)
    if out.exists() and not overwrite:
        return 0
    _hole(Path(png), out, holes, gray)
    return 1


def stage_mask(base: Path, *, overwrite: bool, workers: int) -> None:
    from mask_image import GRAY

    for src, dst in (("resized", "staging"), ("heldout", "heldout_staging")):
        rows_path = base / src / "boxes.jsonl"
        if not rows_path.is_file():
            print(f"[mask] {rows_path} missing — run cut.py first", flush=True)
            continue
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        jobs = [
            (
                str(base / src / r["rel"]),
                str(base / dst / r["rel"]),
                r["holes"],
                GRAY,
                overwrite,
            )
            for r in rows
        ]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            written = sum(ex.map(_mask_one, jobs, chunksize=8))
        print(
            f"[mask] {src} → {dst}: {written} written, {len(jobs) - written} skipped",
            flush=True,
        )


def stage_encode(base: Path, a) -> None:
    inpaint = _load_inpaint_prep()
    for src, dst in (("resized", "lora"), ("staging", "cond")):
        stats = inpaint.stage_encode(
            base / src,
            base / dst,
            vae_path=a.vae,
            batch_size=a.batch_size,
            chunk_size=a.chunk_size,
            recursive=True,
        )
        print(
            f"[encode] {src} → {dst}: {stats.written} cached, {stats.skipped} skipped",
            flush=True,
        )


def stage_text(base: Path, a) -> None:
    inpaint = _load_inpaint_prep()
    stats = inpaint.stage_text(
        base / "resized",
        base / "text",
        qwen3_path=a.qwen3,
        dit_path=a.dit,
        t5_tokenizer_path=None,
        batch_size=a.text_batch_size,
        recursive=True,
        shuffle_variants=0,
        tag_dropout_rate=0.0,
        staging=base / "staging",
        vocab_pack=a.vocab_pack,
    )
    print(f"[text] {stats.written} cached, {stats.skipped} skipped", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stages", nargs="+", choices=("mask", "encode", "text"))
    ap.add_argument("--edition", default="en", choices=("en", "ja", "ko"))
    ap.add_argument("--root", default="post_image_dataset/render")
    ap.add_argument(
        "--overwrite", action="store_true", help="mask: re-hole staged PNGs"
    )
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=64)
    ap.add_argument(
        "--qwen3", default="models/text_encoders/qwen_3_06b_base.safetensors"
    )
    ap.add_argument(
        "--dit", default="models/diffusion_models/anima-base-v1.0.safetensors"
    )
    ap.add_argument("--text_batch_size", type=int, default=16)
    ap.add_argument(
        "--vocab_pack",
        default=None,
        help="'' = stock tokenizer (the EN arm), a pack path prefix (JA-SHIP / JA-RAND), "
        "omitted = configs/base.toml default",
    )
    a = ap.parse_args()
    base = REPO / a.root / a.edition
    if "mask" in a.stages:
        stage_mask(base, overwrite=a.overwrite, workers=a.workers)
    if "encode" in a.stages:
        stage_encode(base, a)
    if "text" in a.stages:
        stage_text(base, a)


if __name__ == "__main__":
    main()
