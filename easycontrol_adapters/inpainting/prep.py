#!/usr/bin/env python3
"""Mask + cache condition latents (and shuffled-caption text) for the inpaint
EasyControl task.

Three idempotent stages (simpler than colorize — the text stage applies no caption
filter, just full captions with shuffle + tag-dropout variants):

1. **Mask** — walk every image under ``--src`` (the existing resized training
   images, the inpaint *targets*), punch one seeded free-form hole per image
   (``mask_image.mask_array``: LaMa-style brush strokes + rects, gray-filled),
   and write the result to ``--staging`` mirroring the source subpath. One mask
   per image, seeded per-stem (``_stable_seed = crc32(stem)``), so holes vary in
   position/shape/size *across* the dataset but each image's hole is fixed (the
   cond cache is keyed ``{stem}_{WxH}_anima.npz`` — one cond per image). Pure
   numpy/cv2, fanned out across a ProcessPool. Skips already-staged PNGs unless
   ``--overwrite``.

2. **Encode** — VAE-encode the masked images into ``--cond_cache_dir`` via
   ``library.preprocess.cache_latents`` (same ``{stem}_{WxH}_anima.npz`` format
   as the target latent cache), at each image's **native size** so the cond
   latent shape matches its target latent exactly. Skips cached resolutions.

3. **Text** — re-encode the *target* captions (no filter, full caption) into
   ``--text_cache_dir`` with ``--text_shuffle_variants`` shuffled + tag-dropped
   variants (``--text_tag_dropout_rate``; the ``@artist`` prefix is auto-protected),
   so the model learns to fill from a partial caption as well as the full spec.
   Reads ``.txt`` from ``--caption_src`` and scopes to the masked staging tree's
   stems. Skip with ``--skip_text`` to reuse the shared LoRA TE cache verbatim.

The cond cache is paired with the original-image target at train time by the
loader's ``cond_cache_dir`` subset knob (stem-matched); the variant text by
``text_cache_dir``. Run from the repo root::

    python easycontrol_adapters/inpainting/prep.py            # full dataset (all 3 stages)
    python easycontrol_adapters/inpainting/prep.py --limit 8  # quick QA batch

Via the task runner the stages are split across two targets (knobs come from the
``[staging]`` / ``[preprocess]`` tables of ``configs/easycontrol/inpaint.toml``)::

    make easycontrol-staging    EASYADAPTER=inpaint   # stage 1 (mask)
    make easycontrol-preprocess EASYADAPTER=inpaint   # stages 2-3 (encode + text)
"""

from __future__ import annotations

import argparse
import os
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from library.preprocess import cache_latents, tqdm_progress
from library.preprocess._dataset import walk_images

from mask_image import mask_array  # sibling module (script dir is sys.path[0])


def _stable_seed(name: str) -> int:
    """Process-stable per-stem seed (Python's hash() is salted; crc32 isn't)."""
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def _save_png_atomic(arr: np.ndarray, out: Path) -> None:
    """Write ``arr`` to ``out`` atomically — temp file in the same dir + os.replace.

    A direct ``Image.save(out)`` interrupted mid-write leaves a truncated PNG that
    the ``out.exists()`` skip-check then keeps forever, blowing up only later at
    VAE decode. Writing to a unique temp in the same directory and ``os.replace``-ing
    it in means the final name only ever appears fully written (atomic on one fs);
    on any failure the temp is removed."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp.png")
    os.close(fd)
    try:
        Image.fromarray(arr).save(tmp)
        os.replace(tmp, out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Worker process state ──────────────────────────────────────────────────────
# The mask math is pure numpy/cv2, so it scales ~linearly with cores; the seeded
# per-stem hole keeps every worker bit-identical to the serial path.
_WORKER: dict = {}


def _worker_init(
    src: Path, staging: Path, overwrite: bool, min_frac: float, max_frac: float
):
    _WORKER.update(
        src=src,
        staging=staging,
        overwrite=overwrite,
        min_frac=min_frac,
        max_frac=max_frac,
    )


def _worker_process(path_str: str) -> int:
    p = Path(path_str)
    rel = p.relative_to(_WORKER["src"])
    out = (_WORKER["staging"] / rel).with_suffix(".png")
    if out.exists() and not _WORKER["overwrite"]:
        return 0
    img = np.array(Image.open(p).convert("RGB"))
    masked = mask_array(
        img,
        _stable_seed(rel.stem),
        min_frac=_WORKER["min_frac"],
        max_frac=_WORKER["max_frac"],
    )
    _save_png_atomic(masked, out)
    return 1


def stage_mask(
    src: Path,
    staging: Path,
    *,
    recursive: bool,
    overwrite: bool,
    limit: int | None,
    workers: int,
    min_frac: float,
    max_frac: float,
) -> tuple[int, int]:
    images = walk_images(src, recursive=recursive)
    if limit:
        images = images[:limit]
    progress = tqdm_progress("Masking")
    progress(0, total=len(images))

    if workers > 1 and len(images) > 1:
        written = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(src, staging, overwrite, min_frac, max_frac),
        ) as ex:
            for w in ex.map(_worker_process, [str(p) for p in images], chunksize=4):
                written += w
                progress(1)
        return len(images), written

    written = 0
    for p in images:
        rel = p.relative_to(src)
        out = (staging / rel).with_suffix(".png")
        if out.exists() and not overwrite:
            progress(1, detail=f"skip {p.name}")
            continue
        img = np.array(Image.open(p).convert("RGB"))
        masked = mask_array(
            img, _stable_seed(rel.stem), min_frac=min_frac, max_frac=max_frac
        )
        _save_png_atomic(masked, out)
        written += 1
        progress(1, detail=p.name)
    return len(images), written


def stage_encode(
    staging: Path,
    cond_cache_dir: Path,
    *,
    vae_path: str,
    batch_size: int,
    chunk_size: int,
    recursive: bool,
    vae_2d: bool = True,
):
    """VAE-encode the masked staging tree into ``cond_cache_dir`` at native size so
    cond latent shape == target latent shape.

    ``vae_2d`` (default ON, matching ``scripts/preprocess/cache_latents.py``) folds
    the causal Conv3d VAE into 2D convs — ~2x faster and, crucially, the SAME path
    the *target* latents in post_image_dataset/lora/ were cached with, so cond and
    target stay numerically consistent. Pass ``--no_vae_2d`` for the stock 3D VAE."""
    from library.models import qwen_vae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading VAE from {vae_path} (2d_fold={vae_2d}) ...")
    vae = qwen_vae.load_vae(
        vae_path,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=chunk_size,
        disable_cache=True,
        vae_2d=vae_2d,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()

    stats = cache_latents(
        staging,
        vae,
        cache_dir=cond_cache_dir,
        recursive=recursive,
        batch_size=batch_size,
        progress=tqdm_progress("Caching cond latents"),
    )

    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def stage_text(
    caption_src: Path,
    text_cache_dir: Path,
    *,
    qwen3_path: str,
    dit_path: str,
    t5_tokenizer_path: str | None,
    batch_size: int,
    recursive: bool,
    shuffle_variants: int,
    tag_dropout_rate: float,
    staging: Path | None = None,
    vocab_pack: str | None = None,
    adapter_outputs: bool = True,
):
    """Cache **full-caption** TE embeddings with shuffle + tag-dropout variants.

    Unlike colorize's text stage there is NO caption filter — inpaint reuses the
    full caption (the hole content is described by it). With ``shuffle_variants > 0``
    each cache holds v0 (the verbatim caption) plus shuffled variants with
    ``tag_dropout_rate`` of the tags dropped — teaching the model to fill from a
    *partial* caption (mostly context-driven) as well as the full spec. The
    ``@artist`` prefix is auto-protected from both shuffle and dropout by
    ``caption_variants`` (no ``caption_transform``/``protect_fn`` needed here).

    When ``staging`` is given (the masked cond tree), encoding is scoped to the
    stems present there so the TE cache mirrors the cond cache. ``None`` / absent /
    empty staging = encode every caption.

    ``vocab_pack`` follows ``scripts/preprocess/cache_text_embeddings.py``: ``None``
    = the config default (``configs/base.toml`` ``vocab_pack``), ``""`` = off. With
    a pack the CJK spans of a caption route onto the pack rows, the rows are hooked
    onto the LLM adapter for the crossattn cache, and every cache is stamped with
    the pack id; EN-only captions are bit-exact either way.

    ``adapter_outputs=False`` writes the pre-adapter layout (Qwen
    ``prompt_embeds`` + T5 ids) instead of ``crossattn_emb`` — for runs that
    train the llm_adapter and so need it live (``cache_llm_adapter_outputs=false``).
    """
    from library.anima import weights as anima_utils
    from library.anima.strategy import AnimaTextEncodingStrategy
    from library.anima.vocab_pack import load_vocab_pack, make_tokenize_strategy
    from library.preprocess import cache_text_embeddings

    text_cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Qwen3 text encoder from {qwen3_path} ...")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        qwen3_path, dtype=torch.bfloat16, device=str(device)
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(t5_tokenizer_path)
    if vocab_pack is None:
        from library.anima.vocab_pack import default_vocab_pack

        vocab_pack = default_vocab_pack()
    pack = load_vocab_pack(vocab_pack)
    if pack is not None:
        print(f"Vocab pack {pack.name}: {pack.rows} ext rows")
    print(f"Loading LLM adapter from {dit_path} ...")
    llm_adapter = anima_utils.load_llm_adapter(
        dit_path, dtype=torch.bfloat16, device=str(device), vocab_pack=pack
    )

    tokenize_strategy = make_tokenize_strategy(
        pack, qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer
    )
    encoding_strategy = AnimaTextEncodingStrategy()

    # The loader reuses the shared T5("") uncond sidecar for caption dropout, staged
    # by `make preprocess`; re-stage idempotently in case this run touches it first.
    from library.preprocess.uncond import (
        DEFAULT_UNCOND_DIR,
        stage_uncond_sidecar_with_models,
    )

    stage_uncond_sidecar_with_models(
        DEFAULT_UNCOND_DIR,
        text_encoder,
        tokenize_strategy,
        encoding_strategy,
        llm_adapter,
        device=device,
        overwrite=False,
    )

    # Scope to the masked staging tree's stems (mirrors the cond cache) when present.
    keep_stems: set[str] | None = None
    if staging is not None and staging.is_dir():
        keep_stems = {p.stem for p in walk_images(staging, recursive=recursive)}
        if not keep_stems:
            print(
                f"Warning: staging tree {staging} is empty; encoding the whole "
                "caption master (run the mask staging step first to scope this)."
            )
            keep_stems = None

    stats = cache_text_embeddings(
        caption_src,
        tokenize_strategy,
        encoding_strategy,
        text_encoder,
        llm_adapter=llm_adapter if adapter_outputs else None,
        device=device,
        cache_dir=text_cache_dir,
        recursive=recursive,
        keep_stems=keep_stems,
        batch_size=batch_size,
        caption_shuffle_variants=shuffle_variants,
        caption_tag_dropout_rate=tag_dropout_rate,
        progress=tqdm_progress("Caching caption text"),
    )

    text_encoder.to("cpu")
    del text_encoder, llm_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="post_image_dataset/resized")
    parser.add_argument(
        "--staging", default="post_image_dataset/easycontrol/inpaint/staging"
    )
    parser.add_argument(
        "--cond_cache_dir", default="post_image_dataset/easycontrol/inpaint/cond"
    )
    parser.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    # Match scripts/preprocess/cache_latents.py: 2D fold ON by default so cond
    # latents are encoded by the SAME VAE path as the target latents in
    # post_image_dataset/lora/ (and ~2x faster). --no_vae_2d for the stock 3D VAE.
    parser.add_argument(
        "--vae_2d",
        "--qwen_image_vae_2d",
        dest="vae_2d",
        action="store_true",
        default=True,
        help="fold the causal Conv3d VAE into 2D convs (image-only). Default ON.",
    )
    parser.add_argument(
        "--no_vae_2d",
        "--qwen_image_vae_3d",
        dest="vae_2d",
        action="store_false",
        help="use the stock 3D causal-Conv3d VAE instead of the 2D fold.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="parallel mask worker processes (1 = serial)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="re-mask staged PNGs that exist"
    )
    parser.add_argument("--limit", type=int, default=None, help="cap #images (QA)")
    parser.add_argument(
        "--min_frac", type=float, default=0.10, help="min hole coverage fraction"
    )
    parser.add_argument(
        "--max_frac", type=float, default=0.40, help="max hole coverage fraction"
    )
    parser.add_argument(
        "--skip_mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip the mask stage (it's the `easycontrol-staging EASYADAPTER=inpaint` "
        "step); --no-skip_mask re-stages inline during preprocess",
    )
    parser.add_argument("--skip_encode", action="store_true")
    parser.add_argument(
        "--skip_text",
        action="store_true",
        help="skip the caption TE stage (reuse the shared LoRA TE cache verbatim)",
    )
    # ── caption text stage (full caption + shuffle/tag-dropout variants) ─────────
    parser.add_argument(
        "--caption_src",
        default="image_dataset",
        help="caption master (.txt sidecars), nested like post_image_dataset/resized",
    )
    parser.add_argument(
        "--text_cache_dir",
        default="post_image_dataset/easycontrol/inpaint/text",
        help="where caption TE caches go; the inpaint dataset's text_cache_dir",
    )
    parser.add_argument(
        "--qwen3", default="models/text_encoders/qwen_3_06b_base.safetensors"
    )
    parser.add_argument(
        "--dit",
        default="models/diffusion_models/anima-base-v1.0.safetensors",
        help="DiT for the LLM adapter (produces crossattn_emb)",
    )
    parser.add_argument("--t5_tokenizer_path", default=None)
    parser.add_argument(
        "--vocab_pack",
        type=str,
        default=None,
        help="CJK vocab pack path prefix (configs/base.toml `vocab_pack`; '' = off). "
        "Routes CJK caption spans onto the pack rows, hooks the rows onto the "
        "LLM adapter for crossattn caching, and stamps the pack id into every "
        "cache written. EN-only captions are bit-exact either way.",
    )
    parser.add_argument(
        "--text_batch_size", type=int, default=16, help="text stage encode batch"
    )
    parser.add_argument(
        "--text_shuffle_variants",
        type=int,
        default=4,
        help="caption variants per image (v0=verbatim, v1+=shuffled + tag-dropped). "
        "0 = single verbatim caption.",
    )
    parser.add_argument(
        "--text_tag_dropout_rate",
        type=float,
        default=0.2,
        help="per-tag dropout in v1+ variants (the @artist prefix is auto-protected). "
        "Ignored when --text_shuffle_variants <= 0.",
    )
    args = parser.parse_args()

    src = Path(args.src)
    staging = Path(args.staging)
    cond_cache_dir = Path(args.cond_cache_dir)

    if not args.skip_mask:
        seen, written = stage_mask(
            src,
            staging,
            recursive=args.recursive,
            overwrite=args.overwrite,
            limit=args.limit,
            workers=args.workers,
            min_frac=args.min_frac,
            max_frac=args.max_frac,
        )
        print(f"\nMask: {written} written, {seen - written} skipped/seen={seen}")

    if not args.skip_encode:
        stats = stage_encode(
            staging,
            cond_cache_dir,
            vae_path=args.vae,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            recursive=args.recursive,
            vae_2d=args.vae_2d,
        )
        print(
            f"\nCond latent caching complete: {stats.written} cached, "
            f"{stats.skipped} skipped (already existed)"
        )

    if not args.skip_text:
        tstats = stage_text(
            Path(args.caption_src),
            Path(args.text_cache_dir),
            qwen3_path=args.qwen3,
            dit_path=args.dit,
            t5_tokenizer_path=args.t5_tokenizer_path,
            batch_size=args.text_batch_size,
            recursive=args.recursive,
            shuffle_variants=args.text_shuffle_variants,
            tag_dropout_rate=args.text_tag_dropout_rate,
            staging=staging,
            vocab_pack=args.vocab_pack,
        )
        print(
            f"\nCaption text caching complete: {tstats.written} cached, "
            f"{tstats.skipped} skipped (already existed)"
        )


if __name__ == "__main__":
    main()
