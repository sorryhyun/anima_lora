#!/usr/bin/env python3
"""Mangafy + cache condition latents (and white-balanced targets + color-only
text) for the colorization EasyControl task. See ``README.md`` for the full
pipeline description; summary of the four idempotent stages:

1. **Mangafy** — screen every color image under ``--src`` to a synthetic B&W
   manga page (``--engine gpu``/``cv2``, both XDoG + algorithmic screentone,
   no downloads), written to ``--staging``. Skips already-staged PNGs unless
   ``--overwrite``.
2. **Encode** — VAE-encode the staged manga images into ``--cond_cache_dir``
   at native size so each cond latent shape matches its target exactly.
3. **Target** — white-balance each color target (see ``wb.py``) and
   VAE-encode into ``--target_cache_dir``. Targets below
   ``--target_drop_sat`` (effectively monochrome) are dropped by deleting
   their cond latents, pairing them out of training.
4. **Text** — re-encode target captions filtered to color tags (plus
   copyright/comic tags by default) into ``--text_cache_dir``, mirroring
   ``--caption_src``. Skipped without ``--qwen3``/``--dit``.

Run from the repo root::

    python easycontrol_adapters/colorization/prep.py            # full dataset
    python easycontrol_adapters/colorization/prep.py --limit 8  # quick QA batch

Via the task runner::

    make easycontrol-staging    EASYADAPTER=colorize   # stage 1 (mangafy)
    make easycontrol-preprocess EASYADAPTER=colorize   # stages 2-4
"""

from __future__ import annotations

import argparse
import os
import tempfile
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from typing import Callable

import numpy as np
import torch
from PIL import Image

from library.preprocess import cache_latents, tqdm_progress
from library.preprocess._dataset import walk_images

# Screener: color RGB uint8 (H,W,3) + per-stem seed → B&W RGB uint8 (H,W,3) same size.
Screener = Callable[[np.ndarray, int], np.ndarray]


def _stable_seed(name: str) -> int:
    """Process-stable per-stem seed (Python's hash() is salted; crc32 isn't)."""
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def _save_png_atomic(arr: np.ndarray, out: Path) -> None:
    """Write ``arr`` to ``out`` atomically — temp file in the same dir + os.replace.

    A direct ``Image.save(out)`` interrupted mid-write leaves a truncated PNG
    that the mangafy skip-check (``out.exists()``) then keeps forever, only
    blowing up later at VAE-decode time. ``os.replace`` is atomic on one
    filesystem, so the final name only ever appears fully written."""
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


def _text_mask_path(mask_dir: Path | None, rel: Path) -> Path | None:
    """Locate the ``{stem}_mask.png`` text mask mirroring ``rel`` under ``mask_dir``.

    Returns ``None`` when masking is off or the page has no mask (most
    pages don't), so the caller leaves the screened result untouched."""
    if mask_dir is None:
        return None
    p = mask_dir / rel.parent / f"{rel.stem}_mask.png"
    return p if p.exists() else None


def _apply_text_mask(
    manga: np.ndarray, img_rgb: np.ndarray, mask_path: Path | None, *, dilate: int
) -> np.ndarray:
    """Paste the source's own grayscale back into the text/speech-bubble regions
    so the screened text stays pixel-exact. No-op without a mask. ``dilate``
    grows the mask a few px to catch anti-aliased glyph fringes."""
    if mask_path is None:
        return manga
    h, w = manga.shape[:2]
    mask = np.asarray(
        Image.open(mask_path).convert("L").resize((w, h), Image.Resampling.NEAREST)
    )
    text = mask < 128  # black = text (merge_masks union polarity)
    if dilate > 0:
        import cv2

        k = np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8)
        text = cv2.dilate(text.astype(np.uint8), k).astype(bool)
    if not text.any():
        return manga
    gray = np.asarray(Image.fromarray(img_rgb).convert("L"))
    out = manga.copy()
    out[text] = np.stack([gray] * 3, axis=-1)[text]
    return out


def _make_screener(engine: str) -> Screener:
    """Resolve ``engine`` to a ``(img, seed) → manga`` callable.

    Imports are deferred so ``cv2`` stays import-light; takes plain
    (picklable) args so it can be rebuilt inside a worker process."""
    if engine == "cv2":
        import cv2

        cv2.setNumThreads(
            1
        )  # one cv2 thread per worker; the pool parallelizes
        from mangafy import mangafy_array

        return lambda img, seed: mangafy_array(img, seed=seed)

    # gpu: same algorithm, but trig screens + XDoG blurs run on CUDA, serial (no pool).
    from mangafy_gpu import mangafy_array_gpu

    return lambda img, seed: mangafy_array_gpu(img, seed=seed)


# ── Worker process state (cv2 engine only) ───────────────────────────────────
# The screener closure isn't picklable, so each worker rebuilds it once in its
# initializer and stashes it (plus constant paths) in a module global.
_WORKER: dict = {}


def _worker_init(
    engine: str,
    src: Path,
    staging: Path,
    overwrite: bool,
    mask_dir: Path | None,
    text_mask_dilate: int,
):
    _WORKER.update(
        screener=_make_screener(engine),
        src=src,
        staging=staging,
        overwrite=overwrite,
        mask_dir=mask_dir,
        text_mask_dilate=text_mask_dilate,
    )


def _worker_process(path_str: str) -> int:
    p = Path(path_str)
    rel = p.relative_to(_WORKER["src"])
    out = (_WORKER["staging"] / rel).with_suffix(".png")
    if out.exists() and not _WORKER["overwrite"]:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    img = np.array(Image.open(p).convert("RGB"))
    manga = _WORKER["screener"](img, _stable_seed(rel.stem))
    manga = _apply_text_mask(
        manga,
        img,
        _text_mask_path(_WORKER["mask_dir"], rel),
        dilate=_WORKER["text_mask_dilate"],
    )
    _save_png_atomic(manga, out)
    return 1


def _mangafy_gpu_overlap(
    images: list[Path],
    src: Path,
    staging: Path,
    *,
    overwrite: bool,
    mask_dir: Path | None,
    text_mask_dilate: int,
    io_workers: int,
    progress: Callable,
) -> tuple[int, int]:
    """GPU mangafy with CPU/IO overlapped against the GPU.

    The screener holds a single GPU stream and stays serial on this thread;
    disk decode and text-mask/PNG write are pure CPU/IO, farmed to thread
    pools to overlap the GPU work. Per-stem seed makes decoding ahead of
    order safe — output is identical."""
    screener = _make_screener("gpu")

    def _decode(p: Path):
        rel = p.relative_to(src)
        out = (staging / rel).with_suffix(".png")
        if out.exists() and not overwrite:
            return rel, out, None  # img=None → skip
        return rel, out, np.array(Image.open(p).convert("RGB"))

    def _finish(manga: np.ndarray, img: np.ndarray, out: Path, rel: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        manga = _apply_text_mask(
            manga, img, _text_mask_path(mask_dir, rel), dilate=text_mask_dilate
        )
        _save_png_atomic(manga, out)

    written = 0
    depth = max(4, io_workers)  # decode look-ahead window (bounds held-decoded memory)
    max_saves = 2 * io_workers  # cap in-flight saves (bounds held manga+img memory)
    it = iter(images)
    decode_q: deque = deque()
    save_q: deque = deque()
    with (
        ThreadPoolExecutor(max_workers=io_workers) as decode_ex,
        ThreadPoolExecutor(max_workers=io_workers) as save_ex,
    ):
        for _ in range(depth):  # prime the decode window
            p = next(it, None)
            if p is None:
                break
            decode_q.append(decode_ex.submit(_decode, p))
        while decode_q:
            rel, out, img = decode_q.popleft().result()
            p = next(it, None)  # refill so a decode is always in flight
            if p is not None:
                decode_q.append(decode_ex.submit(_decode, p))
            if img is None:
                progress(1, detail=f"skip {out.name}")
                continue
            manga = screener(img, _stable_seed(rel.stem))  # GPU, serial on this thread
            save_q.append(save_ex.submit(_finish, manga, img, out, rel))
            written += 1
            progress(1, detail=rel.name)
            while len(save_q) >= max_saves:  # backpressure on the save side
                save_q.popleft().result()
        for f in save_q:  # drain remaining writes
            f.result()
    return len(images), written


def stage_mangafy(
    src: Path,
    staging: Path,
    *,
    engine: str,
    recursive: bool,
    overwrite: bool,
    limit: int | None,
    workers: int,
    mask_dir: Path | None,
    text_mask_dilate: int,
    only_stems: frozenset[str] | None = None,
    exclude_stems: frozenset[str] = frozenset(),
) -> tuple[int, int]:
    images = walk_images(src, recursive=recursive)
    # Tag selection: keep only_stems (None = all), then drop exclude_stems.
    if only_stems is not None or exclude_stems:
        pre = len(images)
        images = [
            p
            for p in images
            if (only_stems is None or p.stem in only_stems)
            and p.stem not in exclude_stems
        ]
        if len(images) != pre:
            print(
                f"Tag filter: keeping {len(images)}/{pre} images for the colorize set"
            )
    if limit:
        images = images[:limit]
    progress = tqdm_progress("Mangafy")
    progress(0, total=len(images))

    # cv2 fans out across processes when there's more than one worker; gpu runs
    # one serial GPU stream (handled below).
    if engine == "cv2" and workers > 1 and len(images) > 1:
        written = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(
                engine,
                src,
                staging,
                overwrite,
                mask_dir,
                text_mask_dilate,
            ),
        ) as ex:
            # small chunks: IPC overhead is negligible vs each image's cv2 work,
            # and fine-grained chunks load-balance variable image sizes well.
            for w in ex.map(_worker_process, [str(p) for p in images], chunksize=2):
                written += w
                progress(1)
        return len(images), written

    # gpu: overlap the CPU/IO decode+mask+write with the serial GPU stream so
    # it isn't starved between images.
    if engine == "gpu" and len(images) > 1:
        return _mangafy_gpu_overlap(
            images,
            src,
            staging,
            overwrite=overwrite,
            mask_dir=mask_dir,
            text_mask_dilate=text_mask_dilate,
            io_workers=max(2, workers),
            progress=progress,
        )

    screener = _make_screener(engine)
    written = 0
    for p in images:
        rel = p.relative_to(src)
        out = (staging / rel).with_suffix(".png")
        if out.exists() and not overwrite:
            progress(1, detail=f"skip {p.name}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        img = np.array(Image.open(p).convert("RGB"))
        manga = screener(img, _stable_seed(rel.stem))
        manga = _apply_text_mask(
            manga, img, _text_mask_path(mask_dir, rel), dilate=text_mask_dilate
        )
        _save_png_atomic(manga, out)
        written += 1
        progress(1, detail=p.name)
    return len(images), written


def _load_encode_vae(vae_path: str, chunk_size: int):
    from library.models import qwen_vae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading VAE from {vae_path} ...")
    vae = qwen_vae.load_vae(
        vae_path,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=chunk_size,
        disable_cache=True,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def _free_vae(vae) -> None:
    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage_encode(
    staging: Path,
    cond_cache_dir: Path,
    *,
    vae_path: str,
    batch_size: int,
    chunk_size: int,
    recursive: bool,
):
    vae = _load_encode_vae(vae_path, chunk_size)
    stats = cache_latents(
        staging,
        vae,
        cache_dir=cond_cache_dir,
        recursive=recursive,
        batch_size=batch_size,
        progress=tqdm_progress("Caching cond latents"),
    )
    _free_vae(vae)
    return stats


def stage_target(
    src: Path,
    staging: Path,
    cond_cache_dir: Path,
    target_cache_dir: Path,
    *,
    vae_path: str,
    batch_size: int,
    chunk_size: int,
    recursive: bool,
    drop_sat: float,
    wb_strength: float,
    wb_max_gain: float,
    workers: int,
    overwrite: bool,
):
    """White-balance + VAE-encode the color targets into ``target_cache_dir``.

    Scoped to the stems the mangafy stage staged. A thumbnail pre-pass
    measures each target's mean HSV saturation; stems below ``drop_sat`` are
    effectively monochrome and get their cond latents **deleted** from
    ``cond_cache_dir``, pairing them out of training. Deletion re-applies on
    every run, so a re-stage always converges to the same pair set.
    """
    import functools

    from wb import apply_whitebalance, mean_saturation

    from library.io.cache import discover_latents_by_stem

    staged_stems = {p.stem for p in walk_images(staging, recursive=recursive)}
    if not staged_stems:
        raise SystemExit(
            f"Staging tree {staging} is empty — run the mangafy staging step first."
        )
    targets = [
        p for p in walk_images(src, recursive=recursive) if p.stem in staged_stems
    ]

    # thumbnail keeps this a ~seconds pass; the saturation statistic is scale-stable
    def _sat(p: Path) -> tuple[str, float]:
        img = Image.open(p).convert("RGB")
        img.thumbnail((256, 256), Image.Resampling.BILINEAR)
        return p.stem, mean_saturation(np.asarray(img))

    progress = tqdm_progress("Target saturation scan")
    progress(0, total=len(targets))
    sats: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max(2, workers)) as ex:
        for stem, sat in ex.map(_sat, targets):
            sats[stem] = sat
            progress(1)

    dropped = {stem for stem, sat in sats.items() if sat < drop_sat}
    kept = frozenset(sats.keys() - dropped)
    print(
        f"Sepia-core drop (mean sat < {drop_sat}): {len(dropped)}/{len(sats)} "
        f"targets dropped, {len(kept)} kept"
    )

    # deleting the cond npz is the exclusion mechanism: the loader only keeps
    # targets with a cached cond latent
    removed = 0
    if dropped:
        by_stem = discover_latents_by_stem(cond_cache_dir)
        for stem in dropped:
            for f in by_stem.get(stem, []):
                f.path.unlink(missing_ok=True)
                removed += 1
        print(f"Removed {removed} cond latent file(s) for dropped stems")

    vae = _load_encode_vae(vae_path, chunk_size)
    stats = cache_latents(
        src,
        vae,
        cache_dir=target_cache_dir,
        recursive=recursive,
        keep_stems=kept,
        image_transform=functools.partial(
            apply_whitebalance, max_gain=wb_max_gain, strength=wb_strength
        ),
        batch_size=batch_size,
        overwrite=overwrite,
        progress=tqdm_progress("Caching WB target latents"),
    )
    _free_vae(vae)
    return stats, len(dropped)


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
    keep_copyright: bool = True,
    keep_comic: bool = False,
    caption_index: str | None = None,
    staging: Path | None = None,
):
    """Cache color-only TE embeddings for the color targets into ``text_cache_dir``.

    Loads Qwen3 + the DiT's LLM adapter, then runs
    ``library.preprocess.cache_text_embeddings`` over ``caption_src`` with
    the color-only caption filter. ``caption_src`` is nested identically to
    ``post_image_dataset/resized`` so TE paths key-match the colorize
    loader's lookup. ``staging`` scopes encoding to the staged stems.

    With ``shuffle_variants > 0`` each cache holds v0 (full color set) plus
    shuffled variants with ``tag_dropout_rate`` of the color tags dropped.

    ``keep_copyright`` places the copyright/series tag first; it rides the
    same tag-dropout as the color tags. ``keep_comic`` adds comic/panel-format
    tags after it, protected from dropout via ``caption_protect_fn``.
    """
    from color_caption import (
        filter_to_colors,
        filter_to_colors_and_protected,
        is_comic_tag,
        load_copyright_tags,
    )

    from library.anima import weights as anima_utils
    from library.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy
    from library.preprocess import cache_text_embeddings

    text_cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Qwen3 text encoder from {qwen3_path} ...")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        qwen3_path, dtype=torch.bfloat16, device=str(device)
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(t5_tokenizer_path)
    print(f"Loading LLM adapter from {dit_path} ...")
    llm_adapter = anima_utils.load_llm_adapter(
        dit_path, dtype=torch.bfloat16, device=str(device)
    )

    tokenize_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer
    )
    encoding_strategy = AnimaTextEncodingStrategy()

    # re-stage the shared uncond sidecar idempotently in case this run is first
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

    # color tags always kept; copyright/comic optionally ride as a leading prefix.
    # only comic is protected from dropout — copyright rides the same tag-dropout.
    caption_protect_fn = None
    copyright_tags: frozenset[str] = frozenset()
    if keep_copyright:
        copyright_tags = (
            load_copyright_tags(caption_index)
            if caption_index
            else load_copyright_tags()
        )
        if not copyright_tags:
            print(
                "Warning: no copyright tags found in caption index "
                f"({caption_index or 'default'}); copyright will not be kept. "
                "Run `make caption-index` or pass --caption_index."
            )

    if keep_copyright or keep_comic:

        def caption_transform(caption: str) -> str:
            return filter_to_colors_and_protected(
                caption,
                copyright_tags,
                keep_copyright=keep_copyright,
                keep_comic=keep_comic,
            )

        def caption_protect_fn(tag: str) -> bool:
            return keep_comic and is_comic_tag(tag)
    else:
        caption_transform = filter_to_colors

    # restrict TE encoding to the staged stems (already tag-filtered by mangafy)
    # rather than re-encoding the whole caption master; None/empty = encode all.
    # --limit only applies to the mangafy/encode stages, not this one.
    keep_stems: set[str] | None = None
    if staging is not None and staging.is_dir():
        keep_stems = {p.stem for p in walk_images(staging, recursive=recursive)}
        if not keep_stems:
            print(
                f"Warning: staging tree {staging} is empty; encoding the whole "
                "caption master (run the mangafy staging step first to scope this)."
            )
            keep_stems = None

    stats = cache_text_embeddings(
        caption_src,
        tokenize_strategy,
        encoding_strategy,
        text_encoder,
        llm_adapter=llm_adapter,
        device=device,
        cache_dir=text_cache_dir,
        recursive=recursive,
        keep_stems=keep_stems,
        batch_size=batch_size,
        caption_transform=caption_transform,
        caption_protect_fn=caption_protect_fn,
        caption_shuffle_variants=shuffle_variants,
        caption_tag_dropout_rate=tag_dropout_rate,
        progress=tqdm_progress("Caching color-only text"),
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
        "--staging", default="post_image_dataset/easycontrol/colorize/staging"
    )
    parser.add_argument(
        "--cond_cache_dir", default="post_image_dataset/easycontrol/colorize/cond"
    )
    parser.add_argument(
        "--target_cache_dir",
        default="post_image_dataset/easycontrol/colorize/target",
        help="white-balanced target-latent cache; the colorize dataset's "
        "latent_cache_dir (TE/PE stay on the shared caches)",
    )
    parser.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    parser.add_argument(
        "--engine",
        choices=["cv2", "gpu"],
        default="gpu",
        help="condition synthesizer (both XDoG+halftone, no downloads): "
        "gpu = on CUDA (fast, serial); cv2 = same math on a CPU ProcessPool",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="cv2 engine: parallel mangafy worker processes (1 = serial). The gpu "
        "engine ignores this (single GPU stream).",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-mangafy staged PNGs that exist; also re-encodes cached "
        "white-balanced target latents (use after changing the WB knobs)",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap #images (QA)")
    parser.add_argument(
        "--only_data_includes",
        nargs="*",
        default=[],
        metavar="TAG",
        help="restrict the colorize set to pages whose caption (in --caption_src) "
        "carries one of these standalone tags (e.g. `genshin_impact`). Empty/omitted "
        "= all pages eligible. Applied before --exclude_data_includes.",
    )
    parser.add_argument(
        "--exclude_data_includes",
        nargs="*",
        default=[],
        metavar="TAG",
        help="drop any page whose caption (in --caption_src) carries one of these "
        "standalone tags (e.g. `comic monochrome`). Net set = only_data_includes − "
        "exclude_data_includes. Filtered pages get no condition synthesized, so the "
        "train-time loader pairs them out automatically.",
    )
    parser.add_argument(
        "--mask_dir",
        default="post_image_dataset/masks",
        help="text/speech-bubble masks (black=text); the source grayscale is pasted "
        "back over these regions so the screened text stays pixel-exact",
    )
    parser.add_argument(
        "--skip_text_mask",
        action="store_true",
        help="don't paste source text back (screen the whole page, garbled text and all)",
    )
    parser.add_argument(
        "--text_mask_dilate",
        type=int,
        default=0,
        help="px to grow the text mask before pasting (catches anti-aliased glyph fringes)",
    )
    parser.add_argument(
        "--skip_mangafy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip the mangafy condition-synthesis stage (it's the "
        "`easycontrol-staging EASYADAPTER=colorize` step); --no-skip_mangafy "
        "re-stages inline during preprocess",
    )
    parser.add_argument("--skip_encode", action="store_true")
    parser.add_argument(
        "--skip_target",
        action="store_true",
        help="skip the white-balanced target-latent stage",
    )
    parser.add_argument(
        "--skip_text", action="store_true", help="skip color-only TE stage"
    )
    # ── white-balanced target stage ──────────────────────────────────────────
    parser.add_argument(
        "--target_drop_sat",
        type=float,
        default=0.10,
        help="drop targets whose mean HSV saturation is below this (effectively "
        "monochrome/sepia-core — white-balancing them yields flat gray); their "
        "cond latents are deleted so the loader pairs them out",
    )
    parser.add_argument(
        "--target_wb_strength",
        type=float,
        default=1.0,
        help="white-balance strength (gains ** strength; <1 = partial correction)",
    )
    parser.add_argument(
        "--target_wb_max_gain",
        type=float,
        default=1.35,
        help="per-channel white-balance gain clamp (larger casts are treated as "
        "an intentional palette and only partially corrected)",
    )
    # ── color-only text stage ────────────────────────────────────────────────
    parser.add_argument(
        "--caption_src",
        default="image_dataset",
        help="caption master (.txt sidecars), nested like post_image_dataset/resized",
    )
    parser.add_argument(
        "--text_cache_dir",
        default="post_image_dataset/easycontrol/colorize/text",
        help="where color-only TE caches go; the colorize dataset's text_cache_dir",
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
        "--text_batch_size", type=int, default=16, help="text stage encode batch"
    )
    parser.add_argument(
        "--text_shuffle_variants",
        type=int,
        default=3,
        help="color-caption variants per image (v0=full colors, v1+=shuffled "
        "+ tag-dropped). 0 = single full-color caption.",
    )
    parser.add_argument(
        "--text_tag_dropout_rate",
        type=float,
        default=0.8,
        help="per-color-tag dropout in v1+ variants — trains robustness to "
        "partial color prompts. Ignored when --text_shuffle_variants <= 0.",
    )
    parser.add_argument(
        "--text_keep_copyright",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="carry the copyright/series tag alongside the color tags (placed "
        "first). Kept but NOT protected — it rides the same tag-dropout as the "
        "color tags (only comic is protected). --no-text_keep_copyright drops "
        "it entirely.",
    )
    parser.add_argument(
        "--text_keep_comic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep comic/panel-format tags (comic, 4koma, …) alongside the color "
        "tags (placed in the protected prefix, immune to tag-dropout) so the "
        "adapter learns the `comic` tag. On by default; "
        "--no-text_keep_comic drops them.",
    )
    parser.add_argument(
        "--caption_index",
        default=None,
        help="caption index JSON with a groups.copyright vocab, used to identify "
        "copyright tags. Defaults to post_image_dataset/captions/caption_index.json.",
    )
    args = parser.parse_args()

    src = Path(args.src)
    staging = Path(args.staging)
    cond_cache_dir = Path(args.cond_cache_dir)

    # net set = only_data_includes - exclude_data_includes (see --help above); a
    # filtered-out page gets no cond latent, so the loader pairs it out downstream.
    from library.datasets.dreambooth import stems_with_any_tag

    only_stems = (
        stems_with_any_tag(args.caption_src, args.only_data_includes)
        if args.only_data_includes
        else None
    )
    exclude_stems = stems_with_any_tag(args.caption_src, args.exclude_data_includes)

    if not args.skip_mangafy:
        seen, written = stage_mangafy(
            src,
            staging,
            engine=args.engine,
            recursive=args.recursive,
            overwrite=args.overwrite,
            limit=args.limit,
            workers=args.workers,
            mask_dir=None if args.skip_text_mask else Path(args.mask_dir),
            text_mask_dilate=args.text_mask_dilate,
            only_stems=only_stems,
            exclude_stems=exclude_stems,
        )
        print(
            f"\nMangafy ({args.engine}): "
            f"{written} written, {seen - written} skipped/seen={seen}"
        )

    if not args.skip_encode:
        stats = stage_encode(
            staging,
            cond_cache_dir,
            vae_path=args.vae,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            recursive=args.recursive,
        )
        print(
            f"\nCond latent caching complete: {stats.written} cached, "
            f"{stats.skipped} skipped (already existed)"
        )

    if not args.skip_target:
        tgt_stats, n_dropped = stage_target(
            src,
            staging,
            cond_cache_dir,
            Path(args.target_cache_dir),
            vae_path=args.vae,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            recursive=args.recursive,
            drop_sat=args.target_drop_sat,
            wb_strength=args.target_wb_strength,
            wb_max_gain=args.target_wb_max_gain,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        print(
            f"\nWB target latent caching complete: {tgt_stats.written} cached, "
            f"{tgt_stats.skipped} skipped (already existed), "
            f"{n_dropped} sepia-core targets dropped"
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
            keep_copyright=args.text_keep_copyright,
            keep_comic=args.text_keep_comic,
            caption_index=args.caption_index,
            staging=staging,
        )
        print(
            f"\nColor-only text caching complete: {tstats.written} cached, "
            f"{tstats.skipped} skipped (already existed)"
        )


if __name__ == "__main__":
    main()
