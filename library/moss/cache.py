"""Precache Qwen3.5-9B text embeddings and FLUX.2 VAE latents for a training folder.

Training never loads the TE (16.8 GB) or the VAE: this runs TE → free → VAE and
leaves two files per image in one flat folder.

    {stem}.moss_te.safetensors
        embeds  (S, 4096) bf16: ``last_hidden_state`` at the caption's true length
        embeds_plain  (S', 4096) bf16: under ``layout_json`` only, the plain Anima
                caption the layout came from (training swaps it in at
                ``plain_caption_rate``)
        embeds_v1, embeds_v2, …  (S_k − P, 4096) bf16: optional tag-dropped
                caption variants (``variant_entries`` / ``add_variants``), swapped
                in at ``caption_variant_rate``. Only the rows after the first
                ``P = variant_prefix`` (metadata) are stored: the TE is causal and
                the variants share ``embeds``' leading rows
        metadata: caption, caption_sha256, caption_format, source_sha256, source,
                  n_text, max_text_tokens, encoder (+ plain_caption_sha256,
                  n_text_plain; + caption_v{k}, variant_prefix with variants)
    {stem}_{W}x{H}.moss_latent.safetensors
        latent  (128, H/16, W/16) fp32: packed 2×2 and BN-normalised, exactly what
                the DiT consumes (``model.encode``, the inverse of the decode)
        metadata: width, height, source, source_size, source_mtime, vae_dtype, resampled

    {stem}.moss_ref.safetensors  (``--pairs`` only)
        latent  (128, h, w) fp32: the pair's condition image, fitted to the
                target's ``W×H`` and then to 1/``ref_scale`` of its sides
        metadata: width, height, target, ref_scale, condition, condition_mtime

``W×H`` is the pixel size the latent was encoded at. Text is never padded: the
DiT slices each context to its true length and the image's rope frame coordinate
equals that length, so a padded cache would train an offset inference never sees.

Invalidation: a TE file is current while its ``caption_sha256``, truncation
length, ``encoder`` and (under ``layout_json``) ``plain_caption_sha256`` match, so
a caption edit re-encodes that caption only. A re-encode keeps the file's caption
variants while its caption is unchanged; after a caption edit they are stale and
dropped, with a warning. A latent file is current while it is newer than its
source image; a latent at a different size for the same stem is deleted when the
new one is written. ``--overwrite`` re-encodes everything.

Sizing (``target_size``): native aspect, both sides multiples of 16, area at most
``resolution²``, **never upscaled**. An image whose sides are already multiples
of 16 and whose area is within ``AREA_TOLERANCE`` of the cap is encoded as-is (the
Anima resized tree is already on a 16-px grid at ~1.05 MP, so it is not resampled
twice). Otherwise it is LANCZOS-resized to cover the target and centre-cropped by
the < 16 px the rounding leaves. Transparent images are composited onto white.

Captions: the ``.txt`` beside each image, stripped, mapped onto MOSS's
layout-JSON prompt by ``anime_tools.captions.layout_json`` (the default
``--caption_format layout_json``; ``caption`` / ``caption_sha256`` are the JSON;
a ``{stem}.layout.txt`` published by anime_tools' Export is read instead) or
verbatim with ``--caption_format text``; ``--caption_format fixed`` encodes
``--fixed_caption`` for every image instead. ``source_sha256`` is always the ``.txt``.
A stem that appears in several subfolders is prefixed with its relative folder
(``sub__dir__stem``). A caption in the Anima resized tree with no PNG beside it
pairs with its ``image_dataset/`` original, minus the excluded ones
(``scan.original_tree``).

    make daemon-run ARGS="scripts/moss/cache.py \\
        --src 'post_image_dataset/resized/channel_(caststation)' \\
        --out 'post_image_dataset/moss/channel_(caststation)'"
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from library.env import resolve_under_home
from library.moss.caching.images import (  # noqa: F401  (this module is their import path)
    AREA_TOLERANCE,
    STRIDE,
    fit,
    load_rgb,
    prefetch,
    ref_size,
    target_size,
    to_pixels,
)
from library.moss.caching.items import (  # noqa: F401
    CacheItem,
    add_variants,
    carried_variants,
    list_items,
    load_item,
    load_plain,
    load_ref,
    load_variant,
    read_te,
    variant_entries,
)
from library.moss.caching.sources import (  # noqa: F401
    Pair,
    caption_converter,
    find_pairs,
    is_layout_json,
    manifest_pairs,
    tag_knowledge_base,
)
from library.moss.scan import (
    LATENT_SUFFIX,
    REF_SUFFIX,
    TE_SUFFIX,
    caption_sha,
    latent_files,
)

ENCODER_ID = (
    "Qwen3.5-9B last_hidden_state, chat template add_generation_prompt+enable_thinking"
)


def encoder_id(precision: str = "bf16") -> str:
    """The ``encoder`` stamp of a TE cache file; bf16 keeps the original string."""
    return ENCODER_ID if precision == "bf16" else f"{ENCODER_ID}, {precision} weights"


def _te_current(
    path: Path,
    caption: str,
    max_tokens: int,
    plain: str = "",
    encoder: str = ENCODER_ID,
) -> bool:
    if not path.exists():
        return False
    try:
        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
    except Exception:
        return False
    return (
        meta.get("caption_sha256") == caption_sha(caption)
        and meta.get("max_text_tokens") == str(max_tokens)
        and meta.get("encoder", ENCODER_ID) == encoder
        and meta.get("plain_caption_sha256", "")
        == (caption_sha(plain) if plain else "")
    )


# ------------------------------------------------------------------ passes


def cache_text(
    pairs,
    out_dir,
    model_dir,
    device,
    max_tokens,
    overwrite,
    caption_format="text",
    precision="bf16",
) -> dict:
    from library.moss.loader import (
        empty_cache,
        encode_prompt,
        load_text_encoder,
        load_tokenizer,
        prompt_ids,
    )

    encoder = encoder_id(precision)
    todo = [
        p
        for p in pairs
        if overwrite
        or not _te_current(
            out_dir / f"{p.stem}{TE_SUFFIX}", p.caption, max_tokens, p.plain, encoder
        )
    ]
    if not todo:
        print("text: all cached", flush=True)
        return {"encoded": 0}
    t0 = time.time()
    tok = load_tokenizer(model_dir)
    longest = max(
        prompt_ids(tok, text, max_tokens).input_ids.shape[1]
        for p in todo
        for text in (p.caption, p.plain or p.caption)
    )
    tok, te = load_text_encoder(
        model_dir, device, longest, tok=tok, precision=precision
    )
    t_load = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    lengths = []
    for i, p in enumerate(todo, 1):
        embeds = encode_prompt(tok, te, p.caption, device, max_tokens).to(
            "cpu", torch.bfloat16
        )
        n = embeds.shape[0]
        lengths.append(n)
        note = "  TRUNCATED" if n >= max_tokens else ""
        path = out_dir / f"{p.stem}{TE_SUFFIX}"
        tensors, meta = carried_variants(path, p.caption)
        tensors["embeds"] = embeds.contiguous()
        meta |= {
            "caption": p.caption,
            "caption_sha256": caption_sha(p.caption),
            "caption_format": caption_format,
            "source_sha256": p.source_sha,
            "n_text": str(n),
            "max_text_tokens": str(max_tokens),
            "encoder": encoder,
            "source": str(p.image),
        }
        plain_note = ""
        if p.plain:
            plain = encode_prompt(tok, te, p.plain, device, max_tokens).to(
                "cpu", torch.bfloat16
            )
            tensors["embeds_plain"] = plain.contiguous()
            meta["plain_caption_sha256"] = caption_sha(p.plain)
            meta["n_text_plain"] = str(plain.shape[0])
            plain_note = f" (plain {plain.shape[0]})"
        print(
            f"  text {i}/{len(todo)} {p.stem}: {n} tokens{plain_note}{note}",
            flush=True,
        )
        save_file(tensors, path, metadata=meta)
    t_enc = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"text: {len(todo)} captions, load {t_load:.1f}s + encode {t_enc:.1f}s, "
        f"tokens {min(lengths)}-{max(lengths)}, peak {peak:.2f} GiB",
        flush=True,
    )
    del tok, te
    empty_cache()
    return {"encoded": len(todo), "load_s": t_load, "encode_s": t_enc, "peak_gib": peak}


def cache_latents(
    pairs, out_dir, model_dir, device, resolution, vae_dtype, overwrite, save_crops
) -> dict:
    from library.moss.loader import empty_cache, load_vae
    from library.moss.model import encode

    plan = []
    for p in pairs:
        with Image.open(p.image) as im:
            size = im.size
        w, h = target_size(*size, resolution)
        path = out_dir / f"{p.stem}_{w}x{h}{LATENT_SUFFIX}"
        current = path.exists() and path.stat().st_mtime >= p.image.stat().st_mtime
        if overwrite or not current:
            plan.append((p, size, w, h, path))
    if not plan:
        print("latents: all cached", flush=True)
        return {"encoded": 0}
    dtype = torch.float32 if vae_dtype == "fp32" else torch.bfloat16
    t0 = time.time()
    vae = load_vae(model_dir, dtype=dtype, device=device)
    t_load = time.time() - t0
    crops = out_dir / "crops"
    if save_crops:
        crops.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    resampled = 0

    def load(job):
        p, _, w, h, _ = job
        image = load_rgb(p.image)
        fitted = fit(image, w, h)
        return fitted, fitted is not image, to_pixels(fitted)

    for i, ((p, size, w, h, path), (fitted, was_resampled, pixels)) in enumerate(
        prefetch(plan, load), 1
    ):
        resampled += was_resampled
        latent = encode(pixels, vae)[0].cpu().contiguous()
        for *_, old in latent_files(out_dir, p.stem):
            if old != path:
                old.unlink()
        save_file(
            {"latent": latent},
            path,
            metadata={
                "width": str(w),
                "height": str(h),
                "source": str(p.image),
                "source_size": f"{size[0]}x{size[1]}",
                "source_mtime": str(p.image.stat().st_mtime),
                "vae_dtype": vae_dtype,
                "resampled": str(was_resampled),
            },
        )
        if save_crops:
            fitted.save(crops / f"{p.stem}.png")
        print(
            f"  latents {i}/{len(plan)} {p.stem}: {size[0]}x{size[1]} -> {w}x{h}",
            flush=True,
        )
    t_enc = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"latents: {len(plan)} images ({resampled} resampled), load {t_load:.1f}s + "
        f"encode {t_enc:.1f}s, peak {peak:.2f} GiB",
        flush=True,
    )
    del vae
    empty_cache()
    return {
        "encoded": len(plan),
        "resampled": resampled,
        "load_s": t_load,
        "encode_s": t_enc,
        "peak_gib": peak,
    }


def cache_refs(
    pairs, refs, out_dir, model_dir, device, resolution, ref_scale, vae_dtype, overwrite
) -> dict:
    """``{stem}.moss_ref`` per pair: the condition fitted to the target's cached
    size, then to :func:`ref_size`. Current while newer than the condition and
    stamped with the same target size and ``ref_scale``."""
    from library.moss.loader import empty_cache, load_vae
    from library.moss.model import encode

    plan = []
    for p in pairs:
        with Image.open(p.image) as im:
            w, h = target_size(*im.size, resolution)
        cond = refs[p.stem]
        path = out_dir / f"{p.stem}{REF_SUFFIX}"
        current = False
        if path.exists() and path.stat().st_mtime >= cond.stat().st_mtime:
            with safe_open(path, "pt") as f:
                meta = f.metadata() or {}
            current = meta.get("target") == f"{w}x{h}" and meta.get("ref_scale") == str(
                ref_scale
            )
        if overwrite or not current:
            plan.append((p.stem, cond, w, h, path))
    if not plan:
        print("refs: all cached", flush=True)
        return {"encoded": 0}
    dtype = torch.float32 if vae_dtype == "fp32" else torch.bfloat16
    vae = load_vae(model_dir, dtype=dtype, device=device)
    t0 = time.time()

    def load(job):
        _, cond, w, h, _ = job
        return to_pixels(fit(fit(load_rgb(cond), w, h), *ref_size(w, h, ref_scale)))

    for i, ((stem, cond, w, h, path), pixels) in enumerate(prefetch(plan, load), 1):
        rw, rh = ref_size(w, h, ref_scale)
        latent = encode(pixels, vae)[0].cpu().contiguous()
        save_file(
            {"latent": latent},
            path,
            metadata={
                "width": str(rw),
                "height": str(rh),
                "target": f"{w}x{h}",
                "ref_scale": str(ref_scale),
                "condition": str(cond),
                "condition_mtime": str(cond.stat().st_mtime),
            },
        )
        print(f"  refs {i}/{len(plan)} {stem}: {w}x{h} -> {rw}x{rh}", flush=True)
    seconds = time.time() - t0
    print(f"refs: {len(plan)} encoded in {seconds:.1f}s", flush=True)
    del vae
    empty_cache()
    return {"encoded": len(plan), "encode_s": seconds, "ref_scale": ref_scale}


def summarize(out_dir: Path) -> dict:
    """Token statistics over the complete pairs in the cache."""
    items = list_items(out_dir)
    if not items:
        return {"items": 0}
    text = [it.n_text for it in items]
    image = [(it.width // STRIDE) * (it.height // STRIDE) for it in items]
    joint = [a + b + it.n_ref for a, b, it in zip(text, image, items)]
    size = sum(p.stat().st_size for p in out_dir.glob("*.moss_*.safetensors"))
    return {
        "items": len(items),
        "text_tokens": {
            "min": min(text),
            "median": statistics.median(text),
            "max": max(text),
        },
        "image_tokens": {
            "min": min(image),
            "median": statistics.median(image),
            "max": max(image),
        },
        "joint_tokens_max": max(joint),
        "shapes": len({(it.width, it.height) for it in items}),
        "bytes": size,
    }


def run_cache(req) -> Path:
    """Both passes over ``req.src``; returns the cache folder."""
    from library.moss.requests import resolve_model_dir

    src = resolve_under_home(req.src)
    out_dir = resolve_under_home(req.out)
    model_dir = resolve_model_dir(req.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = None
    fixed = req.caption_format == "fixed"
    fmt = "text" if fixed else req.caption_format
    if req.pairs:
        src = resolve_under_home(req.pairs)
        if not src.is_file():
            raise SystemExit(f"no pair list at {src} (make i2i-stage TASK=<task>)")
        pairs, refs = manifest_pairs(src, fmt)
    else:
        pairs = find_pairs(src, fmt)
    if not pairs:
        raise SystemExit(f"no image/caption pairs under {src}")
    if fixed:
        pairs = [replace(p, caption=req.fixed_caption.strip()) for p in pairs]
    print(f"{len(pairs)} pairs from {src}, model {model_dir}", flush=True)
    device = torch.device("cuda")
    report = {"src": str(src), "resolution": req.resolution}
    if not req.skip_text:
        report["text"] = cache_text(
            pairs,
            out_dir,
            model_dir,
            device,
            req.max_text_tokens,
            req.overwrite,
            req.caption_format,
            req.text_encoder,
        )
    if not req.skip_latents:
        report["latents"] = cache_latents(
            pairs,
            out_dir,
            model_dir,
            device,
            req.resolution,
            req.vae_dtype,
            req.overwrite,
            req.save_crops,
        )
    if refs is not None and not req.skip_latents:
        report["refs"] = cache_refs(
            pairs,
            refs,
            out_dir,
            model_dir,
            device,
            req.resolution,
            req.ref_scale,
            req.vae_dtype,
            req.overwrite,
        )
    report["summary"] = summarize(out_dir)
    (out_dir / "cache_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"cache at {out_dir}", flush=True)
    return out_dir
