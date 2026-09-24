"""Precache Qwen3-VL text embeddings and VAE latents for a training folder.

Both encoders are far too large to keep beside the transformer, so training
never sees them: this runs the same phase split as ``smoke_t2i.py`` (text
encoder block-swapped onto the card, dropped, then the VAE) and leaves two
files per image.

    {stem}.te.safetensors      prompt_embeds (L, 4096) + prompt_embeds_mask (L,)
    {stem}.latent.safetensors  latents (T, 64), packed, mean/std normalised

They are written separately so either stage can be re-run alone — a caption
edit needs the text pass only. Both skip on existence unless ``--overwrite``.

Nothing is cropped and nothing is padded, on either side:

- Each image keeps its **native aspect ratio** and lands at ~``resolution``^2
  pixels with both edges on a multiple of 32 (``calculate_dimensions``, the
  pipeline's own). A square crop would throw away most of a 0.44-aspect image.
- Text length stays at its natural value per caption. The model's rope gives the
  image block a frame position equal to the text length, so padding during
  training would teach the adapter an offset that inference never reproduces.

Both leave the joint sequence varying per sample, which is why ``train.py``
compiles the blocks dynamically.

    make daemon-run ARGS="scripts/qwen21/cache.py \
        --src 'post_image_dataset/resized/channel_(caststation)'"
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

from library.env import resolve_under_home
from library.qwen21.loader import (
    TEXT_ENCODER_BLOCKS,
    drop_text_encoder,
    empty_cache,
    free_vram_gb,
    load_pipeline,
    load_text_encoder,
    place,
)
from library.qwen21.requests import CacheRequest, resolve_model_dir
from library.qwen21.scan import duplicate_stems, find_images

VAE_SCALE_FACTOR = 16


def find_pairs(src: Path) -> list[tuple[str, Path, str]]:
    """``(stem, image path, caption)`` for every image with a caption sidecar,
    subfolders included. The cache is flat, so a stem seen twice is an error."""
    images = find_images(src)
    dupes = duplicate_stems(images)
    if dupes:
        lines = [
            f"  {stem}: {', '.join(str(p) for p in paths)}"
            for stem, paths in dupes.items()
        ]
        raise SystemExit(
            f"{len(dupes)} file names appear in more than one subfolder — the "
            "cache is keyed by file name, rename them first:\n" + "\n".join(lines[:20])
        )
    pairs = []
    for path in images:
        sidecar = path.with_suffix(".txt")
        if not sidecar.exists():
            print(f"skip {path.name}: no caption sidecar", flush=True)
            continue
        pairs.append((path.stem, path, sidecar.read_text(encoding="utf-8").strip()))
    return pairs


def target_size(image: Image.Image, resolution: int) -> tuple[int, int]:
    """``(width, height)`` at ~``resolution``^2 pixels, native aspect kept.

    The pipeline's own ``calculate_dimensions``: nothing is cropped, and both
    edges land on a multiple of 32 — which is what keeps the latent token count
    divisible by 4, as the target image's ``img_mask`` slots require.
    """
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
        calculate_dimensions,
    )

    width, height, _ = calculate_dimensions(
        resolution * resolution, image.width / image.height
    )
    return width, height


def cache_text(
    pairs: list[tuple[str, Path, str]],
    out_dir: Path,
    model_dir: Path,
    device: torch.device,
    te_blocks_to_swap: int | None,
    overwrite: bool,
) -> None:
    todo = [
        p
        for p in pairs
        if overwrite or not (out_dir / f"{p[0]}.te.safetensors").exists()
    ]
    if not todo:
        print("text: all cached", flush=True)
        return

    te = load_text_encoder(model_dir)
    pipe = load_pipeline(model_dir, components=("text_encoder",), text_encoder=te)
    attached = place(
        te,
        TEXT_ENCODER_BLOCKS,
        device,
        blocks_to_swap=te_blocks_to_swap,
        label="text_encoder",
    )

    t0 = time.time()
    lengths = []
    with torch.no_grad():
        for i, (stem, _path, caption) in enumerate(todo, 1):
            print(f"  text {i}/{len(todo)} {stem}", flush=True)
            embeds, mask, _pad = pipe.encode_prompt(prompt=caption, device=device)
            embeds = embeds[0].detach().to("cpu", torch.bfloat16)
            if mask is None:
                mask = torch.ones(embeds.shape[0], dtype=torch.int64)
            else:
                mask = mask[0].detach().to("cpu", torch.int64)
            save_file(
                {"prompt_embeds": embeds.contiguous(), "prompt_embeds_mask": mask},
                out_dir / f"{stem}.te.safetensors",
                metadata={"tokens": str(embeds.shape[0])},
            )
            lengths.append(embeds.shape[0])
    print(
        f"text: {len(todo)} captions in {time.time() - t0:.1f}s  "
        f"tokens min {min(lengths)} max {max(lengths)}  "
        f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
        flush=True,
    )

    if attached is not None:
        attached.detach()
    del te
    drop_text_encoder(pipe)
    del pipe
    empty_cache()


def cache_latents(
    pairs: list[tuple[str, Path, str]],
    out_dir: Path,
    model_dir: Path,
    device: torch.device,
    resolution: int,
    overwrite: bool,
    save_crops: bool,
) -> None:
    todo = [
        p
        for p in pairs
        if overwrite or not (out_dir / f"{p[0]}.latent.safetensors").exists()
    ]
    if not todo:
        print("latents: all cached", flush=True)
        return

    pipe = load_pipeline(model_dir, components=("vae",))
    pipe.vae.to(device)
    pipe.vae.enable_tiling()
    generator = torch.Generator(device=device).manual_seed(0)
    crops = out_dir / "crops"
    if save_crops:
        crops.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    shapes: list[tuple[int, int, int]] = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i, (stem, path, _caption) in enumerate(todo, 1):
            print(f"  latents {i}/{len(todo)} {stem}", flush=True)
            # RGBA, not RGB: the 2.1 VAE encoder takes 4 channels, which is why
            # the pipeline converts its condition images the same way.
            image = Image.open(path).convert("RGBA")
            width, height = target_size(image, resolution)
            # (1, 4, 1, H, W) for the 3D VAE, then normalised and packed to
            # (T, 64). `preprocess` does the aspect-preserving resize itself.
            pixels = pipe.image_processor.preprocess(
                image, width=width, height=height
            ).unsqueeze(2)
            if save_crops:
                image.resize((width, height), Image.LANCZOS).convert("RGB").save(
                    crops / f"{stem}.png"
                )
            pixels = pixels.to(device=device, dtype=pipe.vae.dtype)
            encoded = pipe._encode_vae_image(pixels, generator)
            latent_h = height // VAE_SCALE_FACTOR
            latent_w = width // VAE_SCALE_FACTOR
            latents = pipe._pack_latents(
                encoded, 1, pipe.vae.config.z_dim, latent_h, latent_w
            )[0]
            shapes.append((width, height, latents.shape[0]))
            save_file(
                {"latents": latents.detach().to("cpu", torch.bfloat16).contiguous()},
                out_dir / f"{stem}.latent.safetensors",
                metadata={
                    "size": f"{width}x{height}",
                    "latent_h": str(latent_h),
                    "latent_w": str(latent_w),
                    "tokens": str(latents.shape[0]),
                    "source_size": f"{image.width}x{image.height}",
                },
            )
    token_counts = sorted(t for _w, _h, t in shapes)
    print(
        f"latents: {len(todo)} images at ~{resolution}^2 px in {time.time() - t0:.1f}s  "
        f"{len({(w, h) for w, h, _t in shapes})} distinct shapes, "
        f"tokens {token_counts[0]}-{token_counts[-1]}  "
        f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
        flush=True,
    )
    del pipe
    empty_cache()


def run_cache(req: CacheRequest) -> Path:
    """Both passes over ``req.src``; returns the cache folder."""
    src = resolve_under_home(req.src)
    out_dir = resolve_under_home(req.out)
    model_dir = resolve_model_dir(req.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(src)
    if not pairs:
        raise SystemExit(f"no image/caption pairs under {src}")
    print(f"{len(pairs)} pairs from {src}, model {model_dir}", flush=True)

    torch.cuda.init()
    device = torch.device("cuda")
    print(f"free VRAM: {free_vram_gb():.2f} GB", flush=True)

    if not req.skip_text:
        cache_text(
            pairs, out_dir, model_dir, device, req.te_blocks_to_swap, req.overwrite
        )
    if not req.skip_latents:
        cache_latents(
            pairs,
            out_dir,
            model_dir,
            device,
            req.resolution,
            req.overwrite,
            req.save_crops,
        )
    print(f"cache at {out_dir}", flush=True)
    return out_dir
