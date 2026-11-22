#!/usr/bin/env python3
"""Turbo end to end in ComfyUI: VAE Decode against the pixel head node.

The ComfyUI app's own startup (``../comfy/main.py`` imported as a module: dynamic VRAM, model
management, ``extra_model_paths.yaml``), then the nodes of a turbo workflow called in order:
UNETLoader (anima-base) → AnimaAdapterLoader (turbo LoRA, ``stream``) → AnimaBlockCompile →
Anima Pixel Head Tap → KSampler (euler, simple, 4 steps, CFG 1). The sampler's LATENT is then
decoded twice, by VAE Decode and by Anima Pixel Decode, each timed with a CUDA sync; both end
with the IMAGE on ComfyUI's intermediate device. Each prompt is also sampled once without the
tap, for the tap's own cost. Text encoding (CLIP Text Encode) is outside the totals.

P3's 16 prompts at their ~1 MP sizes; one warmup render + both decodes per (stream, size) is
dropped. The Adapter Loader's fp32 stream (the head's training features); ``--streams
fp32,bf16`` adds ComfyUI's own bf16 forward. Head vs VAE PSNR on the same latent.

``--scale 2``: the chain VAE Decode → ResShift SR Upscale (the ×2 student at the node's
defaults, ``ComfyUI-Distilled-ResShift``) against Anima Pixel Decode with a ×2 head file (the node
decodes at the file's scale); PSNR and VGG-LPIPS of the head against the chain. ``--lora`` / ``--pixel_head``
take a path outside ComfyUI's folders (the turbo point of the ×2 heads is v4_longer).

    make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/comfy_e2e.py \
        --label comfy_turbo_mmdd_hhmm"
    make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/comfy_e2e.py \
        --scale 2 --lora output/ckpt/release/anima_turbo_4step_v4_longer.safetensors \
        --pixel_head <run>/head_x2.safetensors --label comfy_x2_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

from library.env import resolve_under_home

COMFY = resolve_under_home("../comfy").resolve()
NODE_PACKS = (
    "comfyui-anima-lora-adapter",
    "comfyui-anima-blockcompile",
    "comfyui-anima-pixel-decoder",
)
RESSHIFT_PACK = "comfyui-distilled-resshift"
MB = 1 << 20


def start_comfy(flags: list[str], packs=NODE_PACKS):
    """Import ComfyUI's ``main.py`` with its own argv: the app's module-level setup runs
    (args, logging, dynamic VRAM, model patcher class), the server does not start."""
    sys.path.insert(0, str(COMFY))
    # no prestartup scripts (ComfyUI-Manager's installs deps); the packs load below
    flags = ["--disable-all-custom-nodes", *flags]
    argv, sys.argv = sys.argv, [str(COMFY / "main.py"), *flags]
    try:
        import main  # noqa: F401
    finally:
        sys.argv = argv
    main.apply_custom_paths()
    import nodes

    for pack in packs:
        ok = asyncio.run(nodes.load_custom_node(str(COMFY / "custom_nodes" / pack)))
        assert ok, f"custom node {pack} failed to load"
    return main, nodes


def main_():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--unet", default="anima-base-v1.0.safetensors")
    p.add_argument("--clip", default="qwen_3_06b_base.safetensors")
    p.add_argument("--vae", default="qwen_image_vae.safetensors")
    p.add_argument("--lora", default="anima_turbo_4step.safetensors")
    p.add_argument("--head", default="anima_pixel_head.safetensors")
    p.add_argument(
        "--pixel_head",
        default=None,
        help="a head file by path (outside models/pixel_heads/); overrides --head",
    )
    p.add_argument(
        "--scale",
        type=int,
        choices=(1, 2),
        default=1,
        help="2: VAE Decode → ResShift ×2 against Anima Pixel Decode at scale 2",
    )
    p.add_argument("--streams", default="fp32")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--only", nargs="*", default=None, help="prompt names (smoke)")
    p.add_argument(
        "--tags",
        default="",
        help="tags appended to each prompt's leading tag sentence, e.g. 'monochrome, comic'",
    )
    p.add_argument("--save_images", type=int, default=16, help="first N, first stream")
    p.add_argument(
        "--comfy_args",
        default="",
        help="extra ComfyUI CLI flags, e.g. '--use-flash-attention'",
    )
    p.add_argument("--label", default=None)
    args = p.parse_args()

    packs = NODE_PACKS + ((RESSHIFT_PACK,) if args.scale == 2 else ())
    comfy_main, nodes = start_comfy(args.comfy_args.split(), packs)
    import comfy.memory_management
    import comfy.model_management
    import folder_paths
    import torch

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from bench._anima import DEFAULT_NEG
    from bench._common import make_run_dir, start_heartbeat, write_result
    from bench.pixel_readout._prompts import PROMPTS
    from library.anima.pixel_head_train import psnr, to_pil

    start_heartbeat(label="comfy_e2e")
    folder_paths.add_model_folder_path("loras", str(resolve_under_home("models/loras")))

    def by_path(folder: str, name: str) -> str:
        """A file given by path → its folder registered under ``folder``, the basename."""
        path = Path(resolve_under_home(name))
        if "/" not in name or not path.is_file():
            return name
        folder_paths.add_model_folder_path(folder, str(path.parent))
        return path.name

    args.lora = by_path("loras", args.lora)
    if args.pixel_head:
        args.head = by_path("pixel_heads", args.pixel_head)
    mapping = nodes.NODE_CLASS_MAPPINGS
    run_dir = make_run_dir("pixel_readout", label=args.label)
    prompts = [x for x in PROMPTS if not args.only or x[0] in args.only]
    if (
        args.tags
    ):  # "masterpiece, …, safe. Prose" → "masterpiece, …, safe, <tags>. Prose"
        prompts = [
            (n, t.replace(". ", f", {args.tags}. ", 1), hw) for n, t, hw in prompts
        ]
    device = comfy.model_management.get_torch_device()

    def sync_time(fn):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return out, dt, (torch.cuda.max_memory_allocated() - base) / MB

    unet = nodes.UNETLoader().load_unet(args.unet, "default")[0]
    clip = nodes.CLIPLoader().load_clip(args.clip, "stable_diffusion")[0]
    vae = nodes.VAELoader().load_vae(args.vae)[0]
    encode = nodes.CLIPTextEncode()
    neg = encode.encode(clip, DEFAULT_NEG)[0]
    conds, te_s = {}, {}
    for name, prompt, _ in prompts:
        (c,), te_s[name], _ = sync_time(lambda: encode.encode(clip, prompt))
        conds[name] = c
    vae_decode = nodes.VAEDecode()
    pixel_decode = mapping["AnimaPixelDecode"]()
    lp = rsd = up = None
    if args.scale == 2:
        import lpips

        rsd = mapping["ResShiftLoader"]().load(
            "x2", "rsd_student_18k.safetensors", "bf16"
        )
        rsd = rsd[0]
        up = mapping["ResShiftUpscale"]()
        lp = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    def vae_path(out):
        """VAE Decode, then (×2) ResShift SR Upscale at the node's defaults."""
        (img,) = vae_decode.decode(vae, out)
        if up is not None:
            (img,) = up.upscale(rsd, img, args.seed, 512, 64, 4, True)
        return img

    def head_path(out):
        return pixel_decode.decode(tapped, out, args.head)[0]  # scale: the file's

    @torch.no_grad()
    def lpips_img(a, b, tile: int = 768) -> float:
        """Area-weighted mean over ``tile``-px tiles of (3, H, W) images in [-1, 1]."""
        a, b = a.to(device), b.to(device)
        h, w = a.shape[-2:]
        tot = 0.0
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                ta = a[None, :, y : y + tile, x : x + tile]
                tb = b[None, :, y : y + tile, x : x + tile]
                tot += lp(ta, tb).mean().item() * ta.shape[-2] * ta.shape[-1]
        return tot / (h * w)

    def latent(h, w):
        return {"samples": torch.zeros(1, 16, 1, h // 8, w // 8)}

    def sample(model, name, h, w, seed):
        return nodes.common_ksampler(
            model, seed, args.steps, args.cfg, "euler", "simple",
            conds[name], neg, latent(h, w),
        )[0]  # fmt: skip

    def to_pm1(img):  # IMAGE (1, H, W, 3) in [0, 1] → (3, H, W) in [-1, 1]
        return img[0].permute(2, 0, 1).float() * 2 - 1

    results = {}
    for si, stream in enumerate(args.streams.split(",")):
        model = mapping["AnimaAdapterLoader"]().apply(
            unet, args.lora, 1.0, 1.0, stream
        )[0]
        if not args.no_compile:
            model = mapping["AnimaBlockCompile"]().patch(model)[0]
        tapped = mapping["AnimaPixelHeadTap"]().patch(model)[0]

        seen = set()
        for name, _, (h, w) in prompts:
            if (h, w) in seen:
                continue
            seen.add((h, w))
            out = sample(tapped, name, h, w, 1000)
            sample(model, name, h, w, 1000)
            vae_path(out)
            head_path(out)
            print(f"[{stream}] warmup {h}x{w} done", flush=True)

        rows = []
        for k, (name, _, (h, w)) in enumerate(prompts):
            _, plain_s, _ = sync_time(lambda: sample(model, name, h, w, args.seed))
            out, dit_s, dit_mb = sync_time(
                lambda: sample(tapped, name, h, w, args.seed)
            )
            img_v, vae_s, vae_mb = sync_time(lambda: vae_path(out))
            img_h, head_s, head_mb = sync_time(lambda: head_path(out))
            row = {
                "name": name,
                "size": [h, w],
                "te_s": te_s[name],
                "sample_plain_s": plain_s,
                "sample_tapped_s": dit_s,
                "sample_peak_mb": dit_mb,
                "vae_s": vae_s,
                "vae_peak_mb": vae_mb,
                "head_s": head_s,
                "head_peak_mb": head_mb,
                "total_vae_s": plain_s + vae_s,
                "total_head_s": dit_s + head_s,
                "psnr_head_vs_vae": psnr(to_pm1(img_h), to_pm1(img_v)),
            }
            if lp is not None:
                row["lpips_head_vs_vae"] = lpips_img(to_pm1(img_h), to_pm1(img_v))
            rows.append(row)
            if si == 0 and k < args.save_images:
                to_pil(to_pm1(img_v)).save(run_dir / f"{name}_vae.png")
                to_pil(to_pm1(img_h)).save(run_dir / f"{name}_head.png")
            print(
                f"[{stream}] {name} {h}x{w}: sample {plain_s:.3f}s (tapped {dit_s:.3f}s)  "
                f"vae {vae_s * 1e3:.0f}ms  head {head_s * 1e3:.0f}ms  "
                f"psnr {row['psnr_head_vs_vae']:.2f}"
                + (f"  lpips {row['lpips_head_vs_vae']:.4f}" if lp is not None else ""),
                flush=True,
            )
        keys = [
            k
            for k in rows[0]
            if k.endswith(("_s", "_mb")) or k.startswith(("psnr", "lpips"))
        ]
        summary = {
            k: {
                "median": statistics.median(r[k] for r in rows),
                "mean": statistics.fmean(r[k] for r in rows),
            }
            for k in keys
        }
        for k, v in summary.items():
            print(f"[{stream}] {k}: median {v['median']:.4f}  mean {v['mean']:.4f}")
        results[stream] = {"summary": summary, "per_image": rows}
        comfy.model_management.unload_all_models()
        model = tapped = None  # noqa: F841
        torch.cuda.empty_cache()

    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "comfy_version": __import__("comfyui_version").__version__,
            "dynamic_vram": bool(comfy.memory_management.aimdo_enabled),
            "attention": "flash"
            if comfy_main.args.use_flash_attention
            else "comfy default",
            "device": comfy.model_management.get_torch_device_name(device),
            "scale": args.scale,
            "vae_path": "VAE Decode → ResShift ×2 (rsd_student_18k, node defaults)"
            if args.scale == 2
            else "VAE Decode",
            "streams": results,
        },
        label=args.label,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main_()
