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

    make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/comfy_e2e.py \
        --label comfy_turbo_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time

from library.env import resolve_under_home

COMFY = resolve_under_home("../comfy").resolve()
NODE_PACKS = (
    "comfyui-anima-lora-adapter",
    "comfyui-anima-blockcompile",
    "comfyui-anima-pixel-decoder",
)
MB = 1 << 20


def start_comfy(flags: list[str]):
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

    for pack in NODE_PACKS:
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
        "--comfy_args", default="", help="extra ComfyUI CLI flags, e.g. '--use-flash-attention'"
    )
    p.add_argument("--label", default=None)
    args = p.parse_args()

    comfy_main, nodes = start_comfy(args.comfy_args.split())
    import comfy.memory_management
    import comfy.model_management
    import folder_paths
    import torch

    from bench._anima import DEFAULT_NEG
    from bench._common import make_run_dir, start_heartbeat, write_result
    from bench.pixel_readout.linear_probe import psnr, to_pil
    from bench.pixel_readout.p3_generated import PROMPTS

    start_heartbeat(label="comfy_e2e")
    folder_paths.add_model_folder_path("loras", str(resolve_under_home("models/loras")))
    mapping = nodes.NODE_CLASS_MAPPINGS
    run_dir = make_run_dir("pixel_readout", label=args.label)
    prompts = [x for x in PROMPTS if not args.only or x[0] in args.only]
    if args.tags:  # "masterpiece, …, safe. Prose" → "masterpiece, …, safe, <tags>. Prose"
        prompts = [(n, t.replace(". ", f", {args.tags}. ", 1), hw) for n, t, hw in prompts]
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
            vae_decode.decode(vae, out)
            pixel_decode.decode(tapped, out, args.head)
            print(f"[{stream}] warmup {h}x{w} done", flush=True)

        rows = []
        for k, (name, _, (h, w)) in enumerate(prompts):
            _, plain_s, _ = sync_time(lambda: sample(model, name, h, w, args.seed))
            out, dit_s, dit_mb = sync_time(
                lambda: sample(tapped, name, h, w, args.seed)
            )
            (img_v,), vae_s, vae_mb = sync_time(lambda: vae_decode.decode(vae, out))
            (img_h,), head_s, head_mb = sync_time(
                lambda: pixel_decode.decode(tapped, out, args.head)
            )
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
            rows.append(row)
            if si == 0 and k < args.save_images:
                to_pil(to_pm1(img_v)).save(run_dir / f"{name}_vae.png")
                to_pil(to_pm1(img_h)).save(run_dir / f"{name}_head.png")
            print(
                f"[{stream}] {name} {h}x{w}: sample {plain_s:.3f}s (tapped {dit_s:.3f}s)  "
                f"vae {vae_s * 1e3:.0f}ms  head {head_s * 1e3:.0f}ms  "
                f"psnr {row['psnr_head_vs_vae']:.2f}",
                flush=True,
            )
        keys = [k for k in rows[0] if k.endswith(("_s", "_mb")) or k.startswith("psnr")]
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
            "streams": results,
        },
        label=args.label,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main_()
