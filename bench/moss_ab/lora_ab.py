#!/usr/bin/env python3
"""LoRA A/B on MOSS-Image Base and SFT: each prompt × seed over ``--arms``.

An arm is ``<sampler>:<multiplier>``; sampler ``euler``, ``er_sde`` or ``cns``
(er_sde with ``--cns_gamma``'s recolored noise); Spectrum at its shipped settings
by default. Same noise per (prompt, seed), so each row differs by the arm only.
Writes one grid per (variant, prompt, seed) into ``--out``. Blocks are compiled
after the LoRA is attached (``--no-compile`` for eager; compare compiled arms only
against compiled arms). Latent names carry a digest of the LoRA (path + mtime),
prompts, steps, cfg, size, compile, Spectrum (and CNS γ on cns arms), so a rerun
resumes only what matches.

    make daemon-run ARGS="bench/moss_ab/lora_ab.py --lora output/ckpt_moss/x.safetensors"
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from bench.moss_ab._render import (
    decoded,
    load_dit,
    lora_id,
    render,
    save_latent,
    settings_tag,
    spectrum_settings,
    uncond_embeds,
)
from library.env import resolve_under_home
from library.moss.cache import caption_converter
from library.moss.cns import MossCNS
from library.moss.loader import TE_PRECISIONS, empty_cache, load_vae
from library.moss.model import snapped_sigmas

PROMPTS = [
    # Tag bag in the training captions' own format, characters it saw.
    "safe, 1girl, hatsune miku, vocaloid, {artist}, aqua eyes, aqua hair, "
    "twintails, smile, peace sign, upper body, looking at viewer, simple background, white background",
    # A character the caststation set never shows.
    "safe, 1girl, akita neru, vocaloid, {artist}, yellow eyes, blonde hair, "
    "side ponytail, holding phone, looking at viewer, upper body, simple background",
    # Natural language, no tag grammar.
    "An illustration in the style of {artist}: a girl with short black hair "
    "in a sailor school uniform sitting on a bench under cherry blossoms, smiling.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument(
        "--artist",
        default="@channel (caststation)",
        help="artist tag substituted into PROMPTS (the LoRA's trigger)",
    )
    ap.add_argument(
        "--prompts_file",
        default="",
        help="one prompt per line ({artist} substituted) in place of PROMPTS",
    )
    ap.add_argument("--variants", default="base,sft")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument(
        "--arms",
        default="er_sde:1.0,er_sde:0.0",
        help="<euler|er_sde|cns>:<multiplier>, comma-separated",
    )
    ap.add_argument(
        "--cns_gamma",
        default="auto",
        help="γ npz for cns arms; auto = the shipped γ for the variant at the nearest cfg",
    )
    ap.add_argument(
        "--spectrum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Spectrum at GenerateRequest's defaults",
    )
    ap.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile the blocks after the LoRA is attached, as generate does",
    )
    ap.add_argument(
        "--prompt_format",
        default="layout_json",
        choices=("text", "layout_json"),
        help="encode PROMPTS as typed or mapped to layout JSON (match the LoRA's cache)",
    )
    ap.add_argument(
        "--te_precision",
        default="bf16",
        choices=TE_PRECISIONS,
        help="prompt TE weights (match the LoRA's cache encoder)",
    )
    ap.add_argument(
        "--stream",
        default="bf16",
        choices=("bf16", "fp32"),
        help="residual stream / modulation / norm precision around the bf16 block GEMMs",
    )
    ap.add_argument(
        "--out",
        default="",
        help="default: output/tests/ab_<lora stem>_<steps>st (+ _fp32s with --stream fp32)",
    )
    args = ap.parse_args()
    w, h = map(int, args.size.split("x"))
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [(a.split(":")[0], float(a.split(":")[1])) for a in args.arms.split(",")]
    out = resolve_under_home(
        args.out
        or f"output/tests/ab_{Path(args.lora).stem}_{args.steps}st"
        + ("_fp32s" if args.stream == "fp32" else "")
    )
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    templates = PROMPTS
    if args.prompts_file:
        lines = resolve_under_home(args.prompts_file).read_text().splitlines()
        templates = [line for line in lines if line.strip()]
    prompts = [
        caption_converter(args.prompt_format)(p.format(artist=args.artist))
        for p in templates
    ]

    lora = resolve_under_home(args.lora)
    embeds, uncond, uncond_mask = uncond_embeds(device, prompts, args.te_precision)
    sigmas = snapped_sigmas(args.steps)
    tag = settings_tag(
        lora=lora_id(lora),
        prompts=prompts,
        steps=args.steps,
        cfg=args.cfg,
        size=args.size,
        compile=args.compile,
        spectrum=spectrum_settings(args.spectrum),
        # bf16 stays out of the digest so latents rendered before the flag resume.
        **({"te": args.te_precision} if args.te_precision != "bf16" else {}),
        **({"stream": args.stream} if args.stream != "bf16" else {}),
    )
    cns_tag = settings_tag(base=tag, cns_gamma=args.cns_gamma)
    lat_dir = out / "latents"
    lat_dir.mkdir(exist_ok=True)
    variants = args.variants.split(",")
    jobs = [
        (v, pi, s, a)
        for v in variants
        for pi in range(len(prompts))
        for s in seeds
        for a in arms
    ]

    def stem(job):
        v, pi, s, (sampler, m) = job
        return f"{v}_p{pi}_s{s}_x{m}_{sampler}_{cns_tag if sampler == 'cns' else tag}"

    def lat_path(job):
        return lat_dir / f"{stem(job)}.safetensors"

    for variant in variants:
        todo = [j for j in jobs if j[0] == variant and not lat_path(j).exists()]
        if not todo:
            continue
        tr, net = load_dit(
            lora, device, variant, compile=args.compile, stream=args.stream
        )
        for job in todo:
            _, pi, seed, (sampler, m) = job
            net.set_multiplier(m)
            cns = (
                MossCNS.from_spec(args.cns_gamma, variant, args.cfg)
                if sampler == "cns"
                else None
            )
            z = render(
                tr,
                embeds[prompts[pi]],
                uncond,
                uncond_mask,
                sigmas,
                seed=seed,
                height=h,
                width=w,
                cfg=args.cfg,
                sampler="euler" if sampler == "euler" else "er_sde",
                spectrum=args.spectrum,
                cns=cns,
            )
            save_latent(z, lat_path(job))
            print(
                f"[{variant}] p{pi} s{seed} {sampler} x{m} -> {lat_path(job).name}",
                flush=True,
            )
        del net, tr
        empty_cache()

    vae = load_vae("", device=device)
    rows = []
    for job in jobs:
        variant, pi, seed, (sampler, m) = job
        path = out / f"{stem(job)}.png"
        decoded(lat_path(job), vae, device, png=path)
        rows.append(
            {
                "variant": variant,
                "prompt": pi,
                "seed": seed,
                "sampler": sampler,
                "multiplier": m,
                "path": str(path),
            }
        )
    for variant in args.variants.split(","):
        for pi in range(len(prompts)):
            for seed in seeds:
                ims = [
                    Image.open(out / f"{stem((variant, pi, seed, a))}.png")
                    for a in arms
                ]
                grid = Image.new("RGB", (w * len(ims), h + 32), "white")
                d = ImageDraw.Draw(grid)
                for i, ((sampler, m), im) in enumerate(zip(arms, ims)):
                    grid.paste(im, (i * w, 32))
                    d.text(
                        (i * w + 8, 8),
                        f"{variant} p{pi} s{seed} {sampler} lora x{m}",
                        fill="black",
                    )
                suffix = (
                    ""
                    if args.arms == ap.get_default("arms")
                    else "_" + args.arms.replace(":", "").replace(",", "_")
                ) + ("" if args.spectrum else "_nospec")
                grid.save(out / f"grid_{variant}_p{pi}_s{seed}{suffix}.png")
    (out / "index.json").write_text(
        json.dumps(
            {"args": vars(args), "tag": tag, "prompts": prompts, "rows": rows},
            indent=2,
        )
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
