#!/usr/bin/env python3
"""E4 render + eval — score the trained arms against the frozen manifest.

Consumes `runs/<stamp>-e4-manifest/e4_manifest.json` (the pre-registered
prompts/seeds/sizes/splits — nothing is re-drawn here) and, per artist:

  * renders every manifest prompt under each arm's checkpoint
    (`e4_{slug}_{arm}_s{seed}`) at the prompt's own free-fit bucket and
    `gen_seed`, 20 steps / CFG 1.0 (the trainer's CMMD convention);
  * saves the PNGs for the eyeball pass;
  * scores per arm: `cmmd_holdout` (PE-Core pooled MMD² vs ALL real holdout
    stems), `cmmd_member` (vs ALL real train stems), the real-vs-real noise
    floor (disjoint holdout halves), and the paired PE-Core cosine between
    arms at matched (prompt, gen_seed).

The reference pools follow the manifest's `eval_protocol` verbatim ("all
holdout stems" — NOT generalize.py's prompt-exclusion variant, which would
leave the thin channel cell with zero refs). Machinery is generalize.py's
(GenerationRequest render loop, decode_latent, PE-Core pool, cmmd_from_pools).

Usage (GPU — submit via daemon)::

    make daemon-run ARGS="project/sigma_lowres/paper_bench/e4_render_eval.py \
        --manifest project/sigma_lowres/paper_bench/runs/20260729-1537-e4-manifest/e4_manifest.json"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from anima_lora import (  # noqa: E402
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_vae,
)
from library.inference import load_shared_models  # noqa: E402
from library.inference.output import decode_latent  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402
from library.training.cmmd import cmmd_from_pools, pool_and_normalize  # noqa: E402
from library.vision.encoder import (  # noqa: E402
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

RESIZED = REPO / "post_image_dataset" / "resized"
ARMS = ("native", "sigma896", "unsafe768")


def slug(artist: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", artist).strip("_")


def _pil_to_minus1to1(img: Image.Image) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(img.convert("RGB"))).float()
    return (t / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)


def render_arm(args, label: str, adapter: Path, prompts: list[dict], device):
    """CPU latents for every manifest prompt under one arm's checkpoint."""
    ckpt = default_checkpoints()
    base_req = GenerationRequest(
        dit=ckpt.dit,
        vae=ckpt.vae,
        text_encoder=ckpt.text_encoder,
        prompt="placeholder",
        save_path="__unused__",
        infer_steps=args.steps,
        guidance_scale=args.cfg,
        seed=0,
        lora_weight=[str(adapter)],
        lora_multiplier=[1.0],
        extra_argv=("--compile_blocks",) if args.compile_blocks else (),
    )
    base_args = base_req.to_args()
    base_args.device = device
    gen_settings = get_generation_settings(base_args)
    shared_models = load_shared_models(base_args)
    shared_models["conds_cache"] = {}

    latents: list[torch.Tensor] = []
    for i, p in enumerate(prompts):
        pa = dataclasses.replace(
            base_req,
            prompt=p["caption"],
            seed=int(p["gen_seed"]),  # frozen in the manifest — paired design
            image_size=(int(p["height"]), int(p["width"])),
        ).to_args()
        pa.device = device
        print(f"[{label}] {i + 1}/{len(prompts)} {p['stem']}", flush=True)
        latents.append(generate(pa, gen_settings, shared_models).detach().to("cpu"))

    shared_models.clear()
    del gen_settings
    clean_memory_on_device(device)
    return latents


def encode_pool(bundle, images: list[torch.Tensor]) -> torch.Tensor:
    pooled = []
    for img in images:
        if img.dim() == 3:
            img = img.unsqueeze(0)
        feats = encode_pe_from_imageminus1to1(bundle, img.to(bundle.device))
        pooled.append(pool_and_normalize(feats[0]).to("cpu"))
    return torch.stack(pooled)


def real_pool(bundle, artist: str, stems: list[str]) -> torch.Tensor:
    imgs = []
    for s in stems:
        p = RESIZED / artist / f"{s}.png"
        with Image.open(p) as im:
            imgs.append(_pil_to_minus1to1(im))
    return encode_pool(bundle, imgs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument(
        "--prompts",
        default=None,
        help="Prompt-amendment JSON (e.g. e4_prompts_sfw.json) whose "
        "cells[artist].prompts replace the manifest's — splits/pools/sizes "
        "still come from the manifest.",
    )
    ap.add_argument("--train_seed", type=int, default=1001)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--artists", nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument(
        "--compile_blocks", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--out", default=None, help="Run dir (default: stamped sibling).")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    artists = args.artists or man["artists"]
    proto = man["eval_protocol"]
    deviates = args.steps != proto["infer_steps"] or args.cfg != proto["cfg"]
    if deviates:
        print(
            f"NOTE: steps/cfg ({args.steps}/{args.cfg}) deviate from the "
            f"pre-registered protocol ({proto['infer_steps']}/{proto['cfg']}) — "
            "recorded in result.json; CMMD here is NOT comparable to the "
            "cfg-1.0 floors/history, arm-vs-arm within this run only.",
            flush=True,
        )

    if args.out:
        out = Path(args.out)
    else:
        stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
        out = HERE / "runs" / f"{stamp}-e4-eval-s{args.train_seed}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result: dict = {
        "manifest": str(args.manifest),
        "train_seed": args.train_seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "arms": list(args.arms),
        "cells": {},
    }

    if deviates:
        result["protocol_deviation"] = (
            f"steps/cfg {args.steps}/{args.cfg} vs pre-registered "
            f"{proto['infer_steps']}/{proto['cfg']}"
        )
    override = json.loads(Path(args.prompts).read_text()) if args.prompts else None
    if override:
        result["prompts_override"] = str(args.prompts)

    for artist in artists:
        cell = man["cells"][artist]
        prompts = override["cells"][artist]["prompts"] if override else cell["prompts"]
        aslug = slug(artist)

        # Phase 1+2 — render + decode + save PNGs per arm. An arm whose PNGs
        # are all already in the run dir is skipped (resume after a partial
        # run); pools always derive from the saved PNGs either way, so both
        # paths feed phase 3 identically.
        for arm in args.arms:
            arm_dir = out / "renders" / aslug / arm
            if all((arm_dir / f"{p['stem']}.png").exists() for p in prompts):
                print(f"[{aslug}/{arm}] renders present — skipping", flush=True)
                continue
            # Launch names may use a shortened artist slug (e.g. e4_channel_*
            # for channel_(caststation)) — try full slug, then the first token.
            candidates = [
                REPO
                / "output"
                / "ckpt"
                / f"e4_{s}_{arm}_s{args.train_seed}.safetensors"
                for s in dict.fromkeys((aslug, aslug.split("_")[0]))
            ]
            ckpt = next((c for c in candidates if c.exists()), None)
            if ckpt is None:
                raise FileNotFoundError(f"missing arm checkpoint, tried {candidates}")
            lats = render_arm(args, f"{aslug}/{arm}", ckpt, prompts, device)

            ckpts = default_checkpoints()
            vae = load_vae(
                ckpts.vae,
                device="cpu",
                disable_mmap=True,
                dtype=torch.bfloat16,
                eval=True,
            )
            arm_dir.mkdir(parents=True, exist_ok=True)
            for p, lat in zip(prompts, lats):
                px = decode_latent(vae, lat, device).to("cpu")
                if px.dim() == 4:
                    px = px[0]
                arr = ((px.float().clamp(-1, 1) + 1) * 127.5).byte()
                Image.fromarray(arr.permute(1, 2, 0).numpy()).save(
                    arm_dir / f"{p['stem']}.png"
                )
            del vae, lats
            clean_memory_on_device(device)

        # Phase 3 — PE-Core pools from the saved PNGs + the two real pools.
        bundle = load_pe_encoder(device)
        gen_pools = {}
        for arm in args.arms:
            arm_dir = out / "renders" / aslug / arm
            imgs = []
            for p in prompts:
                with Image.open(arm_dir / f"{p['stem']}.png") as im:
                    imgs.append(_pil_to_minus1to1(im))
            gen_pools[arm] = encode_pool(bundle, imgs)
        ref_hold = real_pool(bundle, artist, cell["holdout"])
        ref_memb = real_pool(bundle, artist, cell["train"])
        del bundle
        clean_memory_on_device(device)

        half = len(ref_hold) // 2
        floor = cmmd_from_pools(ref_hold[:half], ref_hold[half:])

        arms_out: dict = {}
        for arm, pool in gen_pools.items():
            arms_out[arm] = {
                "cmmd_holdout": float(cmmd_from_pools(ref_hold, pool)),
                "cmmd_member": float(cmmd_from_pools(ref_memb, pool)),
            }

        # Paired PE-Core cosine at matched (prompt, gen_seed); pools are
        # unit-norm so the dot IS the cosine.
        paired: dict = {}
        arm_list = list(gen_pools)
        for i, a in enumerate(arm_list):
            for b in arm_list[i + 1 :]:
                cos = (gen_pools[a] * gen_pools[b]).sum(dim=1)
                paired[f"{a}~{b}"] = {
                    "mean": float(cos.mean()),
                    "min": float(cos.min()),
                    "per_prompt": [round(float(c), 4) for c in cos],
                }

        result["cells"][artist] = {
            "n_prompts": len(prompts),
            "n_ref_holdout": len(ref_hold),
            "n_ref_member": len(ref_memb),
            "noise_floor": float(floor),
            "arms": arms_out,
            "paired_cosine": paired,
        }
        print(f"[{aslug}] floor={floor:.4f} " + json.dumps(arms_out), flush=True)

    (out / "result.json").write_text(json.dumps(result, indent=1))

    lines = ["# E4 eval (exercise pass)", ""]
    for artist, c in result["cells"].items():
        lines += [
            f"## {artist}",
            "",
            f"- prompts {c['n_prompts']} · refs {c['n_ref_holdout']}/"
            f"{c['n_ref_member']} (holdout/member) · noise floor "
            f"**{c['noise_floor']:.4f}**",
            "",
            "| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |",
            "|---|---|---|---|",
        ]
        for arm, m in c["arms"].items():
            lines.append(
                f"| {arm} | {m['cmmd_holdout']:.4f} | {m['cmmd_member']:.4f} | "
                f"{m['cmmd_member'] - m['cmmd_holdout']:+.4f} |"
            )
        lines += ["", "| pair | paired cos (mean / min) |", "|---|---|"]
        for pair, pc in c["paired_cosine"].items():
            lines.append(f"| {pair} | {pc['mean']:.4f} / {pc['min']:.4f} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
