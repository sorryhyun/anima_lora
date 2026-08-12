#!/usr/bin/env python3
"""E25c eyeball (claim-free) — the rescond projection on the PLAIN BASE model.

Renders the frozen SFW paper prompts (`e4_prompts_sfw.json`) on the base DiT
with NO LoRA dW at all, sweeping the res-cond conditioning input
s in {-0.415, -0.193, 0, +0.193, +0.415} against a `plain` arm (no projection
attached), one contact sheet per corpus for the eyeball pass.

The projection comes from each corpus's Stage-2 rescond s1001 checkpoint but
its dW never loads — the (proj, s) tuple is installed directly on the DiT
(`model._sigma_lowres_res_cond`, the same seam `_attach_res_cond` uses; eager
region, so one shared compiled model serves every arm). Note `plain` is NOT
the same as `s0`: the trained W*phi(0) offset is part of what the rescond run
converged to, so on the base model even s=0 is a foreign perturbation.

Registration status: descriptive curiosity OUTSIDE the frozen E25c material —
the frozen sweep (rescond checkpoints rendered at s != 0) is deliberately NOT
touched here; no verdict, no read, eyeball sheets only.

Usage (GPU — submit via daemon)::

    make daemon-run ARGS="project/sigma_lowres/paper_bench/experiments/e25c/e25c_eyeball_base.py"

Resumable: latents then PNGs are skipped when already on disk.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER_BENCH = HERE.parents[1]  # project/sigma_lowres/paper_bench
RUNS = PAPER_BENCH / "runs"
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from safetensors import safe_open  # noqa: E402

from anima_lora import (  # noqa: E402
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_vae,
)
from library.inference import load_shared_models  # noqa: E402
from library.inference.models import load_dit_model  # noqa: E402
from library.inference.output import decode_latent  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

S_VALUES = (-0.415, -0.193, 0.0, 0.193, 0.415)
CARRIERS = {
    "hews": "e25b2_hews_rescond_s1001.safetensors",
    "channel_(caststation)": "e25b2_channel_rescond_s1001.safetensors",
}


def slug(artist: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", artist).strip("_")


def arm_name(s: float | None) -> str:
    return "plain" if s is None else f"s{s:+.3f}"


def load_proj(path: Path) -> torch.Tensor:
    with safe_open(str(path), framework="pt") as f:
        if "sigma_lowres_res_cond_proj" not in f.keys():
            raise RuntimeError(f"{path}: no sigma_lowres_res_cond_proj tensor")
        stamped = dict(f.metadata() or {}).get("ss_sigma_lowres_res_cond") == "true"
        if not stamped:
            raise RuntimeError(f"{path}: missing ss_sigma_lowres_res_cond stamp")
        return f.get_tensor("sigma_lowres_res_cond_proj")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--prompts",
        default=str(RUNS / "20260729-1537-e4-manifest" / "e4_prompts_sfw.json"),
    )
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument(
        "--compile_blocks", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--out", default=None, help="Run dir (default: stamped sibling).")
    args = ap.parse_args()

    if args.out:
        out = Path(args.out)
    else:
        stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
        out = RUNS / f"{stamp}-e25c-eyeball-base"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pdata = json.loads(Path(args.prompts).read_text())
    cells = {a: c["prompts"] for a, c in pdata["cells"].items() if a in CARRIERS}
    projs = {a: load_proj(REPO / "output" / "ckpt" / CARRIERS[a]) for a in cells}
    arms = [None, *S_VALUES]  # None = plain

    # ---- Phase 1: latents (one shared plain-base DiT, projection toggled) ----
    todo = [
        (artist, s, p)
        for artist, prompts in cells.items()
        for s in arms
        for p in prompts
        if not (
            out / "latents" / slug(artist) / arm_name(s) / f"{p['stem']}.pt"
        ).exists()
    ]
    if todo:
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
            extra_argv=("--compile_blocks",) if args.compile_blocks else (),
        )
        base_args = base_req.to_args()
        base_args.device = device
        gen_settings = get_generation_settings(base_args)
        shared_models = load_shared_models(base_args)
        shared_models["conds_cache"] = {}
        model = load_dit_model(base_args, device, torch.bfloat16)
        shared_models["model"] = model

        for i, (artist, s, p) in enumerate(todo):
            if s is None:
                if hasattr(model, "_sigma_lowres_res_cond"):
                    del model._sigma_lowres_res_cond
            else:
                model._sigma_lowres_res_cond = (
                    projs[artist].to(device=device, dtype=torch.float32),
                    float(s),
                )
            pa = dataclasses.replace(
                base_req,
                prompt=p["caption"],
                seed=int(p["gen_seed"]),
                image_size=(int(p["height"]), int(p["width"])),
            ).to_args()
            pa.device = device
            print(
                f"[{i + 1}/{len(todo)}] {slug(artist)}/{arm_name(s)} {p['stem']}",
                flush=True,
            )
            lat = generate(pa, gen_settings, shared_models).detach().to("cpu")
            lat_path = out / "latents" / slug(artist) / arm_name(s) / f"{p['stem']}.pt"
            lat_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(lat, lat_path)

        shared_models.clear()
        del gen_settings, model
        clean_memory_on_device(device)

    # ---- Phase 2: decode PNGs (DiT freed first — 16GB card) ----
    pngs_todo = [
        (artist, s, p)
        for artist, prompts in cells.items()
        for s in arms
        for p in prompts
        if not (
            out / "renders" / slug(artist) / arm_name(s) / f"{p['stem']}.png"
        ).exists()
    ]
    if pngs_todo:
        vae = load_vae(
            default_checkpoints().vae,
            device="cpu",
            disable_mmap=True,
            dtype=torch.bfloat16,
            eval=True,
        )
        for artist, s, p in pngs_todo:
            lat = torch.load(
                out / "latents" / slug(artist) / arm_name(s) / f"{p['stem']}.pt",
                weights_only=True,
            )
            px = decode_latent(vae, lat, device).to("cpu")
            if px.dim() == 4:
                px = px[0]
            arr = ((px.float().clamp(-1, 1) + 1) * 127.5).byte()
            png_path = out / "renders" / slug(artist) / arm_name(s) / f"{p['stem']}.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr.permute(1, 2, 0).numpy()).save(png_path)
        del vae
        clean_memory_on_device(device)

    # ---- Phase 3: contact sheets (rows = prompts, cols = plain + s sweep) ----
    thumb_w, gap, header_h = 300, 6, 34
    try:
        font = ImageFont.load_default(size=20)
        small = ImageFont.load_default(size=14)
    except TypeError:  # older PIL
        font = small = ImageFont.load_default()
    for artist, prompts in cells.items():
        aslug = slug(artist)
        cols = [arm_name(s) for s in arms]
        rows = []
        for p in prompts:
            thumbs = []
            for c in cols:
                with Image.open(out / "renders" / aslug / c / f"{p['stem']}.png") as im:
                    h = round(im.height * thumb_w / im.width)
                    thumbs.append(im.resize((thumb_w, h), Image.LANCZOS))
            rows.append((p, thumbs))
        sheet_w = len(cols) * (thumb_w + gap) + gap
        sheet_h = header_h + sum(max(t.height for t in ts) + gap + 18 for _, ts in rows)
        sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
        draw = ImageDraw.Draw(sheet)
        for j, c in enumerate(cols):
            label = c + (" (trained pt)" if c == "s+0.000" else "")
            draw.text((gap + j * (thumb_w + gap), 7), label, fill="black", font=font)
        y = header_h
        for p, thumbs in rows:
            draw.text(
                (gap, y + 2),
                f"{p['stem']}  gen_seed={p['gen_seed']}  {p['width']}x{p['height']}",
                fill="gray",
                font=small,
            )
            y += 18
            row_h = max(t.height for t in thumbs)
            for j, t in enumerate(thumbs):
                sheet.paste(t, (gap + j * (thumb_w + gap), y))
            y += row_h + gap
        sheet.save(out / f"sheet_{aslug}.png")
        print(f"sheet: {out / f'sheet_{aslug}.png'}", flush=True)

    (out / "result.json").write_text(
        json.dumps(
            {
                "what": "E25c claim-free eyeball — rescond projection on the plain "
                "base DiT (no LoRA dW), SFW paper prompts, plain vs s sweep. "
                "Descriptive only; NOT the frozen E25c sweep (rescond ckpts at "
                "s!=0 deliberately unrendered).",
                "prompts": str(args.prompts),
                "carriers": {a: CARRIERS[a] for a in cells},
                "s_values": list(S_VALUES),
                "arms": [arm_name(s) for s in arms],
                "steps": args.steps,
                "cfg": args.cfg,
                "n_renders": sum(len(p) for p in cells.values()) * len(arms),
            },
            indent=1,
        )
    )
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
