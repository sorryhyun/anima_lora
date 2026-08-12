#!/usr/bin/env python3
"""E25e knob eyeball (claim-free) — the TRAINED centered checkpoint under
the E25c-style ``--res_cond_s`` sweep.

Registration status: the E25e kill switches allow exactly this as a
descriptive render ("a centered checkpoint under the E25c-style
``--res_cond_s`` knob sweeps the differential axis only — descriptive if
ever rendered, own registration before any claim"). No verdict, no read,
eyeball sheets only. E25c's frozen sweep (25b checkpoints) is untouched.

What it does: for one rescond_c carrier per corpus (the seed with the
largest learned differential norm), renders 5 frozen SFW paper prompts
(the contact-sheet subset) at 1024-native with the full LoRA loaded and
the conditioning input swept over s in {-0.415, -0.193, 0, +0.193,
+0.415}, plus a ``plain`` arm with the projection detached. For a
centered checkpoint plain and s=0 must be bit-identical (delta and
gradient exactly zero at s=0 for any projection) — asserted in vivo on
the latents and recorded in ``identity_check.json``. One contact sheet
per corpus: rows = prompts, cols = plain + s sweep.

Usage (GPU — submit via daemon)::

    make daemon-run ARGS="project/sigma_lowres/paper_bench/experiments/e25e/e25e_knob_eyeball.py"

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
PAPER_BENCH = HERE.parents[1]
RUNS = PAPER_BENCH / "runs"
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from anima_lora import (  # noqa: E402
    GenerationRequest,
    default_checkpoints,
    get_generation_settings,
    generate,
    load_vae,
)
from library.inference import load_shared_models  # noqa: E402
from library.inference.models import load_dit_model  # noqa: E402
from library.inference.output import decode_latent  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

S_VALUES = (-0.415, -0.193, 0.0, 0.193, 0.415)
# carrier = the rescond_c seed with the largest learned differential norm
# (e25e_read.json descriptives: hews s1001 0.077 @768, channel s1003 0.037)
CARRIERS = {
    "hews": "e25e_hews_rescond_c_s1001.safetensors",
    "channel_(caststation)": "e25e_channel_rescond_c_s1003.safetensors",
}
N_PROMPTS = 5  # evenly spaced over the frozen prompt list (contact-sheet subset)


def slug(artist: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", artist).strip("_")


def arm_name(s: float | None) -> str:
    return "plain" if s is None else f"s{s:+.3f}"


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
        out = RUNS / f"{stamp}-e25e-knob-eyeball"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pdata = json.loads(Path(args.prompts).read_text())
    cells = {}
    for artist, cell in pdata["cells"].items():
        if artist not in CARRIERS:
            continue
        prompts = cell["prompts"]
        idx = [
            round(i * (len(prompts) - 1) / (N_PROMPTS - 1)) for i in range(N_PROMPTS)
        ]
        cells[artist] = [prompts[i] for i in idx]
    arms = [None, *S_VALUES]  # None = plain (projection detached)

    # ---- Phase 1: latents (one LoRA-loaded DiT per corpus, s toggled) ------
    for artist, prompts in cells.items():
        aslug = slug(artist)
        todo = [
            (s, p)
            for s in arms
            for p in prompts
            if not (out / "latents" / aslug / arm_name(s) / f"{p['stem']}.pt").exists()
        ]
        if not todo:
            continue
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
            lora_weight=[str(REPO / "output" / "ckpt" / CARRIERS[artist])],
            extra_argv=("--compile_blocks",) if args.compile_blocks else (),
        )
        base_args = base_req.to_args()
        base_args.device = device
        gen_settings = get_generation_settings(base_args)
        shared_models = load_shared_models(base_args)
        shared_models["conds_cache"] = {}
        model = load_dit_model(base_args, device, torch.bfloat16)
        shared_models["model"] = model
        attached = model._sigma_lowres_res_cond  # (proj, 0.0, centered=True)
        assert attached[2], f"{CARRIERS[artist]}: expected a centered stamp"

        for i, (s, p) in enumerate(todo):
            if s is None:
                if hasattr(model, "_sigma_lowres_res_cond"):
                    del model._sigma_lowres_res_cond
            else:
                model._sigma_lowres_res_cond = (attached[0], float(s), True)
            pa = dataclasses.replace(
                base_req,
                prompt=p["caption"],
                seed=int(p["gen_seed"]),
                image_size=(int(p["height"]), int(p["width"])),
            ).to_args()
            pa.device = device
            print(
                f"[{aslug} {i + 1}/{len(todo)}] {arm_name(s)} {p['stem']}", flush=True
            )
            lat = generate(pa, gen_settings, shared_models).detach().to("cpu")
            lat_path = out / "latents" / aslug / arm_name(s) / f"{p['stem']}.pt"
            lat_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(lat, lat_path)

        shared_models.clear()
        del gen_settings, model
        clean_memory_on_device(device)

    # ---- Phase 1b: centered identity check — plain must equal s+0.000 ------
    ident = {}
    for artist, prompts in cells.items():
        aslug = slug(artist)
        for p in prompts:
            a = torch.load(
                out / "latents" / aslug / "plain" / f"{p['stem']}.pt",
                weights_only=True,
            )
            b = torch.load(
                out / "latents" / aslug / "s+0.000" / f"{p['stem']}.pt",
                weights_only=True,
            )
            ident[f"{aslug}/{p['stem']}"] = {
                "bit_identical": bool(torch.equal(a, b)),
                "max_abs_diff": float((a - b).abs().max()),
            }
    (out / "identity_check.json").write_text(json.dumps(ident, indent=1))
    n_ok = sum(v["bit_identical"] for v in ident.values())
    print(f"centered identity plain==s0: {n_ok}/{len(ident)} bit-identical", flush=True)

    # ---- Phase 2: decode PNGs (DiT freed first — 16GB card) ----------------
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

    # ---- Phase 3: contact sheets (rows = prompts, cols = plain + sweep) ----
    thumb_w, gap, header_h = 340, 8, 40
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
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
            rows.append((p["stem"], thumbs))
        row_h = max(t.height for _, ts in rows for t in ts)
        W = len(cols) * (thumb_w + gap) + gap
        H = header_h + len(rows) * (row_h + gap) + gap
        sheet = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(sheet)
        for j, c in enumerate(cols):
            d.text(
                (gap + j * (thumb_w + gap) + thumb_w // 2, header_h // 2),
                c,
                fill="black",
                font=font,
                anchor="mm",
            )
        for ri, (stem, thumbs) in enumerate(rows):
            y = header_h + gap + ri * (row_h + gap)
            for j, t in enumerate(thumbs):
                sheet.paste(t, (gap + j * (thumb_w + gap), y))
            d.text((gap + 4, y + 4), stem, fill="red", font=font)
        sheet_path = out / f"{aslug}_knob_sweep.jpg"
        sheet.save(sheet_path, quality=92)
        print(f"wrote {sheet_path}", flush=True)


if __name__ == "__main__":
    main()
