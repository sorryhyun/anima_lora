#!/usr/bin/env python3
"""E25d RUN + READ — truthful low-res inference labeling (see README.md).

Per Stage-2 rescond checkpoint (2 corpora x 3 seeds), per frozen SFW paper
prompt, three arms in one boot:

  * ``ref1024``  — native free-fit bucket, s = 0
  * ``plain896`` — the trained 896 sibling grid (``demote_bucket_for``), s = 0
  * ``cond896``  — same grid, s = -0.193 (the truthful trained label)

plain896/cond896 share latent shape + seed => identical init noise; the only
difference is the res-cond delta. 25d-1 gate: paired dcos vs ref1024, median
per ckpt, TRANSFER at >=5/6 positive (frozen in README before any render).

Usage (GPU — submit via daemon)::

    make daemon-run ARGS="project/sigma_lowres/paper_bench/experiments/e25d/e25d_render_read.py"

Resumable within one boot (boot-id stamped; a reboot discards latents).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER_BENCH = HERE.parents[1]
RUNS = PAPER_BENCH / "runs"
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from anima_lora import (  # noqa: E402
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_vae,
)
from library.datasets.buckets import demote_bucket_for  # noqa: E402
from library.inference import load_shared_models  # noqa: E402
from library.inference.models import load_dit_model  # noqa: E402
from library.inference.output import decode_latent  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402
from library.training.cmmd import pool_and_normalize  # noqa: E402
from library.vision.encoder import (  # noqa: E402
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

S_TRUTHFUL = -0.193
ARMS = ("ref1024", "plain896", "cond896")
HIGH_BAND_FNORM = 0.5  # frozen descriptive band: f/f_nyquist >= 0.5
CELLS = {  # corpus short name -> (prompt cell key, ckpt corpus token)
    "hews": "hews",
    "channel": "channel_(caststation)",
}
SEEDS = (1001, 1002, 1003)


def slug(artist: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", artist).strip("_")


def arm_size(arm: str, p: dict) -> tuple[int, int]:
    """(H, W) render size for an arm; hard-fails off-band prompts."""
    w, h = int(p["width"]), int(p["height"])
    if arm == "ref1024":
        return (h, w)
    bucket = demote_bucket_for(w, h, 1024, 896)
    if bucket is None:
        raise RuntimeError(f"prompt {p['stem']} ({w}x{h}) not in the 1024 band")
    bw, bh = bucket
    return (bh, bw)


def _pil_to_minus1to1(img: Image.Image) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(img.convert("RGB"))).float()
    return (t / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)


def high_band_energy(img: Image.Image) -> float:
    """Mean log power at normalized radial frequency >= HIGH_BAND_FNORM."""
    g = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
    f = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    p = np.abs(f) ** 2
    hh, ww = p.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    r = np.sqrt(((yy - hh / 2) / (hh / 2)) ** 2 + ((xx - ww / 2) / (ww / 2)) ** 2)
    mask = (r >= HIGH_BAND_FNORM) & (r <= 1.0)
    return float(np.log10(p[mask].mean() + 1e-12))


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
        out = RUNS / f"{stamp}-e25d"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Boot guard: renders compare within one boot only.
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    boot_file = out / "boot_id"
    if boot_file.exists() and boot_file.read_text().strip() != boot:
        raise RuntimeError(
            f"{out} holds latents from a previous boot — wipe the run dir "
            "and resubmit (no cross-boot render comparisons)."
        )
    boot_file.write_text(boot)

    pdata = json.loads(Path(args.prompts).read_text())
    cells = {short: pdata["cells"][full]["prompts"] for short, full in CELLS.items()}

    ckpts = [
        (
            short,
            seed,
            REPO / "output" / "ckpt" / f"e25b2_{short}_rescond_s{seed}.safetensors",
        )
        for short in CELLS
        for seed in SEEDS
    ]
    for _, _, path in ckpts:
        if not path.exists():
            raise FileNotFoundError(path)

    # ---- Phase 1: latents (one DiT load per ckpt; s mutated between arms) ----
    base_ckpt = default_checkpoints()
    for short, seed, ckpt_path in ckpts:
        prompts = cells[short]
        cell_dir = out / "latents" / f"{short}_s{seed}"
        pending = [
            (arm, p)
            for arm in ARMS
            for p in prompts
            if not (cell_dir / arm / f"{p['stem']}.pt").exists()
        ]
        if not pending:
            print(f"[{short}_s{seed}] latents present — skipping", flush=True)
            continue
        base_req = GenerationRequest(
            dit=base_ckpt.dit,
            vae=base_ckpt.vae,
            text_encoder=base_ckpt.text_encoder,
            prompt="placeholder",
            save_path="__unused__",
            infer_steps=args.steps,
            guidance_scale=args.cfg,
            seed=0,
            lora_weight=[str(ckpt_path)],
            lora_multiplier=[1.0],
            extra_argv=("--compile_blocks",) if args.compile_blocks else (),
        )
        base_args = base_req.to_args()
        base_args.device = device
        gen_settings = get_generation_settings(base_args)
        shared_models = load_shared_models(base_args)
        shared_models["conds_cache"] = {}
        model = load_dit_model(base_args, device, torch.bfloat16)
        shared_models["model"] = model
        rc = getattr(model, "_sigma_lowres_res_cond", None)
        if rc is None:
            raise RuntimeError(f"{ckpt_path}: res-cond projection did not attach")
        proj = rc[0]

        for i, (arm, p) in enumerate(pending):
            # Same eager (proj, s) seam the trainer mutates per-step.
            model._sigma_lowres_res_cond = (
                proj,
                S_TRUTHFUL if arm == "cond896" else 0.0,
            )
            pa = dataclasses.replace(
                base_req,
                prompt=p["caption"],
                seed=int(p["gen_seed"]),
                image_size=arm_size(arm, p),
            ).to_args()
            pa.device = device
            print(
                f"[{short}_s{seed}] {i + 1}/{len(pending)} {arm} {p['stem']}",
                flush=True,
            )
            lat = generate(pa, gen_settings, shared_models).detach().to("cpu")
            lat_path = cell_dir / arm / f"{p['stem']}.pt"
            lat_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(lat, lat_path)

        shared_models.clear()
        del gen_settings, model
        clean_memory_on_device(device)

    # ---- Phase 2: decode PNGs (DiT freed first) ----
    vae = None
    for short, seed, _ in ckpts:
        for arm in ARMS:
            for p in cells[short]:
                png = out / "renders" / f"{short}_s{seed}" / arm / f"{p['stem']}.png"
                if png.exists():
                    continue
                if vae is None:
                    vae = load_vae(
                        base_ckpt.vae,
                        device="cpu",
                        disable_mmap=True,
                        dtype=torch.bfloat16,
                        eval=True,
                    )
                lat = torch.load(
                    out / "latents" / f"{short}_s{seed}" / arm / f"{p['stem']}.pt",
                    weights_only=True,
                )
                px = decode_latent(vae, lat, device).to("cpu")
                if px.dim() == 4:
                    px = px[0]
                arr = ((px.float().clamp(-1, 1) + 1) * 127.5).byte()
                png.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr.permute(1, 2, 0).numpy()).save(png)
    if vae is not None:
        del vae
        clean_memory_on_device(device)

    # ---- Phase 3: 25d-1 read (paired PE-cos) + spectral descriptive ----
    bundle = load_pe_encoder(device)

    def pe_vec(png: Path, bundle=bundle) -> torch.Tensor:
        with Image.open(png) as im:
            feats = encode_pe_from_imageminus1to1(
                bundle, _pil_to_minus1to1(im).to(bundle.device)
            )
        return pool_and_normalize(feats[0]).to("cpu")

    per_ckpt: dict = {}
    for short, seed, _ in ckpts:
        key = f"{short}_s{seed}"
        rows = []
        for p in cells[short]:
            vecs = {
                arm: pe_vec(out / "renders" / key / arm / f"{p['stem']}.png")
                for arm in ARMS
            }
            cos_plain = float(vecs["plain896"] @ vecs["ref1024"])
            cos_cond = float(vecs["cond896"] @ vecs["ref1024"])
            hb = {
                arm: high_band_energy(
                    Image.open(out / "renders" / key / arm / f"{p['stem']}.png")
                )
                for arm in ARMS
            }
            rows.append(
                {
                    "stem": p["stem"],
                    "cos_plain896": cos_plain,
                    "cos_cond896": cos_cond,
                    "dcos": cos_cond - cos_plain,
                    "high_band": hb,
                }
            )
        dcs = [r["dcos"] for r in rows]
        per_ckpt[key] = {
            "median_dcos": statistics.median(dcs),
            "positive_prompts": sum(1 for d in dcs if d > 0),
            "n_prompts": len(dcs),
            "baseline_gap_median_cos_plain896": statistics.median(
                r["cos_plain896"] for r in rows
            ),
            "per_prompt": rows,
        }
        print(
            f"[{key}] median dcos {per_ckpt[key]['median_dcos']:+.4f} "
            f"({per_ckpt[key]['positive_prompts']}/{len(dcs)} prompts positive)",
            flush=True,
        )
    del bundle
    clean_memory_on_device(device)

    pos = [k for k, v in per_ckpt.items() if v["median_dcos"] > 0]
    neg = [k for k, v in per_ckpt.items() if v["median_dcos"] < 0]
    both_corpora_pos = any(k.startswith("hews") for k in pos) and any(
        k.startswith("channel") for k in pos
    )
    if len(pos) >= 5 and both_corpora_pos:
        verdict = "TRANSFER"
    elif len(neg) >= 5:
        verdict = "ANTI"
    else:
        verdict = "NULL"

    # ---- Phase 4: eyeball sheets per (corpus, seed) ----
    thumb_w, gap, header_h = 340, 6, 34
    try:
        font = ImageFont.load_default(size=20)
        small = ImageFont.load_default(size=14)
    except TypeError:
        font = small = ImageFont.load_default()
    labels = {
        "ref1024": "ref 1024 (s=0)",
        "plain896": "896 s=0 (false label)",
        "cond896": f"896 s={S_TRUTHFUL} (truthful)",
    }
    for short, seed, _ in ckpts:
        key = f"{short}_s{seed}"
        rows_img = []
        for p in cells[short]:
            thumbs = []
            for arm in ARMS:
                with Image.open(out / "renders" / key / arm / f"{p['stem']}.png") as im:
                    h = round(im.height * thumb_w / im.width)
                    thumbs.append(im.resize((thumb_w, h), Image.LANCZOS))
            rows_img.append((p, thumbs))
        sheet_w = len(ARMS) * (thumb_w + gap) + gap
        sheet_h = header_h + sum(
            max(t.height for t in ts) + gap + 18 for _, ts in rows_img
        )
        sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
        draw = ImageDraw.Draw(sheet)
        for j, arm in enumerate(ARMS):
            draw.text(
                (gap + j * (thumb_w + gap), 7), labels[arm], fill="black", font=font
            )
        y = header_h
        for p, thumbs in rows_img:
            draw.text(
                (gap, y + 2),
                f"{p['stem']}  gen_seed={p['gen_seed']}",
                fill="gray",
                font=small,
            )
            y += 18
            for j, t in enumerate(thumbs):
                sheet.paste(t, (gap + j * (thumb_w + gap), y))
            y += max(t.height for t in thumbs) + gap
        sheet.save(out / f"sheet_{key}.png")

    result = {
        "experiment": "e25d",
        "boot_id": boot,
        "prompts": str(args.prompts),
        "steps": args.steps,
        "cfg": args.cfg,
        "s_truthful": S_TRUTHFUL,
        "high_band_fnorm": HIGH_BAND_FNORM,
        "verdict_25d1": {
            "verdict": verdict,
            "positive_ckpts": pos,
            "negative_ckpts": neg,
            "both_corpora_among_positives": both_corpora_pos,
            "gate": "median dcos > 0 on >=5/6 ckpts (both corpora among "
            "positives) => TRANSFER; >=5/6 negative => ANTI; else NULL",
        },
        "per_ckpt": per_ckpt,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(f"25d-1 verdict: {verdict}  ({len(pos)}/6 positive)", flush=True)
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
