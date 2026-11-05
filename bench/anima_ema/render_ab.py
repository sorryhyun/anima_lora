#!/usr/bin/env python3
"""Adapter EMA A/B on Anima: one training run's raw weights against its EMA spans.

The run (``ema_window`` + ``ema_extra``) writes the main EMA file, the raw final
weights at ``<name>/<name>-raw`` (with ``ema_save_raw``) and each extra span at ``<name>/<name>-ema<span>``,
so every arm comes from one trajectory. ``--ref`` adds other LoRAs as columns.
Each arm renders every prompt x seed in one ``inference.py --from_file`` load at the
``make test`` recipe; one grid per prompt (rows = seeds, columns = arms).

    make daemon-run ARGS="--stall-timeout 900 bench/anima_ema/render_ab.py \
        --run output/ckpt/anima_lora_ama_ema5e5.safetensors \
        --ref ama4_2e-5=output/ckpt/anima_lora_ama4.safetensors"
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from bench._common import make_run_dir, write_result
from library.env import resolve_under_home

PROMPTS = [
    "safe, 1girl, solo, @ama mitsuki, blonde hair, school uniform, classroom, sitting at desk, smile, looking at viewer",
    "safe, 1boy, 1girl, @ama mitsuki, hetero, hug, indoors, curtains",
    "safe, 1girl, solo, @ama mitsuki, maid, holding a tray, cafe interior",
    "safe, 1girl, solo, @ama mitsuki, upper body, portrait, simple background",
    "safe, 1girl, solo, @ama mitsuki, cat ears, hoodie, night, city street, neon lights, from side",
    "sensitive, 1girl, original, @ama mitsuki, black bikini, black hair, short hair, innertube, beach, day, blue sky, looking at viewer, standing",
]
SEEDS = [1, 2]

INFER = [
    "inference.py",
    "--dit",
    "models/diffusion_models/anima-base-v1.0.safetensors",
    "--text_encoder",
    "models/text_encoders/qwen_3_06b_base.safetensors",
    "--vae",
    "models/vae/qwen_image_vae.safetensors",
    "--vae_chunk_size",
    "64",
    "--vae_disable_cache",
    "--attn_mode",
    "flash",
    "--lora_multiplier",
    "1.0",
    "--negative_prompt",
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
    "--image_size",
    "1024",
    "1024",
    "--infer_steps",
    "28",
    "--flow_shift",
    "3.0",
    "--sampler",
    "euler",
    "--guidance_scale",
    "4.0",
]


def run_arms(run: Path) -> dict[str, Path]:
    stem, side = run.stem, run.parent / run.stem
    raw = side / f"{stem}-raw{run.suffix}"
    arms = {"raw": raw} if raw.exists() else {}
    arms["ema_main"] = run
    for f in sorted(side.glob(f"{stem}-ema*{run.suffix}")):
        arms[f"ema{f.stem.rsplit('-ema', 1)[1]}"] = f
    return arms


def _deltas(path: Path) -> dict[str, torch.Tensor]:
    """``up @ down`` per module; the alpha / rank scale is shared across arms."""
    sd = load_file(str(path))
    out = {}
    for k, up in sd.items():
        if k.endswith(".lora_up.weight"):
            down = sd[k.replace(".lora_up.", ".lora_down.")]
            out[k[: -len(".lora_up.weight")]] = up.float().flatten(
                1
            ) @ down.float().flatten(1)
    return out


def weight_delta(arms: dict[str, Path]) -> dict[str, dict[str, float]]:
    """Median and min over modules of ||dW_arm|| / ||dW_raw|| and cos(dW_arm, dW_raw)."""
    if "raw" not in arms:
        return {}
    raw = _deltas(arms["raw"])
    out = {}
    for arm, path in arms.items():
        if arm == "raw":
            continue
        d = _deltas(path)
        keys = [k for k in raw if k in d and raw[k].shape == d[k].shape]
        if not keys:
            continue
        ratio = torch.tensor(
            [d[k].norm() / raw[k].norm().clamp_min(1e-12) for k in keys]
        )
        cos = torch.tensor(
            [F.cosine_similarity(d[k].flatten(), raw[k].flatten(), dim=0) for k in keys]
        )
        out[arm] = {
            "norm_ratio_median": round(ratio.median().item(), 4),
            "cos_median": round(cos.median().item(), 4),
            "cos_min": round(cos.min().item(), 4),
        }
    return out


def seed_of(p: int, s: int) -> int:
    return 1000 * (p + 1) + s


def render(arm: str, lora: Path, out: Path) -> dict[int, Path]:
    out.mkdir(parents=True, exist_ok=True)
    done = _by_seed(out)
    if len(done) == len(PROMPTS) * len(SEEDS):
        print(f"[{arm}] reusing {out}", flush=True)
        return done
    lines = [f"{pr} --d {seed_of(i, s)}" for i, pr in enumerate(PROMPTS) for s in SEEDS]
    prompt_file = out / "prompts.txt"
    prompt_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        sys.executable,
        *INFER,
        "--lora_weight",
        str(lora),
        "--from_file",
        str(prompt_file),
        "--save_path",
        str(out),
    ]
    print(f"[{arm}] {lora}", flush=True)
    subprocess.run(cmd, check=True, cwd=resolve_under_home("."))
    return _by_seed(out)


def _by_seed(out: Path) -> dict[int, Path]:
    """inference.py names a render ``<time>_<seed>_.png``."""
    by_seed = {}
    for f in out.glob("*.png"):
        m = re.search(r"_(\d+)_?$", f.stem)
        if m:
            by_seed[int(m.group(1))] = f
    return by_seed


def grid(cells: list[list[Path]], cols: list[str], rows: list[str], path: Path):
    tile = 512
    pad = 28
    img = Image.new("RGB", (pad + tile * len(cols), pad + tile * len(rows)), "white")
    draw = ImageDraw.Draw(img)
    for c, name in enumerate(cols):
        draw.text((pad + c * tile + 6, 6), name, fill="black")
    for r, name in enumerate(rows):
        draw.text((2, pad + r * tile + 6), name, fill="black")
        for c, f in enumerate(cells[r]):
            im = Image.open(f).convert("RGB")
            im.thumbnail((tile, tile))
            img.paste(im, (pad + c * tile, pad + r * tile))
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="main EMA file of the run")
    ap.add_argument("--ref", action="append", default=[], help="name=path, repeatable")
    ap.add_argument(
        "--skip",
        default="",
        help="comma-separated arms not to render (weight_delta still reads them)",
    )
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--run_dir", default=None, help="existing results dir: reuse its finished arms"
    )
    args = ap.parse_args()

    run = resolve_under_home(args.run)
    arms = run_arms(run)
    for spec in args.ref:
        name, path = spec.split("=", 1)
        arms[name] = resolve_under_home(path)
    missing = [str(p) for p in arms.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing arms: {missing}")

    run_dir = (
        resolve_under_home(args.run_dir)
        if args.run_dir
        else make_run_dir("anima_ema", label=args.label or run.stem)
    )
    skip = {a.strip() for a in args.skip.split(",") if a.strip()}
    names = [a for a in arms if a not in skip]
    renders = {a: render(a, arms[a], run_dir / "renders" / a) for a in names}
    grids = []
    for i, prompt in enumerate(PROMPTS):
        cells = [[renders[a][seed_of(i, s)] for a in names] for s in SEEDS]
        path = run_dir / f"grid_p{i}.png"
        grid(cells, names, [f"s{s}" for s in SEEDS], path)
        grids.append(str(path))
    write_result(
        run_dir,
        script=__file__,
        args={
            **vars(args),
            "arms": {a: str(p) for a, p in arms.items()},
            "prompts": PROMPTS,
            "seeds": SEEDS,
            "recipe": "28 st euler, cfg 4, shift 3, 1024x1024",
        },
        metrics={"weight_delta": weight_delta(arms)},
        label=args.label or run.stem,
        artifacts=grids,
    )
    print(f"grids in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
