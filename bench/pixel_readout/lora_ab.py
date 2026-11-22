#!/usr/bin/env python3
"""Renders and pair sheets for a LoRA A/B (first run on the pixel-LPIPS arms).

Each arm's LoRA renders the ``docs/anima/findings/lora_recipe.md`` prompts p0 – p3 (the
artist tag filled in) at ``--seeds``, at the ``make test`` recipe: 28 steps Euler, CFG 4,
shift 3, 1024², blocks compiled. Per treatment arm and (prompt, seed) one sheet of two
columns, control and treatment: ``labelled/<arm>/`` with the arm names, and, with
``--blind_sets`` (one set name per arm), a blind set in the ``anima-blind-pairs`` checkout
in the layout of the lora_recipe A/Bs: ``sets/<name>/pNN.webp`` with the sides shuffled
and marked A / B, a ``README.md`` that scrolls through them and an empty ``verdicts.tsv``;
the key goes to ``output/blind/<name>/key.json``. Nothing is committed or pushed here. The
envelope also takes each file's median ‖(alpha / rank)·up @ down‖.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/lora_ab.py --label lpab_mmdd_hhmm --control lpctl_1121_1900 --arms lpr03_1121_1900 lpr10_1121_1900 --blind_sets s45_LPIPS03_ama s45_LPIPS10_ama"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from bench._common import make_run_dir, start_heartbeat, write_result
from library.env import resolve_under_home

BODIES = [
    "safe, 1girl, solo, {a}, blonde hair, school uniform, classroom, sitting at desk, smile, looking at viewer",
    "safe, 1boy, 1girl, {a}, hetero, hug, indoors, curtains",
    "safe, 1girl, solo, {a}, maid, holding a tray, cafe interior",
    "sensitive, 1girl, original, {a}, black bikini, black hair, short hair, innertube, beach, day, blue sky, looking at viewer, standing",
]
INFER = [
    "inference.py",
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
    "--compile_blocks",
]


def seed_of(p: int, s: int) -> int:
    return 1000 * (p + 1) + s


def by_seed(out: Path) -> dict[int, Path]:
    """inference.py names a render ``<time>_<seed>_.png``."""
    found = {}
    for f in out.glob("*.png"):
        m = re.search(r"_(\d+)_?$", f.stem)
        if m:
            found[int(m.group(1))] = f
    return found


def render(lora: Path, out: Path, prompts: list[str], seeds: list[int]):
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"{pr} --d {seed_of(i, s)}" for i, pr in enumerate(prompts) for s in seeds]
    (out / "prompts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [sys.executable, *INFER, "--lora_weight", str(lora)]
    cmd += ["--from_file", str(out / "prompts.txt"), "--save_path", str(out)]
    print(f"render {lora.name}", flush=True)
    subprocess.run(cmd, check=True, cwd=resolve_under_home("."))
    return by_seed(out)


def dw_norm(path: Path) -> float:
    """Median over modules of ‖(alpha / rank) · up @ down‖."""
    sd = load_file(str(path))
    norms = []
    for k, up in sd.items():
        if not k.endswith(".lora_up.weight"):
            continue
        down = sd[k.replace(".lora_up.", ".lora_down.")].float().flatten(1)
        alpha = sd.get(k.replace(".lora_up.weight", ".alpha"))
        scale = float(alpha) / down.shape[0] if alpha is not None else 1.0
        norms.append((scale * up.float().flatten(1) @ down).norm().item())
    return float(torch.tensor(norms).median())


def pair(left: Path, right: Path, names: tuple[str, str] | None, path: Path) -> None:
    a, b = Image.open(left).convert("RGB"), Image.open(right).convert("RGB")
    top = 24 if names else 0
    canvas = Image.new("RGB", (a.width + b.width + 8, top + a.height), "white")
    canvas.paste(a, (0, top))
    canvas.paste(b, (a.width + 8, top))
    if names:
        d = ImageDraw.Draw(canvas)
        d.text((6, 4), names[0], fill="black")
        d.text((a.width + 14, 4), names[1], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def blind_pair(a: Path, b: Path, out: Path) -> None:
    """The blind repo's sheet: the two renders side by side under an A and a B."""
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    gap, h = 24, max(ia.height, ib.height)
    sheet = Image.new("RGB", (ia.width + ib.width + gap, h + 40), "white")
    sheet.paste(ia, (0, 40))
    sheet.paste(ib, (ia.width + gap, 40))
    d = ImageDraw.Draw(sheet)
    d.text((ia.width // 2 - 8, 12), "A", fill="black")
    d.text((ia.width + gap + ib.width // 2 - 8, 12), "B", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90, method=6)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label", default=None)
    p.add_argument("--control", required=True, help="output_name of the control LoRA")
    p.add_argument("--arms", nargs="+", required=True, help="treatment output_names")
    p.add_argument("--ckpt_dir", default="output/ckpt")
    p.add_argument("--artist", default="@ama mitsuki")
    p.add_argument("--seeds", default="42,43,44,45")
    p.add_argument(
        "--blind_sets",
        nargs="*",
        default=None,
        help="one set name per arm, written under <blind_repo>/sets/",
    )
    p.add_argument("--blind_repo", default="../anima-blind-pairs")
    p.add_argument("--seed", type=int, default=0, help="shuffles the blind sides")
    args = p.parse_args()
    start_heartbeat(label="lora_ab")
    seeds = [int(s) for s in args.seeds.split(",")]
    prompts = [b.format(a=args.artist) for b in BODIES]
    ckpt = Path(resolve_under_home(args.ckpt_dir))
    run_dir = make_run_dir("pixel_readout", args.label)
    sets = args.blind_sets or []
    assert not sets or len(sets) == len(args.arms), "--blind_sets: one name per arm"
    repo = Path(resolve_under_home(args.blind_repo))
    for name in sets:
        assert not (repo / "sets" / name).exists(), f"blind set {name} exists"

    names = [args.control, *args.arms]
    files = {n: ckpt / f"{n}.safetensors" for n in names}
    for f in files.values():
        assert f.exists(), f"no LoRA {f}"
    shots = {
        n: render(files[n], run_dir / "renders" / n, prompts, seeds) for n in names
    }
    dw = {n: round(dw_norm(files[n]), 5) for n in names}

    rng = random.Random(args.seed)
    for k, arm in enumerate(args.arms):
        cells = [(i, sd) for i in range(len(prompts)) for sd in seeds]
        for i, sd in cells:
            rs = seed_of(i, sd)
            pair(
                shots[args.control][rs],
                shots[arm][rs],
                ("control", arm),
                run_dir / "labelled" / arm / f"p{i}_s{sd}.png",
            )
        if not sets:
            continue
        name, out = sets[k], repo / "sets" / sets[k]
        rng.shuffle(cells)
        key = {}
        readme = [
            f"# {name}\n",
            "Same prompt, same seed, two different training recipes. Which side is "
            "better overall (style fidelity to the artist, anatomy, stray dataset "
            "artifacts such as speech bubbles)? Fill `verdicts.tsv` (A / B / tie / "
            "skip) and commit. Sides are shuffled per pair; the key is not in this repo.\n",
        ]
        for n, (i, sd) in enumerate(cells, 1):
            pid = f"p{n:02d}"
            left, right = rng.sample([args.control, arm], 2)
            rs = seed_of(i, sd)
            blind_pair(shots[left][rs], shots[right][rs], out / f"{pid}.webp")
            key[pid] = {"A": left, "B": right, "prompt": i, "render_seed": sd}
            readme.append(f"## {pid}\n\n`{prompts[i]}`\n\n![{pid}]({pid}.webp)\n")
        (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
        (out / "verdicts.tsv").write_text(
            "pair\tverdict\tnote\n" + "".join(f"{q}\t\t\n" for q in key),
            encoding="utf-8",
        )
        kdir = Path(resolve_under_home("output/blind")) / name
        kdir.mkdir(parents=True, exist_ok=True)
        (kdir / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
        print(f"{name}: {len(key)} pairs -> {out}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "dw_median": dw,
            "dw_ratio_to_control": {
                n: round(dw[n] / dw[args.control], 4) for n in args.arms
            },
            "prompts": prompts,
            "render_seeds": [seed_of(i, s) for i in range(len(prompts)) for s in seeds],
            "recipe": "28 steps euler, cfg 4, shift 3, 1024x1024, compile_blocks",
            "blind_sets": dict(zip(sets, args.arms)),
        },
        label=args.label,
    )
    print(json.dumps({"dw_median": dw}, indent=1))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
