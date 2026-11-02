#!/usr/bin/env python3
"""The reference_adapters A/B on the four swap_text eval pages
(``docs/moss/proposal/reference_adapters.md``).

``render``: one LoRA, same prompt, size, seed and sampler as the control renders (GPU,
through the daemon):

    make daemon-run ARGS="--stall-timeout 600 project/int8_train/src/ref_adapters_eval.py render --lora <lora> --name <results dir>"

``score``: mean |ΔL| (0-255) against the source outside the text boxes, per run, and
sheets of each text box at 2x (CPU, ``<out>/<page>.png``):

    .venv/bin/python project/int8_train/src/ref_adapters_eval.py score --out <dir> <label>=<results dir> ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RESULTS = Path("output/moss_i2i_eval/swap_text/results")
INPUTS = Path("output/moss_i2i_eval/swap_text/inputs")
CONTROL = RESULTS / "v3_en400_ko1709_e2_ema_bf16_fp32s"
PAGES = ("1483610_0007_en", "3124789_0042_en", "589883_0004_ko", "737486_0019_ko")
PAD, ZOOM = 24, 2


def render(lora: str, name: str) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from library.moss.generate import run_generate
    from library.moss.requests import GenerateRequest

    for page in PAGES:
        control = json.load(open(CONTROL / page / "manifest.json"))["images"][0]
        argv = [
            "--prompt",
            (INPUTS / f"{page}.txt").read_text().strip(),
            "--prompt_format",
            "text",
            "--lora",
            lora,
            "--width",
            str(control["width"]),
            "--height",
            str(control["height"]),
            "--seed",
            str(control["seed"]),
            "--out_dir",
            str(RESULTS / name / page),
            "--ref_image",
            str(INPUTS / f"{page}.src.webp"),
            "--stream",
            "fp32",
            "--blocks_to_swap",
            "10",
        ]
        run_generate(GenerateRequest.from_argv(argv))


def boxes(page: str, size: tuple[int, int]) -> list[tuple[str, tuple]]:
    w, h = size
    out = []
    for item in json.loads((INPUTS / f"{page}.txt").read_text())["items"]:
        y0, x0, y1, x1 = item["bbox"]
        out.append(
            (
                item["text"],
                (
                    max(0, x0 + w // 2 - PAD),
                    max(0, y0 + h // 2 - PAD),
                    min(w, x1 + w // 2 + PAD),
                    min(h, y1 + h // 2 + PAD),
                ),
            )
        )
    return out


def outside_delta_l(src: Image.Image, im: Image.Image, page: str) -> float:
    diff = np.abs(
        np.asarray(src.convert("L"), np.float32)
        - np.asarray(im.convert("L"), np.float32)
    )
    keep = np.ones(diff.shape, bool)
    for _, (x0, y0, x1, y1) in boxes(page, src.size):
        keep[y0:y1, x0:x1] = False
    return float(diff[keep].mean())


def score(runs: dict[str, Path], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 18)
    table = {}
    for page in PAGES:
        src = Image.open(INPUTS / f"{page}.src.webp").convert("RGB")
        images = {"src": src}
        for label, folder in runs.items():
            png = sorted((folder / page).glob("*.png"))[0]
            images[label] = Image.open(png).convert("RGB").resize(src.size)
            table.setdefault(label, {})[page] = outside_delta_l(
                src, images[label], page
            )
        rows = [
            (
                text,
                [
                    im.crop(box).resize(
                        ((box[2] - box[0]) * ZOOM, (box[3] - box[1]) * ZOOM),
                        Image.LANCZOS,
                    )
                    for im in images.values()
                ],
            )
            for text, box in boxes(page, src.size)
        ]
        cell = max(c.width for _, cs in rows for c in cs) + 8
        heights = [max(c.height for c in cs) + 38 for _, cs in rows]
        sheet = Image.new("RGB", (cell * len(images), 56 + sum(heights)), "white")
        d = ImageDraw.Draw(sheet)
        for i, label in enumerate(images):
            d.text((i * cell + 4, 4), label, fill="black", font=font)
        d.text((4, 28), page, fill="gray", font=font)
        y = 56
        for (text, crops), height in zip(rows, heights):
            d.text((4, y), text, fill="black", font=font)
            for i, crop in enumerate(crops):
                sheet.paste(crop, (i * cell + 4, y + 30))
            y += height
        sheet.save(out / f"{page}.png")
    (out / "outside_delta_l.json").write_text(json.dumps(table, indent=2))
    print("| run | " + " | ".join(PAGES) + " |")
    for label, row in table.items():
        print(f"| {label} | " + " | ".join(f"{row[p]:.1f}" for p in PAGES) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--lora", required=True)
    r.add_argument("--name", required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("runs", nargs="+", help="label=<results dir, under RESULTS>")
    args = ap.parse_args()
    if args.cmd == "render":
        render(args.lora, args.name)
    else:
        runs = dict(x.split("=", 1) for x in args.runs)
        score({k: RESULTS / v for k, v in runs.items()}, Path(args.out))


if __name__ == "__main__":
    main()
