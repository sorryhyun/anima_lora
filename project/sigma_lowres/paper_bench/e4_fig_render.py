#!/usr/bin/env python3
"""E4 figure-candidate re-render — candidate stems only, sheet assembly.

Re-renders the chosen figure-candidate stems (default: the three from the
20260730-e4-fig-candidates pass) under every arm x train-seed checkpoint at a
chosen steps/cfg (default 30 / 1.0 — the steps-30 look requested for the
paper figures), then assembles one comparison sheet per stem:
columns = real (holdout) + arms, rows = train seeds (real only on row 1).

Prompts/sizes/gen_seeds come frozen from the SFW prompt amendment; rendering
machinery is e4_render_eval's (imported).

Usage (GPU — submit via daemon)::

    make daemon-run ARGS="project/sigma_lowres/paper_bench/e4_fig_render.py"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from e4_render_eval import RESIZED, render_arm, slug  # noqa: E402
from anima_lora import default_checkpoints, load_vae  # noqa: E402
from library.inference.output import decode_latent  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

DEFAULT_PROMPTS = (
    HERE / "runs" / "20260729-1537-e4-manifest" / "e4_prompts_sfw.json"
)
DEFAULT_STEMS = ("10607820", "14296235", "8508115")
ARMS = ("native", "sigma896", "896only", "unsafe768")

CELL_H = 410
LABEL_H = 24
SEED_H = 16
PAD = 6


def find_ckpt(aslug: str, arm: str, seed: int) -> Path:
    candidates = [
        REPO / "output" / "ckpt" / f"e4_{s}_{arm}_s{seed}.safetensors"
        for s in dict.fromkeys((aslug, aslug.split("_")[0]))
    ]
    ckpt = next((c for c in candidates if c.exists()), None)
    if ckpt is None:
        raise FileNotFoundError(f"missing arm checkpoint, tried {candidates}")
    return ckpt


def fit(img: Image.Image, h: int) -> Image.Image:
    w = round(img.width * h / img.height)
    return img.resize((w, h), Image.LANCZOS)


def assemble(stem: str, artist: str, arms: list[str], seeds: list[int], out: Path):
    real = fit(Image.open(RESIZED / artist / f"{stem}.png").convert("RGB"), CELL_H)
    grid = {
        (arm, seed): fit(
            Image.open(
                out / "renders" / slug(artist) / f"{arm}_s{seed}" / f"{stem}.png"
            ).convert("RGB"),
            CELL_H,
        )
        for arm in arms
        for seed in seeds
    }
    col_ws = [real.width] + [
        max(grid[(arm, seed)].width for seed in seeds) for arm in arms
    ]
    col_x = []
    x = 0
    for w in col_ws:
        col_x.append(x)
        x += w + PAD
    total_w = x - PAD
    total_h = LABEL_H + len(seeds) * (SEED_H + CELL_H + PAD) - PAD

    sheet = Image.new("RGB", (total_w, total_h), (12, 12, 12))
    draw = ImageDraw.Draw(sheet)
    for cx, name in zip(col_x, ["real (holdout)", *arms]):
        draw.text((cx + 2, 6), name, fill=(230, 230, 230))
    y = LABEL_H
    for r, seed in enumerate(seeds):
        draw.text((2, y + 1), f"s{seed}", fill=(120, 160, 255))
        y += SEED_H
        if r == 0:
            sheet.paste(real, (col_x[0], y))
        for c, arm in enumerate(arms, start=1):
            sheet.paste(grid[(arm, seed)], (col_x[c], y))
        y += CELL_H + PAD
    path = out / f"cand_{stem}.png"
    sheet.save(path)
    print(f"sheet {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--stems", nargs="+", default=list(DEFAULT_STEMS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--train_seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument(
        "--compile_blocks", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--out", default=None, help="Run dir (default: stamped name).")
    args = ap.parse_args()

    override = json.loads(Path(args.prompts).read_text())
    wanted = set(args.stems)
    cells = {
        artist: [p for p in c["prompts"] if p["stem"] in wanted]
        for artist, c in override["cells"].items()
    }
    cells = {a: ps for a, ps in cells.items() if ps}
    found = {p["stem"] for ps in cells.values() for p in ps}
    if found != wanted:
        raise SystemExit(f"stems not found in prompt file: {wanted - found}")

    out = Path(
        args.out
        or HERE / "runs" / f"20260730-e4-fig-candidates-{args.steps}steps"
    )
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for artist, prompts in cells.items():
        aslug = slug(artist)
        for arm in args.arms:
            for seed in args.train_seeds:
                arm_dir = out / "renders" / aslug / f"{arm}_s{seed}"
                if all((arm_dir / f"{p['stem']}.png").exists() for p in prompts):
                    print(f"[{aslug}/{arm}/s{seed}] present — skipping", flush=True)
                    continue
                ckpt = find_ckpt(aslug, arm, seed)
                lats = render_arm(
                    args, f"{aslug}/{arm}/s{seed}", ckpt, prompts, device
                )
                vae = load_vae(
                    default_checkpoints().vae,
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

    for artist, prompts in cells.items():
        for p in prompts:
            assemble(p["stem"], artist, list(args.arms), args.train_seeds, out)

    (out / "result.json").write_text(
        json.dumps(
            {
                "prompts": str(args.prompts),
                "stems": list(args.stems),
                "arms": list(args.arms),
                "train_seeds": list(args.train_seeds),
                "steps": args.steps,
                "cfg": args.cfg,
            },
            indent=1,
        )
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
