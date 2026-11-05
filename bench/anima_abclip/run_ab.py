#!/usr/bin/env python3
"""Blind A/B of two LoRA recipes over several training seeds and artists.

Arm A is the shipped default (``configs/methods/lora.toml``: 2e-5 cosine, bs1,
clip 1.0); arm B is 5e-5 constant, GA2, median clip k 2 per micro-batch
(docs/anima/proposal/lora_batch_clip_ema.md). Each (artist, seed) trains both
arms; ``abclip_<artist>_<arm>_s<seed>.safetensors`` in ``output/ckpt``.

``render``: every LoRA renders the same prompts x render seeds at the
``make test`` recipe (one ``inference.py --from_file`` load per LoRA, reused
on rerun) and its median ``||dW||`` goes to the bench envelope.
``make``: one blind set per artist in ``../anima-blind-pairs/sets/<set>/``, A|B
sides shuffled per pair, the key in ``output/blind/<set>/key.json`` (outside
the repo). ``score``: wins per arm, a binomial sign test over pairs, and a
sign-flip test over (artist, seed) clusters, the unit training noise acts on.

    make daemon-run ARGS="--stall-timeout 900 bench/anima_abclip/run_ab.py render"
    .venv/bin/python bench/anima_abclip/run_ab.py make --push
    .venv/bin/python bench/anima_abclip/run_ab.py score
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from bench._common import make_run_dir, write_result
from bench.anima_ema.render_ab import _by_seed, render
from library.env import resolve_under_home

ARTISTS = {"cast": "@channel (caststation)", "ama": "@ama mitsuki"}
SEEDS = [42, 43, 44, 45]
ARMS = ["A", "B"]
BODIES = [
    "safe, 1girl, solo, {a}, blonde hair, school uniform, classroom, sitting at desk, smile, looking at viewer",
    "safe, 1boy, 1girl, {a}, hetero, hug, indoors, curtains",
    "safe, 1girl, solo, {a}, maid, holding a tray, cafe interior",
    "sensitive, 1girl, original, {a}, black bikini, black hair, short hair, innertube, beach, day, blue sky, looking at viewer, standing",
]
RENDER_SEEDS = [1, 2]
SET_PREFIX = "s29_ABCLIP"

CKPT = resolve_under_home("output/ckpt")
RENDERS = resolve_under_home("output/tests/abclip")
KEYS = resolve_under_home("output/blind")
GIT = resolve_under_home("..") / "anima-blind-pairs"


def lora(art: str, arm: str, seed: int) -> Path:
    return CKPT / f"abclip_{art}_{arm}_s{seed}.safetensors"


def render_seed(p: int, s: int) -> int:
    return 1000 * (p + 1) + s


def dw_norm(path: Path) -> float:
    """Median over modules of ``||up @ down||`` (alpha / rank shared by every arm)."""
    sd = load_file(str(path))
    norms = [
        (
            up.float().flatten(1)
            @ sd[k.replace(".lora_up.", ".lora_down.")].float().flatten(1)
        )
        .norm()
        .item()
        for k, up in sd.items()
        if k.endswith(".lora_up.weight")
    ]
    return float(torch.tensor(norms).median())


def cmd_render(args) -> None:
    import bench.anima_ema.render_ab as rab

    run_dir = make_run_dir("anima_abclip")
    disp = {}
    for art, tag in ARTISTS.items():
        rab.PROMPTS = [b.format(a=tag) for b in BODIES]
        rab.SEEDS = RENDER_SEEDS
        for arm, seed in itertools.product(ARMS, SEEDS):
            f = lora(art, arm, seed)
            if not f.exists():
                print(f"missing {f}, skipped", flush=True)
                continue
            render(f.stem, f, RENDERS / f.stem)
            disp[f.stem] = round(dw_norm(f), 5)
    write_result(
        run_dir,
        script=__file__,
        args={
            "artists": ARTISTS,
            "seeds": SEEDS,
            "bodies": BODIES,
            "render_seeds": RENDER_SEEDS,
        },
        metrics={"dw_median": disp},
    )
    print(json.dumps(disp, indent=1))


def compose(a: Path, b: Path, out: Path) -> None:
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


def set_name(art: str) -> str:
    return f"{SET_PREFIX}_{art}"


def cmd_make(args) -> None:
    rng = random.Random(args.shuffle_seed)
    for art, tag in ARTISTS.items():
        name = set_name(art)
        cells = [
            (seed, p, s)
            for seed in SEEDS
            for p in range(len(BODIES))
            for s in RENDER_SEEDS
        ]
        rng.shuffle(cells)
        out = GIT / "sets" / name
        key, readme = (
            {},
            [
                f"# {name}\n",
                "Same prompt, same seed, two different training recipes. Which side is "
                "better overall (style fidelity to the artist, anatomy, stray dataset "
                "artifacts such as speech bubbles)? Fill `verdicts.tsv` (A / B / tie / "
                "skip) and commit. Sides are shuffled per pair; the key is not in this repo.\n",
            ],
        )
        for i, (seed, p, s) in enumerate(cells, 1):
            pair = f"p{i:02d}"
            imgs = {
                arm: _by_seed(RENDERS / lora(art, arm, seed).stem).get(
                    render_seed(p, s)
                )
                for arm in ARMS
            }
            if any(v is None for v in imgs.values()):
                raise SystemExit(f"{art} s{seed} prompt {p} seed {s}: render missing")
            left, right = rng.sample(ARMS, 2)
            compose(imgs[left], imgs[right], out / f"{pair}.webp")
            key[pair] = {
                "A": left,
                "B": right,
                "seed": seed,
                "prompt": p,
                "render_seed": s,
            }
            readme.append(
                f"## {pair}\n\n`{BODIES[p].format(a=tag)}`\n\n![{pair}]({pair}.webp)\n"
            )
        (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
        (out / "verdicts.tsv").write_text(
            "pair\tverdict\tnote\n" + "".join(f"{p}\t\t\n" for p in key),
            encoding="utf-8",
        )
        (KEYS / name).mkdir(parents=True, exist_ok=True)
        (KEYS / name / "key.json").write_text(
            json.dumps(key, indent=1), encoding="utf-8"
        )
        print(f"{name}: {len(key)} pairs -> {out}")
    if args.push:
        sets = [f"sets/{set_name(a)}" for a in ARTISTS]
        subprocess.run(["git", "add", *sets], cwd=GIT, check=True)
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                f"blind sets {', '.join(set_name(a) for a in ARTISTS)}: "
                f"{len(SEEDS) * len(BODIES) * len(RENDER_SEEDS)} pairs each (A default vs B median-clip GA2)",
            ],
            cwd=GIT,
            check=True,
        )
        subprocess.run(["git", "push"], cwd=GIT, check=True)


def binom_tail(k: int, n: int) -> float:
    """P(X >= k), X ~ Bin(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n


def cmd_score(args) -> None:
    if not args.no_pull:
        subprocess.run(["git", "pull", "--ff-only"], cwd=GIT, check=True)
    wins = defaultdict(int)
    cluster = defaultdict(lambda: [0, 0])  # (art, seed) -> [A wins, B wins]
    by_prompt = defaultdict(lambda: [0, 0])
    lines = []
    for art in ARTISTS:
        name = set_name(art)
        key = json.loads((KEYS / name / "key.json").read_text(encoding="utf-8"))
        rows = (
            (GIT / "sets" / name / "verdicts.tsv")
            .read_text(encoding="utf-8")
            .splitlines()[1:]
        )
        for row in rows:
            pair, verdict, *_ = (row.split("\t") + ["", ""])[:3]
            verdict = verdict.strip().upper()
            k = key[pair]
            if verdict in ("A", "B"):
                arm = k[verdict]
                wins[arm] += 1
                cluster[(art, k["seed"])][ARMS.index(arm)] += 1
                by_prompt[(art, k["prompt"])][ARMS.index(arm)] += 1
            else:
                wins[verdict.lower() or "blank"] += 1
            lines.append(
                f"| {name} | {pair} | s{k['seed']} | p{k['prompt']} | {k['A']} | {k['B']} | {verdict} |"
            )

    n = wins["A"] + wins["B"]
    # sign-flip over clusters: statistic = sum of (B - A) per cluster
    diffs = [b - a for a, b in cluster.values()]
    obs = sum(diffs)
    flips = [
        sum(s * d for s, d in zip(signs, diffs))
        for signs in itertools.product((1, -1), repeat=len(diffs))
    ]
    p_flip = sum(f >= obs for f in flips) / len(flips) if diffs else float("nan")

    out = [
        f"# blind A/B — {SET_PREFIX}\n",
        "A = default (2e-5 cosine bs1 clip 1.0), B = 5e-5 constant GA2 median k2 per micro\n",
        f"| arm | wins |\n|---|---:|\n| A | {wins['A']} |\n| B | {wins['B']} |\n"
        + "".join(f"| {k} | {v} |\n" for k, v in wins.items() if k not in ARMS),
        f"Binomial sign test, B wins {wins['B']}/{n}: one-sided p = {binom_tail(wins['B'], n):.4f}"
        if n
        else "no graded pairs",
        f"\nCluster sign-flip over {len(diffs)} (artist, seed) pairs, B − A = {obs}: one-sided p = {p_flip:.4f}\n",
        "| artist | seed | A | B |\n|---|---|---:|---:|\n"
        + "".join(
            f"| {a} | s{s} | {v[0]} | {v[1]} |\n"
            for (a, s), v in sorted(cluster.items())
        ),
        "| artist | prompt | A | B |\n|---|---|---:|---:|\n"
        + "".join(
            f"| {a} | p{p} | {v[0]} | {v[1]} |\n"
            for (a, p), v in sorted(by_prompt.items())
        ),
        "| set | pair | seed | prompt | A | B | verdict |\n|---|---|---|---|---|---|---|\n"
        + "\n".join(lines),
    ]
    report = resolve_under_home("bench/anima_abclip/report.md")
    report.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:5]))
    print(f"-> {report}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("render")
    mk = sub.add_parser("make")
    mk.add_argument("--shuffle_seed", type=int, default=29)
    mk.add_argument("--push", action="store_true")
    sc = sub.add_parser("score")
    sc.add_argument("--no_pull", action="store_true")
    args = ap.parse_args()
    {"render": cmd_render, "make": cmd_make, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
