#!/usr/bin/env python3
"""Blind A/B of two LoRA recipes over several training seeds and artists.

Experiments (``--exp``, docs/anima/findings/lora_gradient_noise_recipe.md):

- ``abclip`` (s29, Phase 1): ``A`` the old default (2e-5 cosine, bs1, clip 1.0)
  against ``B`` (5e-5 constant, GA2, median clip k 2 per micro-batch), on
  caststation and ama mitsuki.
- ``nomask`` (s31): T-LoRA as published (mask in training and at inference) against
  the same recipe without the mask, sweetonedollar, seeds 42-44.
- ``sigshift`` (s32, proposal item 1): the shipped recipe (no mask) against the
  same with ``timestep_sampling = "shift"``, ``discrete_flow_shift = 2``, on
  caststation and sweetonedollar, seeds 42-44; ``mad`` prints the cross-seed
  spread per arm.
- ``abclip2`` (s33, proposal §0): s29 rerun with ``use_timestep_mask`` off in
  both arms and B on one clip window, ama mitsuki / sweetonedollar / caststation,
  B rescaled to A's median ``||dW||``.
- ``sigband`` (s30, Phase 2): ``global`` (recipe B, the shipped
  ``configs/methods/lora.toml``; reuses the s29 B LoRAs) against ``band`` (B with
  ``--clip_sigma_bands 0.2,0.4,0.6,0.8``), on caststation and sweetonedollar, at
  new render seeds. The band LoRA is rescaled to the global LoRA's median
  ``||dW||`` of the same (artist, seed) before rendering (``match``), so the
  pairs compare the direction and not the step length.
- ``rank`` (s34): the shipped recipe at rank 32 / alpha 128 (the s31 / s32
  ``nomask`` LoRAs) against rank 16 / alpha 64, caststation and sweetonedollar,
  seeds 42-44, r16 rescaled to r32's median ``||dW||``; ``mad`` first.
- ``alpha`` (s35): the shipped recipe (alpha 128, lr 5e-5) against alpha 512,
  lr 1.25e-5 (same alpha x lr, a quarter of the bilinear drift; arXiv
  2606.12883), caststation and sweetonedollar, seeds 42-44, matched ``||dW||``.
- ``alpha_sincos`` (s36): the same pair on sincos (354 images, ~1416 updates),
  seeds 42-43, four render seeds.
- ``norepa`` (s37): the s35 a512 LoRAs (relational REPA on) against the same
  recipe with ``use_repa=false``, caststation, seeds 42-44, matched ``||dW||``.
- ``norepa_sweet`` (s38): s37 on sweetonedollar.

``train``: queues the missing LoRAs on the daemon. ``render``: (``match`` first
when the experiment asks for it) every LoRA renders the prompts x render seeds
at the ``make test`` recipe, and its median ``||dW||`` goes to the bench
envelope. ``make``: one blind set per artist in
``../anima-blind-pairs/sets/<set>/``, sides shuffled per pair, the key in
``output/blind/<set>/key.json`` (outside the repo). ``score``: wins per arm, a
binomial sign test over pairs, and a sign-flip test over (artist, seed)
clusters, the unit training noise acts on. ``answer``: unblinded copies,
``output/blind/<prefix>_answer_<artist>/<arm>/<pair>.png``.

    .venv/bin/python bench/anima_abclip/run_ab.py --exp sigband train
    make daemon-run ARGS="--queue --stall-timeout 900 bench/anima_abclip/run_ab.py --exp sigband render"
    .venv/bin/python bench/anima_abclip/run_ab.py --exp sigband make --push
    .venv/bin/python bench/anima_abclip/run_ab.py --exp sigband score
    .venv/bin/python bench/anima_abclip/run_ab.py --exp sigband answer
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from bench._common import make_run_dir, write_result
from bench.anima_ema.render_ab import INFER, _by_seed, render
from library.env import resolve_under_home

TAGS = {
    "cast": ("@channel (caststation)", "channel_(caststation)/*"),
    "ama": ("@ama mitsuki", "ama_mitsuki/*"),
    "sweet": ("@sweetonedollar", "sweetonedollar/*"),
    "sincos": ("@sincos", "sincos/*"),
}
BODIES = [
    "safe, 1girl, solo, {a}, blonde hair, school uniform, classroom, sitting at desk, smile, looking at viewer",
    "safe, 1boy, 1girl, {a}, hetero, hug, indoors, curtains",
    "safe, 1girl, solo, {a}, maid, holding a tray, cafe interior",
    "sensitive, 1girl, original, {a}, black bikini, black hair, short hair, innertube, beach, day, blue sky, looking at viewer, standing",
]
SEEDS = [42, 43, 44, 45]

CKPT = resolve_under_home("output/ckpt")
KEYS = resolve_under_home("output/blind")
GIT = resolve_under_home("..") / "anima-blind-pairs"


@dataclass(frozen=True)
class Exp:
    artists: tuple[str, ...]
    arms: tuple[str, str]  # (control, treatment); statistic = treatment − control
    stems: dict  # arm -> output_name pattern with {art} and {seed}
    train_args: dict  # arm -> extra train.py flags
    describe: str
    render_seeds: tuple[int, ...]
    set_prefix: str
    renders: str
    shuffle_seed: int
    report: str
    match_dw: bool = False  # rescale the treatment to the control's ||dW||
    seeds: tuple[int, ...] = tuple(SEEDS)
    render_args: dict | None = None  # arm -> extra inference.py flags


EXPS = {
    "abclip": Exp(
        artists=("cast", "ama"),
        arms=("A", "B"),
        stems={"A": "abclip_{art}_A_s{seed}", "B": "abclip_{art}_B_s{seed}"},
        train_args={},
        describe="A = default (2e-5 cosine bs1 clip 1.0), B = 5e-5 constant GA2 median k2 per micro",
        render_seeds=(1, 2),
        set_prefix="s29_ABCLIP",
        renders="output/tests/abclip",
        shuffle_seed=29,
        report="bench/anima_abclip/report.md",
    ),
    "sigband": Exp(
        artists=("cast", "sweet"),
        arms=("global", "band"),
        stems={
            "global": "abclip_{art}_B_s{seed}",
            "band": "sigband_{art}_band_s{seed}",
        },
        train_args={"global": "", "band": "--clip_sigma_bands 0.2,0.4,0.6,0.8"},
        describe="global = recipe B (median clip k2 per micro, one window), band = B with "
        "one window per σ band (0.2 / 0.4 / 0.6 / 0.8); band rescaled to global's median ||dW||",
        render_seeds=(3, 4),
        set_prefix="s30_SIGBAND",
        renders="output/tests/sigband",
        shuffle_seed=30,
        report="bench/anima_abclip/report_sigband.md",
        match_dw=True,
    ),
    "nomask": Exp(
        artists=("sweet",),
        arms=("band", "nomask"),
        stems={
            "band": "sigband_{art}_band_s{seed}",
            "nomask": "nomask_{art}_s{seed}",
        },
        train_args={"band": "", "nomask": "--network_args use_timestep_mask=false"},
        describe="band = T-LoRA as published: the shipped lora.toml (B + σ-band clip, "
        "use_timestep_mask, min_rank 1) rendered with the same mask per step "
        "(--lora_timestep_mask); nomask = the same recipe with use_timestep_mask off, full "
        "rank in training and inference (static merge); nomask rescaled to band's median "
        "||dW||",
        render_args={"band": ["--lora_timestep_mask"]},
        render_seeds=(5, 6),
        set_prefix="s31_NOMASK",
        renders="output/tests/nomask",
        shuffle_seed=31,
        report="bench/anima_abclip/report_nomask.md",
        match_dw=True,
        seeds=(42, 43, 44),
    ),
    "sigshift": Exp(
        artists=("cast", "sweet"),
        arms=("shift1", "shift2"),
        stems={
            "shift1": "nomask_{art}_s{seed}",
            "shift2": "sigshift_{art}_s{seed}",
        },
        train_args={
            "shift1": "--network_args use_timestep_mask=false",
            "shift2": "--network_args use_timestep_mask=false "
            "--timestep_sampling shift --discrete_flow_shift 2",
        },
        describe="shift1 = the shipped lora.toml (B + σ-band clip, no mask; σ ~ "
        "logit-normal(0, 1)); shift2 = the same with timestep_sampling shift, "
        "discrete_flow_shift 2 (σ >= 0.8 draws 7.4 % -> 24 %); shift2 rescaled to "
        "shift1's median ||dW||",
        render_seeds=(7, 8),
        set_prefix="s32_SIGSHIFT",
        renders="output/tests/sigshift",
        shuffle_seed=32,
        report="bench/anima_abclip/report_sigshift.md",
        match_dw=True,
        seeds=(42, 43, 44),
    ),
    "abclip2": Exp(
        artists=("ama", "sweet", "cast"),
        arms=("A", "B"),
        stems={"A": "abclip2_{art}_A_s{seed}", "B": "abclip2_{art}_B_s{seed}"},
        train_args={
            "A": "--network_args use_timestep_mask=false --learning_rate 2e-5 "
            "--lr_scheduler cosine --gradient_accumulation_steps 1 --clip_mode fixed "
            "--max_grad_norm 1.0 --clip_sigma_bands ''",
            "B": "--network_args use_timestep_mask=false --clip_sigma_bands ''",
        },
        describe="s29 without the T-LoRA mask: A = old default (2e-5 cosine, bs1, fixed "
        "clip 1.0), B = 5e-5 constant GA2, median clip k2 per micro, one window (no σ "
        "bands); both use_timestep_mask off, EMA 0.1; B rescaled to A's median ||dW||",
        render_seeds=(9, 10),
        set_prefix="s33_ABCLIP2",
        renders="output/tests/abclip2",
        shuffle_seed=33,
        report="bench/anima_abclip/report_abclip2.md",
        match_dw=True,
    ),
    "rank": Exp(
        artists=("cast", "sweet"),
        arms=("r32", "r16"),
        stems={"r32": "nomask_{art}_s{seed}", "r16": "rank16_{art}_s{seed}"},
        train_args={
            "r32": "--network_args use_timestep_mask=false",
            "r16": "--network_args use_timestep_mask=false --network_dim 16 "
            "--network_alpha 64 --grad_sigma_log",
        },
        describe="r32 = the shipped lora.toml (rank 32 / alpha 128, B + σ-band clip, no "
        "mask; the s31 / s32 LoRAs); r16 = the same at rank 16 / alpha 64 (alpha / rank "
        "4 in both); r16 rescaled to r32's median ||dW||",
        render_seeds=(11, 12),
        set_prefix="s34_RANK",
        renders="output/tests/rank",
        shuffle_seed=34,
        report="bench/anima_abclip/report_rank.md",
        match_dw=True,
        seeds=(42, 43, 44),
    ),
    "alpha": Exp(
        artists=("cast", "sweet"),
        arms=("a128", "a512"),
        stems={"a128": "nomask_{art}_s{seed}", "a512": "alpha512_{art}_s{seed}"},
        train_args={
            "a128": "--network_args use_timestep_mask=false",
            "a512": "--network_args use_timestep_mask=false --network_alpha 512 "
            "--learning_rate 1.25e-5",
        },
        describe="a128 = the shipped lora.toml (rank 32 / alpha 128, lr 5e-5, B + σ-band "
        "clip, no mask; the s31 / s32 LoRAs); a512 = alpha 512, lr 1.25e-5 (same "
        "alpha x lr, so the same first-order step and 1/4 of the bilinear drift, "
        "arXiv 2606.12883); a512 rescaled to a128's median ||dW||",
        render_seeds=(13, 14),
        set_prefix="s35_ALPHA",
        renders="output/tests/alpha",
        shuffle_seed=35,
        report="bench/anima_abclip/report_alpha.md",
        match_dw=True,
        seeds=(42, 43, 44),
    ),
    "alpha_sincos": Exp(
        artists=("sincos",),
        arms=("a128", "a512"),
        stems={"a128": "a128_{art}_s{seed}", "a512": "alpha512_{art}_s{seed}"},
        train_args={
            "a128": "--network_args use_timestep_mask=false",
            "a512": "--network_args use_timestep_mask=false --network_alpha 512 "
            "--learning_rate 1.25e-5",
        },
        describe="s35 on a 354-image set (~1416 updates against ~120): a128 = the shipped "
        "lora.toml (rank 32 / alpha 128, lr 5e-5); a512 = alpha 512, lr 1.25e-5; a512 "
        "rescaled to a128's median ||dW||",
        render_seeds=(15, 16, 17, 18),
        set_prefix="s36_ALPHASINCOS",
        renders="output/tests/alpha_sincos",
        shuffle_seed=36,
        report="bench/anima_abclip/report_alpha_sincos.md",
        match_dw=True,
        seeds=(42, 43),
    ),
    "norepa": Exp(
        artists=("cast",),
        arms=("repa", "norepa"),
        stems={"repa": "alpha512_{art}_s{seed}", "norepa": "norepa512_{art}_s{seed}"},
        train_args={
            "repa": "--network_args use_timestep_mask=false --network_alpha 512 "
            "--learning_rate 1.25e-5",
            "norepa": "--network_args use_timestep_mask=false use_repa=false "
            "--network_alpha 512 --learning_rate 1.25e-5",
        },
        describe="repa = the s35 a512 LoRAs (alpha 512, lr 1.25e-5, relational REPA "
        "weight 0.05 at layer 8); norepa = the same with use_repa off; norepa rescaled "
        "to repa's median ||dW||",
        render_seeds=(19, 20),
        set_prefix="s37_NOREPA",
        renders="output/tests/norepa",
        shuffle_seed=37,
        report="bench/anima_abclip/report_norepa.md",
        match_dw=True,
        seeds=(42, 43, 44),
    ),
    "norepa_sweet": Exp(
        artists=("sweet",),
        arms=("repa", "norepa"),
        stems={"repa": "alpha512_{art}_s{seed}", "norepa": "norepa512_{art}_s{seed}"},
        train_args={
            "repa": "--network_args use_timestep_mask=false --network_alpha 512 "
            "--learning_rate 1.25e-5",
            "norepa": "--network_args use_timestep_mask=false use_repa=false "
            "--network_alpha 512 --learning_rate 1.25e-5",
        },
        describe="s37 on sweetonedollar: repa = the s35 a512 LoRAs, norepa = the same "
        "with use_repa off; norepa rescaled to repa's median ||dW||",
        render_seeds=(19, 20),
        set_prefix="s38_NOREPA",
        renders="output/tests/norepa",
        shuffle_seed=38,
        report="bench/anima_abclip/report_norepa_sweet.md",
        match_dw=True,
        seeds=(42, 43, 44),
    ),
}


def lora(e: Exp, art: str, arm: str, seed: int) -> Path:
    return CKPT / f"{e.stems[arm].format(art=art, seed=seed)}.safetensors"


def rendered(e: Exp, art: str, arm: str, seed: int) -> Path:
    """The file that renders: the dW-matched copy for a matched treatment."""
    f = lora(e, art, arm, seed)
    if e.match_dw and arm == e.arms[1]:
        return f.with_name(f"{f.stem}_dwm.safetensors")
    return f


def render_seed(p: int, s: int) -> int:
    return 1000 * (p + 1) + s


def dw_norm(path: Path) -> float:
    """Median over modules of ``||(alpha / rank) · up @ down||``.

    Envelopes before s35 recorded ``||up @ down||`` without the scale; every arm
    there ran alpha / rank 4, so their ratios and match factors are unchanged."""
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


def cmd_train(e: Exp, args) -> None:
    for art, seed, arm in itertools.product(e.artists, e.seeds, e.arms):
        f = lora(e, art, arm, seed)
        if f.exists():
            continue
        extra = f"--queue --seed {seed} --path_pattern '{TAGS[art][1]}' {e.train_args.get(arm, '')} --output_name {f.stem}"
        print(f"queue {f.stem}: {extra}", flush=True)
        if not args.dry_run:
            subprocess.run(
                ["make", "lora", f"ARGS={extra}"],
                cwd=resolve_under_home("."),
                check=True,
                stdout=subprocess.DEVNULL,
            )


def cmd_match(e: Exp) -> dict:
    """Write ``<treatment>_dwm.safetensors``: ``lora_up`` scaled so the median
    ``||dW||`` equals the control's at the same (artist, seed)."""
    factors = {}
    ctl, trt = e.arms
    for art, seed in itertools.product(e.artists, e.seeds):
        src, ref = lora(e, art, trt, seed), lora(e, art, ctl, seed)
        if not (src.exists() and ref.exists()):
            print(f"match: missing {src.name} or {ref.name}, skipped", flush=True)
            continue
        factor = dw_norm(ref) / dw_norm(src)
        factors[src.stem] = round(factor, 5)
        out = rendered(e, art, trt, seed)
        if out.exists():
            continue
        with safe_open(str(src), "pt") as f:
            meta = dict(f.metadata() or {})
        sd = load_file(str(src))
        for k in sd:
            if k.endswith(".lora_up.weight"):
                sd[k] = (sd[k].float() * factor).to(sd[k].dtype)
        meta["ss_dw_match_factor"] = f"{factor:.6f}"
        meta["ss_dw_match_ref"] = ref.name
        save_file(sd, str(out), metadata=meta)
        print(f"match {src.stem}: x{factor:.4f} -> {out.name}", flush=True)
    return factors


def cmd_render(e: Exp, args) -> None:
    import bench.anima_ema.render_ab as rab

    factors = cmd_match(e) if e.match_dw else {}
    run_dir = make_run_dir("anima_abclip", label=args.exp)
    disp = {}
    for art in e.artists:
        rab.PROMPTS = [b.format(a=TAGS[art][0]) for b in BODIES]
        rab.SEEDS = list(e.render_seeds)
        for arm, seed in itertools.product(e.arms, e.seeds):
            f = rendered(e, art, arm, seed)
            if not f.exists():
                print(f"missing {f}, skipped", flush=True)
                continue
            rab.INFER = (
                INFER + ["--compile_blocks"] + (e.render_args or {}).get(arm, [])
            )
            render(f.stem, f, resolve_under_home(e.renders) / f.stem)
            disp[f.stem] = round(dw_norm(f), 5)
            raw = lora(e, art, arm, seed)
            if raw != f:
                disp[raw.stem] = round(dw_norm(raw), 5)
    write_result(
        run_dir,
        script=__file__,
        args={
            "exp": args.exp,
            "artists": list(e.artists),
            "seeds": e.seeds,
            "bodies": BODIES,
            "render_seeds": list(e.render_seeds),
            "render_args": e.render_args or {},
        },
        metrics={"dw_median": disp, "dw_match_factor": factors},
    )
    print(json.dumps({"dw_median": disp, "dw_match_factor": factors}, indent=1))


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


def set_name(e: Exp, art: str) -> str:
    return f"{e.set_prefix}_{art}"


def cmd_make(e: Exp, args) -> None:
    rng = random.Random(
        e.shuffle_seed if args.shuffle_seed is None else args.shuffle_seed
    )
    renders = resolve_under_home(e.renders)
    for art in e.artists:
        tag = TAGS[art][0]
        name = set_name(e, art)
        cells = [
            (seed, p, s)
            for seed in e.seeds
            for p in range(len(BODIES))
            for s in e.render_seeds
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
                arm: _by_seed(renders / rendered(e, art, arm, seed).stem).get(
                    render_seed(p, s)
                )
                for arm in e.arms
            }
            if any(v is None for v in imgs.values()):
                raise SystemExit(f"{art} s{seed} prompt {p} seed {s}: render missing")
            left, right = rng.sample(list(e.arms), 2)
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
        sets = [f"sets/{set_name(e, a)}" for a in e.artists]
        subprocess.run(["git", "add", *sets], cwd=GIT, check=True)
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                f"blind sets {', '.join(set_name(e, a) for a in e.artists)}: "
                f"{len(e.seeds) * len(BODIES) * len(e.render_seeds)} pairs each "
                f"({e.arms[0]} vs {e.arms[1]})",
            ],
            cwd=GIT,
            check=True,
        )
        subprocess.run(["git", "push"], cwd=GIT, check=True)


def binom_tail(k: int, n: int) -> float:
    """P(X >= k), X ~ Bin(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n


def cmd_score(e: Exp, args) -> None:
    if not args.no_pull:
        subprocess.run(["git", "pull", "--ff-only"], cwd=GIT, check=True)
    ctl, trt = e.arms
    wins = defaultdict(int)
    cluster = defaultdict(lambda: [0, 0])  # (art, seed) -> [ctl wins, trt wins]
    by_prompt = defaultdict(lambda: [0, 0])
    lines = []
    for art in e.artists:
        name = set_name(e, art)
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
                cluster[(art, k["seed"])][e.arms.index(arm)] += 1
                by_prompt[(art, k["prompt"])][e.arms.index(arm)] += 1
            else:
                wins[verdict.lower() or "blank"] += 1
            lines.append(
                f"| {name} | {pair} | s{k['seed']} | p{k['prompt']} | {k['A']} | {k['B']} | {verdict} |"
            )

    n = wins[ctl] + wins[trt]
    # sign-flip over clusters: statistic = sum of (treatment − control) per cluster
    diffs = [b - a for a, b in cluster.values()]
    obs = sum(diffs)
    flips = [
        sum(s * d for s, d in zip(signs, diffs))
        for signs in itertools.product((1, -1), repeat=len(diffs))
    ]
    p_flip = sum(f >= obs for f in flips) / len(flips) if diffs else float("nan")

    out = [
        f"# blind A/B — {e.set_prefix}\n",
        e.describe + "\n",
        f"| arm | wins |\n|---|---:|\n| {ctl} | {wins[ctl]} |\n| {trt} | {wins[trt]} |\n"
        + "".join(f"| {k} | {v} |\n" for k, v in wins.items() if k not in e.arms),
        f"Binomial sign test, {trt} wins {wins[trt]}/{n}: one-sided p = {binom_tail(wins[trt], n):.4f}"
        if n
        else "no graded pairs",
        f"\nCluster sign-flip over {len(diffs)} (artist, seed) pairs, {trt} − {ctl} = {obs}: one-sided p = {p_flip:.4f}\n",
        f"| artist | seed | {ctl} | {trt} |\n|---|---|---:|---:|\n"
        + "".join(
            f"| {a} | s{s} | {v[0]} | {v[1]} |\n"
            for (a, s), v in sorted(cluster.items())
        ),
        f"| artist | prompt | {ctl} | {trt} |\n|---|---|---:|---:|\n"
        + "".join(
            f"| {a} | p{p} | {v[0]} | {v[1]} |\n"
            for (a, p), v in sorted(by_prompt.items())
        ),
        "| set | pair | seed | prompt | A | B | verdict |\n|---|---|---|---|---|---|---|\n"
        + "\n".join(lines),
    ]
    report = resolve_under_home(e.report)
    report.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:5]))
    print(f"-> {report}")


def cmd_answer(e: Exp, args) -> None:
    renders = resolve_under_home(e.renders)
    for art in e.artists:
        key = json.loads(
            (KEYS / set_name(e, art) / "key.json").read_text(encoding="utf-8")
        )
        out = KEYS / f"{e.set_prefix.split('_')[0]}_answer_{art}"
        for arm in e.arms:
            (out / arm).mkdir(parents=True, exist_ok=True)
        for pair, k in key.items():
            for arm in e.arms:
                src = _by_seed(renders / rendered(e, art, arm, k["seed"]).stem)[
                    render_seed(k["prompt"], k["render_seed"])
                ]
                shutil.copy2(src, out / arm / f"{pair}.png")
        print(f"{out}: {len(key)} pairs x {list(e.arms)}")


def cmd_mad(e: Exp, args) -> None:
    """Cross-training-seed spread per arm: mean absolute pixel difference (0-255)
    between every two training seeds' renders of the same prompt and render seed."""
    import numpy as np

    renders = resolve_under_home(e.renders)
    out = {}
    for art, arm in itertools.product(e.artists, e.arms):
        by = {s: _by_seed(renders / rendered(e, art, arm, s).stem) for s in e.seeds}
        vals = []
        for p, rs in itertools.product(range(len(BODIES)), e.render_seeds):
            k = render_seed(p, rs)
            for a, b in itertools.combinations(e.seeds, 2):
                if k in by[a] and k in by[b]:
                    ia = np.asarray(Image.open(by[a][k]).convert("RGB"), np.float32)
                    ib = np.asarray(Image.open(by[b][k]).convert("RGB"), np.float32)
                    vals.append(float(np.abs(ia - ib).mean()))
        out[f"{art}/{arm}"] = round(sum(vals) / len(vals), 2) if vals else None
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--exp", choices=sorted(EXPS), default="abclip")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dry_run", action="store_true")
    sub.add_parser("match")
    sub.add_parser("render")
    mk = sub.add_parser("make")
    mk.add_argument("--shuffle_seed", type=int, default=None)
    mk.add_argument("--push", action="store_true")
    sc = sub.add_parser("score")
    sc.add_argument("--no_pull", action="store_true")
    sub.add_parser("answer")
    sub.add_parser("mad")
    args = ap.parse_args()
    e = EXPS[args.exp]
    {
        "train": cmd_train,
        "match": lambda e, a: cmd_match(e),
        "render": cmd_render,
        "make": cmd_make,
        "score": cmd_score,
        "answer": cmd_answer,
        "mad": cmd_mad,
    }[args.cmd](e, args)


if __name__ == "__main__":
    main()
