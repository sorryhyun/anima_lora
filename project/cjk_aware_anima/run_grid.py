#!/usr/bin/env python3
"""Render one prompt file across N already-trained arms (+ an optional no-LoRA
base reference) at a set of seeds, as ONE daemon command job.

`run_unmask_r2.py` renders a single arm and only after a cache/train stage;
this is the render-only, multi-arm sibling used to re-eval finished
checkpoints against a *different* prompt set. Sampler flags are copied from
`run_unmask_r2.py` verbatim so grids are comparable across both drivers.

    make daemon-run ARGS="--label v2grid --stall-timeout 0 \
        project/cjk_aware_anima/run_grid.py \
        --prompts project/cjk_aware_anima/assets/unmask_eval_prompts_v2.txt \
        --eval_dir output/tests/cjk_unmask_evalv2 \
        --arm OCR128=cjk_unmask_ocr_a128 --arm PLAIN128=cjk_unmask_plain_a128 \
        --base --seeds 42 7 1234"

Arms land in ``<eval_dir>/arm<NAME>_s<seed>`` (the layout
``probes/blind_pairs.py`` and ``probes/unmask_grid_judge.py`` expect); the
base reference lands in ``<eval_dir>/base_s<seed>``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = sys.executable

SAMPLER = [
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


def run(stage: str, argv: list[str]) -> None:
    print(f"\n=== [{stage}] {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=REPO, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=ckpt_stem",
        help="repeatable; ckpt_stem resolves to output/ckpt/<stem>.safetensors",
    )
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 1234])
    ap.add_argument("--lora_multiplier", default="1.0")
    ap.add_argument(
        "--base", action="store_true", help="also render a no-LoRA reference"
    )
    opts = ap.parse_args()

    arms = []
    for spec in opts.arm:
        name, _, stem = spec.partition("=")
        if not stem:
            sys.exit(f"--arm wants NAME=ckpt_stem, got {spec!r}")
        lora = REPO / "output" / "ckpt" / f"{stem}.safetensors"
        if not lora.exists():
            sys.exit(f"missing checkpoint: {lora}")
        arms.append((name, lora))
    prompts = (
        REPO / opts.prompts
        if not Path(opts.prompts).is_absolute()
        else Path(opts.prompts)
    )
    if not prompts.exists():
        sys.exit(f"missing prompts: {prompts}")
    n_rows = len(
        [ln for ln in prompts.read_text(encoding="utf-8").splitlines() if ln.strip()]
    )
    print(
        f"{len(arms)} arm(s){' + base' if opts.base else ''} x {len(opts.seeds)} seed(s) "
        f"x {n_rows} rows = {(len(arms) + bool(opts.base)) * len(opts.seeds) * n_rows} images",
        flush=True,
    )

    jobs = [
        (
            f"arm{n}",
            ["--lora_multiplier", opts.lora_multiplier, "--lora_weight", str(w)],
        )
        for n, w in arms
    ]
    if opts.base:
        jobs.append(("base", []))
    for tag, extra in jobs:
        for seed in opts.seeds:
            run(
                f"gen {tag} s{seed}",
                [
                    PY,
                    "inference.py",
                    *SAMPLER,
                    "--from_file",
                    str(prompts),
                    *extra,
                    "--seed",
                    str(seed),
                    "--save_path",
                    f"{opts.eval_dir}/{tag}_s{seed}",
                ],
            )
    print("\n=== done ->", opts.eval_dir, flush=True)


if __name__ == "__main__":
    main()
