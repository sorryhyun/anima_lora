#!/usr/bin/env python3
"""Arm C3 pipeline as ONE daemon command job: ext TE re-cache with the
glossary-r2 pack -> arm-C3 LoRA train -> 3-seed eval grid.

Stages run as direct subprocesses (never nested daemon jobs — that deadlocks
the serial queue, same rule as scripts/soup/pipeline.py). Queue it *behind*
the distill job that writes the pack::

    make daemon-run ARGS="--label unmask-c3-r2 --stall-timeout 0 \
        project/cjk_aware_anima/run_unmask_r2.py --queue"

Stage 1 re-uses the PP-OCR records + mirror from arm C2 (captions identical);
stage 3 renders ``assets/unmask_eval_prompts.txt`` at seeds 42/7/1234 into
``output/tests/cjk_unmask_eval2/armC3_s*`` next to the C2 grid.

Multi-training-seed arms (C10, plan_base1 B3) are one job per seed: the first
builds the mirror + TE cache, the rest pass ``--skip_cache`` and their own
``--method`` / ``--arm`` (``seed`` lives in the method toml).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = sys.executable

SEEDS = (42, 7, 1234)


def run(stage: str, argv: list[str]) -> None:
    print(f"\n=== [{stage}] {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=REPO, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ext_prefix", default="output/ckpt/cjk_vocab_pack_synthja_v5")
    ap.add_argument("--method", default="cjk_unmask_c3")
    ap.add_argument("--te_out", default="post_image_dataset/cjk_unmask/te/sincos_r2")
    ap.add_argument(
        "--mirror",
        default="post_image_dataset/cjk_unmask/mirror_sincos_ppocr",
        help="caption mirror dir; use a fresh one for a new records/format so "
        "the trained arms' mirrors stay as trained.",
    )
    ap.add_argument("--eval_dir", default="output/tests/cjk_unmask_eval2")
    ap.add_argument(
        "--prompts",
        default=str(HERE / "assets" / "unmask_eval_prompts.txt"),
        help="eval prompt file, one row per line (v2: assets/unmask_eval_prompts_v2.txt)",
    )
    ap.add_argument("--arm", default="armC3")
    ap.add_argument(
        "--records",
        default="post_image_dataset/cjk_unmask/ocr_records_sincos_ppocr.jsonl",
        help="OCR records; the _v2 file carries reading order + the ー/ニ/tally "
        "post-processing (anime_tools.ocr._text).",
    )
    ap.add_argument(
        "--ocr_format",
        default="order",
        choices=("order", "tags", "presence", "sentence"),
        help="cache_te_ext --ocr_format; C2–C9 were 'tags', C10 (plan_base1 B3) "
        "'sentence' on the hybrid records.",
    )
    ap.add_argument("--skip_cache", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS),
        help="grid seeds (default 42 7 1234); use fresh ones for a re-blind",
    )
    opts = ap.parse_args()

    pack = REPO / f"{opts.ext_prefix}.safetensors"
    if not opts.skip_cache and not pack.exists():
        sys.exit(f"ext pack missing: {pack} (distill job not finished?)")

    if not opts.skip_cache:
        run(
            "cache",
            [
                PY,
                str(HERE / "datasets" / "cache_te_ext.py"),
                "--shard",
                "sincos",
                "--records",
                opts.records,
                "--mirror",
                opts.mirror,
                "--ext_prefix",
                opts.ext_prefix,
                "--out",
                opts.te_out,
                "--ocr_format",
                opts.ocr_format,
            ],
        )

    if not opts.skip_train:
        run(
            "train",
            [
                PY,
                "train.py",
                "--method",
                opts.method,
                "--preset",
                "default",
                "--methods_subdir",
                "gui-methods/custom",
                # stamps ss_ext_pack_sha (D1): the LoRA is coupled to the pack
                # its TE caches were encoded through.
                "--ext_pack",
                opts.ext_prefix,
            ],
        )

    lora = f"output/ckpt/{opts.method}.safetensors"
    base = [
        PY,
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
        "--lora_weight",
        lora,
        "--from_file",
        opts.prompts,
    ]
    for seed in opts.seeds:
        run(
            f"gen s{seed}",
            base
            + [
                "--seed",
                str(seed),
                "--save_path",
                f"{opts.eval_dir}/{opts.arm}_s{seed}",
            ],
        )
    print("\n=== done:", lora, "->", opts.eval_dir, flush=True)


if __name__ == "__main__":
    main()
