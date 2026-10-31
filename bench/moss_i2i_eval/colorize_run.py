#!/usr/bin/env python3
"""Run a colorize LoRA as one daemon job: cache → train → eval of the final LoRA.

Each stage is a subprocess; the run stops at the first that fails.
``scripts/moss/colorize_cache.py`` builds ``post_image_dataset/moss/<name>``; the
training recipe is ``moss_colorize_3k_ocr``'s (``configs/moss/i2i/colorize/train.toml``).
OCR captions run up to ~2.2k text rows, so training starts on the worst case
(``--fit_report``) with expandable segments. ``--no-vision`` / ``--ref_scale`` /
``--ref_isolate`` / ``--holdout_json`` / ``--train_json`` reach the cache and trainer; ``--ocr_from``
copies another cache's ``ocr.json`` in first so the captions match its reads.

    make daemon-run ARGS="--queue --stall-timeout 1800 bench/moss_i2i_eval/colorize_run.py --name <name>"
"""

import argparse
import shutil
import subprocess
import sys

from library.env import resolve_under_home
from library.moss.requests import ColorizeCacheRequest, TrainRequest

SRC = "bench/moss_i2i_eval"


def run(argv):
    print(f"[colorize] {' '.join(argv)}", flush=True)
    subprocess.run([sys.executable, *argv], check=True, cwd=resolve_under_home("."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=1667, help="illustration pairs")
    ap.add_argument("--manga", type=int, default=1667, help="manga pages")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--ref_scale", type=int, default=2)
    ap.add_argument("--vision", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ref_isolate", action="store_true")
    ap.add_argument("--holdout_json", default="")
    ap.add_argument("--train_json", default="", help="stems to train on, no draw")
    ap.add_argument("--ocr_from", default="", help="cache whose ocr.json to reuse")
    ap.add_argument("--arms", default="", help="colorize_eval.py --arms")
    ap.add_argument("--blocks_to_swap", type=int, default=28)
    ap.add_argument("--activation_memory_budget", type=float, default=0.15)
    args = ap.parse_args()
    cache = f"post_image_dataset/moss/{args.name}"
    out = f"output/ckpt_moss_i2i/{args.name}.safetensors"

    if args.ocr_from:
        dst = resolve_under_home(cache) / "ocr.json"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolve_under_home(args.ocr_from) / "ocr.json", dst)
    cache_req = ColorizeCacheRequest(
        out=cache,
        n=args.n,
        manga=args.manga,
        ref_scale=args.ref_scale,
        vision=args.vision,
        holdout_json=args.holdout_json,
        train_json=args.train_json,
    )
    run([cache_req.SCRIPT, *cache_req.to_argv()])
    train = TrainRequest(
        cache=cache,
        output=out,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        save_every_epochs=1,
        plain_caption_rate=0.3 if args.vision else 0.0,
        caption_variant_rate=0.3,
        ref_isolate=args.ref_isolate,
        int8="off",
        stream="bf16",
        ema_card=False,
        activation_memory_budget=args.activation_memory_budget,
        blocks_to_swap=args.blocks_to_swap,
        fit_report=True,
        expandable_segments=True,
    )
    run([train.SCRIPT, *train.to_argv()])
    arms = ["--arms", args.arms] if args.arms else []
    run([f"{SRC}/colorize_eval.py", "--lora", out, "--cache", cache, *arms])


if __name__ == "__main__":
    main()
