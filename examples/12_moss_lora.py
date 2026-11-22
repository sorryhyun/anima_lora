#!/usr/bin/env python3
"""MOSS-Image: cache → LoRA training → generation with that LoRA.

MOSS-Image is a separate line from Anima (5B DiT, Qwen3.5-9B text encoder, FLUX.2
VAE; `library/moss/CLAUDE.md`), so none of `anima_lora`'s Anima entry points
apply. Each job is a torch-free request dataclass in `library.moss.requests`
(`CacheRequest` / `TrainRequest` / `GenerateRequest`) that holds every flag and its
default, plus a `run_*` function that does the work:

    request                      work                          sidecar CLI
    CacheRequest                 library.moss.cache.run_cache   scripts/moss/cache.py
    TrainRequest                 library.moss.train.run_train   scripts/moss/train.py
    GenerateRequest              library.moss.generate.run_generate  scripts/moss/generate.py

Settings resolve field default → TOML (`configs/moss/{cache,lora}.toml`, flat field
names) → your overrides. `req.to_argv()` gives the same job as CLI flags, which is
what the GUI and the daemon run.

Run from the repo root (anima_lora/) after `make download-model ARGS=moss-base`:

    # print the three requests as CLI argv (no torch, no GPU)
    python examples/12_moss_lora.py

    # queue all three on the GPU daemon; they run in order
    python examples/12_moss_lora.py --submit --src post_image_dataset/resized/my_set

    # or run one stage in this process
    python examples/12_moss_lora.py --run cache    --src post_image_dataset/resized/my_set
    python examples/12_moss_lora.py --run train    --epochs 8
    python examples/12_moss_lora.py --run generate --prompt "…"

Cache, train and generate must agree on the text encoder, caption format and prompt
tail: the TE cache and the LoRA stamp them, and generate's `prompt_tail auto` follows
the LoRA. The defaults here leave all three at the shipped values (int8 TE,
`layout_json`, `sure`). Captions come from `.txt` files beside the images in `--src`.
Training and generation default to the int8 DiT (16 GB cards).
"""

from __future__ import annotations

import argparse
import os
import shlex

# Torch-free: building and printing requests loads no model code.
from library.moss.requests import CacheRequest, GenerateRequest, TrainRequest

CACHE_CONFIG = "configs/moss/cache.toml"
TRAIN_CONFIG = "configs/moss/lora.toml"
# The GUI's budget for MOSS jobs: the TE load and the first compile print nothing
# for minutes.
STALL_TIMEOUT = 900.0


def build_requests(
    opts: argparse.Namespace,
) -> tuple[CacheRequest, TrainRequest, GenerateRequest]:
    # load_config checks the TOML's keys and types against the dataclass fields.
    cache = CacheRequest(
        **{
            **CacheRequest.load_config(CACHE_CONFIG),
            "src": opts.src,
            "out": opts.cache_dir,
        }
    )
    train = TrainRequest(
        **{
            **TrainRequest.load_config(TRAIN_CONFIG),
            "cache": opts.cache_dir,
            "output": opts.lora,
            "epochs": opts.epochs,
        }
    )
    generate = GenerateRequest(
        prompt=opts.prompt,
        # layout_json: an Anima-format prompt mapped onto MOSS's layout JSON, as the
        # cache's caption_format did. Use "text" for a prompt already in JSON.
        prompt_format=cache.caption_format,
        text_encoder=cache.text_encoder,
        lora=opts.lora,
        lora_multiplier=opts.multiplier,
        seed=opts.seed,
        out_dir=opts.out_dir,
        # steps 50, CFG 3, uncond skips block 22, shift 5, ER-SDE: the reference
        # pipeline's sampler, left at the field defaults.
    )
    return cache, train, generate


def print_requests(reqs) -> None:
    for req in reqs:
        print(f"# {type(req).__name__}")
        print(shlex.join(["python", req.SCRIPT, *req.to_argv()]) + "\n")


def submit(reqs) -> None:
    """Queue each request as a daemon command job, the way the GUI does."""
    from anima_daemon.client import ensure_daemon

    client = ensure_daemon()
    for req in reqs:
        label = f"moss {req.SCRIPT.rsplit('/', 1)[-1].removesuffix('.py')}"
        job = client.submit_command(
            label=label,
            argv=[req.SCRIPT, *req.to_argv()],
            stall_timeout=STALL_TIMEOUT,
            start=True,
        )
        print(f"queued {label}: job {job.get('job_id')}")
    print("follow with: make run-status  /  make daemon-wait JOB=<id>")


def run_inline(stage: str, cache, train, generate) -> None:
    """One stage in this process. Set the allocator env before torch loads, as the
    sidecars do."""
    if stage == "cache":
        from library.moss.cache import run_cache

        print(f"cache → {run_cache(cache)}")
    elif stage == "train":
        if train.expandable_segments:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from library.moss.train import run_train

        print(f"LoRA → {run_train(train)}")
    else:
        if generate.blocks_to_swap:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from library.moss.generate import run_generate

        print(f"images → {run_generate(generate)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--submit",
        action="store_true",
        help="queue cache → train → generate on the GPU daemon",
    )
    mode.add_argument(
        "--run",
        choices=["cache", "train", "generate"],
        help="run one stage in this process",
    )
    p.add_argument("--src", default="post_image_dataset/resized")
    p.add_argument("--cache_dir", default="post_image_dataset/moss/example_12")
    p.add_argument("--lora", default="output/ckpt_moss/example_12.safetensors")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument(
        "--prompt",
        default="1girl, solo, long silver hair, red eyes, black school uniform, "
        "sitting on a classroom windowsill, sunset",
    )
    p.add_argument("--multiplier", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="output/tests/example_12")
    opts = p.parse_args()

    reqs = build_requests(opts)
    if opts.submit:
        submit(reqs)
    elif opts.run:
        run_inline(opts.run, *reqs)
    else:
        print_requests(reqs)


if __name__ == "__main__":
    main()
