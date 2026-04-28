#!/usr/bin/env python3
"""APEX cold-start bench — train rank-0 APEX on the local dataset, measure NFE.

Phase 0 measured −48% NFE=1 W1 from cold start in a 2D toy. This bench checks
whether that regression carries over to real DiT training when the dataset
overlaps the base model's pretraining distribution. Trains 2 cold-start
variants (warmup ratios 0.20 / 0.50) and sweeps inference at NFE ∈ {1, 2, 4}.

Variants (both cold-start, both rank 64):
    cold_w20   — apex_warmup_ratio = 0.20, max_train_epochs = 2
    cold_w50   — apex_warmup_ratio = 0.50, max_train_epochs = 2

Both use ``configs/gui-methods/apex_coldstart.toml`` as the base config.
``cold_w50`` overrides only ``--apex_warmup_ratio`` via CLI — same budget, same
post-warmup APEX schedule, just more pure-L_sup time before adversarial terms
kick in.

Inference mirrors ``cmd_test_apex`` (euler sampler, guidance_scale=1.0,
flash attn, 1024², seed 42 in the base) but sweeps NFE ∈ {1, 2, 4} ×
seed ∈ {41, 42, 43}.

Outputs:
    output/ckpt/anima_apex_bench_<variant>.safetensors                    (weights)
    output/bench/apex_coldstart/<variant>/nfe<N>/<time>_<seed>.png        (images)

Usage:
    python scripts/bench_apex_coldstart.py                          # full run
    python scripts/bench_apex_coldstart.py --variants cold_w20      # one variant
    python scripts/bench_apex_coldstart.py --skip-train             # reuse ckpts
    python scripts/bench_apex_coldstart.py --skip-infer             # train only
    PRESET=low_vram python scripts/bench_apex_coldstart.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# (bench tag, extra train-arg overrides on top of gui-methods/apex_coldstart.toml)
VARIANTS: list[tuple[str, list[str]]] = [
    ("cold_w20", []),
    ("cold_w50", ["--apex_warmup_ratio", "0.50"]),
]

NFES: tuple[int, ...] = (1, 2, 4)
SEEDS: tuple[int, ...] = (41, 42, 43)

# Mirrors scripts/tasks/_common.py::INFERENCE_BASE + cmd_test_apex's apex flags
# (euler / guidance 1.0 / NFE swept below). Kept verbatim here so the bench is
# self-contained and a future _common.py refactor can't silently shift results.
INFERENCE_BASE = [
    PY, "inference.py",
    "--dit", "models/diffusion_models/anima-preview3-base.safetensors",
    "--text_encoder", "models/text_encoders/qwen_3_06b_base.safetensors",
    "--vae", "models/vae/qwen_image_vae.safetensors",
    "--vae_chunk_size", "64", "--vae_disable_cache",
    "--attn_mode", "flash",
    "--lora_multiplier", "1.0",
    "--prompt",
    "masterpiece, best quality, score_7, safe. An anime girl wearing a black "
    "tank-top and denim shorts is standing outdoors. She's holding a "
    "rectangular sign out in front of her that reads \"ANIMA\". She's looking "
    "at the viewer with a smile. The background features some trees and blue "
    "sky with clouds.",
    "--negative_prompt",
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
    "--image_size", "1024", "1024",
    "--flow_shift", "1.0",
    "--sampler", "euler",
    "--guidance_scale", "1.0",
]


def run(cmd: list[str]) -> None:
    print(f"  > {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[bench] command failed (rc={result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def train_variant(tag: str, extra: list[str], preset: str) -> None:
    output_name = f"anima_apex_bench_{tag}"
    run([
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "3",
        "--mixed_precision", "bf16",
        "train.py",
        "--method", "apex_coldstart",
        "--methods_subdir", "gui-methods",
        "--preset", preset,
        "--seed", "42",
        "--output_name", output_name,
        *extra,
    ])


def infer_variant(tag: str) -> None:
    ckpt = ROOT / "output" / "ckpt" / f"anima_apex_bench_{tag}.safetensors"
    if not ckpt.is_file():
        print(
            f"[bench] checkpoint missing: {ckpt} — skipping inference for {tag}",
            file=sys.stderr,
        )
        return
    base_dir = ROOT / "output" / "bench" / "apex_coldstart" / tag
    for nfe in NFES:
        # One subdir per NFE so the auto-generated {time}_{seed}.png names
        # from save_images() don't blur which step count produced them.
        nfe_dir = base_dir / f"nfe{nfe}"
        nfe_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            run([
                *INFERENCE_BASE,
                "--lora_weight", str(ckpt),
                "--infer_steps", str(nfe),
                "--seed", str(seed),
                "--save_path", str(nfe_dir),
            ])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--preset",
        default=os.environ.get("PRESET", "default"),
        help="Hardware preset passed to train.py (default: $PRESET or 'default').",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=[v[0] for v in VARIANTS],
        choices=[v[0] for v in VARIANTS],
        help="Subset of variants to run.",
    )
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse existing output/ckpt/anima_apex_bench_*.safetensors; only run inference.",
    )
    ap.add_argument(
        "--skip-infer", action="store_true", help="Train only; no inference sweep."
    )
    args = ap.parse_args()

    selected = [v for v in VARIANTS if v[0] in args.variants]

    if not args.skip_train:
        for tag, extra in selected:
            print(f"\n=== [bench] train {tag} ===", flush=True)
            train_variant(tag, extra, args.preset)

    if not args.skip_infer:
        for tag, _ in selected:
            print(f"\n=== [bench] infer {tag} NFE={NFES} seeds={SEEDS} ===", flush=True)
            infer_variant(tag)


if __name__ == "__main__":
    main()
