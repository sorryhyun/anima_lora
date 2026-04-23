#!/usr/bin/env python3
"""Benchmark 4 LoRA variants — train each, then sweep inference across seeds.

Variants (all at r=16, 2 epochs, sample_ratio=0.2):
    lora         — plain LoRA                         (gui-methods/lora.toml)
    ortholora    — OrthoLoRA                          (gui-methods/ortholora.toml)
    tlora        — T-LoRA without ortho               (gui-methods/tlora.toml + use_ortho=false)
    ortho_tlora  — OrthoLoRA + T-LoRA (ortho+tstep)   (gui-methods/tlora.toml as-is)

Inference mirrors Makefile::TEST_COMMON (flash attn, er_sde, 30 steps, cfg=4,
1024²) and runs once per variant per seed in {41, 42, 43}.

Outputs:
    output/ckpt/anima_bench_<variant>.safetensors      (weights)
    output/bench/<variant>/<time>_<seed>.png            (images)

Usage:
    python scripts/bench_methods.py                          # full sweep
    python scripts/bench_methods.py --variants lora tlora    # subset
    python scripts/bench_methods.py --skip-train             # reuse ckpts, re-infer
    python scripts/bench_methods.py --skip-infer             # train only
    PRESET=low_vram python scripts/bench_methods.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# (bench tag, gui-methods variant file, extra train-arg overrides)
VARIANTS: list[tuple[str, str, list[str]]] = [
    ("lora",        "lora",      []),
    ("ortholora",   "ortholora", []),
    ("tlora",       "tlora",     ["--network_args", "use_ortho=false"]),
    ("ortho_tlora", "tlora",     []),
]

SEEDS: tuple[int, ...] = (41, 42, 43)

COMMON_TRAIN_ARGS = [
    "--network_dim", "16",
    "--network_alpha", "16",
    "--learning_rate", "5e-5",
    "--max_train_epochs", "2",
    "--save_every_n_epochs", "2",
    "--sample_ratio", "0.2",
    # Fix training seed so every variant sees the same shuffle / noise / caption
    # rotation — isolates method differences from dataloader RNG.
    # (sample_ratio's 20% pick is already deterministic via validation_seed.)
    "--seed", "42",
]

# Mirrors Makefile::TEST_COMMON.
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
    "--infer_steps", "30",
    "--flow_shift", "1.0",
    "--sampler", "er_sde",
    "--guidance_scale", "4.0",
]


def run(cmd: list[str]) -> None:
    print(f"  > {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[bench] command failed (rc={result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def train_variant(tag: str, gui_variant: str, extra: list[str], preset: str) -> None:
    output_name = f"anima_bench_{tag}"
    run([
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "3",
        "--mixed_precision", "bf16",
        "train.py",
        "--method", gui_variant,
        "--methods_subdir", "gui-methods",
        "--preset", preset,
        *COMMON_TRAIN_ARGS,
        "--output_name", output_name,
        *extra,
    ])


def infer_variant(tag: str) -> None:
    ckpt = ROOT / "output" / "ckpt" / f"anima_bench_{tag}.safetensors"
    if not ckpt.is_file():
        print(f"[bench] checkpoint missing: {ckpt} — skipping inference for {tag}",
              file=sys.stderr)
        return
    save_dir = ROOT / "output" / "bench" / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        run([
            *INFERENCE_BASE,
            "--lora_weight", str(ckpt),
            "--seed", str(seed),
            "--save_path", str(save_dir),
        ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default=os.environ.get("PRESET", "default"),
                    help="Hardware preset passed to train.py (default: $PRESET or 'default').")
    ap.add_argument("--variants", nargs="+", default=[v[0] for v in VARIANTS],
                    choices=[v[0] for v in VARIANTS],
                    help="Subset of variants to run.")
    ap.add_argument("--skip-train", action="store_true",
                    help="Reuse existing output/ckpt/anima_bench_*.safetensors; only run inference.")
    ap.add_argument("--skip-infer", action="store_true", help="Train only; no inference sweep.")
    args = ap.parse_args()

    selected = [v for v in VARIANTS if v[0] in args.variants]

    if not args.skip_train:
        for tag, gui_variant, extra in selected:
            print(f"\n=== [bench] train {tag} ({gui_variant}) ===", flush=True)
            train_variant(tag, gui_variant, extra, args.preset)

    if not args.skip_infer:
        for tag, _, _ in selected:
            print(f"\n=== [bench] infer {tag} seeds={SEEDS} ===", flush=True)
            infer_variant(tag)


if __name__ == "__main__":
    main()
