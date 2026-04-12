#!/usr/bin/env python3
"""Cross-platform task runner — replaces Makefile for Windows compatibility.

Usage:
    python tasks.py <command> [extra args...]

Examples:
    python tasks.py lora
    python tasks.py lora --network_dim 32 --max_train_epochs 64
    python tasks.py test
    python tasks.py test-spectrum
    python tasks.py download-models
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LORA_DIR = ROOT.parent / "comfy" / "ComfyUI" / "models" / "loras"


def latest_output(prefix: str = "") -> Path:
    """Return the most recently modified .safetensors file in output/ matching prefix."""
    outputs = sorted(
        (
            f
            for f in (ROOT / "output").glob("*.safetensors")
            if f.name.startswith(prefix)
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        label = f"'{prefix}*.safetensors'" if prefix else "*.safetensors"
        print(f"No {label} files found in output/", file=sys.stderr)
        sys.exit(1)
    return outputs[0]


def latest_lora() -> Path:
    return latest_output()


def run(cmd: list[str], **kwargs):
    """Run a subprocess, exit on failure."""
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=kwargs.pop("cwd", ROOT), **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


def accelerate_launch(*args: str):
    """Launch training via accelerate with extra CLI args forwarded."""
    run(
        [
            "accelerate",
            "launch",
            "--num_cpu_threads_per_process",
            "3",
            "--mixed_precision",
            "bf16",
            "train.py",
            *args,
        ]
    )


# ── Training ──────────────────────────────────────────────────────────


def cmd_lora(extra):
    accelerate_launch("--config_file", "configs/training_config_plain.toml", *extra)


def cmd_lora_fast(extra):
    accelerate_launch("--config_file", "configs/training_config_fast_16gb.toml", *extra)


def cmd_lora_low_vram(extra):
    accelerate_launch("--config_file", "configs/training_config_low_vram.toml", *extra)


def cmd_dora(extra):
    accelerate_launch("--config_file", "configs/training_config_dora.toml", *extra)


def cmd_tdora(extra):
    accelerate_launch(
        "--config_file", "configs/training_config_doratimestep.toml", *extra
    )


def cmd_tlora(extra):
    accelerate_launch("--config_file", "configs/training_config_tlora.toml", *extra)


def cmd_hydralora(extra):
    accelerate_launch("--config_file", "configs/training_config_hydralora.toml", *extra)


def cmd_postfix(extra):
    accelerate_launch("--config_file", "configs/training_config_postfix.toml", *extra)


def cmd_prefix(extra):
    accelerate_launch("--config_file", "configs/training_config_prefix.toml", *extra)


# ── Inference ─────────────────────────────────────────────────────────

INFERENCE_BASE = [
    "python",
    "inference.py",
    "--dit",
    "models/diffusion_models/anima-preview3-base.safetensors",
    "--text_encoder",
    "models/text_encoders/qwen_3_06b_base.safetensors",
    "--vae",
    "models/vae/qwen_image_vae.safetensors",
    "--vae_chunk_size",
    "64",
    "--vae_disable_cache",
    "--attn_mode",
    "flash",  # flash4 not supported yet (flash-attention-sm120 disabled)
    "--lora_multiplier",
    "1.0",
    "--prompt",
    "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top"
    " and denim shorts is standing outdoors. She's holding a rectangular sign out in"
    ' front of her that reads "ANIMA". She\'s looking at the viewer with a smile. The'
    " background features some trees and blue sky with clouds.",
    "--negative_prompt",
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
    "--image_size",
    "1024",
    "1024",
    "--infer_steps",
    "30",
    "--flow_shift",
    "1.0",
    "--sampler",
    "er_sde",
    "--guidance_scale",
    "4.0",
    "--seed",
    "42",
    "--save_path",
    "test_output",
]


def cmd_test(extra):
    run([*INFERENCE_BASE, "--lora_weight", str(latest_lora()), *extra])


def cmd_test_prefix(extra):
    run(
        [*INFERENCE_BASE, "--prefix_weight", str(latest_output("anima_prefix")), *extra]
    )


def cmd_test_postfix(extra):
    run(
        [
            *INFERENCE_BASE,
            "--postfix_weight",
            str(latest_output("anima_postfix")),
            *extra,
        ]
    )


def cmd_test_spectrum(extra):
    run(
        [
            *INFERENCE_BASE,
            "--lora_weight",
            str(latest_lora()),
            "--spectrum",
            "--spectrum_window_size",
            "2.0",
            "--spectrum_flex_window",
            "0.25",
            "--spectrum_warmup",
            "7",
            "--spectrum_w",
            "0.3",
            "--spectrum_m",
            "3",
            "--spectrum_lam",
            "0.1",
            "--spectrum_stop_caching_step",
            "29",
            "--spectrum_calibration",
            "0.0",
            *extra,
        ]
    )


# ── Utilities ─────────────────────────────────────────────────────────


def cmd_sync(_extra):
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "output").glob("*.safetensors"):
        shutil.copy2(f, LORA_DIR / f.name)
        print(f"  Copied {f.name}")


def cmd_step(extra):
    run(["python", "scripts/graft_step.py", *extra])


def cmd_preprocess_resize(extra):
    run(
        [
            "python",
            "scripts/resize_images.py",
            "--src",
            "image_dataset",
            "--dst",
            "post_image_dataset",
            *extra,
        ]
    )


def cmd_preprocess_vae(extra):
    run(
        [
            "python",
            "scripts/cache_latents.py",
            "--dir",
            "post_image_dataset",
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
            *extra,
        ]
    )


def cmd_preprocess_te(extra):
    run(
        [
            "python",
            "scripts/cache_text_embeddings.py",
            "--dir",
            "post_image_dataset",
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--caption_shuffle_variants",
            "16",
            *extra,
        ]
    )


def cmd_preprocess(extra):
    cmd_preprocess_resize(extra)
    cmd_preprocess_vae(extra)
    cmd_preprocess_te(extra)


def cmd_comfy_batch(extra):
    workflow = extra[0] if extra else "workflows/lora-batch.json"
    remaining = extra[1:] if extra else []
    run(["python", "scripts/comfy_batch.py", workflow, *remaining])


# ── Downloads ─────────────────────────────────────────────────────────


def cmd_download_sam3(_extra):
    (ROOT / "models" / "sam3").mkdir(parents=True, exist_ok=True)
    run(["huggingface-cli", "download", "facebook/sam3", "--local-dir", "models/sam3"])


def cmd_download_mit(_extra):
    (ROOT / "models" / "mit").mkdir(parents=True, exist_ok=True)
    run(
        [
            "huggingface-cli",
            "download",
            "a-b-c-x-y-z/Manga-Text-Segmentation-2025",
            "model.pth",
            "--local-dir",
            "models/mit",
        ]
    )


def cmd_download_anima(_extra):
    for d in ["diffusion_models", "text_encoders", "vae"]:
        (ROOT / "models" / d).mkdir(parents=True, exist_ok=True)
    run(
        [
            "huggingface-cli",
            "download",
            "circlestone-labs/Anima",
            "split_files/diffusion_models/anima-preview3-base.safetensors",
            "split_files/text_encoders/qwen_3_06b_base.safetensors",
            "split_files/vae/qwen_image_vae.safetensors",
            "--local-dir",
            "models",
            "--include",
            "split_files/*",
        ]
    )
    split = ROOT / "models" / "split_files"
    for subdir in ["diffusion_models", "text_encoders", "vae"]:
        src = split / subdir
        dst = ROOT / "models" / subdir
        if src.exists():
            for f in src.iterdir():
                shutil.move(str(f), str(dst / f.name))
    if split.exists():
        shutil.rmtree(split)


def cmd_download_models(_extra):
    cmd_download_anima(_extra)
    cmd_download_sam3(_extra)
    cmd_download_mit(_extra)


# ── Masking ───────────────────────────────────────────────────────────


def cmd_mask_sam(extra):
    run(
        [
            "python",
            "scripts/generate_masks.py",
            "--config",
            "configs/sam_mask.yaml",
            "--image-dir",
            "post_image_dataset",
            "--mask-dir",
            "masks_sam",
            "--checkpoint",
            "models/sam3/sam3.pt",
            "--batch-size",
            "2",
            *extra,
        ]
    )


def cmd_mask_mit(extra):
    run(
        [
            "python",
            "scripts/generate_masks_mit.py",
            "--image-dir",
            "post_image_dataset",
            "--mask-dir",
            "masks_mit",
            "--model-path",
            "models/mit/model.pth",
            *extra,
        ]
    )


def cmd_mask(extra):
    if not (ROOT / "masks_sam").is_dir():
        cmd_mask_sam([])
    if not (ROOT / "masks_mit").is_dir():
        cmd_mask_mit([])
    run(
        [
            "python",
            "scripts/merge_masks.py",
            "masks_sam",
            "masks_mit",
            "--output-dir",
            "masks",
            *extra,
        ]
    )


def cmd_mask_clean(_extra):
    for d in ["masks", "masks_sam", "masks_mit"]:
        p = ROOT / d
        if p.exists():
            shutil.rmtree(p)
            print(f"  Removed {d}/")


def cmd_gui(_extra):
    run(["python", "-m", "gui"])


def cmd_invert(extra):
    run(
        [
            "python",
            "scripts/invert_embedding.py",
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--attn_mode",
            "flash",
            "--image_dir",
            "post_image_dataset",
            "--num_images",
            "10",
            "--shuffle",
            "--steps",
            "500",
            "--lr",
            "0.01",
            "--output_dir",
            "inversions",
            "--log_block_grads",
            *extra,
        ]
    )


# ── CLI ───────────────────────────────────────────────────────────────

COMMANDS = {
    "lora": (cmd_lora, "Standard LoRA training"),
    "lora-fast": (cmd_lora_fast, "Fast LoRA training (16GB, no block swap)"),
    "lora-low-vram": (cmd_lora_low_vram, "LoRA training (low VRAM)"),
    "dora": (cmd_dora, "DoRA training"),
    "tdora": (cmd_tdora, "DoRA + timestep masking"),
    "tlora": (cmd_tlora, "T-LoRA: OrthoLoRA + timestep masking"),
    "hydralora": (cmd_hydralora, "HydraLoRA: multi-style MoE routing"),
    "postfix": (cmd_postfix, "Postfix tuning (LLM adapter cross-attn)"),
    "prefix": (cmd_prefix, "Prefix tuning (T5-space, cache-compatible)"),
    "test": (cmd_test, "Inference with latest LoRA"),
    "test-prefix": (cmd_test_prefix, "Inference with latest prefix weight"),
    "test-postfix": (cmd_test_postfix, "Inference with latest postfix weight"),
    "test-spectrum": (cmd_test_spectrum, "Spectrum-accelerated inference"),
    "sync": (cmd_sync, "Copy outputs to ComfyUI loras dir"),
    "step": (cmd_step, "Run one GRAFT iteration"),
    "preprocess": (
        cmd_preprocess,
        "Full preprocessing (resize + VAE + text embeddings)",
    ),
    "preprocess-resize": (cmd_preprocess_resize, "Resize images to bucket resolutions"),
    "preprocess-vae": (cmd_preprocess_vae, "Cache VAE latents"),
    "preprocess-te": (cmd_preprocess_te, "Cache text encoder embeddings"),
    "comfy-batch": (cmd_comfy_batch, "Run ComfyUI batch workflow"),
    "download-models": (cmd_download_models, "Download all models"),
    "download-anima": (cmd_download_anima, "Download Anima model"),
    "download-sam3": (cmd_download_sam3, "Download SAM3 model"),
    "download-mit": (cmd_download_mit, "Download MIT model"),
    "mask": (cmd_mask, "Generate SAM3 + MIT masks, then merge"),
    "mask-sam": (cmd_mask_sam, "Generate SAM3 masks only"),
    "mask-mit": (cmd_mask_mit, "Generate MIT masks only"),
    "mask-clean": (cmd_mask_clean, "Remove all generated masks"),
    "gui": (cmd_gui, "Launch PySide6 GUI"),
    "invert": (cmd_invert, "Embedding inversion (image → text embedding)"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python tasks.py <command> [extra args...]\n")
        print("Commands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:20s} {desc}")
        print("\nExtra arguments are forwarded to the underlying command.")
        print("Example: python tasks.py lora --network_dim 32 --max_train_epochs 64")
        sys.exit(0)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run 'python tasks.py --help' for available commands", file=sys.stderr)
        sys.exit(1)

    extra = sys.argv[2:]
    fn, _ = COMMANDS[command]
    fn(extra)


if __name__ == "__main__":
    main()
