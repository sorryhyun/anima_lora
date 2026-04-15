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

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def _preset(default: str = "default") -> str:
    return os.environ.get("PRESET", default)


def latest_output(prefix: str = "", exclude: str | None = None) -> Path:
    """Return the most recently modified .safetensors file in output/ matching prefix.

    If `exclude` is given, any filename containing that substring is skipped. Useful
    to disambiguate overlapping prefixes (e.g. anima_postfix vs anima_postfix_exp).
    HydraLoRA multi-head sibling files (`*_moe.safetensors`) and backup files
    (containing `.bak.`) are always excluded.
    """
    outputs = sorted(
        (
            f
            for f in (ROOT / "output").glob("*.safetensors")
            if f.name.startswith(prefix)
            and not f.name.endswith("_moe.safetensors")
            and ".bak." not in f.name
            and (exclude is None or exclude not in f.name)
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


def latest_hydra() -> Path:
    """Latest HydraLoRA multi-head file (`anima_hydra*_moe.safetensors`).

    Router-live inference needs the moe sibling, not the baked-down file.
    The baked-down `anima_hydra*.safetensors` collapses experts to a uniform
    average, which defeats the layer-local routing objective — use it only
    as a ComfyUI-compat fallback.
    """
    outputs = sorted(
        (
            f
            for f in (ROOT / "output").glob("anima_hydra*_moe.safetensors")
            if ".bak." not in f.name
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        print(
            "No 'anima_hydra*_moe.safetensors' files found in output/ "
            "(train with `make hydralora` to produce one)",
            file=sys.stderr,
        )
        sys.exit(1)
    return outputs[0]


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


def train(method: str, extra, preset: str | None = None):
    """Launch training for a given method + preset (PRESET env overrides default)."""
    accelerate_launch(
        "--method", method, "--preset", preset or _preset(), *extra
    )


# ── Training ──────────────────────────────────────────────────────────


def cmd_lora(extra):
    train("lora", extra)


def cmd_lora_fast(extra):
    train("lora", extra, preset=_preset("fast_16gb"))


def cmd_lora_low_vram(extra):
    train("lora", extra, preset=_preset("low_vram"))


def cmd_dora(extra):
    train("dora", extra)


def cmd_tdora(extra):
    train("doratimestep", extra)


def cmd_tlora(extra):
    train("tlora", extra)


def cmd_hydralora(extra):
    train("hydralora", extra)


def cmd_apex(extra):
    train("apex", extra)


def cmd_postfix(extra):
    train("postfix", extra)


def cmd_postfix_exp(extra):
    train("postfix_exp", extra)


def cmd_postfix_func(extra):
    train("postfix_func", extra)


def cmd_prefix(extra):
    train("prefix", extra)


# ── Inference ─────────────────────────────────────────────────────────

INFERENCE_BASE = [
    PY,
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


def cmd_test_apex(extra):
    run(
        [
            *INFERENCE_BASE,
            "--lora_weight",
            str(latest_output("anima_apex")),
            "--infer_steps",
            "4",
            "--guidance_scale",
            "1.0",
            "--sampler",
            "euler",
            *extra,
        ]
    )


def cmd_test_hydra(extra):
    # Uses the moe sibling (router-live); static-merge is auto-skipped in
    # library/inference_pipeline.py:_is_hydra_moe detection.
    run([*INFERENCE_BASE, "--lora_weight", str(latest_hydra()), *extra])


def cmd_test_prefix(extra):
    run(
        [*INFERENCE_BASE, "--prefix_weight", str(latest_output("anima_prefix")), *extra]
    )


def cmd_test_postfix(extra):
    # exclude both _exp and _func so the vanilla postfix target doesn't grab them
    outputs = sorted(
        (
            f
            for f in (ROOT / "output").glob("anima_postfix*.safetensors")
            if "_exp" not in f.name and "_func" not in f.name
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        print("No 'anima_postfix*.safetensors' files found in output/", file=sys.stderr)
        sys.exit(1)
    run(
        [
            *INFERENCE_BASE,
            "--postfix_weight",
            str(outputs[0]),
            *extra,
        ]
    )


def cmd_test_postfix_exp(extra):
    run(
        [
            *INFERENCE_BASE,
            "--postfix_weight",
            str(latest_output("anima_postfix_exp")),
            *extra,
        ]
    )


def cmd_test_postfix_func(extra):
    run(
        [
            *INFERENCE_BASE,
            "--postfix_weight",
            str(latest_output("anima_postfix_func")),
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


def cmd_step(extra):
    run([PY, "scripts/graft_step.py", *extra])


def cmd_preprocess_resize(extra):
    run(
        [
            PY,
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
            PY,
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
            PY,
            "scripts/cache_text_embeddings.py",
            "--dir",
            "post_image_dataset",
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--caption_shuffle_variants",
            "8",
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
    run([PY, "scripts/comfy_batch.py", workflow, *remaining])


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
            PY,
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
            PY,
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
            PY,
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
    run([PY, "-m", "gui"])


def cmd_invert(extra):
    run(
        [
            PY,
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
    "apex": (cmd_apex, "APEX distillation (condition-shift self-adversarial)"),
    "postfix": (cmd_postfix, "Postfix tuning (LLM adapter cross-attn)"),
    "postfix-exp": (
        cmd_postfix_exp,
        "Postfix experiment: caption-conditional MLP, end-of-sequence splice",
    ),
    "postfix-func": (
        cmd_postfix_func,
        "Postfix-exp + functional MSE loss vs inversion runs",
    ),
    "prefix": (cmd_prefix, "Prefix tuning (T5-space, cache-compatible)"),
    "test": (cmd_test, "Inference with latest LoRA"),
    "test-apex": (cmd_test_apex, "Inference with latest APEX LoRA"),
    "test-hydra": (cmd_test_hydra, "Inference with latest HydraLoRA moe (router-live)"),
    "test-prefix": (cmd_test_prefix, "Inference with latest prefix weight"),
    "test-postfix": (cmd_test_postfix, "Inference with latest postfix weight"),
    "test-postfix-exp": (
        cmd_test_postfix_exp,
        "Inference with latest postfix-exp weight",
    ),
    "test-postfix-func": (
        cmd_test_postfix_func,
        "Inference with latest postfix-func weight",
    ),
    "test-spectrum": (cmd_test_spectrum, "Spectrum-accelerated inference"),
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
