"""Embedding-inversion entry-points (invert / invert-ref / test-invert / bench-inversion)."""

from __future__ import annotations

import os
import sys

from ._common import PY, run


def cmd_invert(extra):
    run(
        [
            PY,
            "scripts/inversion/invert_embedding.py",
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
            "--aggregate_by",
            "3",
            "--save_per_run",
            "--probe_functional",
            "--probe_blocks",
            "8,12,16,20",
            "--output_dir",
            "output/inversions",
            "--log_block_grads",
            *extra,
        ]
    )


def cmd_invert_ref(extra):
    """Reference-image inversion (scripts/inversion/invert_reference.py).

    Image source:
        - If REF_IMAGE is set, use that file.
        - Otherwise pick a random file from REF_IMAGE_DIR (default post_image_dataset).
          Re-running picks a new random image each time.
    Optional env: REF_TEMPLATE (default "a photo"), REF_K (default 8),
                  REF_STEPS (default 100), REF_LR (default 0.01),
                  REF_NAME (default "latest" -> output/ckpt/anima_ref_latest.safetensors),
                  REF_SAVE_PATH (overrides REF_NAME),
                  REF_SWAP (blocks_to_swap, default 0).
    """
    import glob
    import random

    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image:
        ref_dir = os.environ.get("REF_IMAGE_DIR", "post_image_dataset")
        candidates = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            candidates.extend(
                glob.glob(os.path.join(ref_dir, "**", ext), recursive=True)
            )
        if not candidates:
            print(
                f"Error: no images found in REF_IMAGE_DIR={ref_dir}/ and REF_IMAGE not set.\n"
                "       Either set REF_IMAGE=path/to/ref.png or point REF_IMAGE_DIR at a "
                "directory with .png/.jpg/.webp files.",
                file=sys.stderr,
            )
            sys.exit(1)
        ref_image = random.choice(candidates)
        print(f"  > random pick from {ref_dir}/: {ref_image}")
    ref_template = os.environ.get("REF_TEMPLATE", "a photo")
    ref_k = os.environ.get("REF_K", "8")
    ref_steps = os.environ.get("REF_STEPS", "100")
    ref_lr = os.environ.get("REF_LR", "0.01")
    ref_name = os.environ.get("REF_NAME", "latest")
    ref_save_path = os.environ.get(
        "REF_SAVE_PATH", f"output/ckpt/anima_ref_{ref_name}.safetensors"
    )
    ref_swap = os.environ.get("REF_SWAP", "0")
    run(
        [
            PY,
            "scripts/inversion/invert_reference.py",
            "--image",
            ref_image,
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--text_encoder",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--attn_mode",
            "flash",
            "--template",
            ref_template,
            "--num_tokens",
            ref_k,
            "--steps",
            ref_steps,
            "--lr",
            ref_lr,
            "--save_path",
            ref_save_path,
            "--blocks_to_swap",
            ref_swap,
            "--verify",
            *extra,
        ]
    )


def cmd_test_invert(extra):
    """Verify a saved embedding inversion via the interpret pass.

    INVERT_NAME env (default ``latest``) selects the inversion bundle name.
    """
    name = os.environ.get("INVERT_NAME", "latest")
    run(
        [
            PY,
            "scripts/inversion/interpret_inversion.py",
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--attn_mode",
            "flash",
            "--name",
            name,
            "--verify",
            "--verify_steps",
            "30",
            *extra,
        ]
    )


def cmd_bench_inversion(extra):
    """Run the inversion-stability benchmark. BENCH_INVERSIONS env (default 5)."""
    num = os.environ.get("BENCH_INVERSIONS", "5")
    run(
        [
            PY,
            "bench/inversion/inversion_stability.py",
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--num_inversions",
            num,
            "--steps",
            "100",
            "--lr",
            "0.01",
            *extra,
        ]
    )
