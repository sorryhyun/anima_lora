"""Default-dataset preprocessing: resize → VAE latents → text-embedding caches."""

from __future__ import annotations

import os

from ._common import PY, _path, run


def cmd_preprocess_resize(extra):
    run(
        [
            PY,
            "preprocess/resize_images.py",
            "--src",
            _path("source_image_dir", "image_dataset"),
            "--dst",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--no_copy_captions",
            *extra,
        ]
    )


def cmd_preprocess_vae(extra):
    run(
        [
            PY,
            "preprocess/cache_latents.py",
            "--dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
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
            "preprocess/cache_text_embeddings.py",
            "--dir",
            _path("source_image_dir", "image_dataset"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--caption_shuffle_variants",
            "4",
            "--caption_tag_dropout_rate",
            "0.1",
            *extra,
        ]
    )


def cmd_preprocess_pe(extra):
    """Cache PE-Core (or other registered) vision-encoder features.

    Reads pre-resized images from ``post_image_dataset/resized/`` (the
    standard LoRA pipeline source) and writes
    ``{stem}_anima_{encoder}.safetensors`` sidecars into the LoRA cache dir
    so the dataset's existing ``cache_dir`` lookup finds them.

    Consumed by methods that align against frozen vision features —
    currently REPA (--use_repa) and IP-Adapter when reading PE features off
    disk. PE_ENCODER env var overrides the encoder registry name (default
    ``pe`` = PE-Core-L14-336).
    """
    encoder = os.environ.get("PE_ENCODER", "pe")
    run(
        [
            PY,
            "preprocess/cache_pe_encoder.py",
            "--dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--encoder",
            encoder,
            *extra,
        ]
    )


def cmd_preprocess(extra):
    cmd_preprocess_resize(extra)
    cmd_preprocess_vae(extra)
    cmd_preprocess_te(extra)
