"""Training entry-points: lora family, apex, postfix, ip-adapter, easycontrol.

Each ``cmd_*`` is a thin shim that translates env vars + extra argv into the
right ``train.py`` (via ``accelerate launch``) or ``preprocess/*.py`` call.
"""

from __future__ import annotations

import os
import sys

from ._common import PY, ROOT, _preset, run, train


def cmd_lora(extra):
    train("lora", extra)


def cmd_lora_fast(extra):
    train("lora", extra, preset=_preset("fast_16gb"))


def cmd_lora_low_vram(extra):
    train("lora", extra, preset=_preset("low_vram"))


def cmd_lora_half(extra):
    train("lora", extra, preset=_preset("half"))


def cmd_lora_gui(extra):
    """Train from configs/gui-methods/<variant>.toml.

    Variant is taken from GUI_PRESETS env var, falling back to the first
    positional extra arg (``python tasks.py lora-gui tlora ...``), then to
    ``lora`` (plain). Extra args after the variant are forwarded as usual.
    """
    variant = os.environ.get("GUI_PRESETS")
    if not variant and extra and not extra[0].startswith("-"):
        variant = extra[0]
        extra = extra[1:]
    variant = variant or "lora"

    expected = ROOT / "configs" / "gui-methods" / f"{variant}.toml"
    if not expected.exists():
        available = sorted(
            p.stem for p in (ROOT / "configs" / "gui-methods").glob("*.toml")
        )
        print(
            f"Unknown gui-methods variant: {variant!r}\n"
            f"Available: {', '.join(available)}",
            file=sys.stderr,
        )
        sys.exit(1)

    train(variant, extra, methods_subdir="gui-methods")


def cmd_apex(extra):
    train("apex", extra)


def cmd_postfix(extra):
    train("postfix", extra)


def cmd_ip_adapter(extra):
    train("ip_adapter", extra)


def cmd_ip_adapter_cache(extra):
    """Pre-cache PE-Core patch features for IP-Adapter training.

    Writes ``{stem}_anima_pe.safetensors`` files into
    ``post_image_dataset/ip-adapter/``. IP_ENCODER env var overrides the
    registry name (default ``pe``).
    """
    encoder = os.environ.get("IP_ENCODER", "pe")
    run(
        [
            PY,
            "preprocess/cache_pe_encoder.py",
            "--dir",
            "ip-adapter-dataset",
            "--cache_dir",
            "post_image_dataset/ip-adapter",
            "--encoder",
            encoder,
            *extra,
        ]
    )


def cmd_ip_adapter_preprocess(extra):
    """Full IP-Adapter preprocess: VAE latents + text-encoder outputs + PE features.

    Source: ``ip-adapter-dataset/``  Caches: ``post_image_dataset/ip-adapter/``.
    """
    encoder = os.environ.get("IP_ENCODER", "pe")
    src = "ip-adapter-dataset"
    dst = "post_image_dataset/ip-adapter"
    run(
        [
            PY,
            "preprocess/cache_latents.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
        ]
    )
    run(
        [
            PY,
            "preprocess/cache_text_embeddings.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--caption_shuffle_variants",
            "4",
        ]
    )
    run(
        [
            PY,
            "preprocess/cache_pe_encoder.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--encoder",
            encoder,
        ]
    )


def cmd_easycontrol(extra):
    train("easycontrol", extra)


def cmd_easycontrol_preprocess(extra):
    """Full EasyControl preprocess: VAE latents + text-encoder outputs.

    Source: ``easycontrol-dataset/``  Caches: ``post_image_dataset/easycontrol/``.
    """
    src = "easycontrol-dataset"
    dst = "post_image_dataset/easycontrol"
    run(
        [
            PY,
            "preprocess/cache_latents.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
        ]
    )
    run(
        [
            PY,
            "preprocess/cache_text_embeddings.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--caption_shuffle_variants",
            "4",
        ]
    )
