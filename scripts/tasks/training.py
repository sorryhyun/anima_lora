"""Training entry-points: lora family, apex, postfix, ip-adapter, easycontrol.

Each ``cmd_*`` is a thin shim that translates env vars + extra argv into the
right ``train.py`` (via ``accelerate launch``) or ``preprocess/*.py`` call.
"""

from __future__ import annotations

import os
import sys

from . import preprocess as _preprocess
from ._common import PY, ROOT, run, train


def cmd_lora(extra):
    train("lora", extra)


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


def cmd_ip_adapter_preprocess(extra):
    """Full IP-Adapter preprocess.

    IP-Adapter shares the LoRA pipeline's data layout — source images live in
    ``image_dataset/`` and caches in ``post_image_dataset/lora/``. This is just
    a convenience alias for ``make preprocess`` + ``make preprocess-pe`` so the
    GUI's IP-Adapter tab and ``make ip-adapter-preprocess`` keep working.
    """
    _preprocess.cmd_preprocess(extra)
    _preprocess.cmd_preprocess_pe(extra)


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
