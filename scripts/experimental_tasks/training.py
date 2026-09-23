"""Experimental training entry-points (soft tokens, BYG data, CJK distill).

Wired up under ``make exp-*`` / ``python tasks.py exp-*``. Each ``cmd_*`` is a
thin shim that translates env vars + extra argv into a ``train.py`` or script
call.
"""

from __future__ import annotations

from scripts.tasks._common import (
    PY,
    run,
    train,
)


def cmd_soft_tokens(extra):
    train("soft_tokens", extra)


def cmd_distill_cjk(extra):
    """Distill the extended T5-side vocab rows (project/cjk_aware_anima 2b).

    Run the gates in order — ``--mode oracle`` (loop + trimming invariant),
    ``--mode capacity`` (can the ext rows express the teacher at all?), then
    ``--mode train``. Output is a vocab pack (``ext_embed.safetensors`` +
    ``.json``), not a LoRA.
    """
    run([PY, "-m", "scripts.distill_cjk.distill", *extra])


def cmd_byg_data(extra):
    """Build BYG edit-tuple sidecars (tag-swap) into ``post_image_dataset/byg/``.

    One offline pass over the captioned corpus emitting
    ``<stem>_byg.safetensors`` (4 encoded role conditionings) per image. Pass
    ``--limit N`` for a quick smoke subset, ``--overwrite`` to rebuild.
    """
    run(
        [
            PY,
            "scripts/byg/build_edit_tuples.py",
            "--dir",
            "image_dataset",
            "--cache_dir",
            "post_image_dataset/byg",
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            *extra,
        ]
    )
