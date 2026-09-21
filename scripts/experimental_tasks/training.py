"""Experimental training entry-points (soft tokens, BYG, CJK distill).

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


def cmd_byg(extra):
    """BYG — Bootstrap Your Generator unpaired instruction editing.

    Plain rank-64 LoRA trained with a multi-forward unpaired objective (bootstrap
    rollout + DDS prior + cycle + identity), conditioned on a parameter-free
    token-concat source latent. Reads ``configs/methods/byg.toml``.

    Run ``exp-byg-data`` first to build the per-image edit-tuple sidecars under
    ``post_image_dataset/byg/``. The image VAE/TE caches are the standard
    ``preprocess`` ones (the source image IS the training image).
    """
    train("byg", extra)


def cmd_cjk_cache(extra):
    """Stage the CJK distillation cache (Qwen hidden states + teacher output).

    One pass over ``post_image_dataset/cjk_distill/pairs.jsonl``. The teacher
    is frozen and both arms share the Qwen side, so this is computed once and
    reused by every ``exp-distill-cjk`` arm — rebuild only when the corpus, the
    tokenizer, or the ext *mapping* changes (not when the ext *values* do).
    The holdout split additionally caches the all-EN reference and the
    unk-wall arms, which is what makes the recovery-fraction metric possible
    without a text encoder at train time.
    """
    run([PY, "-m", "scripts.distill_cjk.cache", *extra])


def cmd_distill_cjk(extra):
    """Distill the extended T5-side vocab rows (project/cjk_aware_anima 2b).

    Run the gates in order — ``--mode oracle`` (loop + trimming invariant),
    ``--mode capacity`` (can the ext rows express the teacher at all?), then
    ``--mode train``. Output is a vocab pack (``ext_embed.safetensors`` +
    ``.json``), not a LoRA.
    """
    run([PY, "-m", "scripts.distill_cjk.distill", *extra])


def cmd_cjk_gates(extra):
    """Phase-2b closing gates G3 + G4 (project/cjk_aware_anima).

    ``G3`` measures the teacher ceiling per register on the whole holdout.
    ``G4`` is corpus health (token-count ratio, occurrence-weighted span
    provenance, ext-row visit bands) plus the trust ablation.

    The driver lives with the line (``project/cjk_aware_anima/gates/``). Its
    siblings there run the same way, by path:
    ``g2.py`` (the loss × parameterization cross-tab) and ``g5.py`` (what the
    settled objective's *exact optimum* scores on the Phase-2c gate).
    """
    run([PY, "project/cjk_aware_anima/gates/g34.py", *extra])


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
