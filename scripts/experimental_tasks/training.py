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
    """Distill the extended T5-side vocab rows (the frozen cjk_aware_anima line, 2b).

    Run the gates in order — ``--mode oracle`` (loop + trimming invariant),
    ``--mode capacity`` (can the ext rows express the teacher at all?), then
    ``--mode train``. Output is a vocab pack (``ext_embed.safetensors`` +
    ``.json``), not a LoRA.
    """
    run([PY, "-m", "scripts.distill_cjk.distill", *extra])


CJK_CORPUS_STAGES = (
    "wikidata_lexicon",
    "tag_glossary",
    "tag_pairs",
    "build_pairs",
    "synth_names",
    "synth_tags",
    "desc_pairs",
    "build_pairs_sym",
    "mt",
)


def cmd_cjk_corpus(extra):
    """Run one corpus-builder stage of the CJK distillation (``scripts/distill_cjk/corpus/``).

    ``ARGS="<stage> [flags]"`` → ``python -m scripts.distill_cjk.corpus.<stage> [flags]``.
    GPU stages (``tag_glossary --mt``, ``mt``) go through the daemon instead:
    ``make daemon-run ARGS="-m scripts.distill_cjk.corpus.tag_glossary --mt"``.
    """
    if not extra or extra[0] not in CJK_CORPUS_STAGES:
        raise SystemExit(
            "usage: make exp-cjk-corpus ARGS='<stage> [flags]'  stages: "
            + ", ".join(CJK_CORPUS_STAGES)
        )
    run([PY, "-m", f"scripts.distill_cjk.corpus.{extra[0]}", *extra[1:]])


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
