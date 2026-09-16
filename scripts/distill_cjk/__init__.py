"""CJK vocab-pack distillation (project/cjk_aware_anima, Phase 2b).

Trains the *extended* rows of the LLM Adapter's T5-side query table so that

    student:  adapter(qwen_hidden(ja), t5ext_ids(ja))
    teacher:  adapter(qwen_hidden(ja), t5_ids(en_translation))   # no grad

match at the adapter output. Both arms share the Qwen side, so the loss
isolates exactly the broken piece; the original 32,128 rows and the EN
tokenize path are never touched, so EN prompts stay bit-identical.

Two CLIs::

    python -m scripts.distill_cjk.cache    # encode pairs + teacher outputs once
    python -m scripts.distill_cjk.distill  # train / capacity / oracle

Modules:

* :mod:`scripts.distill_cjk.config`    — CLI → frozen dataclass.
* :mod:`scripts.distill_cjk.ext_table` — parameterization ladder + split embedding.
* :mod:`scripts.distill_cjk.data`      — pairs, span alignment, on-disk cache.
* :mod:`scripts.distill_cjk.losses`    — L_flat / L_span / L_attn / L_pool.
* :mod:`scripts.distill_cjk.attn_bank` — DiT cross-attn K/V probe bank (no DiT load).

Why not ``train.py``: there is no DiT, no VAE, no latents and no sampler here —
the whole loop is a 6-block adapter forward over cached text features.
"""
