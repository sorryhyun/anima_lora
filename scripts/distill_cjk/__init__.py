"""CJK vocab-pack distillation (the frozen ``project/finished/cjk_aware_anima`` line, Phase 2b).

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
* :mod:`scripts.distill_cjk.rows`      — per-row bookkeeping / holdout.
* :mod:`scripts.distill_cjk.adapter_lora` — ext-gated LoRA on the adapter's Linears.
* :mod:`scripts.distill_cjk.build_query_bank` — cross-attn probe queries for ``attn``.
* :mod:`scripts.distill_cjk.corpus`    — the pair-file builders (glossary, lexicon,
  ``build_pairs`` + synth registers); ``make exp-cjk-corpus ARGS='<stage> …'``.

The loop runs outside ``train.py``: no DiT, VAE, latents or sampler, only a 6-block
adapter forward over cached text features.
"""
