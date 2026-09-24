"""Corpus builders for the CJK vocab-pack distillation (moved here from the frozen
``project/cjk_aware_anima/datasets/`` line on 2026-09-24 so a pack stays rebuildable).

Every module is a CLI (``python -m scripts.distill_cjk.corpus.<stage>``); the pair
files land in ``post_image_dataset/cjk_distill/`` and the intermediate assets
(glossaries, lexicon, wiki dump, MT caches) in
``post_image_dataset/cjk_distill/assets/`` (``tag_glossary.ASSETS``). Order for a
JA pack: ``wikidata_lexicon`` → ``tag_glossary`` (``--mt`` on the daemon) →
``tag_pairs`` → ``build_pairs`` → ``synth_names`` → ``synth_tags``; KO / ZH add
``--lang`` passes of the same stages plus ``desc_pairs`` (KO) and
``build_pairs_sym`` (symbol register). ``kanji_allow`` and ``mt`` are libraries.
Hand-pinned wordings live beside the code (``tag_overrides*.json``). Rebuild recipes for the shipped packs:
``docs/methods/cjk_vocab_pack.md``.
"""
