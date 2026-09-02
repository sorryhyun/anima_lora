# temp_plan — JA glossary r2 → re-distill (2026-09-02, scratch)

Temporary working plan; fold into `plan.md` / `findings.md` once the chain
lands, then delete this file.

## Why the chain was stopped

The r1 override round (351 fixes, occ>100) was applied and the JA re-distill
chain (cache → synthja_v5 → synthjako3) was queued. Two things happened:

1. The first cache attempt died at 236/518 shards on **disk full** (data
   volume 100%). `cache_synth3` (193G, the v4/jako2 JA cache) was deleted by
   user call to make room; the volume is now at 50%.
2. The r2 review (occ 10–100 tail, 1,704 rows) came back with **~900 more
   proposals** covering ~33k occurrences — far more than expected. A cache
   built now would bake the pre-r2 surfaces into ~190G of teacher/student
   embeddings and need rebuilding again. So the chain was killed
   (`8e17ba` cache / `725c89` v5 / `87c595` jako3 → stopped) and the partial
   `cache_synth4` removed. **Nothing is cached right now; v4 / jako2 packs
   still serve.**

## Steps

1. [x] **r2 review lands** (2026-09-02 11:00; applied 11:18 with `frieren → フリーレン` / `sousou no frieren → 葬送のフリーレン` instead of the `(キャラ)` suffix, plus 52 extra fixes in `datasets/tag_overrides_ja.proposed_r2b.json` — few-shot exemplar leaks, 腐 wiki picks, `fate/extra`, `genderswap (mtf)`, emoticon identities) → `assets/tag_glossary_review_ja_r2.md` +
   `datasets/tag_overrides_ja.proposed_r2.json` (~900 entries). User skims
   the md (sections 1–4 = confident changes, 5 = uncertain rows the agent
   left out). Known judgement calls to look at: `frieren → フリーレン(キャラ)`
   (collision with the copyright title), `fangs → 八重歯`, `impregnation →
   種付け` (correct: the act, 妊娠 stays for `pregnant`).
2. [x] **Apply r2** (overrides 371 → 1324; reselect 11:35 after the guard landed — see "Reselect guards" below): merge the picked entries into `datasets/tag_overrides.json`
   (r1 pattern: `dict(ov).update(prop)`), then
   `python project/cjk_aware_anima/datasets/tag_glossary.py --lang ja
   --reselect project/cjk_aware_anima/datasets/assets/tag_glossary_ja.json`
   (CPU, seconds; back the pre-r2 glossary up as `tag_glossary_ja.pre_r2.json`
   first — reselect reads the prior build's stored candidates). Verify every
   proposal took, and check the tail for the systematic MT error classes the
   r2 report names — a rule-based pass below occ 10 may be worth it before
   caching, since each cache is ~65 GPU-min + ~190G.
3. [x] **Rebuild corpora** (11:36 — 63,241 base / 238,113 synth / 240,131 synth_tags pairs; 6,544 ext rows visited) (CPU, ~3 min total, no backups needed):
   `build_pairs.py --lang ja --commentary
   project/cjk_aware_anima/datasets/assets/commentary_pairs_7k.jsonl`
   (the default commentary path is stale — without the flag D2 is silently
   skipped and the corpus drops from 63k to 54k pairs) →
   `synth_names.py` → `synth_tags.py` (defaults; writes `pairs_synth.jsonl`,
   `pairs_synth_tags.jsonl` ≈ 240k pairs).
4. [x] **Re-queue the chain** (11:37 — jobs `86ed07` cache / `01cd6c` v5 / `2c4662` jako3, labels `*-glossary-r2`; 280G free at submit) (daemon, FIFO; ~2.5 h GPU; needs ≥200G free on
   the data volume — check `df` first):
   - cache: `make daemon-run ARGS="--label 2c-cache-synth4 --queue -m
     scripts.distill_cjk.cache --pairs post_image_dataset/cjk_distill/pairs_synth_tags.jsonl
     --cache_dir post_image_dataset/cjk_distill/cache_synth4 --holdout 500"`
   - synthja_v5: the v4 argv (`2c-synthja-v4-kanjifilter`, job
     20260831-111615-69960b) with `cache_synth4` and
     `--out output/ckpt/cjk_vocab_pack_synthja_v5 --label 2c-synthja-v5-glossary-r2`
   - synthjako3: the jako2 argv (job 20260901-095430-8fd30a) with
     `cache_synth4`, `--init_pack output/ckpt/cjk_vocab_pack_synthja_v5`,
     `--out output/ckpt/cjk_vocab_pack_synthjako3`; KO caches (`cache_ko`,
     `cache_desc_ko`) are reused unchanged.
   The exact three commands were used verbatim on 2026-09-02 10:34 (jobs
   `8e17ba` / `725c89` / `87c595`); copy them from those job records.
5. [ ] **Gates** before calling v5 the new pack: distill readouts vs v4
   (holdout recovery_attn per register, F3-style span loss), the 2c render
   grid at the same seeds (`bench/cjk_adapter/run_bench.py --ext --ext_prefix
   output/ckpt/cjk_vocab_pack_synthja_v5 --prompts …`, ≥3 seeds per K3),
   EN bit-exact test.
6. [ ] **Then** the morpheme-row minting on top of v5 (user picks in
   `assets/ja_mint_morphemes_pick.txt`: 女の子 赤面 ヘア ノー ビキニ ハート
   セックス ブラ). Needs a JA boundary design in
   `HybridT5Encoder._encode_cjk_words` — at minimum a block list so ブラ does
   not fire inside ブラウス / ブラック / ブラウン (~12% of ブラ matches), and
   a clause-start anchor for the tag register. Mint corpus via
   `mint_corpus.py` + `--span_focus_from` (plan_ko3 F2: batch loss leaves
   minted rows flat).

## Reselect guards (landed 2026-09-02, `tag_glossary.py`, uncommitted)

The r2 tail showed a systematic MT contamination class: `mt.py`'s 16
`TAG_FEWSHOT` exemplars leak into translations of tags the model finds
meaningless — whole-list echoes (`:t` → `:女の子1人の全身写真、シンプルな背景…`)
and word substitutions (`pantyhose only` → `ニーハイのみ`, `holding shorts` →
`ショートヘアを持つ`). Fixed at the selection layer, veto-only (never requires
Japaneseness — emoticon rule):

- `contaminated()` rejects candidates with U+FFFD / 2+ fewshot words (any
  source), `、。` (MT only — real titles carry 、), an unlicensed exemplar
  word (`EXEMPLAR_LICENCE`, MT only; 全身鏡 stays), or a fujoshi register
  marker (`REGISTER_LICENCE`: 腐/ホモ/やおい/BL unless the tag is about it).
  Dropped strings recorded in `rejected_contaminated`; raw MT stays in `mt_ja`.
- `is_symbol_tag()` → `passthrough` for emoticon/symbol tags (`:d`, `\m/`,
  `!`), pinned like artist handles. Applies to KO too on its next reselect.
- `tag_counts()` now goes through `parse_caption` — the old `split(",")`
  minted 30 clause-fragment tags (`weight. On the left`); all gone after
  reselect, period-named titles (`c.c.`, `takt op.`) keep their dot.
- Tests in `tests/test_cjk_glossary.py` (84 pass). Isolated effect on the
  pre-r2 glossary: 103 flips, unresolved 705 → 742 (+37 tags / ~100 occ).

Follow-up NOT done: `build_pairs.split_caption` still hand-splits with
`split(",")` — clause fragments now fall to latin passthrough in composed
pairs instead of junk JA, but the composer should use the grammar too.

## Parked alongside

- Unmask arm C2 (PP-OCR captions, `cjk_unmask_c2.safetensors`) rendered ≈ C;
  needs the render-level scorer (text-mask area + Tagger recall) to rank —
  96 renders in `output/tests/cjk_unmask_eval2/` are the first input.
- `nimi nightmare` (occ 9) dropped to unresolved in the r1 reselect; one
  override line restores it if wanted.
