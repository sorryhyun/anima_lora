# CJK vocab pack

Prompt and caption in Japanese / Korean / Chinese through a text-encoder
asset, not a LoRA. The pack is a table of extra T5-side embedding rows
(`ext_embed [rows, 1024]`, ids ≥ 32128) plus a JSON sidecar with the
segmentation / row maps. English text is untouched: a prompt or caption with no
routed character tokenizes the same with or without the pack. On by
default: `configs/base.toml` ships `vocab_pack` pointing at the
shipped pack, `make download-models` installs it, and the loader fetches it
itself if the default is missing. Setting the key to `""` turns the whole path
off (stock tokenizer).

Public pack: <https://huggingface.co/sorryhyun/anima-vocab-pack-cjk>
(`anima_cjk_vocab_pack_jp_v1.{safetensors,json}`, ~285 MB; the model card carries the
training label). The builder is `_archive/bench/cjk_adapter/build_ext.py` (ext table) +
`_archive/scripts/distill_cjk/` (corpus builders under `corpus/`, cache, distill) — see
[Rebuilding a pack](#rebuilding-a-pack).

## Intended use

Japanese in a prompt or caption is **text to draw in the image**. Put the
string you want rendered in double quotes inside an English clause; the rest
of the prompt stays English tags:

```
1girl, solo, speech bubble. Text reads as "何言ってるの?".
1girl, solo, speech bubble, japanese text. Japanese text reads as "はい".
```

The `japanese text` tag (and the `Japanese` in the clause) is optional. The
shipped pack's JA rows are trained on rendering composites in this frame, so
Japanese written as tags (`猫耳`, `銀髪`) is not the intended use — write
those in English (`cat ears`, `silver hair`). Captions for LoRA training follow
the same frame; the export stage's `--combine_ocr` already writes it
(`… Japanese text reads as "…"`). KO / ZH spans route to glossary-derived tag
rows; rendering them is not trained.

## Default on, how to turn off

```bash
make download-models              # first-run set — includes the pack (→ models/vocab_packs/anima_cjk_vocab_pack_jp_v1.{safetensors,json})
make download-vocab-pack          # re-fetch just the pack
# configs/base.toml (the shipped default)
vocab_pack = "models/vocab_packs/anima_cjk_vocab_pack_jp_v1"
# off: stock tokenizer
vocab_pack = ""
make preprocess-te ARGS=--overwrite   # after any change, only if a caption carries CJK (see below)
```

If the configured prefix is the shipped default and the pair is not on disk, the
loader (`resolve_pack_prefix`) fetches the catalog row itself with one printed
line. A custom `vocab_pack` path is never auto-fetched: a missing pair there is a
`FileNotFoundError` with the download hint.

| Surface | Selection | Notes |
|---|---|---|
| `configs/base.toml` `vocab_pack` | path prefix of the pair (shipped default: the CJK pack); `""` = off | The one key every surface below defaults to. `ANIMA_VOCAB_PACK` env overrides it (like `ANIMA_DIT`). |
| `train.py` | `--vocab_pack` (config chain fills it; `--ext_pack` is an alias) | Routes inline TE caching + sample prompts, hooks the rows for sampling. Training steps read only the caches. |
| `make preprocess-te` | forwarded automatically when the key is set | Caches are encoded through the pack (T5 ids and `crossattn_emb`) and stamped with its digest. |
| `inference.py` / `make test` / `make gen` | `--vocab_pack PREFIX` overrides, `--no_vocab_pack` forces off, default = the key | Tokenizer + `llm_adapter.embed` hook, same table as the caches. |
| `GenerationRequest` | `vocab_pack=…` / `no_vocab_pack=True` | `examples/09_cjk_vocab_pack.py`; the diffusers variant is `examples/10_cjk_vocab_pack_diffusers.py`. |
| ComfyUI | `AnimaVocabPackLoader` (Adapter node ≥ 3.9; `_jp_v1` needs ≥ 3.14) | Same hook design. `AnimaMergedLoader` (≥ 3.15) loads a DiT with the pack merged in (see below). |
| Python | `anima_lora.load_vocab_pack` / `attach_vocab_pack` / `VocabPack` | Primitives in `library/anima/vocab_pack.py`; `ext_vocab.py` owns the encoder + digest. |

## What it patches

1. Tokenizer — `VocabPackTokenizeStrategy` (subclass of the stock
   `AnimaTokenizeStrategy`) re-routes the T5 id stream of any text that
   carries a routed character through `HybridT5Encoder`. The Qwen3 side (the
   actual text encoder) is untouched; the stream is still EOS-terminated and
   max-padded (the padding-as-attention-sink invariant holds).
2. Embedding table — `attach_vocab_pack` installs a hook pair on
   `llm_adapter.embed`: a pre-hook clamps ext ids to `<unk>` and remembers the
   positions, a forward hook overwrites those positions with pack rows. The
   module keeps its 32128-row state dict, so `make merge`, checkpoint saves and
   metadata are unaffected and the pack composes with any DiT or LoRA. The table
   stays on CPU; only the rows a batch uses are gathered.

Both patches are applied from one memoised `load_vocab_pack(prefix)` so the
strategy and the DiT loader see the same table.

## Cache invalidation

TE caches invalidate on **caption mtime only** (`_cache_is_current` re-encodes a
stem whose cache is older than its caption or `.variants.txt`) — no content hash
of the encoder state. Enabling, disabling or swapping a pack changes the cached T5
ids and `crossattn_emb` for every caption that carries CJK without touching the
caption, so the caches go stale.

Every cache written through a pack carries `vocab_pack` / `vocab_pack_sha`
in its safetensors metadata. At train start the cache check compares the
stamp with the active pack and warns once per mismatch kind per run (pack
→ none, none → pack, pack A → pack B) — one line, not one per file; caches with
no stamp log a single `none → pack` line. The fix is `make preprocess-te
ARGS=--overwrite`; for EN-only captions the line is informational.

EN-only datasets are unaffected either way (identical ids, identical caches).

## What works / what does not

- Works: mixed EN + CJK prompts; symbols the stock T5 cannot spell (the pack's
  symbol block, e.g. `♡`); KO / ZH spans route to glossary-derived tag rows.
- Encode rules (json `fold` / `dots`): `〜` → `～`, `―` → `ー`, `，` → `、`; a dot
  run → its own `…` row (one `・` → `.`, 2–3 dots → `…`, 4+ → `……`). `...` / `…`
  take the row only beside Japanese text, so an English prompt is unchanged. A
  reader without `dots` support (an older trainer or ComfyUI node) silently skips
  the dot rule.
- Renders: quoted Japanese text drawn into the image —
  `speech bubble, japanese text. … Japanese text reads as "はい".` Single kana /
  kanji and very short words render some of the time, seed-dependent; longer
  words and sentences mostly do not.
- JA written as tags: the kana / kanji rows are trained on quoted-text rendering,
  not tag identity — write tags in English.
- Does not: full-CJK rare-kanji character names do not compose —
  type them in latin (`hakurei reimu`). Free-form CJK sentences are a
  tokenization path, not a translation: the rows carry tag identity, not
  grammar.
- `HybridT5Encoder` also handles a quote-partitioned isotropic block (`iso`)
  when a local pack carries one; the published pack has none and routes every
  CJK span to the trained rows.
- `ext_vocab.materialize` refuses a pack that carries a line block
  (`mapping["line"]`).

## The published pack: `_jp_v1`

1 573 trained rows (kana 163, kanji 1 410) on the punct pack (`_preview51`'s
encode rules and its retrained marks `～ … ♡ ♥ 、 。「」`, carried as is); digest
`ce0ec15f7168…`. Kanji with a row cover 96.7 % of Manga109-s kanji
occurrences (`_preview51`: 92.8 %). Trained glyph list:
`anima_cjk_vocab_pack_jp_v1_trained.json` on the Hub. Recipe, all in
`project/cjk_anima_reseed/` (`_archive/plan.md` § 3):

1. **A — identity** (`retrain_kanji_b5`, `project/finished/cjk_anima_scale/`): 225 new
   kanji trained cold on the scale table, every other row frozen at
   `_preview51`'s.
2. **B — dialogue** (`sent_kanji_225`): every row free from A's, Manga109
   dialogue lines, sent_kanji_pres's recipe (λ 10 · L_pres at σ 0.8–0.9).
3. **Transplant** (`transplant.py`): B's 225 onto sent_kanji_pres's rows. On
   the 96-string ruler it reads as pres on every measure
   (`results/20261008-2025-ruler-sensitive-seed_1008/`).

Against `_preview51` (pres's read, `progress.md` § 4): glyph F1 +0.029
(p 0.0035), the page up (en_match +0.097, p 5e-8), cer n.s.; it draws text
about a third smaller. Moving caches from `_preview51`: `make preprocess-te
ARGS=--overwrite` for CJK captions.

**ComfyUI without a pack loader:** `sorryhyun/anima-jp-extended` is the base
DiT with `_jp_v1` merged in (`scripts/toolkits/merge_vocab_pack.py` →
`merge_pack_into_dit`: the rows as `llm_adapter.ext_embed.weight`, the JSON in
`ss_ext_pack_mapping`), loaded by `AnimaMergedLoader` (Adapter node ≥ 3.15.0).
A stock UNet loader loads it as plain Anima. anima_lora does not read the
merged file; it uses `vocab_pack` with the stock DiT.

## Rebuilding a pack

**Archived 2026-10-22.** The builder code is in the gitignored `_archive/` tree (`_archive/bench/cjk_adapter/`, `_archive/scripts/distill_cjk/`, outputs in `_archive/bench/cjk_distill/`) and the `exp-distill-cjk` / `exp-cjk-corpus` targets are gone; the commands below are the record of what ran, not a runnable recipe.

Four stages, all from the repo root; GPU stages go through the daemon. Every
argv below is what actually ran (daemon `job.json` records and the distill
`result.json` envelopes under `_archive/bench/cjk_distill/results/`), except where a
line says the flags were not recorded.

1. **Ext table** — `_archive/bench/cjk_adapter/build_ext.py` →
   `_archive/bench/cjk_adapter/assets/ext_embed.{safetensors,json}`. The id mapping is
   tokenizer-deterministic, so a rebuild changes row *values* only; distill
   caches (keyed on ids) survive it. Loads the Qwen3 encoder for the contextual
   char init, so: `make daemon-run ARGS="_archive/bench/cjk_adapter/build_ext.py"` (the
   v2 default, `--map procrustes-mix --char-init contextual`, symbol block on —
   69,558 rows). `--no-symbols` reproduces the 58,968-row table.
2. **Corpus** — `_archive/scripts/distill_cjk/corpus/` (`python -m
   scripts.distill_cjk.corpus.<stage> [flags]`).
   Pair files land in `post_image_dataset/cjk_distill/`, intermediates
   (glossaries, Wikidata lexicon, the danbooru wiki dump, MT caches) in
   `post_image_dataset/cjk_distill/assets/`. `tag_overrides*.json` beside the
   code are the hand-pinned wordings and win over every source.
3. **Cache** — `python -m scripts.distill_cjk.cache --pairs <jsonl> --cache_dir
   <dir> --holdout 500` per pair file (daemon; encodes both arms once).
4. **Distill** — `make daemon-run ARGS="-m scripts.distill_cjk.distill …"` →
   `output/ckpt/cjk_vocab_pack_<name>.{safetensors,json}`; publish the pair as
   is (the loader reads the json's routing maps; the Hub repo also carries the
   Qwen3 tokenizer files next to it).

### `synthja_v4` — the JA tag tier

CPU stages, in order (`--lang ja` is the default everywhere; the exact flags of
the JA glossary / pairs runs were not recorded — the defaults are the recipe):

```bash
python -m scripts.distill_cjk.corpus.wikidata_lexicon          # EN↔JA proper nouns (Wikidata, CC0)
make daemon-run ARGS="-m scripts.distill_cjk.corpus.tag_glossary --mt"   # wiki other_names → lexicon → MT residue (Hy-MT2-7B)
python -m scripts.distill_cjk.corpus.tag_pairs                 # fill-only from p1atdev/danbooru-ja-tag-pair
python -m scripts.distill_cjk.corpus.build_pairs               # pairs.jsonl: tags / tags_alt / names (+ D6 quotes)
python -m scripts.distill_cjk.corpus.synth_names --context both # + names_synth / names_synth_ja → pairs_synth.jsonl
python -m scripts.distill_cjk.corpus.synth_tags                # + tags_synth_ja (under-floor tags) → pairs_synth_tags.jsonl
```

Then cache and distill (`result.json` of `20260831-1221-2c-synthja-v4-kanjifilter`):

```bash
P=post_image_dataset/cjk_distill/pairs_synth_tags.jsonl; C=post_image_dataset/cjk_distill/cache_synth3
make daemon-run ARGS="-m scripts.distill_cjk.cache --pairs $P --cache_dir $C --holdout 500"
make daemon-run ARGS="-m scripts.distill_cjk.distill --pairs $P --cache_dir $C \
  --train_registers tags,tags_alt,names,names_synth,names_synth_ja,tags_synth_ja \
  --register_sampling names_synth_ja:0.2,names_synth:0.5 --register_span_scale names_synth:en_pinned=0.3 \
  --param global --rank 64 --loss span --trust provenance --min_visits 5 --holdout 500 \
  --steps 12000 --batch_size 32 --lr 0.001 --eval_every 250 --eval_limit 256 \
  --ext_prefix _archive/bench/cjk_adapter/assets/ext_embed \
  --out output/ckpt/cjk_vocab_pack_synthja_v4 --label 2c-synthja-v4-kanjifilter"
```

`v4` = `v3` + the allowed-kanji filter (`kanji_allow.ALLOWED`, jōyō + jinmeiyō
+ the reviewed hyōgai whitelist) applied inside `tag_glossary`; the ext table
was the v1 build (`--map ridge --char-init fragment-mean`).

### `synthjakozh1sym_r256` — the JA+KO+ZH distill

Corpus (CPU). The JA file is `pairs_tags.jsonl` (`build_pairs` + `synth_tags`,
registers `tags,tags_alt,names,tags_synth_ja`); KO and ZH are `--lang` passes
of the same stages plus the KO description register and the ZH `tags_zh_hant`
sibling (OpenCC s2t, emitted by `build_pairs --lang zh`). Flags on record:

```bash
# JA
make daemon-run ARGS="-m scripts.distill_cjk.corpus.tag_glossary --mt"
python -m scripts.distill_cjk.corpus.tag_glossary --lang ja --reselect post_image_dataset/cjk_distill/assets/tag_glossary_ja.json # CPU re-pick over stored candidates
python -m scripts.distill_cjk.corpus.build_pairs --lang ja        # pairs.jsonl, 63,241 pairs (a --commentary D2 file was passed; D2 is span-less and not a trained register)
python -m scripts.distill_cjk.corpus.synth_tags                   # tags_synth_ja
# the merge that wrote pairs_tags.jsonl (tags, tags_alt, names, tags_synth_ja — no names_synth) was not recorded
# KO
python -m scripts.distill_cjk.corpus.tag_glossary --lang ko --reselect # sources: KR KB (models/danbooru_tags_classified.csv) → wiki → MT
python -m scripts.distill_cjk.corpus.build_pairs --lang ko --alt-register # pairs_ko.jsonl: tags_ko / tags_alt_ko / names_ko
python -m scripts.distill_cjk.corpus.synth_names --lang ko --context ja --max-names 500 --floor 60 --max-per-name 40 # → pairs_synth_ko.jsonl
python -m scripts.distill_cjk.corpus.desc_pairs                   # pairs_desc_ko.jsonl (desc_ko: EN wiki sentence ↔ KO KB summary)
# ZH
python -m scripts.distill_cjk.corpus.tag_glossary --lang zh       # KB-first ranking, JA-kanji inventory guard
python -m scripts.distill_cjk.corpus.build_pairs --lang zh        # pairs_zh.jsonl (+ tags_zh_hant)
python -m scripts.distill_cjk.corpus.synth_tags --lang zh         # → pairs_synth_tags_zh.jsonl
```

Caches (daemon jobs `20260902-184212-622c9c`, `20260903-162103-*`; the four
were re-staged on the 69,558-row table after the symbol-block `build_ext`):

```bash
D=post_image_dataset/cjk_distill
for pair in tags:pairs_tags ko:pairs_synth_ko desc_ko:pairs_desc_ko zh:pairs_synth_tags_zh; do
  make daemon-run ARGS="-m scripts.distill_cjk.cache --pairs $D/${pair#*:}.jsonl --cache_dir $D/cache_${pair%%:*} --holdout 500"
done
```

Distill (daemon job `20260903-173932-6db847`, verbatim):

```bash
D=post_image_dataset/cjk_distill
make daemon-run ARGS="-m scripts.distill_cjk.distill --pairs $D/pairs_tags.jsonl \
  --cache_dir $D/cache_tags,$D/cache_ko,$D/cache_desc_ko,$D/cache_zh \
  --train_registers tags,tags_alt,names,tags_synth_ja,tags_ko,tags_alt_ko,names_ko,names_synth_ko,desc_ko,tags_zh,tags_alt_zh,names_zh,tags_zh_hant,tags_synth_zh \
  --holdout 500 --param global --rank 256 --min_visits 5 --loss span --trust provenance \
  --attn_blocks 0,13,27 --attn_queries 64 --mode train --steps 12000 --batch_size 32 --lr 0.001 \
  --eval_every 250 --eval_limit 1200 --ext_prefix _archive/bench/cjk_adapter/assets/ext_embed \
  --out output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256 --label 2c-synthjakozh1sym-r256"
```

The symbol rows ride along untrained in this pack (the pool touches 33 of
6,118 `sym` rows); the `tags_sym` register that teaches them is
`build_pairs_sym` → `cache_sym` → the same distill with `cache_sym` appended
and `tags_sym` in `--train_registers`.
The published `_preview` / `_preview2` packs are this pack plus a baked
wake-line delta (`scripts/toolkits/bake_vocab_pack.py`), not a re-distill.

## Unmask recipe

Manga pages can train with text masks off when the in-image text is OCR'd into
the caption and encoded through the pack (`masked_loss=false` + OCR captions +
`vocab_pack`). Unmasking without the captions reproduces the text spam, so the
three go together. OCR text clauses come from the caption stages (`make
caption-full`, attached by the export stage's `--combine_ocr`). Configs:
`cjk_unmask_*` under `configs/gui-methods/custom/`; evidence:
`_archive/cjk_aware_anima/reports/0901_unmask_ab.md`.
