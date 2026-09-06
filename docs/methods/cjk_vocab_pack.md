# CJK vocab pack

Prompt and caption in Japanese / Korean / Chinese through a **text-encoder
asset**, not a LoRA. The pack is a table of extra T5-side embedding rows
(`ext_embed [rows, 1024]`, ids ≥ 32128) plus a JSON sidecar with the
segmentation / row maps. English text is untouched: a prompt or caption with no
routed character tokenizes bit-identically with or without the pack, and the
whole path is inert while the key is empty (the shipped default).

Public pack: <https://huggingface.co/sorryhyun/anima-vocab-pack-cjk>
(`anima_cjk_vocab_pack.{safetensors,json}`, ~285 MB; the model card carries the
training label). Research history and the pack builder live under
`project/cjk_aware_anima/` and `bench/cjk_adapter/`; this page is the shipped
surface only.

## Enable

```bash
make download-vocab-pack          # → models/vocab_packs/anima_cjk_vocab_pack.{safetensors,json}
# configs/base.toml
vocab_pack = "models/vocab_packs/anima_cjk_vocab_pack"
make preprocess-te ARGS=--overwrite   # only if any caption carries CJK (see below)
```

| Surface | Selection | Notes |
|---|---|---|
| `configs/base.toml` `vocab_pack` | path prefix of the pair; `""` = off | The one key every surface below defaults to. `ANIMA_VOCAB_PACK` env overrides it (like `ANIMA_DIT`). |
| `train.py` | `--vocab_pack` (config chain fills it; `--ext_pack` is the pre-v2 alias) | Routes inline TE caching + sample prompts, hooks the rows for sampling, stamps `ss_ext_pack` / `ss_ext_pack_sha` on the LoRA. Training steps read only the caches. |
| `make preprocess-te` | forwarded automatically when the key is set | Caches are encoded through the pack (T5 ids **and** `crossattn_emb`) and stamped with its digest. |
| `inference.py` / `make test` / `make gen` | `--vocab_pack PREFIX` overrides, `--no_vocab_pack` forces off, default = the key | Tokenizer + `llm_adapter.embed` hook, same table as the caches. |
| `GenerationRequest` | `vocab_pack=…` / `no_vocab_pack=True` | `examples/09_cjk_vocab_pack.py`; the diffusers variant is `examples/10_cjk_vocab_pack_diffusers.py`. |
| ComfyUI | `AnimaVocabPackLoader` (Adapter node ≥ 3.9) | Same hook design; compares the LoRA's `ss_ext_pack_sha` against its loaded pack. |
| Python | `anima_lora.load_vocab_pack` / `attach_vocab_pack` / `VocabPack` | Primitives in `library/anima/vocab_pack.py`; `ext_vocab.py` owns the encoder + digest. |

## What it patches

1. **Tokenizer** — `VocabPackTokenizeStrategy` (subclass of the stock
   `AnimaTokenizeStrategy`) re-routes the **T5 id stream** of any text that
   carries a routed character through `HybridT5Encoder`. The Qwen3 side (the
   actual text encoder) is untouched; the stream is still EOS-terminated and
   max-padded (the padding-as-attention-sink invariant holds).
2. **Embedding table** — `attach_vocab_pack` installs a hook pair on
   `llm_adapter.embed`: a pre-hook clamps ext ids to `<unk>` and remembers the
   positions, a forward hook overwrites those positions with pack rows. The
   module keeps its 32128-row state dict, so `make merge`, checkpoint saves and
   metadata are unaffected and the pack composes with any DiT or LoRA. The table
   stays on CPU; only the rows a batch uses are gathered.

Both patches are applied from one memoised `load_vocab_pack(prefix)` so the
strategy and the DiT loader see the same table.

## Cache invalidation (read this)

TE caches skip on **existence only** — no content hash. Enabling, disabling or
swapping a pack changes the cached T5 ids and `crossattn_emb` for every caption
that carries CJK, but the files still exist, so the trainer would silently use
them. Two guards:

- Every cache written through a pack carries `vocab_pack` / `vocab_pack_sha`
  in its safetensors metadata. At train start the cache check compares the
  stamp with the active pack and **warns once per mismatch kind** (pack → none,
  none → pack, pack A → pack B). The fix is always `make preprocess-te
  ARGS=--overwrite`.
- A LoRA trained through a pack carries `ss_ext_pack` / `ss_ext_pack_sha`.
  Loading it with no pack, or a different one, logs a warning naming both.

EN-only datasets are unaffected either way (identical ids, identical caches).

## What works / what does not

- **Works**: danbooru-style tags in JA behave like their English spelling in
  same-seed grids (`猫耳` ≈ `cat ears`); mixed EN + CJK prompts; symbols the
  stock T5 cannot spell (the pack's symbol block, e.g. `♡`); KO / ZH tag rows
  are trained (glossary-derived) but were not grid-validated as widely as JA.
- **Does not**: full-CJK rare-kanji **character names** do not compose —
  type them in latin (`hakurei reimu`). Free-form CJK sentences are a
  tokenization path, not a translation: the rows carry tag identity, not
  grammar.
- **Not in this pack**: the quote-partitioned isotropic block (`iso`) used by
  the manga-unmask line (`project/cjk_aware_anima_dit/`) is a research build
  (`output/ckpt/*_isoq`), not published. `HybridT5Encoder` handles it when a
  local pack carries one; the shipped pack routes every CJK span to the
  trained rows.

## Unmask recipe (not shipped as a variant yet)

The reason the pack is a trainer path: manga pages train with text masks
**off** when the in-image text is OCR'd into the caption and encoded through
the pack (`masked_loss=false` + OCR captions + `vocab_pack`). Unmasking
without the captions reproduces the text spam, so it is a bundle, not a
toggle. The recipe and its evidence live in
`project/cjk_aware_anima/reports/0901_unmask_ab.md` and the `cjk_unmask_*`
configs under `configs/gui-methods/custom/`; the OCR caption stage is still
research-side (`docs/v2_release_plan.md` B5/B6).
