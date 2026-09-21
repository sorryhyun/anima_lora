# deploy_plan — the form of the shipped weights

Index: [`README.md`](README.md). What gets shipped and when is
[`plan.md`](plan.md) § 3 (`preview2`, v2.0.0.beta2). This file is only the file
form. The 2026-09-17 version (Hub folder layout, `comfy/` / `diffusers/`
exports, gates, migration) is archived as
`_archive/cjk_renderable_anima/deploy_plan_2026_09_17.md`.

## A vocab pack pair

The render line ships as an ordinary **vocab pack**: one `.safetensors` + one
`.json` with the same stem, in the same format as `anima_cjk_vocab_pack`.

- **`.safetensors`** — the pack's ext-row table (69 558 rows × 1 024) with the
  trained delta summed into the rows it trained; every other row is the base
  pack's, byte for byte. The header carries the usual `anima_*` stamps plus an
  `anima_render` summary.
- **`.json`** — the routing map (`qwen` / `char` / `sym` / `route` → rows),
  unchanged from the base pack, so tokenization and EN prompts are untouched.
  It gains a `render` block (source arm, ext ids, row → piece text, scale, base
  pack digest, git rev) and marks the summed rows with provenance tier
  `render`. Every loader refuses a table without its json.

No tokenizer files, no DiT weights, no LoRA: the DiT's `llm_adapter.embed`
stays at 32 128 rows and the pack is looked up through the existing embed hook.

## Bake

```
ext_embed[e − 32128] += delta.raw[i] · delta.row_scale      for every e in delta.ext_ids
```

This is `ExtDelta`'s forward hook at scale 1, applied once to the stored table,
so the baked pack renders what the evaluated arm rendered
(`tests/test_bake_vocab_pack.py`).

```bash
.venv/bin/python scripts/toolkits/bake_vocab_pack.py \
    output/wake_probe/<arm dir> --out models/vocab_packs/<pack name>
```

Each pack gets its own directory holding exactly one pair (`vocab_pack` accepts
such a directory); `--comfy_dir` also symlinks the pair into a ComfyUI
`vocab_packs/` folder.

## What loads it

The one `vocab_pack` key — training, TE caching, `inference.py --vocab_pack`,
`GenerationRequest` — and the ComfyUI `AnimaVocabPackLoader` node (≥ 3.9.1),
all without a code change. A baked pack has a new `pack_digest`: TE caches of
CJK captions need `make preprocess-te ARGS=--overwrite`, and LoRAs stamped with
another pack warn on load.

## Shipped and next

- **Shipped:** `anima_cjk_vocab_pack_preview` at the root of
  `sorryhyun/anima-vocab-pack-cjk` — baked from `sent_s24k_a1_s05` (503 rows,
  sha `5f52aefce82a…`); the trainer's default since it shipped.
- **Next:** `anima_cjk_vocab_pack_preview2` — the merged JA table after the
  sentence run (`plan.md` § 3), the default from v2.0.0.beta2.
