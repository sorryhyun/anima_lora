# deploy_plan — `sorryhyun/anima-vocab-pack-cjk`, v2 layout (2026-09-14)

> Forward plan for publishing the render line's table. Nothing here is
> uploaded yet. Training phases and their gates stay in [`plan.md`](plan.md);
> this file is only packaging, surfaces, gates before upload, and the Hub
> migration. Decisions the user still owes are collected at the end (D1–D6).

## Target Hub layout

The repo stays `sorryhyun/anima-vocab-pack-cjk`. The current release moves
to `old/`; three new folders serve three kinds of user.

```
README.md                      model card (rewritten; license block fixed, see License)
LICENSE.md                     CircleStone Labs Non-Commercial License v1.2, verbatim
NOTICE                         attribution notice + "modified by" statement (§3b, §3d-i)
modular_model_index.json       only under D2a — diffusers entry point, components → diffusers/…

old/                           the 2026-09-06 release, byte-identical (server-side copy)
  anima_cjk_vocab_pack.safetensors / .json
  tokenizer_qwen3/

delta/                         framework-neutral pack pair — same format as today
  anima_cjk_render_pack_v1.safetensors / .json
  tokenizer_qwen3/

comfy/                         single file for ComfyUI models/diffusion_models/
  anima-base-v1.0-cjk-render-v1.safetensors

diffusers/                     circlestone-shaped modular pipeline, only our parts uploaded
  modular_model_index.json     (D2b / D2c; under D2a this lives at the root)
  text_conditioner/config.json, diffusion_pytorch_model.safetensors
  text_encoder_block/modular_config.json, pack_text_encoder.py, ext_vocab.py, pack.json
```

## The constraint every folder inherits

In every Anima stack the T5-side id stream is produced on the **text side**
(ComfyUI's CLIP tokenizer, diffusers' `text_encoder` block, anima_lora's
tokenize strategy), and the DiT only looks the ids up. The pack touches both
halves: routed spans get ext ids (tokenizer), and those ids get rows
(`llm_adapter.embed` / `text_conditioner.embed`). A merged weight file can
carry the rows but cannot change tokenization, so **no folder is code-free**.
What "merged" buys is one download and one input instead of a pair plus a
wiring step. The model card must say this plainly, or `comfy/` reads as a
drop-in that silently does nothing for CJK.

## What each folder is

### `delta/` — the pack pair (assumption D6)

The base 69,558-row table with the render rows summed in, plus the json.
Same file format, same stem rule and same two patch points as the current
release, so it works unchanged with node ≥ 3.9.1 `AnimaVocabPackLoader`,
anima_lora's `vocab_pack` key, and `examples/09` / `examples/10`. Against
`old/` only the baked rows differ (~410 of 69,558). The json gains a
`render` block (ext ids, `row_text`, source arm, git rev, gate numbers),
provenance tier `render` on those rows, and a new label; `pack_digest`
changes, which is what makes stale TE caches and LoRA stamps warn.

Optional extra: `delta/render_rows_v1.safetensors` (`[N, 1024]` + ext ids,
~2 MB) for someone layering the render rows onto their own pack. Cheap; ship
only if asked.

### `comfy/` — one file in `models/diffusion_models/`

Contents: the `anima-base-v1.0` state dict **unchanged**, including
`llm_adapter.embed.weight` at 32 128 rows, plus one extra tensor
`vocab_pack.ext_embed [69558, 1024]` and the pack json in the safetensors
header (`anima_vocab_pack_json`, plus the usual `anima_*` stamps).

Why the embed is not widened: ComfyUI core hardcodes
`operations.Embedding(32128, …)` (`comfy/ldm/anima/model.py:159`). A widened
tensor is a size-mismatch error even under `strict=False`, so a widened file
would not load in stock ComfyUI at all.

Behaviour:

- **Stock `UNETLoader`**: loads it (`load_state_dict(strict=False)` logs one
  `unet unexpected: vocab_pack.ext_embed`) and behaves exactly as stock
  base-v1.0; CJK degrades to `<unk>` as before.
- **Node**: a new `AnimaVocabPackCheckpointLoader(unet_name, clip) → (MODEL,
  CLIP)` in ComfyUI-Anima_lora-Adapter reads table + json from the same
  file and runs the existing loader's `apply()` on them (tokenizer wrap +
  embed hook). It needs its own file read, because the stock loader drops
  unexpected keys before a downstream node could see them. Node release
  3.11.

One file per base DiT, ~4.3 GB each (base is bf16; the ext table in bf16
adds ~142 MB). **v1 ships base-v1.0 only**, the DiT every render row was
trained against. Aesthetic and turbo files exist only if G4 passes (D5).

### `diffusers/` — loads like `circlestone-labs/Anima-Base-v1.0-Diffusers`

A modular pipeline whose unchanged components **point at circlestone's
repo** (`transformer`, `vae`, `text_encoder`, `tokenizer`, `t5_tokenizer`,
`scheduler` via `pretrained_model_name_or_path` in the index), so ~5.6 GB is
neither re-uploaded nor redistributed. Ours:

- `text_conditioner/`: circlestone's conditioner with `embed` widened to
  101 686 rows and `target_vocab_size: 101686` in `config.json`.
  `AnimaTextConditioner` builds `nn.Embedding(target_vocab_size, …)`
  (diffusers 0.39.0), so it loads natively.
- `text_encoder_block/`: example 10's `PackTextEncoderStep` as remote code
  (`auto_map` in `modular_config.json`), `ext_vocab.py` beside it, the pack
  json. Stock ids for prompts with no routed character, as today.

**Loading constraint found (diffusers 0.39.0):**
`ModularPipeline.from_pretrained` reads `modular_model_index.json` from the
repo **root** only; its config-loading kwargs carry no `subfolder`. So
`from_pretrained("sorryhyun/anima-vocab-pack-cjk", subfolder="diffusers")`
does not exist. Options (D2):

- **D2a (recommended)**: put `modular_model_index.json` at the root with
  components pointing into `diffusers/…` subfolders. Users then call
  `ModularPipeline.from_pretrained("sorryhyun/anima-vocab-pack-cjk",
  trust_remote_code=True)`, which is circlestone's call with our repo id.
  Cost: one small json beside the four folders.
- **D2b**: no root file; the card documents
  `snapshot_download(allow_patterns="diffusers/*")` and loading the local
  directory.
- **D2c**: a separate `sorryhyun/anima-vocab-pack-cjk-diffusers` repo.

Unverified until smoke S-diffusers: remote-code blocks resolved from a
**subfolder** (`ModularPipelineBlocks.from_pretrained` forwards `subfolder` to
`get_class_from_dynamic_module`, but no end-to-end run exists). If that
fails, the block files move to the root under D2a.

## What gets baked

Source: the `trained.pt` of the table that passes its gate. v1 candidate is
P0b (`output/wake_probe/encoder_wdsek_w120_s24k_p0b/trained.pt`), unless D1
says to wait for P2.

Formula, for every ext id `e` in `delta.ext_ids` (all of them, exactly as
evaluated, so G0 can be exact):

```
ext_embed[e − 32128] += delta.raw[i] · delta.row_scale
```

This is `ExtDelta`'s forward hook at `scale = 1`, so the eval renders are
what ships. Held-out and eval-only rows carry `g` only; they are baked too,
so the shipped table equals the evaluated one.

Card claims for a P0b-sourced v1, and nothing beyond them:

- Renders **one unit** per quoted string: a kana (basic, voiced, small), one
  of the 100 kanji, or one of the 112 trained words. Multi-piece strings
  render one piece until P1/P2 (Run 3 `line` 0/32).
- Through the trained clause shape `Japanese text reads as "…"`.
- On base-v1.0 (others per G4).
- Every existing pack behaviour (tags, EN bit-exact) per G1/G2.

## The risk to measure first: render rows are tag rows

The shipped pack routes every CJK span, quoted or not, to the same rows.
The rows the render delta moves are the ones JA tag prompts address: 人 女 男
子 目 気 among the kanji, and every kana inside words. `女の子, 青い目` now
reads rows trained to draw glyphs, which risks text leaking into tag-prompt
renders or tag meaning drifting. **No wake-line eval has measured a tag
prompt.**

G2 measures it. If it fails, the fix is the D1 quote partition: `iso` block
plus `route.quotes`, so quoted content uses mirror rows at an offset. It is
already in `HybridT5Encoder` and node 3.10.0. But the render addresses move,
so the rows retrain on the mirror block. That decision is cheapest **before
P1**; run G2 on the P0b table as soon as it exists. (isoq's s20 loss was
tag-caption quality through the mirror block, which is a different use; it
does not decide this.)

## Gates before any upload

| gate | what | pass |
|---|---|---|
| G0 bake | baked pack (no hook) vs hook eval, same seeds, groups `single single_ext single_kanji word en` | fp32: identical latents; bf16: max \|Δpixel\| ≤ 1/255 |
| G1 EN | ids with/without pack on `EN_WORDS` + `order_probe.py` words; EN renders | ids identical, renders bit-exact |
| G2 tag path | the pack's JA / KO / ZH tag grid, same seeds, `old/` vs baked | (a) detector text-box count on tag renders not above `old/`, both readers; (b) PE-cos baked-vs-old inside the seed-twin floor |
| G3 native | `native` stage, 8 kana, on the baked pack | clause text lands in the scene; clause-free prompts unchanged |
| G4 cross-base | `eval` with `ANIMA_DIT` = aesthetic-v1.1, turbo-v1.1 (turbo: 4 steps, cfg 1), groups `single single_kanji en` | ≥ 0.8× base-v1.0's counts → that base may be claimed and get a `comfy/` file |
| G5 surfaces | S-comfy, S-diffusers, S-delta below | all three |
| G6 license | LICENSE.md + NOTICE at root, card license block | present |

Surface smokes (G5). Each renders the same prompt and seed as the
anima_lora render where the surface allows it.

- **S-comfy**: stock `UNETLoader` loads the file with exactly one
  unexpected-key warning and its EN render equals base-v1.0's; the new node
  renders `Japanese text reads as "何"` as 何.
- **S-diffusers**: the D2 entry point loads with `trust_remote_code=True`;
  its image matches example 10 run on `delta/` (same seed, sampler).
- **S-delta**: node ≥ 3.9.1 `AnimaVocabPackLoader` on the pair; `examples/09`
  and `examples/10 --dry_run` on the `delta/` stem; anima_lora with
  `vocab_pack` pointing at it.

## License (blocking; applies to every folder)

All four folders read as **Derivatives** under the CircleStone Labs
Non-Commercial License v1.2 §1a, which names "textual inversions based on a
CircleStone Model"; a text-embedding table trained through the frozen
model is that. `comfy/` also Distributes the base weights themselves. §3
then requires:

- a copy of the license for recipients (§3a);
- the Attribution Notice verbatim (§3b) plus a statement that we modified
  the model (§3d-i);
- no terms that conflict with the license, and its disclaimer at least as
  protective (§3d-ii);
- non-commercial use only, which carries over to Derivatives (§2a).

The current card says `license: apache-2.0`, which conflicts. The new card
uses `license: other`, `license_name: circlestone-labs-non-commercial-license`,
`license_link: LICENSE.md`, with LICENSE.md and NOTICE at the root. The same
question applies to `sorryhyun/anima-turbo-4step` (also tagged apache-2.0);
it is out of scope here and noted only.

## Migration: current release → `old/`

Hub mechanics: one `create_commit` with `CommitOperationCopy` (server-side
LFS copy, no re-upload) root → `old/`, and later `CommitOperationDelete`
on the roots.

What reads the root paths today:

- `library/downloads.py` (`VOCAB_PACK_REPO` + `VOCAB_PACK_STEM`), used by
  `make download-models`, `make download-vocab-pack` and the loader
  auto-fetch in `resolve_pack_prefix`;
- `examples/09`, `examples/10`;
- the current card and the ComfyUI node README;
- `docs/methods/cjk_vocab_pack.md` and the guidebook in four languages.

`vocab_pack` is on by default since v2, so every checkout older than the
trainer release fails its first-run download the moment the root files are
deleted. The users are Arca Live (KR) plus CN/JP.

Order:

1. **M1: Hub commit A.** Copy root → `old/`, keep the roots. Nothing breaks.
2. **M2: trainer release.** The catalog row points at the subfolder path
   with a `revision` pin. Update examples 09/10 and the docs (translator
   agent for the guidebook). D3 decides the default pack. Switching to
   `delta/` changes the digest, which means `make preprocess-te
   ARGS=--overwrite` for CJK captions and warnings on LoRAs stamped with the
   old pack; both go in the release notes.
3. **M3: Hub commit B.** Upload `delta/`, `comfy/`, `diffusers/` (+ root
   json under D2a), LICENSE.md, NOTICE and the new card. Release node 3.11
   with the checkpoint loader and README.
4. **M4: Hub commit C.** Delete the root files after the D4 window; the card
   keeps a "moved to `old/`" line.

Collapsing M1 and M4 into one commit is possible if the break is accepted;
the release notes then carry the manual-download line.

## Build tooling (not written yet)

| file | does |
|---|---|
| `scripts/toolkits/bake_vocab_pack.py` | `trained.pt` + base pack → pack pair; stamps `anima_*` metadata (replaces the ad-hoc `save_file(..., metadata=)` stamping of 2026-09-06) |
| `scripts/toolkits/export_pack_comfy.py` | base DiT + pack → `comfy/` file (32 128-row embed kept, `vocab_pack.ext_embed` + header json) |
| `scripts/toolkits/export_pack_diffusers.py` | widened `text_conditioner/` + modular index with circlestone pointers + block dir |

The block dir's `ext_vocab.py` is generated from `library/anima/ext_vocab.py`
by the vendor-sync step, not copied by hand. Add it as a `make vendor-sync`
target.

Invariant tests, each on a tiny synthetic table:

- bake equals hook;
- EN ids are bit-exact;
- the comfy export keeps the 32 128-row embed and round-trips the header json;
- the widened conditioner loads through `from_pretrained`.

Uploads use `hf upload` for `delta/` and `diffusers/`, and
`upload-large-folder` or `create_commit` for the 4.3 GB `comfy/` file.

| folder | size |
|---|---|
| `old/` | ~297 MB (pack 285 MB + tokenizer 11.5 MB) |
| `delta/` | ~297 MB |
| `comfy/` | ~4.3 GB per base |
| `diffusers/` | ~270 MB conditioner + ext rows in its dtype (~140–285 MB) |

## Decisions owed

- **D1**: first table to ship. The P0b singles table, with the single-unit
  claim, or wait for P2 strings.
- **D2**: diffusers entry point. Root json (a, recommended), subfolder
  snapshot (b), or a separate repo (c).
- **D3**: the trainer's default pack after M2. Stay on `old/` until G2 passes,
  or switch to `delta/`.
- **D4**: how long root files stay before M4.
- **D5**: `comfy/` bases. base-v1.0 only, or also aesthetic / turbo after G4.
- **D6**: what `delta/` means. Assumed: the framework-neutral pack pair, like
  today. Alternative: only the diffusers `text_conditioner` component, swapped
  in with `pipe.update_components`.
