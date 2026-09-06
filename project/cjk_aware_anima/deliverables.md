# CJK-aware Anima — deliverables

What exists, where it lives, and what state it is in. Verdicts are in
[`findings.md`](findings.md); remaining work in [`plan.md`](plan.md). Nothing
here duplicates measured tables — those stay with the reports and
`bench/*/results/`.

## Code

| Piece | Where | State |
|---|---|---|
| Ext table build (Qwen CJK tokens → T5 space, anchor init, byte-permutation tie-break) | `bench/cjk_adapter/build_ext.py`, `ext_vocab.build_ext_table` → `bench/cjk_adapter/assets/ext_embed.{safetensors,json}` | done; rebuild only after an init change |
| Hybrid encoder (CJK spans → ext ids; bit-identical on pure EN) | `library/anima/ext_vocab.py::HybridT5Encoder` (canonical since 2026-09-01; `bench/cjk_adapter/ext_vocab.py` is a re-export shim) | promoted; also on the Adapter node's `_vendor` surface |
| Probe / acceptance harness (arm sweep, `--ext`, `--prompts` render grid, `--adapter_lora`) | `bench/cjk_adapter/run_bench.py` | done |
| Residual name probe (adapter-space floor gate) | `bench/cjk_adapter/residual_probe.py` | done; diagnostic only |
| Distill loop: corpus cache (process-pool stager, 67 pairs/s), ext-table ladder, objectives (`span`/`attn`/`flat`), register sampling / span scaling, warm start | `scripts/distill_cjk/{cache,config,data,distill,ext_table,losses,attn_bank}.py` (`make exp-cjk-cache` / `exp-distill-cjk`) | done |
| Real-query bank for the attention readout | `scripts/distill_cjk/build_query_bank.py` → `bench/cjk_distill/assets/query_bank.safetensors` | done |
| Ext-gated adapter LoRA (`g = any(id ≥ 32128)`, forward-hook delta, standard `lora_A/lora_B` keys) | `scripts/distill_cjk/adapter_lora.py`, `--adapter_lora r=…` | done, inert without the flag; **does not ship** (plan3 closed) |
| One-off gate drivers (G2, G3/G4, G5, coverage, separability) | [`gates/`](gates/) (`make exp-cjk-gates`) | done |
| Tests | `tests/test_cjk_distill.py` (G1 EN bit-exactness, 23 cases), `tests/test_cjk_glossary.py` (24 invariants), `tests/test_cjk_adapter_lora.py` | green |
| `process_escape` mojibake fix | `c8cf3ce2` | shipped |

## Data builders (`datasets/` — measured numbers in [`datasets/README.md`](datasets/README.md))

| Builder | Output | Notes |
|---|---|---|
| `wikidata_lexicon.py` | EN↔JA/KO/ZH proper-noun lexicon (CC0) | ≥2-token + `P31/P279* Q95074` guard; 0/89 artists |
| `tag_glossary.py` (`--mt` GPU; `--reselect` CPU) | `assets/tag_glossary_ja.json` + `tag_glossary_review.md` (400 sourced rows) | 14,678/14,753 tags, 99.86% coverage; `.pre_item2.json` snapshot since pruned |
| `tag_overrides.json` | hand-pinned wordings, beat every source on rebuild | review signed off 2026-08-30 — no new overrides, glossary/pairs/`cache_synth2` unchanged |
| `tag_pairs.py` | fill-only from `p1atdev/danbooru-ja-tag-pair-20241015`; arbiter candidates | `via: tagpair` 0.6 / `tagpair_verified` 1.0 |
| `build_pairs.py` (multi-root, `--commentary`, `names` register) | `post_image_dataset/cjk_distill/{pairs,pairs_synth}.jsonl` + `coverage.json` + `spotcheck.md` | 45,230 base pairs; dedup by artist-relative path |
| `synth_names.py --context ja\|en\|both` | `names_synth{,_ja}` registers (wiki ∩ tag-pair canonical names, rarity-weighted to a visit floor) | 261k pairs with `both` |
| `commentary.py` (+`--mt`, Hy-MT2-7B) | `assets/commentary_ja.jsonl` — 73,015 native-JA records, 9,068 paired | MT resumable at 5,721/69,668; inert under span |
| `mt.py` | `MTEngine` (Hy-MT2), prompt-keyed resume cache, `--probe` idiom bench | |
| `lovehina.py` | native-register JA eval set (D7) | held out, never trained |
| `manga_text.py` | danbooru text-detection pilot (986 regions, logprob gate) | glyph-line input only |

Caches on disk: `post_image_dataset/cjk_distill/cache_synth2` (~155–170 G,
261k pairs, JA-context names) is the kept training cache; `cache/` and
`cache_synth/` were deleted 2026-08-28 and rebuild from the kept jsonl
(~15 min / ~1 h on the daemon).

## Trained packs (`output/ckpt/cjk_vocab_pack_*.{safetensors,json}`)

| pack | recipe | role |
|---|---|---|
| **`synthjakozh1sym_r256`** | cold joint JA+KO+ZH on the r256 recipe, 69,558-row table with the symbol block + `route` rule (findings §10–§11) | **the shipped test pack** — published 2026-09-06 as `anima_cjk_vocab_pack.{safetensors,json}` at https://huggingface.co/sorryhyun/anima-vocab-pack-cjk (metadata-stamped, + Qwen3 tokenizer files; the node needs ≥ 3.9.1 for symbol routing). KO/ZH rows trained but never render-grid validated — the README says so |
| `synthja_v4` | v3 + the 2b allowed-kanji filter (jōyō+jinmeiyō whitelist; report 0831_kanji_filter) | the first test release (2026-09-01, `anima_ja_vocab_pack.*`; the `-ja` repo was renamed to `-cjk` and the JA files were removed from the Hub 2026-09-06 — local copy is the record) |
| `synthja_v3` | v2 corpus + §5a `tags_synth_ja` (2,249 under-floor tag pairs; report 0831 §6) | superseded by `synthja_v4` — keeps every v2 gain, adds c1/c2/c3/t2/t6; t3 armor still open |
| `synthja_v2` | first pack on the rebuilt corpus (name-axis fix + `, ` joiner) | superseded by `synthja_v3` same day |
| `synthja` | `param=global`, `loss=span`, registers `tags,tags_alt,names,names_synth,names_synth_ja`, 12k steps, `、`-joined corpus | superseded by `synthja_v2`; keep as the pre-rebuild reference |
| `synthja_lora16{,_reg}` (+ `.adapter_lora.safetensors`) | plan3 arms | kept as reference for a future DiT-side-target line; nothing ships |
| `synthja_attn`, `synth`, `synth_bal`, `names`, `item2`, `tagpair`, `wide`, `global{,_row}` | the measured ladder | superseded; safe to delete once `findings.md` is trusted |

Envelopes: `bench/cjk_distill/results/<stamp>-<label>/`; render grids:
`bench/cjk_adapter/results/<stamp>-<label>-grid/` (labels match the pack).

## Eval assets (`assets/`)

`ja_eval_prompts.json` (20 prompts, five registers — the binding grid),
`ja_eval_prompts_names_mixed.json`, `ja_eval_prompts_residual.json`
(name-stripped twins), `grid_labels.json` (eyeball labels, 5 arms × 7–9 prompts),
`separability_phase02{,_fixed}.json`.

## Ship contract

The v1 artifact is a **vocab pack**, not a LoRA: `cjk_vocab_pack_synthja_v3.safetensors`
(rows + global correction baked) + `.json` (segmentation rules, char/token →
row-id map, per-row provenance `trained` / `zero-shot`, training metadata).
Rare kanji character names are **out of scope for v1** (users type
`hakurei reimu` latin — the mixed register works); zh/ko rows are physically
present but untrained (never-visited rows stay at zero-shot init and are
flagged; demote to `<unk>` in the JSON if they poison neighbours).

Surfaces (status notes inline in [`plan.md`](plan.md) Phase 3 — as of
2026-09-01 the ComfyUI node + HF test release exist; the in-repo shim does not):

- **In-repo**: strategy shim routes CJK spans through `HybridT5Encoder` when
  the sidecar is present (flag to disable); `load_dit_model` appends rows to
  `llm_adapter.embed`. Composes with every checkpoint / DiT LoRA (disjoint
  parameters). TE caching uses the same strategy, so JA captions cache with
  ext ids; **JA TE caches must be regenerated** after the shim lands (EN caches
  untouched by construction).
- **ComfyUI**: one node `(MODEL, CLIP, vocab_pack) → (MODEL, CLIP)` in
  `ComfyUI-Anima_lora-Adapter` — wraps the CLIP's t5xxl tokenize path, object-
  patches the adapter embed (forward-hook-not-override). Endgame: upstream.
- Release asset pattern: the CNS γ npz (must be attached to the release tag).

## Hard blocker on ship — CLEARED 2026-08-30

**Human sign-off on `assets/tag_glossary_review.md` → `datasets/tag_overrides.json`** — done; the 22 pinned overrides stood, no re-cache needed (rows unchanged, stager is pair-keyed). Rationale kept for the record:
36.6% of span tokens are `mt_unverified`; a wrong wording trains that tag's
rows toward the wrong meaning and no aggregate metric can see it (G4b). Two
review axes: the polysemy class (`bow`) and katakana-vs-kanji (`armor`→鎧 vs
`bed`→床). Known junk already spotted: `touhou → おは東方`. The fix path is a
CPU `--reselect`, then one ~20 GPU-min retrain of `synthja` and a grid.
