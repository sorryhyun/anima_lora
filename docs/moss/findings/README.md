# MOSS-Image findings

Measurements of MOSS-Image (`library/moss/`) that are not a shipped feature's own doc. Each doc
separates what was measured (with its run) from untested hypotheses. Run logs, probe code
and raw numbers live in the matching `bench/moss_*/` or `_archive/bench/moss_*/` folder.

## Prompt and text encoder

| Doc | What it covers |
|---|---|
| [chat_template.md](chat_template.md) | §1: the 10 chat-template tokens are MOSS's attention sink, an OFF-gate plus a per-head bias written into every image token; trained LoRAs barely move it. §2: an assistant-turn tail after `<think>\n` (best `\n</think>\n\nSure,`) sharpens hands and backgrounds without a LoRA; the image reads `</think>`, a caption-dependent key. A LoRA trained and rendered on the tail beats its control and keeps most of the gain without it. Basis of the `prompt_tail = sure` default. |
| [bbox_grounding.md](bbox_grounding.md) | A layout item's bbox cuts the in-box loss 14 % at σ 0.9: the two-block text refiner builds a σ-independent location code that the DiT reads over blocks ~14–35. The sanitize LoRA does its high-σ edit in blocks 26–35. |
| [text_stream_page_state.md](text_stream_page_state.md) | The template rows run a per-block trajectory the image needs and no page state it reads; a layout item's rows pick up the page from block 26 and the image reads it back in blocks 26–41, as strongly as it reads the caption's colour items. |
| [ko_token_split.md](ko_token_split.md) | swap_text's Korean wrong syllables follow Qwen tokenization: the same letters split differently come out right, more epochs move nothing. Basis of `token_split = syllable`. |
| [vision_tower_reference.md](vision_tower_reference.md) | An image put through Qwen3.5's own vision tower conditions MOSS zero-shot: content and coarse layout come back, pixel alignment does not. |

## Sampling and numerics

| Doc | What it covers |
|---|---|
| [sigma_staircase.md](sigma_staircase.md) | One forward's `x0_pred` resolves at Anima's σ knee, but layout is set by σ 0.87–0.80, steps 22–28 of 50. Spectrum's shipped growing window beats log-SNR, step-even and NFE-16 schedules against it; an open-loop skip-cost score ranks them backwards. |

Closed numerics, kernel and checkpoint checks (fp32 stream, FA2 tile, bf16 HMMA, the heal
finetune) are in [`../reports/`](../reports/README.md).

## Training recipe

| Doc | What it covers |
|---|---|
| [t2i_lora_recipe.md](t2i_lora_recipe.md) | How the t2i defaults were set on caststation: 5e-5 / clip 1 over looser clips; Muon spreads the same step over ~3× the rank directions; `weight_svd` feeds the loud channel ch 42 and the input DC, kaiming does not, and with kaiming channel scaling does nothing; an adapter EMA keeps lr 1e-4 on the prompt. |
| [mod_rank_i2i.md](mod_rank_i2i.md) | `mod_rank` 16 on i2i: the copied tokens' loss drops, preservation outside the edit is visibly better, the edit ties; on sanitize it renders warmer with the tone term. The reference-only part of the modulation delta is shared across i2i tasks. |
| [reference_adapters.md](reference_adapters.md) | Separate adapters for the reference rows on swap_text: k / v exports alone are 29 % faster and lose the copy; a full bank of its own ties on copy, costs 1.2 GB and drops untranslated glyphs. The shared bank stays. |
| [colorize_v5_control.md](colorize_v5_control.md) | Colorize v5: control captions (box-only text items, character colours, no OCR text) in place of v4's OCR caption, the arm that settled the recipe and the shipped v5. |
