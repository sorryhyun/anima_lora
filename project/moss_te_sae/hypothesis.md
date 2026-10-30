# hypothesis — the caption is a user turn to an instruct model (2026-10-29)

Status: hypothesis; test 3 run (2026-10-29), tests 1 / 2 / 4 not. The two binding arms
from text_read_path.md §4 ran on 2026-10-30 ([trigger_binding](reports/trigger_binding.md)). Every test below needs the 9B TE on the GPU (no CPU
runs while a training job holds the box).

MOSS encodes every caption as a **user turn of the post-trained Qwen3.5-9B** (instruct,
not Base: README "Facts settled") and reads `last_hidden_state` (final norm of layer 31):

```
<|im_start|> user \n  {caption}  <|im_end|> \n <|im_start|> assistant \n <think> \n
```

The instruct model is trained to read that slot as a request it is about to answer. A
tag bag or a layout-JSON block is neither a question nor an instruction. The question is
whether that framing shapes the caption rows in a way that explains why a LoRA trigger
does not bind on MOSS while it does on Anima. Anima differs on every axis at once: its TE
is **Qwen3-0.6B Base** (`qwen_3_06b_base.safetensors`), fed the raw caption with no chat
template (`library/anima/strategy.py::tokenize`), and its `llm_adapter` output is
indexed by T5 vocab rows (`cache_llm_adapter_outputs = true`). An Anima-vs-MOSS
difference alone cannot be pinned on the framing.

## What is already measured

- **The DiT can use a name as a key.** Neru, GUMI, Haku, MEIKO render from the name
  although the TE does not know them; splicing the name's own rows into an invented-name
  caption keeps 0.96–0.98 of the name's one-step effect ([name_rows](reports/name_rows.md)). So MOSS has per-name slots for names it saw in pretraining.
- **A trained LoRA does not use one.** `caststation_ema097_lr1e4` keeps cos 0.74 / 0.83
  (σ 0.3 / 0.7) with an empty prompt; artist and character tokens each carry 2.2 % of
  its gradient × input, the top tokens are commas and JSON delimiters ([text_read_path](reports/text_read_path.md) §4). The trigger is written as text on a name plate in all three
  optimizer arms (`docs/moss/findings/muon_lora.md`).
- **The style is not routed through the template rows.** Keeping a LoRA's delta off the
  10 template rows leaves the style in place and changes content details on 4 of 8 seeds
  (`docs/moss/findings/attention_sink_chat_template.md`, "Trained LoRAs barely move it").
  With the empty-prompt result, the style shift lives mostly on the image rows.
- **The massive channels carry no per-token content.** ch 42 at the `ffn_up_gate` input:
  |x̄| / mean|x| = 1.00 on blocks 12–41, max/mean 1.3–2 late, adaLN shift −4.3. Cutting it
  (`svd_drop_loud 1`) moved DC and composition, not style uptake
  (`docs/moss/findings/dcfree_svd_init.md`).
- **The template moves caption rows off the Base distribution.** The Base-trained SAE
  reconstructs raw text better than the same text inside the template: FVU 0.21 vs 0.25
  (layer 31), 0.24 vs 0.41 (layer 15) on the NL prompts ([sae_fit](reports/sae_fit.md)).
  Confounded: the SAE is Base, the TE is instruct, so this mixes "instruct weights" with
  "user-turn framing". Test 3 ([text_read_path](reports/text_read_path.md) §3) separates
  them: the template alone moves Base's layout rows 0.10, the weights alone 0.24, both
  0.60 at L31. The L15 FVU gap was most likely the raw sink token.

## H1 — caption rows encode a request, not a description

Inside a user turn the instruct TE writes onto each caption token what it needs to
answer: task type, language, format, what is being asked. The content of a tag is
carried as part of that request representation, mixed with its neighbours.

- Predicts: the same caption encoded raw (no template) and in the shipped template
  differs on caption rows by much more than the INT8 / bf16 gap (2.1 %), more at layer 15
  than at 31 (the SAE FVU gap above).
- Predicts: an explicit request wrapper (`Draw this image: {caption}`) moves caption rows
  less than the shift from raw to templated, because the model already reads the turn
  as a request.

## H2 — context dominates a new name's rows

At the last layer an instruct model's state is shaped toward the next token. For a name
the TE does not know, what comes next (`,`, `"`, `}`) and the surrounding request decide
most of the row; the name itself decides little. A row with little name-specific
content gives the LoRA no stable key, so its gradient goes to the rows that pool the
whole caption (commas, delimiters) and to the image rows, where the update is the same
for every prompt.

- Fits: Neru's SAE top-100 overlaps an invented name more than Miku (60 / 54 vs 47 / 45
  at L31); 2.2 % gradient share on name tokens; the empty-prompt cos.
- Predicts: for N names × M contexts, the share of row variance explained by the name
  (vs the context) is small on MOSS, at the TE output and at `text_embedder` out, and
  large on Anima's `llm_adapter` output for the same names in tag captions.
- Predicts: names the DiT knows (Miku, Neru) keep a larger name share than invented ones
  on MOSS — the DiT learned keys from whatever name-specific part survives.

## H3 — non-English triggers survive as names

The community report (issue #112): Chinese or Russian character triggers recall a
character better than English words, and keep two characters' outfits apart. An English
word inside an English request is read for its meaning and absorbed into the request; a
foreign-script span in an English caption is read as a proper name and keeps a distinct
row.

- Predicts: under H2's decomposition, the name share is higher for a CJK or Cyrillic
  trigger than for an English-word trigger of the same token length.
- Predicts: a LoRA trained with a CJK trigger puts more gradient × input on the trigger
  (`src/lora_use.py`) and less of its effect survives an empty prompt.

## Alternatives that explain the same failures

- **Prior, not framing.** MOSS is a general model with little anime-tag pretraining; a
  LoRA has no existing slot to sharpen, as Anima's tag-trained base has. H2's
  decomposition separates the two: framing predicts a small name share for every new
  name, the prior predicts nothing about rows.
- **The objective, not the TE.** A LoRA fitting 30 images in one style is best served by
  a prompt-independent shift whatever the conditioning looks like. The test is H3's
  training arm: if a distinct trigger row still gets ~2 % of the gradient, the trigger is
  not the bottleneck.
- **Instruct weights, not the user-turn framing.** H1's raw-vs-template read cannot tell
  these apart. Qwen3.5-9B-Base (the Qwen-Scope SAEs are trained on it; not downloaded)
  separates them on the TE side only: raw text through Base vs through instruct. The DiT
  never saw Base states, so Base is not a conditioning candidate.

## Tests (queued for a free GPU; TE-only unless marked)

1. **Wrapper sweep (H1).** 50 layout-JSON + 50 NL captions × {shipped template, raw,
   template with a request prefix}. Per-row relative change vs shipped at layers 15 / 31
   and at `text_embedder` out; Base-SAE FVU per wrapper. Kill H1 if the template-vs-raw
   change on caption rows is within ~2× the INT8 gap.
2. **Name × context decomposition (H2, H3).** ~20 names (DiT-known, TE-known, invented;
   English-word, CJK, Cyrillic triggers) × ~20 contexts from the `*_layout` caches.
   Two-way variance split of the name rows at TE out / `text_embedder` out; the same names
   through Anima's `llm_adapter` (cheap). Kill H2 if MOSS's name share is close to
   Anima's.
3. **Base contrast (alternative 3).** Qwen3.5-9B-Base vs instruct × {raw, shipped
   template}: a 2 × 2 that separates the weights from the framing (Base's tokenizer ships
   the same chat template). Base is at `/media/sorryhyun/data/qwen35_base/Qwen3.5-9B-Base`
   (bf16, 4 shards, downloaded 2026-10-29).
4. **Training arm (H3, ask first, ~25 min).** The channel set with its trigger replaced by
   a CJK / Cyrillic string; read `lora_use.py` (trigger share, empty-prompt cos) against
   `caststation_ema097_lr1e4`.
