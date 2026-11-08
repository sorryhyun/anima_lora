# Request: isolate the text stream as the reference is isolated

Opened 2026-11-08. Not started.

## Why

MOSS is single-stream: the sequence is `[image | text | reference]`, and the text rows run all
42 blocks on every step at the image's timestep modulation. The reference used to work the same
way. `ref_isolate` (reference queries read only reference keys) and a fixed `ref_timestep` made
the reference stream depend on the reference alone, so `RefCache` runs it once per generation.

The text stream has no such switch, and on dense pages it is not small. The dense swap_text
smoke (`configs/swap_dense_{k,c}_s500.toml`) has captions of up to 2,088 tokens beside 4,104
image and 4,104 reference tokens, about a fifth of the rows. That sample set the training
peak: swap 36 needed `activation_memory_budget` 0.05.

## The question

Can the text rows' attention be restricted the way the reference's is, and what does the model
lose?

- **Cross read:** text queries read text keys only, not image or reference keys. Image queries
  still read the text keys, so the prompt still conditions the image.
- **Intra read:** text queries are restricted further within the text, e.g. each layout-JSON item
  reads itself plus the template rows, or a causal mask.

With the cross read cut and a fixed text timestep, the text stream depends on the prompt alone.
It could run once per generation, like `RefCache`. In training, its per-block K / V could even be
cached beside the TE file, so the text rows would never run through the blocks.

## What the text stream may be doing that a cut would lose

1. **Image-dependent text states:** text rows that read the image and carry what they read back
   to the image through their keys.
2. **Registers and sinks:** the chat-template rows are MOSS's attention sink
   (`docs/moss/findings/attention_sink_chat_template.md`; arXiv 2607.19139 on template tokens as
   implicit registers).
3. **Timestep-dependent text keys:** the text rows take the image's σ modulation on every step.

## Substitutes worth trying

- **Soft tokens:** learned rows per block (and per σ band), in the style of Anima's soft tokens
  (`configs/methods/soft_tokens.toml`), appended to the text keys. They could cover 2 and 3.
  They cannot cover 1, which depends on the image; measure how much of 1 matters first.
- **`trigger_registers`:** MOSS's learned text rows exist only at the input, so they are a weaker
  stand-in for 2.

## First steps

1. **Forward-only probe on Base, no training:** how much attention mass text queries put on image
   keys, per block, and how much a render (CFG 3, skip-22 uncond) changes with the cross read cut,
   at the image's timestep and at a fixed one. `bench/moss_sinks/probe_attention.py` has the hooks.
   If Base barely moves, isolation is nearly free.
2. **If it moves:** a 500-step swap_text arm with the text isolated, with and without per-block
   soft tokens, against the dense smoke pair on `inputs_ko10/`.

## Code it would need

- A `text_isolate` beside `ref_isolate` in `model._attention`: text rows `[n_img, ref_start)` read
  only those keys.
- A `text_timestep` beside `ref_timestep` (`ref_modulation` stacking a third modulation).
- A text cache like `RefCache`.
- Stamping in the LoRA, so `generate` follows it.
- Soft tokens: a per-block row bank on the adapters, held outside the blocks like the other
  adapters (`library/moss/lora.py`).
