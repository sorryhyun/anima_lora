# The chat template: MOSS-Image's attention sink, and the assistant-turn tail after it

Two reads of the same 10 rows. §1 (2026-10-18): the template is the image's attention sink, an
OFF-gate plus a per-head bias, and trained LoRAs barely move it. §2 (2026-11-03): appending an
assistant-turn tail after the template (best `\n</think>\n\nSure,`) sharpens renders; the image
reads `</think>`, the first heavy key that varies with the caption. §2 is the basis of the
`prompt_tail = sure` default.

## Setting

MOSS is a single-stream DiT: the sequence is `[image (H·W) | text (S)]` through all 42
blocks, and the text is never padded. Each prompt is wrapped in the Qwen chat template
before the text encoder and ends at the generation prompt with thinking on:
`<|im_start|>user\n{caption}<|im_end|>\n<|im_start|>assistant\n<think>\n` — 3 prefix rows,
the caption, 7 suffix rows ([layout](../structure.md#21-qwen35-9b-one-prompt-per-forward)).

The uncond prompt is `''`, so the uncond text segment is these 10 tokens alone. Qwen is
causal, so the 3 prefix tokens have the same TE states for every prompt; the 7 suffix
tokens read the caption. On Anima the zero-pad tail of the 512-token context was the
null-attention sink (`_archive/null_key/report.md`: K = 0, V = 0, adding only to the softmax
denominator). MOSS has no pads, and the per-query OFF-gate the pad tail gave Anima has no
other carrier in a MOSS block (`gate_a` is per channel and time-only). arXiv 2607.19139
("Text Template Tokens Are Implicit Semantic Registers in DiTs", Qwen-Image) describes
template tokens taking this role.

## 1. The template is the attention sink

Measured 2026-10-18 on MOSS-Image Base. Probe, runs and per-block tables:
`bench/moss_sinks/` (`probe_attention.py`, `README.md`). All runs: 1024², 30-step Euler,
CFG 3, uncond skips block 22; anime_sunset + ja_manga × seeds 1000–1003; read on steps
0 3 7 11 15 19 23 29.

### Where attention concentrates

- On 35 of 42 blocks, ≥ 25 % of heads put ≥ 30 % of their image-query mass on the top 1 %
  of keys, and those keys are the template. Across heads, the top-1 % share tracks the
  template's mass at r = 0.98.
- Per key, image queries give `<|im_start|>` ~300× the mass of an image token, each prefix
  token ~220×, each suffix token ~80×, a caption token ~2.5×.
- The template takes 0.24–0.34 of image-query mass on average, U-shaped over the run
  (highest at the first and last steps). Text queries also put ~0.35 of their mass on it.
- By block: strongest on 19–21 and 25–29 (~10 of 24 heads put ≥ half their mass there),
  weakest on 0, 4, 7 and 35–41. Blocks 36–41 keep 0.8–0.88 of the mass on image keys.
- Cond and uncond look almost the same; uncond puts a little more on the template at step
  0 (0.38).

### No image-token sink

- With the 18 massive-activation channels left out
  (`library/moss/calibration/massive_channels.npz`), image-token norm max/median is 1.1–1.8
  on every block. The high norms are on the text side: over the image median, prefix ~21×
  (up to 39×), suffix ~10×, `<|im_start|>` ~7×, caption ~5.5×. Only image-derived massive
  channels were left out, so part of the text-side ratio may be the text stream's own.
- The massive channels themselves (1356, 42, …) are a near-constant offset on every image
  token (DC 0.98–0.99 of the energy on blocks 0–18, `project/finished/moss_lora/done/dave.md`), not a
  few outlier tokens.

### What the template writes

Values also carry that offset, so writes are measured on values centred on the image keys'
mean v̄_img. Since Σp = 1, `out − v̄_img = Σ p·(v − v̄_img)`: a key whose value is v̄_img
writes nothing (the analogue of Anima's V = 0).

- The template writes 0.15–0.51 of the centred image-query output, 1.13–2.1× its mass share
  (~0.46–0.51 on blocks 19–21 and 25–29). It is not a null sink.
- It writes nearly the same vector for every image query: coherence 0.89–0.98 on blocks
  0–40, against 0.18–0.65 for the image keys' write. Per head, that is a fixed vector added
  to every image token, scaled by the mass the query puts on the template.
- Centred value norms relative to image values: `<|im_start|>` 0.43–0.9 (closest to null),
  prefix 0.5–1.5 past block 7, suffix 1.0–2.4. On blocks 1–7 caption and suffix values are
  3.5–9.4×.
- Cond vs uncond template write cosine: 0.93–0.96 on blocks 0–21, 0.76–0.94 after block 22.
  Under CFG the template write mostly cancels in cond − uncond.
- Across the two prompts (cond): cosine 0.70–0.88. The prompts also produce different
  trajectories, so this mixes template content with image state.

### Trained LoRAs barely move it

Two caststation layout LoRAs (r32 α128, SVD init) at ×1.0. Step 0 is the clean read (same
x_T and σ). Base's seed-to-seed CV of step-0 template mass is 0.5 %.

| | lr1e4, no channel scaling / dc_isolate | lr5e5, cs 0.5 + dc_isolate |
|---|---|---|
| step-0 template mass, mean \|Δ\| / base | 5.4 % (≤ 20 %, blocks 5, 6, 17, 23) | 3.0 % (≤ 11 %, block 8) |
| template write vs base, cosine (mean) | 0.905 | 0.948 |
| template write norm ratio | 1.06 | 1.04 |

The shift is resolved on 33–35 of 42 blocks but is smaller than the template mass swing
within one Base run (±15 % over the steps). The two LoRAs differ in lr, channel scaling and
dc_isolate at once, and a LoRA edits both the template rows' K / V and the image rows' Q, so
neither the gap between them nor the source of the shift is attributed.

Renders (`_archive/bench/moss_sinks/render_template_mask.py`, lr5e5 LoRA, raw-text prompts,
seeds 1000–1003; `output/tests/sinks_lr5e5dc_30st/`): keeping the LoRA delta off the 10
template rows leaves the style (faces, line work) in place but changes content details on 4
of 8 seeds (collar, ribbon colour, pose, expression), some back toward Base. Mean pixel
difference lora ↔ masked 5.6–24.9, against 32.5–55.4 for base ↔ lora.

### Reading

The 10 template tokens do two jobs: the OFF-gate Anima's pads were (a head with nothing to
read parks mass there), and a learned, mostly prompt-independent per-head bias channel into
every image token.

## 2. An assistant-turn tail after the template sharpens renders

Measured 2026-11-03 on MOSS-Image Base, no LoRA. Script:
`project/moss_te_sae/src/base_te_render.py`; renders in `output/tests/base_te_render/`
(eval prompts) and `output/tests/base_te_render_kikuri2/` (explicit prompts).

### Arms

The TE is causal, so a tail appended to the prompt text leaves the earlier rows' states
unchanged, except where the tokenizer merges the template's last `\n` with a tail starting
in `\n` into one `\n\n` token (`ok`, `sure`): that one suffix row changes token. Arms append
a tail to the cond prompt and, unless noted, the same tail to the empty uncond prompt:

| arm | appended to the prompt text | tokens after `<think>` | rows added |
|---|---|---|---|
| `instruct` | nothing (shipped) | `\n` | 0 |
| `instruct_ok` | `\n</think>\n\nOK` (closed empty think block, Qwen's non-thinking form) | `\n\n` `</think>` `\n\n` `OK` | 3 |
| `instruct_sure` | `\n</think>\n\nSure,` | `\n\n` `</think>` `\n\n` `Sure` `,` | 4 |
| `instruct_think` | `Okay,` (think block left open) | `\n` `Okay` `,` | 2 |
| `instruct_ok_cond` | `instruct_ok`'s tail on cond only; uncond shipped | as `ok` | 3 / 0 |
| `instruct_pad4` | four `\n` token ids (198) appended to the ids, no content | `\n` + 4 × `\n` | 4 |

Shipped INT8 TE, layout-JSON prompts, ER-SDE 50, CFG 3, Spectrum, fp32 stream, 1024²,
seeds 1 / 2. Prompts: the three `bench/moss_ab/lora_ab.py` PROMPTS (Miku and Neru tag
bags, an NL schoolgirl prompt, caststation artist tag) and two explicit tag prompts of an
adult character (Hiroi Kikuri, solo and missionary). 10 prompt × seed pairs per arm.

### Renders

Mean |Δpixel| (0–255) against `instruct`, range over the 10 pairs: `ok` 12–31,
`ok_cond` 12–32, `sure` 11–32, `think` 9–24, `pad4` 13–27. `ok` against `ok_cond`:
3.5–15 (mean 8).

- **The uncond tail barely matters.** `ok_cond` renders close to `ok` (8 against 12–31
  from shipped), so the effect comes from the cond rows, not from a changed CFG direction.
- **Length alone changes the picture but not the quality.** `pad4` (4 rows, as `sure`;
  one more than `ok`) moves the image as
  far as `ok` and, by the user's review, gives a different picture at shipped quality.
  The tokens' content carries the gain.
- **By the user's review, the tail gains detail consistently and `sure` gains the most.**
  `ok` sharpened hands and backgrounds against shipped on the eval and the explicit
  prompts; `sure` was the most detailed of `ok` / `sure` / `think`, and showed less of
  the unrequested or unintended "preparing to refuse" look on the explicit prompts.

Unblinded and two seeds per prompt; no tagger or metric score backs the quality reading.

### Attention on the tail

`project/moss_te_sae/src/tail_attn.py` (`template_attn.py`'s read with the tail rows as
a fifth key segment): Base, Euler 30 steps, CFG 3, no Spectrum, seed 1, 1024², the three
eval prompts, cond branch; `output/tests/tail_attn/20261103-1338/`. Mass and value
contribution `‖Σ_seg p·v‖ / ‖Σ p·v‖` of the image queries, mean over heads and blocks,
early / mid / late step thirds:

| tail | tail mass | tail contrib | suffix mass | `</think>` | answer rows | peak blocks |
|---|---|---|---|---|---|---|
| none | — | — | 0.11–0.13 | — | — | — |
| `ok` | 0.010–0.018 | 0.022–0.039 | 0.10–0.12 | 0.010–0.015 | `OK` 0.0003 | 19 / 15 / 12 / 10 / 17 |
| `sure` | 0.011–0.021 | 0.023–0.041 | 0.10–0.11 | 0.011–0.016 | `Sure` 0.0002, `,` 0.0002 | 19 / 15 / 12 / 17 / 20 |
| `pad4` | 0.007–0.010 | 0.019–0.037 | 0.11–0.12 | — | first `\n` 0.0065, rest ≤ 0.001 | 25 / 23 / 29 / 19 / 34 |

- **The image reads `</think>`, not the answer word.** `</think>` becomes the fifth
  heaviest key after the four sink rows (`user`, `assistant`, `<|im_start|>` ~0.07,
  `<|im_end|>` ~0.03). `OK` / `Sure` / `,` get 0.0002–0.0003, as the shipped suffix's
  caption-dependent rows do. The tail's mass comes out of the suffix (mostly
  `assistant`); the caption rows keep theirs.
- **It writes more than its mass.** Contribution 2–4 % on 1–2 % of the mass: not a
  sink that takes mass and returns little.
- **`ok` / `sure` are read in blocks 9–20, `pad4` in 23–34.** The extra `\n` rows act
  as a copy of the template's last `\n` (0.0065 vs 0.007) and are read late.
- **`</think>` is the first heavy key that varies with the caption.** Cross-prompt spread
  of each row's value (`mean_p ‖v_p − v̄‖ / ‖v̄‖`, mean over blocks, step 0 / 29): the
  four sink rows 0.02–0.03 in every arm; `</think>` 0.29 / 0.23 (`ok`), 0.23 / 0.19
  (`sure`); the answer rows 0.27–0.49. `pad4`'s one read row (the first extra `\n`) is
  prompt-constant, 0.02 / 0.02. Every shipped row the image reads heavily is constant
  (`project/moss_te_sae/reports/text_read_path.md` §1), so `ok` / `sure` add a caption-dependent key at sink-adjacent
  mass, and `pad4` adds another constant one.
- **Caption queries read the tail too**: 2–3 % of their mass (`</think>` 0.012–0.017;
  `Sure` 0.0015 and `,` 0.003, against `OK` 0.002).

The TE is causal, so `</think>`'s TE row is the same under `ok` and `sure`. The text
refiner (`text_embedder`, 2 bidirectional blocks) and the joint blocks let the answer
rows write into it before the image reads it, which is the path by which `Sure,` and
`OK` can differ while the image barely reads either directly.

### Base TE arms (same run)

Qwen3.5-9B Base in place of the instruct TE: the shipped template (`base`), instruct's
template rows around Base's caption rows (`splice`), Base without the 7 suffix rows
(`base_nosuffix`), Base with `Describe this image: ` before the caption
(`base_describe`). None rendered better than `instruct`; `base_nosuffix` did not
collapse (the prefix keeps 3 of the 4 heavy sink rows) but rendered flatter, more
saturated, doll-like. Base's conditioning stays linearly close to instruct's (linear CKA
0.84 at the text-embedder output over 230 captions) with its variance on fewer
directions (participation ratio 72 vs 113); `project/moss_te_sae/src/te_stats.py`,
`output/tests/te_stats/20261103-1241/`.

### A LoRA trained with the tail

`prompt_tail = sure` on `CacheRequest` / `GenerateRequest` (`library/moss/prompt_tail.py`),
stamped in the TE cache and the LoRA; generate's `auto` follows the LoRA. Measured
2026-11-03 on caststation (30 images), INT8 TE for cache and render, `configs/moss/lora.toml`
with lr 1e-4 and `ema_window` 0.139 (8 epochs, 240 steps, seed 0). The two arms differ only
in the TE cache: `channel_(caststation)_layout` (shipped) against `_layout_sure`. Renders:
ER-SDE 50 + Spectrum, CFG 3, bf16 DiT and stream, 1024², the three `lora_ab.py` prompts × 2
seeds; `output/tests/te8_tail/`.

| arm | LoRA | render tail |
|---|---|---|
| control | `caststation_te8_int8` | off |
| `sure` | `caststation_te8_sure` | sure (`auto`) |
| `sure` LoRA, no tail | `caststation_te8_sure` | off |

- **By the user's review, `sure` renders significantly better than the control**: more
  detailed clothes and hands (Neru's vest and ring), livelier expressions, at the same
  style strength. One `sure` render (Neru, seed 4) has a blank margin along the top and left.
- **Most of the gain survives without the tail at render time**: the `sure` LoRA rendered
  without the tail sits close to the `sure` arm and away from the control, so the
  difference comes mostly from training on the tail cache.
- Training loss is the same in both arms (final epoch 0.2839 / 0.2840).

One training seed per arm and 6 renders per arm. The control LoRA with a tail at render
time was not rendered.

Each tail cache row is the shipped row bit-for-bit in 57 of 60 sequences. In 3
(`12248671` full, `8293513` / `8508115` plain) the rows from position 64 on differ by up to
0.43 relative, which fits the GDN layers' 64-token chunks changing numerics with the total
length (not checked further).

`prompt_tail = sure` became the cache and t2i generate default on 2026-11-03 on the strength of this run; the i2i
configs pin `off`.

## Hypotheses and open checks (not tested)

On the sink (§1):

- **Editing the template is off-distribution.** A shortened or altered template, or real
  values in its place, would break the gate the way real pad states collapsed Anima's
  images. (§2's tail appends after the template and leaves its rows in place.)
- **The prefix is a pure register; the suffix carries some caption content.** Basis:
  identical prefix TE states across prompts, larger suffix value norms. A per-class
  cross-prompt read at matched latents (same x_t, two prompts, one forward) would separate
  this from the trajectory difference.
- **Caption format moves the gate.** A long layout-JSON caption gives image queries more
  real keys, so template mass may drop with caption length.
- **Null-text optimisation on MOSS moves the sink.** Null-TTA (arXiv 2511.20889) optimises
  the CFG null embedding; on MOSS that embedding is the template span, so it would act on
  the gate and the bias rather than a semantic direction.
- **A LoRA could use the template as a global lever**, the token-axis counterpart of the
  channel DC that `dc_isolate` removes. The measured LoRAs move its attention only a little,
  but the masked renders show the template-row delta changes attributes. Whether masking
  helps is not read yet (layout-JSON prompts, more seeds).
- **A sink-bias knob at inference.** A per-block log-bias on the template keys (Anima's
  `_ctx_k_bias` form) would set how far each head shuts off, without retraining.

On the tail (§2):

- Instruct reads a user turn as a request (`project/moss_te_sae/hypothesis.md`, H1).
  Ending the prompt mid-answer, with an affirmative start (`Sure,`), puts the last rows
  in a state of complying with the caption rather than deciding whether and how to
  answer, and the DiT reads that state.
- The `</think>` row carries it, read in blocks 9–20, with the answer rows reaching it
  through the text refiner and the joint blocks. Untested: cutting the image's keys to
  `</think>`, or to the answer rows, per arm (`template_causal.py`'s cut arms).
- More training seeds and a blind pick for the tail LoRA.
