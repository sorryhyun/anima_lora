# The chat template is MOSS-Image's attention sink

Measured 2026-10-18 on MOSS-Image Base. Probe, runs and per-block tables:
`bench/moss_sinks/` (`probe_attention.py`, `README.md`).

## Setting

MOSS is a single-stream DiT: the sequence is `[image (H·W) | text (S)]` through all 42
blocks, and the text is never padded. Each prompt is wrapped in the Qwen chat template
before the text encoder, 3 prefix and 7 suffix tokens
([layout](../structure.md#21-qwen35-9b-one-prompt-per-forward)).

The uncond prompt is `''`, so the uncond text segment is these 10 tokens alone. Qwen is
causal, so the 3 prefix tokens have the same TE states for every prompt; the 7 suffix
tokens read the caption. On Anima the zero-pad tail of the 512-token context was the
null-attention sink (`_archive/null_key/report.md`: K = 0, V = 0, adding only to the softmax
denominator). MOSS has no pads, and the per-query OFF-gate the pad tail gave Anima has no
other carrier in a MOSS block (`gate_a` is per channel and time-only). arXiv 2607.19139
("Text Template Tokens Are Implicit Semantic Registers in DiTs", Qwen-Image) describes
template tokens taking this role.

All runs: 1024², 30-step Euler, CFG 3, uncond skips block 22; anime_sunset + ja_manga ×
seeds 1000–1003; read on steps 0 3 7 11 15 19 23 29.

## Measured

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

## Reading

The 10 template tokens do two jobs: the OFF-gate Anima's pads were (a head with nothing to
read parks mass there), and a learned, mostly prompt-independent per-head bias channel into
every image token.

## Hypotheses (not tested)

- **Editing the template is off-distribution.** A shortened or altered template, or real
  values in its place, would break the gate the way real pad states collapsed Anima's
  images.
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
