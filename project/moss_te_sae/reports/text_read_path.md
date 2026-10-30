# How the DiT reads the prompt, and what a trained LoRA uses

Scripts: `src/template_attn.py` + `src/template_attn_cmp.py` (20261029-2131 Base,
20261029-2140 Base + `caststation_ema097_mod16`), `src/template_causal.py`
(20261029-2202), `src/base_contrast.py` (20261029-2123), `src/lora_use.py`
(20261027-1626, `caststation_ema097_lr1e4`). Results in `output/tests/{template_attn,
template_causal,base_contrast}/<stamp>/`.

Setup for the attention runs: the 7 A/B eval prompts (`bench/moss_ab/lora_ab.py` PROMPTS +
`project/finished/moss_lora/composition_prompts.txt`, caststation trigger, layout JSON,
INT8 TE), Base DiT, Euler 30 steps, CFG 3, no Spectrum, seed 1, 1024², cond branch. A
prompt is 3 prefix rows (`<|im_start|>user\n`), 39–69 caption rows and 7 suffix rows
(`<|im_end|>\n<|im_start|>assistant\n<think>\n`). The earlier finding
`docs/moss/findings/attention_sink_chat_template.md` (2026-10-18) measured the same thing
on two plain-text prompts × 4 seeds; this reproduces it on layout JSON and adds the
per-row and causal reads.

## Claims

1. **The 10 template rows take ~95 % of the image tokens' text attention; the caption
   rows get 1–3 %.** The heavy template rows carry the same value for every prompt.
2. **The template gates, the caption carries.** Swapping another prompt's suffix rows
   changes nothing; cutting the caption keys loses the prompt; cutting the template keys
   collapses the image into texture.
3. **The chat template reshapes the caption rows only together with the instruct
   weights.** On Base the template moves layout rows 0.10, the weights alone 0.24, both
   0.60 (L31), and the shift grows along the caption in instruct and fades in Base.
4. **A style LoRA does not use the caption path.** `caststation_ema097_lr1e4` keeps
   cos 0.74 / 0.83 of its effect on an empty prompt; artist and character tokens each get
   2.2 % of its gradient × input; `mod16` leaves attention mass within 0.003 of Base and
   shifts values the same way for every prompt.

For LoRA triggers: a prompt reaches the image through a narrow path (the 1–3 % of
image attention on caption rows); the rest is a constant gate. A LoRA fitting a style
moves the image without that path. To make a trigger bind, the training signal has to go
through the caption-row reads. A template-side or global adapter would not do it. The
two levers now in `library/moss/train.py` (`delta_rows`, `preserve_weight`) come from
this; `hypothesis.md` holds the framing hypotheses.

## 1. Where the image reads the text

Image queries, mean over blocks (early / mid / late thirds of the steps):

| keys | mass | value contribution `‖Σ_seg p·v‖ / ‖Σ p·v‖` |
|---|---|---|
| image | 0.70 / 0.72 / 0.69 | 0.83 / 0.84 / 0.81 |
| prefix (3 rows) | 0.16 / 0.16 / 0.18 | 0.14 / 0.15 / 0.17 |
| caption body (~60 rows) | **0.026 / 0.013 / 0.010** | 0.08 / 0.05 / 0.04 |
| suffix (7 rows) | 0.12 / 0.11 / 0.13 | 0.15 / 0.14 / 0.15 |

- Per template token: `user` 0.076, `assistant` 0.073, prefix `<|im_start|>` 0.067,
  `<|im_end|>` 0.034, prefix `\n` 0.019; the other five suffix rows 0.0005–0.007 each.
  Per row ~0.05 for a prefix row against ~0.0003 for a caption row.
- The caption is read directly only by single heads in a few blocks: max over heads 0.17
  in blocks 8 / 19, 0.12–0.13 in blocks 18 / 35 / 37 / 40; the block mean is ≤ 0.04.
- The value contribution is uncentred, so the shared offset inflates it; the finding's
  centred write (0.15–0.51) is the better measure of how much the template writes.

**Cross-prompt spread of each row's value** (`mean_p ‖v_p − v̄‖ / ‖v̄‖`, mean over blocks;
at step 0 every prompt starts from the same x_T, so the spread is the prompt alone):

| row | `<\|im_start\|>` | `user` | `\n` | `<\|im_end\|>` | `\n` | `<\|im_start\|>` | `assistant` | `\n` | `<think>` | `\n` | caption mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mass | 0.067 | 0.076 | 0.019 | 0.034 | 0.0005 | 0.0013 | 0.073 | 0.0035 | 0.0007 | 0.0073 | ~0.02 total |
| spread, step 0 | 0.03 | 0.03 | 0.01 | 0.03 | 0.70 | 0.39 | 0.03 | 0.21 | 0.11 | 0.02 | 0.33 |
| spread, step 29 | 0.02 | 0.02 | 0.01 | 0.02 | 0.75 | 0.28 | 0.02 | 0.13 | 0.20 | 0.03 | 0.27 |

The four heavy rows are prompt-constant (absolute spread 1–2 on ‖v̄‖ 59–98; the caption
mean row 33 on 101): a fixed per-head bias, not a channel. The suffix rows that do vary
with the caption (the `\n` after `<|im_end|>`, 0.70) are the ones the image ignores. So
the caption reaches the image almost only through the 1–3 % of mass on its own rows.

## 2. Causal check

Every arm changes the cond branch only; uncond stays the shipped empty prompt. Reference
= the Base latent above (same eager path). Tagger scores: `own` = recall of the prompt's
tags (6 tag prompts), `donor` = recall of the donor's tags the prompt lacks, `J` =
Jaccard of the kept tag set against the base render, `px` = mean |Δ| against it (0–255;
two different prompts' base renders differ by 60–85).

| arm | own | donor | J | px |
|---|---|---|---|---|
| base | 1.00 | 0.00–0.55 | 1 | 0 |
| `swap_suffix` (7 suffix rows from prompt `(i+3) % 7`; prefix rows are bit-identical across prompts) | 1.00 | **= base on every prompt** | 0.77–1.00 | 0.4–13.5 |
| `no_caption_keys` (image queries cannot see caption rows, all blocks) | 0.09–0.38 | 0.00–0.22 | 0.08–0.26 | 52–71 |
| `no_template_keys` (image queries cannot see the 10 template rows) | 0.00–0.15 | 0.00–0.14 | 0.00–0.07 | 64–102 |

- Suffix rows carry no usable content although, in the causal TE, they read the whole
  caption: composition, pose and colours stay (largest change Neru's sleeve, px 13.5), no
  donor tag appears.
- Without the caption rows the prompt is gone: every render becomes one default figure
  (a girl in a blue jacket, arms spread, often a speech-bubble sketch). A little leaks
  through the template rows (black bob on p5, grey hair on p0 / p2); CFG 3 scales what
  remains. Recall 0.09–0.38 is generic tags (`1girl`, `looking at viewer`).
- Without the template rows all seven renders turn into texture noise, presumably because
  the mass parked on the template lands on image and caption keys. This is the
  off-distribution failure the finding predicted for editing the template.

Limits: one seed, seven prompts, Base only; the `no_*` arms cut every block at once. A
per-block cut would say which blocks carry the caption read (single heads in blocks 8 /
19 / 35 / 37 are the candidates).

## 3. Base vs instruct × raw vs template

`hypothesis.md` test 3. The SAE probe's 230 captions (150 layout-JSON, 80 NL) through
Qwen3.5-9B-Base and the instruct TE (bf16), each raw and in the shipped template; caption
rows compared position by position, the first 4 body tokens dropped (the raw sink). Mean
per-token `‖b − a‖ / ‖a‖`, layout / NL:

| edge | L15 | L31 | `out` | `proj` |
|---|---|---|---|---|
| framing on Base (raw → template) | 0.10 / 0.21 | 0.10 / 0.25 | 0.11 / 0.24 | 0.12 / 0.27 |
| framing on instruct | 0.27 / 0.24 | **0.57 / 0.40** | 0.56 / 0.37 | 0.58 / 0.40 |
| weights on raw text (Base → instruct) | 0.19 / 0.17 | 0.24 / 0.24 | 0.23 / 0.21 | 0.25 / 0.24 |
| weights inside the template | 0.32 / 0.25 | 0.60 / 0.44 | 0.61 / 0.42 | 0.67 / 0.48 |
| total (Base raw → shipped) | 0.36 / 0.33 | 0.60 / 0.49 | 0.61 / 0.46 | 0.68 / 0.54 |

- An interaction: template alone 0.10, weights alone 0.24, both 0.60 on layout rows.
  Shipped is as far from instruct-raw as from Base-raw. Centred cosine at L31: 0.99 /
  0.76 / 0.96 / 0.72 for the four edges.
- It grows along the caption in instruct and fades in Base (layout `out`, body position
  0–16 / 16–64 / 64+: Base 0.19 / 0.11 / 0.09, instruct 0.37 / 0.54 / 0.59). The template
  sets a state the instruct model holds over the whole caption.
- Mostly late-layer, and larger on layout-JSON than NL (Base shows the opposite order).
- H1's kill test (framing change within ~2× the 2.1 % INT8 gap) does not fire: ~26× the
  gap at `out`. That shows the framing matters to the rows, not that the rows encode a
  request (test 1).

## 4. What a trained LoRA reads

**Output side** (`lora_use.py`, `caststation_ema097_lr1e4`). 8 training captions + 4
`hews_layout` captions, σ 0.3 / 0.7, Δ = v_lora − v_base:

- ‖Δ‖ / ‖v‖ = 5.5 % on its own captions, 6.2 % on another artist's.
- Caption edits leave ‖Δ‖ unchanged (0.99–1.01). cos(Δ_edit, Δ_full) at σ 0.3 / 0.7:
  no artist 0.82 / 0.94, no characters 0.81 / 0.91, no tag bag 0.78 / 0.89, plain caption
  0.79 / 0.92, artist only 0.75 / 0.83, **empty prompt 0.74 / 0.83**.
- Gradient × input on the full caption: tag bag 39 %, items 35 %, template 12.5 %,
  artist 2.2 %, characters 2.2 %. Top tokens are commas and JSON delimiters, which pool
  context.

`@channel (caststation)` is not a trigger: the LoRA is a prompt-independent style shift.

**Attention side** (`template_attn_cmp.py`, Base + `caststation_ema097_mod16`, same
prompts, seed and sampler):

- Every segment's mass and contribution stays within 0.003 of Base (all step thirds,
  image and caption queries), and so does every template token's mass. The finding's
  lr1e4 / lr5e5 moved step-0 template mass 5.4 % / 3.0 %; mod16 ~1 %.
- The only attention change: the heads that read the caption directly in late blocks
  weaken (block 37 max head 0.126 → 0.075, block 35 0.125 → 0.095, block 36 0.069 →
  0.057). Block means stay.
- At step 0 the value change is small and the same for every prompt: 0.01–0.07 on
  template rows, 0.05 on the caption mean; pairwise cosine of the per-prompt deltas
  0.60–0.77 on the heavy rows, 0.70 on the caption mean. The prompt-independent shift,
  seen on the values.
- mod16 makes the late suffix rows more alike across prompts (absolute spread of `\n` /
  `<think>` / `\n` 38 / 69 / 20 → 28 / 30 / 8), but those rows get little mass.
