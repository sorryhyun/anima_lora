# Anima: an attention-alignment loss for OCR text in LoRA training

Status: proposal, 2026-11-20. Two probes are run (`bench/xattn_loc/`); the loss and the A/B below
are not built. The next step is the A/B in § 5.

The idea (user, 2026-11-20): a caption's quoted OCR string has a known place on its training
image, the OCR line box. During LoRA training, add a term that puts the string tokens'
cross-attention on that box. The shipped vocab pack rows (`jp_v1`) stay frozen and only the LoRA
trains; the first target is the channel caststation LoRA.

## 1. Sources

- UDiffText (arXiv 2312.04884, PDF read 2026-11-20): SD 2.0 inpainting, a character-level text
  encoder, and a local attention loss that ties each character token's cross-attn map to that
  character's segmentation mask (Eq. 8), trained on the cross-attn layers only (75.9 M of 891 M
  parameters). Ablation, sequence accuracy on SynthText reconstruction (Table 2): base 8.0, +
  character encoder 40.0, + local attention loss 54.0, + recognition loss 65.0, + inference-time
  latent refinement 76.0. English, at most 12 characters, the text region given as an input mask.
- ABSS (arXiv 2605.19532, PDF read 2026-11-20): early cross-attn mass on the subject tokens
  ranks seeds. On recent DiTs the selection gain is small (Hunyuan-DiT DrawBench HPS 0.2982 →
  0.3002). It is where this line started and sets nothing below.
- `project/cjk_anima_reseed/reports/xattn_box_2026_10_15.md`: a training-free logit bias that
  pins the string's keys to a box at σ ≥ 0.85, every block. String keys alone: +19 letters a
  page, g_f1 −0.094. With the `speech bubble` / `japanese text` tag tokens pinned too: in_box
  +0.067 (p 0.033), every text read not significant, repetition unchanged.
- `project/cjk_anima_reseed/idea.md` and `status.md` § 5 – 6: the box-share FM loss gives every
  row in a box the same credit (`probe_twin`: neighbours' rows turn 0.6 – 0.9 as much as the
  swapped glyph's own); per-occurrence credit is listed as not tried.
- `project/finished/cjk_anima_scale/future.md` § 1: training the rows themselves on real OCR
  pages dropped every read (words 16 → 0 / 104). The rows are frozen here.

## 2. Evidence: where the string's attention is today

Both probes read the base DiT with the shipped pack, no LoRA, compile off, and report the eager
softmax map (head mean) at the string's T5 positions. "Lift" is the share of a map inside a box
over the box's share of the image tokens; 1 is no preference.

### 2.1 Real training pages (`bench/xattn_loc/probe.py`)

Run `bench/xattn_loc/results/20261120-1256-base/` (local): 694 pages whose caption carries a text
clause and whose OCR sidecar holds the line's box, 2,987 strings, one forward per σ on
`x_t = (1 − σ)·x0 + σ·ε` with the cached v0 context. FM MSE 0.074 – 0.197 across the five σ.
Blocks 13 – 15, mean over strings:

| σ | own line box | other lines' boxes | random caption tokens | order ρ | own-cell share |
|---|---|---|---|---|---|
| 1.0 | 1.08 | 1.06 | 1.00 | 0.11 | 1.01 |
| 0.9 | 1.44 | 1.35 | 1.00 | 0.25 | 1.04 |
| 0.75 | 2.10 | 1.77 | 1.03 | 0.49 | 1.14 |
| 0.6 | 2.38 | 1.85 | 1.06 | 0.64 | 1.26 |
| 0.45 | 2.45 | 1.81 | 1.08 | 0.70 | 1.35 |

- Blocks: the lift is at blocks 8 and 13 – 15; blocks 20 – 27 read 1.00 at every σ. Heads (mean
  over the five σ): (15, 1) 3.77, (14, 15) 3.50, (13, 7) 3.37, (8, 14) 2.73; the block means are
  1.34 – 1.54.
- Own against other lines, on the 2,803 strings with another line on the page: own is the larger
  for 51 % of strings at σ 0.9, 56 % at 0.75, 64 % at 0.45.
- Order (848 one-row / one-column lines of ≥ 4 tokens): ρ of token order against the attention
  centroid along the line; the own-cell share is against chance 1.
- The `speech bubble` / `japanese text` tag tokens over the union of the page's text boxes
  (167 pages): 1.2 – 2.0 at σ 0.9 on blocks 4 – 16, and 1.1 – 1.3 already at σ 1.0.
- Caststation alone (24 strings): own 1.98 / 4.10 / 5.60 and other 1.62 / 2.64 / 3.04 at σ 0.9 /
  0.75 / 0.45.

### 2.2 The generation path (`bench/xattn_loc/gen_probe.py`)

Run `bench/xattn_loc/results/20261120-1303-gen-ruler/` (local): the reseed line's dialogue ruler
(96 strings, each in its own image's caption), seeds 0 and 1, 28 steps, cfg 4, 192 renders, read
by the tuned reader with the ruler's scores. Reads: g_f1 0.245, g_r 0.592, g_p 0.196, 40.5 letters
and 8.1 text boxes drawn a page, exact 2.6 %, ≤ 2 edits 16.1 %.

The string's map over every text box the page letters (blocks 13 – 15, band means over the
sampler's steps), against the read:

| | σ ≥ 0.85 | σ 0.6 – 0.85 |
|---|---|---|
| lift, renders at or above the median g_f1 / below it | 1.96 / 1.52 | 3.65 / 2.14 |
| Spearman with g_f1, 192 renders | 0.50 | 0.63 |
| same string, two seeds: the higher lift has the higher g_f1 | 67 / 94 | 72 / 94 |
| Spearman with g_f1 given the letters drawn, 192 renders | 0.13 | 0.18 |
| the same on the seed-pair differences | 0.28 | 0.39 |
| Spearman with g_r | 0.13 | 0.11 |

- Letters drawn against g_f1: Spearman −0.69. Lift against letters drawn: −0.63 (σ ≥ 0.85).
- On the 86 renders with both kinds of box, the boxes holding the string's letters take a lift of
  1.97 against 1.54 for the other text boxes at σ ≥ 0.85 (the larger on 73 %), 3.64 against 2.20
  at σ 0.6 – 0.85 (80 %).
- Order: 38 renders have an elongated box holding the string; ρ against cer −0.48 (σ ≥ 0.85) and
  −0.67 (σ 0.6 – 0.85).

Spread: unknown for both probes (one ε per page, two seeds per string). The σ 1.0 row and the
random-token column are the meter's nulls.

## 3. Reading

- The routing exists and is narrow: four blocks, about one head in each. A loss has a small,
  named target.
- At the σ where a page's text layout is set (≥ 0.85) the string knows "a text region" and little
  more: its own line is not preferred over another line, and glyph order is near absent. Both
  appear lower down, where the real `x_t` already shows the letters, so on real pages most of the
  low-σ alignment is likely the model following what it sees.
- On the generation path alignment goes with the read, also inside a string across seeds, which
  was the criterion set before the run. Most of that relation runs through how many letters the
  page draws, and none of it through recall. The loss should be expected to move where text goes
  and how much junk is lettered, not which glyphs come out.
- Cause is open. A page laid out with many bubbles would lower the lift and raise the junk
  together, and the ruler's captions are their source pages' captions, some of which ask for
  such pages.
- `xattn_box` forced the same thing at inference on every block and found no gain in the reads.
  Its two lessons carry over: the tag tokens have to move with the string, and pulling the
  string's mass off the rest of the page is not free.

## 4. The loss

Per training item with a text clause, at a step with σ ≥ 0.6:

- `A_b[q, c]`: block `b`'s cross-attn probability from image token `q` to T5 position `c`, head
  mean, softmax over all 512 positions as the model computes it. Blocks `B = {8, 13, 14, 15}`.
- For a string `s` with positions `C_s` and its line box `R_s` (image tokens whose centre lies in
  the box, dilated half a token): `p_s(q) ∝ Σ_b Σ_{c ∈ C_s} A_b[q, c]`, normalised over `q`.
- `L_str = − mean_s log Σ_{q ∈ R_s} p_s(q)`.
- The tag tokens (`speech bubble`, `japanese text`, where the caption has them) get the same
  term against the union of the page's usable OCR boxes: `L_tag`.
- `L = L_FM + λ · (L_str + L_tag)`.

Each map is normalised over the image, so the term says where a token looks and leaves how much
it is attended alone; the padding sinks and the token's mass are not targets.

A second form, for the arm in § 5: `L_str` with the denominator restricted to the page's text
boxes, `− log (own / (own + other text boxes))`. It only moves the string between text regions
and leaves its mass on the rest of the page, which is the part `xattn_box` found load-bearing. It
needs a second line on the page.

Inputs, all on disk: the line boxes in `post_image_dataset/ocr/<set>/{stem}.ocr.txt`
(`anime_tools.captions.ocr_sidecar.text_boxes`, the same floors the caption clause uses), and the
token positions from `HybridT5Encoder.encode_aligned` on the caption variant the step drew.

Build notes:

- A `MethodAdapter` beside REPA's (`library/training/repa.py`). The blocks' `_forward` is
  compiled, so the query is rebuilt outside the graph from the block's input (a pre-hook on
  `block.__call__`, as REPA's hook sits): the block's self-attn half is run again for the four
  blocks, with grad. Step time and VRAM cost are not measured.
- The strategy draws the caption variant (`v0` – `v3`) and drops the index. The adapter needs it
  in the batch.
- A flipped or cropped item needs its boxes moved the same way, or is left out of the term.

## 5. The A/B (next)

Question: does the term change what a caststation LoRA letters, on strings it was not trained on?
A gain in the reads or on the user's read of the sheets → the 694-page arm (§ 6). Level with the
twin → the same A/B on the 694-page set once before the line closes, since caststation gives the
term few steps. A loss (more junk, damaged style) → the contrast form only, or stop.

Parent: the shipped recipe (`configs/methods/lora.toml`: r32 / α128, LR 5e-5, GA 2, 8 epochs,
shuffled caption variants, REPA off, compile on) on channel caststation, 30 images, about 120
updates.

| arm | change from the parent |
|---|---|
| `plain` | none |
| `twin` | the training seed |
| `box_lo` | + λ · (L_str + L_tag), λ at a gradient-norm ratio of 0.1 to FM |
| `box_hi` | the same at a ratio of 0.5 |
| `contrast` | the second form of L_str, at the ratio `box_lo` / `box_hi` read better at |

- λ: no record sets it, so two values are run. Each is fixed from the ratio of the two terms'
  gradient norms on the LoRA parameters over the first 20 term-bearing steps of `plain`'s own
  data and σ draws.
- What the term sees on this set: 12 of the 30 images carry a text clause (24 strings), and σ ≥
  0.6 is about a third of the draws under the sigmoid sampler at shift 1, so roughly 30 of the
  240 micro-steps. The run cannot separate "the term does nothing" from "the term had too few
  steps".

Read, same script and settings for every arm (28 steps, cfg 4, seeds 0 – 2):

- Working set, to build: the 12 caststation captions that carry text, each with its quoted
  strings swapped for dialogue-ruler strings of the same length (the cjk_rl pool's rule). The
  tuned reader's g_p, letters drawn, g_r, g_f1 and the on-string share of the text area, each
  arm beside `plain`.
- Sheets for the user: `plain` against one arm, two columns, one (prompt, seed) per file, no arm
  labels, shuffled per prompt. The user's read decides.
- Style: the four LoRA-recipe prompts (`docs/anima/findings/lora_recipe.md`) on the same sheets,
  to see that the term costs the LoRA nothing elsewhere.
- Bars: the `plain` / `twin` gap on each number. An arm inside it is recorded as level.
- Ship set, read only if an arm is taken forward: the 96 dialogue-ruler strings with the LoRA on.
  The base's read on it is § 2.2 (g_f1 0.245, g_p 0.196, 40.5 letters).

Preconditions, each checked at smoke scale:

- The adapter's maps equal the probe's on the same item and σ (the probe is the reference).
- The token positions it uses match the cached mask of the variant the step drew.
- The term's gradient reaches the LoRA at a size the optimizer moves, and step time and VRAM with
  the adapter on are known before the five runs are queued.

Cost: the adapter and its checks, then five runs of a few minutes each plus renders and reads.
One to two hours in all.

## 5a. Runs (2026-11-20)

### Facts

- Caststation (`bench/xattn_align/run_ab.py table`, 12 text prompts × seeds 0 – 2, n = 36
  renders per arm): plain `xaplain_1120_1357`, twin `xatwin_1120_1358`, box_lo
  `xaboxlo_1120_1359` (λ 0.00469), box_hi `xaboxhi_1120_1400` (λ 0.02347), contrast
  `xacon_1120_1451` (box_lo's λ).

  | arm | g_p | g_r | g_f1 | drawn | a_p | text_area |
  |---|---|---|---|---|---|---|
  | plain | 0.303 | 0.569 | 0.372 | 31.9 | 0.293 | 0.041 |
  | twin | 0.344 | 0.323 | 0.291 | 13.7 | 0.338 | 0.018 |
  | box_lo | 0.410 | 0.445 | 0.399 | 17.9 | 0.494 | 0.021 |
  | box_hi | 0.267 | 0.286 | 0.249 | 14.9 | 0.168 | 0.017 |
  | contrast | 0.324 | 0.490 | 0.371 | 24.9 | 0.367 | 0.037 |

  The user's partial picks: box_lo 9 : 7 against plain (2 ties; 12 of 18 sheets were style),
  box_hi 5 : 6 or 4 : 7. box_hi dropped.
- Sincos, at caststation's λ (`--set sincos`; plain `xsplain_1120_1511`, box_lo
  `xsboxlo_1120_1512`, n = 36): plain g_p 0.167, g_r 0.493, g_f1 0.216, drawn 43.8, a_p 0.188,
  text_area 0.061; box_lo 0.199, 0.485, 0.242, 33.4, 0.228, 0.081, one exact string of 36.
- Sincos blind set `s44_XATTN_sincos` (24 pairs: 12 text prompts at seed 0, 4 style prompts ×
  3 seeds; sides shuffled; key and verdicts in `output/blind/s44_XATTN_sincos/` (local), the set
  removed from the blind repo after scoring): box_lo 11, plain 12, tie 1; text box_lo 8 : 4;
  style plain 8 : 3 + 1 tie (style 3 plain 3 : 0, style 0 box_lo 2 : 1).
- The user's read (2026-11-20): "파볼만한 가치는 있는거같아 이를테면 ぱん 이 p쪽은 パン 식으로
  '외워서 낸듯한' 글자가 나왔거든" — on the prompt quoting `ぱん` (sincos prompt 11, pair
  p11), plain drew `パン`.
- Bars: the caststation plain / twin gap on g_f1 is 0.08; no twin on sincos, so its spread
  is unknown.

### Interpretation

- On text the term reads as a gain on both sets' g_f1 (inside the caststation twin gap) and on
  the user's text picks on sincos (8 : 4); style lost 3 : 8 on sincos, which the term was meant
  to leave alone. Whether the style losses are distortion (the user's first criterion) or a
  style shift is not separated.
- The `ぱん` → `パン` case is the failure the term targets: a string written as a word the model
  has memorised rather than as the quoted letters. The line goes on (§ 6).

## 6. After the A/B

- The 694-page set: one LoRA on every page with OCR text, where the term has thousands of
  term-bearing steps, read on the dialogue ruler.
- Per-glyph targets: on one-row / one-column lines the box divides into cells, token `k` to cell
  `k`. § 2.1's order column is the baseline (ρ 0.25 at σ 0.9).
- The four heads instead of four block means, if the block-mean term proves diffuse.

## 7. What this does not address

- Glyph identity: neither probe ties alignment to whether the asked letters come out.
- Slot count and repetition, which `xattn_box` found do not follow placement.
- Long strings: the ruler's long bin reads exact 0 / 64 on the base.
