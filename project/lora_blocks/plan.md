# Plan (2026-11-07)

**First (2026-11-11): the copy block's reference stream, `ref_adapters kv` or shared bank (`off`).**
Two copy blocks on the same data and training time: `c_fill_bw10_mod16_s800` (shared bank, 800
steps, 32.4 min) and `1111_1546` (its twin with `ref_adapters kv`, 1,050 steps, ~33 min; config
`configs/1111_1546.toml`). Decided by a 300-step sanitize fine-tune from each, the P2 recipe
(`_archive/configs/lora_blocks/sanitize_ft_cbw10_s300.toml`; the kv arm adds `ref_adapters = "kv"`),
read on the sanitize eval (`ref_gate.py --part sanitize --arms ctrl,cut`, five pages, seeds 0–2),
judged on `ctrl`: a fine-tune learns what to remove from its pairs and ships without a mask; `cut` is a side read.
Queued 2026-11-11: `configs/1111_1630.toml` (shared bank) and `configs/1111_1631.toml` (kv), renders
in `output/tests/ref_gate/20261111-sanft-1111_163{0,1}`.
Read 2026-11-11 by the user (renders side by side in
`output/tests/ref_gate/sanft_compare_20261111/`): **no winner, the two split the job**. kv picks
the right targets and keeps the rest of the page, but draws the inside of a target badly (residue
in usual_case1 s1/s2, a red blob where hard_to_sanitize2's red balloons were). Shared bank draws
the inside of a target well, but picks targets oddly (the anger-mark bubble and balloon outlines on
hard_to_sanitize1) and keeps the page a little less (the sweater pattern smudged in usual_case1
s0/s1). Reading (mine): under kv the reference rows run on Base and the image rows read them
through k/v exports, so the reference stays exact (selection, preservation) but the hole is
filled by reading what is there; under the shared bank the adapted reference rows can be
rewritten, which fills the hole and also blurs what the reference says.
The copy line and the tasks built on it follow the winner.

The problem that sets the order is Korean strokes. colorize v5 and swap_text ksplit e2 both
write the right syllables with soft, uneven strokes (user, 2026-11-07), and swap_text
still gets an occasional syllable wrong. The VAE is ruled out: a round trip of real Korean
pages leaves the strokes as they were (2026-10-27), so the DiT redraws the text. The
gate reads and the gate's shipping (the P0 follow-ups) are parked until this is understood.

The two tasks want opposite things from the text:

| task | what should happen to the text | why the strokes go soft | lever |
|---|---|---|---|
| colorize | nothing: the glyphs stay as drawn | the page is redrawn, glyphs with it, instead of copied | copy: G inside the text boxes, or C |
| swap_text | redrawn in another language | drawing quality of L's part, learned from edition twins | draw: L on crisp lettering data |

## Steps

**0. Read the strokes per glyph (no GPU).** Crops of every Korean line in the existing
renders against the source (colorize) or the target edition (swap_text), each glyph
labelled: right and crisp / right and soft / wrong syllable. Renders:
`output/moss_i2i_eval/swap_text/results/v3_en400_ko1709_ksplit_e2_ema_bf16_fp32s[_s1,_s2]`
and `ksplit_e3_ema[_s1,_s2]`; `output/moss_i2i_eval/colorize/results/{v5_eval,v5_holdout_comic_s1}`.
A stroke number to go with the sheets: the ink-edge gradient magnitude per glyph against
the same glyph in the source / target crop. Writes `reports/ko_strokes.md`. Decides which
task goes first and gives later arms their read.

**1. colorize: the copy inside the text boxes. Done, negative**
([reports/colorize_inboost.md](reports/colorize_inboost.md)). `inboost` on pages 03 / 05 /
09 / 14 of `input/to_colorize4` makes the strokes heavier and leaves every deformation in
place (the stray dot in `생` on page 05 under every arm). The gate is the wrong tool for
the strokes; steps 2 and 3 are the ones that can change them. A non-DiT bypass for colorize
alone: composite the source's text pixels back into the OCR boxes after decode (the text
should not change), at the cost of coloured lettering.

**2. P1: train C. Done** ([reports/p1_c_fill.md](reports/p1_c_fill.md)). 400 steps
(16 min) on clean illustrations with random text-layout holes under `ref_gate cut`. C + `cut`
on AnimeText boxes removes the sanitize eval pages' stamps and watermark on every seed with
no sanitize pair; the copy head forms by step 200. Outside the hole it darkens one page
(no tone term). `cut` replaces `diag` as the training and removal gate.

**3. A Hangul block: one LoRA part that writes Korean well, reused by every task. v1 data negative** ([reports/hangul_s500.md](reports/hangul_s500.md)), **v2 data works** ([reports/hangul_v2_s500.md](reports/hangul_v2_s500.md)): right syllables, bold clean strokes, uncaptioned text kept; e2 is already crisp on the eval pages, so the soft-page read (step 0) is still owed. L as
proposed is swap_text's drawing; the more useful form is a block for Korean lettering on
its own, which swap_text, colorize (where a glyph has to be redrawn), uncensor / sanitize
pages with text, and t2i would all stack or init from.

- Data: synthetic, so every target stroke is crisp. Korean lines rendered with the fonts
  in `project/cjk_anima_reseed/assets/fonts/` (`ko_extra/`) into blank balloons, caption
  boxes and signs on real pages, at manga glyph sizes. Pair = (the page with the region
  blank or holding other text, the page with the rendered line); the caption is the
  layout item with the string and its bbox, as swap_text's. The edition-twin pages keep a
  share so the style stays the manga's.
- Tokenization as swap_text ksplit (`token_split = "syllable"`): the syllables have to
  reach the glyphs they were trained on, or the block learns them twice.
- Trained from C with `boost` outside the boxes, so its updates go to the strokes and not
  to the copy.
- Reads: step 0's per-glyph labels and stroke number; wrong syllables; held-out syllables
  (the block should draw ones its data did not render, as the Anima jamo line measured
  with its H set, `project/cjk_ko/findings.md`).
- Gate as a block: stacked on swap_text (or as its concat init, against the en50_ko150
  micro from kaiming at equal steps), softer → crisp on the eval pages with no more wrong
  syllables; stacked on colorize v5 on the to_colorize4 pages, strokes no softer than step
  1's best arm. Run length is the user's call before it is queued.

**4. P2: sanitize assembled.** The zero-shot half is in the P1 report. Left: a short
fine-tune from C on 50 sanitize pairs (and on all 277) against the same from kaiming, for
the tone and the marks the mask misses.

## Parked

- P0 follow-ups: own-cell boost, σ schedules, `boost` on sanitize, cut + boost.
- Shipping G: port to `generate.py`, the Reference node mask input.
- K (colorize's drawing block), D in the DiT, instruction assembly.
- The induction-style upstream ablation ([findings/copy.md](findings/copy.md)): changes no
  decision yet.
