# length proposal — the line as long as the word (2026-10-01, draft)

The live open question of the line. It takes over `plan_retrain.md` § 4
"Doubling" (archived 2026-10-01): `retrain_kana`'s `dup` rose over `p1_mix`'s
(42 vs 31 / 64), and the seed still doubles on the `sent` grid (dup
100 / 184, `かかんがが`, `こんにちちは`).

## Where the line got to

**The concept: attack the base where it is confused, in its own layout.**
Asked for Japanese, the base already draws a text region and fills it with
pseudo-Japanese (`reports/polish_seed_2026_09_30.md`, `sigma_split`: bubbles with vertical lines,
a subtitle bar, a banner). The rows do not have to invent a layout; they
have to make the base's own text region say the word. `garble_replace`
built that literally — the base's garble erased and replaced by a real line
at its px — and it moved the layout the way no fill-rule composite did
(box IoU vs EN ref 0.20 → 0.33), while the strings did not read.

What the garble arms and their follow-ups settled
(`reports/garble_replace_2026_09_30.md`, `reports/delta_scale_2026_10_01.md`,
`reports/sigma_split_2026_09_30.md`):

- **The repeats are leftover slots.** The text region's span is set by σ 0.95
  with or without trained rows; the glyph count inside it at σ ≈ 0.9. The word
  is written into those slots and the slots left over repeat its glyphs
  (traj: warm fills the seed's span with smaller glyphs, `こんにちはは`). A
  weaker Δ lengthens the line in the same banner (Δ 0.9: dup 100 → 124).
- **Order information is not the limit.** The slot share of the adapter output
  falls with the reads across arms, but restoring it (Δ 0.75: slot variance
  2×) does not bring the reads back; identity falls first.
- **The count is decided where only large glyphs resolve.** Garble glyphs are
  ≈ 14 px: below σ 0.85 the rows learn small glyphs (more slots); at 0.8–0.95
  they learn the canvas's bubbles and no string (`windows.py` C.2: 0.8–0.95
  is dead even at 48 px).

Closed by these reads: a Δ scale or row cap; garble items at the base's px
at any band; an order-paired loss on routed singles (concatenated single
rows carry no order leverage at any σ —
`../finished/cjk_renderable_anima/findings.md` § Settled); ΔFM (plain FM's
target with less variance — same findings, closed 2026-09-18).

## Proposal: shorten the span, where it is set

Repeats = span − word. Two levers follow, at two σ:

- **Span** (σ ≥ 0.9): the text region's width at σ 0.95. The rows already act
  on it there — the seed's large banner vs the raw pack's small one is the
  rows' doing above the switch (`sigma_split` `hi` / `lo`), and `short50hb`
  at 0.8–0.95 learned its canvases' bubbles. What no row has learned is a
  span of the word's length: the seed's word items (`scene_window`, the text
  filling 0.7–1.0 of a bubble at ≈ 40 px) train at 0.5–0.7 and 0.3–0.5
  (`builder.TABLE` b0507 / b0305), below where the span is set, and the
  garble canvases were sentence-span by construction. 0.8–0.95 is dead for
  glyph identity at 48 px (`windows.py` C.2); the span is layout, which is
  the one thing that did train there.
- **Count** (σ 0.8–0.9): a counterfactual pair — A = the word once,
  banner-size, filling the region; B = the same span with one slot more, a
  glyph of the word repeated (`こんにちはは`); input B noised, caption A,
  target toward A (`../finished/cjk_renderable_anima/idea.md`'s CF). Two
  caveats the record puts on it: the CF argument was made for σ ≤ 0.7, where
  the input decides and the residual `(x0_A − x0_B)/σ` grows as σ falls — at
  0.85 the correction term is 0.18 · (x0_B − x0_A) on an input that is 15 %
  B; and no leverage read covers the pair — Gate 0's 0.197 / 0.155 at σ 0.8
  is single glyphs, its strings were read at ≤ 48 px (dead at 0.8–0.9), and
  the ceiling table has no cell for strings above 48 px
  (`band_experiment_results.md` § 2).

Span first: it is the lever at the σ where the quantity is set, its item is
already in the recipe table, and it is one band change. Count is the route
if the span does not move.

**Step 0 (no training, minutes):** `sigma_split --traj` on the `sent` grid
with the seed rows, x̂0 at σ 0.95 / 0.9 / 0.85, the same (prompt, seed) under
three captions — the word, the word with one glyph repeated, a 2–3-glyph
word — reading `box` / `box h` of the text region per σ. The span tracks the
caption's glyph count at 0.95 → the caption reaches the span, step 1 trains
it. The span is the same for 2, 5 and 6 glyphs → it is the base's prior on
`japanese text`, and the count route is read on the same run's x_t at 0.85:
caption A vs B, `move` on the region as `cf_sense` reads it — ≥ 0.1 opens
it, ≈ 0 and the line sits at the base's span.

**Step 1 (≈ 25 min):** one micro arm on the 57 `sent` singles, warm — the
seed's own item mix plus its `scene_window` items rebanded to 0.85–0.95
(`experiments/garble_replace`'s `reband` leg, pointed at the seed run's
items — a source-path knob the leg lacks today; past `windows.SIGMA_MAX`
0.9, as `short50hb` was). Read on `sent` (dup, glyphs per read) and the traj
leg (`sigma_split --traj --rows`): span at σ 0.95 and glyph count at 0.9
against the seed's.

## Result (2026-10-01): step 1 ran, the span lever is closed

`experiments/span_reband` (the b0507 windows at 0.85–0.95, filtered to
bubbles of ≤ 2 columns, warm): the banner's span does not shorten; the rows
refill it with smaller glyphs (dup 100 → 113, official 29 → 16). The
`inject_count` read put the count's commit between σ 0.95 and 0.9 without
`cf_sense`. Both in `proposal_seed_synthesis.md`, which takes over the
question.
