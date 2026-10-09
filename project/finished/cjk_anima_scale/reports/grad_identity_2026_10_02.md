# grad_identity — how much of a cold row's gradient is glyph identity, by px × σ (2026-10-02)

`experiments/grad_identity` — a training-free read of `hypothesis.md` H1 on
the grid_44 train leg's initial point: the 81 hiragana + ー rows cold at the
pack rows, everything else at the old seed, routed, raw pack. Two passes, both
at the gradient of record (the box-share FM loss, `train.BOX_SHARE*`,
`GRID_BOX`, plain MSE on a lone 1×1), σ swept over 0.2 … 0.95 with one ε per
(item, σ).

**Answer.** Where the identity signal leaves the gradient, by px tier:

| tier | slot px | identity peak (σ of max ‖I‖) | identity share at half its plateau | share < 0.1 from | EN ceiling (§ 2: grid cell / flat letter, peak · live) | grid_44 band | share across that band | b7593 (0.75–0.93) |
|---|---|---|---|---|---|---|---|---|
| `grid_16` | 15 | 0.4 | ≈ 0.62 | 0.7 | 16 px: 0.5 · 0.4–0.6 / 0.4 · 0.2–0.6 | 0.3–0.5 | 0.31 → 0.19 | 0.08 → 0.01 |
| `grid_29` | 28 | 0.5 | ≈ 0.72 | 0.8 | 24–32 px: 0.5–0.6 · 0.4–0.7 / 0.5 · 0.35–0.7 | 0.5–0.7 | 0.32 → 0.19 | 0.12 → 0.03 |
| `grid_44` | 44 | 0.6 | ≈ 0.76 | 0.9 | 48 px: 0.8 · 0.7–0.8 / 0.6 · 0.5–0.7 | 0.7–0.9 | 0.36 → 0.07 | 0.25 → 0.07 |
| `lone_16` | 15 | flat (‖I‖ 2–5 × 1e-3) | ≈ 0.53 | 0.75 (0.11 at 0.8) | 16 px flat: 0.4 · 0.2–0.6 | 0.3–0.5 | 0.60 → 0.35 | 0.10 → 0.02 |
| `lone_28` | 28 | ≈ 0.75 (‖I‖ ≤ 8 × 1e-3) | ≈ 0.66 | 0.85 | 24–32 px flat: 0.5 · 0.35–0.7 | 0.5–0.7 | 0.69 → 0.19 | 0.16 → 0.04 |
| `lone_44` | 45 | 0.7–0.8 (‖I‖ ≤ 13 × 1e-3) | ≈ 0.77 | 0.9 | 48 / 64 px flat: 0.6 · 0.5–0.7 / 0.7 · 0.6–0.8 | 0.7–0.9 | 0.66 → 0.07 | 0.42 → 0.07 |
| `bubble1_52` (§ 5) | 50 | 0.7 | ≈ 0.79 | never (0.27 at 0.95) | 48 px: as `grid_44` | 0.7–0.9 (`TABLE`'s) | 0.63 → 0.27 | 0.58 → 0.27 |
| `bubble1_32` (§ 5) | 32 | 0.5 | ≈ 0.60 | never (0.19 at 0.95) | 24–32 px: as `grid_29` | 0.5–0.7 | 0.78 → 0.38 | 0.24 → 0.23 |
| `bubbleN_34` (§ 5) | 36 | 0.6 | ≈ 0.68 | 0.95 | — | 0.5–0.7 | 0.46 → 0.27 | 0.25 → 0.13 |
| `bubbleN_18` (§ 5) | 19 | 0.4 | ≈ 0.52 | 0.6 | — | 0.3–0.5 | 0.38 → 0.22 | 0.10 → 0.06 |

"Identity share" = f, the share of row u's gradient energy that changes when
the glyph drawn in its slot changes (pass 2, § 3); ‖I‖ = ‖g‖·√f, its size.
Ceiling px are font px, slot px are ink px (≈ 0.8 × font px for these
renders); the law keys on ink px, as here.

- **The true glyph never separates from the layout in the brief's sense.**
  In no tier, at no σ, plain or clause, is the drawn glyph's row (pass 1) or
  the drawn glyph's image (pass 2) more distinct from the rest than a wrong
  one is: no (tier, σ) cell has the excess over the null above zero at
  p < 0.01, and pooled the sign is the other way. At the cold start no row
  matches its glyph yet, so "the right one" is not a contrast the gradient
  carries.
- **What the gradient does carry is how much it depends on the glyph drawn**,
  and that is the H1 quantity. Holding the row fixed and swapping the glyph
  in the image (pass 2), f sits on a plateau at low σ and falls with σ; it
  falls **earlier the smaller the glyph**, and its half point (0.62 / 0.72 /
  0.76 for 15 / 28 / 44 px grids; 0.53 / 0.66 / 0.77 lone) **is the EN
  ceiling's upper edge for that px** (0.6 / 0.7 / 0.8). The grids' identity
  peak (0.4 / 0.5 / 0.6) is the ceiling's flat-letter peak, one step under
  its grid-cell peak.
- **There is no lower edge in f.** Below its peak the share holds, but the
  gradient's size collapses (`grid_44`: ‖g‖ 1.4 × 1e-3 at σ 0.2 vs 76 × 1e-3
  at 0.6; `grid_16` only 3× smaller at 0.2). A band's lower edge is an
  exposure edge, not an identity one.
- **Against grid_44's bands:** the 15 and 28 px tiers train at their peak,
  between plateau and half (grids f 0.31 → 0.19); the 44 px tier trains at 0.7–0.9,
  above its ‖I‖ peak, and its upper half (0.8–0.9) gives f ≤ 0.15. **b7593**
  (every item at 0.75–0.93) puts every 15–28 px tier at f ≤ 0.16 and mostly
  under 0.1 — a step-0 gradient over 90 % the same whichever glyph is drawn,
  H1's "layout alone", measured; `grid_small b7593` read 0
  (`../cjk_anima_reseed/reports/grid_small_lone_2026_10_02.md`).

- **The bubble tiers follow the same law, and a window's rows are not told
  apart** (§ 5, added the same evening). The ‖I‖ peak is ordered by px
  across forms — σ 0.4 (15–19 px), 0.5 (28–32), 0.6 (36–44), 0.7 (50) — and
  `TABLE`'s bands sit on it. In a window, a row whose own glyph did not
  change takes as large a glyph-dependent share from the swap of slot k as
  slot k's own row does (f_cross ≈ f_own at every σ): the identity signal of
  a draw is paid to every row of the window, H2 on the gradient.

## 1. Setup

| | pass 1 (`r1`) | pass 2 (`render`) |
|---|---|---|
| contrast | different rows: the true caption C vs C^{v_j} (slot k's glyph u → v_j), same image | one row (u), the true caption C, the image re-drawn with slot k's glyph u (A) or v_j (B_j) |
| items | 40 per tier × 9 tiers (360); clause: 16 per tier (144) | 40 per tier × 6 grid / lone tiers (240, none dropped) |
| captions | `data_recap_b7593` (plain) + `data` (position clause) | plain |
| m, σ | 3; 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 | same |
| job | `20261002-195428-37dfb2` (38.5 min) | `20261002-203655-1e4c1d` (18.5 min) |
| result | `experiments/grad_identity/results/20261002-1954-r1/` | `experiments/grad_identity/results/20261002-2036-render/` |

Both: `output/cjk_anima_scale/run1002_grid_44/` items (the recap dir's
latents in pass 1), rows from `Rows(warm=None, frozen=…, context=SEED_ROWS_0921)`
— the 82 rows are the only ext rows the captions carry, so all of them sit at
the pack rows; job logs `pack anima_cjk_vocab_pack raw sha 7b9fce0bb57b;
ANIMA_VOCAB_GLYPH_ROUTE=1`, set in-process. u is one of the 81 hiragana
occurring once in the item; v_j are hiragana absent from the item and from
u's dakuten / small-kana family. Every caption set is checked on its T5 ids
(wrong differs from true at exactly one position, u's row against v_j's).
Each results dir holds `plan.json`, `grads_<mode>.pt` (the 1 + m gradient
rows per item × σ), `per_item.jsonl`, `summary.json` and the full
`report.md` (per tier × σ: id2 · null · excess, direction-only, norm-only,
f / ‖I‖, dloss / ‖g‖ / cos, and per form × slot px).

**Batching (pass 1).** One batch of the 1 + m captions over the repeated
noisy latent gives every row's gradient in one backward. Checked before the
loop on 4 items × σ 0.4 / 0.8 (`verify.json`): a caption never touches
another's row (the others' rows are exactly zero); row u's gradient with the
three wrong captions replaced by three copies of C moves by rel 0.013 —
inside the run-to-run floor (same batch twice 0.018, reversed 0.017, B = 1
twice 0.018). B = 1 against B = 4 differs by rel **0.12** (median; up to 0.8
on small gradients) with matching losses to 1e-3 — bf16 kernel shape, not
batching; the loop runs one shape throughout. (`20261002-194820-45136a`
stopped at the first, stricter gate, B = 1 vs B = 4 at cos > 0.98;
`20261002-195126-3f915f` is the diagnostic that split the two; `…-194547-b86330`
died on a σ assert that did not allow the bf16 cast.)

**Per-sample gradient (pass 2).** All 1 + m samples carry row u, so a zero
`(B, D)` tensor is added at u's position after the rows; its gradient is each
sample's own. Σ_b over it equals ∂raw_u exactly on the 20 checked reads.

**Renders.** `data.grid.render_grid` with the record's units, grid, canvas,
bubble and fill and `grid_small`'s `bubble_fit` / `cell_jitter` (the grid_44
build's), one seeded rng per item: fresh twins of the records, not their
pixels. The font list is cut to the fonts covering every glyph involved
(14–15 of the set), so `pick_font` draws one font for A and every B_j. Kept
iff the pixel difference lies inside slot k's cell and every other cell's ink
box is unchanged: 240 / 240. Peeked: only the slot's glyph (and its fitted
bubble) changes.

## 2. Pass 1 — the true row against other rows in its slot (the brief's design)

| (plain, all 9 tiers) | σ 0.2 | 0.4 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| cos(g_true, g_wrong) | 0.15 | 0.17 | 0.16 | 0.16 | 0.15 | 0.15 | 0.15 |
| cos(g_wrong, g_wrong) | 0.16 | 0.16 | 0.16 | 0.16 | 0.14 | 0.14 | 0.15 |
| items with excess_dir > 0 (of 360) | 163 | 170 | 169 | 169 | 172 | 167 | 175 |

- Different rows' gradients in one slot are **near-orthogonal at every σ**
  (cos ≈ 0.15; f ≈ 0.85): a row's gradient is mostly a function of its own
  vector, whatever the image. The mean of other rows' gradients is therefore
  not "what any row standing in the slot receives" for u — the brief's layout
  term is a different row's Jacobian, and the identity / layout split does not
  exist in row space at the cold start.
- id2 vs null (the brief's quantity) is ≈ 1.1–1.4 vs 1.1–1.4 in every cell
  (clause, 16 items, up to 1.57); direction-only excess per tier, pooled over
  σ, −0.015 … +0.005 plain, −0.015 … +0.017 clause; **no (tier, σ) cell
  above zero** at p < 0.01 on any excess, plain or clause. Pooled, the true row is slightly
  *closer* to the others (plain excess_dir −0.007, 1 820 / 3 960 > 0,
  p 4e-7; clause n.s.).
- Caption leverage at the cold start is nil: dloss = L(wrong) − L(true)
  median 0.0000 in every cell, pooled 2 060 / 3 960 > 0 (p 0.012), clause
  p 0.41; the one cell at p < 0.01 of the 198 is clause `bubble1_32` σ 0.3
  (15 / 16). Plain and clause do not differ.
- The magnitude estimator's **median** excess is negative everywhere
  (−0.05 … −0.10, p to 1e-10) under exchangeability: id2's three terms share
  one denominator ‖g_true‖, null's three do not, so id2 is the more skewed
  (row gradient norms spread with log-sd 0.59). Its mean is +0.02. The
  direction-only and norm-only columns carry the read.

## 3. Pass 2 — one row, the glyph drawn swapped

f = 1 − cos(g(u | B_j), g(u | B_k)), median over items (IQR in the results
report). With g = L + I(glyph), f is I's share of the energy; the bf16 floor
is f ≈ 0.01 (B-shape numerics, rel 0.12), the run-to-run floor 2e-4.

| tier | px | σ 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grid_44` | 44 | 0.48 | 0.58 | 0.45 | 0.44 | 0.40 | 0.36 | 0.25 | 0.15 | 0.11 | 0.07 | 0.01 |
| `grid_29` | 28 | 0.42 | 0.34 | 0.36 | 0.32 | 0.33 | 0.19 | 0.12 | 0.10 | 0.08 | 0.03 | 0.01 |
| `grid_16` | 15 | 0.33 | 0.31 | 0.29 | 0.19 | 0.17 | 0.09 | 0.08 | 0.06 | 0.02 | 0.01 | 0.01 |
| `lone_44` | 45 | 0.55 | 0.71 | 0.76 | 0.85 | 0.71 | 0.66 | 0.42 | 0.30 | 0.17 | 0.07 | 0.01 |
| `lone_28` | 28 | 0.56 | 0.73 | 0.63 | 0.69 | 0.52 | 0.19 | 0.16 | 0.11 | 0.04 | 0.04 | 0.01 |
| `lone_16` | 15 | 0.64 | 0.60 | 0.44 | 0.35 | 0.23 | 0.16 | 0.10 | 0.11 | 0.04 | 0.02 | 0.01 |

‖I‖ = ‖g‖·√f and ‖g_true‖ (× 1e-3, medians):

| tier | | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grid_44` | ‖I‖ | 1.2 | 2.3 | 4.9 | 13.6 | **46.9** | 35.4 | 34.6 | 26.9 | 20.7 | 14.5 | 9.9 |
| | ‖g‖ | 1.4 | 3.1 | 6.2 | 20 | 76 | 65 | 64 | 63 | 66 | 69 | 90 |
| `grid_29` | ‖I‖ | 3.1 | 6.4 | 21.2 | **48.5** | 40.5 | 26.1 | 22.1 | 24.0 | 19.2 | 11.3 | 10.1 |
| | ‖g‖ | 4.2 | 12 | 33 | 82 | 53 | 53 | 69 | 63 | 68 | 75 | 88 |
| `grid_16` | ‖I‖ | 20.0 | 28.7 | **34.6** | 23.7 | 16.3 | 23.4 | 27.3 | 21.9 | 13.3 | 10.2 | 10.2 |
| | ‖g‖ | 25 | 58 | 68 | 54 | 46 | 81 | 73 | 93 | 104 | 92 | 115 |
| `lone_44` | ‖I‖ | 0.9 | 1.8 | 4.1 | 8.3 | 10.9 | 12.2 | 12.5 | **12.8** | 6.5 | 7.4 | 10.1 |
| `lone_28` | ‖I‖ | 1.1 | 2.9 | 4.8 | 5.5 | 5.1 | 7.2 | **8.1** | 5.9 | 4.5 | 6.5 | 7.5 |
| `lone_16` | ‖I‖ | 2.4 | 2.4 | 2.0 | 2.3 | 2.9 | **5.1** | 4.4 | 4.3 | 3.3 | 5.0 | 7.4 |

- **Monotone in σ, ordered by px.** At σ 0.7 the grids give f 0.36 / 0.19 /
  0.09 for 44 / 28 / 15 px; at 0.8, 0.15 / 0.10 / 0.06; at 0.95 every tier
  is at the bf16 floor.
- **The lone tiers** have a larger share (0.55–0.85 plateau) but a small
  gradient: their loss is the plain canvas mean (`src` font), where a 15 px
  glyph is ≈ 0.1 % of a 512² canvas; ‖g‖ stays under 36 × 1e-3 up to σ 0.9.
  ‖I‖ is 3–10× under the grids' and has no clear peak at 15 px; the 0.95
  values (‖g‖ ≈ 0.1) are the global composition, at f ≈ 0.01.
- **The drawn glyph being u buys nothing; at 28–45 px it costs a little.**
  The true render's gradient sits *closer* to the common part than a wrong
  render's: direction-only excess per tier, pooled over σ, `lone_44` −0.060
  (150 / 440 > 0, p 2e-11), `grid_44` −0.033 (p 9e-6), `lone_28` −0.033
  (p 0.002), `grid_29` −0.027 (p 3e-4), `grid_16` −0.010 (p 0.08), `lone_16`
  −0.007 (p 0.06). The base (its Qwen side, the pack rows) already expects u
  a little where the glyph is large enough, so the matching image leaves a
  smaller glyph-specific residual; at 15 px it cannot see it.
- dloss (L(B_j) − L(A) under C) is not separable from zero in any cell; the
  grids lean positive at σ ≤ 0.4 (69–72 of 120, p 0.035–0.12).

## 4. What it does not show

- **Training.** A step-0 gradient is necessary, not sufficient: Adam
  rescales per coordinate, the rows move off the pack rows within a few
  hundred steps (`rel` ≈ 1 by the end of grid_44's run), and the identity
  share of a moving row is not read here. f says what a draw *offers* the
  row, not what the run keeps.
- **Different rows start from different vectors.** That is what sinks pass
  1 (§ 2) and why pass 2 holds the row fixed; pass 2's f is per row, medians
  over 40 different u per tier.
- **Bubble tiers in pass 2**: § 5, below.
- **Clause captions in pass 2**, and any warm or trained rows.
- **The pixels of record.** Pass 2's renders are fresh twins of the records'
  specs (font list cut to the covering fonts); their slot px match the tiers'
  (15 / 28 / 44–45).
- **Numerics.** Gradients are bf16-autocast at B = 4, as in training; a
  kernel-shape change moves a row's gradient by rel 0.12, so f ≤ 0.01 is
  floor.

## 5. Pass 2 on the bubble tiers (`scene`, the same evening)

`--render --render_tiers bubble1_52 bubble1_32 bubbleN_34 bubbleN_18`, job
`20261002-223707-64728b` (18.0 min), `experiments/grad_identity/results/20261002-2237-scene/`.
Same initial point, σ grid, m = 3 and one ε per (item, σ) as § 3.

- **Items.** 40 per `bubble1` tier, 60 per `bubbleN` tier (200). The three
  grid_44 tiers from `run1002_grid_44/data_recap_b7593` (scene captions are
  the same in both data dirs, 3 280 / 3 280); `bubble1_52` is not a grid_44
  tier — its records are `retrain_kana/data`'s (`b0709` / `scene_single`,
  hiragana only), read at the grid_44 leg's initial point.
- **Renders.** `render_into_scene(ref_text=…)`: the record's scene, fill and
  orientation, one font (of the 15 covering every glyph involved) and one
  seeded rng for A and every B_j — one fit, the sibling's glyphs in place,
  pixel-identical outside the two text boxes (the renderer's assert). Fresh
  twins: twin px median 50 / 32 / 36 / 19, the records' 50 / 32 / 35 / 18.
  A window is kept iff every A / B_j difference spans ≤ 1.3 glyph pitch along
  the text axis, at slot k: 6 dropped (4 + 2). Peeked: only slot k's glyph
  changes.
- **Loss.** `train.train`'s rule — a scene item takes the box share
  (`BOX_SHARE` 0.25, log to 0.5 at 8 glyphs); the box is the union of the
  1 + m ink boxes, the same for every sample.
- **A window's other rows.** `_SlotGrad` takes several rows: the same
  backward reads every other row of the window (1–5 per item, each once in
  the caption, its own glyph unchanged). Σ_b ∂z_b = ∂raw exactly on the 44
  checked (own and cross rows).

f, median over items (the results report has the IQR — wide on the scenes,
`bubble1_52` at σ 0.8: 0.25–0.76):

| tier | px | σ 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bubble1_52` | 50 | 0.91 | 0.99 | 0.88 | 0.85 | 0.81 | 0.63 | 0.58 | 0.44 | 0.47 | 0.27 | 0.27 |
| `bubble1_32` | 32 | 0.89 | 0.94 | 0.83 | 0.78 | 0.45 | 0.38 | 0.24 | 0.28 | 0.29 | 0.23 | 0.19 |
| `bubbleN_34` | 36 | 0.58 | 0.60 | 0.58 | 0.46 | 0.39 | 0.27 | 0.25 | 0.15 | 0.13 | 0.13 | 0.06 |
| `bubbleN_18` | 19 | 0.49 | 0.38 | 0.35 | 0.22 | 0.09 | 0.08 | 0.10 | 0.09 | 0.06 | 0.12 | 0.05 |

‖I‖ and ‖g_true‖ (× 1e-3, medians, per row):

| tier | | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bubble1_52` | ‖I‖ | 4.9 | 8.3 | 21.8 | 45.8 | 104.7 | **123.4** | 91.6 | 72.6 | 58.3 | 39.9 | 27.8 |
| | ‖g‖ | 3.5 | 7.4 | 23 | 47 | 130 | 150 | 140 | 130 | 96 | 84 | 59 |
| `bubble1_32` | ‖I‖ | 13.7 | 33.7 | 115.1 | **153.8** | 119.2 | 95.0 | 84.4 | 74.6 | 62.8 | 49.8 | 33.3 |
| | ‖g‖ | 10 | 25 | 63 | 160 | 180 | 180 | 180 | 160 | 120 | 110 | 97 |
| `bubbleN_34` | ‖I‖ | 1.6 | 3.2 | 7.4 | 19.8 | **26.0** | 17.4 | 18.5 | 18.9 | 18.7 | 14.8 | 13.6 |
| | ‖g‖ | 2.0 | 4.3 | 10 | 30 | 45 | 36 | 31 | 40 | 48 | 39 | 51 |
| `bubbleN_18` | ‖I‖ | 11.3 | 15.3 | **20.8** | 14.0 | 13.1 | 13.7 | 16.3 | 22.9 | 17.2 | 13.8 | 9.7 |
| | ‖g‖ | 14 | 29 | 38 | 33 | 38 | 50 | 63 | 77 | 71 | 46 | 45 |

- **The px law holds across forms.** The ‖I‖ peak: 0.4 (`bubbleN_18`, 19 px;
  `grid_16`), 0.5 (`bubble1_32`, 32 px; `grid_29`), 0.6 (`bubbleN_34`,
  36 px; `grid_44`), 0.7 (`bubble1_52`, 50 px). `TABLE`'s bands hold the
  peak in every bubble tier (at their lower edge for the two `bubble1`).
  `bubbleN_18`'s second bump at 0.8 is ‖g‖, at f 0.09.
- **f's level is the form's, its fall the px's.** The plateau is ≈ 0.9 where
  the glyph is alone in its box (`bubble1`, as the lone tiers), ≈ 0.6 → 0.4
  in a window, 0.3–0.5 in a grid. The half point runs ≈ 0.1 σ under a grid
  cell of the same px for the small scene tiers — windows 0.52 (19 px) /
  0.68 (36 px), `bubble1_32` 0.60, against the grids' 0.62 (15 px) / 0.72
  (28 px) — and meets it at 50 px (`bubble1_52` 0.79; `grid_44` 0.76,
  `lone_44` 0.77).
- **A `bubble1` never goes glyph-blind**: f 0.19–0.27 at σ 0.95, where every
  grid / lone tier is at the floor (0.01) — a quarter of its loss sits on
  the one ink box that changed.
- **b7593 on the windows is § 3's reading again**: `bubbleN_18` at 0.75–0.93
  gives f 0.06–0.12 — the b0305 windows of `b0305_reband`, ≈ 90 %
  glyph-independent at the band that broke the layout; `bubbleN_34` 0.25 →
  0.13. At its own band `bubbleN_18` reads 0.38 → 0.22.

### A window's other rows

Slot k's glyph swapped in the image; f on slot k's own row against the mean
f over the window's other rows, and items with f_own > f_cross (of 60):

| tier | | σ 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bubbleN_34` | f_own | 0.58 | 0.60 | 0.58 | 0.46 | 0.39 | 0.27 | 0.25 | 0.15 | 0.13 | 0.13 | 0.06 |
| | f_cross | 0.52 | 0.54 | 0.56 | 0.44 | 0.34 | 0.25 | 0.26 | 0.15 | 0.14 | 0.11 | 0.07 |
| | own > cross | 35 | 36 | 35 | 35 | 39 | 35 | 35 | 31 | 32 | 30 | 23 |
| | ‖I_own‖ · ‖I_cross‖ | 1.6 · 1.5 | 3.2 · 3.3 | 7.4 · 6.1 | 19.8 · 18.0 | 26.0 · 25.3 | 17.4 · 15.2 | 18.5 · 16.0 | 18.9 · 17.2 | 18.7 · 16.5 | 14.8 · 14.1 | 13.6 · 14.9 |
| `bubbleN_18` | f_own | 0.49 | 0.38 | 0.35 | 0.22 | 0.09 | 0.08 | 0.10 | 0.09 | 0.06 | 0.12 | 0.05 |
| | f_cross | 0.42 | 0.39 | 0.32 | 0.20 | 0.10 | 0.08 | 0.10 | 0.10 | 0.07 | 0.10 | 0.05 |
| | own > cross | 38 | 38 | 40 | 32 | 31 | 30 | 28 | 29 | 20 | 29 | 25 |
| | ‖I_own‖ · ‖I_cross‖ | 11.3 · 13.2 | 15.3 · 16.6 | 20.8 · 20.1 | 14.0 · 14.8 | 13.1 · 14.6 | 13.7 · 15.6 | 16.3 · 18.7 | 22.9 · 21.5 | 17.2 · 21.6 | 13.8 · 15.8 | 9.7 · 10.4 |

- **The swap of slot k reaches every row of the window as much as slot k's
  own.** f_cross tracks f_own over the whole σ range and ‖I_cross‖ ≈
  ‖I_own‖; no (tier, σ) cell separates at p < 0.01. What the gradient says
  about the glyph drawn in a slot is not addressed to that slot's row —
  `hypothesis.md` H2 ("nothing ties `a`'s gradient to the place `a` is
  drawn"), read on the cold-start gradient.
- **The own row's edge is small and only where the glyph resolves.** Per
  item, mean over σ ≤ 0.5: `bubbleN_18` f_own − f_cross median +0.021,
  46 / 60 > 0 (p 4e-5); `bubbleN_34` +0.009, 37 / 60 (p 0.09). Above that
  `bubbleN_18` has none (0.6–0.75: 30 / 60; ≥ 0.8: 22 / 60) and
  `bubbleN_34` stays where it was (37 / 60; ≥ 0.8: 28 / 60).
- **Rows next to slot k take a little more than rows further off** at low σ
  (`bubbleN_18` σ 0.2–0.4: 0.44 / 0.40 / 0.33 against 0.37 / 0.31 / 0.27;
  `bubbleN_34` σ 0.3–0.4: 0.60 / 0.58 against 0.53 / 0.53, within 0.04
  elsewhere).
- **It is a shared amount, not a shared vector.** The change of the own
  row's gradient between two renders and a cross row's over the same two:
  cos ≈ 0.09 at every σ, the same as the two rows' gradients themselves
  (0.09) and as two cross rows' changes (0.08). Each row takes the swap
  through its own Jacobian (§ 2) — `shared_dir`'s "the layout break is in
  the per-row residual", from the gradient side.

### Per draw, by form

‖I‖ at its peak (× 1e-3, per row; § 3 for the grid / lone tiers):

| px | `bubble1` | `grid` | `bubbleN` | `lone` |
|---|---|---|---|---|
| 15–19 | — | 34.6 (σ 0.4) | 20.8 (0.4) | ≤ 5.1 |
| 28–36 | 153.8 (0.5) | 48.5 (0.5) | 26.0 (0.6) | 8.1 |
| 44–50 | 123.4 (0.7) | 46.9 (0.6) | — | 12.8 |

- A lone glyph that takes the box share (`bubble1`) is paid ≈ 3× a grid
  cell and 10–20× a lone 1×1 under the plain canvas loss, at the same px —
  the price of § Open's "should a lone 1×1 take the box share", with a
  scene around the glyph instead of a blank canvas.
- A window row is paid about half a grid cell's, and by the table above not
  for its own slot.
- By window length (per row, at the peak): `bubbleN_18` 2–3 glyphs 42–45
  against 4–6 glyphs 10–14 (21 / 39 items); `bubbleN_34` 45 against 24
  (13 / 47). Fewer rows, more each.
- Lines against columns (`bubbleN_34`, 20 / 40 items): the lines' ‖I‖ peaks
  at 0.7 (39.9; 32.0 at 0.8) where the columns' is at 0.5–0.6 (21–22) and
  12–14 by 0.7–0.8; f at 0.8–0.85 is 0.24 / 0.22 against 0.13 / 0.10. Lines
  live on the `sl1w` pool only, so pool and orientation are one variable
  here.

Matching and leverage, pooled over σ:

- The true render again sits closer to the common part than a wrong one
  (§ 3): direction-only excess `bubble1_52` −0.053 (180 / 440 > 0, p 2e-4),
  `bubbleN_18` −0.031 (267 / 660, p 1e-6), `bubble1_32` −0.030 (p 0.009),
  `bubbleN_34` −0.012 (n.s.).
- dloss (L(B_j) − L(A) under C) leans positive on the windows —
  `bubbleN_18` 385 / 660 > 0 (p 2e-5), `bubbleN_34` 368 / 660 (p 0.004),
  cell medians mostly +0.2 … +0.6 × 1e-3; one cell at p < 0.01 (`bubbleN_18` σ 0.2,
  42 / 60) — and not on `bubble1` (217, 221 / 440).

Not shown here: twins, not the records' pixels (and `bubble1_52`'s are
another run's records); the by-length and by-orientation cells are the
tiers' own mix, not balanced draws; trained rows.

## Open

- Pass 2 on trained rows (`experiments/grid_44_cold_hira_recap_b7593`,
  `retrain_kana`): where training bought identity, does the true render's
  gradient separate from the wrong renders' (the sign of § 3's matching
  excess flipping), and at which σ?
- The window read on trained rows: does f_own pull away from f_cross once
  the rows carry identity (`retrain_kana`, where windows composed), or does
  a window keep paying every row for every slot?
- A band chosen by ‖I‖ rather than by the ceiling: 0.3–0.5 (15 px), 0.4–0.6
  (28 px), 0.5–0.7 (44 px grid) are the windows around the per-draw identity
  peak; grid_44 trains the 44 px tier one step above it.
- The lone tiers' plain canvas loss gives them 3–10× less identity per draw
  than a grid cell of the same px; whether a lone 1×1 should take the box
  share (`GRID_BOX` covers `src` grid only) is a trainer question this read
  only prices.
