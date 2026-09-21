# Exposure ledger — what a draw buys, by item type (2026-09-21)

> Every cold, single-glyph rows arm on record with a countable exposure, read
> two ways: **identity** (the flat one-bubble `single` eval) and **native**
> (scene prompt + clause, あ か す 日, both readers, of 64). Three item types:
> **flat** (font glyph in a white bubble — the `single` eval's own template),
> **scene** (composite into a generated scene's bubble), **grid** (k units in
> k cells, one position clause each).
>
> **Identity is bought by any draw that shows the glyph large; native needs
> scene draws, but a starved row takes most of its first native from the grid
> too.** In scene-draw equivalents a grid *cell* draw averages **≈ 0.4 for
> identity, ≈ 0.35 for native `en`, ≈ 0.55 for native `swap`** over the seven
> mix / grid arms (§ 2) — high (0.5–0.9) while the row has < 50 scene draws,
> low (0.04–0.13 on native) once it is at the knee. The 374-row gate arms
> (6 k steps, grid 50 %) read `step1_0920`'s identity and native at a ninth of
> its steps. A flat draw is
> worth the most for the flat ruler and nothing for native on its own (2/64),
> but a 10 % flat share is what holds `swap` and frame independence. The curve
> is per row and holds from 24 to 252 units; at 374–433 rows the same draws buy
> less, and the loss (ΔFM vs plain) moves the full-table numbers more than any
> item type does.

Exposure here = draws of an item that contains the row: `steps × batch ×
(items with the row) / (items in the dir)` — the sampler is an epoch over
items, so this is exact up to one epoch's remainder. A grid item pays every
cell's row (6.3 cell draws per item on the even mix).

## 1. This session's arms (raw pack, plain FM, lr 1e-3 cosine, σ 0.7–0.9, `--box_share 0.25`)

Median over the 18 eval rows; native exposure is the four native rows' own.

| arm | units | steps | scene draws / row | grid cell draws / row | `single` exact | glyph in any read | native `en` / `swap` | en cos |
|---|---|---|---|---|---|---|---|---|
| `single_m24_a500` | 24 | 500 | 82 | 0 | 6/36 (0.17) | 9 | — | — |
| `single_m24_a750` | 24 | 750 | 123 | 0 | 12/36 (0.33) | 14 | 17 / 10 | 0.918 |
| `single_m24_a1500` | 24 | 1 500 | 246 | 0 | 28/36 (0.78) | 30 | 49 / 29 | 0.891 |
| `single_m24_a3000` | 24 | 3 000 | 492 | 0 | 29/36 (0.81) | 34 | 44 / 32 | 0.889 |
| `grid_m24_g500` | 24 | 500 | 0 | 528 | 14/36 (0.39) | 22 | 13 / 3 | 0.931 |
| `grid_m24_g1500` | 24 | 1 500 | 0 | 1 584 | 17/36 (0.47) | 22 | — | — |
| `mix_m24_m1500` (50 %) | 24 | 1 500 | 122 | 792 | 27/36 (0.75) | 36 | 25 / 24 | 0.925 |
| `mix25_m24_m1500` (25 %) | 24 | 1 500 | 183 | 398 | 30/36 (0.83) | 33 | 37 / 27 | 0.904 |
| `single_k174_a3000` | 252 | 3 000 | 66 | 0 | 5/36 (0.14) | 8 | 10 / 1 of 48¹ | 0.936 |
| `mix_k174_m3000` (50 %) | 252 | 3 000 | 39 | 147 | 13/36 (0.36) | 19 | 13 / 11 of 48¹ | 0.928 |
| `s1_k174_w3000` (word cells) | 252 | 3 000 | 39 | 75 (+ 206 in words) | 7/36 (0.19) | 16 | 12 / 3 of 48¹ | 0.931 |

¹ the kana block has no 日 row, so 16 of each clause's 64 prompts cannot hit.
(`grid_m0b_s1_2026_09_21.md` printed the `hit vl` column, 11, for the first
arm; *both readers* is 10.)

**The 374-row gate arms** (2026-09-21; 452 pool units, plain 1e-3, 6 000
steps, 5 000 scene + 5 000 grid items, the deck dealt by pool weight; jobs
`20260921-071912-{2ae353,aed21a}`, natives `…-{1593e9,a26cd2}`, in-frame
`…-{cd8ecd,0cca1b}`):

| arm | grid mix | scene / cell draws per row | `single` / ext / small / kanji (of 36) | `single` seed 0 / 1 | native `en` / `swap` | in-frame found k = 4 (flat · bubble) |
|---|---|---|---|---|---|---|
| `rows_g1_even_p6k` | `2x2,3x3,2x3,3x2` (6.28 cells / item) | 32 / 202 | **20 / 12 / 3 / 2** | 14 / 6 | 20 / 25 | 0.56 · 0.59 |
| `rows_g1_dense_p6k` | `2x2:1,3x3:2,2x3:2,3x2:2` (6.58) | 32 / 211 | 15 / 10 / 1 / 2 | 12 / 3 | 25 / 20 | 0.66 · 0.59 |
| `rows_g1_even_pair_d6k` (scene ΔFM + grid plain, 2e-3) | even | 32 / 202 | 18 / 10 / 0 / 4 | 13 / 5 | 17 / 11 | 0.50 · 0.44 |
| `rows_g1_even75_p4k4` (75 % grid, 4 400 steps) | even | **12 / 222** | 12 / 12 / 0 / 3 | 10 / 2 | **5 / 4** | 0.62 · 0.59 |
| `step1_0920` (53 k, ΔFM, scene-only) | — | 568 / 0 | 20 / 8 / 0 / 8 | — | 19 / 8 | — |

Even is taken: identity 37 against 28 over the four groups, native 45 against
45, and dense is the worse one on seed 1 (3/18). Kana identity and native are
at `step1_0920`'s level with 32 scene draws per row; **kanji are not (2/36 in
both, reads of 3–4 glyphs)** — 200 of the 374 rows, and the grid mix does not
move them. Per native row (`en` / `swap` of 16): あ 11 / 12, か 0 / 0, す 6 / 4,
日 3 / 9 (even) — か has the fewest scene draws (22) and reads nothing.

**Cell draws are not interchangeable with scene draws** (the 75 % arm, full
read in `grid_g1_2026_09_21.md`): the same 234 draws per row as the 50 % arm,
20 fewer of them scene draws, and `single` falls 20 → 12, native 45 → 9 of 128
while the in-frame `found` is the best of the four. The rates of § 2 are
therefore rates *given* the row's scene draws — a cell draw multiplies what the
scene draws carry out of the grid frame; it does not stand in for them. § 5's
per-step arithmetic holds at a fixed share, not across shares.

**The scene-only identity curve, per row** (all scene-only rows of both
tables pooled by their own exposure): 60–100 draws → 9/68 (0.13), 100–160 →
12/36 (0.33), 160–300 → 28/36 (0.78), 300–1 000 → 29/36 (0.81). The 252-unit
table sits on the 24-row curve (66 → 0.14 against 82 → 0.17): **up to ≈ 250
units the curve is per row, not per table.** The knee is 125 → 250 draws; above
250 this ruler is ceilinged (≈ 0.8 — the "micro arms are ceilinged" rule).

**Native per row** (both readers, of 16 per clause):

| row | A750 scene ≈ 130 | A1500 ≈ 260 | A3000 ≈ 520 | grid-only 528 cells | mix 50 % ≈ 125 + 790 | mix 25 % ≈ 190 + 397 |
|---|---|---|---|---|---|---|
| あ `en` / `swap` | 9 / 9 | 15 / 15 | 15 / 13 | 5 / 1 | 10 / 12 | 9 / 12 |
| か | 1 / 0 | 13 / 2 | 7 / 6 | 1 / 0 | 4 / 3 | 11 / 3 |
| す | 6 / 1 | 15 / 7 | 15 / 9 | 5 / 0 | 10 / 5 | 13 / 6 |
| 日 | 1 / 0 | 6 / 5 | 7 / 4 | 2 / 2 | 1 / 4 | 4 / 6 |

Native has the same knee (130 → 260 scene draws: 17 → 49) and no gain from
260 → 520 (49 → 44; か falls 13 → 7). Rows differ more than arms do: あ reads
at 130 draws, か and 日 need 260 and 日 never passes 7/16.

## 2. What a grid cell draw is worth, in scene draws

Read each mix off the scene-only curve (linear between the 123- and 246-draw
points) and divide the excess by the grid draws.

| arm | ruler | reads like scene draws | its scene draws | excess from grid | per cell draw |
|---|---|---|---|---|---|
| mix k174 (39 + 147) | identity 0.36 | ≈ 131 | 39 | ≈ 92 | **≈ 0.6** |
| grid-only g500 (0 + 528) | identity 0.39 | ≈ 139 | 0 | ≈ 139 | ≈ 0.26 |
| grid-only g1500 (0 + 1 584) | identity 0.47 | ≈ 161 | 0 | ≈ 161 | ≈ 0.10 (marginal 0.02) |
| mix 50 % m24 (122 + 792) | identity 0.75 | ≈ 238 | 122 | ≈ 116 | ≥ 0.15 (ceilinged) |
| mix 25 % m24 (183 + 398) | identity 0.83 | ≥ 260 | 183 | ≥ 77 | ≥ 0.19 (ceilinged) |
| mix 50 % m24 | native `en` 25 | ≈ 154 | 122 | ≈ 32 | **≈ 0.04** |
| mix 25 % m24 | native `en` 37 | ≈ 200 | 183 | ≈ 17 | ≈ 0.04 |
| mix 50 % m24 | native `swap` 24 | ≈ 214 | 122 | ≈ 92 | **≈ 0.12** |
| mix 25 % m24 | native `swap` 27 | ≈ 233 | 183 | ≈ 50 | ≈ 0.13 |
| g1 even (32 + 202), 374 rows | identity 0.56 | ≈ 186 | 29 | ≈ 157 | ≈ 0.78 |
| g1 dense (32 + 211) | identity 0.42 | ≈ 148 | 29 | ≈ 119 | ≈ 0.56 |
| g1 even / dense | native `en` 20 / 25 | ≈ 135 / 154 | 33 | ≈ 102 / 121 | ≈ 0.50 / 0.57 |
| g1 even / dense | native `swap` 25 / 20 | ≈ 220 / 188 | 33 | ≈ 187 / 155 | ≈ 0.92 / 0.73 |
| mix k174 (native rows 33 + 147; of 48 → × 64/48) | native `en` 13 / `swap` 11 | ≈ 124 / 153 | 33 | ≈ 91 / 120 | ≈ 0.6 / 0.8 |

**Planning rates — the plain mean of the estimates above** (user, 2026-09-21:
one number, not a range): identity **≈ 0.4** (0.6, 0.26, 0.10, ≥ 0.15, ≥ 0.19,
0.78, 0.56), native `en` **≈ 0.35** (0.04, 0.04, 0.50, 0.57, 0.6), native
`swap` **≈ 0.55** (0.12, 0.13, 0.92, 0.73, 0.8). The spread is one variable:
rows with < 50 scene draws and ≈ 150–200 cell draws read 0.5–0.9 on every
ruler, rows at the knee (120–190 scene draws, 400–800 cells) read ≥ 0.15 on
identity and 0.04–0.13 on native — a grid draw is worth most where the row has
nothing else. The equivalents of the 374-row arms are read off the small-table
curve, which flatters them if anything (§ 4: the same draws buy less there).

- **Identity:** ≈ 0.4 on average, 0.6–0.8 where nothing is ceilinged. Per
  *item* (6.3 cells) that is ≈ 2.5 scene items of identity for one grid item.
  The m24 mixes only give a lower bound — both sit on the ruler's ceiling.
- **Grid alone saturates**: 528 → 1 584 cell draws moved identity 0.39 → 0.47
  and glyph-in-read not at all, while the same table renders 94–100 % of its
  units in its own frame (in-frame read). With no scene draw the cap is
  transfer to a single bubble, ≈ 140–160 scene-draw equivalents, not learning.
- **Native `en`:** 0.04 at the knee, 0.5–0.6 on starved rows (mean 0.35).
  **Native `swap`:** 0.12 / 0.7–0.9 (mean 0.55), above `en` in every arm.
  `swap` is the clause the grid
  caption does not use (`English text reads as`), i.e. the grid pays
  frame-independent identity more than it pays the trained clause in a scene.
- One run per point, two seeds, 36 / 64 prompts: a ±4 read moves these by
  a third. The ordering (identity ≫ `swap` > `en`) is the result, not the
  decimals.

## 3. Flat draws (history, 2026-09-14 → 09-16; raw pack, plain FM, 6 rows, 1 330 draws per row)

The only clean flat-share series — same rows, same total draws:

| flat share | flat / scene draws per row | identity (of 12) | native `en` | hit & scene kept | native `swap` | en cos |
|---|---|---|---|---|---|---|
| 60 % (`m6 0.4`) | 800 / 530 | 11–12 | 57 | 23 | 10 | 0.797 |
| 10 % (`m6c9`) | 133 / 1 200 | 12 | **60** | **41** | **23** | 0.860 |
| 0 % (`m6c10`) | 0 / 1 330 | 10 | 46 | — | 5 | 0.856 |
| 100 % (P0b, 367 rows, 234 draws — encoder hybrid) | 234 / 0 | 36/36 | hit 32, kept 28 | **2/64** | — | — |

- **A flat draw is the cheapest identity on the flat ruler** (36/36 at 234
  draws per row on a full table — the eval prompt is the training template
  verbatim) and buys **no native**: the glyph arrives with its white canvas
  (hit & kept 2/64; every flat donor transplants at 0/64, `findings.md`).
- **A small flat share is not inert.** 10 % → 0 % at equal total draws cost
  identity 12 → 10, native `en` 60 → 46 and `swap` 23 → 5 (日 rendered as Latin
  "a"). The report's reading: flat holds identity and *frame independence*.
  Above 10 % it trades scene for canvas (`en` cos 0.86 → 0.80, hit & kept
  41 → 23).
- The grid is the same kind of item — mechanical, large glyph, no scene — and
  lands in the same place: strong on identity and `swap`, weak on `en`. What
  it adds over flat is k rows per item and no white-bubble canvas to absorb.

## 4. The full-table rows (where the per-row curve stops holding)

| arm | rows | flat / scene draws per row | loss | identity `single` / ext / kanji (of 36) | native `en` / `swap` |
|---|---|---|---|---|---|
| `full_s53k_qoff` (09-16; 7.5 % of items were words) | 433 | ≈ 49 / 441 | plain 1e-3 | 13 / 18 / 18 | 36 / 18 |
| `step1_0919` | 374 | 0 / 568 | ΔFM 2e-3 | 10 / 5 / 3 | 8 / 4 |
| `step1_0920` | 374 | 0 / 568 | ΔFM 2e-3 + `--box_share 0.25` | 20 / 8 / 8 | 19 / 8 |
| 12-row raw controls at 500 scene draws (09-19/20) | 12 | 0 / 500 | plain → 7/24; plain + box_share → 10/24; ΔFM + box_share → 15/24 | | plain + box_share `en` 26/64 |

- At ≈ 500 scene draws per row the 24-row table reads 0.81 and the 433-row
  table 0.36: past ≈ 250 units the same draws buy less (the old 1 330 / 670 /
  490 → 100 / 75 / 36 % curve is this, on 10 % flat + 90 % scene items, read
  on three different rulers).
- The row *set* matters as much: the 12-row set (dakuten rows, 500 draws,
  plain + box_share) reads 10/24 where m24's basic kana read 28/36 at 246.
  Rulers are comparable only within a row set.
- The loss moves the full table more than the item type: plain 36 / 18 native
  against ΔFM 8 / 4 → 19 / 8 at more draws. No full-table plain run on the
  current data exists (`plan.md` S1a), so the grid gate arms (plain) have no
  full-table control but each other.

## 5. Per-step efficiency and steps per row — grid 50 % + scene

Batch 4, grid 50 % by items = 2 scene items + 2 grid items per step, 6.3 cells
per grid item (even mix), § 2's planning rates (0.4 / 0.35 / 0.55).

| per step, in scene-draw equivalents | scene-only | grid 50 % mix | ratio |
|---|---|---|---|
| raw draws | 4 scene | 2 scene + 12.6 cell | — |
| identity | 4 | 2 + 12.6 × 0.4 = **7.0** | × 1.8 |
| native `en` | 4 | 2 + 12.6 × 0.35 = **6.4** | × 1.6 |
| native `swap` | 4 | 2 + 12.6 × 0.55 = **8.9** | × 2.2 |

Steps per row (vocab entry) to clear the knee — 250 scene-draw equivalents per
row on a small table (≤ ≈ 250 units: identity 0.78, native `en` 49/64), and the
old planning number of 1 000 per row at 374 rows (§ 4):

| target | scene-only | grid 50 % mix |
|---|---|---|
| identity, small table (250) | 63 steps / row | **36** |
| native `en`, small table (250) | 63 | **39** |
| identity, 374-row table (1 000) | 250 steps / row = 94 k | **143** = 53 k |
| native `en`, 374-row table (1 000) | 250 = 94 k | **156** = 58 k |

Two budgets read off it:

- **80 k steps, 374 rows, grid 50 %:** 428 scene + ≈ 2 700 cell draws per row =
  identity ≈ 1 500 equivalents (2.7 × `step1_0920`'s 568), native `en` ≈ 1 370,
  `swap` ≈ 1 900. The rates were measured over a row's first 150–800 cell
  draws and grid-only was flat past ≈ 500, so past ≈ 800 cells these are an
  extrapolation — the floor of the same budget is its scene draws alone, 428.
- **Raising the share does not buy the steps back.** 75 % pays 19.8 draws per
  step against 14.6, but at matched total exposure it read `single` 12 against
  20 and native 9 against 45: a run is as short as its scene budget allows.
- **The 6 k gate arms:** 32 scene + ≈ 200 cell draws per row ≈ 113 identity
  equivalents by the mean rate; they read ≈ 150–190 (`single` 15–20/36), i.e.
  on the starved-row rates, as § 2 says they should.

## 6. What it says about the run that replaces step 1

1. **Scene draws still set the native ceiling; the grid sets how fast a row
   gets off the floor.** At the knee a cell draw adds 0.04 of a scene draw of
   `en`; on a starved row 0.5. か, with 22 scene draws, reads 0/32 native in
   both gate arms — the grid does not replace the row's own scene draws.
2. **The grid is de-risked at full inventory.** Six thousand steps at grid
   50 % read `step1_0920`'s kana identity (20 / 12 vs 20 / 8) and more than its
   native (20 / 25 vs 19 / 8) — nine times fewer steps, plain loss. What it
   has not shown is kanji (2/36) and anything past ≈ 200 cell draws per row at
   this scale; grid-only was flat past ≈ 500 at 24 rows, and an 80 k run puts
   ≈ 2 700 on every row. The share question (25 / 50 / 75 %) is therefore a
   question about the *late* run, which a 6 k arm only partly reads.
3. **A flat share was dropped on 2026-09-19 without an A/B on this recipe.**
   The one series that varied it says 10 % beats 0 % on identity, `en` and
   `swap`. The grid's flat frame looks like it supplies that now: the gate
   arms read `swap` 25 / 20 against `step1_0920`'s 8, so a flat share is not
   owed unless a scene-heavy share loses `swap` again.
4. Rows are not equal: か and 日 need ≈ 2 × あ's draws for native. A per-row
   budget (the `--row_boost` sampler) has a measurable target here.
