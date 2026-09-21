# plan_grid — k units per canvas inside step 1 (2026-09-20, state of 2026-09-21)

Index: [`README.md`](README.md). The live plan is [`plan.md`](plan.md); this
file is one change to its *Step 1* data, and it replaces
[`plan_step1.md`](plan_step1.md) (dropped when its boost gate read ≈ 0).
Measurements: [`reports/position_probe_2026_09_20.md`](reports/position_probe_2026_09_20.md),
[`reports/grid_s0_2026_09_20.md`](reports/grid_s0_2026_09_20.md),
[`reports/grid_m0_2026_09_21.md`](reports/grid_m0_2026_09_21.md),
[`reports/grid_m0b_s1_2026_09_21.md`](reports/grid_m0b_s1_2026_09_21.md),
[`reports/exposure_ledger_2026_09_21.md`](reports/exposure_ledger_2026_09_21.md)
(what a draw buys by item type),
[`reports/grid_g1_2026_09_21.md`](reports/grid_g1_2026_09_21.md) (the 374-row
gate arms).

> Every launch states its pack:
> `ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack` (raw).

## State

**The single-cell grid is de-risked; the run that replaces `step1_0920` is
next.** A mix of scene singles and grid items (never grid alone) reads
`step1_0920`'s kana identity and more than its native at a ninth of its steps.
Word cells (S1) did not pass cold and wait on a warm-started pair. The gate
arms settled the run's recipe — **even grids, 50 %, plain** — and left two
things open: kanji (2–4/36 at 6 k in every arm) and the quality of native
(plain wipes the scene on seed 0, paired repeats the glyph).

| gate | table | read | verdict |
|---|---|---|---|
| position probe | base model, EN | clause → cell lift +0.6–0.7 at k = 4, clause beats caption order | the address exists untrained |
| S0 | 24 rows, grid only | G500 14/36 vs singles 6/36 at equal steps; G1500 17/36, native 13 vs singles 29/36, 44 | a grid draw teaches identity; grid alone is not a seed table |
| in-frame read | S0 / M0 tables | the grid table renders 94–100 % of its units in its own frame | S0's gap was transfer to a single bubble, not learning |
| M0 | 24 rows, 1 500 steps | 50 % mix 27/36, native 25 / 24 vs A750 (equal scene draws) 12/36, 17 / 10 and A1500 28/36, 49 / 29; 25 % mix 30/36, 37 / 27 | the grid adds; native is paid in scene draws; a two-glyph answer to a one-kana caption survives on seed 1 |
| M0b | 252 units, 3 000 steps | singles 5/36, native 10 / 1; 50 % mix **13/36**, 13 / **11** | pass — the grid matters where singles are starved |
| S1 (cold) | 252 units, word cells in half the grid items | held piece hit +0.105 [+0.034, +0.178] (gate +0.15), trained −0.011, `single` 13 → 7, native `swap` 11 → 3 | not a pass; rows at 7–13/36 give a word nothing to combine |
| G1 even / dense | 374 rows, 6 000 steps, grid 50 %, plain | even **20 / 12 / 3 / 2** (single / ext / small / kanji), native 20 / 25; dense 15 / 10 / 1 / 2, 25 / 20; `step1_0920` (53 k, ΔFM) 20 / 8 / 0 / 8, 19 / 8 | even taken (identity 37 vs 28, dense worse on seed 1); kanji unmoved |
| G1 paired | even mix, `--pair_loss 1 --lr_rows 2e-3` (scene batches ΔFM, grid batches plain) | 18 / 10 / 0 / 4, native 17 / 11, en cos `swap` 0.961 vs 0.947; scenes intact, misses are repeats of the right glyph (ああああ) | plain stays (tie on identity, 18 % slower); plain's native hits are wipe-inflated |
| G1 75 % | even mix, 4 400 steps = the 50 % arm's 234 draws per row (12 scene + 222 cell) | 12 / 12 / 0 / 3 (seed 1 2/18), native **5 / 4**, in-frame found 0.62 · 0.59 (best of four) | scene draws do not trade for cell draws; no step-1 time saved by share |

## Why a grid

- Step 1 spent 53 k steps on one glyph per canvas: 374 rows, one row per item,
  each row updated once in ≈ 94 steps.
- A grid item holds k units in k cells. It pays k rows per step, every row
  meets random neighbours in the caption, the glyph stays large, and it needs
  no real word — so it reaches the 78 kanji rows that no word carries.
- The base model already routes a quoted text to a cell through a position
  clause; the credit for cell i goes to clause i without training anything.
- What it lacks is frame: every item holds 4–9 units, so grid-only answers a
  one-kana caption with a three-glyph line. Scene singles are where the rows
  get the frame, the trigger and "one unit" — hence a mix in one run.

## The recipe (as built)

`--grid` beside `--scenes`; the grid items join the S-line mix, batches stay
one *(shape, source)*, the share is the item ratio (`--n_items` : `--n_grid`).

- **Deal, don't sample.** Units come off a shuffled deck of the singles pool —
  every unit by its weight, so a small kana's 6 digraphs share one unit's mass
  as they do in the scene singles — and a unit never repeats inside an item.
  Unit → cell is the shuffle; **clauses stay in reading order** (shuffled
  clause order lost units at k = 9 in the probe).
- Grids `2x2,3x3,2x3,3x2` at equal weight (512², 512², 416×624, 624×416 — one
  token family, 6.3 cells per item); digraphs go into every grid, 3x3 included.
- **Count stays predictable from the caption** (`findings.md`): k clauses = k
  units, one bare `reads as` clause = one unit.
- Loss: a grid item has no `ref_file`, so under `--pair_loss 1` grid batches
  train plain and scene batches ΔFM. ΔFM on the grid itself is not planned.
- σ 0.7–0.9 (identity band) while cells are singles.
- Exposure arithmetic (ledger § 5, mean rates): a cell draw ≈ 0.4 scene draw
  of identity, ≈ 0.35 of native `en`, ≈ 0.55 of `swap`; at grid 50 % a step is
  worth 7.0 / 6.4 / 8.9 scene draws against scene-only's 4. The rates are
  front-loaded (0.5–0.9 on rows under 50 scene draws, 0.04–0.13 on native at
  the knee), and nothing measures a row past ≈ 800 cell draws.

## The run that replaces step 1 (M2)

| knob | value | why |
|---|---|---|
| inventory | `step1_0920`'s (374 rows, `*1` weights), cold | the comparator is that table |
| data | 20 000 scene singles + 20 000 grid items, even mix, bubble 0.5 | more items = more distinct company per row; render cost is CPU minutes |
| share | **50 %** | 75 % at matched exposure kept the rows in the grid frame (native 5 / 4); M0 had 25 % ahead of 50 % only where singles were not starved |
| steps | 80 000 (≈ 9.4 h plain at 2.38 it/s) | 428 scene + ≈ 2 700 cell draws per row; scene draws alone are 75 % of `step1_0920`'s |
| loss | **plain** (`--pair_loss 0 --lr_rows 1e-3`) | paired ties identity, loses native on glyph repeats, 18 % slower |
| native read | `--native_floor 1` (hit & scene kept) + the sheets | plain's seed-0 hits are a white canvas with one glyph; hits alone overstate it |
| σ / lr / box | 0.7–0.9, cosine, `--box_share 0.25` | the singles recipe of record |

Optional, not built: a share schedule (grid-heavy early, scene-only tail for
the last ≈ 15 %) — the grid's contribution is front-loaded, native is paid in
cumulative scene draws, and the tail is the one lever that addresses both
native failure shapes of G1 (plain's wipe, paired's repeat). One sampler
change. The 75 % arm says the *total* scene budget cannot shrink, so a
schedule moves scene draws late, it does not remove them.

**Pass:** `single` / ext / kanji at or above 20 / 8 / 8 and native at or above
19 / 8, `single_kanji` up, no three-glyph answers to a one-kana caption on the
sheets. **The real read (M1's):** an unchanged step 2 (`step2_0920b` data argv,
Round 2 train argv, 6 k) on the new table, `row_dose.py` against the seed — the
< 400 bins, +0.000 today, are the target. Identity up with those bins flat
closes the line as "identity got cheaper, sentence content is another problem".

## S1 — word cells (parked)

`--grid_words <tsv>`: cells of 2x2 / 2x3 / 3x2 hold 2–4-piece real words whose
every piece is a trained row, drawn by row (rows cycled, least-used word, cap
25), held by string → `gword` / `gword_held`, never 3x3, a bubble cell takes
one glyph fewer than a flat one (`grid.max_glyphs`, 56 px floor). The kana
block carries 2 121 such words; the 24 rows of M0 carry 84, so S1 has no micro
arm. Cold it is not a pass (table above). **Owed:** the single-cell mix and the
word-cell mix both warm-started from a learned table (`--init_rows … --lr_warmup
500 --init_anchor μ`), 3 000 steps, `row_dose.py --seed` read; gate +0.15 on
held piece hit with `single` and native inside the rerun floor. The natural
seed is M2's table, so S1 follows M2.

## Code

In: `src/data/grid.py` (`--grid`, `--n_grid`, `--grid_bubble_frac`,
`--grid_fill_min/max`, weighted deck; `--grid_words` + `--grid_word_frac / _pieces
/ _cap / _held / _min_glyph`), `common.prompts.grid_caption`, eval groups
`gword` / `gword_held` (`row_dose.py`, `sub_exact.py`; `row_dose.py` counts a
grid item per cell), `probe/position_probe.py --arm_dir <arm> --units …` (the
in-frame read → `<arm>/inframe/`), tests + CLI golden.

Owed: the share schedule (optional, above); in-box weighting for grid items
from the recorded per-cell `boxes` (only if a read asks — `--box_share` applies
to scene items only).

## Not this plan

- Grid alone as a seed table (S0: 17/36, native 13 vs 29/36, 44).
- A denser grid mix (G1: identity 28 vs 37, worse on seed 1).
- Budget arithmetic in draws per row alone; row oversampling of an unchanged
  pool (the boost gate).
- Clause-order shuffling at k = 9; the `reads as "A", "B"` list form as an
  address.
- ΔFM on grid items; random multi-glyph strings in a cell (standing rule: real
  words).
- Reading a mix against a singles arm at equal *steps*: the control is the
  mix's own scene draws (M0's A750).
