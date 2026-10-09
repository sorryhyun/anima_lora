# hypothesis — what a row learns from an item, and where it acts (2026-10-01)

(The pre-retrain `hypothesis.md` is under `_archive/`.)

"Layout" here is the text's own geometry — where the text block sits, its
span, slot count, glyph size, line count and direction, its bubble or box —
not the scene around it; which glyph fills a slot is "identity".

## H1 — a row is one vector at every σ

A row is added at encode (`ExtDelta` on `llm_adapter.embed`), so it has no σ
input: the band an item trains at decides **what** the row learns, and the
row then says it at **every** σ of every caption that carries it.

- An item teaches what the FM target resolves at its band. At σ ≥ 0.9 that
  is layout (span, slot count, glyph size, line count — `findings.md`
  § 1–8); identity resolves only where the item's glyphs resolve, lower the
  smaller they are (§ 4, § 7, § 8).
- An item trained above where its glyphs resolve teaches its **layout alone**.
  At inference that layout lands in the high-σ steps that set the layout,
  so it overrides the layout of every caption the row appears in.
- So band and px are one choice: an item belongs at the σ where its glyphs
  resolve, and no higher.

## H2 — the rows of an item share one signal

Every row of an item's caption is trained on the same loss over the same
text region: nothing ties `a`'s gradient to the place `a` is drawn (the
loss box is the region, and the glyph tokens' cross-attention is diffuse —
`docs/findings/crossattn_self_attn_dominance.md` Result 4). So a large part
of each row's update is one direction shared by the rows that co-occur —
"fill the text region" — and that, not a per-glyph change, is what the
rows bring to a caption: a region filled to the items' extent, with
whatever glyphs are known (the leftover-slot repeats, `findings.md` § 1–2).

- Stage B (09-26, `_archive/reports/stage_b_2026_09_26.md`): the donors'
  mean Δ added to other rows moved composition (≤ 1 edit 11 → 66 / 160) and
  the repeats with it (25 → 53 / 320) — a shared, glyph-free direction.
- One-vocab items (grid_single, scene_single) carry no shared credit; the
  seed's singles draw 68 / 72 kana in a 3 × 3 grid (§ 7).
- `b0305_reband`'s update (164 rows moved, |d| ≈ 29 % of a row): the mean
  direction holds 12 % of the update's energy, median cos(d_i, mean) 0.32.

H1 says why the signal an item gives is layout (its band), H2 why the rows
of an item cannot tell it apart. Both can hold.

## What it explains

- `b0305_reband` (10-01): the seed's 12–24 px kana windows, warm at
  0.75–0.93, μ 0.02. The floor's top banner turns into small columns and
  sentence-length lines in bubbles — the b0305 items' layout — at an
  ordinary drift (warm_cos 0.959, vs 0.965 for the garble μ 0.02 arm at
  0.6–0.85, which kept the layout).
- `garble_replace` short50hb (0.8–0.95, 14 px): the rows learn the canvas,
  not the line's length (`reports/garble_replace_2026_09_30.md`).
- The polish passes moved layout, not strings (README, 10-01).

## Predictions

0. **No training, both at once** (`experiments/shared_dir`): split the
   `b0305_reband` update into its mean and residual and render the `sent`
   ruler with seed + mean, seed + residual, the arm, and the seed rows above
   0.8 with the arm's (or the residual) below. H2: seed + mean breaks the
   layout, seed + residual keeps it. H1: seed + residual breaks it too, and
   the seed above 0.8 restores it.

1. **No training: gate the trained rows by σ.** With `sigma_split`'s switch,
   the seed rows above 0.8 and `b0305_reband`'s rows below. The floor's
   layout comes back; whatever identity the rows gained survives.
2. **Same lines, larger.** b0305's dialogue lines re-rendered at 40–64 px
   and trained at 0.75–0.93 keep the floor's layout and move identity.
3. **The reverse.** Rows trained only low (0.3–0.5) still act on layout
   through the high-σ steps. `findings.md` § 5 already sees part of it: dup
   falls 100 → 50–58 when the rows stop at 0.8.

If (1) restores the layout but the low-σ rows alone read no better than the
floor, the band carried nothing but layout. If (2) breaks the layout too,
the claim is wrong: layout leaks in whatever band the item trains at.

## Result — prediction 0 (`experiments/shared_dir`, 2026-10-01)

`b0305_reband`'s update split into mean + residual, `sent` ruler, seed 0
(92 keys), paired against the floor. Job `20261001-191758-82b5e9`,
`experiments/shared_dir/results/20261001-1946-sd0_s0/`, sheets under
`output/cjk_anima_scale/experiments/shared_dir_b0305_reband_warm/sheets/`.

| rows (above 0.8 / below) | official | ≤ 1 edit | dup | box | box_h | flat_white |
|---|---|---|---|---|---|---|
| floor (seed / seed) | 15 | 45 | 40 | 0.149 | 0.189 | 0.240 |
| `full` (arm / arm) | 1 | 8 | 52 | 0.090 | 0.340 | 0.320 |
| `nomean` (seed + residual) | 1 | 8 | 55 | 0.109 | 0.309 | 0.321 |
| `mean` (seed + mean) | 5 | 22 | 54 | 0.134 | 0.169 | 0.250 |
| `s_full` (seed / arm) | 3 | 25 | 43 | 0.133 | 0.202 | 0.242 |
| `s_nomean` (seed / seed + residual) | 6 | 32 | 44 | 0.150 | 0.204 | 0.241 |

- **The layout break is in the per-row residual, and it acts above 0.8.**
  `nomean` breaks the layout as `full` does (small columns, sentence-length
  lines, white boxes); with the seed rows above 0.8 the floor's layout is
  back on every placement measure. H1 holds; H2's mean direction is not
  what moves the layout.
- **The mean direction keeps the layout and adds glyphs to it.** `mean`
  draws the floor's banner with one more glyph in it (こんにちは →
  `こんにちちは`, `こんにこちは`): dup 40 → 54 (20 gained / 6 lost,
  p 0.009), ≤ 1 edit 45 → 22 but 58 → 47 once repeats are collapsed. The
  shared direction exists and it is "more glyphs in the region" — H2's
  signal, at the scale of the repeats, not of the layout.
- **The arm's rows carry no identity gain.** Below 0.8 alone they cost
  strings (`s_full` ≤ 1 edit 25, `s_nomean` 32, floor 45): prediction 1's
  "whatever identity the rows gained survives" has nothing to keep.
- Only the mean was removed. The residual's first principal direction holds
  24 % of the update's energy (the mean 12 %), so a shared subspace beyond
  the mean is not excluded.
- The kanji-only strings (山田太郎, 日本人, 大丈夫, 何時間) read the same in
  every arm: the arm moved kana rows only.


## Result — predictions 1–2 (`experiments/span_reband` + `shared_dir`, 2026-10-01)

Prediction 2 ran as `span_reband` (the seed's b0507 windows, ≈ 34 px, at
0.85–0.95, warm μ 0.02; `proposal_seed_synthesis.md` § 1): the floor's
layout holds (the banner stays) and identity does not move — it falls
(official 29 → 16, ≤ 1 edit 92 → 70) while the banner's glyphs get smaller
and more (dup 100 → 113). Prediction 1 on that arm (`shared_dir`, seed 0):
`full_s` (arm above 0.8 / seed below) = `full` on every count, `s_full`
(seed above / arm below) = the floor. The arm's whole effect is above 0.8;
the gate has nothing to restore. H1 holds a third time; what the band
teaches is the item's glyph size relative to its region, and the region
itself stays the base's (`proposal_seed_synthesis.md` § Reading).
