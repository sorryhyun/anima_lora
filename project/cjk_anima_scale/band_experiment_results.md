# band_experiment_results — which σ band, keyed on what (2026-09-23)

**The vocab band law** — the line's name for this document ([`README.md`](README.md)).
Not a theory: every row below is a measured read, and it holds over the
sizes, layouts and units those reads covered.

The verdict of `plan_band.md` (closed and deleted 2026-09-23; git `f5cd4c0c`).
Every number here is in `project/cjk_renderable_anima/reports/`
(`cf_rebin_gate0_2026_09_23.md`, `cf_band_a1_2026_09_23.md` with its A.2
section, `band_b1_2026_09_23.md`).

## Answer

**A row's training band is keyed on its glyph count. Rendered glyph px sets
the floor that band may reach, and a grid cell sits one step higher.**

- **Single-glyph rows train at 0.7–0.9** — at scene size (48 px, B.1) and at
  grid size (85–200 px, the ceiling). Step 1a keeps its band.
- **Multi-glyph one-token rows train at 0.5–0.7** at scene size (35 px,
  `micro_cf_0922`). Step 1b keeps its band.
- The per-px ceiling table (§ 2) is where the caption's leverage *lives*,
  not the band a row trains in: a single row needs the upper half of its
  live band (the σ that decides "one glyph, once"), a piece can sit on the
  peak because its count is in the token.
- Ink (font weight) and the bubble ellipse move nothing worth a cell.
- Small glyphs are a band question, not a capability question: a 16 px
  letter carries more caption leverage than a 24 px one, over a band twice
  as wide, centred at σ 0.4.

## 1. Hypotheses, as read

| hypothesis | verdict | where |
|---|---|---|
| H-size — band = f(px) | **holds at the ceiling**: peak σ 0.4 (12–16 px) → 0.5 (20–32) → 0.6 (48) → 0.7 (64–96) → 0.8 (128), monotone, no plateau. In training it sets the floor, not the band | A.1 + A.2 |
| H-ink — band = f(ink per glyph) | **half a grid step**: Light = Regular; Black moves one near-tie cell (96 px) and the live floor by one step at 32–48. Black 48 px (ink 8.1) peaks 0.6 while Regular 64 px (ink 8.6) peaks 0.7, so px, not ink, is the variable. Stage C not run | A.1 runs 2–3 |
| H-count — one glyph vs a piece at fixed size | **holds in training, sign as the recipe assumed**: singles 0.7–0.9 (S, 48 px), pieces 0.5–0.7 (M, 35 px). At the ceiling a string peaks 0.1 *above* a letter at the same px, so the training preference is not the ceiling's: a single row buys the count at high σ, a piece has it in its token | B.1 vs `micro_cf_0922`; A.1 |
| H-complexity — kanji vs kana at fixed px | **read 2026-09-23 (`plan_kanji.md`): ink does not move the band, structure moves where leverage sits.** Three strata of 8 (kana / simple kanji at the same ink / dense kanji at 2 ×) on 48 px composites, 0.7–0.9 vs 0.8–0.95: every stratum collapses on the higher arm (native 71 → 18 of 192) and every arm's leverage peaks at 0.7 — **0.8–0.95 is dead at 48 px**. Simple straight-stroke kanji sit one σ step above kana at equal ink (C.1, both layouts) and tolerate the high arm; dense kanji sit at the kana σ with half the leverage and read 0 / 64 native on both arms. Kanji take the kana band; density is an exposure / px question, not a band. Over `kanji:200` (96 pairs, 48 px): ink per glyph moves the soft peak **down** (r −0.44, and shrinks it), straightness (orientation-entropy descriptor, independent of ink) moves it up by +0.03 σ per quartile range (partial +0.2) — neither is a band term | `reports/cf_kanji_c1_2026_09_23.md`, `reports/band_c2_kanji_2026_09_23.md` |
| H-layout — scene / grid / flat | **ellipse = flat** to the second decimal at every px; **grid +0.1–0.2 σ from 32 px on, half the leverage**, down to 16 px. Not read in training (B.2 / B.3 dropped, § 4) | A.0, A.1 runs 4–5, A.2 |
| H-item — per-item bands in mixed-size data | **not triggered**: both halves of step 1a want 0.7–0.9. Re-scoped to the mixed-px sentence data (§ 4) | — |

## 2. The ceiling table (EN, base model, Noto Serif CJK Regular)

Peak σ / live band (mean move ≥ 0.1) by font px. Strings above font 48
crop on the 512 canvas and are not cells.

| font px | single letter, flat = bubble | two-word string, flat | single letter, grid cell |
|---|---|---|---|
| 12 | 0.4 / 0.2–0.6 | 0.35 / 0.2–0.6 | 0.3 / 0.3–0.5 |
| 16 | 0.4 / 0.2–0.6 | 0.4 / 0.25–0.6 | 0.5 / 0.4–0.6 |
| 20 | 0.5 / 0.25–0.6 | 0.5 / 0.35–0.6 | 0.6 / 0.4–0.6 |
| 24 | 0.5 / 0.35–0.7 | 0.6 / 0.4–0.7 | 0.5 / 0.4–0.7 |
| 32 | 0.5 / 0.4–0.7 | 0.6 / 0.5–0.7 | 0.6 / 0.6–0.7 |
| 48 | 0.6 / 0.5–0.7 | 0.7 / 0.6–0.8 | 0.8 / 0.7–0.8 |
| 64 | 0.7 / 0.6–0.8 | – | 0.8 / 0.8 |
| 96 | 0.7 / 0.7–0.8 | – | 0.8 / 0.8 |
| 128 | 0.8 / 0.8 | – | 0.9 / 0.9 |

Use: the lower edge is the floor a training band for that item may reach;
the band a row *trains* in is the count rule above, placed inside this
window. A 16 px multi-glyph scene item therefore trains at ≈ 0.25–0.6.

## 3. The training reads

| cell | rows | px | 0.5–0.7 | 0.7–0.9 | winner | ruler |
|---|---|---|---|---|---|---|
| scene × pieces (M, `micro_cf_0922`) | 16 | 35 | exact 7/32, native 6 / 3 of 64 | 1/32, 1 / 0 | **0.5–0.7** | both |
| scene × singles (S, `band_s_0923`) | 24 | 48 | exact 25/36, native 49 / 30 of 192 | 29/36, 84 / 55 | **0.7–0.9** | native (exact under the seed floor, same direction) |

Signatures: the low band's single-glyph misses are the right glyph drawn as
a run (いい, ののののの); the high band's are identity swaps (お→む, マ→チ).
The trained rows' own leverage (`cf_sense_ja` at 48 px) peaks at 0.7 on
both arms, one step above the EN ceiling's 0.6.

A note on the plan's premise: single-glyph scene composites are **48–53
px**, not 38 — a single glyph takes the bubble fit and the pool has no
smaller bubble. B.1 therefore ran as scene × as-built, which is the size
step 1a actually trains at; the "small" single cell does not exist without
shrinking the glyph below the fit, which the plan rules out.

## 4. Left unrun, and why

- **H-count is still size-confounded** (S at 48 px vs M at 35). The ceiling
  puts both in one live band, so count is the only explanation standing,
  but no training cell has both blocks at one size. The clean cell is
  pieces at ≈ 48 px (fill 1.0 in the large-bubble scenes), 2 arms; the
  35 px single is not buildable.
- **Grid band confirmed at the ceiling only.** B.3 (grid × large × S at
  both bands) was dropped on the ceiling's 0.8–0.9; the ceiling
  mispredicted B.1's single cell, so this is a confirmation left open, not
  a decision. B.1's rule (singles want the top half) still says 0.7–0.9.
- **B.2** (grid × small × S) needed a 0.5–0.7 pick and did not run.
- **Stage C** ran after all (`plan_kanji.md`, 2026-09-23): the ink gate had only seen Latin weight (1.7 ×), not kanji ink (2.6 ×). Read above; `bk_lo` (0.5–0.7 on kanji) unrun — no stratum asked to go down.
- **Stage D** is re-scoped: its case is a data dir mixing 16 and 30 px
  sentence text (ceiling 0.25–0.6 vs 0.4–0.6), which is step 2 territory
  and outside this plan's § 6.

## 5. What the recipe takes from this

- `recipe.md` step 1a (singles, 0.7–0.9) and step 1b (pieces, 0.5–0.7)
  stand as written; the band is chosen by the row's glyph count, which is
  why they are two runs and not one run with `--t_band_multi`.
- Any new item kind gets its band from (count rule) ∩ (§ 2 window for its
  px, +0.1 for a grid cell). For 16 px text in scenes
  (`--scene_min_glyph 16`, multi-glyph only — a single glyph never renders
  that small in a bubble): 0.25–0.6.
- The per-px table is also the argument for the pack serving downstream
  LoRA training on real pages: the DiT reads a 16 px glyph from the
  caption as readily as a 24 px one, lower in σ.

## 6. Next cells (not part of this plan)

1. **Switch vs accumulate** (Kim et al. 2025, easy-to-hard = high σ first,
   lower clusters accumulated): warm from `bs_hi`, 40 steps/row, 0.5–0.7 vs
   0.5–0.9. Decides whether step 1 becomes high → accumulate; can change
   the step 1 structure, so it goes first.
2. **16 px multi-glyph cell**: step 2-style data with `--scene_min_glyph
   16`, band 0.25–0.7 vs the per-item band. Opens step 2's band.
3. **Dense kanji: px or exposure?** Block K on the step-1a recipe (grid
   50 %, 0.7–0.9, same budget): K_hi read 0 / 64 native and near-neighbour
   swaps on 48 px scene-only draws; `step1_0921` bought its kanji identity
   on 85–200 px cells. One arm, ≈ 40 min + native.
4. **Dead rows** (マ メ ロ の イ at 0–1 of 8 on both B.1 arms):
   `table_geometry.py` on `bs_hi` before spending steps on them.
