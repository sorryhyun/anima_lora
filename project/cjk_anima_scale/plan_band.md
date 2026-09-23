# plan_band — which σ band, keyed on what (2026-09-23)

**Closed 2026-09-23** — verdict in
[`band_experiment_results.md`](band_experiment_results.md): the band is
keyed on glyph count, px sets the floor, grid one step higher. This file
is the plan as it ran, with each stage's read under it.

Question: does the σ band a row should train in depend on the **layout kind**
(scene composite / grid cell / flat), on **glyph count** (one glyph vs a
multi-glyph one-token piece), on **glyph complexity** (kana vs many-stroke
kanji), or only on the **rendered glyph size** that those happen to set? And
when training renders one row at **several sizes**, does the item's band
follow the item or the row? Today every one of these is confounded with size
in the data, so no existing read separates them. This plan makes size a
controlled variable and reads each factor at fixed size.

Companion: [`recipe.md`](recipe.md) (what the table of record trains with).
Code: `project/cjk_renderable_anima/src/`. Per-stage reads land in
`project/cjk_renderable_anima/reports/` as they finish; **once the
experiment is complete, the consolidated result is written to
[`band_experiment_results.md`](band_experiment_results.md) in this
directory** (user, 2026-09-23) — this file stays the plan, that file is the
verdict.

## 1. What is already known, and what it does not settle

Free reads on the cf_sense pairs (no training; EN = the ceiling, JA = trained
rows). Binned by per-glyph rendered area in latent cells² (64 ≈ 64 × 64 px),
EN and JA fall on one curve:

| cells²/glyph | EN peak σ | JA peak σ | typical item |
|---|---|---|---|
| < 25 | 0.6 (live 0.5–0.7) | – | EN 2-word string, 11 letters |
| 25–50 | 0.7 | – | EN 7-letter word |
| 50–100 | 0.8 | 0.7 | JA 4–5-glyph piece, big flat render |
| 100–400 | 0.8 | 0.8 | JA 2–3-glyph piece, big flat render |
| 400+ | – | 0.9 | JA 2-glyph piece |

Binned by **ink per glyph** (cells² of ink inside the box, so size × stroke
density) the two languages line up more closely still, and the same variable
separates glyphs of one size: among the 48 single-kana pairs (all ≈ 370
cells²), the low-ink third (へ ニ シ く) peaks at 0.8 with 0.26 and still
holds 0.08 at 0.7; the high-ink third (は す チ せ) peaks at 0.9 with 0.20
and reads 0.02 at 0.7.

| ink/glyph (cells²) | EN peak σ | JA peak σ |
|---|---|---|
| 0–4 | 0.6 | – |
| 4–8 | 0.7 | 0.7 |
| 8–16 | 0.8 | 0.7 |
| 16–32 | 0.8 | 0.8 |
| 32–64 | – | 0.8 |
| 64+ | – | 0.9 |

Reading: the more ink the input carries, the earlier (higher σ) the glyph is
legible from the input, and the caption's leverage moves up with it. Size,
glyph count (through the fit) and stroke density are three ways of changing
the ink. Kanji have not been probed at all (no kanji pair in any cf_sense
run); by ink they should sit above kana of the same px.

Two things the existing pairs do not control, so the tables above are hints,
not reads:

- **Layout is mixed inside every cell.** `_render_pair` takes
  `sample_layout`'s default `bubble_frac=0.6`, so each Gate 0 curve is a
  60 / 40 bubble / flat average, and the EN caption is the speech-bubble
  template (`TPL_EN`) for the flat 40 % too.
- **Font weight is drawn per pair.** `find_fonts` globs all seven Noto Serif
  CJK weights (ExtraLight → Black) plus the bold and medium gothics, and
  `pick_font` takes one per pair. The ink spread inside a size bin is largely
  the font draw. The single-kana "thirds" are the 24 `id` pairs of one run
  (the merge run's `id` read is bit-identical to s30k's), 8 pairs per third.

Both are recoverable post hoc — the rng is seeded (`30_000 + seed`), so each
Gate 0 pair's layout and font re-derive — and Stage A pins both.

**Every existing training read is confounded with size:**

| item kind | glyph px (median) | cells²/glyph | band it has been trained or probed at |
|---|---|---|---|
| scene composite, single (step 1 / 2 data) | 53 (p10 41 – p90 79; the 38 this row carried until 2026-09-23 was the 2-glyph number) | ≈ 44 | trained 0.7–0.9 only (`step1_0921`); B.1 (`bs_lo` / `bs_hi`, 48 px as built) |
| scene composite, multi-glyph piece | 35 | ≈ 19 | 0.5–0.7 > 0.6–0.8 > 0.7–0.9 (`micro_cf_0922`, exact 7 / 2 / 1 of 32, native 9 / 3 / 1 of 128, cf_sense at 0.7 0.080 / 0.049 / 0.011); monotone in the band, so the floor below 0.5 is untested |
| step 2 `short` / `sentence` | 32 / 25 | ≈ 16 / 10 | 0.35–0.7 (`band2`) |
| grid cell (`--grid_fill 0.5–0.8` of a 170–256 px cell) | 85–200 | 110–600 | trained 0.7–0.9 only |
| flat single (`layout v1`: 110–200 px; `jitter`: down to 60) | 110–200 | 190–600 | every W-line read; "identity at σ 0.8" |
| cf_sense probe layout (`sample_layout`: 320/n – 400/n px per glyph) | 80–200 | 100–600 | the Gate 0 numbers |

So: the single-glyph band 0.7–0.9 was established on 110–200 px flat renders;
the multi-glyph 0.5–0.7 result is on 35 px composites; grids were never
trained below 0.7 and never at small size; the probe never saw a 35 px glyph;
kanji were never probed and never trained outside 0.7–0.9. "Singles want
0.7–0.9, pieces want 0.5–0.7" and "35 px wants 0.5–0.7, 120 px wants 0.7–0.9"
fit the same data.

## 2. Hypotheses

- **H-ink**: band = f(ink per glyph). Size, count and complexity act only
  through the ink; layout adds nothing.
- **H-size**: band = f(rendered glyph px); complexity adds nothing at fixed px.
- **H-count**: at fixed size, a one-glyph row and a multi-glyph one-token row
  want different bands (a single glyph has no internal structure for the
  caption to decide; a piece does).
- **H-complexity**: at fixed px, a many-stroke kanji wants a band above a
  simple kana's (predicted by H-ink; distinguishes H-ink from H-size).
- **H-layout**: at fixed size and count, scene / grid / flat differ (bubble
  context, cell grid context, blank canvas).
- **H-item**: with one row rendered at mixed sizes, per-item bands (each draw
  at the σ its own size wants) beat one band for the row.

The design reads all of them; the training mix then gets a per-item band from
whichever factors survive.

## 3. Instruments

**Ruler A — cf_sense (no training, the ceiling).** EN text, base model, so
the number is a property of the DiT, not of any table. Needs knobs it does
not have (`src/eval/cf_sense.py` → `sample_layout` draws 320/n – 400/n px per
glyph, a bubble on 60 % of draws, one font per pair):

- `--cf_glyph_px 24,32,…` — a list, drawn **per item** and written to the
  record, so one run covers every size of a cell and the report bins by the
  recorded px (one seed stream, one launch per layout × font instead of
  one per px).
- `--cf_layout flat|bubble|grid` — `flat` = `bubble_frac 0` **and** the plain
  caption template (`TPL_PLAIN`'s EN form); `bubble` = the existing white
  ellipse at `bubble_frac 1`, with `TPL_EN`; `grid` = the pair inside one
  cell of a 3 × 3 grid. The bubble is the render's own ellipse, **not a scene
  composite** — cf_sense draws through `render_string`, and an EN scene
  composite (compositor + an EN `TPL_SCENE`) is not built; rung B.2 reads
  scene-vs-grid in training, and only a ceiling-vs-training disagreement
  there would bring the composite probe back.
- `--cf_font <path>` — pins the font for the run and writes it to the item;
  the weight axis is Noto Serif CJK Light vs Black at one px. Without the pin
  the within-cell spread is the font draw, not σ.
- `--cf_text letter|string2` — `_en_pairs` only builds whole words and
  two-word strings; the 1-letter cell needs it.

For complexity, JA on the base model is not a ceiling (no row), so Ruler A
reads complexity through EN weight (light vs black at one px) and through JA
rows only after they are trained. 1.5 min per 48-pair run.

**Ruler B — training arms (what actually matters).** Cold rows (24 kana in
S, the done 16 pieces in M), same rows across every arm of a block, equal
draws per row, two bands per cell. Reads: `single_*` exact on the run's eval set, `native` (both-hit
of 64 per clause, `en` / `swap`, the rows themselves as `--native_chars`),
en cos. The eval set is the same renders for every arm of a block.

Size is set by the data stage, per layout:

| layout | knob | small (≈ 35 px) | large (≈ 120 px) |
|---|---|---|---|
| scene composite | the bubble pool + `--scene_fill`; `--scene_min_glyph` is a floor | as built (fill 0.7): **pieces** 34–36 px (`data_micro_*`); **singles** 48 px, p10 37 (`data_band_s_0923`) — one glyph fills the bubble's short side, so the pool has no 35 px single without shrinking below the fit (the jitter objection below) | fill 1.0 restricted to the large-bubble scenes (`--scene_min_tokens` / a pool filter), or a `--scene_glyph_px` cap |
| grid cell | `--grid_fill_min/max` (share of the cell short side) | 0.15–0.25 | 0.5–0.8 (as built) |
| flat | `--layout` (`v1` 110–200 px, `jitter` ≥ 60) | needs a `--flat_glyph_px` range | `v1` as built |

**Not `--scene_size_jitter`.** It shrinks the glyph inside a bubble the base
drew for something larger, an image the base never draws (`findings.md`); in
a scene the glyph is small because the bubble is small, so small-scene size
comes from the pool, and large-scene size from the pool's large bubbles. The
two cells that need a new data-stage knob (small flat, large scene) are not
on the Stage B ladder; large scene is built only if Stage D runs. Every built
data dir prints its per-kind glyph px and
ink-per-glyph distribution (box area / glyph count, ink pixels / glyph count
from `train.jsonl` + the render), and an arm is not run unless the median
lands in its target ± 20 %.

## 4. Design

### Stage A — the ceiling map (EN, no training, ≈ 50 runs × 1.5 min)

**A.0, no GPU — done 2026-09-23**
(`reports/cf_rebin_gate0_2026_09_23.md`, `src/probe/cf_rebin.py`): each Gate 0
pair's font and bubble flag replayed from the seeded rng, ink read off the
render, the reads rebinned. Peak σ is monotone in px and in ink (40–60 px →
0.70, 60–100 px → 0.80, < 40 px strings → 0.60), ellipse vs flat moves no
peak at fixed px, and fonts are unreadable at 1–6 pairs each. So A.1's run 4
(ellipse) is the one cell A.0 already answers; it stays as the pinned-font
control but is the first to drop if time is short. Runs 1–3 (px curve, Light
vs Black) are the ones A.0 cannot give.

**A.1:** five runs, px drawn per item from {24, 32, 48, 64, 96, 128}, text
{1 letter, 2-word string} (the 3-letter word is the middle of the two and is
dropped), 16 pairs per (px, text) cell, σ grid **0.35 / 0.4 / 0.5 / 0.6 /
0.7 / 0.8 / 0.9** (0.4 added: the low-σ addendum put the string floor there
and order wakes at 0.4, so a small-px string may peak below 0.5):

| run | layout | font | reads |
|---|---|---|---|
| 1 | flat | Noto Serif CJK Regular | the px curve at fixed everything (H-size / H-count) |
| 2 | flat | Light | H-ink: ink changes, px does not |
| 3 | flat | Black | H-ink, the other side |
| 4 | ellipse | Regular | H-layout vs run 1 |
| 5 | grid cell | Regular | H-layout vs run 1 |

192 pairs × 7 σ per run ≈ 6 min; ≈ 30 min total. Output: peak σ and the live
band (move ≥ 0.1) per cell, and the same numbers replotted against ink per
glyph.

Decides: whether peak σ is a function of px alone or of ink (runs 1–3 at one
px separate them), whether a single letter follows the same curve as a
string at the same px (H-count at the ceiling), and whether flat / ellipse /
grid differ at fixed px and count (H-layout before any training;
scene-vs-ellipse is not read here, § 3). **This is the gate for Stage B**:
a factor that does not move the ceiling peak by ≥ 0.1 σ at any px is not
given a training cell.

**A.1 — done 2026-09-23** (`reports/cf_band_a1_2026_09_23.md`, 10
launches, 25 min). Peak σ is monotone in font px (letter 0.5 / 0.5 / 0.6 /
0.7 / 0.7 / 0.8 at 24 / 32 / 48 / 64 / 96 / 128); a two-word string peaks
**0.1 above** a letter at the same px (H-count holds, sign reversed: count
pulls up); ellipse = flat; the grid moves the peak up 0.1–0.2 σ from 32 px
on with half the leverage (H-layout holds for grid only); Black vs Light
moves one near-tie cell (96 px) and the live floor by one step at 32–48
(H-ink ≈ a half-step, under the 0.1 gate — **Stage C does not run**).
Strings at font ≥ 64 crop on the 512 canvas, so the valid string cells are
24 / 32 / 48. Ceiling predictions for B: S at 48 px in a bubble → live
0.5–0.7 (`bs_lo`); M at 35 px → 0.5–0.8; grid cells at 85–200 px → 0.8–0.9
(B.3's question, answered: step 1a's grid half is banded right).

**A.2 — the small end, 16 px (queued 2026-09-23).** Scope change (user):
the pack has to serve downstream LoRA training on real images, and real
images carry 16 px glyphs, so the band table must cover 16 px, not stop at
24. A.1's grid strings drawn at 13–16 px peaked at σ 0.35–0.4 — the floor of
the grid — with half the 24 px leverage, so the peak may sit below it. Four
launches, px {12, 16, 20, 24, 32}, σ {0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
0.5, 0.6}, Regular: flat × {letter, string2}, grid × {letter, string2}
(`--eval_tag a2`). Reads the 16 px peak and live band, and whether the grid
shift (+0.1–0.2 σ) holds at the small end. Training consequence: the 16 px
cell is **scene × multi-glyph × `--scene_min_glyph 16`** — the floor is a
capacity (`region_capacity`: how many glyphs a bubble holds at that px),
so lowering it lets long texts into small bubbles and the glyph lands at
bubble ÷ glyphs, the small dialogue text of a real page. A single glyph
always takes the bubble fit (pool p10 37 px), so a single row meets 16 px
only inside a text run, in training as in real data. The cell needs its
own band below 0.5, which no rung of Stage B covers yet; the rung is added
after A.2 reads.
**Read 2026-09-23** (A.2 section of `reports/cf_band_a1_2026_09_23.md`):
a 12–16 px letter carries *more* leverage than 24 px (0.41 vs 0.30) over
0.2–0.6; the size curve stays monotone (peak 0.4 at 12–16, 0.5 at 20–32);
grid +0.1 σ and half the leverage down to 16 px. **A 16 px scene string
trains at 0.25–0.6** (centre 0.4), below `band2`'s 0.35–0.7 — a
`--scene_min_glyph 16` sentence dir wants 0.25–0.7 or the per-item band.
Small glyphs are a band question, not a capability question.

### Stage B — training arms (JA rows, the confirmatory read)

Ruler A predicts the band; B checks that a *trained* row's preference follows
the ceiling (Gate 0: leverage sits where the row was trained, singles at 78 %
of the ceiling), at the two sizes the recipe is in. It is a ladder, not a
cross: each rung is one two-band pair, and the next rung runs only if the
previous one left the question open.

Row blocks, both cold, the same rows across every arm:

- **S** — 24 kana (12 hiragana, 12 katakana, no dakuten pairs). 24, not 16:
  power is rows, not seeds, and this block carries every rung below.
- **M** — the 16 kana-only one-token pieces of `ja_micro_cf_0922.txt`
  (2–4 glyphs, untrained ranks 1 901+), already run.

Per arm: one layout's data only (no mixing), the `micro_warm_0923` draw
budget (≈ 190 scene draws per row, so steps scale with rows: 24 rows ≈ 4 500
× batch 4), `--seeds 2`, bands {0.5–0.7, 0.7–0.9}.

1. **Scene × small × S** (2 arms) — the rung recipe.md's step 1a stands on:
   do singles at composite size also prefer 0.5–0.7? Read beside the done
   M pair (`mcf_plain_hi` / `mcf_plain_lo`, scene × small × M) — that
   comparison is H-count in training at one size and one layout, for free.
   **Run 2026-09-23 as scene × as-built × S** (`data_band_s_0923`, arms
   `bs_lo` / `bs_hi`, 4 500 steps): the S singles land at 48 px (p10 37),
   not 35 — the ± 20 % gate fails on the pool itself (§ 3), and a fill 0.5
   build would be the shrunk-in-bubble image § 6 rules out. This is the
   size step 1a's scene half actually trains at, so the 1a question is read
   as intended; the H-count read against M is **not** at one size (48 vs
   35 px) and leans on Stage A's letter-vs-string2 curves at 32 / 48 px.
   Rows: あいおなぬのまむめ アイナネヌマムメヨ やられ ラルロ (`chars:`
   base, exact reads the first 18; native reads all 24 on 4 prompts).
   **Read 2026-09-23** (`reports/band_b1_2026_09_23.md`): **0.7–0.9** —
   native en both-hit 84 vs 49 of 192, swap 55 vs 30 (seed floor 19 / 8),
   exact 29 vs 25 of 36 (under the floor, no vote), the row's own
   leverage at 0.7 on both arms. Low-band misses are runs of the right
   glyph, high-band misses are identity swaps. Against M (35 px pieces →
   0.5–0.7) the band is keyed on **glyph count**, not px; the ceiling
   predicted 0.5–0.7 for this cell and was wrong about the trained row —
   it bounds where leverage can reach, not which half a single row needs.
   Rung 2 does not run, rung 3 is answered at the ceiling.
2. **Grid × small × S** (2 arms; `--grid_fill_min 0.15 --grid_fill_max
   0.25`, the knob exists). Runs only if rung 1 picked 0.5–0.7. With flat
   large already known to want 0.7–0.9 (the W-line, and `--t_max 0.6` lost
   on flat singles) this is the crossed cell: grid-small agreeing with
   scene-small says size, grid-small agreeing with flat-large says layout.
3. **Grid × large × S** (2 arms) — the grid as step 1a trains it, read at
   both bands for the first time. Runs only if rung 2 said layout (grid
   behaves like flat, not like scene), or if rung 1 picked 0.7–0.9 and the
   question is whether the grid half of 1a could do better lower.

Dropped from the cross, and why: **flat × large** (the W-line already read
it; a protocol-reproduction control costs 1.3 h and changes no decision),
**grid × M** and **flat × M** (no recipe cell trains pieces in a grid or
flat; 1b is scene-only), **flat × small** and **scene × large** (each needs a
new data knob; Stage A's ellipse and grid runs read the layout question at
the ceiling, and rung 2 reads it in training). Any of them comes back only
if Stage A shows a layout effect ≥ 0.1 σ that rung 2 cannot attribute.

Each arm also gets `cf_sense --cf_rows piece` at the *training* px, so the
row's leverage is read where it trained.

### Stage C — complexity (kanji), 0 or 2 arms

Gated on Stage A runs 2–3: if Light and Black peak at the same σ at every px
(ink does not move the ceiling at fixed px), C does not run and kanji take
the kana band for their px. Otherwise one pair: row block **K** = 16 kanji
in two strata of 8 by stroke count (≤ 6: 人 日 口 山 川 …; ≥ 12: 議 識 繊 響 …,
from the `kanji:200` inventory ranked by ink in the training font), scene ×
small × K × band {0.5–0.7, 0.7–0.9}, read exact and native per stratum.

Decides H-complexity in training: if the ≥ 12-stroke stratum prefers a band
above the ≤ 6-stroke stratum at the same px, ink (not px) is the key. Also
the first read of kanji below 0.7 at all; `single_kanji` has only ever been
trained at 0.7–0.9 (17–18 / 36 on the band2 arms vs 21–24 for kana).

### Stage D — mixed-size training (H-item), 0 or 2 arms

Runs only if B found the two sizes want different bands (one band winning
everywhere makes it moot). Data = the union of rung 1's scene-small dir and a
scene-large dir for block S (equal halves; the large-composite knob from
§ 5 is owed here, not before). Two arms, same draws:

- **fixed wide**: one band spanning both sizes' winners (e.g. 0.5–0.9).
- **per-item**: each draw's band from its own size (the `_remap_band` change
  in § 5), the two sizes' winners.

The "fixed narrow" arm is dropped: B's own reads already say a narrow band
loses at the size it was not chosen for. Read exact and native at **both**
sizes (the eval set carries a small and a large render of every row).
Decides whether mixed-size data needs per-item bands or whether a wide band
pays the same. If per-item wins, the same switch covers grid + scene mixes
in step 1a.

### Reading rule

Per cell, the winning band is the one ahead on exact **and** native both-hit
(en + swap); a split is a tie. Then:

- **H-size / H-ink holds** if rung 2 agrees with rung 1 (small wants the
  same band on scene and grid) and small ≠ large; Stage A runs 2–3 and
  Stage C say which of the two. → per-item band from the item's px or ink,
  one table for everything.
- **H-count holds** if S (rung 1) and M (done) pick different bands at
  scene × small. → band keyed on (px, count).
- **H-layout holds** if rung 2 disagrees with rung 1 at the same size and
  count. → band keyed on (px, layout); the training mix then needs per-kind
  bands, which `_remap_band` does not do today.
- Anything else (interactions) → the table of record trains each kind in its
  own run at its own band, as recipe.md already does for 1a / 1b.

**Noise floor = the two training seeds.** A cell votes only if the band
difference exceeds the larger of (a) the two seeds' disagreement within
either arm and (b) 3 exact / 3 native hits, the movement the 40-step grid
read showed on nothing. A cell under the floor is "no preference" and does
not vote.

### Budget

| stage | arms | GPU | runs if |
|---|---|---|---|
| A.0 | rebin, no GPU | 0 | always |
| A.1 | 5 cf_sense runs | ≈ 0.5 h | always |
| B.1 | 2 (S, scene small) | 2 × (≈ 40 + 9 + 2) min ≈ 1.7 h | always |
| B.2 | 2 (S, grid small) | ≈ 1.7 h | B.1 picked 0.5–0.7 |
| B.3 | 2 (S, grid large) | ≈ 1.7 h | B.2 said layout, or B.1 picked 0.7–0.9 |
| C | 0 or 2 | ≈ 1.3 h | A.1 shows an ink effect at fixed px |
| D | 0 or 2 | ≈ 1.7 h | B found two bands for two sizes |

Worst case ≈ 8.5 h against the ≈ 20 h cross; the likely path (A, B.1, B.2,
D) ≈ 5.5 h. B.1 alone decides step 1a's band; B.2 decides how the band is
keyed; B.3 decides whether the grid half of 1a keeps 0.7–0.9; C decides
whether kanji get their own band; D decides how a mixed-size run is banded.
Stop after any rung whose answer already fixes the recipe.

## 5. Code owed, by the stage that first needs it

Shipped 2026-09-23 (tests in `tests/test_wake_src.py`, CLI golden
regenerated):

- **A.0**: `src/probe/cf_rebin.py` — replays `cf_sense`'s seeded draws to
  tag each pre-knob pair with its font / bubble / px, reads ink off the
  render, bins by px / ink / font / bubble.
- **A.1** (`cf_sense`): `--cf_glyph_px 24,32,…` (cycled over the items and
  recorded; `--cf_per_pair` = its length gives every pair every px),
  `--cf_layout mixed|flat|bubble|grid` (`mixed` = the Gate 0 draw and its
  bubble caption on every pair; `flat` / `bubble` caption what was drawn,
  `TPL_PLAIN_EN` is new; `grid` = the centre cell of a 3 × 3 flat grid under
  the data stage's grid caption, EN only), `--cf_font <path>` (recorded),
  `--cf_text letter|word|string2`. Every item records `px` / `font` /
  `bubble` / `glyphs` / `ink_a` / `ink_b`; the report gains by-px and by-font
  tables; the `.pt` carries a `meta` block. The layout and font draws are
  consumed whatever the knobs say, so a run's pair sequence is the Gate 0
  sequence for the same seed.
- **B.1** (data stage): `src/common/render/ink.py::ink_pixels` (background
  = the ring outside the box, so a dark canvas reads the same as a bubble);
  every boxed record gets `glyphs` / `ink` / `box_area`, and the build prints
  per kind the median (p10–p90) glyph px and ink per glyph in cells² — the
  ± 20 % arm gate of § 3 reads that line. Rung B.2 uses the existing
  `--grid_fill_min/max`.

Still owed, **D** only: a scene glyph cap or a large-bubble pool filter for large
  composites, and `_remap_band` (`src/train/stage.py`) keyed on the item's
  own glyph px or ink (`box`, `ink`, glyph count on the record) instead of
  the single / multi split that classifies a grid item by its joined cell
  text. `--flat_glyph_px` is not owed by any rung.

## 6. What this plan does not do

- It does not touch step 2's band (sentence items at 25–32 px; `band2`'s
  0.35–0.7 is the read of record there and the size curve agrees with it).
- It does not re-open CF (killed at Gate 2, `micro_cf_0922`).
- It does not vary steps per row; every arm uses the 190-draw budget so the
  band is the only difference inside a cell.
- It does not use `--scene_size_jitter` for size control (§ 3).
