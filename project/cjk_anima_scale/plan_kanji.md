# plan_kanji — does a kanji's ink move its band? (Stage C, reopened 2026-09-23)

`plan_band.md` Stage C was gated off on the A.1 ink read (Light vs Black
Latin at one px moves the peak by ≈ 0.05 σ, `reports/cf_band_a1_2026_09_23.md`
§ 2). That gate never saw a kanji: Black carries 1.5–1.7 × Regular's ink,
while a 12-stroke kanji at the same px carries **2.6 × a kana's** (§ 1). The
question stays open and this plan reads it. `band_experiment_results.md`
keeps the verdict; this file is the plan.

**Status 2026-09-23**: C.0 / C.1 / C.2 done — `reports/cf_kanji_c1_2026_09_23.md`,
`reports/band_c2_kanji_2026_09_23.md`. Verdict: § 4 row 2 (px + count stand;
kanji take the kana band; 0.8–0.95 dead at 48 px; dense kanji 0 / 64 native on
both arms → the px-vs-exposure cell, `band_experiment_results.md` § 6 item 3).
`bk_lo` not run. Straightness read done (C.2 report, last section): ink moves the peak down (r −0.44), straightness up by +0.03 σ — neither a band term.

## 0. Question

At fixed px and fixed count (one glyph, one row), does a kanji with 2–3 × a
kana's ink want a band above the kana's 0.7–0.9? Three answers are possible
and each changes the recipe differently:

- **H-ink**: yes, the high-ink stratum's winner sits above kana's. A 48 px
  kanji then behaves like a 64–96 px Latin letter (same ink, A.1 peak 0.7
  against 0.6) and `kanji:200` gets its own band (or an ink-keyed per-item
  band).
- **H-size + count**: no, every stratum picks 0.7–0.9. Kanji take the kana
  band for their px, and the kanji deficit (`single_kanji` 18 / 36 vs kana
  24 / 36 on the band2 arms) is exposure, not band.
- **structure**: low-ink kanji ≠ kana at equal ink. Neither px nor ink is
  the key; open.

No ceiling exists for this: JA on the base model has zero leverage at every
σ (`floor` rows, `cf_sense_gate0`), and Latin cannot be drawn at kanji ink.
So the ceiling ruler reads kanji only through trained rows (C.1) and the
decision is a training read (C.2).

## 1. C.0 — the ink strata (no GPU, done 2026-09-23)

`common.render.ink.ink_pixels` on single glyphs, Noto Serif CJK, ink in
latent cells² (÷ 64), box = the glyph bbox:

| stratum | 32 px Regular | 48 px Regular | 96 px Regular | 48 px Black |
|---|---|---|---|---|
| S kana (the 24 B.1 rows) | 2.5 | 5.2 [3.3–8.0] | 19.0 | 9.4 |
| K_lo (人 日 口 山 川 大 木 中 …) | 2.8 | 6.2 [1.6–8.9] | 23.6 | 11.3 |
| K_mid (気 時 間 語 理 感 …) | 4.7 | 10.9 [7.9–12.1] | 36.7 | 16.6 |
| K_hi (議 識 観 護 験 難 警 顔 …) | 6.1 | 13.7 [12.8–15.6] | 48.1 | 20.6 |

Reads off the table:

- K_lo and kana are one ink bin (5–8). K_hi is a bin of its own (13–14),
  tighter than any kana bin, and above every kana at 48 px.
- K_hi at 48 px Regular (13.7) = Regular 64 px Black (14.3) = the ink A.1
  read a 0.7 peak at; kana / K_lo at 48 px (5–6) is A.1's 48 px Regular
  cell (5.4, peak 0.6). **H-ink predicts one full step**, not the half-step
  the gate tripped on.
- Font weight shifts every stratum together (Black ≈ 1.7 ×) and keeps the
  order; the training font draw does not blur the strata.

Row block **K** = 24 rows, three strata of 8, every kanji one of the 200
`kanji:200` rows of `rows_step1_0921_s30k` (so C.1 reads the same glyphs
C.2 trains), ranked by ink at 48 px Regular over those 200 (一 二 上 下 are
strokes, not glyphs — out):

| stratum | rows | ink/glyph cells², 48 px Regular |
|---|---|---|
| kana | あ ぬ ま め な お の む | 7.0 (6.0–8.0) |
| K_lo | 人 大 口 中 日 手 女 水 | 7.1 (4.6–7.3) |
| K_hi | 願 構 備 動 聞 精 離 魔 | 13.3 (12.7–14.6) |

Typed so that `exact`'s first 18 hold 6 of each:

```
chars:あぬまめなお人大口中日手願構備動聞精のむ女水離魔
```

The data stage prints the per-stratum ink medians from the render; an arm
does not run unless kana / K_lo / K_hi land at ≈ 7 / 7 / 13 ± 20 % and
glyph px ≈ 48 for all three (a single glyph takes the bubble fit, so px is
fixed by construction — the same reason B.1 landed at 48). The `units:` line
confirms 24 rows before anything runs.

## 2. C.1 — the free read (≈ 5 min GPU, one tool change)

`cf_sense --cf_lang ja` on a table that already holds kanji rows trained at
one band (`rows_step1_0921_s30k`: `kanji:200` ×2, 0.7–0.9), id pairs drawn
**within** an ink stratum, `--cf_glyph_px 32,48,96`, σ 0.35 … 0.9,
`--cf_layout flat --cf_font <Regular>`. The record already carries `ink_a`
/ `ink_b`, so `probe/cf_rebin.py` bins the read by ink at fixed px.

Tool change: `src/eval/cf_sense.py::_ja_pairs` keeps only `t in KANA`. Add
`--cf_units kana|kanji|chars:<list>` (default `kana`, so every existing run
re-derives) and, for `kanji`, pair within ink terciles measured at the run's
px so an id pair never crosses strata.

Reads, per stratum × px: peak σ, live band (move ≥ 0.1), magnitude at 0.7.
This is the ink read at **fixed training band and fixed px** — the two
confounds of every earlier kanji number removed at once. Its limit is Gate
0's rule: leverage lands where the row was trained, so a stratum cannot
show a peak far outside 0.7–0.9; what it can show is the live band's edge
(does K_hi hold 0.9 where kana has dropped to 0.05?) and the 32 px cell
(does K_hi at 32 px, ink 6, read like kana at 48, ink 5?). A K_hi edge one
step above kana's at 48 px is the H-ink signature and makes C.2 a
confirmation; no difference makes C.2 the only read, not a reason to skip
it.

## 3. C.2 — the training read (3 arms, ≈ 2.7 h; 2 arms ≈ 1.8 h)

The B.1 recipe unchanged except the rows and a third band:

```
data  … --data_tag band_k_0923 --seed 0 --scenes s1,s1w,sl1w,ja_comic --scene_one_bubble ja_comic \
        --single_scenes s1,s1w --single_max_ar 2 --units chars:<K, § 1 order> --scene_mix single=1.0 \
        --n_items 3000 --scene_frac 1.0 --natural_frac 0 --strings_frac 0 --flat_bubble 1.0 \
        --scene_fill 0.7 --scene_min_glyph 28 --scene_max_lines 1 --scene_vertical 1 \
        --shapes 448,512:2,448x512,512x448
train … --train_steps 4500 --batch 4 --compile 1 --lr_rows 1e-3 --lr_decay cosine --free_residual 1e-3 \
        --box_share 0.25 --pair_loss 0 --c_flat 0 --seeds 2 --no_floor --eval_groups single,en \
        --t_min 0.5 --t_max 0.7  --arm_tag bk_lo
        --t_min 0.7 --t_max 0.9  --arm_tag bk_mid
        --t_min 0.8 --t_max 0.95 --arm_tag bk_hi
native … --native_chars <24 rows> --native_clauses en,swap --native_limit 4 --seeds 2 --delta_parts full
cf_sense … --cf_lang ja --cf_units chars:<kana>/<K_lo>/<K_hi> --cf_rows single --cf_pairs 48 --cf_per_pair 3 \
        --cf_glyph_px 32,48,96 --cf_t 0.35,0.5,0.6,0.7,0.8,0.9 --eval_tag px
```

Same draw budget as B.1 (190 scene draws per row), same rows on every arm,
cold rows, raw pack sha checked in every submit shell (the wake-line rule).

- `bk_mid` is kana's winner and the anchor. `bk_hi` is H-ink's prediction
  for K_hi (a 0.7-peak stratum wants the top half above 0.7). `bk_lo` is
  kana's loser and is the first arm to drop if time is short: it adds
  whether K_hi loses it *harder* than kana (a monotone three-point curve
  per stratum), not the sign.
- 0.95 needs a check that the sampler / `_remap_band` accept `t_max` above
  0.9 — every band so far topped at 0.9.

Rulers, all per stratum (the block is 8 rows a stratum, so power is the
stratum's rows × 2 seeds × 4 prompts × 2 clauses = 128 native reads, 12
exact reads per arm):

| ruler | read | floor |
|---|---|---|
| `exact` (single group, 6 rows a stratum × 2 seeds = 12) | hits | under the seed floor at 12 — direction only, no vote (B.1) |
| `native` both-hit (`en` / `swap`, 8 rows × 4 × 2 seeds = 64 per clause) | hits a stratum | B.1's whole-block floor 19 / 192 → ≈ 6 / 64 a stratum; a pick needs a gap above that on both clauses |
| `cf_sense_ja` on the arm's own rows, 48 px | peak / live per stratum | 0.00 floor rows |
| miss signatures (sheets) | runs of the right glyph (low band) vs identity swaps (high band), as B.1 | per-glyph, with sheets — totals hid a か/日 reversal once |

## 4. Verdict rules

Read the winner band per stratum on `native` (the ruler that voted in B.1),
`exact` and `cf_sense` as direction:

| kana | K_lo | K_hi | verdict | recipe |
|---|---|---|---|---|
| mid | mid | **hi** | H-ink holds in training | `kanji:200` trains at 0.8–0.95, or `--t_band_multi` keyed on ink per item; the § 2 table gets an ink column |
| mid | mid | mid | px + count stand; ink is the A.1 half-step at every ink | kanji take the kana band; the kanji deficit is exposure (`README` "kanji at scale is an exposure budget") |
| mid | **≠ mid** | any | structure, not ink (K_lo is at kana's ink) | open; the next cell is K_lo vs kana at 96 px |
| any | any | lo | against every read so far | re-check the strata's rendered px before believing it |

A stratum whose native gap is inside the floor on both clauses has no vote;
two of three strata voting is enough for row 1 or 2, one is not.

## 5. Left out, and why

- **Kanji pieces / words** — no recipe cell trains kanji outside `kanji:200`
  singles; pieces have their band from count (`micro_cf_0922`).
- **Kanji in a grid** — the grid shift is a layout term already read at the
  ceiling (+0.1–0.2 σ); it stacks on whatever this plan finds.
- **Weight (Light / Black) on kanji** — § 1: weight moves every stratum
  together and A.1 already priced it at a half-step.
- **Stroke count as its own variable** — ink is the measured quantity;
  stroke count only picked the candidates. K_mid (10–12) is not a stratum:
  three strata of 8 already fill the 24-row block and the mid bin's
  prediction is between the other two.
- **A JA ceiling** — does not exist (§ 0); C.1 is the nearest thing.
