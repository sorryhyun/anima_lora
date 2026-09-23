# band C.2 — kana / simple kanji / dense kanji at 0.7–0.9 vs 0.8–0.95 (2026-09-23)

`project/cjk_anima_scale/plan_kanji.md` § 3: the training read. Block K —
24 cold rows in three ink strata of 8 (C.0), on B.1's scene composites at
the bubble fit (48–55 px), B.1's train argv, two bands. The 0.5–0.7 arm was
left out by decision (user, 2026-09-23) and is queued only if a stratum's
`bk_mid` − `bk_hi` gap asks for it; C.1 (`cf_kanji_c1_2026_09_23.md`)
predicted that no stratum would prefer the higher arm.

| stratum | rows | ink/glyph cells² (C.0, 48 px Regular) | rendered px / ink (data stage) |
|---|---|---|---|
| kana | あ ぬ ま め な お の む | 7.0 | 48 / 10.8 |
| K_lo | 人 大 口 中 日 手 女 水 | 7.1 | 51 / 10.3 |
| K_hi | 願 構 備 動 聞 精 離 魔 | 13.3 | 55 / 19.8 |

```
data   … --data_tag band_k_0923 --units chars:あぬまめなお人大口中日手願構備動聞精のむ女水離魔 <B.1 scene argv>
         → units: base 24 (restricted); ink single: n 3000, glyph px 52 (41–80)
train  … <B.1 train argv: 4500 steps, batch 4, lr_rows 1e-3 cosine, box_share 0.25, seeds 2> \
         --t_min 0.7 --t_max 0.9  --arm_tag bk_mid   |   --t_min 0.8 --t_max 0.95 --arm_tag bk_hi
native … --native_chars <24 rows> --native_clauses en,swap --native_limit 4 --seeds 2 --delta_parts full
cf_sense … --cf_lang ja --cf_units chars:<kana>/<K_lo>/<K_hi> --cf_pairs 48 --cf_glyph_px 48 --eval_tag px48
```

Jobs: data `20260923-141618-97bb92`; train `20260923-143154-84479a` (`bk_mid`,
37.1 min) / `-d67d01` (`bk_hi`); natives `20260923-151615-443aeb` (27 min) /
`20260923-154447-3969e2`; cf_sense `…-151615-ce8c26` / `…-154447-cf3f99`. Raw
pack (sha 7b9fce0b…) in every submit shell. Data dir gate: per-stratum px
48 / 51 / 55, ink 10.8 / 10.3 / 19.8 (kana = K_lo, K_hi 1.8 ×) — passed.

## Reads

**Exact** (`single`, the first 18 rows = 6 a stratum × 2 seeds):

| stratum | `bk_mid` 0.7–0.9 | `bk_hi` 0.8–0.95 |
|---|---|---|
| kana | **10**/12 | 3/12 |
| K_lo | 6/12 | **7**/12 |
| K_hi | **3**/12 | 0/12 |
| all | 19/36 | 10/36 |

**Native both-hit** (8 rows × 4 scene prompts × 2 seeds = 64 a clause):

| stratum | `bk_mid` en / swap | `bk_hi` en / swap |
|---|---|---|
| kana | **42 / 33** | 9 / 5 |
| K_lo | **29 / 27** | 9 / 6 |
| K_hi | **0 / 0** | 0 / 0 |
| all | 71 / 60 of 192 | 18 / 11 |

Per row, `bk_mid` en: あ8 ぬ3 ま6 め6 な6 お6 の0 む7 · 人8 大6 口0 中0 日0
手5 女6 水4 · every K_hi row 0. `bk_hi` en: あ2 め5 な2 · 人4 手2 · rest ≤ 1.

**cf_sense on the arm's own rows, 48 px, id pairs, mean move by σ**:

| arm | stratum | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | peak |
|---|---|---|---|---|---|---|---|
| `bk_mid` | kana | 0.06 | 0.14 | **0.18** | 0.10 | 0.06 | 0.7 |
| `bk_mid` | K_lo | 0.03 | 0.20 | **0.34** | 0.17 | 0.04 | 0.7 |
| `bk_mid` | K_hi | 0.06 | 0.12 | **0.14** | 0.12 | 0.06 | 0.7 |
| `bk_hi` | kana | 0.04 | 0.06 | 0.06 | 0.05 | 0.04 | flat |
| `bk_hi` | K_lo | 0.02 | 0.15 | **0.26** | 0.13 | 0.01 | 0.7 |
| `bk_hi` | K_hi | 0.04 | 0.05 | 0.07 | 0.06 | 0.03 | flat |

## Verdict

1. **No stratum wants a band above 0.7–0.9.** `bk_hi` is a collapse on
   every ruler for every stratum (native 71 → 18, exact 19 → 10), and its
   rows' leverage still peaks at 0.7 — trained at 0.8–0.95, the rows put
   what little they have where the DiT has leverage to give, not where they
   were trained. Gate 0's rule ("leverage lands where the row trained")
   stops at the ceiling's edge: the EN ceiling at 48 px is 0.06 at 0.9, and
   a band that lives there trains nothing. **0.8–0.95 is a dead band at
   48 px** for kana and kanji alike.
2. **H-ink is rejected in training.** The dense stratum did not prefer the
   higher arm (3 → 0 exact, 0 → 0 native); ink does not move the band. The
   `plan_kanji.md` § 4 verdict is row 2: **px + count stand; kanji take the
   kana band**.
3. **The one structural signal is K_lo's tolerance of the higher band**:
   exact 6 → 7 where kana fell 10 → 3, native 29 → 9 where kana fell 42 →
   9, and the largest leverage on both arms (0.34 / 0.26 vs kana 0.18 /
   0.06). This is the C.1 read again (K_lo peaks one step above kana at the
   same ink) — simple straight-stroke kanji carry their identity higher in
   σ than a kana of equal ink. It is not a band preference (K_lo still reads
   best at 0.7–0.9), so it does not change a recipe; it is the reason the
   straightness read (§ Straightness) exists.
4. **Dense kanji at 48 px in a scene: 0 of 64 native on both arms**, with
   exact 3/12 on `bk_mid` whose misses are near-neighbour swaps (願 → 顔,
   聞 → 闘, 精 → 樺) and whose native renders draw a dense-kanji *texture*
   of the right size and weight but never the glyph (`sheet_願_en.png`).
   At 190 steps a row, scene-only, the row has found the neighbourhood and
   not the identity. This is not a band result (both arms agree); it is
   the cell the C.1 amplitude (K_hi at half of K_lo) and `step1_0921`'s
   own kanji split (ink < 9: 9/12, ink ≥ 11: 4/10 exact) pointed at, and
   it separates two causes that this run cannot: **exposure** (more draws)
   vs **px** (`step1_0921` bought its kanji identity on 85–200 px grid
   cells; a 14-stroke glyph at 48 px is 6 latent cells wide). The next
   cell is block K on the step-1a recipe (grid 50 %, 0.7–0.9, same budget)
   — if K_hi wakes there, dense kanji are a px question and `stage0709`
   needs grid cells for them; if not, an exposure one.

Against the plan's expectations: the ceiling's "K_hi at 48 px = Black 64
px Latin = peak 0.7" was right about the σ (everything peaks at 0.7 here)
and wrong about what ink does — it lowers leverage rather than raising the
band. `bk_lo` (0.5–0.7) stays unrun: nothing in these reads says a stratum
wants to go *down* either (K_hi's `bk_mid` curve is centred at 0.7 like the
others), and the singles rule from B.1 holds for all three.

## Straightness — small, real, and the smaller of two terms

The K_lo signal is not ink (§ Verdict 3), and among the C.1 descriptors
(fill, coarse share, stroke density) K_lo is the *sparsest* stratum while
peaking highest, so no density scalar predicts it. `common.render.ink.
glyph_features` adds **straightness** = 1 − normalised entropy of the
magnitude-weighted gradient-orientation histogram (straight strokes at any
angle concentrate it, curves spread it; kana 0.06, K_lo 0.31, K_hi 0.27,
Latin 0.27 at 48 px Regular, against fill 0.20 / 0.15 / 0.30 / 0.27 — so
separated from density) and an axis share (± 10° of horizontal /
vertical). Every cf_sense item now records both; the report bins by
straightness tercile.

Two runs on `rows_step1_0921_s30k`, flat, Regular, 48 px, σ 0.35 … 0.9:
`kanji:200` in ink terciles, 96 id pairs (job `20260923-163457-00d5c9`,
6 min) and the trained kana, 48 pairs (`…-4cc5c1`, 3 min). Ruler per pair:
the **soft peak** (move-weighted mean σ over the live part; the 0.1-grid
argmax is too coarse for a 0.03 effect) and the amplitude.

| kanji:200 by straightness quartile | straight | ink | soft peak σ | amp |
|---|---|---|---|---|
| Q1 curviest (楽 法 先 待 愛 …) | 0.18 | 9.8 | 0.624 | 0.20 |
| Q2 | 0.23 | 10.4 | 0.622 | 0.22 |
| Q3 | 0.28 | 9.4 | 0.633 | 0.25 |
| Q4 straightest (日 相 少 言 自 …) | 0.35 | 8.2 | **0.658** | 0.26 |

| per-pair correlation | kanji:200 (96) | kana (48) | pooled (144) |
|---|---|---|---|
| straightness → soft peak | +0.33 | +0.16 | +0.02 |
| ink → soft peak | **−0.44** | −0.25 | **−0.46** |
| straightness → soft peak, ink held | +0.20 | +0.06 | +0.19 |
| ink → soft peak, straightness held | −0.37 | −0.20 | **−0.49** |
| ink → amplitude | −0.23 | −0.15 | −0.40 |
| axis share → anything | ≤ 0.13 | ≤ 0.17 | ≤ 0.27 (negative) |

Reads:

1. **Ink is the larger term and its sign is the opposite of H-ink**: at
   fixed px, more ink per glyph puts the leverage *lower* in σ (−0.45) and
   makes it smaller (−0.4). Dense glyphs are decided later, with less
   caption leverage — the C.1 / C.2 picture, now over 200 kanji.
2. **Straightness is real, independent of ink, and small**: +0.2 partial,
   +0.03 σ from the curviest to the straightest quartile. It is the K_lo
   effect (K_lo = the straight, sparse kanji: 0.31 / ink 6.8) and it is
   why K_lo sat one grid step above kana in C.1 — the two terms add there
   (straightest *and* least ink). Axis alignment adds nothing; angle
   concentration is what counts, not horizontality.
3. **Neither is a band term.** +0.03 σ typical, +0.1 at the extreme of the
   two-term sum, all inside 0.7–0.9 for singles. `windows.py` stays keyed
   on px and count. Ink enters the recipe only as a budget weight (dense
   rows need more draws, or the px cell in `band_experiment_results.md`
   § 6 item 3), straightness not at all.
