# progress — where the reseed line stands (2026-10-07)

How rows are trained, the data, what an arm is judged on, and the best arm
against what ships (preview51). Every number is from a dated report
(`reports/`, `_archive/reports/`) or a ruler result (`results/`).

**Now.** `sent_kanji_f0` — preview51's kana and kanji rows warm, every row
free, lr 2e-4, the norm pull off, 70 % dialogue lines — is the line's best
arm. Against preview51 it letters less junk and more of the string (glyph
F1 0.271 vs 0.241, p 0.0095), draws the target kanji far more often (kanji
recall 0.336 vs 0.193, p 0.015), with the page unchanged. It pays on short
kana strings (exact 8 → 3 of 46). Long strings within 2 edits stay 0 / 32
for every arm.

**10-07.** `sent_kanji_pres` = f0 + λ 10 · L_pres at σ 0.8–0.9 every 2nd
step keeps f0's text (cer, le2, g_f1 n.s.) and lifts the page on every page
read (`en_match` +0.097, `en_tok_out` +0.041, `iou_en` +0.075, p ≤ 5e-7),
the gain held where the text area did; it shrinks the text a third and
leans kana recall down (−0.066, p 0.09)
(`reports/sent_kanji_pres_2026_10_07.md`).

## 1. How the rows are trained

A table's rows (one per glyph) are split as **stick** = the family's mean
row and **spikes / ball** = each row less that mean
(`_archive/reports/stick_2026_10_03.md`: cold rows are a burr on a stick,
the stick 29–34 % of the delta energy, the spikes near-orthogonal). What a
run trains:

| mode | moves | held | run key | arms |
|---|---|---|---|---|
| **cold** | every row, from the seed's | — | `seed` | kana, kana_mix, kana_up, kana_big |
| **stick** | the mean only | the spikes | `stick_from` (+ `rows_from`) | stick_*, stick_rk_*, sent_stick |
| **ball** | the spikes | the mean, put back each step | `ball_on` (+ `warm`) | ball_rk*, sent_ball, sent_ball_lr2 |
| **whole** (warm) | every row, from shipped rows | — | `rows_from` alone | sent_whole, sent_kanji, sent_kanji_f0 |

What each mode settled:

- **Cold never beat the old seed.** Every cold kana table (banner era,
  10-02 – 10-04) read under `retrain_kana`; on the dialogue ruler none beat
  it either (`reports/ruler_2026_10_05.md` § 3–4). kana_up's +0.1 upper edge
  won the banner grid and lost on dialogue.
- **Stick.** On retrain_kana's rows the stick carries the word fit and any
  re-fit loses it (`stick_rk`); on dialogue, the stick alone (`sent_stick`)
  runs 41° and loses long strings (cer +0.061, p 0.009). Stick moves cost
  short strings far less than spike turns of the same size.
- **Ball.** The ball is learned in the low band and a ball can be moved
  between sticks (`gs_rkstick`), but every warm ball pass turned the spikes
  and lost the short strings: `sent_ball` exact 1 / 8 vs preview51
  (p 0.039); at lr 2e-4 the same; a **random** turn of the same size
  (`rand_turn`) costs the same — the ball run's gradient carries no reading
  signal at that size (`reports/sent_ball_2026_10_05.md` § 6).
- **Whole.** Freeing the stick wins back most of what the spike turn cost:
  `sent_whole` ties preview51 (cer −0.017, p 0.3). This is the mode in use.
- **The kanji free too** (`sent_kanji`, 1 348 rows, kana step × 0.12 by
  `row_lr`). The trainer's norm pull (`FREE_RESIDUAL` 1e-3 · ‖f‖²) under
  AdamW walks a row absent from the batch back to the pack row: the rare
  kanji ended at length 0.008. **`free_residual = 0`** (`sent_kanji_f0`)
  holds them (row cos 0.954, 1 185 / 1 185 nearest their start). Any warm run
  with rare rows needs the pull off (`reports/sent_kanji_2026_10_06.md` § 2).

Recipe of record (`configs/sent_kanji_f0.toml`): rows from
`seed_fixed_1005_stick080` (= preview51), the punct pack, 40 steps / row
(53 920 steps × batch 4), lr 2e-4 cosine, μ 0, free_residual 0, kana
row_lr 0.12.

## 2. The data

`sent_kanji`'s build (sent_kanji_f0 trains on it): 134 800 items, 1 348
rows, lines from every Manga109-s text on the pack's rows
(`~/manga109s/derived/dialogue_pack.tsv`, 82 971 lines). The grid / lone
tiers are off (share 0) since the banner era.

| tier | form | σ band | ink px (median) | share | items | horizontal | with kanji | kanji / glyphs |
|---|---|---|---|---|---|---|---|---|
| bubble1_52 | one glyph in a bubble | 0.55–0.8 | 52 | 2 % | 2 696 | — | 91 % | 91 % |
| bubble1_32 | one glyph in a bubble | 0.35–0.6 | 36 | 3 % | 4 044 | — | 86 % | 86 % |
| bubbleN_34 | 2–6 glyph window | 0.45–0.7 | 35 | 15 % | 20 220 | 29 % | 92 % | 48 % |
| bubbleN_18 | 2–6 glyph window | 0.2–0.5 | 17.5 | 10 % | 13 480 | 21 % | 93 % | 43 % |
| sent_34 | dialogue line, 2–3 columns, 8–10 cells | 0.45–0.7 | 33 | 40 % | 53 920 | 0 | 65 % | 16 % |
| sent_22 | dialogue line, 2–3 columns, 8–14 cells | 0.3–0.6 | 22 | 30 % | 40 440 | 0 | 73 % | 16 % |

- **Almost all vertical.** `sent` is tategaki by construction
  (`recipes.sent`: `vertical_only`, `tategaki`, `vert_forms`); left-to-right
  lines come only from bubbleN (`HORIZONTAL_FRAC` 0.3, on the `sl1w` scenes):
  8 759 items, 6.8 % of the multi-glyph items.
- **Kanji density is in the bubbles**, by count in the lines: windows are
  cut from the dialogue at any offset (2–6 glyphs, not at word bounds:
  `離せよパ`), so most hold a kanji; a dialogue line is ~16 % kanji. The
  kanji are 22.5 % of the trained glyph occurrences.
- By kana occurrence (sent_ball's build, kana rows only): singles 0.8 %,
  windows 16 %, dialogue lines 83 %.

## 3. What an arm is judged on

`criteria.md` (the ruler as built: `reports/ruler_2026_10_05.md`). In short:
96 strings of the training set's bubble dialogue, 32 each short (2–4 glyphs)
/ mid (5–9) / long (10–20), one render each against a hand-written EN
reference on the same prompt, read paired per string (a direction by the
sign over strings). Text headline: glyph F1 (`g_f1`, `ruler.py`
`score_page`: `g_p` the string's letters among all drawn, `g_r` the string's
letters drawn; `drawn` letters on the page, `a_p` the on-string share of the
text area); exact / ≤ 2 edits / cer stay in the tables but sit at 0–1 past a
word for every arm. Page: `en_match` (an unrelated page sits at 0.02),
`en_tok_out` beside it; by eye: paste / wipe / banner.

## 4. Progress against preview51

`results/20261006-1502-ruler-sensitive-glyph_f1/` (all 96 strings).
preview51 = `seed_fixed_1005_stick080@punct`; the old floor =
`seed_retrain_0930` (the seed before the punct fix and the × 0.8 stick).

| arm | g_f1 | g_p | g_r | g_r_kanji | drawn | a_p | exact | cer | en_match |
|---|---|---|---|---|---|---|---|---|---|
| retrain_kana | 0.317 | 0.264 | 0.620 | 0.086 | 29.5 | 0.410 | 12 | 0.595 | 0.328 |
| seed_retrain_0930 | **0.319** | 0.257 | 0.680 | 0.328 | 32.8 | 0.422 | **13** | **0.560** | 0.360 |
| preview51 | 0.241 | 0.170 | 0.635 | 0.193 | 45.6 | 0.227 | 9 | 0.681 | 0.436 |
| sent_whole | 0.260 | 0.193 | 0.647 | 0.167 | 39.8 | 0.266 | 6 | 0.664 | 0.437 |
| **sent_kanji_f0** | 0.271 | 0.202 | 0.670 | **0.336** | 37.3 | 0.318 | 3 | 0.689 | 0.436 |
| sent_kanji_pres ¹ | 0.270 | 0.224 | 0.611 | 0.294 | 35.2 | 0.270 | 4 | 0.676 | **0.533** |

Paired (mean Δ, better / worse strings, p):

| pair | g_f1 | drawn | en_match | cer |
|---|---|---|---|---|
| f0 vs preview51 | +0.030, 60 / 34, **0.0095** | −8.3, 0.011 | −0.001, 0.66 | +0.009, 1.0 |
| preview51 vs seed_retrain_0930 | −0.079, 21 / 74, **4e-8** | +12.8, 2e-6 | +0.076, **5e-5** | +0.121, 1e-4 |
| f0 vs seed_retrain_0930 | −0.048, 43 / 53, 0.36 | +4.5, 0.04 | +0.075, **0.001** | +0.129, 8e-4 |
| pres ¹ vs preview51 | +0.029, 61 / 32, **0.0035** | −10.4, 2e-5 | +0.097, **5e-8** | −0.005, 0.51 |

¹ `results/20261007-2048-ruler-sensitive-sent_kanji_pres/` (same renders
for the other arms, from cache).

- **What preview51 bought and paid.** Over the old floor it won the page
  (en_match +0.076: the scene kept, fewer banners and pastes, the green leaf
  gone) and paid in text: it letters 13 more glyphs a page, mostly junk
  (F1 −0.079).
- **What the dialogue arms win back.** f0 takes back two thirds of those
  extra letters (drawn −8.3 of the +12.8) and lifts F1 to where it no longer separates from the old
  floor (p 0.36) — **with preview51's page kept** (en_match +0.075 over the
  floor). On long pages F1 0.331 vs the floor's 0.335 (preview51 0.261).
- **Kanji.** f0's kanji recall 0.336 (preview51 0.193, sent_whole 0.167);
  the kanji rows trained with the kana add it (f0 vs sent_whole +0.169,
  p 0.023).
- **Not yet won.** cer is still the old floor's loss (+0.129, p 8e-4); short
  kana exact falls with every warm pass (preview51 8, sent_whole 5, f0 3 of
  the 46 kana-only strings); long within 2 edits 0 / 32 for every table,
  and long recall does not move (g_r 0.56–0.59 for every arm): the pages
  letter less junk, not more of the string.

## 5. Settled in the banner era (10-02 – 10-04)

Read on the banner grid (`probe_split`'s plain read), not on dialogue —
verdicts about the banner. Reports in `_archive/reports/`.

- σ bands: an item trained above where its glyphs resolve teaches layout
  only (`grad_identity`); the gradient's own bands tie the band law's
  (`grad_bands`); no lower edge loses.
- Identity does not need large glyphs, but cold small-glyph grids read under
  retrain_kana (`grid_small_lone`); a 65 px tier ties (`grid_64`).
- Upper edges + 0.1 buys banner words and costs the scene (`kana_up`); the
  gain and the cost are one layout move (`probe_split`).
- Stick re-fits move nothing on kana_up's rows and lose the word fit on
  retrain_kana's (`stick_fit`, `stick_scene`, `stick_rk`, `stick_rk_jt50`).
- The `japanese text` tag dropped half the time loses words (`stick_rk_jt50`).

## Open

- **The short kana loss.** Every warm pass loses short kana strings; a run
  with the kana held at preview51 and only the kanji free splits whether the
  kana rows or the kanji under the same lines cost them.
- **Long strings.** No arm moves long recall; the data is 70 % 8–14 cell
  lines and the ruler's long bin is 10–20 glyphs.
- **Horizontal text** is 6.8 % of the multi-glyph items and the ruler holds
  no horizontal read; whatever f0 learned is vertical.
- The ruler's strings are not held out of the window pool (`criteria.md`).
