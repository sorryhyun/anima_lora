# status — where the reseed line stands (2026-10-09)

Folds `progress.md` (10-07), `plan.md`, `idea.md` / `idea2.md` / `idea3.md`,
`structure_candidate.md` and `task_report.md` § 2 / § 4, all now in
`_archive/`. Every number is from a dated report (`reports/`,
`_archive/reports/`) or a ruler result (`results/`).

**Freeze point.** The tree before the prune is `b3dee68e`
(`b3dee68ecdb189009a5bd5fd8f26c14e97948733`, `cjk-reseed`). The archived
configs, probes and `stick_fit.py` run from a checkout of that commit.

## 1. Recipe of record and lineage

| step | what | where |
|---|---|---|
| punct pack | `punct_pack.py`: the raw pack + wider folds, dot runs, a `…` row | `models/vocab_packs/anima_cjk_vocab_pack_punct` |
| `punct` | the six mark rows `～ … ♡ ♥ 、 。` cold, every other row at `seed_retrain_0930` (`configs/punct.toml`, `reports/ruler_2026_10_05.md`) | `output/cjk_anima_reseed/punct` |
| `seed_fixed_1005` | `seed_retrain_0930` + punct's six mark rows, on the punct pack (user 10-05) | `output/cjk_anima_scale/seed_fixed_1005` |
| preview51 | each family's mean × 0.8 (kana 167, kanji 1 185), the mark rows unscaled | `output/cjk_anima_scale/seed_fixed_1005_stick080` |
| `sent_kanji_pres` | whole warm from preview51, every row free, free_residual 0, lr 2e-4, λ 10 · L_pres at σ 0.8–0.9 every 2nd step, data = `sent_kanji`'s build (`configs/sent_kanji_pres.toml`) | `output/cjk_anima_reseed/sent_kanji_pres` |
| the 225 cold (stage A) | scale's `retrain_kanji_b5`: the 225 new kanji cold on preview51, ink budget (158 at 337.5 / row, 67 at 225) | `output/cjk_anima_scale/retrain_kanji_b5{,_stick080}` |
| the 225 warm (stage B) | `sent_kanji_225`: pres's recipe over every row, `focus` = the 225, 9 000 steps; only the 225 are kept (`configs/sent_kanji_225.toml`) | `output/cjk_anima_reseed/sent_kanji_225` |
| `seed_1008` = jp_v1 | `transplant.py write`: pres's 2 686 ids + the 225 at B's values; baked on the punct pack, `--glyph_route` (sha `ce0ec15f7168…`) | `output/cjk_anima_reseed/seed_1008`, `models/vocab_packs/anima_cjk_vocab_pack_{seed_1008,jp_v1}` |

- **Why two stages for new kanji** (`_archive/plan.md` § 3): reseed's warm
  recipe (40 steps / row at 2e-4, norm pull off) cannot make a cold row; the
  scale recipe's ink budget (C3: 225 / row gave identity, 450 restored the
  dense ones) can. The warm pass is what moves dialogue.
- **The transplant read** (`transplant.py stick`): the 225's mean sits 12.2°
  off pres's kanji stick (16.0° before B), length × 1.02 (inside the random
  225-subset range); adding pres's Δstick turns it under a degree and makes it
  × 1.12 long, so the plain transplant shipped.
- **The 225 on the ruler** (`results/20261008-2025-ruler-sensitive-seed_1008/`):
  every read n.s. against pres on all 96 and the 42 unseen. No ruler string
  holds one of the 225, so the ruler says they cost pres nothing; whether they
  draw is unread. The new-kanji string set (`_archive/plan.md` § 4, strings in
  `$MANGA109S/derived/b5_held.tsv`) was not built (user 10-08: by eye in
  ComfyUI).
- **Session noise.** 95 / 96 of seed_1008's renders differ from pres's cached
  ones on rows equal on every id pres holds: mean |Δ| median 3.8 / 255, max
  18.9.

## 2. The ruler and the standing table

`criteria.md` (built: `reports/ruler_2026_10_05.md`). 96 strings of the
training set's bubble dialogue, 32 each short (2–4) / mid (5–9) / long
(10–20), one render each against a hand-written EN reference on the same
prompt, read paired per string. Text headline: glyph F1 (`g_f1`); page:
`en_match`, `en_tok_out`.

`results/20261006-1502-ruler-sensitive-glyph_f1/`; preview51 =
`seed_fixed_1005_stick080@punct`; the old floor = `seed_retrain_0930`.

| arm | g_f1 | g_p | g_r | g_r_kanji | drawn | a_p | exact | cer | en_match |
|---|---|---|---|---|---|---|---|---|---|
| retrain_kana | 0.317 | 0.264 | 0.620 | 0.086 | 29.5 | 0.410 | 12 | 0.595 | 0.328 |
| seed_retrain_0930 | **0.319** | 0.257 | 0.680 | 0.328 | 32.8 | 0.422 | **13** | **0.560** | 0.360 |
| preview51 | 0.241 | 0.170 | 0.635 | 0.193 | 45.6 | 0.227 | 9 | 0.681 | 0.436 |
| sent_whole | 0.260 | 0.193 | 0.647 | 0.167 | 39.8 | 0.266 | 6 | 0.664 | 0.437 |
| sent_kanji_f0 | 0.271 | 0.202 | 0.670 | **0.336** | 37.3 | 0.318 | 3 | 0.689 | 0.436 |
| sent_kanji_pres ¹ | 0.270 | 0.224 | 0.611 | 0.294 | 35.2 | 0.270 | 4 | 0.676 | **0.533** |
| **seed_1008** ² | 0.269 | 0.219 | 0.629 | 0.280 | — | — | 5 | 0.677 | — |

¹ `results/20261007-2048-ruler-sensitive-sent_kanji_pres/`.
² `results/20261008-2025-ruler-sensitive-seed_1008/`; le2 12 (pres 13).

| pair (mean Δ, better / worse, p) | g_f1 | drawn | en_match | cer |
|---|---|---|---|---|
| f0 vs preview51 | +0.030, 60 / 34, **0.0095** | −8.3, 0.011 | −0.001, 0.66 | +0.009, 1.0 |
| preview51 vs seed_retrain_0930 | −0.079, 21 / 74, **4e-8** | +12.8, 2e-6 | +0.076, **5e-5** | +0.121, 1e-4 |
| f0 vs seed_retrain_0930 | −0.048, 43 / 53, 0.36 | +4.5, 0.04 | +0.075, **0.001** | +0.129, 8e-4 |
| pres vs preview51 | +0.029, 61 / 32, **0.0035** | −10.4, 2e-5 | +0.097, **5e-8** | −0.005, 0.51 |

- **preview51 over the old floor** won the page (scene kept, fewer banners and
  pastes, the green leaf gone) and paid in text: 13 more glyphs a page, mostly
  junk.
- **f0** takes back two thirds of those letters, F1 no longer separates from
  the old floor, preview51's page kept; kanji recall 0.336 (preview51 0.193).
- **pres** = f0 + L_pres keeps f0's text (cer, le2, g_f1 n.s.) and lifts every
  page read (`en_match` +0.097, `en_tok_out` +0.041, `iou_en` +0.075,
  p ≤ 5e-7); it shrinks the text a third and leans kana recall down (−0.066,
  p 0.09) (`reports/sent_kanji_pres_2026_10_07.md`).
- **Not won.** cer is still the old floor's (f0 +0.129, p 8e-4); short kana
  exact falls with every warm pass (preview51 8, sent_whole 5, f0 3 of 46
  kana-only strings); long within 2 edits 0 / 32 for every arm; long recall
  flat (g_r 0.56–0.59 for every arm).

## 3. Modes settled

A family's rows split as **stick** (the mean row) + **spikes / ball** (rows
less it).

- **Cold never beat the old seed.** Every cold kana table read under
  `retrain_kana` on the banner grid and on the dialogue ruler
  (`reports/ruler_2026_10_05.md` § 3–4).
- **Stick re-fits lose the word fit** (`stick_rk`); on dialogue the stick
  alone (`sent_stick`) runs 41° and loses long strings (cer +0.061, p 0.009).
- **Any warm ball turn costs short strings as a random turn does**:
  `sent_ball` exact 1 / 8 vs preview51 (p 0.039), the same at lr 2e-4, and
  `rand_turn` (the same turn sizes, random directions) costs the same
  (`_archive/reports/sent_ball_2026_10_05.md` § 6).
- **Whole-warm is the mode**: freeing the stick wins back what the spike turn
  cost (`sent_whole` ties preview51, cer −0.017, p 0.3).
- **Any warm run with rare rows needs `free_residual = 0`**: the norm pull
  under AdamW walks a row absent from the batch back to the pack row (rare
  kanji ended at length 0.008); with it off, row cos 0.954 and 1 185 / 1 185
  nearest their start (`reports/sent_kanji_2026_10_06.md` § 2).

## 4. Banner-era verdicts (10-02 – 10-04)

Read on the banner grid, not on dialogue; reports in `_archive/reports/`.

- An item trained above where its glyphs resolve teaches layout only
  (`grad_identity`); the gradient's own bands tie the band law's
  (`grad_bands`).
- Identity does not need large glyphs, but cold small-glyph grids read under
  retrain_kana (`grid_small_lone`); a 65 px tier ties (`grid_64`).
- Upper edges + 0.1 buy banner words and cost the scene, one layout move
  (`kana_up`, `probe_split`).
- Stick re-fits move nothing on kana_up's rows and lose the word fit on
  retrain_kana's (`stick_fit`, `stick_scene`, `stick_rk`).
- The `japanese text` tag dropped half the time loses words
  (`stick_rk_jt50`).

## 5. Probes that closed with no lever

Reports and probe code in `_archive/`.

- **probe_geom** (10-06): box weighting is inert on row geometry.
- **probe_cf** (10-06): counterfactual-input FM leaves the ∂v/∂e scatter.
- **probe_scene** (10-06): scene diversity is not the scatter.
- **probe_accum** (10-06): per-row accumulation ≈ plain AdamW.
- **idea2 / probe_jl** (10-07) and its PE lens (10-08): stopped. No row
  direction moves the text box without the page at σ 0.8–0.9 (cross-fit λ
  ~1–2 there), in latent cells or in PE features; A / B fits unstable.
- **idea3 / probe_twin** (10-08): stopped. Glyph identity has no directions
  of its own (cross-fit λ / r 0.96–1.28); the neighbours' rows turn 0.6–0.9
  as much as the row's own.

## 6. Not tried

From `_archive/idea.md` (10-05 review; its dialogue-length data became the
`sent` tiers):

- Per-occurrence credit: a glyph's local loss backpropagated only through
  its own occurrence's embedding lookup.
- An OCR / spelling loss on decoded x̂₀ text crops.
- A quote-scoped sequence adapter over the quoted string's adapter outputs
  (needs a runtime component; not a static pack).
- The new-kanji string set (§ 1).

## 7. Reading a row

From `_archive/structure_candidate.md` (on `sent_kanji_f0`, kana / kanji).
`trained.pt` `delta` is an **offset**: offset = `raw × row_scale`, and the
row the model sees = pack row + offset.

```
row_i = m_pack + s + (1 − α_i)·q_i + e_i
```

| term | what | size (kana / kanji) |
|---|---|---|
| `m_pack` | the family's mean pack row, frozen; ≈ T5's mean (cos 0.72 / 0.81) | \|99\| / \|101\| |
| `s` | the trained stick = the family's mean offset; a free bias, ⟂ `m_pack` | \|121\| / \|127\| |
| `q_i` | the row's pack spike (pack row − `m_pack`): Qwen semantics | \|159\| / \|183\| |
| `α_i` | how much training shrank `q_i` (uniform, done in the cold seed) | 0.38 / 0.32 |
| `e_i` | the offset spike with `q̂_i` projected out: **the glyph** | ≈ \|209\| / \|211\| |

- Norms on the whole row (an origin-centred shell, |row| 275 / 289, cv
  0.05 / 0.07); glyph identity on `e_i` (bitmap Spearman 0.28 / 0.18, kana NN
  rank 1); meaning on `q_i` (meaning pairs +0.21, look-alikes +0.09).
  Uncentred row cos is mostly `s` and `m_pack`.
- Taking out `m_pack` or swapping `q_i` breaks the render (7 / 7); projecting
  `q_i` out keeps composition (7 / 7, by eye).
- A new cold row inherits `m_pack` and `q_i` from its pack row and `s` from
  its family; only `e_i` has to be learned.

## 8. KO / ZH

`reports/kozh16_2026_10_09.md` (8 Hangul + 8 hanzi cold beside seed_1008):
each script grows its own stick ~57° off both seed sticks, its spikes sit
outside both seed balls (0.09–0.12 of their energy in the seed balls' top-40
vs 0.28–0.30 held-out); hanzi 6 / 8 exact, Hangul 2 / 8 exact with the other
6 one jamo off. Next: `proposal_jamo.md`.

From `_archive/task_report.md` § 2 / § 4:

- **seed_1008's balls**: sticks |kana| 119 / |kanji| 125, cos 0.757; ball
  radius 209 / 211; kanji spikes in kana's top-40 0.107 (kana half → other
  half 0.237, random 0.039).
- **Fonts**: the 12 OFL faces under `kozh/` (FONTS.md), kept out of the
  top-level dir because several ZH faces cover kana. `TanukiMagic.ttf` maps
  你 to an empty outline: any run drawing 你 drops that face.
- **Windows** need every glyph to be a row, so 8 Hangul make no window from
  any corpus; the row set must cover common syllables first. Corpus
  candidates: LCCC (ZH, MIT), songys Chatbot_data (KO, MIT, 11.8 k pairs),
  OpenSubtitles v2018 ko / zh_cn (unclear copyright), SmileStyle (KO,
  CC BY-NC). The JA pool format is `line\tbook\tn_pieces`.
- **Captions**: `korean text` / `chinese text` and `Korean / Chinese text
  reads as`, not the JA words.
- **`个` is in Shift-JIS**, so script detection by encoding mislabels it; the
  language is named per row (`lang`).
- **Bake with `--base …_punct`**: without it `bake_vocab_pack.py` takes the
  configured `jp_v1`.

## Open

- **The short kana loss.** Every warm pass loses short kana strings; a run
  with the kana held at preview51 and only the kanji free would split whether
  the kana rows or the kanji under the same lines cost them.
- **Long strings.** No arm moves long recall; the data is 70 % 8–14 cell
  lines and the ruler's long bin is 10–20 glyphs.
- **Horizontal text** is 6.8 % of the multi-glyph items and the ruler holds no
  horizontal read.
- **The ruler's strings are not held out of the window pool**
  (`criteria.md`).
- **Release remainder**: the 225 unread (§ 1).
