# findings — where the text is decided on the trajectory (2026-10-01)

Five no-training reads on the base and the seed rows (`seed_retrain_0930`,
routed), all through `experiments/sigma_split` on the `sent` grid (4 prompts ×
2 seeds, 512², 28 steps, cfg 4, flow shift 3). They started as
`proposal_length.md` step 0 and ended up questioning the band table the seed
was trained on (`builder.TABLE` b0507 / b0305, `windows.py`'s 12–24 px rows).

## Verdict

- **Glyph identity is decided at σ 0.85–0.7 on the generation trajectory,
  small text included, when the model has an identity for the glyphs.** The
  base writing English at ≈ 12–18 px in a speech bubble reads 61 % of the
  words at σ 0.85, 80 % at 0.75, 83 % at the end (§ 4).
- **The base's own Japanese garble is the trajectory of a model with no JA
  identity, not a property of small text.** Its columns are placed by 0.69;
  the strokes — real kana and kanji in nonsense order — keep changing until
  ≈ 0.33 and follow x_t, not the caption (`reports/sigma_split_2026_09_30.md`).
- **So b0305 (0.3–0.5) and b0507 (0.5–0.7) train where identity is already
  set at inference.** Neither has a trajectory read behind it: b0305's
  12–24 px row is ceiling-only (`cf_band_a1`, teacher-forced on a noised clean
  render), and b0507 for words was borrowed from the piece read
  (`micro_cf_0922`), which per-glyph routing no longer uses (こんにちは encodes
  to five single rows). § 8 qualifies this for the seed's small JA: with the
  conditional dropped at 0.8, its b0305 strings do not survive.
- **The `japanese text` tag sets a text area beyond the quoted word.** With it,
  a 2-glyph word gets the long banner or a second garble line; without it, the
  line shrinks to the word (§ 3). It does not set the one leftover slot of a
  5-glyph word.
- **With the seed rows only above σ 0.8, layout and slots come out as the
  floor's and a third of the strings survive (§ 5).** The large glyphs are
  committed above 0.8; smaller ones are rewritten by the base below it.
  Identity needs the rows across 0.9–0.7.
- **The repeats are the base banner's ≈ 6 slots minus the word.** A caption
  that fills them reads better: こんにちはは 4 / 8 vs こんにちは 2 / 8, and
  こんにちは！ (！ folds to base T5) 5 / 8 with 7 / 8 free of repeats (§ 1–2).

- **The σ that sets a glyph follows its rendered size, not the script.** EN
  words in a 3 × 3 grid read by 0.85–0.75 on the base (§ 6); the seed's
  trained singles in the same grid (≈ 100 px) are set by 0.9 and survive the
  conditional dropped at 0.8, bar dakuten / handakuten (§ 7); the seed's own
  b0305 captions lose every string when it is dropped at 0.8 (§ 8). Slot count
  and layout are set at σ ≥ 0.9 in all three.

Not yet a band change: the rows are one vector at every σ, so what a low band
teaches still acts at high σ. Which band to train in is a training read; these
reads say where the rows have to act.

## 1. Span vs the caption's glyph count (`--span`)

The same (prompt, seed) under はい / やったネ / やったネネ / こんにちは /
こんにちはは, seed rows both sides, x̂0 at σ 0.95 / 0.9 / 0.85 / 0. Job
`20261001-143001-53e843`, `results/20261001-1433-span/`.

- No number reads the span at σ 0.95. The OCR box is 0 at 0.95 in every traj
  run (nothing boxes a blur), and the x̂0 pixel difference between two
  captions covers 86–99 % of the canvas from 0.95 on (a caption change moves
  the whole scene). Read on the sheets.
- In most samples the banner has the same width for 2, 4, 5 and 6 glyphs.
  はい, final image of 8: 4 fill a full-width banner with ≈ 6 glyphs
  (`はにはにきい`, `麦どはもいい`, `はまんはない`), 3 write a short はい plus a
  separate garble line, 1 writes はい alone.
- Between 4 and 6 glyphs the glyph size adapts (やったネ comes out larger in
  the same width); at 2 it does not.
- こんにちは repeats a glyph in 6 / 8 (`こんにちちは`, `こにちちは`).
  こんにちはは reads 4 / 8 official against こんにちは's 2 / 8: the caption
  that matches the slots reads better.
- やったネネ never reads doubled: the second ネ becomes ト or 木 (0 / 8).

## 2. ！ in the leftover slot

Same grid, captions こんにちは！ and やったネ！. The pack's encode fold sends ！
to base T5 (`!"` is one stock piece), so the slot is the base's to fill; ネ
keeps its row (checked with the pack encoder). Job `20261001-144937-fbc22f`.
Final images, by eye:

| | こんにちは | こんにちは！ | やったネ | やったネ！ |
|---|---|---|---|---|
| a glyph of the word repeated | 6 | 1 | 0 | 3 |
| clean word (+ ！) | 2 | 4 | 6 | 2 |
| official (sfx ∧ VL, punctuation stripped) | 2 | 5 | 5 | 0 |

- こんにちは！: 6 slots. The repeats go; 2 renders put a smudge between は and
  ！, 1 writes ごんにちは！, 1 keeps こんにちはは!. The separate garble lines of
  the plain caption are gone too.
- やったネ！: 5 slots, one still left over. It goes to a repeat at the front
  (`ややった`, `やっった`), and the glyph before ！ loses its identity
  (ネ → ホ / え / な / ヌ in 6 / 8 vs 2 / 8 plain). Not tokenization; the cause
  is open.
- The readers strip punctuation and the sfx reader drops ！, so whether the
  sixth slot holds ！ is a sheet read.

## 3. The `japanese text` tag (`--span_caption notag`)

`{p}, japanese text. Japanese text reads as "X".` → `{p}. Text reads as "X".`,
same seven captions. Job `20261001-145852-2c3dc5`.

| | with the tag | without |
|---|---|---|
| はい: banner filled with ≈ 6 glyphs | 4 | 0 |
| はい: separate garble line | 4 | 0 |
| はい official | 3 | 7 |
| はい text box (share of canvas) | 0.130 | 0.030 |
| こんにちは: a glyph repeated | 6 | 6 |
| official, 7 captions summed | 19 | 21 |
| ≤ 1 edit, 7 captions summed | 37 | 40 |

- Without the tag はい's line is the word's length in 8 / 8, with smaller
  glyphs; the scene composition moves too.
- The 5-glyph leftover slot does not move (こんにちは 6 / 8 either way), and
  こんにちは！ still helps (official 6 / 8).
- Per string the signs disagree (はい +4, やったネ −2, こんにちは −1,
  こんにちはは −2, こんにちは！ +1, やったネ！ +2): no direction over the set.
- The `target` ruler's user captions carry `speech bubble, japanese text` in
  5 of 7 lines; the training captions swap `english text` → `japanese text`
  on canvases mostly drawn under EN captions (`src/data/synth.py`
  `scene_caption`).

## 4. Small English on the base (`--en_small`)

The base alone (row Δ 0), `{p}, speech bubble, english text. English text
reads as "…".` at three lengths, x̂0 at 11 σ + the final image, VL word
recall (8 renders). Job `20261001-161626-956c3d`,
`results/20261001-1620-en_small/`.

| σ | `hello` | 30 chars | 75 chars |
|---|---|---|---|
| 0.9 | 0.00 | 0.00 | 0.18 |
| 0.85 | 0.38 | 0.57 | 0.61 |
| 0.8 | 0.62 | 0.73 | 0.67 |
| 0.75 | 0.50 | 0.84 | 0.80 |
| 0.7 | 0.62 | 0.98 | 0.81 |
| 0.6 | 1.00 | 1.00 | 0.83 |
| 0 | 1.00 | 0.98 | 0.83 |

- The 75-char sentence comes out as two bubble lines or a subtitle line at
  ≈ 12–18 px (sheet estimate); one render is a 7–8 px subtitle, also formed
  at 0.85.
- σ 0.9: the bubble and the line are placed, the letters a smear. σ 0.85: most
  words read. σ 0.75–0.7: the final text bar a few letters — the final
  misspellings (`befor might`, `lister`) are fixed there too.
- `hello` finishes at 0.6: it is drawn large, and large strokes refine later.
- Against the ceiling table: `cf_band_a1` put 12–16 px EN leverage at
  0.2–0.6 by reading a noised clean render. On the trajectory the base has
  committed the text by 0.7.

## 5. Seed rows above σ 0.8, no string below (switch 0.8, `hi` / `unk` / `uncond`)

The seed rows act while σ ≥ 0.8; below it the conditional is the JA caption
with the raw pack rows (`hi`), the JA caption under the stock T5 tokenizer —
the base with no pack, the quote one `<unk>` (`unk`), or the negative
embedding (`uncond`, CFG collapses to the unconditional model). The full
`sent` grid, 184 renders per arm, paired against the floor. Job
`20261001-164156-00b8c7`, `results/20261001-1719-s08_nostring/` (`lo` /
`garble` are the 09-30 renders, re-read).

| arm (above 0.8 / below) | official | ≤ 1 edit | contained | dup | box | box h | box IoU vs EN ref |
|---|---|---|---|---|---|---|---|
| floor (seed / seed) | 29 | 92 | 64 | 100 | 0.156 | 0.180 | 0.198 |
| `hi` (seed / raw rows) | 13 (+3 / −19) | 33 | 20 | 58 | 0.152 | 0.173 | 0.198 |
| `unk` (seed / no pack) | 10 (+1 / −20) | 38 | 17 | 50 | 0.151 | 0.174 | 0.203 |
| `uncond` (seed / none) | 11 (+2 / −20) | 41 | 20 | 94 | 0.149 | 0.169 | 0.207 |
| `garble` (base garble / seed) | 16 (+11 / −24) | 61 | 47 | 123 | 0.095 | 0.253 | 0.172 |

- **Layout, slot count, glyph size, colour and position are set above 0.8.**
  `hi` / `unk` / `uncond` have the floor's placement to the third decimal, and
  on the sheets every changed string sits in the floor's slots at the floor's
  weight and colour (floor `こんにちちは` → unk `よんなさ日楸`, same six slots).
- **Identity is split across 0.8.** With no string below it, a third of the
  official reads survive (10–13 of 29, ≤ 1 edit 33–41 of 92) — the largest
  glyphs: top banners (こんにちは, カメラ) keep their string, bottom banners,
  subtitles and small lines are rewritten by the base into other kana in the
  same slots (たいせつ 0 / 8 on the peek). With the base's garble layout above
  0.8 and the rows only below, 16. Neither side alone reaches the floor.
- **The raw pack rows below 0.8 add nothing over no pack** (`hi` 13 vs `unk`
  10, `uncond` 11).
- **Part of the repeats is written below 0.8.** dup falls from 100 to 50–58
  when the rows stop at 0.8 (`hi`, `unk`): the leftover slots then take other
  kana, not the word's glyphs. `uncond` keeps 94.

Matches § 4: identity forms over 0.85–0.7, large glyphs first, so 0.8 cuts
through it. The rows have to act across 0.9–0.7, the smaller the glyph the
lower; 0.8–0.95 alone cannot carry it.

## 6. A 3 × 3 EN word grid on the base (`--en_grid`)

The base alone (row Δ 0) on grid_single's own flat caption (`white
background, …, english text. On the top left, English text reads as "SNOW".
…`, nine distinct 3–5-letter words from a 36-word pool), 8 word sets × 2
seeds, 512², x̂0 per σ, VL word recall (whole image). Job
`20261001-172804-a5096f`, `results/20261001-1731-en_grid/`.

| σ | 1.0 | 0.95 | 0.9 | 0.85 | 0.8 | 0.75 | 0.7 | 0.6 | 0.5 | 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| recall | 0.00 | 0.00 | 0.16 | 0.60 | 0.76 | 0.84 | 0.84 | 0.84 | 0.84 | 0.84 |
| all 9 words | 0 | 0 | 0 | 0 | 3 | 7 | 7 | 7 | 7 | 7 / 16 |

- The rows are placed at σ 1.0 (ink blobs in the cell rows), the slots at
  0.95. Letters form at 0.9, most words read at 0.85, the misspellings are
  fixed by 0.8–0.75; below 0.75 only colour and stroke weight change. Every
  final misspelling (`FLUE`, `FOG`, `TREY`, `APLE`) has its shape at ≈ 0.8.
- Seed 0: recall 0.99, a clean 3 × 3. Seed 1: 0.69, every miss a layout miss —
  the middle row empty or one word, the top / bottom rows 4-slot bands
  filled with a repeat (`SNOW GOLD GOLD YES`, `RAIN MUSIC MUSIC GO`,
  `FISH NIGHT NIGHT OK`), the slot count already visible at 0.95. The base's
  own EN shows § 2's leftover-slot repeat.

## 7. The seed rows on grid_single's caption (`--ja_grid [--ja_below uncond --switch 0.8]`)

The seed rows on b0709's grid_single caption, 3 × 3, flat / `multiple
speech bubbles` frame alternating, one trained single per cell: 8 kana
grids (random hiragana / katakana, no small kana or punctuation) and 8 kanji
grids (`ja_retrain_kanji_b1`, the most frequent), one seed, 512². Read per
cell (the image cut into its 3 × 3) and anywhere (whole-image reads). Jobs
`20261001-173914-2a059b` (seed rows throughout), `20261001-175750-da0707`
(seed rows above 0.8, the negative embedding below — § 5's `uncond`), same
noise.

| cell hits / 72 | 0.95 | 0.9 | 0.85 | 0.8 | 0.75 | 0.6 | final | final, uncond < 0.8 |
|---|---|---|---|---|---|---|---|---|
| kana | 37 | 55 | 61 | 63 | 68 | 67 | 64 | 61 |
| kanji | 27 | 49 | 56 | 52 | 49 | 48 | 45 | 47 |

- The grid renders as the training grids do: font-like glyphs on white, one
  per cell. The bubble frame draws no bubbles (8 / 8 a flat grid).
- Kana: 68 / 72 right by eye; the misses are shape neighbours (ガ → が,
  ク → タ, グ → ダ) and one glyph of the grid repeated (ロ → ヲ). Kanji: the
  low-stroke frequent ones are clean (力 本 水 先 火 犬 社 手 少 体); the misses
  are a similar kanji (忘 → 志, 貴 → 責 / 賁, 安 → 友, 完 → 売) or a plausible
  non-kanji of the right radicals (勝, 場, 羅, 嫌), one a seal-like box.
- A step above EN: layout and colour at σ 1.0, the glyph outline at 0.95, its
  identity by 0.9 — a wrong kanji is wrong from 0.95 and never corrected. The
  kanji read falls 56 → 45 after 0.85 with the shapes unchanged on the sheet:
  outlines and gradients added late (kanji4) cost the reader, not the glyph.
- With the conditional dropped at 0.8 the grid comes out the same — layout,
  colour, size, weight and most identities. What changes is dakuten /
  handakuten (が ↔ ガ, ペ → ベ, じ → し, ハ → パ, ダ → グ, ポ → ボ) and a few shape
  neighbours (ウ → ツ, タ → ク); as many of these flips fix a miss as make
  one, so the totals hold. The diacritic is decided below 0.8 even on a
  ≈ 100 px glyph.

## 8. The seed's b0305 items as prompts, uncond below 0.8 (`--b0305`)

Eight b0305 `scene_window` training items (12–24 px dialogue windows) —
4 from `retrain_kana`, 4 from `retrain_kanji_b4` — each at its own caption and
shape × 2 seeds: seed rows throughout vs seed rows above 0.8 and the negative
embedding below. Job `20261001-180228-5c1091` (rendered and read; it crashed
at the sheet on the pruned training `img/`, so no `result.json` — the reads
are `sigma_split_b0305/b0305_reads.json`, the sheet rebuilt on CPU).

| | official | CER (VL) |
|---|---|---|
| seed rows throughout | 7 / 16 | 0.29 |
| uncond below 0.8 | 0 / 16 | 0.80 |

- Scene, bubble / sign placement and slot count are the same in both arms.
- What is drawn large keeps most of its string: `っとおし` → `つどおし` (a
  ≈ 100 px sign), `偉いさ` → `俥いさ` / `損いさ` (the kanji alone moves).
  What is drawn small is rewritten in its slots: `んだよウチ` → `眞ようす`,
  `なアロワナ` → garble, `虚をつい` → `国七11`.
- The renders do not keep the item's px: the same caption gives a large sign
  or bubble on one seed (kana0 s0, b42) and a small line on the other.
- So for small text the seed's identity is written below 0.8 — the reading
  "b0305 trains after the commit" (Verdict) holds for small EN on the base,
  not for the seed's small JA. As in § 5, uncond drops the scene conditional
  too; `hi` / `unk` there scored the same.

**The mirror (10-02): the base above 0.8, the seed rows below.** Same 16
renders, two more arms — `garble0.8` (the item's tags with no quote clause
above the switch) and `unk0.8` (the item's caption under the stock tokenizer,
no pack). Job `20261002-095155-8eec49`, `results/20261002-0954-b0305_mirror/`,
sheet `sigma_split_b0305/sheet.png` (train item | 4 arms).

| above 0.8 / below | official | any reader exact | CER (VL) | CER ≤ 0.25 |
|---|---|---|---|---|
| seed / seed | 7 / 16 | 7 | 0.29 | 8 |
| seed / uncond | 0 | 0 | 0.80 | 0 |
| base garble / seed | 1 | 1 | 0.62 | 2 |
| no pack / seed | 0 | 2 | 0.55 | 6 |

- The split does not hold on the seed: 0–1 of 16 against 7.
- The rows below 0.8 do write the word's glyphs into the base's layout
  (CER 0.80 → 0.55–0.62), into more slots than the word has: `んだだよよウチ`,
  `んだよよよウチチ`, `なアロロナ`, `っととまおし`, `虚ををつい`. The kanji
  items read 1 of 8 (`偉いさ`, one seed).
- On the sheet the two mirror arms keep the base's small bubbles and small
  glyphs; the seed arm draws the same word large in one sign or bubble
  (§ 8 above: the renders do not keep the item's px).
- n = 16 on training captions of 3–5 glyphs; no sentence-length caption is
  read.

## Against the earlier reads

No earlier record contradicts § 1–8; where wording differs, the setup does.

- **FreeText Stage-2** (`_archive/bench/freetext/stage2_report.md`): "glyph
  strokes are resolved in the tail (σ < 0.45)" was read on Korean glyphs the
  base has no identity for — § 4's JA garble, not § 4's EN. Narrowed: strokes
  move late only when the model has no identity for the glyph.
- **Cross-attn drive** (`docs/findings/crossattn_self_attn_dominance.md`): the
  velocity reads say text only rescales below σ ≈ 0.85, but its Result 5
  measures a glyph-specific drive bump over 0.8 → 0.45 (peak 0.66) — the
  velocity side of § 5 / § 8 (dropping the conditional below 0.8 rewrites
  small text). Result 4 (the `korean text` tag keeps its attention mass, the
  glyph tokens fade) is the attention side of § 3.
- **Commitment timing**: `_archive/bench/cross_attn_drive/report.md` (bubble
  84 % committed by σ 0.8), `docs/findings/traj_stats_front_loaded_commitment.md`
  (channel profile fixed from σ ≈ 0.92; its commit-CDF counts the last
  change, colour and stroke weight included, hence 18 % at 0.8),
  `docs/findings/foveated_denoise.md` (composition locks above 0.75) — all
  consistent with layout at ≥ 0.9 and identity by 0.85–0.7.
  `docs/findings/sigma_signal_where_anima_resolves.md` ("0.95 is mostly a
  blur") is teacher-forced, not the trajectory's x̂0.
- **Ceiling vs trajectory**: `cf_band_a1` is not the first —
  `../finished/cjk_renderable_anima/reports/cf_sense_gate0_2026_09_22.md`
  (single leverage peak 0.8, two-word strings 0.5–0.7) and that line's kana
  classifier (identity ≈ 0.8, EN order / count peak 0.65) are teacher-forced
  too and peak below where the trajectory commits.
- **Repeats**: the base's EN repeat in a 3 × 3 grid (§ 6) was already in
  `../finished/cjk_renderable_anima/reports/position_probe_2026_09_20.md`
  (k = 9 shuffles: `GO GO`, `RUN RUN`). `cjk_renderable_anima/findings.md`
  read the JA repeats as the rows' count prior (singles-only → one,
  strings-only → several); both readings stay open (below).

## Open

- The JA counterpart of § 4: a long JA line in a bubble with the seed rows,
  x̂0 per σ. The seed learned its small text at 0.3–0.5; the prediction is
  no identity at 0.85–0.7 and strokes moving late.
- Where the 5-glyph leftover slot comes from, now that the tag is out: the
  base's `reads as` prior or the rows' shared count habit (the same captions
  at Δ 0). The base side has § 6 and position_probe's `GO GO`; the
  rows side has cjk_renderable_anima's singles-only / strings-only count read.
- A training read for the bands: the seed's item mix rebanded to 0.7–0.9,
  b0305's 12–24 px items out (or re-rendered larger), micro arm on the 57
  `sent` singles.

Code: `experiments/sigma_split/run_exp.py` (`--span [--span_caption notag]`,
`--en_small`, `--en_grid`, `--ja_grid [--ja_below … --switch …]`, `--b0305`,
arms `unk` / `uncond`). Renders under
`output/cjk_anima_scale/experiments/sigma_split_{span,span_notag,en_small,s0.8,en_grid,ja_grid,ja_grid_uncond0.8,b0305}/`.
