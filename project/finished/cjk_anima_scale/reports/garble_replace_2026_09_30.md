# garble_replace — the base's garble canvases, only the string replaced (2026-09-30)

`experiments/garble_replace`: `plan_garble_replace.md` § 1–2 on a 31-canvas
pool, then its § 4 warm vs cold, read on the `sent` grid against the seed's
routed floor. Results only.

**Result.**
- **warm** keeps the floor's layout, the large banner. Its string degrades:
  official 29 → 5 / 184, ≤ 1 edit 92 → 29, dup 100 → 137. Katakana and
  kanji words hold best, hiragana words fall to 0.
- **cold** writes into speech bubbles and dialogue boxes: box 0.156 → 0.065,
  en_cos at the floor's. No string reads (0 / 184 official, 0 ≤ 1 edit).
  Whatever the caption asks for, it draws long pseudo-Japanese lines.

## Pool `scenes_garble`

- Frames `ja_garble_saying` (`{pro} is saying something.`) and
  `ja_garble_tag` (no clause). Both have generals `{bubble}, japanese text`,
  no quote and no anchor.
- The judge treats them as JA frames, plus `--scene_max_regions 2`:
  overlapping boxes merge, and more than 2 regions rejects the scene as
  `multi_box`.
- Argv: `--scene_tag garble --seed 5 --scene_shapes
  384x640,448x640,448x576,512x512:0.33 --scene_frames
  ja_garble_saying,ja_garble_tag --scene_min_box 40 --scene_max_regions 2
  --scene_n 200 --scene_prune 1`. No `comic` tag. Job
  `20260930-170917-c990f5`.
- **31 / 200 kept (16 %).**

  | reason | share |
  |---|---|
  | `multi_box` | 66 % |
  | `open_bubble` | 12 % |
  | `erase_miss` | 4 % |
  | `small_box` | 2 % |
  | `bubble_leak` | 2 % |

  - By frame, `saying` kept 18 / 69 and `tag` kept 13 / 131.
  - 512² kept 6 / 22, the portrait shapes 25 / 178.
- The `multi_box` rejects mostly show 3–6 real bubbles, or a multi-panel page.
- Regions per kept scene: 1 on 13 scenes, 2 on 18. That is 49 regions, 47 of
  them vertical and 94 % taller than wide.

## Data

`output/cjk_anima_scale/run0930_garble_replace/data/`, CPU.

- **Measure.** Each region's garble ink is taken as > 80 grey from the ring
  fill.
  - Its columns are the ink runs across the box, cut where a column holds
    under 15 % of the densest column's ink.
  - px = the ink span across ÷ (1 + (k − 1) · 1.15).
  - Each column keeps its own centre and its first and last ink along.
  - Result: px min 9.8, p25 12.4, **median 14.4**, max 25.1. Columns
    k = 1: 7, 2: 25, 3: 15, 4: 2.
- **Lines.**
  - Source: `polish_seed`'s pool (manga109s, normalized, seed singles only, no
    doubled glyph, no held trigram, routed), with the length window widened
    to 2–32. That is 34 536 lines.
  - A region takes a line whose length fits its columns at pitch
    0.95–1.25 × px.
  - When fewer than 20 single lines fit, two lines are joined with `、`
    (bare after `。！？…`).
  - Lines are dealt so the least-drawn of the 57 singles comes first.
- **Render.**
  - The stage's `erase_paint` (ring-median, inside the bubble interior) on
    every region.
  - The line is split in proportion to the garble's column lengths. Column i
    sits on garble column i and runs from its first ink to its last, at the
    measured px, with no tilt.
  - Ink is the median of the garble's darkest 30 % of stroke pixels.
  - Fonts: the render set minus the faces whose 14 px ink coverage is
    under 0.14 (NotoSerif ExtraLight / Light / Regular, koyomiyuru). That
    leaves 12 faces, 27–41 regions each.
- **Caption.** The canvas prompt verbatim + one ` Japanese text reads as "X".`
  per region, right bubble first (`plan_garble_replace.md` § 2).
- **Volume.** 31 canvases × 8 lines = **248 items** (243 distinct captions),
  329 distinct lines.
  - Band 0.6–0.85 on every item.
  - Loss boxes are the drawn glyph boxes (`layout: grid` + `boxes`).
- For reference, `windows.py`'s rows put 12–24 px singles and multis at
  0.3–0.5; 0.6–0.85 is their band for 24–64 px pieces / multis and ≥ 40 px
  singles.

## Training

Both arms have the same items and the `cjk_scale/train.py` constants:
lr 1e-3 cosine, warmup 513, batch 4, box share 0.25 → 0.5 cap, routed.

- **Rows.** The 57 singles of the 23 `sent` strings. The 436 other ext rows
  the captions touch are frozen at the seed (`seed_retrain_0930`).
- **Steps.** 90 steps / row = **5 130 steps**, i.e. 20 520 draws.
  - Each item is drawn about 83 times.
  - Draws per row (items containing it × 82.7): min 1 324, median 1 655,
    mean 3 655, max 11 832 (い).
  - Distinct lines per row: median 19, 丈 only 2. Distinct canvases per row:
    median 17, 丈 6.
  - 52 % of all drawn glyphs are the 57.

| arm | start | μ | in-box loss (step 1 → end) | out-box | rows at the end |
|---|---|---|---|---|---|
| warm (job `…fe1d43`, 36 min) | seed rows | 0.1 | 0.135 → 0.1125 | 0.083 | cos to seed 0.996 (min 0.992), Δ norm 270 → 268 |
| cold (job `…728ebe`, 35.5 min) | pack rows | 0 | 0.129 → 0.103 | 0.082 | Δ norm 0 → 119 (step 1 000) → 160 (flat from ≈ 3 000; seed ≈ 268) |

## Reads — `sent`, 184 renders per arm

Grid: 23 strings × 4 prompts × 2 seeds, `en` clause, 512², 28 steps, cfg 4,
routed. Paired per render against the seed's routed floor cache. Jobs
`20260930-174716-cd94f8` (warm), `…-2f421d` (cold).

| arm | official | contained | ≤ 1 edit | ≤ 2 edit | dup | kana |
|---|---|---|---|---|---|---|
| floor | 29 | 64 | 92 | 129 | 100 | 169 |
| warm | 5 (+1 / −25, p 8e-7) | 23 (+10 / −51) | 29 (+3 / −66, p 2e-16) | 60 | 137 (+53 / −16, p 9e-6) | 171 |
| cold | 0 (−29, p 4e-9) | 0 | 0 (−92, p 4e-28) | 1 | 114 (+56 / −42, p 0.19) | 184 |

Per string (official / ≤ 1 edit / dup, of 8):

| string | floor | warm | cold |
|---|---|---|---|
| こんにちは | 2 / 6 / 6 | 0 / 1 / 7 | 0 / 0 / 4 |
| たいせつ | 0 / 5 / 2 | 0 / 0 / 7 | 0 / 0 / 7 |
| かなしい | 2 / 4 / 6 | 0 / 0 / 6 | 0 / 0 / 5 |
| かんがえ | 1 / 1 / 7 | 0 / 0 / 8 | 0 / 0 / 3 |
| たすけて | 1 / 5 / 5 | 0 / 0 / 8 | 0 / 0 / 7 |
| こうえん | 0 / 1 / 8 | 0 / 0 / 8 | 0 / 0 / 5 |
| てつだう | 1 / 2 / 5 | 0 / 0 / 8 | 0 / 0 / 4 |
| ことば | 1 / 5 / 3 | 0 / 0 / 8 | 0 / 0 / 7 |
| なにしてる | 0 / 4 / 6 | 0 / 0 / 6 | 0 / 0 / 3 |
| おしい | 1 / 7 / 7 | 0 / 2 / 7 | 0 / 0 / 7 |
| やったネ | 5 / 7 / 1 | 1 / 2 / 4 | 0 / 0 / 1 |
| ちょっと来い | 2 / 7 / 4 | 0 / 0 / 8 | 0 / 0 / 7 |
| 愛してる | 0 / 4 / 3 | 0 / 0 / 5 | 0 / 0 / 3 |
| はい | 3 / – / 3 | 1 / – / 4 | 0 / – / 5 |
| テレビ | 0 / 3 / 3 | 0 / 3 / 3 | 0 / 0 / 5 |
| カメラ | 2 / 6 / 2 | 1 / 5 / 4 | 0 / 0 / 6 |
| パソコン | 3 / 6 / 3 | 1 / 4 / 6 | 0 / 0 / 4 |
| アイドル | 2 / 6 / 5 | 1 / 4 / 6 | 0 / 0 / 5 |
| 山田太郎 | 0 / 3 / 4 | 0 / 3 / 4 | 0 / 0 / 6 |
| 小山田 | 1 / 5 / 4 | 0 / 4 / 4 | 0 / 0 / 6 |
| 日本人 | 2 / 3 / 4 | 0 / 0 / 4 | 0 / 0 / 4 |
| 大丈夫 | 0 / 0 / 7 | 0 / 0 / 6 | 0 / 0 / 4 |
| 何時間 | 0 / 2 / 2 | 0 / 1 / 6 | 0 / 0 / 6 |

- **warm:**
  - The katakana words and 山田太郎 / 小山田 keep most of their ≤ 1 edit:
    テレビ 3 → 3, カメラ 6 → 5, パソコン 6 → 4, アイドル 6 → 4,
    山田太郎 3 → 3, 小山田 5 → 4.
  - The all-hiragana words and ちょっと来い / 日本人 / 愛してる go to 0–2.
- **cold:** 0 on every string.

## Placement

Same measures as `sigma_split`.

| arm | box | box h | flat white | en_cos | en_cos_out | box IoU vs EN ref |
|---|---|---|---|---|---|---|
| floor | 0.156 | 0.180 | 0.132 | 0.922 | 0.923 | 0.198 |
| warm | 0.119 | 0.152 | 0.102 | 0.932 | 0.933 | 0.297 |
| cold | 0.065 | 0.197 | 0.129 | 0.922 | 0.922 | 0.243 |

## On the sheets

Sheets: `output/cjk_anima_scale/experiments/garble_replace_compare/`. The
overview is `sheet_overview_p0s0.png` (23 strings × p0 s0); per string,
`sheet_<string>.png` (8 renders). Columns seed | cold | warm.

- **warm:**
  - The floor's banner, in the same place and size, on nearly every render.
  - The string doubles or garbles: `おおしいし`, `かかんがが`, `ことととば`,
    `こんにちはは`, `カ・ジメラ`.
- **cold:**
  - No banner.
  - Hiragana strings get one or two vertical speech bubbles beside the
    character, filled with long pseudo-Japanese columns.
  - Katakana / kanji strings (カメラ, テレビ, パソコン, 日本人, 愛してる)
    often get a visual-novel dialogue box at the bottom, with 2–3 lines of
    small text.
  - The pose and framing move more than in the floor or warm.
  - Reads are sentence-length (`次理表締め日は`, `大囲瑠去捨…`,
    `麦奚洸んこむ併は揚いでま…`) on every string.

Results: `experiments/garble_replace/results/20260930-1743-data/` (the data
build), `…/20260930-1745-warm/`, `…/20260930-1821-cold/` (training),
`…/20260930-1857-read-warm/`, `…/20260930-1907-read-cold/`. Rows:
`output/cjk_anima_scale/experiments/garble_replace_{warm,cold}/trained.pt`.
Pool: `output/cjk_anima_scale/scenes_garble/`.

## Follow-ups (2026-09-30 – 10-01)

Four more arms, all at 5 130 steps (90 / row × 57 rows, batch 4), read on the
same `sent` grid against the same floor. Each varies one thing against a run
above.

- **quoted:** captions only; images, boxes and lines are those of `data`.
  - `ja_garble_saying` items: `… is saying something.` becomes
    `She is saying "X" "Y".`, with no `reads as` clause (the `ja_saying` form).
  - Other items: one clause, `Japanese text reads as "X" "Y".`
  - 200 of 248 captions change. A single-region tag item reads the same
    either way.
- **grid20 / grid50:** the quoted items plus 3×3 `grid_single` items over the
  same 57 singles (the b0709 tier's params: fill 0.3–0.8 of the cell, half in
  bubbles), 20 % (62) / 50 % (248) of the items. The grids train at σ 0.7–0.9
  and the garble items keep 0.6–0.85.
- **warm μ 0.02:** the `data` items (old captions, no grid), warm from the
  seed rows, anchor μ 0.02 instead of 0.1.

Data is rebuilt from `data`, not redrawn. With 破線G out of `assets/fonts/`,
a fresh `--legs data` deals other fonts and lines: only 19 of 248 items keep
their lines.

| arm | official | contained | ≤ 1 edit | ≤ 2 edit | dup | kana |
|---|---|---|---|---|---|---|
| floor | 29 | 64 | 92 | 129 | 100 | 169 |
| cold (above) | 0 | 0 | 0 | 1 | 114 | 184 |
| cold, quoted | 0 | 1 | 0 | 0 | 115 | 183 |
| cold, quoted + grid20 | 0 | 0 | 0 | 5 | 79 (+38 / −59 vs floor, p 0.04) | 175 |
| cold, quoted + grid50 | 1 | 3 | 3 (+2 / −91, p 9e-25) | 11 | 87 | 178 |
| warm μ 0.1 (above) | 5 | 23 | 29 | 60 | 137 | 171 |
| warm μ 0.02 | 3 | 27 | 24 (+6 / −74, p 5e-16) | 50 | 144 (+65 / −21, p 2e-6) | 168 |

| arm | box | box h | flat white | box IoU vs EN ref |
|---|---|---|---|---|
| floor | 0.156 | 0.180 | 0.132 | 0.198 |
| cold (above) | 0.065 | 0.197 | 0.129 | 0.243 |
| cold, quoted | 0.078 | 0.219 | 0.132 | 0.236 |
| cold, quoted + grid20 | 0.115 | 0.197 | 0.095 | 0.250 |
| cold, quoted + grid50 | 0.130 | 0.177 | 0.071 | 0.290 |
| warm μ 0.02 | 0.102 | 0.189 | 0.109 | 0.217 |

- **Captions:** no effect. The quoted cold arm matches the old cold on every
  count.
- **Grids:** the box grows toward the floor's (0.078 → 0.115 → 0.130) and dup
  drops. The ≤ 2-edit count goes 0 → 5 → 11, far below the floor's 129.
- **Sheets (`かなしい`):** every cold arm still draws the canvas's vertical
  bubbles, filled with long unrelated lines. No render gets the large centred
  word the floor draws.
- **μ:** 0.02 is not better than 0.1. The differences are inside noise at
  184 renders.

Results: `experiments/garble_replace/results/20260930-2217-cold-quoted/`,
`…/20260930-2316-mix-grid20/`, `…/20260930-2316-cold-grid20/`,
`…/20261001-0059-mix-grid50/`, `…/20261001-0059-cold-grid50/`,
`…/20261001-0008-warm-mu002/`. Data:
`output/cjk_anima_scale/run0930_garble_replace/data_{quoted,grid20,grid50}/`.
Rows: `output/cjk_anima_scale/experiments/garble_replace_{cold_quoted,cold_grid20,cold_grid50,warm_mu002}/`.

## Follow-ups (10-01): no humans, short lines, inverse frequency

Three more arms, same `sent` grid, same floor, 5 130 steps each. Each varies
one thing against an arm above.

- **grid50nh** (cold): `data_grid50`, captions only. The 137 bubble grids
  (`manga, multiple speech bubbles, …`) take `no humans` after `manga`; the
  111 flat grids already carried it. Images, order and latents are
  grid50's (`--legs no_humans`).
- **short50** (warm μ 0.02): a fresh `--legs data --short_variants 4`. Of a
  canvas's 8 draws, the last 4 take a 2–4-glyph pool line instead of a
  long one, at the garble's px, in one column centred on the garble's ink;
  the rest of the erased region stays blank. Same canvases, fonts, band and
  quoted captions.
  - 124 / 248 items short (196 regions: 2 glyphs 18, 3 60, 4 118); 5 890
    short pool lines, 195 drawn. 15 of the 196 carry `・` (`そうー・`,
    `え・と・`), kept.
  - Short lines leave out any piece of a held string and any line carrying
    a ≤ 2-glyph held string (はい: the trigram hold does not reach it).
    丈 gets no short line (its only short word is 大丈夫).
  - The long half is redrawn (破線G out, other draws), so short50 vs the
    μ 0.02 arm also differs in captions (quoted) and long lines.
- **inv_freq** (warm μ 0.02, `data_short50`): each single's update ×
  `min(1, median / items carrying it)` after every step
  (`train.train(row_step_scale=…)`). Median 16 items, max 85 (っ). 26 rows
  scale below 1: っ 0.19, な 0.21, い て ん 0.23, か 0.24, し 0.25, …; kanji
  and most katakana stay 1. It is the update that is scaled, not the
  gradient: AdamW normalizes a per-row gradient scale away.

| arm | official | contained | ≤ 1 edit | ≤ 2 edit | dup | kana |
|---|---|---|---|---|---|---|
| floor | 29 | 64 | 92 | 129 | 100 | 169 |
| cold, quoted + grid50 (above) | 1 | 3 | 3 | 11 | 87 | 178 |
| cold, quoted + grid50nh | 1 | 3 | 2 | 10 | 70 | 180 |
| warm μ 0.02 (above) | 3 | 27 | 24 | 50 | 144 | 168 |
| warm μ 0.02, short50 | 2 | 19 | 18 | 51 | 133 | 171 |
| warm μ 0.02, short50, inv_freq | 2 | 11 | 9 | 28 | 142 | 177 |

Paired, [gained, lost, p]:

- grid50nh vs grid50: ≤ 2 edit 6 / 7 (p 1.0), dup 31 / 48 (p 0.07). The dup
  drop does not hold to one string (山田太郎 たいせつ やったネ ことば −4,
  こうえん +4).
- short50 vs μ 0.02: ≤ 1 edit 14 / 20 (p 0.39), ≤ 2 edit 33 / 32 (p 1.0),
  dup 25 / 36 (p 0.2).
- inv_freq vs short50: ≤ 1 edit 7 / 16 (p 0.09), **≤ 2 edit 9 / 32
  (p 4e-4)**, contained 6 / 14 (p 0.12). Hiragana strings ≤ 2 edit
  23 → 12, the others 28 → 16: the slowed rows' strings fall, and so do the
  rest.

| arm | box | box h | flat white | en_cos | box IoU vs EN ref |
|---|---|---|---|---|---|
| floor | 0.156 | 0.180 | 0.132 | 0.922 | 0.198 |
| cold, quoted + grid50 | 0.130 | 0.177 | 0.071 | 0.929 | 0.290 |
| cold, quoted + grid50nh | 0.120 | 0.168 | 0.069 | 0.933 | 0.317 |
| warm μ 0.02 | 0.102 | 0.189 | 0.109 | 0.931 | 0.217 |
| warm μ 0.02, short50 | 0.073 | 0.198 | 0.123 | 0.929 | 0.254 |
| warm μ 0.02, short50, inv_freq | 0.073 | 0.142 | 0.134 | 0.928 | 0.331 |

Training: grid50nh's in-box / out-box loss and Δ norm track grid50's within
0.001 / 1 % at every step (same seed, same batch order). inv_freq's mean Δ
norm peaks at 272.3 vs short50's 274.5 (seed ≈ 268); in-box loss sits
0.0005 above short50's throughout.

On the sheets (`garble_replace_compare/short50/`, columns EN ref | floor |
μ 0.02 | short50 | inv_freq; `overview_p0s0.png` + one sheet per string):

- **EN ref:** the base writes English small and horizontal — a line at the
  top, a black subtitle bar at the bottom, or a small `hi` in a corner —
  often sentence-length pseudo-English. No banner, no speech bubble.
- **floor:** the large banner where EN ref puts its small line.
- **μ 0.02 / short50:** no banner; 2–3 speech bubbles with 2–3-column
  lines. short50's bubbles reuse the target's glyphs (`かしてかいい`,
  `がしきしいきいい`) where μ 0.02's read as unrelated lines; no render
  draws the short word alone and stops. Where short50 keeps a banner it
  is smaller and fainter (小山田, 日本人 tiny at the top; パソコン → `オヌン`).
- **inv_freq:** fewer bubbles; small horizontal lines at the top or along
  the bottom, where EN ref writes (`かかな応さぃなしいしいい`,
  `かか 暮しなな帰ししいい`). The target's glyphs recur, repeated and
  mixed; the string does not read.

- **no humans:** no effect on the strings.
- **Short lines:** move layout (box 0.102 → 0.073), not length. Every
  render still writes a line, not the word.
- **Inverse frequency:** the strings get worse, not better (≤ 2 edit
  51 → 28). The layout moves furthest toward EN ref of any arm (IoU 0.331,
  box h 0.142, flat white back at the floor's).

Results: `experiments/garble_replace/results/20261001-0759-data-grid50nh/`,
`…/20261001-0759-cold-grid50nh/`, `…/20261001-0811-data-short50/`,
`…/20261001-0848-warm-short50-mu002/`,
`…/20261001-0948-warm-short50-mu002-invfreq/`. Jobs `…075957-47effd`,
`…081331-a77066`, `…094838-cdbfdf`. Data:
`output/cjk_anima_scale/run0930_garble_replace/data_{grid50nh,short50}/`.
Rows: `output/cjk_anima_scale/experiments/garble_replace_{cold_grid50nh,warm_short50,warm_short50if}/`.

## Follow-up (10-01): x̂0 per σ on the trained rows

`plan_garble_replace.md` § 4's `--traj` leg, run 10-01 through `sigma_split
--traj --rows <arm> --traj_conds ja_seed` (the arm's rows at every σ), on
the seed, warm (μ 0.1) and cold rows. Samples: こんにちは / やったネ × the 4 grid
prompts, seed 0. x̂0 at the step nearest σ 1.0 / 0.95 / 0.9 / 0.85 / 0.8 /
0.75 / 0.7 / 0.5 / 0, each read.

| rows | σ 0.9 | σ 0.84 | σ 0 | glyphs read at σ 0 (target 5 / 4) |
|---|---|---|---|---|
| seed | ≤ 1 edit 4 / 8, box 0.037 | 7 / 8, box 0.121 | 6 / 8, hit 4 | 5 4 6 4 6 4 5 5 |
| warm | 0, box 0 | 2 / 8, box 0.039 | 2 / 8, hit 0 | 6 5 8 4 8 14 8 7 |
| cold | 0, box 0.005 (tall: box h 0.06) | 0, box h 0.17 | 0 | 23 4 14 6 33 12 19 11 |

- σ 0.95: seed and warm draw the same blurred banner, same place and span.
- σ 0.9: the seed fills it with 4–5 large glyphs that already read. Warm
  fills the same span with a row of smaller glyphs (`こんにちがつぬいぐり`) —
  more slots than the word.
- σ 0.84 → 0.7: warm's string settles into those slots; the extra slots take
  a repeated glyph (`こんにちはは`, `こんくにちちはは`, `ややったたえっ…`) and
  stay to the end.
- Cold: tall vertical columns from σ 0.9 on, sentence-length.
- Warm moves the string's commit from σ 0.9 (seed) to 0.85: inside its band
  (0.6–0.85, σ drawn as 0.6 + 0.25 · sigmoid(z), ⅔ of draws in 0.67–0.78).

The span is set by σ 0.95 with or without the trained rows; the glyph count
inside it is set at σ ≈ 0.9, just above the band. The repeats are the slots
left over once the word is written. Not tested, but consistent with it: the
garble items draw their glyphs at the canvas's px (median 14.4), and warm's
glyphs come out smaller in the same span.

Results: `experiments/sigma_split/results/20261001-1217-traj-seed/`,
`…-1218-traj-garble_replace_warm/`, `…-1220-traj-garble_replace_cold/`.
Sheets: `output/cjk_anima_scale/experiments/sigma_split_s0.5/traj_<rows>/sheet_traj_ja_seed.png`.

## Follow-up (10-01): short50 at σ 0.8–0.95

One arm against warm μ 0.02 short50: the same items (`--legs reband --tag
short50hb --mix_from short50 --band 0.8 0.95`: images, captions, order and
latents of `data_short50`, every band → 0.8–0.95), same trainer constants,
5 130 steps. Job `20261001-122410-18dd04`.

| arm | official | contained | ≤ 1 edit | ≤ 2 edit | dup | kana |
|---|---|---|---|---|---|---|
| floor | 29 | 64 | 92 | 129 | 100 | 169 |
| warm μ 0.02, short50 (0.6–0.85) | 2 | 19 | 18 | 51 | 133 | 171 |
| warm μ 0.02, short50 at 0.8–0.95 | 1 | 3 | 7 (+3 / −88 vs floor, p 1e-22) | 25 | 120 | 178 |

| arm | box | box h | flat white | en_cos | box IoU vs EN ref |
|---|---|---|---|---|---|
| floor | 0.156 | 0.180 | 0.132 | 0.922 | 0.198 |
| short50 (0.6–0.85) | 0.073 | 0.198 | 0.123 | 0.929 | 0.254 |
| short50 at 0.8–0.95 | 0.078 | 0.309 | 0.174 | 0.917 | 0.084 |

Training: in-box loss 0.174 → 0.146 by step 2 500 (−16 %; short50 at its
band −9 %, 0.125 → 0.113), out-box flat at 0.124; warm_cos to the seed 0.930
at step 2 650 (short50 0.961).

traj (`sigma_split --traj --rows garble_replace_warm_short50hb`, same
samples and σ as above):

- Two to four large white bubble blobs at σ 1.0 / 0.95, before the figures —
  the garble canvases' bubbles. No banner at any σ.
- The bubbles hold vertical columns of small glyphs that only form at
  σ ≤ 0.5, sentence-length (`どこ夢んして ど疾な感る…`); 0 / 8 ≤ 1 edit at
  every σ (short50 at its band: 2–3 / 8 from σ 0.8).

At 0.8–0.95 the rows learn the canvas — its bubbles — not the line's length.
The short line's glyphs are drawn at the garble's px (median 14.4); at
σ ≥ 0.8 they are below what the FM target resolves, so the only gradient
the rows get there is layout. `windows.py` says the same of glyph identity:
0.8–0.95 is dead at 48 px (C.2), and 12–24 px singles sit at 0.3–0.5.

Results: `experiments/garble_replace/results/20261001-1223-reband-short50hb/`,
`…/20261001-1224-warm-short50hb-mu002/`, `experiments/sigma_split/results/20261001-1314-traj-garble_replace_warm_short50hb/`, `…/20261001-1316-traj-garble_replace_warm_short50/`.
Rows: `output/cjk_anima_scale/experiments/garble_replace_warm_short50hb/`.
