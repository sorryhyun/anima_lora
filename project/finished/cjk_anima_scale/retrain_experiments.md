# retrain_experiments — what plan_retrain read and landed (2026-09-28)

The record behind `plan_retrain.md`: why the singles re-seed cold, the case
for per-glyph routing, the windowed word pool, checks C0–C3, the
`retrain_kana` run and its read, and the code that landed. What is still to
run is in `plan_retrain.md`.

## 1. Why, and the scope

hypothesis.md § 4 P1 / P1b. On Stage B's 36 donor kana, rows trained
**cold from the pack rows** on in-word items + the lone singles group
(`p1_mix`) hold the floor's singles (official 82 vs 91 / 144, p 0.69;
repeat 30 vs 29) and compose (こんにちは ≤ 1 edit 0 → 11 / 16, exact
0 → 5, `dup` 4). The adapter reads them in context (c4 0.817; seed 0.977).
The same items warm from the seed (Stage B) compose less (6 / 16) and
double (`dup` 13), and a norm cap is nearly redundant (cold rows stop at
≈ 230–240 on their own). The seed's rows carry a context-immune direction.
Training on top of it keeps that direction.

Target: a new seed whose singles are cold-trained on lone + in-word data,
replacing `paths.SEED_ROWS` for every later run, and every JA caption
addressed through those singles (§ 2).

The seed (`rows_step1_0921_merged`, 2 274 rows; `step1_0921` 374 at 30 k
steps + `step1_0921z` 1 900 at 152 k, the research line's lone-style data):

| kind | rows | in the retrain |
|---|---|---|
| hiragana | 80 | retrain cold |
| katakana | 81 | retrain cold |
| kanji | 783 | retrain cold |
| other singles (punctuation etc.) | 9 | retrain cold, lone only |
| frequent glyphs with no single row yet | ≈ 560 (§ 2) | train cold (C-k's 388 are 386 of them) |
| pieces (2 / 3 / 4 / 5 glyphs) | 797 / 354 / 140 / 29 | **not trained**: § 2 routes around them |

## 2. Pieces: per-glyph routing (hypothesis.md § 3, P2)

**Per-glyph routing** is an encoder flag. Every JA Qwen token goes, on the
T5 side, to its glyphs' single rows; the Qwen text is untouched.
`HybridT5Encoder` already regroups byte fragments per char, and the flag
makes that the only path. The T5 side becomes ≈ 1 500 single rows under
27 k Qwen JA tokens (**Qwen / T5 ≈ 3**, EN's is 2.4), and every row is
shared by every word it appears in. Piece rows go unused, so the piece
question disappears. It was gated on P1 ("P2 follows P0 / P1 … only if P1
composes"), and P1b composes.

For it:
- **こんにちは**: the piece row renders 0 on every arm on record (300f
  0 → 0; B0 at × 1 and × 3). Spelled on `p1_mix`'s singles: ≤ 1 edit
  11 / 16, exact 5. Spell 2026-09-26 had the same direction on the seed
  (trained singles 13 / 32 within 2 edits vs the piece row's 4).
- **Coverage** (manga109s dialogue pool, CPU 2026-09-28). The seed's 953
  singles already hold **96.6 %** of the JA glyph occurrences (941 of 1 890
  distinct glyphs):

  | most frequent glyphs | occurrences covered | not in the seed's singles | of them in C-k |
  |---|---|---|---|
  | 1 000 | 99.06 % | 161 (159 kanji) | 46 |
  | 1 500 | 99.85 % | 563 (559 kanji) | 386 |
  | 2 000 | 100 % | — | — |

  Pieces cover ≈ 85 % of lines with 1 900 cold rows
  (`step1_0921z`). Singles reach 99.85 % with ≈ 1 500.
- **の / を** are addressable. The spelled-caption problem (their
  space-prefixed form is another Qwen token) is a tokenization artefact of
  spelling. Routing maps the glyph to its single row at the encoder.
- plan_2900 B and C-p (the piece runs, ≈ 18 h) are no longer needed.

Against it, or unread before C0:
- **The routed caption had not been rendered.** T5 gets per-glyph ids and
  Qwen gets the whole word (c1's `ja_hybrid`). c1 found Qwen's word context
  moves the adapter's context cos by nothing (0.965 → 0.965), but that is
  an adapter read, not a render. C0 / C2 rendered it: routed = spelled
  (§ 4).
- **Doubling reaches every caption.** Every JA string becomes a run of
  single rows, so the in-word `dup` of the singles (4 / 16 on `p1_mix`) is
  what every caption carries. C2 and `retrain_kana` measured it (§ 4, § 5).
- The piece training on record (300f, `300f_sp`, the long pieces) is sunk.

## 3. Data: the windowed word pool

Two of the three tiers existed already:

- **lone** = the production `b0709` group (`scene_single` + `grid_single`,
  σ 0.7–0.9), at share 0.5 of the kind's items (P1b's 1 : 2 lone to in-word).
- **in-word, count** = Stage B's `scene_single_small` in `b0507`
  (weight 0.3).
- **in-word, spelled** = Stage B's `scene_spelled` (unspaced image, spaced
  caption) in `b0507` / `b0305`. **Its word source is the new part.** After
  C0, the caption is the routed unspaced form (`scene_window`) instead of
  the spaced one.

Stage B took whole dialogue lines of 2–6 glyphs, every glyph a donor, none
repeated. Over the seed's 953 singles (の / を excluded, as spelled), the
dialogue pool gives:

| word source | words | kana median words / glyph | kanji median | kanji with 0 | kanji < 5 |
|---|---|---|---|---|---|
| whole lines, 2–6 glyphs | 7 607 | 188 (hira), 39 (kata) | **3** | **101** | **496** |
| windows of lines, 2–4 glyphs | 204 053 | 725 (kata) | 90 | 2 | 4 |
| windows of lines, 2–6 glyphs | 380 447 | 1 406 (kata) | 162 | 2 | 3 |

(CPU, 2026-09-28; the two kanji at 0 do not occur in the corpus at all.)

Whole lines cover kana only. Kanji need **windows**: any substring of a
dialogue line whose glyphs are all singles of the run, none repeated. A
window can cross a word boundary. Nothing on record says the DiT needs
real words (JA is rendered, never read for meaning), and C3 checked it.
Rules carried over from Stage B:
- A word draw picks a glyph uniformly, then one of its words, so exposure
  is per row.
- No glyph repeated inside a word, so training does not teach doubling.
- The read words are trigram-held-out from the pool.
- Every spelled word must encode to its glyphs' single ids and nothing
  else (`stage_b.check_spelling`). A word that does not is dropped.

So the retrain needed one production change, not a new data pipeline: a
windowed recipe + word pool in `recipes.py`, and the single kind's groups
in `builder.TABLE` (§ 6). The builder, pools, trainer, eval and floor stay
as they are. That is why the retrain lives in the line and not in a new
project.

## 4. Checks C0–C3

Each was launched by hand after the previous read.

| # | what | answers | GPU |
|---|---|---|---|
| **C0** | **P2 render.** The routing flag (encoder + the eval path), then `p1_mix`'s rows on the unspaced こんにちは: routed vs spelled (cached, 11 / 16) vs unrouted (the seed's piece row, cached on the floor) | Does a routed caption render what the spelled one does? Routed ≈ spelled (≤ 1 edit within noise of 11 / 16) → § 2 holds and pieces stay out. Routed ≪ spelled → Qwen's word context interferes at render, and the piece question reopens (a cold-piece micro in P1's shape). | ≈ 16–32 renders |
| C1 | `p1_lone`: 36 donors, cold, `b0709` only, 90 steps / row; render read + c4 | Is the in-word tier load-bearing, or is cold alone enough? Lone-only cold composes like `p1_mix` → § 3's new tier is not needed, and the retrain is the existing TABLE, cold. Ends context-immune (c4 ≫ 0.82) with no composition → the in-word tier stays. | ≈ 25 min + read |
| C2 | the composition read widened: 6–8 held-in kana words (donor glyphs, trigram-held-out), spelled **and routed**, floor rendered **once**; re-read seed / Stage B / `p1_cold` / `p1_mix` (/ `p1_lone`) | `p1_mix`'s composition on more than one word × 16 renders, and C0 on more than one word; in-word `dup` with it | floor ≈ 256 renders + ≈ 256 per arm |
| C3 | 36 cold kanji donors (mixed density, some from the ≈ 560 with no row yet), windowed in-word pool + lone, 225 / row; read the kanji singles + 4–6 held-in kanji-bearing words (routed) | Does the recipe carry from kana to kanji: identity at the cold-kanji budget, composition with windows? It also sets the kanji steps / row. | ≈ 36 × 225 steps ≈ 1 h + read |

**C0 read (2026-09-28, `experiments/p2_route/results/20260928-0100-c0/`,
job `20260928-010053-5dd1b7`): routing holds, pieces stay out.**
こんにちは (en, 16 renders), ≤ 1 edit / official / contained:

| rows | routed | spelled | unrouted (piece row) |
|---|---|---|---|
| `p1_mix` | **14** / 8 / 11 | 11 / 5 / 6 | 0 / 0 / 0 |
| seed (floor) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 (`dup` 6) |

Routed vs spelled on `p1_mix`: ≤ 1 edit 3 / 0 (p 0.25), contained 5 / 0
(p 0.06), `dup` 3 vs 4. Every routed render carried one ext row per glyph.
Qwen's whole-word context does not interfere at render, and may help. One
word; C2 widens it.

- Spelling is not a neutral address beyond の / を: a spaced kanji is a
  space-prefixed Qwen token with its own row (`" 名"` → row 675, 名 → 22).
  Routing sends both to the glyph's own row, so kanji data (C3, the
  retrain) uses routed captions.
- The flag: `mapping["glyph_route"]` (in the digest only when set) or
  `ANIMA_VOCAB_GLYPH_ROUTE=1|0` (overrides, read in
  `VocabPack.build_encoder`; the line's TE cache key carries it). Every JA
  Qwen token (27 143) splits; none keeps its piece row. Experiments set the
  env var in-process, never in the submit shell (a daemon booted from that
  shell would route every later job).

**C1 read (2026-09-28, `experiments/p1_cap/results/20260928-0103-lone1/`,
job `20260928-010327-fab1ec`): the in-word tier is load-bearing.**
`p1_lone` = the 36 donors cold on the production table alone (`b0709`,
2 400 lone items, 90 steps / row, data `run0928_p1_lone`):

| donor keys (en) | floor | Stage B | `p1_cold` | `p1_mix` | **`p1_lone`** |
|---|---|---|---|---|---|
| end row norm | ≈ 320 | 311 | 229 | 240 | **230** |
| こんにちは ≤ 1 edit / 16 (spelled) | 0 | 6 | 12 | 11 | **0** |
| こんにちは ≤ 2 edits | 0 | 13 | 14 | 12 | 1 |
| singles official / 144 | 91 | 43 | 29 | 82 | **74** |
| singles repeat | 29 | 50 | 43 | 30 | 25 |

A cold start alone stops the rows at the same norm as the in-word arms and
composes nothing. The norm is not what composes; the in-word items are.
Singles: 74 vs the floor's 91 (paired 18 / 35, p 0.027), near `p1_mix`'s
82. So § 3's new tier stays. At the adapter (`ctx_trigger --probe c4`,
`results/20260928-0431-c4lone/`) the lone rows read less context than
`p1_mix`'s at nearly the same norm: out cos 0.893 at 228 vs 0.817 at 238
(seed 0.977 at 321, pack rows 0.624 at 182). The in-word items move the
rows' direction, not their size.

**C2 read (2026-09-28, `experiments/p2_route/results/20260928-0138-c2/`,
job `20260928-010523-19dc00`): `p1_mix` composes on eight words, and
routed = spelled everywhere.** こんにちは + たいせつ かなしい かんがえ
たすけて こうえん てつだう ことば (donor glyphs, trigram-held-out from the
donor words), en, 8 × 16 renders per condition:

| arm | ≤ 1 edit routed / spelled | ≤ 2 (routed) | official | `dup` | ≤ 1 with doubles collapsed |
|---|---|---|---|---|---|
| floor (seed) | 1 / 1 | 14 | 0 | 9 | 1 |
| Stage B (warm) | 23 / 24 | 55 | 2 | 81 | 46 |
| `p1_cold` | 65 / 65 | 96 | 23 | 60 | 79 |
| **`p1_mix`** | **80 / 75** | **110** | **26** | 55 | **105** |
| `p1_lone` | 9 / 10 | 37 | 0 | 40 | 10 |

- **Routing**: routed vs spelled within noise on every arm (`p1_mix` ≤ 1
  edit 15 / 10, p 0.42; floor ≤ 2 edits 2 / 10, p 0.04, both near zero).
  C0 generalises: § 2 holds.
- **`p1_mix` composes on every word** (≤ 1 edit 6–15 / 16 per word; vs the
  floor 79 / 0, p 3e-24), ahead of Stage B (62 / 5) and at or above
  `p1_cold` (34 / 19, p 0.053; ≤ 2 edits p 0.034) while holding the
  singles (C1 table).
- **`p1_lone` barely moves** (9 / 128): C1's one-word read holds on eight.
- **Doubling is the remaining cost**: 55 / 128 `p1_mix` words carry a
  doubled glyph, and collapsing them lifts ≤ 1 edit 80 → 105. It is below
  Stage B's (81) but far above the floor's 9, whose renders are mostly not
  the word.

**C3 read (2026-09-28, `experiments/c3_kanji/results/20260928-0246-c3/`,
job `20260928-011133-aa3912`): composition carries to kanji; identity is
bought for new kanji and lost for the seed's dense ones at 225 / row.**
36 kanji (stage_i's 12 with no seed row, dense_a0's 12 seed rows, 日本人大丈夫何時
(seed), 死父誰名 (no row)), cold, on `p1_mix`'s merged rows as context
(`train(…, context=)`), `p1_mix`'s table with windows (2–4 glyphs of
dialogue lines, 4 370, kanji-first draw; 輩 12, 精 / 奥 17 the thinnest) as
the in-word tier, routed captions, 6 000 items, 225 steps / row (59.5 min).
End row norm ≈ 270 (the kanji pack rows start at ≈ 203).

| read (en, 16 renders / key) | floor | `p1_mix` (kana context only) | **`c3_kanji`** |
|---|---|---|---|
| words ≤ 1 edit / 96 (routed) | 3 | 7 | **36** |
| words ≤ 2 edits | 28 | 33 | 68 |
| words official | 0 | 3 | 13 |
| words `dup` | 22 | 24 | 39 |
| new kanji (stage_i 12) official / 192 | 0 | = floor | **51** |
| new kanji contained | 4 | = floor | 104 |
| seed kanji (dense_a0 12) official / 192 | 65 | = floor | **31** |
| seed kanji contained | 110 | = floor | 93 |

Words, per word (≤ 1 edit / 16): 日本人 14, 大丈夫 9, 小山田 6, 愛してる 4,
何時間 3, 山田太郎 0 (≤ 2 edits 7). Paired vs the floor: words ≤ 1 edit
35 / 2 (p 1e-8); vs `p1_mix` 33 / 4 (p 1e-6), so it is the kanji rows, not
the kana context.

- **Windows compose.** Substrings that cross word boundaries were enough;
  nothing here needs real words.
- **New kanji get identity** at 225 / row (stage_i's lone-only I0 at
  90 / 270 contained 59 / 117, `budget.RULES`); 郎 野 太 stay at 0.
- **The seed's dense kanji lose half their identity** (official 65 → 31,
  paired 14 / 48, p 2e-5; 感 愛 飲 最 様 at 0). The kana run at 135 held
  the floor (C1 table: 82 vs 91). For kanji, 225 / row does not, so the
  kanji steps / row are not set yet (`plan_retrain.md` § 1).
- Doubling rises with composition, as on kana (`dup` 39 vs 22).

**C3 at 450 / row (2026-09-28, `experiments/c3_kanji/results/20260928-1145-c3s450/`,
job `20260928-114509-3798dd`): the seed's dense kanji come back to the
floor; the extra steps go to ink-dense glyphs.** Same data, same context,
`--steps 450` (16 200 steps, 119 min, arm `experiments/c3_kanji_450/`).
End row norm ≈ 285 (dnorm 248, peaked ≈ 254 mid-run, like `retrain_kana`).

| read (en, 16 renders / key) | floor | 225 / row | **450 / row** |
|---|---|---|---|
| seed kanji (dense_a0 12) official / 192 | 65 | 31 | **57** |
| seed kanji contained | 110 | 93 | 115 |
| new kanji (stage_i 12) official / 192 | 0 | 51 | **63** |
| new kanji contained | 4 | 104 | 141 |
| words ≤ 1 edit / 96 (routed) | 3 | 36 | 37 |
| words official | 0 | 13 | 8 |

- **Seed kanji at the floor**: 450 vs floor official 28 / 36 (p 0.38);
  vs 225 32 / 6 (p 2e-5). 感 0 → 4, 最 0 → 1; 愛 飲 様 stay 0 (floor
  2 / 2 / 0). New kanji contained vs 225 51 / 14 (p 5e-6).
- **Words do not move** (≤ 1 edit 14 / 13 vs 225): the extra steps buy
  identity, not composition.
- **By ink, not by group.** stage_i's 12 hold 地 道 郎 場 野 at ink
  10.5–11.6 (`cf_sense._glyph_ink`, cells² at 48 px), next to dense_a0's
  11.2–12.9. Split at ink 10 (official / contained, of 16 per glyph):

  | ink | glyphs | 225 / row | 450 / row |
  |---|---|---|---|
  | < 10 (7) | 小太天山田空星 | 38 / 78 | 34 / 81 |
  | ≥ 10.5 (17) | the rest | 44 / 119 | **86 / 175** |

  At 225, ink and official correlate −0.38 over the 24; at 450, −0.05; the
  225 → 450 gain correlates +0.41 with ink. Light glyphs are done at 225
  (天 14 → 9, 山 8 → 4); dense ones double (地 1 → 10, 奥 4 → 10, 野 0 → 5).
  7 vs 17 glyphs, 16 renders each, and dense_a0's glyphs had seed rows
  (trained cold here all the same): a small read.
- **The kanji budget (user, 2026-09-28): ink < 10 → 225 / row, ink ≥ 10 →
  337 / row** (the midpoint of 225 and 450, unmeasured; 450 is the read
  point). Distribution over `retrain_kanji`'s glyphs (dialogue_2_10
  ranking, seed singles from `SEED_ROWS`; recomputed 2026-09-28 at 942
  covered / 557 new at the 1 500 cut, vs § 2's 941 / 559):

  | cut | kanji | ink < 10 | ink ≥ 10 | all 450 | **225 / 337** | all 225 |
  |---|---|---|---|---|---|---|
  | top 1 000 | 941 (783 + 158) | 413 (44 %) | 528 | 423 k | **271 k** | 212 k |
  | top 1 500 | 1 340 (783 + 557) | 542 (40 %) | 798 | 603 k | **391 k** | 302 k |

  Ink over the 1 340: median 10.5, p10 7.5, p90 12.7, max < 16 (騙 襲 鐵).
  At 8.3 k steps / h local, 391 k ≈ 47 h; ≈ 16 h on a G4 (≈ 25 k / h).

## 5. `retrain_kana`

Budget: cold singles use `budget.py`'s cold single row; the mix multiplies
by (all items / in-word items) = 1.5, so the in-word items keep their
exposure (P1b's 90 → 135). The kana row is P1b's point (cold hiragana,
singles at the floor); katakana at 135 was unread before this run.

**Launched 2026-09-28** (`configs/runs/retrain_kana.toml`, data job
`20260928-075359-9f6158`, train job `20260928-075839-76c2d9`): 174 singles
(the seed's 80 hira + 82 kata + 8 punctuation, + ヴヶゔヵ), cold,
135 / row = 23 490 steps, 17 400 items (5 800 per group). Windows
178 846, 163 glyphs (ヂ ヵ have none: lone only), per glyph median 1 612.
`read` (held out of the windows): C2's eight words, なにしてる, テレビ
カメラ パソコン アイドル. Trained 183 min, end row norm ≈ 250 (peaked
≈ 291 mid-run).

**Read (2026-09-28, `experiments/retrain_read/results/20260928-1102-kana/`,
job `20260928-110210-54da19`): 174 rows compose like `p1_mix`'s 36 and
hold the singles; doubling rose.** A smaller grid than C2: 4 prompts × 2
seeds = 8 renders / key, routed; words `en` (C2's eight pair with the
floor's and `p1_mix`'s cached `native_route/`, prompts < 4), singles
`swap` (hiragana floor from `native_spell/`, katakana floor rendered once).

| words (en, / 8 per word) | ≤ 1 edit | ≤ 2 | official | `dup` | ≤ 1, doubles collapsed |
|---|---|---|---|---|---|
| floor, C2's 8 (cache) | 1 / 64 | 7 | 0 | 4 | 1 |
| `p1_mix`, C2's 8 (cache) | 34 | 55 | 11 | 31 | 51 |
| **`retrain_kana`, C2's 8** | **29** | 48 | 10 | **42** | 48 |
| `retrain_kana`, なにしてる (held out) | 4 / 8 | 5 | 0 | 5 | 5 |
| `retrain_kana`, 4 katakana words | 20 / 32 | 26 | 6 | 15 | 22 |

| singles (swap) | official | contained | paired official vs floor |
|---|---|---|---|
| 8 hiragana: floor / `retrain_kana` | 23 / 17 of 64 | 46 / 44 | 2 / 8, p 0.11 |
| 6 katakana: floor / `retrain_kana` | 7 / 4 of 48 | 25 / 29 | 2 / 5, p 0.45 |

- **Composition holds at 174 rows**: vs the floor ≤ 1 edit 28 / 0
  (p 7e-9); vs `p1_mix` 13 / 18 (p 0.47), ≤ 2 edits 5 / 12 (p 0.14).
- **Katakana composes at 135 / row**: カメラ パソコン 6, アイドル 5, テレビ 3
  of 8 (no floor rendered; routed words read ≈ 0 on the floor).
- **Doubling is the cost**: `dup` 42 vs `p1_mix`'s 31 (paired 20 / 9,
  p 0.06), and collapsing doubles lifts ≤ 1 edit 29 → 48 / 64. Why it
  rose is unread (`plan_retrain.md` § 4). In row space the donors carry
  more of the shared direction than `p1_mix`'s (0.43 vs 0.28), a candidate
  only (`reports/row_geometry_2026_09_28.md` § 6).
- **Singles hold**: official dips without significance, contained flat
  (C1's `p1_mix` had the same shape, 82 vs 91). ノ reads 0 on both arms
  (the reader).

## 6. Code that landed (2026-09-28)

0. (C0) Per-glyph routing: an encoder flag in
   `library/anima/ext_vocab.py::HybridT5Encoder`, reached from inference
   and from the line's eval path (the `native` stage encodes through the
   pipeline). Default off until the new seed is baked
   (`tests/test_ext_vocab_glyph_route.py`).
1. `recipes.py`: `scene_window` (C3's routed windows, not the spaced
   `scene_spelled`) + `scene_single_small` (Stage B, byte-faithful), and the
   windowed word pool (`window_pool`: letters only, 2–6 glyphs, no repeats,
   trigram hold-out of the run's `read`; `routed_windows`: the encoding
   check).
2. `builder.TABLE`: the single kind's three groups (§ 3): lone `b0709` at
   0.5, `b0507` windows 0.7 + count 0.3, `b0305` windows. A build with
   windows stamps `glyph_route` in `build.json` and writes `windows.json`.
3. `budget.py`: singles start cold (`COLD_KINDS`); the cold single row
   splits by script (kana 90 from P1b, kanji 150); `mix_factor` = Σ of
   the kind's shares (× 1.5).
4. `train.py`: `cold=None` = the rule; a `glyph_route` data dir trains
   routed (env set in-process); the steps take `mix_factor`. The context
   override is `train(…, context=)` (C3); `scale.py` does not expose it
   yet (`plan_retrain.md` § 2).
