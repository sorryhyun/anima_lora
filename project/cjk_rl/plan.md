# Plan: the glyph stage on G-D (opened 2026-11-25)

The glyph stage trains one row a glyph on the OCR reader's loss. How it runs is in
[README.md](README.md), what it has measured in [reports/](reports/). This file holds the next
step, what is parked and the runs done. A run's settings, question and gate go in its config
header (`configs/<handle>_mmdd_hhmm.toml`) when it is submitted, and this file then names the
run.

## 6. Where the kanji run starts — to decide

`g2b_1126_0058` is done ([reports/g2b_2026_11_26.md](reports/g2b_2026_11_26.md)): the kana
from jp_v1's rows at 16 guarded steps read alone on 1 716 of 2 624 pages, against 1 686 and
1 785 for `g2a_1125_1642`'s 4- and 6-step rows from the shared start. The user's read of its
sheets (g2a's 6-step rows against its 16-step rows, 164 files, sent 11-26) is not made yet.
Nothing is queued. The user's calls:

- The start of the kanji rows: the shared start or jp_v1's rows. For the shared start, the
  shift is g1long's 16-row mean as g2a used it, or a mean over the 164 kana rows (g2a's rows
  from jp_v1's: cosine 0.99 with g1long's mean; g2b's: 0.81, at half the length).
- The guards on that run. The user (11-26): the return should fire where the guard has, or on
  a stricter condition. `[back] over` is built for it (0.5: a best only where the taper
  shortens the step); the return rule replayed on logged scores is in the report of item 6's
  run and below. `taper` and `near_score` ran as built.
- Which kana rows ship: g2a's 6-step rows are the pack `anima_cjk_vocab_pack_jp_v1_g2a`; g2b
  reads higher on 64 rows and lower on 76.

Facts kept from the run's planning (11-26):

- The return replayed on logged scores (the rows then move as logged): drop 4 with no floor,
  with `over` 0.5, and drop 6 with `over` 0.5 give 21, 18 and 11 returns on
  `g1long_1125_1335`'s 384 steps, 5, 3 and 3 on `g1shared_1125_1538`'s 128 (あ at step 8
  under each).
- Two things were filed as "past the glyph". A row that reads is moved a full step by the
  few renders that fail (ぬ in `g1long_1125_1335`: 14 – 16 of 16 at steps 12 – 16, 7 – 11 at
  steps 21 – 24). A row that doubles its glyph (タ, ミ, こ in `g2a_1125_1642`) never read
  over 9, 6 and 4 of 16 on the training side.
- `reader_attn = "sdpa"` was added to `g2a_1125_1642`'s config on 11-26: it ran before
  FlashAttention 2 became the default, and `start_from` compares the key.
- A data build beside the run (the user, 11-26): `split_1126_0910` set aside 2 002 crops
  where the tuned reader and hayai split on the asked glyph, for the tuned reader's training
  later (`output/cjk_rl/reader_split/split_1126_0910/`, local; the report's second section).

## 5. After the kana — to decide

`g2a_1125_1642` is done ([reports/g2a_2026_11_25.md](reports/g2a_2026_11_25.md)); the user's
read of its sheets is not made yet. Nothing is queued. What the report leaves to decide, each
with its cost:

- The shapes the two scripts share (へ べ ぺ / ヘ ベ ペ) folded as the small kana are, in the
  read and the loss. The read can be recounted from `records.json` with no render; the loss
  changes only for a later run.
- The rows in `not_learned` that are the row's own (こ, the voiced rows ぎ ぞ ば ぶ ゲ ヅ, and
  を タ モ ン): more steps from their 6-step save. About 12 rows at 8 more steps is 20 min of
  training and four short reads.
- The detector's misses on large one- and two-stroke glyphs (く ノ ヘ ロ ン): whether the
  no-box term should be off for a row once its glyph reads, and how such pages are counted.
- The frequent kanji ([proposal_glyph.md](proposal_glyph.md)): 1 410 rows at 6 steps are
  about 29 h of training, and a read of all of them is 22 560 renders an arm, about 3.3 h.
  Whether they run this way, at 4 steps (most of the kana's gain), or on the cheaper step of
  § 2 is the user's call. `skip_lone_read` (built 11-25, off unless a config sets it) takes
  about 0.17 s off a 3.04 s group, about 1.6 h of the 29; a log with it on has `lone` in
  place of box / low on one-box pages.

- The plain frame (the user, 11-25, on ComfyUI renders of the g2a pack: no lettering without
  `speech bubble`). Read ([plain](reports/plain_2026_11_25.md),
  [pscreen](reports/pscreen_2026_11_25.md), [mix](reports/mix_2026_11_26.md)): g2a's 6-step
  rows carry to captions with no bubble tag and are about at EN's level there; the model
  itself letters EN on a fifth of plain pages and almost never on a scene of two figures;
  16 rows trained in both frames on the plain screen's scenes (`mix_1125_2338`) read under
  g2a's in both frames. Nothing queued. Open, the user's call: leave the kana on the bubble
  frame's recipe, or a different plain-frame arm (its scenes, its share of the renders).

## Parked

- A step that shrinks with the mean gradient's norm: offered for item 4 and left out on the
  user's call (11-25); not run. Item 6 tapers the step by the row's training-side count
  instead.
- A term for text outside the picked box (the detector's score there, pushed down), for the
  rows that double their glyph. It would be a page term, which the stage's rules leave out;
  offered to the user on 11-26, not taken up.
- The no-box term off for a row once its glyph reads (item 5's third point): offered again
  on 11-26, not in item 6.
- The shared part kept as its own vector that every row's renders train: the cheaper step the
  1 410 kanji may need ([proposal_glyph.md](proposal_glyph.md) § 2).
- G1's permuted-label control: not run.
- The x_3 cache: set for the runs after the micro arm and dropped the same day (the user,
  11-25). What it costs is not measured.
- After the kana: [proposal_glyph.md](proposal_glyph.md) (the frequent kanji, words, transfer
  and ship reads). The sentence arms of [proposal.md](proposal.md) wait behind the stage.

## Done

Solo is of 256 held-out pages (16 glyphs, 4 read scenes, 4 seeds), the start rows first;
item 4's is of 2 624 (164 glyphs).

| item | what | run | report | number | verdict |
|---|---|---|---|---|---|
| – | G0a – c: the tagless frame, the terms' spread, the pixel head against the VAE | `probes/glyph_g0.py` | [glyph_g0](reports/glyph_g0_2026_11_25.md) | pixel head against VAE: sign test p ≥ 0.19 on every term | – |
| – | the scene pool and the crossed grid | `probes/glyph_scenes.py` | [glyph_scenes](reports/glyph_scenes_2026_11_25.md) | 25 of 128 candidates selected; solo EN 0.57, hiragana 0.27, kanji 0.05 | – |
| 1 | G-D's path, and G0d | `g0d_1125_1119` | [gd_g0d](reports/gd_g0d_2026_11_25.md) | the last step's ce gradient five decades over Adam's ε (bar: two) | passed |
| 2 | G1's micro arm: 16 rows, 8 steps | `g1micro_1125_1148` | [g1micro](reports/g1micro_2026_11_25.md) | solo 59 → 132 | adopted on the user's read |
| – | hayai v2.7 beside the tuned reader | `hayai_1125_1234`, `g0h_1125_1243` | [reader_pair](reports/reader_pair_2026_11_25.md), [hayai_g0](reports/hayai_g0_2026_11_25.md) | hayai counts the micro arm 60 → 125; the two row gradients' cosine 0.21 | – |
| – | the reader side of a group, timed | `room_1125_1324` | [reader_room](reports/reader_room_2026_11_25.md) | 0.75 s a group | – |
| – | a one-box training page left unread (`skip_lone_read`) | `room_1125_2155`, `lone_1125_2200` (smoke) | [reader_room](reports/reader_room_2026_11_25.md), second part | read 0.317 → 0.148 s a group; same box and image gradient on 256 of 256 pages | – |
| 3 | G1 long: 16 rows, 24 steps, hayai's term | `g1long_1125_1335` | [g1long](reports/g1long_2026_11_25.md) | solo 59 → 136 → 175 → 199 at 8, 16, 24 steps | adopted on the user's read |
| – | the 24-step rows on the base model, sign prompt | `sign_1125_1504` | [g1long](reports/g1long_2026_11_25.md), second part | 4 glyphs, 2 seeds | – |
| 3b | the row line: the rows' walk, their place in the table, forward reads | `line_1125_1517`, `line_1125_1527` | [row_line](reports/row_line_2026_11_25.md) | solo at twice the 8-step displacement 122, at the other rows' mean displacement 112, at a drawn direction 59 | – |
| 3c | G1 from the shared start: 16 rows, 8 steps | `g1shared_1125_1538` | [g1shared](reports/g1shared_2026_11_25.md) | solo 112 → 148 → 156 at 0, 4, 8 steps (hayai 104 → 147 → 167) | adopted on the user's read |
| 4 | G2a: the 81 hiragana and the 83 katakana, 6 steps from the shared start, small kana folded | `g2a_1125_1642` | [g2a](reports/g2a_2026_11_25.md) | solo of 2 624: 743 → 1 322 → 1 686 → 1 785 at the start, the shifted start, 4, 6 steps (hayai 835 → 1 420 → 1 762 → 1 840) | – |
| – | g2a's rows read with no bubble tag | `plain_1125_2249` | [plain](reports/plain_2026_11_25.md) | 16 rows, solo of 256 plain pages 22 → 93; on scene 35 0 → 4 of 64 | – |
| – | EN words on the same scenes, both frames | `enplain_1125_2258`, `enbubble_1125_2258` | [plain](reports/plain_2026_11_25.md) | 16 words, solo of 256: plain 71, bubble 104 | – |
| – | the plain screen: new scenes by their EN count with no bubble tag | `pscreen_1125_2306` | [pscreen](reports/pscreen_2026_11_25.md) | 160 scenes, EN solo 0.205; 24 at 8 of 16 or more | – |
| – | 16 rows in the plain and bubble frames on 12 of the plain screen's scenes, 6 steps | `mix_1125_2338` | [mix](reports/mix_2026_11_26.md) | solo of 512 a frame, this run against g2a's rows: plain 202 against 238, bubble 297 against 329 | – |
| 6 | G2b: the kana from jp_v1's rows, 16 steps, with `taper`, `[back]`, `near_score` | `g2b_1126_0058` | [g2b](reports/g2b_2026_11_26.md) | solo of 2 624: 743 → 1 118 → 1 393 → 1 566 → 1 716 at 4, 8, 12, 16 steps (g2a 1 686 and 1 785 at 4 and 6); rows against g2a's 6-step rows 64 up, 76 down | – |
| – | the crops the two readers split on | `split_1126_0910` | [g2b](reports/g2b_2026_11_26.md), second part | 2 002 crops: hayai only 1 308, tuned only 694 | – |

- The verdicts on items 2, 3 and 3c are the user's, in each report's Facts. Item 3c's is the
  call for item 4: the shared start at 6 steps.
- hayai's term at weight 0.2 is the user's call (11-25), made on the two hayai reports.
- Item 4 ran on the user's calls (11-25): 6 steps a row from the shared start, the two
  scripts in one run, and the small kana folded onto their full-size glyphs (`fold_small`)
  after a first submit (`g2a_1125_1630`) was stopped 6 min in. Its 28 rows in a group
  (17 not learned, 9 past their glyph, 12 a reader's problem) are in the report.
- Item 4's 6-step rows as a pack (the user, 11-25): `anima_cjk_vocab_pack_jp_v1_g2a`
  (`models/vocab_packs/`, local; linked into ComfyUI's `models/vocab_packs` for
  `AnimaVocabPackLoader`), baked from the run's `rows.pt` onto `anima_cjk_vocab_pack_punct`
  with `scripts/toolkits/bake_vocab_pack.py --glyph_route`. It is `jp_v1` with the 164 kana
  rows replaced (checked: no other row and no routing map differs); the rows in a group are
  in it as trained.
