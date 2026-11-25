# Plan: the glyph stage on G-D (opened 2026-11-25)

The glyph stage trains one row a glyph on the OCR reader's loss. How it runs is in
[README.md](README.md), what it has measured in [reports/](reports/). This file holds the next
step, what is parked and the runs done. A run's settings, question and gate go in its config
header (`configs/<handle>_mmdd_hhmm.toml`) when it is submitted, and this file then names the
run.

## 4. G2a — the hiragana and the katakana, 6 steps from the shared start

The user's call (11-25, after `g1shared_1125_1538`): "시작행 + 16행 평균에서 음... 아예 그냥
6스텝만, 히라가나+카타카나 학습으로 하면 어떨까? 이것만으로도 일단 큰 도약이고 동시에 효율적이니깐.
어차피 문장런도 나중에 생각해야 하고." It replaces the call after `g1long_1125_1335` (16 steps a
row from the start rows, the hiragana and the katakana as two runs). Not queued; it runs on
the user's word.

Facts:

- Rows: the 81 hiragana and the 83 katakana with one trained row in the frame, in one run.
  Each starts at its start row (`seed_1008`) plus the mean 24-step displacement of
  `g1long_1125_1335`'s 16 rows (`[start_shift]`, `src/g1.py`); for the eight hiragana among
  those 16 the row's own displacement is left out of the mean, as in `g1shared_1125_1538`.
- Steps a row: 6, fixed (the user's call). That is 96 renders a row, a pass and a half: the
  full first pass, then the first 8 of a row's 16 scenes in the second pass's order.
- Everything else as `g1shared_1125_1538` ran it: scenes, split, update (0.03 row norms a
  step, 16 renders a step), `hayai_weight` 0.2, no x_3 cache. Parent: `g1shared_1125_1538`;
  what changes is the rows and the steps (8 → 6).
- Saves, each read: the start rows, the shifted start, 4 steps and 6 steps.
- To build before it runs: the katakana as a class in `src/g1.py` (`CLASSES` holds `hira` and
  `kanji`) and a class a config may leave out; a cap on a row's steps (`max_steps`: a row at
  the cap is passed over) with saves named by the steps taken; the read's analysis below (a
  script over `records.json` and `log.jsonl`).
- Read, per row, the start rows against the shifted start and each save on the read scenes
  and seeds (16 pages a row and arm): solo under the tuned reader and under hayai, on_page,
  extra, no-box pages, the picked crop's ce; the training log's count a step.
- The analysis the user asked for after `g1long_1125_1335`, per row, with the groups written here before
  the run:
  - not learned: solo under 8 of 16 at 6 steps under both readers, with the shifted start's
    solo and the training-side count a step beside it (still climbing, or level).
  - pushed past its glyph: solo at 6 steps under the 4-step save's or the shifted start's by
    4 or more of 16, or the picked boxes reading the glyph with a second character (あ's
    あら in `g1shared_1125_1538`).
  - a reader's problem: the two readers apart by 4 or more of 16, or the boxes read as a
    look-alike of the asked glyph. Look-alikes counted apart from misses: the small kana
    against the full-size ones (っ / つ, ゃ / や, ァ / ア …), the two scripts' shared shapes
    (へ / ヘ, べ / ベ, ぺ / ペ, り / リ), and katakana against kanji (ニ / 二, ロ / 口, カ / 力,
    タ / 夕, エ / 工, ト / 卜, ハ / 八, ミ / 三, チ / 千). For each such row, what the boxes
    read as, by reader.
  - labelled sheets (start against 6 steps) of the rows in any group, to the user.
- Bars: spread unknown. References for the eight hiragana the runs share (ひ, あ, ぬ, う, さ,
  が, め, を): `g1shared_1125_1538`'s shifted start and 4- and 8-step saves, and
  `g1long_1125_1335`'s 8- and 16-step saves, with other rows in the same run this time.
- Cost: 3.05 s a group, 24 groups a row: 164 rows 3.3 h. Reads: 2 624 renders an arm, about
  24 min, four arms 1.6 h. About 5 h in all (the earlier call: about 10 h).

Interpretation:

- Six steps were not read on held-out pages. `g1shared_1125_1538` reads 148 of 256 at 4 steps
  and 156 at 8 (hayai 147, 167), and あ fell between its 6th step (13 of 16 on the training
  side) and its 8th; six takes most of the gain and stops before that fall in the one case
  seen. The 4-step save is read so that a row better at 4 than at 6 shows.
- The shared displacement comes from 8 hiragana and 8 kanji. Its two class means agree
  (cosine 0.61), and the run's read of the shifted start says whether it carries to the
  katakana and to hiragana outside the 16.
- A fixed 6 leaves slow rows (を, 楽 in the 16) short of their ceiling. The analysis names
  them; whether they get more steps is decided on its table.
- `g1long_1125_1335`'s failures sat in the kanji (行, 二 against ニ). The kana run says
  whether slow rows and reader confusions are rare there before the kanji are taken on.
- The cost grows with the row count, since a render teaches the one row in its caption. The
  1 410 kanji at 6 steps are about 29 h of training; whether they need a cheaper step still
  ([proposal_glyph.md](proposal_glyph.md) § 2) is read after the kana.

## Parked

- A row stopped by its own training-side count, and a step that shrinks with the mean
  gradient's norm. Both were offered for item 4 and left out on the user's call (11-25);
  neither has run. The cases they are for: ぬ in `g1long_1125_1335`, あ in
  `g1shared_1125_1538`.
- The shared part kept as its own vector that every row's renders train: the cheaper step the
  1 410 kanji may need ([proposal_glyph.md](proposal_glyph.md) § 2).
- G1's permuted-label control: not run.
- The x_3 cache: set for the runs after the micro arm and dropped the same day (the user,
  11-25). What it costs is not measured.
- After the kana: [proposal_glyph.md](proposal_glyph.md) (the frequent kanji, words, transfer
  and ship reads). The sentence arms of [proposal.md](proposal.md) wait behind the stage.

## Done

Solo is of 256 held-out pages (16 glyphs, 4 read scenes, 4 seeds), the start rows first.

| item | what | run | report | number | verdict |
|---|---|---|---|---|---|
| – | G0a – c: the tagless frame, the terms' spread, the pixel head against the VAE | `probes/glyph_g0.py` | [glyph_g0](reports/glyph_g0_2026_11_25.md) | pixel head against VAE: sign test p ≥ 0.19 on every term | – |
| – | the scene pool and the crossed grid | `probes/glyph_scenes.py` | [glyph_scenes](reports/glyph_scenes_2026_11_25.md) | 25 of 128 candidates selected; solo EN 0.57, hiragana 0.27, kanji 0.05 | – |
| 1 | G-D's path, and G0d | `g0d_1125_1119` | [gd_g0d](reports/gd_g0d_2026_11_25.md) | the last step's ce gradient five decades over Adam's ε (bar: two) | passed |
| 2 | G1's micro arm: 16 rows, 8 steps | `g1micro_1125_1148` | [g1micro](reports/g1micro_2026_11_25.md) | solo 59 → 132 | adopted on the user's read |
| – | hayai v2.7 beside the tuned reader | `hayai_1125_1234`, `g0h_1125_1243` | [reader_pair](reports/reader_pair_2026_11_25.md), [hayai_g0](reports/hayai_g0_2026_11_25.md) | hayai counts the micro arm 60 → 125; the two row gradients' cosine 0.21 | – |
| – | the reader side of a group, timed | `room_1125_1324` | [reader_room](reports/reader_room_2026_11_25.md) | 0.75 s a group | – |
| 3 | G1 long: 16 rows, 24 steps, hayai's term | `g1long_1125_1335` | [g1long](reports/g1long_2026_11_25.md) | solo 59 → 136 → 175 → 199 at 8, 16, 24 steps | adopted on the user's read |
| – | the 24-step rows on the base model, sign prompt | `sign_1125_1504` | [g1long](reports/g1long_2026_11_25.md), second part | 4 glyphs, 2 seeds | – |
| 3b | the row line: the rows' walk, their place in the table, forward reads | `line_1125_1517`, `line_1125_1527` | [row_line](reports/row_line_2026_11_25.md) | solo at twice the 8-step displacement 122, at the other rows' mean displacement 112, at a drawn direction 59 | – |
| 3c | G1 from the shared start: 16 rows, 8 steps | `g1shared_1125_1538` | [g1shared](reports/g1shared_2026_11_25.md) | solo 112 → 148 → 156 at 0, 4, 8 steps (hayai 104 → 147 → 167) | adopted on the user's read |

- The verdicts on items 2, 3 and 3c are the user's, in each report's Facts. Item 3c's is the
  call for item 4: the shared start at 6 steps.
- hayai's term at weight 0.2 is the user's call (11-25), made on the two hayai reports.
