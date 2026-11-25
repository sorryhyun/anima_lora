# Plan: the glyph stage on G-D (opened 2026-11-25)

The glyph-first stage on G-D: the OCR reader's loss carried back to the rows. Items 1 to 3
(the path, G0d, G1's micro arm, G1 long) are done; item 4 (the hiragana and the katakana at
16 steps a row) is next and runs on the user's word. What comes after the
kana is in [proposal_glyph.md](proposal_glyph.md); results go to [reports/](reports/). A
run's settings, question and gate go in its config header (`configs/<handle>_mmdd_hhmm.toml`)
when it is submitted, and this file then names the run.

## Carried

Facts:

- Policy: `m15v11` (merged15 + the turbo v1.1 delta), 4 steps, cfg 1, shift 3, 768², pixel-head
  decode. Trains: the rows (`ExtDelta`), warm from `seed_1008` (jp_v1). Frozen: the DiT, the
  delta, the pixel head, the reader.
- Frame: `<scene>. Text reads as "<string>".`, no text tag.
- The user's calls (2026-11-25):
  - G-D is the arm; G-R is out of the glyph stage.
  - Scenes come from artists whose captions hold little text, or from captions that had no OCR
    clause; the selected prompts live in `assets/glyph_scenes.json` (gitignored).
  - The selection rule is the relaxed one: the plain no-clause renders letter nothing, the
    bubble ones on at most 1 of 4 seeds, and the probe strings (`stop`, う) read exactly on 2 of
    their 8 bubble renders or more (`glyph_scenes.py select`). 25 of the 128 screened
    candidates pass (two rounds), 23 of them from the low-text folders, 13 folders in all.
  - A box the OCR reader reads under a confidence of about 0.6 counts as "nothing drawn", and
    the loss is told so.
  - A page where AnimeText finds no box gets the detector's own score as its signal to draw.
  - Renders go 4 images a forward; reads go 16 pages a call through the detector and the reader.
- [reports/glyph_g0_2026_11_25.md](reports/glyph_g0_2026_11_25.md): the pixel head reads as
  the VAE does (sign test p ≥ 0.19 on every term, Euler and SDE). Without `speech bubble` half
  the pages hold no letters; with it 11 %.
- [reports/glyph_scenes_2026_11_25.md](reports/glyph_scenes_2026_11_25.md): the screen (two
  rounds) and the selection. On four clean scenes, bubble frame, 16 renders a string: `solo` EN
  0.57, hiragana 0.27, kanji 0.05.
- The reader's confidence on that crossed grid, rows arm (896 pages; counted from the run's
  `reads.json`, renders `output/cjk_rl/glyph_scenes/m15v11_px/cross/seed_1008/`, local): no box
  on 193 pages, a box but none at 0.6 or over on 195 more. Of the 130 boxes that read the asked
  glyph exactly, 9 are under 0.6; of the 660 that read something else, 36 %.
- G1's loop as it ran (`src/g1.py`, `configs/g1micro_1125_1148.toml`): the last step's
  backward, 1 render a forward; the reader's cross-entropy on the picked crop, the detector's
  score on a page with no box, no page term; a render's row gradient clipped at 0.12; a row
  steps every 16 renders along the momentum (0.7) of its mean gradients, 0.03 row norms a step.
  Training scenes: the 16 of the 9 folders that hold no read scene; read scenes 6, 12, 45, 15.
- The user's calls (11-25) on the loop: no page term; the split by folder; an x_3 cache
  refreshed every 4 passes on runs after the micro arm. Later the same day: item 3 runs
  without the cache.
- The user's call (11-25): from the next run on, the loss carries hayai v2.7's cross-entropy of
  the glyph on the same crop at weight 0.2 (`hayai_weight`, `src/g1.py`; two readers for
  robustness). [reports/reader_pair_2026_11_25.md](reports/reader_pair_2026_11_25.md)
  (`hayai_1125_1234`): the two readers on the micro arm's pages and where they differ.
  [reports/hayai_g0_2026_11_25.md](reports/hayai_g0_2026_11_25.md) (`g0h_1125_1243`): the two
  terms' row gradients are of one size (hayai / tuned 0.93 at the median) and share a direction
  on a render (cosine 0.21, null 0); hayai's term costs 0.05 s a render.
- Cost, measured: 2.90 s a group of 4 renders, 12 s a row step, 13.1 GiB peak.
- The reader side of a group alone (`room_1125_1324`,
  [reports/reader_room_2026_11_25.md](reports/reader_room_2026_11_25.md)): read 0.35 s and
  loss 0.41 s a group with hayai's term, the GPU's kernels busy for 33 % and 40 % of them; a
  decode cap saves 0.008 s, and 4 pages a call saves 0.24 s against the 0.40 s forward it
  needs.
- Rows with one trained row in the frame: 81 hiragana, 83 katakana.
- Built: `probes/turbo_read.py` (render: all text prepared before the loop; read),
  `probes/glyph_g0.py` (frames, the grid's numbers, the terms), `probes/glyph_scenes.py` (the
  screen, the crossed grid), `src/gd.py` (G-D's path) and `probes/gd_g0d.py`.

## 1. G-D's path, and G0d — done

`src/gd.py`, `probes/gd_g0d.py`, run `g0d_1125_1119`
([reports/gd_g0d_2026_11_25.md](reports/gd_g0d_2026_11_25.md)).

Facts:

- The path's three checks hold. A backward forward holds 1 render on the 16 GB card.
- The last step's ce gradient on a render that misses its glyph is five decades over Adam's ε
  (bar: two).
- Two seeds of a glyph have gradient cosine 0.01, the null between glyphs 0.003 – 0.017; the
  part that follows the label has cosine 0.025 between two renders of a glyph against 0.001.

## 2. G1's micro arm — done

Run `g1micro_1125_1148` ([reports/g1micro_2026_11_25.md](reports/g1micro_2026_11_25.md)).

Facts:

- 16 rows, 8 steps each (128 renders a row), 24.7 min.
- Read on the held-out scenes, 256 pages an arm, start → trained: solo 59 → 132 (13 glyphs
  up, 1 down, p 0.002), the next glyph's solo 0 → 0, extra letters a page 3.44 → 2.34, no-box
  pages 44 → 23. 人 0 → 7; を 2 → 0, 行 0 → 0, 楽 0 → 1. The best row reads 15 of 16.
- The user's read of the blind sheets (11-25): success, clear at a glance.
- Not run: the permuted-label control.
- A second reader on the same pages (`hayai_1125_1234`,
  [reports/reader_pair_2026_11_25.md](reports/reader_pair_2026_11_25.md)): hayai v2.7 counts
  solo 60 → 125, 12 glyphs up, 2 down.

Interpretation:

- Eight steps along the averaged gradient move a row toward its glyph on scenes it was not
  trained on. No row is at its ceiling, so the steps a row needs are still open; item 3 reads
  them.

## 3. G1 long — the sixteen rows to their ceiling — done

Run `g1long_1125_1335` (`configs/g1long_1125_1335.toml`,
[reports/g1long_2026_11_25.md](reports/g1long_2026_11_25.md)). Parent: `g1micro_1125_1148`.

Facts:

- 24 steps a row in 78.0 min (3.05 s a group), 14.2 GiB peak.
- Solo on the held-out scenes, of 256, start → 8 → 16 → 24 steps: 59 → 136 → 175 → 199 (hayai
  on the same boxes: 59 → 136 → 171 → 189); 15 glyphs up and none down at 16 and at 24; the
  null 0 throughout; no-box pages 44 → 14.
- From 16 to 24 steps: を 4 → 10, 楽 9 → 14, 人 8 → 13, が 11 → 14; the rows at 12 – 16 by the
  16-step save move by at most 3. ぬ 15 → 13, and on the training scenes its count of renders
  whose picked box reads the glyph falls from 15.0 to 9.0 of 16 over the last two passes.
- 行 0 at every save. 二 7 by the tuned reader and 0 by hayai, which reads the same boxes as ニ.
- The 8-step save against the parent (hayai's term the one difference): 136 against 132.
- The 24-step rows on the base model in the sign prompt (`sign_1125_1504`, the same report's
  second part; う, が, 楽, 人 at 2 seeds): the trained rows letter the glyph once on all 8
  renders (Claude's labelled read), the start rows repeat it or add other characters on 5 of 8.
- The user's read of the blind sheets (start against the 24-step save), 11-25: the training
  helped, plain without a blind read; and the question whether about 16 steps are enough.

As it was set up:

- From the start rows again, 6 passes (24 steps a row), `hayai_weight` 0 → 0.2 (the user's
  call), no x_3 cache (the user's call, 11-25): every pass draws its own seeds, as the parent
  did. Nothing else changed. Two things change together: length and the hayai term. Passes 0
  and 1 run on the parent's seeds and order, so the 8-step save differs from the parent by the
  term alone.
- Built for it: `save_passes` in `src/g1.py` (the rows written and read in the process after
  the named passes, the start rows read before training). The reader side stays as it is
  (`room_1125_1324`).
- The rows are saved and read at 8, 16 and 24 steps on the parent's read scenes and seeds,
  with the parent's tables and blind sheets (start against the last save).
- Read: solo by save, per glyph, under the tuned reader and under hayai (`hy_solo`, the same
  boxes); the steps at which a glyph stops rising; を, 行, 楽; ぬ and 死, whose pages changed
  without a gain; the crops one reader alone reads as the glyph, start against the last save;
  the 8-step save against the parent's 132 (hayai's count of the parent: 125, WebP).
- Bars: spread unknown. The 8-step save against the parent differs by the hayai term; no
  twin of the parent ran, so the pair sets no bar.
- Cost: about 80 min of training (1 536 groups at 3.1 s; the hayai term adds 0.2 s a group)
  and about 10 min of reads.

Interpretation:

- This sets item 4's steps a row. What the cache costs is not read by this run.
- Rows that already read well may be pushed off by more steps. If the 16- or 24-step save
  shows it, item 4 stops a row by its own training-side reads.
- As read: held-out solo rises at every save and the late rows need the 24 steps; one fast row
  (ぬ) was better at 16. The user's call for item 4 (11-25): 16 steps, fixed.

## 4. G2a — the hiragana and the katakana

The user's call (11-25, after item 3's sheets): "히라가나 및 가타카나로 가고, 16스텝, 특히 못배우는
경우나 ocr문제는 한자에 몇몇 있는것으로 보이니 히라 가타 보고 분석". Not queued; it runs on the
user's word.

Facts:

- Rows: the 81 hiragana and the 83 katakana with one trained row in the frame, from the start
  rows (`seed_1008`). G1's loop as `g1long_1125_1335` ran it: scenes, split, update,
  `hayai_weight` 0.2, no x_3 cache. Parent: `g1long_1125_1335`; what changes is the rows and
  the steps.
- Steps a row: 16 (4 passes), fixed (the user's call). Saves read at 8 and 16 steps.
- Two runs, the hiragana first, then the katakana.
- To build before it runs: the katakana as a class in `src/g1.py` (`CLASSES` holds `hira` and
  `kanji`) and a class a config may leave out; the read's analysis below (a script over
  `records.json` and `log.jsonl`).
- Read, per row, the start rows against each save on the read scenes and seeds (16 pages a row
  and arm): solo under the tuned reader and under hayai, on_page, extra, no-box pages, the
  picked crop's ce; the training log's count a pass.
- The analysis the user asked for, per row, with the groups written here before the run:
  - not learned: solo under 8 of 16 at 16 steps under both readers, with the training-side
    count a pass beside it (still climbing, or level).
  - a reader's problem: the two readers apart by 4 or more of 16, or the boxes read as a
    look-alike of the asked glyph. Look-alikes counted apart from misses: the small kana
    against the full-size ones (っ / つ, ゃ / や, ァ / ア …), the two scripts' shared shapes
    (へ / ヘ, べ / ベ, ぺ / ペ, り / リ), and katakana against kanji (ニ / 二, ロ / 口, カ / 力,
    タ / 夕, エ / 工, ト / 卜, ハ / 八, ミ / 三, チ / 千). For each such row, what the boxes
    read as, by reader.
  - labelled sheets (start against 16 steps) of the rows in either group, to the user.
- Bars: spread unknown. Item 3's 16-step save is the reference for the eight hiragana it
  shares (ひ 12, あ 14, ぬ 15, う 12, さ 13, が 11, め 15, を 4 of 16), with other rows in the
  same run this time.
- Cost: 12.2 s a row step (`g1long_1125_1335`: 3.05 s a group). 81 rows at 16 steps 4.4 h,
  83 rows 4.5 h; reads about 11 min an arm and run, three arms a run. About 10 h in all.

Interpretation:

- Item 3's failures sit in the kanji: 行 never started, and 二 is read as ニ by one reader.
  Among its eight hiragana only を was slow (4 of 16 at 16 steps, 10 at 24). The kana runs say
  whether slow rows and reader confusions are rare there, as that sample suggests, before the
  kanji are taken on.
- A fixed 16 leaves a row like を short of its ceiling. The analysis names those rows; whether
  they get more steps is decided on its table.
- The cost grows with the row count, since a render teaches the one row in its caption. The
  1 410 kanji need a cheaper step ([proposal_glyph.md](proposal_glyph.md) § 2).
