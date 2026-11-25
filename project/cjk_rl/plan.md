# Plan: the glyph stage on G-D (opened 2026-11-25)

The glyph-first stage on G-D: the OCR reader's loss carried back to the rows. Items 1 and 2
(the path, G0d, G1's micro arm) are done; item 3 is the next run. What comes after the
hiragana is in [proposal_glyph.md](proposal_glyph.md); results go to [reports/](reports/). A
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
  refreshed every 4 passes on runs after the micro arm.
- The user's call (11-25): from the next run on, the loss carries hayai v2.7's cross-entropy of
  the glyph on the same crop at weight 0.2 (`hayai_weight`, `src/g1.py`; two readers for
  robustness). [reports/reader_pair_2026_11_25.md](reports/reader_pair_2026_11_25.md)
  (`hayai_1125_1234`): the two readers on the micro arm's pages and where they differ.
  [reports/hayai_g0_2026_11_25.md](reports/hayai_g0_2026_11_25.md) (`g0h_1125_1243`): the two
  terms' row gradients are of one size (hayai / tuned 0.93 at the median) and share a direction
  on a render (cosine 0.21, null 0); hayai's term costs 0.05 s a render.
- Cost, measured: 2.90 s a group of 4 renders, 12 s a row step, 13.1 GiB peak.
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

## 3. G1 long — the sixteen rows to their ceiling

Proposed (Claude), not queued. Parent: `g1micro_1125_1148`.

Facts:

- From the start rows again, 6 passes (24 steps a row) with the x_3 cache on (the input of
  the last step kept per glyph, scene and seed; drawn again, on new seeds, every 4 passes) and
  `hayai_weight` 0 → 0.2 (the user's call). Nothing else changed. Three things change together:
  length, the cache and the hayai term. The read at 8 steps differs from the parent by the
  cache and the term, and the run cannot separate those two.
- To build before it runs: the cache and the saves at 8, 16 and 24 steps with a read each
  (`src/g1.py` holds neither yet).
- The rows are saved and read at 8, 16 and 24 steps on the parent's read scenes and seeds,
  with the parent's tables and blind sheets (start against the last save).
- Read: solo by save, per glyph, under the tuned reader and under hayai (`hy_solo`, the same
  boxes); the steps at which a glyph stops rising; を, 行, 楽; ぬ and 死, whose pages changed
  without a gain; the crops one reader alone reads as the glyph, start against the last save;
  the 8-step save against the parent's 132 (hayai's count of the parent: 125, WebP).
- Bars: spread unknown. The 8-step save against the parent is the first pair of runs that
  differ only by the cache and the seeds of pass 1.
- Cost: about 60 min of training (two passes at 3.1 s a group, four estimated at 1.9 s; the
  hayai term adds 0.2 s a group) and about 10 min of reads.

Interpretation:

- This sets item 4's steps a row and says what the cache costs before 81 rows run on it.
- Rows that already read well may be pushed off by more steps. If the 16- or 24-step save
  shows it, item 4 stops a row by its own training-side reads.

## 4. G2a — the hiragana

After item 3. Proposed (Claude), not queued.

Facts:

- Rows: the 81 hiragana with one trained row in the frame, G1's loop, scenes and split.
- Steps a row: the save of item 3 at which solo stops rising.
- Read: every row on the read scenes and seeds against the start rows (81 × 16 pages an arm,
  about 11 min an arm), blind sheets for a sample of rows picked before the run.
- Cost: 12 s a row step without the cache. 81 rows at 8 steps 2.1 h, at 16 steps 4.3 h, at 24
  steps 6.5 h; with the cache an estimated two thirds of that.

Interpretation:

- The cost grows with the row count, since a render teaches the one row in its caption. The
  katakana (83) double it; the 1 410 kanji need a cheaper step
  ([proposal_glyph.md](proposal_glyph.md) § 2).
