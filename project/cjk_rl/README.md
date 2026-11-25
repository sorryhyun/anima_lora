# cjk_rl — OCR-reward RL on the JA vocab-pack rows

Opened 2026-11-13 out of the reseed line (`../cjk_anima_reseed/`). The line trains the JA rows
of the vocab pack (jp_v1's rows) on a reward read off the model's own renders by the OCR
reader, with the 4-step turbo stack as the policy. The target is sentence fidelity: the asked
string, in order, at its length, with no invented dialogue around it.

It has two stages. The first (2026-11-13) ran GRPO on sentence captions; its arm R changed
nothing on the ruler ([reports/p1_2026_11_13.md](reports/p1_2026_11_13.md)), and its other
arms ([proposal.md](proposal.md)) wait. The second, the glyph stage (2026-11-25), is the one
running: one glyph a caption, and the reader's loss carried back to that glyph's row by
gradient (G-D). The next step is in [plan.md](plan.md), item 5.

## The glyph stage, as it runs

The idea (user, 2026-11-25): a JA row exists only to be lettered, so a row should draw its
glyph more reliably than the model draws EN text. Code: `src/gd.py` (the path), `src/g1.py`
(the loop and its read).

### What is trained

- A row is one row of the T5-side embedding table that the vocab pack adds to
  `llm_adapter.embed`. As the model sees it, a row is its pack row
  (`anima_cjk_vocab_pack_punct`) plus a delta; the delta starts at jp_v1's (`seed_1008`), and
  only the delta of the glyphs in the run moves.
- Frozen: the DiT (`m15v11`: merged15 plus the turbo v1.1 delta), the pixel head, both
  readers and the detector.
- Lengths are in row norms, the mean norm of a pack row (206.26).

### A render

- Caption: `<scene>. Text reads as "<glyph>".`, the scene with `speech bubble` in it and no
  text tag (the `bubble` frame). The quote holds one trained row.
- 4 Euler steps, cfg 1, shift 3, 768², decoded by the pixel head.
- A group is one glyph on one training scene at 4 seeds.

### From the reader back to the row

- Steps 0 – 2 run for the group's 4 renders in one forward with no graph. The last step runs
  once a render with a graph; its output is the pixel head's image.
- The page is read as any page is: AnimeText finds the boxes, the tuned reader reads each.
  One box is picked (`gd.pick`): among boxes read at confidence 0.6 or over, the one closest
  to the glyph; with none at 0.6, the best of the rest.
- The loss on a page with a box is the tuned reader's cross-entropy of the glyph on the
  picked crop, plus 0.2 times hayai v2.7's cross-entropy on the same crop. A page with no box
  takes −log of the detector's top text score, its signal to draw at all. There is no page
  term.
- The gradient goes from the loss to the image, through the last step to the text embedding
  (the earlier steps are not differentiated), and through the LLM adapter to the glyph's row.

### A row's step

- A render's row gradient is clipped at 0.12 and summed; after 16 renders (4 groups) their
  mean joins the row's momentum (0.7), and the row moves 0.03 row norms against the
  momentum's direction. The step has that length whatever the gradient's size.
- A pass is every (glyph, training scene) once, shuffled, 4 steps a row; each pass draws new
  seeds. No x_3 cache. With `max_steps` a row at that many steps is passed over.
- A render teaches the one row in its caption, so the cost grows with the row count: 3.05 s a
  group, 12 s a row step, 14.2 GiB peak on the 16 GB card.

### The shared start

- The rows of `g1long_1125_1335` moved partly in one common direction. With `[start_shift]` a
  row begins at its start plus the mean displacement of that run's 16 rows (its own left out
  when it is one of them), 0.18 – 0.20 row norms, before its first step.

### Scenes and the read

- Scenes are 25 text-free scene prompts picked by render (`assets/glyph_scenes.json`), split
  by artist folder: 16 training scenes, 4 read scenes (6, 12, 45, 15) from folders no
  training scene comes from.
- The read: every glyph on the read scenes at seeds 0 – 3, 16 pages a glyph and arm, the
  start rows against each save. A page counts as `solo` when a box reads exactly the glyph,
  `on_page` when some box holds it, and carries `extra`, the letters read beyond the glyph's.
  The next glyph's `solo` on the same pages is the null. hayai reads the tuned reader's boxes
  (`hy_solo`). Sheets go to the user with the arms as rows A / B in a drawn order.

### What the reports say about it

- 16 rows trained this way read alone on 59 of 256 held-out pages before and on 136, 175 and
  199 after 8, 16 and 24 steps ([g1long](reports/g1long_2026_11_25.md)). A first look on the
  base model in a sign prompt (8 renders) shows the trained rows lettering one clean glyph.
- A step's direction is mostly noise, and the noise costs little; twice the step loses
  ([row_line](reports/row_line_2026_11_25.md)).
- A trained row keeps its length and its neighbours and turns about 15°; part of the turn is
  common to all rows, and that part alone lifts a row that spent no render (same report).
- From the shared start a row needs 4 – 6 steps fewer, and a row that already reads can be
  pushed past its glyph by the fixed step ([g1shared](reports/g1shared_2026_11_25.md)).
- The 164 kana rows read alone on 0.28 of held-out pages before, 0.50 at the shared start and
  0.68 after 6 steps; 28 rows sit in a group of the plan's, many of them a reader's naming of
  a shape two glyphs share ([g2a](reports/g2a_2026_11_25.md)).

## Files

- [plan.md](plan.md) — the glyph stage's next step, what is parked, and the runs done.
- [turbo_refl.md](turbo_refl.md) — the method as a note: the reward carried through the
  distilled student to rows the base model reads, what has been measured, the conditions it
  seems to need, open points.
- [proposal_glyph.md](proposal_glyph.md) — what follows the kana: the frequent kanji and
  words (G2), transfer and ship reads (G3).
- [proposal.md](proposal.md) — the sentence stage: reward, arms R / A (GRPO), D (reader-NLL
  reward gradient) and R-credit, phases P1 – P3 / D0 – D1 and their gates, the SFX track.
- [reports/](reports/) — finished probes:
  - [ruler_turbo_2026_11_13.md](reports/ruler_turbo_2026_11_13.md) — the 64-string ruler on
    base / merged14 / merged14 + v3: the soup loses the string, the turbo is a fair proxy.
  - [p0_2026_11_13.md](reports/p0_2026_11_13.md) — P0: sampler (SDE a = 0.7), group spread
    (passes), cost of one update.
  - [p1_2026_11_13.md](reports/p1_2026_11_13.md) — P1: 300 GRPO groups on the rows change
    nothing on the ruler; arm R fails.
  - [reward_reader_2026_11_13.md](reports/reward_reader_2026_11_13.md) — the reader beside the
    update: int8 / fp8 dropped, group-batched read without the whole page, tuned reader only.
  - [position_read_2026_11_25.md](reports/position_read_2026_11_25.md) — the ruler's reads of
    record aligned per bubble: hits by position, substitutions, and the EN references' own
    text on the base, merged14 and the turbo.
  - [turbo_read_2026_11_25.md](reports/turbo_read_2026_11_25.md) — fresh renders on merged15 +
    the turbo v1.1 delta: the ruler at four seeds with its EN references, and a kanji grid.
  - [glyph_g0_2026_11_25.md](reports/glyph_g0_2026_11_25.md) — G0a – c of the glyph proposal:
    the tagless frame with and without `speech bubble` against one-token EN words, the reward
    terms' spread inside a group, the pixel head against the VAE.
  - [glyph_scenes_2026_11_25.md](reports/glyph_scenes_2026_11_25.md) — the glyph stage's scene
    pool picked by render (128 text-free captions in two rounds, by artist folder; 25
    selected), and every glyph on the same four scenes.
  - [gd_g0d_2026_11_25.md](reports/gd_g0d_2026_11_25.md) — G0d: G-D's path built and checked,
    its memory and seconds, and the size and direction of the gradient it puts on a row.
  - [g1micro_2026_11_25.md](reports/g1micro_2026_11_25.md) — G1's micro arm: 16 rows, 8 steps
    each on the reader's loss; solo on held-out scenes 59 → 132 of 256.
  - [reader_pair_2026_11_25.md](reports/reader_pair_2026_11_25.md) — hayai v2.7 Supernova
    beside the tuned reader on g1micro's read pages and the crossed grid: hayai counts
    g1micro's solo 60 → 125; the crops the two differ on.
  - [hayai_g0_2026_11_25.md](reports/hayai_g0_2026_11_25.md) — G0h: hayai's cross-entropy
    beside the tuned reader's on G1's path; the two row gradients' sizes and cosine.
  - [g1long_2026_11_25.md](reports/g1long_2026_11_25.md) — G1 long: the 16 rows over 24 steps
    with hayai's term, read at 8, 16 and 24; solo on held-out scenes 59 → 136 → 175 → 199 of
    256. Second part: the 24-step rows on the base model in the sign prompt (`sign_1125_1504`,
    4 glyphs, 2 seeds).
  - [reader_room_2026_11_25.md](reports/reader_room_2026_11_25.md) — the reader side of G1's
    group alone: 0.75 s a group by part, the GPU busy for a third of it; a decode cap and 4
    pages a call timed.
  - [row_line_2026_11_25.md](reports/row_line_2026_11_25.md) — G1 long's rows: a step's
    direction against a walk on noise, and the rows read with no training at twice their
    8-step displacement (122 of 256) and at the other rows' mean displacement (59 → 112; a
    drawn direction 59).
  - [g1shared_2026_11_25.md](reports/g1shared_2026_11_25.md) — G1 from a shared start: 8
    steps a row after the other rows' mean displacement; solo 112 → 148 → 156 of 256 at 0, 4
    and 8 steps (the parent 136 at 8, 175 at 16); 行 starts, あ falls to あら.
  - [g2a_2026_11_25.md](reports/g2a_2026_11_25.md) — G2a: the 81 hiragana and the 83
    katakana, 6 steps a row from the shared start, small kana folded; solo 743 → 1 322 →
    1 686 → 1 785 of 2 624 at the start, the shifted start, 4 and 6 steps; the rows not
    learned, past their glyph, or misread by a reader.
- `src/pool.py` — P1's prompt pool: the training captions with each quoted OCR string swapped
  for another OCR string of its length (Manga109-s, or another caption's), nothing else
  changed; → `output/cjk_rl/pool/pool.json`.
- `src/reward.py` — the read terms: r_seq over every asked string (one bubble each), g_p over
  all of them.
- `src/grpo.py` — the GRPO loop (arm R).
- `src/gd.py` — G-D's path: the policy with a step-by-step backward to the rows, and the
  reader's cross-entropy and the detector's score on a page tensor.
- `src/g1.py` — G1's loop on that path (a row steps along the momentum of its renders' mean
  gradient, from the start rows or from a shared start) and its read on held-out scenes, at
  each save, with blind sheets.
- `probes/` — `grpo_p0.py` (P0), `reader_batch.py` (the reward reader). Both run on the
  reseed line's package, ruler and readers (`../cjk_anima_reseed/{reseed,src}`).
  `position_read.py` (CPU) reads the ruler results' `renders.json`. `turbo_read.py` renders
  and reads the ruler and a kanji grid on `m15v11` (pixel-head decode by default); its render
  (every caption's text prepared before the loop, 4 images a forward) and read (16 pages a
  call) are the ones `glyph_g0.py` runs on.
  `glyph_g0.py` and `glyph_scenes.py` are the glyph stage's forward probes; `gd_g0d.py`
  measures the row gradient of `src/gd.py`.
  `reader_pair.py` reads pages on disk with the tuned reader and hayai v2.7 on the same
  crops; `hayai_g0.py` measures hayai's loss beside the tuned reader's on G1's path.
  `reader_room.py` times the reader side of G1's group by part, without the DiT, on pages a
  G1 run wrote. `base_sign.py` renders a G1 save's rows on another model of the ruler's table
  in a prompt of its config and reads them beside the start rows. `row_walk.py` (CPU) counts
  how far a G1 run's rows moved and how much of it is one direction, `row_geom.py` (CPU) where
  the trained rows sit in the table before and after; `row_line.py` reads a run's saved rows
  displaced along what they learned, with no training. `g2_rows.py` (CPU) reads a G1 run's
  records and log per row and sorts the rows into the plan's groups (not learned, pushed
  past the glyph, a reader's problem), with labelled sheets of those rows.
- `configs/` — one file a run, its question and gate in the header.
- `assets/glyph_scenes.json` (gitignored) — the selected scene prompts, written by
  `glyph_scenes.py screen`.
- `results/` (gitignored) — one directory per probe run; renders under `output/cjk_rl/`.

## Rules

Every read is the tuned PaddleOCR-VL-1.6 reader (`anime_tools.ocr.reader.OcrReader`); stock
VL16 is not a reader.

The line (the user's calls, 2026-11-13):

- The goal is the intended sentence (order, length, no invented text), not crisper glyphs.
  Rewards and reads are sequence-level. SFX is in scope and needs a style guard.
- Sampler: SDE a = 0.7.
- Prompt pool: every caption with a JA text / SFX clause, only the quoted strings swapped. It
  holds out the ruler's images and 5-grams and the `$CHARACTER_REF_EXCLUDE` tags. No rating
  filter.
- Rows can learn order (proposal § 2), so arm A is a sample-count fallback, not an order fix.
- Arm A starts with the `src/grpo.py` fixes: summed log π, G 16–24.

The glyph stage (the user's calls, 2026-11-25):

- G-D is the arm; G-R (GRPO on the glyph reward) is out of the stage.
- Scenes come from artists whose captions hold little text, or from captions that had no OCR
  clause, picked by the relaxed rule of
  [reports/glyph_scenes_2026_11_25.md](reports/glyph_scenes_2026_11_25.md). Training and read
  scenes are split by artist folder.
- A box the reader reads under a confidence of about 0.6 counts as nothing drawn. A page where
  AnimeText finds no box takes the detector's own score as its signal to draw.
- No page term in the loss. No x_3 cache.
- The loss carries hayai v2.7's cross-entropy of the glyph on the same crop at weight 0.2
  (two readers for robustness).
- Renders go 4 images a forward; reads go 16 pages a call.
- Words are real words only, no random strings.
