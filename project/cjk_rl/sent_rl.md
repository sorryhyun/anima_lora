# Sentence RL — kana sentences from kmix (rough plan, opened 2026-11-26)

What the sentence run trains, on what strings, and how it is read. The run's settings,
question and gate go in its config header when it is submitted; this file is the plan before
that and names the records it rests on. Where it stands in the line: [plan.md](plan.md) item 7.

## Decided (the user's calls, 2026-11-26)

- Start: the kana rows `kmix_1126_1158` (g2b's 16-step rows with ten of g2a's).
- Strings: the OCR reader's training labels, as the reader's rule normalises them
  (`project/ocr_reader/core/crop_dataset.py::normalize_target`, `TARGET_NORM = 10`).
  Punctuation stays in. A blank heart is written as the filled one (`♡` → `♥`) in the quote
  and in the loss's target: the reader in use does not read `♡` yet.
- The read's strings come from the reader's val split and are in no train label.
- A small kana is trained as itself, its target and its read the small glyph (`fold_small`
  off). kmix's 20 small rows were trained with the full-size glyph as their target.
- Guards: `taper`, `near_score` 0.5, `[back]` with `over` 0.5.
- Ship: the rows this run trains from kmix, if they read well, go out as a v2 preview with
  the kana alone. The kanji are taken as a matter of scale and come after.
- The backward covers the last two steps, σ 0.75 → 0.5 → 0 (on `tswap_1126_1209`'s table).
- The tuned reader runs on FlashAttention 2 with `reader_compile` on (on `timeb_1126_1238`'s
  seconds, `reports/reader_room_2026_11_25.md`, third part). `skip_lone_read` does not apply:
  the run needs every page's read for the breaks and the score.

## Facts it rests on

- `wread_1126_1207` (forward only, 48 pool strings, 96 pages a length, jp_v1's rows → kmix's
  rows): the whole string alone in a box 19 → 52 at 2 glyphs, 2 → 21 at 3, 2 → 13 at 4,
  0 → 0 at 5 – 8; share of positions hit 0.41 → 0.76, 0.28 → 0.62, 0.27 → 0.53, 0.20 → 0.35;
  kmix's hit share by position 0.66, 0.52, 0.47, 0.38, 0.26; glyphs drawn over glyphs asked
  0.84 at 5 – 8; the first glyph alone on 11 of 384 pages. Spread unknown.
  `results/20261126-1215-row_line-wread_1126_1207/` (local).
- `tswap_1126_1209` (forward only, kmix's rows: a render's first k steps on string A, the
  rest on string B; k = 0 is B's own render, k = 2 gives B to σ 0.75 → 0.5 → 0, k = 3 to
  σ 0.5 → 0 alone; small kana folded; 128 pages a cell for a glyph, 96 for a length).
  Spread unknown. `results/20261126-1227-text_swap-tswap_1126_1209/` (local).

  | | B alone, k 0 / 2 / 3 | A alone, k 0 / 2 / 3 | B's positions hit, k 0 / 2 / 3 | A's positions hit, k 0 / 2 / 3 |
  |---|---|---|---|---|
  | a glyph | 80 / 71 / 14 | 0 / 7 / 49 | 0.70 / 0.63 / 0.18 | 0.00 / 0.06 / 0.41 |
  | 2 glyphs | 63 / 46 / 10 | 0 / 1 / 19 | 0.82 / 0.72 / 0.28 | 0.05 / 0.06 / 0.37 |
  | 3 | 22 / 20 / 5 | 0 / 0 / 4 | 0.64 / 0.57 / 0.28 | 0.12 / 0.12 / 0.31 |
  | 4 | 13 / 7 / 1 | 0 / 0 / 1 | 0.55 / 0.50 / 0.27 | 0.04 / 0.05 / 0.20 |
  | 5 – 8 | 0 / 0 / 0 | 0 / 0 / 0 | 0.36 / 0.35 / 0.22 | 0.12 / 0.11 / 0.18 |
- Where a quote's characters go on the T5 side (the active pack with `glyph_route`, jp_v1's
  delta in row norms): a kana its own row (1.2 – 1.5); `ー` its own row (1.52; `―` folds to
  it); `「` `」` 0.9; `。` 0.48, `、` 0.35; `〜` and `~` one row, 0.42; `♥` 0.46, `♡` 0.48, a row
  each; a dot run inside a string one row, 0.38; `！` `？` fold to ASCII and are T5's own
  tokens, with no row to train.
- The reader's main labels (`load_mainsub`, `OCR:` rows), distinct targets made of kana and
  those marks: train 28 975 (2 – 4 glyphs 9 552, 5 – 8 11 739, 9 – 12 4 570, 13 – 20 2 565,
  over 20 460), val 3 517, of them 2 396 in no train label. 12 547 of the train targets hold
  a line break; `…` is in 5 738, `ー` in 5 628, `!` in 4 913, `?` in 2 827, `♡` in 244.
- The current pool (`output/cjk_rl/pool/pool.json`) holds 355 kana-only strings.
- `reports/gd_g0d_2026_11_25.md` (single glyphs): a step's share of the four-step ce gradient
  on a row, steps 0 – 3 (σ 1 → 0.9 → 0.75 → 0.5 → 0): 0.48 / 0.40 / 0.33 / 0.08; one step's
  backward 0.95 s for a group's 4 renders (a group 3.0 – 3.2 s with the last step's alone).

## The run

What stays as the glyph stage has it (README, "The glyph stage, as it runs"): the bubble
frame, 4 Euler steps at cfg 1 and shift 3, the pixel head, the box picked by its sequence
score to the string, the tuned reader's cross-entropy of the string on the picked crop plus
0.2 of hayai's, the detector's term on a page with no box, no page term, the fixed-length
step along a row's momentum.

What changes:

- The quote holds a string, and a render's gradient goes to every row in it. A row steps when
  16 renders that hold it have come in, whatever strings they were.
- Rows trained: the 164 kana, `ー`, `。`, `、`, `～`, the dot row, `♥`, `「`, `」`, each from
  where kmix or jp_v1 has it. `♡`'s row is not in the run.
- A group is one string on one training scene at 4 seeds. A string is drawn once; the scenes
  go round the 16 training scenes.
- Strings: drawn from the train labels over lengths 2 – 12 (longer ones wait), with the
  ruler's 5-grams and `$CHARACTER_REF_EXCLUDE` held out as the pool's rule has it. The draw
  is a local file (Manga109-s is not redistributed); a config names the file and its seed.
- Line breaks: the quote holds the string with no break. The loss's target takes its breaks
  from the page's own read, aligned to the string, so where a render wraps costs nothing.
- A row's score for the guards: over the renders that hold it, the positions of its glyph
  that the aligned read hits; a look-alike counts 0.5.
- The backward: the last two steps (σ 0.75 → 0.5 → 0), the user's call on
  `tswap_1126_1209`'s table. The image's cotangent goes back through the last step to its
  input, and from there through the step before; the two steps' cotangents of the embedding
  are summed before the rows. One step's graph is alive at a time. A one-step arm on the
  same draw of strings is not planned.

The read, at each save, against kmix's rows:

- Strings of the val split that are in no train label, by length (2 – 4, 5 – 8, 9 – 12), on
  the read scenes and seeds: the whole string alone, positions hit, glyphs drawn over asked,
  under the tuned reader and hayai. Two counts side by side: with punctuation
  (`textnorm.exact_key`) and without it (the line's `norm`, as every read so far).
- A sample of train-label strings on the same scenes, for the tuned reader's count over
  hayai's on strings it was trained to write against strings it was not.
- The 164 kana alone, for what the run costs them.
- Blind sheets, a string a file, kmix's rows against the run's. The user's read decides.

Cost, to be set in the config: a group is about 4.0 s with the two-step backward (3.05 s
with the last step's alone, and 0.95 s a step's backward), so 1 000 groups are 67 min. A group of a 7-glyph string is 28
row-renders, against 4 in the glyph stage.

## Interpretation

- The rows trained alone already carry into short strings, so the run starts from a working
  state, and what it has to mend is glyphs dropped from a string and the fall along its
  length.
- A row is one vector at every position, so order and count over a long string may be out
  of its reach; the read by length shows where that begins.
- The swap puts most of what a page letters in the step σ 0.75 → 0.5: with B's text on the
  last two steps the page is close to B's own render, and with it on the last step alone
  the page mostly keeps A, for a single glyph as for a string. The last step still moves
  the read part of the way, and its gradient alone took the single glyphs from 0.28 to 0.68
  of their pages, since a row moved by it acts at every step. So the swap does not say the
  one-step backward fails on strings; it says the two-step backward reaches the step where
  the string is set.
- Reusing the reader's labels as prompts does not touch the reader. What it can do is flatter
  the read: a reader writes a sentence it was trained on more readily from poor glyphs. The
  val-only strings, hayai beside the tuned reader and the user's sheets are the checks.
- Frequent rows (い, っ, ん, ー) will take many more steps than rare ones. The guards shorten
  a row's step as it reads; whether a cap on a row's steps is needed as well is open.

## Open

- A cap on a row's steps beside the guards.
- The mix of lengths in the draw, and when strings over 12 glyphs come in.
- `♡` at ship: its row is left as jp_v1 has it; a fold of `♡` to `♥` in the pack would send
  both to the trained row.
- The kana that few label strings hold (ヂ, ヅ, ゔ, ヲ): whether single-glyph groups are mixed
  in for them.

## Order of work

1. `tswap_1126_1209`'s read.
2. `src/g1.py`: a quote of several rows, a row's renders counted over strings, the score per
   position, the target's breaks from the read, the backward over two steps (a key, 1 as
   the runs so far).
3. The draw of strings as a local file, counted once when built (lengths, rows covered).
4. A smoke, then a micro arm (about 30 min) with its read, then the run on the user's word.
