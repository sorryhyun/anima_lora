# Plan: the glyph stage on G-D (opened 2026-11-25)

The next steps of the glyph-first stage: build the path that carries the OCR reader's loss back
to the rows (G-D), measure it (G0d), then train 16 rows (G1). What comes after G1 is in
[proposal_glyph.md](proposal_glyph.md); results go to [reports/](reports/). A run's settings,
question and gate go in its config header (`configs/<handle>_mmdd_hhmm.toml`) when it is
submitted, and this file then names the run.

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
- Built: `probes/turbo_read.py` (render: all text prepared before the loop; read),
  `probes/glyph_g0.py` (frames, the grid's numbers, the terms), `probes/glyph_scenes.py` (the
  screen, the crossed grid).

## 1. G-D's path, and G0d

Not started. No run yet; the probe will be `probes/gd_g0d.py` on `src/gd.py`.

Facts:

- The loss, a render: the reader's teacher-forced cross-entropy of the asked string on a crop,
  plus a page term, the pixel MSE to the anchor render outside the crop. The anchor is the
  scene with no clause, plain frame, same noise.
- The crop comes from an ordinary read of the same render (detector and reader, detached).
  Three cases, counted on the crossed grid's 896 JA pages:
  - a box read at confidence 0.6 or over (57 % of the pages): the crop is that box (the best
    `r_seq` to the asked string, then the highest confidence);
  - boxes, none at 0.6 (22 %): "nothing drawn" by the user's rule. The cross-entropy on the
    best of those boxes is already that signal: it is defined on any crop and is high there;
  - no box at all (22 %): no crop, so the reader's loss has nothing to act on. The page's
    term is the detector's score (the user's call, 11-25): AnimeText is a torch YOLO12 whose
    per-anchor text score is differentiable in the pixels, so −log of its top score on the
    page asks for a text block, and the reader's loss names the glyph once a box is there.
    Not taken: the reader's cross-entropy on the whole page (the reader was tuned on crops).
- Gradient path: reader → crop pixels → pixel head → DiT step(s) → llm_adapter → rows.
- To build:
  - the detector's forward on the pixel tensor with its scores kept in the graph
    (`AnimeTextDetector.forward_batch` runs under `inference_mode` on a letterboxed array);
  - the reader with tensors in and labels in (`OcrReader` decodes PIL crops with `generate`
    under `inference_mode`); the crop taken from the pixel tensor by resampling;
  - a pixel tap that keeps the graph (`PixelTap` detaches);
  - the backward through the last step, and through all four with checkpointing.
- G0d measures, on the 16 rows of item 2 and scenes 6 and 12, 4 seeds each:
  - the row gradient's size against Adam's ε, for the last step alone and for all four;
  - peak VRAM and seconds a step for both, the reader resident;
  - the same on pages with no box for the detector-score term, and its cosine to the row's
    gradient on pages that hold the glyph;
  - whether the gradient is about the glyph: the cosine of a row's gradient between two seeds,
    against the cosine between its gradient under the asked label and under another glyph's
    label on the same render.
- Read: Claude's, numbers only; it sets item 2's settings (which steps, the page term's weight
  from the two terms' gradient norms on the rows, the learning rate). No renders are judged.
- Bars: a row gradient two decades over Adam's ε (P1's was at it,
  [reports/p1_2026_11_13.md](reports/p1_2026_11_13.md)); the fit is the 16 GB card. The cosine
  check has no bar: spread unknown.
- Cost: the build is code; the probe is under 10 minutes of GPU.

Interpretation:

- The reader's loss needs something in the crop to pull on. A box with the wrong strokes is
  the common case on kanji (the crossed grid's sheets), and there the loss has a target per
  position from one sample.
- With a box on the page the 0.6 line changes nothing in the loss: the cross-entropy is
  continuous and already large on a box the reader is unsure of. The line matters for which
  box is taken and for the reads. Of the exact reads 7 % sit under it.
- The no-box pages are the ones that need a term of their own. The detector's score says
  "letter something" in the detector's terms, the same eyes that decide whether a crop exists,
  and it is one small backward more. Its risk is a text-like scribble that satisfies the
  detector alone; the reader's loss takes over as soon as a box appears, and the read of
  record is the sampled read.
- If G0d finds the score's gradient pointing away from the glyph's (a negative cosine), the
  fallback is to skip those pages: a row is shared by every seed, so a glyph learned where a
  box exists may bring the blank share down by itself.
- Layout is set in the early steps, so the last step alone may reach strokes and not whether a
  glyph is drawn. The four-step backward is the fallback for the "nothing drawn" branch.

## 2. G1 — sixteen rows

After item 1; its step count, steps of backward, page weight and learning rate come from G0d.

Facts:

- Rows, picked across the crossed grid's `solo` range (out of 16): ひ 11, あ 8, ぬ 7, う 5, さ 4,
  が 2, め 1, を 1; 死 7, 王 4, 太 3, 二 3, 味 2, 楽 1, 人 0, 行 0.
- Frame: `speech bubble` in (proposed: G-D needs a crop, and the plain frame leaves half the
  pages blank).
- Scenes: the 25 of `assets/glyph_scenes.json` (the relaxed rule), split into training scenes
  and read scenes no update sees; the split is fixed in the run's config header, by folder so
  that a read scene's artist is not trained on. A further screen round
  (`glyph_scenes.py screen --round N`, 64 fresh candidates of the low-text folders, about 12
  minutes, about 11 more scenes) widens the pool if the split leaves either side thin.
- Control: a run of the same length with the 16 labels permuted among the rows (proposed).
- Read of record: the glyph grid on the held-out scenes at 4 seeds, `solo`, `on_page`, `extra`
  per glyph with the null of `glyph_g0.py`, against the start rows, paired over glyphs; the 16
  rows' neighbours on the grid, which must not move; steps to the first gain per row. Sheets go
  to the user, whose read decides.
- Bars: spread unknown.
- Cost: unknown until G0d gives the step's seconds.

Interpretation:

- The permuted run separates "the row learned its glyph" from "the rows learned to fill a
  bubble": a gain in `solo` that the permuted run shares is the second.
- を, 人 and 行 start at or near zero, so they show whether the loss can start a glyph and not
  only sharpen one.
