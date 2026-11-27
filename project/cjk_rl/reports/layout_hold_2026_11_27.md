# Layout hold — what fixes a page's bubble: the (scene, seed), the string, the steps before σ 0.75 (2026-11-27)

One forward-only probe, `lhold_1127_1851` (`probes/layout_hold.py`), on kmix's rows, read with
the tuned reader as it stood before the 11-27 retrain. Opened on the user's word (11-27), on a
layout scaffold for the sentence loop (a list of screened (scene, seed), or a LoRA / cached state
on the steps before σ 0.75): "어차피 검증되면 새로운 방법에 리더만 올리면 되니까 probe 해볼수있음
해보자". No training.

## Facts

- 768 pages: `wread_1126_1207`'s 12 strings of 2 glyphs (short) and 12 of 5 – 8 (long) on read
  scenes 6, 12, 45, 15. Own pages at seeds 0 – 3 (384); swap pages (the first k steps on string
  A, the rest on B, A and B of different lengths, both ways) at seeds 0, 1 and k = 1, 2 (384).
  A (scene, seed) is a cell below; there are 16.
- Two meters of a page's layout. The picked box (`gd.pick`): its centre, and its way (tall /
  wide / square at 1.15, as `orient_read.py`). The bubble: the bright region under the picked
  box, compared between pages by IoU; it is new in this probe.
- The bubble's misses: of the 384 own pages 40 have no bright region under the box and 3 run
  over half the page. Claude viewed 4 of the 24 row sheets (16 pages): on 1 page the region runs
  into a white background under the cap, on 2 a shaded bubble gives a fragment, and scene 45's
  pages letter on a dark bubble (no bright region). The picked box's columns do not use it.
- Spread unknown (no twin). Results (local):
  `results/20261127-1900-layout_hold-lhold_1127_1851/` (`tables.json` from `--tabulate`), pages
  and row sheets under `output/cjk_rl/layout_hold/lhold_1127_1851/`.

Own pages of different strings, in pairs:

| level | lengths | centres within 0.1 of the side (pairs) | bubble IoU ≥ 0.5 (pairs) | bubble IoU, median | same way |
|---|---|---|---|---|---|
| same scene and seed | short – short | 0.825 (1 034) | 0.602 (931) | 0.634 | 0.443 |
| same scene and seed | long – long | 0.830 (1 003) | 0.621 (768) | 0.633 | 0.546 |
| same scene and seed | short – long | 0.802 (2 222) | 0.581 (1 836) | 0.579 | 0.463 |
| same scene, another seed | short – short | 0.198 (3 102) | 0.029 (2 690) | 0.0 | 0.345 |
| same scene, another seed | long – long | 0.181 (3 005) | 0.015 (2 256) | 0.0 | 0.347 |
| same scene, another seed | short – long | 0.195 (6 663) | 0.022 (5 361) | 0.0 | 0.345 |
| another scene | short – short | 0.298 (12 408) | 0.062 (10 573) | 0.0 | 0.367 |
| another scene | long – long | 0.250 (12 018) | 0.040 (8 401) | 0.0 | 0.365 |
| another scene | short – long | 0.274 (26 645) | 0.050 (20 611) | 0.0 | 0.368 |

Own pages by length (192 each):

| length | boxes at 0.6 | tall / wide / square / no box | bubble's share of the page, median | bubble h / w, median | positions hit | whole string |
|---|---|---|---|---|---|---|
| short | 1.00 | 55 / 93 / 42 / 2 | 0.017 | 0.848 | 0.784 | 113 |
| long | 0.97 | 75 / 83 / 29 / 5 | 0.019 | 0.857 | 0.348 | 0 |

- The long strings' positions hit by the picked box's way: tall 0.343 (75 pages), wide 0.365
  (83), square 0.377 (29).
- A cell's 12 long strings: tall on 0 to 10 of the 12 (0, 0, 1, 1, 2, 2, 3, 4, 4, 7, 8, 8, 8, 8,
  9, 10 over the 16 cells); positions hit 0.122 to 0.554 a cell (74 positions each). At the
  overall tall share 0.391 a cell of 12 draws 8 or more with probability 0.050 (0.8 of 16 cells;
  6 seen) and 1 or fewer with 0.023 (0.4; 4 seen). Split over the strings in two halves of 6,
  the cells' two values correlate at 0.81 for positions hit and 0.83 for the tall share.

Swap pages against A's and B's own pages of the same cell (96 pages a row; the IoU over the 74 –
78 where all three pages have a bubble):

| A → B | k | IoU to A's / to B's, median | A's to B's own | closer to A's | way as A's / as B's | B's positions, swap / own | B whole, swap / own |
|---|---|---|---|---|---|---|---|
| short → long | 1 | 0.704 / 0.913 | 0.650 | 0.378 | 0.427 / 0.740 | 0.355 / 0.375 | 0 / 0 |
| short → long | 2 | 0.978 / 0.639 | 0.665 | 0.811 | 0.656 / 0.490 | 0.272 / 0.375 | 0 / 0 |
| long → short | 1 | 0.657 / 0.967 | 0.652 | 0.160 | 0.521 / 0.760 | 0.781 / 0.812 | 55 / 58 |
| long → short | 2 | 0.966 / 0.679 | 0.632 | 0.769 | 0.750 / 0.583 | 0.688 / 0.812 | 40 / 58 |

- `tswap_1126_1209`, A and B of one length, B's positions at k 0 / 2: 0.82 / 0.72 at 2 glyphs,
  0.36 / 0.35 at 5 – 8 ([sent_prep](sent_prep_2026_11_26.md)).
- Claude's read, arm labels visible, 4 row sheets (A's own, k = 1, k = 2, B's own): the k = 2
  page keeps A's bubble in place and outline with B's glyphs in it, on 2 of the 4 in fainter
  strokes than either own page; the k = 1 page looks like B's own. Six long strings of cell
  (12, 3): one bubble at the same place on all six, tall and narrow on five; four pages letter
  3 – 4 kana in a column where 5 – 7 were asked, two letter Latin-like text.

## Interpretation

- A (scene, seed) carries where the bubble is and, on about 0.6 of pairs, its outline, whatever
  the string; the scene alone carries neither (another seed of the same scene sits at another
  scene's level). A list of screened cells is then a layout bank that changes nothing in the
  render: the string stays on all four steps.
- A cell is also a stable property for the two things a screen would be on: whether long
  strings stand tall there, and how much of them reads there (the split halves at 0.83 and
  0.81, over other strings). The counts above are on the strings the cells would be picked on;
  a screen needs a second set of strings for its read.
- The bubble does not grow with the string: its share and shape are the same at 2 glyphs and at
  5 – 8, and in the viewed tall cell the long strings lose glyphs to a small bubble. A stable
  layout is not a layout with room, so the screen that matters is on the long strings' read a
  cell, not on the way alone. The bubble's medians rest on the pages the meter caught.
- The bubble is set by σ 0.75, and what the string adds to it comes in the step σ 0.9 → 0.75:
  the text of the last two steps leaves A's bubble as it is, the text from step 1 on gives B's.
  A scaffold on the first two steps (a cached state, or a LoRA on those steps) would therefore
  hold its layout through two bare steps. Its cost shows on the long strings: 0.10 of 0.375
  positions under a bubble made for 2 glyphs, where a same-length swap cost 0.01. A cached
  state would have to be made a length; the cell list does not pay this.
- The two steps the sentence loop differentiates do not move the bubble or the way, so the
  row gradient as it stands cannot reach either. For vertical lettering, the step that would
  have to be in the backward is σ 0.9 → 0.75 (`back_steps = 3`).
- Not measured: whether rows trained on screened cells read better on pages nobody screened.
  That is the micro arm, and it waits on the reader.

# A cell's room, and SAM3 on the bubble

Two follow-ups on the same pages, on the user's questions (11-27): "sam3로 안잡히나?", "n글자별로
맞는 프롬프트나 레이아웃이 있다?".

## Facts

- Counted from the run's `records.json` on its own pages; no script kept. Over the 16 cells:
  the long strings' positions hit against the short strings' correlate at 0.72; against the
  cell's median picked-box area on its long pages at 0.52; against its median bubble share at
  0.63.
- A cell's median picked-box area (share of the page) on its long and on its short pages
  differs by under 0.005 in 14 of the 16 cells (for instance 0.004 / 0.004, 0.008 / 0.008,
  0.010 / 0.008); the other two are 0.066 / 0.039 and 0.041 / 0.033.
- The long strings' positions hit by quartile of the page's picked-box area (187 pages with a
  box): 0.266, 0.360, 0.401, 0.438. Glyphs drawn over asked, a cell: 0.34 to 1.30, under 0.67
  in 10 of the 16.
- By scene, the long strings' positions hit a cell: scene 6 0.31 – 0.51, scene 12 0.35 – 0.55,
  scene 15 0.15 – 0.35, scene 45 0.13 – 0.40.
- SAM3 (`anime_tools.masking.cli.probe_sam_masks`, the prompt `speech bubble`, threshold 0.3)
  on 12 pages, the four pages of three row sheets: a bubble found on 12 of 12 at scores
  0.90 – 0.95, with its box on the bubble in Claude's view of the overlays. The 12 hold the
  three kinds the bright region missed: a white background (1 page), a shaded bubble (2), an
  outlined bubble on a dark ground (3, where the mask is the outline and fills 0.11 – 0.13 of
  its box). One page has two bubbles and both are found. Output not kept.

## Interpretation

- The cells do not sort by length: a cell that reads long strings reads short ones too, and
  the lettered area is the cell's, the same at 2 glyphs as at 5 – 8. What differs between
  cells is room, and a long string reads more where there is more of it. A list of cells
  would then carry each cell's room, with a string drawn onto cells that hold its length,
  not a layout a length.
- SAM3's box is the meter for that room: it does not depend on how many glyphs came out, as
  the picked box does, and it counts the bubbles, which a screen for a bubble a string needs.
- The prompt's part is not measured here: four scenes, and the seeds of one scene spread as
  far as the scenes do. A tag for a large or tall bubble is an untried lever.
