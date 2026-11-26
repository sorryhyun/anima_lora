# The glyph stage's scene pool, picked by render, and the glyphs crossed with four scenes (2026-11-25)

After `glyph_g0_2026_11_25.md`: which training captions letter nothing by themselves, whether the
artist folder or the caption predicts it, and each glyph's read on the same four scenes. The
user's calls (11-25): pick scenes from artists whose captions hold little text, or from captions
that had no OCR clause; the selected prompts go to `assets/` (gitignored).

## Facts

Probe `../probes/glyph_scenes.py` (its docstring holds the design). Stack, frames, decode and
reads as G0: `m15v11`, 4 Euler steps, cfg 1, 768², pixel head, seeds 0 – 3, 4 images a forward,
16 pages a read call. Spread between two runs of one setting: unknown.

- `screen`: job `20261125-101807-4a613f`, 1 536 renders, 12.6 min; result
  `../results/20261125-1030-glyph_scenes-screen/` (local; `scenes.json` holds every candidate).
- `cross`: job `20261125-103113-ddecf7`, 1 024 renders, 9.2 min; result
  `../results/20261125-1040-glyph_scenes-cross/` (local).
- `screen --round 2`: job `20261125-104920-2bdd03`, 1 536 renders, 12.4 min; result
  `../results/20261125-1101-glyph_scenes-screen/` (local). 64 fresh candidates of the `low`
  folders.
- Selected prompts: `../assets/glyph_scenes.json` (local), written by `glyph_scenes.py select`
  from both rounds: 25 scenes with both captions and their numbers.

### Screen — 64 candidates

Every candidate is a caption with no text clause and no lettering tag. `low`: 44 from the 23
folders whose captions carry a text clause at most 5 % and a lettering tag at most 12 % of the
time; `other`: 20 from the remaining folders. Of the 3 242 captions 723 carry a text clause.
A scene is clean in a frame when none of its 4 no-clause renders has a read letter.

| group | scenes | clean, plain | clean, bubble | clean in both | letters a no-clause render, plain / bubble |
|---|---|---|---|---|---|
| low | 44 | 38 | 11 | 10 | 0.23 / 5.43 |
| other | 20 | 9 | 4 | 4 | 2.51 / 6.35 |

`solo` of the two probe strings (`stop`, rows off; う, rows on), and the letters beside them:

| scenes | stop, plain | stop, bubble | う, plain | う, bubble | extra, stop / う in a bubble |
|---|---|---|---|---|---|
| low (44) | 0.28 | 0.42 | 0.08 | 0.16 | 3.2 / 3.7 |
| other (20) | 0.14 | 0.35 | 0.03 | 0.18 | 5.9 / 6.9 |
| clean in both (14) | 0.21 | 0.23 | 0.07 | 0.11 | 2.1 / 1.7 |
| not clean (50) | 0.24 | 0.45 | 0.06 | 0.18 | 4.6 / 5.6 |

- Of the 14 clean scenes, 6 never read `stop` or う exactly in either frame; scenes 6 and 12
  read `stop` 4 of 4 in a bubble.
- Claude's read (`sheets/screen_bubble_08.png`, labels visible): the scenes where both strings
  sit alone in a bubble on most seeds are single figures on a white ground (12, 13). Scene 13
  is not clean: one of its no-clause renders letters a line. Scene 12 counts as clean while
  its no-clause seed 0 shows a bubble of small Latin lettering the reader returned nothing for.

### Round 2 and the selection

| round, group | scenes | clean, plain | clean, bubble | clean in both | letters a no-clause render, plain / bubble | stop, plain / bubble | う, plain / bubble |
|---|---|---|---|---|---|---|---|
| 1, low | 44 | 38 | 11 | 10 | 0.23 / 5.43 | 0.28 / 0.42 | 0.08 / 0.16 |
| 2, low | 64 | 52 | 19 | 17 | 0.43 / 5.26 | 0.29 / 0.40 | 0.08 / 0.18 |

The selection rule (the user's call, 11-25, the relaxed one): the plain no-clause renders
letter nothing, the bubble ones on at most 1 of 4 seeds, and the two probe strings read exactly
on 2 of their 8 bubble renders or more.

| candidates | screened | selected |
|---|---|---|
| low, round 1 | 44 | 12 |
| low, round 2 | 64 | 11 |
| other, round 1 | 20 | 2 |
| all | 128 | 25 |

- The 25 come from 13 folders (one folder 5, one 4, five 2, six 1). 10 of them are clean in
  both frames; on 14 the probe strings read exactly on 4 of 8 bubble renders or more.
- The low folders hold 622 text-free captions; 108 are screened.

### Cross — every string on scenes 6, 12, 15, 45 (bubble frame, 16 renders a string)

The four clean scenes with the most exact reads of the probe strings, one a folder.

| class | renders | solo | on_page | solo and nothing else | extra a page | no letters read |
|---|---|---|---|---|---|---|
| hiragana (24) | 384 | 0.27 | 0.34 | 0.24 | 2.1 | 0.19 |
| kanji (32) | 512 | 0.05 | 0.06 | 0.04 | 4.6 | 0.27 |
| EN (8) | 128 | 0.57 | 0.63 | 0.52 | 2.0 | 0.18 |

`solo` out of 16, per string:

- hiragana: ひ 11, く 9, ち 9, あ 8, ね 8, ぬ 7, ぽ 6, う 5, ぱ 5, ぜ 5, ゔ 5, ろ 4, さ 4, と 3,
  が り れ わ る 2, も め を 1, っ ゃ 0.
- kanji: 死 7, 王 4, 太 二 可 3, 味 2, 持 楽 性 他 1; the other 22 are 0.
- EN: open 13, stop 11, help 11, love 9, no 9, home 9, exit 6, yes 5.

`solo` by scene, hiragana / kanji / EN: scene 6 35 of 96, 7 of 128, 19 of 32; scene 12 40, 12,
28; scene 15 13, 5, 15; scene 45 14, 2, 11.

Inside a (string, scene) group of 4 seeds: `r_seq` not constant in 0.67 of the hiragana groups,
0.16 of the kanji groups, 0.75 of the EN groups; mean std of `r_seq` 0.32, 0.07, 0.32.

- Claude's read (`sheets/cross_kanji_00.png`, labels visible): on scene 12 seeds 1 and 2 draw one
  kanji-sized mark in a bubble for nearly every kanji asked, most of them not the asked one
  (行 read as イチ, 生 as 王, 待 as 緋); seed 0 draws the same bubble of small Latin lettering
  whatever is asked.

### The reader's confidence on the crossed grid

Counted from the run's `reads.json`, rows arm, 896 pages (renders
`output/cjk_rl/glyph_scenes/m15v11_px/cross/seed_1008/`, local).

- No box on 193 pages; a box but none at a confidence of 0.6 or over on 195 more.
- Of the 130 boxes that read the asked glyph exactly, 9 are under 0.6; of the 660 that read
  something else, 36 %.
- The user's call (11-25): a box the reader reads under about 0.6 counts as nothing drawn, and
  a page where AnimeText finds no box takes the detector's own score as its signal to draw.

### Cost

Sampling 0.40 s an image at 4 a forward (0.39 s at 8 in G0), the four steps 1.2 s a forward;
100 % GPU use, 5.2 GB, 299 W of 300 W during the render. Text for 384 captions 7.7 s. Reads 512
pages in 0.5 min.

## Interpretation

- The artist folder predicts the plain frame (38 of 44 scenes clean against 9 of 20); the
  caption having no OCR clause does not by itself. Neither predicts the bubble frame: three
  scenes in four fill the bubble on their own.
- Being clean and taking the string pull apart. The scenes that letter nothing without a clause
  also letter the asked string less, and the scenes where a string lands best fill a bubble by
  themselves on some seeds. The relaxed rule keeps both: a no-clause bubble may letter on one
  seed in four, and the page term is taken on the plain anchor.
- The two rounds agree on every rate of the low folders (the probe strings' reads within 0.02),
  so the screen's rates carry to fresh captions of those folders; about one candidate in five
  passes the rule.
- On the same scenes the order is EN 0.57, hiragana 0.27, kanji 0.05: the rows are under the EN
  level by a factor of two for kana and ten for kanji once the scene is held fixed.
- The kanji sheets show a glyph-sized mark in the right place with the wrong strokes. That is
  the case G-D's per-position cross-entropy is for, and the box exists on about three pages in
  four.
- G1's sixteen rows across the range: ひ あ ぬ う さ が め を and 死 王 太 二 味 楽 人 行 (the last
  two never read). The small kana have no size reference alone and stay out.
