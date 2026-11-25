# hayai v2.7 Supernova beside the tuned reader on the glyph stage's pages (2026-11-25)

Covers run `hayai_1125_1234` (`../configs/hayai_1125_1234.toml`, `../probes/reader_pair.py`,
daemon job `20261125-123445-851ddb`): pages already on disk read again by AnimeText, then by
the tuned PaddleOCR-VL-1.6 reader and by hayai-ocr v2.7 Supernova on the same crops. Result:
`../results/20261125-1234-reader_pair-hayai_1125_1234/` (local: `result.json`, `reads.json` a
page, `sheets/` with `key.json` and `claude_read.json`, the blind zip sent to the user).

## Facts

### The run

- 1 536 pages, 1 427 boxes, 2.2 min; peak allocated 6.19 GiB with the detector and both readers
  resident.
- hayai's read (`model.ocr`, auto patches, greedy): 505 crops in 1.93 s, 922 in 3.01 s. The
  detector and the tuned reader together: 21.4 s for 512 pages, 44.8 s for 1 024.
- hayai's teacher-forced forward (the path a loss on it would use) puts its first argmax on the
  first token of its own read on 1 415 of 1 427 boxes.
- The pages are WebP; the records they come from were read in process.

### g1micro's read pages under both readers

`g1micro_1125_1148`'s 512 pages (16 glyphs × scenes 6, 12, 45, 15 × seeds 0 – 3), start rows →
trained rows. The trained rows took the tuned reader's loss; hayai was not in that run.

| | tuned reader | hayai v2.7 |
|---|---|---|
| solo, of 256 | 58 → 132 | 60 → 125 |
| on_page, of 256 | 78 → 144 | 75 → 135 |
| extra letters a page | 3.27 → 2.18 | 2.35 → 2.05 |
| glyphs up / down (sign test) | 13 / 1 (p 0.002) | 12 / 2 (p 0.013) |
| cross-entropy of the asked glyph on the reader's picked crop | 3.99 → 2.31 | 3.87 → 2.45 |

| glyph | tuned, solo / 16 | hayai, solo / 16 |
|---|---|---|
| ひ | 8 → 11 | 8 → 10 |
| あ | 8 → 15 | 8 → 15 |
| ぬ | 9 → 9 | 10 → 9 |
| う | 4 → 14 | 6 → 13 |
| さ | 3 → 9 | 1 → 8 |
| が | 2 → 7 | 2 → 7 |
| め | 3 → 13 | 3 → 14 |
| を | 2 → 0 | 2 → 2 |
| 死 | 8 → 9 | 7 → 8 |
| 王 | 4 → 10 | 5 → 10 |
| 太 | 2 → 12 | 2 → 9 |
| 二 | 3 → 4 | 0 → 1 |
| 味 | 2 → 11 | 4 → 12 |
| 楽 | 0 → 1 | 1 → 0 |
| 人 | 0 → 7 | 1 → 7 |
| 行 | 0 → 0 | 0 → 0 |

- Pages by the two readers' solo, start / trained: both 51 / 119, the tuned reader only 7 / 13,
  hayai only 9 / 6, neither 189 / 118.
- The two cross-entropies of the asked glyph over the 505 boxes: Spearman 0.893.
- Bars: spread unknown. Two reads of one picture by the tuned reader put 26 of 128 pages in
  another case (`gd_g0d_2026_11_25.md`).

### The crossed grid

`glyph_scenes.py cross`'s pages, start rows (`seed_1008`): 56 JA glyphs × 16 renders, and the 8
EN words × 16 on the base without rows.

| | pages | no box | tuned: solo / on_page / extra a page | hayai: solo / on_page / extra a page |
|---|---|---|---|---|
| JA | 896 | 193 | 128 / 160 / 3.57 | 140 / 176 / 2.74 |
| EN | 128 | 20 | 73 / 80 / 1.98 | 70 / 74 / 1.05 |

- JA pages by the two readers' solo: both 109, the tuned reader only 19, hayai only 31, neither
  737. EN: 66, 7, 4, 51.
- Per glyph the two counts differ by at most 3 of 16, except `help` (tuned 11, hayai 6).

### Boxes

| | g1micro (505 boxes) | crossed grid (922 boxes) |
|---|---|---|
| the two reads are the same text | 257 | 365 |
| both read the asked string | 170 | 178 |
| the tuned reader only | 21 | 26 |
| hayai only | 15 | 36 |
| neither | 299 | 682 |
| empty read, tuned / hayai | 14 / 35 | 28 / 99 |

Confidence under 0.6 (the plan's "nothing drawn" line; hayai's is the mean probability of its
own read's tokens under the teacher-forced forward), both sets:

| | tuned reader | hayai |
|---|---|---|
| boxes that read the asked string | 24 of 395 | 16 of 399 |
| boxes that read something else | 355 of 990 | 285 of 894 |

### Claude's read of the crops the readers differ on

98 crops (47 the tuned reader only, 51 hayai only), numbered and shuffled, the asked string
under each, no reads and no case shown; judged before `key.json` was opened
(`sheets/claude_read.json`). Y: the crop is the asked string alone. N: it is not. The rest
undecided.

| | Y | N | undecided |
|---|---|---|---|
| the tuned reader only (47) | 14 | 19 | 14 |
| hayai only (51) | 20 | 9 | 22 |

- By set, the reader the judgment agrees with, tuned / hayai / undecided: g1micro 13 / 13 / 10,
  crossed JA 10 / 17 / 24, crossed EN 0 / 9 / 2.
- Of the 14 Y on the tuned reader's side, 8 are 二, which hayai reads as katakana ニ; 4 more are
  a dropped mark (ゔ → う, が → か, 太 → 大 twice).
- Of the 19 N on the tuned reader's side: 5 are `help` read on a bubble lettered `hep` (hayai:
  `Hep`, `hep`, `hep...`, `hepり♡`), 1 is `open` on `Opeen`, 3 are 死 on a glyph with a missing
  stroke (hayai: 兎, 兀) at confidence 0.85 – 0.96, 2 are さ on a き-shaped glyph at 0.99, 1 is
  太 at 0.98 on a mark that is not a kanji.
- Of the 9 N on hayai's side: 5 are crops too small to judge at the sheet's size (く, う twice,
  早, あ), on which hayai read the asked glyph without being given it.
- Of the 20 Y on hayai's side: 5 are the glyph beside a heart, star or small mark the tuned
  reader kept (`う♡`, `NO♡`, `げ★`, `ちぃ`); the others are a neighbouring glyph read by the
  tuned reader (れ → わ twice, 人 → ト twice, ぜ → ザ / ビ, ぬ → `Va` / `∆a`, 王 → エ).

### The user's read

Not given yet. The numbered sheets went to the user as a zip (keys left out).

## Interpretation

- hayai counts g1micro's gain as the tuned reader does: 60 → 125 against 58 → 132, with twelve
  glyphs up. The rows were trained against the tuned reader alone, so a reader that took no part
  in the training seeing the same rise says the rows learned glyphs and not the tuned reader's
  habits. This stands in for part of what the permuted-label control was meant to show; it does
  not replace it.
- On single glyphs neither reader is clearly the better judge. Where they differ, Claude's
  read sides with hayai somewhat more often on the crossed grid and evenly on g1micro, with a
  third of the crops undecided. The config header's third answer applies: a split.
- The two readers fail differently. The tuned reader completes what it expects (`help` on
  `hep`, 死 and さ on malformed glyphs, at high confidence) and keeps decorations, which costs
  solo. hayai reads what is drawn, drops decorations, reads small text, and has its own blind
  spots: 二 as ニ, dakuten on ゔ, and more empty reads.
- The completion habit matters more for the sentence arms than for the glyph stage. A reward
  that reads `help` on `hep` pays for a misspelled string, which is the opposite of the line's
  goal (the asked string, in order, at its length). Five of five `hep` bubbles and one `Opeen`
  are few and all EN; whether the tuned reader does the same on JA strings is open.
- As a loss, hayai would rank crops much as the tuned reader's loss does (Spearman 0.89), so a
  G1 arm on hayai's loss is expected to learn similar rows; the cases where it would differ
  are the ones above (it would not reward the malformed 死, and it would fight 二). Its read
  is about ten times faster, but the read (detector included) and the reader's loss are 0.7 s
  of a 2.9 s group, so a row step would cost under a quarter less even with both at zero.

## Next

Open, for the user: the glyph stage stays on the tuned reader's loss (plan.md item 3 as
written); hayai can be added to G1's read as a second column at about 2 s a read of 512
pages. A G1 micro arm on hayai's loss (about 25 min, plus the tensor path for its patches) and
a string-level check of the completion habit on JA (the ruler's renders, forward only) are
proposed, neither queued.
