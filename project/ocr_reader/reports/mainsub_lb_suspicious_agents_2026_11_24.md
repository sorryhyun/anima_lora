# mainsub_det: the `linebreak suspicious` sort, top 1,600 rows, fixed by Sonnet agents (2026-11-24)

The rows whose main is one Manga109 `<text>` box with automatic line breaks waiting for review
(`linebreak_auto.py`'s ink-projection cut), largest read distance first, as the review page lists
them. Paths under `output/` are local.

## Facts

### What the box held

- The page box of these rows: the annotation's characters with `linebreak_auto.py`'s breaks (ink
  projection, the label cut in proportion to each column's ink length) and `round_dots`. The meta
  line's `label:` is the annotation without breaks. These mains stay out of train `OCR:` until
  reviewed.
- Claude's read of 10 random rows of the top 100 (crops and labels visible): the proposed breaks
  were right on none; the annotation's characters were right on all 10. Types: a one-column crop
  cut in two (ハ strokes, a balloon tail, a ruby column; 7), ruby beside the text with an empty sub
  (3), real multi-line text cut at the wrong place (3). Top-100 counts: sub empty on 100, `rb` 0 on
  100.
- Three rows the user showed (`Hamlet_122_00`, `Hamlet_082_02`, `OL_Lunch_029_18`): ruby between
  columns filled the gaps, so the projection merged columns (8 printed columns → 4 segments); dot
  runs split by a break stayed unfolded (`…⏎…`, `.⏎....`, `・・・・⏎・・・・・・`), 4 rows of 8,392.
- The user's call: no bulk change to the dot rule (`ellipsis_fold`); agents fix the rows by hand.

### Agent passes

- Rows taken from the built page's data in the page's sort order (`lbs` descending, then id),
  checked against the user's screenshot; given the annotation label (not the proposal), the current
  sub, the in-crop items and hayai's read. Crops upscaled to a short side of 300 px. Rules:
  `RULES.md`, checker `check.py` (output/ocr/linebreak/agent/lb_sus/top600/, local). Dots: per
  printed line, `…` × round(n / 3), at least one. Sub: one line per main line carrying ruby, groups
  separated by a space, then any other legible text.
- Pass 1: top 600 (read distance 101.0 → 0.69), 10 agents × 60. Pass 2: the next 1,000 (0.687 →
  0.273), 10 agents × 100, with one added line: type ruby only as far as the printed kana show.
  All 20 batches passed `check.py`.

| pass | break | one | mismatch | unsure | sub filled | agent main = page box |
|---|--:|--:|--:|--:|--:|--:|
| top 600 | 380 | 203 | 13 | 4 | 200 | 170 |
| next 1,000 | 911 | 35 | 52 | 2 | 339 | 546 |

- Claude's read (crops, statuses visible):
  - pass 1: 24 random `break` / `one` rows, 24 mains broken as printed; one sub doubtful (#145, `とり`
    over an unreadable ruby on 取). All 17 `mismatch` / `unsure` rows: 12 mismatch mains match the
    crop (天上板, ヒマワリ, プライベート, 板 / 。, 「ニッポン」って, ホント / バカだったよな, …).
  - pass 2: 24 random `break` / `one` rows without a note, 24 mains broken as printed (#147, #915
    slanted handwriting, split debatable). 12 chosen mismatches: 10 match the crop (text the crop
    does not show cut, characters as printed, `えーと` to the sub); #103 (main emptied, effect lines
    only) and #706 (`ひろかおずです`) held.
- Held, not applied: pass 1 five rows (4 unsure, `TouyouKidan_078_06` with glyphs below 恩 not
  typed); pass 2 36 rows (29 with a ruby the agent could not read or read in part, 2 unsure, #103,
  #706, #87 cut text not typed, #783 an unreadable mark beside 瞳, #416 dots left as `••••••`).
  Lists: `lb600_held.tsv`, `lb1000_held.tsv` beside the passes (local).
- Applied to `assets/m109_mainsub.tsv` (local) through `apply_review.py`, as `edit` rows with the
  agent's main and sub (`-` = no sub): 595 + 964 rows, file 1,098 → 2,657 rows.
- The HF build of `load_mainsub` (`probes/build_hf_mainsub.py`, local folder `hf_mainsub/`) was
  rebuilt on these labels and the 11-23 build deleted, on the user's word.

## Interpretation

- The read distance ranks segmentation failures, not doubtful breaks: on narrow column strips the
  reader hallucinates (LaTeX, Thai), so the top of the sort is mostly one-column crops and columns
  merged by ruby. Lower in the sort the projection is more often right (546 of 1,000 against 170 of
  600), and almost every row is multi-column.
- Ruby is the common cause of both problems the user saw: it bridges column gaps for the
  projection, and the rows had no sub for it. The 539 filled subs are the first `OCR2:` furigana on
  these crops; the held rows are the ones where an empty or partial sub would train a wrong
  negative.
