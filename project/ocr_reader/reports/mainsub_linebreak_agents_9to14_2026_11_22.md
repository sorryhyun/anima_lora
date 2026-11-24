# main / sub line breaks: Sonnet agents on the unbroken speech mains of 9–14 characters (2026-11-22)

The tier below [`mainsub_linebreak_agents_2026_11_22.md`](mainsub_linebreak_agents_2026_11_22.md),
by the same method. The policy is [`main-sub-approach.md`](../main-sub-approach.md). Paths under
`output/` and `assets/` are local.

## Facts

The user's call, after seeing the count and 12 random crops of the waiting rows: "흠 일단 a만 하자
b, c는 일단 학습에서 빠져있으니 괜찮을듯". a = this tier. b / c = the waiting mains of 15+ (3,764) and
9–14 characters (3,245), which stay out of train `OCR:` and are not filled.

- Input: the 2,247 train `mainsub_det` mains that are speech with no proposal, no line break and
  9–14 characters.
- 12 Sonnet agents, 187–188 rows each, 1.4–5.7 minutes and 141k–176k tokens an agent (1.85 M in
  all). Instructions: `output/ocr/linebreak/agent/speech9/README.md`, the 15+ pass's with three
  additions (count the columns first, about half are one line; text the crop's edge cuts into is
  ignored; marks the label lacks are a note, not a `mismatch`). All 12 passed `check.py`.
- The agents' statuses: break 1,202, one 997, mismatch 42, unsure 6.
- Claude's read, statuses visible:
  - 16 random `break` / `one` rows per batch (seed 0), 192 in all: 187 broken as printed
    (97.4 %; Wilson 95 % interval 94.0–98.9 %).
  - the rows the agents named as uncertain or held, `one` on a crop of a multi-column shape,
    `break` on a crop of a one-line shape, a break at a space or closing mark, and equal labels
    broken differently.
- Claude's corrections are 29 rows in `fixes.tsv`, 5 of them from the random sample: 13 given
  more breaks, 7 fewer, 1 moved, 7 to `mismatch`, 1 to `unsure`. The agents' files are unchanged.

| final status | rows | in training |
|---|--:|---|
| break | 1,199 | the label with its breaks |
| one | 992 | unchanged, one line as printed |
| mismatch | 49 | unchanged, one line (`held.tsv`) |
| unsure | 7 | unchanged, one line (`held.tsv`) |

- 19 `break` / `one` rows print a mark the label lacks (quote marks, arrows, notes, dots, dashes):
  `missing_marks.tsv`, labels not changed.
- `assets/m109_breaks.tsv` went from 1,259 to 2,458 rows, the earlier ones as they were.
- Dataset diff against the state before: 1,199 train `OCR:` targets changed, each in line breaks
  only; rows, order and val unchanged. Train `OCR:` rows with no break 54,879 → 53,680 of
  137,876. On the rebuilt page 1,199 texts changed and none of the rows of exports g, d and b.

## Interpretation

- At the sampled rate the agents broke about 57 of their 2,199 `break` / `one` rows wrongly
  (24–132 over the interval). The targeted reads fixed 16 outside the sample, so roughly 40
  probably remain, mostly slanted handwriting.
- Unlike the 15+ pass the errors go both ways: at this length a single column with a gap also
  gets broken.
