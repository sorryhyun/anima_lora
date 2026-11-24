# main / sub line breaks: Sonnet agents on the unbroken speech mains of 15+ characters (2026-11-22)

The policy is [`main-sub-approach.md`](../main-sub-approach.md); the count these rows come from is
[`mainsub_labels_2026_11_22.md`](mainsub_labels_2026_11_22.md). Paths under `output/` and `assets/`
are local.

## Facts

The user's call: "대사 15자+ 인거 sonnet 에이전트 돌려서 줄바꿈 채워달라고 하자".

- Input: the 1,498 train `mainsub_det` mains that are speech with no proposal, no line break and
  15+ characters.
- 8 Sonnet agents, 187–188 rows each, read every crop from sheets of 8 and placed breaks only
  (`output/ocr/linebreak/agent/speech15/README.md`). Statuses: `break`, `one`, `mismatch`
  (breaks alone cannot make the label right), `unsure`. All 8 outputs passed `check.py` (the
  characters equal the label's, whitespace aside). 4–9 minutes an agent.
- Claude's read, labels visible: 16 random `break` / `one` rows per batch (seed 0), 128 in
  all, 123 broken as printed (96.1 %; Wilson 95 % interval 91.2–98.3 %). The 5 wrong rows each
  join two adjacent columns or rows into one line.
- Rows the agents reported as uncertain were read too. With the sample's, Claude's corrections
  are 25 rows in `fixes.tsv` (10 re-broken, 14 to `mismatch`, 1 to `unsure`); the agents' files
  are unchanged.
- Batch 2 flagged no mismatch; 5 of the rows Claude read there were mismatches.
- A column-count check by ink projection was tried for the merged-column error and not used:
  the first setting found fewer columns than the agents on most rows, the second flagged 0 of
  6 known merges.

| final status | rows | in training |
|---|--:|---|
| break | 1,259 | the label with its breaks |
| one | 154 | unchanged, one line as printed |
| mismatch | 76 | unchanged, one line (`held.tsv`) |
| unsure | 9 | unchanged, one line (`held.tsv`) |

- The breaks are in `assets/m109_breaks.tsv` (source, id, main), placed into the label's own
  characters. The loader and the page read it (`with_breaks`); a reviewed row overrides it, and
  a row whose proposed characters change stops taking it.
- Dataset diff against the state before: 1,259 train `OCR:` targets changed, each in line
  breaks only; rows, order and val unchanged. Train `OCR:` rows with no break 56,138 → 54,879
  of 137,876.
- `Arisa_081_04` was held at the diff: its label types quote marks as `゛`, which
  `normalize_target` folds onto the kana (`言ってた゛やくそく゛` trains as `言ってだやくそぐ`).
  A search of the training set for the same pattern found two more: `EvaLady_034_15` (`゛まて゛`
  trains as `゙まで`) and `EvaLady_056_21` (`関係なく゛故障゛` as `関係なぐ故障゙`). Not changed.

## Interpretation

- At the sampled rate about 50 of the 1,413 `break` / `one` rows still join two lines (22–113
  over the interval), mostly among slanted handwriting.
- The mismatches are label errors of the annotation or text the crop cuts. They are likely
  undercounted, since agents differed in how often they flagged them.
