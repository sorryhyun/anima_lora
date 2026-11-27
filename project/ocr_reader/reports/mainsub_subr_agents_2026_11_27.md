# main / sub: every unchecked ruby sub in the training quarter, checked and retyped by Sonnet agents (2026-11-27)

Paths under `output/` and the scratchpad are local.

## Facts

- Set: the `OCR2:` sub rows of `load_mainsub('train', seed=0)` from Manga109 with no page review
  and no agent label, after pass `subq1127`: 8,567 rows. 5,138 of them are headed by an unreviewed
  ruby read (`ruby_raw`, the fixed `RUBY_RAW_SHARE` quarter); 3,429 are not (their furigana were
  fixed per `<text>` box by the earlier `furigana_find.py` agents; 3,409 `neg_speech`, 20
  `pos_mixed`). The user asked for the quarter.
- Rules: `subq1127`'s `RULES.md` and `check.py` unchanged; crops upscaled to a short side of 300 px.
- Sample first (the user: 200 random): 4 Sonnet agents × 50, `output/ocr/mainsub/agent/subr_1127/`.
  ok 80, wrong kana 48, furigana missing 44, extra kana 1, spacing / breaks only 25, unsure 2.
- Then the other 4,938 (the user): 99 Sonnet agents × 50 (the last 38), at most 20 at a time,
  `output/ocr/mainsub/agent/subr_all_1127/`. All 103 batches passed `check.py`.
- Whole quarter, 5,138 rows, sorted by comparing the proposal with the agent's sub with
  whitespace removed:

| verdict | rows |
|---|--:|
| ok | 2,394 |
| fixed, other kana | 1,272 |
| fixed, kana added only (furigana missing) | 1,058 |
| fixed, kana removed only | 10 |
| fixed, spaces / line breaks only | 369 |
| unsure | 27 |
| empty | 8 |

- Content wrong (other kana, added, removed): 2,340 of 5,138, 45.5 %.
- Main fixes: 4 (two mains that repeat their lines once more than the crop prints).
- Claude's read of 4 random `fixed` rows from the sample (crops, verdicts visible): all 4 right
  (`すいはん き`, `むり`, `フル/い`, `わたし/あい`).
- Agents split furigana groups differently: some per kanji (`がっ こう`), some per word
  (`おやこどん`); `RULES.md` says one space between groups and does not define a group.
- Written to `assets/m109_agent_subs.tsv` (local; pass `subr1127`; 1,541 → 6,679 rows; the
  previous file in the scratchpad as `m109_agent_subs.before_subr1127.tsv`): ok / fixed → `typed`
  (5,103); unsure and the 8 `empty` (all on a main with kanji) → `other` (35).
- Loader after the change (seed 0): `OCR2:` sub 10,347 → 10,312, sub_neg 6,803 → 6,795; val
  unchanged (sub 1,159).

## Interpretation

- About half the unflagged ruby reads that trained were wrong, the same rate as the sample; the
  `subq1127` flags had caught the obvious ones, not most of them. With these replaced, about 2,340
  of the 10,312 `OCR2:` sub positives carry a different target than before.
- In the missing-furigana rows read here the dropped furigana sat on columns after the first,
  which suggests the ruby reader keeps the first column's only; these rows taught the sub prompt
  to leave furigana out.
- Group spacing inside a sub line now varies by agent; a rule for it, then a fold, would make the
  targets consistent.
- The 3,429 non-ruby rows were not checked; they went through an agent pass per `<text>` box
  before, and a 16-row sheet read as furigana matching the main.
