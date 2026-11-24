# mainsub_det: the rows hayai v2.7 disagrees with, checked by Sonnet agents (tier 1, 2026-11-24)

The user's review of the top of the `hayai disagrees` sort, its error types, and the agent pass
over the next 861 rows. First pass: [`hayai_first_pass_2026_11_24.md`](hayai_first_pass_2026_11_24.md).

## Facts

### Page fixes before the pass

- 406 `mainsub_det` rows had an empty main: a detector box over 0.30 – 0.50 of one speech
  `<text>` box (the detector splits what the annotation joins), under the trim gate of 0.5; hayai
  read text on every one, so they topped the sort at distance 1.0. `m109_mainsub.py`: `TRIM` 0.5 →
  0.3, and a crop with a label of its own is not trimmed (`ARMS_046_20` kept `ズズン`). Rebuilt:
  406 rows changed, all from an empty main; 0 non-empty mains changed. Trim rows 576 → 982.
- `ko_label_review.py`: `hayai_distance` folds hiragana = katakana and small = large kana when the
  main is one kana or the same kana two or three times (`short_kana`; the user's rule): 79 rows
  leave the sort (`へへへ` / `ヘヘヘ`, `ぁ` / `あ`, `オ` / `ォ`).

### The user's review (export of 70 rows, applied)

- `neg_speech` 24 edit / 5 ok / 5 drop; `pos` 9 edit / 22 ok / 2 drop; `pos_mixed` 1 / 1 / 0, one
  row a sub only. 24 of 33 new mains are within 0.2 of hayai's read.
- Types: text the annotation lacks is the largest in the crop (`TRRRR`, `大当り`, `WAITING ROOM`,
  `I wanna⏎Destroy⏎passer-!`, unannotated speech), the old main to the sub; every trim row in the
  export came from the 0.3 – 0.5 band and was edited or dropped (10 of 10); placeholders (`～～～`
  over English, `・・`); `う` → `ゔ` ×8; drops where hayai read only `...` or a long invented
  sentence. hayai wrong (the `ok` rows): hand-drawn SFX (`まる` → `み`, `ダダダ` → `HAHA`), dakuten
  on SFX, repeat counts, single symbols.

### Agent pass

- Set: unreviewed train rows of `mainsub_det` that hayai disagrees with, not only in notation
  (kana size / script, `ー～っ`, punctuation), and either a trim row or distance > 0.4: 861.
- 22 Sonnet agents × 40 (the last 21), the crop upscaled to a short side of 300 px, with the
  main, sub, hayai's read and the partly visible items. Rules: `RULES.md`. Inputs, outputs and the
  merge script: `output/ocr/hayai/agent_tier1/` (local).
- Agent verdicts:

  | cat | edit | ok | drop | unsure |
  |---|---|---|---|---|
  | `neg_speech` | 101 | 57 | 15 | 4 |
  | `neg_speech` trim | 284 | 11 | 12 | 1 |
  | `pos` | 87 | 163 | 88 | 6 |
  | `pos_mixed` | 16 | 13 | 3 | 0 |

- Claude's read (crops, labels visible): six `pos` drops of batch 2 (`はた`, `が~っ!`, `どきっ`,
  `ぐつ`, `ぽか`, `もく⏎もく`) are legible stylized SFX matching the annotation. 18 `pos` glyph
  edits: 15 match the crop (`うう` → `ララ`, `シュウ` → `ミュウ`, `ふん` → `ぶん`, `ガバ` → `ザバ`, …);
  `すく` → `すし`, `カン` → `カニ`, `ぬおおおお` → `ぬああああ` ambiguous. Batch 3's trim edits cut
  the main to what shows, with column breaks and furigana in the sub.
- Merge (`merge_tier1.py`): every `pos` drop (91) and 10 speech drops whose note is only "too
  small / stylized" kept the annotation, not exported; the 3 ambiguous glyph edits not exported;
  2 edits held for the user (furigana in the crop, agent sub empty: `BEMADER_P_047_27`,
  `ToutaMairimasu_037_11`); `ok` rows exported where the page main and sub equal what the agent
  saw and no line break waits (234; 10 skipped); unsure not exported (11).
- Applied to `assets/m109_mainsub.tsv` (local): the user's export 70 rows, then 734 agent rows
  (edit 483, ok 234, drop 17; 172 with a sub). File 294 → 1,098 rows.

## Interpretation

- Where hayai disagrees strongly, the crop usually holds something the annotation does not
  describe: unannotated larger text, or, on the new trim rows, a different part of the page. The
  agents' speech and trim edits follow that.
- The agents drop hand-drawn SFX they find stylized although the annotator's read is right; on
  SFX their drops are not a signal, their glyph edits mostly are.
- The empty subs of the applied edit rows train as `OCR2:` negatives; furigana an agent did not
  type would make a wrong negative. Two such rows were found where the page had ruby reads; rows
  without a ruby read were not checked for it.
