# ocr_reader — main / sub: open questions and pending work

Definitions and the dataset: [`main-sub-approach.md`](main-sub-approach.md). A finished item
leaves this file for a report.

## Policy, for the user

- "Largest" when a dense block meets a few large glyphs. `GarakutayaManta_006_34`: the
  proposal's main was `ひ～` (the larger glyphs), the user's review has the English paragraph as
  main and `ひ～` as sub. That main is over the cap, so only the sub trains.
  `AosugiruHaru_031_20` is the same case unreviewed (main `ググッ`, the ranking list as sub).
- When a sub counts as complete. Most `OCR2:` positives are unreviewed ruby reads (about half
  right in a 14-row read, `reports/mainsub_data_2026_11_21.md`).
- `drop` boxes (an SFX cut partway, 2,678 train) and `neg_other` boxes (no annotation inside,
  22,718 train) stay out; under main / sub a cut SFX could be sub.
- How a main / sub read is scored. The v7 gates read per kind (COO SFX / speech, doujin SFX,
  K3); the val subs are unreviewed ruby reads.

## Work

- Pick out the suspect crops among the ~20k SFX-only `OCR2:` negatives (text the annotation does
  not cover, marks that may be glyphs): 2 with text and 8 unsure in a 200-crop agent check
  ([report](reports/mainsub_sfx_agents_2026_11_23.md)). Method not chosen; the next training run
  follows it (the user, 2026-11-23).
- `AosugiruHaru_028_21`: the main `アハハハ` covers only part of the SFX (a `キ…` column beside
  it); no one has read the whole. Trains as proposed.
- 141 speech mains held by the line-break agents (15+ characters: 76 label and crop differ, 9
  with unclear columns; 9–14: 49 and 7; `output/ocr/linebreak/agent/speech{15,9}/held.tsv`,
  local) still train as one line. Out of `OCR:` until reviewed, like the waiting rows: proposed,
  not decided.
- 19 mains of 9–14 characters print a mark their label lacks
  (`output/ocr/linebreak/agent/speech9/missing_marks.tsv`, local): not fixed.
- 3 labels type quote marks as `゛`, which `normalize_target` folds onto the kana: not fixed.
- A main repeated in the sub when the two differ in width (`HarukaRefrain_002_11`: main `８年`,
  sub `8年` / `ひっ`): the proposal's repeat check does not fold full-width digits.
- The 8,412 waiting mains come back row by row as they are reviewed. The agents do not fill
  them for now (the user, 2026-11-22: "b, c는 일단 학습에서 빠져있으니 괜찮을듯").
- Val never had the line-break pass outside the speech manifest. Proposed, not decided.

## Reports

- [`mainsub_data_2026_11_21.md`](reports/mainsub_data_2026_11_21.md): the detector crops, the
  proposals, the line-break and furigana passes, the agent reviews.
- [`mainsub_1121_1806.md`](reports/mainsub_1121_1806.md): the first run (val main 86.4 %, sub
  49.2 %) and its v7 gates.
- [`mainsub_labels_2026_11_22.md`](reports/mainsub_labels_2026_11_22.md): the loader changes
  of 11-22 and the count of missing line breaks.
- [`mainsub_linebreak_agents_2026_11_22.md`](reports/mainsub_linebreak_agents_2026_11_22.md)
  and [`mainsub_linebreak_agents_9to14_2026_11_22.md`](reports/mainsub_linebreak_agents_9to14_2026_11_22.md):
  the agents' breaks on 15+ and 9–14 character speech.
- [`mainsub_1122_1446.md`](reports/mainsub_1122_1446.md): the sync-free training forward.
- [`mainsub_1123_1328.md`](reports/mainsub_1123_1328.md): `OCR2:` negatives from swap_relax and
  reviewed JA rows (main 85.4 %, sub 51.0 %, empty on the 292 JA crops 72.6 %).
- [`mainsub_sfx_agents_2026_11_23.md`](reports/mainsub_sfx_agents_2026_11_23.md): the SFX-only
  check (190 / 2 / 8 of 200) and the agent subs of the 636 SFX crops with speech partly inside.
