# ocr_reader — main / sub: open questions and pending work

Definitions, labeling and normalization: [`README.md`](README.md). A finished item
leaves this file; the facts behind each item and the reports are in `_archive/ocr_reader/mainsub/`
(local; `mainsub_plan_2026_11_27.md` is this file before the cut).

## Open, for the user

- "Largest" when a dense block meets a few large glyphs: `AosugiruHaru_031_20` unreviewed
  (proposed main `ググッ`, the ranking list as sub), the same case as the reviewed
  `GarakutayaManta_006_34`.
- When a sub counts as complete. Most `OCR2:` positives are unreviewed ruby reads.
- How a main / sub read is scored. The v7 gates read per kind (COO SFX / speech, doujin SFX, K3);
  the val subs are unreviewed ruby reads.
- How the `OCR2:` sub text enters a caption (`anime_tools` reads `OCR:` only).
- The line-break pass on val outside the speech manifest: proposed, not decided.
- Labels holding text outside a `part` crop (ガ / カ, 下だ！！).

## Waiting on a run

No run yet on the current labels and targets: `TARGET_NORM = 13` (drawn-out lines `ー` / `~`,
`core/textnorm.py` fold 7), the `subq1127` / `subr1127` sub fixes, and `assets/anima_mainsub.tsv`
in `load_mainsub`. The local HF build of `load_mainsub` predates them. `anime_tools`'
`ocr.reader.fold_drawn_lines` (norm 13) sits uncommitted in the sibling checkout, pin unchanged.

## Work not done

- Unreviewed subs: the ruby subs outside the training quarter (they do not train), the 3,429
  non-ruby unreviewed subs, a rule for furigana group spacing.
- `hayai disagrees` tiers 2 (distance ≤ 0.4, about 620) and 3 (same length, other glyphs, about
  2,900). `BEMADER_P_047_27` and `ToutaMairimasu_037_11` (furigana untyped) wait for the user.
- Manga109 test SFX rows v7 misses that `sfxpc_1121_0117` reads (156 against 108), and the 11
  non-empty `OCR2:` answers on the 292 crops. Not read.
- Mains out of train `OCR:` until reviewed: 8,412 waiting on a line-break proposal, 162 in
  `assets/m109_break_held.tsv`; 6,833 `mainsub_det` rows left on the `linebreak suspicious` sort.
- `reader_split_mainsub.tsv` is not read by training.
- Label fixes not made: 19 mains of 9–14 characters missing a printed mark
  (`output/ocr/linebreak/agent/speech9/missing_marks.tsv`); the main-in-sub repeat check does not
  fold full-width digits (`HarukaRefrain_002_11`); `AosugiruHaru_028_21` main covers part of the
  SFX.
- Shelved (the user: furigana and line breaks first): finding the crops with text among the
  ~20k SFX-only `OCR2:` negatives.
