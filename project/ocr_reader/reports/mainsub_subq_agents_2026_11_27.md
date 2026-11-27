# main / sub: duplicate check on the training set, and the suspect subs in train fixed by Sonnet agents (2026-11-27)

Paths under `output/` and the scratchpad are local.

## Duplicates: Facts

- Set: `load_mainsub('train' / 'val')` on the labels of 2026-11-27 morning (`TARGET_NORM = 12`),
  train 157,001 rows, val 12,802.
- Same crop path and prompt: 0 in train, 0 in val. Pixel-identical crops (md5) among the 139,696
  train `OCR:` rows: 37 rows in 12 groups, all gelnote UI chips (`구독`, `공유`, `내 페이지` × 7,
  `时间 / 近代` …). dHash (256 bit) within 10 bits and the same target: 147 pairs, gelnote and
  `swaprelax_v5`.
- Manga109 test (5,117) and the doujin gate (904): no duplicate path or md5; no test book in train.
- Detector boxes on one page, one inside the other (containment > 0.5) with the same main: 784
  pairs, 1,568 crops (train); IoU 0.2 – 0.5. Three pairs read by Claude: the same text cut tight
  and loose.
- Label repetition: `m109_pos` (SFX main) 20,974 rows, 8,095 distinct targets (`はっ` 168, `ザワ`
  148, `ザッ` 142); `m109_neg_speech` 94,224 / 82,643; KO / ZH 5 – 11 % repeats.
- Misses read as a train label at least 10 times as frequent and 3× the target's count
  (`mainsub_1127_0706`): Manga109 test SFX 71 of 881, speech 5 of 265, doujin SFX 22 of 417.
- Doujin SFX misses whose read is a train label: `mainsub_1126_2213` 121 (112 of them a target
  with a heart, read without one), `mainsub_1127_0028` 101 (94), `mainsub_1127_0706` 71 (54).

## Duplicates: Interpretation

- The set holds next to no duplicate crops; the tight / loose pairs act as a scale augmentation.
- The label prior explains under a tenth of the misses; the doujin misses that land on a train
  label are the heart drops `mainsub_plan.md` already records, not repetition.

## Suspect subs: Facts

- Flags on an unreviewed, unagented sub (`subp`): repeats a line or a 2+ character part of the
  main, holds a kanji, is not kana only, is longer than 3 × the main's kanji, or the main has no
  kanji. Over the whole train proposal file 2,402 rows (ruby reads 2,096).
- Set (the user, 2026-11-27: the rows that train, val included): the flagged `OCR2:` sub rows of
  `load_mainsub` at seed 0, 903 (train 824: ruby reads 540, others 284; val 79).
- 16 Sonnet agents × 56 – 57 rows, crops upscaled to a short side of 300 px, given the main, the
  proposed sub and the annotated items. Rules, checker and outputs:
  `output/ocr/mainsub/agent/subq_1127/` (`RULES.md`, `check.py`, `batch_*/out.tsv`,
  `merged.parquet`). All 16 batches passed `check.py`.

| rows | ok | fixed | empty | unsure |
|---|--:|--:|--:|--:|
| train, ruby read | 22 | 373 | 123 | 22 |
| train, other | 218 | 51 | 12 | 3 |
| val | 17 | 53 | 7 | 2 |
| all | 257 | 477 | 142 | 27 |

- Per flag: repeats the main 185 (fixed 120, empty 46, ok 15); main without kanji 224 (ok 172).
  The 15 `ok` repeats are furigana that spell kana also in the main (`楽しみ` → `たの`).
- Main fixes: 4 (`EvaLady_097_00` first column `滅亡よ!` added, `Saisoku_095_18` `パアアア`,
  `HinagikuKenzan_055_23` the slanted aside moved to the sub, `EvaLady_005_01` `バイザー・`).
- Claude's read (crops, verdicts visible) of 6 random `fixed` and 2 random `empty` rows: 5 fixed
  right; `KarappoHighschool_109_07` merged two columns' furigana into one line where the proposal
  had them apart. Both `empty` crops print furigana of a few pixels (`待` ま, `行` い).
- Written to `assets/m109_agent_subs.tsv` (local; pass `subq1127`; file 638 → 1,541 rows, the
  previous file in the scratchpad): ok / fixed → `typed`; unsure → `other`; empty → `empty` when
  the main has no kanji (11), else `other` (131).
- Loader after the change (seed 0): `OCR2:` sub 10,471 → 10,311, sub_neg 6,834 → 6,794; val sub
  1,168 → 1,159.
- `RUBY_RAW_SHARE`'s quarter now hashes the id with a fixed seed (`RUBY_RAW_HASH_SEED = 0`), not the
  run's seed (the user, 2026-11-27: the ruby subs that do not train are left out for good).
  `OCR2:` sub positives at seed 0 and seed 1 are now the same 10,311 rows.

## Suspect subs: Interpretation

- About 9 in 10 flagged ruby reads were wrong or empty (373 + 123 of 540), so the flags pick out
  bad ruby reads well; on the non-ruby rows the flags were mostly false alarms (218 of 284 ok).
- An `empty` verdict on a main with kanji is not reliable at this crop size (2 of 2 read printed
  furigana), so those rows are kept out of `OCR2:` rather than trained as negatives.
- The val sub metric moved by 9 rows and 53 corrected targets; val sub reads before and after
  this pass are not on the same targets.
