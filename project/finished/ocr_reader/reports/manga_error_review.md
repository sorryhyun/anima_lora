# Manga109 review: the two-prompt reader's errors, detector misses, missed polygons restored (2026-11-20)

The reader is `sfxpc_1119_0020` (the 2-epoch `sfxpc_1118_0412` recipe; `sfxpc_1119_2348` has no
COO eval yet), against v6 on the same 4 px COO test crops. The proposal is
[`docs/proposal/ocr_sfx_prefix.md`](../../../../docs/proposal/ocr_sfx_prefix.md).

## 1. COO test errors by kind

### Facts

Rows: `output/ocr/eval/sfxpc_1119_0020_cap4_{sfx,ocr}_test.jsonl`, `v6_cap4_ocr_test.jsonl` (local).
Kinds come from a string rule over label and read, first match wins.

COO SFX under `SFX:`, n 2,558: 0020 79.2 %, v6 85.2 %. Of the paired rows, v6 alone is right on 233 and 0020 alone on 80.

| kind | 0020 | v6 |
|---|---|---|
| misread, sim ≥ 0.5 / < 0.5 | 139 / 112 | 116 / 69 |
| dakuten / small kana only (`キィィィィ` → `キイイイイ`) | 94 | 89 |
| empty answer | 68 | — |
| dots / long vowel / heart only | 42 | 41 |
| cut short | 33 | 40 |
| trailing extra (`バシャ` → `バシャバ`) | 28 | 11 |
| runaway | 14 | 11 |

- Of the 68 empty `SFX:` answers, `OCR:` reads 56 exactly (`うーん`, `よいしょっと`, `ナハハ`).
  `SFX:` with an `OCR:` fallback on empty scores 81.4 %.
- COO speech under `OCR:`, n 2,559: 0020 88.8 %, v6 88.4 %. When runs of `ー`, `~` and `っ` are folded to one character (and dot runs to `…`), the scores are
  94.6 % and 94.1 %. The GT has a doubled `ーー` / `～～` in 134 rows.

### Interpretation

- The 56 recovered empties lose only their kind. The pipeline reads every box under both
  prompts, so the text survives as a speech line.
- The speech gap to the fold is a labelling convention (Manga109 types a long dash as `ーー`), and both
  readers share it. A scoring-key fold would move both by about 6 points.
- The SFX gap to v6 is mostly misreads, which fits the SFX share of the gradient
  ([`ocr_additionals.md`](../../../../docs/proposal/ocr_additionals.md) §2).

## 2. Detector misses on the COO test pages

### Facts

The 200 test pages of `output/ocr/eval/det_animetext.jsonl` / `det_ppocr_manga.jsonl` (local)
hold 990 SFX. AnimeText leaves 178 of them (18 %) below 0.5 bbox coverage.

| AnimeText-missed SFX | n |
|---|---|
| PP-OCR misses too | 129 |
| one PP-OCR box covers it | 43 |
| PP-OCR fragments cover it | 6 |

Readers on the same pages: covered 0020 84.1 %, v6 89.0 % (n 812); missed 68.8 %, 74.5 % (n 157).

### Interpretation

A PP-OCR union would add up to 43 / 990 SFX. PP-OCR returns 47 boxes a page, and whether the
reader answers `""` on a box with no text is unmeasured.

## 3. Missed train polygons by sub-type, and the restore

### Facts

The 9,353 train COO polygons typed `missed` (no detector box covers 20 % of the polygon; `sfxp_types_train.parquet`) are
sub-typed by `build_sfx_prefix.missed_sub`, first rule that fits. The test column applies the
same rules to the 200 det pages (bbox coverage; `covered` = ≥ 0.8).

| sub-type | rule | train | test n | test 0020 / v6 |
|---|---|---|---|---|
| a_punct | marks only (`!`, `?`) | 334 | 2 | — |
| b_page_edge | touches the page edge | 1,094 | 26 | 26.9 / 46.2 % |
| c_huge | long side ≥ 300 px | 1,120 | 24 | 66.7 / 70.8 % |
| d_joined | joined polygon | 128 | 5 | — |
| e_dark_bg | 4 px ring mean < 150 grey | 1,424 | 13 | 61.5 / 92.3 % |
| f_rotated | ≥ 25° off axis | 1,123 | 25 | 72.0 / 68.0 % |
| g_tiny | long side < 40 px | 779 | 10 | 80.0 / 90.0 % |
| h_one_char | one character | 787 | 8 | 75.0 / 62.5 % |
| i_plain | none of the above | 2,564 | 42 | 92.9 / 90.5 % |
| (covered) | | | 795 | 84.7 / 89.2 % |

- Of the pairs where only one reader is right (v6 / 0020), 61 / 25 are on covered polygons and 0 / 1 on i_plain.
- Contact sheets (`~/missed_sheets/`, local; red = COO polygon, blue / green = detector boxes)
  show i_plain as drawn SFX (outline letters, brush strokes in speed lines), e_dark_bg as white
  on black or tone, g_tiny as small hand-lettered background SFX.
- The user's decision: restore i_plain and a_punct whole, and g_tiny and e_dark_bg without the
  crops another COO SFX reaches (`o_sfx ≥ 0.1`, the largest share of another polygon's bbox inside
  the crop's bbox). The user read the i_plain / a_punct polygon crops as clean enough.
- `build_sfx_prefix.py --missed` → `derived/manifest_sfxp_missed.parquet` (local): every missed
  polygon with `sub` and `o_sfx`, crops cut as the gate crops (`pilot.deskew_crop`, capped margin)
  under `derived/crops_sfxp/missed/train/`: i_plain 2,564, e_dark_bg 1,163, g_tiny 572,
  a_punct 334 (4,633; none lost). `load_sfxp(missed_manifest=…)` /
  `finetune_vl16_lora.py --sfxp_missed sfxp_missed` adds them as `SFX:` positives, train only;
  `OCR:` negatives unchanged. No run has used them yet.

### Interpretation

- The restore adds SFX signal on drawn and inverted lettering, which v7 never trained on and v6 did.
  The test sample does not show i_plain as a gap. e_dark_bg is where v6 leads, on 13 polygons.
- The restored rows raise `SFX:` positives from 21,332 to about 26k, so the `SFX:` negatives rise with them
  (`neg_share`).

## 4. Training target cap

### Facts

- `crop_dataset.MAX_TARGET_CHARS` was 96, and longer targets were sliced. 53 KO labels
  (swaprelax_v5 afterword / colophon paragraphs, 2–14 lines) and 14 Manga109 train speech labels
  were longer. 2348 reads 6 of the 53 to exactly 96 characters. `660538_0027_ko_2` (265 characters) stops at 193,
  at the reader's 200-token decode cap (`anime_tools.ocr.reader.MAX_NEW_TOKENS`).
- AnimeText boxes 52 of the 53 paragraphs as one block, with no smaller box inside
  (`post_image_dataset/i2i/swap_text_relax/det.json`, local).
- Counts over a cap (KO / Manga109 train speech): 120 → 26 / 4, 150 → 14 / 3, 200 → 10 / 2.
- The user's decision: cap 150, longer rows dropped instead of cut (`cap_targets`). `load_sfxp`
  train drops 17 rows (109,356 → 109,339); val unchanged (5,328). The eval decode cap
  (`eval_manga109.MAX_NEW_TOKENS` 96) is unchanged.

## State

The Manga109 side of the reader has been reviewed through. Next: the Korean label set (the review page loop,
`probes/ko_label_review.py` → `apply_ko_review.py`).
