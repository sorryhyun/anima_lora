# hayai-ocr v2.7 Supernova over every Manga109 detector box, and its place on the review page (2026-11-24)

A first-pass read for the boxes with no label and a second read beside the main / sub
proposals. The policy: [`main-sub-approach.md`](../main-sub-approach.md); the page: README § Rules.

## Facts

- The user (2026-11-24): "아직 레이블 안된거 hayai 2.7 supernova로 레이블링 초벌 쓰고 싶어"; the
  rows picked: the boxes with no annotation, the mains waiting on a line break, every unreviewed
  main / sub row. On the licence: README § Labels.
- `probes/hayai_read.py`, `JustANormalTinkerer/hayai-ocr-v2.7-supernova` at `90fc6148`,
  `max_patches="auto"`, batch 64: all 164,090 boxes of `manifest_sfxp_det.parquet` (train
  150,081, val 14,009) in 302 s, 544 boxes/s. Job `20261124-090142-1b699b`, rc 0. Output
  `output/ocr/hayai/hayai27_det.jsonl` (local).
- A box of `manifest_sfxp_det_all` is read from its crop on disk (134,613); the other 29,477 are
  cut from the page with the training margin.
- No read is empty or missing, on any cat: `neg_other` 22,718 train / 1,955 val, `drop` 2,678 /
  168 included.
- No read holds a line break (0 of 164,090).
- On the page (`ko_label_review.py --hayai`, built 09:07):
  - `mainsub_det` 122,853 rows, each with hayai's read. 12,947 differ from the main
    (whitespace-blind, `exact_key`'s folds, a read equal to main + sub counted as a match);
    2,923 of them are among the 8,412 mains waiting on a line break. By distance to the main
    (1 − similarity): under 0.2 8,397, 0.2 – 0.4 2,804, 0.4 – 0.6 902, 0.6 – 0.8 265, over 0.8 579.
  - `mainsub_nolabel` 25,396 rows (`neg_other` and `drop`, train), the main box holding hayai's
    read; 91 over the training cap are not listed. Their crops were cut to
    `derived/crops_sfxp/det_nolabel/train/` (local).
- Six rows read at distance 0.333: `え──!!` read `えー!!`, `そお？` read `そぉ?`, `ワー⏎ワー` read
  `ワー` (twice), `わ――――っ!!!` read `わーっ!!!`, `ブー` read `ブ`.
- `apply_ko_review.py` takes `mainsub_nolabel` rows into `assets/m109_mainsub.tsv`.
  `crop_dataset.load_mainsub` reads `manifest_sfxp_det_all` only, so a reviewed no-label row does
  not train.

## Interpretation

- hayai answers something on every box, so on the no-label rows an empty main has to come from
  the reviewer: a box of drawing with no text still shows a read.
- With no line breaks in its reads, hayai says nothing about the mains waiting on a break beyond
  whether their characters match.
- The low-distance disagreements look like spelling of lengthening marks and small kana; the
  sort puts the largest distances first for that reason. The share of real label errors among
  the 12,947 is not known.
