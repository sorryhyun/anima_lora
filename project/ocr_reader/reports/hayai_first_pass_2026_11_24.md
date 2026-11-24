# hayai-ocr v2.7 Supernova over every Manga109 detector box, and its place on the review page (2026-11-24)

A first-pass read for the boxes with no label and a second read beside the main / sub
proposals. The policy: [`main-sub-approach.md`](../main-sub-approach.md); the page: README § Rules.

## Facts

- The user (2026-11-24): "아직 레이블 안된거 hayai 2.7 supernova로 레이블링 초벌 쓰고 싶어"; the
  rows picked: the boxes with no annotation, the mains waiting on a line break, every unreviewed
  main / sub row. On the licence: README § Labels.
- `labeling/hayai_read.py`, `JustANormalTinkerer/hayai-ocr-v2.7-supernova` at `90fc6148`,
  `max_patches="auto"`, batch 64: all 164,090 boxes of `manifest_sfxp_det.parquet` (train
  150,081, val 14,009) in 302 s, 544 boxes/s. Job `20261124-090142-1b699b`, rc 0. Output
  `output/ocr/hayai/hayai27_det.jsonl` (local).
- A box of `manifest_sfxp_det_all` is read from its crop on disk (134,613); the other 29,477 are
  cut from the page with the training margin.
- No read is empty or missing, on any cat: `neg_other` 22,718 train / 1,955 val, `drop` 2,678 /
  168 included.
- No read holds a line break (0 of 164,090).
- On the page (`review_page.py --hayai`, built 09:07):
  - `mainsub_det` 122,853 rows, each with hayai's read. 12,947 differ from the main
    (whitespace-blind, `exact_key`'s folds, a read equal to main + sub counted as a match);
    2,923 of them are among the 8,412 mains waiting on a line break. By distance to the main
    (1 − similarity): under 0.2 8,397, 0.2 – 0.4 2,804, 0.4 – 0.6 902, 0.6 – 0.8 265, over 0.8 579.
  - `mainsub_nolabel` 25,396 rows (`neg_other` and `drop`, train), the main box holding hayai's
    read; 91 over the training cap are not listed. Their crops were cut to
    `derived/crops_sfxp/det_nolabel/train/` (local).
- Six rows read at distance 0.333: `え──!!` read `えー!!`, `そお？` read `そぉ?`, `ワー⏎ワー` read
  `ワー` (twice), `わ――――っ!!!` read `わーっ!!!`, `ブー` read `ブ`.
- `apply_review.py` takes `mainsub_nolabel` rows into `assets/m109_mainsub.tsv`.
  `crop_dataset.load_mainsub` reads `manifest_sfxp_det_all` only, so a reviewed no-label row does
  not train.

## Facts — the kozh and hito crops (added 14:45)

- The user (2026-11-24): "gelnote나 swaprelax, hito v4도 hayai로 disagrees scoring 해보자".
  `hayai_read.py --set kozh hito --max_new_tokens 256`, the trained crops of
  `manifest_sfxp_kozh` and `manifest_pseudo_hito_v4`: 39,981 crops at 476 crops/s, job
  `20261124-144313-4ef189`, rc 0. Output `output/ocr/hayai/hayai27_kozh_hito.jsonl` (local).
  No read is missing or empty; none holds a line break.
- Rows whose label hayai's read does not match (`hayai_distance` against the label, sub empty):

  | source | rows | disagree | = 0.001–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | > 0.8 | read has kana / hanja the label lacks |
  |---|---|---|---|---|---|---|---|---|
  | `gelnote_ko` | 1,697 | 150 (9 %) | 69 | 17 | 21 | 12 | 31 | 77 |
  | `swaprelax_v5` | 20,327 | 5,761 (28 %) | 3,015 | 639 | 603 | 593 | 911 | 2,605 |
  | `hito_v4_ko` | 6,904 | 963 (14 %) | 351 | 137 | 186 | 94 | 195 | 464 |
  | `hito_v4_ja` | 8,926 | 412 (5 %) | 348 | 50 | 11 | 1 | 2 | 0 (Hangul) |
  | `gelnote_zh` | 2,127 | 599 (28 %) | 234 | 225 | 84 | 20 | 36 | — |

- Sampled KO reads with a script slip: `찔러주니까` read `찔た주니까`, `자유` read `ス유`, `선생님` read
  `色人!!`, `응…` read `9...`, `아♥` read `OH♡`. ZH reads come back in Japanese forms (`说` read `説`,
  `晓` read `暁`). `gelnote_zh` is not on the page.
- Page (`--hayai_kozh`, built 14:45): the four sources above show the read with its distance;
  `hyx` marks a script slip and puts the row last in both hayai sorts; `hayai near miss` lists the
  disagreeing rows smallest distance first.

## Interpretation

- hayai answers something on every box, so on the no-label rows an empty main has to come from
  the reviewer: a box of drawing with no text still shows a read.
- With no line breaks in its reads, hayai says nothing about the mains waiting on a break beyond
  whether their characters match.
- On the Korean rows hayai's own errors dominate the large distances: about half of the
  disagreements are script slips, and most of the rest above 0.4 on sampled rows are misreads
  of short interjections (`아` read `Oト`, `하아` read `計計`). The near-miss end is where a label
  typo would show; `hito_v4_ja` disagrees on 5 %, mostly `…` / `♡` / small kana.
- The low-distance disagreements look like spelling of lengthening marks and small kana; the
  sort puts the largest distances first for that reason. The share of real label errors among
  the 12,947 is not known.
