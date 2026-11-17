# OCR reader: an `SFX:` task prompt

Status: proposed (2026-11-17).

Goal: retrain the manga reader (PaddleOCR-VL, Hub `v6/`, `anime_tools.ocr.reader`) with a
second task prompt. `SFX:` returns only the sound effects in the crop, or an empty string when
there are none. `OCR:` stays as it is and reads everything. The reader then decides from the
image whether a line is SFX, instead of the text rules deciding it from the read string afterwards.

## Why

- **The text rules miss often.** `anime_tools.captions.ocr_sfx.line_kind` decides speech vs SFX
  from the string alone (a vowel initial means speech, a repeat means SFX, …).
  - On the sincos hand labels it calls 113 of 597 SFX speech (`おっ`, `へこ♡`, `フリ♡`, `たぷん♡`,
    `はぁ♡`) and 12 of 272 speech lines SFX (`ムラッ`, `ダメッ♡`).
  - On the Manga109 train split it calls 10,522 of 38,582 COO SFX speech, and 2,124 of 38,634
    speech lines SFX.
  - These strings are ambiguous as text. In the image they are not: SFX is drawn, speech is
    lettered.
- **Dialogue in SFX crops.** 24.8 % of the COO SFX train crops take in at least 15 % of a
  neighbouring annotation at the current 0.12 × long-side pad. In 3,661 of them the neighbour is
  speech. `SFX:` is trained to leave that text out.

## Training data

All rows are cut from Manga109 annotations, where COO polygons and `<text>` boxes are both on
the page.

| prompt | crop | target |
|---|---|---|
| `SFX:` | COO SFX | the SFX label |
| `SFX:` | Manga109 speech | the COO SFX inside the crop, or `""` |
| `OCR:` | as v6 | as v6 |

- **Crop margin** for the rebuilt crops: min(0.12 · long, 0.3 · short, 4 px). At a 4 px cap,
  2.8 % of the SFX crops take in a neighbour (24.8 % at the current pad). Contact sheets of
  current vs 4 px show no clipped strokes.
- **Pseudo manifests** (gelnote KO / ZH, `pseudo_swaprelax_v5`) tag every row `kind = sfx`,
  although they are speech. Retag them as speech before building the `SFX:` rows. They then join
  as `SFX:` → `""` negatives or as `OCR:` rows only.
- There are no doujin SFX training rows. sincos stays eval-only.

## Gate

Criteria are set before the run:

- **Kind**: on the sincos 869 rows, a non-empty `SFX:` read counts as SFX. Compare against
  `kind_hand`. Baseline is the text rule: 484 / 597 SFX, 260 / 272 speech.
- **SFX read**: `SFX:` on the 597 SFX rows, strict and ♡-blind, against v6's `OCR:` (349 / 413).
- **No regression on `OCR:`**:
  - COO test SFX 85.3 % and speech 88.9 % (v6);
  - K3 KO 28 / 48, ZH 30 / 48.

## Downstream

- `anime_tools.ocr.reader`: reads with a chosen prompt (`OCR:` or `SFX:`).
- The OCR stage takes each line's kind from the `SFX:` read. `line_kind` remains for the reads of
  other readers.
- Cut a new anime_tools tag and bump the pin; upload the weights as Hub `v7/`.
