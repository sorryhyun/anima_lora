# OCR reader: `OCR:` for speech, `SFX:` for sound effects

Status: data built, 1k-step pilot and a 2-epoch run stopped after epoch 1 (2026-11-17).

Goal: retrain the manga reader (PaddleOCR-VL, Hub `v6/`, `anime_tools.ocr.reader`) with two task
prompts. `OCR:` reads speech and answers `""` on a crop holding only SFX. `SFX:` reads the sound
effects and answers `""` on a crop holding only speech. The reader then decides from the image
whether a line is SFX, instead of the text rules deciding it from the read string afterwards.

## Why

- **The text rules miss often.** `anime_tools.captions.ocr_sfx.line_kind` decides speech vs SFX
  from the string alone (a vowel initial means speech, a repeat means SFX, …).
  - On the sincos hand labels it calls 113 of 597 SFX speech (`おっ`, `へこ♡`, `フリ♡`, `たぷん♡`,
    `はぁ♡`) and 12 of 272 speech lines SFX (`ムラッ`, `ダメッ♡`).
  - On the Manga109 train split it calls 10,522 of 38,582 COO SFX speech, and 2,124 of 38,634
    speech lines SFX.
- **Dialogue in SFX crops.** 24.8 % of the COO SFX train crops take in at least 15 % of a
  neighbouring annotation at the current 0.12 × long-side pad. In 3,661 of them the neighbour is
  speech.

## Training data

| prompt | rows | target |
|---|---|---|
| `OCR:` | Manga109 speech (`<text>` boxes) | the speech |
| `OCR:` | KO / ZH (gelnote KO / ZH, `pseudo_swaprelax_v5`) | labels as v6 |
| `OCR:` | detector boxes holding SFX and speech | the speech only |
| `OCR:` negatives | detector boxes holding only SFX | `""` |
| `SFX:` | detector boxes holding SFX (with or without speech) | the SFX only |
| `SFX:` negatives | detector boxes holding only speech, KO / ZH crops (half each) | `""` |

- Negatives are at most 20 % of each prompt's rows.
- **Detector boxes**: the OCR stage's AnimeText detector (`anime_tools.ocr.animetext`, `denest`
  inner) over every Manga109 train / val page, so the crops are the boxes the reader gets at
  inference. Labels come from the Manga109 annotations by pixel overlap with each COO polygon:
  `pf` = share of the polygon inside the crop, `cf` = share of the crop the polygon covers.
  - `pf ≥ 0.8`: the SFX is in the target.
  - `pf < 0.2` and `cf < 0.2`: a graze, ignored.
  - Anything else (a polygon cut partway, or a fragment of a large SFX): the box is dropped.
  - A crop holds speech when a `<text>` box has ≥ 30 % of its area in it. The `OCR:` row of a
    mixed box needs every `<text>` box touching it to lie ≥ 80 % inside.
  - On a 160-page sample: 639 SFX-only, 16 mixed, 2,463 speech-only, 66 dropped of 3,660 boxes.
- **Crop margin**, every crop: min(0.12 · long, 0.3 · short, 4 px). KO / ZH are re-cut from their
  pages at this margin.
- There are no doujin SFX training rows. sincos stays eval-only.
- Build: `project/finished/ocr_reader/ocr/build_sfx_prefix.py`; loader `crop_dataset.load_sfxp`;
  training `finetune_vl16_lora.py --sfx_prefix`.

## Gate

| | prompt | v6 (`OCR:`) |
|---|---|---|
| COO test SFX | `SFX:` and `OCR:` | 85.3 % |
| COO test speech | `OCR:` | 88.9 % |
| doujin SFX (sincos 597, strict / ♡-blind) | `SFX:` | 349 / 413 |

COO test crops come in two margins: the v6 cut (0.12 × long side) and the capped margin
(`build_sfx_prefix.py --gt test`, `eval_manga109.py --manifest sfxp_gt`; doujin
`eval_sfx.py --pad cap4`).

## Pilot: 1k steps from the stock model (`vl16_sfxp_1k`)

v6's recipe (tower LoRA r64, LM full FT 1e-5, bs 8 × 2), `--sfx_prefix --steps 1000`: 16k crops,
about 1 / 12 of v6's steps.

| | v6, 0.12 pad | v6, 4 px | 1k, 0.12 pad | 1k, 4 px |
|---|---|---|---|---|
| COO SFX, `SFX:` | — | — | 64.5 % | 67.3 % |
| COO SFX, `OCR:` | 85.3 % | 85.2 % | 2.5 % | 2.7 % |
| COO speech, `OCR:` | 88.9 % | 88.4 % | 84.7 % | 85.0 % |
| doujin SFX, strict / ♡-blind | 349 / 413 | 292 / 423 | 90 / 221 | 97 / 249 |

- The kind split is learnt: `OCR:` answers `""` on 96.5 % of COO SFX crops and on 1.8 % of speech
  crops; `SFX:` answers `""` on 0.9 % of COO SFX and 0.7 % of doujin SFX.
- Matching the eval crops to the training crops helps, although the 4 px crop holds less of the
  page: the 1k reader gains 2.8 points on COO SFX and 28 ♡-blind lines on doujin. v6, trained on
  the 0.12 pad, is flat on COO; its doujin strict drop (349 → 292) is trailing hearts clipped, as
  ♡-blind rises 413 → 423.
- The pilot's reading gap to v6 looked like undertraining. The 2-epoch run below says SFX is not
  only that.

## Run `sfxp_1117_1816`: stopped after epoch 1 (6,790 of 13,580 steps)

Batching for this run (`finetune_vl16_lora.py`):

- `--token_budget 1536`: batches cut on padded patch tokens (largest crop × crops,
  `crop_dataset.token_batches(padded=True)`), not on area.
- `--max_tokens 2048`: training crops capped at 2,048 patch tokens (max_pixels 401,408); the
  73 crops above it are large-glyph notes and SFX, read the same at the lower resolution. Val keeps
  the processor default.
- bs 2 × 8 without gradient checkpointing: 14.8 crops/s, peak 11.5 GB, against 10.7–11.0 crops/s
  (peak 7.1 GB) for bs 8 × 2 with checkpointing. bs 8 without checkpointing runs out of memory on a
  16 GB card at 6k tokens per batch, with SDPA and with flash-attn 2 alike.

| | v6, 0.12 pad | v6, 4 px | 1k, 4 px | ep1, 0.12 pad | ep1, 4 px |
|---|---|---|---|---|---|
| COO SFX, `SFX:` | — | — | 67.3 % | 66.1 % | 71.0 % |
| COO SFX, `OCR:` | 85.3 % | 85.2 % | 2.7 % | 6.3 % | 7.9 % |
| COO speech, `OCR:` | 88.9 % | 88.4 % | 85.0 % | 85.4 % | 85.8 % |
| doujin SFX, strict / ♡-blind | 349 / 413 | 292 / 423 | 97 / 249 | 199 / 362 | 183 / 358 |

- Val at ep1: SFX 79.6 %, speech 87.0 %, `OCR:` empty on SFX 99.5 %, `SFX:` empty on speech 96.7 %.
- v6's own run gained 4.2 (SFX) and 4.0 (speech) val points from ep1 to ep2. The same step would
  put speech near v6 and leave COO SFX about 10 points short.
- On the 4 px COO SFX crops, v6 is right and ep1 wrong on 430 (the reverse on 67): 248 misread
  glyphs (ぱち → ばち, ギシッ → ゴシッ), 107 trailing extras (は → は…), 61 empty answers, 14 cut
  short.
- Open: the reader trains SFX on detector boxes and is scored on COO polygon crops; v6 trained
  on the polygon crops. A contact sheet of the two comes next.

## Next

- Compare the detector-box SFX training crops with the COO polygon crops before training again.

## Downstream

- `anime_tools.ocr.reader`: reads with a chosen prompt (`OCR:` or `SFX:`).
- The OCR stage takes each line's kind from the two reads. `line_kind` remains for the reads of
  other readers.
- Cut a new anime_tools tag and bump the pin; upload the weights as Hub `v7/`.
