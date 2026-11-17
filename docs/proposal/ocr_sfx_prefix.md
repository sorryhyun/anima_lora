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

## Detector boxes against COO polygons (2026-11-17)

Every train COO polygon typed against the AnimeText boxes of its page
(`project/finished/ocr_reader/probes/sfxp_crop_sheet.py`, one contact sheet per type):

| type | polygons | share | |
|---|---|---|---|
| same: one box, this SFX alone, crop ≈ COO crop | 16,312 | 42.2 % | kept |
| missed: no box covers 20 % of it | 9,353 | 24.2 % | out (as before) |
| merged: 2+ SFX in one box | 7,687 | 19.9 % | order-doubtful boxes cut |
| fragment: the detector splits or cuts it | 1,741 | 4.5 % | out (as before) |
| graze / cut / mixed | 895 / 694 / 492 | 5.4 % | kept |
| dup: inside 2+ kept boxes (nested boxes) | 609 | 1.6 % | outer boxes cut |
| drop_nb: fits a box dropped for a cut neighbour | 566 | 1.5 % | out (as before) |
| loose: one SFX in a much larger box | 286 | 0.7 % | cut |

- **missed / fragment** stay out of training: AnimeText will not detect them at inference either.
  The COO gate still scores them, so part of the COO SFX gap to v6 sits on crops the pipeline
  never produces.
- **loose**: v6 reads text beyond the target in 15 % of these crops (titles, signs, asides
  annotated in neither COO nor `<text>`) against 0.1 % of `same`.
- **merged**: the build joins right to left, then top down. That is right for vertical columns
  side by side and wrong for most stacks (135 of 251 with differing texts) and horizontal rows
  (45 of 46); scattered SFX have no order. v6 reads these boxes in yet another order.
- 394 `SFX:` negatives held part of an SFX (`neg_graze`).

`build_sfx_prefix.py --clean` writes `manifest_sfxp_det_clean.parquet` (the default of
`load_sfxp` / `--sfxp_det`): dup outer 532, loose 286, merged with doubtful order 620 (repeats and
side-by-side vertical columns kept), `neg_graze` 394. Train rows 107,879 → 105,949: `SFX:`
positives 22,770 → 21,332, negatives 5,692 → 5,332, `OCR:` speech −106.

KO labels (`probes/ko_label_review.py`, applied by `probes/apply_ko_review.py`): gelnote KO
pseudo labels take the teacher's spacing and line breaks, which the whitespace-blind agreement
does not check. Hand review fixed spacing and breaks on about 160 rows; 10 duplicate gelnote notes
(same or nested box) dropped; spaces at line edges removed.

### 1k on the cleaned data (`sfxpc_1117_2110`, the pilot's recipe)

| | pilot `vl16_sfxp_1k`, 4 px | cleaned, 4 px | per crop (cleaned only / pilot only) |
|---|---|---|---|
| COO SFX, `SFX:` | 67.3 % | 66.6 % | 188 / 207 (z −1.0) |
| COO SFX, `OCR:` | 2.7 % | 2.7 % | |
| COO speech, `OCR:` | 85.0 % | 84.9 % | |
| doujin SFX, strict / ♡-blind | 97 / 249 | 129 / 210 | strict z +3.3, ♡-blind z −3.3 |

No change on COO, as expected: 1,000 steps see about 200 of the removed boxes. The doujin moves
(more trailing hearts, more misread glyphs such as びく → ぐく) touch no data the clean changed and
are read as the spread between two undertrained 1k runs.

### swaprelax_v5 boxes and the second review (2026-11-17)

- **Nested boxes.** swaprelax boxes are AnimeText's, and the swap_text read kept the outer box of
  a nested pair (the raw boxes are in the relax tree's `det.json`). 983 rows hold a smaller
  detector box. The v6 confidence does not separate the bad ones (the read of the outer crop is
  right); the share of the label the inner boxes' v6 reads cover does. `probes/nested_inner_read.py`
  → `probes/apply_nested_fix.py`: coverage ≥ 0.85 takes the inner box (loose outer box, label
  kept; the union when the detector split one block), below drops the row (a line of a block, or
  several blocks merged): 500 re-boxed, 387 dropped.
- **Unswapped groups.** The staged target is the JA page with the clean KO groups pasted in, so
  108 rows held JA lettering under a KO label (target equals the JA source inside the box).
  `probes/unswapped_ja.py` turns them into JA rows: the JA page's AnimeText box, the crop from the
  JA source, v6's JA read (conf ≥ 0.9, one row per JA box): 91 relabelled, 16 dropped. The JA
  labels are not audited.
- Both decisions sit in `swap_relax_v5/{nested_fix,unswapped_ja}.json`, which its `build.py`
  applies on a rebuild. Two more review exports applied (161 edits, 1 drop).
- `manifest_sfxp_kozh`: 24,574 rows at `sfxpc_1117_2110` → 24,160; about 1,450 rows removed or
  changed, all `OCR:` KO / ZH speech (1.4 % of the 1k run's train rows).

Batching defaults in `finetune_vl16_lora.py`: bs 4 × 4 without gradient checkpointing, batches
cut at 3,072 padded patch tokens, crops capped at 2,048 tokens, no stock val before training.
97 % of the crops sit at the processor's min_pixels floor (600–700 tokens), so the budget only
splits the rare large ones (98 % of batches hold 4 crops, 3.5 % padding). 16.7 crops/s at 11.8 GB
peak on the costliest batches, against 13.7 for bs 2 × 8 and 13.0 for bs 8 × 2 with
checkpointing; GPU utilisation at bs 2 was 70–80 % at ~220 W, the kernels too small to fill the
card. Removing the per-step host syncs (loss `.item()`, label length) changed nothing.

### 1k on the fixed data, bs 4 × 4 (`sfxpc_1117_2300`)

18.3 crops/s over the run (13.0 for `sfxpc_1117_2110`); peak 14.1 GB, above the smoke test's
11.8 because long labels make the LM side of a batch dear, which `--largest_first` (vision cost)
does not reach.

| | pilot | `2110` | `2300` | paired z, 2300 vs 2110 |
|---|---|---|---|---|
| COO SFX, `SFX:`, 4 px | 67.3 % | 66.6 % | 66.5 % | −0.1 (192 / 193) |
| COO SFX, `OCR:` | 2.7 % | 2.7 % | 3.5 % | |
| COO speech, `OCR:` | 85.0 % | 84.9 % | 85.3 % | +1.1 |
| doujin SFX, strict / ♡-blind | 97 / 249 | 129 / 210 | 114 / 215 | −1.7 / +0.5 |
| val SFX | | 73.9 % | 76.4 % | +2.7 |
| val speech | | 87.5 % | 87.7 % | +0.5 |
| val `OCR:` empty on SFX | | 97.7 % | 95.4 % | −2.9 |
| val `SFX:` empty on speech | | 89.9 % | 87.6 % | −2.0 |

COO and doujin do not move. Val SFX gains while both negatives lose, and the batching changed
with the data, so neither is credited. None of these sets holds Korean speech
(`manifest_sfxp_kozh` is train-only; val and COO speech are Manga109), so the KO label fixes are
not measured here; that needs a held-out KO set.

### 1k after the KO review and target norm 6 (`sfxpc_1118_0031`)

Same recipe as `2300`, on the data after three more KO review exports (179 edits), three box
fixes (two loose JA boxes tightened, one two-balloon box split into two rows), the swaprelax crops
recut on the current staged pages (159 had been cut before the 11-09 restage; two KO labels whose
group the restage left in Japanese dropped) and `TARGET_NORM = 6` (curly double quotes → `"`, a
lone conjoining jamo → the typed compatibility jamo, `ᄏ` → `ㅋ`). 17.8 crops/s, peak 14.1 GB.

| | `2300` | `0031` | paired z |
|---|---|---|---|
| COO SFX, `SFX:`, 4 px | 66.5 % | 66.6 % | +0.1 (184 / 182) |
| COO SFX, `OCR:` | 3.5 % | 2.0 % | −4.2 |
| COO speech, `OCR:` | 85.4 % | 86.8 % | +3.0 |
| doujin SFX, strict / ♡-blind | 114 / 215 | 173 / 253 | +5.9 / +3.4 |
| doujin, 86 user-checked | 18 | 23 | +1.5 |
| val SFX | 76.4 % | 74.8 % | −1.6 |
| val speech | 87.7 % | 87.8 % | +0.2 |
| val `OCR:` empty on SFX | 95.4 % | 99.5 % | +4.8 |
| val `SFX:` empty on speech | 87.6 % | 93.4 % | +4.4 |

Both negatives recover what `2300` lost, and COO SFX read under `OCR:` falls with them: the two
prompts separate better. Doujin strict moves most, and part of it is the heart (213 predictions
carry one against 150); strict moved −15 between `2110` and `2300`, so the ♡-blind +38 is the
firmer number. The data edits, the norm change and the batch order all moved at once, so no single
one is credited. Rescoring `2300` under norm 6 moves only COO speech (2,184 → 2,196). Korean
speech is still unmeasured.

## Next

- The 2-epoch run on the cleaned data (`sfxpc_1118_*`, queued 11-18).
- Score the COO gate on the polygons a detector box covers, beside the full set.

## Downstream

- `anime_tools.ocr.reader`: reads with a chosen prompt (`OCR:` or `SFX:`).
- The OCR stage takes each line's kind from the two reads. `line_kind` remains for the reads of
  other readers.
- Cut a new anime_tools tag and bump the pin; upload the weights as Hub `v7/`.
