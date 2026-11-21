# ocr_reader — main / sub approach (`OCR:` / `OCR2:`)

The plan that follows v7's two prompts (`OCR:` speech / `SFX:` sound effects,
[`docs/proposal/ocr_sfx_prefix.md`](../../../docs/proposal/ocr_sfx_prefix.md)). Opened 2026-11-21
on the user's call. Labels are under review. Training does not read them yet.

## Prompts and label rules

### Facts (the user's calls, 2026-11-21)

- `OCR:` reads the main text, `OCR2:` the sub text of the same crop. The pair replaces the
  speech / SFX split. The kind of a line (speech or SFX) comes from the read text
  (`line_kind`), not from the prompt.
- main = the largest text in the crop. sub = everything else visible in the crop, one item
  per line, in reading order. Furigana is sub.
- Ignoring visible text is the harm. Text cut by the crop edge goes into sub as far as it
  shows.
- The label follows the 4 px training crop as shown. The 12 % crops (`manifest.parquet`,
  the page's `m109_sfx` reference rows) are not part of this plan.
- `OCR2:` rows only where the label is complete. A crop with no kanji cannot carry ruby,
  so its main-only label is complete and it gives an `OCR2:` negative.

Examples:

| crop | main | sub |
|---|---|---|
| `Nekodama_13_013_25` | the large `!` | the small speech |
| `PLANET7_3_003_15` | `0!!` | `ゼロ ズバン! !!` |
| `BakuretsuKungFuGirl_038_0000a4fc` (speech) | the speech | the ruby over 冷凍 and the smaller hand-lettered `ふざけてんの？` |
| `GakuenNoise_048_0001a044` (speech) | `死にさらせっっ!!` (the label) | `ス` (an SFX cut by the edge, as it shows) |

In `BakuretsuKungFuGirl_038_0000a4fc` the Manga109 `<text>` annotation already joins
`ふざけてんの？` into the speech label, so that crop's main loses it.

## Training crops and the proposals

### Facts

`probes/m109_mainsub.py` lists every Manga109 crop the v7 mix trains on (train split).
Each COO polygon and `<text>` box on the page is scored by the share of its convex hull
inside the crop:

- 0.8 or more: in the crop. main = the item with the largest glyph (`sqrt(hull area /
  glyphs)`), joined with same-kind items of at least 0.7× that glyph and with repeats of
  the same text. sub = the other items.
- 0.05 to 0.8: partly visible. Shown on the review page, never prefilled.

On speech crops the furigana source (`probes/furigana_find.py`, ruby read ×3 at 12 px pad,
the agents' fixes where they reviewed the row) heads the sub proposal.

| source | crops | rows | sub proposed | partly visible text |
|---|---|---|---|---|
| `mainsub_det` | detector boxes (`manifest_sfxp_det_clean`) | 36,363 (pos 20,978, pos_mixed 354, neg_speech 15,031) | 282 | 1,333 |
| `mainsub_speech` | `<text>` boxes (`manifest_sfxp_speech`: sfxp_speech + m109_lines) | 41,769 | 11,424 (11,412 from ruby) | 1,109 |
| `mainsub_missed` | missed COO polygons (`manifest_sfxp_missed`) | 4,633 | 14 | 653 |

Furigana finder, as measured when it was built: 74 % of the ruby groups over kanji read as
kana, about 80 % of those right. Agent review of the 2,000 most doubtful furigana rows:
edit 1,773, ok 82, none 74, unsure 71.

Not covered by the annotations: unannotated text (signs, titles, asides outside a box) and
smaller lettering inside one `<text>` box (`ふざけてんの？` above).

### Interpretation

- The annotations prefill sub on few SFX crops (296 of 41k). Most subs there come from text
  only partly visible, which the reviewer types.
- `neg_speech` crops were `SFX:` negatives. Under main / sub they are speech crops; the ones
  whose every speech box is cut are the "sub only" rows.

## Review loop

```
ANIMA_MANGA109S_ROOT=… python project/finished/ocr_reader/probes/m109_mainsub.py \
    --out output/ocr/mainsub/m109_mainsub.jsonl
python project/finished/ocr_reader/probes/ko_label_review.py --out ~/ko_label_review \
    --flags <ko_reads>_flags.parquet --doujin_reads <sfx>_cap4_sfx.jsonl \
    --ruby output/ocr/furigana/ruby_all_read_u3p12.jsonl --ruby_fix output/ocr/furigana/agent \
    --mainsub output/ocr/mainsub/m109_mainsub.jsonl
python project/finished/ocr_reader/probes/apply_ko_review.py <export.tsv>
```

- The page's `has` filter lists rows by what their boxes hold: main only, sub only, main +
  sub, partly visible text. Emptying the main box makes a sub-only row.
- A `mainsub_speech` row with ruby takes its furigana row's sub edits as its starting sub.
- Apply writes the `mainsub_*` rows to `assets/m109_mainsub.tsv` (source, id, status, main,
  sub) and the other sources' subs to `assets/sub_labels.tsv` (KO / ZH, furigana). The
  manifests are not touched by `mainsub_*` rows.

## Open

- A loader that builds `OCR:` / `OCR2:` rows from the manifests plus the two TSVs, in place
  of `crop_dataset.load_sfxp`'s speech / SFX split.
- Which crops count as complete for `OCR2:` before the review covers them.
- The gate under main / sub: the v7 gates (COO SFX / speech, doujin SFX, K3) read per kind;
  how a main / sub read is scored is not set.
