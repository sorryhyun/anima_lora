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
- main / sub is read on the boxes AnimeText gives. The COO polygons it misses
  (`manifest_sfxp_missed`, the `--sfxp_missed` rows) are out: they were there for the
  `SFX:` prompt.
- `OCR2:` rows only where the label is complete. A crop with no kanji cannot carry ruby,
  so its main-only label is complete and it gives an `OCR2:` negative.

Examples:

| crop | main | sub |
|---|---|---|
| `Nekodama_13_013_25` | the large `!` | the small speech |
| `PLANET7_3_003_15` | `0!!` | `ゼロ ズバン! !!` |
| `BakuretsuKungFuGirl_038_0000a4fc` (`<text>` crop) | the speech | the ruby over 冷凍 and the smaller hand-lettered `ふざけてんの？` |
| `GakuenNoise_048_0001a044` (`<text>` crop) | `死にさらせっっ!!` (the label) | `ス` (an SFX cut by the edge, as it shows) |

The Manga109 `<text>` annotation of `BakuretsuKungFuGirl_038_0000a4fc` joins `ふざけてんの？`
into the speech label. AnimeText gives the two as separate boxes: `BakuretsuKungFuGirl_038_07`
holds the main block (the annotation box 73 % inside), `ふざけてんの？` has a box of its own.

## Training crops and the proposals

### Facts

- main / sub is read on the boxes AnimeText gives (the user's call, 2026-11-21, after the
  missed polygons went: "let's try it"). The `<text>`-box crops (`manifest_sfxp_speech`)
  are out of the review; their texts (line breaks, review edits) still label the speech
  inside the detector crops.
- Detector coverage of the 39,029 train COO polygons (`sfxp_types_train.parquet`, written by
  `build_sfx_prefix.py --clean`): missed 9,353, fragment 1,741, the rest in or near a box.
- `build_sfx_prefix.py --det_all` → `manifest_sfxp_det_all.parquet`: every `--clean` box
  holding text, each with a crop. The 95,193 `neg_speech` boxes `--neg_keep 0.15` had left
  unwritten were cut (train + val, 16 s). `load_sfxp` does not read this manifest.

`probes/m109_mainsub.py` scores each COO polygon and `<text>` box on the page by the share of
its convex hull inside the crop:

- 0.8 or more: in the crop. main = the item with the largest glyph (`sqrt(hull area /
  glyphs)`), joined with same-kind items of at least 0.7× that glyph and with repeats of
  the same text. sub = the other items.
- 0.05 to 0.8: partly visible, shown on the review page. A crop with nothing whole inside
  takes a speech box that is at least half inside as its main (`trim`), cut to what shows
  (`trim_lines`): the `<text>` box's columns come from the ink projection
  (`linebreak_auto.segment`); a column stays when half of it is inside, a glyph when its
  centre is (glyphs spread over the column's ink). Applied only when the ink is one column
  or the label's `\n` lines match the columns; an unbroken multi-column label stays whole.
  180 of the 576 `trim` rows are cut (201 after the line-break pass below). In a 16-row sample, 12 read as the crop shows, 3
  drop one glyph too many, 1 is wrong (MisutenaideDaisy_053_21).
  DollGun_015_14 (one column, the detector cut below `だ`): `な…なんだ　今…` → `な…なんだ`.
- Speech texts take `ellipsis_fold` (the speech labels' dot spelling) on every `<text>`
  box, including the 60,810 outside `manifest_sfxp_speech` that carry the raw annotation. A text
  with Hangul is left as is: Korean keeps `...` and `…` apart.
- The furigana source (`probes/furigana_find.py`, ruby read ×3 at 12 px pad, the agents'
  fixes where they reviewed the row) heads the sub proposal, through the `<text>` ids in
  the crop. A `trim` row takes only the ruby groups whose centre is in its crop
  (`trim_ruby`).
- The 60,810 `<text>` boxes outside `manifest_sfxp_speech` (2026-11-21,
  `probes/mainsub_text.py` → `manifest_mainsub_text.parquet`, crops under
  `crops_sfxp/mainsub_text/`) went through the speech labels' passes: `linebreak_auto.py
  inkread --set mainsub_text` (stock reader, 113,461 column crops, 663 s) and `apply --set
  mainsub_text` (gate passed 32,182; 6,398 to `linebreak_suspicious_mainsub_text.jsonl`, not
  on the page; 35,122 of 60,810 now broken); then `furigana_find.py find --manifest
  mainsub_text --name ruby_mt` and `read` (32,081 groups on 16,836 crops, 508 s; aligned
  over a kanji 5,029 before the breaks, 26,553 after; read as kana 23,287). The page and
  `m109_mainsub.py` take `ruby_m109_read_u3p12.jsonl` = `ruby_all` + `ruby_mt`.

Train split, `mainsub_det` (`output/ocr/mainsub/m109_mainsub.jsonl`, local):

| crops | 122,853: pos 20,978, pos_mixed 354, neg_speech 101,521 |
|---|---|
| sub proposed | 26,794 (26,468 from ruby; 11,744 before the `mainsub_text` pass) |
| partly visible text | 3,469 |
| main to trim | 576 |
| main and sub both empty | neg_speech 982 before `trim` |

One `<text>` box and no SFX in the crop: 98,854 crops. IoU of the detector box with that
box's 4 px cut: median 0.845; 5 / 25 / 75 / 90th percentile 0.683 / 0.786 / 0.884 / 0.911;
≥ 0.9 15.6 %, < 0.5 1.6 %.

Furigana finder, as measured when it was built: 74 % of the ruby groups over kanji read as
kana, about 80 % of those right. Agent review of the 2,000 most doubtful furigana rows:
edit 1,773, ok 82, none 74, unsure 71. On 14 `ruby_mt` rows read against the crop: 7
right, 3 partial (groups missed: 秘薬, 復活, 藩主), 4 poor (misreads ヴェラしゃ / おば,
junk `゙` / `ら`). Line breaks on 10 `mainsub_text` rows: 10 right.

The user's 12 `trim` edits of 11-21 against the rebuilt proposals: main as edited on 4;
ruby close on 1 (`さ きすけさま` for `き き すけさま`). Misses: horizontal text has no
furigana pass (`furigana_find` skips `orient == horizontal`, `１年前`), the finder found 1
of the 4 groups on AkkeraKanjinchou_050_16, two partly inside boxes in one crop
(MisutenaideDaisy_097_40).

Not covered by the annotations: unannotated text (signs, titles, asides outside a box) and
smaller lettering inside one `<text>` box.

### Interpretation

- The detector boxes sit close to the `<text>` boxes but are not copies of them (median IoU
  0.845 at the same margin), so the earlier "AnimeText learned Manga109" read is not settled
  by box geometry alone.
- Where the two disagree on what is one block (`BakuretsuKungFuGirl_038_07`), the detector
  box follows the lettering and the annotation does not; those crops are the `trim` rows.
- The annotations prefill a sub on few crops apart from ruby. Most other subs come from text
  only partly visible, which the reviewer types.

## Review loop

```
ANIMA_MANGA109S_ROOT=… python project/finished/ocr_reader/ocr/build_sfx_prefix.py --det_all
ANIMA_MANGA109S_ROOT=… python project/finished/ocr_reader/probes/m109_mainsub.py \
    --out output/ocr/mainsub/m109_mainsub.jsonl --ruby output/ocr/furigana/ruby_m109_read_u3p12.jsonl
python project/finished/ocr_reader/probes/ko_label_review.py --out ~/ko_label_review \
    --flags <ko_reads>_flags.parquet --doujin_reads <sfx>_cap4_sfx.jsonl \
    --ruby output/ocr/furigana/ruby_m109_read_u3p12.jsonl --ruby_fix output/ocr/furigana/agent \
    --mainsub output/ocr/mainsub/m109_mainsub.jsonl
python project/finished/ocr_reader/probes/apply_ko_review.py <export.tsv>
```

- The page's `has` filter lists rows by what their boxes hold: main only, sub only, main +
  sub, partly visible text. Emptying the main box makes a sub-only row. The default
  `suspicion` sort puts `trim` rows, then partly visible text, first.
- A row holding one `<text>` box with ruby takes its furigana row's sub edits as its
  starting sub.
- The export holds only what differs from the applied state (the page reads the assets at
  build time), so rebuild the page after each apply.
- Apply writes the `mainsub_*` rows to `assets/m109_mainsub.tsv` (source, id, status, main,
  sub) and the other sources' subs to `assets/sub_labels.tsv` (KO / ZH, furigana). The
  manifests are not touched by `mainsub_*` rows.

## Open

- A loader that builds `OCR:` / `OCR2:` rows from `manifest_sfxp_det_all`, the KO / ZH
  manifests and the two TSVs, in place of `crop_dataset.load_sfxp`'s speech / SFX split.
- Which crops count as complete for `OCR2:` before the review covers them.
- `drop` boxes (an SFX cut partway, 2,678 train) and `neg_other` boxes (no annotation inside,
  22,718 train) stay out; under main / sub a cut SFX could be sub.
- The gate under main / sub: the v7 gates (COO SFX / speech, doujin SFX, K3) read per kind;
  how a main / sub read is scored is not set.
