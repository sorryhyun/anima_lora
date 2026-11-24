# main / sub data: detector crops, proposals, line-break and furigana passes (2026-11-21)

The policy these serve is [`main-sub-approach.md`](../main-sub-approach.md). Paths under `output/`,
`assets/` and the Manga109 derived root are local.

## Facts

- main / sub is read on the boxes AnimeText gives (the user's call, 2026-11-21, after the
  missed polygons went: "let's try it"). The `<text>`-box crops (`manifest_sfxp_speech`)
  are out of the review; their texts (line breaks, review edits) still label the speech
  inside the detector crops.
- Detector coverage of the 39,029 train COO polygons (`sfxp_types_train.parquet`, written by
  `build_sfx_prefix.py --clean`): missed 9,353, fragment 1,741, the rest in or near a box.
- `build_sfx_prefix.py --det_all` → `manifest_sfxp_det_all.parquet`: every `--clean` box
  holding text, each with a crop. The 95,193 `neg_speech` boxes `--neg_keep 0.15` had left
  unwritten were cut (train + val, 16 s). `load_sfxp` does not read this manifest.

`labeling/m109_mainsub.py` scores each COO polygon and `<text>` box on the page by the share of
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
- The furigana source (`labeling/furigana_find.py`, ruby read ×3 at 12 px pad, the agents'
  fixes where they reviewed the row) heads the sub proposal, through the `<text>` ids in
  the crop. A `trim` row takes only the ruby groups whose centre is in its crop
  (`trim_ruby`).
- The 60,810 `<text>` boxes outside `manifest_sfxp_speech` (2026-11-21,
  `probes/mainsub_text.py` → `manifest_mainsub_text.parquet`, crops under
  `crops_sfxp/mainsub_text/`) went through the speech labels' passes: `linebreak_auto.py
  inkread --set mainsub_text` (stock reader, 113,461 column crops, 663 s) and `apply --set
  mainsub_text` (gate passed 32,182; 6,398 to `linebreak_suspicious_mainsub_text.jsonl`;
  35,122 of 60,810 now broken); then `furigana_find.py find --manifest
  mainsub_text --name ruby_mt` and `read` (32,081 groups on 16,836 crops, 508 s; aligned
  over a kanji 5,029 before the breaks, 26,553 after; read as kana 23,287). The page and
  `m109_mainsub.py` take `ruby_m109_read_u3p12.jsonl` = `ruby_all` + `ruby_mt` + the
  horizontal reads below.
- Horizontal text (`furigana_find.py find --orient horizontal`): a crop counts as horizontal
  when its longest x ink run (gaps ≤ 3 px closed) is 1.5× its longest y run
  (`ink_horizontal`); the box shape does not tell (most crops tagged `horizontal` hold
  vertical columns in a wide balloon) and neither do the gap widths (tried first: a single
  column has no gap across). The crop is turned 90° clockwise for the finder and the boxes
  turned back. 2,186 crops judged horizontal; 299 with groups, 527 groups, 290 read as kana;
  49 replace a vertical record of the same box. In a 12-crop sheet 10 were horizontal text;
  ruby on the top row is often missed (it touches the crop edge and `ink_mask` drops it).
- `m109_mainsub.py --ruby … --ruby_fix output/ocr/furigana/agent` writes `subp`: per
  `<text>` box in the crop the agents' sub where they reviewed it, else the reads, a trim
  crop the ruby inside it, then the annotated sub items. The training loader reads `subp`;
  the page assembles its own (with the furigana rows' notes and sub state).

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
edit 1,773, ok 82, none 74, unsure 71. The same review of the 2,000 most doubtful `ruby_mt`
rows (`agent/mt_top2000`, 11-21): edit 1,661, ok 141, none 100, unsure 98; 12 rows read
against their sheets, 12 right. On 14 `ruby_mt` rows read against the crop: 7
right, 3 partial (groups missed: 秘薬, 復活, 藩主), 4 poor (misreads ヴェラしゃ / おば,
junk `゙` / `ら`). Line breaks on 10 `mainsub_text` rows: 10 right.

The user's 12 `trim` edits of 11-21 against the rebuilt proposals: main as edited on 4;
ruby close on 1 (`さ きすけさま` for `き き すけさま`). Misses: horizontal text has no
furigana pass (`furigana_find` skips `orient == horizontal`, `１年前`), the finder found 1
of the 4 groups on AkkeraKanjinchou_050_16, two partly inside boxes in one crop
(MisutenaideDaisy_097_40).

Not covered by the annotations: unannotated text (signs, titles, asides outside a box) and
smaller lettering inside one `<text>` box.

The example behind the detector / annotation split: the Manga109 `<text>` annotation of
`BakuretsuKungFuGirl_038_0000a4fc` joins `ふざけてんの？` into the speech label. AnimeText gives
the two as separate boxes: `BakuretsuKungFuGirl_038_07` holds the main block (the annotation
box 73 % inside), `ふざけてんの？` has a box of its own.

## Interpretation

- The detector boxes sit close to the `<text>` boxes but are not copies of them (median IoU
  0.845 at the same margin), so the earlier "AnimeText learned Manga109" read is not settled
  by box geometry alone.
- Where the two disagree on what is one block (`BakuretsuKungFuGirl_038_07`), the detector
  box follows the lettering and the annotation does not; those crops are the `trim` rows.
- The annotations prefill a sub on few crops apart from ruby. Most other subs come from text
  only partly visible, which the reviewer types.
