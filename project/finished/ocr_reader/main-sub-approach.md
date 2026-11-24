# ocr_reader — main / sub (`OCR:` / `OCR2:`): definitions and dataset

What main and sub are, and how the training set is built from them. Runs and label passes are in
`reports/mainsub_*.md`; open questions and pending work are in [`mainsub_plan.md`](mainsub_plan.md);
the review page and its procedure are in the README's § Rules.

## The task

- Input: one crop, an AnimeText detector box cut with a 4 px margin. The reader is meant to read
  the detector's output, so the detector box is the unit, not the annotation's polygon or box.
- `OCR:` reads the crop's main text, `OCR2:` its sub text. Either can be speech or SFX: the prompt
  does not carry the kind, the kind of a line comes from the text (`line_kind`).

## Definitions

### main

- The text with the largest glyphs in the crop (the size of the printed characters, not the
  length of the text or the area of the block).
- Several items count as one main when they are of the same kind and their glyphs are at least
  0.7× the largest, or when they repeat the same text at any size. They are written one per
  line, right to left, then top to bottom.
- Line breaks are the printed lines: one per column of vertical text (right to left), one per
  row of horizontal text.

### sub

- Every other text visible in the crop, one item per line, in reading order: furigana, smaller
  speech or SFX, hand-lettered asides, smaller lettering that the annotation joins into the same
  `<text>` box, text cut by the crop edge, numbers drawn in the art (a clock face, a floor
  display; the user, 2026-11-23).
- A label that leaves out visible text teaches the reader to skip text, so a sub is complete or
  it is not trained (§ What trains).

### The crop decides

- The label follows the crop as shown, not the page and not the annotation.
- Text cut by the crop edge is typed as far as it shows. What only the page shows (the rest of a
  cut line, a neighbouring balloon outside the box) is not typed.
- A mark that cannot be told apart from drawing (a stroke cut by the edge, a symbol in the art)
  is not typed; a crop whose only other marks are of that kind has an empty sub (the user,
  2026-11-23: "그림인지 글자인지 모르면 출력 안하는게 맞는듯").

### Empty labels

- Empty main: a crop with only sub text. It trains as `OCR:` `""` only when reviewed as such.
- Empty sub: the crop has no sub. It trains as `OCR2:` `""` only where that is known (§ Sources,
  the negatives). Anywhere else an empty sub means "not known" and gives no `OCR2:` row.

### Spelling

- Japanese speech takes `ellipsis_fold`. Korean keeps `...` and `…` apart.
- Japanese lettering left under a Korean label stays Japanese.
- Targets are normalized by `normalize_target`; a target over 150 normalized characters is
  dropped, main and sub each on their own.

### Examples

| crop | shows | main | sub | from |
|---|---|---|---|---|
| `UchuKigekiM774_020_24` | a title, two balloons cut by the edge | `第4幕 宇宙さそりの逆襲 前編` | the cut speech as far as it shows (`もうよ` / `なよ` / `体に毒` …) | reviewed |
| `LoveHina_vol01_051_29` | `いてっ`; the next balloon's `…` is on the page only | `いてっ` | empty | reviewed |
| `JijiBabaFight_045_17` | the SFX `わはは` over a balloon | `わはは` | `もうけ` / `もうけ` | proposal |
| `AkkeraKanjinchou_051_11` | speech with ruby and a small SFX | `変なの…` | `へん` / `ヒソ` | proposal |
| `GarakutayaManta_006_34` | an English paragraph and a large `ひ～` | the paragraph | `ひ～` | reviewed (overrides glyph size; `mainsub_plan.md`) |

## Dataset

### Sources

| source | split | `OCR:` | `OCR2:` |
|---|---|---|---|
| Manga109 detector crops (`mainsub_det`: `manifest_sfxp_det_all`, labels `output/ocr/mainsub/m109_mainsub[_val].jsonl`) | train, val | main | sub; `""` on a row reviewed `ok` with an empty sub, and on an SFX-only crop (`sfx_only`: one SFX, no other annotated text whole or partly inside) |
| KO / ZH (`manifest_sfxp_kozh`) | train | the label | `assets/sub_labels.tsv`, else `assets/kozh_sfx.tsv`; `""` on a `swaprelax_v5` crop with neither |

Not in the set: the COO polygons the detector misses, the `<text>`-box crops
(`manifest_sfxp_speech`) and the older 12 % crops (`manifest.parquet`). The `<text>` labels
still label the speech inside the detector crops.

### Manga109 labels

Proposed by `probes/m109_mainsub.py` from the annotations. An agent label
(`assets/m109_agent_subs.tsv`, local: source, id, verdict, main, sub, pass) replaces the proposal, and a
reviewed row (`assets/m109_mainsub.tsv`: source, id, status, main, sub) replaces both.

- An annotated item (COO SFX polygon or `<text>` box) is in the crop at 0.8 of its convex hull
  or more, partly visible from 0.05 to 0.8.
- main and sub follow § Definitions over the items in the crop; glyph size is
  `sqrt(hull area / glyphs)`.
- sub (`subp`): per `<text>` box, the furigana first (the agents' fix where they reviewed it, else
  the ruby read), then the other items in the crop.
- With no item whole inside, a speech box at least 0.3 inside becomes the main, cut to the
  columns and glyphs that show (`trim`).
- An unreviewed main takes the line breaks agents read off its crop (`assets/m109_breaks.tsv`);
  its characters stay the proposal's.
- The annotations do not cover unannotated text (signs, titles, asides outside a box) or smaller
  lettering inside one `<text>` box; only review adds those.

### What trains

`crop_dataset.load_mainsub`, per Manga109 crop:

| state | `OCR:` | `OCR2:` |
|---|---|---|
| reviewed `drop` | out | out |
| reviewed | the main as reviewed | the sub as reviewed; `""` if empty (train) |
| agent `typed` | the agent's main if set, else the proposed main | the agent's sub |
| agent `empty` | the same | `""` (train, SFX-only group) |
| agent `other` (text inside, not typed) | the same | out |
| unreviewed `trim` | out | out |
| unreviewed, main empty | out | out |
| unreviewed, line breaks waiting on review or held (`m109_break_held.tsv`) (train) | out | as the rows below |
| unreviewed, partly visible text in the crop | the proposed main | out |
| unreviewed, SFX only (train) | the proposed main | `""` |
| unreviewed, other | the proposed main | the proposed sub, if not empty |

- Negatives (target `""`) are at most `--neg_share` (0.2) of each prompt's rows. The reviewed
  Manga109 negatives go in first; the rest of the share is split 1 : 1 between SFX-only crops
  and `swaprelax_v5` crops, each drawn at random.
- Of the unreviewed train subs headed by a ruby read no agent fixed (`ruby_raw` in
  `m109_mainsub.jsonl`), a quarter trains (`RUBY_RAW_SHARE`; the user,
  2026-11-23): 5,724 of 22,762 at seed 0, a hash of seed and id.
- Val reads `main` / `main_neg` / `sub`; the best checkpoint is picked by `main_exact`.

Train rows on 2026-11-23, with the agent labels, the held line breaks out and a quarter of the
unreviewed ruby subs (`load_mainsub('train')`):

| prompt | metric | rows | from |
|---|---|--:|---|
| `OCR:` | `main` | 113,566 | Manga109 (SFX main 21,205, speech main 92,036, no annotated item whole inside 325) |
| `OCR:` | `main_kozh` | 24,145 | KO / ZH |
| `OCR:` | `main_neg` | 3 | Manga109, reviewed |
| `OCR2:` | `sub` | 9,869 | Manga109 9,839 (unreviewed ruby reads 5,724 at seed 0), KO / ZH 30 |
| `OCR2:` | `sub_neg` | 6,726 | SFX-only 3,240 (of 20,490 with the agents' empty ones), `swaprelax_v5` 3,240 (of 6,423), Manga109 reviewed 246; sized on the subs before the ruby quarter |
