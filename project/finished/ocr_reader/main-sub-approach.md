# ocr_reader — the main / sub policy (`OCR:` / `OCR2:`)

What is main, what is sub and which labels train. It follows v7's two prompts (`OCR:` speech /
`SFX:` sound effects, [`docs/proposal/ocr_sfx_prefix.md`](../../../docs/proposal/ocr_sfx_prefix.md)).
Opened 2026-11-21 on the user's call. Finished work is in the reports listed at the end.

## Facts

The rules are the user's calls of 2026-11-21 and 22, except where a line names the code that
approximates them.

### The split

- `OCR:` reads the main text, `OCR2:` the sub text of the same crop. The pair replaces the
  speech / SFX split: either can be speech or SFX, and the kind of a line comes from the read
  text (`line_kind`), not from the prompt.
- main = the largest text in the crop.
- sub = everything else visible in the crop, one item per line, in reading order: furigana,
  smaller speech or SFX, hand-lettered asides, smaller lettering that the annotation joins
  into the same `<text>` box, text cut by the crop edge.
- Ignoring visible text is the harm.

### What a label may hold

- The crop decides. The label follows the 4 px training crop as shown, not the page and not
  the annotation.
- Cut text is typed as far as it shows. What the crop does not show is left empty
  ("크롭에서 안보이면 비우는게 맞을듯"): no text or dots that only the page shows.
- Crops are the boxes AnimeText gives (`mainsub_det`). Out: the COO polygons it misses
  (`manifest_sfxp_missed`, which were there for the `SFX:` prompt), the `<text>`-box crops
  (`manifest_sfxp_speech`) and the 12 % crops (`manifest.parquet`, the page's `m109_sfx`
  reference rows). The `<text>` texts still label the speech inside the detector crops.
- A main's line breaks are the printed lines: one per column of vertical text (right to
  left), one per row of horizontal text.
- Korean keeps `...` and `…` apart. Japanese speech takes `ellipsis_fold`.
- Japanese lettering left under a Korean label stays Japanese.

### Examples

Reviewed training crops (`assets/m109_mainsub.tsv`, local):

| crop | shows | main | sub |
|---|---|---|---|
| `UchuKigekiM774_020_24` | a title, two balloons cut by the edge | `第4幕 宇宙さそりの逆襲 前編` | the cut speech as far as it shows (`もうよ` / `なよ` / `体に毒` …) |
| `LoveHina_vol01_051_29` | `いてっ`; the next balloon's `…` is on the page only | `いてっ` | empty |

Crops from before the detector crops became the only source (they show the split, they do
not train):

| crop | main | sub |
|---|---|---|
| `Nekodama_13_013_25` | the large `!` | the small speech |
| `PLANET7_3_003_15` | `0!!` | `ゼロ ズバン! !!` |
| `BakuretsuKungFuGirl_038_0000a4fc` | the speech | the ruby over 冷凍 and the smaller hand-lettered `ふざけてんの？` |
| `GakuenNoise_048_0001a044` | `死にさらせっっ!!` (the label) | `ス` (an SFX cut by the edge, as it shows) |

### An empty label

- An empty sub means "not known", never "this crop has none". It gives no `OCR2:` row,
  whatever its source or review state. `OCR2:` negatives exist only once the user sets them
  up.
- An empty main trains as `OCR:` `""` only when the row was reviewed as such (a crop with
  only sub text). An unreviewed empty main is out: the first run read text on all 59 val ones.

### What trains

`crop_dataset.load_mainsub`, per crop:

| state | `OCR:` | `OCR2:` |
|---|---|---|
| reviewed `drop` | out | out |
| reviewed | the main as reviewed | the sub as reviewed, if not empty |
| unreviewed, main empty | out | out |
| unreviewed `trim` (main only partly inside) | out | out |
| unreviewed, line breaks waiting on the page (train only) | out | as the two rows below |
| unreviewed, partly visible text in the crop | the proposed main | out |
| unreviewed, other | the proposed main | the proposed sub, if not empty |
| KO / ZH (train only) | the label | `sub_labels.tsv`, else `kozh_sfx.tsv`, if not empty |

An unreviewed main takes the line breaks agents read off its crop, where it has them
(`assets/m109_breaks.tsv`, `with_breaks`); the characters stay the proposal's.

Any target over 150 normalized characters is dropped, main and sub each on its own.
`OCR:` negatives are at most 20 % (`--neg_share`). Val metrics are `main` / `main_neg` / `sub`
(best by `main_exact`).

### How the proposal approximates the split

`probes/m109_mainsub.py`, from the annotations. A reviewed row overrides it.

- An annotated item is in the crop at 0.8 of its hull or more, partly visible from 0.05.
- main = the in-crop item with the largest glyph (`sqrt(hull area / glyphs)`), joined with
  same-kind items of at least 0.7× that glyph and with repeats of the same text at any size.
  Several are written one per line, right to left, then top to bottom.
- sub = the ruby reads first (the agents' fix where they reviewed the box), then the other
  in-crop items. Partly visible text is only a hint on the page.
- With nothing whole inside, a speech box at least half inside becomes the main, cut to the
  columns and glyphs that show (`trim`).
- It cannot see unannotated text (signs, titles, asides outside a box) or smaller lettering
  inside one `<text>` box. The reviewer adds those.

### Review

- `ok` or an edit confirms both boxes as they stand, the proposed sub included.
- Emptying the main box makes a sub-only row.
- The `has` filter lists rows by what the boxes hold (main only, sub only, both, partly
  visible text). The default `suspicion` sort puts `trim`, then partly visible text, first.
- A row holding one `<text>` box with ruby starts from its furigana row's sub edits.
- The export holds what differs from the applied state. Rebuild the page after each apply.
- Apply writes `mainsub_*` rows to `assets/m109_mainsub.tsv` (source, id, status, main, sub)
  and other sources' subs to `assets/sub_labels.tsv`. The manifests are not touched by
  `mainsub_*` rows.

```
python project/finished/ocr_reader/probes/apply_ko_review.py <export.tsv>
python project/finished/ocr_reader/probes/ko_label_review.py --out ~/ko_label_review \
    --flags <ko_reads>_flags.parquet --doujin_reads <sfx>_cap4_sfx.jsonl \
    --ruby output/ocr/furigana/ruby_all_read_u3p12.jsonl --ruby_fix output/ocr/furigana/agent \
    --mainsub output/ocr/mainsub/m109_mainsub.jsonl
```

Both need `ANIMA_MANGA109S_ROOT`. Rebuilding the proposals:

```
python project/finished/ocr_reader/ocr/build_sfx_prefix.py --det_all
python project/finished/ocr_reader/probes/m109_mainsub.py \
    --out output/ocr/mainsub/m109_mainsub.jsonl \
    --ruby output/ocr/furigana/ruby_m109_read_u3p12.jsonl --ruby_fix output/ocr/furigana/agent
```

## Open

Policy, for the user:

- "Largest" when a dense block meets a few large glyphs. `GarakutayaManta_006_34`: the
  proposal's main was `ひ～` (the larger glyphs), the user's review has the English paragraph
  as main and `ひ～` as sub. That main is over the cap, so only the sub trains.
- How "no sub" is marked, for the `OCR2:` negatives.
- When a sub counts as complete. The loader's rule is a first cut: most `OCR2:` positives
  are unreviewed ruby reads (about half right in a 14-row read, the data report).
- `drop` boxes (an SFX cut partway, 2,678 train) and `neg_other` boxes (no annotation inside,
  22,718 train) stay out; under main / sub a cut SFX could be sub.
- How a main / sub read is scored. The v7 gates read per kind (COO SFX / speech, doujin SFX,
  K3); the val subs are unreviewed ruby reads.

Work:

- 141 speech mains held by the line-break agents (15+ characters: 76 label and crop differ, 9
  with unclear columns; 9–14: 49 and 7; `output/ocr/linebreak/agent/speech{15,9}/held.tsv`,
  local) still train as one line. Out of `OCR:` until reviewed, like the waiting rows:
  proposed, not decided.
- 19 mains of 9–14 characters print a mark their label lacks
  (`output/ocr/linebreak/agent/speech9/missing_marks.tsv`, local): not fixed.
- 3 labels type quote marks as `゛`, which `normalize_target` folds onto the kana (the
  line-break agents' report): not fixed.
- The 8,412 waiting mains come back row by row as they are reviewed. The agents do not fill
  them for now (the user, 2026-11-22: "b, c는 일단 학습에서 빠져있으니 괜찮을듯").
- Val never had the line-break pass outside the speech manifest. Proposed, not decided.

## Reports

- [`mainsub_data_2026_11_21.md`](reports/mainsub_data_2026_11_21.md): the detector crops, the
  proposals, the line-break and furigana passes, the agent reviews.
- [`mainsub_1121_1806.md`](reports/mainsub_1121_1806.md): the first run (val main 86.4 %, sub
  49.2 %) and its v7 gates.
- [`mainsub_labels_2026_11_22.md`](reports/mainsub_labels_2026_11_22.md): the loader changes
  of 11-22 and the count of missing line breaks.
- [`mainsub_linebreak_agents_2026_11_22.md`](reports/mainsub_linebreak_agents_2026_11_22.md):
  the agents' breaks on 15+ character speech (1,259 applied, 85 held).
- [`mainsub_linebreak_agents_9to14_2026_11_22.md`](reports/mainsub_linebreak_agents_9to14_2026_11_22.md):
  the same on 9–14 characters (1,199 applied, 56 held).
