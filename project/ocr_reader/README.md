# ocr_reader — PaddleOCR-VL manga reader (JA + KO + ZH), main / sub

PaddleOCR-VL 1.6 fine-tuned on manga crops (Hub `sorryhyun/paddleocr-vl-1.6-manga-lora`, read by
`anime_tools.ocr.reader`). This file holds the main / sub approach, how labels are made and how
text is normalized. Open work: [`mainsub_plan.md`](mainsub_plan.md). Versions, gates, reports and
earlier versions of this file: `_archive/ocr_reader/` (local).

## main / sub

### The task

- Input: one crop, an AnimeText detector box cut with a 4 px margin. The reader reads the
  detector's output, so the detector box is the unit, not the annotation's polygon or box.
- `OCR:` reads the crop's main text, `OCR2:` its sub text. Either can be speech or SFX: the prompt
  does not carry the kind, the kind of a line comes from the text (`line_kind`).

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
  display).
- A label that leaves out visible text teaches the reader to skip text, so a sub is complete or
  it is not trained (§ What trains).

### The crop decides

- The label follows the crop as shown, not the page and not the annotation.
- Text cut by the crop edge is typed as far as it shows. What only the page shows (the rest of a
  cut line, a neighbouring balloon outside the box) is not typed.
- A mark that cannot be told apart from drawing (a stroke cut by the edge, a symbol in the art)
  is not typed; a crop whose only other marks are of that kind has an empty sub.

### Empty labels

- Empty main: a crop with only sub text. It trains as `OCR:` `""` only when reviewed as such.
- Empty sub: the crop has no sub. It trains as `OCR2:` `""` only where that is known (§ Sources,
  the negatives). Anywhere else an empty sub means "not known" and gives no `OCR2:` row.

### Examples

| crop | shows | main | sub | from |
|---|---|---|---|---|
| `UchuKigekiM774_020_24` | a title, two balloons cut by the edge | `第4幕 宇宙さそりの逆襲 前編` | the cut speech as far as it shows (`もうよ` / `なよ` / `体に毒` …) | reviewed |
| `LoveHina_vol01_051_29` | `いてっ`; the next balloon's `…` is on the page only | `いてっ` | empty | reviewed |
| `JijiBabaFight_045_17` | the SFX `わはは` over a balloon | `わはは` | `もうけ` / `もうけ` | proposal |
| `AkkeraKanjinchou_051_11` | speech with ruby and a small SFX | `変なの…` | `へん` / `ヒソ` | proposal |
| `GarakutayaManta_006_34` | an English paragraph and a large `ひ～` | the paragraph | `ひ～` | reviewed (overrides glyph size) |

### Sources

| source | split | `OCR:` | `OCR2:` |
|---|---|---|---|
| Manga109 detector crops (`mainsub_det`: `manifest_sfxp_det_all`, labels `output/ocr/mainsub/m109_mainsub[_val].jsonl`) | train, val | main | sub; `""` on a row reviewed `ok` with an empty sub, and on an SFX-only crop (one SFX, no other annotated text whole or partly inside) |
| KO / ZH (`manifest_sfxp_kozh`) | train | the label | `assets/sub_labels.tsv`, else `assets/kozh_sfx.tsv`; `""` on a `swaprelax_v5` crop with neither |
| Anima low-confidence reads (`anima_lowconf`: `assets/anima_mainsub.tsv`, crops under `ANIMA_LOWCONF_ROOT`) | train | the reviewed main (`""` = no text) | the reviewed sub; `""` when empty, with the reviewed Manga109 negatives |

Not in the set: the COO polygons the detector misses, the `<text>`-box crops
(`manifest_sfxp_speech`) and the older 12 % crops (`manifest.parquet`). The `<text>` labels
still label the speech inside the detector crops.

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
  `m109_mainsub.jsonl`), one fixed quarter trains (`RUBY_RAW_SHARE`, picked by a hash of the id,
  the same for every seed); the other three quarters never train.
- A target over 150 normalized characters is dropped, main and sub each on their own.
- Val reads `main` / `main_neg` / `sub`; the best checkpoint is picked by `main_exact`.

## Labeling

### Proposals

Manga109 rows are proposed by `labeling/m109_mainsub.py` from the annotations. An agent label
(`assets/m109_agent_subs.tsv`: source, id, verdict, main, sub, pass) replaces the proposal, and a
reviewed row (`assets/m109_mainsub.tsv`: source, id, status, main, sub) replaces both.

- An annotated item (COO SFX polygon or `<text>` box) is in the crop at 0.8 of its convex hull
  or more, partly visible from 0.05 to 0.8.
- main and sub follow § main / sub over the items in the crop; glyph size is
  `sqrt(hull area / glyphs)`.
- sub (`subp`): per `<text>` box, the furigana first (the agents' fix where they reviewed it, else
  the ruby read), then the other items in the crop.
- With no item whole inside, a speech box at least 0.3 inside becomes the main, cut to the
  columns and glyphs that show (`trim`).
- An unreviewed main takes the line breaks agents read off its crop (`assets/m109_breaks.tsv`);
  its characters stay the proposal's.
- The annotations do not cover unannotated text (signs, titles, asides outside a box) or smaller
  lettering inside one `<text>` box; only review adds those.

Rebuilding the proposals (after a change to the annotations, the ruby reads or `m109_mainsub.py`):

```
python project/ocr_reader/ocr/build_sfx_prefix.py --det_all
python project/ocr_reader/labeling/m109_mainsub.py \
    --out output/ocr/mainsub/m109_mainsub.jsonl \
    --ruby output/ocr/furigana/ruby_m109_read_u3p12.jsonl --ruby_fix output/ocr/furigana/agent
```

### Label sources

- hayai-ocr v2.7 Supernova reads are allowed as first-pass labels (`labeling/hayai_read.py`;
  Apache-2.0 weights, the user's call).
- Never label from `animetext-ocr`, and never train on AnimeText rows or pseudo rows derived from
  them (CC-BY-NC-SA; a shipped reader is built without them).
- `swaprelax_v5` was audited by agents and the user read every gelnote KO row: neither set is
  "unreviewed".
- Which sources give `OCR2:` negatives is the user's call (§ Sources).
- Mains waiting on a line-break proposal stay out of training; agents do not fill them.
- Merged blocks go.

### The review page

- `~/ko_label_review/index.html`, built by `labeling/review_page.py`, is a local page (not an
  Artifact). Its state is in the browser's localStorage; Export writes
  `~/Downloads/ko_label_review.tsv`. The `ko_label_review` name stays on the folder, the
  localStorage key `ko_label_review_v1` and the export: renaming any of them loses the saved edits.
- Tab in a main box marks the row `ok` as the box stands and moves to the next row. `ok` or an
  edit confirms both boxes as they stand, the proposed sub included (a wrong ruby read becomes a
  reviewed positive). Emptying the main box makes a sub-only row.
- The page lists only sources that train. Rows over the training cap (main > 150 normalized
  characters) are hidden and still export.
- The export holds changed rows only, against the assets and manifests at build time: rebuild the
  page after each apply. A fix made outside the page is undone by the browser's saved edit at the
  next export; drop the row from a later export if it carries the old text again.
- From every export, filter `3261138_279400` and the stale `軍神か……` edit
  (`Nekodama_007_0004b534` → `…`).
- A `mainsub_*` row shows hayai's read (one string, no line breaks); the `hayai disagrees` sort
  lists the rows whose main it does not match, largest distance first. A row already in
  `assets/m109_mainsub.tsv` shows the applied main and sub and starts reviewed. The page shows no
  proposal text.
- The `has` filter lists rows by what the boxes hold (main only, sub only, both, partly visible
  text). The default `suspicion` sort puts `trim`, then partly visible text, first.
- A row holding one `<text>` box with ruby starts from its furigana row's sub edits.
- Where apply writes: `mainsub_*` → `assets/m109_mainsub.tsv`; other sources' subs →
  `assets/sub_labels.tsv`; `k3_ko` / `k3_zh` (the K3 gate) → `assets/k3_gelnote.tsv` (`ok` / edit
  → `checked`, drop → `skip`); `anima_lowconf` → `assets/anima_mainsub.tsv`; `reader_split` →
  `assets/reader_split_mainsub.tsv` (not trained yet). The manifests are not touched.
- `anima_lowconf`: the v7 reads of the Anima training set at score <= 0.85, the main box holding
  the read; `sincos/` is left out (the doujin gate's source). Crops and `rows.jsonl` under
  `ANIMA_LOWCONF_ROOT` (`/media/sorryhyun/new/dataset/anima_ocr_lowconf`).
- `reader_split`: cjk_rl kana render crops that one reader (hayai v2.7 or the tuned reader) reads
  as the asked glyph and the other does not, both at confidence 0.9 or over
  (`--reader_split_min_conf`); the main box holds the read that matched; the `ks` sort puts the
  hayai-only crops first, grouped by asked glyph.

### After each export

1. Copy it to `assets/review_<MMDD><letter>.tsv`, run `labeling/apply_review.py <tsv>` with
   `ANIMA_MANGA109S_ROOT` exported (earlier edits show as "stale": already applied), then move it
   to `_archive/ocr_reader/assets_applied_exports/`. The user deletes the Downloads file.
2. Check the crops of a small export and report suspected typos with paste-ready text. Fix only
   on the user's word.
3. Retrain only when asked.
4. Rebuild the page:

   ```
   python project/ocr_reader/labeling/review_page.py --out ~/ko_label_review \
       --flags output/ocr/eval/ko_reads_v7_flags.parquet \
       --doujin_reads output/ocr/eval/sfx_mainsub_1124_0151_cap4_ocr.jsonl \
       --k3_reads output/ocr/eval/k3_mainsub_1124_0151.jsonl \
       --ruby output/ocr/furigana/ruby_all_read_u3p12.jsonl --ruby_fix output/ocr/furigana/agent \
       --mainsub output/ocr/mainsub/m109_mainsub.jsonl --hayai output/ocr/hayai/hayai27_det.jsonl \
       --hayai_kozh output/ocr/hayai/hayai27_kozh_hito.jsonl --anima_lowconf --reader_split
   ```

   The `--flags`, `--doujin_reads` and `--k3_reads` files are v7's reads; a new reader swaps them.

### Agent passes

- Three Sonnet agents per batch of ~32 rows run `labeling/label_agent.md`, each in its own
  `output/ocr/<source>/agents/b<NN>_a<k>/` folder (`batch.tsv` in, `out.tsv` out) and with no
  shared script: agents sharing a scratchpad overwrote each other's files.
- The page puts the majority main in the box where two agree and lists every agent's read; the
  `agents unsure` sort puts the non-unanimous rows first, Hangul first.
- `labeling/anima_agents_apply.py` writes every row where two agents agree with status `agent`;
  split rows stay on the page. A touched `agent` row exports as the user's `ok` / edit.
- Agents miss hand-lettered Korean SFX most, and split sub lines by label line or by printed
  column inconsistently.

## Normalization

One fold (`core/textnorm.py`, `fold_glyphs`) feeds the training target, the scoring keys and
`anime_tools`' read form, so the reader never learns a spelling the pipeline discards. The current
version is `TARGET_NORM = 13`.

- Training target (`training_target`): keeps line breaks as `\n` and keeps `♥` (filled) and `♡`
  (outline) apart; `♡` and `💢` are LM tokens of their own (`core/vocab_ext.py`).
- Scoring key (`exact_key`): the same folds plus every heart → `♡`, every dot run → `…`,
  whitespace deleted. `linebreak_blind` also accepts a break against a space or none.

Decided folds, in order:

1. Spacing dakuten `゛` / `゜` → the combining marks, before NFKC.
2. NFKC (fullwidth `！？．～` → ASCII, halfwidth kana composed, `‼` → `!!`).
3. A lone Hangul conjoining jamo → the compatibility jamo (`ㅋㅋ`, not `ᄏᄏ`).
4. Symbols: `❤` → `♥` (key: every heart → `♡`); `〜` / `〰` → `~`; `─` / `—` → `―`; curly
   quotes → `"`; `◯` / `〇` → `○`; `☓✕✖╳✗✘` → `×`. Latin `O` / `x` / `X` stay.
5. In a string holding Hangul, every dash → `-`.
6. Dots: a `・` run → `…` per three dots, at least one (`・・・・・` → `……`); a lone `・` stays.
   `...` / `…` / `……` stay as labelled in the target; `.` runs stay (Korean keeps `..` / `...`
   apart from `…`). Japanese speech labels take `ellipsis_fold` (`core/labels.py`).
7. Drawn-out lines: outside a Hangul string, a run of `ー` / `―` / hyphens (any mix, any count) →
   one `ー`, Chinese included; a `~` run → one `~`. A hyphen run touching an ASCII letter or digit
   stays one `-` (`747-400D`); in a Hangul string a `ー` / `-` run is one `-` and a lone `ー`
   stays. The user: the long mark is what the speech lines train a drawn-out sound as, and one
   spelling is worth more than Chinese dash compatibility.

Spelling rules on the label side: Japanese lettering left under a Korean label stays Japanese;
never strip whitespace from a target; check the tokenizer before choosing a target spelling.
A change of fold bumps `TARGET_NORM` and is mirrored in `anime_tools.ocr.reader`.
