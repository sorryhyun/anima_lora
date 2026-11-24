# ocr_reader — PaddleOCR-VL manga reader (JA + KO + ZH), main / sub

PaddleOCR-VL 1.6 fine-tuned on manga crops, shipped as Hub `sorryhyun/paddleocr-vl-1.6-manga-lora`
and read by `anime_tools.ocr.reader`. The record before main / sub (v4–v6, the two-prompt reader)
is in `_archive/ocr_reader/history/` (local; `README.md` there). In git, the same files are at
`73050123` under `project/finished/ocr_reader/` and `docs/proposal/`.

## State

- Shipped: v7 = `mainsub_1124_0151`, Hub `v7/` (`59768fc6`), the default reader of `anime_tools`
  v0.14.0 under `OCR:` only. `OCR2:` is not read there (the user, 2026-11-24: "일단은 ocr만
  쓰는식으로 하자"); v6 and `v7_preview` are its opt-ins.
- Now: the user reviews labels on the local page (§ Rules); exports are applied as they come.
- Definitions and the dataset: [`main-sub-approach.md`](main-sub-approach.md). Open questions,
  pending work and the report index: [`mainsub_plan.md`](mainsub_plan.md).

## Versions

| Hub dir | run | `anime_tools` pin | change | against the previous | report |
|---|---|---|---|---|---|
| `v3/` | `vl16_b2_norm4` | — | JA only: tower + projector full, LM r16 LoRA | sincos SFX strict 289 / 597, ♡-blind 336 | `finished/cjk_aware_anima_dit/findings.md` |
| `v4/` | `vl16_p2b_kozh_ep2` | 0.7.12 | tower r64 LoRA + LM full FT, gelnote KO/ZH pseudo rows, 2 epochs | sincos SFX strict 289 → 346; KO 1 → 26 / 48 | `history/reports/p3_kozh_ep2.md` |
| `v5/` (`fb08a10b`) | `vl16_p2b_kozh_nl` | 0.9.5 | line breaks kept (`TARGET_NORM` 5), crops trimmed at the page edge | sincos ♡-blind 386 → 414 (z +3.0); K3 KO 26 → 22 (z −1.4) | `history/README.md` § v5 |
| `v6/` (`57e7c650`) | `vl16_p2b_kozh_nl_swaprelax` | 0.9.8 | + `pseudo_swaprelax_v5` (20,745 KO speech lines) | K3 KO 22 → 28 (z +2.1); JA flat | `history/README.md` § v6 |
| `v7_preview/` | `sfxpc_1118_0412` | opt-in | two prompts, `OCR:` speech / `SFX:` SFX | — | `history/proposals/ocr_sfx_prefix.md` |
| `v7/` (`59768fc6`) | `mainsub_1124_0151` | 0.14.0 | main / sub (`OCR:` / `OCR2:`), detector boxes at 4 px | vs v6: doujin SFX strict 294 → 334 (+95 / −55); Manga109 test SFX 85.2 → 80.8 % (+83 / −196), speech 88.4 → 89.0 %; K3 KO 28 → 30, ZH 30 → 33 | [`mainsub_1124_0151.md`](reports/mainsub_1124_0151.md) |

K3 figures in the table are on the labels before the 2026-11-24 review (`assets/k3_gelnote.pre_review.tsv`).
On the reviewed labels (`review_1124b`: 53 edits, mostly line breaks, 91 rows → `checked`; 48 KO + 48 ZH
scored), `eval_k3.py --compare` rescores v6 → v7 as KO 26 → 29 (+5 / −2), ZH 31 → 35 (+6 / −2).
`spaced` now counts line breaks on ZH (a break is a space there and ZH labels had none), so ZH
`spaced` is not comparable across the review.

Report paths under `history/` are in `_archive/ocr_reader/` (local). Sincos figures are on the 597-row gate below; v5 is paired against v4 rescored on the rebuilt crops.

## Layout

| path | holds |
|---|---|
| `core/` | the library every script imports (package `project.ocr_reader.core`): `paths.py` (repo and dataset paths, `ANIMA_*_ROOT` overrides), `manga109.py` (corpus access), `crop_dataset.py` (`load_mainsub`, the training set) + `augment.py`, `textnorm.py` (targets and keys), `readers.py` (`READERS`, `Vl16Reader` with its `task` prompt), `scoring.py` (`score` / `summary` / `exact_key` / `normalize_ja`), `crops.py` (`deskew_crop`, `margin_box`, ink projection), `labels.py` (`ellipsis_fold`, `line_broken`, `agent_subs`), `regions.py` (main / sub `Item`), `det_types.py` (detector-box `classify`), `hayai_beam.py`, `vl16_compile.py` |
| `ocr/` | training and eval scripts: `finetune_vl16_lora.py`, `build_sfx_prefix.py` (the detector crops and their manifests), `build_manga109_crops.py` (the grey `manifest.parquet` the others start from), `eval_sfx.py` / `eval_k3.py` / `eval_manga109.py`, `rescore_eval.py` |
| `labeling/` | the review page and the label pipeline: `review_page.py` (builds the page), `apply_review.py` (applies an export), `m109_mainsub.py` (Manga109 main / sub proposals), `furigana_find.py`, `linebreak_auto.py`, `hayai_read.py`, `ko_read.py` (reader flags for the KO rows) |
| `probes/` | main / sub measurements: `eval_mainsub.py`, `eval_subneg.py`, `sfxonly_suspect.py`, `mainsub_text.py`, `sfxp_crop_sheet.py` (sheets of the `core.det_types` types), `build_hf_mainsub.py` (the local HF dataset build) |
| `gate/` | builders of the frozen gate sets: `build_eval_data.py`, `k3_gelnote.py` |
| `assets/` | label TSVs; local except `k3_gelnote.tsv`, `kozh_sfx.tsv`, `sfx_labels_sincos_597.tsv` |
| `eval_data/` | frozen gate images (local) |
| `reports/` | written reports (`mainsub_*`, `hayai_*`, `manga_error_review.md`); `reports/eval/` holds the eval scripts' generated tables (local) |
| `tests/` | `textnorm`, `deskew_crop` |

Names that outlived their meaning:

- `sfxp` / `sfxpc` come from the two-prompt reader. The main / sub set still lives in its
  manifests and crop trees on disk (`manifest_sfxp_det_all.parquet`, `crops_sfxp/`), built by
  `ocr/build_sfx_prefix.py`; `sfxpc_*` runs are two-prompt runs.
- The review page keeps the name `ko_label_review` wherever the browser state hangs on it: the page
  folder `~/ko_label_review/`, the localStorage key `ko_label_review_v1` and the export
  `~/Downloads/ko_label_review.tsv`. Renaming any of them loses the saved edits.

## The gate — sincos 597

This line scores sincos with
**`--labels project/ocr_reader/assets/sfx_labels_sincos_597.tsv`** (933 rows,
597 scored SFX). It is the finished line's 617-row file minus the 8 pages that
were deliberately removed from the dataset (2026-10-10). Every arm, v3
included, is scored on it. Never mix it with the `/ 617` figures in
`finished/cjk_aware_anima_dit/eval.md`. v3 on this basis: strict 289, ♡-blind 336.

Metrics (since 2026-11-24, the user: heart-blind retired, line-break-blind added): `exact`
(`textnorm.exact_key`, whitespace-blind) and `lb` (`textnorm.linebreak_blind`: the same glyphs,
the same 띄어쓰기, a line break on either side matching a space or none) in every eval;
`eval_k3` keeps `spaced` (a break counts as a space). The ♡-blind figures in the Versions table
predate this. v7 on sincos: exact 334, lb 334.

```
ANIMA_MANGA109S_ROOT=/media/sorryhyun/new/dataset/manga109s/Manga109s_released_2026_05_21 make daemon-run ARGS="--stall-timeout 900 \
  project/ocr_reader/ocr/eval_sfx.py --reader vl16 --ckpt output/ocr/<run>/ep1 \
  --name <run>_597 --labels project/ocr_reader/assets/sfx_labels_sincos_597.tsv"
```

## Gate images — `eval_data/` (gitignored)

Labels stay in the tracked `assets/` TSVs. The images every gate reads are
frozen under `project/ocr_reader/eval_data/` by `gate/build_eval_data.py`:

- `sincos/crops/<row>.png` + `index.tsv`: the 933 label rows cut at pad 0.12,
  the same cut `eval_sfx.py` makes. `sincos/pages/` holds the 155 pages behind
  them. `eval_sfx.py` reads the frozen crops first (at pad 0.12), then the
  frozen pages; the gate does not depend on `post_image_dataset/`.
- `k3_gelnote/crops/` + `pages/`: the 100 K3 rows of `assets/k3_gelnote.tsv`.

Manga109-s (corpus + `derived/` manifests and crops) lives at
`/media/sorryhyun/new/dataset/manga109s/`. `~/manga109s` is a symlink to it,
so older commands still resolve.

## Inherited — read before any arm

- Baseline ledger (rescored key, on 617 rows): `finished/cjk_aware_anima_dit/findings.md`
  § "The reader ledger". On a sincos-sized set the binomial SE is about 12 lines, so a gap
  smaller than that is not a finding.
- The "Do not re-propose" list in that same `findings.md` still applies. Most
  relevant here:
  - Never screen or size a KO/ZH pool with B′.
  - Never label from `animetext-ocr` (those reads come from hayai, and they
    carry no 띄어쓰기).
  - Never judge a KO/ZH arm on the JA gates alone.
  - Never strip whitespace from training targets.
  - Check the tokenizer before choosing any target spelling.
- Licence: AnimeText is CC-BY-NC-SA. A shipped reader must be built without
  it, and without any pseudo rows derived from it.
- Code: shared code goes in `core/`; a script imports `project.ocr_reader.core.*` and never another
  script. Scripts run by path (`make daemon-run ARGS="project/ocr_reader/ocr/eval_sfx.py …"`).
  The Ovis / PP-OCRv6 / package readers, the pseudo-labeller, the page-detector and hito evals
  are in `_archive/ocr_reader/code/` (local; in git before the 2026-11-24 split). `eval_sfx.py`
  defaults to the 597 labels; every eval script writes its table to `reports/eval/`. Every GPU
  step goes through the daemon.
- Hardware: RTX 5070 Ti 16 GB. No bitsandbytes on cu132.

## Rules — the label review loop (v7, main / sub)

### The page

- `~/ko_label_review/index.html` is a local page (the user's call: not an Artifact). Its state is
  in the browser's localStorage; Export writes `~/Downloads/ko_label_review.tsv`. Tab in a main
  box marks the row `ok` as the box stands and moves to the next row.
- The export holds changed rows only, against the assets and manifests at build time: rebuild the
  page after each apply.
- A fix made outside the page is undone by the browser's saved edit at the next export. Drop the
  row from a later export if it carries the old text again.
- From every export, filter `3261138_279400` and the stale `軍神か……` edit
  (`Nekodama_007_0004b534` → `…`).
- A reviewed row exports its sub proposal untouched, so a wrong ruby read becomes a reviewed
  positive.
- `ok` or an edit confirms both boxes as they stand, the proposed sub included. Emptying the main
  box makes a sub-only row.
- `mainsub_nolabel` lists the detector boxes with no label (no annotation inside, or an SFX cut
  by the box), the main box holding hayai v2.7's read as a first pass. hayai answers on every
  box: a crop with no text is emptied by the reviewer. Its exports go to `m109_mainsub.tsv`;
  the loader does not train them yet.
- A `mainsub_*` row shows hayai's read (one string, no line breaks); the `hayai disagrees` sort
  lists the rows whose main it does not match, largest distance first.
- A `mainsub_*` row in `assets/m109_mainsub.tsv` (the user's or an agent pass's) shows the applied
  main and sub and starts reviewed; hayai's distance is to them, and an applied drop leaves the
  hayai sorts. Before 2026-11-24 the page showed the annotation's proposal on these rows. The
  page shows no proposal text.
- The `has` filter lists rows by what the boxes hold (main only, sub only, both, partly visible
  text). The default `suspicion` sort puts `trim`, then partly visible text, first.
- A row holding one `<text>` box with ruby starts from its furigana row's sub edits.
- Apply writes `mainsub_*` rows to `assets/m109_mainsub.tsv` and other sources' subs to
  `assets/sub_labels.tsv`. `k3_ko` / `k3_zh` are the K3 gate's rows: apply writes them back to
  `assets/k3_gelnote.tsv` (`ok` / edit → `checked`, drop → `skip`); the K3 figures move with them. The manifests are not touched by `mainsub_*` rows.

### After each export

1. Copy it to `assets/`, run `labeling/apply_review.py <tsv>` with `ANIMA_MANGA109S_ROOT`
   exported (earlier edits show as "stale": already applied). The user deletes the Downloads
   file. Name the copy `review_<MMDD><letter>.tsv`; once applied it may be deleted (the exports up
   to 11-22 are in `_archive/ocr_reader/assets_applied_exports/`, local). The gate / mark / sub /
   mainsub TSVs stay.
2. Check the crops of a small export and report suspected typos with paste-ready text. Fix only
   on the user's word.
3. Retrain only when asked; export `ANIMA_MANGA109S_ROOT` at submit.
4. Rebuild the page:

   ```
   python project/ocr_reader/labeling/review_page.py --out ~/ko_label_review \
       --flags output/ocr/eval/ko_reads_v7_flags.parquet \
       --doujin_reads output/ocr/eval/sfx_mainsub_1124_0151_cap4_ocr.jsonl \
       --k3_reads output/ocr/eval/k3_mainsub_1124_0151.jsonl \
       --ruby output/ocr/furigana/ruby_all_read_u3p12.jsonl --ruby_fix output/ocr/furigana/agent \
       --mainsub output/ocr/mainsub/m109_mainsub.jsonl --hayai output/ocr/hayai/hayai27_det.jsonl \
       --hayai_kozh output/ocr/hayai/hayai27_kozh_hito.jsonl
   ```

   The `--flags`, `--doujin_reads` and `--k3_reads` files are v7's (`ko_read.py`, `eval_sfx.py`,
   `eval_k3.py` on `mainsub_1124_0151`); a new reader swaps them.

   Rebuilding the Manga109 proposals (after a change to the annotations, the ruby reads or
   `m109_mainsub.py`):

   ```
   python project/ocr_reader/ocr/build_sfx_prefix.py --det_all
   python project/ocr_reader/labeling/m109_mainsub.py \
       --out output/ocr/mainsub/m109_mainsub.jsonl \
       --ruby output/ocr/furigana/ruby_m109_read_u3p12.jsonl --ruby_fix output/ocr/furigana/agent
   ```

### Labels (the user's rules)

- What a label holds (the crop decides, empty labels, spelling): `main-sub-approach.md`
  § Definitions.
- Merged blocks go.
- `swaprelax_v5` was audited by agents and the user read every gelnote KO row: neither set is
  "unreviewed".
- hayai-ocr v2.7 Supernova reads are allowed as first-pass labels (`labeling/hayai_read.py`). Its
  card lists AnimeText-derived training sets and an Apache-2.0 licence for the weights; the user,
  2026-11-24: "hayai는 아파치 라이센스라고 하니 뭐 라이센스 이슈는 괜찮을듯 문제가 생기면 hayai가
  알아서 하겠지". The rules under "Inherited" on `animetext-ocr` labels and AnimeText rows stand
  for those sources themselves.
- Which sources give `OCR2:` negatives is the user's call (`main-sub-approach.md` § Sources).
- Mains waiting on a line-break proposal stay out of training; agents do not fill them.
- Rows over the training cap (main > 150 normalized characters) are hidden on the page and
  still export.

### Traps

- A 2-epoch run runs out of memory on 16 GB without `--ckpt_lm_only` or `--grad_ckpt`.
- Eval `exact` is whitespace-stripped, so line breaks never move it.
- There is no KO val split; K3's 48 KO rows are the only KO gate.
- Parallel agents sharing one scratchpad overwrote each other's helper scripts and outputs: each
  agent gets a private output directory and no shared script.
- Agents split sub lines by label line or by printed column inconsistently.
