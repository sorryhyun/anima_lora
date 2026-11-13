# cjk_ko

Hangul rows on KO dialogue: `proposal_jamo.md` § 4 (phase 2; the proposal,
its phase-1 configs, probes and reports moved here from the reseed line on
10-14) of the reseed line (`../cjk_anima_reseed/`), opened 2026-10-11 on the `cjk-ko` branch. The
start is `jamo_j256_ko`'s merged rows: seed_1008's JA rows, the 256 syllables
J256_ko trained, and the other 10 916 at their jamo composition
(`reports/jamo_j256_ko_2026_10_11.md`: H exact 8 / 32, jamo F1 0.46).

The line owns its configs, corpus and reads; the builder and trainer are the
reseed line's `reseed/` package, run through `run.py`. Outputs:
`output/cjk_ko/<run>/`.

## Code

| file | what |
|---|---|
| `corpus.py` | CPU: songys Chatbot_data (MIT, `~/ko_corpus/songys_chatbot/`) → `~/ko_corpus/derived/`: `chatbot_lines.tsv` (the line file, `lines = "ko_chat"`), `chatbot_held.tsv` (the ruler's 96 strings, 32 each short 2–4 / mid 5–9 / long 10–20 syllables), `chatbot_rows.txt` (the line file's syllables by frequency). Normalisation and the hold-out rule in its docstring |
| `run.py` | front door: `run.py <run> data [--frac f] \| train` → the reseed line's `run.py` on `configs/<run>.toml` |
| `configs/ko_chat_warm.toml` | warm A: every syllable of the lines free, warm from `cjk_anima_reseed/jamo_j256_ko`, sent_kanji_pres's recipe (lr 2e-4, free_residual 0, 40 steps a row, L_pres λ 10 at σ 0.8–0.9 every 2nd step) |

| `configs/ko_micro_{a,b,bp}.toml` | the micro arms (user 10-11): the 192 most frequent syllables, 20 steps a row (3 840), no L_pres, on the lines of those 192 (`corpus.py --rows 192` → `ko_chat_192`). A = free rows; B = the jamo factors only, warm from j256_ko's (`factor_from`); B′ = B + a per-syllable residual (`factor_residual`). One data build (`ko_micro_a`), B / B′ `data_from` it |
| `configs/jamo_j1024_ko.toml` | the J line to 1024 (user 10-11): jamo_j256_ko's rows + the next 768 chatbot syllables by frequency (KS X 1001, H out; `corpus.py --jamo 1024` → `assets/jamo1024.json`) composed from the jamo factors, plus the 51 compat jamo as free rows (`Jamo`'s own rows); jamo_j256_ko's recipe and data cold on seed_1008, 56 a row (60 200 steps) |
| `configs/jamo_j256_ko_g44.toml` | jamo_j256_ko with the 16 px tiers cut (user 10-12): lone_16 / grid_16 5 % each, grid_44 31.7 %, the rest as built; same rows, recipe, steps |
| `configs/jamo_j256_ko_part.toml` | jamo_j256_ko + the 48 partial forms (user 10-12): the medial vowels ᅡ … ᅵ and finals ᆨ … ᇂ drawn with the initial left out (`jamo.PART_SPELL`, Noto Serif CJK), composed without the absent parts (ᅥ = b + V[ㅓ]); the opt-in `partial` tier at 15 %, J256_ko's mix at 85 %, 56 a row (17 024 steps) |
| `configs/jamo_j256_ko_cf.toml` | jamo_j256_ko + counterfactual syllables (user 10-12; `../cjk_anima_reseed/reseed/counterfactual.py`): each trained syllable less its initial (언 → ㅓ ㄴ, b + V + F) or its vowel (언 → ㅇ ㄴ, b + C + F), the 287 forms on hosts; less the final = the CV syllable as a row (41, H out); the counterfactual tier 25 %, steps held at 14 336 (`steps`) |
| `configs/jamo_j1024_ko_cf.toml` | jamo_j1024_ko + jamo_j256_ko_cf's counterfactual forms at 25 % of the items — 15 % on the flat canvas (`counterfactual`) and 10 % in a scene bubble (user 10-13: `cf_bubble1_52` / `cf_bubble1_32` at 5.5 / 4.5, bubble1_52 / bubble1_32's bands and px_keep, Noto Serif CJK) — steps held at 60 200; `faces.legible` on (user 10-12, `../cjk_anima_reseed/reseed/legible.py`): from 28 font px a face is re-picked for a glyph whose strokes it fuses (median gap < 1.5 px or p10 < 0.8 px) — ~7 % of the 28–40 px draws, the 16 px tiers untouched. Not run yet |
| `configs/ko_warm_cf_{a,b,bp}.toml` | the micro arms on `jamo_j1024_ko_cf` (user 10-13, `plan_warm_cf.md`): ko_micro_{a,b,bp} with `rows_from` / `factor_from` moved to `cjk_ko/jamo_j1024_ko_cf`, on ko_micro_a's data (the 10-11 mix, plan § 3 option 1) |
| `configs/ko_mix2_cf_{a,b,bp}.toml` | the cf micro arms on plan § 3 option 2's mix (user 10-13): bubble1 20 / bubbleN 40 / sentko 40, steps held at 3 840; `ko_mix2_cf_a` builds, B / B′ `data_from` it |
| `configs/ko_mix3_cf_bp.toml` | B′ alone on bubble1 40 / bubbleN 30 / sentko 30 (10-14, after the mix2 read); its own build |
| `configs/ko_x5_cf_bp.toml` | ko_mix3_cf_bp on × 5 rows (user 10-14): the 960 most frequent chatbot syllables (`corpus.py --rows 960` → `assets/micro960.json`, `ko_chat_960`: 15 269 lines), 20 a row (19 200 steps); read with `micro_read.py --sets 960` (held 92) |
| `configs/jamo_{j64,f64,j96,j128,j64_112,lone,j64_lone,d128,j256,j128_ko,j256_ko}.toml` | the jamo line's phase 1 (moved from the reseed line 10-14; `proposal_jamo.md`): `jamo_j64` / `jamo_f64` (factor vs free rows, one data dir), `jamo_j96` / `jamo_j128` (the factor arm on the larger sets), `jamo_j64_112` (J64 at J128's total steps), `jamo_lone` / `jamo_j64_lone` (the jamo curriculum), `jamo_d128`, `jamo_j256`, `jamo_j128_ko` (J128's data redrawn: `faces`, `pose`), `jamo_j256_ko` (J256 on that build). Their outputs of record stay in `output/cjk_anima_reseed/<run>/`; a re-run from here writes `output/cjk_ko/<run>/` |
| `probes/jamo_sets.py` | CPU: `proposal_jamo`'s syllable sets (J64 ⊂ J96 ⊂ J128 by jamo-cell coverage, H held out) and the Hangul font check → `../cjk_anima_reseed/assets/jamo_sets.json` |
| `probes/jamo_read.py` | `proposal_jamo` phase 1's reads: `fit` (CPU, R3: the jamo model fitted to F64's free rows, LOO cos; writes the regressed arm `jamo_f64_reg`), `render [--held a,b]` (GPU: each syllable alone in a KO bubble, the trained 64 and H, or H only under the named arms) → `results/<ts>-jamo-read/` |
| `probes/jamo_vl.py` | GPU: the jamo arms read by VL — `cal` (font-drawn lone glyphs: the reader's ceiling) / `score [--groups H,words]` (the cached renders); v4 (`OcrReader`) free read by default, `--readers v4,stock` / `--forced` opt-in; the vision tower runs once per crop (`VisionCache`) → `results/<ts>-jamo-vl-<mode>/` (`reports/jamo_curriculum_2026_10_11.md` § 6) |
| `probes/jamo_vl_sheet.py` | CPU: a `jamo_vl.py score` run glyph by glyph — one row per target, one column per arm, the render with v4's whole-image read under it (green exact, amber one jamo shared) + `reads.md`; per arm the glyph and positioned-jamo P / R / F1 of what was drawn against the target, and the jamo F1 against the other targets (chance) → `results/<ts>-jamo-vl-sheet/` (`reports/jamo_vl_reads_2026_10_11.md`) |
| `probes/ko_reader_cal.py` | GPU: `vl` on font-drawn lone Hangul through the ruler's crop path, per syllable / face / px → `results/<ts>-ko-reader-cal/` |
| `probes/hangul_init.py` | GPU (daemon): a ridge from PE-Spatial-B16 4×4 glyph features (seed_1008's katakana + kanji rows) predicts every trained syllable's row; read against j256_ko / j256_ko_cf / j1024_ko (rank, same-layout rank, the runs' own agreement as ceiling), least-squares onto b + C + V + F and compared with the trained factors → `output/cjk_ko/hangul_init/pe_spatial.pt` for `factor_init = { glyph, radius }` (`reports/hangul_init_2026_10_15.md`) |
| `probes/micro_read.py` | GPU: `render` (each read item alone in a KO bubble, seed 0, under `floor` = jamo_j256_ko and the micro arms; cached under `output/cjk_ko/micro_read/render/`) and `score` (v4 free read; per arm × group exact and glyph / jamo P R F1 with chance) → `results/<ts>-micro-read/`. Groups (`assets/micro192.json`): every third of the 192 (64), the next 64 by frequency (in no line), the 27 held strings on the 192; `--seeds` reads several renders an item |
| `probes/native_read.py` | GPU: the arms off the bubble (user 10-14) — `render` (48 strings of 1–5 syllables on the 192, each hung as a KO clause off one of the eight `NATIVE_PROMPTS`, seed 0, 512², plus the 8 EN references `English text reads as "hi"`, delta off; cached under `output/cjk_ko/native_read/render/`) and `score` (v4 free read as micro_read, per arm × length; `en_cos` / `en_cos_out` / `box_iou` against the EN reference) → `results/<ts>-native-read/` |

```bash
.venv/bin/python project/cjk_ko/corpus.py
.venv/bin/python project/cjk_ko/run.py ko_chat_warm data --frac 0.03   # sizes: sheet_<tier>.png
.venv/bin/python project/cjk_ko/run.py ko_chat_warm data
make daemon-run ARGS="project/cjk_ko/run.py ko_chat_warm train"
.venv/bin/python -m pytest project/cjk_ko/tests
```

## What the reseed package does for a KO run

Added 10-11 (`reseed/config.py` docstring, `table.py`, `recipes.py`); a JA
run draws what it drew.

- `rows_from = "<line>/<run>"` warms from `output/<line>/<run>/trained.pt`.
- A config outside the reseed line writes to `output/<its line>/<run>/`.
- `lines = "ko_chat"` (`LINE_LANG`: korean) gives a `lang` run its windows
  and `sent` lines (no JA dataset text joins them) and stamps
  `glyph_route_ko` in `build.json`, so the trainer routes Hangul tokens per
  syllable.
- `sentko_34` / `sentko_22` (opt-in): a KO line in 1–4 horizontal lines
  broken at its spaces, at sent_34 / sent_22's font px and bands; lengths in
  cells, spaces not counted; 70 % of the lines left-aligned, the rest
  centred (`SENT_H_LEFT`). A KO window (`bubbleN`) is a whole word of the
  lines (`word_pool`: 2–6 syllables, edge `?!…` dropped), never a piece of
  one, drawn horizontal on any scene. Neither caption says `horizontal`.
- A line's spaces pass the encoding check (no T5 row; the routed stream
  drops them); its letters count Hangul syllables.
- L_pres's EN caption names `Korean text` / `korean text` English too.

## Reads

`findings.md`: the J line's verdict so far — better data and counterfactual
syllables move H, tier reshuffles and × 4 scale do not.

- `reports/x5_native_2026_10_14.md`: × 5 rows (`ko_x5_cf_bp`) and the
  native read — the best held strings of any arm (glyph F1 .24 → .45 on the
  192, .18 → .37 on the 960; exact 0), lone syllables at the start's level
  or under (960 trained 14 / 64 = the start's), H 4 / 32 as mix3. In the
  scene prompts x5 letters near-miss strings (잊어야하나? → 얼어야아나?;
  3- / 5-syllable glyph F1 .47 / .49) but draws a line for a lone syllable
  (exact 0 / 48); the scene holds on every arm (en_cos_out .93–.94).

- `reports/mix2_2026_10_14.md`: the cf micro arms on bubble1 20 / bubbleN
  40 / sentko 40 — B / B′ lose less of the start (trained 0–1 → 6–7, next
  1–3 → 5–8), B′ leads held (glyph F1 0.30, jamo F1 0.57 over chance 0.31,
  exact 0 / 27); A unmoved. The line habit still on; § 4 not met. § 4 of
  it: B′ on bubble1 40 / bubbleN 30 / sentko 30 (`ko_mix3_cf_bp`) holds the
  start — trained 21 / 64 (glyph F1 0.35), next 16, held glyph F1 0.24.
- `reports/warm_cf_2026_10_13.md`: the micro arms on jamo_j1024_ko_cf (the
  10-11 mix) — held strings no better than 10-11 (B′_cf glyph F1 0.26 vs
  ko_micro_bp 0.25, exact 0 / 27 on every arm); the line habit back (trained
  21 → A_cf 11, B / B′ 0–1 / 64); B / B′ take next 22 → 1–3. A_cf keeps more
  of its start than 10-11 A did (11 vs 1 / 64). The mix comes first.
- `reports/micro_2026_10_11.md`: the micro arms A / B / B′ — every arm
  learned to draw a line for a lone syllable (exact 18 → 1–2 / 64 against
  j256_ko); B′ leads on the held strings (glyph F1 0.27 vs A 0.18); B / B′
  carry the line habit to untrained syllables. The data mix comes first.
- `reports/jamo_j1024_ko_2026_10_12.md`: the J line to 1024 syllables + the
  51 jamo — trained rows read as J256's own (new rows exact 5 → 17 / 61), the
  jamo draw (~30 / 51, as jamo_lone alone); H flat (8 → 7 / 32): scale buys
  trained rows, not composition. A better start for the dialogue warm.
- `reports/jamo_j256_ko_g44_2026_10_12.md`: the 16 px tiers cut into
  grid_44 — no gain; H and trained fall (H exact 8 → 6 / 32, glyph F1 0.31 →
  0.22). Size and σ band move together in the table, so it does not say
  which carried the loss.
- `reports/jamo_j256_ko_cf_2026_10_12.md`: counterfactual syllables (one
  jamo left out, 25 % of the items) — the first J arm to move H (exact 8 →
  10 / 32, glyph F1 0.31 → 0.35, jamo F1 0.46 → 0.52), next 8 → 13 / 64, the
  gain on syllables J256 did not train (13 → 20 / 93); held strings fall
  (glyph F1 0.17 → 0.09). Seeds 0–3 keep the direction, smaller: H 18 → 26
  / 128, next 17 → 32 / 256 (H short of significance). Forms vs the tier
  itself not separated.

## Open

- **The line habit** (10-14, `reports/x5_native_2026_10_14.md`): every
  dialogue arm draws a line for a short item, in a bubble or a scene; x5
  made it no better. Option 3 (+ the counterfactual tier) still needs the
  `Jamo.merge_all` residual fix.
- **jamo_j1024_ko_cf, ready for a RunPod 5090** (10-12; no 5090 in stock that
  night, not launched). Config and `faces.legible` committed; a 3 % build
  checked locally (3 274 items, 1 718 rows; the bubble1_32 sheet draws no
  fused glyph). On the pod, as J256_ko: the tracked code (`git ls-files -z`)
  plus 1.24 GB of inputs over 8 `rsync -aL --mkpath` shards —
  `../cjk_anima_reseed/assets/fonts/`, `output/cjk_anima_scale/scenes_{s1,
  s1s,s1w,sl1w,ja_comic}/`, `output/cjk_anima_scale/experiments/
  scene_{colorful,bubble_check}.json`, `models/vocab_packs/
  anima_cjk_vocab_pack_punct.{json,safetensors}`,
  `output/cjk_anima_reseed/seed_1008/trained.pt` — and the
  `output/wake_probe/scenes_{s1,s1w,sl1w,ja_comic}` links with plain
  `rsync -a` (the scenes' `scenes.jsonl` paths go through them). Then
  `run.py jamo_j1024_ko_cf data --workers 12 && run.py jamo_j1024_ko_cf
  train`; pull back `trained.pt`, `train_log.json`, `train_record.json`,
  `eval_coverage.json`, the data's manifests and `sheet_*.png`. Estimate:
  ~110 k items, ~10 min build, train ~3.7 h at J256_ko's 4.5 it/s (local:
  J1024_ko took 419.8 min at 2.39 it/s), ~4–4.5 h and ~$5 on SECURE. The
  cf masks need `uharfbuzz` + `freetype-py` (not in pyproject; a `uv sync`
  drops them): `uv pip install uharfbuzz freetype-py` on the pod before
  `data`. Read against jamo_j1024_ko: cf (flat and bubble) and the legible
  re-pick move together.
- **The read.** No KO ruler yet: the 96 held strings are cut out of the data
  now; the renderer and v4 per-glyph read are still to build (the JA ruler's
  shape, `criteria.md`). Also H (the jamo line's 32 held syllables) — they
  are all in the chatbot rows, so warm trains them directly.
- **Mode B**: the factors warm with a per-syllable residual (`proposal_jamo`
  § 4), so syllables outside the lines move too.
- **hitocrawl**: 133 KO editions (3 569 pages) to OCR into a second line
  file — manga register, licence unclear.
- The T5 stream holds no space between KO words (the routed encoder drops
  it); only the Qwen text keeps the word breaks.
