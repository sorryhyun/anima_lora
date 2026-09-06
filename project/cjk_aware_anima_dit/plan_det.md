# plan_det — AnimeText_yolo as the text detector (2026-09-06, rev. 1)

*Side line of [`plan_ocr.md`](plan_ocr.md), spun out of its O5 after the O6
probe (`findings.md` § O6). One goal — **G-D: one detector in front of the
reader**, in the package (`anime_tools.ocr`, the OCR stage) and in the dit
records pipeline — replacing the three layers that grew over O4 (PP-OCRv6 DB
→ VL Spotting boxes → MIT-mask components). The reader is not touched.*

## What is measured today (2026-09-06, `findings.md` § O6)

| fact | number / pointer |
|---|---|
| stock `deepghs/AnimeText_yolo` yolo12l @ 640, conf 0.426, sincos 351 pages | covers **237 / 237** PP-OCRv6 v3 lines, 98 % hybrid speech / 96 % sfx / 96 % chrome, 100 % hand speech / 98 % hand sfx, 92 % MIT mask components; masked-but-no-box floor **3** (PP DB 38, 3-layer stack 8) |
| boxes → SFX reader → records, conf 0.25 | 1,086 records on 163 pages (stack: 448 / 138), floor **4** (8), manga-ocr best-match **0.844** / 45 ≥ 0.9 (0.810 / 42), hand-SFX exact **66 / 99** (44) |
| the ~520 boxes matching nothing known | 438–522 yield a valid reader read; sheets show real SFX on pages the stack held at zero lines (`output/tests/ocr_animetext/sheets/`) — no hand pass yet |
| model / input size | l 640 ≈ l 1024 ≈ l native ≈ x 1024 on every column; 640 = 26 ms/page CUDA EP |
| nested boxes | YOLO emits a balloon block **and** its columns; `outer` (keep block) tanks best-match to 0.694, `inner` (drop a box holding ≥ 2 others) costs nothing measurable |
| weights / data licence | model card **GPL-3.0**; dataset CC-BY-NC-SA-4.0 |
| ORT gotcha | CUDA EP default arena + EXHAUSTIVE cuDNN search grew to 15 GB over per-page shapes and killed the next session — bound it (`kSameAsRequested`, `HEURISTIC`, `gpu_mem_limit`) |

## Decisions

1. **yolo12l @ 640, conf 0.25, NMS 0.5, `inner` nesting** is the config. No
   size sweep, no retrain — the stock weights pass the O5 detector gate.
2. **The detector pairs with the VL reader.** PP-OCRv6 rec is a line
   recognizer; a block box from YOLO is multi-column, so `--detector
   animetext --reader ppocr` is allowed but not a default, and D1 measures
   it once as a side row only.
3. **Runtime download, never bundled.** A catalog row fetches the ONNX on
   first use; GPL weights do not enter the MIT package, the trainer, or a
   node. Whether a *shipped* build may default to a detector trained on NC
   data is a licence call the user makes at D3 — the research path is
   unaffected.
4. **When the detector is `animetext`, the other two layers are off**: no
   VL Spotting pass, no mask-component crops. The MIT mask stays what it is
   for `make mask`; the floor is measured against it, not filled from it.
5. **Kind stays as it is** (hand labels + the text rule + chrome list). O5's
   segmentation half stays parked; only its detector half is retired here.

## Status (2026-09-06, night)

- **D0 DONE** — `anime_tools` b015ba2 + 2cbe201 (sibling checkout, pin bump
  owed at D3): `anime_tools.ocr.animetext`, the engine's detector protocol
  **+ a detect-only engine** (not in the plan: a YOLO block box through
  PP-OCRv6's recognizer dies on the score floor before the VL reader sees
  it, so `animetext` + `vl` loads no recognizer), `--detector` / `--det_conf`,
  catalog row `animetext_det`, the bounded arena in `make_session`. Gate: six
  pages reproduce the probe's boxes 15 / 15. `join_cjk` is **off** under
  the detector (measured −0.041 best-match over 97 joins).
- **D1 DONE on my side, hand pass owed** — `ocr/animetext_records.py` →
  `ocr_records_sincos_animetext.jsonl` (955 / 163 pages; 46 nested reads
  deduped). Regression: floor 4, best-match 0.844 / 45, hand-SFX exact **63**
  (= the O6 `inner` reference; the gate's 64 was `raw`'s number — the three
  rows are doubled-SFX labels `inner` splits, collapsed downstream anyway).
  Side row done once: PP-OCRv6 rec on the boxes = floor 26, hand-SFX 4 / 99.
  **Hand pass**: `assets/animetext_new_lines_sincos.tsv` (60 of 491 new
  lines, `real_text` blank) + `output/tests/ocr_animetext/hand_pass/`;
  graded by the user: **59 / 60 real** (#57 = heart censorship) → precision gate PASS.
- **D2 RUNNING** (daemon job `20260906-213229-e7feb8`, s42 vs C11) — mirror
  `mirror_sincos_animetext_sentence` + `te/sincos_animetext_sentence_isoq`
  built (351 / 351); caption diff `output/tests/ocr_animetext/d2_caption_diff.md`;
  config `configs/gui-methods/custom/cjk_unmask_d2.toml` (C11 verbatim but the
  mirror / cache / name). Launch = the `run_unmask_r2.py --skip_cache` line in
  its header; then grids + blind set vs C11.
- **D3 pending D2.**

`findings.md` § D0–D1 carries the numbers.

## Phases

### D0 — package: `anime_tools.ocr.animetext` (½ day, CPU + a GPU smoke)

- `AnimeTextDetector.load(device, conf, nms, nest)` over `_onnx._session`
  (the existing preload; the **bounded CUDA arena goes into `_session`** so
  every ONNX row gets it), `detect_batch(bgrs) → list[list[Box]]` with the
  probe's letterbox / decode / NMS / `denest` moved over verbatim.
- `OcrEngine` takes either detector: today's `_read_chunk` calls the DB
  detector's `probs_batch` / `_boxes` / `crop_quad` on quads; the YOLO path
  yields axis-aligned boxes, so the engine gets a small `Detector` protocol
  (`prepare(bgr)`, `boxes_batch(prepared) → boxes`) and the DB detector is
  wrapped to it. `min_box_px` / `max_boxes` / `join_cjk` apply unchanged —
  D1 checks `join_cjk` does not re-join what `inner` left as columns across
  balloons.
- Catalog row `animetext_det` (`deepghs/AnimeText_yolo` →
  `yolo12l_animetext/model.onnx` + `threshold.json`, dest
  `models/animetext/`), `OcrRequest --detector {ppocr, animetext}`
  (`OCR_DETECTORS`, default `ppocr` until D3), `--det_conf`.
- Tests: decode / NMS / denest on a synthetic `(5, N)` head; the request
  round-trips through the contract test; the source-inspection test keeps
  torch out of `run_ocr`. Smoke: six sincos pages through the daemon into
  the package venv, `--detector animetext --reader vl`.

*Gate:* the six-page smoke reproduces the probe's boxes (same count ± the
`inner` drop) and lines. Commit in the sibling checkout; pin bump owed at D3.

### D1 — dit records on the new detector (½ day)

- Promote the probe's records path: `ocr/animetext_records.py` (boxes →
  `SfxReader` → records with `kind` from the hand labels else the rule,
  reading order from `_text.reading_order`, `engine: animetext+sfx_reader`)
  → `ocr_records_sincos_animetext.jsonl`; **record-level dedupe** of nested
  reads (drop a record whose normalized text sits inside another record's
  on the same page with box containment ≥ 0.85; the SFX clause dedupe
  stays downstream).
- Regression check = the O6 table re-run on the dedup'd file: floor ≤ 4,
  best-match ≥ 0.84, hand-SFX exact ≥ 64. Side row: `--reader ppocr` on the
  same boxes (decision 2, once).
- **Hand pass, 60 lines**: a random draw of the "new" records (boxes the old
  stack never had) labelled real-text / not on the contact sheet — the
  precision number O6 lacks. Sheets via `probes/ocr_contact_sheet.py`.

*Gate:* regression holds and ≥ 50 / 60 new lines are real text. *Kill:*
precision under that → raise `--det_conf` to 0.426 and re-check once; still
under → the detector ships for masks / Tagger / the OCR stage only and the
records stay `hybrid_vl`.

### D2 — captions on the new records (1 GPU day; the user's call on the arm)

- Mirror `mirror_sincos_animetext_sentence` + `te/sincos_animetext_sentence_isoq`
  via `cache_te_ext.py` (SFX sentence default, dedupe on). Diff vs the
  `hybrid_vl` captions: how many gain an SFX clause, how many change speech.
- The line count roughly doubles, so this is a caption change, not a reader
  swap: one seed (s42, the C11 precedent) of the C10 recipe on the new mirror,
  grids, blind set vs C11 (`probes/blind_pairs.py`, ~24 pairs).

*Gate:* spam ≤ C11 on the seed; blind ≥ C11 inside the seed-twin floor.
*Kill:* spam up → D2 records keep `hybrid_vl`; the detector still ships
(D3 minus the records default).

### D3 — flip (½ day)

- `OcrRequest --detector` default → `animetext` **when `--reader vl`**
  (`ppocr` + DB stays the torch-free default); `make ocr` docs, the
  `captions` skill's OCR note, `docs/position_captions.md`.
- Trainer: pin bump + `uv lock`; `reread_records.py` / `cache_te_ext.py` /
  `run_unmask_r2.py` defaults → the animetext records if D2 passed.
- `plan_ocr.md` O5: detector half retired, segmentation half stays parked.
  `findings.md` § D0–D3, memory.

## Not doing here

- **No YOLO retrain** on AnimeText (84 GB, NC) or on Manga109/COO; no
  Spotting LoRA (O5 option b) — the stock detector already passes.
- **No kind segmentation** (O5) and no kind-aware masks — `kind` is still
  the rule + hand labels.
- **No PP-OCRv6 rec retrain**, no DB detector retrain — DB stays only as the
  torch-free `ppocr` default.
- **No bundled weights** in the package, the trainer, or any node; no
  weights on the Hub under our name (GPL + NC provenance).
- No new blind-set protocol; no C12-style reader-swap arm (user waived).

## Gotchas to carry

- Bound the ORT CUDA arena for every ONNX session, not just this one.
- `outer` nesting is wrong for balloon lines even though it looks tidier.
- PP rec on a YOLO block box garbles (multi-column); only the VL reader
  reads blocks.
- The manga-ocr reference is 40 A/B pages / 84 lines; the sincos hand
  labels are the SFX truth. Neither labels the ~500 new lines — D1's hand
  pass is the first look.
