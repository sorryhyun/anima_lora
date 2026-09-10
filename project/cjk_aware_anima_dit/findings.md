# CJK-aware Anima, DiT side — findings

Settled verdicts of this line, one entry per phase, evidence pointer beside
each. The encoder-side verdicts it builds on are in
[`../cjk_aware_anima/findings.md`](../cjk_aware_anima/findings.md) (read-only).

Line frozen 2026-09-08. The plans and the dated reports moved to
`_archive/cjk_aware_anima_dit/{plans,reports}/` (gitignored, private mirror);
every `reports/…` and `plan*.md` link in this file resolves there.
[`plan.md`](plan.md) is the freeze note — what shipped, what never ran, and
where the measured headroom is.

This file is a digest, deduped 2026-09-08. Full tables, per-orientation and
per-length breakdowns, worst-N rows and the raw predictions live in the
archived reports and in `output/ocr/eval/*.jsonl`; what stays here is the
verdict, the numbers a later decision turns on, and the mechanism notes that
exist nowhere else.

---

## Comparability — read this before comparing any two numbers

Three things moved under this line's numbers. Two of them make older figures
non-convertible, not merely shifted.

### 1. The sincos label basis changed (2026-09-06) — no conversion exists

`assets/sfx_labels_sincos.tsv` was re-based onto the AnimeText detector's boxes
(`plan_det.md` D1, `ocr/relabel_animetext.py`).

| basis | rows scored | SFX gate rows | speech / chrome |
|---|---|---|---|
| PP-box (retired) | 338 | 99 hand-typed, of which a 71-row subset was the gate number | 213 / 26 |
| AnimeText (current) | 949 | 617 | 293 / 39 |

The detector finds far more lines on the same 351 pages, so the row set
differs: **`n / 71` is not `n × 617/71`.** Any section below dated 2026-09-05 →
09-06 evening quotes the PP-box basis; it is marked where it appears.

The AnimeText basis itself moved twice under user label passes — 619 → 617 SFX
(three `＾＾＾` scribbles dropped, one blank row joined), speech 294 → 293,
chrome 35 → 39, `checked` 70 → 86. Every `/ 619` figure in the archived
reports and plans is pre-revision. The ledger below is on 617.

### 2. The eval key was re-based (2026-09-08 00:12, `acd41d72`)

`exact_key` gained the ellipsis fold (each dot run → one `…`), the eval half of
the O4e guard fix. Every row measured before that scores `・・・`, `...` and
`…` as three different reads — worth up to +495 lines on COO speech,
+37…+154 on COO SFX, +4…+9 on the sincos gate.

`ocr/rescore_eval.py` re-derives `exact` for every stored `output/ocr/eval/*.jsonl`
from its own `pred_norm` + `text` columns (CPU, no model re-run). The ledger
below is the output of that script — every row on one key and one basis. The
archived `reports/ocr_eval*.md` files were never regenerated (`--write` not
run), so a report file and this ledger will disagree by exactly the fold.

### 3. Harness and jitter floors

- The VL readers carry a ±1-line batching jitter at different `--bs`; a
  1-line move is not a result.
- The same weights read 312 (`eval_sfx.py`) and 316
  (`context_margin_sweep.py`, pad 0.12) on the same 617 rows with the same key.
  Compare rows produced by one harness.
- On either basis the sincos speech / chrome rows are not an accuracy
  metric: their `text_hand` is PP-OCRv6 record text on almost every row, so
  reading `♡` correctly scores as a miss (§ O3 label audit).

---

## The reader ledger (all rows on the current key, `ocr/rescore_eval.py`)

COO = Manga109-s test ∩ the official COO book split (2,558 SFX / 2,559 speech).
sincos = the doujin gate, 617 hand-labelled SFX rows. Published COO baseline:
TRBA+2D 81.2 % on the full 10-book test (ours is 6 of those 10 books).

| reader | COO SFX | COO speech | sincos SFX / 617 |
|---|---|---|---|
| stock manga-ocr | 28.9 % | 81.0 % | 6 (1.0 %) |
| stock PP-OCRv6 rec | 7.7 % | 16.1 %† | — |
| stock PaddleOCR-VL-1.6 | 31.6 % | 82.8 % | 19 (3.1 %) |
| stock HunyuanOCR-1.5, official zh prompt | 9.5 % | 37.7 % | 12 (1.9 %) |
| stock HunyuanOCR-1.5, ja prompt | 13.0 % | 43.3 % | 21 (3.4 %) |
| manga-ocr FT lr 2e-5 (arm A) | 73.6 % | 80.7 % | 108 (17.5 %) |
| manga-ocr FT lr 5e-5 (arm A, best) | 76.0 % | 80.1 % | 127 (20.6 %) |
| VL-1.6 LoRA, tower frozen (arm B) | 66.6 % | 86.2 % | 110 (17.8 %) |
| B′ `vl16_tower_lr1e-5` — the published reader | 83.2 % | 88.3 % | 312 (50.6 %) |
| B′ + col100 (1.6 % colorized append) | 84.9 % | 87.8 % | 323 (52.4 %) |
| B′ + col1500 swap (22.3 % colorized) | 85.5 % | 88.4 % | 281 (45.5 %) |
| B′ × 3 epochs | 87.3 % | 89.2 % | 306 (49.6 %) |
| B′ from LP-FT (arm B → B′ order) | 84.3 % | 87.6 % | 264 (42.8 %) |
| B′ from an SSL-adapted tower | 84.6 % | 88.2 % | 307 (49.8 %) |
| shipped `anime_tools.ocr.sfx` (B′ + decode guard) | — | — | 311 (50.4 %) |
| the pipeline's own records (`--reader record`) | — | — | 317 (51.4 %) |
| outside: hayai-ocr v2.1 | 78.2 % | 88.0 % | 180 (29.2 %) |
| outside: hayai-ocr v2.1.5 | 60.7 % | 85.5 % | 316 (51.2 %) |

† not a valid speech control — the speech crops are whole bubble boxes
(multi-column) and PP's single-line CTC head cannot read them without its own
detector in front.

What the ledger says as a whole:

1. The tower was the doujin gap. Every row under 130 on sincos has either a
   frozen tower or no fine-tune. Unfreezing it (B → B′) is the single largest
   move this line made: 110 → 312.
2. In-domain and the doujin gate decouple, in five arms running. col100,
   col1500-swap, ×3 epochs, LP-FT, tower SSL: each moved COO by +1…+4 points
   and left sincos flat or worse. The remaining headroom is on the label
   side, not the representation side — see § Tower for the ♡ evidence.
3. The gate's top is a ~10-line band at ~50 %, shared by B′ (312), the
   shipped guarded reader (311), the pipeline records (317), col100 (323) and
   an outside 150 M model (hayai v2.1.5, 316). Nothing in this line separated
   them; the binomial SE at n = 617 is 2.0 points ≈ 12 lines.
4. One unresolved inconsistency: stock VL-1.6's COO runaway count is
   recorded as 331 (§ O2's corrected table) and 91 (§ Outside readers · hayai's table) for
   what should be the same row. `runaway` is untouched by the rescore, so the
   fold does not explain it; the decode cap went 48 → 96 between the two, which
   is the only known difference. Neither number was ever re-derived. Do not
   quote a stock-VL runaway count without re-running it.

---

## D0 — ISO1 vs C9 direct blind set: flat (2026-09-05)

`reports/blind_s13_ISO1_vs_C9.md`: 48 pairs, 16 v2 rows × seeds 6/7/8, both
arms fresh to the grader. ISO1 23 – C9 20, tie 5; rows 6-6; p 0.76. The
isotropic table and the trained r256 pack are indistinguishable for unmask
training on this grid.

- The transitivity claim ISO1 ≈ HOT > C9 (s12 + s11) does not survive the
  direct test. Transitivity has now failed twice in this protocol (s03/s04,
  s12/s13). Do not chain blind sets — pair the arms you want to compare.
- Pooled s01–s13: a content-free table is never worse than the trained pack for
  the OCR route, and rows must exist (C9 > P). The isotropic block is the
  OCR-route default on cost grounds (seed-generated, deterministic, no
  distill), not on quality grounds.
- `k_norm` strips row scale on the K path, which is why HOT (norm ×5) ≈ ISO1
  (s12). Do not run norm / gain arms on the table.

Gate outcome: proceed to D1 with the isotropic block for 「…」 spans; bare
CJK tags keep the trained rows.

## OCR reader A/B — VL-1.6 is not a stock upgrade over PP-OCRv6 (2026-09-05)

`reports/0905_paddleocr_vl16_vs_ppocrv6.md`; probes `probes/ocr_vl16_ab.py`,
`probes/ocr_vl16_prompt_batch.py`. 40 sincos pages, VL read three ways (page
`Spotting:`, page `OCR:`, `OCR:` on PP's own quads), disputed lines checked
against the pixels.

Verdict as of that date: character accuracy ties; VL keeps ♡ / `ー` / small
kana (30 lines vs 4), finds ~2× the lines (260 vs 132), and rewrites toward the
likelier word (`狠狠地`→`狼狽地`, `おい`→`あい`). Prompt hints are noise —
settled, never reopen. Batching is the only wall lever.

This verdict was later superseded on the model axis, not the prompt axis: a
VL fine-tune with the tower unfrozen (§ O2b) beat PP-OCRv6 outright, and the
detector was replaced too (§ O6, § D0–D1). The prompt-engineering half stands.

## D1 — deterministic table + route partition + LoRA stamp (2026-09-05)

Gate: PASSED (sanity) — arm `C9ISOQ` (the C9 recipe re-cached through the
partitioned pack) renders inside the same-recipe seed-twin floor: 64-px L1 to
C9 0.075 ± 0.042 vs the twin's 0.087 ± 0.042. A mild grayscale tilt (mean
saturation 0.208 vs C9's 0.255, 9 vs 6 low-saturation images of 24) matches
another C9 variant and is not separable at n = 24. Not a blind set; the gate
only asked for "inside the floor".

Shipped (contract: `docs/experimental/cjk_ext_vocab_coverage.md` § "Quote
partition"; tests `tests/test_ext_vocab_iso.py`, 58 passing):

- `library/anima/ext_vocab.py` — `iso_block` / `IsoSpec` / `materialize_iso`
  (seed-regenerated, byte-equal across machines), `Route.quotes` +
  `quote_spans`, the span rule before `segment_runs`, `pack_digest`.
- `make_random_pack.py --mode iso | iso-partition`. Norm default = the native
  T5 mean row norm 212.165 (measured off `llm_adapter.embed`); ISO1 had used
  the trained mean 203.9. Pack built:
  `cjk_vocab_pack_synthjakozh1sym_r256_isoq` (sha `2cf81cbc…`).
- `ss_ext_pack_sha` / `ss_ext_pack` stamped by `train.py --ext_pack`;
  `load_dit_model` warns on a stamped LoRA with no pack; Adapter node 3.10.0
  compares digests in either node order and regenerates seed-only blocks.
  Node committed locally, never pushed / registry-published.
- Grammar: anime_tools `efb235c` — a comma or `. On the` inside `「」『』""` is
  content, `compose_caption` round-trips.

Design facts worth keeping: the mirror is a full row-for-row copy (one id map
serves both blocks; 285 MB fp32 shipped, or 0 bytes seed-only), quoted content
bypasses minted-word rows and the C fallback, and the rule is inert unless
both `iso` and `route.quotes` are present — every pre-existing pack, cache and
blind set is untouched.

## OCR eyeball + SFX handling (2026-09-05)

The eyeball that produced `plan_base1.md`. Four things it established:

- What C2–C9 actually trained on: `mirror_sincos_ppocr` was built from the
  v1 records; v2's reading-order rewrite never reached a trained arm.
- VL is an extra detector, not a manga-SFX reader. On a masked-no-line page
  it reads the one solid-fill SFX and misses six hand-lettered pink ones —
  identically at 896 px and at 3048 px, so resolution is not the limit.
- `datasets/ocr_sfx.py` (torch-free text rules: kanji / >6 kana /
  vowel-or-h-row initial → speech; repeated unit, voiced initial, lexicon onset,
  sokuon → SFX) and `--ocr_format sentence` in `cache_te_ext.py`. GOTCHA that
  drove B2: the anime_tools grammar had no header for these sentences, so a
  re-parse glued the first onto the last tag — the append had to be
  string-level until `TEXT_PREFIXES` landed.
- SAM3 `speech bubble` is not the speech/SFX signal (user's suggestion,
  measured): balloons on 34 / 97 pages, 69 of 228 lines inside one, and as a
  veto it moved one line. Outside-a-balloon is not SFX (narration,
  floating dialogue, chrome) — tried, added more errors than it fixed. SAM3 is
  out of this path unless a balloon soft prompt is trained (`--prompt_embed`).

Side gotcha, recurring: `build_mirror`'s symlink creation fails sporadically on
the ntfs3-mounted dataset volume — fall back to `os.link`, or retry once.

## B0 — hybrid OCR records on sincos: floor 38 → 23 (2026-09-05)

Full tables `reports/0905_b0_hybrid_records.md`. PP-OCRv6 v3 records + VL
`Spotting:` on all 351 pages + VL `OCR:` on every PP box, merged.

Heading corrected 2026-09-08: `44 → 27` was the first merge, on the v2 PP
file. The v3 space-join re-merge is what stands. The full floor chain for
sincos' 133 masked pages: PP v2 44 → PP v3 38 → hybrid 23 → 8
with the SFX reader + MIT-mask components (§ O4) → 4 with the AnimeText
detector (§ D0–D1).*

| | PP-OCRv6 v3 | hybrid |
|---|---|---|
| pages with any line (351) | 103 | 123 |
| lines | 237 | 338 |
| masked-but-no-line floor (133 masked) | 38 | 23 |
| best-match sim to manga-ocr (84 ref lines) | 0.751 (35 ≥ 0.9) | 0.786 (38) |

Gate PASS. What the merge had to do differently from the plan, all still
true of any two-detector merge:

- IoU 0.5 is too strict for columns — a 30 px vertical column boxed 12 px
  apart is IoU 0.42 with byte-identical text, because VL quads are per column
  while PP records are `join_cjk`-joined blocks. Match at IoU ≥ 0.3 or
  containment ≥ 0.5 or touching + text sim ≥ 0.75, after `join_cjk`-ing the
  VL side.
- Spotting hallucinates a page caption — two quads covering the whole page;
  gate on area (> 40 %).
- A second reader needs three guards, not one. Beyond the runaway / 2×-length
  test: never accept a read that loses a heart PP had (PP drops ♡, never invents
  it); a weak-score re-read must agree with the matched Spotting read; a
  symbol dispute may move symbols only. Under them 0 of 9 symbol disputes
  survived — the second reader's symbol job delivered nothing on sincos.
- Joined blocks keep their boundaries as a space (`JOIN_SEP = " "`,
  anime_tools `cd75591`): a space inside a vertical sentence costs a reader
  nothing, a lost list boundary is gone for good. With boundaries kept, the
  replaced lines move toward manga-ocr (0.485 → 0.512) instead of a wash —
  the second reader had been penalised for glued rows, not for its letters.
- SFX lines out of captions (user's call): the records keep `kind: sfx`, the
  mirror builder drops it, until a reader can read hand-lettered onomatopoeia.
  Reversed at § O4b once one could.

## B3 — arm C10: sentence captions pass the gate at the floor (2026-09-06)

`reports/0906_c10_sentence_captions.md`. Blind `s14_C10_vs_C9ISOQ`, 48 pairs,
fresh seeds: C10 21 – C9ISOQ 15, tie 12 — flat inside the s02 seed-twin
floor (p 0.41). Spam ~2 events on each of 3 training seeds; adherence recall
back to C9's level.

Gate PASS at the floor → the sentence shape is the default caption shape; no
blind-visible gain claimed. Grader caveat: the grader picked side B in 31/36
decisive pairs, a side habit shared by every arm since C8.

Instrument added here: `probes/grid_spam_tally.py`.

## O0 / O1 — the SFX reader line: split, crops, stock baselines (2026-09-06)

Split = the official COO `books_{train,val,test}` ∩ Manga109-s `books.txt` →
74 / 7 / 6 books (`assets/coo_split_manga109s.json`). No private split: a
by-name cut would make the held-out number incomparable to the COO leaderboard.
Crops: `ocr/build_manga109_crops.py`, pad 12 %, min side 16, orientation
preserved → 43,535 COO lines (train 38,582 / val 2,395 / test 2,558; 1,724
truncation joins) + a count-matched 43,589 speech crops, 3.7 GB under
`~/manga109s/derived/`. Never in-tree. Stock rows are in the ledger.

Facts that shaped everything after:

- Char coverage. SFX: 175 / 181 chars, 99.90 % of occurrences in
  manga-ocr's WordPiece vocab — no tokenizer surgery needed. Speech: 2,571 /
  2,689 chars but only 93.8 % of occurrences, and the misses are not
  glyphs — full-width punctuation, the ideographic space, and newlines (1,989
  rows; Manga109 `<text>` keeps line breaks). → NFKC-fold + strip all
  whitespace before tokenising, or a twentieth of the speech characters
  becomes `[UNK]`.
- 1 : 1 by count is 5.4 : 1 by characters (speech p50 11 vs SFX 3). If a
  speech control ever slips, weight by tokens, do not redraw.
- Joined truncation lines are near-unreadable stock (98 lines, manga-ocr 1
  exact). They are real COO test items; a fine-tune may want them weighted down.
- Augmentation (`ocr/augment.py`): pad jitter 5–25 %, ±8°, colour tint
  (pink/red/plum strokes over skin/pastel), gamma, invert, scale, JPEG. The tint
  pass is the cheap half of the domain lever — the colorized-COO lever (§ O3)
  owns backgrounds with real art.
- Daemon gotcha for every script on this line: only `ANIMA_`-prefixed env
  reaches a job, hence `ANIMA_MANGA109S_ROOT` / `ANIMA_ANIMETEXT_ROOT`.

### O1 correction — `deskew_crop` transposed every axis-aligned box

Found while standing up the sincos gate: stock manga-ocr read the hand-labelled
crops at 0 / 99, and the dumped crops were strips through one glyph. Cause:
OpenCV ≥ 4.5 `minAreaRect` reports an axis-aligned 30×120 box as size
(120, 30) at 90°; the pilot's `deskew_crop` took `angle − 90` without
swapping the extents, so every polygon whose reported angle was > 45° — all
Manga109 `<text>` boxes, all sincos record boxes, a large share of COO polygons
— was cropped as a transposed rectangle around the right centre. In the O1
manifest 88 % of speech boxes were taller than wide but only 17 % of the crops
were.

Fixed (swap `w, h` with the angle); all 87,124 crops rebuilt; every pre-fix
stock row, smoke run and first O2 launch was discarded. The pilot-era sincos
"~12 / 71" came through the same function and is not a clean reference either.

## O2 — the two fine-tune bases: in-domain PASS, doujin gate MISS on both

*Superseded the same day by § O2b. Sincos figures here are `/ 71`, PP-box basis.*

| arm | recipe | wall | COO test SFX (as measured then) | sincos / 71 (♡-blind) |
|---|---|---|---|---|
| A · manga-ocr lr 5e-5 | full FT, bs 64, 4 ep | 20 min | 73.5 % | 10 (12) |
| B · VL-1.6 LoRA lr 1e-4 | r 16 on 126 LM proj (6.0 M), tower frozen, 2 ep | 85 min | 64.7 % | 13 (19) |

Gate: COO reported ✓; sincos ≥ 35 / 71 ✗; both speech controls ✓.

- In-domain, fine-tuning works and manga-ocr wins it — 26 → 73.5 % in 20
  GPU-min, speech control untouched, no runaways, at 1/10 VL's wall.
- Out-of-domain, both bases miss. Arm A trips the kill clause literally
  (< 25 while COO ≥ 70 %). The residual is no longer garbage (99-row SFX sim
  0.31 → 0.67–0.70) but it is the doujin surface: `びくん` for `びく♡`, `ぱんッ`
  for `ぱん♡` — the heart decoded as the katakana ending COO taught — plus
  pink-outline confusions (`ぱ/は/ば`).
- 1 : 1 by count held both speech controls, so the 1 : 2 arm is not needed.

Engineering note that generalises: VL's first launch OOMed in the loss — the
native forward materialises fp32 logits over the 103k vocab for every image
token, 4.7 GB on a large-crop batch. Fixed with left-padding +
`logits_to_keep` = target length (CE on the suffix only; peak 13 → 3 GB).

## O2b — arm B′: the vision tower was the doujin gap (2026-09-06) — the pick

Same crops, mix, LoRA and lr as arm B, plus the NaViT tower + projector trained
in full (fp32 master, lr 1e-5, 439 M params), 1 epoch, ~90 min, 12.1 GB.
Reports `reports/ocr_eval{,_sfx}_vl16_tower_lr1e-5.md`; ledger row above.

All four O2 gate clauses hold → PASS, the first arm to pass the doujin gate,
without O3. (Clauses were read on the PP-box basis: sincos 38 / 71.)

- The domain gap was a tower problem, not a decoder-prior problem. Arm B
  moved the in-domain number and barely the doujin one; letting the tower see
  the crops does both in one epoch.
- Hearts are read natively — strict vs ♡-blind gap 3 lines. The planned
  heart-patching rule is moot, so decision 1 resolves to VL outright (it
  passes and removes the rule; the tie-break to manga-ocr never engages).
- Cost accepted: ~10× manga-ocr's wall per crop; deployment = torch + remote
  modeling files + adapter 24 MB + tower 878 MB; a runaway guard is
  mandatory before wiring (189 on COO test; the count is left unguarded in
  every table on purpose).
- Residual for a later lift: 8+-char lines, square multi-line SFX blocks, the
  `ぱん♡` family.

Published: `sorryhyun/paddleocr-vl-1.6-manga-lora` (adapter +
`tower.safetensors` + card with the two-step load — peft merge, then
`load_state_dict(strict=False)` of the tower — the runaway caveat, and the
Manga109-s / COO citations). Weights only, no crops. Re-verified against the
Hub 2026-09-07 (sha `3b5fe022`): it carries B′'s numbers, i.e.
`vl16_tower_lr1e-5`. Neither `ep3` nor `col100` was ever pushed.

## O4 — the reader wired in: `anime_tools.ocr.sfx`, floor 23 → 8 (2026-09-06)

*Sincos figures `/ 71`, PP-box basis.*

The package. `anime_tools.ocr.sfx.SfxReader` (rev 46ebbb5; `peft` became a
package dependency), B′'s weights from two catalog rows — `vl16_base` and
`sfx_reader` — fetched on first load. A crop reader only.

The decode guard is area-tied, not aspect-tied. The first guard capped a
read at `4 × longer/shorter + 6` characters: it held the SFX gate and silently
threw away 60 % of the speech reads (a multi-column balloon block is square
and holds 20 characters; sincos speech sim 0.910 → 0.454), and
`max_new_tokens = 32` truncated long lines (~1 token per CJK character; speech
runs to 57). Shipped: cap = crop area / (16 px)², floor 12; 80 new tokens; the
repetition test owns the runaways. `guard` runs at apply time on the cached
raw decode, so a guard change never costs a GPU pass.

Records. One GPU pass over 486 crops (338 hybrid records + 148 MIT-mask
components). `kind` now comes from the hand labels where one exists
(`kind_src: hand`), the v1 rule elsewhere.

| | PP v3 | hybrid (B0) | + SFX reader | `--reread all` |
|---|---|---|---|---|
| lines / pages | 237 / 103 | 338 / 123 | 448 / 138 | 448 / 138 |
| floor (133 masked) | 38 | 23 | 8 | 8 |
| best-match to manga-ocr (84 ref) | 0.751 (35) | 0.786 (38) | 0.800 (39) | 0.810 (42) |
| hand-SFX (99) exact / sim | — | 1 / 0.479 | 44 / 0.873 | 44 / 0.873 |

Mask components: 148 cropped → 110 added, real lettering on the sheet; 15
floor pages recovered. The 8 still empty have components under 32 px or reads
under the floor.

"Just run all of OCR through VL" (user) — measured, wins modestly.
`--reread all` replaces 298 of 338 records. The speech rows cannot be judged
on the hand labels (the incumbent scores 0.998 by construction), but on the
independent manga-ocr reference the all-VL file is the best of the four, and
the replacements read as fixes — hearts restored, `ムうムう` → `ムラムラ` — with
a regression tail (`バスト91` → `バスト9`) and the B0 space between joined
columns dropped (VL reads a block as one string). Kept single-variable for
C11; adopted as the records default at § O4c.

O5 parked (user's call). `ocr/kind_seg.py` — a `segmentation_models_pytorch`
U-Net over Manga109-s spreads, classes bg / speech / sfx, with box-level evals
and an ONNX export — is written and CPU-smoked, never trained. Resume = the
four commands in its module doc, ~1–2 GPU-h.

## O4b — arm C11 (SFX sentence) vs C10: flat both halves (2026-09-06)

One training seed (s42, user's call, not the three the plan asked for).
`reports/unmask_grid_judge_c11.md`.

- Spam ~2 = C10's ~2 — the same two base-habit cells. Gate half PASS.
- Adherence flat; the one recall gap is the `comic, 2koma` row that C9ISOQ
  also lost, at n = 3 renders.
- Blind `s15_C11_vs_C10`, 24 pairs: C11 11 – C10 9, tie 4 — inside the
  seed-twin floor (s02 15–9). Grader sides balanced, so the s14 side-bias flag
  does not apply. Gate half PASS as written.

`DROP_KINDS` flipped the same evening: `{chrome}` only, the SFX sentence is
the default, `--drop_sfx` reproduces the C2–C10 caption.

## O4c — SFX dedupe + all-VL records as the default (2026-09-06, night)

Two user calls off the caption sheet, no arm.

1. The SFX clause is deduplicated per sound unit. `ocr_sfx.dedupe_sfx` —
   `sfx_key` = the kana core minus sokuon / long-vowel marks, folded to its
   minimal repeating unit; first in reading order kept; speech never
   deduplicated here. `じゅぽ, じゅぽ, じゅぽじゅぽ` → `じゅぽ`. The key folds
   hearts, so the first read's decoration wins.
2. The all-VL re-read is the records default. 82 of 132 captions and 184
   lines differ from the SFX-only file: about half spacing / halfwidth
   punctuation / hearts, half real repairs, with a regression tail on digits and
   short lines. C11's config and mirror stay as trained; no C12 arm was run.

Package side the same night: the OCR stage gained `--reader {ppocr,vl}` +
`--mask_dir` / `--comp_min_side` / `--comp_max` / `--vl_batch_size`. Torch stays
out of `run_ocr` (the ONNX-device pin holds). Artefact to know: a PP box of
screentone that VL reads as `s v .l √2` passes the floors — the dit pipeline's
rule-1b / symbol filters are not in the stage.

## O4d — a caption says each line once; a box too small never gets there (2026-09-07)

Two user calls off `probes/ocr_merge_sheet.py`, no arm. Both in the package
(anime_tools `95e1a22`); the research builder mirrors only the first, so every
arm through C11 trained without these — a rebuilt cache is not the corpus they
saw.

1. Speech is deduplicated on exact text. A page of panting read as `はあ`
   seven times said it seven times. 141 of 2,167 det-passing speech lines
   (6.5 %) on 90 of 616 pages repeat a neighbour verbatim. Deliberately not
   `sfx_key`: that folds `はっ` with `はー` and `んっ♡` into `ん` (266 lines,
   12.3 %) — two different words of dialogue are two lines however alike they
   sound.
2. A glyph floor, not a box floor. `OcrLine.glyph_px` = `sqrt(w·h / len)` is
   the em of the line whichever way it runs; `DEFAULT_MIN_GLYPH` 16 px. Plain
   box area does not separate — `ドキ` at 17×42 (714 px², a good read) is
   smaller than an 858 px² shop sign read as four kana at 14.6 px a glyph.
   The em also catches correct reads of 13–15 px watermarks, credit lines and
   narration strips, which nothing trained at this resolution can render.
   Cost: 173 more lines dropped (4.7 %), 31 of 727 sidecars emptied. Floors
   of 14 (2.4 %) and 18 (9.6 %) were the alternatives; 16 is where the reads
   visibly stop being text (hand-checked across the 13–19 px bands).

## O4e — the guard was eating the longest line on the page (2026-09-08)

Three user picks off the merge sheet, no arm. Two mechanisms, one a bug in the
shipped guard.

The bug. The largest text on a page — a three-column balloon, a three-column
narration — was in neither sidecar although the detector had it (0.89 / 0.96)
and the reader read it (0.92 / 0.98). `sfx.is_runaway` threw them away: the
reader spells an ellipsis as `......` or `・・・・・・`, never `…`; six dots hold
the trigram `...` four times over, and the test fires on any 3-gram ≥ 3 times.
So every dialogue line of ≥ 9 characters that paused twice was a "runaway".

Measured over all 859 pages: 94 lines on 64 pages were runaway only
because of the dots (median 17 chars, p90 40 — the longest speech on each page,
concentrated on the wordy artists) against 28 real runaways, 17 length-capped,
194 correctly dropped as ASCII-only / one glyph, 60 with no letter.

The other mechanism, not fixed: Latin text hallucinated as kana (`Zzz...` →
`ててて…`, `Hi-!!` → `ウー!!`). Since `skip_en` drops every correct Latin read,
the only Latin that ever reaches a caption is a kana hallucination of it. No
text rule separates these; the reader score is not clean. Known miss.

Shipped (anime_tools `8ebaf58`): `sfx.normalize_read` folds every dot run to
one `…` and every heart to `♡`, before the guard, not at export (a hundred
dots alone fold to `…` and then correctly fail `has_script`); `has_script` no
longer counts `ー っ ッ ゝ ゞ ヽ ヾ` as a letter (motion lines read as kana); the
eval key folds the same way — which is the re-basing § Comparability describes.

Sidecars over the whole tree: 4,218 → 4,330 lines, det+glyph passing 3,475 →
3,633, characters +3.2 %; `・・・`/`...`/`…` 442/304/0 → 0/0/920; `♥`/`♡`
332/889 → 0/1,226.

The research copies (`../cjk_aware_anima/datasets/build_ocr_records.py`
`is_runaway` / `_normalize_read`) are deliberately not changed — they are the
keys of the recorded PP-vs-VL A/B and would re-score it.

## O3 — colorized COO: PASS as augmentation, LOSS as replacement

### The pilot (2026-09-06 night) — `reports/ocr_colorize_pilot_*.md`

`ocr/colorize_manga109.py`: the EasyControl colorize LoRA repaints Manga109-s
spreads as flat-tint doujin pages, leaving the lettering alone.

- The whole-spread form fails the gate. A Manga109 "page" is a two-page
  spread; free-fit into the 1024 band shrinks it to 1216×864 and the trip back
  blurs small kanji — stock speech exact fell 67 → 42 % on the same polygons.
- Splitting the spread into halves (864×1216 ≈ native) with the `comic` prompt
  passes both clauses: stroke IoU 0.90 / 0.92, ≥ 0.8 on 90 % / 99 % of crops;
  reads agree at better than the rate the source read agrees with the label; B′
  reads colorized at 84.9 vs 89.7 % SFX at equal sim.
- 16 steps ≡ 28 steps on every gate number at 0.59× the wall — the reference
  image carries the layout, the sampler only fills tint.
- The crop builder drops colorized crops below IoU d1 0.8 (~10 % of SFX).

### col100 — a 1.6 % append (2026-09-06)

In-domain +1.7, sincos +11 lines (312 → 323) — inside 1 σ (SE 12 lines) either
way, and the +10 lever gate was never approached on the basis it was written
for. Verdict: does not move the target at this share. The pre-registered
`--extra_repeat 8` arm (≈ 10 % share) was withdrawn by the user and is still
untested.

Label audit after the diff sheet (user: "col100이 훨씬 정확한데??") — the more
important finding. The sincos label file is two different things by kind:

- Speech (213) and chrome (26): `text_hand` is PP-OCRv6 record text on 211/213
  and 21/26 rows — `status=checked` certified the kind, not the text. PP
  reads no hearts, so every ♡ a reader reads inside a balloon scores as a
  miss. col100 differs from the label only by hearts on 28 speech rows; 16
  crops eyeballed, every heart is on the page. "Speech exact 59 → 54" is
  agreement with PP-OCRv6, not accuracy. Never judge a speech re-read on these
  labels.
- SFX (99, hand-typed): two genuine label errors found and fixed, both rows
  col100 read right. The other 26 rows where both readers agree against the
  label are shared misses — the small heart at the end of a burst, `…` vs
  `・・・` — not label errors.

### col1500sw — replacing the grey originals costs the gate (2026-09-07)

The user's framing ("can we drop corresponding black/white ones?"): each
colorized crop stands in for its grey original, so the training set stays
B′'s size (77,164 rows) and only 22.3 % of its appearance changes. New code:
`crop_dataset.load_split(..., extra_replace=True)` + `--extra_replace`. The keys
are an exact subset, so the swap is clean and the SFX/speech balance is
identical — a genuine single-variable comparison against B′.

In-domain +2.3 points, sincos −31 lines (312 → 281, −5.1 points). At n = 617
the binomial SE is 2.0 points, so this is ≈ 2.5 σ and mean sim moves with it
(0.852 → 0.837) — real, unlike col100's +1.6 inside 1 σ. Damage
concentrated in square SFX blocks: 53.8 → 33.3 %.

Verdict: the grey originals are load-bearing. Colorized COO helps as
augmentation and hurts as replacement. Repainting is not a relabelling of the
target domain — it deletes the real screentone appearance for a fifth of the set
and substitutes a synthetic recolour, and the doujin residual (outlined,
heart-terminated bursts) is not what the recolour supplies. The in-domain gain
is the tell that it did learn something: it moved toward the COO test set and
away from the target.

Untested point on this axis: the append-at-17k arm (18.2 % share) — whether
col100's +1.6 grows or saturates at 12× the share. Launched, then killed when
the user asked for the swap instead.

## O2 follow-up — B′ × 3 epochs: in-domain +4.1, sincos flat (2026-09-07)

Fresh 3-epoch schedule (not a warm restart), B′'s recipe otherwise unchanged.
Val SFX 80.8 → 85.7 → 88.3 %; `best` = ep3.

In-domain it is the largest single lift since the tower unfroze — every
orientation and every length bin ≥ 2 moves up, the 8+-char bin most (25.9 →
41.4 %), and it is the best COO reader in the ledger on both columns.
On the target it is flat: 312 → 306, heart-blind 363 → 349, mean sim and the
≥ 0.8 share unchanged. Row-level 53 better / 62 worse, the losses the same shape
as col100's (`♡` read as `☆`/`ッ`/dropped, consonant swaps on outlined kana).

Verdict: more epochs buy COO, not the doujin gap. The residual on sincos is a
domain residual. Ship decision unchanged; ep3 is a strictly better COO
reader if that ever matters. Weights local only.

## O2 follow-up — LP-FT (arm B → B′ order): closed (2026-09-08)

The user's question: does linear-probe-then-fine-tune (Kumar et al., ICLR
2022) buy anything here, given ep3 showed the schedule is not the lever. Stage 1
already existed as arm B, so the cost was near zero. Recipe:
`--init_adapter output/ocr/vl16_lr1e-4/ep2 --train_tower` with B′'s exact
stage-2 flags (new flag: load a trained adapter dir as the starting point). The
head start is real — step-25 loss 0.206 vs B′'s 0.914 from scratch.

Corrected 2026-09-08 (deduplication pass). This entry originally read
"in-domain speech +4.8 … the best speech reader so far", comparing LP-FT's
new-key row against B′'s old-key row. On one key (§ Comparability) the
comparison is:

| | COO SFX | COO speech | sincos / 617 |
|---|---|---|---|
| B′ | 83.2 % | 88.3 % | 312 |
| LP-FT | 84.3 % | 87.6 % | 264 |

So LP-FT is +1.1 SFX, −0.7 speech in-domain — not a speech win at all, and
not the best speech reader (that is ep3 at 89.2 %) — and −48 lines on the
target. The verdict is unchanged and now rests on cleaner ground: LP-FT is
closed for this reader.

Row-level vs B′ on sincos: 38 better / 78 worse; heart-blind 63 / 77 — so it is
not only hearts, but hearts are the largest single bucket (of the 497 rows whose
label carries `♡`, B′ emits a heart on 313 and LP-FT on 247).

Why, in the paper's own terms. LP-FT works by preserving pretrained
features — the head is near its optimum when the backbone unfreezes, so the
backbone moves less. § O2b established the doujin gap is a tower problem: the
target needs the tower to change. Starting from a decoder that has already
spent two epochs fitting grey COO locks in the COO prior before the tower sees a
gradient. The mechanism that makes LP-FT good in-domain is the one that costs
it here. If a warm start is ever wanted again, stage 1 must have seen the
target distribution.

(Launch gotcha: the first run trained a full epoch and crashed in val scoring —
anime_tools `62f6fc3` retired `ocr._text.normalize_ja` with the CTC recognizer,
and the script saved weights only after scoring. Fixed both ways —
`eval_manga109.py` now vendors that normaliser verbatim so every row stays on
one scorer, and the epoch dir is saved before val.)*

## Outside readers

### hayai-ocr, scored on both evals at its author's request (2026-09-07)

`JustANormalTinkerer` opened discussion #1 on the Hub repo. ~150 M params
(a `siglip2-base-patch16-naflex` tower + a 12-layer causal decoder) against B′'s
0.9 B. v2.1 and v2.1.5 are git branches; `main` is v2.0. Decoded by its own
card's recipe. Ledger rows above.

1. v2.1 is the best zero-shot reader measured in-domain — 78.2 % COO SFX
   without ever seeing COO from us, against 28.9 / 31.6 % for the two stock
   bases and 76.0 % for our own 4-epoch manga-ocr fine-tune. Caveat we cannot
   settle: its base mix (`hayai-dataset-merged`, now private) is not auditable,
   so Manga109 overlap is unknown. The sincos set carries no such risk.
2. v2.1.5 matches B′ on the doujin set at ~1/6 the parameters (316 vs 312).
   It is not a worse checkpoint, it is a differently aimed one — its fine-tune
   set is modern scanlation manga, JA + KO, i.e. exactly our surface, and not
   Manga109's 80s–00s printed B&W, where it drops 20 points. The two branches
   move in opposite directions across the two evals: the same decoupling B′'s
   own arms show, from the other side.
3. v2.1's low strict number is a heart bug, not a kana gap. 496 of the 617
   labels end in `♡`; v2.1 reads the kana right and drops the heart on 130
   lines. Ranked ♡-blind the order is B′ > v2.1.5 > v2.1. Where all three lose
   is length: v2.1 collapses past 6 characters.
4. B′ keeps the quality margin even where it ties on exact (sim 0.855 vs
   0.789). Whether that matters depends on the consumer: for a deduped SFX
   caption clause a near-miss and a miss cost the same, so a 150 M reader at
   30 crops/s is a real alternative if the reader ever needs to get cheaper.
   Not a ship decision — B′ stays.

The discussion's second ask, publishing the test set, is not answered: the
sincos labels are useless without the doujin pages (the user's own dataset), and
the COO half needs Manga109-s, which may not be redistributed at all.

### HunyuanOCR-1.5 — measured stock, out (2026-09-08)

`tencent/HunyuanOCR` (1 B, Tencent Hunyuan Community licence, native in
transformers ≥ 5.13 — no remote code). Full write-up
`reports/0908_hunyuan_vs_vl16.md`; prompt sweep `reports/0908_hunyuan_prompt_sweep.md`.

1. It is not a Japanese manga reader. In-domain it is below stock
   manga-ocr on every column. On the doujin gate it ties stock VL-1.6 (21 vs
   19) but both sit on the ~3 % floor this line exists to lift, so the tie
   carries no information. Nothing here reopens the base decision.
2. Half the miss is the prompt naming no language. Under the official
   Chinese instruction 51 % of its COO SFX reads contain no kana at all (840
   of 2,558 pure Han: `ドドド` → `咚咚`) vs 4 % for VL-1.6. A Japanese
   instruction drops that to 18 %; an English one is worse than the Chinese.
   Language prior, not legibility — and the lever is one upstream's own client
   will not let a user pull (it exposes `--task-type`, never a free prompt).
3. What the prompt does not repair is small kana. Folding
   `っゃゅょぁぃぅぇぉ` on both sides rescues +6.8 points of its COO speech
   vs +0.8 for VL-1.6. It also emits furigana as its own interleaved line.
   Small kana + `ー` + `♡` are precisely what VL-1.6 was picked for.
4. A fine-tune arm has a weak prior. It would start 18 SFX / 39 speech
   points behind B′'s starting point with systematic kana errors, on an untried
   architecture, while B′ and hayai v2.1.5 already bracket the gate at ~50 %.
   Not run. Complementarity is real but one-sided: 108 COO SFX lines Hunyuan-ja
   reads and VL-1.6 misses, against 547 the other way.

Wall (batch-matched, 600 crops, bs 32): Hunyuan 20.2 crops/s vs VL-1.6 26.6.
Its `min_pixels` is 262144, so a 40×60 SFX crop is upscaled to ≥ 256 visual
tokens — cost per crop is near-flat in crop size.

## O6 → D0–D1 — `deepghs/AnimeText_yolo` replaces the whole detection stack

*Sincos figures in this section are `/ 99` on the PP-box labels, which this same
night's re-base retired.*

O6, the probe (2026-09-06 night). The stock YOLO12 `text_block` detector, no
training, in front of the SFX reader.

| known lines | covered by stock yolo12l @ 640, conf 0.426 |
|---|---|
| PP-OCRv6 v3 records (237) | 237 / 237 |
| hybrid_vl speech / sfx / chrome | 98 % / 96 % / 96 % |
| hand rows speech / sfx | 100 % / 98 % |
| MIT mask components, min side 32 | 92 % |
| masked-but-no-box floor (133 masked) | 3 (PP DB 38, the full 3-layer stack 8) |

Read-level, boxes → SfxReader → records: floor 8 → 4, best-match to
manga-ocr 0.810 → 0.844, hand-SFX exact 44 → 63–66. Of ~520 boxes
matching nothing known, 438 yield a valid read, and the sheets show real SFX
on pages the stack held at zero lines.

Verdict: PP DB, VL Spotting and the mask-component crops are all replaceable
by this one detector — which also retires the detector half of O5.

Facts to carry:

- Model and input size are a wash (l 640 ≈ l 1024 ≈ l native ≈ x 1024 on
  every column); 640 is 26 ms/page on the CUDA EP vs 310 ms native.
- Nested boxes: YOLO emits a balloon block and its columns. `outer`
  (keep the block) tanks best-match to 0.694 — block reads don't match balloon
  lines; `inner` (drop a box holding ≥ 2 others) costs nothing measurable.
- `join_cjk` over YOLO boxes loses (−0.041 best-match: SFX beside a balloon
  gets pulled into it). The stage never joins under `animetext`.
- PP-OCRv6 recognition on block boxes is unusable (floor 26, hand-SFX 4/99),
  measured once — a YOLO block box is multi-column and dies on the score floor
  before the reader sees it. Hence the detect-only engine: `animetext` + `vl`
  loads no recognizer at all. This was not foreseen in the plan.
- ORT gotcha, applies to every ONNX session: the CUDA EP's default BFC arena
  + EXHAUSTIVE cuDNN search grew to 15 GB over per-page shapes and killed the
  next session's `cublasCreate`. Bound it (`kSameAsRequested`, `HEURISTIC`,
  `gpu_mem_limit`).
- Licence: model card GPL-3.0, dataset CC-BY-NC-SA-4.0. Runtime download,
  never bundled into the MIT package, the trainer, or a node.

D0 — package (anime_tools `b015ba2` + `2cbe201`): `ocr.animetext.AnimeTextDetector`,
a three-call `Detector` protocol (`prepare` / `forward_batch` / `boxes`) so DB
and YOLO share the crop and the size filters, the catalog row `animetext_det`
(`stages=()`, so the stage bar never demands GPL weights), `--detector` /
`--det_conf`, the bounded arena in `make_session`, 29 weights-free tests. Gate:
six pages reproduce the probe's boxes 15/15. PASS.

D1 — records (`ocr/animetext_records.py`): 1,146 boxes = the probe's count;
reads → records with a record-level dedupe (a record whose normalized text
sits inside another's with box containment ≥ 0.85 — the column read repeated by
its block read; 46 drops) → `ocr_records_sincos_animetext.jsonl`, 955 records
on 163 pages, floor 4, best-match 0.844.

Gate: floor ✓, best-match ✓, hand-SFX exact ≥ 64 → 63. The 64 was `raw`'s
number; under `inner` (the chosen config) the reference is 63 and the file
reproduces it exactly. The three rows `raw` gets are one shape — a hand label
spanning a doubled SFX whose block `inner` drops for its two repeats, which the
SFX-clause dedupe collapses anyway. Reading the gate against the `inner`
reference: PASS; as written: −1, explained.

Hand pass, the precision number O6 lacked. 491 records sit on boxes no older
record covers; 60 drawn and graded by the user: 59 / 60 real lettering (the
one exception is pink hearts drawn as censorship). Precision gate PASS; the
kill (raise `--det_conf` to 0.426) is not triggered.

## D2 killed, D3 — the flip: AnimeText + VL are the defaults (2026-09-06, late night)

D2 killed. The user stopped the arm at step 715 / 2808 ("kill current daemon
run and just proceed d3 … retiring ppocr"), so the D2 gate was never
measured and the records default flipped without a verdict. The mirror, the
isoq TE cache, the caption diff and `cjk_unmask_d2.toml` stay on disk; the arm
is re-launchable from the config header.

What D2 would have tested, and is therefore open: the animetext captions are
a caption-count change, not a reader swap — 151 of 351 captions differ
from `hybrid_vl`, SFX lines 142 → 353, speech 250 → 356, 27 pages newly carry
text. Whether roughly twice the SFX per page trains cleaner or spammier than C11
is unknown. Any future DiT arm on this shard trains on the animetext captions
by default; compare against C11 with that caveat, or re-run D2.

D3 — the flip (anime_tools `ae6f33e`, pushed and pinned). `OcrRequest`
defaults to `detector="animetext"`, `reader="vl"` — every build, not only
under `--reader vl` as the plan wrote: the user's licence call is that a default
fetching GPL-3.0 weights (NC data) at runtime is fine, since nothing is bundled.
`OcrRequest()` is therefore detect-only; `--detector ppocr --reader ppocr`
remains the explicit torch-free pair, and `--mask_dir` still requires
`--detector ppocr`. Trainer defaults follow; `reread_records.py` carries a
superseded banner and stays for C10/C11 reproducibility.

Stage gotcha: the OCR stage by module needs explicit `--dst
post_image_dataset/resized --ocr_dir post_image_dataset/ocr`; its bare defaults
are `workspace/…`.

## Plain vs OCR captions on sincos — the clauses are neutral (2026-09-08)

The plain control the line had owed since 2026-09-01, re-run against what the
package publishes today (AnimeText + VL, the O4c–O4e caption rules, clauses
composed by `with_ocr_clause` off the sidecar tree — what any user of the package
gets, not a research records file). Two arms, one training seed each, masks
off in both, latents shared bit-identically, one variable. Full design + tables
`reports/0908_plain_vs_ocr.md`, `reports/0908_alpha128.md`,
`reports/0908_v2_prompts_a128.md`.

| blind set | arms | result |
|---|---|---|
| `s16` (α32, v1 prompts, 24 pairs) | OCR vs PLAIN | 14 – 9, 1 tie |
| `s17` (α128, v1 prompts, 24 pairs) | OCR128 vs PLAIN128 | 10 – 12, 2 ties |
| `s18` (α128, v2 hard prompts, 32 pairs) | OCR128 vs PLAIN128 | 11 – 15, 6 ties |

Verdict: three blind reads, three ties. Pooled 35–36 on 71 decisive
pairs, one-sided p = 0.50. Every automated readout is flat in all three
(`cos→base` matches to four decimals; adherence differs by less than the
row-to-row spread). So the shipped speech + SFX clauses are neutral for image
quality on this shard — still shippable (they cost nothing), but the "captions
are load-bearing" claim is closed in the negative for the render axis. It
rests only on the 2026-09-01 unmask A/B/C, where arm C beat spam, not
quality. Do not spend another render-only re-eval here; moving it needs a
text-requesting prompt grid, or a second training seed per arm.

Four instrument findings that outlive the pair:

- α is a pure output-scale knob here, not an LR proxy. `network_alpha`
  32 → 128 scaled ‖ΔW‖ 3.7× with the learned factors unchanged (‖A,B‖ +1.8 %) —
  Adam normalises the ~16× effective-step argument away. At α/r = 1 the
  adapter barely moves the base, and the spread between two training seeds of
  the same arm was larger than any between-arm gap this line ever measured.
- A more specified prompt is a less sensitive instrument — the opposite of
  the intuition that sent us there. Arm-vs-arm PE cos went 0.9794 (v1) → 0.9884
  (v2) while the same-arm different-seed floor rose 0.9347 → 0.9738: v2 rows
  carry 12–15 tags and named characters, so they pin the composition and leave
  the seed and the adapter less room.
- Two text-requesting rows cannot measure the text axis. "OCR writes 47
  lines vs PLAIN's 35" is one cell; by glyph area the sign flips. n = 4 per arm.
- The spam tally changed units at D3 — `grid_spam_tally.py` was ported to
  the detect-only engine, so boxes carry no text and `n_lines` / `glyph_frac`
  are the tally. Never compare these counts to a pre-D3 tally, C10's "~2 on
  3 seeds" included. Within this set, box counts are a wash, and adapter
  magnitude is not what produced arm B's spam (α128 reproduces the wash).

## Context is not the lever — margin / marker / page-text sweep (2026-09-08)

Question: the 12 %-pad crop looks too tight to infer a hard SFX from; would a
page-level context channel (PE-Core / PE-Spatial as a prefix) help? Measure
whether context carries signal through the reader's own tower before building a
modality bridge. `reports/0908_context_margin_sweep.md`.

Arms: pad 0.12 / 0.35 / 0.7 / 1.5 × a red box drawn around the 12 % crop × an
oracle page-text arm (the page's Manga109 `<text>` lines quoted in the
prompt), on three readers.

Monotone on every reader: more frame never beats the tight crop. B′ 316 →
294 (pad 0.35) → 45 (pad 1.5); stock VL 21 → 4; Hunyuan 21 → 16. `contains`
(label ⊂ prediction) says the target is found less often with more context, so
the surrounding pixels distract rather than disambiguate. The marker works
on Hunyuan (runaways 28 → 0 at pad 1.5) but still loses to pad 0.12. Oracle
dialogue in the prompt is below baseline.

Verdict: a learned page-context channel — strictly weaker than oracle text and
than the reader's own encoder on the same pixels — is closed as a direction.
B′ degrades fastest, as expected for a reader trained on 12 % crops.

## Tower — where B′'s misses are, and what label-free adaptation does (2026-09-08)

Motivation: B → B′ (tower unfrozen) took sincos 110 → 312 on the same
labels, and its garbage misses (sim < 0.5) 185 → 43 — perception was the
bottleneck. Can the tower be adapted with no labels on in-domain crops?
`plan_ssl_tower.md`.

B′'s miss profile (computed on the pre-rekey 313-miss set; the fold moves
the count to 305 and none of the classes): 160 near (sim ≥ 0.8), 110 mid, 43
garbage, 0 empty; 55 are ♡-only (`びく♡` → `びく`), 61 ♡/〜/ー-only. ♡ is in
497 / 617 gate rows and 337 / 77k COO train rows — an LM-side label gap no
vision-only method moves. This is the single most useful number in the section:
it is why the remaining headroom is on the label side.

Full FT is out on this hardware. The official recipe (ERNIEKit
`paddleocr_vl_sft.md`) is Full FT at lr 5e-6 on ~30k labelled samples — it
confirms the tower is meant to be trained for font-style shifts and says
nothing about the no-label case. fp32 AdamW states for 800 M params do not fit
on 16 GB beside activations, and `bitsandbytes` 0.49.2 ships no
`libbitsandbytes_cuda132.so` for torch 2.12+cu132 (`AdamW8bit` fails at the
first `step()`). bnb removed from the deps.

Corpus: `deepghs/AnimeText` test split (73,725 images) already on the volume;
`ocr/animetext_crops.py` cuts text boxes at 12 % pad — 7.6 boxes / image,
140,132 crops from the first 19,514 images. Mostly manga bubbles + hand-lettered
SFX, closer to doujin than the "anime scene text" framing suggested.
CC-BY-NC-SA → research build only.

### S0 — a pixel target collapses the tower; a feature target does not

Pixel SimMIM (mask 60 % of 2×2 blocks after the patch conv, linear head →
588 pixels, L1): masked L1 2.15 → 0.37 in 30 steps and the tower collapsed —
median relative ΔW 4e-4, yet cosine to stock features 0.38, feature norm
×0.22, the untouched LM reads `""` on every crop, and a 30-step SFT from it
scores 0 % where the stock-tower SFT smoke scores 48 %. Bisect: embeddings /
post-LN innocent; any 9-layer block of encoder weights alone does it.

Mechanism: a pixel target on the final features drags the representation out
of the space the projector reads, in one coordinated direction per Adam step —
small weights, large features. Lowering the lr slows it; it does not change the
destination. The objective has to live in the space the projector reads.

Feature target (masked-token features regressed to the frozen stock
tower's `last_hidden_state` on the unmasked crop — data2vec / BEiT-v2 with the
base model as its own tokenizer, + 0.1× the same loss on unmasked tokens):
read-through PASS (cosine 0.967, norm ×1.09, 6/6 reads), SFT smoke 39.1 %
SFX vs 48.4 % for the stock-tower smoke — same ballpark at 30 steps.

### S1/S2 — the lift is not there; the line closes

Reports `reports/ocr_eval{,_sfx}_vl16_tower_ssl.md`.

S1 read-through gate PASS: 1 epoch over the 140k crops, held-out masked L1
0.235 → 0.084, cosine to stock 0.981, norm ×1.035, 6/6 non-empty reads.
One epoch of in-domain SSL leaves the tower inside the space the projector reads.

S2, the same SFT from that init (both rows on one key and one harness):
sincos 312 → 307, heart-blind 375 → 380, sim 0.852 → 0.855; COO SFX
83.2 → 84.6 %, speech 88.3 → 88.2 %.

Verdict: −5 on the gate, +38 lines in-domain. The plan's `≥ +15` bar is not
met and its kill rule (`S2 ≤ B′ + 10` with the Phase-0 gate passed) fires:
label-free tower adaptation is closed as a lever for this reader. S3 (the full
560k crops, ep2, an in-domain doujin corpus) is not run — the kill rule was
written to stop exactly that follow-up — and no lr / mask-ratio sweep follows.

This is the fifth arm running where in-domain and the doujin gate decouple. The
open lever is ♡-bearing and small-kana labels (synthetic or pseudo), which
O3's synth-SFX bullet describes and which was never built.

Weights local only; the shipped reader stays B′. An AnimeText-derived tower would
have been research-only (NC) regardless.
