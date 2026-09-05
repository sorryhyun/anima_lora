# plan_base1 — a better OCR base before D2: hybrid reader, SFX handling, arm C10 (2026-09-05)

*Sub-plan of [`plan.md`](plan.md). D2 builds the OCR-diverse corpus on top of
whatever reader and caption shape this line trusts; today's eyeball
(`findings.md`, "OCR eyeball + SFX handling") says the current base — PP-OCRv6
alone, lines as `「…」` tags, no speech/SFX distinction — is not what D2 should
freeze. This plan fixes the base on `sincos` (351 images, 133 text-masked,
the one shard every unmask arm has run on) and validates it with one training
arm, C10, before the corpus job runs. It does not reopen the reader A/B
(`reports/0905_paddleocr_vl16_vs_ppocrv6.md` stands: PP-OCRv6 is the
recognizer of record; VL-1.6 is complementary, never the replacement).*

## What the eyeball showed (measured 2026-09-05)

| fact | number / pointer |
|---|---|
| masked-but-no-line floor | 44 / 133 sincos pages; the misses are hand-lettered SFX (`ぱんぱん`, `びくっ`) and loose kana on the art, not balloons |
| PP-OCRv6 on SFX | garbles what it does detect: `でくv` for びくっ, `Kッ4vv`, `ゴmvvv`, `はんぱん` for ぱんぱん |
| VL-1.6 Spotting on two floor pages | `12440144`: one SFX (`ばるん`) found, six pink SFX missed, identical at 896 px and 3048 px — resolution is not the limit; `6067089`: the balloon PP dropped is found (`もしかして興奮してるー？`), 興→無 rewrite on the original, page `OCR:` reads it right |
| trained captions vs records | C2–C9 trained on the **v1** records (97 stems, top-to-bottom order); v2's reading order (58 / 96 stems differ) never reached an arm |
| speech / SFX by text rules | 19 SFX / 209 speech of 228 lines; 15 of 96 captions get an SFX sentence (`datasets/ocr_sfx.py`) |
| SAM3 `speech bubble` | balloons on 34 / 97 pages, 69 / 228 lines inside one; as a veto it moved **one** line (`バスト91`); misses plain rounded balloons (`カリカリ`, 6813398) |
| UI chrome in the records | `ツイート`, `ポスト`, `完了にする`, `★お気に入り` (screenshot pages) — neither speech nor SFX, currently captioned as one or the other |
| sentence format | `… . Japanese text reads as "…", "…". Japanese SFX reads as "…".` appends string-level; the anime_tools grammar has no header for it (a re-parse glues the first sentence to the last tag) |

## Decisions

1. **PP-OCRv6 stays the recognizer; VL-1.6 is the complement.** Two jobs
   only: (a) *detector* — page `Spotting:` adds lines PP's DB head never
   boxed (balloons like 6067089's, SFX like `ばるん`); (b) *second reader on
   PP's own quads* where PP is weak — score < 0.85, a symbol (`♡ ー ～`) in
   dispute, or a line the SFX rule flags (PP's SFX reads are the garbage).
   Not prompt engineering, not a swap (plan.md not-doing list holds).
2. **SFX handling is a records field, not a caption-time guess.** Every
   record carries `kind ∈ {speech, sfx, chrome}`; the mirror builder reads
   it and never looks at pixels. `chrome` is dropped from both sentences.
   **Amended 2026-09-05 (user):** `sfx` is dropped from captions too, for
   now — neither reader can read hand-lettered onomatopoeia, so an SFX
   sentence would teach garbage; the records keep the lines for B1's labels
   and for a later reader (a light OCR fine-tune on SFX crops — future work,
   not this plan). C10's sentence caption is speech-only.
5. **A joined block keeps its boundaries.** `anime_tools` `JOIN_SEP = " "`
   (rev cd75591): the columns / rows PP-OCRv6 joins into one record are
   separated by a space, so a profile card's fields (9410777) stay distinct;
   VL crop rows and Spotting blocks get the same treatment.
3. **Sentence-shaped captions are the C10 arm, not yet the D2 default.**
   The shape is decided by C10's gate, not by taste.
4. **SAM3 is out of the speech/SFX path.** It is not the lever at this
   recall; reopen only with a balloon soft prompt trained the way the
   subject prompt was (`--prompt_embed`), which is not this plan's work.

## Phases

### B0 — hybrid records on sincos (½ day; one daemon GPU job) — **DONE 2026-09-05, gate PASS**

*Result (`findings.md` "B0" + addendum): on the v3 space-join PP pass, floor
**38 → 23** on 133 masked pages (first pass on the v2 file: 44 → 27),
manga-ocr similarity 0.751 → 0.786 on the 40 A/B pages, VL-only lines on the
sheet are real balloons / SFX; `kind: sfx` is kept in the records but dropped
from captions (decision 2, amended). Deviations from the text below, all measured: matching
is IoU ≥ 0.3 / containment ≥ 0.5 / text-sim fallback after `join_cjk` on the
VL lines (IoU 0.5 fails on offset thin columns); rule 1b carries three extra
guards (never lose a heart PP had, weak re-reads need Spotting corroboration,
symbol disputes may not change letters) — under them 0 / 9 symbol disputes
and 37 / 70 second reads survive; two full-page Spotting quads are gated by
area. `kind` is the v1 rule + chrome list until B1. Raw VL outputs are cached
(`ocr_raw_vl16_sincos.jsonl`) so `--stage merge` re-runs on CPU.*


`datasets/build_ocr_records.py` (new; the D2 builder grows from it):

- Inputs: the PP-OCRv6 v2 records (boxes, scores, reading order); the resized
  pages. Runs VL-1.6 page `Spotting:` (×2 upscale below 1500 px, bs 8,
  `use_cache=True`, the `probes/ocr_vl16_ab.py` recipe) on every page of the
  shard, not only the 97 with PP lines — the 44 floor pages are the point.
- Merge: VL quads → axis bounds; IoU ≥ 0.5 with a PP box = the same line
  (PP text kept unless rule 1b fires → VL `OCR:` on PP's quad, repetition
  guard: reject a read whose 3-gram repeats or that is longer than 2× the
  PP read); VL-only lines enter with `engine=vl16_spotting` after the
  floors (`min_chars`, ASCII-only, tally) and a chrome/garbage gate.
  Reading order recomputed over the merged boxes with
  `anime_tools.ocr.reading_order` (page-aware, right-to-left).
- Every record gets `kind` from B1's rule (text + box size); `engine`,
  `score`, `post` as before. Output
  `post_image_dataset/cjk_unmask/ocr_records_sincos_hybrid.jsonl`.
- Chore: `build_mirror` symlink creation fails sporadically on the ntfs3
  dataset volume — fall back to `os.link` (same filesystem) or retry once,
  and say which happened.

*Gate:* the floor: masked-but-no-line count on sincos, PP alone (44) vs
hybrid — report it; the contact sheet (`probes/ocr_contact_sheet.py`, the
session script promoted) shows VL-only lines in their own colour for an
eyeball; on the 40 A/B pages the hybrid's similarity to manga-ocr is ≥ PP's
(no regression from the second reader). *Kill:* floor unchanged and VL-only
lines are chrome/SFX garbage on the sheet → VL stays a floor instrument
only, C10 runs on v2 records.

### B1 — SFX handling v2 (½ day CPU + ~1 h of hand labels)

- **Labels first.** `assets/sfx_labels_sincos.tsv`: every hybrid line, hand
  labelled `speech | sfx | chrome` off `bubble_kind.pdf`-style crops. The
  rule is tuned against it and reported as accuracy; no rule change without
  the number.
- **Rule v2** (`datasets/ocr_sfx.py`): the v1 text rules plus (a) a glyph
  size feature from the records alone — box short side vs the page's median
  column width (sincos p50 58 px; SFX lettering runs to p90 124 px) — as the
  tiebreak for short non-vocal kana the text rules leave as speech;
  (b) `chrome`: ASCII/kanji UI strings from a small list (`ツイート`,
  `ポスト`, `完了にする`, `お気に入り`, `バスト`, `cm`) and any line inside a
  box that PP scored ≥ 0.98 on a screenshot-flat background — measured, not
  assumed; (c) the SFX onset lexicon grown from the VL-only SFX reads of B0
  (what the hand labels call SFX and the rules missed).
- Reader-side SFX fix: rule 1b sends every `sfx` line's PP quad through VL
  `OCR:`; the sheet shows PP vs VL side by side for those.
- **Vocal moans** (`あっ…うっ…`, `おおおん`) stay speech — they are a mouth
  in the caption's sense; if the labels disagree, that is the one rule the
  labels may flip.

*Gate:* ≥ 95 % agreement with the labels on sincos, chrome dropped from
captions, the mirror builder reading `kind` only.

### B2 — grammar-native sentences (½ day; `anime_tools`) — **DONE 2026-09-05**

`TEXT_PREFIXES = ("Japanese text reads as ", "Japanese SFX reads as ")` as a
clause kind beside `On the` / `In the` in
`anime_tools.captions.position_clauses`: parse keeps each as its own clause
(tags = the quoted lines), compose renders them last, after position
clauses; `"` pairs already opaque (D1 rev). Test the C10 caption round-trips
byte-identical and the shuffled-variants pass leaves the tail whole. Pinned
rev bump. Until this lands the string-level append in `cache_te_ext.py`
holds (nothing tag-level runs on a mirror caption after it).

*Landed (anime_tools b453cc2, pinned): a text clause is a `PositionClause`
with an empty `position`, the quoted lines as tags and `is_text` set
(`text_clause(lines)` builds one); `compose_caption` always renders text
clauses last; `has_clauses` stays position-only (`has_text_clauses` is the
other question); variants, `correct_caption`'s clause drop and
`flatten_caption` pass a text clause through verbatim; a quoted
punctuation-only line at the caption's end sheds the period. Trainer side:
`cache_te_ext.py`'s `sentence` format now composes through the grammar
(`ocr_text_clauses`), and — per the amended decision 2 — a line the
`ocr_sfx` rule reads as SFX is **skipped**, not sentenced (speech-only C10
captions; the SFX prefix stays registered in the grammar for when a reader
can read them). `tests/test_cjk_ocr_captions.py` pins the byte-identical
round-trip and the whole tail under `clause_dropout_rate=1.0`.*

### B3 — arm C10 (one evening on the daemon; 3 seeds) — **LAUNCHED 2026-09-05 23:40**

*Daemon jobs `20260905-234025-83503f` (seed 42, builds
`mirror_sincos_hybrid_sentence` + `te/sincos_hybrid_sentence_isoq`),
`-fa1a24` (seed 7), `-ba0ac2` (seed 1234); configs
`cjk_unmask_c10{,_s7,_s1234}.toml` (gitignored dir); grids
`armC10{,s7,s1234}_s{42,7,1234}`. CPU dry run before launch: 105 / 351
captions carry a speech sentence (hybrid records: 259 speech · 71 sfx · 8
chrome lines; sfx + chrome dropped), 13 of them after a position clause, 0
round-trip or variant-tail failures.*

C9ISOQ's recipe verbatim (quote-partitioned pack
`cjk_vocab_pack_synthjakozh1sym_r256_isoq`, `"…"` spans on the isotropic
mirror block, rank 32, 8 epochs, `blocks_to_swap 0`) with the two changes
this plan is about: **hybrid records** (B0) and **sentence captions with the
speech/SFX split** (B1/B2). `configs/gui-methods/custom/cjk_unmask_c10.toml`
(own `text_cache_dir` and `output_name`; latents shared with C2–C9), launched
by `run_unmask_r2.py --records …_hybrid.jsonl --mirror mirror_sincos_hybrid_sentence
--ocr_format sentence --ext_prefix …_isoq --method cjk_unmask_c10`, seeds
42 / 7 / 1234 (principle 3). C9ISOQ is the control: same pack, same latents,
tags format on v1 records — the diff is reader + caption shape and nothing
else.

Readouts, all existing: the 8-row unmask grid spam tally (C-series ledger,
`findings.md` §10: C2 0 · C8 ~2 · C9 ~2), a blind pair set C10 vs C9ISOQ
(protocol in memory / `probes/blind_pairs.py`, ≥ 24 pairs, user grades), and
the `japanese text` presence-prompt row (does a text address alone spam
more or less than before). One new cheap one: prompt the top SFX sentence
(`Japanese SFX reads as "ぱんぱん"`) and count OCR'd glyph pixels in the
render with PP-OCRv6 — not a quality metric, a spam-direction check.

*Gate:* C10 spam events ≤ C9ISOQ on 3 seeds **and** blind ≥ C9ISOQ inside
the seed-twin floor (s02: 9-15 on 24) — then sentence captions on hybrid
records become D2's default caption shape. *Kill:* spam up on ≥ 2 seeds →
the sentence shape (text made more attributable) renders more text, and D2
keeps `tags`; blind clearly below → same. Either way the hybrid *records*
survive (a reader is not a caption shape); only the shape is on trial.

### B4 — fold into plan.md (¼ day)

D2's reader paragraph becomes "hybrid records via `build_ocr_records.py`",
its caption paragraph reads the C10 verdict, the floor-measurement clause
is done (B0's number), and SAM3 joins the not-doing list. `findings.md`
gets the B0 floor, B1 accuracy, C10 tally.

## Not doing here

- **No SAM3 / MIT bubble geometry** in the speech/SFX rule (decision 4). MIT
  is the text-pixel mask already merged into `masks/`; it says *text*, not
  *balloon*, and is not the missing signal either.
- **No VL fine-tune, no VL prompt sweeps**, no VL as sole reader (plan.md).
- **No romanised SFX** (`pan pan`): the lines keep their glyphs — the ext
  rows are the address, and romaji would ride stock T5 pieces.
- **No new pack, no new rank** — C10 rides the D1 pack as is.
- **No corpus-scale job before B3's gate** — the 873 + paired corpus wait.

## Order and budget

B0 (½ d + ~10 min GPU: 351 pages Spotting at ~1.4 pages/s, crops at bs 32)
→ B1 (½ d + labels) → B2 (½ d, can overlap B1) → B3 (3 seeds × ~40 min cache
+ train on 351 images ≈ 2 GPU-h, grids + blind set the next morning) →
B4. Three working days; every GPU step is a daemon job.

## Deliverables

- `datasets/build_ocr_records.py`; `ocr_records_sincos_hybrid.jsonl`;
  `probes/ocr_contact_sheet.py` (promoted from the 2026-09-05 session
  script, with the VL-only colour).
- `datasets/ocr_sfx.py` v2 + `assets/sfx_labels_sincos.tsv` + the accuracy
  number in `findings.md`; `kind` in the records; chrome dropped.
- `anime_tools`: `TEXT_PREFIXES` clause kind + tests + pinned rev bump.
- `configs/gui-methods/custom/cjk_unmask_c10.toml`; grids
  `output/tests/cjk_unmask_eval2/armC10_s*`; blind set `s14` (C10 vs C9ISOQ);
  `reports/09xx_c10_sentence_captions.md`.
- Done today: `datasets/ocr_sfx.py` v1, `--ocr_format sentence`,
  `tests/test_cjk_ocr_captions.py`, `output/tests/{ocr_contact_sheet,
  vl16_single*, sam_bubbles}/` (scratch), `findings.md` entry; **B0** —
  `datasets/build_ocr_records.py`, `ocr_records_sincos_hybrid.jsonl`,
  `probes/ocr_contact_sheet.py`, `reports/0905_b0_hybrid_records.md`, chrome
  drop + link fallback in `cache_te_ext.py`.
