# CJK-aware Anima, DiT side — findings

Settled verdicts of this line, one entry per phase, evidence pointer beside
each. The encoder-side verdicts it builds on are in
[`../cjk_aware_anima/findings.md`](../cjk_aware_anima/findings.md) (read-only).

## D0 — ISO1 vs C9 direct blind set: flat (2026-09-05)

`reports/blind_s13_ISO1_vs_C9.md` (in the old line's `reports/`): 48 pairs,
16 v2 rows × seeds 6/7/8, both arms fresh to the grader. **ISO1 23 – C9 20,
tie 5; rows 6-6 (tie 4); p 0.76.** The isotropic table and the trained
r256 pack are indistinguishable for unmask training on this grid.

- The transitivity claim ISO1 ≈ HOT > C9 (s12 + s11) does **not** survive
  the direct test; transitivity has now failed twice in this protocol
  (s03/s04, s12/s13). Do not chain blind sets — pair the arms you want to
  compare.
- Pooled s01–s13: a content-free table is never worse than the trained pack
  for the OCR route, and rows must exist (C9 > P). The isotropic block is
  therefore the OCR-route default on **cost** grounds (seed-generated,
  deterministic, no distill), not on quality grounds. The hypothesis doc's
  "structured low-rank spread hurts" mechanism is weakened, not confirmed.
- Mechanism note carried from `bench/frontload_text_boost`: `k_norm`
  strips row scale on the K path, which is why HOT (norm ×5) ≈ ISO1 (s12).

Gate outcome: **proceed to D1** with the isotropic block for 「…」 spans;
bare CJK tags keep the trained rows (plan principle 2).

## OCR reader for D2/D3 — PaddleOCR-VL-1.6 is not an upgrade over PP-OCRv6 (2026-09-05)

`reports/0905_paddleocr_vl16_vs_ppocrv6.md`; probes `probes/ocr_vl16_ab.py`,
`probes/ocr_vl16_prompt_batch.py`; raw outputs `output/tests/vl16_{ab,prompt}/`.
40 sincos pages with PP-OCRv6 sidecars, VL-1.6 read three ways (page
`Spotting:`, page `OCR:`, `OCR:` on PP-OCRv6's own quads); disputed lines
checked against the pixels.

- **Character accuracy is a wash on the same crops** (69/132 identical after
  punctuation normalization; each reader wins about half the disputed
  lines; vs manga-ocr 0.767 PP / 0.774 VL). VL is not a drop-in accuracy
  upgrade — consistent with the public manga fine-tune figure (stock model
  27 % sentence accuracy on Manga109 crops).
- **VL wins symbols and recall.** Hearts survive (30 lines vs 4), `ー` and
  small kana come back as themselves (what `anime_tools.ocr._text`'s
  `normalize_ja` patches by hand), and page `Spotting:` finds 260 lines vs
  PP-OCRv6's 132 — SFX, chat chrome — with per-column boxes and silence on
  the tally-mark / screentone pages.
- **VL loses by rewriting.** Page-level modes swap the printed word for a
  likelier one (`狠狠地`→`狼狽地`, `おい`→`あい`, `喘いで`→`噛いで`) and
  greedy decoding runs away on short SFX crops (2/132 crops, 1/40 pages).
  PP-OCRv6 garbles but never rewrites. **This decides the D3 CER judge**:
  a reader with an LM prior would "repair" a half-rendered line toward the
  prompted word and inflate the held-out-vs-never-seen gap. The render→OCR
  CER instrument uses PP-OCRv6.
- **Prompt hints are not a lever.** The chat template concatenates text
  after the fixed task token; "vertical / right-to-left / Japanese manga"
  hints churn 40–60 % of crop outputs in random directions (sim vs PP
  0.845 → 0.844 / 0.848 / 0.858) and make hinted `Spotting:` 2× slower,
  lower (0.69 → 0.63), and degenerate on 2–3 of 12 pages. Native Spotting
  order is not R→L either (39 vs 69 adjacent pairs) — `reading_order`
  stays geometric.
- **Batching is the throughput lever.** Left-padded, area-sorted crops:
  4.0 → 18.5 crops/s at bs 32 (2.6 GB); pages 0.56 → 1.40 pages/s at bs 8
  (4.5 GB). Outputs churn at the byte level (`♥`↔`❤️`, ellipsis lengths,
  column-break placement) with unchanged line counts and similarity.
  Two shipped gotchas: `generation_config.json` has `use_cache: false`
  (vision tower re-run per token; `use_cache=True` is byte-identical at
  11×), and transformers 5.16's image processor exposes `size`, not the
  card's `min_pixels`.

Decision for the plan: **D2's records stay PP-OCRv6** (detector +
confidence + v2 post-processing, unchanged); VL-1.6 enters only as an
optional hybrid pass — `OCR:` on PP-OCRv6's quads with a repetition guard,
preferred where PP's score is low or the two disagree on symbols — and as a
**detector for the "masked but no OCR line" floor**: its Spotting recall is
the one thing that could shrink sincos' 44-of-133, which caps every unmask
arm. Measure that floor with both detectors in D2 before deciding whether
the hybrid pass is built at all.

## D1 — deterministic table + route partition + LoRA stamp (2026-09-05)

**Gate: PASSED (sanity)** — daemon job `20260905-210248-114990` (rc 0):
arm `C9ISOQ` = the C9 recipe re-cached through the partitioned pack, trained
2,808 steps, 8-row grid at seeds 42/7/1234 into
`output/tests/cjk_unmask_eval2/armC9ISOQ_s*`, read against `armC9_s*` with
the same-recipe seed twin `armC9s2_s*` as the floor (prompts are CJK-free).

| check | result |
|---|---|
| stamp on the LoRA | `ss_ext_pack_sha` = `2cf81cbc…` = the pack's digest; `ss_ext_pack` = pack stem |
| restaged TE caches (702 captions, 194 with 「」) | 7,656 tokens on the mirror, 1,843 on trained rows (delimiters + bare CJK), 0 `<unk>` |
| render distance, 64-px L1 to C9, 24 rows | C9ISOQ 0.075 ± 0.042 vs seed-twin 0.087 ± 0.042; ISOQ the closer one in 14/24 |
| colour saturation (mean; images < 0.12) | C9 0.255 (6/24) · C9s2 0.275 (6/24) · **C9ISOQ 0.208 (9/24)** · C9trigpol 0.209 (7/24) |

Inside the floor on the pixel metric; a mild grayscale/sketch tilt (3 more
low-saturation images than either C9 seed) that matches another C9 variant
(trigpol) and is not separable at n = 24 — rows are near-identical at seed
1234, and the rows that diverge at seed 42 (2, 7, 8) diverge between the
two C9 seeds too. Not a blind set; if the partition ever needs a ranking
claim, run one (s14 C9ISOQ vs C9), but D1's gate only asked for "inside the
floor" and it is. D2 proceeds through this pack.

What exists now (pointers, not repeats — contract in
`docs/experimental/cjk_ext_vocab_coverage.md` §"Quote partition"):

- `library/anima/ext_vocab.py`: `iso_block` / `IsoSpec` / `materialize_iso`
  (seed-regenerated isotropic mirror, NumPy legacy stream, byte-equal
  across machines), `Route.quotes` + `quote_spans` (one regex, non-nesting),
  `HybridT5Encoder.encode_cjk_run` (span rule before `segment_runs`; EN
  bit-identical by construction), `pack_digest`.
- `make_random_pack.py --mode iso | iso-partition [--no-iso-rows] [--norm]`
  (norm default = native T5 mean row norm 212.165, measured off the DiT's
  `llm_adapter.embed`; ISO1 had used the trained mean 203.9).
  Built: `output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256_isoq` (sha
  `2cf81cbc…`; mirror rows 69,558–139,116, PR 1009).
- Stamp: `train.py --ext_pack` → `ss_ext_pack_sha` / `ss_ext_pack`
  (`run_unmask_r2.py` passes it); `load_dit_model` warns on a stamped LoRA
  with no pack; Adapter node 3.10.0 compares digests in either node order
  (`vocab_pack.check_pack_vs_adapter`, `adapter._record_ext_pack_stamp`),
  regenerates seed-only blocks, cuts routed runs at quote boundaries in
  `VocabPackTokenizer`. Vendor tree re-synced; node committed locally, not
  yet pushed / registry-published (publish with the first partitioned pack).
- Grammar: anime_tools `efb235c` (`quoted_spans`; comma / `. On the` inside
  `「」『』""` is content; `compose_caption` round-trips) — pin bumped,
  `uv lock` + `uv sync` done. `cache_te_ext._quote_safe` keeps commas now.
- Tests: `tests/test_ext_vocab_iso.py` (determinism, EN bit-exact, quoted
  content only on the mirror with `「」`/`黒髪` staying on trained rows, three
  spellings → same ids, `"…"` order phrase routes, ASCII inside quotes stays
  spiece, digest invariance under regeneration, grammar, inference warn) +
  the earlier route tests: 58 passed.

Design choices worth knowing: the mirror is a full row-for-row copy (so
one id map serves both blocks; 285 MB fp32 shipped, or 0 bytes seed-only),
quoted content bypasses minted-word rows and the C fallback (those are
trained content), and the rule is inert unless *both* `iso` and
`route.quotes` are present — every existing pack, cache and blind set is
untouched.


## OCR eyeball + SFX handling (2026-09-05)

Contact sheets in `output/tests/ocr_contact_sheet/` (scratch, not tracked):
`sincos_ppocr_v2_sentence.pdf` (140 tiles = 133 masked ∪ 96 with PP-OCRv6 v2
lines; mask / SAM3-bubble tints, v2 boxes in reading order, the proposed
sentence caption, the tags actually trained), `bubble_kind.pdf` (one crop per
line, speech vs SFX with the balloon containment).

- **What trained (C2–C9)**: `mirror_sincos_ppocr` was built from the **v1**
  records (97 stems, `tags` format); v2's reading-order rewrite never reached
  a trained arm — 58 / 96 stems match v2 in order, 91 / 97 match v1 exactly.
- **PaddleOCR-VL-1.6, two masked-no-line pages** (`output/tests/vl16_single*/`):
  `12440144` — VL reads exactly one SFX (`ばるん`, the one column with solid
  fill) on the resized ×2 page and on the 3048×4080 original alike, and
  misses the six hand-lettered pink SFX (`ぱんぱん` ×2, `びくっ` ×2, `おっ`,
  `お`); resolution is not the limit. `6067089` — VL finds the right-hand
  balloon PP-OCRv6 dropped (`もしかして興奮してるー？`; Spotting rewrites
  興→無 on the original, page `OCR:` reads it right) and PP's `かわいいなー♡`,
  not PP's `ちらっ`. Same verdict as the A/B: VL is an extra detector, not a
  manga-SFX reader.
- **SFX handling** — `datasets/ocr_sfx.py` (torch-free, text-only rules:
  kanji / >6 kana / vowel-or-h-row initial → speech; repeated unit, voiced
  initial, lexicon onset, sokuon → SFX; optional `in_bubble` veto) and a
  `--ocr_format sentence` in `cache_te_ext.py`: the caption tail becomes
  `Japanese text reads as "…", "…". Japanese SFX reads as "…".` (speech
  first, SFX second, reading order inside each, ASCII quotes so the D1 span
  rule keys on them, native glyphs — the ext rows are the address). On the
  v2 records: 19 SFX / 209 speech lines, 15 of 96 captions get an SFX
  sentence. PP-OCRv6's SFX reads are the weak link (`でくv` for びくっ,
  `Kッ4vv`, `ゴmvvv`), and UI chrome (`ツイート`, `ポスト`, `完了にする`) is
  neither class. GOTCHA: the anime_tools grammar has no header for these
  sentences — a re-parse glues the first onto the last tag — so the append
  is string-level; a `TEXT_PREFIXES` clause kind in the package would make
  it grammar-native. Tests: `tests/test_cjk_ocr_captions.py`.
- **SAM3 `speech bubble` as the speech/SFX signal** (user's suggestion; run
  `-m anime_tools.masking.cli.generate_masks --prompts 'speech bubble'
  --focus-prompts none --dilate 0` → `output/tests/sam_bubbles/sincos/`):
  balloons found on **34 / 97** pages (median 5 % of the page), 69 of 228
  lines sit inside one; the veto changed **one** line vs the text rules
  (`バスト91`, a profile card). The rules already agree with the balloons
  where SAM3 finds them; where it misses (`カリカリ` in a plain rounded
  balloon, 6813398) the text rule is what's left. Outside-a-balloon is *not*
  SFX (narration, floating dialogue, chrome) — tried, added more errors than
  it fixed. The veto is in `line_kind(text, in_bubble=…)` but not wired into
  the mirror builder (records carry no balloon field yet).
- Side gotcha: `build_mirror` symlink creation fails sporadically on the
  ntfs3-mounted dataset volume (`FileNotFoundError` from `os.symlink` on a
  path that exists; `ln -s` by hand works) — the preview mirror was built
  under the session scratchpad; a dangling-link guard was added.

## B0 — hybrid OCR records on sincos: floor 44 → 27 (2026-09-05)

`plan_base1.md` B0. `datasets/build_ocr_records.py` (old tree, beside
`ocr_sfx.py`): PP-OCRv6 v2 records + PaddleOCR-VL-1.6 `Spotting:` on all 351
pages (×2 upscale, bs 8, `use_cache=True`, 3.5 min on the daemon, 4.5 GB
peak) + VL `OCR:` on every PP box (227 crops, bs 32). Raw VL outputs cached
in `post_image_dataset/cjk_unmask/ocr_raw_vl16_sincos.jsonl` so the merge is a
CPU re-run (`--stage merge`). Output `ocr_records_sincos_hybrid.jsonl` (326
lines / 118 pages; every record carries `kind`, `engine`, and `pp_text` /
`vl_text` / `rule1b` where a second read happened). Report with the two
tables (VL-only lines, rule 1b replacements) →
`reports/0905_b0_hybrid_records.md`; sheets
`output/tests/ocr_contact_sheet/sincos_hybrid{,_vl,_floor}.pdf`
(`probes/ocr_contact_sheet.py`, promoted; magenta = VL-only, orange = re-read).

| | PP-OCRv6 v2 | hybrid |
|---|---|---|
| pages with any line (351) | 96 | 118 |
| lines | 227 | 326 |
| **masked-but-no-line floor** (133 masked) | **44** | **27** |
| best-match sim to manga-ocr, 84 ref lines / 40 A/B pages | 0.752 (35 ≥ 0.9) | 0.772 (36 ≥ 0.9) |
| replaced lines only (14) | 0.456 | 0.451 |

**Gate: PASS** — floor down 17 pages, similarity ≥ PP (no regression from the
second reader), and the VL-only lines on the sheet are real balloons / SFX
(`ばるん`, `ぱんッぱんッ`, `ぶちゃぶちゃ`, `もじもじ`, `禁止ですよぶ？`,
`えっ…`), not chrome. 99 VL-only lines kept (60 dropped by the PP floors,
2 full-page quads, 1 duplicate quad). C10 runs on the hybrid records.

What the merge had to do differently from the plan text:

- **IoU 0.5 is too strict for columns.** A 30 px vertical column that VL and
  PP box 12 px apart is IoU 0.42 with byte-identical text; VL quads are per
  column while PP records are `join_cjk`-joined blocks. Fix: `join_cjk` the
  VL lines first, then match at IoU ≥ 0.3 **or** containment ≥ 0.5 **or**
  touching boxes with text sim ≥ 0.75. 198 / 227 PP lines matched a VL block.
- **Spotting hallucinates a page caption**: two quads covering the whole page
  (`だなんか` on [0,0,704,1487]) — dropped by an area gate (> 40 % of the
  page). Those two pages return to the floor, correctly.
- **Rule 1b needed three more guards** beyond the runaway / 2× length one:
  (a) never accept a read that loses a heart PP had (PP drops ♡, never
  invents it — 3 / 9 symbol disputes went the wrong way without this);
  (b) a *weak*-score re-read must agree (sim ≥ 0.5) with the matched Spotting
  read — two independent VL readings — else PP's text stays (`おいしそう` →
  `おぃ～う`, `ブルン` → `ぐにソ`, `先輩？` → `先非事？` were all rejected
  by this; `借てきたよ` → `借りてきたよ`, `特别` → `特別`, `おじーまっ` →
  `おじさまっ` pass); (c) a *symbol* dispute may move symbols only — a read
  that changes a letter (`ご主人様` → `ごー主人様`, `一発` → `ー発`) is
  rejected. Net: 56 weak + 9 symbol + 5 sfx re-reads → **37 replaced, 25
  rejected; 0 of the 9 symbol disputes survived** — the second reader's
  symbol job delivered nothing on sincos (the crop read either dropped the
  heart too or rewrote a letter). The replaced-lines similarity is a wash
  (0.456 → 0.451): mostly SFX garbage for SFX garbage, as the A/B predicted.
- `kind` is the **v1 text rule + a chrome word list** (251 speech · 67 sfx ·
  8 chrome); the rule's h-row miss is visible on the sheet (`はんv`, `はちゃ`,
  `ふくっ` land as speech) — B1's labels. The mirror builder
  (`cache_te_ext.ocr_lines_by_stem`) now drops `kind: chrome` records before
  any format sees them; the speech/SFX split still comes from the text rule
  until B1 wires `kind` through.
- Chore: `build_mirror` retries the symlink once, then hard-links, and says
  which happened (`_link_image`).

**Addendum (same evening) — two user decisions, both landed.**

1. **SFX lines are out of the captions for now.** The VL logic stays, the
   `kind: sfx` records stay in the file (B1 still labels them), but
   `cache_te_ext.ocr_lines_by_stem` drops `{chrome, sfx}` (`DROP_KINDS`) before
   any format sees them — no `Japanese SFX reads as` sentence, no `「ぱんぱん」`
   tag, until a reader can actually read hand-lettered onomatopoeia (both
   PP-OCRv6 and VL-1.6 garble it; a light OCR fine-tune on SFX crops is the
   likely route, not this plan's work). C10 therefore trains on speech only.
2. **Joined blocks keep their boundaries as a space.** 9410777 (a profile
   card: `椎名真昼ちゃん / 身長：156cm / おぱい：成長中 / すきなもの：…`) came
   out of PP as one glued string. `anime_tools.ocr._text._merge` now joins a
   block's columns / rows with `JOIN_SEP = " "` (package rev **cd75591**,
   pinned + `uv sync`; a space inside a vertical sentence costs a reader
   nothing, a lost list boundary is gone for good). Sidecars are post-join, so
   PP-OCRv6 was re-run on sincos (`-m anime_tools.stages.cli.ocr_captions
   --ocr_dir post_image_dataset/cjk_unmask/ocr_v3/sincos --apply`, daemon,
   ~1 min) and the builder grew `--sidecars` (records straight from
   `{stem}.ocr.txt`, gate 0.70 → `ocr_records_sincos_ppocr_v3.jsonl`) plus a
   box-matched alignment of the cached VL crop reads onto the re-derived
   records, so the GPU stage did not rerun. VL crop rows and VL-only blocks
   get the same space; VL's LaTeX wrapping of measurements
   (`身長: \( 156 \, cm \)`) is stripped.

Re-merged on v3 (`reports/0905_b0_hybrid_records.md` is this version):

| | PP-OCRv6 v3 | hybrid |
|---|---|---|
| pages with any line (351) | 103 | 123 |
| lines | 237 | 338 (138 carry a space) |
| **masked-but-no-line floor** (133 masked) | **38** | **23** |
| best-match sim to manga-ocr, 84 ref lines / 40 A/B pages | 0.751 (35 ≥ 0.9) | 0.786 (38 ≥ 0.9) |
| replaced lines only (13) | 0.485 | 0.512 |

The fresh PP pass alone already lands 7 more pages than the v2 file (sidecar
floor 0.6 vs whatever the old pass used; the 0.70 record gate is the same),
so the PP-alone floor is 38 not 44; hybrid takes it to 23. With row
boundaries kept, the replaced lines now move *toward* manga-ocr (0.485 →
0.512) instead of a wash — the second reader was being penalised for glued
rows, not for its letters. Rule 1b: 67 weak + 9 symbol + 6 sfx → 28 replaced,
51 rejected. Gate still PASS; C10 runs on these records with SFX dropped.

## B3 — arm C10: sentence captions on hybrid records pass the gate at the floor (2026-09-06)

Full write-up `reports/0906_c10_sentence_captions.md`. Blind
`s14_C10_vs_C9ISOQ` (48 pairs, fresh seeds 9/10/11): **C10 21 – C9ISOQ 15,
tie 12, rows 8-6** — flat inside the s02 seed-twin floor (pair p 0.41);
the grader picked side B in 31/36 decisive pairs (first set with a side
bias; sides balanced 24/24 so the arm total is unaffected in expectation,
and the flipped-convention reading 15–21 is also flat). Spam tally on the
8-row grids (lenient OCR flag + eyeball, ledger convention): **C10 ~2 / ~2
/ ~2 vs C9ISOQ ~2** — equality on all three seeds, the r6 s42 banner is a
base habit shared by every arm since C8. Tagger adherence recall C10
0.912 / 0.896 / 0.886 vs C9ISOQ 0.844 (back to C9's 0.912), driven by the
`comic` row. Plain-prompt spam probe: C10 25 % images with text vs
C9ISOQ 50 % (n = 12). **Gate PASS at the floor → sentence shape is D2's
default caption shape; no blind-visible gain claimed.** Instrument added:
`probes/grid_spam_tally.py` (CPU PP-OCRv6 flag pass over a grid dir set).

## O0 — SFX reader line: in-domain split + stock baselines (2026-09-06)

`plan_ocr.md` O0. Split = official COO `books_{train,val,test}` ∩ Manga109-s
`books.txt` → **74 / 7 / 6 books** (`assets/coo_split_manga109s.json`; the
COO lists are CRLF, `unsplit` = 0). Test crops built by
`ocr/build_manga109_crops.py --split test` (pilot `deskew_crop`, pad 12 %,
min side 16, orientation preserved): **2,558 COO lines** after truncation-link
joining (98 joined; the earlier "2,759 polygons" counted the `<onomatopoeia_link*>`
elements too) + **2,559 speech** `<text>` boxes, a per-book count-matched
draw (seed 0). SFX len p50 / p90 / max = 3 / 5 / 17, min side p10 / p50 =
33 / 76 px, 30 % vertical; speech 11 / 25 / 90, 44 / 88 px (bubble boxes are
multi-column, 6 % "vertical" by aspect). One crop dropped (< 16 px).

Scorer `ocr/eval_manga109.py` (daemon jobs `20260906-113412-{e51d7b,5a3a59,7d8758}`;
reports `reports/ocr_eval_{manga_ocr,ppocr,vl16}.md`, predictions
`output/ocr/eval/<reader>_test.jsonl`). **exact** = NFKC + whitespace-stripped
equality; **sim** = `build_ocr_records.sim` after `normalize_ja`.

| reader (stock) | SFX exact | SFX sim | SFX runaway | speech exact | speech sim | crops/s |
|---|---|---|---|---|---|---|
| manga-ocr-base | **16.4 %** (419 / 2,558) | 0.336 | 0 | 31.9 % | **0.824** | 270 |
| PP-OCRv6 rec (ONNX, upstream rotate rule) | 3.0 % | 0.093 | 0 | 0.9 %† | 0.082† | 355 |
| PaddleOCR-VL-1.6 crop `OCR:` (no guard) | **20.1 %** (515) | 0.449 | **105** (4.1 %) | 34.3 % | 0.845 | 40 |

† not a valid speech control: the speech crops are whole bubble boxes
(multi-column), which PP's single-line CTC head cannot read without its own
detector in front; it stays in the table as the "rec on the same crops" row
only. On the single-line SFX crops PP is what `plan_base1.md` observed — garble.

Reading the SFX rows:

- The published COO TRBA+2D baseline is 81.2 % on the full 10-book test; both
  stock readers are at 16–20 % on 6 of those books. That is the gap O2's
  fine-tune has to close, and it is not a punctuation artefact: punctuation-
  blind exact (also stripping `!?…・ー～`) is 486 / 2,558 (19.0 %) for manga-ocr,
  583 (22.8 %) for VL.
- manga-ocr degrades with length (24 % exact at 2 chars → 0 % at ≥ 7) and does
  worst on **horizontal** SFX (11.5 % vs 28.8 % square, 16.8 % vertical) — the
  speech-prior story from the sincos peek holds here: the worst lines are
  fluent speech (`パタパタ` → `「おめのお兄さん`, `カチーン!!` → `それは、それでも、`)
  or `...`.
- VL is 3.7 pts better on SFX exact and +0.11 sim, orientation-flat, but
  4.1 % of SFX crops and 6.8 % of speech crops run away (`ひと`×20,
  `えーーー…`) with plain greedy decoding — far above the A/B's 2 / 132 on
  sincos crops; a decode-time guard is a wiring knob, not a baseline
  correction, so the stock row stays unguarded.
- **Joined truncation lines are near-unreadable stock** (98 lines: manga-ocr
  1 exact, sim 0.11) — two glyph runs under one `minAreaRect`; the O1 builder
  keeps them (they are real COO test items) but the fine-tune may want them
  weighted down or cropped as parts.
- Speech control for O2: manga-ocr sim 0.824 / VL 0.845 are the "≥ stock −
  0.01" reference rows. The speech *exact* rate (32–34 %) is low because the
  boxes are multi-line bubbles and Manga109's transcriptions carry `‼`/`…`
  variants; 1,170 / 2,559 match punctuation-blind. sim is the control metric.

O0 gate: **PASS** (split written + asserted, three stock rows on the COO test
crops + speech control). Gotcha for every script on this line: the daemon
forwards only `ANIMA_`-prefixed env to its jobs, so the roots are
`ANIMA_MANGA109S_ROOT` / `ANIMA_ANIMETEXT_ROOT`. Still owed from O0, off the
critical path: the sincos hand-label draft (`assets/sfx_labels_sincos.tsv`)
+ `ocr/eval_sfx.py`, needed by the O2 gate.

## O1 — SFX reader line: COO + speech crops, all splits (2026-09-06)

`plan_ocr.md` O1. `ocr/build_manga109_crops.py --split train --split val
--workers 10` (books in parallel; 8 min wall, 97 CPU-min) on top of O0's test
cut. Same recipe throughout: pilot `deskew_crop`, pad 12 %, min side 16,
orientation preserved, speech = per-book count-matched `<text>` draw (seed 0).
Output `~/manga109s/derived/crops/<split>/<kind>/` (3.7 GB, 87,124 PNGs) +
`manifest.parquet` (one row per crop: split / kind / id / book / page / text /
joined / orient / w / h / poly / path). Never in-tree.

| split | kind | crops | joined | len p50 / p90 / max | min side p10 / p50 | vertical |
|---|---|---|---|---|---|---|
| train | sfx | 38,582 | 1,562 | 3 / 5 / 28 | 33 / 76 | 0.32 |
| train | speech | 38,634 | 0 | 11 / 25 / 243 | 44 / 89 | 0.08 |
| val | sfx | 2,395 | 64 | 3 / 5 / 16 | 30 / 64 | 0.34 |
| val | speech | 2,396 | 0 | 9 / 24 / 89 | 44 / 85 | 0.07 |
| test | sfx | 2,558 | 98 | 3 / 5 / 17 | 33 / 76 | 0.30 |
| test | speech | 2,559 | 0 | 11 / 25 / 90 | 44 / 88 | 0.06 |

**43,535 COO lines** kept (45,422 polygons − 1,724 truncation joins − 55
min-side drops, all in train/val) over 74 / 7 / 6 books; SFX orientation by
crop aspect: 49 % horizontal, 32 % vertical, 19 % square.

| text len | 1 | 2 | 3 | 4 | 5 | 6–8 | 9–12 | 13+ |
|---|---|---|---|---|---|---|---|---|
| sfx | 6.6 % | 41.5 % | 27.3 % | 11.5 % | 6.2 % | 6.1 % | 0.6 % | 0.1 % |
| speech | 1.5 % | 4.9 % | 5.2 % | 6.1 % | 5.4 % | 16.1 % | 18.3 % | 42.5 % |

Char coverage against manga-ocr's WordPiece vocab (`vocab.txt`, `##` stripped):

- **SFX: 175 / 181 chars, 99.90 % of occurrences** (128,798); missing
  `゛ ゔ ♫ ♬ ゜ ♩` — the O0 table's number reproduced on the built crops.
  Inventory 67.8 % katakana / 29.2 % hiragana / 3.0 % symbol; 99 heart lines.
- **Speech: 2,571 / 2,689 chars but only 93.8 % of occurrences** — the misses
  are *not* glyphs: full-width `！ ？ ～ ･ ‼ １２３ ＡＮ（）`, `…`/`‥`,
  the ideographic space (930 rows) and **newlines (1,989 rows — Manga109's
  `<text>` transcriptions keep line breaks)**. → **O2 target rule:** NFKC-fold
  + strip all whitespace before tokenising (the same fold the scorer's `exact`
  already applies), so the replay set trains the vocab it has instead of
  emitting `[UNK]` on a twentieth of the speech characters. `…` → `...` under
  NFKC is in vocab.
- 1 : 1 by *count* is 5.4 : 1 by *characters* against SFX (speech p50 11 vs 3)
  — decision 2's caveat quantified; the 1 : 2 count arm would be ~11 : 1 by
  tokens, so if the speech control slips, weight by tokens rather than
  re-drawing.

**Augmentation (decided here, applied at train time)** — `ocr/augment.py`,
`Augment(seed)` on the BGR crop, independent Bernoulli draws: pad jitter
5–25 % (p .8; inward cut / outward border-colour fill around the fixed 12 %
crop), ±8° rotation (p .5), colour tint (p .35: darkness → alpha, strokes in
pink / red / plum / black over skin / pastel / pink backgrounds, 15 % of tints
white-on-dark), gamma 0.6–1.5 ± 30 levels (p .6), invert (p .1), scale
0.5–1.0 down-up (p .4), JPEG q 30–95 (p .5). `--demo` contact sheet at
`~/manga109s/derived/aug_demo.png` checked by eye: the tint pass yields the
pink-on-skin surface sincos has; outward pad shows as a flat frame (border
median), acceptable. The tint is the cheap half of decision 4 — the colorized
COO lever (O3) still owns backgrounds with real art.

O1 gate: **PASS** (43,535 ≥ 40k COO crops; test 2,558 ≥ 2k). Next: O2 on
the daemon — `finetune_manga_ocr.py` first (the default base), the VL-1.6
crop LoRA second; the sincos label draft + `eval_sfx.py` remain owed before
either gate is read.

## O1 correction — `deskew_crop` transposed every axis-aligned box (2026-09-06)

Found while standing up the sincos gate (`ocr/eval_sfx.py`): stock manga-ocr
read the hand-labelled SFX crops at 0 / 99 and the speech control at sim
0.34, and the dumped crops were strips through one glyph. Cause: OpenCV ≥ 4.5
`minAreaRect` reports an axis-aligned 30×120 box as size **(120, 30) at 90°**;
the pilot's `deskew_crop` (`../cjk_aware_anima/datasets/manga_text.py`) took
`angle − 90` without swapping the extents, so every polygon whose reported
angle was > 45° — all Manga109 `<text>` boxes, all sincos record boxes, and a
large share of COO polygons — was cropped as a **transposed rectangle** around
the right centre. In the O1 manifest 88 % of speech boxes were taller than wide
but only 17 % of the crops were. Fixed (swap `w, h` with the angle), all
87,124 crops rebuilt with `--overwrite` (9 min; vertical share now 0.57–0.71
instead of 0.06–0.34; 56 min-side drops), and **every O0 stock row, the smoke
runs and the first O2 launches were discarded** — the O0 numbers above are on
the transposed crops and are superseded by the re-run rows in § O2. The
pilot-era sincos "~12 / 71" (`plan_base1.md`, by-eye against the records)
was produced through the same function and is not a clean reference either;
the gate now reads `eval_sfx.py`'s strict exact (hearts count) beside a
heart-blind exact.

## O2 — SFX reader fine-tunes, both bases (2026-09-06): in-domain PASS, doujin gate MISS on both → O3

*Superseded the same day by § O2b below: unfreezing VL's vision tower passes the doujin gate outright, so O3 is no longer on arm B's path.*

`plan_ocr.md` O2 on the **corrected** O1 crops (§ O1 correction). Scorers:
`ocr/eval_manga109.py` (COO test + speech control) and `ocr/eval_sfx.py`
(sincos hand labels, `assets/sfx_labels_sincos.tsv` — 338 rows drafted off
the contact sheets, **corrected by the user 2026-09-06** (323 checked, 15
draft): 99 `sfx` / 213 `speech` / 26 `chrome`; 34 records the v1 rule
called speech are hand-lettered SFX by eye (`ぱん♡` read as `はんv` etc.), 17
are overlay captions / signs / clothing print → `chrome`; 6 of the 71
`kind: sfx` records are speech / chrome by eye). `exact` folds `♥→♡`,
`〜→~`; a **heart-blind exact** rides beside it because manga-ocr almost never
emits `♡` and the pilot's "~12 / 71" was counted without hearts (and through
the transposing crop).

**Corrected O0 stock rows** (COO test, 2,558 SFX + 2,559 speech; replaces the
transposed-crop table in § O0):

| reader (stock) | SFX exact | SFX sim | runaway | speech exact | speech sim | sincos gate / 71 (♡-blind) | sincos SFX sim | sincos speech sim |
|---|---|---|---|---|---|---|---|---|
| manga-ocr-base | 26.2 % | 0.478 | 0 | 62.1 % | **0.975** | 2 (4) | 0.315 | 0.646 |
| PP-OCRv6 rec | 7.2 % | 0.194 | 0 | 13.0 %† | 0.297† | — | — | — |
| PaddleOCR-VL-1.6 crop `OCR:` | **30.2 %** | 0.545 | 331 | 63.4 % | 0.976 | 2 (6) | 0.464 | 0.856 |

† single-line CTC head on multi-line bubble crops — not a valid speech row.
With real crops the speech control is 0.975 (was 0.824 on transposed crops)
and stock manga-ocr is at 26 % on COO (was 16 %).

**The two arms** (train 77,164 crops 1 : 1, val 4,791 each epoch):

| arm | recipe | wall | val SFX exact stock → best | COO test SFX exact | COO speech sim | COO runaway | sincos gate / 71 (♡-blind) | sincos SFX sim (99) | sincos speech sim |
|---|---|---|---|---|---|---|---|---|---|
| A · manga-ocr lr 2e-5 | full FT, bs 64, 4 ep | 20 min | 32.8 → 73.3 | 71.2 % | 0.975 (= stock) | 0 | 9 (13) | 0.667 | 0.747 |
| A · manga-ocr lr 5e-5 | same | 20 min | 32.8 → **74.9** | **73.5 %** | 0.975 (= stock) | 0 | 10 (12) | 0.664 | 0.721 |
| B · VL-1.6 LoRA lr 1e-4 | r 16 on 126 LM proj (6.0 M), bs 16, 2 ep | 85 min | 33.9 → 66.2 | 64.7 % | **0.981** | **194** (24 sfx + 170 speech) | **13 (19)** | **0.698** | **0.889** |

Gate (per base): COO test reported ✓; sincos SFX exact ≥ 35 / 71 ✗ (9–13);
sincos speech ≥ stock − 0.01 ✓ (both up); COO speech ≥ stock − 0.01 ✓.

Reading it:

- **In-domain: fine-tuning works, manga-ocr wins it.** 26 → 73.5 % COO test
  exact in 20 GPU-min, 8 pts under the published TRBA+2D (81.2 % on the
  10-book test), speech control untouched, no runaways. VL's LoRA reaches
  64.7 % in 4× the wall and keeps its runaway class (194 on the test crops,
  168 on val speech even after tuning) — a decode guard would be mandatory
  before any wiring. Both arm-A curves were still climbing +2 pts/epoch at
  epoch 4; the 8-epoch run was cancelled to keep the day on the gate.
- **Out-of-domain: the doujin gap is real; both bases miss the gate.**
  sincos gate 2 → 9–13 / 71 strict, 4 → 12–19 heart-blind, against ≥ 35.
  Arm A trips the kill clause literally (< 25 while COO ≥ 70 %) → **O3 is
  mandatory before wiring**; arm B is under both thresholds. The residual is
  no longer garbage (99-row SFX sim 0.31 → 0.67–0.70; half the rows at sim
  ≥ 0.8): manga-ocr reads `びくん` for `びく♡`, `ぱんッ` for `ぱん♡`, `ガクン` for
  `ガク♡` — the heart decoded as the katakana ending COO taught (`ン`/`ッ`) —
  plus pink-outline confusions (`ぱ/は/ば`, `ぶっ`, `くにくに`). VL keeps more
  hearts (its strict/♡-blind gap is 6 rows vs manga-ocr's 2–4 but from a
  higher base) and reads the sincos *speech* far better (0.889 vs 0.75),
  which is the pink hand-lettered bubbles. This is the surface decision 4
  predicted: lettering style + hearts → **synth doujin SFX first** (hearts
  at sincos' rate, outlined kana over doujin backgrounds), colorized COO
  second.
- **Pick for O3:** run O3 on **manga-ocr** first (10× cheaper per crop, no
  runaways, higher COO, gate within noise of VL's); VL rides along only if
  synth + colorized lift manga-ocr short of the gate, since its native heart
  handling is the one thing it does that rules would otherwise have to.
- 1 : 1 by count held both speech controls, so the 1 : 2 arm is not needed.

Arm B engineering note: the first launch OOMed in the loss — the native
forward materialises fp32 logits over the 103k vocab for every image token,
4.7 GB on a large-crop batch — fixed by left-padding and `logits_to_keep` =
target length (CE on the suffix only; peak 13 → 3 GB, ~35 crops/s).
Deployment shape for a VL pick would be torch + remote modeling files + the
batching rules + a runaway guard — the tie-break decision 1 already makes.

O2 gate: **in-domain PASS, doujin gate MISS on both bases** (`findings` rows
above are the O3 reference). Artifacts: `output/ocr/{mocr_lr2e-5,mocr_lr5e-5,vl16_lr1e-4}/best`,
`reports/ocr_eval_{manga_ocr,ppocr,vl16,mocr_lr2e-5,mocr_lr5e-5,vl16_lr1e-4}.md`,
`reports/ocr_eval_sfx_*.md`. The VL adapter is on the Hub as a **private**
research checkpoint: `sorryhyun/paddleocr-vl-1.6-manga-sfx-lora` (renamed `…-manga-lora` and made public with the O2b weights, see § O2b; model card
carries the recipe, both eval tables, the runaway caveat and the Manga109-s /
COO citations; adapter weights only). Next *as written then*: O3 synth on arm A — overtaken by § O2b.

## O2b — arm B′: VL-1.6 LoRA + vision-tower full FT (2026-09-06): doujin gate PASS, VL is the pick

The frozen tower was the bottleneck. Same crops, mix, LoRA and lr as arm B,
plus the NaViT tower + projector trained in full (fp32 master copy, lr 1e-5,
439 M params in 443 tensors; `finetune_vl16_lora.py --train_tower --tower_lr
1e-5`, bs 8 × grad-accum 2 = the same effective 16, **1 epoch** = 4,822 steps,
~90 min, 12.1 GB peak). Val SFX exact 86.2 % after the single epoch (arm B
reached 66.2 % after two). Eval jobs `20260906-161652-{b5ba49,c41369}`,
reports `reports/ocr_eval_{sfx_,}vl16_tower_lr1e-5.md`.

| arm | COO test SFX exact | COO SFX sim | COO speech sim | COO runaway | sincos gate / 71 (♡-blind) | sincos SFX exact / sim (99) | sincos speech sim (213) |
|---|---|---|---|---|---|---|---|
| VL-1.6 stock | 30.2 % | 0.545 | 0.976 | 331 | 2 (6) | — / 0.464 | 0.856 |
| A · manga-ocr lr 5e-5 | 73.5 % | 0.884 | 0.975 | 0 | 10 (12) | — / 0.664 | 0.721 |
| B · VL LoRA, tower frozen, 2 ep | 64.7 % | 0.816 | 0.981 | 194 | 13 (19) | 13.1 % / 0.698 | 0.889 |
| **B′ · VL LoRA + tower FT, 1 ep** | **81.7 %** | **0.927** | **0.986** | 189 (25 sfx + 164 speech) | **38 (41)** | **45.5 % / 0.868** | **0.910** |

**O2 gate, arm B′:** COO test reported ✓ (81.7 %, at the published 81.2 % on
our 6-book subset); sincos SFX exact ≥ 35 / 71 ✓ (**38**); sincos speech sim ≥
stock − 0.01 ✓ (0.910 vs 0.856); COO speech sim ≥ O0 stock − 0.01 ✓ (0.986 vs
0.976). **PASS** — the first arm to pass the doujin gate, without O3.

Reading it:

- **The domain gap was a tower problem, not a decoder-prior problem.** Arm B
  moved the in-domain number and barely the doujin one; letting the tower see
  the crops does both in one epoch (13 → 38 / 71, sim 0.75 → 0.90). The
  decoder-side lever the plan queued for O3 (synth outlined kana, colorized
  COO) is not needed to pass; it stays available as a *lift*, not a rescue.
- **Hearts are read natively.** Strict vs ♡-blind gap is 3 lines (38 / 41);
  misses are mostly `♥` for `♡`, which `exact` already folds. Decision 6's
  heart-patching rule is moot for this pick.
- **Decision 1 resolves to VL.** It passes and removes the heart rule, so it
  wins outright (the tie-break to manga-ocr never engages). Cost accepted:
  ~10× manga-ocr's wall per crop, deployment = torch + remote modeling files +
  adapter 24 MB **+ tower 878 MB**, and a runaway guard is mandatory before
  wiring (189 on COO test, `びく♡` → `ぐくーーー…` on sincos; the count is left
  unguarded in every table on purpose).
- **Residual** for a later lift: 8+-char lines 0 / 5, square multi-line SFX
  blocks (17 rows, 0.59 → weakest orientation), `ぱん♡` family
  (`ぱィ♥` / `ぱ人♡` / `ぱく`). The curve was still rising at epoch 1; a 2–3
  epoch run and a tower-lr sweep (3e-6 / 3e-5) are the cheap next arms if O4
  wants more margin, but neither gates O4.

Published: **`sorryhyun/paddleocr-vl-1.6-manga-lora`** (public, 2026-09-06;
the `…-manga-sfx-lora` repo renamed in place, old URL redirects) — adapter +
`tower.safetensors` + card with both eval tables, the two-step load (peft
merge, then `load_state_dict(strict=False)` of the tower), runaway caveat,
Manga109-s / COO citations. Weights only; no crops.

Next: O4 — `build_ocr_records.py --sfx_reader` with the VL reader + a decode
guard, `anime_tools.ocr.sfx` in the VL deployment shape, re-measure the sincos
floor, then arm C11. O3 levers are demoted to optional lift.

## O4 — the SFX reader wired in: `anime_tools.ocr.sfx`, records re-read, floor 23 → 8 (2026-09-06)

`plan_ocr.md` O4, first half (the records + the package); arm C11 is
running (§ O4b when its grids land).

**The package.** The reader ships as `anime_tools.ocr.sfx.SfxReader`
(anime_tools **46ebbb5**, pinned + `uv sync`; `peft` is a package dependency
now): B′'s weights from two catalog rows — `vl16_base`
(`PaddlePaddle/PaddleOCR-VL-1.6`, 1.9 GB, `models/paddleocr_vl_1.6`) and
`sfx_reader` (`sorryhyun/paddleocr-vl-1.6-manga-lora`, adapter 24 MB + tower
878 MB, `models/paddleocr_vl_1.6_manga_lora`) — fetched on first load (the
Hub path verified: `eval_sfx.py --reader sfx` with no `--ckpt` downloaded,
merged and reproduced **38 / 71**, 15.9 crops/s at bs 16). A crop reader
only (`read` / `read_boxes`); no stage uses it yet, so the rows carry no
`stages`. The rest of O4's wiring lives in the dit tree as
`ocr/reread_records.py` (a new script rather than a flag on the 900-line
`build_ocr_records.py`; it imports that file's `overlap` / `record_kind` /
`floor_count`).

**The decode guard is area-tied, not aspect-tied.** The first guard capped a
read at `4 × longer/shorter + 6` characters: it held the SFX gate (38 / 71)
and silently threw away 60 % of the *speech* reads (a multi-column balloon
block is square and holds 20 characters; sincos speech sim 0.910 → 0.454),
and `max_new_tokens = 32` truncated the long lines (the tokenizer spends ~1
token per CJK character; speech runs to 57). Shipped: cap = crop area / (16
px)², floor 12; 80 new tokens; the repetition test (`is_runaway`, unchanged
from B0) owns the runaways. On the 338 hand labels the guarded reader is
speech exact 60 / sim 0.865 / **0 runaways** vs 59 / 0.910 / 11 unguarded —
the 0.045 is the eleven runaways scoring empty instead of half-right, and in
the pipeline a rejected read keeps the previous text, so nothing is lost
there. `guard` runs at apply time on the cached raw decode, so a guard
change never costs a GPU pass.

**Records** (`ocr_raw_sfx_sincos.jsonl`, one GPU pass: 486 crops = the 338
hybrid records + 148 MIT-mask components, 14 crops/s; `--stage apply` on
CPU). `kind` now comes from the **hand labels** for the 338 matched records
(B1's file wired through: `kind_src: hand`) and the v1 rule elsewhere.

| | PP-OCRv6 v3 | hybrid (B0) | **+ SFX reader, `--reread sfx`** | `--reread all` |
|---|---|---|---|---|
| pages with any line (351) | 103 | 123 | **138** | 138 |
| lines | 237 | 338 | **448** | 448 |
| **masked-but-no-line floor** (133) | 38 | 23 | **8** | 8 |
| best-match sim to manga-ocr, 84 ref lines / 40 A/B pages | 0.751 (35 ≥ 0.9) | 0.786 (38) | 0.800 (39) | **0.810 (42)** |
| sincos gate, 71 `kind: sfx` records, exact (♡-blind) | — | 4 (10) | **37 (40)** | 37 (40) |
| hand-SFX rows (99) exact / sim | — | 1 / 0.479 | **44 / 0.873** | 44 / 0.873 |

`--reread sfx`: 99 SFX records re-read → 97 replaced, 1 guard-rejected
(`びくひく・・・・・♡` for `ぐくぐく…`, area cap). Mask components: 148 cropped
→ **110 added** (7 guard, 25 under the 2-char floor, 6 symbol-only); the
sheet (`output/tests/ocr_contact_sheet/sincos_hybrid_sfx.pdf`, blue = mask
component) shows real lettering — `びくっ`, `ドチュ♥ドチュ♥`, `パシッ♡`,
`くにくに♡`, `ぬぽっ♡ / ぬぱっ♡`, `ムラッ ×2` — plus a tail of 2-glyph reads
(`ハハ`, `ハン`, `あ♥`) the rule calls speech. 15 floor pages recovered; the
8 still empty (10542078 10732203 11883907 14068612 14216300 6437445 9410775
9830919) have mask components under 32 px or reads under the floor. Kind
over the file: speech 254 · sfx 168 · chrome 26.

**"Just run all of OCR through VL" (user, mid-session) — measured, and it
wins modestly.** `--reread all` replaces 298 of 338 records (11 rejected):
the speech rows *cannot* be judged on the hand labels (their `text_hand` is
the record text unless obviously wrong, so the incumbent scores 0.998 by
construction), but on the independent manga-ocr reference the all-VL file
is the best of the four (0.810 / 42 ≥ 0.9 vs hybrid 0.786 / 38), and the
replacements read as fixes — hearts restored (`センパイ♥おなほの…♥`, `も~♡特別
だよ~?♡`), `ムうムう` → `ムラムラ`, `おち人ぽ` → `おちんぽ`, `おじさLちLぽ` →
`おじさんちんぽ` — with a few regressions (`我慢できない` → `でさない`, one
garble for another on 10792115) and the B0 space between joined columns
dropped (VL reads a block as one string). Decision: **C11 stays
single-variable** (SFX-only re-read + the SFX sentence, so the blind set
isolates the caption shape); the all-VL records are the **D2 records
recommendation** on B0's own metric (floor equal, reference sim up), and a
C11-on-all-VL seed is the cheap follow-up if C11 passes.

**C11 launched — one training seed by the user's call** (job
`20260906-165905-d6eb86`, s42; the s7 / s1234 jobs were queued and killed
unstarted, 2026-09-06 17:05 — "시드 하나만 하자"; ~1 h): C10's recipe on
`ocr_records_sincos_hybrid_sfx.jsonl` with `--keep_sfx` — `DROP_KINDS`
loses `sfx`, the caption gains `Japanese SFX reads as "…"` after the speech
clause from the records' `kind` (`cache_te_ext.ocr_records_by_stem` /
`ocr_text_clauses(kinds, sfx_sentence)`, tests in
`tests/test_cjk_ocr_captions.py`); 87 of 132 captioned stems carry an SFX
clause. Config `configs/gui-methods/custom/cjk_unmask_c11.toml` (the `_s7` / `_s1234` twins exist, unused).

**O5 parked (user's call, 17:15 — "O5는 일단 냅두자").** The segmenter is
written and CPU-smoked, not trained: `ocr/kind_seg.py` — a
`segmentation_models_pytorch` U-Net (`resnet34`) over Manga109-s spreads,
classes bg / speech (`<text>` fill) / sfx (COO polygon fill, wins on
overlap), 768-px crops at native resolution, weighted CE + dice, with
box-level evals (`eval-val` / `eval-sincos`: kind accuracy and SFX recall
against the hand labels beside the v1 rule, SFX components on the 133
masked pages beside the O4 reader) and an ONNX export. The five queued jobs
(`20260906-170305-*`) were killed unstarted. Resume = the four commands in
its module doc, ~1–2 GPU-h.

## O4b — arm C11 (SFX sentence on the SFX-reader records) vs C10: spam equal, blind set owed (2026-09-06)

One training seed (s42, user's call), grids `output/tests/cjk_unmask_eval2/armC11_s{42,7,1234}`,
final avr_loss 0.074 (C10's band). Control = C10 s42 (same pack, latents,
recipe; speech-only sentence).

**Spam tally, 8-row grids × 3 render seeds** (`probes/grid_spam_tally.py
reports/grid_spam_tally_c11.json`, lenient PP-OCRv6 flag pass + eyeball with
the ledger convention):

| arm | events | cells |
|---|---|---|
| C11 (s42) | ~2 | s42 r6 maid-café banner (`Maife` + menu board); s7 r3 hug with three JA speech bubbles + hearts. s7 r6 chalkboard menu is diegetic and **identical in C10's s7 r6**; s7 r8 comic row (bubble `きすてー` + scribbled SFX) excluded by convention; s42 r7 "59 % glyph" flag is one false box on a clean portrait |
| C10 (s42) | ~2 | s42 r6 banner; s7 r3 hug with two JA bubbles |

Same two base-habit cells, **C11 ≤ C10 → spam gate half PASS**. The extra
bubble on s7 r3 (three vs two) is the one place the SFX sentence might be
visible; a single cell at one seed is not a signal. s1234 clean for both.

**Tagger adherence** (`reports/unmask_grid_judge_c11.md`): C11 prob 0.736 /
recall 0.869 vs C10 0.734 / 0.912 (C10's three seeds: 0.912 / 0.896 /
0.886); the recall gap is the `comic, 2koma` row (0.58 vs 0.83), the row
C9ISOQ also lost, at n = 3 renders. cos→base 0.979 vs 0.984, cos→sincos
0.903 vs 0.905: neither arm moved.

**Blind set `s15_C11_vs_C10`**: 24 pairs (8 rows × render seeds 12 / 13 / 14)
(`regrid_set.py`, job `20260906-173543-4855a4`, pushed to the private
pairs repo) — **the user grades**; gate half 2 (blind ≥ C10 inside the
seed-twin floor, s02 15–9 on 24) reads off `probes/blind_pairs.py score
--set s15_C11_vs_C10`.

**s15 GRADED 2026-09-06** (user pasted one 24-char a/b string, `-` = tie;
private repo commit d604a27; `reports/blind_s15_C11_vs_C10.md` in the old
project dir): **C11 11 – C10 9, tie 4** — flat, inside the seed-twin floor
(s02 15–9 on 24). Rows split 3–3 (C11 sweeps r1 3–0 and edges r5 / r7; C10
edges r4 / r6 / r8; r2 / r3 mixed with ties). Grader sides balanced (A 11 / B
9), so the s14 side-bias flag does not apply here. Reading: the SFX sentence
neither helps nor hurts on 24 pairs — **blind half of the gate PASSES as
written** (blind ≥ C10 inside the floor), with the caveat that the spam half
was measured on one training seed (s42, the user's call), not the three the
plan asked for. **`DROP_KINDS` flipped** (user's call, same evening):
`cache_te_ext.DROP_KINDS = {chrome}`, the SFX sentence is the default
(`--keep_sfx` stays as a parsing no-op), `--drop_sfx` (also on
`run_unmask_r2.py`) reproduces the C2–C10 caption via `SFX_DROPPED`; the
closed text-binding probe pins `SFX_DROPPED` so its caption is unchanged.
C11's caption is D2's default from here.

## O4c — SFX dedupe + all-VL records as the default, no arm (2026-09-06, night)

Two user calls after eyeballing the SFX caption sheet
(`probes/sfx_caption_sheet.py`: crop as the reader saw it → read / raw /
previous text → caption clause, `output/tests/ocr_contact_sheet/sfx_caption*.pdf`).

**1. "쥬포 쥬포쥬포 는 빼도 될듯" — the SFX clause is deduplicated.**
`ocr_sfx.dedupe_sfx` (torch-free, `sfx_key` = kana core minus sokuon /
long-vowel marks folded to its minimal repeating unit; first in reading
order kept; speech never deduplicated) runs inside
`cache_te_ext.ocr_text_clauses`. On sincos: 168 SFX lines → 148, 17 of the
87 SFX captions change (`じゅぽ, じゅぽ, じゅぽじゅぽ` → `じゅぽ`; `パン♥, パン♥`
→ `パン♥`; `ぱちゅ, ぱちゅ♡` → `ぱちゅ` — the key folds hearts, so the first
read's decoration wins). Tests in `tests/test_cjk_ocr_captions.py`.

**2. "let's make vl be default … we don't really have to do another c arm
test" — the all-VL re-read is the records default.** Every line through the
fine-tuned VL reader (`--reread all`, detection still PP-OCRv6 + VL
Spotting boxes; 11 of 338 reads guard-rejected keep the old text). Against
the SFX-only file (both deduped): 82 of 132 captions and 184 lines differ —
90 only spacing / halfwidth `?!~` / `・・・` / hearts, 94 real character
changes, mostly repairs (`ムうムう` → `ムラムラ`, `オプパコ` → `オフパコ`,
`温水く人！！` → `温水くん!!`, `トしーナーちん` → `トレーナーちゃん`, `狼狽地…前驚`
→ `狠狠地…前輩♥`) with a regression tail (`バスト91` → `バスト9`, `団体様 2名`
→ `団体様20名`, `アリッ` → `フリ♥`, `ピストン` → `ビストン`, `おまんこ` →
`おまんご`). Sheet `output/tests/ocr_contact_sheet/vl_vs_sfx.pdf` (119
tiles, changed crops + the SFX-only caption struck under the all-VL one),
list `vl_vs_sfx_lines.txt`. Defaults flipped: `reread_records.py --reread
all` → `ocr_records_<shard>_hybrid_vl.jsonl`; `cache_te_ext.py --records`
default = that file; `run_unmask_r2.py` defaults = hybrid_vl records +
`mirror_sincos_hybrid_vl_sentence` + `te/sincos_hybrid_vl_sentence_isoq` +
the isoq pack + `sentence`. Both caches built (351 / 0 failed; the deduped
SFX-only pair `mirror_…_sfx_sentence_dd` / `te/…_sfx_sentence_dd_isoq` exists
too). C11's config + mirror stay as trained. **Package side (same night):** `anime_tools`'s OCR stage gained
`--reader {ppocr,vl}` + `--mask_dir` / `--comp_min_side` / `--comp_max` /
`--vl_batch_size` (`anime_tools/ocr/reread.py`: `reread_lines` +
`RereadEngine`, the dit tree's `reread_records.py` logic minus the hand
labels and `kind` — the sidecar has no kind column; a VL-only mask line
carries score `0.000`). Torch stays out of `run_ocr` (`_vl_engine` helper,
the ONNX-device pin holds); 953 package tests pass; smoke on six sincos
pages through the daemon reproduced the records' lines (`ドチュ♥`, `びくっ`,
`ムラッ` ×2, `じゅぽ` ×3, `ブルン♥`). One artefact to know: a PP box of
screentone (`10985746`, 0.62) that VL reads as `s v .l √2` passes the floors
— the dit pipeline's rule-1b / symbol filters are not in the stage.
Uncommitted in `../anime_tools`; pin bump + `uv sync` owed before the
trainer's `make` wrappers can reach it.
