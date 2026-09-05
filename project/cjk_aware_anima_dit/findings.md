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
