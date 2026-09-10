# plan_render — can the DiT *write* KO/JA from ext rows? (2026-09-10)

A feasibility probe, not a method. Every arm this line ran (unmask, OCR-vs-PLAIN,
SHIPRAND-vs-SHIP) was a setting where the caption's text was never needed to
solve the loss, and the verdict — rows are addresses, content is inert
([`findings.md`](findings.md) § s21) — is a verdict about *those* settings.
Nothing has ever forced the DiT to read an ext-row sequence and render its
glyphs. This plan builds the smallest setting that does.

**The task.** Bubble fill as an EasyControl condition: cond = a manga crop with
its speech-text boxes gray-holed, target = the same crop intact, caption = the
tag bag plus the holed lines as a text clause. The transcript is the only
source of the glyphs; the loss cannot be solved without reading it, or by
memorising the page (see R2 for why that escape hatch must be measured, not
assumed shut).

Corpus: the paired-edition corpus (plan principle 9 — never pathed, listed or
sampled here; its own docs carry the detail). It has JA / EN / KO editions of
the same pages, so the **EN edition is a free positive control**. The first
pass is **EN + JA** (2026-09-10, the user's call: most pages are Japanese and
the question is the ext rows, not one script); KO follows once EN/JA read. It
has no ZH edition; this plan does not answer ZH.

The real use this feeds: an EasyControl i2i that fills the bubbles of a manga
panel — so a training sample is a **panel** (cut) with its bubbles holed, not
a window around one bubble.

## What exists (reuse, don't rebuild)

| piece | where | state |
|---|---|---|
| EasyControl inpaint task (gray hole → fill, caption steers) | `configs/easycontrol/inpaint.toml`, `easycontrol_adapters/inpainting/prep.py` | shipped; mask stage is LaMa-style random strokes |
| text-removal twin task (the reverse direction) | `configs/easycontrol/near_twins.toml` | shipped; shape reference only |
| detector | `anime_tools.ocr.animetext.AnimeTextDetector` (yolo12l @ 640, conf 0.25) | shipped, line boxes not bubble boxes |
| KO reader | `stock` PaddleOCR-VL-1.6 (`ocr/pseudo_label.py` `Vl16Sweeper`), spacing-preserving | plan2 K2 teacher; hayai is voter only |
| text clause grammar | `anime_tools.captions.position_clauses.text_clause` | **JA-hardcoded**: `TEXT_PREFIXES = ("Japanese text reads as ", "Japanese SFX reads as ")` — left as is for now (research run; post-process the exported captions trainer-side) |
| pack-aware tokenizer | `library.anima.vocab_pack.make_tokenize_strategy` | production path; **inpaint prep's `stage_text` is pack-aware since 2026-09-10** (`--vocab_pack`, config default when omitted; CPU-verified: KO clause routes to ext ids, EN bit-exact). `colorization/` and `region/` preps still build the stock strategy |
| panel (frame) detector | `models/manga109_yolo/` = `deepghs/manga109_yolo` v2023.12.07_l_yv11 ONNX (`frame` class, F1 0.92) | fetched ad hoc for the probe, not a catalog row; white-gutter XY-cut was tried and fails on this corpus's full-bleed colour pages |
| S0 / S0b script | `render/corpus_boxes.py` — `det` / `read` / `bubbles` / `panels` / `sheet` | outputs under `output/render/<name>/`, keyed by corpus-relative path; `bubbles` needs `--corpus` since S0b (balloon probe) |
| published pack | `output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256` | KO/ZH rows trained, never render-validated |
| geometry-matched random pack | `output/ckpt/cjk_vocab_pack_random_r256` | the s21 control; same json/rows/geometry, random content |
| CER judge | `../cjk_aware_anima/probes/text_bind_judge.py` | PP-OCRv6, JA only; the CER/NFKC/montage scaffolding is reusable |
| inference | `make test-easycontrol` `_ADAPTERS` table (`scripts/tasks/inference.py`) | needs a `render` row |

## Design decisions (fixed before anything runs)

1. **Crops are panels (Manga109 `frame` boxes), multi-bubble from v0.** A
   page with no detected frame is one whole-page sample. Bubbles inside a
   panel are listed in manga reading order (right→left, then top→bottom). A
   bubble assigned to no panel is dropped.
2. **The hole is the text box (the AnimeText block), not the balloon.** The
   outline stays as a position cue and the task stays glyph rendering. Text
   boxes the guard rejects (Japanese left in an EN page, SFX, credits) are
   **left intact**: they sit in both cond and target and are copied through.
   A *holed* box without its own clause is what must never happen.
3. **768 tier, never 512.** § 512-is-not-a-text-resolution
   ([`findings.md`](findings.md)): at 0.5× downscale 18 % of glyph columns
   fall under 16 px. Gate each sample on its **post-resize glyph height ≥
   20 px** (block height ÷ the read's line count — the detector gives blocks,
   not lines); drop the rest. Report the kept fraction.
4. **Caption = fixed minimal bag + one text clause.** No tagger pass in v0
   (`manga, speech bubble, japanese text` / `english text` + clause). The clause
   header must say the language: extend `TEXT_PREFIXES` in anime_tools
   (`Korean text reads as `, `English text reads as `) and bump the pin, or —
   for v0 only — compose the clause locally with the JA header and accept the
   mislabel. Prefer the pin bump; the header is a tag the DiT can use.
5. **Transcript is never dropped.** `inpaint.toml`'s
   `text_tag_dropout_rate = 0.8` would delete the clause in most variants. The
   render descriptor sets `text_shuffle_variants = 0` (v0) or protects the
   text clause the way `@artist` is protected. `caption_dropout_rate` stays
   low (0.1) so the uncond branch exists for CFG.
6. **TE caches go through the pack** — DONE 2026-09-10 in the inpaint prep's
   `stage_text` (`--vocab_pack`; config default when omitted). Without it a
   CJK clause encodes to stock-T5 byte fallback and the arm measures nothing.
7. **Dedicated dataset dir, dedicated cache dir.** Never `target_res` flips
   on the production trees (standing rule). Tree:
   `post_image_dataset/render/<ed>/{resized,lora,staging,cond,text}` per
   edition `ed ∈ {en, ko, ja}`; `boxes.jsonl` beside `resized/`.
8. **Split by artist dir.** Two artist dirs held out for every edition; the
   same two across editions. Artist names never appear in this tree's docs.
   Corpus sample: `--per_work 4` seeded pages per work (~1.1k pages per
   edition, ~2.7–3.1k panel samples each) — the feasibility budget.
9. **Recipe = `inpaint.toml` as the smoke recipe, then scale.** dim 32, lr
   2e-5, `b_cond_init = -6`, `apply_ffn_lora = 1`, 4 epochs. Glyph learning is
   expected to need more than context fill; the EN control decides whether the
   recipe or the script is the limit (R1).

## Arms

| arm | edition | pack | question |
|---|---|---|---|
| **EN** | en | stock tokenizer (no pack) | positive control: Anima writes Latin natively (`docs/findings/freetext_text_rendering.md`), so a failing EN arm is a broken pipeline, not a pack verdict |
| **JA-SHIP** | ja | `synthjakozh1sym_r256` | the question, on the corpus's majority script |
| **JA-RAND** | ja | `random_r256` | s21's control on a task where content *could* matter: SHIP ≈ RAND means the DiT learns row→glyph from data alone and the pack's training is inert here too; SHIP > RAND is the first content signal this line has ever seen |
| KO (later) | ko | `synthjakozh1sym_r256` | typeset hangul; after EN/JA read, same pipeline (`--editions ko`, `stock` teacher per plan2 K2) |

JA-SHIP and JA-RAND share latents and cond caches; only `text/` differs
(re-cache with `--vocab_pack`). One training seed each for v0; a second seed
before any claim beyond "works / does not".

## Rulers

R0 **reader floor.** `stock` CER on the *target* crops' holed lines (the
ground truth is the teacher's own read, so this is decode noise, expected ≈ 0
for typeset KO). Any arm's CER is read against this floor.

R1 **held-out CER.** Held-out artist panels, holes from the real boxes,
caption from the real read, adapter fills, `stock` reads each fill, CER vs
the caption line. Whitespace-blind CER is the headline (`exact_key` convention,
[`whitespace_fixed.md`](whitespace_fixed.md)); a spaced-CER column sits
beside it and is reported, never gated. Floors: `anima_inpaint` with the same
caption (never saw a text clause: CER ≈ 1), and no-adapter base.

R2 **caption swap — the memorisation control.** On *training* crops, replace
the clause with another training crop's line (matched length). Read the fill.
Report the follows-caption rate: fraction of fills whose CER to the swapped
line is lower than to the original. § 14 says a once-seen line binds to the
image, not the rows; the cond stream carries the page, so that hatch is open
here. **A KO arm that clears R1 but fails R2 is a memoriser, not a writer.**

R3 **legibility eyeball.** One contact sheet per arm, 24 held-out cells,
target | cond | fill. The reader can score a fill that a person cannot read
(and the reverse); the sheet is what the CER is checked against.

Gates: **G0** EN arm R1 CER ≤ 0.3 and R2 ≥ 0.8 → pipeline is sound. **G1**
JA-SHIP R1 CER ≤ 0.5 and R2 ≥ 0.7 → the DiT can be taught to write JA from
ext rows; the line reopens with a real budget. G1 failing while G0 passes is
the clean negative this plan exists to produce.

## Steps

**S0 — boxes / reads / panels — RAN 2026-09-10** (`render/corpus_boxes.py`,
`output/render/s0/`, `--editions en ja --per_work 4`): det 1,118 EN + 1,122 JA
pages (15.7k / 18.4k text boxes, 3.1 / 3.4k frames, 72 s each on the daemon),
`stock` read every box (~14 min per edition), bubbles + panels + a 40-cell
sheet per edition, reviewed cell by cell by a general-purpose agent.

| | EN | JA |
|---|---|---|
| bubbles accepted | 8,357 / 14,031 (rejects: 5,639 Japanese-left-in) | 10,730 / 15,529 (rejects: 3,481 SFX, 874 non-text, 431 short) |
| panel samples with text | 2,714 (1,954 multi-bubble) | 3,083 (2,416 multi-bubble) |
| review: crop OK / flawed | 30 / 10 | 34 / 6 |
| review: boxes on non-speech | 13 % | **25 %** (hand-lettered SFX) |
| review: OCR exact / minor / wrong / unreadable | 67 / 17 / 11 / 29 | 69 / 32 / **52** / 6 |
| review: order violations | 4 | 7 |

Verdict: EN usable for a rough run as is; **JA needs a fix first** (a third of
reads wrong, a quarter of boxes on SFX, `stock` hallucinates on SFX bursts —
"ウウウ…" ×200 on one box). The four systematic faults and their fixes are S0b (below).

**S0b — det post-processing — RAN 2026-09-10** (`corpus_boxes.py`, `output/
render/s0b/`, S0's boxes and reads symlinked in; CPU only — S0b changes no
box, so S0's one-read-per-box stays valid and the "re-read the survivors"
pass was not needed). The bubble stage grew four guards and the panel stage
three fixes; every threshold was calibrated on the S0 JA sheet cells already
eyeballed, then two 40-cell draws per edition were reviewed by a
general-purpose agent (one draw before, one after the last rule change).

Guards (`bubbles`, in order — the first hit is the `reason`):

| guard | rule | EN rejects | JA rejects |
|---|---|---|---|
| `repeat` | one 1–3-char unit repeated back to back over ≥ 20 stripped chars and ≥ 60 % of the read; non-Latin reads ≥ 20 chars also fail on < 0.34 distinct chars / char | 0 | 142 |
| `short_kana` | JA, kana-only, ≤ 2 content kana after stripping ー〜っ and punctuation | — | 3,196 |
| `no_balloon` | the band just outside each member box — middle half of each side, corners excluded — is < 60 % white *and* < 60 % flat (± 15 of its median), *and* the box interior is < 70 % white | 1,435 | 1,356 |
| `overlap` | a member box overlaps another bubble's (IoU > 0.1 or one inside the other), whatever that bubble's verdict — the padded read crop saw both | 86 | 63 |

Accepted: EN 6,836 / 14,031, JA 5,973 / 15,529 (S0: 8,357 / 10,730).

Three rules were wrong on the first try and the reviews caught each: a
distinct-char ratio flags every long EN sentence (26-letter alphabet), so it
applies to non-Latin reads only; a run-length rule flags typeset `AH! AH!
AH!` / `NO NO NO` inside balloons, so the loop signature is absolute length
(≥ 20 chars); a full ring around the box fails every wide typeset EN block
(corners land past the balloon) and every tinted colour-page balloon (white
≠ flat), hence the side-strip band with the flatness alternative, plus the
interior-white second chance for text that fills its balloon to the outline.
A `fill < 0.4 → reject` rule was measured and **not** added: it removes
7.7 % of accepted JA bubbles, mostly real inverted (white-on-black) and
tinted balloons, for a third of the residual non-speech. Detector confidence
does not separate brush lettering from balloons (0.66–0.94 either way).

Panel fixes (`panels`): frames < 3 % of the page dropped (77 EN / 88 JA), a
frame containing ≥ 2 other frames dropped (39 / 29 — the whole-page box over
real panels), a bubble goes to the *smallest* frame holding its centre (S0
double-assigned nested frames), the crop grows to hold every accepted bubble
(419 / 404 samples), and reading order is a greedy row band one median
bubble tall, right→left inside a row. Slant proxy (panel overlaps a sibling
frame by > 10 % of its area): **617 EN / 677 JA of ~2.5k samples** — a
quarter; not built for, see below. Samples: **2,507 EN / 2,503 JA** (1,646 /
1,517 multi-bubble); 314 / 151 accepted bubbles fall in no frame (orphans,
dropped).

Review (40 cells per draw; the S0 row is the plan's S0 table):

| | EN S0 | EN draw 1 | EN draw 2 | JA S0 | JA draw 1 | JA draw 2 |
|---|---|---|---|---|---|---|
| crop OK / flawed | 30 / 10 | 35 / 5 | 36 / 4 | 34 / 6 | 38 / 2 | 36 / 4 |
| red boxes | — | 78 | 96 | — | 98 | 83 |
| boxes on non-speech | 13 % | 5.1 % | 10.4 % | **25 %** | 12.2 % | 10.8 % |
| OCR wrong | 11 / 124 | 11.5 % | 9.4 % | **52 / 159** | 12.2 % | 16.9 % |
| order violations | 4 | 3 | 8 | 7 | 4 | 2 |
| false rejects (blue) | — | 13 | 9 | — | 9 | 7 |

Draw 1 predates the interior-white second chance (10 of its 13 EN false
rejects were `no_balloon` on plain bubbles). Two draws of 40 put the
sampling noise at ± 5 points, so the gate reads as follows.

**Gate verdict: JA marginal, EN clear.** JA wrong 12–17 % against ≤ 15 %,
non-speech 11–12 % against ≤ 10 %, EN non-speech 5–10 % / wrong 9–12 %,
crops 36–38 / 40 both. The two JA residues are one residue: **every
accepted brush-lettered box OCRs to garbage** (9 of the 14 wrong reads in
draw 2 are its 9 non-speech boxes; on true balloon text JA wrong is ≈ 6 %).
The balloon probe passes brush lettering on white or flat backgrounds and
nothing image-side cheaply separates it. The lever left is **reader
disagreement**: a second read (hayai, the plan2 voter) of the accepted
boxes only (~6k per edition, a short daemon job), rejecting a box whose two
reads disagree past a CER threshold — garbage reads are unstable, typeset
reads are not. It is the one S0b item that needs the GPU and is not built;
the alternative is to cut now and carry ~11 % of holed boxes with a garbage
clause into the EN/JA smoke.

Known residuals, recorded not fixed: reading order is row-major, so a
spread or illustration page with two columns of balloons interleaves them
(EN cells 3 / 18 / 32, JA 36 / 38 / 39 of the draws); `short_kana` also
drops はい / うん (37 + 33 of 1,487 two-kana rejects, the rest moans); `sfx`
(`line_kind`, anime_tools) mislabels stuttered speech (ご ご ごめん) and
かわっ — pinned, not ours to fix here; the detector silently misses large
plain bubbles on several cells (no box at all — a detector limit, not a
filter one); duplicated reads inside one box (a line emitted twice) are
cosmetic and the cut stage can de-dup adjacent repeats; ♥ / small-kana drops
are cosmetic. Slanted-gutter cut-throughs are ~25 % of samples by the AABB
proxy but only 2–4 of 40 crops per draw were judged flawed by them — the
proxy over-counts; nothing is built for them.

**S0c — cut — RAN 2026-09-10** (`render/cut.py`; decision: cut now, the
S0b residue is ~11 % of holed boxes with a garbage clause, shared by every
arm — a G1 miss must be re-run behind the reader-disagreement pass before
it counts as the clean negative). Panel crop → free-fit onto the 768 band
(`anime_tools.buckets.freefit_bucket`) → `<ed>/resized/<artist>/<work>/
<work>_<page>_p<k>.png` + `.txt` + `boxes.jsonl` (holes, intact boxes,
caption, scale). Caption = bag + text clause; JA through `anime_tools`'
`text_clause`, EN composed locally with `English text reads as` (the
`PositionClause` renderer only knows the JA prefixes on v0.6.2, so a custom
prefix must not go through it). Glyph gate ≥ 20 px (box thickness over the
raw read's line count, scaled):

| | EN | JA |
|---|---|---|
| train / held-out samples | 1,847 / 425 | 1,760 / 392 |
| dropped under 20 px | 235 (kept 90.6 %) | 351 (kept 86.0 %) |
| glyph px p10 / p50 / p90 | 21 / 62 / 168 | 17 / 38 / 102 |

Held-out = 2 artist dirs shared across editions, seeded from the middle
third by sample count (844 of 5,010 samples). Adjacent duplicate reads in a
bubble collapse; JA member boxes join with nothing (S0's `samples` joined
with a space).

**S1 — trainer prep — RAN 2026-09-10** (`render/prep_render.py mask | encode |
text`; `configs/easycontrol/render_{en,ja}.toml`; `render_en` / `render_ja`
rows in `_EASY_ADAPTERS` and the inference `_ADAPTERS`). `mask` gray-fills the
holes (`mask_image.GRAY`) into `staging/` and `heldout_staging/`; `encode`
caches target and cond latents at native size through the inpaint prep's
`stage_encode`; `text` is the inpaint prep's pack-aware `stage_text` with
zero variants (verbatim caption — decision 5), EN on the stock tokenizer
(`--vocab_pack ""`), JA on `synthjakozh1sym_r256` (NB `models/vocab_packs/
anima_cjk_vocab_pack` is a *different* safetensors with the same json — the
descriptor pins the path). Eyeballed target | cond pairs: holes sit on the
accepted bubbles' text only; moans and hand-lettered SFX stay intact.

**S2 — EN smoke — RAN 2026-09-10** (`make easycontrol EASYADAPTER=render_en
--queue`, job `20260910-170900-be1182`): inpaint recipe, caption dropout 0.1,
save every 2 epochs. **1.67 it/s at 768, 7,388 steps, 74 min wall** (default
preset, no block swap, no OOM). Loss average 0.048 → 0.011 by epoch 2 → 0.010
at the end — steep enough that R2 is the number to watch. One-cell judge
smoke on the epoch-2 weight: R1 0.32 vs floors 0.63 (inpaint) / 0.83 (base),
R2 1/1; the sheet shows four short bubbles filled near-exactly and one long
sentence degrading into pseudo-words. Full judges (48 held-out + 24 swap) on
epochs 2 and 4: jobs `20260910-182517-257aac` / `-e1b3c0`.

**S3 — judge — BUILT** (`render/judge.py fill read report`): one loaded DiT
stack, the arm's EasyControl network applied once and re-primed per cell
(`set_cond` + `precompute_cond_kv`), `library.inference.generate` at the
panel's own size, 30 steps, cfg 3.5; floors = `anima_inpaint_girl_preview_v1`
(the only inpaint weight on disk) and the bare base; swap cells replace a
training panel's lines with length-matched donors from the training pool.
`stock` reads each holed box (12 % pad) off the fill and the target. CER
whitespace-blind headline, spaced column, clipped at 1. ~10 s per cell.

**S4 — JA-SHIP + JA-RAND.** JA prep queued behind the judges (job
`20260910-182517-29b3da`, SHIP pack); the RAND arm re-runs only `text` with
`random_r256` into a second `text/` dir and a `render_ja_rand.toml` pointing
at it. Two runs on the daemon, ~70 min each. G1.

**S5 — verdict** into [`findings.md`](findings.md) as a new § (render), the
report under `reports/`, and the closed-lines memory. If G1 passes: the
follow-up is a tagger pass on the bag and the KO arm; if it fails with G0
passing: JA glyph rendering needs DiT weights the LoRA + pack cannot supply,
and that is the end of the DiT-side vocab claim.

## What this plan does not answer

- **ZH** — no ZH edition in the corpus. A ZH arm needs a different source.
- **Page composition** — whether the model places a bubble it invents. Holes
  are given; only what goes inside is learned.
- **SFX** — hand-lettered onomatopoeia are excluded from the holes in v0
  (speech only; `line_kind` plus the S0b guard). The SFX reader line is
  separate and shipped.
- **Whether the pack's *training* matters** beyond SHIP vs RAND on this one
  task. A SHIP ≈ RAND read here is consistent with s21 and is not a new
  negative.

## Known traps

- `parse_caption` was quote-blind on 2026-09-05 (inner comma splits the tag);
  the fix was assigned to anime_tools. Verify on the pinned tag (v0.6.2)
  before composing a caption with a comma inside the quoted line; a KO line
  with `,` is common.
- Inpaint's `_inpaint_prep_paths` pins `--src` to the shared corpus; the
  render handler must not inherit it.
- The AnimeText detector emits **block** boxes for horizontal text (a whole
  balloon, 2–3 lines); `line_h` in the S0 files is block height, and EN pages
  keep ~37 % of their text boxes in Japanese (SFX, moans) — those are guard
  rejects by design, not detector misses.
- EasyControl at 768 with a two-stream forward needs more dynamo graphs than
  `block._forward` (closed line `dynamo_limit_contextvar`); block-compile
  first on OOM, not grad checkpointing.
- Every GPU step goes through the daemon (`make daemon-run` / `--queue`); a
  bare background process dies at ~1 min.
- Do not compare a CER here to any `/ 617` or `/ 71` figure in
  [`findings.md`](findings.md); different units, different task.
