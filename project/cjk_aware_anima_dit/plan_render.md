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
| S0 script | `render/corpus_boxes.py` — `det` / `read` / `bubbles` / `panels` / `sheet` | outputs under `output/render/<name>/`, keyed by corpus-relative path |
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
"ウウウ…" ×200 on one box). The four systematic faults and their fixes are S0b.

**S0b — det post-processing + re-read (NEXT STEP).** All in
`corpus_boxes.py`; the det pass need not re-run (boxes and frames are on
disk), only `read` after the box filter, then bubbles → panels → sheet → the
same agent review.

1. **SFX / hallucination guard, before reading** (JA first): drop a text box
   that (a) is short and kana-only by `line_kind` *after* the read, (b) has a
   repetition ratio above a threshold (`ウウウ…` loops; cap read length), or
   (c) has no balloon outline — probe the box's dilated ring for a mostly
   white/uniform band, the cheapest "is this a bubble" test on a colour page.
   Re-read only the survivors so the OCR pass is smaller, not larger.
2. **Overlap rule**: two accepted boxes that overlap (IoU > 0.1 or one
   contains the other) reject both — merged-neighbour reads ("S DAY FTER AY
   OF SEX") and 6-bubbles-in-one-box came from these.
3. **Frame fixes**: a bubble whose box crosses the panel edge grows the panel
   to contain it (never a sample whose target text is half outside the crop);
   panels below a minimum area are dropped; slanted-panel cut-throughs
   (en 16/18/26/32/39, ja 38 on the sheet) are the remaining case — measure
   how many samples they are before building anything for them. Frameless
   pages (214 EN / 98 JA) as whole-page samples are fine at 768.
4. **Reading order**: widen the row band in `_reading_order` (a bubble
   height, not half of it) — the current band is so narrow it sorts y-first.
5. Sheet render: long JA captions overprint — wrap by glyph count, not chars.

Gate for S0b: the agent's re-review reads JA wrong ≤ 15 % and boxes on
non-speech ≤ 10 %, EN unchanged or better. Then cut.

**S0c — cut**: panel crops → `post_image_dataset/render/<ed>/resized/` +
`.txt` captions (`<bag>. <clause>` with the holed boxes' lines in order) +
`boxes.jsonl`; long edge sized for the 768 tier; drop a sample whose smallest
holed box is under 20 px tall after resize (the block height over the read's
line count is the glyph-height proxy — the detector gives blocks, not lines).

**S1 — trainer prep** (`project/cjk_aware_anima_dit/render/prep.py`, a thin
sibling of the inpaint prep): mask stage takes `--boxes boxes.jsonl` and holes
the boxes instead of random strokes; encode stage caches target *and* cond
latents at native size via `library.preprocess.cache_latents`; text stage is
the inpaint prep's now pack-aware `stage_text` (decision 6). Descriptor
`configs/easycontrol/render.toml` (`name` reroutes per edition), a `render`
row in `_EASY_ADAPTERS` and in the inference `_ADAPTERS` table.

**S2 — EN smoke.** `make easycontrol EASYADAPTER=render … --queue` at the
inpaint recipe. Judge (S3) at the end of epoch 2 and 4. G0 decides whether S4
runs at this recipe or a scaled one.

**S3 — judge** (`render/judge.py`): R0–R3 in one script, reader = `stock`
for JA and EN, PP-OCRv6 Latin as a second EN read if `stock` disagrees with
itself. Reuses `text_bind_judge.py`'s CER/NFKC/montage code, not its reader.

**S4 — JA-SHIP + JA-RAND.** Same latents, two text caches, two runs on the
daemon. G1. KO is the same pipeline afterwards.

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
