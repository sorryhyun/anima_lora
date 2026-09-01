# Unmask A/B — text masks off + OCR-quoted captions (2026-09-01)

**Status: CLOSED same-day, hypothesis confirmed with attribution.** Full
3-seed grid: arm C (unmask + OCR-quoted captions) is no-worse and
spam-cleaner than masked arm A on every seed. Decomposition arm B (unmask
alone) then **reproduced the spam** — giant decorative gibberish on the cafe
row, a large pseudo-JA-filled speech bubble on the comic row, cat ears +
signature scribbles on the portrait row — so the health is attributable to
**the captions explaining the text, not to unmasking**. See §Attribution.

## Goal (the line's final reframe, set this day)

The CJK line's ultimate criterion is **not** KO-prompt UX and not a
text-rendering feature: it is that **manga data trains healthily with the
text masks OFF** — captions that carry the in-image text make text pixels
attributable, so unmasked training neither corrupts general quality nor
teaches unconditional text spam. MT/substitution is not a direction for this
goal (it destroys the surface), and the coarse-tag arm (`japanese text` alone,
unmasked) was already tried before this line and lost to masking — settled,
do not re-run. The loss curve cannot see this property; the readout is
render-level (spam rate + adherence + quality), never fm loss.

## Design

Shard: `sincos` (350 imgs, 133 with text masks; heavy typeset text, some
pages Chinese-typeset). Plain LoRA dim32 / lr 2e-5 / 8 epochs / batch 1,
identical in both arms; latents shared bit-identically.

| arm | loss mask | captions | TE encode |
|---|---|---|---|
| A | text masks ON (production) | production captions | stock caches |
| C | **OFF** (`masked_loss=false`) | + OCR quote tags (73 imgs, 148 lines) | **synthjako2 ext pack** re-cache |

`base` = no-LoRA reference renders.

Pipeline (all landed this day):

- `datasets/ocr_text_captions.py` — mask-complement CCs, dilation-merged to
  bubble level, padded crops → batched manga-ocr with mean-token logprobs
  (reuses `manga_text.MangaOCR`), gates: logprob > −0.3, area ≥ 400 mask-px,
  junk regex. Output: sidecar captions + `ocr_records_sincos.jsonl`.
  Quality after bubble-merge: median line 12 chars, full sentences
  (`「いや、ちがう...八奈見さん、ここじゃまずいって!」`).
- `datasets/cache_te_ext.py` — temp mirror (symlinked resized imgs, corrected
  captions + variants with quote tags appended via `parse_caption` /
  `compose_caption`), then the standard `cache_text_embeddings` stage with a
  tokenize strategy routing the T5 side through `HybridT5Encoder` and the
  synthjako2 rows appended to the adapter embed. 351 caches; sanity: an OCR'd
  caption fires 35 ext rows; CJK-free captions encode bit-identically to
  stock (copied-through imgs are an in-run control).
- `configs/gui-methods/custom/cjk_unmask_{a,c}.toml` — the two arms;
  arm C redirects `text_cache_dir` only (latents shared with A).
- Eval prompts: `assets/unmask_eval_prompts.txt` — 8 text-free rows
  (row 8 `comic, 2koma` asks layout, still no text), rendered per arm at
  seeds 42/7/1234 (`output/tests/cjk_unmask_eval/`).

Both training runs cache-verified at launch (arm C log: "Skipping Qwen3 …
all text-encoder outputs cached", zero latent re-encodes — the ext
conditioning is what trained). Note inference conditioning is *identical*
across arms (stock encoder, no CJK in eval prompts): every A↔C render
difference is weights-only.

## s42 readouts (2 of 8 rows inspected; single seed — treat accordingly)

- **Row 6, maid/cafe** (model eyeball): tag adherence perfect in all three
  arms. base **spams large decorative gibberish latin signage** on a
  text-free prompt ("CIRGO MO, TAFE"); arm A ≈ base (same spam, near-same
  composition — masked training leaves the base's text habit untouched, as
  expected). **Arm C drops the decorative spam**: text only as small
  chalkboard-menu scribbles (diegetic), and the cafe interior is the most
  fully realized of the three. Composition diverges from base far more than
  A does — consistent with unmasking admitting more training signal.
- **Row 7, upper body/portrait** (user eyeball): base and arm A both
  hallucinate **cat ears** not in the prompt; **arm C does not** — the
  adherence win at this seed is C's.

## Full-grid readout (s7 + s1234, model eyeball of A vs C pairs)

Text-spam events on text-free prompts, 8 rows × 3 seeds:

- **Arm A, 3 clear spam events**: s42 row 6 (giant decorative gibberish
  letters), **s7 row 3 (two spawned speech bubbles with pseudo-JA scribbles
  on a plain hug prompt — the masked arm inventing bubbles)**, s7 row 8
  (caption box + pseudo-JA text block beyond the asked comic layout).
- **Arm C, zero bubble/decorative events**: worst cases are diegetic or
  marginal — chalkboard menu scribbles (s42 row 6), a small signature
  scribble (s1234 row 6), small kana SFX in the comic row (s7 row 8).
- **Shared habits (not C regressions)**: `@handle`-style watermark scribbles
  appear in BOTH arms at s1234 row 7, and punctuation SFX in both comic
  rows — sincos signature/SFX marks apparently survive arm A's masks too.
- Quality/adherence: C ≈ A everywhere else; several rows near-identical
  (rows 1, 3@s1234, 5, 7@s7). Cat-ears hallucination: base+A only at s42
  row 7; both arms at s1234 row 6.

Mechanistic note on the s7 bubble spawn: masking removes the text pixels
from the loss but the *bubble shapes* and their co-occurrence context still
train, with the contents never supervised — the masked arm can learn
"bubble with mush inside" as a compositional element. The captioned unmasked
arm gets the contents supervised *and* explained, and did not spawn bubbles
anywhere in the grid.

## Verdict after the grid

Consistent across all three seeds: the unmasked + OCR-captioned arm shows
**no quality degradation and fewer text artifacts than the masked
baseline** — the masked arm is the one that spawns bubbles and gibberish
blocks. This despite the enablers being individually mediocre: OCR lines
are gappy fragments of the real pages (73/133 imgs pass gates), and the
synthjako2 pack's semantic performance on names is known-weak. The bar was
"not worse than masking"; the grid reads "better".

## Attribution (arm B readout, same day)

Arm B (unmask + production captions, stock TE, no OCR;
`configs/gui-methods/custom/cjk_unmask_b.toml`, renders
`output/tests/cjk_unmask_eval/armB_s*`) **spams**:

- s42 row 6: the giant decorative gibberish returns ("CDRGO MA / LH MAL"),
  plus a signature scribble — the row where C stayed clean.
- s7 row 8: a large spiky speech bubble **filled** with pseudo-JA text plus
  scattered scribbles — heavier than A's caption box, no comparison to C's
  small SFX.
- s42 row 7: cat-ears hallucination (like base/A; C had none) + signature.
- s7 row 3 stayed clean (A's bubble spawn did not reproduce here — single
  instance variance both ways).

Net: **B ≥ A in text artifacts; C is the clean arm.** Unmasking alone does
not explain C — the OCR-quoted captions (through the ext encode) are the
load-bearing element, which re-confirms the settled coarse-tag verdict
("unexplained text unmasked = worse than masking") on this exact
shard/recipe. The line's hypothesis — captioned text pixels become
attributable and stop being poison — survives its decomposition test.

## Caveats / owed before a real verdict

1. ~~Single seed~~ Full 3-seed grid read (§above); eyeball only — no
   quantitative tagger-judge count yet (deferred by user call).
2. Positive control unrun: does C render *asked-for* text
   (`japanese text, 「…」` prompt through the ext encoder — needs merge or
   run_bench-style encode)? Not required for the unmask-health gate, but it
   is the attribution mechanism's direct signature.
3. ~~Confound~~ → arm B queued (see verdict section). Note the settled
   coarse-tag experiment was a different shard/recipe; B is the same-setup
   re-measurement.
4. sincos has Chinese-typeset pages; OCR wrote them as kanji strings. Fine
   for attribution, wrong as transcription.
5. One shard, one recipe, 8 epochs — scale/duration effects (spam often
   grows with training length) unmeasured.
