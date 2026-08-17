# CJK-aware Anima — plan2: the glyph line (OCR captions + unmasked text training)

*Line home: [`plan.md`](plan.md) is the vocab-pack line (Phases 2–3, text-encoder
side). This document promotes its deferred Phase 4 into its own line: make the
DiT **render** Japanese text, by OCR'ing in-image JA into captions and paying
loss on the text pixels it currently masks away. Written 2026-08-17.*

Status: **PROPOSED** — ceilings resolved ([`done2.md`](done2.md), 2026-08-17:
VAE-recon and census skipped by user call, char-row separability measured
green + init collision fixed); Phase 1 is the next work item.

## Goal, stated as the invariant it creates

Today the pipeline treats in-image text as noise: `make mask` segments it
(stroke-accurate UNet++ + ComicTextDetector block gate) and `masked_loss`
zeroes its pixels out of every training step. The goal is to flip that for
*captioned* text: an image whose caption carries the verbatim OCR string
「営業中」 trains with those pixels **in** the loss, so the DiT binds the ext
token ids of 営業中 to their glyphs. End state: a JA prompt with quoted text
renders that text, in-register (signs, speech bubbles, sfx), and Japanese is
embedded *as Japanese* end-to-end — ext rows on the encoder side, glyphs on
the image side. No EN-id detour anywhere on this path.

The deliverable is an **ordinary DiT LoRA** (every existing loader) that
*depends on* the vocab pack being loaded, exactly as plan.md's deployment
section scopes it. The vocab pack ships meaning; this LoRA ships glyphs.

## Why this is startable now (what is already measured/built)

- **The encoder side of quotes is done.** `quote_preserved` passes verbatim
  through ext rows at student-vs-teacher cos **0.988**
  ([report](report_0816_phase2.md)) — quoted strings arrive at the DiT as
  stable, well-formed ext-id sequences today. What's missing is purely that
  the DiT has never seen a glyph with loss on it.
- **The masks already exist** (`post_image_dataset/masks/`, stroke-level,
  ctd-gated). Phase 1 reuses them as the *localization* of what to unmask; no
  new segmentation work.
- **The OCR pilot ran** (`datasets/manga_text.py`,
  [`report`](datasets/assets/manga_text_pilot/report.md), 986 regions):
  register split (34.5% line / 21.5% sfx / 46.2% translatable), a working
  logprob confidence gate, and per-line polygons on 73,725 danbooru images.
  `done.md` rejected it *as distillation corpus* and explicitly kept it *for
  Phase 4* — this line is that use. **OCR backend quality is a swappable
  component, not a design pillar** (user call, 2026-08-17): manga-ocr + the
  measured logprob gate is the default; any better reader drops in behind the
  same sidecar format.
- **The name-grid render (2026-08-17, `bench/cjk_adapter/results/20260817-0924/`)
  settled what this line does NOT need to carry**: character *identity* lives
  in EN tag-id sequences and is served by the (orthogonal, optional) name
  dictionary or the D5 register. Glyphs are the one thing no encoder-side
  trick can buy — this line is the only path to them.

## Dependency pulled forward from plan.md Phase 3

Training image-level with JA captions requires **TE caching through the ext
encoder** (JA caption → ext ids → cached embeddings). That is the
"strategy shim in preprocess" slice of plan.md's Phase 3 ship list — it
becomes a **prerequisite build item here** (the rest of Phase 3, ComfyUI node
etc., stays where it is). Padding invariant applies unchanged; regenerating
TE caches for JA captions is expected and budgeted (EN caches untouched by
construction).

## Phase 1 — data build

*(The Phase 0 ceilings this plan originally opened with are resolved —
verdicts, the separability numbers, and the byte-permutation init fix live in
[`done2.md`](done2.md).)*

- **OCR sidecars**: `{stem}_ocr.json` next to the mask —
  `[{poly, text, logprob, register}]`, backend-agnostic (backend + version
  recorded per record). Default backend manga-ocr per line, fed by the
  existing mask/polygon geometry; the pilot's logprob gate (≈ −0.3, re-tune
  on the build corpus) drops garbage reads. SFX policy v1: **captioned as text like any
  other region** if it passes the gate (it's renderable content in-register),
  revisit only if Phase 2 shows it poisons dialogue text.
- **Caption composition**: append quoted strings to the existing caption in
  the D6/quote template register the encoder already handles —
  `「{text}」と書かれた{carrier}` where a carrier is inferable (sign/bubble),
  bare `「{text}」` otherwise. Position-free in v1. Multi-region: one clause
  per surviving region, reading order top-right→bottom-left for vertical.
  Captions with OCR clauses are JA-bearing → TE-cached through the ext shim.
- **Selective unmask — the core safety rule.** For each OCR'd image, emit a
  training mask variant where **captioned** text regions are 1 (trainable)
  and **uncaptioned** text regions (gate-failed, unreadable) stay 0. Text the
  caption doesn't describe must never get loss — that asymmetry is the whole
  reason plan.md ordered "keep masking ON until this phase". Text-free images
  are untouched and double as the natural negative class (no quote in caption
  → no text in image).
- **Wiring note (verify at build time)**: two mask paths exist —
  `train.py`'s `masked_loss` (alpha-mask channel) and `CachedDataset`'s
  `mask_dir` key consumed by the bespoke loops. This line trains through
  `train.py`, so the flipped masks must land in the path train.py actually
  reads for the chosen dataset config; confirm before the pilot, don't assume.

## Phase 2 — pilot train + gates

Ordinary `train.py --method` run (plain LoRA stack, text-bearing subset +
text-free dilution), vocab pack loaded, TE caches ext-id. Small first: the
question is binding, not quality. Gates, in order of what kills the line vs
tunes it:

- **Render→OCR round-trip** on held-out strings: the owed D6 instrument
  (same template, different quoted strings — never a cosine, per G3) becomes
  real here as CER between prompted and OCR-read rendered text. This is the
  binding gate.
- **No-text-spam**: text-free prompts (the existing 2c grid is the ready-made
  set) must not sprout glyphs. Compare against base-model renders, same seed.
- **EN regression**: G1 stays green (the LoRA composes with, not replaces,
  the pack; EN prompts must be unaffected at multiplier 0 and acceptably
  affected at ship multiplier).
- **Tag-register regression**: the 2c per-register readout re-run with the
  LoRA active — glyph training must not degrade what 2c bought.
- **Masking sanity**: a probe arm trained with unmask-everything (no caption
  gating) is the predicted-failure control — if it does *not* underperform
  the gated arm, the selective-unmask machinery is dead weight; if it does,
  the safety rule is measured, not assumed.

## Escalation rung — glyph conditioning via EasyControl (enter only on measured failure)

"Does the DiT need a mechanism to *recognize* Japanese?" splits in two, and
only half is open. **Semantic recognition is settled**: Qwen reads JA natively
and the vocab pack gives the T5-side ids meaning — measured end-to-end (2c
renders, quote register 0.988). A new cross-attn layer for *meaning* is
plan.md's 2-ii escalation, still gated on a capacity signal that has never
appeared. **Glyph identification** is the genuinely open half: caption-only
supervision has to bind thousands of visually-fine-grained kanji classes, and
the CJK text-rendering literature (AnyText family) says the strong method is
explicit glyph conditioning — render the quoted string with a font, feed it
as an image condition.

The repo already owns that mechanism: **EasyControl** (frozen DiT, extended
self-attn image conditioning, per-block cond LoRA). The rung, if Phase 2's
direct arm fails its CER gate:

- condition channel = font-rendered glyph image of the caption's quoted
  string (layout-free v1: white-on-black text strip);
- training pairs come **for free and without OCR** — composite font-rendered
  text onto text-free images (synthetic data is how AnyText/TextDiffuser
  train), with the OCR'd real corpus as the eval/adaptation set;
- Phase 2's direct arm stays the ablation baseline, and the separability
  measurement ([`done2.md`](done2.md)) is method-agnostic — nothing run
  before this rung is wasted.

Favorable prior for the direct arm worth stating: our char ext rows are
one-id-per-character (28,017 rows), a *better* binding granularity than the
BPE fragments EN models learned letters from — and `param=global` measurably
preserved (indeed widened) the Qwen-init separation between characters
([`done2.md`](done2.md)).

## Phase 3 — scale + ship

Scale along whatever Phase 2 shows binds (glyph-size floor, char coverage,
register mix). Ship as `anima_ja_glyph_lora.safetensors` + snapshot, release
asset alongside the vocab pack, loader docs stating the dependency. ComfyUI:
rides the existing Adapter Loader; needs the vocab-pack node from plan.md
Phase 3 to be usable there — the two ship items converge, neither blocks the
other's training work.

## Risks

1. **VAE glyph floor** — accepted unmeasured (user call: qwen-image shows the
   VAE family carries glyphs); worst case scopes the line to large text,
   doesn't kill it. Revisit only on recon-shaped Phase 2 failures.
2. **OCR noise trains wrong glyphs** — logprob gate + selective unmask;
   backend swappable by design. Residual risk is *systematic* misreads
   (pilot's 五月蠅い→"Stylish" class), which land as wrong caption text, not
   wrong pixels — same failure class as 2a's `mt_unverified`, and the same
   mitigation applies (provenance-weighted trust if it shows up in Phase 2).
3. **Text spam** — gated in Phase 2; the text-free negative class is free.
4. **Kanji long tail** — char ext rows exist for everything; *visual* visits
   won't. Expect the 2c lesson to repeat (concept ≠ fidelity floor); the
   char census over the Phase 1 OCR output (the `gates/coverage.py` analog)
   makes it measurable before it's a surprise.
5. **Register imbalance** (vertical/horizontal/sfx) — census alongside the
   Phase 1 build, mix knob in Phase 2; don't pre-tune.
6. **Interaction with σ-demote / free-fit** — none expected (masks ride the
   latent grid like today), but the flipped-mask path must be checked against
   demoted sibling latents if `sigma_lowres` is on for the pilot; simplest:
   pilot without it.
