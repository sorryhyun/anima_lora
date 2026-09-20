# freetext — what the shelved FreeText line (2026-06) says to this one

> Light note, 2026-09-16. FreeText (*Training-Free Text Rendering in
> Diffusion Transformers via Attention Localization and Spectral Glyph
> Injection*, arXiv:2601.00535) was evaluated on Anima on 2026-06-01 and
> shelved; the record is [`docs/findings/freetext_text_rendering.md`](../../docs/findings/freetext_text_rendering.md),
> the driver + paper PDF `_archive/bench/freetext/`. It is the same problem
> as this line — OOD-script text on Anima — approached from the opposite
> side (training-free latent injection vs a learned address), and one of
> its verdicts reads differently after the wake line.

## What it found, in two lines

1. **Stage 1 (localization) — GO.** Anima's own image→text cross-attention
   (two-stream, no RoPE, so an eager `softmax(QKᵀ)` recompute is faithful)
   localises where the text goes: entity-token attention concentrates
   2–3.6× over uniform on the sign region, strongest in blocks L6–L17 at
   mid σ. No external detector. Three Anima-specific deviations from the
   paper were load-bearing (concentration ranking, entity tokens only —
   the zeroed padding field is the only sink and it points at the body —,
   mass + centroid scoring).
2. **Stage 2 (SGMI, masked latent replacement of a glyph raster) — NO-GO
   for native text.** Korean became legible only as a pasted patch; the EN
   control showed injection *degrades* text the base draws natively.

## The re-read

FreeText's root cause was *"Anima has no native Korean glyph prior — the
DiT's visual glyph head is missing"*, with a Korean-glyph LoRA as the
prescribed fix. The wake line says otherwise: the frozen DiT renders a
requested kana from an ext-row delta alone (W1, 2026-09-13), and
[`reports/krzh16_2026_09_16.md`](reports/krzh16_2026_09_16.md) trained 8
Hangul rows on the rows recipe and got Hangul-shaped glyphs (감 exact by
the VL reader, 몹 → 몸, 가 → 기 — jamo-level near misses). The units were
there; the wall FreeText hit is that latent injection bypasses the text
path and so never gives the DiT an *address* for the glyph. Its "a light
hint gets corrected away (안녕 → 안긴)" is the categorical-address pull
of [`findings_seed.md`](findings_seed.md) seen from the latent side: the
DiT snaps a near-glyph to the nearest unit it knows. The negative result
stands (injection ≠ native); its explanation does not.

## The one thing this line can take: an attention ruler

Every placement / wipe read here is post-render (detector box IoU, `en
cos` vs the `"hi"` reference, two OCR readers). Stage 1 is a render-free
read of **where an ext-row token attaches**, per block and per σ. It
would answer, without a new arm:

- **wipe** — does a row that overrides the scene spread its attention
  over the whole grid, while a kept render concentrates it in the bubble?
  (The residual wipe is the delta norm, `synth.md` *Risks the build
  carries*; no data-mix arm moves it.)
- **frame binding** — under `swap` (EN caption, word swapped) does か's
  attention leave the bubble, or stay and draw Latin strokes? (か 0/16
  under swap, あ 11/16 — per glyph, not per arm.)
- **katakana** — the 53k miss is not row-space interference
  (`findings_seed.md`); "render / reader side or DiT-side" is the open
  split. If katakana rows attach like hiragana rows and still fail, it is
  DiT-side / reader-side; if they never attach, it is the address.

Caveat before trusting it: FreeText read *entity* tokens (the sign) at
mid σ; row identity here is decided at σ ≈ 0.8 (`findings.md`). Whether
the ext-row token's own map is readable in the 0.7–0.9 band is the first
thing to check — the localizer's contrast was already the fragile part in
June (abstain on low lift).

Instrument: `_archive/bench/freetext/stage1_localize.py` (`capture_maps`,
`dump_maps_npz`) + `stage1.py`; hook it behind the `ExtDelta` hook
(`src/common/hooks.py`) on a trained table and read the map for the row
token vs the frame tokens. Not built; no job.

## Ideas noted, not planned

- **Inference-time spatial gating** — mask image tokens outside R from
  the ext-row tokens, so the delta cannot override the scene. Training-free,
  composes with a shipped pack; costs a localization pass and touches the
  attention dispatch invariants (`networks/CLAUDE.md`). Only worth it if
  the attention ruler above shows the wipe *is* attention spread.
- **SGMI on the base for katakana vs hiragana** — a training-free "does
  the DiT hold katakana units" probe. Weak signal by FreeText's own EN
  control (injection hurts even renderable text); the attention ruler asks
  the same question cheaper.
- **Paste on-manifold lessons** — FreeText's flat-DC / polarity findings
  are the same family as this line's "bubble is a canvas / erase patch"
  risks; `anchor_ink`, tilt and the font pool already cover the part that
  matters.

## Not a connection

SGMI as an inference path. This line's target is native rendering, and
FreeText's EN control already showed injection breaks native text.
