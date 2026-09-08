# FreeText on Anima — localization works, native OOD text rendering doesn't

This records the evaluation of FreeText (*Training-Free Text Rendering in
Diffusion Transformers via Attention Localization and Spectral Glyph Injection*,
arXiv:2601.00535) on Anima, run 2026-06-01. It was not promoted to a shipped
method, but it is not a dead end either — it produced one genuinely reusable
capability and one clean, well-triangulated negative result.

Two-line summary:
1. GO — training-free writing-region localization works on Anima. We can find
   where text goes by reading Anima's own image→text cross-attention, with no
   external OCR / layout predictor / VLM detector. This is reusable beyond text.
2. NO-GO — training-free *native* OOD glyph rendering. Stage-2 (SGMI) can make
   Korean (a script Anima can't natively draw) legible, but only as a pasted
   patch, never as a sign the model drew itself. The wall is Anima's native
   Korean glyph capability, not the method — and an English control proves it.

Archived bench (driver, machinery, all runs + images): `_archive/bench/freetext/`
(report `stage2_report.md`, Stage-1 writeup `stage1_progress.md`). Paper PDF archived
alongside. Related memory: [[project_freetext_phase0_localization_go]],
[[project_freetext_stage2_sgmi_deepwindow]], [[project_sigma_signal_resolves_by_045]].

---

## The method (faithful summary)

FreeText decomposes text rendering into where and what, both training-free:

- Stage 1 — localization (paper Eqs 1–6). Read head-averaged I2T attention
  `A(t,l)`, average over the target token span (+ sink tokens), giving a per-block
  per-timestep map `M(t,l)`. Select informative `(t,l)` pairs, aggregate, then
  topology-refine (neighborhood smooth → Otsu → DBSCAN component pick) → binary
  latent writing mask `R`.
- Stage 2 — SGMI (Eqs 7–14). Rasterize the target string into `R`, VAE-encode
  → glyph latent `z_ref`. Per step: noise-align to σ, Log-Gabor band-pass (keep
  mid-freq stroke structure, kill low-freq background), inject over a short mid-early
  window with cosine-annealed weight via masked replacement
  `z̃ = (I − λR)⊙z + λR⊙z_sgmi` (Eq 14, cited as Blended Latent Diffusion).

Key structural fact: Stage-2's injection is masked latent replacement. Our
implementation was faithful — the difference in outcome is not a mechanism bug, it
is that the paper only ever applies this to scripts the base can already draw.

---

## Stage 1 — GO: localization without external models

Anima is a two-stream cross-attention DiT (`crossattn_emb` = Qwen3 last_hidden_state),
cross-attn has no RoPE, so an eager `softmax(QK^T)` recompute reproduces the fused
kernel faithfully and exposes a direct image→text attention map. Entity-token
attention concentrates 2–3.6× over uniform on the sign region, strongest in
shallow-mid blocks (L6–L17) at mid timesteps — matching the paper's Fig. 3.

Three Anima-specific deviations from the paper were load-bearing (paper-literal
gave whole-figure masks):
1. Selection = direct concentration ranking, not soft-IoU (the paper's reference
   `Y` doesn't exist at deploy; consensus rewards diffuse deep-block maps).
2. Anchor = entity tokens only, not entity+sink. Anima's only attention sink is
   the zeroed padding field (Qwen3-base adds no special tokens, `n_special=0`),
   and it points at the body — including it drags localization off-target.
3. Region score = total mass + mass-weighted centroid, not Eq-5 peakiness (peaky
   scoring fingers face/ViT-edge sinks; the sign is a broad band).

This is the reusable win: Anima's endogenous attention is a usable, external-model-
free localizer for "where is concept X in the image." Validated across text-content
variation (n=3), and the WHERE survives even for un-renderable text (entity
attention still lands on the sign for Korean the model can't draw). The fragile part
is the contrast of the raw map → extractor robustness (contrast-invariant threshold +
centroid seeding + abstain on low lift).

---

## Stage 2 — NO-GO: legible, but never native

SGMI can drive OOD Korean to legible, correct glyphs on Anima, but only with a
recipe that departs from the paper, and the result is always a patch:

| lever | finding |
|---|---|
| window depth | Paper window σ∈[0.6,0.8] stops before strokes lock (σ≲0.45 tail "corrects" 안녕→안긴). Extending the hard-replace to σ=0.35 locks strokes → legible. Legibility lever. |
| high-pass injection | The flat grey board is a low-freq artifact (white-on-black raster's DC). A one-sided high-pass (keep the band-pass's low-cut, drop its high-cut) lets the model paint a textured board while strokes survive. Un-greys the board. Reframes "Log-Gabor harmful": only its high-cut harms OOD strokes; its low-cut is the cure. |
| dark-on-light polarity | Default raster is white-on-black — inverse of the dark-ink-on-light-card a base draws natively. `fg=0.1/bg=0.9` injects on-manifold color. Best config: `hp_deep_dl` (high-pass + dark-on-light) — bold dark text on a textured card. |

Falsified levers (negative results, all in the archived report):
- Late soft-release tail — inert; by σ=0.40 the flat card is committed, a deep-tail
  cosine release can't re-color it.
- Color jitter on the raster — the premise is right (a flat fill encodes to a
  near-constant, off-manifold latent), but synthetic injection is the wrong delivery:
  full-chroma → CFG amplifies to a pastel rainbow; luminance-dominant → washes text
  contrast on the flat card and corrupts strokes → garbled text on the high-pass
  path (which keeps only the stroke high-freq the jitter perturbs). The principled
  flat-fix is high-pass itself — strip the flat DC and let the model paint
  on-manifold texture; don't inject texture and fight the model.
- Ink-shaped mask — backfires (floats distorted strokes on the body); full
  rectangular-region injection hands the denoiser a "blank board" prior it commits to.

### The English control settles it

Rendering text Anima can draw ("ANIMA"), same girl-sign composition:
- `base` (no injection): fully native — crisp black text on a clean white bordered
  card, gripped and shaded, integrated in the scene.
- Every injection variant degrades it, including the gentle paper recipe (native
  white sign → flat olive card, faint text).

So the injection is the source of the patch look, not the cure. On renderable
text, doing *nothing* beats SGMI. SGMI only "helps" where the base can't draw the
glyph — and there it can only ever produce a non-native patch, because it overwrites
the model's own latent with foreign raster content.

---

## Root cause and where native lives

The non-nativeness is structural, two reinforcing reasons:
1. Masked replacement is a patch by construction — the model never generates
   the sign+text as part of its own coherent image, so it reads as what it is.
2. Anima has no native Korean glyph prior to integrate a gentle hint into (Qwen3
   already understands Korean semantically — the gap is purely the DiT's visual
   glyph head). A light nudge gets "corrected" away, forcing hard replacement.

The paper's own §4.2.3 caveat agrees: SGMI "strengthens glyph structure rather than
enabling unseen characters from scratch." It never had to fight a base that can't
draw the script; on Anima + Korean, we do.

Paths to native OOD text (neither pursued here):
- Small Korean-glyph LoRA — closes exactly the missing visual head; inference then
  needs no injection (same as English). Highest-confidence path, and home turf for
  this repo. Qwen3's existing Korean semantics means the gap is narrow.
- Replacement → guidance — instead of overwriting `R`, steer the model's own `x̂₀`
  toward the glyph (`E = ‖M⊙(x̂₀ − z_ref)‖²`, nudge the velocity by `∇E`) so the model
  paints the sign itself. The only training-free route that could be native, but
  speculative on a base with zero Korean prior, and it needs a gradient through the
  DiT (collides with the block-swap offloader — see
  [[project_blockswap_extra_forwards_gradcache]]).

## Reusable lessons

- Endogenous I2T attention is a training-free, external-model-free localizer on
  Anima (two-stream cross-attn, no RoPE → readable). Reuse for any "where is X."
- Anima's only attention sink is the zeroed padding field (`n_special=0`), and it
  points at the body — exclude it from any attention-anchoring.
- **Training-free injection cannot manufacture a capability the base lacks** — it can
  patch, not teach. Native unseen-script text needs base capability (LoRA), not a
  better injection recipe.
- "Flat fill is off-manifold" is real, and high-pass (let the model paint the
  low-freq) is the right fix, not synthetic noise injection.
